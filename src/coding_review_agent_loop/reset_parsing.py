"""Pure rate-limit reset-time parsing (stdlib only).

Moved verbatim out of ``agent_failure`` so ``transient`` can use it without an
import cycle; ``agent_failure`` re-exports every name.
"""

from __future__ import annotations

import datetime
import os
import re
import zoneinfo

# Parse "Retry-After: N" (HTTP header) or "retry after N" or "retryDelay: Ns" (gRPC).
_RETRY_AFTER_SECONDS_RE = re.compile(
    r"\bretry[- ]after[:\s]+(\d+)\b"
    r"|\bretry[_-]?delay[:\s]+['\"]?(\d+)s['\"]?",
    re.I,
)
# Parse "try again in Xd Yh Zm Ws".
_TRY_AGAIN_IN_RE = re.compile(
    r"\btry\s+again\s+in\s+"
    r"(?:(?P<d>\d+)\s*d(?:ays?)?\s*)?"
    r"(?:(?P<h>\d+)\s*h(?:r|ours?)?\s*)?"
    r"(?:(?P<m>\d+)\s*m(?:in(?:utes?)?)?\s*)?"
    r"(?:(?P<s>\d+)\s*s(?:ec(?:onds?)?)?)?",
    re.I,
)
# Parse "reset in Xh Ym" / "resets in X hours" / "resets in 2 days".
_RESET_IN_RE = re.compile(
    r"\brese(?:t|ts)\s+in\s+"
    r"(?:(?P<d>\d+)\s*d(?:ays?)?\s*)?"
    r"(?:(?P<h>\d+)\s*h(?:r|ours?)?\s*)?"
    r"(?:(?P<m>\d+)\s*m(?:in(?:utes?)?)?\s*)?"
    r"(?:(?P<s>\d+)\s*s(?:ec(?:onds?)?)?)?",
    re.I,
)
# Parse ISO 8601 timestamps (used to compute reset delta from now).
_ISO_TIMESTAMP_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)"
)
# Parse Claude Code session-limit messages such as
# "resets 1:30am (America/Los_Angeles)".
_ABSOLUTE_RESET_TIME_RE = re.compile(
    r"\brese(?:t|ts)(?:\s+at)?\s+"
    r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*"
    r"(?P<ampm>a\.?m\.?|p\.?m\.?)\s*"
    r"\((?P<timezone>[A-Za-z0-9_./+-]+)\)",
    re.I,
)
# Parse Codex plan-limit wording (#1281): "try again at 3:05 PM",
# "try again at 3:05 PM on 23 Oct" and "try again at Oct 23rd, 2026 3:05 PM".
# The time has no zone, so it is read in the host's local zone.
_CLOCK = r"(?P<{p}hour>\d{{1,2}}):(?P<{p}minute>\d{{2}})\s*(?P<{p}ampm>[ap]\.?m)\b\.?"
_TRY_AGAIN_AT_RE = re.compile(
    r"\btry\s+again\s+at\s+(?:"
    r"(?P<month_a>[A-Za-z]{3,9})\.?\s+(?P<day_a>\d{1,2})(?:st|nd|rd|th)?,?\s+"
    r"(?P<year_a>\d{4}),?\s+(?:at\s+)?" + _CLOCK.format(p="a_") +
    r"|" + _CLOCK.format(p="b_") +
    r"(?:\s+on\s+(?P<day_b>\d{1,2})(?:st|nd|rd|th)?\s+(?P<month_b>[A-Za-z]{3,9})\b\.?)?"
    r")",
    re.I,
)
_MONTHS = {
    name: index
    for index, names in enumerate(
        (
            ("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"),
            ("may",), ("jun", "june"), ("jul", "july"), ("aug", "august"),
            ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"),
            ("dec", "december"),
        ),
        start=1,
    )
    for name in names
}
_LOCALTIME_PATH = "/etc/localtime"


def _load_localtime_zone() -> zoneinfo.ZoneInfo | None:
    """The host's /etc/localtime zone, or None when it cannot be read."""
    try:
        with open(_LOCALTIME_PATH, "rb") as handle:
            return zoneinfo.ZoneInfo.from_file(handle)
    except (OSError, ValueError):
        return None


def _host_local_zone() -> zoneinfo.ZoneInfo | None:
    """Host zone with its real DST rules, or None; never substitutes another zone.

    An explicit TZ is authoritative: a ':'-prefixed or plain IANA key or an
    absolute tzfile path. Anything else (empty, a POSIX rule string, an unknown
    key) yields None instead of falling back to /etc/localtime.
    """
    if "TZ" not in os.environ:
        return _load_localtime_zone()
    value = os.environ["TZ"]
    if value.startswith(":"):
        value = value[1:]
    if not value:
        return None
    try:
        if os.path.isabs(value):
            with open(value, "rb") as handle:
                return zoneinfo.ZoneInfo.from_file(handle)
        return zoneinfo.ZoneInfo(value)
    except (OSError, ValueError, zoneinfo.ZoneInfoNotFoundError):
        return None


def _to_24_hour(hour: int, ampm: str) -> int | None:
    if not 1 <= hour <= 12:
        return None
    if ampm.lower().replace(".", "") == "am":
        return 0 if hour == 12 else hour
    return 12 if hour == 12 else hour + 12


def _local_occurrences(
    date: datetime.date, hour: int, minute: int, tz: zoneinfo.ZoneInfo
) -> list[datetime.datetime]:
    """UTC instants of a local wall time; empty in a spring-forward gap."""
    found: list[datetime.datetime] = []
    for fold in (0, 1):
        local = datetime.datetime(date.year, date.month, date.day, hour, minute, fold=fold, tzinfo=tz)
        as_utc = local.astimezone(datetime.timezone.utc)
        if as_utc.astimezone(tz).replace(tzinfo=None) != local.replace(tzinfo=None, fold=0):
            continue
        if as_utc not in found:
            found.append(as_utc)
    return sorted(found)


def _parse_try_again_at_seconds(
    text: str,
    *,
    now_utc: datetime.datetime | None = None,
    local_tz: zoneinfo.ZoneInfo | None = None,
) -> int | None:
    m = _TRY_AGAIN_AT_RE.search(text)
    if not m:
        return None
    if now_utc is None:
        now_utc = datetime.datetime.now(datetime.timezone.utc)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=datetime.timezone.utc)
    else:
        now_utc = now_utc.astimezone(datetime.timezone.utc)
    tz = local_tz if local_tz is not None else _host_local_zone()
    if tz is None:
        return None
    prefix = "a_" if m.group("a_hour") else "b_"
    hour = _to_24_hour(int(m.group(f"{prefix}hour")), m.group(f"{prefix}ampm"))
    minute = int(m.group(f"{prefix}minute"))
    if hour is None or not 0 <= minute <= 59:
        return None
    today = now_utc.astimezone(tz).date()
    month_name = m.group("month_a") or m.group("month_b")
    if month_name is None:
        # A bare clock time: the next future occurrence, today or tomorrow. A
        # wall time that does not exist (a spring-forward gap) is None, never
        # silently rolled to another day.
        for date in (today, today + datetime.timedelta(days=1)):
            occurrences = _local_occurrences(date, hour, minute, tz)
            if not occurrences:
                return None
            future = [t for t in occurrences if t > now_utc]
            if future:
                return int((future[0] - now_utc).total_seconds())
        return None
    month = _MONTHS.get(month_name.lower())
    if month is None:
        return None
    day = int(m.group("day_a") or m.group("day_b"))
    year = int(m.group("year_a")) if m.group("year_a") else today.year
    try:
        date = datetime.date(year, month, day)
    except ValueError:
        return None
    # A dated time that is not in the future fails safe to None.
    future = [t for t in _local_occurrences(date, hour, minute, tz) if t > now_utc]
    if not future:
        return None
    return int((future[0] - now_utc).total_seconds())


def _parse_absolute_reset_seconds(
    text: str,
    *,
    now_utc: datetime.datetime | None = None,
) -> int | None:
    m = _ABSOLUTE_RESET_TIME_RE.search(text)
    if not m:
        return None

    try:
        tz = zoneinfo.ZoneInfo(m.group("timezone"))
    except zoneinfo.ZoneInfoNotFoundError:
        return None

    if now_utc is None:
        now_utc = datetime.datetime.now(datetime.timezone.utc)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=datetime.timezone.utc)
    else:
        now_utc = now_utc.astimezone(datetime.timezone.utc)

    hour = int(m.group("hour"))
    minute = int(m.group("minute") or 0)
    ampm = m.group("ampm").lower().replace(".", "")
    if not 1 <= hour <= 12 or not 0 <= minute <= 59:
        return None
    if ampm == "am":
        hour = 0 if hour == 12 else hour
    else:
        hour = 12 if hour == 12 else hour + 12

    now_local = now_utc.astimezone(tz)
    reset_local = now_local.replace(
        hour=hour,
        minute=minute,
        second=0,
        microsecond=0,
    )
    if reset_local <= now_local:
        reset_local += datetime.timedelta(days=1)

    return int((reset_local.astimezone(datetime.timezone.utc) - now_utc).total_seconds())


def _parse_rate_limit_reset_seconds(
    text: str,
    *,
    now_utc: datetime.datetime | None = None,
) -> int | None:
    """Extract the reset wait time in seconds from a rate-limit error message.

    Returns None if the reset time cannot be reliably parsed.
    """
    m = _RETRY_AFTER_SECONDS_RE.search(text)
    if m:
        val = m.group(1) or m.group(2)
        if val:
            return int(val)

    m = _TRY_AGAIN_IN_RE.search(text)
    if m and any(m.group(g) for g in ("d", "h", "m", "s")):
        return (
            int(m.group("d") or 0) * 86400
            + int(m.group("h") or 0) * 3600
            + int(m.group("m") or 0) * 60
            + int(m.group("s") or 0)
        )

    m = _RESET_IN_RE.search(text)
    if m and any(m.group(g) for g in ("d", "h", "m", "s")):
        return (
            int(m.group("d") or 0) * 86400
            + int(m.group("h") or 0) * 3600
            + int(m.group("m") or 0) * 60
            + int(m.group("s") or 0)
        )

    m = _ISO_TIMESTAMP_RE.search(text)
    if m:
        try:
            ts_str = m.group(1).replace(" ", "T")
            if not ts_str.endswith("Z"):
                ts_str += "Z"
            ts = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            base_now = now_utc or datetime.datetime.now(datetime.timezone.utc)
            if base_now.tzinfo is None:
                base_now = base_now.replace(tzinfo=datetime.timezone.utc)
            else:
                base_now = base_now.astimezone(datetime.timezone.utc)
            delta = int((ts - base_now).total_seconds())
            if delta > 0:
                return delta
        except (ValueError, OverflowError):
            pass

    reset_secs = _parse_try_again_at_seconds(text, now_utc=now_utc)
    if reset_secs is not None:
        return reset_secs

    reset_secs = _parse_absolute_reset_seconds(text, now_utc=now_utc)
    if reset_secs is not None:
        return reset_secs

    return None
