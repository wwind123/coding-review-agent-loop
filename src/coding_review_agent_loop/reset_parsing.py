"""Pure rate-limit reset-time parsing (stdlib only).

Moved verbatim out of ``agent_failure`` so ``transient`` can use it without an
import cycle; ``agent_failure`` re-exports every name.
"""

from __future__ import annotations

import datetime
import re
import zoneinfo

# Parse "Retry-After: N" (HTTP header) or "retry after N" or "retryDelay: Ns" (gRPC).
_RETRY_AFTER_SECONDS_RE = re.compile(
    r"\bretry[- ]after[:\s]+(\d+)\b"
    r"|\bretry[_-]?delay[:\s]+['\"]?(\d+)s['\"]?",
    re.I,
)
# Parse "try again in Xh Ym Zs".
_TRY_AGAIN_IN_RE = re.compile(
    r"\btry\s+again\s+in\s+"
    r"(?:(?P<h>\d+)\s*h(?:r|ours?)?\s*)?"
    r"(?:(?P<m>\d+)\s*m(?:in(?:utes?)?)?\s*)?"
    r"(?:(?P<s>\d+)\s*s(?:ec(?:onds?)?)?)?",
    re.I,
)
# Parse "reset in Xh Ym" / "resets in X hours".
_RESET_IN_RE = re.compile(
    r"\brese(?:t|ts)\s+in\s+"
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
    if m and any(m.group(g) for g in ("h", "m", "s")):
        return (
            int(m.group("h") or 0) * 3600
            + int(m.group("m") or 0) * 60
            + int(m.group("s") or 0)
        )

    m = _RESET_IN_RE.search(text)
    if m and any(m.group(g) for g in ("h", "m", "s")):
        return (
            int(m.group("h") or 0) * 3600
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

    reset_secs = _parse_absolute_reset_seconds(text, now_utc=now_utc)
    if reset_secs is not None:
        return reset_secs

    return None
