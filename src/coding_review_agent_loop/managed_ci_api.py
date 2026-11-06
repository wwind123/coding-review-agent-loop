"""Managed-CI read helpers and shared list-failure diagnostics."""
from __future__ import annotations
from .managed_ci_tuple import MANAGED_LABEL
import functools
from collections.abc import Mapping
from .github import note_host_footer_observed
from .protocol_markers import strip_known_host_footer
import json
import re
import threading
from pathlib import Path
from typing import Literal
from .config import AgentLoopConfig
from .errors import AgentLoopError
from .logging import log
from .github_retry import describe_gh_failure, run_gh_read
from .runner import Runner
from .workdirs import github_api_cwd
from .managed_ci_types import (
    CompleteLabelHistory,
    LabelEvent,
    ManagedActorVariable,
    ManagedCiProbeContext,
    UnknownLabelHistory,
    WORKFLOW_FILE,
)


_LIST_FAILURE = threading.local()


def _last_list_failure() -> str:
    """Reason of the most recent ``_api_list`` failure on this thread ('' if it succeeded)."""
    return getattr(_LIST_FAILURE, "reason", "")


def _surfaces_list_failures(func):
    """Append GitHub list-read failure diagnostics to terminal errors.

    Many managed-CI gates fail closed on an unreadable list and later raise a
    generic refusal.  Within one outermost gate call, every list failure is
    remembered, so the terminal ``AgentLoopError`` carries the final stderr and
    retry attempt history even when the failing read was consumed by a helper
    that only returns ``None``.  Existing fail-closed decisions are unchanged.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        outermost = getattr(_LIST_FAILURE, "depth", 0) == 0
        if outermost:
            _LIST_FAILURE.failures = []
        _LIST_FAILURE.depth = getattr(_LIST_FAILURE, "depth", 0) + 1
        try:
            return func(*args, **kwargs)
        except AgentLoopError as exc:
            failures = list(getattr(_LIST_FAILURE, "failures", ()) or ())
            missing = [reason for reason in failures if reason not in str(exc)]
            if missing and type(exc) in (AgentLoopError,):
                detail = "\n".join(f"GitHub list read failure: {reason}" for reason in missing)
                raise AgentLoopError(f"{exc}\n{detail}") from exc
            raise
        finally:
            _LIST_FAILURE.depth -= 1
            if outermost:
                _LIST_FAILURE.failures = None

    return wrapper


def _probe(runner: Runner, context: ManagedCiProbeContext, endpoint: str, *, raw: bool = False):
    command = [context.gh_cmd, "api", endpoint]
    if raw:
        command.extend(["-H", "Accept: application/vnd.github.raw+json"])
    return runner.run(command, cwd=context.cwd, check=False)


def _probe_json(runner: Runner, context: ManagedCiProbeContext, endpoint: str) -> tuple[dict[str, object] | None, object]:
    result = _probe(runner, context, endpoint)
    if result.returncode != 0:
        return None, result
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None, result
    return (payload if isinstance(payload, dict) else None), result


def _probe_json_list(
    runner: Runner, context: ManagedCiProbeContext, endpoint: str
) -> tuple[list[object] | None, object]:
    """Read a GitHub list response without treating a valid array as malformed."""
    result = _probe(runner, context, endpoint)
    if result.returncode != 0:
        return None, result
    try:
        payload = json.loads(result.stdout or "")
    except json.JSONDecodeError:
        return None, result
    return (payload if isinstance(payload, list) else None), result


def _is_http_error(result: object, *statuses: int) -> bool:
    """`gh api` uses exit 1 for HTTP errors, so inspect its diagnostic text."""
    combined = " ".join(
        str(getattr(result, field, "") or "") for field in ("stdout", "stderr")
    ).casefold()
    return any(str(status) in combined for status in statuses)


def _is_plan_limited_error(result: object) -> bool:
    """Recognize GitHub's plan restriction independently of endpoint wording."""
    combined = " ".join(
        str(getattr(result, field, "") or "") for field in ("stdout", "stderr")
    ).casefold()
    return "upgrade to github pro" in combined or "make this repository public" in combined


_GH_HTTP_STATUS_LINE = re.compile(r"^gh: .*\(HTTP (\d{3})\)[ \t\r]*$", re.MULTILINE)


def _http_status(result: object) -> int | None:
    """Return the HTTP status from gh's own stderr diagnostic, strictly.

    ``gh api`` writes the API response body to stdout, and that body may quote
    arbitrary text, so stdout is never inspected.  Only stderr lines shaped
    like ``gh: <message> (HTTP 403)`` count, and every such line must agree.
    A missing, mid-line, or conflicting status yields ``None`` so no
    status-specific fallback applies.
    """
    if getattr(result, "returncode", 0) == 0:
        return None
    stderr = str(getattr(result, "stderr", "") or "")
    statuses = {int(match) for match in _GH_HTTP_STATUS_LINE.findall(stderr)}
    if len(statuses) != 1:
        return None
    return next(iter(statuses))


def _read_managed_actor_variable(
    runner: Runner, gh_cmd: str, repo: str, cwd: Path
) -> ManagedActorVariable:
    """Classify the ``AGENT_LOOP_MANAGED_ACTOR`` read for every identity check."""
    return _read_actions_variable(runner, gh_cmd, repo, cwd, "AGENT_LOOP_MANAGED_ACTOR")


def _read_actions_variable(
    runner: Runner, gh_cmd: str, repo: str, cwd: Path, name: str
) -> ManagedActorVariable:
    """Classify one repository Actions variable read.

    A strict HTTP 403 is checked before the legacy 404 text match, so a
    refused read whose body happens to mention 404 is never "absent".
    """
    endpoint = f"repos/{repo}/actions/variables/{name}"
    result = runner.run([gh_cmd, "api", endpoint], cwd=cwd, check=False)
    if result.returncode == 0:
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            payload = None
        value = payload.get("value") if isinstance(payload, dict) else None
        if isinstance(value, str):
            return ManagedActorVariable(value, "readable", f"{endpoint} is readable")
        if isinstance(payload, dict):
            # Historically a readable response without a value advertises no
            # actor (an identity mismatch), not an unreadable probe.
            return ManagedActorVariable(None, "absent", f"{endpoint} has no string value")
        return ManagedActorVariable(None, "error", f"{endpoint} returned a malformed response")
    if _http_status(result) == 403:
        return ManagedActorVariable(
            None, "unreadable", f"{endpoint} is not readable by this token (HTTP 403)"
        )
    if _is_http_error(result, 404):
        return ManagedActorVariable(None, "absent", f"{endpoint} is not configured")
    status = _http_status(result)
    return ManagedActorVariable(
        None, "error",
        f"{endpoint} could not be read" + (f" (HTTP {status})" if status is not None else ""),
    )


def _resolve_advertised_actor(
    variable: ManagedActorVariable, configured: str | None
) -> tuple[str | None, Literal["variable", "asserted"] | None]:
    """Return the advertised actor and where it came from.

    Only a strict 403 lets the operator's trusted actor stand in.  A readable
    variable always wins, and absence or any other error yields no actor.
    """
    if variable.status == "readable":
        return variable.value, "variable"
    configured = (configured or "").strip()
    if variable.status == "unreadable" and configured:
        return configured, "asserted"
    return None, None


_ASSERTED_ACTOR_LOGGED: set[tuple[str, str]] = set()


def _log_asserted_actor(config: AgentLoopConfig, actor: str) -> None:
    key = (config.repo.casefold(), actor.casefold())
    if key in _ASSERTED_ACTOR_LOGGED:
        return
    _ASSERTED_ACTOR_LOGGED.add(key)
    log(
        config,
        "AGENT_LOOP_MANAGED_ACTOR is not readable by this token (HTTP 403); "
        f"--managed-ci-trusted-actor={actor} is asserted and unverified locally. "
        "The workflow enforces vars.AGENT_LOOP_MANAGED_ACTOR server-side: a mismatch leaves "
        "ordinary PR CI unsuppressed and fails managed dispatch validation.",
    )


def _advertised_managed_actor(runner: Runner, config: AgentLoopConfig) -> str | None:
    """Resolve the advertised actor for a config-bearing identity check."""
    variable = _read_managed_actor_variable(
        runner, config.gh_cmd, config.repo, github_api_cwd()
    )
    advertised, source = _resolve_advertised_actor(variable, config.managed_ci_trusted_actor)
    if source == "asserted" and advertised is not None:
        _log_asserted_actor(config, advertised)
    return advertised


def _probe_raw_workflow(
    runner: Runner, context: ManagedCiProbeContext, base: str
) -> tuple[str | None, object]:
    result = _probe(
        runner, context,
        f"repos/{context.repo}/contents/.github/workflows/{WORKFLOW_FILE}?ref={base}", raw=True,
    )
    return ((result.stdout or "") if result.returncode == 0 else None), result


def _historical_managed_label_event(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    actor_login: str,
    actor_id: int,
) -> tuple[int, str, int] | None:
    events = _managed_label_event_history(
        runner,
        config=config,
        pr_number=pr_number,
        actor_login=actor_login,
        actor_id=actor_id,
    )
    return events[-1] if events else None


def _managed_label_event_history(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    actor_login: str,
    actor_id: int,
) -> list[tuple[int, str, int]] | None:
    """Return every actor-owned managed-label application in timeline order."""
    events = _api_list(
        runner, config, f"repos/{config.repo}/issues/{pr_number}/events?per_page=100"
    )
    if events is None:
        return None
    candidates: list[tuple[int, str, int]] = []
    for event in events:
        label = event.get("label") if isinstance(event.get("label"), dict) else {}
        actor = event.get("actor") if isinstance(event.get("actor"), dict) else {}
        event_id = event.get("id")
        login = actor.get("login")
        identity = actor.get("id")
        if (
            label.get("name") == MANAGED_LABEL
            and event.get("event") == "labeled"
            and isinstance(event_id, int)
            and isinstance(login, str)
            and isinstance(identity, int)
            and login.casefold() == actor_login.casefold()
            and identity == actor_id
        ):
            candidates.append((event_id, login, identity))
    return sorted(candidates, key=lambda item: item[0])


def _api_list_detailed(
    runner: Runner, config: AgentLoopConfig, endpoint: str
) -> tuple[list[dict[str, object]] | None, str]:
    payload, reason = _api_list_detailed_inner(runner, config, endpoint)
    _LIST_FAILURE.reason = reason if payload is None else ""
    if payload is None and reason:
        failures = getattr(_LIST_FAILURE, "failures", None)
        if failures is not None and reason not in failures:
            failures.append(reason)
    return payload, reason


def _api_list_detailed_inner(
    runner: Runner, config: AgentLoopConfig, endpoint: str
) -> tuple[list[dict[str, object]] | None, str]:
    """Fetch a paginated GitHub list; on failure return ``(None, reason)``.

    ``reason`` carries the final stderr and retry attempt history so raising
    callers can include it while keeping their fail-closed behavior.
    """
    result = run_gh_read(
        runner, [config.gh_cmd, "api", "--paginate", endpoint], cwd=github_api_cwd(), check=False
    )
    if result.returncode != 0:
        reason = describe_gh_failure(result)
        log(config, f"GitHub list read of {endpoint} failed: {reason}")
        return None, reason
    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return None, f"{endpoint} returned malformed JSON"
    if not isinstance(payload, list):
        return None, f"{endpoint} returned a non-list payload"
    # GitHub CLI 2.45 emits one flat array for paginated array endpoints.
    # Never silently discard malformed entries: doing so could turn an
    # incomplete timeline into an apparently complete ownership record.
    if not all(isinstance(item, dict) for item in payload):
        return None, f"{endpoint} returned malformed entries"
    return payload, ""


def _with_reason(message: str, reason: str) -> str:
    return f"{message}\n{reason}" if reason else message


def _api_list(runner: Runner, config: AgentLoopConfig, endpoint: str) -> list[dict[str, object]] | None:
    """Fetch a paginated GitHub list, returning None for an uninspectable response."""
    return _api_list_detailed(runner, config, endpoint)[0]


def label_event_history(
    runner: Runner, *, config: AgentLoopConfig, pr_number: int, label_name: str
) -> CompleteLabelHistory | UnknownLabelHistory:
    """Read one label's latest labeled/unlabeled event completely, or ``Unknown``.

    Missing actor IDs, malformed pagination and a malformed latest event are
    unreadable, never an empty history.
    """
    events, reason = _api_list_detailed(
        runner, config, f"repos/{config.repo}/issues/{pr_number}/events?per_page=100"
    )
    if events is None:
        return UnknownLabelHistory(reason or "event list unreadable")
    latest: LabelEvent | None = None
    for event in events:
        kind = event.get("event")
        if not isinstance(kind, str) or not kind:
            # An envelope without a usable discriminator could be a label
            # transition; the history is unreadable, never merely filtered.
            return UnknownLabelHistory("a timeline event lacked an event kind")
        if kind not in {"labeled", "unlabeled"}:
            continue
        # A label transition whose label cannot be identified might belong to
        # this label; the history is then unreadable, never merely absent.
        label = event.get("label")
        name = label.get("name") if isinstance(label, dict) else None
        if not isinstance(name, str) or not name:
            return UnknownLabelHistory("a label transition event lacked a label name")
        if name != label_name:
            continue
        actor = event.get("actor")
        event_id = event.get("id")
        login = actor.get("login") if isinstance(actor, dict) else None
        actor_id = actor.get("id") if isinstance(actor, dict) else None
        if (
            isinstance(event_id, bool) or not isinstance(event_id, int) or event_id < 1
            or not isinstance(login, str) or not login
            or isinstance(actor_id, bool) or not isinstance(actor_id, int) or actor_id < 1
        ):
            return UnknownLabelHistory(f"a `{label_name}` event lacked an id or actor")
        if latest is not None and event_id <= latest.event_id:
            return UnknownLabelHistory(f"`{label_name}` event ids were not strictly increasing")
        latest = LabelEvent(event_id, str(event["event"]), login, actor_id)
    return CompleteLabelHistory(latest)


def _active_managed_label_event(
    runner: Runner, *, config: AgentLoopConfig, pr_number: int
) -> tuple[int, str, int] | None:
    """Return the last label application, but only while it remains active.

    Timeline provenance is an enforcement input for adoption.  Treat missing
    actor IDs, malformed pagination, and an intervening unlabel as untrusted.
    """
    history = label_event_history(
        runner, config=config, pr_number=pr_number, label_name=MANAGED_LABEL
    )
    if not isinstance(history, CompleteLabelHistory):
        return None
    latest = history.latest
    if latest is None or latest.kind != "labeled":
        return None
    return latest.event_id, latest.actor_login, latest.actor_id


def _read_repo_default_branch(runner: Runner, gh_cmd: str, repo: str, cwd: Path) -> str | None:
    """The live default branch, or ``None`` when it cannot be established."""
    result = runner.run([gh_cmd, "api", f"repos/{repo}"], cwd=cwd, check=False)
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None
    value = payload.get("default_branch") if isinstance(payload, dict) else None
    return value if isinstance(value, str) and value else None


def _api_json(runner: Runner, config: AgentLoopConfig, endpoint: str, *, quiet: bool = False) -> dict[str, object]:
    result = runner.run(
        [config.gh_cmd, "api", endpoint], cwd=github_api_cwd(), check=False
    )
    if result.returncode != 0:
        if quiet:
            return {}
        raise AgentLoopError(f"Managed-CI API request failed: {endpoint}.")
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise AgentLoopError(f"Managed-CI API response was invalid JSON: {endpoint}.") from exc
    return payload if isinstance(payload, dict) else {}


def _normalized_comment_body(
    comment: Mapping[str, object], *, config: AgentLoopConfig | None
) -> str:
    """Return one raw ``_api_list`` comment body with the host footer removed once.

    This is the managed-CI comment-ingestion boundary (#1043).  Strict record
    parsers downstream never strip, so exactly one known host footer is
    tolerated along any path and a doubled footer still reaches the parser.
    """
    raw = comment.get("body")
    if not isinstance(raw, str):
        return ""
    body, footered = strip_known_host_footer(raw)
    if footered:
        note_host_footer_observed(config, "managed-CI record re-read")
    return body
