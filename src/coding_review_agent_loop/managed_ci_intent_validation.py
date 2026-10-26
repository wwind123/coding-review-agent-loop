"""Pure managed-CI intent validation, mirrored by the base workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import json
import re

from .protocol_markers import KNOWN_HOST_COMMENT_FOOTER


# Mirror of the base workflow's MANAGED_CI_V2_VALIDATOR block in
# .github/workflows/ci.yml.  Keep the key set, patterns, states, envelope
# forms, and failure order in lockstep with that block; the page-level
# parity test runs both against one corpus.
INTENT_REQUIRED_KEYS = frozenset({
    "version", "repository", "pr", "expected_head_sha", "base_ref",
    "workflow_revision", "generation", "nonce", "created_at", "state",
    "run_id", "run_attempt", "terminal_run_id", "terminal_run_attempt",
    "terminal_outcome", "terminal_attempts",
})
INTENT_ALLOWED_STATES = frozenset({"prepared", "dispatch-requested", "attached", "completed"})
INTENT_SHA_PATTERN = r"[0-9a-f]{40}"
INTENT_NONCE_PATTERN = r"[A-Za-z0-9_-]{32}"
_INTENT_VISIBLE_LINE = r"(?:Managed CI authorization for exact head (?P<visible_sha>[0-9a-f]{40})\.\n\n)?"
_INTENT_RECORD = r"<!-- AGENT_MANAGED_CI_INTENT_V2 (?P<payload>.*?) -->"
_INTENT_ENVELOPE_RE = re.compile(r"^" + _INTENT_VISIBLE_LINE + _INTENT_RECORD + r"$")
_INTENT_FOOTERED_ENVELOPE_RE = re.compile(
    _INTENT_VISIBLE_LINE + _INTENT_RECORD + re.escape(KNOWN_HOST_COMMENT_FOOTER)
)


@dataclass(frozen=True)
class IntentBinding:
    """The PR/head tuple a workflow binds an intent record to."""

    repository: str | None
    pr: int | None
    expected_head_sha: str | None
    base_ref: str | None
    workflow_revision: str | None
    nonce: object
    require_generation: bool = True


@dataclass(frozen=True)
class IntentPageVerdict:
    """The workflow's decision for one requested nonce over one comment page.

    ``recoverable`` is a lone valid ``prepared`` record: the workflow fails it
    as not yet a dispatch authorization, but the producer may advance it.
    """

    outcome: Literal["fail", "recoverable", "authorized"]
    record: dict[str, object] | None = None
    comment_id: int | None = None
    reason: str | None = None


def match_intent_envelope(
    body: object, *, visible_capable: bool, host_footer_capable: bool
) -> re.Match[str] | None:
    """Match one comment body exactly as the base workflow's envelope does.

    The bare and visible-line forms are matched against the stripped body.
    The footered form is matched against the raw body, only when the
    workflow advertises it, so no whitespace or doubled footer is admitted.
    """
    if not isinstance(body, str):
        return None
    match = _INTENT_ENVELOPE_RE.fullmatch(body.strip())
    if match is None and host_footer_capable:
        match = _INTENT_FOOTERED_ENVELOPE_RE.fullmatch(body)
    if match is None:
        return None
    if match.group("visible_sha") is not None and not visible_capable:
        return None
    return match


def _exact_int(value: object, *, positive: bool = False) -> bool:
    return type(value) is int and (not positive or value > 0)


def _exact_string(value: object, pattern: str | None = None) -> bool:
    return type(value) is str and (pattern is None or re.fullmatch(pattern, value) is not None)


def validate_intent_record(
    record: dict[str, object], match: re.Match[str], *, binding: IntentBinding
) -> str | None:
    """Return the workflow's failure reason for one same-nonce record, if any."""
    visible_sha = match.group("visible_sha")
    if visible_sha is not None and visible_sha != record.get("expected_head_sha"):
        return "visible intent line disagrees with the authorized head"
    if set(record) != INTENT_REQUIRED_KEYS:
        return "trusted intent has an invalid schema"
    checks: tuple[tuple[bool, str], ...] = (
        (_exact_string(record.get("repository"), r"[^/\s]+/[^/\s]+"), "repository"),
        (_exact_int(record.get("pr"), positive=True), "pr"),
        (_exact_string(record.get("expected_head_sha"), INTENT_SHA_PATTERN), "expected_head_sha"),
        (_exact_string(record.get("base_ref"), r"[^\s]+"), "base_ref"),
        (_exact_string(record.get("workflow_revision"), INTENT_SHA_PATTERN), "workflow_revision"),
    )
    for valid, name in checks:
        if not valid:
            return "invalid " + name
    if binding.require_generation or record.get("generation") is not None:
        if not _exact_string(record.get("generation"), r"[A-Za-z0-9_-]+"):
            return "invalid generation"
    if not _exact_string(record.get("nonce"), INTENT_NONCE_PATTERN):
        return "invalid nonce"
    if not _exact_int(record.get("created_at"), positive=True):
        return "invalid created_at"
    state = record.get("state")
    if not _exact_string(state, r"(?:prepared|dispatch-requested|attached|completed)"):
        return "invalid state"
    if state not in INTENT_ALLOWED_STATES:
        return "invalid lifecycle state"
    run_id, run_attempt = record.get("run_id"), record.get("run_attempt")
    if state in {"prepared", "dispatch-requested"}:
        if run_id is not None or run_attempt is not None:
            return "early lifecycle state has run fields"
    elif not _exact_int(run_id, positive=True) or not _exact_int(run_attempt, positive=True):
        return "attached lifecycle state lacks paired run fields"
    terminal_id = record.get("terminal_run_id")
    terminal_attempt = record.get("terminal_run_attempt")
    if terminal_id is not None and not _exact_int(terminal_id, positive=True):
        return "invalid terminal run ID"
    if terminal_id is None and terminal_attempt is not None:
        return "terminal attempt has no terminal run ID"
    if terminal_id is not None and not _exact_int(terminal_attempt, positive=True):
        return "invalid terminal run attempt"
    outcome = record.get("terminal_outcome")
    if outcome not in {None, "no-status"}:
        return "invalid terminal outcome"
    attempts = record.get("terminal_attempts")
    if not isinstance(attempts, list):
        return "terminal attempt history is not a list"
    seen_attempts: set[tuple[object, object]] = set()
    for item in attempts:
        if not isinstance(item, dict) or set(item) != {"run_id", "run_attempt"}:
            return "malformed terminal attempt history"
        item_id, item_attempt = item["run_id"], item["run_attempt"]
        if not _exact_int(item_id, positive=True):
            return "invalid terminal history run ID"
        if not _exact_int(item_attempt, positive=True):
            return "invalid terminal history run attempt"
        key = (item_id, item_attempt)
        if key in seen_attempts:
            return "duplicate terminal attempt history"
        seen_attempts.add(key)
    terminal_key = (terminal_id, terminal_attempt)
    if terminal_id is not None and terminal_key not in seen_attempts:
        return "terminal run is absent from terminal attempt history"
    if state == "prepared" and (
        terminal_id is not None or terminal_attempt is not None or outcome is not None or attempts
    ):
        return "prepared intent has terminal lifecycle fields"
    if state == "dispatch-requested" and terminal_id is not None and outcome != "no-status":
        return "dispatch-requested retry lacks no-status outcome"
    if state == "attached" and outcome is not None:
        return "attached intent has terminal outcome"
    if state == "completed" and outcome == "no-status" and (run_id, run_attempt) != terminal_key:
        return "completed no-status record is not bound to its terminal run"
    if state in {"attached", "completed"} and outcome is None and (run_id, run_attempt) in seen_attempts:
        return "active run is already terminal history"
    expected = {
        "repository": binding.repository,
        "pr": binding.pr,
        "expected_head_sha": binding.expected_head_sha,
        "base_ref": binding.base_ref,
        "workflow_revision": binding.workflow_revision,
        "nonce": binding.nonce,
    }
    if any(record[key] != value for key, value in expected.items()):
        return "trusted intent binding drifted"
    return None


def _walk_intent_page(
    page: list[object],
    *,
    requested_nonce: object,
    check_nonce: bool,
    trusted_login: str,
    trusted_id: int | None,
    visible_capable: bool,
    host_footer_capable: bool,
    binding: IntentBinding | None,
) -> tuple[str | None, list[tuple[int | None, dict[str, object]]]]:
    """Walk one page in the workflow's exact order.

    Returns the first failure reason (naming the comment) and the validated
    records for ``requested_nonce``.  With ``check_nonce`` false the walk
    stops before nonce comparison, reporting only nonce-independent failures.
    """
    validated: list[tuple[int | None, dict[str, object]]] = []
    for index, comment in enumerate(page):
        if not isinstance(comment, dict):
            return f"comment #{index}: malformed comment object", validated
        comment_id = comment.get("id") if _exact_int(comment.get("id")) else None
        label = f"comment {comment_id}" if comment_id is not None else f"comment #{index}"
        user = comment.get("user")
        if not isinstance(user, dict) or type(user.get("login")) is not str:
            return f"{label}: malformed comment author", validated
        if user["login"] != trusted_login:
            continue
        if trusted_id is not None and user.get("id") != trusted_id:
            return f"{label}: trusted comment author ID drifted", validated
        body = comment.get("body")
        if type(body) is not str:
            return f"{label}: trusted comment body is not text", validated
        match = match_intent_envelope(
            body, visible_capable=visible_capable, host_footer_capable=host_footer_capable
        )
        if match is None:
            continue
        try:
            record = json.loads(match.group("payload"))
        except json.JSONDecodeError:
            return f"{label}: trusted intent envelope contains malformed JSON", validated
        if not isinstance(record, dict):
            return f"{label}: trusted intent payload is not an object", validated
        if record.get("version") != 2:
            return f"{label}: unsupported intent version", validated
        if not check_nonce or record.get("nonce") != requested_nonce:
            continue
        assert binding is not None
        reason = validate_intent_record(record, match, binding=binding)
        if reason is not None:
            return f"{label}: {reason}", validated
        validated.append((comment_id, record))
    return None, validated


def intent_page_pre_nonce_fatal(
    page: list[object],
    *,
    trusted_login: str,
    trusted_id: int | None,
    visible_capable: bool,
    host_footer_capable: bool,
) -> str | None:
    """Return the first failure the workflow raises for every requested nonce."""
    reason, _ = _walk_intent_page(
        page,
        requested_nonce=None,
        check_nonce=False,
        trusted_login=trusted_login,
        trusted_id=trusted_id,
        visible_capable=visible_capable,
        host_footer_capable=host_footer_capable,
        binding=None,
    )
    return reason


def classify_intent_page(
    page: list[object],
    *,
    requested_nonce: object,
    trusted_login: str,
    trusted_id: int | None,
    visible_capable: bool,
    host_footer_capable: bool,
    binding: IntentBinding,
) -> IntentPageVerdict:
    """Mirror the workflow's ``validate()`` decision for one requested nonce.

    Freshness is deliberately not judged here: the workflow checks the age
    window separately in its dispatch validator.
    """
    reason, validated = _walk_intent_page(
        page,
        requested_nonce=requested_nonce,
        check_nonce=True,
        trusted_login=trusted_login,
        trusted_id=trusted_id,
        visible_capable=visible_capable,
        host_footer_capable=host_footer_capable,
        binding=binding,
    )
    if reason is not None:
        return IntentPageVerdict("fail", reason=reason)
    if len(validated) != 1:
        return IntentPageVerdict(
            "fail", reason="expected exactly one fresh intent for requested nonce"
        )
    comment_id, record = validated[0]
    if record["state"] == "prepared":
        return IntentPageVerdict(
            "recoverable",
            record=record,
            comment_id=comment_id,
            reason="prepared intent is not a dispatch authorization",
        )
    return IntentPageVerdict("authorized", record=record, comment_id=comment_id)


