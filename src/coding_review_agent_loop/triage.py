"""Advisory issue triage before an issue run (#1434, stage 2 of #1430).

``agent-loop issue N --triage`` arms a one-shot hook on the run's
:class:`~coding_review_agent_loop.usage.RunUsageContext`.  The hook fires at the
run's first agent dispatch, so every startup guard and refusal has already run.
It makes one validated model call that recommends ``discuss-first``,
``plan-first`` or ``direct``, posts the recommendation on the issue, and logs it.
The route the run takes is always the one the operator selected: nothing here
reads or changes a routing argument, and any failure is logged and ignored.

Signals and similar past issues are computed deterministically from the issue
text and an optional stage-1 ``history export`` file; the model only reads them.
"""

from __future__ import annotations

import base64
import binascii
import datetime
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace as dataclasses_replace
from pathlib import Path

from .agents.base import AgentName
from .agents.registry import agent_display_name
from .config import AgentLoopConfig, configured_model_for, reviewers
from .errors import AgentLoopError
from .github import IssueContext, post_trusted_issue_comment
from .history_export import (
    SCHEMA_VERSION as HISTORY_SCHEMA_VERSION,
    EVIDENCE_VERIFIED,
    SNAPSHOT_DIGEST_NORMALIZED,
    SNAPSHOT_DIGEST_VERIFIED,
    SNAPSHOT_NEVER_EDITED,
    SNAPSHOT_TIME_INFERRED,
)
from .logging import log
from .prompts import build_triage_prompt
from .protocol_markers import ISSUE_COMMENT_SURFACE, TrustedBody, sanitize_untrusted_prose
from .response_validation import TriageAnswer, parse_triage_response
from .reviewer_seats import ReviewerSeat, SeatAgent, seat_backend
from .runner import Runner
from .usage import RunUsageContext
from .validated_agent import _run_validated_agent

TRIAGE_RECORD_MARKER = "AGENT_ISSUE_TRIAGE"
TRIAGE_RECORD_KIND = "issue_triage"
TRIAGE_RECORD_SCHEMA_VERSION = 1
TRIAGE_COMMENT_HEADING = "### Advisory issue triage"
MAX_SIMILAR_ISSUES = 5

HINT_DISCUSS = "discuss-first"
HINT_DIRECT = "direct"

SIGNAL_OPEN_QUESTION = "open-design-question"
SIGNAL_EDGE_CASES = "edge-cases-without-closed-criteria"
SIGNAL_COMPETING_DESIGNS = "competing-designs"
SIGNAL_PRIOR_STALL = "prior-stalled-plan"
SIGNAL_LOCAL_CHANGE = "named-local-change"
SIGNAL_SPECIFIED_BEHAVIOR = "specified-behavior"
SIGNAL_CLOSED_CRITERIA = "closed-acceptance-criteria"

SIGNAL_HINTS: Mapping[str, str] = {
    SIGNAL_OPEN_QUESTION: HINT_DISCUSS,
    SIGNAL_EDGE_CASES: HINT_DISCUSS,
    SIGNAL_COMPETING_DESIGNS: HINT_DISCUSS,
    SIGNAL_PRIOR_STALL: HINT_DISCUSS,
    SIGNAL_LOCAL_CHANGE: HINT_DIRECT,
    SIGNAL_SPECIFIED_BEHAVIOR: HINT_DIRECT,
    SIGNAL_CLOSED_CRITERIA: HINT_DIRECT,
}
# Names a model reason may cite besides the deterministic signals.
REASON_ONLY_SIGNALS = ("similar-issue-outcome", "code-context", "no-signal")
TRIAGE_REASON_SIGNALS = (*SIGNAL_HINTS, *REASON_ONLY_SIGNALS)

USABLE_SNAPSHOT_STATUSES = frozenset({
    SNAPSHOT_DIGEST_VERIFIED, SNAPSHOT_DIGEST_NORMALIZED, SNAPSHOT_NEVER_EDITED, SNAPSHOT_TIME_INFERRED,
})

_OPEN_QUESTION_RE = re.compile(
    r"\b(?:how|what|which|where|when) should\b[^?\n]*\?"
    r"|\bshould (?:we|it|this|they)\b[^?\n]*\?"
    r"|\bwhich (?:approach|design|option|policy)\b",
    re.I,
)
_EDGE_CASE_RE = re.compile(
    r"\b(?:edge[- ]cases?|corner[- ]cases?|degraded|malformed|mixed-type|non-list"
    r"|unexpected (?:input|values?|shapes?)|invalid (?:input|values?|shapes?))\b",
    re.I,
)
_CLOSED_CRITERIA_RE = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*)?(?:acceptance(?: criteria)?|scope decisions|out of scope"
    r"|closed (?:scope|list))\b",
    re.I | re.M,
)
_COMPETING_RE = re.compile(
    r"\balternatively\b|\boption [a-c1-3]\b|\beither\b[^.\n]{0,80}\bor\b|\btrade-?offs?\b"
    r"|\bcompeting (?:designs?|approaches)\b|\bpolicy (?:choices?|decisions?)\b|\bversus\b",
    re.I,
)
_STALL_TEXT_RE = re.compile(
    r"\bstalled\b|\bstep[- ]back\b|\bstopped for a human decision\b|\bhuman[- ]decision stop\b",
    re.I,
)
_FILE_PATH_RE = re.compile(
    r"(?<![\w/.-])(?:[\w.-]+/)*[\w-][\w.-]*\.(?:py|sh|md|toml|ya?ml|json|js|ts|cfg|ini)\b"
)
_BEHAVIOR_RE = re.compile(
    r"\b(?:should|could also|must|needs? to|would)\s+(?:also\s+)?"
    r"(?:accept|reject|refuse|return|raise|print|emit|report|exit|skip|ignore|use|include|exclude)\b",
    re.I,
)
_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = frozenset(
    "the and for with that this from into when then than are was were not but can its"
    " should would could also have has had will issue run runs".split()
)
_SNIPPET_CHARS = 160


@dataclass(frozen=True)
class TriageSignal:
    name: str
    route_hint: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class SimilarIssue:
    issue_number: int
    score: float
    row: Mapping[str, object]


@dataclass(frozen=True)
class TriageBackend:
    agent: AgentName
    model: str | None
    reason: str


@dataclass(frozen=True)
class TriageOutcome:
    answer: TriageAnswer
    backend: TriageBackend
    signals: tuple[TriageSignal, ...]
    similar: tuple[SimilarIssue, ...]
    comment_posted: bool


# ---------------------------------------------------------------------------
# Deterministic signals
# ---------------------------------------------------------------------------


def _snippet(text: str, start: int, end: int) -> str:
    left = max(text.rfind("\n", 0, start), text.rfind(". ", 0, start))
    left = 0 if left < 0 else left + 1
    right = text.find("\n", end)
    right = len(text) if right < 0 else right
    snippet = " ".join(text[left:right].split())
    return snippet if len(snippet) <= _SNIPPET_CHARS else snippet[: _SNIPPET_CHARS - 3] + "..."


def _matches(pattern: re.Pattern[str], text: str, limit: int = 2) -> tuple[str, ...]:
    found: list[str] = []
    for match in pattern.finditer(text):
        snippet = _snippet(text, match.start(), match.end())
        if snippet not in found:
            found.append(snippet)
        if len(found) >= limit:
            break
    return tuple(found)


def _pair_value(row: Mapping[str, object], column: str) -> object:
    pair = row.get(column)
    if isinstance(pair, Mapping) and pair.get("evidence") == EVIDENCE_VERIFIED:
        return pair.get("value")
    return None


def _stall_count(row: Mapping[str, object]) -> int:
    total = 0
    for column in ("human_decision_stops", "step_backs"):
        value = _pair_value(row, column)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            total += value
    return total


def extract_signals(
    title: str | None, body: str | None, *, similar: Sequence[SimilarIssue] = ()
) -> tuple[TriageSignal, ...]:
    """Deterministic route signals from the issue text and retrieved outcomes."""
    text = f"{title or ''}\n{body or ''}"
    signals: list[TriageSignal] = []

    def add(name: str, evidence: Sequence[str]) -> None:
        if evidence:
            signals.append(TriageSignal(name, SIGNAL_HINTS[name], tuple(evidence)))

    closed = bool(_CLOSED_CRITERIA_RE.search(text))
    add(SIGNAL_OPEN_QUESTION, _matches(_OPEN_QUESTION_RE, text))
    edge_terms = sorted({match.group(0).lower() for match in _EDGE_CASE_RE.finditer(text)})
    if len(edge_terms) >= 2 and not closed:
        add(SIGNAL_EDGE_CASES, (
            "Edge-case terms: " + ", ".join(edge_terms[:6]) + "; no closed acceptance criteria.",
        ))
    add(SIGNAL_COMPETING_DESIGNS, _matches(_COMPETING_RE, text))
    stall_evidence = list(_matches(_STALL_TEXT_RE, text, limit=1))
    for item in similar:
        count = _stall_count(item.row)
        if count:
            stall_evidence.append(
                f"Similar issue #{item.issue_number} recorded {count} step-back(s) or human-decision stop(s)."
            )
    add(SIGNAL_PRIOR_STALL, stall_evidence)
    paths = sorted({match.group(0) for match in _FILE_PATH_RE.finditer(text)})
    if 1 <= len(paths) <= 3:
        add(SIGNAL_LOCAL_CHANGE, ("Names " + ", ".join(paths) + ".",))
    add(SIGNAL_SPECIFIED_BEHAVIOR, _matches(_BEHAVIOR_RE, text))
    if closed:
        add(SIGNAL_CLOSED_CRITERIA, _matches(_CLOSED_CRITERIA_RE, text, limit=1))
    return tuple(signals)


# ---------------------------------------------------------------------------
# Leak-free retrieval over a stage-1 history export
# ---------------------------------------------------------------------------


def _parse_time(value: object) -> datetime.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=datetime.timezone.utc)


def load_triage_history(path: Path) -> tuple[Mapping[str, object], ...]:
    """Rows of one local ``history export`` JSON file; unknown versions are refused."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AgentLoopError(f"Triage history {path} is unreadable: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != HISTORY_SCHEMA_VERSION
        or not isinstance(payload.get("rows"), list)
    ):
        raise AgentLoopError(
            f"Triage history {path} is not a schema_version {HISTORY_SCHEMA_VERSION} history export."
        )
    return tuple(row for row in payload["rows"] if isinstance(row, dict))


def retrieval_exclusion(
    row: Mapping[str, object], *, repo: str, issue_number: int, cutoff: datetime.datetime
) -> str | None:
    """Why ``row`` may not be retrieved at cutoff ``T``; ``None`` when eligible.

    A row is eligible only when its outcome was complete and available strictly
    before ``T`` and its run-start snapshot is usable and dated before ``T``.
    """
    if row.get("schema_version") != HISTORY_SCHEMA_VERSION:
        return "unknown row schema"
    if not isinstance(row.get("repo"), str) or str(row["repo"]).lower() != repo.lower():
        return "another repository"
    if row.get("issue_number") == issue_number:
        return "the issue itself"
    if row.get("outcome_complete") is not True:
        return "outcome not complete"
    available = _parse_time(row.get("outcome_available_at"))
    if available is None or available >= cutoff:
        return "outcome not available before the cutoff"
    snapshot = row.get("snapshot")
    if not isinstance(snapshot, Mapping) or snapshot.get("status") not in USABLE_SNAPSHOT_STATUSES:
        return "no usable run-start snapshot"
    as_of = _parse_time(snapshot.get("as_of"))
    if as_of is None or as_of >= cutoff:
        return "snapshot not dated before the cutoff"
    return None


def _tokens(text: str) -> frozenset[str]:
    return frozenset(
        token for token in _TOKEN_RE.findall(text.lower()) if len(token) >= 3 and token not in _STOPWORDS
    )


def retrieve_similar_issues(
    rows: Sequence[Mapping[str, object]],
    *,
    repo: str,
    issue_number: int,
    cutoff: datetime.datetime,
    query: str,
    limit: int = MAX_SIMILAR_ISSUES,
) -> tuple[SimilarIssue, ...]:
    """Up to ``limit`` eligible rows by token-set similarity, deterministically ordered."""
    query_tokens = _tokens(query)
    scored: list[SimilarIssue] = []
    for row in rows:
        if retrieval_exclusion(row, repo=repo, issue_number=issue_number, cutoff=cutoff) is not None:
            continue
        snapshot = row["snapshot"]
        assert isinstance(snapshot, Mapping)
        row_tokens = _tokens(f"{snapshot.get('title') or ''}\n{snapshot.get('body') or ''}")
        union = query_tokens | row_tokens
        score = len(query_tokens & row_tokens) / len(union) if union else 0.0
        if score > 0:
            scored.append(SimilarIssue(int(row["issue_number"]), round(score, 4), row))
    scored.sort(key=lambda item: (-item.score, item.issue_number))
    return tuple(scored[:limit])


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def resolve_triage_backend(config: AgentLoopConfig) -> TriageBackend:
    """Primary plan reviewer, else the first codex reviewer, else the first reviewer."""
    configured = reviewers(config)
    if config.primary_plan_reviewer is not None:
        agent, reason = config.primary_plan_reviewer, "the primary plan reviewer"
    else:
        codex = next((reviewer for reviewer in configured if seat_backend(reviewer) == "codex"), None)
        if codex is not None:
            agent, reason = codex, "the first configured codex reviewer"
        elif configured:
            agent, reason = configured[0], "the first configured reviewer"
        else:
            raise AgentLoopError("Triage needs a configured reviewer to select its backend.")
    if config.triage_model:
        return TriageBackend(agent, config.triage_model, f"{reason}, with the model from --triage-model")
    return TriageBackend(agent, configured_model_for(config, agent, role="reviewer"), reason)


def triage_invocation(config: AgentLoopConfig, backend: TriageBackend) -> tuple[AgentName, AgentLoopConfig]:
    """The agent and config the triage turn runs with; only the model may differ."""
    agent = backend.agent
    if not config.triage_model:
        return agent, config
    model = config.triage_model
    if isinstance(agent, SeatAgent):
        seat = ReviewerSeat(
            seat_id=str(agent), backend=agent.backend, model_chain=(model,),
            effort=agent.effort, workdir=agent.workdir,
        )
        return SeatAgent(seat, agent.workdir), config
    overrides: dict[str, object] = {
        "codex": {"reviewer_codex_model": model},
        "claude": {"reviewer_claude_model": model},
        "gemini": {"gemini_model": model},
        "antigravity": {"antigravity_models": (model,), "antigravity_model": None},
    }.get(str(agent), {})
    return agent, dataclasses_replace(config, **overrides)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _plain(text: str) -> str:
    return sanitize_untrusted_prose(" ".join(text.split()))


def render_signals_block(signals: Sequence[TriageSignal]) -> str:
    if not signals:
        return "- None of the deterministic signals fired."
    return "\n".join(
        f"- `{signal.name}` (points to {signal.route_hint}): "
        + "; ".join(f'"{_plain(item)}"' for item in signal.evidence)
        for signal in signals
    )


def _display(row: Mapping[str, object], column: str) -> str:
    pair = row.get(column)
    if isinstance(pair, Mapping) and pair.get("evidence") == EVIDENCE_VERIFIED:
        return str(pair.get("value"))
    return "unavailable"


def render_similar_block(similar: Sequence[SimilarIssue], *, history_note: str) -> str:
    if not similar:
        return f"- None. {history_note}"
    lines = [f"- {history_note}"]
    for item in similar:
        snapshot = item.row.get("snapshot") or {}
        title = _plain(str(snapshot.get("title") or "(untitled)"))[:200] if isinstance(snapshot, Mapping) else ""
        lines.append(
            f"- #{item.issue_number} (similarity {item.score:.2f}): {title}. Route taken: "
            f"{item.row.get('route', 'unavailable')}; plan rounds {_display(item.row, 'plan_rounds')}; "
            f"step-backs {_display(item.row, 'step_backs')}; human-decision stops "
            f"{_display(item.row, 'human_decision_stops')}; PR rounds {_display(item.row, 'pr_rounds')}; "
            f"merged {_display(item.row, 'merged')}."
        )
    return "\n".join(lines)


_PROMPT_SNAPSHOT_BODY_CHARS = 4_000


def render_similar_prompt_block(similar: Sequence[SimilarIssue], *, history_note: str) -> str:
    """The model's view of each retrieved issue: its run-start snapshot and full outcome row.

    The comment uses the shorter :func:`render_similar_block`; the prompt keeps
    every outcome column, including plan size and infrastructure evidence, so
    the model can tell difficult work from infrastructure stops.
    """
    if not similar:
        return f"- None. {history_note}"
    parts = [f"- {history_note}"]
    for item in similar:
        snapshot = item.row.get("snapshot")
        snapshot = snapshot if isinstance(snapshot, Mapping) else {}
        body = str(snapshot.get("body") or "")
        if len(body) > _PROMPT_SNAPSHOT_BODY_CHARS:
            body = body[:_PROMPT_SNAPSHOT_BODY_CHARS] + "\n[snapshot body truncated]"
        outcome = {key: value for key, value in item.row.items() if key != "snapshot"}
        outcome["snapshot_status"] = snapshot.get("status")
        outcome["snapshot_as_of"] = snapshot.get("as_of")
        parts.append(
            "\n".join([
                f"### Similar issue #{item.issue_number} (similarity {item.score:.2f})",
                f"Run-start title: {sanitize_untrusted_prose(str(snapshot.get('title') or '(untitled)'))}",
                "Run-start body:",
                sanitize_untrusted_prose(body) or "(empty)",
                "Outcome row (history export):",
                sanitize_untrusted_prose(json.dumps(outcome, indent=2, sort_keys=True, ensure_ascii=False)),
            ])
        )
    return "\n\n".join(parts)


def _encode_record(payload: Mapping[str, object]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


_RECORD_RE = re.compile(rf"<!--\s*{TRIAGE_RECORD_MARKER}:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->")


def parse_triage_record(body: str) -> dict | None:
    """Decode the typed triage record of one comment body, or ``None``.

    Callers must pass only bodies authored by the authenticated orchestrator
    actor; a foreign copy of the record carries no authority.
    """
    match = _RECORD_RE.search(body or "")
    if match is None:
        return None
    try:
        payload = json.loads(base64.urlsafe_b64decode(match.group("payload").encode("ascii")))
    except (ValueError, binascii.Error):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("kind") != TRIAGE_RECORD_KIND
        or payload.get("schema_version") != TRIAGE_RECORD_SCHEMA_VERSION
    ):
        return None
    return payload


def render_triage_comment(
    *,
    config: AgentLoopConfig,
    issue_number: int,
    outcome_answer: TriageAnswer,
    backend: TriageBackend,
    signals: Sequence[TriageSignal],
    similar: Sequence[SimilarIssue],
    history_note: str,
    operator_route: str,
    triaged_at: datetime.datetime,
) -> TrustedBody:
    answer = outcome_answer
    headline = f"Recommended route: **{answer.route}**"
    if answer.unsure:
        headline += f" (unsure: {answer.unsure_reason}; unsure answers are reported as plan-first)"
    model = backend.model or "provider default"
    lines = [
        TRIAGE_COMMENT_HEADING,
        "",
        headline,
        "",
        f"This recommendation is advisory. This run continues with the route the operator "
        f"selected: `{operator_route}`.",
        "",
        f"Triage backend: {agent_display_name(backend.agent)}, model `{model}` ({backend.reason}).",
        "",
        "Reasons:",
        *(f"- `{reason.signal}`: {_plain(reason.explanation)}" for reason in answer.reasons),
        "",
        "Deterministic signals:",
        render_signals_block(signals),
        "",
        "Similar past issues:",
        render_similar_block(similar, history_note=history_note),
        "",
    ]
    payload = {
        "schema_version": TRIAGE_RECORD_SCHEMA_VERSION,
        "kind": TRIAGE_RECORD_KIND,
        "repo": config.repo,
        "issue_number": issue_number,
        "route": answer.route,
        "model_route": answer.model_route[:40],
        "unsure": answer.unsure,
        "operator_route": operator_route,
        "backend": seat_backend(backend.agent),
        "agent": str(backend.agent),
        "model": backend.model,
        "selection_reason": backend.reason,
        "signals": [signal.name for signal in signals],
        "reason_signals": [reason.signal for reason in answer.reasons],
        "similar_issues": [item.issue_number for item in similar],
        "triaged_at": triaged_at.isoformat(),
    }
    lines += [f"<!-- {TRIAGE_RECORD_MARKER}: {_encode_record(payload)} -->", "", "-- Orchestrator"]
    return TrustedBody.canonical(
        "\n".join(lines), surface=ISSUE_COMMENT_SURFACE, expected_tokens=(TRIAGE_RECORD_MARKER,)
    )


# ---------------------------------------------------------------------------
# The triage turn and its one-shot hook
# ---------------------------------------------------------------------------


def run_issue_triage(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    usage_context: RunUsageContext,
    issue_context: IssueContext,
    operator_route: str,
    now: Callable[[], datetime.datetime] | None = None,
) -> TriageOutcome | None:
    """Run one advisory triage turn; log and swallow every failure."""
    issue_number = issue_context.number
    try:
        cutoff = (now or (lambda: datetime.datetime.now(datetime.timezone.utc)))()
        backend = resolve_triage_backend(config)
        log(
            config,
            f"Issue #{issue_number}: triage backend {agent_display_name(backend.agent)} "
            f"model={backend.model or 'provider default'} ({backend.reason})",
        )
        rows: tuple[Mapping[str, object], ...] = ()
        if config.triage_history is None:
            history_note = "No outcome history file was given (--triage-history), so zero similar past issues were retrieved."
        else:
            try:
                rows = load_triage_history(config.triage_history)
                history_note = (
                    "Retrieved from the history export, using only outcomes complete before "
                    f"{cutoff.isoformat()}."
                )
            except AgentLoopError as exc:
                log(config, f"Issue #{issue_number}: {exc}; triage continues with zero similar issues.")
                history_note = "The outcome history file was unreadable, so zero similar past issues were retrieved."
        similar = retrieve_similar_issues(
            rows,
            repo=config.repo,
            issue_number=issue_number,
            cutoff=cutoff,
            query=f"{issue_context.title or ''}\n{issue_context.body or ''}",
        )
        signals = extract_signals(issue_context.title, issue_context.body, similar=similar)
        agent, triage_config = triage_invocation(config, backend)
        prompt = build_triage_prompt(
            issue_context,
            triage_config,
            agent=agent,
            signals_block=render_signals_block(signals),
            similar_block=render_similar_prompt_block(similar, history_note=history_note),
            signal_names=TRIAGE_REASON_SIGNALS,
        )
        response = _run_validated_agent(
            runner,
            agent=agent,
            config=triage_config,
            prompt=prompt,
            marker_description="structured issue_triage JSON object",
            validate=lambda text: parse_triage_response(text, allowed_signals=TRIAGE_REASON_SIGNALS),
            usage_context=usage_context,
            role="reviewer",
            label="triage",
            operation_description="advisory issue triage",
        )
        answer = response.marker_value
        if not isinstance(answer, TriageAnswer):
            raise AgentLoopError("Triage turn returned no triage answer.")
    except Exception as exc:  # noqa: BLE001 - triage is advisory; the run continues.
        log(
            config,
            f"Issue #{issue_number}: advisory triage failed ({type(exc).__name__}: {exc}); "
            f"no recommendation was posted. Continuing with the operator-selected route "
            f"`{operator_route}`.",
        )
        return None
    log(
        config,
        f"Issue #{issue_number}: advisory triage recommends `{answer.route}`"
        + (f" (unsure: {answer.unsure_reason})" if answer.unsure else "")
        + f"; signals: {', '.join(signal.name for signal in signals) or 'none'}; "
        f"similar issues: {', '.join(f'#{item.issue_number}' for item in similar) or 'none'}. "
        f"The run continues with the operator-selected route `{operator_route}`.",
    )
    posted = False
    try:
        post_trusted_issue_comment(
            runner,
            config=config,
            issue_number=issue_number,
            body=render_triage_comment(
                config=config,
                issue_number=issue_number,
                outcome_answer=answer,
                backend=backend,
                signals=signals,
                similar=similar,
                history_note=history_note,
                operator_route=operator_route,
                triaged_at=cutoff,
            ),
        )
        posted = True
    except Exception as exc:  # noqa: BLE001 - a failed post never stops the run.
        log(
            config,
            f"Issue #{issue_number}: posting the triage comment failed ({type(exc).__name__}: {exc}); "
            f"continuing with the operator-selected route `{operator_route}`.",
        )
    return TriageOutcome(answer, backend, signals, similar, posted)


def arm_issue_triage(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    usage_context: RunUsageContext,
    issue_context: IssueContext,
    operator_route: str,
) -> bool:
    """Arm the one-shot pre-first-dispatch triage hook; arming has no side effect."""
    armed = usage_context.arm_pre_first_dispatch_hook(
        lambda: run_issue_triage(
            runner,
            config=config,
            usage_context=usage_context,
            issue_context=issue_context,
            operator_route=operator_route,
        )
    )
    if armed:
        log(config, f"Issue #{issue_context.number}: advisory triage will run before the first agent turn.")
    return armed


__all__ = [
    "SimilarIssue",
    "TriageBackend",
    "TriageOutcome",
    "TriageSignal",
    "arm_issue_triage",
    "extract_signals",
    "load_triage_history",
    "parse_triage_record",
    "render_similar_block",
    "render_similar_prompt_block",
    "render_triage_comment",
    "resolve_triage_backend",
    "retrieval_exclusion",
    "retrieve_similar_issues",
    "run_issue_triage",
    "triage_invocation",
]
