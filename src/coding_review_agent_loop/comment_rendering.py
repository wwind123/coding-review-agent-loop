"""Public comment and canonical plan rendering helpers."""

from __future__ import annotations

import html
import json
import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .decomposition import _decode_json_payload, _encode_json_payload
from .errors import AgentLoopError
from .expected_closure import normalize_issue_ids
from .agents.registry import agent_display_name, agent_signature
from .protocol import (
    DEGRADED_ROW_CLAIM_DIAGNOSTIC,
    PARSE_DEGRADATION_RENDER_LIMIT,
    UNAPPROVED_ROW_CLAIM_DIAGNOSTIC,
    ANY_HEADING_RE,
    HTML_COMMENT_RE,
    HUMAN_REQUIREMENTS_ADDRESSED_MARKER,
    HUMAN_REQUIREMENTS_ADDRESSED_RE,
    HUMAN_REQUIREMENTS_DIRECT_DISCUSSION_ACK_RE,
    HUMAN_REQUIREMENTS_HEADING_RE,
    PLAN_STATE_RE,
    PRIOR_UNRESOLVED_ITEM_DISPOSITIONS_HEADING_RE,
    PRIOR_UNRESOLVED_PLAN_ITEM_DISPOSITIONS_HEADING_RE,
    SIGNATURE_RE,
    AgentUnavailable,
    DeferredStage,
    TypedPlanStages,
    HumanRequirementDisposition,
    MACHINE_AUTHORITY,
    ParsedDiscussAgenda,
    ParsedDiscussAnswer,
    ParsedDiscussFinalSynthesis,
    ParsedFailedDiscussResponse,
    ParsedDiscussRoundSynthesis,
    ParsedDiscussResponse,
    ParsedDiscussReview,
    ParsedPlanReview,
    ParsedReview,
    ReviewItemDisposition,
    StructuredCoderFollowup,
    StructuredIssueImplementation,
    StructuredPlanState,
    StructuredPlanRevision,
    RiskTestMatrix,
    RiskTestMatrixChange,
    RiskTestMatrixEvidence,
    parse_risk_test_matrix,
    parse_risk_test_matrix_changes,
    parse_risk_test_matrix_evidence,
    risk_test_matrix_identity,
    ExecutionStrategyRecommendation,
    EXECUTION_TOPOLOGY_SOURCE,
    UnresolvedReviewItem,
    parse_human_requirements_acknowledgement,
    review_freeform_summary_text,
)
from .unresolved_items import HUMAN_REQUIREMENTS_ACK_ITEM_ID, MERGE_CONFLICT_ITEM_ID
from .protocol_markers import sanitize_historical_text
from .round_transport import (
    MAX_GITHUB_BODY_CHARS,
    execution_recommendation_section_boundary,
    risk_test_matrix_section_boundary,
)
from .test_runtime import (
    DEFAULT_TEST_TIMEOUT_SECONDS,
    TestRuntimeConfigurationError,
    managed_wrapper_traversal,
    parse_managed_test_invocation,
    resolve_timeout_seconds,
)
from .local_test_evidence import decode_bounded_evidence, redact_test_command

if TYPE_CHECKING:
    from .agents.base import AgentName
    from .config import AgentLoopConfig
    from .round_state import PostedRoundMetadata

ITEM_SUMMARY_LIMIT = 100
PLAN_EXPECTED_CLOSING_MARKER = "AGENT_PLAN_EXPECTED_CLOSING_ISSUES"
PLAN_EXPECTED_CLOSING_MARKER_RE = re.compile(
    rf"<!--\s*{PLAN_EXPECTED_CLOSING_MARKER}:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.IGNORECASE,
)
EXECUTION_RECOMMENDATION_MARKER = "AGENT_EXECUTION_RECOMMENDATION"
EXECUTION_RECOMMENDATION_MARKER_RE = re.compile(
    rf"<!--\s*{EXECUTION_RECOMMENDATION_MARKER}:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.IGNORECASE,
)
# Reverse map display-name -> agent. agent_display_name is config-independent, so
# this is safe to build at import; the signature itself is resolved per-call via
# agent_signature(agent, config) so it can reflect the configured model (#332).
_AGENT_BY_DISPLAY_NAME = {
    agent_display_name(agent): agent
    for agent in ("claude", "codex", "gemini", "antigravity")
}


def _render_test_command_for_comment(
    command: str,
    *,
    config: AgentLoopConfig | None = None,
) -> str:
    """Hide managed wrapper plumbing while preserving ordinary reports exactly."""
    try:
        tokens = shlex.split(command)
        # Traverse leading assignments and supported execution prefixes with
        # the same contract the checkout guard and citation projection use, so
        # a prefix-wrapped clause such as `timeout 1800 agent-loop run-tests
        # --memory-dir ... -- pytest` does not fall through to verbatim
        # rendering and publish the operator's memory directory.
        prefix_len = managed_wrapper_traversal(tokens).effective_head_index
        if prefix_len is None:
            return command
        parsed = parse_managed_test_invocation(
            tokens[prefix_len:], allow_command_name_launcher=True
        )
    except (ValueError, TestRuntimeConfigurationError):
        return command
    if parsed is None:
        return command
    chosen = parsed.timeout_seconds
    policy = (
        config.coder_test_command_timeout_seconds
        if config is not None
        else DEFAULT_TEST_TIMEOUT_SECONDS
    )
    try:
        chosen = resolve_timeout_seconds(chosen, policy_ceiling=policy)
    except TestRuntimeConfigurationError:
        return command
    if chosen is None:
        chosen = policy
    if float(chosen).is_integer():
        timeout = str(int(chosen))
    else:
        timeout = f"{chosen:g}"
    return (
        f"{shlex.join([*tokens[:prefix_len], *parsed.inner_argv])} "
        f"(agent-loop instrumented; whole-command timeout {timeout}s)"
    )


def _render_test_commands_for_comment(
    commands: Sequence[str], *, config: AgentLoopConfig | None = None
) -> list[str]:
    return [_render_test_command_for_comment(command, config=config) for command in commands]


def _render_test_observation_citations(
    citations: Sequence[object],
    *,
    local_test_evidence: str | None = None,
    current_test_turn_id: str | None = None,
) -> str:
    """Correlate receipt claims with the sanitized parent journal."""
    evidence = decode_bounded_evidence(local_test_evidence) if local_test_evidence else None
    by_receipt = {
        item.receipt_id: item
        for item in (evidence.observations if evidence is not None else ())
        if item.receipt_id
    }
    lines = ["### Test observation receipts"]
    cited: set[str] = set()
    citation_uses: dict[str, set[tuple[str, str]]] = {}
    for citation in citations:
        receipt_id = sanitize_historical_text(str(getattr(citation, "receipt_id", "")))
        command = sanitize_historical_text(str(getattr(citation, "command", "")))
        claim = sanitize_historical_text(str(getattr(citation, "claim", "")))
        safe_command, _identifiers, _caveats = redact_test_command(command)
        citation_uses.setdefault(receipt_id, set()).add((safe_command, claim))
    for citation in citations:
        command = sanitize_historical_text(str(getattr(citation, "command", "")))
        receipt_id = sanitize_historical_text(str(getattr(citation, "receipt_id", "")))
        claim = sanitize_historical_text(str(getattr(citation, "claim", "")))
        safe_command, _identifiers, _caveats = redact_test_command(command)
        observed = by_receipt.get(receipt_id)
        supported = observed is not None
        reason = "verified against the parent journal"
        if len(citation_uses.get(receipt_id, ())) > 1:
            supported = False
            reason = "unverified: conflicting uses of one receipt"
        elif observed is None:
            reason = "unverified: unknown or cross-turn receipt"
        elif current_test_turn_id is None or observed.turn_id != current_test_turn_id:
            supported = False
            reason = "unverified: unknown or cross-turn receipt"
        elif redact_test_command(observed.command)[0] != safe_command:
            supported = False
            reason = "unverified: command disagrees with the parent journal"
        elif claim == "base-reproduction" and observed.attribution.state != "base-reproduction":
            supported = False
            reason = "unverified: receipt does not support a base reproduction"
        if supported:
            cited.add(receipt_id)
        lines.append(
            f"- `{safe_command}` — receipt `{receipt_id[:256]}` — `{claim}` ({reason}; CI remains authoritative)"
        )
    if evidence is not None:
        for item in evidence.observations:
            if (
                item.receipt_id
                and item.receipt_id not in cited
                and item.is_failure
                and item.provenance == "parent-observed"
                and not item.superseded_by
            ):
                safe_command, _identifiers, _caveats = redact_test_command(item.command)
                lines.append(
                    f"- `{safe_command}` — receipt `{item.receipt_id[:256]}` — uncited authoritative `{item.outcome}`"
                )
    return "\n".join(lines)


def render_agent_unavailable_comment(unavailable: AgentUnavailable, *, signature: str) -> str:
    """Render a protocol-valid ``agent_unavailable`` envelope that parse_agent_unavailable accepts.

    Used only for orchestrator-synthesized outcomes (e.g. an exhausted
    completion-recovery attempt, #588) where there is no agent-authored
    envelope to post verbatim. The JSON payload, footer, and signature must
    exactly match the grammar in protocol.py's
    _consume_agent_unavailable_footer_and_signature: no prose between the JSON
    and the footer, and nothing but the signature after it.
    """
    payload = {
        "schema_version": unavailable.schema_version,
        "kind": "agent_unavailable",
        "retryable": unavailable.retryable,
        "category": unavailable.category,
        "summary": unavailable.summary,
        "suggested_action": unavailable.suggested_action,
    }
    return f"{json.dumps(payload, ensure_ascii=False)}\n<!-- AGENT_UNAVAILABLE -->\n-- {signature}"


def _review_freeform_summary_text(text: str) -> str:
    return review_freeform_summary_text(text)


def _normalize_item_summary(text: str, *, limit: int = ITEM_SUMMARY_LIMIT) -> str:
    # Prior-item text is historical ledger data, not a fresh agent output.  It
    # must be neutralized before truncation so a marker cannot survive in a
    # later public rendering or be recreated at the truncation boundary.
    text = sanitize_historical_text(text)
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)+", "", stripped)
        stripped = " ".join(stripped.split())
        if len(stripped) <= limit:
            return stripped
        if limit <= 3:
            return stripped[:limit]
        return stripped[: limit - 3].rstrip() + "..."
    return "No summary provided."


def _item_label_status(item: UnresolvedReviewItem) -> str:
    return item.source_status or item.status


def _public_reviewer_name(
    name: str,
    config: AgentLoopConfig | None = None,
    model_used: str | None = None,
    *,
    role: str | None = None,
) -> str:
    agent = _AGENT_BY_DISPLAY_NAME.get(name)
    if agent is None and name in {"claude", "codex", "gemini", "antigravity"}:
        agent = name  # type: ignore[assignment]
    if agent is None:
        return name
    # `model_used` is the producing agent's actual model (footers). Historical
    # attributions pass the producing role so configured role overrides remain
    # accurate when no observed model is available.
    return agent_signature(agent, config, model_used, role=role)


def _comment_signature(
    agent: str,
    config: AgentLoopConfig | None = None,
    model_used: str | None = None,
) -> str:
    return _public_reviewer_name(agent, config, model_used)


def _format_unresolved_item_label(
    item: UnresolvedReviewItem, config: AgentLoopConfig | None = None
) -> str:
    summary = _normalize_item_summary(item.text)
    status = _item_label_status(item)
    if item.item_id == HUMAN_REQUIREMENTS_ACK_ITEM_ID:
        return f"Human-requirements acknowledgement item, round {item.source_round}: {summary}"
    if item.item_id == MERGE_CONFLICT_ITEM_ID:
        return f"Merge conflict item, round {item.source_round}: {summary}"
    if item.authority == MACHINE_AUTHORITY or item.obligation_kind is not None:
        return (
            f"Machine obligation ({item.obligation_kind or 'unknown'}), "
            f"lifecycle {item.lifecycle or 'unknown'}, round {item.source_round}: {summary}"
        )
    phrases = {
        "blocking": "Blocking issue",
        "same-pr": "Same-PR follow-up",
        "same-plan": "Same-plan follow-up",
        "future": "Future follow-up",
    }
    phrase = phrases.get(status, "Unresolved item")
    reviewer_name = _public_reviewer_name(item.reviewer, config, role="reviewer")
    return f"{phrase} from {reviewer_name}, round {item.source_round}: {summary}"


def _render_disposition_status(disposition: ReviewItemDisposition) -> str:
    labels = {
        "resolved": "RESOLVED",
        "blocking": "BLOCKING",
        "same-pr": "SAME-PR",
        "same-plan": "SAME-PLAN",
        "future": "FUTURE FOLLOW-UP",
    }
    rendered = labels.get(disposition.disposition, disposition.disposition)
    if disposition.note:
        rendered = f"{rendered}: {disposition.note}"
    return rendered


def _render_prior_dispositions_section(
    *,
    heading: str,
    prior_items: Sequence[UnresolvedReviewItem],
    dispositions: Sequence[ReviewItemDisposition],
    config: AgentLoopConfig | None = None,
) -> str:
    item_by_id = {item.item_id: item for item in prior_items}
    lines = [heading]
    for disposition in dispositions:
        item = item_by_id.get(disposition.item_id)
        if item is None:
            raise AgentLoopError(
                f"Renderer encountered unknown prior item ID {disposition.item_id!r}; "
                f"allowed IDs: {sorted(item_by_id)}"
            )
        # A colon after [item-id] makes bare statuses hidden Markdown link definitions.
        lines.append(
            f"- [{disposition.item_id}] {_render_disposition_status(disposition)}"
        )
        lines.append(f"  - Original finding: {_format_unresolved_item_label(item, config)}")
    return "\n".join(lines)


def _replace_structured_section(
    body: str,
    *,
    heading_re: re.Pattern[str],
    replacement: str,
) -> str:
    lines = body.splitlines()
    output: list[str] = []
    index = 0
    replaced = False
    while index < len(lines):
        line = lines[index]
        if not replaced and heading_re.match(line):
            output.extend(replacement.splitlines())
            replaced = True
            index += 1
            while index < len(lines):
                current = lines[index]
                if (
                    PRIOR_UNRESOLVED_ITEM_DISPOSITIONS_HEADING_RE.match(current)
                    or PRIOR_UNRESOLVED_PLAN_ITEM_DISPOSITIONS_HEADING_RE.match(current)
                    or (
                        current.strip()
                        and (
                            ANY_HEADING_RE.match(current)
                            or HTML_COMMENT_RE.match(current)
                            or SIGNATURE_RE.match(current)
                        )
                    )
                ):
                    break
                index += 1
            continue
        output.append(line)
        index += 1
    return "\n".join(output) if replaced else body


def _append_before_trailing_metadata(body: str, section: str) -> str:
    lines = body.splitlines()
    index = len(lines)
    while index > 0 and not lines[index - 1].strip():
        index -= 1
    metadata_start = index
    while metadata_start > 0:
        candidate = lines[metadata_start - 1]
        if not candidate.strip() or HTML_COMMENT_RE.match(candidate) or SIGNATURE_RE.match(candidate):
            metadata_start -= 1
            continue
        break
    if metadata_start == index:
        return body.rstrip() + "\n\n" + section
    prefix = "\n".join(lines[:metadata_start]).rstrip()
    suffix = "\n".join(lines[metadata_start:]).lstrip("\n")
    parts = [prefix, section, suffix]
    return "\n\n".join(part for part in parts if part)


def render_canonical_plan_steps(plan_steps: Sequence[str]) -> str:
    return "\n".join(f"{index}. {step}" for index, step in enumerate(plan_steps, start=1))


RISK_TEST_MATRIX_MARKER = "AGENT_RISK_TEST_MATRIX"
RISK_TEST_MATRIX_MARKER_RE = re.compile(
    rf"<!--\s*{RISK_TEST_MATRIX_MARKER}:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.IGNORECASE,
)
_RISK_TEST_MATRIX_BOUNDARY_RE = re.compile(
    r"(?m)^<!--\s*risk-test-matrix-section:\s*[0-9a-f]{64}\s*-->[ \t]*$",
    re.IGNORECASE,
)
_RISK_TEST_MATRIX_HEADING_RE = re.compile(
    r"(?m)^### Risk-based mode and transition test matrix[ \t]*$",
)
_RISK_TEST_MATRIX_SECTION_TAIL_RE = re.compile(
    r"(?m)^(?:#{2,3}[ \t]+|<!--\s*(?:execution-recommendation-section|AGENT_PLAN_STATE):)",
)


def extract_risk_test_matrix_section(text: str) -> str | None:
    """Extract one complete write-time matrix section from rendered text.

    The structured payload is the authority during recovery, so this helper is
    intentionally limited to the persistence boundary. It prevents a posted
    plan from carrying extra reviewer-visible matrix prose after an otherwise
    valid payload marker while still allowing the normal section separator and
    subsequent plan/protocol sections.
    """
    boundaries = list(_RISK_TEST_MATRIX_BOUNDARY_RE.finditer(text))
    headings = list(_RISK_TEST_MATRIX_HEADING_RE.finditer(text))
    markers = list(RISK_TEST_MATRIX_MARKER_RE.finditer(text))
    if len(boundaries) != 1 or len(headings) != 1 or len(markers) != 1:
        return None
    boundary = boundaries[0]
    marker = markers[0]
    if marker.start() < boundary.end() or headings[0].start() < boundary.end():
        return None

    tail_match = _RISK_TEST_MATRIX_SECTION_TAIL_RE.search(text, marker.end())
    tail_end = tail_match.start() if tail_match is not None else len(text)
    if text[marker.end():tail_end].strip():
        return None
    return text[boundary.start():marker.end()]


def render_risk_test_matrix_section(
    matrix: RiskTestMatrix,
    changes: Sequence[RiskTestMatrixChange] = (),
) -> str:
    """Render the matrix projection from validated semantics only."""
    parsed = parse_risk_test_matrix(matrix)
    identity = risk_test_matrix_identity(parsed, changes)
    payload = {
        "contract_version": 1,
        "matrix": parsed.to_payload(),
        "changes": [change.to_payload() for change in changes],
        "identity": identity,
    }
    encoded = _encode_json_payload(payload)
    lines = [
        risk_test_matrix_section_boundary(identity),
        "### Risk-based mode and transition test matrix",
    ]
    if parsed.applicability == "not-applicable":
        lines.append(f"- **Applicability:** not applicable — {sanitize_historical_text(parsed.not_applicable_rationale or '')}")
    else:
        lines.extend([
            "- **Applicability:** applicable",
            "",
            "| ID | Scenario | Entry path / mode | Initial state | Event | Expected outcome | Forbidden side effects | Proposed test | Owner |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for row in parsed.rows:
            def cell(value: str) -> str:
                return sanitize_historical_text(value).replace("|", "\\|").replace("\n", " ")
            forbidden = "; ".join(row.forbidden_side_effects) or "None"
            proposed = f"{row.proposed_test_level} / {row.proposed_test_location}"
            lines.append(
                "| " + " | ".join([
                    f"`{cell(row.row_id)}`", cell(row.label), cell(row.entry_path_or_mode),
                    cell(row.initial_state), cell(row.event), cell(row.expected_outcome),
                    cell(forbidden), cell(proposed), f"`{cell(row.execution_owner)}`",
                ]) + " |"
            )
    if parsed.important_exclusions:
        lines.extend(["", "**Important exclusions:**", *[f"- {sanitize_historical_text(item)}" for item in parsed.important_exclusions]])
    if changes:
        lines.extend(["", "#### Matrix draft change audit"])
        lines.extend(
            f"- `{sanitize_historical_text(change.operation)}` `{', '.join(sanitize_historical_text(item) for item in change.row_ids)}` — {sanitize_historical_text(change.rationale)}"
            for change in changes
        )
    lines.append(f"<!-- {RISK_TEST_MATRIX_MARKER}: {encoded} -->")
    return "\n".join(lines)


def decode_risk_test_matrix_marker(
    encoded: str, *, bodies: Sequence[str] = ()
) -> dict[str, object]:
    payload = _decode_json_payload(encoded, marker_name=RISK_TEST_MATRIX_MARKER)
    if _encode_json_payload(payload) != encoded:
        raise AgentLoopError(f"Invalid {RISK_TEST_MATRIX_MARKER} payload.")
    if "$round_transport_risk_test_matrix" in payload:
        from .round_transport import hydrate_mapping

        hydrated, missing = hydrate_mapping(
            {"risk_test_matrix_marker": payload}, bodies
        )
        recovered = hydrated.get("risk_test_matrix_marker")
        if missing or not isinstance(recovered, str):
            raise AgentLoopError(
                f"Invalid {RISK_TEST_MATRIX_MARKER} payload: sidecar unavailable."
            )
        try:
            payload = json.loads(recovered)
        except json.JSONDecodeError as exc:
            raise AgentLoopError(
                f"Invalid {RISK_TEST_MATRIX_MARKER} payload: sidecar is not JSON."
            ) from exc
    if not isinstance(payload, dict) or set(payload) != {"contract_version", "matrix", "changes", "identity"}:
        raise AgentLoopError(f"Invalid {RISK_TEST_MATRIX_MARKER} payload.")
    if payload["contract_version"] != 1:
        raise AgentLoopError(f"Unsupported {RISK_TEST_MATRIX_MARKER} contract version.")
    matrix = parse_risk_test_matrix(payload["matrix"])
    changes = parse_risk_test_matrix_changes(payload["changes"])
    identity = risk_test_matrix_identity(matrix, changes)
    if payload["identity"] != identity:
        raise AgentLoopError(f"Invalid {RISK_TEST_MATRIX_MARKER} identity.")
    return payload


_MATRIX_EVIDENCE_RENDER_MODES = frozenset({"full", "delta"})


def _positive_round(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


@dataclass(frozen=True)
class MatrixEvidenceRenderDecision:
    """The single full-or-delta choice for a visible matrix-evidence section.

    Presentation only (#959): canonical evidence in round metadata and the
    matrix sidecar always keep every row.  ``anchor_round`` is the coder
    record round that rendered the full row list and is what callers persist.
    """

    mode: str
    anchor_round: int
    previous_evidence: RiskTestMatrixEvidence | None = None
    previous_round: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in _MATRIX_EVIDENCE_RENDER_MODES:
            raise ValueError("matrix evidence render mode must be full or delta")
        if not _positive_round(self.anchor_round):
            raise ValueError("matrix evidence anchor round must be a positive integer")
        if self.mode == "full" and (
            self.previous_evidence is not None or self.previous_round is not None
        ):
            raise ValueError("a full matrix evidence decision carries no previous evidence")
        if self.mode == "delta" and (
            self.previous_evidence is None or not _positive_round(self.previous_round)
        ):
            raise ValueError("a delta matrix evidence decision requires previous evidence")


def _matrix_evidence_row_key(row: object) -> str:
    # The per-turn receipt_id is ignored: a row re-cited with a fresh receipt
    # but the same claims is not a presentation change.
    payload = row.to_payload()  # type: ignore[attr-defined]
    payload["evidence_citations"] = [
        {"command": item["command"], "claim": item["claim"]}
        for item in payload["evidence_citations"]
    ]
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def resolve_matrix_evidence_render(
    current: RiskTestMatrixEvidence,
    previous_metadata: "PostedRoundMetadata | None",
    current_round: int,
) -> MatrixEvidenceRenderDecision:
    """Choose full or delta presentation against the previous coder record.

    ``current_round`` is the round number of the coder metadata record being
    published.  Never raises on unusable prior presentation data: any doubt
    falls back to a full render anchored at ``current_round``.
    """
    full = MatrixEvidenceRenderDecision(mode="full", anchor_round=current_round)
    if previous_metadata is None:
        return full
    raw_previous = getattr(previous_metadata, "risk_test_matrix_evidence", None)
    if raw_previous is None:
        return full
    try:
        previous = parse_risk_test_matrix_evidence(raw_previous)
    except Exception:  # noqa: BLE001 - prior presentation data must never abort a round
        return full
    if previous.matrix_identity != current.matrix_identity:
        return full
    if {row.row_id for row in previous.rows} != {row.row_id for row in current.rows}:
        return full
    previous_round = getattr(previous_metadata, "round_number", None)
    if not _positive_round(previous_round):
        return full
    status = getattr(previous_metadata, "risk_test_matrix_evidence_full_round_status", "absent")
    if status == "valid":
        anchor = getattr(previous_metadata, "risk_test_matrix_evidence_full_round", None)
        if not _positive_round(anchor) or anchor > previous_round:
            return full
    elif status == "absent":
        # A pre-#959 record rendered the full row list in its own comment.
        anchor = previous_round
    else:
        return full
    return MatrixEvidenceRenderDecision(
        mode="delta",
        anchor_round=anchor,
        previous_evidence=previous,
        previous_round=previous_round,
    )


def _matrix_evidence_display_text(value: object) -> str:
    # Row text is agent-influenced and sits inside a renderer-owned
    # <details> wrapper, so it must not be able to emit HTML tags (a raw
    # </details> would close the wrapper early) or line breaks (which could
    # start a new Markdown block).  Escape after protocol-record neutralization.
    text = sanitize_historical_text(str(value))
    text = " ".join(text.splitlines())
    return html.escape(text, quote=False)


def _render_risk_test_matrix_evidence(
    evidence: RiskTestMatrixEvidence | None,
    *,
    render_decision: MatrixEvidenceRenderDecision | None = None,
    diagnostics: Sequence[object] = (),
) -> str | None:
    if evidence is None:
        return None
    safe = _matrix_evidence_display_text
    rows = sorted(evidence.rows, key=lambda row: (row.status == "verified", row.row_id))
    lines = [
        "### Risk-based mode and transition test matrix evidence",
        f"- Matrix identity: `{safe(evidence.matrix_identity)}`",
    ]
    # Outside the collapsed block, so an operator sees that coverage was
    # dropped rather than silently reduced (#920, #926).
    lines.extend(
        f"- Dropped claim: {safe(getattr(diagnostic, 'message', ''))}"
        for diagnostic in diagnostics
        if getattr(diagnostic, "code", None)
        in {UNAPPROVED_ROW_CLAIM_DIAGNOSTIC, DEGRADED_ROW_CLAIM_DIAGNOSTIC}
    )
    unchanged_line: str | None = None
    if render_decision is not None and render_decision.mode == "delta":
        assert render_decision.previous_evidence is not None
        previous_keys = {
            row.row_id: _matrix_evidence_row_key(row)
            for row in render_decision.previous_evidence.rows
        }
        shown = [
            row for row in rows
            if previous_keys.get(row.row_id) != _matrix_evidence_row_key(row)
        ]
        unchanged = len(rows) - len(shown)
        summary = (
            f"{len(shown)} changed {'row' if len(shown) == 1 else 'rows'}; "
            f"{unchanged} unchanged since round {int(render_decision.previous_round)}"
        )
        unchanged_line = (
            f"{unchanged} {'row' if unchanged == 1 else 'rows'} unchanged since round "
            f"{int(render_decision.previous_round)}; full matrix in round "
            f"{int(render_decision.anchor_round)}."
        )
    else:
        shown = rows
        summary = f"Full matrix evidence ({len(rows)} {'row' if len(rows) == 1 else 'rows'})"
    lines.extend(["", "<details>", f"<summary>{summary}</summary>", ""])
    for row in shown:
        lines.append(f"- **{safe(row.row_id)}** — `{safe(row.status)}`")
        lines.append(f"  - Tests: {', '.join(safe(item) for item in row.test_identifiers) or 'none'}")
        lines.append(f"  - Locations: {', '.join(safe(item) for item in row.test_locations) or 'none'}")
        lines.append(f"  - Workflow path: {safe(row.workflow_path_claim)}")
        lines.append(f"  - Expected outcome assertions: {'; '.join(safe(item) for item in row.outcome_assertions) or 'none'}")
        lines.append(f"  - Forbidden-effect assertions: {'; '.join(safe(item) for item in row.forbidden_effect_assertions) or 'none'}")
        if row.evidence_citations:
            lines.append("  - Evidence citations: " + "; ".join(
                f"{safe(citation.command)} [{safe(citation.receipt_id)}; {safe(citation.claim)}]"
                for citation in row.evidence_citations
            ))
        if row.caveats:
            lines.append("  - Caveats: " + "; ".join(safe(item) for item in row.caveats))
    if unchanged_line is not None:
        if shown:
            lines.append("")
        lines.append(unchanged_line)
    lines.extend(["", "</details>"])
    return "\n".join(lines)


def render_expected_closing_issue_declaration(
    issue_ids: Sequence[int] | None,
) -> str | None:
    """Render an authoritative plan declaration, retaining omitted vs empty."""
    if issue_ids is None:
        return None
    normalized = normalize_issue_ids(issue_ids, field_name="additional_closing_issue_ids")
    assert normalized is not None
    payload = {"issue_ids": list(normalized)}
    encoded = _encode_json_payload(payload)
    visible = ", ".join(f"#{item}" for item in normalized) or "none"
    return "\n".join(
        [
            "### Additional issues completed by this implementation PR",
            f"Only these additional same-repository issues are part of this single-PR closing contract: {visible}.",
            "Incidental mentions, `Refs` links, related issues, staged parents, unselected stages, deferred work, and plan actions are not declarations.",
            f"<!-- {PLAN_EXPECTED_CLOSING_MARKER}: {encoded} -->",
        ]
    )


def decode_expected_closing_issue_declaration(encoded: str) -> tuple[int, ...]:
    payload = _decode_json_payload(encoded, marker_name=PLAN_EXPECTED_CLOSING_MARKER)
    if _encode_json_payload(payload) != encoded:
        raise AgentLoopError(
            f"Invalid {PLAN_EXPECTED_CLOSING_MARKER} payload: non-canonical encoding."
        )
    if set(payload) != {"issue_ids"}:
        raise AgentLoopError(f"Invalid {PLAN_EXPECTED_CLOSING_MARKER} payload.")
    normalized = normalize_issue_ids(
        payload["issue_ids"], field_name=f"{PLAN_EXPECTED_CLOSING_MARKER}.issue_ids"
    )
    assert normalized is not None
    return normalized


def render_human_requirement_dispositions(
    dispositions: Sequence[HumanRequirementDisposition],
    *,
    heading: str = "### Human requirement dispositions",
) -> str | None:
    if not dispositions:
        return None
    return "\n".join(
        [heading]
        + [
            f"- **{item.requirement_id}** — `{item.disposition}`: {item.evidence}"
            for item in dispositions
        ]
    )


# ``PARSE_DEGRADATION_RENDER_LIMIT`` is defined in protocol.py, beside the
# citation drop bound that must never exceed it, and re-exported here (#927).
TEST_OBSERVATION_DEGRADATIONS_HEADING = "### Test observation parse degradations"


def render_test_observation_degradations_section(records: Sequence[object]) -> str | None:
    """Render dropped follow-up citation records in full (#927)."""
    return render_parse_degradations_section(
        records, heading=TEST_OBSERVATION_DEGRADATIONS_HEADING
    )


def _degradation_cell(text: object) -> str:
    return " ".join(
        sanitize_historical_text(str(text))
        .replace("`", "'")
        .replace("<", "\u2039")
        .replace(">", "\u203a")
        .split()
    )[:200]


def render_parse_degradations_section(
    records: Sequence[object], *, heading: str = "### Parse degradations"
) -> str | None:
    """Render bounded, sanitized parse degradation records (#924).

    An operator sees reduced coverage rather than silent loss.  Renders
    nothing when there are no records.
    """
    if not records:
        return None
    lines = [
        heading,
        "The orchestrator degraded these elements instead of rejecting the response:",
    ]
    for record in list(records)[:PARSE_DEGRADATION_RENDER_LIMIT]:
        lines.append(
            f"- `{_degradation_cell(getattr(record, 'element_path', ''))}` — rule "
            f"`{_degradation_cell(getattr(record, 'rule', ''))}`; observed "
            f"`{_degradation_cell(getattr(record, 'observed_preview', ''))}`; outcome "
            f"`{_degradation_cell(getattr(record, 'outcome', ''))}`"
        )
    omitted = len(records) - PARSE_DEGRADATION_RENDER_LIMIT
    if omitted > 0:
        lines.append(f"- {omitted} more record(s) omitted.")
    return "\n".join(lines)


def render_decomposition_degradation_comment(records: Sequence[object]) -> str | None:
    """Plain parent-issue comment for an accepted decomposition's records."""
    section = render_parse_degradations_section(
        records, heading="### Decomposition parse degradations"
    )
    if section is None:
        return None
    return section + "\n\n-- coding-review-agent-loop"


def render_refused_decomposition_comment(
    records: Sequence[object], *, diagnostic: str
) -> str:
    """Plain parent-issue comment for a decomposition refused by its contract.

    Posted once before the exhaustion error propagates; it carries no managed
    record, and nothing was checkpointed or created.
    """
    section = render_parse_degradations_section(
        records, heading="### Decomposition parse degradations"
    )
    lines = [section] if section is not None else [
        "### Decomposition parse degradations",
        "The required `architecture_impact` assessment was omitted.",
    ]
    lines.append(
        "The decomposition was refused: no child issue was created and no topology "
        "checkpoint was published."
    )
    lines.append(f"Diagnostic: {_degradation_cell(diagnostic)}")
    return "\n".join(lines) + "\n\n-- coding-review-agent-loop"


def render_deferred_stages_section(deferred_stages: Sequence[DeferredStage]) -> str | None:
    """Render declared deferred stages so they carry into the plan's markdown.

    Deferred stages must survive canonical-plan rendering (#476): they feed
    subject hashing, stored plan state, reviewer prompts, and resume, so
    scope narrowing stays a mechanical signal instead of prose reviewers
    might miss.

    The human-readable `- {title}: {summary}` bullets are for reviewers; they
    are NOT what resume parses back, because a title containing its own colon
    (e.g. "Stage 2: API follow-up") would corrupt a naive split-on-first-colon
    parse. An `AGENT_DEFERRED_STAGES` HTML-comment marker carries the
    structured title/summary pairs verbatim so they round-trip exactly
    regardless of their text content (#492 review).
    """
    if not deferred_stages:
        return None
    lines = ["### Deferred stages (not in this plan)"]
    for stage in deferred_stages:
        lines.append(f"- {stage.title}: {stage.summary}")
    lines.append(f"<!-- AGENT_DEFERRED_STAGES: {_encode_deferred_stages_marker(deferred_stages)} -->")
    return "\n".join(lines)


def render_typed_plan_stages_section(stages: TypedPlanStages) -> str | None:
    """Render #585 categories; only child stages are eligible for filing."""
    groups = (
        ("Child stages (eligible for child issues)", stages.child_stages),
        ("External dependencies (linked, never created)", stages.external_dependencies),
        ("Deferred work (recorded only)", stages.deferred_work),
        ("Plan actions (recorded only)", stages.plan_actions),
    )
    if not any(entries for _, entries in groups):
        return None
    lines = ["### Structured scope categories"]
    for title, entries in groups:
        if entries:
            lines.append(f"#### {title}")
            lines.extend(f"- {entry.title}: {entry.summary}" for entry in entries)
    payload = {
        "child_stages": [
            {"title": item.title, "summary": item.summary}
            for item in stages.child_stages
        ],
        "external_dependencies": [
            {"title": item.title, "summary": item.summary}
            for item in stages.external_dependencies
        ],
        "deferred_work": [
            {"title": item.title, "summary": item.summary}
            for item in stages.deferred_work
        ],
        "plan_actions": [
            {"title": item.title, "summary": item.summary}
            for item in stages.plan_actions
        ],
    }
    lines.append(f"<!-- AGENT_TYPED_PLAN_STAGES: {_encode_json_payload(payload)} -->")
    return "\n".join(lines)


def _sanitize_execution_payload(value: object) -> object:
    if isinstance(value, str):
        return sanitize_historical_text(value)
    if isinstance(value, dict):
        return {str(key): _sanitize_execution_payload(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_sanitize_execution_payload(child) for child in value]
    return value


def render_execution_recommendation_section(
    recommendation: ExecutionStrategyRecommendation,
) -> str:
    """Render the complete v1 recommendation and its lossless sidecar.

    The sidecar is a distinct marker so legacy typed-stage extraction cannot
    accidentally treat reviewed v1 topology as executable child issues.
    """
    payload = _sanitize_execution_payload(recommendation.to_payload())
    assert isinstance(payload, dict)
    encoded = _encode_json_payload(payload)
    lines = [
        execution_recommendation_section_boundary(encoded),
        "### Execution strategy recommendation (v1)",
        f"- `topology_source`: `{EXECUTION_TOPOLOGY_SOURCE}`",
        f"- `strategy`: `{sanitize_historical_text(recommendation.strategy)}`",
        f"- `staging_feasibility`: `{sanitize_historical_text(recommendation.staging_feasibility)}`",
        f"- `rationale`: {sanitize_historical_text(recommendation.rationale)}",
        "",
        "#### `scope_items`",
    ]
    for item in recommendation.scope_items:
        lines.extend(
            [
                f"- `{sanitize_historical_text(item.scope_item_id)}`",
                f"  - `requirement`: {sanitize_historical_text(item.requirement)}",
                "  - `acceptance_criteria`:",
            ]
        )
        if item.acceptance_criteria:
            lines.extend(
                f"    - {sanitize_historical_text(value)}"
                for value in item.acceptance_criteria
            )
        else:
            lines.append("    - None")

    lines.extend(["", "#### `coupling_constraints`"])
    if recommendation.coupling_constraints:
        for constraint in recommendation.coupling_constraints:
            lines.extend(
                [
                    f"- `{sanitize_historical_text(constraint.constraint_id)}`",
                    "  - `scope_item_ids`: "
                    + ", ".join(
                        f"`{sanitize_historical_text(value)}`"
                        for value in constraint.scope_item_ids
                    ),
                    f"  - `rationale`: {sanitize_historical_text(constraint.rationale)}",
                ]
            )
    else:
        lines.append("- None")

    if recommendation.one_shot_delivery is not None:
        delivery = recommendation.one_shot_delivery
        lines.extend(["", "#### `one_shot_delivery`"])
        lines.extend(
            [
                "- `deliverables`:",
                *(
                    [f"  - {sanitize_historical_text(value)}" for value in delivery.deliverables]
                    or ["  - None"]
                ),
                "- `acceptance_criteria`:",
                *(
                    [
                        f"  - {sanitize_historical_text(value)}"
                        for value in delivery.acceptance_criteria
                    ]
                    or ["  - None"]
                ),
                "- `covered_scope_item_ids`: "
                + ", ".join(
                    f"`{sanitize_historical_text(value)}`"
                    for value in delivery.covered_scope_item_ids
                ),
            ]
        )

    lines.extend(["", "#### `child_stages`"])
    if recommendation.child_stages:
        for stage in recommendation.child_stages:
            lines.extend(
                [
                    f"- `{sanitize_historical_text(stage.stage_id)}` (`position`: {stage.position})",
                    f"  - `title`: {sanitize_historical_text(stage.title)}",
                    f"  - `summary`: {sanitize_historical_text(stage.summary)}",
                    "  - `deliverables`:",
                    *(
                        [f"    - {sanitize_historical_text(value)}" for value in stage.deliverables]
                        or ["    - None"]
                    ),
                    "  - `non_goals`:",
                    *(
                        [f"    - {sanitize_historical_text(value)}" for value in stage.non_goals]
                        or ["    - None"]
                    ),
                    "  - `acceptance_criteria`:",
                    *(
                        [
                            f"    - {sanitize_historical_text(value)}"
                            for value in stage.acceptance_criteria
                        ]
                        or ["    - None"]
                    ),
                    "  - `depends_on_stage_ids`: "
                    + ", ".join(
                        f"`{sanitize_historical_text(value)}`"
                        for value in stage.depends_on_stage_ids
                    )
                    if stage.depends_on_stage_ids
                    else "  - `depends_on_stage_ids`: None",
                    f"  - `dependency_notes`: {sanitize_historical_text(stage.dependency_notes)}",
                    f"  - `automation`: `{sanitize_historical_text(stage.automation)}`",
                    f"  - `rollout_risk`: {sanitize_historical_text(stage.rollout_risk)}",
                    "  - `compatibility_constraints`:",
                    *(
                        [
                            f"    - {sanitize_historical_text(value)}"
                            for value in stage.compatibility_constraints
                        ]
                        or ["    - None"]
                    ),
                    "  - `covered_scope_item_ids`: "
                    + ", ".join(
                        f"`{sanitize_historical_text(value)}`"
                        for value in stage.covered_scope_item_ids
                    ),
                ]
            )
            # Rendered only when declared so canonical text of plans approved
            # before the disposition contract (#808) is unchanged.
            if stage.execution_disposition is not None:
                disposition = stage.execution_disposition
                lines.extend(
                    [
                        "  - `execution_disposition`:",
                        f"    - `disposition`: `{sanitize_historical_text(disposition.disposition)}`",
                        f"    - `rationale`: {sanitize_historical_text(disposition.rationale)}",
                        "    - `unresolved_design_decisions`:",
                        *(
                            [
                                f"      - {sanitize_historical_text(value)}"
                                for value in disposition.unresolved_design_decisions
                            ]
                            or ["      - None"]
                        ),
                    ]
                )
    else:
        lines.append("- None")

    def render_allocation(title: str, allocation: object) -> None:
        lines.extend(["", f"#### {title}"])
        lines.append(f"- `status`: `{sanitize_historical_text(allocation.status)}`")
        lines.append("- `deliverables`:")
        lines.extend(
            f"  - {sanitize_historical_text(value)}" for value in allocation.deliverables
        )
        if not allocation.deliverables:
            lines.append("  - None")
        lines.append("- `acceptance_criteria`:")
        lines.extend(
            f"  - {sanitize_historical_text(value)}"
            for value in allocation.acceptance_criteria
        )
        if not allocation.acceptance_criteria:
            lines.append("  - None")
        lines.append(
            "- `covered_scope_item_ids`: "
            + (
                ", ".join(
                    f"`{sanitize_historical_text(value)}`"
                    for value in allocation.covered_scope_item_ids
                )
                if allocation.covered_scope_item_ids
                else "None"
            )
        )

    render_allocation("`retained_parent_work`", recommendation.retained_parent_work)
    render_allocation("`final_integration_work`", recommendation.final_integration_work)
    lines.extend(["", "#### `caveats`"])
    lines.extend(
        f"- {sanitize_historical_text(value)}" for value in recommendation.caveats
    )
    if not recommendation.caveats:
        lines.append("- None")
    lines.append(
        "The same complete recommendation is retained in the bounded sidecar below for lossless resume."
    )
    lines.append(f"<!-- {EXECUTION_RECOMMENDATION_MARKER}: {encoded} -->")
    return "\n".join(lines)


def decode_execution_recommendation_marker(
    encoded: str, *, bodies: Sequence[str] = ()
) -> dict[str, object]:
    payload = _decode_json_payload(encoded, marker_name=EXECUTION_RECOMMENDATION_MARKER)
    if _encode_json_payload(payload) != encoded:
        raise AgentLoopError(
            f"Invalid {EXECUTION_RECOMMENDATION_MARKER} payload: non-canonical encoding."
        )
    if "$round_transport_execution_recommendation" in payload:
        from .round_transport import hydrate_mapping

        hydrated, missing = hydrate_mapping(
            {"execution_recommendation": payload}, bodies
        )
        if missing or not isinstance(hydrated.get("execution_recommendation"), str):
            raise AgentLoopError(
                f"Invalid {EXECUTION_RECOMMENDATION_MARKER} payload: sidecar unavailable."
            )
        try:
            recovered = json.loads(hydrated["execution_recommendation"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise AgentLoopError(
                f"Invalid {EXECUTION_RECOMMENDATION_MARKER} payload: sidecar is not JSON."
            ) from exc
        if not isinstance(recovered, dict):
            raise AgentLoopError(f"Invalid {EXECUTION_RECOMMENDATION_MARKER} payload.")
        payload = recovered
    return payload


def _encode_deferred_stages_marker(deferred_stages: Sequence[DeferredStage]) -> str:
    return _encode_json_payload(
        {"stages": [{"title": stage.title, "summary": stage.summary} for stage in deferred_stages]}
    )


DEFERRED_STAGES_MARKER_RE = re.compile(
    r"<!--\s*AGENT_DEFERRED_STAGES:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.I,
)


def decode_deferred_stages_marker(encoded: str) -> tuple[DeferredStage, ...]:
    payload = _decode_json_payload(encoded, marker_name="AGENT_DEFERRED_STAGES")
    stages_payload = payload.get("stages")
    if not isinstance(stages_payload, list):
        raise AgentLoopError("Invalid AGENT_DEFERRED_STAGES payload.")
    stages: list[DeferredStage] = []
    for entry in stages_payload:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("title"), str)
            or not isinstance(entry.get("summary"), str)
        ):
            raise AgentLoopError("Invalid AGENT_DEFERRED_STAGES payload.")
        stages.append(DeferredStage(title=entry["title"], summary=entry["summary"]))
    return tuple(stages)


def render_canonical_plan_revision(
    parsed_revision: StructuredPlanRevision,
    prior_items: Sequence[UnresolvedReviewItem],
    config: AgentLoopConfig | None = None,
) -> str:
    sections = [parsed_revision.summary.strip()]
    if prior_items or parsed_revision.prior_plan_item_dispositions:
        sections.append(
            _render_prior_dispositions_section(
                heading="### Prior plan item dispositions",
                prior_items=prior_items,
                dispositions=parsed_revision.prior_plan_item_dispositions,
                config=config,
            )
        )
    else:
        sections.append("### Prior plan review item dispositions\n- None.")
    sections.append(
        "\n".join(
            [
                "### Plan steps",
                render_canonical_plan_steps(parsed_revision.plan_steps),
            ]
        )
    )
    if parsed_revision.risk_test_matrix is not None:
        sections.append(render_risk_test_matrix_section(parsed_revision.risk_test_matrix, parsed_revision.risk_test_matrix_changes))
    expected_section = render_expected_closing_issue_declaration(
        parsed_revision.additional_closing_issue_ids
    )
    if expected_section:
        sections.append(expected_section)
    human_section = render_human_requirement_dispositions(parsed_revision.human_requirement_dispositions)
    if human_section:
        sections.append(human_section)
    deferred_section = render_deferred_stages_section(parsed_revision.deferred_stages)
    if deferred_section:
        sections.append(deferred_section)
    typed_section = render_typed_plan_stages_section(parsed_revision.typed_stages)
    if typed_section:
        sections.append(typed_section)
    if parsed_revision.execution_recommendation is not None:
        sections.append(render_execution_recommendation_section(parsed_revision.execution_recommendation))
    return "\n\n".join(sections)


def render_canonical_plan_state(
    parsed_plan: StructuredPlanState,
    config: AgentLoopConfig | None = None,
) -> str:
    """Render a first-round plan using the same canonical rules as revisions."""
    sections = [parsed_plan.summary.strip(), "### Plan steps", render_canonical_plan_steps(parsed_plan.plan_steps)]
    if parsed_plan.risk_test_matrix is not None:
        sections.append(render_risk_test_matrix_section(parsed_plan.risk_test_matrix, parsed_plan.risk_test_matrix_changes))
    expected_section = render_expected_closing_issue_declaration(parsed_plan.additional_closing_issue_ids)
    if expected_section:
        sections.append(expected_section)
    human_section = render_human_requirement_dispositions(parsed_plan.human_requirement_dispositions)
    if human_section:
        sections.append(human_section)
    deferred_section = render_deferred_stages_section(parsed_plan.deferred_stages)
    if deferred_section:
        sections.append(deferred_section)
    typed_section = render_typed_plan_stages_section(parsed_plan.typed_stages)
    if typed_section:
        sections.append(typed_section)
    if parsed_plan.execution_recommendation is not None:
        sections.append(render_execution_recommendation_section(parsed_plan.execution_recommendation))
    return "\n\n".join(sections)


def _render_public_review_comment(
    body: str,
    *,
    review_kind: str,
    prior_items: Sequence[UnresolvedReviewItem],
    dispositions: Sequence[ReviewItemDisposition],
    new_items: Sequence[UnresolvedReviewItem],
    config: AgentLoopConfig | None = None,
) -> str:
    rendered = body
    if prior_items:
        heading = "### Prior unresolved item dispositions"
        heading_re = PRIOR_UNRESOLVED_ITEM_DISPOSITIONS_HEADING_RE
        if review_kind == "plan":
            heading = "### Prior unresolved plan item dispositions"
            heading_re = PRIOR_UNRESOLVED_PLAN_ITEM_DISPOSITIONS_HEADING_RE
        rendered = _replace_structured_section(
            rendered,
            heading_re=heading_re,
            replacement=_render_prior_dispositions_section(
                heading=heading,
                prior_items=prior_items,
                dispositions=dispositions,
                config=config,
            ),
        )
    return rendered


def _render_public_pr_review_comment(
    parsed_review: ParsedReview,
    *,
    reviewer: str,
    human_requirements_resolved_flag: bool,
    prior_items: Sequence[UnresolvedReviewItem],
    dispositions: Sequence[ReviewItemDisposition],
    config: AgentLoopConfig | None = None,
    model_used: str | None = None,
) -> str:
    sections: list[str] = [f"**Review verdict:** {parsed_review.state.title()}"]
    if parsed_review.summary and parsed_review.summary.strip() != "Review complete.":
        sections.append(parsed_review.summary.strip())
    if parsed_review.blocking_items:
        sections.append(
            "\n".join(
                [
                    "### Blocking issues",
                    *[f"- {item.text}" for item in parsed_review.blocking_items],
                ]
            )
        )
    if parsed_review.followups.same_pr:
        sections.append(
            "\n".join(
                [
                    "### Same-PR follow-ups",
                    *[f"- {item.text}" for item in parsed_review.followups.same_pr],
                ]
            )
        )
    if parsed_review.followups.future:
        sections.append(
            "\n".join(
                [
                    "### Future follow-ups",
                    *[f"- {item.text}" for item in parsed_review.followups.future],
                ]
            )
        )
    if prior_items:
        sections.append(
            _render_prior_dispositions_section(
                heading="### Prior unresolved item dispositions",
                prior_items=prior_items,
                dispositions=dispositions,
                config=config,
            )
        )
    footer: list[str] = []
    if human_requirements_resolved_flag:
        footer.append("<!-- HUMAN_REQUIREMENTS_RESOLVED -->")
    footer.append(f"<!-- AGENT_STATE: {parsed_review.state} -->")
    footer.append(f"-- {_public_reviewer_name(reviewer, config, model_used)}")
    return "\n\n".join(section for section in sections if section) + (
        ("\n\n" if sections else "") + "\n".join(footer)
    )


def _render_public_plan_review_comment(
    parsed_review: ParsedPlanReview,
    *,
    reviewer: str,
    prior_items: Sequence[UnresolvedReviewItem],
    dispositions: Sequence[ReviewItemDisposition],
    human_requirements_resolved_flag: bool = False,
    config: AgentLoopConfig | None = None,
    model_used: str | None = None,
) -> str:
    sections: list[str] = [f"**Review verdict:** {parsed_review.state.title()}"]
    if parsed_review.summary and parsed_review.summary.strip() != "Plan review complete.":
        sections.append(parsed_review.summary.strip())
    if parsed_review.items.blocking:
        sections.append(
            "\n".join(
                [
                    "### Blocking plan issues",
                    *[f"- {item.text}" for item in parsed_review.items.blocking],
                ]
            )
        )
    if parsed_review.items.same_plan:
        sections.append(
            "\n".join(
                [
                    "### Same-plan follow-ups",
                    *[f"- {item.text}" for item in parsed_review.items.same_plan],
                ]
            )
        )
    if parsed_review.items.future:
        sections.append(
            "\n".join(
                [
                    "### Future follow-ups",
                    *[f"- {item.text}" for item in parsed_review.items.future],
                ]
            )
        )
    if prior_items:
        sections.append(
            _render_prior_dispositions_section(
                heading="### Prior unresolved plan item dispositions",
                prior_items=prior_items,
                dispositions=dispositions,
                config=config,
            )
        )
    human_section = render_human_requirement_dispositions(
        parsed_review.human_requirement_dispositions
    )
    if human_section:
        sections.append(human_section)
    footer: list[str] = []
    if human_requirements_resolved_flag:
        footer.append("<!-- HUMAN_REQUIREMENTS_RESOLVED -->")
    footer.extend(
        [
            f"<!-- AGENT_PLAN_STATE: {parsed_review.state} -->",
            f"-- {_public_reviewer_name(reviewer, config, model_used)}",
        ]
    )
    return "\n\n".join(section for section in sections if section) + (
        ("\n\n" if sections else "") + "\n".join(footer)
    )


def coder_followup_head_unchanged_notice(head_sha: str) -> str:
    """Orchestrator note for a coder follow-up that left the PR head unchanged.

    Limited to observable state: an equal PR head cannot rule out an
    unpushed local commit, only that no new commit is visible (#1034).
    """
    return (
        f"> Orchestrator note: the PR head was unchanged at `{head_sha}` after this "
        "follow-up; no new commit is visible on the PR branch. The coder statements "
        "below are claims about the existing head, not changes attributed to this turn."
    )


def add_coder_followup_head_unchanged_notice(body: str, head_sha: str) -> str:
    """Frame a freeform coder follow-up body with the head-unchanged note."""
    return coder_followup_head_unchanged_notice(head_sha) + "\n\n" + body


def _render_public_coder_followup_comment(
    parsed_followup: StructuredCoderFollowup,
    *,
    agent: str,
    prior_items: Sequence[UnresolvedReviewItem] = (),
    config: AgentLoopConfig | None = None,
    model_used: str | None = None,
    local_test_evidence: str | None = None,
    current_test_turn_id: str | None = None,
    matrix_evidence_render_decision: MatrixEvidenceRenderDecision | None = None,
    head_unchanged_sha: str | None = None,
) -> str:
    item_by_id = {item.item_id: item for item in prior_items}

    def render_item(item_id: str, *, note_label: str, note: str | None, placeholder: str | None) -> list[str]:
        item = item_by_id.get(item_id)
        if item is None:
            lines = [f"- {item_id}: Item context unavailable in current round metadata."]
        else:
            lines = [f"- {item_id}: {_format_unresolved_item_label(item, config)}"]
        if note:
            lines.append(f"  - {note_label}: {note}")
        elif placeholder:
            lines.append(f"  - {note_label}: {placeholder}")
        return lines

    addressed_items: list[str] = []
    for item_id in parsed_followup.addressed_items:
        addressed_items.extend(
            render_item(
                item_id,
                note_label="Coder claim" if head_unchanged_sha else "Resolution",
                note=parsed_followup.addressed_item_notes.get(item_id),
                placeholder=None,
            )
        )
    if not addressed_items:
        addressed_items = ["- None."]

    remaining_items: list[str] = []
    for item_id in parsed_followup.remaining_items:
        remaining_items.extend(
            render_item(
                item_id,
                note_label="Reason",
                note=parsed_followup.remaining_item_notes.get(item_id),
                placeholder="No reason provided by coder.",
            )
        )
    if not remaining_items:
        remaining_items = ["- None."]

    disputed_items: list[str] = []
    for item_id in parsed_followup.disputed_items:
        disputed_items.extend(
            render_item(
                item_id,
                note_label="Counter-evidence",
                note=parsed_followup.dispute_evidence.get(item_id),
                placeholder="No evidence provided by coder.",
            )
        )

    if head_unchanged_sha:
        # The PR head did not move, so the coder's statements are claims about
        # the existing head, not changes attributed to this turn (#1034).
        sections = [
            "## Coder follow-up",
            coder_followup_head_unchanged_notice(head_unchanged_sha),
            f"Coder summary (claim, unverified): {parsed_followup.summary.strip()}",
            "\n".join(
                [
                    f"### Claimed already present at `{head_unchanged_sha}` (PR head unchanged)",
                    *addressed_items,
                ]
            ),
            "\n".join(["### Remaining items", *remaining_items]),
        ]
    else:
        sections = [
            "## Coder follow-up",
            parsed_followup.summary.strip(),
            "\n".join(["### Addressed items", *addressed_items]),
            "\n".join(["### Remaining items", *remaining_items]),
        ]
    if disputed_items:
        sections.append("\n".join(["### Disputed items", *disputed_items]))
    if parsed_followup.tests_run:
        sections.append(
            "\n".join(
                ["### Tests run", *[
                    f"- {test}"
                    for test in _render_test_commands_for_comment(
                        parsed_followup.tests_run, config=config
                    )
                ]]
            )
        )
    if parsed_followup.test_observations or local_test_evidence:
        sections.append(_render_test_observation_citations(
            parsed_followup.test_observations,
            local_test_evidence=local_test_evidence,
            current_test_turn_id=current_test_turn_id,
        ))
    citation_degradations = render_test_observation_degradations_section(
        parsed_followup.test_observation_degradations
    )
    if citation_degradations:
        sections.append(citation_degradations)
    matrix_evidence = _render_risk_test_matrix_evidence(
        parsed_followup.risk_test_matrix_evidence,
        render_decision=matrix_evidence_render_decision,
        diagnostics=parsed_followup.risk_test_matrix_diagnostics,
    )
    if matrix_evidence:
        sections.append(matrix_evidence)
    if parsed_followup.human_requirement_dispositions:
        sections.append(
            "\n".join(
                [
                    "### Human requirements",
                    *[
                        f"- {disposition.requirement_id}: {disposition.disposition} — {disposition.evidence}"
                        for disposition in parsed_followup.human_requirement_dispositions
                    ],
                ]
            )
        )
    sections.append(f"<!-- AGENT_STATE: {parsed_followup.state} -->")
    sections.append(f"-- {_comment_signature(agent, config, model_used)}")
    return "\n\n".join(section for section in sections if section)


def _render_public_issue_implementation_comment(
    parsed: StructuredIssueImplementation,
    *,
    agent: str,
    config: AgentLoopConfig | None = None,
    model_used: str | None = None,
    local_test_evidence: str | None = None,
    current_test_turn_id: str | None = None,
) -> str:
    """Render an implementation result without exposing its JSON envelope."""
    has_blocked_requirement = any(
        item.disposition == "blocked"
        for item in parsed.human_requirement_dispositions
    )
    if has_blocked_requirement and parsed.pr_number is not None:
        result = (
            f"Rejected for handoff: reported PR #{parsed.pr_number} was not accepted "
            "because a signed human requirement is blocked."
        )
    elif parsed.pr_number is None:
        result = "No pull request was accepted for handoff."
    else:
        result = f"Pull request reported: #{parsed.pr_number}."
    sections = ["## Issue implementation", parsed.summary.strip(), f"### Result\n{result}"]
    if parsed.tests_run:
        sections.append(
            "\n".join(
                ["### Tests run", *[
                    f"- {test}"
                    for test in _render_test_commands_for_comment(
                        parsed.tests_run, config=config
                    )
                ]]
            )
        )
    if parsed.test_observations or local_test_evidence:
        sections.append(_render_test_observation_citations(
            parsed.test_observations,
            local_test_evidence=local_test_evidence,
            current_test_turn_id=current_test_turn_id,
        ))
    citation_degradations = render_test_observation_degradations_section(
        parsed.test_observation_degradations
    )
    if citation_degradations:
        sections.append(citation_degradations)
    matrix_evidence = _render_risk_test_matrix_evidence(
        parsed.risk_test_matrix_evidence,
        diagnostics=parsed.risk_test_matrix_diagnostics,
    )
    if matrix_evidence:
        sections.append(matrix_evidence)
    human_section = render_human_requirement_dispositions(
        parsed.human_requirement_dispositions
    )
    if human_section:
        sections.append(human_section)
    sections.extend(
        [
            f"<!-- AGENT_STATE: {parsed.state} -->",
            f"-- {_comment_signature(agent, config, model_used)}",
        ]
    )
    return "\n\n".join(section for section in sections if section)


def _extract_plan_human_requirements_block(text: str) -> str:
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return ""
    decoder = json.JSONDecoder()
    try:
        payload, end = decoder.raw_decode(stripped)
    except ValueError:
        return ""
    if not isinstance(payload, dict):
        return ""
    trailing = stripped[end:].lstrip()
    state_match = PLAN_STATE_RE.search(trailing)
    if state_match is None:
        return ""
    return trailing[: state_match.start()].strip()


def _extract_plan_revision_human_requirements_block(text: str) -> str:
    return _extract_plan_human_requirements_block(text)


# Bounded visible digest for structured plan coder comments (#948).  The
# complete plan always travels in authenticated round metadata; the digest only
# bounds the human-facing prose.  Each budgeted section owns a fixed share that
# already includes its heading and omitted-entry line, and unused share is never
# redistributed, so the budgeted output is bounded by construction for any
# entry count or string length.
COMPACT_PLAN_DIGEST_BUDGET_CHARS = 12_000
COMPACT_PLAN_DIGEST_NOTICE = (
    "> **Compact plan digest.** The complete plan exceeded the comment budget, so this "
    "comment shows a bounded summary. The complete canonical plan is preserved in this "
    "round's authenticated attachments and is what reviewers, resume and approval use."
)
_COMPACT_ELLIPSIS = " …"
_COMPACT_SUMMARY_SHARE = 2_000
_COMPACT_STEPS_SHARE = 4_800
_COMPACT_PRIOR_DISPOSITIONS_SHARE = 1_400
_COMPACT_CLOSING_SHARE = 500
_COMPACT_DEFERRED_SHARE = 900
_COMPACT_TYPED_CATEGORY_SHARE = 400
_COMPACT_ENTRY_CHARS = 300
_COMPACT_TITLE_CHARS = 120
_COMPACT_HUMAN_EVIDENCE_CHARS = 300
_COMPACT_ACK_PROSE_LINES = 20
_COMPACT_OMITTED_LINE = (
    "- ... {omitted} more of {total} omitted; complete list in the authenticated attachments"
)
_COMPACT_REQUIREMENT_ID_RE = re.compile(
    r"\b(?:Requirement\s+)?hr-[0-9a-f]{64}\b|\bRequirement\s+\d+\b", re.I
)


def _compact_clip(text: str, limit: int, *, keep_newlines: bool = False) -> str:
    """Sanitize, flatten and clip ``text`` to at most ``limit`` characters."""
    safe = sanitize_historical_text(text).strip()
    if not keep_newlines:
        safe = re.sub(r"\s+", " ", safe)
    if len(safe) <= limit:
        return safe
    clipped = safe[: max(0, limit - len(_COMPACT_ELLIPSIS))].rstrip() + _COMPACT_ELLIPSIS
    # Clipping sanitized text cannot normally create a reserved record; a second
    # pass keeps that a checked property rather than an assumption.
    return sanitize_historical_text(clipped)[:limit]


def _compact_budgeted_section(
    heading: str, entries: Sequence[str], *, share: int, entry_chars: int
) -> str:
    """Render entries in canonical order until the fixed section share is used."""
    total = len(entries)
    reserve = len(_COMPACT_OMITTED_LINE.format(omitted=total, total=total)) + 1
    lines = [heading]
    used = len(heading)
    if used + reserve > share:
        raise AgentLoopError("Compact plan digest section share is too small for its heading.")
    shown = 0
    for index, entry in enumerate(entries):
        line = _compact_clip(entry, entry_chars)
        cost = len(line) + 1
        needed = reserve if index + 1 < total else 0
        if used + cost + needed > share:
            break
        lines.append(line)
        used += cost
        shown += 1
    if shown < total:
        lines.append(_COMPACT_OMITTED_LINE.format(omitted=total - shown, total=total))
    rendered = "\n".join(lines)
    if len(rendered) > share:
        raise AgentLoopError("Compact plan digest section exceeded its share.")
    return rendered


def _compact_human_requirement_dispositions(
    dispositions: Sequence[HumanRequirementDisposition],
) -> str | None:
    """List every requirement ID; only explanatory evidence is clipped."""
    if not dispositions:
        return None
    return "\n".join(
        ["### Human requirement dispositions"]
        + [
            f"- **{item.requirement_id}** — `{item.disposition}`: "
            f"{_compact_clip(item.evidence, _COMPACT_HUMAN_EVIDENCE_CHARS)}"
            for item in dispositions
        ]
    )


def _compact_acknowledgement_line(line: str) -> str:
    safe = re.sub(r"[ \t]+$", "", sanitize_historical_text(line))
    if len(safe) <= _COMPACT_HUMAN_EVIDENCE_CHARS:
        return safe
    cut = _COMPACT_HUMAN_EVIDENCE_CHARS
    required = [
        *_COMPACT_REQUIREMENT_ID_RE.finditer(safe),
        *HUMAN_REQUIREMENTS_DIRECT_DISCUSSION_ACK_RE.finditer(safe),
    ]
    # Never cut inside an ID or the acknowledgement sentence: a partial
    # "Requirement 12" would otherwise read as a different requirement.
    for match in required:
        if match.start() < cut < match.end():
            cut = match.start()
    kept = safe[:cut].rstrip() + _COMPACT_ELLIPSIS
    tail = [match.group(0) for match in required if match.start() >= cut]
    if tail:
        kept += " " + "; ".join(tail)
    return kept


def _compact_human_requirements_block(block: str) -> str:
    """Keep the record, heading, every ID line and the direct-discussion sentence."""
    if not block:
        return ""
    lines: list[str] = []
    prose_lines = 0
    dropped = 0
    for line in block.splitlines():
        if HUMAN_REQUIREMENTS_ADDRESSED_RE.search(line) and HTML_COMMENT_RE.match(line):
            lines.append(HUMAN_REQUIREMENTS_ADDRESSED_MARKER)
            continue
        essential = (
            HUMAN_REQUIREMENTS_HEADING_RE.match(line)
            or _COMPACT_REQUIREMENT_ID_RE.search(line)
            or HUMAN_REQUIREMENTS_DIRECT_DISCUSSION_ACK_RE.search(line)
        )
        if not essential:
            if not line.strip():
                if lines and lines[-1]:
                    lines.append("")
                continue
            prose_lines += 1
            if prose_lines > _COMPACT_ACK_PROSE_LINES:
                dropped += 1
                continue
        lines.append(_compact_acknowledgement_line(line))
    if dropped:
        lines.append(f"_{dropped} further explanatory line(s) omitted; no requirement ID was omitted._")
    compact = "\n".join(lines).strip()
    original = parse_human_requirements_acknowledgement(block)
    rendered = parse_human_requirements_acknowledgement(compact)
    direct = HUMAN_REQUIREMENTS_DIRECT_DISCUSSION_ACK_RE.search
    if (
        original.addressed_ids != rendered.addressed_ids
        or original.marker_present != rendered.marker_present
        or original.section_present != rendered.section_present
        or bool(direct(original.section_text)) != bool(direct(rendered.section_text))
    ):
        # Fail toward size, never toward a changed acknowledgement: the
        # transport's attributed overflow then fails closed if this cannot fit.
        return block
    return compact


def _render_compact_plan_digest(
    parsed: StructuredPlanState | StructuredPlanRevision,
    *,
    title: str,
    raw_text: str,
    agent: str,
    config: AgentLoopConfig | None,
    model_used: str | None,
) -> str:
    budgeted: list[str] = [
        COMPACT_PLAN_DIGEST_NOTICE,
        _compact_clip(parsed.summary, _COMPACT_SUMMARY_SHARE, keep_newlines=True),
    ]
    prior_dispositions = getattr(parsed, "prior_plan_item_dispositions", ())
    if prior_dispositions:
        budgeted.append(
            _compact_budgeted_section(
                "### Prior plan item dispositions (digest)",
                [
                    f"- [{item.item_id}] {_render_disposition_status(replace(item, note=None))}"
                    for item in prior_dispositions
                ],
                share=_COMPACT_PRIOR_DISPOSITIONS_SHARE,
                entry_chars=_COMPACT_TITLE_CHARS,
            )
        )
    budgeted.append(
        _compact_budgeted_section(
            "### Plan steps (digest)",
            [f"{index}. {step}" for index, step in enumerate(parsed.plan_steps, start=1)],
            share=_COMPACT_STEPS_SHARE,
            entry_chars=_COMPACT_ENTRY_CHARS,
        )
    )
    if parsed.additional_closing_issue_ids is not None:
        budgeted.append(
            _compact_budgeted_section(
                "### Additional issues completed by this implementation PR (digest)",
                [f"- #{item}" for item in parsed.additional_closing_issue_ids] or ["- none"],
                share=_COMPACT_CLOSING_SHARE,
                entry_chars=_COMPACT_TITLE_CHARS,
            )
        )
    if parsed.deferred_stages:
        budgeted.append(
            _compact_budgeted_section(
                "### Deferred stages (not in this plan; digest)",
                [f"- {stage.title}" for stage in parsed.deferred_stages],
                share=_COMPACT_DEFERRED_SHARE,
                entry_chars=_COMPACT_TITLE_CHARS,
            )
        )
    for heading, entries in (
        ("#### Child stages (digest)", parsed.typed_stages.child_stages),
        ("#### External dependencies (digest)", parsed.typed_stages.external_dependencies),
        ("#### Deferred work (digest)", parsed.typed_stages.deferred_work),
        ("#### Plan actions (digest)", parsed.typed_stages.plan_actions),
    ):
        if entries:
            budgeted.append(
                _compact_budgeted_section(
                    heading,
                    [f"- {entry.title}" for entry in entries],
                    share=_COMPACT_TYPED_CATEGORY_SHARE,
                    entry_chars=_COMPACT_TITLE_CHARS,
                )
            )
    budgeted_text = "\n\n".join(section for section in budgeted if section)
    if len(budgeted_text) > COMPACT_PLAN_DIGEST_BUDGET_CHARS:
        raise AgentLoopError(
            "Compact plan digest exceeded its aggregate budget of "
            f"{COMPACT_PLAN_DIGEST_BUDGET_CHARS} characters."
        )
    sections = [title, budgeted_text]
    # Visible-anchor records: emitted by the unchanged section renderers so the
    # transport's authenticated reference rewrites keep applying to them.
    if parsed.risk_test_matrix is not None:
        sections.append(
            render_risk_test_matrix_section(parsed.risk_test_matrix, parsed.risk_test_matrix_changes)
        )
    human_section = _compact_human_requirement_dispositions(parsed.human_requirement_dispositions)
    if human_section:
        sections.append(human_section)
    if parsed.execution_recommendation is not None:
        sections.append(render_execution_recommendation_section(parsed.execution_recommendation))
    acknowledgement = _compact_human_requirements_block(
        _extract_plan_human_requirements_block(raw_text)
    )
    if acknowledgement:
        sections.append(acknowledgement)
    sections.append(f"<!-- AGENT_PLAN_STATE: {parsed.state} -->")
    sections.append(f"-- {_comment_signature(agent, config, model_used)}")
    return "\n\n".join(section for section in sections if section)


def _render_public_plan_revision_comment(
    parsed_revision: StructuredPlanRevision,
    *,
    prior_items: Sequence[UnresolvedReviewItem],
    raw_text: str,
    agent: str,
    config: AgentLoopConfig | None = None,
    model_used: str | None = None,
    compact: bool = False,
) -> str:
    if compact:
        return _render_compact_plan_digest(
            parsed_revision,
            title="## Revised plan",
            raw_text=raw_text,
            agent=agent,
            config=config,
            model_used=model_used,
        )
    sections = ["## Revised plan", render_canonical_plan_revision(parsed_revision, prior_items, config)]
    human_requirements_block = _extract_plan_revision_human_requirements_block(raw_text)
    if human_requirements_block:
        sections.append(human_requirements_block)
    sections.append(f"<!-- AGENT_PLAN_STATE: {parsed_revision.state} -->")
    sections.append(f"-- {_comment_signature(agent, config, model_used)}")
    return "\n\n".join(section for section in sections if section)


def _render_public_plan_state_comment(
    parsed_plan: StructuredPlanState,
    *,
    agent: str,
    config: AgentLoopConfig | None = None,
    model_used: str | None = None,
    compact: bool = False,
    raw_text: str = "",
) -> str:
    # ``raw_text`` is read only by the compact digest: the parsed fresh plan
    # does not retain the raw acknowledgement block.  The full rendering
    # ignores it and stays byte-identical.
    if compact:
        return _render_compact_plan_digest(
            parsed_plan,
            title="## Plan",
            raw_text=raw_text,
            agent=agent,
            config=config,
            model_used=model_used,
        )
    sections = [
        "## Plan",
        parsed_plan.summary.strip(),
        "\n".join(["### Plan steps", render_canonical_plan_steps(parsed_plan.plan_steps)]),
    ]
    if parsed_plan.risk_test_matrix is not None:
        sections.append(render_risk_test_matrix_section(parsed_plan.risk_test_matrix, parsed_plan.risk_test_matrix_changes))
    expected_section = render_expected_closing_issue_declaration(
        parsed_plan.additional_closing_issue_ids
    )
    if expected_section:
        sections.append(expected_section)
    human_section = render_human_requirement_dispositions(parsed_plan.human_requirement_dispositions)
    if human_section:
        sections.append(human_section)
    deferred_section = render_deferred_stages_section(parsed_plan.deferred_stages)
    if deferred_section:
        sections.append(deferred_section)
    typed_section = render_typed_plan_stages_section(parsed_plan.typed_stages)
    if typed_section:
        sections.append(typed_section)
    if parsed_plan.execution_recommendation is not None:
        sections.append(render_execution_recommendation_section(parsed_plan.execution_recommendation))
    sections.append(f"<!-- AGENT_PLAN_STATE: {parsed_plan.state} -->")
    sections.append(f"-- {_comment_signature(agent, config, model_used)}")
    return "\n\n".join(section for section in sections if section)


def normalize_freeform_signature(
    text: str,
    agent: AgentName,
    config: AgentLoopConfig | None,
    model_used: str | None,
) -> str:
    """Replace or append the trailing agent signature on a free-form response.

    Walks back over trailing blank lines and HTML comments to find the last
    SIGNATURE_RE line and replaces it with the canonical qualified form. If no
    signature is found, one is appended. Structured responses already receive
    canonical signatures via render_public_agent_comment; this function is for
    free-form/legacy text that is posted verbatim.
    """
    canonical = f"-- {agent_signature(agent, config, model_used)}"
    lines = text.rstrip("\n").splitlines()
    tail = len(lines)
    while tail > 0 and (
        not lines[tail - 1].strip() or HTML_COMMENT_RE.match(lines[tail - 1])
    ):
        tail -= 1
    if tail > 0 and SIGNATURE_RE.match(lines[tail - 1]):
        lines[tail - 1] = canonical
        return "\n".join(lines)
    return f"{text.rstrip(chr(10))}\n{canonical}"


def render_public_agent_comment(
    *,
    kind: str,
    parsed: (
        ParsedReview
        | ParsedPlanReview
        | StructuredCoderFollowup
        | StructuredPlanRevision
        | StructuredPlanState
        | ParsedDiscussReview
        | StructuredIssueImplementation
    ),
    agent: str,
    config: AgentLoopConfig | None = None,
    model_used: str | None = None,
    prior_items: Sequence[UnresolvedReviewItem] = (),
    dispositions: Sequence[ReviewItemDisposition] = (),
    raw_text: str = "",
    human_requirements_resolved_flag: bool = False,
    round_number: int = 1,
    local_test_evidence: str | None = None,
    current_test_turn_id: str | None = None,
    compact: bool = False,
    matrix_evidence_render_decision: MatrixEvidenceRenderDecision | None = None,
    head_unchanged_sha: str | None = None,
) -> str:
    """Render a parsed agent response and stamp the agent/model signature.

    This is the single render-and-sign seam used by CLI and skill mode. Callers
    resolve where the actual model came from, then pass it here; footer signing
    stays owned by the renderer.
    """
    if kind == "pr_review":
        if not isinstance(parsed, ParsedReview):
            raise AgentLoopError("render_public_agent_comment expected ParsedReview.")
        return _render_public_pr_review_comment(
            parsed,
            reviewer=agent,
            human_requirements_resolved_flag=human_requirements_resolved_flag,
            prior_items=prior_items,
            dispositions=dispositions,
            config=config,
            model_used=model_used,
        )
    if kind == "plan_review":
        if not isinstance(parsed, ParsedPlanReview):
            raise AgentLoopError("render_public_agent_comment expected ParsedPlanReview.")
        return _render_public_plan_review_comment(
            parsed,
            reviewer=agent,
            prior_items=prior_items,
            dispositions=dispositions,
            human_requirements_resolved_flag=human_requirements_resolved_flag,
            config=config,
            model_used=model_used,
        )
    if kind == "coder_followup":
        if not isinstance(parsed, StructuredCoderFollowup):
            raise AgentLoopError("render_public_agent_comment expected StructuredCoderFollowup.")
        return _render_public_coder_followup_comment(
            parsed,
            agent=agent,
            prior_items=prior_items,
            config=config,
            model_used=model_used,
            local_test_evidence=local_test_evidence,
            current_test_turn_id=current_test_turn_id,
            matrix_evidence_render_decision=matrix_evidence_render_decision,
            head_unchanged_sha=head_unchanged_sha,
        )
    if kind == "issue_implementation":
        if not isinstance(parsed, StructuredIssueImplementation):
            raise AgentLoopError(
                "render_public_agent_comment expected StructuredIssueImplementation."
            )
        return _render_public_issue_implementation_comment(
            parsed,
            agent=agent,
            config=config,
            model_used=model_used,
            local_test_evidence=local_test_evidence,
            current_test_turn_id=current_test_turn_id,
        )
    if kind == "plan_revision":
        if not isinstance(parsed, StructuredPlanRevision):
            raise AgentLoopError("render_public_agent_comment expected StructuredPlanRevision.")
        return _render_public_plan_revision_comment(
            parsed,
            prior_items=prior_items,
            raw_text=raw_text,
            agent=agent,
            config=config,
            model_used=model_used,
            compact=compact,
        )
    if kind == "plan_state":
        if not isinstance(parsed, StructuredPlanState):
            raise AgentLoopError("render_public_agent_comment expected StructuredPlanState.")
        return _render_public_plan_state_comment(
            parsed,
            agent=agent,
            config=config,
            model_used=model_used,
            compact=compact,
            raw_text=raw_text,
        )
    if kind == "discuss_review":
        if not isinstance(parsed, ParsedDiscussReview):
            raise AgentLoopError("render_public_agent_comment expected ParsedDiscussReview.")
        return _render_public_discuss_review_comment(
            parsed,
            reviewer=agent,
            round_number=round_number,
            config=config,
            model_used=model_used,
        )
    if kind == "discuss_answer":
        if not isinstance(parsed, ParsedDiscussAnswer):
            raise AgentLoopError("render_public_agent_comment expected ParsedDiscussAnswer.")
        return _render_public_discuss_answer_comment(
            parsed, reviewer=agent, round_number=round_number, config=config, model_used=model_used
        )
    raise AgentLoopError(f"Unknown render kind: {kind}")


_DISCUSS_OUTCOME_LABELS = {
    "implement": "Implement",
    "do-not-implement": "Do Not Implement",
    "needs-human": "Needs Human Review",
    "split": "Split",
}

_DISCUSS_RESEARCH_STATUS_LABELS = {
    "sourced": "done, sourced facts cited below",
    "not-needed": "not needed (no external-fact trigger applies)",
    "unavailable": "unavailable — related claims are judgment, not sourced fact",
    "inconclusive": "inconclusive — related claims are judgment, not sourced fact",
}


def _render_public_discuss_review_comment(
    parsed: ParsedDiscussReview,
    *,
    reviewer: str,
    round_number: int = 1,
    config: AgentLoopConfig | None = None,
    model_used: str | None = None,
) -> str:
    outcome_label = _DISCUSS_OUTCOME_LABELS.get(parsed.outcome, parsed.outcome)
    sections: list[str] = [
        f"## Round {round_number}: {reviewer} position",
        f"**Vote:** {outcome_label} (`{parsed.outcome}`)",
        parsed.rationale.strip(),
    ]
    if parsed.rebuttal:
        sections.append("### Rebuttal\n\n" + parsed.rebuttal.strip())
    if parsed.analyzer_framing == "misframed":
        note = (parsed.framing_note or "").strip()
        sections.append("### Analyzer framing correction\n\n" + note)
    elif parsed.analyzer_framing == "accurate":
        framing_line = "**Analyzer framing:** accurate"
        if parsed.framing_note:
            framing_line += f" — {parsed.framing_note.strip()}"
        sections.append(framing_line)
    if parsed.split_proposals:
        proposals = "\n".join(f"- {proposal}" for proposal in parsed.split_proposals)
        sections.append("### Proposed sub-issues\n\n" + proposals)
    if parsed.research_status is not None:
        label = _DISCUSS_RESEARCH_STATUS_LABELS.get(parsed.research_status, parsed.research_status)
        sections.append(f"**Research:** {label} (`{parsed.research_status}`)")
    if parsed.research_target:
        sections.append(
            "### Research intent\n\n"
            f"- Target: `{parsed.research_target}`\n"
            + "\n".join(f"- Question: {question}" for question in parsed.research_questions)
        )
    if parsed.sourced_facts:
        facts = "\n".join(
            f"- {fact.fact} — source: {fact.source}" for fact in parsed.sourced_facts
        )
        sections.append("### Sourced facts\n\n" + facts)
    sections.append(f"-- {_comment_signature(reviewer, config, model_used)}")
    return "\n\n".join(section for section in sections if section)


def _render_public_discuss_answer_comment(
    parsed: ParsedDiscussAnswer, *, reviewer: str, round_number: int = 1,
    config: AgentLoopConfig | None = None, model_used: str | None = None,
) -> str:
    sections = [f"## Round {round_number}: {reviewer} answer", f"**Position:** `{parsed.position}`"]
    if parsed.answer:
        sections.append("### Answer\n\n" + parsed.answer.strip())
    sections.append("### Rationale\n\n" + parsed.rationale.strip())
    sections.append(f"**Confidence:** `{parsed.confidence}`")
    sections.extend(_render_discuss_unresolved_item_sections([parsed]))
    if parsed.rebuttal:
        sections.append("### Rebuttal\n\n" + parsed.rebuttal.strip())
    if parsed.research_status is not None:
        sections.append(f"**Research:** `{parsed.research_status}`")
    if parsed.research_target:
        sections.append(
            "### Research intent\n\n"
            f"- Target: `{parsed.research_target}`\n"
            + "\n".join(f"- Question: {question}" for question in parsed.research_questions)
        )
    if parsed.sourced_facts:
        sections.append("### Sourced facts\n\n" + "\n".join(f"- {f.fact} — source: {f.source}" for f in parsed.sourced_facts))
    sections.append(f"-- {_comment_signature(reviewer, config, model_used)}")
    return "\n\n".join(sections)


def _render_discuss_unresolved_item_sections(
    votes: Sequence[ParsedDiscussAnswer],
) -> list[str]:
    headings = (
        ("blocker", "Blockers"),
        ("human-decision", "Human decisions"),
        ("follow-up", "Non-blocking follow-ups"),
    )
    sections: list[str] = []
    for status, heading in headings:
        seen: set[str] = set()
        texts: list[str] = []
        for vote in votes:
            for item in vote.unresolved_items:
                if item.status == status and item.text not in seen:
                    seen.add(item.text)
                    texts.append(item.text)
        if texts:
            sections.append(f"### {heading}\n\n" + "\n".join(f"- {text}" for text in texts))
    return sections


def _render_discuss_agenda_lines(votes: Sequence[ParsedDiscussResponse]) -> list[str]:
    lines: list[str] = []
    for vote in votes:
        if isinstance(vote, ParsedDiscussAnswer):
            position = vote.position
            detail = vote.answer or vote.rationale
        else:
            position = vote.outcome
            detail = vote.rationale
        lines.append(f"- {vote.reviewer} held `{position}`: {detail}")
    return lines


def _render_analyzer_agenda_lines(agenda: ParsedDiscussAgenda) -> list[str]:
    lines: list[str] = []
    if agenda.consensus:
        lines.append("Analyzer-extracted consensus so far (not debater-confirmed):")
        lines.extend(f"- {point}" for point in agenda.consensus)
    if agenda.disagreements:
        if lines:
            lines.append("")
        lines.append("Open disagreements:")
        for disagreement in agenda.disagreements:
            lines.append(f"- **{disagreement.topic}**")
            for name, position in disagreement.positions:
                lines.append(f"  - {name}: {position}")
            lines.append(
                f"  - Question for next round: {disagreement.question_for_next_round}"
            )
    if agenda.missing_facts:
        if lines:
            lines.append("")
        lines.append("Missing facts:")
        lines.extend(f"- {fact}" for fact in agenda.missing_facts)
    if agenda.research_required and agenda.research_questions:
        if lines:
            lines.append("")
        lines.append("Research brief for the next round (answer with cited sources):")
        for index, question in enumerate(agenda.research_questions):
            target = (agenda.research_question_targets[index]
                      if index < len(agenda.research_question_targets) else "legacy/unclassified")
            suffix = f" (target: `{target}`)" if target != "legacy/unclassified" else ""
            lines.append(f"- {question}{suffix}")
    return lines


def _render_discuss_research_section(
    *,
    research_mode: str,
    reviewer_votes: Sequence[ParsedDiscussResponse],
    round_history: Sequence[Sequence[ParsedDiscussResponse]] | None,
    evidence_reconciliation: dict[str, object] | None = None,
) -> list[str]:
    """Render the final-summary research section (#477).

    Keeps debater-cited external facts distinct from agent judgment, and makes
    missing, unavailable, or inconclusive research explicit instead of letting
    stale assumptions read as fact.
    """
    lines = ["### Research", "", f"Research policy: `{research_mode}`."]
    if research_mode == "none":
        lines.append("")
        lines.append("Online research was disabled; all positions are agent judgment.")
        return lines
    lines.append("")
    successful_votes = [v for v in reviewer_votes if not isinstance(v, ParsedFailedDiscussResponse)]
    for vote in successful_votes:
        if vote.research_status is None:
            lines.append(f"- {vote.reviewer}: no research status reported")
        else:
            label = _DISCUSS_RESEARCH_STATUS_LABELS.get(
                vote.research_status, vote.research_status
            )
            lines.append(f"- {vote.reviewer}: {label} (`{vote.research_status}`)")
        if getattr(vote, "research_target", None):
            lines.append(f"  Target: `{vote.research_target}`")
            lines.extend(f"  Question: {question}" for question in vote.research_questions)
    # Final summaries use the reconciled ledger.  Individual debater comments
    # remain the complete append-only audit trail.
    if evidence_reconciliation is not None:
        rendered = evidence_reconciliation.get("rendered", [])
        lines.append("Sourced facts cited by debaters are reported evidence unless directly verified below.")
        current = [item for item in rendered if "status" in item]
        history = [item for item in rendered if "action" in item]
        for status, heading in (
            ("verified", "Verified evidence"),
            ("reported-but-unverified", "Reported but unverified"),
            ("missing", "Missing facts"),
        ):
            entries = [item for item in current if item.get("status") == status]
            if entries:
                lines.extend(["", f"### {heading}", ""])
                for item in entries:
                    sources = "; ".join(item.get("sources", [])) or "no source supplied"
                    contributors = ", ".join(item.get("contributors", []))
                    refs = ", ".join(item.get("ids", []))
                    lines.append(f"- {item.get('fact')} — contributors: {contributors}; source: {sources} (`{refs}`)")
        if history:
            lines.extend(["", "### Retracted or superseded history", ""])
            for item in history:
                lines.append(f"- {item.get('action')}: {item.get('fact')} (`{item.get('id')}`); reason: {item.get('reason')}")
        omitted = evidence_reconciliation.get("omitted_entries", 0)
        lines.extend(["", f"Audit: {evidence_reconciliation.get('observation_count', 0)} observations, {evidence_reconciliation.get('update_count', 0)} updates; ledger digest `{evidence_reconciliation.get('digest')}`."])
        if omitted:
            lines.append(f"{omitted} lower-priority ledger entries were omitted from this bounded summary; round comments and replay metadata retain the raw audit trail.")
    else:
        # Compatibility for callers rendering an old transcript without the
        # persisted reconciliation artifact.
        fact_rounds = round_history if round_history else [reviewer_votes]
        sourced: list[str] = []
        seen: set[tuple[str, str, str]] = set()
        for votes in fact_rounds:
            for vote in votes:
                if isinstance(vote, ParsedFailedDiscussResponse):
                    continue
                for fact in vote.sourced_facts:
                    key = (vote.reviewer, fact.fact, fact.source)
                    if key not in seen:
                        seen.add(key)
                        sourced.append(f"- {vote.reviewer}: {fact.fact} — source: {fact.source}")
        if sourced:
            lines.extend(["", "Sourced facts cited by debaters (everything else above is agent judgment):", *sourced])
    statuses = [vote.research_status for vote in successful_votes]
    if statuses and all(status == "not-needed" for status in statuses):
        lines.append("")
        lines.append(
            "All debaters determined external research was unnecessary for this question."
        )
    gap_reviewers = [
        vote.reviewer
        for vote in successful_votes
        if vote.research_status in {"unavailable", "inconclusive"}
    ]
    if gap_reviewers:
        lines.append("")
        lines.append(
            f"Research was unavailable or inconclusive for {', '.join(gap_reviewers)}; "
            "treat their related claims as judgment, not sourced fact."
        )
    unreported = [vote.reviewer for vote in successful_votes if vote.research_status is None]
    if unreported:
        lines.append("")
        lines.append(
            f"No research status was reported by {', '.join(unreported)}; treat their "
            "claims as judgment, not sourced fact."
        )
    return lines


def _render_discuss_failed_debater_lines(
    failed_debaters: Sequence[tuple[str, str]],
    *,
    is_final: bool = False,
) -> list[str]:
    """Surface failed/timed-out debaters in the round summary (#475)."""
    if not failed_debaters:
        return []
    lines = ["", "### Debater failures", ""]
    for name, category in failed_debaters:
        lines.append(f"- {name}: no vote this round ({category})")
    if is_final:
        lines.append("")
        lines.append(
            "This round completed with partial results, so it cannot declare "
            "final consensus."
        )
    return lines


def render_discuss_round_summary_comment(
    *,
    is_final: bool,
    subject: str,
    round_number: int = 1,
    reviewer_votes: Sequence[ParsedDiscussResponse],
    outcome: str | None = None,
    consensus_kind: str = "unanimous",
    round_history: Sequence[Sequence[ParsedDiscussResponse]] | None = None,
    split_proposals: Sequence[str] | None = None,
    analyzer_agenda: ParsedDiscussAgenda | None = None,
    prior_analyzer_agenda: ParsedDiscussAgenda | None = None,
    final_analyzer_agenda: ParsedDiscussAgenda | None = None,
    analyzer_name: str | None = None,
    research_mode: str | None = None,
    failed_debaters: Sequence[tuple[str, str]] = (),
    result_mode: str = "triage",
    semantic_comparison: dict[str, object] | None = None,
    evidence_reconciliation: dict[str, object] | None = None,
    round_synthesis: ParsedDiscussRoundSynthesis | None = None,
    final_synthesis: ParsedDiscussFinalSynthesis | None = None,
) -> str:
    """Render the orchestrator/analyzer round-summary comment.

    When `is_final` is False this closes out an inconclusive round with an
    agenda for the next round; when True it renders the same consensus/deadlock
    content previously produced by `render_discuss_consensus_comment`, plus the
    marker that final-only idempotency and legacy detection rely on. When an
    analyzer agenda is supplied, the next-round agenda section shows the
    analyzer's structured output (attributed and auditable) instead of the
    mechanical per-vote lines, and final summaries add an analyzer-extracted
    consensus section kept distinct from the debater vote table.
    """
    split_proposals = list(split_proposals or ())
    if result_mode == "answer":
        return _render_discuss_answer_summary(
            is_final=is_final, subject=subject, round_number=round_number,
            reviewer_votes=reviewer_votes, consensus_kind=consensus_kind,
            round_history=round_history, analyzer_agenda=analyzer_agenda,
            prior_analyzer_agenda=prior_analyzer_agenda,
            final_analyzer_agenda=final_analyzer_agenda,
            analyzer_name=analyzer_name, research_mode=research_mode,
            failed_debaters=failed_debaters, outcome=outcome,
            semantic_comparison=semantic_comparison, evidence_reconciliation=evidence_reconciliation,
            round_synthesis=round_synthesis, final_synthesis=final_synthesis,
        )
    if not is_final:
        lines: list[str] = [
            f"## Round {round_number} summary: Consensus Pending",
            "",
            "| Reviewer | Outcome | Rationale |",
            "| --- | --- | --- |",
        ]
        for vote in reviewer_votes:
            rationale = vote.rationale.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {vote.reviewer} | {vote.outcome} | {rationale} |")
        lines.extend(_render_discuss_failed_debater_lines(failed_debaters))
        if split_proposals:
            lines.append("")
            lines.append("### Proposed sub-issues raised this round")
            lines.append("")
            for proposal in split_proposals:
                lines.append(f"- {proposal}")
        lines.append("")
        if analyzer_agenda is not None:
            heading = f"### Agenda for round {round_number + 1}"
            if analyzer_name:
                heading += f" (analyzer: {analyzer_name})"
            lines.append(heading)
            lines.append("")
            lines.extend(_render_analyzer_agenda_lines(analyzer_agenda))
        else:
            lines.append(f"### Agenda for round {round_number + 1}")
            lines.append("")
            lines.extend(_render_discuss_agenda_lines(reviewer_votes))
        lines.append("")
        lines.append("-- Orchestrator")
        return "\n".join(lines)

    if consensus_kind not in {"unanimous", "converged", "deadlock"}:
        raise AgentLoopError("consensus_kind must be `unanimous`, `converged`, or `deadlock`.")
    if outcome is None:
        raise AgentLoopError("outcome is required when is_final=True.")
    outcome_heading = {
        "implement": "Consensus: Implement",
        "do-not-implement": "Consensus: Do Not Implement",
        "needs-human": "Consensus: Needs Human Review",
        "split": "Consensus: Split",
    }.get(outcome, f"Consensus: {outcome}")
    if consensus_kind == "deadlock":
        outcome_heading = "Consensus: Needs Human Review (Deadlock)"
    lines = [
        f"## {outcome_heading}",
        "",
        f"Consensus kind: `{consensus_kind}` after round {round_number}.",
        "",
        "| Reviewer | Outcome | Rationale |",
        "| --- | --- | --- |",
    ]
    for vote in reviewer_votes:
        rationale = vote.rationale.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {vote.reviewer} | {vote.outcome} | {rationale} |")
    lines.extend(_render_discuss_failed_debater_lines(failed_debaters, is_final=True))
    rebuttal_votes = [vote for vote in reviewer_votes if vote.rebuttal]
    if rebuttal_votes:
        lines.append("")
        lines.append("### Final rebuttals")
        lines.append("")
        for vote in rebuttal_votes:
            lines.append(f"- {vote.reviewer}: {vote.rebuttal}")
    if consensus_kind == "deadlock":
        lines.append("")
        lines.append("### Core disagreement")
        lines.append("")
        lines.extend(_render_discuss_agenda_lines(reviewer_votes))
    if outcome == "split" and split_proposals:
        lines.append("")
        lines.append("### Proposed sub-issues")
        lines.append("")
        for proposal in split_proposals:
            lines.append(f"- {proposal}")
    # `analyzer_agenda` is retained as the non-final API for compatibility.
    # Final callers must pass a final-only analysis and a distinct historical agenda.
    if final_analyzer_agenda is not None:
        heading = "### Final analyzer observations (not debater-confirmed)"
        if analyzer_name:
            heading = f"### Final analyzer observations (analyzer: {analyzer_name}; not debater-confirmed)"
        lines.append("")
        lines.append(heading)
        lines.append("")
        lines.append(
            "The debater vote table above is authoritative. These advisory observations "
            "were derived only from final-round responses and are not debater-confirmed."
        )
        lines.append("")
        agenda_lines = _render_analyzer_agenda_lines(final_analyzer_agenda)
        lines.extend(agenda_lines if agenda_lines else ["(the analyzer extracted no points)"])
    if prior_analyzer_agenda is not None:
        lines.extend(["", "### Agenda before final round", "", "Historical analyzer agenda only; it is not a statement of current disagreements.", ""])
        agenda_lines = _render_analyzer_agenda_lines(prior_analyzer_agenda)
        lines.extend(agenda_lines if agenda_lines else ["(the analyzer extracted no points)"])
    if research_mode is not None:
        lines.append("")
        lines.extend(
            _render_discuss_research_section(
                research_mode=research_mode,
                reviewer_votes=reviewer_votes,
                round_history=round_history,
                evidence_reconciliation=evidence_reconciliation,
            )
        )
    if round_history:
        lines.append("")
        lines.append("### Round history")
        lines.append("")
        for index, votes in enumerate(round_history, start=1):
            rendered_votes = ", ".join(f"{vote.reviewer}: `{vote.outcome}`" for vote in votes)
            lines.append(f"- Round {index}: {rendered_votes}")
    lines.append("")
    lines.append("-- Orchestrator")
    lines.append(f"<!-- AGENT_DISCUSS_CONSENSUS: {subject} -->")
    return "\n".join(lines)


def _safe_synthesis_text(text: str) -> str:
    return sanitize_historical_text(text)


def _render_synthesis_consensus(item: object) -> list[str]:
    text = _safe_synthesis_text(item.text)
    references = ", ".join(
        f"{_safe_synthesis_text(reference.reviewer)} (round {reference.round})"
        for reference in item.references
    )
    return [f"- {text} _(supported by {references})_"]


def _render_synthesis_disagreement(item: object) -> list[str]:
    lines = [f"- **{_safe_synthesis_text(item.topic)}**"]
    for position in item.positions:
        names = ", ".join(_safe_synthesis_text(name) for name in position.reviewers)
        lines.append(f"  - {names}: {_safe_synthesis_text(position.position)}")
    lines.append(f"  - Decision needed: {_safe_synthesis_text(item.decision_needed)}")
    return lines


def _render_round_synthesis_lead(
    synthesis: ParsedDiscussRoundSynthesis, *, round_number: int
) -> list[str]:
    respondents = ", ".join(_safe_synthesis_text(name) for name in synthesis.responding_reviewers)
    lines = [f"## Round {round_number} discussion state", "", "### Current consensus"]
    if respondents:
        lines.append(f"Among responding debaters: {respondents}.")
    lines.extend(
        [_render_synthesis_consensus(item)[0] for item in synthesis.consensus]
        or ["- None established yet."]
    )
    lines.extend(["", "### Active disagreements"])
    lines.extend(
        line for item in synthesis.disagreements for line in _render_synthesis_disagreement(item)
    )
    if not synthesis.disagreements:
        lines.append("- None currently identified.")
    lines.extend(["", "### Changes this round"])
    if synthesis.changes:
        for change in synthesis.changes:
            refs = ", ".join(
                f"{_safe_synthesis_text(ref.reviewer)} (round {ref.round})"
                for ref in change.references
            )
            lines.append(
                f"- **{_safe_synthesis_text(change.kind)}** "
                f"{_safe_synthesis_text(change.topic)}: {_safe_synthesis_text(change.text)} "
                f"_(reported by {refs})_"
            )
    else:
        lines.append("- No state changes identified.")
    lines.extend(["", "### Missing facts"])
    lines.extend(
        f"- {_safe_synthesis_text(item)}" for item in synthesis.missing_facts
    )
    if not synthesis.missing_facts:
        lines.append("- None identified.")
    lines.extend(["", "### Next-round focus"])
    lines.extend(
        f"- {_safe_synthesis_text(item)}" for item in synthesis.next_round_focus
    )
    if not synthesis.next_round_focus:
        lines.append("- No additional focus identified.")
    return lines


def _render_final_synthesis_lead(
    synthesis: ParsedDiscussFinalSynthesis, *, round_number: int
) -> list[str]:
    lines = [
        "## Executive conclusion",
        "",
        "### Outcome",
        "",
        f"`{_safe_synthesis_text(synthesis.classification)}` after round {round_number}.",
        "",
        "### Agreed conclusions",
    ]
    lines.extend(
        line for item in synthesis.agreed_conclusions for line in _render_synthesis_consensus(item)
    )
    if not synthesis.agreed_conclusions:
        lines.append("- None established.")
    lines.extend(["", "### Remaining disagreements"])
    lines.extend(
        line
        for item in synthesis.remaining_disagreements
        for line in _render_synthesis_disagreement(item)
    )
    if not synthesis.remaining_disagreements:
        lines.append("- None; the configured mechanical outcome is supported by the final responses.")
    lines.extend(["", "### Next action", "", _safe_synthesis_text(synthesis.next_action)])
    return lines


def _bounded_answer_audit_excerpt(text: str, *, remaining: int) -> str:
    safe = sanitize_historical_text(text).replace("|", "\\|").replace("\n", " ")
    limit = min(1_500, max(0, remaining))
    if len(safe) > limit:
        return safe[: max(0, limit - 1)] + "…"
    return safe


def _render_discuss_answer_summary(*, is_final: bool, subject: str, round_number: int,
    reviewer_votes: Sequence[ParsedDiscussAnswer], consensus_kind: str,
    round_history: Sequence[Sequence[ParsedDiscussAnswer]] | None,
    analyzer_agenda: ParsedDiscussAgenda | None, prior_analyzer_agenda: ParsedDiscussAgenda | None,
    final_analyzer_agenda: ParsedDiscussAgenda | None, analyzer_name: str | None,
    research_mode: str | None, failed_debaters: Sequence[tuple[str, str]], outcome: str | None,
    semantic_comparison: dict[str, object] | None = None,
    evidence_reconciliation: dict[str, object] | None = None,
    round_synthesis: ParsedDiscussRoundSynthesis | None = None,
    final_synthesis: ParsedDiscussFinalSynthesis | None = None,
    _bounded_audit: bool = False) -> str:
    if round_synthesis is not None or final_synthesis is not None:
        # Keep the pre-synthesis renderer as a bounded audit section. The
        # executive state remains the only primary reading path.
        lead = (
            _render_final_synthesis_lead(final_synthesis, round_number=round_number)
            if final_synthesis is not None
            else _render_round_synthesis_lead(round_synthesis, round_number=round_number)
        )
        audit = _render_discuss_answer_summary(
            is_final=is_final, subject=subject, round_number=round_number,
            reviewer_votes=reviewer_votes, consensus_kind=consensus_kind,
            round_history=round_history, analyzer_agenda=analyzer_agenda,
            prior_analyzer_agenda=prior_analyzer_agenda,
            final_analyzer_agenda=final_analyzer_agenda, analyzer_name=analyzer_name,
            research_mode=research_mode, failed_debaters=failed_debaters, outcome=outcome,
            semantic_comparison=semantic_comparison,
            evidence_reconciliation=evidence_reconciliation,
            _bounded_audit=True,
        )
        # Long answer bodies remain available in their per-agent comments. Keep
        # the summary audit useful but bounded when the lead is present.
        if len(("\n".join(lead) + audit).encode("utf-8")) > MAX_GITHUB_BODY_CHARS:
            audit = "The complete debater responses and provenance remain available in the per-agent audit comments."
        return "\n".join(lead) + "\n\n<details>\n<summary>Audit details</summary>\n\n" + audit + "\n\n</details>"
    if not is_final:
        heading = f"## Round {round_number} summary: Answer Pending"
    elif outcome == "needs-human":
        heading = "## Needs Human Decision"
    elif outcome == "deadlock":
        heading = "## Deadlock"
    else:
        heading = "## " + ("Consensus Answer" if consensus_kind == "unanimous" else "Converged Answer")
    lines = [heading, "", f"Consensus kind: `{consensus_kind}` after round {round_number}.", ""]
    if is_final and outcome not in {"needs-human", "deadlock"}:
        answers = [v.answer for v in reviewer_votes if v.answer]
        if answers:
            answer = semantic_comparison.get("confirmed_answer") or semantic_comparison.get("shared_recommendation") if semantic_comparison else answers[0]
            lines.extend(["### Answer", "", str(answer), ""])
    lines.extend(["| Reviewer | Position | Confidence | Answer |", "| --- | --- | --- | --- |"])
    remaining_excerpt_bytes = 6_000
    for vote in reviewer_votes:
        if _bounded_audit:
            answer = _bounded_answer_audit_excerpt(
                vote.answer or "(no asserted answer)", remaining=remaining_excerpt_bytes
            )
            remaining_excerpt_bytes = max(0, remaining_excerpt_bytes - len(answer.encode("utf-8")))
        else:
            answer = (vote.answer or "(no asserted answer)").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {vote.reviewer} | {vote.position} | {vote.confidence} | {answer} |")
    if failed_debaters:
        lines.extend(_render_discuss_failed_debater_lines(failed_debaters, is_final=is_final))
    item_sections = _render_discuss_unresolved_item_sections(reviewer_votes)
    for section in item_sections:
        lines.extend(["", section])
    if is_final:
        statuses = {item.status for vote in reviewer_votes for item in vote.unresolved_items}
        if outcome == "needs-human":
            lines.extend(["", "The run needs human input because the listed human decisions remain."])
        elif outcome == "deadlock" and "blocker" in statuses:
            lines.extend(["", "The run is deadlocked because the listed blockers remain."])
        elif outcome not in {"deadlock", "needs-human"}:
            lines.extend(["", "The answer can proceed because no material items remain; non-blocking follow-ups do not block it."])
    if outcome == "deadlock":
        if not any(item.status == "blocker" for vote in reviewer_votes for item in vote.unresolved_items):
            lines.extend(["", "### Unresolved disagreement", "", "The debaters did not converge on one normalized answer, or a required semantic check or debater failed."])
    if semantic_comparison is not None:
        lines.extend(["", "### Semantic comparison (advisory; not a debater vote)", "",
            f"Analyzer: {semantic_comparison.get('analyzer', 'configured analyzer')}",
            f"Classification: `{semantic_comparison.get('classification', 'failed')}`"])
        if semantic_comparison.get("shared_recommendation"):
            lines.extend(["", "Shared recommendation:", str(semantic_comparison["shared_recommendation"])])
        decisions = semantic_comparison.get("remaining_decisions", ())
        if decisions:
            lines.extend(["", "Residual decisions:", *[f"- {item}" for item in decisions]])
        evidence = semantic_comparison.get("evidence", ())
        if evidence:
            lines.extend(["", "Evidence:"])
            for item in evidence:
                reviewer = getattr(item, "reviewer", None)
                supports = getattr(item, "supports", None)
                if reviewer and supports:
                    lines.append(f"- {reviewer}: {supports}")
        if semantic_comparison.get("confirmed_answer"):
            lines.extend(["", "The recommendation above was explicitly confirmed by every debater."])
    if not is_final and analyzer_agenda is not None:
        heading = f"### Agenda for round {round_number + 1}"
        if analyzer_name:
            heading += f" (analyzer: {analyzer_name})"
        lines.extend(["", heading, "", *_render_analyzer_agenda_lines(analyzer_agenda)])
    if final_analyzer_agenda is not None:
        lines.extend(["", "### Final analyzer observations (not debater-confirmed)", "", "The analyzer is non-authoritative; these observations use only final-round responses.", "", *_render_analyzer_agenda_lines(final_analyzer_agenda)])
    if prior_analyzer_agenda is not None:
        lines.extend(["", "### Agenda before final round", "", "Historical analyzer agenda only; it is not current-state analysis.", "", *_render_analyzer_agenda_lines(prior_analyzer_agenda)])
    if research_mode is not None:
        lines.extend(["", *_render_discuss_research_section(research_mode=research_mode, reviewer_votes=reviewer_votes, round_history=round_history, evidence_reconciliation=evidence_reconciliation)])
    lines.extend(["", "-- Orchestrator", f"<!-- AGENT_DISCUSS_CONSENSUS: {subject} -->"] if is_final else ["", "-- Orchestrator"])
    return "\n".join(lines)


_PLAN_STRICT_PRE_PANEL_PREFIX = "strict pre-panel fallback:"
_PLAN_POST_PANEL_PREFIX = "post-panel fallback:"
_PLAN_OPERATOR_OVERRIDE_PREFIX = "operator force-full:"


def render_plan_scheduling_audit(
    *,
    phase: str,
    reason: str,
    selected: Sequence[str],
    paused: Sequence[tuple[str, str]],
    primary: str | None,
    active_owners: Sequence[str],
    plan_subject: str,
    panel_evidence: bool,
    degraded_history_class: str,
    degraded_history_reason: str | None,
    force_full: bool,
    force_full_source: str | None,
    calls_avoided: int,
    unqualified_artifacts: Sequence[str] = (),
) -> str:
    """Render the posted planning scheduler record (#905, from #841).

    The wording deliberately distinguishes the strict pre-panel fallback, the
    post-panel fallback, and the operator override so an auditor can tell which
    rule selected the board without re-deriving it.

    The decision kind is keyed on authoritative decision data, not on the reason
    prefix alone: the scheduler also stamps the post-panel prefix on the
    *ordinary* owner-scoped remediation decision, which is not a fallback, so
    labelling it one would contradict the phase, selected-reviewer, and
    force-full fields rendered beside it.  The remediation board size is read
    from the paused list rather than assumed: owner-scoped selection equals the
    complete configured board whenever every secondary owns an active finding.
    """
    if reason.startswith(_PLAN_OPERATOR_OVERRIDE_PREFIX) or force_full_source == "operator":
        kind = "operator override (qualified panel opening, source `operator`)"
    elif reason.startswith(_PLAN_STRICT_PRE_PANEL_PREFIX):
        kind = "strict pre-panel fallback (primary-only, no latch)"
    elif reason.startswith(_PLAN_POST_PANEL_PREFIX) and phase == "full-board":
        kind = "post-panel fallback (complete board" + (
            ", automatic latch)" if force_full else ")"
        )
    elif phase == "remediation":
        kind = (
            "owner-scoped remediation decision ("
            + ("partial board" if paused else "complete board")
            + (", automatic latch)" if force_full else ", no automatic latch)")
        )
    else:
        kind = "ordinary staged planning decision"
    lines = [
        "Plan review scheduling audit.",
        "",
        f"- Decision kind: {kind}",
        f"- Phase: `{phase}`",
        f"- Candidate plan: `{plan_subject}`",
        f"- Selected reviewers: {', '.join(selected) or 'none'}",
        "- Paused reviewers: "
        + (
            "; ".join(f"{name} ({why})" for name, why in paused)
            if paused
            else "none"
        ),
        f"- Primary plan reviewer: {primary or '(none)'}",
        f"- Active finding owners: {', '.join(active_owners) or '(none)'}",
        f"- Qualified panel opening recorded: {'yes' if panel_evidence else 'no'}",
        f"- Force-full: {force_full} (source: {force_full_source or 'none'})",
        f"- Scheduler calls avoided cumulatively: {calls_avoided}",
        f"- Scheduling reason: {reason}",
    ]
    if degraded_history_class and degraded_history_class != "intact":
        lines.append(
            f"- Degraded planning history class: `{degraded_history_class}`"
            + (f" ({degraded_history_reason})" if degraded_history_reason else "")
        )
    if unqualified_artifacts:
        rendered = list(unqualified_artifacts[:6])
        suffix = " ..." if len(unqualified_artifacts) > 6 else ""
        lines.append(
            "- Unqualified pre-opening artifacts (not approvals, not ownership): "
            + "; ".join(rendered)
            + suffix
        )
    lines.extend(["", "-- Orchestrator"])
    return "\n".join(lines)


def render_plan_phase_advance(
    *,
    next_round_number: int,
    plan_subject: str,
    missing_reviewers: Sequence[str],
    phase: str,
) -> str:
    """Render the reviewer-only planning phase-advance record."""
    return "\n".join(
        [
            f"Plan review phase advance to round {next_round_number}.",
            "",
            "No planner turn is invoked for this round: the candidate plan is "
            f"byte-identical (`{plan_subject}`) and no must-fix plan item remains.",
            "",
            f"- Outstanding phase after this advance: `{phase}`",
            "- Reviewer(s) still missing a qualifying exact-plan approval: "
            + (", ".join(missing_reviewers) or "none"),
            "",
            "-- Orchestrator",
        ]
    )
