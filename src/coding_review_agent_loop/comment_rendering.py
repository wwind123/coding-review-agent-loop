"""Public comment and canonical plan rendering helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .errors import AgentLoopError
from .agents.registry import agent_display_name, agent_signature
from .protocol import (
    ANY_HEADING_RE,
    HTML_COMMENT_RE,
    PLAN_STATE_RE,
    PRIOR_UNRESOLVED_ITEM_DISPOSITIONS_HEADING_RE,
    PRIOR_UNRESOLVED_PLAN_ITEM_DISPOSITIONS_HEADING_RE,
    SIGNATURE_RE,
    ParsedDiscussAgenda,
    ParsedDiscussReview,
    ParsedPlanReview,
    ParsedReview,
    ReviewItemDisposition,
    StructuredCoderFollowup,
    StructuredPlanState,
    StructuredPlanRevision,
    UnresolvedReviewItem,
    review_freeform_summary_text,
)
from .unresolved_items import HUMAN_REQUIREMENTS_ACK_ITEM_ID

if TYPE_CHECKING:
    from .agents.base import AgentName
    from .config import AgentLoopConfig

ITEM_SUMMARY_LIMIT = 100
# Reverse map display-name -> agent. agent_display_name is config-independent, so
# this is safe to build at import; the signature itself is resolved per-call via
# agent_signature(agent, config) so it can reflect the configured model (#332).
_AGENT_BY_DISPLAY_NAME = {
    agent_display_name(agent): agent
    for agent in ("claude", "codex", "gemini", "antigravity")
}


def _review_freeform_summary_text(text: str) -> str:
    return review_freeform_summary_text(text)


def _normalize_item_summary(text: str, *, limit: int = ITEM_SUMMARY_LIMIT) -> str:
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
) -> str:
    agent = _AGENT_BY_DISPLAY_NAME.get(name)
    if agent is None and name in {"claude", "codex", "gemini", "antigravity"}:
        agent = name  # type: ignore[assignment]
    if agent is None:
        return name
    # `model_used` is the producing agent's actual model (footers). Prior-item
    # attributions pass model_used=None and use the configured model (or the
    # generic signature when nothing is declared) — no cross-agent leakage.
    return agent_signature(agent, config, model_used)


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
    phrases = {
        "blocking": "Blocking issue",
        "same-pr": "Same-PR follow-up",
        "same-plan": "Same-plan follow-up",
        "future": "Future follow-up",
    }
    phrase = phrases.get(status, "Unresolved item")
    reviewer_name = _public_reviewer_name(item.reviewer, config)
    return f"{phrase} from {reviewer_name}, round {item.source_round}: {summary}"


def _render_disposition_status(disposition: ReviewItemDisposition) -> str:
    labels = {
        "resolved": "resolved",
        "blocking": "still blocking",
        "same-pr": "same-pr",
        "same-plan": "same-plan",
        "future": "future follow-up",
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
        lines.append(
            f"- [{disposition.item_id}] {_format_unresolved_item_label(item, config)}"
            f" -> {_render_disposition_status(disposition)}"
        )
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


def _render_public_coder_followup_comment(
    parsed_followup: StructuredCoderFollowup,
    *,
    agent: str,
    prior_items: Sequence[UnresolvedReviewItem] = (),
    config: AgentLoopConfig | None = None,
    model_used: str | None = None,
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
                note_label="Resolution",
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
            "\n".join(["### Tests run", *[f"- {test}" for test in parsed_followup.tests_run]])
        )
    sections.append(f"<!-- AGENT_STATE: {parsed_followup.state} -->")
    sections.append(f"-- {_comment_signature(agent, config, model_used)}")
    return "\n\n".join(section for section in sections if section)


def _extract_plan_revision_human_requirements_block(text: str) -> str:
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return ""
    decoder = json.JSONDecoder()
    payload, end = decoder.raw_decode(stripped)
    if not isinstance(payload, dict):
        return ""
    trailing = stripped[end:].lstrip()
    state_match = PLAN_STATE_RE.search(trailing)
    if state_match is None:
        return ""
    return trailing[: state_match.start()].strip()


def _render_public_plan_revision_comment(
    parsed_revision: StructuredPlanRevision,
    *,
    prior_items: Sequence[UnresolvedReviewItem],
    raw_text: str,
    agent: str,
    config: AgentLoopConfig | None = None,
    model_used: str | None = None,
) -> str:
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
) -> str:
    sections = [
        "## Plan",
        parsed_plan.summary.strip(),
        "\n".join(["### Plan steps", render_canonical_plan_steps(parsed_plan.plan_steps)]),
        f"<!-- AGENT_PLAN_STATE: {parsed_plan.state} -->",
        f"-- {_comment_signature(agent, config, model_used)}",
    ]
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
    ),
    agent: str,
    config: AgentLoopConfig | None = None,
    model_used: str | None = None,
    prior_items: Sequence[UnresolvedReviewItem] = (),
    dispositions: Sequence[ReviewItemDisposition] = (),
    raw_text: str = "",
    human_requirements_resolved_flag: bool = False,
    round_number: int = 1,
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
        )
    if kind == "plan_state":
        if not isinstance(parsed, StructuredPlanState):
            raise AgentLoopError("render_public_agent_comment expected StructuredPlanState.")
        return _render_public_plan_state_comment(
            parsed,
            agent=agent,
            config=config,
            model_used=model_used,
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
    if parsed.sourced_facts:
        facts = "\n".join(
            f"- {fact.fact} — source: {fact.source}" for fact in parsed.sourced_facts
        )
        sections.append("### Sourced facts\n\n" + facts)
    sections.append(f"-- {_comment_signature(reviewer, config, model_used)}")
    return "\n\n".join(section for section in sections if section)


def _render_discuss_agenda_lines(votes: Sequence[ParsedDiscussReview]) -> list[str]:
    return [f"- {vote.reviewer} held `{vote.outcome}`: {vote.rationale}" for vote in votes]


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
        lines.extend(f"- {question}" for question in agenda.research_questions)
    return lines


def _render_discuss_research_section(
    *,
    research_mode: str,
    reviewer_votes: Sequence[ParsedDiscussReview],
    round_history: Sequence[Sequence[ParsedDiscussReview]] | None,
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
    for vote in reviewer_votes:
        if vote.research_status is None:
            lines.append(f"- {vote.reviewer}: no research status reported")
        else:
            label = _DISCUSS_RESEARCH_STATUS_LABELS.get(
                vote.research_status, vote.research_status
            )
            lines.append(f"- {vote.reviewer}: {label} (`{vote.research_status}`)")
    fact_rounds = round_history if round_history else [reviewer_votes]
    sourced: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for votes in fact_rounds:
        for vote in votes:
            for fact in vote.sourced_facts:
                key = (vote.reviewer, fact.fact, fact.source)
                if key not in seen:
                    seen.add(key)
                    sourced.append(f"- {vote.reviewer}: {fact.fact} — source: {fact.source}")
    if sourced:
        lines.append("")
        lines.append(
            "Sourced facts cited by debaters (everything else above is agent judgment):"
        )
        lines.extend(sourced)
    statuses = [vote.research_status for vote in reviewer_votes]
    if statuses and all(status == "not-needed" for status in statuses):
        lines.append("")
        lines.append(
            "All debaters determined external research was unnecessary for this question."
        )
    gap_reviewers = [
        vote.reviewer
        for vote in reviewer_votes
        if vote.research_status in {"unavailable", "inconclusive"}
    ]
    if gap_reviewers:
        lines.append("")
        lines.append(
            f"Research was unavailable or inconclusive for {', '.join(gap_reviewers)}; "
            "treat their related claims as judgment, not sourced fact."
        )
    unreported = [vote.reviewer for vote in reviewer_votes if vote.research_status is None]
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
    reviewer_votes: Sequence[ParsedDiscussReview],
    outcome: str | None = None,
    consensus_kind: str = "unanimous",
    round_history: Sequence[Sequence[ParsedDiscussReview]] | None = None,
    split_proposals: Sequence[str] | None = None,
    analyzer_agenda: ParsedDiscussAgenda | None = None,
    analyzer_name: str | None = None,
    research_mode: str | None = None,
    failed_debaters: Sequence[tuple[str, str]] = (),
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
    if analyzer_agenda is not None:
        heading = "### Analyzer-extracted consensus (not debater-confirmed)"
        if analyzer_name:
            heading = f"### Analyzer-extracted consensus (analyzer: {analyzer_name}; not debater-confirmed)"
        lines.append("")
        lines.append(heading)
        lines.append("")
        lines.append(
            "The debater vote table above is the authoritative consensus. The analyzer's "
            "last agenda extracted the following; it may misclassify consensus or omit "
            "minority arguments."
        )
        lines.append("")
        agenda_lines = _render_analyzer_agenda_lines(analyzer_agenda)
        lines.extend(agenda_lines if agenda_lines else ["(the analyzer extracted no points)"])
    if research_mode is not None:
        lines.append("")
        lines.extend(
            _render_discuss_research_section(
                research_mode=research_mode,
                reviewer_votes=reviewer_votes,
                round_history=round_history,
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
