"""Discussion response parsing and canonical synthesis serialization."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from .errors import AgentLoopError
from .protocol_discuss_types import (
    DiscussUnresolvedItem,
    DISCUSS_UNRESOLVED_ITEM_STATUS_VALUES,
    DISCUSS_EVIDENCE_STATUS_VALUES,
    DISCUSS_EVIDENCE_UPDATE_ACTIONS,
    DISCUSS_VERIFICATION_BASES,
    DiscussEvidenceClaim,
    DiscussEvidenceUpdate,
    ParsedDiscussAnswer,
    DiscussSemanticEvidence,
    ParsedDiscussSemanticComparison,
    ParsedDiscussAnswerConfirmation,
    ParsedDiscussEvidenceReconciliation,
    DiscussSourcedFact,
    ParsedDiscussReview,
    DISCUSS_OUTCOME_VALUES,
    DISCUSS_ANALYZER_FRAMING_VALUES,
    DISCUSS_RESEARCH_STATUS_VALUES,
    DISCUSS_RESEARCH_TARGET_VALUES,
    DiscussAgendaDisagreement,
    DISCUSS_SYNTHESIS_MAX_ENTRIES,
    DISCUSS_SYNTHESIS_MAX_NEXT_ROUND_FOCUS,
    DISCUSS_SYNTHESIS_MAX_TEXT_BYTES,
    DISCUSS_SYNTHESIS_MAX_CANONICAL_BYTES,
    DISCUSS_SYNTHESIS_CHANGE_KINDS,
    DISCUSS_SYNTHESIS_CLASSIFICATIONS,
    DiscussSynthesisResponseReference,
    DiscussSynthesisConsensus,
    DiscussSynthesisPosition,
    DiscussSynthesisDisagreement,
    DiscussSynthesisChange,
    ParsedDiscussRoundSynthesis,
    ParsedDiscussFinalSynthesis,
    ParsedDiscussAgenda,
)
from .protocol_common import (
    PLAN_STATE_RE,
    _consume_structured_footer_and_signature,
    _expect_bool,
    _expect_exact_keys,
    _expect_int,
    _expect_non_empty_string,
    _expect_object,
    _expect_optional_string_list,
    _expect_string_list,
    _extract_json_object_prefix,
    _require_supported_schema_version,
    normalize_response_file_structured_text,
)


def _extract_structured_discuss_review_payload(
    text: str, *, context_label: str = "Structured discuss review"
) -> dict[str, object] | None:
    text, _status = normalize_response_file_structured_text(text)
    extracted = _extract_json_object_prefix(text)  # shape-check: fatal:unparseable-envelope
    if extracted is None:
        return None
    payload, trailing = extracted
    result = _consume_structured_footer_and_signature(  # shape-check: fatal:unparseable-envelope
        payload=payload,
        trailing=trailing,
        state_re=PLAN_STATE_RE,
        state_marker_name="AGENT_PLAN_STATE",
        context_label=context_label,
    )
    if result is None:
        return None
    footer_match = PLAN_STATE_RE.search(trailing)
    if footer_match is not None:
        footer_state = footer_match.group(1).lower()
        if footer_state != "approved":
            raise AgentLoopError(  # shape-check: fatal:unparseable-envelope
                f"{context_label} footer AGENT_PLAN_STATE must be `approved`; got `{footer_state}`."
            )
    return result

def _extract_structured_discuss_agenda_payload(text: str) -> dict[str, object] | None:
    text, _status = normalize_response_file_structured_text(text)
    extracted = _extract_json_object_prefix(text)  # shape-check: fatal:unparseable-envelope
    if extracted is None:
        return None
    payload, trailing = extracted
    result = _consume_structured_footer_and_signature(  # shape-check: fatal:unparseable-envelope
        payload=payload,
        trailing=trailing,
        state_re=PLAN_STATE_RE,
        state_marker_name="AGENT_PLAN_STATE",
        context_label="Structured discuss agenda",
    )
    if result is None:
        return None
    footer_match = PLAN_STATE_RE.search(trailing)
    if footer_match is not None:
        footer_state = footer_match.group(1).lower()
        if footer_state != "approved":
            raise AgentLoopError(  # shape-check: fatal:unparseable-envelope
                f"Structured discuss agenda footer AGENT_PLAN_STATE must be `approved`; got `{footer_state}`."
            )
    return result

def _parse_discuss_research(
    value: object,
) -> tuple[str, tuple[DiscussSourcedFact, ...], str | None, tuple[str, ...]]:
    payload = _expect_object(value, context="discuss_review.research")  # shape-check: fatal:no-conservative-reading
    _expect_exact_keys(  # shape-check: fatal:no-conservative-reading
        payload,
        context="discuss_review.research",
        required={"status"},
        optional={"sourced_facts", "target", "questions"},
    )
    status = _expect_non_empty_string(payload["status"], context="discuss_review.research.status")  # shape-check: fatal:no-conservative-reading
    if status not in DISCUSS_RESEARCH_STATUS_VALUES:
        rendered = ", ".join(sorted(DISCUSS_RESEARCH_STATUS_VALUES))
        raise AgentLoopError(f"discuss_review.research.status must be one of: {rendered}")  # shape-check: fatal:no-conservative-reading
    facts_value = payload.get("sourced_facts", [])
    if not isinstance(facts_value, list):
        raise AgentLoopError("discuss_review.research.sourced_facts must be a JSON array.")  # shape-check: fatal:no-conservative-reading
    sourced_facts: list[DiscussSourcedFact] = []
    for index, item in enumerate(facts_value):
        context = f"discuss_review.research.sourced_facts at index {index}"
        fact_payload = _expect_object(item, context=context)  # shape-check: fatal:no-conservative-reading
        _expect_exact_keys(fact_payload, context=context, required={"fact", "source"})  # shape-check: fatal:no-conservative-reading
        sourced_facts.append(
            DiscussSourcedFact(
                fact=_expect_non_empty_string(fact_payload["fact"], context=f"{context}.fact"),  # shape-check: fatal:no-conservative-reading
                source=_expect_non_empty_string(  # shape-check: fatal:no-conservative-reading
                    fact_payload["source"], context=f"{context}.source"
                ),
            )
        )
    if status == "sourced" and not sourced_facts:
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            "discuss_review.research.sourced_facts must be non-empty when status is `sourced`."
        )
    if status != "sourced" and sourced_facts:
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            "discuss_review.research.sourced_facts requires status `sourced`."
        )
    has_target = "target" in payload
    has_questions = "questions" in payload
    if has_target != has_questions:
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            "discuss_review.research.target and research.questions must be supplied together."
        )
    target: str | None = None
    questions: tuple[str, ...] = ()
    if has_target:
        target = _expect_non_empty_string(payload["target"], context="discuss_review.research.target")  # shape-check: fatal:no-conservative-reading
        if target not in DISCUSS_RESEARCH_TARGET_VALUES:
            rendered = ", ".join(sorted(DISCUSS_RESEARCH_TARGET_VALUES))
            raise AgentLoopError(f"discuss_review.research.target must be one of: {rendered}")  # shape-check: fatal:no-conservative-reading
        questions = _expect_string_list(  # shape-check: fatal:no-conservative-reading
            payload["questions"],
            context="discuss_review.research.questions",
            item_context="discuss_review.research.questions",
        )
        if not questions:
            raise AgentLoopError("discuss_review.research.questions must be non-empty when research intent is supplied.")  # shape-check: fatal:no-conservative-reading
        if status == "not-needed":
            raise AgentLoopError("discuss_review.research intent requires an active research status.")  # shape-check: fatal:no-conservative-reading
    return status, tuple(sourced_facts), target, questions

def _parse_discuss_evidence(
    value: object,
) -> tuple[tuple[DiscussEvidenceClaim, ...], tuple[DiscussEvidenceUpdate, ...]]:
    """Parse explicit evidence without trying to infer meaning from prose.

    This deliberately stays optional so persisted pre-#535 transcript responses
    can still be replayed.  Legacy ``research.sourced_facts`` are projected by
    the reconciliation layer as reported observations.
    """
    payload = _expect_object(value, context="discuss evidence")  # shape-check: fatal:no-conservative-reading
    _expect_exact_keys(payload, context="discuss evidence", required={"claims", "updates"})  # shape-check: fatal:no-conservative-reading
    claims_value = payload["claims"]
    updates_value = payload["updates"]
    if not isinstance(claims_value, list) or not isinstance(updates_value, list):
        raise AgentLoopError("discuss evidence.claims and evidence.updates must be JSON arrays.")  # shape-check: fatal:no-conservative-reading
    claims: list[DiscussEvidenceClaim] = []
    for index, item in enumerate(claims_value):
        context = f"discuss evidence.claims at index {index}"
        claim = _expect_object(item, context=context)  # shape-check: fatal:no-conservative-reading
        _expect_exact_keys(  # shape-check: fatal:no-conservative-reading
            claim, context=context, required={"fact", "status"},
            optional={"source", "verification_basis"},
        )
        fact = _expect_non_empty_string(claim["fact"], context=f"{context}.fact")  # shape-check: fatal:no-conservative-reading
        status = _expect_non_empty_string(claim["status"], context=f"{context}.status")  # shape-check: fatal:no-conservative-reading
        if status not in DISCUSS_EVIDENCE_STATUS_VALUES:
            raise AgentLoopError("discuss evidence claim status must be verified, reported-but-unverified, or missing.")  # shape-check: fatal:no-conservative-reading
        source = (_expect_non_empty_string(claim["source"], context=f"{context}.source")  # shape-check: fatal:no-conservative-reading
                  if "source" in claim else None)
        basis = (_expect_non_empty_string(claim["verification_basis"], context=f"{context}.verification_basis")  # shape-check: fatal:no-conservative-reading
                 if "verification_basis" in claim else None)
        if status == "missing" and (source is not None or basis is not None):
            raise AgentLoopError("missing evidence claims cannot carry a source or verification basis.")  # shape-check: fatal:no-conservative-reading
        if status == "verified":
            if basis not in DISCUSS_VERIFICATION_BASES or source is None:
                raise AgentLoopError("verified evidence requires source and verification_basis external-source-inspected or checkout-inspected.")  # shape-check: fatal:no-conservative-reading
            if basis == "checkout-inspected" and not re.fullmatch(r"[^\s:][^:]*:\d+", source):
                raise AgentLoopError("checkout-inspected verified evidence requires a repository-relative path:line source.")  # shape-check: fatal:no-conservative-reading
        elif basis is not None:
            raise AgentLoopError("only verified evidence may carry verification_basis.")  # shape-check: fatal:no-conservative-reading
        claims.append(DiscussEvidenceClaim(fact=fact, status=status, source=source, verification_basis=basis))
    updates: list[DiscussEvidenceUpdate] = []
    for index, item in enumerate(updates_value):
        context = f"discuss evidence.updates at index {index}"
        update = _expect_object(item, context=context)  # shape-check: fatal:no-conservative-reading
        _expect_exact_keys(  # shape-check: fatal:no-conservative-reading
            update, context=context, required={"action", "target_observation_id", "reason"},
            optional={"replacement_claim_index"},
        )
        action = _expect_non_empty_string(update["action"], context=f"{context}.action")  # shape-check: fatal:no-conservative-reading
        if action not in DISCUSS_EVIDENCE_UPDATE_ACTIONS:
            raise AgentLoopError("discuss evidence update action must be retract or supersede.")  # shape-check: fatal:no-conservative-reading
        replacement = update.get("replacement_claim_index")
        if replacement is not None and (not isinstance(replacement, int) or isinstance(replacement, bool) or replacement < 0 or replacement >= len(claims)):
            raise AgentLoopError("evidence replacement_claim_index must reference a claim in the same response.")  # shape-check: fatal:no-conservative-reading
        updates.append(DiscussEvidenceUpdate(
            action=action,
            target_observation_id=_expect_non_empty_string(update["target_observation_id"], context=f"{context}.target_observation_id"),  # shape-check: fatal:no-conservative-reading
            reason=_expect_non_empty_string(update["reason"], context=f"{context}.reason"),  # shape-check: fatal:no-conservative-reading
            replacement_claim_index=replacement,
        ))
    return tuple(claims), tuple(updates)

def parse_structured_discuss_review(
    text: str, *, reviewer: str, round_number: int = 1, research_mode: str | None = None
) -> ParsedDiscussReview | None:
    payload = _extract_structured_discuss_review_payload(  # shape-check: fatal:unparseable-envelope
        text, context_label="Structured discuss review"
    )
    if payload is None:
        return None
    _require_supported_schema_version(payload)  # shape-check: fatal:kind-or-version-mismatch
    kind = payload.get("kind")
    if isinstance(kind, str) and kind != "discuss_review":
        raise AgentLoopError("Structured response kind mismatch: expected `discuss_review`.")  # shape-check: fatal:unparseable-envelope
    _expect_exact_keys(  # shape-check: fatal:unparseable-envelope
        payload,
        context="discuss_review",
        required={"schema_version", "kind", "outcome", "rationale"},
        optional={"split_proposals", "rebuttal", "analyzer_framing", "framing_note", "research", "evidence"},
    )
    outcome = _expect_non_empty_string(payload["outcome"], context="discuss_review.outcome")  # shape-check: fatal:no-conservative-reading
    if outcome not in DISCUSS_OUTCOME_VALUES:
        rendered = ", ".join(sorted(DISCUSS_OUTCOME_VALUES))
        raise AgentLoopError(f"discuss_review.outcome must be one of: {rendered}")  # shape-check: fatal:no-conservative-reading
    rationale = _expect_non_empty_string(payload["rationale"], context="discuss_review.rationale")  # shape-check: fatal:no-conservative-reading
    split_proposals = _expect_optional_string_list(  # shape-check: fatal:no-conservative-reading
        payload,
        "split_proposals",
        context="discuss_review.split_proposals",
        item_context="discuss_review.split_proposals",
    )
    if outcome == "split" and not split_proposals:
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            "discuss_review.split_proposals must be non-empty when outcome is `split`."
        )
    rebuttal = None
    if "rebuttal" in payload:
        rebuttal = _expect_non_empty_string(payload["rebuttal"], context="discuss_review.rebuttal")  # shape-check: fatal:no-conservative-reading
    if round_number > 1 and rebuttal is None:
        raise AgentLoopError("discuss_review.rebuttal is required for debate rounds.")  # shape-check: fatal:no-conservative-reading
    analyzer_framing = None
    if "analyzer_framing" in payload:
        analyzer_framing = _expect_non_empty_string(  # shape-check: fatal:no-conservative-reading
            payload["analyzer_framing"], context="discuss_review.analyzer_framing"
        )
        if analyzer_framing not in DISCUSS_ANALYZER_FRAMING_VALUES:
            rendered = ", ".join(sorted(DISCUSS_ANALYZER_FRAMING_VALUES))
            raise AgentLoopError(f"discuss_review.analyzer_framing must be one of: {rendered}")  # shape-check: fatal:no-conservative-reading
    framing_note = None
    if "framing_note" in payload:
        framing_note = _expect_non_empty_string(  # shape-check: fatal:no-conservative-reading
            payload["framing_note"], context="discuss_review.framing_note"
        )
    if analyzer_framing == "misframed" and framing_note is None:
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            "discuss_review.framing_note is required when analyzer_framing is `misframed`."
        )
    if framing_note is not None and analyzer_framing is None:
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            "discuss_review.framing_note requires analyzer_framing to be set."
        )
    research_status: str | None = None
    sourced_facts: tuple[DiscussSourcedFact, ...] = ()
    research_target: str | None = None
    research_questions: tuple[str, ...] = ()
    if "research" in payload:
        research_status, sourced_facts, research_target, research_questions = _parse_discuss_research(payload["research"])  # shape-check: fatal:no-conservative-reading
    evidence_claims, evidence_updates = ((), ())
    if "evidence" in payload:
        evidence_claims, evidence_updates = _parse_discuss_evidence(payload["evidence"])  # shape-check: fatal:no-conservative-reading
    if research_mode == "required":
        if research_status is None:
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                "discuss_review.research is required when the research policy is `required`."
            )
        if research_status == "not-needed":
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                "discuss_review.research.status must not be `not-needed` when the "
                "research policy is `required`; use `sourced`, `unavailable`, or "
                "`inconclusive`."
            )
    return ParsedDiscussReview(
        outcome=outcome,
        rationale=rationale,
        split_proposals=split_proposals,
        reviewer=reviewer,
        rebuttal=rebuttal,
        analyzer_framing=analyzer_framing,
        framing_note=framing_note,
        research_status=research_status,
        sourced_facts=sourced_facts,
        research_target=research_target,
        research_questions=research_questions,
        evidence_claims=evidence_claims,
        evidence_updates=evidence_updates,
    )

def validate_structured_discuss_review(
    text: str, *, reviewer: str, round_number: int = 1, research_mode: str | None = None
) -> ParsedDiscussReview:
    parsed = parse_structured_discuss_review(  # shape-check: fatal:no-conservative-reading
        text, reviewer=reviewer, round_number=round_number, research_mode=research_mode
    )
    if parsed is not None:
        return parsed
    raise AgentLoopError("Discuss review did not use the required structured format.")  # shape-check: fatal:no-conservative-reading

DISCUSS_ANSWER_POSITION_VALUES = frozenset({"answer", "needs-human"})

DISCUSS_ANSWER_CONFIDENCE_VALUES = frozenset({"low", "medium", "high"})

def parse_structured_discuss_answer(
    text: str, *, reviewer: str, round_number: int = 1, research_mode: str | None = None
) -> ParsedDiscussAnswer | None:
    return _parse_structured_discuss_answer(  # shape-check: fatal:no-conservative-reading
        text, reviewer=reviewer, round_number=round_number, research_mode=research_mode,
        allow_legacy_open_questions=False,
    )

def parse_legacy_structured_discuss_answer(
    text: str, *, reviewer: str, round_number: int = 1, research_mode: str | None = None
) -> ParsedDiscussAnswer | None:
    """Decode only persisted pre-#536 answer votes.

    New agent output must use ``unresolved_items``.  Legacy untyped questions
    are conservatively mapped to blockers for asserted answers, and to human
    decisions for escalations, so a resumed run cannot silently proceed.
    """
    return _parse_structured_discuss_answer(  # shape-check: fatal:no-conservative-reading
        text, reviewer=reviewer, round_number=round_number, research_mode=research_mode,
        allow_legacy_open_questions=True,
    )

def _parse_discuss_unresolved_items(value: object, *, context: str) -> tuple[DiscussUnresolvedItem, ...]:
    if not isinstance(value, list):
        raise AgentLoopError(f"{context} must be a JSON array.")  # shape-check: fatal:no-conservative-reading
    items: list[DiscussUnresolvedItem] = []
    for index, item in enumerate(value):
        item_context = f"{context}[{index}]"
        if not isinstance(item, dict):
            raise AgentLoopError(f"{item_context} must be an object.")  # shape-check: fatal:no-conservative-reading
        _expect_exact_keys(item, context=item_context, required={"status", "text"})  # shape-check: fatal:no-conservative-reading
        status = _expect_non_empty_string(item["status"], context=f"{item_context}.status")  # shape-check: fatal:no-conservative-reading
        if status not in DISCUSS_UNRESOLVED_ITEM_STATUS_VALUES:
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                f"{item_context}.status must be one of: blocker, human-decision, follow-up."
            )
        items.append(DiscussUnresolvedItem(
            status=status,
            text=_expect_non_empty_string(item["text"], context=f"{item_context}.text"),  # shape-check: fatal:no-conservative-reading
        ))
    return tuple(items)

def _parse_structured_discuss_answer(
    text: str, *, reviewer: str, round_number: int, research_mode: str | None,
    allow_legacy_open_questions: bool,
) -> ParsedDiscussAnswer | None:
    payload = _extract_structured_discuss_review_payload(  # shape-check: fatal:unparseable-envelope
        text, context_label="Structured discuss answer"
    )
    if payload is None:
        return None
    _require_supported_schema_version(payload)  # shape-check: fatal:kind-or-version-mismatch
    if payload.get("kind") != "discuss_answer":
        raise AgentLoopError("Structured response kind mismatch: expected `discuss_answer`.")  # shape-check: fatal:no-conservative-reading
    item_key = "open_questions" if allow_legacy_open_questions else "unresolved_items"
    _expect_exact_keys(  # shape-check: fatal:no-conservative-reading
        payload,
        context="discuss_answer",
        required={"schema_version", "kind", "position", "rationale", "confidence", item_key},
        optional={"answer", "rebuttal", "analyzer_framing", "framing_note", "research", "evidence"},
    )
    position = _expect_non_empty_string(payload["position"], context="discuss_answer.position")  # shape-check: fatal:no-conservative-reading
    if position not in DISCUSS_ANSWER_POSITION_VALUES:
        raise AgentLoopError("discuss_answer.position must be `answer` or `needs-human`.")  # shape-check: fatal:no-conservative-reading
    rationale = _expect_non_empty_string(payload["rationale"], context="discuss_answer.rationale")  # shape-check: fatal:no-conservative-reading
    confidence = _expect_non_empty_string(payload["confidence"], context="discuss_answer.confidence")  # shape-check: fatal:no-conservative-reading
    if confidence not in DISCUSS_ANSWER_CONFIDENCE_VALUES:
        raise AgentLoopError("discuss_answer.confidence must be one of: low, medium, high.")  # shape-check: fatal:no-conservative-reading
    if allow_legacy_open_questions:
        legacy_questions = _expect_string_list(  # shape-check: fatal:no-conservative-reading
            payload["open_questions"], context="discuss_answer.open_questions",
            item_context="discuss_answer.open_questions",
        )
        legacy_status = "human-decision" if position == "needs-human" else "blocker"
        unresolved_items = tuple(
            DiscussUnresolvedItem(status=legacy_status, text=question)
            for question in legacy_questions
        )
    else:
        unresolved_items = _parse_discuss_unresolved_items(  # shape-check: fatal:no-conservative-reading
            payload["unresolved_items"], context="discuss_answer.unresolved_items"
        )
    answer = None
    if "answer" in payload:
        answer = _expect_non_empty_string(payload["answer"], context="discuss_answer.answer")  # shape-check: fatal:no-conservative-reading
    if position == "answer" and answer is None:
        raise AgentLoopError("discuss_answer.answer is required when position is `answer`.")  # shape-check: fatal:no-conservative-reading
    if position == "needs-human":
        if answer is not None:
            raise AgentLoopError("discuss_answer.answer must be omitted when position is `needs-human`.")  # shape-check: fatal:no-conservative-reading
        if not any(item.status == "human-decision" for item in unresolved_items):
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                "discuss_answer.unresolved_items must include a human-decision for `needs-human`."
            )
    rebuttal = None
    if "rebuttal" in payload:
        rebuttal = _expect_non_empty_string(payload["rebuttal"], context="discuss_answer.rebuttal")  # shape-check: fatal:no-conservative-reading
    if round_number > 1 and rebuttal is None:
        raise AgentLoopError("discuss_answer.rebuttal is required for debate rounds.")  # shape-check: fatal:no-conservative-reading
    analyzer_framing = payload.get("analyzer_framing")
    if analyzer_framing is not None:
        analyzer_framing = _expect_non_empty_string(analyzer_framing, context="discuss_answer.analyzer_framing")  # shape-check: fatal:no-conservative-reading
        if analyzer_framing not in DISCUSS_ANALYZER_FRAMING_VALUES:
            raise AgentLoopError("discuss_answer.analyzer_framing must be `accurate` or `misframed`.")  # shape-check: fatal:no-conservative-reading
    framing_note = payload.get("framing_note")
    if framing_note is not None:
        framing_note = _expect_non_empty_string(framing_note, context="discuss_answer.framing_note")  # shape-check: fatal:no-conservative-reading
    if analyzer_framing == "misframed" and framing_note is None:
        raise AgentLoopError("discuss_answer.framing_note is required when analyzer_framing is `misframed`.")  # shape-check: fatal:no-conservative-reading
    if framing_note is not None and analyzer_framing is None:
        raise AgentLoopError("discuss_answer.framing_note requires analyzer_framing to be set.")  # shape-check: fatal:no-conservative-reading
    research_status, sourced_facts = (None, ())
    research_target: str | None = None
    research_questions: tuple[str, ...] = ()
    if "research" in payload:
        research_status, sourced_facts, research_target, research_questions = _parse_discuss_research(payload["research"])  # shape-check: fatal:no-conservative-reading
    evidence_claims, evidence_updates = ((), ())
    if "evidence" in payload:
        evidence_claims, evidence_updates = _parse_discuss_evidence(payload["evidence"])  # shape-check: fatal:no-conservative-reading
    if research_mode == "required" and (research_status is None or research_status == "not-needed"):
        raise AgentLoopError("discuss_answer.research with sourced, unavailable, or inconclusive status is required.")  # shape-check: fatal:no-conservative-reading
    return ParsedDiscussAnswer(
        position=position, rationale=rationale, confidence=confidence,
        unresolved_items=unresolved_items, reviewer=reviewer, answer=answer,
        rebuttal=rebuttal, analyzer_framing=analyzer_framing, framing_note=framing_note,
        research_status=research_status, sourced_facts=sourced_facts,
        research_target=research_target, research_questions=research_questions,
        evidence_claims=evidence_claims, evidence_updates=evidence_updates,
    )

def validate_structured_discuss_answer(
    text: str, *, reviewer: str, round_number: int = 1, research_mode: str | None = None
) -> ParsedDiscussAnswer:
    parsed = parse_structured_discuss_answer(text, reviewer=reviewer, round_number=round_number, research_mode=research_mode)  # shape-check: fatal:no-conservative-reading
    if parsed is None:
        raise AgentLoopError("Discuss answer did not use the required structured format.")  # shape-check: fatal:no-conservative-reading
    return parsed

def validate_structured_discuss_evidence_reconciliation(
    text: str, *, observation_ids: Sequence[str], observation_statuses: dict[str, str],
) -> ParsedDiscussEvidenceReconciliation:
    payload = _extract_structured_discuss_review_payload(text, context_label="Evidence reconciliation")  # shape-check: fatal:unparseable-envelope
    if payload is None:
        raise AgentLoopError("Evidence reconciliation did not use the required structured format.")  # shape-check: fatal:no-conservative-reading
    _require_supported_schema_version(payload)  # shape-check: fatal:kind-or-version-mismatch
    _expect_exact_keys(payload, context="discuss_evidence_reconciliation", required={"schema_version", "kind", "groups"})  # shape-check: fatal:unparseable-envelope
    if payload.get("kind") != "discuss_evidence_reconciliation" or not isinstance(payload["groups"], list):
        raise AgentLoopError("Expected discuss_evidence_reconciliation groups.")  # shape-check: fatal:no-conservative-reading
    known = set(observation_ids)
    used: set[str] = set()
    groups: list[tuple[str, ...]] = []
    for index, value in enumerate(payload["groups"]):
        if not isinstance(value, list) or len(value) < 2 or not all(isinstance(item, str) and item for item in value):
            raise AgentLoopError(f"evidence reconciliation group {index} must contain at least two observation IDs.")  # shape-check: fatal:no-conservative-reading
        group = tuple(value)
        if any(item not in known for item in group) or any(item in used for item in group):
            raise AgentLoopError("evidence reconciliation groups must use each supplied ID at most once.")  # shape-check: fatal:payload-bound
        if len({observation_statuses[item] for item in group}) != 1:
            raise AgentLoopError("evidence reconciliation can group only compatible active statuses.")  # shape-check: fatal:no-conservative-reading
        used.update(group)
        groups.append(group)
    return ParsedDiscussEvidenceReconciliation(groups=tuple(groups))

DISCUSS_SEMANTIC_CLASSIFICATIONS = frozenset({
    "equivalent", "compatible_with_residual_decisions", "material_conflict"
})

def validate_structured_discuss_semantic_comparison(
    text: str, *, reviewers: Sequence[str]
) -> ParsedDiscussSemanticComparison:
    payload = _extract_structured_discuss_review_payload(text, context_label="Semantic comparison")  # shape-check: fatal:unparseable-envelope
    if payload is None:
        raise AgentLoopError("Semantic comparison did not use the required structured format.")  # shape-check: fatal:no-conservative-reading
    _require_supported_schema_version(payload)  # shape-check: fatal:kind-or-version-mismatch
    _expect_exact_keys(payload, context="discuss_semantic_comparison",  # shape-check: fatal:unparseable-envelope
        required={"schema_version", "kind", "classification", "shared_recommendation", "remaining_decisions", "evidence"})
    if payload.get("kind") != "discuss_semantic_comparison":
        raise AgentLoopError("Structured response kind mismatch: expected `discuss_semantic_comparison`.")  # shape-check: fatal:unparseable-envelope
    classification = _expect_non_empty_string(payload["classification"], context="discuss_semantic_comparison.classification")  # shape-check: fatal:no-conservative-reading
    if classification not in DISCUSS_SEMANTIC_CLASSIFICATIONS:
        raise AgentLoopError("Unsupported semantic comparison classification.")  # shape-check: fatal:no-conservative-reading
    shared = _expect_non_empty_string(payload["shared_recommendation"], context="discuss_semantic_comparison.shared_recommendation")  # shape-check: fatal:no-conservative-reading
    decisions = _expect_string_list(payload["remaining_decisions"], context="discuss_semantic_comparison.remaining_decisions", item_context="discuss_semantic_comparison.remaining_decisions")  # shape-check: fatal:no-conservative-reading
    if len(set(item.casefold() for item in decisions)) != len(decisions):
        raise AgentLoopError("Semantic comparison remaining_decisions must be deduplicated.")  # shape-check: fatal:no-conservative-reading
    if classification == "equivalent" and decisions:
        raise AgentLoopError("Equivalent semantic comparisons cannot have remaining decisions.")  # shape-check: fatal:no-conservative-reading
    if classification == "compatible_with_residual_decisions" and not decisions:
        raise AgentLoopError("Compatible semantic comparisons require remaining decisions.")  # shape-check: fatal:no-conservative-reading
    evidence_payload = payload["evidence"]
    if not isinstance(evidence_payload, list) or not evidence_payload:
        raise AgentLoopError("Semantic comparison evidence must be a non-empty list.")  # shape-check: fatal:no-conservative-reading
    evidence: list[DiscussSemanticEvidence] = []
    for index, item in enumerate(evidence_payload):
        item_payload = _expect_object(item, context=f"discuss_semantic_comparison.evidence[{index}]")  # shape-check: fatal:no-conservative-reading
        _expect_exact_keys(item_payload, context=f"discuss_semantic_comparison.evidence[{index}]", required={"reviewer", "supports"})  # shape-check: fatal:no-conservative-reading
        evidence.append(DiscussSemanticEvidence(
            reviewer=_expect_non_empty_string(item_payload["reviewer"], context="semantic evidence reviewer"),  # shape-check: fatal:no-conservative-reading
            supports=_expect_non_empty_string(item_payload["supports"], context="semantic evidence supports"),  # shape-check: fatal:no-conservative-reading
        ))
    expected = set(reviewers)
    actual = {item.reviewer for item in evidence}
    if actual != expected or len(evidence) != len(expected):
        raise AgentLoopError("Semantic comparison evidence must cover exactly every final-round reviewer.")  # shape-check: fatal:no-conservative-reading
    return ParsedDiscussSemanticComparison(classification, shared, decisions, tuple(evidence))

def validate_structured_discuss_answer_confirmation(text: str, *, reviewer: str) -> ParsedDiscussAnswerConfirmation:
    payload = _extract_structured_discuss_review_payload(text, context_label="Answer confirmation")  # shape-check: fatal:unparseable-envelope
    if payload is None:
        raise AgentLoopError("Answer confirmation did not use the required structured format.")  # shape-check: fatal:no-conservative-reading
    _require_supported_schema_version(payload)  # shape-check: fatal:kind-or-version-mismatch
    _expect_exact_keys(payload, context="discuss_answer_confirmation",  # shape-check: fatal:unparseable-envelope
        required={"schema_version", "kind", "decision", "rationale", "reviewer"}, optional={"answer"})
    if payload.get("kind") != "discuss_answer_confirmation":
        raise AgentLoopError("Structured response kind mismatch: expected `discuss_answer_confirmation`.")  # shape-check: fatal:unparseable-envelope
    response_reviewer = _expect_non_empty_string(payload["reviewer"], context="discuss_answer_confirmation.reviewer")  # shape-check: fatal:no-conservative-reading
    if response_reviewer != reviewer:
        raise AgentLoopError("Answer confirmation reviewer does not match the debater.")  # shape-check: fatal:no-conservative-reading
    decision = _expect_non_empty_string(payload["decision"], context="discuss_answer_confirmation.decision")  # shape-check: fatal:no-conservative-reading
    if decision not in {"confirm", "refine"}:
        raise AgentLoopError("Answer confirmation decision must be `confirm` or `refine`.")  # shape-check: fatal:no-conservative-reading
    answer = payload.get("answer")
    if decision == "confirm" and answer is not None:
        raise AgentLoopError("Confirm responses must not include an answer.")  # shape-check: fatal:no-conservative-reading
    if decision == "refine":
        answer = _expect_non_empty_string(answer, context="discuss_answer_confirmation.answer")  # shape-check: fatal:no-conservative-reading
    return ParsedDiscussAnswerConfirmation(response_reviewer, decision,
        _expect_non_empty_string(payload["rationale"], context="discuss_answer_confirmation.rationale"), answer)  # shape-check: fatal:no-conservative-reading

def _parse_discuss_agenda_disagreement(
    value: object, *, context: str
) -> DiscussAgendaDisagreement:
    payload = _expect_object(value, context=context)  # shape-check: fatal:no-conservative-reading
    _expect_exact_keys(  # shape-check: fatal:no-conservative-reading
        payload,
        context=context,
        required={"topic", "positions", "question_for_next_round"},
    )
    positions_payload = _expect_object(payload["positions"], context=f"{context}.positions")  # shape-check: fatal:no-conservative-reading
    if not positions_payload:
        raise AgentLoopError(f"{context}.positions must not be empty.")  # shape-check: fatal:no-conservative-reading
    positions = tuple(
        (
            _expect_non_empty_string(name, context=f"{context}.positions key"),  # shape-check: fatal:no-conservative-reading
            _expect_non_empty_string(position, context=f"{context}.positions[{name!r}]"),  # shape-check: fatal:no-conservative-reading
        )
        for name, position in positions_payload.items()
    )
    return DiscussAgendaDisagreement(
        topic=_expect_non_empty_string(payload["topic"], context=f"{context}.topic"),  # shape-check: fatal:no-conservative-reading
        positions=positions,
        question_for_next_round=_expect_non_empty_string(  # shape-check: fatal:no-conservative-reading
            payload["question_for_next_round"],
            context=f"{context}.question_for_next_round",
        ),
    )

def _bounded_discuss_synthesis_text(value: object, *, context: str) -> str:
    text = _expect_non_empty_string(value, context=context)  # shape-check: fatal:no-conservative-reading
    if len(text.encode("utf-8")) > DISCUSS_SYNTHESIS_MAX_TEXT_BYTES:
        raise AgentLoopError(  # shape-check: fatal:payload-bound
            f"{context} exceeds {DISCUSS_SYNTHESIS_MAX_TEXT_BYTES} UTF-8 bytes."
        )
    return text

def _bounded_discuss_synthesis_list(
    value: object,
    *,
    context: str,
    item_context: str,
    maximum: int = DISCUSS_SYNTHESIS_MAX_ENTRIES,
) -> tuple[str, ...]:
    values = _expect_string_list(value, context=context, item_context=item_context)  # shape-check: fatal:no-conservative-reading
    if len(values) > maximum:
        raise AgentLoopError(f"{context} may contain at most {maximum} item(s).")  # shape-check: fatal:payload-bound
    bounded = tuple(
        _bounded_discuss_synthesis_text(item, context=f"{item_context} at index {index}")  # shape-check: fatal:no-conservative-reading
        for index, item in enumerate(values)
    )
    if len({item.casefold() for item in bounded}) != len(bounded):
        raise AgentLoopError(f"{context} must not contain duplicate entries.")  # shape-check: fatal:no-conservative-reading
    return bounded

def _parse_discuss_synthesis_reference(
    value: object, *, context: str
) -> DiscussSynthesisResponseReference:
    payload = _expect_object(value, context=context)  # shape-check: fatal:no-conservative-reading
    _expect_exact_keys(payload, context=context, required={"reviewer", "round"})  # shape-check: fatal:no-conservative-reading
    reviewer = _bounded_discuss_synthesis_text(  # shape-check: fatal:no-conservative-reading
        payload["reviewer"], context=f"{context}.reviewer"
    )
    round_number = _expect_int(payload["round"], context=f"{context}.round")  # shape-check: fatal:no-conservative-reading
    if round_number < 1:
        raise AgentLoopError(f"{context}.round must be at least 1.")  # shape-check: fatal:no-conservative-reading
    return DiscussSynthesisResponseReference(reviewer=reviewer, round=round_number)

def _parse_discuss_synthesis_references(
    value: object, *, context: str
) -> tuple[DiscussSynthesisResponseReference, ...]:
    if not isinstance(value, list):
        raise AgentLoopError(f"{context} must be a JSON array.")  # shape-check: fatal:no-conservative-reading
    if not value:
        raise AgentLoopError(f"{context} must not be empty.")  # shape-check: fatal:no-conservative-reading
    if len(value) > DISCUSS_SYNTHESIS_MAX_ENTRIES:
        raise AgentLoopError(  # shape-check: fatal:payload-bound
            f"{context} may contain at most {DISCUSS_SYNTHESIS_MAX_ENTRIES} item(s)."
        )
    references = tuple(
        _parse_discuss_synthesis_reference(item, context=f"{context}[{index}]")  # shape-check: fatal:no-conservative-reading
        for index, item in enumerate(value)
    )
    keys = [(item.reviewer.casefold(), item.round) for item in references]
    if len(set(keys)) != len(keys):
        raise AgentLoopError(f"{context} must not contain duplicate references.")  # shape-check: fatal:no-conservative-reading
    return references

def _parse_discuss_synthesis_consensus(
    value: object, *, context: str
) -> DiscussSynthesisConsensus:
    payload = _expect_object(value, context=context)  # shape-check: fatal:no-conservative-reading
    _expect_exact_keys(payload, context=context, required={"text", "references"})  # shape-check: fatal:no-conservative-reading
    return DiscussSynthesisConsensus(
        text=_bounded_discuss_synthesis_text(payload["text"], context=f"{context}.text"),  # shape-check: fatal:no-conservative-reading
        references=_parse_discuss_synthesis_references(  # shape-check: fatal:no-conservative-reading
            payload["references"], context=f"{context}.references"
        ),
    )

def _parse_discuss_synthesis_position(
    value: object, *, context: str
) -> DiscussSynthesisPosition:
    payload = _expect_object(value, context=context)  # shape-check: fatal:no-conservative-reading
    _expect_exact_keys(payload, context=context, required={"reviewers", "position"})  # shape-check: fatal:no-conservative-reading
    reviewers = _bounded_discuss_synthesis_list(  # shape-check: fatal:no-conservative-reading
        payload["reviewers"],
        context=f"{context}.reviewers",
        item_context=f"{context}.reviewers",
        maximum=DISCUSS_SYNTHESIS_MAX_ENTRIES,
    )
    return DiscussSynthesisPosition(
        reviewers=reviewers,
        position=_bounded_discuss_synthesis_text(  # shape-check: fatal:no-conservative-reading
            payload["position"], context=f"{context}.position"
        ),
    )

def _parse_discuss_synthesis_disagreement(
    value: object, *, context: str
) -> DiscussSynthesisDisagreement:
    payload = _expect_object(value, context=context)  # shape-check: fatal:no-conservative-reading
    _expect_exact_keys(  # shape-check: fatal:no-conservative-reading
        payload, context=context, required={"topic", "positions", "decision_needed"}
    )
    positions_value = payload["positions"]
    if not isinstance(positions_value, list):
        raise AgentLoopError(f"{context}.positions must be a JSON array.")  # shape-check: fatal:no-conservative-reading
    if not positions_value:
        raise AgentLoopError(f"{context}.positions must not be empty.")  # shape-check: fatal:no-conservative-reading
    if len(positions_value) > DISCUSS_SYNTHESIS_MAX_ENTRIES:
        raise AgentLoopError(  # shape-check: fatal:payload-bound
            f"{context}.positions may contain at most {DISCUSS_SYNTHESIS_MAX_ENTRIES} item(s)."
        )
    positions = tuple(
        _parse_discuss_synthesis_position(item, context=f"{context}.positions[{index}]")  # shape-check: fatal:no-conservative-reading
        for index, item in enumerate(positions_value)
    )
    all_reviewers = [reviewer for item in positions for reviewer in item.reviewers]
    if len({reviewer.casefold() for reviewer in all_reviewers}) != len(all_reviewers):
        raise AgentLoopError(f"{context}.positions must not repeat a reviewer.")  # shape-check: fatal:no-conservative-reading
    return DiscussSynthesisDisagreement(
        topic=_bounded_discuss_synthesis_text(payload["topic"], context=f"{context}.topic"),  # shape-check: fatal:no-conservative-reading
        positions=positions,
        decision_needed=_bounded_discuss_synthesis_text(  # shape-check: fatal:no-conservative-reading
            payload["decision_needed"], context=f"{context}.decision_needed"
        ),
    )

def _parse_discuss_synthesis_change(
    value: object, *, context: str
) -> DiscussSynthesisChange:
    payload = _expect_object(value, context=context)  # shape-check: fatal:no-conservative-reading
    _expect_exact_keys(  # shape-check: fatal:no-conservative-reading
        payload, context=context, required={"kind", "topic", "text", "references"}
    )
    kind = _bounded_discuss_synthesis_text(payload["kind"], context=f"{context}.kind")  # shape-check: fatal:no-conservative-reading
    if kind not in DISCUSS_SYNTHESIS_CHANGE_KINDS:
        raise AgentLoopError(  # shape-check: fatal:kind-or-version-mismatch
            f"{context}.kind must be one of: "
            + ", ".join(sorted(DISCUSS_SYNTHESIS_CHANGE_KINDS))
        )
    return DiscussSynthesisChange(
        kind=kind,
        topic=_bounded_discuss_synthesis_text(payload["topic"], context=f"{context}.topic"),  # shape-check: fatal:no-conservative-reading
        text=_bounded_discuss_synthesis_text(payload["text"], context=f"{context}.text"),  # shape-check: fatal:no-conservative-reading
        references=_parse_discuss_synthesis_references(  # shape-check: fatal:no-conservative-reading
            payload["references"], context=f"{context}.references"
        ),
    )

def _parse_round_synthesis_payload(
    payload: dict[str, object], *, context: str = "discuss_round_synthesis"
) -> ParsedDiscussRoundSynthesis:
    _require_supported_schema_version(payload)  # shape-check: fatal:kind-or-version-mismatch
    _expect_exact_keys(  # shape-check: fatal:no-conservative-reading
        payload,
        context=context,
        required={
            "schema_version", "kind", "consensus", "disagreements", "changes",
            "missing_facts", "next_round_focus", "responding_reviewers",
        },
    )
    if payload.get("kind") != "discuss_round_synthesis":
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            "Structured response kind mismatch: expected `discuss_round_synthesis`."
        )
    consensus = tuple(
        _parse_discuss_synthesis_consensus(item, context=f"{context}.consensus[{index}]")  # shape-check: fatal:no-conservative-reading
        for index, item in enumerate(
            _bounded_synthesis_object_list(payload["consensus"], context=f"{context}.consensus")  # shape-check: fatal:no-conservative-reading
        )
    )
    disagreements = tuple(
        _parse_discuss_synthesis_disagreement(  # shape-check: fatal:no-conservative-reading
            item, context=f"{context}.disagreements[{index}]"
        )
        for index, item in enumerate(
            _bounded_synthesis_object_list(payload["disagreements"], context=f"{context}.disagreements")  # shape-check: fatal:no-conservative-reading
        )
    )
    changes = tuple(
        _parse_discuss_synthesis_change(item, context=f"{context}.changes[{index}]")  # shape-check: fatal:no-conservative-reading
        for index, item in enumerate(
            _bounded_synthesis_object_list(payload["changes"], context=f"{context}.changes")  # shape-check: fatal:no-conservative-reading
        )
    )
    topics = [item.topic.casefold() for item in disagreements]
    if len(set(topics)) != len(topics):
        raise AgentLoopError(f"{context}.disagreements must not contain duplicate topics.")  # shape-check: fatal:no-conservative-reading
    missing_facts = _bounded_discuss_synthesis_list(  # shape-check: fatal:no-conservative-reading
        payload["missing_facts"], context=f"{context}.missing_facts", item_context=f"{context}.missing_facts"
    )
    next_focus = _bounded_discuss_synthesis_list(  # shape-check: fatal:no-conservative-reading
        payload["next_round_focus"], context=f"{context}.next_round_focus", item_context=f"{context}.next_round_focus",
        maximum=DISCUSS_SYNTHESIS_MAX_NEXT_ROUND_FOCUS,
    )
    responding_reviewers = _bounded_discuss_synthesis_list(  # shape-check: fatal:no-conservative-reading
        payload["responding_reviewers"], context=f"{context}.responding_reviewers", item_context=f"{context}.responding_reviewers"
    )
    return ParsedDiscussRoundSynthesis(
        consensus=consensus, disagreements=disagreements, changes=changes,
        missing_facts=missing_facts, next_round_focus=next_focus,
        responding_reviewers=responding_reviewers,
    )

def _bounded_synthesis_object_list(value: object, *, context: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise AgentLoopError(f"{context} must be a JSON array.")  # shape-check: fatal:no-conservative-reading
    if len(value) > DISCUSS_SYNTHESIS_MAX_ENTRIES:
        raise AgentLoopError(  # shape-check: fatal:payload-bound
            f"{context} may contain at most {DISCUSS_SYNTHESIS_MAX_ENTRIES} item(s)."
        )
    return [_expect_object(item, context=f"{context}[{index}]") for index, item in enumerate(value)]  # shape-check: fatal:no-conservative-reading

def _parse_final_synthesis_payload(
    payload: dict[str, object], *, context: str = "discuss_final_synthesis"
) -> ParsedDiscussFinalSynthesis:
    _require_supported_schema_version(payload)  # shape-check: fatal:kind-or-version-mismatch
    _expect_exact_keys(  # shape-check: fatal:no-conservative-reading
        payload,
        context=context,
        required={
            "schema_version", "kind", "classification", "agreed_conclusions",
            "remaining_disagreements", "next_action",
        },
    )
    if payload.get("kind") != "discuss_final_synthesis":
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            "Structured response kind mismatch: expected `discuss_final_synthesis`."
        )
    classification = _bounded_discuss_synthesis_text(  # shape-check: fatal:no-conservative-reading
        payload["classification"], context=f"{context}.classification"
    )
    if classification not in DISCUSS_SYNTHESIS_CLASSIFICATIONS:
        raise AgentLoopError(f"{context}.classification is not supported.")  # shape-check: fatal:no-conservative-reading
    agreed = tuple(
        _parse_discuss_synthesis_consensus(item, context=f"{context}.agreed_conclusions[{index}]")  # shape-check: fatal:no-conservative-reading
        for index, item in enumerate(
            _bounded_synthesis_object_list(payload["agreed_conclusions"], context=f"{context}.agreed_conclusions")  # shape-check: fatal:no-conservative-reading
        )
    )
    disagreements = tuple(
        _parse_discuss_synthesis_disagreement(  # shape-check: fatal:no-conservative-reading
            item, context=f"{context}.remaining_disagreements[{index}]"
        )
        for index, item in enumerate(
            _bounded_synthesis_object_list(payload["remaining_disagreements"], context=f"{context}.remaining_disagreements")  # shape-check: fatal:no-conservative-reading
        )
    )
    topics = [item.topic.casefold() for item in disagreements]
    if len(set(topics)) != len(topics):
        raise AgentLoopError(f"{context}.remaining_disagreements must not contain duplicate topics.")  # shape-check: fatal:no-conservative-reading
    if classification == "consensus" and disagreements:
        raise AgentLoopError("A consensus final synthesis cannot contain remaining disagreements.")  # shape-check: fatal:no-conservative-reading
    if classification == "consensus" and not agreed:
        raise AgentLoopError("A consensus final synthesis must include an agreed conclusion.")  # shape-check: fatal:no-conservative-reading
    if classification == "near_consensus" and not disagreements:
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            "A near-consensus final synthesis must include a remaining disagreement."
        )
    if classification == "material_deadlock" and not disagreements and not agreed:
        raise AgentLoopError("A material-deadlock final synthesis must describe the residual state.")  # shape-check: fatal:no-conservative-reading
    return ParsedDiscussFinalSynthesis(
        classification=classification,
        agreed_conclusions=agreed,
        remaining_disagreements=disagreements,
        next_action=_bounded_discuss_synthesis_text(  # shape-check: fatal:no-conservative-reading
            payload["next_action"], context=f"{context}.next_action"
        ),
    )

def _disc_synthesis_reference_payload(
    reference: DiscussSynthesisResponseReference,
) -> dict[str, object]:
    return {"reviewer": reference.reviewer, "round": reference.round}

def _disc_synthesis_consensus_payload(item: DiscussSynthesisConsensus) -> dict[str, object]:
    return {
        "text": item.text,
        "references": [_disc_synthesis_reference_payload(ref) for ref in item.references],
    }

def _disc_synthesis_disagreement_payload(item: DiscussSynthesisDisagreement) -> dict[str, object]:
    return {
        "topic": item.topic,
        "positions": [
            {"reviewers": list(position.reviewers), "position": position.position}
            for position in item.positions
        ],
        "decision_needed": item.decision_needed,
    }

def _disc_synthesis_change_payload(item: DiscussSynthesisChange) -> dict[str, object]:
    return {
        "kind": item.kind,
        "topic": item.topic,
        "text": item.text,
        "references": [_disc_synthesis_reference_payload(ref) for ref in item.references],
    }

def discuss_round_synthesis_payload(synthesis: ParsedDiscussRoundSynthesis) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "discuss_round_synthesis",
        "consensus": [_disc_synthesis_consensus_payload(item) for item in synthesis.consensus],
        "disagreements": [_disc_synthesis_disagreement_payload(item) for item in synthesis.disagreements],
        "changes": [_disc_synthesis_change_payload(item) for item in synthesis.changes],
        "missing_facts": list(synthesis.missing_facts),
        "next_round_focus": list(synthesis.next_round_focus),
        "responding_reviewers": list(synthesis.responding_reviewers),
    }

def discuss_final_synthesis_payload(synthesis: ParsedDiscussFinalSynthesis) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "discuss_final_synthesis",
        "classification": synthesis.classification,
        "agreed_conclusions": [_disc_synthesis_consensus_payload(item) for item in synthesis.agreed_conclusions],
        "remaining_disagreements": [_disc_synthesis_disagreement_payload(item) for item in synthesis.remaining_disagreements],
        "next_action": synthesis.next_action,
    }

def serialize_discuss_round_synthesis(synthesis: ParsedDiscussRoundSynthesis) -> str:
    payload = discuss_round_synthesis_payload(synthesis)
    _parse_round_synthesis_payload(payload)  # shape-check: fatal:orchestrator-authored
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(serialized.encode("utf-8")) > DISCUSS_SYNTHESIS_MAX_CANONICAL_BYTES:
        raise AgentLoopError("Canonical discuss round synthesis exceeds 16,000 UTF-8 bytes.")  # shape-check: fatal:orchestrator-authored
    return serialized

def serialize_discuss_final_synthesis(synthesis: ParsedDiscussFinalSynthesis) -> str:
    payload = discuss_final_synthesis_payload(synthesis)
    _parse_final_synthesis_payload(payload)  # shape-check: fatal:orchestrator-authored
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(serialized.encode("utf-8")) > DISCUSS_SYNTHESIS_MAX_CANONICAL_BYTES:
        raise AgentLoopError("Canonical discuss final synthesis exceeds 16,000 UTF-8 bytes.")  # shape-check: fatal:orchestrator-authored
    return serialized

def _parse_canonical_discuss_synthesis(text: str, *, kind: str) -> dict[str, object] | None:
    if not isinstance(text, str) or len(text.encode("utf-8")) > DISCUSS_SYNTHESIS_MAX_CANONICAL_BYTES:
        return None
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("kind") != kind:
        return None
    return value

def parse_structured_discuss_round_synthesis(text: str) -> ParsedDiscussRoundSynthesis | None:
    payload = _extract_structured_discuss_review_payload(  # shape-check: fatal:unparseable-envelope
        text, context_label="Structured discuss round synthesis"
    )
    if payload is None:
        return None
    if payload.get("kind") != "discuss_round_synthesis":
        return None
    return _parse_round_synthesis_payload(payload)  # shape-check: fatal:no-conservative-reading

def validate_structured_discuss_round_synthesis(text: str) -> ParsedDiscussRoundSynthesis:
    parsed = parse_structured_discuss_round_synthesis(text)  # shape-check: fatal:no-conservative-reading
    if parsed is None:
        raise AgentLoopError("Discuss round synthesis did not use the required structured format.")  # shape-check: fatal:no-conservative-reading
    return parsed

def parse_structured_discuss_final_synthesis(text: str) -> ParsedDiscussFinalSynthesis | None:
    payload = _extract_structured_discuss_review_payload(  # shape-check: fatal:unparseable-envelope
        text, context_label="Structured discuss final synthesis"
    )
    if payload is None:
        return None
    if payload.get("kind") != "discuss_final_synthesis":
        return None
    return _parse_final_synthesis_payload(payload)  # shape-check: fatal:no-conservative-reading

def validate_structured_discuss_final_synthesis(text: str) -> ParsedDiscussFinalSynthesis:
    parsed = parse_structured_discuss_final_synthesis(text)  # shape-check: fatal:no-conservative-reading
    if parsed is None:
        raise AgentLoopError("Discuss final synthesis did not use the required structured format.")  # shape-check: fatal:no-conservative-reading
    return parsed

def parse_canonical_discuss_round_synthesis(text: str) -> ParsedDiscussRoundSynthesis | None:
    payload = _parse_canonical_discuss_synthesis(text, kind="discuss_round_synthesis")
    if payload is None:
        return None
    return _parse_round_synthesis_payload(payload)  # shape-check: fatal:orchestrator-authored

def parse_canonical_discuss_final_synthesis(text: str) -> ParsedDiscussFinalSynthesis | None:
    payload = _parse_canonical_discuss_synthesis(text, kind="discuss_final_synthesis")
    if payload is None:
        return None
    return _parse_final_synthesis_payload(payload)  # shape-check: fatal:orchestrator-authored

def parse_structured_discuss_agenda(text: str) -> ParsedDiscussAgenda | None:
    payload = _extract_structured_discuss_agenda_payload(text)  # shape-check: fatal:unparseable-envelope
    if payload is None:
        return None
    _require_supported_schema_version(payload)  # shape-check: fatal:kind-or-version-mismatch
    kind = payload.get("kind")
    if isinstance(kind, str) and kind != "discuss_agenda":
        raise AgentLoopError("Structured response kind mismatch: expected `discuss_agenda`.")  # shape-check: fatal:unparseable-envelope
    _expect_exact_keys(  # shape-check: fatal:unparseable-envelope
        payload,
        context="discuss_agenda",
        required={"schema_version", "kind", "consensus", "disagreements"},
        optional={
            "missing_facts", "research_required", "research_questions",
            "research_question_targets", "round_synthesis",
        },
    )
    consensus = _expect_string_list(  # shape-check: fatal:no-conservative-reading
        payload["consensus"],
        context="discuss_agenda.consensus",
        item_context="discuss_agenda.consensus",
    )
    disagreements_value = payload["disagreements"]
    if not isinstance(disagreements_value, list):
        raise AgentLoopError("discuss_agenda.disagreements must be a JSON array.")  # shape-check: fatal:no-conservative-reading
    disagreements = tuple(
        _parse_discuss_agenda_disagreement(  # shape-check: fatal:no-conservative-reading
            item, context=f"discuss_agenda.disagreements at index {index}"
        )
        for index, item in enumerate(disagreements_value)
    )
    missing_facts = _expect_optional_string_list(  # shape-check: fatal:no-conservative-reading
        payload,
        "missing_facts",
        context="discuss_agenda.missing_facts",
        item_context="discuss_agenda.missing_facts",
    )
    research_required = False
    if "research_required" in payload:
        research_required = _expect_bool(  # shape-check: fatal:no-conservative-reading
            payload["research_required"], context="discuss_agenda.research_required"
        )
    research_questions = _expect_optional_string_list(  # shape-check: fatal:no-conservative-reading
        payload,
        "research_questions",
        context="discuss_agenda.research_questions",
        item_context="discuss_agenda.research_questions",
    )
    research_question_targets = _expect_optional_string_list(  # shape-check: fatal:no-conservative-reading
        payload,
        "research_question_targets",
        context="discuss_agenda.research_question_targets",
        item_context="discuss_agenda.research_question_targets",
    )
    for target in research_question_targets:
        if target not in DISCUSS_RESEARCH_TARGET_VALUES:
            rendered = ", ".join(sorted(DISCUSS_RESEARCH_TARGET_VALUES))
            raise AgentLoopError(f"discuss_agenda.research_question_targets must use only: {rendered}")  # shape-check: fatal:no-conservative-reading
    if research_question_targets and len(research_question_targets) != len(research_questions):
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            "discuss_agenda.research_question_targets must align one-to-one with research_questions."
        )
    if research_required and not research_questions:
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            "discuss_agenda.research_questions must be non-empty when "
            "research_required is true."
        )
    if research_questions and not research_required:
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            "discuss_agenda.research_questions requires research_required to be true."
        )
    round_synthesis = None
    if "round_synthesis" in payload:
        round_synthesis_payload = _expect_object(  # shape-check: fatal:no-conservative-reading
            payload["round_synthesis"], context="discuss_agenda.round_synthesis"
        )
        round_synthesis = _parse_round_synthesis_payload(  # shape-check: fatal:no-conservative-reading
            round_synthesis_payload, context="discuss_agenda.round_synthesis"
        )
    return ParsedDiscussAgenda(
        consensus=consensus,
        disagreements=disagreements,
        missing_facts=missing_facts,
        research_required=research_required,
        research_questions=research_questions,
        research_question_targets=research_question_targets,
        round_synthesis=round_synthesis,
    )

def validate_structured_discuss_agenda(text: str) -> ParsedDiscussAgenda:
    parsed = parse_structured_discuss_agenda(text)  # shape-check: fatal:no-conservative-reading
    if parsed is not None:
        return parsed
    raise AgentLoopError("Discuss agenda did not use the required structured format.")  # shape-check: fatal:no-conservative-reading
