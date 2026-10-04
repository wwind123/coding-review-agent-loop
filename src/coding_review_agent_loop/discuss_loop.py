"""The discuss loop and its analyzers.

Extracted from ``orchestrator.py`` as a pure move (#1181, #1196); the
orchestrator re-exports every name defined here.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace as dataclasses_replace
from pathlib import Path
from .agents.base import AgentName
from .agents.registry import agent_display_name, get_backend
from .workdir_claims import claimed_run
from .config import AgentLoopConfig, ensure_agent_workdirs, resolve_base_branch
from .child_topology import NeedsHumanDecision
from .errors import (
    AgentInvocationError,
    CheckoutVerificationError,
    AgentLoopError,
    QuotaResetExceededError,
)
from .github import (
    strip_bot_login_suffix,
    IssueContext,
    get_issue_context,
    post_issue_comment,
    validate_open_issue,
)
from .split_materialization import (
    DISCUSS_SPLIT_MARKER_RE,
    SPLIT_STAGE_HANDOFF_MARKER_RE,
    UNFILED_SPLIT_WARNING_MARKER_RE,
    dedupe_split_stage_proposals,
    has_unfiled_split_warning,
    materialize_split_proposals,
    post_unfiled_split_warning,
    split_stage_proposal_from_text,
)
from .logging import log
from .evidence_reconciliation import (
    bounded_reconciliation_candidates,
    collect_evidence_observations,
    reconcile_evidence,
)
from .memory import AgentMemoryContext, prepare_agent_memory
from .prompts import (
    build_discuss_agenda_prompt,
    build_discuss_round_synthesis_prompt,
    build_discuss_final_synthesis_prompt,
    build_discuss_final_analysis_prompt,
    build_discuss_evidence_reconciliation_prompt,
    build_discuss_answer_confirmation_prompt,
    build_discuss_semantic_comparison_prompt,
    build_discuss_review_prompt,
)
from .protocol import (
    DISCUSS_RESEARCH_TARGET_VALUES,
    DISCUSS_SYNTHESIS_MAX_ENTRIES,
    DISCUSS_SYNTHESIS_MAX_TEXT_BYTES,
    ParsedDiscussAgenda,
    ParsedDiscussEvidenceReconciliation,
    ParsedDiscussAnswer,
    ParsedDiscussFinalSynthesis,
    ParsedDiscussRoundSynthesis,
    DiscussSynthesisConsensus,
    DiscussSynthesisDisagreement,
    DiscussSynthesisPosition,
    DiscussSynthesisResponseReference,
    DiscussUnresolvedItem,
    ParsedDiscussSemanticComparison,
    ParsedDiscussResponse,
    ParsedDiscussReview,
    failed_discuss_review_category,
    failed_discuss_review_placeholder,
    failed_discuss_answer_placeholder,
    is_failed_discuss_response,
    validate_structured_discuss_agenda,
    parse_structured_discuss_final_synthesis,
    validate_structured_discuss_final_synthesis,
    validate_structured_discuss_round_synthesis,
    validate_structured_discuss_review,
    validate_structured_discuss_answer,
    validate_structured_discuss_answer_confirmation,
    validate_structured_discuss_evidence_reconciliation,
    validate_structured_discuss_semantic_comparison,
    serialize_discuss_round_synthesis,
    serialize_discuss_final_synthesis,
)
from .runner import Runner
from .usage import RunUsageContext
from .workdir_guard import validate_checkout_inspected_evidence
from .comment_rendering import (
    render_discuss_round_summary_comment,
    render_public_agent_comment,
    _render_discuss_agenda_lines,
)
from .round_state import (
    PostedRoundMetadata,
    ROUND_RESUME_MARKER_RE,
    _attach_round_metadata,
    _decode_discuss_vote,
    _extract_round_metadata_records,
    _resume_discuss_round,
)
from .round_transport import decode_mapping, is_round_transport_sidecar
from .agent_failure import ValidatedAgentResponse, _metadata_identity_fields
from .validated_agent import (
    _new_usage_context,
    _begin_run_telemetry,
    _end_run_telemetry,
    _persist_usage_summary,
    _run_validated_agent,
)
from .review_rounds import _ensure_parallel_discuss_workdirs


DISCUSS_CONSENSUS_MARKER_RE = re.compile(
    r"<!--\s*AGENT_DISCUSS_CONSENSUS:\s*([0-9a-f]+)\s*-->",
    re.I,
)


def _is_bot_authored_discuss_comment(body: str) -> bool:
    if is_round_transport_sidecar(body):
        return True
    if DISCUSS_CONSENSUS_MARKER_RE.search(body):
        return True
    # Split-materialization comments (#476) can land on the same issue a
    # discuss run is evaluating (the parent and the discuss subject are the
    # same issue); they must not perturb subject hashing or be forwarded to
    # debaters as fresh human discussion on a later rerun.
    if (
        DISCUSS_SPLIT_MARKER_RE.search(body)
        or UNFILED_SPLIT_WARNING_MARKER_RE.search(body)
        or SPLIT_STAGE_HANDOFF_MARKER_RE.search(body)
    ):
        return True
    match = ROUND_RESUME_MARKER_RE.search(body)
    if match is None:
        return False
    # Read only ``flow``, which never spills: a full decode would reject a
    # round comment whose growth fields are unhydrated spill references.
    try:
        payload = decode_mapping(match.group("payload"))
    except AgentLoopError:
        return False
    return payload.get("flow") == "discuss"


def _discuss_subject(issue_context: IssueContext) -> str:
    text = (issue_context.title or "") + "\n\n" + (issue_context.body or "")
    non_bot_bodies = [
        c.body
        for c in issue_context.comments
        if c.body and not _is_bot_authored_discuss_comment(c.body)
    ]
    if non_bot_bodies:
        text += "\n\n" + "\n\n".join(non_bot_bodies)
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _merge_discuss_split_proposals(votes: Sequence[ParsedDiscussReview]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for vote in votes:
        for proposal in vote.split_proposals:
            if proposal not in seen:
                seen.add(proposal)
                merged.append(proposal)
    return merged


def _detect_discuss_consensus(
    votes: list[ParsedDiscussReview],
) -> tuple[str, list[str]] | None:
    if not votes:
        return None
    outcome = votes[0].outcome
    if any(vote.outcome != outcome for vote in votes):
        return None
    split_proposals = _merge_discuss_split_proposals(votes) if outcome == "split" else []
    return outcome, split_proposals


def _normalize_discuss_answer(answer: str) -> str:
    return " ".join(answer.strip().split()).casefold()


def _detect_discuss_answer_consensus(
    responses: Sequence[ParsedDiscussAnswer], *, partial: bool = False
) -> tuple[str, list[str]] | None:
    if partial or not responses:
        return None
    if _discuss_has_material_items(responses):
        # A final round applies explicit outcome precedence. Before then,
        # classified material prevents normalized-text convergence.
        if not all(response.position == "needs-human" for response in responses):
            return None
    if all(response.position == "needs-human" for response in responses):
        return "needs-human", []
    answers = [response.answer for response in responses]
    if all(answer is not None for answer in answers):
        normalized = {_normalize_discuss_answer(answer or "") for answer in answers}
        if len(normalized) == 1:
            return "answer", []
    return None


def _aggregate_discuss_unresolved_items(
    responses: Sequence[ParsedDiscussAnswer],
) -> tuple[DiscussUnresolvedItem, ...]:
    """Return current-round classified items, deduplicated within a status.

    We deliberately retain identical text under different statuses: a blocker
    must not disappear merely because another debater calls it a follow-up.
    """
    seen: set[tuple[str, str]] = set()
    merged: list[DiscussUnresolvedItem] = []
    for response in responses:
        for item in response.unresolved_items:
            key = (item.status, item.text)
            if key not in seen:
                seen.add(key)
                merged.append(item)
    return tuple(merged)


def _discuss_has_material_items(responses: Sequence[ParsedDiscussAnswer]) -> bool:
    return any(
        item.status in {"blocker", "human-decision"}
        for item in _aggregate_discuss_unresolved_items(responses)
    )


def _final_discuss_answer_item_outcome(
    responses: Sequence[ParsedDiscussAnswer],
) -> str | None:
    statuses = {item.status for item in _aggregate_discuss_unresolved_items(responses)}
    if "human-decision" in statuses:
        return "needs-human"
    if "blocker" in statuses:
        return "deadlock"
    return None


def _handle_discuss_split_outcome(
    runner: Runner,
    *,
    issue_number: int,
    config: AgentLoopConfig,
    subject: str,
    split_proposals: Sequence[str],
    final_votes: Sequence[ParsedDiscussReview],
    issue_comments: Sequence[object],
    post_warning_comment: bool,
) -> NeedsHumanDecision | None:
    """Materialize (or warn about) a discuss `split` consensus's proposed sub-issues (#476).

    Called both when the final summary is freshly posted and on a resumed/
    already-final run, so enabling `--materialize-split-issues` on a rerun
    still files the proposals instead of leaving them stuck in comments.
    Idempotent: `materialize_split_proposals` finds prior children via the
    parent marker and creates nothing when they already cover every proposal.
    """
    if not split_proposals:
        return
    proposals = dedupe_split_stage_proposals(
        [split_stage_proposal_from_text(proposal) for proposal in split_proposals]
    )
    if config.materialize_split_issues:
        rationale = tuple(
            (vote.reviewer, vote.rationale) for vote in final_votes if vote.outcome == "split"
        )
        result = materialize_split_proposals(
            runner,
            config=config,
            parent_issue=issue_number,
            subject=subject,
            proposals=proposals,
            rationale=rationale,
            issue_comments=issue_comments,
        )
        if isinstance(result, NeedsHumanDecision):
            return result
        return
    log(
        config,
        f"discuss: split follow-ups are NOT filed as issues for #{issue_number}; rerun with "
        "--materialize-split-issues or file them manually.",
    )
    if post_warning_comment and not has_unfiled_split_warning(
        issue_comments, issue_number=issue_number, subject=subject
    ):
        post_unfiled_split_warning(
            runner,
            config=config,
            issue_number=issue_number,
            subject=subject,
            proposals=proposals,
        )


def _recover_final_discuss_split_proposals(
    issue_context: IssueContext,
    *,
    subject: str,
    configured_reviewers: Sequence[AgentName],
    reviewer_workdirs: Mapping[str, Path],
) -> tuple[list[str], list[ParsedDiscussReview]] | None:
    """Legacy fallback (#476): reconstruct final-round votes and merged split
    proposals from debater comment metadata when `PostedRoundMetadata` has no
    `split_proposals` recorded on the final summary (comments posted before
    this field existed)."""
    records = _extract_round_metadata_records(issue_context.comments, flow="discuss")
    subject_records = [record for record in records if record.metadata.subject == subject]
    if not subject_records:
        return None
    final_round_number = max(record.metadata.round_number for record in subject_records)
    debater_records = [
        record
        for record in subject_records
        if record.metadata.round_number == final_round_number and record.metadata.role == "debater"
    ]
    if not debater_records:
        return None
    configured_reviewer_names = [agent_display_name(agent) for agent in configured_reviewers]
    by_name = {record.metadata.agent: record for record in debater_records}
    final_votes: list[ParsedDiscussReview] = []
    for name in configured_reviewer_names:
        record = by_name.get(name)
        if record is None:
            continue
        vote = _decode_discuss_vote(
            record,
            round_number=final_round_number,
            reviewer_workdirs=reviewer_workdirs,
        )
        # This is triage-only legacy recovery. A mixed or malformed transcript
        # must not be interpreted as a split consensus.
        if not isinstance(vote, ParsedDiscussReview):
            return None
        final_votes.append(vote)
    if not final_votes:
        return None
    consensus = _detect_discuss_consensus(final_votes)
    if consensus is None or consensus[0] != "split":
        return None
    return consensus[1], final_votes


_DISCUSS_AGENDA_SUPPORT_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "could",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "if",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "so",
        "than",
        "that",
        "the",
        "their",
        "this",
        "to",
        "use",
        "was",
        "we",
        "what",
        "when",
        "whether",
        "which",
        "while",
        "with",
        "would",
        # Low-signal discuss/analyzer boilerplate. These terms are common in
        # agenda summaries and should not validate invented content alone.
        "approach",
        "change",
        "custom",
        "existing",
        "implementation",
        "issue",
        "next",
        "objection",
        "question",
        "round",
        "scope",
        "strategy",
    }
)


def _normalize_discuss_agenda_phrase(text: str) -> str:
    return " ".join(re.findall(r"[A-Za-z0-9_]+", text.lower()))


def _tokenize_discuss_agenda_support(
    text: str,
    *,
    ignored_names: Sequence[str] = (),
) -> tuple[str, ...]:
    ignored_tokens = {
        token
        for name in ignored_names
        for token in re.findall(r"[A-Za-z0-9_]+", name.lower())
    }
    tokens: list[str] = []
    for token in re.findall(r"[A-Za-z0-9_]+", text.lower()):
        if len(token) <= 2 and not token.isdigit():
            continue
        if token in ignored_tokens or token in _DISCUSS_AGENDA_SUPPORT_STOP_WORDS:
            continue
        tokens.append(token)
    return tuple(tokens)


@dataclass(frozen=True)
class _DiscussAgendaSupportCorpus:
    phrase_segments: tuple[str, ...]
    tokens: frozenset[str]


def _build_discuss_agenda_support_corpus(
    *,
    issue_context: IssueContext,
    round_history: Sequence[Sequence[ParsedDiscussResponse]],
    prior_agenda: ParsedDiscussAgenda | None,
    configured_reviewers: Sequence[AgentName],
    analyzer: AgentName,
) -> _DiscussAgendaSupportCorpus:
    segments: list[str] = []
    phrase_segments: list[str] = []

    def add(value: object, *, phrase_support: bool = False) -> None:
        if isinstance(value, str) and value.strip():
            segments.append(value)
            if phrase_support:
                phrase_segments.append(value)

    add(issue_context.title, phrase_support=True)
    add(issue_context.body, phrase_support=True)
    for comment in issue_context.comments:
        # Token parity across transports: REST spells app logins with `[bot]`.
        add(strip_bot_login_suffix(comment.author))
        add(comment.created_at)
        add(comment.body, phrase_support=True)
    for reviewer in configured_reviewers:
        add(agent_display_name(reviewer))
    for round_index, votes in enumerate(round_history, start=1):
        add(f"Round {round_index}")
        for vote in votes:
            add(vote.reviewer)
            if isinstance(vote, ParsedDiscussAnswer):
                add(vote.position)
                add(vote.answer, phrase_support=True)
                add(vote.rationale, phrase_support=True)
                for item in vote.unresolved_items:
                    add(item.status)
                    add(item.text, phrase_support=True)
            elif isinstance(vote, ParsedDiscussReview):
                add(vote.outcome)
                add(vote.rationale, phrase_support=True)
                for proposal in vote.split_proposals:
                    add(proposal, phrase_support=True)
            else:
                add("failed")
                add(vote.category, phrase_support=True)
            add(getattr(vote, "rebuttal", None), phrase_support=True)
            add(getattr(vote, "analyzer_framing", None))
            add(getattr(vote, "framing_note", None), phrase_support=True)
            add(getattr(vote, "research_status", None))
            add(getattr(vote, "research_target", None))
            for question in getattr(vote, "research_questions", ()):
                add(question, phrase_support=True)
            for fact in getattr(vote, "sourced_facts", ()):
                add(fact.fact, phrase_support=True)
                add(fact.source, phrase_support=True)
    if prior_agenda is not None:
        for point in prior_agenda.consensus:
            add(point, phrase_support=True)
        for disagreement in prior_agenda.disagreements:
            add(disagreement.topic, phrase_support=True)
            for name, position in disagreement.positions:
                add(name)
                add(position, phrase_support=True)
            add(disagreement.question_for_next_round, phrase_support=True)
        for fact in prior_agenda.missing_facts:
            add(fact, phrase_support=True)
        for question in prior_agenda.research_questions:
            add(question, phrase_support=True)
        for target in prior_agenda.research_question_targets:
            add(target)

    ignored_names = [
        agent_display_name(analyzer),
        *(agent_display_name(r) for r in configured_reviewers),
    ]
    tokens = frozenset(
        token
        for segment in segments
        for token in _tokenize_discuss_agenda_support(segment, ignored_names=ignored_names)
    )
    return _DiscussAgendaSupportCorpus(
        phrase_segments=tuple(
            normalized
            for segment in phrase_segments
            if (normalized := _normalize_discuss_agenda_phrase(segment))
        ),
        tokens=tokens,
    )


def _discuss_agenda_text_has_support(
    text: str,
    *,
    corpus: _DiscussAgendaSupportCorpus,
    ignored_names: Sequence[str],
) -> bool:
    tokens = _tokenize_discuss_agenda_support(text, ignored_names=ignored_names)
    if tokens:
        supported = sum(1 for token in set(tokens) if token in corpus.tokens)
        required = 1 if len(set(tokens)) <= 4 else 2
        return supported >= required
    normalized = _normalize_discuss_agenda_phrase(text)
    return bool(
        normalized
        and any(
            normalized == segment or normalized in segment
            for segment in corpus.phrase_segments
        )
    )


def _validate_discuss_analyzer_agenda_fidelity(
    agenda: ParsedDiscussAgenda,
    *,
    issue_context: IssueContext,
    round_history: Sequence[Sequence[ParsedDiscussResponse]],
    prior_agenda: ParsedDiscussAgenda | None,
    configured_reviewers: Sequence[AgentName],
    analyzer: AgentName,
) -> None:
    allowed_names = {agent_display_name(reviewer) for reviewer in configured_reviewers}
    targets = agenda.research_question_targets
    if targets and len(targets) != len(agenda.research_questions):
        raise AgentLoopError(
            "analyzer agenda research_question_targets must align one-to-one with research_questions."
        )
    invalid_targets = sorted(set(targets) - DISCUSS_RESEARCH_TARGET_VALUES)
    if invalid_targets:
        raise AgentLoopError(
            "analyzer agenda used invalid research target(s): " + ", ".join(invalid_targets)
        )
    unknown_names = sorted(
        {name for disagreement in agenda.disagreements for name, _position in disagreement.positions}
        - allowed_names
    )
    if unknown_names:
        raise AgentLoopError(
            "analyzer agenda used unknown debater name(s): " + ", ".join(unknown_names)
        )

    ignored_names = [agent_display_name(analyzer), *sorted(allowed_names)]
    corpus = _build_discuss_agenda_support_corpus(
        issue_context=issue_context,
        round_history=round_history,
        prior_agenda=prior_agenda,
        configured_reviewers=configured_reviewers,
        analyzer=analyzer,
    )

    fields: list[tuple[str, str]] = []
    fields.extend(("consensus", point) for point in agenda.consensus)
    for disagreement in agenda.disagreements:
        fields.append(("disagreement topic", disagreement.topic))
        fields.extend(("position", position) for _name, position in disagreement.positions)
        fields.append(("question_for_next_round", disagreement.question_for_next_round))
    fields.extend(("missing_fact", fact) for fact in agenda.missing_facts)
    fields.extend(("research_question", question) for question in agenda.research_questions)

    for field, text in fields:
        if not _discuss_agenda_text_has_support(text, corpus=corpus, ignored_names=ignored_names):
            raise AgentLoopError(f"analyzer agenda {field} lacks transcript support: {text}")


def _disc_synthesis_vote_text(vote: ParsedDiscussResponse) -> str:
    parts = [
        str(getattr(vote, "answer", "") or ""),
        str(getattr(vote, "rationale", "") or ""),
        str(getattr(vote, "rebuttal", "") or ""),
    ]
    parts.extend(
        str(item.text)
        for item in getattr(vote, "unresolved_items", ())
    )
    parts.extend(str(item) for item in getattr(vote, "split_proposals", ()))
    return " ".join(parts)


def _disc_synthesis_text_supported_by_vote(text: str, vote: ParsedDiscussResponse) -> bool:
    """Conservative lexical support check over one current response only."""
    normalized = _normalize_discuss_agenda_phrase(text)
    source = _normalize_discuss_agenda_phrase(_disc_synthesis_vote_text(vote))
    if not normalized or not source:
        return False
    # Keep synthesis fidelity at least as strict as agenda fidelity. In
    # particular, boilerplate such as "that", "must", and "should" is not
    # evidence for an otherwise unsupported claim.
    if not _tokenize_discuss_agenda_support(normalized):
        return False
    corpus = _DiscussAgendaSupportCorpus(
        phrase_segments=(source,),
        tokens=frozenset(_tokenize_discuss_agenda_support(source)),
    )
    return _discuss_agenda_text_has_support(
        normalized, corpus=corpus, ignored_names=()
    )


def _validate_discuss_round_synthesis_fidelity(
    synthesis: ParsedDiscussRoundSynthesis,
    *,
    round_number: int,
    current_votes: Sequence[ParsedDiscussResponse],
    prior_synthesis: ParsedDiscussRoundSynthesis | None,
    configured_reviewers: Sequence[AgentName],
) -> None:
    """Validate cumulative synthesis against current responses and legal state transitions."""
    successful = tuple(vote for vote in current_votes if not is_failed_discuss_response(vote))
    if len(successful) < 2:
        raise AgentLoopError("round synthesis requires at least two successful responders.")
    configured_names = {agent_display_name(agent) for agent in configured_reviewers}
    current_names = {vote.reviewer for vote in successful}
    if synthesis.responding_reviewers and set(synthesis.responding_reviewers) != current_names:
        raise AgentLoopError("round synthesis responding_reviewers must equal current responders.")
    if not synthesis.responding_reviewers:
        raise AgentLoopError("round synthesis responding_reviewers must not be empty.")
    if not current_names <= configured_names:
        raise AgentLoopError("round synthesis contains an unknown current responder.")

    # The current response set is the only source available to the validator;
    # prior references are legal only when they identify an existing carried
    # state, never when they are used as support for a new claim.
    current_by_name = {vote.reviewer: vote for vote in successful}

    def validate_refs(
        references: Sequence[DiscussSynthesisResponseReference], *,
        require_current: bool = False,
    ) -> None:
        if not references:
            raise AgentLoopError("round synthesis statements require response references.")
        for reference in references:
            if reference.reviewer not in configured_names:
                raise AgentLoopError(
                    f"round synthesis references unknown reviewer {reference.reviewer!r}."
                )
            if reference.round > round_number:
                raise AgentLoopError("round synthesis references a future round.")
            if require_current and (
                reference.round != round_number or reference.reviewer not in current_names
            ):
                raise AgentLoopError("new round synthesis claims must reference current responders.")

    prior_consensus = {item.text.casefold(): item for item in (prior_synthesis.consensus if prior_synthesis else ())}
    prior_disagreements = {
        item.topic.casefold(): item for item in (prior_synthesis.disagreements if prior_synthesis else ())
    }
    prior_missing_facts = {
        item.casefold() for item in (prior_synthesis.missing_facts if prior_synthesis else ())
    }
    prior_next_focus = {
        item.casefold() for item in (prior_synthesis.next_round_focus if prior_synthesis else ())
    }
    prior_change_topics = {
        item.topic.casefold() for item in (prior_synthesis.changes if prior_synthesis else ())
    }
    prior_settled_topics = {
        item.text.casefold() for item in (prior_synthesis.consensus if prior_synthesis else ())
    }
    prior_settled_topics.update(
        change.topic.casefold()
        for change in (prior_synthesis.changes if prior_synthesis else ())
        if change.kind in {"resolved", "retracted"}
    )

    consensus_texts: set[str] = set()
    for item in synthesis.consensus:
        key = item.text.casefold()
        if key in consensus_texts:
            raise AgentLoopError("round synthesis consensus must not contain duplicates.")
        consensus_texts.add(key)
        carried = prior_consensus.get(key)
        if carried is not None:
            if item.text != carried.text or item.references != carried.references:
                raise AgentLoopError(
                    "carried round synthesis consensus must preserve its validated text "
                    "and response references"
                )
            validate_refs(item.references)
        else:
            validate_refs(item.references, require_current=True)
            if any(not _disc_synthesis_text_supported_by_vote(item.text, vote) for vote in successful):
                raise AgentLoopError(
                    f"round synthesis consensus lacks independent current support: {item.text}"
                )

    active_topics: set[str] = set()
    for item in synthesis.disagreements:
        topic_key = item.topic.casefold()
        if topic_key in active_topics:
            raise AgentLoopError("round synthesis disagreements must not repeat topics.")
        reopened = any(
            change.kind == "reopened" and change.topic.casefold() == topic_key
            for change in synthesis.changes
        )
        if topic_key in prior_settled_topics and not reopened:
            raise AgentLoopError(
                "round synthesis cannot reactivate a settled topic without a reopened change."
            )
        active_topics.add(topic_key)
        if not any(_disc_synthesis_text_supported_by_vote(item.topic, vote) for vote in successful):
            raise AgentLoopError(f"round synthesis disagreement topic lacks current support: {item.topic}")
        named: set[str] = set()
        for position in item.positions:
            for reviewer in position.reviewers:
                if reviewer in named or reviewer not in current_names:
                    raise AgentLoopError("round synthesis disagreement has unknown or duplicate reviewer.")
                named.add(reviewer)
                if not _disc_synthesis_text_supported_by_vote(position.position, current_by_name[reviewer]):
                    raise AgentLoopError(
                        f"round synthesis position lacks support from {reviewer}: {position.position}"
                    )
        if not _disc_synthesis_text_supported_by_vote(item.decision_needed, successful[0]):
            # A decision question can be phrased across responses; require at
            # least one current response to carry its material vocabulary.
            if not any(_disc_synthesis_text_supported_by_vote(item.decision_needed, vote) for vote in successful):
                raise AgentLoopError(
                    f"round synthesis decision_needed lacks current support: {item.decision_needed}"
                )

    for change in synthesis.changes:
        topic_key = change.topic.casefold()
        validate_refs(change.references, require_current=True)
        if not any(
            _disc_synthesis_text_supported_by_vote(change.text, vote)
            for vote in successful
        ):
            raise AgentLoopError(f"round synthesis change lacks current support: {change.text}")
        if change.kind == "resolved":
            if prior_synthesis is None or topic_key not in prior_disagreements:
                raise AgentLoopError("resolved synthesis changes must match a prior active disagreement.")
            if {ref.reviewer for ref in change.references} != current_names:
                raise AgentLoopError("resolved synthesis changes require every current responder.")
            if topic_key in active_topics:
                raise AgentLoopError("a resolved disagreement cannot remain active in the same snapshot.")
        elif change.kind == "reopened":
            if prior_synthesis is None or (
                topic_key not in prior_consensus
                and topic_key not in prior_disagreements
                and topic_key not in prior_change_topics
            ):
                raise AgentLoopError("reopened synthesis changes must match prior consensus or resolved state.")
            if topic_key not in active_topics:
                raise AgentLoopError("reopened synthesis changes must have an active disagreement.")
        elif change.kind in {"retracted", "refined"}:
            if prior_synthesis is None or (
                topic_key not in prior_consensus
                and topic_key not in prior_disagreements
                and topic_key not in prior_change_topics
            ):
                raise AgentLoopError("synthesis change does not match prior state.")
        elif change.kind == "introduced" and prior_synthesis is not None and (
            topic_key in prior_change_topics
            or topic_key in prior_disagreements
        ):
            raise AgentLoopError("introduced synthesis change repeats prior state.")

    for field_name, values in (
        ("missing_facts", synthesis.missing_facts),
        ("next_round_focus", synthesis.next_round_focus),
    ):
        for value in values:
            if value.casefold() in (
                prior_missing_facts if field_name == "missing_facts" else prior_next_focus
            ):
                continue
            if not any(_disc_synthesis_text_supported_by_vote(value, vote) for vote in successful):
                raise AgentLoopError(f"round synthesis {field_name} lacks current support: {value}")


def _mechanical_discuss_final_classification(
    *, outcome: str, consensus_kind: str | None, votes: Sequence[ParsedDiscussResponse]
) -> str:
    if outcome == "answer" and consensus_kind in {
        "unanimous", "converged", "semantic-equivalent", "debater-confirmed",
    }:
        return "consensus"
    if (
        outcome == "needs-human"
        and consensus_kind in {"unanimous", "converged"}
        and any(
            item.status == "human-decision"
            for vote in votes
            for item in getattr(vote, "unresolved_items", ())
        )
    ):
        return "near_consensus"
    if outcome == "deadlock" or consensus_kind in {
        "deadlock", "material-conflict", "semantic-comparison-failed",
        "confirmation-failed", "confirmation-disagreement",
    }:
        return "material_deadlock"
    raise AgentLoopError(
        f"unsupported mechanical answer classification: outcome={outcome!r}, kind={consensus_kind!r}"
    )


def _validate_discuss_final_synthesis_fidelity(
    synthesis: ParsedDiscussFinalSynthesis,
    *,
    expected_classification: str,
    final_votes: Sequence[ParsedDiscussResponse],
    round_number: int,
    configured_reviewers: Sequence[AgentName],
) -> None:
    if synthesis.classification != expected_classification:
        raise AgentLoopError("final synthesis classification does not match the mechanical result.")
    successful = tuple(vote for vote in final_votes if not is_failed_discuss_response(vote))
    if len(successful) < 2:
        raise AgentLoopError("final synthesis requires at least two successful responders.")
    configured_names = {agent_display_name(agent) for agent in configured_reviewers}
    names = {vote.reviewer for vote in successful}
    if not names <= configured_names:
        raise AgentLoopError("final synthesis contains an unknown responder.")

    def refs_for_current(references: Sequence[DiscussSynthesisResponseReference]) -> None:
        if {ref.reviewer for ref in references} != names or any(
            ref.round != round_number for ref in references
        ):
            raise AgentLoopError("final synthesis references must cover each final responder exactly once.")

    for item in synthesis.agreed_conclusions:
        refs_for_current(item.references)
        if any(not _disc_synthesis_text_supported_by_vote(item.text, vote) for vote in successful):
            raise AgentLoopError(f"final synthesis agreement lacks independent support: {item.text}")
    for disagreement in synthesis.remaining_disagreements:
        if not any(_disc_synthesis_text_supported_by_vote(disagreement.topic, vote) for vote in successful):
            raise AgentLoopError(f"final synthesis topic lacks final-round support: {disagreement.topic}")
        named: set[str] = set()
        for position in disagreement.positions:
            for reviewer in position.reviewers:
                if reviewer in named or reviewer not in names:
                    raise AgentLoopError("final synthesis has an unknown or duplicate reviewer position.")
                named.add(reviewer)
                vote = next(vote for vote in successful if vote.reviewer == reviewer)
                if not _disc_synthesis_text_supported_by_vote(position.position, vote):
                    raise AgentLoopError(
                        f"final synthesis position lacks support from {reviewer}: {position.position}"
                    )
        if not any(
            _disc_synthesis_text_supported_by_vote(disagreement.decision_needed, vote)
            for vote in successful
        ):
            raise AgentLoopError(
                f"final synthesis decision_needed lacks final-round support: {disagreement.decision_needed}"
            )


def _run_discuss_analyzer(
    runner: Runner,
    *,
    issue_number: int,
    config: AgentLoopConfig,
    analyzer: AgentName,
    memory: AgentMemoryContext | None,
    issue_context: IssueContext,
    round_number: int,
    round_history: Sequence[Sequence[ParsedDiscussResponse]],
    prior_agenda: ParsedDiscussAgenda | None,
    prior_round_synthesis: ParsedDiscussRoundSynthesis | None,
    configured_reviewers: Sequence[AgentName],
    usage_context: RunUsageContext,
) -> tuple[ParsedDiscussAgenda | None, str | None, ParsedDiscussRoundSynthesis | None, str | None]:
    """Run the optional discuss analyzer after a non-final round.

    Returns (parsed_agenda, raw_response, synthesis, raw_synthesis_response).
    Analyzer/agenda failures fall back to the mechanical agenda; a nested
    synthesis fidelity failure drops only that advisory extension and keeps
    the validated agenda. Only a quota-reset stop or a checkout-verification refusal propagates because the
    whole run must pause.
    """
    analyzer_name = agent_display_name(analyzer)
    log(
        config,
        f"discuss: invoking analyzer {analyzer_name} on issue #{issue_number} "
        f"(after round {round_number})",
    )
    try:
        response = _run_validated_agent(
            runner,
            agent=analyzer,
            config=config,
            prompt=build_discuss_agenda_prompt(
                issue_number,
                config,
                analyzer=analyzer,
                memory=memory,
                issue_context=issue_context,
                round_number=round_number,
                round_history=round_history,
                prior_agenda=prior_agenda,
                prior_round_synthesis=prior_round_synthesis,
                research_mode=config.discuss_research,
            ),
            marker_description="<!-- AGENT_PLAN_STATE: approved -->",
            validate=validate_structured_discuss_agenda,
            usage_context=usage_context,
            use_repair=True,
            repair_expected_kind="discuss_agenda",
            role="analyzer",
            label=f"discuss-analyzer-r{round_number}",
            operation_description="discuss analyzer",
        )
    except (QuotaResetExceededError, CheckoutVerificationError):
        raise
    except AgentLoopError as exc:
        log(
            config,
            f"discuss: analyzer {analyzer_name} failed ({exc}); falling back to the "
            f"mechanical agenda for round {round_number + 1}",
        )
        return None, None, None, None
    parsed = response.marker_value
    assert isinstance(parsed, ParsedDiscussAgenda)
    try:
        _validate_discuss_analyzer_agenda_fidelity(
            parsed,
            issue_context=issue_context,
            round_history=round_history,
            prior_agenda=prior_agenda,
            configured_reviewers=configured_reviewers,
            analyzer=analyzer,
        )
    except AgentLoopError as exc:
        log(
            config,
            f"discuss: analyzer {analyzer_name} agenda rejected ({exc}); falling back to the "
            f"mechanical agenda for round {round_number + 1}",
        )
        return None, None, None, None
    synthesis = (
        parsed.round_synthesis
        if config.discuss_result_mode == "answer"
        else None
    )
    if synthesis is not None:
        try:
            _validate_discuss_round_synthesis_fidelity(
                synthesis,
                round_number=round_number,
                current_votes=round_history[-1] if round_history else (),
                prior_synthesis=prior_round_synthesis,
                configured_reviewers=configured_reviewers,
            )
        except AgentLoopError as exc:
            log(
                config,
                f"discuss: analyzer {analyzer_name} synthesis rejected ({exc}); "
                "falling back to the mechanical agenda",
            )
            # The synthesis is advisory. Preserve the independently validated
            # agenda and its raw audit response when only the nested extension
            # fails fidelity validation.
            return parsed, response.text, None, None
        return parsed, response.text, synthesis, None

    if config.discuss_result_mode != "answer" or not round_history:
        return parsed, response.text, None, None
    # Legacy agendas are accepted, but answer mode gets one bounded extension
    # call from the same configured analyzer. The ordinary debaters and coder
    # never become synthesis agents implicitly.
    try:
        fallback = _run_validated_agent(
            runner,
            agent=analyzer,
            config=config,
            prompt=build_discuss_round_synthesis_prompt(
                issue_number,
                config,
                analyzer=analyzer,
                round_number=round_number,
                current_responses=round_history[-1],
                prior_synthesis=prior_round_synthesis,
            ),
            marker_description="<!-- AGENT_PLAN_STATE: approved -->",
            validate=validate_structured_discuss_round_synthesis,
            usage_context=usage_context,
            use_repair=True,
            repair_expected_kind="discuss_round_synthesis",
            role="analyzer",
            label=f"discuss-round-synthesis-r{round_number}",
            operation_description="discuss round synthesis",
        )
        fallback_synthesis = fallback.marker_value
        assert isinstance(fallback_synthesis, ParsedDiscussRoundSynthesis)
        _validate_discuss_round_synthesis_fidelity(
            fallback_synthesis,
            round_number=round_number,
            current_votes=round_history[-1],
            prior_synthesis=prior_round_synthesis,
            configured_reviewers=configured_reviewers,
        )
        return parsed, response.text, fallback_synthesis, fallback.text
    except (QuotaResetExceededError, CheckoutVerificationError):
        raise
    except (AgentLoopError, AgentInvocationError) as exc:
        log(
            config,
            f"discuss: round synthesis fallback unavailable ({exc}); retaining the "
            "legacy agenda/mechanical rendering",
        )
        return parsed, response.text, None, None


def _validate_discuss_final_analyzer_fidelity(
    agenda: ParsedDiscussAgenda,
    *,
    final_votes: Sequence[ParsedDiscussResponse],
    configured_reviewers: Sequence[AgentName],
    analyzer: AgentName,
) -> None:
    """Validate an advisory final pass against final debater text only (#529)."""
    if not final_votes or any(is_failed_discuss_response(vote) for vote in final_votes):
        raise AgentLoopError("final analyzer requires successful final-round responses.")
    if agenda.research_required or agenda.research_questions or agenda.research_question_targets:
        raise AgentLoopError("final analyzer output must not include next-round research fields.")
    allowed_names = {agent_display_name(reviewer) for reviewer in configured_reviewers}
    final_names = {vote.reviewer for vote in final_votes}
    unknown_names = sorted(
        {name for disagreement in agenda.disagreements for name, _position in disagreement.positions}
        - allowed_names.intersection(final_names)
    )
    if unknown_names:
        raise AgentLoopError("final analyzer used unknown or absent debater name(s): " + ", ".join(unknown_names))
    empty_context = IssueContext(0, "", None, None, None, ())
    corpus = _build_discuss_agenda_support_corpus(
        issue_context=empty_context,
        round_history=(tuple(final_votes),),
        prior_agenda=None,
        configured_reviewers=configured_reviewers,
        analyzer=analyzer,
    )
    ignored_names = [agent_display_name(analyzer), *sorted(allowed_names)]
    fields: list[tuple[str, str]] = [("consensus", point) for point in agenda.consensus]
    for disagreement in agenda.disagreements:
        fields.append(("disagreement topic", disagreement.topic))
        fields.extend(("position", position) for _name, position in disagreement.positions)
        fields.append(("question_for_next_round", disagreement.question_for_next_round))
    fields.extend(("missing_fact", fact) for fact in agenda.missing_facts)
    for field, text in fields:
        if not _discuss_agenda_text_has_support(text, corpus=corpus, ignored_names=ignored_names):
            raise AgentLoopError(f"final analyzer {field} lacks final-round support: {text}")


def _run_discuss_final_analyzer(
    runner: Runner,
    *,
    issue_number: int,
    config: AgentLoopConfig,
    analyzer: AgentName | None,
    round_number: int,
    final_votes: Sequence[ParsedDiscussResponse],
    configured_reviewers: Sequence[AgentName],
    usage_context: RunUsageContext,
    mechanical_classification: str | None = None,
) -> tuple[ParsedDiscussAgenda | None, str | None, ParsedDiscussFinalSynthesis | None]:
    """Best-effort final-only analyzer pass; failures never affect finalization."""
    if analyzer is None or not final_votes or any(is_failed_discuss_response(vote) for vote in final_votes):
        return None, None, None


    analyzer_name = agent_display_name(analyzer)
    try:
        def validate_final_output(text: str) -> object:
            if config.discuss_result_mode == "answer":
                return validate_structured_discuss_final_synthesis(text)
            parsed_synthesis = parse_structured_discuss_final_synthesis(text)
            if parsed_synthesis is not None:
                return parsed_synthesis
            return validate_structured_discuss_agenda(text)

        response = _run_validated_agent(
            runner, agent=analyzer, config=config,
            prompt=(
                build_discuss_final_synthesis_prompt(
                    issue_number,
                    config,
                    analyzer=analyzer,
                    round_number=round_number,
                    final_responses=final_votes,
                    mechanical_classification=mechanical_classification or "material_deadlock",
                )
                if config.discuss_result_mode == "answer"
                else build_discuss_final_analysis_prompt(
                    issue_number, config, analyzer=analyzer, round_number=round_number,
                    final_votes=final_votes,
                )
            ),
            marker_description="<!-- AGENT_PLAN_STATE: approved -->",
            validate=validate_final_output, usage_context=usage_context,
            use_repair=True,
            repair_expected_kind=(
                "discuss_final_synthesis"
                if config.discuss_result_mode == "answer"
                else "discuss_agenda"
            ),
            role="analyzer",
            label=(
                f"discuss-final-synthesis-r{round_number}"
                if config.discuss_result_mode == "answer"
                else f"discuss-final-analyzer-r{round_number}"
            ),
            operation_description="final discuss synthesis",
        )
        parsed = response.marker_value
        if isinstance(parsed, ParsedDiscussFinalSynthesis):
            if mechanical_classification is None:
                raise AgentLoopError("final synthesis requires a mechanical classification.")
            _validate_discuss_final_synthesis_fidelity(
                parsed,
                expected_classification=mechanical_classification,
                final_votes=final_votes,
                round_number=round_number,
                configured_reviewers=configured_reviewers,
            )
            return None, response.text, parsed
        assert isinstance(parsed, ParsedDiscussAgenda)
        _validate_discuss_final_analyzer_fidelity(
            parsed, final_votes=final_votes, configured_reviewers=configured_reviewers, analyzer=analyzer,
        )
        return parsed, response.text, None
    except (QuotaResetExceededError, CheckoutVerificationError):
        raise
    except (AgentLoopError, AgentInvocationError) as exc:
        log(config, f"discuss: final analyzer {analyzer_name} unavailable ({exc}); omitting advisory observations")
        return None, None, None


def _run_discuss_evidence_reconciler(
    runner: Runner, *, issue_number: int, config: AgentLoopConfig, analyzer: AgentName | None,
    subject: str, round_history: Sequence[Sequence[ParsedDiscussResponse]], usage_context: RunUsageContext,
) -> tuple[tuple[tuple[str, ...], ...], str | None]:
    """Best-effort semantic grouping, isolated from the final agenda analyzer."""
    if analyzer is None:
        return (), None
    observations, updates = collect_evidence_observations(subject, round_history)
    candidates = bounded_reconciliation_candidates(observations, updates)
    candidate_ids = {str(candidate["id"]) for candidate in candidates}
    statuses = {item.observation_id: item.status for item in observations if item.observation_id in candidate_ids}
    if len(candidates) < 2:
        return (), None
    try:
        response = _run_validated_agent(
            runner, agent=analyzer, config=config,
            prompt=build_discuss_evidence_reconciliation_prompt(issue_number, config, analyzer=analyzer, candidates=candidates),
            marker_description="<!-- AGENT_PLAN_STATE: approved -->",
            validate=lambda text: validate_structured_discuss_evidence_reconciliation(text, observation_ids=tuple(statuses), observation_statuses=statuses),
            usage_context=usage_context, use_repair=True,
            repair_expected_kind="discuss_evidence_reconciliation", role="analyzer",
            label="discuss-evidence-reconciliation", operation_description="evidence reconciliation",
        )
        parsed = response.marker_value
        assert isinstance(parsed, ParsedDiscussEvidenceReconciliation)
        return parsed.groups, response.text
    except (QuotaResetExceededError, CheckoutVerificationError):
        raise
    except Exception as exc:
        log(config, f"discuss: evidence reconciler unavailable ({exc}); using exact-match ledger")
        return (), None


@dataclass(frozen=True)
class _DiscussDebaterTurnResult:
    """Outcome of one debater turn: a validated response or a captured failure.

    Worker threads in the parallel path return these instead of raising, so
    exceptions never cross the thread boundary; the main thread applies the
    failure policy after all futures settle (#475).
    """

    reviewer_name: str
    response: ValidatedAgentResponse | None = None
    error: AgentLoopError | None = None

    @property
    def failure_category(self) -> str:
        return getattr(self.error, "failure_category", None) or "error"


def _validate_structured_discuss_vote_with_evidence(
    text: str,
    *,
    reviewer_name: str,
    round_number: int,
    config: AgentLoopConfig,
    assigned_workdir: Path,
) -> ParsedDiscussResponse:
    """Parse a debater's structured vote, then reject any checkout-inspected
    evidence claim whose path:line reference does not resolve inside the
    reviewer's own assigned checkout right now (#541)."""
    parsed = (
        validate_structured_discuss_answer(
            text, reviewer=reviewer_name, round_number=round_number, research_mode=config.discuss_research
        )
        if config.discuss_result_mode == "answer"
        else validate_structured_discuss_review(
            text, reviewer=reviewer_name, round_number=round_number, research_mode=config.discuss_research
        )
    )
    validate_checkout_inspected_evidence(parsed.evidence_claims, assigned_workdir=assigned_workdir)
    return parsed


def _run_discuss_debater_turn(
    runner: Runner,
    *,
    reviewer: AgentName,
    reviewer_name: str,
    config: AgentLoopConfig,
    prompt: str,
    round_number: int,
    usage_context: RunUsageContext,
) -> ValidatedAgentResponse:
    """Run one debater turn from a prebuilt prompt.

    Never posts to GitHub or mutates shared round state, so it is safe to call
    from a worker thread; the caller posts the comment after the round's
    synchronization point.
    """
    assigned_workdir = get_backend(reviewer).workdir(config)
    return _run_validated_agent(
        runner,
        agent=reviewer,
        config=config,
        prompt=prompt,
        marker_description="<!-- AGENT_PLAN_STATE: approved -->",
        validate=lambda text, r=reviewer_name, rn=round_number, wd=assigned_workdir: (
            _validate_structured_discuss_vote_with_evidence(
                text, reviewer_name=r, round_number=rn, config=config, assigned_workdir=wd
            )
        ),
        usage_context=usage_context,
        use_repair=True,
        repair_expected_kind="discuss_answer" if config.discuss_result_mode == "answer" else "discuss_review",
        role="reviewer",
        label=f"discuss-r{round_number}",
        timeout_seconds=config.discuss_debater_timeout,
        operation_description="discuss review",
    )


def _run_discuss_semantic_finalization(
    runner: Runner, *, issue_number: int, config: AgentLoopConfig, answers: Sequence[ParsedDiscussAnswer],
    configured_reviewers: Sequence[AgentName], usage_context: RunUsageContext,
) -> tuple[str, str, dict[str, object] | None]:
    """Fail-closed semantic answer evaluation. Exact equality is intentionally outside this helper."""
    analyzer = config.discuss_analyzer
    if analyzer is None or len(answers) != len(configured_reviewers):
        return "deadlock", "deadlock", None
    if any(item.position != "answer" or not item.answer for item in answers) or _discuss_has_material_items(answers):
        return "deadlock", "deadlock", None
    names = [item.reviewer for item in answers]
    try:
        response = _run_validated_agent(
            runner, agent=analyzer, config=config,
            prompt=build_discuss_semantic_comparison_prompt(issue_number, config, answers=answers),
            marker_description="<!-- AGENT_PLAN_STATE: approved -->",
            validate=lambda text: validate_structured_discuss_semantic_comparison(text, reviewers=names),
            usage_context=usage_context, use_repair=True,
            repair_expected_kind="discuss_semantic_comparison", role="analyzer",
            label="discuss-semantic-comparison", operation_description="semantic answer comparison",
        )
        comparison = response.marker_value
        assert isinstance(comparison, ParsedDiscussSemanticComparison)
    except CheckoutVerificationError:
        raise
    except (AgentLoopError, AgentInvocationError):
        # Preserve an explicit audit record even when the comparator itself
        # fails. This distinguishes its fail-closed deadlock from a purely
        # textual disagreement in both the public summary and round metadata.
        return "deadlock", "semantic-comparison-failed", {
            "classification": "failed",
            "analyzer": agent_display_name(analyzer),
        }
    audit: dict[str, object] = {
        "classification": comparison.classification,
        "shared_recommendation": comparison.shared_recommendation,
        "remaining_decisions": comparison.remaining_decisions,
        "evidence": comparison.evidence,
        "analyzer": agent_display_name(analyzer),
    }
    if comparison.classification == "equivalent":
        return "answer", "semantic-equivalent", audit
    if comparison.classification == "material_conflict":
        return "deadlock", "material-conflict", audit
    confirmations = []
    try:
        for reviewer in configured_reviewers:
            reviewer_name = agent_display_name(reviewer)
            response = _run_validated_agent(
                runner, agent=reviewer, config=config,
                prompt=build_discuss_answer_confirmation_prompt(
                    issue_number, config, reviewer=reviewer,
                    shared_recommendation=comparison.shared_recommendation,
                    remaining_decisions=comparison.remaining_decisions),
                marker_description="<!-- AGENT_PLAN_STATE: approved -->",
                validate=lambda text, name=reviewer_name: validate_structured_discuss_answer_confirmation(text, reviewer=name),
                usage_context=usage_context, use_repair=True,
                repair_expected_kind="discuss_answer_confirmation", role="reviewer",
                label="discuss-answer-confirmation", timeout_seconds=config.discuss_debater_timeout,
                operation_description="semantic answer confirmation",
            )
            confirmations.append(response.marker_value)
    except CheckoutVerificationError:
        raise
    except (AgentLoopError, AgentInvocationError):
        return "deadlock", "confirmation-failed", audit
    effective = [comparison.shared_recommendation if item.decision == "confirm" else item.answer for item in confirmations]
    audit["confirmations"] = tuple(confirmations)
    if all(answer and _normalize_discuss_answer(answer) == _normalize_discuss_answer(effective[0] or "") for answer in effective):
        audit["confirmed_answer"] = effective[0]
        return "answer", "debater-confirmed", audit
    return "deadlock", "confirmation-disagreement", audit


def _adapt_discuss_final_synthesis(
    *,
    outcome: str,
    consensus_kind: str | None,
    final_votes: Sequence[ParsedDiscussResponse],
    round_number: int,
    semantic_comparison: Mapping[str, object] | None = None,
) -> ParsedDiscussFinalSynthesis | None:
    """Reuse mechanically verified artifacts without another analyzer call."""
    def cap(text: str) -> str:
        raw = str(text).strip().encode("utf-8")[:DISCUSS_SYNTHESIS_MAX_TEXT_BYTES]
        return raw.decode("utf-8", "ignore").strip() or "(not stated)"

    def key(text: str) -> str:
        return cap(text).casefold()

    successful_votes = tuple(
        vote for vote in final_votes if not is_failed_discuss_response(vote)
    )
    answers = [vote for vote in successful_votes if isinstance(vote, ParsedDiscussAnswer)]
    if len(successful_votes) > DISCUSS_SYNTHESIS_MAX_ENTRIES:
        # The protocol cannot represent a complete reference set above this
        # bound. Let the normal fail-closed rendering handle that case.
        references = ()
    else:
        references = tuple(
            DiscussSynthesisResponseReference(vote.reviewer, round_number)
            for vote in successful_votes
        )
    if (
        outcome == "answer"
        and consensus_kind in {"unanimous", "converged"}
        and len(answers) == len(successful_votes)
        and references
        and answers
        and all(vote.answer for vote in answers)
    ):
        # This is the exact-text path. `_detect_discuss_answer_consensus` has
        # already established normalized equality, so no model call is needed.
        return ParsedDiscussFinalSynthesis(
            classification="consensus",
            agreed_conclusions=(
                DiscussSynthesisConsensus(text=cap(answers[0].answer or ""), references=references),
            ),
            remaining_disagreements=(),
            next_action="Proceed with the shared recommendation.",
        )
    if semantic_comparison is not None:
        classification = semantic_comparison.get("classification")
        if (
            classification == "equivalent"
            and semantic_comparison.get("shared_recommendation")
            and references
        ):
            return ParsedDiscussFinalSynthesis(
                classification="consensus",
                agreed_conclusions=(
                    DiscussSynthesisConsensus(
                        text=cap(str(semantic_comparison["shared_recommendation"])),
                        references=references,
                    ),
                ),
                remaining_disagreements=(),
                next_action="Proceed with the shared recommendation.",
            )
        if consensus_kind == "debater-confirmed" and semantic_comparison.get("confirmed_answer"):
            if not references:
                return None
            return ParsedDiscussFinalSynthesis(
                classification="consensus",
                agreed_conclusions=(
                    DiscussSynthesisConsensus(
                        text=cap(str(semantic_comparison["confirmed_answer"])),
                        references=references,
                    ),
                ),
                remaining_disagreements=(),
                next_action="Proceed with the debater-confirmed recommendation.",
            )
    if outcome == "needs-human" and consensus_kind in {"unanimous", "converged"}:
        human_items = [
            (vote, item)
            for vote in successful_votes
            for item in getattr(vote, "unresolved_items", ())
            if item.status == "human-decision"
        ]
        if human_items:
            groups: dict[str, list[tuple[ParsedDiscussResponse, DiscussUnresolvedItem]]] = {}
            for vote, item in human_items:
                bucket = groups.setdefault(key(item.text), [])
                if any(existing_vote.reviewer.casefold() == vote.reviewer.casefold() for existing_vote, _ in bucket):
                    # Multiple equivalent items from one reviewer must not
                    # create a duplicate reviewer position in one topic.
                    continue
                bucket.append((vote, item))
            groups = dict(list(groups.items())[:DISCUSS_SYNTHESIS_MAX_ENTRIES])
            disagreements = tuple(
                DiscussSynthesisDisagreement(
                    topic=cap(items[0][1].text),
                    positions=tuple(
                        DiscussSynthesisPosition(
                            reviewers=(vote.reviewer,), position=cap(item.text)
                        )
                        for vote, item in items[:DISCUSS_SYNTHESIS_MAX_ENTRIES]
                    ),
                    decision_needed=cap(items[0][1].text),
                )
                for items in groups.values()
            )
            shared_rationales = [cap(vote.rationale) for vote in successful_votes]
            agreed_conclusions = ()
            if (
                references
                and shared_rationales
                and len({_normalize_discuss_answer(item) for item in shared_rationales}) == 1
            ):
                agreed_conclusions = (
                    DiscussSynthesisConsensus(
                        text=shared_rationales[0], references=references
                    ),
                )
            return ParsedDiscussFinalSynthesis(
                classification="near_consensus",
                agreed_conclusions=agreed_conclusions,
                remaining_disagreements=disagreements,
                next_action=cap(
                    "A human must decide: "
                    + "; ".join(item.topic for item in disagreements)
                ),
            )
    return None


def _safe_discuss_synthesis_serialization(
    synthesis: ParsedDiscussRoundSynthesis | ParsedDiscussFinalSynthesis | None,
    *,
    final: bool,
    config: AgentLoopConfig,
    location: str,
) -> tuple[ParsedDiscussRoundSynthesis | ParsedDiscussFinalSynthesis | None, str | None]:
    """Drop advisory synthesis that cannot be durably encoded.

    Rendering must remain fail-closed even when a mechanically assembled
    candidate or a legacy structured response is individually valid but cannot
    fit the canonical sidecar budget.
    """
    if synthesis is None:
        return None, None
    try:
        serialized = (
            serialize_discuss_final_synthesis(synthesis)  # type: ignore[arg-type]
            if final
            else serialize_discuss_round_synthesis(synthesis)  # type: ignore[arg-type]
        )
    except Exception as exc:
        log(config, f"discuss: dropping {location} synthesis that could not be serialized ({exc})")
        return None, None
    return synthesis, serialized


def _post_discuss_debater_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    reviewer_name: str,
    parsed: ParsedDiscussResponse,
    response: ValidatedAgentResponse,
    round_number: int,
    subject: str,
) -> None:
    post_issue_comment(
        runner,
        config=config,
        issue_number=issue_number,
        body=_attach_round_metadata(
            render_public_agent_comment(
                kind="discuss_answer" if config.discuss_result_mode == "answer" else "discuss_review",
                parsed=parsed,
                agent=reviewer_name,
                config=config,
                model_used=response.model_used,
                round_number=round_number,
            ),
            PostedRoundMetadata(
                flow="discuss",
                role="debater",
                agent=reviewer_name,
                round_number=round_number,
                subject=subject,
                raw_structured_coder_response=response.text,
                model_used=response.model_used,
                **_metadata_identity_fields(response),
                research_mode=config.discuss_research,
                result_mode=config.discuss_result_mode,
            ),
        ),
    )


def _validate_discuss_evidence_update_targets(
    parsed: ParsedDiscussResponse, *, subject: str, round_history: Sequence[Sequence[ParsedDiscussResponse]],
) -> None:
    """Reject active malformed updates; legacy replay merely audits them."""
    if not isinstance(parsed, (ParsedDiscussReview, ParsedDiscussAnswer)):
        return
    observations, _updates = collect_evidence_observations(subject, round_history)
    allowed = {item.observation_id for item in observations}
    for update in parsed.evidence_updates:
        if not update.target_observation_id.startswith(f"{subject}-") or update.target_observation_id not in allowed:
            raise AgentLoopError(f"evidence update targets unknown or cross-subject observation: {update.target_observation_id}")


def _run_discuss_loop(
    runner: Runner,
    *,
    issue_number: int,
    config: AgentLoopConfig,
    usage_context: RunUsageContext,
    discuss_max_rounds: int = 2,
) -> int:
    from .config import reviewers as _reviewers
    if discuss_max_rounds < 0:
        raise AgentLoopError("--discuss-max-rounds must be zero or greater.")
    issue_context = get_issue_context(runner, config=config, issue_number=issue_number)
    subject = _discuss_subject(issue_context)
    configured_reviewers = list(_reviewers(config))
    reviewer_workdirs = {
        agent_display_name(agent): get_backend(agent).workdir(config) for agent in configured_reviewers
    }
    resume_state = _resume_discuss_round(
        issue_context.comments, subject=subject, configured_reviewers=configured_reviewers,
        reviewer_workdirs=reviewer_workdirs, result_mode=config.discuss_result_mode,
    )

    def _resolve_final_split_proposals() -> tuple[list[str], Sequence[ParsedDiscussReview]] | None:
        if config.discuss_result_mode == "answer":
            return None
        # Resuming (or already-final) reruns must materialize a split consensus
        # instead of silently skipping (#476): prefer the split proposals
        # recorded directly on the final summary metadata, and fall back to
        # reconstructing them from debater comment metadata for legacy
        # summaries that predate the `split_proposals` field.
        if resume_state is not None and resume_state.done and resume_state.round_history:
            final_votes = resume_state.round_history[-1]
            consensus = _detect_discuss_consensus(list(final_votes))
            if consensus is not None and consensus[0] == "split" and consensus[1]:
                return consensus[1], final_votes
        return _recover_final_discuss_split_proposals(
            issue_context, subject=subject, configured_reviewers=configured_reviewers,
            reviewer_workdirs=reviewer_workdirs,
        )

    already_final = False
    for comment in issue_context.comments or []:
        body = comment.body or ""
        m = DISCUSS_CONSENSUS_MARKER_RE.search(body)
        if m and m.group(1).lower() == subject:
            already_final = True
            break
    if already_final or (resume_state is not None and resume_state.done):
        log(config, f"discuss: found matching consensus for issue #{issue_number}; skipping debate")
        recovered = _resolve_final_split_proposals()
        if recovered is not None:
            split_proposals, final_votes = recovered
            split_result = _handle_discuss_split_outcome(
                runner,
                issue_number=issue_number,
                config=config,
                subject=subject,
                split_proposals=split_proposals,
                final_votes=final_votes,
                issue_comments=issue_context.comments,
                post_warning_comment=False,
            )
            if isinstance(split_result, NeedsHumanDecision):
                print(json.dumps(split_result.as_dict(), sort_keys=True))
                return 2
        elif config.materialize_split_issues:
            log(
                config,
                f"discuss: issue #{issue_number} has a final consensus comment but no "
                "recoverable split-proposal metadata; nothing to materialize (or this was not "
                "a `split` consensus).",
            )
        return 0
    memory = prepare_agent_memory(runner, config)
    analyzer = config.discuss_analyzer
    analyzer_name = agent_display_name(analyzer) if analyzer is not None else None
    prompt_issue_context = issue_context
    if analyzer is not None:
        # Agenda-focused analyzer mode: prior rounds reach debaters only through
        # the structured agenda (or the analyzer via round_history), so strip
        # discuss-flow bot comments from the prompt-facing issue context. Plain
        # mode keeps the full context unchanged.
        prompt_issue_context = dataclasses_replace(
            issue_context,
            comments=tuple(
                comment
                for comment in (issue_context.comments or ())
                if not (comment.body and _is_bot_authored_discuss_comment(comment.body))
            ),
        )
    if analyzer is not None and discuss_max_rounds == 0:
        log(
            config,
            "discuss: --discuss-analyzer is set but --discuss-max-rounds=0 leaves no "
            "non-final round, so the analyzer will not run.",
        )
    if resume_state is not None:
        round_history: list[list[ParsedDiscussReview]] = [list(votes) for votes in resume_state.round_history]
        start_round_number = resume_state.next_round_number
        prior_round_agenda: list[str] = list(resume_state.prior_round_agenda)
        prior_analyzer_agenda: ParsedDiscussAgenda | None = (
            resume_state.prior_analyzer_agenda if analyzer is not None else None
        )
        prior_round_synthesis: ParsedDiscussRoundSynthesis | None = (
            resume_state.prior_round_synthesis if analyzer is not None else None
        )
        in_progress_votes: dict[str, ParsedDiscussResponse] = dict(resume_state.in_progress_votes)
        if round_history or in_progress_votes:
            log(config, f"discuss: resuming issue #{issue_number} at round {start_round_number}")
    else:
        round_history = []
        start_round_number = 1
        prior_round_agenda = []
        prior_analyzer_agenda = None
        prior_round_synthesis = None
        in_progress_votes = {}
    max_round_number = discuss_max_rounds + 1
    if start_round_number > max_round_number:
        if not round_history:
            raise AgentLoopError(
                "discuss: resumed state expects round "
                f"{start_round_number} but --discuss-max-rounds={discuss_max_rounds} allows only "
                f"{max_round_number} round(s), and no completed round was found to finalize from. "
                "Rerun with a --discuss-max-rounds at least as large as the rounds already posted "
                "on the issue, or repair the discuss transcript."
            )
        log(
            config,
            f"discuss: resumed round {start_round_number} exceeds --discuss-max-rounds="
            f"{discuss_max_rounds} (allows {max_round_number} round(s)); finalizing from the last "
            "completed round instead of starting a new one.",
        )
        final_round_number = len(round_history)
        final_votes = round_history[-1]
        # A resumed partial round (#475) may carry placeholder votes; keep the
        # vote table to real positions and surface the rest as failures.
        final_successful_votes = [
            vote for vote in final_votes if not is_failed_discuss_response(vote)
        ]
        final_failed_debaters = tuple(
            (
                vote.reviewer,
                (
                    failed_discuss_review_category(vote)
                    if isinstance(vote, ParsedDiscussReview)
                    else vote.category
                ),
            )
            for vote in final_votes
            if is_failed_discuss_response(vote)
        )
        final_synthesis: ParsedDiscussFinalSynthesis | None = None
        final_analyzer_agenda: ParsedDiscussAgenda | None = None
        final_analyzer_response_raw: str | None = None
        final_synthesis_source: str | None = None
        final_mechanical_classification = (
            _mechanical_discuss_final_classification(
                outcome="needs-human", consensus_kind="deadlock", votes=final_votes
            )
            if config.discuss_result_mode == "answer"
            else None
        )
        final_analyzer_agenda, final_analyzer_response_raw, final_synthesis = _run_discuss_final_analyzer(
            runner,
            issue_number=issue_number,
            config=config,
            analyzer=analyzer,
            round_number=final_round_number,
            final_votes=final_successful_votes,
            configured_reviewers=configured_reviewers,
            usage_context=usage_context,
            mechanical_classification=final_mechanical_classification,
        )
        if final_synthesis is not None:
            final_synthesis_source = "final-analyzer"
        final_synthesis, final_synthesis_serialized = _safe_discuss_synthesis_serialization(
            final_synthesis,
            final=True,
            config=config,
            location="resumed final",
        )
        evidence_groups, evidence_reconciler_raw = _run_discuss_evidence_reconciler(
            runner, issue_number=issue_number, config=config, analyzer=analyzer, subject=f"issue-{issue_number}",
            round_history=round_history, usage_context=usage_context,
        )
        evidence_reconciliation = reconcile_evidence(f"issue-{issue_number}", round_history, evidence_groups)
        evidence_reconciliation["raw_evidence_reconciler_response"] = evidence_reconciler_raw
        summary_body = render_discuss_round_summary_comment(
            round_number=final_round_number,
            reviewer_votes=final_successful_votes,
            is_final=True,
            subject=subject,
            outcome="needs-human",
            consensus_kind="deadlock",
            round_history=round_history,
            split_proposals=[],
            prior_analyzer_agenda=prior_analyzer_agenda,
            final_analyzer_agenda=final_analyzer_agenda,
            analyzer_name=analyzer_name,
            research_mode=config.discuss_research,
            failed_debaters=final_failed_debaters,
            result_mode=config.discuss_result_mode,
            final_synthesis=final_synthesis,
            evidence_reconciliation=evidence_reconciliation,
        )
        post_issue_comment(
            runner,
            config=config,
            issue_number=issue_number,
            body=_attach_round_metadata(
                summary_body,
                PostedRoundMetadata(
                    flow="discuss",
                    role="summary",
                    agent="Orchestrator",
                    round_number=final_round_number,
                    subject=subject,
                    is_final=True,
                    consensus_kind="deadlock",
                    agenda=(),
                    final_analyzer_response=final_analyzer_response_raw,
                    final_synthesis=final_synthesis_serialized,
                    synthesis_provenance=(
                        {
                            "source": final_synthesis_source or "final-analyzer",
                            "round": final_round_number,
                        }
                        if final_synthesis is not None else None
                    ),
                    research_mode=config.discuss_research,
                    failed_debaters=final_failed_debaters,
                    evidence_reconciliation=evidence_reconciliation,
                    result_mode=config.discuss_result_mode,
                ),
            ),
        )
        log(
            config,
            f"discuss: posted final summary comment for issue #{issue_number} "
            "(outcome: needs-human; kind: deadlock)",
        )
        return 0
    for round_number in range(start_round_number, max_round_number + 1):
        prior_round_votes = round_history[-1] if round_history else []
        votes_by_name: dict[str, ParsedDiscussResponse] = {}
        failures_by_name: dict[str, _DiscussDebaterTurnResult] = {}
        pending: list[AgentName] = []
        for reviewer in configured_reviewers:
            reviewer_name = agent_display_name(reviewer)
            resumed_vote = (
                in_progress_votes.get(reviewer_name) if round_number == start_round_number else None
            )
            if resumed_vote is not None:
                log(
                    config,
                    f"discuss: resuming {reviewer_name}'s posted round {round_number} position "
                    f"on issue #{issue_number}",
                )
                votes_by_name[reviewer_name] = resumed_vote
            else:
                pending.append(reviewer)

        def _build_debater_prompt(reviewer: AgentName) -> str:
            return build_discuss_review_prompt(
                issue_number,
                config,
                reviewer=reviewer,
                memory=memory,
                issue_context=prompt_issue_context,
                round_number=round_number,
                prior_round_votes=prior_round_votes,
                prior_round_agenda=prior_round_agenda,
                analyzer_agenda=prior_analyzer_agenda,
                research_mode=config.discuss_research,
            )

        if config.discuss_parallel and pending:
            # Same-round debaters run concurrently; prompts are built up front
            # from shared pre-round state and comments are posted only after
            # every future settles, so no debater can see a co-debater's
            # in-progress round-N output. Zero-pending resumes skip this branch
            # entirely (no executor is constructed).
            prompts = {
                agent_display_name(reviewer): _build_debater_prompt(reviewer)
                for reviewer in pending
            }
            pending_names = [agent_display_name(reviewer) for reviewer in pending]
            log(
                config,
                f"discuss: invoking {', '.join(pending_names)} in parallel on "
                f"issue #{issue_number} (round {round_number})",
            )

            def _debater_worker(reviewer: AgentName, reviewer_name: str) -> _DiscussDebaterTurnResult:
                try:
                    response = _run_discuss_debater_turn(
                        runner,
                        reviewer=reviewer,
                        reviewer_name=reviewer_name,
                        config=config,
                        prompt=prompts[reviewer_name],
                        round_number=round_number,
                        usage_context=usage_context,
                    )
                    parsed = response.marker_value
                    assert isinstance(parsed, (ParsedDiscussReview, ParsedDiscussAnswer))
                    _validate_discuss_evidence_update_targets(
                        parsed, subject=f"issue-{issue_number}", round_history=round_history
                    )
                except AgentLoopError as exc:
                    # Includes QuotaResetExceededError: captured here and
                    # re-raised on the main thread with priority.
                    return _DiscussDebaterTurnResult(reviewer_name=reviewer_name, error=exc)
                return _DiscussDebaterTurnResult(reviewer_name=reviewer_name, response=response)

            executor = ThreadPoolExecutor(
                max_workers=len(pending), thread_name_prefix=f"discuss-r{round_number}"
            )
            try:
                futures = {
                    agent_display_name(reviewer): executor.submit(
                        contextvars.copy_context().run,
                        _debater_worker,
                        reviewer,
                        agent_display_name(reviewer),
                    )
                    for reviewer in pending
                }
                # The analyzer synchronization point: wait for every debater.
                turn_results = {name: future.result() for name, future in futures.items()}
            except KeyboardInterrupt:
                # Workers never receive the terminal SIGINT; kill their agent
                # process groups so their wait loops return and the shutdown
                # below completes promptly, then propagate the interrupt.
                runner.terminate_active_processes()
                raise
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
            for name in pending_names:
                turn = turn_results[name]
                if turn.error is None:
                    parsed = turn.response.marker_value
                    assert isinstance(parsed, (ParsedDiscussReview, ParsedDiscussAnswer))
                    _post_discuss_debater_comment(
                        runner,
                        config=config,
                        issue_number=issue_number,
                        reviewer_name=name,
                        parsed=parsed,
                        response=turn.response,
                        round_number=round_number,
                        subject=subject,
                    )
                    votes_by_name[name] = parsed
                else:
                    failures_by_name[name] = turn
            # Successful votes are posted above even when the round is about
            # to abort, so a rerun resumes them instead of re-invoking.
            for name in pending_names:
                turn = failures_by_name.get(name)
                if turn is not None and isinstance(
                    turn.error, (QuotaResetExceededError, CheckoutVerificationError)
                ):
                    raise turn.error
            if failures_by_name and config.discuss_on_debater_failure == "fail":
                raise next(iter(failures_by_name.values())).error
        else:
            for reviewer in pending:
                reviewer_name = agent_display_name(reviewer)
                log(
                    config,
                    f"discuss: invoking {reviewer_name} on issue #{issue_number} "
                    f"(round {round_number})",
                )
                try:
                    response = _run_discuss_debater_turn(
                        runner,
                        reviewer=reviewer,
                        reviewer_name=reviewer_name,
                        config=config,
                        prompt=_build_debater_prompt(reviewer),
                        round_number=round_number,
                        usage_context=usage_context,
                    )
                except (QuotaResetExceededError, CheckoutVerificationError):
                    raise
                except AgentLoopError as exc:
                    if config.discuss_on_debater_failure == "fail":
                        raise
                    failures_by_name[reviewer_name] = _DiscussDebaterTurnResult(
                        reviewer_name=reviewer_name, error=exc
                    )
                    log(
                        config,
                        f"discuss: {reviewer_name} failed round {round_number} "
                        f"({failures_by_name[reviewer_name].failure_category}); continuing "
                        "per --discuss-on-debater-failure=partial",
                    )
                    continue
                parsed = response.marker_value
                assert isinstance(parsed, (ParsedDiscussReview, ParsedDiscussAnswer))
                _validate_discuss_evidence_update_targets(
                    parsed, subject=f"issue-{issue_number}", round_history=round_history
                )
                _post_discuss_debater_comment(
                    runner,
                    config=config,
                    issue_number=issue_number,
                    reviewer_name=reviewer_name,
                    parsed=parsed,
                    response=response,
                    round_number=round_number,
                    subject=subject,
                )
                votes_by_name[reviewer_name] = parsed

        failed_debaters: list[tuple[str, str]] = []
        reviewer_votes: list[ParsedDiscussResponse] = []
        for reviewer in configured_reviewers:
            reviewer_name = agent_display_name(reviewer)
            vote = votes_by_name.get(reviewer_name)
            if vote is not None:
                reviewer_votes.append(vote)
                continue
            failure = failures_by_name[reviewer_name]
            category = failure.failure_category
            failed_debaters.append((reviewer_name, category))
            reviewer_votes.append(
                failed_discuss_answer_placeholder(reviewer_name, category)
                if config.discuss_result_mode == "answer"
                else failed_discuss_review_placeholder(reviewer_name, category)
            )
        if failed_debaters:
            # Reached only under the "partial" policy: continue when at least
            # two debaters produced votes; a partial round can never declare
            # final consensus because the placeholder outcome differs.
            if len(votes_by_name) < 2:
                log(
                    config,
                    "discuss: --discuss-on-debater-failure=partial requires at least two "
                    f"successful debater votes in round {round_number}, got {len(votes_by_name)}",
                )
                raise next(iter(failures_by_name.values())).error
            log(
                config,
                f"discuss: continuing round {round_number} with partial results; failed "
                "debater(s): "
                + ", ".join(f"{name} ({category})" for name, category in failed_debaters),
            )
        successful_votes = [
            vote for vote in reviewer_votes
            if not is_failed_discuss_response(vote)
        ]
        round_history.append(reviewer_votes)
        if config.discuss_result_mode == "answer":
            answer_votes = [vote for vote in successful_votes if isinstance(vote, ParsedDiscussAnswer)]
            consensus = _detect_discuss_answer_consensus(answer_votes, partial=bool(failed_debaters))
        else:
            consensus = _detect_discuss_consensus(reviewer_votes)  # type: ignore[arg-type]
        is_final = consensus is not None or round_number == max_round_number
        # The completed current round is authoritative: a later round can
        # clear/reclassify earlier items. At the final round, classified
        # material selects a fail-closed outcome before text comparison.
        if (
            is_final and config.discuss_result_mode == "answer" and not failed_debaters
            and len(answer_votes) == len(configured_reviewers)
        ):
            material_outcome = _final_discuss_answer_item_outcome(answer_votes)
            if material_outcome is not None:
                consensus = (material_outcome, [])
        semantic_comparison: dict[str, object] | None = None
        semantic_finalization_ran = False
        # Keep normalized equality as the zero-call fast path.  Only a complete,
        # final all-answer round is eligible for the configured independent analyzer.
        if (
            is_final and consensus is None and config.discuss_result_mode == "answer"
            and not failed_debaters
            and len(successful_votes) == len(configured_reviewers)
            and all(isinstance(vote, ParsedDiscussAnswer) and vote.position == "answer" and vote.answer for vote in successful_votes)
            and not _discuss_has_material_items(
                [vote for vote in successful_votes if isinstance(vote, ParsedDiscussAnswer)]
            )
        ):
            semantic_finalization_ran = True
            outcome, semantic_kind, semantic_comparison = _run_discuss_semantic_finalization(
                runner, issue_number=issue_number, config=config,
                answers=[vote for vote in successful_votes if isinstance(vote, ParsedDiscussAnswer)],
                configured_reviewers=configured_reviewers, usage_context=usage_context,
            )
            consensus_kind = semantic_kind
            consensus = (outcome, []) if outcome == "answer" else None
        if consensus is None:
            outcome = "deadlock" if config.discuss_result_mode == "answer" else "needs-human"
            round_split_proposals: list[str] = (
                [] if config.discuss_result_mode == "answer" else _merge_discuss_split_proposals(successful_votes)  # type: ignore[arg-type]
            )
            consensus_kind = None if not is_final else (
                semantic_kind if semantic_finalization_ran else "deadlock"
            )
        else:
            outcome, round_split_proposals = consensus
            if semantic_comparison is None:
                consensus_kind = ("unanimous" if len(round_history) == 1 else "converged") if is_final else None
        next_analyzer_agenda: ParsedDiscussAgenda | None = None
        analyzer_response_raw: str | None = None
        next_round_synthesis: ParsedDiscussRoundSynthesis | None = None
        raw_synthesis_response: str | None = None
        final_analyzer_agenda: ParsedDiscussAgenda | None = None
        final_analyzer_response_raw: str | None = None
        final_synthesis: ParsedDiscussFinalSynthesis | None = None
        final_synthesis_source: str | None = None
        if not is_final and analyzer is not None:
            (
                next_analyzer_agenda,
                analyzer_response_raw,
                next_round_synthesis,
                raw_synthesis_response,
            ) = _run_discuss_analyzer(
                runner,
                issue_number=issue_number,
                config=config,
                analyzer=analyzer,
                memory=memory,
                issue_context=prompt_issue_context,
                round_number=round_number,
                round_history=round_history,
                prior_agenda=prior_analyzer_agenda,
                prior_round_synthesis=prior_round_synthesis,
                configured_reviewers=configured_reviewers,
                usage_context=usage_context,
            )
        elif is_final and config.discuss_result_mode == "answer":
            final_mechanical_classification = _mechanical_discuss_final_classification(
                outcome=outcome, consensus_kind=consensus_kind, votes=reviewer_votes
            )
            final_synthesis = _adapt_discuss_final_synthesis(
                outcome=outcome,
                consensus_kind=consensus_kind,
                final_votes=successful_votes,
                round_number=round_number,
                semantic_comparison=semantic_comparison,
            )
            if final_synthesis is not None:
                final_synthesis_source = (
                    "semantic-comparison"
                    if semantic_comparison is not None
                    else "mechanical-result"
                )
            # Near-consensus candidates are only a bounded mechanical
            # fallback. Give the configured analyzer the opportunity to
            # recover agreed recommendations and accurately summarize the
            # residual human decisions.
            if final_synthesis is None or final_mechanical_classification == "near_consensus":
                (
                    final_analyzer_agenda,
                    final_analyzer_response_raw,
                    analyzer_synthesis,
                ) = _run_discuss_final_analyzer(
                    runner,
                    issue_number=issue_number,
                    config=config,
                    analyzer=analyzer,
                    round_number=round_number,
                    final_votes=successful_votes,
                    configured_reviewers=configured_reviewers,
                    usage_context=usage_context,
                    mechanical_classification=final_mechanical_classification,
                )
                # The mechanical/adapted synthesis is a safe fallback. An
                # unavailable, malformed, or rejected advisory analyzer must
                # not erase it and return to the pre-synthesis rendering.
                if analyzer_synthesis is not None:
                    final_synthesis = analyzer_synthesis
                    final_synthesis_source = "final-analyzer"
        elif is_final:
            # Semantic finalization already used the configured analyzer to
            # compare the completed final-round answers. When every debater
            # confirms that recommendation, keep that audit as the final
            # analyzer record rather than invoking the same analyzer again
            # for advisory observations over identical input.
            final_analyzer_agenda, final_analyzer_response_raw, _unused_final_synthesis = _run_discuss_final_analyzer(
                runner,
                issue_number=issue_number,
                config=config,
                analyzer=analyzer,
                round_number=round_number,
                final_votes=successful_votes,
                configured_reviewers=configured_reviewers,
                usage_context=usage_context,
            )
        evidence_reconciliation = None
        if is_final:
            evidence_groups, evidence_reconciler_raw = _run_discuss_evidence_reconciler(
                runner, issue_number=issue_number, config=config, analyzer=analyzer, subject=f"issue-{issue_number}",
                round_history=round_history, usage_context=usage_context,
            )
            evidence_reconciliation = reconcile_evidence(f"issue-{issue_number}", round_history, evidence_groups)
            evidence_reconciliation["raw_evidence_reconciler_response"] = evidence_reconciler_raw
        next_round_synthesis, round_synthesis_serialized = _safe_discuss_synthesis_serialization(
            next_round_synthesis,
            final=False,
            config=config,
            location="round",
        )
        raw_synthesis_response = (
            raw_synthesis_response if next_round_synthesis is not None else None
        )
        final_synthesis, final_synthesis_serialized = _safe_discuss_synthesis_serialization(
            final_synthesis,
            final=True,
            config=config,
            location="final",
        )
        if final_synthesis is None:
            final_synthesis_source = None
        summary_body = render_discuss_round_summary_comment(
            round_number=round_number,
            # The vote table and agenda draw from real positions only; failed
            # debaters surface in the dedicated failures section instead.
            reviewer_votes=successful_votes,
            is_final=is_final,
            subject=subject,
            outcome=outcome if is_final else None,
            consensus_kind=consensus_kind,
            round_history=round_history if is_final else None,
            split_proposals=round_split_proposals,
            analyzer_agenda=next_analyzer_agenda,
            prior_analyzer_agenda=prior_analyzer_agenda if is_final else None,
            final_analyzer_agenda=final_analyzer_agenda,
            analyzer_name=analyzer_name,
            research_mode=config.discuss_research,
            failed_debaters=tuple(failed_debaters),
            result_mode=config.discuss_result_mode,
            semantic_comparison=semantic_comparison,
            evidence_reconciliation=evidence_reconciliation,
            round_synthesis=next_round_synthesis,
            final_synthesis=final_synthesis,
        )
        agenda = () if is_final else tuple(_render_discuss_agenda_lines(successful_votes))
        post_issue_comment(
            runner,
            config=config,
            issue_number=issue_number,
            body=_attach_round_metadata(
                summary_body,
                PostedRoundMetadata(
                    flow="discuss",
                    role="summary",
                    agent="Orchestrator",
                    round_number=round_number,
                    subject=subject,
                    is_final=is_final,
                    consensus_kind=consensus_kind,
                    agenda=agenda,
                    analyzer_response=analyzer_response_raw,
                    final_analyzer_response=final_analyzer_response_raw,
                    round_synthesis=round_synthesis_serialized,
                    final_synthesis=final_synthesis_serialized,
                    raw_synthesis_response=raw_synthesis_response,
                    synthesis_provenance=(
                        {
                            "source": (
                                "round-analyzer" if raw_synthesis_response is None
                                else "round-fallback"
                            ),
                            "round": round_number,
                        }
                        if next_round_synthesis is not None else (
                            {
                                "source": final_synthesis_source or "final-analyzer",
                                "round": round_number,
                            }
                            if final_synthesis is not None else None
                        )
                    ),
                    research_mode=config.discuss_research,
                    failed_debaters=tuple(failed_debaters),
                    split_proposals=tuple(round_split_proposals) if is_final else (),
                    result_mode=config.discuss_result_mode,
                    evidence_reconciliation=evidence_reconciliation,
                ),
            ),
        )
        if is_final:
            log(
                config,
                f"discuss: posted final summary comment for issue #{issue_number} "
                f"(outcome: {outcome}; kind: {consensus_kind})",
            )
            if outcome == "split":
                split_result = _handle_discuss_split_outcome(
                    runner,
                    issue_number=issue_number,
                    config=config,
                    subject=subject,
                    split_proposals=round_split_proposals,
                    final_votes=successful_votes,
                    issue_comments=issue_context.comments,
                    post_warning_comment=True,
                )
                if isinstance(split_result, NeedsHumanDecision):
                    print(json.dumps(split_result.as_dict(), sort_keys=True))
                    return 2
            return 0
        log(
            config,
            f"discuss: posted round {round_number} summary comment for issue #{issue_number}; "
            f"continuing to round {round_number + 1}",
        )
        prior_round_agenda = list(agenda)
        prior_analyzer_agenda = next_analyzer_agenda
        prior_round_synthesis = next_round_synthesis
    return 0


@claimed_run("discuss", "issue_number")
def run_discuss_loop(
    runner: Runner,
    *,
    issue_number: int,
    config: AgentLoopConfig,
    discuss_max_rounds: int = 2,
    usage_context: RunUsageContext | None = None,
) -> int:
    owned_usage_context = usage_context is None
    usage_context = usage_context or _new_usage_context(config)
    telemetry_token = _begin_run_telemetry(
        runner, config, usage_context, owned_usage_context, issue_number=issue_number
    )
    try:
        if config.discuss_parallel:
            _ensure_parallel_discuss_workdirs(config)
        config = resolve_base_branch(config, runner)
        ensure_agent_workdirs(config, runner)
        log(config, f"discuss: validating issue #{issue_number}")
        validate_open_issue(runner, config=config, issue_number=issue_number)
        return _run_discuss_loop(
            runner,
            issue_number=issue_number,
            config=config,
            usage_context=usage_context,
            discuss_max_rounds=discuss_max_rounds,
        )
    finally:
        _end_run_telemetry(runner, telemetry_token)
        if owned_usage_context:
            _persist_usage_summary(config, usage_context)
