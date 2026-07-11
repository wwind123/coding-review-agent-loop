"""Round metadata persistence and resume helpers."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from .agents.base import AgentName
from .agents.registry import agent_display_name
from .errors import AgentLoopError
from .protocol import (
    HTML_COMMENT_RE,
    SIGNATURE_RE,
    ParsedDiscussAgenda,
    ParsedDiscussAnswer,
    ParsedDiscussResponse,
    ParsedDiscussReview,
    ReviewItemDisposition,
    UnresolvedReviewItem,
    failed_discuss_review_placeholder,
    failed_discuss_answer_placeholder,
    parse_structured_discuss_agenda,
    parse_structured_discuss_review,
    parse_structured_discuss_answer,
    parse_legacy_structured_discuss_answer,
)
from .unresolved_items import _apply_unresolved_item_dispositions

ROUND_RESUME_MARKER_RE = re.compile(r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", re.I)


@dataclass(frozen=True)
class PostedRoundMetadata:
    flow: str
    role: str
    agent: str
    round_number: int
    subject: str
    prior_items: tuple[UnresolvedReviewItem, ...] = ()
    dispositions: tuple[ReviewItemDisposition, ...] = ()
    new_items: tuple[UnresolvedReviewItem, ...] = ()
    state: str | None = None
    canonical_plan: str | None = None
    raw_structured_coder_response: str | None = None
    compact_prior_summaries: tuple[str, ...] = ()
    usage: dict | None = None
    model_used: str | None = None
    consensus_kind: str | None = None
    is_final: bool = False
    agenda: tuple[str, ...] = ()
    # Raw structured discuss-agenda response from the optional analyzer (#467).
    # None on final summaries, plain-mode rounds, and legacy comments.
    analyzer_response: str | None = None
    # Raw structured final-only analyzer response (#529).  Kept separate from
    # the prior-round agenda so resumes/audit cannot treat history as current.
    final_analyzer_response: str | None = None
    # Discuss research policy in effect when the comment was posted (#477).
    # None on non-discuss flows and legacy comments.
    research_mode: str | None = None
    # Debaters that failed or timed out in a partial round (#475), as
    # (display name, failure category) pairs on the round summary. Resume
    # treats these debaters' missing comments as accounted for.
    failed_debaters: tuple[tuple[str, str], ...] = ()
    # Merged discuss split proposals recorded on a final summary comment
    # (#476), so a resumed run can materialize them into child issues without
    # having to reconstruct them from debater comment metadata. Empty on
    # non-final, non-split, and legacy comments.
    split_proposals: tuple[str, ...] = ()
    # Missing metadata decodes as triage for legacy transcripts.
    result_mode: str = "triage"
    # Canonical bounded #535 evidence artifact on final summaries.  Keeping it
    # here makes resume idempotent without feeding evidence into analyzer agenda.
    evidence_reconciliation: dict | None = None


@dataclass(frozen=True)
class PostedRoundRecord:
    index: int
    metadata: PostedRoundMetadata
    body: str


@dataclass(frozen=True)
class ResumedRoundSelection:
    anchor_record: PostedRoundRecord
    current_round_records: tuple[PostedRoundRecord, ...]


@dataclass(frozen=True)
class ResumedReviewRound:
    round_number: int
    prior_items: tuple[UnresolvedReviewItem, ...]
    coder_output: str | None
    completed_reviews: tuple[PostedRoundRecord, ...]
    next_unresolved_item_number: int
    ledger_may_be_incomplete: bool = False
    compact_prior_summaries: tuple[str, ...] = ()
    unrecorded_head_advance: bool = False


def _serialize_unresolved_item(item: UnresolvedReviewItem) -> dict[str, object]:
    return {
        "item_id": item.item_id,
        "reviewer": item.reviewer,
        "source_round": item.source_round,
        "text": item.text,
        "status": item.status,
        "source_status": item.source_status,
        "notes": list(item.notes),
    }


def _deserialize_unresolved_item(payload: object) -> UnresolvedReviewItem:
    if not isinstance(payload, dict):
        raise AgentLoopError("Invalid round metadata unresolved-item payload.")
    raw_notes = payload.get("notes") or []
    notes = tuple(str(note) for note in raw_notes) if isinstance(raw_notes, list) else ()
    return UnresolvedReviewItem(
        item_id=str(payload["item_id"]),
        reviewer=str(payload["reviewer"]),
        source_round=int(payload["source_round"]),
        text=str(payload["text"]),
        status=str(payload["status"]),
        source_status=str(payload["source_status"]) if payload.get("source_status") is not None else None,
        notes=notes,
    )


def _serialize_disposition(disposition: ReviewItemDisposition) -> dict[str, object]:
    return {
        "item_id": disposition.item_id,
        "reviewer": disposition.reviewer,
        "disposition": disposition.disposition,
        "note": disposition.note,
    }


def _deserialize_disposition(payload: object) -> ReviewItemDisposition:
    if not isinstance(payload, dict):
        raise AgentLoopError("Invalid round metadata disposition payload.")
    return ReviewItemDisposition(
        item_id=str(payload["item_id"]),
        reviewer=str(payload["reviewer"]),
        disposition=str(payload["disposition"]),
        note=str(payload["note"]) if payload.get("note") is not None else None,
    )


def _encode_round_metadata(metadata: PostedRoundMetadata) -> str:
    payload = {
        "flow": metadata.flow,
        "role": metadata.role,
        "agent": metadata.agent,
        "round_number": metadata.round_number,
        "subject": metadata.subject,
        "prior_items": [_serialize_unresolved_item(item) for item in metadata.prior_items],
        "dispositions": [_serialize_disposition(item) for item in metadata.dispositions],
        "new_items": [_serialize_unresolved_item(item) for item in metadata.new_items],
        "state": metadata.state,
        "canonical_plan": metadata.canonical_plan,
        "raw_structured_coder_response": metadata.raw_structured_coder_response,
        "compact_prior_summaries": list(metadata.compact_prior_summaries),
        "usage": metadata.usage,
        "model_used": metadata.model_used,
        "consensus_kind": metadata.consensus_kind,
        "is_final": metadata.is_final,
        "agenda": list(metadata.agenda),
        "analyzer_response": metadata.analyzer_response,
        "final_analyzer_response": metadata.final_analyzer_response,
        "research_mode": metadata.research_mode,
        "failed_debaters": [list(pair) for pair in metadata.failed_debaters],
        "split_proposals": list(metadata.split_proposals),
        "result_mode": metadata.result_mode,
        "evidence_reconciliation": metadata.evidence_reconciliation,
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    return encoded.decode("ascii")


def _decode_round_metadata(encoded: str) -> PostedRoundMetadata:
    try:
        raw = base64.urlsafe_b64decode(encoded.encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise AgentLoopError("Invalid AGENT_LOOP_META payload.")
        return PostedRoundMetadata(
            flow=str(payload["flow"]),
            role=str(payload["role"]),
            agent=str(payload["agent"]),
            round_number=int(payload["round_number"]),
            subject=str(payload["subject"]),
            prior_items=tuple(_deserialize_unresolved_item(item) for item in payload.get("prior_items", [])),
            dispositions=tuple(_deserialize_disposition(item) for item in payload.get("dispositions", [])),
            new_items=tuple(_deserialize_unresolved_item(item) for item in payload.get("new_items", [])),
            state=str(payload["state"]) if payload.get("state") is not None else None,
            canonical_plan=(
                str(payload["canonical_plan"])
                if payload.get("canonical_plan") is not None
                else None
            ),
            raw_structured_coder_response=(
                str(payload["raw_structured_coder_response"])
                if payload.get("raw_structured_coder_response") is not None
                else None
            ),
            compact_prior_summaries=tuple(
                str(summary) for summary in payload.get("compact_prior_summaries", [])
            ),
            usage=payload.get("usage") if isinstance(payload.get("usage"), dict) else None,
            model_used=str(payload["model_used"]) if payload.get("model_used") is not None else None,
            consensus_kind=str(payload["consensus_kind"]) if payload.get("consensus_kind") is not None else None,
            is_final=bool(payload.get("is_final", False)),
            agenda=tuple(str(item) for item in payload.get("agenda", [])),
            analyzer_response=(
                str(payload["analyzer_response"])
                if payload.get("analyzer_response") is not None
                else None
            ),
            final_analyzer_response=(
                str(payload["final_analyzer_response"])
                if payload.get("final_analyzer_response") is not None
                else None
            ),
            research_mode=(
                str(payload["research_mode"])
                if payload.get("research_mode") is not None
                else None
            ),
            failed_debaters=tuple(
                (str(pair[0]), str(pair[1]))
                for pair in payload.get("failed_debaters", [])
                if isinstance(pair, (list, tuple)) and len(pair) == 2
            ),
            split_proposals=tuple(str(item) for item in payload.get("split_proposals", [])),
            result_mode=str(payload.get("result_mode", "triage")),
            evidence_reconciliation=(
                payload.get("evidence_reconciliation")
                if isinstance(payload.get("evidence_reconciliation"), dict)
                else None
            ),
        )
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise AgentLoopError("Invalid AGENT_LOOP_META payload.") from exc


def _attach_round_metadata(body: str, metadata: PostedRoundMetadata) -> str:
    marker = f"<!-- AGENT_LOOP_META: {_encode_round_metadata(metadata)} -->"
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
    prefix = "\n".join(lines[:metadata_start]).rstrip("\n")
    suffix = "\n".join(lines[metadata_start:]).lstrip("\n")
    if not prefix:
        return "\n".join(part for part in (marker, suffix) if part)
    if not suffix:
        return "\n".join((prefix, marker))
    return "\n".join((prefix, marker, suffix))


def _strip_round_metadata(body: str) -> str:
    cleaned = re.sub(
        r"\n?\s*<!--\s*AGENT_LOOP_META:\s*[A-Za-z0-9+/=_-]+\s*-->\s*\n?",
        "\n",
        body,
        flags=re.I,
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _extract_round_metadata_records(comments: Sequence[object], *, flow: str) -> tuple[PostedRoundRecord, ...]:
    records: list[PostedRoundRecord] = []
    for index, comment in enumerate(comments):
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        matches = list(ROUND_RESUME_MARKER_RE.finditer(body))
        if not matches:
            continue
        metadata = _decode_round_metadata(matches[-1].group("payload"))
        if metadata.flow != flow:
            continue
        records.append(
            PostedRoundRecord(
                index=index,
                metadata=metadata,
                body=_strip_round_metadata(body),
            )
        )
    return tuple(records)


def _prior_item_ledger_signature(items: Sequence[UnresolvedReviewItem]) -> tuple[tuple[str, str, int, str, str, str | None, tuple[str, ...]], ...]:
    return tuple(
        (
            item.item_id,
            item.reviewer,
            item.source_round,
            item.text,
            item.status,
            item.source_status,
            item.notes,
        )
        for item in items
    )


def _select_current_round_records(
    records: Sequence[PostedRoundRecord],
    *,
    subject: str,
) -> ResumedRoundSelection | None:
    subject_records = [record for record in records if record.metadata.subject == subject]
    if not subject_records:
        return None
    anchor_record = subject_records[-1]
    anchor_metadata = anchor_record.metadata
    prior_items_signature = _prior_item_ledger_signature(anchor_metadata.prior_items)
    current_round_records = tuple(
        record
        for record in subject_records
        if record.metadata.round_number == anchor_metadata.round_number
        and _prior_item_ledger_signature(record.metadata.prior_items) == prior_items_signature
    )
    latest_coder_record = next(
        (
            record
            for record in reversed(current_round_records)
            if record.metadata.role == "coder"
        ),
        None,
    )
    if latest_coder_record is not None:
        current_round_records = tuple(
            record for record in current_round_records if record.index >= latest_coder_record.index
        )
    return ResumedRoundSelection(
        anchor_record=anchor_record,
        current_round_records=current_round_records,
    )


def _max_unresolved_item_number_from_records(records: Sequence[PostedRoundRecord]) -> int:
    max_number = 0
    for record in records:
        for item in (*record.metadata.prior_items, *record.metadata.new_items):
            match = re.fullmatch(r"item-(\d+)", item.item_id)
            if match:
                max_number = max(max_number, int(match.group(1)))
    return max_number


def _active_pr_items(items: Sequence[UnresolvedReviewItem]) -> list[UnresolvedReviewItem]:
    return [item for item in items if item.status in {"blocking", "same-pr"}]


def _append_active_pr_new_items(
    active_items: list[UnresolvedReviewItem],
    records: Sequence[PostedRoundRecord],
) -> None:
    seen_item_ids = {item.item_id for item in active_items}
    for record in records:
        for item in record.metadata.new_items:
            if item.status not in {"blocking", "same-pr"} or item.item_id in seen_item_ids:
                continue
            active_items.append(item)
            seen_item_ids.add(item.item_id)


def _aggregate_record_dispositions(
    records: Sequence[PostedRoundRecord],
) -> dict[str, list[ReviewItemDisposition]]:
    dispositions_by_item: dict[str, list[ReviewItemDisposition]] = {}
    for record in records:
        for disposition in record.metadata.dispositions:
            dispositions_by_item.setdefault(disposition.item_id, []).append(disposition)
    return dispositions_by_item


def _recover_unrecorded_pr_head_advance(
    records: Sequence[PostedRoundRecord],
    *,
    head_sha: str,
) -> ResumedReviewRound | None:
    prior_records = [record for record in records if record.metadata.subject != head_sha]
    if not prior_records:
        return None
    latest_prior_subject = prior_records[-1].metadata.subject
    selection = _select_current_round_records(records, subject=latest_prior_subject)
    if selection is None:
        return None

    current_round_records = selection.current_round_records
    anchor_metadata = selection.anchor_record.metadata
    latest_coder_record = next(
        (
            record
            for record in reversed(current_round_records)
            if record.metadata.role == "coder"
        ),
        None,
    )
    reviewer_records_after_coder = tuple(
        record
        for record in current_round_records
        if record.metadata.role == "reviewer"
        and (latest_coder_record is None or record.index > latest_coder_record.index)
    )
    all_reviewer_records = tuple(
        record for record in current_round_records if record.metadata.role == "reviewer"
    )

    if latest_coder_record is not None and not reviewer_records_after_coder:
        round_number = latest_coder_record.metadata.round_number
        recovered_items = _active_pr_items(latest_coder_record.metadata.prior_items)
        coder_output = latest_coder_record.metadata.raw_structured_coder_response or latest_coder_record.body
        compact_prior_summaries = latest_coder_record.metadata.compact_prior_summaries
    elif latest_coder_record is not None:
        round_number = latest_coder_record.metadata.round_number
        recovered_items, _future_items = _apply_unresolved_item_dispositions(
            latest_coder_record.metadata.prior_items,
            _aggregate_record_dispositions(reviewer_records_after_coder),
            retain_future=False,
        )
        recovered_items = _active_pr_items(recovered_items)
        _append_active_pr_new_items(recovered_items, reviewer_records_after_coder)
        coder_output = latest_coder_record.metadata.raw_structured_coder_response or latest_coder_record.body
        compact_prior_summaries = latest_coder_record.metadata.compact_prior_summaries
    elif all_reviewer_records:
        round_number = anchor_metadata.round_number
        recovered_items, _future_items = _apply_unresolved_item_dispositions(
            anchor_metadata.prior_items,
            _aggregate_record_dispositions(all_reviewer_records),
            retain_future=False,
        )
        recovered_items = _active_pr_items(recovered_items)
        _append_active_pr_new_items(recovered_items, all_reviewer_records)
        coder_output = None
        compact_prior_summaries = ()
    else:
        return None

    if not recovered_items:
        return None

    return ResumedReviewRound(
        round_number=round_number,
        prior_items=tuple(recovered_items),
        coder_output=coder_output,
        completed_reviews=(),
        next_unresolved_item_number=_max_unresolved_item_number_from_records(records) + 1,
        ledger_may_be_incomplete=True,
        compact_prior_summaries=compact_prior_summaries,
        unrecorded_head_advance=True,
    )


def _latest_prior_pr_subject_is_coherent(records: Sequence[PostedRoundRecord], *, head_sha: str) -> bool:
    prior_records = [record for record in records if record.metadata.subject != head_sha]
    if not prior_records:
        return True
    latest_prior_subject = prior_records[-1].metadata.subject
    selection = _select_current_round_records(records, subject=latest_prior_subject)
    if selection is None:
        return False
    return any(
        record.metadata.role in {"coder", "reviewer"}
        for record in selection.current_round_records
    )


def _plan_subject(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _resume_pr_round(
    comments: Sequence[object],
    *,
    head_sha: str | None,
    configured_reviewers: Sequence[AgentName],
) -> ResumedReviewRound | None:
    if not head_sha:
        return None
    records = _extract_round_metadata_records(comments, flow="pr")
    if not records:
        return None
    selection = _select_current_round_records(records, subject=head_sha)
    if selection is None:
        recovered = _recover_unrecorded_pr_head_advance(records, head_sha=head_sha)
        if recovered is not None:
            return recovered
        if _latest_prior_pr_subject_is_coherent(records, head_sha=head_sha):
            return None
        latest_prior_subject = records[-1].metadata.subject
        raise AgentLoopError(
            "PR head advanced without a recorded coder follow-up and the metadata-backed "
            "handoff could not be recovered safely. "
            f"Current head: {head_sha}. Latest recorded metadata subject: {latest_prior_subject}. "
            "Rerun after posting a valid structured coder follow-up for the current head "
            "or repair the metadata-backed handoff."
        )
    current_round_records = selection.current_round_records
    anchor_metadata = selection.anchor_record.metadata
    latest_coder_record = next(
        (
            record
            for record in reversed(current_round_records)
            if record.metadata.role == "coder"
        ),
        None,
    )
    reviewer_records: dict[str, PostedRoundRecord] = {}
    configured_reviewer_names = {agent_display_name(agent) for agent in configured_reviewers}
    for record in current_round_records:
        metadata = record.metadata
        if metadata.role != "reviewer" or metadata.agent not in configured_reviewer_names:
            continue
        reviewer_records[metadata.agent] = record
    if latest_coder_record is None and not reviewer_records:
        return None
    prior_items = anchor_metadata.prior_items
    round_number = anchor_metadata.round_number
    ledger_may_be_incomplete = (
        len(anchor_metadata.prior_items) == 0
        and any(
            record.metadata.new_items
            for record in records
            if record.metadata.subject == anchor_metadata.subject
            and record.metadata.round_number < anchor_metadata.round_number
        )
    )
    return ResumedReviewRound(
        round_number=round_number,
        prior_items=prior_items,
        coder_output=(
            latest_coder_record.metadata.raw_structured_coder_response
            or latest_coder_record.body
            if latest_coder_record is not None
            else None
        ),
        completed_reviews=tuple(reviewer_records[agent_display_name(agent)] for agent in configured_reviewers if agent_display_name(agent) in reviewer_records),
        next_unresolved_item_number=_max_unresolved_item_number_from_records(
            [record for record in records if record.metadata.subject == head_sha]
        )
        + 1,
        ledger_may_be_incomplete=ledger_may_be_incomplete,
        compact_prior_summaries=(
            latest_coder_record.metadata.compact_prior_summaries
            if latest_coder_record is not None
            else ()
        ),
    )


def _resume_plan_round(
    comments: Sequence[object],
    *,
    configured_reviewers: Sequence[AgentName],
) -> tuple[str, ResumedReviewRound] | None:
    records = _extract_round_metadata_records(comments, flow="plan")
    if not records:
        return None
    latest_coder_record = next((record for record in reversed(records) if record.metadata.role == "coder"), None)
    if latest_coder_record is None:
        return None
    selection = _select_current_round_records(records, subject=latest_coder_record.metadata.subject)
    if selection is None:
        return None
    current_round_records = selection.current_round_records
    anchor_metadata = selection.anchor_record.metadata
    current_plan = latest_coder_record.metadata.canonical_plan or latest_coder_record.body
    coder_output = latest_coder_record.metadata.raw_structured_coder_response or current_plan
    ledger_may_be_incomplete = (
        len(anchor_metadata.prior_items) == 0
        and any(
            record.metadata.new_items
            for record in records
            if record.metadata.subject == anchor_metadata.subject
            and record.metadata.round_number < anchor_metadata.round_number
        )
    )
    reviewer_records: dict[str, PostedRoundRecord] = {}
    configured_reviewer_names = {agent_display_name(agent) for agent in configured_reviewers}
    for record in current_round_records:
        metadata = record.metadata
        if metadata.role != "reviewer" or metadata.agent not in configured_reviewer_names:
            continue
        reviewer_records[metadata.agent] = record
    return (
        current_plan,
        ResumedReviewRound(
            round_number=anchor_metadata.round_number,
            prior_items=anchor_metadata.prior_items,
            coder_output=coder_output,
            completed_reviews=tuple(reviewer_records[agent_display_name(agent)] for agent in configured_reviewers if agent_display_name(agent) in reviewer_records),
            next_unresolved_item_number=_max_unresolved_item_number_from_records(
                [record for record in records if record.metadata.subject == anchor_metadata.subject]
            )
            + 1,
            ledger_may_be_incomplete=ledger_may_be_incomplete,
            compact_prior_summaries=latest_coder_record.metadata.compact_prior_summaries,
        ),
    )


@dataclass(frozen=True)
class ResumedDiscussState:
    done: bool
    round_history: tuple[tuple[ParsedDiscussResponse, ...], ...]
    next_round_number: int
    prior_round_agenda: tuple[str, ...] = ()
    prior_analyzer_agenda: ParsedDiscussAgenda | None = None
    in_progress_votes: dict[str, ParsedDiscussResponse] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.in_progress_votes is None:
            object.__setattr__(self, "in_progress_votes", {})


def _decode_analyzer_agenda(raw: str | None) -> ParsedDiscussAgenda | None:
    """Reparse a persisted analyzer agenda; corrupt/legacy payloads resume in plain mode."""
    if not raw:
        return None
    try:
        return parse_structured_discuss_agenda(raw)
    except AgentLoopError:
        return None


def _decode_discuss_vote(record: PostedRoundRecord, *, round_number: int) -> ParsedDiscussResponse:
    text = record.metadata.raw_structured_coder_response or record.body
    if record.metadata.result_mode == "answer":
        try:
            vote = parse_structured_discuss_answer(
                text, reviewer=record.metadata.agent, round_number=round_number
            )
        except AgentLoopError:
            # Only persisted round metadata gets the explicitly opt-in legacy
            # decoder. Its exact-key validation still rejects malformed/mixed
            # payloads rather than hiding a corrupt transcript.
            vote = parse_legacy_structured_discuss_answer(
                text, reviewer=record.metadata.agent, round_number=round_number
            )
    else:
        vote = parse_structured_discuss_review(text, reviewer=record.metadata.agent, round_number=round_number)
    if vote is None:
        raise AgentLoopError(
            f"Could not decode discuss response metadata for {record.metadata.agent} "
            f"(round {round_number}); the posted comment's structured payload is missing "
            "or invalid."
        )
    return vote


def _resume_discuss_round(
    comments: Sequence[object],
    *,
    subject: str,
    configured_reviewers: Sequence[AgentName],
    result_mode: str = "triage",
) -> ResumedDiscussState | None:
    records = _extract_round_metadata_records(comments, flow="discuss")
    subject_records = [record for record in records if record.metadata.subject == subject]
    if not subject_records:
        return None
    stored_modes = {record.metadata.result_mode for record in subject_records}
    if stored_modes != {result_mode}:
        raise AgentLoopError(
            "Discuss transcript result mode conflicts with the requested mode; "
            "use the same --discuss-result-mode used to create the transcript."
        )
    configured_reviewer_names = [agent_display_name(agent) for agent in configured_reviewers]
    rounds: dict[int, list[PostedRoundRecord]] = {}
    for record in subject_records:
        rounds.setdefault(record.metadata.round_number, []).append(record)

    round_history: list[tuple[ParsedDiscussResponse, ...]] = []
    prior_round_agenda: tuple[str, ...] = ()
    prior_analyzer_agenda: ParsedDiscussAgenda | None = None
    next_round_number = 1
    in_progress_votes: dict[str, ParsedDiscussResponse] = {}
    for round_number in sorted(rounds):
        round_records = rounds[round_number]
        summary_record = next(
            (record for record in round_records if record.metadata.role == "summary"), None
        )
        debater_records: dict[str, PostedRoundRecord] = {}
        for record in round_records:
            if record.metadata.role == "debater":
                debater_records[record.metadata.agent] = record
        if summary_record is not None:
            # Debaters recorded as failed in a partial round (#475) post no
            # comment; their summary metadata accounts for the gap.
            failed_by_name = dict(summary_record.metadata.failed_debaters)
            missing = [
                name
                for name in configured_reviewer_names
                if name not in debater_records and name not in failed_by_name
            ]
            if missing:
                raise AgentLoopError(
                    "Discuss round metadata is inconsistent: round "
                    f"{round_number} has a summary comment but is missing debater "
                    f"comments for: {', '.join(missing)}."
                )
            votes = tuple(
                _decode_discuss_vote(debater_records[name], round_number=round_number)
                if name in debater_records
                else (
                    failed_discuss_answer_placeholder(name, failed_by_name[name])
                    if result_mode == "answer"
                    else failed_discuss_review_placeholder(name, failed_by_name[name])
                )
                for name in configured_reviewer_names
            )
            round_history.append(votes)
            prior_round_agenda = summary_record.metadata.agenda
            prior_analyzer_agenda = _decode_analyzer_agenda(summary_record.metadata.analyzer_response)
            if summary_record.metadata.is_final:
                return ResumedDiscussState(
                    done=True,
                    round_history=tuple(round_history),
                    next_round_number=round_number,
                    prior_round_agenda=prior_round_agenda,
                    prior_analyzer_agenda=prior_analyzer_agenda,
                )
            next_round_number = round_number + 1
        else:
            next_round_number = round_number
            for name, record in debater_records.items():
                in_progress_votes[name] = _decode_discuss_vote(record, round_number=round_number)
            break
    return ResumedDiscussState(
        done=False,
        round_history=tuple(round_history),
        next_round_number=next_round_number,
        prior_round_agenda=prior_round_agenda,
        prior_analyzer_agenda=prior_analyzer_agenda,
        in_progress_votes=in_progress_votes,
    )
