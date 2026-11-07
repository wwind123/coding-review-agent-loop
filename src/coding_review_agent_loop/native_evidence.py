"""Conservative native evidence matching and provisional authorization."""
from __future__ import annotations
import re
import hashlib
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING
from dataclasses import replace
from .protocol import EVIDENCE_OBLIGATION_KIND

if TYPE_CHECKING:
    from .protocol import UnresolvedReviewItem
    from .round_metadata import (
        PostedRoundRecord,
    )

HEX_TOKEN = re.compile(r"(?<![a-zA-Z0-9_])[0-9a-fA-F]{7,40}(?![a-zA-Z0-9_])")


def known_pr_heads(records, current=None):
    """Only admitted PR subjects, never SHA mentions in untrusted prose."""
    return tuple(sorted({h.lower() for h in [current, *(r.metadata.subject for r in records if r.metadata.flow == "pr")]
                         if h and re.fullmatch(r"[0-9a-fA-F]{40}", h)}))


def native_history(runner, config, pr_number, comments):
    """Admit head subjects and requester votes from the orchestrator's actor."""
    from .github import merge_pr_comment_transport_identity, resolve_authenticated_github_actor
    from .round_state import _extract_round_metadata_records, _recovery_record_admitted
    if config.dry_run or not comments:
        return ()
    raw = _extract_round_metadata_records(comments, flow="pr")
    if not any(r.metadata.evidence_requests or any(native(i) for i in (*r.metadata.prior_items, *r.metadata.new_items)) for r in raw):
        return ()
    hydrated = tuple(merge_pr_comment_transport_identity(
        runner, config=config, pr_number=pr_number, comments=comments))
    actor = resolve_authenticated_github_actor(runner, config=config)
    return tuple(r for r in _extract_round_metadata_records(hydrated, flow="pr")
                 if _recovery_record_admitted(r, hydrated, actor))


def recognized_head(token, heads):
    matches = {h.lower() for h in heads if h.lower().startswith(token.lower())}
    return next(iter(matches)) if len(matches) == 1 else None


def names_head(text, head, heads):
    return any(recognized_head(m.group(), heads) == head.lower() for m in HEX_TOKEN.finditer(text))


def request_key(text, heads):
    """Tagged segments prevent literal prose from colliding with a head placeholder."""
    from .protocol import _normalized_review_item_text
    normalized = _normalized_review_item_text(text)
    parts, start = [], 0
    for match in HEX_TOKEN.finditer(normalized):
        if recognized_head(match.group(), heads):
            parts.extend((("text", normalized[start:match.start()]), ("head", "")))
            start = match.end()
    return (*parts, ("text", normalized[start:]))


def contextual_identity(text, reviewer, heads):
    from .unresolved_items import evidence_obligation_identity
    key = request_key(text, heads)
    if not any(kind == "head" for kind, _ in key):
        return evidence_obligation_identity(text, reviewer)
    digest = hashlib.sha256(json.dumps(["native-head-context-v1", key, reviewer],
                                     ensure_ascii=False).encode("utf-8")).hexdigest()[:24]
    return f"{EVIDENCE_OBLIGATION_KIND}:{digest}"


def native(item):
    return item.obligation_kind == EVIDENCE_OBLIGATION_KIND and item.converted_row_id is None and not item.screenshot_request


def open_native(item):
    return native(item) and item.status in {"blocking", "same-pr"} and item.lifecycle in {"evidence_deferred", "evidence_frozen"}


def identities(item):
    return tuple(dict.fromkeys((item.obligation_identity, *item.alias_obligation_identities)))


def merge_items(items, text=None, identity=None):
    ordered = sorted(items, key=lambda i: int(i.item_id.split("-")[-1]))
    first = ordered[0]
    return replace(first, text=text if text is not None else first.text,
        alias_item_ids=tuple(dict.fromkeys(a for i in ordered for a in (i.item_id, *i.alias_item_ids) if a != first.item_id)),
        alias_obligation_identities=tuple(dict.fromkeys(a for a in (
            *(a for i in ordered for a in identities(i)), identity) if a and a != first.obligation_identity)),
        notes=tuple(dict.fromkeys(n for i in ordered for n in i.notes)))


def consolidate_ledger(items, heads):
    result = []
    for item in items:
        if open_native(item):
            item = merge_items([item], identity=contextual_identity(item.text, item.reviewer, heads))
        match = next((i for i, prior in enumerate(result) if open_native(item) and open_native(prior)
                      and item.reviewer == prior.reviewer and request_key(item.text, heads) == request_key(prior.text, heads)), None)
        if match is None:
            result.append(item)
        else:
            result[match] = merge_items([result[match], item], text=item.text)
    return result


# Only an affirmative imperative clause with a bounded item list.
CONSOLIDATE = re.compile(r"(?:^|[.!?]\s+)consolidate\s+(item-\d+(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+)item-\d+)*)\s+into\s+(?:this|one|a single|the existing)\b", re.I)


def consolidation_targets(text, items, reviewer):
    matches = list(CONSOLIDATE.finditer(text))
    if len(matches) != 1:
        return []
    ids = re.findall(r"item-\d+", matches[0].group(1))
    if len(ids) > 12 or len(ids) != len(set(ids)):
        return []
    targets = []
    for item_id in ids:
        candidates = [i for i in items if item_id in (i.item_id, *i.alias_item_ids)]
        if len(candidates) != 1 or not open_native(candidates[0]) or candidates[0].reviewer != reviewer:
            return []
        if candidates[0] not in targets:
            targets.append(candidates[0])
    return targets


def citation_ids(value):
    from .errors import AgentLoopError
    if not isinstance(value, list) or any(not isinstance(i, str) or not i.strip() for i in value) or len(value) != len(set(value)):
        raise AgentLoopError("evidence_requirement_ids must be an array of distinct non-empty strings")
    return tuple(value)


def provisional_clearances(items, records, requirements, *, head, round_number, board):
    """Use this round's latest requester attestations and effective live signed objects.

    The existing trust policy owns requirement admission. A resolved citation
    attests adequacy; a SHA mention alone never authorizes acceptance.
    """
    from .review_scope import is_complete_code_review
    latest = {}
    for record in records:
        m = record.metadata
        if m.role == "reviewer" and is_complete_code_review(m) and m.subject == head and m.round_number == round_number:
            latest[m.agent] = m
    heads = known_pr_heads(records, head)
    live = {r.requirement_id: r for r in requirements}
    kept, cleared = [], []
    for item in items:
        required = {item.reviewer} if item.reviewer in board else set(board)
        accepted = set()
        if open_native(item):
            for reviewer in required:
                review = latest.get(reviewer)
                if review is None:
                    continue
                votes = [d for d in review.dispositions if d.item_id in (item.item_id, *item.alias_item_ids)]
                if not votes or any(d.disposition != "resolved" for d in votes):
                    continue
                if all(d.evidence_requirement_ids and all(
                    rid in review.surfaced_reviewer_requirement_ids and rid in live
                    and names_head(live[rid].body, head, heads)
                    for rid in d.evidence_requirement_ids) for d in votes):
                    accepted.add(reviewer)
        if required and required <= accepted:
            cleared.extend((identity, head) for identity in identities(item) if identity)
        else:
            kept.append(item)
    return kept, cleared


def _append_recorded_evidence_requests(
    active_items: list[UnresolvedReviewItem],
    reviewer_records: Sequence[PostedRoundRecord],
    *,
    records: Sequence[PostedRoundRecord],
    admitted_records: Sequence[PostedRoundRecord] | None = None,
) -> None:
    """Re-derive deferred evidence from the reviewer records that requested it.

    Applied after dispositions and new items, in the same order as the live
    loop, so a deferred request is durable from the moment its reviewer record
    was posted (#1068).
    """
    from .round_state import _max_unresolved_item_number_from_records
    from .unresolved_items import _upsert_evidence_obligation
    admitted_records = records if admitted_records is None else admitted_records
    admitted_indexes = {r.index for r in admitted_records}
    boundary = max((r.index for r in reviewer_records), default=-1)
    next_number = max(
        _max_unresolved_item_number_from_records([r for r in records if r.index <= boundary]),
        max(
            (
                int(match.group(1))
                for item in active_items
                if (match := re.fullmatch(r"item-(\d+)", item.item_id))
            ),
            default=0,
        ),
    ) + 1
    for record in reviewer_records:
        for request in record.metadata.evidence_requests:
            updated, consumed = _upsert_evidence_obligation(
                active_items,
                item_number=next_number,
                reviewer=record.metadata.agent,
                text=request,
                source_round=record.metadata.round_number,
                current_head_sha=record.metadata.subject if record.index in admitted_indexes else None,
                known_heads=known_pr_heads([r for r in admitted_records if r.index <= record.index]),
            )
            active_items[:] = updated
            if consumed:
                next_number += 1



def restore_for_new_input(items, resume, requirements, records, next_number):
    """A reviewer invalidated by new input must still see its durable request."""
    if resume is None or resume.reconciled:
        return list(items), next_number
    current_ids = {r.requirement_id for r in requirements}
    reviews = [r for r in resume.completed_reviews
               if set(r.metadata.surfaced_reviewer_requirement_ids) != current_ids]
    result = list(items)
    _append_recorded_evidence_requests(result, reviews, records=records)
    return result, max(next_number, max((int(i.item_id.split("-")[-1]) + 1
        for i in result if re.fullmatch(r"item-\d+", i.item_id)), default=1))


def publication_dispositions(parsed, text, reviewer, items):
    """Keep code approval normalized, but revoke conflicting alias attestations.

    Native blocking votes remain in the reconciled ledger's notes. They must
    not turn the durable code approval into a contradictory approval carrier.
    A conflicting resolved citation loses its authorization on publication.
    """
    from .protocol import parse_review, parse_structured_pr_review
    carrier = parse_structured_pr_review(text, reviewer=reviewer, architecture_status_mode="legacy")
    raw = (carrier or parse_review(text, reviewer=reviewer)).dispositions
    conflicts = set()
    for item in items:
        if not native(item):
            continue
        keys = {item.item_id, *item.alias_item_ids}
        votes = [d for d in raw if d.item_id in keys]
        if any(d.disposition == "resolved" for d in votes) and any(d.disposition != "resolved" for d in votes):
            conflicts.update(keys)
    return tuple(replace(d, evidence_requirement_ids=(),
                         note=(d.note or "") + " Conflicting native alias votes do not authorize acceptance.")
                 if d.item_id in conflicts else d for d in parsed.dispositions)


def reopen_cleared_requests(items, boundary, records, head, *, admitted_records):
    """A release authorizes only its own head; restore its native requests on push."""
    prior = [r for r in records if r.index <= boundary.index]
    admitted_prior = [r for r in admitted_records if r.index <= boundary.index]
    old_head = boundary.metadata.subject
    cleared = {identity for identity, h in boundary.metadata.evidence_clearances if h == old_head}
    if not cleared:
        return list(items)
    # Requests are ingested after snapshots. A same-response clearance may
    # leave only the requester record carrying the consolidated lineage.
    groups = {}
    for record in prior:
        groups.setdefault((record.metadata.subject, record.metadata.round_number), []).append(record)
    candidates = []
    for group in groups.values():
        snapshots = {item.item_id: item for record in group
                     for item in (*record.metadata.prior_items, *record.metadata.new_items)
                     if native(item) and set(identities(item)) & cleared}
        candidates = [snapshots.pop(item.item_id, item) for item in candidates]
        candidates.extend(snapshots.values())
        retired = {alias for item in candidates for alias in item.alias_item_ids}
        candidates = [item for item in candidates if item.item_id not in retired]
        _append_recorded_evidence_requests(candidates,
            [r for r in group if r.metadata.role == "reviewer"], records=prior,
            admitted_records=admitted_prior)
        candidates = consolidate_ledger(candidates, known_pr_heads(
            [r for r in admitted_prior if r.index <= max(g.index for g in group)]))
    restored = [item for item in candidates if native(item) and set(identities(item)) & cleared]
    result = list(items)
    present = {identity for item in result for identity in identities(item)}
    for item in restored:
        if not set(identities(item)) & present:
            result.append(replace(item, status="blocking", source_status="blocking",
                lifecycle="evidence_deferred", candidate_head_sha=None))
            present.update(identities(item))
    return result
