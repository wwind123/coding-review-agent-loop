"""One record-derived no-progress detector for the review loops (#2067, #2097).

The detector replaces the PR step-back sibling, non-convergence,
approach-alternative and unchanged-head stops with one rule: stop when the K
most recent completed rounds share the same subject (the PR head SHA, or the
canonical plan hash) and the same non-empty set of blocking item IDs, and the
live candidate still equals that repeated observation.

Everything here is pure.  Observations are derived only from persisted round
records (plus, at the coder-dispatch site, the just-settled current round), so
a fresh run and a resumed run reach the same decision for the same history.
The unreviewed live candidate is never counted as a completed round: a head
that moved after the last review always goes to review first.

Legacy step-back bookkeeping (``step_back_entries``/``step_back_status``) is
never read here; tests/test_no_progress.py guards that statically.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .errors import render_human_decision
from .review_scope import is_complete_code_review
from .round_closure import (
    closed_round_reviewer_indexes,
    is_round_closure_boundary,
    is_unvalidated_closure_summary,
)

if TYPE_CHECKING:
    from .round_metadata import PostedRoundRecord


DEFAULT_NO_PROGRESS_ROUNDS = 3

# Leading bytes of the no-progress stop after the ``PR #N: `` prefix; history
# export recognizes them as ``pr-no-progress``.
NO_PROGRESS_PREFIX_TEXT = "human decision required: no progress across "

NO_PROGRESS_SHOWN_ITEMS = 20
NO_PROGRESS_ITEM_TEXT_LIMIT = 240
NO_PROGRESS_EXCERPT_LIMIT = 2000

_BLOCKING_STATUSES = frozenset({"blocking", "same-pr"})


@dataclass(frozen=True)
class NoProgressObservation:
    """One completed round: its subject and the blocking IDs it settled on."""

    round_number: int
    subject: str
    blocking_ids: frozenset[str]


@dataclass(frozen=True)
class NoProgressDecision:
    """The repeated observation that stops the run."""

    subject: str
    blocking_ids: frozenset[str]
    rounds: tuple[int, ...]
    k: int


def is_blocking_ledger_item(item: object) -> bool:
    """Whether a ledger item counts toward the no-progress blocking set.

    Blocking and same-PR findings count.  Future items, inert tombstones and
    converted findings, and machine or human exact-head obligations (CI
    failures, evidence requests, requirement acknowledgements) do not, which
    matches the reviewer-owned mandatory set step-back tracking uses.
    """
    if getattr(item, "status", None) not in _BLOCKING_STATUSES:
        return False
    if getattr(item, "authority", None) is not None:
        return False
    if getattr(item, "obligation_kind", None) is not None:
        return False
    return not bool(getattr(item, "is_inert_ledger_record", False))


def blocking_item_ids(items: Iterable[object]) -> frozenset[str]:
    """The no-progress blocking ID set of a ledger."""
    return frozenset(
        str(item.item_id) for item in items if is_blocking_ledger_item(item)  # type: ignore[attr-defined]
    )


def no_progress_decision(
    observations: Iterable[NoProgressObservation],
    *,
    live_subject: str | None,
    live_blocking_ids: Iterable[str],
    k: int,
    through_round: int | None = None,
) -> NoProgressDecision | None:
    """The stop decision, or ``None`` to continue.

    Fires only when the last ``k`` observations have contiguous round numbers
    (ending at ``through_round`` when given), the same non-empty subject and
    the same non-empty blocking ID set, and the live candidate equals that
    repeated observation.  The live candidate is compared, never appended.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        return None
    by_round: dict[int, NoProgressObservation] = {}
    for observation in observations:
        by_round[observation.round_number] = observation
    ordered = [by_round[number] for number in sorted(by_round)]
    if len(ordered) < k:
        return None
    window = ordered[-k:]
    if through_round is not None and window[-1].round_number != through_round:
        return None
    if any(
        later.round_number != earlier.round_number + 1
        for earlier, later in zip(window, window[1:])
    ):
        return None
    first = window[0]
    if not first.subject or not first.blocking_ids:
        return None
    if any(
        observation.subject != first.subject or observation.blocking_ids != first.blocking_ids
        for observation in window[1:]
    ):
        return None
    if live_subject != first.subject or frozenset(live_blocking_ids) != first.blocking_ids:
        return None
    return NoProgressDecision(
        subject=first.subject,
        blocking_ids=first.blocking_ids,
        rounds=tuple(observation.round_number for observation in window),
        k=k,
    )


def pr_no_progress_observations(
    records: Sequence["PostedRoundRecord"],
    *,
    current: NoProgressObservation | None = None,
) -> tuple[NoProgressObservation, ...]:
    """One observation per completed PR round, derived from durable records.

    Round N is observed through the newest PR coder record numbered N+1: it
    carries the settled round-end ledger (``prior_items``) and the head the
    follow-up was dispatched against (``followup_dispatch_head``).  Round N is
    not an observation, which breaks any window through it, when:

    * round N has a closure boundary or an unvalidated closure summary;
    * round N has no complete, non-inert review posted before that coder record;
    * there is no coder record N+1;
    * the dispatch head differs from round N's reviewed head; or
    * a historical record without a dispatch head has no single reviewed head.

    ``current`` is the just-settled round at the coder-dispatch site, which has
    no coder record yet; it replaces any durable observation of its round
    unless that round number carries a closure boundary.
    """
    ordered = sorted(
        (record for record in records if getattr(record.metadata, "flow", None) == "pr"),
        key=lambda record: record.index,
    )
    inert = closed_round_reviewer_indexes(ordered, flow="pr")
    closed_rounds = {
        record.metadata.round_number
        for record in ordered
        if is_round_closure_boundary(record.metadata)
        or is_unvalidated_closure_summary(record.metadata)
    }
    reviews: dict[int, list["PostedRoundRecord"]] = {}
    coders: dict[int, "PostedRoundRecord"] = {}
    for record in ordered:
        metadata = record.metadata
        if metadata.role == "coder":
            coders[metadata.round_number] = record
        elif is_complete_code_review(metadata) and record.index not in inert:
            reviews.setdefault(metadata.round_number, []).append(record)
    observations: dict[int, NoProgressObservation] = {}
    for coder_round, coder in coders.items():
        review_round = coder_round - 1
        if review_round < 1 or review_round in closed_rounds:
            continue
        round_reviews = [
            review for review in reviews.get(review_round, ()) if review.index < coder.index
        ]
        if not round_reviews:
            continue
        subjects = {str(review.metadata.subject or "") for review in round_reviews}
        dispatch_head = getattr(coder.metadata, "followup_dispatch_head", None)
        if dispatch_head:
            if subjects != {dispatch_head}:
                continue
            subject = str(dispatch_head)
        else:
            if len(subjects) != 1:
                continue
            (subject,) = subjects
            if not subject:
                continue
        observations[review_round] = NoProgressObservation(
            review_round, subject, blocking_item_ids(coder.metadata.prior_items)
        )
    if current is not None and current.round_number not in closed_rounds:
        # A closure boundary excludes its round number at both sites, so a
        # replacement review that reuses the number (a legacy supersession)
        # counts neither live nor on resume.
        observations[current.round_number] = current
    return tuple(observations[number] for number in sorted(observations))


def defers_to_evidence_stall(
    item_rows: Mapping[str, Sequence[str]], unsatisfied_rows: frozenset[str] | None
) -> bool:
    """Whether the evidence-stall stop may own this window, so it decides first.

    True only when every repeated item is a finding tagged with rows that are
    all unsatisfied at the dispatch head.  Any untagged item, a tag naming a
    satisfied row, or no usable row set means the stall can never own it.
    """
    if not unsatisfied_rows or not item_rows:
        return False
    return all(rows and set(rows) <= unsatisfied_rows for rows in item_rows.values())


def _clean_text(text: object, limit: int) -> str:
    cleaned = " ".join(str(text).split()).replace("`", "'")
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1].rstrip() + "…"
    return cleaned


def _first_line(text: object) -> str:
    return next((part.strip() for part in str(text).splitlines() if part.strip()), "")


def render_no_progress_item_lines(
    item_ids: Iterable[str], item_texts: Mapping[str, str] | None = None
) -> tuple[str, ...]:
    """Sorted, bounded ``- [id] first line`` entries with an omitted-count suffix."""
    texts = item_texts or {}
    ordered = sorted(item_ids)
    lines = [
        f"- [{_clean_text(item_id, 80)}] "
        + (_clean_text(_first_line(texts.get(item_id, "")), NO_PROGRESS_ITEM_TEXT_LIMIT)
           or "(no text recorded)")
        for item_id in ordered[:NO_PROGRESS_SHOWN_ITEMS]
    ]
    if len(ordered) > NO_PROGRESS_SHOWN_ITEMS:
        lines.append(f"- and {len(ordered) - NO_PROGRESS_SHOWN_ITEMS} more")
    return tuple(lines)


def citation_diagnosis_sentence(diagnosis: object) -> str:
    """The non-array ``test_observations`` sentence (#1626), or ``""``."""
    if diagnosis is None or not getattr(diagnosis, "non_array_drop", False):
        return ""
    count = getattr(diagnosis, "uncited_receipts", None)
    receipts = (
        f"{int(count)} run-tests receipt(s) from that turn went uncited"
        if isinstance(count, int) and not isinstance(count, bool)
        else "that turn's run-tests receipts went uncited (count not recorded)"
    )
    return (
        "The coder's test_observations were not a JSON array (same-agent re-asks "
        f"attempted: {'yes' if getattr(diagnosis, 'repair_attempted', False) else 'no'}); "
        f"{receipts}, so the blocker may be a tool format problem rather than a code defect."
    )


def render_no_progress_message(
    *,
    lead: str,
    subject_label: str,
    subject: str,
    k: int,
    rounds: Sequence[int],
    item_lines: Sequence[str],
    route: str | None = None,
    evidence: str | None = None,
    citation_diagnosis: str | None = None,
    approach_alternative: tuple[int, str] | None = None,
    pr_size: str | None = None,
) -> str:
    """The single no-progress human-decision message (exit 4).

    Optional sections appear only when present; an unavailable measurement
    is omitted rather than shown as a placeholder.
    """
    first, last = (rounds[0], rounds[-1]) if rounds else (0, 0)
    sections = [
        f"{lead}{NO_PROGRESS_PREFIX_TEXT}{k} consecutive completed round(s): rounds "
        f"{first}-{last} ended with the same {subject_label} `{subject}` and the same "
        f"blocking items (--no-progress-rounds {k}). No further coder or reviewer turn "
        "was invoked.",
        "Unchanged blocking items:\n" + "\n".join(item_lines),
    ]
    if pr_size:
        sections.append(
            "Total PR size (a scope-growth signal; compare it with the issue's suggested "
            f"direction): {pr_size}"
        )
    if evidence and evidence.strip():
        sections.append("Latest coder turn evidence:" + evidence.rstrip())
    if citation_diagnosis:
        sections.append(citation_diagnosis)
    if approach_alternative is not None:
        coder_round, excerpt = approach_alternative
        excerpt = excerpt.strip()
        if len(excerpt) > NO_PROGRESS_EXCERPT_LIMIT:
            excerpt = excerpt[: NO_PROGRESS_EXCERPT_LIMIT - 1].rstrip() + "…"
        sections.append(
            f"Step-back approach alternative proposed at round {coder_round} (excerpt): "
            f"{excerpt or '(empty)'}"
        )
    sections.append(
        "Decide how to proceed:\n"
        "1. Continue: rerun with a larger `--no-progress-rounds`; the run stays bounded "
        "by `--max-rounds`.\n"
        "2. Accept a stated limitation or narrow the scope with a signed human decision "
        "through the existing signed-requirement path that resolves or defers these items, "
        "then rerun the same command.\n"
        f"3. Redesign, or push a change yourself, and rerun: a new {subject_label} starts "
        "a new window."
    )
    if route and route.strip():
        sections.append(route.strip())
    return render_human_decision(
        what="\n".join(sections),
        clear=(
            "Resolve or defer the unchanged items with a signed human decision (option 2), "
            "or push a change, then rerun the same command; the other options are listed "
            "above."
        ),
        safe_to_automate=False,
        automation_note=(
            "deciding whether to continue, narrow or redesign a loop that made no progress "
            "is a human judgement."
        ),
    )
