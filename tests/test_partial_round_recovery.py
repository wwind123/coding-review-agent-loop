"""Partial-round recovery bounded by a trusted conversion record (#1664)."""

from __future__ import annotations

import random
import string

from coding_review_agent_loop.evidence_conversion import ConversionAuditRow, EvidenceConversionAudit
from coding_review_agent_loop.github import IssueComment
from coding_review_agent_loop.partial_round_recovery import compute_partial_round_recovery
from coding_review_agent_loop.pr_loop_support import comment_identity_key
from coding_review_agent_loop.review_scheduling import make_contract
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    _attach_round_metadata,
    _resume_pr_round,
)
from coding_review_agent_loop.round_transport import prepare_round_comment

ACTOR = ("bot", 4242)
HEAD = "h"
ROUND = 3
REVIEWERS = ("Codex", "Gemini")


def _big_text(seed):
    rnd = random.Random(seed)
    return "".join(rnd.choice(string.ascii_letters + " ") for _ in range(300_000))


def _meta(role="reviewer", agent="Codex", phase="publication", **extra):
    scheduler = {}
    if phase == "scheduler-prelaunch":
        scheduler = dict(
            scheduler_contract=make_contract(REVIEWERS, "selective-intermediate", None).as_dict(),
            scheduler_previous_sha=None, scheduler_current_sha=HEAD,
            scheduler_obligation_digest="0" * 16,
            scheduler_selected_reviewers=REVIEWERS, scheduler_paused_reviewers=(),
            scheduler_reasons=("full board",), scheduler_final_sweep=False,
            scheduler_force_full=False, scheduler_calls_avoided=0,
        )
    if role == "reviewer":
        extra.setdefault("state", "blocking" if agent == "Gemini" else "approved")
    return PostedRoundMetadata(
        flow="pr", role=role, agent=agent, round_number=ROUND, subject=HEAD, phase=phase,
        **scheduler, **extra,
    )


def _conversion_meta():
    audit = EvidenceConversionAudit(
        source="auto", matrix_identity="m" * 64, head=HEAD,
        rows=(ConversionAuditRow(
            row_id="row-a", obligation_identity="human-exact-head-evidence:row:x",
            obligation_item_id="item-2", owners=("Gemini",), converted_item_ids=(),
            attempt_record_rounds=(1, 2),
        ),),
    )
    return PostedRoundMetadata(
        flow="pr", role="summary", agent="Orchestrator", round_number=ROUND, subject=HEAD,
        state="blocking", phase="evidence-conversion", evidence_conversion=audit,
        conversion_review_round=ROUND,
    )


class _History:
    """A PR comment history with REST identities: round 3, a conversion, a partial repeat."""

    def __init__(self):
        self.comments: list[IssueComment] = []

    def add(self, body, *, author=ACTOR[0]):
        number = len(self.comments) + 1
        self.comments.append(IssueComment(
            author=author, created_at=f"2026-01-01T00:00:{number:02d}Z", body=str(body),
            comment_id=1000 + number, author_id=ACTOR[1] if author == ACTOR[0] else 7,
            url=f"https://example.test/c/{1000 + number}",
        ))
        return number - 1

    def add_record(self, metadata, text="body", **kwargs):
        parts = prepare_round_comment(_attach_round_metadata(text, metadata))
        return [self.add(part, **kwargs) for part in parts][-1]

    def orphan(self, metadata, **kwargs):
        parts = prepare_round_comment(_attach_round_metadata("orphan", metadata))
        return [self.add(part, **kwargs) for part in parts[:-1]]

    def ids(self, indexes):
        return {1000 + index + 1 for index in indexes}


def _conversion_round(*, before_conversion=None, after_conversion=None):
    history = _History()
    history.add_record(_meta(role="coder", agent="Claude", phase="authoritative"))
    history.add_record(_meta(role="summary", agent="Orchestrator", phase="scheduler-prelaunch"))
    pre_review = history.add_record(_meta(agent="Gemini"), text="Gemini blocks on row-a evidence.")
    pre_reconciliation = history.add_record(
        _meta(role="summary", agent="Orchestrator", phase="reconciliation")
    )
    if before_conversion is not None:
        before_conversion(history)
    conversion = history.add_record(_conversion_meta(), text="Risk-matrix evidence converted.")
    repeat_checkpoint = history.add_record(
        _meta(role="summary", agent="Orchestrator", phase="scheduler-prelaunch")
    )
    if after_conversion is not None:
        after_conversion(history)
    repeat_review = history.add_record(_meta(agent="Codex"), text="Codex reviews the converted ledger.")
    return history, {
        "pre_review": pre_review, "pre_reconciliation": pre_reconciliation,
        "conversion": conversion, "repeat_checkpoint": repeat_checkpoint,
        "repeat_review": repeat_review,
    }


def _resolver(guard_key):
    """Locate the guard's conversion record by identity; only REST-admitted copies count."""

    def resolve(history):
        for index, comment in enumerate(history):
            if comment_identity_key(comment) == guard_key and comment.author_id == ACTOR[1]:
                return index
        return None

    return resolve


def _recover(snapshot, rest, *, resolver):
    return compute_partial_round_recovery(
        snapshot=tuple(snapshot), read_rest=lambda: tuple(rest), flow="pr",
        round_number=ROUND, subject=HEAD, reviewer_names=REVIEWERS,
        resume=lambda remaining: _resume_pr_round(
            remaining, head_sha=HEAD, configured_reviewers=("codex", "gemini"),
            reconciliation_mode="aggregate", trusted_actor=ACTOR,
        ),
        conversion_boundary=resolver,
    )


def _listed(recovery):
    return {target.comment_id for target in recovery.targets}


def test_recovery_verifies_while_retaining_the_conversion_and_the_earlier_pass():
    """`recovery-verified-with-boundary`."""
    history, at = _conversion_round()
    key = comment_identity_key(history.comments[at["conversion"]])
    recovery = _recover(history.comments, history.comments, resolver=_resolver(key))
    assert recovery.verified, recovery.reason
    assert _listed(recovery) == history.ids([at["repeat_review"]])
    retained = history.ids([at["conversion"], at["pre_review"], at["pre_reconciliation"]])
    assert not retained & _listed(recovery)


def test_without_a_boundary_the_recovery_listing_is_unchanged():
    """`recovery-verified-with-boundary` negative: no resolver keeps today's listing."""
    history, at = _conversion_round()
    recovery = _recover(history.comments, history.comments, resolver=None)
    # Today's analysis treats the earlier pass and the conversion as this round's records.
    assert history.ids([at["pre_review"], at["repeat_review"]]) <= _listed(recovery)
    assert history.ids([at["conversion"]]) <= _listed(recovery)


def test_a_rest_history_with_shifted_indexes_resolves_the_boundary_by_identity():
    """`recovery-rest-shifted-history`: a snapshot index is never applied to REST."""
    history, at = _conversion_round()
    rest = [
        IssueComment(
            author="someone", created_at="2025-12-31T00:00:00Z", body="An earlier unrelated comment.",
            comment_id=999, author_id=7, url="https://example.test/c/999",
        ),
        *history.comments,
    ]
    key = comment_identity_key(history.comments[at["conversion"]])
    recovery = _recover(history.comments, rest, resolver=_resolver(key))
    assert recovery.verified, recovery.reason
    # The REST indexes are shifted by one; the target is still only the repeat review.
    assert _listed(recovery) == history.ids([at["repeat_review"]])


def test_a_rest_history_without_the_admitted_conversion_falls_back_unverified():
    """`recovery-rest-shifted-history`: the boundary cannot be located in REST."""
    history, at = _conversion_round()
    rest = list(history.comments)
    # The REST copy of the conversion comment carries no admissible identity.
    conversion = rest[at["conversion"]]
    rest[at["conversion"]] = IssueComment(
        author=conversion.author, created_at=conversion.created_at, body=conversion.body,
        comment_id=conversion.comment_id, author_id=None, url=conversion.url,
    )
    key = comment_identity_key(conversion)
    recovery = _recover(history.comments, rest, resolver=_resolver(key))
    assert not recovery.verified
    assert recovery.reason == "comment ids could not be read completely"
    # Computed from the snapshot with its own resolved boundary.
    assert _listed(recovery) == history.ids([at["repeat_review"]])
    # A REST history that lacks the comment entirely is likewise snapshot-based.
    missing = [c for i, c in enumerate(history.comments) if i != at["conversion"]]
    recovery = _recover(history.comments, missing, resolver=_resolver(key))
    assert not recovery.verified
    assert _listed(recovery) == history.ids([at["repeat_review"]])


def test_orphan_attachments_are_attributed_only_after_the_conversion():
    """`recovery-orphan-attachment-window`."""
    orphans: dict[str, list[int]] = {}

    def before(history):
        orphans["pre"] = history.orphan(_meta(agent="Gemini", canonical_reviewer_response=_big_text(1)))

    def after(history):
        orphans["post_same"] = history.orphan(_meta(agent="Codex", canonical_reviewer_response=_big_text(2)))
        orphans["post_other"] = history.orphan(
            _meta(agent="Gemini", canonical_reviewer_response=_big_text(3)), author="stranger"
        )

    history, at = _conversion_round(before_conversion=before, after_conversion=after)
    assert orphans["pre"] and orphans["post_same"] and orphans["post_other"]
    key = comment_identity_key(history.comments[at["conversion"]])
    recovery = _recover(history.comments, history.comments, resolver=_resolver(key))
    uncertain = {target.comment_id for target in recovery.uncertain}
    listed = _listed(recovery)
    assert not history.ids(orphans["pre"]) & (listed | uncertain)
    assert history.ids(orphans["post_same"]) <= listed
    assert history.ids(orphans["post_other"]) <= uncertain
    assert history.ids([at["repeat_review"]]) <= listed
    assert not history.ids([at["conversion"], at["pre_review"]]) & listed
