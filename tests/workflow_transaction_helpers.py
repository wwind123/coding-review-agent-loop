"""Builders for authenticated comment envelopes and transaction records (#827)."""

from __future__ import annotations

from dataclasses import replace

from coding_review_agent_loop.github import AuthenticatedComment, AuthenticatedCommentView
from coding_review_agent_loop.issue_pr_handoff import (
    format_issue_pr_handoff_comment,
    format_issue_pr_handoff_v2_comment,
)
from coding_review_agent_loop.plan_review_scheduling import PlanCandidateKey, make_plan_contract
from coding_review_agent_loop.pr_contract import (
    format_pr_contract_comment,
    format_pr_contract_v2_comment,
    make_pr_contract,
)
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    _approved_plan_hash,
    _attach_round_metadata,
    _plan_subject,
)
from coding_review_agent_loop.workflow_transaction import (
    ENTRY_AUTHORIZATION,
    ENTRY_HANDOFF,
    ENTRY_INITIAL_CODER_ROUND,
    ENTRY_PR_CONTRACT,
    FLOW_APPROVED_PLAN,
    FLOW_ISSUE,
    PHASE_ABORTED,
    PHASE_COMMITTED,
    RECORD_SET_ENTRY_NAMES,
    STATUS_INHERITED,
    STATUS_NOT_APPLICABLE,
    STATUS_UNPUBLISHED,
    EntryOutcome,
    SchedulerCheckpointRef,
    WorkflowTransactionRecord,
    WorkflowTransition,
    derive_handoff_metadata,
    derive_pr_contract,
    format_transaction_record_comment,
    not_applicable,
    prepared_record,
    reissued,
)

REPO = "OWNER/REPO"
ISSUE = 813
PR = 826
ACTOR = ("agent-loop-bot", 4242)
FOREIGN = ("mallory", 666)
HEAD_1 = "a" * 40
HEAD_2 = "b" * 40
PLAN = "Approved plan: publish workflow metadata as one transaction."
PLAN_HASH = _approved_plan_hash(PLAN)
PLAN_SUBJECT = _plan_subject(PLAN)


def stamp(second: int) -> str:
    return f"2026-09-21T10:{second // 60:02d}:{second % 60:02d}Z"


def comment(
    comment_id: int,
    body: str,
    *,
    surface: str = f"pr#{PR}",
    author: tuple[str, int] = ACTOR,
    second: int | None = None,
) -> AuthenticatedComment:
    created = stamp(comment_id if second is None else second)
    return AuthenticatedComment(
        surface=surface,
        comment_id=comment_id,
        author_login=author[0],
        author_id=author[1],
        created_at=created,
        updated_at=created,
        body=str(body),
    )


def view(surface: str, *comments: AuthenticatedComment, actor=ACTOR) -> AuthenticatedCommentView:
    ordered = sorted(comments, key=lambda item: item.comment_id)
    return AuthenticatedCommentView(
        surface=surface,
        actor_login=actor[0],
        actor_id=actor[1],
        authored=tuple(item for item in ordered if item.author_id == actor[1]),
        ignored_foreign=tuple(item for item in ordered if item.author_id != actor[1]),
    )


def pr_view(*comments, actor=ACTOR):
    return view(f"pr#{PR}", *comments, actor=actor)


def issue_view(*comments, number: int = ISSUE, actor=ACTOR):
    return view(f"issue#{number}", *comments, actor=actor)


def record_set(**overrides):
    defaults = {
        ENTRY_HANDOFF: reissued(ENTRY_HANDOFF),
        ENTRY_PR_CONTRACT: reissued(ENTRY_PR_CONTRACT),
        ENTRY_AUTHORIZATION: not_applicable(ENTRY_AUTHORIZATION),
        ENTRY_INITIAL_CODER_ROUND: reissued(ENTRY_INITIAL_CODER_ROUND),
    }
    names = {
        "handoff": ENTRY_HANDOFF,
        "contract": ENTRY_PR_CONTRACT,
        "authorization": ENTRY_AUTHORIZATION,
        "coder_round": ENTRY_INITIAL_CODER_ROUND,
    }
    for key, entry in overrides.items():
        defaults[names[key]] = entry
    return tuple(defaults[name] for name in RECORD_SET_ENTRY_NAMES)


def direct_intent(**overrides) -> WorkflowTransition:
    fields = dict(
        repository=REPO,
        primary_issue=ISSUE,
        pr_number=PR,
        base="main",
        head_sha=HEAD_1,
        origin_flow=FLOW_ISSUE,
        approved_plan_hash=None,
        expected_closing_issue_ids=(ISSUE,),
        scheduler_checkpoint=SchedulerCheckpointRef(absence_reason="flow-without-plan-review"),
        record_set=record_set(),
    )
    fields.update(overrides)
    return WorkflowTransition(**fields)


def plan_intent(**overrides) -> WorkflowTransition:
    fields = dict(
        origin_flow=FLOW_APPROVED_PLAN,
        approved_plan_hash=PLAN_HASH,
        scheduler_checkpoint=SchedulerCheckpointRef(absence_reason="no-plan-scheduler-records"),
    )
    fields.update(overrides)
    return direct_intent(**fields)


def prepared_comment(comment_id: int, intent: WorkflowTransition, *, author=ACTOR, second=None):
    body = format_transaction_record_comment(
        prepared_record(intent, writer_login=author[0], writer_id=author[1])
    )
    return comment(comment_id, body, author=author, second=second)


def outcomes_for(intent: WorkflowTransition, published: dict[str, int], *, aborted=False):
    result = []
    for entry in intent.record_set:
        if entry.inherited is not None:
            result.append(EntryOutcome(entry.name, status=STATUS_INHERITED))
        elif entry.disposition == "not-applicable":
            result.append(EntryOutcome(entry.name, status=STATUS_NOT_APPLICABLE))
        elif entry.name in published:
            result.append(EntryOutcome(entry.name, comment_id=published[entry.name]))
        else:
            assert aborted, f"{entry.name} needs a published comment ID"
            result.append(EntryOutcome(entry.name, status=STATUS_UNPUBLISHED))
    return tuple(result)


def terminal_comment(
    comment_id: int,
    intent: WorkflowTransition,
    *,
    prepared_id: int,
    published: dict[str, int] | None = None,
    phase: str = PHASE_COMMITTED,
    abort_reason: str | None = None,
    author=ACTOR,
    second=None,
):
    record = WorkflowTransactionRecord(
        phase=phase,
        transaction_id=intent.transaction_id,
        prepared_comment_id=prepared_id,
        outcomes=outcomes_for(intent, published or {}, aborted=phase == PHASE_ABORTED),
        abort_reason=abort_reason,
    )
    return comment(
        comment_id, format_transaction_record_comment(record), author=author, second=second
    )


def v2_contract_comment(comment_id: int, intent, *, author=ACTOR, **supersession):
    return comment(
        comment_id,
        format_pr_contract_v2_comment(derive_pr_contract(intent, **supersession)),
        author=author,
    )


def v2_handoff_comment(comment_id: int, intent, *, author=ACTOR):
    return comment(
        comment_id,
        format_issue_pr_handoff_v2_comment(derive_handoff_metadata(intent), repo=REPO),
        surface=f"issue#{intent.primary_issue}",
        author=author,
    )


def v1_contract(origin_flow=FLOW_ISSUE, ids=(ISSUE,), supersedes_hash=None):
    return make_pr_contract(
        repository=REPO,
        pr_number=PR,
        origin_flow=origin_flow,
        expected_closing_issue_ids=ids,
        primary_issue_number=ISSUE,
        supersedes_hash=supersedes_hash,
    )


def v1_contract_comment(comment_id: int, contract=None, *, author=ACTOR):
    return comment(comment_id, format_pr_contract_comment(contract or v1_contract()), author=author)


def v1_handoff_comment(
    comment_id: int, *, flow=FLOW_ISSUE, plan_hash=None, author=ACTOR, ids=None
):
    body = format_issue_pr_handoff_comment(
        issue_number=ISSUE,
        pr_number=PR,
        pr_url=f"https://github.com/{REPO}/pull/{PR}",
        pr_head_sha=HEAD_1,
        flow=flow,
        plan_hash=plan_hash,
        expected_closing_issue_ids=ids,
    )
    return comment(comment_id, body, surface=f"issue#{ISSUE}", author=author)


def plan_record_comment(
    comment_id: int, plan: str = PLAN, *, number: int = ISSUE, author=ACTOR, second=None
):
    body = _attach_round_metadata(
        "Plan candidate.\n-- Claude",
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=1,
            subject=_plan_subject(plan),
            canonical_plan=plan,
        ),
    )
    return comment(comment_id, body, surface=f"issue#{number}", author=author, second=second)


def plan_key(plan: str = PLAN, **overrides) -> PlanCandidateKey:
    fields = dict(
        subject=_plan_subject(plan),
        aggregate_plan_identity="aggregate",
        execution_strategy_identity="strategy",
        risk_test_matrix_identity="matrix",
        surfaced_requirement_id_digest="requirements",
        execution_strategy_contract_version=1,
    )
    fields.update(overrides)
    return PlanCandidateKey(**fields)


def scheduler_comment(
    comment_id: int,
    plan: str = PLAN,
    *,
    key: PlanCandidateKey | None = None,
    number: int = ISSUE,
    author=ACTOR,
    second=None,
    round_number: int = 1,
    surface: str | None = None,
):
    subject = _plan_subject(plan)
    key = key or plan_key(plan)
    body = _attach_round_metadata(
        f"Plan review scheduling audit {comment_id}.\n\n-- Orchestrator",
        PostedRoundMetadata(
            flow="plan",
            role="summary",
            agent="Orchestrator",
            round_number=round_number,
            subject=subject,
            phase="scheduler-prelaunch",
            scheduler_contract=make_plan_contract(
                ("Codex", "Gemini"), "primary-then-panel", "Codex"
            ).as_dict(),
            scheduler_obligation_digest="0" * 16,
            scheduler_selected_reviewers=("Codex",),
            scheduler_paused_reviewers=(("Gemini", "primary phase"),),
            scheduler_reasons=("primary phase",),
            scheduler_final_sweep=False,
            scheduler_force_full=False,
            scheduler_calls_avoided=1,
            scheduler_phase="primary",
            scheduler_primary_reviewer="Codex",
            plan_candidate_key=key.as_dict(),
        ),
    )
    return comment(
        comment_id, body, surface=surface or f"issue#{number}", author=author, second=second
    )


def plan_reviewer_comment(comment_id: int, plan: str = PLAN, *, number: int = ISSUE):
    body = _attach_round_metadata(
        "Plan review.\n<!-- AGENT_PLAN_STATE: approved -->\n-- Codex",
        PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject=_plan_subject(plan),
            state="approved",
        ),
    )
    return comment(comment_id, body, surface=f"issue#{number}")


def pr_review_comment(
    comment_id: int,
    *,
    head: str = HEAD_1,
    plan_hash: str | None = PLAN_HASH,
    plan_subject: str | None = PLAN_SUBJECT,
    author=ACTOR,
    second=None,
):
    body = _attach_round_metadata(
        f"Review {comment_id}.\n<!-- AGENT_STATE: approved -->\n-- Codex",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject=head,
            state="approved",
            approved_plan_hash=plan_hash,
            approved_plan_subject=plan_subject,
        ),
    )
    return comment(comment_id, body, author=author, second=second)


def with_created_at(item: AuthenticatedComment, created_at: str) -> AuthenticatedComment:
    return replace(item, created_at=created_at, updated_at=created_at)
