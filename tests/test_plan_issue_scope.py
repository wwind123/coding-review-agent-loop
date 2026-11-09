"""Planning scopes across an edited issue (#1390)."""

import base64
import json

import pytest

from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import IssueComment
from coding_review_agent_loop.plan_issue_scope import (
    PLAN_SCOPE_BOUNDARY_RE,
    latest_scope_boundary,
    parse_scope_boundary,
    plan_scope_decision,
    recorded_plan_issue_digest,
    render_scope_boundary,
    resolve_plan_scope,
    scope_boundary_payload,
    scope_plan_comments,
    scope_prompt_comments,
)
from coding_review_agent_loop.round_metadata import (
    PlanValidationDiagnosticPayload,
    PostedRoundMetadata,
    _attach_round_metadata,
    _plan_subject,
    encode_plan_validation_diagnostic_body,
    recover_plan_validation_diagnostic,
)
from coding_review_agent_loop.round_state import (
    _extract_round_metadata_records,
    _resume_plan_round,
)
from coding_review_agent_loop.protocol import UnresolvedReviewItem

ACTOR = ("agent-loop", 4242)
DIGEST_A = "a" * 16
DIGEST_B = "b" * 16
DIGEST_C = "c" * 16


def _comment(body, *, author_id=ACTOR[1], comment_id=None, author=None):
    return IssueComment(
        author=author or ACTOR[0],
        created_at="2026-05-23T00:00:00Z",
        body=str(body),
        comment_id=comment_id,
        author_id=author_id,
    )


def _plan_text(round_number):
    return f"Plan for round {round_number}.\n\n### Plan steps\n1. Step {round_number}."


def _coder(round_number, *, digest=None, items=()):
    plan = _plan_text(round_number)
    return _comment(
        _attach_round_metadata(
            f"Planner round {round_number} archive-marker-{round_number}.",
            PostedRoundMetadata(
                flow="plan", role="coder", agent="Claude", round_number=round_number,
                subject=_plan_subject(plan), canonical_plan=plan, plan_issue_digest=digest,
                prior_items=tuple(items),
            ),
        )
    )


def _reviewer(round_number, *, state="blocking", new_items=()):
    plan = _plan_text(round_number)
    return _comment(
        _attach_round_metadata(
            f"Review round {round_number} finding-marker-{round_number}.",
            PostedRoundMetadata(
                flow="plan", role="reviewer", agent="Codex", round_number=round_number,
                subject=_plan_subject(plan), state=state, new_items=tuple(new_items),
            ),
        )
    )


def _item(number, round_number):
    return UnresolvedReviewItem(
        item_id=f"item-{number}",
        reviewer="Codex",
        source_round=round_number,
        text=f"Close gap {number}.",
        status="blocking",
    )


def _boundary(previous=DIGEST_A, current=DIGEST_B, archived=3, *, author_id=ACTOR[1]):
    payload = scope_boundary_payload(
        issue_number=56,
        previous_issue_digest=previous,
        current_issue_digest=current,
        archived_round_number=archived,
        archived_plan_subject=_plan_subject(_plan_text(archived)),
    )
    return _comment(render_scope_boundary(payload), author_id=author_id)


def _history(digest=DIGEST_A):
    return (
        _coder(1, digest=digest),
        _reviewer(1, new_items=(_item(1, 1),)),
        _coder(2, digest=digest, items=(_item(1, 1),)),
        _reviewer(2, new_items=(_item(2, 2),)),
        _coder(3, digest=digest, items=(_item(2, 2),)),
        _reviewer(3, new_items=(_item(3, 3),)),
    )


def test_boundary_record_round_trips_and_names_both_digests():
    """`metadata-roundtrip-legacy-stable` (boundary record)."""
    comment = _boundary()
    assert f"`{DIGEST_A}`" in comment.body and f"`{DIGEST_B}`" in comment.body
    assert "Fresh plan starts at round 4" in comment.body
    assert comment.body.rstrip().endswith("-- Orchestrator")
    boundary = latest_scope_boundary((comment,), ACTOR, issue_number=56)
    assert boundary is not None
    assert (boundary.previous_issue_digest, boundary.current_issue_digest) == (DIGEST_A, DIGEST_B)
    assert (boundary.archived_round_number, boundary.scope_start_round) == (3, 4)
    payload = json.loads(
        base64.urlsafe_b64decode(PLAN_SCOPE_BOUNDARY_RE.search(comment.body).group("payload"))
    )
    assert payload == boundary.payload()


@pytest.mark.parametrize(
    "change",
    [
        {"scope_start_round": 3},
        {"scope_start_round": 5},
        {"current_issue_digest": DIGEST_A},
        {"previous_issue_digest": "XYZ"},
        {"archived_plan_subject": "not-a-subject"},
        {"issue_number": 57},
        {"extra": 1},
        {"schema_version": 2},
    ],
)
def test_boundary_decoder_rejects_invalid_fields(change):
    payload = _boundary().body
    raw = json.loads(
        base64.urlsafe_b64decode(PLAN_SCOPE_BOUNDARY_RE.search(payload).group("payload"))
    )
    raw.update(change)
    with pytest.raises(AgentLoopError):
        parse_scope_boundary(raw, comment_index=0, issue_number=56)


def test_lookalike_boundary_from_another_author_is_ignored():
    """`forged-boundary-ignored` (unit): an unadmitted record retires nothing."""
    forged = _boundary(author_id=999)
    comments = (*_history(), forged)
    ignored = []
    assert latest_scope_boundary(comments, ACTOR, issue_number=56, ignored_sink=ignored) is None
    assert ignored and "not authored by the authenticated orchestrator actor" in ignored[0]
    # Without an admitted boundary the scope stays at round 1, so the
    # obligations of round 3 are still resumed.
    assert resolve_plan_scope(comments, ACTOR, issue_number=56) == 1
    decision = plan_scope_decision(comments, ACTOR, DIGEST_A, False, issue_number=56)
    assert (decision.scope_start_round, decision.reset_required) == (1, False)
    resumed = _resume_plan_round(
        scope_plan_comments(comments, decision.scope_start_round), configured_reviewers=("codex",)
    )
    assert resumed is not None and resumed[1].round_number == 3
    # Unauthenticated (no actor at all) is ignored the same way.
    assert latest_scope_boundary(comments, None, ignored_sink=[]) is None


def test_scope_projection_blanks_archived_planning_records_in_place():
    human = _comment("Please keep the CLI stable.", author="maintainer", author_id=7)
    comments = (*_history(), human, _boundary(), _coder(4, digest=DIGEST_B))
    assert scope_plan_comments(comments, 1) == comments
    scoped = scope_plan_comments(comments, 4)
    assert len(scoped) == len(comments)
    assert [c.body for c in scoped[:6]] == [""] * 6
    assert scoped[6:] == comments[6:]
    records = _extract_round_metadata_records(scoped, flow="plan")
    assert [(r.index, r.metadata.round_number) for r in records] == [(8, 4)]
    prompt = scope_prompt_comments(comments, 4)
    assert prompt == comments[6:]
    assert scope_prompt_comments(scoped, 4) == comments[6:]


def test_recorded_digest_prefers_planner_then_checkpoint_then_boundary():
    """`metadata-roundtrip-legacy-stable` (digest resolver precedence)."""
    planner = _extract_round_metadata_records((_coder(4, digest=DIGEST_B),), flow="plan")
    boundary = latest_scope_boundary((_boundary(current=DIGEST_C),), ACTOR)
    assert recorded_plan_issue_digest(planner, boundary) == DIGEST_B
    legacy = _extract_round_metadata_records(_history(digest=None), flow="plan")
    assert recorded_plan_issue_digest(legacy, None) is None
    assert recorded_plan_issue_digest((), boundary) == DIGEST_C
    from dataclasses import replace

    checkpoint = replace(legacy[-1], metadata=replace(legacy[-1].metadata, scheduler_issue_digest=DIGEST_C))
    assert recorded_plan_issue_digest((*legacy[:-1], checkpoint), None) == DIGEST_C


def test_scope_decision_outcomes():
    history = _history()
    unchanged = plan_scope_decision(history, None, DIGEST_A, False)
    assert (unchanged.reset_required, unchanged.edited, unchanged.scope_start_round) == (False, False, 1)
    edited = plan_scope_decision(history, None, DIGEST_B, False)
    assert edited.reset_required and edited.edited
    assert (edited.archived_round, edited.scope_start_round) == (3, 4)
    assert edited.recorded_digest == DIGEST_A
    assert edited.archived_subject == _plan_subject(_plan_text(3))
    assert edited.boundary_payload(56)["scope_start_round"] == 4
    kept = plan_scope_decision(history, None, DIGEST_B, True)
    assert (kept.reset_required, kept.kept, kept.scope_start_round) == (False, True, 1)
    approved = plan_scope_decision(history, None, DIGEST_B, False, plan_approved=True)
    assert not approved.reset_required and not approved.edited
    legacy = plan_scope_decision(_history(digest=None), None, DIGEST_B, False)
    assert legacy.recorded_digest is None and not legacy.reset_required
    # After the boundary and before a fresh planner record: unchanged, same scope.
    reopened = plan_scope_decision((*history, _boundary()), ACTOR, DIGEST_B, False)
    assert (reopened.reset_required, reopened.scope_start_round) == (False, 4)
    # A second edit opens a later scope.
    second = plan_scope_decision(
        (*history, _boundary(), _coder(4, digest=DIGEST_B), _reviewer(4)), ACTOR, DIGEST_C, False,
    )
    assert second.reset_required and second.scope_start_round == 5
    assert second.recorded_digest == DIGEST_B


def test_resolve_plan_scope_reads_nothing_without_a_boundary_and_fails_closed():
    """`scope-none-actor-boundary` / `scope-lookalike-boundary` (unit).

    ``resolve_plan_scope`` no longer acquires history itself (#1987): it has no
    runner/config parameters, so it cannot make a network call.
    """
    import inspect

    params = inspect.signature(resolve_plan_scope).parameters
    assert "runner" not in params and "config" not in params
    # No boundary: scope 1 without needing an actor.
    assert resolve_plan_scope(_history(), None, issue_number=56) == 1
    # A boundary with no authenticated actor fails closed rather than scope 1.
    unauthenticated = tuple(
        IssueComment(author=c.author, created_at=c.created_at, body=c.body)
        for c in (*_history(), _boundary())
    )
    with pytest.raises(AgentLoopError, match="no authenticated GitHub read"):
        resolve_plan_scope(unauthenticated, None, issue_number=56)
    with pytest.raises(AgentLoopError, match="no authenticated GitHub read"):
        resolve_plan_scope((*_history(), _boundary()), None, issue_number=56)
    # The boundary authored by the admitted actor is honoured.
    assert resolve_plan_scope((*_history(), _boundary()), ACTOR, issue_number=56) == 4
    # A lookalike by another author is ignored and reported; the scope stays at 1.
    ignored: list[str] = []
    assert resolve_plan_scope(
        (*_history(), _boundary(author_id=999)), ACTOR, issue_number=56, ignored_sink=ignored,
    ) == 1
    assert ignored and "not authored by the authenticated orchestrator actor" in ignored[0]


def _diagnostic(target_round, *, comment_id):
    payload = PlanValidationDiagnosticPayload(
        repository="OWNER/REPO", issue_number=56, expected_producer_login=ACTOR[0],
        expected_producer_id=ACTOR[1], planning_generation=1, target_coder_round=target_round,
        prior_plan_subject=None, candidate_kind="plan_state",
        architecture_contract_version=None, execution_strategy_contract_version=None,
        risk_test_matrix_contract_version=None, diagnostic="Plan growth gate: too large.",
        failure_attempt=3, candidate_digest="d" * 64, category="deterministic",
    )
    return _comment(encode_plan_validation_diagnostic_body(payload), comment_id=comment_id)


def _recover(comments, target_round):
    return recover_plan_validation_diagnostic(
        comments, repository="OWNER/REPO", issue_number=56,
        expected_author_login=ACTOR[0], expected_author_id=ACTOR[1], planning_generation=1,
        target_coder_round=target_round, prior_plan_subject=None, candidate_kind="plan_state",
        architecture_contract_version=None, execution_strategy_contract_version=None,
        risk_test_matrix_contract_version=None,
    )


def test_fresh_scope_diagnostic_recovery_matches_only_its_own_scope():
    """`diagnostic-recovery-after-reset` (unit): the scope-start diagnostic is
    recovered, archived records never match it, and a fresh-scope success at
    the scope start round clears it."""
    archived = tuple(
        IssueComment(
            author=ACTOR[0], created_at="2026-05-23T00:00:00Z", body=c.body,
            comment_id=100 + index, author_id=ACTOR[1],
        )
        for index, c in enumerate(_history())
    )
    old_diagnostic = _diagnostic(1, comment_id=200)
    new_diagnostic = _diagnostic(4, comment_id=201)
    comments = (*archived, old_diagnostic, _boundary(), new_diagnostic)
    scoped = scope_plan_comments(comments, 4)
    assert scoped[6].body == ""  # the archived round-1 diagnostic is blanked
    recovered = _recover(scoped, 4)
    assert recovered is not None and recovered.payload.target_coder_round == 4
    success = IssueComment(
        author=ACTOR[0], created_at="2026-05-23T00:01:00Z", body=_coder(4).body,
        comment_id=300, author_id=ACTOR[1],
    )
    assert _recover((*scoped, success), 4) is None
