"""Managed-CI PR qualification reads the plan binding from the PR side (#966)."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_loop_helpers import make_config

import coding_review_agent_loop.orchestrator as orchestrator
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import IssueComment, IssueContext
from coding_review_agent_loop.managed_ci import (
    ManagedCiIssueAuthorization,
    format_issue_created_authorization_comment,
    verify_managed_pr_plan_binding,
)
from coding_review_agent_loop.round_state import (
    ApprovedPlanContext,
    PostedRoundMetadata,
    _attach_round_metadata,
)
from coding_review_agent_loop.runner import CommandResult

PLAN = "8c6ed707e3700a21"


class CommentsRunner:
    """Serve the PR comment REST list; any other command is a test failure."""

    def __init__(self, comments, *, returncode=0):
        self.comments = comments
        self.returncode = returncode
        self.commands = []

    def run(self, args, *, cwd=None, check=True, **_kwargs):
        self.commands.append(list(args))
        assert "repos/OWNER/REPO/issues/7/comments?per_page=100" in args
        return CommandResult(
            list(args), Path(cwd or "."), json.dumps(self.comments), "", self.returncode
        )


def _root(**overrides):
    record = ManagedCiIssueAuthorization(
        kind="creation", repository="OWNER/REPO", issue_number=959, pr_number=7,
        base_ref="main", head_sha="head-0", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="root",
        label_event_id=101, approved_plan_hash=PLAN,
    )
    return replace(record, **overrides)


def _continuity(**overrides):
    return _root(
        kind="continuity", head_sha="head-1", nonce="next", predecessor_head="head-0",
        predecessor_comment_id=41, round_comment_ids=(88, 89), **overrides,
    )


def _auth_comment(comment_id, record, *, login="agent-loop", user_id=1):
    return {
        "id": comment_id,
        "user": {"login": login, "id": user_id},
        "body": str(format_issue_created_authorization_comment(record)),
    }


def _round_comment(comment_id, *, role, subject, round_number, state=None):
    return {
        "id": comment_id,
        "user": {"login": "agent-loop", "id": 1},
        "body": _attach_round_metadata(
            f"{role} round",
            PostedRoundMetadata(
                flow="pr", role=role, agent="agent-loop",
                round_number=round_number, subject=subject, state=state,
            ),
        ),
    }


def _chain(continuity=None):
    return [
        _auth_comment(41, _root()),
        _round_comment(88, role="reviewer", subject="head-0", round_number=1, state="blocking"),
        _round_comment(89, role="coder", subject="head-1", round_number=2),
        _auth_comment(100, continuity or _continuity()),
    ]


def _config(tmp_path, **overrides):
    overrides.setdefault("managed_ci_trusted_actor", "agent-loop")
    return make_config(tmp_path, managed_ci=True, **overrides)


def _verify(tmp_path, comments, *, live_head="head-1", plan=PLAN, **config_overrides):
    verify_managed_pr_plan_binding(
        CommentsRunner(comments), config=_config(tmp_path, **config_overrides),
        pr_number=7, issue_number=959, live_head=live_head, approved_plan_hash=plan,
    )


def test_continuity_chain_to_the_live_head_binds_the_plan(tmp_path):
    _verify(tmp_path, _chain())


def test_creation_record_at_the_live_head_binds_the_plan(tmp_path):
    _verify(tmp_path, [_auth_comment(41, _root())], live_head="head-0")


@pytest.mark.parametrize(
    ("comments", "kwargs", "reason"),
    [
        ([], {}, "no authorization record exists"),
        (None, {"live_head": "head-2"}, "no authenticated chain reaches the live head"),
        (None, {"plan": "0" * 16}, "names a different approved plan"),
        (None, {"managed_ci_trusted_actor": "someone-else"}, "not authored by the trusted actor"),
        (None, {"live_head": None}, "live head is unknown"),
    ],
)
def test_binding_fails_closed(tmp_path, comments, kwargs, reason):
    with pytest.raises(AgentLoopError, match=reason):
        _verify(tmp_path, _chain() if comments is None else comments, **kwargs)


def test_binding_rejects_a_record_for_another_pr_or_issue(tmp_path):
    comments = _chain() + [_auth_comment(120, _root(issue_number=960))]
    with pytest.raises(AgentLoopError, match="different repository, issue, PR, or base"):
        _verify(tmp_path, comments)


def test_binding_rejects_a_record_from_a_foreign_author(tmp_path):
    comments = _chain() + [_auth_comment(120, _root(), login="mallory", user_id=9)]
    with pytest.raises(AgentLoopError, match="not authored by the trusted actor"):
        _verify(tmp_path, comments)


def test_binding_rejects_a_chain_whose_root_names_another_plan(tmp_path):
    comments = _chain()
    comments[0] = _auth_comment(41, _root(approved_plan_hash="0" * 16))
    with pytest.raises(AgentLoopError, match="names a different approved plan"):
        _verify(tmp_path, comments)


def test_binding_rejects_continuity_without_its_round_records(tmp_path):
    comments = [item for item in _chain() if item["id"] not in {88, 89}]
    with pytest.raises(AgentLoopError, match="no authenticated chain reaches the live head"):
        _verify(tmp_path, comments)


def test_binding_rejects_an_uninspectable_comment_list(tmp_path):
    runner = CommentsRunner([], returncode=1)
    with pytest.raises(AgentLoopError, match="could not be inspected"):
        verify_managed_pr_plan_binding(
            runner, config=_config(tmp_path), pr_number=7, issue_number=959,
            live_head="head-1", approved_plan_hash=PLAN,
        )


def _snapshot(tmp_path, monkeypatch, *, managed_ci, issue_comments=()):
    issue = IssueContext(
        number=959, repo="OWNER/REPO", title="t", body="b", url=None,
        comments=tuple(issue_comments),
    )
    plan = ApprovedPlanContext(
        canonical_text="plan", plan_hash=PLAN, plan_subject="subject", availability="available",
    )
    pr_context = SimpleNamespace(
        metadata=SimpleNamespace(head_sha="head-1", base_branch="main"),
        comments=(),
        human_requirements=(),
        architecture_identity_changed=False,
    )
    calls = []
    monkeypatch.setattr(orchestrator, "get_pr_review_context", lambda *a, **k: pr_context)
    monkeypatch.setattr(orchestrator, "get_issue_context", lambda *a, **k: issue)
    monkeypatch.setattr(
        orchestrator, "_revalidate_pr_architecture_identity", lambda *a, **k: (None, False)
    )
    monkeypatch.setattr(
        orchestrator, "recover_approved_plan_context", lambda *a, **k: plan
    )
    monkeypatch.setattr(
        orchestrator, "verify_managed_pr_plan_binding", lambda *a, **k: calls.append(k)
    )
    config = make_config(tmp_path, managed_ci=managed_ci, managed_ci_trusted_actor="agent-loop")
    result = orchestrator._fresh_pr_qualification_snapshot(
        object(), config=config, pr_number=7, issue_context=issue,
        parent_issue_context=None, approved_plan_context=plan,
    )
    return result, calls


def test_managed_qualification_uses_the_pr_side_binding_without_issue_handoff(
    tmp_path, monkeypatch
):
    (_context, _ids, plan, _config), calls = _snapshot(tmp_path, monkeypatch, managed_ci=True)
    assert plan.plan_hash == PLAN
    assert calls == [{
        "config": calls[0]["config"], "pr_number": 7, "issue_number": 959,
        "live_head": "head-1", "approved_plan_hash": PLAN,
    }]


def test_unmanaged_qualification_still_requires_the_issue_side_handoff(tmp_path, monkeypatch):
    with pytest.raises(AgentLoopError, match="Approved-plan/handoff identity changed"):
        _snapshot(tmp_path, monkeypatch, managed_ci=False)


def test_unrelated_issue_prose_is_not_a_handoff(tmp_path, monkeypatch):
    prose = IssueComment(author="agent-loop", created_at=None, body="status note: handoff pending")
    with pytest.raises(AgentLoopError, match="Approved-plan/handoff identity changed"):
        _snapshot(tmp_path, monkeypatch, managed_ci=False, issue_comments=(prose,))
