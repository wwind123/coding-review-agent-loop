"""PR-loop tests for pr_loop: merge handling (#1542)."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from coding_review_agent_loop.runner import CommandResult
from coding_review_agent_loop.cli import AgentLoopError, run_pr_loop
from coding_review_agent_loop.github import (
    PullRequestCheck,
    PullRequestMetadata,
)
from coding_review_agent_loop.managed_ci import (
    ManagedCiContract,
    ManagedCiOutcome,
    OrdinaryRecoveryCapability,
)
from agent_loop_helpers import (
    FakeRunner,
    make_config,
    structured_pr_review,
)
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.plan_first_loop as plan_first_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
from agent_loop_helpers import patch_attr_in
from pr_loop_test_helpers import (
    _carried_ci_obligations,
    _watch_check_board,
)


def test_pr_loop_runs_tests_and_merge_only_after_codex_approval(tmp_path):
    runner = FakeRunner(codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"])
    config = make_config(
        tmp_path,
        auto_merge=True,
        test_command=("pytest", "tests/test_agent_loop.py"),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    assert ["pytest", "tests/test_agent_loop.py"] in commands
    assert [
        "gh",
        "api",
        "repos/OWNER/REPO/commits/6367c48dd193d56ea7b0baad25b19455e529f5ee/check-runs",
    ] in commands
    assert [
        "gh",
        "api",
        "repos/OWNER/REPO/commits/6367c48dd193d56ea7b0baad25b19455e529f5ee/status",
    ] in commands
    assert [
        "gh",
        "api",
        "repos/OWNER/REPO/branches/main/protection/required_status_checks",
    ] in commands
    assert ["gh", "pr", "merge", "77", "--repo", "OWNER/REPO", "--merge", "--match-head-commit", "6367c48dd193d56ea7b0baad25b19455e529f5ee"] in commands


def test_ordinary_recovery_readies_and_merges_exact_head(monkeypatch, tmp_path):
    runner = FakeRunner()
    config = make_config(tmp_path, auto_merge=True)
    capability = OrdinaryRecoveryCapability(
        pr_number=77, repository="OWNER/REPO", base_ref="main", expected_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        released_label_event_id=101, released_at=100,
    )
    validations = []
    merged = []
    monkeypatch.setattr(pr_loop_support, "refresh_ordinary_recovery_capability", lambda *args, **kwargs: capability)
    monkeypatch.setattr(
        pr_loop_support,
        "wait_for_ordinary_recovery",
        lambda *args, **kwargs: ManagedCiOutcome(
            status="passed",
            checks=_watch_check_board("passing"),
            head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        ),
    )
    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_implementation, plan_first_loop, issue_loop,
    ), "get_observed_pr_review_context",
        lambda *args, **kwargs: SimpleNamespace(metadata=PullRequestMetadata(
            number=77, repo="OWNER/REPO", title="draft", head_branch="feature",
            base_branch="main", head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee", url=None,
        )),
    )
    monkeypatch.setattr(
        pr_loop_support,
        "validate_ordinary_recovery_capability",
        lambda *args, **kwargs: validations.append(kwargs.get("require_draft", True)) or True,
    )
    monkeypatch.setattr(
        pr_loop_support,
        "merge_pr",
        lambda *args, **kwargs: merged.append(kwargs["expected_head_sha"]),
    )

    pr_loop_support._finalize_ordinary_recovery_merge(
        runner, config=config, pr_number=77, capability=capability,
    )

    assert validations == [True, None]
    assert merged == ["6367c48dd193d56ea7b0baad25b19455e529f5ee"]
    assert ["gh", "pr", "ready", "77", "--repo", "OWNER/REPO"] in [
        command for command, _cwd in runner.commands
    ]


@pytest.mark.parametrize("board", ["absent", "neutral", "skipped", "forbidden"])
def test_ordinary_recovery_does_not_ready_or_merge_without_authoritative_board(
    monkeypatch, tmp_path, board
):
    runner = FakeRunner()
    config = make_config(tmp_path, auto_merge=True)
    capability = OrdinaryRecoveryCapability(
        pr_number=77,
        repository="OWNER/REPO",
        base_ref="main",
        expected_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        released_label_event_id=101,
        released_at=100,
    )
    if board == "absent":
        snapshot = None
    elif board == "forbidden":
        snapshot = _watch_check_board("passing", protection="forbidden")
    else:
        snapshot = _watch_check_board(
            "passing",
            passing=(PullRequestCheck(name="test", kind="check_run", status=board),),
        )
    monkeypatch.setattr(
        pr_loop_support, "refresh_ordinary_recovery_capability", lambda *args, **kwargs: capability
    )
    monkeypatch.setattr(
        pr_loop_support,
        "wait_for_ordinary_recovery",
        lambda *args, **kwargs: ManagedCiOutcome(
            status="passed", checks=snapshot, head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee"
        ),
    )
    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_implementation, plan_first_loop, issue_loop,
    ), "get_observed_pr_review_context",
        lambda *args, **kwargs: SimpleNamespace(metadata=PullRequestMetadata(
            number=77,
            repo="OWNER/REPO",
            title="draft",
            head_branch="feature",
            base_branch="main",
            head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            url=None,
        )),
    )
    monkeypatch.setattr(
        pr_loop_support,
        "merge_pr",
        lambda *args, **kwargs: pytest.fail("non-authoritative recovery must not merge"),
    )

    ordinary_item = _carried_ci_obligations()[1]
    items = [ordinary_item]
    assert not pr_loop_support._finalize_ordinary_recovery_checked(
        runner,
        config=config,
        pr_number=77,
        round_number=1,
        items=items,
        current_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        capability=capability,
    )
    assert items == [ordinary_item]
    assert not any(command[:3] == ["gh", "pr", "ready"] for command, _cwd in runner.commands)
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def test_ordinary_recovery_refuses_head_changed_before_ready(monkeypatch, tmp_path):
    runner = FakeRunner()
    config = make_config(tmp_path, auto_merge=True)
    capability = OrdinaryRecoveryCapability(
        pr_number=77, repository="OWNER/REPO", base_ref="main", expected_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        released_label_event_id=101, released_at=100,
    )
    monkeypatch.setattr(pr_loop_support, "refresh_ordinary_recovery_capability", lambda *args, **kwargs: capability)
    monkeypatch.setattr(
        pr_loop_support,
        "wait_for_ordinary_recovery",
        lambda *args, **kwargs: ManagedCiOutcome(status="head_changed", head_sha="4ebf8a1679f50e9e49fc965072f996bfe0dc18c6"),
    )

    with pytest.raises(AgentLoopError, match="head_changed"):
        pr_loop_support._finalize_ordinary_recovery_merge(
            runner, config=config, pr_number=77, capability=capability,
        )

    assert not any(command[:3] == ["gh", "pr", "ready"] for command, _cwd in runner.commands)


def test_ordinary_recovery_refuses_provenance_change_after_ready(monkeypatch, tmp_path):
    runner = FakeRunner()
    config = make_config(tmp_path, auto_merge=True)
    capability = OrdinaryRecoveryCapability(
        pr_number=77, repository="OWNER/REPO", base_ref="main", expected_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        released_label_event_id=101, released_at=100,
    )
    validations = iter([True, False])
    monkeypatch.setattr(pr_loop_support, "refresh_ordinary_recovery_capability", lambda *args, **kwargs: capability)
    monkeypatch.setattr(
        pr_loop_support,
        "wait_for_ordinary_recovery",
        lambda *args, **kwargs: ManagedCiOutcome(
            status="passed",
            checks=_watch_check_board("passing"),
            head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        ),
    )
    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_implementation, plan_first_loop, issue_loop,
    ), "get_observed_pr_review_context",
        lambda *args, **kwargs: SimpleNamespace(metadata=PullRequestMetadata(
            number=77, repo="OWNER/REPO", title="draft", head_branch="feature",
            base_branch="main", head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee", url=None,
        )),
    )
    monkeypatch.setattr(
        pr_loop_support,
        "validate_ordinary_recovery_capability",
        lambda *args, **kwargs: next(validations),
    )

    with pytest.raises(AgentLoopError, match="after `gh pr ready`"):
        pr_loop_support._finalize_ordinary_recovery_merge(
            runner, config=config, pr_number=77, capability=capability,
        )

    assert any(command[:3] == ["gh", "pr", "ready"] for command, _cwd in runner.commands)
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def test_ordinary_recovery_merge_failure_leaves_pr_ready(monkeypatch, tmp_path):
    runner = FakeRunner()
    config = make_config(tmp_path, auto_merge=True)
    capability = OrdinaryRecoveryCapability(
        pr_number=77, repository="OWNER/REPO", base_ref="main", expected_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        released_label_event_id=101, released_at=100,
    )
    monkeypatch.setattr(pr_loop_support, "refresh_ordinary_recovery_capability", lambda *args, **kwargs: capability)
    monkeypatch.setattr(
        pr_loop_support,
        "wait_for_ordinary_recovery",
        lambda *args, **kwargs: ManagedCiOutcome(
            status="passed",
            checks=_watch_check_board("passing"),
            head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        ),
    )
    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_implementation, plan_first_loop, issue_loop,
    ), "get_observed_pr_review_context",
        lambda *args, **kwargs: SimpleNamespace(metadata=PullRequestMetadata(
            number=77, repo="OWNER/REPO", title="draft", head_branch="feature",
            base_branch="main", head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee", url=None,
        )),
    )
    monkeypatch.setattr(pr_loop_support, "validate_ordinary_recovery_capability", lambda *args, **kwargs: True)

    def fail_merge(*args, **kwargs):
        raise AgentLoopError("merge failed")

    monkeypatch.setattr(pr_loop_support, "merge_pr", fail_merge)

    with pytest.raises(AgentLoopError, match="merge failed"):
        pr_loop_support._finalize_ordinary_recovery_merge(
            runner, config=config, pr_number=77, capability=capability,
        )

    assert any(command[:3] == ["gh", "pr", "ready"] for command, _cwd in runner.commands)
    assert not any("convert-to-draft" in command for command, _cwd in runner.commands)


# --- #510: merge ambiguity through every real merge entry path -----------------------


class _AmbiguousMergeRunner(FakeRunner):
    """A FakeRunner whose ``gh pr merge`` fails with a scripted 502.

    ``events`` records every merge attempt so a test can order merges against
    gate re-execution; the PR is reported OPEN at the approved head afterwards.
    """

    def __init__(self, *args, merge_script=("502",), events=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.merge_script = list(merge_script)
        self.events = events if events is not None else []
        self.merge_view = {
            "state": "OPEN", "mergedAt": None, "headRefOid": "6367c48dd193d56ea7b0baad25b19455e529f5ee", "mergeCommit": None,
        }

    def run(self, args, *, cwd, input_text=None, check=True, env=None):
        cmd = [str(a) for a in args]
        if cmd[:3] == ["gh", "pr", "merge"]:
            super().run(args, cwd=cwd, input_text=input_text, check=False, env=env)
            self.events.append("merge")
            action = self.merge_script.pop(0) if self.merge_script else "ok"
            where = Path(cwd) if cwd else Path(".")
            if action == "502":
                return CommandResult(cmd, where, "", "non-200 OK status code: 502 Bad Gateway", 1)
            return CommandResult(cmd, where, "", "", 0)
        if cmd[:3] == ["gh", "pr", "view"] and "state,mergedAt,headRefOid,mergeCommit" in cmd:
            where = Path(cwd) if cwd else Path(".")
            return CommandResult(cmd, where, json.dumps(self.merge_view), "", 0)
        return super().run(args, cwd=cwd, input_text=input_text, check=check, env=env)


def _merge_commands(runner):
    return [cmd for cmd, _cwd in runner.commands if cmd[:3] == ["gh", "pr", "merge"]]


def test_full_board_merge_502_is_one_attempt_and_a_rerun_regates_before_merging(tmp_path):
    events = []
    approval = "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    runner = _AmbiguousMergeRunner(codex_outputs=[approval, approval], events=events)
    config = make_config(tmp_path, auto_merge=True, test_command=("pytest", "tests/test_agent_loop.py"))
    original = pr_loop.get_pr_checks
    with patch.object(
        pr_loop, "get_pr_checks",
        side_effect=lambda *a, **k: events.append("full-board-gate") or original(*a, **k),
    ):
        with pytest.raises(AgentLoopError, match="rerun to re-execute every merge gate"):
            run_pr_loop(runner, pr_number=77, config=config)
        assert len(_merge_commands(runner)) == 1
        assert "--match-head-commit" in _merge_commands(runner)[0]
        assert events.count("merge") == 1

        # The rerun re-executes the full-board gate before any new merge attempt.
        assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(_merge_commands(runner)) == 2
    first_merge = events.index("merge")
    second_merge = len(events) - 1 - events[::-1].index("merge")
    assert "full-board-gate" in events[first_merge + 1:second_merge]


def test_managed_merge_502_is_one_attempt_and_a_rerun_requalifies_before_merging(tmp_path, monkeypatch):
    events = []
    approval = structured_pr_review(state="approved", summary="Approved.")
    runner = _AmbiguousMergeRunner(codex_outputs=[approval, approval], events=events)
    config = make_config(tmp_path, watch_pending_ci=True, auto_merge=True)
    monkeypatch.setattr(pr_loop, "activate_managed_ci", lambda *a, **k: ManagedCiContract())
    monkeypatch.setattr(
        pr_loop, "dispatch_final_qualification", lambda *a, **k: events.append("dispatch")
    )
    monkeypatch.setattr(
        pr_loop, "wait_for_final_qualification",
        lambda *a, **k: events.append("qualified") or ManagedCiOutcome(status="passed", head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee"),
    )

    with pytest.raises(AgentLoopError, match="rerun to re-execute every merge gate"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert len(_merge_commands(runner)) == 1
    assert "--match-head-commit" in _merge_commands(runner)[0]
    assert events == ["dispatch", "qualified", "merge"]

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(_merge_commands(runner)) == 2
    assert events == [
        "dispatch", "qualified", "merge", "dispatch", "qualified", "merge",
    ]  # no merge without a fresh exact-head qualification


def test_ordinary_recovery_merge_502_is_one_attempt_and_a_rerun_revalidates_before_merging(
    monkeypatch, tmp_path
):
    events = []
    runner = _AmbiguousMergeRunner(events=events)
    config = make_config(tmp_path, auto_merge=True)
    capability = OrdinaryRecoveryCapability(
        pr_number=77, repository="OWNER/REPO", base_ref="main", expected_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        released_label_event_id=101, released_at=100,
    )
    monkeypatch.setattr(pr_loop_support, "refresh_ordinary_recovery_capability", lambda *a, **k: capability)
    monkeypatch.setattr(
        pr_loop_support, "wait_for_ordinary_recovery",
        lambda *a, **k: events.append("recovery-gate") or ManagedCiOutcome(
            status="passed", checks=_watch_check_board("passing"), head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        ),
    )
    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_implementation, plan_first_loop, issue_loop,
    ), "get_observed_pr_review_context",
        lambda *a, **k: SimpleNamespace(metadata=PullRequestMetadata(
            number=77, repo="OWNER/REPO", title="draft", head_branch="feature",
            base_branch="main", head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee", url=None,
        )),
    )
    monkeypatch.setattr(
        pr_loop_support, "validate_ordinary_recovery_capability",
        lambda *a, **k: events.append("capability") or True,
    )

    with pytest.raises(AgentLoopError, match="rerun to re-execute every merge gate"):
        pr_loop_support._finalize_ordinary_recovery_merge(
            runner, config=config, pr_number=77, capability=capability,
        )
    assert len(_merge_commands(runner)) == 1
    assert "--match-head-commit" in _merge_commands(runner)[0]
    first_pass = list(events)
    assert first_pass[-1] == "merge" and "recovery-gate" in first_pass and "capability" in first_pass

    pr_loop_support._finalize_ordinary_recovery_merge(
        runner, config=config, pr_number=77, capability=capability,
    )

    assert len(_merge_commands(runner)) == 2
    second_pass = events[len(first_pass):]
    assert second_pass[-1] == "merge"
    assert "recovery-gate" in second_pass and "capability" in second_pass  # gates re-ran first


@pytest.mark.parametrize("path", ["full-board", "ordinary-recovery"])
def test_merge_502_that_actually_merged_is_recovered_with_no_second_merge(tmp_path, monkeypatch, path):
    runner = _AmbiguousMergeRunner()
    runner.merge_view = {"state": "MERGED", "mergedAt": "t", "headRefOid": "6367c48dd193d56ea7b0baad25b19455e529f5ee", "mergeCommit": {"oid": "m"}}
    if path == "full-board":
        approval = "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        runner.codex_outputs = [approval]
        config = make_config(tmp_path, auto_merge=True, test_command=("pytest", "tests/test_agent_loop.py"))
        assert run_pr_loop(runner, pr_number=77, config=config) == 0
    else:
        config = make_config(tmp_path, auto_merge=True)
        capability = OrdinaryRecoveryCapability(
            pr_number=77, repository="OWNER/REPO", base_ref="main", expected_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            released_label_event_id=101, released_at=100,
        )
        monkeypatch.setattr(pr_loop_support, "refresh_ordinary_recovery_capability", lambda *a, **k: capability)
        monkeypatch.setattr(
            pr_loop_support, "wait_for_ordinary_recovery",
            lambda *a, **k: ManagedCiOutcome(
                status="passed", checks=_watch_check_board("passing"), head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            ),
        )
        patch_attr_in(monkeypatch, (
            pr_loop_support, pr_loop, issue_implementation, plan_first_loop, issue_loop,
        ), "get_observed_pr_review_context",
            lambda *a, **k: SimpleNamespace(metadata=PullRequestMetadata(
                number=77, repo="OWNER/REPO", title="draft", head_branch="feature",
                base_branch="main", head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee", url=None,
            )),
        )
        monkeypatch.setattr(pr_loop_support, "validate_ordinary_recovery_capability", lambda *a, **k: True)
        pr_loop_support._finalize_ordinary_recovery_merge(
            runner, config=config, pr_number=77, capability=capability,
        )
    assert len(_merge_commands(runner)) == 1


class _FailingReleaseRunner(FakeRunner):
    """Label DELETEs fail with a 503 so the PR-loop cleanup cannot release."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deletes = 0

    def run(self, args, *, cwd, input_text=None, check=True, env=None):
        cmd = [str(a) for a in args]
        where = Path(cwd) if cwd else Path(".")
        if "DELETE" in cmd and any(a.endswith("/labels/agent-loop-managed") for a in cmd):
            super().run(args, cwd=cwd, input_text=input_text, check=False, env=env)
            self.deletes += 1
            return CommandResult(cmd, where, "", "HTTP 503 Service Unavailable", 1)
        return super().run(args, cwd=cwd, input_text=input_text, check=check, env=env)


def test_pr_loop_cleanup_refusal_carries_the_release_diagnostic(tmp_path, monkeypatch):
    approval = structured_pr_review(state="approved", summary="Approved.")
    runner = _FailingReleaseRunner(codex_outputs=[approval])
    config = make_config(
        tmp_path, watch_pending_ci=True, auto_merge=True, managed_ci=True,
        managed_ci_trusted_actor="agent-loop",
    )
    contract = ManagedCiContract(
        protocol_version=2, adopted_existing_pr=True, invocation_applied_label=True,
        active_label_event_id=None,
    )
    monkeypatch.setattr(pr_loop, "activate_managed_ci", lambda *a, **k: contract)
    monkeypatch.setattr(pr_loop, "revalidate_adopted_managed_ci", lambda *a, **k: True)
    monkeypatch.setattr(pr_loop, "dispatch_final_qualification", lambda *a, **k: None)

    monkeypatch.setattr(
        pr_loop, "wait_for_final_qualification",
        lambda *a, **k: (_ for _ in ()).throw(AgentLoopError("qualification aborted")),
    )

    with pytest.raises(AgentLoopError) as raised:
        run_pr_loop(runner, pr_number=77, config=config)

    notes = " ".join(getattr(raised.value, "__notes__", []))
    assert "requires manual label removal" in notes or "requires manual label removal" in str(raised.value)
    assert "503 Service Unavailable" in notes + str(raised.value)
    assert runner.deletes >= 1


class _ReadyAmbiguityRunner(FakeRunner):
    """``gh pr ready`` returns a 503; the PR then drifts or has actually changed."""

    def __init__(self, *args, drift=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.drift = drift
        self.pr_payload["isDraft"] = True

    def run(self, args, *, cwd, input_text=None, check=True, env=None):
        cmd = [str(a) for a in args]
        if cmd[:3] == ["gh", "pr", "ready"]:
            super().run(args, cwd=cwd, input_text=input_text, check=False, env=env)
            if self.drift:
                self.pr_payload["headRefOid"] = "d" * 40
            else:
                self.pr_payload["isDraft"] = False  # accepted despite the 503
            where = Path(cwd) if cwd else Path(".")
            return CommandResult(cmd, where, "", "HTTP 503 Service Unavailable", 1)
        return super().run(args, cwd=cwd, input_text=input_text, check=check, env=env)


def _ordinary_recovery_ready_setup(monkeypatch, runner_tmp):
    capability = OrdinaryRecoveryCapability(
        pr_number=77, repository="OWNER/REPO", base_ref="main", expected_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        released_label_event_id=101, released_at=100,
    )
    validations = []
    monkeypatch.setattr(pr_loop_support, "refresh_ordinary_recovery_capability", lambda *a, **k: capability)
    monkeypatch.setattr(
        pr_loop_support, "wait_for_ordinary_recovery",
        lambda *a, **k: ManagedCiOutcome(
            status="passed", checks=_watch_check_board("passing"), head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        ),
    )
    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_implementation, plan_first_loop, issue_loop,
    ), "get_observed_pr_review_context",
        lambda *a, **k: SimpleNamespace(metadata=PullRequestMetadata(
            number=77, repo="OWNER/REPO", title="draft", head_branch="feature",
            base_branch="main", head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee", url=None,
        )),
    )
    monkeypatch.setattr(
        pr_loop_support, "validate_ordinary_recovery_capability",
        lambda *a, **k: validations.append(k.get("require_draft", True)) or True,
    )
    return capability, validations


def test_ordinary_recovery_ready_503_with_head_drift_refuses_with_one_ready_and_no_merge(
    monkeypatch, tmp_path
):
    runner = _ReadyAmbiguityRunner(drift=True)
    config = make_config(tmp_path, auto_merge=True)
    capability, validations = _ordinary_recovery_ready_setup(monkeypatch, tmp_path)

    with pytest.raises(AgentLoopError, match="Unable to mark recovered PR #77 ready") as raised:
        pr_loop_support._finalize_ordinary_recovery_merge(
            runner, config=config, pr_number=77, capability=capability,
        )

    ready = [c for c, _ in runner.commands if c[:3] == ["gh", "pr", "ready"]]
    assert len(ready) == 1  # never replayed against the drifted head
    assert _merge_commands(runner) == []
    assert validations == [True]  # the post-write verification was never reached
    assert "503 Service Unavailable" in str(raised.value)


def test_ordinary_recovery_same_head_accepted_ready_reaches_post_write_verification(
    monkeypatch, tmp_path
):
    runner = _ReadyAmbiguityRunner(drift=False)
    config = make_config(tmp_path, auto_merge=True)
    capability, validations = _ordinary_recovery_ready_setup(monkeypatch, tmp_path)

    pr_loop_support._finalize_ordinary_recovery_merge(
        runner, config=config, pr_number=77, capability=capability,
    )

    ready = [c for c, _ in runner.commands if c[:3] == ["gh", "pr", "ready"]]
    assert len(ready) == 1
    assert validations == [True, None]  # the existing post-write verification ran
    assert len(_merge_commands(runner)) == 1  # and only then was the exact head merged
