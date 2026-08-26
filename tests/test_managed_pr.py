import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.managed_ci import ManagedCiCreationIntent, UNPROTECTED_OVERRIDE_TRAILER
from coding_review_agent_loop.managed_pr import (
    SOURCE_MARKER,
    _compose_body,
    create_managed_pr,
)
from coding_review_agent_loop.runner import CommandResult, Runner

from agent_loop_helpers import make_config


class ManagedPrRunner(Runner):
    def __init__(
        self,
        *,
        label_failure: bool = False,
        label_exists: bool = True,
        cleanup_failure: bool = False,
        existing_pulls=None,
        post_create_pulls=None,
    ):
        super().__init__()
        self.label_failure = label_failure
        self.label_exists = label_exists
        self.cleanup_failure = cleanup_failure
        self.existing_pulls = existing_pulls or []
        self.post_create_pulls = post_create_pulls
        self.pull_reads = 0
        self.calls: list[tuple[list[str], str | None]] = []
        self.branch_reads = 0

    def run(self, args, *, cwd, input_text=None, check=True, env=None):
        cmd = [str(item) for item in args]
        self.calls.append((cmd, input_text))
        if "--method" not in cmd:
            assert cmd[-1] == "repos/OWNER/REPO/labels/agent-loop-managed"
            return CommandResult(cmd, Path(cwd), "{}", "", 0 if self.label_exists else 1)
        method = cmd[cmd.index("--method") + 1]
        endpoint = cmd[cmd.index("--method") + 2]
        payload: object = {}
        returncode = 0
        stderr = ""
        if method == "GET" and "/branches/" in endpoint:
            self.branch_reads += 1
            payload = {"commit": {"sha": "a" * 40}}
        elif method == "GET" and endpoint.endswith("/pulls?state=open"):
            self.pull_reads += 1
            payload = (
                self.post_create_pulls
                if self.pull_reads > 1 and self.post_create_pulls is not None
                else self.existing_pulls
            )
        elif method == "POST" and endpoint.endswith("/pulls"):
            payload = {"number": 77}
        elif method == "POST" and endpoint.endswith("/labels") and self.label_failure:
            returncode = 1
            stderr = "label failed"
        elif self.cleanup_failure and method in {"PATCH", "DELETE"}:
            returncode = 1
            stderr = "cleanup failed"
        return CommandResult(cmd, Path(cwd), json.dumps(payload), stderr, returncode)


def _config(tmp_path):
    return replace(
        make_config(tmp_path, auto_merge=True),
        base="main",
        managed_ci_trusted_actor="wwind123",
        allow_unprotected_managed_ci=True,
    )


def _intent(*args, branch, **kwargs):
    return ManagedCiCreationIntent(
        branch=branch,
        trusted_actor="wwind123",
        protection_mode="plan_limited",
        audit_nonce="fresh-nonce",
    )


def test_create_managed_pr_opens_labeled_draft_and_correlates_override(tmp_path):
    runner = ManagedPrRunner()
    config = replace(
        _config(tmp_path),
        invocation_argv=(
            "agent-loop", "managed-pr", "--head", "fix/direct-change", "--title", "Fix direct change",
        ),
    )

    with patch("coding_review_agent_loop.managed_pr.preflight_managed_ci_creation", _intent):
        handoff = create_managed_pr(
            runner,
            config=config,
            source_branch="fix/direct-change",
            title="Fix direct change",
            body="## Summary\n\nKeeps real newlines.",
        )

    assert handoff.pr_number == 77
    assert handoff.config.managed_ci_expected_override_nonce == "fresh-nonce"
    assert handoff.config.pr_origin_flow == "managed-pr"
    assert handoff.config.invocation_argv == ()
    create_call = next(
        (cmd, body) for cmd, body in runner.calls
        if "--method" in cmd and cmd[cmd.index("--method") + 1] == "POST" and cmd[-3].endswith("/pulls")
    )
    created_payload = json.loads(create_call[1])
    assert created_payload["draft"] is True
    assert created_payload["head"].startswith("agent-loop/managed-direct-")
    assert "## Summary\n\nKeeps real newlines." in created_payload["body"]
    assert SOURCE_MARKER in created_payload["body"]
    assert f"{UNPROTECTED_OVERRIDE_TRAILER} nonce=fresh-nonce" in created_payload["body"]
    label_payload = next(
        json.loads(body) for cmd, body in runner.calls if cmd[-3].endswith("/labels")
    )
    assert label_payload == {"labels": ["agent-loop-managed"]}
    assert runner.branch_reads == 2


def test_empty_override_nonce_is_rejected_at_body_composition(tmp_path):
    with pytest.raises(AgentLoopError, match="Invalid AGENT_MANAGED_CI_UNPROTECTED_OVERRIDE_V1"):
        _compose_body(
            "",
            source_branch="fix/empty-nonce",
            source_sha="a" * 40,
            override_nonce="",
        )


def test_create_managed_pr_creates_documented_label_before_applying_it(tmp_path):
    runner = ManagedPrRunner(label_exists=False)

    with patch("coding_review_agent_loop.managed_pr.preflight_managed_ci_creation", _intent):
        create_managed_pr(
            runner,
            config=_config(tmp_path),
            source_branch="fix/missing-label",
            title="Create label",
            body="",
        )

    label_create_index = next(
        index
        for index, (cmd, _body) in enumerate(runner.calls)
        if cmd[:4] == ["gh", "api", "--method", "POST"]
        and cmd[4] == "repos/OWNER/REPO/labels"
    )
    label_apply_index = next(
        index
        for index, (cmd, _body) in enumerate(runner.calls)
        if cmd[:4] == ["gh", "api", "--method", "POST"]
        and cmd[4] == "repos/OWNER/REPO/issues/77/labels"
    )
    create_cmd, _ = runner.calls[label_create_index]
    assert create_cmd[-6:] == [
        "-f", "name=agent-loop-managed", "-f", "color=1f6feb", "-f",
        "description=Suppress intermediate CI; agent-loop dispatches exact-head final CI",
    ]
    assert label_create_index < label_apply_index


def test_create_managed_pr_refuses_existing_open_pr_before_writes(tmp_path):
    runner = ManagedPrRunner(existing_pulls=[{"number": 42, "state": "open"}])

    with pytest.raises(AgentLoopError, match="already has an open PR.*#42"):
        create_managed_pr(
            runner,
            config=_config(tmp_path),
            source_branch="fix/already-open",
            title="Duplicate",
            body="",
        )

    assert not any(
        cmd[cmd.index("--method") + 1] != "GET" for cmd, _ in runner.calls
    )


def test_create_managed_pr_closes_partial_draft_when_labeling_fails(tmp_path):
    runner = ManagedPrRunner(label_failure=True)

    with (
        patch("coding_review_agent_loop.managed_pr.preflight_managed_ci_creation", _intent),
        pytest.raises(AgentLoopError, match="label failed"),
    ):
        create_managed_pr(
            runner,
            config=_config(tmp_path),
            source_branch="fix/label-failure",
            title="Label failure",
            body="",
        )

    methods_and_endpoints = [
        (cmd[cmd.index("--method") + 1], cmd[cmd.index("--method") + 2])
        for cmd, _ in runner.calls
        if "--method" in cmd
    ]
    assert ("PATCH", "repos/OWNER/REPO/pulls/77") in methods_and_endpoints
    assert any(method == "DELETE" and "/git/refs/" in endpoint for method, endpoint in methods_and_endpoints)


def test_create_managed_pr_closes_partial_draft_on_concurrent_duplicate(tmp_path):
    runner = ManagedPrRunner(
        post_create_pulls=[
            {"number": 77, "state": "open"},
            {"number": 78, "state": "open"},
        ]
    )

    with (
        patch("coding_review_agent_loop.managed_pr.preflight_managed_ci_creation", _intent),
        pytest.raises(AgentLoopError, match="Another PR was opened"),
    ):
        create_managed_pr(
            runner,
            config=_config(tmp_path),
            source_branch="fix/raced",
            title="Raced PR",
            body="",
        )

    assert any(
        "--method" in cmd
        and cmd[cmd.index("--method") + 1] == "PATCH"
        and cmd[cmd.index("--method") + 2].endswith("/pulls/77")
        for cmd, _ in runner.calls
    )


def test_create_managed_pr_reports_incomplete_rollback(tmp_path):
    runner = ManagedPrRunner(label_failure=True, cleanup_failure=True)

    with (
        patch("coding_review_agent_loop.managed_pr.preflight_managed_ci_creation", _intent),
        pytest.raises(AgentLoopError, match="automatic cleanup was incomplete.*close PR #77.*delete branch"),
    ):
        create_managed_pr(
            runner,
            config=_config(tmp_path),
            source_branch="fix/rollback-failure",
            title="Rollback failure",
            body="",
        )


@pytest.mark.parametrize("source_branch", ["", "refs/heads/fix/x", "owner:fix/x", "main"])
def test_create_managed_pr_rejects_invalid_source_branch_before_github(tmp_path, source_branch):
    runner = ManagedPrRunner()

    with pytest.raises(AgentLoopError):
        create_managed_pr(
            runner,
            config=_config(tmp_path),
            source_branch=source_branch,
            title="Invalid source",
            body="",
        )

    assert runner.calls == []


@pytest.mark.parametrize("marker", [SOURCE_MARKER, UNPROTECTED_OVERRIDE_TRAILER])
def test_create_managed_pr_rejects_reserved_body_markers(tmp_path, marker):
    runner = ManagedPrRunner()

    with pytest.raises(AgentLoopError, match="reserved managed-PR protocol marker"):
        create_managed_pr(
            runner,
            config=_config(tmp_path),
            source_branch="fix/body",
            title="Reserved marker",
            body=f"Unexpected {marker}",
        )

    assert runner.calls == []
