"""Scoped default-branch closure verifies evidence and confirms every write."""

import pytest

import coding_review_agent_loop.integration_close as closure
from coding_review_agent_loop.github import IssueContext
from agent_loop_helpers import make_config
from default_branch_close_helpers import DefaultCloseRunner, MERGE_SHA


def recover(tmp_path, runner, **overrides):
    return closure.close_staged_child_after_default_merge(
        runner, config=make_config(tmp_path, **overrides), pr_number=7,
        issue_context=IssueContext(42, "OWNER/REPO", "Child", "", "url", ()),
    )


@pytest.mark.parametrize("body", ["Fixes #42", "> Quoted example: Fixes #43\n\nFixes #42"])
def test_closes_with_commit_and_confirms(tmp_path, body):
    runner = DefaultCloseRunner()
    runner.live_pr["body"] = body
    assert recover(tmp_path, runner) == "closed"
    assert len(runner.closes) == 1
    cmd = runner.closes[0]
    assert cmd[cmd.index("--reason") + 1] == "completed"
    assert "PR #7" in cmd[-1] and MERGE_SHA in cmd[-1]
    assert runner.evidence_commands[-1] == ["gh", "api", "repos/OWNER/REPO/issues/42"]
    assert recover(tmp_path, runner) == "already-closed"
    assert len(runner.closes) == 1


@pytest.mark.parametrize("body", ["", "Refs #42", "> Fixes #42", "> Example:\nFixes #42", "```\nFixes #42\n```", "`Fixes #42`", "Fixes #43", "Fixes OTHER/REPO#42"])
def test_missing_affirmative_closing_reference_never_writes(tmp_path, body):
    runner = DefaultCloseRunner()
    runner.live_pr["body"] = body
    assert recover(tmp_path, runner) == "unresolved"
    assert not runner.closes


@pytest.mark.parametrize("failure", [
    "pr-read", "pr-malformed", "not-merged", "missing-merged", "merge-date", "empty-merge-date", "base", "repo", "default-read",
    "default-missing", "commit-read", "commit-missing", "commit-malformed", "child-read", "child-unknown",
    "recorded-unreadable", "recorded-conflict",
])
def test_unverified_evidence_never_writes(tmp_path, monkeypatch, failure):
    runner = DefaultCloseRunner()
    if failure == "pr-read":
        runner.unreadable.add("repos/OWNER/REPO/pulls/7")
    if failure == "pr-malformed":
        runner.live_pr = []
    if failure == "empty-merge-date":
        runner.live_pr["merged_at"] = ""
    if failure == "not-merged":
        runner.live_pr["merged"] = False
    if failure == "missing-merged":
        runner.live_pr.pop("merged")
    if failure == "merge-date":
        runner.live_pr["merged_at"] = None
    if failure == "base":
        runner.live_pr["base"]["ref"] = "other"
    if failure == "repo":
        runner.live_pr["base"]["repo"]["full_name"] = "OTHER/REPO"
    if failure == "default-read":
        runner.unreadable.add("repos/OWNER/REPO")
    if failure == "default-missing":
        runner.default = {}
    if failure == "commit-read":
        runner.unreadable.add("commit")
    if failure == "commit-missing":
        runner.commit = {}
    if failure == "commit-malformed":
        runner.commit = {"mergeCommit": {"oid": "short"}}
    if failure == "child-read":
        runner.unreadable.add("repos/OWNER/REPO/issues/42")
    if failure == "child-unknown":
        runner.state = "unknown"
    if failure == "recorded-unreadable":
        def unavailable(*_a, **_kw):
            raise closure._RecordedBaseUnavailable("unreadable")
        monkeypatch.setattr(closure, "_authorized_base", unavailable)
    if failure == "recorded-conflict":
        monkeypatch.setattr(closure, "_authorized_base", lambda *_a: "integration/old")
    assert recover(tmp_path, runner) == "unresolved"
    assert not runner.closes


@pytest.mark.parametrize("failure", ["close", "unconfirmed", "readback"])
def test_closure_failure_retries_from_live_state(tmp_path, failure):
    runner = DefaultCloseRunner()
    runner.close_rc = int(failure == "close")
    runner.confirm = failure != "unconfirmed"
    original = runner.run
    def run(args, **kwargs):
        result = original(args, **kwargs)
        if list(args)[:3] == ["gh", "issue", "close"] and failure == "readback":
            runner.unreadable.add("repos/OWNER/REPO/issues/42")
        return result
    runner.run = run
    assert recover(tmp_path, runner) == "unresolved"
    runner.run = original
    runner.unreadable.clear()
    runner.close_rc = 0
    runner.confirm = True
    assert recover(tmp_path, runner) == ("already-closed" if failure == "readback" else "closed")
    assert len(runner.closes) == (1 if failure == "readback" else 2)


def test_already_closed_race_and_dry_run_do_not_write(tmp_path):
    runner = DefaultCloseRunner()
    runner.state = "closed"
    assert recover(tmp_path, runner) == "already-closed"
    assert not runner.closes
    runner.evidence_commands.clear()
    assert recover(tmp_path, runner, dry_run=True) == "skipped"
    assert not runner.evidence_commands
