"""Orchestrator-level tests for GitHub merge-conflict detection and routing (#606)."""
from unittest.mock import patch

import coding_review_agent_loop.orchestrator as orchestrator
from coding_review_agent_loop.cli import run_pr_loop

from agent_loop_helpers import FakeRunner, command_index, make_config


def _codex_exec_commands(runner):
    return [cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]


def _claude_commands(runner):
    return [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]


def _check_runs_commands(runner):
    return [
        cmd
        for cmd, _cwd in runner.commands
        if cmd[:2] == ["gh", "api"] and cmd[2].endswith("/check-runs")
    ]


def _merge_commands(runner):
    return [cmd for cmd, _cwd in runner.commands if cmd[:3] == ["gh", "pr", "merge"]]


def test_conflict_at_round_start_skips_reviewers_and_routes_to_coder(tmp_path):
    runner = FakeRunner(
        claude_outputs=["Resolved the conflict.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        mergeability_payloads=[
            {
                "mergeable": "CONFLICTING",
                "mergeStateStatus": "DIRTY",
                "headRefOid": "abc123",
                "baseRefName": "main",
            },
        ],
    )
    config = make_config(tmp_path)

    result = run_pr_loop(runner, pr_number=77, config=config)

    assert result == 0
    # No reviewer invocation and no CI checks fetch happened before the coder
    # resolved the conflict: only one codex call (round 2's review, after the
    # conflict was resolved) and it comes after the single claude call.
    assert len(_codex_exec_commands(runner)) == 1
    assert len(_claude_commands(runner)) == 1
    claude_index = command_index(runner.commands, ["claude"])
    codex_index = command_index(runner.commands, ["codex", "exec"])
    assert claude_index < codex_index
    # Zero check-run/status/branch-protection calls happened before the coder
    # was dispatched to resolve the conflict.
    check_runs_index_before_claude = [
        cmd for cmd, _cwd in runner.commands[:claude_index]
        if cmd[:2] == ["gh", "api"] and cmd[2].endswith("/check-runs")
    ]
    assert check_runs_index_before_claude == []


def test_conflict_after_review_before_merge_blocks_watcher_and_merge(tmp_path):
    runner = FakeRunner(
        claude_outputs=["Resolved the conflict.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        mergeability_payloads=[
            {
                "mergeable": "MERGEABLE",
                "mergeStateStatus": "CLEAN",
                "headRefOid": "abc123",
                "baseRefName": "main",
            },
            {
                "mergeable": "CONFLICTING",
                "mergeStateStatus": "DIRTY",
                "headRefOid": "abc123",
                "baseRefName": "main",
            },
        ],
    )
    config = make_config(tmp_path, auto_merge=True)

    with patch(
        "coding_review_agent_loop.orchestrator.watch_pr_checks",
        wraps=orchestrator.watch_pr_checks,
    ) as watch_spy, patch("coding_review_agent_loop.orchestrator.merge_pr") as merge_spy:
        result = run_pr_loop(runner, pr_number=77, config=config)

    assert result == 0
    # The merge-gate probe (right before CI checks) caught the conflict, so
    # CI was never polled and the merge was never attempted in round 1;
    # both only happen after the coder resolves and round 2 re-approves.
    assert watch_spy.call_count == 1
    assert merge_spy.call_count == 1
    assert len(_codex_exec_commands(runner)) == 2
    assert len(_claude_commands(runner)) == 1


def test_unknown_mergeability_does_not_block_normal_flow(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={"mergeable": "UNKNOWN", "mergeStateStatus": "UNKNOWN"},
    )
    config = make_config(tmp_path, auto_merge=True, mergeability_poll_attempts=1)

    result = run_pr_loop(runner, pr_number=77, config=config)

    assert result == 0
    assert len(_merge_commands(runner)) == 1
    assert not any("conflict" in comment.lower() for comment in runner.comments)


def test_persistent_conflict_with_unchanged_head_stops_cleanly(tmp_path):
    runner = FakeRunner(
        claude_outputs=["Tried to resolve.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
        mergeability_payloads=[
            {
                "mergeable": "CONFLICTING",
                "mergeStateStatus": "DIRTY",
                "headRefOid": "abc123",
                "baseRefName": "main",
            },
            {
                "mergeable": "CONFLICTING",
                "mergeStateStatus": "DIRTY",
                "headRefOid": "abc123",
                "baseRefName": "main",
            },
        ],
        advance_pr_head_on_coder_followup=False,
    )
    config = make_config(tmp_path)

    result = run_pr_loop(runner, pr_number=77, config=config)

    assert result == 0
    # Only one coder round was dispatched; the second round detected the
    # unchanged head and stopped cleanly instead of dispatching again.
    assert len(_claude_commands(runner)) == 1
    assert any("still conflicted" in comment for comment in runner.comments)


def test_confirmed_conflict_survives_transient_unknown_reprobe_on_same_head(tmp_path):
    """Regression (#609 review): DIRTY at round start, the coder is
    dispatched but does not push (head unchanged), and the next round's probe
    comes back `UNKNOWN` (a transient GitHub failure) for that same head. The
    confirmed conflict must be preserved rather than silently cleared -- which
    would otherwise let reviewers run, GitHub checks be fetched, and (with
    auto-merge) a merge be attempted against a branch GitHub last told us was
    still conflicted. Instead, the unchanged-head bounded-progress guard
    should fire: no second coder round, no reviewer, no checks fetch, no
    merge."""
    runner = FakeRunner(
        claude_outputs=["Tried to resolve.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        mergeability_payloads=[
            {
                "mergeable": "CONFLICTING",
                "mergeStateStatus": "DIRTY",
                "headRefOid": "abc123",
                "baseRefName": "main",
            },
            {"mergeable": "UNKNOWN", "mergeStateStatus": "UNKNOWN"},
        ],
        advance_pr_head_on_coder_followup=False,
    )
    config = make_config(tmp_path, auto_merge=True, mergeability_poll_attempts=1)

    result = run_pr_loop(runner, pr_number=77, config=config)

    assert result == 0
    assert len(_claude_commands(runner)) == 1
    assert _codex_exec_commands(runner) == []
    assert _check_runs_commands(runner) == []
    assert _merge_commands(runner) == []
    assert any("still conflicted" in comment for comment in runner.comments)


def test_conflict_resolution_requires_fresh_ci_and_review(tmp_path):
    """A resolution push is treated as code-changing: stale approvals/checks
    are not reused, and the new head goes through a full reviewer + CI cycle
    before merge."""
    runner = FakeRunner(
        claude_outputs=["Resolved the conflict.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        mergeability_payloads=[
            {
                "mergeable": "CONFLICTING",
                "mergeStateStatus": "DIRTY",
                "headRefOid": "abc123",
                "baseRefName": "main",
            },
        ],
    )
    config = make_config(tmp_path, auto_merge=True)

    result = run_pr_loop(runner, pr_number=77, config=config)

    assert result == 0
    assert len(_codex_exec_commands(runner)) == 1
    assert len(_merge_commands(runner)) == 1
    merge_index = command_index(runner.commands, ["gh", "pr", "merge"])
    codex_index = command_index(runner.commands, ["codex", "exec"])
    claude_index = command_index(runner.commands, ["claude"])
    assert claude_index < codex_index < merge_index
