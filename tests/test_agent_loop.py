import ast
import base64
import contextlib
import datetime
import errno
import inspect
import json
import os
import sys
import textwrap
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

import orchestrator_split_guard
import coding_review_agent_loop.cli as cli_module
import coding_review_agent_loop.prompts as prompts_module

from coding_review_agent_loop.agents.base import AgentResult, with_public_response_file_instruction
from coding_review_agent_loop.cli import (
    AgentLoopError,
    Runner,
    build_parser,
    config_from_args,
    ensure_log_dir_ignored,
    is_clarification_request,
    main,
    run_issue_loop,
    run_pr_loop,
    run_task_loop,
)
from coding_review_agent_loop.comment_rendering import normalize_freeform_signature
from coding_review_agent_loop.errors import HumanDecisionRequiredError
from coding_review_agent_loop.config import (
    default_agent_memory_dir,
    default_agent_workdir,
    default_run_worktree_root,
    default_cache_root,
    resolve_base_branch,
)
from coding_review_agent_loop.errors import QuotaResetExceededError
from coding_review_agent_loop.github import IssueComment
from coding_review_agent_loop.managed_ci import ManagedCiReadiness, ProtectionAssessment
from coding_review_agent_loop.managed_pr import ManagedPrHandoff
from coding_review_agent_loop.runner import ExecutableIdentity
from coding_review_agent_loop.panel_evidence import PostedRoundMetadata, _plan_subject
from coding_review_agent_loop.agent_failure import (
    ValidatedAgentResponse,
    _parse_rate_limit_reset_seconds,
)
from coding_review_agent_loop.discuss_loop import _attach_round_metadata
from coding_review_agent_loop.round_state import _decode_round_metadata, _encode_round_metadata, _strip_round_metadata
from coding_review_agent_loop.pr_loop_support import _resume_plan_round
from coding_review_agent_loop.plan_first_loop import render_canonical_plan_revision
from coding_review_agent_loop.response_validation import render_public_agent_comment
from coding_review_agent_loop.protocol import (
    UnresolvedReviewItem,
    parse_plan_item_dispositions,
    parse_plan_review,
    parse_pr_review,
    validate_structured_issue_implementation,
    validate_structured_coder_followup,
    validate_structured_plan_revision,
)
from agent_loop_helpers import (
    FakeRunner,
    command_index,
    make_config,
    prior_item_dispositions,
    prior_plan_item_dispositions,
    structured_coder_followup,
    structured_issue_implementation,
    structured_plan_review,
    structured_plan_revision,
    structured_pr_review,
)
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.plan_first_loop as plan_first_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.triage as triage
import coding_review_agent_loop.validated_agent as validated_agent
from agent_loop_helpers import patch_attr_in, patch_object_in


def test_pre_review_tests_can_be_disabled(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Created PR.\nTests: pytest passed.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={"body": "Fixes #56"},
    )
    config = make_config(
        tmp_path,
        test_command=("pytest", "tests/test_agent_loop.py"),
        pre_review_tests=False,
    )

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    first_test = commands.index(["pytest", "tests/test_agent_loop.py"])
    first_review = command_index(runner.commands, ["codex", "exec"])
    assert first_review < first_test
    assert commands.count(["pytest", "tests/test_agent_loop.py"]) == 1


def test_issue_mode_passes_complete_expected_closing_set_to_coder(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={"body": "Fixes #56\nFixes #847"},
    )
    config = make_config(tmp_path, expected_closing_issue_ids=(847,))

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    coder_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "separate GitHub closing phrase/reference pair" in coder_prompt
    assert "Closes #847" in coder_prompt


def test_approved_plan_additional_closing_issue_reaches_implementation_flow(tmp_path):
    plan_output = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_state",
                "state": "blocking",
                "summary": "Implement the change.",
                "plan_steps": ["Implement the change."],
                "additional_closing_issue_ids": [847],
                "human_requirement_dispositions": [],
                "architecture_impact": {
                    "status": "unchanged",
                    "rationale": "No architectural contract changed.",
                    "affected_components": [], "dependencies": [],
                    "execution_data_flows": [], "persistence": [],
                    "public_contracts": [], "security_boundaries": [],
                    "canonical_document_action": "no-change",
                    "canonical_document_path": None,
                    "canonical_document_rationale": "",
                },
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = FakeRunner(
        claude_outputs=[
            plan_output,
            "Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->",
        ],
        codex_outputs=[
            structured_plan_review(summary="Plan approved."),
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        pr_payload={"body": "Fixes #56\nFixes #847"},
    )
    config = make_config(tmp_path, plan_execution_mode="implement-one-shot")

    assert run_issue_loop(
        runner,
        issue_number=56,
        config=config,
        plan_first=True,
        implement_after_approval=True,
    ) == 0

    implementation_prompt = next(
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] and "Implement the approved plan" in cmd[-1]
    )
    assert "Closes #847" in implementation_prompt


def test_staged_child_closes_child_and_refs_parent(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        issue_payload={
            "body": "<!-- AGENT_SPLIT_CHILD: parent=12 key=" + "a" * 64 + " -->",
        },
        pr_payload={"body": "Fixes #56\nRefs #12"},
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    coder_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "Closes #56" in coder_prompt
    assert "Refs #12" in coder_prompt
    assert "Do NOT use a closing keyword" in coder_prompt


def test_direct_pr_without_metadata_remains_contract_unknown(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={"body": "Fixes #56"},
    )
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert not any(
        "AGENT_PR_EXPECTED_CLOSING_ISSUES" in comment
        for comment in runner.pr_payload["comments"]
    )


def test_direct_pr_explicit_metadata_uses_managed_origin_and_all_ids(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={"body": "Closes #847\nCloses #848"},
    )
    config = make_config(
        tmp_path,
        expected_closing_issue_ids=(847, 848),
        pr_origin_flow="managed-pr",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    contract_comments = [
        comment
        for comment in runner.pr_payload["comments"]
        if "AGENT_PR_EXPECTED_CLOSING_ISSUES" in comment["body"]
    ]
    assert len(contract_comments) == 1
    assert "Origin flow: managed-pr" in contract_comments[0]["body"]


def test_direct_pr_supersession_lineage_survives_plain_reruns(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        pr_payload={"body": "Closes #847\nCloses #848"},
    )

    assert run_pr_loop(
        runner,
        pr_number=77,
        config=make_config(tmp_path, expected_closing_issue_ids=(847,)),
    ) == 0
    assert sum(
        "AGENT_PR_EXPECTED_CLOSING_ISSUES" in comment["body"]
        for comment in runner.pr_payload["comments"]
    ) == 1

    assert run_pr_loop(
        runner,
        pr_number=77,
        config=make_config(
            tmp_path,
            expected_closing_issue_ids=(847, 848),
            supersede_expected_closing_contract=True,
        ),
    ) == 0
    assert sum(
        "AGENT_PR_EXPECTED_CLOSING_ISSUES" in comment["body"]
        for comment in runner.pr_payload["comments"]
    ) == 2

    assert run_pr_loop(runner, pr_number=77, config=make_config(tmp_path)) == 0
    assert sum(
        "AGENT_PR_EXPECTED_CLOSING_ISSUES" in comment["body"]
        for comment in runner.pr_payload["comments"]
    ) == 2


def test_ensure_log_dir_ignored_does_not_overwrite_existing_file(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    gitignore = log_dir / ".gitignore"
    gitignore.write_text("custom\n", encoding="utf-8")

    ensure_log_dir_ignored(log_dir)

    assert gitignore.read_text(encoding="utf-8") == "custom\n"


def test_logging_module_imports_without_fcntl(monkeypatch):
    """The shared logging module must remain importable on Windows."""
    import importlib

    import coding_review_agent_loop.logging as logging_module

    monkeypatch.setitem(sys.modules, "fcntl", None)
    try:
        reloaded = importlib.reload(logging_module)
        assert reloaded.fcntl is None
        assert reloaded.datetime_stamp()
    finally:
        monkeypatch.delitem(sys.modules, "fcntl", raising=False)
        importlib.reload(logging_module)


def test_gemini_public_response_file_is_inside_git_dir(tmp_path):
    valid = "LGTM from response file.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(gemini_outputs=["stdout should be ignored"], public_response_outputs=[valid])
    config = make_config(tmp_path, reviewer="gemini")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments == [f"**Review verdict:** Approved\n\n{valid}"]
    gemini_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["gemini"]]
    assert len(gemini_commands) == 1
    prompt = "\n".join(gemini_commands[0])
    expected_prefix = str(config.gemini_dir / ".git" / "agent-loop" / "responses" / "gemini")
    assert expected_prefix in prompt
    assert "/tmp/coding-review-agent-loop/responses/" not in prompt


def test_gemini_public_response_file_resolves_worktree_git_dir(tmp_path):
    valid = "LGTM from response file.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(gemini_outputs=["stdout should be ignored"], public_response_outputs=[valid])
    config = make_config(tmp_path, reviewer="gemini")
    git_dir = tmp_path / "main-repo" / ".git" / "worktrees" / "gemini"
    git_dir.mkdir(parents=True)
    (config.gemini_dir / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments == [f"**Review verdict:** Approved\n\n{valid}"]
    gemini_call = next(cmd for cmd, _cwd in runner.commands if cmd[:1] == ["gemini"])
    assert str(git_dir / "agent-loop" / "responses" / "gemini") in gemini_call[2]
    assert str(config.gemini_dir / ".git" / "agent-loop") not in gemini_call[2]


# ---------------------------------------------------------------------------
# Issue #271: coder_followup path through attempt_envelope_normalization
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Issue #275: strip_unknown_prior_item_dispositions with tightly-packed input
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Issue #274: combined envelope+disposition strip via _run_validated_agent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected_secs", [
    ("Retry-After: 3600", 3600),
    ("retry after 1800", 1800),
    ("retryDelay: '7200s'", 7200),
    ("try again in 2h 30m", 9000),
    ("try again in 45m", 2700),
    ("resets in 1h", 3600),
    ("reset in 5m", 300),
])
def test_parse_rate_limit_reset_seconds(text, expected_secs):
    assert _parse_rate_limit_reset_seconds(text) == expected_secs


def test_parse_rate_limit_reset_seconds_claude_absolute_time():
    now = datetime.datetime(2026, 6, 3, 5, 33, 48, tzinfo=datetime.timezone.utc)
    text = "You've hit your session limit · resets 1:30am (America/Los_Angeles)"

    assert _parse_rate_limit_reset_seconds(text, now_utc=now) == 10572


@pytest.mark.parametrize("text", [
    "HTTP 429: rate limit exceeded.",
    "Too many requests.",
    "quota exceeded",
])
def test_parse_rate_limit_reset_seconds_returns_none_when_unparseable(text):
    assert _parse_rate_limit_reset_seconds(text) is None


def test_quota_reset_exceeded_error_exit_code():
    assert QuotaResetExceededError.EXIT_CODE == 3


def test_agent_memory_default_parent_ignores_generated_contents(tmp_path):
    runner = FakeRunner(codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"])
    # The memory directory stays outside every checkout: the review-round
    # spool beside it must not be reviewer-reachable (#1560).
    config = make_config(tmp_path, agent_memory_dir=tmp_path / "outside" / ".agent-loop" / "memory")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    gitignore = tmp_path / "outside" / ".agent-loop" / ".gitignore"
    assert gitignore.read_text(encoding="utf-8") == "*\n!.gitignore\n"


def test_agent_memory_does_not_ignore_custom_parent_directory(tmp_path):
    runner = FakeRunner(codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"])
    memory_dir = tmp_path / "custom-memory"
    config = make_config(tmp_path, agent_memory_dir=memory_dir)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert not (tmp_path / ".gitignore").exists()


def test_agent_memory_detects_changed_files_since_previous_commit(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        git_head="def456",
        changed_files=["src/coding_review_agent_loop/prompts.py", "tests/test_agent_loop.py"],
    )
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "last-analyzed-commit").write_text("abc123\n", encoding="utf-8")
    config = make_config(tmp_path, agent_memory_dir=memory_dir)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    diff_commands = [cmd for cmd, _cwd in runner.commands if cmd[:3] == ["git", "diff", "--name-only"]]
    assert ["git", "diff", "--name-only", "abc123..def456"] in diff_commands
    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "src/coding_review_agent_loop/prompts.py" in prompt
    assert "tests/test_agent_loop.py" in prompt
    assert (memory_dir / "last-analyzed-commit").read_text(encoding="utf-8") == "def456\n"


def test_agent_memory_logs_when_changed_file_diff_falls_back(tmp_path, capsys):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        git_head="def456",
        diff_returncode=128,
        diff_stderr="fatal: bad revision 'abc123..def456'",
    )
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "last-analyzed-commit").write_text("abc123\n", encoding="utf-8")
    config = make_config(tmp_path, agent_memory_dir=memory_dir, quiet=False)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    captured = capsys.readouterr()
    assert "Could not diff agent memory baseline abc123..def456" in captured.err
    assert "treating all tracked files as changed" in captured.err
    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "README.md" in prompt


def test_test_profile_records_provided_test_command(tmp_path):
    runner = FakeRunner(codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"])
    memory_dir = tmp_path / "memory"
    config = make_config(
        tmp_path,
        agent_memory_dir=memory_dir,
        test_command=("python", "-m", "pytest", "-q"),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    profile = (memory_dir / "test-profile.md").read_text(encoding="utf-8")
    assert "`python -m pytest -q`" in profile
    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "prefer verified test commands from the execution profile" in prompt


def test_agent_memory_can_be_disabled(tmp_path):
    runner = FakeRunner(codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"])
    memory_dir = tmp_path / "memory"
    config = make_config(tmp_path, agent_memory=False, agent_memory_dir=memory_dir)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert not memory_dir.exists()
    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "Agent memory context:" not in prompt


def test_resume_plan_round_marks_empty_ledger_incomplete_after_same_subject_prior_new_items():
    plan = "Plan text."
    subject = _plan_subject(plan)
    prior_new_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Prior same-plan item.",
        status="blocking",
    )
    prior_review_comment = _attach_round_metadata(
        "Prior plan review.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject=subject,
            prior_items=(),
            new_items=(prior_new_item,),
            state="blocking",
        ),
    )
    current_coder_comment = _attach_round_metadata(
        plan + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=subject,
            prior_items=(),
            canonical_plan=plan,
        ),
    )

    resumed = _resume_plan_round(
        [
            IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=prior_review_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:01:00Z", body=current_coder_comment),
        ],
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    assert resumed[1].ledger_may_be_incomplete is True


def test_shared_workdir_requires_explicit_override(tmp_path):
    runner = FakeRunner()
    shared = tmp_path / "repo"
    shared.mkdir()
    config = make_config(tmp_path, claude_dir=shared, codex_dir=shared)

    with pytest.raises(AgentLoopError, match="same directory"):
        run_pr_loop(runner, pr_number=77, config=config)


def test_gemini_shared_workdir_requires_explicit_override(tmp_path):
    runner = FakeRunner()
    shared = tmp_path / "repo"
    shared.mkdir()
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        codex_dir=shared,
        gemini_dir=shared,
    )

    with pytest.raises(AgentLoopError, match="same directory"):
        run_pr_loop(runner, pr_number=77, config=config)


def test_missing_agent_workdirs_are_created(tmp_path):
    runner = FakeRunner(
        claude_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude"],
    )
    claude_dir = tmp_path / "missing" / "claude"
    codex_dir = tmp_path / "missing" / "codex"
    config = make_config(
        tmp_path,
        claude_dir=claude_dir,
        codex_dir=codex_dir,
        coder="codex",
        reviewer="claude",
        create_dirs=False,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert claude_dir.is_dir()
    assert codex_dir.is_dir()


def test_missing_gemini_workdir_is_created_when_configured(tmp_path):
    runner = FakeRunner(
        gemini_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"],
    )
    gemini_dir = tmp_path / "missing" / "gemini"
    config = make_config(
        tmp_path,
        reviewer="gemini",
        gemini_dir=gemini_dir,
        create_dirs=False,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert gemini_dir.is_dir()


def test_non_codex_loop_uses_active_workdir_for_github_and_tests(tmp_path):
    runner = FakeRunner(
        gemini_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"],
    )
    codex_dir = tmp_path / "inactive" / "codex"
    config = make_config(
        tmp_path,
        claude_dir=tmp_path / "missing" / "claude",
        codex_dir=codex_dir,
        gemini_dir=tmp_path / "missing" / "gemini",
        coder="claude",
        reviewer="gemini",
        test_command=("pytest", "tests/test_agent_loop.py"),
        create_dirs=False,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert not codex_dir.exists()
    github_or_test_cwds = [
        cwd
        for cmd, cwd in runner.commands
        if cmd[:1] == ["gh"] or cmd == ["pytest", "tests/test_agent_loop.py"]
    ]
    assert github_or_test_cwds
    bootstrap_pr_queries = [
        cwd
        for cmd, cwd in runner.commands
        if cmd[:3] == ["gh", "pr", "view"]
        and "number,title,headRefName,baseRefName,headRefOid,url,body,comments,reviews" in cmd
        and cwd != config.claude_dir
    ]
    assert bootstrap_pr_queries == [Path.cwd()]
    assert set(github_or_test_cwds) == {Path.cwd(), config.claude_dir}


def test_omitted_agent_dirs_default_to_repo_scoped_temp_checkouts(monkeypatch, tmp_path):
    parser = build_parser()
    cache_home = tmp_path / "cache"
    monkeypatch.setattr("coding_review_agent_loop.config.sys.platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    args = parser.parse_args([
        "task",
        "Fix the bug",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "codex",
        "--reviewer",
        "claude",
    ])

    config = config_from_args(args, FakeRunner())

    token = config.run_token
    assert token
    for agent, path in (
        ("codex", config.codex_dir),
        ("claude", config.claude_dir),
        ("gemini", config.gemini_dir),
        ("antigravity", config.antigravity_dir),
    ):
        assert path.parent == default_run_worktree_root("OWNER/REPO", agent).resolve()
        assert path.name == token
    assert dict(config.default_checkout_stores) == {
        agent: default_agent_workdir("OWNER/REPO", agent).resolve()
        for agent in ("claude", "codex", "gemini", "antigravity")
    }
    assert set(config.auto_agent_dirs) == {"claude", "codex", "gemini", "antigravity"}
    assert config.agent_memory_dir == (
        cache_home / "coding-review-agent-loop" / "repos" / "OWNER-REPO" / "memory"
    ).resolve()
    # The default log dir is durable and outside the removable worktree.
    artifacts = (cache_home / "coding-review-agent-loop" / "run-artifacts" / "OWNER-REPO" / token)
    assert config.log_dir == artifacts / ".agent-loop-logs"
    assert config.codex_dir not in config.log_dir.parents
    # A second invocation gets its own token and worktrees.
    other = config_from_args(args, FakeRunner())
    assert other.run_token != token
    assert other.codex_dir != config.codex_dir
    # A log dir that aliases a runs/ root through a symlink is rejected.
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "scratch"))
    runs_root = default_run_worktree_root("OWNER/REPO", "codex").resolve()
    runs_root.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(runs_root, target_is_directory=True)
    alias_args = parser.parse_args([
        "task", "Fix the bug", "--repo", "OWNER/REPO", "--coder", "codex",
        "--reviewer", "claude", "--log-dir", str(alias / "logs"),
    ])
    with pytest.raises(AgentLoopError, match="--log-dir"):
        config_from_args(alias_args, FakeRunner())
    # A relative --agent-memory-dir stays repo-keyed, not under the worktree.
    memory_args = parser.parse_args([
        "task", "Fix the bug", "--repo", "OWNER/REPO", "--coder", "codex",
        "--reviewer", "claude", "--agent-memory-dir", "my-memory",
    ])
    memory_config = config_from_args(memory_args, FakeRunner())
    assert memory_config.agent_memory_dir == (
        cache_home / "coding-review-agent-loop" / "repos" / "OWNER-REPO" / "my-memory"
    ).resolve()


def test_run_tests_preflight_is_a_non_mutating_success_path(monkeypatch, capsys):
    assert main(["run-tests", "--preflight"]) == 0
    assert capsys.readouterr().out.strip() == "agent-loop preflight: verified"


def test_config_enables_ci_watch_with_auto_merge_without_rebuilding_antigravity_model(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr", "77", "--repo", "OWNER/REPO", "--auto-merge",
        "--antigravity-model", "Gemini 3.1 Pro (High)",
    ])
    config = config_from_args(
        args,
        FakeRunner(),
        invocation_argv=("agent-loop", "pr", "77"),
    )
    assert config.watch_pending_ci is True
    assert config.watch_pending_ci_explicit is False
    assert config.managed_ci_pr_mode is True
    assert config.invocation_argv[-1] == "77"
    assert config.antigravity_models == ("Gemini 3.1 Pro (High)",)


def test_config_records_explicit_managed_ci_without_implying_auto_merge(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr", "77", "--repo", "OWNER/REPO", "--managed-ci",
        "--managed-ci-trusted-actor", "agent-loop",
    ])

    config = config_from_args(args, FakeRunner())

    assert config.managed_ci is True
    assert config.auto_merge is False
    assert config.effective_managed_ci is True
    assert config.watch_pending_ci is False


def test_explicit_managed_ci_requires_trusted_actor_before_agent_preflight(capsys):
    assert main(["pr", "77", "--repo", "OWNER/REPO", "--managed-ci"]) == 1
    assert "--managed-ci requires --managed-ci-trusted-actor" in capsys.readouterr().err


def test_cli_distinguishes_human_decision_from_generic_failure(capsys):
    with patch.object(
        cli_module,
        "run_pr_loop",
        side_effect=HumanDecisionRequiredError("Reviewer positions conflict."),
    ):
        assert main(["pr", "77", "--repo", "OWNER/REPO"]) == 4

    stderr = capsys.readouterr().err
    assert "HUMAN DECISION REQUIRED" in stderr
    assert "Reviewer positions conflict." in stderr


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["pr", "77", "--repo", "OWNER/REPO", "--managed-ci-fresh"],
         "--managed-ci-fresh requires --managed-ci"),
        (["pr", "77", "--repo", "OWNER/REPO", "--managed-ci", "--managed-ci-fresh",
          "--managed-ci-trusted-actor", "agent-loop"],
         "--managed-ci-fresh requires --allow-unprotected-managed-ci"),
        (["pr", "77", "--repo", "OWNER/REPO", "--managed-ci", "--managed-ci-fresh",
          "--allow-unprotected-managed-ci", "--managed-ci-trusted-actor", "agent-loop"],
         "--managed-ci-fresh in PR mode requires --managed-ci-issue"),
        (["pr", "77", "--repo", "OWNER/REPO", "--managed-ci-issue", "56"],
         "--managed-ci-issue requires --managed-ci-fresh"),
    ],
)
def test_managed_ci_fresh_cli_rejects_incomplete_authorization(argv, message, capsys):
    assert main(argv) == 1
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["issue", "56", "--repo", "OWNER/REPO", "--managed-ci",
         "--managed-ci-trusted-actor", "agent-loop", "--allow-unreadable-protection"],
        ["pr", "77", "--repo", "OWNER/REPO", "--managed-ci",
         "--managed-ci-trusted-actor", "agent-loop", "--allow-unreadable-protection"],
        ["pr", "77", "--repo", "OWNER/REPO", "--auto-merge",
         "--managed-ci-trusted-actor", "agent-loop", "--managed-ci-adopt-existing-pr",
         "--allow-unreadable-protection"],
        ["managed-pr", "--repo", "OWNER/REPO", "--head", "fix/direct-change",
         "--title", "Direct change", "--managed-ci",
         "--managed-ci-trusted-actor", "agent-loop", "--allow-unreadable-protection"],
    ],
)
def test_allow_unreadable_protection_requires_the_unprotected_waiver(
    argv, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(cli_module, "config_from_args", lambda *args, **kwargs: make_config(tmp_path))
    # The validation error precedes every GitHub call, including base resolution.
    monkeypatch.setattr(
        cli_module, "resolve_base_branch",
        lambda *_a, **_k: pytest.fail("GitHub was read before flag validation"),
    )
    assert main(argv) == 1
    assert (
        "--allow-unreadable-protection requires --allow-unprotected-managed-ci"
        in capsys.readouterr().err
    )


def test_allow_unreadable_protection_with_adoption_keeps_adoption_refusal(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(cli_module, "config_from_args", lambda *args, **kwargs: make_config(tmp_path))
    assert main([
        "pr", "77", "--repo", "OWNER/REPO", "--auto-merge",
        "--managed-ci-trusted-actor", "agent-loop", "--managed-ci-adopt-existing-pr",
        "--allow-unprotected-managed-ci", "--allow-unreadable-protection",
    ]) == 1
    assert (
        "--allow-unprotected-managed-ci cannot be used with --managed-ci-adopt-existing-pr"
        in capsys.readouterr().err
    )


def test_config_records_unreadable_protection_waiver():
    args = build_parser().parse_args([
        "issue", "1040", "--repo", "OWNER/REPO", "--managed-ci",
        "--managed-ci-trusted-actor", "agent-loop", "--allow-unprotected-managed-ci",
        "--allow-unreadable-protection",
    ])

    config = config_from_args(args, FakeRunner())

    assert config.allow_unprotected_managed_ci is True
    assert config.allow_unreadable_protection is True


def test_config_records_explicit_existing_pr_managed_ci_adoption(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr", "77", "--repo", "OWNER/REPO", "--auto-merge",
        "--managed-ci-trusted-actor", "agent-loop", "--managed-ci-adopt-existing-pr",
    ])

    config = config_from_args(args, FakeRunner())

    assert config.managed_ci_adopt_existing_pr is True


def test_config_records_per_invocation_unprotected_managed_ci_override(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "issue", "658", "--repo", "OWNER/REPO", "--auto-merge",
        "--managed-ci-trusted-actor", "agent-loop", "--allow-unprotected-managed-ci",
    ])

    config = config_from_args(args, FakeRunner())

    assert config.allow_unprotected_managed_ci is True


def test_managed_pr_parser_accepts_direct_branch_metadata():
    args = build_parser().parse_args([
        "managed-pr",
        "--repo", "OWNER/REPO",
        "--head", "fix/direct-change",
        "--title", "Fix direct change",
        "--body-file", "description.md",
        "--auto-merge",
        "--managed-ci-trusted-actor", "agent-loop",
        "--allow-unprotected-managed-ci",
    ])

    assert args.command == "managed-pr"
    assert args.head == "fix/direct-change"
    assert args.title == "Fix direct change"
    assert args.body_file == Path("description.md")
    assert args.allow_unprotected_managed_ci is True


def test_managed_pr_parser_accepts_manual_managed_ci():
    args = build_parser().parse_args([
        "managed-pr",
        "--repo", "OWNER/REPO",
        "--head", "fix/direct-change",
        "--title", "Fix direct change",
        "--managed-ci",
        "--managed-ci-trusted-actor", "agent-loop",
    ])

    assert args.command == "managed-pr"
    assert args.managed_ci is True
    assert args.auto_merge is False


def test_managed_pr_requires_managed_ci_or_auto_merge(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "config_from_args", lambda *args, **kwargs: make_config(tmp_path))

    assert main([
        "managed-pr", "--repo", "OWNER/REPO", "--head", "fix/direct-change", "--title", "Direct change",
    ]) == 1

    assert "managed-pr requires --managed-ci or --auto-merge" in capsys.readouterr().err


def test_managed_pr_requires_trusted_actor(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "config_from_args", lambda *args, **kwargs: make_config(tmp_path))

    assert main([
        "managed-pr", "--repo", "OWNER/REPO", "--head", "fix/direct-change", "--title", "Direct change",
        "--auto-merge",
    ]) == 1

    assert "managed-pr requires --managed-ci-trusted-actor" in capsys.readouterr().err


def test_managed_pr_refuses_dry_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "config_from_args", lambda *args, **kwargs: make_config(tmp_path))

    assert main([
        "managed-pr", "--repo", "OWNER/REPO", "--head", "fix/direct-change", "--title", "Direct change",
        "--managed-ci", "--managed-ci-trusted-actor", "agent-loop", "--dry-run",
    ]) == 1

    assert "managed-pr does not support --dry-run" in capsys.readouterr().err


def test_managed_pr_hands_created_draft_to_pr_loop(tmp_path, monkeypatch):
    config = make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop")
    handoff_config = replace(config, invocation_argv=())
    handoff = ManagedPrHandoff(77, handoff_config, "a" * 40, "agent-loop/managed-direct-token")
    captured = {}
    monkeypatch.setattr(cli_module, "config_from_args", lambda *args, **kwargs: config)
    monkeypatch.setattr(cli_module, "resolve_base_branch", lambda config, runner: config)
    monkeypatch.setattr(cli_module, "ensure_agent_workdirs", lambda config, runner: None)

    def create(runner, *, config, source_branch, title, body):
        captured["create"] = (config, source_branch, title, body)
        return handoff

    def run(runner, *, pr_number, config, workdirs_ready=False):
        captured["run"] = (pr_number, config, workdirs_ready)
        return 23

    monkeypatch.setattr(cli_module, "create_managed_pr", create)
    monkeypatch.setattr(cli_module, "run_pr_loop", run)

    assert main([
        "managed-pr", "--repo", "OWNER/REPO", "--head", "fix/direct-change", "--title", "Direct change",
        "--managed-ci", "--managed-ci-trusted-actor", "agent-loop",
    ]) == 23

    assert captured["create"][1:] == ("fix/direct-change", "Direct change", "")
    assert captured["run"] == (77, handoff_config, True)


def test_unprotected_managed_ci_override_rejects_existing_pr_adoption(capsys):
    assert main([
        "pr", "77", "--repo", "OWNER/REPO", "--auto-merge",
        "--managed-ci-trusted-actor", "agent-loop", "--managed-ci-adopt-existing-pr",
        "--allow-unprotected-managed-ci",
    ]) == 1
    assert "cannot be used with --managed-ci-adopt-existing-pr" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            ["pr", "77", "--managed-ci-fresh"],
            "--managed-ci-fresh requires --managed-ci",
        ),
        (
            [
                "pr", "77", "--managed-ci", "--managed-ci-fresh",
                "--managed-ci-trusted-actor", "agent-loop",
            ],
            "--managed-ci-fresh requires --allow-unprotected-managed-ci",
        ),
        (
            [
                "pr", "77", "--managed-ci", "--managed-ci-fresh",
                "--allow-unprotected-managed-ci",
            ],
            "--managed-ci requires --managed-ci-trusted-actor",
        ),
        (
            [
                "pr", "77", "--managed-ci", "--managed-ci-fresh",
                "--managed-ci-trusted-actor", "agent-loop",
                "--allow-unprotected-managed-ci",
            ],
            "--managed-ci-fresh in PR mode requires --managed-ci-issue",
        ),
        (
            [
                "pr", "77", "--managed-ci", "--managed-ci-issue", "56",
                "--managed-ci-trusted-actor", "agent-loop",
            ],
            "--managed-ci-issue requires --managed-ci-fresh",
        ),
    ],
)
def test_managed_ci_fresh_recovery_cli_validation_errors_are_explicit(
    tmp_path, monkeypatch, capsys, arguments, message,
):
    monkeypatch.setattr(
        cli_module,
        "config_from_args",
        lambda *_args, **_kwargs: make_config(tmp_path),
    )

    assert main(arguments) == 1
    assert message in capsys.readouterr().err


def test_managed_ci_preflight_reports_the_resolved_base(monkeypatch, capsys):
    result = ManagedCiReadiness(
        "strict_ready", "private", "agent-loop", "agent-loop", True, True,
        ProtectionAssessment("strict", "classic", "admins enforced"), (), (), "develop",
    )
    monkeypatch.setattr(cli_module, "evaluate_managed_ci_readiness", lambda *args, **kwargs: result)

    assert main(["managed-ci", "preflight", "--repo", "OWNER/REPO", "--trusted-actor", "agent-loop"]) == 0

    assert "base: develop" in capsys.readouterr().out


@pytest.mark.parametrize(("state", "expected"), [("override_eligible", 10), ("indeterminate", 11)])
def test_managed_ci_preflight_uses_documented_nonready_exit_codes(monkeypatch, state, expected):
    result = ManagedCiReadiness(
        state, "private", "agent-loop", "agent-loop", True, True,
        ProtectionAssessment("voluntary", "classic", "not enforced"), (), (), "main",
    )
    monkeypatch.setattr(cli_module, "evaluate_managed_ci_readiness", lambda *args, **kwargs: result)

    assert main(["managed-ci", "preflight", "--repo", "OWNER/REPO", "--trusted-actor", "agent-loop"]) == expected


def test_config_does_not_enable_ci_watch_without_auto_merge(tmp_path):
    parser = build_parser()
    args = parser.parse_args(["pr", "77", "--repo", "OWNER/REPO"])

    config = config_from_args(args, FakeRunner())

    assert config.watch_pending_ci is False


def test_config_allows_disabling_auto_merge_ci_watch(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr", "77", "--repo", "OWNER/REPO", "--auto-merge", "--no-watch-pending-ci",
    ])

    config = config_from_args(args, FakeRunner())

    assert config.watch_pending_ci is False
    assert config.watch_pending_ci_explicit is True


@pytest.mark.parametrize(
    "legacy_args",
    [["--ci-check-name", "legacy-ci"], ["--ci-check-name=legacy-ci"]],
)
def test_retired_ci_name_option_is_rejected(legacy_args, capsys):
    parser = build_parser()
    arguments = ["pr", "77", "--repo", "OWNER/REPO", "--auto-merge", *legacy_args]

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(arguments)

    assert exc_info.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_auto_merge_legacy_ci_watch_option_emits_warning(tmp_path, monkeypatch, capsys):
    config = make_config(
        tmp_path,
        auto_merge=True,
        watch_pending_ci=False,
        watch_pending_ci_explicit=True,
    )
    monkeypatch.setattr(cli_module, "config_from_args", lambda *args, **kwargs: config)
    monkeypatch.setattr(cli_module, "run_pr_loop", lambda *args, **kwargs: 0)

    assert main([
        "pr", "77", "--repo", "OWNER/REPO", "--auto-merge",
        "--no-watch-pending-ci",
    ]) == 0

    warning = capsys.readouterr().err
    assert "--no-watch-pending-ci is retained for compatibility" in warning


def test_config_honors_explicit_ci_watch_without_auto_merge(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr", "77", "--repo", "OWNER/REPO", "--watch-pending-ci",
    ])

    config = config_from_args(args, FakeRunner())

    assert config.watch_pending_ci is True
    assert config.auto_merge is False


@pytest.mark.parametrize(
    ("coder", "reviewer", "missing_command", "override_flag"),
    [
        ("claude", "codex", "missing-claude", "--claude-cmd"),
        ("claude", "gemini", "missing-gemini", "--gemini-cmd"),
    ],
)
def test_config_preflight_rejects_missing_agent_before_repo_detection(
    monkeypatch,
    coder,
    reviewer,
    missing_command,
    override_flag,
):
    parser = build_parser()
    args = parser.parse_args([
        "task",
        "Fix the bug",
        "--coder",
        coder,
        "--reviewer",
        reviewer,
        f"--{coder}-cmd",
        missing_command if override_flag == f"--{coder}-cmd" else coder,
        f"--{reviewer}-cmd",
        missing_command if override_flag == f"--{reviewer}-cmd" else reviewer,
    ])
    detection_calls = []
    monkeypatch.setattr(
        "coding_review_agent_loop.config.detect_repo",
        lambda *call_args: detection_calls.append(call_args),
    )
    monkeypatch.setattr(
        "coding_review_agent_loop.config.shutil.which",
        lambda command: None if command == missing_command else f"/bin/{command}",
    )

    with pytest.raises(
        AgentLoopError,
        match=rf"{missing_command} CLI not found on PATH.*{override_flag}",
    ):
        config_from_args(args, Runner())

    assert detection_calls == []


def test_config_preflight_checks_only_unique_configured_agents(monkeypatch, tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "task",
        "Fix the bug",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "codex",
        "--reviewer",
        "codex",
        "--codex-dir",
        str(tmp_path / "codex"),
    ])
    checked = []

    def fake_which(command):
        checked.append(command)
        return f"/bin/{command}"

    monkeypatch.setattr("coding_review_agent_loop.config.shutil.which", fake_which)

    config = config_from_args(args, Runner())

    assert config.coder == "codex"
    assert checked == ["codex", "agy"]


def test_config_preflight_accepts_custom_absolute_command(tmp_path):
    command = tmp_path / "custom-codex"
    command.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    command.chmod(0o755)
    parser = build_parser()
    args = parser.parse_args([
        "task",
        "Fix the bug",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "codex",
        "--reviewer",
        "codex",
        "--codex-cmd",
        str(command),
    ])

    config = config_from_args(args, Runner())

    assert config.codex_cmd == str(command)


def test_config_preflight_skips_dry_run_command_preview(monkeypatch):
    parser = build_parser()
    args = parser.parse_args([
        "task",
        "Fix the bug",
        "--repo",
        "OWNER/REPO",
        "--dry-run",
        "--claude-cmd",
        "missing-claude",
        "--codex-cmd",
        "missing-codex",
    ])
    monkeypatch.setattr(
        "coding_review_agent_loop.config.shutil.which",
        lambda command: pytest.fail(f"unexpected preflight for {command}"),
    )

    config = config_from_args(args, Runner(dry_run=True))

    assert config.dry_run is True


def test_preflight_absolute_path_valid(tmp_path):
    """Absolute path to an existing executable passes preflight and is stored."""
    from coding_review_agent_loop.config import preflight_agent_commands

    parser = build_parser()
    args = parser.parse_args([
        "pr", "77", "--repo", "OWNER/REPO",
        "--claude-cmd", sys.executable,
        "--codex-cmd", "codex",
    ])
    args.coder = "claude"
    runner = Runner()
    preflight_agent_commands(args, runner, ())
    assert runner._resolved_commands[sys.executable] == sys.executable


def test_preflight_absolute_path_not_found_gives_path_error(tmp_path):
    """Nonexistent absolute path gives 'not found or not executable', not 'not found on PATH'."""
    from coding_review_agent_loop.config import preflight_agent_commands

    nonexistent = str(tmp_path / "no-such-binary")
    parser = build_parser()
    args = parser.parse_args([
        "pr", "77", "--repo", "OWNER/REPO",
        "--claude-cmd", nonexistent,
    ])
    args.coder = "claude"
    with pytest.raises(AgentLoopError, match="not found or not executable"):
        preflight_agent_commands(args, Runner(), ())


def test_preflight_absolute_path_dangling_symlink_gives_path_error(tmp_path):
    """Dangling absolute-path symlink gives 'not found or not executable', not 'not found on PATH'."""
    from coding_review_agent_loop.config import preflight_agent_commands

    dangling = tmp_path / "dangling-claude"
    dangling.symlink_to(tmp_path / "missing-target")
    parser = build_parser()
    args = parser.parse_args([
        "pr", "77", "--repo", "OWNER/REPO",
        "--claude-cmd", str(dangling),
    ])
    args.coder = "claude"
    with pytest.raises(AgentLoopError, match="not found or not executable"):
        preflight_agent_commands(args, Runner(), ())


def test_preflight_bare_name_not_found_gives_path_message(monkeypatch):
    """Bare name not on PATH still gives 'not found on PATH' message."""
    from coding_review_agent_loop.config import preflight_agent_commands

    monkeypatch.setattr("coding_review_agent_loop.config.shutil.which", lambda cmd: None)
    parser = build_parser()
    args = parser.parse_args([
        "pr", "77", "--repo", "OWNER/REPO",
        "--claude-cmd", "missing-bare-name",
    ])
    args.coder = "claude"
    with pytest.raises(AgentLoopError, match="not found on PATH"):
        preflight_agent_commands(args, Runner(), ())


def test_omitted_cli_base_is_preserved_for_runtime_resolution(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
    ])

    assert config_from_args(args, FakeRunner()).base is None


def test_max_rounds_cli_default_is_shared_constant(tmp_path):
    from coding_review_agent_loop.config import DEFAULT_MAX_ROUNDS

    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
    ])

    assert args.max_rounds == DEFAULT_MAX_ROUNDS == 10
    assert config_from_args(args, FakeRunner()).max_rounds == DEFAULT_MAX_ROUNDS


def test_discuss_max_rounds_cli_is_discuss_only():
    parser = build_parser()
    args = parser.parse_args([
        "discuss",
        "56",
        "--repo",
        "OWNER/REPO",
        "--discuss-max-rounds",
        "4",
    ])

    assert args.discuss_max_rounds == 4

    with pytest.raises(SystemExit):
        parser.parse_args([
            "pr",
            "77",
            "--repo",
            "OWNER/REPO",
            "--discuss-max-rounds",
            "4",
        ])


def test_pre_review_tests_cli_defaults_on_and_can_be_disabled(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "task",
        "Fix the bug",
        "--repo",
        "OWNER/REPO",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
    ])
    config = config_from_args(args, FakeRunner())
    assert config.pre_review_tests is True

    args = parser.parse_args([
        "task",
        "Fix the bug",
        "--repo",
        "OWNER/REPO",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
        "--no-pre-review-tests",
    ])
    config = config_from_args(args, FakeRunner())
    assert config.pre_review_tests is False


@pytest.mark.parametrize("repo", ["OWNER", "OWNER/", "/REPO", "OWNER/REPO/EXTRA"])
def test_default_agent_workdir_rejects_invalid_repo_formats(repo):
    with pytest.raises(AgentLoopError, match="OWNER/REPO"):
        default_agent_workdir(repo, "codex")


def test_default_agent_memory_dir_uses_xdg_cache_and_repo_scope(monkeypatch, tmp_path):
    cache_home = tmp_path / "xdg-cache"
    monkeypatch.setattr("coding_review_agent_loop.config.sys.platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))

    assert default_agent_memory_dir("OWNER/REPO") == (
        cache_home / "coding-review-agent-loop" / "repos" / "OWNER-REPO" / "memory"
    )


def test_default_cache_root_uses_posix_home_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr("coding_review_agent_loop.config.sys.platform", "linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    assert default_cache_root() == tmp_path / ".cache" / "coding-review-agent-loop"


@pytest.mark.parametrize(
    ("platform", "home_parts"),
    [
        ("darwin", ("Library", "Caches", "coding-review-agent-loop")),
        ("win32", ("AppData", "Local", "coding-review-agent-loop", "Cache")),
    ],
)
def test_default_cache_root_uses_platform_home_fallbacks(
    monkeypatch,
    tmp_path,
    platform,
    home_parts,
):
    monkeypatch.setattr("coding_review_agent_loop.config.sys.platform", platform)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)

    assert default_cache_root() == tmp_path.joinpath(*home_parts)


def test_default_cache_root_uses_windows_local_app_data(monkeypatch, tmp_path):
    local_app_data = tmp_path / "local-app-data"
    monkeypatch.setattr("coding_review_agent_loop.config.sys.platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))

    assert default_cache_root() == local_app_data / "coding-review-agent-loop" / "Cache"


@pytest.mark.parametrize("repo", ["OWNER", "OWNER/", "/REPO", "OWNER/REPO/EXTRA"])
def test_default_agent_memory_dir_rejects_invalid_repo_formats(repo):
    with pytest.raises(AgentLoopError, match="OWNER/REPO"):
        default_agent_memory_dir(repo)


@pytest.mark.parametrize("mode", ["ignore", "summarize", "issue", "fix-and-summarize", "fix-and-issue"])
def test_approved_followups_cli_mode_is_configurable(tmp_path, mode):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--approved-followups",
        mode,
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
        "--gemini-dir",
        str(tmp_path / "gemini"),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.approved_followups == mode


@pytest.mark.parametrize(
    "mode",
    ["plan-only", "decompose-only", "implement-one-shot", "implement-by-phase", "auto"],
)
def test_plan_execution_mode_cli_is_configurable(tmp_path, mode):
    parser = build_parser()
    args = parser.parse_args([
        "issue",
        "56",
        "--repo",
        "OWNER/REPO",
        "--plan-first",
        "--plan-execution-mode",
        mode,
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
        "--gemini-dir",
        str(tmp_path / "gemini"),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.plan_execution_mode == mode


def test_plan_first_implementation_coder_cli_is_configurable(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "issue",
        "56",
        "--repo",
        "OWNER/REPO",
        "--plan-first",
        "--implement-after-approval",
        "--coder",
        "claude",
        "--claude-model",
        "claude-fable-5",
        "--implementation-coder",
        "codex",
        "--implementation-coder-model",
        "gpt-5.5",
        "--implementation-codex-reasoning-effort",
        "xhigh",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
        "--gemini-dir",
        str(tmp_path / "gemini"),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.coder == "claude"
    assert config.claude_model == "claude-fable-5"
    assert config.implementation_coder == "codex"
    assert config.implementation_coder_model == "gpt-5.5"
    assert config.implementation_codex_reasoning_effort == "xhigh"


def test_implementation_coder_options_require_issue_plan_first(capsys):
    assert main(["pr", "77", "--repo", "OWNER/REPO", "--implementation-coder", "codex"]) == 1
    assert (
        "--implementation-coder options are only supported with issue --plan-first"
        in capsys.readouterr().err
    )

    assert main(["issue", "56", "--repo", "OWNER/REPO", "--implementation-coder", "codex"]) == 1
    assert "--implementation-coder options require --plan-first" in capsys.readouterr().err

    assert main([
        "pr", "77", "--repo", "OWNER/REPO", "--implementation-claude-effort", "high"
    ]) == 1
    assert (
        "--implementation-coder options are only supported with issue --plan-first"
        in capsys.readouterr().err
    )

    assert main([
        "issue", "56", "--repo", "OWNER/REPO", "--implementation-claude-effort", "high"
    ]) == 1
    assert "--implementation-coder options require --plan-first" in capsys.readouterr().err


def test_plan_first_post_approval_options_require_plan_first(capsys):
    assert main(["issue", "56", "--repo", "OWNER/REPO", "--implement-after-approval"]) == 1
    assert "--implement-after-approval requires --plan-first" in capsys.readouterr().err

    assert (
        main([
            "issue",
            "56",
            "--repo",
            "OWNER/REPO",
            "--plan-execution-mode",
            "implement-one-shot",
        ])
        == 1
    )
    assert "--plan-execution-mode requires --plan-first" in capsys.readouterr().err


def test_preapproval_plan_execution_mode_reads_have_explicit_auto_coverage():
    """Keep every pre-approval mode consumer tied to the automatic policy contract."""
    from coding_review_agent_loop.config import PLAN_EXECUTION_MODES

    assert "auto" in PLAN_EXECUTION_MODES
    # The prompt guard reads the mode only through the shared predicate, which
    # is defined once in config and pinned against all five modes below.
    guard_source = inspect.getsource(prompts_module._phased_plan_guard)
    assert "phased_delivery_guard_active(config.plan_execution_mode)" in guard_source
    assert "in {" not in guard_source
    from coding_review_agent_loop.config import phased_delivery_guard_active

    assert {m for m in PLAN_EXECUTION_MODES if not phased_delivery_guard_active(m)} == {
        "decompose-only", "implement-by-phase", "auto",
    }
    sources = (inspect.getsource(cli_module.main),)
    for source in sources:
        tree = ast.parse(textwrap.dedent(source))
        mode_branches = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            condition = ast.unparse(node.test)
            if "plan_execution_mode" not in condition:
                continue
            if not any(operator in condition for operator in (" == ", " != ", " in ")):
                continue
            mode_branches.append(ast.unparse(node))
        assert mode_branches
        assert all("auto" in branch for branch in mode_branches)


@pytest.mark.parametrize("mode", ["plan-only", "decompose-only", "implement-by-phase", "auto"])
def test_cli_rejects_alias_with_conflicting_explicit_execution_mode(tmp_path, capsys, mode):
    result = main([
        "issue", "56", "--repo", "OWNER/REPO", "--plan-first",
        "--plan-execution-mode", mode, "--implement-after-approval",
        "--claude-dir", str(tmp_path / "claude"),
        "--codex-dir", str(tmp_path / "codex"),
        "--gemini-dir", str(tmp_path / "gemini"),
    ])

    assert result == 1
    assert "--implement-after-approval is only compatible" in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["decompose-only", "implement-by-phase", "auto"])
def test_cli_rejects_split_materialization_with_decomposition_modes(tmp_path, capsys, mode):
    result = main([
        "issue", "56", "--repo", "OWNER/REPO", "--plan-first",
        "--plan-execution-mode", mode, "--materialize-split-issues",
        "--claude-dir", str(tmp_path / "claude"),
        "--codex-dir", str(tmp_path / "codex"),
        "--gemini-dir", str(tmp_path / "gemini"),
        "--dry-run",
    ])

    assert result == 1
    assert "cannot be combined" in capsys.readouterr().err


def test_cli_rejects_split_stage_with_auto(tmp_path, capsys):
    result = main([
        "issue", "56", "--repo", "OWNER/REPO", "--plan-first",
        "--plan-execution-mode", "auto", "--split-stage", "1",
        "--claude-dir", str(tmp_path / "claude"),
        "--codex-dir", str(tmp_path / "codex"),
        "--gemini-dir", str(tmp_path / "gemini"),
        "--dry-run",
    ])

    assert result == 1
    assert "cannot be combined" in capsys.readouterr().err


def test_flat_child_limit_cli_is_configurable(tmp_path):
    args = build_parser().parse_args([
        "issue", "56", "--repo", "OWNER/REPO", "--flat-child-limit", "23",
        "--claude-dir", str(tmp_path / "claude"),
        "--codex-dir", str(tmp_path / "codex"),
        "--gemini-dir", str(tmp_path / "gemini"),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.flat_child_limit == 23


def test_implementation_codex_reasoning_effort_requires_codex_but_not_model(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "issue",
        "56",
        "--repo",
        "OWNER/REPO",
        "--plan-first",
        "--implement-after-approval",
        "--coder",
        "claude",
        "--implementation-codex-reasoning-effort",
        "high",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
        "--gemini-dir",
        str(tmp_path / "gemini"),
    ])
    with pytest.raises(AgentLoopError, match="requires --implementation-coder codex"):
        config_from_args(args, FakeRunner())

    args = parser.parse_args([
        "issue",
        "56",
        "--repo",
        "OWNER/REPO",
        "--plan-first",
        "--implement-after-approval",
        "--coder",
        "codex",
        "--implementation-codex-reasoning-effort",
        "high",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
        "--gemini-dir",
        str(tmp_path / "gemini"),
    ])
    config = config_from_args(args, FakeRunner())
    assert config.implementation_codex_reasoning_effort == "high"


def test_implementation_claude_effort_requires_claude_but_not_model(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "issue",
        "56",
        "--repo",
        "OWNER/REPO",
        "--plan-first",
        "--implement-after-approval",
        "--coder",
        "codex",
        "--implementation-claude-effort",
        "high",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
        "--gemini-dir",
        str(tmp_path / "gemini"),
    ])
    with pytest.raises(AgentLoopError, match="requires --implementation-coder claude"):
        config_from_args(args, FakeRunner())

    args = parser.parse_args([
        "issue",
        "56",
        "--repo",
        "OWNER/REPO",
        "--plan-first",
        "--implement-after-approval",
        "--coder",
        "claude",
        "--implementation-claude-effort",
        "high",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
        "--gemini-dir",
        str(tmp_path / "gemini"),
    ])
    config = config_from_args(args, FakeRunner())
    assert config.implementation_claude_effort == "high"


@pytest.mark.parametrize("reviewer", ["gemini", "codex"])
@pytest.mark.parametrize("parallel", [False, True])
def test_issue_loop_plan_first_can_switch_implementation_coder_end_to_end(tmp_path, reviewer, parallel):
    from coding_review_agent_loop.config import resolve_invocation

    plan_text = "Approved plan.\n<!-- AGENT_PLAN_STATE: approved -->\n-- Anthropic Claude"
    plan_review_text = structured_plan_review(summary="Plan approved.")
    pr_review_text = structured_pr_review(state="approved", summary="PR approved.")
    plan_review_marker = parse_plan_review(plan_review_text, reviewer="Google Gemini")
    pr_review_marker = parse_pr_review(pr_review_text, reviewer="Google Gemini")
    implementation_text = structured_issue_implementation(
        summary="Created PR 77.",
        pr_number=77,
        reviewer="OpenAI Codex",
    )
    calls = []
    runner = FakeRunner(pr_payload={"body": "Fixes #56"})
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer=reviewer,
        review_parallel=parallel,
        reviewer_codex_model="review-model",
        reviewer_codex_reasoning_effort="medium",
        implementation_coder="codex",
        implementation_coder_model="gpt-5.5",
        implementation_codex_reasoning_effort="high",
    )

    def fake_run_validated_agent(runner_arg, *, agent, config, session_id=None, **kwargs):
        invocation = resolve_invocation(config, provider=agent, role=kwargs.get("role"))
        calls.append(
            {
                "agent": agent,
                "coder": config.coder,
                "codex_model": config.codex_model,
                "codex_reasoning_effort": config.codex_reasoning_effort,
                "implementation_codex_reasoning_effort": config.implementation_codex_reasoning_effort,
                "session_id": session_id,
                "operation": kwargs.get("operation_description"),
                "model": invocation.configured_model,
                "effort": invocation.resolved_effort,
            }
        )
        operation = kwargs.get("operation_description")
        if operation == "planning":
            return ValidatedAgentResponse(
                text=plan_text,
                model_used="claude-fable-5",
                session_id="planning-session",
                marker_value=None,
            )
        if operation == "plan review":
            return ValidatedAgentResponse(
                text=plan_review_text,
                model_used="gemini-3-pro",
                session_id=None,
                marker_value=plan_review_marker,
            )
        if operation == "approved-plan implementation":
            # This fake stands in for run_agent_result: the coder commits in its own
            # checkout, and the real boundary then records that as the new baseline.
            from coding_review_agent_loop.checkout_verification import record_checkout_baseline
            from coding_review_agent_loop.workdirs import agent_workdir

            runner_arg._last_agent_cwd = agent_workdir(config, agent)
            runner_arg.git_head = "def456"
            record_checkout_baseline(
                config, runner_arg, agent_workdir(config, agent), source="coder turn"
            )
            return ValidatedAgentResponse(
                text=implementation_text,
                model_used="gpt-5.5 (high)",
                session_id="implementation-session",
                marker_value=validate_structured_issue_implementation(implementation_text),
            )
        return ValidatedAgentResponse(
            text=pr_review_text,
            model_used="gemini-3-pro",
            session_id=None,
            marker_value=pr_review_marker,
        )

    with patch_object_in((validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage, issue_loop), "_run_validated_agent",
        side_effect=fake_run_validated_agent,
    ):
        assert (
            run_issue_loop(
                runner,
                issue_number=56,
                config=config,
                plan_first=True,
                implement_after_approval=True,
            )
            == 0
        )

    assert [call["agent"] for call in calls] == ["claude", reviewer, "codex", reviewer]
    implementation_call = calls[2]
    assert implementation_call["coder"] == "codex"
    assert implementation_call["codex_model"] == "gpt-5.5"
    assert implementation_call["codex_reasoning_effort"] == ""
    assert implementation_call["implementation_codex_reasoning_effort"] == "high"
    assert implementation_call["session_id"] is None
    assert calls[0]["coder"] == "claude"
    assert (implementation_call["model"], implementation_call["effort"]) == ("gpt-5.5", "high")
    if reviewer == "codex":
        for call in (calls[1], calls[3]):
            assert (call["model"], call["effort"]) == ("review-model", "medium")


def test_explicit_agent_dirs_are_preserved_when_others_default(tmp_path):
    parser = build_parser()
    codex_dir = tmp_path / "codex"
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "codex",
        "--reviewer",
        "claude",
        "--codex-dir",
        str(codex_dir),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.codex_dir == codex_dir
    assert config.claude_dir.parent == default_run_worktree_root("OWNER/REPO", "claude").resolve()
    assert dict(config.default_checkout_stores)["claude"] == default_agent_workdir("OWNER/REPO", "claude").resolve()
    assert "codex" not in dict(config.default_checkout_stores)
    assert set(config.auto_agent_dirs) == {"claude", "gemini", "antigravity"}


def test_relative_log_dir_defaults_under_active_coder_workdir(tmp_path):
    parser = build_parser()
    claude_dir = tmp_path / "claude"
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "claude",
        "--reviewer",
        "gemini",
        "--claude-dir",
        str(claude_dir),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.log_dir == claude_dir / ".agent-loop-logs"


def test_agent_memory_flags_configure_memory_dir_and_refresh(tmp_path):
    parser = build_parser()
    codex_dir = tmp_path / "codex"
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "codex",
        "--reviewer",
        "claude",
        "--codex-dir",
        str(codex_dir),
        "--no-agent-memory",
        "--refresh-agent-memory",
        "--refresh-test-profile",
        "--agent-memory-dir",
        "custom-memory",
    ])

    config = config_from_args(args, FakeRunner())

    assert config.agent_memory is False
    assert config.refresh_agent_memory is True
    assert config.refresh_test_profile is True
    assert config.agent_memory_dir == codex_dir / "custom-memory"


def test_agent_memory_explicit_absolute_dir_is_resolved(tmp_path):
    parser = build_parser()
    memory_dir = tmp_path / "memory-parent" / ".." / "agent-memory"
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--agent-memory-dir",
        str(memory_dir),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.agent_memory_dir == memory_dir.resolve()


def test_agent_memory_default_ignores_active_coder_workdir(tmp_path, monkeypatch):
    parser = build_parser()
    cache_home = tmp_path / "cache"
    codex_dir = tmp_path / "codex"
    monkeypatch.setattr("coding_review_agent_loop.config.sys.platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "codex",
        "--reviewer",
        "claude",
        "--codex-dir",
        str(codex_dir),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.agent_memory_dir == (
        cache_home / "coding-review-agent-loop" / "repos" / "OWNER-REPO" / "memory"
    ).resolve()
    assert codex_dir not in config.agent_memory_dir.parents


def test_auto_created_agent_dir_is_cloned_before_use(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    codex_dir = tmp_path / "tmp-root" / "owner-repo" / "codex" / "repo"
    config = make_config(
        tmp_path,
        claude_dir=tmp_path / "explicit-claude",
        codex_dir=codex_dir,
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
    )
    config.claude_dir.mkdir(parents=True)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert ["git", "init", "-q", "-b", "main"] in [
        cmd for cmd, _cwd in runner.commands
    ]
    assert codex_dir.is_dir()


def test_clean_existing_auto_agent_dir_is_synced(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    assert ["git", "fetch", "origin"] in commands
    assert ["git", "switch", "main"] in commands
    assert ["git", "reset", "--hard", runner.git_head] in commands


@pytest.mark.parametrize("mode", ["issue", "task"])
def test_issue_and_task_loops_use_repo_default_when_base_is_omitted(tmp_path, mode):
    runner = FakeRunner(
        claude_outputs=[
            "Implemented.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={"baseRefName": "develop", "body": "Fixes #56"},
        repo_default_branch="develop",
    )
    config = make_config(
        tmp_path,
        base=None,
        reviewer="codex",
        auto_agent_dirs=("claude", "codex"),
    )

    if mode == "issue":
        assert run_issue_loop(runner, issue_number=56, config=config) == 0
    else:
        assert run_task_loop(runner, task_text="Add /healthz endpoint.", config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    assert ["git", "switch", "develop"] in commands
    assert not any("origin/main" in arg for cmd in commands for arg in cmd)


@pytest.mark.parametrize("mode", ["issue", "task"])
def test_direct_issue_and_task_implementation_turns_run_as_coder_role(tmp_path, monkeypatch, mode):
    """#1077: without role="coder" a sandboxed run gives the turn the read-only grant."""
    runner = FakeRunner(
        claude_outputs=[
            "Implemented.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={"body": "Fixes #56"},
    )
    config = make_config(tmp_path, reviewer="codex", auto_agent_dirs=("claude", "codex"))
    real_run_validated_agent = validated_agent._run_validated_agent
    roles: dict[str, object] = {}

    def recording_run_validated_agent(*args, **kwargs):
        roles.setdefault(kwargs.get("operation_description"), kwargs.get("role"))
        return real_run_validated_agent(*args, **kwargs)

    patch_attr_in(monkeypatch, (validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage, issue_loop), "_run_validated_agent", recording_run_validated_agent)

    if mode == "issue":
        assert run_issue_loop(runner, issue_number=56, config=config) == 0
    else:
        assert run_task_loop(runner, task_text="Add /healthz endpoint.", config=config) == 0

    assert roles[f"{mode} implementation"] == "coder"


def test_unresolved_base_metadata_produces_targeted_override_error(tmp_path):
    runner = FakeRunner(
        pr_payload={"baseRefName": None},
        repo_default_branch=None,
        repo_default_branch_returncode=1,
    )
    config = make_config(tmp_path, base=None, reviewer="codex")

    with pytest.raises(
        AgentLoopError,
        match=r"Unable to resolve a base branch for OWNER/REPO.*--base <branch>",
    ):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:2] == ["git", "switch"] for cmd, _cwd in runner.commands)


def test_dry_run_base_resolution_defaults_to_main_without_github_query(tmp_path):
    runner = FakeRunner()
    config = make_config(tmp_path, base=None, dry_run=True)

    resolved = resolve_base_branch(config, runner)

    assert resolved.base == "main"
    assert not any(cmd[:1] == ["gh"] for cmd, _cwd in runner.commands)


def test_reviewer_checkout_is_refreshed_to_pr_head_before_review(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path, reviewer="codex")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    review_index = command_index(runner.commands, ["codex", "exec"])
    fetch_index = command_index(runner.commands, ["git", "fetch", "origin"], start=0)
    pr_fetch_index = command_index(
        runner.commands,
        ["git", "fetch", "origin", "+pull/77/head:refs/remotes/origin/pr/77"],
    )
    checkout_index = command_index(
        runner.commands,
        ["git", "checkout", "--detach", runner.pr_payload["headRefOid"]],
    )
    head_index = command_index(runner.commands, ["git", "rev-parse", "HEAD"], start=checkout_index)

    assert commands[pr_fetch_index] == ["git", "fetch", "origin", "+pull/77/head:refs/remotes/origin/pr/77"]
    assert fetch_index < pr_fetch_index < checkout_index < head_index < review_index


def test_reviewer_checkout_refreshes_each_round_before_review(tmp_path):
    runner = FakeRunner(
        claude_outputs=["Fixed.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
        codex_outputs=[
            "Please fix it.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    pr_fetches = [
        index
        for index, cmd in enumerate(commands)
        if cmd == ["git", "fetch", "origin", "+pull/77/head:refs/remotes/origin/pr/77"]
    ]
    review_indices = [index for index, cmd in enumerate(commands) if cmd[:2] == ["codex", "exec"]]

    assert len(pr_fetches) == 3
    assert len(review_indices) == 2
    assert pr_fetches[0] < review_indices[0]
    assert pr_fetches[1] < review_indices[1]


def test_dirty_default_reviewer_checkout_is_cleaned_before_review(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        git_status=" M stale.py\n",
    )
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        coder="claude",
        reviewer="codex",
        auto_agent_dirs=("claude", "codex"),
        create_dirs=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    assert run_pr_loop(runner, pr_number=77, config=config, workdirs_ready=True) == 0

    reset_index = command_index(runner.commands, ["git", "reset", "--hard"])
    clean_index = command_index(runner.commands, ["git", "clean", "-fd"])
    review_index = command_index(runner.commands, ["codex", "exec"])

    assert reset_index < clean_index < review_index


def test_dirty_explicit_reviewer_checkout_fails_before_review_invocation(tmp_path):
    runner = FakeRunner(
        codex_outputs=["This should not run.\n<!-- AGENT_STATE: approved -->"],
        git_status=" M stale.py\n",
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", agent_memory=False)

    with pytest.raises(AgentLoopError, match="--codex-dir is dirty"):
        run_pr_loop(runner, pr_number=77, config=config, workdirs_ready=True)

    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


def test_dirty_existing_auto_agent_dir_is_cleaned_before_sync(tmp_path, capsys):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        git_status=" M file.py\n",
    )
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        coder="codex",
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
        quiet=False,
    )
    config.gemini_dir.mkdir(parents=True)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    assert ["git", "reset", "--hard"] in commands
    assert ["git", "clean", "-fd"] in commands
    assert ["git", "reset", "--hard", runner.git_head] in commands
    captured = capsys.readouterr()
    assert f"Cleaning dirty default codex workdir: {codex_dir}" in captured.err


def test_dirty_explicit_agent_dir_fails_clearly(tmp_path):
    runner = FakeRunner(git_status=" M file.py\n")
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        coder="codex",
        reviewer="codex",
        create_dirs=False,
    )
    config.gemini_dir.mkdir(parents=True)

    with pytest.raises(AgentLoopError, match="--codex-dir is dirty"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


@pytest.mark.parametrize("loop_name", ["issue", "task"])
def test_dirty_explicit_coder_dir_fails_before_issue_or_task_coder_invocation(tmp_path, loop_name):
    runner = FakeRunner(
        git_status=" M file.py\n",
        codex_outputs=[
            "Implemented.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
    )
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        coder="codex",
        reviewer="codex",
        create_dirs=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    with pytest.raises(AgentLoopError, match="--codex-dir is dirty"):
        if loop_name == "issue":
            run_issue_loop(runner, issue_number=56, config=config)
        else:
            run_task_loop(runner, task_text="Add /healthz endpoint.", config=config)

    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


def test_explicit_agent_dir_must_match_requested_repo(tmp_path):
    runner = FakeRunner(git_remote="git@github.com:OTHER/REPO.git")
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        coder="codex",
        reviewer="codex",
        create_dirs=False,
    )
    config.gemini_dir.mkdir(parents=True)

    with pytest.raises(AgentLoopError, match="operator-trusted endpoint"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


def test_existing_auto_agent_dir_must_be_git_checkout(tmp_path):
    runner = FakeRunner(git_inside=False)
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    with pytest.raises(AgentLoopError, match="not a git checkout"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


def test_stale_default_workdir_only_logs_is_recreated(tmp_path, capsys):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    (codex_dir / ".agent-loop-logs").mkdir()

    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
        quiet=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    init_cmds = [(cmd, cwd) for cmd, cwd in runner.commands if cmd[:2] == ["git", "init"]]
    assert any(cwd == codex_dir for _, cwd in init_cmds), "Expected fresh init of stale workdir"

    captured = capsys.readouterr()
    assert "Stale default codex workdir detected" in captured.err
    assert "recreating" in captured.err


def test_stale_default_workdir_with_unknown_files_still_fails(tmp_path):
    runner = FakeRunner(git_inside=False)
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    (codex_dir / ".agent-loop-logs").mkdir()
    (codex_dir / "some-user-file.py").write_text("# user work", encoding="utf-8")

    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    with pytest.raises(AgentLoopError, match="not a git checkout"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:3] == ["gh", "repo", "clone"] for cmd, _cwd in runner.commands)


def test_stale_default_workdir_empty_is_recreated(tmp_path, capsys):
    """An empty workdir (no .git, no files) is treated as stale and re-cloned."""
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()  # exists but empty

    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
        quiet=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    init_cmds = [(cmd, cwd) for cmd, cwd in runner.commands if cmd[:2] == ["git", "init"]]
    assert any(cwd == codex_dir for _, cwd in init_cmds), "Expected fresh init of empty stale workdir"

    captured = capsys.readouterr()
    assert "Stale default codex workdir detected" in captured.err
    assert "recreating" in captured.err


def test_stale_default_workdir_git_only_is_recreated(tmp_path, capsys):
    """A workdir with only a .git dir (no working tree) is treated as stale and re-cloned."""
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    (codex_dir / ".git").mkdir()  # .git present, but no source files

    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
        quiet=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    init_cmds = [(cmd, cwd) for cmd, cwd in runner.commands if cmd[:2] == ["git", "init"]]
    assert any(cwd == codex_dir for _, cwd in init_cmds), "Expected fresh init of git-only stale workdir"

    captured = capsys.readouterr()
    assert "Stale default codex workdir detected" in captured.err
    assert "recreating" in captured.err


def test_explicit_dir_not_git_checkout_is_not_recreated(tmp_path):
    runner = FakeRunner(git_inside=False)
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    (codex_dir / ".agent-loop-logs").mkdir()

    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        reviewer="codex",
        create_dirs=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    with pytest.raises(AgentLoopError, match="not a git checkout"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:3] == ["gh", "repo", "clone"] for cmd, _cwd in runner.commands)


def test_existing_auto_agent_dir_must_match_requested_repo(tmp_path):
    runner = FakeRunner(git_remote="git@github.com:OTHER/REPO.git")
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    with pytest.raises(AgentLoopError, match="operator-trusted endpoint"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


def test_agent_workdir_existing_file_fails_clearly(tmp_path):
    runner = FakeRunner()
    claude_path = tmp_path / "claude-file"
    claude_path.write_text("not a dir", encoding="utf-8")
    config = make_config(tmp_path, claude_dir=claude_path, create_dirs=False)

    with pytest.raises(AgentLoopError, match="not a directory"):
        run_pr_loop(runner, pr_number=77, config=config)


def test_gemini_workdir_existing_file_fails_clearly(tmp_path):
    runner = FakeRunner()
    gemini_path = tmp_path / "gemini-file"
    gemini_path.write_text("not a dir", encoding="utf-8")
    config = make_config(
        tmp_path,
        reviewer="gemini",
        gemini_dir=gemini_path,
        create_dirs=False,
    )

    with pytest.raises(AgentLoopError, match="not a directory"):
        run_pr_loop(runner, pr_number=77, config=config)


def test_config_allows_same_coder_and_reviewer(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "codex",
        "--reviewer",
        "codex",
        "--codex-dir",
        str(tmp_path / "codex"),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.coder == "codex"
    assert config.reviewer == ("codex",)


def test_config_allows_coder_in_multiple_reviewers(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "codex",
        "--reviewer",
        "claude",
        "--reviewer",
        "codex",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.coder == "codex"
    assert config.reviewer == ("claude", "codex")


def test_config_accepts_gemini_as_coder_and_reviewer(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "gemini",
        "--reviewer",
        "claude",
        "--reviewer",
        "gemini",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--gemini-dir",
        str(tmp_path / "gemini"),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.coder == "gemini"
    assert config.reviewer == ("claude", "gemini")
    assert config.gemini_dir == tmp_path / "gemini"


@pytest.mark.parametrize(
    ("coder", "reviewer"),
    [
        ("agy", "codex"),
        ("codex", "agy"),
        ("antigravity", "codex"),
    ],
)
def test_config_normalizes_antigravity_agent_names(tmp_path, coder, reviewer):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--coder",
        coder,
        "--reviewer",
        reviewer,
        "--codex-dir",
        str(tmp_path / "codex"),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.coder == ("antigravity" if coder == "agy" else coder)
    assert config.reviewer == (
        "antigravity" if reviewer == "agy" else reviewer,
    )


def test_config_rejects_duplicate_reviewers(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--reviewer",
        "codex",
        "--reviewer",
        "codex",
        "--codex-dir",
        str(tmp_path / "codex"),
    ])

    with pytest.raises(AgentLoopError, match="same agent more than once"):
        config_from_args(args, FakeRunner())


@pytest.mark.parametrize("seat_flags,expected", [
    (["--reviewer-seat", "agy=agy", "--seat-model", "agy=Model"], "reserved"),
    (["--reviewer-seat", "Antigravity=agy", "--seat-model", "Antigravity=Model"], "reserved"),
    (["--reviewer-seat", "OpenAI Codex=codex", "--seat-model", "OpenAI Codex=Model"], "reserved"),
    (["--reviewer-seat", "Orchestrator=codex", "--seat-model", "Orchestrator=m"], "reserved"),
    (["--reviewer-seat", "a=agy", "--reviewer-seat", "a=agy", "--seat-model", "a=Model"], "Duplicate"),
    (["--reviewer-seat", "a=agy", "--reviewer-seat", "A=agy", "--seat-model", "a=Model"], "lowercase ASCII"),
    (["--reviewer-seat", "a=agy"], "requires --seat-model"),
    (["--reviewer-seat", "a=codex", "--seat-model", "a=one", "--seat-model", "a=two"], "fallback"),
    (["--reviewer-seat", "a=agy", "--seat-model", "a=one", "--seat-effort", "a=high"], "unsupported"),
    (["--reviewer-seat", "a=agy", "--seat-model", "a=Gemini 3.8 Flash (High)",
      "--reviewer-seat", "b=agy", "--seat-model", "b=gemini-3.8-flash-high"], "overlap"),
    (["--reviewer-seat", "a=agy", "--seat-model", "a=Claude Opus 5.5 (Medium)",
      "--reviewer-seat", "b=agy", "--seat-model", "b=claude-opus-5-5"], "overlap"),
    (["--reviewer-seat", "a=agy", "--seat-model", "a=Claude Opus 5.5",
      "--reviewer-seat", "b=agy", "--seat-model", "b=claude-opus-5-5-medium"], "overlap"),
    (["--reviewer-seat", "a=agy", "--seat-model", "a=Model A",
      "--antigravity-models", "Model B"], "not named antigravity seats"),
    (["--reviewer-seat", "a=codex", "--seat-model", "a=Model A",
      "--reviewer-codex-model", "Model B"], "requires an explicit legacy"),
])
def test_named_seat_validation_precedes_dispatch(tmp_path, monkeypatch, capsys, seat_flags, expected):
    monkeypatch.setattr(cli_module, "run_pr_loop", lambda *a, **k: pytest.fail("agent dispatched"))
    code = cli_module.main(["pr", "77", "--repo", "OWNER/REPO", *seat_flags])
    assert code == 1
    assert expected in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_valid_named_seats_reach_pr_loop_with_independent_identity(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(cli_module, "run_pr_loop", lambda *a, **k: seen.append(k["config"]) or 0)
    code = cli_module.main([
        "pr", "77", "--repo", "OWNER/REPO", "--dry-run", "--reviewer-seat", "gemini-seat=agy",
        "--seat-model", "gemini-seat=Gemini 3.8 Flash (High)",
        "--reviewer-seat", "claude-seat=agy", "--seat-model", "claude-seat=Claude Opus 5.5 (Medium)",
    ])
    assert code == 0
    assert len(seen) == 1
    assert tuple(map(str, seen[0].reviewer)) == ("gemini-seat", "claude-seat")
    assert seen[0].reviewer[0].backend == seen[0].reviewer[1].backend == "antigravity"
    assert seen[0].reviewer[0].workdir != seen[0].reviewer[1].workdir


@pytest.mark.parametrize("entry", [
    ["pr", "77"], ["issue", "77"], ["task", "Fix the bug"],
    ["managed-pr", "--head", "feature", "--title", "Fix the bug"],
])
def test_named_seats_configure_each_pr_entry_path(entry):
    args = build_parser().parse_args([
        *entry, "--repo", "OWNER/REPO", "--dry-run",
        "--reviewer-seat", "flash=agy", "--seat-model", "flash=Model A",
        "--reviewer-seat", "opus=agy", "--seat-model", "opus=Model B",
    ])
    config = config_from_args(args, Runner(dry_run=True))
    assert tuple(map(str, config.reviewer)) == ("flash", "opus")


def test_named_primary_cli_selects_seat_identity():
    args = build_parser().parse_args([
        "pr", "77", "--repo", "OWNER/REPO", "--dry-run",
        "--reviewer-seat", "flash=agy", "--seat-model", "flash=Model A",
        "--reviewer-seat", "opus=agy", "--seat-model", "opus=Model B",
        "--pr-review-policy", "primary-then-panel", "--primary-reviewer-seat", "opus",
    ])
    config = config_from_args(args, Runner(dry_run=True))
    assert config.primary_reviewer is config.reviewer[1]


def test_named_primary_reaches_pr_loop_through_main(monkeypatch):
    seen = []
    monkeypatch.setattr(cli_module, "run_pr_loop", lambda *a, **k: seen.append(k["config"]) or 0)
    code = cli_module.main([
        "pr", "77", "--repo", "OWNER/REPO", "--dry-run",
        "--reviewer-seat", "flash=agy", "--seat-model", "flash=Model A",
        "--reviewer-seat", "opus=agy", "--seat-model", "opus=Model B",
        "--pr-review-policy", "primary-then-panel", "--primary-reviewer-seat", "opus",
    ])
    assert code == 0
    assert len(seen) == 1
    assert seen[0].primary_reviewer is seen[0].reviewer[1]


def test_default_named_seat_checkout_uses_claimed_run_worktree(monkeypatch, tmp_path):
    from coding_review_agent_loop.config import run_worktree_store, ensure_temp_checkout
    from coding_review_agent_loop import run_worktrees
    from coding_review_agent_loop.workdir_claims import workdir_claim_scope

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    args = build_parser().parse_args([
        "pr", "77", "--repo", "OWNER/REPO", "--dry-run",
        "--reviewer-seat", "flash=agy", "--seat-model", "flash=Model A",
    ])
    config = config_from_args(args, Runner(dry_run=True))
    seat = config.reviewer[0]
    assert run_worktree_store(config, seat.workdir) == default_agent_workdir(
        "OWNER/REPO", "flash"
    ).resolve()
    calls = []
    monkeypatch.setattr(run_worktrees, "prepare_store", lambda *a, **k: "head")
    monkeypatch.setattr(run_worktrees, "prune_dead_worktrees", lambda *a, **k: None)
    monkeypatch.setattr(run_worktrees, "store_lock", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(run_worktrees, "add_run_worktree", lambda *a, **k: calls.append("add"))
    monkeypatch.setattr(run_worktrees, "identify_owned_worktree", lambda *a, **k: True)
    monkeypatch.setattr(run_worktrees, "remove_run_worktree", lambda *a, **k: calls.append("remove"))
    monkeypatch.setattr("coding_review_agent_loop.config.verify_before_sync", lambda *a, **k: None)
    monkeypatch.setattr("coding_review_agent_loop.config._sync_base_branch", lambda *a, **k: None)
    with workdir_claim_scope():
        ensure_temp_checkout(seat.workdir, agent=seat, config=config, runner=FakeRunner())
    assert calls == ["add", "remove"]


def test_discuss_rejects_named_seats(capsys):
    code = cli_module.main([
        "discuss", "77", "--repo", "OWNER/REPO", "--reviewer-seat", "model-a=claude",
        "--seat-model", "model-a=Model A",
    ])
    assert code == 1
    assert "discuss does not support named" in capsys.readouterr().err


def test_named_seat_rejects_legacy_implicit_model_and_coder_workdir(tmp_path, capsys):
    common = ["pr", "77", "--repo", "OWNER/REPO", "--reviewer-seat", "model-a=agy",
              "--seat-model", "model-a=Gemini 3.8 Flash (High)"]
    assert cli_module.main([*common, "--reviewer", "agy"]) == 1
    assert "overlaps the legacy" in capsys.readouterr().err
    checkout = tmp_path / "checkout"
    assert cli_module.main([
        "pr", "77", "--repo", "OWNER/REPO", "--coder", "claude",
        "--claude-dir", str(checkout), "--reviewer-seat", "model-a=codex",
        "--seat-model", "model-a=Model A", "--seat-dir", f"model-a={checkout}",
    ]) == 1
    assert "workdir collides with coder" in capsys.readouterr().err
    assert not checkout.exists()


@pytest.mark.parametrize("backend", ["codex", "claude", "gemini"])
def test_named_seat_rejects_unbound_legacy_model_before_dispatch(
    tmp_path, monkeypatch, capsys, backend,
):
    monkeypatch.setattr(cli_module, "run_pr_loop", lambda *a, **k: pytest.fail("agent dispatched"))
    checkout = tmp_path / "seat-checkout"
    assert cli_module.main([
        "pr", "77", "--repo", "OWNER/REPO", "--reviewer", backend,
        "--reviewer-seat", f"alternate={backend}", "--seat-model", "alternate=Model A",
        "--seat-dir", f"alternate={checkout}",
    ]) == 1
    assert f"Legacy {backend} reviewer has an implicit model" in capsys.readouterr().err
    assert not checkout.exists()


@pytest.mark.parametrize("backend,effort_flag", [
    ("codex", "--codex-reasoning-effort"),
    ("claude", "--claude-effort"),
])
def test_named_seat_rejects_unowned_backend_effort_before_dispatch(
    tmp_path, monkeypatch, capsys, backend, effort_flag,
):
    monkeypatch.setattr(cli_module, "run_pr_loop", lambda *a, **k: pytest.fail("agent dispatched"))
    checkout = tmp_path / "seat-checkout"
    coder = "claude" if backend == "codex" else "codex"
    assert cli_module.main([
        "pr", "77", "--repo", "OWNER/REPO", "--coder", coder,
        "--reviewer-seat", f"alternate={backend}", "--seat-model", "alternate=Model A",
        "--seat-dir", f"alternate={checkout}", effort_flag, "high",
    ]) == 1
    error = capsys.readouterr().err
    assert effort_flag in error
    assert "configures a coder or legacy default reviewer" in error
    assert not checkout.exists()


def test_named_plan_seat_preserves_implementation_coder_model_and_effort(
    tmp_path, monkeypatch,
):
    captured = {}
    monkeypatch.setattr(cli_module, "run_issue_loop", lambda *a, **k: captured.update(k) or 0)
    checkout = tmp_path / "seat-checkout"
    assert cli_module.main([
        "issue", "77", "--repo", "OWNER/REPO", "--plan-first", "--coder", "codex",
        "--implementation-coder", "claude", "--claude-model", "claude-fable-5",
        "--claude-effort", "high", "--reviewer-seat", "alternate=claude",
        "--seat-model", "alternate=claude-sonnet-5", "--seat-dir", f"alternate={checkout}",
    ]) == 0
    assert captured["plan_first"] is True
    assert captured["config"].implementation_coder == "claude"
    assert len(captured["config"].reviewer_seats) == 1


def test_named_plan_primary_cli_selects_seat_before_dispatch(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(cli_module, "run_issue_loop", lambda *a, **k: captured.update(k) or 0)
    assert cli_module.main([
        "issue", "77", "--repo", "OWNER/REPO", "--plan-first",
        "--reviewer-seat", "first=codex", "--seat-model", "first=Model A",
        "--reviewer-seat", "second=claude", "--seat-model", "second=Model B",
        "--plan-review-policy", "primary-then-panel",
        "--primary-plan-reviewer-seat", "second",
        "--seat-dir", f"first={tmp_path / 'first'}",
        "--seat-dir", f"second={tmp_path / 'second'}",
    ]) == 0
    config = captured["config"]
    assert tuple(str(seat) for seat in config.reviewer) == ("first", "second")
    assert str(config.primary_plan_reviewer) == "second"
    assert config.primary_plan_reviewer.backend == "claude"


def test_named_seat_rejects_legacy_explicit_reviewer_workdir(tmp_path, capsys):
    checkout = tmp_path / "reviewer-checkout"
    assert cli_module.main([
        "pr", "77", "--repo", "OWNER/REPO", "--coder", "claude",
        "--codex-dir", str(checkout), "--reviewer", "codex",
        "--reviewer-seat", "other=claude", "--seat-model", "other=Model A",
        "--seat-dir", f"other={checkout}",
    ]) == 1
    assert "workdir collides with legacy codex reviewer" in capsys.readouterr().err
    assert not checkout.exists()


@pytest.mark.parametrize("legacy_reviewers", [
    ("agy", "agy"),
    ("agy", "antigravity"),
])
@pytest.mark.parametrize("primary", [False, True])
def test_named_seat_rejects_duplicate_legacy_reviewers_before_phase_gate(
    tmp_path, monkeypatch, capsys, legacy_reviewers, primary,
):
    monkeypatch.setattr(cli_module, "run_pr_loop", lambda *a, **k: pytest.fail("agent dispatched"))
    checkout = tmp_path / "seat-checkout"
    flags = ["--pr-review-policy", "primary-then-panel", "--primary-reviewer-seat", "model-a"] if primary else []
    assert cli_module.main([
        "pr", "77", "--repo", "OWNER/REPO", "--reviewer-seat", "model-a=codex",
        "--seat-model", "model-a=Model A", "--seat-dir", f"model-a={checkout}",
        "--reviewer", legacy_reviewers[0],
        "--reviewer", legacy_reviewers[1], *flags,
    ]) == 1
    assert "--reviewer cannot include the same agent more than once" in capsys.readouterr().err
    assert not checkout.exists()


def test_named_seat_rejects_implementation_coder_workdir_before_phase_gate(tmp_path, monkeypatch, capsys):
    checkout = tmp_path / "implementation-checkout"
    monkeypatch.setattr(cli_module, "run_issue_loop", lambda *a, **k: pytest.fail("agent dispatched"))
    assert cli_module.main([
        "issue", "77", "--repo", "OWNER/REPO", "--plan-first", "--coder", "codex",
        "--implementation-coder", "claude", "--claude-dir", str(checkout),
        "--reviewer-seat", "model-a=agy", "--seat-model", "model-a=Model A",
        "--seat-dir", f"model-a={checkout}",
    ]) == 1
    assert "workdir collides with implementation coder" in capsys.readouterr().err
    assert not checkout.exists()


@pytest.mark.parametrize("policy,primary,expected", [
    ("--pr-review-policy", "--primary-reviewer-seat", "requires at least one secondary reviewer"),
    ("--plan-review-policy", "--primary-plan-reviewer-seat", "requires at least one secondary reviewer"),
])
def test_named_primary_requires_secondary_before_phase_gate(capsys, policy, primary, expected):
    assert cli_module.main([
        "pr", "77", "--repo", "OWNER/REPO", "--reviewer-seat", "only=codex",
        "--seat-model", "only=Model A", policy, "primary-then-panel", primary, "only",
    ]) == 1
    assert expected in capsys.readouterr().err


def test_config_rejects_alias_and_canonical_duplicate_reviewers(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--reviewer",
        "agy",
        "--reviewer",
        "antigravity",
        "--codex-dir",
        str(tmp_path / "codex"),
    ])

    with pytest.raises(AgentLoopError, match="same agent more than once"):
        config_from_args(args, FakeRunner())


@pytest.mark.parametrize("max_rounds", ["0", "-1"])
def test_config_rejects_non_positive_max_rounds(tmp_path, max_rounds):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--max-rounds",
        max_rounds,
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
    ])

    with pytest.raises(AgentLoopError, match="--max-rounds must be greater than zero"):
        config_from_args(args, FakeRunner())


def test_config_defaults_ci_queued_grace_seconds(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
    ])
    config = config_from_args(args, FakeRunner())

    assert config.ci_queued_grace_seconds == 1200


def test_config_parses_custom_ci_queued_grace_seconds(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--ci-queued-grace-seconds",
        "600",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
    ])
    config = config_from_args(args, FakeRunner())

    assert config.ci_queued_grace_seconds == 600


@pytest.mark.parametrize("ci_queued_grace_seconds", ["0", "-1"])
def test_config_rejects_non_positive_ci_queued_grace_seconds(tmp_path, ci_queued_grace_seconds):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--ci-queued-grace-seconds",
        ci_queued_grace_seconds,
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
    ])

    with pytest.raises(AgentLoopError, match="--ci-queued-grace-seconds must be greater than zero"):
        config_from_args(args, FakeRunner())


def test_config_defaults_do_not_bypass_agent_permissions(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
    ])
    config = config_from_args(args, FakeRunner())

    assert config.claude_args == ()
    assert config.codex_args == ()
    assert config.gemini_args == ()


def test_config_can_opt_into_dangerous_agent_permissions(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
        "--dangerous-agent-permissions",
    ])
    config = config_from_args(args, FakeRunner())

    assert config.claude_args == ("--dangerously-skip-permissions",)
    assert config.codex_args == ("--dangerously-bypass-approvals-and-sandbox",)
    assert config.gemini_args == ("--yolo", "--skip-trust")


def test_explicit_agent_args_replace_dangerous_profile(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
        "--dangerous-agent-permissions",
        "--claude-arg=--permission-mode",
        "--claude-arg=acceptEdits",
        "--codex-arg=--sandbox",
        "--codex-arg=workspace-write",
        "--gemini-arg=--approval-mode",
        "--gemini-arg=auto_edit",
    ])
    config = config_from_args(args, FakeRunner())

    assert config.claude_args == ("--permission-mode", "acceptEdits")
    assert config.codex_args == ("--sandbox", "workspace-write")
    assert config.gemini_args == ("--approval-mode", "auto_edit")


def _permission_args(tmp_path, *extra):
    return build_parser().parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
        *extra,
    ])


SANDBOXED_HELPERS = (
    "--repair-backend", "claude", "--repair-model", "claude-haiku-4-5",
    "--semantic-followup-backend", "claude",
)


def test_agent_permissions_sandboxed_parses_with_empty_static_args(tmp_path):
    config = config_from_args(
        _permission_args(tmp_path, "--agent-permissions", "sandboxed", *SANDBOXED_HELPERS),
        FakeRunner(),
    )
    assert config.agent_permissions == "sandboxed"
    assert config.claude_args == config.codex_args == config.gemini_args == config.antigravity_args == ()


def test_agent_permissions_dangerous_is_byte_identical_to_legacy_flag(tmp_path):
    legacy = config_from_args(_permission_args(tmp_path, "--dangerous-agent-permissions"), FakeRunner())
    explicit = config_from_args(_permission_args(tmp_path, "--agent-permissions", "dangerous"), FakeRunner())
    both = config_from_args(
        _permission_args(tmp_path, "--dangerous-agent-permissions", "--agent-permissions", "dangerous"),
        FakeRunner(),
    )
    default = config_from_args(_permission_args(tmp_path, "--agent-permissions", "default"), FakeRunner())
    for config in (legacy, explicit, both):
        assert config.agent_permissions == "dangerous"
        assert config.claude_args == ("--dangerously-skip-permissions",)
        assert config.codex_args == ("--dangerously-bypass-approvals-and-sandbox",)
        assert config.gemini_args == ("--yolo", "--skip-trust")
    assert default.agent_permissions == "default"
    assert default.claude_args == default.codex_args == default.gemini_args == ()


@pytest.mark.parametrize("mode", ["sandboxed", "default"])
def test_dangerous_flag_conflicts_with_other_agent_permissions(tmp_path, mode):
    args = _permission_args(tmp_path, "--dangerous-agent-permissions", "--agent-permissions", mode, *SANDBOXED_HELPERS)
    with pytest.raises(AgentLoopError, match="conflicts with --agent-permissions"):
        config_from_args(args, FakeRunner())


@pytest.mark.parametrize(
    "passthrough",
    [
        "--claude-arg=--permission-mode",
        "--claude-arg=--dangerously-skip-permissions",
        "--codex-arg=-c",
        "--codex-arg=--config=sandbox_mode=danger-full-access",
        "--gemini-arg=--yolo",
        "--antigravity-arg=--x",
    ],
)
def test_sandboxed_rejects_every_passthrough_agent_arg(tmp_path, passthrough):
    args = _permission_args(tmp_path, "--agent-permissions", "sandboxed", *SANDBOXED_HELPERS, passthrough)
    option = passthrough.split("=", 1)[0]
    with pytest.raises(AgentLoopError, match=f"{option} .* is not allowed with --agent-permissions sandboxed"):
        config_from_args(args, FakeRunner())


def test_sandboxed_names_default_repair_backend_before_command_preflight(tmp_path):
    args = _permission_args(tmp_path, "--agent-permissions", "sandboxed", "--semantic-followup-backend", "claude")
    with pytest.raises(AgentLoopError, match="--repair-backend claude\\|codex --repair-model MODEL"):
        config_from_args(args, FakeRunner())
    args = _permission_args(
        tmp_path, "--agent-permissions", "sandboxed", "--repair-backend", "codex", "--repair-model", "m"
    )
    with pytest.raises(AgentLoopError, match="--no-semantic-followup-dedupe"):
        config_from_args(args, FakeRunner())


def test_sandboxed_codex_pr_coder_fails_before_any_agent_runs(tmp_path, monkeypatch, capsys):
    from coding_review_agent_loop import cli as cli_module

    monkeypatch.setattr(
        cli_module, "run_pr_loop", lambda *a, **k: pytest.fail("no agent may run")
    )
    code = cli_module.main([
        "pr", "77", "--repo", "OWNER/REPO", "--dry-run",
        "--claude-dir", str(tmp_path / "claude"), "--codex-dir", str(tmp_path / "codex"),
        "--subprocess-log-dir", str(tmp_path / "logs"),
        "--agent-permissions", "sandboxed", "--coder", "codex", "--reviewer", "claude",
        *SANDBOXED_HELPERS,
    ])
    assert code == 1
    assert "cannot run a committing Codex coder" in capsys.readouterr().err


def test_resume_plan_round_prefers_latest_metadata_ledger_for_same_plan_replay():
    current_plan = "Revised plan.\n- Add the active-ledger replay test.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    subject = _plan_subject(current_plan)
    stale_item = UnresolvedReviewItem(
        item_id="item-3",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Stale plan replay item.",
        status="same-plan",
        source_status="same-plan",
    )
    active_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Active plan replay item.",
        status="same-plan",
        source_status="same-plan",
    )
    stale_coder_comment = _attach_round_metadata(
        current_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=subject,
            prior_items=(stale_item,),
            canonical_plan=current_plan,
        ),
    )
    stale_reviewer_comment = _attach_round_metadata(
        "Still needs work."
        + prior_plan_item_dispositions("[item-3] same-plan: stale replay")
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent="Codex",
            round_number=2,
            subject=subject,
            prior_items=(stale_item,),
            dispositions=(
                parse_plan_item_dispositions(
                    prior_plan_item_dispositions("[item-3] same-plan: stale replay"),
                    reviewer="OpenAI Codex",
                )[0],
            ),
            state="blocking",
        ),
    )
    active_coder_comment = _attach_round_metadata(
        current_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=subject,
            prior_items=(active_item,),
            canonical_plan=current_plan,
        ),
    )
    active_reviewer_comment = _attach_round_metadata(
        "Plan looks sound."
        + prior_plan_item_dispositions("[item-1] resolved")
        + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- Google Gemini",
        PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent="Gemini",
            round_number=2,
            subject=subject,
            prior_items=(active_item,),
            dispositions=(
                parse_plan_item_dispositions(
                    prior_plan_item_dispositions("[item-1] resolved"),
                    reviewer="Google Gemini",
                )[0],
            ),
            state="approved",
        ),
    )

    resumed = _resume_plan_round(
        [
            IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=stale_coder_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:01:00Z", body=stale_reviewer_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:02:00Z", body=active_coder_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:03:00Z", body=active_reviewer_comment),
        ],
        configured_reviewers=("codex", "gemini"),
    )

    assert resumed is not None
    current_plan_text, resumed_state = resumed
    assert current_plan_text == current_plan
    assert [item.item_id for item in resumed_state.prior_items] == ["item-1"]
    assert resumed_state.next_unresolved_item_number == 4
    assert [record.metadata.agent for record in resumed_state.completed_reviews] == ["Gemini"]


def test_resume_plan_round_prefers_canonical_plan_metadata():
    public_body = (
        "Revised plan summary.\n\n### Plan steps\n1. Public body copy.\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    canonical_plan = (
        "Revised plan summary.\n\n### Prior plan review item dispositions\n- None.\n\n"
        "### Plan steps\n1. Canonical copy."
    )
    coder_comment = _attach_round_metadata(
        public_body,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=_plan_subject(canonical_plan),
            prior_items=(),
            canonical_plan=canonical_plan,
        ),
    )

    resumed = _resume_plan_round(
        [IssueComment(author="bot", created_at="2026-05-20T09:00:00Z", body=coder_comment)],
        configured_reviewers=("codex", "gemini"),
    )

    assert resumed is not None
    current_plan, state = resumed
    assert current_plan == canonical_plan
    assert state.coder_output == canonical_plan


def test_resume_plan_round_prefers_structured_plan_revision_metadata_for_coder_output():
    public_body = (
        "## Revised plan\n\nRevised plan summary.\n\n### Plan steps\n1. Public body copy.\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    raw_structured_revision = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_revision",
                "state": "blocking",
                "summary": "Revised plan summary.",
                "prior_plan_item_dispositions": [],
                "plan_steps": ["Canonical copy."],
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    parsed = validate_structured_plan_revision(raw_structured_revision)
    assert parsed is not None
    canonical_plan = render_canonical_plan_revision(parsed, ())
    coder_comment = _attach_round_metadata(
        public_body,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=_plan_subject(canonical_plan),
            prior_items=(),
            canonical_plan=canonical_plan,
            raw_structured_coder_response=raw_structured_revision,
        ),
    )

    resumed = _resume_plan_round(
        [IssueComment(author="bot", created_at="2026-05-20T09:00:00Z", body=coder_comment)],
        configured_reviewers=("codex", "gemini"),
    )

    assert resumed is not None
    current_plan, state = resumed
    assert current_plan == canonical_plan
    assert state.coder_output == raw_structured_revision
    assert validate_structured_plan_revision(state.coder_output) is not None
    assert '"kind": "plan_revision"' not in _strip_round_metadata(coder_comment)


def test_resume_plan_round_falls_back_to_raw_body_for_markdown_plan():
    plan = "Revised plan.\n- Add state reconstruction.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    coder_comment = _attach_round_metadata(
        plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=_plan_subject(plan),
            prior_items=(),
        ),
    )

    resumed = _resume_plan_round(
        [IssueComment(author="bot", created_at="2026-05-20T09:00:00Z", body=coder_comment)],
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    current_plan, state = resumed
    assert current_plan == plan
    assert state.coder_output == plan


def test_plan_subject_ignores_trailing_whitespace_added_by_metadata_round_trip():
    plan = "Revised plan.\n- Add state reconstruction.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"

    attached = _attach_round_metadata(
        f"{plan}\n",
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=_plan_subject(f"{plan}\n"),
            prior_items=(),
        ),
    )

    assert _plan_subject(f"{plan}\n") == _plan_subject(_strip_round_metadata(attached))


def test_round_metadata_round_trip_preserves_canonical_plan():
    metadata = PostedRoundMetadata(
        flow="plan",
        role="coder",
        agent="Claude",
        round_number=2,
        subject="abc",
        canonical_plan="Summary\n\n### Plan steps\n1. Canonical step.",
        approved_plan_hash="0123456789abcdef",
        approved_plan_subject="f" * 64,
    )

    decoded = _decode_round_metadata(_encode_round_metadata(metadata))
    assert decoded.canonical_plan == metadata.canonical_plan
    assert decoded.approved_plan_hash == metadata.approved_plan_hash
    assert decoded.approved_plan_subject == metadata.approved_plan_subject


def test_round_metadata_round_trip_preserves_split_proposals():
    metadata = PostedRoundMetadata(
        flow="discuss",
        role="summary",
        agent="Orchestrator",
        round_number=1,
        subject="abc",
        is_final=True,
        split_proposals=("Auth flow", "Billing flow"),
    )

    decoded = _decode_round_metadata(_encode_round_metadata(metadata))

    assert decoded.split_proposals == metadata.split_proposals


def test_decode_old_round_metadata_defaults_split_proposals_to_empty():
    payload = {
        "flow": "discuss",
        "role": "summary",
        "agent": "Orchestrator",
        "round_number": 1,
        "subject": "abc",
        "prior_items": [],
        "dispositions": [],
        "new_items": [],
        "state": None,
        "canonical_plan": None,
        "raw_structured_coder_response": None,
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).decode("ascii")

    assert _decode_round_metadata(encoded).split_proposals == ()


def test_round_metadata_round_trip_preserves_compact_prior_summaries():
    metadata = PostedRoundMetadata(
        flow="plan",
        role="coder",
        agent="Claude",
        round_number=2,
        subject="abc",
        compact_prior_summaries=("[item-1] resolved: full prior text",),
    )

    decoded = _decode_round_metadata(_encode_round_metadata(metadata))

    assert decoded.compact_prior_summaries == metadata.compact_prior_summaries


def test_decode_old_round_metadata_defaults_compact_prior_summaries_to_empty():
    payload = {
        "flow": "plan",
        "role": "coder",
        "agent": "Claude",
        "round_number": 2,
        "subject": "abc",
        "prior_items": [],
        "dispositions": [],
        "new_items": [],
        "state": None,
        "canonical_plan": None,
        "raw_structured_coder_response": None,
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).decode("ascii")

    assert _decode_round_metadata(encoded).compact_prior_summaries == ()


def test_resume_plan_round_restores_compact_prior_summaries_across_subject_change():
    old_plan = "Old plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    new_plan = "New plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    old_subject = _plan_subject(old_plan)
    new_subject = _plan_subject(new_plan)
    old_summary = "[item-1] resolved: old-subject resolved summary"
    old_coder_comment = _attach_round_metadata(
        old_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=old_subject,
            compact_prior_summaries=(old_summary,),
        ),
    )
    new_coder_comment = _attach_round_metadata(
        new_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=3,
            subject=new_subject,
            compact_prior_summaries=(old_summary,),
        ),
    )

    resumed = _resume_plan_round(
        [
            IssueComment(author="bot", created_at="2026-05-20T09:00:00Z", body=old_coder_comment),
            IssueComment(author="bot", created_at="2026-05-20T09:10:00Z", body=new_coder_comment),
        ],
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    _current_plan, state = resumed
    assert state.compact_prior_summaries == (old_summary,)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"flow": "plan"},
        {
            "flow": "plan",
            "role": "coder",
            "agent": "Claude",
            "round_number": "not-an-int",
            "subject": "abc",
        },
    ],
)
def test_decode_round_metadata_rejects_missing_or_invalid_required_fields(payload):
    encoded = json.dumps(payload).encode("utf-8")

    with pytest.raises(AgentLoopError, match="Invalid AGENT_LOOP_META payload"):
        _decode_round_metadata(encoded=base64.urlsafe_b64encode(encoded).decode("ascii"))


def test_is_clarification_request_detects_marker():
    assert is_clarification_request("need more info\n<!-- AGENT_CLARIFY -->")
    assert is_clarification_request("<!-- agent_clarify -->")
    assert not is_clarification_request("done\n<!-- AGENT_STATE: blocking -->")


def test_is_clarification_request_state_marker_after_clarify_takes_precedence():
    # AGENT_PLAN_STATE after inline AGENT_CLARIFY example: issue #216 / #278 shape.
    # Inline (non-standalone) AGENT_CLARIFY never triggers, regardless of state markers.
    plan_with_embedded_clarify = (
        "Here is my plan.\n\n"
        "If I needed clarification I would emit <!-- AGENT_CLARIFY --> as a marker.\n\n"
        "But I have enough information, so here is the full plan:\n\n"
        "1. Do step one\n2. Do step two\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    assert not is_clarification_request(plan_with_embedded_clarify)

    # Inline AGENT_CLARIFY without any state marker: still not clarification.
    inline_only = (
        "The protocol supports <!-- AGENT_CLARIFY --> for clarification requests.\n\n"
        "Here is my fix."
    )
    assert not is_clarification_request(inline_only)

    # AGENT_STATE after inline AGENT_CLARIFY example: PR/coder blocking response.
    pr_response_with_embedded_clarify = (
        "The protocol supports <!-- AGENT_CLARIFY --> for clarification requests.\n\n"
        "Here is my fix.\n\n"
        "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    assert not is_clarification_request(pr_response_with_embedded_clarify)

    # AGENT_PR after inline AGENT_CLARIFY example: coder PR-creation response.
    pr_created_with_embedded_clarify = (
        "Use <!-- AGENT_CLARIFY --> if you need more info.\n\n"
        "Implemented the fix.\n\n"
        "<!-- AGENT_PR: 42 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    assert not is_clarification_request(pr_created_with_embedded_clarify)

    # PR URL after inline AGENT_CLARIFY: treated as final state marker.
    pr_url_with_embedded_clarify = (
        "Use <!-- AGENT_CLARIFY --> for questions.\n\n"
        "See https://github.com/OWNER/REPO/pull/99 for the PR."
    )
    assert not is_clarification_request(pr_url_with_embedded_clarify)

    # Real clarification request: standalone AGENT_CLARIFY is the final marker.
    real_clarify = "Which endpoint should I use?\n<!-- AGENT_CLARIFY -->\n-- Anthropic Claude"
    assert is_clarification_request(real_clarify)

    # Standalone AGENT_CLARIFY on its own line, after a state marker in prose.
    clarify_after_state = (
        "Round 1 ended with <!-- AGENT_STATE: blocking -->, but I still need more info.\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(clarify_after_state)

    # Standalone AGENT_CLARIFY on its own line, appearing after AGENT_PLAN_STATE in prose.
    plan_state_in_prose_clarify_last = (
        "The previous round used <!-- AGENT_PLAN_STATE: blocking --> to signal issues,\n"
        "but now I have a question:\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(plan_state_in_prose_clarify_last)


def test_is_clarification_request_standalone_marker_positional_semantics():
    # Standalone AGENT_PLAN_STATE footer appearing BEFORE a standalone AGENT_CLARIFY
    # does NOT suppress it — AGENT_CLARIFY is the final marker and wins.
    plan_footer_then_clarify_appendix = (
        "Plan content.\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n"
        "-- Anthropic Claude\n\n"
        "Appendix:\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(plan_footer_then_clarify_appendix)

    # Standalone AGENT_STATE appearing BEFORE AGENT_CLARIFY also does not suppress.
    state_then_clarify = (
        "<!-- AGENT_STATE: blocking -->\n\n"
        "Note:\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(state_then_clarify)

    # Standalone AGENT_STATE appearing AFTER AGENT_CLARIFY does suppress it.
    clarify_then_state = (
        "<!-- AGENT_CLARIFY -->\n"
        "<!-- AGENT_STATE: blocking -->"
    )
    assert not is_clarification_request(clarify_then_state)

    # Inline (non-standalone) AGENT_STATE in prose does NOT suppress AGENT_CLARIFY —
    # it may be a quoted reference to a previous round's state.
    inline_state_then_clarify = (
        "Round 1 ended with <!-- AGENT_STATE: blocking -->, but I still need more info.\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(inline_state_then_clarify)

    # Inline AGENT_PLAN_STATE in prose also does not suppress.
    inline_plan_state_then_clarify = (
        "The previous round used <!-- AGENT_PLAN_STATE: blocking --> to signal issues,\n"
        "but now I have a question:\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(inline_plan_state_then_clarify)


def test_is_clarification_request_pr_marker_takes_precedence():
    # AGENT_PR: N standalone marker appearing AFTER AGENT_CLARIFY suppresses it.
    pr_after_clarify = (
        "<!-- AGENT_CLARIFY -->\n"
        "Actually I have enough info.\n"
        "<!-- AGENT_PR: 55 -->"
    )
    assert not is_clarification_request(pr_after_clarify)

    # AGENT_PR: N standalone marker appearing BEFORE AGENT_CLARIFY does NOT suppress —
    # AGENT_CLARIFY is the final marker and wins.
    pr_before_clarify = (
        "<!-- AGENT_PR: 55 -->\n"
        "<!-- AGENT_STATE: blocking -->\n\n"
        "Note:\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(pr_before_clarify)


def test_is_clarification_request_ignores_fenced_code_block_examples():
    # AGENT_CLARIFY on its own line inside a backtick fence: not clarification.
    fenced_no_state = (
        "Here's how the marker looks:\n\n"
        "```\n"
        "<!-- AGENT_CLARIFY -->\n"
        "```\n\n"
        "That's all."
    )
    assert not is_clarification_request(fenced_no_state)

    # Fenced example with AGENT_PLAN_STATE after the block: still not clarification.
    fenced_with_plan_state = (
        "Protocol example:\n\n"
        "```\n"
        "<!-- AGENT_CLARIFY -->\n"
        "```\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    assert not is_clarification_request(fenced_with_plan_state)

    # Fenced example where the code block appears AFTER a state marker: not clarification.
    state_then_fenced = (
        "<!-- AGENT_PLAN_STATE: blocking -->\n\n"
        "Appendix:\n\n"
        "```\n"
        "<!-- AGENT_CLARIFY -->\n"
        "```"
    )
    assert not is_clarification_request(state_then_fenced)

    # Tilde fence also excluded.
    tilde_fenced = (
        "~~~\n"
        "<!-- AGENT_CLARIFY -->\n"
        "~~~"
    )
    assert not is_clarification_request(tilde_fenced)

    # Real standalone AGENT_CLARIFY outside a fence: still detected.
    outside_fence = (
        "```\n"
        "some code\n"
        "```\n\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(outside_fence)

    # AGENT_CLARIFY both inside and outside a fence: outside occurrence is active.
    inside_and_outside = (
        "```\n"
        "<!-- AGENT_CLARIFY -->\n"
        "```\n\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(inside_and_outside)


def test_is_clarification_request_requires_clarify_at_end():
    # Non-blank, non-signature content after AGENT_CLARIFY means it's embedded.
    embedded_with_trailing = (
        "<!-- AGENT_CLARIFY -->\n\n"
        "Some trailing prose that isn't a signature.\n"
        "<!-- AGENT_STATE: blocking -->"
    )
    # AGENT_STATE suppresses it via the presence-based check above.
    assert not is_clarification_request(embedded_with_trailing)

    # Standalone AGENT_CLARIFY with only blank lines after it: valid.
    clarify_then_blank = "<!-- AGENT_CLARIFY -->\n\n"
    assert is_clarification_request(clarify_then_blank)

    # Standalone AGENT_CLARIFY with only a signature after it: valid.
    clarify_then_sig = "<!-- AGENT_CLARIFY -->\n-- Anthropic Claude\n"
    assert is_clarification_request(clarify_then_sig)

    # Standalone AGENT_CLARIFY with real prose content after it (no state marker):
    # should NOT be treated as an active clarification.
    clarify_then_prose = (
        "<!-- AGENT_CLARIFY -->\n\n"
        "Continuing thoughts about the plan.\n"
    )
    assert not is_clarification_request(clarify_then_prose)

    # AGENT_CLARIFY in plan body with plan footer after: suppressed by state marker
    # (presence-based check catches it before trailing-content check).
    in_plan_body = (
        "Here are my questions:\n\n"
        "<!-- AGENT_CLARIFY -->\n\n"
        "More explanation here.\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n"
    )
    assert not is_clarification_request(in_plan_body)

    # Multiple AGENT_CLARIFY; last one has only signature trailing: valid.
    multi_clarify = (
        "First question set:\n<!-- AGENT_CLARIFY -->\n\nOther text.\n\n"
        "<!-- AGENT_CLARIFY -->\n-- Anthropic Claude\n"
    )
    assert is_clarification_request(multi_clarify)

    # Multiple AGENT_CLARIFY; last one has prose trailing: not valid.
    multi_clarify_bad = (
        "<!-- AGENT_CLARIFY -->\n\n"
        "<!-- AGENT_CLARIFY -->\n\n"
        "But wait, there's more content.\n"
    )
    assert not is_clarification_request(multi_clarify_bad)


# ---------------------------------------------------------------------------
# Reverse flow: Codex creates PR, Claude reviews
# ---------------------------------------------------------------------------


def test_public_response_file_instruction_mentions_plan_revision_human_ack_exception(tmp_path):
    prompt = with_public_response_file_instruction(
        "Review the PR.",
        tmp_path / "response.md",
    )

    assert "For structured plan revisions only" in prompt
    assert "<!-- HUMAN_REQUIREMENTS_ADDRESSED -->" in prompt
    assert "`### Human requirements` section after the JSON object" in prompt
    assert "before the\n`AGENT_PLAN_STATE` footer" in prompt


# ---------------------------------------------------------------------------
# Repair pass tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# New tests for issue #246: repair approved reviews with active prior dispositions
# ---------------------------------------------------------------------------


# --- repair.py prompt content tests ---


# --- _reviewer_human_requirements_instruction tests ---


# --- _surfaced_reviewer_requirement_ids tests ---


# --- PR loop repair-first tests ---


# --- Plan loop repair-first tests ---


# --- Protocol regression tests ---


# ---------------------------------------------------------------------------
# Round 2 follow-up tests: same-pr/same-plan followup recording and FORMAT fix
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests for issue #273: deterministic recovery of same-round prior-item dispositions
# ---------------------------------------------------------------------------


# --- Unit tests for strip_unknown_prior_item_dispositions ---


# --- Integration tests via _run_validated_agent ---


# ---------------------------------------------------------------------------
# Antigravity (agy) backend + Gemini retirement guidance (#215)
# ---------------------------------------------------------------------------


def test_runner_pty_reports_tty_and_strips_ansi(tmp_path):
    """The real PTY path: the child sees a TTY and ANSI codes are stripped."""
    import sys
    from coding_review_agent_loop.runner import Runner, strip_ansi

    assert strip_ansi("\x1b[31mred\x1b[0m\r\ndone") == "red\ndone"

    program = (
        "import sys\n"
        "sys.stdout.write('istty=%s\\n' % sys.stdout.isatty())\n"
        "sys.stdout.write('\\x1b[32mGREEN\\x1b[0m\\n')\n"
    )
    log_path = tmp_path / "logs" / "pty.log"
    result = Runner().run_with_log(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        log_path=log_path,
        label="PtyProbe",
        progress_interval_seconds=999,
        check=True,
        use_pty=True,
    )
    assert "istty=True" in result.stdout
    assert "GREEN" in result.stdout
    assert "\x1b[" not in result.stdout  # ANSI stripped from captured output
    assert result.returncode == 0


def test_runner_pty_preserves_bytes_and_reports_unexpected_master_read_failure(
    tmp_path, monkeypatch
):
    import errno
    import pty
    import coding_review_agent_loop.runner as runner_module

    program = (
        "import sys, time\n"
        "sys.stdout.write('captured-before-read-failure\\n'); sys.stdout.flush()\n"
        "time.sleep(1)\n"
    )
    calls = [0]
    master_fd: list[int] = []
    real_openpty = pty.openpty
    real_read = runner_module.os.read

    def capture_master_fd():
        pair = real_openpty()
        master_fd.append(pair[0])
        return pair

    monkeypatch.setattr(pty, "openpty", capture_master_fd)

    def fail_after_first_read(fd, size):
        if not master_fd or fd != master_fd[0]:
            return real_read(fd, size)
        calls[0] += 1
        if calls[0] == 1:
            # Seed the real PTY capture buffer deterministically; process
            # scheduling may otherwise let the child close before the first
            # select/read cycle is reached.
            return b"captured-before-read-failure\n"
        raise OSError(errno.EBADF, "injected PTY read failure")

    monkeypatch.setattr(runner_module.os, "read", fail_after_first_read)
    result = Runner().run_with_log(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        log_path=tmp_path / "logs" / "pty-read-failure.log",
        label="PtyReadFailureProbe",
        progress_interval_seconds=999,
        check=False,
        use_pty=True,
    )

    assert "captured-before-read-failure" in result.stdout
    assert result.capture_diagnostics == (
        "capture_read_failed:OSError:[Errno 9] injected PTY read failure",
    )


def test_runner_keeps_output_when_capture_path_is_unlinked(tmp_path):
    """An open capture descriptor survives cleanup unlinking its pathname."""
    program = (
        "import os, sys, time\n"
        "path = sys.argv[1]\n"
        "print('before-unlink', flush=True)\n"
        "os.unlink(path)\n"
        "print('after-unlink', flush=True)\n"
    )
    log_path = tmp_path / "logs" / "unlinked.log"
    result = Runner().run_with_log(
        [sys.executable, "-c", program, str(log_path)],
        cwd=tmp_path,
        log_path=log_path,
        label="UnlinkProbe",
        progress_interval_seconds=999,
    )
    assert result.returncode == 0
    assert "before-unlink" in result.stdout
    assert "after-unlink" in result.stdout
    assert result.capture_diagnostics == ()


def test_runner_classifies_invalid_capture_encoding_without_crashing(tmp_path):
    program = "import sys; sys.stdout.buffer.write(b'\\xff'); sys.stdout.flush()"
    result = Runner().run_with_log(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        log_path=tmp_path / "logs" / "invalid-utf8.log",
        label="InvalidEncodingProbe",
        progress_interval_seconds=999,
    )
    assert result.returncode == 0
    assert result.capture_diagnostics[0].startswith("capture_read_failed:UnicodeDecodeError:")


@pytest.mark.parametrize("use_pty", [False, True])
def test_runner_execution_observation_elapsed_begins_at_spawn(tmp_path, use_pty):
    """Pipe and PTY observations use the same post-spawn elapsed interval."""
    result = Runner().run_with_log(
        [sys.executable, "-c", "print('done')"],
        cwd=tmp_path,
        log_path=tmp_path / "logs" / f"observation-{use_pty}.log",
        label="Observation probe",
        progress_interval_seconds=999,
        use_pty=use_pty,
    )

    assert result.observation is not None
    assert result.observation.elapsed_seconds == (
        result.observation.exit_monotonic - result.observation.spawn_monotonic
    )


def test_wait_for_executable_stability_accepts_absolute_command_with_matching_identity(monkeypatch, tmp_path):
    """Explicit updater diagnostics may recover an absolute Claude override."""
    import coding_review_agent_loop.runner as runner_module

    command = str(tmp_path / "claude")
    identity = ExecutableIdentity(command, command, (1, 1, 1), (1, 1, 1))
    monkeypatch.setattr(runner_module, "executable_identity", lambda _command: identity)

    assert Runner().wait_for_executable_stability(command)


def test_runner_with_log_passes_large_input_on_stdin(tmp_path):
    import sys

    prompt = "x" * 150_000
    result = Runner().run_with_log(
        [sys.executable, "-c", "import sys; print(len(sys.stdin.read()))"],
        cwd=tmp_path,
        log_path=tmp_path / "logs" / "stdin.log",
        label="Stdin probe",
        progress_interval_seconds=999,
        input_text=prompt,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == str(len(prompt))
    log_text = (tmp_path / "logs" / "stdin.log").read_text(encoding="utf-8")
    assert f"# stdin\n{prompt}\n" in log_text


@pytest.mark.parametrize("use_pty", [False, True])
def test_runner_retries_dangling_symlink_spawn_and_recovers(
    monkeypatch,
    tmp_path,
    use_pty,
):
    import coding_review_agent_loop.runner as runner_module

    command_name = "bare-agent"
    missing_target = tmp_path / "updating-agent-target"
    command = tmp_path / command_name
    command.symlink_to(missing_target)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    runner = Runner()
    runner.remember_agent_command(command_name, str(command), "--codex-cmd")
    original_popen = runner_module.subprocess.Popen
    popen_calls = []
    sleep_calls = []

    def flaky_popen(*args, **kwargs):
        popen_calls.append(args[0])
        if len(popen_calls) == 1:
            raise FileNotFoundError(command_name)
        return original_popen(*args, **kwargs)

    def restore_command(delay):
        sleep_calls.append(delay)
        command.unlink()
        command.symlink_to(sys.executable)

    monkeypatch.setattr(runner_module.subprocess, "Popen", flaky_popen)
    monkeypatch.setattr(runner_module.time, "sleep", restore_command)

    result = runner.run_with_log(
        [command_name, "-c", "print('recovered')"],
        cwd=tmp_path,
        log_path=tmp_path / "logs" / f"retry-{use_pty}.log",
        label="Retry probe",
        progress_interval_seconds=999,
        use_pty=use_pty,
    )

    assert result.returncode == 0
    assert "recovered" in result.stdout
    assert len(popen_calls) == 2
    assert sleep_calls[0] == 2
    assert all(delay == 1 for delay in sleep_calls[1:])


@pytest.mark.parametrize("use_pty", [False, True])
def test_runner_dangling_symlink_spawn_retry_is_bounded(
    monkeypatch,
    tmp_path,
    use_pty,
):
    import coding_review_agent_loop.runner as runner_module

    command_name = "bare-agent"
    command = tmp_path / command_name
    command.symlink_to(tmp_path / "missing-target")
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    runner = Runner()
    runner.remember_agent_command(command_name, str(command), "--codex-cmd")
    popen_calls = []
    sleep_calls = []

    def missing_popen(*args, **kwargs):
        popen_calls.append(args[0])
        raise FileNotFoundError(command_name)

    monkeypatch.setattr(runner_module.subprocess, "Popen", missing_popen)
    monkeypatch.setattr(
        runner_module.time,
        "sleep",
        lambda delay: sleep_calls.append(delay),
    )

    with pytest.raises(
        AgentLoopError,
        match=r"CLI not found on PATH.*--codex-cmd",
    ):
        runner.run_with_log(
            [command_name, "--version"],
            cwd=tmp_path,
            log_path=tmp_path / "logs" / f"bounded-{use_pty}.log",
            label="Bounded retry probe",
            progress_interval_seconds=999,
            use_pty=use_pty,
        )

    assert len(popen_calls) == 3
    assert sleep_calls == [2, 2]


@pytest.mark.parametrize("use_pty", [False, True])
def test_runner_retries_preflighted_command_disappearance_and_recovers(
    monkeypatch,
    tmp_path,
    use_pty,
):
    """A bare CLI that vanishes after preflight can return during spawn retry."""
    import coding_review_agent_loop.runner as runner_module

    command_name = "bare-agent"
    command = tmp_path / command_name
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    runner = Runner()
    runner.remember_agent_command(
        command_name,
        str(tmp_path / "preflight-agent-path"),
        "--codex-cmd",
    )
    original_popen = runner_module.subprocess.Popen
    original_which = runner_module.shutil.which
    popen_calls = []
    which_calls = []
    sleep_calls = []

    def tracking_which(name):
        which_calls.append(name)
        return original_which(name)

    def flaky_popen(*args, **kwargs):
        popen_calls.append(args[0])
        if len(popen_calls) < 3:
            raise FileNotFoundError(command_name)
        return original_popen(*args, **kwargs)

    def restore_command(delay):
        sleep_calls.append(delay)
        if delay == 2 and not command.exists():
            command.symlink_to(sys.executable)

    monkeypatch.setattr(runner_module.shutil, "which", tracking_which)
    monkeypatch.setattr(runner_module.subprocess, "Popen", flaky_popen)
    monkeypatch.setattr(runner_module.time, "sleep", restore_command)

    result = runner.run_with_log(
        [command_name, "-c", "print('recovered after preflight')"],
        cwd=tmp_path,
        log_path=tmp_path / "logs" / f"preflight-retry-{use_pty}.log",
        label="Preflight retry probe",
        progress_interval_seconds=999,
        use_pty=use_pty,
    )

    assert result.returncode == 0
    assert "recovered after preflight" in result.stdout
    assert len(popen_calls) == 3
    # Each candidate spawn is resolved immediately before its attempt so the
    # returned observation belongs to the successful attempt; the post-exit
    # sample adds one final lookup.
    assert which_calls == [command_name] * 7
    assert runner._resolved_commands[command_name] == str(command)
    assert sleep_calls[:2] == [2, 2]
    assert all(delay == 1 for delay in sleep_calls[2:])


@pytest.mark.parametrize("use_pty", [False, True])
def test_runner_preflighted_command_disappearance_retry_is_bounded(
    monkeypatch,
    tmp_path,
    use_pty,
):
    """A post-preflight PATH disappearance gives a specific bounded error."""
    import coding_review_agent_loop.runner as runner_module

    command_name = "bare-agent"
    runner = Runner()
    runner.remember_agent_command(
        command_name,
        str(tmp_path / "preflight-agent-path"),
        "--codex-cmd",
    )
    popen_calls = []
    which_calls = []
    sleep_calls = []

    def missing_popen(*args, **kwargs):
        popen_calls.append(args[0])
        raise FileNotFoundError(command_name)

    def missing_which(name):
        which_calls.append(name)
        return None

    monkeypatch.setattr(runner_module.shutil, "which", missing_which)
    monkeypatch.setattr(runner_module.subprocess, "Popen", missing_popen)
    monkeypatch.setattr(
        runner_module.time,
        "sleep",
        lambda delay: sleep_calls.append(delay),
    )

    with pytest.raises(
        AgentLoopError,
        match=r"disappeared after successful preflight and may be updating",
    ):
        runner.run_with_log(
            [command_name, "--version"],
            cwd=tmp_path,
            log_path=tmp_path / "logs" / f"preflight-bounded-{use_pty}.log",
            label="Preflight bounded retry probe",
            progress_interval_seconds=999,
            use_pty=use_pty,
        )

    assert len(popen_calls) == 3
    assert which_calls == [command_name] * 7
    assert sleep_calls == [2, 2]


@pytest.mark.parametrize("use_pty", [False, True])
def test_runner_preflighted_command_permission_error_does_not_retry(
    monkeypatch,
    tmp_path,
    use_pty,
):
    """Only FileNotFoundError and preflighted ENOEXEC/ETXTBSY can retry."""
    import coding_review_agent_loop.runner as runner_module

    command_name = "bare-agent"
    runner = Runner()
    runner.remember_agent_command(command_name, "/bin/bare-agent", "--codex-cmd")
    popen_calls = []
    sleep_calls = []

    def denied_popen(*args, **kwargs):
        popen_calls.append(args[0])
        raise PermissionError(command_name)

    monkeypatch.setattr(runner_module.subprocess, "Popen", denied_popen)
    monkeypatch.setattr(
        runner_module.time,
        "sleep",
        lambda delay: sleep_calls.append(delay),
    )

    with pytest.raises(PermissionError):
        runner.run_with_log(
            [command_name, "--version"],
            cwd=tmp_path,
            log_path=tmp_path / "logs" / f"permission-no-retry-{use_pty}.log",
            label="Permission no-retry probe",
            progress_interval_seconds=999,
            use_pty=use_pty,
        )

    assert len(popen_calls) == 1
    assert sleep_calls == []


@pytest.mark.parametrize("use_pty", [False, True])
def test_runner_preflighted_exec_format_error_retries(monkeypatch, tmp_path, use_pty):
    """A bare preflighted CLI may be briefly non-executable during auto-update."""
    import errno
    import coding_review_agent_loop.runner as runner_module

    command_name = "bare-agent"
    runner = Runner()
    runner.remember_agent_command(command_name, "/tmp/bare-agent", "--claude-cmd")
    popen_calls = []
    sleep_calls = []

    def invalid_popen(*args, **kwargs):
        popen_calls.append(args[0])
        raise OSError(errno.ENOEXEC, "Exec format error")

    monkeypatch.setattr(runner_module.subprocess, "Popen", invalid_popen)
    monkeypatch.setattr(
        runner_module.time,
        "sleep",
        lambda delay: sleep_calls.append(delay),
    )

    with pytest.raises(AgentLoopError, match="remained temporarily non-executable"):
        runner.run_with_log(
            [command_name, "--version"],
            cwd=tmp_path,
            log_path=tmp_path / "logs" / f"exec-format-retry-{use_pty}.log",
            label="Exec format retry probe",
            progress_interval_seconds=999,
            use_pty=use_pty,
        )

    assert len(popen_calls) == 3
    assert sleep_calls == [2, 2]


@pytest.mark.parametrize("command", ["bare-agent", "/tmp/bare-agent"])
def test_runner_unpreflighted_or_absolute_exec_format_error_does_not_retry(
    monkeypatch,
    tmp_path,
    command,
):
    """ENOEXEC is fail-closed for unpreflighted bare and absolute commands."""
    import errno
    import coding_review_agent_loop.runner as runner_module

    runner = Runner()
    popen_calls = []
    sleep_calls = []

    def invalid_popen(*args, **kwargs):
        popen_calls.append(args[0])
        raise OSError(errno.ENOEXEC, "Exec format error")

    monkeypatch.setattr(runner_module.subprocess, "Popen", invalid_popen)
    monkeypatch.setattr(
        runner_module.time,
        "sleep",
        lambda delay: sleep_calls.append(delay),
    )

    with pytest.raises(OSError) as exc_info:
        runner.run_with_log(
            [command, "--version"],
            cwd=tmp_path,
            log_path=tmp_path / "logs" / "exec-format-no-retry.log",
            label="Exec format no-retry probe",
            progress_interval_seconds=999,
        )

    assert exc_info.value.errno == errno.ENOEXEC
    assert len(popen_calls) == 1
    assert sleep_calls == []


def test_runner_missing_command_without_dangling_evidence_does_not_retry(
    monkeypatch,
    tmp_path,
):
    import coding_review_agent_loop.runner as runner_module

    popen_calls = []
    sleep_calls = []
    which_calls = []

    def runtime_only_which(command):
        which_calls.append(command)
        return "/tmp/runtime-agent" if len(which_calls) == 1 else None

    def missing_popen(*args, **kwargs):
        popen_calls.append(args[0])
        raise FileNotFoundError("missing-agent")

    monkeypatch.setattr(runner_module.shutil, "which", runtime_only_which)
    monkeypatch.setattr(runner_module.subprocess, "Popen", missing_popen)
    monkeypatch.setattr(
        runner_module.time,
        "sleep",
        lambda delay: sleep_calls.append(delay),
    )

    with pytest.raises(AgentLoopError, match="missing-agent CLI not found on PATH"):
        Runner().run_with_log(
            ["missing-agent", "--version"],
            cwd=tmp_path,
            log_path=tmp_path / "logs" / "missing.log",
            label="Missing probe",
            progress_interval_seconds=999,
        )

    assert len(popen_calls) == 1
    assert which_calls == ["missing-agent"] * 3
    assert sleep_calls == []


@pytest.mark.parametrize("use_pty", [False, True])
def test_runner_absolute_path_spawn_does_not_retry(monkeypatch, tmp_path, use_pty):
    """Absolute-path FileNotFoundError raises immediately (no retry, no sleep)."""
    import coding_review_agent_loop.runner as runner_module

    abs_cmd = str(tmp_path / "no-such-binary")
    popen_calls = []
    sleep_calls = []

    def missing_popen(*args, **kwargs):
        popen_calls.append(args[0])
        raise FileNotFoundError(abs_cmd)

    monkeypatch.setattr(runner_module.subprocess, "Popen", missing_popen)
    monkeypatch.setattr(
        runner_module.time, "sleep", lambda delay: sleep_calls.append(delay),
    )

    with pytest.raises(AgentLoopError, match="not found or not executable"):
        Runner().run_with_log(
            [abs_cmd, "--version"],
            cwd=tmp_path,
            log_path=tmp_path / "logs" / f"abs-no-retry-{use_pty}.log",
            label="Absolute-path no-retry probe",
            progress_interval_seconds=999,
            use_pty=use_pty,
        )

    assert len(popen_calls) == 1
    assert sleep_calls == []


def _self_update_command(tmp_path, monkeypatch, absolute):
    command = tmp_path / "bare-agent"
    command.symlink_to(sys.executable)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    return str(command) if absolute else "bare-agent"


@pytest.mark.parametrize("use_pty", [False, True])
@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("errno_value", [errno.ETXTBSY, errno.ENOEXEC])
def test_runner_preflighted_self_update_exec_error_recovers(
    monkeypatch, tmp_path, use_pty, absolute, errno_value
):
    """#1226: a preflighted CLI busy/non-executable mid-update recovers after one backoff."""
    import coding_review_agent_loop.runner as runner_module

    command = _self_update_command(tmp_path, monkeypatch, absolute)
    runner = Runner()
    runner.remember_agent_command(command, str(tmp_path / "bare-agent"), "--claude-cmd")
    original_popen = runner_module.subprocess.Popen
    popen_calls = []
    sleep_calls = []

    def flaky_popen(*args, **kwargs):
        popen_calls.append(args[0])
        if len(popen_calls) == 1:
            raise OSError(errno_value, os.strerror(errno_value))
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(runner_module.subprocess, "Popen", flaky_popen)
    monkeypatch.setattr(runner_module.time, "sleep", lambda delay: sleep_calls.append(delay))

    result = runner.run_with_log(
        [command, "-c", "print('recovered')"],
        cwd=tmp_path,
        log_path=tmp_path / "logs" / f"self-update-{use_pty}.log",
        label="Self-update probe",
        progress_interval_seconds=999,
        use_pty=use_pty,
    )

    assert result.returncode == 0
    assert "recovered" in result.stdout
    assert len(popen_calls) == 2
    assert sleep_calls[0] == 2


@pytest.mark.parametrize("use_pty", [False, True])
@pytest.mark.parametrize("command", ["bare-agent", "/opt/bin/bare-agent"])
@pytest.mark.parametrize(
    ("errno_value", "expected"),
    [
        (errno.ETXTBSY, "remained temporarily busy"),
        (errno.ENOEXEC, "remained temporarily non-executable"),
    ],
)
def test_runner_preflighted_self_update_exec_error_is_bounded(
    monkeypatch, tmp_path, use_pty, command, errno_value, expected
):
    import coding_review_agent_loop.runner as runner_module

    runner = Runner()
    runner.remember_agent_command(command, "/opt/bin/bare-agent", "--claude-cmd")
    popen_calls = []
    sleep_calls = []

    def busy_popen(*args, **kwargs):
        popen_calls.append(args[0])
        raise OSError(errno_value, os.strerror(errno_value))

    monkeypatch.setattr(runner_module.subprocess, "Popen", busy_popen)
    monkeypatch.setattr(runner_module.time, "sleep", lambda delay: sleep_calls.append(delay))

    with pytest.raises(AgentLoopError, match=rf"{expected}.*--claude-cmd"):
        runner.run_with_log(
            [command, "--version"],
            cwd=tmp_path,
            log_path=tmp_path / "logs" / f"busy-{use_pty}.log",
            label="Busy probe",
            progress_interval_seconds=999,
            use_pty=use_pty,
        )

    assert len(popen_calls) == runner_module._SPAWN_ATTEMPTS
    assert sleep_calls == [2] * (runner_module._SPAWN_ATTEMPTS - 1)


@pytest.mark.parametrize("command", ["bare-agent", "/opt/bin/bare-agent"])
def test_runner_unpreflighted_text_busy_does_not_retry(monkeypatch, tmp_path, command):
    import coding_review_agent_loop.runner as runner_module

    popen_calls = []
    sleep_calls = []

    def busy_popen(*args, **kwargs):
        popen_calls.append(args[0])
        raise OSError(errno.ETXTBSY, "Text file busy")

    monkeypatch.setattr(runner_module.subprocess, "Popen", busy_popen)
    monkeypatch.setattr(runner_module.time, "sleep", lambda delay: sleep_calls.append(delay))

    with pytest.raises(OSError) as exc_info:
        Runner().run_with_log(
            [command, "--version"],
            cwd=tmp_path,
            log_path=tmp_path / "logs" / "busy-no-retry.log",
            label="Busy no-retry probe",
            progress_interval_seconds=999,
        )

    assert exc_info.value.errno == errno.ETXTBSY
    assert len(popen_calls) == 1
    assert sleep_calls == []


@pytest.mark.parametrize("use_pty", [False, True])
def test_runner_preflighted_absolute_command_disappearance_recovers(
    monkeypatch, tmp_path, use_pty
):
    """#1226: a preflighted absolute --<agent>-cmd briefly missing is retried, bounded."""
    import coding_review_agent_loop.runner as runner_module

    command = str(tmp_path / "abs-agent")
    runner = Runner()
    runner.remember_agent_command(command, command, "--claude-cmd")
    original_popen = runner_module.subprocess.Popen
    popen_calls = []
    sleep_calls = []

    def flaky_popen(*args, **kwargs):
        popen_calls.append(args[0])
        if len(popen_calls) == 1:
            raise FileNotFoundError(command)
        return original_popen(*args, **kwargs)

    def restore_command(delay):
        sleep_calls.append(delay)
        if not os.path.lexists(command):
            os.symlink(sys.executable, command)

    monkeypatch.setattr(runner_module.subprocess, "Popen", flaky_popen)
    monkeypatch.setattr(runner_module.time, "sleep", restore_command)

    result = runner.run_with_log(
        [command, "-c", "print('absolute recovered')"],
        cwd=tmp_path,
        log_path=tmp_path / "logs" / f"abs-retry-{use_pty}.log",
        label="Absolute retry probe",
        progress_interval_seconds=999,
        use_pty=use_pty,
    )

    assert result.returncode == 0
    assert "absolute recovered" in result.stdout
    assert len(popen_calls) == 2
    assert sleep_calls[0] == 2


def test_runner_preflighted_absolute_command_disappearance_is_bounded(monkeypatch, tmp_path):
    import coding_review_agent_loop.runner as runner_module

    command = str(tmp_path / "abs-agent")
    runner = Runner()
    runner.remember_agent_command(command, command, "--claude-cmd")
    popen_calls = []
    sleep_calls = []

    def missing_popen(*args, **kwargs):
        popen_calls.append(args[0])
        raise FileNotFoundError(command)

    monkeypatch.setattr(runner_module.subprocess, "Popen", missing_popen)
    monkeypatch.setattr(runner_module.time, "sleep", lambda delay: sleep_calls.append(delay))

    with pytest.raises(AgentLoopError, match="disappeared after successful preflight"):
        runner.run_with_log(
            [command, "--version"],
            cwd=tmp_path,
            log_path=tmp_path / "logs" / "abs-bounded.log",
            label="Absolute bounded probe",
            progress_interval_seconds=999,
        )

    assert len(popen_calls) == runner_module._SPAWN_ATTEMPTS
    assert sleep_calls == [2] * (runner_module._SPAWN_ATTEMPTS - 1)


def test_runner_unpreflighted_dangling_symlink_fails_fast(monkeypatch, tmp_path):
    """A cached dangling candidate is not retry evidence without preflight (#1226)."""
    import coding_review_agent_loop.runner as runner_module

    command_name = "bare-agent"
    command = tmp_path / command_name
    command.symlink_to(sys.executable)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    runner = Runner()
    popen_calls = []
    sleep_calls = []

    def dangling_popen(*args, **kwargs):
        popen_calls.append(args[0])
        # The runtime resolution was cached; the target now disappears.
        command.unlink()
        command.symlink_to(tmp_path / "missing-target")
        raise FileNotFoundError(command_name)

    monkeypatch.setattr(runner_module.subprocess, "Popen", dangling_popen)
    monkeypatch.setattr(runner_module.time, "sleep", lambda delay: sleep_calls.append(delay))

    with pytest.raises(AgentLoopError, match="bare-agent CLI not found on PATH"):
        runner.run_with_log(
            [command_name, "--version"],
            cwd=tmp_path,
            log_path=tmp_path / "logs" / "dangling-fast.log",
            label="Dangling fail-fast probe",
            progress_interval_seconds=999,
        )

    assert runner._resolved_commands[command_name] == str(command)
    assert len(popen_calls) == 1
    assert sleep_calls == []


# ---------------------------------------------------------------------------
# Dynamic model-specific signatures (#332)
# ---------------------------------------------------------------------------


def test_agent_signature_generic_without_config():
    from coding_review_agent_loop.agents.registry import agent_signature
    assert agent_signature("codex") == "OpenAI Codex"
    assert agent_signature("antigravity") == "Google Antigravity"


def test_agent_signature_uses_configured_model(tmp_path):
    from coding_review_agent_loop.agents.registry import agent_signature
    config = make_config(tmp_path, codex_model="gpt-5.2-codex", codex_reasoning_effort="high")
    assert agent_signature("codex", config) == "OpenAI Codex: gpt-5.2-codex (high)"
    assert (
        agent_signature("codex", make_config(tmp_path, codex_model="gpt-5.5"))
        == "OpenAI Codex: gpt-5.5 (medium)"
    )
    assert (
        agent_signature("claude", make_config(tmp_path, claude_model="claude-sonnet"))
        == "Anthropic Claude: claude-sonnet (medium)"
    )
    assert (
        agent_signature(
            "codex", make_config(tmp_path, codex_reasoning_effort="high")
        )
        == "OpenAI Codex: unknown model (high)"
    )
    # antigravity model is always declared (effort already embedded).
    assert agent_signature("antigravity", config) == "Google Antigravity: Gemini 3.8 Flash (High)"
    # gemini with no declared model falls back to the generic signature.
    assert agent_signature("gemini", make_config(tmp_path)) == "Google Gemini"


def test_agent_signature_model_used_overrides_config(tmp_path):
    from coding_review_agent_loop.agents.registry import agent_signature
    config = make_config(tmp_path, antigravity_model="Gemini 3.1 Pro (High)")
    # #333 fallback: the model that actually ran wins over the configured one.
    assert (
        agent_signature("antigravity", config, "Gemini 3.5 Flash (High)")
        == "Google Antigravity: Gemini 3.5 Flash (High)"
    )


def test_config_rejects_model_arg_conflicts(tmp_path):
    for kwargs in (
        {"codex_model": "gpt-5", "codex_args": ("--model", "other")},
        {"codex_reasoning_effort": "high", "codex_args": ("-c", 'model_reasoning_effort="low"')},
        {"gemini_model": "g", "gemini_args": ("--model", "other")},
        {"claude_model": "c", "claude_args": ("--model", "other")},
        {"antigravity_args": ("--model", "x")},
    ):
        with pytest.raises(AgentLoopError, match="conflicts with"):
            make_config(tmp_path, **kwargs)


def test_config_allows_codex_effort_without_model(tmp_path):
    # Rollout evidence may identify the model later; model omission is valid and
    # the effort remains independently reportable.
    config = make_config(tmp_path, codex_reasoning_effort="high")
    assert config.codex_model == ""
    from coding_review_agent_loop.config import resolve_invocation
    assert resolve_invocation(config, provider="codex").resolved_effort == "high"
    # With a model it's accepted as well.
    config = make_config(tmp_path, codex_model="gpt-5", codex_reasoning_effort="high")
    assert config.codex_reasoning_effort == "high"


def test_effort_resolution_preserves_omitted_vs_explicit_and_role_precedence(tmp_path):
    from coding_review_agent_loop.config import resolve_invocation

    omitted = make_config(tmp_path, coder="claude")
    resolved = resolve_invocation(omitted, provider="claude", role="coder")
    assert (resolved.resolved_effort, resolved.effort_source) == ("medium", "tool_default")

    explicit = make_config(tmp_path, codex_reasoning_effort="high")
    agent_wide = resolve_invocation(explicit, provider="codex", role="reviewer")
    assert (agent_wide.resolved_effort, agent_wide.effort_source) == ("high", "agent_wide")

    implementation = make_config(
        tmp_path,
        coder="codex",
        codex_reasoning_effort="high",
        implementation_coder="codex",
        implementation_codex_reasoning_effort="xhigh",
    )
    role_override = resolve_invocation(
        implementation, provider="codex", role="coder", implementation=True
    )
    assert (role_override.resolved_effort, role_override.effort_source) == (
        "xhigh",
        "role_override",
    )


def test_provider_switch_does_not_carry_effort_between_claude_and_codex(tmp_path):
    from coding_review_agent_loop.config import resolve_invocation

    config = make_config(
        tmp_path,
        coder="claude",
        claude_effort="high",
        codex_reasoning_effort="xhigh",
        implementation_coder="codex",
    )
    assert resolve_invocation(config, provider="claude").resolved_effort == "high"
    assert resolve_invocation(config, provider="codex", implementation=True).resolved_effort == "xhigh"


@pytest.mark.parametrize("value", ["", "invalid", "MAX"])
def test_claude_effort_values_are_validated(tmp_path, value):
    if value == "":
        assert make_config(tmp_path, claude_effort=value).claude_effort == ""
    else:
        with pytest.raises(AgentLoopError, match="--claude-effort"):
            make_config(tmp_path, claude_effort=value)


def test_config_allows_declared_model_without_conflict(tmp_path):
    config = make_config(tmp_path, codex_model="gpt-5", gemini_model="g", claude_model="c")
    assert config.codex_model == "gpt-5"
    assert config.gemini_model == "g"
    assert config.claude_model == "c"


def test_codex_backend_passes_model_and_effort(tmp_path):
    from coding_review_agent_loop.agents.codex import CodexBackend
    runner = FakeRunner(codex_outputs=[("STATE: approved\n\nok", 0)])
    config = make_config(tmp_path, codex_model="gpt-5.2-codex", codex_reasoning_effort="high")
    result = CodexBackend().run(runner, config, "Review", run_id="r")
    cmd = runner.commands[-1][0]
    assert cmd[cmd.index("--model") + 1] == "gpt-5.2-codex"
    assert 'model_reasoning_effort="high"' in cmd
    assert result.model_used == "gpt-5.2-codex (high)"


def test_gemini_backend_passes_model_and_sets_model_used(tmp_path):
    import coding_review_agent_loop.agents.gemini as gm
    runner = FakeRunner(gemini_outputs=[("STATE: approved\n\nok", 0)])
    config = make_config(tmp_path, gemini_model="gemini-3.5-flash")
    result = gm.BACKEND.run(runner, config, "Review", run_id="r")
    cmd = runner.commands[-1][0]
    assert cmd[cmd.index("--model") + 1] == "gemini-3.5-flash"
    assert result.model_used == "gemini-3.5-flash"


def test_claude_backend_passes_model_when_declared(tmp_path):
    from coding_review_agent_loop.agents.claude import ClaudeBackend
    runner = FakeRunner(claude_outputs=[("STATE: approved\n\nok", 0)])
    config = make_config(tmp_path, claude_model="opus")
    ClaudeBackend().run(runner, config, "Review", run_id="r")
    cmd = next(cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert cmd[cmd.index("--model") + 1] == "opus"


def test_public_reviewer_name_config_aware_no_leakage(tmp_path):
    from coding_review_agent_loop.comment_rendering import _public_reviewer_name
    config = make_config(tmp_path, codex_model="gpt-5", antigravity_model="Gemini 3.1 Pro (High)")
    assert _public_reviewer_name("Codex", config) == "OpenAI Codex: gpt-5 (medium)"
    assert _public_reviewer_name("Antigravity", config) == "Google Antigravity: Gemini 3.1 Pro (High)"
    # No declared model → generic; unknown display name → passthrough.
    assert _public_reviewer_name("Claude", config) == "Anthropic Claude"
    assert _public_reviewer_name("Codex") == "OpenAI Codex"
    assert _public_reviewer_name("Somebody") == "Somebody"


def test_render_public_agent_comment_stamps_model_for_every_kind():
    model = "Gemini 3.1 Pro (High)"

    pr_review = parse_pr_review(
        structured_pr_review(state="approved", reviewer="Google Antigravity"),
        reviewer="Google Antigravity",
    )
    plan_review = parse_plan_review(
        structured_plan_review(state="approved", reviewer="Google Antigravity"),
        reviewer="Google Antigravity",
    )
    coder_followup = validate_structured_coder_followup(
        structured_coder_followup(state="approved", reviewer="Google Antigravity")
    )
    plan_revision = validate_structured_plan_revision(
        structured_plan_revision(reviewer="Google Antigravity")
    )
    assert coder_followup is not None
    assert plan_revision is not None

    rendered = [
        render_public_agent_comment(
            kind="pr_review",
            parsed=pr_review,
            agent="Antigravity",
            dispositions=pr_review.dispositions,
            model_used=model,
        ),
        render_public_agent_comment(
            kind="plan_review",
            parsed=plan_review,
            agent="Antigravity",
            dispositions=plan_review.dispositions,
            model_used=model,
        ),
        render_public_agent_comment(
            kind="coder_followup",
            parsed=coder_followup,
            agent="antigravity",
            model_used=model,
        ),
        render_public_agent_comment(
            kind="plan_revision",
            parsed=plan_revision,
            agent="antigravity",
            raw_text=structured_plan_revision(reviewer="Google Antigravity"),
            model_used=model,
        ),
    ]

    assert all(comment.endswith(f"-- Google Antigravity: {model}") for comment in rendered)


# ---------------------------------------------------------------------------
# Antigravity prompt — turn-end requirement (#385)
# ---------------------------------------------------------------------------


def test_base_response_file_instruction_includes_must_write_before_turn_ends(tmp_path):
    from coding_review_agent_loop.agents.base import with_public_response_file_instruction
    composed = with_public_response_file_instruction("BASE PROMPT", tmp_path / "response.md")
    assert "before your turn ends" in composed


# ── Tests: issue #400 – toolPermission: "strict" injection for reviewer ────────


def test_run_agent_result_passes_role_to_backend(tmp_path, monkeypatch):
    """run_agent_result threads role= through to the backend's run() method."""
    from coding_review_agent_loop.agents.registry import run_agent_result
    from coding_review_agent_loop.agents.base import AgentResult
    from coding_review_agent_loop.agents import registry as reg_mod

    captured: dict = {}

    class TrackingBackend:
        name = "gemini"
        display_name = "Gemini"
        signature = "Google Gemini"

        def workdir(self, config):
            return tmp_path

        def default_args(self, *, dangerous):
            return ()

        def run(
            self,
            runner,
            config,
            prompt,
            session_id=None,
            run_id=None,
            role=None,
            label=None,
            timeout_seconds=None,
        ):
            captured["role"] = role
            return AgentResult(text="ok")

    monkeypatch.setitem(reg_mod.BACKENDS, "gemini", TrackingBackend())
    config = make_config(tmp_path, reviewer="gemini")
    run_agent_result(FakeRunner(), agent="gemini", config=config, prompt="Test", role="reviewer")
    assert captured["role"] == "reviewer"


@pytest.mark.parametrize("use_pty", [False, True])
def test_runner_records_attempt_local_executable_observation(tmp_path, use_pty):
    """Both runner paths retain pre/post bare-command evidence for updater policy."""
    result = Runner().run_with_log(
        ["python3", "-c", "print('ok')"],
        cwd=tmp_path,
        log_path=tmp_path / f"observation-{use_pty}.log",
        label="observation",
        progress_interval_seconds=999,
        use_pty=use_pty,
    )
    assert result.returncode == 0
    assert result.observation is not None
    assert result.observation.before.path
    assert result.observation.after.path
    assert result.observation.elapsed_seconds >= 0


@pytest.mark.parametrize("use_pty", [False, True])
def test_runner_observation_before_is_from_successful_spawn_attempt(
    monkeypatch, tmp_path, use_pty
):
    """A failed spawn's identity is discarded before a later retry succeeds."""
    import coding_review_agent_loop.runner as runner_module

    command_name = "observed-agent"
    command = tmp_path / command_name
    command.symlink_to(sys.executable)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    runner = Runner()
    runner.remember_agent_command(command_name, str(command), "--codex-cmd")
    first = ExecutableIdentity("/failed", "/failed", (1, 1, 1), (1, 1, 1))
    successful = ExecutableIdentity("/successful", "/successful", (1, 2, 2), (1, 2, 2))
    after = ExecutableIdentity("/after", "/after", (1, 3, 3), (1, 3, 3))
    identities = iter([first, successful, after])
    original_popen = runner_module.subprocess.Popen
    popen_calls = []

    def flaky_popen(*args, **kwargs):
        popen_calls.append(args[0])
        if len(popen_calls) == 1:
            raise FileNotFoundError(command_name)
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(runner_module, "executable_identity", lambda _command: next(identities))
    monkeypatch.setattr(runner_module.subprocess, "Popen", flaky_popen)
    monkeypatch.setattr(runner_module.time, "sleep", lambda _delay: None)

    result = runner.run_with_log(
        [command_name, "-c", "print('observed')"],
        cwd=tmp_path,
        log_path=tmp_path / "logs" / f"successful-observation-{use_pty}.log",
        label="Successful observation",
        progress_interval_seconds=999,
        use_pty=use_pty,
    )

    assert len(popen_calls) == 2
    assert result.observation is not None
    assert result.observation.before == successful
    assert result.observation.after == after


# ---------------------------------------------------------------------------
# Shared PR review guidance unit and integration tests (#413, #417)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PostedRoundMetadata.model_used and normalize_freeform_signature (#416)
# ---------------------------------------------------------------------------


def test_posted_round_metadata_model_used_round_trip():
    metadata = PostedRoundMetadata(
        flow="plan",
        role="coder",
        agent="Claude",
        round_number=1,
        subject="abc",
        model_used="gpt-5.5 (medium)",
    )
    decoded = _decode_round_metadata(_encode_round_metadata(metadata))
    assert decoded.model_used == "gpt-5.5 (medium)"


def test_posted_round_metadata_model_used_none_round_trip():
    metadata = PostedRoundMetadata(
        flow="plan",
        role="coder",
        agent="Claude",
        round_number=1,
        subject="abc",
        model_used=None,
    )
    decoded = _decode_round_metadata(_encode_round_metadata(metadata))
    assert decoded.model_used is None


def test_posted_round_metadata_acquisition_fields_round_trip_and_legacy_default():
    metadata = PostedRoundMetadata(
        flow="plan", role="reviewer", agent="Claude", round_number=1,
        subject="abc", acquisition_outcome="accepted_nonzero_exit",
        acquisition_returncode=1,
    )
    decoded = _decode_round_metadata(_encode_round_metadata(metadata))
    assert decoded.acquisition_outcome == "accepted_nonzero_exit"
    assert decoded.acquisition_returncode == 1

    legacy_payload = {
        "flow": "plan", "role": "reviewer", "agent": "Claude",
        "round_number": 1, "subject": "abc",
    }
    legacy = _decode_round_metadata(base64.urlsafe_b64encode(
        json.dumps(legacy_payload).encode("utf-8")
    ).decode("ascii"))
    assert legacy.acquisition_outcome == "success"
    assert legacy.acquisition_returncode is None


def test_posted_round_metadata_model_used_backward_compat():
    payload = {
        "flow": "plan",
        "role": "coder",
        "agent": "Claude",
        "round_number": 1,
        "subject": "abc",
        "prior_items": [],
        "dispositions": [],
        "new_items": [],
        "state": None,
        "canonical_plan": None,
        "raw_structured_coder_response": None,
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).decode("ascii")
    decoded = _decode_round_metadata(encoded)
    assert decoded.model_used is None


def test_resume_plan_round_preserves_stored_model_used():
    plan = "Plan content.\n<!-- AGENT_PLAN_STATE: approved -->\n-- Anthropic Claude"
    coder_comment = _attach_round_metadata(
        plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=1,
            subject=_plan_subject(plan),
        ),
    )
    review_text = structured_plan_review(state="approved", summary="LGTM.")
    reviewer_comment = _attach_round_metadata(
        review_text,
        PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject=_plan_subject(plan),
            state="approved",
            model_used="gpt-5.5 (medium)",
        ),
    )
    resumed = _resume_plan_round(
        [
            IssueComment(author="bot", created_at="2026-05-20T09:00:00Z", body=coder_comment),
            IssueComment(author="bot", created_at="2026-05-20T09:01:00Z", body=reviewer_comment),
        ],
        configured_reviewers=("codex",),
    )
    assert resumed is not None
    _current_plan, state = resumed
    assert state.completed_reviews[0].metadata.model_used == "gpt-5.5 (medium)"


def test_normalize_freeform_signature_replaces_existing(tmp_path):
    config = make_config(tmp_path)
    result = normalize_freeform_signature(
        "Plan text.\n-- OpenAI Codex",
        agent="codex",
        config=config,
        model_used="gpt-5.5 (medium)",
    )
    assert result.endswith("-- OpenAI Codex: gpt-5.5 (medium)")
    assert "Plan text." in result


def test_normalize_freeform_signature_appends_when_absent(tmp_path):
    config = make_config(tmp_path)
    result = normalize_freeform_signature(
        "Plan text without a signature.",
        agent="codex",
        config=config,
        model_used="gpt-5.5 (medium)",
    )
    assert result.endswith("-- OpenAI Codex: gpt-5.5 (medium)")
    assert "Plan text without a signature." in result


def test_normalize_freeform_signature_skips_html_comments(tmp_path):
    config = make_config(tmp_path)
    text = "Plan text.\n-- OpenAI Codex\n<!-- AGENT_PLAN_STATE: approved -->"
    result = normalize_freeform_signature(
        text,
        agent="codex",
        config=config,
        model_used="gpt-5.5 (medium)",
    )
    assert "-- OpenAI Codex: gpt-5.5 (medium)" in result
    assert "<!-- AGENT_PLAN_STATE: approved -->" in result
    assert result.endswith("<!-- AGENT_PLAN_STATE: approved -->")


def test_normalize_freeform_signature_no_duplicate_when_already_canonical(tmp_path):
    config = make_config(tmp_path)
    canonical = "Plan text.\n-- OpenAI Codex: gpt-5.5 (medium)"
    result = normalize_freeform_signature(
        canonical,
        agent="codex",
        config=config,
        model_used="gpt-5.5 (medium)",
    )
    assert result == canonical


# ── _failure_suggestion unit tests ─────────────────────────────────────────


# --- host footer log once per owning invocation (#1043) ----------------------

_OWNING_RUN_ENTRIES = {"run_issue_loop", "run_task_loop", "run_pr_loop", "run_discuss_loop"}


def _orchestrator_tree():
    # Spans orchestrator.py and every module extracted from it (#1181), so
    # moved run entries and nested run_pr_loop calls stay in scope.
    return orchestrator_split_guard.combined_split_tree()


def test_orchestrator_tree_includes_registered_extracted_modules(monkeypatch):
    monkeypatch.setattr(orchestrator_split_guard, "EXTRACTED_MODULES", ("usage",))
    functions = {node.name for node in _orchestrator_tree().body if isinstance(node, ast.FunctionDef)}
    facade_source = orchestrator_split_guard.module_path("orchestrator").read_text(encoding="utf-8")
    facade_functions = {node.name for node in ast.parse(facade_source).body if isinstance(node, ast.FunctionDef)}
    assert facade_functions <= functions
    usage_source = orchestrator_split_guard.module_path("usage").read_text(encoding="utf-8")
    usage_functions = {node.name for node in ast.parse(usage_source).body if isinstance(node, ast.FunctionDef)}
    assert usage_functions and usage_functions <= functions


def test_only_owning_run_entries_create_a_usage_context():
    callers = {}
    for function in ast.walk(_orchestrator_tree()):
        if not isinstance(function, ast.FunctionDef):
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_new_usage_context":
                callers.setdefault(function.name, []).append(node)
    callers.pop("_new_usage_context", None)
    assert set(callers) == _OWNING_RUN_ENTRIES
    # Each owning entry creates a context (and so resets the latch) only when
    # it did not receive one, so a nested run shares the outer latch.
    for function in ast.walk(_orchestrator_tree()):
        if isinstance(function, ast.FunctionDef) and function.name in _OWNING_RUN_ENTRIES:
            owned = [
                node for node in ast.walk(function)
                if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or)
                and getattr(node.values[0], "id", None) == "usage_context"
                and isinstance(node.values[1], ast.Call)
                and getattr(node.values[1].func, "id", None) == "_new_usage_context"
            ]
            assert len(owned) == 1, function.name


def test_nested_run_pr_loop_calls_pass_the_outer_usage_context():
    nested = [
        node for node in ast.walk(_orchestrator_tree())
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "run_pr_loop"
    ]
    assert nested
    for call in nested:
        assert any(keyword.arg == "usage_context" for keyword in call.keywords), call.lineno


def _sub_item_stall_args(tmp_path, *extra):
    return build_parser().parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        *extra,
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
    ])


def test_sub_item_stall_rounds_defaults_to_three_and_zero_disables(tmp_path):
    from coding_review_agent_loop.config import DEFAULT_SUB_ITEM_STALL_ROUNDS

    omitted = _sub_item_stall_args(tmp_path)
    assert omitted.sub_item_stall_rounds == DEFAULT_SUB_ITEM_STALL_ROUNDS == 3
    assert config_from_args(omitted, FakeRunner()).sub_item_stall_rounds == 3
    disabled = _sub_item_stall_args(tmp_path, "--sub-item-stall-rounds", "0")
    assert config_from_args(disabled, FakeRunner()).sub_item_stall_rounds == 0
    assert config_from_args(disabled, FakeRunner()).max_rounds == omitted.max_rounds


def test_sub_item_stall_rounds_rejects_negative_values(tmp_path):
    args = _sub_item_stall_args(tmp_path, "--sub-item-stall-rounds", "-1")
    with pytest.raises(AgentLoopError, match="--sub-item-stall-rounds must be zero or greater"):
        config_from_args(args, FakeRunner())


def _preflight_args(monkeypatch, extra, missing=("agy",)):
    args = build_parser().parse_args(
        ["task", "Fix the bug", "--coder", "codex", "--reviewer", "claude", *extra]
    )
    detection_calls = []
    which_calls = []
    monkeypatch.setattr(
        "coding_review_agent_loop.config.detect_repo",
        lambda *call_args: detection_calls.append(call_args),
    )

    def fake_which(command):
        which_calls.append(command)
        return None if command in missing else f"/bin/{command}"

    monkeypatch.setattr("coding_review_agent_loop.config.shutil.which", fake_which)
    return args, detection_calls, which_calls


def _assert_repair_hint(message):
    assert "--repair-backend" in message
    assert "default is antigravity" in message
    assert "--repair-backend codex --repair-model MODEL" in message
    assert "--repair-backend claude --repair-model MODEL" in message
    assert "--antigravity-cmd" in message


@pytest.mark.parametrize("extra", [[], ["--repair-backend", "antigravity"]])
def test_config_preflight_names_repair_backend_when_agy_missing(monkeypatch, extra):
    args, detection_calls, _ = _preflight_args(monkeypatch, extra)

    with pytest.raises(AgentLoopError, match="agy CLI not found on PATH") as exc:
        config_from_args(args, Runner())

    _assert_repair_hint(str(exc.value))
    assert detection_calls == []


def test_config_preflight_repair_hint_for_absolute_antigravity_cmd(monkeypatch):
    args, _, _ = _preflight_args(
        monkeypatch, ["--antigravity-cmd", "/missing/agy"], missing=("/missing/agy",)
    )

    with pytest.raises(AgentLoopError, match="not found or not executable") as exc:
        config_from_args(args, Runner())

    _assert_repair_hint(str(exc.value))


def test_config_preflight_antigravity_role_keeps_generic_message(monkeypatch):
    args = build_parser().parse_args(
        ["task", "Fix the bug", "--coder", "codex", "--reviewer", "antigravity"]
    )
    monkeypatch.setattr(
        "coding_review_agent_loop.config.shutil.which",
        lambda command: None if command == "agy" else f"/bin/{command}",
    )

    with pytest.raises(AgentLoopError) as exc:
        config_from_args(args, Runner())

    assert str(exc.value) == "agy CLI not found on PATH; install it or pass --antigravity-cmd <path>."


def test_config_preflight_default_repair_backend_is_antigravity(monkeypatch):
    args, _, _ = _preflight_args(monkeypatch, [], missing=())
    assert args.repair_backend == "antigravity"
    monkeypatch.setattr(
        "coding_review_agent_loop.config.detect_repo",
        lambda *a: (_ for _ in ()).throw(RuntimeError("past preflight")),
    )
    with pytest.raises(RuntimeError, match="past preflight"):
        config_from_args(args, Runner())


def test_config_preflight_explicit_codex_repair_does_not_look_up_agy(monkeypatch):
    args, _, which_calls = _preflight_args(
        monkeypatch, ["--repair-backend", "codex", "--repair-model", "X"]
    )
    monkeypatch.setattr(
        "coding_review_agent_loop.config.detect_repo",
        lambda *a: (_ for _ in ()).throw(RuntimeError("past preflight")),
    )
    with pytest.raises(RuntimeError, match="past preflight"):
        config_from_args(args, Runner())
    assert "agy" not in which_calls
    assert args.repair_backend == "codex"
    assert args.repair_model == ["X"]


def test_config_preflight_codex_repair_without_model_example(monkeypatch):
    args, _, _ = _preflight_args(
        monkeypatch, ["--repair-backend", "codex", "--repo", "owner/repo"]
    )
    with pytest.raises(AgentLoopError, match="requires an explicit --repair-model") as exc:
        config_from_args(args, Runner())
    assert "--repair-model MODEL" in str(exc.value)


def test_repair_backend_help_names_default_and_alternative(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["task", "x", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "default: antigravity" in help_text
    assert "agy CLI even when no role uses Antigravity" in help_text
    assert "--repair-backend codex|claude --repair-model MODEL" in help_text


# ── #1269: validated-agent loop driven with Codex provider-error streams ──────


# --- Edited-issue planning scope metadata and opt-out (#1390) -----------------


def test_plan_issue_digest_round_trips_and_legacy_records_stay_byte_stable():
    """`metadata-roundtrip-legacy-stable`."""
    plan = "Plan.\n\n### Plan steps\n1. Step."
    legacy = PostedRoundMetadata(
        flow="plan", role="coder", agent="Claude", round_number=2,
        subject=_plan_subject(plan), canonical_plan=plan,
    )
    encoded = _encode_round_metadata(legacy)
    decoded = _decode_round_metadata(encoded)
    assert decoded.plan_issue_digest is None
    # A record without the field re-encodes byte-identically, with no new key.
    assert _encode_round_metadata(decoded) == encoded
    stamped = replace(legacy, plan_issue_digest="0123456789abcdef")
    stamped_encoded = _encode_round_metadata(stamped)
    assert stamped_encoded != encoded
    round_tripped = _decode_round_metadata(stamped_encoded)
    assert round_tripped.plan_issue_digest == "0123456789abcdef"
    assert _encode_round_metadata(round_tripped) == stamped_encoded
    # Never a scheduler checkpoint: the planner record keeps scheduler state absent.
    assert round_tripped.scheduler_metadata_status == "absent"
    assert round_tripped.scheduler_issue_digest is None
    # A malformed or non-plan digest decodes as absent.
    assert _decode_round_metadata(
        _encode_round_metadata(replace(legacy, plan_issue_digest="NOT-HEX"))
    ).plan_issue_digest is None


def test_plan_resume_keep_on_issue_edit_flag_parses_and_requires_plan_first(capsys):
    """`flag-not-inherited` (CLI/config)."""
    parser = build_parser()
    base = ["issue", "56", "--repo", "OWNER/REPO"]
    assert parser.parse_args([*base, "--plan-first"]).plan_resume_keep_on_issue_edit is False
    with_flag = parser.parse_args([*base, "--plan-first", "--plan-resume-keep-on-issue-edit"])
    assert with_flag.plan_resume_keep_on_issue_edit is True
    assert config_from_args(with_flag, FakeRunner()).plan_resume_keep_on_issue_edit is True
    assert config_from_args(
        parser.parse_args([*base, "--plan-first"]), FakeRunner()
    ).plan_resume_keep_on_issue_edit is False

    assert main([*base, "--plan-resume-keep-on-issue-edit"]) == 1
    assert "--plan-resume-keep-on-issue-edit requires --plan-first" in capsys.readouterr().err
    assert main(["pr", "7", "--repo", "OWNER/REPO", "--plan-resume-keep-on-issue-edit"]) == 1
    assert "--plan-resume-keep-on-issue-edit requires issue --plan-first" in (
        capsys.readouterr().err
    )


def test_plan_resume_keep_on_issue_edit_is_reset_in_derived_configs(tmp_path):
    """`flag-not-inherited`: approving-board, isolated-provider and recovery configs."""
    from coding_review_agent_loop.managed_ci import _render_recovery_command
    from coding_review_agent_loop.plan_verification import PlanVerificationContext
    from coding_review_agent_loop.semantic_dedupe import _isolated_provider_config
    from coding_review_agent_loop.reviewer_floor import BoardSeat

    config = make_config(
        tmp_path, reviewer=("codex", "claude"), plan_resume_keep_on_issue_edit=True,
        invocation_argv=(
            "agent-loop", "issue", "56", "--plan-first",
            "--plan-resume-keep-on-issue-edit", "--reviewer", "codex",
        ),
    )
    context = PlanVerificationContext(
        issue_number=56,
        original_reviewers=("Codex",),
        effective_reviewers=("Codex",),
        original_seats=(BoardSeat(seat_id="Codex", backend="codex"),),
        effective_seats=(BoardSeat(seat_id="Codex", backend="codex"),),
        policy="all-reviewers",
        primary_reviewer=None,
        seat_binding=None,
        signed_amendment_count=0,
    )
    assert context.config_for(config).plan_resume_keep_on_issue_edit is False
    isolated, isolated_dir = _isolated_provider_config(config, "claude", "")
    try:
        assert isolated.plan_resume_keep_on_issue_edit is False
    finally:
        import shutil

        shutil.rmtree(isolated_dir, ignore_errors=True)
    for target in ("issue", "pr"):
        rendered = _render_recovery_command(
            config, target=target, identifier=7, managed_ci=True,
            preserve_managed_options=True,
        )
        assert "--plan-resume-keep-on-issue-edit" not in rendered
