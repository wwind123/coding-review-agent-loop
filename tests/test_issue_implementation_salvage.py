"""Issue-implementation tests: salvage capture on failed implementation turns (#1543)."""

import json

import pytest
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.decomposition import approved_plan_hash
from coding_review_agent_loop.errors import QuotaResetExceededError
from coding_review_agent_loop.github import get_issue_context
import coding_review_agent_loop.agent_failure as agent_failure
import coding_review_agent_loop.issue_implementation as issue_implementation
from issue_loop_test_helpers import FakeRunner, _write_salvage_summary
from coding_review_agent_loop.salvage import (
    SalvageContext,
    capture_salvage_artifacts,
    post_salvage_comment,
)
from agent_loop_helpers import (
    make_config,
    structured_plan_review,
    structured_plan_revision,
)
import coding_review_agent_loop.validated_agent as validated_agent_module


def _salvage_dirs(config):
    salvage_root = config.log_dir / "salvage"
    if not salvage_root.exists():
        return []
    return sorted(path for path in salvage_root.iterdir() if path.is_dir())


def test_failed_issue_implementation_with_diff_writes_salvage_artifacts(tmp_path):
    patch_text = (
        "diff --git a/src/coding_review_agent_loop/cli.py b/src/coding_review_agent_loop/cli.py\n"
        "--- a/src/coding_review_agent_loop/cli.py\n"
        "+++ b/src/coding_review_agent_loop/cli.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    runner = FakeRunner(
        claude_outputs=[("quota exceeded; reset in 10m", 1)],
        post_agent_git_status=" M src/coding_review_agent_loop/cli.py\n?? scratch-note.md\n",
        post_agent_git_diff=patch_text,
        post_agent_git_diff_stat=" src/coding_review_agent_loop/cli.py | 2 +-\n",
        post_agent_git_diff_check="src/coding_review_agent_loop/cli.py:1: trailing whitespace.\n",
        post_agent_git_diff_check_returncode=2,
    )
    config = make_config(tmp_path)

    with pytest.raises(QuotaResetExceededError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config)

    message = str(excinfo.value)
    assert "Implementation salvage artifacts were written to" in message
    assert "patch:" in message
    assert "No non-empty public response file was produced at expected path" in message
    assert "no result was recorded because the agent command exited with quota/session-limit status" in message
    salvage_dir = _salvage_dirs(config)[0]
    assert str(salvage_dir / "salvage-summary.md") in message
    assert (salvage_dir / "partial.patch").read_text(encoding="utf-8") == patch_text
    assert "?? scratch-note.md" in (salvage_dir / "changed-files.txt").read_text(
        encoding="utf-8"
    )
    assert "2 +-" in (salvage_dir / "diff-stat.txt").read_text(encoding="utf-8")
    assert "trailing whitespace" in (salvage_dir / "diff-check.txt").read_text(
        encoding="utf-8"
    )

    summary = (salvage_dir / "salvage-summary.md").read_text(encoding="utf-8")
    assert "No\nsuccessful response, review result, or pull request should be inferred" in summary
    assert "Public response file: missing" in summary
    assert "Required marker status: missing or invalid" in summary
    assert "Untracked files appear in `changed-files.txt`" in summary

    metadata = json.loads((salvage_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["repo"] == "OWNER/REPO"
    assert metadata["issue_number"] == 56
    assert metadata["scope"] == "issue-implementation"
    assert metadata["agent"] == "claude"
    assert metadata["failure_category"] == "transient"
    assert metadata["response_file_missing"] is True
    assert metadata["diff_check_returncode"] == 2


def test_failed_issue_implementation_salvage_oserror_preserves_original_failure(
    tmp_path, capsys, monkeypatch
):
    runner = FakeRunner(
        claude_outputs=[("quota exceeded; reset in 10m", 1)],
        post_agent_git_diff="diff --git a/file.txt b/file.txt\n",
    )
    config = make_config(tmp_path, quiet=False)

    def fail_capture(*args, **kwargs):
        raise OSError("simulated disk full")

    monkeypatch.setattr(agent_failure, "capture_salvage_artifacts", fail_capture)

    with pytest.raises(QuotaResetExceededError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config)

    message = str(excinfo.value)
    assert "quota exhausted" in message
    assert "Rerun when quota resets" in message
    assert "Implementation salvage was attempted for issue implementation" in message
    assert "capture failed (simulated disk full)" in message
    assert "preserving the original agent failure" in message
    assert _salvage_dirs(config) == []
    assert "salvage capture failed (simulated disk full)" in capsys.readouterr().err


def test_failed_issue_implementation_without_diff_writes_no_salvage_patch(tmp_path):
    runner = FakeRunner(
        claude_outputs=["Created local notes but forgot the required marker."],
        post_agent_git_status=" M src/coding_review_agent_loop/cli.py\n",
        post_agent_git_diff="",
    )
    config = make_config(tmp_path, agent_max_retries=0)

    with pytest.raises(AgentLoopError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config)

    message = str(excinfo.value)
    assert "Implementation salvage was attempted for issue implementation" in message
    assert "no tracked/staged `git diff HEAD --binary` existed" in message
    assert "no patch artifacts were created" in message
    assert _salvage_dirs(config) == []


def test_failed_issue_implementation_with_untracked_only_diff_reports_untracked_only(tmp_path):
    runner = FakeRunner(
        claude_outputs=["Created local notes but forgot the required marker."],
        post_agent_git_status="?? scratch-note.md\n",
        post_agent_git_diff="",
    )
    config = make_config(tmp_path, agent_max_retries=0)

    with pytest.raises(AgentLoopError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config)

    message = str(excinfo.value)
    assert "only untracked files were present" in message
    assert "no tracked/staged `git diff HEAD --binary` existed" in message
    assert _salvage_dirs(config) == []


def test_plan_revision_quota_failure_reports_response_file_without_recording(tmp_path):
    valid_revision = structured_plan_revision(
        summary="Revised plan with a regression test.",
        prior_plan_item_dispositions=[
            {"item_id": "item-1", "disposition": "resolved", "note": "Added the test step."}
        ],
        plan_steps=["Add the regression test.", "Run the focused suite."],
    )
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            ("quota exceeded; reset in 10m", 1),
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Missing a regression test.",
                blocking_plan_issues=["Missing a regression test."],
            )
        ],
        # A failed exit may only be salvaged when its current-attempt artifact
        # passes the normal revision validator.
        public_response_outputs=["", "", "partial revision"],
    )
    config = make_config(tmp_path, agent_max_retries=0)

    with pytest.raises(QuotaResetExceededError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    message = str(excinfo.value)
    assert "No implementation salvage was attempted because this was plan revision" in message
    assert "not a mutating implementation attempt" in message
    assert "A public response file exists at" in message
    assert "no result was recorded because the agent command exited with quota/session-limit status" in message
    assert "Revised plan with a regression test" not in "".join(runner.comments)


def test_plan_review_failure_without_response_file_reports_non_mutating_skip(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=["Plan review without the required structured response."],
    )
    config = make_config(tmp_path, agent_max_retries=0)

    with pytest.raises(AgentLoopError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    message = str(excinfo.value)
    assert "No implementation salvage was attempted because this was plan review" in message
    assert "not a mutating implementation attempt" in message
    assert "No non-empty public response file was produced at expected path" in message
    assert "no result was recorded because the public response failed validation" in message


def test_issue_implementation_rerun_prompt_includes_latest_salvage_summary(tmp_path):
    config = make_config(tmp_path)
    _write_salvage_summary(
        config,
        name="old",
        summary="old failed attempt summary",
        created_at_ns=1,
    )
    _write_salvage_summary(
        config,
        name="new",
        summary="new failed attempt summary\nPartial patch: `/tmp/new.patch`",
        created_at_ns=2,
    )
    runner = FakeRunner(
        claude_outputs=[
            "Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    coder_prompt = next(
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] and "Fix GitHub issue #56" in cmd[-1]
    )
    assert "Previous implementation attempt salvage:" in coder_prompt
    assert "new failed attempt summary" in coder_prompt
    assert "old failed attempt summary" not in coder_prompt
    assert "Do not auto-apply the patch" in coder_prompt
    assert "cherry-pick or ignore it" in coder_prompt
    assert "selectively" in coder_prompt


def test_failed_issue_implementation_posts_github_salvage_comment(tmp_path):
    runner = FakeRunner(
        claude_outputs=[("quota exceeded; reset in 10m", 1)],
        post_agent_git_status=" M src/coding_review_agent_loop/cli.py\n",
        post_agent_git_diff=(
            "diff --git a/src/coding_review_agent_loop/cli.py b/src/coding_review_agent_loop/cli.py\n"
            "--- a/src/coding_review_agent_loop/cli.py\n"
            "+++ b/src/coding_review_agent_loop/cli.py\n"
            "@@ -1 +1 @@\n-old\n+new\n"
        ),
        post_agent_git_diff_stat=" src/coding_review_agent_loop/cli.py | 2 +-\n",
    )
    config = make_config(tmp_path)

    with pytest.raises(QuotaResetExceededError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config)

    message = str(excinfo.value)
    assert "A GitHub salvage comment was posted to issue #56." in message
    assert len(runner.issue_comments) == 1
    posted_body = runner.issue_comments[0]["body"]
    assert "<!-- AGENT_SALVAGE:" in posted_body
    assert "Implementation salvage breadcrumb" in posted_body


def test_failed_issue_implementation_salvage_comment_disabled_posts_nothing(tmp_path):
    runner = FakeRunner(
        claude_outputs=[("quota exceeded; reset in 10m", 1)],
        post_agent_git_status=" M src/coding_review_agent_loop/cli.py\n",
        post_agent_git_diff=(
            "diff --git a/src/coding_review_agent_loop/cli.py b/src/coding_review_agent_loop/cli.py\n"
            "--- a/src/coding_review_agent_loop/cli.py\n"
            "+++ b/src/coding_review_agent_loop/cli.py\n"
            "@@ -1 +1 @@\n-old\n+new\n"
        ),
    )
    config = make_config(tmp_path, salvage_comments=False)

    with pytest.raises(QuotaResetExceededError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config)

    message = str(excinfo.value)
    assert "No GitHub salvage comment was posted." in message
    assert runner.issue_comments == []


def _build_remote_salvage_comment(
    tmp_path,
    *,
    issue_number=56,
    scope="issue-implementation",
    approved_plan_hash_value=None,
    patch_text=None,
    failure_reason="remote-only failure summary",
    patch_max_bytes=20000,
):
    patch_text = patch_text or (
        "diff --git a/file.txt b/file.txt\n--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new\n"
    )
    checkout = tmp_path / "remote-source-checkout"
    checkout.mkdir(parents=True, exist_ok=True)
    seed_runner = FakeRunner(
        git_diff=patch_text,
        git_status=" M file.txt\n",
        git_diff_stat=" file.txt | 2 +-\n",
    )
    context = SalvageContext(
        repo="OWNER/REPO",
        issue_number=issue_number,
        scope=scope,
        agent="claude",
        approved_plan_hash=approved_plan_hash_value,
    )
    artifacts = capture_salvage_artifacts(
        seed_runner,
        checkout=checkout,
        log_dir=tmp_path / "remote-source-logs",
        context=context,
        failure_category="transient",
        failure_reason=failure_reason,
        required_marker="<!-- AGENT_PR: <number> -->",
        result=None,
    )
    post_config = make_config(tmp_path, salvage_comment_patch_max_bytes=patch_max_bytes)
    posted = post_salvage_comment(
        seed_runner,
        config=post_config,
        artifacts=artifacts,
        context=context,
        failure_category="transient",
        failure_reason=failure_reason,
    )
    assert posted
    return seed_runner.issue_comments[-1]


def test_issue_implementation_rerun_discovers_remote_salvage_when_local_log_dir_is_empty(tmp_path):
    remote_comment = _build_remote_salvage_comment(
        tmp_path,
        issue_number=56,
        scope="issue-implementation",
        failure_reason="remote-only failure summary",
    )
    runner = FakeRunner(
        issue_comments=[remote_comment],
        claude_outputs=[
            "Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    coder_prompt = next(
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] and "Fix GitHub issue #56" in cmd[-1]
    )
    assert "Previous implementation attempt salvage:" in coder_prompt
    assert "recovered from a GitHub issue comment" in coder_prompt
    assert "remote-only failure summary" in coder_prompt
    assert "```diff" in coder_prompt
    assert "-old" in coder_prompt and "+new" in coder_prompt
    # The raw AGENT_SALVAGE breadcrumb comment must not also be rendered
    # verbatim via the ordinary issue-context comment block (#507 follow-up):
    # it is consumed only through the parsed salvage_summary injection above.
    assert "AGENT_SALVAGE" not in coder_prompt
    assert "Implementation salvage breadcrumb" not in coder_prompt


def test_issue_implementation_rerun_remote_salvage_with_omitted_patch_renders_local_only_note(tmp_path):
    oversized_patch = (
        "diff --git a/file.txt b/file.txt\n--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new\n"
    ) + ("+padding\n" * 5000)
    remote_comment = _build_remote_salvage_comment(
        tmp_path,
        issue_number=56,
        scope="issue-implementation",
        patch_text=oversized_patch,
        patch_max_bytes=100,
        failure_reason="remote failure with an oversized patch",
    )
    runner = FakeRunner(
        issue_comments=[remote_comment],
        claude_outputs=[
            "Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    coder_prompt = next(
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] and "Fix GitHub issue #56" in cmd[-1]
    )
    assert "recovered from a GitHub issue comment" in coder_prompt
    assert "local-only" in coder_prompt
    assert "```diff" not in coder_prompt


def test_issue_implementation_rerun_ignores_remote_salvage_for_a_different_issue(tmp_path):
    remote_comment = _build_remote_salvage_comment(
        tmp_path,
        issue_number=999,
        scope="issue-implementation",
        failure_reason="unrelated issue failure",
    )
    runner = FakeRunner(
        issue_comments=[remote_comment],
        claude_outputs=[
            "Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    coder_prompt = next(
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] and "Fix GitHub issue #56" in cmd[-1]
    )
    assert "Previous implementation attempt salvage:" not in coder_prompt


def test_approved_plan_implementation_rerun_discovers_remote_salvage_with_matching_plan_hash(tmp_path):
    approved_plan = "Plan:\n- Do the thing."
    plan_hash = approved_plan_hash(approved_plan)
    remote_comment = _build_remote_salvage_comment(
        tmp_path,
        issue_number=56,
        scope="approved-plan-implementation",
        approved_plan_hash_value=plan_hash,
        failure_reason="remote approved-plan failure",
    )
    runner = FakeRunner(
        issue_comments=[remote_comment],
        claude_outputs=[
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path)
    issue_context = get_issue_context(runner, config=config, issue_number=56)
    usage_context = validated_agent_module._new_usage_context(config)

    result = issue_implementation._implement_approved_issue(
        runner,
        issue_number=56,
        approved_plan=approved_plan,
        config=config,
        memory=None,
        issue_context=issue_context,
        coder_session_id=None,
        usage_context=usage_context,
    )

    assert result == 0
    coder_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "Previous implementation attempt salvage:" in coder_prompt
    assert "recovered from a GitHub issue comment" in coder_prompt
    assert "remote approved-plan failure" in coder_prompt


@pytest.mark.parametrize("terminal_marker, expected_state", [
    ("<!-- AGENT_STATE: blocking -->", "blocking"),
    ("<!-- AGENT_CLARIFY -->", "clarification"),
])
def test_approved_plan_no_pr_terminal_result_bypasses_human_requirements_and_pr_checks(
    tmp_path, terminal_marker, expected_state
):
    runner = FakeRunner(
        issue_payload={"body": "Keep the public API stable."},
        claude_outputs=[f"Cannot proceed.\n<!-- AGENT_PR: 0 -->\n{terminal_marker}"],
    )
    config = make_config(tmp_path)
    issue_context = get_issue_context(runner, config=config, issue_number=56)

    with pytest.raises(AgentLoopError, match=f"implementation is {expected_state}"):
        issue_implementation._implement_approved_issue(
            runner,
            issue_number=56,
            approved_plan="Plan:\n- Do the thing.",
            config=config,
            memory=None,
            issue_context=issue_context,
            coder_session_id=None,
            usage_context=validated_agent_module._new_usage_context(config),
        )

    assert not any(cmd[:3] == ["gh", "pr", "view"] for cmd, _cwd in runner.commands)
    assert not any(cmd[:1] == ["codex"] for cmd, _cwd in runner.commands)
    # A genuine coder-declared no-PR terminal is posted to the issue like a
    # PR-success implementation result already is (#588).
    if expected_state == "blocking":
        assert runner.comments[0].startswith("## Issue implementation")
        assert "Cannot proceed." in runner.comments[0]
        assert "No pull request was accepted for handoff." in runner.comments[0]
    else:
        assert runner.comments == [
            f"Cannot proceed.\n<!-- AGENT_PR: 0 -->\n{terminal_marker}\n-- Anthropic Claude: unknown model (medium)"
        ]


def test_approved_plan_implementation_rerun_ignores_remote_salvage_on_plan_hash_mismatch(tmp_path):
    approved_plan = "Plan:\n- Do the current thing."
    stale_plan_hash = approved_plan_hash("Plan:\n- Do the old thing.")
    remote_comment = _build_remote_salvage_comment(
        tmp_path,
        issue_number=56,
        scope="approved-plan-implementation",
        approved_plan_hash_value=stale_plan_hash,
        failure_reason="stale plan failure",
    )
    runner = FakeRunner(
        issue_comments=[remote_comment],
        claude_outputs=[
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path)
    issue_context = get_issue_context(runner, config=config, issue_number=56)
    usage_context = validated_agent_module._new_usage_context(config)

    result = issue_implementation._implement_approved_issue(
        runner,
        issue_number=56,
        approved_plan=approved_plan,
        config=config,
        memory=None,
        issue_context=issue_context,
        coder_session_id=None,
        usage_context=usage_context,
    )

    assert result == 0
    coder_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "Previous implementation attempt salvage:" not in coder_prompt


TERMINAL_PATCH = "diff --git a/file.txt b/file.txt\n--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new\n"


def _run_terminal(runner, config, approved):
    if not approved:
        return run_issue_loop(runner, issue_number=56, config=config)
    return issue_implementation._implement_approved_issue(
        runner, issue_number=56, approved_plan="Plan:\n- Do the thing.",
        config=config, memory=None,
        issue_context=get_issue_context(runner, config=config, issue_number=56),
        coder_session_id=None, usage_context=validated_agent_module._new_usage_context(config),
    )


@pytest.mark.parametrize("approved", [False, True])
@pytest.mark.parametrize("outcome", ["structured", "blocking", "clarification"])
def test_valid_terminal_issue_preserves_work_and_rerun_context(tmp_path, approved, outcome):
    from agent_loop_helpers import structured_issue_implementation
    from coding_review_agent_loop.salvage import latest_salvage_context
    text = (structured_issue_implementation(pr_number=None, summary="Required test timed out.")
            if outcome == "structured" else "Cannot proceed.\n" + (
                "<!-- AGENT_CLARIFY -->" if outcome == "clarification" else "<!-- AGENT_STATE: blocking -->"))
    runner = FakeRunner(claude_outputs=[text], post_agent_git_diff=TERMINAL_PATCH,
                        post_agent_git_status="M  file.txt\n?? ignored.txt\n")
    config = make_config(tmp_path)
    with pytest.raises(AgentLoopError):
        _run_terminal(runner, config, approved)
    directory, = _salvage_dirs(config)
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["failure_category"] == "valid-terminal-stop"
    assert metadata["scope"] == ("approved-plan-implementation" if approved else "issue-implementation")
    assert (directory / "partial.patch").read_text() == TERMINAL_PATCH
    assert "valid terminal response" in (directory / "salvage-summary.md").read_text()
    assert "failed before" not in (directory / "salvage-summary.md").read_text()
    assert "valid terminal response" in runner.issue_comments[0]["body"]
    kwargs = dict(repo=config.repo, issue_number=56, scope=metadata["scope"],
                  approved_plan_hash=metadata["approved_plan_hash"])
    # Stable absolute directory permits local discovery; a new default directory
    # needs a remote breadcrumb and must not search other runs.
    local = latest_salvage_context(config.log_dir, [], **kwargs)
    assert local and str(directory) in local
    from coding_review_agent_loop.github import IssueComment
    comment = IssueComment(author="test", created_at="2026-11-01T00:00:00Z", body=runner.issue_comments[0]["body"])
    remote = latest_salvage_context(tmp_path / "next-run", comments=[comment], **kwargs)
    assert remote and "valid terminal response" in remote
    assert latest_salvage_context(tmp_path / "next-run", [], **kwargs) is None
    assert not any(cmd[:3] == ["gh", "pr", "view"] for cmd, _ in runner.commands)
    assert not any(cmd[:1] == ["codex"] for cmd, _ in runner.commands)


@pytest.mark.parametrize("approved", [False, True])
@pytest.mark.parametrize("policy", ["disabled", "dry-run", "post-error", "capture-error", "clean", "untracked"])
def test_terminal_issue_degradation_preserves_reason(tmp_path, monkeypatch, capsys, approved, policy):
    import coding_review_agent_loop.salvage as salvage
    from agent_loop_helpers import structured_issue_implementation
    config = make_config(tmp_path, quiet=False, salvage_comments=policy != "disabled", dry_run=policy == "dry-run")
    runner = FakeRunner(claude_outputs=[structured_issue_implementation(pr_number=None)],
                        post_agent_git_diff="" if policy in {"clean", "untracked"} else TERMINAL_PATCH,
                        post_agent_git_status="?? ignored.txt\n" if policy == "untracked" else "")
    def fail(*args, **kwargs):
        raise OSError("disk or network unavailable")
    if policy == "capture-error":
        monkeypatch.setattr(salvage, "capture_salvage_artifacts", fail)
    if policy == "post-error":
        monkeypatch.setattr(salvage, "post_salvage_comment", fail)
    with pytest.raises(AgentLoopError, match="implementation is blocking") as stop:
        _run_terminal(runner, config, approved)
    message = str(stop.value) + capsys.readouterr().err
    if policy == "capture-error":
        assert "capture failed" in message
    elif policy in {"clean", "untracked"}:
        assert "No eligible tracked/staged diff" in message
        assert "Untracked-only contents" in message
        assert not _salvage_dirs(config)
    else:
        assert "partial.patch" in message
        assert "only recovery route" in message
        assert "stable absolute --log-dir" in message
        assert _salvage_dirs(config)
    assert not any("Implementation salvage breadcrumb" in c["body"] for c in runner.issue_comments)


@pytest.mark.parametrize("approved", [False, True])
@pytest.mark.parametrize("discovery", ["stable-local", "remote", "disabled", "post-error", "dry-run", "stale-plan"])
def test_terminal_issue_rerun_prompt_discovery(tmp_path, monkeypatch, approved, discovery):
    from dataclasses import replace
    import coding_review_agent_loop.salvage as salvage
    from agent_loop_helpers import structured_issue_implementation
    config = make_config(tmp_path, salvage_comments=discovery != "disabled", dry_run=discovery == "dry-run")
    text = structured_issue_implementation(pr_number=None, summary="Saved terminal work from prior run.")
    first = FakeRunner(claude_outputs=[text], post_agent_git_diff=TERMINAL_PATCH)
    if discovery == "post-error":
        monkeypatch.setattr(salvage, "post_salvage_comment", lambda *a, **kw: False)
    with pytest.raises(AgentLoopError):
        _run_terminal(first, config, approved)
    if discovery != "stable-local":
        config = replace(config, log_dir=tmp_path / "different-run-logs")
    comments = first.issue_comments
    if discovery == "stable-local":
        comments = []
    if discovery == "stale-plan":
        # Use a direct-issue capture on an approved-plan restart or vice versa;
        # the existing exact hash tests additionally cover changed plan content.
        approved = not approved
    second = FakeRunner(claude_outputs=[structured_issue_implementation(pr_number=None)], issue_comments=comments)
    with pytest.raises(AgentLoopError):
        _run_terminal(second, config, approved)
    prompt = next(cmd[-1] for cmd, _ in second.commands if cmd[:1] == ["claude"])
    expected = discovery in {"stable-local", "remote"}
    assert ("Previous implementation attempt salvage:" in prompt) is expected
    if expected:
        assert "Saved terminal work from prior run." in prompt
        assert "valid terminal response" in prompt
        assert "The previous attempt failed." not in prompt
    else:
        assert "valid terminal response without an accepted PR handoff" not in prompt
