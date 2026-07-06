import coding_review_agent_loop.salvage as salvage_module
from coding_review_agent_loop.github import IssueComment
from coding_review_agent_loop.salvage import (
    RemoteSalvageRecord,
    SalvageContext,
    capture_salvage_artifacts,
    find_latest_remote_salvage,
    latest_salvage_context,
    post_salvage_comment,
    render_remote_salvage_summary,
)
from agent_loop_helpers import FakeRunner, make_config

CLEAN_PATCH = (
    "diff --git a/file.txt b/file.txt\n"
    "--- a/file.txt\n"
    "+++ b/file.txt\n"
    "@@ -1 +1 @@\n"
    "-old\n"
    "+new\n"
)


def _capture(tmp_path, *, patch_text=CLEAN_PATCH, context=None, subdir="checkout"):
    checkout = tmp_path / subdir
    checkout.mkdir(parents=True, exist_ok=True)
    runner = FakeRunner(
        git_diff=patch_text,
        git_status=" M file.txt\n",
        git_diff_stat=" file.txt | 2 +-\n",
        git_diff_check="",
    )
    context = context or SalvageContext(
        repo="OWNER/REPO",
        issue_number=56,
        scope="issue-implementation",
        agent="claude",
        run_id="run-1",
        approved_plan_hash="hash-1",
    )
    artifacts = capture_salvage_artifacts(
        runner,
        checkout=checkout,
        log_dir=tmp_path / "logs",
        context=context,
        failure_category="transient",
        failure_reason="agent failed",
        required_marker="<!-- AGENT_PR: <number> -->",
        result=None,
    )
    assert artifacts is not None
    return runner, artifacts, context


def _comments_from_runner(runner):
    return [
        IssueComment(author="bot", created_at=raw["createdAt"], body=raw["body"])
        for raw in runner.issue_comments
    ]


def test_post_salvage_comment_round_trips_metadata_fields(tmp_path):
    runner, artifacts, context = _capture(tmp_path)
    config = make_config(tmp_path)

    posted = post_salvage_comment(
        runner,
        config=config,
        artifacts=artifacts,
        context=context,
        failure_category="transient",
        failure_reason="agent failed",
    )

    assert posted is True
    assert len(runner.issue_comments) == 1
    body = runner.issue_comments[0]["body"]
    assert "<!-- AGENT_SALVAGE:" in body

    record = find_latest_remote_salvage(
        _comments_from_runner(runner),
        repo="OWNER/REPO",
        issue_number=56,
        scope="issue-implementation",
        approved_plan_hash="hash-1",
    )
    assert record is not None
    assert record.repo == "OWNER/REPO"
    assert record.issue_number == 56
    assert record.scope == "issue-implementation"
    assert record.agent == "claude"
    assert record.run_id == "run-1"
    assert record.approved_plan_hash == "hash-1"
    assert record.failure_category == "transient"
    assert record.failure_reason == "agent failed"
    assert record.patch_included is True
    assert record.patch_text == CLEAN_PATCH


def test_patch_omitted_when_over_size_limit(tmp_path):
    oversized_patch = CLEAN_PATCH + ("+padding line\n" * 5000)
    runner, artifacts, context = _capture(tmp_path, patch_text=oversized_patch)
    config = make_config(tmp_path, salvage_comment_patch_max_bytes=100)

    post_salvage_comment(
        runner,
        config=config,
        artifacts=artifacts,
        context=context,
        failure_category="transient",
        failure_reason="agent failed",
    )

    record = find_latest_remote_salvage(
        _comments_from_runner(runner),
        repo="OWNER/REPO",
        issue_number=56,
        scope="issue-implementation",
        approved_plan_hash="hash-1",
    )
    assert record.patch_exists is True
    assert record.patch_included is False
    assert record.patch_text is None
    assert "local-only" in runner.issue_comments[0]["body"]


def test_patch_omitted_when_binary(tmp_path):
    binary_patch = CLEAN_PATCH + "GIT binary patch\n1234 bytes of nonsense\n"
    runner, artifacts, context = _capture(tmp_path, patch_text=binary_patch)
    config = make_config(tmp_path)

    post_salvage_comment(
        runner,
        config=config,
        artifacts=artifacts,
        context=context,
        failure_category="transient",
        failure_reason="agent failed",
    )

    record = find_latest_remote_salvage(
        _comments_from_runner(runner),
        repo="OWNER/REPO",
        issue_number=56,
        scope="issue-implementation",
        approved_plan_hash="hash-1",
    )
    assert record.patch_included is False
    assert record.patch_text is None


def test_patch_omitted_when_secret_pattern_detected(tmp_path):
    secret_patch = CLEAN_PATCH + "+aws_key = AKIAABCDEFGHIJKLMNOP\n"
    runner, artifacts, context = _capture(tmp_path, patch_text=secret_patch)
    config = make_config(tmp_path)

    post_salvage_comment(
        runner,
        config=config,
        artifacts=artifacts,
        context=context,
        failure_category="transient",
        failure_reason="agent failed",
    )

    record = find_latest_remote_salvage(
        _comments_from_runner(runner),
        repo="OWNER/REPO",
        issue_number=56,
        scope="issue-implementation",
        approved_plan_hash="hash-1",
    )
    assert record.patch_included is False
    assert record.patch_text is None


def test_post_salvage_comment_skips_ineligible_scopes(tmp_path):
    runner, artifacts, context = _capture(
        tmp_path,
        context=SalvageContext(
            repo="OWNER/REPO",
            issue_number=56,
            scope="pr-followup",
            agent="claude",
        ),
    )
    config = make_config(tmp_path)

    posted = post_salvage_comment(
        runner,
        config=config,
        artifacts=artifacts,
        context=context,
        failure_category="transient",
        failure_reason="agent failed",
    )

    assert posted is False
    assert runner.issue_comments == []


def test_post_salvage_comment_skips_when_disabled_or_dry_run(tmp_path):
    runner, artifacts, context = _capture(tmp_path)

    disabled_config = make_config(tmp_path, salvage_comments=False)
    assert (
        post_salvage_comment(
            runner,
            config=disabled_config,
            artifacts=artifacts,
            context=context,
            failure_category="transient",
            failure_reason="agent failed",
        )
        is False
    )

    dry_run_config = make_config(tmp_path, dry_run=True)
    assert (
        post_salvage_comment(
            runner,
            config=dry_run_config,
            artifacts=artifacts,
            context=context,
            failure_category="transient",
            failure_reason="agent failed",
        )
        is False
    )
    assert runner.issue_comments == []


def test_post_salvage_comment_swallows_posting_failure(tmp_path, monkeypatch, capsys):
    runner, artifacts, context = _capture(tmp_path)
    config = make_config(tmp_path, quiet=False)

    def boom(*args, **kwargs):
        raise salvage_module.AgentLoopError("gh unavailable")

    monkeypatch.setattr(salvage_module, "post_issue_comment", boom)

    posted = post_salvage_comment(
        runner,
        config=config,
        artifacts=artifacts,
        context=context,
        failure_category="transient",
        failure_reason="agent failed",
    )

    assert posted is False
    assert "salvage comment posting failed" in capsys.readouterr().err


def test_render_remote_salvage_summary_includes_patch_verbatim_when_included():
    record = RemoteSalvageRecord(
        created_at_ns=1,
        repo="OWNER/REPO",
        issue_number=56,
        scope="issue-implementation",
        agent="claude",
        run_id="run-1",
        approved_plan_hash="hash-1",
        failure_category="transient",
        failure_reason="agent failed",
        changed_files=" M file.txt",
        diff_stat=" file.txt | 2 +-",
        local_directory="/tmp/some/old/workdir/salvage/run-1-claude-issue-implementation",
        patch_exists=True,
        patch_included=True,
        patch_text=CLEAN_PATCH,
    )

    summary = render_remote_salvage_summary(record)

    assert "recovered from a GitHub issue comment" in summary
    assert CLEAN_PATCH in summary
    assert "local-only" not in summary


def test_render_remote_salvage_summary_omits_patch_block_when_not_included():
    record = RemoteSalvageRecord(
        created_at_ns=1,
        repo="OWNER/REPO",
        issue_number=56,
        scope="issue-implementation",
        agent="claude",
        run_id="run-1",
        approved_plan_hash=None,
        failure_category="transient",
        failure_reason="agent failed",
        changed_files=" M file.txt",
        diff_stat=" file.txt | 2 +-",
        local_directory="/tmp/some/old/workdir/salvage/run-1-claude-issue-implementation",
        patch_exists=True,
        patch_included=False,
        patch_text=None,
    )

    summary = render_remote_salvage_summary(record)

    assert "local-only" in summary
    assert "```diff" not in summary


def test_latest_salvage_context_merge_prefers_newer_remote(tmp_path, monkeypatch):
    counter = iter([100, 200])
    monkeypatch.setattr(salvage_module.time, "time_ns", lambda: next(counter))

    runner, artifacts, context = _capture(tmp_path)
    config = make_config(tmp_path)
    post_salvage_comment(
        runner,
        config=config,
        artifacts=artifacts,
        context=context,
        failure_category="transient",
        failure_reason="remote is newer",
    )

    summary = latest_salvage_context(
        tmp_path / "logs",
        _comments_from_runner(runner),
        repo="OWNER/REPO",
        issue_number=56,
        scope="issue-implementation",
        approved_plan_hash="hash-1",
    )
    assert summary is not None
    assert "recovered from a GitHub issue comment" in summary
    assert "remote is newer" in summary


def test_latest_salvage_context_merge_ties_prefer_local(tmp_path, monkeypatch):
    monkeypatch.setattr(salvage_module.time, "time_ns", lambda: 100)

    runner, artifacts, context = _capture(tmp_path)
    config = make_config(tmp_path)
    post_salvage_comment(
        runner,
        config=config,
        artifacts=artifacts,
        context=context,
        failure_category="transient",
        failure_reason="remote failure reason",
    )

    summary = latest_salvage_context(
        tmp_path / "logs",
        _comments_from_runner(runner),
        repo="OWNER/REPO",
        issue_number=56,
        scope="issue-implementation",
        approved_plan_hash="hash-1",
    )
    assert summary is not None
    assert "recovered from a GitHub issue comment" not in summary
    assert "The claude agent failed before producing a valid public response" in summary


def test_find_latest_remote_salvage_filters_plan_hash_scope_repo_issue(tmp_path):
    runner, artifacts, context = _capture(tmp_path)
    config = make_config(tmp_path)
    post_salvage_comment(
        runner,
        config=config,
        artifacts=artifacts,
        context=context,
        failure_category="transient",
        failure_reason="agent failed",
    )
    comments = _comments_from_runner(runner)

    assert find_latest_remote_salvage(
        comments, repo="OWNER/OTHER", issue_number=56, scope="issue-implementation", approved_plan_hash="hash-1"
    ) is None
    assert find_latest_remote_salvage(
        comments, repo="OWNER/REPO", issue_number=99, scope="issue-implementation", approved_plan_hash="hash-1"
    ) is None
    assert find_latest_remote_salvage(
        comments, repo="OWNER/REPO", issue_number=56, scope="approved-plan-implementation", approved_plan_hash="hash-1"
    ) is None
    assert find_latest_remote_salvage(
        comments, repo="OWNER/REPO", issue_number=56, scope="issue-implementation", approved_plan_hash="different-hash"
    ) is None
    assert find_latest_remote_salvage(
        comments, repo="OWNER/REPO", issue_number=56, scope="issue-implementation", approved_plan_hash="hash-1"
    ) is not None


def test_find_latest_remote_salvage_skips_malformed_markers():
    comments = [
        IssueComment(author="bot", created_at="2026-01-01T00:00:00Z", body="<!-- AGENT_SALVAGE: not-valid-base64!! -->"),
        IssueComment(author="bot", created_at="2026-01-01T00:00:01Z", body="<!-- AGENT_SALVAGE: aGVsbG8= -->"),
    ]

    assert find_latest_remote_salvage(
        comments, repo="OWNER/REPO", issue_number=56, scope="issue-implementation", approved_plan_hash=None
    ) is None
