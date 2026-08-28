"""Tests for GitHub mergeability probing and its full-board watcher use (#606)."""
import json

from coding_review_agent_loop.comment_rendering import _format_unresolved_item_label
from coding_review_agent_loop.github import (
    PullRequestMergeability,
    PullRequestMetadata,
    get_pr_mergeability,
    watch_pr_checks,
)
from coding_review_agent_loop.protocol import UnresolvedReviewItem, validate_structured_coder_followup
from coding_review_agent_loop.runner import CommandResult, Runner
from coding_review_agent_loop.unresolved_items import (
    MERGE_CONFLICT_ITEM_ID,
    _reconcile_merge_conflict_item,
    _validate_structured_coder_followup_items,
)

from agent_loop_helpers import make_config, structured_coder_followup


class _StubMergeabilityRunner(Runner):
    def __init__(self, *, mergeability_responses=None, check_runs_responses=None):
        super().__init__(dry_run=False)
        self.mergeability_responses = list(mergeability_responses or [])
        self.check_runs_responses = list(
            check_runs_responses
            or [{"check_runs": [{"name": "test", "status": "completed", "conclusion": "success"}]}]
        )
        self.sleep_calls: list[str] = []
        self.commands: list[list[str]] = []

    def run(self, args, *, cwd, input_text=None, check=True, env=None):
        cmd = [str(a) for a in args]
        self.commands.append(cmd)
        if cmd[:1] == ["sleep"]:
            self.sleep_calls.append(cmd[1])
            return CommandResult(cmd, cwd, "", "", 0)
        if (
            cmd[:3] == ["gh", "pr", "view"]
            and "--json" in cmd
            and cmd[cmd.index("--json") + 1] == "mergeable,mergeStateStatus,headRefOid,baseRefName"
        ):
            if not self.mergeability_responses:
                raise AssertionError("no more scripted mergeability responses")
            response = self.mergeability_responses.pop(0)
            payload, returncode = response if isinstance(response, tuple) else (response, 0)
            stdout = payload if isinstance(payload, str) else json.dumps(payload)
            return CommandResult(cmd, cwd, stdout, "", returncode)
        if cmd[:3] == ["gh", "pr", "view"] and "--jq" in cmd and ".headRefOid" in cmd:
            return CommandResult(cmd, cwd, "headsha1\n", "", 0)
        if cmd[:2] == ["gh", "api"] and cmd[2].endswith("/check-runs"):
            payload = self.check_runs_responses.pop(0) if self.check_runs_responses else {"check_runs": []}
            return CommandResult(cmd, cwd, json.dumps(payload), "", 0)
        if cmd[:2] == ["gh", "api"] and cmd[2].endswith("/status"):
            return CommandResult(cmd, cwd, json.dumps({"statuses": []}), "", 0)
        if cmd[:2] == ["gh", "api"] and cmd[2].endswith("/protection/required_status_checks"):
            return CommandResult(cmd, cwd, json.dumps({"contexts": []}), "", 0)
        raise AssertionError(f"unexpected command: {cmd}")


def _metadata(**overrides):
    fields = dict(
        number=77,
        repo="OWNER/REPO",
        title="Title",
        head_branch="feature",
        base_branch="main",
        head_sha="headsha1",
        url="https://github.com/OWNER/REPO/pull/77",
    )
    fields.update(overrides)
    return PullRequestMetadata(**fields)


# --- get_pr_mergeability precedence -----------------------------------------


def test_conflicting_mergeable_is_conflicted(tmp_path):
    config = make_config(tmp_path)
    runner = _StubMergeabilityRunner(
        mergeability_responses=[
            {"mergeable": "CONFLICTING", "mergeStateStatus": "DIRTY", "headRefOid": "h1", "baseRefName": "main"}
        ]
    )

    result = get_pr_mergeability(runner, config=config, pr_number=77)

    assert result.state == "conflicted"
    assert result.mergeable_raw == "CONFLICTING"
    assert result.merge_state_raw == "DIRTY"
    assert result.head_sha == "h1"
    assert result.base_branch == "main"


def test_dirty_with_null_mergeable_is_conflicted(tmp_path):
    config = make_config(tmp_path)
    runner = _StubMergeabilityRunner(
        mergeability_responses=[
            {"mergeable": None, "mergeStateStatus": "DIRTY", "headRefOid": "h1", "baseRefName": "main"}
        ]
    )

    result = get_pr_mergeability(runner, config=config, pr_number=77)

    assert result.state == "conflicted"


def test_mergeable_state(tmp_path):
    config = make_config(tmp_path)
    runner = _StubMergeabilityRunner(
        mergeability_responses=[
            {"mergeable": "MERGEABLE", "mergeStateStatus": "CLEAN", "headRefOid": "h1", "baseRefName": "main"}
        ]
    )

    result = get_pr_mergeability(runner, config=config, pr_number=77)

    assert result.state == "mergeable"


def test_explicit_unknown_bounded_repoll_then_settles_unknown(tmp_path):
    config = make_config(tmp_path, mergeability_poll_attempts=3, mergeability_poll_interval_seconds=7)
    runner = _StubMergeabilityRunner(
        mergeability_responses=[
            {"mergeable": "UNKNOWN", "mergeStateStatus": "UNKNOWN"},
            {"mergeable": "UNKNOWN", "mergeStateStatus": "UNKNOWN"},
            {"mergeable": "UNKNOWN", "mergeStateStatus": "UNKNOWN"},
        ]
    )

    result = get_pr_mergeability(runner, config=config, pr_number=77)

    assert result.state == "unknown"
    assert result.mergeable_raw == "UNKNOWN"
    assert runner.sleep_calls == ["7", "7"]


def test_explicit_unknown_resolves_within_budget(tmp_path):
    config = make_config(tmp_path, mergeability_poll_attempts=3, mergeability_poll_interval_seconds=2)
    runner = _StubMergeabilityRunner(
        mergeability_responses=[
            {"mergeable": "UNKNOWN", "mergeStateStatus": "UNKNOWN"},
            {"mergeable": "MERGEABLE", "mergeStateStatus": "CLEAN", "headRefOid": "h1", "baseRefName": "main"},
        ]
    )

    result = get_pr_mergeability(runner, config=config, pr_number=77)

    assert result.state == "mergeable"
    assert runner.sleep_calls == ["2"]


def test_null_missing_mergeable_no_conflict_evidence_is_unknown_without_polling(tmp_path):
    config = make_config(tmp_path, mergeability_poll_attempts=3)
    runner = _StubMergeabilityRunner(
        mergeability_responses=[{"mergeable": None, "mergeStateStatus": "BEHIND"}]
    )

    result = get_pr_mergeability(runner, config=config, pr_number=77)

    assert result.state == "unknown"
    assert runner.sleep_calls == []


def test_gh_exit_failure_is_unknown(tmp_path):
    config = make_config(tmp_path, mergeability_poll_attempts=3)
    runner = _StubMergeabilityRunner(mergeability_responses=[({}, 1)])

    result = get_pr_mergeability(runner, config=config, pr_number=77)

    assert result.state == "unknown"
    assert runner.sleep_calls == []


def test_invalid_json_is_unknown(tmp_path):
    config = make_config(tmp_path, mergeability_poll_attempts=3)
    runner = _StubMergeabilityRunner(mergeability_responses=["not json"])

    result = get_pr_mergeability(runner, config=config, pr_number=77)

    assert result.state == "unknown"
    assert runner.sleep_calls == []


def test_dry_run_never_calls_gh(tmp_path):
    config = make_config(tmp_path, dry_run=True)
    runner = _StubMergeabilityRunner()

    result = get_pr_mergeability(runner, config=config, pr_number=77)

    assert result.state == "unknown"
    assert runner.commands == []


# --- full-board watcher mergeability outcomes -------------------------------


def test_watcher_returns_merge_conflict_before_check_board_poll(tmp_path):
    config = make_config(tmp_path, ci_timeout_seconds=120, ci_poll_interval_seconds=10)
    runner = _StubMergeabilityRunner(
        mergeability_responses=[
            {"mergeable": "CONFLICTING", "mergeStateStatus": "DIRTY", "headRefOid": "h1", "baseRefName": "main"}
        ],
    )

    outcome = watch_pr_checks(runner, config, 77, metadata=_metadata())

    assert outcome.status == "merge_conflict"
    assert outcome.mergeability is not None
    assert outcome.mergeability.state == "conflicted"
    # The conflict is checked before any check-run poll or sleep this attempt.
    assert runner.check_runs_responses  # untouched
    assert runner.sleep_calls == []


def test_watcher_passes_full_board_when_not_conflicted(tmp_path):
    config = make_config(tmp_path, ci_timeout_seconds=120, ci_poll_interval_seconds=10)
    runner = _StubMergeabilityRunner(
        mergeability_responses=[
            {"mergeable": "MERGEABLE", "mergeStateStatus": "CLEAN", "headRefOid": "h1", "baseRefName": "main"}
        ],
    )

    outcome = watch_pr_checks(runner, config, 77, metadata=_metadata())

    assert outcome.status == "passed"
    assert outcome.head_sha == "headsha1"


def test_watcher_reprobes_conflict_on_later_poll(tmp_path):
    config = make_config(tmp_path, ci_timeout_seconds=120, ci_poll_interval_seconds=10)
    runner = _StubMergeabilityRunner(
        mergeability_responses=[
            {"mergeable": "MERGEABLE", "mergeStateStatus": "CLEAN", "headRefOid": "h1", "baseRefName": "main"},
            {"mergeable": "CONFLICTING", "mergeStateStatus": "DIRTY", "headRefOid": "h1", "baseRefName": "main"},
        ],
        check_runs_responses=[
            {"check_runs": [{"name": "test", "status": "in_progress", "conclusion": None}]},
        ],
    )

    outcome = watch_pr_checks(runner, config, 77, metadata=_metadata())

    assert outcome.status == "merge_conflict"
    assert outcome.mergeability is not None
    assert outcome.mergeability.state == "conflicted"


# --- _reconcile_merge_conflict_item ------------------------------------------


def test_reconcile_adds_blocking_item_when_conflicted():
    mergeability = PullRequestMergeability(
        state="conflicted",
        mergeable_raw="CONFLICTING",
        merge_state_raw="DIRTY",
        head_sha="abc123",
        base_branch="main",
    )

    items = _reconcile_merge_conflict_item(
        [], mergeability=mergeability, source_round=1, current_head_sha="abc123"
    )

    assert len(items) == 1
    assert items[0].item_id == MERGE_CONFLICT_ITEM_ID
    assert items[0].status == "blocking"
    assert "main" in items[0].text
    assert "abc123" in items[0].text
    # The confirmed head is recorded so a later transient `unknown` probe on
    # the same head can be told apart from a genuinely resolved conflict.
    assert any("abc123" in note for note in items[0].notes)


def test_reconcile_clears_item_when_mergeable():
    prior = [
        UnresolvedReviewItem(
            item_id=MERGE_CONFLICT_ITEM_ID,
            reviewer="Orchestrator",
            source_round=1,
            text="stale conflict text",
            status="blocking",
            source_status="blocking",
            notes=("confirmed-head:h1",),
        )
    ]
    mergeability = PullRequestMergeability(
        state="mergeable", mergeable_raw="MERGEABLE", merge_state_raw="CLEAN", head_sha="h2", base_branch="main"
    )

    items = _reconcile_merge_conflict_item(
        prior, mergeability=mergeability, source_round=2, current_head_sha="h2"
    )

    assert items == []


def test_reconcile_no_op_when_unknown_without_prior_conflict():
    """A fresh `unknown` with nothing previously confirmed must never create
    a blocker -- a transient GitHub mergeability computation window must not
    trigger an unnecessary coder round."""
    mergeability = PullRequestMergeability(
        state="unknown", mergeable_raw=None, merge_state_raw=None, head_sha=None, base_branch=None
    )

    items = _reconcile_merge_conflict_item(
        [], mergeability=mergeability, source_round=2, current_head_sha="h1"
    )

    assert items == []


def test_reconcile_preserves_confirmed_conflict_when_unknown_on_same_head():
    """Regression (#609 review): once a conflict is confirmed on a head, a
    later probe returning `unknown` for that *same* head (a probe hiccup, not
    new information -- e.g. the coder round made no push) must not silently
    clear the blocker. Clearing it would let reviewers, checks, or a merge
    proceed against a branch GitHub last told us is still conflicted, and
    would bypass the unchanged-head bounded-progress guard, which only fires
    while the synthetic item remains present."""
    prior = [
        UnresolvedReviewItem(
            item_id=MERGE_CONFLICT_ITEM_ID,
            reviewer="Orchestrator",
            source_round=1,
            text="PR branch has a merge conflict with `main`.",
            status="blocking",
            source_status="blocking",
            notes=("confirmed-head:abc123",),
        )
    ]
    mergeability = PullRequestMergeability(
        state="unknown", mergeable_raw=None, merge_state_raw=None, head_sha=None, base_branch=None
    )

    items = _reconcile_merge_conflict_item(
        prior, mergeability=mergeability, source_round=2, current_head_sha="abc123"
    )

    assert len(items) == 1
    assert items[0].item_id == MERGE_CONFLICT_ITEM_ID
    assert items[0].status == "blocking"


def test_reconcile_clears_item_when_unknown_after_head_advanced():
    """Once the head genuinely changes (a resolution attempt happened), a
    fresh `unknown` for that new commit is ordinary -- not a hiccup on the
    known-conflicted head -- so it clears normally and lets the round
    re-evaluate the new head."""
    prior = [
        UnresolvedReviewItem(
            item_id=MERGE_CONFLICT_ITEM_ID,
            reviewer="Orchestrator",
            source_round=1,
            text="PR branch has a merge conflict with `main`.",
            status="blocking",
            source_status="blocking",
            notes=("confirmed-head:abc123",),
        )
    ]
    mergeability = PullRequestMergeability(
        state="unknown", mergeable_raw=None, merge_state_raw=None, head_sha=None, base_branch=None
    )

    items = _reconcile_merge_conflict_item(
        prior, mergeability=mergeability, source_round=2, current_head_sha="def456"
    )

    assert items == []


def test_reconcile_preserves_other_items():
    other_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Fix the null check.",
        status="blocking",
        source_status="blocking",
    )
    mergeability = PullRequestMergeability(
        state="conflicted", mergeable_raw="CONFLICTING", merge_state_raw="DIRTY", head_sha="h1", base_branch="main"
    )

    items = _reconcile_merge_conflict_item(
        [other_item], mergeability=mergeability, source_round=2, current_head_sha="h1"
    )

    assert other_item in items
    assert any(item.item_id == MERGE_CONFLICT_ITEM_ID for item in items)


# --- structured coder followup validation excludes the synthetic item -------


def test_structured_followup_excludes_merge_conflict_item_from_allowed_ids():
    conflict_item = UnresolvedReviewItem(
        item_id=MERGE_CONFLICT_ITEM_ID,
        reviewer="Orchestrator",
        source_round=1,
        text="conflict",
        status="blocking",
        source_status="blocking",
    )
    other_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Fix the null check.",
        status="blocking",
        source_status="blocking",
    )
    text = structured_coder_followup(addressed_items=["item-1"])
    parsed = validate_structured_coder_followup(text)

    # Does not raise: item-1 is classified and the synthetic conflict item is
    # neither required nor referenced.
    _validate_structured_coder_followup_items(
        parsed, unresolved_items=[conflict_item, other_item]
    )


def test_structured_followup_rejects_merge_conflict_item_reference():
    conflict_item = UnresolvedReviewItem(
        item_id=MERGE_CONFLICT_ITEM_ID,
        reviewer="Orchestrator",
        source_round=1,
        text="conflict",
        status="blocking",
        source_status="blocking",
    )
    text = structured_coder_followup(addressed_items=[MERGE_CONFLICT_ITEM_ID])
    parsed = validate_structured_coder_followup(text)

    try:
        _validate_structured_coder_followup_items(parsed, unresolved_items=[conflict_item])
    except Exception as exc:  # AgentLoopError
        assert "unknown" in str(exc).lower()
    else:
        raise AssertionError("expected an error for referencing the synthetic conflict item")


# --- comment rendering label --------------------------------------------------


def test_merge_conflict_item_label():
    item = UnresolvedReviewItem(
        item_id=MERGE_CONFLICT_ITEM_ID,
        reviewer="Orchestrator",
        source_round=3,
        text="PR branch has a merge conflict with `main`.",
        status="blocking",
        source_status="blocking",
    )

    label = _format_unresolved_item_label(item)

    assert "Merge conflict item, round 3" in label
