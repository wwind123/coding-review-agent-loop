"""Same-model panel note in round reconciliation summaries (#1236)."""
from unittest.mock import patch

import pytest

from agent_loop_helpers import *  # noqa: F403
from coding_review_agent_loop.comment_rendering import (
    canonical_model_identity,
    same_model_panel_note,
)


def test_canonical_model_identity_unifies_claude_and_agy_labels():
    assert canonical_model_identity("claude-opus-5-5") == canonical_model_identity(
        "Claude Opus 5.5 (Medium)"
    )
    assert canonical_model_identity("claude-opus-5-5-medium") == "claude-opus-5-5"
    assert canonical_model_identity("Gemini 3.1 Pro (High)") != canonical_model_identity("claude-opus-5-5")
    assert canonical_model_identity(None) == "" and canonical_model_identity("  ") == ""


def test_note_names_both_reviewers_and_readable_model():
    note = same_model_panel_note(
        {"Claude": "claude-opus-5-5", "Antigravity": "Claude Opus 5.5 (Medium)", "Codex": "gpt-6.1"}
    )
    assert note == "note: Claude and Antigravity both reviewed with Claude Opus 5.5"


def test_note_omitted_for_distinct_or_unknown_models():
    assert same_model_panel_note({"A": "m1", "B": "m2"}) is None
    assert same_model_panel_note({"A": None, "B": ""}) is None
    assert same_model_panel_note({}) is None
    assert same_model_panel_note(
        {"Codex": "unknown model (medium)", "Claude": "unknown model (medium)"}
    ) is None


def _shared_config(tmp_path, **overrides):
    return make_config(
        tmp_path, reviewer=("codex", "gemini"), codex_model="shared-model-1",
        gemini_model="shared-model-1", **overrides,
    )


@pytest.mark.parametrize("parallel", [False, True])
def test_pr_round_with_shared_model_posts_note_in_reconciliation(tmp_path, parallel):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(summary="Codex ok.")],
        gemini_outputs=[structured_pr_review(summary="Gemini ok.", reviewer="Google Gemini")],
    )
    config = _shared_config(tmp_path, review_parallel=parallel)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    reconciliations = [c for c in runner.comments if "reconciliation" in c]
    assert len(reconciliations) == 1
    assert "note: " in reconciliations[0] and "both reviewed with shared-model-1" in reconciliations[0]


def test_sequential_pr_round_without_shared_model_posts_no_reconciliation(tmp_path):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(summary="Codex ok.")],
        gemini_outputs=[structured_pr_review(summary="Gemini ok.", reviewer="Google Gemini")],
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), codex_model="model-a", gemini_model="model-b"
    )
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert not any("reconciliation" in c for c in runner.comments)
    assert not any("note: " in c and "reviewed with" in c for c in runner.comments)


@pytest.mark.parametrize("parallel", [False, True])
def test_plan_round_with_shared_model_posts_note(tmp_path, parallel):
    from test_review_parallel import _initial_plan

    runner = FakeRunner(
        claude_outputs=[_initial_plan()],
        codex_outputs=[structured_plan_review(summary="Codex plan ok.")],
        gemini_outputs=[structured_plan_review(summary="Gemini plan ok.", reviewer="Google Gemini")],
    )
    config = _shared_config(tmp_path, review_parallel=parallel)
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    reconciliations = [c for c in runner.comments if "reconciliation" in c]
    assert len(reconciliations) == 1
    assert "both reviewed with shared-model-1" in reconciliations[0]


def test_resume_before_reconciliation_posts_note_exactly_once(tmp_path):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(summary="Codex ok.")],
        gemini_outputs=[structured_pr_review(summary="Gemini ok.", reviewer="Google Gemini")],
    )
    config = _shared_config(tmp_path, review_parallel=True)
    real_post = orchestrator.post_pr_comment

    def interrupt(*args, **kwargs):
        if "reconciliation" in kwargs["body"]:
            raise KeyboardInterrupt
        return real_post(*args, **kwargs)

    with patch.object(orchestrator, "post_pr_comment", side_effect=interrupt):
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=config)
    assert not any("reconciliation" in c for c in runner.comments)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    posted = [c for c in runner.comments if "reconciliation" in c]
    assert len(posted) == 1 and "both reviewed with shared-model-1" in posted[0]
