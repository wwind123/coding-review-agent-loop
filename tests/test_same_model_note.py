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


# --- loop-level publication contract (#1236) --------------------------------

from coding_review_agent_loop import pr_loop, plan_first_loop

AGY_QUOTA = (
    "error: RESOURCE_EXHAUSTED (code 429): Resource has been exhausted (e.g. check quota).",
    1,
)
NOTE_OPUS = "note: Claude and Antigravity both reviewed with Claude Opus 5.5"


def _agent_calls(runner):
    return [c[0] for c, _cwd in runner.commands if c and c[0] in {"claude", "codex", "gemini", "agy"}]


def _fallback_pair_runner(*, fall_back: bool, plan: bool = False):
    if plan:
        from test_review_parallel import _initial_plan

        review = structured_plan_review(summary="Plan ok.", reviewer="Google Antigravity")
        claude_review = structured_plan_review(summary="Claude plan ok.", reviewer="Claude")
        outputs = {"codex_outputs": [_initial_plan()]}
    else:
        review = structured_pr_review(summary="Agy ok.", reviewer="Google Antigravity")
        claude_review = structured_pr_review(summary="Claude ok.", reviewer="Claude")
        outputs = {}
    agy = ([AGY_QUOTA] if fall_back else []) + [(review, 0)]
    return FakeRunner(claude_outputs=[claude_review], antigravity_outputs=agy, **outputs)


def _pair_config(tmp_path, parallel, plan=False):
    return make_config(
        tmp_path, reviewer=("claude", "antigravity"), claude_model="claude-opus-5-5",
        coder="codex", review_parallel=parallel,
    )


def _run(kind, runner, config):
    if kind == "pr":
        return run_pr_loop(runner, pr_number=77, config=config)
    return run_issue_loop(runner, issue_number=56, config=config, plan_first=True)


@pytest.mark.parametrize("kind", ["pr", "plan"])
@pytest.mark.parametrize("parallel", [False, True])
def test_claude_agy_opus_fallback_pair_posts_note_once_and_resume_is_idempotent(
    tmp_path, kind, parallel
):
    runner = _fallback_pair_runner(fall_back=True, plan=kind == "plan")
    config = _pair_config(tmp_path, parallel)
    logs: list[str] = []
    module = pr_loop if kind == "pr" else plan_first_loop
    with patch.object(module, "log", lambda _c, m: logs.append(m)):
        assert _run(kind, runner, config) == 0
    posted = [c for c in runner.comments if "reconciliation" in c]
    assert len(posted) == 1 and NOTE_OPUS in posted[0]
    assert any(NOTE_OPUS in line for line in logs)
    calls = _agent_calls(runner)
    # Resume after the summary was posted: no re-invocation and no duplicate.
    assert _run(kind, runner, config) == 0
    assert _agent_calls(runner) == calls
    assert len([c for c in runner.comments if "reconciliation" in c]) == 1


@pytest.mark.parametrize("kind", ["pr", "plan"])
@pytest.mark.parametrize("parallel", [False, True])
def test_resolved_models_not_configured_models_decide_the_note(tmp_path, kind, parallel):
    # Opus is in the configured chain, but agy served Gemini: no shared model.
    runner = _fallback_pair_runner(fall_back=False, plan=kind == "plan")
    config = _pair_config(tmp_path, parallel)
    assert _run(kind, runner, config) == 0
    assert not any("reviewed with" in c for c in runner.comments)
    if not parallel:
        assert not any("reconciliation" in c for c in runner.comments)


@pytest.mark.parametrize("kind", ["pr", "plan"])
@pytest.mark.parametrize("parallel", [False, True])
def test_resume_before_reconciliation_posts_note_once(tmp_path, kind, parallel):
    runner = _fallback_pair_runner(fall_back=True, plan=kind == "plan")
    config = _pair_config(tmp_path, parallel)
    attr = "post_pr_comment" if kind == "pr" else "post_issue_comment"
    real_post = getattr(orchestrator, attr)

    def interrupt(*args, **kwargs):
        if "reconciliation" in kwargs["body"]:
            raise KeyboardInterrupt
        return real_post(*args, **kwargs)

    with patch.object(orchestrator, attr, side_effect=interrupt):
        with pytest.raises(KeyboardInterrupt):
            _run(kind, runner, config)
    assert not any("reconciliation" in c for c in runner.comments)
    calls = _agent_calls(runner)
    assert _run(kind, runner, config) == 0
    assert _agent_calls(runner) == calls  # same-round reviews resumed, not re-invoked
    posted = [c for c in runner.comments if "reconciliation" in c]
    assert len(posted) == 1 and NOTE_OPUS in posted[0]


def test_carried_prior_round_approval_never_contributes_and_note_resets_per_round(tmp_path):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(summary="Codex ok.")] * 3,
        gemini_outputs=[structured_pr_review(summary="G ok.", reviewer="Google Gemini")],
        antigravity_outputs=[
            (structured_pr_review(
                state="blocking", summary="A blocks.", reviewer="Google Antigravity",
                blocking_items=["Fix it."]), 0),
            (structured_pr_review(
                summary="A ok.", reviewer="Google Antigravity",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}]), 0),
        ],
        claude_outputs=[structured_coder_followup(summary="Fixed.", addressed_items=["item-1"])],
    )
    runner.advance_pr_head_on_coder_followup = False
    config = make_config(
        tmp_path, reviewer=("codex", "gemini", "antigravity"),
        pr_review_policy="primary-then-panel", primary_reviewer="codex",
        gemini_model="shared-model-1", antigravity_models=("shared-model-1",), max_rounds=6,
    )
    logs: list[str] = []
    with patch.object(pr_loop, "log", lambda _c, m: logs.append(m)):
        assert run_pr_loop(runner, pr_number=77, config=config) == 0
    # Round 2: Gemini and Antigravity both reviewed fresh; round 3: Gemini's
    # approval is only carried, so no note even though the models match.
    notes = [c for c in runner.comments if "reviewed with shared-model-1" in c]
    assert len(notes) == 1 and "reconciliation: settled reviewers: Antigravity, Gemini" in notes[0]
    round3 = next(c for c in runner.comments if c.startswith("PR review round 3 reconciliation"))
    assert "reviewed with" not in round3
    assert sum("reviewed with shared-model-1" in line for line in logs) == 1


# --- review round 2: checkpoint resume, evidence pass, unknown models ----------

@pytest.mark.parametrize("kind", ["pr", "plan"])
@pytest.mark.parametrize("parallel", [False, True])
def test_resume_after_reconciliation_posted_before_finalization(tmp_path, kind, parallel):
    runner = _fallback_pair_runner(fall_back=True, plan=kind == "plan")
    config = _pair_config(tmp_path, parallel)
    attr = "post_pr_comment" if kind == "pr" else "post_issue_comment"
    real_post = getattr(orchestrator, attr)

    def post_then_interrupt(*args, **kwargs):
        result = real_post(*args, **kwargs)
        if "reconciliation" in kwargs["body"]:
            raise KeyboardInterrupt
        return result

    with patch.object(orchestrator, attr, side_effect=post_then_interrupt):
        with pytest.raises(KeyboardInterrupt):
            _run(kind, runner, config)
    assert len([c for c in runner.comments if "reconciliation" in c]) == 1
    calls = _agent_calls(runner)
    logs: list[str] = []
    module = pr_loop if kind == "pr" else plan_first_loop
    with patch.object(module, "log", lambda _c, m: logs.append(m)):
        assert _run(kind, runner, config) == 0
    assert _agent_calls(runner) == calls
    posted = [c for c in runner.comments if "reconciliation" in c]
    assert len(posted) == 1 and NOTE_OPUS in posted[0]
    assert any(NOTE_OPUS in line for line in logs)


@pytest.mark.parametrize("kind", ["pr", "plan"])
@pytest.mark.parametrize("parallel", [False, True])
def test_resume_before_reconciliation_logs_the_note(tmp_path, kind, parallel):
    runner = _fallback_pair_runner(fall_back=True, plan=kind == "plan")
    config = _pair_config(tmp_path, parallel)
    attr = "post_pr_comment" if kind == "pr" else "post_issue_comment"
    real_post = getattr(orchestrator, attr)

    def interrupt(*args, **kwargs):
        if "reconciliation" in kwargs["body"]:
            raise KeyboardInterrupt
        return real_post(*args, **kwargs)

    with patch.object(orchestrator, attr, side_effect=interrupt):
        with pytest.raises(KeyboardInterrupt):
            _run(kind, runner, config)
    logs: list[str] = []
    module = pr_loop if kind == "pr" else plan_first_loop
    with patch.object(module, "log", lambda _c, m: logs.append(m)):
        assert _run(kind, runner, config) == 0
    assert any(NOTE_OPUS in line for line in logs)


@pytest.mark.parametrize("kind", ["pr", "plan"])
@pytest.mark.parametrize("parallel", [False, True])
def test_unknown_resolved_models_never_produce_a_note(tmp_path, kind, parallel):
    if kind == "plan":
        from test_review_parallel import _initial_plan

        runner = FakeRunner(
            claude_outputs=[structured_plan_review(summary="C ok.", reviewer="Claude")],
            codex_outputs=[_initial_plan(), structured_plan_review(summary="X ok.")],
        )
        config = make_config(tmp_path, reviewer=("claude", "codex"), coder="codex",
                             review_parallel=parallel)
    else:
        runner = FakeRunner(
            claude_outputs=[structured_pr_review(summary="C ok.", reviewer="Claude")],
            codex_outputs=[structured_pr_review(summary="X ok.")],
        )
        config = make_config(tmp_path, reviewer=("claude", "codex"), review_parallel=parallel)
    assert _run(kind, runner, config) == 0
    assert not any("reviewed with" in c for c in runner.comments)
    if not parallel:
        assert not any("reconciliation" in c for c in runner.comments)


def test_evidence_response_pass_gains_no_reconciliation_summary(tmp_path):
    from test_orchestrator_pr import _EVIDENCE_TEXT, _evidence_review, _signed

    runner = FakeRunner(
        codex_outputs=[_evidence_review(evidence=[_EVIDENCE_TEXT])],
        gemini_outputs=[_evidence_review(reviewer="Google Gemini")],
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), codex_model="shared-model-1",
        gemini_model="shared-model-1",
    )
    with pytest.raises(orchestrator.HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    before = len([c for c in runner.comments if "reconciliation" in c])
    assert before == 1  # the authoritative round carries the note
    runner.pr_payload["comments"].append(_signed("Live suite at abc123: 11 passed.", 1))
    from test_orchestrator_pr import _resolve

    runner.codex_outputs.append(_evidence_review(dispositions=[_resolve()], hr=True))
    runner.gemini_outputs.append(
        _evidence_review(
            reviewer="Google Gemini", hr=True,
            dispositions=[_resolve("item-1", "Codex owns this request.")],
        )
    )
    logs: list[str] = []
    with patch.object(pr_loop, "log", lambda _c, m: logs.append(m)):
        assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert runner.codex_outputs == [] and runner.gemini_outputs == []  # pass really ran
    assert len([c for c in runner.comments if "reconciliation" in c]) == before
    assert any("reviewed with shared-model-1" in line for line in logs)  # log line only
