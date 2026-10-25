"""PR-loop tests for review_step_back and finding_history: PR convergence (#1542)."""

import json
import re
from unittest.mock import patch

import pytest

from coding_review_agent_loop.runner import CommandResult
from coding_review_agent_loop.cli import AgentLoopError, run_pr_loop
from coding_review_agent_loop.errors import HumanDecisionRequiredError
from coding_review_agent_loop.github import (
    CiWatchOutcome,
    PullRequestCheck,
)
from agent_loop_helpers import (
    FakeRunner,
    blocking_issues,
    make_config,
    prior_item_dispositions,
    structured_coder_followup,
    structured_pr_review,
)
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.plan_first_loop as plan_first_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.triage as triage
import coding_review_agent_loop.validated_agent as validated_agent
from agent_loop_helpers import patch_attr_in
from pr_loop_test_helpers import (
    SB_HEAD,
    _StepBackRunner,
    _step_back_coders,
    _watch_check_board,
)


# --- PR fix-loop step-back (#1251, stage 2) ---------------------------------


def _sb_review(
    blocking_items=None, *, resolved=(), carried=(), state=None, summary="review",
    signature="OpenAI Codex",
):
    state = state or ("blocking" if blocking_items or carried else "approved")
    dispositions = [{"item_id": item, "disposition": "resolved"} for item in resolved]
    dispositions += [{"item_id": item, "disposition": "blocking", "note": "still open"} for item in carried]
    return (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": state,
                "summary": summary,
                "blocking_items": blocking_items or [],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": dispositions,
            }
        )
        + f"\n<!-- AGENT_STATE: {state} -->\n-- {signature}"
    )


def _sb_config(tmp_path, **overrides):
    overrides.setdefault("max_rounds", 8)
    return make_config(tmp_path, reviewer=("codex",), **overrides)


def _sb_coder_prompts(runner):
    return [
        "\n".join(command) for command, _cwd in runner.commands
        if command and command[0] == "claude"
    ]


def _sb_codex_prompts(runner):
    return [
        "\n".join(command) for command, _cwd in runner.commands
        if command and command[0] == "codex"
    ]


def _three_clustered_blocks():
    return [
        _sb_review(["gap one at src/spool.py:124-131"]),
        _sb_review(["gap two at src/spool.py:131-136"], resolved=["item-1"]),
        _sb_review(["gap three at src/spool.py:140"], resolved=["item-2"]),
    ]


def test_clustered_new_finding_blocks_trigger_a_step_back_turn_and_a_bound_sweep(tmp_path):
    """`pr-clustered-k-blocks`."""
    runner = _StepBackRunner(
        claude_outputs=_step_back_coders(),
        codex_outputs=[*_three_clustered_blocks(), _sb_review(resolved=["item-3"])],
    )
    assert run_pr_loop(runner, pr_number=77, config=_sb_config(tmp_path)) == 0
    coder_prompts = _sb_coder_prompts(runner)
    assert len(coder_prompts) == 3
    assert all("STEP-BACK TURN" not in prompt for prompt in coder_prompts[:2])
    assert "STEP-BACK TURN" in coder_prompts[2]
    assert "src/spool.py` lines 124-140" in coder_prompts[2]
    assert "small, localized cleanup" not in coder_prompts[2]
    review_prompts = _sb_codex_prompts(runner)
    assert len(review_prompts) == 4
    assert all("Enumerate ALL remaining instances" not in p for p in review_prompts[:3])
    assert "Enumerate ALL remaining instances" in review_prompts[3]


def test_clustered_blocks_trigger_a_step_back_turn_under_review_parallel(tmp_path):
    """`pr-parallel-trigger` (#1271): provisional publication + reconciliation records."""
    from coding_review_agent_loop.github import get_pr_review_context
    from coding_review_agent_loop.round_state import _extract_round_metadata_records

    runner = _StepBackRunner(
        claude_outputs=_step_back_coders(),
        codex_outputs=[*_three_clustered_blocks(), _sb_review(resolved=["item-3"])],
    )
    config = _sb_config(tmp_path, review_parallel=True)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    records = _extract_round_metadata_records(
        get_pr_review_context(runner, config=config, pr_number=77).comments, flow="pr"
    )
    reviews = [r.metadata for r in records if r.metadata.role == "reviewer"]
    assert any(m.phase == "publication" and m.new_items == () for m in reviews)
    assert any(
        r.metadata.phase == "reconciliation" and r.metadata.new_items for r in records
    )
    coder_prompts = _sb_coder_prompts(runner)
    assert len(coder_prompts) == 3
    assert all("STEP-BACK TURN" not in prompt for prompt in coder_prompts[:2])
    assert "STEP-BACK TURN" in coder_prompts[2]


def test_step_back_entry_is_recorded_on_the_coder_record_and_not_repeated(tmp_path):
    from coding_review_agent_loop.github import get_pr_review_context
    from coding_review_agent_loop.round_state import _extract_round_metadata_records

    runner = _StepBackRunner(
        claude_outputs=_step_back_coders(),
        codex_outputs=[*_three_clustered_blocks(), _sb_review(resolved=["item-3"])],
    )
    config = _sb_config(tmp_path)
    run_pr_loop(runner, pr_number=77, config=config)
    comments = get_pr_review_context(runner, config=config, pr_number=77).comments
    records = _extract_round_metadata_records(comments, flow="pr")
    stepped = [r for r in records if r.metadata.role == "coder" and r.metadata.step_back_entries]
    assert len(stepped) == 1
    (entry,) = stepped[0].metadata.step_back_entries
    assert entry["reviewer"] == "Codex" and entry["trigger_round"] == 3
    assert entry["trigger_head"] == f"{2:040x}"
    assert entry["anchor"] == {"path": "src/spool.py", "start": 124, "end": 140}
    assert stepped[0].metadata.round_number == 4
    assert stepped[0].metadata.subject == f"{3:040x}"


def test_unrelated_blocks_do_not_trigger_a_step_back(tmp_path):
    """`pr-no-trigger`."""
    runner = _StepBackRunner(
        claude_outputs=[structured_coder_followup(addressed_items=[f"item-{n}"]) for n in (1, 2, 3)],
        codex_outputs=[
            _sb_review(["gap at src/a.py:10"]),
            _sb_review(["gap at src/b.py:10"], resolved=["item-1"]),
            _sb_review(["gap at src/c.py:10"], resolved=["item-2"]),
            _sb_review(resolved=["item-3"]),
        ],
    )
    assert run_pr_loop(runner, pr_number=77, config=_sb_config(tmp_path)) == 0
    assert all("STEP-BACK TURN" not in prompt for prompt in _sb_coder_prompts(runner))
    assert all("Enumerate ALL remaining" not in p for p in _sb_codex_prompts(runner))


def test_disabling_the_pr_step_back_restores_the_ordinary_prompt(tmp_path):
    runner = _StepBackRunner(
        claude_outputs=_step_back_coders(),
        codex_outputs=[*_three_clustered_blocks(), _sb_review(resolved=["item-3"])],
    )
    config = _sb_config(tmp_path, pr_step_back_rounds=0)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert all("STEP-BACK TURN" not in prompt for prompt in _sb_coder_prompts(runner))
    assert all("Enumerate ALL remaining" not in p for p in _sb_codex_prompts(runner))


def _run_to_sibling(tmp_path, sibling, *, config=None, **runner_kwargs):
    runner = _StepBackRunner(
        claude_outputs=_step_back_coders(),
        codex_outputs=[*_three_clustered_blocks(), _sb_review([sibling], resolved=["item-3"])],
        **runner_kwargs,
    )
    with pytest.raises(HumanDecisionRequiredError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=config or _sb_config(tmp_path))
    return runner, str(excinfo.value)


def test_a_sibling_in_the_sweep_review_stops_for_a_human_decision_before_another_coder_turn(tmp_path):
    """`pr-sibling-escalation`."""
    runner, message = _run_to_sibling(tmp_path, "another branch at src/spool.py:134")
    assert "human decision required" in message
    assert "Generalization: any spool record that fails a validation step is malformed." in message
    assert "--pr-step-back-rounds 0" in message and "accept a stated limitation" in message
    assert len(_sb_coder_prompts(runner)) == 3  # no coder turn after the sweep
    assert len(_sb_codex_prompts(runner)) == 4


def test_the_cli_maps_a_sibling_escalation_to_the_human_decision_exit_code(tmp_path, capsys):
    """`sibling-cli-exit-code`: the live sibling stop exits 4, not the generic 1."""
    import coding_review_agent_loop.cli as cli_module

    runner = _StepBackRunner(
        claude_outputs=_step_back_coders(),
        codex_outputs=[
            *_three_clustered_blocks(),
            _sb_review(["another branch at src/spool.py:134"], resolved=["item-3"]),
        ],
    )
    config = _sb_config(tmp_path)

    def drive_the_fixture(*_args, **_kwargs):
        return run_pr_loop(runner, pr_number=77, config=config)

    with patch.object(cli_module, "run_pr_loop", side_effect=drive_the_fixture):
        assert cli_module.main(["pr", "77", "--repo", "OWNER/REPO"]) == 4
    stderr = capsys.readouterr().err
    assert "HUMAN DECISION REQUIRED" in stderr
    assert "--pr-step-back-rounds 0" in stderr
    assert len(_sb_coder_prompts(runner)) == 3  # no coder turn after the sweep


def test_a_sweep_sibling_located_only_in_a_sub_item_escalates(tmp_path):
    """`pr-sweep-sub-item-location`."""
    sibling = {
        "text": "More spool validation branches are misclassified",
        "sub_items": ["branch at src/spool.py:130 treats it as absent", "branch at src/other.py:9"],
    }
    runner, message = _run_to_sibling(tmp_path, sibling)
    assert "human decision required" in message and "src/spool.py:130-130" in message
    assert len(_sb_coder_prompts(runner)) == 3


def test_a_sweep_finding_outside_the_cluster_does_not_escalate(tmp_path):
    runner = _StepBackRunner(
        claude_outputs=[*_step_back_coders(), structured_coder_followup(addressed_items=["item-4"])],
        codex_outputs=[
            *_three_clustered_blocks(),
            _sb_review(["unrelated at src/other.py:5"], resolved=["item-3"]),
            _sb_review(resolved=["item-4"]),
        ],
    )
    assert run_pr_loop(runner, pr_number=77, config=_sb_config(tmp_path)) == 0
    # The fourth coder turn is an ordinary follow-up, not a second step-back.
    assert "STEP-BACK TURN" not in _sb_coder_prompts(runner)[3]


def test_a_carried_item_does_not_escalate_and_no_second_step_back_is_issued(tmp_path):
    runner = _StepBackRunner(
        claude_outputs=[*_step_back_coders(), structured_coder_followup(addressed_items=["item-3"])],
        codex_outputs=[
            *_three_clustered_blocks(),
            _sb_review(carried=["item-3"]),
            _sb_review(resolved=["item-3"]),
        ],
    )
    assert run_pr_loop(runner, pr_number=77, config=_sb_config(tmp_path)) == 0
    prompts = _sb_coder_prompts(runner)
    assert len(prompts) == 4 and "STEP-BACK TURN" not in prompts[3]


def test_a_step_back_that_shifts_lines_by_one_hundred_still_escalates_a_true_sibling(tmp_path):
    """`pr-head-coordinate-mapping` through the orchestrator."""
    hunks = "@@ -10,0 +11,100 @@\n"
    _runner, message = _run_to_sibling(
        tmp_path, "another branch at src/spool.py:230", hunks=hunks
    )
    assert "human decision required" in message and "SHIFTED" in message
    assert "src/spool.py` lines 224-240" in message


def test_unrelated_code_shifted_into_the_old_coordinates_does_not_escalate(tmp_path):
    hunks = "@@ -10,0 +11,100 @@\n"
    runner = _StepBackRunner(
        claude_outputs=[*_step_back_coders(), structured_coder_followup(addressed_items=["item-4"])],
        codex_outputs=[
            *_three_clustered_blocks(),
            _sb_review(["code now at src/spool.py:130"], resolved=["item-3"]),
            _sb_review(resolved=["item-4"]),
        ],
        hunks=hunks,
    )
    assert run_pr_loop(runner, pr_number=77, config=_sb_config(tmp_path)) == 0


def test_an_unmappable_anchor_falls_back_to_any_finding_on_the_path(tmp_path):
    hunks = "@@ -124,17 +123,0 @@\n"
    _runner, message = _run_to_sibling(
        tmp_path, "another branch at src/spool.py:900", hunks=hunks
    )
    assert "UNMAPPABLE" in message and "human decision required" in message


def test_resume_after_the_step_back_record_rebuilds_the_sweep_without_a_second_step_back(tmp_path):
    """`pr-resume-mid-step-back`."""
    runner = _StepBackRunner(
        claude_outputs=_step_back_coders(),
        codex_outputs=_three_clustered_blocks(),
    )
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        # The step-back coder record is published, then the sweep review cannot run.
        run_pr_loop(runner, pr_number=77, config=_sb_config(tmp_path))
    assert len(_sb_coder_prompts(runner)) == 3
    runner.codex_outputs.append(_sb_review(resolved=["item-3"]))
    assert run_pr_loop(runner, pr_number=77, config=_sb_config(tmp_path)) == 0
    assert len(_sb_coder_prompts(runner)) == 3  # still one step-back turn, nothing re-dispatched
    assert "Enumerate ALL remaining instances" in _sb_codex_prompts(runner)[3]


def _gemini_review(*args, **kwargs):
    return _sb_review(*args, signature="Google Gemini", **kwargs)


def _two_reviewer_board(*, gemini_paths):
    """Codex and Gemini each block three rounds; item IDs interleave per round.

    Every reviewer dispositions every prior item, as the aggregate ledger requires.
    """
    codex, gemini = [], []
    for n, line in enumerate((124, 131, 140)):
        prior = [f"item-{2 * n - 1}", f"item-{2 * n}"] if n else []
        codex.append(_sb_review([f"codex gap at src/spool.py:{line}"], resolved=prior))
        gemini.append(
            _gemini_review([f"gemini gap at {gemini_paths[n]}:{10 + n}"], resolved=prior)
        )
    coders = [
        structured_coder_followup(addressed_items=[f"item-{2 * n + 1}", f"item-{2 * n + 2}"])
        for n in range(3)
    ]
    return codex, gemini, coders


def _two_reviewer_config(tmp_path):
    return make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=8)


def _sb_gemini_prompts(runner):
    return [
        "\n".join(command) for command, _cwd in runner.commands
        if command and command[0] == "gemini"
    ]


def test_one_trigger_on_a_multi_reviewer_board_leaves_the_other_reviewer_ordinary(tmp_path):
    codex, gemini, coders = _two_reviewer_board(
        gemini_paths=("src/a.py", "src/b.py", "src/c.py")
    )
    runner = _StepBackRunner(
        claude_outputs=coders,
        codex_outputs=[*codex, _sb_review(resolved=["item-5", "item-6"])],
        gemini_outputs=[*gemini, _gemini_review(resolved=["item-5", "item-6"])],
    )
    assert run_pr_loop(runner, pr_number=77, config=_two_reviewer_config(tmp_path)) == 0
    assert "STEP-BACK TURN" in _sb_coder_prompts(runner)[2]
    assert "src/a.py" not in _sb_coder_prompts(runner)[2].split("Clusters:")[1].split("Recent")[0]
    assert "Enumerate ALL remaining instances" in _sb_codex_prompts(runner)[3]
    assert "Enumerate ALL remaining instances" not in _sb_gemini_prompts(runner)[3]


def _simultaneous_runner(gemini_outputs_tail):
    codex, gemini, coders = _two_reviewer_board(
        gemini_paths=("src/other.py", "src/other.py", "src/other.py")
    )
    return _StepBackRunner(
        claude_outputs=coders,
        codex_outputs=[*codex, _sb_review(resolved=["item-5", "item-6"])],
        gemini_outputs=[*gemini, *gemini_outputs_tail],
    )


def test_simultaneous_clusters_share_one_coder_turn_with_one_bound_sweep_each(tmp_path):
    """`pr-simultaneous-clusters`."""
    from coding_review_agent_loop.github import get_pr_review_context
    from coding_review_agent_loop.round_state import _extract_round_metadata_records

    runner = _simultaneous_runner([_gemini_review(resolved=["item-5", "item-6"])])
    config = _two_reviewer_config(tmp_path)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    step_back_prompts = [p for p in _sb_coder_prompts(runner) if "STEP-BACK TURN" in p]
    assert len(step_back_prompts) == 1
    clusters = step_back_prompts[0].split("Clusters:")[1]
    assert clusters.index("Codex") < clusters.index("Gemini")  # configured-reviewer order
    assert "src/spool.py" in clusters and "src/other.py" in clusters
    assert "Enumerate ALL remaining instances" in _sb_codex_prompts(runner)[3]
    assert "Enumerate ALL remaining instances" in _sb_gemini_prompts(runner)[3]
    records = _extract_round_metadata_records(
        get_pr_review_context(runner, config=config, pr_number=77).comments, flow="pr"
    )
    entries = [
        entry for r in records if r.metadata.role == "coder" for entry in r.metadata.step_back_entries
    ]
    assert [entry["reviewer"] for entry in entries] == ["Codex", "Gemini"]


def test_resume_after_one_sweep_review_is_published_requests_only_the_missing_sweep(tmp_path):
    runner = _simultaneous_runner([])  # Gemini's sweep review cannot run the first time
    config = _two_reviewer_config(tmp_path)
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=config)
    codex_before = len(_sb_codex_prompts(runner))
    runner.gemini_outputs.append(_gemini_review(resolved=["item-5", "item-6"]))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert len(_sb_codex_prompts(runner)) == codex_before  # the published sweep is not re-requested
    assert "Enumerate ALL remaining instances" in _sb_gemini_prompts(runner)[-1]
    assert len([p for p in _sb_coder_prompts(runner) if "STEP-BACK TURN" in p]) == 1


def test_a_carried_in_window_item_keeps_the_episode_open_and_a_later_sibling_escalates(tmp_path):
    """`pr-clearance-window-membership`: live; the unit tests replay the same records."""
    runner = _StepBackRunner(
        claude_outputs=[*_step_back_coders(), structured_coder_followup(addressed_items=["item-3"])],
        codex_outputs=[
            *_three_clustered_blocks(),
            _sb_review(carried=["item-3"]),
            _sb_review(["late sibling at src/spool.py:135"], carried=["item-3"]),
        ],
    )
    with pytest.raises(HumanDecisionRequiredError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=_sb_config(tmp_path))
    assert "human decision required" in str(excinfo.value)
    assert len(_sb_coder_prompts(runner)) == 4  # no fifth coder turn


# --- review round 1 fixes (#1265) -------------------------------------------


def test_a_sweep_sibling_arriving_exactly_at_the_round_limit_still_stops_for_a_human_decision(tmp_path):
    runner = _StepBackRunner(
        claude_outputs=_step_back_coders(),
        codex_outputs=[*_three_clustered_blocks(), _sb_review(["sibling at src/spool.py:134"], resolved=["item-3"])],
    )
    with pytest.raises(HumanDecisionRequiredError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=_sb_config(tmp_path, max_rounds=4))
    message = str(excinfo.value)
    assert "human decision required" in message and "Reached max rounds" not in message
    assert "Generalization:" in message
    assert len(_sb_coder_prompts(runner)) == 3


def test_primary_then_panel_step_back_sweeps_only_the_primary_before_any_secondary(tmp_path):
    runner = _StepBackRunner(
        claude_outputs=_step_back_coders(),
        codex_outputs=[*_three_clustered_blocks(), _sb_review(resolved=["item-3"])],
        gemini_outputs=[_gemini_review()],
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), max_rounds=8,
        pr_review_policy="primary-then-panel", primary_reviewer="codex",
    )
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    commands = [c[0] for c, _cwd in runner.commands if c and c[0] in {"codex", "gemini"}]
    first_gemini = commands.index("gemini")
    assert commands[:first_gemini] == ["codex"] * 4  # no secondary before exact-head primary approval
    assert len([p for p in _sb_coder_prompts(runner) if "STEP-BACK TURN" in p]) == 1
    assert "Enumerate ALL remaining instances" in _sb_codex_prompts(runner)[3]
    assert all("Enumerate ALL remaining instances" not in p for p in _sb_gemini_prompts(runner))


def test_resume_after_a_carried_in_window_sweep_review_keeps_the_episode_and_stops_on_a_sibling(tmp_path):
    runner = _StepBackRunner(
        claude_outputs=[*_step_back_coders(), structured_coder_followup(addressed_items=["item-3"])],
        codex_outputs=[*_three_clustered_blocks(), _sb_review(carried=["item-3"])],
    )
    config = _sb_config(tmp_path)
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        # Round 4's sweep review (carrying the in-window item) and the ordinary fourth
        # coder turn are published; the run is interrupted before round 5's review.
        run_pr_loop(runner, pr_number=77, config=config)
    assert len(_sb_coder_prompts(runner)) == 4
    runner.codex_outputs.append(_sb_review(["late sibling at src/spool.py:135"], carried=["item-3"]))
    with pytest.raises(HumanDecisionRequiredError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=config)
    message = str(excinfo.value)
    assert "human decision required" in message
    assert "Generalization: any spool record that fails a validation step is malformed." in message
    assert len(_sb_coder_prompts(runner)) == 4  # no coder turn after the sibling


def test_resume_after_the_kth_review_is_published_still_issues_the_step_back_turn(tmp_path):
    """Interrupted between the K-th clustered review and its coder dispatch."""
    runner = _StepBackRunner(
        claude_outputs=_step_back_coders()[:2],  # the third (step-back) coder turn cannot run
        codex_outputs=_three_clustered_blocks(),
    )
    config = _sb_config(tmp_path)
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_pr_loop(runner, pr_number=77, config=config)
    runner.claude_outputs.append(_step_back_coders()[2])
    runner.codex_outputs.append(_sb_review(resolved=["item-3"]))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    prompts = _sb_coder_prompts(runner)
    assert "STEP-BACK TURN" in prompts[-1]  # the resumed dispatch, not an ordinary follow-up
    assert len([p for p in prompts[:2] if "STEP-BACK TURN" in p]) == 0
    assert "Enumerate ALL remaining instances" in _sb_codex_prompts(runner)[-1]


def test_a_rerun_after_a_sibling_stop_does_not_repeat_the_stop_for_the_old_sibling(tmp_path):
    """The operator pushes a redesign or a documented limitation; the next review
    blocks on something outside the cluster. The stale sibling must not stop the run."""
    runner, message = _run_to_sibling(tmp_path, "another branch at src/spool.py:134")
    assert "human decision required" in message
    assert len(_sb_coder_prompts(runner)) == 3
    runner._move_head(f"{9:040x}", cwd=None)  # the operator's push
    runner.pr_payload["headRefOid"] = f"{9:040x}"
    # The pushed head is recovered as an unrecorded advance: the coder checks the
    # stopped round's item, then the reviewer blocks on something outside the cluster.
    runner.claude_outputs.extend(
        [
            structured_coder_followup(addressed_items=["item-4"]),
            structured_coder_followup(addressed_items=["item-5"]),
        ]
    )
    runner.codex_outputs.extend(
        [
            _sb_review(["unrelated at src/other.py:5"], resolved=["item-4"]),
            _sb_review(resolved=["item-5"]),
        ]
    )
    assert run_pr_loop(runner, pr_number=77, config=_sb_config(tmp_path)) == 0
    assert len(_sb_coder_prompts(runner)) == 5  # three before the stop, two after the push
    assert all("STEP-BACK TURN" not in p for p in _sb_coder_prompts(runner)[3:])


def test_malformed_step_back_history_suppresses_the_sibling_stop(tmp_path, monkeypatch, capsys):
    """`sibling-no-stop-paths`: a decode failure logs and suppresses, never stops."""
    import coding_review_agent_loop.pr_loop_support as support
    from coding_review_agent_loop import review_step_back

    real_history = review_step_back.pr_step_back_history

    def degraded_once_recorded(records, reviewer):
        entries, degraded = real_history(records, reviewer)
        # Once the step-back entry is recorded, report its history as malformed.
        return entries, degraded or bool(entries)

    monkeypatch.setattr(support._step_back, "pr_step_back_history", degraded_once_recorded)
    runner = _StepBackRunner(
        claude_outputs=[*_step_back_coders(), structured_coder_followup(addressed_items=["item-4"])],
        codex_outputs=[
            *_three_clustered_blocks(),
            _sb_review(["another branch at src/spool.py:134"], resolved=["item-3"]),
            _sb_review(resolved=["item-4"]),
        ],
    )
    config = _sb_config(tmp_path, pr_non_convergence_rounds=0, quiet=False)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    prompts = _sb_coder_prompts(runner)
    assert len(prompts) == 4 and "STEP-BACK TURN" not in prompts[3]
    assert "PR step-back suppressed for Codex: malformed step-back history" in capsys.readouterr().err


def _external_push_after_the_kth_review(tmp_path, hunks):
    """Interrupt between the K-th clustered review and its coder dispatch, then push."""
    pushed = f"{9:040x}"
    runner = _StepBackRunner(
        claude_outputs=_step_back_coders()[:2],
        codex_outputs=_three_clustered_blocks(),
        hunks=hunks,
        shift_head=pushed,
    )
    config = _sb_config(tmp_path)
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_pr_loop(runner, pr_number=77, config=config)
    runner._move_head(pushed, cwd=None)  # an operator's refactor lands first
    runner.pr_payload["headRefOid"] = pushed
    runner.claude_outputs.append(structured_coder_followup(addressed_items=["item-3"]))
    runner.codex_outputs.append(_sb_review(resolved=["item-3"]))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    return runner


def test_an_external_push_that_deletes_the_cluster_suppresses_the_stale_step_back(tmp_path):
    runner = _external_push_after_the_kth_review(tmp_path, "@@ -100,60 +99,0 @@\n")
    prompts = _sb_coder_prompts(runner)
    # The first (interrupted) dispatch was a step-back on the then-current head; the
    # resumed one, on the pushed head, is ordinary because the code is now unmappable.
    assert "STEP-BACK TURN" in prompts[2]
    assert "STEP-BACK TURN" not in prompts[-1]
    assert all("Enumerate ALL remaining" not in p for p in _sb_codex_prompts(runner)[3:])


def test_an_external_push_that_shifts_the_cluster_maps_the_step_back_to_the_pushed_head(tmp_path):
    runner = _external_push_after_the_kth_review(tmp_path, "@@ -10,0 +11,100 @@\n")
    assert "src/spool.py` lines 224-240" in _sb_coder_prompts(runner)[-1]


# --- #1273: proactive generalization guidance and finding history ------------


def _history_runner(coder_outputs):
    return _StepBackRunner(
        claude_outputs=coder_outputs,
        codex_outputs=[*_three_clustered_blocks(), _sb_review(resolved=["item-3"])],
    )


def test_coder_prompts_show_prior_findings_and_fixes_when_a_sibling_arrives(tmp_path):
    """`fix-cutoff-sibling`, `pr-followup-and-same-pr`: no step-back involved."""
    runner = _history_runner(_step_back_coders())
    assert run_pr_loop(runner, pr_number=77, config=_sb_config(tmp_path, pr_step_back_rounds=0)) == 0
    prompts = _sb_coder_prompts(runner)
    assert all("Proactive generalization" in prompt for prompt in prompts)
    assert all("Generalization: this generalizes the fix for" in prompt for prompt in prompts)
    assert "no earlier-round findings or fixes" in prompts[0]
    # Round 2 prompt: the round-1 finding (resolved) with its location, and the
    # round-1 fix published under round 2; the round-2 sibling is only in the review.
    second = prompts[1]
    assert "Codex finding item-1 (resolved) src/spool.py:124-131" in second
    assert "fix by " in second
    history = second.split("Earlier-round history for this run", 1)[1].split("review:", 1)[0]
    assert "item-2" not in history and "src/spool.py:131-136" not in history


def test_declared_generalization_is_logged_with_its_tag(tmp_path, capsys):
    """`generalization-logged`: the step-back turn's declaration is step-back-directed."""
    runner = _history_runner(_step_back_coders())
    assert run_pr_loop(runner, pr_number=77, config=_sb_config(tmp_path, quiet=False)) == 0
    err = capsys.readouterr().err
    assert err.count("declared a generalization") == 1
    assert "declared a generalization (step-back-directed): Generalization: any spool record" in err


def test_undeclared_generalization_logs_nothing_and_proactive_tag_without_step_back(
    tmp_path, capsys
):
    coders = [
        structured_coder_followup(addressed_items=["item-1"]),
        structured_coder_followup(
            addressed_items=["item-2"],
            addressed_item_notes={
                "item-2": "Generalization: this generalizes the fix for [item-1]: any decode error"
            },
        ),
        structured_coder_followup(addressed_items=["item-3"]),
    ]
    runner = _history_runner(coders)
    assert run_pr_loop(runner, pr_number=77, config=_sb_config(tmp_path, pr_step_back_rounds=0, quiet=False)) == 0
    err = capsys.readouterr().err
    assert err.count("declared a generalization") == 1
    assert "(proactive)" in err


def test_an_advisory_history_failure_keeps_guidance_and_the_run_continues(
    tmp_path, monkeypatch, capsys
):
    """`history-decode-failure` case A."""
    from coding_review_agent_loop import finding_history

    def boom(item):
        raise RuntimeError("projection exploded")

    monkeypatch.setattr(finding_history, "project_finding", boom)
    runner = _history_runner(_step_back_coders())
    config = _sb_config(tmp_path, pr_step_back_rounds=0, quiet=False)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    prompts = _sb_coder_prompts(runner)
    assert all("Proactive generalization" in prompt for prompt in prompts)
    assert all("history is unavailable this turn" in prompt for prompt in prompts)
    assert "finding history unavailable: RuntimeError: projection exploded" in capsys.readouterr().err


def test_ci_repair_dispatch_shows_the_earlier_ci_failure_by_check_name(
    tmp_path, monkeypatch, capsys
):
    """`ci-repair-dispatch`: a CI failure on head A is history when head B fails."""
    runner = FakeRunner(
        codex_outputs=[
            "Fix the application bug.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Still failing CI."
            + prior_item_dispositions("[item-1] resolved", "[item-2] still blocking: the unit-a check still fails in src/app.py:3 and needs a test fix")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "All good."
            + prior_item_dispositions("[item-2] resolved", "[item-3] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Fixed the application and CI.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Fixed lint.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
    )
    counter = iter(range(1, 10))
    original = validated_agent._run_validated_agent

    def run(*args, **kwargs):
        response = original(*args, **kwargs)
        if kwargs.get("role") == "coder":
            runner.pr_payload["headRefOid"] = f"repaired-{next(counter)}"
        return response

    patch_attr_in(monkeypatch, (
        validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage,
        issue_loop,
    ), "_run_validated_agent", run)

    def boards(*_a, **_k):
        head = runner.pr_payload["headRefOid"]
        if head == "repaired-2":
            return _watch_check_board("passing")
        name = "unit-a" if head != "repaired-1" else "lint-b"
        failed = PullRequestCheck(
            name=name, kind="check_run", status="failure", url=f"https://example.test/{name}"
        )
        return _watch_check_board("failing", failing=(failed,))

    monkeypatch.setattr(pr_loop, "get_pr_checks", boards)
    monkeypatch.setattr(pr_loop, "watch_pr_checks", lambda *a, **k: CiWatchOutcome(
        status="passed", pr_checks=_watch_check_board("passing"),
        head_sha=runner.pr_payload["headRefOid"], attempts_used=1
    ))
    assert run_pr_loop(
        runner, pr_number=77,
        config=make_config(tmp_path, watch_pending_ci=True, quiet=False, max_rounds=4),
    ) == 0
    prompts = [
        "\n".join(cmd) for cmd, _cwd in runner.commands if cmd and cmd[0] == "claude"
    ]
    assert len(prompts) == 2
    assert all("Proactive generalization" in prompt for prompt in prompts)
    second = prompts[1]
    history = second.split("Earlier-round history for this run", 1)[1].split("\n\n", 1)[0]
    assert history.count("CI github-pr-checks on") == 1
    # The head-A failure is superseded by the live failure on head B (round 2).
    assert "CI github-pr-checks on abc123 (superseded by the failure on repaired-1)" in history
    assert "Failing checks: unit-a" in history
    assert "lint-b" not in history
    assert "Failing checks: unit-a" in history
    assert "GitHub PR checks are failing" not in history
    assert "Reviewed head" not in history


def test_deferred_finding_stays_deferred_across_ci_repair_dispatches_and_on_replay(
    tmp_path, monkeypatch
):
    """`deferred-status-sticky`: approval defers item-2, CI forces two later dispatches."""
    from coding_review_agent_loop.finding_history import FindingHistoryLedger
    from coding_review_agent_loop.github import get_pr_review_context
    from coding_review_agent_loop.round_state import (
        _extract_round_metadata_records,
        canonical_history_item_outcomes,
    )

    runner = FakeRunner(
        codex_outputs=[
            "Two gaps."
            + blocking_issues("Gap one in src/a.py:10", "Gap two in src/b.py:20")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Defer gap two."
            + prior_item_dispositions(
                "[item-1] resolved", "[item-2] future follow-up: separate follow-up work"
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "Still only CI."
            + prior_item_dispositions("[item-3] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "Done."
            + prior_item_dispositions("[item-3] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Fixed one.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Fixed CI 1.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Fixed CI 2.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
    )
    counter = iter(range(1, 10))
    original = validated_agent._run_validated_agent

    def run(*args, **kwargs):
        response = original(*args, **kwargs)
        if kwargs.get("role") == "coder":
            runner.pr_payload["headRefOid"] = f"repaired-{next(counter)}"
        return response

    patch_attr_in(monkeypatch, (
        validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage,
        issue_loop,
    ), "_run_validated_agent", run)

    def boards(*_a, **_k):
        head = runner.pr_payload["headRefOid"]
        if head == "repaired-3":
            return _watch_check_board("passing")
        if head == "abc123":
            return _watch_check_board("passing")
        failed = PullRequestCheck(
            name=f"check-{head}", kind="check_run", status="failure",
            url=f"https://example.test/{head}",
        )
        return _watch_check_board("failing", failing=(failed,))

    monkeypatch.setattr(pr_loop, "get_pr_checks", boards)
    monkeypatch.setattr(pr_loop, "watch_pr_checks", lambda *a, **k: CiWatchOutcome(
        status="passed", pr_checks=_watch_check_board("passing"),
        head_sha=runner.pr_payload["headRefOid"], attempts_used=1
    ))
    config = make_config(
        tmp_path, watch_pending_ci=True, max_rounds=6, pr_review_context_mode="compact"
    )
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    prompts = [
        "\n".join(cmd) for cmd, _cwd in runner.commands if cmd and cmd[0] == "claude"
    ]
    assert len(prompts) == 3
    for prompt in prompts[1:]:
        assert "Codex finding item-2 (deferred) src/b.py:20" in prompt
    assert "Codex finding item-1 (resolved) src/a.py:10" in prompts[1]
    records = _extract_round_metadata_records(
        get_pr_review_context(runner, config=config, pr_number=77).comments, flow="pr"
    )
    outcomes = canonical_history_item_outcomes(
        records, reconciliation_mode="aggregate", same_status="same-pr"
    )
    assert outcomes.get("item-2") == "deferred"
    assert outcomes.get("item-1") == "resolved"
    resumed = FindingHistoryLedger("pr")
    resumed.seed(records, outcomes=outcomes)
    resumed.observe_reconciled([])
    assert "Codex finding item-2 (deferred)" in resumed.view(99).body


def test_deferred_finding_survives_a_real_resume_and_reintroduction_flips_it_open(
    tmp_path, monkeypatch
):
    """`deferred-status-sticky`: stop after two dispatches, resume run_pr_loop, reintroduce."""
    import dataclasses as dc

    from coding_review_agent_loop.errors import AgentLoopError
    from coding_review_agent_loop.github import get_pr_review_context
    from coding_review_agent_loop.round_state import (
        PostedRoundMetadata,
        PostedRoundRecord,
        _extract_round_metadata_records,
        canonical_history_item_outcomes,
    )

    runner = FakeRunner(
        codex_outputs=[
            "Two gaps."
            + blocking_issues("Gap one in src/a.py:10", "Gap two in src/b.py:20")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Defer gap two."
            + prior_item_dispositions(
                "[item-1] resolved", "[item-2] future follow-up: separate follow-up work"
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "Still only CI."
            + prior_item_dispositions("[item-3] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "Done."
            + prior_item_dispositions("[item-3] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Fixed one.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Fixed CI 1.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Fixed CI 2.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
    )
    counter = iter(range(1, 10))
    original = validated_agent._run_validated_agent

    def run(*args, **kwargs):
        response = original(*args, **kwargs)
        if kwargs.get("role") == "coder":
            runner.pr_payload["headRefOid"] = f"repaired-{next(counter)}"
        return response

    patch_attr_in(monkeypatch, (
        validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage,
        issue_loop,
    ), "_run_validated_agent", run)

    def boards(*_a, **_k):
        head = runner.pr_payload["headRefOid"]
        if head in {"repaired-3", "abc123"}:
            return _watch_check_board("passing")
        failed = PullRequestCheck(
            name=f"check-{head}", kind="check_run", status="failure",
            url=f"https://example.test/{head}",
        )
        return _watch_check_board("failing", failing=(failed,))

    monkeypatch.setattr(pr_loop, "get_pr_checks", boards)
    monkeypatch.setattr(pr_loop, "watch_pr_checks", lambda *a, **k: CiWatchOutcome(
        status="passed", pr_checks=_watch_check_board("passing"),
        head_sha=runner.pr_payload["headRefOid"], attempts_used=1
    ))
    first = make_config(
        tmp_path, watch_pending_ci=True, max_rounds=3, pr_review_context_mode="compact"
    )
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=first)
    claude_after_first = [
        cmd for cmd, _cwd in runner.commands if cmd and cmd[0] == "claude"
    ]
    assert len(claude_after_first) == 2

    second = make_config(
        tmp_path, watch_pending_ci=True, max_rounds=6, pr_review_context_mode="compact"
    )
    assert run_pr_loop(runner, pr_number=77, config=second) == 0
    prompts = [
        "\n".join(cmd) for cmd, _cwd in runner.commands if cmd and cmd[0] == "claude"
    ]
    assert len(prompts) == 3
    # The third dispatch ran in a NEW invocation, seeded only from published records.
    assert "Codex finding item-2 (deferred) src/b.py:20" in prompts[2]
    assert "Codex finding item-1 (resolved) src/a.py:10" in prompts[2]

    records = _extract_round_metadata_records(
        get_pr_review_context(runner, config=second, pr_number=77).comments, flow="pr"
    )
    before = canonical_history_item_outcomes(
        records, reconciliation_mode="aggregate", same_status="same-pr"
    )
    assert before["item-2"] == "deferred"
    item2 = next(
        item
        for record in records
        for item in record.metadata.new_items
        if item.item_id == "item-2"
    )
    reintroduced = PostedRoundRecord(
        index=max(record.index for record in records) + 1,
        metadata=PostedRoundMetadata(
            flow="pr", role="coder", agent="Claude", round_number=99, subject="late",
            prior_items=(dc.replace(item2, status="blocking", source_status="blocking"),),
        ),
        body="",
    )
    after = canonical_history_item_outcomes(
        [*records, reintroduced], reconciliation_mode="aggregate", same_status="same-pr"
    )
    assert after["item-2"] == "active"


def test_future_finding_cleared_in_full_context_mode_is_resolved_live_and_on_resume(
    tmp_path, monkeypatch
):
    """Retained future item resolved by the reconciler while CI forces other dispatches."""
    import dataclasses as dc  # noqa: F401

    from coding_review_agent_loop.errors import AgentLoopError
    from coding_review_agent_loop.github import get_pr_review_context
    from coding_review_agent_loop.round_state import (
        _extract_round_metadata_records,
        canonical_history_item_outcomes,
    )

    runner = FakeRunner(
        codex_outputs=[
            "Two gaps."
            + blocking_issues("Gap one in src/a.py:10", "Gap two in src/b.py:20")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Defer gap two."
            + prior_item_dispositions(
                "[item-1] resolved", "[item-2] future follow-up: separate follow-up work"
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "Gap two was fixed after all."
            + prior_item_dispositions(
                "[item-2] resolved", "[item-3] resolved"
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "Done."
            + prior_item_dispositions("[item-3] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Fixed one.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Fixed CI 1.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Fixed CI 2.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
    )
    counter = iter(range(1, 10))
    original = validated_agent._run_validated_agent

    def run(*args, **kwargs):
        response = original(*args, **kwargs)
        if kwargs.get("role") == "coder":
            runner.pr_payload["headRefOid"] = f"repaired-{next(counter)}"
        return response

    patch_attr_in(monkeypatch, (
        validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage,
        issue_loop,
    ), "_run_validated_agent", run)

    def boards(*_a, **_k):
        head = runner.pr_payload["headRefOid"]
        if head in {"repaired-3", "abc123"}:
            return _watch_check_board("passing")
        failed = PullRequestCheck(
            name=f"check-{head}", kind="check_run", status="failure",
            url=f"https://example.test/{head}",
        )
        return _watch_check_board("failing", failing=(failed,))

    monkeypatch.setattr(pr_loop, "get_pr_checks", boards)
    monkeypatch.setattr(pr_loop, "watch_pr_checks", lambda *a, **k: CiWatchOutcome(
        status="passed", pr_checks=_watch_check_board("passing"),
        head_sha=runner.pr_payload["headRefOid"], attempts_used=1
    ))
    config = make_config(
        tmp_path, watch_pending_ci=True, max_rounds=6, pr_review_context_mode="full"
    )
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    prompts = [
        "\n".join(cmd) for cmd, _cwd in runner.commands if cmd and cmd[0] == "claude"
    ]
    assert len(prompts) == 3

    def history(prompt):
        return prompt.split("Earlier-round history for this run", 1)[1].split("\n\n", 1)[0]

    assert "Codex finding item-2 (deferred) src/b.py:20" in history(prompts[1])
    assert "Codex finding item-2 (resolved) src/b.py:20" in history(prompts[2])
    assert "(deferred)" not in history(prompts[2])
    records = _extract_round_metadata_records(
        get_pr_review_context(runner, config=config, pr_number=77).comments, flow="pr"
    )
    outcomes = canonical_history_item_outcomes(
        records, reconciliation_mode="aggregate", same_status="same-pr"
    )
    assert outcomes["item-2"] == "resolved"


def test_future_finding_reintroduced_as_blocking_in_full_context_mode_is_open_live(
    tmp_path, monkeypatch
):
    """Retained future item resolved by the reconciler while CI forces other dispatches."""
    import dataclasses as dc  # noqa: F401

    from coding_review_agent_loop.errors import AgentLoopError
    from coding_review_agent_loop.github import get_pr_review_context
    from coding_review_agent_loop.round_state import (
        _extract_round_metadata_records,
        canonical_history_item_outcomes,
    )

    runner = FakeRunner(
        codex_outputs=[
            "Two gaps."
            + blocking_issues("Gap one in src/a.py:10", "Gap two in src/b.py:20")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Defer gap two."
            + prior_item_dispositions(
                "[item-1] resolved", "[item-2] future follow-up: separate follow-up work"
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "Gap two matters now."
            + prior_item_dispositions(
                "[item-2] still blocking: src/b.py:20 must be fixed in this PR after all",
                "[item-3] resolved",
            )
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Done."
            + prior_item_dispositions("[item-2] resolved", "[item-3] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Fixed one.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Fixed CI 1.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Fixed CI 2.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
    )
    counter = iter(range(1, 10))
    original = validated_agent._run_validated_agent

    def run(*args, **kwargs):
        response = original(*args, **kwargs)
        if kwargs.get("role") == "coder":
            runner.pr_payload["headRefOid"] = f"repaired-{next(counter)}"
        return response

    patch_attr_in(monkeypatch, (
        validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage,
        issue_loop,
    ), "_run_validated_agent", run)

    def boards(*_a, **_k):
        head = runner.pr_payload["headRefOid"]
        if head in {"repaired-3", "abc123"}:
            return _watch_check_board("passing")
        failed = PullRequestCheck(
            name=f"check-{head}", kind="check_run", status="failure",
            url=f"https://example.test/{head}",
        )
        return _watch_check_board("failing", failing=(failed,))

    monkeypatch.setattr(pr_loop, "get_pr_checks", boards)
    monkeypatch.setattr(pr_loop, "watch_pr_checks", lambda *a, **k: CiWatchOutcome(
        status="passed", pr_checks=_watch_check_board("passing"),
        head_sha=runner.pr_payload["headRefOid"], attempts_used=1
    ))
    config = make_config(
        tmp_path, watch_pending_ci=True, max_rounds=6, pr_review_context_mode="full"
    )
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    prompts = [
        "\n".join(cmd) for cmd, _cwd in runner.commands if cmd and cmd[0] == "claude"
    ]
    assert len(prompts) == 3

    def history(prompt):
        return prompt.split("Earlier-round history for this run", 1)[1].split("\n\n", 1)[0]

    assert "Codex finding item-2 (deferred) src/b.py:20" in history(prompts[1])
    # Authoritative reintroduction: a reviewer blocks the carried future item again.
    assert "Codex finding item-2 (open) src/b.py:20" in history(prompts[2])
    assert "(deferred)" not in history(prompts[2])


def test_future_finding_cleared_in_the_first_resumed_round_is_resolved_not_deferred(
    tmp_path, monkeypatch
):
    """Resume BEFORE clearance (full context): stored ledger carries the future item."""
    runner = FakeRunner(
        codex_outputs=[
            "Two gaps."
            + blocking_issues("Gap one in src/a.py:10", "Gap two in src/b.py:20")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Defer gap two."
            + prior_item_dispositions(
                "[item-1] resolved", "[item-2] future follow-up: separate follow-up work"
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Fixed one.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Fixed CI 1.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
    )
    counter = iter(range(1, 10))
    original = validated_agent._run_validated_agent

    def run(*args, **kwargs):
        response = original(*args, **kwargs)
        if kwargs.get("role") == "coder":
            runner.pr_payload["headRefOid"] = f"repaired-{next(counter)}"
        return response

    patch_attr_in(monkeypatch, (
        validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage,
        issue_loop,
    ), "_run_validated_agent", run)

    def boards(*_a, **_k):
        head = runner.pr_payload["headRefOid"]
        if head in {"repaired-3", "abc123"}:
            return _watch_check_board("passing")
        failed = PullRequestCheck(
            name=f"check-{head}", kind="check_run", status="failure",
            url=f"https://example.test/{head}",
        )
        return _watch_check_board("failing", failing=(failed,))

    monkeypatch.setattr(pr_loop, "get_pr_checks", boards)
    monkeypatch.setattr(pr_loop, "watch_pr_checks", lambda *a, **k: CiWatchOutcome(
        status="passed", pr_checks=_watch_check_board("passing"),
        head_sha=runner.pr_payload["headRefOid"], attempts_used=1
    ))
    config = make_config(
        tmp_path, watch_pending_ci=True, max_rounds=6, pr_review_context_mode="full"
    )
    # First invocation: runs out of scripted reviewer output right after two dispatches.
    with pytest.raises(Exception):
        run_pr_loop(runner, pr_number=77, config=config)
    assert len([c for c, _ in runner.commands if c and c[0] == "claude"]) == 2

    # Second invocation resumes with item-2 still future in the stored ledger; the
    # first resumed review clears it while CI still forces a dispatch.
    runner.codex_outputs.extend([
        "Gap two was fixed."
        + prior_item_dispositions("[item-2] resolved", "[item-3] resolved")
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        "Done."
        + prior_item_dispositions("[item-3] resolved")
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
    ])
    runner.claude_outputs.append(
        "Fixed CI 2.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    prompts = [
        "\n".join(cmd) for cmd, _cwd in runner.commands if cmd and cmd[0] == "claude"
    ]
    assert len(prompts) == 3

    def history(prompt):
        return prompt.split("Earlier-round history for this run", 1)[1].split("\n\n", 1)[0]

    assert "Codex finding item-2 (deferred) src/b.py:20" in history(prompts[1])
    assert "Codex finding item-2 (resolved) src/b.py:20" in history(prompts[2])
    assert "(deferred)" not in history(prompts[2])


def test_future_finding_cleared_by_subitem_completion_is_resolved_live_and_after_resume(
    tmp_path, monkeypatch
):
    """Derived clearance (every sub-item closed) of a retained future finding.

    Full context keeps item-2 in the ledger as `future`.  The first resumed
    review closes all its sub-items with a note-less `blocking` entry, which the
    reconciler turns into a clearance, and raises item-4 so the coder is still
    dispatched.  Both the live dispatch and the dispatch after a further resume
    (whose stored ledger no longer holds item-2) must report it resolved.
    """
    gap_two = {"text": "Gap two in src/b.py:20", "sub_items": ["branch x", "branch y"]}
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking", blocking_items=["Gap one in src/a.py:10", gap_two]
            ),
            structured_pr_review(
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"},
                    {"item_id": "item-2", "disposition": "future", "note": "follow-up work"},
                ],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(addressed_items=["item-1"], remaining_items=["item-2"]),
            structured_coder_followup(summary="Fixed CI.", remaining_items=["item-2", "item-3"]),
        ],
    )
    counter = iter(range(1, 10))
    original = validated_agent._run_validated_agent

    def run(*args, **kwargs):
        response = original(*args, **kwargs)
        if kwargs.get("role") == "coder":
            runner.pr_payload["headRefOid"] = f"repaired-{next(counter)}"
        return response

    patch_attr_in(monkeypatch, (
        validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage,
        issue_loop,
    ), "_run_validated_agent", run)

    def boards(*_a, **_k):
        head = runner.pr_payload["headRefOid"]
        if head != "repaired-1":
            return _watch_check_board("passing")
        failed = PullRequestCheck(
            name=f"check-{head}", kind="check_run", status="failure",
            url=f"https://example.test/{head}",
        )
        return _watch_check_board("failing", failing=(failed,))

    monkeypatch.setattr(pr_loop, "get_pr_checks", boards)
    monkeypatch.setattr(pr_loop, "watch_pr_checks", lambda *a, **k: CiWatchOutcome(
        status="passed", pr_checks=_watch_check_board("passing"),
        head_sha=runner.pr_payload["headRefOid"], attempts_used=1
    ))
    config = make_config(
        tmp_path, watch_pending_ci=True, max_rounds=8, pr_review_context_mode="full"
    )

    def prompts():
        return ["\n".join(cmd) for cmd, _cwd in runner.commands if cmd and cmd[0] == "claude"]

    def history(prompt):
        return prompt.split("Earlier-round history for this run", 1)[1].split("\n\n", 1)[0]

    # First invocation stops when scripted reviewer output runs out.
    with pytest.raises(Exception):
        run_pr_loop(runner, pr_number=77, config=config)
    assert len(prompts()) == 2
    assert "Codex finding item-2 (deferred) src/b.py:20" in history(prompts()[1])

    # Second invocation: the first resumed review closes every sub-item of item-2.
    runner.codex_outputs.append(
        structured_pr_review(
            state="blocking",
            blocking_items=["Gap three in src/c.py:30"],
            prior_item_dispositions=[
                {
                    "item_id": "item-2",
                    "disposition": "blocking",
                    "sub_item_dispositions": {"item-2.s1": "resolved", "item-2.s2": "resolved"},
                },
                {"item_id": "item-3", "disposition": "resolved"},
            ],
        )
    )
    runner.claude_outputs.append(
        structured_coder_followup(remaining_items=["item-3", "item-4"])
    )
    with pytest.raises(Exception):
        run_pr_loop(runner, pr_number=77, config=config)
    assert len(prompts()) == 3
    assert "Codex finding item-2 (resolved) src/b.py:20" in history(prompts()[2])
    assert "(deferred)" not in history(prompts()[2])

    # Third invocation: history is seeded from records whose stored ledger no
    # longer holds item-2, so the replay itself must report the clearance.
    runner.codex_outputs.extend([
        structured_pr_review(
            state="blocking",
            prior_item_dispositions=[
                {"item_id": "item-3", "disposition": "resolved"},
                {"item_id": "item-4", "disposition": "blocking", "note": "still open"},
            ],
        ),
        structured_pr_review(
            prior_item_dispositions=[
                {"item_id": "item-3", "disposition": "resolved"},
                {"item_id": "item-4", "disposition": "resolved"},
            ]
        ),
    ])
    runner.claude_outputs.append(
        structured_coder_followup(remaining_items=["item-3", "item-4"])
    )
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert len(prompts()) == 4
    assert "Codex finding item-2 (resolved) src/b.py:20" in history(prompts()[3])
    assert "(deferred)" not in history(prompts()[3])


def test_compact_resume_cannot_reintroduce_a_deferred_finding_under_its_old_id(
    tmp_path, monkeypatch
):
    """`deferred-status-sticky`, compact mode: why same-ID reintroduction is unreachable.

    Compact reconciliation (`retain_future=False`) moves a deferred item out of
    the ledger, so after a real resume the reviewer cannot disposition it again:
    the authoritative validator treats its ID as an unknown prior item and it
    never re-enters the ledger.  A concern raised again is a NEW finding ID, and
    the deferred one stays deferred in the coder's history.  (Same-ID reintroduction is covered for
    full context, where the item stays in the ledger as `future`.)
    """
    from coding_review_agent_loop.errors import AgentLoopError

    runner = FakeRunner(
        codex_outputs=[
            "Two gaps."
            + blocking_issues("Gap one in src/a.py:10", "Gap two in src/b.py:20")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Defer gap two."
            + prior_item_dispositions(
                "[item-1] resolved", "[item-2] future follow-up: separate follow-up work"
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Fixed one.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Fixed CI 1.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
    )
    counter = iter(range(1, 10))
    original = validated_agent._run_validated_agent

    def run(*args, **kwargs):
        response = original(*args, **kwargs)
        if kwargs.get("role") == "coder":
            runner.pr_payload["headRefOid"] = f"repaired-{next(counter)}"
        return response

    patch_attr_in(monkeypatch, (
        validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage,
        issue_loop,
    ), "_run_validated_agent", run)

    def boards(*_a, **_k):
        head = runner.pr_payload["headRefOid"]
        if head != "repaired-1":
            return _watch_check_board("passing")
        failed = PullRequestCheck(
            name=f"check-{head}", kind="check_run", status="failure",
            url=f"https://example.test/{head}",
        )
        return _watch_check_board("failing", failing=(failed,))

    monkeypatch.setattr(pr_loop, "get_pr_checks", boards)
    monkeypatch.setattr(pr_loop, "watch_pr_checks", lambda *a, **k: CiWatchOutcome(
        status="passed", pr_checks=_watch_check_board("passing"),
        head_sha=runner.pr_payload["headRefOid"], attempts_used=1
    ))
    config = make_config(
        tmp_path, watch_pending_ci=True, max_rounds=8, pr_review_context_mode="compact"
    )
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=config)

    def prompts():
        return ["\n".join(cmd) for cmd, _cwd in runner.commands if cmd and cmd[0] == "claude"]

    def history(prompt):
        return prompt.split("Earlier-round history for this run", 1)[1].split("\n\n", 1)[0]

    assert len(prompts()) == 2
    assert "Codex finding item-2 (deferred) src/b.py:20" in history(prompts()[1])

    # Resumed invocation: the reviewer tries to block item-2 under its old ID and
    # also raises the concern again.  The old ID is not a carried prior item, so
    # it never re-enters the ledger; the concern lives on only as new item-4.
    runner.codex_outputs.extend([
        structured_pr_review(
            state="blocking",
            blocking_items=["Gap two again in src/b.py:20"],
            prior_item_dispositions=[
                {"item_id": "item-2", "disposition": "blocking", "note": "src/b.py:20"},
                {"item_id": "item-3", "disposition": "resolved"},
            ],
        ),
        structured_pr_review(
            prior_item_dispositions=[
                {"item_id": "item-3", "disposition": "resolved"},
                {"item_id": "item-4", "disposition": "resolved"},
            ]
        ),
    ])
    runner.claude_outputs.append(
        structured_coder_followup(addressed_items=["item-4"], remaining_items=["item-3"])
    )
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert len(prompts()) == 3
    resumed_review = next(c for c in runner.comments if "Gap two again" in c)
    assert "[item-2]" not in resumed_review
    assert "Codex finding item-2 (deferred) src/b.py:20" in history(prompts()[2])
    assert "item-4" not in history(prompts()[2])
    assert "Gap two again in src/b.py:20" in prompts()[2]


def test_ci_history_repeated_snapshot_and_older_open_failure_at_later_dispatches(
    tmp_path, monkeypatch
):
    """`ci-repair-dispatch` (ordinary CI) across four coder dispatches.

    Head A (abc123) fails unit-a in round 1; head B (repaired-1) fails lint-b in
    round 2 and, because the second coder turn pushes nothing, again in round 3
    (a repeated snapshot of the same failure); round 4 sees pending checks on a
    new head.  B must stay one instance first observed in round 2, rendered as an
    older OPEN entry at dispatches 3 and 4 (machine qualification is approval
    gated, so reviewer `resolved` votes never clear it), with A superseded.
    """

    def review(*dispositions, blocking=(), state="blocking"):
        return structured_pr_review(
            state=state,
            blocking_items=list(blocking),
            prior_item_dispositions=[
                {"item_id": item, "disposition": verdict}
                | ({"note": "still"} if verdict == "blocking" else {})
                for item, verdict in dispositions
            ],
        )

    runner = FakeRunner(
        codex_outputs=[
            review(blocking=["Bug one in src/app.py:3"]),
            *[review(("item-1", "blocking"), ("item-2", "resolved"))] * 3,
            review(("item-1", "resolved"), ("item-2", "resolved"), state="approved"),
        ],
        claude_outputs=[structured_coder_followup(remaining_items=["item-1", "item-2"])] * 4,
    )
    heads = iter(["repaired-1", "repaired-1", "repaired-3", "repaired-4"])
    original = validated_agent._run_validated_agent

    def run(*args, **kwargs):
        response = original(*args, **kwargs)
        if kwargs.get("role") == "coder":
            runner.pr_payload["headRefOid"] = next(heads)
        return response

    patch_attr_in(monkeypatch, (
        validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage,
        issue_loop,
    ), "_run_validated_agent", run)

    def boards(*_a, **_k):
        head = runner.pr_payload["headRefOid"]
        failing = {"abc123": "unit-a", "repaired-1": "lint-b"}.get(head)
        if head == "repaired-3":
            return _watch_check_board(
                "pending",
                pending=(PullRequestCheck(name="lint-b", kind="check_run", status="pending"),),
            )
        if failing is None:
            return _watch_check_board("passing")
        return _watch_check_board("failing", failing=(PullRequestCheck(
            name=failing, kind="check_run", status="failure",
            url=f"https://example.test/{failing}",
        ),))

    monkeypatch.setattr(pr_loop, "get_pr_checks", boards)
    monkeypatch.setattr(pr_loop, "watch_pr_checks", lambda *a, **k: CiWatchOutcome(
        status="passed", pr_checks=_watch_check_board("passing"),
        head_sha=runner.pr_payload["headRefOid"], attempts_used=1
    ))
    assert run_pr_loop(
        runner, pr_number=77, config=make_config(tmp_path, watch_pending_ci=True, max_rounds=8),
    ) == 0
    prompts = [
        "\n".join(cmd) for cmd, _cwd in runner.commands if cmd and cmd[0] == "claude"
    ]
    assert len(prompts) == 4
    # The lint-b failure on head B was really observed twice (rounds 2 and 3).
    assert sum(
        1 for c in runner.comments
        if c.startswith("GitHub PR checks are failing") and "lint-b" in c
    ) == 2

    def history(prompt):
        return prompt.split("Earlier-round history for this run", 1)[1].split("\n\n", 1)[0]

    def rounds(text):
        return {
            block.split(":", 1)[0]: block for block in re.split(r"\n(?=Round \d+:)", text)
        }

    superseded = "item-2: CI github-pr-checks on abc123 (superseded by the failure on repaired-1)"
    b_open = "item-2: CI github-pr-checks on repaired-1 (open)"
    # Dispatch 2: B is the current round's failure, so only A is history.
    assert superseded in history(prompts[1]) and "repaired-1 (open)" not in history(prompts[1])
    for prompt in prompts[2:]:
        text = history(prompt)
        by_round = rounds(text)
        assert superseded in by_round["Round 1"] and "unit-a (failure)" in by_round["Round 1"]
        # One instance, first observed in round 2, despite the round-3 repeat.
        assert b_open in by_round["Round 2"] and "lint-b (failure)" in by_round["Round 2"]
        assert text.count("CI github-pr-checks on") == 2
        assert "CI github-pr-checks" not in by_round.get("Round 3", "")
        assert "repaired-3" not in text
        assert "GitHub PR checks are failing" not in text


# ---------------------------------------------------------------------------
# PR fix-loop non-convergence stop (#1372)
# ---------------------------------------------------------------------------

NC_A_TEXT = "race: a same-UID attacker swaps the config at src/secure_git.py:10"
NC_B_TEXT = "race: the worktree path is reused at src/run_worktrees.py:900"
NC_MERGE_BASE = "e" * 40


class _NonConvergenceRunner(_StepBackRunner):
    """Step-back runner with scripted ``git diff --shortstat`` and ``git merge-base``."""

    def __init__(
        self, *, shortstat=" 3 files changed, 120 insertions(+), 4 deletions(-)\n",
        git_returncode=0, git_error=None, **kwargs,
    ):
        super().__init__(**kwargs)
        self.shortstat = shortstat
        self.git_returncode = git_returncode
        self.git_error = git_error
        self.measurement_calls = []

    def run(self, args, *, cwd, input_text=None, check=True, env=None):
        cmd = [str(arg) for arg in args]
        shortstat = cmd[:3] == ["git", "diff", "--shortstat"]
        # Only the measurement's ``origin/<base>`` lookup is scripted; the plain-base
        # fallback and unrelated merge-base calls keep the fake's empty default.
        merge_base = cmd[:2] == ["git", "merge-base"] and cmd[2:3] == ["origin/main"]
        if shortstat or merge_base:
            self.measurement_calls.append(cmd)
            if self.git_error is not None:
                raise self.git_error
            cmd, cwd_path = self._record_command(cmd, cwd)
            stdout = self.shortstat if shortstat else f"{NC_MERGE_BASE}\n"
            return CommandResult(cmd, cwd_path, stdout, "", self.git_returncode)
        return super().run(args, cwd=cwd, input_text=input_text, check=check, env=env)


def _nc_reviews(rounds=4, *, review=_sb_review):
    return [review([NC_A_TEXT, NC_B_TEXT])] + [
        review(carried=["item-1", "item-2"]) for _ in range(rounds - 1)
    ]


def _nc_coders(count=3):
    return [structured_coder_followup(addressed_items=["item-1", "item-2"]) for _ in range(count)]


def _nc_records(runner, config):
    from coding_review_agent_loop.github import get_pr_review_context
    from coding_review_agent_loop.round_state import _extract_round_metadata_records

    return _extract_round_metadata_records(
        get_pr_review_context(runner, config=config, pr_number=77).comments, flow="pr"
    )


def _nc_stop(runner, config):
    with pytest.raises(HumanDecisionRequiredError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=config)
    return str(excinfo.value)


def test_items_persisting_across_files_stop_before_a_fourth_coder_turn(tmp_path):
    """`nc-persist-spread`, `nc-production-numbering`."""
    runner = _NonConvergenceRunner(claude_outputs=_nc_coders(), codex_outputs=_nc_reviews())
    config = _sb_config(tmp_path)
    message = _nc_stop(runner, config)
    assert "human decision required: Codex kept the same item(s) blocking across 3" in message
    assert "not converging" in message and "No further coder turn was invoked" in message
    assert f"[item-1] (attempts: 3) {NC_A_TEXT}" in message
    assert f"[item-2] (attempts: 3) {NC_B_TEXT}" in message
    assert "reviews at rounds 1-4" in message
    assert "3 files changed, 120 insertions(+), 4 deletions(-)" in message
    assert "unavailable" not in message
    assert "`--pr-non-convergence-rounds 0`" in message
    assert len(_sb_coder_prompts(runner)) == 3  # no fourth coder turn
    assert all("STEP-BACK TURN" not in p for p in _sb_coder_prompts(runner))
    records = _nc_records(runner, config)
    assert [r.metadata.round_number for r in records if r.metadata.role == "reviewer"] == [1, 2, 3, 4]
    coder_records = [r for r in records if r.metadata.role == "coder"]
    # Production numbering: review r is followed by coder record r+1.
    assert [r.metadata.round_number for r in coder_records] == [2, 3, 4]
    assert not any(r.metadata.step_back_entries for r in coder_records)
    # The growth is measured from the first reviewed head, the size from the merge base.
    diffs = [c for c in runner.measurement_calls if c[:3] == ["git", "diff", "--shortstat"]]
    assert diffs[0][3] == SB_HEAD and diffs[-1][3] == NC_MERGE_BASE


def test_the_live_dispatch_snapshot_lacks_the_newest_coder_record_and_still_stops(
    tmp_path, monkeypatch
):
    """`nc-live-snapshot`: the pre-check passes on the snapshot's reviews alone."""
    import coding_review_agent_loop.pr_loop_support as support

    seen = []
    original = support._pr_step_back_records_may_matter

    def spy(records, tracked, **kwargs):
        result = original(records, tracked, **kwargs)
        seen.append(
            (
                kwargs["round_number"],
                sorted(r.metadata.round_number for r in records if r.metadata.role == "coder"),
                result,
            )
        )
        return result

    monkeypatch.setattr(support, "_pr_step_back_records_may_matter", spy)
    runner = _NonConvergenceRunner(claude_outputs=_nc_coders(), codex_outputs=_nc_reviews())
    _nc_stop(runner, _sb_config(tmp_path, pr_step_back_rounds=0))
    round_four = [entry for entry in seen if entry[0] == 4]
    assert round_four and round_four[-1][2] is True
    assert 4 not in round_four[-1][1]  # coder record 4 was posted after the snapshot
    assert len(_sb_coder_prompts(runner)) == 3


def test_a_rerun_after_the_stop_stops_again_with_identical_text_and_no_new_state(tmp_path):
    """`nc-resume`."""
    runner = _NonConvergenceRunner(claude_outputs=_nc_coders(), codex_outputs=_nc_reviews())
    config = _sb_config(tmp_path)
    first = _nc_stop(runner, config)
    comments_before = len(_nc_records(runner, config))
    second = _nc_stop(runner, config)
    assert second == first
    assert len(_nc_records(runner, config)) == comments_before
    assert len(_sb_coder_prompts(runner)) == 3


def test_resume_after_an_interrupted_kth_review_reaches_the_same_stop(tmp_path):
    """`nc-resume`: the run is interrupted before the K-th review, then rerun."""
    runner = _NonConvergenceRunner(claude_outputs=_nc_coders(), codex_outputs=_nc_reviews(3))
    config = _sb_config(tmp_path)
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert len(_sb_coder_prompts(runner)) == 3
    runner.codex_outputs.extend(_nc_reviews(4)[3:])
    resumed = _nc_stop(runner, config)
    live_runner = _NonConvergenceRunner(claude_outputs=_nc_coders(), codex_outputs=_nc_reviews())
    live = _nc_stop(live_runner, _sb_config(tmp_path / "live"))
    assert resumed == live
    assert len(_sb_coder_prompts(runner)) == 3


def test_disabling_step_back_keeps_the_non_convergence_stop(tmp_path):
    """`nc-flag-independence`."""
    runner = _NonConvergenceRunner(claude_outputs=_nc_coders(), codex_outputs=_nc_reviews())
    message = _nc_stop(runner, _sb_config(tmp_path, pr_step_back_rounds=0))
    assert "not converging" in message
    assert len(_sb_coder_prompts(runner)) == 3


def test_disabling_non_convergence_restores_the_prior_behaviour(tmp_path):
    """`nc-flag-independence`: the loop keeps dispatching the coder."""
    runner = _NonConvergenceRunner(
        claude_outputs=_nc_coders(4),
        codex_outputs=[*_nc_reviews(), _sb_review(resolved=["item-1", "item-2"])],
    )
    config = _sb_config(tmp_path, pr_non_convergence_rounds=0)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert len(_sb_coder_prompts(runner)) == 4
    assert not runner.measurement_calls


def test_non_convergence_stops_under_review_parallel(tmp_path):
    """`nc-review-parallel`: publication records resolve through reconciliation."""
    runner = _NonConvergenceRunner(claude_outputs=_nc_coders(), codex_outputs=_nc_reviews())
    config = _sb_config(tmp_path, review_parallel=True)
    message = _nc_stop(runner, config)
    assert "[item-1] (attempts: 3)" in message and "[item-2] (attempts: 3)" in message
    assert any(
        r.metadata.phase == "publication" for r in _nc_records(runner, config)
        if r.metadata.role == "reviewer"
    )
    assert len(_sb_coder_prompts(runner)) == 3


def test_primary_then_panel_tracks_only_the_primary(tmp_path):
    """`nc-primary-then-panel`: the paused panel is not needed to trigger."""
    runner = _NonConvergenceRunner(
        claude_outputs=_nc_coders(),
        codex_outputs=_nc_reviews(),
        gemini_outputs=[_gemini_review(resolved=["item-1", "item-2"])],
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), max_rounds=8,
        pr_review_policy="primary-then-panel", primary_reviewer="codex",
    )
    message = _nc_stop(runner, config)
    assert "Codex kept the same item(s) blocking" in message
    assert not _sb_gemini_prompts(runner)
    assert len(_sb_coder_prompts(runner)) == 3


def test_non_convergence_wins_over_a_fresh_clustered_step_back(tmp_path):
    """`nc-precedence`: both qualify at round 3; no step-back entry is recorded."""
    runner = _NonConvergenceRunner(
        claude_outputs=[
            structured_coder_followup(addressed_items=["item-1"]),
            structured_coder_followup(addressed_items=["item-1", "item-2"]),
        ],
        codex_outputs=[
            _sb_review(["gap one at src/spool.py:124-131"]),
            _sb_review(["gap two at src/spool.py:131-136"], carried=["item-1"]),
            _sb_review(["gap three at src/spool.py:140"], carried=["item-1", "item-2"]),
        ],
    )
    config = _sb_config(tmp_path, pr_non_convergence_rounds=2)
    message = _nc_stop(runner, config)
    assert "[item-1] (attempts: 2)" in message and "item-3" not in message
    assert len(_sb_coder_prompts(runner)) == 2
    assert all("STEP-BACK TURN" not in p for p in _sb_coder_prompts(runner))
    assert not any(
        r.metadata.step_back_entries for r in _nc_records(runner, config)
        if r.metadata.role == "coder"
    )


def test_a_clustered_sibling_escalation_stops_before_non_convergence(tmp_path):
    """`nc-precedence`: a pending sibling escalation is reported first."""
    runner = _NonConvergenceRunner(
        claude_outputs=_step_back_coders(),
        codex_outputs=[
            *_three_clustered_blocks(),
            _sb_review(["another branch at src/spool.py:134"], carried=["item-3"]),
        ],
    )
    with pytest.raises(HumanDecisionRequiredError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=_sb_config(tmp_path, pr_non_convergence_rounds=1))
    message = str(excinfo.value)
    assert "still raised a new sibling finding" in message
    assert "not converging" not in message
    assert len(_sb_coder_prompts(runner)) == 3


@pytest.mark.parametrize(
    "runner_kwargs",
    [
        {"git_returncode": 128},
        {"shortstat": "fatal: bad revision\n"},
        {"git_error": OSError("git is missing")},
        {"git_error": AgentLoopError("runner refused")},
    ],
    ids=["non-zero", "unparseable", "oserror", "agent-loop-error"],
)
def test_diff_measurement_failures_render_unavailable_and_still_stop(tmp_path, runner_kwargs):
    """`nc-diff-failure`: the runner glue never raises a non-decision error."""
    runner = _NonConvergenceRunner(
        claude_outputs=_nc_coders(), codex_outputs=_nc_reviews(), **runner_kwargs
    )
    message = _nc_stop(runner, _sb_config(tmp_path))
    assert "Diff growth since the first attempt" in message
    growth = message.split("Diff growth since the first attempt", 1)[1].splitlines()[0]
    size = message.split("Total PR size", 1)[1].splitlines()[0]
    assert growth.endswith("unavailable") and size.endswith("unavailable")
    assert runner.measurement_calls
    assert len(_sb_coder_prompts(runner)) == 3


def test_diff_measurement_failure_through_the_dispatch_decision_directly(tmp_path):
    """`nc-diff-failure`: drive ``_pr_step_back_decision`` with a failing runner."""
    import coding_review_agent_loop.pr_loop_support as support
    from coding_review_agent_loop.github import get_pr_review_context

    runner = _NonConvergenceRunner(claude_outputs=_nc_coders(), codex_outputs=_nc_reviews())
    config = _sb_config(tmp_path)
    _nc_stop(runner, config)
    comments = get_pr_review_context(runner, config=config, pr_number=77).comments
    head = runner.pr_payload["headRefOid"]
    for failure in ({"git_returncode": 1}, {"git_error": OSError("boom")}):
        runner.git_returncode = failure.get("git_returncode", 0)
        runner.git_error = failure.get("git_error")
        with pytest.raises(HumanDecisionRequiredError) as excinfo:
            support._pr_step_back_decision(
                runner, config, pr_number=77, round_number=4, snapshot_comments=comments,
                tracked=("Codex",), head_sha=head, base_branch="main",
            )
        assert str(excinfo.value).count("unavailable") == 2
    assert len(_sb_coder_prompts(runner)) == 3


def test_enabling_the_gate_on_longer_history_measures_growth_from_the_first_attempt(tmp_path):
    """The attempt scan reaches past K, and the growth baseline follows it."""
    runner = _NonConvergenceRunner(claude_outputs=_nc_coders(4), codex_outputs=_nc_reviews(5))
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_pr_loop(runner, pr_number=77, config=_sb_config(tmp_path, pr_non_convergence_rounds=0))
    assert len(_sb_coder_prompts(runner)) == 5  # the fifth dispatch ran out of script
    runner.measurement_calls.clear()
    message = _nc_stop(runner, _sb_config(tmp_path, pr_non_convergence_rounds=3))
    assert "[item-1] (attempts: 4)" in message and "reviews at rounds 1-5" in message
    assert f"(`{SB_HEAD[:12]}` to" in message
    growth = [c for c in runner.measurement_calls if c[:3] == ["git", "diff", "--shortstat"]][0]
    assert growth[3] == SB_HEAD and growth[4] == runner.pr_payload["headRefOid"]
    assert len(_sb_coder_prompts(runner)) == 5


# --- PR step-back approach alternative (#1628) -------------------------------


def _approach_alternative_coders():
    return [
        structured_coder_followup(addressed_items=[f"item-{n}"]) for n in (1, 2)
    ] + [
        structured_coder_followup(
            addressed_items=["item-3"],
            summary="Approach alternative: drop the predecessor-head disqualifier.",
        )
    ]


def test_an_approach_alternative_step_back_stops_live_for_a_human_decision(tmp_path):
    """`approach-alt-live-stop`."""
    runner = _StepBackRunner(
        claude_outputs=_approach_alternative_coders(),
        codex_outputs=[*_three_clustered_blocks(), _sb_review(resolved=["item-3"])],
    )
    with pytest.raises(HumanDecisionRequiredError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=_sb_config(tmp_path))
    message = str(excinfo.value)
    assert "the coder proposes replacing the approved approach" in message
    assert "drop the predecessor-head disqualifier." in message
    assert "rerun with the scope change approved" in message
    assert "accept a stated limitation" in message and "--pr-step-back-rounds 0" in message
    assert len(_sb_coder_prompts(runner)) == 3  # no further coder turn
    assert len(_sb_codex_prompts(runner)) == 3  # no sweep review of the proposal head
    assert not any("gh workflow run" in " ".join(cmd) for cmd, _cwd in runner.commands)
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _cwd in runner.commands)


def test_a_resumed_run_stops_before_reviewing_the_approach_alternative_head(tmp_path):
    """`approach-alt-resume-stop`: the record was posted, the live stop never ran."""
    runner = _StepBackRunner(
        claude_outputs=_approach_alternative_coders(),
        codex_outputs=_three_clustered_blocks(),
    )
    with patch.object(pr_loop, "_pr_step_back_approach_alternative_stop"):
        with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
            run_pr_loop(runner, pr_number=77, config=_sb_config(tmp_path))
    assert len(_sb_coder_prompts(runner)) == 3
    reviews_before = len(_sb_codex_prompts(runner))  # includes the failed sweep attempt
    runner.codex_outputs.append(_sb_review(resolved=["item-3"]))
    with pytest.raises(HumanDecisionRequiredError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=_sb_config(tmp_path))
    assert "drop the predecessor-head disqualifier." in str(excinfo.value)
    assert len(_sb_coder_prompts(runner)) == 3
    assert len(_sb_codex_prompts(runner)) == reviews_before  # no review of that head
    # Continuing to generalize disables the stop and reviews the head.
    assert run_pr_loop(
        runner, pr_number=77, config=_sb_config(tmp_path, pr_step_back_rounds=0)
    ) == 0
    assert len(_sb_codex_prompts(runner)) == reviews_before + 1


def test_a_generalization_step_back_still_runs_the_sweep_review(tmp_path):
    """`generalization-unchanged`."""
    runner = _StepBackRunner(
        claude_outputs=_step_back_coders(),
        codex_outputs=[*_three_clustered_blocks(), _sb_review(resolved=["item-3"])],
    )
    assert run_pr_loop(runner, pr_number=77, config=_sb_config(tmp_path)) == 0
    assert "Enumerate ALL remaining instances" in _sb_codex_prompts(runner)[3]


def test_a_sibling_escalation_lists_the_scope_change_route(tmp_path):
    """`generalization-unchanged`: the sibling stop keeps its prefix and adds the route."""
    _runner, message = _run_to_sibling(tmp_path, "another branch at src/spool.py:134")
    assert "human decision required: the coder generalized a class of findings" in message
    assert "approve a scope change (re-plan or amend the issue)" in message
    assert "re-offer" not in message


def test_disabled_step_back_never_stops_on_an_approach_alternative_summary(tmp_path):
    """`approach-alt-malformed`: step-back rounds 0 never stops."""
    runner = _StepBackRunner(
        claude_outputs=_approach_alternative_coders(),
        codex_outputs=[*_three_clustered_blocks(), _sb_review(resolved=["item-3"])],
    )
    config = _sb_config(tmp_path, pr_step_back_rounds=0)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0


# --- Durable PR human-decision diagnostics (#1640) ---------------------------


_DIAGNOSTIC_LEAD = "PR review scheduling diagnostic ("


def _hd_diagnostics(runner):
    return [
        comment["body"] for comment in runner.pr_payload.get("comments", [])
        if comment["body"].startswith(_DIAGNOSTIC_LEAD)
    ]


def _assert_one_hd_diagnostic(runner, message, *, label, reason):
    from coding_review_agent_loop.history_export import (
        KIND_HUMAN_DECISION_STOP,
        recognize_diagnostic,
    )
    from coding_review_agent_loop.round_transport import ROUND_RESUME_MARKER_RE

    diagnostics = _hd_diagnostics(runner)
    assert diagnostics == [pr_loop.render_pr_human_decision_diagnostic(label, message)]
    assert not ROUND_RESUME_MARKER_RE.search(diagnostics[0])  # audit text, never state
    recognized = recognize_diagnostic(diagnostics[0])
    assert recognized is not None
    assert (recognized.kind, recognized.reason, recognized.terminal) == (
        KIND_HUMAN_DECISION_STOP, reason, True,
    )
    return diagnostics[0]


def test_a_sibling_escalation_posts_one_bounded_scheduling_diagnostic(tmp_path):
    """`sibling-stop-posts`."""
    runner, message = _run_to_sibling(tmp_path, "another branch at src/spool.py:134")
    _assert_one_hd_diagnostic(runner, message, label="round 4", reason="pr-step-back-sibling")
    assert len(_sb_coder_prompts(runner)) == 3  # no coder turn after the stop


def test_an_immediate_non_convergence_stop_posts_one_scheduling_diagnostic(tmp_path):
    """`nonconv-immediate-posts`."""
    runner = _NonConvergenceRunner(claude_outputs=_nc_coders(), codex_outputs=_nc_reviews())
    config = _sb_config(tmp_path)
    message = _nc_stop(runner, config)
    _assert_one_hd_diagnostic(runner, message, label="round 4", reason="pr-non-convergence")
    assert len(_sb_coder_prompts(runner)) == 3
    # The plain diagnostic never breaks round-record decoding on a rerun.
    assert _nc_stop(runner, config) == message
    assert len(_hd_diagnostics(runner)) == 2  # one per invocation stop


def test_a_live_approach_alternative_stop_posts_one_scheduling_diagnostic(tmp_path):
    """`approach-live-order` (non-managed path)."""
    runner = _StepBackRunner(
        claude_outputs=_approach_alternative_coders(),
        codex_outputs=[*_three_clustered_blocks(), _sb_review(resolved=["item-3"])],
    )
    with pytest.raises(HumanDecisionRequiredError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=_sb_config(tmp_path))
    _assert_one_hd_diagnostic(
        runner, str(excinfo.value), label="round 3",
        reason="pr-step-back-approach-alternative",
    )
    assert len(_sb_codex_prompts(runner)) == 3


def test_a_resumed_approach_alternative_stop_posts_a_new_diagnostic_from_round_records(tmp_path):
    """`approach-resume-posts`: an earlier diagnostic is on the thread; the rerun
    re-derives the stop from round records and posts one more."""
    from coding_review_agent_loop.history_export import recognize_diagnostic

    runner = _StepBackRunner(
        claude_outputs=_approach_alternative_coders(),
        codex_outputs=[*_three_clustered_blocks(), _sb_review(resolved=["item-3"])],
    )
    config = _sb_config(tmp_path)
    with pytest.raises(HumanDecisionRequiredError) as live:
        run_pr_loop(runner, pr_number=77, config=config)
    assert len(_hd_diagnostics(runner)) == 1
    reviews_before = len(_sb_codex_prompts(runner))
    with pytest.raises(HumanDecisionRequiredError) as resumed:
        run_pr_loop(runner, pr_number=77, config=config)
    assert str(resumed.value) == str(live.value)
    diagnostics = _hd_diagnostics(runner)
    assert len(diagnostics) == 2
    assert recognize_diagnostic(diagnostics[1]).reason == "pr-step-back-approach-alternative"
    assert len(_sb_codex_prompts(runner)) == reviews_before  # no reviewer invocation
    assert len(_sb_coder_prompts(runner)) == 3
    assert _nc_records(runner, config)  # records still decode around the diagnostics


class _ProxyTempfile:
    """``tempfile`` stand-in whose handle fails to write the diagnostic body."""

    def __init__(self, real, *, fail_on):
        self._real = real
        self._fail_on = fail_on

    def __getattr__(self, name):
        return getattr(self._real, name)

    def NamedTemporaryFile(self, *args, **kwargs):  # noqa: N802 - mirrors tempfile
        inner = self._real.NamedTemporaryFile(*args, **kwargs)
        fail_on = self._fail_on

        class _Handle:
            name = inner.name

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return inner.__exit__(*exc)

            def write(self, text):
                if text.startswith(fail_on):
                    raise OSError(28, "No space left on device")
                return inner.write(text)

        return _Handle()


def _failing_diagnostic_post(error):
    """``post_pr_comment`` that fails only for the human-decision diagnostic."""
    real_post = pr_loop.post_pr_comment

    def post(runner, *, config, pr_number, body, **kwargs):
        if isinstance(body, str) and body.startswith(_DIAGNOSTIC_LEAD):
            raise error
        return real_post(runner, config=config, pr_number=pr_number, body=body, **kwargs)

    return post


def _capture_step_back_decision_error(monkeypatch):
    raised = []
    original = pr_loop._pr_step_back_decision

    def spy(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except HumanDecisionRequiredError as exc:
            raised.append(exc)
            raise

    monkeypatch.setattr(pr_loop, "_pr_step_back_decision", spy)
    return raised


@pytest.mark.parametrize("failure", ["github", "transport-overflow", "temp-file"])
def test_a_failed_diagnostic_post_keeps_the_original_human_decision_stop(
    tmp_path, monkeypatch, capsys, failure
):
    """`post-failure-exit4`: the original exception object propagates unchanged."""
    import tempfile

    import coding_review_agent_loop.github as github_module
    from coding_review_agent_loop.round_transport import RoundCommentOverflowError

    reference_runner, reference = _run_to_sibling(
        tmp_path / "reference", "another branch at src/spool.py:134"
    )
    raised = _capture_step_back_decision_error(monkeypatch)
    if failure == "temp-file":
        monkeypatch.setattr(
            github_module, "tempfile", _ProxyTempfile(tempfile, fail_on=_DIAGNOSTIC_LEAD)
        )
    else:
        error = (
            AgentLoopError("gh pr comment failed: HTTP 502")
            if failure == "github"
            else RoundCommentOverflowError("GitHub comment body exceeds 60000 characters")
        )
        monkeypatch.setattr(pr_loop, "post_pr_comment", _failing_diagnostic_post(error))
    runner, message = _run_to_sibling(
        tmp_path / "failing", "another branch at src/spool.py:134",
        config=_sb_config(tmp_path / "failing", quiet=False),
    )
    assert message == reference
    assert raised and isinstance(raised[-1], HumanDecisionRequiredError)
    assert len(_hd_diagnostics(runner)) == 0
    assert len(_sb_coder_prompts(runner)) == 3
    assert "could not post the human-decision diagnostic" in capsys.readouterr().err
    assert len(_hd_diagnostics(reference_runner)) == 1


def test_a_failed_post_reraises_the_same_exception_object(tmp_path, monkeypatch):
    """`post-failure-exit4`: identity, not only an equal message."""
    raised = _capture_step_back_decision_error(monkeypatch)

    monkeypatch.setattr(
        pr_loop, "post_pr_comment", _failing_diagnostic_post(OSError(5, "Input/output error"))
    )
    runner = _StepBackRunner(
        claude_outputs=_step_back_coders(),
        codex_outputs=[
            *_three_clustered_blocks(),
            _sb_review(["another branch at src/spool.py:134"], resolved=["item-3"]),
        ],
    )
    with pytest.raises(HumanDecisionRequiredError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=_sb_config(tmp_path))
    assert excinfo.value is raised[-1]


def test_the_cli_keeps_exit_four_when_the_diagnostic_post_fails(tmp_path, monkeypatch, capsys):
    """`post-failure-exit4`: exit code and CLI text are unchanged."""
    import coding_review_agent_loop.cli as cli_module

    monkeypatch.setattr(
        pr_loop, "post_pr_comment",
        _failing_diagnostic_post(AgentLoopError("gh pr comment failed: HTTP 502")),
    )
    runner = _NonConvergenceRunner(claude_outputs=_nc_coders(), codex_outputs=_nc_reviews())
    config = _sb_config(tmp_path)

    def drive_the_fixture(*_args, **_kwargs):
        return run_pr_loop(runner, pr_number=77, config=config)

    with patch.object(cli_module, "run_pr_loop", side_effect=drive_the_fixture):
        assert cli_module.main(["pr", "77", "--repo", "OWNER/REPO"]) == 4
    stderr = capsys.readouterr().err
    assert "HUMAN DECISION REQUIRED" in stderr
    assert "not converging" in stderr and "--pr-non-convergence-rounds 0" in stderr


def test_an_oversized_non_convergence_stop_posts_a_bounded_diagnostic(tmp_path, monkeypatch):
    """`oversize-diagnostic-bounded`: through the real post and transport path."""
    import coding_review_agent_loop.review_step_back as step_back_module
    from coding_review_agent_loop.history_export import recognize_diagnostic
    from coding_review_agent_loop.round_transport import MAX_GITHUB_BODY_CHARS

    real_render = step_back_module.render_pr_non_convergence_decision
    extra = "\n".join(
        f"- [item-{n}] (attempts: 3) " + "persisting detail " * 12 for n in range(3, 400)
    )

    def many_items(*args, **kwargs):
        rendered = real_render(*args, **kwargs)
        head, sep, tail = rendered.partition("\nDiff growth")
        return head + "\n" + extra + sep + tail

    monkeypatch.setattr(step_back_module, "render_pr_non_convergence_decision", many_items)
    runner = _NonConvergenceRunner(claude_outputs=_nc_coders(), codex_outputs=_nc_reviews())
    message = _nc_stop(runner, _sb_config(tmp_path))
    assert len(pr_loop.render_pr_scheduling_diagnostic("round 4", message)) > MAX_GITHUB_BODY_CHARS
    assert message.endswith("blocking finding.")  # the exception text is never bounded
    (posted,) = _hd_diagnostics(runner)
    assert posted == pr_loop.render_pr_human_decision_diagnostic("round 4", message)
    assert len(posted) <= pr_loop.PR_HUMAN_DECISION_DIAGNOSTIC_LIMIT
    first_line = message.split("\n", 1)[0]
    assert posted.startswith(pr_loop.render_pr_scheduling_diagnostic("round 4", first_line + "\n"))
    assert posted.endswith("the full text is in the CLI output of this run)")
    assert recognize_diagnostic(posted).reason == "pr-non-convergence"


def test_disabled_gates_post_no_human_decision_diagnostic(tmp_path):
    """`gate-disabled-silent`."""
    runner = _NonConvergenceRunner(
        claude_outputs=_nc_coders(4),
        codex_outputs=[*_nc_reviews(), _sb_review(resolved=["item-1", "item-2"])],
    )
    assert run_pr_loop(
        runner, pr_number=77, config=_sb_config(tmp_path, pr_non_convergence_rounds=0)
    ) == 0
    assert _hd_diagnostics(runner) == []
    runner = _StepBackRunner(
        claude_outputs=_approach_alternative_coders(),
        codex_outputs=[*_three_clustered_blocks(), _sb_review(resolved=["item-3"])],
    )
    assert run_pr_loop(
        runner, pr_number=77, config=_sb_config(tmp_path / "sb", pr_step_back_rounds=0)
    ) == 0
    assert _hd_diagnostics(runner) == []
