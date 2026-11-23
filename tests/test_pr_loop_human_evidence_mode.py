"""PR-loop integration tests for ``--pr-human-evidence freeze|advisory|off`` (#2063)."""

import dataclasses

import pytest

import coding_review_agent_loop.human_evidence_mode as human_evidence_mode
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.issue_recovery as issue_recovery
import coding_review_agent_loop.plan_execution as plan_execution
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
from coding_review_agent_loop.cli import AgentLoopError, run_pr_loop
from coding_review_agent_loop.config import AgentLoopConfig
from coding_review_agent_loop.errors import HumanDecisionRequiredError
from coding_review_agent_loop.github import _parse_pr_human_requirements
from coding_review_agent_loop.human_evidence_mode import HUMAN_CHECKS_FLOW, HumanCheck
from coding_review_agent_loop.round_state import _extract_round_metadata_records
from agent_loop_helpers import FakeRunner, make_config, patch_attr_in, structured_coder_followup
from pr_loop_test_helpers import (
    _EVIDENCE_TEXT,
    _association,
    _classified_review,
    _evidence_review,
    _freeze_records,
    _pr_records,
    _release_records,
    _signed,
)
from types import SimpleNamespace

HEAD = "6367c48dd193d56ea7b0baad25b19455e529f5ee"
DEFAULT_MODE = AgentLoopConfig.__dataclass_fields__["pr_human_evidence"].default
_CONTEXT_MODULES = (pr_loop_support, pr_loop, issue_implementation, issue_loop, issue_recovery, plan_execution)


def _config(tmp_path, **overrides):
    """The shared fixture pins freeze; drop the pin to get the production default."""
    mode = overrides.pop("pr_human_evidence", DEFAULT_MODE)
    return dataclasses.replace(make_config(tmp_path, **overrides), pr_human_evidence=mode)


def _check_comments(runner):
    return [c["body"] for c in runner.pr_payload["comments"]
            if c["body"].startswith("## Human checks requested")]


def _check_records(runner):
    return _extract_round_metadata_records(
        [SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]], flow=HUMAN_CHECKS_FLOW)


def _merge_calls(runner):
    return [cmd for cmd, _cwd in runner.commands if cmd[:3] == ["gh", "pr", "merge"]]


def _reviewer_calls(runner):
    return [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["codex"]]


def _coder_calls(runner):
    return [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]


def _evidence_items(items):
    return [item for item in items if item.obligation_kind == "human-exact-head-evidence"]


def _captured_log(monkeypatch):
    lines = []
    monkeypatch.setattr(human_evidence_mode, "log", lambda _config, message: lines.append(message))
    return lines


# --- advisory approval outcome --------------------------------------------------------


def test_default_mode_with_evidence_request_and_auto_merge_leaves_pr_approved_and_unmerged(
    tmp_path, capsys, monkeypatch
):
    """Row advisory-default-auto-merge (issue acceptance 1)."""
    completions = []
    monkeypatch.setattr(pr_loop, "_record_staged_parent_completion_after_merge",
                        lambda *a, **k: completions.append("staged"))
    runner = FakeRunner(codex_outputs=[_evidence_review(evidence=[_EVIDENCE_TEXT])])
    config = _config(tmp_path, auto_merge=True)
    assert config.pr_human_evidence == "advisory"
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert _freeze_records(runner) == []
    assert _merge_calls(runner) == [] and completions == []
    bodies = _check_comments(runner)
    assert len(bodies) == 1 and _EVIDENCE_TEXT in bodies[0] and HEAD in bodies[0]
    records = _check_records(runner)
    assert [r.metadata.subject for r in records] == [HEAD]
    [check] = [HumanCheck.from_mapping(entry) for entry in records[0].metadata.human_checks]
    assert check.text == _EVIDENCE_TEXT and check.kind == "native" and check.reviewer == "Codex"
    out = capsys.readouterr().out
    assert f"PR #77 approved but not merged: 1 human check(s) requested; see the 'Human checks requested'" in out
    assert "merged" not in out.replace("not merged", "")
    # No blocking evidence obligation was ever persisted on a PR record ledger.
    assert not any(_evidence_items(r.metadata.prior_items) for r in _pr_records(runner))


def test_advisory_without_auto_merge_posts_list_and_takes_approval_exit(tmp_path, capsys):
    """Row advisory-approval-variants (request, no --auto-merge)."""
    runner = FakeRunner(codex_outputs=[_evidence_review(evidence=[_EVIDENCE_TEXT])])
    assert run_pr_loop(runner, pr_number=77, config=_config(tmp_path, auto_merge=False)) == 0
    assert len(_check_comments(runner)) == 1 and _freeze_records(runner) == []
    out = capsys.readouterr().out
    assert "PR #77 approved by" in out and "approved but not merged" not in out


def test_advisory_empty_list_with_auto_merge_merges_as_today(tmp_path, capsys):
    """Row advisory-approval-variants (no request, --auto-merge)."""
    runner = FakeRunner(codex_outputs=[_evidence_review()])
    assert run_pr_loop(runner, pr_number=77, config=_config(tmp_path, auto_merge=True)) == 0
    assert _merge_calls(runner) and _check_comments(runner) == []
    assert "approved but not merged" not in capsys.readouterr().out


def test_advisory_finalization_blocker_fails_closed_before_list_or_exit(tmp_path, monkeypatch):
    """Row advisory-finalization-blocker."""
    def blocked(**kwargs):
        raise AgentLoopError("uncleared machine revalidation candidate blocks finalization")

    monkeypatch.setattr(pr_loop, "_ensure_finalization_ready", blocked)
    runner = FakeRunner(codex_outputs=[_evidence_review(evidence=[_EVIDENCE_TEXT])])
    with pytest.raises(AgentLoopError, match="uncleared machine revalidation candidate"):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path, auto_merge=True))
    assert _check_comments(runner) == [] and _merge_calls(runner) == []


# --- freeze and off --------------------------------------------------------------------


def test_explicit_freeze_reproduces_todays_freeze_record_and_stop(tmp_path):
    """Row freeze-mode-unchanged (issue acceptance 2)."""
    runner = FakeRunner(codex_outputs=[_evidence_review(evidence=[_EVIDENCE_TEXT])])
    with pytest.raises(HumanDecisionRequiredError, match=f"frozen at head {HEAD}") as excinfo:
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path, auto_merge=True, pr_human_evidence="freeze"))
    assert excinfo.value.EXIT_CODE == 4
    [freeze] = _freeze_records(runner)
    assert [item.text for item in _evidence_items(freeze.metadata.prior_items)] == [_EVIDENCE_TEXT]
    assert _check_comments(runner) == [] and _merge_calls(runner) == []
    assert not any(r.metadata.human_checks for r in _pr_records(runner))


def test_freeze_mode_prompt_text_is_unchanged_and_advisory_says_non_blocking():
    from coding_review_agent_loop import prompts

    assert prompts.pr_review_evidence_request_guidance(SimpleNamespace(pr_human_evidence="freeze")) \
        is prompts.PR_REVIEW_EVIDENCE_REQUEST_GUIDANCE
    advisory = prompts.pr_review_evidence_request_guidance(SimpleNamespace(pr_human_evidence="advisory"))
    assert "non-blocking human checks" in advisory and "Signed-source classification is not requested" in advisory
    assert "ignored" in prompts.pr_review_evidence_request_guidance(SimpleNamespace(pr_human_evidence="off"))


def test_off_mode_drops_requests_with_a_log_line_and_merges(tmp_path, monkeypatch, capsys):
    """Row off-mode-drops."""
    lines = _captured_log(monkeypatch)
    runner = FakeRunner(codex_outputs=[_evidence_review(evidence=[_EVIDENCE_TEXT])])
    assert run_pr_loop(runner, pr_number=77, config=_config(tmp_path, auto_merge=True, pr_human_evidence="off")) == 0
    assert any("dropped 1 exact-head evidence request(s) (--pr-human-evidence off)" in line for line in lines)
    assert _merge_calls(runner) and _check_comments(runner) == [] and _freeze_records(runner) == []
    assert not any(_evidence_items(r.metadata.prior_items) or r.metadata.human_checks for r in _pr_records(runner))
    assert "approved but not merged" not in capsys.readouterr().out


# --- signed-source classification bypass ------------------------------------------------


@pytest.mark.parametrize("mode", ["advisory", "off"])
def test_inconsistent_classification_causes_no_reask_or_stop_outside_freeze(tmp_path, mode):
    """Row advisory-signed-classification-bypass (live validation and publication)."""
    runner = FakeRunner()
    runner.pr_payload["comments"].append(_signed("A human must accept the policy at this exact head.", 1))
    rid = _parse_pr_human_requirements(runner.pr_payload)[0].requirement_id
    # A human-only role naming no emitted request: freeze re-asks; here it is ignored.
    runner.codex_outputs = [_classified_review(
        [_association(rid, "human-only", request="Unemitted request")], hr=True, evidence=[_EVIDENCE_TEXT])]
    assert run_pr_loop(runner, pr_number=77, config=_config(tmp_path, pr_human_evidence=mode)) == 0
    assert runner.codex_outputs == [] and len(_reviewer_calls(runner)) == 1
    assert _freeze_records(runner) == []
    reviewer = [r for r in _pr_records(runner) if r.metadata.role == "reviewer"][-1]
    assert reviewer.metadata.requirement_evidence == ()
    assert len(_check_comments(runner)) == (1 if mode == "advisory" else 0)


def test_freeze_mode_still_reasks_the_same_inconsistent_classification(tmp_path):
    """Contrast for advisory-signed-classification-bypass: freeze keeps validation."""
    runner = FakeRunner()
    runner.pr_payload["comments"].append(_signed("A human must accept the policy at this exact head.", 1))
    rid = _parse_pr_human_requirements(runner.pr_payload)[0].requirement_id
    bad = _classified_review([_association(rid, "human-only", request="Unemitted request")],
                             hr=True, evidence=[_EVIDENCE_TEXT])
    runner.codex_outputs = [bad, bad, bad]
    with pytest.raises((AgentLoopError, HumanDecisionRequiredError)):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path, pr_human_evidence="freeze"))
    assert len(_reviewer_calls(runner)) >= 2


# --- resume and migration ----------------------------------------------------------------


def _frozen(tmp_path, **runner_kwargs):
    runner = FakeRunner(codex_outputs=[_evidence_review(evidence=[_EVIDENCE_TEXT])], **runner_kwargs)
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path, pr_human_evidence="freeze"))
    assert len(_freeze_records(runner)) == 1
    return runner


@pytest.mark.parametrize("auto_merge", [True, False])
def test_frozen_pr_resumed_in_advisory_releases_without_reviewers(tmp_path, capsys, auto_merge):
    """Row resume-frozen-into-advisory (issue acceptance 3)."""
    runner = _frozen(tmp_path)
    reviewers_before = len(_reviewer_calls(runner))
    capsys.readouterr()
    assert run_pr_loop(runner, pr_number=77, config=_config(tmp_path, auto_merge=auto_merge)) == 0
    assert len(_reviewer_calls(runner)) == reviewers_before and _coder_calls(runner) == []
    [release] = _release_records(runner)
    assert release.metadata.evidence_release.reason == "advisory-mode"
    assert release.metadata.subject == HEAD
    assert "Exact-head evidence freeze released (advisory mode)" in release.body
    assert "'Human checks requested' list" in release.body
    assert not _evidence_items(release.metadata.prior_items)
    assert [HumanCheck.from_mapping(e).text for e in release.metadata.human_checks] == [_EVIDENCE_TEXT]
    assert len(_freeze_records(runner)) == 1
    bodies = _check_comments(runner)
    assert len(bodies) == 1 and _EVIDENCE_TEXT in bodies[0]
    assert _merge_calls(runner) == []
    out = capsys.readouterr().out
    assert ("approved but not merged" in out) is auto_merge


def test_frozen_pr_with_failing_check_resumed_in_advisory_takes_machine_gate_release(tmp_path):
    """Row resume-frozen-machine-gate."""
    runner = _frozen(tmp_path, claude_outputs=[structured_coder_followup(addressed_items=["item-2"])])
    runner.pr_check_runs_payload = {
        "check_runs": [{"name": "test", "status": "completed", "conclusion": "failure"}]}
    runner.codex_outputs.append(_evidence_review(
        dispositions=[{"item_id": "item-2", "disposition": "resolved", "note": "Checks fixed."}]))
    real_run = runner.run

    def checks_recover_after_coder(args, **kwargs):
        if _coder_calls(runner):
            runner.pr_check_runs_payload = {
                "check_runs": [{"name": "test", "status": "completed", "conclusion": "success"}]}
        return real_run(args, **kwargs)

    runner.run = checks_recover_after_coder
    assert run_pr_loop(runner, pr_number=77, config=_config(tmp_path, max_rounds=3)) == 0
    releases = _release_records(runner)
    assert [r.metadata.evidence_release.reason for r in releases] == ["machine-gate"]
    assert not _evidence_items(releases[0].metadata.prior_items)
    assert {i.obligation_kind for i in releases[0].metadata.prior_items} >= {"github-pr-checks"}
    assert [HumanCheck.from_mapping(e).text for e in releases[0].metadata.human_checks] == [_EVIDENCE_TEXT]
    assert len(_coder_calls(runner)) == 1 and len(_freeze_records(runner)) == 1
    assert _EVIDENCE_TEXT in _check_comments(runner)[-1]


def test_malformed_persisted_freeze_still_blocks_in_advisory(tmp_path):
    """Row malformed-evidence-record."""
    from test_pr_loop_evidence_freeze import _retag_freeze_budget

    runner = _frozen(tmp_path)
    _retag_freeze_budget(runner, evidence_identities=("not-the-frozen-identity",))
    with pytest.raises(AgentLoopError, match="malformed or contradictory persisted exact-head evidence"):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path, auto_merge=True))
    assert _merge_calls(runner) == [] and _release_records(runner) == []


def test_advisory_list_survives_restart_without_reposting(tmp_path):
    """Row advisory-resume-idempotent."""
    runner = FakeRunner(codex_outputs=[_evidence_review(evidence=[_EVIDENCE_TEXT])])
    config = _config(tmp_path, auto_merge=True)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert len(_check_comments(runner)) == 1
    runner.codex_outputs.append(_evidence_review())
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert len(_check_comments(runner)) == 1 and _merge_calls(runner) == []


@pytest.mark.parametrize("mode", ["advisory", "off"])
def test_interrupted_ordinary_review_resumed_without_freeze_is_normalized(tmp_path, mode, monkeypatch):
    """Row advisory-ordinary-resume-normalized."""
    code = "Guard the pre-spawn git status call."
    runner = FakeRunner(codex_outputs=[_evidence_review(state="blocking", blocking=[code], evidence=[_EVIDENCE_TEXT])])
    # Interrupted after the reviews: the round limit stops before any coder or freeze.
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path, max_rounds=1, pr_human_evidence="freeze"))
    assert _freeze_records(runner) == []
    assert any(r.metadata.evidence_requests == (_EVIDENCE_TEXT,) for r in _pr_records(runner))
    lines = _captured_log(monkeypatch)
    seen = []
    real_ready = pr_loop._ensure_finalization_ready

    def ready(**kwargs):
        seen.append(_evidence_items(kwargs["items"]))
        return real_ready(**kwargs)

    monkeypatch.setattr(pr_loop, "_ensure_finalization_ready", ready)
    runner.codex_outputs.append(_evidence_review(
        dispositions=[{"item_id": "item-1", "disposition": "resolved", "note": "Fixed."}]))
    runner.claude_outputs.append(structured_coder_followup(addressed_items=["item-1"]))
    assert run_pr_loop(runner, pr_number=77, config=_config(tmp_path, max_rounds=3, pr_human_evidence=mode)) == 0
    assert _freeze_records(runner) == [] and seen and not any(seen)
    coder_records = [r for r in _pr_records(runner) if r.metadata.role == "coder"]
    assert coder_records and not any(_evidence_items(r.metadata.prior_items) for r in coder_records)
    bodies = _check_comments(runner)
    if mode == "advisory":
        assert len(bodies) == 1 and bodies[0].count(_EVIDENCE_TEXT) == 1
    else:
        assert bodies == [] and any("dropped" in line for line in lines)


def test_advisory_list_is_restored_as_obligations_when_resumed_in_freeze(tmp_path):
    """Row advisory-to-freeze-carried (native request)."""
    runner = FakeRunner(codex_outputs=[_evidence_review(evidence=[_EVIDENCE_TEXT])])
    assert run_pr_loop(runner, pr_number=77, config=_config(tmp_path)) == 0
    [check] = [HumanCheck.from_mapping(e) for e in _check_records(runner)[0].metadata.human_checks]
    runner.codex_outputs.append(_evidence_review())
    with pytest.raises(HumanDecisionRequiredError, match=f"frozen at head {HEAD}"):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path, pr_human_evidence="freeze"))
    [freeze] = _freeze_records(runner)
    [item] = _evidence_items(freeze.metadata.prior_items)
    assert item.item_id == check.origin_item.item_id
    assert item.obligation_identity == check.origin_item.obligation_identity
    assert item.resolution_owners == ("Codex",) and item.reviewer == "Codex"


def test_head_moving_before_finalization_in_advisory_reviews_the_new_head(tmp_path, monkeypatch):
    """Row advisory-head-revalidation (fresh run, head moves at the gate read)."""
    runner = FakeRunner(codex_outputs=[_evidence_review(evidence=[_EVIDENCE_TEXT])])
    real_context = pr_loop_support.get_observed_pr_review_context
    moved = []
    new_head = "0b3d8b29493059afd7f9912106279c4643ac4939"

    def head_moves_before_gate(runner_, **kwargs):
        if not moved and not runner.codex_outputs:
            moved.append(True)
            runner.pr_payload["headRefOid"] = new_head
            runner.git_head = new_head
        return real_context(runner_, **kwargs)

    patch_attr_in(monkeypatch, _CONTEXT_MODULES, "get_observed_pr_review_context", head_moves_before_gate)
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path, max_rounds=1, auto_merge=True))
    assert moved and _merge_calls(runner) == []
    assert not any(HEAD in body for body in _check_comments(runner))


def test_new_signed_input_before_finalization_in_advisory_refreshes_same_head(tmp_path, monkeypatch):
    """Row advisory-head-revalidation (new signed requirement after the surfaced baseline)."""
    from test_pr_loop_evidence_freeze import _inject_signed_input_before

    runner = FakeRunner(codex_outputs=[
        _evidence_review(evidence=[_EVIDENCE_TEXT]),
        _evidence_review(hr=True),
    ])
    injected = _inject_signed_input_before(
        monkeypatch, runner, body="Also keep the legacy flag.", when=lambda: len(runner.codex_outputs) == 1)
    assert run_pr_loop(runner, pr_number=77, config=_config(tmp_path, auto_merge=True)) == 0
    assert injected and runner.codex_outputs == [] and len(_reviewer_calls(runner)) == 2
    assert _merge_calls(runner) == [] and len(_check_comments(runner)) == 1


def test_migrated_freeze_with_new_signed_input_refreshes_before_approval(tmp_path):
    """Row advisory-head-revalidation (freeze migration with signed input after the freeze)."""
    runner = _frozen(tmp_path)
    runner.pr_payload["comments"].append(_signed("Also keep the legacy flag.", 3))
    runner.codex_outputs.append(_evidence_review(hr=True))
    reviewers_before = len(_reviewer_calls(runner))
    assert run_pr_loop(runner, pr_number=77, config=_config(tmp_path, auto_merge=True)) == 0
    assert [r.metadata.evidence_release.reason for r in _release_records(runner)][0] == "advisory-mode"
    # The migrated approval did not cover the new signed input: one refresh review ran.
    assert len(_reviewer_calls(runner)) == reviewers_before + 1
    assert _merge_calls(runner) == [] and len(_check_comments(runner)) == 1


def test_stall_decision_and_operator_conversion_are_bypassed_outside_freeze(tmp_path, monkeypatch):
    """Row advisory-stall-conversion-bypass (dispatch path and config validation)."""
    def no_stall(*args, **kwargs):
        raise AssertionError("the evidence-stall decision must not run outside freeze")

    monkeypatch.setattr(pr_loop, "_pr_evidence_stall_decision", no_stall)
    code = "Guard the pre-spawn git status call."
    runner = FakeRunner(
        codex_outputs=[
            _evidence_review(state="blocking", blocking=[code], evidence=[_EVIDENCE_TEXT]),
            _evidence_review(dispositions=[{"item_id": "item-1", "disposition": "resolved", "note": "Fixed."}]),
        ],
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
    )
    assert run_pr_loop(runner, pr_number=77, config=_config(tmp_path, max_rounds=3)) == 0
    assert len(_coder_calls(runner)) == 1 and len(_check_comments(runner)) == 1
    with pytest.raises(AgentLoopError, match="--pr-convert-evidence-rows requires --pr-human-evidence freeze"):
        _config(tmp_path, pr_convert_evidence_rows=("row-a",))
    with pytest.raises(AgentLoopError, match="requires --pr-human-evidence freeze"):
        make_config(tmp_path, pr_convert_evidence_rows=("row-a",), pr_human_evidence="advisory")


# --- screenshot coverage is a separate channel ------------------------------------------

from test_pr_loop_review_images import visual_setup  # noqa: E402,F401 - pytest fixture
from test_review_images_coverage import responder, typed_review, records as visual_records  # noqa: E402


@pytest.mark.parametrize("mode", ["advisory", "off"])
def test_screenshot_only_request_still_freezes_outside_freeze_mode(visual_setup, monkeypatch, mode):
    """Row screenshot-channel-preserved (screenshot-only pending request at finalization)."""
    config, _images, _ = visual_setup
    config = dataclasses.replace(config, pr_human_evidence=mode)
    responder(monkeypatch, accepted=False)
    runner = FakeRunner(codex_outputs=[typed_review()])
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    freezes = [r for r in visual_records(runner) if r.metadata.evidence_freeze]
    assert len(freezes) == 1
    frozen = _evidence_items(freezes[0].metadata.prior_items)
    assert frozen and all(item.screenshot_request for item in frozen)
    assert _check_comments(runner) == []
    assert not any(r.metadata.human_checks for r in visual_records(runner))


@pytest.mark.parametrize("mode", ["advisory", "off"])
def test_mixed_freeze_is_not_released_by_the_mode_change(visual_setup, monkeypatch, mode):
    """Row screenshot-channel-preserved (mixed freeze resumed outside freeze mode)."""
    config, _images, _ = visual_setup
    responder(monkeypatch, accepted=False)
    runner = FakeRunner(codex_outputs=[typed_review(legacy=["Operator must verify the unrelated audit"])])
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    [freeze] = [r for r in visual_records(runner) if r.metadata.evidence_freeze]
    kinds = {bool(item.screenshot_request) for item in _evidence_items(freeze.metadata.prior_items)}
    assert kinds == {True, False}
    seen = []
    real_normalize = human_evidence_mode.HumanEvidenceRun.normalize

    def spy(self, items, **kwargs):
        result = real_normalize(self, items, **kwargs)
        seen.append(([c.text for c in self.checks], [i for i in _evidence_items(result)]))
        return result

    monkeypatch.setattr(human_evidence_mode.HumanEvidenceRun, "normalize", spy)
    runner.simulate_new_process()
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=dataclasses.replace(config, pr_human_evidence=mode))
    assert not [r for r in visual_records(runner) if r.metadata.evidence_release]
    checks, remaining = seen[-1]
    assert remaining and all(item.screenshot_request for item in remaining)
    assert checks == (["Operator must verify the unrelated audit"] if mode == "advisory" else [])


def test_persisted_human_only_classification_does_not_route_the_source_away_in_advisory(tmp_path, capsys):
    """Row advisory-signed-classification-bypass (historical routing into a reviewer-skipping migration)."""
    runner = FakeRunner()
    runner.pr_payload["comments"].append(_signed("A human must accept the policy at this exact head.", 1))
    rid = _parse_pr_human_requirements(runner.pr_payload)[0].requirement_id
    runner.codex_outputs = [_classified_review(
        [_association(rid, "human-only", request=_EVIDENCE_TEXT)], evidence=[_EVIDENCE_TEXT])]
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path, pr_human_evidence="freeze"))
    [freeze] = _freeze_records(runner)
    reviewer = [r for r in _pr_records(runner) if r.metadata.role == "reviewer"][-1]
    assert [e["role"] for e in reviewer.metadata.requirement_evidence] == ["human-only"]
    seen = []
    real_reconcile = pr_loop._reconcile_human_requirements_ack_item

    def spy(items, **kwargs):
        seen.append((tuple(r.requirement_id for r in kwargs["human_requirements"]),
                     tuple(kwargs.get("reviewer_owned_requirement_ids", ()))))
        return real_reconcile(items, **kwargs)

    patcher = pytest.MonkeyPatch()
    patcher.setattr(pr_loop, "_reconcile_human_requirements_ack_item", spy)
    try:
        assert run_pr_loop(runner, pr_number=77, config=_config(tmp_path, auto_merge=True)) == 0
    finally:
        patcher.undo()
    assert _EVIDENCE_TEXT in _check_comments(runner)[-1]
    assert "approved but not merged" in capsys.readouterr().out
    # The migration released the freeze without a classification stop, and the
    # persisted human-only association did not make the source reviewer-owned.
    assert [r.metadata.evidence_release.reason for r in _release_records(runner)][:1] == ["advisory-mode"]
    assert seen and seen[0] == ((rid,), ())
    assert _merge_calls(runner) == []
