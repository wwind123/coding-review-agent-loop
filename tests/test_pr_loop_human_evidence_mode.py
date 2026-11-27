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
from coding_review_agent_loop.human_evidence_mode import HUMAN_CHECKS_FLOW, HumanCheck, HumanEvidenceRun
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


# --- review round 1 fixes ------------------------------------------------------------------


def _recovery_capability():
    from coding_review_agent_loop.managed_ci_types import OrdinaryRecoveryCapability

    return OrdinaryRecoveryCapability(pr_number=77, repository="OWNER/REPO", base_ref="main",
                                      expected_head_sha=HEAD, released_label_event_id=101, released_at=100)


@pytest.mark.parametrize("outcome", ["pending", "failed", "not_started", "non_authoritative", "passed"])
def test_ordinary_recovery_runs_the_advisory_hold_only_after_authoritative_recovery_ci(
    tmp_path, monkeypatch, outcome
):
    """Item-1: the advisory hold never pre-empts ordinary-recovery CI validation."""
    from coding_review_agent_loop.github import PullRequestCheck, PullRequestMetadata
    from coding_review_agent_loop.managed_ci_types import ManagedCiOutcome
    from pr_loop_test_helpers import _watch_check_board

    capability = _recovery_capability()
    board = (_watch_check_board("passing", passing=(PullRequestCheck(name="test", kind="check_run", status="neutral"),))
             if outcome == "non_authoritative" else _watch_check_board("passing"))
    status = "passed" if outcome in {"passed", "non_authoritative"} else outcome
    monkeypatch.setattr(pr_loop_support, "refresh_ordinary_recovery_capability", lambda *a, **k: capability)
    monkeypatch.setattr(pr_loop_support, "wait_for_ordinary_recovery",
                        lambda *a, **k: ManagedCiOutcome(status=status, checks=board, head_sha=HEAD))
    patch_attr_in(monkeypatch, _CONTEXT_MODULES, "get_observed_pr_review_context",
                  lambda *a, **k: SimpleNamespace(metadata=PullRequestMetadata(
                      number=77, repo="OWNER/REPO", title="draft", head_branch="feature",
                      base_branch="main", head_sha=HEAD, url=None)))
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *a, **k: pytest.fail("advisory hold must not merge"))
    holds = []
    runner = FakeRunner()

    def finalize():
        return pr_loop_support._finalize_ordinary_recovery_checked(
            runner, config=_config(tmp_path, auto_merge=True), pr_number=77, round_number=1, items=[],
            current_head_sha=HEAD, capability=capability, hold=lambda: holds.append(True) or True)

    if outcome in {"pending", "failed"}:
        with pytest.raises(AgentLoopError, match="did not qualify the exact head"):
            finalize()
    else:
        assert finalize() is False
    assert holds == ([True] if outcome == "passed" else [])
    assert not any(cmd[:3] in (["gh", "pr", "ready"], ["gh", "pr", "merge"]) for cmd, _cwd in runner.commands)


@pytest.mark.parametrize("mode", ["advisory", "off"])
def test_finalization_gate_never_reopens_a_converted_tombstone_outside_freeze(tmp_path, mode):
    """Item-2: a cleared converted row at an earlier head proceeds instead of freezing."""
    from test_pr_loop_evidence_conversion import START_HEAD, _comment, _plan_context
    from coding_review_agent_loop.evidence_conversion import ConversionAuditRow, EvidenceConversionAudit
    from coding_review_agent_loop.round_metadata import PostedRoundMetadata, _attach_round_metadata
    from coding_review_agent_loop.unresolved_items import _upsert_converted_evidence_obligation

    context = _plan_context("row-a")
    _ledger, obligation, _ = _upsert_converted_evidence_obligation(
        [], item_number=2, row_id="row-a", matrix_identity=context.risk_test_matrix_identity,
        head_sha="old-head", owners=("Codex",), source_round=2, explanation="")
    audit = EvidenceConversionAudit(
        source="operator", matrix_identity=context.risk_test_matrix_identity, head="old-head",
        rows=(ConversionAuditRow(row_id="row-a", obligation_identity=obligation.obligation_identity,
                                 obligation_item_id="item-2", owners=("Codex",), converted_item_ids=(),
                                 attempt_record_rounds=()),))
    conversion = PostedRoundMetadata(
        flow="pr", role="summary", agent="Orchestrator", round_number=2, subject="old-head",
        prior_items=(obligation,), state="blocking", phase="evidence-conversion", evidence_conversion=audit)
    runner = FakeRunner(pr_payload={"comments": [_comment(_attach_round_metadata("Converted.", conversion))]})
    stale = dataclasses.replace(obligation, status="resolved", lifecycle="cleared", candidate_head_sha="old-head")
    items = [stale]
    outcome = pr_loop_support._evidence_freeze_gate(
        runner, config=_config(tmp_path, pr_human_evidence=mode), pr_number=77, round_number=3,
        head_sha=START_HEAD, items=items, surfaced_requirement_ids=(), collect_requirement_ids=lambda _ctx: (),
        allowed_rounds=8, watch_failure_extension_used=False, watch_head_extension_used=False,
        approved_plan_context=context)
    assert outcome.action == "proceed" and items == [stale]
    assert _freeze_records(runner) == []


def test_native_evidence_reaching_the_freeze_write_outside_freeze_fails_closed(tmp_path):
    """Item-2 generalization: no native human evidence freeze can be written outside freeze mode."""
    from coding_review_agent_loop.unresolved_items import _upsert_evidence_obligation

    items, _ = _upsert_evidence_obligation([], item_number=1, reviewer="Codex", text=_EVIDENCE_TEXT,
                                           source_round=1, current_head_sha=HEAD)
    runner = FakeRunner()
    with pytest.raises(AgentLoopError, match="outside --pr-human-evidence freeze"):
        pr_loop_support._evidence_freeze_gate(
            runner, config=_config(tmp_path), pr_number=77, round_number=1, head_sha=HEAD, items=items,
            surfaced_requirement_ids=(), collect_requirement_ids=lambda _ctx: (), allowed_rounds=5,
            watch_failure_extension_used=False, watch_head_extension_used=False)
    assert _freeze_records(runner) == []


def test_more_than_fifty_requests_survive_restart_and_restoration_losslessly(tmp_path):
    """Item-3: the durable list is lossless; only the rendered comment is bounded."""
    texts = [f"Run live scenario {n} against authenticated CLIs at this head." for n in range(60)]
    runner = FakeRunner(codex_outputs=[_evidence_review(evidence=texts)])
    assert run_pr_loop(runner, pr_number=77, config=_config(tmp_path)) == 0
    [record] = _check_records(runner)
    assert sorted(HumanCheck.from_mapping(e).text for e in record.metadata.human_checks) == sorted(texts)
    assert "and 10 more" in _check_comments(runner)[0]
    runner.codex_outputs.append(_evidence_review())
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path, pr_human_evidence="freeze"))
    [freeze] = _freeze_records(runner)
    assert sorted(i.text for i in _evidence_items(freeze.metadata.prior_items)) == sorted(texts)


@pytest.mark.parametrize("bad_kind", [[], {}, 7, None])
def test_malformed_kind_is_dropped_through_full_round_metadata_decoding(bad_kind):
    """Item-4: a non-string kind drops only that entry, never the record."""
    from coding_review_agent_loop.round_metadata import (
        PostedRoundMetadata, _decode_round_metadata, _encode_round_metadata)
    from coding_review_agent_loop.unresolved_items import _upsert_evidence_obligation

    items, _ = _upsert_evidence_obligation([], item_number=1, reviewer="Codex", text=_EVIDENCE_TEXT,
                                           source_round=1, current_head_sha=HEAD)
    good = HumanCheck("Codex", _EVIDENCE_TEXT, 1, HEAD, "native", items[0]).as_dict()
    metadata = PostedRoundMetadata(flow="pr", role="summary", agent="Orchestrator", round_number=1,
                                   subject=HEAD, human_checks=(good, {**good, "kind": bad_kind}))
    decoded = _decode_round_metadata(_encode_round_metadata(metadata))
    assert decoded.human_checks == (good,)


# --- item-5 workflow coverage -----------------------------------------------------------


def _malformed_association_review(**kwargs):
    import json

    raw, footer = _evidence_review(**kwargs).split("\n<!--", 1)
    payload = json.loads(raw)
    payload["requirement_evidence"] = [{"requirement_id": "R1"}]  # malformed shape
    return json.dumps(payload) + "\n<!--" + footer


@pytest.mark.parametrize("mode", ["advisory", "off"])
def test_published_malformed_association_is_reparsed_on_same_round_resume_without_reask(
    tmp_path, monkeypatch, mode
):
    """Item-5.s1: same-round resume reparses the canonical response with associations ignored."""
    from pr_loop_test_helpers import _InterruptedAfterReviews

    runner = FakeRunner(
        codex_outputs=[_malformed_association_review(evidence=[_EVIDENCE_TEXT])],
        gemini_outputs=[_evidence_review(reviewer="Google Gemini")],
    )
    config = _config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True, pr_human_evidence=mode)
    real = pr_loop.post_pr_comment

    def post(*args, **kwargs):
        real(*args, **kwargs)
        if len([r for r in _pr_records(runner) if r.metadata.role == "reviewer"]) == 2:
            raise _InterruptedAfterReviews()

    with monkeypatch.context() as patch:
        patch.setattr(pr_loop, "post_pr_comment", post)
        with pytest.raises(_InterruptedAfterReviews):
            run_pr_loop(runner, pr_number=77, config=config)
    codex_record = next(r for r in _pr_records(runner) if r.metadata.agent == "Codex")
    assert '"requirement_id": "R1"' in codex_record.metadata.canonical_reviewer_response
    reparsed = []
    # Resume rebuilds published reviews through the shared reconstruction helper (#2064).
    from coding_review_agent_loop import round_closure_writer
    real_parse = round_closure_writer.parse_structured_pr_review

    def spy(text, **kwargs):
        result = real_parse(text, **kwargs)
        reparsed.append((kwargs.get("parse_requirement_evidence"), result.requirement_evidence if result else None))
        return result

    monkeypatch.setattr(round_closure_writer, "parse_structured_pr_review", spy)
    calls_before = len(_reviewer_calls(runner))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert reparsed and all(flag is False and evidence == () for flag, evidence in reparsed)
    assert len(_reviewer_calls(runner)) == calls_before  # no re-ask or fresh review
    assert _freeze_records(runner) == []
    assert len(_check_comments(runner)) == (1 if mode == "advisory" else 0)


@pytest.mark.parametrize("mode", ["advisory", "off"])
def test_closed_incomplete_round_evidence_is_normalized_through_rollover(tmp_path, monkeypatch, mode):
    """Item-5.s2: closure restoration and completed-review replay never reach dispatch or finalization."""
    from coding_review_agent_loop.round_closure import ROUND_CLOSED_INCOMPLETE_PHASE, build_round_closure_payload
    from coding_review_agent_loop.round_metadata import PostedRoundMetadata, _attach_round_metadata
    from pr_loop_test_helpers import _CLOSURE_HEAD, _append_orchestrator_record, _InterruptedAfterReviews

    runner = FakeRunner(
        codex_outputs=[_evidence_review(evidence=[_EVIDENCE_TEXT])],
        gemini_outputs=[_evidence_review(reviewer="Google Gemini")],
    )
    config = _config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True, pr_human_evidence=mode)
    real = pr_loop.post_pr_comment

    def post(*args, **kwargs):
        real(*args, **kwargs)
        if len([r for r in _pr_records(runner) if r.metadata.role == "reviewer"]) == 2:
            raise _InterruptedAfterReviews()

    with monkeypatch.context() as patch:
        patch.setattr(pr_loop, "post_pr_comment", post)
        with pytest.raises(_InterruptedAfterReviews):
            run_pr_loop(runner, pr_number=77, config=config)
    payload = build_round_closure_payload(
        flow="pr", round_number=1, subject=_CLOSURE_HEAD,
        published=[("Codex", None), ("Gemini", None)],
        unreplayable=[("Gemini", "published review could not be replayed")],
        evidence_obligations=[{"reviewer": "Codex", "source_round": 1, "head": _CLOSURE_HEAD,
                               "text": _EVIDENCE_TEXT, "item_number": 1}],
        item_number_high_water=1,
    )
    assert payload is not None
    _append_orchestrator_record(runner, _attach_round_metadata("Review round 1 closed, incomplete.", PostedRoundMetadata(
        flow="pr", role="summary", agent="Orchestrator", round_number=1, subject=_CLOSURE_HEAD,
        phase=ROUND_CLOSED_INCOMPLETE_PHASE, round_closure=payload)))
    lines = _captured_log(monkeypatch)
    seen = []
    real_ready = pr_loop._ensure_finalization_ready

    def ready(**kwargs):
        seen.append(_evidence_items(kwargs["items"]))
        return real_ready(**kwargs)

    monkeypatch.setattr(pr_loop, "_ensure_finalization_ready", ready)
    runner.codex_outputs.append(_evidence_review(evidence=[_EVIDENCE_TEXT]))
    runner.gemini_outputs.append(_evidence_review(reviewer="Google Gemini"))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert seen and not any(seen) and _freeze_records(runner) == []
    later = [r for r in _pr_records(runner) if r.metadata.round_number >= 2]
    assert later and not any(_evidence_items(r.metadata.prior_items) for r in later)
    bodies = _check_comments(runner)
    if mode == "advisory":
        assert len(bodies) == 1 and bodies[0].count(_EVIDENCE_TEXT) == 1
    else:
        assert bodies == [] and any("dropped" in line for line in lines)


def test_carried_earlier_round_request_is_restored_in_freeze(tmp_path):
    """Item-5.s3 (carried approval): an earlier-round advisory request returns as its obligation."""
    code = "Guard the pre-spawn git status call."
    runner = FakeRunner(
        codex_outputs=[_evidence_review(evidence=[_EVIDENCE_TEXT])],
        gemini_outputs=[_evidence_review(state="blocking", blocking=[code], reviewer="Google Gemini")],
    )
    two = dict(reviewer=("codex", "gemini"))
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path, max_rounds=1, **two))
    fixed = {"item_id": "item-1", "disposition": "resolved", "note": "Fixed."}
    keep = {"item_id": "item-2", "disposition": "blocking", "note": "Human evidence still needed."}
    runner.codex_outputs.extend([_evidence_review(state="blocking", dispositions=[fixed, keep])] * 2)
    runner.gemini_outputs.extend([_evidence_review(reviewer="Google Gemini", dispositions=[
        fixed, {**keep, "disposition": "resolved", "note": "Not my request."}])] * 2)
    runner.claude_outputs.append(structured_coder_followup(addressed_items=["item-1"]))
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head"):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path, max_rounds=4, pr_human_evidence="freeze", **two))
    [freeze] = _freeze_records(runner)
    [item] = _evidence_items(freeze.metadata.prior_items)
    assert item.text == _EVIDENCE_TEXT and item.reviewer == "Codex" and item.resolution_owners == ("Codex",)
    assert item.source_round == 1


def test_converted_row_freeze_advisory_freeze_round_trip_keeps_provenance_and_clears(tmp_path, monkeypatch):
    """Item-5.s3 (converted row): restored once with its provenance, then cleared by its row clearance."""
    from test_pr_loop_evidence_conversion import (
        _evidence_items as row_items, _frozen_after_auto_conversion, _plan_context, _review)

    context = _plan_context("row-a", "row-b")
    runner, config = _frozen_after_auto_conversion(tmp_path, monkeypatch, context)
    head = "525753a4b04bd57d874211ba1fade250c3ff6d82"
    [original] = row_items(_freeze_records(runner)[0].metadata.prior_items, "row-a")
    advisory = dataclasses.replace(config, pr_human_evidence="advisory")
    assert run_pr_loop(runner, pr_number=77, config=advisory, approved_plan_context=context) == 0
    [release] = _release_records(runner)
    assert release.metadata.evidence_release.reason == "advisory-mode"
    [check] = [HumanCheck.from_mapping(e) for e in release.metadata.human_checks]
    assert check.kind == "converted" and "`row-a`" in _check_comments(runner)[-1]
    # Back to freeze: the row is restored from its origin item and freezes again.
    with pytest.raises(HumanDecisionRequiredError, match=f"frozen at head {head}"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    [restored] = row_items(_freeze_records(runner)[-1].metadata.prior_items, "row-a")
    assert (restored.item_id, restored.obligation_identity, restored.converted_matrix_identity,
            restored.resolution_owners, restored.reviewer) == (
        original.item_id, original.obligation_identity, original.converted_matrix_identity,
        original.resolution_owners, original.reviewer)
    # Its existing row clearance resolves it.
    runner.pr_payload["comments"].append(
        _signed(f"Ran the row-a scenario at {head} against the live stack: all checks passed.", 1))
    runner.codex_outputs.append(_review(resolved=[restored.item_id], hr=True))
    assert run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context) == 0
    cleared = [r for r in _release_records(runner) if r.metadata.evidence_release.reason == "evidence-cleared"]
    assert cleared and [ref.row_id for ref in cleared[-1].metadata.signed_evidence_references] == ["row-a"]


# --- review round 2 fixes ------------------------------------------------------------------


def test_multi_owner_converted_row_round_trip_keeps_owner_voting_and_clears_after_all_owners(
    tmp_path, monkeypatch
):
    """Item-5.s3: a two-owner converted row survives freeze -> advisory -> freeze and needs both owners."""
    from test_pr_loop_evidence_conversion import (
        _evidence_items as row_items, _plan_context, _review, _stalled_two_reviewer_runner)

    context = _plan_context("row-a", "row-b")
    runner, config = _stalled_two_reviewer_runner(tmp_path, monkeypatch, context)
    runner.codex_outputs.append(_review(kept=["item-2", "item-3"], state="blocking"))
    runner.gemini_outputs.append(_review(kept=["item-2", "item-3"], state="blocking", reviewer="Google Gemini"))
    head = "525753a4b04bd57d874211ba1fade250c3ff6d82"
    with pytest.raises(HumanDecisionRequiredError, match=f"frozen at head {head}"):
        run_pr_loop(runner, pr_number=77, approved_plan_context=context,
                    config=dataclasses.replace(config, pr_convert_evidence_rows=("row-a", "row-b")))
    [original] = row_items(_freeze_records(runner)[-1].metadata.prior_items, "row-b")
    assert original.resolution_owners == ("Codex", "Gemini")

    # advisory: the freeze is released and both rows move to the list.
    assert run_pr_loop(runner, pr_number=77, approved_plan_context=context,
                       config=dataclasses.replace(config, pr_human_evidence="advisory")) == 0
    release = [r for r in _release_records(runner) if r.metadata.evidence_release.reason == "advisory-mode"][-1]
    listed = {HumanCheck.from_mapping(e).origin_item.converted_row_id for e in release.metadata.human_checks}
    assert listed == {"row-a", "row-b"}

    # freeze again: the two-owner row is restored once with its provenance.
    with pytest.raises(HumanDecisionRequiredError, match=f"frozen at head {head}"):
        run_pr_loop(runner, pr_number=77, approved_plan_context=context, config=config)
    [restored] = row_items(_freeze_records(runner)[-1].metadata.prior_items, "row-b")
    assert len(row_items(_freeze_records(runner)[-1].metadata.prior_items, "row-a")) == 1
    assert (restored.item_id, restored.obligation_identity, restored.converted_matrix_identity,
            restored.resolution_owners, restored.owner_states) == (
        original.item_id, original.obligation_identity, original.converted_matrix_identity,
        original.resolution_owners, original.owner_states)
    row_a = row_items(_freeze_records(runner)[-1].metadata.prior_items, "row-a")[0]

    # Only Codex accepts: row-b stays frozen while Gemini has not accepted.
    runner.pr_payload["comments"].append(_signed(f"Ran rows a and b at {head}: all checks passed.", 5))
    both = [row_a.item_id, restored.item_id]
    runner.codex_outputs.append(_review(resolved=both, hr=True))
    runner.gemini_outputs.append(_review(resolved=[row_a.item_id], kept=[restored.item_id],
                                         state="blocking", hr=True, reviewer="Google Gemini"))
    with pytest.raises(HumanDecisionRequiredError, match=f"frozen at head {head}"):
        run_pr_loop(runner, pr_number=77, approved_plan_context=context, config=config)
    [pending] = [i for i in row_items(_freeze_records(runner)[-1].metadata.prior_items, "row-b")
                 if i.lifecycle == "evidence_frozen"]
    assert pending.item_id == restored.item_id
    assert dict(pending.owner_states).get("Gemini") != "resolved"

    # Gemini accepts too: the row clears through its real clearance and the freeze releases.
    runner.pr_payload["comments"].append(_signed(f"Gemini's row-b scenario at {head} also passed.", 6))
    runner.codex_outputs.append(_review(resolved=[restored.item_id], hr=True))
    runner.gemini_outputs.append(_review(resolved=[restored.item_id], hr=True, reviewer="Google Gemini"))
    assert run_pr_loop(runner, pr_number=77, approved_plan_context=context, config=config) == 0
    cleared = [r for r in _release_records(runner) if r.metadata.evidence_release.reason == "evidence-cleared"]
    assert cleared and "row-b" in {ref.row_id for ref in cleared[-1].metadata.signed_evidence_references}


def _varied_requests(round_number, count=40):
    import hashlib

    return [
        f"Round {round_number} live scenario {n}: "
        + "".join(hashlib.sha256(f"{round_number}-{n}-{k}".encode()).hexdigest() for k in range(6))
        for n in range(count)
    ]


def test_accumulated_advisory_list_overflows_into_lossless_sidecars(tmp_path, monkeypatch):
    """Item-6: a list too large for one comment spills, survives restart and restores in freeze."""
    from coding_review_agent_loop import round_transport

    requests = [_varied_requests(r) for r in (1, 2, 3)]
    runner = FakeRunner(
        codex_outputs=[
            _evidence_review(state="blocking", blocking=["Fix retry bound one."], evidence=requests[0]),
            _evidence_review(state="blocking", blocking=["Fix retry bound two."], evidence=requests[1],
                             dispositions=[{"item_id": "item-1", "disposition": "resolved", "note": "Fixed."}]),
            _evidence_review(evidence=requests[2],
                             dispositions=[{"item_id": "item-42", "disposition": "resolved", "note": "Fixed."}]),
        ],
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"]),
                        structured_coder_followup(addressed_items=["item-42"])],
    )
    assert run_pr_loop(runner, pr_number=77, config=_config(tmp_path, max_rounds=4)) == 0
    spilled = [key for c in runner.pr_payload["comments"] for key in round_transport.attachment_keys(c["body"])
               if key[1] == "human_checks"]
    assert spilled, "the accumulated list must use sidecar transport"
    expected = sorted(text for batch in requests for text in batch)
    [record] = _check_records(runner)
    assert sorted(HumanCheck.from_mapping(e).text for e in record.metadata.human_checks) == expected
    # Restart: rebuilt from the hydrated record, not reposted.
    runner.codex_outputs.append(_evidence_review())
    assert run_pr_loop(runner, pr_number=77, config=_config(tmp_path, max_rounds=6)) == 0
    assert len(_check_comments(runner)) == 1
    # Freeze: every check comes back as its obligation.  Today's freeze notice
    # lists every request in its visible body, a pre-existing size limit
    # outside this change, so the notice is shortened; the ledger is not.
    monkeypatch.setattr(pr_loop_support, "_render_evidence_freeze_notice",
                        lambda **kwargs: "## Exact-head evidence freeze")
    runner.codex_outputs.append(_evidence_review())
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path, max_rounds=8, pr_human_evidence="freeze"))
    [freeze] = _freeze_records(runner)
    assert sorted(i.text for i in _evidence_items(freeze.metadata.prior_items)) == expected


@pytest.mark.parametrize("sidecars", ["present", "missing", "corrupt"])
def test_spilled_human_checks_hydrate_or_fail_closed_when_a_sidecar_is_unavailable(sidecars):
    """Items 6 and 7: the spilled list hydrates losslessly; an unavailable sidecar fails the record."""
    from coding_review_agent_loop.round_metadata import PostedRoundMetadata, _attach_round_metadata
    from coding_review_agent_loop.round_state import _decode_round_metadata_records
    from coding_review_agent_loop.round_transport import attachment_keys, prepare_round_comment
    from coding_review_agent_loop.unresolved_items import _upsert_evidence_obligation

    checks = []
    for n, text in enumerate(_varied_requests(9, count=320), start=1):
        items, _ = _upsert_evidence_obligation([], item_number=n, reviewer="Codex", text=text,
                                               source_round=1, current_head_sha=HEAD)
        checks.append(HumanCheck("Codex", text, 1, HEAD, "native", items[0]).as_dict())
    body = _attach_round_metadata("Human checks.", PostedRoundMetadata(
        flow=HUMAN_CHECKS_FLOW, role="summary", agent="Orchestrator", round_number=1,
        subject=HEAD, human_checks=tuple(checks)))
    bodies = [str(part) for part in prepare_round_comment(body)]
    assert any(key[1] == "human_checks" for b in bodies for key in attachment_keys(b))
    anchor = next(b for b in bodies if "AGENT_LOOP_META" in b)
    kept = {"present": bodies, "missing": [anchor], "corrupt": [_corrupt_sidecar(b) for b in bodies]}[sidecars]
    if sidecars == "present":
        [record] = _decode_round_metadata_records([SimpleNamespace(body=b) for b in kept], flow=HUMAN_CHECKS_FLOW)
        assert record.metadata.human_checks == tuple(checks)
    else:
        with pytest.raises(AgentLoopError, match="Incomplete round metadata: human_checks"):
            _decode_round_metadata_records([SimpleNamespace(body=b) for b in kept], flow=HUMAN_CHECKS_FLOW)


def _corrupt_sidecar(body):
    """Flip the payload of every human_checks transport attachment in ``body``."""
    import base64
    import json as _json
    from coding_review_agent_loop.round_transport import ROUND_TRANSPORT_SIDECAR_RE

    def flip(match):
        raw = match.group("payload")
        pad = "=" * (-len(raw) % 4)
        item = _json.loads(base64.urlsafe_b64decode(raw + pad).decode())
        if item.get("field") != "human_checks":
            return match.group(0)
        item["data"] = item["data"][::-1]
        encoded = base64.urlsafe_b64encode(_json.dumps(item).encode()).decode()
        return match.group(0).replace(raw, encoded)

    return ROUND_TRANSPORT_SIDECAR_RE.sub(flip, body)


class _StopAfterRelease(Exception):
    pass


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_release_boundary_with_unavailable_list_sidecar_fails_closed_without_losing_checks(
    tmp_path, monkeypatch, damage
):
    """Item-7: a lost or corrupt list sidecar on the advisory-mode release never reads as empty."""
    import hashlib
    from coding_review_agent_loop import round_transport

    texts = [f"Check {n} " + hashlib.sha256(str(n).encode()).hexdigest() for n in range(40)]
    runner = FakeRunner(codex_outputs=[_evidence_review(evidence=texts)])
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path, pr_human_evidence="freeze"))

    # Advisory migration, interrupted right after the release record is posted.
    real_post = pr_loop_support.post_pr_comment

    def post_then_stop(*args, **kwargs):
        real_post(*args, **kwargs)
        if "freeze released (advisory mode)" in str(kwargs.get("body", "")):
            raise _StopAfterRelease()

    with monkeypatch.context() as patch:
        patch.setattr(pr_loop_support, "post_pr_comment", post_then_stop)
        # A smaller comment budget forces the release record's list into sidecars
        # without inflating the fixture (the technique test_round_transport uses).
        patch.setattr(round_transport, "MAX_GITHUB_BODY_CHARS", 4_000)
        patch.setattr(round_transport, "_PART_CHARS", 2_500)
        with pytest.raises(_StopAfterRelease):
            run_pr_loop(runner, pr_number=77, config=_config(tmp_path, auto_merge=True))
    [release] = _release_records(runner)
    assert release.metadata.evidence_release.reason == "advisory-mode"
    assert not _evidence_items(release.metadata.prior_items)
    comments = runner.pr_payload["comments"]
    carriers = [i for i, c in enumerate(comments)
                if any(k[1] == "human_checks" for k in round_transport.attachment_keys(c["body"]))]
    assert carriers, "the release record's list must use sidecar transport"
    saved = {i: comments[i]["body"] for i in carriers}
    if damage == "missing":
        runner.pr_payload["comments"] = [c for i, c in enumerate(comments) if i not in carriers]
    else:
        for i in carriers:
            comments[i]["body"] = _corrupt_sidecar(comments[i]["body"])

    with pytest.raises(AgentLoopError, match="human_checks"):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path, auto_merge=True))
    assert _merge_calls(runner) == [] and _check_comments(runner) == []

    # Restoring the transport recovers every check; still no merge.
    if damage == "missing":
        runner.pr_payload["comments"] = comments
    for i, body in saved.items():
        comments[i]["body"] = body
    assert run_pr_loop(runner, pr_number=77, config=_config(tmp_path, auto_merge=True)) == 0
    [record] = _check_records(runner)
    assert sorted(HumanCheck.from_mapping(e).text for e in record.metadata.human_checks) == sorted(texts)
    assert _merge_calls(runner) == []


# --- reviewer-visible list and requester withdrawal (#2105, #2106) --------------------------

_OTHER_TEXT = "Run the authenticated live-CLI smoke scenario at this exact head."
_LATER_HEAD = "0b3d8b29493059afd7f9912106279c4643ac4939"
_SECTION = "Advisory human checks (non-blocking, not in prior items)"


def _id_for(text, reviewer="Codex"):
    from coding_review_agent_loop.human_evidence_mode import display_id
    from coding_review_agent_loop.unresolved_items import evidence_obligation_identity

    return display_id(evidence_obligation_identity(text, reviewer))


def _withdrawing(review, *ids):
    import json

    raw, footer = review.split("\n<!--", 1)
    payload = json.loads(raw)
    payload["withdrawn_human_checks"] = list(ids)
    return json.dumps(payload) + "\n<!--" + footer


def _listed(record):
    return sorted(HumanCheck.from_mapping(e).text for e in record.metadata.human_checks)


def reviewer_head(runner):
    return [r for r in _pr_records(runner) if r.metadata.role == "reviewer"][-1].metadata.subject


def _orchestrator_records(runner):
    return [r for r in _pr_records(runner) if r.metadata.agent == "Orchestrator"]


class _Interrupted(Exception):
    pass


def _interrupt_withdrawal(monkeypatch):
    real = HumanEvidenceRun.withdraw

    def interrupted(self, *args, **kwargs):
        monkeypatch.setattr(HumanEvidenceRun, "withdraw", real)
        raise _Interrupted("interrupted after the reviewer record was published")

    monkeypatch.setattr(HumanEvidenceRun, "withdraw", interrupted)


def test_withdrawal_that_empties_the_list_lets_auto_merge_merge_the_later_head(tmp_path, capsys):
    """Row withdraw-empties-automerge."""
    code = "Guard the pre-spawn git status call."
    runner = FakeRunner(
        codex_outputs=[
            _evidence_review(state="blocking", blocking=[code], evidence=[_EVIDENCE_TEXT]),
            _withdrawing(_evidence_review(
                dispositions=[{"item_id": "item-1", "disposition": "resolved", "note": "Fixed."}]),
                _id_for(_EVIDENCE_TEXT)),
        ],
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
    )
    assert run_pr_loop(runner, pr_number=77, config=_config(tmp_path, auto_merge=True)) == 0
    first_prompt, second_prompt = (cmd[-1] for cmd in _reviewer_calls(runner))
    assert _SECTION not in first_prompt
    assert _id_for(_EVIDENCE_TEXT) in second_prompt and "requested by Codex" in second_prompt
    [merge] = _merge_calls(runner)
    # The normal exact-head merge path, at the later head the reviewers approved.
    assert merge[merge.index("--match-head-commit") + 1] == reviewer_head(runner)
    assert _check_comments(runner) == []
    reviewer = [r for r in _pr_records(runner) if r.metadata.role == "reviewer"][-1]
    assert reviewer.metadata.withdrawn_human_checks == (_id_for(_EVIDENCE_TEXT),)
    assert "approved but not merged" not in capsys.readouterr().out
    assert reviewer_head(runner) != HEAD


def _seeded_list_comment(runner, *checks):
    from test_pr_loop_evidence_conversion import _comment
    from coding_review_agent_loop.round_metadata import PostedRoundMetadata, _attach_round_metadata

    runner.pr_payload["comments"].append(_comment(_attach_round_metadata(
        "## Human checks requested\n\nSeeded.", PostedRoundMetadata(
            flow=HUMAN_CHECKS_FLOW, role="summary", agent="Orchestrator", round_number=1,
            subject="0" * 40, state="approved", human_checks=tuple(c.as_dict() for c in checks)))))


def _native_and_converted_checks():
    from coding_review_agent_loop.unresolved_items import (
        _upsert_converted_evidence_obligation, _upsert_evidence_obligation)

    [native], _ = _upsert_evidence_obligation([], item_number=91, reviewer="Codex", text=_EVIDENCE_TEXT,
                                             source_round=1, current_head_sha="0" * 40)
    _ledger, row, _ = _upsert_converted_evidence_obligation(
        [], item_number=92, row_id="row-a", matrix_identity="e" * 64, head_sha="0" * 40,
        owners=("Codex",), source_round=1, explanation="")
    return (HumanCheck("Codex", native.text, 1, "0" * 40, "native", native),
            HumanCheck(row.reviewer, row.text, 1, "0" * 40, "converted", row))


def test_advisory_list_is_rendered_to_reviewers_with_ids_and_guidance(tmp_path):
    """Row advisory-render."""
    from coding_review_agent_loop.human_evidence_mode import display_id

    native, row = _native_and_converted_checks()
    runner = FakeRunner(codex_outputs=[_evidence_review()])
    _seeded_list_comment(runner, native, row)
    assert run_pr_loop(runner, pr_number=77, config=_config(tmp_path)) == 0
    [prompt] = [cmd[-1] for cmd in _reviewer_calls(runner)]
    section = prompt[prompt.index(_SECTION):]
    assert f"`{display_id(native.identity)}` (requested by Codex, round 1, head `{'0' * 40}`): {_EVIDENCE_TEXT}" in section
    assert f"`{display_id(row.identity)}` (risk-matrix row `row-a`, round 1, head `{'0' * 40}`)" in section
    assert "Do not re-request a check that is already listed" in section
    assert "never put their IDs in `prior_item_dispositions`" in section
    # Not carried as prior items: neither origin item ID reaches the ledger block.
    assert "item-91" not in prompt and "item-92" not in prompt


def test_compact_reviewer_prompt_puts_the_list_in_the_volatile_tail(tmp_path):
    """Row advisory-render (compact builder)."""
    from coding_review_agent_loop import prompt_support, prompts
    from coding_review_agent_loop.human_evidence_mode import render_reviewer_context

    context = render_reviewer_context(_native_and_converted_checks())
    prompt = prompts.build_review_prompt(77, 2, _config(tmp_path), reviewer="codex", compact_context=True,
                                         human_checks_context=context)
    marker = prompt.index(prompt_support.COMPACT_PR_REVIEW_VOLATILE_TAIL_MARKER)
    assert prompt.index(_SECTION) > marker


@pytest.mark.parametrize("mode", ["freeze", "off"])
def test_freeze_and_off_render_nothing_and_ignore_withdrawals(tmp_path, monkeypatch, mode):
    """Row freeze-off-no-render."""
    lines = _captured_log(monkeypatch)
    evidence = [_EVIDENCE_TEXT] if mode == "freeze" else []
    runner = FakeRunner(codex_outputs=[_withdrawing(_evidence_review(evidence=evidence), _id_for(_EVIDENCE_TEXT))])
    config = _config(tmp_path, auto_merge=True, pr_human_evidence=mode)
    if mode == "freeze":
        with pytest.raises(HumanDecisionRequiredError, match=f"frozen at head {HEAD}"):
            run_pr_loop(runner, pr_number=77, config=config)
        [freeze] = _freeze_records(runner)
        # The field never clears a frozen exact-head obligation.
        assert [i.text for i in _evidence_items(freeze.metadata.prior_items)] == [_EVIDENCE_TEXT]
    else:
        assert run_pr_loop(runner, pr_number=77, config=config) == 0
        assert _merge_calls(runner)
    [prompt] = [cmd[-1] for cmd in _reviewer_calls(runner)]
    assert _SECTION not in prompt and "withdrawn_human_checks" not in prompt
    assert any(f"ignored 1 withdrawn human check ID(s) from Codex (--pr-human-evidence {mode})" in line
               for line in lines)
    assert runner.codex_outputs == []  # no re-ask
    reviewer = [r for r in _pr_records(runner) if r.metadata.role == "reviewer"][-1]
    assert reviewer.metadata.withdrawn_human_checks == ()


def test_partial_withdrawal_reposts_the_reduced_list_once_and_still_holds(tmp_path, capsys):
    """Row withdraw-partial."""
    code = "Guard the pre-spawn git status call."
    runner = FakeRunner(
        codex_outputs=[
            _evidence_review(state="blocking", blocking=[code], evidence=[_EVIDENCE_TEXT]),
            _withdrawing(_evidence_review(
                dispositions=[{"item_id": "item-1", "disposition": "resolved", "note": "Fixed."}]),
                _id_for(_EVIDENCE_TEXT)),
        ],
        gemini_outputs=[_evidence_review(evidence=[_OTHER_TEXT], reviewer="Google Gemini"),
                        _evidence_review(reviewer="Google Gemini",
                                         dispositions=[{"item_id": "item-1", "disposition": "resolved"}])],
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
    )
    config = _config(tmp_path, auto_merge=True, reviewer=("codex", "gemini"))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert runner.codex_outputs == [] and runner.gemini_outputs == []
    # The reduced list is posted once, at the holding head, and the hold still applies.
    assert [_listed(r) for r in _check_records(runner)] == [[_OTHER_TEXT]]
    assert [r.metadata.subject for r in _check_records(runner)] == [reviewer_head(runner)]
    assert _merge_calls(runner) == []
    assert "approved but not merged: 1 human check(s) requested" in capsys.readouterr().out
    # The other reviewer's entry stays.
    [remaining] = [HumanCheck.from_mapping(e) for e in _check_records(runner)[-1].metadata.human_checks]
    assert remaining.reviewer == "Gemini"
    # A rerun at the same head neither re-posts nor merges.
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert len(_check_records(runner)) == 1 and _merge_calls(runner) == []


def test_blocking_review_withdrawal_is_applied_before_the_reconciliation_record(tmp_path):
    """Row withdraw-before-reconciliation."""
    runner = FakeRunner(
        codex_outputs=[
            _evidence_review(evidence=[_EVIDENCE_TEXT]),
            _withdrawing(_evidence_review(state="blocking", dispositions=[{
                "item_id": "item-1", "disposition": "blocking",
                "note": "The status call is still unguarded at line 12; add the guard and a test."}]),
                _id_for(_EVIDENCE_TEXT)),
        ],
        gemini_outputs=[
            _evidence_review(state="blocking", blocking=["Guard the pre-spawn git status call."],
                             reviewer="Google Gemini"),
            _evidence_review(reviewer="Google Gemini",
                             dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
    )
    config = _config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    with pytest.raises(AgentLoopError, match="still reported blocking issues after round 2"):
        run_pr_loop(runner, pr_number=77, config=dataclasses.replace(config, max_rounds=2))
    assert runner.codex_outputs == [] and runner.gemini_outputs == []
    records = _orchestrator_records(runner)
    # The round-2 reviewers saw the listed request, so the withdrawal had something to remove.
    assert all(_id_for(_EVIDENCE_TEXT) in cmd[-1] for cmd in _reviewer_calls(runner)[-1:])
    [reconciliation] = [r for r in records if r.metadata.phase == "reconciliation" and r.metadata.round_number == 2]
    assert _listed(reconciliation) == []
    assert all(_listed(r) == [] for r in records if r.index >= reconciliation.index)
    # A resume from that record does not bring the entry back.
    runner.claude_outputs.append(structured_coder_followup(addressed_items=["item-1"]))
    runner.codex_outputs.append(_evidence_review(
        dispositions=[{"item_id": "item-1", "disposition": "resolved", "note": "Fixed."}]))
    runner.gemini_outputs.append(_evidence_review(
        reviewer="Google Gemini", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]))
    assert run_pr_loop(runner, pr_number=77, config=dataclasses.replace(config, max_rounds=5)) == 0
    assert _SECTION not in _reviewer_calls(runner)[-1][-1]
    assert _check_comments(runner) == []
    assert all(_listed(r) == [] for r in _orchestrator_records(runner) if r.index >= reconciliation.index)


@pytest.mark.parametrize("parallel", [True, False], ids=["parallel-publication", "sequential-authoritative"])
def test_interrupted_withdrawal_survives_resume_without_resurrecting_the_entry(tmp_path, monkeypatch, parallel):
    """Row resume-after-withdrawal."""
    runner = FakeRunner(codex_outputs=[_evidence_review(evidence=[_EVIDENCE_TEXT])])
    config = _config(tmp_path, auto_merge=True, review_parallel=parallel)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert len(_check_comments(runner)) == 1 and _merge_calls(runner) == []
    # A later head: the request made at the early head is now stale.
    runner._move_head(_LATER_HEAD, cwd=None)
    runner.pr_payload["headRefOid"] = _LATER_HEAD
    runner.codex_outputs.append(_withdrawing(_evidence_review(), _id_for(_EVIDENCE_TEXT)))
    _interrupt_withdrawal(monkeypatch)
    with pytest.raises(_Interrupted):
        run_pr_loop(runner, pr_number=77, config=config)
    [withdrawing] = [r for r in _pr_records(runner) if r.metadata.withdrawn_human_checks]
    assert withdrawing.metadata.phase == ("publication" if parallel else "authoritative")
    assert (withdrawing.metadata.canonical_reviewer_response is None) is (not parallel)
    # Every orchestrator-side record still lists the entry.
    assert _listed(_check_records(runner)[-1]) == [_EVIDENCE_TEXT]
    seen = []
    real_hold = HumanEvidenceRun.hold
    monkeypatch.setattr(HumanEvidenceRun, "hold", lambda self, *a, **k: seen.append(self.checks) or real_hold(self, *a, **k))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert runner.codex_outputs == [] and seen and all(checks == () for checks in seen)
    assert len(_check_comments(runner)) == 1, "no stale human-checks comment is reposted"
    assert _merge_calls(runner)


@pytest.mark.parametrize("variant", ["recorded", "interrupted"])
def test_freeze_resume_after_a_withdrawal_restores_only_the_remaining_check(tmp_path, monkeypatch, variant):
    """Row freeze-resume-after-withdrawal."""
    runner = FakeRunner(codex_outputs=[_evidence_review(evidence=[_EVIDENCE_TEXT, _OTHER_TEXT])])
    config = _config(tmp_path, auto_merge=True)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert _listed(_check_records(runner)[-1]) == sorted([_EVIDENCE_TEXT, _OTHER_TEXT])
    runner._move_head(_LATER_HEAD, cwd=None)
    runner.pr_payload["headRefOid"] = _LATER_HEAD
    runner.codex_outputs.append(_withdrawing(_evidence_review(), _id_for(_EVIDENCE_TEXT)))
    if variant == "recorded":
        assert run_pr_loop(runner, pr_number=77, config=config) == 0
        assert _listed(_check_records(runner)[-1]) == [_OTHER_TEXT]
    else:
        _interrupt_withdrawal(monkeypatch)
        with pytest.raises(_Interrupted):
            run_pr_loop(runner, pr_number=77, config=config)
        assert _listed(_check_records(runner)[-1]) == sorted([_EVIDENCE_TEXT, _OTHER_TEXT])
    restores = []
    real_normalize = HumanEvidenceRun.normalize

    def normalize(self, items, **kwargs):
        pending = self.restore_pending
        result = real_normalize(self, items, **kwargs)
        if pending:
            restores.append(sorted(i.text for i in _evidence_items(result)))
        return result

    monkeypatch.setattr(HumanEvidenceRun, "normalize", normalize)
    runner.codex_outputs.append(_evidence_review())
    with pytest.raises(HumanDecisionRequiredError, match=f"frozen at head {_LATER_HEAD}"):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path, auto_merge=True, pr_human_evidence="freeze"))
    assert restores == [[_OTHER_TEXT]]
    [freeze] = _freeze_records(runner)
    assert [i.text for i in _evidence_items(freeze.metadata.prior_items)] == [_OTHER_TEXT]
    assert _merge_calls(runner) == []


def test_ordinary_recovery_hold_says_the_pr_is_still_a_draft_and_leaves_it_draft(tmp_path, monkeypatch, capsys):
    """Row recovery-hold-draft (#2106)."""
    from coding_review_agent_loop.github import PullRequestMetadata
    from coding_review_agent_loop.managed_ci_types import ManagedCiOutcome
    from pr_loop_test_helpers import _watch_check_board

    capability = _recovery_capability()
    monkeypatch.setattr(pr_loop_support, "refresh_ordinary_recovery_capability", lambda *a, **k: capability)
    monkeypatch.setattr(pr_loop_support, "wait_for_ordinary_recovery", lambda *a, **k: ManagedCiOutcome(
        status="passed", checks=_watch_check_board("passing"), head_sha=HEAD))
    patch_attr_in(monkeypatch, _CONTEXT_MODULES, "get_observed_pr_review_context",
                  lambda *a, **k: SimpleNamespace(metadata=PullRequestMetadata(
                      number=77, repo="OWNER/REPO", title="draft", head_branch="feature",
                      base_branch="main", head_sha=HEAD, url=None)))
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *a, **k: pytest.fail("advisory hold must not merge"))
    config = _config(tmp_path, auto_merge=True)
    run = HumanEvidenceRun(config, 77)
    run.checks = _native_and_converted_checks()[:1]
    runner = FakeRunner()
    assert pr_loop_support._finalize_ordinary_recovery_checked(
        runner, config=config, pr_number=77, round_number=1, items=[], current_head_sha=HEAD,
        capability=capability,
        hold=lambda: run.hold(runner, head_sha=HEAD, round_number=1, merging=True, draft=True)) is False
    assert "PR #77 approved but not merged and still a draft: 1 human check(s)" in capsys.readouterr().out
    [body] = _check_comments(runner)
    assert "PR #77 is still a draft" in body
    assert not any(cmd[:3] in (["gh", "pr", "ready"], ["gh", "pr", "merge"]) for cmd, _cwd in runner.commands)


def test_only_the_ordinary_recovery_hold_sites_pass_draft():
    """Rows recovery-hold-draft and nonrecovery-hold-wording: call-site wiring in pr_loop."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(pr_loop))
    holds = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "hold"
             and isinstance(node.func.value, ast.Name) and node.func.value.id == "human_evidence"]
    # The ordinary-recovery sites pass the hold as `hold=lambda: human_evidence.hold(...)`.
    recovery_holds = {id(keyword.value.body) for keyword in ast.walk(tree) if isinstance(keyword, ast.keyword)
                      and keyword.arg == "hold" and isinstance(keyword.value, ast.Lambda)}
    drafted = [any(k.arg == "draft" and k.value.value is True for k in call.keywords) for call in holds]
    assert len(holds) == 6 and drafted.count(True) == 3
    assert all(any(k.arg == "draft" for k in call.keywords) == (id(call) in recovery_holds) for call in holds)


@pytest.mark.parametrize("mode", ["advisory", "freeze"])
def test_returning_to_a_previously_reviewed_head_does_not_resurrect_a_withdrawn_check(
    tmp_path, monkeypatch, capsys, mode
):
    """Items 1 and 2: an older review replayed after its check was withdrawn stays withdrawn."""
    runner = FakeRunner(codex_outputs=[_evidence_review(evidence=[_EVIDENCE_TEXT])])
    config = _config(tmp_path, auto_merge=True)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert _listed(_check_records(runner)[-1]) == [_EVIDENCE_TEXT]
    # Head B: the requester withdraws the sole check (no merge yet).
    runner._move_head(_LATER_HEAD, cwd=None)
    runner.pr_payload["headRefOid"] = _LATER_HEAD
    runner.codex_outputs.append(_withdrawing(_evidence_review(), _id_for(_EVIDENCE_TEXT)))
    assert run_pr_loop(runner, pr_number=77, config=dataclasses.replace(config, auto_merge=False)) == 0
    assert runner.codex_outputs == [] and _merge_calls(runner) == []
    # The branch returns to head A, whose original review requested the check.
    runner._move_head(HEAD, cwd=None)
    runner.pr_payload["headRefOid"] = HEAD
    runner.codex_outputs.append(_evidence_review())  # only used if the head is reviewed again
    capsys.readouterr()
    seen = []
    real_hold = HumanEvidenceRun.hold
    monkeypatch.setattr(HumanEvidenceRun, "hold",
                        lambda self, *a, **k: seen.append(self.checks) or real_hold(self, *a, **k))
    assert run_pr_loop(runner, pr_number=77, config=dataclasses.replace(config, pr_human_evidence=mode)) == 0
    assert all(checks == () for checks in seen)
    assert _freeze_records(runner) == []
    assert len(_check_comments(runner)) == 1
    [merge] = _merge_calls(runner)
    assert merge[merge.index("--match-head-commit") + 1] == HEAD
    assert "approved but not merged" not in capsys.readouterr().out


_THIRD_HEAD = "9d2a7c1f0e4b5a6c7d8e9f0a1b2c3d4e5f6a7b8c"


def _move_to(runner, head):
    runner._move_head(head, cwd=None)
    runner.pr_payload["headRefOid"] = head


def test_returning_to_the_withdrawing_head_does_not_cancel_a_later_re_request(tmp_path, capsys):
    """Item-3: request at A, withdraw at B, re-request at C, then return to B: the check still holds."""
    runner = FakeRunner(codex_outputs=[_evidence_review(evidence=[_EVIDENCE_TEXT])])
    config = _config(tmp_path, auto_merge=False)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    _move_to(runner, _LATER_HEAD)
    runner.codex_outputs.append(_withdrawing(_evidence_review(), _id_for(_EVIDENCE_TEXT)))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    _move_to(runner, _THIRD_HEAD)
    runner.codex_outputs.append(_evidence_review(evidence=[_EVIDENCE_TEXT]))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert runner.codex_outputs == [] and _listed(_check_records(runner)[-1]) == [_EVIDENCE_TEXT]
    # Back to head B, whose recorded review withdrew the check before C re-requested it.
    _move_to(runner, _LATER_HEAD)
    capsys.readouterr()
    calls = len(_reviewer_calls(runner))
    assert run_pr_loop(runner, pr_number=77, config=dataclasses.replace(config, auto_merge=True)) == 0
    assert len(_reviewer_calls(runner)) == calls, "head B's recorded review is reused"
    assert _merge_calls(runner) == []
    assert "approved but not merged: 1 human check(s) requested" in capsys.readouterr().out
    assert _listed(_check_records(runner)[-1]) == [_EVIDENCE_TEXT]


@pytest.mark.parametrize("rerun_mode", ["advisory", "freeze"])
def test_an_ignored_withdrawal_stays_ignored_after_resume(tmp_path, capsys, rerun_mode):
    """Item-4: a stale ID sent beside a new request is ignored live and on every resume."""
    runner = FakeRunner(codex_outputs=[_withdrawing(_evidence_review(evidence=[_EVIDENCE_TEXT]),
                                                    _id_for(_EVIDENCE_TEXT))])
    config = _config(tmp_path, auto_merge=True)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    [reviewer] = [r for r in _pr_records(runner) if r.metadata.role == "reviewer"]
    assert reviewer.metadata.withdrawn_human_checks == (_id_for(_EVIDENCE_TEXT),)
    assert _listed(_check_records(runner)[-1]) == [_EVIDENCE_TEXT] and _merge_calls(runner) == []
    capsys.readouterr()
    runner.codex_outputs.append(_evidence_review())  # only used if the head is reviewed again
    rerun = dataclasses.replace(config, pr_human_evidence=rerun_mode)
    if rerun_mode == "freeze":
        with pytest.raises(HumanDecisionRequiredError, match=f"frozen at head {HEAD}"):
            run_pr_loop(runner, pr_number=77, config=rerun)
        [freeze] = _freeze_records(runner)
        assert [i.text for i in _evidence_items(freeze.metadata.prior_items)] == [_EVIDENCE_TEXT]
    else:
        assert run_pr_loop(runner, pr_number=77, config=rerun) == 0
        assert "approved but not merged: 1 human check(s) requested" in capsys.readouterr().out
        assert len(_check_comments(runner)) == 1
    assert _merge_calls(runner) == []
