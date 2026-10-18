"""PR-loop tests for panel_evidence: the human-only exact-head evidence freeze (#1542)."""

import dataclasses
import re

import pytest
from coding_review_agent_loop.cli import AgentLoopError, run_pr_loop
from coding_review_agent_loop.errors import HumanDecisionRequiredError
from coding_review_agent_loop.managed_ci import (
    ManagedCiContract,
    ManagedCiOutcome,
)
from coding_review_agent_loop.round_state import (
    _attach_round_metadata,
    _decode_round_metadata,
    _strip_round_metadata,
)
from agent_loop_helpers import (
    FakeRunner,
    make_config,
    structured_coder_followup,
)
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.plan_first_loop as plan_first_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
from agent_loop_helpers import patch_attr_in
from pr_loop_test_helpers import (
    _EVIDENCE_TEXT,
    _evidence_review,
    _freeze_records,
    _pr_records,
    _release_records,
    _resolve,
    _signed,
)


# --- Human-only exact-head evidence freeze (#1068) ---------------------------


def _keep(item_id="item-1", note="The signed output omits the sandbox escape case; rerun it."):
    return {"item_id": item_id, "disposition": "blocking", "note": note}


def _phases(runner):
    return [record.metadata.phase for record in _pr_records(runner)]


def _frozen_runner(tmp_path, *, codex_outputs=(), **runner_kwargs):
    """Run one clean round that ends in a freeze at abc123."""
    runner = FakeRunner(
        codex_outputs=[_evidence_review(evidence=[_EVIDENCE_TEXT]), *codex_outputs],
        **runner_kwargs,
    )
    return runner


def test_clean_board_with_evidence_publishes_one_freeze_and_exits_at_decision_boundary(tmp_path):
    """Row freeze-on-clean-board (main)."""
    runner = _frozen_runner(tmp_path)
    config = make_config(tmp_path)

    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123") as excinfo:
        run_pr_loop(runner, pr_number=77, config=config)
    assert excinfo.value.EXIT_CODE == 4

    freezes = _freeze_records(runner)
    assert len(freezes) == 1
    freeze = freezes[0].metadata
    assert freeze.subject == "abc123"
    assert freeze.evidence_freeze.frozen_head == "abc123"
    assert freeze.evidence_freeze.allowed_rounds == config.max_rounds
    evidence = [item for item in freeze.prior_items if item.obligation_kind == "human-exact-head-evidence"]
    assert [(item.lifecycle, item.candidate_head_sha) for item in evidence] == [
        ("evidence_frozen", "abc123")
    ]
    assert freeze.evidence_freeze.evidence_identities == (evidence[0].obligation_identity,)
    body = freezes[0].body
    assert "Evidence requested at `abc123`. No further code changes will be accepted" in body
    assert _EVIDENCE_TEXT in body and "-- Human Reviewer" in body
    # The freeze notice is the first public comment that shows the request.
    first_public = next(
        index for index, comment in enumerate(runner.comments) if _EVIDENCE_TEXT in comment
    )
    assert "## Exact-head evidence freeze" in runner.comments[first_public]
    commands = [cmd for cmd, _cwd in runner.commands]
    assert not any(cmd[:1] == ["claude"] for cmd in commands)
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd in commands)
    # The reviewer's own record persisted the typed request.
    reviewer_records = [record for record in _pr_records(runner) if record.metadata.role == "reviewer"]
    assert reviewer_records[0].metadata.evidence_requests == (_EVIDENCE_TEXT,)


def test_frozen_rerun_without_new_signed_input_invokes_no_agent_and_posts_nothing(tmp_path):
    """Rows frozen-rerun-signed-input and new-obligation-while-frozen (pending variant)."""
    runner = _frozen_runner(
        tmp_path,
        pr_payload={"comments": [_signed("Keep backwards compatibility.", 1)]},
    )
    runner.codex_outputs[0] = _evidence_review(evidence=[_EVIDENCE_TEXT], hr=True)
    config = make_config(tmp_path)
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    posted = len(runner.pr_payload["comments"])

    # Checks are only pending now: the freeze stays and nothing runs.
    runner.pr_check_runs_payload = {
        "check_runs": [{"name": "test", "status": "in_progress", "conclusion": None}]
    }
    # A deleted signed comment is not new input either.
    runner.pr_payload["comments"].pop(0)
    posted -= 1
    commands_before = len(runner.commands)
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert len(runner.pr_payload["comments"]) == posted
    assert not any(
        cmd[:1] in (["codex"], ["claude"]) for cmd, _cwd in runner.commands[commands_before:]
    )
    assert len(_freeze_records(runner)) == 1


def test_frozen_rerun_detects_an_edited_signed_comment_by_identity(tmp_path):
    """Row frozen-rerun-signed-input (edited variant, created_at unchanged)."""
    signed = _signed("Keep backwards compatibility.", 1)
    runner = _frozen_runner(tmp_path, pr_payload={"comments": [signed]})
    runner.codex_outputs[0] = _evidence_review(evidence=[_EVIDENCE_TEXT], hr=True)
    config = make_config(tmp_path)
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)

    signed["body"] = "Keep backwards compatibility. Live suite at abc123: 11 passed.\n\n-- Human Reviewer"
    runner.codex_outputs.append(_evidence_review(dispositions=[_resolve()], hr=True))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert runner.codex_outputs == []
    releases = _release_records(runner)
    assert [record.metadata.evidence_release.reason for record in releases] == ["evidence-cleared"]


def test_failed_freeze_write_leaves_no_freeze_and_rerun_posts_exactly_one(tmp_path, monkeypatch):
    """Row freeze-write-interrupted."""
    runner = _frozen_runner(tmp_path)
    config = make_config(tmp_path)
    real_post = pr_loop_support.post_pr_comment
    failures = []

    def flaky_post(runner_, *, config, pr_number, body):
        if "## Exact-head evidence freeze" in str(body) and not failures:
            failures.append(body)
            raise AgentLoopError("simulated GitHub outage while posting the freeze")
        return real_post(runner_, config=config, pr_number=pr_number, body=body)

    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_loop,
    ), "post_pr_comment", flaky_post)
    with pytest.raises(AgentLoopError, match="simulated GitHub outage"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert failures
    assert "evidence-freeze" not in _phases(runner)
    assert not any(record.metadata.evidence_freeze for record in _pr_records(runner))

    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    assert len(_freeze_records(runner)) == 1
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    assert len(_freeze_records(runner)) == 1
    assert runner.codex_outputs == []


def _claude_prompts(runner):
    return ["\n".join(cmd) for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]


def test_evidence_stays_deferred_across_heads_while_code_findings_remain(tmp_path):
    """Row evidence-deferred-across-heads (integration)."""
    code_item = {"text": "Guard the pre-spawn git status call.", "fix_scope": ["src/coding_review_agent_loop/cli.py"]}
    runner = FakeRunner(
        codex_outputs=[
            _evidence_review(state="blocking", blocking=[code_item["text"]]),
            _evidence_review(
                state="blocking",
                dispositions=[
                    _keep("item-1", "The guard still misses the submodule branch."),
                    {"item_id": "item-2", "disposition": "resolved", "note": "Not my request."},
                ],
            ),
        ],
        gemini_outputs=[
            _evidence_review(evidence=[_EVIDENCE_TEXT], reviewer="Google Gemini"),
            _evidence_review(
                state="blocking",
                evidence=[_EVIDENCE_TEXT],
                reviewer="Google Gemini",
                dispositions=[
                    {"item_id": "item-1", "disposition": "resolved", "note": "Deferring to Codex."},
                    _keep("item-2", "Still required at the next clean head."),
                ],
            ),
        ],
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=2)

    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=config)
    message = str(excinfo.value)
    assert "reviewer-owned findings: Codex (item-1)" in message
    assert "final barrier" in message

    assert "evidence-freeze" not in _phases(runner)
    assert not any("Exact-head evidence freeze" in comment for comment in runner.comments)
    prompts = _claude_prompts(runner)
    assert len(prompts) == 1
    assert _EVIDENCE_TEXT not in prompts[0]
    # The deferred request stays in round metadata only: no public comment
    # (reviewer, coder, or stop message) shows its text before a freeze.
    assert not any(_EVIDENCE_TEXT in comment for comment in runner.comments)
    # One deduplicated obligation, still deferred, carried to the new head.
    records = _pr_records(runner)
    latest = [record for record in records if record.metadata.subject == "abc123-coder-1"][-1]
    evidence = [
        item for item in latest.metadata.prior_items
        if item.obligation_kind == "human-exact-head-evidence"
    ]
    assert len(evidence) == 1 and evidence[0].lifecycle == "evidence_deferred"
    assert evidence[0].reviewer == "Gemini"


def test_failing_checks_after_clean_reviews_go_to_the_coder_without_a_freeze(tmp_path):
    """Row clean-reviews-check-not-green (failing variant)."""
    runner = FakeRunner(
        codex_outputs=[
            _evidence_review(evidence=[_EVIDENCE_TEXT]),
            _evidence_review(
                evidence=[_EVIDENCE_TEXT],
                dispositions=[_resolve("item-1", "Deferred evidence; not a code issue."), _resolve("item-2", "Fixed.")],
            ),
        ],
        claude_outputs=[structured_coder_followup(addressed_items=["item-2"])],
        pr_check_runs_payload={
            "check_runs": [{"name": "test", "status": "completed", "conclusion": "failure"}]
        },
    )
    config = make_config(tmp_path, max_rounds=2)
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=config)
    assert len(_claude_prompts(runner)) == 1
    assert "evidence-freeze" not in _phases(runner)


def test_pending_checks_stop_names_deferred_evidence_and_restart_rederives_it(tmp_path, capsys):
    """Row clean-reviews-check-not-green (pending variant, then restart)."""
    runner = _frozen_runner(
        tmp_path,
        pr_check_runs_payload={
            "check_runs": [{"name": "test", "status": "in_progress", "conclusion": None}]
        },
    )
    config = make_config(tmp_path)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    out = capsys.readouterr().out
    assert "Human-only exact-head evidence remains the final barrier" in out
    assert any("final barrier" in comment for comment in runner.comments)
    assert "evidence-freeze" not in _phases(runner)
    assert not any(_EVIDENCE_TEXT in comment for comment in runner.comments)

    # Restart once checks pass: the deferred obligation is re-derived from the
    # reviewer record alone, so the freeze is published without a new review.
    runner.pr_check_runs_payload = {
        "check_runs": [{"name": "test", "status": "completed", "conclusion": "success"}]
    }
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    assert len(_freeze_records(runner)) == 1
    assert runner.codex_outputs == []


def _managed_setup(monkeypatch, *, outcome):
    contract = ManagedCiContract(protocol_version=2, protection_mode="strict")
    events = []
    monkeypatch.setattr(pr_loop, "activate_managed_ci", lambda *args, **kwargs: contract)
    monkeypatch.setattr(
        pr_loop, "dispatch_final_qualification",
        lambda *args, **kwargs: events.append("dispatch"),
    )
    monkeypatch.setattr(
        pr_loop, "wait_for_final_qualification",
        lambda *args, **kwargs: events.append("wait") or outcome,
    )
    monkeypatch.setattr(
        pr_loop, "publish_manual_v2_qualification",
        lambda *args, **kwargs: pytest.fail("a pending evidence obligation must not qualify"),
    )
    monkeypatch.setattr(
        pr_loop_support, "merge_pr",
        lambda *args, **kwargs: pytest.fail("a pending evidence obligation must not merge"),
    )
    return events


def test_managed_qualification_completes_before_the_freeze(tmp_path, monkeypatch):
    """Row managed-qualification-before-freeze (success)."""
    runner = _frozen_runner(tmp_path)
    config = make_config(tmp_path, managed_ci=True, max_rounds=1)
    events = _managed_setup(monkeypatch, outcome=ManagedCiOutcome(status="passed", head_sha="abc123"))
    real_post = pr_loop_support.post_pr_comment

    def recording_post(runner_, *, config, pr_number, body):
        if "## Exact-head evidence freeze" in str(body):
            events.append("freeze")
        return real_post(runner_, config=config, pr_number=pr_number, body=body)

    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_loop,
    ), "post_pr_comment", recording_post)
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    assert events == ["dispatch", "wait", "freeze"]


def test_managed_qualification_failure_is_repaired_without_a_freeze(tmp_path, monkeypatch):
    """Row managed-qualification-before-freeze (failure variant)."""
    runner = _frozen_runner(tmp_path)
    config = make_config(tmp_path, managed_ci=True, max_rounds=1)
    _managed_setup(
        monkeypatch,
        outcome=ManagedCiOutcome(status="failed", head_sha="abc123", failure_details=("unit tests failed",)),
    )
    runner.claude_outputs.append(structured_coder_followup(addressed_items=["item-2"]))
    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=config)
    assert not isinstance(excinfo.value, HumanDecisionRequiredError)
    # The failure became the managed obligation and went to the coder.
    prompts = _claude_prompts(runner)
    assert len(prompts) == 1 and "unit tests failed" in prompts[0]
    assert _EVIDENCE_TEXT not in prompts[0]
    assert "evidence-freeze" not in _phases(runner)


def _inject_signed_input_before(monkeypatch, runner, *, body, when):
    """Post a signed comment the first time ``when()`` holds at a context read."""
    real_context = pr_loop_support.get_pr_review_context
    injected = []

    def context_with_injection(runner_, **kwargs):
        if not injected and when():
            injected.append(True)
            runner.pr_payload["comments"].append(_signed(body, 40 + len(runner.pr_payload["comments"])))
        return real_context(runner_, **kwargs)

    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_implementation, plan_first_loop, issue_loop,
    ), "get_pr_review_context", context_with_injection)
    return injected


@pytest.mark.parametrize("max_rounds", [5, 1])
def test_signed_input_arriving_before_the_freeze_runs_a_same_round_refresh(
    tmp_path, monkeypatch, max_rounds
):
    """Row signed-input-before-freeze (including R == effective allowed_rounds)."""
    runner = _frozen_runner(
        tmp_path,
        codex_outputs=[
            _evidence_review(
                evidence=[_EVIDENCE_TEXT], hr=True,
                dispositions=[_resolve("item-1", "Still needed; not a code issue.")],
            )
        ],
    )
    config = make_config(tmp_path, max_rounds=max_rounds)
    injected = _inject_signed_input_before(
        monkeypatch, runner, body="Also keep the legacy flag.",
        when=lambda: len(runner.codex_outputs) == 1,
    )
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    assert injected and runner.codex_outputs == []
    records = _pr_records(runner)
    pass_reviews = [
        record for record in records
        if record.metadata.role == "reviewer" and record.metadata.phase == "evidence-response"
    ]
    assert [record.metadata.round_number for record in pass_reviews] == [1]
    freezes = _freeze_records(runner)
    assert len(freezes) == 1 and freezes[0].metadata.round_number == 1
    new_ids = set(pass_reviews[0].metadata.surfaced_reviewer_requirement_ids)
    assert new_ids and new_ids <= set(freezes[0].metadata.evidence_freeze.signed_requirement_ids_at_freeze)


def test_external_push_breaks_the_freeze_and_the_new_head_is_reviewed_in_full(tmp_path):
    """Row external-push-breaks-freeze (integration)."""
    runner = _frozen_runner(tmp_path)
    config = make_config(tmp_path)
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)

    runner.pr_payload["headRefOid"] = "def456"
    runner.git_head = "def456"
    runner.simulate_new_process()  # the rerun is a fresh agent-loop process
    runner.codex_outputs.append(
        _evidence_review(
            evidence=[_EVIDENCE_TEXT],
            dispositions=[_resolve("item-1", "Evidence for abc123 does not cover def456.")],
        )
    )
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head def456"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert runner.codex_outputs == []
    assert any("was broken by an external push" in comment for comment in runner.comments)
    freezes = _freeze_records(runner)
    assert [record.metadata.subject for record in freezes] == ["abc123", "def456"]
    assert not _claude_prompts(runner)


def test_kept_evidence_reposts_a_still_frozen_record_and_reexits(tmp_path):
    """Row evidence-response-outcome (kept variant)."""
    runner = _frozen_runner(tmp_path)
    config = make_config(tmp_path)
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    runner.pr_payload["comments"].append(_signed("Live suite at abc123: 10 passed, 1 failed.", 1))
    runner.codex_outputs.append(_evidence_review(state="blocking", dispositions=[_keep()], hr=True))
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123"):
        run_pr_loop(runner, pr_number=77, config=config)
    freezes = _freeze_records(runner)
    assert len(freezes) == 2
    assert "still in effect" in freezes[-1].body
    assert freezes[-1].metadata.evidence_freeze.signed_requirement_ids_at_freeze
    assert not _claude_prompts(runner)
    codex_prompts = ["\n".join(cmd) for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]
    response_prompt = " ".join(codex_prompts[-1].split())
    assert "evidence-response re-review at frozen head `abc123`" in response_prompt
    assert _EVIDENCE_TEXT in response_prompt
    # The kept obligation stays frozen at the same head and round.
    assert freezes[-1].metadata.round_number == freezes[0].metadata.round_number


def test_cleared_evidence_with_signed_input_during_the_pass_refreshes_at_the_ceiling(
    tmp_path, monkeypatch
):
    """Row evidence-response-outcome (injected-comment variant at the ceiling)."""
    runner = _frozen_runner(tmp_path)
    config = make_config(tmp_path, max_rounds=1)
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    runner.pr_payload["comments"].append(_signed("Live suite at abc123: 11 passed.", 1))
    runner.codex_outputs.extend(
        [
            _evidence_review(dispositions=[_resolve()], hr=True),
            _evidence_review(hr=True, summary="Refreshed; nothing further."),
        ]
    )
    injected = _inject_signed_input_before(
        monkeypatch, runner, body="Please also note the run in the changelog.",
        when=lambda: len(runner.codex_outputs) == 1,
    )
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert injected and runner.codex_outputs == []
    reasons = [record.metadata.evidence_release.reason for record in _release_records(runner)]
    assert reasons == ["evidence-cleared", "refresh-clean"]
    assert {record.metadata.round_number for record in _release_records(runner)} == {1}


def test_restart_after_cleared_release_refreshes_for_newer_signed_input(tmp_path, monkeypatch):
    """Row evidence-response-outcome (restart variant)."""
    runner = _frozen_runner(tmp_path)
    config = make_config(tmp_path)
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    runner.pr_payload["comments"].append(_signed("Live suite at abc123: 11 passed.", 1))
    runner.codex_outputs.append(_evidence_review(dispositions=[_resolve()], hr=True))
    real_checks = pr_loop.get_pr_checks
    stops = []

    def stop_after_release(*args, **kwargs):
        if _release_records(runner) and not stops:
            stops.append(True)
            raise AgentLoopError("simulated process stop before finalization")
        return real_checks(*args, **kwargs)

    monkeypatch.setattr(pr_loop, "get_pr_checks", stop_after_release)
    with pytest.raises(AgentLoopError, match="simulated process stop"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert [record.metadata.evidence_release.reason for record in _release_records(runner)] == [
        "evidence-cleared"
    ]

    runner.pr_payload["comments"].append(_signed("One more requirement: keep the flag.", 2))
    runner.codex_outputs.append(_evidence_review(hr=True, summary="Checked the new requirement."))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert runner.codex_outputs == []
    reasons = [record.metadata.evidence_release.reason for record in _release_records(runner)]
    assert reasons == ["evidence-cleared", "refresh-clean"]


def _retag_freeze_budget(runner, **budget):
    """Rewrite the latest freeze record's budget fields in place."""
    comments = runner.pr_payload["comments"]
    for comment in reversed(comments):
        if "AGENT_LOOP_META" not in comment["body"]:
            continue
        metadata = _decode_round_metadata(
            re.search(r"<!-- AGENT_LOOP_META: (\S+) -->", comment["body"]).group(1)
        )
        if metadata.phase != "evidence-freeze":
            continue
        payload = dataclasses.replace(metadata.evidence_freeze, **budget)
        comment["body"] = _attach_round_metadata(
            _strip_round_metadata(comment["body"]),
            dataclasses.replace(metadata, evidence_freeze=payload),
        )
        return
    raise AssertionError("no freeze record")


def test_freeze_at_the_ceiling_is_answered_twice_across_a_restart(tmp_path):
    """Row freeze-at-ceiling-repeated-response."""
    runner = _frozen_runner(tmp_path)
    config = make_config(tmp_path, max_rounds=1)
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    # A watcher extension had raised the effective ceiling before the freeze.
    _retag_freeze_budget(runner, allowed_rounds=2, watch_failure_extension_used=True)

    runner.pr_payload["comments"].append(_signed("Live suite at abc123: 10 passed, 1 failed.", 1))
    runner.codex_outputs.append(_evidence_review(state="blocking", dispositions=[_keep()], hr=True))
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    still_frozen = _freeze_records(runner)[-1].metadata
    assert still_frozen.round_number == 1
    assert still_frozen.evidence_freeze.allowed_rounds == 2
    assert still_frozen.evidence_freeze.watch_failure_extension_used is True

    runner.pr_payload["comments"].append(_signed("Rerun at abc123: 11 passed, 0 failed.", 2))
    runner.codex_outputs.append(_evidence_review(dispositions=[_resolve()], hr=True))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert runner.codex_outputs == []
    assert all(record.metadata.round_number == 1 for record in _release_records(runner))


def test_out_of_bound_freeze_budget_keeps_the_freeze_and_posts_one_correction(tmp_path):
    """Row freeze-at-ceiling-repeated-response (out-of-bound budget variant)."""
    runner = _frozen_runner(tmp_path)
    config = make_config(tmp_path, max_rounds=1)
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    _retag_freeze_budget(runner, allowed_rounds=99)

    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123"):
        run_pr_loop(runner, pr_number=77, config=config)
    freezes = _freeze_records(runner)
    assert len(freezes) == 2
    corrected = freezes[-1].metadata
    assert corrected.evidence_freeze.allowed_rounds == 1
    assert corrected.evidence_freeze.watch_failure_extension_used is False
    assert corrected.evidence_freeze.evidence_identities == freezes[0].metadata.evidence_freeze.evidence_identities
    # A corrected record is not re-posted on the next rerun.
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    assert len(_freeze_records(runner)) == 2

    runner.pr_payload["comments"].append(_signed("Live suite at abc123: 11 passed.", 1))
    runner.codex_outputs.append(_evidence_review(dispositions=[_resolve()], hr=True))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0


def test_interrupted_evidence_response_pass_is_rerun_in_full(tmp_path, monkeypatch):
    """Row evidence-response-interrupted."""
    runner = FakeRunner(
        codex_outputs=[_evidence_review(evidence=[_EVIDENCE_TEXT])],
        gemini_outputs=[_evidence_review(reviewer="Google Gemini")],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    freeze = _freeze_records(runner)[0].metadata

    runner.pr_payload["comments"].append(_signed("Live suite at abc123: 11 passed.", 1))
    runner.codex_outputs.append(_evidence_review(dispositions=[_resolve()], hr=True))
    real_post = pr_loop_support.post_pr_comment
    stopped = []

    def stop_after_first_response_record(runner_, *, config, pr_number, body):
        result = real_post(runner_, config=config, pr_number=pr_number, body=body)
        if not stopped and "evidence-response" in _phases(runner):
            stopped.append(True)
            raise AgentLoopError("simulated process stop mid-pass")
        return result

    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_loop,
    ), "post_pr_comment", stop_after_first_response_record)
    with pytest.raises(AgentLoopError, match="simulated process stop mid-pass"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert "evidence-release" not in _phases(runner)
    assert runner.gemini_outputs == []

    # The partial transcript cannot clear anything: the rerun restores the
    # freeze (ledger, budget, signed baseline) and reruns the whole board.
    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop, issue_loop), "post_pr_comment", real_post)
    runner.codex_outputs.append(_evidence_review(dispositions=[_resolve()], hr=True))
    runner.gemini_outputs.append(
        _evidence_review(
            reviewer="Google Gemini", hr=True,
            dispositions=[_resolve("item-1", "Codex owns this request.")],
        )
    )
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert runner.codex_outputs == [] and runner.gemini_outputs == []
    releases = _release_records(runner)
    assert len(releases) == 1 and releases[0].metadata.evidence_release.reason == "evidence-cleared"
    assert releases[0].metadata.evidence_release.allowed_rounds == freeze.evidence_freeze.allowed_rounds


def test_evidence_revealing_a_defect_releases_the_freeze_and_dispatches_the_coder(tmp_path):
    """Row evidence-reveals-defect."""
    runner = _frozen_runner(
        tmp_path,
        claude_outputs=[
            structured_coder_followup(
                addressed_items=["item-2"], human_requirement_ids=["Requirement 1"]
            )
        ],
    )
    config = make_config(tmp_path, max_rounds=2)
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    runner.pr_payload["comments"].append(
        _signed("Live suite at abc123: 10 passed, 1 failed (sandbox escape via submodule).", 1)
    )
    runner.codex_outputs.extend(
        [
            _evidence_review(
                state="blocking",
                blocking=["The submodule config gate is bypassable, as the signed live run shows."],
                dispositions=[_keep("item-1", "Evidence must be rerun at the fix head.")],
                hr=True,
            ),
            _evidence_review(
                state="blocking",
                dispositions=[
                    _keep("item-1", "Rerun the live suite at the fix head."),
                    _resolve("item-2", "The gate now covers submodules."),
                ],
                hr=True,
            ),
        ]
    )
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123-coder-1"):
        run_pr_loop(runner, pr_number=77, config=config)
    releases = _release_records(runner)
    assert [record.metadata.evidence_release.reason for record in releases] == ["findings"]
    released = [
        item for item in releases[0].metadata.prior_items
        if item.obligation_kind == "human-exact-head-evidence"
    ]
    assert [item.lifecycle for item in released] == ["evidence_deferred"]
    prompts = _claude_prompts(runner)
    assert len(prompts) == 1 and "submodule config gate" in prompts[0]
    # The fix head gets a fresh freeze; abc123 evidence did not carry forward.
    assert [record.metadata.subject for record in _freeze_records(runner)] == [
        "abc123", "abc123-coder-1"
    ]


def test_failing_check_while_frozen_releases_before_repair(tmp_path):
    """Row new-obligation-while-frozen (failing check, below the ceiling)."""
    runner = _frozen_runner(tmp_path, claude_outputs=[structured_coder_followup(addressed_items=["item-2"])])
    config = make_config(tmp_path, max_rounds=3)
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    runner.pr_check_runs_payload = {
        "check_runs": [{"name": "test", "status": "completed", "conclusion": "failure"}]
    }
    # After the repair the checks pass again at the new head.
    runner.codex_outputs.append(
        _evidence_review(
            state="blocking",
            dispositions=[_keep("item-1", "Rerun at the repair head."), _resolve("item-2", "Checks fixed.")],
        )
    )

    real_run = runner.run

    def checks_recover_after_coder(args, **kwargs):
        if _claude_prompts(runner):
            runner.pr_check_runs_payload = {
                "check_runs": [{"name": "test", "status": "completed", "conclusion": "success"}]
            }
        return real_run(args, **kwargs)

    runner.run = checks_recover_after_coder
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123-coder-1"):
        run_pr_loop(runner, pr_number=77, config=config)
    phases = _phases(runner)
    release_index = phases.index("evidence-release")
    releases = _release_records(runner)
    assert releases[0].metadata.evidence_release.reason == "machine-gate"
    kinds = {item.obligation_kind for item in releases[0].metadata.prior_items}
    assert {"github-pr-checks", "human-exact-head-evidence"} <= kinds
    coder_records = [record for record in _pr_records(runner) if record.metadata.role == "coder"]
    assert coder_records and coder_records[0].metadata.round_number == 2
    assert coder_records[0].index > _release_records(runner)[0].index
    assert release_index < len(phases)


def test_merge_conflict_while_frozen_releases_before_conflict_resolution(tmp_path):
    """Row new-obligation-while-frozen (merge-conflict variant)."""
    runner = _frozen_runner(tmp_path)
    config = make_config(tmp_path, max_rounds=3)
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    runner.mergeability_payloads = [
        {"mergeable": "CONFLICTING", "mergeStateStatus": "DIRTY", "headRefOid": "abc123", "baseRefName": "main"},
    ]
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=config)
    releases = _release_records(runner)
    assert [record.metadata.evidence_release.reason for record in releases] == ["machine-gate"]
    assert "merge conflict" in releases[0].body
    prompts = _claude_prompts(runner)
    assert len(prompts) == 1 and _EVIDENCE_TEXT not in prompts[0]


def test_machine_gate_release_at_the_ceiling_survives_restart_without_refreezing(tmp_path):
    """Row new-obligation-while-frozen (R == effective allowed_rounds, then restart)."""
    runner = _frozen_runner(tmp_path)
    config = make_config(tmp_path, max_rounds=1)
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    runner.pr_check_runs_payload = {
        "check_runs": [{"name": "test", "status": "completed", "conclusion": "failure"}]
    }
    with pytest.raises(AgentLoopError, match="github-pr-checks") as first:
        run_pr_loop(runner, pr_number=77, config=config)
    assert not isinstance(first.value, HumanDecisionRequiredError)
    assert len(_release_records(runner)) == 1
    comments_before = len(runner.pr_payload["comments"])

    with pytest.raises(AgentLoopError, match="github-pr-checks") as second:
        run_pr_loop(runner, pr_number=77, config=config)
    assert not isinstance(second.value, HumanDecisionRequiredError)
    assert len(_release_records(runner)) == 1
    assert len(_freeze_records(runner)) == 1
    assert len(runner.pr_payload["comments"]) == comments_before
    assert not _claude_prompts(runner)


def test_release_budget_below_the_ceiling_is_restored_after_restart(tmp_path, monkeypatch):
    """Row new-obligation-while-frozen (watcher-extended budget, then restart)."""
    runner = _frozen_runner(tmp_path, claude_outputs=[structured_coder_followup(addressed_items=["item-2"])])
    config = make_config(tmp_path, max_rounds=1)
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    _retag_freeze_budget(runner, allowed_rounds=2, watch_failure_extension_used=True)
    runner.pr_check_runs_payload = {
        "check_runs": [{"name": "test", "status": "completed", "conclusion": "failure"}]
    }
    real_post = pr_loop_support.post_pr_comment
    stops = []

    def stop_after_release(runner_, *, config, pr_number, body):
        if _release_records(runner) and not stops:
            stops.append(True)
            raise AgentLoopError("simulated process stop after the release")
        return real_post(runner_, config=config, pr_number=pr_number, body=body)

    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_loop,
    ), "post_pr_comment", stop_after_release)
    with pytest.raises(AgentLoopError, match="simulated process stop"):
        run_pr_loop(runner, pr_number=77, config=config)
    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop, issue_loop), "post_pr_comment", real_post)
    release = _release_records(runner)[0].metadata
    assert release.evidence_release.allowed_rounds == 2
    assert _pr_records(runner)[-1].metadata.phase == "evidence-release"

    # Restart: the release budget authorizes the R+1 repair round.
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=config)
    coder_records = [record for record in _pr_records(runner) if record.metadata.role == "coder"]
    assert len(coder_records) == 1 and coder_records[0].metadata.round_number == 2
    assert len(_freeze_records(runner)) == 1


def test_malformed_persisted_freeze_blocks_without_approval(tmp_path):
    """Row malformed-evidence-record (orchestrator)."""
    runner = _frozen_runner(tmp_path)
    config = make_config(tmp_path)
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    _retag_freeze_budget(runner, evidence_identities=("not-the-frozen-identity",))
    runner.pr_payload["comments"].append(_signed("Live suite at abc123: 11 passed.", 1))
    with pytest.raises(AgentLoopError, match="malformed or contradictory persisted exact-head evidence"):
        run_pr_loop(runner, pr_number=77, config=config)
    commands = [cmd for cmd, _cwd in runner.commands]
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd in commands)
    assert not _release_records(runner)


def test_selective_policy_signed_input_with_pending_evidence_stays_in_the_round(
    tmp_path, monkeypatch
):
    """Row signed-input-before-freeze under selective scheduling at the ceiling."""
    runner = FakeRunner(
        codex_outputs=[
            _evidence_review(evidence=[_EVIDENCE_TEXT]),
            _evidence_review(
                evidence=[_EVIDENCE_TEXT], hr=True,
                dispositions=[_resolve("item-1", "Still needed; not a code issue.")],
            ),
        ],
        gemini_outputs=[
            _evidence_review(reviewer="Google Gemini"),
            _evidence_review(
                reviewer="Google Gemini", hr=True,
                dispositions=[_resolve("item-1", "Codex owns this request.")],
            ),
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        pr_review_policy="selective-intermediate",
        max_rounds=1,
    )
    injected = _inject_signed_input_before(
        monkeypatch, runner, body="Keep the legacy flag.",
        when=lambda: len(runner.codex_outputs) == 1 and len(runner.gemini_outputs) == 1,
    )
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert injected
    assert runner.codex_outputs == [] and runner.gemini_outputs == []
    freezes = _freeze_records(runner)
    assert len(freezes) == 1 and freezes[0].metadata.round_number == 1


def test_head_moving_at_the_gate_read_publishes_no_freeze(tmp_path, monkeypatch):
    """Row freeze-on-clean-board (variant: live head is H' at the last read)."""
    runner = _frozen_runner(tmp_path)
    config = make_config(tmp_path, max_rounds=1)
    real_context = pr_loop_support.get_pr_review_context
    moved = []

    def head_moves_before_gate(runner_, **kwargs):
        if not moved and not runner.codex_outputs:
            moved.append(True)
            runner.pr_payload["headRefOid"] = "def456"
            runner.git_head = "def456"
        return real_context(runner_, **kwargs)

    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_implementation, plan_first_loop, issue_loop,
    ), "get_pr_review_context", head_moves_before_gate)
    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=config)
    assert moved
    assert not isinstance(excinfo.value, HumanDecisionRequiredError)
    assert "final barrier" in str(excinfo.value) or "exact-head evidence" in str(excinfo.value)
    assert "evidence-freeze" not in _phases(runner)
    assert not any("Exact-head evidence freeze" in comment for comment in runner.comments)


def test_refresh_pass_with_deferred_evidence_survives_a_pending_check_stop(tmp_path, monkeypatch):
    """Rows signed-input-before-freeze and clean-reviews-check-not-green: a
    refresh pass ends with one terminal record even when checks then pend."""
    runner = _frozen_runner(
        tmp_path,
        codex_outputs=[
            _evidence_review(
                evidence=[_EVIDENCE_TEXT], hr=True,
                dispositions=[_resolve("item-1", "Still needed; not a code issue.")],
            )
        ],
    )
    config = make_config(tmp_path)
    _inject_signed_input_before(
        monkeypatch, runner, body="Keep the legacy flag.",
        when=lambda: len(runner.codex_outputs) == 1,
    )
    real_checks = pr_loop.get_pr_checks
    pending = {"on": True}

    def checks_pending_after_refresh(*args, **kwargs):
        if pending["on"] and not runner.codex_outputs:
            runner.pr_check_runs_payload = {
                "check_runs": [{"name": "test", "status": "in_progress", "conclusion": None}]
            }
        return real_checks(*args, **kwargs)

    monkeypatch.setattr(pr_loop, "get_pr_checks", checks_pending_after_refresh)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    releases = _release_records(runner)
    assert [record.metadata.evidence_release.reason for record in releases] == ["refresh-clean"]
    pass_reviews = [
        record for record in _pr_records(runner)
        if record.metadata.phase == "evidence-response"
    ]
    assert len(pass_reviews) == 1 and pass_reviews[0].index < releases[0].index
    released = releases[0].metadata
    assert set(pass_reviews[0].metadata.surfaced_reviewer_requirement_ids) == set(
        released.evidence_release.signed_requirement_ids_surfaced
    )
    assert [item.lifecycle for item in released.prior_items
            if item.obligation_kind == "human-exact-head-evidence"] == ["evidence_deferred"]
    assert "evidence-freeze" not in _phases(runner)

    # Restart once checks pass: the pass is kept, no reviewer re-runs, and the
    # deferred evidence is frozen with the refreshed baseline.
    pending["on"] = False
    runner.pr_check_runs_payload = {
        "check_runs": [{"name": "test", "status": "completed", "conclusion": "success"}]
    }
    commands_before = len(runner.commands)
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert not any(cmd[:1] == ["codex"] for cmd, _cwd in runner.commands[commands_before:])
    freezes = _freeze_records(runner)
    assert len(freezes) == 1
    assert set(freezes[0].metadata.evidence_freeze.signed_requirement_ids_at_freeze) == set(
        released.evidence_release.signed_requirement_ids_surfaced
    )
