"""validated_agent tests: retry, repair, normalization, replay and usage-context recovery (#1544)."""

import errno
import json
import time
from unittest.mock import patch

import pytest

from coding_review_agent_loop.agents.base import AgentResult
from coding_review_agent_loop.agents.gemini import PUBLIC_RESPONSE_MARKER
from coding_review_agent_loop.cli import (
    AgentLoopError,
    run_pr_loop,
)
from coding_review_agent_loop.containment import ContainmentEvidence
from coding_review_agent_loop.errors import AgentInvocationError, QuotaResetExceededError
from coding_review_agent_loop.runner import CommandResult, ExecutableIdentity, ExecutionObservation
from coding_review_agent_loop.usage import RunUsageContext
from coding_review_agent_loop.validated_agent import CompletionRecoveryPolicy, _run_validated_agent
from coding_review_agent_loop.pr_loop import _validate_coder_followup_response, _validate_review_response
from coding_review_agent_loop.plan_first_loop import _validate_plan_review_response
from coding_review_agent_loop.protocol import (
    UnresolvedReviewItem,
    validate_structured_issue_implementation,
)
from coding_review_agent_loop.salvage import SalvageContext
from agent_loop_helpers import (
    FakeRunner,
    make_config,
    structured_coder_followup,
    structured_issue_implementation,
    structured_plan_review,
    structured_plan_state,
    structured_pr_review,
)
import coding_review_agent_loop.plan_first_loop as plan_first_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.response_validation as response_validation
import coding_review_agent_loop.validated_agent as validated_agent
from agent_loop_helpers import patch_attr_in, patch_object_in
from agent_loop_helpers import _CAPACITY


def test_provider_cli_effort_rejection_is_classified_without_retry(tmp_path):
    runner = FakeRunner(claude_outputs=[("unknown option: --effort", 1)])
    config = make_config(tmp_path, coder="claude", agent_max_retries=0)

    with pytest.raises(AgentInvocationError) as exc_info:
        _run_validated_agent(
            runner,
            agent="claude",
            config=config,
            prompt="Review the PR.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: text,
            role="reviewer",
        )

    assert exc_info.value.failure_category == "unsupported_effort"
    assert "provider CLI rejected the explicit --effort setting" in str(exc_info.value)
    assert not any(command[:1] == ["sleep"] for command, _cwd in runner.commands)


def test_gemini_pre_marker_429_does_not_suppress_structured_review_repair(tmp_path):
    malformed_public_review = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                # Issue #871: preservation now also runs on the legacy repair
                # hook for reviewer kinds, so the repaired summary must carry
                # the reviewer's own wording rather than invented text.
                "summary": "Review passed after repair.",
                "blocking_items": [],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": [],
            }
        )
        + "\nProse between JSON and footer should be repaired.\n"
        "<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    )
    raw_stdout = (
        "Attempt 1 failed with status 429. Retrying with backoff... "
        "No capacity available for model gemini-3-flash-preview on the server.\n"
        f"{PUBLIC_RESPONSE_MARKER}\n"
        f"{malformed_public_review}"
    )
    repaired_review = structured_pr_review(
        state="approved",
        summary="Review passed after repair.",
        reviewer="Google Gemini",
    )
    runner = FakeRunner(gemini_outputs=[{"stdout": raw_stdout}])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)
    captured_repairs = []

    def fake_attempt_repair(raw: str, gemini_cmd: str, *, expected_kind: str | None = None, unresolved_item_ids=None, surfaced_requirement_ids=None, requires_direct_discussion_ack=False, allowed_prior_item_ids=None, unknown_prior_item_ids=None, same_round_context=None) -> str | None:
        captured_repairs.append(raw)
        assert expected_kind == "pr_review"
        return repaired_review

    with patch("coding_review_agent_loop.validated_agent.attempt_repair", fake_attempt_repair):
        assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert captured_repairs == [malformed_public_review]
    assert "429" not in captured_repairs[0]
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)
    assert any("Review passed after repair." in comment for comment in runner.comments)


def test_gemini_response_file_repair_ignores_raw_stdout_transient_diagnostics(tmp_path):
    malformed_public_review = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                # Issue #871: the repaired summary must carry the reviewer's own
                # wording, and an approved source may not carry a current-scope
                # finding that repair would have to drop to keep the approval.
                "summary": "Response file review passed after repair.",
                "blocking_items": [],
                "same_pr_followups": "not-a-list",
                "future_followups": [],
                "prior_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    )
    repaired_review = structured_pr_review(
        state="approved",
        summary="Response file review passed after repair.",
        reviewer="Google Gemini",
    )
    runner = FakeRunner(
        gemini_outputs=[
            {"stdout": "Attempt 1 failed with status 429. No capacity available, then recovered."}
        ],
        public_response_outputs=[{"text": malformed_public_review}],
    )
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)
    captured_repairs = []

    def fake_attempt_repair(raw: str, gemini_cmd: str, *, expected_kind: str | None = None, unresolved_item_ids=None, surfaced_requirement_ids=None, requires_direct_discussion_ack=False, allowed_prior_item_ids=None, unknown_prior_item_ids=None, same_round_context=None) -> str | None:
        captured_repairs.append(raw)
        return repaired_review

    with patch("coding_review_agent_loop.validated_agent.attempt_repair", fake_attempt_repair):
        assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert captured_repairs == [malformed_public_review]
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)
    assert any("Response file review passed after repair." in comment for comment in runner.comments)


def test_diagnostic_shaped_public_response_remains_transient(tmp_path):
    public_response = (
        f"{PUBLIC_RESPONSE_MARKER}\n"
        "HTTP 429 Too Many Requests: rate limit exceeded.\n"
        "<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    )
    runner = FakeRunner(gemini_outputs=[{"stdout": public_response}])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.validated_agent.attempt_repair") as repair_mock:
        with pytest.raises(AgentLoopError) as exc_info:
            run_pr_loop(runner, pr_number=77, config=config)

    repair_mock.assert_not_called()
    assert "Failure category: transient" in str(exc_info.value)
    assert "Suggestion:" in str(exc_info.value)


def test_codex_chatgpt_unsupported_model_public_response_skips_retry_repair_and_salvages(tmp_path):
    error_payload = json.dumps(
        {
            "error": {
                "type": "invalid_request_error",
                "message": (
                    "The 'gpt-5.5-pro' model is not supported when using "
                    "Codex with a ChatGPT account."
                ),
            }
        }
    )
    runner = FakeRunner(
        codex_outputs=[{"stdout": error_payload, "public_response": "", "returncode": 0}],
        public_response_outputs=[{"text": error_payload}],
        git_diff=(
            "diff --git a/src/coding_review_agent_loop/cli.py "
            "b/src/coding_review_agent_loop/cli.py\n"
            "--- a/src/coding_review_agent_loop/cli.py\n"
            "+++ b/src/coding_review_agent_loop/cli.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        ),
        git_status=" M src/coding_review_agent_loop/cli.py\n",
    )
    config = make_config(
        tmp_path,
        coder="codex",
        reviewer="codex",
        codex_model="gpt-5.5-pro",
        agent_max_retries=2,
    )
    from coding_review_agent_loop.checkout_verification import establish_initial_baseline

    # The scripted dirty tree is this checkout's prepared state.
    establish_initial_baseline(config, runner, config.codex_dir)

    with patch("coding_review_agent_loop.validated_agent.attempt_repair") as repair_mock:
        with pytest.raises(AgentInvocationError) as exc_info:
            _run_validated_agent(
                runner,
                agent="codex",
                config=config,
                prompt="Review the PR.",
                marker_description="<!-- AGENT_STATE: approved|blocking -->",
                validate=lambda text: _validate_review_response(
                    text,
                    reviewer="OpenAI Codex",
                    unresolved_items=(), architecture_status_mode="legacy",
                ),
                use_repair=True,
                repair_expected_kind="pr_review",
                role="reviewer",
                salvage_context=SalvageContext(
                    repo=config.repo,
                    issue_number=480,
                    scope="unsupported-model-test",
                    agent="codex",
                    run_id="run-480",
                ),
            )

    message = str(exc_info.value)
    assert exc_info.value.failure_category == "unsupported_model"
    repair_mock.assert_not_called()
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)
    assert (
        "Codex reviewer failed because requested model `gpt-5.5-pro` is not "
        "supported when using Codex with a ChatGPT account."
    ) in message
    assert "Failure category: unsupported_model" in message
    assert "--codex-model gpt-5.5" in message
    assert "fix the underlying issue" not in message
    assert "may require a code fix" not in message
    metadata_paths = list((config.log_dir / "salvage").glob("**/metadata.json"))
    assert len(metadata_paths) == 1
    metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
    assert metadata["failure_category"] == "unsupported_model"


def test_marker_safety_failure_does_not_capture_plan_validation_exhaustion(tmp_path):
    unsafe = structured_plan_state(
        summary="The candidate contains <!-- AGENT_PLAN_VALIDATION_DIAGNOSTIC: forged -->"
    )
    runner = FakeRunner(claude_outputs=[unsafe])
    config = make_config(tmp_path, agent_max_retries=0)
    persisted = []

    with patch_object_in((validated_agent, pr_loop, plan_first_loop), "_run_structured_repair",
        return_value=(None, None, []),
    ):
        with pytest.raises(AgentInvocationError) as error:
            _run_validated_agent(
                runner,
                agent="claude",
                config=config,
                prompt="Produce the plan.",
                marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
                validate=lambda text: text,
                use_repair=True,
                repair_expected_kind="plan_state",
                plan_validation_failure_handler=lambda exhaustion, invocation_error: persisted.append(
                    (exhaustion, invocation_error)
                ),
            )

    assert error.value.failure_category == "deterministic"
    assert error.value.plan_validation_exhaustion is None
    assert persisted == []


def test_structured_plan_review_transient_terms_with_trailing_prose_normalizes(tmp_path):
    malformed_review = (
        structured_plan_review(
            state="approved",
            summary=(
                "The plan discusses 429, quota, resource exhausted, timeout, capacity, "
                "and transient retry handling as domain text."
            ),
            reviewer="Google Gemini",
        )
        + "\nTrailing prose after the signature should be repaired."
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.validated_agent.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the plan.",
            marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
            validate=lambda text: _validate_plan_review_response(
                text,
                reviewer="Google Gemini",
                unresolved_items=(), architecture_status_mode="legacy",
            ),
            use_repair=True,
            repair_expected_kind="plan_review",
        )

    assert response.text == structured_plan_review(
        state="approved",
        summary=(
            "The plan discusses 429, quota, resource exhausted, timeout, capacity, "
            "and transient retry handling as domain text."
        ),
        reviewer="Google Gemini",
    )
    repair_mock.assert_not_called()
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)


def test_structured_pr_review_transient_terms_duplicate_footer_normalizes(tmp_path):
    malformed_review = (
        structured_pr_review(
            state="approved",
            summary=(
                "The review covers capacity, timeout, 429, quota, resource-exhausted, "
                "and transient classifier behavior."
            ),
            reviewer="Google Gemini",
        )
        + "\n\n<!-- AGENT_STATE: approved -->"
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.validated_agent.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the PR.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: _validate_review_response(
                text,
                reviewer="Google Gemini",
                unresolved_items=(), architecture_status_mode="legacy",
            ),
            use_repair=True,
            repair_expected_kind="pr_review",
        )

    assert response.text == structured_pr_review(
        state="approved",
        summary=(
            "The review covers capacity, timeout, 429, quota, resource-exhausted, "
            "and transient classifier behavior."
        ),
        reviewer="Google Gemini",
    )
    repair_mock.assert_not_called()
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)


def test_structured_coder_followup_transient_terms_before_footer_runs_repair(tmp_path):
    unresolved_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="OpenAI Codex",
            source_round=1,
            text="Add timeout regression coverage.",
            status="blocking",
        ),
    )
    malformed_followup = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "approved",
                "summary": "Updated timeout and capacity handling without treating prose as transient.",
                "addressed_items": ["item-1"],
                "remaining_items": [],
                "human_requirement_dispositions": [],
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n## Changes made\nMentioned timeout and capacity in prose before the footer.\n"
        "<!-- AGENT_STATE: approved -->\n-- Anthropic Claude"
    )
    repaired_followup = structured_coder_followup(
        state="approved",
        summary="Updated timeout and capacity handling.",
        addressed_items=["item-1"],
        remaining_items=[],
        reviewer="Anthropic Claude",
    )
    runner = FakeRunner(claude_outputs=[malformed_followup])
    config = make_config(tmp_path, coder="claude", agent_max_retries=0)

    with patch("coding_review_agent_loop.validated_agent.attempt_repair", return_value=repaired_followup) as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="claude",
            config=config,
            prompt="Address review feedback.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: _validate_coder_followup_response(
                text,
                unresolved_items=unresolved_items,
                human_requirements=(), architecture_status_mode="legacy",
            ),
            use_repair=True,
            repair_expected_kind="coder_followup",
        )

    assert response.text == repaired_followup
    repair_mock.assert_called_once_with(
        malformed_followup,
        config.gemini_cmd,
        expected_kind="coder_followup",
    )
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)


def test_run_validated_agent_recovers_coder_followup_from_message_text_when_response_file_markdown(tmp_path):
    unresolved_items = (
        UnresolvedReviewItem(
            item_id="item-8",
            reviewer="Orchestrator",
            source_round=4,
            text="Acknowledge signed human requirements.",
            status="blocking",
        ),
    )
    valid_followup = structured_coder_followup(
        state="approved",
        summary="Acknowledged the signed human requirements.",
        addressed_items=["item-8"],
        remaining_items=[],
        reviewer="OpenAI Codex",
    )
    markdown_response_file = (
        "### Human requirements\n\n"
        "Acknowledged.\n\n"
        "<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n"
        "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    )
    runner = FakeRunner(
        codex_outputs=[{"public_response": valid_followup, "stdout": "diagnostic output"}],
        public_response_outputs=[{"text": markdown_response_file}],
    )
    config = make_config(tmp_path, coder="codex", agent_max_retries=0)

    response = _run_validated_agent(
        runner,
        agent="codex",
        config=config,
        prompt="Address feedback.",
        marker_description="<!-- AGENT_STATE: approved|blocking -->",
        validate=lambda text: _validate_coder_followup_response(
            text,
            unresolved_items=unresolved_items,
            human_requirements=(), architecture_status_mode="legacy",
        ),
        repair_expected_kind="coder_followup",
    )

    assert response.text == valid_followup
    assert response.marker_value.addressed_items == ("item-8",)


def test_run_validated_agent_recovers_fenced_coder_followup_from_raw_stdout(tmp_path):
    unresolved_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="OpenAI Codex",
            source_round=1,
            text="Fix the bug.",
            status="blocking",
        ),
    )
    valid_followup = structured_coder_followup(
        state="approved",
        addressed_items=["item-1"],
        remaining_items=[],
        reviewer="OpenAI Codex",
    )
    json_part, footer = valid_followup.split("\n<!-- AGENT_STATE:", 1)
    fenced_stdout = f"tool diagnostic\n```json\n{json_part}\n```\n<!-- AGENT_STATE:{footer}"
    runner = FakeRunner(
        codex_outputs=[{"public_response": "legacy markdown", "stdout": fenced_stdout}],
        public_response_outputs=[{"text": "### Update\nFixed it.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"}],
    )
    config = make_config(tmp_path, coder="codex", agent_max_retries=0)

    response = _run_validated_agent(
        runner,
        agent="codex",
        config=config,
        prompt="Address feedback.",
        marker_description="<!-- AGENT_STATE: approved|blocking -->",
        validate=lambda text: _validate_coder_followup_response(
            text,
            unresolved_items=unresolved_items,
            human_requirements=(), architecture_status_mode="legacy",
        ),
        repair_expected_kind="coder_followup",
    )

    assert response.text == valid_followup


def test_gemini_duplicate_trailing_agent_state_marker_normalizes_without_repair(tmp_path):
    malformed_public_review = (
        structured_pr_review(
            state="approved",
            summary="Found one issue.",
            reviewer="Google Gemini",
        )
        + "\n\n<!-- AGENT_STATE: approved -->"
    )
    raw_stdout = f"{PUBLIC_RESPONSE_MARKER}\n{malformed_public_review}"
    runner = FakeRunner(gemini_outputs=[{"stdout": raw_stdout}])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.validated_agent.attempt_repair") as repair_mock:
        assert run_pr_loop(runner, pr_number=77, config=config) == 0

    repair_mock.assert_not_called()
    assert any("Found one issue." in comment for comment in runner.comments)
    assert all(comment.count("<!-- AGENT_STATE: approved -->") == 1 for comment in runner.comments)


def test_gemini_pre_marker_429_malformed_public_response_fails_deterministically(tmp_path):
    malformed_public_review = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "Found one issue.",
                "blocking_items": [],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": [],
            }
        )
        + "\nExtra prose before the footer.\n"
        "<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    )
    raw_stdout = (
        "Attempt 1 failed with status 429. No capacity available for model gemini.\n"
        f"{PUBLIC_RESPONSE_MARKER}\n"
        f"{malformed_public_review}"
    )
    runner = FakeRunner(gemini_outputs=[{"stdout": raw_stdout}])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.validated_agent.attempt_repair", return_value="still invalid"):
        with pytest.raises(AgentLoopError) as exc_info:
            run_pr_loop(runner, pr_number=77, config=config)

    message = str(exc_info.value)
    # The internal category stays deterministic (no transient auto-retry), but
    # the operator label reflects a stochastic schema miss (#957).
    assert getattr(exc_info.value, "failure_category", "deterministic") == "deterministic"
    assert "Failure category: schema-validation" in message
    assert "Failure category: transient" not in message


def test_malformed_structured_review_model_support_terms_still_runs_repair(tmp_path):
    malformed_review = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": (
                    "The review discusses unsupported model and model availability "
                    "diagnostics as domain text."
                ),
                "blocking_items": [],
                "same_pr_followups": "not-a-list",
                "future_followups": [],
                "prior_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    )
    repaired_review = structured_pr_review(
        state="approved",
        # Issue #871: grounded in the source summary above, which repair may
        # re-envelope but never replace with invented text.
        summary=(
            "The review discusses unsupported model and model availability "
            "diagnostics as domain text."
        ),
        reviewer="Google Gemini",
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.validated_agent.attempt_repair", return_value=repaired_review) as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the PR.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: _validate_review_response(
                text,
                reviewer="Google Gemini",
                unresolved_items=(), architecture_status_mode="legacy",
            ),
            use_repair=True,
            repair_expected_kind="pr_review",
        )

    assert response.text == repaired_review
    repair_mock.assert_called_once_with(
        malformed_review,
        config.gemini_cmd,
        expected_kind="pr_review",
    )
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)


def test_coder_validation_keeps_acquisition_catalog_through_format_repair(tmp_path, monkeypatch):
    malformed = structured_coder_followup(summary="Needs envelope repair") + "\ntrailing text"
    repaired = structured_coder_followup(summary="Repaired semantic response")
    acquisition_observation = object()
    result = AgentResult(
        text=malformed,
        returncode=0,
        test_turn_id="coder-turn",
        test_turn_observations=(acquisition_observation,),
    )
    runner = FakeRunner()
    seen_catalogs = []

    def validate(text):
        catalog = response_validation._current_test_turn_observations(runner)
        seen_catalogs.append(catalog)
        if text != repaired:
            raise AgentLoopError("malformed response")
        assert catalog == (acquisition_observation,)
        return text

    def repair_with_new_broker_turn(raw, *, runner, validate, **kwargs):
        # A real repair starts a new containment/broker turn. The acquisition
        # context must still resolve the coder's original selector catalog.
        runner._latest_test_turn_id = "repair-turn"
        runner._local_test_observations = []
        parsed = validate(repaired)
        return repaired, parsed, []

    patch_attr_in(monkeypatch, (validated_agent, pr_loop, plan_first_loop), "_run_structured_repair", repair_with_new_broker_turn)
    with patch_object_in((validated_agent, response_validation), "run_agent_result", return_value=result):
        response = _run_validated_agent(
            runner,
            agent="claude",
            config=make_config(tmp_path),
            prompt="Provide the coder follow-up.",
            marker_description="structured coder follow-up",
            validate=validate,
            role="coder",
            use_repair=True,
            repair_expected_kind="coder_followup",
        )

    assert response.text == repaired
    assert response.acquisition_test_turn_id == "coder-turn"
    assert response.acquisition_test_observations == (acquisition_observation,)
    assert len(seen_catalogs) >= 2
    assert all(catalog == (acquisition_observation,) for catalog in seen_catalogs)


def test_claude_completion_recovery_keeps_original_acquisition_catalog(
    tmp_path, monkeypatch
):
    original_observation = object()
    recovery_observation = object()
    original_result = AgentResult(
        text="I'll wait for the background tests.",
        session_id="claude-session",
        test_turn_id="original-coder-turn",
        test_turn_observations=(original_observation,),
    )
    recovery_text = structured_issue_implementation(pr_number=77)
    recovery_result = AgentResult(
        text=recovery_text,
        session_id="claude-session",
        test_turn_id="completion-recovery-turn",
        test_turn_observations=(recovery_observation,),
    )
    runner = FakeRunner()
    config = make_config(tmp_path, coder="claude")

    def validate(text):
        if text == recovery_text:
            return validate_structured_issue_implementation(text)
        raise AgentLoopError("response deferred to background work")

    with patch_object_in((validated_agent, response_validation), "run_agent_result",
        side_effect=[original_result, recovery_result],
    ):
        response = _run_validated_agent(
            runner,
            agent="claude",
            config=config,
            prompt="Finish the implementation.",
            marker_description="structured issue implementation",
            validate=validate,
            role="coder",
            completion_recovery=CompletionRecoveryPolicy(issue_number=56),
        )

    assert response.marker_value.pr_number == 77
    assert response.acquisition_test_turn_id == "original-coder-turn"
    assert response.acquisition_test_observations == (original_observation,)


def test_claude_self_update_replay_recovers_valid_response_with_remaining_timeout(tmp_path):
    """A spawned updater interruption gets one stable, bounded full replay."""
    identity = ExecutableIdentity("claude", "claude", (1, 1, 1), (1, 1, 1))
    observation = ExecutionObservation(
        spawn_wall_time=1,
        spawn_monotonic=time.monotonic() - 1,
        exit_monotonic=time.monotonic(),
        elapsed_seconds=1,
        before=identity,
        after=identity,
        interrupted=False,
    )

    class ObservedClaudeRunner(FakeRunner):
        def __init__(self):
            super().__init__(claude_outputs=[
                ("Loading...\nInstalling Claude Code v2.1.226", 1),
                ("overloaded_error", 1),
                (structured_pr_review(state="approved", summary="Recovered.", reviewer="Anthropic Claude"), 0),
            ])
            self.timeouts = []
            self.claude_log_paths = []

        def run_with_log(self, *args, timeout_seconds=None, **kwargs):
            self.timeouts.append(timeout_seconds)
            if args[0][:1] == ["claude"]:
                self.claude_log_paths.append(kwargs["log_path"])
            result = super().run_with_log(*args, timeout_seconds=timeout_seconds, **kwargs)
            if result.args[:1] == ["claude"]:
                return CommandResult(
                    result.args, result.cwd, result.stdout, result.stderr, result.returncode, observation
                )
            return result

        def wait_for_executable_stability(self, command, *, deadline=None):
            assert command == "claude"
            assert deadline is not None
            return True

    runner = ObservedClaudeRunner()
    config = make_config(tmp_path, reviewer="claude", agent_max_retries=1)
    response = _run_validated_agent(
        runner,
        agent="claude",
        config=config,
        prompt="Review the PR.",
        marker_description="<!-- AGENT_STATE: approved|blocking -->",
        validate=lambda text: _validate_review_response(
            text, reviewer="Anthropic Claude", unresolved_items=(), architecture_status_mode="legacy"
        ),
        timeout_seconds=12,
    )

    assert response.marker_value.state == "approved"
    assert len([cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]) == 3
    assert runner.timeouts[0] == 12
    assert 0 < runner.timeouts[1] < 12
    assert runner.timeouts[2] == 12
    assert runner.claude_log_paths[0].name.endswith("-attempt1.log")
    assert runner.claude_log_paths[1].name.endswith("-self-update-attempt2.log")
    assert runner.claude_log_paths[2].name.endswith("-attempt2.log")


def test_claude_self_update_stability_failure_preserves_ordinary_retry_budget(tmp_path):
    """A transient updater failure still receives the configured ordinary retry."""
    identity = ExecutableIdentity("claude", "claude", (1, 1, 1), (1, 1, 1))
    observation = ExecutionObservation(1, time.monotonic(), time.monotonic(), 1, identity, identity, False)

    class UnstableClaudeRunner(FakeRunner):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.claude_log_paths = []

        def run_with_log(self, *args, **kwargs):
            if args[0][:1] == ["claude"]:
                self.claude_log_paths.append(kwargs["log_path"])
            result = super().run_with_log(*args, **kwargs)
            if result.args[:1] == ["claude"]:
                return CommandResult(
                    result.args, result.cwd, result.stdout, result.stderr, result.returncode, observation
                )
            return result

        def wait_for_executable_stability(self, command, *, deadline=None):
            return False

    runner = UnstableClaudeRunner(claude_outputs=[
        ("fatal: auto-update in progress\noverloaded_error", 1),
        (structured_pr_review(state="approved", summary="Recovered by ordinary retry.", reviewer="Anthropic Claude"), 0),
    ])
    config = make_config(tmp_path, reviewer="claude", agent_max_retries=1)
    response = _run_validated_agent(
        runner,
        agent="claude",
        config=config,
        prompt="Review the PR.",
        marker_description="<!-- AGENT_STATE: approved|blocking -->",
        validate=lambda text: _validate_review_response(
            text, reviewer="Anthropic Claude", unresolved_items=(), architecture_status_mode="legacy"
        ),
    )

    assert response.marker_value.state == "approved"
    assert len([cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]) == 2
    assert [path.name.rsplit("-", 1)[-1] for path in runner.claude_log_paths] == [
        "attempt1.log",
        "attempt2.log",
    ]


def test_claude_self_update_stability_failure_sets_final_category(tmp_path):
    """An exhausted failed stability check remains explicitly diagnosable."""
    identity = ExecutableIdentity("claude", "claude", (1, 1, 1), (1, 1, 1))
    observation = ExecutionObservation(1, time.monotonic(), time.monotonic(), 1, identity, identity, False)

    class UnstableClaudeRunner(FakeRunner):
        def run_with_log(self, *args, **kwargs):
            result = super().run_with_log(*args, **kwargs)
            if result.args[:1] == ["claude"]:
                return CommandResult(
                    result.args, result.cwd, result.stdout, result.stderr, result.returncode, observation
                )
            return result

        def wait_for_executable_stability(self, command, *, deadline=None):
            return False

    runner = UnstableClaudeRunner(claude_outputs=[
        ("fatal: auto-update in progress\\noverloaded_error", 1),
        ("ordinary failure", 1),
    ])
    config = make_config(tmp_path, reviewer="claude", agent_max_retries=1)

    with pytest.raises(AgentInvocationError) as error:
        _run_validated_agent(
            runner,
            agent="claude",
            config=config,
            prompt="Review the PR.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: _validate_review_response(
                text, reviewer="Anthropic Claude", unresolved_items=(), architecture_status_mode="legacy"
            ),
        )

    assert error.value.failure_category == "self-update-interruption"
    assert "wait for the Claude Code self-update to finish" in str(error.value)


def _refused_claude_result(tmp_path, *, raw_output="fatal: auto-update in progress", kind="changed-status"):
    identity = ExecutableIdentity("claude", "claude", (1, 1, 1), (1, 1, 1))
    observation = ExecutionObservation(1, 1, 2, 1, identity, identity, False)
    command_result = CommandResult(
        ["claude", "--print"], tmp_path, raw_output, "", 1, observation
    )
    return AgentResult(
        text="",
        raw_output=raw_output,
        returncode=1,
        log_path=tmp_path / "claude-attempt1.log",
        command_result=command_result,
        self_update_reason="Claude Code updater diagnostic",
        self_update_replay_refusal_kind=kind,
        self_update_replay_refusal_detail=(
            "Claude self-update replay refused: dirty workdir status changed during invocation"
            if kind == "changed-status"
            else "Claude self-update replay refused: after workdir snapshot unavailable"
        ),
    )


def test_claude_replay_refusal_is_diagnostic_only_and_does_not_wait_or_override_category(tmp_path):
    result = _refused_claude_result(tmp_path)
    runner = FakeRunner()
    config = make_config(tmp_path, reviewer="claude", agent_max_retries=0)
    usage = RunUsageContext(run_id="run-refusal", summary_path=tmp_path / "usage.json")

    with patch_object_in((validated_agent, response_validation), "run_agent_result", return_value=result):
        with pytest.raises(AgentInvocationError) as error:
            _run_validated_agent(
                runner,
                agent="claude",
                config=config,
                prompt="Review the PR.",
                marker_description="<!-- AGENT_STATE: approved|blocking -->",
                validate=lambda text: _validate_review_response(
                    text, reviewer="Anthropic Claude", unresolved_items=(), architecture_status_mode="legacy"
                ),
                usage_context=usage,
            )

    assert error.value.failure_category == "deterministic"
    assert "dirty workdir" in str(error.value)
    assert not any(command[:1] == ["sleep"] for command, _cwd in runner.commands)
    assert usage.records[0].outcome == "self_update_replay_refused_changed_workdir"
    assert usage.records[0].log_path == str(result.log_path)


def test_transient_claude_replay_refusal_keeps_provider_retry_and_category(tmp_path):
    refused = _refused_claude_result(tmp_path, raw_output="overloaded_error")
    valid = AgentResult(
        text=structured_pr_review(
            state="approved", summary="Recovered after refusal.", reviewer="Anthropic Claude"
        ),
        returncode=0,
    )
    runner = FakeRunner()
    config = make_config(tmp_path, reviewer="claude", agent_max_retries=1)

    with patch_object_in((validated_agent, response_validation), "run_agent_result", side_effect=[refused, valid]):
        response = _run_validated_agent(
            runner,
            agent="claude",
            config=config,
            prompt="Review the PR.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: _validate_review_response(
                text, reviewer="Anthropic Claude", unresolved_items=(), architecture_status_mode="legacy"
            ),
        )

    assert response.marker_value.state == "approved"
    assert any(command[:1] == ["sleep"] for command, _cwd in runner.commands)


def test_transient_claude_replay_refusal_only_retains_transient_category(tmp_path):
    refused = _refused_claude_result(tmp_path, raw_output="overloaded_error")
    runner = FakeRunner()
    config = make_config(tmp_path, reviewer="claude", agent_max_retries=0)

    with patch_object_in((validated_agent, response_validation), "run_agent_result", return_value=refused):
        with pytest.raises(AgentInvocationError) as error:
            _run_validated_agent(
                runner,
                agent="claude",
                config=config,
                prompt="Review the PR.",
                marker_description="<!-- AGENT_STATE: approved|blocking -->",
                validate=lambda text: _validate_review_response(
                    text, reviewer="Anthropic Claude", unresolved_items=(), architecture_status_mode="legacy"
                ),
            )

    assert error.value.failure_category == "transient"
    assert "dirty workdir" in str(error.value)


def _observed_codex_result(tmp_path, *, text="", raw_output="", returncode=1, changed=True):
    before = ExecutableIdentity(
        "/tmp/codex", "/tmp/codex-target", (1, 1, 100_000_000_000), (1, 2, 100_000_000_000)
    )
    after = (
        ExecutableIdentity(
            "/tmp/codex", "/tmp/codex-new-target", (1, 3, 101_000_000_000), (1, 4, 101_000_000_000)
        )
        if changed
        else before
    )
    observation = ExecutionObservation(100, 100, 101, 1, before, after, False)
    command_result = CommandResult(
        ["codex", "exec", "--json"], tmp_path, raw_output, "", returncode, observation
    )
    return AgentResult(
        text=text,
        raw_output=raw_output,
        returncode=returncode,
        command_result=command_result,
        self_update_reason=("Codex executable changed during invocation" if changed else None),
    )


def test_codex_executable_replacement_replay_uses_fresh_full_timeout(tmp_path):
    valid = structured_pr_review(
        state="approved", summary="Recovered.", reviewer="OpenAI Codex"
    )
    results = iter(
        [
            _observed_codex_result(tmp_path, raw_output="overloaded_error"),
            AgentResult(text=valid, raw_output=valid, returncode=0),
        ]
    )
    runner = FakeRunner()
    stability_calls = []
    runner.wait_for_executable_stability = lambda command, *, deadline=None: (
        stability_calls.append((command, deadline)) or True
    )
    config = make_config(tmp_path, reviewer="codex", agent_max_retries=1)

    with patch_object_in((validated_agent, response_validation), "run_agent_result", side_effect=results) as run_mock:
        response = _run_validated_agent(
            runner,
            agent="codex",
            config=config,
            prompt="Review the PR.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: _validate_review_response(
                text, reviewer="OpenAI Codex", unresolved_items=(), architecture_status_mode="legacy"
            ),
            timeout_seconds=60,
        )

    assert response.marker_value.state == "approved"
    assert stability_calls == [(config.codex_cmd, None)]
    assert run_mock.call_count == 2
    assert run_mock.call_args_list[0].kwargs["timeout_seconds"] == 60
    assert run_mock.call_args_list[1].kwargs["timeout_seconds"] == 60
    assert "attempt_suffix" not in run_mock.call_args_list[0].kwargs
    assert run_mock.call_args_list[1].kwargs["attempt_suffix"] == "executable-replacement-attempt2"


def test_codex_replacement_replay_does_not_consume_ordinary_retry(tmp_path):
    valid = structured_pr_review(
        state="approved", summary="Recovered after retry.", reviewer="OpenAI Codex"
    )
    results = iter(
        [
            _observed_codex_result(tmp_path),
            _observed_codex_result(
                tmp_path, raw_output="overloaded_error", changed=False
            ),
            AgentResult(text=valid, raw_output=valid, returncode=0),
        ]
    )
    runner = FakeRunner()
    runner.wait_for_executable_stability = lambda command, *, deadline=None: True
    config = make_config(tmp_path, reviewer="codex", agent_max_retries=1)

    with patch_object_in((validated_agent, response_validation), "run_agent_result", side_effect=results) as run_mock:
        response = _run_validated_agent(
            runner,
            agent="codex",
            config=config,
            prompt="Review the PR.",
            marker_description="test",
            validate=lambda text: _validate_review_response(
                text, reviewer="OpenAI Codex", unresolved_items=(), architecture_status_mode="legacy"
            ),
        )

    assert response.marker_value.state == "approved"
    assert run_mock.call_count == 3
    assert run_mock.call_args_list[1].kwargs["attempt_suffix"] == "executable-replacement-attempt2"
    assert "attempt_suffix" not in run_mock.call_args_list[2].kwargs
    assert len([cmd for cmd, _cwd in runner.commands if cmd[:1] == ["sleep"]]) == 1


def test_codex_replacement_exhaustion_keeps_specific_terminal_category(tmp_path):
    results = iter(
        [
            _observed_codex_result(tmp_path),
            _observed_codex_result(tmp_path, raw_output="ordinary failure", changed=False),
        ]
    )
    runner = FakeRunner()
    runner.wait_for_executable_stability = lambda command, *, deadline=None: True
    config = make_config(tmp_path, reviewer="codex", agent_max_retries=1)

    with patch_object_in((validated_agent, response_validation), "run_agent_result", side_effect=results):
        with pytest.raises(AgentInvocationError) as error:
            _run_validated_agent(
                runner,
                agent="codex",
                config=config,
                prompt="Review the PR.",
                marker_description="test",
                validate=lambda text: _validate_review_response(
                    text, reviewer="OpenAI Codex", unresolved_items=(), architecture_status_mode="legacy"
                ),
            )

    assert error.value.failure_category == "executable-replacement"
    assert "Codex executable replacement" in str(error.value)
    assert "wait for the Codex executable replacement" in str(error.value)


def test_codex_unstable_replacement_retains_ordinary_retry(tmp_path):
    valid = structured_pr_review(
        state="approved", summary="Recovered by ordinary retry.", reviewer="OpenAI Codex"
    )
    results = iter(
        [
            _observed_codex_result(tmp_path),
            AgentResult(text=valid, raw_output=valid, returncode=0),
        ]
    )
    runner = FakeRunner()
    runner.wait_for_executable_stability = lambda command, *, deadline=None: False
    config = make_config(tmp_path, reviewer="codex", agent_max_retries=1)

    # The first failed replacement observation remains subject to normal
    # retry accounting; this test uses a scripted valid ordinary retry.
    with patch_object_in((validated_agent, response_validation), "run_agent_result", side_effect=results) as run_mock:
        response = _run_validated_agent(
            runner,
            agent="codex",
            config=config,
            prompt="Review the PR.",
            marker_description="test",
            validate=lambda text: _validate_review_response(
                text, reviewer="OpenAI Codex", unresolved_items=(), architecture_status_mode="legacy"
            ),
        )

    assert response.marker_value.state == "approved"
    assert run_mock.call_count == 2
    assert "attempt_suffix" not in run_mock.call_args_list[1].kwargs


def test_codex_replacement_context_does_not_change_long_quota_exit_priority(tmp_path):
    result = _observed_codex_result(
        tmp_path,
        raw_output="HTTP 429: rate limit exceeded. Retry-After: 3600",
    )
    runner = FakeRunner()
    runner.wait_for_executable_stability = lambda command, *, deadline=None: False
    config = make_config(tmp_path, reviewer="codex", agent_max_retries=1)

    with patch_object_in((validated_agent, response_validation), "run_agent_result", return_value=result):
        with pytest.raises(QuotaResetExceededError) as error:
            _run_validated_agent(
                runner,
                agent="codex",
                config=config,
                prompt="Review the PR.",
                marker_description="test",
                validate=lambda text: _validate_review_response(
                    text, reviewer="OpenAI Codex", unresolved_items=(), architecture_status_mode="legacy"
                ),
            )

    assert QuotaResetExceededError.EXIT_CODE == 3
    assert "quota exhausted" in str(error.value).lower()
    assert "Codex executable replacement" in str(error.value)


def _observed_replacement_result(
    tmp_path, *, command, reason, raw_output="", returncode=1, model_used=None
):
    identity = ExecutableIdentity(
        command, f"{command}-target", (1, 1, 90_000_000_000), (1, 2, 90_000_000_000)
    )
    changed = ExecutableIdentity(
        command, f"{command}-new-target", (1, 3, 100_000_000_000), (1, 4, 100_000_000_000)
    )
    observation = ExecutionObservation(100, 100, 101, 1, identity, changed, False)
    return AgentResult(
        text="",
        raw_output=raw_output,
        returncode=returncode,
        model_used=model_used,
        command_result=CommandResult(
            [command], tmp_path, raw_output, "", returncode, observation
        ),
        self_update_reason=reason,
    )


def test_gemini_replacement_replay_uses_fresh_timeout_and_provider_suffix(tmp_path):
    valid = structured_pr_review(
        state="approved", summary="Gemini recovered.", reviewer="Google Gemini"
    )
    results = iter(
        [
            _observed_replacement_result(
                tmp_path,
                command="gemini",
                reason="Gemini executable changed during invocation",
            ),
            AgentResult(text=valid, raw_output=valid, returncode=0, model_used="gemini-model"),
        ]
    )
    runner = FakeRunner()
    stability_calls = []
    runner.wait_for_executable_stability = lambda command, *, deadline=None: (
        stability_calls.append((command, deadline)) or True
    )
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch_object_in((validated_agent, response_validation), "run_agent_result", side_effect=results) as run_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the PR.",
            marker_description="test",
            validate=lambda text: _validate_review_response(
                text, reviewer="Google Gemini", unresolved_items=(), architecture_status_mode="legacy"
            ),
            timeout_seconds=60,
        )

    assert response.marker_value.state == "approved"
    assert stability_calls == [(config.gemini_cmd, None)]
    assert run_mock.call_args_list[0].kwargs["timeout_seconds"] == 60
    assert run_mock.call_args_list[1].kwargs["timeout_seconds"] == 60
    assert run_mock.call_args_list[1].kwargs["attempt_suffix"] == "executable-replacement-attempt2"


def test_gemini_unstable_replacement_preserves_ordinary_retry(tmp_path):
    valid = structured_pr_review(
        state="approved", summary="Ordinary retry recovered.", reviewer="Google Gemini"
    )
    results = iter(
        [
            _observed_replacement_result(
                tmp_path,
                command="gemini",
                reason="Gemini executable changed during invocation",
                returncode=0,
            ),
            AgentResult(text=valid, raw_output=valid, returncode=0),
        ]
    )
    runner = FakeRunner()
    runner.wait_for_executable_stability = lambda command, *, deadline=None: False
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=1)

    with patch_object_in((validated_agent, response_validation), "run_agent_result", side_effect=results) as run_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the PR.",
            marker_description="test",
            validate=lambda text: _validate_review_response(
                text, reviewer="Google Gemini", unresolved_items=(), architecture_status_mode="legacy"
            ),
        )

    assert response.marker_value.state == "approved"
    assert run_mock.call_count == 2
    assert "attempt_suffix" not in run_mock.call_args_list[1].kwargs


def test_claude_quiet_unstable_replacement_keeps_ordinary_retry(tmp_path):
    valid = structured_pr_review(
        state="approved", summary="Ordinary retry recovered.", reviewer="Anthropic Claude"
    )
    results = iter(
        [
            _observed_replacement_result(
                tmp_path,
                command="claude",
                reason="Claude Code updater diagnostic",
            ),
            AgentResult(text=valid, raw_output=valid, returncode=0),
        ]
    )
    runner = FakeRunner()
    runner.wait_for_executable_stability = lambda command, *, deadline=None: False
    config = make_config(tmp_path, reviewer="claude", agent_max_retries=1)

    with patch_object_in((validated_agent, response_validation), "run_agent_result", side_effect=results) as run_mock:
        response = _run_validated_agent(
            runner,
            agent="claude",
            config=config,
            prompt="Review the PR.",
            marker_description="test",
            validate=lambda text: _validate_review_response(
                text, reviewer="Anthropic Claude", unresolved_items=(), architecture_status_mode="legacy"
            ),
        )

    assert response.marker_value.state == "approved"
    assert run_mock.call_count == 2
    assert run_mock.call_args_list[1].kwargs["attempt_suffix"] == "attempt2"


def test_antigravity_quiet_unstable_replacement_reaches_model_fallback(tmp_path):
    valid = structured_pr_review(
        state="approved", summary="Fallback recovered.", reviewer="Google Antigravity"
    )
    results = iter(
        [
            _observed_replacement_result(
                tmp_path,
                command="agy",
                reason="Antigravity executable changed during invocation",
                returncode=0,
                model_used="ModelA",
            ),
            AgentResult(text="", raw_output="quota exceeded", returncode=1, model_used="ModelA"),
            AgentResult(text=valid, raw_output=valid, returncode=0, model_used="ModelB"),
        ]
    )
    runner = FakeRunner()
    runner.wait_for_executable_stability = lambda command, *, deadline=None: False
    config = make_config(
        tmp_path,
        reviewer="antigravity",
        antigravity_models=("ModelA", "ModelB"),
        agent_max_retries=1,
    )
    models: list[tuple[str, ...]] = []

    def mock_run(_runner, *, config, **_kwargs):
        models.append(config.antigravity_models)
        return next(results)

    with patch_object_in((validated_agent, response_validation), "run_agent_result", side_effect=mock_run) as run_mock:
        response = _run_validated_agent(
            runner,
            agent="antigravity",
            config=config,
            prompt="Review the PR.",
            marker_description="test",
            validate=lambda text: _validate_review_response(
                text, reviewer="Google Antigravity", unresolved_items=(), architecture_status_mode="legacy"
            ),
        )

    assert response.marker_value.state == "approved"
    assert models == [("ModelA",), ("ModelA",), ("ModelB",)]
    assert "attempt_suffix" not in run_mock.call_args_list[1].kwargs
    assert "attempt_suffix" not in run_mock.call_args_list[2].kwargs


def test_antigravity_replacement_replay_does_not_advance_model_state(tmp_path):
    valid = structured_pr_review(
        state="approved", summary="Antigravity recovered.", reviewer="Google Antigravity"
    )
    results = iter(
        [
            _observed_replacement_result(
                tmp_path,
                command="agy",
                reason="Antigravity executable changed during invocation",
                model_used="ModelA",
            ),
            AgentResult(text="", raw_output="overloaded_error", returncode=1, model_used="ModelA"),
            AgentResult(text="", raw_output="quota exceeded", returncode=1, model_used="ModelA"),
            AgentResult(text=valid, raw_output=valid, returncode=0, model_used="ModelB"),
        ]
    )
    runner = FakeRunner()
    runner.wait_for_executable_stability = lambda command, *, deadline=None: True
    config = make_config(
        tmp_path,
        reviewer="antigravity",
        antigravity_models=("ModelA", "ModelB"),
        agent_max_retries=1,
    )
    models: list[tuple[str, ...]] = []

    def mock_run(_runner, *, config, **_kwargs):
        models.append(config.antigravity_models)
        return next(results)

    with patch_object_in((validated_agent, response_validation), "run_agent_result", side_effect=mock_run) as run_mock:
        response = _run_validated_agent(
            runner,
            agent="antigravity",
            config=config,
            prompt="Review the PR.",
            marker_description="test",
            validate=lambda text: _validate_review_response(
                text, reviewer="Google Antigravity", unresolved_items=(), architecture_status_mode="legacy"
            ),
            timeout_seconds=60,
        )

    assert response.marker_value.state == "approved"
    assert models == [("ModelA",), ("ModelA",), ("ModelA",), ("ModelB",)]
    assert run_mock.call_args_list[1].kwargs["attempt_suffix"] == "executable-replacement-attempt2"
    assert "attempt_suffix" not in run_mock.call_args_list[2].kwargs
    assert "attempt_suffix" not in run_mock.call_args_list[3].kwargs


@pytest.mark.parametrize(
    ("text", "returncode"),
    [("", 1), ("not a structured response", 0)],
)
def test_orchestrator_retries_capture_diagnostics_as_tooling_failure(
    tmp_path, text, returncode
):
    """Capture-read diagnostics remain retryable for failed or invalid responses."""
    from unittest.mock import patch

    config = make_config(tmp_path, coder="codex", reviewer="codex", agent_max_retries=1)
    command_result = CommandResult(
        ["codex"],
        tmp_path,
        "",
        "",
        returncode,
        capture_diagnostics=("capture_read_failed:OSError: injected",),
    )
    result = AgentResult(
        text=text,
        raw_output=text,
        returncode=returncode,
        command_result=command_result,
    )

    with patch_object_in((validated_agent, response_validation), "run_agent_result", return_value=result
    ) as run_mock:
        with pytest.raises(AgentInvocationError) as exc_info:
            _run_validated_agent(
                FakeRunner(),
                agent="codex",
                config=config,
                prompt="Review the PR.",
                marker_description="test",
                validate=lambda value: _validate_review_response(
                    value, reviewer="OpenAI Codex", unresolved_items=(), architecture_status_mode="legacy"
                ),
            )

    assert run_mock.call_count == 2
    assert exc_info.value.failure_category == (
        "deterministic" if returncode != 0 else "transient"
    )


def test_orchestrator_classifies_resource_exhaustion_with_diagnostics(tmp_path):
    config = make_config(tmp_path, coder="codex", reviewer="codex", agent_max_retries=0)
    result = AgentResult(
        text="",
        raw_output="",
        returncode=137,
        containment=ContainmentEvidence(
            backend="systemd-cgroup-v2",
            termination_cause="oom",
            applicable_limit="MemoryMax/MemorySwapMax",
            diagnostics=("systemd scope result=oom-kill",),
        ),
    )

    with patch_object_in((validated_agent, response_validation), "run_agent_result", return_value=result
    ):
        with pytest.raises(AgentInvocationError) as exc_info:
            _run_validated_agent(
                FakeRunner(),
                agent="codex",
                config=config,
                prompt="Review the PR.",
                marker_description="test",
                validate=lambda value: value,
            )

    assert exc_info.value.failure_category == "resource-exhausted"
    assert "MemoryMax/MemorySwapMax" in str(exc_info.value)
    assert "systemd-cgroup-v2" in str(exc_info.value)


def test_orchestrator_target_exec_error_without_command_result_is_retryable(tmp_path):
    class TargetExecRunner(FakeRunner):
        def __init__(self):
            super().__init__()
            self.decisions = []

        def target_exec_retry_decision(self, command, errno_value):
            self.decisions.append((command, errno_value))
            return True, "target executable retryable"

    config = make_config(tmp_path, coder="codex", reviewer="codex", agent_max_retries=0)
    result = AgentResult(
        text="",
        raw_output="",
        returncode=127,
        containment=ContainmentEvidence(
            backend="systemd-cgroup-v2",
            termination_cause="target-exec-error",
            target_exec_errno=2,
            cleanup_confirmed=True,
            diagnostics=("target exec failed (2): missing",),
        ),
    )
    runner = TargetExecRunner()

    with patch_object_in((validated_agent, response_validation), "run_agent_result", return_value=result
    ):
        with pytest.raises(AgentInvocationError) as exc_info:
            _run_validated_agent(
                runner,
                agent="codex",
                config=config,
                prompt="Review the PR.",
                marker_description="test",
                validate=lambda value: value,
            )

    # The typed runner decision outranks textual classification (#1226).
    assert exc_info.value.failure_category == "transient"
    assert runner.decisions
    assert all(command == "" and errno_value == 2 for command, errno_value in runner.decisions)
    assert "target executable retryable" in str(exc_info.value)


def _contained_target_exec_result(tmp_path, command, errno_value=errno.ENOENT):
    return AgentResult(
        text="",
        raw_output="",
        returncode=127,
        command_result=CommandResult([command, "exec"], tmp_path, "", "", 127),
        containment=ContainmentEvidence(
            backend="systemd-cgroup-v2",
            termination_cause="target-exec-error",
            target_exec_errno=errno_value,
            cleanup_confirmed=True,
            diagnostics=(f"target exec failed ({errno_value}): [Errno {errno_value}] missing",),
        ),
    )


def _preflighted_target_exec_runner(monkeypatch, command, stable=True):
    import coding_review_agent_loop.runner as runner_module

    runner = FakeRunner()
    runner.remember_agent_command(command, command, "--claude-cmd")
    # The updater has already re-created the binary when the decision runs.
    monkeypatch.setattr(runner_module.shutil, "which", lambda _name: "/opt/bin/claude")
    monkeypatch.setattr(runner_module.os.path, "exists", lambda _path: True)
    stability_calls = []
    runner.wait_for_executable_stability = lambda cmd, *, deadline=None: (
        stability_calls.append(cmd) or stable
    )
    return runner, stability_calls


def _codex_review_validator(text):
    return _validate_review_response(
        text, reviewer="OpenAI Codex", unresolved_items=(), architecture_status_mode="legacy"
    )


@pytest.mark.parametrize("command", ["claude", "/home/user/.npm-global/bin/claude"])
def test_validated_agent_recovers_after_contained_enoent_during_self_update(
    monkeypatch, tmp_path, capsys, command
):
    """#1225/#1226: a mid-run self-update ENOENT is transient and relaunched once."""
    runner, stability_calls = _preflighted_target_exec_runner(monkeypatch, command)
    valid = structured_pr_review(state="approved", summary="Recovered.", reviewer="OpenAI Codex")
    results = iter(
        [
            _contained_target_exec_result(tmp_path, command),
            AgentResult(text=valid, raw_output=valid, returncode=0),
        ]
    )
    config = make_config(
        tmp_path, reviewer="codex", agent_max_retries=2, quiet=False
    )

    with patch_object_in((validated_agent, response_validation), "run_agent_result", side_effect=results) as run_mock:
        response = _run_validated_agent(
            runner,
            agent="codex",
            config=config,
            prompt="Review the PR.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=_codex_review_validator,
        )

    assert response.marker_value.state == "approved"
    assert run_mock.call_count == 2
    assert stability_calls == [command]
    stderr = capsys.readouterr().err
    assert "transient failure" in stderr
    assert "deterministic failure" not in stderr
    assert "temporarily unavailable/being replaced after successful preflight" in stderr


def test_validated_agent_exhausted_contained_enoent_stays_transient(
    monkeypatch, tmp_path, capsys
):
    command = "/home/user/.npm-global/bin/claude"
    runner, stability_calls = _preflighted_target_exec_runner(monkeypatch, command)
    config = make_config(
        tmp_path, reviewer="codex", agent_max_retries=2, quiet=False
    )

    with patch_object_in((validated_agent, response_validation), "run_agent_result",
        side_effect=lambda *a, **k: _contained_target_exec_result(tmp_path, command),
    ) as run_mock:
        with pytest.raises(AgentInvocationError) as exc_info:
            _run_validated_agent(
                runner,
                agent="codex",
                config=config,
                prompt="Review the PR.",
                marker_description="<!-- AGENT_STATE: approved|blocking -->",
                validate=_codex_review_validator,
            )

    assert exc_info.value.failure_category == "transient"
    assert "temporarily unavailable/being replaced after successful preflight" in str(
        exc_info.value
    )
    assert run_mock.call_count == config.agent_max_retries + 1
    # One bounded stability wait per actual relaunch, none after exhaustion.
    assert stability_calls == [command] * config.agent_max_retries
    stderr = capsys.readouterr().err
    assert "deterministic failure" not in stderr


@pytest.mark.parametrize("stable", [True, False])
def test_validated_agent_target_exec_relaunch_ignores_stability_result(
    monkeypatch, tmp_path, stable
):
    command = "claude"
    runner, stability_calls = _preflighted_target_exec_runner(
        monkeypatch, command, stable=stable
    )
    valid = structured_pr_review(state="approved", summary="Recovered.", reviewer="OpenAI Codex")
    results = iter(
        [
            _contained_target_exec_result(tmp_path, command, errno.ETXTBSY),
            AgentResult(text=valid, raw_output=valid, returncode=0),
        ]
    )
    config = make_config(tmp_path, reviewer="codex", agent_max_retries=1)

    with patch_object_in((validated_agent, response_validation), "run_agent_result", side_effect=results) as run_mock:
        response = _run_validated_agent(
            runner,
            agent="codex",
            config=config,
            prompt="Review the PR.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=_codex_review_validator,
        )

    assert response.marker_value.state == "approved"
    assert run_mock.call_count == 2
    assert stability_calls == [command]


def test_validated_agent_target_exec_without_retry_budget_skips_stability_wait(
    monkeypatch, tmp_path
):
    command = "claude"
    runner, stability_calls = _preflighted_target_exec_runner(monkeypatch, command)
    config = make_config(tmp_path, reviewer="codex", agent_max_retries=0)

    with patch_object_in((validated_agent, response_validation), "run_agent_result",
        side_effect=lambda *a, **k: _contained_target_exec_result(tmp_path, command),
    ) as run_mock:
        with pytest.raises(AgentInvocationError) as exc_info:
            _run_validated_agent(
                runner,
                agent="codex",
                config=config,
                prompt="Review the PR.",
                marker_description="<!-- AGENT_STATE: approved|blocking -->",
                validate=_codex_review_validator,
            )

    assert exc_info.value.failure_category == "transient"
    assert run_mock.call_count == 1
    assert stability_calls == []


@pytest.mark.parametrize(
    ("containment_kwargs", "expected_category"),
    [
        (
            {"termination_cause": "oom", "applicable_limit": "MemoryMax", "cleanup_confirmed": True},
            "resource-exhausted",
        ),
        (
            {"termination_cause": "target-exec-error", "cleanup_confirmed": False},
            "containment-indeterminate",
        ),
    ],
)
def test_validated_agent_target_exec_retry_keeps_safety_precedence(
    monkeypatch, tmp_path, containment_kwargs, expected_category
):
    command = "claude"
    runner, stability_calls = _preflighted_target_exec_runner(monkeypatch, command)
    result = AgentResult(
        text="",
        raw_output="",
        returncode=127,
        command_result=CommandResult([command], tmp_path, "", "", 127),
        containment=ContainmentEvidence(
            backend="systemd-cgroup-v2",
            target_exec_errno=errno.ENOENT,
            diagnostics=("target exec failed (2): missing",),
            **containment_kwargs,
        ),
    )
    config = make_config(tmp_path, reviewer="codex", agent_max_retries=2)

    with patch_object_in((validated_agent, response_validation), "run_agent_result", return_value=result) as run_mock:
        with pytest.raises(AgentInvocationError) as exc_info:
            _run_validated_agent(
                runner,
                agent="codex",
                config=config,
                prompt="Review the PR.",
                marker_description="<!-- AGENT_STATE: approved|blocking -->",
                validate=_codex_review_validator,
            )

    assert exc_info.value.failure_category == expected_category
    assert run_mock.call_count == 1
    assert stability_calls == []


def test_antigravity_reviewer_capacity_retry_falls_back_with_repair_enabled(tmp_path):
    """Reviewer repair paths share Antigravity's retry budget across models."""
    from coding_review_agent_loop.agents.base import AgentResult
    from unittest.mock import patch

    config = make_config(
        tmp_path,
        reviewer="antigravity",
        agent_max_retries=1,
        antigravity_models=("ModelA", "ModelB"),
    )
    results = iter((
        AgentResult(
            text="",
            raw_output="Error: Our servers are experiencing high traffic right now, please try again in a minute.",
            returncode=1,
            model_used="ModelA",
        ),
        AgentResult(text="", raw_output="quota exceeded please try again", returncode=1, model_used="ModelA"),
        AgentResult(text="valid review", returncode=0, model_used="ModelB"),
    ))
    models: list[tuple[str, ...]] = []

    def mock_run(_runner, *, config, **_kwargs):
        models.append(config.antigravity_models)
        return next(results)

    with patch_object_in((validated_agent, response_validation), "run_agent_result", mock_run):
        response = _run_validated_agent(
            FakeRunner(), agent="antigravity", config=config, prompt="Review.",
            marker_description="test", validate=lambda text: text, use_repair=True, role="reviewer",
        )

    assert response.text == "valid review"
    assert response.model_used == "ModelB"
    assert models == [("ModelA",), ("ModelA",), ("ModelB",)]


def test_antigravity_capacity_exhaustion_is_reported_transient(tmp_path):
    """A real orchestration retry path keeps capacity exhaustion transient."""
    from coding_review_agent_loop.agents.base import AgentResult
    from coding_review_agent_loop.errors import AgentInvocationError
    from unittest.mock import patch

    config = make_config(tmp_path, reviewer="antigravity", agent_max_retries=1, antigravity_models=("ModelA",))
    result = AgentResult(
        text="",
        raw_output="Error: Our servers are experiencing high traffic right now, please try again in a minute.",
        returncode=1,
        model_used="ModelA",
    )

    with patch_object_in((validated_agent, response_validation), "run_agent_result", return_value=result):
        with pytest.raises(AgentInvocationError, match="transient"):
            _run_validated_agent(
                FakeRunner(), agent="antigravity", config=config, prompt="Review.",
                marker_description="test", validate=lambda text: text, use_repair=True, role="reviewer",
            )


def test_reviewer_and_coder_call_sites_pass_correct_role(tmp_path):
    """_run_validated_agent propagates role= to run_agent_result correctly."""
    from unittest.mock import patch
    from coding_review_agent_loop.agents.base import AgentResult

    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)
    captured_roles: list = []

    def mock_run(
        runner,
        *,
        agent,
        config,
        prompt,
        session_id=None,
        run_id=None,
        role=None,
        label=None,
        timeout_seconds=None,
    ):
        captured_roles.append(role)
        return AgentResult(text="ok")

    with patch_object_in((validated_agent, response_validation), "run_agent_result", mock_run):
        _run_validated_agent(
            FakeRunner(),
            agent="gemini",
            config=config,
            prompt="Review the plan.",
            marker_description="test",
            validate=lambda text: text,
            role="reviewer",
        )

    assert captured_roles == ["reviewer"]
    captured_roles.clear()

    with patch_object_in((validated_agent, response_validation), "run_agent_result", mock_run):
        _run_validated_agent(
            FakeRunner(),
            agent="gemini",
            config=config,
            prompt="Implement.",
            marker_description="test",
            validate=lambda text: text,
        )

    assert captured_roles == [None]


@pytest.mark.parametrize("artifact", ["", "not a structured review", "wrong reviewer response"])
def test_failed_exit_rejects_empty_malformed_or_wrong_role_response_artifact(tmp_path, artifact):
    runner = FakeRunner(
        gemini_outputs=[("command failed", 1)], public_response_outputs=[artifact]
    )
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    def validate(text):
        if text == "valid review":
            return text
        raise AgentLoopError("review does not match the expected reviewer role")

    with pytest.raises(AgentInvocationError):
        _run_validated_agent(
            runner, agent="gemini", config=config, prompt="Review the PR.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->", validate=validate,
        )
    assert len([cmd for cmd, _cwd in runner.commands if cmd[:1] == ["gemini"]]) == 1


def test_host_footer_log_latch_resets_with_each_owned_usage_context(tmp_path, monkeypatch):
    import coding_review_agent_loop.github as github_module

    lines = []
    monkeypatch.setattr(github_module, "log", lambda _config, message: lines.append(message))
    config = make_config(tmp_path)
    # An earlier invocation in this process left the latch set.
    github_module.reset_host_footer_log_latch()
    github_module.note_host_footer_observed(config, "earlier invocation")
    assert len(lines) == 1

    # Two successive owning invocations each log once, however often the
    # footer is observed on write and re-read paths.
    for expected in (2, 3):
        validated_agent._new_usage_context(config)
        for context in ("write", "re-read", "nested PR run"):
            github_module.note_host_footer_observed(config, context)
        assert len(lines) == expected

    # With no footer observed, an owning invocation logs nothing.
    validated_agent._new_usage_context(config)
    assert len(lines) == 3


def _codex_stream(*, message=None, error=None, stderr=None, tool=True):
    lines = []
    if tool:
        lines.append(json.dumps({"type": "item.completed", "item": {
            "type": "command_execution", "aggregated_output": "auth.captureOwner() billing"}}))
    if message is not None:
        lines.append(json.dumps({"type": "error", "message": message}))
        lines.append(json.dumps({"type": "turn.failed", "error": {"message": message}}))
    if error is not None:
        lines.append(json.dumps({"type": "turn.failed", "error": error}))
    if stderr:
        lines.append(stderr)
    return "\n".join(lines)


def _drive_codex(tmp_path, raws, *, retries=2, containment=None):
    """Run _run_validated_agent with a Codex fake; return (calls, sleeps, response|exc)."""
    from coding_review_agent_loop.errors import AgentInvocationError
    from unittest.mock import patch

    config = make_config(tmp_path, reviewer="codex", agent_max_retries=retries)
    outcomes = list(raws)
    calls = {"n": 0}

    def mock_run(_runner, **_kwargs):
        raw = outcomes[min(calls["n"], len(outcomes) - 1)]
        calls["n"] += 1
        if raw is None:
            return AgentResult(text="valid", returncode=0, provider="codex")
        return AgentResult(text="", raw_output=raw, returncode=1, provider="codex", containment=containment)

    runner = FakeRunner()
    with patch_object_in((validated_agent, response_validation), "run_agent_result", mock_run):
        try:
            out = _run_validated_agent(
                runner, agent="codex", config=config, prompt="Review.",
                marker_description="test", validate=lambda text: text, role="reviewer",
            )
        except AgentInvocationError as exc:
            out = exc
    sleeps = [c for c, _cwd in runner.commands if c and c[0] == "sleep"]
    return calls["n"], sleeps, out


def test_codex_capacity_with_auth_tool_output_is_retried_then_succeeds(tmp_path):
    n, sleeps, out = _drive_codex(tmp_path, [_codex_stream(message=_CAPACITY), None])
    assert n == 2 and out.text == "valid"
    assert len(sleeps) == 1


def test_codex_capacity_retry_exhausted_reports_transient(tmp_path):
    n, sleeps, out = _drive_codex(tmp_path, [_codex_stream(message=_CAPACITY)], retries=2)
    assert n == 3 and len(sleeps) == 2
    text = str(out)
    assert "transient" in text
    assert "billing" not in text.lower() and "authenticated" not in text


@pytest.mark.parametrize(
    "raw,expected_calls,needle",
    [
        (_codex_stream(message="401 Unauthorized"), 1, "authenticated"),
        (_codex_stream(message="invalid api key"), 1, "authenticated"),
        (_codex_stream(message="401 Unauthorized", stderr="warning: model at capacity, retrying"), 1, "authenticated"),
        (_codex_stream(message="429 insufficient_quota: You exceeded your current quota, please check your plan."), 1, "billing"),
        (_codex_stream(error={"message": "request failed", "status": 401}), 1, "authenticated"),
        (_codex_stream(error={"status": 429, "code": "insufficient_quota"}), 1, "billing"),
    ],
)
def test_codex_non_retryable_structured_errors_are_not_retried(tmp_path, raw, expected_calls, needle):
    n, sleeps, out = _drive_codex(tmp_path, [raw])
    assert n == expected_calls and sleeps == []
    assert "non-retryable" in str(out) and needle in str(out)


@pytest.mark.parametrize(
    "raw",
    [
        _codex_stream(message="429 Too Many Requests: rate limit reached", stderr="billing notice"),
        _codex_stream(error={"message": "request failed", "status": 503}),
        _codex_stream(error={"message": "request failed", "status": "503"}),
        _codex_stream(message="429 RESOURCE_EXHAUSTED"),
    ],
)
def test_codex_transient_structured_errors_are_retried(tmp_path, raw):
    n, sleeps, out = _drive_codex(tmp_path, [raw, None])
    assert n == 2 and out.text == "valid" and len(sleeps) == 1


def test_codex_provider_resource_exhausted_is_transient_not_host(tmp_path):
    n, _sleeps, out = _drive_codex(tmp_path, [_codex_stream(message="429 RESOURCE_EXHAUSTED")], retries=0)
    assert "transient" in str(out) and "resource-exhausted" not in str(out)


@pytest.mark.parametrize(
    "raw",
    [
        '{"type":"item.completed","item":{"aggregated_output":"auth.captureOwner() invalid api key timeout',
        _codex_stream(tool=True),
    ],
)
def test_codex_streams_without_error_evidence_are_deterministic(tmp_path, raw):
    n, sleeps, out = _drive_codex(tmp_path, [raw])
    assert n == 1 and sleeps == []
    assert "deterministic" in str(out)


def test_typed_host_exhaustion_beats_provider_capacity(tmp_path):
    evidence = ContainmentEvidence(
        backend="systemd-cgroup-v2", termination_cause="oom", cleanup_confirmed=True,
        applicable_limit="MemoryMax",
    )
    n, sleeps, out = _drive_codex(tmp_path, [_codex_stream(message=_CAPACITY)], containment=evidence)
    assert n == 1 and sleeps == []
    assert "resource-exhausted" in str(out)


def test_codex_valid_event_then_truncated_item_is_deterministic_without_retry(tmp_path):
    raw = "\n".join([
        json.dumps({"type": "thread.started", "thread_id": "t"}),
        '{"type":"item.completed","item":{"aggregated_output":"auth.captureOwner() invalid api key timeout',
    ])
    n, sleeps, out = _drive_codex(tmp_path, [raw])
    assert n == 1 and sleeps == []
    text = str(out)
    assert "deterministic" in text and "transient" not in text and "authenticated" not in text


def test_codex_stderr_decisive_evidence_in_omitted_middle_is_not_retried(tmp_path):
    stderr = "timeout " + "n" * 7000 + " 401 Unauthorized " + "n" * 7000 + " timeout"
    raw = json.dumps({"type": "thread.started", "thread_id": "t"}) + "\n" + stderr
    n, sleeps, out = _drive_codex(tmp_path, [raw, None])
    assert n == 1 and sleeps == []
    assert "non-retryable" in str(out)


def test_codex_oversized_structured_billing_error_gets_billing_suggestion(tmp_path):
    msg = "n" * 7000 + " insufficient_quota " + "n" * 7000
    n, sleeps, out = _drive_codex(tmp_path, [_codex_stream(message=msg)])
    assert n == 1 and sleeps == []
    text = str(out)
    assert "non-retryable" in text and "billing" in text and "authenticated" not in text


# --- Codex plan usage limits (#1281) -----------------------------------------

_USAGE_HEAD = (
    "You've hit your usage limit. Upgrade to Pro (https://openai.com/chatgpt/pricing), "
    "visit https://chatgpt.com/codex/settings/usage to purchase more credits"
)


def _short_clock(minutes_ahead=2):
    import datetime
    import zoneinfo

    target = datetime.datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles")) + datetime.timedelta(
        minutes=minutes_ahead
    )
    return target.strftime("%I:%M %p").lstrip("0")


@pytest.fixture
def _la_tz(monkeypatch):
    monkeypatch.setenv("TZ", "America/Los_Angeles")


def _spy_reset_parse(monkeypatch):
    parsed: list[str] = []
    real = validated_agent._parse_rate_limit_reset_seconds

    def spy(text, **kwargs):
        parsed.append(text)
        return real(text, **kwargs)

    monkeypatch.setattr(validated_agent, "_parse_rate_limit_reset_seconds", spy)
    return parsed


@pytest.mark.parametrize("shape", ["message", "turn_failed"])
def test_codex_usage_limit_long_reset_raises_quota_reset_once(tmp_path, _la_tz, shape):
    message = f"{_USAGE_HEAD} or try again at Oct 23rd, 2099 3:05 PM."
    raw = (
        _codex_stream(message=message)
        if shape == "message"
        else _codex_stream(error={"message": message})
    )
    calls = {"n": 0}

    def mock_run(_runner, **_kwargs):
        calls["n"] += 1
        return AgentResult(text="", raw_output=raw, returncode=1, provider="codex")

    config = make_config(tmp_path, reviewer="codex", agent_max_retries=2)
    runner = FakeRunner()
    with patch_object_in((validated_agent, response_validation), "run_agent_result", mock_run):
        with pytest.raises(QuotaResetExceededError) as excinfo:
            _run_validated_agent(
                runner, agent="codex", config=config, prompt="Review.",
                marker_description="test", validate=lambda text: text, role="reviewer",
            )
    assert calls["n"] == 1
    assert [c for c, _cwd in runner.commands if c and c[0] == "sleep"] == []
    text = str(excinfo.value)
    assert "Codex quota exhausted. Reset in" in text
    assert "deterministic" not in text and "billing / credit balance" not in text


@pytest.mark.parametrize(
    "raw",
    [
        _codex_stream(message=f"{_USAGE_HEAD}."),
        _codex_stream(error={"code": "usage_limit_reached", "message": "Usage limit reached"}),
    ],
)
def test_codex_usage_limit_without_reset_is_bounded_transient_with_quota_suggestion(tmp_path, raw):
    n, sleeps, out = _drive_codex(tmp_path, [raw], retries=2)
    assert n == 3 and len(sleeps) == 2
    assert isinstance(out, AgentInvocationError)
    assert out.failure_category == "transient"
    text = str(out)
    assert "deterministic" not in text and "non-retryable" not in text
    assert "wait for quota reset" in text
    assert "billing / credit balance" not in text


def test_codex_usage_limit_short_reset_follows_ordinary_retry(tmp_path, _la_tz):
    raw = _codex_stream(message=f"{_USAGE_HEAD} or try again at {_short_clock()}.")
    n, sleeps, out = _drive_codex(tmp_path, [raw, None], retries=2)
    assert n == 2 and len(sleeps) == 1 and out.text == "valid"


def test_codex_earlier_long_reset_does_not_drive_quota_stop_without_terminal_reset(
    tmp_path, monkeypatch
):
    raw = "\n".join([
        json.dumps({"type": "error", "message": "rate limit reached; try again in 2 days"}),
        json.dumps({"type": "turn.failed", "error": {"message": f"{_USAGE_HEAD}."}}),
    ])
    parsed = _spy_reset_parse(monkeypatch)
    n, sleeps, out = _drive_codex(tmp_path, [raw], retries=2)
    assert n == 3 and len(sleeps) == 2
    assert isinstance(out, AgentInvocationError)
    assert out.failure_category == "transient"
    assert parsed and all("2 days" not in text for text in parsed)
    # The verdict text still carries the supporting error for diagnostics.
    from coding_review_agent_loop import transient

    assert "try again in 2 days" in transient.classify_codex_failure(raw).text


def test_codex_earlier_long_reset_yields_to_terminal_short_reset(tmp_path, _la_tz, monkeypatch):
    raw = "\n".join([
        json.dumps({"type": "error", "message": "rate limit reached; try again in 2 days"}),
        json.dumps({"type": "turn.failed", "error": {
            "message": f"{_USAGE_HEAD} or try again at {_short_clock()}."}}),
    ])
    parsed = _spy_reset_parse(monkeypatch)
    n, sleeps, out = _drive_codex(tmp_path, [raw, None], retries=2)
    assert n == 2 and len(sleeps) == 1 and out.text == "valid"
    assert parsed and all("2 days" not in text for text in parsed)


@pytest.mark.parametrize(
    "error",
    [
        {"message": f"{_USAGE_HEAD}.", "status": 401},
        {"message": f"unauthorized: {_USAGE_HEAD}."},
    ],
)
def test_codex_usage_limit_with_auth_is_not_retried(tmp_path, error):
    n, sleeps, out = _drive_codex(tmp_path, [_codex_stream(error=error)], retries=2)
    assert n == 1 and sleeps == []
    assert out.failure_category == "non-retryable"


def test_guard_correction_freezes_its_own_acquisition_catalog(tmp_path):
    from coding_review_agent_loop.protocol import validate_structured_coder_followup
    rejected = structured_coder_followup(tests_run=['pytest tests/test_api.py /outside/test_api.py'])
    accepted = structured_coder_followup(tests_run=['pytest tests/test_api.py'])
    first, second = object(), object()
    results = [
        AgentResult(text=rejected, returncode=0, session_id='coder-session', test_turn_id='rejected', test_turn_observations=(first,)),
        AgentResult(text=accepted, returncode=0, session_id='coder-session', test_turn_id='correction', test_turn_observations=(second,)),
    ]
    runner = FakeRunner()
    config = make_config(tmp_path, agent_max_retries=0)
    seen = []
    def validate(text):
        seen.append(response_validation._current_test_turn_observations(runner))
        return response_validation._validate_followup_workdir_report(
            validate_structured_coder_followup(text), config=config,
        )
    with patch_object_in((validated_agent, response_validation), 'run_agent_result', side_effect=results) as invoke:
        response = _run_validated_agent(
            runner, agent='claude', config=config, prompt='Correct PR feedback.',
            marker_description='structured coder followup', validate=validate,
            role='coder', use_repair=True, repair_expected_kind='coder_followup',
            reask_on_evidence_rejection=True,
        )
    assert invoke.call_count == 2
    assert len(seen) >= 2
    assert all(catalog == (first,) for catalog in seen[:-1])
    assert seen[-1] == (second,)
    assert response.acquisition_test_turn_id == 'correction'
    assert response.acquisition_test_observations == (second,)
    assert invoke.call_args_list[1].kwargs['session_id'] == 'coder-session'


def test_followup_workdir_composition_preserves_operational_errors(tmp_path, monkeypatch):
    from coding_review_agent_loop.protocol import validate_structured_coder_followup
    parsed = validate_structured_coder_followup(structured_coder_followup())
    failure = AgentLoopError('Operational context unavailable')
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(response_validation, '_degrade_out_of_checkout_tests', fail)
    with pytest.raises(AgentLoopError) as caught:
        response_validation._validate_followup_workdir_report(parsed, config=make_config(tmp_path))
    assert caught.value is failure
