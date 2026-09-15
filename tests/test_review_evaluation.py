import json

import pytest

from agent_loop_helpers import make_config
from coding_review_agent_loop.cli import build_parser
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.review_evaluation import (
    evaluate_frozen_artifacts,
    load_frozen_artifacts,
    render_evaluation_report,
)
from coding_review_agent_loop.review_scheduling import (
    ReviewSchedulingContract,
    ReviewObligation,
    SchedulerSnapshot,
    TransitionClassification,
    policy_capabilities,
    select_reviewers,
)


def _primary_contract():
    return ReviewSchedulingContract(
        required_reviewers=("OpenAI Codex", "Anthropic Claude", "Google Gemini"),
        policy="primary-then-panel",
        primary_reviewer="OpenAI Codex",
        broad_rules=(".github/**",),
    )


def test_primary_policy_requires_member_and_secondary():
    with pytest.raises(AgentLoopError, match="at least one secondary"):
        ReviewSchedulingContract(
            required_reviewers=("OpenAI Codex",),
            policy="primary-then-panel",
            primary_reviewer="OpenAI Codex",
        )
    with pytest.raises(AgentLoopError, match="member"):
        ReviewSchedulingContract(
            required_reviewers=("OpenAI Codex", "Anthropic Claude"),
            policy="primary-then-panel",
            primary_reviewer="Google Gemini",
        )


def test_primary_policy_selects_primary_then_independent_panel_then_owners():
    contract = _primary_contract()
    snapshot = SchedulerSnapshot(
        previous_sha=None,
        current_sha="head",
        contract=contract,
        obligations=(),
    )
    primary = select_reviewers(snapshot, TransitionClassification("broad", "initial"))
    assert primary.selected_reviewers == ("OpenAI Codex",)
    assert primary.phase == "primary"
    assert primary.calls_avoided == 2

    panel = select_reviewers(
        snapshot,
        TransitionClassification("broad", "same head"),
        qualifying_approvals=("OpenAI Codex",),
        final_sweep=True,
    )
    assert panel.selected_reviewers == ("Anthropic Claude", "Google Gemini")
    assert panel.phase == "final-secondary-sweep"

    primary_after_change = select_reviewers(
        SchedulerSnapshot(
            previous_sha="old", current_sha="new", contract=contract,
            obligations=(),
        ),
        TransitionClassification("narrow", "scoped fix"),
    )
    assert primary_after_change.selected_reviewers == ("OpenAI Codex",)
    assert primary_after_change.phase == "primary"

    remediation = select_reviewers(
        SchedulerSnapshot(
            previous_sha="old", current_sha="new", contract=contract,
            obligations=(
                ReviewObligation(
                    item_id="finding-1", status="blocking", scope=("src/a.py",),
                    resolution_owners=("Anthropic Claude",),
                    pending_owners=("Anthropic Claude",),
                ),
            ),
        ),
        TransitionClassification("narrow", "scoped fix"),
    )
    assert remediation.selected_reviewers == ("OpenAI Codex", "Anthropic Claude")
    assert remediation.phase == "remediation"


def test_policy_capabilities_are_explicit_and_existing_defaults_unchanged(tmp_path):
    assert not policy_capabilities("all-reviewers").scheduler_enabled
    assert policy_capabilities("selective-intermediate").owner_scoped_reconciliation
    assert policy_capabilities("primary-then-panel").requires_primary

    config = make_config(
        tmp_path,
        reviewer=("codex", "claude"),
        pr_review_policy="primary-then-panel",
        primary_reviewer="codex",
    )
    assert config.primary_reviewer == "codex"


def test_primary_cli_options_are_parsed_together():
    args = build_parser().parse_args(
        [
            "pr", "77", "--pr-review-policy", "primary-then-panel",
            "--primary-reviewer", "codex", "--reviewer", "codex", "--reviewer", "claude",
        ]
    )
    assert args.pr_review_policy == "primary-then-panel"
    assert args.primary_reviewer == "codex"


def test_offline_evaluation_is_deterministic_and_marks_missing_measurements():
    artifacts = {
        "schema_version": 1,
        "runs": [
            {
                "run_id": "r1",
                "policy": "primary-then-panel",
                "primary_reviewer": "primary",
                "primary_approval_round": 1,
                "panel_approval_round": 3,
                "findings": [
                    {"id": "primary-finding", "severity": "high", "contributors": ["primary"]},
                    {"id": "panel-finding", "severity": "critical", "contributors": ["secondary"]},
                ],
                "metrics": {"reviewer_calls": 3},
            }
        ],
    }
    first = evaluate_frozen_artifacts(artifacts)
    second = evaluate_frozen_artifacts(json.loads(json.dumps(artifacts)))
    assert first == second
    row = first["policies"]["primary-then-panel"]
    assert row["marginal_findings_beyond_primary"] == {"secondary": ["panel-finding"]}
    assert row["severity_weighted_marginal_findings"] == {"secondary": 5}
    assert row["metrics"]["reviewer_calls"]["value"] == 3
    assert row["metrics"]["tokens"]["status"] == "unavailable"
    assert "primary-then-panel" in render_evaluation_report(first, human=True)


def test_offline_evaluation_loads_local_json_and_rejects_live_schema(tmp_path):
    path = tmp_path / "runs.json"
    path.write_text(json.dumps({"schema_version": 1, "runs": []}), encoding="utf-8")
    assert load_frozen_artifacts(path)["runs"] == []
    path.write_text(json.dumps({"schema_version": 2, "runs": []}), encoding="utf-8")
    with pytest.raises(AgentLoopError, match="schema"):
        load_frozen_artifacts(path)
