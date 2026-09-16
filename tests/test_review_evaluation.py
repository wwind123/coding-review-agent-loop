import json
from pathlib import Path

import pytest

from coding_review_agent_loop.cli import main as cli_main
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.review_evaluation import (
    POLICIES,
    evaluate_frozen_artifacts,
    load_frozen_artifacts,
    render_evaluation_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ARTIFACTS = REPO_ROOT / "docs" / "evaluation" / "frozen_review_artifacts.json"
FIXTURE_REPORT = REPO_ROOT / "docs" / "evaluation" / "frozen_review_report.json"
VERIFIED = {"source": "frozen-run-log", "verified": True}


def _run(**overrides):
    run = {
        "run_id": "r1",
        "policy": "primary-then-panel",
        "primary_reviewer": "primary",
        "primary_approval_round": 1,
        "panel_approval_round": 3,
        "provenance": VERIFIED,
        "label_provenance": {"source": "maintainer-triage", "verified": True},
        "findings": [
            {"id": "primary-finding", "severity": "high", "valid": True, "contributors": ["primary"]},
            {"id": "panel-finding", "severity": "critical", "valid": True, "contributors": ["secondary"]},
            {"id": "withdrawn", "severity": "high", "valid": False, "contributors": ["secondary"]},
        ],
        "metrics": {"reviewer_calls": 3},
    }
    run.update(overrides)
    return run


def test_offline_evaluation_is_deterministic_and_marks_missing_measurements():
    artifacts = {"schema_version": 1, "runs": [_run()]}
    first = evaluate_frozen_artifacts(artifacts)
    second = evaluate_frozen_artifacts(json.loads(json.dumps(artifacts)))
    assert first == second
    row = first["policies"]["primary-then-panel"]
    assert row["valid_unique_findings"] == {"value": 2, "status": "verified"}
    assert row["marginal_findings_beyond_primary"] == {
        "status": "verified", "value": {"secondary": ["r1:panel-finding"]},
    }
    assert row["severity_weighted_marginal_findings"] == {"status": "verified", "value": {"secondary": 5}}
    assert row["primary_to_panel_approval_regressions"] == {"value": [2], "status": "verified"}
    assert row["metrics"]["reviewer_calls"] == {"value": 3, "status": "verified"}
    assert row["metrics"]["tokens"]["status"] == "unavailable"
    assert "tokens: unavailable" in render_evaluation_report(first, human=True)
    assert "primary-then-panel" in render_evaluation_report(first, human=True)
    for policy in POLICIES:
        assert policy in first["policies"]


def test_unlabeled_findings_and_unverified_label_provenance_are_unavailable_not_verified():
    unlabeled = evaluate_frozen_artifacts(
        {
            "schema_version": 1,
            "runs": [
                _run(findings=[{"id": "f", "severity": "high", "contributors": ["secondary"]}])
            ],
        }
    )["policies"]["primary-then-panel"]
    assert unlabeled["valid_unique_findings"]["status"] == "unavailable"
    assert unlabeled["valid_unique_findings"]["unlabeled_findings"] == 1
    assert unlabeled["marginal_findings_beyond_primary"]["status"] == "unavailable"
    assert unlabeled["severity_weighted_marginal_findings"]["status"] == "unavailable"

    unverified = evaluate_frozen_artifacts(
        {"schema_version": 1, "runs": [_run(label_provenance=None)]}
    )["policies"]["primary-then-panel"]
    assert unverified["valid_unique_findings"]["status"] == "unavailable"
    assert "provenance" in unverified["valid_unique_findings"]["reason"]

    false_provenance = evaluate_frozen_artifacts(
        {"schema_version": 1, "runs": [_run(label_provenance={"source": "guess", "verified": False})]}
    )["policies"]["primary-then-panel"]
    assert false_provenance["marginal_findings_beyond_primary"]["status"] == "unavailable"


def test_metrics_without_verified_provenance_are_unavailable():
    no_provenance = evaluate_frozen_artifacts(
        {"schema_version": 1, "runs": [_run(provenance=None)]}
    )["policies"]["primary-then-panel"]
    assert no_provenance["metrics"]["reviewer_calls"]["status"] == "unavailable"
    assert "r1" in no_provenance["metrics"]["reviewer_calls"]["reason"]
    assert no_provenance["primary_to_panel_approval_regressions"]["status"] == "unavailable"

    per_metric = evaluate_frozen_artifacts(
        {
            "schema_version": 1,
            "runs": [
                _run(
                    metrics={"reviewer_calls": 3, "escaped_defects": 0},
                    metric_provenance={"escaped_defects": {"source": "unlabeled", "verified": False}},
                )
            ],
        }
    )["policies"]["primary-then-panel"]
    assert per_metric["metrics"]["reviewer_calls"]["status"] == "verified"
    assert per_metric["metrics"]["escaped_defects"]["status"] == "unavailable"

    # Derived avoided calls also need run provenance.
    derived = evaluate_frozen_artifacts(
        {
            "schema_version": 1,
            "runs": [
                _run(
                    metrics={},
                    rounds=[
                        {"required_reviewers": ["p", "s1", "s2"], "selected_reviewers": ["p"]},
                        {"required_reviewers": ["p", "s1", "s2"], "selected_reviewers": ["s1", "s2"]},
                    ],
                ),
                _run(run_id="r2", provenance=None, metrics={}, rounds=[
                    {"required_reviewers": ["p", "s1"], "selected_reviewers": ["p"]},
                ]),
            ],
        }
    )["policies"]["primary-then-panel"]
    assert derived["metrics"]["calls_avoided"]["status"] == "unavailable"
    only_verified = evaluate_frozen_artifacts(
        {
            "schema_version": 1,
            "runs": [
                _run(
                    metrics={},
                    rounds=[
                        {"required_reviewers": ["p", "s1", "s2"], "selected_reviewers": ["p"]},
                        {"required_reviewers": ["p", "s1", "s2"], "selected_reviewers": ["s1", "s2"]},
                    ],
                )
            ],
        }
    )["policies"]["primary-then-panel"]
    assert only_verified["metrics"]["calls_avoided"] == {"value": 3, "status": "verified"}


def test_findings_are_namespaced_by_run_and_marginal_rows_are_not_applicable_without_primary():
    report = evaluate_frozen_artifacts(
        {
            "schema_version": 1,
            "runs": [
                _run(run_id="a", findings=[
                    {"id": "finding-1", "severity": "high", "valid": True, "contributors": ["primary"]},
                ]),
                _run(run_id="b", findings=[
                    {"id": "finding-1", "severity": "high", "valid": True, "contributors": ["secondary"]},
                ]),
                _run(run_id="c", policy="all-reviewers", primary_reviewer=None, findings=[
                    {"id": "finding-1", "severity": "low", "valid": True, "contributors": ["x"]},
                ]),
                _run(run_id="d", policy="selective-intermediate", primary_reviewer=None, findings=[]),
            ],
        }
    )
    staged = report["policies"]["primary-then-panel"]
    # Two distinct findings even though both use the ID "finding-1".
    assert staged["valid_unique_findings"] == {"value": 2, "status": "verified"}
    # Run a's primary coverage does not suppress run b's secondary finding.
    assert staged["marginal_findings_beyond_primary"]["value"] == {"secondary": ["b:finding-1"]}
    for policy in ("all-reviewers", "selective-intermediate"):
        row = report["policies"][policy]
        assert row["marginal_findings_beyond_primary"]["status"] == "not-applicable"
        assert row["severity_weighted_marginal_findings"]["status"] == "not-applicable"
        assert row["primary_to_panel_approval_regressions"]["status"] == "not-applicable"
    assert report["policies"]["all-reviewers"]["valid_unique_findings"] == {"value": 1, "status": "verified"}
    # A historical full-board run may declare a hypothetical primary for coverage comparison.
    hypothetical = evaluate_frozen_artifacts(
        {"schema_version": 1, "runs": [_run(policy="all-reviewers")]}
    )["policies"]["all-reviewers"]
    assert hypothetical["marginal_findings_beyond_primary"]["status"] == "verified"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"findings": [{"id": "dup", "valid": True}, {"id": "dup", "valid": True}]}, "repeats finding ID"),
        ({"findings": [{"id": "x", "valid": "yes"}]}, "valid label must be boolean"),
        ({"primary_reviewer": None}, "without a primary_reviewer"),
        ({"provenance": {"source": "", "verified": True}}, "non-empty source"),
        ({"provenance": {"source": "log", "verified": "true"}}, "boolean verified"),
        ({"label_provenance": "verified"}, "must be an object or null"),
        ({"metric_provenance": ["x"]}, "metric_provenance must be an object"),
        ({"policy": "unknown"}, "unsupported policy"),
    ],
)
def test_schema_validation_rejects_untrustworthy_artifacts(tmp_path, overrides, message):
    path = tmp_path / "runs.json"
    path.write_text(json.dumps({"schema_version": 1, "runs": [_run(**overrides)]}), encoding="utf-8")
    with pytest.raises(AgentLoopError, match=message):
        load_frozen_artifacts(path)


def test_offline_evaluation_loads_local_json_and_rejects_live_schema(tmp_path):
    path = tmp_path / "runs.json"
    path.write_text(json.dumps({"schema_version": 1, "runs": []}), encoding="utf-8")
    assert load_frozen_artifacts(path)["runs"] == []
    path.write_text(json.dumps({"schema_version": 2, "runs": []}), encoding="utf-8")
    with pytest.raises(AgentLoopError, match="schema"):
        load_frozen_artifacts(path)


def test_checked_in_fixture_report_is_reproducible_and_cli_never_touches_github(tmp_path, capsys):
    import coding_review_agent_loop.review_evaluation as module

    assert not any(name in module.__dict__ for name in ("github", "run_pr_loop", "post_pr_comment"))
    expected = json.loads(FIXTURE_REPORT.read_text(encoding="utf-8"))
    report = evaluate_frozen_artifacts(load_frozen_artifacts(FIXTURE_ARTIFACTS))
    assert report == expected
    selective = report["policies"]["selective-intermediate"]
    assert selective["metrics"]["escaped_defects"]["status"] == "unavailable"
    assert selective["marginal_findings_beyond_primary"]["status"] == "not-applicable"
    assert report["policies"]["primary-then-panel"]["metrics"]["calls_avoided"] == {
        "value": 1, "status": "verified",
    }

    output = tmp_path / "report.json"
    assert cli_main(["review-evaluation", str(FIXTURE_ARTIFACTS), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected
    assert cli_main(["evaluate-reviews", str(FIXTURE_ARTIFACTS), "--format", "text"]) == 0
    assert "Frozen PR review policy evaluation" in capsys.readouterr().out
    assert not any(command[:1] == ["gh"] for command in [])
    assert cli_main(["review-evaluation", str(tmp_path / "missing.json")]) == 1
