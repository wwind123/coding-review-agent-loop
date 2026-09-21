import json
from pathlib import Path

import pytest

from coding_review_agent_loop.cli import main as cli_main
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.review_evaluation import (
    FLOW_POLICIES,
    FLOWS,
    PLAN_POLICIES,
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


def test_partial_approval_round_data_makes_regressions_unavailable_not_partially_verified():
    complete = _run(run_id="complete")
    missing_panel = _run(run_id="missing-panel", panel_approval_round=None)
    row = evaluate_frozen_artifacts(
        {"schema_version": 1, "runs": [complete, missing_panel]}
    )["policies"]["primary-then-panel"]["primary_to_panel_approval_regressions"]
    assert row["status"] == "unavailable"
    assert row["value"] is None
    assert "missing-panel" in row["reason"]
    assert "complete" not in row["reason"].replace("incomplete", "")

    missing_primary = _run(run_id="missing-primary", primary_approval_round=None)
    row = evaluate_frozen_artifacts(
        {"schema_version": 1, "runs": [complete, missing_primary]}
    )["policies"]["primary-then-panel"]["primary_to_panel_approval_regressions"]
    assert row["status"] == "unavailable"
    assert "missing-primary" in row["reason"]

    unverified = _run(run_id="unverified", provenance={"source": "log", "verified": False})
    row = evaluate_frozen_artifacts(
        {"schema_version": 1, "runs": [complete, unverified]}
    )["policies"]["primary-then-panel"]["primary_to_panel_approval_regressions"]
    assert row["status"] == "unavailable"
    assert "unverified" in row["reason"]
    assert "provenance" in row["reason"]

    both_complete = _run(run_id="second", primary_approval_round=2, panel_approval_round=2)
    row = evaluate_frozen_artifacts(
        {"schema_version": 1, "runs": [complete, both_complete]}
    )["policies"]["primary-then-panel"]["primary_to_panel_approval_regressions"]
    assert row == {"value": [2, 0], "status": "verified"}


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


def test_missing_severity_on_valid_marginal_finding_makes_weighted_row_unavailable():
    report = evaluate_frozen_artifacts(
        {
            "schema_version": 1,
            "runs": [
                _run(findings=[
                    {"id": "primary-finding", "severity": "high", "valid": True, "contributors": ["primary"]},
                    {"id": "weighted", "severity": "low", "valid": True, "contributors": ["secondary"]},
                    # A serious finding whose severity label is absent must not
                    # silently weigh zero and vanish from the comparison.
                    {"id": "unlabeled-severity", "valid": True, "contributors": ["secondary"]},
                ]),
            ],
        }
    )["policies"]["primary-then-panel"]
    # Unique and marginal coverage do not depend on severity and stay verified.
    assert report["valid_unique_findings"] == {"value": 3, "status": "verified"}
    assert report["marginal_findings_beyond_primary"]["status"] == "verified"
    assert report["marginal_findings_beyond_primary"]["value"] == {
        "secondary": ["r1:unlabeled-severity", "r1:weighted"],
    }
    weighted = report["severity_weighted_marginal_findings"]
    assert weighted["status"] == "unavailable"
    assert weighted["value"] is None
    assert "r1:unlabeled-severity" in weighted["reason"]
    assert weighted["unweighted_findings"] == 1
    assert "severity-weighted marginal findings: unavailable" in render_evaluation_report(
        {"policies": {"primary-then-panel": report}}, human=True
    )

    # A missing severity on an invalid or primary-covered finding never enters
    # the weighted sum, so the weighted row stays verified.
    unaffected = evaluate_frozen_artifacts(
        {
            "schema_version": 1,
            "runs": [
                _run(findings=[
                    {"id": "primary-finding", "valid": True, "contributors": ["primary"]},
                    {"id": "withdrawn", "valid": False, "contributors": ["secondary"]},
                    {"id": "panel-finding", "severity": "critical", "valid": True, "contributors": ["secondary"]},
                ]),
            ],
        }
    )["policies"]["primary-then-panel"]
    assert unaffected["severity_weighted_marginal_findings"] == {"status": "verified", "value": {"secondary": 5}}


@pytest.mark.parametrize("severity", ["blocker", "", "  ", "HIGHEST", 3, True, ["high"]])
def test_unknown_severity_on_a_finding_fails_closed_instead_of_weighing_zero(tmp_path, severity):
    run = _run(findings=[
        {"id": "primary-finding", "severity": "high", "valid": True, "contributors": ["primary"]},
        {"id": "odd", "severity": severity, "valid": True, "contributors": ["secondary"]},
    ])
    path = tmp_path / "runs.json"
    path.write_text(json.dumps({"schema_version": 1, "runs": [run]}), encoding="utf-8")
    with pytest.raises(AgentLoopError, match="severity"):
        load_frozen_artifacts(path)
    # Direct evaluation of an unvalidated artifact fails the same way rather
    # than publishing a verified weighted row that omits the finding.
    with pytest.raises(AgentLoopError, match="severity"):
        evaluate_frozen_artifacts({"schema_version": 1, "runs": [run]})


def test_severity_labels_are_case_insensitive_and_weighted_after_validation(tmp_path):
    path = tmp_path / "runs.json"
    path.write_text(
        json.dumps({"schema_version": 1, "runs": [_run(findings=[
            {"id": "primary-finding", "severity": "High", "valid": True, "contributors": ["primary"]},
            {"id": "panel-finding", "severity": " CRITICAL ", "valid": True, "contributors": ["secondary"]},
        ])]}),
        encoding="utf-8",
    )
    loaded = load_frozen_artifacts(path)
    assert [finding["severity"] for finding in loaded["runs"][0]["findings"]] == ["high", "critical"]
    row = evaluate_frozen_artifacts(loaded)["policies"]["primary-then-panel"]
    assert row["severity_weighted_marginal_findings"] == {"status": "verified", "value": {"secondary": 5}}


@pytest.mark.parametrize(
    "contributors",
    [
        None,  # absent key entirely
        "secondary",  # wrong type: string instead of array
        [],  # empty array
        ["primary", ""],  # partially invalid: blank entry
        ["primary", "   "],  # partially invalid: whitespace-only entry
        ["primary", 3],  # partially invalid: non-string entry
        [None],
        {"name": "secondary"},  # wrong type: object
    ],
)
def test_missing_or_malformed_contributors_on_a_valid_finding_fail_closed(tmp_path, contributors):
    finding = {"id": "security-gap", "severity": "critical", "valid": True}
    if contributors is not None:
        finding["contributors"] = contributors
    run = _run(findings=[
        {"id": "primary-finding", "severity": "high", "valid": True, "contributors": ["primary"]},
        finding,
    ])
    path = tmp_path / "runs.json"
    path.write_text(json.dumps({"schema_version": 1, "runs": [run]}), encoding="utf-8")
    with pytest.raises(AgentLoopError, match="contributor"):
        load_frozen_artifacts(path)
    # Direct evaluation of an unvalidated artifact must not silently drop the
    # valid finding and publish a verified unique-coverage count of one.
    with pytest.raises(AgentLoopError, match="contributor"):
        evaluate_frozen_artifacts({"schema_version": 1, "runs": [run]})


def test_contributors_are_stripped_and_deduplicated_but_never_dropped(tmp_path):
    path = tmp_path / "runs.json"
    path.write_text(
        json.dumps({"schema_version": 1, "runs": [_run(findings=[
            {"id": "primary-finding", "severity": "high", "valid": True, "contributors": ["primary"]},
            {"id": "shared", "severity": "critical", "valid": True, "reviewers": [" secondary ", "secondary", "other"]},
        ])]}),
        encoding="utf-8",
    )
    loaded = load_frozen_artifacts(path)
    assert loaded["runs"][0]["findings"][1]["contributors"] == ["secondary", "other"]
    row = evaluate_frozen_artifacts(loaded)["policies"]["primary-then-panel"]
    assert row["valid_unique_findings"] == {"value": 2, "status": "verified"}
    assert row["marginal_findings_beyond_primary"]["value"] == {
        "other": ["r1:shared"],
        "secondary": ["r1:shared"],
    }
    assert row["severity_weighted_marginal_findings"] == {
        "status": "verified",
        "value": {"other": 5, "secondary": 5},
    }


def test_duplicate_run_identity_within_a_policy_is_rejected_not_collapsed(tmp_path):
    first = _run(run_id="pr-1", findings=[
        {"id": "finding-1", "severity": "high", "valid": True, "contributors": ["primary"]},
    ])
    second = _run(run_id="pr-1", findings=[
        {"id": "finding-1", "severity": "critical", "valid": True, "contributors": ["secondary"]},
    ])
    path = tmp_path / "runs.json"
    path.write_text(json.dumps({"schema_version": 1, "runs": [first, second]}), encoding="utf-8")
    with pytest.raises(AgentLoopError, match="repeat run ID 'pr-1' for policy 'primary-then-panel'"):
        load_frozen_artifacts(path)
    # The evaluator itself refuses the collision even when validation is
    # bypassed, so the two distinct findings can never be collapsed into one.
    with pytest.raises(AgentLoopError, match="repeat run ID"):
        evaluate_frozen_artifacts({"schema_version": 1, "runs": [first, second]})

    # Distinct run IDs for the same records keep both findings visible.
    distinct = evaluate_frozen_artifacts(
        {"schema_version": 1, "runs": [first, dict(second, run_id="pr-2")]}
    )["policies"]["primary-then-panel"]
    assert distinct["valid_unique_findings"] == {"value": 2, "status": "verified"}
    assert distinct["marginal_findings_beyond_primary"]["value"] == {"secondary": ["pr-2:finding-1"]}
    assert distinct["severity_weighted_marginal_findings"] == {"status": "verified", "value": {"secondary": 5}}

    # The same run ID under a different policy is a different namespace, not a
    # duplicate: policies are compared independently.
    cross_policy = evaluate_frozen_artifacts(
        {"schema_version": 1, "runs": [first, dict(second, policy="all-reviewers")]}
    )
    assert cross_policy["policies"]["primary-then-panel"]["valid_unique_findings"]["value"] == 1
    assert cross_policy["policies"]["all-reviewers"]["valid_unique_findings"]["value"] == 1

    # Omitted run IDs default to the record index and therefore stay unique.
    indexed = [dict(_run(), run_id=None) for _ in range(2)]
    for run in indexed:
        del run["run_id"]
    path.write_text(json.dumps({"schema_version": 1, "runs": indexed}), encoding="utf-8")
    assert [run["run_id"] for run in load_frozen_artifacts(path)["runs"]] == ["1", "2"]


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {"findings": [
                {"id": "dup", "valid": True, "contributors": ["primary"]},
                {"id": "dup", "valid": True, "contributors": ["primary"]},
            ]},
            "repeats finding ID",
        ),
        ({"findings": [{"id": "x", "severity": "urgent", "valid": True, "contributors": ["primary"]}]}, "unsupported severity"),
        ({"findings": [{"id": "x", "severity": "high", "valid": True}]}, "no contributors array"),
        ({"findings": [{"id": "x", "severity": "high", "valid": False, "contributors": []}]}, "at least one contributor"),
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
    # The fixture carries both planning policies, reported separately from PR.
    plan = report["flows"]["plan"]["policies"]
    assert set(plan) == set(PLAN_POLICIES)
    assert plan["all-reviewers"]["metrics"]["reviewer_calls"] == {"value": 6, "status": "verified"}
    assert plan["all-reviewers"]["metrics"]["escaped_defects"] == {"value": 1, "status": "verified"}
    assert plan["primary-then-panel"]["metrics"]["reviewer_calls"] == {"value": 4, "status": "verified"}
    assert plan["primary-then-panel"]["metrics"]["tokens"] == {"value": 16000, "status": "verified"}
    assert plan["primary-then-panel"]["metrics"]["elapsed_seconds"] == {"value": 260, "status": "verified"}
    assert plan["primary-then-panel"]["metrics"]["calls_avoided"] == {"value": 3, "status": "verified"}
    # Escaped plan defects were never measured for the staged run; the row says
    # so instead of borrowing the full-board figure.
    assert plan["primary-then-panel"]["metrics"]["escaped_defects"]["status"] == "unavailable"
    # Reviewer overlap and severity-weighted marginal findings are reported for
    # both planning policies, so staged planning can be compared against the
    # full-board baseline rather than against a not-applicable row.
    for policy in PLAN_POLICIES:
        assert plan[policy]["marginal_findings_beyond_primary"]["status"] == "verified"
        assert plan[policy]["severity_weighted_marginal_findings"] == {
            "status": "verified", "value": {"Anthropic Claude": 3},
        }
    assert plan["all-reviewers"]["marginal_findings_beyond_primary"]["value"] == {
        "Anthropic Claude": ["historical-plan-all-001:plan-untested-fallback"],
    }
    assert plan["primary-then-panel"]["marginal_findings_beyond_primary"]["value"] == {
        "Anthropic Claude": ["historical-plan-primary-panel-001:plan-untested-fallback"],
    }
    assert plan["primary-then-panel"]["primary_to_panel_approval_regressions"] == {
        "value": [1], "status": "verified",
    }
    # The full-board planning run has no primary-to-panel transition to
    # measure, so that row is unavailable naming the run rather than estimated.
    regressions = plan["all-reviewers"]["primary_to_panel_approval_regressions"]
    assert regressions["status"] == "unavailable"
    assert "historical-plan-all-001" in regressions["reason"]

    output = tmp_path / "report.json"
    assert cli_main(["review-evaluation", str(FIXTURE_ARTIFACTS), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected
    assert cli_main(["evaluate-reviews", str(FIXTURE_ARTIFACTS), "--format", "text"]) == 0
    rendered = capsys.readouterr().out
    assert "Frozen PR review policy evaluation" in rendered
    assert "Frozen plan review policy evaluation" in rendered
    assert not any(command[:1] == ["gh"] for command in [])
    assert cli_main(["review-evaluation", str(tmp_path / "missing.json")]) == 1


def _write(tmp_path, runs, name="runs.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"schema_version": 1, "runs": runs}), encoding="utf-8")
    return path


def _plan_run(**overrides):
    return _run(flow="plan", **overrides)


def test_runs_default_to_the_pr_flow_and_reject_an_unknown_flow(tmp_path):
    loaded = load_frozen_artifacts(_write(tmp_path, [_run()]))
    assert [run["flow"] for run in loaded["runs"]] == ["pr"]

    path = _write(tmp_path, [_run(flow="issue")])
    with pytest.raises(AgentLoopError, match="unsupported flow"):
        load_frozen_artifacts(path)
    for bad in ("", "   ", 3, [], {}):
        with pytest.raises(AgentLoopError, match="flow"):
            load_frozen_artifacts(_write(tmp_path, [_run(flow=bad)]))


def test_legacy_pr_only_artifact_without_flow_produces_byte_identical_pr_rows(tmp_path):
    legacy_runs = [
        {key: value for key, value in _run(run_id="legacy").items()},
        {key: value for key, value in _run(run_id="legacy-full", policy="all-reviewers").items()},
    ]
    legacy = evaluate_frozen_artifacts(load_frozen_artifacts(_write(tmp_path, legacy_runs)))
    labeled_runs = [dict(run, flow="pr") for run in legacy_runs]
    labeled = evaluate_frozen_artifacts(load_frozen_artifacts(_write(tmp_path, labeled_runs, "labeled.json")))
    assert legacy["flows"]["pr"]["policies"] == labeled["flows"]["pr"]["policies"]
    # The historical ``policies`` key keeps naming the PR rows exactly.
    assert legacy["policies"] == legacy["flows"]["pr"]["policies"]
    assert legacy["flows"]["plan"]["run_count"] == 0
    for policy in PLAN_POLICIES:
        assert legacy["flows"]["plan"]["policies"][policy]["run_count"] == 0
        assert legacy["flows"]["plan"]["policies"][policy]["status"] == "unavailable"


def test_planning_runs_never_contaminate_pr_rows_and_pr_runs_never_contaminate_planning_rows(tmp_path):
    pr_only = evaluate_frozen_artifacts(
        load_frozen_artifacts(_write(tmp_path, [_run(run_id="shared")], "pr.json"))
    )
    plan_only = evaluate_frozen_artifacts(
        load_frozen_artifacts(_write(tmp_path, [_plan_run(run_id="shared", metrics={"reviewer_calls": 99})], "plan.json"))
    )
    # The same policy name and the same run ID in both flows.
    mixed = evaluate_frozen_artifacts(
        load_frozen_artifacts(
            _write(
                tmp_path,
                [_run(run_id="shared"), _plan_run(run_id="shared", metrics={"reviewer_calls": 99})],
                "mixed.json",
            )
        )
    )
    assert mixed["flows"]["pr"]["policies"] == pr_only["flows"]["pr"]["policies"]
    assert mixed["flows"]["plan"]["policies"] == plan_only["flows"]["plan"]["policies"]
    pr_row = mixed["flows"]["pr"]["policies"]["primary-then-panel"]
    plan_row = mixed["flows"]["plan"]["policies"]["primary-then-panel"]
    assert pr_row["run_count"] == 1 and plan_row["run_count"] == 1
    # Calls, findings, and every other measurement stay unpooled.
    assert pr_row["metrics"]["reviewer_calls"] == {"value": 3, "status": "verified"}
    assert plan_row["metrics"]["reviewer_calls"] == {"value": 99, "status": "verified"}
    assert pr_row["valid_unique_findings"] == {"value": 2, "status": "verified"}
    assert plan_row["valid_unique_findings"] == {"value": 2, "status": "verified"}
    assert pr_row["marginal_findings_beyond_primary"]["value"] == {"secondary": ["shared:panel-finding"]}
    assert plan_row["marginal_findings_beyond_primary"]["value"] == {"secondary": ["shared:panel-finding"]}


def test_run_identity_uniqueness_is_per_flow_and_policy(tmp_path):
    # The same ID in two flows is two namespaces, not a duplicate.
    both_flows = load_frozen_artifacts(
        _write(tmp_path, [_run(run_id="r"), _plan_run(run_id="r")], "both.json")
    )
    assert [(run["flow"], run["run_id"]) for run in both_flows["runs"]] == [("pr", "r"), ("plan", "r")]

    duplicate = _write(tmp_path, [_plan_run(run_id="r"), _plan_run(run_id="r")], "dup.json")
    with pytest.raises(AgentLoopError, match="repeat run ID 'r' for policy 'primary-then-panel' in flow 'plan'"):
        load_frozen_artifacts(duplicate)
    with pytest.raises(AgentLoopError, match="repeat run ID"):
        evaluate_frozen_artifacts(
            {"schema_version": 1, "runs": [_plan_run(run_id="r"), _plan_run(run_id="r")]}
        )


def test_a_pr_only_policy_is_rejected_on_a_planning_run(tmp_path):
    assert FLOW_POLICIES["plan"] == PLAN_POLICIES
    assert set(FLOWS) == {"pr", "plan"}
    with pytest.raises(AgentLoopError, match="unsupported policy 'selective-intermediate' for flow 'plan'"):
        load_frozen_artifacts(
            _write(tmp_path, [_plan_run(policy="selective-intermediate", primary_reviewer=None)], "bad.json")
        )
    # The same policy stays valid on the PR flow.
    assert load_frozen_artifacts(
        _write(tmp_path, [_run(policy="selective-intermediate", primary_reviewer=None)], "ok.json")
    )["runs"][0]["policy"] == "selective-intermediate"


def test_text_report_titles_and_aggregates_each_flow_separately():
    rendered = render_evaluation_report(
        evaluate_frozen_artifacts(
            {"schema_version": 1, "runs": [_run(), _plan_run(run_id="p1", metrics={"reviewer_calls": 99})]}
        ),
        human=True,
    )
    pr_section, plan_section = rendered.split("Frozen plan review policy evaluation")
    assert pr_section.startswith("Frozen PR review policy evaluation")
    # Only the PR flow reports the PR-only policy.
    assert "selective-intermediate" in pr_section
    assert "selective-intermediate" not in plan_section
    assert "reviewer_calls: 3" in pr_section
    assert "reviewer_calls: 99" in plan_section
    assert "reviewer_calls: 99" not in pr_section


@pytest.mark.parametrize("flow", ["issue", "PR-flow", "", "   ", 0, [], {}, 3, True])
def test_direct_evaluation_rejects_a_malformed_flow_instead_of_dropping_or_defaulting(flow):
    # Direct evaluation of an unvalidated artifact is an exercised public path.
    # An unknown flow must not silently drop the run from every report, and a
    # falsy wrong-typed flow must not be coerced into the pr default.
    run = _run(run_id="odd-flow", flow=flow)
    with pytest.raises(AgentLoopError, match="flow"):
        evaluate_frozen_artifacts({"schema_version": 1, "runs": [run]})


def test_only_an_absent_flow_key_defaults_to_pr_and_an_explicit_null_is_rejected(tmp_path):
    # The compatibility exception covers legacy PR artifacts that carry no
    # flow key at all. An explicit null is a labeled run missing its label, so
    # assigning it to pr could let a planning run contaminate the PR rows.
    absent = _run(run_id="absent-flow")
    assert "flow" not in absent
    report = evaluate_frozen_artifacts({"schema_version": 1, "runs": [absent]})
    assert report["flows"]["pr"]["run_count"] == 1
    assert report["flows"]["plan"]["run_count"] == 0
    assert report["flows"]["pr"]["policies"]["primary-then-panel"]["run_count"] == 1
    assert load_frozen_artifacts(_write(tmp_path, [absent]))["runs"][0]["flow"] == "pr"

    explicit_null = _run(run_id="null-flow", flow=None)
    for artifacts in (
        {"schema_version": 1, "runs": [explicit_null]},
        {"schema_version": 1, "runs": [_run(), explicit_null]},
    ):
        with pytest.raises(AgentLoopError, match="explicit null flow"):
            evaluate_frozen_artifacts(artifacts)
    with pytest.raises(AgentLoopError, match="explicit null flow"):
        load_frozen_artifacts(_write(tmp_path, [explicit_null], "null.json"))


def test_direct_evaluation_normalizes_flow_case_and_surrounding_space():
    report = evaluate_frozen_artifacts(
        {"schema_version": 1, "runs": [_run(run_id="cased", flow=" Plan ")]}
    )
    assert report["flows"]["plan"]["run_count"] == 1
    assert report["flows"]["pr"]["run_count"] == 0
    # A duplicate identity is still caught after normalization.
    with pytest.raises(AgentLoopError, match="repeat run ID 'cased'"):
        evaluate_frozen_artifacts(
            {
                "schema_version": 1,
                "runs": [_run(run_id="cased", flow="plan"), _run(run_id="cased", flow=" PLAN ")],
            }
        )


def test_direct_evaluation_rejects_a_policy_foreign_to_its_flow_instead_of_dropping_it():
    # A policy string valid in the pr flow must not be silently unassignable in
    # the plan flow: counting the run in flows.plan.run_count while no plan
    # policy row holds it would hide its calls, tokens, latency, findings, and
    # escaped defects from every report.
    with pytest.raises(AgentLoopError, match="unsupported policy 'selective-intermediate' for flow 'plan'"):
        evaluate_frozen_artifacts(
            {
                "schema_version": 1,
                "runs": [
                    _plan_run(
                        run_id="foreign-policy",
                        policy="selective-intermediate",
                        primary_reviewer=None,
                    )
                ],
            }
        )
    with pytest.raises(AgentLoopError, match="policy"):
        evaluate_frozen_artifacts(
            {"schema_version": 1, "runs": [_plan_run(run_id="blank-policy", policy="   ")]}
        )
    # The same policy stays valid on the pr flow through the direct path.
    pr_report = evaluate_frozen_artifacts(
        {
            "schema_version": 1,
            "runs": [_run(run_id="pr-selective", policy="selective-intermediate", primary_reviewer=None)],
        }
    )
    assert pr_report["flows"]["pr"]["policies"]["selective-intermediate"]["run_count"] == 1


@pytest.mark.parametrize("entry", ["oops", 3, None, ["run"], True])
def test_direct_evaluation_rejects_a_non_object_run_instead_of_dropping_it(entry):
    # The loading path already fails closed on a non-object run; the direct
    # path must too, or the entry contributes to no flow run_count, no policy
    # row, and no diagnostic.
    with pytest.raises(AgentLoopError, match="must be an object"):
        evaluate_frozen_artifacts({"schema_version": 1, "runs": [_run(), entry]})


def test_every_counted_run_belongs_to_exactly_one_policy_row_of_its_flow():
    report = evaluate_frozen_artifacts(
        {
            "schema_version": 1,
            "runs": [
                _run(run_id="pr-staged"),
                _run(run_id="pr-full", policy="all-reviewers", primary_reviewer=None),
                _plan_run(run_id="plan-staged"),
                _plan_run(run_id="plan-full", policy="all-reviewers", primary_reviewer=None),
            ],
        }
    )
    for flow in FLOWS:
        row = report["flows"][flow]
        counted = sum(policy_row["run_count"] for policy_row in row["policies"].values())
        assert counted == row["run_count"]


# --- Review contract dimension (#894) --------------------------------------

REAL_RUN_ARTIFACTS = REPO_ROOT / "docs" / "evaluation" / "review_contract_runs.json"
REAL_RUN_REPORT = REPO_ROOT / "docs" / "evaluation" / "review_contract_report.json"
LABEL_EVIDENCE = {"source": "tool commit abc123 for every review round", "verified": True}
CONTRACT_VALUE_KEYS = (
    "review_rounds",
    "reviewer_calls",
    "coder_followup_rounds",
    "escaped_defects",
    "review_rounds_per_run",
    "reviewer_calls_per_run",
    "escaped_defects_per_run",
)


def _contract_run(run_id, policy, contract, *, rounds, calls, escaped=0, followups=1, **overrides):
    run = _run(
        run_id=run_id,
        policy=policy,
        review_contract=contract,
        review_contract_provenance=LABEL_EVIDENCE,
        metrics={
            "review_rounds": rounds,
            "reviewer_calls": calls,
            "coder_followup_rounds": followups,
            "escaped_defects": escaped,
        },
    )
    run.update(overrides)
    return run


def _cell(report, contract, policy, flow="pr"):
    return report["flows"][flow]["review_contracts"][contract]["policies"][policy]


def _verified(value):
    return {"value": value, "status": "verified"}


def _stratified_runs():
    return [
        _contract_run("base-all-1", "all-reviewers", "first-finding-permitted", rounds=6, calls=18, escaped=1),
        _contract_run("base-all-2", "all-reviewers", "first-finding-permitted", rounds=4, calls=12),
        _contract_run("base-panel-1", "primary-then-panel", "first-finding-permitted", rounds=11, calls=13),
        _contract_run("exh-all-1", "all-reviewers", "exhaustive", rounds=3, calls=9),
        _contract_run("exh-panel-1", "primary-then-panel", "exhaustive", rounds=4, calls=6, escaped=1),
        _contract_run("exh-panel-2", "primary-then-panel", "exhaustive", rounds=2, calls=4),
        _contract_run("exh-panel-3", "primary-then-panel", "exhaustive", rounds=3, calls=5),
    ]


def test_absent_review_contract_defaults_without_verifying_and_leaves_policy_rows_identical(tmp_path):
    legacy = [_run(run_id="legacy-1"), _run(run_id="legacy-2", policy="all-reviewers")]
    loaded = load_frozen_artifacts(_write(tmp_path, legacy))
    # A defaulted label is not written into the normalized run, so it stays
    # distinguishable from an explicit one and the artifact hash is unchanged.
    assert all("review_contract" not in run for run in loaded["runs"])
    report = evaluate_frozen_artifacts(loaded)
    labelled = [dict(run, review_contract="first-finding-permitted") for run in legacy]
    labelled_report = evaluate_frozen_artifacts(load_frozen_artifacts(_write(tmp_path, labelled, "labelled.json")))
    assert json.dumps(report["policies"], sort_keys=True) == json.dumps(labelled_report["policies"], sort_keys=True)
    for flow in FLOWS:
        assert json.dumps(report["flows"][flow]["policies"], sort_keys=True) == json.dumps(
            labelled_report["flows"][flow]["policies"], sort_keys=True
        )
    for policy, run_id in (("primary-then-panel", "legacy-1"), ("all-reviewers", "legacy-2")):
        cell = _cell(report, "first-finding-permitted", policy)
        assert cell["run_count"] == 1
        for key in CONTRACT_VALUE_KEYS:
            assert cell[key]["status"] == "unavailable"
            assert cell[key]["value"] is None
            assert run_id in cell[key]["reason"]
        other = "all-reviewers" if policy == "primary-then-panel" else "primary-then-panel"
        assert run_id not in json.dumps(_cell(report, "first-finding-permitted", other))
    for flow in FLOWS:
        for policy in FLOW_POLICIES[flow]:
            cell = _cell(report, "exhaustive", policy, flow)
            assert cell["run_count"] == 0
            for key in CONTRACT_VALUE_KEYS:
                assert cell[key] == {
                    "value": None,
                    "status": "unavailable",
                    "reason": "no frozen runs for this review contract and policy",
                }


def test_contract_cells_are_stratified_by_policy_and_never_pooled(tmp_path):
    runs = _stratified_runs()
    report = evaluate_frozen_artifacts(load_frozen_artifacts(_write(tmp_path, runs)))
    assert report == evaluate_frozen_artifacts({"schema_version": 1, "runs": runs}) | {
        "artifact_sha256": report["artifact_sha256"]
    }

    base_all = _cell(report, "first-finding-permitted", "all-reviewers")
    assert base_all["run_count"] == 2
    assert base_all["review_rounds"] == _verified(10)
    assert base_all["reviewer_calls"] == _verified(30)
    assert base_all["coder_followup_rounds"] == _verified(2)
    assert base_all["escaped_defects"] == _verified(1)
    assert base_all["review_rounds_per_run"] == _verified(5.0)
    assert base_all["reviewer_calls_per_run"] == _verified(15.0)
    assert base_all["escaped_defects_per_run"] == _verified(0.5)

    base_panel = _cell(report, "first-finding-permitted", "primary-then-panel")
    assert base_panel["run_count"] == 1
    assert base_panel["review_rounds_per_run"] == _verified(11.0)
    assert base_panel["reviewer_calls_per_run"] == _verified(13.0)

    exh_all = _cell(report, "exhaustive", "all-reviewers")
    assert exh_all["run_count"] == 1
    assert exh_all["reviewer_calls_per_run"] == _verified(9.0)

    exh_panel = _cell(report, "exhaustive", "primary-then-panel")
    assert exh_panel["run_count"] == 3
    assert exh_panel["review_rounds"] == _verified(9)
    assert exh_panel["reviewer_calls"] == _verified(15)
    assert exh_panel["review_rounds_per_run"] == _verified(3.0)
    assert exh_panel["reviewer_calls_per_run"] == _verified(5.0)
    assert exh_panel["escaped_defects_per_run"] == _verified(1 / 3)

    # Shape: both contracts, every policy of the flow, and no rollup anywhere.
    for flow in FLOWS:
        section = report["flows"][flow]["review_contracts"]
        assert set(section) == {"first-finding-permitted", "exhaustive"}
        for contract_section in section.values():
            assert set(contract_section) == {"policies"}
            assert set(contract_section["policies"]) == set(FLOW_POLICIES[flow])
    assert "selective-intermediate" not in report["flows"]["plan"]["review_contracts"]["exhaustive"]["policies"]
    assert "review_contracts" not in report
    assert set(report["flows"]["pr"]) == {"title", "run_count", "policies", "review_contracts"}

    # Changing a run under one policy never changes another policy's cells.
    changed = _stratified_runs()
    changed[0]["metrics"]["reviewer_calls"] = 99
    changed[0]["metrics"]["review_rounds"] = 50
    changed_report = evaluate_frozen_artifacts({"schema_version": 1, "runs": changed})
    assert _cell(changed_report, "first-finding-permitted", "all-reviewers") != base_all
    for contract, policy in (
        ("first-finding-permitted", "primary-then-panel"),
        ("first-finding-permitted", "selective-intermediate"),
        ("exhaustive", "all-reviewers"),
        ("exhaustive", "primary-then-panel"),
    ):
        assert _cell(changed_report, contract, policy) == _cell(report, contract, policy)


def test_pr_and_plan_contract_cells_never_pool():
    runs = [
        _contract_run("shared", "all-reviewers", "exhaustive", rounds=2, calls=6),
        _contract_run("shared", "all-reviewers", "exhaustive", rounds=5, calls=20, flow="plan"),
    ]
    report = evaluate_frozen_artifacts({"schema_version": 1, "runs": runs})
    assert _cell(report, "exhaustive", "all-reviewers", "pr")["reviewer_calls"] == _verified(6)
    assert _cell(report, "exhaustive", "all-reviewers", "plan")["reviewer_calls"] == _verified(20)
    assert _cell(report, "exhaustive", "all-reviewers", "pr")["run_count"] == 1
    assert _cell(report, "exhaustive", "all-reviewers", "plan")["run_count"] == 1


@pytest.mark.parametrize(
    "label_provenance",
    [None, {"source": "unconfirmed recollection", "verified": False}],
)
def test_unevidenced_label_makes_only_its_own_cell_unavailable(label_provenance):
    runs = _stratified_runs()
    runs[4]["review_contract_provenance"] = label_provenance
    if label_provenance is None:
        del runs[4]["review_contract_provenance"]
    report = evaluate_frozen_artifacts({"schema_version": 1, "runs": runs})
    cell = _cell(report, "exhaustive", "primary-then-panel")
    assert cell["run_count"] == 3
    for key in CONTRACT_VALUE_KEYS:
        assert cell[key]["status"] == "unavailable"
        assert "exh-panel-1" in cell[key]["reason"]
        assert "exh-panel-2" not in cell[key]["reason"]
    assert _cell(report, "exhaustive", "all-reviewers")["reviewer_calls_per_run"] == _verified(9.0)
    assert _cell(report, "first-finding-permitted", "primary-then-panel")["review_rounds_per_run"] == _verified(11.0)
    text = render_evaluation_report(report, human=True).split("Frozen plan review policy evaluation")[0]
    assert "before/after comparison unavailable for primary-then-panel: no fully verified cell under exhaustive" in text
    assert "before/after comparison unavailable for all-reviewers" not in text


def test_unverified_metric_provenance_makes_only_that_value_of_that_cell_unavailable():
    runs = _stratified_runs()
    runs[5]["metric_provenance"] = {
        "escaped_defects": {"source": "merged 2026-09-20; 14-day window still open", "verified": False}
    }
    report = evaluate_frozen_artifacts({"schema_version": 1, "runs": runs})
    cell = _cell(report, "exhaustive", "primary-then-panel")
    for key in ("escaped_defects", "escaped_defects_per_run"):
        assert cell[key]["status"] == "unavailable"
        assert cell[key]["reason"].endswith("for runs: exh-panel-2")
    assert cell["review_rounds_per_run"] == _verified(3.0)
    assert cell["reviewer_calls_per_run"] == _verified(5.0)
    assert cell["coder_followup_rounds"] == _verified(3)
    assert _cell(report, "exhaustive", "all-reviewers")["escaped_defects_per_run"] == _verified(0.0)

    # A run whose run-level provenance is unverified loses every value, again
    # only in its own cell, and a partial sum is never reported as verified.
    runs = _stratified_runs()
    runs[1]["provenance"] = {"source": "hand-copied", "verified": False}
    report = evaluate_frozen_artifacts({"schema_version": 1, "runs": runs})
    cell = _cell(report, "first-finding-permitted", "all-reviewers")
    for key in CONTRACT_VALUE_KEYS:
        assert cell[key]["status"] == "unavailable"
        assert cell[key]["reason"].endswith("for runs: base-all-2")
    assert _cell(report, "exhaustive", "all-reviewers")["review_rounds_per_run"] == _verified(3.0)


def test_text_report_compares_contracts_side_by_side_within_each_policy():
    report = evaluate_frozen_artifacts({"schema_version": 1, "runs": _stratified_runs()})
    text = render_evaluation_report(report, human=True)
    assert text.count("Review contract comparison (within scheduling policy)") == len(FLOWS)
    pr_text, plan_text = text.split("Frozen plan review policy evaluation")
    assert pr_text.index("calls_avoided") < pr_text.index("Review contract comparison")
    assert "  primary-then-panel (first-finding-permitted 1 runs, exhaustive 3 runs)" in pr_text
    assert "    review_rounds_per_run: first-finding-permitted=11.0 | exhaustive=3.0" in pr_text
    assert "    reviewer_calls_per_run: first-finding-permitted=15.0 | exhaustive=9.0" in pr_text
    assert "before/after comparison unavailable for all-reviewers" not in pr_text
    assert "before/after comparison unavailable for primary-then-panel" not in pr_text
    # No run exists under selective-intermediate, or anywhere in the plan flow.
    assert (
        "before/after comparison unavailable for selective-intermediate: "
        "no fully verified cell under first-finding-permitted, exhaustive"
    ) in pr_text
    assert "selective-intermediate" not in plan_text
    assert "before/after comparison unavailable for all-reviewers" in plan_text

    # A policy verified under only one contract is reported as not comparable.
    one_sided = [run for run in _stratified_runs() if run["run_id"] != "exh-all-1"]
    text = render_evaluation_report(
        evaluate_frozen_artifacts({"schema_version": 1, "runs": one_sided}), human=True
    )
    pr_text = text.split("Frozen plan review policy evaluation")[0]
    assert "reviewer_calls_per_run: first-finding-permitted=15.0 | exhaustive=unavailable" in pr_text
    assert "before/after comparison unavailable for all-reviewers: no fully verified cell under exhaustive" in pr_text


@pytest.mark.parametrize("label", [None, "", "   ", 7, True, ["exhaustive"], "thorough"])
def test_invalid_explicit_review_contract_is_rejected_at_load_direct_evaluation_and_cli(tmp_path, label):
    run = _run(review_contract=label)
    path = _write(tmp_path, [run])
    with pytest.raises(AgentLoopError, match="review_contract"):
        load_frozen_artifacts(path)
    with pytest.raises(AgentLoopError, match="run 0.*review_contract"):
        evaluate_frozen_artifacts({"schema_version": 1, "runs": [run]})
    output = tmp_path / "report.json"
    assert cli_main(["review-evaluation", str(path), "--output", str(output)]) == 1
    assert not output.exists()


def test_review_contract_label_is_normalized_and_not_part_of_run_identity(tmp_path):
    loaded = load_frozen_artifacts(_write(tmp_path, [_run(review_contract="  Exhaustive ")]))
    assert loaded["runs"][0]["review_contract"] == "exhaustive"
    assert loaded["runs"][0]["review_contract_provenance"] is None
    duplicate = [
        _run(review_contract="exhaustive"),
        _run(review_contract="first-finding-permitted"),
    ]
    with pytest.raises(AgentLoopError, match="repeat run ID"):
        load_frozen_artifacts(_write(tmp_path, duplicate, "dup.json"))
    with pytest.raises(AgentLoopError, match="repeat run ID"):
        evaluate_frozen_artifacts({"schema_version": 1, "runs": duplicate})


@pytest.mark.parametrize(
    "provenance",
    ["run-log", [], {"verified": True}, {"source": " ", "verified": True}, {"source": "log", "verified": "yes"}],
)
def test_malformed_review_contract_provenance_is_rejected(tmp_path, provenance):
    run = _run(review_contract="exhaustive", review_contract_provenance=provenance)
    with pytest.raises(AgentLoopError, match="review_contract provenance"):
        load_frozen_artifacts(_write(tmp_path, [run]))
    with pytest.raises(AgentLoopError, match="review_contract provenance"):
        evaluate_frozen_artifacts({"schema_version": 1, "runs": [run]})


def test_label_provenance_cannot_vouch_for_a_defaulted_label(tmp_path):
    run = _run(review_contract_provenance=LABEL_EVIDENCE)
    with pytest.raises(AgentLoopError, match="without an explicit review_contract"):
        load_frozen_artifacts(_write(tmp_path, [run]))
    with pytest.raises(AgentLoopError, match="without an explicit review_contract"):
        evaluate_frozen_artifacts({"schema_version": 1, "runs": [run]})


def test_fixture_runs_are_unlabelled_and_land_in_unavailable_baseline_cells():
    raw = json.loads(FIXTURE_ARTIFACTS.read_text(encoding="utf-8"))
    assert len(raw["runs"]) == 5
    for run in raw["runs"]:
        assert run["run_id"].startswith("historical-")
        assert "review_contract" not in run
        assert "review_contract_provenance" not in run
    report = json.loads(FIXTURE_REPORT.read_text(encoding="utf-8"))
    for flow in FLOWS:
        for policy in FLOW_POLICIES[flow]:
            baseline = _cell(report, "first-finding-permitted", policy, flow)
            assert baseline["run_count"] == 1
            assert _cell(report, "exhaustive", policy, flow)["run_count"] == 0
            for contract in ("first-finding-permitted", "exhaustive"):
                for key in CONTRACT_VALUE_KEYS:
                    assert _cell(report, contract, policy, flow)[key]["status"] == "unavailable"


def _real_run_invariant_violations(runs):
    """Return every documented real-run invariant a raw run list breaks."""
    violations = []
    for index, run in enumerate(runs):
        name = f"run {run.get('run_id', index)!r}"
        if "review_contract" not in run:
            violations.append(f"{name} has no explicit review_contract")
        label = run.get("review_contract_provenance")
        if not isinstance(label, dict) or not str(label.get("source") or "").strip():
            violations.append(f"{name} has no review_contract_provenance source")
        per_metric = run.get("metric_provenance") if isinstance(run.get("metric_provenance"), dict) else {}
        sources = [run.get("provenance"), run.get("label_provenance"), label, *per_metric.values()]
        if any(
            isinstance(entry, dict) and str(entry.get("source") or "").strip().startswith("frozen-fixture:")
            for entry in sources
        ):
            violations.append(f"{name} uses a frozen-fixture: provenance source")
        metrics = run.get("metrics") if isinstance(run.get("metrics"), dict) else {}
        if "escaped_defects" in metrics or "escaped_defects" in run:
            escaped = per_metric.get("escaped_defects")
            if not isinstance(escaped, dict) or not str(escaped.get("source") or "").strip():
                violations.append(f"{name} has no metric_provenance.escaped_defects source")
    return violations


def test_real_run_pair_is_reproducible_and_every_real_run_meets_the_freezing_invariants(tmp_path):
    """Pins no metric literal, so stage-2 data-only additions need no test edit."""
    raw = json.loads(REAL_RUN_ARTIFACTS.read_text(encoding="utf-8"))
    assert _real_run_invariant_violations(raw["runs"]) == []
    fixture_ids = {run["run_id"] for run in json.loads(FIXTURE_ARTIFACTS.read_text(encoding="utf-8"))["runs"]}
    assert not fixture_ids & {run.get("run_id") for run in raw["runs"]}

    report = evaluate_frozen_artifacts(load_frozen_artifacts(REAL_RUN_ARTIFACTS))
    assert report == json.loads(REAL_RUN_REPORT.read_text(encoding="utf-8"))
    assert sum(report["flows"][flow]["run_count"] for flow in FLOWS) == len(raw["runs"])
    output = tmp_path / "report.json"
    assert cli_main(["review-evaluation", str(REAL_RUN_ARTIFACTS), "--output", str(output)]) == 0
    assert output.read_text(encoding="utf-8") == REAL_RUN_REPORT.read_text(encoding="utf-8")

    # A cell with no runs is unavailable; while the artifact is empty that is
    # every cell, and later additions only ever fill their own cells.
    for flow in FLOWS:
        for contract in ("first-finding-permitted", "exhaustive"):
            for policy in FLOW_POLICIES[flow]:
                cell = _cell(report, contract, policy, flow)
                if cell["run_count"] == 0:
                    for key in CONTRACT_VALUE_KEYS:
                        assert cell[key]["status"] == "unavailable"


def test_real_run_invariants_accept_documented_runs_and_reject_shortcuts(tmp_path):
    good = _contract_run(
        "pr-893", "primary-then-panel", "first-finding-permitted", rounds=11, calls=13,
        metric_provenance={
            "escaped_defects": {
                "source": "merged 2026-09-19; window end 2026-10-03; observed 2026-10-04; issues and PRs referencing the merge",
                "verified": True,
            }
        },
    )
    assert _real_run_invariant_violations([good]) == []
    assert _real_run_invariant_violations([]) == []
    load_frozen_artifacts(_write(tmp_path, [good]))

    defaulted = {key: value for key, value in good.items() if key != "review_contract"}
    assert any("explicit review_contract" in item for item in _real_run_invariant_violations([defaulted]))
    unevidenced = {key: value for key, value in good.items() if key != "review_contract_provenance"}
    assert any("review_contract_provenance" in item for item in _real_run_invariant_violations([unevidenced]))
    run_level_only = {key: value for key, value in good.items() if key != "metric_provenance"}
    assert any("escaped_defects source" in item for item in _real_run_invariant_violations([run_level_only]))
    synthetic = dict(good, provenance={"source": "frozen-fixture:historical", "verified": True})
    assert any("frozen-fixture:" in item for item in _real_run_invariant_violations([synthetic]))
    # A run that froze no escaped-defect metric needs no per-metric entry.
    no_escape = dict(run_level_only, metrics={"review_rounds": 2, "reviewer_calls": 3})
    assert _real_run_invariant_violations([no_escape]) == []
