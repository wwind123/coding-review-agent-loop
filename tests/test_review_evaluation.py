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
    assert plan["primary-then-panel"]["severity_weighted_marginal_findings"] == {
        "status": "verified", "value": {"Anthropic Claude": 3},
    }
    assert plan["primary-then-panel"]["primary_to_panel_approval_regressions"] == {
        "value": [1], "status": "verified",
    }

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
