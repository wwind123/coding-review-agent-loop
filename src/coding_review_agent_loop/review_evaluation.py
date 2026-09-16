"""Deterministic, offline comparison of PR review scheduling policies.

The evaluator consumes local frozen run artifacts only.  It never imports the
GitHub client, starts a reviewer, or writes to a repository.  Artifact fields
that are absent or not trustworthy are rendered as ``unavailable`` rather than
being estimated from unrelated measurements.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping

from .errors import AgentLoopError

POLICIES = ("all-reviewers", "selective-intermediate", "primary-then-panel")
SEVERITY_WEIGHTS = {"critical": 5, "high": 4, "medium": 3, "low": 2, "info": 1}
_METRIC_KEYS = (
    "review_rounds",
    "coder_followup_rounds",
    "reviewer_calls",
    "tokens",
    "elapsed_seconds",
    "estimated_cost",
    "ci_failures",
    "escaped_defects",
    "false_positives",
    "withdrawals",
    "disagreements",
    "calls_avoided",
)


def _nonblank_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AgentLoopError(f"Frozen evaluation {label} must be a non-empty string.")
    return value.strip()


def _number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _as_list(value: object, label: str) -> list[object]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise AgentLoopError(f"Frozen evaluation {label} must be an array.")
    return value


def _finding_contributors(finding: Mapping[str, object]) -> tuple[str, ...]:
    raw = finding.get("contributors", finding.get("reviewers", ()))
    if not isinstance(raw, list):
        return ()
    return tuple(dict.fromkeys(name.strip() for name in raw if isinstance(name, str) and name.strip()))


def _finding_id(finding: Mapping[str, object]) -> str | None:
    value = finding.get("finding_id", finding.get("id"))
    return value.strip() if isinstance(value, str) and value.strip() else None


def _metric_value(run: Mapping[str, object], key: str) -> int | float | None:
    metrics = run.get("metrics")
    if isinstance(metrics, dict):
        value = _number(metrics.get(key))
        if value is not None:
            return value
    return _number(run.get(key))


def _derived_avoided_calls(run: Mapping[str, object]) -> int | None:
    """Derive omissions only from complete per-round board snapshots."""
    rounds = run.get("rounds")
    if not isinstance(rounds, list):
        return None
    total = 0
    for round_data in rounds:
        if not isinstance(round_data, dict):
            return None
        required = round_data.get("required_reviewers")
        selected = round_data.get("selected_reviewers")
        if not isinstance(required, list) or not isinstance(selected, list):
            return None
        if any(not isinstance(name, str) for name in [*required, *selected]):
            return None
        total += max(0, len(set(required)) - len(set(selected)))
    return total


def _provenance(value: object, label: str) -> dict[str, object] | None:
    """Return a normalized provenance record or ``None`` when absent.

    A provenance record is trustworthy only when it names a non-empty
    ``source`` and explicitly states ``verified: true``.  Anything else is
    treated as unverified and the dependent measurement is unavailable.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise AgentLoopError(f"Frozen evaluation {label} provenance must be an object or null.")
    source = value.get("source")
    verified = value.get("verified")
    if not isinstance(source, str) or not source.strip():
        raise AgentLoopError(f"Frozen evaluation {label} provenance requires a non-empty source.")
    if not isinstance(verified, bool):
        raise AgentLoopError(f"Frozen evaluation {label} provenance requires a boolean verified flag.")
    return {"source": source.strip(), "verified": verified}


def _is_verified(provenance: Mapping[str, object] | None) -> bool:
    return provenance is not None and provenance.get("verified") is True


def _validate_run(run: object, index: int) -> dict[str, object]:
    if not isinstance(run, dict):
        raise AgentLoopError(f"Frozen evaluation run {index} must be an object.")
    policy = _nonblank_string(run.get("policy"), f"run {index} policy")
    if policy not in POLICIES:
        raise AgentLoopError(f"Frozen evaluation run {index} has unsupported policy {policy!r}.")
    run_id = _nonblank_string(run.get("run_id", str(index + 1)), f"run {index} ID")
    run_provenance = _provenance(run.get("provenance"), f"run {run_id}")
    label_provenance = _provenance(run.get("label_provenance"), f"run {run_id} label")
    raw_metric_provenance = run.get("metric_provenance")
    if raw_metric_provenance is not None and not isinstance(raw_metric_provenance, dict):
        raise AgentLoopError(f"Frozen evaluation run {run_id} metric_provenance must be an object.")
    metric_provenance = {
        str(key): _provenance(value, f"run {run_id} metric {key}")
        for key, value in (raw_metric_provenance or {}).items()
    }
    findings = _as_list(run.get("findings"), f"run {index} findings")
    normalized_findings: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for finding_index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            raise AgentLoopError(f"Frozen evaluation finding {run_id}/{finding_index} must be an object.")
        finding_id = _finding_id(finding)
        if finding_id is None:
            raise AgentLoopError(f"Frozen evaluation finding {run_id}/{finding_index} has no ID.")
        if finding_id in seen_ids:
            raise AgentLoopError(f"Frozen evaluation run {run_id} repeats finding ID {finding_id!r}.")
        seen_ids.add(finding_id)
        valid = finding.get("valid")
        if valid is not None and not isinstance(valid, bool):
            raise AgentLoopError(f"Frozen evaluation finding {run_id}/{finding_id} valid label must be boolean.")
        normalized_findings.append(
            {
                "id": finding_id,
                "contributors": list(_finding_contributors(finding)),
                "severity": finding.get("severity") if isinstance(finding.get("severity"), str) else None,
                # ``None`` means unlabeled: never inferred as valid.
                "valid": valid,
            }
        )
    primary = run.get("primary_reviewer")
    if primary is not None and (not isinstance(primary, str) or not primary.strip()):
        raise AgentLoopError(f"Frozen evaluation run {run_id} primary_reviewer must be a non-empty string or null.")
    if policy == "primary-then-panel" and primary is None:
        raise AgentLoopError(f"Frozen evaluation run {run_id} uses primary-then-panel without a primary_reviewer.")
    metrics = run.get("metrics")
    if metrics is not None and not isinstance(metrics, dict):
        raise AgentLoopError(f"Frozen evaluation run {run_id} metrics must be an object.")
    return {
        "run_id": run_id,
        "policy": policy,
        "primary_reviewer": primary.strip() if isinstance(primary, str) else None,
        "findings": normalized_findings,
        "metrics": dict(metrics) if isinstance(metrics, dict) else {},
        "rounds": run.get("rounds"),
        "primary_approval_round": run.get("primary_approval_round"),
        "panel_approval_round": run.get("panel_approval_round"),
        "provenance": run_provenance,
        "label_provenance": label_provenance,
        "metric_provenance": metric_provenance,
    }


def load_frozen_artifacts(path: str | Path) -> dict[str, object]:
    """Load and validate a local JSON artifact without any external effects."""
    artifact_path = Path(path)
    try:
        raw = artifact_path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AgentLoopError(f"Unable to read frozen evaluation artifacts: {artifact_path}") from exc
    if isinstance(payload, list):
        payload = {"schema_version": 1, "runs": payload}
    if not isinstance(payload, dict):
        raise AgentLoopError("Frozen evaluation artifacts must contain an object or run array.")
    if payload.get("schema_version", 1) != 1:
        raise AgentLoopError("Unsupported frozen evaluation artifact schema version.")
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise AgentLoopError("Frozen evaluation artifacts require a runs array.")
    return {"schema_version": 1, "runs": [_validate_run(run, index) for index, run in enumerate(runs)]}


def _unavailable(reason: str | None = None) -> dict[str, object]:
    row: dict[str, object] = {"value": None, "status": "unavailable"}
    if reason:
        row["reason"] = reason
    return row


def _not_applicable(reason: str) -> dict[str, object]:
    return {"value": None, "status": "not-applicable", "reason": reason}


def _metric_provenance_for(run: Mapping[str, object], key: str) -> Mapping[str, object] | None:
    per_metric = run.get("metric_provenance")
    if isinstance(per_metric, dict) and per_metric.get(key) is not None:
        return per_metric.get(key)
    run_level = run.get("provenance")
    return run_level if isinstance(run_level, dict) else None


def _metric_measurement(run: Mapping[str, object], key: str) -> int | float | None:
    """Return a verified measurement or ``None`` when absent or unproven."""
    if not _is_verified(_metric_provenance_for(run, key)):
        return None
    value = _metric_value(run, key)
    if value is None and key == "calls_avoided":
        value = _derived_avoided_calls(run)
    return value


def evaluate_frozen_artifacts(artifacts: Mapping[str, object]) -> dict[str, object]:
    """Return a stable report for all three policies.

    Every measurement is reported as ``verified`` only when the frozen artifact
    carries trustworthy provenance for it; otherwise it is ``unavailable``.
    Findings are namespaced by run so identical IDs across runs never collide.
    """
    runs = artifacts.get("runs")
    if not isinstance(runs, list):
        raise AgentLoopError("Validated frozen artifacts require a runs array.")
    canonical = json.dumps(dict(artifacts), separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "frozen_review_policy_evaluation",
        "artifact_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "policies": {},
    }
    for policy in POLICIES:
        policy_runs = [run for run in runs if isinstance(run, dict) and run.get("policy") == policy]
        finding_by_reviewer: dict[str, set[tuple[str, str]]] = defaultdict(set)
        weighted_by_reviewer: dict[str, int] = defaultdict(int)
        valid_finding_keys: set[tuple[str, str]] = set()
        primary_covered: set[tuple[str, str]] = set()
        unlabeled_findings = 0
        unverified_label_runs: list[str] = []
        runs_with_primary = 0
        for run in policy_runs:
            run_id = str(run.get("run_id"))
            primary = run.get("primary_reviewer")
            if isinstance(primary, str):
                runs_with_primary += 1
            if not _is_verified(run.get("label_provenance")):
                unverified_label_runs.append(run_id)
            for finding in run.get("findings", []):
                if not isinstance(finding, dict):
                    continue
                valid = finding.get("valid")
                if valid is None:
                    unlabeled_findings += 1
                    continue
                if valid is not True:
                    continue
                finding_id = finding.get("id")
                contributors = tuple(finding.get("contributors", ()))
                if not isinstance(finding_id, str) or not contributors:
                    continue
                key = (run_id, finding_id)
                valid_finding_keys.add(key)
                if isinstance(primary, str) and primary in contributors:
                    primary_covered.add(key)
                weight = SEVERITY_WEIGHTS.get(str(finding.get("severity")).lower(), 0)
                for reviewer in contributors:
                    finding_by_reviewer[reviewer].add(key)
                    if key not in primary_covered:
                        weighted_by_reviewer[reviewer] += weight
        labels_verified = bool(policy_runs) and not unverified_label_runs and unlabeled_findings == 0
        coverage_reason = None
        if not policy_runs:
            coverage_reason = "no frozen runs for this policy"
        elif unverified_label_runs:
            coverage_reason = (
                "finding labels lack verified provenance for runs: "
                + ", ".join(sorted(unverified_label_runs))
            )
        elif unlabeled_findings:
            coverage_reason = f"{unlabeled_findings} finding(s) have no explicit valid label"
        if labels_verified:
            coverage: dict[str, object] = {"value": len(valid_finding_keys), "status": "verified"}
        else:
            coverage = _unavailable(coverage_reason)
            if unlabeled_findings:
                coverage["unlabeled_findings"] = unlabeled_findings
        if not policy_runs or runs_with_primary == 0:
            marginal: dict[str, object] = _not_applicable(
                "no run for this policy declares a primary reviewer"
            )
            weighted: dict[str, object] = _not_applicable(
                "no run for this policy declares a primary reviewer"
            )
        elif runs_with_primary != len(policy_runs):
            marginal = _unavailable("only some runs declare a primary reviewer")
            weighted = _unavailable("only some runs declare a primary reviewer")
        elif not labels_verified:
            marginal = _unavailable(coverage_reason)
            weighted = _unavailable(coverage_reason)
        else:
            marginal = {
                "status": "verified",
                "value": {
                    reviewer: sorted(f"{run_id}:{finding_id}" for run_id, finding_id in keys - primary_covered)
                    for reviewer, keys in sorted(finding_by_reviewer.items())
                    if keys - primary_covered
                },
            }
            weighted = {
                "status": "verified",
                "value": {
                    reviewer: weight
                    for reviewer, weight in sorted(weighted_by_reviewer.items())
                    if weight
                },
            }
        metric_rows: dict[str, object] = {}
        for key in _METRIC_KEYS:
            if not policy_runs:
                metric_rows[key] = _unavailable("no frozen runs for this policy")
                continue
            values = [_metric_measurement(run, key) for run in policy_runs]
            if any(value is None for value in values):
                missing = [
                    str(run.get("run_id"))
                    for run, value in zip(policy_runs, values)
                    if value is None
                ]
                metric_rows[key] = _unavailable(
                    "measurement absent or without verified provenance for runs: "
                    + ", ".join(missing)
                )
            else:
                metric_rows[key] = {"value": sum(values), "status": "verified"}
        # Primary-to-panel regressions are a whole-policy measurement: every
        # primary-bearing run must carry both approval-round endpoints with
        # verified run provenance, otherwise a partial list would make an
        # incomplete dataset look complete.  Missing data is reported as
        # unavailable, naming the affected runs, never silently dropped.
        regressions: list[float | int] = []
        runs_missing_rounds: list[str] = []
        runs_without_provenance: list[str] = []
        for run in policy_runs:
            run_id = str(run.get("run_id"))
            first = _number(run.get("primary_approval_round"))
            last = _number(run.get("panel_approval_round"))
            if first is None or last is None:
                runs_missing_rounds.append(run_id)
                continue
            if not _is_verified(run.get("provenance")):
                runs_without_provenance.append(run_id)
                continue
            regressions.append(last - first)
        if not policy_runs or runs_with_primary == 0:
            regression_row: dict[str, object] = _not_applicable(
                "no run for this policy declares a primary reviewer"
            )
        elif runs_with_primary != len(policy_runs):
            regression_row = _unavailable("only some runs declare a primary reviewer")
        elif runs_missing_rounds or runs_without_provenance:
            reasons: list[str] = []
            if runs_missing_rounds:
                reasons.append(
                    "approval-round measurement absent for runs: "
                    + ", ".join(runs_missing_rounds)
                )
            if runs_without_provenance:
                reasons.append(
                    "approval-round data lacks verified run provenance for runs: "
                    + ", ".join(runs_without_provenance)
                )
            regression_row = _unavailable("; ".join(reasons))
        else:
            regression_row = {"value": regressions, "status": "verified"}
        report["policies"][policy] = {
            "run_count": len(policy_runs),
            "status": "verified" if policy_runs else "unavailable",
            "valid_unique_findings": coverage,
            "marginal_findings_beyond_primary": marginal,
            "severity_weighted_marginal_findings": weighted,
            "primary_to_panel_approval_regressions": regression_row,
            "metrics": metric_rows,
        }
    return report


def render_evaluation_report(report: Mapping[str, object], *, human: bool = False) -> str:
    if not human:
        return json.dumps(dict(report), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    lines = ["Frozen PR review policy evaluation", f"Artifact: {report.get('artifact_sha256', 'unavailable')}"]
    policies = report.get("policies", {})
    for policy in POLICIES:
        row = policies.get(policy, {}) if isinstance(policies, dict) else {}
        lines.append(f"\n{policy}: {row.get('status', 'unavailable')} ({row.get('run_count', 0)} runs)")
        for label, key in (
            ("valid unique findings", "valid_unique_findings"),
            ("marginal findings beyond primary", "marginal_findings_beyond_primary"),
            ("severity-weighted marginal findings", "severity_weighted_marginal_findings"),
            ("primary-to-panel approval regressions", "primary_to_panel_approval_regressions"),
        ):
            cell = row.get(key, _unavailable())
            if isinstance(cell, dict):
                status = cell.get("status", "unavailable")
                value = cell.get("value")
                lines.append(f"  {label}: {value if status == 'verified' else status}")
            else:
                lines.append(f"  {label}: {cell}")
        metrics = row.get("metrics", {})
        if isinstance(metrics, dict):
            for key in _METRIC_KEYS:
                value = metrics.get(key, _unavailable())
                if isinstance(value, dict):
                    status = value.get("status", "unavailable")
                    lines.append(f"  {key}: {value.get('value') if status == 'verified' else status}")
                else:
                    lines.append(f"  {key}: {value}")
    return "\n".join(lines) + "\n"


# Concise aliases for callers and integrations.
evaluate = evaluate_frozen_artifacts
render_report = render_evaluation_report
