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


def _validate_run(run: object, index: int) -> dict[str, object]:
    if not isinstance(run, dict):
        raise AgentLoopError(f"Frozen evaluation run {index} must be an object.")
    policy = _nonblank_string(run.get("policy"), f"run {index} policy")
    if policy not in POLICIES:
        raise AgentLoopError(f"Frozen evaluation run {index} has unsupported policy {policy!r}.")
    run_id = _nonblank_string(run.get("run_id", str(index + 1)), f"run {index} ID")
    findings = _as_list(run.get("findings"), f"run {index} findings")
    normalized_findings: list[dict[str, object]] = []
    for finding_index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            raise AgentLoopError(f"Frozen evaluation finding {run_id}/{finding_index} must be an object.")
        finding_id = _finding_id(finding)
        if finding_id is None:
            raise AgentLoopError(f"Frozen evaluation finding {run_id}/{finding_index} has no ID.")
        normalized_findings.append(
            {
                "id": finding_id,
                "contributors": list(_finding_contributors(finding)),
                "severity": finding.get("severity") if isinstance(finding.get("severity"), str) else None,
                "valid": finding.get("valid", True) is True,
            }
        )
    primary = run.get("primary_reviewer")
    if primary is not None and not isinstance(primary, str):
        raise AgentLoopError(f"Frozen evaluation run {run_id} primary_reviewer must be a string or null.")
    return {
        "run_id": run_id,
        "policy": policy,
        "primary_reviewer": primary.strip() if isinstance(primary, str) else None,
        "findings": normalized_findings,
        "metrics": dict(run.get("metrics")) if isinstance(run.get("metrics"), dict) else {},
        "rounds": run.get("rounds"),
        "primary_approval_round": run.get("primary_approval_round"),
        "panel_approval_round": run.get("panel_approval_round"),
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


def _unavailable() -> dict[str, object]:
    return {"value": None, "status": "unavailable"}


def evaluate_frozen_artifacts(artifacts: Mapping[str, object]) -> dict[str, object]:
    """Return a stable report for all three policies."""
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
        finding_by_reviewer: dict[str, set[str]] = defaultdict(set)
        weighted_by_reviewer: dict[str, int] = defaultdict(int)
        valid_finding_ids: set[str] = set()
        primary_marginal: set[str] = set()
        for run in policy_runs:
            primary = run.get("primary_reviewer")
            for finding in run.get("findings", []):
                if not isinstance(finding, dict) or finding.get("valid", True) is not True:
                    continue
                finding_id = finding.get("id")
                contributors = tuple(finding.get("contributors", ()))
                if not isinstance(finding_id, str) or not contributors:
                    continue
                valid_finding_ids.add(finding_id)
                if isinstance(primary, str) and primary in contributors:
                    primary_marginal.add(finding_id)
                for reviewer in contributors:
                    finding_by_reviewer[reviewer].add(finding_id)
                    weight = finding.get("severity")
                    if finding_id not in primary_marginal:
                        weighted_by_reviewer[reviewer] += SEVERITY_WEIGHTS.get(str(weight).lower(), 0)
        marginal = {
            reviewer: sorted(ids - primary_marginal)
            for reviewer, ids in sorted(finding_by_reviewer.items())
            if ids - primary_marginal
        }
        metric_rows: dict[str, object] = {}
        for key in _METRIC_KEYS:
            values = [_metric_value(run, key) for run in policy_runs]
            if key == "calls_avoided" and not any(value is not None for value in values):
                values = [_derived_avoided_calls(run) for run in policy_runs]
            if not values or any(value is None for value in values):
                metric_rows[key] = _unavailable()
            else:
                metric_rows[key] = {"value": sum(values), "status": "verified"}
        regressions: list[float | int] = []
        for run in policy_runs:
            first = _number(run.get("primary_approval_round"))
            last = _number(run.get("panel_approval_round"))
            if first is not None and last is not None:
                regressions.append(last - first)
        report["policies"][policy] = {
            "run_count": len(policy_runs),
            "status": "verified" if policy_runs else "unavailable",
            "valid_unique_findings": len(valid_finding_ids),
            "marginal_findings_beyond_primary": marginal,
            "severity_weighted_marginal_findings": {
                reviewer: weight
                for reviewer, weight in sorted(weighted_by_reviewer.items())
                if weight
            },
            "primary_to_panel_approval_regressions": (
                {"value": regressions, "status": "verified"}
                if regressions else _unavailable()
            ),
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
        lines.append(f"  valid unique findings: {row.get('valid_unique_findings', 'unavailable')}")
        lines.append(f"  severity-weighted marginal findings: {row.get('severity_weighted_marginal_findings', {})}")
        metrics = row.get("metrics", {})
        if isinstance(metrics, dict):
            for key in _METRIC_KEYS:
                value = metrics.get(key, _unavailable())
                lines.append(f"  {key}: {value.get('value') if isinstance(value, dict) else value}")
    return "\n".join(lines) + "\n"


# Concise aliases for callers and integrations.
evaluate = evaluate_frozen_artifacts
render_report = render_evaluation_report
