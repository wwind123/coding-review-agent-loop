"""Deterministic, offline comparison of review scheduling policies.

The evaluator consumes local frozen run artifacts only.  It never imports the
GitHub client, starts a reviewer, or writes to a repository.  Artifact fields
that are absent or not trustworthy are rendered as ``unavailable`` rather than
being estimated from unrelated measurements.

Runs carry a ``flow`` discriminator (``pr`` or ``plan``).  PR review and issue
plan review share the ``all-reviewers`` and ``primary-then-panel`` policy
names, so aggregation, run-identity uniqueness, and report titling are all per
flow: a planning run can never pool its calls, tokens, latency, findings, or
escaped defects into a PR row, or the reverse.  A run that omits ``flow``
defaults to ``pr``, so PR-only artifacts written before the flow dimension keep
loading and produce byte-identical PR rows.

Runs may also carry a ``review_contract`` label (``first-finding-permitted``,
the historical reviewer contract, or ``exhaustive``) with optional
``review_contract_provenance`` evidence.  The contract comparison is reported
per ``(flow, review_contract, policy)`` cell and is never pooled across
scheduling policies: a policy determines reviewer calls and influences rounds
and escapes, so a pooled cohort would attribute scheduling effects to the
prompt contract.  An absent label defaults to ``first-finding-permitted`` but,
lacking label evidence, never yields a verified cell.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping

from .errors import AgentLoopError

FLOWS = ("pr", "plan")
DEFAULT_FLOW = "pr"
POLICIES = ("all-reviewers", "selective-intermediate", "primary-then-panel")
PLAN_POLICIES = ("all-reviewers", "primary-then-panel")
# Each flow declares the policies it can actually run.  ``selective-intermediate``
# is a PR-only policy and is rejected on a planning run rather than silently
# producing an always-empty planning row.
FLOW_POLICIES: dict[str, tuple[str, ...]] = {"pr": POLICIES, "plan": PLAN_POLICIES}
# Sentinel distinguishing an absent ``flow`` key, which defaults for legacy
# PR-only artifacts, from an explicitly present null value, which does not.
_ABSENT = object()
REVIEW_CONTRACTS = ("first-finding-permitted", "exhaustive")
DEFAULT_REVIEW_CONTRACT = "first-finding-permitted"
# Summed per (flow, review contract, policy) cell, then divided by the cell's
# run count for the ``*_per_run`` values named on the right.
_CONTRACT_METRIC_KEYS = ("review_rounds", "reviewer_calls", "coder_followup_rounds", "escaped_defects")
_CONTRACT_PER_RUN_KEYS = {
    "review_rounds": "review_rounds_per_run",
    "reviewer_calls": "reviewer_calls_per_run",
    "escaped_defects": "escaped_defects_per_run",
}
CONTRACT_COMPARISON_TITLE = "Review contract comparison (within scheduling policy)"
FLOW_TITLES = {
    "pr": "Frozen PR review policy evaluation",
    "plan": "Frozen plan review policy evaluation",
}
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


def _finding_contributors(finding: Mapping[str, object], label: str) -> tuple[str, ...]:
    """Return the deduplicated reviewer identities that contributed a finding.

    Contributor metadata must be a non-empty array of nonblank reviewer
    identities.  Absent, wrong-typed, empty, or partially invalid contributor
    data fails closed: silently normalizing it to an empty tuple would drop a
    valid finding from unique, marginal, and severity-weighted coverage while
    the row still reported ``verified``.
    """
    raw = finding.get("contributors", finding.get("reviewers"))
    if raw is None:
        raise AgentLoopError(f"Frozen evaluation finding {label} has no contributors array.")
    if not isinstance(raw, list):
        raise AgentLoopError(f"Frozen evaluation finding {label} contributors must be an array of reviewer names.")
    names: list[str] = []
    for index, name in enumerate(raw):
        if not isinstance(name, str) or not name.strip():
            raise AgentLoopError(
                f"Frozen evaluation finding {label} contributor {index} must be a non-empty reviewer name."
            )
        names.append(name.strip())
    contributors = tuple(dict.fromkeys(names))
    if not contributors:
        raise AgentLoopError(f"Frozen evaluation finding {label} must name at least one contributor.")
    return contributors


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


def _severity(value: object, label: str) -> str | None:
    """Return a canonical severity label, ``None`` when absent, or fail closed.

    A severity outside ``SEVERITY_WEIGHTS`` is rejected rather than silently
    weighted as zero, which would erase a serious finding from the
    severity-weighted comparison.
    """
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise AgentLoopError(f"Frozen evaluation finding {label} severity must be a non-empty string or null.")
    severity = value.strip().lower()
    if severity not in SEVERITY_WEIGHTS:
        raise AgentLoopError(
            f"Frozen evaluation finding {label} has unsupported severity {value!r}; "
            f"expected one of: {', '.join(SEVERITY_WEIGHTS)}."
        )
    return severity


def _run_flow(value: object, label: str) -> str:
    """Return the validated flow for a run; only an absent key defaults.

    A PR-only artifact written before the flow dimension existed carries no
    ``flow`` key at all, and that is the whole compatibility exception:
    ``_ABSENT`` defaults to ``pr`` so such artifacts keep loading unchanged.
    An explicitly present ``"flow": null`` is a labeled run whose label is
    missing, not a legacy record, so it is rejected rather than assigned to
    the PR rows where a planning run's measurements could contaminate them.
    """
    if value is _ABSENT:
        return DEFAULT_FLOW
    if value is None:
        raise AgentLoopError(
            f"Frozen evaluation {label} has an explicit null flow; omit the field entirely "
            f"to keep the legacy {DEFAULT_FLOW!r} default, or name one of: {', '.join(FLOWS)}."
        )
    flow = _nonblank_string(value, f"{label} flow").lower()
    if flow not in FLOWS:
        raise AgentLoopError(
            f"Frozen evaluation {label} has unsupported flow {flow!r}; "
            f"expected one of: {', '.join(FLOWS)}."
        )
    return flow


def _run_policy(value: object, flow: str, label: str) -> str:
    """Return the run's policy, validated against the policies its flow runs.

    A policy name valid in one flow is not automatically valid in the other:
    ``selective-intermediate`` is PR-only.  Resolving the policy here, rather
    than matching the raw value against a flow's policy list during
    partitioning, keeps a run that names a foreign policy from disappearing
    from every policy row while still being counted in its flow.
    """
    policy = _nonblank_string(value, f"{label} policy")
    if policy not in FLOW_POLICIES[flow]:
        raise AgentLoopError(
            f"Frozen evaluation {label} has unsupported policy {policy!r} for flow {flow!r}; "
            f"expected one of: {', '.join(FLOW_POLICIES[flow])}."
        )
    return policy


def _run_review_contract(value: object, label: str) -> str:
    """Return the validated review contract for a run; only an absent key defaults.

    Mirrors ``_run_flow``: a run frozen before the contract dimension carries
    no ``review_contract`` key and defaults to the historical
    ``first-finding-permitted`` contract.  An explicitly present null, blank,
    non-string, or unknown value is a labeled run whose label is unusable, so
    it is rejected rather than silently assigned to the default cohort.
    """
    if value is _ABSENT:
        return DEFAULT_REVIEW_CONTRACT
    if value is None:
        raise AgentLoopError(
            f"Frozen evaluation {label} has an explicit null review_contract; omit the field "
            f"entirely to keep the legacy {DEFAULT_REVIEW_CONTRACT!r} default, or name one of: "
            f"{', '.join(REVIEW_CONTRACTS)}."
        )
    contract = _nonblank_string(value, f"{label} review_contract").lower()
    if contract not in REVIEW_CONTRACTS:
        raise AgentLoopError(
            f"Frozen evaluation {label} has unsupported review_contract {contract!r}; "
            f"expected one of: {', '.join(REVIEW_CONTRACTS)}."
        )
    return contract


def _run_review_contract_provenance(run: Mapping[str, object], label: str) -> dict[str, object] | None:
    """Return the run's label evidence, rejecting evidence for a defaulted label.

    Provenance attached to a run that names no ``review_contract`` would vouch
    for a label nobody wrote down, so it is rejected; a defaulted label can
    therefore never produce a verified contract cell.
    """
    provenance = _provenance(run.get("review_contract_provenance"), f"{label} review_contract")
    if provenance is not None and "review_contract" not in run:
        raise AgentLoopError(
            f"Frozen evaluation {label} carries review_contract_provenance without an explicit "
            "review_contract label; name the contract the evidence supports."
        )
    return provenance


def _resolved_contracts(
    runs: list[Mapping[str, object]],
) -> list[tuple[str, dict[str, object] | None]]:
    """Pair each run with its validated review contract and label provenance."""
    resolved: list[tuple[str, dict[str, object] | None]] = []
    for index, run in enumerate(runs):
        label = f"run {index}"
        contract = _run_review_contract(run.get("review_contract", _ABSENT), label)
        resolved.append((contract, _run_review_contract_provenance(run, label)))
    return resolved


def _resolved_runs(runs: list[object]) -> list[tuple[Mapping[str, object], str, str]]:
    """Pair each run with its validated flow and policy, failing closed.

    Every path that partitions runs resolves both values here, so a run that
    is not an object, carries an unknown flow, or names a policy foreign to
    that flow can never silently drop out of every report, and a wrong-typed
    or blank flow can never be coerced into the ``pr`` default.  Only an
    absent ``flow`` key defaults; an explicitly present null is rejected.
    """
    resolved: list[tuple[Mapping[str, object], str, str]] = []
    for index, run in enumerate(runs):
        label = f"run {index}"
        if not isinstance(run, dict):
            raise AgentLoopError(f"Frozen evaluation {label} must be an object.")
        flow = _run_flow(run.get("flow", _ABSENT), label)
        resolved.append((run, flow, _run_policy(run.get("policy"), flow, label)))
    return resolved


def _reject_duplicate_runs(runs: list[Mapping[str, object]]) -> None:
    """Fail closed when two records share a flow, policy, and run ID.

    Findings are namespaced by ``(run_id, finding_id)`` within a flow's policy,
    so a repeated run identity would silently collapse distinct findings and
    undercount coverage.  The same run ID under a different flow is a different
    namespace, because PR and planning runs are compared independently.
    """
    seen: set[tuple[str, str, str]] = set()
    for run, flow, policy in _resolved_runs(runs):
        key = (flow, policy, str(run.get("run_id")))
        if key in seen:
            raise AgentLoopError(
                f"Frozen evaluation artifacts repeat run ID {key[2]!r} for policy {key[1]!r} "
                f"in flow {key[0]!r}; run identities must be unique within a flow and policy."
            )
        seen.add(key)


def _validate_run(run: object, index: int) -> dict[str, object]:
    if not isinstance(run, dict):
        raise AgentLoopError(f"Frozen evaluation run {index} must be an object.")
    flow = _run_flow(run.get("flow", _ABSENT), f"run {index}")
    policy = _run_policy(run.get("policy"), flow, f"run {index}")
    run_id = _nonblank_string(run.get("run_id", str(index + 1)), f"run {index} ID")
    review_contract = _run_review_contract(run.get("review_contract", _ABSENT), f"run {run_id}")
    review_contract_provenance = _run_review_contract_provenance(run, f"run {run_id}")
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
                "contributors": list(_finding_contributors(finding, f"{run_id}/{finding_id}")),
                # ``None`` means no severity label; it is never weighted as zero.
                "severity": _severity(finding.get("severity"), f"{run_id}/{finding_id}"),
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
    normalized: dict[str, object] = {
        "run_id": run_id,
        "flow": flow,
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
    # The contract keys are carried through only when the run names them, so a
    # defaulted label stays distinguishable from an explicit one and an
    # artifact frozen before the contract dimension keeps its artifact hash.
    if "review_contract" in run:
        normalized["review_contract"] = review_contract
        normalized["review_contract_provenance"] = review_contract_provenance
    return normalized


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
    validated = [_validate_run(run, index) for index, run in enumerate(runs)]
    _reject_duplicate_runs(validated)
    return {"schema_version": 1, "runs": validated}


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


def _policy_row(policy_runs: list[Mapping[str, object]]) -> dict[str, object]:
    """Return one policy row computed from that policy's runs within one flow.

    The caller partitions runs by flow first, so no measurement, finding, or
    escaped defect from one flow can ever reach another flow's row.
    """
    finding_by_reviewer: dict[str, set[tuple[str, str]]] = defaultdict(set)
    weighted_by_reviewer: dict[str, int] = defaultdict(int)
    valid_finding_keys: set[tuple[str, str]] = set()
    primary_covered: set[tuple[str, str]] = set()
    unlabeled_findings = 0
    unweighted_findings: list[str] = []
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
            finding_id = _finding_id(finding)
            if finding_id is None:
                raise AgentLoopError(f"Frozen evaluation run {run_id} contains a valid finding without an ID.")
            # Re-validate rather than skip: a valid finding must never be
            # dropped from coverage because its contributor data is absent
            # or malformed.
            contributors = _finding_contributors(finding, f"{run_id}/{finding_id}")
            key = (run_id, finding_id)
            valid_finding_keys.add(key)
            if isinstance(primary, str) and primary in contributors:
                primary_covered.add(key)
            severity = _severity(finding.get("severity"), f"{run_id}/{finding_id}")
            if severity is None and key not in primary_covered:
                # Only marginal findings enter the weighted sum; a missing
                # severity on one of them makes the sum unavailable.
                unweighted_findings.append(f"{run_id}:{finding_id}")
            for reviewer in contributors:
                finding_by_reviewer[reviewer].add(key)
                if key not in primary_covered and severity is not None:
                    weighted_by_reviewer[reviewer] += SEVERITY_WEIGHTS[severity]
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
        if unweighted_findings:
            # A valid finding without a severity label cannot be weighted;
            # reporting a partial sum as verified would hide it.
            weighted = _unavailable(
                "valid finding(s) have no severity label: " + ", ".join(sorted(unweighted_findings))
            )
            weighted["unweighted_findings"] = len(unweighted_findings)
        else:
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
    return {
        "run_count": len(policy_runs),
        "status": "verified" if policy_runs else "unavailable",
        "valid_unique_findings": coverage,
        "marginal_findings_beyond_primary": marginal,
        "severity_weighted_marginal_findings": weighted,
        "primary_to_panel_approval_regressions": regression_row,
        "metrics": metric_rows,
    }


def _contract_cell(
    cell_runs: list[tuple[Mapping[str, object], Mapping[str, object] | None]],
) -> dict[str, object]:
    """Return one ``(flow, review contract, policy)`` cell from only its own runs.

    A value is ``verified`` only when every run in the cell has both the
    verified underlying measurement and verified ``review_contract_provenance``.
    A defaulted or unevidenced label makes every value of the cell
    ``unavailable``, naming the runs, so a mislabelled run can never yield a
    verified before/after comparison.  Nothing is estimated and an empty cell
    is never divided.
    """
    cell: dict[str, object] = {"run_count": len(cell_runs)}
    unlabelled = [
        str(run.get("run_id")) for run, provenance in cell_runs if not _is_verified(provenance)
    ]
    for key in _CONTRACT_METRIC_KEYS:
        per_run_key = _CONTRACT_PER_RUN_KEYS.get(key)
        if not cell_runs:
            row = _unavailable("no frozen runs for this review contract and policy")
            per_run = dict(row)
        elif unlabelled:
            row = _unavailable(
                "review contract label absent or without verified provenance for runs: "
                + ", ".join(unlabelled)
            )
            per_run = dict(row)
        else:
            values = [_metric_measurement(run, key) for run, _ in cell_runs]
            missing = [
                str(run.get("run_id")) for (run, _), value in zip(cell_runs, values) if value is None
            ]
            if missing:
                row = _unavailable(
                    "measurement absent or without verified provenance for runs: "
                    + ", ".join(missing)
                )
                per_run = dict(row)
            else:
                total = sum(values)
                row = {"value": total, "status": "verified"}
                per_run = {"value": total / len(cell_runs), "status": "verified"}
        cell[key] = row
        if per_run_key is not None:
            cell[per_run_key] = per_run
    return cell


def evaluate_frozen_artifacts(artifacts: Mapping[str, object]) -> dict[str, object]:
    """Return a stable report for every flow and its policies.

    Aggregation is per flow: PR runs and planning runs are partitioned before
    any policy row is computed, so runs that share a policy name across flows
    are never pooled.  Every measurement is reported as ``verified`` only when
    the frozen artifact carries trustworthy provenance for it; otherwise it is
    ``unavailable``.  Findings are namespaced by run so identical IDs across
    runs never collide.
    """
    runs = artifacts.get("runs")
    if not isinstance(runs, list):
        raise AgentLoopError("Validated frozen artifacts require a runs array.")
    # Resolve and validate every run's shape, flow, and policy once, before
    # duplicate detection and partitioning, so direct evaluation of an
    # unvalidated artifact fails closed exactly as loading does and no run can
    # be dropped from every report or counted in a flow while belonging to
    # none of that flow's policy rows.
    resolved = _resolved_runs(list(runs))
    _reject_duplicate_runs([run for run, _, _ in resolved])
    contracts = _resolved_contracts([run for run, _, _ in resolved])
    canonical = json.dumps(dict(artifacts), separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "frozen_review_policy_evaluation",
        "artifact_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "flows": {},
    }
    for flow in FLOWS:
        flow_runs = [
            (run, run_policy) for run, run_flow, run_policy in resolved if run_flow == flow
        ]
        flow_contract_runs = [
            (run, run_policy, contract, provenance)
            for (run, run_flow, run_policy), (contract, provenance) in zip(resolved, contracts)
            if run_flow == flow
        ]
        report["flows"][flow] = {
            "title": FLOW_TITLES[flow],
            "run_count": len(flow_runs),
            "policies": {
                policy: _policy_row(
                    [run for run, run_policy in flow_runs if run_policy == policy]
                )
                for policy in FLOW_POLICIES[flow]
            },
            # Partitioned by flow, then contract, then policy.  There is
            # deliberately no per-flow or cross-policy contract rollup.
            "review_contracts": {
                contract: {
                    "policies": {
                        policy: _contract_cell(
                            [
                                (run, provenance)
                                for run, run_policy, run_contract, provenance in flow_contract_runs
                                if run_contract == contract and run_policy == policy
                            ]
                        )
                        for policy in FLOW_POLICIES[flow]
                    }
                }
                for contract in REVIEW_CONTRACTS
            },
        }
    # Backward-compatible alias: the PR flow's rows stay reachable at the
    # historical ``policies`` key so PR-only consumers keep working.
    report["policies"] = report["flows"][DEFAULT_FLOW]["policies"]
    return report


def _flow_policy_rows(report: Mapping[str, object], flow: str) -> Mapping[str, object]:
    """Return one flow's policy rows, tolerating a PR-only legacy report."""
    flows = report.get("flows")
    if isinstance(flows, dict):
        row = flows.get(flow)
        policies = row.get("policies") if isinstance(row, dict) else None
        return policies if isinstance(policies, dict) else {}
    if flow != DEFAULT_FLOW:
        return {}
    policies = report.get("policies")
    return policies if isinstance(policies, dict) else {}


def _flow_contract_cells(report: Mapping[str, object], flow: str) -> Mapping[str, object]:
    """Return one flow's review-contract section, tolerating older reports."""
    flows = report.get("flows")
    row = flows.get(flow) if isinstance(flows, dict) else None
    contracts = row.get("review_contracts") if isinstance(row, dict) else None
    return contracts if isinstance(contracts, dict) else {}


def _contract_cell_value(cell: object, key: str) -> tuple[str, bool]:
    row = cell.get(key) if isinstance(cell, dict) else None
    if isinstance(row, dict) and row.get("status") == "verified":
        return str(row.get("value")), True
    status = row.get("status", "unavailable") if isinstance(row, dict) else "unavailable"
    return str(status), False


def _render_contract_comparison(report: Mapping[str, object], flow: str) -> list[str]:
    """Render both contracts side by side, one scheduling policy at a time."""
    contracts = _flow_contract_cells(report, flow)
    lines = [f"\n{CONTRACT_COMPARISON_TITLE}"]
    comparison_keys = (*_CONTRACT_PER_RUN_KEYS.values(),)
    for policy in FLOW_POLICIES[flow]:
        cells = {}
        for contract in REVIEW_CONTRACTS:
            section = contracts.get(contract)
            policies = section.get("policies") if isinstance(section, dict) else None
            cells[contract] = policies.get(policy, {}) if isinstance(policies, dict) else {}
        counts = ", ".join(
            f"{contract} {cells[contract].get('run_count', 0)} runs" for contract in REVIEW_CONTRACTS
        )
        lines.append(f"  {policy} ({counts})")
        lacking: list[str] = []
        for key in comparison_keys:
            rendered = []
            for contract in REVIEW_CONTRACTS:
                text, verified = _contract_cell_value(cells[contract], key)
                rendered.append(f"{contract}={text}")
                if not verified and contract not in lacking:
                    lacking.append(contract)
            lines.append(f"    {key}: " + " | ".join(rendered))
        if lacking:
            lines.append(
                f"    before/after comparison unavailable for {policy}: no fully verified cell under "
                + ", ".join(lacking)
            )
    return lines


def render_evaluation_report(report: Mapping[str, object], *, human: bool = False) -> str:
    if not human:
        return json.dumps(dict(report), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    lines: list[str] = []
    for flow in FLOWS:
        policies = _flow_policy_rows(report, flow)
        # Each flow is titled and aggregated separately so a planning row is
        # never read as a PR row.
        lines.append(FLOW_TITLES[flow])
        lines.append(f"Artifact: {report.get('artifact_sha256', 'unavailable')}")
        for policy in FLOW_POLICIES[flow]:
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
        lines.extend(_render_contract_comparison(report, flow))
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


# Concise aliases for callers and integrations.
evaluate = evaluate_frozen_artifacts
render_report = render_evaluation_report
