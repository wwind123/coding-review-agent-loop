"""Managed-CI protection assessment and explicit waiver reconciliation."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import replace
from typing import Literal
from .config import AgentLoopConfig
from .runner import Runner
from .managed_ci_status import FINAL_CONTEXT
from .managed_ci_types import (
    ManagedCiProbeContext,
    ProtectionAssessment,
    _HIDDEN_BYPASS_DETAIL,
    _OVERRIDE_PROTECTION_STATES,
    _UNPROTECTED_WAIVER,
    _UNREADABLE_PROTECTION_DETAIL,
    _UNREADABLE_WAIVER,
)
from .managed_ci_api import (
    _http_status,
    _is_http_error,
    _is_plan_limited_error,
    _probe_json,
    _probe_json_list,
)


def waivable_protection_states(config: AgentLoopConfig) -> frozenset[str]:
    """Return the non-strict protection states this invocation explicitly waives."""
    if not config.allow_unprotected_managed_ci:
        return frozenset()
    if config.allow_unreadable_protection:
        return _OVERRIDE_PROTECTION_STATES
    return frozenset({"voluntary", "plan_limited"})


def waiver_flags_for_protection(state: str) -> str:
    """Render the CLI flags needed to waive ``state`` explicitly."""
    if state == "unreadable":
        return "--allow-unprotected-managed-ci --allow-unreadable-protection"
    return "--allow-unprotected-managed-ci"


def _waiver_for_protection(state: str) -> str | None:
    """Map a persisted protection state to its persisted waiver value."""
    if state in {"voluntary", "plan_limited"}:
        return _UNPROTECTED_WAIVER
    if state == "unreadable":
        return _UNREADABLE_WAIVER
    return None


def reconcile_live_protection(
    persisted: str | None, live: ProtectionAssessment, *, config: AgentLoopConfig
) -> ProtectionAssessment:
    """Continue a persisted ``unreadable`` PR on a host that reads ``voluntary`` (#1063).

    ``unreadable`` already waives an unknown, non-strict base, so a live
    ``voluntary`` reading (this host can see more, and it is still not strict)
    is covered by the same explicit waivers.  The PR keeps its persisted
    state so every authorization and resume record stays consistent.  Every
    other disagreement, notably a base that became strict, is returned as-is
    and still refuses.
    """
    if (
        persisted == "unreadable"
        and live.state == "voluntary"
        and "unreadable" in waivable_protection_states(config)
    ):
        return replace(
            live,
            state="unreadable",
            detail=f"persisted unreadable; live assessment is voluntary ({live.detail})",
        )
    return live


def _has_final_context(payload: dict[str, object]) -> bool:
    contexts = payload.get("contexts")
    checks = payload.get("checks")
    return (
        isinstance(contexts, list) and FINAL_CONTEXT in contexts
    ) or (
        isinstance(checks, list)
        and any(isinstance(check, dict) and check.get("context") == FINAL_CONTEXT for check in checks)
    )


_RULESET_ENFORCEMENT_VALUES = frozenset({"active", "evaluate", "disabled"})


_BYPASS_ACTOR_TYPES = frozenset({
    "Integration", "OrganizationAdmin", "RepositoryRole", "Team", "DeployKey",
})


_BYPASS_ACTOR_TYPES_WITHOUT_ID = frozenset({"OrganizationAdmin", "DeployKey"})


_BYPASS_MODES = frozenset({"always", "pull_request"})


def _ruleset_detail_problem(ruleset: Mapping[str, object]) -> str | None:
    """Return why a decoded ruleset detail is malformed, or ``None``.

    This is an allowlist: anything it does not recognize is malformed, so an
    unexpected shape can never be read as "no enforcement".
    """
    rules = ruleset.get("rules")
    if not isinstance(rules, list):
        return "'rules' is not a list"
    for rule in rules:
        if not isinstance(rule, dict) or not isinstance(rule.get("type"), str):
            return "a rule entry has no string 'type'"
        if rule["type"] == "required_status_checks":
            parameters = rule.get("parameters")
            checks = parameters.get("required_status_checks") if isinstance(parameters, dict) else None
            if not isinstance(checks, list) or any(
                not isinstance(check, dict) or not isinstance(check.get("context"), str)
                for check in checks
            ):
                return "a required_status_checks rule has malformed 'required_status_checks'"
    # Every set membership test first requires a string: JSON lists and
    # objects are unhashable and must be reported as malformed, not raise.
    enforcement = ruleset.get("enforcement")
    if not isinstance(enforcement, str) or enforcement not in _RULESET_ENFORCEMENT_VALUES:
        return "'enforcement' is not a known value"
    if "bypass_actors" in ruleset:
        actors = ruleset["bypass_actors"]
        if not isinstance(actors, list):
            return "'bypass_actors' is not a list"
        for actor in actors:
            if not isinstance(actor, dict):
                return "a bypass actor is not an object"
            actor_type = actor.get("actor_type")
            actor_id = actor.get("actor_id")
            if not isinstance(actor_type, str) or actor_type not in _BYPASS_ACTOR_TYPES:
                return "a bypass actor has an unknown 'actor_type'"
            id_valid = (
                isinstance(actor_id, int) and not isinstance(actor_id, bool) and actor_id > 0
            ) or (actor_id is None and actor_type in _BYPASS_ACTOR_TYPES_WITHOUT_ID)
            if not id_valid:
                return "a bypass actor has an invalid 'actor_id'"
            if "bypass_mode" in actor and (
                not isinstance(actor["bypass_mode"], str)
                or actor["bypass_mode"] not in _BYPASS_MODES
            ):
                return "a bypass actor has an unknown 'bypass_mode'"
    return None


def _ruleset_detail_well_formed(ruleset: Mapping[str, object]) -> bool:
    return _ruleset_detail_problem(ruleset) is None


def _ruleset_bypass_state(ruleset: Mapping[str, object]) -> Literal["none", "present", "unknown"]:
    """Whether the ruleset's bypass actors are visibly empty, present, or hidden.

    GitHub returns ``bypass_actors`` only to callers with write/admin access to
    the ruleset, so an absent key does not prove the ruleset is unbypassable.
    """
    if "bypass_actors" not in ruleset:
        return "unknown"
    actors = ruleset["bypass_actors"]
    if isinstance(actors, list) and not actors:
        return "none"
    return "present"


def _ruleset_requires_final_context(rules: object) -> bool:
    """Exact match on a required_status_checks context; no substring search."""
    if not isinstance(rules, list):
        return False
    for rule in rules:
        if not isinstance(rule, dict) or rule.get("type") != "required_status_checks":
            continue
        parameters = rule.get("parameters")
        checks = parameters.get("required_status_checks") if isinstance(parameters, dict) else None
        if isinstance(checks, list) and any(
            isinstance(check, dict) and check.get("context") == FINAL_CONTEXT for check in checks
        ):
            return True
    return False


def assess_exact_head_protection(
    runner: Runner, *, context: ManagedCiProbeContext, base: str
) -> ProtectionAssessment:
    """Classify GitHub enforcement conservatively.

    A required context alone is voluntary when administrators can bypass it.
    Rulesets are inspected as an additional independent enforcement source;
    evaluate-mode and bypass actors are deliberately not strict.

    When the classic protection endpoint refuses this token with a strict
    HTTP 403 (#1040), protection is indeterminate rather than absent.  It is
    classified as the distinct ``unreadable`` state only when the effective
    rules were fully inspected and show no strict enforcement; after such a
    403 a ruleset proves strict only when its bypass actors are visible and
    empty.  Every other failure on that path stays ``indeterminate``.
    """
    required, required_result = _probe_json(
        runner, context, f"repos/{context.repo}/branches/{base}/protection/required_status_checks"
    )
    classic_state: Literal["voluntary", "plan_limited", "indeterminate"] | None = None
    classic_detail = "final-ci/exact-head is not independently required"
    classic_forbidden = False
    if required is None:
        if _is_plan_limited_error(required_result):
            classic_state = "plan_limited"
            classic_detail = "GitHub plan/API does not permit branch protection"
        elif _http_status(required_result) == 403:
            # Checked before the legacy 404 text match, so a refused read
            # whose body mentions 404 is never mistaken for "not configured".
            classic_forbidden = True
            classic_state = "indeterminate"
            classic_detail = (
                "required-status protection could not be inspected "
                f"(repos/{context.repo}/branches/{base}/protection/required_status_checks: HTTP 403)"
            )
        elif _is_http_error(required_result, 404):
            classic_state = "voluntary"
            classic_detail = "required status protection is not configured"
        else:
            classic_state = "indeterminate"
            classic_detail = "required-status protection could not be inspected"
    elif _has_final_context(required):
        admins, admins_result = _probe_json(
            runner, context, f"repos/{context.repo}/branches/{base}/protection/enforce_admins"
        )
        if admins is not None and admins.get("enabled") is True:
            return ProtectionAssessment("strict", "classic", "final-ci/exact-head is required and admins are enforced")
        if admins is not None and admins.get("enabled") is False:
            classic_state = "voluntary"
            classic_detail = "administrators can bypass required status checks"
        elif admins_result.returncode != 0:
            classic_state = "indeterminate"
            classic_detail = "admin enforcement could not be inspected"
            if _http_status(admins_result) == 403:
                classic_forbidden = True
        else:
            classic_state = "indeterminate"
            classic_detail = "admin enforcement response was malformed"

    rules, rules_result = _probe_json_list(runner, context, f"repos/{context.repo}/rules/branches/{base}")
    if rules is None:
        # A private-Free repository can reject both protection APIs.  The
        # ruleset probe cannot turn an already known plan restriction into an
        # unknown result, so retain the actionable classification.
        if classic_state == "plan_limited":
            return ProtectionAssessment("plan_limited", "classic", classic_detail)
        if classic_forbidden:
            # Only gh's own strict 404/422 status proves "no rules" here.
            if _http_status(rules_result) in {404, 422}:
                return ProtectionAssessment("unreadable", "classic", _UNREADABLE_PROTECTION_DETAIL)
            return ProtectionAssessment("indeterminate", "rulesets", "effective branch rules could not be inspected")
        if _is_http_error(rules_result, 404, 422):
            return ProtectionAssessment(classic_state or "voluntary", "classic" if classic_state else "none", classic_detail)
        return ProtectionAssessment("indeterminate", "rulesets", "effective branch rules could not be inspected")
    voluntary = False
    hidden_bypass = False
    rules_fully_inspected = True
    malformed_detail: str | None = None
    for entry in rules:
        if not isinstance(entry, dict):
            rules_fully_inspected = False
            malformed_detail = malformed_detail or "an effective rule entry is not an object"
            continue
        ruleset_id = entry.get("ruleset_id") or entry.get("id")
        if not isinstance(ruleset_id, int):
            rules_fully_inspected = False
            malformed_detail = malformed_detail or "an effective rule entry has no integer ruleset id"
            continue
        ruleset, ignored = _probe_json(runner, context, f"repos/{context.repo}/rulesets/{ruleset_id}")
        if ruleset is None:
            return ProtectionAssessment("indeterminate", "rulesets", "an applicable ruleset could not be inspected")
        problem = _ruleset_detail_problem(ruleset)
        if problem is not None:
            rules_fully_inspected = False
            malformed_detail = malformed_detail or f"ruleset {ruleset_id}: {problem}"
            if classic_forbidden:
                continue
        if not _ruleset_requires_final_context(ruleset.get("rules")):
            continue
        if classic_forbidden:
            bypass_state = _ruleset_bypass_state(ruleset)
            non_bypassable = bypass_state == "none"
        else:
            bypass_state = None
            non_bypassable = not ruleset.get("bypass_actors")
        if ruleset.get("enforcement") != "active" or not non_bypassable:
            voluntary = True
            if ruleset.get("enforcement") == "active" and bypass_state == "unknown":
                hidden_bypass = True
            continue
        return ProtectionAssessment("strict", "ruleset", "active ruleset requires final-ci/exact-head without bypass actors")
    if classic_forbidden:
        if not rules_fully_inspected:
            return ProtectionAssessment(
                "indeterminate", "rulesets",
                f"effective branch rules were malformed ({malformed_detail}); "
                "applicable enforcement could not be inspected",
            )
        return ProtectionAssessment(
            "unreadable", "classic",
            _UNREADABLE_PROTECTION_DETAIL + (_HIDDEN_BYPASS_DETAIL if hidden_bypass else ""),
        )
    if classic_state == "indeterminate":
        return ProtectionAssessment("indeterminate", "classic", classic_detail)
    if classic_state == "plan_limited":
        return ProtectionAssessment("plan_limited", "classic", classic_detail)
    return ProtectionAssessment(
        "voluntary", "rulesets" if voluntary else ("classic" if classic_state else "none"),
        "matching ruleset is bypassable/evaluate-mode" if voluntary else classic_detail,
    )
