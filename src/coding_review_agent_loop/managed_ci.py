"""External driver for repositories that implement managed exact-head CI."""

from __future__ import annotations

import json
import re
import secrets
import shlex
import time
from datetime import datetime
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from .ci_health import (
    CiInfrastructureStall,
    PullRequestCheck,
    PullRequestChecks,
    is_wholly_infrastructure_blocked,
)
from .config import AgentLoopConfig
from .errors import AgentLoopError
from .github import (
    PullRequestMergeability,
    PullRequestMetadata,
    get_pr_checks,
    get_pr_head_sha,
    get_pr_mergeability,
    parse_strong_issue_reference_evidence,
)
from .logging import log
from .runner import Runner
from .workdirs import active_workdir
from .protocol_markers import (
    PR_BODY_SURFACE,
    PR_COMMENT_SURFACE,
    TrustedBody,
    scan_reserved_markers,
)


MANAGED_LABEL = "agent-loop-managed"
MANAGED_OPT_OUT_LABEL = "agent-loop-managed-opt-out"
QUALIFIED_LABEL = "agent-loop-exact-head-qualified"
QUALIFICATION_MARKER = "AGENT_LOOP_MANAGED_CI_QUALIFIED_V2"
FINAL_CONTEXT = "final-ci/exact-head"
READINESS_CONTEXT = "agent-loop/round-readiness"
WORKFLOW_FILE = "ci.yml"
_CONTRACT_MARKERS = (MANAGED_LABEL, FINAL_CONTEXT, "expected_head_sha")
V2_MARKER = "AGENT_LOOP_MANAGED_CI_V2"
INTENT_MARKER = "AGENT_MANAGED_CI_INTENT_V2"
V2_FEATURE_MARKERS = (V2_MARKER, "workflow_dispatch", "managed_nonce", FINAL_CONTEXT)
# Adoption is deliberately not a v2 feature marker: repositories which have
# deployed the safe issue-created draft flow do not implicitly opt in to
# suppressing CI for existing PRs.
V2_ADOPTION_MARKER = "AGENT_LOOP_MANAGED_CI_V2_PR_ADOPTION"
V2_ADOPTION_FEATURE_MARKERS = (V2_ADOPTION_MARKER,)
RECOVERY_MARKER = "AGENT_LOOP_MANAGED_CI_UNLABELED_RECOVERY_V1"
UNPROTECTED_OVERRIDE_TRAILER = "AGENT_MANAGED_CI_UNPROTECTED_OVERRIDE_V1"
_TERMINAL_CI_STATUSES = frozenset({
    "success", "failure", "error", "cancelled", "timed_out",
    "action_required", "startup_failure", "stale",
})


@dataclass(frozen=True)
class ManagedCiProbeContext:
    """The deliberately small, read-only context used by standalone preflight."""

    repo: str
    gh_cmd: str
    cwd: Path


@dataclass(frozen=True)
class ProtectionAssessment:
    """Whether GitHub, rather than this process, enforces the final context."""

    state: Literal["strict", "voluntary", "plan_limited", "indeterminate"]
    source: str
    detail: str


@dataclass(frozen=True)
class ManagedCiReadiness:
    """Stable, non-mutating readiness result consumed by CLI and runtime gates."""

    state: Literal["strict_ready", "override_eligible", "ordinary_fallback", "invalid", "indeterminate"]
    repo_visibility: str | None
    actor: str | None
    advertised_actor: str | None
    workflow_v2: bool
    recovery_capable: bool
    protection: ProtectionAssessment
    reasons: tuple[str, ...]
    remediation: tuple[str, ...]
    # The evaluator resolves the repository default before it probes the
    # workflow.  Keep that value so preflight reports the branch it assessed.
    base: str | None = None


PREFLIGHT_STRICT_READY = 0
PREFLIGHT_KNOWN_NOT_READY = 10
PREFLIGHT_INDETERMINATE = 11


@dataclass
class ManagedCiContract:
    workflow_file: str = WORKFLOW_FILE
    protocol_version: int = 1
    base_ref: str | None = None
    trusted_actor_login: str | None = None
    trusted_actor_id: int | None = None
    workflow_revision: str | None = None
    intent_comment_id: int | None = None
    nonce: str | None = None
    attached_run_id: int | None = None
    run_attempt: int | None = None
    pr_number: int | None = None
    expected_head_sha: str | None = None
    repository: str | None = None
    created_at: int | None = None
    adopted_existing_pr: bool = False
    issue_created_pr: bool = False
    guard_head_sha: str | None = None
    active_label_event_id: int | None = None
    invocation_applied_label: bool = False
    protection_mode: str | None = None
    audit_nonce: str | None = None
    audit_comment_id: int | None = None
    # A resumed invocation always gets a new generation. Historical intent
    # comments remain diagnostic history and are never adopted as this run's
    # dispatch authorization.
    intent_generation: str | None = None
    intent_state: str | None = None
    terminal_run_id: int | None = None
    terminal_run_attempt: int | None = None
    terminal_attempts: tuple[tuple[int, int | None], ...] = ()
    activation_path: Literal["managed", "ordinary_fallback"] = "managed"
    ordinary_recovery: "OrdinaryRecoveryCapability | None" = None
    # Derived from the exact base workflow at activation. Retain it for a
    # delayed dispatch-time fallback so a suppressed draft is released only
    # when that same workflow advertises an unlabeled CI route.
    ordinary_recovery_capable: bool = False
    # Explicit lifecycle provenance.  These fields are intentionally not
    # inferred from public mode flags after activation.
    origin: Literal["issue-created", "source-managed"] | None = None
    lifecycle: Literal[
        "creation", "draft-labeled", "draft-unlabeled-reentry", "ready-unlabeled-reentry"
    ] | None = None
    authenticated_resume: "AuthenticatedManagedResume | None" = None


@dataclass(frozen=True)
class ManagedCiCreationIntent:
    """Instructions that make the opened event atomically recognizable as v2."""

    branch: str
    trusted_actor: str
    protection_mode: str = "strict"
    audit_nonce: str | None = None


@dataclass(frozen=True)
class ManagedCiOverrideRecord:
    """Canonical override record parsed from one trusted PR surface.

    The same marker has deliberately different schemas on a PR body and on
    an actor-authored audit comment.  Keeping that distinction here prevents
    one consumer from accepting a permissive form that another rejects.
    """

    nonce: str
    fields: tuple[tuple[str, str], ...]

    def field_map(self) -> dict[str, str]:
        return dict(self.fields)


@dataclass(frozen=True)
class AuthenticatedIssueCreatedHandoff:
    """Read-only proof for the issue-created draft -> PR-loop transition."""

    pr_number: int
    issue_number: int
    repository: str
    base_ref: str
    head_sha: str
    branch: str
    trusted_actor_login: str
    trusted_actor_id: int
    protection_mode: str
    override_nonce: str | None
    active_label_event_id: int | None = None
    lifecycle: Literal[
        "draft-labeled", "draft-unlabeled-reentry", "ready-unlabeled-reentry"
    ] = "draft-labeled"


@dataclass(frozen=True)
class AuthenticatedManagedResume:
    """Read-only authority for a managed-CI resume.

    A resume is distinct from a just-created handoff.  The latter authorizes
    the first activation only; this record is minted by re-reading a durable
    PR and carries the lifecycle state that was actually authenticated.
    """

    origin: Literal["issue-created", "source-managed"]
    lifecycle: Literal[
        "draft-labeled", "draft-unlabeled-reentry", "ready-unlabeled-reentry"
    ]
    issue_created_handoff: AuthenticatedIssueCreatedHandoff | None = None
    source_branch: str | None = None
    source_sha: str | None = None
    managed_branch: str | None = None
    override_nonce: str | None = None


def parse_managed_ci_override_record(
    body: str,
    *,
    surface: str,
    schema: Literal["body", "audit"],
    required: bool = False,
    expected_nonce: str | None = None,
    additional_allowed_tokens: frozenset[str] = frozenset(),
) -> ManagedCiOverrideRecord | None:
    """Parse one canonical managed-CI override record and reject every variant.

    ``TrustedBody`` supplies the registry/surface/canonical-wire checks.  This
    helper owns the inner field schema so body records cannot quietly acquire
    audit provenance fields and audit comments cannot omit their provenance.
    """
    occurrences = scan_reserved_markers(body)
    override = [
        occurrence for occurrence in occurrences
        if occurrence.definition.token == UNPROTECTED_OVERRIDE_TRAILER
    ]
    allowed = set(additional_allowed_tokens)
    allowed.add(UNPROTECTED_OVERRIDE_TRAILER)
    unexpected = [
        occurrence.definition.token for occurrence in occurrences
        if occurrence.definition.token not in allowed
    ]
    if unexpected:
        raise AgentLoopError(
            "Managed-CI override carrier contains unrelated reserved protocol record(s): "
            + ", ".join(sorted(set(unexpected)))
        )
    if not override:
        if required:
            raise AgentLoopError("Managed-CI override record is required for this handoff.")
        # Still canonicalize allowed companion records, including malformed
        # bare-token fallbacks, before returning absence.
        if occurrences:
            TrustedBody.canonical(
                body,
                surface=surface,
                expected_tokens={item.definition.token for item in occurrences},
            )
        return None
    if len(override) != 1:
        raise AgentLoopError("Managed-CI override carrier contains duplicate override records.")
    TrustedBody.canonical(
        body,
        surface=surface,
        expected_tokens={item.definition.token for item in occurrences},
    )
    parts = override[0].text.strip().split()
    if not parts or parts[0] != UNPROTECTED_OVERRIDE_TRAILER:
        raise AgentLoopError("Managed-CI override record has an invalid schema.")
    fields: list[tuple[str, str]] = []
    for token in parts[1:]:
        key, separator, value = token.partition("=")
        if not separator or not key or not value or any(key == old for old, _ in fields):
            raise AgentLoopError("Managed-CI override record has an invalid schema.")
        fields.append((key, value))
    values = dict(fields)
    body_keys = {"nonce"}
    audit_keys = {
        "nonce", "repo", "base", "head", "protection", "active_label_event_id",
        "resume_from", "provenance_head", "generation",
    }
    if schema == "body":
        if set(values) != body_keys:
            raise AgentLoopError("Managed-CI PR-body override record must contain only its nonce.")
    else:
        if not {"nonce", "repo", "base", "head", "protection"}.issubset(values) or set(values) - audit_keys:
            raise AgentLoopError("Managed-CI override audit record has an invalid provenance schema.")
    nonce = values.get("nonce")
    if not nonce:
        raise AgentLoopError("Managed-CI override record has no nonce.")
    if expected_nonce is not None and nonce != expected_nonce:
        raise AgentLoopError("Managed-CI override record does not match this creation handoff.")
    return ManagedCiOverrideRecord(nonce=nonce, fields=tuple(fields))


@dataclass(frozen=True)
class OrdinaryRecoveryCapability:
    """Invocation-owned proof that this run deliberately released managed CI."""

    pr_number: int
    repository: str
    base_ref: str
    expected_head_sha: str
    released_label_event_id: int | None
    released_at: int
    prior_run_ids: frozenset[int] = frozenset()


@dataclass(frozen=True)
class ManagedCiOutcome:
    status: Literal[
        "passed",
        "failed",
        "timeout",
        "head_changed",
        "merge_conflict",
        "infrastructure_stall",
        "terminal_without_status",
        "not_started",
    ]
    checks: PullRequestChecks | None = None
    mergeability: PullRequestMergeability | None = None
    head_sha: str | None = None
    stall: CiInfrastructureStall | None = None
    failure_details: tuple[str, ...] = ()
    run_id: int | None = None
    run_attempt: int | None = None
    workflow_status: str | None = None
    workflow_conclusion: str | None = None


@dataclass(frozen=True)
class ManagedCiRunSnapshot:
    run_id: int
    run_attempt: int | None
    status: str | None
    conclusion: str | None


def activate_managed_ci(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    metadata: PullRequestMetadata,
    managed_resume: AuthenticatedManagedResume | None = None,
    resume_origin: Literal["issue-created", "source-managed"] | None = None,
) -> ManagedCiContract | None:
    """Activate a repository-advertised managed-CI contract.

    Repositories without any contract markers retain legacy behavior. A partial
    contract fails closed because applying its suppression label could otherwise
    disable hosted tests without a usable final qualification path.
    """
    if not config.effective_managed_ci or config.dry_run:
        return None

    workflow_ref = metadata.base_branch or config.base
    workflow_endpoint = f"repos/{config.repo}/contents/.github/workflows/{WORKFLOW_FILE}"
    if workflow_ref:
        workflow_endpoint += f"?ref={workflow_ref}"
    result = runner.run(
        [
            config.gh_cmd,
            "api",
            workflow_endpoint,
            "-H",
            "Accept: application/vnd.github.raw+json",
        ],
        cwd=active_workdir(config),
        check=False,
    )
    if result.returncode != 0:
        if config.managed_ci:
            raise AgentLoopError(
                "--managed-ci could not read the base workflow; managed exact-head qualification "
                "is unavailable and no ordinary fallback was selected."
            )
        return None
    workflow = result.stdout or ""
    present = tuple(marker for marker in _CONTRACT_MARKERS if marker in workflow)
    if not present:
        if config.managed_ci:
            raise AgentLoopError(
                "--managed-ci requested managed exact-head qualification, but the base workflow "
                "does not advertise a complete v2 managed-CI contract."
            )
        return None
    missing = tuple(marker for marker in _CONTRACT_MARKERS if marker not in workflow)
    if missing:
        raise AgentLoopError(
            "Repository CI advertises an incomplete managed-CI contract; missing marker(s): "
            + ", ".join(missing)
        )

    # v2 is deliberately an explicit migration.  A v1 workflow continues to
    # use the post-open label handoff below; a v2 workflow must prove both the
    # invoking identity and the complete PR creation tuple before it can make
    # an automatic pull_request matrix disappear.
    if V2_MARKER in workflow:
        missing_v2 = tuple(marker for marker in V2_FEATURE_MARKERS if marker not in workflow)
        if missing_v2:
            raise AgentLoopError(
                "Repository CI advertises an incomplete managed-CI v2 contract; missing marker(s): "
                + ", ".join(missing_v2)
            )
        contract = _activate_v2_managed_ci(
            runner,
            config=config,
            pr_number=pr_number,
            metadata=metadata,
            managed_resume=managed_resume,
            resume_origin=resume_origin,
        )
        if contract is not None or not config.managed_ci_adopt_existing_pr:
            if contract is None and config.managed_ci:
                raise AgentLoopError(
                    f"PR #{pr_number} does not match the authenticated issue-created managed-CI "
                    "tuple; use --managed-ci-adopt-existing-pr for an eligible existing PR."
                )
            return contract
        # A complete ordinary v2 workflow is not an adoption contract.  This
        # path is intentionally quiet/fallback-compatible when the optional
        # marker has not been deployed.
        adopted = _activate_v2_existing_pr_adoption(
            runner,
            config=config,
            pr_number=pr_number,
            metadata=metadata,
        )
        if adopted is None and config.managed_ci:
            raise AgentLoopError(
                f"--managed-ci-adopt-existing-pr could not safely adopt PR #{pr_number}; "
                "no suppression or qualification was claimed."
            )
        return adopted

    if config.managed_ci:
        raise AgentLoopError(
            "--managed-ci is unsupported for the legacy v1 managed-CI contract; deploy the complete v2 workflow."
        )

    pr_result = runner.run(
        [config.gh_cmd, "api", f"repos/{config.repo}/pulls/{pr_number}"],
        cwd=active_workdir(config),
        check=False,
    )
    if pr_result.returncode != 0:
        raise AgentLoopError(f"Unable to validate managed-CI eligibility for PR #{pr_number}.")
    try:
        pr_data = json.loads(pr_result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise AgentLoopError(
            f"Managed-CI eligibility response for PR #{pr_number} was invalid JSON."
        ) from exc
    head = pr_data.get("head") or {}
    head_repo = (head.get("repo") or {}).get("full_name")
    live_head_sha = head.get("sha")
    live_head_ref = head.get("ref")
    base_ref = (pr_data.get("base") or {}).get("ref")
    if not isinstance(head_repo, str) or head_repo.casefold() != config.repo.casefold():
        return None
    if not metadata.base_branch or base_ref != metadata.base_branch:
        return None
    if not metadata.head_sha:
        raise AgentLoopError(f"PR #{pr_number} has no head SHA; managed CI cannot be activated.")
    if not metadata.head_branch or live_head_ref != metadata.head_branch:
        raise AgentLoopError(
            f"PR #{pr_number} has no stable same-repository head branch for managed CI."
        )
    if live_head_sha != metadata.head_sha:
        raise AgentLoopError(
            f"PR #{pr_number} moved from {metadata.head_sha} to "
            f"{live_head_sha or 'an unknown head'} "
            "while managed CI was being activated; rerun against the live head."
        )

    labels = {
        label.get("name")
        for label in pr_data.get("labels") or []
        if isinstance(label, dict) and isinstance(label.get("name"), str)
    }
    if not ensure_managed_label(runner, config=config):
        raise AgentLoopError(f"Unable to create the `{MANAGED_LABEL}` label.")
    if MANAGED_LABEL not in labels:
        prior_workflow_run_ids = _workflow_run_ids(
            runner,
            config=config,
            head_sha=metadata.head_sha,
        )
        apply_result = runner.run(
            [
                config.gh_cmd,
                "api",
                "--method",
                "POST",
                f"repos/{config.repo}/issues/{pr_number}/labels",
                "-f",
                f"labels[]={MANAGED_LABEL}",
            ],
            cwd=active_workdir(config),
            check=False,
        )
        if apply_result.returncode != 0:
            raise AgentLoopError(f"Unable to apply `{MANAGED_LABEL}` to PR #{pr_number}.")
        try:
            _wait_for_label_handoff(
                runner,
                config=config,
                pr_number=pr_number,
                metadata=metadata,
                prior_run_ids=prior_workflow_run_ids,
            )
        except AgentLoopError as exc:
            remove_result = runner.run(
                [
                    config.gh_cmd,
                    "api",
                    "--method",
                    "DELETE",
                    f"repos/{config.repo}/issues/{pr_number}/labels/{MANAGED_LABEL}",
                ],
                cwd=active_workdir(config),
                check=False,
            )
            if remove_result.returncode != 0:
                raise AgentLoopError(
                    f"{exc} Cleanup also failed: `{MANAGED_LABEL}` remains applied and "
                    "suppresses hosted CI until it is removed."
                ) from exc
            raise
    log(config, f"PR #{pr_number}: activated managed exact-head CI")
    return ManagedCiContract()


def _probe(runner: Runner, context: ManagedCiProbeContext, endpoint: str, *, raw: bool = False):
    command = [context.gh_cmd, "api", endpoint]
    if raw:
        command.extend(["-H", "Accept: application/vnd.github.raw+json"])
    return runner.run(command, cwd=context.cwd, check=False)


def _probe_json(runner: Runner, context: ManagedCiProbeContext, endpoint: str) -> tuple[dict[str, object] | None, object]:
    result = _probe(runner, context, endpoint)
    if result.returncode != 0:
        return None, result
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None, result
    return (payload if isinstance(payload, dict) else None), result


def _probe_json_list(
    runner: Runner, context: ManagedCiProbeContext, endpoint: str
) -> tuple[list[object] | None, object]:
    """Read a GitHub list response without treating a valid array as malformed."""
    result = _probe(runner, context, endpoint)
    if result.returncode != 0:
        return None, result
    try:
        payload = json.loads(result.stdout or "")
    except json.JSONDecodeError:
        return None, result
    return (payload if isinstance(payload, list) else None), result


def _is_http_error(result: object, *statuses: int) -> bool:
    """`gh api` uses exit 1 for HTTP errors, so inspect its diagnostic text."""
    combined = " ".join(
        str(getattr(result, field, "") or "") for field in ("stdout", "stderr")
    ).casefold()
    return any(str(status) in combined for status in statuses)


def _is_plan_limited_error(result: object) -> bool:
    """Recognize GitHub's plan restriction independently of endpoint wording."""
    combined = " ".join(
        str(getattr(result, field, "") or "") for field in ("stdout", "stderr")
    ).casefold()
    return "upgrade to github pro" in combined or "make this repository public" in combined


def _probe_raw_workflow(
    runner: Runner, context: ManagedCiProbeContext, base: str
) -> tuple[str | None, object]:
    result = _probe(
        runner, context,
        f"repos/{context.repo}/contents/.github/workflows/{WORKFLOW_FILE}?ref={base}", raw=True,
    )
    return ((result.stdout or "") if result.returncode == 0 else None), result


def _has_final_context(payload: dict[str, object]) -> bool:
    contexts = payload.get("contexts")
    checks = payload.get("checks")
    return (
        isinstance(contexts, list) and FINAL_CONTEXT in contexts
    ) or (
        isinstance(checks, list)
        and any(isinstance(check, dict) and check.get("context") == FINAL_CONTEXT for check in checks)
    )


def assess_exact_head_protection(
    runner: Runner, *, context: ManagedCiProbeContext, base: str
) -> ProtectionAssessment:
    """Classify GitHub enforcement conservatively.

    A required context alone is voluntary when administrators can bypass it.
    Rulesets are inspected as an additional independent enforcement source;
    evaluate-mode and bypass actors are deliberately not strict.
    """
    required, required_result = _probe_json(
        runner, context, f"repos/{context.repo}/branches/{base}/protection/required_status_checks"
    )
    classic_state: Literal["voluntary", "plan_limited", "indeterminate"] | None = None
    classic_detail = "final-ci/exact-head is not independently required"
    if required is None:
        if _is_plan_limited_error(required_result):
            classic_state = "plan_limited"
            classic_detail = "GitHub plan/API does not permit branch protection"
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
        if _is_http_error(rules_result, 404, 422):
            return ProtectionAssessment(classic_state or "voluntary", "classic" if classic_state else "none", classic_detail)
        return ProtectionAssessment("indeterminate", "rulesets", "effective branch rules could not be inspected")
    voluntary = False
    for entry in rules:
        if not isinstance(entry, dict):
            continue
        ruleset_id = entry.get("ruleset_id") or entry.get("id")
        if not isinstance(ruleset_id, int):
            continue
        ruleset, ignored = _probe_json(runner, context, f"repos/{context.repo}/rulesets/{ruleset_id}")
        if ruleset is None:
            return ProtectionAssessment("indeterminate", "rulesets", "an applicable ruleset could not be inspected")
        contexts = json.dumps(ruleset.get("rules") or [])
        if FINAL_CONTEXT not in contexts:
            continue
        if ruleset.get("enforcement") != "active" or ruleset.get("bypass_actors"):
            voluntary = True
            continue
        return ProtectionAssessment("strict", "ruleset", "active ruleset requires final-ci/exact-head without bypass actors")
    if classic_state == "indeterminate":
        return ProtectionAssessment("indeterminate", "classic", classic_detail)
    if classic_state == "plan_limited":
        return ProtectionAssessment("plan_limited", "classic", classic_detail)
    return ProtectionAssessment(
        "voluntary", "rulesets" if voluntary else ("classic" if classic_state else "none"),
        "matching ruleset is bypassable/evaluate-mode" if voluntary else classic_detail,
    )


def evaluate_managed_ci_readiness(
    runner: Runner, *, context: ManagedCiProbeContext, base: str | None, trusted_actor: str
) -> ManagedCiReadiness:
    """Read every v2 prerequisite without writes or AgentLoopConfig construction."""
    repo, repo_result = _probe_json(runner, context, f"repos/{context.repo}")
    if repo is None:
        return ManagedCiReadiness("indeterminate", None, None, None, False, False,
            ProtectionAssessment("indeterminate", "none", "repository metadata unavailable"),
            ("repository metadata could not be read",), ())
    resolved_base = base or (repo.get("default_branch") if isinstance(repo.get("default_branch"), str) else None)
    visibility = "private" if repo.get("private") is True else "public" if repo.get("private") is False else None
    if not resolved_base:
        return ManagedCiReadiness("indeterminate", visibility, None, None, False, False,
            ProtectionAssessment("indeterminate", "none", "base branch unavailable"), ("base branch is unavailable",), ())
    workflow, workflow_result = _probe_raw_workflow(runner, context, resolved_base)
    if workflow is None:
        if _is_http_error(workflow_result, 404):
            return ManagedCiReadiness(
                "ordinary_fallback", visibility, None, None, False, False,
                ProtectionAssessment("voluntary", "none", "base workflow is absent"),
                ("base workflow is absent",),
                ("deploy the documented managed-CI v2 workflow",),
                resolved_base,
            )
        return ManagedCiReadiness(
            "indeterminate", visibility, None, None, False, False,
            ProtectionAssessment("indeterminate", "none", "base workflow could not be read"),
            ("a required read-only GitHub probe failed",), (), resolved_base,
        )
    who, _ = _probe_json(runner, context, "user")
    variable, variable_result = _probe_json(runner, context, f"repos/{context.repo}/actions/variables/AGENT_LOOP_MANAGED_ACTOR")
    actor = who.get("login") if who and isinstance(who.get("login"), str) else None
    advertised = variable.get("value") if variable and isinstance(variable.get("value"), str) else None
    protection = assess_exact_head_protection(runner, context=context, base=resolved_base)
    if who is None or (variable is None and not _is_http_error(variable_result, 404)):
        return ManagedCiReadiness("indeterminate", visibility, actor, advertised, False, False, protection,
            ("a required read-only GitHub probe failed",), ())
    core = V2_MARKER in workflow
    complete = core and all(marker in workflow for marker in V2_FEATURE_MARKERS)
    # Pre-marker v2 fixtures/workflows that contain no pull_request trigger at
    # all never suppress an opening matrix. Treat them as legacy-compatible;
    # any workflow that does opt into pull_request suppression must advertise
    # the explicit recovery marker and unlabeled activity.
    recovery = (RECOVERY_MARKER in workflow and "unlabeled" in workflow) or "pull_request" not in workflow
    expected_actor = trusted_actor.strip()
    identity_ok = bool(actor and advertised and expected_actor and actor.casefold() == expected_actor.casefold() == advertised.casefold())
    if not core:
        return ManagedCiReadiness("ordinary_fallback", visibility, actor, advertised, False, recovery, protection,
            ("base workflow does not advertise managed-CI v2",), ("deploy the documented managed-CI v2 workflow",), resolved_base)
    if not complete or not recovery:
        missing = "complete v2 contract" if not complete else "unlabeled recovery contract"
        return ManagedCiReadiness("invalid", visibility, actor, advertised, complete, recovery, protection,
            (f"workflow lacks the {missing}",), ("add the documented workflow markers and pull_request unlabeled trigger",), resolved_base)
    if not identity_ok:
        return ManagedCiReadiness("ordinary_fallback", visibility, actor, advertised, complete, recovery, protection,
            ("authenticated login, trusted actor, and AGENT_LOOP_MANAGED_ACTOR do not match",),
            (f"gh variable set AGENT_LOOP_MANAGED_ACTOR --repo {shlex.quote(context.repo)} --body {shlex.quote(expected_actor)}",), resolved_base)
    if protection.state == "strict":
        return ManagedCiReadiness("strict_ready", visibility, actor, advertised, complete, recovery, protection, (), (), resolved_base)
    if protection.state in {"voluntary", "plan_limited"}:
        return ManagedCiReadiness("override_eligible", visibility, actor, advertised, complete, recovery, protection,
            (protection.detail,), ("configure non-bypassable final-ci/exact-head protection, or use --allow-unprotected-managed-ci for this invocation",), resolved_base)
    return ManagedCiReadiness("indeterminate", visibility, actor, advertised, complete, recovery, protection,
        (protection.detail,), (), resolved_base)


def render_managed_ci_preflight(result: ManagedCiReadiness, *, repo: str, base: str, trusted_actor: str) -> str:
    lines = [
        f"repository: {repo}", f"base: {result.base or base}", f"visibility: {result.repo_visibility or 'unknown'}",
        f"workflow: {'v2 complete' if result.workflow_v2 else 'ordinary/absent'}",
        f"recovery: {'unlabeled capable' if result.recovery_capable else 'missing'}",
        f"actor: authenticated={result.actor or 'unknown'} trusted={trusted_actor} variable={result.advertised_actor or 'missing'}",
        f"protection: {result.protection.state} ({result.protection.source}; {result.protection.detail})",
        f"result: {result.state}",
    ]
    lines.extend(f"reason: {reason}" for reason in result.reasons)
    lines.extend(f"remediation: {command}" for command in result.remediation)
    return "\n".join(lines)


def preflight_managed_ci_creation(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int | None = None,
    branch: str | None = None,
) -> ManagedCiCreationIntent | None:
    """Return an atomic creation intent only for an authenticated v2 rollout.

    This happens before the coder is prompted, avoiding the opened-event race:
    the PR is born as a recognizable reserved draft and is labeled before the
    orchestrator continues, or it is ordinary CI.
    """
    if (issue_number is None) == (branch is None):
        raise AgentLoopError("Managed-CI creation preflight requires exactly one issue number or branch.")
    if branch is not None and not branch.startswith("agent-loop/managed-"):
        raise AgentLoopError("Managed-CI creation branches must use the reserved `agent-loop/managed-` prefix.")
    if not config.effective_managed_ci or config.dry_run or not config.managed_ci_trusted_actor or not config.base:
        return None
    source_context = ManagedCiProbeContext(config.repo, config.gh_cmd, active_workdir(config))
    source_workflow, _ = _probe_raw_workflow(runner, source_context, config.base)
    if source_workflow is None or V2_MARKER not in source_workflow:
        if config.managed_ci:
            raise AgentLoopError(
                "--managed-ci requires a complete v2 workflow on the resolved base branch; "
                "v1 or ordinary CI cannot qualify a managed head."
            )
        return None
    if any(marker not in source_workflow for marker in V2_FEATURE_MARKERS):
        raise AgentLoopError("Repository CI advertises an incomplete managed-CI v2 contract.")
    legacy_non_suppressing = "pull_request" not in source_workflow
    readiness = evaluate_managed_ci_readiness(
        runner,
        context=source_context,
        base=config.base,
        trusted_actor=config.managed_ci_trusted_actor,
    )
    if readiness.state == "indeterminate" and not legacy_non_suppressing:
        if config.managed_ci:
            raise AgentLoopError(
                "--managed-ci could not determine the repository's exact-head protection; "
                "no PR was created."
            )
        log(config, "Managed-CI readiness could not be determined; continuing with ordinary CI.")
        return None
    if readiness.state == "indeterminate":
        who = _api_json(runner, config, "user", quiet=True)
        advertised = _api_json(
            runner, config, f"repos/{config.repo}/actions/variables/AGENT_LOOP_MANAGED_ACTOR", quiet=True
        ).get("value")
        actor = who.get("login") if isinstance(who.get("login"), str) else None
        if not isinstance(actor, str) or not isinstance(advertised, str) or (
            actor.casefold() != config.managed_ci_trusted_actor.casefold()
            or actor.casefold() != advertised.casefold()
        ):
            if config.managed_ci:
                raise AgentLoopError(
                    "--managed-ci authentication does not match the configured trusted actor; no PR was created."
                )
            return None
        readiness = ManagedCiReadiness(
            "override_eligible", None, actor, advertised, True, True,
            ProtectionAssessment("voluntary", "legacy", "legacy non-suppressing workflow"), (), (),
        )
    if readiness.state == "override_eligible" and not (config.allow_unprotected_managed_ci or legacy_non_suppressing):
        if config.managed_ci:
            raise AgentLoopError(
                "--managed-ci requires protected final-ci/exact-head or "
                "--allow-unprotected-managed-ci; no PR was created."
            )
        return None
    if readiness.state != "strict_ready" and not (
        readiness.state == "override_eligible" and (config.allow_unprotected_managed_ci or legacy_non_suppressing)
    ):
        if config.managed_ci:
            raise AgentLoopError(
                "--managed-ci repository readiness is not sufficient for suppression and exact-head qualification; "
                "no PR was created."
            )
        return None
    return ManagedCiCreationIntent(
        branch=branch or f"agent-loop/managed-{issue_number}",
        trusted_actor=readiness.actor or config.managed_ci_trusted_actor,
        protection_mode=readiness.protection.state,
        audit_nonce=secrets.token_urlsafe(18) if config.allow_unprotected_managed_ci else None,
    )


def _issue_created_tuple(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    issue_number: int,
    metadata: PullRequestMetadata,
    expected_branch: str,
    expected_nonce: str | None,
    protection_mode: str,
    lifecycle: Literal["draft-labeled", "draft-unlabeled", "ready-unlabeled"] = "draft-labeled",
) -> AuthenticatedIssueCreatedHandoff:
    """Read and verify the immutable tuple before any handoff publication.

    The GraphQL view is useful to the review loop but cannot attest author,
    same-repository head, labels, and draft state together.  Require it to
    agree with the REST tuple and re-read the REST tuple before returning so a
    transient body/head race cannot escape into a writer.
    """
    if not config.base or metadata.number != pr_number or metadata.repo.casefold() != config.repo.casefold():
        raise AgentLoopError("Managed-CI issue handoff does not match the configured repository/base.")
    body = metadata.body or ""
    record = parse_managed_ci_override_record(
        body,
        surface=PR_BODY_SURFACE,
        schema="body",
        required=expected_nonce is not None,
        expected_nonce=expected_nonce,
    )
    if expected_nonce is None and record is not None:
        raise AgentLoopError("Strict managed-CI issue handoff must not contain an override record.")
    linked = parse_strong_issue_reference_evidence(body, repo=config.repo, issue_number=issue_number)
    if len(linked) != 1:
        raise AgentLoopError("Managed-CI issue handoff requires one canonical closing reference to its issue.")
    who = _api_json(runner, config, "user", quiet=True)
    actor_login = who.get("login") if isinstance(who.get("login"), str) else None
    actor_id = who.get("id") if isinstance(who.get("id"), int) else None
    variable = _api_json(
        runner, config,
        f"repos/{config.repo}/actions/variables/AGENT_LOOP_MANAGED_ACTOR",
        quiet=True,
    )
    advertised = variable.get("value") if isinstance(variable.get("value"), str) else None
    configured = (config.managed_ci_trusted_actor or "").strip()
    if (
        not configured or not actor_login or actor_id is None or not advertised
        or actor_login.casefold() != configured.casefold()
        or actor_login.casefold() != advertised.casefold()
    ):
        trusted_actor = advertised or actor_login or configured
        if trusted_actor:
            remediation_config = replace(config, managed_ci_trusted_actor=trusted_actor)
            remediation = render_managed_ci_resume_command(
                remediation_config,
                pr_number=pr_number,
                issue_number=issue_number,
                managed_ci=True,
            )
        else:
            remediation = render_managed_ci_resume_command(
                config,
                pr_number=pr_number,
                issue_number=issue_number,
                managed_ci=True,
            ) + " --managed-ci-trusted-actor '<trusted-actor>'"
        raise AgentLoopError(
            "Managed-CI issue handoff actor authentication does not match repository settings. "
            f"Observed identities: gh login=`{actor_login or '<missing>'}`, "
            f"--managed-ci-trusted-actor=`{configured or '<missing>'}`, "
            f"AGENT_LOOP_MANAGED_ACTOR=`{advertised or '<missing>'}`; expected all three to match. "
            f"Rerun with the configured trusted actor: `{remediation}`."
        )

    remediation = render_managed_ci_resume_command(
        config,
        pr_number=pr_number,
        issue_number=issue_number,
        managed_ci=True,
    )

    def check_rest(pr: dict[str, object]) -> tuple[str, str]:
        head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
        base = pr.get("base") if isinstance(pr.get("base"), dict) else {}
        author = pr.get("user") if isinstance(pr.get("user"), dict) else {}
        head_repo = head.get("repo") if isinstance(head.get("repo"), dict) else {}
        labels = {
            item.get("name") for item in (pr.get("labels") or [])
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        sha = head.get("sha") if isinstance(head.get("sha"), str) else None
        branch = head.get("ref") if isinstance(head.get("ref"), str) else None
        rest_body = pr.get("body") if isinstance(pr.get("body"), str) else ""
        lifecycle_matches = (
            pr.get("draft") is True and MANAGED_LABEL in labels
            if lifecycle == "draft-labeled"
            else pr.get("draft") is True and MANAGED_LABEL not in labels
            if lifecycle == "draft-unlabeled"
            else pr.get("draft") is False and MANAGED_LABEL not in labels
        )
        if (
            pr.get("state") not in {"open", "OPEN"}
            or not lifecycle_matches
            or head_repo.get("full_name", "").casefold() != config.repo.casefold()
            or branch != expected_branch
            or base.get("ref") != config.base
            or (lifecycle == "draft-labeled" and MANAGED_LABEL not in labels)
            or (lifecycle != "draft-labeled" and MANAGED_LABEL in labels)
            or author.get("login", "").casefold() != actor_login.casefold()
            or author.get("id") != actor_id
            or sha != metadata.head_sha
            or branch != metadata.head_branch
            or base.get("ref") != metadata.base_branch
            or rest_body != body
        ):
            raise AgentLoopError(
                "Managed-CI issue-created opening tuple is missing or changed. "
                f"The PR was left unchanged. Resume with `{remediation}`."
            )
        return sha, branch

    first = _api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}", quiet=True)
    sha, branch = check_rest(first)
    second = _api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}", quiet=True)
    live_sha, live_branch = check_rest(second)
    if (live_sha, live_branch) != (sha, branch):
        raise AgentLoopError(
            "Managed-CI issue-created opening tuple changed during validation. "
            f"The PR was left unchanged. Resume with `{remediation}`."
        )
    return AuthenticatedIssueCreatedHandoff(
        pr_number=pr_number,
        issue_number=issue_number,
        repository=config.repo,
        base_ref=config.base,
        head_sha=sha,
        branch=branch,
        trusted_actor_login=actor_login,
        trusted_actor_id=actor_id,
        protection_mode=protection_mode,
        override_nonce=record.nonce if record is not None else None,
        lifecycle={
            "draft-labeled": "draft-labeled",
            "draft-unlabeled": "draft-unlabeled-reentry",
            "ready-unlabeled": "ready-unlabeled-reentry",
        }[lifecycle],
    )


def authenticate_issue_created_handoff(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    intent: ManagedCiCreationIntent,
    issue_number: int,
    pr_number: int,
    metadata: PullRequestMetadata,
) -> AuthenticatedIssueCreatedHandoff:
    """Authenticate a just-created issue draft before *any* remote write."""
    return _issue_created_tuple(
        runner,
        config=config,
        pr_number=pr_number,
        issue_number=issue_number,
        metadata=metadata,
        expected_branch=intent.branch,
        expected_nonce=intent.audit_nonce,
        protection_mode=intent.protection_mode,
        lifecycle="draft-labeled",
    )


def revalidate_issue_created_handoff(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    handoff: AuthenticatedIssueCreatedHandoff,
    metadata: PullRequestMetadata,
) -> AuthenticatedIssueCreatedHandoff:
    """Re-read an authenticated handoff at PR-loop entry before any writer."""
    if handoff.repository.casefold() != config.repo.casefold() or handoff.base_ref != config.base:
        raise AgentLoopError("Managed-CI handoff does not belong to this invocation.")
    validated = _issue_created_tuple(
        runner,
        config=config,
        pr_number=handoff.pr_number,
        issue_number=handoff.issue_number,
        metadata=metadata,
        expected_branch=handoff.branch,
        expected_nonce=handoff.override_nonce,
        protection_mode=handoff.protection_mode,
        lifecycle=(
            "ready-unlabeled"
            if handoff.lifecycle == "ready-unlabeled-reentry"
            else "draft-unlabeled"
            if handoff.lifecycle == "draft-unlabeled-reentry"
            else "draft-labeled"
        ),
    )
    if handoff.active_label_event_id is not None:
        event = _active_managed_label_event(runner, config=config, pr_number=handoff.pr_number)
        if (
            event is None
            or event[0] != handoff.active_label_event_id
            or event[1].casefold() != handoff.trusted_actor_login.casefold()
            or event[2] != handoff.trusted_actor_id
        ):
            raise AgentLoopError("Managed-CI direct-resume label provenance changed before activation.")
        validated = replace(validated, active_label_event_id=handoff.active_label_event_id)
    return validated


def recover_issue_created_handoff(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    metadata: PullRequestMetadata,
    issue_number: int | None = None,
) -> AuthenticatedIssueCreatedHandoff | None:
    """Reconstruct a durable issue-created resume without reusing authority.

    Canonical issue reruns and direct PR reruns use the same read-only proof.
    The body nonce, when present, is provenance only; the current lifecycle,
    head, author, base, label, and issue association are reauthenticated.
    """
    if not (config.managed_ci_pr_mode or issue_number is not None):
        return None
    body = metadata.body or ""
    branch = metadata.head_branch or ""
    match = re.fullmatch(r"agent-loop/managed-([1-9]\d*)", branch)
    if match is None:
        return None
    branch_issue_number = int(match.group(1))
    if issue_number is not None and issue_number != branch_issue_number:
        command = render_managed_ci_resume_command(
            config,
            pr_number=pr_number,
            issue_number=issue_number,
            managed_ci=True,
        )
        raise AgentLoopError(
            f"Managed-CI issue association targets issue #{issue_number}, but the authenticated "
            f"managed branch targets issue #{branch_issue_number}. The PR was left unchanged. "
            f"Rerun the canonical issue command after correcting the issue association: `{command}`"
        )
    issue_number = branch_issue_number
    record = None
    if UNPROTECTED_OVERRIDE_TRAILER in body:
        record = parse_managed_ci_override_record(
            body, surface=PR_BODY_SURFACE, schema="body", required=True,
        )
    pr = _api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}", quiet=True)
    labels = {
        item.get("name") for item in (pr.get("labels") or [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    draft = pr.get("draft")
    if draft is True and MANAGED_LABEL in labels:
        lifecycle = "draft-labeled"
    elif draft is True and MANAGED_LABEL not in labels and config.managed_ci:
        # A failed explicit managed-CI invocation deliberately releases its
        # suppression label.  Re-admit that durable draft state only when the
        # operator explicitly asks for managed qualification again.
        lifecycle = "draft-unlabeled"
    elif draft is False and MANAGED_LABEL not in labels:
        lifecycle = "ready-unlabeled"
    else:
        if draft is True:
            state = "draft/labeled" if MANAGED_LABEL in labels else "draft/unlabeled"
        elif draft is False:
            state = "ready/labeled" if MANAGED_LABEL in labels else "ready/unlabeled"
        else:
            state = "unknown/labeled" if MANAGED_LABEL in labels else "unknown/unlabeled"
        command = render_managed_ci_resume_command(
            config,
            pr_number=pr_number,
            issue_number=issue_number,
            managed_ci=True,
        )
        raise AgentLoopError(
            f"Managed-CI issue-created resume for PR #{pr_number} observed mixed lifecycle state "
            f"{state}; expected draft/labeled, draft/unlabeled with explicit managed CI, or "
            f"ready/unlabeled. The PR was left unchanged. Resume with `{command}`."
        )
    # A resumed nonce is provenance only.  Its value is checked against the
    # live body, then activation mints a fresh audit/generation.
    handoff = _issue_created_tuple(
        runner,
        config=config,
        pr_number=pr_number,
        issue_number=issue_number,
        metadata=metadata,
        expected_branch=branch,
        expected_nonce=record.nonce if record is not None else None,
        protection_mode="voluntary" if record is not None else "strict",
        lifecycle=lifecycle,
    )
    if lifecycle == "draft-labeled":
        event = _active_managed_label_event(runner, config=config, pr_number=pr_number)
        if (
            event is None
            or event[1].casefold() != handoff.trusted_actor_login.casefold()
            or event[2] != handoff.trusted_actor_id
        ):
            raise AgentLoopError(
                "Managed-CI issue-created resume requires an actor-owned active managed-label event."
            )
        handoff = replace(handoff, active_label_event_id=event[0])
    if record is not None:
        comments = _api_list(
            runner, config, f"repos/{config.repo}/issues/{pr_number}/comments?per_page=100"
        )
        if comments is None:
            raise AgentLoopError("Managed-CI issue-created resume could not inspect override audit provenance.")
        if any(
            UNPROTECTED_OVERRIDE_TRAILER in (
                comment.get("body") if isinstance(comment.get("body"), str) else ""
            )
            for comment in comments
        ) and _find_resume_audit(
            runner,
            config=config,
            pr_number=pr_number,
            actor_login=handoff.trusted_actor_login,
            actor_id=handoff.trusted_actor_id,
            base_ref=handoff.base_ref,
        ) is None:
            raise AgentLoopError(
                "Managed-CI issue-created resume found malformed or uncorrelated override audit provenance."
            )
    return handoff


def authenticate_source_managed_resume(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    source_branch: str,
    source_sha: str,
    managed_branch: str,
    override_nonce: str | None,
) -> AuthenticatedManagedResume:
    """Authenticate the durable lifecycle of a recovered managed-pr origin."""
    pr = _api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}", quiet=True)
    labels = {
        item.get("name") for item in (pr.get("labels") or [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if pr.get("draft") is True and MANAGED_LABEL in labels:
        lifecycle: Literal[
            "draft-labeled", "draft-unlabeled-reentry", "ready-unlabeled-reentry"
        ] = "draft-labeled"
    elif pr.get("draft") is True and MANAGED_LABEL not in labels and config.managed_ci:
        lifecycle = "draft-unlabeled-reentry"
    elif pr.get("draft") is False and MANAGED_LABEL not in labels:
        lifecycle = "ready-unlabeled-reentry"
    else:
        draft = pr.get("draft")
        if draft is True:
            state = "draft/labeled" if MANAGED_LABEL in labels else "draft/unlabeled"
        elif draft is False:
            state = "ready/labeled" if MANAGED_LABEL in labels else "ready/unlabeled"
        else:
            state = "unknown/labeled" if MANAGED_LABEL in labels else "unknown/unlabeled"
        command = render_managed_ci_resume_command(config, pr_number=pr_number, managed_ci=True)
        raise AgentLoopError(
            f"Managed source PR #{pr_number} observed mixed lifecycle state {state}; "
            f"expected draft/labeled, draft/unlabeled with explicit managed CI, or authenticated "
            f"ready/unlabeled. The PR was left unchanged. Resume with `{command}`."
        )
    return AuthenticatedManagedResume(
        origin="source-managed",
        lifecycle=lifecycle,
        source_branch=source_branch,
        source_sha=source_sha,
        managed_branch=managed_branch,
        override_nonce=override_nonce,
    )


_RECOVERY_VALUE_OPTIONS = frozenset({
    "--repo", "--base", "--claude-dir", "--codex-dir", "--gemini-dir", "--antigravity-dir",
    "--coder", "--reviewer", "--max-rounds", "--managed-ci-trusted-actor",
    "--implementation-coder", "--implementation-coder-model",
    "--implementation-codex-reasoning-effort", "--claude-cmd", "--codex-cmd",
    "--gemini-cmd", "--antigravity-cmd", "--antigravity-print-timeout-seconds",
    "--repair-backend", "--repair-model", "--repair-timeout-seconds",
    "--antigravity-model", "--antigravity-models", "--antigravity-quota-signatures",
    "--codex-model", "--codex-reasoning-effort", "--gemini-model", "--claude-model",
    "--gh-cmd",
    "--claude-arg", "--codex-arg", "--gemini-arg", "--antigravity-arg",
    "--test-command", "--ci-timeout-seconds",
    "--ci-poll-interval-seconds", "--ci-startup-timeout-seconds",
    "--ci-queued-grace-seconds", "--mergeability-poll-attempts",
    "--mergeability-poll-interval-seconds", "--log-dir", "--subprocess-log-dir",
    "--progress-interval-seconds", "--agent-max-retries", "--agent-retry-backoff-seconds",
    "--agent-memory-dir", "--salvage-comment-patch-max-bytes", "--planning-context-mode",
    "--pr-review-context-mode", "--expected-closing-issue",
    "--plan-execution-mode", "--flat-child-limit", "--split-stage", "--head", "--title", "--body-file",
    "--approved-followups",
})
_RECOVERY_NARGS_VALUE_OPTIONS = frozenset({
    "--antigravity-models", "--antigravity-quota-signatures", "--agent-retry-backoff-seconds",
})
_ISSUE_ONLY_RECOVERY_OPTIONS = frozenset({
    "--plan-first", "--implement-after-approval", "--plan-execution-mode",
    "--materialize-split-issues", "--split-stage", "--implementation-coder",
    "--implementation-coder-model", "--implementation-codex-reasoning-effort",
})
_PR_ONLY_RECOVERY_OPTIONS = frozenset({"--managed-ci-adopt-existing-pr"})
_MANAGED_PR_ONLY_RECOVERY_OPTIONS = frozenset({"--head", "--title", "--body-file"})
_MANAGED_RECOVERY_OPTIONS = frozenset({
    "--managed-ci", "--managed-ci-trusted-actor", "--allow-unprotected-managed-ci",
    "--managed-ci-adopt-existing-pr",
})
def _option_name(token: str) -> str:
    return token.split("=", 1)[0]


def _recovery_option_end(argv: list[str], index: int) -> int:
    """Return the first token after one option and its value payload."""
    token = argv[index]
    name = _option_name(token)
    if "=" in token or name not in _RECOVERY_VALUE_OPTIONS:
        return index + 1
    if name in _RECOVERY_NARGS_VALUE_OPTIONS:
        index += 1
        while index < len(argv) and (
            not argv[index].startswith("-") or argv[index] == "-"
        ):
            index += 1
        return index
    if index + 1 < len(argv) and (
        not argv[index + 1].startswith("-") or argv[index + 1] == "-"
    ):
        return index + 2
    return index + 1


def _find_recovery_positional_index(argv: list[str], command_index: int) -> int | None:
    """Find an issue/PR identifier even when options precede it."""
    index = command_index + 1
    while index < len(argv):
        if argv[index].startswith("-"):
            index = _recovery_option_end(argv, index)
            continue
        try:
            int(argv[index])
        except ValueError:
            return None
        return index
    return None


def _strip_recovery_options(
    argv: list[str], *, names: frozenset[str]
) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        name = _option_name(token)
        if name in names:
            index = _recovery_option_end(argv, index)
            continue
        result.append(token)
        index += 1
    return result


def _has_option(argv: list[str], name: str) -> bool:
    return any(_option_name(token) == name for token in argv)


def _render_recovery_command(
    config: AgentLoopConfig,
    *,
    target: Literal["issue", "pr"],
    identifier: int,
    managed_ci: bool,
    preserve_managed_options: bool = False,
    include_context: bool = True,
) -> str:
    """Build one parser-valid, shell-quoted recovery command.

    Invocation tokens are replayed when available.  Retargeting removes only
    options that the target subparser cannot accept, while preserving repeated
    common options and their complete value payloads.
    """
    if not config.invocation_argv:
        if not include_context:
            command = ["agent-loop", target, str(identifier)]
            if config.auto_merge:
                command.append("--auto-merge")
            elif config.watch_pending_ci:
                command.append("--watch-pending-ci")
            return shlex.join(command)
        command = ["agent-loop", target, str(identifier), "--repo", config.repo]
        if config.base:
            command.extend(("--base", config.base))
        if config.auto_merge:
            command.append("--auto-merge")
        if config.watch_pending_ci and not config.auto_merge:
            command.append("--watch-pending-ci")
        if managed_ci:
            command.append("--managed-ci")
            if config.managed_ci_trusted_actor:
                command.extend(("--managed-ci-trusted-actor", config.managed_ci_trusted_actor))
            if config.allow_unprotected_managed_ci:
                command.append("--allow-unprotected-managed-ci")
        return shlex.join(command)

    command = list(config.invocation_argv)
    command_index = next(
        (index for index, token in enumerate(command) if token in {"issue", "pr", "managed-pr"}),
        None,
    )
    if command_index is None:
        # Programmatic callers occasionally preserve only a binary path.  A
        # deterministic fallback is safer than attaching options to an
        # unknown parser shape.
        command = ["agent-loop", target, str(identifier), "--repo", config.repo]
    else:
        source = command[command_index]
        remove = set()
        if target == "pr":
            remove.update(_ISSUE_ONLY_RECOVERY_OPTIONS)
            remove.update(_MANAGED_PR_ONLY_RECOVERY_OPTIONS)
        else:
            remove.update(_PR_ONLY_RECOVERY_OPTIONS)
            if source == "managed-pr":
                remove.update(_MANAGED_PR_ONLY_RECOVERY_OPTIONS)
        if not managed_ci:
            if not preserve_managed_options:
                remove.update(_MANAGED_RECOVERY_OPTIONS)
        if config.auto_merge:
            remove.update({"--watch-pending-ci", "--no-watch-pending-ci"})
        command = _strip_recovery_options(command, names=frozenset(remove))
        if source in {"issue", "pr"}:
            # Remove the source command's positional before inserting the
            # retargeted identifier. This also handles `issue --repo X 123`.
            positional_index = _find_recovery_positional_index(command, command_index)
            if positional_index is not None:
                del command[positional_index]
        command_index = next(
            index for index, token in enumerate(command) if token in {"issue", "pr", "managed-pr"}
        )
        command[command_index] = target
        if command_index + 1 >= len(command) or command[command_index + 1].startswith("-"):
            command.insert(command_index + 1, str(identifier))
        else:
            command[command_index + 1] = str(identifier)

    if managed_ci:
        if not _has_option(command, "--managed-ci"):
            command.append("--managed-ci")
        if config.managed_ci_trusted_actor:
            command = _strip_recovery_options(
                command, names=frozenset({"--managed-ci-trusted-actor"})
            )
            command.extend(("--managed-ci-trusted-actor", config.managed_ci_trusted_actor))
        if config.allow_unprotected_managed_ci and not _has_option(command, "--allow-unprotected-managed-ci"):
            command.append("--allow-unprotected-managed-ci")
    return shlex.join(command)


def render_managed_ci_resume_command(
    config: AgentLoopConfig,
    *,
    pr_number: int,
    managed_ci: bool,
    preserve_managed_options: bool = False,
    issue_number: int | None = None,
    include_context: bool = True,
) -> str:
    """Render the deterministic, parser-valid, shell-quoted recovery contract."""
    target: Literal["issue", "pr"] = "pr"
    identifier = pr_number
    if config.invocation_argv:
        command_index = next(
            (
                index
                for index, token in enumerate(config.invocation_argv)
                if token in {"issue", "pr", "managed-pr"}
            ),
            None,
        )
        if command_index is not None and config.invocation_argv[command_index] == "issue":
            target = "issue"
            identifier = issue_number if issue_number is not None else pr_number
            if issue_number is None:
                positional_index = _find_recovery_positional_index(
                    list(config.invocation_argv), command_index
                )
                if positional_index is not None:
                    identifier = int(config.invocation_argv[positional_index])
    return _render_recovery_command(
        config,
        target=target,
        identifier=identifier,
        managed_ci=managed_ci,
        preserve_managed_options=preserve_managed_options,
        include_context=include_context,
    )


def _restore_ordinary_ci_after_v2_fallback(
    runner: Runner, *, config: AgentLoopConfig, pr_number: int, reason: str
) -> None:
    """Remove the issue-created suppression label before returning to ordinary CI."""
    result = runner.run(
        [
            config.gh_cmd, "api", "--method", "DELETE",
            f"repos/{config.repo}/issues/{pr_number}/labels/{MANAGED_LABEL}",
        ],
        cwd=active_workdir(config), check=False,
    )
    if result.returncode != 0:
        raise AgentLoopError(
            f"Managed-CI v2 could not activate ({reason}) and `{MANAGED_LABEL}` could not be removed. "
            "Remove the label before relying on ordinary CI."
        )
    log(config, f"PR #{pr_number}: removed `{MANAGED_LABEL}`; continuing with ordinary CI ({reason})")
    if config.managed_ci:
        raise AgentLoopError(
            f"--managed-ci requested qualification, but activation failed ({reason}). "
            f"PR #{pr_number} is now draft and unlabeled; this run did NOT qualify its head. "
            "Rerun the same managed-CI command to start a fresh cycle."
        )


def _release_for_ordinary_recovery(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    base_ref: str,
    expected_head_sha: str,
    active_event: tuple[int, str, int] | None,
    reason: str,
    recovery_capable: bool,
) -> OrdinaryRecoveryCapability | None:
    """Release the exact active label and return a narrowly scoped capability.

    The capability is created only after the caller has already authenticated
    the managed draft tuple (same-repository branch, trusted author, draft,
    label, base, and exact head) and the caller is handling a temporarily
    missing timeline event. GitHub can return an incomplete event history
    immediately after a label transition, so that specific race may still use
    the authenticated tuple as provenance. A readable event owned by another
    identity remains fail-closed: the label can be removed to restore ordinary
    CI, but the caller must not receive a capability to ready or merge. The
    later exact-head ordinary-CI gate remains mandatory before readiness or
    merge.
    """
    if not recovery_capable:
        log(
            config,
            f"PR #{pr_number}: ordinary recovery was not selected because the base workflow "
            "does not prove an unlabeled pull_request trigger",
        )
        return None
    prior_run_ids: set[int] = set()
    try:
        prior_run_ids = _workflow_run_ids(runner, config=config, head_sha=expected_head_sha)
    except AgentLoopError:
        log(config, f"PR #{pr_number}: could not baseline ordinary recovery runs ({reason})")
    if active_event is not None:
        current_event = _active_managed_label_event(runner, config=config, pr_number=pr_number)
        if current_event != active_event:
            raise AgentLoopError(
                f"PR #{pr_number} managed-label ownership changed before ordinary release; "
                "leaving the label untouched and no merge will be attempted."
            )
    result = runner.run(
        [
            config.gh_cmd, "api", "--method", "DELETE",
            f"repos/{config.repo}/issues/{pr_number}/labels/{MANAGED_LABEL}",
        ],
        cwd=active_workdir(config), check=False,
    )
    if result.returncode != 0:
        raise AgentLoopError(
            f"Managed-CI v2 could not activate ({reason}) and `{MANAGED_LABEL}` could not be removed."
        )
    log(config, f"PR #{pr_number}: selected ordinary unlabeled recovery ({reason})")
    if config.managed_ci:
        raise AgentLoopError(
            f"--managed-ci requested qualification, but activation failed ({reason}). "
            f"PR #{pr_number} is now draft and unlabeled; this run did NOT qualify its head. "
            "The previously advertised manual-merge state is suspended. Rerun the same "
            "managed-CI command to qualify a fresh live head."
        )
    return OrdinaryRecoveryCapability(
        pr_number=pr_number,
        repository=config.repo,
        base_ref=base_ref,
        expected_head_sha=expected_head_sha,
        released_label_event_id=active_event[0] if active_event is not None else None,
        released_at=int(time.time()),
        prior_run_ids=frozenset(prior_run_ids),
    )


def refresh_ordinary_recovery_capability(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    capability: OrdinaryRecoveryCapability,
) -> OrdinaryRecoveryCapability | None:
    """Rebind ordinary recovery to the live draft head before finalization.

    Review rounds may push commits after the managed label was released.  The
    original capability must not reject that expected progress, nor may it
    authorize a different PR or a relabeled PR.  A newly observed head is
    newer than this invocation's release observation, so its run baseline
    starts empty and only runs observed for that exact head are eligible.
    """
    pr = _api_json(
        runner, config, f"repos/{config.repo}/pulls/{capability.pr_number}", quiet=True,
    )
    head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
    base = pr.get("base") if isinstance(pr.get("base"), dict) else {}
    head_repo = head.get("repo") if isinstance(head.get("repo"), dict) else {}
    live_head = head.get("sha") if isinstance(head.get("sha"), str) else None
    if not live_head or not isinstance(head_repo.get("full_name"), str):
        return None
    if (
        pr.get("state") not in {None, "open", "OPEN"}
        or pr.get("draft") is not True
        or base.get("ref") != capability.base_ref
        or head_repo["full_name"].casefold() != capability.repository.casefold()
        or _active_managed_label_event(runner, config=config, pr_number=capability.pr_number) is not None
    ):
        return None
    if live_head == capability.expected_head_sha:
        return capability
    return replace(
        capability,
        expected_head_sha=live_head,
        prior_run_ids=frozenset(),
    )


def _parse_override_audit(body: str) -> dict[str, str] | None:
    """Parse the documented richer audit form through the shared validator."""
    try:
        record = parse_managed_ci_override_record(
            body,
            surface=PR_COMMENT_SURFACE,
            schema="audit",
            required=False,
        )
    except AgentLoopError:
        return None
    return None if record is None else record.field_map()


def _find_resume_audit(
    runner: Runner, *, config: AgentLoopConfig, pr_number: int, actor_login: str, actor_id: int,
    base_ref: str,
) -> tuple[int, dict[str, str]] | None:
    """Find exactly one actor-owned old issue audit for resume provenance."""
    comments = _api_list(runner, config, f"repos/{config.repo}/issues/{pr_number}/comments?per_page=100")
    if comments is None:
        return None
    candidates: list[tuple[int, dict[str, str]]] = []
    malformed = False
    for comment in comments:
        body = comment.get("body") if isinstance(comment.get("body"), str) else ""
        if UNPROTECTED_OVERRIDE_TRAILER not in body:
            continue
        parsed = _parse_override_audit(body)
        user = comment.get("user") if isinstance(comment.get("user"), dict) else {}
        if parsed is None or user.get("login") != actor_login or user.get("id") != actor_id:
            malformed = True
            continue
        if parsed["repo"].casefold() != config.repo.casefold() or parsed["base"] != base_ref:
            malformed = True
            continue
        cid = comment.get("id")
        if not isinstance(cid, int):
            malformed = True
            continue
        candidates.append((cid, parsed))
    if malformed or not candidates:
        return None
    # Multiple valid audits are normal after a safe retry. The newest actor-
    # owned record is the latest provenance, while all malformed/mismatched
    # records still fail closed above.
    return sorted(candidates, key=lambda item: item[0])[-1]


def _activate_v2_managed_ci(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    metadata: PullRequestMetadata,
    managed_resume: AuthenticatedManagedResume | None = None,
    resume_origin: Literal["issue-created", "source-managed"] | None = None,
) -> ManagedCiContract | None:
    """Validate the non-forgeable v2 opening tuple without mutating a PR.

    Labels and branch names are only continuity/convention signals here.  The
    REST author identity is checked against both the configured actor and the
    repository Actions variable, which is not contributor-editable.
    """
    configured = (config.managed_ci_trusted_actor or "").strip()
    if not configured:
        return None
    who = _api_json(runner, config, "user")
    actor_login = who.get("login") if isinstance(who.get("login"), str) else None
    actor_id = who.get("id") if isinstance(who.get("id"), int) else None
    if not actor_login or actor_id is None or actor_login.casefold() != configured.casefold():
        return None
    variable = _api_json(
        runner, config, f"repos/{config.repo}/actions/variables/AGENT_LOOP_MANAGED_ACTOR", quiet=True
    )
    advertised = variable.get("value") if isinstance(variable.get("value"), str) else None
    if not advertised or advertised.casefold() != actor_login.casefold():
        return None

    # Resolve workflow source from the protected base ref.  The dispatch later
    # uses the same ref, so PR-authored YAML cannot redefine qualification.
    base_ref = metadata.base_branch or config.base
    if not base_ref:
        return None
    base_workflow = runner.run(
        [
            config.gh_cmd,
            "api",
            f"repos/{config.repo}/contents/.github/workflows/{WORKFLOW_FILE}?ref={base_ref}",
            "-H",
            "Accept: application/vnd.github.raw+json",
        ],
        cwd=active_workdir(config),
        check=False,
    )
    if base_workflow.returncode != 0:
        return None
    workflow_text = base_workflow.stdout or ""
    if any(marker not in workflow_text for marker in V2_FEATURE_MARKERS):
        raise AgentLoopError("The base branch does not contain the complete managed-CI v2 workflow.")
    ordinary_recovery_capable = (
        RECOVERY_MARKER in workflow_text and "pull_request" in workflow_text and "unlabeled" in workflow_text
    )

    pr = _api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}")
    head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
    base = pr.get("base") if isinstance(pr.get("base"), dict) else {}
    author = pr.get("user") if isinstance(pr.get("user"), dict) else {}
    labels = {
        item.get("name") for item in (pr.get("labels") or [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    head_repo = head.get("repo") if isinstance(head.get("repo"), dict) else {}
    head_repo_name = head_repo.get("full_name") if isinstance(head_repo.get("full_name"), str) else None
    live_sha = head.get("sha") if isinstance(head.get("sha"), str) else None
    head_ref = head.get("ref") if isinstance(head.get("ref"), str) else None
    author_login = author.get("login") if isinstance(author.get("login"), str) else None
    author_id = author.get("id") if isinstance(author.get("id"), int) else None
    # Reserved branches are generated by the typed creation prompt. A mere
    # lookalike branch cannot pass without all the other authenticated fields.
    immutable_tuple = (
        head_repo_name is not None and head_repo_name.casefold() == config.repo.casefold()
        and author_login is not None and author_login.casefold() == actor_login.casefold()
        and author_id == actor_id
        and isinstance(head_ref, str) and head_ref.startswith("agent-loop/managed-")
        and base.get("ref") == base_ref
        and live_sha is not None and live_sha == metadata.head_sha
        and pr.get("state") not in {"closed", "CLOSED"}
    )
    if not immutable_tuple:
        return None

    # Explicit existing-PR adoption has a separate tuple and label handshake;
    # never let the issue-created lifecycle classifier intercept it.
    if (
        managed_resume is None
        and resume_origin is None
        and config.managed_ci_adopt_existing_pr
    ):
        return None

    # Keep the low-level activation API safe for callers that enter through the
    # public PR mode without first calling the orchestration recovery helper.
    # The immutable tuple above is still authenticated before this lifecycle
    # is selected; the orchestrator supplies the richer resume record itself.
    if managed_resume is None and config.managed_ci_pr_mode:
        if pr.get("draft") is False and MANAGED_LABEL not in labels:
            lifecycle = "ready-unlabeled-reentry"
        elif pr.get("draft") is True and MANAGED_LABEL in labels:
            lifecycle = "draft-labeled"
        elif pr.get("draft") is True and MANAGED_LABEL not in labels:
            lifecycle = "draft-unlabeled-reentry"
        else:
            # The orchestration recovery path reports mixed lifecycle states;
            # retain the low-level API's historical no-match result for an
            # unrecognized pre-authenticated PR.
            return None
        managed_resume = AuthenticatedManagedResume(
            origin="issue-created", lifecycle=lifecycle,
        )

    origin = (
        managed_resume.origin
        if managed_resume is not None
        else resume_origin
        or ("source-managed" if config.pr_origin_flow == "managed-pr" else "issue-created")
    )
    lifecycle = managed_resume.lifecycle if managed_resume is not None else "draft-labeled"

    # A successful explicit manual run leaves a managed PR ready and
    # unlabeled. Re-entry is a privileged mutation: it is allowed only when
    # the authenticated lifecycle record and the invocation both request
    # managed CI explicitly. Implicit auto-merge must leave this state alone.
    if lifecycle == "ready-unlabeled-reentry":
        if not config.managed_ci:
            command = render_managed_ci_resume_command(
                config, pr_number=pr_number, managed_ci=True,
            )
            raise AgentLoopError(
                f"PR #{pr_number} is an authenticated managed {origin} PR in the ready/unlabeled "
                "re-entry state. It was left unchanged; rerun with explicit `--managed-ci`: "
                f"{command}"
            )
        undo = runner.run(
            [config.gh_cmd, "pr", "ready", "--undo", str(pr_number), "--repo", config.repo],
            cwd=active_workdir(config), check=False,
        )
        refreshed = _api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}", quiet=True)
        refreshed_labels = {
            item.get("name") for item in (refreshed.get("labels") or [])
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        if undo.returncode != 0 or refreshed.get("draft") is not True or MANAGED_LABEL in refreshed_labels:
            raise AgentLoopError(
                f"--managed-ci re-entry could not make PR #{pr_number} draft and unlabeled; "
                "its prior qualified/manual-merge state remains unchanged and no qualification was claimed."
            )
        pr = refreshed
        head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
        base = pr.get("base") if isinstance(pr.get("base"), dict) else {}
        author = pr.get("user") if isinstance(pr.get("user"), dict) else {}
        head_repo = head.get("repo") if isinstance(head.get("repo"), dict) else {}
        labels = {
            item.get("name") for item in (pr.get("labels") or [])
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        live_sha = head.get("sha") if isinstance(head.get("sha"), str) else None
        if (
            pr.get("draft") is not True
            or pr.get("state") in {"closed", "CLOSED"}
            or base.get("ref") != base_ref
            or live_sha != metadata.head_sha
            or head.get("ref") != head_ref
            or head_repo.get("full_name", "").casefold() != config.repo.casefold()
            or author.get("login") != actor_login
            or author.get("id") != actor_id
            or MANAGED_LABEL in labels
        ):
            # There is no trusted active label to remove in the unlabeled
            # re-entry case. Report the draft/unlabeled state without claiming
            # qualification; the next invocation can retry reconstruction.
            raise AgentLoopError(
                f"--managed-ci re-entry reconstruction for PR #{pr_number} failed after "
                "the ready-to-draft transition; the head is NOT qualified by this run. "
                "The PR must be draft and unlabeled before rerunning managed qualification."
            )
    elif lifecycle == "draft-unlabeled-reentry" and (
        pr.get("draft") is not True or MANAGED_LABEL in labels
    ):
        draft = pr.get("draft")
        if draft is True:
            state = "draft/labeled"
        elif draft is False:
            state = "ready/labeled" if MANAGED_LABEL in labels else "ready/unlabeled"
        else:
            state = "unknown/labeled" if MANAGED_LABEL in labels else "unknown/unlabeled"
        command = render_managed_ci_resume_command(config, pr_number=pr_number, managed_ci=True)
        raise AgentLoopError(
            f"Managed-CI {origin} draft re-entry for PR #{pr_number} observed {state}; "
            f"the PR was left unchanged. Resume with `{command}` after restoring draft/unlabeled state."
        )
    elif lifecycle == "draft-unlabeled-reentry" and not config.managed_ci:
        command = render_managed_ci_resume_command(config, pr_number=pr_number, managed_ci=True)
        raise AgentLoopError(
            f"PR #{pr_number} is an authenticated managed {origin} draft/unlabeled re-entry. "
            f"It was left unchanged; rerun with explicit `--managed-ci`: {command}"
        )
    elif lifecycle != "draft-unlabeled-reentry" and (
        pr.get("draft") is not True or MANAGED_LABEL not in labels
    ):
        if managed_resume is None and resume_origin is None:
            return None
        draft = pr.get("draft")
        if draft is True:
            state = "draft/labeled" if MANAGED_LABEL in labels else "draft/unlabeled"
        elif draft is False:
            state = "ready/labeled" if MANAGED_LABEL in labels else "ready/unlabeled"
        else:
            state = "unknown/labeled" if MANAGED_LABEL in labels else "unknown/unlabeled"
        command = render_managed_ci_resume_command(config, pr_number=pr_number, managed_ci=True)
        raise AgentLoopError(
            f"Managed-CI {origin} resume for PR #{pr_number} observed {state}; "
            f"expected draft/labeled, draft/unlabeled, or authenticated ready/unlabeled. "
            f"The PR was left unchanged. Resume with `{command}`."
        )
    if MANAGED_LABEL not in labels:
        if not ensure_managed_label(runner, config=config):
            raise AgentLoopError(f"Unable to create the `{MANAGED_LABEL}` label.")
        applied_label = runner.run(
            [
                config.gh_cmd, "api", "--method", "POST",
                f"repos/{config.repo}/issues/{pr_number}/labels",
                "-f", f"labels[]={MANAGED_LABEL}",
            ], cwd=active_workdir(config), check=False,
        )
        if applied_label.returncode != 0:
            raise AgentLoopError(f"Unable to apply `{MANAGED_LABEL}` to PR #{pr_number}.")
        label_applied = True
        labels.add(MANAGED_LABEL)
    else:
        label_applied = False

    # A direct `pr` retry has a separate, deliberately narrower contract. It
    # may resume only an issue-created draft whose immutable timeline facts
    # still identify this trusted actor. The old body nonce is provenance, not
    # authorization; this invocation mints a new nonce and intent generation.
    active_event: tuple[int, str, int] | None = None
    if config.managed_ci_pr_mode or config.managed_ci:
        active_event = _active_managed_label_event(runner, config=config, pr_number=pr_number)
        if active_event is None:
            recovery = _release_for_ordinary_recovery(
                runner, config=config, pr_number=pr_number, base_ref=base_ref,
                expected_head_sha=live_sha, active_event=active_event,
                reason="the active managed-label event is temporarily unreadable",
                recovery_capable=ordinary_recovery_capable,
            )
            return ManagedCiContract(
                activation_path="ordinary_fallback",
                ordinary_recovery=recovery,
            )
        if active_event[1].casefold() != actor_login.casefold() or active_event[2] != actor_id:
            _release_for_ordinary_recovery(
                runner, config=config, pr_number=pr_number, base_ref=base_ref,
                expected_head_sha=live_sha, active_event=active_event,
                reason="the active managed-label event is not actor-owned",
                recovery_capable=ordinary_recovery_capable,
            )
            return ManagedCiContract(
                activation_path="ordinary_fallback",
                ordinary_recovery=None,
            )

    # Only workflows with a pull_request route can suppress an opening matrix.
    # Legacy dispatch-only v2 deployments remain compatible, while modern
    # suppression-capable workflows must retain strict protection or supply a
    # live, explicit waiver and its auditable PR trailer.
    protection = ProtectionAssessment("strict", "legacy", "dispatch-only legacy workflow")
    override_nonce: str | None = None
    resume_audit_id: int | None = None
    resume_provenance_head: str | None = None
    if "pull_request" in workflow_text:
        protection = assess_exact_head_protection(
            runner,
            context=ManagedCiProbeContext(config.repo, config.gh_cmd, active_workdir(config)),
            base=base_ref,
        )
        if protection.state != "strict" and managed_resume is not None:
            if not config.allow_unprotected_managed_ci or protection.state not in {"voluntary", "plan_limited"}:
                recovery = _release_for_ordinary_recovery(
                    runner, config=config, pr_number=pr_number, base_ref=base_ref,
                    expected_head_sha=live_sha, active_event=active_event,
                    reason="strict protection is unavailable and the explicit waiver is absent",
                    recovery_capable=ordinary_recovery_capable,
                )
                return ManagedCiContract(
                    activation_path="ordinary_fallback", ordinary_recovery=recovery,
                )
            prior_audit = _find_resume_audit(
                runner, config=config, pr_number=pr_number,
                actor_login=actor_login, actor_id=actor_id, base_ref=base_ref,
            )
            if prior_audit is None:
                recovery = _release_for_ordinary_recovery(
                    runner, config=config, pr_number=pr_number, base_ref=base_ref,
                    expected_head_sha=live_sha, active_event=active_event,
                    reason="no unambiguous actor-owned issue-created override audit exists",
                    recovery_capable=ordinary_recovery_capable,
                )
                return ManagedCiContract(activation_path="ordinary_fallback", ordinary_recovery=recovery)
            resume_audit_id, prior_fields = prior_audit
            resume_provenance_head = prior_fields.get("head")
        elif protection.state != "strict":
            body = pr.get("body") if isinstance(pr.get("body"), str) else ""
            if not config.allow_unprotected_managed_ci or protection.state not in {"voluntary", "plan_limited"}:
                _restore_ordinary_ci_after_v2_fallback(
                    runner, config=config, pr_number=pr_number,
                    reason="strict protection or the explicit override is unavailable",
                )
                return None
            try:
                override = parse_managed_ci_override_record(
                    body,
                    surface=PR_BODY_SURFACE,
                    schema="body",
                    required=True,
                    expected_nonce=config.managed_ci_expected_override_nonce,
                )
            except AgentLoopError as error:
                _restore_ordinary_ci_after_v2_fallback(
                    runner, config=config, pr_number=pr_number,
                    reason=str(error),
                )
                return None
            assert override is not None
            override_nonce = override.nonce
            audit_body = TrustedBody.canonical(
                (
                    f"{UNPROTECTED_OVERRIDE_TRAILER} nonce={override_nonce} repo={config.repo} "
                    f"base={base_ref} head={live_sha} protection={protection.state}\n\n"
                    "Voluntary gate: GitHub cannot prevent manual merges, other automation, "
                    "compromised credentials, or an agent-loop defect from bypassing it."
                ),
                expected_tokens=(UNPROTECTED_OVERRIDE_TRAILER,),
            )
            audit_body.validate_for_surface(PR_COMMENT_SURFACE)
            audit = runner.run(
                [
                    config.gh_cmd, "api", "--method", "POST", f"repos/{config.repo}/issues/{pr_number}/comments",
                    "-f", "body=" + str(audit_body),
                ], cwd=active_workdir(config), check=False,
            )
            if audit.returncode != 0:
                _restore_ordinary_ci_after_v2_fallback(
                    runner, config=config, pr_number=pr_number,
                    reason="the override audit comment could not be recorded",
                )
                return None
            try:
                audit_id = json.loads(audit.stdout or "{}").get("id")
            except json.JSONDecodeError:
                audit_id = None
            if not isinstance(audit_id, int):
                _restore_ordinary_ci_after_v2_fallback(
                    runner, config=config, pr_number=pr_number,
                    reason="the override audit comment response was malformed",
                )
                return None
        else:
            audit_id = None
    else:
        audit_id = None

    if managed_resume is not None:
        # Re-read both the PR and the active timeline event immediately before
        # the audit write. A successful earlier probe cannot authorize a raced
        # head/base/draft/label transition.
        live_pr = _api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}", quiet=True)
        live_head = live_pr.get("head") if isinstance(live_pr.get("head"), dict) else {}
        live_base = live_pr.get("base") if isinstance(live_pr.get("base"), dict) else {}
        live_author = live_pr.get("user") if isinstance(live_pr.get("user"), dict) else {}
        live_event = _active_managed_label_event(runner, config=config, pr_number=pr_number)
        live_labels = {
            item.get("name") for item in (live_pr.get("labels") or [])
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        if (
            live_pr.get("state") not in {None, "open", "OPEN"}
            or live_pr.get("draft") is not True
            or live_base.get("ref") != base_ref
            or live_head.get("sha") != live_sha
            or live_head.get("ref") != head_ref
            or live_head.get("repo", {}).get("full_name", "").casefold() != config.repo.casefold()
            or live_author.get("login") != actor_login
            or live_author.get("id") != actor_id
            or MANAGED_LABEL not in live_labels
            or live_pr.get("body") != pr.get("body")
            or live_event != active_event
        ):
            recovery = _release_for_ordinary_recovery(
                runner, config=config, pr_number=pr_number, base_ref=base_ref,
                expected_head_sha=live_sha, active_event=live_event,
                reason="the immutable resume tuple changed before activation",
                recovery_capable=ordinary_recovery_capable,
            )
            return ManagedCiContract(activation_path="ordinary_fallback", ordinary_recovery=recovery)
        if protection.state != "strict":
            override_nonce = secrets.token_urlsafe(24)
            resume_body = (
                f"{UNPROTECTED_OVERRIDE_TRAILER} nonce={override_nonce} repo={config.repo} "
                f"base={base_ref} head={live_sha} protection={protection.state} "
                f"active_label_event_id={active_event[0]} resume_from={resume_audit_id} "
                f"provenance_head={resume_provenance_head or 'unknown'} generation={secrets.token_urlsafe(12)}\n\n"
                "Resume provenance only: the prior issue-created audit is not an authorization token."
            )
            trusted_resume_body = TrustedBody.canonical(
                resume_body,
                expected_tokens=(UNPROTECTED_OVERRIDE_TRAILER,),
            )
            trusted_resume_body.validate_for_surface(PR_COMMENT_SURFACE)
            audit = runner.run(
                [
                    config.gh_cmd, "api", "--method", "POST",
                    f"repos/{config.repo}/issues/{pr_number}/comments", "-f", f"body={trusted_resume_body}",
                ], cwd=active_workdir(config), check=False,
            )
            if audit.returncode != 0:
                recovery = _release_for_ordinary_recovery(
                    runner, config=config, pr_number=pr_number, base_ref=base_ref,
                    expected_head_sha=live_sha, active_event=active_event,
                    reason="the fresh resume audit could not be recorded",
                    recovery_capable=ordinary_recovery_capable,
                )
                return ManagedCiContract(activation_path="ordinary_fallback", ordinary_recovery=recovery)
            try:
                audit_id = json.loads(audit.stdout or "{}").get("id")
            except json.JSONDecodeError:
                audit_id = None
            if not isinstance(audit_id, int):
                recovery = _release_for_ordinary_recovery(
                    runner, config=config, pr_number=pr_number, base_ref=base_ref,
                    expected_head_sha=live_sha, active_event=active_event,
                    reason="the fresh resume audit response was malformed",
                    recovery_capable=ordinary_recovery_capable,
                )
                return ManagedCiContract(activation_path="ordinary_fallback", ordinary_recovery=recovery)
        else:
            audit_id = None
    workflow_revision = _api_json(runner, config, f"repos/{config.repo}/commits/{base_ref}", quiet=True)
    revision = workflow_revision.get("sha") if isinstance(workflow_revision.get("sha"), str) else None
    generation = secrets.token_urlsafe(16) if (managed_resume is not None or config.managed_ci) else None
    log(
        config,
        f"PR #{pr_number}: selected managed resume (fresh generation)"
        if managed_resume is not None
        else f"PR #{pr_number}: activated authenticated managed exact-head CI v2",
    )
    return ManagedCiContract(
        protocol_version=2,
        base_ref=base_ref,
        trusted_actor_login=actor_login,
        trusted_actor_id=actor_id,
        workflow_revision=revision,
        protection_mode=protection.state,
        audit_nonce=override_nonce,
        audit_comment_id=audit_id if isinstance(audit_id, int) else None,
        intent_generation=generation,
        active_label_event_id=active_event[0] if active_event is not None else None,
        issue_created_pr=origin == "issue-created",
        invocation_applied_label=label_applied,
        ordinary_recovery_capable=ordinary_recovery_capable,
        origin=origin,
        lifecycle=lifecycle,
        authenticated_resume=managed_resume,
    )


def _api_list(runner: Runner, config: AgentLoopConfig, endpoint: str) -> list[dict[str, object]] | None:
    """Fetch a paginated GitHub list, returning None for an uninspectable response."""
    result = runner.run(
        [config.gh_cmd, "api", "--paginate", endpoint], cwd=active_workdir(config), check=False
    )
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list):
        return None
    # GitHub CLI 2.45 emits one flat array for paginated array endpoints.
    # Never silently discard malformed entries: doing so could turn an
    # incomplete timeline into an apparently complete ownership record.
    if not all(isinstance(item, dict) for item in payload):
        return None
    return payload


def _active_managed_label_event(
    runner: Runner, *, config: AgentLoopConfig, pr_number: int
) -> tuple[int, str, int] | None:
    """Return the last label application, but only while it remains active.

    Timeline provenance is an enforcement input for adoption.  Treat missing
    actor IDs, malformed pagination, and an intervening unlabel as untrusted.
    """
    events = _api_list(
        runner, config, f"repos/{config.repo}/issues/{pr_number}/events?per_page=100"
    )
    if events is None:
        return None
    latest: dict[str, object] | None = None
    for event in events:
        label = event.get("label") if isinstance(event.get("label"), dict) else {}
        if label.get("name") == MANAGED_LABEL and event.get("event") in {"labeled", "unlabeled"}:
            latest = event
    if latest is None or latest.get("event") != "labeled":
        return None
    actor = latest.get("actor") if isinstance(latest.get("actor"), dict) else {}
    event_id = latest.get("id")
    login, actor_id = actor.get("login"), actor.get("id")
    if not isinstance(event_id, int) or not isinstance(login, str) or not isinstance(actor_id, int):
        return None
    return event_id, login, actor_id


def _has_exact_head_protection(runner: Runner, *, config: AgentLoopConfig, base_ref: str) -> bool:
    """Compatibility predicate for adoption; strict means non-bypassable."""
    return assess_exact_head_protection(
        runner,
        context=ManagedCiProbeContext(config.repo, config.gh_cmd, active_workdir(config)),
        base=base_ref,
    ).state == "strict"


def _publish_adoption_guard(
    runner: Runner, *, config: AgentLoopConfig, head_sha: str
) -> bool:
    result = runner.run(
        [
            config.gh_cmd, "api", "--method", "POST", f"repos/{config.repo}/statuses/{head_sha}",
            "-f", "state=pending", "-f", f"context={FINAL_CONTEXT}",
            "-f", "description=Managed CI adoption awaits exact-head qualification",
        ], cwd=active_workdir(config), check=False,
    )
    return result.returncode == 0


def ensure_managed_label(runner: Runner, *, config: AgentLoopConfig) -> bool:
    """Create the repository label only when GitHub reports it absent."""
    existing = runner.run(
        [config.gh_cmd, "api", f"repos/{config.repo}/labels/{MANAGED_LABEL}"],
        cwd=active_workdir(config), check=False,
    )
    if existing.returncode == 0:
        return True
    created = runner.run(
        [
            config.gh_cmd, "api", "--method", "POST", f"repos/{config.repo}/labels",
            "-f", f"name={MANAGED_LABEL}", "-f", "color=1f6feb",
            "-f", "description=Suppress intermediate CI; agent-loop dispatches exact-head final CI",
        ], cwd=active_workdir(config), check=False,
    )
    return created.returncode == 0


def _adoption_identity(
    runner: Runner, *, config: AgentLoopConfig
) -> tuple[str, int] | None:
    configured = (config.managed_ci_trusted_actor or "").strip()
    if not configured:
        return None
    who = _api_json(runner, config, "user", quiet=True)
    login, actor_id = who.get("login"), who.get("id")
    advertised = _api_json(
        runner, config, f"repos/{config.repo}/actions/variables/AGENT_LOOP_MANAGED_ACTOR", quiet=True
    ).get("value")
    if (
        not isinstance(login, str) or not isinstance(actor_id, int)
        or not isinstance(advertised, str)
        or login.casefold() != configured.casefold()
        or login.casefold() != advertised.casefold()
    ):
        return None
    return login, actor_id


def _activate_v2_existing_pr_adoption(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    metadata: PullRequestMetadata,
) -> ManagedCiContract | None:
    """Explicitly adopt an already-open same-repository PR into v2.

    All mutations occur only after capability, identity, branch protection and
    stable-head checks.  The workflow remains the security boundary; this
    client-side handshake is deliberately fail-closed hygiene.
    """
    identity = _adoption_identity(runner, config=config)
    if identity is None:
        return None
    actor_login, actor_id = identity
    base_ref = metadata.base_branch or config.base
    if not base_ref:
        return None
    workflow = runner.run(
        [config.gh_cmd, "api", f"repos/{config.repo}/contents/.github/workflows/{WORKFLOW_FILE}?ref={base_ref}",
         "-H", "Accept: application/vnd.github.raw+json"],
        cwd=active_workdir(config), check=False,
    )
    if workflow.returncode != 0:
        return None
    source = workflow.stdout or ""
    # Incomplete optional adoption advertisement is unsupported rather than a
    # reason to break the existing issue-created v2 route.
    if any(marker not in source for marker in (*V2_FEATURE_MARKERS, *V2_ADOPTION_FEATURE_MARKERS)):
        return None
    pr = _api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}", quiet=True)
    head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
    base = pr.get("base") if isinstance(pr.get("base"), dict) else {}
    head_repo = head.get("repo") if isinstance(head.get("repo"), dict) else {}
    labels = {
        item.get("name") for item in (pr.get("labels") or [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    live_sha, head_ref = head.get("sha"), head.get("ref")
    if (
        pr.get("state") not in {None, "open"}
        or not isinstance(head_repo.get("full_name"), str)
        or head_repo["full_name"].casefold() != config.repo.casefold()
        or not isinstance(head_ref, str) or not head_ref
        or not isinstance(live_sha, str) or live_sha != metadata.head_sha
        or base.get("ref") != base_ref
        or MANAGED_OPT_OUT_LABEL in labels
        or not _has_exact_head_protection(runner, config=config, base_ref=base_ref)
    ):
        return None
    # Publish the required blocking context before any possible suppression.
    if not _publish_adoption_guard(runner, config=config, head_sha=live_sha):
        return None
    existing = _active_managed_label_event(runner, config=config, pr_number=pr_number)
    applied = False
    if MANAGED_LABEL not in labels:
        if not ensure_managed_label(runner, config=config):
            return None
        created = runner.run(
            [config.gh_cmd, "api", "--method", "POST", f"repos/{config.repo}/issues/{pr_number}/labels",
             "-f", f"labels[]={MANAGED_LABEL}"], cwd=active_workdir(config), check=False,
        )
        if created.returncode != 0:
            return None
        applied = True
        existing = _active_managed_label_event(runner, config=config, pr_number=pr_number)
    if existing is None or existing[1].casefold() != actor_login.casefold() or existing[2] != actor_id:
        # A newly-created but unprovable label must not remain as our claimed
        # suppression.  The release helper fresh-checks event ownership.
        provisional = ManagedCiContract(
            protocol_version=2, adopted_existing_pr=True, active_label_event_id=existing[0] if existing else None,
            invocation_applied_label=applied,
        )
        release_adopted_managed_ci(runner, config=config, pr_number=pr_number, contract=provisional)
        return None
    revision = _api_json(runner, config, f"repos/{config.repo}/commits/{base_ref}", quiet=True).get("sha")
    log(config, f"PR #{pr_number}: activated authenticated managed exact-head CI v2 adoption")
    return ManagedCiContract(
        protocol_version=2, base_ref=base_ref, trusted_actor_login=actor_login,
        trusted_actor_id=actor_id, workflow_revision=revision if isinstance(revision, str) else None,
        adopted_existing_pr=True, guard_head_sha=live_sha, active_label_event_id=existing[0],
        invocation_applied_label=applied,
        intent_generation=(
            secrets.token_urlsafe(16)
            if config.managed_ci
            else None
        ),
    )


def revalidate_adopted_managed_ci(
    runner: Runner, *, config: AgentLoopConfig, pr_number: int, metadata: PullRequestMetadata,
    contract: ManagedCiContract,
) -> bool:
    """Recheck adoption security inputs before filtering or final dispatch."""
    if not contract.adopted_existing_pr or not contract.base_ref:
        return True
    pr = _api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}", quiet=True)
    head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
    labels = {
        item.get("name") for item in (pr.get("labels") or [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    live_sha = head.get("sha")
    if (
        not isinstance(live_sha, str) or live_sha != metadata.head_sha
        or MANAGED_OPT_OUT_LABEL in labels
        or MANAGED_LABEL not in labels
        or not _has_exact_head_protection(runner, config=config, base_ref=contract.base_ref)
    ):
        return False
    event = _active_managed_label_event(runner, config=config, pr_number=pr_number)
    if (
        event is None or event[0] != contract.active_label_event_id
        or event[1].casefold() != (contract.trusted_actor_login or "").casefold()
        or event[2] != contract.trusted_actor_id
    ):
        return False
    if contract.guard_head_sha != live_sha:
        if not _publish_adoption_guard(runner, config=config, head_sha=live_sha):
            return False
        contract.guard_head_sha = live_sha
    return True


def release_adopted_managed_ci(
    runner: Runner, *, config: AgentLoopConfig, pr_number: int, contract: ManagedCiContract,
    force: bool = False,
) -> bool:
    """Remove an invocation-owned adoption label without removing a later label.

    If post-apply provenance is unreadable, the label cannot safely remain as
    our claimed suppression.  It is then removed unconditionally; when an
    event ID was recorded, retain the normal fresh event-ID ownership check.
    """
    if not (
        contract.adopted_existing_pr
        or contract.issue_created_pr
        or contract.origin == "source-managed"
    ) or (
        not contract.invocation_applied_label and not force
    ):
        return True
    if contract.active_label_event_id is not None:
        event = _active_managed_label_event(runner, config=config, pr_number=pr_number)
        if event is None or event[0] != contract.active_label_event_id:
            return True
    result = runner.run(
        [config.gh_cmd, "api", "--method", "DELETE", f"repos/{config.repo}/issues/{pr_number}/labels/{MANAGED_LABEL}"],
        cwd=active_workdir(config), check=False,
    )
    return result.returncode == 0


def _release_managed_label_for_manual_qualification(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    contract: ManagedCiContract,
) -> None:
    """Release only the authenticated active suppression before manual exit."""
    event = _active_managed_label_event(runner, config=config, pr_number=pr_number)
    if (
        event is None
        or contract.active_label_event_id is None
        or event[0] != contract.active_label_event_id
        or event[1].casefold() != (contract.trusted_actor_login or "").casefold()
        or event[2] != contract.trusted_actor_id
    ):
        raise AgentLoopError(
            f"PR #{pr_number} managed-label provenance changed before manual qualification; "
            "the head is not qualified and no manual merge command is safe."
        )
    result = runner.run(
        [
            config.gh_cmd, "api", "--method", "DELETE",
            f"repos/{config.repo}/issues/{pr_number}/labels/{MANAGED_LABEL}",
        ], cwd=active_workdir(config), check=False,
    )
    if result.returncode != 0:
        raise AgentLoopError(
            f"PR #{pr_number} qualified CI passed, but `{MANAGED_LABEL}` could not be removed; "
            "the PR remains suppressed and must not be manually merged until the label is removed."
        )


def publish_manual_v2_qualification(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    expected_head_sha: str,
    contract: ManagedCiContract,
    reviewers: tuple[str, ...],
) -> str:
    """Publish a SHA-bound manual result, release suppression, and ready the PR."""
    if contract.protocol_version != 2:
        raise AgentLoopError("Explicit managed-CI manual qualification requires protocol v2.")
    if get_pr_head_sha(runner, config, pr_number) != expected_head_sha:
        raise AgentLoopError(
            f"PR #{pr_number} head changed before manual qualification publication; no merge is safe."
        )

    # The bare qualified label is intentionally not used for a manual result.
    # Remove a historical copy if one exists so it cannot outlive its SHA.
    pr = _api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}")
    labels = {
        item.get("name") for item in (pr.get("labels") or [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if QUALIFIED_LABEL in labels:
        removed = runner.run(
            [
                config.gh_cmd, "api", "--method", "DELETE",
                f"repos/{config.repo}/issues/{pr_number}/labels/{QUALIFIED_LABEL}",
            ], cwd=active_workdir(config), check=False,
        )
        if removed.returncode != 0:
            raise AgentLoopError(f"Unable to clear stale `{QUALIFIED_LABEL}` from PR #{pr_number}.")

    _release_managed_label_for_manual_qualification(
        runner, config=config, pr_number=pr_number, contract=contract,
    )
    after_release = _api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}")
    after_head = after_release.get("head") if isinstance(after_release.get("head"), dict) else {}
    after_labels = {
        item.get("name") for item in (after_release.get("labels") or [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if (
        after_head.get("sha") != expected_head_sha
        or MANAGED_LABEL in after_labels
        or (contract.adopted_existing_pr and after_release.get("draft") is True)
    ):
        raise AgentLoopError(
            f"PR #{pr_number} changed while releasing managed CI; its exact head is not safely published."
        )

    if not contract.adopted_existing_pr:
        ready = runner.run(
            [config.gh_cmd, "pr", "ready", str(pr_number), "--repo", config.repo],
            cwd=active_workdir(config), check=False,
        )
        if ready.returncode != 0:
            raise AgentLoopError(f"Unable to mark qualified PR #{pr_number} ready for manual review.")
        after_ready = _api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}")
        ready_head = after_ready.get("head") if isinstance(after_ready.get("head"), dict) else {}
        ready_labels = {
            item.get("name") for item in (after_ready.get("labels") or [])
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        if ready_head.get("sha") != expected_head_sha or after_ready.get("draft") is True or MANAGED_LABEL in ready_labels:
            raise AgentLoopError(
                f"PR #{pr_number} changed while being made ready; the approved head is not safely published."
            )

    run_text = str(contract.attached_run_id) if contract.attached_run_id is not None else "unknown"
    attempt_text = str(contract.run_attempt) if contract.run_attempt is not None else "unknown"
    reviewer_text = ",".join(reviewers) or "unknown"
    body = (
        f"<!-- {QUALIFICATION_MARKER} repo={config.repo} pr={pr_number} base={contract.base_ref or config.base} "
        f"protocol=2 qualified_head={expected_head_sha} reviewers={reviewer_text} "
        f"protection={contract.protection_mode or 'unknown'} nonce={contract.nonce or 'unknown'} "
        f"run_id={run_text} attempt={attempt_text} generation={contract.intent_generation or 'unknown'} -->\n\n"
        f"Managed exact-head CI qualified `{expected_head_sha}` for manual merge. "
        "The managed suppression label was released and this SHA is the only advertised merge target."
    )
    if contract.protection_mode != "strict":
        body += (
            " GitHub cannot force a human or other automation to use this SHA or the guarded command "
            "after agent-loop exits."
        )
    trusted_body = TrustedBody.canonical(body, expected_tokens=(QUALIFICATION_MARKER,))
    trusted_body.validate_for_surface(PR_COMMENT_SURFACE)
    posted = runner.run(
        [
            config.gh_cmd, "api", "--method", "POST",
            f"repos/{config.repo}/issues/{pr_number}/comments", "-f", f"body={trusted_body}",
        ], cwd=active_workdir(config), check=False,
    )
    if posted.returncode != 0:
        raise AgentLoopError(f"Unable to publish the SHA-bound qualification audit for PR #{pr_number}.")
    try:
        audit_id = json.loads(posted.stdout or "{}").get("id")
    except json.JSONDecodeError:
        audit_id = None
    if not isinstance(audit_id, int):
        raise AgentLoopError(f"Qualification audit for PR #{pr_number} returned no comment ID.")
    contract.audit_comment_id = audit_id
    if get_pr_head_sha(runner, config, pr_number) != expected_head_sha:
        raise AgentLoopError(
            f"PR #{pr_number} head changed after qualification publication; rerun review and exact-head CI."
        )
    return expected_head_sha


def _api_json(runner: Runner, config: AgentLoopConfig, endpoint: str, *, quiet: bool = False) -> dict[str, object]:
    result = runner.run(
        [config.gh_cmd, "api", endpoint], cwd=active_workdir(config), check=False
    )
    if result.returncode != 0:
        if quiet:
            return {}
        raise AgentLoopError(f"Managed-CI API request failed: {endpoint}.")
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise AgentLoopError(f"Managed-CI API response was invalid JSON: {endpoint}.") from exc
    return payload if isinstance(payload, dict) else {}


def managed_label_present(runner: Runner, *, config: AgentLoopConfig, pr_number: int) -> bool | None:
    """Return None when label state cannot be proven, never treating it as safe."""
    pr = _api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}", quiet=True)
    if not pr:
        return None
    labels = pr.get("labels")
    if not isinstance(labels, list):
        return None
    return any(isinstance(label, dict) and label.get("name") == MANAGED_LABEL for label in labels)


def intermediate_managed_checks(checks: PullRequestChecks) -> PullRequestChecks:
    """Hide contract-controlled absence while reviewers inspect an intermediate head.

    The managed route intentionally does not create the repository's test
    matrix contexts, so branch protection reports those contexts as missing
    until the exact-head workflow is dispatched. Actually observed non-final
    checks remain visible, especially failures from independent integrations.
    """
    pending = tuple(check for check in checks.pending if check.name != FINAL_CONTEXT)
    failing = tuple(check for check in checks.failing if check.name != FINAL_CONTEXT)
    passing = tuple(check for check in checks.passing if check.name != FINAL_CONTEXT)
    required = tuple(name for name in checks.required_checks if name != FINAL_CONTEXT)
    if failing:
        state = "failing"
    elif pending:
        state = "pending"
    elif passing:
        state = "passing"
    elif checks.check_query_status == "unavailable":
        state = "unavailable"
    else:
        state = "no_checks"
    return replace(
        checks,
        state=state,
        required_checks=required,
        passing=passing,
        pending=pending,
        failing=failing,
        missing_required=(),
        infrastructure_stalls=tuple(
            stall for stall in checks.infrastructure_stalls if stall.name != FINAL_CONTEXT
        ),
    )


def publish_round_readiness(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    head_sha: str,
) -> None:
    result = runner.run(
        [
            config.gh_cmd,
            "api",
            "--method",
            "POST",
            f"repos/{config.repo}/statuses/{head_sha}",
            "-f",
            "state=success",
            "-f",
            f"context={READINESS_CONTEXT}",
            "-f",
            "description=Configured local pre-review verification passed",
        ],
        cwd=active_workdir(config),
        check=False,
    )
    if result.returncode != 0:
        raise AgentLoopError(f"Unable to publish `{READINESS_CONTEXT}` for {head_sha}.")


def dispatch_final_qualification(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    expected_head_sha: str,
    head_ref: str,
    contract: ManagedCiContract,
) -> None:
    live_head = get_pr_head_sha(runner, config, pr_number)
    if live_head != expected_head_sha:
        raise AgentLoopError(
            f"PR #{pr_number} head moved from approved SHA {expected_head_sha} "
            f"to {live_head} before CI dispatch."
        )
    if contract.protocol_version == 2:
        _dispatch_v2_qualification(
            runner,
            config=config,
            pr_number=pr_number,
            expected_head_sha=expected_head_sha,
            contract=contract,
        )
        return
    result = runner.run(
        [
            config.gh_cmd,
            "api",
            "--method",
            "POST",
            f"repos/{config.repo}/actions/workflows/{contract.workflow_file}/dispatches",
            "-f",
            f"ref={head_ref}",
            "-f",
            f"inputs[pr_number]={pr_number}",
            "-f",
            f"inputs[expected_head_sha]={expected_head_sha}",
        ],
        cwd=active_workdir(config),
        check=False,
    )
    if result.returncode != 0:
        raise AgentLoopError(
            f"Unable to dispatch managed final CI for PR #{pr_number} at {expected_head_sha}."
        )
    log(config, f"PR #{pr_number}: dispatched managed final CI at {expected_head_sha}")


def _dispatch_v2_qualification(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    expected_head_sha: str,
    contract: ManagedCiContract,
) -> None:
    """Create/resume one authenticated intent and converge duplicate dispatches.

    GitHub dispatches are asynchronous: discovering first prevents normal
    retries from creating a second run, while selecting the newest surviving
    run makes a crash between dispatch and ledger PATCH deterministic.
    """
    if not contract.base_ref or not contract.trusted_actor_login or contract.trusted_actor_id is None:
        raise AgentLoopError("Managed-CI v2 contract is missing authenticated dispatch provenance.")
    current_base = _api_json(
        runner, config, f"repos/{config.repo}/commits/{contract.base_ref}", quiet=True
    ).get("sha")
    if (
        contract.workflow_revision
        and isinstance(current_base, str)
        and current_base != contract.workflow_revision
    ):
        raise AgentLoopError(
            "The managed-CI base workflow revision changed after activation; rerun review before dispatch."
        )
    try:
        _ensure_v2_intent(
            runner, config=config, pr_number=pr_number, expected_head_sha=expected_head_sha, contract=contract
        )
    except AgentLoopError as exc:
        if not config.managed_ci_pr_mode and contract.authenticated_resume is None:
            raise
        event = _active_managed_label_event(runner, config=config, pr_number=pr_number)
        recovery = _release_for_ordinary_recovery(
            runner, config=config, pr_number=pr_number, base_ref=contract.base_ref,
            expected_head_sha=expected_head_sha, active_event=event,
            reason=f"fresh intent ledger could not be reconciled ({exc})",
            recovery_capable=contract.ordinary_recovery_capable,
        )
        contract.activation_path = "ordinary_fallback"
        contract.ordinary_recovery = recovery
        log(config, f"PR #{pr_number}: managed resume ledger failed; selected ordinary recovery")
        return
    attached = _discover_v2_run(runner, config=config, contract=contract, retries=3)
    if attached is None:
        _patch_intent(runner, config=config, contract=contract, state="dispatch-requested")
        result = runner.run(
            [
                config.gh_cmd,
                "api",
                "--method",
                "POST",
                f"repos/{config.repo}/actions/workflows/{contract.workflow_file}/dispatches",
                "-f",
                f"ref={contract.base_ref}",
                "-f",
                "inputs[protocol_version]=2",
                "-f",
                f"inputs[pr_number]={pr_number}",
                "-f",
                f"inputs[expected_head_sha]={expected_head_sha}",
                "-f",
                f"inputs[managed_nonce]={contract.nonce}",
            ],
            cwd=active_workdir(config),
            check=False,
        )
        if result.returncode != 0:
            raise AgentLoopError(
                f"Unable to dispatch managed final CI for PR #{pr_number} at {expected_head_sha}."
            )
        attached = _discover_v2_run(runner, config=config, contract=contract, retries=3)
    if attached is not None:
        contract.attached_run_id = attached[0]
        contract.run_attempt = attached[1]
        _patch_intent(runner, config=config, contract=contract, state="attached")
        log(config, f"PR #{pr_number}: attached managed-CI v2 run {attached[0]}")
    else:
        # Do not guess a run ID. The waiter will continue bounded discovery;
        # an uncorrelated green status is never accepted in the meantime.
        log(config, f"PR #{pr_number}: dispatch accepted; waiting for run visibility")


def _intent_body(contract: ManagedCiContract, *, pr_number: int, expected_head_sha: str, state: str) -> TrustedBody:
    payload = {
        "version": 2,
        "repository": contract.repository,
        "pr": pr_number,
        "expected_head_sha": expected_head_sha,
        "base_ref": contract.base_ref,
        "workflow_revision": contract.workflow_revision,
        "generation": contract.intent_generation,
        "nonce": contract.nonce,
        "created_at": contract.created_at,
        "state": state,
        "run_id": contract.attached_run_id,
        "run_attempt": contract.run_attempt,
        "terminal_run_id": contract.terminal_run_id,
        "terminal_run_attempt": contract.terminal_run_attempt,
        "terminal_attempts": [
            {"run_id": run_id, "run_attempt": run_attempt}
            for run_id, run_attempt in contract.terminal_attempts
        ],
    }
    return TrustedBody.canonical(
        f"<!-- {INTENT_MARKER} {json.dumps(payload, separators=(',', ':'), sort_keys=True)} -->",
        expected_tokens=(INTENT_MARKER,),
    )


def _ensure_v2_intent(
    runner: Runner, *, config: AgentLoopConfig, pr_number: int, expected_head_sha: str, contract: ManagedCiContract
) -> None:
    contract.pr_number = pr_number
    contract.expected_head_sha = expected_head_sha
    contract.repository = config.repo
    comments = _api_list(
        runner, config, f"repos/{config.repo}/issues/{pr_number}/comments?per_page=100"
    )
    if comments is None:
        raise AgentLoopError("Unable to inspect managed-CI v2 intent history.")
    matching: list[tuple[int, dict[str, object]]] = []
    for raw in comments:
        body = raw.get("body")
        author = raw.get("user") if isinstance(raw.get("user"), dict) else {}
        if not isinstance(body, str) or INTENT_MARKER not in body:
            continue
        if author.get("login") != contract.trusted_actor_login or author.get("id") != contract.trusted_actor_id:
            continue
        try:
            encoded = body.split(INTENT_MARKER, 1)[1].split("-->", 1)[0].strip()
            intent = json.loads(encoded)
        except (IndexError, json.JSONDecodeError):
            continue
        if not isinstance(intent, dict):
            continue
        if intent.get("repository") != config.repo:
            continue
        if contract.intent_generation is not None and intent.get("generation") != contract.intent_generation:
            prior_run = intent.get("run_id")
            if isinstance(prior_run, int):
                log(
                    config,
                    f"PR #{pr_number}: superseding prior managed-CI generation run {prior_run}; "
                    "it will not be attached to this invocation",
                )
            continue
        if intent.get("pr") == pr_number and intent.get("expected_head_sha") == expected_head_sha:
            cid = raw.get("id")
            if isinstance(cid, int):
                matching.append((cid, intent))
    distinct = {str(item.get("nonce")) for _, item in matching}
    if len(distinct) > 1:
        raise AgentLoopError("Competing managed-CI v2 intent comments exist for this PR head.")
    if matching:
        contract.intent_comment_id, intent = matching[-1]
        nonce = intent.get("nonce")
        if not isinstance(nonce, str) or not nonce:
            raise AgentLoopError("Managed-CI v2 intent comment lacks a nonce.")
        contract.nonce = nonce
        contract.created_at = intent.get("created_at") if isinstance(intent.get("created_at"), int) else None
        contract.attached_run_id = intent.get("run_id") if isinstance(intent.get("run_id"), int) else None
        contract.run_attempt = intent.get("run_attempt") if isinstance(intent.get("run_attempt"), int) else None
        contract.intent_state = intent.get("state") if isinstance(intent.get("state"), str) else None
        terminal_run_id = intent.get("terminal_run_id")
        terminal_run_attempt = intent.get("terminal_run_attempt")
        encoded_attempts = intent.get("terminal_attempts")
        terminal_attempts: list[tuple[int, int | None]] = []
        if isinstance(encoded_attempts, list):
            for encoded in encoded_attempts:
                if not isinstance(encoded, dict) or not isinstance(encoded.get("run_id"), int):
                    continue
                encoded_attempt = encoded.get("run_attempt")
                terminal_attempts.append(
                    (encoded["run_id"], encoded_attempt if isinstance(encoded_attempt, int) else None)
                )
        if isinstance(terminal_run_id, int) and isinstance(terminal_run_attempt, int):
            contract.terminal_run_id = terminal_run_id
            contract.terminal_run_attempt = terminal_run_attempt
        if contract.intent_state == "terminal-no-status" and not terminal_attempts:
            if contract.attached_run_id is not None:
                contract.terminal_run_id = contract.attached_run_id
                contract.terminal_run_attempt = contract.run_attempt
        legacy_key = (contract.terminal_run_id, contract.terminal_run_attempt)
        if legacy_key[0] is not None and legacy_key not in terminal_attempts:
            terminal_attempts.append((legacy_key[0], legacy_key[1]))
        contract.terminal_attempts = tuple(terminal_attempts)
        if (
            contract.intent_state == "terminal-no-status"
            and (contract.attached_run_id, contract.run_attempt) in contract.terminal_attempts
        ):
            # The attached attempt is the one that already stopped. Leave
            # discovery responsible for finding a rerun or fresh dispatch.
            contract.attached_run_id = None
            contract.run_attempt = None
        return
    contract.nonce = secrets.token_urlsafe(24)
    contract.created_at = int(time.time())
    body = _intent_body(contract, pr_number=pr_number, expected_head_sha=expected_head_sha, state="prepared")
    body.validate_for_surface(PR_COMMENT_SURFACE)
    created = runner.run(
        [config.gh_cmd, "api", "--method", "POST", f"repos/{config.repo}/issues/{pr_number}/comments", "-f", f"body={body}"],
        cwd=active_workdir(config), check=False,
    )
    if created.returncode != 0:
        raise AgentLoopError("Unable to create managed-CI v2 intent ledger comment.")
    try:
        payload = json.loads(created.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise AgentLoopError("Managed-CI intent create response was invalid JSON.") from exc
    if not isinstance(payload.get("id"), int):
        raise AgentLoopError("Managed-CI intent comment was created without an ID.")
    contract.intent_comment_id = payload["id"]
    contract.intent_state = "prepared"


def _patch_intent(runner: Runner, *, config: AgentLoopConfig, contract: ManagedCiContract, state: str) -> None:
    if contract.intent_comment_id is None or contract.nonce is None:
        return
    body = _intent_body(
        contract,
        pr_number=contract.pr_number or 0,
        expected_head_sha=contract.expected_head_sha or "",
        state=state,
    )
    body.validate_for_surface(PR_COMMENT_SURFACE)
    # The immutable identifying fields are already in the original marker;
    # state updates add only live provenance and are still actor-owned.
    updated = runner.run(
        [config.gh_cmd, "api", "--method", "PATCH", f"repos/{config.repo}/issues/comments/{contract.intent_comment_id}", "-f", f"body={body}"],
        cwd=active_workdir(config), check=False,
    )
    if updated.returncode != 0:
        raise AgentLoopError("Unable to persist managed-CI v2 intent state.")
    contract.intent_state = state


def _discover_v2_snapshot(
    runner: Runner, *, config: AgentLoopConfig, contract: ManagedCiContract,
    retries: int, include_terminal: bool = False,
) -> ManagedCiRunSnapshot | None:
    if not contract.nonce:
        return None
    run_name = _v2_run_name(contract)
    for attempt in range(retries):
        result = runner.run(
            [config.gh_cmd, "api", f"repos/{config.repo}/actions/workflows/{contract.workflow_file}/runs?event=workflow_dispatch&per_page=100"],
            cwd=active_workdir(config), check=False,
        )
        if result.returncode == 0:
            try:
                runs = json.loads(result.stdout or "{}").get("workflow_runs", [])
            except (json.JSONDecodeError, AttributeError):
                runs = []
            candidates = [
                run
                for run in runs
                if isinstance(run, dict) and _is_v2_intent_run(run, contract=contract, run_name=run_name)
            ]
            survivors = [
                run for run in candidates
                if include_terminal or run.get("status") != "completed"
                or run.get("conclusion") != "cancelled"
            ]
            excluded_attempts = set(contract.terminal_attempts)
            if contract.terminal_run_id is not None:
                excluded_attempts.add((contract.terminal_run_id, contract.terminal_run_attempt))
            if excluded_attempts:
                survivors = [run for run in survivors if not _v2_terminal_attempt_excluded(
                    run.get("id"), run.get("run_attempt"), excluded_attempts
                )]
            if survivors:
                newest = sorted(survivors, key=lambda run: (str(run.get("created_at") or ""), int(run.get("id") or 0)))[-1]
                rid = newest.get("id")
                if isinstance(rid, int):
                    attempt_no = newest.get("run_attempt")
                    return ManagedCiRunSnapshot(
                        rid,
                        attempt_no if isinstance(attempt_no, int) else None,
                        newest.get("status") if isinstance(newest.get("status"), str) else None,
                        newest.get("conclusion") if isinstance(newest.get("conclusion"), str) else None,
                    )
        if attempt < retries - 1:
            runner.run(["sleep", str(min(config.ci_poll_interval_seconds, 2))], cwd=active_workdir(config))
    return None


def _discover_v2_run(
    runner: Runner, *, config: AgentLoopConfig, contract: ManagedCiContract, retries: int,
) -> tuple[int, int | None] | None:
    snapshot = _discover_v2_snapshot(
        runner, config=config, contract=contract, retries=retries, include_terminal=False
    )
    return (snapshot.run_id, snapshot.run_attempt) if snapshot is not None else None


def _refresh_v2_attached_run(
    runner: Runner, *, config: AgentLoopConfig, contract: ManagedCiContract
) -> ManagedCiRunSnapshot | None:
    """Read the current attempt for the already-attached workflow run.

    GitHub increments ``run_attempt`` when a run is re-run.  The run ID remains
    the durable correlation key, so status validation must follow that current
    attempt rather than the attempt present when agent-loop dispatched it.
    """
    if contract.attached_run_id is None or not contract.nonce:
        return None
    result = runner.run(
        [
            config.gh_cmd,
            "api",
            f"repos/{config.repo}/actions/runs/{contract.attached_run_id}",
        ],
        cwd=active_workdir(config),
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        run = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None
    if not isinstance(run, dict) or run.get("id") != contract.attached_run_id:
        return None
    run_name = _v2_run_name(contract)
    if not _is_v2_intent_run(run, contract=contract, run_name=run_name):
        return None
    attempt = run.get("run_attempt")
    return ManagedCiRunSnapshot(
        contract.attached_run_id,
        attempt if isinstance(attempt, int) else contract.run_attempt,
        run.get("status") if isinstance(run.get("status"), str) else None,
        run.get("conclusion") if isinstance(run.get("conclusion"), str) else None,
    )


def _v2_run_name(contract: ManagedCiContract) -> str:
    return f"managed-ci-v2 nonce={contract.nonce}"


def _is_v2_intent_run(
    run: dict[str, object], *, contract: ManagedCiContract, run_name: str
) -> bool:
    """Validate the API fields available for a dispatched base workflow."""
    if (
        run_name not in (run.get("name"), run.get("display_title"))
        or run.get("event") != "workflow_dispatch"
    ):
        return False
    path = run.get("path")
    if isinstance(path, str) and f".github/workflows/{WORKFLOW_FILE}" not in path:
        return False
    if isinstance(run.get("head_branch"), str) and contract.base_ref and run["head_branch"] != contract.base_ref:
        return False
    if isinstance(run.get("head_sha"), str) and contract.workflow_revision and run["head_sha"] != contract.workflow_revision:
        return False
    for field in ("actor", "triggering_actor"):
        identity = run.get(field)
        if not isinstance(identity, dict):
            continue
        login = identity.get("login")
        actor_id = identity.get("id")
        if (
            login != contract.trusted_actor_login
            or actor_id != contract.trusted_actor_id
        ):
            return False
    return True


def wait_for_final_qualification(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    metadata: PullRequestMetadata,
    contract: ManagedCiContract | None = None,
) -> ManagedCiOutcome:
    expected_head = metadata.head_sha
    if not expected_head:
        raise AgentLoopError(f"PR #{pr_number} has no approved head SHA.")
    attempts = max(1, config.ci_timeout_seconds // config.ci_poll_interval_seconds)
    latest: PullRequestChecks | None = None
    terminal_confirmation: tuple[int, int | None] | None = None
    terminal_confirmation_seen = False
    for attempt in range(attempts):
        live_head = get_pr_head_sha(runner, config, pr_number)
        if live_head != expected_head:
            return ManagedCiOutcome(status="head_changed", checks=latest, head_sha=live_head)
        mergeability = get_pr_mergeability(runner, config=config, pr_number=pr_number)
        if mergeability.state == "conflicted":
            return ManagedCiOutcome(
                status="merge_conflict",
                checks=latest,
                mergeability=mergeability,
                head_sha=live_head,
            )
        latest = get_pr_checks(runner, config=config, metadata=metadata)
        run_snapshot: ManagedCiRunSnapshot | None = None
        if contract is not None and contract.protocol_version == 2:
            if contract.attached_run_id is None:
                run_snapshot = _discover_v2_snapshot(
                    runner, config=config, contract=contract, retries=1, include_terminal=True
                )
                if run_snapshot is not None:
                    contract.attached_run_id, contract.run_attempt = run_snapshot.run_id, run_snapshot.run_attempt
                    _patch_intent(runner, config=config, contract=contract, state="attached")
            else:
                run_snapshot = _refresh_v2_attached_run(
                    runner, config=config, contract=contract
                )
                if run_snapshot is not None and (
                    run_snapshot.run_attempt is not None
                    and run_snapshot.run_attempt != contract.run_attempt
                ):
                    contract.run_attempt = run_snapshot.run_attempt
                    _patch_intent(runner, config=config, contract=contract, state="attached")
            final = _v2_correlated_status(
                runner, config=config, expected_head=expected_head, contract=contract
            )
        else:
            final = _find_context(latest, FINAL_CONTEXT)
        status = final.status.lower() if final is not None else "pending"
        log(config, f"Managed CI context '{FINAL_CONTEXT}' status: {status}")
        complete_board = (
            latest.check_query_status == "ok"
            and latest.branch_protection_status in {"configured", "not_found", "forbidden"}
            and latest.state == "passing"
            and bool(latest.passing)
            and not latest.pending
            and not latest.missing_required
        )
        if status == "success" and complete_board:
            if contract is not None and contract.protocol_version == 2:
                _patch_intent(runner, config=config, contract=contract, state="completed")
            return ManagedCiOutcome(status="passed", checks=latest, head_sha=live_head)
        correlated_terminal = final is not None and final.status.lower() in _TERMINAL_CI_STATUSES
        if (
            contract is not None and contract.protocol_version == 2
            and run_snapshot is not None
            and (run_snapshot.status or "").lower() == "completed"
            and not correlated_terminal
        ):
            key = (run_snapshot.run_id, run_snapshot.run_attempt)
            if key != terminal_confirmation:
                terminal_confirmation = key
                terminal_confirmation_seen = False
                # A publisher may race the workflow's completed transition. Re-read
                # immediately before starting the bounded confirmation window.
                final = _v2_correlated_status(
                    runner, config=config, expected_head=expected_head, contract=contract
                )
                if final is not None and final.status.lower() in _TERMINAL_CI_STATUSES:
                    status = final.status.lower()
                    if status == "success" and complete_board:
                        _patch_intent(runner, config=config, contract=contract, state="completed")
                        return ManagedCiOutcome(status="passed", checks=latest, head_sha=live_head)
                else:
                    terminal_confirmation_seen = True
            elif terminal_confirmation_seen:
                contract.terminal_run_id = run_snapshot.run_id
                contract.terminal_run_attempt = run_snapshot.run_attempt
                terminal_key = (run_snapshot.run_id, run_snapshot.run_attempt)
                if terminal_key not in contract.terminal_attempts:
                    contract.terminal_attempts += (terminal_key,)
                _patch_intent(runner, config=config, contract=contract, state="terminal-no-status")
                return ManagedCiOutcome(
                    status="terminal_without_status", checks=latest, head_sha=live_head,
                    run_id=run_snapshot.run_id, run_attempt=run_snapshot.run_attempt,
                    workflow_status=run_snapshot.status, workflow_conclusion=run_snapshot.conclusion,
                )
        if status in _TERMINAL_CI_STATUSES - {"success"}:
            details = ()
            if contract is not None and contract.protocol_version == 2:
                _patch_intent(runner, config=config, contract=contract, state="completed")
                details = _v2_failed_jobs(runner, config=config, run_id=contract.attached_run_id)
            return ManagedCiOutcome(
                status="failed", checks=latest, head_sha=live_head, failure_details=details
            )
        stall_checks = intermediate_managed_checks(latest)
        if is_wholly_infrastructure_blocked(stall_checks):
            stall = CiInfrastructureStall(checks=stall_checks.infrastructure_stalls)
            return ManagedCiOutcome(
                status="infrastructure_stall",
                checks=latest,
                head_sha=live_head,
                stall=stall,
            )
        if attempt < attempts - 1:
            runner.run(["sleep", str(config.ci_poll_interval_seconds)], cwd=active_workdir(config))
    return ManagedCiOutcome(status="timeout", checks=latest, head_sha=expected_head)


def _v2_correlated_status(
    runner: Runner, *, config: AgentLoopConfig, expected_head: str, contract: ManagedCiContract
) -> PullRequestCheck | None:
    """Return only the status published by the attached, current run.

    A context name is deliberately insufficient: an older duplicate or a
    contributor-created status must remain pending, including if it is red.
    """
    if contract.attached_run_id is None or not contract.nonce:
        return None
    result = runner.run(
        [config.gh_cmd, "api", "--paginate", f"repos/{config.repo}/commits/{expected_head}/statuses?per_page=100"],
        cwd=active_workdir(config), check=False,
    )
    if result.returncode != 0:
        return None
    try:
        pages = _decode_paginated_json(result.stdout)
    except (json.JSONDecodeError, TypeError):
        return None
    statuses = [item for page in pages for item in page]
    required = {f"nonce={contract.nonce}", f"run_id={contract.attached_run_id}"}
    if isinstance(contract.run_attempt, int):
        required.add(f"attempt={contract.run_attempt}")
    excluded = _v2_terminal_attempt_excluded(
        contract.attached_run_id, contract.run_attempt, contract.terminal_attempts
    )
    matches: list[tuple[int, str | None, dict[str, object]]] = []
    for position, raw in enumerate(statuses):
        if not isinstance(raw, dict) or raw.get("context") != FINAL_CONTEXT:
            continue
        description = raw.get("description") if isinstance(raw.get("description"), str) else ""
        target = raw.get("target_url") if isinstance(raw.get("target_url"), str) else ""
        creator = raw.get("creator") if isinstance(raw.get("creator"), dict) else {}
        login = creator.get("login") if isinstance(creator.get("login"), str) else None
        creator_id = creator.get("id") if isinstance(creator.get("id"), int) else None
        allowed_publisher = (
            (login or "").casefold() == "github-actions[bot]"
            or (login == contract.trusted_actor_login and creator_id == contract.trusted_actor_id)
        )
        description_tokens = {token.strip() for token in description.split(";") if token.strip()}
        if not allowed_publisher or not required.issubset(description_tokens):
            continue
        target_path = urlparse(target).path.rstrip("/")
        target_segments = target_path.split("/")
        if (
            len(target_segments) < 3
            or target_segments[-3:-1] != ["actions", "runs"]
            or target_segments[-1] != str(contract.attached_run_id)
        ):
            continue
        if excluded:
            continue
        created_value = raw.get("created_at")
        created = created_value if isinstance(created_value, str) else None
        if created is not None:
            try:
                datetime.fromisoformat(created.replace("Z", "+00:00"))
            except ValueError:
                continue
        matches.append((position, created, raw))
    if not matches:
        return None
    def sort_key(item: tuple[int, str | None, dict[str, object]]) -> tuple[int, str, int]:
        pos, created, _ = item
        return (1 if created else 0, created or "", -pos)
    raw = sorted(matches, key=sort_key)[-1][2]
    description = raw.get("description") if isinstance(raw.get("description"), str) else ""
    target = raw.get("target_url") if isinstance(raw.get("target_url"), str) else ""
    creator = raw.get("creator") if isinstance(raw.get("creator"), dict) else {}
    login = creator.get("login") if isinstance(creator.get("login"), str) else None
    creator_id = creator.get("id") if isinstance(creator.get("id"), int) else None
    return PullRequestCheck(
        name=FINAL_CONTEXT,
        kind="status_context",
        status=str(raw.get("state") or "pending"),
        url=target or None,
        run_id=str(contract.attached_run_id),
        created_at=raw.get("created_at") if isinstance(raw.get("created_at"), str) else None,
        completed_at=raw.get("updated_at") if isinstance(raw.get("updated_at"), str) else None,
        creator_login=login,
        creator_id=creator_id,
        description=description,
    )


def _decode_paginated_json(text: str) -> list[list[object]]:
    """Decode gh --paginate output, including concatenated JSON arrays."""
    decoder = json.JSONDecoder()
    pages: list[list[object]] = []
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset >= len(text):
            break
        value, end = decoder.raw_decode(text, offset)
        if not isinstance(value, list):
            raise json.JSONDecodeError("page is not an array", text, offset)
        pages.append(value)
        offset = end
    return pages


def _v2_terminal_attempt_excluded(
    run_id: object, attempt: object, exclusions: object
) -> bool:
    if not isinstance(run_id, int) or not isinstance(exclusions, (set, tuple, list)):
        return False
    for entry in exclusions:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            continue
        rid, stored_attempt = entry
        if rid != run_id or not isinstance(rid, int):
            continue
        if isinstance(stored_attempt, int) and stored_attempt == attempt:
            return True
        if stored_attempt is None and attempt is None:
            return True
    return False


def _v2_failed_jobs(runner: Runner, *, config: AgentLoopConfig, run_id: int | None) -> tuple[str, ...]:
    if run_id is None:
        return ()
    result = runner.run(
        [config.gh_cmd, "api", "--paginate", f"repos/{config.repo}/actions/runs/{run_id}/jobs?filter=latest&per_page=100"],
        cwd=active_workdir(config), check=False,
    )
    if result.returncode != 0:
        return (f"Managed CI run {run_id} failed; job details were unavailable.",)
    raw = result.stdout or ""
    # Without --slurp, gh 2.45 concatenates object pages. Decode every object
    # instead of accepting only the first page (or treating the stream as bad).
    decoder = json.JSONDecoder()
    pages: list[dict[str, object]] = []
    offset = 0
    try:
        while offset < len(raw):
            while offset < len(raw) and raw[offset].isspace():
                offset += 1
            if offset == len(raw):
                break
            page, end = decoder.raw_decode(raw, offset)
            if not isinstance(page, dict):
                raise ValueError("jobs page is not an object")
            pages.append(page)
            offset = end
    except (json.JSONDecodeError, ValueError):
        return (f"Managed CI run {run_id} failed; job details were unavailable.",)
    jobs: list[object] = []
    for payload in pages:
        page_jobs = payload.get("jobs")
        if not isinstance(page_jobs, list):
            return (f"Managed CI run {run_id} failed; job details were unavailable.",)
        jobs.extend(page_jobs)
    terminal = {"failure", "cancelled", "timed_out", "action_required", "startup_failure"}
    details = []
    for job in jobs if isinstance(jobs, list) else []:
        if not isinstance(job, dict) or str(job.get("conclusion") or "").lower() not in terminal:
            continue
        name = job.get("name") if isinstance(job.get("name"), str) else "unnamed job"
        conclusion = str(job.get("conclusion"))
        url = job.get("html_url") if isinstance(job.get("html_url"), str) else ""
        details.append(f"{name}: {conclusion}" + (f" ({url})" if url else ""))
    return tuple(details) or (f"Managed CI run {run_id} failed; no failed job was exposed.",)


def prepare_v2_merge(
    runner: Runner, *, config: AgentLoopConfig, pr_number: int, expected_head_sha: str, contract: ManagedCiContract
) -> None:
    """Publish continuity before readying the PR, then re-check its exact head."""
    if contract.protocol_version != 2:
        return
    labelled = runner.run(
        [config.gh_cmd, "api", "--method", "POST", f"repos/{config.repo}/issues/{pr_number}/labels", "-f", f"labels[]={QUALIFIED_LABEL}"],
        cwd=active_workdir(config), check=False,
    )
    if labelled.returncode != 0:
        raise AgentLoopError(f"Unable to apply `{QUALIFIED_LABEL}` before readying PR #{pr_number}.")
    pr = _api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}")
    if pr.get("draft") is True:
        ready = runner.run(
            [config.gh_cmd, "pr", "ready", str(pr_number), "--repo", config.repo],
            cwd=active_workdir(config), check=False,
        )
        if ready.returncode != 0:
            raise AgentLoopError(f"Unable to mark qualified PR #{pr_number} ready for review.")
    if get_pr_head_sha(runner, config, pr_number) != expected_head_sha:
        raise AgentLoopError(f"PR #{pr_number} head changed while it was being readied for merge.")


def _find_context(checks: PullRequestChecks, name: str) -> PullRequestCheck | None:
    for check in (*checks.failing, *checks.pending, *checks.passing):
        if check.name == name:
            return check
    return None


def _workflow_run_ids(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    head_sha: str,
) -> set[int]:
    payload = _workflow_runs_payload(runner, config=config, head_sha=head_sha)
    return {
        run["id"]
        for run in payload
        if isinstance(run, dict) and isinstance(run.get("id"), int)
    }


def _workflow_runs_payload(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    head_sha: str,
) -> list[dict[str, object]]:
    result = runner.run(
        [
            config.gh_cmd,
            "api",
            f"repos/{config.repo}/actions/workflows/{WORKFLOW_FILE}/runs"
            f"?event=pull_request&head_sha={head_sha}&per_page=20",
        ],
        cwd=active_workdir(config),
        check=False,
    )
    if result.returncode != 0:
        raise AgentLoopError("Unable to inspect managed-CI workflow runs.")
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise AgentLoopError("Managed-CI workflow-run response was invalid JSON.") from exc
    runs = payload.get("workflow_runs")
    if not isinstance(runs, list):
        raise AgentLoopError("Managed-CI workflow-run response omitted `workflow_runs`.")
    return runs


def wait_for_ordinary_recovery(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    capability: OrdinaryRecoveryCapability,
    metadata: PullRequestMetadata,
) -> ManagedCiOutcome:
    """Wait for a post-unlabel current-head run and a complete ordinary board."""
    expected_head = capability.expected_head_sha
    attempts = max(1, config.ci_timeout_seconds // config.ci_poll_interval_seconds)
    startup_attempt_limit = max(
        1,
        (config.ci_startup_timeout_seconds + config.ci_poll_interval_seconds - 1)
        // config.ci_poll_interval_seconds,
    )
    latest: PullRequestChecks | None = None
    for attempt in range(attempts):
        live_head = get_pr_head_sha(runner, config, capability.pr_number)
        if live_head != expected_head:
            return ManagedCiOutcome(status="head_changed", checks=latest, head_sha=live_head)
        mergeability = get_pr_mergeability(runner, config=config, pr_number=capability.pr_number)
        if mergeability.state == "conflicted":
            return ManagedCiOutcome(
                status="merge_conflict", checks=latest, mergeability=mergeability, head_sha=live_head,
            )
        runs = _workflow_runs_payload(runner, config=config, head_sha=expected_head)
        recovery_runs = []
        for run in runs:
            if not isinstance(run, dict) or run.get("id") in capability.prior_run_ids:
                continue
            if run.get("event") not in {None, "pull_request"}:
                continue
            if isinstance(run.get("head_sha"), str) and run["head_sha"] != expected_head:
                continue
            recovery_runs.append(run)
        latest = get_pr_checks(runner, config=config, metadata=metadata)
        if latest.state == "failing":
            return ManagedCiOutcome(status="failed", checks=latest, head_sha=live_head)
        if is_wholly_infrastructure_blocked(latest):
            return ManagedCiOutcome(
                status="infrastructure_stall", checks=latest, head_sha=live_head,
                stall=CiInfrastructureStall(checks=latest.infrastructure_stalls),
            )
        completed = [run for run in recovery_runs if run.get("status") == "completed"]
        if completed and any(run.get("conclusion") != "success" for run in completed):
            return ManagedCiOutcome(status="failed", checks=latest, head_sha=live_head)
        reliable = latest.check_query_status == "ok" and latest.branch_protection_status in {
            "configured", "not_found", "forbidden",
        }
        if (
            recovery_runs
            and any(run.get("status") == "completed" and run.get("conclusion") == "success" for run in completed)
            and latest.state == "passing"
            and bool(latest.passing)
            and reliable
            and not latest.pending
            and not latest.missing_required
        ):
            log(config, f"PR #{capability.pr_number}: ordinary unlabeled recovery passed at {expected_head}")
            return ManagedCiOutcome(status="passed", checks=latest, head_sha=live_head)
        if not recovery_runs and attempt + 1 >= startup_attempt_limit:
            return ManagedCiOutcome(status="not_started", checks=latest, head_sha=live_head)
        if attempt < attempts - 1:
            runner.run(["sleep", str(config.ci_poll_interval_seconds)], cwd=active_workdir(config))
    return ManagedCiOutcome(status="timeout", checks=latest, head_sha=expected_head)


def validate_ordinary_recovery_capability(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    capability: OrdinaryRecoveryCapability,
    require_draft: bool | None = True,
) -> bool:
    """Revalidate the invocation-owned fallback tuple before readiness/merge."""
    pr = _api_json(runner, config, f"repos/{config.repo}/pulls/{capability.pr_number}", quiet=True)
    head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
    base = pr.get("base") if isinstance(pr.get("base"), dict) else {}
    head_repo = head.get("repo") if isinstance(head.get("repo"), dict) else {}
    event = _active_managed_label_event(
        runner, config=config, pr_number=capability.pr_number,
    )
    return bool(
        pr.get("state") in {None, "open"}
        and (require_draft is None or pr.get("draft") is require_draft)
        and base.get("ref") == capability.base_ref
        and isinstance(head_repo.get("full_name"), str)
        and head_repo["full_name"].casefold() == capability.repository.casefold()
        and head.get("sha") == capability.expected_head_sha
        and event is None
    )


def _wait_for_label_handoff(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    metadata: PullRequestMetadata,
    prior_run_ids: set[int],
) -> None:
    assert metadata.head_sha is not None
    attempts = max(1, config.ci_timeout_seconds // config.ci_poll_interval_seconds)
    for attempt in range(attempts):
        live_head = get_pr_head_sha(runner, config, pr_number)
        if live_head != metadata.head_sha:
            raise AgentLoopError(
                f"PR #{pr_number} head moved while waiting for the managed-label CI handoff."
            )
        runs = _workflow_runs_payload(runner, config=config, head_sha=metadata.head_sha)
        new_runs = [
            run
            for run in runs
            if isinstance(run.get("id"), int) and run["id"] not in prior_run_ids
        ]
        if any(run.get("status") == "completed" for run in new_runs):
            checks = get_pr_checks(runner, config=config, metadata=metadata)
            final = _find_context(checks, FINAL_CONTEXT)
            if final is None or final.status.lower() not in {"pending", "queued", "in_progress"}:
                raise AgentLoopError(
                    f"Managed-label handoff for PR #{pr_number} completed without publishing "
                    f"`{FINAL_CONTEXT}=pending`."
                )
            return
        if attempt < attempts - 1:
            runner.run(["sleep", str(config.ci_poll_interval_seconds)], cwd=active_workdir(config))
    raise AgentLoopError(
        f"Managed-label handoff for PR #{pr_number} did not complete within "
        f"{config.ci_timeout_seconds}s."
    )
