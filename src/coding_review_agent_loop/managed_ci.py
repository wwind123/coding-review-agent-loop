"""External driver for repositories that implement managed exact-head CI."""

from __future__ import annotations

import json
import secrets
import shlex
import time
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
)
from .logging import log
from .runner import Runner
from .workdirs import active_workdir


MANAGED_LABEL = "agent-loop-managed"
MANAGED_OPT_OUT_LABEL = "agent-loop-managed-opt-out"
QUALIFIED_LABEL = "agent-loop-exact-head-qualified"
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
    guard_head_sha: str | None = None
    active_label_event_id: int | None = None
    invocation_applied_label: bool = False
    protection_mode: str | None = None
    audit_nonce: str | None = None
    audit_comment_id: int | None = None


@dataclass(frozen=True)
class ManagedCiCreationIntent:
    """Instructions that make the opened event atomically recognizable as v2."""

    branch: str
    trusted_actor: str
    protection_mode: str = "strict"
    audit_nonce: str | None = None


@dataclass(frozen=True)
class ManagedCiOutcome:
    status: Literal[
        "passed",
        "failed",
        "timeout",
        "head_changed",
        "merge_conflict",
        "infrastructure_stall",
    ]
    checks: PullRequestChecks | None = None
    mergeability: PullRequestMergeability | None = None
    head_sha: str | None = None
    stall: CiInfrastructureStall | None = None
    failure_details: tuple[str, ...] = ()


def activate_managed_ci(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    metadata: PullRequestMetadata,
) -> ManagedCiContract | None:
    """Activate a repository-advertised managed-CI contract for auto-merge runs.

    Repositories without any contract markers retain legacy behavior. A partial
    contract fails closed because applying its suppression label could otherwise
    disable hosted tests without a usable final qualification path.
    """
    if not config.auto_merge or config.dry_run:
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
        return None
    workflow = result.stdout or ""
    present = tuple(marker for marker in _CONTRACT_MARKERS if marker in workflow)
    if not present:
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
        )
        if contract is not None or not config.managed_ci_adopt_existing_pr:
            return contract
        # A complete ordinary v2 workflow is not an adoption contract.  This
        # path is intentionally quiet/fallback-compatible when the optional
        # marker has not been deployed.
        return _activate_v2_existing_pr_adoption(
            runner,
            config=config,
            pr_number=pr_number,
            metadata=metadata,
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
    label_result = runner.run(
        [config.gh_cmd, "api", f"repos/{config.repo}/labels/{MANAGED_LABEL}"],
        cwd=active_workdir(config),
        check=False,
    )
    if label_result.returncode != 0:
        create_result = runner.run(
            [
                config.gh_cmd,
                "api",
                "--method",
                "POST",
                f"repos/{config.repo}/labels",
                "-f",
                f"name={MANAGED_LABEL}",
                "-f",
                "color=1f6feb",
                "-f",
                "description=Suppress intermediate CI; agent-loop dispatches exact-head final CI",
            ],
            cwd=active_workdir(config),
            check=False,
        )
        if create_result.returncode != 0:
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


def _probe_raw_workflow(runner: Runner, context: ManagedCiProbeContext, base: str) -> str | None:
    result = _probe(
        runner, context,
        f"repos/{context.repo}/contents/.github/workflows/{WORKFLOW_FILE}?ref={base}", raw=True,
    )
    return (result.stdout or "") if result.returncode == 0 else None


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
    if required is None:
        stderr = (required_result.stderr or "").lower()
        if "upgrade to github pro" in stderr or "make this repository public" in stderr:
            return ProtectionAssessment("plan_limited", "classic", "GitHub plan/API does not permit branch protection")
        if required_result.returncode == 404:
            return ProtectionAssessment("voluntary", "classic", "required status protection is not configured")
        return ProtectionAssessment("indeterminate", "classic", "required-status protection could not be inspected")

    classic_state: Literal["voluntary", "indeterminate"] | None = None
    classic_detail = "final-ci/exact-head is not independently required"
    if _has_final_context(required):
        admins, admins_result = _probe_json(
            runner, context, f"repos/{context.repo}/branches/{base}/protection/enforce_admins"
        )
        # Empty responses are accepted only for compatibility with older gh
        # test doubles; GitHub's real endpoint is a JSON object with enabled.
        if admins is not None and (not admins or admins.get("enabled") is True):
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

    rules, rules_result = _probe_json(runner, context, f"repos/{context.repo}/rules/branches/{base}")
    if rules is None:
        if rules_result.returncode in {404, 422}:
            return ProtectionAssessment(classic_state or "voluntary", "classic" if classic_state else "none", classic_detail)
        return ProtectionAssessment("indeterminate", "rulesets", "effective branch rules could not be inspected")
    entries = rules.get("rules") if isinstance(rules.get("rules"), list) else []
    voluntary = False
    for entry in entries:
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
    workflow = _probe_raw_workflow(runner, context, resolved_base)
    who, _ = _probe_json(runner, context, "user")
    variable, variable_result = _probe_json(runner, context, f"repos/{context.repo}/actions/variables/AGENT_LOOP_MANAGED_ACTOR")
    actor = who.get("login") if who and isinstance(who.get("login"), str) else None
    advertised = variable.get("value") if variable and isinstance(variable.get("value"), str) else None
    protection = assess_exact_head_protection(runner, context=context, base=resolved_base)
    if workflow is None or who is None or (variable is None and variable_result.returncode not in {404}):
        return ManagedCiReadiness("indeterminate", visibility, actor, advertised, False, False, protection,
            ("a required read-only GitHub probe failed",), ())
    core = V2_MARKER in workflow
    complete = core and all(marker in workflow for marker in V2_FEATURE_MARKERS)
    # Pre-marker v2 fixtures/workflows that contain no pull_request trigger at
    # all never suppress an opening matrix. Treat them as legacy-compatible;
    # any workflow that does opt into pull_request suppression must advertise
    # the explicit recovery marker and unlabeled activity.
    recovery = (RECOVERY_MARKER in workflow and "unlabeled" in workflow) or "pull_request" not in workflow
    identity_ok = bool(actor and advertised and trusted_actor.strip() and actor.casefold() == trusted_actor.casefold() == advertised.casefold())
    if not core:
        return ManagedCiReadiness("ordinary_fallback", visibility, actor, advertised, False, recovery, protection,
            ("base workflow does not advertise managed-CI v2",), ("deploy the documented managed-CI v2 workflow",))
    if not complete or not recovery:
        missing = "complete v2 contract" if not complete else "unlabeled recovery contract"
        return ManagedCiReadiness("invalid", visibility, actor, advertised, complete, recovery, protection,
            (f"workflow lacks the {missing}",), ("add the documented workflow markers and pull_request unlabeled trigger",))
    if not identity_ok:
        return ManagedCiReadiness("ordinary_fallback", visibility, actor, advertised, complete, recovery, protection,
            ("authenticated login, trusted actor, and AGENT_LOOP_MANAGED_ACTOR do not match",),
            (f"gh variable set AGENT_LOOP_MANAGED_ACTOR --repo {shlex.quote(context.repo)} --body {shlex.quote(trusted_actor)}",))
    if protection.state == "strict":
        return ManagedCiReadiness("strict_ready", visibility, actor, advertised, complete, recovery, protection, (), ())
    if protection.state in {"voluntary", "plan_limited"}:
        return ManagedCiReadiness("override_eligible", visibility, actor, advertised, complete, recovery, protection,
            (protection.detail,), ("configure non-bypassable final-ci/exact-head protection, or use --allow-unprotected-managed-ci for this invocation",))
    return ManagedCiReadiness("indeterminate", visibility, actor, advertised, complete, recovery, protection,
        (protection.detail,), ())


def render_managed_ci_preflight(result: ManagedCiReadiness, *, repo: str, base: str, trusted_actor: str) -> str:
    lines = [
        f"repository: {repo}", f"base: {base}", f"visibility: {result.repo_visibility or 'unknown'}",
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
    runner: Runner, *, config: AgentLoopConfig, issue_number: int
) -> ManagedCiCreationIntent | None:
    """Return an atomic creation intent only for an authenticated v2 rollout.

    This happens before the coder is prompted, avoiding the opened-event race:
    the PR is born as a draft with its managed label, or it is ordinary CI.
    """
    if not config.auto_merge or config.dry_run or not config.managed_ci_trusted_actor or not config.base:
        return None
    source_context = ManagedCiProbeContext(config.repo, config.gh_cmd, active_workdir(config))
    source_workflow = _probe_raw_workflow(runner, source_context, config.base)
    if source_workflow is None or V2_MARKER not in source_workflow:
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
        raise AgentLoopError("Managed-CI readiness could not be determined; refusing to suppress CI.")
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
            return None
        readiness = ManagedCiReadiness(
            "override_eligible", None, actor, advertised, True, True,
            ProtectionAssessment("voluntary", "legacy", "legacy non-suppressing workflow"), (), (),
        )
    if readiness.state == "override_eligible" and not (config.allow_unprotected_managed_ci or legacy_non_suppressing):
        return None
    if readiness.state != "strict_ready" and not (
        readiness.state == "override_eligible" and (config.allow_unprotected_managed_ci or legacy_non_suppressing)
    ):
        return None
    workflow = source_workflow
    if workflow is None or V2_MARKER not in workflow:
        return None
    if any(marker not in workflow for marker in V2_FEATURE_MARKERS):
        raise AgentLoopError("Repository CI advertises an incomplete managed-CI v2 contract.")
    return ManagedCiCreationIntent(
        branch=f"agent-loop/managed-{issue_number}",
        trusted_actor=readiness.actor or config.managed_ci_trusted_actor,
        protection_mode=readiness.protection.state,
        audit_nonce=secrets.token_urlsafe(18) if config.allow_unprotected_managed_ci else None,
    )


def _activate_v2_managed_ci(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    metadata: PullRequestMetadata,
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
    managed_tuple = (
        head_repo_name is not None and head_repo_name.casefold() == config.repo.casefold()
        and author_login is not None and author_login.casefold() == actor_login.casefold()
        and author_id == actor_id
        and isinstance(head_ref, str) and head_ref.startswith("agent-loop/managed-")
        and MANAGED_LABEL in labels
        and pr.get("draft") is True
        and base.get("ref") == base_ref
        and live_sha is not None and live_sha == metadata.head_sha
    )
    if not managed_tuple:
        return None
    # Only workflows with a pull_request route can suppress an opening matrix.
    # Legacy dispatch-only v2 deployments remain compatible, while modern
    # suppression-capable workflows must retain strict protection or supply a
    # live, explicit waiver and its auditable PR trailer.
    protection = ProtectionAssessment("strict", "legacy", "dispatch-only legacy workflow")
    override_nonce: str | None = None
    if "pull_request" in workflow_text:
        protection = assess_exact_head_protection(
            runner,
            context=ManagedCiProbeContext(config.repo, config.gh_cmd, active_workdir(config)),
            base=base_ref,
        )
        if protection.state != "strict":
            body = pr.get("body") if isinstance(pr.get("body"), str) else ""
            marker_line = next((line.strip() for line in body.splitlines() if line.strip().startswith(UNPROTECTED_OVERRIDE_TRAILER)), "")
            if not config.allow_unprotected_managed_ci or protection.state not in {"voluntary", "plan_limited"} or not marker_line:
                return None
            parts = marker_line.split("nonce=", 1)
            if len(parts) != 2 or not parts[1].strip():
                return None
            override_nonce = parts[1].strip().split()[0]
            audit = runner.run(
                [
                    config.gh_cmd, "api", "--method", "POST", f"repos/{config.repo}/issues/{pr_number}/comments",
                    "-f", "body=" + (
                        f"{UNPROTECTED_OVERRIDE_TRAILER} nonce={override_nonce} repo={config.repo} "
                        f"base={base_ref} head={live_sha} protection={protection.state}\n\n"
                        "Voluntary gate: GitHub cannot prevent manual merges, other automation, "
                        "compromised credentials, or an agent-loop defect from bypassing it."
                    ),
                ], cwd=active_workdir(config), check=False,
            )
            if audit.returncode != 0:
                return None
            try:
                audit_id = json.loads(audit.stdout or "{}").get("id")
            except json.JSONDecodeError:
                audit_id = None
        else:
            audit_id = None
    else:
        audit_id = None
    workflow_revision = _api_json(runner, config, f"repos/{config.repo}/commits/{base_ref}", quiet=True)
    revision = workflow_revision.get("sha") if isinstance(workflow_revision.get("sha"), str) else None
    log(config, f"PR #{pr_number}: activated authenticated managed exact-head CI v2")
    return ManagedCiContract(
        protocol_version=2,
        base_ref=base_ref,
        trusted_actor_login=actor_login,
        trusted_actor_id=actor_id,
        workflow_revision=revision,
        protection_mode=protection.state,
        audit_nonce=override_nonce,
        audit_comment_id=audit_id if isinstance(audit_id, int) else None,
    )


def _api_list(runner: Runner, config: AgentLoopConfig, endpoint: str) -> list[dict[str, object]] | None:
    """Fetch a paginated GitHub list, returning None for an uninspectable response."""
    result = runner.run(
        [config.gh_cmd, "api", "--paginate", "--slurp", endpoint], cwd=active_workdir(config), check=False
    )
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list):
        return None
    # `gh api --paginate --slurp` returns one array per page.  Accept a plain
    # list too so single-page API implementations remain compatible.
    if all(isinstance(item, list) for item in payload):
        return [entry for page in payload for entry in page if isinstance(entry, dict)]
    return [item for item in payload if isinstance(item, dict)]


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


def _ensure_managed_label(runner: Runner, *, config: AgentLoopConfig) -> bool:
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
        if not _ensure_managed_label(runner, config=config):
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
    runner: Runner, *, config: AgentLoopConfig, pr_number: int, contract: ManagedCiContract
) -> bool:
    """Remove an invocation-owned adoption label without removing a later label.

    If post-apply provenance is unreadable, the label cannot safely remain as
    our claimed suppression.  It is then removed unconditionally; when an
    event ID was recorded, retain the normal fresh event-ID ownership check.
    """
    if not contract.adopted_existing_pr or not contract.invocation_applied_label:
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
    _ensure_v2_intent(
        runner, config=config, pr_number=pr_number, expected_head_sha=expected_head_sha, contract=contract
    )
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


def _intent_body(contract: ManagedCiContract, *, pr_number: int, expected_head_sha: str, state: str) -> str:
    payload = {
        "version": 2,
        "repository": contract.repository,
        "pr": pr_number,
        "expected_head_sha": expected_head_sha,
        "base_ref": contract.base_ref,
        "workflow_revision": contract.workflow_revision,
        "nonce": contract.nonce,
        "created_at": contract.created_at,
        "state": state,
        "run_id": contract.attached_run_id,
        "run_attempt": contract.run_attempt,
    }
    return f"<!-- {INTENT_MARKER} {json.dumps(payload, separators=(',', ':'), sort_keys=True)} -->"


def _ensure_v2_intent(
    runner: Runner, *, config: AgentLoopConfig, pr_number: int, expected_head_sha: str, contract: ManagedCiContract
) -> None:
    contract.pr_number = pr_number
    contract.expected_head_sha = expected_head_sha
    contract.repository = config.repo
    comments_result = runner.run(
        [config.gh_cmd, "api", "--paginate", f"repos/{config.repo}/issues/{pr_number}/comments?per_page=100"],
        cwd=active_workdir(config), check=False,
    )
    comments: list[object] = []
    if comments_result.returncode == 0:
        try:
            parsed = json.loads(comments_result.stdout or "[]")
            comments = parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError as exc:
            raise AgentLoopError("Managed-CI intent comments response was invalid JSON.") from exc
    matching: list[tuple[int, dict[str, object]]] = []
    for raw in comments:
        if not isinstance(raw, dict):
            continue
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
        return
    contract.nonce = secrets.token_urlsafe(24)
    contract.created_at = int(time.time())
    body = _intent_body(contract, pr_number=pr_number, expected_head_sha=expected_head_sha, state="prepared")
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


def _patch_intent(runner: Runner, *, config: AgentLoopConfig, contract: ManagedCiContract, state: str) -> None:
    if contract.intent_comment_id is None or contract.nonce is None:
        return
    body = _intent_body(
        contract,
        pr_number=contract.pr_number or 0,
        expected_head_sha=contract.expected_head_sha or "",
        state=state,
    )
    # The immutable identifying fields are already in the original marker;
    # state updates add only live provenance and are still actor-owned.
    updated = runner.run(
        [config.gh_cmd, "api", "--method", "PATCH", f"repos/{config.repo}/issues/comments/{contract.intent_comment_id}", "-f", f"body={body}"],
        cwd=active_workdir(config), check=False,
    )
    if updated.returncode != 0:
        raise AgentLoopError("Unable to persist managed-CI v2 intent state.")


def _discover_v2_run(runner: Runner, *, config: AgentLoopConfig, contract: ManagedCiContract, retries: int) -> tuple[int, int | None] | None:
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
            survivors = [run for run in candidates if run.get("status") != "completed" or run.get("conclusion") != "cancelled"]
            if survivors:
                newest = sorted(survivors, key=lambda run: (str(run.get("created_at") or ""), int(run.get("id") or 0)))[-1]
                rid = newest.get("id")
                if isinstance(rid, int):
                    attempt_no = newest.get("run_attempt")
                    return rid, attempt_no if isinstance(attempt_no, int) else None
        if attempt < retries - 1:
            runner.run(["sleep", str(min(config.ci_poll_interval_seconds, 2))], cwd=active_workdir(config))
    return None


def _refresh_v2_attached_run_attempt(
    runner: Runner, *, config: AgentLoopConfig, contract: ManagedCiContract
) -> int | None:
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
    return attempt if isinstance(attempt, int) else None


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
        if contract is not None and contract.protocol_version == 2:
            if contract.attached_run_id is None:
                attached = _discover_v2_run(runner, config=config, contract=contract, retries=1)
                if attached is not None:
                    contract.attached_run_id, contract.run_attempt = attached
                    _patch_intent(runner, config=config, contract=contract, state="attached")
            else:
                current_attempt = _refresh_v2_attached_run_attempt(
                    runner, config=config, contract=contract
                )
                if current_attempt is not None and current_attempt != contract.run_attempt:
                    contract.run_attempt = current_attempt
                    _patch_intent(runner, config=config, contract=contract, state="attached")
            final = _v2_correlated_status(
                runner, config=config, expected_head=expected_head, contract=contract
            )
        else:
            final = _find_context(latest, FINAL_CONTEXT)
        status = final.status.lower() if final is not None else "pending"
        log(config, f"Managed CI context '{FINAL_CONTEXT}' status: {status}")
        if status == "success":
            if contract is not None and contract.protocol_version == 2:
                _patch_intent(runner, config=config, contract=contract, state="completed")
            return ManagedCiOutcome(status="passed", checks=latest, head_sha=live_head)
        if status in {
            "failure",
            "error",
            "cancelled",
            "timed_out",
            "action_required",
            "startup_failure",
            "stale",
        }:
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
        [config.gh_cmd, "api", f"repos/{config.repo}/commits/{expected_head}/status"],
        cwd=active_workdir(config), check=False,
    )
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None
    statuses = payload.get("statuses") if isinstance(payload, dict) else None
    if not isinstance(statuses, list):
        return None
    required = (
        f"nonce={contract.nonce}",
        f"run_id={contract.attached_run_id}",
    )
    if contract.run_attempt is not None:
        required += (f"attempt={contract.run_attempt}",)
    for raw in statuses:
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
        if not allowed_publisher or not set(required).issubset(description_tokens):
            continue
        target_path = urlparse(target).path.rstrip("/")
        target_segments = target_path.split("/")
        if (
            len(target_segments) < 3
            or target_segments[-3:-1] != ["actions", "runs"]
            or target_segments[-1] != str(contract.attached_run_id)
        ):
            continue
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
    return None


def _v2_failed_jobs(runner: Runner, *, config: AgentLoopConfig, run_id: int | None) -> tuple[str, ...]:
    if run_id is None:
        return ()
    result = runner.run(
        [config.gh_cmd, "api", "--paginate", f"repos/{config.repo}/actions/runs/{run_id}/jobs?filter=latest&per_page=100"],
        cwd=active_workdir(config), check=False,
    )
    if result.returncode != 0:
        return (f"Managed CI run {run_id} failed; job details were unavailable.",)
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return (f"Managed CI run {run_id} failed; job details were unavailable.",)
    jobs = payload.get("jobs") if isinstance(payload, dict) else None
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
