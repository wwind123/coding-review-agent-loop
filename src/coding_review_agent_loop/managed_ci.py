"""External driver for repositories that implement managed exact-head CI."""

from __future__ import annotations

from .managed_ci_tuple import (
    MANAGED_LABEL,
)

from .managed_ci_result import VerifiedManagedResult

import json
import re
import secrets
import shlex
import time
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from .ci_health import (
    CiInfrastructureStall,
    FAILED_JOB_CONCLUSIONS,
    board_tests_passed,
    check_name_matches,
    classify_run_jobs,
    PullRequestCheck,
    PullRequestChecks,
    V2_RUN_NAME_PREFIX,
    is_wholly_infrastructure_blocked,
)
from .config import AgentLoopConfig
from .errors import AgentLoopError
from .github import (
    _fetch_protocol_comment_envelope,
    read_actions_run,
    read_run_jobs,
    reconciled_pr_ready,
    resolve_authenticated_github_actor,
    PullRequestMetadata,
    board_protection_is_reliable,
    get_pr_checks,
    protection_awaits_readiness,
    get_pr_head_sha,
    get_pr_mergeability,
    note_host_footer_observed,
    patch_verified_trusted_protocol_comment_observed,
    post_verified_trusted_pr_protocol_comment,
    post_verified_trusted_pr_protocol_comment_observed,
    WrittenProtocolComment,
)
from .logging import log
from .managed_ci_bases import base_is_trusted
from .managed_ci_intent_validation import (
    INTENT_NONCE_PATTERN,
    IntentBinding,
)
from . import github_retry
from .github_retry import (
    DEFAULT_POLICY,
    GitHubAttempt,
    attempt_from_result,
    classify_gh_failure,
    describe_gh_failure,
)
from .runner import CommandResult, Runner
from .workdirs import active_workdir, github_api_cwd
from .protocol_markers import (
    PR_BODY_SURFACE,
    PR_COMMENT_SURFACE,
    TrustedBody,
    protocol_record_label,
    strip_known_host_footer,
)
from .managed_ci_status import FINAL_CONTEXT
from . import (
    managed_ci_api,
    managed_ci_authorization,
    managed_ci_failure,
    managed_ci_intent_validation,
    managed_ci_protection,
    managed_ci_reconciliation,
    managed_ci_recovery_commands,
    managed_ci_status,
)


# Support imports used by the retained driver gates.
from .managed_ci_types import (
    MANAGED_OPT_OUT_LABEL,
    QUALIFIED_LABEL,
    QUALIFICATION_MARKER,
    MANAGED_PLUMBING_JOB_NAMES,
    PLUMBING_STATUS_CONTEXTS,
    READINESS_CONTEXT,
    WORKFLOW_FILE,
    _CONTRACT_MARKERS,
    V2_MARKER,
    INTENT_MARKER,
    V2_FEATURE_MARKERS,
    V2_ADOPTION_FEATURE_MARKERS,
    RECOVERY_MARKER,
    VISIBLE_INTENT_MARKER,
    HOST_FOOTER_INTENT_MARKER,
    UNPROTECTED_OVERRIDE_TRAILER,
    TRUSTED_BASES_MARKER,
    TRUSTED_BASES_VARIABLE,
    ISSUE_AUTHORIZATION_MARKER,
    _TERMINAL_CI_STATUSES,
    ManagedCiProbeContext,
    ProtectionAssessment,
    ManagedCiReadiness,
    _OVERRIDE_PROTECTION_STATES,
    V1_DISPATCH_RECORD_LIMIT,
    ManagedV1DispatchRecord,
    ManagedCiContract,
    ManagedCiCreationIntent,
    AuthenticatedIssueCreatedHandoff,
    ManagedCiIssueAuthorization,
    AuthenticatedManagedResume,
    OrdinaryRecoveryCapability,
    ManagedCiOutcome,
    ManagedCiRunSnapshot,
    CompleteLabelHistory,
    UnknownLabelHistory,
    ReconciledLabelWrite,
)


# Authorization imports used by the retained driver gates.
from .managed_ci_authorization import (
    ChainAuthContext,
)


@managed_ci_api._surfaces_list_failures
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

    workflow_ref = _workflow_source_ref(runner, config, metadata.base_branch or config.base)
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
        adoption_diagnostics: list[str] = []
        adopted = _activate_v2_existing_pr_adoption(
            runner,
            config=config,
            pr_number=pr_number,
            metadata=metadata,
            diagnostics=adoption_diagnostics,
        )
        if adopted is None and config.managed_ci:
            raise AgentLoopError(
                f"--managed-ci-adopt-existing-pr could not safely adopt PR #{pr_number}; "
                "no suppression or qualification was claimed."
                + "".join(adoption_diagnostics)
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
        apply_result = _label_add(
            runner, config=config, pr_number=pr_number, label_name=MANAGED_LABEL
        ).result
        if apply_result.returncode != 0:
            raise AgentLoopError(
                f"Unable to apply `{MANAGED_LABEL}` to PR #{pr_number}.{failure_suffix(apply_result)}"
            )
        try:
            _wait_for_label_handoff(
                runner,
                config=config,
                pr_number=pr_number,
                metadata=metadata,
                prior_run_ids=prior_workflow_run_ids,
            )
        except AgentLoopError as exc:
            remove_result = _label_remove(
                runner, config=config, pr_number=pr_number, label_name=MANAGED_LABEL
            ).result
            if remove_result.returncode != 0:
                raise AgentLoopError(
                    f"{exc} Cleanup also failed: `{MANAGED_LABEL}` remains applied and "
                    f"suppresses hosted CI until it is removed.{failure_suffix(remove_result)}"
                ) from exc
            raise
    log(config, f"PR #{pr_number}: activated managed exact-head CI")
    return ManagedCiContract()


def evaluate_managed_ci_readiness(
    runner: Runner, *, context: ManagedCiProbeContext, base: str | None, trusted_actor: str
) -> ManagedCiReadiness:
    """Read every v2 prerequisite without writes or AgentLoopConfig construction."""
    repo, repo_result = managed_ci_api._probe_json(runner, context, f"repos/{context.repo}")
    if repo is None:
        return ManagedCiReadiness("indeterminate", None, None, None, False, False,
            ProtectionAssessment("indeterminate", "none", "repository metadata unavailable"),
            ("repository metadata could not be read",), ())
    resolved_base = base or (repo.get("default_branch") if isinstance(repo.get("default_branch"), str) else None)
    visibility = "private" if repo.get("private") is True else "public" if repo.get("private") is False else None
    if not resolved_base:
        return ManagedCiReadiness("indeterminate", visibility, None, None, False, False,
            ProtectionAssessment("indeterminate", "none", "base branch unavailable"), ("base branch is unavailable",), ())
    default_branch = repo.get("default_branch") if isinstance(repo.get("default_branch"), str) else None
    workflow_ref = default_branch or resolved_base
    workflow, workflow_result = managed_ci_api._probe_raw_workflow(runner, context, workflow_ref)
    if workflow is None:
        if managed_ci_api._is_http_error(workflow_result, 404):
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
    base_problem = trusted_base_problem(
        runner, gh_cmd=context.gh_cmd, repo=context.repo, cwd=context.cwd, base=resolved_base,
        default_branch=default_branch, workflow_text=workflow,
    )
    if base_problem is not None:
        return ManagedCiReadiness(
            "invalid", visibility, None, None, False, False,
            ProtectionAssessment("indeterminate", "none", "base is not a trusted managed-CI target"),
            (base_problem,),
            (f"add the base to the {TRUSTED_BASES_VARIABLE} repository variable, or use the default branch",),
            resolved_base,
        )
    who, _ = managed_ci_api._probe_json(runner, context, "user")
    variable = managed_ci_api._read_managed_actor_variable(runner, context.gh_cmd, context.repo, context.cwd)
    actor = who.get("login") if who and isinstance(who.get("login"), str) else None
    advertised, advertised_source = managed_ci_api._resolve_advertised_actor(variable, trusted_actor)
    protection = managed_ci_protection.assess_exact_head_protection(runner, context=context, base=resolved_base)
    if who is None:
        return ManagedCiReadiness("indeterminate", visibility, actor, advertised, False, False, protection,
            ("a required read-only GitHub probe failed: the authenticated user could not be read",), ())
    if variable.status == "error" or (variable.status == "unreadable" and advertised is None):
        return ManagedCiReadiness("indeterminate", visibility, actor, advertised, False, False, protection,
            (f"a required read-only GitHub probe failed: {variable.detail}",),
            (
                ("pass --managed-ci-trusted-actor so it can stand in for the unreadable variable",)
                if variable.status == "unreadable" else ()
            ))
    core = V2_MARKER in workflow
    complete = core and all(marker in workflow for marker in V2_FEATURE_MARKERS)
    # Pre-marker v2 fixtures/workflows that contain no pull_request trigger at
    # all never suppress an opening matrix. Treat them as legacy-compatible;
    # any workflow that does opt into pull_request suppression must advertise
    # the explicit recovery marker and unlabeled activity.
    recovery = (RECOVERY_MARKER in workflow and "unlabeled" in workflow) or "pull_request" not in workflow
    expected_actor = trusted_actor.strip()
    identity_ok = bool(actor and advertised and expected_actor and actor.casefold() == expected_actor.casefold() == advertised.casefold())
    source = advertised_source
    if not core:
        return ManagedCiReadiness("ordinary_fallback", visibility, actor, advertised, False, recovery, protection,
            ("base workflow does not advertise managed-CI v2",), ("deploy the documented managed-CI v2 workflow",), resolved_base,
            advertised_actor_source=source)
    if not complete or not recovery:
        missing = "complete v2 contract" if not complete else "unlabeled recovery contract"
        return ManagedCiReadiness("invalid", visibility, actor, advertised, complete, recovery, protection,
            (f"workflow lacks the {missing}",), ("add the documented workflow markers and pull_request unlabeled trigger",), resolved_base,
            advertised_actor_source=source)
    if not identity_ok:
        return ManagedCiReadiness("ordinary_fallback", visibility, actor, advertised, complete, recovery, protection,
            ("authenticated login, trusted actor, and AGENT_LOOP_MANAGED_ACTOR do not match",),
            (f"gh variable set AGENT_LOOP_MANAGED_ACTOR --repo {shlex.quote(context.repo)} --body {shlex.quote(expected_actor)}",), resolved_base,
            advertised_actor_source=source)
    if protection.state == "strict":
        return ManagedCiReadiness("strict_ready", visibility, actor, advertised, complete, recovery, protection, (), (), resolved_base,
            advertised_actor_source=source)
    # Readiness is independent of this invocation's waiver flags; only the
    # authorization gates consult waivable_protection_states(config).
    if protection.state in _OVERRIDE_PROTECTION_STATES:
        if protection.state == "unreadable":
            remediation = (
                "grant administration read, or configure a readable non-bypassable "
                "final-ci/exact-head ruleset",
                "or use --allow-unprotected-managed-ci --allow-unreadable-protection for this invocation",
            )
        else:
            remediation = (
                "configure non-bypassable final-ci/exact-head protection, or use "
                "--allow-unprotected-managed-ci for this invocation",
            )
        return ManagedCiReadiness("override_eligible", visibility, actor, advertised, complete, recovery, protection,
            (protection.detail,), remediation, resolved_base, advertised_actor_source=source)
    return ManagedCiReadiness("indeterminate", visibility, actor, advertised, complete, recovery, protection,
        (protection.detail,), (), resolved_base, advertised_actor_source=source)


def render_managed_ci_preflight(result: ManagedCiReadiness, *, repo: str, base: str, trusted_actor: str) -> str:
    variable = result.advertised_actor or "missing"
    if result.advertised_actor_source == "asserted":
        variable += " (asserted; unverified locally, enforced by workflow)"
    lines = [
        f"repository: {repo}", f"base: {result.base or base}", f"visibility: {result.repo_visibility or 'unknown'}",
        f"workflow: {'v2 complete' if result.workflow_v2 else 'ordinary/absent'}",
        f"recovery: {'unlabeled capable' if result.recovery_capable else 'missing'}",
        f"actor: authenticated={result.actor or 'unknown'} trusted={trusted_actor} variable={variable}",
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
    source_workflow, _ = managed_ci_api._probe_raw_workflow(
        runner, source_context, _workflow_source_ref(runner, config, config.base)
    )
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
    if readiness.advertised_actor_source == "asserted" and readiness.advertised_actor:
        managed_ci_api._log_asserted_actor(config, readiness.advertised_actor)
    if readiness.state == "indeterminate" and not legacy_non_suppressing:
        failed_reads = "; ".join(
            dict.fromkeys((readiness.protection.detail, *readiness.reasons))
        )
        if config.managed_ci:
            raise AgentLoopError(
                "--managed-ci could not determine the repository's exact-head protection "
                f"({failed_reads}). No waiver applies to an indeterminate result; "
                "no PR was created."
            )
        log(
            config,
            f"Managed-CI readiness could not be determined ({failed_reads}); continuing with ordinary CI.",
        )
        return None
    if readiness.state == "indeterminate":
        who = managed_ci_api._api_json(runner, config, "user", quiet=True)
        advertised = managed_ci_api._advertised_managed_actor(runner, config)
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
    explicit_waiver = readiness.protection.state in managed_ci_protection.waivable_protection_states(config)
    if readiness.state == "override_eligible" and not (explicit_waiver or legacy_non_suppressing):
        if config.managed_ci:
            raise AgentLoopError(
                "--managed-ci requires protected final-ci/exact-head or "
                f"{managed_ci_protection.waiver_flags_for_protection(readiness.protection.state)} "
                f"(protection is {readiness.protection.state}: {readiness.protection.detail}); "
                "no PR was created."
            )
        return None
    if readiness.state != "strict_ready" and not (
        readiness.state == "override_eligible" and (explicit_waiver or legacy_non_suppressing)
    ):
        if config.managed_ci:
            reasons = "; ".join(readiness.reasons)
            raise AgentLoopError(
                "--managed-ci repository readiness is not sufficient for suppression and exact-head qualification"
                + (f" ({reasons})" if reasons else "")
                + "; no PR was created."
            )
        return None
    return ManagedCiCreationIntent(
        branch=branch or f"agent-loop/managed-{issue_number}",
        trusted_actor=readiness.actor or config.managed_ci_trusted_actor,
        protection_mode=readiness.protection.state,
        audit_nonce=(secrets.token_urlsafe(18) if explicit_waiver else None),
    )


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
    pr = managed_ci_api._api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}", quiet=True)
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
        command = managed_ci_recovery_commands.render_managed_ci_resume_command(config, pr_number=pr_number, managed_ci=True)
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


def _issue_created_rendering_issue_number(
    managed_resume: AuthenticatedManagedResume | None,
    *,
    origin: str,
    head_ref: str | None,
) -> int | None:
    """Return an issue hint for diagnostics, never for authorization."""
    if origin != "issue-created":
        return None
    handoff = managed_resume.issue_created_handoff if managed_resume is not None else None
    if handoff is not None and handoff.issue_number > 0:
        return handoff.issue_number
    return managed_ci_authorization.managed_branch_issue_number(head_ref)


def plumbing_check_names(config: AgentLoopConfig | None = None) -> frozenset[str]:
    """The managed plumbing catalog plus explicit --ci-plumbing-check names."""
    extra = tuple(getattr(config, "ci_plumbing_checks", ()) or ()) if config is not None else ()
    return MANAGED_PLUMBING_JOB_NAMES | PLUMBING_STATUS_CONTEXTS | frozenset(
        name for name in extra if name
    )


def is_plumbing_check(name: str, extra_names: Iterable[str] = ()) -> bool:
    """Exact whole-name or final-segment catalog membership; never a substring."""
    return check_name_matches(
        name,
        MANAGED_PLUMBING_JOB_NAMES | PLUMBING_STATUS_CONTEXTS | frozenset(extra_names),
    )


# Failed-activation state reporting (#1067).
#
# One constant names every REST ``pulls/{n}`` field of the activation tuple.
# The entry snapshot, the post-failure measured read, the tuple comparison,
# and the printed inspection command are all derived from it, so the operator
# is always shown exactly the fields the run itself compared.
_ACTIVATION_TUPLE_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("head_repo", ("head", "repo", "full_name")),
    ("head_ref", ("head", "ref")),
    ("head_sha", ("head", "sha")),
    ("base_ref", ("base", "ref")),
    ("author_login", ("user", "login")),
    ("author_id", ("user", "id")),
    ("state", ("state",)),
    ("draft", ("draft",)),
    ("labels", ("labels",)),
)
# Fields compared for identity; draft and labels are lifecycle state instead.
_ACTIVATION_IDENTITY_FIELDS: tuple[str, ...] = (
    "head_repo", "head_ref", "head_sha", "base_ref", "author_login", "author_id", "state",
)
_CASEFOLDED_IDENTITY_FIELDS = frozenset({"head_repo", "author_login"})
_INTEGRITY_RELEASE_REASONS = frozenset({"foreign-actor", "event-changed"})


def _strict_int(value: object) -> int | None:
    """Return a real JSON integer; a JSON boolean is never an identity."""
    return value if type(value) is int else None


def _str_field(obj: object, *path: str) -> str | None:
    """Walk nested objects; any non-object hop or non-string leaf is None."""
    current = obj
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current if isinstance(current, str) else None


def _strict_label_names(payload: Mapping[str, object]) -> frozenset[str] | None:
    """Return label names only when every entry is an object with a string name."""
    labels = payload.get("labels")
    if not isinstance(labels, list):
        return None
    names: set[str] = set()
    for item in labels:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            return None
        names.add(item["name"])
    return frozenset(names)


@dataclass(frozen=True)
class _PrStateSnapshot:
    """Three-valued PR state: every field is known or ``None`` (unreadable)."""

    head_repo: str | None
    head_ref: str | None
    head_sha: str | None
    base_ref: str | None
    author_login: str | None
    author_id: int | None
    state: str | None
    draft: bool | None
    labeled: bool | None

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "_PrStateSnapshot":
        paths = dict(_ACTIVATION_TUPLE_FIELDS)
        user = payload.get("user")
        draft = payload.get("draft")
        label_names = _strict_label_names(payload)
        return cls(
            head_repo=_str_field(payload, *paths["head_repo"]),
            head_ref=_str_field(payload, *paths["head_ref"]),
            head_sha=_str_field(payload, *paths["head_sha"]),
            base_ref=_str_field(payload, *paths["base_ref"]),
            author_login=_str_field(payload, *paths["author_login"]),
            author_id=_strict_int(user.get("id")) if isinstance(user, dict) else None,
            state=_str_field(payload, *paths["state"]),
            draft=draft if isinstance(draft, bool) else None,
            labeled=None if label_names is None else MANAGED_LABEL in label_names,
        )

    def unknown_fields(self) -> list[str]:
        values = {
            "head_repo": self.head_repo, "head_ref": self.head_ref,
            "head_sha": self.head_sha, "base_ref": self.base_ref,
            "author_login": self.author_login, "author_id": self.author_id,
            "state": self.state, "draft": self.draft, "labels": self.labeled,
        }
        return [name for name, _path in _ACTIVATION_TUPLE_FIELDS if values[name] is None]

    def render(self) -> str:
        draft = {True: "draft", False: "ready", None: "of unknown draft state"}[self.draft]
        labeled = {True: "labeled", False: "unlabeled", None: "with unreadable labels"}[self.labeled]
        text = f"{draft}/{labeled}" if self.draft is not None and self.labeled is not None else (
            f"{draft}, {labeled}"
        )
        unreadable = [name for name in self.unknown_fields() if name not in {"draft", "labels"}]
        if unreadable:
            text += " (" + ", ".join(f"{name} unreadable" for name in unreadable) + ")"
        return text


def _entry_tuple_mismatches(entry: _PrStateSnapshot, live: _PrStateSnapshot) -> list[str]:
    """Name every identity field that differs or is unknown on either side."""
    changed: list[str] = []
    for name in _ACTIVATION_IDENTITY_FIELDS:
        before = getattr(entry, name)
        after = getattr(live, name)
        if before is None or after is None:
            changed.append(name)
        elif name in _CASEFOLDED_IDENTITY_FIELDS:
            if before.casefold() != after.casefold():
                changed.append(name)
        elif before != after:
            changed.append(name)
    return changed


@dataclass
class _FailedActivationContext:
    """Mutation facts this activation observed directly, never inferred ones.

    ``label_post_acknowledged`` means only that GitHub accepted this run's
    idempotent label request; the label may already have been applied by a
    concurrent invocation, so no report ever says this run owns the label.
    """

    entry: _PrStateSnapshot
    recovery_capable: bool = False
    drafted_by_this_run: bool = False
    ready_undo_ambiguous: bool = False
    label_post_attempted: bool = False
    label_post_acknowledged: bool = False
    label_post_ambiguous: bool = False
    # Application id proven by a recovered (ambiguous) label add; any later
    # cleanup DELETE is fresh-checked against it so a replacement is never removed.
    recovered_label_event_id: int | None = None
    label_release_attempted: bool = False
    label_released: bool = False
    release_reason: str | None = None
    release_reason_kind: Literal["foreign-actor", "event-changed"] | None = None
    # Set when a release was skipped because the base workflow has no
    # unlabeled recovery route; it vetoes any automatic readiness write.
    recovery_incapable_fallback: bool = False
    resume_command: str | None = None
    report: str | None = None

    @property
    def mutated(self) -> bool:
        return (
            self.label_release_attempted
            or self.drafted_by_this_run
            or self.ready_undo_ambiguous
            or self.label_post_attempted
        )


def _read_failed_activation_state(
    runner: Runner, config: AgentLoopConfig, pr_number: int
) -> _PrStateSnapshot | None:
    """Measure the live PR without ever raising.

    ``None`` means the whole state is unreadable: a failed or unlaunchable
    command, invalid JSON, or a non-object payload.  A readable object (even
    ``{}``) yields per-field unknowns instead.
    """
    try:
        result = runner.run(
            [config.gh_cmd, "api", f"repos/{config.repo}/pulls/{pr_number}"],
            cwd=active_workdir(config), check=False,
        )
        if result.returncode != 0:
            return None
        payload = json.loads(result.stdout or "")
        if not isinstance(payload, dict):
            return None
        return _PrStateSnapshot.from_payload(payload)
    except Exception:
        return None


def _pr_inspection_command(config: AgentLoopConfig, pr_number: int) -> str:
    """Render a non-mutating REST read of every activation tuple field.

    The query tolerates a null ``head.repo`` and flags malformed label entries
    and containers instead of erroring or printing an empty list.
    """
    parts = []
    for name, path in _ACTIVATION_TUPLE_FIELDS:
        if name == "labels":
            parts.append(
                'labels: (if (.labels | type) == "array" then [.labels[] | '
                'if type == "object" and (.name | type) == "string" then .name '
                "else {malformed: .} end] else {malformed_container: .labels} end)"
            )
        elif len(path) == 1:
            parts.append(f"{name}: .{path[0]}")
        else:
            parts.append(f"{name}: (.{'.'.join(path)}? // null)")
    query = "{" + ", ".join(parts) + "}"
    return shlex.join(
        [config.gh_cmd, "api", f"repos/{config.repo}/pulls/{pr_number}", "--jq", query]
    )


def _expected_tuple_text(entry: _PrStateSnapshot) -> str:
    values = []
    for name in _ACTIVATION_IDENTITY_FIELDS:
        value = getattr(entry, name)
        values.append(f"{name}={value if value is not None else 'unknown'}")
    return ", ".join(values)


def _failed_activation_resume_command(
    config: AgentLoopConfig, pr_number: int, context: _FailedActivationContext
) -> str:
    if context.resume_command:
        return context.resume_command
    return managed_ci_recovery_commands.render_managed_ci_resume_command(config, pr_number=pr_number, managed_ci=True)


def _report_failed_activation_state(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    context: _FailedActivationContext,
) -> str:
    """Describe the measured post-failure state, restoring readiness only when proven.

    This never raises: the original activation error must always survive.  It
    is idempotent per context, so a second call makes no second readiness
    write.
    """
    if context.report is not None:
        return context.report
    try:
        text = _build_failed_activation_report(
            runner, config=config, pr_number=pr_number, context=context
        )
    except Exception as exc:
        # Remedy (C) even here: the non-mutating inspection plus the resume
        # command, each rendered independently so neither can mask the other.
        try:
            inspect = (
                f" Do not ready or unlabel the PR. Inspect it with "
                f"`{_pr_inspection_command(config, pr_number)}`"
            )
        except Exception:
            inspect = ""
        try:
            resume_command = _failed_activation_resume_command(config, pr_number, context)
            resume = (
                f", then rerun `{resume_command}`." if inspect else f" Resume with `{resume_command}`."
            )
        except Exception:
            resume = "." if inspect else ""
        text = (
            f"Post-failure state report unavailable ({type(exc).__name__}); "
            f"the current state of PR #{pr_number} could not be re-read.{inspect}{resume}"
        )
    context.report = text
    return text


def _build_failed_activation_report(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    context: _FailedActivationContext,
) -> str:
    entry = context.entry
    measured = _read_failed_activation_state(runner, config, pr_number)
    resume = _failed_activation_resume_command(config, pr_number, context)
    inspect = _pr_inspection_command(config, pr_number)
    first_measured = measured
    # The opening sentence is filled in last, from the final measured state.
    lines = [""]
    facts = []
    if context.drafted_by_this_run:
        facts.append("its ready-to-draft request was acknowledged")
    if context.ready_undo_ambiguous:
        facts.append("its ready-to-draft request was not confirmed")
    if context.label_post_acknowledged:
        facts.append(f"its `{MANAGED_LABEL}` label request was acknowledged")
    elif context.label_post_ambiguous:
        facts.append(f"its `{MANAGED_LABEL}` label request was not confirmed")
    elif context.label_post_attempted:
        facts.append(f"its `{MANAGED_LABEL}` label request was attempted")
    if context.label_released:
        facts.append("its label DELETE was acknowledged")
    elif context.label_release_attempted:
        facts.append("its label DELETE was not confirmed")
    if facts:
        lines.append("This run's recorded requests: " + "; ".join(facts) + ".")
    if context.recovery_incapable_fallback and not context.label_release_attempted:
        lines.append(
            f"This run made no `{MANAGED_LABEL}` label DELETE because the base workflow does not "
            "prove an unlabeled pull_request recovery route."
        )

    integrity = context.release_reason_kind in _INTEGRITY_RELEASE_REASONS
    if context.release_reason_kind == "foreign-actor":
        lines.append(
            "Restoration is refused for ownership integrity: the active managed-label event's "
            "ownership was not the trusted actor's."
        )
    elif context.release_reason_kind == "event-changed":
        lines.append(
            "Restoration is refused for ownership integrity: the active managed-label event "
            "changed or could not be verified; its owner is not asserted."
        )

    if context.label_released:
        lines.append(
            "The label release is the deliberate fail-closed return to ordinary CI "
            f"(#663/#667); this run does not reapply `{MANAGED_LABEL}`, because reapplying "
            "it would suppress ordinary CI on an unqualified head and mint a new label event."
        )

    mismatches = _entry_tuple_mismatches(entry, measured) if measured is not None else []
    safe_manual_ready = (
        not integrity
        and not context.recovery_incapable_fallback
        and measured is not None
        and not measured.unknown_fields()
        and measured.draft is True
        and measured.labeled is False
        and not mismatches
    )
    remedy = "C"
    readiness_attempted = False
    if entry.draft is False:
        restoration, remedy, after, readiness_attempted = _evaluate_ready_restoration(
            runner, config=config, pr_number=pr_number, context=context,
            measured=measured, mismatches=mismatches, integrity=integrity,
            safe_manual_ready=safe_manual_ready,
        )
        lines.append(restoration)
        if readiness_attempted:
            # The readiness command may have changed the PR: the read-back is
            # the current state, and an unreadable read-back is unknown.
            measured = after
            mismatches = _entry_tuple_mismatches(entry, after) if after is not None else []
    elif (
        not integrity
        and not context.recovery_incapable_fallback
        and measured is not None
        and not mismatches
        and measured.draft is True
        and measured.labeled is False
    ):
        remedy = "resume"

    if context.label_release_attempted and measured is not None:
        # Claims about suppression use the final measured state only.
        if measured.labeled is True:
            # Only an acknowledged DELETE followed by a labeled read is an
            # observed re-add; an unconfirmed DELETE proves no transition.
            presence = "is present again" if context.label_released else "is present"
            lines.append(
                f"`{MANAGED_LABEL}` {presence}; suppression is still active and ordinary "
                "CI has not resumed."
            )
        elif measured.labeled is False and context.recovery_capable:
            lines.append(
                "Ordinary CI can resume through the base workflow's unlabeled recovery route."
            )

    lines.extend(_lifecycle_sentences(entry, measured))

    now = measured.render() if measured is not None else "unknown: the current state could not be re-read"
    if entry.draft is False and readiness_attempted:
        lines[0] = (
            f"Before this run PR #{pr_number} was {entry.render()}; after the failure and before "
            f"the readiness attempt it was {first_measured.render()}; it is now {now}."
        )
    else:
        lines[0] = f"Before this run PR #{pr_number} was {entry.render()}; it is now {now}."

    changed = mismatches + [
        name for name in (measured.unknown_fields() if measured is not None else [])
        if name not in mismatches
    ]
    if measured is not None and measured.labeled is True:
        changed.append("labels (labeled)")
    if remedy == "A":
        lines.append(
            f"Manual undo: confirm every field with `{inspect}` (expected "
            f"{_expected_tuple_text(entry)}, draft=true, no `{MANAGED_LABEL}` label), then run "
            f"`{shlex.join([config.gh_cmd, 'pr', 'ready', str(pr_number), '--repo', config.repo])}`; "
            f"no qualification is implied. To retry managed qualification instead, rerun `{resume}`."
        )
    elif remedy == "restored":
        lines.append(f"To retry managed qualification, rerun `{resume}`.")
    elif remedy == "resume":
        lines.append(f"Resume with `{resume}`.")
    else:
        detail = f"; changed, unreadable, or labeled: {', '.join(changed)}" if changed else ""
        lines.append(
            f"Do not ready or unlabel the PR. Inspect it with `{inspect}` (expected "
            f"{_expected_tuple_text(entry)}{detail}), then rerun `{resume}`, which "
            "re-authenticates the tuple and the label-event ownership before any write."
        )
    return " ".join(lines)


def _evaluate_ready_restoration(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    context: _FailedActivationContext,
    measured: _PrStateSnapshot | None,
    mismatches: list[str],
    integrity: bool,
    safe_manual_ready: bool,
) -> tuple[str, str, _PrStateSnapshot | None, bool]:
    """Return (restoration sentence, remedy kind, read-back state, readiness attempted)."""
    entry = context.entry
    if measured is None:
        return "Readiness was not restored: the current state is unknown.", "C", None, False
    if measured.draft is False and not context.drafted_by_this_run:
        # No acknowledged conversion: nothing to restore, and no readiness
        # change is attributed to this run.
        clean = not mismatches and not measured.unknown_fields() and measured.labeled is False
        return (
            f"No readiness restoration is needed: the PR is measured {measured.render()}"
            + (f" (changed fields: {', '.join(mismatches)})" if mismatches else "")
            + " and no ready-to-draft conversion by this run was acknowledged.",
            # A nonzero ready-to-draft exit is ambiguous, an integrity reason
            # vetoes the bare resume, and a base without the unlabeled
            # recovery route always gets inspection.
            "resume"
            if clean
            and not context.ready_undo_ambiguous
            and not integrity
            and not context.recovery_incapable_fallback
            else "C",
            None,
            False,
        )
    if measured.draft is False:
        return (
            f"Readiness was not restored by this run: the PR is already ready (measured "
            f"{measured.render()}"
            + (f"; changed fields: {', '.join(mismatches)}" if mismatches else "")
            + "), and this run does not know who made it ready.",
            "C",
            None,
            False,
        )
    if integrity:
        return "Automatic readiness restoration was refused for ownership integrity.", "C", None, False
    if context.recovery_incapable_fallback:
        return (
            "Automatic readiness restoration was refused: the base workflow has no unlabeled "
            "recovery route, so this run released no label and makes no readiness write.",
            "C",
            None,
            False,
        )
    if not context.drafted_by_this_run:
        if context.ready_undo_ambiguous:
            return (
                "Readiness was not restored: attribution is ambiguous because this run's "
                "ready-to-draft request was not confirmed.",
                "A" if safe_manual_ready else "C",
                None,
                False,
            )
        return "Readiness was not restored: this run made no acknowledged draft conversion.", "C", None, False
    if entry.labeled is not False or entry.unknown_fields():
        return "Readiness was not restored: the entry state was not fully known.", "C", None, False
    unknown = measured.unknown_fields()
    if unknown:
        return f"Readiness was not restored: the state is unknown ({', '.join(unknown)}).", "C", None, False
    if measured.labeled:
        return "Readiness was not restored: the PR is still labeled.", "C", None, False
    if mismatches:
        return (
            f"Readiness was not restored: tuple fields changed ({', '.join(mismatches)}).",
            "C",
            None,
            False,
        )
    ready = reconciled_pr_ready(
        runner, config=config, pr_number=pr_number, expected_head_sha=entry.head_sha,
    )
    after = _read_failed_activation_state(runner, config, pr_number)
    if ready.returncode != 0:
        after_safe = (
            after is not None
            and not after.unknown_fields()
            and after.draft is True
            and after.labeled is False
            and not _entry_tuple_mismatches(entry, after)
        )
        return (
            "Readiness was not restored: the readiness command failed"
            + (f"; the PR is now {after.render()}" if after is not None else "; the state could not be re-read")
            + "." + failure_suffix(ready),
            "A" if after_safe else "C",
            after,
            True,
        )
    after_changed = (
        _entry_tuple_mismatches(entry, after) + after.unknown_fields() if after is not None else []
    )
    if after is not None and not after_changed and after.draft is False and after.labeled is False:
        return (
            "Restored to ready/unlabeled as found; no qualification is claimed for this head.",
            "restored",
            after,
            True,
        )
    if after is None:
        detail = "the state could not be re-read"
    else:
        problems = list(dict.fromkeys(after_changed))
        if after.draft is True:
            problems.append("draft")
        if after.labeled is True:
            problems.append("labels (labeled)")
        detail = f"the read-back shows {after.render()}; mismatched: {', '.join(problems)}"
    return (
        f"Readiness command acknowledged; restoration not verified ({detail}).",
        "C",
        after,
        True,
    )


def _lifecycle_sentences(entry: _PrStateSnapshot, measured: _PrStateSnapshot | None) -> list[str]:
    """Emit lifecycle claims only for states the post-failure read proves."""
    if measured is None:
        return []
    sentences = []
    if entry.draft is False and measured.draft is not False and measured.draft is not None:
        sentences.append("The previously advertised manual-merge state is suspended.")
    if measured.draft is True and measured.labeled is False:
        sentences.append("The PR is in the documented draft/unlabeled re-entry state.")
    elif entry.draft is True and entry.labeled is True:
        if measured.draft is True and measured.labeled is True:
            sentences.append("It remains draft/labeled and managed qualification was abandoned.")
        elif measured.draft is False and measured.labeled is True:
            sentences.append(
                "It is now ready/labeled and this run made no readiness change; PR-loop entry "
                "treats ready/labeled as a retained qualified label, and this run claims no "
                "qualification for this head."
            )
    return sentences


def _measured_state_line(runner: Runner, config: AgentLoopConfig, pr_number: int) -> str:
    """Context-free measured current-state sentence; never raises."""
    measured = _read_failed_activation_state(runner, config, pr_number)
    if measured is None:
        return f"The current state of PR #{pr_number} could not be re-read."
    return f"PR #{pr_number} is now {measured.render()}."


def _fallback_state_report(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    context: _FailedActivationContext,
) -> str | None:
    """Log the measured report for a non-raising fallback.

    A report is produced after a recorded mutation, and also when a release
    was skipped on a recovery-incapable base: the caller then prints a
    lifecycle outcome, which must come from a measurement (#1067).
    """
    if not context.mutated and not context.recovery_incapable_fallback:
        return None
    report = _report_failed_activation_state(
        runner, config=config, pr_number=pr_number, context=context
    )
    log(config, f"PR #{pr_number}: {report}")
    return report


def _ordinary_fallback_contract(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    context: _FailedActivationContext,
    recovery: OrdinaryRecoveryCapability | None,
) -> ManagedCiContract:
    return ManagedCiContract(
        activation_path="ordinary_fallback",
        ordinary_recovery=recovery,
        state_report=_fallback_state_report(
            runner, config=config, pr_number=pr_number, context=context
        ),
    )


@contextmanager
def _failed_activation_guard(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    context: _FailedActivationContext,
) -> Iterator[None]:
    """Append the measured report once to any exception after a recorded mutation.

    The original exception object, type, and traceback are preserved.  With
    no recorded mutation the exception passes through unchanged, because the
    pre-existing "left unchanged" wording is then accurate.
    """
    try:
        yield
    except Exception as exc:
        if not context.mutated or getattr(exc, "_managed_state_reported", False):
            raise
        report = _report_failed_activation_state(
            runner, config=config, pr_number=pr_number, context=context
        )
        if isinstance(exc, AgentLoopError) and exc.args and isinstance(exc.args[0], str):
            exc.args = (f"{exc.args[0]} {report}", *exc.args[1:])
        else:
            exc.add_note(report)
        exc._managed_state_reported = True  # type: ignore[attr-defined]
        raise


def _require_recovered_label_ownership(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    context: _FailedActivationContext | None,
) -> int | None:
    """Fresh-check a recovered label application before any cleanup DELETE.

    Returns the recovered event id (to pass as the removal's owned id), or
    ``None`` when no label write was recovered.  When ownership cannot be
    established (unreadable history, or a different application is now the
    latest event) cleanup is refused and the label left untouched.
    """
    recovered = context.recovered_label_event_id if context is not None else None
    if recovered is None:
        return None
    history = managed_ci_api.label_event_history(
        runner, config=config, pr_number=pr_number, label_name=MANAGED_LABEL
    )
    latest = history.latest if isinstance(history, CompleteLabelHistory) else None
    if latest is None or latest.kind != "labeled" or latest.event_id != recovered:
        if context is not None:
            context.release_reason_kind = "event-changed"
        raise AgentLoopError(
            f"PR #{pr_number} `{MANAGED_LABEL}` ownership could not be re-established after "
            f"the recovered label write (recovered event {recovered}); cleanup was refused and "
            "the label was left untouched."
        )
    return recovered


def _restore_ordinary_ci_after_v2_fallback(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    reason: str,
    report_context: _FailedActivationContext | None = None,
    recovery_capable: bool = True,
) -> None:
    """Remove the issue-created suppression label before returning to ordinary CI.

    Operator text states only acknowledged or unconfirmed requests; the
    post-failure state comes from the activation guard's measured report
    (or, without a context, from a context-free measured line).  On a base
    without the unlabeled recovery route the label is retained: removing it
    would start no CI and would discard the only evidence of managed entry.
    """
    if report_context is not None:
        report_context.release_reason = reason
        report_context.recovery_capable = recovery_capable
    if not recovery_capable:
        if report_context is not None:
            report_context.recovery_incapable_fallback = True
        message = (
            f"Managed-CI v2 could not activate ({reason}); this run made no `{MANAGED_LABEL}` "
            "label DELETE because the base workflow does not prove an unlabeled "
            "pull_request recovery route. "
            + _measured_state_line(runner, config, pr_number)
        )
        if config.managed_ci:
            command = managed_ci_recovery_commands.render_managed_ci_resume_command(config, pr_number=pr_number, managed_ci=True)
            raise AgentLoopError(
                f"--managed-ci requested qualification, but activation failed. {message} "
                f"This run did NOT qualify the head of PR #{pr_number}. Restore the base "
                f"workflow's unlabeled recovery route, then rerun `{command}`."
            )
        log(config, f"PR #{pr_number}: {message}")
        return
    if report_context is not None:
        report_context.label_release_attempted = True
    recovered_owned = _require_recovered_label_ownership(
        runner, config=config, pr_number=pr_number, context=report_context,
    )
    result = _label_remove(
        runner, config=config, pr_number=pr_number, label_name=MANAGED_LABEL,
        owned_event_id=recovered_owned,
    ).result
    if result.returncode != 0:
        message = (
            f"Managed-CI v2 could not activate ({reason}); the `{MANAGED_LABEL}` label DELETE "
            "was not confirmed."
        )
        if report_context is None:
            message += " " + _measured_state_line(runner, config, pr_number)
        raise AgentLoopError(message + failure_suffix(result))
    if report_context is not None:
        report_context.label_released = True
    log(config, f"PR #{pr_number}: label DELETE acknowledged ({reason})")
    if config.managed_ci:
        # This helper is used before an issue-created handoff has been
        # authenticated. It must not advertise the exceptional issue-created
        # grant for an adoption/source-managed invocation or invent an issue
        # scope from configuration. Recovery with an authenticated issue hint
        # is rendered by _release_for_ordinary_recovery instead.
        remedy = "Rerun the managed-CI command after restoring the required workflow state."
        state = "" if report_context is not None else _measured_state_line(runner, config, pr_number) + " "
        raise AgentLoopError(
            f"--managed-ci requested qualification, but activation failed ({reason}). "
            f"{state}This run did NOT qualify the head of PR #{pr_number}. " + remedy
        )


@managed_ci_api._surfaces_list_failures
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
    fresh_issue_number: int | None = None,
    fresh_authorization_allowed: bool = False,
    protection_state: str | None = None,
    report_context: _FailedActivationContext | None = None,
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

    With ``report_context`` (activation callers) the release records its
    DELETE on the context and leaves the post-failure state to the guard's
    measured report.  Without it (the dispatch-time caller) the return value,
    capability, and DELETE behavior are unchanged and only a context-free
    measured current-state line is added.
    """
    if report_context is not None:
        report_context.release_reason = reason
        report_context.recovery_capable = recovery_capable
    if not recovery_capable:
        if report_context is not None:
            report_context.recovery_incapable_fallback = True
        log(
            config,
            f"PR #{pr_number}: ordinary recovery was not selected because the base workflow "
            "does not prove an unlabeled pull_request trigger",
        )
        if report_context is None:
            log(config, _measured_state_line(runner, config, pr_number))
        return None
    prior_run_ids: set[int] = set()
    try:
        prior_run_ids = _workflow_run_ids(runner, config=config, head_sha=expected_head_sha)
    except AgentLoopError:
        log(config, f"PR #{pr_number}: could not baseline ordinary recovery runs ({reason})")
    if active_event is not None:
        current_event = managed_ci_api._active_managed_label_event(runner, config=config, pr_number=pr_number)
        if current_event != active_event:
            if report_context is not None:
                report_context.release_reason_kind = "event-changed"
            raise AgentLoopError(
                f"PR #{pr_number} managed-label ownership changed before ordinary release; "
                "leaving the label untouched and no merge will be attempted."
            )
    if report_context is not None:
        report_context.label_release_attempted = True
    recovered_owned = _require_recovered_label_ownership(
        runner, config=config, pr_number=pr_number, context=report_context,
    )
    result = _label_remove(
        runner, config=config, pr_number=pr_number, label_name=MANAGED_LABEL,
        owned_event_id=active_event[0] if active_event is not None else recovered_owned,
    ).result
    if result.returncode != 0:
        message = f"Managed-CI v2 could not activate ({reason}); the label DELETE was not confirmed."
        if report_context is None:
            message += " " + _measured_state_line(runner, config, pr_number)
        raise AgentLoopError(message + failure_suffix(result))
    if report_context is not None:
        report_context.label_released = True
    log(
        config,
        f"PR #{pr_number}: label DELETE acknowledged; ordinary unlabeled recovery selected ({reason})",
    )
    if config.managed_ci:
        fresh = None
        # With a known state, the fresh command must pass that state's own
        # waiver gate.  Before assessment, the fresh command re-runs its gate.
        waiver_allows_fresh = (
            protection_state in managed_ci_protection.waivable_protection_states(config)
            if protection_state is not None
            else config.allow_unprotected_managed_ci
        )
        if (
            fresh_authorization_allowed
            and waiver_allows_fresh
            and fresh_issue_number is not None
            and _reason_allows_fresh_issue_created_authorization(reason)
        ):
            fresh = managed_ci_recovery_commands.render_managed_ci_resume_command(
                config,
                pr_number=pr_number,
                issue_number=fresh_issue_number,
                managed_ci=True,
                fresh_authorization=True,
                fresh_issue_number=fresh_issue_number,
            )
        if report_context is not None and fresh is not None:
            report_context.resume_command = fresh
        remedy = (
            "Use the explicit fresh issue-created authorization command: "
            f"`{fresh}`."
            if fresh is not None
            else "Restore the stated managed-CI prerequisite before retrying; no "
            "issue-created authorization grant was inferred for this failure."
            + (_missing_waiver_note(config, protection_state) if protection_state else "")
        )
        state = "" if report_context is not None else _measured_state_line(runner, config, pr_number) + " "
        raise AgentLoopError(
            f"--managed-ci requested qualification, but activation failed ({reason}). "
            f"{state}This run did NOT qualify the head of PR #{pr_number}. " + remedy
        )
    if report_context is None:
        log(config, _measured_state_line(runner, config, pr_number))
    return OrdinaryRecoveryCapability(
        pr_number=pr_number,
        repository=config.repo,
        base_ref=base_ref,
        expected_head_sha=expected_head_sha,
        released_label_event_id=active_event[0] if active_event is not None else None,
        released_at=int(time.time()),
        prior_run_ids=frozenset(prior_run_ids),
    )


def _missing_waiver_note(config: AgentLoopConfig, protection_state: str) -> str:
    """Name the waiver flag(s) a known non-strict state still needs, if any."""
    if (
        protection_state not in _OVERRIDE_PROTECTION_STATES
        or protection_state in managed_ci_protection.waivable_protection_states(config)
    ):
        return ""
    return (
        f" Protection is {protection_state}; fresh authorization is unavailable without "
        f"{managed_ci_protection.waiver_flags_for_protection(protection_state)}."
    )


def _reason_allows_fresh_issue_created_authorization(reason: str) -> bool:
    """Return whether an explicit issue-created grant can repair ``reason``.

    Ordinary release is also used by adoption and source-managed paths.  A
    generic ``managed_ci`` flag is therefore not enough to advertise the
    exceptional issue-created grant: ownership, waiver, tuple, and publication
    failures have different remedies and several cannot be repaired by a new
    grant at all.
    """
    return reason in {
        "no fully bound actor-owned issue-created authorization reaches the live head",
        "the durable issue-created authorization is bound to an older head; "
        "no trusted continuity record authorizes the live head",
    }


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
    pr = managed_ci_api._api_json(
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
        or managed_ci_api._active_managed_label_event(runner, config=config, pr_number=capability.pr_number) is not None
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
        record = managed_ci_authorization.parse_managed_ci_override_record(
            body,
            surface=PR_COMMENT_SURFACE,
            schema="audit",
            required=False,
        )
    except AgentLoopError:
        return None
    return None if record is None else record.field_map()


def verify_managed_pr_plan_binding(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    issue_number: int,
    live_head: str | None,
    approved_plan_hash: str,
    retired_plan_hashes: Collection[str] = (),
) -> None:
    """Require the PR-side authorization chain to bind this PR to the plan.

    Managed-CI issue-created runs deliberately do not post the issue-side
    handoff record; their durable binding is the trusted actor's PR-comment
    authorization chain (#966).  Qualification reads the binding from there
    with the same substance the issue-side check enforces: the chain must name
    this repository, issue, PR, and approved plan hash, and its unique terminal
    must be the live head, linked back to a creation or fresh root.
    """
    failure = (
        "Approved-plan/handoff identity changed or disappeared during PR qualification; "
        "the managed-CI authorization chain does not bind PR #{pr} to approved plan "
        "{plan} at the live head ({reason}). Stale approvals cannot be used for this head."
    )

    def fail(reason: str) -> AgentLoopError:
        return AgentLoopError(
            failure.format(pr=pr_number, plan=approved_plan_hash, reason=reason)
        )

    trusted_actor = (config.managed_ci_trusted_actor or "").strip()
    if not trusted_actor:
        raise fail("no trusted managed-CI actor is configured")
    if not live_head:
        raise fail("the live head is unknown")
    retired = frozenset(retired_plan_hashes)
    if approved_plan_hash in retired:
        retired = frozenset()
    comments, reason = managed_ci_api._api_list_detailed(
        runner, config, f"repos/{config.repo}/issues/{pr_number}/comments?per_page=100"
    )
    if comments is None:
        raise fail(managed_ci_api._with_reason("the PR authorization comments could not be inspected", reason))
    records: list[tuple[int, ManagedCiIssueAuthorization]] = []
    for comment in comments:
        body = managed_ci_api._normalized_comment_body(comment, config=config)
        if ISSUE_AUTHORIZATION_MARKER not in body:
            continue
        try:
            authorization = managed_ci_authorization.parse_issue_created_authorization_comment(body)
        except AgentLoopError as exc:
            raise fail("an authorization record is malformed") from exc
        if authorization is None:
            continue
        user = comment.get("user") if isinstance(comment.get("user"), dict) else {}
        comment_id = comment.get("id")
        login = user.get("login")
        if (
            not isinstance(comment_id, int)
            or not isinstance(login, str)
            or login.casefold() != trusted_actor.casefold()
            or authorization.actor_login.casefold() != trusted_actor.casefold()
            or user.get("id") != authorization.actor_id
        ):
            raise fail("an authorization record is not authored by the trusted actor")
        if (
            authorization.repository.casefold() != config.repo.casefold()
            or authorization.pr_number != pr_number
            or authorization.issue_number != issue_number
            or (config.base and authorization.base_ref != config.base)
        ):
            raise fail("an authorization record names a different repository, issue, PR, or base")
        records.append((comment_id, authorization))
    superseded = managed_ci_authorization._superseded_authorization_ids(records)
    records = [item for item in records if item[0] not in superseded]
    if not records:
        raise fail("no authorization record exists")
    by_comment_id = dict(records)
    children_by_predecessor: dict[ManagedCiIssueAuthorization, set[str]] = {}
    for _comment_id, authorization in records:
        if authorization.kind == "continuity" and authorization.predecessor_comment_id is not None:
            predecessor = by_comment_id.get(authorization.predecessor_comment_id)
            if predecessor is not None and predecessor.head_sha == authorization.predecessor_head:
                children_by_predecessor.setdefault(predecessor, set()).add(authorization.head_sha)
    if any(len(heads) > 1 for heads in children_by_predecessor.values()):
        raise fail("the authorization chain forks")
    comments_by_id = managed_ci_authorization._comments_by_id(comments)
    # Recovery-shaped chains (#1367) are admitted only with the same actor,
    # waiver, config-aware protection and label-event provenance publication
    # applies.  Load the actor-owned label history only when a continuity
    # record extends a fresh grant, the one place that shape can occur; a
    # missing history leaves it None, so such a chain fails closed.
    actor_logins = {
        str((comments_by_id.get(comment_id) or {}).get("user", {}).get("login"))
        for comment_id, _authorization in records
    }
    actor_ids = {authorization.actor_id for _comment_id, authorization in records}
    bases = {authorization.base_ref for _comment_id, authorization in records}
    valid_label_event_ids: frozenset[int] | None = None
    opening_override_nonce: str | None = None
    if (
        len(actor_logins) == 1
        and len(actor_ids) == 1
        and any(
            authorization.kind == "continuity"
            and (predecessor := by_comment_id.get(authorization.predecessor_comment_id or 0))
            is not None
            and predecessor.kind == "fresh"
            for _comment_id, authorization in records
        )
    ):
        valid_label_event_ids = managed_ci_authorization.load_actor_label_event_ids(
            runner,
            config=config,
            pr_number=pr_number,
            actor_login=next(iter(actor_logins)),
            actor_id=next(iter(actor_ids)),
        )
        try:
            live_pr = managed_ci_api._api_json(
                runner, config, f"repos/{config.repo}/pulls/{pr_number}", quiet=True
            )
            opening_override_nonce = managed_ci_authorization._opening_override_nonce(
                live_pr.get("body") if isinstance(live_pr.get("body"), str) else None
            )
        except AgentLoopError:
            # Unreadable: a recovery-shaped chain's creation root fails closed.
            opening_override_nonce = None
    chain_context = ChainAuthContext(
        config=config,
        actor_login=(
            next(iter(actor_logins)) if len(actor_logins) == 1 else trusted_actor
        ),
        actor_id=next(iter(actor_ids)) if len(actor_ids) == 1 else -1,
        repository=config.repo,
        issue_number=issue_number,
        pr_number=pr_number,
        base_ref=config.base or (next(iter(bases)) if len(bases) == 1 else ""),
        approved_plan_hash=approved_plan_hash,
        retired_plan_hashes=retired,
        valid_label_event_ids=valid_label_event_ids,
        opening_override_nonce=opening_override_nonce,
    )
    from .reviewer_seats import reviewer_seat_binding

    # Recovery links bind seat-bound reviews; re-check them against this
    # run's configured board exactly as publication did.
    expected_seat_binding = reviewer_seat_binding(config)
    bound_terminals: set[ManagedCiIssueAuthorization] = set()
    for terminal_comment_id, terminal in records:
        if terminal.head_sha != live_head:
            continue
        authenticated = managed_ci_authorization._authenticate_authorization_chain(
            terminal_comment_id,
            by_comment_id,
            comments_by_id,
            comments,
            chain_context,
            expected_seat_binding=expected_seat_binding,
        )
        if authenticated is None:
            continue
        chain = list(authenticated)
        if all(
            record.approved_plan_hash != approved_plan_hash
            and record.approved_plan_hash in retired
            for record in chain
        ):
            # A whole chain under verified retired plans is history left at
            # the live head by a signed rebind that kept the head (#993).
            continue
        if any(record.approved_plan_hash != approved_plan_hash for record in chain):
            raise fail("the authorization chain names a different approved plan")
        bound_terminals.add(terminal)
    if not bound_terminals:
        raise fail("no authenticated chain reaches the live head")
    if len(bound_terminals) != 1:
        # Byte-equivalent retries collapse in the set.  As in resume, a single
        # fresh grant at the creation head supersedes that creation record;
        # any other pair of distinct live-head terminals is ambiguous.
        fresh_roots = {record for record in bound_terminals if record.kind == "fresh"}
        if not (
            {record.kind for record in bound_terminals} <= {"creation", "fresh"}
            and len(fresh_roots) == 1
        ):
            raise fail("more than one distinct authorization terminal reaches the live head")


def _find_resume_audit(
    runner: Runner, *, config: AgentLoopConfig, pr_number: int, actor_login: str, actor_id: int,
    base_ref: str, issue_number: int | None = None, live_head: str | None = None,
    expected_handoff: AuthenticatedIssueCreatedHandoff | None = None,
    expected_protection: str | None = None,
    require_actor_owned_label_event: bool = False,
) -> tuple[int, dict[str, str]] | None:
    """Find one actor-owned PR-comment authorization for resume provenance."""
    comments = managed_ci_api._api_list(runner, config, f"repos/{config.repo}/issues/{pr_number}/comments?per_page=100")
    if comments is None:
        return None
    if (
        expected_handoff is not None
        and expected_protection is not None
        and expected_handoff.protection_mode != expected_protection
    ):
        # Every record would have to equal both values, so none can match.
        return None
    valid_label_event_ids: set[int] | None = None
    if expected_handoff is not None or require_actor_owned_label_event:
        valid_label_event_ids = managed_ci_authorization._actor_owned_label_event_ids(
            runner, config=config, pr_number=pr_number,
            actor_login=actor_login, actor_id=actor_id,
        )
        if valid_label_event_ids is None:
            return None
    if expected_protection is not None:
        check_protection: str | None = expected_protection
    elif expected_handoff is not None:
        check_protection = expected_handoff.protection_mode
    else:
        check_protection = None
    candidates: list[tuple[int, dict[str, str]]] = []
    authorization_candidates: list[tuple[int, ManagedCiIssueAuthorization]] = []
    malformed = False
    superseded = managed_ci_authorization._superseded_authorization_comment_ids(
        comments, config=config, actor_login=actor_login, actor_id=actor_id,
    )
    for comment in comments:
        body = managed_ci_api._normalized_comment_body(comment, config=config)
        if ISSUE_AUTHORIZATION_MARKER in body:
            if comment.get("id") in superseded:
                # Retired by a later fresh grant (#1065): history only.
                continue
            try:
                authorization = managed_ci_authorization.parse_issue_created_authorization_comment(body)
            except AgentLoopError:
                malformed = True
                continue
            if authorization is None or not managed_ci_authorization._authorization_record_valid_for_handoff(
                authorization,
                comment,
                config=config,
                handoff=expected_handoff,
                actor_login=actor_login,
                actor_id=actor_id,
                base_ref=base_ref,
                pr_number=pr_number,
                issue_number=issue_number,
                valid_label_event_ids=valid_label_event_ids,
                check_protection=check_protection,
                plan_scope_authenticated=True,
            ):
                malformed = True
                continue
            if managed_ci_authorization._is_retired_plan_history(authorization, expected_handoff):
                # History under a plan a verified signed rebind replaced (#993):
                # neither a competing grant nor part of the live chain.
                continue
            cid = comment["id"]
            candidates.append(
                (
                    cid,
                    {
                        "nonce": authorization.nonce,
                        "repo": authorization.repository,
                        "base": authorization.base_ref,
                        "head": authorization.head_sha,
                        "protection": authorization.protection,
                        "active_label_event_id": str(authorization.label_event_id),
                        "kind": authorization.kind,
                        "issue": str(authorization.issue_number),
                        "pr": str(authorization.pr_number),
                    },
                )
            )
            authorization_candidates.append((cid, authorization))
            continue
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
    if malformed:
        return None
    # The legacy trailer-only audit is useful as historical diagnostic text,
    # but it is not a substitute for the centrally bound authorization record.
    # If no versioned record exists, it must not authorize an issue-created
    # resume merely because its comment author is trusted.
    if candidates and not authorization_candidates and (
        expected_handoff is not None or require_actor_owned_label_event
    ):
        return None
    if authorization_candidates:
        # Exact byte-equivalent retries are harmless.  Distinct creation or
        # continuity payloads are authoritative only when they form one
        # unique root-to-live-head chain; forks and unexplained heads fail
        # closed instead of letting the newest comment win.
        unique: dict[ManagedCiIssueAuthorization, list[int]] = {}
        for comment_id, authorization in authorization_candidates:
            unique.setdefault(authorization, []).append(comment_id)
        records = [
            (comment_id, authorization)
            for authorization, comment_ids in unique.items()
            for comment_id in comment_ids
        ]
        if live_head is None:
            selected = sorted(records, key=lambda item: item[0])[-1]
            return (
                selected[0],
                {
                    "nonce": selected[1].nonce,
                    "repo": selected[1].repository,
                    "base": selected[1].base_ref,
                    "head": selected[1].head_sha,
                    "protection": selected[1].protection,
                    "active_label_event_id": str(selected[1].label_event_id),
                    "kind": selected[1].kind,
                    "issue": str(selected[1].issue_number),
                    "pr": str(selected[1].pr_number),
                },
            )
        by_comment_id = {comment_id: authorization for comment_id, authorization in records}
        by_head: dict[str, list[tuple[int, ManagedCiIssueAuthorization]]] = {}
        for comment_id, authorization in records:
            by_head.setdefault(authorization.head_sha, []).append((comment_id, authorization))
        # A retry may publish byte-equivalent copies of one authorization with
        # different comment IDs.  Treat those IDs as aliases of one logical
        # predecessor before checking outgoing edges; otherwise a fork can be
        # hidden by putting each child on a different duplicate copy.
        children_by_predecessor: dict[ManagedCiIssueAuthorization, set[str]] = {}
        for _comment_id, authorization in records:
            if (
                authorization.kind == "continuity"
                and authorization.predecessor_comment_id is not None
                and authorization.predecessor_head is not None
            ):
                predecessor = by_comment_id.get(authorization.predecessor_comment_id)
                if predecessor is not None and predecessor.head_sha == authorization.predecessor_head:
                    children_by_predecessor.setdefault(predecessor, set()).add(
                        authorization.head_sha
                    )
        if any(len(heads) > 1 for heads in children_by_predecessor.values()):
            return None
        terminal = by_head.get(live_head, [])
        chain_context = ChainAuthContext(
            config=config,
            actor_login=actor_login,
            actor_id=actor_id,
            repository=config.repo,
            issue_number=issue_number,
            pr_number=pr_number,
            base_ref=base_ref,
            approved_plan_hash=(
                expected_handoff.approved_plan_hash if expected_handoff is not None else None
            ),
            retired_plan_hashes=(
                frozenset(expected_handoff.retired_plan_hashes)
                if expected_handoff is not None else frozenset()
            ),
            valid_label_event_ids=(
                frozenset(valid_label_event_ids)
                if valid_label_event_ids is not None else None
            ),
            check_protection=check_protection,
            handoff=expected_handoff,
        )
        comments_by_id = managed_ci_authorization._comments_by_id(comments)
        from .reviewer_seats import reviewer_seat_binding

        expected_seat_binding = reviewer_seat_binding(config)
        valid_terminals: list[tuple[int, ManagedCiIssueAuthorization]] = []
        for terminal_comment_id, terminal_record in terminal:
            if managed_ci_authorization._authenticate_authorization_chain(
                terminal_comment_id,
                by_comment_id,
                comments_by_id,
                comments,
                chain_context,
                expected_seat_binding=expected_seat_binding,
            ) is not None:
                valid_terminals.append((terminal_comment_id, terminal_record))
        if expected_handoff is not None and expected_handoff.active_label_event_id is not None:
            # Only the terminal record must point at the active event.  Older
            # roots and continuity ancestors may legitimately reference prior
            # actor-owned label applications in the same chain.
            valid_terminals = [
                item
                for item in valid_terminals
                if item[1].label_event_id == expected_handoff.active_label_event_id
            ]
        if expected_handoff is not None and expected_handoff.authorization_comment_id is not None:
            valid_terminals = [
                item
                for item in valid_terminals
                if item[0] == expected_handoff.authorization_comment_id
            ]
        distinct_terminals = {record for _comment_id, record in valid_terminals}
        if len(distinct_terminals) != 1:
            # A fresh operator grant at the exact creation head supersedes the
            # creation checkpoint.  This compatibility case repairs records
            # produced by older clients which published both; multiple grants
            # or any competing continuity terminal remain ambiguous.
            root_kinds = {record.kind for record in distinct_terminals}
            fresh_roots = [
                item for item in valid_terminals if item[1].kind == "fresh"
            ]
            creation_roots = [
                item for item in valid_terminals if item[1].kind == "creation"
            ]
            if root_kinds <= {"creation", "fresh"} and len(
                {record for _cid, record in fresh_roots}
            ) == 1 and creation_roots:
                valid_terminals = fresh_roots
            else:
                return None
        if valid_terminals:
            selected = max(valid_terminals, key=lambda item: item[0])
            return (
                selected[0],
                {
                    "nonce": selected[1].nonce,
                    "repo": selected[1].repository,
                    "base": selected[1].base_ref,
                    "head": selected[1].head_sha,
                    "protection": selected[1].protection,
                    "active_label_event_id": str(selected[1].label_event_id),
                    "kind": selected[1].kind,
                    "issue": str(selected[1].issue_number),
                    "pr": str(selected[1].pr_number),
                },
            )
        return None
    if not candidates:
        return None
    # Multiple valid audits are normal after a safe retry. The newest actor-
    # owned record is the latest provenance, while all malformed/mismatched
    # records still fail closed above.
    return sorted(candidates, key=lambda item: item[0])[-1]


@managed_ci_api._surfaces_list_failures
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
    who = managed_ci_api._api_json(runner, config, "user")
    actor_login = who.get("login") if isinstance(who.get("login"), str) else None
    actor_id = _strict_int(who.get("id"))
    if not actor_login or actor_id is None or actor_login.casefold() != configured.casefold():
        return None
    advertised = managed_ci_api._advertised_managed_actor(runner, config)
    if not advertised or advertised.casefold() != actor_login.casefold():
        return None

    # Resolve workflow source from the protected base ref.  The dispatch later
    # uses the same ref, so PR-authored YAML cannot redefine qualification.
    base_ref = metadata.base_branch or config.base
    if not base_ref:
        return None
    dispatch_ref = _workflow_source_ref(runner, config, base_ref)
    base_workflow = runner.run(
        [
            config.gh_cmd,
            "api",
            f"repos/{config.repo}/contents/.github/workflows/{WORKFLOW_FILE}?ref={dispatch_ref}",
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
    base_problem = trusted_base_problem(
        runner, gh_cmd=config.gh_cmd, repo=config.repo, cwd=github_api_cwd(), base=base_ref,
        default_branch=dispatch_ref, workflow_text=workflow_text,
    )
    if base_problem is not None:
        raise AgentLoopError(f"Managed CI cannot target this PR's base: {base_problem}.")
    ordinary_recovery_capable = (
        RECOVERY_MARKER in workflow_text and "pull_request" in workflow_text and "unlabeled" in workflow_text
    )
    visible_intent_capable = VISIBLE_INTENT_MARKER in workflow_text
    host_footer_capable = HOST_FOOTER_INTENT_MARKER in workflow_text

    pr = managed_ci_api._api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}")
    head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
    base = pr.get("base") if isinstance(pr.get("base"), dict) else {}
    author = pr.get("user") if isinstance(pr.get("user"), dict) else {}
    head_repo_name = _str_field(pr, "head", "repo", "full_name")
    live_sha = head.get("sha") if isinstance(head.get("sha"), str) else None
    head_ref = head.get("ref") if isinstance(head.get("ref"), str) else None
    author_login = author.get("login") if isinstance(author.get("login"), str) else None
    author_id = _strict_int(author.get("id"))
    pr_state = pr.get("state")
    # Reserved branches are generated by the typed creation prompt. A mere
    # lookalike branch cannot pass without all the other authenticated fields.
    immutable_tuple = (
        head_repo_name is not None and head_repo_name.casefold() == config.repo.casefold()
        and author_login is not None and author_login.casefold() == actor_login.casefold()
        and author_id == actor_id
        and isinstance(head_ref, str) and head_ref.startswith("agent-loop/managed-")
        and base.get("ref") == base_ref
        and live_sha is not None and live_sha == metadata.head_sha
        # Type-safe: an array or object state must not raise TypeError here.
        and not (isinstance(pr_state, str) and pr_state in {"closed", "CLOSED"})
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

    # Strict pre-write gate (#1067).  It runs only for an authenticated
    # managed candidate, after the adoption short-circuit, so a PR that is not
    # ours keeps its ``None`` route whatever its labels, draft, or state.
    if pr_state != "open":
        raise AgentLoopError(
            f"PR #{pr_number} is not provably open; no label, draft, or readiness write was made."
        )
    strict_labels = _strict_label_names(pr)
    entry = _PrStateSnapshot.from_payload(pr)
    if strict_labels is None or entry.draft is None:
        raise AgentLoopError(
            f"PR #{pr_number} labels or draft state could not be read strictly; no label, draft, "
            "or readiness write was made."
        )
    labels = set(strict_labels)

    # Every exit after this point that follows a PR mutation recorded on the
    # context carries one measured before/now report (#1067).
    context = _FailedActivationContext(entry=entry, recovery_capable=ordinary_recovery_capable)
    with _failed_activation_guard(runner, config=config, pr_number=pr_number, context=context):
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
        issue_hint = _issue_created_rendering_issue_number(
            managed_resume, origin=origin, head_ref=head_ref
        )

        # Authenticate durable issue-created authority before any ready/draft or
        # label mutation.  In particular, draft/unlabeled recovery must not POST a
        # new managed label and only then discover missing, stale, or forged
        # provenance.
        protection = ProtectionAssessment("strict", "legacy", "dispatch-only legacy workflow")
        resume_audit_id: int | None = None
        resume_provenance_head: str | None = None
        if managed_resume is not None and MANAGED_LABEL in labels:
            preflight_event = managed_ci_api._active_managed_label_event(
                runner, config=config, pr_number=pr_number
            )
            if preflight_event is None:
                recovery = _release_for_ordinary_recovery(
                    runner, config=config, pr_number=pr_number, base_ref=base_ref,
                    report_context=context,
                    expected_head_sha=live_sha, active_event=None,
                    reason="the active managed-label event is temporarily unreadable",
                    recovery_capable=ordinary_recovery_capable,
                    fresh_issue_number=issue_hint,
                    fresh_authorization_allowed=origin == "issue-created",
                )
                return _ordinary_fallback_contract(runner, config=config, pr_number=pr_number, context=context, recovery=recovery)
            if (
                preflight_event[1].casefold() != actor_login.casefold()
                or preflight_event[2] != actor_id
            ):
                context.release_reason_kind = "foreign-actor"
                _release_for_ordinary_recovery(
                    runner, config=config, pr_number=pr_number, base_ref=base_ref,
                    report_context=context,
                    expected_head_sha=live_sha, active_event=preflight_event,
                    reason="the active managed-label event is not actor-owned",
                    recovery_capable=ordinary_recovery_capable,
                    fresh_issue_number=issue_hint,
                    fresh_authorization_allowed=origin == "issue-created",
                )
                return _ordinary_fallback_contract(runner, config=config, pr_number=pr_number, context=context, recovery=None)
        if "pull_request" in workflow_text:
            protection = managed_ci_protection.assess_exact_head_protection(
                runner,
                context=ManagedCiProbeContext(config.repo, config.gh_cmd, active_workdir(config)),
                base=base_ref,
            )
            if protection.state != "strict" and managed_resume is not None:
                handoff = managed_resume.issue_created_handoff
                if handoff is not None:
                    protection = managed_ci_protection.reconcile_live_protection(
                        handoff.protection_mode, protection, config=config,
                    )
                if protection.state not in managed_ci_protection.waivable_protection_states(config):
                    reason = "strict protection is unavailable and the explicit waiver is absent"
                    if MANAGED_LABEL in labels:
                        existing_event = managed_ci_api._active_managed_label_event(
                            runner, config=config, pr_number=pr_number
                        )
                        recovery = _release_for_ordinary_recovery(
                            runner, config=config, pr_number=pr_number, base_ref=base_ref,
                            report_context=context,
                            expected_head_sha=live_sha, active_event=existing_event, reason=reason,
                            recovery_capable=ordinary_recovery_capable,
                            fresh_issue_number=issue_hint,
                            fresh_authorization_allowed=origin == "issue-created",
                            protection_state=protection.state,
                        )
                        return _ordinary_fallback_contract(runner, config=config, pr_number=pr_number, context=context, recovery=recovery)
                    waiver_hint = (
                        f" Protection is {protection.state} ({protection.detail}); it requires "
                        f"{managed_ci_protection.waiver_flags_for_protection(protection.state)}."
                        if protection.state in _OVERRIDE_PROTECTION_STATES
                        else ""
                    )
                    raise AgentLoopError(
                        f"--managed-ci requested qualification, but activation failed ({reason}). "
                        f"PR #{pr_number} was left unchanged and this run did NOT qualify its head."
                        f"{waiver_hint} Restore the stated managed-CI prerequisite before retrying."
                    )
                prior_audit = (
                    _find_resume_audit(
                        runner, config=config, pr_number=pr_number,
                        actor_login=actor_login, actor_id=actor_id, base_ref=base_ref,
                        issue_number=issue_hint, live_head=live_sha,
                        expected_handoff=handoff, expected_protection=protection.state,
                        require_actor_owned_label_event=True,
                    )
                    if handoff is not None
                    else None
                )
                if prior_audit is None:
                    reason = "no fully bound actor-owned issue-created authorization reaches the live head"
                    if MANAGED_LABEL in labels:
                        existing_event = managed_ci_api._active_managed_label_event(
                            runner, config=config, pr_number=pr_number
                        )
                        recovery = _release_for_ordinary_recovery(
                            runner, config=config, pr_number=pr_number, base_ref=base_ref,
                            report_context=context,
                            expected_head_sha=live_sha, active_event=existing_event, reason=reason,
                            recovery_capable=ordinary_recovery_capable,
                            fresh_issue_number=issue_hint,
                            fresh_authorization_allowed=origin == "issue-created",
                            protection_state=protection.state,
                        )
                        return _ordinary_fallback_contract(runner, config=config, pr_number=pr_number, context=context, recovery=recovery)
                    if origin == "issue-created" and issue_hint is not None:
                        fresh = managed_ci_recovery_commands.render_managed_ci_resume_command(
                            config, pr_number=pr_number, issue_number=issue_hint,
                            managed_ci=True, fresh_authorization=True,
                            fresh_issue_number=issue_hint,
                        )
                        remedy = f"Use the explicit fresh issue-created authorization command: `{fresh}`."
                    else:
                        remedy = (
                            "Restore the source-managed authorization prerequisite; issue-created "
                            "fresh authorization cannot adopt this PR."
                        )
                    raise AgentLoopError(
                        f"--managed-ci requested qualification, but activation failed ({reason}). "
                        f"PR #{pr_number} was left unchanged and this run did NOT qualify its head. "
                        + remedy
                    )
                resume_audit_id, prior_fields = prior_audit
                resume_provenance_head = prior_fields.get("head")
                if prior_fields.get("kind") and resume_provenance_head != live_sha:
                    reason = (
                        "the durable issue-created authorization is bound to an older head; "
                        "no trusted continuity record authorizes the live head"
                    )
                    if MANAGED_LABEL in labels:
                        existing_event = managed_ci_api._active_managed_label_event(
                            runner, config=config, pr_number=pr_number
                        )
                        recovery = _release_for_ordinary_recovery(
                            runner, config=config, pr_number=pr_number, base_ref=base_ref,
                            report_context=context,
                            expected_head_sha=live_sha, active_event=existing_event,
                            reason=reason, recovery_capable=ordinary_recovery_capable,
                            fresh_issue_number=issue_hint,
                            fresh_authorization_allowed=origin == "issue-created",
                            protection_state=protection.state,
                        )
                        return _ordinary_fallback_contract(runner, config=config, pr_number=pr_number, context=context, recovery=recovery)
                    if (
                        origin == "issue-created"
                        and protection.state in managed_ci_protection.waivable_protection_states(config)
                        and issue_hint is not None
                    ):
                        fresh = managed_ci_recovery_commands.render_managed_ci_resume_command(
                            config,
                            pr_number=pr_number,
                            issue_number=issue_hint,
                            managed_ci=True,
                            fresh_authorization=True,
                            fresh_issue_number=issue_hint,
                        )
                        remedy = f"Use the explicit fresh issue-created authorization command: `{fresh}`."
                    else:
                        remedy = (
                            "Restore the issue-created authorization prerequisite before retrying; "
                            "the live head is not qualified."
                            + _missing_waiver_note(config, protection.state)
                        )
                    raise AgentLoopError(
                        f"--managed-ci requested qualification, but activation failed ({reason}). "
                        f"PR #{pr_number} was left unchanged and this run did NOT qualify its head. "
                        + remedy
                    )

        # Strict protection removes the need for an unprotected waiver and its
        # versioned PR-comment authorization record. An unlabeled draft still
        # needs evidence that this exact issue-created PR previously entered the
        # managed lifecycle before this invocation reapplies the suppression
        # label. The authenticated strict tuple plus an actor-owned historical
        # label event is that separate strict-protection proof; waiver records are
        # deliberately not accepted or minted for this path.
        if (
            managed_resume is not None
            and origin == "issue-created"
            and lifecycle == "draft-unlabeled-reentry"
            and protection.state == "strict"
        ):
            historical_label_events = managed_ci_api._managed_label_event_history(
                runner,
                config=config,
                pr_number=pr_number,
                actor_login=actor_login,
                actor_id=actor_id,
            )
            if historical_label_events is None:
                reason = managed_ci_api._with_reason(
                    "the actor-owned managed-label history is temporarily unreadable",
                    managed_ci_api._last_list_failure(),
                )
            elif not historical_label_events:
                reason = "no actor-owned historical managed-label event authenticates strict re-entry"
            else:
                reason = None
            if reason is not None:
                command = managed_ci_recovery_commands.render_managed_ci_resume_command(
                    config, pr_number=pr_number, managed_ci=True,
                )
                raise AgentLoopError(
                    f"--managed-ci requested qualification, but activation failed because {reason}. "
                    f"PR #{pr_number} was left draft and unlabeled; no label, dispatch, readiness, "
                    f"or qualification write was made. Reapply `{MANAGED_LABEL}` as the configured "
                    f"trusted actor through the repository's managed-CI controls, then retry "
                    f"`{command}`."
                )

        # A successful explicit manual run leaves a managed PR ready and
        # unlabeled. Re-entry is a privileged mutation: it is allowed only when
        # the authenticated lifecycle record and the invocation both request
        # managed CI explicitly. Implicit auto-merge must leave this state alone.
        if lifecycle == "ready-unlabeled-reentry":
            if not config.managed_ci:
                command = managed_ci_recovery_commands.render_managed_ci_resume_command(
                    config, pr_number=pr_number, managed_ci=True,
                )
                raise AgentLoopError(
                    f"PR #{pr_number} is an authenticated managed {origin} PR in the ready/unlabeled "
                    "re-entry state. It was left unchanged; rerun with explicit `--managed-ci`: "
                    f"{command}"
                )
            undo = reconciled_pr_ready(
                runner, config=config, pr_number=pr_number,
                expected_head_sha=metadata.head_sha, undo=True,
            )
            # A zero exit is GitHub's acknowledgement of this invocation's
            # conversion; a nonzero exit is ambiguous (the PR may be draft by
            # another actor) and never licenses an automatic readiness write.
            if undo.returncode == 0:
                context.drafted_by_this_run = True
            else:
                context.ready_undo_ambiguous = True
            refreshed = managed_ci_api._api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}", quiet=True)
            refreshed_labels = _strict_label_names(refreshed)
            if (
                undo.returncode != 0
                or refreshed.get("draft") is not True
                or refreshed_labels is None
                or MANAGED_LABEL in refreshed_labels
            ):
                raise AgentLoopError(
                    f"--managed-ci re-entry could not make PR #{pr_number} provably draft and "
                    "unlabeled; the ready-to-draft request was "
                    f"{'acknowledged' if undo.returncode == 0 else 'not confirmed'} and no "
                    f"qualification was claimed.{failure_suffix(undo) if undo.returncode != 0 else ''}"
                )
            pr = refreshed
            head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
            base = pr.get("base") if isinstance(pr.get("base"), dict) else {}
            author = pr.get("user") if isinstance(pr.get("user"), dict) else {}
            labels = set(refreshed_labels)
            live_sha = head.get("sha") if isinstance(head.get("sha"), str) else None
            refreshed_repo = _str_field(pr, "head", "repo", "full_name")
            refreshed_state = pr.get("state")
            if (
                pr.get("draft") is not True
                or refreshed_state != "open"
                or base.get("ref") != base_ref
                or live_sha != metadata.head_sha
                or head.get("ref") != head_ref
                or refreshed_repo is None
                or refreshed_repo.casefold() != config.repo.casefold()
                or author.get("login") != actor_login
                or _strict_int(author.get("id")) != actor_id
                or MANAGED_LABEL in labels
            ):
                # There is no trusted active label to remove in the unlabeled
                # re-entry case; the guard appends the measured state.
                raise AgentLoopError(
                    f"--managed-ci re-entry reconstruction for PR #{pr_number} failed after "
                    "the ready-to-draft transition; the head is NOT qualified by this run."
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
            command = managed_ci_recovery_commands.render_managed_ci_resume_command(config, pr_number=pr_number, managed_ci=True)
            raise AgentLoopError(
                f"Managed-CI {origin} draft re-entry for PR #{pr_number} observed {state}; "
                f"the PR was left unchanged. Resume with `{command}` after restoring draft/unlabeled state."
            )
        elif lifecycle == "draft-unlabeled-reentry" and not config.managed_ci:
            command = managed_ci_recovery_commands.render_managed_ci_resume_command(config, pr_number=pr_number, managed_ci=True)
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
            command = managed_ci_recovery_commands.render_managed_ci_resume_command(config, pr_number=pr_number, managed_ci=True)
            raise AgentLoopError(
                f"Managed-CI {origin} resume for PR #{pr_number} observed {state}; "
                f"expected draft/labeled, draft/unlabeled, or authenticated ready/unlabeled. "
                f"The PR was left unchanged. Resume with `{command}`."
            )
        recovered_label_event_id: int | None = None
        if MANAGED_LABEL not in labels:
            if not ensure_managed_label(runner, config=config):
                raise AgentLoopError(f"Unable to create the `{MANAGED_LABEL}` label.")
            context.label_post_attempted = True
            label_outcome = _label_add(
                runner, config=config, pr_number=pr_number, label_name=MANAGED_LABEL
            )
            applied_label = label_outcome.result
            recovered_label_event_id = label_outcome.event_id
            context.recovered_label_event_id = recovered_label_event_id
            if applied_label.returncode != 0:
                # GitHub may still have applied the label; this is an
                # ambiguous request, never evidence of label ownership.
                context.label_post_ambiguous = True
                raise AgentLoopError(
                    f"Unable to apply `{MANAGED_LABEL}` to PR #{pr_number}; the label request "
                    f"was not confirmed.{failure_suffix(applied_label)}"
                )
            context.label_post_acknowledged = True
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
            active_event = managed_ci_api._active_managed_label_event(runner, config=config, pr_number=pr_number)
            if active_event is None:
                recovery = _release_for_ordinary_recovery(
                    runner, config=config, pr_number=pr_number, base_ref=base_ref,
                    report_context=context,
                    expected_head_sha=live_sha, active_event=active_event,
                    reason="the active managed-label event is temporarily unreadable",
                    recovery_capable=ordinary_recovery_capable,
                    fresh_issue_number=issue_hint,
                    fresh_authorization_allowed=origin == "issue-created",
                )
                return _ordinary_fallback_contract(runner, config=config, pr_number=pr_number, context=context, recovery=recovery)
            if recovered_label_event_id is not None and active_event[0] != recovered_label_event_id:
                # The label write was recovered as application E, but the timeline
                # now shows a different application; never claim or delete it.
                context.release_reason_kind = "event-changed"
                raise AgentLoopError(
                    f"PR #{pr_number} `{MANAGED_LABEL}` application changed after the recovered "
                    f"label write (recovered event {recovered_label_event_id}, observed "
                    f"{active_event[0]}); the label was left untouched and no qualification "
                    "is claimed."
                )
            if active_event[1].casefold() != actor_login.casefold() or active_event[2] != actor_id:
                context.release_reason_kind = "foreign-actor"
                _release_for_ordinary_recovery(
                    runner, config=config, pr_number=pr_number, base_ref=base_ref,
                    report_context=context,
                    expected_head_sha=live_sha, active_event=active_event,
                    reason="the active managed-label event is not actor-owned",
                    recovery_capable=ordinary_recovery_capable,
                    fresh_issue_number=issue_hint,
                    fresh_authorization_allowed=origin == "issue-created",
                )
                return _ordinary_fallback_contract(runner, config=config, pr_number=pr_number, context=context, recovery=None)

        # Only workflows with a pull_request route can suppress an opening matrix.
        # Legacy dispatch-only v2 deployments remain compatible, while modern
        # suppression-capable workflows must retain strict protection or supply a
        # live, explicit waiver and its auditable PR trailer.
        override_nonce: str | None = None
        if "pull_request" in workflow_text:
            if protection.state != "strict" and managed_resume is None:
                body = pr.get("body") if isinstance(pr.get("body"), str) else ""
                if protection.state not in managed_ci_protection.waivable_protection_states(config):
                    _restore_ordinary_ci_after_v2_fallback(
                        runner, config=config, pr_number=pr_number,
                        reason="strict protection or the explicit override is unavailable",
                        report_context=context,
                        recovery_capable=ordinary_recovery_capable,
                    )
                    _fallback_state_report(runner, config=config, pr_number=pr_number, context=context)
                    return None
                try:
                    override = managed_ci_authorization.parse_managed_ci_override_record(
                        body,
                        surface=PR_BODY_SURFACE,
                        schema="body",
                        required=True,
                        expected_nonce=config.managed_ci_expected_override_nonce,
                        additional_allowed_tokens=(
                            frozenset({"AGENT_MANAGED_PR_SOURCE_V1"})
                            if origin == "source-managed" else frozenset()
                        ),
                    )
                except AgentLoopError as error:
                    _restore_ordinary_ci_after_v2_fallback(
                        runner, config=config, pr_number=pr_number,
                        reason=str(error),
                        report_context=context,
                        recovery_capable=ordinary_recovery_capable,
                    )
                    _fallback_state_report(runner, config=config, pr_number=pr_number, context=context)
                    return None
                assert override is not None
                override_nonce = override.nonce
                # The bare protocol trailer is anchored to a line start, so the
                # label paragraph above it does not move the matched span.
                audit_label = protocol_record_label(
                    "managed_ci_override_audit", pr_number=pr_number, head_sha=live_sha
                )
                audit_body = TrustedBody.canonical(
                    (
                        f"{audit_label}\n\n"
                        f"{UNPROTECTED_OVERRIDE_TRAILER} nonce={override_nonce} repo={config.repo} "
                        f"base={base_ref} head={live_sha} protection={protection.state}\n\n"
                        "Voluntary gate: GitHub cannot prevent manual merges, other automation, "
                        "compromised credentials, or an agent-loop defect from bypassing it."
                    ),
                    expected_tokens=(UNPROTECTED_OVERRIDE_TRAILER,),
                )
                try:
                    audit_id = post_verified_trusted_pr_protocol_comment(
                        runner,
                        config=config,
                        pr_number=pr_number,
                        body=audit_body,
                        expected_author_login=actor_login,
                        expected_author_id=actor_id,
                    )
                except AgentLoopError:
                    _restore_ordinary_ci_after_v2_fallback(
                        runner, config=config, pr_number=pr_number,
                        reason="the override audit comment could not be recorded and verified",
                        report_context=context,
                        recovery_capable=ordinary_recovery_capable,
                    )
                    _fallback_state_report(runner, config=config, pr_number=pr_number, context=context)
                    return None
            else:
                audit_id = None
        else:
            audit_id = None

        if managed_resume is not None:
            # Re-read both the PR and the active timeline event immediately before
            # the audit write. A successful earlier probe cannot authorize a raced
            # head/base/draft/label transition.
            live_pr = managed_ci_api._api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}", quiet=True)
            live_head = live_pr.get("head") if isinstance(live_pr.get("head"), dict) else {}
            live_base = live_pr.get("base") if isinstance(live_pr.get("base"), dict) else {}
            live_author = live_pr.get("user") if isinstance(live_pr.get("user"), dict) else {}
            live_event = managed_ci_api._active_managed_label_event(runner, config=config, pr_number=pr_number)
            live_labels = _strict_label_names(live_pr)
            live_state = live_pr.get("state")
            live_repo = _str_field(live_pr, "head", "repo", "full_name")
            if (
                not (live_state is None or (isinstance(live_state, str) and live_state in {"open", "OPEN"}))
                or live_pr.get("draft") is not True
                or live_base.get("ref") != base_ref
                or live_head.get("sha") != live_sha
                or live_head.get("ref") != head_ref
                or live_repo is None
                or live_repo.casefold() != config.repo.casefold()
                or live_author.get("login") != actor_login
                or _strict_int(live_author.get("id")) != actor_id
                or live_labels is None
                or MANAGED_LABEL not in live_labels
                or live_pr.get("body") != pr.get("body")
                or live_event != active_event
            ):
                recovery = _release_for_ordinary_recovery(
                    runner, config=config, pr_number=pr_number, base_ref=base_ref,
                    report_context=context,
                    expected_head_sha=live_sha, active_event=live_event,
                    reason="the immutable resume tuple changed before activation",
                    recovery_capable=ordinary_recovery_capable,
                    fresh_issue_number=issue_hint,
                    fresh_authorization_allowed=origin == "issue-created",
                )
                return _ordinary_fallback_contract(runner, config=config, pr_number=pr_number, context=context, recovery=recovery)
            if protection.state != "strict":
                override_nonce = secrets.token_urlsafe(24)
                resume_label = protocol_record_label(
                    "managed_ci_resume_audit", pr_number=pr_number, head_sha=live_sha
                )
                resume_body = (
                    f"{resume_label}\n\n"
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
                try:
                    audit_id = post_verified_trusted_pr_protocol_comment(
                        runner,
                        config=config,
                        pr_number=pr_number,
                        body=trusted_resume_body,
                        expected_author_login=actor_login,
                        expected_author_id=actor_id,
                    )
                except AgentLoopError:
                    recovery = _release_for_ordinary_recovery(
                        runner, config=config, pr_number=pr_number, base_ref=base_ref,
                        report_context=context,
                        expected_head_sha=live_sha, active_event=active_event,
                        reason="the fresh resume audit could not be recorded and verified",
                        recovery_capable=ordinary_recovery_capable,
                        fresh_issue_number=issue_hint,
                        fresh_authorization_allowed=origin == "issue-created",
                    )
                    return _ordinary_fallback_contract(runner, config=config, pr_number=pr_number, context=context, recovery=recovery)
            else:
                audit_id = None
        workflow_revision = managed_ci_api._api_json(runner, config, f"repos/{config.repo}/commits/{dispatch_ref}", quiet=True)
        revision = workflow_revision.get("sha") if isinstance(workflow_revision.get("sha"), str) else None
        # Every v2 activation is generation-scoped.  ``auto_merge`` is an implicit
        # managed-CI request, so it must mint the same producer field as explicit
        # ``--managed-ci`` and resume paths.
        generation = secrets.token_urlsafe(16)
        log(
            config,
            f"PR #{pr_number}: selected managed resume (fresh generation)"
            if managed_resume is not None
            else f"PR #{pr_number}: activated authenticated managed exact-head CI v2",
        )
        return ManagedCiContract(
            protocol_version=2,
            base_ref=base_ref,
            dispatch_ref=dispatch_ref,
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
            visible_intent_capable=visible_intent_capable,
            host_footer_capable=host_footer_capable,
            origin=origin,
            lifecycle=lifecycle,
            authenticated_resume=managed_resume,
        )


def _live_label_presence(
    runner: Runner, *, config: AgentLoopConfig, pr_number: int, label_name: str
) -> bool | None:
    """Whether the live PR carries ``label_name``; ``None`` when unreadable."""
    try:
        payload = _read_pr_payload(
            runner, config=config, pr_number=pr_number, cwd=active_workdir(config)
        )
    except Exception:
        return None
    names = _label_names_or_none(payload)
    return None if names is None else label_name in names


def failure_suffix(result: object) -> str:
    """The final diagnostic plus attempt history, as a message suffix (may be empty)."""
    detail = describe_gh_failure(result)  # type: ignore[arg-type]
    return f"\n{detail}" if detail else ""


def reconciled_label_write(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    label_name: str,
    op: str,
    write: Callable[[], CommandResult],
    owned_event_id: int | None = None,
) -> ReconciledLabelWrite:
    """Add (``op='add'``) or remove (``op='remove'``) one label with reconciliation (#510).

    Before the first attempt the exact label's latest event is captured: for an
    add, the baseline a newer event is judged against; for a remove, the
    application being removed (``owned_event_id`` when the caller holds it).
    Without a readable baseline the write is a single attempt.  After a
    transient failure the exact label's event history is re-read (every label
    transition must be well formed) and, before any replay, the live label
    presence must agree with it.

    Add: a newer ``labeled`` event by the authenticated actor id is success; a
    newer foreign or ``unlabeled`` event fails closed; no newer event with the
    label absent replays.  Remove: a newer ``unlabeled`` by the actor is success;
    the label still present as the same application replays; a replacement
    ``labeled`` event (same actor or foreign), a foreign unlabel, or an
    unreadable or inconsistent history is never replayed against.  Failed
    outcomes carry the final diagnostic and attempt history.  Permanent failures
    pass through unchanged; dry-run bypasses all of it.
    """
    if config.dry_run:
        return ReconciledLabelWrite(write())
    history_log: list[GitHubAttempt] = []

    def failed(result: CommandResult, outcome: str) -> ReconciledLabelWrite:
        return ReconciledLabelWrite(
            github_retry._with_history(result, history_log, exhausted=outcome == "exhausted"),
            outcome=outcome,
        )

    target_id: int | None = owned_event_id
    baseline: CompleteLabelHistory | UnknownLabelHistory | None = None
    if op == "add" or target_id is None:
        baseline = managed_ci_api.label_event_history(
            runner, config=config, pr_number=pr_number, label_name=label_name
        )
    if op == "remove" and target_id is None and isinstance(baseline, CompleteLabelHistory):
        if baseline.latest is not None and baseline.latest.kind == "labeled":
            target_id = baseline.latest.event_id
    reconcilable = (
        isinstance(baseline, CompleteLabelHistory) if op == "add" else target_id is not None
    )
    active = DEFAULT_POLICY
    result: CommandResult | None = None
    for number in range(1, active.attempts + 1):
        started = time.monotonic()
        result = write()
        if result.returncode == 0 or classify_gh_failure(result) == "permanent":
            return ReconciledLabelWrite(result)
        history_log.append(attempt_from_result(number, result, started))
        if not reconcilable:
            log(config, f"PR #{pr_number}: `{label_name}` {op} is ambiguous and its ownership baseline is unavailable; not replayed")
            return failed(result, "unknown")
        history = managed_ci_api.label_event_history(
            runner, config=config, pr_number=pr_number, label_name=label_name
        )
        if not isinstance(history, CompleteLabelHistory):
            log(config, f"PR #{pr_number}: `{label_name}` {op} is ambiguous and its history is unreadable; not replayed")
            return failed(result, "unknown")
        try:
            actor_id = resolve_authenticated_github_actor(runner, config=config)[1]
        except Exception:
            return failed(result, "unknown")
        latest = history.latest
        if op == "add":
            assert isinstance(baseline, CompleteLabelHistory)
            before = baseline.latest.event_id if baseline.latest is not None else 0
            if latest is not None and latest.event_id > before:
                if latest.kind == "labeled" and latest.actor_id == actor_id:
                    log(config, f"PR #{pr_number}: `{label_name}` was applied despite a transient error (event {latest.event_id})")
                    return ReconciledLabelWrite(
                        github_retry.as_success(result), latest.event_id, "adopted"
                    )
                return failed(result, "foreign")
            if latest is not None and latest.kind == "labeled":
                # Present with no newer event: not the state this add expects.
                return failed(result, "unknown")
            if _live_label_presence(
                runner, config=config, pr_number=pr_number, label_name=label_name
            ) is not False:
                # History says absent but the live label is present or unreadable.
                return failed(result, "unknown")
        else:
            assert target_id is not None
            if latest is None:
                return failed(result, "unknown")
            if latest.kind == "unlabeled":
                if latest.event_id > target_id and latest.actor_id == actor_id:
                    log(config, f"PR #{pr_number}: `{label_name}` was removed despite a transient error")
                    return ReconciledLabelWrite(github_retry.as_success(result), outcome="adopted")
                return failed(result, "foreign")
            if latest.event_id != target_id:
                log(config, f"PR #{pr_number}: `{label_name}` was re-applied by a different event; not removed")
                return failed(result, "replacement")
            if _live_label_presence(
                runner, config=config, pr_number=pr_number, label_name=label_name
            ) is not True:
                return failed(result, "unknown")
        if number < active.attempts:
            github_retry.sleep_before_retry(active, number)
    assert result is not None
    return failed(result, "exhausted")


def _label_add(
    runner: Runner, *, config: AgentLoopConfig, pr_number: int, label_name: str
) -> ReconciledLabelWrite:
    command = [
        config.gh_cmd, "api", "--method", "POST",
        f"repos/{config.repo}/issues/{pr_number}/labels", "-f", f"labels[]={label_name}",
    ]
    return reconciled_label_write(
        runner, config=config, pr_number=pr_number, label_name=label_name, op="add",
        write=lambda: runner.run(command, cwd=active_workdir(config), check=False),
    )


def _label_remove(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    label_name: str,
    owned_event_id: int | None = None,
    cwd: Path | None = None,
) -> ReconciledLabelWrite:
    command = [
        config.gh_cmd, "api", "--method", "DELETE",
        f"repos/{config.repo}/issues/{pr_number}/labels/{label_name}",
    ]
    return reconciled_label_write(
        runner, config=config, pr_number=pr_number, label_name=label_name, op="remove",
        owned_event_id=owned_event_id,
        write=lambda: runner.run(command, cwd=cwd or active_workdir(config), check=False),
    )


def _has_exact_head_protection(runner: Runner, *, config: AgentLoopConfig, base_ref: str) -> bool:
    """Compatibility predicate for adoption; strict means non-bypassable."""
    return managed_ci_protection.assess_exact_head_protection(
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
    if created.returncode == 0:
        return True
    if classify_gh_failure(created) != "transient":
        return False
    # A transient failure may have created the label: re-read its definition.
    reread = runner.run(
        [config.gh_cmd, "api", f"repos/{config.repo}/labels/{MANAGED_LABEL}"],
        cwd=active_workdir(config), check=False,
    )
    return reread.returncode == 0


def _adoption_identity(
    runner: Runner, *, config: AgentLoopConfig
) -> tuple[str, int] | None:
    configured = (config.managed_ci_trusted_actor or "").strip()
    if not configured:
        return None
    who = managed_ci_api._api_json(runner, config, "user", quiet=True)
    login, actor_id = who.get("login"), who.get("id")
    advertised = managed_ci_api._advertised_managed_actor(runner, config)
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
    diagnostics: list[str] | None = None,
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
    dispatch_ref = _workflow_source_ref(runner, config, base_ref)
    workflow = runner.run(
        [config.gh_cmd, "api", f"repos/{config.repo}/contents/.github/workflows/{WORKFLOW_FILE}?ref={dispatch_ref}",
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
    if trusted_base_problem(
        runner, gh_cmd=config.gh_cmd, repo=config.repo, cwd=github_api_cwd(), base=base_ref,
        default_branch=dispatch_ref, workflow_text=source,
    ) is not None:
        return None
    pr = managed_ci_api._api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}", quiet=True)
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
    existing = managed_ci_api._active_managed_label_event(runner, config=config, pr_number=pr_number)
    applied = False
    recovered_event_id: int | None = None
    if MANAGED_LABEL not in labels:
        if not ensure_managed_label(runner, config=config):
            return None
        label_outcome = _label_add(
            runner, config=config, pr_number=pr_number, label_name=MANAGED_LABEL
        )
        created = label_outcome.result
        if created.returncode != 0:
            if diagnostics is not None:
                diagnostics.append(
                    f"\nThe `{MANAGED_LABEL}` label write failed ({label_outcome.outcome})."
                    + failure_suffix(created)
                )
            return None
        applied = True
        recovered_event_id = label_outcome.event_id
        existing = managed_ci_api._active_managed_label_event(runner, config=config, pr_number=pr_number)
    replaced = (
        recovered_event_id is not None
        and (existing is None or existing[0] != recovered_event_id)
    )
    if replaced or existing is None or existing[1].casefold() != actor_login.casefold() or existing[2] != actor_id:
        # A newly-created but unprovable label must not remain as our claimed
        # suppression.  The release helper fresh-checks event ownership, so a
        # replacement application (not the recovered one) is left untouched.
        provisional = ManagedCiContract(
            protocol_version=2, adopted_existing_pr=True,
            active_label_event_id=(
                recovered_event_id if recovered_event_id is not None
                else existing[0] if existing else None
            ),
            invocation_applied_label=applied,
        )
        if not release_adopted_managed_ci(
            runner, config=config, pr_number=pr_number, contract=provisional
        ) and diagnostics is not None:
            diagnostics.append(
                f"\nThe unprovable `{MANAGED_LABEL}` label could not be released."
                + provisional.release_diagnostic
            )
        return None
    revision = managed_ci_api._api_json(runner, config, f"repos/{config.repo}/commits/{dispatch_ref}", quiet=True).get("sha")
    log(config, f"PR #{pr_number}: activated authenticated managed exact-head CI v2 adoption")
    return ManagedCiContract(
        protocol_version=2, base_ref=base_ref, dispatch_ref=dispatch_ref, trusted_actor_login=actor_login,
        trusted_actor_id=actor_id, workflow_revision=revision if isinstance(revision, str) else None,
        adopted_existing_pr=True, guard_head_sha=live_sha, active_label_event_id=existing[0],
        invocation_applied_label=applied,
        visible_intent_capable=VISIBLE_INTENT_MARKER in source,
        host_footer_capable=HOST_FOOTER_INTENT_MARKER in source,
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
    pr = managed_ci_api._api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}", quiet=True)
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
    event = managed_ci_api._active_managed_label_event(runner, config=config, pr_number=pr_number)
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


@managed_ci_api._surfaces_list_failures
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
        event = managed_ci_api._active_managed_label_event(runner, config=config, pr_number=pr_number)
        if event is None or event[0] != contract.active_label_event_id:
            return True
    result = _label_remove(
        runner, config=config, pr_number=pr_number, label_name=MANAGED_LABEL,
        owned_event_id=contract.active_label_event_id,
    ).result
    # The bool API is unchanged; the final write diagnostic and attempt history
    # ride on the contract so the public cleanup refusal can report them.
    contract.release_diagnostic = "" if result.returncode == 0 else failure_suffix(result)
    return result.returncode == 0


def _label_names_or_none(payload: object) -> set[str] | None:
    """Return label names only for a well-formed PR payload, else None.

    Only a JSON object whose ``labels`` is a list of objects with string
    ``name`` fields proves label state; anything else is unreadable.
    """
    if not isinstance(payload, dict):
        return None
    labels = payload.get("labels")
    if not isinstance(labels, list):
        return None
    names: set[str] = set()
    for item in labels:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            return None
        names.add(item["name"])
    return names


def _read_pr_payload(
    runner: Runner, *, config: AgentLoopConfig, pr_number: int, cwd: Path
) -> dict[str, object] | None:
    """Read the live PR in ``cwd``; None for any failed or malformed read."""
    result = runner.run(
        [config.gh_cmd, "api", f"repos/{config.repo}/pulls/{pr_number}"],
        cwd=cwd, check=False,
    )
    if result.returncode != 0 or not (result.stdout or "").strip():
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _delete_managed_label(
    runner: Runner, *, config: AgentLoopConfig, pr_number: int, cwd: Path
) -> bool:
    """DELETE the managed label; success or a strict HTTP 404 means it is absent.

    Only gh's own unambiguous status diagnostic proves a 404, so incidental
    or conflicting ``HTTP 404`` text never counts as absence.
    """
    result = _label_remove(
        runner, config=config, pr_number=pr_number, label_name=MANAGED_LABEL, cwd=cwd,
    ).result
    if result.returncode == 0:
        return True
    return managed_ci_api._http_status(result) == 404


def _label_event_owned_by_contract(
    event: tuple[int, str, int] | None, contract: ManagedCiContract
) -> bool:
    return bool(
        event is not None
        and contract.active_label_event_id is not None
        and event[0] == contract.active_label_event_id
        and event[1].casefold() == (contract.trusted_actor_login or "").casefold()
        and event[2] == contract.trusted_actor_id
    )


@managed_ci_api._surfaces_list_failures
def _verify_manual_qualification_label_provenance(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    contract: ManagedCiContract,
) -> None:
    """Require the contract's authenticated label event to still be active.

    This never removes the label: a successful manual qualification retains it
    so no ``unlabeled`` event re-runs ordinary CI on the qualified head.
    """
    event = managed_ci_api._active_managed_label_event(runner, config=config, pr_number=pr_number)
    if not _label_event_owned_by_contract(event, contract):
        raise AgentLoopError(
            f"PR #{pr_number} managed-label provenance changed before manual qualification; "
            "the head is not qualified and no manual merge command is safe."
        )


def _release_label_after_failed_publication(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    contract: ManagedCiContract,
) -> str | None:
    """Return a failed publication to ordinary CI; never raises.

    A confirmed-absent label needs no write.  A contract-owned label, a label
    whose provenance is unreadable (fail-open, as in
    ``release_adopted_managed_ci``) and a label on an unreadable PR are
    removed.  A readable foreign label event is left in place and reported.
    Returns a message fragment when the label may still suppress ordinary CI.
    """
    manual = (
        f"`{MANAGED_LABEL}` could not be removed and still suppresses ordinary CI; "
        "remove it manually"
    )
    try:
        cwd = active_workdir(config)
        try:
            payload = _read_pr_payload(runner, config=config, pr_number=pr_number, cwd=cwd)
        except Exception:
            payload = None
        labels = _label_names_or_none(payload)
        if labels is not None:
            if MANAGED_LABEL not in labels:
                return None
            try:
                event = managed_ci_api._active_managed_label_event(runner, config=config, pr_number=pr_number)
            except Exception:
                event = None
            if event is None:
                log(
                    config,
                    f"PR #{pr_number}: managed-label provenance is unreadable after a failed "
                    f"qualification publication; removing `{MANAGED_LABEL}` fail-open",
                )
            elif not _label_event_owned_by_contract(event, contract):
                return (
                    f"a different `{MANAGED_LABEL}` label event is active and was left untouched; "
                    "ordinary CI may still be suppressed; remove it manually if it is stale"
                )
        try:
            removed = _delete_managed_label(runner, config=config, pr_number=pr_number, cwd=cwd)
        except Exception:
            removed = False
        if removed:
            log(
                config,
                f"PR #{pr_number}: released `{MANAGED_LABEL}` after a failed qualification "
                "publication; ordinary CI resumes",
            )
            return None
        return manual
    except Exception:
        return manual


def _attach_cleanup_fragment(original: BaseException, fragment: str) -> None:
    """Attach a cleanup fragment without changing the original error type."""
    sentence = f"{fragment[0].upper()}{fragment[1:]}."
    if isinstance(original, AgentLoopError) and original.args and isinstance(original.args[0], str):
        original.args = (f"{original.args[0]} {sentence}", *original.args[1:])
    else:
        original.add_note(sentence)


@managed_ci_api._surfaces_list_failures
def publish_manual_v2_qualification(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    expected_head_sha: str,
    contract: ManagedCiContract,
    reviewers: tuple[str, ...],
    verified_result: VerifiedManagedResult | None = None,
) -> str:
    """Publish a SHA-bound manual result on a ready PR that keeps its label.

    The managed label is retained on success, so no ``unlabeled`` event
    re-runs ordinary CI on the qualified head; it has no effect once the PR
    is ready.  Any failure releases an owned or unprovable label before the
    original error propagates, independent of the orchestrator's
    interrupted-run preservation.
    """
    try:
        return _publish_manual_v2_qualification(
            runner,
            config=config,
            pr_number=pr_number,
            expected_head_sha=expected_head_sha,
            contract=contract,
            reviewers=reviewers,
            verified_result=verified_result,
        )
    except BaseException as original:
        fragment = _release_label_after_failed_publication(
            runner, config=config, pr_number=pr_number, contract=contract,
        )
        if fragment:
            _attach_cleanup_fragment(original, fragment)
        raise original


def _publish_manual_v2_qualification(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    expected_head_sha: str,
    contract: ManagedCiContract,
    reviewers: tuple[str, ...],
    verified_result: VerifiedManagedResult | None = None,
) -> str:
    if contract.protocol_version != 2:
        raise AgentLoopError("Explicit managed-CI manual qualification requires protocol v2.")
    if verified_result is not None:
        managed_ci_reconciliation.require_verified_result(runner, config=config, pr_number=pr_number,
                                expected_head_sha=expected_head_sha, contract=contract, result=verified_result)
    if get_pr_head_sha(runner, config, pr_number) != expected_head_sha:
        raise AgentLoopError(
            f"PR #{pr_number} head changed before manual qualification publication; no merge is safe."
        )

    # The bare qualified label is intentionally not used for a manual result.
    # Remove a historical copy if one exists so it cannot outlive its SHA.
    pr = managed_ci_api._api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}")
    labels = {
        item.get("name") for item in (pr.get("labels") or [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if QUALIFIED_LABEL in labels:
        removed = _label_remove(
            runner, config=config, pr_number=pr_number, label_name=QUALIFIED_LABEL
        ).result
        if removed.returncode != 0:
            raise AgentLoopError(
                f"Unable to clear stale `{QUALIFIED_LABEL}` from PR #{pr_number}.{failure_suffix(removed)}"
            )

    not_published = (
        f"PR #{pr_number} changed before manual qualification publication; "
        "its exact head is not safely published."
    )
    guard = managed_ci_api._api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}")
    guard_head = guard.get("head") if isinstance(guard.get("head"), dict) else {}
    if (
        guard_head.get("sha") != expected_head_sha
        # An adopted PR is never readied here, so it must already be exactly
        # non-draft; a missing or non-boolean value is not proof.
        or (contract.adopted_existing_pr and guard.get("draft") is not False)
    ):
        raise AgentLoopError(not_published)
    require_recorded_base(
        runner, config=config, pr_number=pr_number, base_ref=contract.base_ref,
        repository=contract.repository or config.repo, head_sha=expected_head_sha,
        action="manual qualification",
    )
    _verify_manual_qualification_label_provenance(
        runner, config=config, pr_number=pr_number, contract=contract,
    )

    changed = not_published
    if not contract.adopted_existing_pr:
        ready = reconciled_pr_ready(
            runner, config=config, pr_number=pr_number, expected_head_sha=expected_head_sha,
        )
        if ready.returncode != 0:
            raise AgentLoopError(
                f"Unable to mark qualified PR #{pr_number} ready for manual review.{failure_suffix(ready)}"
            )
        changed = (
            f"PR #{pr_number} changed while being made ready; the approved head is not safely published."
        )
    after = managed_ci_api._api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}")
    after_head = after.get("head") if isinstance(after.get("head"), dict) else {}
    after_labels = _label_names_or_none(after)
    if (
        after_head.get("sha") != expected_head_sha
        or after.get("draft") is not False
        or after_labels is None
        or MANAGED_LABEL not in after_labels
    ):
        raise AgentLoopError(changed)
    _verify_manual_qualification_label_provenance(
        runner, config=config, pr_number=pr_number, contract=contract,
    )

    require_recorded_base(
        runner, config=config, pr_number=pr_number, base_ref=contract.base_ref,
        repository=contract.repository or config.repo, head_sha=expected_head_sha,
        action="manual qualification record",
    )
    run_text = str(contract.attached_run_id) if contract.attached_run_id is not None else "unknown"
    attempt_text = str(contract.run_attempt) if contract.run_attempt is not None else "unknown"
    result_fields = (managed_ci_reconciliation.historical_qualification_fields(verified_result) if verified_result else
                     f"nonce={contract.nonce or 'unknown'} run_id={run_text} attempt={attempt_text} "
                     f"generation={contract.intent_generation or 'unknown'}")
    reviewer_text = ",".join(reviewers) or "unknown"
    qualification_label = protocol_record_label(
        "managed_ci_qualified_head", pr_number=pr_number, head_sha=expected_head_sha
    )
    body = (
        f"{qualification_label}\n\n"
        f"<!-- {QUALIFICATION_MARKER} repo={config.repo} pr={pr_number} base={contract.base_ref or config.base} "
        f"protocol=2 qualified_head={expected_head_sha} reviewers={reviewer_text} "
        f"protection={contract.protection_mode or 'unknown'} {result_fields} -->\n\n"
        f"Managed exact-head CI qualified `{expected_head_sha}` for manual merge. "
        "The managed label is retained so ordinary CI does not re-run on this qualified head; "
        "it has no effect while the PR is ready. Removing the label, or re-running agent-loop "
        "on this PR, returns it to ordinary CI."
    )
    if contract.protection_mode != "strict":
        body += (
            " GitHub cannot force a human or other automation to use this SHA or the guarded command "
            "after agent-loop exits."
        )
    trusted_body = TrustedBody.canonical(body, expected_tokens=(QUALIFICATION_MARKER,))
    # The qualified-head record has no parser or recovery consumer; its canonical
    # marker span is preserved and the write is verified before it is adopted.
    publication_id = post_verified_trusted_pr_protocol_comment(
        runner,
        config=config,
        pr_number=pr_number,
        body=trusted_body,
        expected_author_login=contract.trusted_actor_login,
        expected_author_id=contract.trusted_actor_id,
    )
    if get_pr_head_sha(runner, config, pr_number) != expected_head_sha:
        raise AgentLoopError(
            f"PR #{pr_number} head changed after qualification publication; rerun review and exact-head CI."
        )
    if verified_result is None:
        contract.audit_comment_id = publication_id
    return expected_head_sha


def retained_managed_label_present(
    runner: Runner, *, config: AgentLoopConfig, pr_number: int, cwd: Path,
) -> bool:
    """Read-only: would ``release_retained_managed_label`` remove the label?"""
    pr = _read_pr_payload(runner, config=config, pr_number=pr_number, cwd=cwd)
    labels = _label_names_or_none(pr)
    return not (
        pr is None
        or labels is None
        or pr.get("state") != "open"
        or pr.get("draft") is not False
        or MANAGED_LABEL not in labels
    )


def release_retained_managed_label(
    runner: Runner, *, config: AgentLoopConfig, pr_number: int, cwd: Path,
) -> bool:
    """Release a label retained on an open ready PR at PR-loop entry.

    A successful manual qualification keeps ``agent-loop-managed`` on the
    ready PR.  Every later invocation first returns such a PR to the
    ready/unlabeled state the lifecycle already handles.  Removing a label
    from a ready PR can only restore ordinary CI, so no provenance check is
    needed.  Drafts, closed PRs, unlabeled PRs and malformed reads are left
    untouched (downstream reads fail on malformed state as before).  Every
    command runs in ``cwd`` because the coder checkout may not exist yet.
    Returns True only when the label was removed and its absence confirmed.
    """
    if not retained_managed_label_present(runner, config=config, pr_number=pr_number, cwd=cwd):
        return False
    failure = AgentLoopError(
        f"PR #{pr_number} is ready and still carries `{MANAGED_LABEL}`; it could not be removed, "
        "so no managed-CI or review work was started. Remove the label manually, then rerun."
    )
    # A runner exception (transport or subprocess failure) at either step is
    # reported with the same manual remedy; the read-back may fail after the
    # DELETE already succeeded, so absence is never assumed.
    try:
        removed = _delete_managed_label(runner, config=config, pr_number=pr_number, cwd=cwd)
    except Exception as exc:
        raise failure from exc
    if not removed:
        raise failure
    try:
        after = _label_names_or_none(
            _read_pr_payload(runner, config=config, pr_number=pr_number, cwd=cwd)
        )
    except Exception as exc:
        raise failure from exc
    if after is None or MANAGED_LABEL in after:
        raise failure
    log(
        config,
        f"PR #{pr_number}: removed `{MANAGED_LABEL}` from the ready PR at entry "
        "(label origin was not checked); "
        "ordinary CI resumes and the run continues as ready/unlabeled",
    )
    return True


def _workflow_source_ref(runner: Runner, config: AgentLoopConfig, requested_base: str | None) -> str:
    """The branch the trusted managed workflow is read from and dispatched on.

    Always the live default branch, never the PR's integration base, so a
    branch cannot supply the workflow that qualifies it (#1285).  Fail-closed:
    when the default branch cannot be read this raises ``AgentLoopError`` and
    never substitutes ``requested_base`` (or any other branch), which is only
    named in the error message.
    """
    default_branch = managed_ci_api._read_repo_default_branch(runner, config.gh_cmd, config.repo, github_api_cwd())
    if default_branch is None:
        raise AgentLoopError(
            f"The default branch of {config.repo} could not be read, so the trusted managed-CI "
            f"workflow source cannot be established (base {requested_base!r} is never used as a "
            "substitute). Retry once repository metadata is readable."
        )
    return default_branch


def _contract_dispatch_ref(runner: Runner, config: AgentLoopConfig, contract: ManagedCiContract) -> str:
    """Resolve the dispatch ref, failing closed for a legacy non-default contract."""
    if contract.dispatch_ref:
        return contract.dispatch_ref
    default_branch = managed_ci_api._read_repo_default_branch(runner, config.gh_cmd, config.repo, github_api_cwd())
    base = contract.base_ref
    if default_branch is None or base is None or base != default_branch:
        raise AgentLoopError(
            "The managed-CI contract carries no default-branch dispatch ref and its base "
            f"{base!r} is not the default branch; the workflow is never dispatched from a "
            "non-default ref. Rerun agent-loop on the PR so a fresh contract is authorized."
        )
    contract.dispatch_ref = default_branch
    return default_branch


def trusted_base_problem(
    runner: Runner,
    *,
    gh_cmd: str,
    repo: str,
    cwd: Path,
    base: str | None,
    default_branch: str | None,
    workflow_text: str | None = None,
    require_marker: bool = True,
) -> str | None:
    """Why ``base`` cannot be a managed-CI target, or ``None`` when it can.

    The default branch is always allowed.  Any other base needs the default
    branch workflow to advertise trusted-base support and the repository
    variable to allow-list it; an unreadable variable never grants trust.
    """
    if not base:
        return None
    if not default_branch:
        return (
            f"base {base!r} cannot be verified: the repository default branch could not be read; "
            "trust is never assumed on a read failure"
        )
    if base == default_branch:
        return None
    if require_marker and (workflow_text is None or TRUSTED_BASES_MARKER not in workflow_text):
        return (
            f"base {base!r} is not the default branch {default_branch!r} and the default-branch "
            f"workflow does not advertise {TRUSTED_BASES_MARKER}; managed CI into an integration "
            f"branch needs the reusable managed-CI workflow with {TRUSTED_BASES_VARIABLE} support"
        )
    variable = managed_ci_api._read_actions_variable(runner, gh_cmd, repo, cwd, TRUSTED_BASES_VARIABLE)
    if variable.status in {"unreadable", "error"}:
        return (
            f"base {base!r} cannot be verified: the {TRUSTED_BASES_VARIABLE} repository variable "
            f"could not be read ({variable.detail}); trust is never assumed on a read failure"
        )
    value = variable.value if variable.status == "readable" and variable.value else ""
    if not base_is_trusted(base, default_branch, value):
        listed = f"currently {value!r}" if value.strip() else "currently unset"
        return (
            f"base {base!r} is not the default branch {default_branch!r} and is not trusted by "
            f"the {TRUSTED_BASES_VARIABLE} repository variable ({listed}; an unset or invalid "
            "variable trusts the default branch only)"
        )
    return None


def enforce_trusted_base_at_startup(runner: Runner, config: AgentLoopConfig) -> None:
    """Refuse an untrusted integration ``--base`` before any agent work (#1285).

    Default-branch runs and runs without managed CI are never affected.
    """
    if not config.effective_managed_ci or config.dry_run or not config.base:
        return
    cwd = github_api_cwd()
    default_branch = managed_ci_api._read_repo_default_branch(runner, config.gh_cmd, config.repo, cwd)
    if default_branch is None:
        raise AgentLoopError(
            f"Managed CI cannot verify --base {config.base!r}: the default branch of "
            f"{config.repo} could not be read, so {TRUSTED_BASES_VARIABLE} trust is never assumed. "
            "No review cycle was spent; retry once repository metadata is readable."
        )
    if default_branch == config.base:
        return
    workflow, _ = managed_ci_api._probe_raw_workflow(
        runner, ManagedCiProbeContext(config.repo, config.gh_cmd, cwd), default_branch
    )
    problem = trusted_base_problem(
        runner, gh_cmd=config.gh_cmd, repo=config.repo, cwd=cwd, base=config.base,
        default_branch=default_branch, workflow_text=workflow,
    )
    if problem:
        raise AgentLoopError(
            f"Managed CI refuses --base {config.base!r}: {problem}. No review cycle was spent. "
            f"Ask the repository owner to add the base to the {TRUSTED_BASES_VARIABLE} repository "
            "variable (exact names or `prefix/*`), or use the default branch."
        )


def require_recorded_base(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    base_ref: str | None,
    repository: str | None,
    head_sha: str | None,
    action: str,
) -> None:
    """Re-read the live PR tuple and refuse when the recorded base drifted.

    Qualification is bound to the repository, base and head it was granted
    for.  On a retarget, a revoked allow-list entry or a moved head the
    qualified label is released and the caller must re-qualify.  GitHub's
    merge API guards only the head SHA, so a retarget between this read and
    the merge call is narrowed but cannot be excluded atomically.
    """
    if not base_ref:
        return
    pr = managed_ci_api._api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}", quiet=True)
    base = pr.get("base") if isinstance(pr.get("base"), dict) else {}
    base_repo = base.get("repo") if isinstance(base.get("repo"), dict) else {}
    head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
    problem: str | None = None
    if not pr:
        problem = "the live pull request could not be read"
    elif base.get("ref") != base_ref:
        problem = f"the live base {base.get('ref')!r} differs from the qualified base {base_ref!r}"
    elif (
        isinstance(base_repo.get("full_name"), str)
        and base_repo["full_name"].casefold() != (repository or config.repo).casefold()
    ):
        problem = "the live base repository differs from the qualified repository"
    elif head_sha is not None and head.get("sha") != head_sha:
        problem = "the live head differs from the qualified head"
    else:
        default_branch = managed_ci_api._read_repo_default_branch(
            runner, config.gh_cmd, config.repo, github_api_cwd()
        )
        if default_branch is None:
            problem = "the repository default branch could not be read"
        else:
            problem = trusted_base_problem(
                runner, gh_cmd=config.gh_cmd, repo=config.repo, cwd=github_api_cwd(),
                base=base_ref, default_branch=default_branch, require_marker=False,
            )
    if problem is None:
        return
    released = _label_remove(
        runner, config=config, pr_number=pr_number, label_name=QUALIFIED_LABEL
    ).result
    suffix = (
        "" if released.returncode == 0
        else f" The `{QUALIFIED_LABEL}` label could not be released.{failure_suffix(released)}"
    )
    raise AgentLoopError(
        f"PR #{pr_number} {action} refused: {problem}. Qualification is invalidated; "
        f"re-qualify the exact head before merging.{suffix}"
    )


def managed_label_present(runner: Runner, *, config: AgentLoopConfig, pr_number: int) -> bool | None:
    """Return None when label state cannot be proven, never treating it as safe."""
    pr = managed_ci_api._api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}", quiet=True)
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
) -> bool:
    """Publish the informational ``agent-loop/round-readiness`` status.

    The status is non-required: nothing gates on it (the authoritative gate is
    the workflow-published exact-head status), so a refused write is logged and
    the round continues.  Some hosts refuse every commit-status write, e.g. a
    Claude Code cloud session's proxy (#1052).  Returns whether it was published.
    """
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
        detail = ((result.stderr or "").strip().splitlines() or [f"exit {result.returncode}"])[-1]
        log(
            config,
            f"Warning: could not publish the non-required `{READINESS_CONTEXT}` status for "
            f"{head_sha} ({detail}); continuing, since nothing gates on it.",
        )
        return False
    return True


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
    requested_at = datetime.now(timezone.utc)
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
    record = ManagedV1DispatchRecord(
        workflow_file=contract.workflow_file,
        ref=head_ref,
        pr_number=pr_number,
        expected_head_sha=expected_head_sha,
        requested_at=requested_at,
        returned_at=datetime.now(timezone.utc),
    )
    contract.v1_dispatch_records = (
        *contract.v1_dispatch_records, record
    )[-V1_DISPATCH_RECORD_LIMIT:]
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
    dispatch_ref = _contract_dispatch_ref(runner, config, contract)
    current_base = managed_ci_api._api_json(
        runner, config, f"repos/{config.repo}/commits/{dispatch_ref}", quiet=True
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
    except (ManagedCiHostFooterIncompatibleError, ManagedCiIntentLedgerError):
        # Terminal operator-action errors: ordinary recovery would release
        # suppression around a ledger the workflow rejects or cannot see.
        raise
    except AgentLoopError as exc:
        if not config.managed_ci_pr_mode and contract.authenticated_resume is None:
            raise
        event = managed_ci_api._active_managed_label_event(runner, config=config, pr_number=pr_number)
        recovery = _release_for_ordinary_recovery(
            runner, config=config, pr_number=pr_number, base_ref=contract.base_ref,
            expected_head_sha=expected_head_sha, active_event=event,
            reason=f"fresh intent ledger could not be reconciled ({exc})",
            recovery_capable=contract.ordinary_recovery_capable,
            fresh_issue_number=(
                contract.authenticated_resume.issue_created_handoff.issue_number
                if (
                    contract.authenticated_resume is not None
                    and contract.authenticated_resume.origin == "issue-created"
                    and contract.authenticated_resume.issue_created_handoff is not None
                )
                else None
            ),
            fresh_authorization_allowed=(
                contract.origin == "issue-created"
                and contract.authenticated_resume is not None
                and contract.authenticated_resume.origin == "issue-created"
            ),
        )
        contract.activation_path = "ordinary_fallback"
        contract.ordinary_recovery = recovery
        log(config, f"PR #{pr_number}: managed resume ledger failed; selected ordinary recovery")
        return

    # A same-nonce attachment is the durable owner of this dispatch.  A
    # transiently empty workflow-runs response must never turn it into a
    # second dispatch or an early record with stale run fields.  Only an
    # attempt explicitly excluded by the terminal-attempt history may be
    # cleared before discovering a replacement.
    if contract.attached_run_id is not None:
        if _v2_attachment_is_excluded(contract):
            contract.attached_run_id = None
            contract.run_attempt = None
            _patch_intent(runner, config=config, contract=contract, state="dispatch-requested")
        else:
            log(
                config,
                f"PR #{pr_number}: retaining managed-CI v2 attachment "
                f"{contract.attached_run_id}/{contract.run_attempt}; waiter owns refresh",
            )
            return
    elif contract.intent_state == "completed" and contract.terminal_outcome == "no-status":
        # The restored terminal attempt was excluded.  Publish the legal
        # early lifecycle transition before looking for a higher attempt or a
        # fresh same-nonce run.
        _patch_intent(runner, config=config, contract=contract, state="dispatch-requested")
    attached = _discover_v2_run(runner, config=config, contract=contract, retries=3)
    if attached is None:
        # This PATCH directly precedes a dispatch, so it renews created_at in
        # place: the workflow's dispatch validator bounds the record's age,
        # and a recovered record may be stale or future-dated.  Same comment,
        # nonce, and generation; no second intent is posted.
        contract.created_at = int(time.time())
        patch_result = _patch_intent(
            runner, config=config, contract=contract, state="dispatch-requested"
        )
        _require_workflow_dispatch_authorization(
            runner, config=config, contract=contract, patch_result=patch_result
        )
        contract.dispatch_issued = True
        result = runner.run(
            [
                config.gh_cmd,
                "api",
                "--method",
                "POST",
                f"repos/{config.repo}/actions/workflows/{contract.workflow_file}/dispatches",
                "-f",
                f"ref={_contract_dispatch_ref(runner, config, contract)}",
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
        contract.dispatch_issued = True
        contract.attached_run_id = attached[0]
        contract.run_attempt = attached[1]
        contract.terminal_outcome = None
        if contract.run_attempt is not None:
            _patch_intent(runner, config=config, contract=contract, state="attached")
            log(config, f"PR #{pr_number}: attached managed-CI v2 run {attached[0]}")
        else:
            # Keep the in-memory correlation key, but do not publish an
            # attached record that the router would reject without an attempt.
            log(config, f"PR #{pr_number}: run {attached[0]} is visible without an attempt; waiting for refresh")
    else:
        # Do not guess a run ID. The waiter will continue bounded discovery;
        # an uncorrelated green status is never accepted in the meantime.
        log(config, f"PR #{pr_number}: dispatch accepted; waiting for run visibility")


def _intent_body(contract: ManagedCiContract, *, pr_number: int, expected_head_sha: str, state: str) -> TrustedBody:
    early_state = state in {"prepared", "dispatch-requested"}
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
        "run_id": None if early_state else contract.attached_run_id,
        "run_attempt": None if early_state else contract.run_attempt,
        "terminal_run_id": contract.terminal_run_id,
        "terminal_run_attempt": contract.terminal_run_attempt,
        "terminal_outcome": contract.terminal_outcome,
        "terminal_attempts": [
            {"run_id": run_id, "run_attempt": run_attempt}
            for run_id, run_attempt in contract.terminal_attempts
        ],
    }
    record = f"<!-- {INTENT_MARKER} {json.dumps(payload, separators=(',', ':'), sort_keys=True)} -->"
    # The intent record is the one protocol comment parsed by the installed
    # base workflow, which fullmatches the whole stripped body.  An older
    # workflow accepts only the bare record (#888), so the fixed visible line
    # is emitted only when the same base workflow advertises that its
    # envelope admits it (#935).  The line is a literal template bound to the
    # payload's head, never free text.
    if contract.visible_intent_capable:
        record = visible_intent_line(expected_head_sha) + "\n\n" + record
    return TrustedBody.canonical(record, expected_tokens=(INTENT_MARKER,))


def visible_intent_line(expected_head_sha: str) -> str:
    """Return the exact visible line a v2 workflow admits ahead of the record."""
    return f"Managed CI authorization for exact head {expected_head_sha}."


def _validate_v2_intent_publication(contract: ManagedCiContract, *, state: str) -> None:
    if state not in {"prepared", "dispatch-requested", "attached", "completed"}:
        raise AgentLoopError(f"Unsupported managed-CI v2 lifecycle state: {state}.")
    if state in {"prepared", "dispatch-requested"} and (
        contract.attached_run_id is not None or contract.run_attempt is not None
    ):
        raise AgentLoopError(
            f"Cannot publish {state} while a managed-CI v2 run attachment is retained."
        )
    if state in {"attached", "completed"} and (
        not isinstance(contract.attached_run_id, int)
        or contract.attached_run_id <= 0
        or not isinstance(contract.run_attempt, int)
        or contract.run_attempt <= 0
    ):
        raise AgentLoopError(
            f"Cannot publish {state} without a positive run_id/run_attempt pair."
        )


def _reset_v2_intent_generation(contract: ManagedCiContract) -> None:
    """Clear every nonce-scoped field before minting a fresh intent."""
    contract.intent_comment_id = None
    contract.nonce = None
    contract.created_at = None
    contract.attached_run_id = None
    contract.run_attempt = None
    contract.terminal_run_id = None
    contract.terminal_run_attempt = None
    contract.terminal_attempts = ()
    contract.terminal_outcome = None
    contract.intent_state = None
    contract.dispatch_issued = False


def _restore_v2_intent_fields(contract: ManagedCiContract, intent: dict[str, object]) -> None:
    """Restore optional fields with record-local defaults, never old-round state."""
    contract.intent_comment_id = None
    contract.nonce = None
    contract.created_at = None
    contract.attached_run_id = None
    contract.run_attempt = None
    contract.terminal_run_id = None
    contract.terminal_run_attempt = None
    contract.terminal_attempts = ()
    contract.terminal_outcome = None
    contract.intent_state = None

    nonce = intent.get("nonce")
    if not isinstance(nonce, str) or not nonce:
        raise AgentLoopError("Managed-CI v2 intent comment lacks a nonce.")
    if not re.fullmatch(INTENT_NONCE_PATTERN, nonce):
        # The workflow requires this exact token shape; never resume a nonce
        # its validator would fail.
        raise ManagedCiIntentLedgerError(
            "Managed-CI v2 intent comment carries a malformed nonce; repair or delete "
            "the intent comment and resume."
        )
    contract.nonce = nonce
    contract.created_at = intent.get("created_at") if isinstance(intent.get("created_at"), int) else None
    contract.attached_run_id = intent.get("run_id") if isinstance(intent.get("run_id"), int) else None
    contract.run_attempt = intent.get("run_attempt") if isinstance(intent.get("run_attempt"), int) else None
    contract.intent_state = intent.get("state") if isinstance(intent.get("state"), str) else None
    terminal_run_id = intent.get("terminal_run_id")
    terminal_run_attempt = intent.get("terminal_run_attempt")
    if isinstance(terminal_run_id, int):
        contract.terminal_run_id = terminal_run_id
        contract.terminal_run_attempt = (
            terminal_run_attempt if isinstance(terminal_run_attempt, int) else None
        )
    terminal_outcome = intent.get("terminal_outcome")
    if terminal_outcome == "no-status" or contract.intent_state == "terminal-no-status":
        contract.terminal_outcome = "no-status"
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
    # Legacy terminal-no-status records did not have a terminal-attempt list;
    # their active attachment is the excluded attempt, including an omitted
    # attempt number.
    if contract.terminal_outcome == "no-status" and not terminal_attempts:
        if contract.attached_run_id is not None:
            terminal_attempts.append((contract.attached_run_id, contract.run_attempt))
    legacy_key = (contract.terminal_run_id, contract.terminal_run_attempt)
    if legacy_key[0] is not None and legacy_key not in terminal_attempts:
        terminal_attempts.append((legacy_key[0], legacy_key[1]))
    contract.terminal_attempts = tuple(terminal_attempts)
    if _v2_attachment_is_excluded(contract):
        contract.attached_run_id = None
        contract.run_attempt = None
    elif contract.attached_run_id is not None and contract.terminal_outcome == "no-status":
        # A record that already carries a different, non-excluded attachment
        # represents an active retry. Its prior no-status result remains in
        # terminal_attempts, but is no longer the current outcome.
        contract.terminal_outcome = None


def _v2_excluded_attempts(contract: ManagedCiContract) -> tuple[tuple[int, int | None], ...]:
    exclusions = list(contract.terminal_attempts)
    if contract.terminal_run_id is not None:
        terminal_key = (contract.terminal_run_id, contract.terminal_run_attempt)
        if terminal_key not in exclusions:
            exclusions.append(terminal_key)
    return tuple(exclusions)


def _v2_attachment_is_excluded(contract: ManagedCiContract) -> bool:
    if contract.attached_run_id is None:
        return False
    return managed_ci_status._v2_terminal_attempt_excluded(
        contract.attached_run_id,
        contract.run_attempt,
        _v2_excluded_attempts(contract),
    )


def managed_ci_attachment_matches_head(contract: ManagedCiContract, head_sha: str | None) -> bool:
    """True only for an attached attempt dispatched for ``head_sha`` (unknown binding fails closed)."""
    return (
        contract.attached_run_id is not None
        and contract.run_attempt is not None
        and bool(head_sha)
        and contract.expected_head_sha == head_sha
    )


def release_stale_managed_ci_attachment(
    contract: ManagedCiContract,
    *,
    head_sha: str | None,
    config: AgentLoopConfig,
    pr_number: int,
) -> bool:
    """Drop in-memory intent state bound to another head so a fresh attempt is dispatched.

    The old head's intent comment is left untouched and its workflow run is
    not cancelled.
    """
    if contract.protocol_version != 2:
        return False
    if contract.attached_run_id is None and contract.nonce is None:
        return False
    if contract.expected_head_sha is not None and contract.expected_head_sha == head_sha:
        return False
    log(
        config,
        f"PR #{pr_number}: managed-CI attempt {contract.attached_run_id}/{contract.run_attempt} "
        f"was dispatched for {contract.expected_head_sha or 'an unknown head'}, not current head "
        f"{head_sha}; treating it as stale and dispatching a fresh attempt",
    )
    _reset_v2_intent_generation(contract)
    return True


def _intent_producer_login(contract: ManagedCiContract) -> str:
    """Return the authenticated producer the intent ledger must be written by."""
    if not contract.trusted_actor_login:
        raise AgentLoopError("Managed-CI v2 intent requires an authenticated trusted actor.")
    return contract.trusted_actor_login


def _intent_producer_id(contract: ManagedCiContract) -> int:
    if not isinstance(contract.trusted_actor_id, int) or contract.trusted_actor_id < 1:
        raise AgentLoopError("Managed-CI v2 intent requires an authenticated trusted actor ID.")
    return contract.trusted_actor_id


def _ensure_v2_intent(
    runner: Runner, *, config: AgentLoopConfig, pr_number: int, expected_head_sha: str, contract: ManagedCiContract,
) -> None:
    contract.pr_number = pr_number
    contract.expected_head_sha = expected_head_sha
    contract.repository = config.repo
    trusted_login = _intent_producer_login(contract)
    trusted_id = _intent_producer_id(contract)
    page = _read_intent_comment_page(runner, config, pr_number)
    # Every nonce-independent failure the workflow would raise for any nonce,
    # including a freshly minted one, stops here: nothing is adopted, posted,
    # or dispatched while such a comment exists.
    fatal = managed_ci_intent_validation.intent_page_pre_nonce_fatal(
        page,
        trusted_login=trusted_login,
        trusted_id=trusted_id,
        visible_capable=contract.visible_intent_capable,
        host_footer_capable=contract.host_footer_capable,
    )
    if fatal is not None:
        raise ManagedCiIntentLedgerError(
            f"PR #{pr_number}: the managed-CI v2 intent ledger holds a comment the base workflow "
            f"rejects ({fatal}); repair or delete that comment and resume."
        )
    # Candidates are trusted, envelope-matching, version-2 object payloads for
    # this PR head and generation, including malformed ones: the workflow's
    # verdict for that nonce decides whether one is adoptable.
    candidates: dict[str, object] = {}
    for raw in page:
        assert isinstance(raw, dict)  # guaranteed by the fatal scan above
        user = raw["user"]
        assert isinstance(user, dict)
        if user.get("login") != trusted_login:
            continue
        match = managed_ci_intent_validation.match_intent_envelope(
            raw.get("body"),
            visible_capable=contract.visible_intent_capable,
            host_footer_capable=contract.host_footer_capable,
        )
        if match is None:
            continue
        intent = json.loads(match.group("payload"))
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
        if intent.get("pr") != pr_number or intent.get("expected_head_sha") != expected_head_sha:
            continue
        nonce = intent.get("nonce")
        candidates[json.dumps(nonce, sort_keys=True)] = nonce
    if len(candidates) > 1:
        raise AgentLoopError("Competing managed-CI v2 intent comments exist for this PR head.")
    if candidates:
        (candidate,) = candidates.values()
        verdict = managed_ci_intent_validation.classify_intent_page(
            page,
            requested_nonce=candidate,
            trusted_login=trusted_login,
            trusted_id=trusted_id,
            visible_capable=contract.visible_intent_capable,
            host_footer_capable=contract.host_footer_capable,
            binding=_intent_binding(contract, nonce=candidate),
        )
        if verdict.outcome == "fail" or verdict.record is None or verdict.comment_id is None:
            raise ManagedCiIntentLedgerError(
                f"PR #{pr_number}: the managed-CI v2 intent for this head is one the base workflow "
                f"rejects ({verdict.reason}); repair or delete the intent comment and resume."
            )
        # A lone valid record of this generation is adopted whatever its
        # age: the dispatch-requested PATCH that precedes a dispatch renews
        # created_at in place.
        _restore_v2_intent_fields(contract, verdict.record)
        contract.intent_comment_id = verdict.comment_id
        return
    _reset_v2_intent_generation(contract)
    contract.nonce = secrets.token_urlsafe(24)
    contract.created_at = int(time.time())
    body = _intent_body(contract, pr_number=pr_number, expected_head_sha=expected_head_sha, state="prepared")
    # The create is verified against the stored body and the authenticated
    # producer; a numeric ID alone is not proof the ledger was persisted.
    written = post_verified_trusted_pr_protocol_comment_observed(
        runner,
        config=config,
        pr_number=pr_number,
        body=body,
        expected_author_login=trusted_login,
        expected_author_id=trusted_id,
    )
    _guard_host_footer_compatibility(contract, written, pr_number=pr_number)
    contract.intent_comment_id = written.comment_id
    contract.intent_state = "prepared"


def _patch_intent(
    runner: Runner, *, config: AgentLoopConfig, contract: ManagedCiContract, state: str
) -> WrittenProtocolComment | None:
    if contract.intent_comment_id is None or contract.nonce is None:
        return None
    _validate_v2_intent_publication(contract, state=state)
    body = _intent_body(
        contract,
        pr_number=contract.pr_number or 0,
        expected_head_sha=contract.expected_head_sha or "",
        state=state,
    )
    # The immutable identifying fields are already in the original marker;
    # state updates add only live provenance and are still actor-owned.  The
    # stored body and producer are read back before the state is recorded.
    written = patch_verified_trusted_protocol_comment_observed(
        runner,
        config=config,
        comment_id=contract.intent_comment_id,
        body=body,
        expected_author_login=_intent_producer_login(contract),
        expected_author_id=_intent_producer_id(contract),
    )
    _guard_host_footer_compatibility(contract, written, pr_number=contract.pr_number)
    contract.intent_state = state
    return written


def _guard_host_footer_compatibility(
    contract: ManagedCiContract, written: WrittenProtocolComment, *, pr_number: int | None
) -> None:
    """Stop when the host footered an intent that the base workflow cannot see.

    An older base workflow fullmatches only the unfootered envelope, so it
    would skip this intent: a dispatch would fail to find its authorization,
    and an already-dispatched run can no longer be trusted to qualify.
    """
    if not written.host_footer_observed or contract.host_footer_capable:
        return
    phase: Literal["pre-dispatch", "post-dispatch"] = (
        "post-dispatch"
        if contract.dispatch_issued or contract.attached_run_id is not None
        else "pre-dispatch"
    )
    raise ManagedCiHostFooterIncompatibleError(pr_number=pr_number, phase=phase, run_id=contract.attached_run_id)


class ManagedCiHostFooterIncompatibleError(AgentLoopError):
    """A footering host wrote an intent that an older base workflow would skip.

    Terminal and exempt from ordinary recovery: the operator must merge the
    workflow that advertises ``AGENT_LOOP_MANAGED_CI_HOST_FOOTER_V1`` first.
    """

    def __init__(
        self,
        *,
        pr_number: int | None,
        phase: Literal["pre-dispatch", "post-dispatch"],
        run_id: int | None = None,
    ) -> None:
        self.phase = phase
        subject = f"PR #{pr_number}" if pr_number is not None else "The managed PR"
        if phase == "pre-dispatch":
            outcome = "No managed-CI run was dispatched."
        else:
            run_text = f" ({run_id})" if run_id is not None else ""
            outcome = (
                f"The managed-CI run already dispatched{run_text} is left in place, "
                "is not cancelled, and does not qualify this head."
            )
        super().__init__(
            f"{subject}: this host appended its known comment footer to the managed-CI intent "
            f"record, but the base workflow does not advertise {HOST_FOOTER_INTENT_MARKER} and "
            f"would skip the footered intent ({phase}). {outcome} Merge the updated "
            f".github/workflows/ci.yml into the base branch, or run from a host that does not "
            "footer comments, then resume."
        )


class ManagedCiIntentLedgerError(AgentLoopError):
    """The PR's intent ledger holds a state the base workflow would fail.

    Terminal and exempt from ordinary recovery; the message names the
    offending comment so the operator can repair or delete it and resume.
    """


class ManagedCiIntentFreshnessError(ManagedCiIntentLedgerError):
    """The renewed intent would fall outside the workflow's age window."""


# The workflow's dispatch validator (MANAGED_CI_V2_DISPATCH_VALIDATOR) bounds
# the record's age in both directions.  The start margin covers queue and
# runner start-up latency between this gate and that validator.
INTENT_MAX_AGE_SECONDS = 15 * 60
INTENT_MAX_FUTURE_SKEW_SECONDS = 5 * 60
DISPATCH_START_MARGIN_SECONDS = 5 * 60


def _intent_binding(contract: ManagedCiContract, *, nonce: object) -> IntentBinding:
    return IntentBinding(
        repository=contract.repository,
        pr=contract.pr_number,
        expected_head_sha=contract.expected_head_sha,
        base_ref=contract.base_ref,
        workflow_revision=contract.workflow_revision,
        nonce=nonce,
    )


def _read_intent_comment_page(
    runner: Runner, config: AgentLoopConfig, pr_number: int
) -> list[object]:
    """Read the PR comment page exactly as the workflow receives it.

    Unlike ``_api_list``, a JSON list is returned unfiltered: a non-dict entry
    must reach the workflow-mirror scan, which fails closed on it as the
    workflow does, instead of collapsing to an uninspectable read.
    """
    result = runner.run(
        [
            config.gh_cmd,
            "api",
            "--paginate",
            f"repos/{config.repo}/issues/{pr_number}/comments?per_page=100",
        ],
        cwd=active_workdir(config),
        check=False,
    )
    if result.returncode != 0:
        raise AgentLoopError("Unable to inspect managed-CI v2 intent history.")
    # An empty body is not an empty ledger: treating it as ``[]`` would mint
    # a fresh intent without having inspected the existing comments.
    try:
        payload = json.loads(result.stdout or "")
    except json.JSONDecodeError as exc:
        raise AgentLoopError("Unable to inspect managed-CI v2 intent history.") from exc
    if not isinstance(payload, list):
        raise AgentLoopError("Unable to inspect managed-CI v2 intent history.")
    # The page stays raw (the envelope mirror owns the footered form), but a
    # footer seen only on this re-read is still reported once per invocation.
    for comment in payload:
        body = comment.get("body") if isinstance(comment, dict) else None
        if isinstance(body, str) and INTENT_MARKER in body and strip_known_host_footer(body)[1]:
            note_host_footer_observed(config, "managed-CI intent re-read")
            break
    return payload


def _require_workflow_dispatch_authorization(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    contract: ManagedCiContract,
    patch_result: WrittenProtocolComment | None,
) -> None:
    """Dispatch only what the workflow would accept, within its age window.

    The page is re-read after the dispatch-requested PATCH and must classify
    as ``authorized`` for this contract's comment.  The age window is checked
    against the PATCH's server ``updated_at`` (or local time when absent),
    never against the comment's creation time, which predates the renewal.
    """
    pr_number = contract.pr_number or 0
    trusted_login = _intent_producer_login(contract)
    trusted_id = _intent_producer_id(contract)
    page = _read_intent_comment_page(runner, config, pr_number)
    fatal = managed_ci_intent_validation.intent_page_pre_nonce_fatal(
        page,
        trusted_login=trusted_login,
        trusted_id=trusted_id,
        visible_capable=contract.visible_intent_capable,
        host_footer_capable=contract.host_footer_capable,
    )
    if fatal is not None:
        raise ManagedCiIntentLedgerError(
            f"PR #{pr_number}: managed-CI dispatch refused because the intent ledger holds a "
            f"comment the base workflow rejects ({fatal}); repair or delete that comment and resume."
        )
    verdict = managed_ci_intent_validation.classify_intent_page(
        page,
        requested_nonce=contract.nonce,
        trusted_login=trusted_login,
        trusted_id=trusted_id,
        visible_capable=contract.visible_intent_capable,
        host_footer_capable=contract.host_footer_capable,
        binding=_intent_binding(contract, nonce=contract.nonce),
    )
    if (
        verdict.outcome != "authorized"
        or verdict.record is None
        or verdict.comment_id != contract.intent_comment_id
    ):
        reason = verdict.reason or (
            f"authorized comment {verdict.comment_id} is not this invocation's intent "
            f"{contract.intent_comment_id}"
        )
        raise ManagedCiIntentLedgerError(
            f"PR #{pr_number}: managed-CI dispatch refused because the base workflow would not "
            f"authorize intent {contract.intent_comment_id} ({reason}); repair or delete the "
            "named comment and resume."
        )
    server_time = patch_result.server_updated_at if patch_result is not None else None
    reference = server_time if server_time is not None else int(time.time())
    source = "server updated_at" if server_time is not None else "local clock"
    created_at = verdict.record["created_at"]
    assert isinstance(created_at, int)
    age = reference - created_at
    if not (
        -INTENT_MAX_FUTURE_SKEW_SECONDS
        <= age
        <= INTENT_MAX_AGE_SECONDS - DISPATCH_START_MARGIN_SECONDS
    ):
        raise ManagedCiIntentFreshnessError(
            f"PR #{pr_number}: managed-CI dispatch refused because intent "
            f"{contract.intent_comment_id} has created_at={created_at}, which is outside the "
            f"workflow's age window at {reference} ({source}). Correct the local clock and resume."
        )


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
                if isinstance(run, dict) and managed_ci_failure._is_v2_intent_run(run, contract=contract, run_name=run_name)
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
                survivors = [run for run in survivors if not managed_ci_status._v2_terminal_attempt_excluded(
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
    runner: Runner, *, config: AgentLoopConfig, contract: ManagedCiContract,
    strict_provenance: bool = False,
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
    if strict_provenance and (
        run.get("head_branch") != (contract.dispatch_ref or contract.base_ref)
        or run.get("head_sha") != contract.workflow_revision
        or not isinstance(run.get("path"), str)
        or run["path"].split("@", 1)[0] != f".github/workflows/{WORKFLOW_FILE}"
        or any(not isinstance(run.get(field), dict)
               or run[field].get("login") != contract.trusted_actor_login
               or type(run[field].get("id")) is not int
               or run[field]["id"] != contract.trusted_actor_id for field in ("actor", "triggering_actor"))
    ):
        return None
    run_name = _v2_run_name(contract)
    if not managed_ci_failure._is_v2_intent_run(run, contract=contract, run_name=run_name):
        return None
    attempt = run.get("run_attempt")
    return ManagedCiRunSnapshot(
        contract.attached_run_id,
        attempt if isinstance(attempt, int) else contract.run_attempt,
        run.get("status") if isinstance(run.get("status"), str) else None,
        run.get("conclusion") if isinstance(run.get("conclusion"), str) else None,
        attempt if type(attempt) is int and attempt > 0 else None,
    )

def _v2_run_name(contract: ManagedCiContract) -> str:
    return f"{V2_RUN_NAME_PREFIX}{contract.nonce}"


def wait_for_final_qualification(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    metadata: PullRequestMetadata,
    contract: ManagedCiContract | None = None,
    attempts: int | None = None,
    min_run_attempt: int | None = None,
    deadline: float | None = None,
) -> ManagedCiOutcome:
    """Wait for the correlated managed exact-head result.

    ``min_run_attempt`` (#1477) ignores correlated terminal statuses and the
    terminal-without-status confirmation until the attached run reaches that
    attempt, so a plumbing rerun is judged only on its own attempt.
    ``deadline`` is the shared monotonic watch deadline; reaching it ends the
    wait with ``timeout`` even while polls remain.
    """
    expected_head = metadata.head_sha
    if not expected_head:
        raise AgentLoopError(f"PR #{pr_number} has no approved head SHA.")
    if (
        contract is not None
        and contract.protocol_version == 2
        and (contract.nonce is not None or contract.attached_run_id is not None)
        and contract.expected_head_sha != expected_head
    ):
        raise AgentLoopError(
            f"managed-CI intent/attempt is bound to head {contract.expected_head_sha or 'unknown'}, "
            f"not {expected_head}; no qualification outcome is reported for this head"
        )
    if attempts is None:
        attempts = max(1, config.ci_timeout_seconds // config.ci_poll_interval_seconds)
    attempts = max(1, attempts)
    latest: PullRequestChecks | None = None
    terminal_confirmation: tuple[int, int | None] | None = None
    terminal_confirmation_seen = False
    last_status: str | None = None
    for attempt in range(attempts):
        live_head = get_pr_head_sha(runner, config, pr_number)
        if live_head != expected_head:
            return ManagedCiOutcome(
                status="head_changed", checks=latest, head_sha=live_head,
                attempts_used=attempt + 1,
            )
        mergeability = get_pr_mergeability(runner, config=config, pr_number=pr_number)
        if mergeability.state == "conflicted":
            return ManagedCiOutcome(
                status="merge_conflict",
                checks=latest,
                mergeability=mergeability,
                head_sha=live_head,
                attempts_used=attempt + 1,
            )
        latest = get_pr_checks(runner, config=config, metadata=metadata)
        run_snapshot: ManagedCiRunSnapshot | None = None
        if contract is not None and contract.protocol_version == 2:
            if _v2_attachment_is_excluded(contract):
                # Terminal attempts stay diagnostic history, but cannot remain
                # the active correlation key for a retry.
                contract.attached_run_id = None
                contract.run_attempt = None
            if contract.attached_run_id is None:
                run_snapshot = _discover_v2_snapshot(
                    runner, config=config, contract=contract, retries=1, include_terminal=True
                )
                if run_snapshot is not None:
                    contract.attached_run_id, contract.run_attempt = run_snapshot.run_id, run_snapshot.run_attempt
                    contract.terminal_outcome = None
                    if contract.run_attempt is not None:
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
                    contract.terminal_outcome = None
                    _patch_intent(runner, config=config, contract=contract, state="attached")
            final = managed_ci_status._v2_correlated_status(
                runner, config=config, expected_head=expected_head, contract=contract
            )
            if run_snapshot is not None and not managed_ci_status._v2_attempt_is_correlatable(run_snapshot.run_attempt):
                log(
                    config,
                    f"PR #{pr_number}: managed-CI v2 run {run_snapshot.run_id} has no positive "
                    "run attempt; terminal status correlation is deferred",
                )
        else:
            final = _find_context(latest, FINAL_CONTEXT)
        stale_attempt = (
            min_run_attempt is not None
            and contract is not None
            and (not isinstance(contract.run_attempt, int) or contract.run_attempt < min_run_attempt)
        )
        if stale_attempt:
            # The pre-rerun attempt's result is not this attempt's evidence.
            final = None
            run_snapshot = None
        status = final.status.lower() if final is not None else "pending"
        last_status = status if final is not None else "absent"
        log(config, f"Managed CI context '{FINAL_CONTEXT}' status: {status}")
        complete_board = managed_ci_status.complete_managed_board(latest)
        if status == "success" and complete_board:
            if contract is not None and contract.protocol_version == 2:
                _patch_intent(runner, config=config, contract=contract, state="completed")
            return ManagedCiOutcome(
                status="passed", checks=latest, head_sha=live_head, attempts_used=attempt + 1,
            )
        correlated_terminal = final is not None and final.status.lower() in _TERMINAL_CI_STATUSES
        if (
            contract is not None and contract.protocol_version == 2
            and run_snapshot is not None
            and (run_snapshot.status or "").lower() == "completed"
            and not correlated_terminal
            and managed_ci_status._v2_attempt_is_correlatable(run_snapshot.run_attempt)
        ):
            key = (run_snapshot.run_id, run_snapshot.run_attempt)
            if key != terminal_confirmation:
                terminal_confirmation = key
                terminal_confirmation_seen = False
                # A publisher may race the workflow's completed transition. Re-read
                # immediately before starting the bounded confirmation window.
                final = managed_ci_status._v2_correlated_status(
                    runner, config=config, expected_head=expected_head, contract=contract
                )
                if final is not None:
                    # Any correlated observation from the reread is the
                    # freshest final state, terminal or not.
                    last_status = final.status.lower()
                if final is not None and final.status.lower() in _TERMINAL_CI_STATUSES:
                    status = final.status.lower()
                    if status == "success" and complete_board:
                        _patch_intent(runner, config=config, contract=contract, state="completed")
                        return ManagedCiOutcome(
                            status="passed", checks=latest, head_sha=live_head,
                            attempts_used=attempt + 1,
                        )
                else:
                    terminal_confirmation_seen = True
            elif terminal_confirmation_seen:
                contract.terminal_run_id = run_snapshot.run_id
                contract.terminal_run_attempt = run_snapshot.run_attempt
                terminal_key = (run_snapshot.run_id, run_snapshot.run_attempt)
                if terminal_key not in contract.terminal_attempts:
                    contract.terminal_attempts += (terminal_key,)
                contract.terminal_outcome = "no-status"
                _patch_intent(runner, config=config, contract=contract, state="completed")
                return ManagedCiOutcome(
                    status="terminal_without_status", checks=latest, head_sha=live_head,
                    run_id=run_snapshot.run_id, run_attempt=run_snapshot.run_attempt,
                    workflow_status=run_snapshot.status, workflow_conclusion=run_snapshot.conclusion,
                    jobs=_v2_jobs(runner, config=config, run_id=run_snapshot.run_id),
                    attempts_used=attempt + 1,
                )
        if status in _TERMINAL_CI_STATUSES - {"success"}:
            details = ()
            jobs = None
            run_id = None
            if contract is not None and contract.protocol_version == 2:
                _patch_intent(runner, config=config, contract=contract, state="completed")
                run_id = contract.attached_run_id
                jobs = _v2_jobs(runner, config=config, run_id=run_id)
                details = _format_v2_failed_jobs(run_id, jobs)
            return ManagedCiOutcome(
                status="failed", checks=latest, head_sha=live_head, failure_details=details,
                run_id=run_id, final_status=status,
                run_attempt=contract.run_attempt if contract is not None else None,
                jobs=jobs, attempts_used=attempt + 1,
            )
        early = managed_ci_failure.observe_active_failure(
            runner, config=config, pr_number=pr_number, metadata=metadata,
            contract=contract, minimum=min_run_attempt,
        ) if run_snapshot is not None and run_snapshot.status == "in_progress" else None
        if early is not None:
            snapshot, jobs = early
            return ManagedCiOutcome(
                status="failed", checks=latest, head_sha=live_head, jobs=jobs,
                run_id=snapshot.run_id, run_attempt=snapshot.live_run_attempt,
                workflow_status=snapshot.status, workflow_conclusion=snapshot.conclusion,
                failure_details=(f"Managed CI run {snapshot.run_id} attempt {snapshot.live_run_attempt}",)
                + _format_v2_failed_jobs(snapshot.run_id, jobs),
                attempts_used=attempt + 1, final_status=last_status,
            )
        stall_checks = intermediate_managed_checks(latest)
        if is_wholly_infrastructure_blocked(stall_checks):
            stall = CiInfrastructureStall(checks=stall_checks.infrastructure_stalls)
            return ManagedCiOutcome(
                status="infrastructure_stall",
                checks=latest,
                head_sha=live_head,
                stall=stall,
                attempts_used=attempt + 1,
                final_status=last_status,
            )
        if deadline is not None and time.monotonic() >= deadline:
            return ManagedCiOutcome(
                status="timeout", checks=latest, head_sha=expected_head,
                attempts_used=attempt + 1, final_status=last_status,
            )
        if attempt < attempts - 1:
            runner.run(["sleep", str(config.ci_poll_interval_seconds)], cwd=active_workdir(config))
    return ManagedCiOutcome(
        status="timeout", checks=latest, head_sha=expected_head, attempts_used=attempts,
        final_status=last_status,
    )


def _v2_jobs(
    runner: Runner, *, config: AgentLoopConfig, run_id: int | None
) -> tuple[tuple[str, str, str], ...] | None:
    """Return the run's latest-attempt jobs as (name, conclusion, url); ``None`` if unreadable."""
    return read_run_jobs(runner, config, run_id)


def _format_v2_failed_jobs(
    run_id: int | None, jobs: tuple[tuple[str, str, str], ...] | None
) -> tuple[str, ...]:
    if run_id is None:
        return ()
    if jobs is None:
        return (f"Managed CI run {run_id} failed; job details were unavailable.",)
    details = [
        f"{name}: {conclusion}" + (f" ({url})" if url else "")
        for name, conclusion, url in jobs
        if conclusion in FAILED_JOB_CONCLUSIONS
    ]
    return tuple(details) or (f"Managed CI run {run_id} failed; no failed job was exposed.",)


def _v2_failed_jobs(runner: Runner, *, config: AgentLoopConfig, run_id: int | None) -> tuple[str, ...]:
    return _format_v2_failed_jobs(run_id, _v2_jobs(runner, config=config, run_id=run_id))


format_failed_jobs = _format_v2_failed_jobs


def classify_managed_jobs(
    jobs: tuple[tuple[str, str, str], ...] | None, plumbing_names: frozenset[str]
) -> Literal["plumbing-only", "test-failure", "not-classifiable"]:
    """Classify a completed managed run from its complete jobs listing (#1477).

    Plumbing-only needs at least one failed job, every failed job catalogued
    plumbing, and every non-plumbing (test) job successful.  An unreadable
    listing, a skipped/pending test or no failed job is not classifiable.
    """
    verdict = classify_run_jobs(jobs, plumbing_names, require_passing_tests=True)
    return "not-classifiable" if verdict == "no-failed-jobs" else verdict


def classify_managed_board(
    checks: PullRequestChecks | None, plumbing_names: frozenset[str]
) -> Literal["ok", "test-failure", "incomplete"]:
    """Gate a managed plumbing decision on the counted PR-head board (#1477).

    The managed run's own jobs are not on the PR-head board, so the board must
    independently show no failing test (test precedence) and be authoritative
    and complete: query ``ok``, complete listing, readable protection, and no
    pending or missing check other than ``final-ci/exact-head`` itself, which
    is legitimately unpublished on the terminal-without-status path.
    """
    if checks is None:
        return "incomplete"
    if any(not check_name_matches(check.name, plumbing_names) for check in checks.failing):
        return "test-failure"
    if not board_tests_passed(checks, plumbing_names):
        return "incomplete"
    if not (
        checks.check_query_status == "ok"
        and checks.listing_complete
        and checks.branch_protection_status in {"configured", "not_found"}
        and not any(name != FINAL_CONTEXT for name in checks.missing_required)
        and not any(check.name != FINAL_CONTEXT for check in checks.pending)
    ):
        return "incomplete"
    return "ok"


@dataclass(frozen=True)
class ManagedRerunResult:
    status: Literal["rerun_started", "refused", "command_failed", "expired"]
    run_id: int | None = None
    run_url: str | None = None
    prior_attempt: int | None = None
    reason: str = ""


def managed_run_url(config: AgentLoopConfig, run_id: int | None) -> str | None:
    return f"https://github.com/{config.repo}/actions/runs/{run_id}" if run_id is not None else None


def rerun_managed_attempt(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    contract: ManagedCiContract,
    expected_head_sha: str | None,
    deadline: float | None = None,
) -> ManagedRerunResult:
    """Fully re-run the attached managed v2 run once, from terminal-without-status only.

    The dispatch validator accepts attempt N+1 only against a completed
    ``no-status`` audit record bound to (run, N), so the authenticated record
    is re-read and required to hold exactly that transition before the write.
    The rerun is full (no ``--failed``): the split publisher rejects a
    validation job carried over from an earlier attempt.
    """
    run_id = contract.attached_run_id
    run_url = managed_run_url(config, run_id)

    def refuse(reason: str, attempt: int | None = None) -> ManagedRerunResult:
        return ManagedRerunResult(
            status="refused", run_id=run_id, run_url=run_url, prior_attempt=attempt, reason=reason,
        )

    if config.dry_run:
        return refuse("dry-run never re-runs CI")
    if contract.protocol_version != 2 or run_id is None or not contract.nonce:
        return refuse("no attached managed v2 run is available")
    if not expected_head_sha or contract.expected_head_sha != expected_head_sha:
        return refuse("the managed contract is not bound to the approved head")
    if contract.intent_comment_id is None:
        return refuse("the managed audit record is unavailable")
    try:
        envelope = _fetch_protocol_comment_envelope(
            runner, config=config, comment_id=contract.intent_comment_id,
            context="Managed-CI audit record",
        )
    except AgentLoopError as error:
        return refuse(f"the managed audit record could not be re-read ({error})")
    user = envelope.get("user") if isinstance(envelope.get("user"), dict) else {}
    try:
        producer_ok = (
            user.get("login") == _intent_producer_login(contract)
            and user.get("id") == _intent_producer_id(contract)
        )
    except AgentLoopError:
        producer_ok = False
    match = managed_ci_intent_validation.match_intent_envelope(
        envelope.get("body"),
        visible_capable=contract.visible_intent_capable,
        host_footer_capable=contract.host_footer_capable,
    )
    if not producer_ok or match is None:
        return refuse("the managed audit record is not the authenticated producer's record")
    try:
        record = json.loads(match.group("payload"))
    except json.JSONDecodeError:
        return refuse("the managed audit record is unreadable")
    if not isinstance(record, dict) or managed_ci_intent_validation.validate_intent_record(
        record, match, binding=_intent_binding(contract, nonce=contract.nonce)
    ) is not None:
        return refuse("the managed audit record does not validate for this contract")
    attempt = record.get("run_attempt")
    history = [
        item.get("run_attempt") for item in record.get("terminal_attempts") or []
        if isinstance(item, dict) and item.get("run_id") == run_id
    ]
    if not (
        record.get("state") == "completed"
        and record.get("terminal_outcome") == "no-status"
        and record.get("run_id") == run_id
        and record.get("terminal_run_id") == run_id
        and isinstance(attempt, int)
        and record.get("terminal_run_attempt") == attempt
        and history
        and max(history) == attempt
    ):
        return refuse(
            "the managed audit record does not hold the completed no-status transition "
            "for the run's current attempt"
        )
    run = read_actions_run(runner, config, run_id)
    if (
        run is None
        or not managed_ci_failure._is_v2_intent_run(run, contract=contract, run_name=_v2_run_name(contract))
        or run.get("status") != "completed"
        or run.get("run_attempt") != attempt
    ):
        return refuse("the attached run does not match the authenticated contract", attempt)
    try:
        login, actor_id = resolve_authenticated_github_actor(runner, config=config)
    except AgentLoopError as error:
        return refuse(f"the authenticated actor is unavailable ({error})", attempt)
    if (
        (login or "").casefold() != (contract.trusted_actor_login or "").casefold()
        or actor_id != contract.trusted_actor_id
    ):
        return refuse("the authenticated actor is not the managed trusted actor", attempt)
    if deadline is not None and time.monotonic() >= deadline:
        # The verification reads consumed the shared watch budget: never
        # start CI that nothing will observe.
        return ManagedRerunResult(
            status="expired", run_id=run_id, run_url=run_url, prior_attempt=attempt,
            reason="the shared CI watch deadline passed before the rerun could be issued",
        )
    result = runner.run(
        [config.gh_cmd, "run", "rerun", str(run_id), "--repo", config.repo],
        cwd=active_workdir(config),
        check=False,
    )
    if result.returncode != 0:
        return ManagedRerunResult(
            status="command_failed", run_id=run_id, run_url=run_url, prior_attempt=attempt,
            reason="gh run rerun failed" + failure_suffix(result),
        )
    return ManagedRerunResult(
        status="rerun_started", run_id=run_id, run_url=run_url, prior_attempt=attempt,
    )


@managed_ci_api._surfaces_list_failures
def prepare_v2_merge(
    runner: Runner, *, config: AgentLoopConfig, pr_number: int, expected_head_sha: str, contract: ManagedCiContract,
    verified_result: VerifiedManagedResult | None = None,
) -> None:
    """Publish continuity before readying the PR, then re-check its exact head."""
    if contract.protocol_version != 2:
        return
    if verified_result is not None:
        managed_ci_reconciliation.require_verified_result(runner, config=config, pr_number=pr_number,
                                expected_head_sha=expected_head_sha, contract=contract, result=verified_result)
    require_recorded_base(
        runner, config=config, pr_number=pr_number, base_ref=contract.base_ref,
        repository=contract.repository or config.repo, head_sha=expected_head_sha,
        action="readiness for merge",
    )
    labelled = _label_add(
        runner, config=config, pr_number=pr_number, label_name=QUALIFIED_LABEL
    ).result
    if labelled.returncode != 0:
        raise AgentLoopError(
            f"Unable to apply `{QUALIFIED_LABEL}` before readying PR #{pr_number}.{failure_suffix(labelled)}"
        )
    pr = managed_ci_api._api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}")
    if pr.get("draft") is True:
        require_recorded_base(
            runner, config=config, pr_number=pr_number, base_ref=contract.base_ref,
            repository=contract.repository or config.repo, head_sha=expected_head_sha,
            action="ready-for-review transition",
        )
        ready = reconciled_pr_ready(
            runner, config=config, pr_number=pr_number, expected_head_sha=expected_head_sha,
        )
        if ready.returncode != 0:
            raise AgentLoopError(
                f"Unable to mark qualified PR #{pr_number} ready for review.{failure_suffix(ready)}"
            )
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
    unverified_green_attempts = 0
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
        # A draft reports DRAFT, never CLEAN; under unreadable protection the
        # finalizer re-checks CLEAN after readiness and before any merge.
        reliable = latest.check_query_status == "ok" and (
            board_protection_is_reliable(latest, mergeability, head_sha=live_head)
            or protection_awaits_readiness(latest, mergeability, head_sha=live_head)
        )
        green_board = (
            recovery_runs
            and any(run.get("status") == "completed" and run.get("conclusion") == "success" for run in completed)
            and latest.check_query_status == "ok"
            and latest.state == "passing"
            and bool(latest.passing)
            and not latest.pending
            and not latest.missing_required
        )
        if green_board and reliable:
            log(config, f"PR #{capability.pr_number}: ordinary unlabeled recovery passed at {expected_head}")
            return ManagedCiOutcome(
                status="passed", checks=latest, mergeability=mergeability, head_sha=live_head,
            )
        if green_board and latest.branch_protection_status == "forbidden":
            # The board is green but neither CLEAN nor a same-head DRAFT can
            # stand in for the unreadable protection.  Stop after the bounded
            # startup window instead of polling to the full CI timeout.
            unverified_green_attempts += 1
            if unverified_green_attempts >= startup_attempt_limit:
                return ManagedCiOutcome(
                    status="protection_unreadable", checks=latest,
                    mergeability=mergeability, head_sha=live_head,
                )
        else:
            unverified_green_attempts = 0
        if not recovery_runs and attempt + 1 >= startup_attempt_limit:
            return ManagedCiOutcome(status="not_started", checks=latest, head_sha=live_head)
        if attempt < attempts - 1:
            runner.run(["sleep", str(config.ci_poll_interval_seconds)], cwd=active_workdir(config))
    if unverified_green_attempts:
        # The budget expired on a green board that only the unreadable
        # protection kept from qualifying; report that cause, not a timeout.
        return ManagedCiOutcome(
            status="protection_unreadable", checks=latest,
            mergeability=mergeability, head_sha=live_head,
        )
    return ManagedCiOutcome(status="timeout", checks=latest, head_sha=expected_head)


def validate_ordinary_recovery_capability(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    capability: OrdinaryRecoveryCapability,
    require_draft: bool | None = True,
) -> bool:
    """Revalidate the invocation-owned fallback tuple before readiness/merge."""
    pr = managed_ci_api._api_json(runner, config, f"repos/{config.repo}/pulls/{capability.pr_number}", quiet=True)
    head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
    base = pr.get("base") if isinstance(pr.get("base"), dict) else {}
    head_repo = head.get("repo") if isinstance(head.get("repo"), dict) else {}
    event = managed_ci_api._active_managed_label_event(
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
