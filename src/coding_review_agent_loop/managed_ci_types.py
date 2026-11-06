"""Shared managed-CI records and constants."""
from __future__ import annotations
from .managed_ci_tuple import MANAGED_LABEL
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from .ci_health import CiInfrastructureStall, PullRequestChecks
from .github import PullRequestMergeability
from .runner import CommandResult
from .managed_ci_status import FINAL_CONTEXT


MANAGED_OPT_OUT_LABEL = "agent-loop-managed-opt-out"


QUALIFIED_LABEL = "agent-loop-exact-head-qualified"


QUALIFICATION_MARKER = "AGENT_LOOP_MANAGED_CI_QUALIFIED_V2"


# Plumbing job identity (#1477): the exact managed workflow job names that
# validate, attest and publish rather than test.  A contract test pins each
# name to a ``name:`` in the managed workflows; callers extend the catalog
# only with explicit --ci-plumbing-check names, never by substring.
MANAGED_PLUMBING_JOB_NAMES = frozenset({
    "Validate managed exact-head dispatch",
    "Verify split attestations",
    "Publish final exact-head status",
})


PLUMBING_STATUS_CONTEXTS = frozenset({FINAL_CONTEXT})


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


# Advertised by a base workflow whose intent validator admits the fixed
# visible authorization line ahead of the record (#935).
VISIBLE_INTENT_MARKER = "AGENT_LOOP_MANAGED_CI_VISIBLE_INTENT_V1"


# Advertised by a base workflow whose intent validator admits the record
# followed by exactly the known host comment footer, matched against the raw
# comment body (#1043).
HOST_FOOTER_INTENT_MARKER = "AGENT_LOOP_MANAGED_CI_HOST_FOOTER_V1"


UNPROTECTED_OVERRIDE_TRAILER = "AGENT_MANAGED_CI_UNPROTECTED_OVERRIDE_V1"


# Default-branch workflows advertise trusted integration-base support with this
# literal; the allow-list itself is the AGENT_LOOP_TRUSTED_BASES repository
# variable, which a PR cannot change (#1285).
TRUSTED_BASES_MARKER = "AGENT_LOOP_MANAGED_CI_TRUSTED_BASES_V1"


TRUSTED_BASES_VARIABLE = "AGENT_LOOP_TRUSTED_BASES"


ISSUE_AUTHORIZATION_MARKER = "AGENT_MANAGED_CI_ISSUE_AUTHORIZATION_V1"


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

    state: Literal["strict", "voluntary", "plan_limited", "unreadable", "indeterminate"]
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
    # ``asserted`` means the Actions variable read was refused with HTTP 403
    # and ``--managed-ci-trusted-actor`` stood in for it.  The value is then
    # unverified locally; the workflow still enforces the real variable.
    advertised_actor_source: Literal["variable", "asserted"] | None = None


@dataclass(frozen=True)
class ManagedActorVariable:
    """One classified read of the ``AGENT_LOOP_MANAGED_ACTOR`` variable."""

    value: str | None
    status: Literal["readable", "absent", "unreadable", "error"]
    detail: str


# The explicit per-invocation waivers.  ``unreadable`` protection is never
# implied by --allow-unprotected-managed-ci alone (#1040).
_UNPROTECTED_WAIVER = "allow-unprotected-managed-ci"


_UNREADABLE_WAIVER = "allow-unreadable-protection"


_OVERRIDE_PROTECTION_STATES = frozenset({"voluntary", "plan_limited", "unreadable"})


_UNREADABLE_PROTECTION_DETAIL = (
    "classic branch protection is not readable by this token (HTTP 403); "
    "effective rules show no strict final-ci/exact-head enforcement"
)


_HIDDEN_BYPASS_DETAIL = (
    "; a ruleset requiring final-ci/exact-head does not expose its bypass actors to this token"
)


PREFLIGHT_STRICT_READY = 0


PREFLIGHT_KNOWN_NOT_READY = 10


PREFLIGHT_INDETERMINATE = 11


# Only the newest records matter to the post-coder-turn audit's skew window.
V1_DISPATCH_RECORD_LIMIT = 8


@dataclass(frozen=True)
class ManagedV1DispatchRecord:
    """One successful managed-CI v1 dispatch made by this process (#1414).

    A v1 run's name is the plain workflow name and GitHub's run listing omits
    dispatch inputs, so the post-coder-turn audit correlates these fields
    with listed runs to avoid attributing the orchestrator's own dispatch to
    the coder.  In-process only: never serialized into any durable record.
    """

    workflow_file: str
    ref: str
    pr_number: int
    expected_head_sha: str
    requested_at: datetime
    returned_at: datetime


@dataclass
class ManagedCiContract:
    workflow_file: str = WORKFLOW_FILE
    protocol_version: int = 1
    base_ref: str | None = None
    # The managed workflow is always read, revision-bound and dispatched on the
    # default branch; ``base_ref`` is the PR's real (possibly integration) base.
    dispatch_ref: str | None = None
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
    # ``completed`` is the only router-compatible terminal lifecycle state.
    # Keep the no-status result as separate round metadata so older routers do
    # not have to parse a producer-only lifecycle value.
    terminal_outcome: Literal["no-status"] | None = None
    activation_path: Literal["managed", "ordinary_fallback"] = "managed"
    ordinary_recovery: "OrdinaryRecoveryCapability | None" = None
    # Derived from the exact base workflow at activation. Retain it for a
    # delayed dispatch-time fallback so a suppressed draft is released only
    # when that same workflow advertises an unlabeled CI route.
    ordinary_recovery_capable: bool = False
    # Derived from the same base workflow. Only a workflow that advertises
    # the visible-intent envelope may receive the prefixed intent body.
    visible_intent_capable: bool = False
    # Derived from the same base workflow. Only a workflow that advertises
    # the host-footer envelope can find an intent a footering host stored.
    host_footer_capable: bool = False
    # True once this generation issued a workflow dispatch or attached an
    # existing run.  The ``dispatch-requested`` lifecycle state is never
    # evidence of a dispatch.
    dispatch_issued: bool = False
    # Explicit lifecycle provenance.  These fields are intentionally not
    # inferred from public mode flags after activation.
    origin: Literal["issue-created", "source-managed"] | None = None
    lifecycle: Literal[
        "creation", "draft-labeled", "draft-unlabeled-reentry", "ready-unlabeled-reentry"
    ] | None = None
    authenticated_resume: "AuthenticatedManagedResume | None" = None
    # Measured before/now report for a non-raising activation fallback that
    # followed a mutation by this run (#1067).  Callers print it instead of a
    # hard-coded post-state sentence.
    state_report: str | None = None
    # Final write diagnostic and attempt history of the last failed
    # release_adopted_managed_ci removal (empty when it succeeded).
    release_diagnostic: str = ""
    # Successful v1 dispatches of this process, newest last, for the
    # post-coder-turn dispatch audit (#1414).  Never persisted.
    v1_dispatch_records: tuple[ManagedV1DispatchRecord, ...] = ()


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
    # The nonce carried by the authenticated PR-opening body, when present.
    # Fresh operator grants have their own nonce and retain this separately so
    # resume validation can distinguish historical opening provenance from the
    # grant that is being activated.
    opening_override_nonce: str | None = None
    active_label_event_id: int | None = None
    lifecycle: Literal[
        "draft-labeled", "draft-unlabeled-reentry", "ready-unlabeled-reentry"
    ] = "draft-labeled"
    authorization_kind: Literal["creation", "fresh", "continuity"] = "creation"
    authorization_comment_id: int | None = None
    approved_plan_hash: str | None = None
    # Approved plans a verified signed child-plan supersession retired (#993).
    # Carried so a fresh re-validation treats their authorizations as history.
    retired_plan_hashes: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ManagedCiIssueAuthorization:
    """Canonical PR-comment authorization for an issue-created managed PR."""

    kind: Literal["creation", "fresh", "continuity"]
    repository: str
    issue_number: int
    pr_number: int
    base_ref: str
    head_sha: str
    actor_login: str
    actor_id: int
    protection: str
    waiver: str
    nonce: str
    label_event_id: int
    predecessor_head: str | None = None
    predecessor_comment_id: int | None = None
    round_comment_ids: tuple[int, ...] = ()
    approved_plan_hash: str | None = None
    # Earlier actor-owned records this fresh grant (#1065) or continuity
    # record (#1069) retires.  A superseded record is history: it is neither
    # a competing grant nor a predecessor, so accumulated unbound records
    # cannot wedge a PR.
    superseded_comment_ids: tuple[int, ...] = ()

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "version": 1,
            "kind": self.kind,
            "repository": self.repository,
            "issue": self.issue_number,
            "pr": self.pr_number,
            "base": self.base_ref,
            "head": self.head_sha,
            "actor": self.actor_login,
            "actor_id": self.actor_id,
            "protection": self.protection,
            "waiver": self.waiver,
            "nonce": self.nonce,
            "label_event_id": self.label_event_id,
        }
        if self.predecessor_head is not None:
            payload["predecessor_head"] = self.predecessor_head
        if self.predecessor_comment_id is not None:
            payload["predecessor_comment_id"] = self.predecessor_comment_id
        if self.round_comment_ids:
            payload["round_comment_ids"] = list(self.round_comment_ids)
        if self.approved_plan_hash is not None:
            payload["approved_plan_hash"] = self.approved_plan_hash
        if self.superseded_comment_ids:
            payload["superseded_comment_ids"] = list(self.superseded_comment_ids)
        return payload


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


# Record kinds that may retire earlier records: the explicit fresh grant
# (#1065) and the ordinary path's continuity record (#1069).  A creation
# checkpoint is the PR's first record and never supersedes.
_SUPERSEDING_AUTHORIZATION_KINDS = frozenset({"fresh", "continuity"})


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
        "protection_unreadable",
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
    # Structured job listing of the attached run (#1477): (name, conclusion,
    # url) per job; ``None`` when the listing was not read or was unreadable.
    jobs: tuple[tuple[str, str, str], ...] | None = None
    attempts_used: int = 0
    # Last observed correlated final-ci/exact-head state, for terminal reports.
    final_status: str | None = None


@dataclass(frozen=True)
class ManagedCiRunSnapshot:
    run_id: int
    run_attempt: int | None
    status: str | None
    conclusion: str | None
    live_run_attempt: int | None = None


@dataclass(frozen=True)
class LabelEvent:
    event_id: int
    kind: str
    actor_login: str
    actor_id: int


@dataclass(frozen=True)
class CompleteLabelHistory:
    """Complete event history of one label: its latest event, or ``None``."""

    latest: LabelEvent | None


@dataclass(frozen=True)
class UnknownLabelHistory:
    reason: str


@dataclass(frozen=True)
class ReconciledLabelWrite:
    """Outcome of a reconciled label write.

    ``result`` keeps the callers' ``returncode`` contract: success (possibly
    synthesized after the history proved the write landed) or the failed result
    that sends the caller down its existing refusal path.  ``event_id`` is the
    adopted ``labeled`` event id when a transient failure was proven accepted.
    """

    result: CommandResult
    event_id: int | None = None
    outcome: str = "direct"
