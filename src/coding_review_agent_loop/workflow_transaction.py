"""Typed cross-surface workflow transaction model (#827, stage A).

One logical workflow transition (issue-to-PR handoff, PR expected-closing
contract, managed-CI authorization, initial coder round) is described by one
immutable :class:`WorkflowTransition` intent.  The canonical hash of that
intent is the transaction ID; both per-surface version-2 payloads are derived
from the intent, so their flow, issue, plan hash, and closing scope cannot
disagree.

A transaction is recorded append-only on the PR conversation: one ``prepared``
record that stores the complete intent, and at most one terminal record
(``committed`` or ``aborted``).  Nothing in this module writes to GitHub and no
existing call site uses it yet: it holds the model, the codecs, and the pure
resolvers.  Every resolver that interprets a version-2 record accepts only an
:class:`~coding_review_agent_loop.github.AuthenticatedCommentView`, so a
comment snapshot that was not bound to the invocation's authenticated actor
cannot reach version-2 interpretation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from .errors import AgentLoopError, WorkflowTransactionError
from .expected_closure import contract_hash, normalize_issue_ids
from .github import (
    ISSUE_THREAD_SURFACE,
    PR_THREAD_SURFACE,
    AuthenticatedComment,
    AuthenticatedCommentView,
    comment_thread_surface,
    parse_comment_thread_surface,
    parse_comment_timestamp,
)
from .issue_pr_handoff import (
    AGENT_ISSUE_PR_HANDOFF_RE,
    IssuePrHandoffLineage,
    IssuePrHandoffMetadata,
    IssuePrHandoffMetadataV2,
    _decode_issue_pr_handoff_metadata,
    decode_issue_pr_handoff_v2,
    encode_issue_pr_handoff_v2,
    issue_pr_handoff_payload_schema_version,
    issue_pr_handoff_record_hash,
    resolve_issue_pr_handoff_lineage,
)
from .pr_contract import (
    PR_EXPECTED_CLOSING_MARKER_RE,
    PrExpectedClosingContract,
    PrExpectedClosingContractV2,
    decode_pr_contract,
    decode_pr_contract_v2,
    find_latest_pr_contract,
    make_pr_contract_v2,
    pr_contract_payload_schema_version,
    pr_contract_record_hash,
)
from .protocol_markers import TrustedBody
from .round_state import (
    PostedRoundRecord,
    _approved_plan_hash,
    _extract_round_metadata_records,
    _legacy_freeform_plan_candidates,
    _plan_subject,
)
from .round_transport import ROUND_RESUME_MARKER_RE

INTENT_SCHEMA_VERSION = 1
RECORD_SCHEMA_VERSION = 1
WORKFLOW_TRANSACTION_MARKER = "AGENT_WORKFLOW_TRANSACTION"
WORKFLOW_TRANSACTION_MARKER_RE = re.compile(
    rf"<!--\s*{WORKFLOW_TRANSACTION_MARKER}:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.I,
)

KIND_INITIAL = "initial"
KIND_LEGACY_ROOT_CORRECTION = "legacy-root-correction"
KIND_PLAN_REPLACEMENT = "plan-replacement"
KIND_FLOW_CORRECTION = "flow-correction"
KIND_CLOSING_WIDENING = "closing-widening"
KIND_MANAGED_CI_CONTINUITY = "managed-ci-continuity"
KIND_HEAD_ADVANCE = "head-advance"
# Successor kinds in the fixed precedence used when several fields differ.
SUCCESSOR_KIND_PRECEDENCE: tuple[str, ...] = (
    KIND_PLAN_REPLACEMENT,
    KIND_FLOW_CORRECTION,
    KIND_CLOSING_WIDENING,
    KIND_MANAGED_CI_CONTINUITY,
    KIND_HEAD_ADVANCE,
)
SUCCESSOR_KINDS = frozenset(
    {KIND_INITIAL, KIND_LEGACY_ROOT_CORRECTION, *SUCCESSOR_KIND_PRECEDENCE}
)

FLOW_ISSUE = "issue-implementation"
FLOW_APPROVED_PLAN = "approved-plan-implementation"
FLOW_DIRECT_PR = "direct-pr"
FLOW_MANAGED_PR = "managed-pr"
ORIGIN_FLOWS = frozenset({FLOW_ISSUE, FLOW_APPROVED_PLAN, FLOW_DIRECT_PR, FLOW_MANAGED_PR})
_ISSUE_ORIGIN_FLOWS = frozenset({FLOW_ISSUE, FLOW_APPROVED_PLAN})

ENTRY_HANDOFF = "issue-pr-handoff"
ENTRY_PR_CONTRACT = "pr-expected-closing-contract"
ENTRY_AUTHORIZATION = "managed-ci-authorization"
ENTRY_INITIAL_CODER_ROUND = "initial-coder-round"
# Fixed order: it is part of the canonical intent serialization.
RECORD_SET_ENTRY_NAMES: tuple[str, ...] = (
    ENTRY_HANDOFF,
    ENTRY_PR_CONTRACT,
    ENTRY_AUTHORIZATION,
    ENTRY_INITIAL_CODER_ROUND,
)

DISPOSITION_REISSUED = "reissued"
DISPOSITION_INHERITED = "inherited"
DISPOSITION_NOT_APPLICABLE = "not-applicable"
_DISPOSITIONS = frozenset(
    {DISPOSITION_REISSUED, DISPOSITION_INHERITED, DISPOSITION_NOT_APPLICABLE}
)

ABSENCE_FLOW_WITHOUT_PLAN_REVIEW = "flow-without-plan-review"
ABSENCE_NO_PLAN_SCHEDULER_RECORDS = "no-plan-scheduler-records"
_ABSENCE_REASONS = frozenset(
    {ABSENCE_FLOW_WITHOUT_PLAN_REVIEW, ABSENCE_NO_PLAN_SCHEDULER_RECORDS}
)

EVIDENCE_PR_ROUND_METADATA = "pr-round-metadata"
EVIDENCE_OPERATOR_ASSERTED = "operator-asserted"

PHASE_PREPARED = "prepared"
PHASE_COMMITTED = "committed"
PHASE_ABORTED = "aborted"
_PHASES = frozenset({PHASE_PREPARED, PHASE_COMMITTED, PHASE_ABORTED})

ABORT_STALE_HEAD = "stale-head"
ABORT_SUPERSEDED_INTENT = "superseded-intent"
ABORT_SIBLING_CANONICAL = "sibling-canonical"
ABORT_REASONS = frozenset({ABORT_STALE_HEAD, ABORT_SUPERSEDED_INTENT, ABORT_SIBLING_CANONICAL})

STATUS_INHERITED = "inherited"
STATUS_NOT_APPLICABLE = "not-applicable"
STATUS_WAIVED_CODER_RESPONSE = "waived-coder-response-unavailable"
STATUS_UNPUBLISHED = "unpublished"
_OUTCOME_STATUSES = frozenset(
    {STATUS_INHERITED, STATUS_NOT_APPLICABLE, STATUS_WAIVED_CODER_RESPONSE, STATUS_UNPUBLISHED}
)

ERA_LEGACY = "legacy"
ERA_TRANSACTION = "transaction"

RECOVERY_RERUN = "rerun to finish the transaction"
RECOVERY_ORIGINAL_ACTOR = "rerun under the original GitHub actor"
RECOVERY_OPERATOR_REVIEW = "operator review of the divergent transaction records"
RECOVERY_RERUN_PLAN_REVIEW = (
    "rerun plan review for the approved candidate so a matching scheduler checkpoint is posted"
)

_HEX64_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_HEAD_SHA_RE = re.compile(r"\A(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_REPOSITORY_RE = re.compile(r"\A[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\Z")
_PLAN_HASH_RE = re.compile(r"\A[0-9a-f]{16}\Z")

CommentOrder = Literal["later", "earlier", "same", "unordered"]


def _fail(message: str) -> AgentLoopError:
    return AgentLoopError(f"Invalid workflow transaction intent: {message}")


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _exact_keys(payload: object, keys: set[str], *, what: str) -> dict[str, object]:
    if not isinstance(payload, dict) or set(payload) != keys:
        raise _fail(f"{what} must carry exactly {', '.join(sorted(keys))}.")
    return payload


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def comment_order(first: object, second: object) -> CommentOrder:
    """Order ``second`` relative to ``first`` under the one shared rule.

    Issue and PR conversation comments share one numeric ID sequence.
    ``second`` is later only when its comment ID is strictly greater and its
    immutable ``created_at`` is not earlier; GitHub timestamps have second
    precision, so a same-second pair is ordered by ID.  Equal IDs are the same
    comment.  A missing or unparseable value, or a timestamp reversal, is
    ``unordered``, which every caller must treat as failure.
    """
    ids: list[int] = []
    stamps = []
    for comment in (first, second):
        comment_id = getattr(comment, "comment_id", None)
        if not _positive_int(comment_id):
            return "unordered"
        try:
            stamps.append(parse_comment_timestamp(getattr(comment, "created_at", None)))
        except AgentLoopError:
            return "unordered"
        ids.append(comment_id)
    if ids[0] == ids[1]:
        return "same"
    if ids[1] > ids[0] and stamps[1] >= stamps[0]:
        return "later"
    if ids[1] < ids[0] and stamps[1] <= stamps[0]:
        return "earlier"
    return "unordered"


def is_strictly_later(first: object, second: object) -> bool:
    return comment_order(first, second) == "later"


# ---------------------------------------------------------------------------
# Typed references and intent components
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommentRef:
    """An authenticated reference to a record that existed before the intent.

    Its comment ID is immutable and is part of the hashed intent.
    """

    surface: str
    comment_id: int
    digest: str

    def __post_init__(self) -> None:
        parse_comment_thread_surface(self.surface)
        if not _positive_int(self.comment_id):
            raise _fail("a comment reference needs a positive numeric comment ID.")
        if not isinstance(self.digest, str) or _HEX64_RE.match(self.digest) is None:
            raise _fail("a comment reference needs a SHA-256 digest.")

    def to_payload(self) -> dict[str, object]:
        return {"surface": self.surface, "comment_id": self.comment_id, "digest": self.digest}

    @classmethod
    def from_payload(cls, payload: object) -> "CommentRef":
        data = _exact_keys(payload, {"surface", "comment_id", "digest"}, what="comment reference")
        return cls(data["surface"], data["comment_id"], data["digest"])  # type: ignore[arg-type]


@dataclass(frozen=True)
class SchedulerCheckpointRef:
    """Either the approved candidate's scheduler checkpoint or why none exists."""

    reference: CommentRef | None = None
    absence_reason: str | None = None

    def __post_init__(self) -> None:
        if (self.reference is None) == (self.absence_reason is None):
            raise _fail(
                "the scheduler checkpoint needs exactly one of a reference or an absence reason."
            )
        if self.reference is not None:
            if not isinstance(self.reference, CommentRef):
                raise _fail("the scheduler checkpoint reference must be a comment reference.")
            if parse_comment_thread_surface(self.reference.surface)[0] != ISSUE_THREAD_SURFACE:
                raise _fail("the scheduler checkpoint may only name an issue-side record.")
        elif self.absence_reason not in _ABSENCE_REASONS:
            raise _fail(f"unknown scheduler checkpoint absence reason {self.absence_reason!r}.")

    def to_payload(self) -> dict[str, object]:
        return {
            "reference": self.reference.to_payload() if self.reference else None,
            "absence_reason": self.absence_reason,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "SchedulerCheckpointRef":
        data = _exact_keys(payload, {"reference", "absence_reason"}, what="scheduler checkpoint")
        reference = data["reference"]
        return cls(
            CommentRef.from_payload(reference) if reference is not None else None,
            data["absence_reason"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class OriginEvidence:
    """Proof that a legacy PR was produced under the named approved plan."""

    kind: str
    plan_hash: str
    plan_subject: str
    reference: CommentRef | None = None
    operator_login: str | None = None
    operator_id: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan_hash, str) or _PLAN_HASH_RE.match(self.plan_hash) is None:
            raise _fail("origin evidence needs an approved-plan hash.")
        if not isinstance(self.plan_subject, str) or _HEX64_RE.match(self.plan_subject) is None:
            raise _fail("origin evidence needs an approved-plan subject.")
        if self.kind == EVIDENCE_PR_ROUND_METADATA:
            if not isinstance(self.reference, CommentRef):
                raise _fail("pr-round-metadata origin evidence needs a comment reference.")
            if parse_comment_thread_surface(self.reference.surface)[0] != PR_THREAD_SURFACE:
                raise _fail("pr-round-metadata origin evidence must name a PR-side record.")
            if self.operator_login is not None or self.operator_id is not None:
                raise _fail("pr-round-metadata origin evidence carries no operator identity.")
        elif self.kind == EVIDENCE_OPERATOR_ASSERTED:
            if self.reference is not None:
                raise _fail("operator-asserted origin evidence carries no comment reference.")
            if (
                not isinstance(self.operator_login, str)
                or not self.operator_login
                or not _positive_int(self.operator_id)
            ):
                raise _fail("operator-asserted origin evidence needs the operator actor.")
        else:
            raise _fail(f"unknown origin evidence kind {self.kind!r}.")

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "plan_hash": self.plan_hash,
            "plan_subject": self.plan_subject,
            "reference": self.reference.to_payload() if self.reference else None,
            "operator_login": self.operator_login,
            "operator_id": self.operator_id,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "OriginEvidence":
        data = _exact_keys(
            payload,
            {"kind", "plan_hash", "plan_subject", "reference", "operator_login", "operator_id"},
            what="origin evidence",
        )
        reference = data["reference"]
        return cls(
            kind=data["kind"],  # type: ignore[arg-type]
            plan_hash=data["plan_hash"],  # type: ignore[arg-type]
            plan_subject=data["plan_subject"],  # type: ignore[arg-type]
            reference=CommentRef.from_payload(reference) if reference is not None else None,
            operator_login=data["operator_login"],  # type: ignore[arg-type]
            operator_id=data["operator_id"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class LegacyRoot:
    """The authenticated version-1 state a ``legacy-root-correction`` repairs."""

    contract: CommentRef
    handoff: CommentRef | None
    origin_evidence: OriginEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.contract, CommentRef):
            raise _fail("a legacy root must name the version-1 PR contract it corrects.")
        if self.handoff is not None and not isinstance(self.handoff, CommentRef):
            raise _fail("a legacy root handoff must be a comment reference or absent.")
        if not isinstance(self.origin_evidence, OriginEvidence):
            raise _fail("a legacy root requires origin evidence.")

    def to_payload(self) -> dict[str, object]:
        return {
            "contract": self.contract.to_payload(),
            "handoff": self.handoff.to_payload() if self.handoff else "absent",
            "origin_evidence": self.origin_evidence.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "LegacyRoot":
        data = _exact_keys(payload, {"contract", "handoff", "origin_evidence"}, what="legacy root")
        handoff = data["handoff"]
        if handoff is None:
            raise _fail("a legacy root handoff must be a reference or the explicit `absent`.")
        return cls(
            contract=CommentRef.from_payload(data["contract"]),
            handoff=None if handoff == "absent" else CommentRef.from_payload(handoff),
            origin_evidence=OriginEvidence.from_payload(data["origin_evidence"]),
        )


@dataclass(frozen=True)
class StagedIdentity:
    """Staged parent/child identity and which of the two owns the approved plan."""

    parent_issue: int
    child_issue: int
    plan_owner: str  # "parent" for a direct-implementation child, else "child"

    def __post_init__(self) -> None:
        if not _positive_int(self.parent_issue) or not _positive_int(self.child_issue):
            raise _fail("staged identity needs positive parent and child issue numbers.")
        if self.parent_issue == self.child_issue:
            raise _fail("staged parent and child must be different issues.")
        if self.plan_owner not in {"parent", "child"}:
            raise _fail("staged identity plan_owner must be `parent` or `child`.")

    def to_payload(self) -> dict[str, object]:
        return {
            "parent_issue": self.parent_issue,
            "child_issue": self.child_issue,
            "plan_owner": self.plan_owner,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "StagedIdentity":
        data = _exact_keys(
            payload, {"parent_issue", "child_issue", "plan_owner"}, what="staged identity"
        )
        return cls(data["parent_issue"], data["child_issue"], data["plan_owner"])  # type: ignore[arg-type]


@dataclass(frozen=True)
class RecordSetEntry:
    """One required record and how this transaction supplies it.

    A ``reissued`` entry carries no comment ID: the comment this transaction
    publishes is a publication result and lives in the terminal record.  An
    ``inherited`` entry names the predecessor chain's existing comment.
    """

    name: str
    disposition: str
    inherited: CommentRef | None = None

    def __post_init__(self) -> None:
        if self.name not in RECORD_SET_ENTRY_NAMES:
            raise _fail(f"unknown record-set entry {self.name!r}.")
        if self.disposition not in _DISPOSITIONS:
            raise _fail(f"unknown record-set disposition {self.disposition!r}.")
        if (self.disposition == DISPOSITION_INHERITED) != (self.inherited is not None):
            raise _fail(
                f"record-set entry {self.name} must carry a predecessor reference exactly "
                "when it is inherited."
            )
        if self.inherited is not None and not isinstance(self.inherited, CommentRef):
            raise _fail("an inherited record-set entry needs a comment reference.")

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "disposition": self.disposition,
            "inherited": self.inherited.to_payload() if self.inherited else None,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "RecordSetEntry":
        data = _exact_keys(payload, {"name", "disposition", "inherited"}, what="record-set entry")
        inherited = data["inherited"]
        return cls(
            data["name"],  # type: ignore[arg-type]
            data["disposition"],  # type: ignore[arg-type]
            CommentRef.from_payload(inherited) if inherited is not None else None,
        )


def reissued(name: str) -> RecordSetEntry:
    return RecordSetEntry(name, DISPOSITION_REISSUED)


def not_applicable(name: str) -> RecordSetEntry:
    return RecordSetEntry(name, DISPOSITION_NOT_APPLICABLE)


def inherited(name: str, reference: CommentRef) -> RecordSetEntry:
    return RecordSetEntry(name, DISPOSITION_INHERITED, reference)


# Which entries each successor kind must reissue and which it must not.
_KIND_MUST_REISSUE: dict[str, frozenset[str]] = {
    KIND_LEGACY_ROOT_CORRECTION: frozenset({ENTRY_HANDOFF, ENTRY_PR_CONTRACT}),
    KIND_PLAN_REPLACEMENT: frozenset({ENTRY_HANDOFF}),
    KIND_FLOW_CORRECTION: frozenset({ENTRY_HANDOFF, ENTRY_PR_CONTRACT}),
    KIND_CLOSING_WIDENING: frozenset({ENTRY_HANDOFF, ENTRY_PR_CONTRACT}),
    KIND_MANAGED_CI_CONTINUITY: frozenset({ENTRY_AUTHORIZATION}),
}
_KIND_MUST_NOT_REISSUE: dict[str, frozenset[str]] = {
    KIND_HEAD_ADVANCE: frozenset({ENTRY_HANDOFF, ENTRY_PR_CONTRACT}),
    KIND_MANAGED_CI_CONTINUITY: frozenset({ENTRY_HANDOFF, ENTRY_PR_CONTRACT}),
}


@dataclass(frozen=True)
class WorkflowTransition:
    """The immutable intent of one workflow transition; its hash is the transaction ID.

    No field can carry a publication-result comment ID, phase, status, or
    writer identity: those live only in the transaction record.
    """

    repository: str
    primary_issue: int | None
    pr_number: int
    base: str
    head_sha: str
    origin_flow: str
    approved_plan_hash: str | None
    expected_closing_issue_ids: tuple[int, ...]
    scheduler_checkpoint: SchedulerCheckpointRef
    record_set: tuple[RecordSetEntry, ...]
    successor_kind: str = KIND_INITIAL
    predecessor_transaction_id: str | None = None
    legacy_root: LegacyRoot | None = None
    staged: StagedIdentity | None = None
    managed_ci_generation: str | None = None
    schema_version: int = INTENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise _fail("schema_version must be an integer.")
        if self.schema_version < 1:
            raise _fail("schema_version must be positive.")
        if not isinstance(self.repository, str) or _REPOSITORY_RE.match(self.repository) is None:
            raise _fail("repository must be OWNER/NAME.")
        if not _positive_int(self.pr_number):
            raise _fail("pr_number must be positive.")
        if self.primary_issue is not None and not _positive_int(self.primary_issue):
            raise _fail("primary_issue must be positive.")
        if not isinstance(self.base, str) or not self.base or self.base != self.base.strip():
            raise _fail("base must be a branch name.")
        if not isinstance(self.head_sha, str) or _HEAD_SHA_RE.match(self.head_sha) is None:
            raise _fail("head_sha must be a full lowercase commit SHA.")
        if self.origin_flow not in ORIGIN_FLOWS:
            raise _fail(f"unknown origin flow {self.origin_flow!r}.")
        if self.origin_flow in _ISSUE_ORIGIN_FLOWS and self.primary_issue is None:
            raise _fail(f"{self.origin_flow} requires a primary issue.")
        if self.origin_flow == FLOW_APPROVED_PLAN:
            if (
                not isinstance(self.approved_plan_hash, str)
                or _PLAN_HASH_RE.match(self.approved_plan_hash) is None
            ):
                raise _fail("approved-plan-implementation requires an approved-plan hash.")
        elif self.approved_plan_hash is not None:
            raise _fail(f"{self.origin_flow} must not carry an approved-plan hash.")
        ids = normalize_issue_ids(
            self.expected_closing_issue_ids, field_name="expected_closing_issue_ids"
        )
        assert ids is not None
        object.__setattr__(self, "expected_closing_issue_ids", ids)
        if self.primary_issue is not None and self.primary_issue not in ids:
            raise _fail(f"the closing contract must retain primary issue #{self.primary_issue}.")
        if self.staged is not None:
            if not isinstance(self.staged, StagedIdentity):
                raise _fail("staged must be a staged identity.")
            if self.staged.child_issue != self.primary_issue:
                raise _fail("the staged child must be the primary issue.")
            if self.staged.parent_issue in ids:
                raise _fail("the staged parent must not be part of the closing contract.")
        if self.managed_ci_generation is not None and (
            not isinstance(self.managed_ci_generation, str) or not self.managed_ci_generation
        ):
            raise _fail("managed_ci_generation must be a non-empty string when present.")
        self._validate_lineage_shape()
        self._validate_scheduler_checkpoint()
        self._validate_record_set()

    # -- validation -------------------------------------------------------

    def _validate_lineage_shape(self) -> None:
        kind = self.successor_kind
        if kind not in SUCCESSOR_KINDS:
            raise _fail(f"unknown successor kind {kind!r}.")
        predecessor = self.predecessor_transaction_id
        if predecessor is not None and (
            not isinstance(predecessor, str) or _HEX64_RE.match(predecessor) is None
        ):
            raise _fail("predecessor_transaction_id must be a transaction ID.")
        if kind == KIND_INITIAL:
            if predecessor is not None or self.legacy_root is not None:
                raise _fail("an initial transaction has neither predecessor nor legacy root.")
        elif kind == KIND_LEGACY_ROOT_CORRECTION:
            if predecessor is not None or not isinstance(self.legacy_root, LegacyRoot):
                raise _fail(
                    "a legacy-root-correction has a legacy root and no predecessor."
                )
            self._validate_legacy_root(self.legacy_root)
        elif predecessor is None or self.legacy_root is not None:
            raise _fail(f"a {kind} transaction has a predecessor and no legacy root.")

    def _validate_legacy_root(self, root: LegacyRoot) -> None:
        pr_surface = comment_thread_surface(PR_THREAD_SURFACE, self.pr_number)
        if self.origin_flow != FLOW_APPROVED_PLAN or self.primary_issue is None:
            raise _fail("a legacy root corrects a PR to the approved-plan flow.")
        if root.contract.surface != pr_surface:
            raise _fail("the legacy-root contract reference must name this PR.")
        if root.handoff is not None and root.handoff.surface != comment_thread_surface(
            ISSUE_THREAD_SURFACE, self.primary_issue
        ):
            raise _fail("the legacy-root handoff reference must name the primary issue.")
        evidence = root.origin_evidence
        if evidence.reference is not None and evidence.reference.surface != pr_surface:
            raise _fail("pr-round-metadata origin evidence must name this PR.")
        if evidence.plan_hash != self.approved_plan_hash:
            raise _fail("the origin-evidence plan hash must equal the approved-plan hash.")

    def _validate_scheduler_checkpoint(self) -> None:
        checkpoint = self.scheduler_checkpoint
        if not isinstance(checkpoint, SchedulerCheckpointRef):
            raise _fail("a scheduler checkpoint reference or absence reason is required.")
        if self.origin_flow != FLOW_APPROVED_PLAN:
            if checkpoint.absence_reason != ABSENCE_FLOW_WITHOUT_PLAN_REVIEW:
                raise _fail(
                    f"{self.origin_flow} has no plan review; its scheduler checkpoint must be "
                    f"`{ABSENCE_FLOW_WITHOUT_PLAN_REVIEW}`."
                )
            return
        if checkpoint.absence_reason == ABSENCE_FLOW_WITHOUT_PLAN_REVIEW:
            raise _fail("an approved-plan flow cannot claim it had no plan review.")
        if checkpoint.reference is not None and checkpoint.reference.surface != (
            comment_thread_surface(ISSUE_THREAD_SURFACE, self.plan_owning_issue)
        ):
            raise _fail("the scheduler checkpoint must live on the plan-owning issue.")

    def _validate_record_set(self) -> None:
        entries = self.record_set
        if not isinstance(entries, tuple) or not all(
            isinstance(entry, RecordSetEntry) for entry in entries
        ):
            raise _fail("record_set must be a tuple of record-set entries.")
        if tuple(entry.name for entry in entries) != RECORD_SET_ENTRY_NAMES:
            raise _fail(
                "record_set must declare exactly " + ", ".join(RECORD_SET_ENTRY_NAMES) + " in order."
            )
        kind = self.successor_kind
        rootless = kind in {KIND_INITIAL, KIND_LEGACY_ROOT_CORRECTION}
        pr_surface = comment_thread_surface(PR_THREAD_SURFACE, self.pr_number)
        for entry in entries:
            if entry.inherited is not None:
                if rootless:
                    raise _fail(f"a {kind} transaction cannot inherit {entry.name}.")
                expected_surface = (
                    comment_thread_surface(ISSUE_THREAD_SURFACE, self.primary_issue)
                    if entry.name == ENTRY_HANDOFF and self.primary_issue is not None
                    else pr_surface
                )
                if entry.inherited.surface != expected_surface:
                    raise _fail(f"inherited {entry.name} names the wrong comment thread.")
            has_entry = entry.name != ENTRY_HANDOFF or self.origin_flow in _ISSUE_ORIGIN_FLOWS
            if (
                has_entry
                and entry.name in _KIND_MUST_REISSUE.get(kind, frozenset())
                and entry.disposition != DISPOSITION_REISSUED
            ):
                raise _fail(f"a {kind} transaction must reissue {entry.name}.")
            if entry.name in _KIND_MUST_NOT_REISSUE.get(kind, frozenset()) and (
                entry.disposition == DISPOSITION_REISSUED
            ):
                raise _fail(f"a {kind} transaction must not reissue {entry.name}.")
        if self.origin_flow not in _ISSUE_ORIGIN_FLOWS and (
            self.entry(ENTRY_HANDOFF).disposition != DISPOSITION_NOT_APPLICABLE
        ):
            raise _fail(f"{self.origin_flow} has no issue-to-PR handoff.")
        if self.origin_flow in _ISSUE_ORIGIN_FLOWS and (
            self.entry(ENTRY_HANDOFF).disposition == DISPOSITION_NOT_APPLICABLE
        ):
            # Without it a committed transaction would leave no discoverable
            # issue-to-PR record.
            raise _fail(f"{self.origin_flow} requires the issue-to-PR handoff entry.")
        if self.staged is None and (
            self.entry(ENTRY_PR_CONTRACT).disposition == DISPOSITION_NOT_APPLICABLE
        ):
            # Only a staged child may be handoff-only on the PR side.
            raise _fail("a non-staged transition requires the PR expected-closing contract entry.")
        authorization = self.entry(ENTRY_AUTHORIZATION)
        if self.managed_ci_generation is None:
            if authorization.disposition != DISPOSITION_NOT_APPLICABLE:
                raise _fail("an unmanaged transition has no managed-CI authorization entry.")
        elif authorization.disposition == DISPOSITION_NOT_APPLICABLE:
            raise _fail("a managed transition must declare its managed-CI authorization entry.")
        elif kind == KIND_HEAD_ADVANCE and authorization.disposition != DISPOSITION_REISSUED:
            # The bound authorization is head-bound.
            raise _fail("a managed head-advance must reissue the managed-CI authorization.")
        if not rootless and (
            self.entry(ENTRY_INITIAL_CODER_ROUND).disposition != DISPOSITION_NOT_APPLICABLE
        ):
            raise _fail("only a first transaction can track the initial coder round.")
        if self.entry(ENTRY_INITIAL_CODER_ROUND).disposition == DISPOSITION_INHERITED:
            raise _fail("the initial coder round is never inherited.")

    # -- accessors --------------------------------------------------------

    def entry(self, name: str) -> RecordSetEntry:
        for item in self.record_set:
            if item.name == name:
                return item
        raise _fail(f"record set has no {name} entry.")

    @property
    def plan_owning_issue(self) -> int:
        """The issue whose conversation holds the approved plan and its checkpoint."""
        if self.staged is not None and self.staged.plan_owner == "parent":
            return self.staged.parent_issue
        if self.primary_issue is None:
            raise _fail("this transition has no plan-owning issue.")
        return self.primary_issue

    @property
    def scope(self) -> tuple[str, int | None, int]:
        return (self.repository.casefold(), self.primary_issue, self.pr_number)

    @property
    def pr_url(self) -> str:
        return f"https://github.com/{self.repository}/pull/{self.pr_number}"

    @property
    def expected_record_set(self) -> tuple[str, ...]:
        return tuple(f"{entry.name} ({entry.disposition})" for entry in self.record_set)

    # -- canonical serialization -----------------------------------------

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "repository": self.repository,
            "primary_issue": self.primary_issue,
            "pr_number": self.pr_number,
            "base": self.base,
            "head_sha": self.head_sha,
            "origin_flow": self.origin_flow,
            "approved_plan_hash": self.approved_plan_hash,
            "expected_closing_issue_ids": list(self.expected_closing_issue_ids),
            "staged": self.staged.to_payload() if self.staged else None,
            "managed_ci_generation": self.managed_ci_generation,
            "scheduler_checkpoint": self.scheduler_checkpoint.to_payload(),
            "record_set": [entry.to_payload() for entry in self.record_set],
            "predecessor_transaction_id": self.predecessor_transaction_id,
            "legacy_root": self.legacy_root.to_payload() if self.legacy_root else None,
            "successor_kind": self.successor_kind,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "WorkflowTransition":
        keys = {
            "schema_version", "repository", "primary_issue", "pr_number", "base", "head_sha",
            "origin_flow", "approved_plan_hash", "expected_closing_issue_ids", "staged",
            "managed_ci_generation", "scheduler_checkpoint", "record_set",
            "predecessor_transaction_id", "legacy_root", "successor_kind",
        }
        data = _exact_keys(payload, keys, what="intent")
        if data["schema_version"] != INTENT_SCHEMA_VERSION or isinstance(
            data["schema_version"], bool
        ):
            raise _fail(f"unsupported intent schema_version {data['schema_version']!r}.")
        raw_ids = data["expected_closing_issue_ids"]
        if not isinstance(raw_ids, list):
            raise _fail("expected_closing_issue_ids must be a list.")
        raw_entries = data["record_set"]
        if not isinstance(raw_entries, list):
            raise _fail("record_set must be a list.")
        intent = cls(
            repository=data["repository"],  # type: ignore[arg-type]
            primary_issue=data["primary_issue"],  # type: ignore[arg-type]
            pr_number=data["pr_number"],  # type: ignore[arg-type]
            base=data["base"],  # type: ignore[arg-type]
            head_sha=data["head_sha"],  # type: ignore[arg-type]
            origin_flow=data["origin_flow"],  # type: ignore[arg-type]
            approved_plan_hash=data["approved_plan_hash"],  # type: ignore[arg-type]
            expected_closing_issue_ids=tuple(raw_ids),
            scheduler_checkpoint=SchedulerCheckpointRef.from_payload(data["scheduler_checkpoint"]),
            record_set=tuple(RecordSetEntry.from_payload(item) for item in raw_entries),
            successor_kind=data["successor_kind"],  # type: ignore[arg-type]
            predecessor_transaction_id=data["predecessor_transaction_id"],  # type: ignore[arg-type]
            legacy_root=(
                LegacyRoot.from_payload(data["legacy_root"])
                if data["legacy_root"] is not None
                else None
            ),
            staged=(
                StagedIdentity.from_payload(data["staged"]) if data["staged"] is not None else None
            ),
            managed_ci_generation=data["managed_ci_generation"],  # type: ignore[arg-type]
        )
        if intent.to_payload() != data:
            raise _fail("intent payload is not canonical.")
        return intent

    @property
    def transaction_id(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_payload())).hexdigest()


def transaction_id(intent: WorkflowTransition) -> str:
    return intent.transaction_id


# ---------------------------------------------------------------------------
# Derivation: both per-surface payloads come from the one intent
# ---------------------------------------------------------------------------


def derive_handoff_metadata(intent: WorkflowTransition) -> IssuePrHandoffMetadataV2:
    """Derive the version-2 issue-side handoff.  No caller supplies a flow string."""
    if intent.origin_flow not in _ISSUE_ORIGIN_FLOWS or intent.primary_issue is None:
        raise _fail(f"{intent.origin_flow} has no issue-to-PR handoff.")
    metadata = IssuePrHandoffMetadataV2(
        issue_number=intent.primary_issue,
        pr_number=intent.pr_number,
        pr_url=intent.pr_url,
        pr_head_sha=intent.head_sha,
        flow=intent.origin_flow,
        plan_hash=intent.approved_plan_hash,
        expected_closing_issue_ids=intent.expected_closing_issue_ids,
        contract_hash=contract_hash(intent.expected_closing_issue_ids),
        transaction_id=intent.transaction_id,
    )
    return decode_issue_pr_handoff_v2(encode_issue_pr_handoff_v2(metadata))


def derive_pr_contract(
    intent: WorkflowTransition,
    *,
    supersession_kind: str | None = None,
    supersedes_record_hash: str | None = None,
) -> PrExpectedClosingContractV2:
    """Derive the version-2 PR-side contract.  No caller supplies a flow string."""
    return make_pr_contract_v2(
        repository=intent.repository,
        pr_number=intent.pr_number,
        origin_flow=intent.origin_flow,
        expected_closing_issue_ids=intent.expected_closing_issue_ids,
        transaction_id=intent.transaction_id,
        primary_issue_number=intent.primary_issue,
        supersession_kind=supersession_kind,
        supersedes_record_hash=supersedes_record_hash,
    )


# ---------------------------------------------------------------------------
# Transaction record codec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntryOutcome:
    """The publication result of one record-set entry, stored only in a terminal record."""

    name: str
    comment_id: int | None = None
    status: str | None = None

    def __post_init__(self) -> None:
        if self.name not in RECORD_SET_ENTRY_NAMES:
            raise AgentLoopError(f"Unknown workflow transaction entry {self.name!r}.")
        if (self.comment_id is None) == (self.status is None):
            raise AgentLoopError(
                f"Workflow transaction entry {self.name} needs exactly one of a comment ID "
                "or a terminal status."
            )
        if self.comment_id is not None and not _positive_int(self.comment_id):
            raise AgentLoopError(f"Workflow transaction entry {self.name} comment ID is invalid.")
        if self.status is not None and self.status not in _OUTCOME_STATUSES:
            raise AgentLoopError(
                f"Workflow transaction entry {self.name} has unknown status {self.status!r}."
            )

    def to_payload(self) -> dict[str, object]:
        return {"name": self.name, "comment_id": self.comment_id, "status": self.status}


@dataclass(frozen=True)
class WorkflowTransactionRecord:
    """One append-only transaction record: ``prepared``, ``committed``, or ``aborted``."""

    phase: str
    transaction_id: str
    intent: WorkflowTransition | None = None
    writer_login: str | None = None
    writer_id: int | None = None
    prepared_comment_id: int | None = None
    outcomes: tuple[EntryOutcome, ...] = ()
    abort_reason: str | None = None
    differing_fields: tuple[tuple[str, str, str], ...] = ()
    schema_version: int = RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        prefix = f"Invalid {WORKFLOW_TRANSACTION_MARKER} record"
        if self.schema_version != RECORD_SCHEMA_VERSION or isinstance(self.schema_version, bool):
            raise AgentLoopError(f"{prefix}: unsupported schema_version.")
        if self.phase not in _PHASES:
            raise AgentLoopError(f"{prefix}: unknown phase {self.phase!r}.")
        if not isinstance(self.transaction_id, str) or _HEX64_RE.match(self.transaction_id) is None:
            raise AgentLoopError(f"{prefix}: transaction_id is invalid.")
        if self.phase == PHASE_PREPARED:
            if not isinstance(self.intent, WorkflowTransition):
                raise AgentLoopError(f"{prefix}: a prepared record stores the complete intent.")
            if self.intent.transaction_id != self.transaction_id:
                raise AgentLoopError(
                    f"{prefix}: the stored intent does not hash to transaction "
                    f"{self.transaction_id}."
                )
            if (
                not isinstance(self.writer_login, str)
                or not self.writer_login
                or not _positive_int(self.writer_id)
            ):
                raise AgentLoopError(f"{prefix}: a prepared record names its writer.")
            if self.prepared_comment_id is not None or self.outcomes or self.abort_reason:
                raise AgentLoopError(f"{prefix}: a prepared record carries no terminal fields.")
            return
        if self.intent is not None or self.writer_login is not None or self.writer_id is not None:
            raise AgentLoopError(f"{prefix}: a terminal record carries no intent or writer.")
        if not _positive_int(self.prepared_comment_id):
            raise AgentLoopError(f"{prefix}: a terminal record binds its prepared comment ID.")
        if tuple(item.name for item in self.outcomes) != RECORD_SET_ENTRY_NAMES:
            raise AgentLoopError(
                f"{prefix}: a terminal record states an outcome for every record-set entry."
            )
        if self.phase == PHASE_COMMITTED:
            if self.abort_reason is not None or self.differing_fields:
                raise AgentLoopError(f"{prefix}: a committed record carries no abort fields.")
            if any(item.status == STATUS_UNPUBLISHED for item in self.outcomes):
                raise AgentLoopError(
                    f"{prefix}: a committed record cannot leave an entry unpublished."
                )
        else:
            if self.abort_reason not in ABORT_REASONS:
                raise AgentLoopError(f"{prefix}: unknown abort reason {self.abort_reason!r}.")
            for item in self.differing_fields:
                if len(item) != 3 or not all(isinstance(part, str) for part in item):
                    raise AgentLoopError(f"{prefix}: differing fields are malformed.")

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "phase": self.phase,
            "transaction_id": self.transaction_id,
        }
        if self.phase == PHASE_PREPARED:
            assert self.intent is not None
            payload["intent"] = self.intent.to_payload()
            payload["writer"] = {"login": self.writer_login, "id": self.writer_id}
            return payload
        payload["prepared_comment_id"] = self.prepared_comment_id
        payload["entries"] = [item.to_payload() for item in self.outcomes]
        if self.phase == PHASE_ABORTED:
            payload["abort_reason"] = self.abort_reason
            payload["differing_fields"] = [
                {"field": name, "stored": stored, "fresh": fresh}
                for name, stored, fresh in self.differing_fields
            ]
        return payload


def prepared_record(
    intent: WorkflowTransition, *, writer_login: str, writer_id: int
) -> WorkflowTransactionRecord:
    return WorkflowTransactionRecord(
        phase=PHASE_PREPARED,
        transaction_id=intent.transaction_id,
        intent=intent,
        writer_login=writer_login,
        writer_id=writer_id,
    )


def encode_transaction_record(record: WorkflowTransactionRecord) -> str:
    return base64.urlsafe_b64encode(_canonical_json(record.to_payload())).decode("ascii")


def decode_transaction_record(encoded: str) -> WorkflowTransactionRecord:
    prefix = f"Invalid {WORKFLOW_TRANSACTION_MARKER} record"
    try:
        payload = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise AgentLoopError(f"{prefix}: payload is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise AgentLoopError(f"{prefix}: payload must be an object.")
    phase = payload.get("phase")
    common = {"schema_version", "phase", "transaction_id"}
    expected = {
        PHASE_PREPARED: common | {"intent", "writer"},
        PHASE_COMMITTED: common | {"prepared_comment_id", "entries"},
        PHASE_ABORTED: common
        | {"prepared_comment_id", "entries", "abort_reason", "differing_fields"},
    }.get(phase)  # type: ignore[arg-type]
    if expected is None or set(payload) != expected:
        raise AgentLoopError(f"{prefix}: unexpected fields for phase {phase!r}.")
    if phase == PHASE_PREPARED:
        writer = payload["writer"]
        if not isinstance(writer, dict) or set(writer) != {"login", "id"}:
            raise AgentLoopError(f"{prefix}: writer must carry exactly login and id.")
        record = WorkflowTransactionRecord(
            phase=PHASE_PREPARED,
            transaction_id=payload["transaction_id"],
            intent=WorkflowTransition.from_payload(payload["intent"]),
            writer_login=writer["login"],
            writer_id=writer["id"],
            schema_version=payload["schema_version"],
        )
    else:
        raw_entries = payload["entries"]
        if not isinstance(raw_entries, list):
            raise AgentLoopError(f"{prefix}: entries must be a list.")
        outcomes = []
        for raw in raw_entries:
            if not isinstance(raw, dict) or set(raw) != {"name", "comment_id", "status"}:
                raise AgentLoopError(f"{prefix}: an entry outcome is malformed.")
            outcomes.append(EntryOutcome(raw["name"], raw["comment_id"], raw["status"]))
        differing: list[tuple[str, str, str]] = []
        for raw in payload.get("differing_fields", []) if phase == PHASE_ABORTED else []:
            if not isinstance(raw, dict) or set(raw) != {"field", "stored", "fresh"}:
                raise AgentLoopError(f"{prefix}: a differing field is malformed.")
            differing.append((raw["field"], raw["stored"], raw["fresh"]))
        if phase == PHASE_ABORTED and not isinstance(payload["differing_fields"], list):
            raise AgentLoopError(f"{prefix}: differing_fields must be a list.")
        record = WorkflowTransactionRecord(
            phase=phase,
            transaction_id=payload["transaction_id"],
            prepared_comment_id=payload["prepared_comment_id"],
            outcomes=tuple(outcomes),
            abort_reason=payload.get("abort_reason"),
            differing_fields=tuple(differing),
            schema_version=payload["schema_version"],
        )
    if encode_transaction_record(record) != encoded:
        raise AgentLoopError(f"{prefix}: payload is not canonically encoded.")
    return record


def format_transaction_record_comment(record: WorkflowTransactionRecord) -> TrustedBody:
    """Render one tool-owned transaction record comment for the PR conversation."""
    encoded = encode_transaction_record(record)
    decode_transaction_record(encoded)
    detail = {
        PHASE_PREPARED: "Prepared: the required records below are still being published.",
        PHASE_COMMITTED: "Committed: every required record was published and read back.",
        PHASE_ABORTED: f"Aborted ({record.abort_reason}): its records grant nothing.",
    }[record.phase]
    lines = [
        "Agent-loop workflow transaction record (machine-readable). "
        "Not an agent response; keep this comment.",
        "",
        f"Transaction: {record.transaction_id}",
        detail,
    ]
    if record.intent is not None:
        lines.append(f"Kind: {record.intent.successor_kind}")
        lines.append("Record set: " + ", ".join(record.intent.expected_record_set))
    lines.extend(
        [
            f"<!-- {WORKFLOW_TRANSACTION_MARKER}: {encoded} -->",
            "-- coding-review-agent-loop",
        ]
    )
    return TrustedBody.canonical(
        "\n".join(lines), expected_tokens=(WORKFLOW_TRANSACTION_MARKER,)
    )


# ---------------------------------------------------------------------------
# Envelope guards
# ---------------------------------------------------------------------------


def _require_view(view: object, *, kind: str, number: int | None = None) -> AuthenticatedCommentView:
    if not isinstance(view, AuthenticatedCommentView):
        raise AgentLoopError(
            "Workflow transaction resolution accepts only an authenticated comment view; "
            "an unauthenticated comment snapshot is never interpreted."
        )
    surface_kind, surface_number = parse_comment_thread_surface(view.surface)
    if surface_kind != kind or (number is not None and surface_number != number):
        expected = f"{kind}#{number}" if number is not None else kind
        raise AgentLoopError(
            f"Authenticated comment view {view.surface} is not the expected {expected} thread."
        )
    return view


def _short(value: str) -> str:
    return value[:12]


# ---------------------------------------------------------------------------
# Lineage resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransactionState:
    """Everything the authenticated PR conversation says about one transaction."""

    intent: WorkflowTransition
    prepared_comment: AuthenticatedComment
    prepared_duplicates: tuple[AuthenticatedComment, ...] = ()
    terminal: WorkflowTransactionRecord | None = None
    terminal_comment: AuthenticatedComment | None = None
    # The exact prepared comment the terminal record binds (it may be a
    # byte-identical duplicate of ``prepared_comment``).
    bound_prepared_comment: AuthenticatedComment | None = None

    @property
    def transaction_id(self) -> str:
        return self.intent.transaction_id

    @property
    def status(self) -> str:
        return self.terminal.phase if self.terminal is not None else PHASE_PREPARED

    @property
    def committed(self) -> bool:
        return self.status == PHASE_COMMITTED

    @property
    def aborted(self) -> bool:
        return self.status == PHASE_ABORTED

    def outcome(self, name: str) -> EntryOutcome | None:
        if self.terminal is None:
            return None
        return next(item for item in self.terminal.outcomes if item.name == name)


@dataclass(frozen=True)
class LegacyRootContext:
    """Authenticated inputs needed to re-validate a legacy root's origin evidence."""

    plan_issue_view: AuthenticatedCommentView
    pr_commit_shas: tuple[str, ...]
    primary_issue_view: AuthenticatedCommentView | None = None


@dataclass(frozen=True)
class TransactionLineage:
    repository: str
    pr_number: int
    surface: str
    transactions: tuple[TransactionState, ...]
    # Non-aborted transactions from the root to the tip.
    chain: tuple[TransactionState, ...]
    ignored_foreign: tuple[str, ...] = ()

    def state(self, tx_id: str) -> TransactionState | None:
        return next((item for item in self.transactions if item.transaction_id == tx_id), None)

    @property
    def latest_committed(self) -> TransactionState | None:
        return next((item for item in reversed(self.chain) if item.committed), None)

    @property
    def pending(self) -> TransactionState | None:
        """The non-terminal prepared transaction a rerun must adopt, if any."""
        tip = self.chain[-1] if self.chain else None
        return tip if tip is not None and not tip.committed else None


def _transaction_error(
    summary: str,
    *,
    states: Sequence[TransactionState] = (),
    transaction_ids: Sequence[str] = (),
    problems: Sequence[str] = (),
    recovery: str = RECOVERY_OPERATOR_REVIEW,
    code: str | None = None,
) -> WorkflowTransactionError:
    first = states[0].intent if states else None
    return WorkflowTransactionError(
        summary,
        transaction_ids=tuple(transaction_ids) or tuple(item.transaction_id for item in states),
        successor_kind=first.successor_kind if first else None,
        expected_record_set=first.expected_record_set if first else (),
        problems=tuple(problems),
        recovery_action=recovery,
        code=code,
    )


def _parse_transaction_records(
    comments: Sequence[AuthenticatedComment],
) -> list[tuple[AuthenticatedComment, WorkflowTransactionRecord]]:
    found: list[tuple[AuthenticatedComment, WorkflowTransactionRecord]] = []
    for comment in comments:
        for match in WORKFLOW_TRANSACTION_MARKER_RE.finditer(comment.body):
            found.append((comment, decode_transaction_record(match.group("payload"))))
    return found


def foreign_transaction_writers(pr_view: AuthenticatedCommentView) -> tuple[str, ...]:
    """Describe transaction records written by any actor other than the current one.

    They are never adopted and grant nothing; an actor change therefore fails
    closed with a diagnostic naming the prior writer.
    """
    _require_view(pr_view, kind=PR_THREAD_SURFACE)
    described: list[str] = []
    for comment in pr_view.ignored_foreign:
        for match in WORKFLOW_TRANSACTION_MARKER_RE.finditer(comment.body):
            try:
                record = decode_transaction_record(match.group("payload"))
                label = f"{record.phase} record of transaction {record.transaction_id}"
            except AgentLoopError:
                label = "malformed transaction record"
            described.append(
                f"ignored-foreign {label} in comment {comment.comment_id} written by "
                f"{comment.author_login} (user ID {comment.author_id})"
            )
    return tuple(described)


def actor_change_error(pr_view: AuthenticatedCommentView) -> WorkflowTransactionError | None:
    """The fail-closed diagnostic for a PR whose transactions another actor wrote."""
    foreign = foreign_transaction_writers(pr_view)
    if not foreign:
        return None
    return WorkflowTransactionError(
        f"Transaction records on {pr_view.surface} were written by another GitHub actor; "
        f"the current actor {pr_view.actor_login} (user ID {pr_view.actor_id}) neither "
        "adopts them nor gains authority from them",
        problems=foreign,
        recovery_action=RECOVERY_ORIGINAL_ACTOR,
        code="actor-change",
    )


def collect_transactions(
    pr_view: AuthenticatedCommentView, *, repository: str, pr_number: int
) -> tuple[TransactionState, ...]:
    """Group the authenticated PR conversation's records by transaction.

    Byte-identical duplicates from interleaved writers canonicalize to the
    earliest comment ID.  A terminal record without its authenticated prepared
    record, or contradictory terminal records, fail closed.
    """
    _require_view(pr_view, kind=PR_THREAD_SURFACE, number=pr_number)
    prepared: dict[str, list[tuple[AuthenticatedComment, WorkflowTransactionRecord]]] = {}
    terminals: dict[str, list[tuple[AuthenticatedComment, WorkflowTransactionRecord]]] = {}
    for comment, record in _parse_transaction_records(pr_view.authored):
        if record.phase == PHASE_PREPARED:
            assert record.intent is not None
            intent = record.intent
            if intent.repository.casefold() != repository.casefold() or intent.pr_number != pr_number:
                raise _transaction_error(
                    f"A prepared transaction record on {pr_view.surface} belongs to another PR",
                    transaction_ids=(record.transaction_id,),
                    problems=(f"contradictory prepared record in comment {comment.comment_id}",),
                )
            if record.writer_id != comment.author_id or record.writer_login != comment.author_login:
                raise _transaction_error(
                    "A prepared transaction record names a writer other than its comment author",
                    transaction_ids=(record.transaction_id,),
                    problems=(f"contradictory prepared record in comment {comment.comment_id}",),
                )
            prepared.setdefault(record.transaction_id, []).append((comment, record))
        else:
            terminals.setdefault(record.transaction_id, []).append((comment, record))
    orphaned = sorted(set(terminals) - set(prepared))
    if orphaned:
        raise _transaction_error(
            "A terminal transaction record has no authenticated prepared record",
            transaction_ids=orphaned,
            problems=tuple(
                f"missing prepared record for terminal comment {comment.comment_id}"
                for tx_id in orphaned
                for comment, _record in terminals[tx_id]
            ),
            code="terminal-without-prepared",
        )
    states: list[TransactionState] = []
    for tx_id, items in prepared.items():
        items.sort(key=lambda pair: pair[0].comment_id)
        canonical_comment, canonical_record = items[0]
        assert canonical_record.intent is not None
        prepared_ids = {comment.comment_id for comment, _record in items}
        terminal_record = terminal_comment = None
        ordered = sorted(terminals.get(tx_id, []), key=lambda pair: pair[0].comment_id)
        if ordered:
            terminal_comment, terminal_record = ordered[0]
            phases = {record.phase for _comment, record in ordered}
            if len(phases) != 1:
                raise _transaction_error(
                    "A transaction has both a committed and an aborted terminal record",
                    transaction_ids=(tx_id,),
                    problems=tuple(
                        f"contradictory {record.phase} record in comment {comment.comment_id}"
                        for comment, record in ordered
                    ),
                    code="contradictory-terminal",
                )
            divergent = [
                (comment, record) for comment, record in ordered if record != terminal_record
            ]
            if divergent:
                # Only byte-identical duplicates canonicalize; a transaction has
                # at most one terminal outcome.
                raise _transaction_error(
                    "A transaction has divergent terminal records of the same phase",
                    transaction_ids=(tx_id,),
                    problems=tuple(
                        f"contradictory {record.phase} record in comment {comment.comment_id}"
                        for comment, record in ordered
                    ),
                    code="contradictory-terminal",
                )
            for comment, record in ordered:
                if record.prepared_comment_id not in prepared_ids:
                    raise _transaction_error(
                        "A terminal transaction record binds a comment that is not its "
                        "prepared record",
                        transaction_ids=(tx_id,),
                        problems=(
                            f"contradictory terminal record in comment {comment.comment_id}",
                        ),
                        code="terminal-without-prepared",
                    )
                bound_prepared = next(
                    item for item, _record in items
                    if item.comment_id == record.prepared_comment_id
                )
                if not is_strictly_later(bound_prepared, comment):
                    raise _transaction_error(
                        "A terminal transaction record is not later than its prepared record",
                        transaction_ids=(tx_id,),
                        problems=(f"unordered terminal record in comment {comment.comment_id}",),
                    )
            _validate_outcomes(canonical_record.intent, terminal_record, terminal_comment)
        states.append(
            TransactionState(
                intent=canonical_record.intent,
                prepared_comment=canonical_comment,
                prepared_duplicates=tuple(comment for comment, _record in items[1:]),
                terminal=terminal_record,
                terminal_comment=terminal_comment,
                bound_prepared_comment=(
                    next(
                        item for item, _record in items
                        if item.comment_id == terminal_record.prepared_comment_id
                    )
                    if terminal_record is not None
                    else None
                ),
            )
        )
    states.sort(key=lambda item: item.prepared_comment.comment_id)
    return tuple(states)


def _validate_outcomes(
    intent: WorkflowTransition,
    terminal: WorkflowTransactionRecord,
    comment: AuthenticatedComment,
) -> None:
    for outcome in terminal.outcomes:
        entry = intent.entry(outcome.name)
        if entry.disposition == DISPOSITION_INHERITED:
            valid = outcome.status == STATUS_INHERITED
        elif entry.disposition == DISPOSITION_NOT_APPLICABLE:
            valid = outcome.status == STATUS_NOT_APPLICABLE
        else:
            valid = outcome.comment_id is not None or (
                outcome.status == STATUS_WAIVED_CODER_RESPONSE
                and outcome.name == ENTRY_INITIAL_CODER_ROUND
            ) or (outcome.status == STATUS_UNPUBLISHED and terminal.phase == PHASE_ABORTED)
        if not valid:
            raise WorkflowTransactionError(
                "A terminal transaction record contradicts its prepared record set",
                transaction_ids=(intent.transaction_id,),
                successor_kind=intent.successor_kind,
                expected_record_set=intent.expected_record_set,
                problems=(
                    f"contradictory outcome for {outcome.name} in comment {comment.comment_id}",
                ),
                recovery_action=RECOVERY_OPERATOR_REVIEW,
            )


def resolve_transaction_lineage(
    pr_view: AuthenticatedCommentView,
    *,
    repository: str,
    pr_number: int,
    legacy_root_context: LegacyRootContext | None = None,
) -> TransactionLineage:
    """Resolve the one canonical transaction chain of a PR, or fail closed.

    Two non-aborted transactions without successor linkage, two non-aborted
    children of one predecessor, a successor over an uncommitted or unknown
    predecessor, and a legacy root that does not re-validate all raise; the
    newest record is never picked silently.
    """
    states = collect_transactions(pr_view, repository=repository, pr_number=pr_number)
    by_id = {item.transaction_id: item for item in states}
    live = [item for item in states if not item.aborted]
    roots = [item for item in live if item.intent.predecessor_transaction_id is None]
    if len(roots) > 1:
        raise _transaction_error(
            f"Divergent same-scope transactions exist on {pr_view.surface} without "
            "successor linkage",
            states=roots,
            problems=tuple(
                f"contradictory {item.intent.successor_kind} transaction {item.transaction_id} "
                f"prepared in comment {item.prepared_comment.comment_id}"
                for item in roots
            ),
            code="divergent-transactions",
        )
    children: dict[str, list[TransactionState]] = {}
    for item in live:
        predecessor_id = item.intent.predecessor_transaction_id
        if predecessor_id is None:
            continue
        predecessor = by_id.get(predecessor_id)
        if predecessor is None or not predecessor.committed:
            raise _transaction_error(
                "A successor transaction names a predecessor that is not committed",
                states=(item,),
                problems=(
                    f"missing committed predecessor {predecessor_id} for transaction "
                    f"{item.transaction_id}",
                ),
                code="uncommitted-predecessor",
            )
        children.setdefault(predecessor_id, []).append(item)
    for predecessor_id, items in children.items():
        if len(items) > 1:
            raise _transaction_error(
                f"Transaction {predecessor_id} has more than one non-aborted successor",
                states=items,
                problems=tuple(
                    f"contradictory successor {item.transaction_id} prepared in comment "
                    f"{item.prepared_comment.comment_id}"
                    for item in items
                ),
                code="divergent-transactions",
            )
    chain: list[TransactionState] = []
    cursor = roots[0] if roots else None
    while cursor is not None:
        chain.append(cursor)
        successors = children.get(cursor.transaction_id, [])
        cursor = successors[0] if successors else None
    if len(chain) != len(live):
        raise _transaction_error(
            "Non-aborted transactions are not connected to one canonical root",
            states=[item for item in live if item not in chain],
            code="divergent-transactions",
        )
    _validate_chain(chain)
    for item in chain:
        if item.intent.legacy_root is None:
            continue
        if legacy_root_context is None:
            raise _transaction_error(
                "A legacy-root-correction transaction cannot be accepted without "
                "re-validating its origin evidence",
                states=(item,),
                code="legacy-root-unvalidated",
            )
        validate_legacy_root(
            item.intent,
            pr_view=pr_view,
            plan_issue_view=legacy_root_context.plan_issue_view,
            pr_commit_shas=legacy_root_context.pr_commit_shas,
            primary_issue_view=legacy_root_context.primary_issue_view,
        )
    return TransactionLineage(
        repository=repository,
        pr_number=pr_number,
        surface=pr_view.surface,
        transactions=states,
        chain=tuple(chain),
        ignored_foreign=foreign_transaction_writers(pr_view),
    )


def _validate_successor_delta(item: TransactionState, before: WorkflowTransition) -> None:
    """A successor's declared kind must be exactly what its delta implies."""
    intent = item.intent
    try:
        expected = successor_kind_for(before, TransitionInputs.of(intent))
    except AgentLoopError as exc:
        raise _transaction_error(
            "A successor transaction changes a field no successor kind may change",
            states=(item,),
            problems=(f"contradictory successor {item.transaction_id}: {exc}",),
            code="successor-kind-mismatch",
        ) from exc
    if expected is None:
        raise _transaction_error(
            "A successor transaction changes nothing relative to its predecessor",
            states=(item,),
            problems=(f"contradictory zero-delta successor {item.transaction_id}",),
            code="successor-kind-mismatch",
        )
    if expected != intent.successor_kind:
        raise _transaction_error(
            f"A successor transaction declares {intent.successor_kind} but its changes "
            f"relative to the predecessor require {expected}",
            states=(item,),
            problems=(f"contradictory successor {item.transaction_id}",),
            code="successor-kind-mismatch",
        )
    if tuple(before.expected_closing_issue_ids) != tuple(intent.expected_closing_issue_ids) and not (
        set(before.expected_closing_issue_ids) < set(intent.expected_closing_issue_ids)
    ):
        raise _transaction_error(
            "A successor transaction may only widen the closing contract to a strict superset",
            states=(item,),
            problems=(f"contradictory closing contract in successor {item.transaction_id}",),
            code="successor-kind-mismatch",
        )


def _validate_chain(chain: Sequence[TransactionState]) -> None:
    effective: dict[str, tuple[int, str | None] | None] = {
        name: None for name in RECORD_SET_ENTRY_NAMES
    }
    previous: TransactionState | None = None
    for item in chain:
        intent = item.intent
        if previous is not None:
            before = previous.intent
            if (
                intent.scope != before.scope
                or intent.base != before.base
                or intent.staged != before.staged
            ):
                raise _transaction_error(
                    "A successor transaction changes the scope of its predecessor",
                    states=(item,),
                    problems=(f"contradictory successor {item.transaction_id}",),
                )
            _validate_successor_delta(item, before)
            if (
                intent.successor_kind != KIND_PLAN_REPLACEMENT
                and intent.scheduler_checkpoint != before.scheduler_checkpoint
            ):
                raise _transaction_error(
                    "A successor transaction must inherit its predecessor's scheduler "
                    "checkpoint reference unchanged",
                    states=(item,),
                    problems=(f"contradictory successor {item.transaction_id}",),
                )
            assert previous.terminal_comment is not None
            if not is_strictly_later(previous.terminal_comment, item.prepared_comment):
                raise _transaction_error(
                    "A successor transaction was not prepared after its predecessor committed",
                    states=(item,),
                    problems=(f"unordered successor {item.transaction_id}",),
                )
        for entry in intent.record_set:
            if entry.inherited is None:
                continue
            current = effective[entry.name]
            if (
                current is None
                or current[0] != entry.inherited.comment_id
                or (current[1] is not None and current[1] != entry.inherited.digest)
            ):
                raise _transaction_error(
                    f"An inherited {entry.name} does not name the predecessor chain's "
                    "canonical record",
                    states=(item,),
                    problems=(
                        f"contradictory inherited {entry.name} reference to comment "
                        f"{entry.inherited.comment_id}",
                    ),
                    code="inherited-mismatch",
                )
        if item.committed:
            for entry in intent.record_set:
                outcome = item.outcome(entry.name)
                assert outcome is not None
                if entry.inherited is not None:
                    effective[entry.name] = (entry.inherited.comment_id, entry.inherited.digest)
                elif entry.disposition == DISPOSITION_REISSUED:
                    effective[entry.name] = (
                        (outcome.comment_id, None) if outcome.comment_id is not None else None
                    )
        previous = item


# ---------------------------------------------------------------------------
# Approved-plan anchor and scheduler checkpoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApprovedPlanAnchor:
    comment: AuthenticatedComment
    plan_hash: str
    plan_subject: str
    canonical_text: str


def _plan_records(
    issue_view: AuthenticatedCommentView,
) -> tuple[tuple[PostedRoundRecord, AuthenticatedComment], ...]:
    records = _extract_round_metadata_records(issue_view.authored, flow="plan")
    return tuple((record, issue_view.authored[record.index]) for record in records)


def _plan_text_candidates(record: PostedRoundRecord) -> tuple[str, ...]:
    # Mirrors recover_approved_plan_context so the anchor and the recovered
    # approved plan are always the same record family.
    raw = record.metadata.canonical_plan
    if raw is None:
        raw = record.metadata.raw_structured_coder_response
    raw_candidates = (raw,) if raw is not None else _legacy_freeform_plan_candidates(record)
    return tuple(item.strip() for item in raw_candidates if item and item.strip())


def resolve_approved_plan_anchor(
    issue_view: AuthenticatedCommentView,
    *,
    plan_hash: str,
    plan_subject: str | None = None,
) -> ApprovedPlanAnchor:
    """Return the earliest authenticated plan-flow record carrying the approved plan.

    Duplicate records with identical canonical text resolve to the earliest;
    differing text, or duplicates whose mutual order is ``unordered``, leave no
    anchor and fail closed.
    """
    _require_view(issue_view, kind=ISSUE_THREAD_SURFACE)
    matches: list[tuple[AuthenticatedComment, str]] = []
    for record, comment in _plan_records(issue_view):
        for text in _plan_text_candidates(record):
            if _approved_plan_hash(text) != plan_hash:
                continue
            if plan_subject is not None and _plan_subject(text) != plan_subject:
                continue
            matches.append((comment, text))
            break
    if not matches:
        raise WorkflowTransactionError(
            f"No authenticated approved-plan record with hash {plan_hash} exists on "
            f"{issue_view.surface}",
            problems=(f"missing approved-plan anchor record for plan {plan_hash}",)
            + issue_view.foreign_diagnostics(),
            recovery_action=RECOVERY_ORIGINAL_ACTOR
            if issue_view.ignored_foreign
            else RECOVERY_OPERATOR_REVIEW,
            code="approved-plan-anchor-missing",
        )
    if len({text for _comment, text in matches}) != 1:
        raise WorkflowTransactionError(
            f"Authenticated plan records matching hash {plan_hash} carry differing text",
            problems=tuple(
                f"contradictory plan record in comment {comment.comment_id}"
                for comment, _text in matches
            ),
            recovery_action=RECOVERY_OPERATOR_REVIEW,
            code="approved-plan-anchor-divergent",
        )
    matches.sort(key=lambda pair: pair[0].comment_id)
    for (first, _a), (second, _b) in zip(matches, matches[1:]):
        if not is_strictly_later(first, second):
            raise WorkflowTransactionError(
                f"Authenticated plan records matching hash {plan_hash} cannot be ordered",
                problems=(
                    f"unordered plan records in comments {first.comment_id} and "
                    f"{second.comment_id}",
                ),
                recovery_action=RECOVERY_OPERATOR_REVIEW,
                code="approved-plan-anchor-unordered",
            )
    comment, text = matches[0]
    return ApprovedPlanAnchor(comment, plan_hash, _plan_subject(text), text)


def round_metadata_digest(comment: AuthenticatedComment) -> str:
    """Digest of the canonical round-metadata payload carried by one comment."""
    matches = list(ROUND_RESUME_MARKER_RE.finditer(comment.body))
    if not matches:
        raise AgentLoopError(
            f"Comment {comment.comment_id} on {comment.surface} carries no round metadata."
        )
    return hashlib.sha256(matches[-1].group("payload").encode("ascii")).hexdigest()


def select_scheduler_checkpoint(
    plan_issue_view: AuthenticatedCommentView | None,
    *,
    origin_flow: str,
    plan_hash: str | None = None,
    plan_subject: str | None = None,
) -> SchedulerCheckpointRef:
    """Bind the approved candidate's own scheduling decision, deterministically.

    The reference is the earliest authenticated issue-side scheduler-prelaunch
    summary whose subject is the approved plan's subject and that is strictly
    later than the approved-plan anchor.  It can never be displaced by records
    appended later, and PR-side scheduler records are never consulted.
    """
    if origin_flow != FLOW_APPROVED_PLAN:
        if origin_flow not in ORIGIN_FLOWS:
            raise _fail(f"unknown origin flow {origin_flow!r}.")
        return SchedulerCheckpointRef(absence_reason=ABSENCE_FLOW_WITHOUT_PLAN_REVIEW)
    view = _require_view(plan_issue_view, kind=ISSUE_THREAD_SURFACE)
    if not isinstance(plan_hash, str):
        raise _fail("approved-plan-implementation requires an approved-plan hash.")
    anchor = resolve_approved_plan_anchor(view, plan_hash=plan_hash, plan_subject=plan_subject)
    records = _plan_records(view)
    invalid = [
        comment for record, comment in records
        if record.metadata.scheduler_metadata_status == "invalid"
    ]
    if invalid:
        raise WorkflowTransactionError(
            f"A plan-flow record on {view.surface} carries scheduler metadata that does not "
            "decode",
            problems=tuple(
                f"contradictory scheduler metadata in comment {comment.comment_id}"
                for comment in invalid
            ),
            recovery_action=RECOVERY_OPERATOR_REVIEW,
            code="scheduler-metadata-invalid",
        )
    scheduled = [
        (record, comment) for record, comment in records
        if record.metadata.scheduler_metadata_status == "valid"
    ]
    if not scheduled:
        return SchedulerCheckpointRef(absence_reason=ABSENCE_NO_PLAN_SCHEDULER_RECORDS)
    candidates: list[AuthenticatedComment] = []
    for record, comment in scheduled:
        metadata = record.metadata
        if metadata.role != "summary" or metadata.phase != "scheduler-prelaunch":
            continue
        if metadata.subject != anchor.plan_subject:
            continue
        key = metadata.plan_candidate_key
        if key is not None and key.get("subject") not in (None, anchor.plan_subject):
            continue
        order = comment_order(anchor.comment, comment)
        if order == "unordered":
            raise WorkflowTransactionError(
                "A scheduler checkpoint cannot be ordered against the approved-plan anchor",
                problems=(
                    f"unordered scheduler checkpoint in comment {comment.comment_id} against "
                    f"anchor comment {anchor.comment.comment_id}",
                ),
                recovery_action=RECOVERY_OPERATOR_REVIEW,
                code="scheduler-checkpoint-unordered",
            )
        if order == "later":
            candidates.append(comment)
    if not candidates:
        raise WorkflowTransactionError(
            f"The planning history on {view.surface} scheduled reviews, but no scheduler "
            f"checkpoint matches approved plan {plan_hash} after its anchor record",
            problems=(
                f"missing scheduler checkpoint for plan subject {anchor.plan_subject} after "
                f"anchor comment {anchor.comment.comment_id}",
            ),
            recovery_action=RECOVERY_RERUN_PLAN_REVIEW,
            code="scheduler-checkpoint-unmatched",
        )
    chosen = min(candidates, key=lambda item: item.comment_id)
    return SchedulerCheckpointRef(
        reference=CommentRef(view.surface, chosen.comment_id, round_metadata_digest(chosen))
    )


def verify_scheduler_checkpoint(
    intent: WorkflowTransition, plan_issue_view: AuthenticatedCommentView | None
) -> None:
    """Re-verify the hashed checkpoint reference; fail closed on any mismatch."""
    reference = intent.scheduler_checkpoint.reference
    if reference is None:
        return
    view = _require_view(
        plan_issue_view, kind=ISSUE_THREAD_SURFACE, number=intent.plan_owning_issue
    )

    def refuse(problem: str, *, recovery: str = RECOVERY_OPERATOR_REVIEW) -> WorkflowTransactionError:
        return WorkflowTransactionError(
            "The transaction's scheduler checkpoint reference no longer authenticates",
            transaction_ids=(intent.transaction_id,),
            successor_kind=intent.successor_kind,
            expected_record_set=intent.expected_record_set,
            problems=(problem,),
            recovery_action=recovery,
            code="scheduler-checkpoint-invalid",
        )

    comment = view.comment(reference.comment_id)
    if comment is None:
        if any(item.comment_id == reference.comment_id for item in view.ignored_foreign):
            raise refuse(
                f"ignored-foreign scheduler checkpoint comment {reference.comment_id}",
                recovery=RECOVERY_ORIGINAL_ACTOR,
            )
        raise refuse(f"missing scheduler checkpoint comment {reference.comment_id}")
    if round_metadata_digest(comment) != reference.digest:
        raise refuse(f"contradictory digest for scheduler checkpoint comment {comment.comment_id}")
    assert intent.approved_plan_hash is not None
    anchor = resolve_approved_plan_anchor(view, plan_hash=intent.approved_plan_hash)
    record = next(
        (item for item, candidate in _plan_records(view) if candidate is comment), None
    )
    if (
        record is None
        or record.metadata.role != "summary"
        or record.metadata.phase != "scheduler-prelaunch"
        or record.metadata.scheduler_metadata_status != "valid"
        or record.metadata.subject != anchor.plan_subject
    ):
        raise refuse(
            f"contradictory scheduler checkpoint comment {comment.comment_id}: it is not a "
            "scheduler-prelaunch record for the approved plan's subject"
        )
    if not is_strictly_later(anchor.comment, comment):
        raise refuse(
            f"unordered scheduler checkpoint comment {comment.comment_id}: it is not strictly "
            f"later than anchor comment {anchor.comment.comment_id}"
        )


# ---------------------------------------------------------------------------
# Legacy root validation
# ---------------------------------------------------------------------------


def find_origin_evidence(
    pr_view: AuthenticatedCommentView,
) -> OriginEvidence | None:
    """Return PR-side round-metadata origin evidence, or ``None`` when none exists.

    Only PR reviewer rounds that ran with approved-plan context write
    ``approved_plan_hash`` and ``approved_plan_subject``.  Every such
    authenticated record must agree on exactly those two fields; the head
    ``subject`` legitimately differs between records of a multi-head PR.
    """
    view = _require_view(pr_view, kind=PR_THREAD_SURFACE)
    carrying = _plan_carrying_pr_records(view)
    if not carrying:
        return None
    _require_evidence_agreement(carrying)
    record, comment = carrying[0]
    assert record.metadata.approved_plan_hash and record.metadata.approved_plan_subject
    return OriginEvidence(
        kind=EVIDENCE_PR_ROUND_METADATA,
        plan_hash=record.metadata.approved_plan_hash,
        plan_subject=record.metadata.approved_plan_subject,
        reference=CommentRef(view.surface, comment.comment_id, round_metadata_digest(comment)),
    )


def _plan_carrying_pr_records(
    view: AuthenticatedCommentView,
) -> list[tuple[PostedRoundRecord, AuthenticatedComment]]:
    """PR reviewer records carrying a complete approved-plan identity.

    Only a PR reviewer round that ran with approved-plan context is evidence.
    A record that carries either plan field without being such a complete
    reviewer record is malformed evidence and fails closed; it is never
    silently dropped from the agreement check.
    """
    carrying: list[tuple[PostedRoundRecord, AuthenticatedComment]] = []
    for record in _extract_round_metadata_records(view.authored, flow="pr"):
        metadata = record.metadata
        if metadata.approved_plan_hash is None and metadata.approved_plan_subject is None:
            continue
        comment = view.authored[record.index]
        if (
            metadata.role != "reviewer"
            or not metadata.approved_plan_hash
            or not metadata.approved_plan_subject
        ):
            raise WorkflowTransactionError(
                "An authenticated PR round-metadata record carries an approved-plan identity "
                "but is not a complete PR reviewer record",
                problems=(
                    f"contradictory approved-plan identity in comment {comment.comment_id} "
                    f"(role {metadata.role})",
                ),
                recovery_action=RECOVERY_OPERATOR_REVIEW,
                code="origin-evidence-conflict",
            )
        carrying.append((record, comment))
    return carrying


def _require_evidence_agreement(
    carrying: Sequence[tuple[PostedRoundRecord, AuthenticatedComment]],
) -> None:
    identities = {
        (record.metadata.approved_plan_hash, record.metadata.approved_plan_subject)
        for record, _comment in carrying
    }
    if len(identities) > 1:
        raise WorkflowTransactionError(
            "Authenticated PR round-metadata records disagree on the approved plan",
            problems=tuple(
                f"contradictory approved-plan identity in comment {comment.comment_id}"
                for _record, comment in carrying
            ),
            recovery_action=RECOVERY_OPERATOR_REVIEW,
            code="origin-evidence-conflict",
        )


def validate_legacy_root(
    intent: WorkflowTransition,
    *,
    pr_view: AuthenticatedCommentView,
    plan_issue_view: AuthenticatedCommentView,
    pr_commit_shas: Sequence[str],
    primary_issue_view: AuthenticatedCommentView | None = None,
) -> None:
    """Re-validate a legacy root against authenticated envelopes; fail closed."""
    root = intent.legacy_root
    if root is None:
        raise _fail("this transition carries no legacy root.")
    pr = _require_view(pr_view, kind=PR_THREAD_SURFACE, number=intent.pr_number)
    plan_view = _require_view(
        plan_issue_view, kind=ISSUE_THREAD_SURFACE, number=intent.plan_owning_issue
    )

    def refuse(problem: str, *, recovery: str = RECOVERY_OPERATOR_REVIEW) -> WorkflowTransactionError:
        return WorkflowTransactionError(
            "The legacy root of a legacy-root-correction does not re-validate",
            transaction_ids=(intent.transaction_id,),
            successor_kind=intent.successor_kind,
            expected_record_set=intent.expected_record_set,
            problems=(problem,),
            recovery_action=recovery,
            code="legacy-root-invalid",
        )

    def authored(view: AuthenticatedCommentView, reference: CommentRef, what: str):
        comment = view.comment(reference.comment_id)
        if comment is None:
            if any(item.comment_id == reference.comment_id for item in view.ignored_foreign):
                raise refuse(
                    f"ignored-foreign {what} comment {reference.comment_id}",
                    recovery=RECOVERY_ORIGINAL_ACTOR,
                )
            raise refuse(f"missing {what} comment {reference.comment_id}")
        return comment

    contract_comment = authored(pr, root.contract, "version-1 PR contract")
    contract_hashes = set()
    for match in PR_EXPECTED_CLOSING_MARKER_RE.finditer(contract_comment.body):
        if pr_contract_payload_schema_version(match.group("payload")) == 1:
            contract_hashes.add(pr_contract_record_hash(decode_pr_contract(match.group("payload"))))
    if contract_hashes != {root.contract.digest}:
        raise refuse(
            f"contradictory version-1 PR contract in comment {contract_comment.comment_id}: "
            "it no longer matches its recorded hash"
        )
    if root.handoff is not None:
        assert intent.primary_issue is not None
        issue_view = primary_issue_view
        if issue_view is None and plan_view.surface == root.handoff.surface:
            issue_view = plan_view
        issue_view = _require_view(
            issue_view, kind=ISSUE_THREAD_SURFACE, number=intent.primary_issue
        )
        handoff_comment = authored(issue_view, root.handoff, "version-1 handoff")
        lineage = resolve_issue_pr_handoff_lineage(
            (handoff_comment,), issue_number=intent.primary_issue, repo=intent.repository
        )
        if lineage is None or issue_pr_handoff_record_hash(lineage.latest) != root.handoff.digest:
            raise refuse(
                f"contradictory version-1 handoff in comment {handoff_comment.comment_id}: "
                "it no longer matches its recorded hash"
            )
    evidence = root.origin_evidence
    try:
        anchor = resolve_approved_plan_anchor(
            plan_view, plan_hash=evidence.plan_hash, plan_subject=evidence.plan_subject
        )
    except WorkflowTransactionError as exc:
        raise refuse(
            f"the approved plan {evidence.plan_hash} named by the origin evidence does not "
            f"authenticate ({exc.code})"
        ) from exc
    if evidence.kind == EVIDENCE_OPERATOR_ASSERTED:
        if evidence.operator_id != pr.actor_id or evidence.operator_login != pr.actor_login:
            raise refuse(
                f"contradictory operator assertion by {evidence.operator_login} "
                f"(user ID {evidence.operator_id})",
                recovery=RECOVERY_ORIGINAL_ACTOR,
            )
        return
    assert evidence.reference is not None
    evidence_comment = authored(pr, evidence.reference, "origin-evidence")
    if round_metadata_digest(evidence_comment) != evidence.reference.digest:
        raise refuse(
            f"contradictory digest for origin-evidence comment {evidence_comment.comment_id}"
        )
    try:
        carrying = _plan_carrying_pr_records(pr)
    except WorkflowTransactionError as exc:
        raise refuse("; ".join(exc.problems)) from exc
    record = next(
        (item for item, comment in carrying if comment is evidence_comment), None
    )
    if record is None:
        raise refuse(
            f"contradictory origin-evidence comment {evidence_comment.comment_id}: it carries "
            "no approved-plan identity"
        )
    try:
        _require_evidence_agreement(carrying)
    except WorkflowTransactionError as exc:
        raise refuse("; ".join(exc.problems)) from exc
    if (
        record.metadata.approved_plan_hash != evidence.plan_hash
        or record.metadata.approved_plan_subject != evidence.plan_subject
    ):
        raise refuse(
            f"contradictory origin-evidence comment {evidence_comment.comment_id}: it names "
            "another approved plan"
        )
    if record.metadata.subject not in set(pr_commit_shas):
        raise refuse(
            f"contradictory origin-evidence comment {evidence_comment.comment_id}: its head "
            f"{record.metadata.subject} is not a commit of PR #{intent.pr_number}"
        )
    if not is_strictly_later(anchor.comment, evidence_comment):
        raise refuse(
            f"unordered origin-evidence comment {evidence_comment.comment_id}: it is not "
            f"strictly later than approved-plan anchor comment {anchor.comment.comment_id}"
        )


# ---------------------------------------------------------------------------
# Era classification
# ---------------------------------------------------------------------------


def _declares_v2(encoded: str, peek) -> bool:
    try:
        return peek(encoded) == 2
    except AgentLoopError:
        return False


def classify_transaction_era(
    pr_view: AuthenticatedCommentView,
    issue_view: AuthenticatedCommentView | None = None,
) -> str:
    """Classify a PR as ``legacy`` or ``transaction`` era from durable records.

    Any authenticated version-2 authority-bearing record makes the PR
    transaction-era even when no transaction record is visible, so deleting or
    hiding a transaction record can only fail closed.  Records from any other
    author never affect the classification.
    """
    view = _require_view(pr_view, kind=PR_THREAD_SURFACE)
    for comment in view.authored:
        if WORKFLOW_TRANSACTION_MARKER_RE.search(comment.body):
            return ERA_TRANSACTION
        for match in PR_EXPECTED_CLOSING_MARKER_RE.finditer(comment.body):
            if _declares_v2(match.group("payload"), pr_contract_payload_schema_version):
                return ERA_TRANSACTION
    if issue_view is not None:
        _, pr_number = parse_comment_thread_surface(view.surface)
        for comment in _require_view(issue_view, kind=ISSUE_THREAD_SURFACE).authored:
            for match in AGENT_ISSUE_PR_HANDOFF_RE.finditer(comment.body):
                encoded = match.group("payload")
                if not _declares_v2(encoded, issue_pr_handoff_payload_schema_version):
                    continue
                try:
                    if decode_issue_pr_handoff_v2(encoded).pr_number != pr_number:
                        continue
                except AgentLoopError:
                    pass
                return ERA_TRANSACTION
    return ERA_LEGACY


# ---------------------------------------------------------------------------
# Version-2-aware lineage entry points (authenticated envelopes only)
# ---------------------------------------------------------------------------


def _split_by_version(
    comments: Sequence[AuthenticatedComment], marker_re: re.Pattern[str], peek, *, what: str
) -> tuple[list[AuthenticatedComment], list[tuple[AuthenticatedComment, str]]]:
    v1_comments: list[AuthenticatedComment] = []
    v2_items: list[tuple[AuthenticatedComment, str]] = []
    for comment in comments:
        payloads = [match.group("payload") for match in marker_re.finditer(comment.body)]
        if not payloads:
            continue
        versions = {2 if _declares_v2(payload, peek) else 1 for payload in payloads}
        if versions == {1}:
            v1_comments.append(comment)
        elif versions == {2}:
            v2_items.extend((comment, payload) for payload in payloads)
        else:
            raise WorkflowTransactionError(
                f"One comment mixes version-1 and version-2 {what} records",
                problems=(f"contradictory {what} records in comment {comment.comment_id}",),
                recovery_action=RECOVERY_OPERATOR_REVIEW,
            )
    if v2_items:
        first_v2 = min(comment.comment_id for comment, _payload in v2_items)
        late = [comment for comment in v1_comments if comment.comment_id > first_v2]
        if late:
            raise WorkflowTransactionError(
                f"A version-1 {what} record was appended after the PR became transaction-era",
                problems=tuple(
                    f"contradictory version-1 {what} record in comment {comment.comment_id}"
                    for comment in late
                ),
                recovery_action=RECOVERY_OPERATOR_REVIEW,
            )
    return v1_comments, v2_items


@dataclass(frozen=True)
class _Group:
    record: object
    encoded: str
    comments: tuple[AuthenticatedComment, ...]

    @property
    def comment_ids(self) -> tuple[int, ...]:
        return tuple(item.comment_id for item in self.comments)

    @property
    def canonical_comment_id(self) -> int:
        return min(self.comment_ids)


def _group_v2(
    items: Sequence[tuple[AuthenticatedComment, str, object, str]], *, what: str
) -> dict[str, _Group]:
    groups: dict[str, _Group] = {}
    for comment, encoded, record, tx_id in items:
        existing = groups.get(tx_id)
        if existing is None:
            groups[tx_id] = _Group(record, encoded, (comment,))
        elif existing.encoded != encoded:
            raise WorkflowTransactionError(
                f"Divergent version-2 {what} records are bound to one transaction",
                transaction_ids=(tx_id,),
                problems=(f"contradictory {what} record in comment {comment.comment_id}",),
                recovery_action=RECOVERY_OPERATOR_REVIEW,
            )
        else:
            groups[tx_id] = _Group(record, encoded, existing.comments + (comment,))
    return groups


def _require_bound_states(
    groups: Mapping[str, _Group], lineage: TransactionLineage, *, what: str
) -> None:
    for tx_id, group in groups.items():
        if lineage.state(tx_id) is None:
            raise WorkflowTransactionError(
                f"A version-2 {what} record is bound to a transaction with no authenticated "
                "prepared record; the PR stays transaction-era and grants nothing",
                transaction_ids=(tx_id,),
                problems=(
                    f"missing prepared record for {what} comment {group.canonical_comment_id}",
                ),
                recovery_action=RECOVERY_OPERATOR_REVIEW,
                code="transaction-record-missing",
            )


def _reissued_group(
    state: TransactionState, groups: Mapping[str, _Group], name: str, *, what: str
) -> tuple[_Group, AuthenticatedComment]:
    """Return the record group and the exact comment the terminal record names.

    Publication order is part of the contract: the named comment must be
    strictly later than the prepared comment the terminal binds and strictly
    earlier than the terminal comment, under the shared two-field rule.  A
    record posted before preparation, inside or after the terminal comment is
    never accepted retroactively.
    """
    group = groups.get(state.transaction_id)
    outcome = state.outcome(name)
    named = (
        next(
            (item for item in group.comments if item.comment_id == outcome.comment_id), None
        )
        if group is not None and outcome is not None and outcome.comment_id is not None
        else None
    )
    if group is None or named is None:
        raise _transaction_error(
            f"A committed transaction's {what} record is missing or is not the comment its "
            "terminal record names",
            states=(state,),
            problems=(f"missing {what} record for transaction {state.transaction_id}",),
            code="record-missing",
        )
    assert state.bound_prepared_comment is not None and state.terminal_comment is not None
    if not (
        is_strictly_later(state.bound_prepared_comment, named)
        and is_strictly_later(named, state.terminal_comment)
    ):
        raise _transaction_error(
            f"A committed transaction's {what} record was not published between its prepared "
            "and terminal records",
            states=(state,),
            problems=(
                f"unordered {what} record in comment {named.comment_id}: expected prepared "
                f"comment {state.bound_prepared_comment.comment_id} < {named.comment_id} < "
                f"terminal comment {state.terminal_comment.comment_id}",
            ),
            code="record-unordered",
        )
    return group, named


@dataclass(frozen=True)
class ResolvedPrContract:
    contract: PrExpectedClosingContract | PrExpectedClosingContractV2 | None
    comment_id: int | None
    record_hash: str | None
    era: str
    # The committed transaction that makes the contract authoritative, if any.
    transaction_id: str | None = None
    # A prepared-only transaction exists; nothing it published counts yet.
    pending_transaction_id: str | None = None


def _v1_contract_comment_id(
    comments: Sequence[AuthenticatedComment], contract: PrExpectedClosingContract
) -> int:
    latest = -1
    for comment in comments:
        for match in PR_EXPECTED_CLOSING_MARKER_RE.finditer(comment.body):
            if decode_pr_contract(match.group("payload")) == contract:
                latest = comment.comment_id
    return latest


def _check_contract_supersession(
    new: PrExpectedClosingContractV2,
    state: TransactionState,
    previous: ResolvedPrContract | None,
    *,
    surface: str,
) -> None:
    def refuse(problem: str) -> WorkflowTransactionError:
        return _transaction_error(
            "A version-2 PR contract does not validly supersede the record before it",
            states=(state,),
            problems=(problem,),
            code="contract-supersession-invalid",
        )

    prior = previous.contract if previous is not None else None
    if prior is None:
        if new.supersession_kind is not None:
            raise refuse("contradictory supersession: there is no record to supersede")
        return
    same_scope = (
        prior.primary_issue_number == new.primary_issue_number
        and tuple(prior.expected_closing_issue_ids) == tuple(new.expected_closing_issue_ids)
    )
    if new.supersession_kind is None:
        if not (same_scope and prior.origin_flow == new.origin_flow):
            raise refuse(
                f"contradictory contract: it differs from comment {previous.comment_id} "
                "without declaring a supersession"
            )
        return
    if new.supersedes_record_hash != previous.record_hash:
        raise refuse(
            "contradictory supersession: the record hash does not match comment "
            f"{previous.comment_id}"
        )
    kind = state.intent.successor_kind
    prior_is_v1 = isinstance(prior, PrExpectedClosingContract)
    if new.supersession_kind == "closing-widening":
        widened = set(prior.expected_closing_issue_ids) < set(new.expected_closing_issue_ids)
        if not (
            widened
            and prior.origin_flow == new.origin_flow
            and prior.primary_issue_number == new.primary_issue_number
            and (kind == KIND_CLOSING_WIDENING or (kind == KIND_INITIAL and prior_is_v1))
        ):
            raise refuse("contradictory closing-widening supersession")
        return
    if not same_scope or prior.origin_flow == new.origin_flow:
        raise refuse("contradictory flow-correction: only the origin flow may change")
    if kind == KIND_FLOW_CORRECTION and not prior_is_v1:
        return
    root = state.intent.legacy_root
    if (
        kind == KIND_LEGACY_ROOT_CORRECTION
        and prior_is_v1
        and root is not None
        and previous.comment_id is not None
        and previous.record_hash is not None
        and root.contract == CommentRef(surface, previous.comment_id, previous.record_hash)
    ):
        return
    raise refuse(
        "contradictory flow-correction: it needs a committed flow-correction transaction over "
        "a version-2 record, or a legacy-root-correction whose legacy root names exactly the "
        "superseded version-1 record"
    )


def resolve_pr_contract_lineage(
    pr_view: AuthenticatedCommentView,
    lineage: TransactionLineage,
    *,
    repository: str,
    pr_number: int,
) -> ResolvedPrContract | None:
    """Resolve the authoritative PR contract across version-1 and version-2 records.

    Version-1 records keep exactly the rules of ``find_latest_pr_contract``.
    A version-2 record counts only when its transaction is committed and
    canonical; records bound to an aborted transaction are inert.
    """
    view = _require_view(pr_view, kind=PR_THREAD_SURFACE, number=pr_number)
    if not isinstance(lineage, TransactionLineage) or lineage.surface != view.surface:
        raise AgentLoopError("PR contract lineage requires this PR's transaction lineage.")
    v1_comments, raw_v2 = _split_by_version(
        view.authored, PR_EXPECTED_CLOSING_MARKER_RE, pr_contract_payload_schema_version,
        what="PR contract",
    )
    v1 = find_latest_pr_contract(v1_comments, repository=repository, pr_number=pr_number)
    decoded = []
    for comment, encoded in raw_v2:
        contract = decode_pr_contract_v2(encoded)
        if contract.repository.casefold() != repository.casefold() or contract.pr_number != pr_number:
            raise AgentLoopError(
                f"A version-2 PR contract record does not belong to {repository} PR #{pr_number}."
            )
        decoded.append((comment, encoded, contract, contract.transaction_id))
    groups = _group_v2(decoded, what="PR contract")
    _require_bound_states(groups, lineage, what="PR contract")
    era = ERA_TRANSACTION if (groups or lineage.transactions) else ERA_LEGACY
    current: ResolvedPrContract | None = None
    if v1 is not None:
        current = ResolvedPrContract(
            v1, _v1_contract_comment_id(v1_comments, v1), pr_contract_record_hash(v1), era
        )
    for state in lineage.chain:
        entry = state.intent.entry(ENTRY_PR_CONTRACT)
        if entry.disposition != DISPOSITION_REISSUED and state.transaction_id in groups:
            raise _transaction_error(
                "A PR contract record is bound to a transaction that does not reissue it",
                states=(state,),
                problems=(
                    "contradictory PR contract record in comment "
                    f"{groups[state.transaction_id].canonical_comment_id}",
                ),
            )
        if not state.committed:
            continue
        if entry.disposition == DISPOSITION_REISSUED:
            group, named = _reissued_group(
                state, groups, ENTRY_PR_CONTRACT, what="PR contract"
            )
            contract = group.record
            assert isinstance(contract, PrExpectedClosingContractV2)
            if contract != derive_pr_contract(
                state.intent,
                supersession_kind=contract.supersession_kind,
                supersedes_record_hash=contract.supersedes_record_hash,
            ):
                raise _transaction_error(
                    "A version-2 PR contract disagrees with its transaction intent",
                    states=(state,),
                    problems=(
                        f"contradictory PR contract record in comment {group.canonical_comment_id}",
                    ),
                )
            _check_contract_supersession(contract, state, current, surface=view.surface)
            current = ResolvedPrContract(
                contract,
                named.comment_id,
                pr_contract_record_hash(contract),
                era,
                transaction_id=state.transaction_id,
            )
        elif entry.inherited is not None:
            if (
                current is None
                or current.comment_id != entry.inherited.comment_id
                or current.record_hash != entry.inherited.digest
            ):
                raise _transaction_error(
                    "An inherited PR contract does not match the predecessor chain's "
                    "canonical record",
                    states=(state,),
                    problems=(
                        "contradictory inherited PR contract reference to comment "
                        f"{entry.inherited.comment_id}",
                    ),
                    code="inherited-mismatch",
                )
            current = ResolvedPrContract(
                current.contract, current.comment_id, current.record_hash, era,
                transaction_id=state.transaction_id,
            )
    pending = lineage.pending
    if current is None:
        if era == ERA_LEGACY:
            return None
        return ResolvedPrContract(
            None, None, None, era,
            pending_transaction_id=pending.transaction_id if pending else None,
        )
    return ResolvedPrContract(
        current.contract, current.comment_id, current.record_hash, era,
        transaction_id=current.transaction_id,
        pending_transaction_id=pending.transaction_id if pending else None,
    )


@dataclass(frozen=True)
class ResolvedHandoff:
    handoff: IssuePrHandoffMetadata | IssuePrHandoffMetadataV2 | None
    comment_id: int | None
    record_hash: str | None
    era: str
    transaction_id: str | None = None
    pending_transaction_id: str | None = None
    v1_lineage: IssuePrHandoffLineage | None = None


def handoff_candidate_pr_numbers(
    issue_view: AuthenticatedCommentView, *, issue_number: int
) -> tuple[int, ...]:
    """PR numbers named by authenticated handoff records of either version, in order."""
    view = _require_view(issue_view, kind=ISSUE_THREAD_SURFACE, number=issue_number)
    numbers: list[int] = []
    for comment in view.authored:
        for match in AGENT_ISSUE_PR_HANDOFF_RE.finditer(comment.body):
            encoded = match.group("payload")
            if _declares_v2(encoded, issue_pr_handoff_payload_schema_version):
                record = decode_issue_pr_handoff_v2(encoded)
                if record.issue_number != issue_number:
                    continue
                number = record.pr_number
            else:
                legacy = _decode_issue_pr_handoff_metadata(encoded)
                if legacy.issue_number != issue_number:
                    continue
                number = legacy.pr_number
            if number not in numbers:
                numbers.append(number)
    return tuple(numbers)


def resolve_handoff_lineage(
    issue_view: AuthenticatedCommentView,
    lineage: TransactionLineage,
    *,
    repository: str,
    issue_number: int,
) -> ResolvedHandoff | None:
    """Resolve the authoritative issue-to-PR handoff for the lineage's PR.

    Version-1 records keep exactly the rules of
    ``resolve_issue_pr_handoff_lineage``.  A version-2 record counts only when
    its transaction is committed and canonical.
    """
    view = _require_view(issue_view, kind=ISSUE_THREAD_SURFACE, number=issue_number)
    if not isinstance(lineage, TransactionLineage):
        raise AgentLoopError("Handoff lineage requires the PR's transaction lineage.")
    v1_comments, raw_v2 = _split_by_version(
        view.authored, AGENT_ISSUE_PR_HANDOFF_RE, issue_pr_handoff_payload_schema_version,
        what="handoff",
    )
    v1_lineage = resolve_issue_pr_handoff_lineage(
        v1_comments, issue_number=issue_number, repo=repository
    )
    decoded = []
    for comment, encoded in raw_v2:
        record = decode_issue_pr_handoff_v2(encoded)
        if record.issue_number != issue_number or record.pr_number != lineage.pr_number:
            continue
        decoded.append((comment, encoded, record, record.transaction_id))
    groups = _group_v2(decoded, what="handoff")
    _require_bound_states(groups, lineage, what="handoff")
    era = ERA_TRANSACTION if (groups or lineage.transactions) else ERA_LEGACY
    current: ResolvedHandoff | None = None
    if v1_lineage is not None and v1_lineage.latest.pr_number == lineage.pr_number:
        current = ResolvedHandoff(
            v1_lineage.latest,
            v1_comments[v1_lineage.latest_comment_index].comment_id,
            issue_pr_handoff_record_hash(v1_lineage.latest),
            era,
            v1_lineage=v1_lineage,
        )
    for state in lineage.chain:
        intent = state.intent
        if intent.primary_issue != issue_number:
            raise _transaction_error(
                f"The PR's transaction chain belongs to issue #{intent.primary_issue}, not "
                f"#{issue_number}",
                states=(state,),
            )
        entry = intent.entry(ENTRY_HANDOFF)
        if entry.disposition != DISPOSITION_REISSUED and state.transaction_id in groups:
            raise _transaction_error(
                "A handoff record is bound to a transaction that does not reissue it",
                states=(state,),
                problems=(
                    "contradictory handoff record in comment "
                    f"{groups[state.transaction_id].canonical_comment_id}",
                ),
            )
        root = intent.legacy_root
        if root is not None:
            named = (
                None if root.handoff is None else (root.handoff.comment_id, root.handoff.digest)
            )
            actual = None if current is None else (current.comment_id, current.record_hash)
            if named != actual:
                raise _transaction_error(
                    "The legacy root's version-1 handoff does not match the authenticated "
                    "issue conversation",
                    states=(state,),
                    problems=("contradictory legacy-root handoff reference",),
                    code="legacy-root-invalid",
                )
        if not state.committed:
            continue
        if entry.disposition == DISPOSITION_REISSUED:
            group, named = _reissued_group(state, groups, ENTRY_HANDOFF, what="handoff")
            if group.record != derive_handoff_metadata(intent):
                raise _transaction_error(
                    "A version-2 handoff disagrees with its transaction intent",
                    states=(state,),
                    problems=(
                        f"contradictory handoff record in comment {group.canonical_comment_id}",
                    ),
                )
            assert isinstance(group.record, IssuePrHandoffMetadataV2)
            current = ResolvedHandoff(
                group.record,
                named.comment_id,
                issue_pr_handoff_record_hash(group.record),
                era,
                transaction_id=state.transaction_id,
                v1_lineage=v1_lineage,
            )
        elif entry.inherited is not None:
            if (
                current is None
                or current.comment_id != entry.inherited.comment_id
                or current.record_hash != entry.inherited.digest
            ):
                raise _transaction_error(
                    "An inherited handoff does not match the predecessor chain's canonical "
                    "record",
                    states=(state,),
                    problems=(
                        "contradictory inherited handoff reference to comment "
                        f"{entry.inherited.comment_id}",
                    ),
                    code="inherited-mismatch",
                )
            current = ResolvedHandoff(
                current.handoff, current.comment_id, current.record_hash, era,
                transaction_id=state.transaction_id, v1_lineage=v1_lineage,
            )
    pending = lineage.pending
    pending_id = pending.transaction_id if pending else None
    if current is None:
        if era == ERA_LEGACY:
            return None
        return ResolvedHandoff(
            None, None, None, era, pending_transaction_id=pending_id, v1_lineage=v1_lineage
        )
    return ResolvedHandoff(
        current.handoff, current.comment_id, current.record_hash, era,
        transaction_id=current.transaction_id,
        pending_transaction_id=pending_id,
        v1_lineage=v1_lineage,
    )


# ---------------------------------------------------------------------------
# Obsolete prepared intent and successor planning (pure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransitionInputs:
    """Freshly derived authenticated inputs a stored intent is compared against."""

    repository: str
    primary_issue: int | None
    pr_number: int
    base: str
    head_sha: str
    origin_flow: str
    approved_plan_hash: str | None
    expected_closing_issue_ids: tuple[int, ...]
    staged: StagedIdentity | None = None
    managed_ci_generation: str | None = None

    @classmethod
    def of(cls, intent: WorkflowTransition) -> "TransitionInputs":
        return cls(
            repository=intent.repository,
            primary_issue=intent.primary_issue,
            pr_number=intent.pr_number,
            base=intent.base,
            head_sha=intent.head_sha,
            origin_flow=intent.origin_flow,
            approved_plan_hash=intent.approved_plan_hash,
            expected_closing_issue_ids=intent.expected_closing_issue_ids,
            staged=intent.staged,
            managed_ci_generation=intent.managed_ci_generation,
        )


_COMPARED_FIELDS: tuple[str, ...] = (
    "repository", "primary_issue", "pr_number", "base", "head_sha", "origin_flow",
    "approved_plan_hash", "expected_closing_issue_ids", "staged", "managed_ci_generation",
)


def _differing_fields(
    stored: TransitionInputs, fresh: TransitionInputs
) -> tuple[tuple[str, str, str], ...]:
    differing: list[tuple[str, str, str]] = []
    for name in _COMPARED_FIELDS:
        before, after = getattr(stored, name), getattr(fresh, name)
        if name == "repository":
            before, after = before.casefold(), after.casefold()
        if name == "expected_closing_issue_ids":
            before, after = tuple(sorted(before)), tuple(sorted(after))
        if before != after:
            differing.append((name, repr(getattr(stored, name)), repr(getattr(fresh, name))))
    return tuple(differing)


@dataclass(frozen=True)
class IntentComparison:
    outcome: Literal["current", "obsolete", "contradiction"]
    differing_fields: tuple[tuple[str, str, str], ...] = ()
    abort_reason: str | None = None
    contradictions: tuple[str, ...] = ()


def compare_prepared_intent(
    stored: WorkflowTransition,
    fresh: TransitionInputs,
    *,
    stored_plan_anchor: AuthenticatedComment | None = None,
    fresh_plan_anchor: AuthenticatedComment | None = None,
) -> IntentComparison:
    """Decide whether an adopted prepared-only intent is current, obsolete, or contradicted.

    Legitimate differences make the prepared transaction obsolete: a different
    live head, an authenticated strictly-later approved plan, a strict-superset
    closing contract with the same primary issue, a different managed-CI
    generation, and an origin-flow correction over a committed predecessor.
    Every other difference is a contradiction: nothing may be written, not even
    an abort.
    """
    differing = _differing_fields(TransitionInputs.of(stored), fresh)
    if not differing:
        return IntentComparison("current")
    contradictions: list[str] = []
    for name, _before, _after in differing:
        if name in {"head_sha", "managed_ci_generation"}:
            continue
        if name == "approved_plan_hash":
            if (
                stored.approved_plan_hash is None
                or fresh.approved_plan_hash is None
                or stored_plan_anchor is None
                or fresh_plan_anchor is None
                or not is_strictly_later(stored_plan_anchor, fresh_plan_anchor)
            ):
                contradictions.append(
                    "approved_plan_hash: the fresh plan does not authenticate as strictly later"
                )
            continue
        if name == "expected_closing_issue_ids":
            if not (
                set(stored.expected_closing_issue_ids) < set(fresh.expected_closing_issue_ids)
                and stored.primary_issue == fresh.primary_issue
            ):
                contradictions.append(
                    "expected_closing_issue_ids: the fresh contract is not a strict superset"
                )
            continue
        if name == "origin_flow" and stored.predecessor_transaction_id is not None:
            continue
        contradictions.append(f"{name}: this field can never change within one scope")
    if contradictions:
        return IntentComparison("contradiction", differing, contradictions=tuple(contradictions))
    reason = (
        ABORT_STALE_HEAD
        if {name for name, _b, _a in differing} == {"head_sha"}
        else ABORT_SUPERSEDED_INTENT
    )
    return IntentComparison("obsolete", differing, abort_reason=reason)


_FIELD_KIND: dict[str, str] = {
    "approved_plan_hash": KIND_PLAN_REPLACEMENT,
    "origin_flow": KIND_FLOW_CORRECTION,
    "expected_closing_issue_ids": KIND_CLOSING_WIDENING,
    "managed_ci_generation": KIND_MANAGED_CI_CONTINUITY,
    "head_sha": KIND_HEAD_ADVANCE,
}
_FIELD_AFFECTS: dict[str, frozenset[str]] = {
    "approved_plan_hash": frozenset({ENTRY_HANDOFF}),
    "origin_flow": frozenset({ENTRY_HANDOFF, ENTRY_PR_CONTRACT}),
    "expected_closing_issue_ids": frozenset({ENTRY_HANDOFF, ENTRY_PR_CONTRACT}),
    "managed_ci_generation": frozenset({ENTRY_AUTHORIZATION}),
    # The bound managed-CI authorization is the only head-bound entry.
    "head_sha": frozenset({ENTRY_AUTHORIZATION}),
}


def successor_kind_for(committed: WorkflowTransition, fresh: TransitionInputs) -> str | None:
    """The highest-precedence successor kind for the fields that differ, or ``None``."""
    names = {name for name, _b, _a in _differing_fields(TransitionInputs.of(committed), fresh)}
    unsupported = names - set(_FIELD_KIND)
    if unsupported:
        raise _fail(
            "no successor kind can change " + ", ".join(sorted(unsupported)) + "."
        )
    kinds = {_FIELD_KIND[name] for name in names}
    return next((kind for kind in SUCCESSOR_KIND_PRECEDENCE if kind in kinds), None)


def plan_successor(
    committed: WorkflowTransition,
    fresh: TransitionInputs,
    *,
    effective_records: Mapping[str, CommentRef],
    scheduler_checkpoint: SchedulerCheckpointRef | None = None,
) -> WorkflowTransition | None:
    """Build the successor of the last committed transaction from fresh inputs.

    Lineage is always computed against the last *committed* transaction, never
    an aborted one.  Entries affected by a differing field are reissued; every
    other entry is inherited by authenticated reference.  Only a plan
    replacement recomputes the scheduler checkpoint reference.
    """
    kind = successor_kind_for(committed, fresh)
    if kind is None:
        return None
    names = {name for name, _b, _a in _differing_fields(TransitionInputs.of(committed), fresh)}
    affected = frozenset().union(*(_FIELD_AFFECTS[name] for name in names))
    entries: list[RecordSetEntry] = []
    for name in RECORD_SET_ENTRY_NAMES:
        if name == ENTRY_INITIAL_CODER_ROUND:
            entries.append(not_applicable(name))
        elif name == ENTRY_AUTHORIZATION:
            if fresh.managed_ci_generation is None:
                entries.append(not_applicable(name))
            elif name in affected or name not in effective_records:
                entries.append(reissued(name))
            else:
                entries.append(inherited(name, effective_records[name]))
        elif name == ENTRY_HANDOFF and fresh.origin_flow not in _ISSUE_ORIGIN_FLOWS:
            entries.append(not_applicable(name))
        elif name in affected:
            entries.append(reissued(name))
        elif name in effective_records:
            entries.append(inherited(name, effective_records[name]))
        else:
            entries.append(not_applicable(name))
    if "approved_plan_hash" in names:
        if scheduler_checkpoint is None:
            raise _fail("a plan replacement must recompute its scheduler checkpoint reference.")
        checkpoint = scheduler_checkpoint
    else:
        checkpoint = committed.scheduler_checkpoint
    return WorkflowTransition(
        repository=committed.repository,
        primary_issue=fresh.primary_issue,
        pr_number=fresh.pr_number,
        base=fresh.base,
        head_sha=fresh.head_sha,
        origin_flow=fresh.origin_flow,
        approved_plan_hash=fresh.approved_plan_hash,
        expected_closing_issue_ids=fresh.expected_closing_issue_ids,
        scheduler_checkpoint=checkpoint,
        record_set=tuple(entries),
        successor_kind=kind,
        predecessor_transaction_id=committed.transaction_id,
        staged=fresh.staged,
        managed_ci_generation=fresh.managed_ci_generation,
    )
