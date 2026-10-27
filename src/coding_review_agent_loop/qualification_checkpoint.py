"""Durable machine qualification checkpoints."""
from __future__ import annotations
from dataclasses import dataclass
from .protocol import MACHINE_OBLIGATION_KINDS, MACHINE_LIFECYCLE_STATES, EVIDENCE_OBLIGATION_KIND
from .managed_ci_result import FailedManagedRun, VerifiedManagedResult

@dataclass(frozen=True)
class QualificationCheckpoint:
    """Bounded durable state for a machine-obligation qualification attempt."""

    obligation_kind: str
    obligation_identity: str
    lifecycle: str
    failed_head_sha: str | None
    candidate_head_sha: str | None
    base_branch: str | None = None
    approval_digest: str | None = None
    plan_digest: str | None = None
    requirements_digest: str | None = None
    acquisition_digest: str | None = None
    scheduler_digest: str | None = None
    qualification_attempt_id: str | None = None
    watch_failure_extension_used: bool = False
    watch_head_extension_used: bool = False
    allowed_rounds: int = 0
    valid: bool = True
    failed_managed_run: FailedManagedRun | None = None
    verified_managed_result: VerifiedManagedResult | None = None

    def __post_init__(self) -> None:
        if not self.valid:
            return
        ref = self.failed_managed_run
        if ref is not None and not isinstance(ref, FailedManagedRun):
            raise ValueError("failed run must be typed")
        if self.verified_managed_result is not None and not isinstance(self.verified_managed_result, VerifiedManagedResult):
            raise ValueError("verified result must be typed")
        if ref is not None and (
            self.obligation_kind != "managed-exact-head-ci"
            or ref.obligation_identity != self.obligation_identity
            or ref.head_sha != self.failed_head_sha
            or ref.base_ref != self.base_branch
        ):
            raise ValueError("failed run contradicts checkpoint")
        if self.verified_managed_result is not None and self.verified_managed_result.failed_run != ref:
            raise ValueError("settlement contradicts failed run")
        if self.obligation_kind not in MACHINE_OBLIGATION_KINDS - {"unknown", EVIDENCE_OBLIGATION_KIND}:
            raise ValueError("qualification checkpoints require a known machine kind")
        if self.lifecycle not in MACHINE_LIFECYCLE_STATES - {"cleared", "evidence_deferred", "evidence_frozen"}:
            raise ValueError("qualification checkpoints require an active lifecycle")
        if self.lifecycle == "repair_required" and self.obligation_kind in {
            "managed-exact-head-ci", "github-pr-checks"
        } and not self.failed_head_sha:
            raise ValueError("CI qualification checkpoints need a failed head")
        if self.lifecycle in {
            "awaiting_current_head_review", "qualification_ready", "qualifying"
        } and not self.candidate_head_sha:
            raise ValueError("qualification-ready checkpoints need a candidate head")
        if self.candidate_head_sha and self.candidate_head_sha == self.failed_head_sha:
            raise ValueError("qualification checkpoints cannot target the failed head")

    def as_dict(self) -> dict[str, object]:
        return {
            "obligation_kind": self.obligation_kind,
            "obligation_identity": self.obligation_identity,
            "lifecycle": self.lifecycle,
            "failed_head_sha": self.failed_head_sha,
            "candidate_head_sha": self.candidate_head_sha,
            "base_branch": self.base_branch,
            "approval_digest": self.approval_digest,
            "plan_digest": self.plan_digest,
            "requirements_digest": self.requirements_digest,
            "acquisition_digest": self.acquisition_digest,
            "scheduler_digest": self.scheduler_digest,
            "qualification_attempt_id": self.qualification_attempt_id,
            "watch_failure_extension_used": self.watch_failure_extension_used,
            "watch_head_extension_used": self.watch_head_extension_used,
            "allowed_rounds": self.allowed_rounds,
            "valid": self.valid,
            **({"failed_managed_run": self.failed_managed_run.as_dict()} if self.failed_managed_run else {}),
            **({"verified_managed_result": self.verified_managed_result.as_dict()} if self.verified_managed_result else {}),
        }

    @classmethod
    def from_mapping(cls, value: object) -> "QualificationCheckpoint":
        if not isinstance(value, dict):
            return cls.invalid("checkpoint is not an object")
        required = {
            "obligation_kind", "obligation_identity", "lifecycle",
            "failed_head_sha", "candidate_head_sha", "base_branch",
            "approval_digest", "plan_digest", "requirements_digest",
            "acquisition_digest", "scheduler_digest", "qualification_attempt_id",
            "watch_failure_extension_used", "watch_head_extension_used",
            "allowed_rounds",
        }
        if not required.issubset(value):
            return cls.invalid("checkpoint is partial")
        strings = (
            "obligation_kind", "obligation_identity", "lifecycle", "failed_head_sha",
            "candidate_head_sha", "base_branch", "approval_digest", "plan_digest",
            "requirements_digest", "acquisition_digest", "scheduler_digest",
            "qualification_attempt_id",
        )
        if any(value[key] is not None and not isinstance(value[key], str) for key in strings):
            return cls.invalid("checkpoint contains a non-string identity")
        if value["obligation_kind"] not in MACHINE_OBLIGATION_KINDS - {"unknown", EVIDENCE_OBLIGATION_KIND}:
            return cls.invalid("checkpoint has an unknown obligation kind")
        if value["lifecycle"] not in MACHINE_LIFECYCLE_STATES - {"cleared", "evidence_deferred", "evidence_frozen"}:
            return cls.invalid("checkpoint has an unknown lifecycle")
        if value["lifecycle"] in {"qualification_ready", "qualifying"}:
            required_identities = (
                "approval_digest",
                "requirements_digest",
                "acquisition_digest",
                "scheduler_digest",
            )
            if any(
                not isinstance(value[key], str) or not value[key].strip()
                for key in required_identities
            ):
                return cls.invalid(
                    "qualification checkpoint is missing a current contract identity"
                )
            if value["lifecycle"] == "qualifying" and (
                not isinstance(value["qualification_attempt_id"], str)
                or not value["qualification_attempt_id"].strip()
            ):
                return cls.invalid(
                    "qualifying checkpoint is missing its qualification attempt identity"
                )
        if (
            not isinstance(value["watch_failure_extension_used"], bool)
            or not isinstance(value["watch_head_extension_used"], bool)
            or isinstance(value["allowed_rounds"], bool)
            or not isinstance(value["allowed_rounds"], int)
            or value["allowed_rounds"] < 0
        ):
            return cls.invalid("checkpoint has invalid budget fields")
        if "valid" in value and value["valid"] is not True:
            return cls.invalid("checkpoint is not marked valid")
        failed = value["failed_head_sha"]
        candidate = value["candidate_head_sha"]
        if candidate is not None and candidate == failed:
            return cls.invalid("checkpoint requalifies the failed head")
        try:
            return cls(
                obligation_kind=value["obligation_kind"],
                obligation_identity=value["obligation_identity"],
                lifecycle=value["lifecycle"],
                failed_head_sha=failed,
                candidate_head_sha=candidate,
                base_branch=value["base_branch"],
                approval_digest=value["approval_digest"],
                plan_digest=value["plan_digest"],
                requirements_digest=value["requirements_digest"],
                acquisition_digest=value["acquisition_digest"],
                scheduler_digest=value["scheduler_digest"],
                qualification_attempt_id=value["qualification_attempt_id"],
                watch_failure_extension_used=value["watch_failure_extension_used"],
                watch_head_extension_used=value["watch_head_extension_used"],
                allowed_rounds=value["allowed_rounds"],
                valid=True,
                failed_managed_run=(FailedManagedRun.from_mapping(value["failed_managed_run"])
                                    if "failed_managed_run" in value else None),
                verified_managed_result=(VerifiedManagedResult.from_mapping(value["verified_managed_result"])
                                         if "verified_managed_result" in value else None),
            )
        except (TypeError, ValueError):
            # A well-typed but contradictory persisted checkpoint must be
            # treated exactly like any other malformed checkpoint. Recovery
            # can then force a fresh board instead of making the whole ledger
            # unresumable or accidentally treating it as ready.
            return cls.invalid("checkpoint contains contradictory lifecycle fields")

    @classmethod
    def invalid(cls, reason: str) -> "QualificationCheckpoint":
        return cls(
            obligation_kind="unknown",
            obligation_identity="invalid-checkpoint",
            lifecycle="repair_required",
            failed_head_sha=None,
            candidate_head_sha=None,
            allowed_rounds=0,
            valid=False,
        )
