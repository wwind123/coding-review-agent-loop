"""Post a coder record whenever a rejected coder follow-up moved the head (#2086).

A coder follow-up that pushed and was then rejected used to leave the new head
with no coder record.  This module turns that rejection into a record:

* **degraded** -- the final rejected candidate is parsed, every droppable
  field (unknown item IDs, citations that do not select a passing receipt of
  the candidate's own acquisition, outside-checkout test commands, malformed
  ``architecture_impact``, missing or invalid human-requirement dispositions)
  is dropped or conservatively normalized with a visible note, and the result
  is revalidated by the unchanged validators under the candidate's frozen
  acquisition catalog;
* **synthesized** -- when nothing parses or a rejection names no droppable
  field, the record is built from git: old head, new head, a bounded diffstat
  and every carried item listed as not claimed.

Dropped fields never become claims: a dropped item ID is not addressed, a
dropped citation is not evidence and a normalized disposition is ``blocked``.
An unchanged head keeps today's rejection behaviour.

When even the minimal synthesized record cannot be transported (the ledger
exceeds a codec size limit, #2153), the pushed head gets a best-effort,
metadata-free notice listing every carried item as not claimed; it is never
admitted as a round record, and the run stops with the original error.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from .agent_failure import ValidatedAgentResponse
from .architecture_contract import _with_validation_context
from .errors import (
    TESTS_RUN_WORKDIR_REASON,
    AgentLoopError,
    CheckoutVerificationError,
    HumanDecisionRequiredError,
    NonRepairableEvidenceRejection,
    QuotaResetExceededError,
    RejectedCandidate,
    TestObservationsShapeError,
)
from .logging import log
from .protocol import StructuredCoderFollowup, _degradable_response_impact
from .protocol_common import (
    partition_item_id_buckets,
    SIGNATURE_RE,
    _extract_json_object_prefix,
    normalize_response_file_structured_text,
)
from .protocol_risk import _observation_semantics, _semantic_execution_ref
from .unresolved_items import HUMAN_REQUIREMENTS_ACK_ITEM_ID

MAX_DEGRADATION_STEPS = 8
DEGRADED_DISPOSITION_EVIDENCE = (
    "Orchestrator degradation: the coder response carried no valid disposition for this "
    "requirement; no coder claim about it was retained."
)
SYNTHESIZED_ID_LISTING_CHARS = 2000
SYNTHESIZED_DIFFSTAT_LINES = 40
SYNTHESIZED_DIFFSTAT_LINE_CHARS = 160
SYNTHESIZED_RECORD_HEADING = "## Orchestrator-synthesized coder record"
UNTRANSPORTABLE_NOTICE_REASON = "its record exceeded the round-metadata transport bound"
UNTRANSPORTABLE_NOTICE_TEXT = (
    "No round metadata could be attached to this notice, so it is not a round record: the "
    "ledger in the latest admitted record stays authoritative, and every item listed above "
    "stays open.\n\nThe PR cannot progress until an operator shrinks the ledger or closes "
    "items in it."
)
_NOTICE_ID_LISTING_TIERS = (2000, 500, 120, 0)
_SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9._:/#-]{1,80}$")


def _label(value: object) -> str:
    """A note-safe rendering of an untrusted identifier."""
    return value if isinstance(value, str) and _SAFE_LABEL_RE.fullmatch(value) else "(unprintable)"


def _labels(values: Sequence[object], limit: int = 8) -> str:
    shown = [f"`{_label(value)}`" for value in values[:limit]]
    if len(values) > limit:
        shown.append(f"and {len(values) - limit} more")
    return ", ".join(shown)


@dataclass(frozen=True)
class DegradedFollowup:
    text: str  # canonical sanitized response text, persisted for resume
    marker_value: StructuredCoderFollowup
    notes: tuple[str, ...]
    human_requirements_degraded: bool


def _observation_value(observation: object, name: str) -> object:
    if isinstance(observation, Mapping):
        return observation.get(name)
    return getattr(observation, name, None)


def _passes(observation: object) -> bool:
    try:
        semantics, _rich = _observation_semantics(observation)
    except Exception:  # noqa: BLE001 - an unreadable receipt is never evidence
        return False
    return semantics.get("outcome") == "passed"


def _drop_unknown_item_ids(payload: dict, allowed: frozenset[str], notes: list[str]) -> None:
    buckets = {
        bucket: payload[bucket]
        for bucket in ("addressed_items", "remaining_items", "disputed_items")
        if isinstance(payload.get(bucket), list)
    }
    kept, dropped = partition_item_id_buckets(buckets, allowed)
    payload.update(kept)
    if not dropped:
        return
    # A dropped ID takes every claim that depends on it.
    for notes_field in ("addressed_item_notes", "remaining_item_notes", "dispute_evidence"):
        mapping = payload.get(notes_field)
        if isinstance(mapping, dict):
            payload[notes_field] = {key: value for key, value in mapping.items() if key not in dropped}
    sub_items = payload.get("addressed_sub_items")
    if isinstance(sub_items, list):
        payload["addressed_sub_items"] = [
            entry for entry in sub_items
            if not (isinstance(entry, dict) and entry.get("item_id") in dropped)
        ]
    notes.append(f"Dropped unknown item ID(s) {_labels(dropped)}; they are not addressed.")


def _drop_inadmissible_citations(
    payload: dict, catalog: Sequence[object], notes: list[str]
) -> None:
    """Keep only citations selecting a passing receipt of the frozen acquisition."""
    passing_receipts = {
        _observation_value(observation, "receipt_id")
        for observation in catalog if _passes(observation)
    }
    passing_refs = {
        _semantic_execution_ref(observation)
        for observation in catalog
        if _passes(observation)
        and (_observation_semantics(observation)[0].get("provenance") == "parent-observed")
    }
    citations = payload.get("test_observations")
    if isinstance(citations, list):
        kept = [
            entry for entry in citations
            if isinstance(entry, dict) and entry.get("receipt_id") in passing_receipts
        ]
        if len(kept) != len(citations):
            dropped = [
                entry.get("receipt_id") if isinstance(entry, dict) else None
                for entry in citations if entry not in kept
            ]
            payload["test_observations"] = kept
            notes.append(
                f"Dropped test_observations citation(s) {_labels(dropped)} that select no passing "
                "receipt of the coder acquisition; they are not evidence."
            )
    claims = payload.get("risk_test_matrix_claims")
    if isinstance(claims, list):
        kept_claims: list[object] = []
        dropped_refs: list[object] = []
        for claim in claims:
            refs = claim.get("execution_refs") if isinstance(claim, dict) else None
            if not isinstance(refs, list):
                kept_claims.append(claim)  # shape defects stay with the validator
                continue
            admissible = [ref for ref in refs if ref in passing_refs]
            dropped_refs.extend(ref for ref in refs if ref not in passing_refs)
            if admissible:
                kept_claims.append({**claim, "execution_refs": admissible})
            elif not refs:
                kept_claims.append(claim)
        if dropped_refs:
            payload["risk_test_matrix_claims"] = kept_claims
            notes.append(
                f"Dropped risk-matrix selector(s) {_labels(dropped_refs)} that are not admissible "
                "passing observations of the coder acquisition; claims left without a selector "
                "were dropped."
            )


def _drop_malformed_architecture_impact(payload: dict, notes: list[str]) -> None:
    if "architecture_impact" not in payload:
        return
    try:
        _degradable_response_impact(
            payload, mode="strict", context="coder_followup.architecture_impact"
        )
        return
    except (AgentLoopError, TypeError, ValueError):
        pass
    payload.pop("architecture_impact", None)
    notes.append("Dropped a malformed architecture_impact; no architecture assessment is recorded.")


def _human_requirements_valid(
    payload: dict,
    *,
    surfaced_requirement_ids: Sequence[str],
    requires_direct_discussion_ack: bool,
    reviewer_owned_requirement_ids: Sequence[str],
) -> bool:
    from .protocol import (
        _expect_human_requirement_dispositions,
        validate_human_requirement_dispositions,
        validate_structured_human_requirements_acknowledgement,
    )

    try:
        acknowledgement = payload["human_requirements"]
        if not isinstance(acknowledgement, dict) or set(acknowledgement) != {
            "addressed_ids", "checked_discussion_directly"
        }:
            return False
        dispositions = _expect_human_requirement_dispositions(
            payload["human_requirement_dispositions"],
            context="coder_followup.human_requirement_dispositions",
        )
        addressed = acknowledgement["addressed_ids"]
        checked = acknowledgement["checked_discussion_directly"]
        if not isinstance(addressed, list) or not all(isinstance(value, str) for value in addressed):
            return False
        if not isinstance(checked, bool):
            return False
        validate_human_requirement_dispositions(
            dispositions,
            surfaced_requirement_ids=surfaced_requirement_ids,
            context="coder_followup.human_requirement_dispositions",
            reviewer_owned_requirement_ids=reviewer_owned_requirement_ids,
        )
        validate_structured_human_requirements_acknowledgement(
            addressed,
            dispositions=dispositions,
            checked_discussion_directly=checked,
            surfaced_requirement_ids=surfaced_requirement_ids,
            requires_direct_discussion_ack=requires_direct_discussion_ack,
            reviewer_owned_requirement_ids=reviewer_owned_requirement_ids,
            degraded_unacknowledged=True,
        )
    except (AgentLoopError, KeyError, TypeError):
        return False
    return True


def _normalize_human_requirements(
    payload: dict, *, surfaced_requirement_ids: Sequence[str], notes: list[str]
) -> None:
    """Replace missing or invalid dispositions with wire-valid ``blocked`` ones."""
    payload["human_requirement_dispositions"] = [
        {
            "requirement_id": requirement_id,
            "disposition": "blocked",
            "evidence": DEGRADED_DISPOSITION_EVIDENCE,
        }
        for requirement_id in surfaced_requirement_ids
    ]
    payload["human_requirements"] = {"addressed_ids": [], "checked_discussion_directly": False}
    if payload.get("state") == "approved":
        payload["state"] = "blocking"
    notes.append(
        "Replaced missing or invalid human_requirement_dispositions: every surfaced signed "
        "requirement is recorded as blocked and not addressed."
        if surfaced_requirement_ids
        else "Dropped invalid human_requirements fields; no signed requirement is addressed."
    )


def _render_candidate(payload: dict, signature: str) -> str:
    state = payload.get("state") if payload.get("state") in {"approved", "blocking"} else "blocking"
    return (
        json.dumps(payload, ensure_ascii=False)
        + f"\n<!-- AGENT_STATE: {state} -->\n{signature}"
    )


def _envelope_changed(trailing: object, rendered: str) -> bool:
    """Whether re-rendering replaced anything but whitespace after the JSON object."""
    rendered_trailer = rendered.split("\n", 1)[1] if "\n" in rendered else ""
    return " ".join(str(trailing).split()) != " ".join(rendered_trailer.split())


def _drop_for_rejection(
    error: AgentLoopError,
    payload: dict,
    *,
    tests_run_admissible: Callable[[str], bool] | None,
) -> str | None:
    """Drop the field a typed rejection names; return the note, or None."""
    if isinstance(error, TestObservationsShapeError) and "test_observations" in payload:
        payload.pop("test_observations")
        return "Dropped a malformed test_observations field; none of it is evidence."
    if not isinstance(error, NonRepairableEvidenceRejection):
        return None
    if error.reason == TESTS_RUN_WORKDIR_REASON and isinstance(payload.get("tests_run"), list):
        entries = payload["tests_run"]
        kept = [
            entry for entry in entries
            if isinstance(entry, str) and tests_run_admissible is not None and tests_run_admissible(entry)
        ]
        payload["tests_run"] = kept
        return (
            f"Dropped {len(entries) - len(kept)} of {len(entries)} tests_run entries run outside the assigned "
            "checkout or against a live target; they are not evidence."
        )
    if error.reason == "report-workdir" and "test_observations" in payload:
        payload.pop("test_observations")
        return "Dropped test_observations citing commands outside the assigned checkout."
    if error.reason in {"non-passing-selector", "launch-integrity"} and "risk_test_matrix_claims" in payload:
        payload.pop("risk_test_matrix_claims")
        return "Dropped risk_test_matrix_claims whose selectors are not admissible evidence."
    return None


def degrade_rejected_followup(
    candidate: RejectedCandidate,
    *,
    validator_factory: Callable[..., Callable[[str], object]],
    runner: object,
    allowed_item_ids: Sequence[str],
    surfaced_requirement_ids: Sequence[str] = (),
    requires_direct_discussion_ack: bool = False,
    reviewer_owned_requirement_ids: Sequence[str] = (),
    tests_run_admissible: Callable[[str], bool] | None = None,
) -> DegradedFollowup | None:
    """Return the degraded follow-up, or None to request git synthesis.

    Every revalidation runs inside ``_with_validation_context`` bound to the
    candidate's own acquisition and closed catalog, never the runner's latest
    test turn.  Any rejection that names no droppable field, or exceeding the
    step bound, returns None; a partially validated response is never used.
    """
    try:
        text, _status = normalize_response_file_structured_text(candidate.text)
        extracted = _extract_json_object_prefix(text)
    except AgentLoopError:
        return None
    if extracted is None:
        return None
    payload, trailing = extracted
    if not isinstance(payload, dict) or payload.get("kind") != "coder_followup":
        return None
    payload = copy.deepcopy(payload)
    signatures = [line.strip() for line in str(trailing).splitlines() if SIGNATURE_RE.match(line.strip())]
    signature = signatures[-1] if signatures else "-- Orchestrator"
    notes: list[str] = []
    allowed = frozenset((*allowed_item_ids, HUMAN_REQUIREMENTS_ACK_ITEM_ID))
    _drop_unknown_item_ids(payload, allowed, notes)
    _drop_inadmissible_citations(payload, candidate.test_turn_observations, notes)
    _drop_malformed_architecture_impact(payload, notes)
    human_requirements_degraded = False
    requirement_context = dict(
        surfaced_requirement_ids=tuple(surfaced_requirement_ids),
        requires_direct_discussion_ack=requires_direct_discussion_ack,
        reviewer_owned_requirement_ids=tuple(reviewer_owned_requirement_ids),
    )
    if not _human_requirements_valid(payload, **requirement_context):
        _normalize_human_requirements(
            payload, surfaced_requirement_ids=surfaced_requirement_ids, notes=notes
        )
        human_requirements_degraded = bool(surfaced_requirement_ids or requires_direct_discussion_ack)
    acknowledgement = payload.get("human_requirements")
    unacknowledged = (
        not surfaced_requirement_ids
        and requires_direct_discussion_ack
        and not (isinstance(acknowledgement, dict) and acknowledgement.get("checked_discussion_directly") is True)
    )
    if unacknowledged:
        # Never fabricated: recorded as not acknowledged.
        human_requirements_degraded = True
        notes.append("The direct-discussion acknowledgement is missing; recorded as not acknowledged.")
    validate = validator_factory("degradable", "degrade", degraded_unacknowledged=unacknowledged)
    acquisition = SimpleNamespace(
        test_turn_id=candidate.test_turn_id or "",
        test_turn_observations=tuple(candidate.test_turn_observations),
    )
    for _step in range(MAX_DEGRADATION_STEPS):
        rendered = _render_candidate(payload, signature)
        try:
            result = _with_validation_context(validate, runner=runner, acquisition=acquisition)(rendered)
        except AgentLoopError as error:
            note = _drop_for_rejection(error, payload, tests_run_admissible=tests_run_admissible)
            if note is None:
                return None
            notes.append(note)
            continue
        if not isinstance(result, StructuredCoderFollowup):
            return None
        if result.architecture_impact_degradations and "architecture_impact" in payload:
            payload.pop("architecture_impact")
            notes.append("Dropped a degraded architecture_impact; no architecture assessment is recorded.")
            continue
        if not notes and _envelope_changed(trailing, rendered):
            # Only the envelope was invalid (prose around the footer): the
            # canonical re-rendering keeps every structured claim.
            notes.append("Dropped text outside the structured response envelope.")
        if not notes:
            # The field validators accept the candidate, so the turn was refused
            # at the seam's architecture contract (missing, undetermined or an
            # out-of-set document path): drop the assessment, never the claims.
            if "architecture_impact" in payload:
                payload.pop("architecture_impact")
                notes.append("Dropped an architecture_impact the response contract refused.")
                continue
            notes.append("The response carried no architecture_impact; no architecture assessment is recorded.")
        return DegradedFollowup(rendered, result, tuple(notes), human_requirements_degraded)
    return None


def tests_run_entry_in_checkout(entry: str, workdir: Path) -> bool:
    """Whether one reported ``tests_run`` entry ran inside the assigned checkout."""
    from .workdir_guard import partition_reported_tests_by_workdir

    try:
        partition = partition_reported_tests_by_workdir([entry], assigned_workdir=workdir)
    except AgentLoopError:
        return False
    return not partition.out_of_checkout


def bounded_diffstat(runner: object, workdir: Path, old_head: str, new_head: str) -> str:
    """A bounded ``git diff --stat`` for the synthesized record; never raises."""
    try:
        result = runner.run(  # type: ignore[attr-defined]
            ["git", "diff", "--stat", old_head, new_head], cwd=workdir, check=False
        )
    except Exception:  # noqa: BLE001 - display-only
        return "(diffstat unavailable)"
    if getattr(result, "returncode", 1) != 0 or not str(getattr(result, "stdout", "")).strip():
        return "(diffstat unavailable)"
    return bound_diffstat_text(str(result.stdout))


def bound_diffstat_text(text: str, max_lines: int = SYNTHESIZED_DIFFSTAT_LINES) -> str:
    lines = [line.rstrip()[:SYNTHESIZED_DIFFSTAT_LINE_CHARS] for line in text.strip().splitlines()]
    lines = [line.replace("`", "'") for line in lines]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    omitted = len(lines) - max_lines
    return "\n".join([*lines[: max_lines - 1], f"... {omitted} more line(s) omitted", lines[-1]])


def synthesize_coder_record_text(
    *,
    old_head: str | None,
    new_head: str | None,
    item_ids: Sequence[str],
    diffstat: str,
    reason: str = "its response could not be validated",
    id_listing_chars: int = SYNTHESIZED_ID_LISTING_CHARS,
) -> str:
    """The visible body of a git-synthesized coder record: no coder claims."""
    shown: list[str] = []
    used = 0
    for item_id in item_ids:
        rendered = f"`{_label(item_id)}`"
        if used + len(rendered) + 2 > id_listing_chars:
            break
        shown.append(rendered)
        used += len(rendered) + 2
    omitted = len(item_ids) - len(shown)
    listing = ", ".join(shown) if shown else "(none)"
    if omitted:
        listing += f" ({omitted} more item(s) omitted from this listing; all remain open)"
    return "\n".join([
        SYNTHESIZED_RECORD_HEADING,
        "",
        f"The coder pushed `{_label(new_head)}` (from `{_label(old_head)}`), but {reason}, so the "
        "orchestrator synthesized this coder record from git. It carries no coder claims.",
        "",
        "**Diffstat:**",
        "",
        "```",
        diffstat or "(empty)",
        "```",
        "",
        f"**Not claimed (every carried item stays open):** {listing}",
        "",
        "<!-- AGENT_STATE: blocking -->",
    ])


class CoderFollowupRecovery:
    """The coder follow-up seam's post-push recovery for one dispatch (#2086)."""

    def __init__(
        self,
        runner: object,
        *,
        config: object,
        pr_number: int,
        pre_turn_head: str,
        ledger: Sequence[object],
        coder_name: str,
        record_round: int,
        validator_factory: Callable[..., Callable[[str], object]],
        human_requirements_context: object,
        reviewer_owned_requirement_ids: Sequence[str] = (),
        allowed_item_ids: Sequence[str] = (),
        workdir: Path,
        observe_head: Callable[[], str | None],
        tests_run_admissible: Callable[[str], bool] | None = None,
        metadata_factory: Callable[..., object] | None = None,
        post_body: Callable[[str], object] | None = None,
        qualification_checkpoint: object = None,
    ) -> None:
        self.runner, self.config, self.pr_number = runner, config, pr_number
        self.pre_turn_head, self.ledger, self.coder_name = pre_turn_head, tuple(ledger), coder_name
        self.record_round, self.validator_factory = record_round, validator_factory
        self.surfaced_requirement_ids = tuple(
            getattr(human_requirements_context, "surfaced_requirement_ids", ()) or ()
        )
        self.requires_direct_discussion_ack = bool(
            getattr(human_requirements_context, "requires_direct_discussion_ack", False)
        )
        self.reviewer_owned_requirement_ids = tuple(reviewer_owned_requirement_ids)
        self.allowed_item_ids = tuple(allowed_item_ids)
        self.workdir, self.observe_head = workdir, observe_head
        self.tests_run_admissible = tests_run_admissible or (
            lambda entry: tests_run_entry_in_checkout(entry, workdir)
        )
        self.metadata_factory, self.post_body = metadata_factory, post_body
        self.qualification_checkpoint = qualification_checkpoint
        self.source: str | None = None
        self.notes: tuple[str, ...] = ()
        self.human_requirements_degraded = False
        self.original_error: BaseException | None = None
        self.record_posted = False
        # Set once a best-effort post was attempted (success or failure), so
        # a later failure path never repeats it (#2153).
        self.record_attempted = False

    @property
    def requirements_in_force(self) -> bool:
        return bool(self.surfaced_requirement_ids or self.requires_direct_discussion_ack)

    def _moved_head(self) -> str | None:
        try:
            head = (self.observe_head() or "").strip()
        except Exception as error:  # noqa: BLE001 - an unobservable head keeps today's path
            log(self.config, f"PR #{self.pr_number}: could not observe the head after a rejected coder turn ({error})")
            return None
        return head if head and head != self.pre_turn_head else None

    def _item_ids(self) -> tuple[str, ...]:
        return tuple(
            item.item_id for item in self.ledger
            if getattr(item, "status", "blocking") in {"blocking", "same-pr", "future"}
        )

    def synthesized_text(self, new_head: str, reason: str = "its response could not be validated") -> str:
        return synthesize_coder_record_text(
            old_head=self.pre_turn_head, new_head=new_head, item_ids=self._item_ids(),
            diffstat=bounded_diffstat(self.runner, self.workdir, self.pre_turn_head, new_head),
            reason=reason,
        )

    def run(self, invoke: Callable[[], ValidatedAgentResponse]) -> ValidatedAgentResponse:
        """Run the coder turn; turn a post-push rejection into a coder record."""
        from .pr_head_sync import PrHeadObservationError

        # Fail-closed errors mean the head or the checkout cannot be trusted:
        # they surface untouched, with no record, diffstat or review (#2086).
        fail_closed = (PrHeadObservationError, CheckoutVerificationError)
        try:
            return invoke()
        except fail_closed:
            raise
        except (QuotaResetExceededError, HumanDecisionRequiredError, KeyboardInterrupt) as error:
            self._post_synthesized_best_effort(error)
            raise
        except AgentLoopError as error:
            response = self._recover(error)
            if response is None:
                raise
            return response

    def _recover(self, error: AgentLoopError) -> ValidatedAgentResponse | None:
        new_head = self._moved_head()
        if new_head is None:
            return None  # unchanged head: today's behaviour
        candidate = getattr(error, "rejected_candidate", None)
        degraded = None
        if isinstance(candidate, RejectedCandidate):
            try:
                degraded = degrade_rejected_followup(
                    candidate,
                    validator_factory=self.validator_factory,
                    runner=self.runner,
                    allowed_item_ids=self.allowed_item_ids,
                    surfaced_requirement_ids=self.surfaced_requirement_ids,
                    requires_direct_discussion_ack=self.requires_direct_discussion_ack,
                    reviewer_owned_requirement_ids=self.reviewer_owned_requirement_ids,
                    tests_run_admissible=self.tests_run_admissible,
                )
            except Exception as secondary:  # noqa: BLE001 - synthesis is the fail-closed fallback
                log(self.config, f"PR #{self.pr_number}: degradation failed ({secondary}); synthesizing from git")
                degraded = None
        self.original_error = error
        session_id = getattr(candidate, "session_id", None)
        model_used = getattr(candidate, "model_used", None)
        if degraded is not None:
            self.source, self.notes = "degraded", degraded.notes
            self.human_requirements_degraded = degraded.human_requirements_degraded
            log(self.config, f"PR #{self.pr_number}: rejected coder follow-up pushed `{new_head}`; "
                f"posting a degraded coder record ({len(degraded.notes)} note(s))")
            return ValidatedAgentResponse(
                text=degraded.text, session_id=session_id, marker_value=degraded.marker_value,
                model_used=model_used,
                acquisition_test_turn_id=candidate.test_turn_id,
                acquisition_test_observations=tuple(candidate.test_turn_observations),
            )
        self.source = "synthesized"
        self.notes = ("The coder response could not be parsed or degraded; this record was synthesized from git.",)
        self.human_requirements_degraded = self.requirements_in_force
        log(self.config, f"PR #{self.pr_number}: rejected coder follow-up pushed `{new_head}`; "
            "posting a git-synthesized coder record")
        return ValidatedAgentResponse(
            text=self.synthesized_text(new_head), session_id=session_id, marker_value="blocking",
            model_used=model_used,
        )

    def _post_synthesized_best_effort(self, error: BaseException) -> None:
        """Record a pushed head before a stopping error re-raises; never raises."""
        if self.metadata_factory is None or self.post_body is None:
            return
        try:
            new_head = self._moved_head()
            if new_head is None:
                return
            self.record_attempted = True
            from .architecture_contract import _architecture_metadata_fields
            from .coder_comment_budget import CoderRecordUntransportableError, bounded_coder_round_comment
            from .round_metadata import _is_followup_dispatch_head

            # An earlier degraded or synthesized recovery keeps its notes, so
            # the record still says the response was degraded (#2153).
            stop_note = (
                f"The coder turn stopped ({type(error).__name__}) after pushing; "
                "this record was synthesized from git."
            )
            prior_notes = self.notes if self.source is not None else ()
            self.source = "synthesized"
            self.notes = tuple(dict.fromkeys((*prior_notes, stop_note)))
            self.human_requirements_degraded = self.requirements_in_force
            text = self.synthesized_text(new_head, reason="the turn stopped before a response was accepted")
            metadata = self.metadata_factory(
                flow="pr", role="coder", agent=self.coder_name, round_number=self.record_round,
                subject=new_head, prior_items=self.ledger,
                qualification_checkpoint=self.qualification_checkpoint,
                followup_dispatch_head=(
                    self.pre_turn_head if _is_followup_dispatch_head(self.pre_turn_head) else None
                ),
                **_architecture_metadata_fields(self.config),  # type: ignore[arg-type]
                **self.metadata_fields(),
            )
            try:
                bounded = bounded_coder_round_comment(
                    text, metadata, pr_number=self.pr_number, head_sha=new_head, freeform_text=text,
                    **self.budget_kwargs(),
                )
            except CoderRecordUntransportableError as untransportable:
                self.post_untransportable_notice(untransportable)
                return
            self.post_body(bounded.posted_body)
            self.record_posted = True
        except Exception as secondary:  # noqa: BLE001 - never mask the stopping error
            log(self.config, f"PR #{self.pr_number}: could not post a synthesized coder record ({secondary})")

    def post_untransportable_notice(self, error: BaseException) -> None:
        """Post a claim-free, metadata-free notice for an untransportable record (#2153).

        The notice carries no reserved protocol marker and no round metadata,
        so it is never admitted as a round record: resume keeps the latest
        admitted ledger with every item open.  Never raises.
        """
        if self.post_body is None:
            return
        try:
            new_head = self._moved_head()
            if new_head is None:
                return
            self.record_attempted = True
            from .round_transport import MAX_GITHUB_BODY_CHARS

            diffstat = bounded_diffstat(self.runner, self.workdir, self.pre_turn_head, new_head)
            body = ""
            for id_chars in _NOTICE_ID_LISTING_TIERS:
                text = synthesize_coder_record_text(
                    old_head=self.pre_turn_head, new_head=new_head, item_ids=self._item_ids(),
                    diffstat=diffstat if id_chars else "(omitted)",
                    reason=UNTRANSPORTABLE_NOTICE_REASON, id_listing_chars=id_chars,
                )
                lines = text.rstrip("\n").splitlines()
                if lines and lines[-1].startswith("<!--"):
                    lines.pop()
                body = "\n".join(lines).rstrip() + "\n\n" + UNTRANSPORTABLE_NOTICE_TEXT
                if len(body) <= MAX_GITHUB_BODY_CHARS:
                    break
            self.post_body(body)
            self.record_posted = True
            log(self.config, f"PR #{self.pr_number}: the coder record for `{new_head}` cannot be "
                f"transported ({error}); posted a metadata-free notice")
        except Exception as secondary:  # noqa: BLE001 - never mask the original error
            log(self.config, f"PR #{self.pr_number}: could not post the untransportable-record notice ({secondary})")

    def bound(self, make_bounded: Callable[[], object]) -> object:
        """Bound the ordinary coder record; an untransportable one gets the notice (#2153).

        Re-raises the recovery's original error when one is set (as ``post``
        does), otherwise the untransportable error itself.
        """
        from .coder_comment_budget import CoderRecordUntransportableError

        try:
            return make_bounded()
        except CoderRecordUntransportableError as untransportable:
            self.post_untransportable_notice(untransportable)
            if self.original_error is not None:
                raise self.original_error from untransportable
            raise

    def record_failed_push(self, error: BaseException) -> None:
        """Record a pushed head a later failure left without a coder record (#2087).

        A failure raised after the coder turn returned (for example evidence
        derivation) used to be covered by the legacy rejected-follow-up record.
        Now the head-moving turn gets a best-effort synthesized coder record,
        unless a record was already posted or a best-effort post was already
        attempted for this turn, or the error is fail-closed (the head or
        checkout cannot be trusted).  A degraded or synthesized recovery whose
        turn fails before its record is posted is covered too (#2153).  Never
        raises: the caller re-raises the original error.
        """
        from .pr_head_sync import PrHeadObservationError

        if self.record_posted or self.record_attempted:
            return
        if isinstance(error, (PrHeadObservationError, CheckoutVerificationError)):
            return
        self._post_synthesized_best_effort(error)

    def metadata_fields(self) -> dict[str, object]:
        if self.source is None:
            return {}
        return {
            "coder_record_source": self.source,
            "coder_record_degradations": tuple(
                note[:300] for note in self.notes[:16]
            ),
            "human_requirements_degraded": self.human_requirements_degraded,
        }

    def budget_kwargs(self) -> dict[str, object]:
        """Inputs for the coder comment budget's synthesized fallback."""
        return {
            "fallback_old_head": self.pre_turn_head,
            "fallback_diffstat": lambda new_head: bounded_diffstat(
                self.runner, self.workdir, self.pre_turn_head, new_head
            ),
            "human_requirements_in_force": self.requirements_in_force,
            "fallback_ledger": self.discarded_response_ledger,
        }

    def adopt(self, bounded: object, metadata: object, output: str | None, items: Sequence[object]):
        """Adopt the posted coder record; return ``(metadata, output, ledger, discarded)``.

        ``discarded`` is true when bounding fell back to a claim-free synthesized
        record: the caller must then drop every other in-memory effect of the
        response (fix history, cited-evidence progress, citation diagnosis),
        exactly as a resume from the stored record would see none of them.
        """
        adopted_metadata, adopted_output = adopt_bounded_coder_comment(bounded, metadata, output)
        if getattr(bounded, "coder_output", None) is None:
            return adopted_metadata, adopted_output, items, False
        return adopted_metadata, adopted_output, self.discarded_response_ledger(items), True

    def discarded_response_ledger(self, items: Sequence[object]) -> list:
        """The ledger once a response is discarded for a claim-free fallback (#2086).

        Every ledger change the discarded response's claims caused is undone,
        keeping unrelated changes: its coder-claim notes are removed, and while
        signed requirements are in force the acknowledgement obligation it may
        have cleared is open again (the pre-turn item when one existed).
        """
        rolled = rollback_discarded_annotations(items, self.ledger)
        if not self.requirements_in_force:
            return rolled
        from .unresolved_items import (
            HUMAN_REQUIREMENTS_DEGRADED_ACK_TEXT,
            _upsert_human_requirements_ack_item,
        )

        prior = next(
            (item for item in self.ledger if getattr(item, "item_id", None) == HUMAN_REQUIREMENTS_ACK_ITEM_ID),
            None,
        )
        if prior is not None:
            return [item for item in rolled if item.item_id != HUMAN_REQUIREMENTS_ACK_ITEM_ID] + [prior]
        return _upsert_human_requirements_ack_item(
            rolled, source_round=max(self.record_round - 1, 1), text=HUMAN_REQUIREMENTS_DEGRADED_ACK_TEXT,
        )

    def post(self, publish: Callable[[], object]) -> object:
        """Post the coder record; a failed degraded post surfaces the original error."""
        if self.source is None:
            result = publish()
            self.record_posted = True
            return result
        try:
            result = publish()
            self.record_posted = True
            return result
        except Exception as secondary:
            log(self.config, f"PR #{self.pr_number}: could not post the {self.source} coder record "
                f"({secondary}); surfacing the original coder error")
            if self.original_error is not None:
                raise self.original_error from secondary
            raise


def rollback_discarded_annotations(items: Sequence[object], pre_turn_ledger: Sequence[object]) -> list:
    """Remove ledger annotations a discarded coder response introduced (#2086).

    Applying a response's claims to the ledger (today: dispute counter-evidence
    notes) happens before the record is bounded.  When the record falls back to
    a claim-free synthesized one, every coder-claim note an item did not carry
    in the pre-turn ledger is removed; all other ledger state is kept.
    """
    from dataclasses import replace as _replace

    from .unresolved_items import CODER_CLAIM_NOTE_PREFIXES

    before = {getattr(item, "item_id", None): tuple(getattr(item, "notes", ())) for item in pre_turn_ledger}
    result = []
    for item in items:
        prior = before.get(item.item_id, ())
        notes = tuple(
            note for note in item.notes
            if note in prior or not note.startswith(CODER_CLAIM_NOTE_PREFIXES)
        )
        result.append(item if notes == item.notes else _replace(item, notes=notes))
    return result


def adopt_bounded_coder_comment(bounded: object, metadata: object, output: str | None):
    """Adopt a synthesized fallback's metadata and claim-free output (#2086)."""
    fallback_output = getattr(bounded, "coder_output", None)
    if fallback_output is None:
        return metadata, output
    return getattr(bounded, "metadata", metadata), fallback_output


__all__ = [
    "CoderFollowupRecovery",
    "DEGRADED_DISPOSITION_EVIDENCE",
    "DegradedFollowup",
    "adopt_bounded_coder_comment",
    "bound_diffstat_text",
    "bounded_diffstat",
    "degrade_rejected_followup",
    "synthesize_coder_record_text",
]
