"""Scheduler metadata codecs, re-exported by round_state."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from .errors import AgentLoopError
from .review_scheduling import FORCE_FULL_SOURCES, ReviewSchedulingContract, SCHEDULER_PHASES


_SCHEDULER_METADATA_KEYS = frozenset(
    {
        "scheduler_contract",
        "scheduler_previous_sha",
        "scheduler_current_sha",
        "scheduler_obligation_digest",
        "scheduler_selected_reviewers",
        "scheduler_paused_reviewers",
        "scheduler_reasons",
        "scheduler_final_sweep",
        "scheduler_force_full",
        "scheduler_calls_avoided",
    }
)
# Planning-only optional checkpoint keys (#1112).  Deliberately outside the
# shared auxiliary set and the serializer's presence test.
_PLAN_ONLY_SCHEDULER_KEYS = frozenset(
    {
        "scheduler_issue_digest",
        "scheduler_stall_reset",
        "scheduler_step_back_deferral",
        "scheduler_execution_mode",
    }
)


def _decode_execution_mode(value: object) -> str | None:
    """Tolerant decode: anything but a known execution mode is None (#1268)."""
    from .config import PLAN_EXECUTION_MODES

    return value if isinstance(value, str) and value in PLAN_EXECUTION_MODES else None

_SCHEDULER_AUXILIARY_KEYS = frozenset(
    {
        "scheduler_phase",
        "scheduler_primary_reviewer",
        "scheduler_approved_reviewers",
        "scheduler_active_owners",
        "scheduler_scope_digest",
        "scheduler_force_full_source",
        "scheduler_plan_previous_key",
    }
)

# Planning scheduler records carry the candidate key in place of the PR-only
# SHA pair, and omit the PR-only broad-rule/scope-digest state entirely.
_PLAN_SCHEDULER_METADATA_KEYS = frozenset(
    {
        "scheduler_contract",
        "scheduler_obligation_digest",
        "scheduler_selected_reviewers",
        "scheduler_paused_reviewers",
        "scheduler_reasons",
        "scheduler_final_sweep",
        "scheduler_force_full",
        "scheduler_calls_avoided",
    }
)
_PR_ONLY_SCHEDULER_KEYS = frozenset(
    {"scheduler_previous_sha", "scheduler_current_sha", "scheduler_scope_digest"}
)


def _decode_common_scheduler_lists(
    payload: Mapping[str, object], contract_reviewers: Sequence[str]
) -> dict[str, object]:
    """Validate the reviewer/reason/counter fields shared by both flows."""
    selected = payload["scheduler_selected_reviewers"]
    paused = payload["scheduler_paused_reviewers"]
    reasons = payload["scheduler_reasons"]
    required_reviewers = set(contract_reviewers)
    if not isinstance(selected, list) or any(not isinstance(name, str) for name in selected):
        raise ValueError("invalid selected reviewer list")
    if len(set(selected)) != len(selected) or not set(selected).issubset(required_reviewers):
        raise ValueError("contradictory selected reviewer list")
    if not isinstance(paused, list):
        raise ValueError("invalid paused reviewer list")
    paused_pairs: list[tuple[str, str]] = []
    for pair in paused:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not isinstance(pair[0], str)
            or not isinstance(pair[1], str)
            or not pair[0]
            or not pair[1]
        ):
            raise ValueError("invalid paused reviewer entry")
        paused_pairs.append((pair[0], pair[1]))
    paused_names = [name for name, _reason in paused_pairs]
    if (
        len(set(paused_names)) != len(paused_names)
        or not set(paused_names).issubset(required_reviewers)
        or set(selected) & set(paused_names)
        or set(selected) | set(paused_names) != required_reviewers
    ):
        raise ValueError("contradictory reviewer scheduling state")
    if not isinstance(reasons, list) or any(
        not isinstance(reason, str) or not reason for reason in reasons
    ):
        raise ValueError("invalid scheduler reasons")
    final_sweep = payload["scheduler_final_sweep"]
    force_full = payload["scheduler_force_full"]
    calls_avoided = payload["scheduler_calls_avoided"]
    if not isinstance(final_sweep, bool) or not isinstance(force_full, bool):
        raise ValueError("invalid scheduler boolean")
    if isinstance(calls_avoided, bool) or not isinstance(calls_avoided, int) or calls_avoided < 0:
        raise ValueError("invalid avoided-call count")
    approved = payload.get("scheduler_approved_reviewers", [])
    if (
        not isinstance(approved, list)
        or any(not isinstance(name, str) for name in approved)
        or len(set(approved)) != len(approved)
        or not set(approved).issubset(required_reviewers)
    ):
        raise ValueError("invalid scheduler approval set")
    owners = payload.get("scheduler_active_owners", [])
    if (
        not isinstance(owners, list)
        or any(not isinstance(name, str) or not name for name in owners)
        or len(set(owners)) != len(owners)
    ):
        raise ValueError("invalid scheduler owner set")
    force_full_source = payload.get("scheduler_force_full_source")
    if force_full_source is not None and (
        not isinstance(force_full_source, str)
        or force_full_source not in FORCE_FULL_SOURCES
        or force_full is not True
    ):
        raise ValueError("invalid scheduler force-full source")
    return {
        "scheduler_selected_reviewers": tuple(selected),
        "scheduler_paused_reviewers": tuple(paused_pairs),
        "scheduler_reasons": tuple(reasons),
        "scheduler_final_sweep": final_sweep,
        "scheduler_force_full": force_full,
        "scheduler_calls_avoided": calls_avoided,
        "scheduler_approved_reviewers": tuple(approved),
        "scheduler_active_owners": tuple(owners),
        "scheduler_force_full_source": force_full_source,
    }


def _decode_plan_scheduler_fields(payload: Mapping[str, object]) -> dict[str, object]:
    """Decode a planning scheduler record through its own contract type.

    Keyed on the record's ``flow`` value so a planning record and a PR record
    can never be mistaken for one another even when both carry
    ``policy: primary-then-panel``.  The outcome vocabulary is deliberately the
    same three values the PR branch uses.
    """
    from .plan_review_scheduling import (
        PLAN_SCHEDULER_PHASES,
        PlanCandidateKey,
        PlanReviewSchedulingContract,
    )

    # Named all-reviewers plans persist a board contract for signed amendment
    # lineage without enabling scheduler selection or carried approvals. The
    # contract is accepted only with a bound seat board and no scheduler vote.
    if (
        isinstance(payload.get("seat_binding"), dict)
        and payload.get("scheduler_contract") is not None
    ):
        try:
            board = PlanReviewSchedulingContract.from_mapping(payload["scheduler_contract"])
        except (AgentLoopError, TypeError, ValueError, KeyError):
            board = None
        if board is not None and board.policy == "all-reviewers" and board.primary_reviewer is None:
            neutral = {
                "scheduler_obligation_digest": None,
                "scheduler_selected_reviewers": [],
                "scheduler_paused_reviewers": [],
                "scheduler_reasons": [],
                "scheduler_final_sweep": None,
                "scheduler_force_full": None,
                "scheduler_calls_avoided": None,
            }
            auxiliary = (
                _SCHEDULER_AUXILIARY_KEYS
                | _PLAN_ONLY_SCHEDULER_KEYS
                | _PR_ONLY_SCHEDULER_KEYS
            )
            if all(payload.get(key) == value for key, value in neutral.items()) and not any(
                payload.get(key) is not None for key in auxiliary
            ):
                return {
                    "scheduler_contract": board.as_dict(),
                    "scheduler_metadata_status": "valid",
                }

    if not _PLAN_SCHEDULER_METADATA_KEYS.issubset(payload.keys()):
        return {"scheduler_metadata_status": "invalid"}
    if any(payload.get(key) is not None for key in _PR_ONLY_SCHEDULER_KEYS):
        # PR-only state on a planning record is internally contradictory.
        return {"scheduler_metadata_status": "invalid"}
    try:
        contract = PlanReviewSchedulingContract.from_mapping(payload["scheduler_contract"])
        digest = payload["scheduler_obligation_digest"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{16}", digest):
            raise ValueError("invalid scheduler obligation digest")
        current_key_payload = payload.get("plan_candidate_key")
        if not isinstance(current_key_payload, dict):
            raise ValueError("planning scheduler record is missing its candidate key")
        # Generation-1 rule: a record missing any key component, or carrying a
        # non-1 execution-strategy contract version, can supply neither an
        # approval nor a panel opening, so it must not decode as valid.
        if not PlanCandidateKey.from_mapping(current_key_payload).complete:
            raise ValueError("incomplete planning candidate key")
        previous_key_payload = payload.get("scheduler_plan_previous_key")
        if previous_key_payload is not None:
            if not isinstance(previous_key_payload, dict):
                raise ValueError("invalid previous planning candidate key")
            if not PlanCandidateKey.from_mapping(previous_key_payload).complete:
                raise ValueError("incomplete previous planning candidate key")
        decoded = _decode_common_scheduler_lists(payload, contract.required_reviewers)
        phase = payload.get("scheduler_phase")
        if phase is not None and (
            not isinstance(phase, str) or phase not in PLAN_SCHEDULER_PHASES
        ):
            raise ValueError("invalid planning scheduler phase")
        primary = payload.get("scheduler_primary_reviewer")
        if primary is not None and (
            not isinstance(primary, str) or primary != contract.primary_reviewer
        ):
            raise ValueError("contradictory planning scheduler primary reviewer")
        issue_digest = payload.get("scheduler_issue_digest")
        if "scheduler_issue_digest" in payload and (
            not isinstance(issue_digest, str)
            or not re.fullmatch(r"[0-9a-f]{16}", issue_digest)
        ):
            raise ValueError("invalid scheduler issue digest")
        stall_reset = False
        if "scheduler_stall_reset" in payload:
            if payload["scheduler_stall_reset"] is not True or phase != "primary":
                raise ValueError("invalid scheduler stall reset marker")
            stall_reset = True
        step_back_deferral = False
        if "scheduler_step_back_deferral" in payload:
            if payload["scheduler_step_back_deferral"] is not True or phase != "primary":
                raise ValueError("invalid scheduler step-back deferral marker")
            step_back_deferral = True
    except (AgentLoopError, TypeError, ValueError, KeyError):
        return {"scheduler_metadata_status": "invalid"}
    return {
        "scheduler_issue_digest": issue_digest,
        "scheduler_execution_mode": _decode_execution_mode(
            payload.get("scheduler_execution_mode")
        ),
        "scheduler_stall_reset": stall_reset,
        "scheduler_step_back_deferral": step_back_deferral,
        "scheduler_contract": contract.as_dict(),
        "scheduler_obligation_digest": digest,
        "scheduler_phase": phase,
        "scheduler_primary_reviewer": primary,
        "scheduler_plan_previous_key": previous_key_payload,
        **decoded,
        "scheduler_metadata_status": "valid",
    }


def _decode_scheduler_fields(payload: Mapping[str, object]) -> dict[str, object]:
    """Decode scheduler metadata, dropping it conservatively when unsafe.

    Scheduler records are an optimization over the required full-board review.
    A malformed or partial record therefore becomes legacy metadata instead of
    being allowed to select a smaller reviewer set during resume.
    """
    scheduler_keys = (
        _SCHEDULER_METADATA_KEYS | _SCHEDULER_AUXILIARY_KEYS | _PLAN_ONLY_SCHEDULER_KEYS
    )
    if not (scheduler_keys & payload.keys()):
        return {"scheduler_metadata_status": "absent"}
    if payload.get("flow") == "plan":
        return _decode_plan_scheduler_fields(payload)
    if _PLAN_ONLY_SCHEDULER_KEYS & payload.keys():
        # Planning-only state on a PR-flow record is contradictory.
        return {"scheduler_metadata_status": "invalid"}
    required = _SCHEDULER_METADATA_KEYS
    if not required.issubset(payload.keys()):
        return {"scheduler_metadata_status": "invalid"}
    try:
        contract = ReviewSchedulingContract.from_mapping(payload["scheduler_contract"])
        previous = payload["scheduler_previous_sha"]
        current = payload["scheduler_current_sha"]
        digest = payload["scheduler_obligation_digest"]
        if previous is not None and (not isinstance(previous, str) or not previous):
            raise ValueError("invalid previous scheduler SHA")
        if not isinstance(current, str) or not current:
            raise ValueError("invalid current scheduler SHA")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{16}", digest):
            raise ValueError("invalid scheduler obligation digest")
        selected = payload["scheduler_selected_reviewers"]
        paused = payload["scheduler_paused_reviewers"]
        reasons = payload["scheduler_reasons"]
        if not isinstance(selected, list) or any(not isinstance(name, str) for name in selected):
            raise ValueError("invalid selected reviewer list")
        if len(set(selected)) != len(selected) or not set(selected).issubset(contract.required_reviewers):
            raise ValueError("contradictory selected reviewer list")
        if not isinstance(paused, list):
            raise ValueError("invalid paused reviewer list")
        paused_pairs: list[tuple[str, str]] = []
        for pair in paused:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not isinstance(pair[0], str)
                or not isinstance(pair[1], str)
                or not pair[0]
                or not pair[1]
            ):
                raise ValueError("invalid paused reviewer entry")
            paused_pairs.append((pair[0], pair[1]))
        paused_names = [name for name, _reason in paused_pairs]
        if (
            len(set(paused_names)) != len(paused_names)
            or not set(paused_names).issubset(contract.required_reviewers)
            or set(selected) & set(paused_names)
            or set(selected) | set(paused_names) != set(contract.required_reviewers)
        ):
            raise ValueError("contradictory reviewer scheduling state")
        if not isinstance(reasons, list) or any(not isinstance(reason, str) or not reason for reason in reasons):
            raise ValueError("invalid scheduler reasons")
        final_sweep = payload["scheduler_final_sweep"]
        force_full = payload["scheduler_force_full"]
        calls_avoided = payload["scheduler_calls_avoided"]
        if not isinstance(final_sweep, bool) or not isinstance(force_full, bool):
            raise ValueError("invalid scheduler boolean")
        if isinstance(calls_avoided, bool) or not isinstance(calls_avoided, int) or calls_avoided < 0:
            raise ValueError("invalid avoided-call count")
        phase = payload.get("scheduler_phase")
        if phase is not None and (not isinstance(phase, str) or phase not in SCHEDULER_PHASES):
            raise ValueError("invalid scheduler phase")
        primary = payload.get("scheduler_primary_reviewer")
        if primary is not None and (
            not isinstance(primary, str) or primary != contract.primary_reviewer
        ):
            raise ValueError("contradictory scheduler primary reviewer")
        approved = payload.get("scheduler_approved_reviewers", [])
        if (
            not isinstance(approved, list)
            or any(not isinstance(name, str) for name in approved)
            or len(set(approved)) != len(approved)
            or not set(approved).issubset(contract.required_reviewers)
        ):
            raise ValueError("invalid scheduler approval set")
        owners = payload.get("scheduler_active_owners", [])
        if (
            not isinstance(owners, list)
            or any(not isinstance(name, str) or not name for name in owners)
            or len(set(owners)) != len(owners)
        ):
            raise ValueError("invalid scheduler owner set")
        scope_digest = payload.get("scheduler_scope_digest")
        if scope_digest is not None and (
            not isinstance(scope_digest, str) or not re.fullmatch(r"[0-9a-f]{16}", scope_digest)
        ):
            raise ValueError("invalid scheduler scope digest")
        force_full_source = payload.get("scheduler_force_full_source")
        if force_full_source is not None and (
            not isinstance(force_full_source, str)
            or force_full_source not in FORCE_FULL_SOURCES
            or force_full is not True
        ):
            raise ValueError("invalid scheduler force-full source")
    except (AgentLoopError, TypeError, ValueError, KeyError):
        return {"scheduler_metadata_status": "invalid"}
    return {
        "scheduler_contract": contract.as_dict(),
        "scheduler_previous_sha": previous,
        "scheduler_current_sha": current,
        "scheduler_obligation_digest": digest,
        "scheduler_selected_reviewers": tuple(selected),
        "scheduler_paused_reviewers": tuple(paused_pairs),
        "scheduler_reasons": tuple(reasons),
        "scheduler_final_sweep": final_sweep,
        "scheduler_force_full": force_full,
        "scheduler_calls_avoided": calls_avoided,
        "scheduler_phase": phase,
        "scheduler_primary_reviewer": primary,
        "scheduler_approved_reviewers": tuple(approved),
        "scheduler_active_owners": tuple(owners),
        "scheduler_scope_digest": scope_digest,
        "scheduler_force_full_source": force_full_source,
        "scheduler_metadata_status": "valid",
    }
