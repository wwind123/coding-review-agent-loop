"""Deterministic assembly for semantic plan revisions.

The model supplies a bounded semantic patch.  This module owns the mutable
boundary: it authenticates the base, validates every operation before making
changes, applies matrix operations simultaneously, and emits the existing
generation-1 ``StructuredPlanRevision`` shape.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .errors import AgentLoopError
from .protocol import (
    RISK_MATRIX_MAX_ROWS,
    ArchitectureImpact,
    DeferredStage,
    ExecutionStrategyRecommendation,
    HumanRequirementDisposition,
    PlanRevisionPatch,
    PlanRevisionPatchOperation,
    RiskTestMatrix,
    RiskTestMatrixChange,
    RiskTestMatrixMetadata,
    RiskTestMatrixRow,
    StructuredPlanRevision,
    StructuredPlanState,
    TypedPlanStages,
    _expect_disposition_list,
    _expect_execution_recommendation,
    _expect_human_requirement_dispositions,
    _expect_optional_issue_id_list,
    _expect_string_list,
    _expect_typed_plan_stages,
    ARCHITECTURE_IMPACT_DECLARED_STATUSES,
    _parse_architecture_impact,
    _expect_deferred_stage_list,
    _extract_structured_plan_revision_payload,
    _parse_risk_test_matrix_contract_fields,
    parse_plan_revision_patch,
    parse_risk_test_matrix,
    validate_risk_test_matrix_revision,
    validate_structured_plan_revision,
    validate_structured_plan_state,
)


ASSEMBLED_PLAN_SIDECAR_SCHEMA_VERSION = 1
ASSEMBLED_PLAN_SIDECAR_KIND = "assembled_plan_sidecar"
AGGREGATE_PLAN_IDENTITY_VERSION = 1


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, separators=(",", ":"), sort_keys=True, ensure_ascii=False
    ).encode("utf-8")


def canonical_json(value: object) -> str:
    """Return the byte-stable JSON representation used by plan identities."""
    return _json_bytes(value).decode("utf-8")


def _deferred_stage_payload(values: Sequence[DeferredStage]) -> list[dict[str, str]]:
    return [{"title": item.title, "summary": item.summary} for item in values]


def _typed_stage_payload(stages: TypedPlanStages) -> dict[str, object]:
    return {
        "child_stages": [
            {"title": item.title, "summary": item.summary}
            for item in stages.child_stages
        ],
        "external_dependencies": _deferred_stage_payload(stages.external_dependencies),
        "deferred_work": _deferred_stage_payload(stages.deferred_work),
        "plan_actions": _deferred_stage_payload(stages.plan_actions),
    }


def _architecture_payload(value: ArchitectureImpact | None) -> dict[str, object] | None:
    if value is None:
        return None
    if value.status not in ARCHITECTURE_IMPACT_DECLARED_STATUSES:
        # A degraded (parser-only) status must never reach a canonical payload
        # or aggregate identity.  Fail closed rather than omitting the key.
        raise AgentLoopError(
            "Canonical plan assembly requires architecture_impact.status `changed` or "
            f"`unchanged`; refusing degraded status {value.status!r}."
        )
    # Keep all fields, including empty arrays, because the object is already a
    # parsed generation-1 value and canonical identity must include its full
    # contract rather than a display-only subset.
    return {
        "status": value.status,
        "rationale": value.rationale,
        "affected_components": list(value.affected_components),
        "dependencies": list(value.dependencies),
        "execution_data_flows": list(value.execution_data_flows),
        "execution_flows": list(value.execution_flows),
        "data_flows": list(value.data_flows),
        "persistence": list(value.persistence),
        "public_contracts": list(value.public_contracts),
        "security_boundaries": list(value.security_boundaries),
        "canonical_document_action": value.canonical_document_action,
        "canonical_document_path": value.canonical_document_path,
        "canonical_document_rationale": value.canonical_document_rationale,
        "uncertainty": list(value.uncertainty),
    }


def _dispositions_payload(values: Sequence[object]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for item in values:
        result.append(
            {
                "item_id": item.item_id,
                "disposition": item.disposition,
                **({"note": item.note} if item.note is not None else {}),
            }
        )
    return result


def _human_dispositions_payload(
    values: Sequence[HumanRequirementDisposition],
) -> list[dict[str, str]]:
    return [
        {
            "requirement_id": item.requirement_id,
            "disposition": item.disposition,
            "evidence": item.evidence,
        }
        for item in values
    ]


def _execution_payload(value: ExecutionStrategyRecommendation | None) -> dict[str, object] | None:
    return value.to_payload() if value is not None else None


def structured_plan_revision_to_payload(
    plan: StructuredPlanRevision | StructuredPlanState,
    *,
    kind: str | None = None,
) -> dict[str, object]:
    """Serialize a parsed generation-1 plan without rendering Markdown."""
    payload: dict[str, object] = {
        "schema_version": plan.schema_version,
        "kind": kind or plan.kind,
        "state": plan.state,
        "summary": plan.summary,
        "plan_steps": list(plan.plan_steps),
    }
    if isinstance(plan, StructuredPlanRevision):
        payload["prior_plan_item_dispositions"] = _dispositions_payload(
            plan.prior_plan_item_dispositions
        )
    if plan.additional_closing_issue_ids is not None:
        payload["additional_closing_issue_ids"] = list(plan.additional_closing_issue_ids)
    if plan.deferred_stages:
        payload["deferred_stages"] = _deferred_stage_payload(plan.deferred_stages)
    typed = _typed_stage_payload(plan.typed_stages)
    for field_name, values in typed.items():
        if values:
            payload[field_name] = values
    if plan.human_requirement_dispositions:
        payload["human_requirement_dispositions"] = _human_dispositions_payload(
            plan.human_requirement_dispositions
        )
    architecture = _architecture_payload(plan.architecture_impact)
    if architecture is not None:
        payload["architecture_impact"] = architecture
    if plan.execution_strategy_contract_version is not None:
        payload["execution_strategy_contract_version"] = plan.execution_strategy_contract_version
        payload["execution_recommendation"] = _execution_payload(plan.execution_recommendation)
    if plan.risk_test_matrix_contract_version is not None:
        payload["risk_test_matrix_contract_version"] = plan.risk_test_matrix_contract_version
        payload["risk_test_matrix"] = (
            plan.risk_test_matrix.to_payload() if plan.risk_test_matrix is not None else None
        )
        payload["risk_test_matrix_changes"] = [
            change.to_payload() for change in plan.risk_test_matrix_changes
        ]
    return payload


def _parse_wire_plan_payload(payload: Mapping[str, object]) -> StructuredPlanRevision | StructuredPlanState:
    """Parse an already extracted JSON object through generation-1 validators."""
    text = (
        json.dumps(dict(payload), separators=(",", ":"), ensure_ascii=False)
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex"
    )
    kind = payload.get("kind")
    if kind == "plan_revision":
        parsed = validate_structured_plan_revision(text)
    elif kind == "plan_state":
        parsed = validate_structured_plan_state(text)
    else:
        raise AgentLoopError("Authenticated plan base must be plan_revision or plan_state.")
    if parsed is None:
        raise AgentLoopError("Authenticated plan base was not a structured generation-1 plan.")
    return parsed


@dataclass(frozen=True)
class AuthenticatedPlanState:
    """Immutable, caller-authenticated generation-1 base for a patch."""

    plan: StructuredPlanRevision | StructuredPlanState | Mapping[str, object]
    round_number: int
    state_identity: str
    approved: bool = False
    canonical_payload: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.plan, Mapping):
            parsed = _parse_wire_plan_payload(self.plan)
        else:
            parsed = self.plan
        if parsed.state != "blocking":
            raise AgentLoopError("Semantic patches require a blocking plan base.")
        if self.round_number < 0:
            raise AgentLoopError("Authenticated plan round_number must be non-negative.")
        if not isinstance(self.state_identity, str) or not re.fullmatch(r"[0-9a-f]{64}", self.state_identity):
            raise AgentLoopError("Authenticated plan state_identity must be a SHA-256 identity.")
        object.__setattr__(self, "plan", parsed)
        if self.canonical_payload is None:
            object.__setattr__(
                self,
                "canonical_payload",
                structured_plan_revision_to_payload(parsed),
            )
        else:
            object.__setattr__(self, "canonical_payload", copy.deepcopy(dict(self.canonical_payload)))
        assert self.canonical_payload is not None
        if aggregate_plan_identity(self.canonical_payload) != self.state_identity:
            raise AgentLoopError("Authenticated plan state_identity does not match canonical plan JSON.")

    @classmethod
    def from_plan(
        cls,
        plan: StructuredPlanRevision | StructuredPlanState | Mapping[str, object],
        *,
        round_number: int,
        approved: bool = False,
    ) -> "AuthenticatedPlanState":
        payload = (
            copy.deepcopy(dict(plan))
            if isinstance(plan, Mapping)
            else structured_plan_revision_to_payload(plan)
        )
        return cls(
            plan=plan,
            round_number=round_number,
            state_identity=aggregate_plan_identity(payload),
            approved=approved,
            canonical_payload=payload,
        )


def _identity_inputs(payload: Mapping[str, object]) -> dict[str, object]:
    """Make the identity's approval-relevant categories explicit."""
    return {
        "version": AGGREGATE_PLAN_IDENTITY_VERSION,
        "canonical_plan": dict(payload),
        "architecture": payload.get("architecture_impact"),
        "execution": {
            "contract_version": payload.get("execution_strategy_contract_version"),
            "recommendation": payload.get("execution_recommendation"),
            "external_dependencies": payload.get("external_dependencies", []),
            "deferred_work": payload.get("deferred_work", []),
            "plan_actions": payload.get("plan_actions", []),
            "deferred_stages": payload.get("deferred_stages", []),
        },
        "matrix": {
            "contract_version": payload.get("risk_test_matrix_contract_version"),
            "value": payload.get("risk_test_matrix"),
            "changes": payload.get("risk_test_matrix_changes", []),
        },
        "closing": payload.get("additional_closing_issue_ids"),
        "human_dispositions": payload.get("human_requirement_dispositions", []),
        "typed_categories": {
            "child_stages": payload.get("child_stages", []),
            "external_dependencies": payload.get("external_dependencies", []),
            "deferred_work": payload.get("deferred_work", []),
            "plan_actions": payload.get("plan_actions", []),
        },
    }


def aggregate_plan_identity(payload: Mapping[str, object]) -> str:
    """Return the SHA-256 identity of a canonical full-state JSON object."""
    return hashlib.sha256(_json_bytes(_identity_inputs(payload))).hexdigest()


@dataclass(frozen=True)
class AssembledPlanSidecar:
    schema_version: int
    kind: str
    response_form: str
    round_number: int
    canonical_json: Mapping[str, object]
    aggregate_identity: str
    raw_patch: Mapping[str, object] | None = None
    # The public Markdown is a separate rendering surface from canonical JSON.
    # Newly published sidecars bind that surface by digest so restart cannot
    # combine an authenticated state with unrelated reviewer-visible prose.
    rendered_plan_identity: str | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "response_form": self.response_form,
            "round_number": self.round_number,
            "canonical_json": copy.deepcopy(dict(self.canonical_json)),
            "aggregate_identity": self.aggregate_identity,
            "raw_patch": copy.deepcopy(dict(self.raw_patch)) if self.raw_patch is not None else None,
            "rendered_plan_identity": self.rendered_plan_identity,
        }

    def encode(self) -> str:
        return canonical_json(self.to_payload())


def make_assembled_plan_sidecar(
    plan: StructuredPlanRevision | StructuredPlanState | Mapping[str, object],
    *,
    round_number: int,
    response_form: str = "semantic-patch-v1",
    raw_patch: Mapping[str, object] | None = None,
    rendered_plan: str | None = None,
) -> AssembledPlanSidecar:
    payload = (
        copy.deepcopy(dict(plan))
        if isinstance(plan, Mapping)
        else structured_plan_revision_to_payload(plan)
    )
    return AssembledPlanSidecar(
        schema_version=ASSEMBLED_PLAN_SIDECAR_SCHEMA_VERSION,
        kind=ASSEMBLED_PLAN_SIDECAR_KIND,
        response_form=response_form,
        round_number=round_number,
        canonical_json=payload,
        aggregate_identity=aggregate_plan_identity(payload),
        raw_patch=raw_patch,
        rendered_plan_identity=(
            rendered_plan_identity(rendered_plan) if rendered_plan is not None else None
        ),
    )


def rendered_plan_identity(text: str) -> str:
    """Return the identity used to bind a rendered canonical plan to a sidecar."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("rendered canonical plan must be non-empty text")
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def decode_assembled_plan_sidecar(value: str | Mapping[str, object]) -> AssembledPlanSidecar:
    try:
        payload = json.loads(value) if isinstance(value, str) else dict(value)
        if not isinstance(payload, dict):
            raise ValueError("sidecar is not an object")
        legacy_expected = {
            "schema_version", "kind", "response_form", "round_number",
            "canonical_json", "aggregate_identity", "raw_patch",
        }
        expected = legacy_expected | {"rendered_plan_identity"}
        if set(payload) != legacy_expected and set(payload) != expected:
            raise ValueError("sidecar keys are not exact")
        if payload["schema_version"] != ASSEMBLED_PLAN_SIDECAR_SCHEMA_VERSION:
            raise ValueError("unsupported sidecar schema")
        if payload["kind"] != ASSEMBLED_PLAN_SIDECAR_KIND:
            raise ValueError("invalid sidecar kind")
        if not isinstance(payload["response_form"], str) or not payload["response_form"].strip():
            raise ValueError("invalid sidecar response form")
        if isinstance(payload["round_number"], bool) or not isinstance(payload["round_number"], int):
            raise ValueError("invalid sidecar round number")
        canonical = payload["canonical_json"]
        if not isinstance(canonical, dict):
            raise ValueError("invalid sidecar canonical JSON")
        identity = payload["aggregate_identity"]
        if not isinstance(identity, str) or aggregate_plan_identity(canonical) != identity:
            raise ValueError("sidecar aggregate identity mismatch")
        raw_patch = payload["raw_patch"]
        if raw_patch is not None and not isinstance(raw_patch, dict):
            raise ValueError("invalid sidecar raw patch provenance")
        rendered_identity = payload.get("rendered_plan_identity")
        if rendered_identity is not None and (
            not isinstance(rendered_identity, str)
            or re.fullmatch(r"[0-9a-f]{64}", rendered_identity) is None
        ):
            raise ValueError("invalid rendered canonical plan identity")
        _parse_wire_plan_payload(canonical)
        return AssembledPlanSidecar(
            schema_version=1,
            kind=ASSEMBLED_PLAN_SIDECAR_KIND,
            response_form=payload["response_form"],
            round_number=payload["round_number"],
            canonical_json=copy.deepcopy(canonical),
            aggregate_identity=identity,
            raw_patch=copy.deepcopy(raw_patch) if raw_patch is not None else None,
            rendered_plan_identity=rendered_identity,
        )
    except (AgentLoopError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise AgentLoopError(f"Invalid assembled plan sidecar: {exc}") from exc


def _base_payload(base: AuthenticatedPlanState) -> dict[str, object]:
    assert base.canonical_payload is not None
    payload = copy.deepcopy(dict(base.canonical_payload))
    payload["kind"] = "plan_revision"
    payload["schema_version"] = 1
    payload["state"] = "blocking"
    payload.setdefault("prior_plan_item_dispositions", [])
    return payload


def _field_payload(value: object) -> object:
    if isinstance(value, ArchitectureImpact):
        return _architecture_payload(value)
    if isinstance(value, ExecutionStrategyRecommendation):
        return value.to_payload()
    if isinstance(value, tuple):
        if value and isinstance(value[0], HumanRequirementDisposition):
            return _human_dispositions_payload(value)  # type: ignore[arg-type]
        if value and isinstance(value[0], DeferredStage):
            return _deferred_stage_payload(value)  # type: ignore[arg-type]
        return [_field_payload(item) for item in value]
    if isinstance(value, RiskTestMatrixMetadata):
        return value.to_payload()
    return value


def _matrix_operations(operations: Sequence[PlanRevisionPatchOperation]) -> tuple[PlanRevisionPatchOperation, ...]:
    return tuple(operation for operation in operations if operation.op.startswith("matrix_"))


def _validate_matrix_operations(
    old: RiskTestMatrix,
    operations: Sequence[PlanRevisionPatchOperation],
) -> None:
    old_by_id = {row.row_id: row for row in old.rows}
    consumers: dict[str, str] = {}
    producers: dict[str, str] = {}
    add_positions: set[int] = set()
    metadata_count = 0
    adds = 0

    def consume(row_id: str, op: str) -> None:
        if row_id not in old_by_id:
            raise AgentLoopError(f"{op} names unknown source row ID `{row_id}`.")
        previous = consumers.get(row_id)
        if previous is not None:
            raise AgentLoopError(f"row ID `{row_id}` is consumed by both {previous} and {op}.")
        consumers[row_id] = op

    def produce(row_id: str, op: str) -> None:
        previous = producers.get(row_id)
        if previous is not None:
            raise AgentLoopError(f"row ID `{row_id}` is produced by both {previous} and {op}.")
        if op in {"matrix_add", "matrix_split", "matrix_merge"} and row_id in old_by_id:
            raise AgentLoopError(f"{op} cannot create row ID `{row_id}` that exists in the base matrix.")
        producers[row_id] = op

    for operation in operations:
        if operation.op == "matrix_metadata_replace":
            metadata_count += 1
            if metadata_count > 1:
                raise AgentLoopError("matrix_metadata_replace may appear at most once per patch.")
            assert isinstance(operation.value, RiskTestMatrixMetadata)
            old_metadata = {
                "applicability": old.applicability,
                "important_exclusions": list(old.important_exclusions),
            }
            if old.not_applicable_rationale is not None:
                old_metadata["not_applicable_rationale"] = old.not_applicable_rationale
            if old_metadata == operation.value.to_payload():
                raise AgentLoopError("matrix_metadata_replace is payload-identical and has no effect.")
            continue
        if operation.op == "matrix_add":
            assert operation.row is not None and operation.final_position is not None
            adds += 1
            if operation.final_position in add_positions:
                raise AgentLoopError("matrix_add operations must use unique final_position values.")
            add_positions.add(operation.final_position)
            produce(operation.row.row_id, operation.op)
        elif operation.op == "matrix_edit":
            assert operation.row is not None and operation.row_id is not None
            consume(operation.row_id, operation.op)
            if operation.row.row_id != operation.row_id:
                raise AgentLoopError("matrix_edit must retain its row_id.")
            produce(operation.row.row_id, operation.op)
            if operation.row.to_payload() == old_by_id[operation.row_id].to_payload():
                raise AgentLoopError(f"matrix_edit for `{operation.row_id}` is payload-identical and has no effect.")
        elif operation.op == "matrix_retire":
            assert operation.row_id is not None
            consume(operation.row_id, operation.op)
        elif operation.op == "matrix_split":
            assert operation.source_row_id is not None
            consume(operation.source_row_id, operation.op)
            for row in operation.target_rows:
                produce(row.row_id, operation.op)
        elif operation.op == "matrix_merge":
            assert operation.target_row is not None
            for source in operation.source_row_ids:
                consume(source, operation.op)
            produce(operation.target_row.row_id, operation.op)

    transformed_count = len(old.rows) - len(consumers)
    for operation in operations:
        if operation.op == "matrix_edit":
            transformed_count += 1
        elif operation.op == "matrix_split":
            transformed_count += len(operation.target_rows)
        elif operation.op == "matrix_merge":
            transformed_count += 1
    final_count = transformed_count + adds
    if final_count > RISK_MATRIX_MAX_ROWS:
        raise AgentLoopError(
            f"assembled risk_test_matrix has {final_count} rows and exceeds the "
            f"{RISK_MATRIX_MAX_ROWS}-row bound; consolidate scenarios explicitly with "
            "matrix_merge or matrix_retire instead of adding rows."
        )
    for position in add_positions:
        if position >= final_count:
            raise AgentLoopError(
                f"matrix_add final_position {position} is out of range for the {final_count}-row final matrix."
            )


def _apply_matrix_operations(
    old_payload: Mapping[str, object],
    operations: Sequence[PlanRevisionPatchOperation],
) -> tuple[dict[str, object], tuple[RiskTestMatrixChange, ...]]:
    old = parse_risk_test_matrix(old_payload, context="authenticated risk_test_matrix")
    matrix_operations = _matrix_operations(operations)
    _validate_matrix_operations(old, matrix_operations)
    if not matrix_operations:
        return copy.deepcopy(dict(old_payload)), ()

    by_source: dict[str, PlanRevisionPatchOperation] = {}
    merge_insertions: dict[int, PlanRevisionPatchOperation] = {}
    additions: list[PlanRevisionPatchOperation] = []
    metadata: PlanRevisionPatchOperation | None = None
    old_index = {row.row_id: index for index, row in enumerate(old.rows)}
    for operation in matrix_operations:
        if operation.op == "matrix_add":
            additions.append(operation)
        elif operation.op == "matrix_metadata_replace":
            metadata = operation
        elif operation.op in {"matrix_edit", "matrix_retire", "matrix_split"}:
            source = operation.row_id or operation.source_row_id
            assert source is not None
            by_source[source] = operation
        elif operation.op == "matrix_merge":
            assert operation.source_row_ids
            insertion = min(old_index[source] for source in operation.source_row_ids)
            merge_insertions[insertion] = operation
            for source in operation.source_row_ids:
                by_source[source] = operation

    transformed: list[RiskTestMatrixRow] = []
    for index, row in enumerate(old.rows):
        operation = by_source.get(row.row_id)
        if operation is None:
            transformed.append(row)
            continue
        if operation.op == "matrix_edit":
            assert operation.row is not None
            transformed.append(operation.row)
        elif operation.op == "matrix_split":
            transformed.extend(operation.target_rows)
        elif operation.op == "matrix_merge":
            if merge_insertions.get(index) is operation:
                assert operation.target_row is not None
                transformed.append(operation.target_row)
        # retire contributes no row.

    final_count = len(transformed) + len(additions)
    final_rows: list[RiskTestMatrixRow | None] = [None] * final_count
    for operation in additions:
        assert operation.row is not None and operation.final_position is not None
        final_rows[operation.final_position] = operation.row
    transformed_iter = iter(transformed)
    for index, row in enumerate(final_rows):
        if row is None:
            final_rows[index] = next(transformed_iter)
    if next(transformed_iter, None) is not None:
        raise AgentLoopError("matrix placement left transformed rows unplaced.")

    old_metadata = {
        "applicability": old_payload["applicability"],
        "important_exclusions": copy.deepcopy(old_payload["important_exclusions"]),
    }
    if "not_applicable_rationale" in old_payload:
        old_metadata["not_applicable_rationale"] = old_payload["not_applicable_rationale"]
    if metadata is not None:
        assert isinstance(metadata.value, RiskTestMatrixMetadata)
        old_metadata = metadata.value.to_payload()
    current_payload = {
        **old_metadata,
        "rows": [row.to_payload() for row in final_rows if row is not None],
    }
    current = parse_risk_test_matrix(current_payload, context="assembled risk_test_matrix")

    changes: list[RiskTestMatrixChange] = []
    old_ids = {row.row_id for row in old.rows}
    new_ids = {row.row_id for row in current.rows}
    if metadata is not None:
        assert metadata.audit_operation is not None and metadata.rationale is not None
        changes.append(
            RiskTestMatrixChange(
                operation=metadata.audit_operation,
                row_ids=tuple(sorted(old_ids | new_ids or {"matrix"})),
                rationale=metadata.rationale,
            )
        )
    class_order = {"matrix_add": 0, "matrix_edit": 1, "matrix_retire": 2, "matrix_split": 3, "matrix_merge": 4}
    structural_changes: list[tuple[int, tuple[str, ...], RiskTestMatrixChange]] = []
    for operation in matrix_operations:
        if operation.op == "matrix_metadata_replace":
            continue
        assert operation.rationale is not None
        if operation.op == "matrix_add":
            assert operation.row is not None
            ids = (operation.row.row_id,)
            audit_operation = "add"
        elif operation.op == "matrix_edit":
            assert operation.row_id is not None
            ids = (operation.row_id,)
            audit_operation = "change"
        elif operation.op == "matrix_retire":
            assert operation.row_id is not None
            ids = (operation.row_id,)
            audit_operation = "retire"
        elif operation.op == "matrix_split":
            assert operation.source_row_id is not None
            ids = (operation.source_row_id, *(row.row_id for row in operation.target_rows))
            audit_operation = "split"
        else:
            assert operation.target_row is not None
            ids = (*operation.source_row_ids, operation.target_row.row_id)
            audit_operation = "merge"
        normalized_ids = tuple(sorted(ids))
        structural_changes.append(
            (
                class_order[operation.op],
                normalized_ids,
                RiskTestMatrixChange(audit_operation, normalized_ids, operation.rationale),
            )
        )
    changes.extend(item[2] for item in sorted(structural_changes, key=lambda item: (item[0], item[1])))
    validate_risk_test_matrix_revision(old, current, changes)
    return current_payload, tuple(changes)


def assemble_authenticated_plan_revision(
    base: AuthenticatedPlanState,
    patch: PlanRevisionPatch | Mapping[str, object],
    *,
    result_round_number: int | None = None,
) -> tuple[StructuredPlanRevision, AssembledPlanSidecar]:
    """Atomically assemble one patch against one authenticated base.

    ``base.round_number`` authenticates the input state, while
    ``result_round_number`` identifies the newly published canonical state.
    The default derives the next round for compatibility with direct callers;
    publication callers should pass the round assigned to the result so a
    restart cannot hydrate the result as its stale predecessor.
    """
    parsed_patch = parse_plan_revision_patch(patch)
    if base.approved:
        raise AgentLoopError("Semantic patches require an unapproved blocking plan base.")
    if result_round_number is None:
        result_round_number = base.round_number + 1
    if isinstance(result_round_number, bool) or result_round_number < 0:
        raise AgentLoopError("Authenticated plan result_round_number must be non-negative.")
    if result_round_number <= base.round_number:
        raise AgentLoopError(
            "Authenticated plan result_round_number must be greater than the authenticated base round."
        )
    if parsed_patch.base_round_number != base.round_number:
        raise AgentLoopError(
            f"Semantic patch base_round_number {parsed_patch.base_round_number} does not match authenticated base round {base.round_number}."
        )
    if parsed_patch.base_state_identity != base.state_identity:
        raise AgentLoopError(
            "Semantic patch base_state_identity does not match the authenticated base identity."
        )

    payload = _base_payload(base)
    replacement_fields: set[str] = set()
    matrix_operations = _matrix_operations(parsed_patch.operations)
    matrix_payload = payload.get("risk_test_matrix")
    if matrix_operations and not isinstance(matrix_payload, dict):
        raise AgentLoopError("Matrix operations require an authenticated generation-1 risk matrix base.")

    # Validate all whole-field operations and conflicts before changing the
    # copied payload.  The matrix validator is likewise run before placement.
    for operation in parsed_patch.operations:
        if operation.op != "replace":
            continue
        assert operation.field is not None
        if operation.field in replacement_fields:
            raise AgentLoopError(f"Field `{operation.field}` has duplicate replace operations.")
        replacement_fields.add(operation.field)
        candidate = _field_payload(operation.value)
        if operation.field in payload and payload[operation.field] == candidate:
            raise AgentLoopError(f"replace for `{operation.field}` is payload-identical and has no effect.")

    if matrix_operations:
        assert isinstance(matrix_payload, dict)
        new_matrix_payload, changes = _apply_matrix_operations(matrix_payload, matrix_operations)
        payload["risk_test_matrix"] = new_matrix_payload
        payload["risk_test_matrix_contract_version"] = 1
        payload["risk_test_matrix_changes"] = [change.to_payload() for change in changes]
    elif "risk_test_matrix" in payload:
        # Audits are derived per revision; a non-matrix semantic edit has no
        # new matrix transition and therefore emits an empty audit.
        if "risk_test_matrix_changes" in payload:
            payload["risk_test_matrix_changes"] = []

    for operation in parsed_patch.operations:
        if operation.op != "replace":
            continue
        assert operation.field is not None
        payload[operation.field] = _field_payload(operation.value)
        if operation.field == "execution_recommendation":
            if "execution_strategy_contract_version" not in payload:
                raise AgentLoopError("execution_recommendation replacement requires an authenticated contract version.")

    payload["kind"] = "plan_revision"
    payload["schema_version"] = 1
    payload["state"] = "blocking"
    payload["summary"] = payload.get("summary", "")
    payload["prior_plan_item_dispositions"] = _dispositions_payload(
        parsed_patch.prior_plan_item_dispositions
    )
    assembled = _parse_wire_plan_payload(payload)
    if not isinstance(assembled, StructuredPlanRevision):
        raise AgentLoopError("Assembler did not emit a generation-1 plan_revision.")
    sidecar = make_assembled_plan_sidecar(
        payload,
        round_number=result_round_number,
        response_form="semantic-patch-v1",
        # Provenance is re-parsed as a wire payload on every restart, and the
        # stored copy is compared for equality against a freshly re-serialized
        # patch.  ``PlanRevisionPatch.to_payload`` is the single normalizer:
        # it emits JSON arrays, so both sides stay byte-comparable (#879).
        raw_patch=parsed_patch.to_payload(),
    )
    return assembled, sidecar


def assemble_plan_revision(
    base: AuthenticatedPlanState,
    patch: PlanRevisionPatch | Mapping[str, object],
) -> StructuredPlanRevision:
    """Convenience API returning only the downstream generation-1 object."""
    return assemble_authenticated_plan_revision(base, patch)[0]


def hydrate_authenticated_plan_state(
    sidecar: AssembledPlanSidecar | str | Mapping[str, object],
    *,
    round_number: int | None = None,
) -> AuthenticatedPlanState:
    """Hydrate a semantic base only from an authenticated canonical sidecar."""
    decoded = sidecar if isinstance(sidecar, AssembledPlanSidecar) else decode_assembled_plan_sidecar(sidecar)
    if round_number is not None and decoded.round_number != round_number:
        raise AgentLoopError("Authenticated sidecar round number does not match the requested base.")
    parsed = _parse_wire_plan_payload(decoded.canonical_json)
    return AuthenticatedPlanState(
        plan=parsed,
        round_number=decoded.round_number,
        state_identity=decoded.aggregate_identity,
        approved=False,
        canonical_payload=decoded.canonical_json,
    )


# Stable aliases make the boundary discoverable to callers while the
# implementation name remains explicit in stack traces and diagnostics.
PlanRevisionPatchV1 = PlanRevisionPatch
AuthenticatedAssembledPlan = AssembledPlanSidecar
plan_revision_identity = aggregate_plan_identity
