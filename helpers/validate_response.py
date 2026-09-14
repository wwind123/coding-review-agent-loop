"""
Validate a structured agent response file against the existing protocol library.

Usage:
  python -m helpers.validate_response \\
    --file PATH --kind KIND [--context-file PATH]

Kinds:
  plan_state       -- coder plan post; validates AGENT_PLAN_STATE marker
  plan_review      -- reviewer plan review structured JSON
  pr_review        -- reviewer PR review structured JSON
  coder_followup   -- coder follow-up structured JSON
  issue_implementation -- issue implementation structured JSON
  plan_revision    -- coder plan revision structured JSON

The optional --context-file is a JSON file with schema:
  {
    "reviewer":            str,
    "prior_items":         [...serialized UnresolvedReviewItem...],
    "current_round_items": [...serialized UnresolvedReviewItem...],
    "human_requirements":  [{"id": str, "text": str, "scope": str}, ...]
  }

Exit 0 on success; exit 1 with diagnostic on failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make src importable when run from the repo root as `python -m helpers.validate_response`
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coding_review_agent_loop.errors import AgentLoopError, UnknownPriorItemDispositionError
from coding_review_agent_loop.protocol import (
    HUMAN_REQUIREMENTS_RESOLVED_RE,
    parse_plan_state,
    validate_structured_plan_state,
    validate_structured_plan_revision,
)
from coding_review_agent_loop.unresolved_items import (
    _validate_plan_review_response,
    _validate_review_response,
    _validate_coder_followup_response,
)
from coding_review_agent_loop.round_state import _deserialize_unresolved_item
from coding_review_agent_loop.github import HumanReviewRequirement


_KINDS = ("plan_state", "plan_review", "pr_review", "coder_followup", "issue_implementation", "plan_revision")


def _load_context(path: str | None) -> dict[str, object]:
    """Load optional context JSON file."""
    if path is None:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"validate_response: could not read context file {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def _deserialize_unresolved_items(raw: list[object]) -> list[object]:
    result = []
    for item in raw:
        result.append(_deserialize_unresolved_item(item))
    return result


def _deserialize_human_requirements(raw: list[object]) -> list[HumanReviewRequirement]:
    result = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        result.append(
            HumanReviewRequirement(
                source_type=str(item.get("source_type", "issue")),
                author=str(item.get("author", "")) or None,
                created_at=str(item.get("created_at", "")) or None,
                url=str(item.get("url", "")) or None,
                body=str(item.get("body", item.get("text", ""))),
            )
        )
    return result


def validate_response_text(
    text: str,
    *,
    kind: str,
    reviewer: str = "Codex",
    prior_items: list[object] | tuple[object, ...] = (),
    current_round_items: list[object] | tuple[object, ...] = (),
    human_requirements: list[object] | tuple[object, ...] = (),
    required_architecture_impact_contract: int = 0,
    require_execution_strategy_contract: int = 0,
    require_risk_test_matrix_contract: int = 0,
    delivered_risk_test_matrix: object = None,
    delivered_risk_test_matrix_identity: str | None = None,
    authoritative_test_observations=None,
    delivered_risk_test_matrix_row_ids=None,
) -> object:
    """Validate response text with the same context used by the helper CLI.

    ``prior_items``/``current_round_items`` and ``human_requirements`` may contain
    either their runtime dataclasses or their serialized dictionary forms.  The
    original ``AgentLoopError`` subclass is intentionally allowed to propagate so
    callers can apply type-specific deterministic recovery.

    For ``kind='plan_revision'``, the unknown-disposition check is unconditional
    regardless of ledger size.  Passing ``prior_items=()`` (the default) or
    ``prior_items=[]`` treats **any** disposition ID as unknown and raises
    ``UnknownPriorItemDispositionError`` — there is no ``if prior_items:`` guard.

    New callers with an empty ledger that want auto-recovery have two options:

    * Call ``_recover_structured_response`` directly with
      ``allowed_prior_item_ids=[]`` — the function's deterministic strip pass
      removes all stray dispositions before revalidation.
    * Route through ``_complete_coder_turn(..., auto_recover=True)`` — the
      integrated path that internally calls ``_recover_structured_response``.

    Note: ``_recover_structured_response`` itself has no ``auto_recover``
    parameter; ``auto_recover`` is a flag on ``_complete_coder_turn``.
    """
    if kind not in _KINDS:
        raise ValueError(f"Unsupported response kind: {kind}")

    deserialized_prior = [
        _deserialize_unresolved_item(item) if isinstance(item, dict) else item
        for item in prior_items
    ]
    deserialized_current = [
        _deserialize_unresolved_item(item) if isinstance(item, dict) else item
        for item in current_round_items
    ]
    deserialized_requirements = [
        _deserialize_human_requirements([item])[0] if isinstance(item, dict) else item
        for item in human_requirements
        if not isinstance(item, dict) or item
    ]

    if kind == "plan_state":
        parsed = validate_structured_plan_state(
            text,
            required_architecture_impact_contract=required_architecture_impact_contract,
            require_execution_strategy_contract=require_execution_strategy_contract,
            require_risk_test_matrix_contract=require_risk_test_matrix_contract,
        )
        if parsed is None:
            if required_architecture_impact_contract == 1:
                raise AgentLoopError(
                    "Fresh plan_state responses must use the structured architecture-impact contract."
                )
            return parse_plan_state(text)
        return parsed
    if kind == "plan_review":
        return _validate_plan_review_response(
            text,
            reviewer=reviewer,
            unresolved_items=deserialized_prior,
            current_round_items=deserialized_current,
        )
    if kind == "pr_review":
        parsed = _validate_review_response(
            text,
            reviewer=reviewer,
            unresolved_items=deserialized_prior,
            current_round_items=deserialized_current,
        )
        marker_present = HUMAN_REQUIREMENTS_RESOLVED_RE.search(text) is not None
        if deserialized_requirements and parsed.state == "approved" and not marker_present:
            raise AgentLoopError(
                "Approved PR review omitted HUMAN_REQUIREMENTS_RESOLVED "
                "after signed human requirements were surfaced. Verify every surfaced "
                "stable ID and include the required resolution marker."
            )
        if not deserialized_requirements and marker_present:
            raise AgentLoopError(
                "PR review emitted HUMAN_REQUIREMENTS_RESOLVED when no signed human "
                "requirements were surfaced."
            )
        return parsed
    if kind == "coder_followup":
        return _validate_coder_followup_response(
            text,
            unresolved_items=deserialized_prior,
            human_requirements=deserialized_requirements or None,
            required_architecture_impact_contract=required_architecture_impact_contract,
            delivered_risk_test_matrix=delivered_risk_test_matrix,
            delivered_risk_test_matrix_identity=delivered_risk_test_matrix_identity,
            required_risk_test_matrix_contract=require_risk_test_matrix_contract,
            authoritative_test_observations=authoritative_test_observations,
            delivered_risk_test_matrix_row_ids=delivered_risk_test_matrix_row_ids,
        )
    if kind == "issue_implementation":
        from coding_review_agent_loop.orchestrator import _validate_issue_implementation_response

        return _validate_issue_implementation_response(
            text,
            human_requirements=deserialized_requirements,
            require_architecture_impact=(required_architecture_impact_contract == 1),
            delivered_risk_test_matrix=delivered_risk_test_matrix,
            delivered_risk_test_matrix_identity=delivered_risk_test_matrix_identity,
            require_risk_test_matrix_contract=(require_risk_test_matrix_contract == 1),
            authoritative_test_observations=authoritative_test_observations,
            delivered_risk_test_matrix_row_ids=delivered_risk_test_matrix_row_ids,
        )

    parsed = validate_structured_plan_revision(
        text,
        required_architecture_impact_contract=required_architecture_impact_contract,
        require_execution_strategy_contract=require_execution_strategy_contract,
        require_risk_test_matrix_contract=require_risk_test_matrix_contract,
    )
    if parsed is None:
        raise AgentLoopError("Response did not parse as a structured plan_revision.")
    allowed_ids = {item.item_id for item in deserialized_prior}
    unknown = {
        disposition.item_id
        for disposition in parsed.prior_plan_item_dispositions
    } - allowed_ids
    if unknown:
        raise UnknownPriorItemDispositionError(
            unknown_ids=tuple(sorted(unknown)),
            allowed_ids=tuple(sorted(allowed_ids)),
            same_round_description=(
                "Same-round findings are informational only and must not be "
                "dispositioned as prior carried items."
            ),
        )
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an agent response file.")
    parser.add_argument("--file", required=True, help="Path to the response file.")
    parser.add_argument(
        "--kind",
        required=True,
        choices=_KINDS,
        help="Kind of response to validate.",
    )
    parser.add_argument(
        "--context-file",
        default=None,
        help="Optional JSON file with reviewer identity and prior item context.",
    )
    parser.add_argument(
        "--require-execution-strategy-contract",
        action="store_true",
        help="Require the fresh generation-1 planning execution recommendation.",
    )
    parser.add_argument(
        "--require-risk-test-matrix-contract",
        action="store_true",
        help="Require the fresh generation-1 risk matrix planning contract.",
    )
    args = parser.parse_args()

    try:
        text = Path(args.file).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"validate_response: cannot read {args.file}: {exc}", file=sys.stderr)
        sys.exit(1)

    ctx = _load_context(args.context_file)
    reviewer = str(ctx.get("reviewer", "Codex"))
    prior_items = _deserialize_unresolved_items(list(ctx.get("prior_items", [])))
    current_round_items = _deserialize_unresolved_items(list(ctx.get("current_round_items", [])))
    raw_human_requirements = list(ctx.get("human_requirements", []))
    human_requirements = _deserialize_human_requirements(raw_human_requirements)
    # A matrix-bearing helper validation has no verified evidence authority
    # unless the bounded local-evidence carrier decodes successfully.  Passing
    # an empty sequence is intentional: it prevents a claimed `verified` row
    # from passing on citation shape alone.
    authoritative_test_observations = []
    local_test_evidence = ctx.get("local_test_evidence")
    if isinstance(local_test_evidence, str):
        from coding_review_agent_loop.local_test_evidence import decode_bounded_evidence

        decoded = decode_bounded_evidence(local_test_evidence)
        if decoded is not None:
            authoritative_test_observations = decoded.observations

    kind = args.kind
    try:
        validate_response_text(
            text,
            kind=kind,
            reviewer=reviewer,
            prior_items=prior_items,
            current_round_items=current_round_items,
            human_requirements=human_requirements,
            require_execution_strategy_contract=(
                1 if args.require_execution_strategy_contract else 0
            ),
            require_risk_test_matrix_contract=(
                1 if args.require_risk_test_matrix_contract else 0
            ),
            delivered_risk_test_matrix=ctx.get("risk_test_matrix"),
            delivered_risk_test_matrix_identity=ctx.get("risk_test_matrix_identity"),
            authoritative_test_observations=authoritative_test_observations,
            delivered_risk_test_matrix_row_ids=ctx.get("risk_test_matrix_expected_row_ids"),
        )
    except AgentLoopError as exc:
        print(f"validation failed: {kind}: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"validation error: {kind}: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"validation passed: {kind}")


if __name__ == "__main__":
    main()
