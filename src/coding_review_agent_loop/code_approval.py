"""Independent complete-code approval lookup for a PR head."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .round_state import PostedRoundRecord
    from .execution_policy import ApprovedPlanContext
from .agents.base import AgentName
from .agents.registry import agent_display_name
from .review_scope import is_complete_code_review
from .review_approval import _contradictory_evidence_approval

def _latest_pr_approved_reviews_for_head(
    comments: Sequence[object],
    *,
    head_sha: str | None,
    configured_reviewers: Sequence[AgentName],
    approved_plan_context: ApprovedPlanContext | None = None,
    human_requirements: Sequence[object] = (),
    reviewer_acquisition_contract: Mapping[str, tuple[object, ...]] | None = None,
    require_architecture_contract: bool = False,
) -> dict[str, PostedRoundRecord]:
    from .round_state import _extract_round_metadata_records

    if not head_sha:
        return {}
    configured_names = {agent_display_name(agent) for agent in configured_reviewers}
    latest_by_reviewer = {}
    records = _extract_round_metadata_records(comments, flow="pr")
    for record in reversed(records):
        metadata = record.metadata
        if (
            metadata.subject != head_sha
            or not is_complete_code_review(metadata)
            or metadata.agent not in configured_names
            or metadata.agent in latest_by_reviewer
        ):
            continue
        if require_architecture_contract and metadata.architecture_contract_version != 1:
            continue
        latest_by_reviewer[metadata.agent] = record

    def acquisition_contract_matches(record: PostedRoundRecord) -> bool:
        if reviewer_acquisition_contract is None:
            return True
        expected = reviewer_acquisition_contract.get(record.metadata.agent)
        if expected is None or len(expected) < 2:
            return False
        if record.metadata.acquisition_outcome != "success":
            return False
        return (
            (record.metadata.configured_model in expected[0]
             if isinstance(expected[0], tuple) else record.metadata.configured_model == expected[0])
            and record.metadata.configured_effort == expected[1]
            and (len(expected) < 3 or record.metadata.provider == expected[2])
        )

    return {
        reviewer: record
        for reviewer, record in latest_by_reviewer.items()
        if record.metadata.state == "approved"
        and not _contradictory_evidence_approval(record.metadata)
        and acquisition_contract_matches(record)
        and (
            approved_plan_context is None
            or (
                record.metadata.approved_plan_hash == approved_plan_context.plan_hash
                and record.metadata.approved_plan_subject == approved_plan_context.plan_subject
            )
        )
        and (
            not human_requirements
            or (
                "HUMAN_REQUIREMENTS_RESOLVED" in record.body
                and {
                    str(getattr(item, "requirement_id", ""))
                    for item in human_requirements
                }.issubset(set(record.metadata.surfaced_reviewer_requirement_ids))
            )
        )
    }
