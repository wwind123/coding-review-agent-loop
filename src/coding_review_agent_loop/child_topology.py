"""Shared planning and preflight primitives for flat child topologies.

The decomposition and split workflows have different payload formats, but
they share one important invariant: a parent may own only one bounded flat
set of children.  This module deliberately contains no GitHub mutation code;
callers can use the result to render and validate every draft before posting a
checkpoint or creating an issue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class NeedsHumanDecision:
    """A deterministic, side-effect-free over-limit outcome."""

    parent_issue: int
    source: str
    requested_desired_count: int
    recognized_existing_count: int
    projected_total: int
    configured_limit: int
    guidance: str = (
        "Consolidate the flat stages or use the hierarchical decomposition design "
        "tracked in #720."
    )

    @property
    def state(self) -> str:
        return "needs-human"

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "parent": self.parent_issue,
            "source": self.source,
            "requested_desired_count": self.requested_desired_count,
            "recognized_existing_count": self.recognized_existing_count,
            "projected_total": self.projected_total,
            "configured_limit": self.configured_limit,
            "guidance": self.guidance,
        }

    def __str__(self) -> str:
        return (
            f"Flat child topology for issue #{self.parent_issue} needs human decision: "
            f"{self.projected_total} projected children exceeds the configured limit "
            f"of {self.configured_limit}. {self.guidance}"
        )


@dataclass(frozen=True)
class ChildTopologyPreflight:
    """The count and identity portion of a topology preflight."""

    desired_count: int
    recognized_existing_count: int
    projected_total: int
    missing_count: int


def parent_child_search_query(parent_issue: int) -> str:
    """Find both flat-child title conventions in one recovery query."""
    return f'"(from #{parent_issue})" in:title OR "[#{parent_issue} stage]" in:title'


def preflight_flat_child_count(
    *,
    parent_issue: int,
    source: str,
    desired_keys: Iterable[str],
    recognized_keys: Iterable[str],
    configured_limit: int,
) -> ChildTopologyPreflight | NeedsHumanDecision:
    """Validate the complete desired topology before any mutation.

    Keys are normalized by callers, but de-duplicating here makes this helper
    safe for both typed and model-generated topology adapters.  Existing
    recognized children count once, while a desired child already present in
    that set does not consume another slot.
    """

    desired = set(desired_keys)
    recognized = set(recognized_keys)
    projected = len(recognized | desired)
    result = ChildTopologyPreflight(
        desired_count=len(desired),
        recognized_existing_count=len(recognized),
        projected_total=projected,
        missing_count=len(desired - recognized),
    )
    if projected > configured_limit:
        return NeedsHumanDecision(
            parent_issue=parent_issue,
            source=source,
            requested_desired_count=len(desired),
            recognized_existing_count=len(recognized),
            projected_total=projected,
            configured_limit=configured_limit,
        )
    return result
