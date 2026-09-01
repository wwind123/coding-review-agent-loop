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

from .github import FoundIssue


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


def parent_child_search_queries(parent_issue: int) -> tuple[str, str]:
    """Return the supported flat-child title searches separately.

    GitHub issue search does not implement the boolean ``OR`` operator used by
    code search. Keeping these as separate queries avoids silently missing one
    of the two historical child-title conventions during recovery.
    """
    return (
        f'"(from #{parent_issue})" in:title',
        f'"[#{parent_issue} stage]" in:title',
    )


def merge_found_issues(result_sets: Iterable[Iterable[FoundIssue]]) -> tuple[FoundIssue, ...]:
    """Merge parent-child search results, de-duplicating by issue number."""
    merged: dict[int, FoundIssue] = {}
    without_number: list[FoundIssue] = []
    for results in result_sets:
        for found in results:
            if found.number is None:
                if found not in without_number:
                    without_number.append(found)
                continue
            previous = merged.get(found.number)
            if previous is None:
                merged[found.number] = found
                continue
            # A repeated result can differ in completeness between queries;
            # preserve whichever non-empty representation is available.
            merged[found.number] = FoundIssue(
                number=found.number,
                title=previous.title or found.title,
                body=previous.body or found.body,
                url=previous.url or found.url,
            )
    return tuple(merged.values()) + tuple(without_number)


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
