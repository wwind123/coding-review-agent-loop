"""Bound tool-created GitHub bodies that embed plan-derived text.

A plan large enough to deserve splitting is large enough to make the issues
generated from it unpublishable: the embedded plan excerpt, proposed scope or
reviewer note pushes the rendered body past GitHub's body limit and creation
fails with a generic "body exceeds 60000 characters" error (#902).

Every tool-created body that embeds such text renders it through
:func:`fit_github_body`.  The caller registers each plan-derived span in render
order as a :class:`BoundedSection`; the limiter shares the remaining room out
between them, so no single span can starve the others, keeps the surrounding
contract sections and markers intact, and points at the canonical source for
the complete text.  When a body still does not fit, the raised diagnostic names
the surface and the section that overflowed instead of the generic size error.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .errors import AgentLoopError
from .round_transport import MAX_GITHUB_BODY_CHARS

#: Headroom kept when shortening so a section boundary, a pointer notice or a
#: trailing marker cannot push a just-fitting body back over the limit.
BODY_SAFETY_MARGIN = 1_000


@dataclass(frozen=True)
class BoundedSection:
    """A plan-derived span of a rendered body that may be shortened."""

    name: str
    """Human-readable section name, used in the overflow diagnostic."""

    text: str
    """The exact text as it appears in the rendered body."""

    pointer: str
    """Where the complete text remains available."""


def _notice(section: BoundedSection) -> str:
    return (
        f"[Shortened to fit the GitHub body limit; only the opening of the "
        f"{section.name} is repeated here. The complete text is in {section.pointer}.]"
    )


def shortened_section(section: BoundedSection, *, budget: int) -> str:
    """Return ``section.text`` cut to ``budget`` characters with a pointer.

    The replacement stays within ``budget`` whenever the budget can hold the
    pointer notice at all; below that it degrades to the notice alone, which is
    the shortest replacement that still says where the full text lives.
    """
    if len(section.text) <= budget:
        return section.text
    notice = _notice(section)
    keep = max(budget - len(notice) - 2, 0)
    if keep == 0:
        return notice
    return f"{section.text[:keep].rstrip()}\n\n{notice}"


def _located_sections(
    body: str, sections: Sequence[BoundedSection]
) -> list[tuple[int, BoundedSection]]:
    """Locate each registered section in ``body``, in render order.

    Sections are matched left to right so a short span that also occurs inside
    a longer one cannot be spliced into the wrong place.
    """
    located: list[tuple[int, BoundedSection]] = []
    cursor = 0
    for section in sections:
        if not section.text:
            continue
        index = body.find(section.text, cursor)
        if index < 0:
            continue
        located.append((index, section))
        cursor = index + len(section.text)
    return located


def _budgets(
    located: Sequence[tuple[int, BoundedSection]], *, available: int
) -> dict[int, int]:
    """Share ``available`` characters between sections, smallest need first.

    Sections that already fit their equal share keep their full text and hand
    the surplus to the larger ones, so one oversized excerpt cannot reduce
    every other section to a bare pointer.
    """
    budgets: dict[int, int] = {}
    order = sorted(located, key=lambda item: len(item[1].text))
    remaining = available
    for position, (index, section) in enumerate(order):
        share = remaining // (len(order) - position)
        budget = min(len(section.text), max(share, 0))
        budgets[index] = budget
        remaining -= budget
    return budgets


def fit_github_body(
    body: str,
    *,
    sections: Sequence[BoundedSection],
    surface: str,
    limit: int = MAX_GITHUB_BODY_CHARS,
    margin: int = BODY_SAFETY_MARGIN,
) -> str:
    """Shorten plan-derived sections until ``body`` fits GitHub's body limit.

    ``sections`` must be registered in the order they appear in ``body``.
    Raises :class:`AgentLoopError` with a surface- and section-specific
    diagnostic when the body still overflows.
    """
    if len(body) <= limit:
        return body
    located = _located_sections(body, sections)
    embedded = sum(len(section.text) for _index, section in located)
    fixed = len(body) - embedded
    available = limit - margin - fixed
    if located and available > 0:
        budgets = _budgets(located, available=available)
        # Splice from the end so the earlier offsets stay valid.
        for index, section in sorted(located, key=lambda item: item[0], reverse=True):
            replacement = shortened_section(section, budget=budgets[index])
            body = body[:index] + replacement + body[index + len(section.text):]
        if len(body) <= limit:
            return body
    raise AgentLoopError(_overflow_diagnostic(body, located, surface=surface, limit=limit))


def _overflow_diagnostic(
    body: str,
    located: Sequence[tuple[int, BoundedSection]],
    *,
    surface: str,
    limit: int,
) -> str:
    excess = len(body) - limit
    if not located:
        culprit = (
            "it embeds no shortenable plan-derived section, so its fixed contract "
            "text is already too large to publish"
        )
    else:
        largest = max(located, key=lambda item: len(item[1].text))[1]
        culprit = (
            f"even with its {len(located)} plan-derived section(s) shortened it does not "
            f"fit; the largest is the {largest.name} section "
            f"({len(largest.text)} characters) and the surrounding fixed contract "
            "text leaves no room for it"
        )
    return (
        f"{surface} exceeds the GitHub body limit of {limit} characters by "
        f"{excess} characters: {culprit}."
    )
