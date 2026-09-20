"""Bound tool-created GitHub bodies that embed plan-derived text.

A plan large enough to deserve splitting is large enough to make the issues
generated from it unpublishable: the embedded plan excerpt, proposed scope or
reviewer note pushes the rendered body past GitHub's body limit and creation
fails with a generic "body exceeds 60000 characters" error (#902).

Every tool-created body that embeds such text renders it through
:func:`fit_github_body`, which shortens the plan-derived sections in place —
largest first, keeping the surrounding contract sections and markers intact —
and points at the canonical source for the complete text.  When a body still
does not fit, the raised diagnostic names the surface and the section that
overflowed instead of the generic size error.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .errors import AgentLoopError
from .round_transport import MAX_GITHUB_BODY_CHARS

#: Headroom kept when shortening so a section boundary, the notice or a
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


def shortened_section(section: BoundedSection, *, budget: int) -> str:
    """Return ``section.text`` cut to ``budget`` characters with a pointer."""
    if len(section.text) <= budget:
        return section.text
    notice = (
        f"[Shortened to fit the GitHub body limit; only the opening of the "
        f"{section.name} is repeated here. The complete text is in {section.pointer}.]"
    )
    keep = max(budget - len(notice) - 2, 0)
    return f"{section.text[:keep].rstrip()}\n\n{notice}"


def fit_github_body(
    body: str,
    *,
    sections: Sequence[BoundedSection],
    surface: str,
    limit: int = MAX_GITHUB_BODY_CHARS,
    margin: int = BODY_SAFETY_MARGIN,
) -> str:
    """Shorten plan-derived sections until ``body`` fits GitHub's body limit.

    ``sections`` are shortened largest first so a single oversized excerpt is
    cut before smaller ones lose any text.  Raises :class:`AgentLoopError` with
    a surface- and section-specific diagnostic when the body still overflows.
    """
    if len(body) <= limit:
        return body
    ordered = sorted(sections, key=lambda section: len(section.text), reverse=True)
    shortened: list[str] = []
    for section in ordered:
        if len(body) <= limit:
            break
        if not section.text or section.text not in body:
            continue
        overflow = len(body) - limit
        budget = max(len(section.text) - overflow - margin, 0)
        replacement = shortened_section(section, budget=budget)
        if len(replacement) >= len(section.text):
            continue
        body = body.replace(section.text, replacement, 1)
        shortened.append(section.name)
    if len(body) <= limit:
        return body
    largest = max(
        (section for section in ordered if section.text and section.text in body),
        key=lambda section: len(section.text),
        default=None,
    )
    if largest is not None:
        culprit = f"the {largest.name} section ({len(largest.text)} characters) still does not fit"
    elif shortened:
        culprit = (
            "its fixed contract sections are too large to publish even with "
            + ", ".join(sorted(set(shortened)))
            + " shortened"
        )
    else:
        culprit = "it has no shortenable plan-derived section"
    raise AgentLoopError(
        f"{surface} exceeds the GitHub body limit of {limit} characters by "
        f"{len(body) - limit} characters: {culprit}."
    )
