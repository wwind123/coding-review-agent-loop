"""Deterministic risk-matrix row binding for untagged evidence-only findings (#1622).

A reviewer finding whose sole demand is admissible test evidence for approved
risk-matrix rows is citation-only only when it carries ``evidence_row_ids``
(#1324/#1338).  When a reviewer omits the tag, this module derives the binding
from the finding text alone, so the orchestrator never asks a model to guess
and never reads coder prose.  The derivation is pure: the same text and the
same :class:`EvidenceBindingContext` always give the same result, which is why
review validation, repair preservation and ledger minting share one context
object per reviewed head.

The rules run in a fixed order:

1. Evidence gate: without evidence-request vocabulary the text is
   ``not-evidence``.
2. Matrix reference: evidence vocabulary with no reference to the matrix (an
   approved row ID, a row-ID-shaped token after "row"/"rows", a collective
   phrase, or any matrix phrase) is ``not-evidence``.
3. Sole demand: any code-demand wording outside a closed list of negated
   phrases gives ``code-demand``; such a finding is never bound or repaired.
4. Cited rows: every row-ID-shaped reference must be enforceable; then the
   finding binds to exactly the cited enforceable rows.  An unknown reference
   gives ``needs-repair``/``unknown-row-reference``.
5. Collective reference without IDs: binds to every enforceable row only when
   the unsatisfied set is known and equals the whole enforceable set;
   otherwise ``needs-repair``/``ambiguous-collective``.
6. Any other matrix reference gives ``needs-repair``/``missing-row-reference``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Iterable, Sequence

BINDING_BOUND = "bound"
BINDING_NEEDS_REPAIR = "needs-repair"
BINDING_CODE_DEMAND = "code-demand"
BINDING_NOT_EVIDENCE = "not-evidence"

REASON_UNKNOWN_ROW_REFERENCE = "unknown-row-reference"
REASON_AMBIGUOUS_COLLECTIVE = "ambiguous-collective"
REASON_MISSING_ROW_REFERENCE = "missing-row-reference"


@dataclass(frozen=True)
class EvidenceBindingContext:
    """The per-head inputs of the derivation, built once and shared.

    ``enforceable_row_ids`` is the approved matrix's enforceable row set.
    ``unsatisfied_row_ids`` is the head-bound unsatisfied set, or ``None``
    when it is unknown (no coder record at the head, or unreadable history);
    only a collective reference depends on it.
    """

    enforceable_row_ids: tuple[str, ...]
    unsatisfied_row_ids: frozenset[str] | None = None


@dataclass(frozen=True)
class EvidenceRowBinding:
    kind: str
    rows: tuple[str, ...] = ()
    reason: str | None = None


_EVIDENCE_VOCABULARY_RE = re.compile(
    r"test[- ]observations?"
    r"|\brun-tests\b"
    r"|\b(?:admissible|passing|captured)\s+(?:[\w-]+\s+){0,2}?(?:evidence|observations?|citations?)\b"
    r"|risk[- ](?:test[- ])?matrix\s+evidence"
    r"|\bcitation\s+evidence\b"
)
_MATRIX_PHRASE_RE = re.compile(r"risk[- ](?:test[- ])?matrix(?:\s+rows?)?|\bmatrix\s+rows?\b")
_COLLECTIVE_RE = re.compile(
    r"risk[- ](?:test[- ])?matrix\s+rows\b"
    r"|\b(?:all|every|each)\s+(?:of\s+the\s+)?(?:(?:risk[- ](?:test[- ])?)?matrix\s+)?rows?\b"
    r"|\bthe\s+(?:\d+|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen)"
    r"\s+(?:(?:risk[- ](?:test[- ])?)?matrix\s+)?rows\b"
)
_ROW_TOKEN = r"(?:`[^`\n]+`|[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+)(?![\w-])"
_ROW_LIST_RE = re.compile(
    r"\brows?(?:\s*:\s*|\s+)(" + _ROW_TOKEN + r"(?:\s*(?:,|/|\band\b|\bor\b)\s*" + _ROW_TOKEN + r")*)"
)
_ROW_TOKEN_RE = re.compile(_ROW_TOKEN)
# The closed list of negated code-change phrases stripped before the
# code-demand scan.  Any other code-demand wording keeps the finding ordinary.
_NEGATED_CODE_CHANGE_RE = re.compile(
    r"\b(?:do\s+not|don['’]t|never)\s+change\s+(?:the\s+)?code\b"
    r"|\bno\s+code\s+changes?\b"
    r"|\bwithout\s+(?:changing|modifying)\s+(?:the\s+)?code\b"
    r"|\bnot\s+to\s+change\s+(?:the\s+)?code\b"
)
_CODE_DEMAND_RE = re.compile(
    r"\b(?:fix(?:es|ed|ing)?|change[sd]?|changing|modif(?:y|ies|ied|ying)"
    r"|update[sd]?|updating|implement(?:s|ed|ing)?|refactor\w*|rewrit\w*"
    r"|defects?|bugs?|regressions?|broken|incorrect(?:ly)?|wrong)\b"
    r"|\bmissing\s+(?:handling|guard)s?\b"
    r"|\b(?:add|remove)\s+(?:a\s+|the\s+)?code\s+paths?\b"
)


def _whole_token_re(row_id: str) -> re.Pattern[str]:
    return re.compile(r"(?<![\w-])" + re.escape(row_id) + r"(?![\w-])")


def _strip_token(token: str) -> str:
    return token[1:-1].strip() if token.startswith("`") and token.endswith("`") else token


def _row_references(text: str, enforceable: Sequence[str]) -> tuple[list[tuple[int, str]], list[str]]:
    """``(position, row id)`` for cited enforceable rows, plus unknown row-shaped references."""
    cited: list[tuple[int, str]] = []
    for row_id in enforceable:
        match = _whole_token_re(row_id).search(text)
        if match is not None:
            cited.append((match.start(), row_id))
    allowed = set(enforceable)
    unknown: list[str] = []
    for list_match in _ROW_LIST_RE.finditer(text):
        for token_match in _ROW_TOKEN_RE.finditer(list_match.group(1)):
            token = _strip_token(token_match.group(0))
            if token in allowed:
                position = list_match.start(1) + token_match.start()
                if all(row_id != token for _, row_id in cited):
                    cited.append((position, token))
            elif token and token not in unknown:
                unknown.append(token)
    return cited, unknown


def derive_evidence_row_binding(
    text: str, context: EvidenceBindingContext
) -> EvidenceRowBinding:
    """Classify one finding text; see the module docstring for the rule order."""
    lowered = text.lower()
    if not _EVIDENCE_VOCABULARY_RE.search(lowered):
        return EvidenceRowBinding(BINDING_NOT_EVIDENCE)
    enforceable = tuple(dict.fromkeys(context.enforceable_row_ids))
    cited, unknown = _row_references(text, enforceable)
    collective = bool(_COLLECTIVE_RE.search(lowered))
    if not (cited or unknown or collective or _MATRIX_PHRASE_RE.search(lowered)):
        return EvidenceRowBinding(BINDING_NOT_EVIDENCE)
    if _CODE_DEMAND_RE.search(_NEGATED_CODE_CHANGE_RE.sub(" ", lowered)):
        return EvidenceRowBinding(BINDING_CODE_DEMAND)
    if unknown:
        return EvidenceRowBinding(BINDING_NEEDS_REPAIR, reason=REASON_UNKNOWN_ROW_REFERENCE)
    if cited:
        return EvidenceRowBinding(
            BINDING_BOUND, rows=tuple(row_id for _, row_id in sorted(cited))
        )
    if collective:
        unsatisfied = context.unsatisfied_row_ids
        if enforceable and unsatisfied is not None and set(unsatisfied) == set(enforceable):
            return EvidenceRowBinding(BINDING_BOUND, rows=enforceable)
        return EvidenceRowBinding(BINDING_NEEDS_REPAIR, reason=REASON_AMBIGUOUS_COLLECTIVE)
    return EvidenceRowBinding(BINDING_NEEDS_REPAIR, reason=REASON_MISSING_ROW_REFERENCE)


def _entry_text_and_shape(entry: object) -> tuple[str, bool, bool] | None:
    """``(text, tagged, has_sub_items)`` of a raw or parsed blocking entry."""
    if isinstance(entry, str):
        return entry, False, False
    if isinstance(entry, dict):
        text = entry.get("text")
        if not isinstance(text, str):
            return None
        return text, bool(entry.get("evidence_row_ids")), bool(entry.get("sub_items"))
    text = getattr(entry, "text", None)
    if not isinstance(text, str):
        return None
    return (
        text,
        bool(getattr(entry, "evidence_row_ids", ())),
        bool(getattr(entry, "sub_items", ())),
    )


def flagged_finding_texts(
    blocking_items: Iterable[object] | None, context: EvidenceBindingContext | None
) -> tuple[str, ...]:
    """Texts of untagged, sub-item-free blocking entries that derive ``needs-repair``.

    Accepts raw review JSON entries (strings or objects) or parsed items, so
    repair preservation computes the flagged set from the original response
    itself, never from error text.  No context flags nothing.
    """
    if context is None:
        return ()
    flagged: list[str] = []
    for entry in blocking_items or ():
        shape = _entry_text_and_shape(entry)
        if shape is None:
            continue
        text, tagged, has_sub_items = shape
        if tagged or has_sub_items:
            continue
        if derive_evidence_row_binding(text, context).kind == BINDING_NEEDS_REPAIR:
            flagged.append(text)
    return tuple(flagged)


def derived_evidence_row_ids(
    text: str,
    *,
    reviewer_tag: Sequence[str],
    has_sub_items: bool,
    context: EvidenceBindingContext | None,
) -> tuple[str, ...]:
    """The tag a newly minted blocking finding carries.

    A reviewer tag always wins verbatim; otherwise only a ``bound``
    derivation for a sub-item-free finding supplies one.
    """
    if reviewer_tag:
        return tuple(reviewer_tag)
    if context is None or has_sub_items:
        return ()
    binding = derive_evidence_row_binding(text, context)
    return binding.rows if binding.kind == BINDING_BOUND else ()
