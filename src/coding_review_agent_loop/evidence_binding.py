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
4. Cited rows: every row-ID-shaped reference must be enforceable, and a row
   list is used only when fully interpreted; then the
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


# Keyword recognition is case-insensitive everywhere (#1622 review): "Rows",
# "AND" or "Risk-Matrix" must never hide a row reference.  Row-ID identity is
# not: IDs are compared exactly, with their original case.
_I = re.IGNORECASE
_EVIDENCE_VOCABULARY_RE = re.compile(
    r"test[- ]observations?"
    r"|\brun-tests\b"
    r"|\b(?:admissible|passing|captured)\s+(?:[\w-]+\s+){0,2}?(?:evidence|observations?|citations?)\b"
    r"|risk[- ](?:test[- ])?matrix\s+evidence"
    r"|\bcitation\s+evidence\b",
    _I,
)
_MATRIX_PHRASE_RE = re.compile(
    r"risk[- ](?:test[- ])?matrix(?:\s+rows?)?|\bmatrix\s+rows?\b", _I
)
_COLLECTIVE_RE = re.compile(
    r"risk[- ](?:test[- ])?matrix\s+rows\b"
    r"|\b(?:all|every|each)\s+(?:of\s+the\s+)?(?:(?:risk[- ](?:test[- ])?)?matrix\s+)?rows?\b"
    r"|\bthe\s+(?:\d+|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen)"
    r"\s+(?:(?:risk[- ](?:test[- ])?)?matrix\s+)?rows\b",
    _I,
)
_ROW_TOKEN = r"(?:`[^`\n]+`|[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+)(?![\w-])"
# One or more separators between list tokens, so ", and", ", or", "and/or",
# "; and", "&" and "as well as" all continue the list (#1622 review item-2).
_ROW_SEPARATOR = (
    r"(?:\s*(?:,|;|/|&|\+|\band\b|\bor\b|\bplus\b|\bas\s+well\s+as\b)\s*)+"
)
_ROW_LIST_RE = re.compile(
    r"\brows?(?:\s*:\s*|\s+)(" + _ROW_TOKEN + r"(?:" + _ROW_SEPARATOR + _ROW_TOKEN + r")*)",
    _I,
)
_ROW_TOKEN_RE = re.compile(_ROW_TOKEN)
# A token in the clause after a parsed row list: backticked, or a bare
# kebab/snake identifier shaped like a row ID.
_TAIL_TOKEN_RE = re.compile(r"`[^`\n]+`|(?<![\w-])[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+(?![\w-])")
# Patterns that EXCUSE a token from being a row reference match whole tokens
# only (#1622 review item-5): a hyphen or underscore continues a kebab/snake
# identifier, so `node-startup` or `run-smoke` is never read as the command
# word `node` or `run`.  Patterns that make the result more conservative (the
# code-demand scan) keep ordinary word boundaries.
_TOKEN_START = r"(?<![\w-])"
_TOKEN_END = r"(?![\w-])"
# Words that introduce a command; the command span runs to the next list
# continuation (comma, semicolon, "and", "or") or the end of the clause.
_COMMAND_INTRO_RE = re.compile(
    _TOKEN_START
    + r"(?:using|via|through|run|running|re-?run|re-?running|execute|executing"
    r"|invoke|invoking|command)"
    + _TOKEN_END,
    _I,
)
_COMMAND_SPAN_END_RE = re.compile(r",|;|" + _TOKEN_START + r"(?:and|or)" + _TOKEN_END, _I)
# A backticked span is a command, not a row ID, when it holds whitespace or
# path/assignment characters, or is or starts with a whole tool-name token.
_COMMAND_CONTENT_RE = re.compile(
    r"\s|[/.=$]|^(?:agent-loop|python\d*|pytest|pip|npm|npx|pnpm|yarn|node|git|gh|uv|make|bash|sh)"
    r"(?=\s|$)",
    _I,
)
# The end of the clause a row list belongs to: sentence punctuation followed
# by whitespace or the end of the text, or a line break.
_CLAUSE_END_RE = re.compile(r"[.!?](?=\s|$)|\n")
# The closed list of negated code-change phrases stripped before the
# code-demand scan.  Any other code-demand wording keeps the finding ordinary.
_NEGATED_CODE_CHANGE_RE = re.compile(
    _TOKEN_START
    + r"(?:(?:do\s+not|don['’]t|never)\s+change\s+(?:the\s+)?code"
    r"|no\s+code\s+changes?"
    r"|without\s+(?:changing|modifying)\s+(?:the\s+)?code"
    r"|not\s+to\s+change\s+(?:the\s+)?code)"
    + _TOKEN_END,
    _I,
)
_CODE_DEMAND_RE = re.compile(
    r"\b(?:fix(?:es|ed|ing)?|change[sd]?|changing|modif(?:y|ies|ied|ying)"
    r"|update[sd]?|updating|implement(?:s|ed|ing)?|refactor\w*|rewrit\w*"
    r"|defects?|bugs?|regressions?|broken|incorrect(?:ly)?|wrong)\b"
    r"|\bmissing\s+(?:handling|guard)s?\b"
    r"|\b(?:add|remove)\s+(?:a\s+|the\s+)?code\s+paths?\b",
    _I,
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
        # A list is bound only when it is fully interpreted.  A row-shaped
        # token left in the same clause after the parsed list, bare or
        # backticked, may be a row the grammar did not reach, so an unknown one
        # forces a binding repair instead of a partial binding.  Command spans
        # are not row references and never count.
        clause_end = _CLAUSE_END_RE.search(text, list_match.end())
        tail = text[list_match.end(): clause_end.start() if clause_end else len(text)]
        for token in _unparsed_row_candidates(tail):
            if token not in allowed and token not in unknown:
                unknown.append(token)
    return cited, unknown


def _command_spans(tail: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for intro in _COMMAND_INTRO_RE.finditer(tail):
        end = _COMMAND_SPAN_END_RE.search(tail, intro.end())
        spans.append((intro.start(), end.start() if end else len(tail)))
    return spans


def _unparsed_row_candidates(tail: str) -> list[str]:
    """Potential row references in the clause after a parsed row list."""
    spans = _command_spans(tail)
    candidates: list[str] = []
    for match in _TAIL_TOKEN_RE.finditer(tail):
        if any(start <= match.start() < end for start, end in spans):
            continue
        raw = match.group(0)
        token = _strip_token(raw)
        if not token or _COMMAND_INTRO_RE.fullmatch(token):
            continue
        if raw.startswith("`") and _COMMAND_CONTENT_RE.search(token):
            continue
        candidates.append(token)
    return candidates


def derive_evidence_row_binding(
    text: str, context: EvidenceBindingContext
) -> EvidenceRowBinding:
    """Classify one finding text; see the module docstring for the rule order."""
    if not _EVIDENCE_VOCABULARY_RE.search(text):
        return EvidenceRowBinding(BINDING_NOT_EVIDENCE)
    enforceable = tuple(dict.fromkeys(context.enforceable_row_ids))
    cited, unknown = _row_references(text, enforceable)
    collective = bool(_COLLECTIVE_RE.search(text))
    if not (cited or unknown or collective or _MATRIX_PHRASE_RE.search(text)):
        return EvidenceRowBinding(BINDING_NOT_EVIDENCE)
    if _CODE_DEMAND_RE.search(_NEGATED_CODE_CHANGE_RE.sub(" ", text)):
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
