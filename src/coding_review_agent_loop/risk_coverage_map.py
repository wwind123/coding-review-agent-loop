"""Deterministic risk-matrix coverage-map completeness check (#1290).

The coder maps every enforceable approved-matrix row to the test(s) covering it
(``risk_test_matrix_claims`` carrying ``test_level``) or declares the row an
explicit gap (``risk_test_matrix_coverage_gaps``).  This module only verifies
completeness and well-formedness:

* every enforceable row is claimed or declared as a gap,
* every cited test path is a regular file in one pinned committed tree, and
* each declared ``test_level`` meets the row's classified required level.

It never runs tests and never judges whether a cited test really exercises its
row; that semantic accuracy stays with the reviewer.  The assessment is
display/gating data only: it is distinct from canonical risk evidence and is
never persisted as authority.
"""

from __future__ import annotations

import posixpath
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .protocol_markers import sanitize_untrusted_prose

LEVEL_RANK = {"unit": 1, "integration": 2, "workflow": 3}

CODE_MISSING_ROW = "missing-row"
CODE_AMBIGUOUS = "ambiguous"
CODE_MISSING_TEST_REFERENCE = "missing-test-reference"
CODE_MISSING_ASSERTION_CLAIM = "missing-assertion-claim"
CODE_NONEXISTENT_TEST_PATH = "nonexistent-test-path"
CODE_MISSING_LEVEL = "missing-level"
CODE_UNDER_LEVEL = "under-level"

_DETAIL_MAX_CHARS = 200
_REGULAR_FILE_MODES = frozenset({"100644", "100755"})

_WORKFLOW_KEYWORDS = ("workflow", "orchestrat", "end-to-end", "e2e")
_INTEGRATION_KEYWORDS = ("integration",)
_UNIT_KEYWORDS = ("unit", "protocol", "parser", "helper")


def classify_required_level(text: object) -> str | None:
    """Classify a row's free-text ``proposed_test_level`` by keyword precedence."""
    if not isinstance(text, str):
        return None
    lowered = text.lower()
    for level, keywords in (
        ("workflow", _WORKFLOW_KEYWORDS),
        ("integration", _INTEGRATION_KEYWORDS),
        ("unit", _UNIT_KEYWORDS),
    ):
        if any(keyword in lowered for keyword in keywords):
            return level
    return None


def coverage_map_applies(plan_context: object) -> bool:
    """True when the approved-plan context carries an applicable, delivered matrix
    with at least one enforceable row for this turn.  Everything else (no plan,
    no/unavailable/not-applicable matrix) is unaffected by the coverage map."""
    if plan_context is None or not getattr(plan_context, "matrix_available", False):
        return False
    payload = getattr(plan_context, "risk_test_matrix_payload", None)
    if not isinstance(payload, Mapping) or payload.get("applicability") != "applicable":
        return False
    row_ids = getattr(plan_context, "risk_test_matrix_expected_row_ids", None)
    return bool(row_ids) and bool(enforceable_rows(payload, row_ids))


def _bounded(text: object) -> str:
    """Flatten, neutralize reserved syntax, and clip decoded display text.

    JSON-decoded coder prose can restore marker syntax a raw-text guard never
    saw, so it is sanitized both before and after clipping (#1290).
    """
    flat = " ".join(sanitize_untrusted_prose(str(text)).split())
    if len(flat) > _DETAIL_MAX_CHARS:
        flat = flat[: _DETAIL_MAX_CHARS - 1] + "…"
    safe = " ".join(sanitize_untrusted_prose(flat).split())
    # No HTML-comment syntax at all: state footers and other comment-shaped
    # records are never display text.
    return _COVERAGE_HEADING_RE.sub(
        "risk-matrix coverage-map (quoted)", safe.replace("<", "&lt;").replace(">", "&gt;")
    )


def normalize_test_path(reference: str, *, identifier: bool) -> str | None:
    """Return a repo-relative POSIX path, or ``None`` when it is not admissible.

    For an identifier the path is the part before the first ``::``.  Absolute
    paths, ``..`` segments, empty paths, and NUL bytes are refused.
    """
    if not isinstance(reference, str) or "\x00" in reference:
        return None
    path = reference.split("::", 1)[0] if identifier else reference
    path = path.strip().replace("\\", "/")
    if not path or path.startswith("/"):
        return None
    if any(segment == ".." for segment in path.split("/")):
        return None
    normalized = posixpath.normpath(path)
    if normalized in {"", "."} or normalized.startswith("../"):
        return None
    return normalized


@dataclass
class CommittedTree:
    """Path probe pinned to one resolved commit SHA.

    ``revision`` is ``None`` for an unavailable tree (the commit object is
    missing locally), which makes every assessment against it ``unverified``.
    A path exists only when ``git ls-tree`` returns exactly one blob entry with
    a regular-file mode; directories, symlinks and gitlinks do not.
    """

    workdir: Path
    revision: str | None
    _memo: dict[str, bool] = field(default_factory=dict, repr=False)

    @property
    def available(self) -> bool:
        return self.revision is not None

    @classmethod
    def resolve(cls, workdir: Path | str, revision_spec: str | None) -> "CommittedTree":
        workdir = Path(workdir)
        if not revision_spec or not isinstance(revision_spec, str) or revision_spec.startswith("-"):
            return cls(workdir, None)
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--verify", "--quiet", f"{revision_spec}^{{commit}}"],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return cls(workdir, None)
        sha = result.stdout.strip()
        if result.returncode != 0 or len(sha) not in {40, 64} or not all(c in "0123456789abcdef" for c in sha):
            return cls(workdir, None)
        return cls(workdir, sha)

    def is_regular_file(self, path: str) -> bool:
        if self.revision is None:
            return False
        if path in self._memo:
            return self._memo[path]
        exists = False
        try:
            result = subprocess.run(
                ["git", "--literal-pathspecs", "ls-tree", "-z", self.revision, "--", path],
                cwd=self.workdir,
                capture_output=True,
                timeout=30,
                check=False,
            )
            if result.returncode == 0:
                entries = [entry for entry in result.stdout.split(b"\x00") if entry]
                if len(entries) == 1:
                    meta, _, name = entries[0].partition(b"\t")
                    parts = meta.decode("ascii", "replace").split()
                    exists = (
                        len(parts) == 3
                        and parts[1] == "blob"
                        and parts[0] in _REGULAR_FILE_MODES
                        and name.decode("utf-8", "replace") == path
                    )
        except (OSError, subprocess.SubprocessError):
            exists = False
        self._memo[path] = exists
        return exists


@dataclass(frozen=True)
class CoverageDeficiency:
    row_id: str
    code: str
    detail: str = ""


@dataclass(frozen=True)
class CoverageRowEntry:
    """Per-row display data for the reviewer's coverage map."""

    row_id: str
    label: str
    required_level: str | None
    required_level_text: str
    kind: str  # "claim" | "gap" | "missing" | "ambiguous"
    test_references: tuple[str, ...] = ()
    declared_level: str | None = None
    gap_reason: str = ""
    gap_correction: str = ""


@dataclass(frozen=True)
class CoverageAssessment:
    """Result of one deterministic check against one pinned revision."""

    revision: str | None
    deficiencies: tuple[CoverageDeficiency, ...] = ()
    entries: tuple[CoverageRowEntry, ...] = ()

    @property
    def tree_available(self) -> bool:
        return self.revision is not None

    @property
    def status(self) -> str:
        if not self.tree_available:
            return "unverified"
        return "incomplete" if self.deficiencies else "complete"

    @property
    def deficient_row_ids(self) -> tuple[str, ...]:
        seen: list[str] = []
        for item in self.deficiencies:
            if item.row_id not in seen:
                seen.append(item.row_id)
        return tuple(seen)


def enforceable_rows(
    matrix_payload: Mapping[str, object] | None,
    enforceable_row_ids: Iterable[str] | None,
) -> tuple[Mapping[str, object], ...]:
    """Return enforceable applicable rows in matrix-row order."""
    if not isinstance(matrix_payload, Mapping) or enforceable_row_ids is None:
        return ()
    allowed = set(enforceable_row_ids)
    rows = matrix_payload.get("rows", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return ()
    return tuple(
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("applicability") in {"applicable", "required"}
        and str(row.get("row_id")) in allowed
    )


def check_coverage_map(
    matrix_payload: Mapping[str, object] | None,
    *,
    enforceable_row_ids: Iterable[str] | None,
    claims: Sequence[object] | None,
    gaps: Sequence[object] | None,
    tree: CommittedTree,
) -> CoverageAssessment:
    """Check completeness and well-formedness of the coder's coverage map.

    ``claims`` are ``SemanticRiskCoverageClaim``-like objects and ``gaps`` are
    ``SemanticRiskCoverageGap``-like objects (duck-typed to keep this module
    free of protocol imports).  Deficiencies are reported in matrix-row order.
    Never executes tests.
    """
    claim_by_row: dict[str, object] = {}
    for claim in claims or ():
        claim_by_row.setdefault(str(getattr(claim, "row_id", "")), claim)
    gap_by_row: dict[str, object] = {}
    for gap in gaps or ():
        gap_by_row.setdefault(str(getattr(gap, "row_id", "")), gap)

    deficiencies: list[CoverageDeficiency] = []
    entries: list[CoverageRowEntry] = []
    for row in enforceable_rows(matrix_payload, enforceable_row_ids):
        row_id = str(row.get("row_id"))
        label = _bounded(row.get("label", ""))
        level_text = str(row.get("proposed_test_level", "") or "")
        required = classify_required_level(level_text)
        claim = claim_by_row.get(row_id)
        gap = gap_by_row.get(row_id)

        def add(code: str, detail: str = "") -> None:
            deficiencies.append(CoverageDeficiency(row_id, code, _bounded(detail)))

        if claim is None and gap is None:
            add(CODE_MISSING_ROW, "neither a test claim nor a declared gap")
            entries.append(CoverageRowEntry(row_id, label, required, level_text, "missing"))
            continue
        if claim is not None and gap is not None:
            add(CODE_AMBIGUOUS, "row is both claimed and declared as a gap")
            entries.append(CoverageRowEntry(row_id, label, required, level_text, "ambiguous"))
            continue
        if gap is not None:
            entries.append(
                CoverageRowEntry(
                    row_id,
                    label,
                    required,
                    level_text,
                    "gap",
                    gap_reason=_bounded(getattr(gap, "reason", "")),
                    gap_correction=_bounded(getattr(gap, "proposed_correction", "")),
                )
            )
            continue

        identifiers = tuple(getattr(claim, "test_identifiers", ()) or ())
        locations = tuple(getattr(claim, "test_locations", ()) or ())
        references = (*identifiers, *locations)
        declared = getattr(claim, "test_level", None)
        if not references:
            add(CODE_MISSING_TEST_REFERENCE, "no test_identifiers or test_locations cited")
        else:
            bad_paths: list[str] = []
            seen: set[str] = set()
            for reference, is_identifier in (
                *((item, True) for item in identifiers),
                *((item, False) for item in locations),
            ):
                path = normalize_test_path(reference, identifier=is_identifier)
                key = path if path is not None else f"\x00{reference}"
                if key in seen:
                    continue
                seen.add(key)
                if path is None or not tree.is_regular_file(path):
                    bad_paths.append(path if path is not None else str(reference))
            if bad_paths and tree.available:
                add(CODE_NONEXISTENT_TEST_PATH, "not a committed regular file: " + ", ".join(bad_paths))
        if not str(getattr(claim, "workflow_path_claim", "") or "").strip() or not tuple(
            getattr(claim, "outcome_assertions", ()) or ()
        ):
            add(CODE_MISSING_ASSERTION_CLAIM, "workflow_path_claim and outcome_assertions are required")
        if required is not None:
            if declared is None:
                add(CODE_MISSING_LEVEL, f"declare test_level (required: {required})")
            elif LEVEL_RANK[declared] < LEVEL_RANK[required]:
                add(CODE_UNDER_LEVEL, f"declared {declared}, required {required}")
        entries.append(
            CoverageRowEntry(
                row_id,
                label,
                required,
                level_text,
                "claim",
                test_references=tuple(_bounded(item) for item in references),
                declared_level=declared,
            )
        )
    return CoverageAssessment(tree.revision, tuple(deficiencies), tuple(entries))


def render_coverage_map(
    assessment: CoverageAssessment | None,
    *,
    reask_used: bool = False,
    discard_note: str | None = None,
) -> str:
    """Render the orchestrator-authored coverage-map section (display only)."""
    if assessment is None:
        return ""
    lines = ["### Risk-matrix coverage map"]
    lines.append(
        "Orchestrator completeness check (existence and level only; whether each "
        "cited test really exercises its row is for the reviewer to judge)."
    )
    for entry in assessment.entries:
        required = entry.required_level or "unclassified (no level check)"
        if entry.kind == "claim":
            declared = entry.declared_level or "undeclared"
            refs = ", ".join(f"`{ref}`" for ref in entry.test_references) or "(none cited)"
            lines.append(
                f"- `{_bounded(entry.row_id)}`: tests {refs}; declared level {declared} vs required {required}"
            )
        elif entry.kind == "gap":
            lines.append(
                f"- `{_bounded(entry.row_id)}`: declared gap (required {required}) - reason: "
                f"{entry.gap_reason}; proposed correction: {entry.gap_correction}"
            )
        elif entry.kind == "ambiguous":
            lines.append(f"- `{_bounded(entry.row_id)}`: claimed and declared as a gap (required {required})")
        else:
            lines.append(f"- `{_bounded(entry.row_id)}`: no test claim and no declared gap (required {required})")
    suffix = " after one coverage re-ask" if reask_used else ""
    if assessment.status == "unverified":
        status = "unverified: authenticated PR tree unavailable" + suffix
    elif assessment.status == "complete":
        status = (
            f"complete at {_bounded(assessment.revision)} "
            f"(deterministic completeness check passed){suffix}"
        )
    else:
        status = (
            f"incomplete at {_bounded(assessment.revision)}{suffix}: "
            + "; ".join(f"{_bounded(item.row_id)}: {item.code}" for item in assessment.deficiencies)
        )
    lines.append(f"- **Status:** {status}")
    if discard_note:
        lines.append(f"- Note: {_bounded(discard_note)}")
    return "\n".join(lines)


COVERAGE_MAP_HEADING = "### Risk-matrix coverage map"


_COVERAGE_HEADING_RE = re.compile(r"risk-matrix\s+coverage\s+map", re.IGNORECASE)


def neutralize_coverage_map_heading(text: str) -> str:
    """Rewrite the coverage-map heading phrase in untrusted text so it can never be
    mistaken for the orchestrator-authored section."""
    return _COVERAGE_HEADING_RE.sub("risk-matrix coverage-map (quoted)", text)


_ANCHOR_HEADINGS = (
    "\n### Result\n",  # implementation comments
    "\n### Addressed items\n",  # coder follow-up comments
    "\n### Remaining items\n",
    "\n### Disputed items\n",
)
_STATUS_RE = re.compile(
    r"- \*\*Status:\*\* (?:(?:complete|incomplete) at (?P<rev>[0-9a-f]{7,64})\b|unverified:)"
)


def extract_coverage_map_section(
    body: str | None, *, expected_revision: str | None = None
) -> str | None:
    """Return the orchestrator-authored coverage-map section of a stored round comment.

    Stored comments may predate the rendering boundary that neutralizes the
    heading in coder prose, so the section is accepted only when it is
    unambiguous: the heading phrase occurs exactly once in the whole body, as a
    section heading after the last result heading, with a well-formed status
    line (naming ``expected_revision`` when one is supplied).  Anything else is
    conservatively omitted rather than guessed at (#1290).
    """
    if not body or len(_COVERAGE_HEADING_RE.findall(body)) != 1:
        return None
    marker = "\n\n" + COVERAGE_MAP_HEADING + "\n"
    index = body.find(marker)
    if index == -1:
        return None
    result_index = max(body.rfind(anchor) for anchor in _ANCHOR_HEADINGS)
    if result_index == -1 or index < result_index:
        return None
    section = body[index + 2 :]
    cut = len(section)
    for cut_marker in ("\n\n### ", "\n\n## ", "\n\n<!--", "\n\n--"):
        found = section.find(cut_marker)
        if found != -1:
            cut = min(cut, found)
    section = section[:cut].strip()
    status = _STATUS_RE.search(section)
    if status is None:
        return None
    revision = status.group("rev")
    if expected_revision and revision and not str(expected_revision).startswith(revision):
        return None
    return section
