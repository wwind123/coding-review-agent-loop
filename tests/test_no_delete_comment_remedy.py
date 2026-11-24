"""No stop message tells the operator to delete comments to recover a round (#2064, #2081).

A partially published review round closes as incomplete and rolls over to the
next round; nothing ever needs deleting.  Two guards keep it that way:

* a fixed set of round-recovery deletion phrases (the removed partial-round
  recovery listing's wording and close variants) must match nothing anywhere
  in ``src/``, comments included;
* a broad operator delete-remedy pattern (an imperative "delete" of a comment,
  record, marker, review, attachment or checkpoint in a message string) may
  match only an explicit allowlist of existing remedies that are not round
  recovery.  Each entry names its file, its exact match count and why it is
  out of scope; a new match anywhere, including a new one in an allowlisted
  file, fails.  The allowlisted remedies are unchanged by #2064.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "coding_review_agent_loop"

ROUND_RECOVERY_PATTERNS = (
    re.compile(r"delete these\b", re.IGNORECASE),
    re.compile(r"partial-round recovery", re.IGNORECASE),
    re.compile(r"delete the listed", re.IGNORECASE),
    re.compile(r"whole round runs again", re.IGNORECASE),
    re.compile(r"records before deleting", re.IGNORECASE),
    re.compile(r"no deletion (?:list|set)", re.IGNORECASE),
    re.compile(r"deletion (?:list|set|targets?)\b", re.IGNORECASE),
    re.compile(r"recovery list", re.IGNORECASE),
    re.compile(r"comments whose deletion", re.IGNORECASE),
    re.compile(r"deleting (?:the|these|its) (?:round|reviews?|records?|comments?)\b", re.IGNORECASE),
)

BROAD_DELETE_REMEDY = re.compile(
    r"\bdelet(?:e|ing)\b[^.;]{0,80}?\b(?:comment|record|marker|review|attachment|checkpoint)s?\b",
    re.IGNORECASE,
)

# file -> (exact number of message strings matching BROAD_DELETE_REMEDY, justification)
ALLOWLIST: dict[str, tuple[int, str]] = {
    "panel_evidence.py": (3, (
        "Signed human amendment comment posted under an unread amendment: the operator "
        "deletes and reposts their own signed input; not round recovery."
    )),
    "board_amendment.py": (3, (
        "Signed human board-amendment record that cannot apply: the operator removes "
        "their own signed board record and changes the reviewer flags; not round recovery."
    )),
    "board_change.py": (2, (
        "Signed human board-change record that cannot apply: same operator-owned signed "
        "board input as board_amendment.py; not round recovery."
    )),
    "pr_board_state.py": (2, (
        "Conflicting issue-to-PR handoff records: association repair of handoff records, "
        "not review-round state."
    )),
    "review_images_state.py": (2, (
        "Unbound visible-defects lifecycle record: an opt-in visual-review record outside "
        "the review round."
    )),
    "managed_ci.py": (5, (
        "Malformed managed-CI intent comments: the managed-CI authorization area is "
        "operator-declared out of scope for #2064."
    )),
}


def _source_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                ids.add(id(body[0].value))
    return ids


def message_strings(source: str) -> list[str]:
    """Every non-docstring string literal; an f-string's placeholders become ``{}``."""
    tree = ast.parse(source)
    docstrings = _docstring_nodes(tree)
    inside_fstring: set[int] = set()
    strings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            parts = []
            for value in node.values:
                inside_fstring.add(id(value))
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
                else:
                    parts.append("{}")
            strings.append("".join(parts))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstrings and id(node) not in inside_fstring):
            strings.append(node.value)
    return strings


def broad_matches(source: str) -> list[str]:
    return [text for text in message_strings(source) if BROAD_DELETE_REMEDY.search(text)]


def test_source_has_no_round_recovery_delete_remedy():
    hits = []
    for path in _source_files():
        text = path.read_text(encoding="utf-8")
        for pattern in ROUND_RECOVERY_PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                hits.append(f"{path.relative_to(ROOT)}:{line}: {match.group(0)!r}")
    assert hits == [], "round-recovery delete remedy found:\n" + "\n".join(hits)


def test_removed_remedy_modules_and_classes_are_gone():
    for module in ("partial_round_recovery.py", "signed_input_supersession.py", "publication_resume.py"):
        assert not (SRC / module).exists(), module
    text = "\n".join(path.read_text(encoding="utf-8") for path in _source_files())
    for name in ("PartialReviewRoundError", "PublicationResumeStop", "classification_resume_kind",
                 "admit_classification_resumes", "compute_partial_round_recovery",
                 "supersede_for_signed_growth", "post_frozen_round_bodies"):
        assert re.search(rf"\b{name}\b", text) is None, name


def test_other_delete_remedies_are_limited_to_the_justified_allowlist():
    found = {}
    for path in _source_files():
        matches = broad_matches(path.read_text(encoding="utf-8"))
        if matches:
            found[str(path.relative_to(SRC))] = matches
    unexpected = {name: matches for name, matches in found.items() if name not in ALLOWLIST}
    assert unexpected == {}, f"new operator delete remedy outside the allowlist: {unexpected}"
    counts = {name: len(matches) for name, matches in found.items()}
    assert counts == {name: count for name, (count, _why) in ALLOWLIST.items()}, (
        "an allowlisted file gained or lost a delete remedy; review it instead of "
        f"widening the allowlist: {found}"
    )
    for name, (_count, why) in ALLOWLIST.items():
        assert why.strip(), f"{name} needs a justification"


def test_broad_pattern_catches_a_new_remedy_and_ignores_docstrings():
    source = '''
def f():
    """Delete the record when the store is recreated."""
    raise RuntimeError("Delete the posted review comments and rerun the same command.")
'''
    assert broad_matches(source) == ["Delete the posted review comments and rerun the same command."]
    fstring = 'x = 1\nmsg = f"repair or delete comment {x} and resume."\n'
    assert broad_matches(fstring) == ["repair or delete comment {} and resume."]
    for pattern_text in ("Delete these 3 comments so the whole round runs again.",
                         "inspect the round's records before deleting anything."):
        assert any(p.search(pattern_text) for p in ROUND_RECOVERY_PATTERNS)
