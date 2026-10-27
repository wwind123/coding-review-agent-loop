"""Source file size guard with a ratcheting allowlist (#1623).

Every ``.py`` file under ``src/coding_review_agent_loop/`` must stay at or
below ``DEFAULT_LIMIT`` lines. Files that were already over the limit when the
guard landed are listed in ``ALLOWLIST`` with their measured size as their own
ceiling: they may shrink but not grow. A pure-move split lowers its file's
ceiling (or deletes the entry) in the same PR.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = ROOT / "src" / "coding_review_agent_loop"

DEFAULT_LIMIT = 6000

# Repo-relative POSIX path -> line ceiling. Each ceiling is the file's size
# when it was listed; lower it (or delete the entry) when the file is split.
ALLOWLIST: dict[str, int] = {
    "src/coding_review_agent_loop/managed_ci.py": 9842,
    "src/coding_review_agent_loop/protocol.py": 10252,
    "src/coding_review_agent_loop/pr_loop.py": 9990,
    "src/coding_review_agent_loop/round_state.py": 6231,
}

SPLIT_GUIDANCE = (
    "See https://github.com/wwind123/coding-review-agent-loop/issues/1623 for "
    "how to split it: move cohesive groups to new sibling modules without "
    "behavior changes, re-export the moved names from the original module, "
    "and list the moved symbols in the PR description."
)


@dataclass(frozen=True)
class Violation:
    path: str
    lines: int
    limit: int
    allowlisted: bool

    def message(self) -> str:
        source = "allowlist ceiling" if self.allowlisted else "default limit"
        return (
            f"{self.path} has {self.lines} lines, over its {source} of "
            f"{self.limit}. {SPLIT_GUIDANCE}"
        )


def count_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _relative(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def scan(
    repo_root: Path,
    package_dir: Path,
    default_limit: int,
    allowlist: dict[str, int],
) -> list[Violation]:
    """Return every ``.py`` file under ``package_dir`` that exceeds its limit."""
    violations = []
    for path in sorted(package_dir.rglob("*.py")):
        rel = _relative(path, repo_root)
        allowlisted = rel in allowlist
        limit = allowlist[rel] if allowlisted else default_limit
        lines = count_lines(path)
        if lines > limit:
            violations.append(Violation(rel, lines, limit, allowlisted))
    return violations


def allowlist_problems(
    repo_root: Path,
    default_limit: int,
    allowlist: dict[str, int],
) -> list[str]:
    """Return stale, unnecessary or malformed allowlist entries."""
    problems = []
    for rel, ceiling in sorted(allowlist.items()):
        if ceiling <= default_limit:
            problems.append(
                f"{rel}: allowlist ceiling {ceiling} is not above the default "
                f"limit {default_limit}; delete the entry."
            )
        path = repo_root / rel
        if not path.is_file():
            problems.append(f"{rel}: allowlisted file no longer exists; delete the entry.")
            continue
        lines = count_lines(path)
        if lines <= default_limit:
            problems.append(
                f"{rel}: now {lines} lines, at or below the default limit "
                f"{default_limit}; delete the allowlist entry."
            )
    return problems


def _report(violations: list[Violation]) -> str:
    return "\n".join(violation.message() for violation in violations)


def test_source_files_within_size_limits():
    violations = scan(ROOT, PACKAGE_DIR, DEFAULT_LIMIT, ALLOWLIST)
    assert not violations, _report(violations)


def test_allowlist_entries_are_current():
    problems = allowlist_problems(ROOT, DEFAULT_LIMIT, ALLOWLIST)
    assert not problems, "\n".join(problems)


# --- helper unit tests over a synthetic tree ---------------------------------


def _write(path: Path, lines: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"x = {i}\n" for i in range(lines)), encoding="utf-8")


@pytest.fixture
def tree(tmp_path):
    package = tmp_path / "src" / "pkg"
    package.mkdir(parents=True)
    return tmp_path, package


def test_scan_passes_file_at_or_under_default_limit(tree):
    root, package = tree
    _write(package / "small.py", 9)
    _write(package / "edge.py", 10)
    assert scan(root, package, 10, {}) == []


def test_scan_reports_unlisted_file_over_default_limit(tree):
    root, package = tree
    _write(package / "big.py", 11)
    violations = scan(root, package, 10, {})
    assert violations == [Violation("src/pkg/big.py", 11, 10, False)]
    message = violations[0].message()
    assert "src/pkg/big.py has 11 lines" in message
    assert "default limit of 10" in message
    assert "issues/1623" in message


def test_scan_honors_allowlist_ceiling_and_fails_one_line_over(tree):
    root, package = tree
    allowlist = {"src/pkg/big.py": 15}
    _write(package / "big.py", 15)
    assert scan(root, package, 10, allowlist) == []
    _write(package / "big.py", 16)
    violations = scan(root, package, 10, allowlist)
    assert violations == [Violation("src/pkg/big.py", 16, 15, True)]
    message = violations[0].message()
    assert "src/pkg/big.py has 16 lines" in message
    assert "allowlist ceiling of 15" in message
    assert "issues/1623" in message


def test_scan_allows_allowlisted_file_to_shrink(tree):
    root, package = tree
    _write(package / "big.py", 12)
    assert scan(root, package, 10, {"src/pkg/big.py": 15}) == []


def test_scan_covers_nested_subpackages_and_reports_all(tree):
    root, package = tree
    _write(package / "agents" / "deep.py", 11)
    _write(package / "plugin" / "inner" / "deeper.py", 12)
    _write(package / "fine.py", 3)
    violations = scan(root, package, 10, {})
    assert [v.path for v in violations] == [
        "src/pkg/agents/deep.py",
        "src/pkg/plugin/inner/deeper.py",
    ]


def test_allowlist_rejects_missing_file(tree):
    root, _package = tree
    problems = allowlist_problems(root, 10, {"src/pkg/gone.py": 20})
    assert problems == ["src/pkg/gone.py: allowlisted file no longer exists; delete the entry."]


def test_allowlist_rejects_file_back_under_default_limit(tree):
    root, package = tree
    _write(package / "split.py", 10)
    problems = allowlist_problems(root, 10, {"src/pkg/split.py": 20})
    assert problems == [
        "src/pkg/split.py: now 10 lines, at or below the default limit 10; "
        "delete the allowlist entry."
    ]


def test_allowlist_rejects_ceiling_not_above_default_limit(tree):
    root, package = tree
    _write(package / "big.py", 11)
    problems = allowlist_problems(root, 10, {"src/pkg/big.py": 10})
    assert problems == [
        "src/pkg/big.py: allowlist ceiling 10 is not above the default limit 10; "
        "delete the entry."
    ]


def test_allowlist_accepts_current_oversize_entry(tree):
    root, package = tree
    _write(package / "big.py", 12)
    assert allowlist_problems(root, 10, {"src/pkg/big.py": 15}) == []


def test_line_count_matches_newline_terminated_wc(tmp_path):
    path = tmp_path / "f.py"
    path.write_text("a\nb\n\nc\n", encoding="utf-8")
    assert count_lines(path) == 4
