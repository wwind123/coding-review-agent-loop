"""Source inventory shared by GitHub call guards across managed-CI moves."""

from pathlib import Path


def managed_ci_source_paths() -> list[Path]:
    # Include future sibling extractions without changing the frozen historical
    # orchestrator registry or leaving call sites behind in the driver.
    package = Path(__file__).resolve().parents[1] / "src/coding_review_agent_loop"
    return sorted(package.glob("managed_ci*.py"))
