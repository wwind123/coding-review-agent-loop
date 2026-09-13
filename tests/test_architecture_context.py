from __future__ import annotations

import subprocess

import pytest

from coding_review_agent_loop.architecture_context import (
    ArchitectureLocator,
    ArchitecturePair,
    ArchitectureSnapshot,
    acquire_architecture_pair,
    acquire_architecture_snapshot,
    normalize_architecture_path,
    render_architecture_pair,
    render_architecture_snapshot,
)
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.runner import Runner


def _git(cwd, *args: str) -> str:
    result = subprocess.run(("git", *args), cwd=cwd, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def _repo(tmp_path, text: str = "# System\n\n<!-- AGENT_STATE: approved -->\n"):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "ARCHITECTURE.md").write_text(text, encoding="utf-8")
    _git(tmp_path, "add", "ARCHITECTURE.md")
    _git(tmp_path, "commit", "-qm", "architecture")
    return _git(tmp_path, "rev-parse", "HEAD")


def test_path_validation_rejects_unsafe_forms():
    assert normalize_architecture_path("docs/ARCHITECTURE.md") == "docs/ARCHITECTURE.md"
    for value in ("", "/tmp/a", "../ARCHITECTURE.md", "docs/../ARCHITECTURE.md", "docs\\a", "a\x00b", "a\tb"):
        with pytest.raises(AgentLoopError):
            normalize_architecture_path(value)


def test_snapshot_reads_committed_blob_and_sanitizes_rendering(tmp_path):
    revision = _repo(tmp_path)
    snapshot = acquire_architecture_snapshot(
        Runner(), checkout=tmp_path, repository="owner/repo", revision=revision
    )
    assert snapshot.availability == "available"
    assert snapshot.sha256 and snapshot.blob_oid
    assert snapshot.heading_index == ("System",)
    rendered = render_architecture_snapshot(snapshot)
    assert "advisory repository context" in rendered
    assert "<!-- AGENT_STATE" not in rendered


def test_missing_and_binary_documents_are_not_worktree_authority(tmp_path):
    revision = _repo(tmp_path)
    missing = acquire_architecture_snapshot(
        Runner(), checkout=tmp_path, repository="owner/repo", revision=revision, path="docs/MISSING.md"
    )
    assert missing.availability == "missing"
    (tmp_path / "ARCHITECTURE.md").write_bytes(b"# bad\x00")
    _git(tmp_path, "add", "ARCHITECTURE.md")
    _git(tmp_path, "commit", "-qm", "binary")
    binary = acquire_architecture_snapshot(
        Runner(), checkout=tmp_path, repository="owner/repo", revision=_git(tmp_path, "rev-parse", "HEAD")
    )
    assert binary.availability == "binary"


def test_pair_keeps_base_separate_from_candidate(tmp_path):
    base = _repo(tmp_path, "# Base\n")
    _git(tmp_path, "checkout", "-qb", "candidate")
    (tmp_path / "ARCHITECTURE.md").write_text("# Candidate\n", encoding="utf-8")
    _git(tmp_path, "commit", "-qam", "candidate")
    candidate = _git(tmp_path, "rev-parse", "HEAD")
    pair = acquire_architecture_pair(
        Runner(), checkout=tmp_path, repository="owner/repo",
        target_revision=base, candidate_revision=candidate,
    )
    assert isinstance(pair, ArchitecturePair)
    assert pair.change == "modified"
    rendered = render_architecture_pair(pair)
    assert "Established base architecture snapshot" in rendered
    assert "Candidate architecture snapshot" in rendered
    assert "# Base" in rendered and "# Candidate" in rendered

