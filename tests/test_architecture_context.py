from __future__ import annotations

import subprocess

import pytest

from coding_review_agent_loop.architecture_context import (
    ArchitecturePair,
    ArchitectureSnapshot,
    acquire_architecture_pair,
    acquire_architecture_snapshot,
    normalize_architecture_path,
    render_architecture_pair,
    render_architecture_snapshot,
)
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.runner import Runner, run_binary_capture


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


def test_symlink_document_is_unavailable(tmp_path):
    revision = _repo(tmp_path)
    (tmp_path / "ARCHITECTURE.md").unlink()
    (tmp_path / "ARCHITECTURE.md").symlink_to("README.md")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "symlink")
    snapshot = acquire_architecture_snapshot(
        Runner(), checkout=tmp_path, repository="owner/repo", revision=_git(tmp_path, "rev-parse", "HEAD")
    )
    assert snapshot.availability == "unavailable"
    assert "regular blob" in (snapshot.diagnostic or "")


def test_pair_marks_both_missing_as_absent(tmp_path):
    _repo(tmp_path)
    (tmp_path / "ARCHITECTURE.md").unlink()
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "without architecture")
    base = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "checkout", "-qb", "candidate")
    candidate = _git(tmp_path, "rev-parse", "HEAD")
    pair = acquire_architecture_pair(
        Runner(), checkout=tmp_path, repository="owner/repo",
        target_revision=base, candidate_revision=candidate,
    )
    assert pair.change == "absent"


def test_pair_marks_added_and_deleted_documents_without_substitution(tmp_path):
    base = _repo(tmp_path, "# Base\n")
    _git(tmp_path, "checkout", "-qb", "candidate")
    (tmp_path / "ARCHITECTURE.md").unlink()
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "delete architecture")
    deleted_revision = _git(tmp_path, "rev-parse", "HEAD")
    deleted = acquire_architecture_pair(
        Runner(), checkout=tmp_path, repository="owner/repo",
        target_revision=base, candidate_revision=deleted_revision,
    )
    assert deleted.change == "deleted"
    assert "Established base architecture snapshot" in render_architecture_pair(deleted)
    assert "Candidate architecture snapshot" in render_architecture_pair(deleted)

    _git(tmp_path, "checkout", "-qb", "no-architecture", base)
    (tmp_path / "ARCHITECTURE.md").unlink()
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "remove architecture for add case")
    base_without_document = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "checkout", "-qb", "added", base_without_document)
    (tmp_path / "ARCHITECTURE.md").write_text("# Proposed\n", encoding="utf-8")
    _git(tmp_path, "add", "ARCHITECTURE.md")
    _git(tmp_path, "commit", "-qm", "add architecture")
    added_revision = _git(tmp_path, "rev-parse", "HEAD")
    added = acquire_architecture_pair(
        Runner(), checkout=tmp_path, repository="owner/repo",
        target_revision=base_without_document, candidate_revision=added_revision,
    )
    assert added.change == "added"
    rendered = render_architecture_pair(added)
    assert "proposal; no established base document exists" in rendered
    assert "# Proposed" in rendered


def test_oversized_architecture_blob_is_unavailable_to_prompt_material(tmp_path):
    revision = _repo(tmp_path, "# System\n" + ("detail " * 100))
    snapshot = acquire_architecture_snapshot(
        Runner(), checkout=tmp_path, repository="owner/repo", revision=revision, max_bytes=8
    )
    assert snapshot.availability == "oversized"
    assert snapshot.content is None


def test_ambiguous_tree_entries_are_rejected(tmp_path):
    class AmbiguousRunner(Runner):
        def run_binary(self, args, **kwargs):
            class Result:
                returncode = 0
                stdout = b"100644 blob " + b"a" * 40 + b"\tARCHITECTURE.md\x00" + b"100644 blob " + b"b" * 40 + b"\tARCHITECTURE.md\x00"
                stderr = b""
            return Result()

    snapshot = acquire_architecture_snapshot(
        AmbiguousRunner(), checkout=tmp_path, repository="owner/repo", revision="r" * 40
    )
    assert snapshot.availability == "missing"


def test_pair_does_not_use_target_tip_without_merge_base(tmp_path):
    base = _repo(tmp_path, "# Base\n")
    _git(tmp_path, "checkout", "--orphan", "unrelated")
    _git(tmp_path, "rm", "-rf", ".")
    (tmp_path / "ARCHITECTURE.md").write_text("# Unrelated\n", encoding="utf-8")
    _git(tmp_path, "add", "ARCHITECTURE.md")
    _git(tmp_path, "commit", "-qm", "unrelated")
    candidate = _git(tmp_path, "rev-parse", "HEAD")
    pair = acquire_architecture_pair(
        Runner(), checkout=tmp_path, repository="owner/repo",
        target_revision=base, candidate_revision=candidate,
    )
    assert pair.merge_base_revision is None
    assert pair.base.availability == "unavailable"
    assert pair.change == "unavailable"
    rendered = render_architecture_pair(pair)
    assert "# Unrelated" in rendered
    assert "candidate edits" not in rendered


def test_pair_resolves_remote_tracking_base_when_local_branch_is_absent(tmp_path):
    base = _repo(tmp_path, "# Base\n")
    _git(tmp_path, "update-ref", "refs/remotes/origin/release", base)
    _git(tmp_path, "checkout", "-qb", "candidate")
    (tmp_path / "ARCHITECTURE.md").write_text("# Candidate\n", encoding="utf-8")
    _git(tmp_path, "commit", "-qam", "candidate")
    candidate = _git(tmp_path, "rev-parse", "HEAD")
    pair = acquire_architecture_pair(
        Runner(), checkout=tmp_path, repository="owner/repo",
        target_revision="release", candidate_revision=candidate,
    )
    assert pair.target_revision == "release"
    assert pair.merge_base_revision == base
    assert pair.change == "modified"
    assert pair.base.content == "# Base\n"


def test_binary_capture_timeout_is_bounded(tmp_path):
    with pytest.raises(AgentLoopError, match="timed out"):
        run_binary_capture(
            ("python3", "-c", "import time; time.sleep(2)"),
            cwd=tmp_path,
            timeout_seconds=0.05,
        )


def test_binary_capture_timeout_is_nonfatal_when_check_is_false(tmp_path):
    result = run_binary_capture(
        ("python3", "-c", "import time; time.sleep(2)"),
        cwd=tmp_path,
        check=False,
        timeout_seconds=0.05,
    )
    assert result.returncode != 0
    assert b"timed out" in result.stderr


def test_pair_budget_retains_both_snapshot_identities(tmp_path):
    base = _repo(tmp_path, "# Base\n\n" + ("base detail " * 300))
    _git(tmp_path, "checkout", "-qb", "candidate")
    (tmp_path / "ARCHITECTURE.md").write_text("# Candidate\n\n" + ("candidate detail " * 300), encoding="utf-8")
    _git(tmp_path, "commit", "-qam", "candidate")
    candidate = _git(tmp_path, "rev-parse", "HEAD")
    pair = acquire_architecture_pair(
        Runner(), checkout=tmp_path, repository="owner/repo",
        target_revision=base, candidate_revision=candidate,
    )
    rendered = render_architecture_pair(pair, max_chars=2_000)
    assert len(rendered) <= 2_000
    assert "Established base architecture snapshot" in rendered
    assert "Candidate architecture snapshot" in rendered
    assert pair.base.blob_oid in rendered
    assert pair.candidate.blob_oid in rendered
    assert pair.base.sha256 in rendered
    assert pair.candidate.sha256 in rendered


def test_invalid_utf8_is_classified_as_binary(tmp_path):
    _repo(tmp_path)
    (tmp_path / "ARCHITECTURE.md").write_bytes(b"# invalid \xff\n")
    _git(tmp_path, "add", "ARCHITECTURE.md")
    _git(tmp_path, "commit", "-qm", "invalid utf8")
    snapshot = acquire_architecture_snapshot(
        Runner(), checkout=tmp_path, repository="owner/repo", revision=_git(tmp_path, "rev-parse", "HEAD")
    )
    assert snapshot.availability == "binary"
    assert "UTF-8" in (snapshot.diagnostic or "")


def test_gitlink_document_is_unavailable(tmp_path):
    revision = _repo(tmp_path, "# System\n")
    _git(tmp_path, "update-index", "--add", "--cacheinfo", f"160000,{revision},ARCHITECTURE.md")
    _git(tmp_path, "commit", "-qm", "architecture gitlink")
    snapshot = acquire_architecture_snapshot(
        Runner(), checkout=tmp_path, repository="owner/repo",
        revision=_git(tmp_path, "rev-parse", "HEAD"),
    )
    assert snapshot.availability == "unavailable"
    assert "regular blob" in (snapshot.diagnostic or "")


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


def test_pair_does_not_call_unreadable_documents_modified(tmp_path):
    pair = acquire_architecture_pair(
        Runner(), checkout=tmp_path, repository="owner/repo",
        target_revision="0" * 40, candidate_revision="1" * 40,
    )
    assert pair.change == "unavailable"
    assert "candidate edits" not in render_architecture_pair(pair)


def test_snapshot_renderer_respects_small_bound(tmp_path):
    revision = _repo(tmp_path, "# System\n\n" + ("detail " * 200))
    snapshot = acquire_architecture_snapshot(
        Runner(), checkout=tmp_path, repository="owner/repo", revision=revision
    )
    rendered = render_architecture_snapshot(snapshot, max_chars=180)
    assert len(rendered) <= 180
    assert "Repository:" in rendered
