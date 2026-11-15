"""Direct process-environment writes never leak between tests (#1990 CI)."""
import os
import subprocess

import pytest

from coding_review_agent_loop.agent_permissions import apply_neutralized_git_env
from process_env_isolation import restored_process_environment


def _git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], text=True, capture_output=True)


@pytest.fixture
def dirty_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    assert _git(repo, "init", "-q").returncode == 0
    (repo / "a.txt").write_text("staged\n")
    assert _git(repo, "add", "a.txt").returncode == 0
    (repo / "a.txt").write_text("unstaged\n")
    return repo


def test_neutralized_git_env_written_directly_is_restored(dirty_repo, tmp_path, monkeypatch):
    marker = tmp_path / "hook-ran"
    monkeypatch.setenv("GIT_EXTERNAL_DIFF", f"touch {marker}")
    monkeypatch.setenv("AGENT_LOOP_LEAK_PROBE", "inherited")
    before = dict(os.environ)
    with restored_process_environment():
        # The exact write establish_sandboxed_run performs, outside monkeypatch.
        apply_neutralized_git_env(os.environ)
        os.environ["AGENT_LOOP_LEAK_PROBE"] = "1"
        assert "GIT_EXTERNAL_DIFF" not in os.environ
        assert os.environ["GIT_TRACE"] == "0"
        inside = _git(dirty_repo, "diff")
        assert inside.returncode == 0, inside.stderr
        assert "+unstaged" in inside.stdout
        assert not marker.exists()
    # The direct write (a removal plus overrides) must not leak (#1990).
    assert dict(os.environ) == before
    assert os.environ["GIT_EXTERNAL_DIFF"] == f"touch {marker}"
    monkeypatch.delenv("GIT_EXTERNAL_DIFF")
    assert _git(dirty_repo, "diff").returncode == 0


@pytest.mark.parametrize("name", ["GIT_EXTERNAL_DIFF", "PATH"])
def test_removed_or_changed_variables_are_restored(name, monkeypatch):
    monkeypatch.setenv(name, os.environ.get(name, "original"))
    original = os.environ[name]
    with restored_process_environment():
        del os.environ[name]
    assert os.environ[name] == original
    with restored_process_environment():
        os.environ[name] = original + "-changed"
    assert os.environ[name] == original
