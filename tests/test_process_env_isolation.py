"""Direct process-environment writes never leak between tests (#1990 CI)."""
import os
import subprocess

import pytest

from coding_review_agent_loop.agent_permissions import neutralized_git_env
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


def test_neutralized_git_env_written_directly_is_restored(dirty_repo):
    before = dict(os.environ)
    with restored_process_environment():
        # The exact write establish_sandboxed_run performs, outside monkeypatch.
        os.environ.update(neutralized_git_env())
        os.environ["AGENT_LOOP_LEAK_PROBE"] = "1"
        assert os.environ["GIT_EXTERNAL_DIFF"] == ""
        # This is what broke test_run_worktrees on the managed CI shard.
        leaked = _git(dirty_repo, "diff")
        assert leaked.returncode != 0 and "external diff" in leaked.stderr
    assert dict(os.environ) == before
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
