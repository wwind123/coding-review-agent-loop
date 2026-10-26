"""Observe real fixture teardown from a surviving parent, before fallback repair."""
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import textwrap

import pytest

from coding_review_agent_loop.agent_permissions import release_unlistable_tree
from private_directory_helpers import run_child, run_owned_child

ROOT = Path(__file__).resolve().parents[1]


def _environment():
    env = dict(os.environ)
    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "AGENT_LOOP_WORKER_CAP_SPEC",
                "AGENT_LOOP_WORKER_CAP_NESTED", "PYTEST_XDIST_WORKER"):
        env.pop(key, None)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "tests"), str(ROOT / "src")])
    return env


def _child(tmp_path, source, *, extra_conftest=""):
    project = tmp_path / "project"
    project.mkdir()
    (project / "conftest.py").write_text(
        (ROOT / "tests/conftest.py").read_text() + "\n" + textwrap.dedent(extra_conftest)
    )
    (project / "test_child.py").write_text(textwrap.dedent(source))
    base = tmp_path / "retained"
    proc = run_child(
        [sys.executable, "-m", "pytest", "-n", "0", "-q", "-p", "no:cacheprovider",
         "--basetemp", str(base)], cwd=project, env=_environment(),
        timeout=120,
    )
    return proc, base


@pytest.mark.parametrize("body_fails", [False, True])
def test_real_pr_loop_finalization(tmp_path, body_fails):
    proc, base = _child(tmp_path, f'''
        import json, stat
        import pytest
        import test_pr_loop_agent_turns as turns

        @pytest.mark.usefixtures("private_response_tempdir")
        def test_lifecycle(tmp_path, monkeypatch):
            original = turns.run_pr_loop
            observations = []
            def observed(*args, **kwargs):
                try:
                    return original(*args, **kwargs)
                finally:
                    private = tmp_path / "agent-cache" / "private"
                    assert stat.S_IMODE(private.stat().st_mode) == 0o300
                    response = tmp_path / "private-response-tmp/coding-review-agent-loop/responses"
                    assert stat.S_IMODE(response.stat().st_mode) == 0o300
                    observations.extend([str(private), str(response)])
            monkeypatch.setattr(turns, "run_pr_loop", observed)
            turns.test_pr_loop_does_not_run_claude_after_final_blocking_round(tmp_path)
            turns.test_pr_loop_resolves_pr_base_before_workdir_setup(tmp_path)
            assert len(observations) == 4
            (tmp_path / "observations.json").write_text(json.dumps(observations))
            assert not {body_fails!r}, "intentional body failure"
    ''')
    assert proc.returncode == int(body_fails), proc.stdout + proc.stderr
    assert "ERROR" not in proc.stdout, proc.stdout
    observations = list(base.glob("test_lifecycle*/observations.json"))
    assert observations
    for private in json.loads(observations[0].read_text()):
        assert Path(private).stat().st_mode & 0o700 == 0o700
    # No release here: the parent must observe actual fixture success.
    shutil.rmtree(base)
    assert not base.exists()


@pytest.mark.parametrize("body_fails", [False, True])
def test_fixture_symlinks_monkeypatch_and_late_finalizer(tmp_path, body_fails):
    external = tmp_path / "external"
    external.mkdir(mode=0o700)
    sentinel = external / "file"
    sentinel.write_text("unchanged")
    sentinel.chmod(0o400)
    external.chmod(0o300)
    before = (external.stat().st_mode, sentinel.stat().st_mode, sentinel.read_bytes())
    proc, base = _child(tmp_path, f'''
        import os
        from pathlib import Path
        import pytest

        @pytest.fixture
        def writer(tmp_path):
            private = tmp_path / "private" / "nested"
            private.mkdir(parents=True)
            yield private
            private.chmod(0o300)  # Resource shutdown can re-harden after the body.
            private.parent.chmod(0o300)

        def test_lifecycle(tmp_path, monkeypatch, writer):
            external = Path({str(external)!r})
            (writer / "dir-link").symlink_to(external, target_is_directory=True)
            (writer / "file-link").symlink_to(external / "file")
            (writer / "dangling").symlink_to(external / "missing")
            writer.chmod(0o300)
            assert writer.stat().st_mode & 0o777 == 0o300
            def broken(*args, **kwargs): raise OSError("patched filesystem")
            monkeypatch.setattr(os, "listdir", broken)
            monkeypatch.setattr(os, "lstat", broken)
            assert not {body_fails!r}, "intentional body failure"
    ''', extra_conftest="""
        _original_verify = _verify_owner_listing
        def _verify_owner_listing(root):
            assert _os.lstat is not _REAL_LSTAT, "lstat patch must still be active"
            assert _os.listdir is not _REAL_LISTDIR, "listdir patch must still be active"
            return _original_verify(root)
    """)
    assert proc.returncode == int(body_fails), proc.stdout + proc.stderr
    assert "ERROR" not in proc.stdout, proc.stdout
    assert before == (external.stat().st_mode, sentinel.stat().st_mode, sentinel.read_bytes())
    shutil.rmtree(base)


@pytest.mark.parametrize("body_fails", [False, True])
@pytest.mark.parametrize("operation", ["chmod", "lstat", "listdir"])
def test_cleanup_postcondition_reports_path(tmp_path, body_fails, operation):
    injection = '''
        _original_restore = _restore_owner_listing
        def _restore_owner_listing(root):
            errors = _original_restore(root)
            global _REAL_LSTAT, _REAL_LISTDIR
            original = _REAL_LSTAT if OP == "lstat" else _REAL_LISTDIR
            def broken(path):
                if str(path).endswith("survivor"):
                    raise PermissionError("injected verification failure")
                return original(path)
            if OP == "lstat": _REAL_LSTAT = broken
            if OP == "listdir": _REAL_LISTDIR = broken
            return errors
    ''' if operation != "chmod" else '''
        _original_chmod = _REAL_CHMOD
        def _REAL_CHMOD(path, mode):
            if str(path).endswith("survivor"):
                raise PermissionError("injected restoration failure")
            return _original_chmod(path, mode)
    '''
    proc, base = _child(tmp_path, f'''
        def test_lifecycle(tmp_path):
            private = tmp_path / "survivor"
            private.mkdir()
            private.chmod(0o300)
            assert not {body_fails!r}, "intentional body failure"
    ''', extra_conftest='OP = '+repr(operation)+'\n'+textwrap.dedent(injection))
    try:
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "tmp_path cleanup failed" in proc.stdout
        assert "survivor" in proc.stdout and operation in proc.stdout
        if body_fails:
            assert "intentional body failure" in proc.stdout
            assert "1 failed, 1 error" in proc.stdout
        else:
            assert "1 passed, 1 error" in proc.stdout
    finally:
        # The injected failure deliberately prevents the child fixture's repair.
        release_unlistable_tree(base)
        shutil.rmtree(base)


@pytest.mark.parametrize("exit_kind", ["success", "failure", "killed"])
def test_generated_response_session_cleanup(tmp_path, exit_kind):
    project = tmp_path / "generated"
    project.mkdir()
    owned = tmp_path / "owned"
    owned.mkdir()
    # Equivalent to historical generated test_probe0/tmp/.../responses layout;
    # historical receipts cannot identify its original producing harness.
    source = '''
        import os, signal, tempfile
        from pathlib import Path
        from coding_review_agent_loop.agent_permissions import harden_response_tree
        from types import SimpleNamespace

        def test_probe():
            root = Path(tempfile.gettempdir()) / "coding-review-agent-loop" / "responses"
            harden_response_tree(SimpleNamespace(repo="repo"))
            assert root.stat().st_mode & 0o777 == 0o300
            if os.environ["EXIT_KIND"] == "killed": os.kill(os.getpid(), signal.SIGKILL)
            assert os.environ["EXIT_KIND"] != "failure", "expected failure"
    '''
    # Use the real response-root creation API rather than synthetic chmod.
    (project / "test_probe.py").write_text(textwrap.dedent(source))
    env = _environment()
    env.update(TMPDIR=str(owned), EXIT_KIND=exit_kind)
    proc = run_owned_child(
        [sys.executable, "-m", "pytest", "-n", "0", "-q", "-p", "no:cacheprovider",
         "--basetemp", str(owned / "pytest")], cwd=project, env=env, timeout=120, owned=owned,
    )
    assert proc.returncode == {"success": 0, "failure": 1, "killed": -signal.SIGKILL}[exit_kind], proc.stdout + proc.stderr
    assert (owned / "coding-review-agent-loop/responses").stat().st_mode & 0o700 == 0o700
    shutil.rmtree(owned)
