"""Parent-owned cleanup for generated sessions that cannot load our conftest."""
import os
import signal
import stat
import subprocess

from coding_review_agent_loop.agent_permissions import release_unlistable_tree


def run_child(argv, *, cwd, env, timeout):
    """Always stop and reap the child group, without repairing its filesystem."""
    proc = subprocess.Popen(
        argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)
    finally:
        # A child can exit while grandchildren survive; stop those too.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()


def release_owned_tree(root):
    """Release and verify only a harness-owned tree whose writers have stopped."""
    release_unlistable_tree(root)
    pending = [root]
    while pending:
        path = pending.pop()
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(info.st_mode):
            continue
        assert info.st_mode & 0o700 == 0o700, f"owner rwx missing: {path}"
        pending.extend(path / name for name in os.listdir(path))


def run_owned_child(argv, *, cwd, env, timeout, owned):
    try:
        return run_child(argv, cwd=cwd, env=env, timeout=timeout)
    finally:
        release_owned_tree(owned)
