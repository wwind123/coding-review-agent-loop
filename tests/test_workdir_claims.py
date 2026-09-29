"""Agent checkout claims (#1127)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from unittest.mock import MagicMock

import pytest

from coding_review_agent_loop import orchestrator, workdir_claims
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.config import ensure_agent_workdirs
from coding_review_agent_loop.workdir_claims import (
    WorkdirClaimedError,
    acquire_workdir_claim,
    claim_agent_workdirs,
    claim_root,
    claimed_run,
    workdir_claim_scope,
)
from agent_loop_helpers import make_config

HOLDER = r"""
import sys
from pathlib import Path
from coding_review_agent_loop import workdir_claims as w
w._set_claim_root_for_tests(Path(sys.argv[1]))
command = sys.argv[3] if sys.argv[3] != "-" else None
number = int(sys.argv[4]) if sys.argv[4] != "-" else None
head = sys.argv[5] if len(sys.argv) > 5 else None
with w.workdir_claim_scope(command=command, number=number, head=head):
    for p in sys.argv[2].split(","):
        w.acquire_workdir_claim(Path(p), agent="claude", repo="OWNER/REPO")
    print("held", flush=True)
    sys.stdin.read()
"""


def start_holder(paths, command="-", number="-", head=None, env=None):
    args = [sys.executable, "-c", HOLDER, str(claim_root()), ",".join(str(p) for p in paths), command, number]
    if head:
        args.append(head)
    proc = subprocess.Popen(
        args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env={**os.environ, **(env or {})}
    )
    assert proc.stdout.readline().strip() == "held"
    return proc


def stop(proc):
    proc.stdin.close()
    proc.wait(timeout=10)


def claim_once(path, agent="claude"):
    with workdir_claim_scope("issue", 5):
        return acquire_workdir_claim(path, agent=agent, repo="OWNER/REPO")


def test_claim_requires_scope(tmp_path):
    with pytest.raises(AgentLoopError):
        acquire_workdir_claim(tmp_path, agent="claude", repo="OWNER/REPO")


@pytest.mark.parametrize(
    "command,number,head,target",
    [
        ("issue", "7", None, "issue #7"),
        ("pr", "8", None, "PR #8"),
        ("discuss", "9", None, "discuss #9"),
        ("task", "-", None, "task"),
        ("managed-pr", "-", "feature-x", "managed-pr from feature-x"),
        ("-", "-", None, "library"),
    ],
)
def test_holder_identity_and_refusal(tmp_path, command, number, head, target):
    proc = start_holder([tmp_path], command, number, head)
    try:
        meta = json.loads(next(claim_root().glob("*.json")).read_text())
        assert meta["target"] == target
        assert meta["command"] == ("library" if command == "-" else command)
        assert meta["number"] == (None if number == "-" else int(number))
        assert meta["pid"] == proc.pid
        with pytest.raises(WorkdirClaimedError) as exc:
            claim_once(tmp_path)
        message = str(exc.value)
        assert f"target {target}" in message
        assert f"pid {proc.pid}" in message
        assert meta["run_id"] in message
        assert "OWNER/REPO" in message
        assert str(next(claim_root().glob("*.lock"))) in message
        if number == "-":
            assert "#" not in target
    finally:
        stop(proc)


def test_second_run_default_slot_no_git_and_holder_untouched(tmp_path):
    checkout = tmp_path / "claude"
    checkout.mkdir()
    (checkout / "dirty.txt").write_text("wip")
    config = make_config(tmp_path, auto_agent_dirs=("claude",))
    runner = MagicMock()
    proc = start_holder([checkout], "issue", "1")
    try:
        before = json.loads(next(claim_root().glob("*.json")).read_text())
        with pytest.raises(WorkdirClaimedError, match="issue #1"):
            ensure_agent_workdirs(config, runner)
        assert runner.method_calls == []
        assert (checkout / "dirty.txt").read_text() == "wip"
        assert json.loads(next(claim_root().glob("*.json")).read_text()) == before
    finally:
        stop(proc)


def test_locks_land_under_claim_root_not_checkout_and_tmpdir_independent(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    other_tmp = tmp_path / "othertmp"
    other_tmp.mkdir()
    proc = start_holder([checkout], "issue", "2", env={"TMPDIR": str(other_tmp)})
    try:
        assert list(checkout.iterdir()) == []
        assert list(other_tmp.iterdir()) == []
        assert list(claim_root().glob("*.lock"))
        with pytest.raises(WorkdirClaimedError):
            claim_once(checkout)
    finally:
        stop(proc)


def test_stale_holder_is_taken_over(tmp_path):
    proc = start_holder([tmp_path], "issue", "3")
    old = json.loads(next(claim_root().glob("*.json")).read_text())
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait(timeout=10)
    assert (claim_root() / next(claim_root().glob("*.json")).name).exists()
    with workdir_claim_scope("pr", 4):
        assert acquire_workdir_claim(tmp_path, agent="claude", repo="OWNER/REPO")
        new = json.loads(next(claim_root().glob("*.json")).read_text())
    assert new["pid"] == os.getpid()
    assert new["run_id"] != old["run_id"]


def test_reentry_same_owner_is_noop_and_nested_scope_joins(tmp_path):
    with workdir_claim_scope("issue", 1) as outer:
        assert acquire_workdir_claim(tmp_path, agent="claude", repo="R")
        with workdir_claim_scope("pr", 2) as inner:
            assert inner is outer
            assert acquire_workdir_claim(tmp_path, agent="claude", repo="R") is False
        # nested exit releases nothing
        assert len(workdir_claims._registry) == 1
        assert json.loads(next(claim_root().glob("*.json")).read_text())["command"] == "issue"
    assert workdir_claims._registry == {}


def test_shared_dir_single_claim_and_other_process_refused(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    config = make_config(
        tmp_path, claude_dir=shared, codex_dir=shared, allow_shared_dir=True, create_dirs=False
    )
    with workdir_claim_scope("issue", 1):
        claim_agent_workdirs(config)
        assert len(workdir_claims._registry) == 1
        # a second holder in another process is refused
        code = (
            "import sys;from pathlib import Path;"
            "from coding_review_agent_loop import workdir_claims as w;"
            "w._set_claim_root_for_tests(Path(sys.argv[1]));"
            "s=w.workdir_claim_scope('issue',9);s.__enter__();"
            "\ntry:\n w.acquire_workdir_claim(Path(sys.argv[2]),agent='codex',repo='R')\n"
            "except w.WorkdirClaimedError: print('refused')"
        )
        out = subprocess.run(
            [sys.executable, "-c", code, str(claim_root()), str(shared)],
            capture_output=True, text=True, timeout=30,
        )
        assert out.stdout.strip() == "refused"


def test_single_run_reuses_existing_checkout_and_releases(tmp_path):
    config = make_config(tmp_path)
    for _ in range(2):
        with workdir_claim_scope("issue", 1):
            claim_agent_workdirs(config)
    assert workdir_claims._registry == {}
    assert not list((tmp_path / "claude").iterdir())


def test_later_path_held_releases_earlier_claim_and_no_cleanup(tmp_path):
    config = make_config(tmp_path, auto_agent_dirs=("claude",))
    (tmp_path / "claude" / "dirty.txt").write_text("wip")
    proc = start_holder([tmp_path / "codex"], "task")
    runner = MagicMock()
    try:
        with pytest.raises(WorkdirClaimedError, match="target task"):
            ensure_agent_workdirs(config, runner)
        assert runner.method_calls == []
        assert (tmp_path / "claude" / "dirty.txt").exists()
        assert workdir_claims._registry == {}
        # the claude lock was taken (metadata was written) and freed again
        with workdir_claim_scope("issue", 1):
            assert acquire_workdir_claim(tmp_path / "claude", agent="claude", repo="R")
    finally:
        stop(proc)


def test_acquisition_order_is_by_agent_name(tmp_path):
    config = make_config(tmp_path)
    order = []
    real = workdir_claims.acquire_workdir_claim

    def spy(path, *, agent, repo):
        order.append(agent)
        return real(path, agent=agent, repo=repo)

    orig = workdir_claims.acquire_workdir_claim
    workdir_claims.acquire_workdir_claim = spy
    try:
        with workdir_claim_scope("issue", 1):
            claim_agent_workdirs(config)
    finally:
        workdir_claims.acquire_workdir_claim = orig
    assert order == sorted(order) == ["claude", "codex"]


def test_unknown_holder_metadata_is_reported_unidentified(tmp_path, monkeypatch):
    monkeypatch.setattr(workdir_claims, "_METADATA_READ_DELAY_SECONDS", 0)
    proc = start_holder([tmp_path], "issue", "1")
    try:
        for meta in claim_root().glob("*.json"):
            meta.unlink()
        with pytest.raises(WorkdirClaimedError) as exc:
            claim_once(tmp_path)
        assert "unidentified holder" in str(exc.value)
        assert ".lock" in str(exc.value)
        assert "pid" not in str(exc.value).split("unidentified")[1].split("lock file")[0]
        assert not list(claim_root().glob("*.json"))
    finally:
        stop(proc)


def test_scope_release_on_return_and_raise(tmp_path):
    with workdir_claim_scope("issue", 1):
        acquire_workdir_claim(tmp_path, agent="claude", repo="R")
    assert not list(claim_root().glob("*.json"))
    with pytest.raises(RuntimeError):
        with workdir_claim_scope("issue", 1):
            acquire_workdir_claim(tmp_path, agent="claude", repo="R")
            raise RuntimeError("boom")
    assert claim_once(tmp_path)


def test_release_keeps_other_holders_record(tmp_path):
    with workdir_claim_scope("issue", 1):
        acquire_workdir_claim(tmp_path, agent="claude", repo="R")
        key = workdir_claims._claim_key(tmp_path)
        meta_path = workdir_claims._registry[key].meta_path
        meta_path.write_text(json.dumps({"pid": 1, "run_id": "someone-else"}))
    assert json.loads(meta_path.read_text())["run_id"] == "someone-else"


def test_unused_default_dirs_not_claimed_and_concurrent_runs_proceed(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    cfg_a = make_config(tmp_path, claude_dir=a / "claude", codex_dir=a / "codex", create_dirs=False)
    cfg_b = make_config(tmp_path, claude_dir=b / "claude", codex_dir=b / "codex", create_dirs=False)
    holder = start_holder([a / "claude", a / "codex"], "issue", "1")
    try:
        with workdir_claim_scope("issue", 2):
            claim_agent_workdirs(cfg_b)
            assert len(workdir_claims._registry) == 2
            assert len(list(claim_root().glob("*.lock"))) == 4
        with pytest.raises(WorkdirClaimedError):
            claim_once(a / "claude")
        assert cfg_a.gemini_dir == cfg_b.gemini_dir
    finally:
        stop(holder)


def test_same_process_owners_refused_and_not_released(tmp_path):
    ready, done = threading.Event(), threading.Event()
    errors = []

    def holder():
        with workdir_claim_scope("issue", 1):
            acquire_workdir_claim(tmp_path, agent="claude", repo="R")
            ready.set()
            done.wait(10)

    thread = threading.Thread(target=holder)
    thread.start()
    assert ready.wait(10)
    try:
        run_id = json.loads(next(claim_root().glob("*.json")).read_text())["run_id"]
        with pytest.raises(WorkdirClaimedError) as exc:
            with workdir_claim_scope("issue", 2):
                acquire_workdir_claim(tmp_path, agent="claude", repo="R")
        errors.append(str(exc.value))
        assert run_id in errors[0]
        assert json.loads(next(claim_root().glob("*.json")).read_text())["run_id"] == run_id
        assert len(workdir_claims._registry) == 1
    finally:
        done.set()
        thread.join()


@pytest.mark.parametrize("fn", ["run_issue_loop", "run_task_loop", "run_pr_loop", "run_discuss_loop"])
def test_run_loops_refuse_before_any_runner_call(tmp_path, fn):
    config = make_config(tmp_path)
    proc = start_holder([tmp_path / "claude"], "issue", "1")
    runner = MagicMock()
    kwargs = {
        "run_issue_loop": {"issue_number": 3},
        "run_task_loop": {"task_text": "x"},
        "run_pr_loop": {"pr_number": 3, "workdirs_ready": True},
        "run_discuss_loop": {"issue_number": 3},
    }[fn]
    try:
        with pytest.raises(WorkdirClaimedError):
            getattr(orchestrator, fn)(runner, config=config, **kwargs)
        assert runner.method_calls == []
        assert workdir_claims._registry == {}
    finally:
        stop(proc)


@pytest.mark.parametrize(
    "command,param,expected",
    [("issue", "n", ("issue", 4, "issue #4")), ("pr", "n", ("pr", 4, "PR #4")),
     ("discuss", "n", ("discuss", 4, "discuss #4")), ("task", None, ("task", None, "task"))],
)
def test_claimed_run_binds_identity_and_nested_joins(tmp_path, command, param, expected):
    config = make_config(tmp_path)
    seen = {}

    @claimed_run("pr", "n")
    def inner(runner, *, n, config):
        seen["inner"] = workdir_claims.current_claim_owner()

    @claimed_run(command, param)
    def outer(runner, *, n=None, config):
        seen["outer"] = workdir_claims.current_claim_owner()
        inner(runner, n=99, config=config)
        seen["claimed"] = len(workdir_claims._registry)

    outer(None, n=4, config=config)
    owner = seen["outer"]
    assert (owner.command, owner.number, owner.target) == expected
    assert seen["inner"] is owner
    assert seen["claimed"] == 2
    assert workdir_claims._registry == {}


def test_all_run_loops_are_claimed():
    for fn in (orchestrator.run_issue_loop, orchestrator.run_task_loop,
               orchestrator.run_pr_loop, orchestrator.run_discuss_loop):
        assert hasattr(fn, "__wrapped__")


def test_cli_main_claims_and_releases(tmp_path, monkeypatch):
    from coding_review_agent_loop import cli

    seen = {}

    def fake_loop(runner, **kwargs):
        seen["owner"] = workdir_claims.current_claim_owner()
        seen["claimed"] = len(workdir_claims._registry)
        return 0

    config = make_config(tmp_path)
    monkeypatch.setattr(cli, "config_from_args", lambda *a, **k: config)
    monkeypatch.setattr(cli, "establish_sandboxed_run", lambda *a, **k: None)
    monkeypatch.setattr(cli, "run_pr_loop", fake_loop)
    assert cli.main(["pr", "12", "--repo", "OWNER/REPO", "--dry-run"]) == 0
    assert (seen["owner"].command, seen["owner"].number, seen["owner"].target) == ("pr", 12, "PR #12")
    assert seen["claimed"] == 2
    assert workdir_claims._registry == {}

    proc = start_holder([tmp_path / "codex"], "task")
    try:
        assert cli.main(["pr", "12", "--repo", "OWNER/REPO", "--dry-run"]) == 1
        assert "owner" in seen and workdir_claims._registry == {}
    finally:
        stop(proc)
