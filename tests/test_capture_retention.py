"""Capture-root retention and pruning (#1572)."""

from __future__ import annotations

import os
import shutil

import pytest

fcntl = pytest.importorskip("fcntl")

import coding_review_agent_loop.logging as logging_module  # noqa: E402
from agent_loop_helpers import make_config  # noqa: E402
from coding_review_agent_loop.config import DEFAULT_SUBPROCESS_LOG_RETENTION_DAYS  # noqa: E402

NOW = 2_000_000_000.0
DAY = 24 * 60 * 60
MARKER = ".agent-loop-capture"


@pytest.fixture(autouse=True)
def _fixed_clock(monkeypatch):
    monkeypatch.setattr(logging_module.time, "time", lambda: NOW)
    leases = {}
    monkeypatch.setattr(logging_module, "_LEASES", leases)
    yield
    for lease in leases.values():
        lease.close()


def _parent(tmp_path):
    parent = tmp_path / "subprocess-logs" / "owner-repo"
    parent.mkdir(parents=True)
    return parent


def _root(parent, name, *, age_days, nested=False):
    root = parent / name
    root.mkdir()
    (root / MARKER).touch()
    (root / ".lease").touch()
    (root / "agent.log").write_text("transcript", encoding="utf-8")
    if nested:
        (root / "sub" / "deeper").mkdir(parents=True)
        (root / "sub" / "deeper" / "file.txt").write_text("x", encoding="utf-8")
    _age(root, age_days)
    return root


def _age(path, age_days):
    stamp = NOW - age_days * DAY
    os.utime(path, (stamp, stamp), follow_symlinks=False)


def test_default_retention_prunes_only_roots_older_than_14_days(tmp_path):
    parent = _parent(tmp_path)
    old = _root(parent, "old", age_days=15)
    young = _root(parent, "young", age_days=13)
    current = parent / "current"
    config = make_config(tmp_path, subprocess_log_dir=current)
    assert config.subprocess_log_retention_days == DEFAULT_SUBPROCESS_LOG_RETENTION_DAYS == 14

    logging_module.agent_log_path(config, "claude")

    assert not old.exists()
    assert (young / "agent.log").is_file()
    assert (current / MARKER).is_file()


def test_custom_retention_uses_the_configured_cutoff(tmp_path):
    parent = _parent(tmp_path)
    old = _root(parent, "old", age_days=4)
    young = _root(parent, "young", age_days=2)

    logging_module._prepare_capture_root(parent / "current", retention_days=3)

    assert not old.exists()
    assert (young / "agent.log").is_file()


def test_zero_retention_never_invokes_the_pruner(tmp_path, monkeypatch):
    parent = _parent(tmp_path)
    ancient = _root(parent, "ancient", age_days=400, nested=True)
    calls = []
    monkeypatch.setattr(logging_module, "_prune_capture_roots", lambda *a, **k: calls.append((a, k)))

    logging_module._prepare_capture_root(parent / "current", retention_days=0)

    assert calls == []
    assert (ancient / "agent.log").is_file()
    assert (ancient / "sub" / "deeper" / "file.txt").is_file()


def test_pruner_itself_is_a_noop_for_zero_retention(tmp_path):
    parent = _parent(tmp_path)
    ancient = _root(parent, "ancient", age_days=400)

    logging_module._prune_capture_roots(parent, parent / "current", retention_days=0)

    assert (ancient / "agent.log").is_file()


def test_agent_log_path_passes_the_configured_retention(tmp_path, monkeypatch):
    parent = _parent(tmp_path)
    seen = []
    monkeypatch.setattr(
        logging_module, "_prune_capture_roots", lambda p, c, *, retention_days: seen.append(retention_days)
    )
    config = make_config(
        tmp_path, subprocess_log_dir=parent / "current", subprocess_log_retention_days=9
    )

    logging_module.agent_log_path(config, "claude")

    assert seen == [9]


def test_live_leased_root_is_never_pruned(tmp_path):
    parent = _parent(tmp_path)
    leased = _root(parent, "leased", age_days=30, nested=True)
    holder = (leased / ".lease").open("a+")
    try:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        logging_module._prune_capture_roots(parent, parent / "current", retention_days=14)
    finally:
        holder.close()

    assert (leased / MARKER).is_file()
    assert (leased / ".lease").is_file()
    assert (leased / "agent.log").is_file()
    assert (leased / "sub" / "deeper" / "file.txt").is_file()


def test_expired_root_with_subdirectories_is_removed_without_following_symlinks(tmp_path):
    parent = _parent(tmp_path)
    outside = tmp_path / "outside"
    (outside / "inner").mkdir(parents=True)
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    (outside / "inner" / "keep.txt").write_text("keep", encoding="utf-8")
    root = _root(parent, "old", age_days=20, nested=True)
    (root / "link-dir").symlink_to(outside, target_is_directory=True)
    (root / "sub" / "nested-link").symlink_to(outside, target_is_directory=True)
    _age(root, 20)

    logging_module._prune_capture_roots(parent, parent / "current", retention_days=14)

    assert not os.path.lexists(root)
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert (outside / "inner" / "keep.txt").is_file()


def test_unmarked_directory_is_never_touched(tmp_path):
    parent = _parent(tmp_path)
    unmarked = parent / "legacy"
    (unmarked / "sub").mkdir(parents=True)
    (unmarked / "sub" / "file.txt").write_text("x", encoding="utf-8")
    _age(unmarked, 60)

    logging_module._prune_capture_roots(parent, parent / "current", retention_days=14)

    assert (unmarked / "sub" / "file.txt").is_file()


def test_roots_outside_the_subprocess_logs_layout_are_not_pruned(tmp_path):
    parent = tmp_path / "custom-captures" / "owner-repo"
    parent.mkdir(parents=True)
    old = _root(parent, "old", age_days=100)

    logging_module._prepare_capture_root(parent / "current", retention_days=1)

    assert (old / "agent.log").is_file()


def test_symlinked_candidate_root_is_never_followed(tmp_path, monkeypatch):
    parent = _parent(tmp_path)
    external_parent = tmp_path / "external"
    external_parent.mkdir()
    external = _root(external_parent, "target", age_days=100, nested=True)
    link = parent / "linked-root"
    link.symlink_to(external, target_is_directory=True)
    _age(link, 100)
    opened = []
    real_open = type(external).open

    def tracking_open(self, *args, **kwargs):
        opened.append(self)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(type(external), "open", tracking_open)

    logging_module._prune_capture_roots(parent, parent / "current", retention_days=14)

    assert link.is_symlink()
    assert opened == []
    for path in (MARKER, ".lease", "agent.log", "sub/deeper/file.txt"):
        assert (external / path).is_file()


def test_symlinked_marker_or_lease_makes_a_root_ineligible(tmp_path):
    parent = _parent(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "file").touch()
    marker_link = parent / "marker-link"
    marker_link.mkdir()
    (marker_link / MARKER).symlink_to(elsewhere / "file")
    (marker_link / "agent.log").touch()
    _age(marker_link, 30)
    lease_link = _root(parent, "lease-link", age_days=30)
    (lease_link / ".lease").unlink()
    (lease_link / ".lease").symlink_to(elsewhere / "file")
    _age(lease_link, 30)

    logging_module._prune_capture_roots(parent, parent / "current", retention_days=14)

    assert (marker_link / "agent.log").exists()
    assert (lease_link / "agent.log").exists()
    assert (elsewhere / "file").is_file()


def test_partial_failure_keeps_the_root_retryable(tmp_path, monkeypatch):
    parent = _parent(tmp_path)
    root = _root(parent, "old", age_days=20, nested=True)
    real_rmtree = shutil.rmtree

    def failing_rmtree(path, *args, **kwargs):
        raise OSError("injected")

    monkeypatch.setattr(logging_module.shutil, "rmtree", failing_rmtree)
    logging_module._prune_capture_roots(parent, parent / "current", retention_days=14)

    assert (root / MARKER).is_file()
    assert (root / ".lease").is_file()
    assert (root / "sub" / "deeper" / "file.txt").is_file()
    assert not (root / "agent.log").exists()
    assert root.stat().st_mtime < NOW - 14 * DAY
    probe = (root / ".lease").open("a+")
    try:
        fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        probe.close()

    monkeypatch.setattr(logging_module.shutil, "rmtree", real_rmtree)
    logging_module._prune_capture_roots(parent, parent / "current", retention_days=14)

    assert not os.path.lexists(root)


def test_explicit_subprocess_log_dir_in_the_layout_is_pruned(tmp_path):
    # #1576: the gate is the path layout (grandparent named subprocess-logs),
    # not whether the capture root is the default cache location.
    parent = tmp_path / "elsewhere" / "subprocess-logs" / "custom"
    parent.mkdir(parents=True)
    old = _root(parent, "old", age_days=100)
    current = parent / "current"
    config = make_config(tmp_path, subprocess_log_dir=current, subprocess_log_retention_days=14)

    logging_module.agent_log_path(config, "claude")

    assert not old.exists()
    assert (current / MARKER).is_file()
