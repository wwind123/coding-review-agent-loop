"""Real-store integration tests for bounded, persistent source health (#1775)."""
import dataclasses
import json
import os
import shutil
import stat
from pathlib import Path

import pytest

from coding_review_agent_loop import checkout_verification as cv
from coding_review_agent_loop import run_worktrees as rw
from coding_review_agent_loop import worktree_link_integrity as health
from coding_review_agent_loop import worktree_setup as setup
from coding_review_agent_loop.config import ensure_agent_workdirs
from coding_review_agent_loop.errors import AgentLoopError, CheckoutVerificationError
from coding_review_agent_loop.runner import Runner
from coding_review_agent_loop.workdir_claims import workdir_claim_scope
from test_run_worktrees import Env
from test_worktree_setup import seeded, setup_command, setup_config, claude_store, runs_in, prepare_full


def source_store(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    store = env.seed_store("claude")
    config = env.config(extra=("--worktree-link", ".venv"))
    return env, store, config


def save(store, config, links=(".venv",)):
    with rw.store_lock(store):
        return health.publish(store, links, config=config)


def test_selected_nested_deletion_survives_restart(tmp_path, monkeypatch):
    env, store, config = source_store(tmp_path, monkeypatch)
    package = store / ".venv" / "package"
    package.mkdir()
    (package / "one").touch()
    (package / "two").touch()
    record = save(store, config)
    (package / "one").unlink()
    cv.reset_checkout_baselines()
    # Read persisted health with fresh registry state; never refresh over damage.
    assert health.inspect(store, (".venv",))[1]
    assert health.health_path(store).read_text() == json.dumps(record, sort_keys=True)
    with pytest.raises(AgentLoopError, match="damaged or stale"):
        rw.prepare_store(store, config=config, runner=Runner())


def test_large_pair_has_bounded_operations_and_payload(tmp_path, monkeypatch):
    env, store, config = source_store(tmp_path, monkeypatch)
    for link in (".venv", "node_modules"):
        root = store / link
        root.mkdir(exist_ok=True)
        for n in range(140):
            package = root / str(n)
            package.mkdir()
            (package / "marker").touch()
    original = health._entries
    observed = []
    witness_probes = []
    original_witness = health._witness
    def witness(fd, path):
        witness_probes.append(path)
        return original_witness(fd, path)
    monkeypatch.setattr(health, '_witness', witness)
    def count(fd, path, **kwargs):
        entries = original(fd, path, **kwargs)
        observed.append((path, len(entries)))
        return entries
    monkeypatch.setattr(health, "_entries", count)
    data = save(store, config, (".venv", "node_modules"))
    for check in (False, True):
        if check:
            observed.clear()
            assert health.inspect(store, (".venv", "node_modules"))[1] is None
        assert len(witness_probes) <= 128
        assert all(len(path.split('/')) <= 4 for path in witness_probes)
        assert len(observed) <= 64
        assert sum(n for _, n in observed) <= 8192
        assert max(len(p.split('/')) if p else 0 for p, _ in observed) <= 4
    assert sum(len(s["witnesses"]) for s in data["sources"].values()) <= 128
    assert len(health.health_path(store).read_bytes()) <= 65536
    assert health._size(data) <= 32768
    assert data["limited"]


def test_saturation_deep_and_unselected_deletion_limits(tmp_path, monkeypatch):
    env, store, config = source_store(tmp_path, monkeypatch)
    root = store / ".venv"
    shutil.rmtree(root)
    root.mkdir()
    for n in range(140):
        (root / f"p{n}").touch()
    deep = root / "a" / "b" / "c" / "d" / "e" / "f"
    deep.mkdir(parents=True)
    (deep / "child").touch()
    data = save(store, config)
    source = data['sources']['.venv']
    witnessed = {path for path, _ in source['witnesses']}
    unselected = [p for p in root.iterdir() if p.is_file() and p.name not in witnessed]
    assert len(unselected) >= 2
    unselected[0].unlink()  # total remains above cap and not a saved witness
    (deep / "child").unlink()
    assert health.inspect(store, (".venv",))[1] is None
    for p in unselected[1:]:
        p.unlink()
    # Crossing below the cap must be detected even by the count alone.
    while len(list(root.iterdir())) >= 128:
        next(p for p in root.iterdir() if p.is_file()).unlink()
    assert 'child count decreased' in health.inspect(store, (".venv",))[1]


@pytest.mark.parametrize("damage", ["record-symlink", "malformed", "identity", "ancestor", "root", "io"])
def test_unsafe_health_refuses_without_bootstrap(tmp_path, monkeypatch, damage):
    env, store, config = source_store(tmp_path, monkeypatch)
    save(store, config)
    path = health.health_path(store)
    if damage == "record-symlink":
        target = tmp_path / "outside"
        target.write_text(path.read_text())
        path.unlink()
        path.symlink_to(target)
    elif damage == "malformed":
        path.write_text('{')
    elif damage == "identity":
        data = json.loads(path.read_text())
        data['store_identity'] = [0, 0]
        path.write_text(json.dumps(data))
    elif damage == "ancestor":
        source = store / '.venv' / 'bin'
        source.rename(store / 'moved')
        source.symlink_to(store / 'moved')
    elif damage == "root":
        source = store / '.venv'
        source.rename(store / 'moved')
        source.symlink_to(store / 'moved')
    else:
        monkeypatch.setattr(health, '_entries', lambda *a, **k: (_ for _ in ()).throw(OSError('unreadable')))
    with pytest.raises(AgentLoopError, match="damaged or stale"):
        rw.prepare_store(store, config=config, runner=Runner())
    if damage == 'record-symlink':
        assert target.read_text().startswith('{')


def test_interior_symlinks_and_content_additions_do_not_rebase(tmp_path, monkeypatch):
    env, store, config = source_store(tmp_path, monkeypatch)
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'a').touch()
    (store / '.venv' / 'interior').symlink_to(outside)
    data = save(store, config)
    (outside / 'a').unlink()
    (store / '.venv' / 'new').touch()
    (store / '.venv' / 'bin' / 'python').write_text('new bytes')
    assert health.inspect(store, ('.venv',))[1] is None
    assert json.loads(health.health_path(store).read_text()) == data
    assert not any('interior/' in p for p, _ in data['sources']['.venv']['counts'])


def test_manual_bootstrap_and_setup_backed_migration(tmp_path, monkeypatch, capsys):
    env, store, config = source_store(tmp_path, monkeypatch)
    rw.prepare_store(store, config=config, runner=Runner())
    data = json.loads(health.health_path(store).read_text())
    assert data['generation']
    counter = tmp_path / 'counter'
    backed = dataclasses.replace(config, worktree_setup=setup_command(counter))
    rw.prepare_store(store, config=backed, runner=Runner())
    health.health_path(store).unlink()  # legacy setup-backed clone
    rw.prepare_store(store, config=backed, runner=Runner())
    rw.prepare_store(store, config=backed, runner=Runner())
    assert runs_in(counter, store) == 2


def test_old_registration_cannot_stale_healthy_new_generation(tmp_path, monkeypatch):
    env, store, config = source_store(tmp_path, monkeypatch)
    old = save(store, config)
    root = tmp_path / 'run'
    root.mkdir()
    (root / '.venv').symlink_to(store / '.venv')
    cv.register_worktree_links(root, {'.venv': str(store / '.venv')}, shared_store=store)
    shutil.rmtree(store / '.venv')
    (store / '.venv').mkdir()
    (store / '.venv' / 'new-generation').touch()
    new = save(store, config)
    assert old['generation'] != new['generation']
    cv._check_shared_health(config, root, post_turn=True)
    assert health.inspect(store, ('.venv',))[1] is None
    assert json.loads(health.health_path(store).read_text()) == new
    (store / '.venv' / 'new-generation').unlink()
    cv._check_shared_health(config, root, post_turn=True)
    assert health.inspect(store, ('.venv',))[1]


def test_depth_four_probes_and_long_optional_paths_reduce_coverage(tmp_path, monkeypatch):
    env, store, config = source_store(tmp_path, monkeypatch)
    root = store / '.venv'
    for n in range(70):
        parent = root / (str(n).zfill(3) + 'a' * 180)
        for depth in range(5):
            parent.mkdir()
            (parent / 'artifact').touch()
            parent /= 'b' * 180
    data = save(store, config)
    assert health._size(data) <= 32768
    assert data['limited']
    assert health.inspect(store, ('.venv',))[1] is None
    # A separate small tree establishes the exact depth boundary.
    shutil.rmtree(root)
    parent = root
    for depth in range(6):
        parent.mkdir()
        (parent / 'artifact').touch()
        parent /= 'next'
    data = save(store, config)
    probes = data['sources']['.venv']['counts']
    assert max(len(p.split('/')) if p else 0 for p, _ in probes) == 4
    (root / 'next/next/next/next/next/artifact').unlink()
    assert health.inspect(store, ('.venv',))[1] is None
    (root / 'next/next/next/next/artifact').unlink()
    assert health.inspect(store, ('.venv',))[1]


@pytest.mark.parametrize('symlink', [False, True])
def test_regular_file_and_root_symlink_sources(tmp_path, monkeypatch, symlink):
    env, store, config = source_store(tmp_path, monkeypatch)
    path = store / 'cache'
    if symlink:
        path.symlink_to(store / '.venv')
    else:
        path.write_text('original')
    save(store, config, ('cache',))
    if symlink:
        (store / '.venv' / 'bin' / 'python').write_text('new bytes')
    else:
        path.write_text('new bytes')
    assert health.inspect(store, ('cache',))[1] is None


def test_record_budget_and_probe_validation(tmp_path, monkeypatch):
    env, store, config = source_store(tmp_path, monkeypatch)
    data = save(store, config)
    for invalid in ('../outside', 'bin/../../outside', '/absolute', 'a/b/c/d/e'):
        altered = json.loads(json.dumps(data))
        altered['sources']['.venv']['counts'].append([invalid, 1])
        health.health_path(store).write_text(json.dumps(altered))
        assert health.inspect(store, ('.venv',))[1]
    health.health_path(store).write_bytes(b' ' * 65537)
    assert health.inspect(store, ('.venv',))[1]


def test_unstable_probe_is_not_accepted(tmp_path, monkeypatch):
    env, store, config = source_store(tmp_path, monkeypatch)
    save(store, config)
    original = health.os.scandir
    def unstable(fd):
        iterator = original(fd)
        (store / '.venv' / 'mutation').touch()
        return iterator
    monkeypatch.setattr(health.os, 'scandir', unstable)
    assert health.inspect(store, ('.venv',))[1]


def test_stale_publication_failure_blocks_in_process_reuse(tmp_path, monkeypatch):
    env, store, config = source_store(tmp_path, monkeypatch)
    save(store, config)
    root = tmp_path / 'run'
    root.mkdir()
    (root / '.venv').symlink_to(store / '.venv')
    cv.register_worktree_links(root, {'.venv': str(store / '.venv')}, shared_store=store)
    artifact = store / '.venv/bin/python'
    artifact.unlink()
    def refuse(*a, **k):
        raise OSError('disk full')
    monkeypatch.setattr(health, 'mark_stale', refuse)
    cv.record_checkout_baseline(config, Runner(), root, source='coder')
    assert cv.has_poisoned_entry(root)
    artifact.touch()
    with pytest.raises(CheckoutVerificationError, match='stale publication failed'):
        cv.verify_checkout(config, Runner(), path=root, agent='claude', purpose='retry')


def test_second_lock_hold_refuses_health_loss_before_worktree_creation(tmp_path, monkeypatch):
    env, store, config = source_store(tmp_path, monkeypatch)
    env.seed_store('codex')
    def lose_health(*args, **kwargs):
        health.health_path(args[0]).unlink()
    monkeypatch.setattr(rw, 'prune_dead_worktrees', lose_health)
    with workdir_claim_scope(command='task'):
        with pytest.raises(AgentLoopError, match='missing or malformed source-health record'):
            prepare_full(config)
        assert not config.claude_dir.exists()


def test_obsolete_configuration_cannot_invalidate_new_setup(tmp_path, monkeypatch):
    env, store, config = source_store(tmp_path, monkeypatch)
    save(store, config)
    root = tmp_path / 'run'
    root.mkdir()
    (root / '.venv').symlink_to(store / '.venv')
    cv.register_worktree_links(root, {'.venv': str(store / '.venv')}, shared_store=store)
    (store / 'cache').write_text('new configuration')
    newer = save(store, config, ('.venv', 'cache'))
    setup._write_setup_record(store, {'version': 1, 'sentinel': 'new setup'})
    before = setup.setup_record_path(store).read_bytes()
    cv._check_shared_health(config, root, post_turn=True)
    assert json.loads(health.health_path(store).read_text()) == newer
    assert setup.setup_record_path(store).read_bytes() == before
    assert health.inspect(store, ('.venv', 'cache'))[1] is None


def test_stale_restart_diagnostic_names_damaged_second_source(tmp_path, monkeypatch):
    env, store, config = source_store(tmp_path, monkeypatch)
    (store / 'node_modules').mkdir()
    artifact = store / 'node_modules' / 'installed'
    artifact.touch()
    save(store, config, ('.venv', 'node_modules'))
    root = tmp_path / 'run'
    root.mkdir()
    mapping = {link: str(store / link) for link in ('.venv', 'node_modules')}
    for link, target in mapping.items():
        (root / link).symlink_to(target)
    cv.register_worktree_links(root, mapping, shared_store=store)
    artifact.unlink()
    cv._check_shared_health(config, root, post_turn=True)
    cv.reset_checkout_baselines()
    _data, reason, link = health.inspect(store, ('.venv', 'node_modules'))
    assert link == 'node_modules'
    diagnostic = health.refusal(store, link, reason)
    assert str(store / 'node_modules') in diagnostic
    assert "--worktree-setup 'npm ci'" in diagnostic
