"""Deterministic duration-balanced CI test sharding (#1235).

Activated only by ``CI_SHARD_INDEX`` / ``CI_SHARD_COUNT`` (1-based index).
``CI_SHARD_MANIFEST`` names the manifest written for the aggregate verifier;
``CI_SHARD_STORE_DURATIONS`` (independent of sharding) records per-test
setup+call+teardown seconds for refreshing ``tests/.test_durations``.

Configuration is read once in ``pytest_configure`` and cached; the conftest
scrubs these variables for every test so nested pytest sessions stay inert
unless they opt in.  Output ownership comes from session state, never from
inherited environment: the manifest is written by the collecting owner (a
plain session or xdist worker ``gw0``) and durations by the reporting owner
(a plain session or the xdist controller).
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from ci_shard_verify import SCHEMA_VERSION, collection_digest

SHARD_ENV_VARS = (
    "CI_SHARD_INDEX",
    "CI_SHARD_COUNT",
    "CI_SHARD_MANIFEST",
    "CI_SHARD_STORE_DURATIONS",
)
DURATIONS_FILE = Path(__file__).parent / ".test_durations"


@dataclass(frozen=True)
class ShardConfig:
    index: int | None
    count: int | None
    manifest: str | None
    store_durations: str | None

    @property
    def sharded(self) -> bool:
        return self.count is not None


_CONFIG_KEY = pytest.StashKey[ShardConfig]()
_FULL_KEY = pytest.StashKey[list]()
_DURATIONS_KEY = pytest.StashKey[dict]()


def parse_env(environ) -> ShardConfig:
    raw_index = environ.get("CI_SHARD_INDEX")
    raw_count = environ.get("CI_SHARD_COUNT")
    index = count = None
    if (raw_index is None) != (raw_count is None):
        raise pytest.UsageError("CI_SHARD_INDEX and CI_SHARD_COUNT must be set together")
    if raw_index is not None:
        try:
            index, count = int(raw_index), int(raw_count)
        except ValueError:
            raise pytest.UsageError(
                f"CI_SHARD_INDEX/CI_SHARD_COUNT must be integers: {raw_index!r}/{raw_count!r}"
            ) from None
        if count < 1 or not 1 <= index <= count:
            raise pytest.UsageError(f"CI_SHARD_INDEX/CI_SHARD_COUNT: shard {index}/{count} is out of range")
    return ShardConfig(
        index=index,
        count=count,
        manifest=environ.get("CI_SHARD_MANIFEST") or None,
        store_durations=environ.get("CI_SHARD_STORE_DURATIONS") or None,
    )


def partition(node_ids, durations, count):
    """Greedy longest-processing-time assignment; returns ``count`` sorted groups."""
    ids = sorted(set(node_ids))
    known = [durations[i] for i in ids if i in durations]
    default = sum(known) / len(known) if known else 1.0
    weight = {i: float(durations.get(i, default)) for i in ids}
    loads = [0.0] * count
    groups: list[list[str]] = [[] for _ in range(count)]
    for node_id in sorted(ids, key=lambda i: (-weight[i], i)):
        target = min(range(count), key=lambda k: (loads[k], k))
        groups[target].append(node_id)
        loads[target] += weight[node_id]
    return [sorted(group) for group in groups]


def check_exactly_once(groups, full_ids) -> None:
    flat = [i for group in groups for i in group]
    if len(flat) != len(set(flat)) or sorted(flat) != sorted(set(full_ids)):
        raise AssertionError("shard groups are not a disjoint cover of the full collection")


@pytest.fixture(autouse=True)
def _scrub_ci_shard_environment(monkeypatch):
    """Nested pytest sessions must not inherit shard settings (#1235).

    The outer session cached its shard configuration at configure time, so
    scrubbing here is safe; tests that exercise sharding opt in explicitly.
    Provided by the plugin so any session that loads it is protected.
    """
    for name in SHARD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _is_worker(config) -> bool:
    return hasattr(config, "workerinput")


def _is_controller(config) -> bool:
    return not _is_worker(config) and config.pluginmanager.get_plugin("dsession") is not None


def _owns_manifest(config) -> bool:
    if _is_worker(config):
        return config.workerinput.get("workerid") == "gw0"
    return not _is_controller(config)


def _owns_durations(config) -> bool:
    return not _is_worker(config)


def _load_durations() -> dict:
    try:
        data = json.loads(DURATIONS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, (int, float))}


def _atomic_write(path: str, payload) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, target)


def _head_sha(config) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(config.rootpath), capture_output=True,
            text=True, check=True, timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def pytest_configure(config):
    cfg = parse_env(os.environ)
    if cfg.manifest:
        cfg = ShardConfig(cfg.index, cfg.count, os.path.abspath(cfg.manifest), cfg.store_durations)
    if cfg.store_durations:
        cfg = ShardConfig(cfg.index, cfg.count, cfg.manifest, os.path.abspath(cfg.store_durations))
    config.stash[_CONFIG_KEY] = cfg
    config.stash[_DURATIONS_KEY] = {}
    config.stash[_FULL_KEY] = []
    if cfg.store_durations and _owns_durations(config):
        config.pluginmanager.register(_DurationRecorder(config), "ci-shard-durations")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    cfg = config.stash[_CONFIG_KEY]
    full = sorted({item.nodeid for item in items})
    config.stash[_FULL_KEY] = full
    if not cfg.sharded:
        return
    groups = partition(full, _load_durations(), cfg.count)
    check_exactly_once(groups, full)
    mine = set(groups[cfg.index - 1])
    kept = [item for item in items if item.nodeid in mine]
    dropped = [item for item in items if item.nodeid not in mine]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
    items[:] = kept
    if cfg.manifest and _owns_manifest(config):
        _atomic_write(cfg.manifest, {
            "schema_version": SCHEMA_VERSION,
            "shard_index": cfg.index,
            "shard_count": cfg.count,
            "full_collection_size": len(full),
            "full_collection_digest": collection_digest(full),
            "selected_ids": sorted(mine),
            "head_sha": _head_sha(config),
            "run_id": os.environ.get("GITHUB_RUN_ID", ""),
            "run_attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT") or 0),
        })


class _DurationRecorder:
    """Per-session recorder; bound to its own config so nested sessions never share state."""

    def __init__(self, config):
        self.config = config

    def pytest_runtest_logreport(self, report):
        if report.when in ("setup", "call", "teardown"):
            durations = self.config.stash[_DURATIONS_KEY]
            durations[report.nodeid] = durations.get(report.nodeid, 0.0) + report.duration

    def pytest_sessionfinish(self, session):
        cfg = self.config.stash[_CONFIG_KEY]
        durations = self.config.stash[_DURATIONS_KEY]
        if durations:
            _atomic_write(cfg.store_durations, {k: round(v, 4) for k, v in durations.items()})
