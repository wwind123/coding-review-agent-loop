"""Run one exact-head test shard exactly as the reusable managed-CI workflow does.

The reusable workflows (``managed-ci.yml`` and ``managed-ci-ordinary.yml``)
check out ``ci/managed`` from their own repository and revision and call this
script for every shard leg.  It is the single place that composes the shard
environment, so an executable adopter test can run the identical composition.

``--shards 1`` runs the test command unchanged: no plugin is injected and no
manifest is written; the workflow's literal result gate covers it.  With more
shards the test command must be a pytest invocation; the callee-owned plugin is
injected with ``-p ci_shard_plugin`` and the leg fails if no manifest appears.

Stdlib only.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

CALLEE_DIR = Path(__file__).resolve().parent
SHARD_ENV_VARS = (
    "CI_SHARD_INDEX",
    "CI_SHARD_COUNT",
    "CI_SHARD_MANIFEST",
    "CI_SHARD_STORE_DURATIONS",
    "CI_SHARD_DURATIONS",
)
_PYTEST_COMMAND = re.compile(r"(?:^|[\s/;&|])pytest(?:\s|$)|-m\s+pytest(?:\s|$)")


class ShardUsageError(ValueError):
    pass


def compose(*, shards: int, index: int, manifest: str, durations: str, base_env) -> dict:
    """Return the child environment for one leg (the base environment when unsharded)."""
    if shards < 1 or not 1 <= index <= shards:
        raise ShardUsageError(f"shard {index}/{shards} is out of range")
    env = {k: v for k, v in base_env.items() if k not in SHARD_ENV_VARS}
    if shards == 1:
        return env
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(CALLEE_DIR), env.get("PYTHONPATH", "")) if part
    )
    env["PYTEST_ADDOPTS"] = " ".join(
        part for part in (env.get("PYTEST_ADDOPTS", ""), "-p ci_shard_plugin") if part
    )
    env["CI_SHARD_INDEX"] = str(index)
    env["CI_SHARD_COUNT"] = str(shards)
    env["CI_SHARD_MANIFEST"] = manifest
    if durations:
        env["CI_SHARD_DURATIONS"] = durations
    return env


def run_leg(*, shards: int, index: int, manifest: str, durations: str, test_command: str, base_env=None) -> int:
    if not test_command.strip():
        raise ShardUsageError("a test command is required")
    if shards > 1 and not _PYTEST_COMMAND.search(test_command):
        raise ShardUsageError("shards > 1 requires a pytest test command")
    env = compose(
        shards=shards, index=index, manifest=manifest, durations=durations,
        base_env=os.environ if base_env is None else base_env,
    )
    if shards > 1:
        # A stale manifest must never satisfy this leg.
        Path(manifest).unlink(missing_ok=True)
    code = subprocess.run(test_command, shell=True, env=env).returncode
    if code == 0 and shards > 1 and not Path(manifest).is_file():
        print(f"shard {index}/{shards} passed but wrote no manifest at {manifest}", file=sys.stderr)
        return 1
    return code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--manifest", default="shard-manifest/manifest.json")
    parser.add_argument("--durations", default="")
    parser.add_argument("--test-command", required=True)
    args = parser.parse_args(argv)
    try:
        return run_leg(
            shards=args.shards, index=args.index, manifest=args.manifest,
            durations=args.durations, test_command=args.test_command,
        )
    except ShardUsageError as exc:
        print(f"run_shard: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
