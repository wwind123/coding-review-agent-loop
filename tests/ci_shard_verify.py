"""Stdlib-only CI shard manifest verifier and canonical encoding helpers (#1235).

The shard pytest plugin (``tests/_ci_shard.py``) imports the encoding helpers
from here, and the CI aggregate jobs sparse-check out *only this file* from the
trusted workflow revision and run it as a script.  Nothing here may import
pytest or any non-stdlib package.

Usage::

    python tests/ci_shard_verify.py MANIFEST_DIR --count N --head SHA \
        --run-id ID --result RESULT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCHEMA_VERSION = 1
MANIFEST_FIELDS = (
    "schema_version",
    "shard_index",
    "shard_count",
    "full_collection_size",
    "full_collection_digest",
    "selected_ids",
    "head_sha",
    "run_id",
    "run_attempt",
)


def collection_digest(node_ids) -> str:
    """sha256 over the UTF-8 newline-joined sorted node ids."""
    return hashlib.sha256("\n".join(sorted(node_ids)).encode("utf-8")).hexdigest()


def _load_manifests(directory: Path, errors: list[str]) -> list[dict]:
    manifests = []
    for path in sorted(directory.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"unreadable manifest {path.name}: {exc}")
            continue
        if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"manifest {path.name} has an unsupported schema")
            continue
        missing = [name for name in MANIFEST_FIELDS if name not in data]
        if missing:
            errors.append(f"manifest {path.name} lacks fields: {', '.join(missing)}")
            continue
        manifests.append(data)
    return manifests


def verify(manifest_dir, count: int, head: str, run_id: str, result: str) -> list[str]:
    """Return a list of problems; empty means the shards qualify."""
    errors: list[str] = []
    if result != "success":
        return [f"shard matrix result is {result!r}, not 'success'"]
    manifests = _load_manifests(Path(manifest_dir), errors)
    valid = []
    for manifest in manifests:
        label = f"shard {manifest.get('shard_index')!r}"
        if str(manifest["run_id"]) != str(run_id):
            errors.append(f"{label}: manifest belongs to run {manifest['run_id']!r}")
            continue
        if manifest["head_sha"] != head:
            errors.append(f"{label}: manifest was produced for head {manifest['head_sha']!r}")
            continue
        ids = manifest["selected_ids"]
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            errors.append(f"{label}: selected_ids is not a list of strings")
            continue
        if len(set(ids)) != len(ids):
            errors.append(f"{label}: duplicate ids within the manifest")
            continue
        valid.append(manifest)

    best: dict[int, dict] = {}
    for manifest in valid:
        index = manifest["shard_index"]
        current = best.get(index)
        attempt = manifest["run_attempt"]
        if current is not None and current["run_attempt"] == attempt:
            errors.append(f"shard {index}: two manifests for attempt {attempt}")
        if current is None or attempt > current["run_attempt"]:
            best[index] = manifest

    expected = set(range(1, count + 1))
    missing = sorted(expected - set(best))
    if missing:
        errors.append(
            f"missing manifests for shards {missing}; re-run all jobs of this workflow run"
        )
    extra = sorted(set(best) - expected)
    if extra:
        errors.append(f"unexpected shard indexes {extra}")
    if errors:
        return errors

    selected = [best[index] for index in sorted(best)]
    if any(m["shard_count"] != count for m in selected):
        errors.append("a manifest disagrees with the expected shard count")
    sizes = {m["full_collection_size"] for m in selected}
    digests = {m["full_collection_digest"] for m in selected}
    if len(sizes) != 1 or len(digests) != 1:
        errors.append("shards saw different collections (size or digest differ)")
    if errors:
        return errors

    union: list[str] = []
    seen: set[str] = set()
    for manifest in selected:
        for node_id in manifest["selected_ids"]:
            if node_id in seen:
                errors.append(f"test selected by more than one shard: {node_id}")
            seen.add(node_id)
            union.append(node_id)
    if errors:
        return errors[:20]
    if len(union) != next(iter(sizes)):
        errors.append(
            f"union size {len(union)} differs from full collection size {next(iter(sizes))}"
        )
    elif collection_digest(union) != next(iter(digests)):
        errors.append("union of selected ids does not match the full collection digest")
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_dir")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--result", default="")
    args = parser.parse_args(argv)
    errors = verify(args.manifest_dir, args.count, args.head, args.run_id, args.result)
    if errors:
        for error in errors:
            print(f"shard verification failed: {error}", file=sys.stderr)
        return 1
    print(f"All {args.count} shards verified: exactly-once coverage of the full collection.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
