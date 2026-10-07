"""Managed CI attestation record writer and fail-closed verifier (#1312, parent #1308).

A caller-owned test job ends with the ``managed-ci-attest`` composite action,
which runs ``write`` here; the reusable publish workflow's read-only verify job
runs ``verify``.  Both run from the callee/action revision, never from the
tested commit.  Stdlib only.

An attestation is an *untrusted correlation claim*: it is written in a runner
that has executed PR-head code, so PR code can forge, alter or suppress it.  The
verifier uses it only for accounting (a declared identity reached its final
step in this run and attempt naming the validated target).  Pass/fail authority
is the Actions jobs API conclusion of the expected job name plus the caller's
``toJSON(needs)`` results.  The record's ``job_status`` is never trusted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SCHEMA = "managed-ci-attestation/1"
RECORD_FILENAME = "attestation.json"
MAX_RECORD_BYTES = 16 * 1024
MAX_EXPECTED = 256
ARTIFACT_PREFIX = "managed-ci-attest-"

RECORD_FIELDS = (
    "schema",
    "attestation_id",
    "job_name",
    "target_sha",
    "head_sha",
    "run_id",
    "run_attempt",
    "repository",
    "job_status",
)
EXPECTED_FIELDS = ("attestation_id", "job_name", "needs_key")

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_REPO_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_ARTIFACT_RE = re.compile(r"^managed-ci-attest-([a-z0-9][a-z0-9._-]{0,63})-attempt-([1-9][0-9]{0,5})$")
_NEEDS_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


def artifact_name(attestation_id: str, run_attempt: int) -> str:
    return f"{ARTIFACT_PREFIX}{attestation_id}-attempt-{run_attempt}"


def parse_artifact_name(name: str) -> tuple[str, int] | None:
    match = _ARTIFACT_RE.fullmatch(name) if isinstance(name, str) else None
    if not match:
        return None
    return match.group(1), int(match.group(2))


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_record(record) -> list[str]:
    """Check a decoded record against the exact ``managed-ci-attestation/1`` schema."""
    if not isinstance(record, dict):
        return ["record is not a JSON object"]
    errors: list[str] = []
    missing = [key for key in RECORD_FIELDS if key not in record]
    extra = sorted(str(key) for key in record if key not in RECORD_FIELDS)
    if missing:
        errors.append(f"record lacks fields: {', '.join(missing)}")
    if extra:
        errors.append(f"record has unknown fields: {', '.join(extra)}")
    if errors:
        return errors
    if record["schema"] != SCHEMA:
        errors.append(f"unsupported schema {record['schema']!r}")
    value = record["attestation_id"]
    if not (isinstance(value, str) and _ID_RE.fullmatch(value)):
        errors.append("attestation_id is malformed")
    value = record["job_name"]
    if not (isinstance(value, str) and 0 < len(value) <= 256):
        errors.append("job_name is malformed")
    for key in ("target_sha", "head_sha"):
        if not (isinstance(record[key], str) and _SHA_RE.fullmatch(record[key])):
            errors.append(f"{key} is not a lowercase 40-hex SHA")
    for key in ("run_id", "run_attempt"):
        if not (_is_int(record[key]) and record[key] > 0):
            errors.append(f"{key} is not a positive integer")
    if not (isinstance(record["repository"], str) and _REPO_RE.fullmatch(record["repository"])):
        errors.append("repository is not owner/name")
    if not isinstance(record["job_status"], str):
        errors.append("job_status is not a string")
    return errors


def _decode_record(content) -> tuple[object, str | None]:
    if isinstance(content, str):
        content = content.encode("utf-8")
    if not isinstance(content, (bytes, bytearray)):
        return None, "record content is not bytes"
    if len(content) > MAX_RECORD_BYTES:
        return None, f"record exceeds {MAX_RECORD_BYTES} bytes"
    try:
        return json.loads(bytes(content).decode("utf-8")), None
    except (ValueError, UnicodeDecodeError) as exc:
        return None, f"record is not valid JSON: {exc}"


def validate_expected(expected) -> list[str]:
    if not isinstance(expected, list) or not expected:
        return ["expected set must be a non-empty JSON array"]
    if len(expected) > MAX_EXPECTED:
        return [f"expected set exceeds {MAX_EXPECTED} entries"]
    errors: list[str] = []
    ids: set = set()
    names: set = set()
    for index, entry in enumerate(expected):
        label = f"expected[{index}]"
        if not isinstance(entry, dict) or set(entry) != set(EXPECTED_FIELDS):
            errors.append(f"{label} must be an object with exactly {', '.join(EXPECTED_FIELDS)}")
            continue
        entry_id, name, key = entry["attestation_id"], entry["job_name"], entry["needs_key"]
        if not (isinstance(entry_id, str) and _ID_RE.fullmatch(entry_id)):
            errors.append(f"{label}.attestation_id is malformed")
        elif entry_id in ids:
            errors.append(f"{label}: duplicate attestation_id {entry_id!r}")
        else:
            ids.add(entry_id)
        if not (isinstance(name, str) and 0 < len(name) <= 256):
            errors.append(f"{label}.job_name is malformed")
        elif name in names:
            errors.append(f"{label}: duplicate job_name {name!r}")
        else:
            names.add(name)
        if not (isinstance(key, str) and _NEEDS_KEY_RE.fullmatch(key)):
            errors.append(f"{label}.needs_key is malformed")
    return errors


def _normalise_artifacts(artifacts):
    if isinstance(artifacts, dict):
        return list(artifacts.items())
    return list(artifacts)


def verify(expected, artifacts, api_jobs, needs_results, run_id, run_attempt, target_sha, repository) -> list[str]:
    """Return failure reasons; an empty list means every rule passed.

    ``artifacts`` maps artifact name -> ``{filename: bytes}`` (or is a list of
    such pairs, so duplicates survive).  ``api_jobs`` is the complete jobs-API
    listing for this run attempt, or ``None`` when it was unavailable.
    """
    errors = validate_expected(expected)
    if errors:
        return errors
    if not (_is_int(run_id) and run_id > 0 and _is_int(run_attempt) and run_attempt > 0):
        return ["publisher run_id and run_attempt must be positive integers"]
    if not (isinstance(target_sha, str) and _SHA_RE.fullmatch(target_sha)):
        return ["target_sha is not a lowercase 40-hex SHA"]
    if not (isinstance(repository, str) and _REPO_RE.fullmatch(repository)):
        return ["repository is not owner/name"]

    by_id = {entry["attestation_id"]: entry for entry in expected}

    # Current-attempt artifact selection by strictly parsed name.
    selected: list[tuple[str, int, object]] = []
    for name, files in _normalise_artifacts(artifacts):
        if not (isinstance(name, str) and name.startswith(ARTIFACT_PREFIX)):
            continue
        parsed = parse_artifact_name(name)
        if parsed is None:
            errors.append(f"malformed attestation artifact name {name!r}")
            continue
        attempt = parsed[1]
        if attempt < run_attempt:
            continue
        if attempt > run_attempt:
            errors.append(f"artifact {name!r} is from a later attempt than {run_attempt}")
            continue
        selected.append((name, parsed[0], files))

    records: dict[str, dict] = {}
    for name, name_id, files in selected:
        if not isinstance(files, dict) or set(files) != {RECORD_FILENAME}:
            found = sorted(map(str, files)) if isinstance(files, dict) else files
            errors.append(f"artifact {name!r} must contain exactly {RECORD_FILENAME}, found {found!r}")
            continue
        record, problem = _decode_record(files[RECORD_FILENAME])
        if problem:
            errors.append(f"artifact {name!r}: {problem}")
            continue
        problems = validate_record(record)
        if problems:
            errors.append(f"artifact {name!r}: " + "; ".join(problems))
            continue
        if record["attestation_id"] != name_id:
            errors.append(f"artifact {name!r}: record attestation_id {record['attestation_id']!r} differs from the name")
            continue
        if record["run_attempt"] != run_attempt:
            errors.append(f"artifact {name!r}: record run_attempt {record['run_attempt']!r} differs from the name")
            continue
        if name_id in records:
            errors.append(f"duplicate attestation for id {name_id!r}")
            continue
        if name_id not in by_id:
            errors.append(f"unexpected attestation id {name_id!r}")
            continue
        records[name_id] = record
    # A duplicate or extra id invalidates the whole set even when a sibling was valid.
    seen = [name_id for _, name_id, _ in selected]
    for name_id in sorted(set(seen)):
        if seen.count(name_id) > 1 and not any(f"duplicate attestation for id {name_id!r}" == e for e in errors):
            errors.append(f"duplicate attestation for id {name_id!r}")

    for entry in expected:
        entry_id = entry["attestation_id"]
        record = records.get(entry_id)
        if record is None:
            if entry_id not in seen:
                errors.append(f"missing attestation for {entry_id!r} in attempt {run_attempt}")
            continue
        if record["job_name"] != entry["job_name"]:
            errors.append(f"{entry_id!r}: record job_name {record['job_name']!r} differs from expected {entry['job_name']!r}")
        for key in ("target_sha", "head_sha"):
            if record[key] != target_sha:
                errors.append(f"{entry_id!r}: record {key} {record[key]!r} is not the validated target")
        if record["run_id"] != run_id:
            errors.append(f"{entry_id!r}: record run_id {record['run_id']!r} is another run")
        if record["run_attempt"] != run_attempt:
            errors.append(f"{entry_id!r}: record run_attempt {record['run_attempt']!r} is another attempt")
        if record["repository"] != repository:
            errors.append(f"{entry_id!r}: record repository {record['repository']!r} is another repository")

    errors.extend(_verify_api_jobs(expected, api_jobs))
    errors.extend(_verify_needs(expected, needs_results))
    return errors


def _verify_api_jobs(expected, api_jobs) -> list[str]:
    if not isinstance(api_jobs, list):
        return ["jobs API listing is unavailable or malformed"]
    if not all(isinstance(job, dict) and isinstance(job.get("name"), str) for job in api_jobs):
        return ["jobs API listing has a malformed job entry"]
    errors: list[str] = []
    for entry in expected:
        matches = [job for job in api_jobs if job["name"] == entry["job_name"]]
        if len(matches) != 1:
            errors.append(f"{entry['attestation_id']!r}: {len(matches)} API jobs named {entry['job_name']!r}, expected exactly one")
        elif matches[0].get("conclusion") != "success":
            errors.append(f"{entry['attestation_id']!r}: API conclusion of {entry['job_name']!r} is {matches[0].get('conclusion')!r}")
    return errors


def _verify_needs(expected, needs_results) -> list[str]:
    if not isinstance(needs_results, dict) or not needs_results:
        return ["needs results must be a non-empty JSON object"]
    errors: list[str] = []
    for key, value in needs_results.items():
        if not (isinstance(value, dict) and isinstance(value.get("result"), str)):
            errors.append(f"needs[{key!r}] is not an object with a string result")
        elif value["result"] != "success":
            errors.append(f"needs[{key!r}] result is {value['result']!r}")
    for entry in expected:
        if entry["needs_key"] not in needs_results:
            errors.append(f"needs lacks expected key {entry['needs_key']!r}")
    return errors


def build_record(*, attestation_id, job_name, target_sha, head_sha, run_id, run_attempt, repository, job_status) -> dict:
    return {
        "schema": SCHEMA,
        "attestation_id": attestation_id,
        "job_name": job_name,
        "target_sha": target_sha,
        "head_sha": head_sha,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "repository": repository,
        "job_status": job_status,
    }


def _git_head(workdir: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=workdir, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"git rev-parse HEAD failed: {result.stderr.strip()}")
    return result.stdout.strip()


def cmd_write(args) -> int:
    if not _SHA_RE.fullmatch(args.target_sha or ""):
        print("refusing to attest: target_sha is not a lowercase 40-hex SHA", file=sys.stderr)
        return 1
    try:
        head = _git_head(args.workdir)
    except (OSError, RuntimeError) as exc:
        print(f"refusing to attest: {exc}", file=sys.stderr)
        return 1
    if head != args.target_sha:
        print(f"refusing to attest: HEAD {head} is not the validated target {args.target_sha}", file=sys.stderr)
        return 1
    try:
        run_id = int(os.environ.get("GITHUB_RUN_ID", ""))
        run_attempt = int(os.environ.get("GITHUB_RUN_ATTEMPT", ""))
    except ValueError:
        print("refusing to attest: GITHUB_RUN_ID/GITHUB_RUN_ATTEMPT are not integers", file=sys.stderr)
        return 1
    record = build_record(
        attestation_id=args.attestation_id,
        job_name=args.job_name,
        target_sha=args.target_sha,
        head_sha=head,
        run_id=run_id,
        run_attempt=run_attempt,
        repository=os.environ.get("GITHUB_REPOSITORY", ""),
        job_status=args.job_status,
    )
    problems = validate_record(record)
    if problems:
        print("refusing to attest: " + "; ".join(problems), file=sys.stderr)
        return 1
    payload = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    if len(payload) > MAX_RECORD_BYTES:
        print(f"refusing to attest: record is {len(payload)} bytes, over the {MAX_RECORD_BYTES} cap", file=sys.stderr)
        return 1
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / RECORD_FILENAME).write_bytes(payload)
    print(artifact_name(args.attestation_id, run_attempt))
    return 0


def _load_artifacts(directory: Path) -> list[tuple[str, dict]]:
    artifacts = []
    for child in sorted(directory.iterdir()):
        if child.is_dir() and not child.is_symlink():
            # Keep each file's path relative to the artifact root so nested or
            # same-basename files cannot collapse into one root attestation.json.
            files = {}
            for p in sorted(child.rglob("*")):
                if p.is_dir() and not p.is_symlink():
                    continue
                key = p.relative_to(child).as_posix()
                files[key if p.is_file() and not p.is_symlink() else key + " (not a regular file)"] = (
                    p.read_bytes() if p.is_file() and not p.is_symlink() else b""
                )
            artifacts.append((child.name, files))
        else:
            artifacts.append((child.name, None))  # a stray non-directory entry fails verification
    return artifacts


def cmd_verify(args) -> int:
    try:
        expected = json.loads(args.expected)
        needs = json.loads(args.needs)
    except ValueError as exc:
        print(f"verification failed: input is not JSON: {exc}", file=sys.stderr)
        return 1
    api_jobs = None
    try:
        raw = json.loads(Path(args.jobs_file).read_text(encoding="utf-8"))
        if isinstance(raw, list):
            api_jobs = raw
    except (OSError, ValueError):
        api_jobs = None
    try:
        artifacts = _load_artifacts(Path(args.artifacts_dir))
        errors = verify(
            expected, artifacts, api_jobs, needs,
            int(args.run_id), int(args.run_attempt), args.target_sha, args.repository,
        )
    except (OSError, ValueError) as exc:
        errors = [f"verification input unreadable: {exc}"]
    for error in errors:
        print(f"attestation verification: {error}", file=sys.stderr)
    return 1 if errors else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    write = sub.add_parser("write", help="write an attestation record after checking HEAD")
    write.add_argument("--target-sha", required=True)
    write.add_argument("--attestation-id", required=True)
    write.add_argument("--job-name", required=True)
    write.add_argument("--job-status", default="success")
    write.add_argument("--output-dir", required=True)
    write.add_argument("--workdir", default=".")
    write.set_defaults(func=cmd_write)
    check = sub.add_parser("verify", help="verify downloaded attestations")
    check.add_argument("--expected", required=True, help="JSON expected set")
    check.add_argument("--needs", required=True, help="JSON toJSON(needs) object")
    check.add_argument("--artifacts-dir", required=True)
    check.add_argument("--jobs-file", required=True, help="JSON list of API jobs")
    check.add_argument("--run-id", required=True)
    check.add_argument("--run-attempt", required=True)
    check.add_argument("--target-sha", required=True)
    check.add_argument("--repository", required=True)
    check.set_defaults(func=cmd_verify)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
