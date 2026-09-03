"""Bounded local test execution policy and advisory runtime memory.

The runtime sidecar is intentionally independent from the markdown repository
profile.  It contains measurements made by agent-loop itself; text narrated by
an agent is never used as a timing sample.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping, Sequence

from .errors import AgentLoopError

DEFAULT_TEST_TIMEOUT_SECONDS = 1800
RUNTIME_SCHEMA_VERSION = 1
RUNTIME_SIDECAR_NAME = "test-runtime.json"
RUNTIME_LOCK_NAME = "test-runtime.json.lock"
MAX_OBSERVATIONS_PER_COHORT = 20
MAX_COHORTS = 200
STALE_AFTER = timedelta(days=30)
_HASHED_ENV_VALUE_RE = re.compile(r"<sha256:[0-9a-f]{16}>")


class TestRuntimeConfigurationError(AgentLoopError):
    """A wrapper policy was malformed or exceeded its inherited ceiling."""


@dataclass(frozen=True)
class ManagedTestInvocation:
    """A validated absolute ``agent-loop run-tests`` invocation."""

    inner_argv: tuple[str, ...]
    timeout_seconds: float | None = None
    memory_dir: Path | None = None
    prefix_argv: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeRecommendation:
    command: str
    fingerprint: str
    successful_samples: int
    median_seconds: float | None
    p95_seconds: float | None
    latest_success_seconds: float | None
    unresolved_timeout_seconds: float | None
    recommended_timeout_seconds: int
    confidence: str
    freshness: str
    ceiling_insufficient: bool = False


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TestRuntimeConfigurationError(f"{name} must be a positive finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TestRuntimeConfigurationError(f"{name} must be a positive finite number.") from exc
    if not math.isfinite(number) or number <= 0:
        raise TestRuntimeConfigurationError(f"{name} must be a positive finite number.")
    return number


def validate_timeout_ceiling(value: object, *, name: str = "timeout ceiling") -> int:
    """Validate a finite, positive integral policy value."""
    number = _finite_positive(value, name=name)
    if not number.is_integer():
        raise TestRuntimeConfigurationError(f"{name} must be a positive integer number of seconds.")
    return int(number)


def resolve_timeout_seconds(
    requested: object | None,
    *,
    policy_ceiling: object | None = None,
    default: int = DEFAULT_TEST_TIMEOUT_SECONDS,
) -> int | float:
    """Resolve one invocation's watchdog before a child process is spawned."""
    ceiling = (
        validate_timeout_ceiling(policy_ceiling, name="AGENT_LOOP_CODER_TEST_TIMEOUT_CEILING_SECONDS")
        if policy_ceiling is not None
        else validate_timeout_ceiling(default, name="default timeout ceiling")
    )
    if requested is None:
        return ceiling
    chosen = _finite_positive(requested, name="--timeout-seconds")
    if chosen > ceiling:
        raise TestRuntimeConfigurationError(
            f"--timeout-seconds ({chosen}) cannot exceed the configured test timeout ceiling ({ceiling})."
        )
    return int(chosen) if chosen.is_integer() else chosen


def inherited_timeout_ceiling(env: Mapping[str, str] | None = None) -> int:
    values = env if env is not None else os.environ
    raw = values.get("AGENT_LOOP_CODER_TEST_TIMEOUT_CEILING_SECONDS")
    if raw is None:
        return DEFAULT_TEST_TIMEOUT_SECONDS
    return validate_timeout_ceiling(raw, name="AGENT_LOOP_CODER_TEST_TIMEOUT_CEILING_SECONDS")


def _is_absolute_executable(token: str) -> bool:
    return bool(token) and Path(token).is_absolute()


def parse_managed_test_invocation(argv: Sequence[str]) -> ManagedTestInvocation | None:
    """Parse the exact wrapper contract, returning ``None`` for a bare command.

    A recognized-looking wrapper with malformed options raises a configuration
    error.  Consumers that inspect untrusted response text can catch that error
    and fail closed to the original command.
    """
    tokens = tuple(str(item) for item in argv)
    prefix_len = 0
    if len(tokens) >= 2 and _is_absolute_executable(tokens[0]) and Path(tokens[0]).name == "agent-loop":
        if tokens[1] != "run-tests":
            return None
        prefix_len = 2
    elif (
        len(tokens) >= 4
        and _is_absolute_executable(tokens[0])
        and tokens[1] == "-m"
        and tokens[2] == "coding_review_agent_loop.cli"
        and tokens[3] == "run-tests"
    ):
        prefix_len = 4
    else:
        return None

    timeout: float | None = None
    memory_dir: Path | None = None
    seen: set[str] = set()
    index = prefix_len
    delimiter = False
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            delimiter = True
            index += 1
            break
        if not token.startswith("-"):
            raise TestRuntimeConfigurationError(
                "managed run-tests options must end with `--` before the inner command."
            )
        name, equals, value = token.partition("=")
        if name not in {"--timeout-seconds", "--memory-dir"} or name in seen:
            raise TestRuntimeConfigurationError(f"unknown or duplicate run-tests option: {token}")
        seen.add(name)
        if not equals:
            if index + 1 >= len(tokens) or tokens[index + 1] == "--":
                raise TestRuntimeConfigurationError(f"{name} requires a value.")
            value = tokens[index + 1]
            index += 1
        if name == "--timeout-seconds":
            timeout = _finite_positive(value, name="--timeout-seconds")
        else:
            if not value or value.startswith("-"):
                raise TestRuntimeConfigurationError("--memory-dir requires a path value.")
            memory_dir = Path(value)
        index += 1
    if not delimiter or index >= len(tokens):
        raise TestRuntimeConfigurationError(
            "managed run-tests requires `--` followed by a non-empty inner command."
        )
    return ManagedTestInvocation(tokens[index:], timeout, memory_dir, tokens[:prefix_len])


def resolve_wrapper_prefix() -> tuple[str, ...] | None:
    """Return a path-stable wrapper prefix suitable for a coder prompt."""
    entry = shutil.which("agent-loop")
    if entry and os.path.isabs(entry) and os.access(entry, os.X_OK):
        return (str(Path(entry).resolve()), "run-tests")
    executable = Path(sys.executable).resolve()
    if executable.is_file() and os.access(executable, os.X_OK):
        return (str(executable), "-m", "coding_review_agent_loop.cli", "run-tests")
    return None


def render_test_wrapper(
    command: Sequence[str],
    *,
    timeout_seconds: int | None = None,
    memory_dir: Path | None = None,
) -> str:
    prefix = resolve_wrapper_prefix()
    if prefix is None:
        return shlex.join(str(item) for item in command)
    options: list[str] = []
    if timeout_seconds is not None:
        options += ["--timeout-seconds", str(timeout_seconds)]
    if memory_dir is not None:
        options += ["--memory-dir", str(memory_dir)]
    return shlex.join((*prefix, *options, "--", *(str(item) for item in command)))


def _relative_or_basename(value: str, cwd: Path) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        try:
            return path.resolve(strict=False).relative_to(cwd.resolve()).as_posix()
        except ValueError:
            return path.name or "<external-path>"
    return value


def _inner_command(argv: Sequence[str]) -> tuple[str, ...]:
    try:
        parsed = parse_managed_test_invocation(argv)
    except TestRuntimeConfigurationError:
        parsed = None
    return parsed.inner_argv if parsed is not None else tuple(str(item) for item in argv)


def normalize_test_command(argv: Sequence[str], *, cwd: Path | None = None) -> str:
    """Canonicalize wrapper spellings while retaining meaningful inner argv."""
    base = (cwd or Path.cwd()).resolve()
    parsed: ManagedTestInvocation | None
    try:
        parsed = parse_managed_test_invocation(argv)
    except TestRuntimeConfigurationError:
        parsed = None
    inner = parsed.inner_argv if parsed is not None else tuple(str(a) for a in argv)
    normalized: list[str] = []
    for index, value in enumerate(inner):
        if "=" in value and value.split("=", 1)[0].replace("_", "").isalnum():
            key, _sep, raw = value.partition("=")
            if _HASHED_ENV_VALUE_RE.fullmatch(raw):
                normalized.append(value)
            else:
                normalized.append(f"{key}=<sha256:{hashlib.sha256(raw.encode()).hexdigest()[:16]}>")
        elif index == 0 or value.startswith("/") or value.startswith("~"):
            normalized.append(_relative_or_basename(value, base))
        else:
            normalized.append(value)
    return shlex.join(normalized)


def _safe_rel(path: Path, cwd: Path) -> str | None:
    try:
        return path.resolve(strict=False).relative_to(cwd.resolve()).as_posix()
    except ValueError:
        return None


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(64 * 1024), b""):
                digest.update(chunk)
    except (OSError, UnicodeError):
        return "unreadable"
    return digest.hexdigest()


def build_input_manifest(argv: Sequence[str], cwd: Path) -> dict[str, str]:
    """Hash cheap, command-relevant checkout inputs without inventory commands."""
    root = cwd.resolve()
    argv = _inner_command(argv)
    names = {
        "pyproject.toml", "pytest.ini", "tox.ini", "setup.cfg", "package.json",
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "uv.lock",
        "Pipfile.lock", "Gemfile.lock", "Cargo.lock", "go.sum", "playwright.config.ts",
        "playwright.config.js", "playwright.config.mjs",
    }
    paths: set[Path] = {root / name for name in names}
    # Explicit target files and directories influence fixtures/configuration.
    for raw in argv:
        candidate_text = (
            raw.split("=", 1)[1]
            if "=" in raw and raw.split("=", 1)[0].startswith("-")
            else raw
        )
        if raw.startswith("-") and "=" not in raw:
            continue
        candidate_path = Path(candidate_text)
        candidate = (
            (root / candidate_text).resolve(strict=False)
            if not candidate_path.is_absolute()
            else candidate_path
        )
        if candidate.is_file():
            paths.add(candidate)
        elif candidate.is_dir():
            for name in ("conftest.py", "pytest.ini", "playwright.config.ts", "playwright.config.js"):
                paths.add(candidate / name)
        parent = candidate if candidate.is_dir() else candidate.parent
        for ancestor in (parent, *parent.parents):
            fixture = ancestor / "conftest.py"
            if fixture == root.parent / "conftest.py":
                break
            paths.add(fixture)
            if ancestor == root:
                break
    manifest: dict[str, str] = {}
    for path in sorted(paths):
        relative = _safe_rel(path, root)
        if relative is None or not path.is_file():
            continue
        manifest[relative] = _hash_file(path)
    return manifest


def _resolve_executable(argv: Sequence[str], cwd: Path, values: Mapping[str, str]) -> str:
    inner = _inner_command(argv)
    if not inner:
        return ""
    raw = inner[0]
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    if raw.startswith((".", "~")) or candidate.parent != Path("."):
        return str((cwd / candidate).resolve(strict=False))
    try:
        resolved = shutil.which(raw, path=values.get("PATH"))
    except TypeError:
        # Test doubles and older Python-compatible shims may only accept the
        # command positional argument.  Fingerprinting must remain best-effort.
        resolved = shutil.which(raw)
    return resolved or raw


def environment_fingerprint(argv: Sequence[str], cwd: Path, *, env: Mapping[str, str] | None = None) -> str:
    values = env if env is not None else os.environ
    executable_path = _resolve_executable(argv, cwd, values)
    try:
        stat = os.stat(executable_path)
        version = f"mtime:{stat.st_mtime_ns}:size:{stat.st_size}"
    except OSError:
        version = "unknown"
    cpu_count = os.cpu_count() or 1
    cpu_bucket = next(
        (bucket for bucket in (1, 2, 4, 8, 16, 32, 64) if cpu_count <= bucket),
        64,
    )
    payload = {
        "runtime": platform.python_version(),
        "platform": platform.system().lower(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "cpu_bucket": cpu_bucket,
        "executable": Path(executable_path).name,
        "version": version,
        "env_assignments": [
            f"{key}=<sha256:{hashlib.sha256(value.encode()).hexdigest()[:16]}>"
            for key, value in sorted(values.items())
            if key in {"PYTHONPATH", "VIRTUAL_ENV", "NODE_PATH", "PATH"}
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _sidecar_payload(memory_dir: Path) -> dict | None:
    path = memory_dir / RUNTIME_SIDECAR_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"schema_version": RUNTIME_SCHEMA_VERSION, "observations": []}
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != RUNTIME_SCHEMA_VERSION:
        return None
    observations = payload.get("observations")
    if not isinstance(observations, list):
        return None
    return {"schema_version": RUNTIME_SCHEMA_VERSION, "observations": [item for item in observations if isinstance(item, dict)]}


def load_runtime_memory(memory_dir: Path) -> list[dict]:
    payload = _sidecar_payload(memory_dir)
    return list(payload["observations"]) if payload is not None else []


def _lock_file(path: Path, timeout: float = 5.0):
    handle = path.open("a+")
    deadline = time.monotonic() + timeout
    if os.name == "nt":
        import msvcrt
        while True:
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return handle
            except OSError:
                if time.monotonic() >= deadline:
                    handle.close()
                    return None
                time.sleep(0.05)
    import fcntl
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except BlockingIOError:
            if time.monotonic() >= deadline:
                handle.close()
                return None
            time.sleep(0.05)


def _unlock_file(handle) -> None:
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def record_test_observation(
    memory_dir: Path | None,
    *,
    argv: Sequence[str],
    cwd: Path,
    outcome: str,
    elapsed_seconds: float,
    attempted_timeout_seconds: int | float,
    policy_ceiling_seconds: int,
    returncode: int | None = None,
    commit: str | None = None,
    environment: Mapping[str, str] | None = None,
    timestamp: datetime | None = None,
) -> bool:
    """Append a bounded observation; persistence failure never affects execution."""
    if memory_dir is None:
        return False
    try:
        memory_dir.mkdir(parents=True, exist_ok=True)
        lock = _lock_file(memory_dir / RUNTIME_LOCK_NAME)
        if lock is None:
            return False
        try:
            payload = _sidecar_payload(memory_dir)
            if payload is None:
                return False
            normalized = normalize_test_command(argv, cwd=cwd)
            fingerprint = environment_fingerprint(argv, cwd, env=environment)
            observation = {
                "normalized_command": normalized,
                "environment_fingerprint": fingerprint,
                "outcome": outcome,
                "elapsed_seconds": round(max(0.0, float(elapsed_seconds)), 6),
                "attempted_timeout_seconds": attempted_timeout_seconds,
                "policy_ceiling_seconds": policy_ceiling_seconds,
                "returncode": returncode,
                "timestamp": (timestamp or _utc_now()).astimezone(timezone.utc).isoformat(),
                "commit": commit or _git_commit(cwd),
                "input_manifest": build_input_manifest(argv, cwd),
            }
            rows = payload["observations"]
            rows.append(observation)
            by_cohort: dict[tuple[str, str], list[dict]] = defaultdict(list)
            for row in rows:
                key = (str(row.get("normalized_command", "")), str(row.get("environment_fingerprint", "")))
                by_cohort[key].append(row)
            cohorts = sorted(
                by_cohort.items(),
                key=lambda pair: max((_timestamp(row.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc) for row in pair[1])),
                reverse=True,
            )[:MAX_COHORTS]
            kept: list[dict] = []
            for _key, group in cohorts:
                group.sort(key=lambda row: _timestamp(row.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
                kept.extend(group[:MAX_OBSERVATIONS_PER_COHORT])
            payload["observations"] = kept
            fd, temp_name = tempfile.mkstemp(prefix=".test-runtime-", suffix=".tmp", dir=memory_dir)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(payload, stream, indent=2, sort_keys=True)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp_name, memory_dir / RUNTIME_SIDECAR_NAME)
            finally:
                try:
                    os.unlink(temp_name)
                except FileNotFoundError:
                    pass
            return True
        finally:
            _unlock_file(lock)
    except (OSError, ValueError, TypeError, TestRuntimeConfigurationError):
        return False


def _git_commit(cwd: Path) -> str | None:
    try:
        result = subprocess.run(("git", "rev-parse", "HEAD"), cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False, timeout=2)
    except OSError:
        return None
    value = result.stdout.strip()
    return value or None


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _next_minute(value: float) -> int:
    return max(60, int(math.ceil(value / 60.0) * 60))


def recommend_timeout(
    memory_dir: Path | None,
    *,
    argv: Sequence[str],
    cwd: Path,
    policy_ceiling_seconds: int,
    now: datetime | None = None,
    normalized_command_override: str | None = None,
    fingerprint_override: str | None = None,
) -> RuntimeRecommendation:
    ceiling = validate_timeout_ceiling(policy_ceiling_seconds, name="policy ceiling")
    normalized = (
        normalized_command_override
        if normalized_command_override is not None
        else normalize_test_command(argv, cwd=cwd)
    )
    fingerprint = (
        fingerprint_override
        if fingerprint_override is not None
        else environment_fingerprint(argv, cwd)
    )
    rows = load_runtime_memory(memory_dir) if memory_dir is not None else []
    current_manifest = build_input_manifest(argv, cwd)
    cutoff = (now or _utc_now()).astimezone(timezone.utc) - STALE_AFTER
    matching: list[dict] = []
    for row in rows:
        if row.get("normalized_command") != normalized or row.get("environment_fingerprint") != fingerprint:
            continue
        stamp = _timestamp(row.get("timestamp"))
        if stamp is None or stamp < cutoff:
            continue
        if row.get("input_manifest") != current_manifest:
            continue
        matching.append(row)
    timestamped: list[tuple[datetime, int, dict]] = []
    for index, row in enumerate(matching):
        stamp = _timestamp(row.get("timestamp"))
        if stamp is not None:
            timestamped.append((stamp, index, row))
    timestamped.sort(key=lambda item: (item[0], item[1]))
    ordered = [row for _stamp, _index, row in timestamped]
    successes = [row for row in ordered if row.get("outcome") == "passed" and isinstance(row.get("elapsed_seconds"), (int, float))]
    success_values = [float(row["elapsed_seconds"]) for row in successes]
    latest_success = next(
        (
            float(row["elapsed_seconds"])
            for row in reversed(ordered)
            if row.get("outcome") == "passed" and isinstance(row.get("elapsed_seconds"), (int, float))
        ),
        None,
    )
    unresolved: float | None = None
    for row in ordered:
        if row.get("outcome") == "timed_out" and isinstance(row.get("attempted_timeout_seconds"), (int, float)):
            unresolved = float(row["attempted_timeout_seconds"])
        elif row.get("outcome") == "passed":
            unresolved = None
    candidate = ceiling
    if success_values:
        candidate = max(0, _next_minute(max(1.25 * _nearest_rank(success_values, 0.95), latest_success + 60)))
    if unresolved is not None:
        candidate = max(candidate, _next_minute(max(1.5 * unresolved, unresolved + 300)))
    clamped = min(ceiling, candidate)
    freshness = "fresh" if matching else "unknown"
    confidence = "high" if len(success_values) >= 3 else ("sparse/low-confidence" if success_values else "unknown")
    insufficient = unresolved is not None and candidate > ceiling
    return RuntimeRecommendation(
        normalized, fingerprint, len(success_values),
        (float(median(success_values)) if success_values else None),
        (_nearest_rank(success_values, 0.95) if success_values else None),
        latest_success, unresolved, int(clamped), confidence, freshness, insufficient,
    )


def render_runtime_context(
    memory_dir: Path | None,
    *,
    commands: Iterable[Sequence[str]],
    cwd: Path,
    policy_ceiling_seconds: int,
    recommendations: Mapping[tuple[str, ...], RuntimeRecommendation] | None = None,
) -> str:
    lines: list[str] = []
    for command in list(commands)[:6]:
        key = tuple(str(item) for item in command)
        recommendation = recommendations.get(key) if recommendations is not None else None
        if recommendation is None:
            recommendation = recommend_timeout(
                memory_dir, argv=command, cwd=cwd,
                policy_ceiling_seconds=policy_ceiling_seconds,
            )
        lines.append(f"- Command: {recommendation.command}")
        if recommendation.successful_samples:
            lines.append(
                f"  Successful samples: {recommendation.successful_samples}; "
                f"median: {recommendation.median_seconds:.0f}s; upper estimate: {recommendation.p95_seconds:.0f}s; "
                f"freshness: {recommendation.freshness}; confidence: {recommendation.confidence}"
            )
        if recommendation.unresolved_timeout_seconds is not None:
            lines.append(f"  Last {recommendation.unresolved_timeout_seconds:.0f}s attempt timed out")
        lines.append(f"  Recommended whole-command timeout: {recommendation.recommended_timeout_seconds}s")
        if recommendation.ceiling_insufficient:
            lines.append("  Warning: ceiling insufficient for the unresolved timeout lower bound")
    return "\n".join(lines)


# Friendly aliases used by integrations and tests.
parse_test_invocation = parse_managed_test_invocation
record_observation = record_test_observation
recommend_test_timeout = recommend_timeout
