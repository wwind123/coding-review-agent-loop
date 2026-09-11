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
import threading
import time
from collections import OrderedDict, defaultdict
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
COMMAND_LANE_LOCK_DIR = "command-lanes"
OVERLAP_REJECTED_EXIT_CODE = 125
OVERLAP_REJECTED_MESSAGE = "agent-loop: identical test command is already running in this invocation lane; wait for it to exit or terminate it explicitly."
MAX_OBSERVATIONS_PER_COHORT = 20
MAX_COHORTS = 200
STALE_AFTER = timedelta(days=30)
LAUNCHER_HEALTH_STALE_AFTER = timedelta(hours=24)
MAX_LAUNCHER_HEALTH_PER_IDENTITY = 8
MAX_LAUNCHER_HEALTH_IDENTITIES = 100
MAX_LAUNCHER_DIAGNOSTIC_CHARS = 240
LAUNCHER_PROBE_TIMEOUT_SECONDS = 5.0
MAX_WRAPPER_PROBE_CANDIDATES = 2
MAX_INNER_PROBE_CANDIDATES = 6
MAX_PREFLIGHT_INVOCATION_BUCKETS = 128
PREFLIGHT_INVOCATION_TTL_SECONDS = 3600.0
_HASHED_ENV_VALUE_RE = re.compile(r"<sha256:[0-9a-f]{16}>")
_DIAGNOSTIC_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|secret|password|token|credential)\s*[:=]\s*[^\s,;]+"
)
_DIAGNOSTIC_ASSIGNMENT_SECRET_RE = re.compile(
    r"(?i)(\b(?:aws[_-](?:access[_-]?key[_-]?id|secret[_-]?access[_-]?key|session[_-]?token)|"
    r"(?:api[_-]?key|secret|password|passwd|token|credential|authorization|private[_-]?key|dsn))\b\s*=\s*)"
    r"[^\s,;&]+"
)
_DIAGNOSTIC_FIELD_SECRET_RE = re.compile(
    r"(?i)(\b(?:[a-z0-9]+[_-])*?(?:api[_-]?key|secret|password|passwd|token|credential|"
    r"authorization|private[_-]?key|dsn)(?:[_-][a-z0-9]+)*\b\s*[:=]\s*)"
    r"[^\s,;&]+"
)
_DIAGNOSTIC_OPTION_SECRET_RE = re.compile(
    r"(?i)(?<![\w-])(--?[^\s=]*(?:api[-_]?key|token|password|passwd|secret|credential|auth)"
    r"[^\s=]*(?:=|\s+))"
    r"[^\s,;&]+"
)
_DIAGNOSTIC_AUTH_HEADER_RE = re.compile(
    r"(?im)(^|\s)((?:proxy-)?authorization\s*:\s*)(?:bearer|basic|token)\s+[^\s,;&]+"
)
_DIAGNOSTIC_URL_USERINFO_RE = re.compile(
    r"(?i)(\b(?:https?|ssh|postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://)[^/@\s]+@"
)

WRAPPER_BOOTSTRAP_STATES = frozenset({"verified", "failed", "unknown"})
INNER_EXEC_STATES = frozenset({"not-attempted", "started", "failed"})
SUITE_START_STATES = frozenset({"not-started", "verified", "unknown"})
LAUNCHER_HEALTH_PROVENANCES = frozenset(
    {"wrapper-probe", "parent-runner", "broker-parent", "agent-reported"}
)


@dataclass(frozen=True)
class LauncherProbeResult:
    """Bounded result of a wrapper or recognized inner-launcher probe."""

    candidate: tuple[str, ...]
    state: str
    diagnostic: str = ""
    identity: str = ""

    def __post_init__(self) -> None:
        if self.state not in WRAPPER_BOOTSTRAP_STATES:
            raise ValueError(f"unknown launcher probe state: {self.state}")


@dataclass(frozen=True)
class LaunchState:
    """Independent wrapper, exec, and suite-boundary state."""

    wrapper_bootstrap: str = "unknown"
    inner_exec: str = "not-attempted"
    suite_start: str = "not-started"
    diagnostic: str = ""
    provenance: str = "parent-runner"

    def __post_init__(self) -> None:
        if self.wrapper_bootstrap not in WRAPPER_BOOTSTRAP_STATES:
            raise ValueError(f"unknown wrapper bootstrap state: {self.wrapper_bootstrap}")
        if self.inner_exec not in INNER_EXEC_STATES:
            raise ValueError(f"unknown inner exec state: {self.inner_exec}")
        if self.suite_start not in SUITE_START_STATES:
            raise ValueError(f"unknown suite start state: {self.suite_start}")


class TestRuntimeConfigurationError(AgentLoopError):
    """A wrapper policy was malformed or exceeded its inherited ceiling."""


class CommandLaneLock:
    """Non-inherited advisory lock for one canonical test command lane."""

    def __init__(self, handle, path: Path, key: str):
        self.handle = handle
        self.path = path
        self.key = key

    @classmethod
    def acquire(
        cls,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
    ) -> "CommandLaneLock | None":
        values = env if env is not None else os.environ
        invocation_id = values.get("AGENT_LOOP_INVOCATION_ID", "standalone")
        normalized = normalize_test_command(argv, cwd=cwd)
        key = f"{cwd.resolve()}\0{normalized}\0{invocation_id}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        root = Path(values.get("XDG_RUNTIME_DIR", "")) / "agent-loop" / COMMAND_LANE_LOCK_DIR
        if str(root) == f"agent-loop/{COMMAND_LANE_LOCK_DIR}":
            root = Path(tempfile.gettempdir()) / "coding-review-agent-loop" / COMMAND_LANE_LOCK_DIR
        try:
            root.mkdir(parents=True, exist_ok=True)
            path = root / f"{digest}.lock"
            handle = path.open("a+")
            if os.name == "nt":
                import msvcrt
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    handle.close()
                    return None
            else:
                import fcntl
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    return None
            # The descriptor is intentionally not inherited by subprocesses.
            os.set_inheritable(handle.fileno(), False)
            return cls(handle, path, key)
        except OSError:
            # A lock failure is not permission to run two potentially expensive
            # copies.  Treat unavailable lock storage as a rejected lane.
            return None

    def close(self) -> None:
        if self.handle.closed:
            return
        try:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            self.handle.close()

    def __enter__(self) -> "CommandLaneLock":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def acquire_command_lane(
    argv: Sequence[str], *, cwd: Path, env: Mapping[str, str] | None = None
) -> CommandLaneLock | None:
    return CommandLaneLock.acquire(argv, cwd=cwd, env=env)


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
    prefix: Sequence[str] | None = None,
) -> str:
    chosen_prefix = tuple(prefix) if prefix is not None else resolve_wrapper_prefix()
    if chosen_prefix is None:
        return shlex.join(str(item) for item in command)
    options: list[str] = []
    if timeout_seconds is not None:
        options += ["--timeout-seconds", str(timeout_seconds)]
    if memory_dir is not None:
        options += ["--memory-dir", str(memory_dir)]
    return shlex.join((*chosen_prefix, *options, "--", *(str(item) for item in command)))


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


def _collapsed_diagnostic(value: object, *, limit: int = MAX_LAUNCHER_DIAGNOSTIC_CHARS) -> str:
    text = " ".join(str(value or "").split())
    text = _DIAGNOSTIC_ASSIGNMENT_SECRET_RE.sub(r"\1<redacted>", text)
    text = _DIAGNOSTIC_OPTION_SECRET_RE.sub(r"\1<redacted>", text)
    text = _DIAGNOSTIC_AUTH_HEADER_RE.sub(r"\1\2<redacted>", text)
    text = _DIAGNOSTIC_URL_USERINFO_RE.sub(r"\1<userinfo:redacted>@", text)
    text = _DIAGNOSTIC_FIELD_SECRET_RE.sub(r"\1<redacted>", text)
    text = _DIAGNOSTIC_SECRET_RE.sub(r"\1=<redacted>", text)
    return text[:limit]


def checkout_identity(cwd: Path) -> str:
    """Hash the canonical assigned checkout without retaining its raw path."""
    canonical = str(cwd.resolve(strict=False))
    return hashlib.sha256(canonical.encode("utf-8", errors="replace")).hexdigest()


def _repository_slug(cwd: Path, repository: str | None = None) -> str:
    if repository and re.fullmatch(r"[^/\s]+/[^/\s]+", repository):
        return repository
    try:
        remote = subprocess.run(
            ("git", "config", "--get", "remote.origin.url"),
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        remote = ""
    match = re.search(r"(?:github\.com[:/])([^/]+/[^/]+?)(?:\.git)?$", remote)
    return match.group(1) if match else "unknown/unknown"


def launcher_environment_fingerprint(
    cwd: Path, *, environment: Mapping[str, str] | None = None
) -> str:
    """Fingerprint only stable runtime identity inputs, never raw environment values."""
    values = environment if environment is not None else os.environ
    payload = {
        "runtime": platform.python_version(),
        "platform": platform.system().lower(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "virtual_env": hashlib.sha256(
            values.get("VIRTUAL_ENV", "").encode("utf-8", errors="replace")
        ).hexdigest(),
        "path": hashlib.sha256(
            values.get("PATH", "").encode("utf-8", errors="replace")
        ).hexdigest(),
        "checkout": checkout_identity(cwd),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _stat_identity(path: Path) -> dict[str, object]:
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path.resolve(strict=False)), "missing": True}
    return {
        "path": str(path.resolve(strict=False)),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
        "mode": stat.st_mode & 0o111,
    }


def _lexical_stat_identity(path: Path) -> dict[str, object]:
    """Stat the directory entry itself so a symlink retarget is observable."""
    lexical = _lexical_absolute(path)
    try:
        stat = lexical.lstat()
    except OSError:
        return {"path": str(lexical), "missing": True}
    return {
        "path": str(lexical),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
        "mode": stat.st_mode & 0o111,
        "symlink": lexical.is_symlink(),
    }


def _shebang_interpreter(path: Path) -> str | None:
    try:
        first = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    except (OSError, IndexError):
        return None
    if not first.startswith("#!"):
        return None
    try:
        parts = shlex.split(first[2:].strip())
    except ValueError:
        return None
    if not parts:
        return None
    if Path(parts[0]).name == "env" and len(parts) > 1:
        if "-S" in parts:
            index = parts.index("-S") + 1
            return parts[index] if index < len(parts) else None
        return parts[1].split("=", 1)[-1] if "=" in parts[1] else parts[1]
    return parts[0]


def _resolve_candidate_path(
    token: str, *, cwd: Path, environment: Mapping[str, str]
) -> Path:
    candidate = Path(token).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    if token.startswith((".", "~")) or candidate.parent != Path("."):
        return (cwd / candidate).resolve(strict=False)
    try:
        resolved = shutil.which(token, path=environment.get("PATH"))
    except TypeError:
        resolved = shutil.which(token)
    return Path(resolved).resolve(strict=False) if resolved else candidate


def _lexical_candidate_path(
    token: str, *, cwd: Path, environment: Mapping[str, str]
) -> Path:
    """Resolve a command token without collapsing symlinks.

    A virtualenv's ``bin/python`` is commonly a symlink.  Its lexical path
    identifies the environment that will supply site-packages, while its
    resolved path identifies the base interpreter.  Both identities are
    needed for safe probing and cache invalidation.
    """
    candidate = Path(token).expanduser()
    if candidate.is_absolute():
        return candidate
    if token.startswith((".", "~")) or candidate.parent != Path("."):
        return cwd / candidate
    try:
        resolved = shutil.which(token, path=environment.get("PATH"))
    except TypeError:
        resolved = shutil.which(token)
    return Path(resolved) if resolved else candidate


def _resolve_shebang_path(
    shebang: str | None, *, cwd: Path, environment: Mapping[str, str]
) -> Path | None:
    if not shebang:
        return None
    return _resolve_candidate_path(shebang, cwd=cwd, environment=environment)


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute path while retaining symlink spelling."""
    return Path(os.path.abspath(os.fspath(path)))


_PYTHON_INTERPRETER_NAME_RE = re.compile(
    r"(?i)(?:python(?:[0-9]+(?:\.[0-9]+)*)?|pypy(?:[0-9]+(?:\.[0-9]+)*)?|py)(?:\.exe)?$"
)


def _python_interpreter_path(
    token: str, *, cwd: Path, environment: Mapping[str, str]
) -> Path | None:
    """Resolve a command token only when it has a recognizable Python identity.

    This is deliberately a filesystem-only check.  It never starts the token
    merely to decide whether it is safe to probe.  In particular, an arbitrary
    shell script named ``python`` is not recognized.  The current interpreter
    may be addressed through any symlink that resolves to it; other candidates
    must be conventionally named, executable Python binaries in a
    ``pyvenv.cfg`` environment.  The bounded ``--version`` probe remains
    authoritative for importability.
    """
    lexical = _lexical_absolute(
        _lexical_candidate_path(token, cwd=cwd, environment=environment)
    )
    try:
        resolved = lexical.resolve(strict=False)
        current = Path(sys.executable).resolve(strict=False)
    except OSError:
        return None
    try:
        same_as_current = resolved == current or os.path.samefile(lexical, current)
    except OSError:
        same_as_current = resolved == current
    if same_as_current:
        return lexical
    if not _PYTHON_INTERPRETER_NAME_RE.fullmatch(lexical.name):
        return None
    if not lexical.is_file() or not os.access(lexical, os.X_OK):
        return None
    environment_root = _lexical_absolute(lexical.parent.parent) if lexical.parent.name.lower() in {"bin", "scripts"} else None
    if environment_root is None or not (environment_root / "pyvenv.cfg").is_file():
        return None
    try:
        with resolved.open("rb") as stream:
            magic = stream.read(4)
    except OSError:
        return None
    if magic not in {b"\x7fELF", b"MZ\x90\x00"}:
        return None
    return lexical


def _interpreter_environment_root(
    interpreter: Path, *, environment: Mapping[str, str]
) -> Path | None:
    """Return a likely environment root without inspecting arbitrary paths."""
    lexical = _lexical_absolute(interpreter)
    if lexical.parent.name.lower() in {"bin", "scripts"}:
        return lexical.parent.parent
    if interpreter.resolve(strict=False) == Path(sys.executable).resolve(strict=False):
        return Path(sys.prefix).resolve(strict=False)
    virtual_env = environment.get("VIRTUAL_ENV")
    if virtual_env:
        return _lexical_absolute(Path(virtual_env).expanduser())
    return None


def _pytest_dependency_paths(
    interpreter: Path, *, environment: Mapping[str, str]
) -> tuple[Path, ...]:
    """Enumerate a small, interpreter-scoped set of pytest dependency paths."""
    root = _interpreter_environment_root(interpreter, environment=environment)
    if root is None:
        return ()
    candidates: list[Path] = []
    # The fixed layouts cover POSIX/Windows virtualenvs and the running
    # interpreter's usual prefix.  Do not recursively search an environment.
    for relative in (
        Path("Lib") / "site-packages",
        Path("lib") / "site-packages",
        Path("lib64") / "site-packages",
    ):
        candidates.append(root / relative)
    lib_dir = root / "lib"
    try:
        versioned_libs = sorted(
            (entry for entry in lib_dir.iterdir() if entry.is_dir() and entry.name.startswith("python")),
            key=lambda entry: entry.name,
        )
    except OSError:
        versioned_libs = []
    candidates.extend(entry / "site-packages" for entry in versioned_libs[:8])
    paths: list[Path] = []
    for site_packages in candidates:
        paths.extend(
            (
                site_packages,
                site_packages / "pytest",
                site_packages / "pytest" / "__init__.py",
                site_packages / "pytest.py",
            )
        )
    return tuple(paths)


def _pytest_dependency_identity(
    interpreter: Path, *, environment: Mapping[str, str]
) -> dict[str, object]:
    """Fingerprint bounded pytest locations for the selected interpreter."""
    paths = _pytest_dependency_paths(interpreter, environment=environment)
    return {
        "interpreter": str(interpreter),
        "paths": [_stat_identity(path) for path in paths],
    }


def launcher_candidate_identity(
    candidate: Sequence[str], *, cwd: Path, environment: Mapping[str, str] | None = None,
    kind: str = "inner",
) -> dict[str, object]:
    """Return a cache/persistence identity with only the managed path verbatim."""
    values = environment if environment is not None else os.environ
    tokens = tuple(str(item) for item in candidate)
    resolved_path = _resolve_candidate_path(tokens[0], cwd=cwd, environment=values) if tokens else Path("")
    lexical_path = _lexical_absolute(
        _lexical_candidate_path(tokens[0], cwd=cwd, environment=values)
    ) if tokens else Path("")
    # Keep managed wrapper paths canonical, but retain the configured lexical
    # path for inner launchers so virtualenv symlinks remain distinguishable.
    path = resolved_path if kind == "wrapper" else lexical_path
    virtual_env = Path(values["VIRTUAL_ENV"]).resolve(strict=False) if values.get("VIRTUAL_ENV") else None
    shebang = _shebang_interpreter(path) if path.is_file() else None
    shebang_path = (
        _resolve_shebang_path(shebang, cwd=cwd, environment=values)
        if kind == "wrapper"
        else (
            _lexical_absolute(_lexical_candidate_path(shebang, cwd=cwd, environment=values))
            if shebang
            else None
        )
    )
    interpreter_path = shebang_path or path
    package_origin = Path(__file__).resolve(strict=False)
    if kind == "inner":
        # Never ask the agent-loop interpreter where pytest lives: that can be
        # a different environment from ``<python> -m pytest``.  The bounded
        # layout identity below is derived from the selected interpreter.
        module_origin = None
        dependency_identity = _pytest_dependency_identity(interpreter_path, environment=values)
    else:
        module_origin = None
        dependency_identity = None
    prefix = (
        tokens[:3]
        if len(tokens) >= 3 and tokens[1:3] == ("-m", "pytest")
        else tokens[:1]
    )
    identity: dict[str, object] = {
        "kind": kind,
        "path": str(path),
        "entry": _stat_identity(path),
        "entry_lexical": _lexical_stat_identity(lexical_path),
        "shebang": shebang,
        "shebang_interpreter": _stat_identity(shebang_path) if shebang_path is not None else None,
        "shebang_interpreter_lexical": (
            _lexical_stat_identity(shebang_path) if shebang_path is not None else None
        ),
        "interpreter": _stat_identity(interpreter_path),
        "interpreter_lexical": _lexical_stat_identity(interpreter_path),
        "package_origin": _stat_identity(package_origin),
        "module_origin": _stat_identity(module_origin) if module_origin is not None else None,
        "pytest_dependency": dependency_identity,
        "virtual_env": _stat_identity(virtual_env) if virtual_env is not None else None,
        "path_environment_sha256": hashlib.sha256(
            values.get("PATH", "").encode("utf-8", errors="replace")
        ).hexdigest(),
        "candidate_prefix": list(prefix),
        "cwd": checkout_identity(cwd),
    }
    return identity


def _identity_key(identity: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()


def _redacted_identity(identity: Mapping[str, object], *, cwd: Path) -> dict[str, object]:
    """Keep the exact managed wrapper path, but redact all other external paths."""
    result: dict[str, object] = {}
    for key, value in identity.items():
        if key == "path":
            result[key] = str(value) if identity.get("kind") == "wrapper" else _relative_or_basename(str(value), cwd)
        elif key == "shebang" and isinstance(value, str):
            result[key] = _relative_or_basename(value, cwd)
        elif isinstance(value, Mapping):
            nested = dict(value)
            nested_path = nested.get("path")
            if isinstance(nested_path, str):
                keep_wrapper_path = (
                    identity.get("kind") == "wrapper"
                    and nested_path == str(identity.get("path"))
                )
                if not keep_wrapper_path:
                    nested["path"] = _relative_or_basename(nested_path, cwd)
            if key == "pytest_dependency":
                interpreter = nested.get("interpreter")
                if isinstance(interpreter, str):
                    nested["interpreter"] = _relative_or_basename(interpreter, cwd)
                paths = nested.get("paths")
                if isinstance(paths, list):
                    nested["paths"] = [
                        {
                            **dict(item),
                            "path": _relative_or_basename(str(item["path"]), cwd),
                        }
                        if isinstance(item, Mapping) and isinstance(item.get("path"), str)
                        else item
                        for item in paths
                    ]
            result[key] = nested
        elif isinstance(value, (list, tuple)):
            result[key] = [
                (
                    str(item)
                    if identity.get("kind") == "wrapper" and key == "candidate_prefix" and index == 0
                    else _relative_or_basename(str(item), cwd)
                    if isinstance(item, str) and Path(item).is_absolute()
                    else item
                )
                for index, item in enumerate(value)
            ]
        else:
            result[key] = value
    return result


def _health_scope(row: Mapping[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row.get("repository", "")),
        str(row.get("checkout_sha256", "")),
        str(row.get("environment_fingerprint", "")),
        str(row.get("candidate_key", "")),
    )


def _sidecar_payload(memory_dir: Path) -> dict | None:
    path = memory_dir / RUNTIME_SIDECAR_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"schema_version": RUNTIME_SCHEMA_VERSION, "observations": [], "launcher_health": []}
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != RUNTIME_SCHEMA_VERSION:
        return None
    observations = payload.get("observations")
    if not isinstance(observations, list):
        return None
    health = payload.get("launcher_health", [])
    if not isinstance(health, list):
        health = []
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "observations": [item for item in observations if isinstance(item, dict)],
        "launcher_health": [item for item in health if isinstance(item, dict)],
    }


def load_runtime_memory(memory_dir: Path) -> list[dict]:
    payload = _sidecar_payload(memory_dir)
    return list(payload["observations"]) if payload is not None else []


def load_launcher_health(memory_dir: Path | None) -> list[dict]:
    """Load health independently; malformed health rows never hide timing rows."""
    if memory_dir is None:
        return []
    payload = _sidecar_payload(memory_dir)
    if payload is None:
        return []
    rows: list[dict] = []
    for row in payload.get("launcher_health", []):
        if not isinstance(row, dict):
            continue
        if (
            isinstance(row.get("repository"), str)
            and bool(row.get("repository"))
            and isinstance(row.get("checkout_sha256"), str)
            and bool(row.get("checkout_sha256"))
            and isinstance(row.get("environment_fingerprint"), str)
            and bool(row.get("environment_fingerprint"))
            and isinstance(row.get("candidate_key"), str)
            and bool(row.get("candidate_key"))
            and isinstance(row.get("candidate_identity"), Mapping)
            and row.get("state") in WRAPPER_BOOTSTRAP_STATES
            and row.get("provenance") in LAUNCHER_HEALTH_PROVENANCES
            and _timestamp(row.get("timestamp")) is not None
            and isinstance(row.get("diagnostic", ""), str)
        ):
            safe = dict(row)
            safe["diagnostic"] = _collapsed_diagnostic(safe.get("diagnostic", ""))
            rows.append(safe)
    return rows


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
    containment: Mapping[str, object] | None = None,
    lane: str | None = None,
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
            if lane is not None:
                observation["lane"] = lane
            if containment is not None:
                # Evidence is already bounded by the runner.  Keep only JSON
                # values and expose unsupported telemetry explicitly.
                observation["containment"] = dict(containment)
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


def record_launcher_health(
    memory_dir: Path | None,
    *,
    cwd: Path,
    candidate: Sequence[str] | Mapping[str, object],
    state: str,
    provenance: str,
    repository: str | None = None,
    environment: Mapping[str, str] | None = None,
    diagnostic: object = "",
    timestamp: datetime | None = None,
) -> bool:
    """Persist bounded launcher health separately from suite timing rows."""
    if memory_dir is None or state not in WRAPPER_BOOTSTRAP_STATES or provenance not in LAUNCHER_HEALTH_PROVENANCES:
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
            if isinstance(candidate, Mapping):
                raw_identity = dict(candidate)
            else:
                raw_identity = launcher_candidate_identity(candidate, cwd=cwd, environment=environment)
            identity = _redacted_identity(raw_identity, cwd=cwd)
            key = _identity_key(raw_identity)
            now = (timestamp or _utc_now()).astimezone(timezone.utc)
            repository_name = _repository_slug(cwd, repository)
            scope = (repository_name, checkout_identity(cwd), launcher_environment_fingerprint(cwd, environment=environment), key)
            rows = []
            cutoff = now - LAUNCHER_HEALTH_STALE_AFTER
            for row in payload.get("launcher_health", []):
                if not isinstance(row, dict):
                    continue
                stamp = _timestamp(row.get("timestamp"))
                if stamp is None:
                    continue
                if _health_scope(row) != scope:
                    rows.append(row)
                elif stamp >= cutoff:
                    rows.append(row)
            if state == "verified":
                rows = [
                    row for row in rows
                    if not (_health_scope(row) == scope and row.get("state") == "failed")
                ]
            row = {
                "repository": repository_name,
                "checkout_sha256": scope[1],
                "environment_fingerprint": scope[2],
                "candidate_key": key,
                "candidate_identity": identity,
                "state": state,
                "timestamp": now.isoformat(),
                "provenance": provenance,
                "diagnostic": _collapsed_diagnostic(diagnostic),
            }
            duplicate_index = next(
                (
                    index
                    for index, existing in enumerate(rows)
                    if _health_scope(existing) == scope
                    and existing.get("state") == row["state"]
                    and existing.get("diagnostic") == row["diagnostic"]
                ),
                None,
            )
            if duplicate_index is None:
                rows.append(row)
            else:
                # A repeated failure is still live evidence. Refresh its
                # timestamp so a continuously broken launcher does not
                # disappear merely because it crossed the 24-hour window.
                rows[duplicate_index] = row
            groups: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
            for existing in rows:
                groups[_health_scope(existing)].append(existing)
            ordered_scopes = sorted(
                groups,
                key=lambda item: max((_timestamp(item_row.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc) for item_row in groups[item]), default=datetime.min.replace(tzinfo=timezone.utc)),
                reverse=True,
            )[:MAX_LAUNCHER_HEALTH_IDENTITIES]
            kept: list[dict] = []
            for group_key in ordered_scopes:
                group = groups[group_key]
                group.sort(key=lambda item: _timestamp(item.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
                kept.extend(group[:MAX_LAUNCHER_HEALTH_PER_IDENTITY])
            payload["launcher_health"] = kept
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


def relevant_launcher_health(
    memory_dir: Path | None,
    *,
    cwd: Path,
    candidate: Sequence[str] | Mapping[str, object] | None = None,
    repository: str | None = None,
    environment: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> list[dict]:
    """Return fresh rows for exactly this repository/checkout/environment."""
    if memory_dir is None:
        return []
    values = environment if environment is not None else os.environ
    key = ""
    if candidate is not None:
        raw = dict(candidate) if isinstance(candidate, Mapping) else launcher_candidate_identity(candidate, cwd=cwd, environment=values)
        key = _identity_key(raw)
    scope = (_repository_slug(cwd, repository), checkout_identity(cwd), launcher_environment_fingerprint(cwd, environment=values), key)
    cutoff = (now or _utc_now()).astimezone(timezone.utc) - LAUNCHER_HEALTH_STALE_AFTER
    rows = []
    for row in load_launcher_health(memory_dir):
        if _health_scope(row)[:3] != scope[:3] or (key and row.get("candidate_key") != key):
            continue
        stamp = _timestamp(row.get("timestamp"))
        if stamp is not None and stamp >= cutoff:
            rows.append(row)
    return rows


def _wrapper_candidates(environment: Mapping[str, str] | None = None) -> list[tuple[str, ...]]:
    values = environment if environment is not None else os.environ
    candidates: list[tuple[str, ...]] = []
    entry = shutil.which("agent-loop", path=values.get("PATH"))
    if entry and os.path.isabs(entry) and os.access(entry, os.X_OK):
        candidates.append((str(Path(entry).resolve()), "run-tests"))
    executable = Path(sys.executable).resolve()
    if executable.is_file() and os.access(executable, os.X_OK):
        fallback = (str(executable), "-m", "coding_review_agent_loop.cli", "run-tests")
        if fallback not in candidates:
            candidates.append(fallback)
    return candidates[:MAX_WRAPPER_PROBE_CANDIDATES]


_WRAPPER_PREFLIGHT_CACHE: dict[tuple[str, str], LauncherProbeResult] = {}
_INNER_PREFLIGHT_CACHE: dict[tuple[str, str], LauncherProbeResult] = {}
_INNER_PREFLIGHT_CANDIDATES: dict[str, set[str]] = defaultdict(set)
_WRAPPER_PREFLIGHT_INFLIGHT: dict[tuple[str, str], threading.Event] = {}
_INNER_PREFLIGHT_INFLIGHT: dict[tuple[str, str], threading.Event] = {}
_INNER_PREFLIGHT_LOCK = threading.Lock()
_PREFLIGHT_INVOCATIONS: OrderedDict[str, float] = OrderedDict()


def _invocation_is_active_locked(invocation: str) -> bool:
    return any(key[0] == invocation for key in (*_WRAPPER_PREFLIGHT_INFLIGHT, *_INNER_PREFLIGHT_INFLIGHT))


def _drop_invocation_locked(invocation: str) -> None:
    _PREFLIGHT_INVOCATIONS.pop(invocation, None)
    _INNER_PREFLIGHT_CANDIDATES.pop(invocation, None)
    for cache in (_WRAPPER_PREFLIGHT_CACHE, _INNER_PREFLIGHT_CACHE):
        for key in tuple(cache):
            if key[0] == invocation:
                del cache[key]


def _prune_invocations_locked(*, protected: set[str] = frozenset()) -> None:
    now = time.monotonic()
    for invocation, last_used in tuple(_PREFLIGHT_INVOCATIONS.items()):
        if invocation in protected or _invocation_is_active_locked(invocation):
            continue
        if (
            now - last_used > PREFLIGHT_INVOCATION_TTL_SECONDS
            or len(_PREFLIGHT_INVOCATIONS) > MAX_PREFLIGHT_INVOCATION_BUCKETS
        ):
            _drop_invocation_locked(invocation)


def _touch_invocation_locked(invocation: str) -> None:
    _PREFLIGHT_INVOCATIONS[invocation] = time.monotonic()
    _PREFLIGHT_INVOCATIONS.move_to_end(invocation)
    _prune_invocations_locked(protected={invocation})


def preflight_wrapper_candidates(
    *,
    cwd: Path | None = None,
    memory_dir: Path | None = None,
    repository: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> tuple[LauncherProbeResult, ...]:
    """Probe at most two safe wrapper prefixes once per invocation/candidate."""
    root = (cwd or Path.cwd()).resolve()
    values = {**os.environ, **environment} if environment is not None else dict(os.environ)
    invocation = values.get("AGENT_LOOP_INVOCATION_ID")
    results: list[LauncherProbeResult] = []
    for candidate in _wrapper_candidates(values):
        identity = launcher_candidate_identity(candidate, cwd=root, environment=values, kind="wrapper")
        cache_key = (invocation, _identity_key(identity)) if invocation else None
        flight: threading.Event | None = None
        owner = True
        result: LauncherProbeResult | None = None
        if cache_key is not None:
            with _INNER_PREFLIGHT_LOCK:
                _touch_invocation_locked(invocation)
                result = _WRAPPER_PREFLIGHT_CACHE.get(cache_key)
                if result is None:
                    flight = _WRAPPER_PREFLIGHT_INFLIGHT.get(cache_key)
                    if flight is not None:
                        owner = False
                    else:
                        flight = threading.Event()
                        _WRAPPER_PREFLIGHT_INFLIGHT[cache_key] = flight
        if not owner:
            assert flight is not None
            if flight.wait(LAUNCHER_PROBE_TIMEOUT_SECONDS + 1.0):
                with _INNER_PREFLIGHT_LOCK:
                    result = _WRAPPER_PREFLIGHT_CACHE.get(cache_key)  # type: ignore[arg-type]
            if result is None:
                result = LauncherProbeResult(
                    candidate,
                    "unknown",
                    "wrapper probe result was not published by its owner",
                    _identity_key(identity),
                )
        elif result is None:
            probe_argv = [*candidate, "--preflight"]
            try:
                completed = subprocess.run(
                    probe_argv,
                    cwd=root,
                    env=values,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=LAUNCHER_PROBE_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired:
                result = LauncherProbeResult(candidate, "failed", "wrapper probe timed out after 5s", _identity_key(identity))
            except OSError as exc:
                result = LauncherProbeResult(candidate, "failed", f"wrapper did not start: {type(exc).__name__}", _identity_key(identity))
            else:
                output = _collapsed_diagnostic((completed.stdout or "") + " " + (completed.stderr or ""))
                success = completed.returncode == 0 and "agent-loop preflight: verified" in output
                explicit_bootstrap = any(token in output.lower() for token in ("modulenotfounderror", "importerror", "no module named", "cannot import"))
                state = "verified" if success else ("failed" if explicit_bootstrap else "unknown")
                result = LauncherProbeResult(candidate, state, output, _identity_key(identity))
            if cache_key is not None:
                assert flight is not None
                with _INNER_PREFLIGHT_LOCK:
                    _WRAPPER_PREFLIGHT_CACHE[cache_key] = result
                    _WRAPPER_PREFLIGHT_INFLIGHT.pop(cache_key, None)
                    _PREFLIGHT_INVOCATIONS[invocation] = time.monotonic()
                    _PREFLIGHT_INVOCATIONS.move_to_end(invocation)
                    flight.set()
                    _prune_invocations_locked()
            if memory_dir is not None and result.state in {"failed", "verified"}:
                record_launcher_health(
                    memory_dir,
                    cwd=root,
                    candidate=identity,
                    state=result.state,
                    provenance="wrapper-probe",
                    repository=repository,
                    environment=values,
                    diagnostic=result.diagnostic,
                )
        results.append(result)
        if result.state == "verified":
            break
    return tuple(results)


def verified_wrapper_prefix(**kwargs: object) -> tuple[str, ...] | None:
    for result in preflight_wrapper_candidates(**kwargs):
        if result.state == "verified":
            return result.candidate
    return None


def recognized_inner_probe(argv: Sequence[str], *, cwd: Path, environment: Mapping[str, str] | None = None) -> tuple[str, ...] | None:
    """Return the only inner launcher forms eligible for a safe bootstrap probe."""
    tokens = tuple(str(item) for item in argv)
    if not tokens:
        return None
    values = environment if environment is not None else os.environ
    first = Path(tokens[0]).name
    if first in {"pytest", "py.test"}:
        executable = tokens[0]
        if not Path(executable).is_absolute() and (tokens[0].startswith((".", "~")) or Path(tokens[0]).parent != Path(".")):
            executable = str((cwd / Path(tokens[0])).resolve(strict=False))
        elif not Path(executable).is_absolute():
            executable = shutil.which(executable, path=(environment or os.environ).get("PATH")) or executable
        return (executable, "--version")
    if len(tokens) >= 3 and tokens[1] == "-m" and tokens[2] == "pytest":
        interpreter = _python_interpreter_path(tokens[0], cwd=cwd, environment=values)
        if interpreter is not None:
            # Preserve the operator's current-interpreter spelling (including
            # a symlink such as ``python3``) while still using the resolved
            # identity for all other interpreters.
            current = Path(sys.executable).resolve(strict=False)
            executable = tokens[0] if interpreter == current else str(interpreter)
            return (executable, "-m", "pytest", "--version")
    return None


def probe_inner_launcher(
    argv: Sequence[str], *, cwd: Path, environment: Mapping[str, str] | None = None,
    environment_is_complete: bool = False,
) -> LauncherProbeResult:
    """Run only a recognized, identity-cached five-second bootstrap probe."""
    values = (
        dict(environment)
        if environment is not None and environment_is_complete
        else ({**os.environ, **environment} if environment is not None else dict(os.environ))
    )
    original = tuple(str(item) for item in argv)
    probe = recognized_inner_probe(original, cwd=cwd, environment=values)
    if probe is None:
        return LauncherProbeResult(original, "unknown", "unrecognized inner launcher")
    identity = launcher_candidate_identity(original, cwd=cwd, environment=values, kind="inner")
    identity_key = _identity_key(identity)
    invocation = values.get("AGENT_LOOP_INVOCATION_ID")
    cache_key = (invocation, identity_key) if invocation else None
    flight: threading.Event | None = None
    owner = True
    if cache_key is not None:
        with _INNER_PREFLIGHT_LOCK:
            assert invocation is not None
            _touch_invocation_locked(invocation)
            cached = _INNER_PREFLIGHT_CACHE.get(cache_key)
            if cached is not None:
                return cached
            flight = _INNER_PREFLIGHT_INFLIGHT.get(cache_key)
            if flight is not None:
                owner = False
            else:
                seen = _INNER_PREFLIGHT_CANDIDATES.setdefault(invocation, set())
                if identity_key not in seen and len(seen) >= MAX_INNER_PROBE_CANDIDATES:
                    return LauncherProbeResult(
                        original,
                        "unknown",
                        f"inner probe candidate limit reached ({MAX_INNER_PROBE_CANDIDATES})",
                        identity_key,
                    )
                seen.add(identity_key)
                flight = threading.Event()
                _INNER_PREFLIGHT_INFLIGHT[cache_key] = flight
    if not owner:
        assert flight is not None
        # The owner has a five-second subprocess watchdog.  A small amount of
        # headroom lets waiters receive its published result without ever
        # starting a duplicate probe if the owner is slow to publish.
        if flight.wait(LAUNCHER_PROBE_TIMEOUT_SECONDS + 1.0):
            with _INNER_PREFLIGHT_LOCK:
                cached = _INNER_PREFLIGHT_CACHE.get(cache_key)  # type: ignore[arg-type]
            if cached is not None:
                return cached
        return LauncherProbeResult(
            original,
            "unknown",
            "inner probe result was not published by its owner",
            identity_key,
        )
    result: LauncherProbeResult | None = None
    try:
        try:
            completed = subprocess.run(
                probe,
                cwd=cwd,
                env=(values if environment_is_complete or environment is not None else None),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=LAUNCHER_PROBE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            result = LauncherProbeResult(original, "failed", "inner bootstrap probe timed out after 5s", identity_key)
        except OSError as exc:
            result = LauncherProbeResult(original, "failed", f"inner launcher did not start: {type(exc).__name__}", identity_key)
        else:
            output = _collapsed_diagnostic((completed.stdout or "") + " " + (completed.stderr or ""))
            if completed.returncode == 0:
                result = LauncherProbeResult(original, "verified", output, identity_key)
            else:
                result = LauncherProbeResult(original, "failed", output or f"bootstrap exited {completed.returncode}", identity_key)
        return result
    finally:
        if cache_key is not None:
            assert flight is not None
            with _INNER_PREFLIGHT_LOCK:
                if result is not None:
                    _INNER_PREFLIGHT_CACHE[cache_key] = result
                _INNER_PREFLIGHT_INFLIGHT.pop(cache_key, None)
                assert invocation is not None
                _PREFLIGHT_INVOCATIONS[invocation] = time.monotonic()
                _PREFLIGHT_INVOCATIONS.move_to_end(invocation)
                flight.set()
                _prune_invocations_locked()


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
    # MemoryHigh/PSI pressure can make a successful run unusually slow.  Keep
    # the gate operationally successful but do not let that sample inflate the
    # learned timeout recommendation.
    successes = [
        row for row in ordered
        if row.get("outcome") == "passed"
        and not bool(isinstance(row.get("containment"), dict) and row["containment"].get("pressure"))
        and isinstance(row.get("elapsed_seconds"), (int, float))
    ]
    success_values = [float(row["elapsed_seconds"]) for row in successes]
    latest_success = next(
        (
            float(row["elapsed_seconds"])
            for row in reversed(ordered)
            if row.get("outcome") == "passed"
            and not bool(isinstance(row.get("containment"), dict) and row["containment"].get("pressure"))
            and isinstance(row.get("elapsed_seconds"), (int, float))
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
