"""Parent-side support for the pytest lifecycle probe (issue #1182).

A direct-pytest run that exits before any test could run (an argument/usage
error, or an interpreter without pytest) produces no evidence about the
change.  Recognising that deterministically needs positive, parent-observed
provenance rather than output text, so the parent:

* injects the stdlib-only ``_agent_loop_lifecycle_probe`` plugin ahead of every
  later plugin, but only into launchers it has validated as a native Python
  interpreter (``LaunchEligibility``);
* reads the plugin's private report back and parses it strictly;
* scans the whole output stream for collection/result markers (a veto only);
* for an interpreter without pytest, runs an independent importability check.

Everything here fails closed: a missing, partial, mismatched or unreadable
signal withdraws provenance and leaves the run an authoritative result.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .test_workers import (
    _PYTHON_FLAG_TOKENS,
    _is_python_name,
    _merge_path_value,
    classify_command,
    plugin_directory,
)

PROBE_MODULE = "_agent_loop_lifecycle_probe"
ENV_PROBE_SPEC = "AGENT_LOOP_LIFECYCLE_PROBE_SPEC"
PROBE_DISABLE_TOKENS = frozenset({f"no:{PROBE_MODULE}"})

PREFLIGHT_TIMEOUT_SECONDS = 5.0
IMPORT_CHECK_TIMEOUT_SECONDS = 10.0
SCAN_MAX_LINES = 60
MAX_PROBE_RECORDS = 16
MAX_PROBE_REPORT_BYTES = 16 * 1024

_PREFLIGHT_CODE = (
    "import importlib.util,sys;"
    f"sys.exit(0 if importlib.util.find_spec('{PROBE_MODULE}') else 3)"
)
_IMPORT_CHECK_CODE = (
    "import importlib.util,sys;"
    "sys.exit(3 if importlib.util.find_spec('pytest') is None else 0)"
)

IdentityTuple = tuple[int, int, int, int]


@dataclass(frozen=True)
class LaunchEligibility:
    """Parent-validated proof that the probe was injected into a native interpreter."""

    shape: str  # "module" (<python> [flags] -m pytest) | "console-script"
    invocation: tuple[str, ...]  # interpreter plus flags/shebang arguments, as invoked
    interpreter_path: str  # realpath of the validated interpreter
    identity: IdentityTuple  # (st_dev, st_ino, st_size, st_mtime_ns)


@dataclass(frozen=True)
class ProbeSetup:
    argv: tuple[str, ...]
    env: dict[str, str]
    report_path: Path
    report_dir: Path
    nonce: str
    eligibility: LaunchEligibility

    def cleanup(self) -> None:
        shutil.rmtree(self.report_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Launcher validation and injection
# ---------------------------------------------------------------------------

_CONSOLE_ARGV0_RE_SUB = "sys.argv[0] = re.sub(r'(-script\\.pyw|\\.exe)?$', '', sys.argv[0])"
_CONSOLE_ARGV0_SUFFIX = "sys.argv[0] = sys.argv[0].removesuffix('.exe')"
_CODING_LINE = "# -*- coding: utf-8 -*-"
_ENV_LAUNCHERS = frozenset({"/usr/bin/env", "/bin/env"})


def _console_script_bodies() -> frozenset[tuple[str, ...]]:
    """Every normalized body (after the shebang) of a generated pytest console script.

    The match is against whole bodies built from literal lines, never per-line
    patterns, so an expression smuggled into an otherwise standard line cannot
    pass (issue #1182).
    """
    bodies: set[tuple[str, ...]] = set()
    for entry in ("console_main", "main"):
        for guard in (True, False):
            for argv0 in (_CONSOLE_ARGV0_RE_SUB, _CONSOLE_ARGV0_SUFFIX):
                imports = ["import re", "import sys"] if argv0 == _CONSOLE_ARGV0_RE_SUB else ["import sys"]
                core = [*imports, f"from pytest import {entry}"]
                if guard:
                    core.append("if __name__ == '__main__':")
                core.extend([argv0, f"sys.exit({entry}())"])
                for coding in (False, True):
                    bodies.add(tuple(([_CODING_LINE] if coding else []) + core))
    return frozenset(bodies)


_CONSOLE_BODIES = _console_script_bodies()


def _resolve_executable(token: str, env: Mapping[str, str], cwd: Path) -> str | None:
    if os.sep in token:
        path = token if os.path.isabs(token) else os.path.join(str(cwd), token)
        return path if os.path.isfile(path) else None
    return shutil.which(token, path=env.get("PATH", os.defpath))


def _console_script_interpreter(
    script: str, env: Mapping[str, str], cwd: Path,
) -> tuple[tuple[str, ...], str] | None:
    """Return (invocation, interpreter) for a standard generated console script."""
    try:
        with open(script, "rb") as handle:
            raw = handle.read(4097)
    except OSError:
        return None
    if len(raw) > 4096:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    lines = text.splitlines()
    if not lines or not lines[0].startswith("#!"):
        return None
    body = tuple(line.strip().replace('"__main__"', "'__main__'") for line in lines[1:] if line.strip())
    if body not in _CONSOLE_BODIES:
        return None
    parts = lines[0][2:].strip().split()
    if not parts:
        return None
    if os.path.basename(parts[0]) == "env":
        # Only the system env launcher, and only as a native executable; any
        # other ``env`` (a trampoline that could add -I) means no injection.
        # Any env option (including -S) also means no injection.
        if parts[0] not in _ENV_LAUNCHERS or _interpreter_is_script(parts[0]):
            return None
        if len(parts) != 2 or parts[1].startswith("-") or "=" in parts[1]:
            return None
        name = parts[1]
        resolved = _resolve_executable(name, env, cwd)
        if resolved is None:
            return None
        return (name,), resolved
    flags = tuple(parts[1:])
    resolved = _resolve_executable(parts[0], env, cwd)
    if resolved is None:
        return None
    return (parts[0], *flags), resolved


def _interpreter_is_script(path: str) -> bool:
    try:
        with open(os.path.realpath(path), "rb") as handle:
            return handle.read(2) == b"#!"
    except OSError:
        return True


def _interpreter_identity(path: str) -> tuple[str, IdentityTuple] | None:
    """Validate a native, non-script, Python-named, executable file."""
    try:
        real = os.path.realpath(path)
        if not _is_python_name(os.path.basename(real)):
            return None
        if not os.access(real, os.R_OK | os.X_OK):
            return None
        with open(real, "rb") as handle:
            if handle.read(2) == b"#!":
                return None
        stat = os.stat(real)
    except OSError:
        return None
    return real, (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _flags_allowed(flags: Sequence[str]) -> bool:
    allowed = _PYTHON_FLAG_TOKENS - {"-E", "-I"}
    return all(flag in allowed for flag in flags)


def _launch_eligibility(
    argv: Sequence[str], env: Mapping[str, str], cwd: Path,
) -> tuple[LaunchEligibility, int] | None:
    """Validate the launcher; return the eligibility and the insertion index."""
    tokens = [str(item) for item in argv]
    shape = classify_command(tokens)
    if shape.command_class != "direct-pytest" or shape.head_index != 0 or shape.pytest_args_start is None:
        return None
    start = shape.pytest_args_start
    name = os.path.basename(tokens[0])
    if name in {"pytest", "py.test"}:
        script = _resolve_executable(tokens[0], env, cwd)
        if script is None:
            return None
        found = _console_script_interpreter(script, env, cwd)
        if found is None:
            return None
        invocation, interpreter = found
        if not _flags_allowed(invocation[1:]):
            return None
        kind = "console-script"
    else:
        module_at = start - 2 if tokens[start - 2:start - 1] == ["-m"] else start - 1
        invocation = tuple(tokens[:module_at])
        if not _flags_allowed(invocation[1:]):
            return None
        interpreter = _resolve_executable(tokens[0], env, cwd)
        if interpreter is None:
            return None
        kind = "module"
    validated = _interpreter_identity(interpreter)
    if validated is None:
        return None
    real, identity = validated
    return LaunchEligibility(kind, tuple(invocation), real, identity), start


def _child_env(env: Mapping[str, str]) -> dict[str, str]:
    child = {str(k): str(v) for k, v in env.items()}
    child.pop(ENV_PROBE_SPEC, None)
    child["PYTHONPATH"] = _merge_path_value(
        child.get("PYTHONPATH"), str(plugin_directory()), separator=os.pathsep,
    )
    return child


def _run_check(
    invocation: Sequence[str], code: str, env: Mapping[str, str], cwd: Path, timeout: float,
) -> int | None:
    try:
        completed = subprocess.run(
            [*invocation, "-c", code],
            cwd=str(cwd), env=dict(env), stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return completed.returncode


def prepare_lifecycle_probe(
    argv: Sequence[str], env: Mapping[str, str] | None, cwd: Path,
) -> ProbeSetup | None:
    """Return the injected argv/env and report, or ``None`` for no injection.

    ``None`` leaves argv and environment byte-identical to the caller's.  Every
    failure (an unsupported launcher, a failed preflight, a report that cannot
    be created) means no injection.
    """
    base = dict(os.environ) if env is None else {str(k): str(v) for k, v in env.items()}
    try:
        eligible = _launch_eligibility(argv, base, cwd)
    except Exception:
        return None
    if eligible is None:
        return None
    eligibility, start = eligible
    child = _child_env(base)
    # Positive preflight through the same execution path the target will use:
    # a wrapper that drops PYTHONPATH (for example by adding -I) fails it.
    if _run_check(eligibility.invocation, _PREFLIGHT_CODE, child, cwd, PREFLIGHT_TIMEOUT_SECONDS) != 0:
        return None
    report_dir: Path | None = None
    try:
        report_dir = Path(tempfile.mkdtemp(prefix="agent-loop-lifecycle-probe-"))
        os.chmod(report_dir, 0o700)
        report = report_dir / "report.jsonl"
        os.close(os.open(report, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    except OSError:
        if report_dir is not None:
            shutil.rmtree(report_dir, ignore_errors=True)
        return None
    nonce = uuid.uuid4().hex
    child[ENV_PROBE_SPEC] = json.dumps({"report": str(report), "nonce": nonce}, sort_keys=True)
    tokens = [str(item) for item in argv]
    injected = tokens[:start] + ["-p", PROBE_MODULE] + tokens[start:]
    return ProbeSetup(tuple(injected), child, report, report_dir, nonce, eligibility)


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeReport:
    records: tuple[Mapping[str, object], ...]

    def of_kind(self, kind: str) -> list[Mapping[str, object]]:
        return [record for record in self.records if record.get("kind") == kind]


def read_probe_report(path: Path | None, nonce: str | None) -> ProbeReport | None:
    """Parse the report strictly; ``None`` for anything missing or corrupt.

    An existing, readable, empty report is an empty ``ProbeReport`` and is
    distinct from ``None``.
    """
    if path is None or not nonce:
        return None
    try:
        with open(path, "rb") as handle:
            raw = handle.read(MAX_PROBE_REPORT_BYTES + 1)
    except OSError:
        return None
    if len(raw) > MAX_PROBE_REPORT_BYTES:
        return None
    if not raw:
        return ProbeReport(())
    if not raw.endswith(b"\n"):
        return None
    records: list[Mapping[str, object]] = []
    pids: set[object] = set()
    try:
        for line in raw.decode("utf-8").splitlines():
            value = json.loads(line)
            if not isinstance(value, dict) or value.get("nonce") != nonce:
                return None
            if not isinstance(value.get("kind"), str):
                return None
            pids.add(value.get("pid"))
            records.append(value)
    except (ValueError, UnicodeDecodeError):
        return None
    if len(records) > MAX_PROBE_RECORDS or len(pids) != 1 or not isinstance(next(iter(pids)), int):
        return None
    return ProbeReport(tuple(records))


# ---------------------------------------------------------------------------
# Whole-stream scan (a veto only)
# ---------------------------------------------------------------------------

_COLLECTION_MARKERS = (
    re.compile(r"^=+ test session starts =+"),
    re.compile(r"^collected \d+ items?"),
    re.compile(r"^collecting \.\.\."),
    re.compile(r"^(platform|rootdir:|plugins:|cachedir:) "),
    re.compile(r"^=+ .*\b(passed|failed|error|errors|skipped|no tests ran|warnings?)\b.* in [\d.]+s"),
    re.compile(r"^(FAILED|ERROR|PASSED|XFAIL|XPASS|SKIPPED) \S"),
    re.compile(r"^!+ .* !+$"),
    re.compile(r"^=+ (FAILURES|ERRORS|short test summary info|warnings summary) =+"),
    re.compile(r"^\S+\.py [.FEsxX]+"),
    re.compile(r"^[.FEsxX]+ +\[ *\d+%\]"),
)
_USAGE_ERROR = re.compile(r"error: (unrecognized arguments|argument |the following arguments are required)")
_USAGE_LINE = re.compile(r"\busage: ")
_NO_PYTEST = re.compile(r"No module named '?pytest'?\s*$")


class PreCollectionScan:
    """Stateful whole-stream scan; markers only ever veto classification."""

    def __init__(self) -> None:
        self.lines = 0
        self.collection_marker_seen = False
        self.usage_error = False
        self.usage_line = False
        self.no_module_pytest = False

    def feed(self, line: str) -> None:
        text = line.rstrip("\r\n")
        self.lines += 1
        if any(marker.search(text) for marker in _COLLECTION_MARKERS):
            self.collection_marker_seen = True
        if _USAGE_ERROR.search(text):
            self.usage_error = True
        if _USAGE_LINE.search(text):
            self.usage_line = True
        if _NO_PYTEST.search(text):
            self.no_module_pytest = True


# ---------------------------------------------------------------------------
# Independent importability check
# ---------------------------------------------------------------------------


def run_independent_import_check(
    eligibility: LaunchEligibility | None,
    argv: Sequence[str],
    env: Mapping[str, str] | None,
    cwd: Path,
) -> str:
    """Return ``pytest-absent`` or ``unavailable``; never runs tests.

    Only a ``-m pytest`` eligibility whose argv[0] still resolves to the very
    file that was validated at launch is checked.
    """
    if eligibility is None or eligibility.shape != "module" or not argv:
        return "unavailable"
    base = dict(os.environ) if env is None else {str(k): str(v) for k, v in env.items()}
    try:
        resolved = _resolve_executable(str(argv[0]), base, cwd)
        validated = _interpreter_identity(resolved) if resolved else None
    except Exception:
        return "unavailable"
    if validated != (eligibility.interpreter_path, eligibility.identity):
        return "unavailable"
    code = _run_check(eligibility.invocation, _IMPORT_CHECK_CODE, base, cwd, IMPORT_CHECK_TIMEOUT_SECONDS)
    return "pytest-absent" if code == 3 else "unavailable"
