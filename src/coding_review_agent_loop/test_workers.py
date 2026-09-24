"""Containment-aware parallel test-worker budget (issue #848).

The budget is derived after containment admission from the limits that
actually apply on each execution path, exported to coder/repair agents as
``AGENT_LOOP_TEST_WORKERS`` and enforced for pytest-xdist by an injected
in-process pytest plugin (``pytest_plugin/_agent_loop_worker_cap.py``).

The plugin is a prompt-slip safety net, not a sandbox: cgroup memory limits
stay the hard boundary.  Everything in this module is side-effect free except
the lock helpers and the report-directory helpers, and every host probe is
injectable so tests never depend on the machine they run on.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import signal
import stat
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .errors import AgentLoopError


ENV_TEST_WORKERS = "AGENT_LOOP_TEST_WORKERS"
ENV_TEST_WORKER_ENFORCEMENT = "AGENT_LOOP_TEST_WORKER_ENFORCEMENT"
ENV_WORKER_CAP_SPEC = "AGENT_LOOP_WORKER_CAP_SPEC"
ENV_WORKER_CAP_NESTED = "AGENT_LOOP_WORKER_CAP_NESTED"
ENV_XDIST_AUTO = "PYTEST_XDIST_AUTO_NUM_WORKERS"
PLUGIN_MODULE = "_agent_loop_worker_cap"
PLUGIN_DISABLE_TOKEN = f"no:{PLUGIN_MODULE}"

WORKER_ENV_NAMES = (
    ENV_TEST_WORKERS,
    ENV_TEST_WORKER_ENFORCEMENT,
    ENV_WORKER_CAP_SPEC,
    ENV_WORKER_CAP_NESTED,
)

ENFORCEMENT_MODES = ("off", "clamp", "refuse")
DEFAULT_ENFORCEMENT = "clamp"
_STRICTNESS = {mode: index for index, mode in enumerate(ENFORCEMENT_MODES)}

GIB = 1024 ** 3
DEFAULT_PER_WORKER_BYTES = GIB
DEFAULT_RESERVE_BYTES = GIB

# Exit statuses shared with the command-lane contract.
WORKER_BUDGET_BUSY_EXIT_CODE = 125
WORKER_BUDGET_REFUSED_EXIT_CODE = 2

WORKER_BUDGET_LOCK_DIR = "worker-budget"
# Host-wide capacity record (issue #987): reservation files sit beside the
# per-invocation locks and a short-lived mutex serializes only the accounting.
HOST_CAPACITY_MUTEX = "host-capacity.mutex"
HOST_RESERVATION_SUFFIX = ".reservation"
ENV_HOST_SHARING = "AGENT_LOOP_TEST_WORKER_HOST_SHARING"
DESCENDANT_TERMINATION_GRACE_SECONDS = 2.0
DESCENDANT_KILL_CONFIRM_SECONDS = 10.0

CAVEAT_UNVERIFIED = "worker-budget-unverified"
CAVEAT_NOT_OBSERVED = "worker-budget-not-observed"
CAVEAT_REFUSED_SESSION = "worker-budget-refused-session"
CAVEAT_PARTIAL = "worker-budget-partial-observation"
CAVEAT_NESTED = "worker-budget-nested-session"
CAVEAT_MIXED = "worker-budget-mixed"
CAVEAT_REMOTE = "worker-budget-remote-gateways"
CAVEAT_EXCEEDED = "worker-budget-exceeded"
CAVEAT_DESCENDANTS = "worker-budget-descendants-terminated"

COHORT_SERIAL = "serial"
COHORT_UNKNOWN = "unknown"


class WorkerBudgetError(AgentLoopError):
    """A worker-budget option or inherited value is invalid."""


# ---------------------------------------------------------------------------
# Budget value and derivation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerBudget:
    """The effective test-worker budget for one execution path."""

    workers: int
    source: str = "derived"  # derived | operator | inherited
    enforcement: str = DEFAULT_ENFORCEMENT
    limiting_factor: str = "cpu"
    inputs: Mapping[str, object] = field(default_factory=dict, compare=False)
    enforced_ceiling: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 1:
            raise WorkerBudgetError("test worker budget must be a positive integer.")
        if self.enforcement not in ENFORCEMENT_MODES:
            raise WorkerBudgetError(f"unknown test worker enforcement mode: {self.enforcement}")
        if self.source not in {"derived", "operator", "inherited"}:
            raise WorkerBudgetError(f"unknown test worker budget source: {self.source}")

    @property
    def enforcing(self) -> bool:
        return self.enforcement != "off"

    def environment(self) -> dict[str, str]:
        return {
            ENV_TEST_WORKERS: str(self.workers),
            ENV_TEST_WORKER_ENFORCEMENT: self.enforcement,
        }

    def describe(self) -> str:
        source = {
            "derived": "derived",
            "operator": "operator-supplied",
            "inherited": "inherited from the parent loop",
        }[self.source]
        ceiling = (
            "enforced memory ceiling present"
            if self.enforced_ceiling
            else "no agent-loop memory ceiling enforced"
        )
        return (
            f"{self.workers} worker(s) ({source}; limiting factor: {self.limiting_factor}; "
            f"enforcement: {self.enforcement}; {ceiling})"
        )


def stricter_mode(left: str, right: str) -> str:
    return left if _STRICTNESS[left] >= _STRICTNESS[right] else right


def parse_enforcement(value: object, *, name: str = "--test-worker-enforcement") -> str:
    text = str(value).strip().lower() if value is not None else ""
    if text not in ENFORCEMENT_MODES:
        raise WorkerBudgetError(f"{name} must be one of clamp, refuse or off.")
    return text


def parse_worker_count(value: object, *, name: str = "--test-workers") -> int:
    if isinstance(value, bool):
        raise WorkerBudgetError(f"{name} must be a positive integer.")
    if isinstance(value, int):
        number = value
    else:
        text = str(value).strip()
        if not text.isdigit():
            raise WorkerBudgetError(f"{name} must be a positive integer.")
        number = int(text)
    if number < 1:
        raise WorkerBudgetError(f"{name} must be a positive integer.")
    return number


def parse_worker_memory(value: object, *, name: str = "--test-worker-memory") -> int:
    from .containment import parse_limit

    try:
        parsed = parse_limit(value, name=name)
    except AgentLoopError as exc:
        raise WorkerBudgetError(str(exc)) from exc
    if parsed is None or parsed <= 0:
        raise WorkerBudgetError(f"{name} must be a finite positive size.")
    return parsed


def probe_cpu_count() -> tuple[int, str]:
    """Return (usable CPUs, probe source); never raises."""
    getter = getattr(os, "sched_getaffinity", None)
    if getter is not None:
        try:
            cpus = getter(0)
        except (AttributeError, OSError, NotImplementedError):
            cpus = None
        if cpus:
            return len(cpus), "affinity"
    try:
        count = os.cpu_count()
    except (NotImplementedError, OSError):  # pragma: no cover - defensive
        count = None
    if count:
        return int(count), "cpu-count"
    return 1, "fallback-1"


def _default_cgroup_reader(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError):
        return None


_CGROUP_LIMIT_FILES = ("cpu.max", "memory.high", "memory.max")


@dataclass(frozen=True)
class AncestryLimits:
    memory: tuple[tuple[str, int], ...] = ()  # (file label, bytes)
    cpu_quotas: tuple[int, ...] = ()  # ceil(quota / period)
    skipped: tuple[str, ...] = ()
    cgroup: str | None = None


def read_ancestry_limits(
    *,
    cgroup_reader: Callable[[Path], str | None] | None = None,
    proc_cgroup: Path = Path("/proc/self/cgroup"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> AncestryLimits:
    """Read cpu.max, memory.high and memory.max along this process's cgroup ancestry.

    Only ``/proc/self/cgroup`` and those three files in each ancestor
    directory are read.  Missing, unreadable or racing files are skipped and
    recorded.
    """
    reader = cgroup_reader or _default_cgroup_reader
    raw = reader(proc_cgroup)
    if not raw:
        return AncestryLimits(skipped=(str(proc_cgroup),))
    relative: str | None = None
    for line in raw.splitlines():
        parts = line.strip().split(":", 2)
        if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
            relative = parts[2]
            break
    if relative is None:
        return AncestryLimits(skipped=("cgroup-v2-unavailable",))
    segments = [segment for segment in relative.strip("/").split("/") if segment and segment not in {".", ".."}]
    memory: list[tuple[str, int]] = []
    quotas: list[int] = []
    skipped: list[str] = []
    for depth in range(len(segments), -1, -1):
        directory = cgroup_root.joinpath(*segments[:depth]) if depth else cgroup_root
        for name in _CGROUP_LIMIT_FILES:
            path = directory / name
            text = reader(path)
            if text is None:
                skipped.append(str(path))
                continue
            value = text.strip()
            if name == "cpu.max":
                pieces = value.split()
                if not pieces or pieces[0] == "max":
                    continue
                try:
                    quota = int(pieces[0])
                    period = int(pieces[1]) if len(pieces) > 1 else 100000
                except ValueError:
                    skipped.append(str(path))
                    continue
                if quota > 0 and period > 0:
                    quotas.append(max(1, math.ceil(quota / period)))
            else:
                if value in {"", "max"}:
                    continue
                try:
                    number = int(value)
                except ValueError:
                    skipped.append(str(path))
                    continue
                if number > 0:
                    memory.append((f"{'/'.join(segments[:depth]) or '/'}:{name}", number))
    return AncestryLimits(tuple(memory), tuple(quotas), tuple(skipped), relative)


def _limit_value(limits: object | None) -> int | None:
    if limits is None:
        return None
    high = getattr(limits, "memory_high", None)
    if high is not None:
        return int(high)
    maximum = getattr(limits, "memory_max", None)
    return int(maximum) if maximum is not None else None


MANAGED_BACKEND = "systemd-cgroup-v2"


def derive_worker_budget(
    *,
    backend: str,
    child_limits: object | None = None,
    aggregate_limits: object | None = None,
    os_headroom_percent: float | None = None,
    per_worker_bytes: int | None = None,
    reserve_bytes: int | None = None,
    enforcement: str = DEFAULT_ENFORCEMENT,
    cpu_reader: Callable[[], tuple[int, str]] | None = None,
    memory_reader: Callable[[], int | None] | None = None,
    cgroup_reader: Callable[[Path], str | None] | None = None,
    ancestry: AncestryLimits | None = None,
) -> WorkerBudget:
    """Pure derivation of a worker budget from the limits that apply.

    Admitted handle limits count only for a managed systemd backend; ancestry
    limits count on every backend; usable host memory is always a candidate
    when readable, so a limit above physical memory never raises the ceiling.
    """
    from .containment import DEFAULT_OS_HEADROOM_PERCENT, probe_host_memory_bytes

    headroom = DEFAULT_OS_HEADROOM_PERCENT if os_headroom_percent is None else float(os_headroom_percent)
    per_worker = int(per_worker_bytes or DEFAULT_PER_WORKER_BYTES)
    reserve = DEFAULT_RESERVE_BYTES if reserve_bytes is None else int(reserve_bytes)
    inputs: dict[str, object] = {"backend": backend, "per_worker_bytes": per_worker, "reserve_bytes": reserve}

    try:
        available, cpu_source = (cpu_reader or probe_cpu_count)()
        available = int(available) if available else 1
    except Exception:  # pragma: no cover - injected readers may misbehave
        available, cpu_source = 1, "fallback-1"
    available = max(1, available)
    inputs["cpu_source"] = cpu_source
    inputs["cpu_available"] = available
    if ancestry is None:
        ancestry = read_ancestry_limits(cgroup_reader=cgroup_reader)
    if ancestry.skipped:
        inputs["skipped"] = list(ancestry.skipped[:16])
    cpu_term = available
    cpu_factor = "cpu"
    for quota in ancestry.cpu_quotas:
        if quota < cpu_term:
            cpu_term = quota
            cpu_factor = "cpu-quota"
    cpu_term = max(1, cpu_term)
    inputs["cpu_term"] = cpu_term

    candidates: list[tuple[int, str]] = []
    enforced = False
    try:
        host_total = (memory_reader or probe_host_memory_bytes)()
    except Exception:  # pragma: no cover - defensive
        host_total = None
    if host_total:
        host_usable = int(math.floor(host_total * (100 - headroom) / 100))
        inputs["host_usable_bytes"] = host_usable
        candidates.append((host_usable, "host-memory"))
    else:
        inputs["host_memory"] = "unreadable"
    if backend == MANAGED_BACKEND:
        child = _limit_value(child_limits)
        if child is not None:
            candidates.append((child, "child-limit"))
            enforced = True
        aggregate = _limit_value(aggregate_limits)
        if aggregate is not None:
            candidates.append((aggregate, "aggregate-limit"))
            enforced = True
    for label, value in ancestry.memory:
        candidates.append((value, "ancestry-limit"))
        enforced = True
        inputs.setdefault("ancestry_limits", []).append(label)  # type: ignore[union-attr]

    if not candidates:
        workers = max(1, cpu_term // 2)
        return WorkerBudget(
            workers, "derived", enforcement, "memory-unknown", inputs, enforced_ceiling=False,
        )
    ceiling, memory_factor = min(candidates, key=lambda item: item[0])
    inputs["memory_ceiling_bytes"] = ceiling
    mem_term = max(1, (ceiling - reserve) // per_worker)
    inputs["memory_term"] = mem_term
    if cpu_term <= mem_term:
        workers, factor = cpu_term, cpu_factor
    else:
        workers, factor = mem_term, memory_factor
    return WorkerBudget(max(1, int(workers)), "derived", enforcement, factor, inputs, enforced)


@dataclass(frozen=True)
class BudgetResolution:
    budget: WorkerBudget
    diagnostics: tuple[str, ...] = ()
    inherited: bool = False


def inherited_budget_values(env: Mapping[str, str]) -> tuple[int | None, str | None, tuple[str, ...]]:
    """Parse inherited control values; unparseable counts fail closed to 1."""
    diagnostics: list[str] = []
    workers: int | None = None
    mode: str | None = None
    raw = env.get(ENV_TEST_WORKERS)
    if raw is not None:
        try:
            workers = parse_worker_count(raw, name=ENV_TEST_WORKERS)
        except WorkerBudgetError:
            workers = 1
            diagnostics.append(
                f"agent-loop: unparseable inherited {ENV_TEST_WORKERS}={raw!r}; using 1 worker"
            )
    raw_mode = env.get(ENV_TEST_WORKER_ENFORCEMENT)
    if raw_mode is not None:
        try:
            mode = parse_enforcement(raw_mode, name=ENV_TEST_WORKER_ENFORCEMENT)
        except WorkerBudgetError:
            mode = "refuse"
            diagnostics.append(
                f"agent-loop: unparseable inherited {ENV_TEST_WORKER_ENFORCEMENT}={raw_mode!r}; using refuse"
            )
    return workers, mode, tuple(diagnostics)


def resolve_worker_budget(
    derived: WorkerBudget,
    *,
    operator_workers: int | None = None,
    operator_enforcement: str | None = None,
    env: Mapping[str, str] | None = None,
    has_parent: bool | None = None,
) -> BudgetResolution:
    """Apply the precedence rules of the approved plan.

    * loop flows and standalone run-tests: ``--test-workers`` replaces the
      derivation (raising or lowering) and ``--test-worker-enforcement`` sets
      the mode;
    * run-tests with a parent (inherited value or broker): child options and
      the local derivation may only lower the budget or tighten the mode.
    """
    values = env if env is not None else {}
    inherited_workers, inherited_mode, diagnostics = inherited_budget_values(values)
    parent = has_parent if has_parent is not None else (
        inherited_workers is not None or inherited_mode is not None
    )
    if parent and (inherited_workers is not None or inherited_mode is not None):
        base = inherited_workers if inherited_workers is not None else derived.workers
        workers = min(base, derived.workers)
        if operator_workers is not None:
            workers = min(workers, operator_workers)
        mode = inherited_mode or DEFAULT_ENFORCEMENT
        if operator_enforcement is not None:
            mode = stricter_mode(mode, operator_enforcement)
        factor = derived.limiting_factor if derived.workers < base else "inherited"
        if operator_workers is not None and operator_workers < min(base, derived.workers):
            factor = "operator"
        return BudgetResolution(
            WorkerBudget(
                max(1, workers), "inherited", mode, factor,
                {**derived.inputs, "inherited_workers": inherited_workers},
                derived.enforced_ceiling,
            ),
            diagnostics,
            True,
        )
    mode = operator_enforcement or derived.enforcement
    if operator_workers is not None:
        return BudgetResolution(
            WorkerBudget(
                operator_workers, "operator", mode, "operator",
                {**derived.inputs, "derived_workers": derived.workers},
                derived.enforced_ceiling,
            ),
            diagnostics,
        )
    return BudgetResolution(replace(derived, enforcement=mode), diagnostics)


# ---------------------------------------------------------------------------
# Command classification and wrapper-side injection
# ---------------------------------------------------------------------------


_PYTHON_FLAG_TOKENS = frozenset({"-B", "-E", "-I", "-O", "-OO", "-s", "-S", "-u", "-b", "-bb", "-q", "-P"})


def _is_python_name(token: str) -> bool:
    name = token.rsplit("/", 1)[-1]
    if name.startswith("python"):
        rest = name[len("python"):]
        return rest == "" or all(ch.isdigit() or ch == "." for ch in rest)
    return False


@dataclass(frozen=True)
class CommandShape:
    command_class: str  # direct-pytest | other
    head_index: int | None
    pytest_args_start: int | None  # index of the first pytest argument


def classify_command(argv: Sequence[str]) -> CommandShape:
    """Classify exactly as direct pytest (after supported prefixes) or other."""
    from .test_runtime import managed_wrapper_traversal

    tokens = tuple(str(item) for item in argv)
    if not tokens:
        return CommandShape("other", None, None)
    head = managed_wrapper_traversal(tokens).effective_head_index
    if head is None or head >= len(tokens):
        return CommandShape("other", None, None)
    name = tokens[head].rsplit("/", 1)[-1]
    if name in {"pytest", "py.test"}:
        return CommandShape("direct-pytest", head, head + 1)
    if _is_python_name(tokens[head]):
        index = head + 1
        while index < len(tokens) and tokens[index] in _PYTHON_FLAG_TOKENS:
            index += 1
        if (
            index + 1 < len(tokens)
            and tokens[index] == "-m"
            and tokens[index + 1] in {"pytest", "py.test"}
        ):
            return CommandShape("direct-pytest", head, index + 2)
        if index < len(tokens) and tokens[index] in {"-mpytest", "-mpy.test"}:
            return CommandShape("direct-pytest", head, index + 1)
    return CommandShape("other", head, None)


def plugin_directory() -> Path:
    """Return the on-disk directory holding the stdlib-only pytest plugin."""
    try:
        from importlib import resources

        candidate = resources.files("coding_review_agent_loop").joinpath("pytest_plugin")
        path = Path(str(candidate))
        if (path / f"{PLUGIN_MODULE}.py").is_file():
            return path
    except (ModuleNotFoundError, TypeError, ValueError):  # pragma: no cover - fallback below
        pass
    return Path(__file__).resolve().parent / "pytest_plugin"


def _is_disable_token(tokens: Sequence[str], index: int) -> int:
    """Return how many tokens a plugin-disabling ``-p`` spelling spans at index."""
    token = tokens[index]
    if token == "-p" and index + 1 < len(tokens) and tokens[index + 1] == PLUGIN_DISABLE_TOKEN:
        return 2
    if token in {f"-p{PLUGIN_DISABLE_TOKEN}", f"-p={PLUGIN_DISABLE_TOKEN}"}:
        return 1
    return 0


@dataclass
class _EnvSegment:
    index: int  # index of the env token (or first leading assignment)
    options_end: int  # first token after the options (where assignments begin)
    ignore_environment: bool = False
    unset: tuple[str, ...] = ()
    is_env: bool = True


def _env_segments(tokens: Sequence[str], head: int | None) -> tuple[list[_EnvSegment], list[int]]:
    """Locate env segments and assignment tokens before the command head."""
    from .test_runtime import (
        _MANAGED_EXECUTION_ASSIGNMENT_RE,
        _MANAGED_EXECUTION_PREFIXES,
        _consume_managed_execution_prefix_options,
    )

    segments: list[_EnvSegment] = []
    assignments: list[int] = []
    limit = head if head is not None else len(tokens)
    index = 0
    assignments_allowed = True
    while index < limit:
        token = tokens[index]
        if assignments_allowed and _MANAGED_EXECUTION_ASSIGNMENT_RE.match(token):
            assignments.append(index)
            index += 1
            continue
        wrapper = token.rsplit("/", 1)[-1]
        if wrapper not in _MANAGED_EXECUTION_PREFIXES:
            break
        next_index = _consume_managed_execution_prefix_options(tokens, index + 1, wrapper)
        if next_index is None:
            break
        if wrapper == "env":
            ignore = False
            unset: list[str] = []
            option_index = index + 1
            while option_index < next_index:
                option = tokens[option_index]
                if option in {"-i", "--ignore-environment", "-"}:
                    ignore = True
                elif option in {"-u", "--unset"} and option_index + 1 < next_index:
                    unset.append(tokens[option_index + 1])
                    option_index += 1
                elif option.startswith("--unset="):
                    unset.append(option.split("=", 1)[1])
                elif option.startswith("-u") and len(option) > 2:
                    unset.append(option[2:])
                option_index += 1
            segments.append(_EnvSegment(index, next_index, ignore, tuple(unset)))
        if wrapper == "timeout":
            next_index += 1
        index = next_index
        assignments_allowed = wrapper == "env"
    return segments, assignments


@dataclass(frozen=True)
class WorkerDecision:
    argv: tuple[str, ...]
    env: dict[str, str]
    command_class: str
    mode: str
    budget: int
    report_path: Path | None = None
    report_dir: Path | None = None
    refused: str | None = None
    notices: tuple[str, ...] = ()

    def cleanup(self) -> None:
        if self.report_dir is not None:
            shutil.rmtree(self.report_dir, ignore_errors=True)


def _merge_path_value(existing: str | None, entry: str, *, separator: str) -> str:
    parts = [part for part in (existing or "").split(separator) if part]
    if entry in parts:
        return separator.join(parts)
    return separator.join([entry, *parts]) if separator == os.pathsep else separator.join([*parts, entry])


def new_report_location() -> tuple[Path, Path]:
    directory = Path(tempfile.mkdtemp(prefix="agent-loop-worker-cap-"))
    os.chmod(directory, 0o700)
    return directory, directory / "report.jsonl"


def apply_worker_budget(
    argv: Sequence[str],
    env: Mapping[str, str],
    cwd: Path,
    budget: WorkerBudget,
    *,
    mode: str | None = None,
    report_location: tuple[Path, Path] | None = None,
) -> WorkerDecision:
    """Compute the effective argv/env for one command under ``budget``.

    Only direct pytest receives argv changes (the plugin token, and removal
    of a plugin-disabling ``-p no:`` in clamp mode).  Every other command
    keeps its argv and gets the same inert control environment in clamp and
    refuse.  Off mode injects nothing.  Worker values are never refused here:
    the plugin judges pytest's final resolved options.
    """
    del cwd  # classification is purely lexical; kept for interface symmetry
    effective_mode = mode or budget.enforcement
    tokens = [str(item) for item in argv]
    shape = classify_command(tokens)
    child_env = {str(key): str(value) for key, value in env.items()}
    child_env.pop(ENV_WORKER_CAP_NESTED, None)
    child_env.pop(ENV_WORKER_CAP_SPEC, None)
    child_env[ENV_TEST_WORKERS] = str(budget.workers)
    child_env[ENV_TEST_WORKER_ENFORCEMENT] = effective_mode
    if effective_mode == "off":
        return WorkerDecision(tuple(tokens), child_env, shape.command_class, effective_mode, budget.workers)

    notices: list[str] = []
    if shape.command_class == "direct-pytest":
        assert shape.pytest_args_start is not None
        index = shape.pytest_args_start
        kept: list[str] = tokens[:index]
        while index < len(tokens):
            span = _is_disable_token(tokens, index)
            if span:
                if effective_mode == "refuse":
                    return WorkerDecision(
                        tuple(tokens), child_env, shape.command_class, effective_mode, budget.workers,
                        refused=(
                            f"agent-loop worker budget refused: `-p {PLUGIN_DISABLE_TOKEN}` disables "
                            "worker-budget enforcement in refuse mode"
                        ),
                    )
                notices.append(
                    f"agent-loop worker budget: removed `-p {PLUGIN_DISABLE_TOKEN}` (clamp mode)"
                )
                index += span
                continue
            if tokens[index] == "--":
                kept.extend(tokens[index:])
                break
            kept.append(tokens[index])
            index += 1
        tokens = kept[: shape.pytest_args_start] + ["-p", PLUGIN_MODULE] + kept[shape.pytest_args_start:]

    report_dir, report_path = report_location or new_report_location()
    spec = json.dumps(
        {"version": 1, "budget": budget.workers, "mode": effective_mode, "report": str(report_path)},
        sort_keys=True,
    )
    plugin_dir = str(plugin_directory())
    auto_cap: str | None = None
    if effective_mode == "clamp":
        caller_auto = child_env.get(ENV_XDIST_AUTO)
        cap = budget.workers
        if caller_auto is not None and caller_auto.strip().isdigit() and int(caller_auto) > 0:
            cap = min(cap, int(caller_auto))
        auto_cap = str(cap)
    controlled: dict[str, str] = {
        ENV_WORKER_CAP_SPEC: spec,
        ENV_TEST_WORKERS: str(budget.workers),
        ENV_TEST_WORKER_ENFORCEMENT: effective_mode,
    }
    if auto_cap is not None:
        controlled[ENV_XDIST_AUTO] = auto_cap

    def pythonpath(existing: str | None) -> str:
        return _merge_path_value(existing, plugin_dir, separator=os.pathsep)

    def plugins(existing: str | None) -> str:
        return _merge_path_value(existing, PLUGIN_MODULE, separator=",")

    child_env["PYTHONPATH"] = pythonpath(child_env.get("PYTHONPATH"))
    child_env["PYTEST_PLUGINS"] = plugins(child_env.get("PYTEST_PLUGINS"))
    child_env.update(controlled)
    if shape.command_class != "direct-pytest":
        # ``other`` commands keep their argv byte-for-byte; a runner whose env
        # segments drop the control environment is simply not observed.
        return WorkerDecision(
            tuple(tokens), child_env, shape.command_class, effective_mode, budget.workers,
            report_path=report_path, report_dir=report_dir, notices=tuple(notices),
        )

    # Rewrite inline env segments of a direct pytest so they cannot disable
    # the plugin or raise the budget.
    shape = classify_command(tokens)
    segments, assignments = _env_segments(tokens, shape.head_index)
    rewritten = list(tokens)
    drop: set[int] = set()
    for position in assignments:
        name, _sep, value = rewritten[position].partition("=")
        if name == ENV_WORKER_CAP_NESTED:
            drop.add(position)
        elif name == "PYTHONPATH":
            rewritten[position] = f"PYTHONPATH={pythonpath(value)}"
        elif name == "PYTEST_PLUGINS":
            rewritten[position] = f"PYTEST_PLUGINS={plugins(value)}"
        elif name in controlled:
            rewritten[position] = f"{name}={controlled[name]}"
    required = [
        f"PYTHONPATH={plugin_dir}",
        f"PYTEST_PLUGINS={PLUGIN_MODULE}",
        *(f"{name}={value}" for name, value in controlled.items()),
    ]
    insertions: dict[int, list[str]] = {}
    for segment in segments:
        removed = segment.ignore_environment or any(
            name in {"PYTHONPATH", "PYTEST_PLUGINS", *controlled} for name in segment.unset
        )
        if not removed:
            continue
        if segment.ignore_environment:
            insert = list(required)
        else:
            insert = []
            for name in segment.unset:
                if name == "PYTHONPATH":
                    insert.append(f"PYTHONPATH={plugin_dir}")
                elif name == "PYTEST_PLUGINS":
                    insert.append(f"PYTEST_PLUGINS={PLUGIN_MODULE}")
                elif name in controlled:
                    insert.append(f"{name}={controlled[name]}")
        insertions.setdefault(segment.options_end, []).extend(insert)
    final: list[str] = []
    for position, token in enumerate(rewritten):
        if position in insertions:
            final.extend(insertions[position])
        if position in drop:
            continue
        final.append(token)
    return WorkerDecision(
        tuple(final), child_env, shape.command_class, effective_mode, budget.workers,
        report_path=report_path, report_dir=report_dir, notices=tuple(notices),
    )


def client_refusal(argv: Sequence[str], budget: WorkerBudget) -> str | None:
    """The only pre-spawn refusal: a plugin-disabling argv token in refuse mode."""
    if budget.enforcement != "refuse":
        return None
    tokens = [str(item) for item in argv]
    shape = classify_command(tokens)
    if shape.command_class != "direct-pytest" or shape.pytest_args_start is None:
        return None
    for index in range(shape.pytest_args_start, len(tokens)):
        if tokens[index] == "--":
            break
        if _is_disable_token(tokens, index):
            return (
                f"agent-loop worker budget refused: `-p {PLUGIN_DISABLE_TOKEN}` disables "
                "worker-budget enforcement in refuse mode"
            )
    return None


# ---------------------------------------------------------------------------
# Lane identity and the per-invocation worker-budget lock
# ---------------------------------------------------------------------------


_WORKER_VALUE_OPTIONS = ("-n", "--numprocesses", "--maxprocesses", "--tx", "--dist")
_CONTROLLED_PREFIX_NAMES = frozenset(
    {
        ENV_WORKER_CAP_SPEC,
        ENV_WORKER_CAP_NESTED,
        ENV_TEST_WORKERS,
        ENV_TEST_WORKER_ENFORCEMENT,
        ENV_XDIST_AUTO,
        "PYTEST_PLUGINS",
        "PYTHONPATH",
    }
)


def strip_worker_tokens(argv: Sequence[str]) -> tuple[tuple[str, ...], bool]:
    """Remove worker-selection tokens, controlled env assignments and ``-p no:``."""
    tokens = [str(item) for item in argv]
    shape = classify_command(tokens)
    if shape.command_class != "direct-pytest" or shape.pytest_args_start is None:
        return tuple(tokens), False
    _segments, assignments = _env_segments(tokens, shape.head_index)
    changed = False
    prefix: list[str] = []
    for position in range(shape.pytest_args_start):
        if position in assignments and tokens[position].split("=", 1)[0] in _CONTROLLED_PREFIX_NAMES:
            changed = True
            continue
        prefix.append(tokens[position])
    kept: list[str] = []
    for position, token in enumerate(prefix):
        # An ``env`` left with nothing but the command is a no-op spelling.
        following = prefix[position + 1] if position + 1 < len(prefix) else ""
        if (
            changed
            and token.rsplit("/", 1)[-1] == "env"
            and following
            and not following.startswith("-")
            and "=" not in following
        ):
            continue
        kept.append(token)
    index = shape.pytest_args_start
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            kept.extend(tokens[index:])
            break
        span = _is_disable_token(tokens, index)
        if span:
            changed = True
            index += span
            continue
        if token == "-d":
            changed = True
            index += 1
            continue
        matched = False
        for option in _WORKER_VALUE_OPTIONS:
            if token == option:
                changed = True
                index += 2
                matched = True
                break
            if token.startswith(option + "=") or (option == "-n" and token.startswith("-n") and len(token) > 2):
                changed = True
                index += 1
                matched = True
                break
        if matched:
            continue
        kept.append(token)
        index += 1
    return tuple(kept), changed


def worker_lane_identity(requested_argv: Sequence[str], *, cwd: Path) -> str:
    """The command-lane identity for every mode and budget.

    Commands without worker-selection tokens keep exactly today's normalized
    key, so the lane only ever serializes more than before.
    """
    from .test_runtime import normalize_test_command

    stripped, changed = strip_worker_tokens(requested_argv)
    return normalize_test_command(stripped if changed else requested_argv, cwd=cwd)


class WorkerBudgetLockError(AgentLoopError):
    """The trusted worker-budget lock directory failed its ownership checks."""


def _validate_private_directory(path: Path, *, allow_shared_mode: bool = False) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise WorkerBudgetLockError(f"cannot inspect worker-budget lock directory {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise WorkerBudgetLockError(f"worker-budget lock directory is not a plain directory: {path}")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise WorkerBudgetLockError(f"worker-budget lock directory is owned by another user: {path}")
    if not allow_shared_mode and info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise WorkerBudgetLockError(f"worker-budget lock directory is group- or world-writable: {path}")


def _ensure_private_directory(path: Path, *, allow_shared_mode: bool = False) -> Path:
    try:
        path.mkdir(mode=0o700, exist_ok=True)
    except OSError as exc:
        raise WorkerBudgetLockError(f"cannot create worker-budget lock directory {path}: {exc}") from exc
    _validate_private_directory(path, allow_shared_mode=allow_shared_mode)
    return path


def worker_budget_lock_root() -> Path:
    """Resolve the worker-budget lock directory without reading any environment."""
    uid = os.getuid() if hasattr(os, "getuid") else 0
    runtime = Path(f"/run/user/{uid}")
    try:
        runtime_info = runtime.stat()
        runtime_ok = stat.S_ISDIR(runtime_info.st_mode) and runtime_info.st_uid == uid
    except OSError:
        runtime_ok = False
    if runtime_ok:
        # /run/user/<uid> is private to the user, so the shared agent-loop
        # directory beneath it only needs the right owner; the lock
        # directory itself must be private.
        base = runtime / "agent-loop"
        _ensure_private_directory(base, allow_shared_mode=True)
        return _ensure_private_directory(base / WORKER_BUDGET_LOCK_DIR)
    base = Path("/tmp") / f"coding-review-agent-loop-{uid}"
    _ensure_private_directory(base)
    return _ensure_private_directory(base / WORKER_BUDGET_LOCK_DIR)


class WorkerBudgetLock:
    """Non-blocking per-invocation lock serializing enforced test commands."""

    def __init__(self, handle, path: Path, key: str):
        self.handle = handle
        self.path = path
        self.key = key
        self.reservation: Path | None = None

    @classmethod
    def acquire(
        cls,
        *,
        invocation_id: str | None,
        cwd: Path,
        root: Path | None = None,
    ) -> tuple["WorkerBudgetLock | None", str | None]:
        """Return (lock, None) or (None, reason) without blocking."""
        try:
            if root is not None:
                root.parent.mkdir(parents=True, exist_ok=True)
                directory = _ensure_private_directory(root)
            else:
                directory = worker_budget_lock_root()
        except (WorkerBudgetLockError, OSError) as exc:
            return None, str(exc)
        scope = invocation_id if invocation_id else str(cwd.resolve())
        key = f"worker-budget\0{scope}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        path = directory / f"{digest}.lock"
        try:
            handle = path.open("a+")
        except OSError as exc:
            return None, f"cannot open worker-budget lock {path}: {exc}"
        try:
            if os.name == "nt":  # pragma: no cover - not supported by the runner
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return None, None
        os.set_inheritable(handle.fileno(), False)
        return cls(handle, path, key), None

    def reserve_host_workers(self, requested: int, pool: int) -> int:
        """Reserve workers against the host-wide capacity record.

        Returns the number of workers granted, 0 when no capacity is left.
        With no other live reservation the full request is granted, so a
        single-loop host behaves exactly as before.  Otherwise the grant is
        capped at ``pool`` minus the workers other live holders reserved.  A
        reservation counts as live only while its holder's per-invocation
        lock is held (by the command or by its group watcher), so a killed
        loop's entry is reclaimed and a reservation is never released while a
        member of the holder's process group survives.
        """
        requested = max(1, int(requested))
        pool = max(1, int(pool))
        directory = self.path.parent
        reserved_by_others = 0
        with _host_capacity_mutex(directory):
            own = self.path.with_suffix(HOST_RESERVATION_SUFFIX)
            for record in sorted(directory.glob("*" + HOST_RESERVATION_SUFFIX)):
                if record == own:
                    continue
                workers = _live_reservation_workers(record)
                if workers is not None:
                    reserved_by_others += workers
            if reserved_by_others == 0:
                granted = requested
            else:
                granted = max(0, min(requested, pool - reserved_by_others))
            if granted == 0:
                return 0
            payload = json.dumps({"version": 1, "workers": granted, "lock": self.path.name}, sort_keys=True)
            temporary = own.with_name(own.name + ".tmp")
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload)
            os.replace(temporary, own)
            self.reservation = own
        return granted

    def _release_reservation(self) -> None:
        record, self.reservation = self.reservation, None
        if record is None:
            return
        try:
            with _host_capacity_mutex(record.parent):
                record.unlink(missing_ok=True)
        except OSError:
            pass  # a leftover entry is reclaimed once this lock is free

    def close(self) -> None:
        if self.handle.closed:
            return
        # Drop the reservation while the lock is still held; close() is only
        # reached once the target's process group is gone.
        self._release_reservation()
        try:
            if os.name != "nt":
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            self.handle.close()

    def hold_until_group_exits(self, pgid: int) -> str:
        """Keep the lock held until process group ``pgid`` has no live member.

        The flock belongs to the open file description, so a detached watcher
        process that inherits the descriptor keeps the lock held after this
        process closes its copy; the lock is released only when the watcher
        sees the group gone and exits.  If the watcher cannot be started, this
        call blocks until the group is gone instead.  Returns "watcher" or
        "waited".
        """
        import subprocess
        import sys

        if self.handle.closed:
            return "waited"
        fd = self.handle.fileno()
        try:
            subprocess.Popen(
                [sys.executable, "-c", _GROUP_WATCHER_SOURCE, str(pgid)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                pass_fds=(fd,), start_new_session=True, close_fds=True,
            )
        except OSError:
            while process_group_alive(pgid):
                time.sleep(0.2)
            self.close()
            return "waited"
        # Drop only this process's reference: no LOCK_UN, which would release
        # the lock shared with the watcher.
        self.handle.close()
        return "watcher"

    def __enter__(self) -> "WorkerBudgetLock":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class _host_capacity_mutex:
    """Blocking exclusive flock held only for the capacity accounting."""

    def __init__(self, directory: Path):
        self._path = directory / HOST_CAPACITY_MUTEX
        self._handle = None

    def __enter__(self) -> "_host_capacity_mutex":
        self._handle = self._path.open("a+")
        if os.name != "nt":
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *_args: object) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            if os.name != "nt":
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _live_reservation_workers(record: Path) -> int | None:
    """Workers held by a live reservation; reclaims and returns None when stale.

    Must be called with the host capacity mutex held.
    """
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
        workers = int(data["workers"])
        lock_name = str(data["lock"])
    except (OSError, ValueError, KeyError, TypeError):
        data = None
    if data is None or "/" in lock_name or workers < 1:
        record.unlink(missing_ok=True)
        return None
    lock_path = record.parent / lock_name
    try:
        handle = lock_path.open("r")
    except FileNotFoundError:
        record.unlink(missing_ok=True)
        return None
    except OSError:
        return workers  # cannot prove the holder dead
    with handle:
        if os.name == "nt":  # pragma: no cover - not supported by the runner
            return workers
        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError:
            return workers  # the holder (or its group watcher) still holds it
        try:
            record.unlink(missing_ok=True)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return None


def host_worker_pool(budget: WorkerBudget) -> int:
    """Total workers the host can run across all loops.

    Computed from host facts only (usable CPUs and usable host memory) so
    every loop on the host agrees on the pool regardless of its own cgroup
    caps or ``--test-workers`` value.  Falls back to probing the host when
    the budget's derivation inputs are unavailable.
    """
    inputs = budget.inputs or {}
    cpus = inputs.get("cpu_available")
    if not isinstance(cpus, int) or isinstance(cpus, bool) or cpus < 1:
        cpus, _source = probe_cpu_count()
    pool = max(1, int(cpus))
    usable = inputs.get("host_usable_bytes")
    if not isinstance(usable, int) or isinstance(usable, bool):
        usable = None
        try:
            from .containment import DEFAULT_OS_HEADROOM_PERCENT, probe_host_memory_bytes

            total = probe_host_memory_bytes()
            if total:
                usable = int(math.floor(total * (100 - DEFAULT_OS_HEADROOM_PERCENT) / 100))
        except Exception:  # pragma: no cover - defensive
            usable = None
    if usable:
        per_worker = inputs.get("per_worker_bytes")
        per_worker = per_worker if isinstance(per_worker, int) and per_worker > 0 else DEFAULT_PER_WORKER_BYTES
        reserve = inputs.get("reserve_bytes")
        reserve = reserve if isinstance(reserve, int) and reserve >= 0 else DEFAULT_RESERVE_BYTES
        pool = min(pool, max(1, (usable - reserve) // per_worker))
    return max(1, pool)


def host_sharing_enabled(env: Mapping[str, str]) -> bool:
    """Host-wide sharing is on unless explicitly opted out."""
    return str(env.get(ENV_HOST_SHARING, "")).strip().lower() not in {"off", "0", "false", "no"}


def host_capacity_busy_message(budget: WorkerBudget, pool: int) -> str:
    return (
        f"agent-loop: worker budget ({budget.workers} worker(s), {budget.enforcement}) is busy: "
        f"other agent-loop runs on this host hold all {pool} shared test worker(s); "
        "wait for them to finish before starting another."
    )


def worker_budget_busy_message(budget: WorkerBudget, reason: str | None = None) -> str:
    text = (
        f"agent-loop: worker budget ({budget.workers} worker(s), {budget.enforcement}) is held by "
        "another test command in this invocation; wait for it to exit before starting another."
    )
    if reason:
        text += f" ({reason})"
    return text


def process_group_members(pgid: int) -> list[int] | None:
    """Live (non-zombie) processes in ``pgid``, or None when /proc is not a
    complete view: /proc cannot be listed, or an entry that still exists
    cannot be read."""
    members: list[int] = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return None
    for name in entries:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", encoding="ascii", errors="replace") as stream:
                text = stream.read()
        except (FileNotFoundError, ProcessLookupError):
            continue  # exited during the scan
        except OSError:
            return None
        fields = text[text.rfind(")") + 2:].split()
        if len(fields) < 3 or fields[0] == "Z":
            continue
        try:
            if int(fields[2]) == pgid:
                members.append(int(name))
        except ValueError:
            continue
    return members


# The liveness probe is kept as source so the lock-holding watcher process
# (see WorkerBudgetLock.hold_until_group_exits) runs exactly the same check.
_GROUP_ALIVE_SOURCE = """
def group_alive(pgid):
    import os
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    # The group exists.  A complete /proc scan may still show only zombies,
    # which hold no memory; anything short of a complete scan counts as alive.
    try:
        entries = os.listdir("/proc")
    except OSError:
        return True
    for name in entries:
        if not name.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % name, encoding="ascii", errors="replace") as stream:
                text = stream.read()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError:
            return True
        fields = text[text.rfind(")") + 2:].split()
        if len(fields) >= 3 and fields[0] != "Z" and fields[2] == str(pgid):
            return True
    return False
"""
_group_namespace: dict = {}
exec(_GROUP_ALIVE_SOURCE, _group_namespace)  # noqa: S102 - trusted, module-local source


def process_group_alive(pgid: int) -> bool:
    """Whether any live process remains in ``pgid``.

    A signal-0 probe of the group decides first, independently of /proc; a
    complete /proc scan may then rule out a group that holds only zombies.
    """
    return bool(_group_namespace["group_alive"](pgid))


_GROUP_WATCHER_SOURCE = _GROUP_ALIVE_SOURCE + """
import sys, time
_pgid = int(sys.argv[1])
while group_alive(_pgid):
    time.sleep(0.2)
"""


def _wait_group_gone(pgid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while True:
        if not process_group_alive(pgid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


@dataclass(frozen=True)
class DescendantTermination:
    count: int
    confirmed: bool


def terminate_process_group_descendants(
    pgid: int,
    *,
    grace_seconds: float = DESCENDANT_TERMINATION_GRACE_SECONDS,
    kill_wait_seconds: float = DESCENDANT_KILL_CONFIRM_SECONDS,
) -> DescendantTermination:
    """Terminate whatever is left in the target's process group.

    The group is always signalled with ``killpg`` (no /proc enumeration is
    needed to decide whether to signal).  After SIGTERM and a grace period
    the group is SIGKILLed, and the call waits (bounded) until no live member
    remains.  ``confirmed`` is False when a member was still alive at the end
    of the wait; the caller must then keep the worker-budget lock held until
    the group is gone (WorkerBudgetLock.hold_until_group_exits).
    """
    members = process_group_members(pgid)
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return DescendantTermination(0, True)
    except OSError:
        pass
    count = len(members) if members is not None else 1
    if _wait_group_gone(pgid, grace_seconds):
        return DescendantTermination(count, True)
    count = max(count, 1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return DescendantTermination(count, True)
    except OSError:
        pass
    return DescendantTermination(count, _wait_group_gone(pgid, kill_wait_seconds))


# ---------------------------------------------------------------------------
# Report analysis and cohort labels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerReportAnalysis:
    workers_cohort: str
    enforcement: str
    caveats: tuple[str, ...] = ()
    notices: tuple[str, ...] = ()
    direct_refusal: bool = False
    sessions: tuple[Mapping[str, object], ...] = ()


def read_report_lines(path: Path | None) -> list[dict]:
    if path is None:
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get("kind"), str):
            rows.append(value)
    return rows


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def argv_only_workers_label(argv: Sequence[str]) -> str:
    """Proven-serial label for off-mode runs and legacy rows.

    Only a direct pytest whose last ``-p`` naming xdist is ``-p no:xdist`` is
    serial; everything else is unknown because other option sources can
    change the gateway count.
    """
    tokens = [str(item) for item in argv]
    shape = classify_command(tokens)
    if shape.command_class != "direct-pytest" or shape.pytest_args_start is None:
        return COHORT_UNKNOWN
    last: str | None = None
    index = shape.pytest_args_start
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            break
        value: str | None = None
        if token == "-p" and index + 1 < len(tokens):
            value = tokens[index + 1]
            index += 1
        elif token.startswith("-p") and len(token) > 2:
            value = token[2:].lstrip("=")
        if value is not None and value in {"xdist", "no:xdist", "xdist.plugin", "no:xdist.plugin"}:
            last = value
        index += 1
    return COHORT_SERIAL if last in {"no:xdist", "no:xdist.plugin"} else COHORT_UNKNOWN


def analyze_worker_report(
    report_path: Path | None,
    *,
    command_class: str,
    mode: str,
    argv: Sequence[str] = (),
) -> WorkerReportAnalysis:
    """Apply the verification rules to the plugin's JSON-lines report."""
    if mode == "off":
        return WorkerReportAnalysis(argv_only_workers_label(argv), "off")
    rows = read_report_lines(report_path)
    nested = [row for row in rows if row.get("kind") == "nested"]
    sessions: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("kind") in {"decision", "confirmed", "refused", "executed"} and isinstance(row.get("session"), str):
            sessions.setdefault(row["session"], []).append(row)
    direct = command_class == "direct-pytest"
    summaries: list[dict[str, object]] = []
    for session, lines in sessions.items():
        kinds = {line.get("kind") for line in lines}
        summaries.append(
            {
                "session": session,
                "refused": "refused" in kinds,
                "confirmed": "confirmed" in kinds,
                "executed": "executed" in kinds,
                "decision_only": kinds == {"decision"},
            }
        )
    if not sessions and not nested:
        if direct:
            return WorkerReportAnalysis(
                COHORT_UNKNOWN, "unverified", (CAVEAT_UNVERIFIED,),
                ("agent-loop worker budget: the pytest plugin wrote no valid report; "
                 "worker-budget enforcement is unverified for this run.",),
            )
        return WorkerReportAnalysis(COHORT_UNKNOWN, "not-observed", (CAVEAT_NOT_OBSERVED,))

    refused_lines = [line for lines in sessions.values() for line in lines if line.get("kind") == "refused"]
    if refused_lines:
        notices = []
        for line in refused_lines:
            notices.append(
                "agent-loop worker budget refused at stage "
                f"{line.get('stage')}: requested {line.get('requested')}, budget {line.get('budget')}"
                + (f", gateways {line.get('total')} (remote {line.get('remote')})" if line.get("total") is not None else "")
            )
        only_session = len(sessions) == 1 and not nested
        if direct and only_session:
            (lines,) = sessions.values()
            kinds = {line.get("kind") for line in lines}
            if kinds == {"refused"}:
                return WorkerReportAnalysis(
                    COHORT_UNKNOWN, "refused", (), tuple(notices), True, tuple(summaries)
                )
        detail = "; ".join(
            f"session {item['session']}: "
            + ", ".join(
                label for label, flag in (
                    ("refused", item["refused"]), ("confirmed", item["confirmed"]),
                    ("executed marker seen", item["executed"]), ("decision only", item["decision_only"]),
                ) if flag
            )
            for item in summaries
        )
        notices.append(f"agent-loop worker budget report (possibly incomplete): {detail}")
        caveats = [CAVEAT_REFUSED_SESSION]
        if nested:
            caveats.append(CAVEAT_NESTED)
        return WorkerReportAnalysis(
            COHORT_UNKNOWN, "refused-in-command", tuple(caveats), tuple(notices), False, tuple(summaries)
        )

    caveats: list[str] = []
    notices: list[str] = []
    confirmations = [
        next(line for line in lines if line.get("kind") == "confirmed")
        for lines in sessions.values()
        if any(line.get("kind") == "confirmed" for line in lines)
    ]
    unconfirmed = [item for item in summaries if not item["confirmed"]]
    effective_values = {_int_or_none(line.get("effective")) for line in confirmations}
    exceeded = any(line.get("action") == "exceeded" for line in confirmations)
    remote = any((_int_or_none(line.get("remote")) or 0) > 0 for line in confirmations)
    if exceeded:
        caveats.append(CAVEAT_EXCEEDED)
        notices.append(
            "agent-loop worker budget WARNING: more worker gateways were created than the budget; "
            "the worker budget was not enforced for this run."
        )
    if unconfirmed or not confirmations:
        caveats.append(CAVEAT_UNVERIFIED)
    if len(effective_values) > 1:
        caveats.append(CAVEAT_MIXED)
    if remote:
        caveats.append(CAVEAT_REMOTE)
    if nested:
        caveats.append(CAVEAT_NESTED)
    if not direct and confirmations:
        caveats.append(CAVEAT_PARTIAL)
    if exceeded:
        enforcement = "exceeded"
    elif not confirmations:
        enforcement = "unverified"
    elif any(line.get("action") == "clamped" for line in confirmations):
        enforcement = "clamped"
    else:
        enforcement = "unchanged"
    for line in confirmations:
        if line.get("action") == "clamped":
            auto = f" (auto resolved {line.get('auto_raw')})" if line.get("auto_raw") is not None else ""
            notices.append(
                "agent-loop worker budget: clamped worker request "
                f"{line.get('requested')}{auto} to {line.get('effective')} gateway(s) (top-level runner)"
            )
    cohort = COHORT_UNKNOWN
    if (
        direct
        and len(sessions) == 1
        and len(confirmations) == 1
        and not nested
        and not unconfirmed
    ):
        line = confirmations[0]
        effective = _int_or_none(line.get("effective"))
        if (
            effective is not None
            and (_int_or_none(line.get("remote")) or 0) == 0
            and line.get("action") in {"clamped", "unchanged"}
        ):
            cohort = COHORT_SERIAL if effective == 0 else str(effective)
    return WorkerReportAnalysis(cohort, enforcement, tuple(caveats), tuple(notices), False, tuple(summaries))


def expected_workers_label(argv: Sequence[str], *, budget: int | None = None, mode: str | None = None) -> str:
    """Best-effort cohort label used to look up a recommendation for ``argv``."""
    if mode not in {"clamp", "refuse"} or budget is None:
        return argv_only_workers_label(argv)
    tokens = [str(item) for item in argv]
    shape = classify_command(tokens)
    if shape.command_class != "direct-pytest" or shape.pytest_args_start is None:
        return COHORT_UNKNOWN
    if argv_only_workers_label(tokens) == COHORT_SERIAL:
        return COHORT_SERIAL
    requested: str | None = None
    index = shape.pytest_args_start
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            break
        if token in {"-n", "--numprocesses"} and index + 1 < len(tokens):
            requested = tokens[index + 1]
            index += 1
        elif token.startswith("--numprocesses="):
            requested = token.split("=", 1)[1]
        elif token.startswith("-n") and len(token) > 2:
            requested = token[2:]
        elif token in {"--tx", "-d", "--dist"} or token.startswith(("--tx=", "--dist=", "--maxprocesses")):
            return COHORT_UNKNOWN
        index += 1
    if requested is None:
        return COHORT_SERIAL
    if not requested.isdigit():
        return COHORT_UNKNOWN
    number = int(requested)
    if number == 0:
        return COHORT_SERIAL
    if mode == "clamp" and number > budget:
        number = budget if budget > 1 else 0
    return COHORT_SERIAL if number == 0 else str(number)


def row_workers_label(row: Mapping[str, object]) -> str:
    """The cohort label of a stored runtime row, classifying legacy rows."""
    value = row.get("workers")
    if isinstance(value, str) and value:
        return value
    argv = row.get("executed_argv") or row.get("argv")
    if isinstance(argv, list):
        return argv_only_workers_label([str(item) for item in argv])
    command = row.get("normalized_command")
    if isinstance(command, str):
        import shlex

        try:
            return argv_only_workers_label(shlex.split(command))
        except ValueError:
            return COHORT_UNKNOWN
    return COHORT_UNKNOWN


# ---------------------------------------------------------------------------
# Parallel-runner detection for prompt guidance
# ---------------------------------------------------------------------------


MAX_DETECTION_FILES = 32
MAX_DETECTION_BYTES = 256 * 1024
_MACHINE_FILES = (
    "pyproject.toml", "setup.cfg", "setup.py", "tox.ini",
    "poetry.lock", "uv.lock", "Pipfile.lock",
)


def _read_capped(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        with path.open("rb") as stream:
            return stream.read(MAX_DETECTION_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return None


def _pytest_parallel_invocation(text: str) -> bool:
    import re

    # Any value counts, including a shell variable such as
    # ``$AGENT_LOOP_TEST_WORKERS``; only an option-looking token does not.
    return bool(
        re.search(r"\bpytest\b[^\n`]*?\s(?:-n|--numprocesses)(?:=|\s*)(?!-)[^\s`]+", text)
    )


def _strip_machine_comments(path: Path, text: str) -> str:
    """Drop comment text so a commented-out requirement is not a requirement."""
    if path.suffix == ".lock" and path.name == "Pipfile.lock":
        return text  # JSON: no comments
    ini_like = path.suffix in {".cfg", ".ini"}
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if ini_like and stripped.startswith(";"):
            continue
        if stripped.startswith("#"):
            continue
        # Trailing ``# comment`` (``#`` after whitespace; URL fragments such as
        # ``#egg=`` have no preceding whitespace and are kept).
        cut = re.search(r"\s#", line)
        lines.append(line[: cut.start()] if cut else line)
    return "\n".join(lines)


def _code_fragments(text: str) -> Iterable[str]:
    import re

    in_fence = False
    block: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith(("```", "~~~")):
            if in_fence:
                yield "\n".join(block)
                block = []
            in_fence = not in_fence
            continue
        if in_fence:
            block.append(line)
        else:
            for span in re.findall(r"`([^`\n]+)`", line):
                yield span
    if in_fence and block:
        yield "\n".join(block)


def detect_parallel_support(root: Path) -> bool:
    """Bounded, text-only detection of pytest-xdist support in named files."""
    import re

    checked = 0
    machine: list[Path] = [root / name for name in _MACHINE_FILES]
    machine.extend(sorted(root.glob("requirements*.txt")))
    requirements_dir = root / "requirements"
    if requirements_dir.is_dir():
        machine.extend(sorted(requirements_dir.glob("*.txt")))
    for path in machine:
        if checked >= MAX_DETECTION_FILES:
            return False
        text = _read_capped(path)
        if text is None:
            continue
        checked += 1
        text = _strip_machine_comments(path, text)
        if re.search(r"(?<![A-Za-z0-9_.-])pytest[-_]xdist(?![A-Za-z0-9_-])", text):
            return True
    workflows = root / ".github" / "workflows"
    if workflows.is_dir():
        for path in sorted(workflows.glob("*.yml")):
            if checked >= MAX_DETECTION_FILES:
                return False
            text = _read_capped(path)
            if text is None:
                continue
            checked += 1
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith(("run:", "- run:")) and _pytest_parallel_invocation(" " + stripped):
                    return True
    docs: list[Path] = []
    for pattern in ("README*", "CONTRIBUTING*"):
        docs.extend(sorted(root.glob(pattern)))
    docs.extend([root / "AGENTS.md", root / "CLAUDE.md"])
    if (root / "docs").is_dir():
        docs.extend(sorted((root / "docs").glob("*.md")))
    for path in docs:
        if checked >= MAX_DETECTION_FILES:
            return False
        text = _read_capped(path)
        if text is None:
            continue
        checked += 1
        for fragment in _code_fragments(text):
            if _pytest_parallel_invocation(" " + fragment):
                return True
    return False


ENFORCEMENT_SENTENCES = {
    "clamp": "Over-budget worker requests are lowered to the budget.",
    "refuse": "Over-budget worker requests are refused and the run does not execute.",
    "off": "The budget is advisory; nothing enforces it.",
}


def render_worker_guidance(budget: WorkerBudget, *, parallel_supported: bool, preliminary: bool = True) -> str:
    """Coder/repair prompt block for the parallel test-worker budget."""
    estimate = "preliminary pre-admission estimate" if preliminary else "resolved budget"
    lines = [
        "Parallel test workers: `$AGENT_LOOP_TEST_WORKERS` in your environment is the "
        "authoritative launch-time worker budget "
        f"({estimate}: {budget.describe()}). "
        + ENFORCEMENT_SENTENCES[budget.enforcement],
    ]
    if parallel_supported:
        lines.append(
            "This repository supports parallel pytest workers: for broad or full-suite runs, "
            "pass `-n $AGENT_LOOP_TEST_WORKERS` (never `-n auto` above the budget); keep focused "
            "single-file runs serial."
        )
    else:
        lines.append(
            "No parallel test runner was detected for this repository; do not add parallel "
            "worker flags."
        )
    lines.append(
        "In clamp and refuse mode only one test command per invocation holds the worker budget "
        "at a time; a concurrent second command gets a worker-budget-busy result."
    )
    return " ".join(lines) + "\n"


def env_has_worker_values(env: Mapping[str, str]) -> bool:
    return any(name in env for name in (ENV_TEST_WORKERS, ENV_TEST_WORKER_ENFORCEMENT))


def budget_from_values(values: Mapping[str, object]) -> tuple[int | None, int | None, str | None]:
    """Parse operator worker options from a config/argparse value mapping."""
    workers = values.get("test_workers")
    memory = values.get("test_worker_memory")
    mode = values.get("test_worker_enforcement")
    return (
        parse_worker_count(workers) if workers is not None else None,
        parse_worker_memory(memory) if memory is not None else None,
        parse_enforcement(mode) if mode is not None else None,
    )
