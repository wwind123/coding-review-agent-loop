"""Process-tree containment for agent-loop invocations.

The containment layer is deliberately independent of the orchestration state
machine.  It can therefore be used by the normal agent runner, the foreground
test gate, and the external skill helper without making GitHub or checkout
state part of the resource-control protocol.

Linux systems with a delegated cgroup-v2 user manager use a transient systemd
scope.  The scope command is intentionally synchronous and foreground:
``systemd-run --user --scope --quiet``.  The target is started by a tiny shim
which writes an authoritative report before attempting the target exec.  This
is important because systemd's numeric exit status alone cannot distinguish a
launcher failure from a target which legitimately returned 1 or 203.

Every other platform, and Linux hosts without the required user-manager
capabilities, uses the runner's process-group termination.  That fallback is
explicitly reported and makes no memory-ceiling claim.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from .errors import AgentLoopError


CONTAINMENT_MODES = frozenset({"auto", "required", "off"})
CONTAINED_ROLES = ("coder", "reviewer", "repair", "test-gate")
DEFAULT_SLICE = "agent-loop.slice"
DEFAULT_OS_HEADROOM_PERCENT = 25.0
DEFAULT_OPTIONAL_COUNTERS = (
    "memory.peak",
    "memory.pressure",
    "memory.events.local",
    "memory.swap.current",
    "memory.swap.events",
    "memory.swap.peak",
)
REQUIRED_COUNTERS = ("memory.events", "pids.events")


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise AgentLoopError(f"{name} must be a finite non-negative number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AgentLoopError(f"{name} must be a finite non-negative number.") from exc
    if not math.isfinite(number) or number < 0:
        raise AgentLoopError(f"{name} must be a finite non-negative number.")
    return number


def host_memory_bytes() -> int:
    """Return physical memory, with a conservative portable fallback."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        value = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        if value > 0:
            return int(value)
    except (AttributeError, OSError, ValueError):
        pass
    return 1024 * 1024 * 1024


def parse_limit(value: object, *, name: str, total_bytes: int | None = None) -> int | None:
    """Parse a byte or percentage limit once.

    ``None``, ``max``, ``infinity`` and ``unlimited`` mean no finite limit.
    Percentages require a host total and use floor rounding.  Byte suffixes
    accept the IEC spellings used by systemd as well as their short forms.
    """
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"", "max", "infinity", "unlimited", "none"}:
            return None
        percent = raw.endswith("%")
        if percent:
            number = _finite_number(raw[:-1].strip(), name=name)
            if number > 100:
                raise AgentLoopError(f"{name} percentage must be between 0 and 100.")
            if total_bytes is None:
                raise AgentLoopError(f"{name} percentage needs a host memory total.")
            return int(total_bytes * number / 100)
        match = re.fullmatch(r"([+]?(?:\d+(?:\.\d*)?|\.\d+))\s*([kmgtpe]?i?b)?", raw)
        if not match:
            raise AgentLoopError(f"{name} must be bytes, a percentage, or 'max'.")
        number = _finite_number(match.group(1), name=name)
        suffix = (match.group(2) or "b").lower()
        multipliers = {
            "b": 1,
            "k": 1000,
            "kb": 1000,
            "m": 1000**2,
            "mb": 1000**2,
            "g": 1000**3,
            "gb": 1000**3,
            "t": 1000**4,
            "tb": 1000**4,
            "p": 1000**5,
            "pb": 1000**5,
            "e": 1000**6,
            "eb": 1000**6,
            "ki": 1024,
            "kib": 1024,
            "mi": 1024**2,
            "mib": 1024**2,
            "gi": 1024**3,
            "gib": 1024**3,
            "ti": 1024**4,
            "tib": 1024**4,
            "pi": 1024**5,
            "pib": 1024**5,
            "ei": 1024**6,
            "eib": 1024**6,
        }
        result = number * multipliers[suffix]
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result = _finite_number(value, name=name)
    else:
        raise AgentLoopError(f"{name} must be bytes, a percentage, or 'max'.")
    if result < 0 or result > (1 << 63) - 1:
        raise AgentLoopError(f"{name} is outside the supported resource range.")
    return int(result)


def parse_tasks_limit(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "max", "infinity", "unlimited", "none"}:
        return None
    number = _finite_number(value, name=name)
    if not number.is_integer() or number < 1:
        raise AgentLoopError(f"{name} must be a positive integer or 'max'.")
    return int(number)


@dataclass(frozen=True)
class ResourceLimits:
    memory_high: int | None
    memory_max: int | None
    memory_swap_max: int | None
    tasks_max: int | None

    def validate(self, *, name: str = "resource limits") -> "ResourceLimits":
        if self.memory_high is not None and self.memory_max is not None and self.memory_high > self.memory_max:
            raise AgentLoopError(f"{name}: MemoryHigh cannot exceed MemoryMax.")
        for field_name in ("memory_high", "memory_max", "memory_swap_max"):
            value = getattr(self, field_name)
            if value is not None and (isinstance(value, bool) or value < 0):
                raise AgentLoopError(f"{name}: {field_name} cannot be negative.")
        if self.tasks_max is not None and self.tasks_max < 1:
            raise AgentLoopError(f"{name}: TasksMax must be positive.")
        return self

    def restrict(self, other: "ResourceLimits") -> "ResourceLimits":
        """Return component-wise stricter finite limits."""
        def minimum(left: int | None, right: int | None) -> int | None:
            if left is None:
                return right
            if right is None:
                return left
            return min(left, right)

        return ResourceLimits(
            minimum(self.memory_high, other.memory_high),
            minimum(self.memory_max, other.memory_max),
            minimum(self.memory_swap_max, other.memory_swap_max),
            minimum(self.tasks_max, other.tasks_max),
        ).validate()

    def as_properties(self) -> tuple[str, ...]:
        def render(value: int | None) -> str:
            return "infinity" if value is None else str(value)
        return (
            f"MemoryHigh={render(self.memory_high)}",
            f"MemoryMax={render(self.memory_max)}",
            f"MemorySwapMax={render(self.memory_swap_max)}",
            f"TasksMax={render(self.tasks_max)}",
        )

    def to_dict(self) -> dict[str, int | None]:
        return {
            "MemoryHigh": self.memory_high,
            "MemoryMax": self.memory_max,
            "MemorySwapMax": self.memory_swap_max,
            "TasksMax": self.tasks_max,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ResourceLimits":
        return cls(
            int(payload["MemoryHigh"]) if payload.get("MemoryHigh") is not None else None,
            int(payload["MemoryMax"]) if payload.get("MemoryMax") is not None else None,
            int(payload["MemorySwapMax"]) if payload.get("MemorySwapMax") is not None else None,
            int(payload["TasksMax"]) if payload.get("TasksMax") is not None else None,
        )


@dataclass(frozen=True)
class ContainmentPolicy:
    mode: str = "auto"
    aggregate: ResourceLimits = field(default_factory=lambda: ResourceLimits(None, None, 0, None))
    roles: tuple[tuple[str, ResourceLimits], ...] = ()
    slice_name: str = DEFAULT_SLICE
    os_headroom_percent: float = DEFAULT_OS_HEADROOM_PERCENT
    cache_dir: Path | None = None
    systemd_run: str = "systemd-run"
    systemctl: str = "systemctl"

    def __post_init__(self) -> None:
        if self.mode not in CONTAINMENT_MODES:
            raise AgentLoopError("containment mode must be 'auto', 'required', or 'off'.")
        if not self.slice_name or "/" in self.slice_name or not self.slice_name.endswith(".slice"):
            raise AgentLoopError("containment slice must be a plain .slice unit name.")
        headroom = _finite_number(self.os_headroom_percent, name="containment OS headroom")
        if headroom >= 100:
            raise AgentLoopError("containment OS headroom must be less than 100%.")
        self.aggregate.validate(name="aggregate resource limits")
        known = dict(self.roles)
        for role, limits in known.items():
            if role not in CONTAINED_ROLES:
                raise AgentLoopError(f"unknown containment role: {role}")
            limits.validate(name=f"{role} resource limits")
            _validate_child_limits(role, limits, self.aggregate)

    def role_limits(self, role: str | None) -> ResourceLimits:
        configured = dict(self.roles)
        return configured.get(role or "coder", configured.get("coder", self.aggregate))

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "slice": self.slice_name,
            "os_headroom_percent": self.os_headroom_percent,
            "aggregate": self.aggregate.to_dict(),
            "roles": {role: limits.to_dict() for role, limits in self.roles},
        }


def _validate_child_limits(role: str, child: ResourceLimits, aggregate: ResourceLimits) -> None:
    for field_name in ("memory_high", "memory_max", "memory_swap_max", "tasks_max"):
        child_value = getattr(child, field_name)
        aggregate_value = getattr(aggregate, field_name)
        if child_value is None and aggregate_value is not None:
            raise AgentLoopError(f"{role} {field_name} cannot be unlimited when the aggregate is finite.")
        if child_value is not None and aggregate_value is not None and child_value > aggregate_value:
            raise AgentLoopError(f"{role} {field_name} cannot exceed the aggregate limit.")


def default_policy(
    *,
    mode: str = "auto",
    memory_high: object | None = None,
    memory_max: object | None = None,
    memory_swap_max: object | None = None,
    tasks_max: object | None = None,
    role_values: Mapping[str, Mapping[str, object | None]] | None = None,
    os_headroom_percent: object = DEFAULT_OS_HEADROOM_PERCENT,
    slice_name: str = DEFAULT_SLICE,
    cache_dir: Path | None = None,
) -> ContainmentPolicy:
    total = host_memory_bytes()
    usable = max(1, int(total * (100 - _finite_number(os_headroom_percent, name="containment OS headroom")) / 100))
    aggregate = ResourceLimits(
        parse_limit(memory_high if memory_high is not None else int(usable * 0.87), name="aggregate MemoryHigh", total_bytes=total),
        parse_limit(memory_max if memory_max is not None else usable, name="aggregate MemoryMax", total_bytes=total),
        parse_limit(memory_swap_max if memory_swap_max is not None else 0, name="aggregate MemorySwapMax", total_bytes=total),
        parse_tasks_limit(tasks_max if tasks_max is not None else min(8192, max(1024, (os.sysconf("SC_CHILD_MAX") if hasattr(os, "sysconf") else 4096))), name="aggregate TasksMax"),
    )
    role_map: dict[str, ResourceLimits] = {}
    values = role_values or {}
    for role in CONTAINED_ROLES:
        raw = values.get(role, {})
        default_role_max = int(aggregate.memory_max * 0.75) if aggregate.memory_max is not None else None
        role_max_value = raw.get("memory_max") if raw.get("memory_max") is not None else default_role_max
        parsed_role_max = parse_limit(role_max_value, name=f"{role} MemoryMax", total_bytes=total)
        default_role_high = int(parsed_role_max * 0.75) if parsed_role_max is not None else None
        if default_role_high is not None and aggregate.memory_high is not None:
            default_role_high = min(default_role_high, aggregate.memory_high)
        role_map[role] = ResourceLimits(
            parse_limit(raw.get("memory_high") if raw.get("memory_high") is not None else default_role_high, name=f"{role} MemoryHigh", total_bytes=total),
            parsed_role_max,
            parse_limit(raw.get("memory_swap_max") if raw.get("memory_swap_max") is not None else aggregate.memory_swap_max, name=f"{role} MemorySwapMax", total_bytes=total),
            parse_tasks_limit(raw.get("tasks_max") if raw.get("tasks_max") is not None else aggregate.tasks_max, name=f"{role} TasksMax"),
        ).validate(name=f"{role} resource limits")
    normalized_cache_dir = Path(cache_dir) if cache_dir is not None else None
    return ContainmentPolicy(
        mode=mode,
        aggregate=aggregate,
        roles=tuple(role_map.items()),
        slice_name=slice_name,
        os_headroom_percent=float(os_headroom_percent),
        cache_dir=normalized_cache_dir,
    )


def policy_from_values(values: Mapping[str, object], *, prefix: str = "containment") -> ContainmentPolicy:
    """Build a policy from argparse/config values, accepting both aggregate aliases."""
    def value(*names: str) -> object | None:
        for name in names:
            if name in values and values[name] is not None:
                return values[name]
        return None

    role_values: dict[str, dict[str, object | None]] = {}
    for role in CONTAINED_ROLES:
        role_values[role] = {
            "memory_high": value(f"{prefix}_{role}_memory_high"),
            "memory_max": value(f"{prefix}_{role}_memory_max"),
            "memory_swap_max": value(f"{prefix}_{role}_memory_swap_max"),
            "tasks_max": value(f"{prefix}_{role}_tasks_max"),
        }
    return default_policy(
        mode=str(value(f"{prefix}_mode") or "auto"),
        memory_high=value(f"{prefix}_aggregate_memory_high", f"{prefix}_memory_high"),
        memory_max=value(f"{prefix}_aggregate_memory_max", f"{prefix}_memory_max"),
        memory_swap_max=value(f"{prefix}_aggregate_memory_swap_max", f"{prefix}_memory_swap_max"),
        tasks_max=value(f"{prefix}_aggregate_tasks_max", f"{prefix}_tasks_max"),
        role_values=role_values,
        os_headroom_percent=(
            value(f"{prefix}_os_headroom_percent")
            if value(f"{prefix}_os_headroom_percent") is not None
            else DEFAULT_OS_HEADROOM_PERCENT
        ),
        slice_name=str(value(f"{prefix}_slice") or DEFAULT_SLICE),
        cache_dir=value(f"{prefix}_cache_dir"),
    )


@dataclass(frozen=True)
class CapabilityManifest:
    backend: str
    ready: bool
    systemd_version: int | None = None
    user_manager: bool = False
    unified_cgroup: bool = False
    supported_counters: tuple[str, ...] = ()
    unavailable_counters: tuple[str, ...] = ()
    required_missing: tuple[str, ...] = ()
    reason: str | None = None
    probe_report: str | None = None

    @property
    def memory_ceiling_claimed(self) -> bool:
        return self.backend == "systemd-cgroup-v2" and self.ready

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "ready": self.ready,
            "systemd_version": self.systemd_version,
            "user_manager": self.user_manager,
            "unified_cgroup": self.unified_cgroup,
            "supported_counters": list(self.supported_counters),
            "unavailable_counters": list(self.unavailable_counters),
            "required_missing": list(self.required_missing),
            "reason": self.reason,
        }


def _systemd_version(binary: str) -> int | None:
    try:
        result = subprocess.run((binary, "--version"), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"systemd\s+(\d+)", result.stdout or result.stderr)
    return int(match.group(1)) if match else None


def _counter_capabilities(cgroup_root: Path) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    supported: list[str] = []
    unavailable: list[str] = []
    required_missing: list[str] = []
    for counter in (*REQUIRED_COUNTERS, *DEFAULT_OPTIONAL_COUNTERS):
        if (cgroup_root / counter).exists():
            supported.append(counter)
        else:
            unavailable.append(counter)
            if counter in REQUIRED_COUNTERS:
                required_missing.append(counter)
    return tuple(supported), tuple(unavailable), tuple(required_missing)


def _probe_argv(policy: ContainmentPolicy, *, unit: str, report: Path) -> list[str]:
    return build_scope_argv(
        policy,
        ("/bin/sleep", "2"),
        report_path=report,
        unit_name=unit,
        limits=policy.aggregate,
    )


def build_scope_argv(
    policy: ContainmentPolicy,
    target_argv: Sequence[str],
    *,
    report_path: Path,
    unit_name: str,
    limits: ResourceLimits,
) -> list[str]:
    """Build the exact foreground systemd scope argv used in production."""
    return [
        policy.systemd_run, "--user", "--scope", "--quiet",
        f"--slice={policy.slice_name}", f"--unit={unit_name}",
        *[arg for prop in limits.as_properties() for arg in ("--property", prop)],
        "--property", "OOMPolicy=kill",
        "--", sys.executable, "-m", "coding_review_agent_loop.containment",
        "--shim", "--report", str(report_path), "--", *map(str, target_argv),
    ]


def preflight_containment(policy: ContainmentPolicy, *, cgroup_root: Path = Path("/sys/fs/cgroup"), probe: bool = True) -> CapabilityManifest:
    """Check the exact systemd launcher path before an agent is started."""
    if policy.mode == "off":
        return CapabilityManifest("process-group", True, reason="containment mode is off")
    if platform.system() != "Linux":
        return _fallback_manifest(policy, "systemd cgroup containment is only available on Linux")
    binary = shutil.which(policy.systemd_run)
    if binary is None:
        return _fallback_manifest(policy, "systemd-run is not installed")
    version = _systemd_version(binary)
    if version is None:
        return _fallback_manifest(policy, "could not determine systemd version")
    if version < 253:
        return _fallback_manifest(policy, f"systemd {version} is older than the required v253 floor")
    if not (cgroup_root / "cgroup.controllers").exists():
        return _fallback_manifest(policy, "the host does not expose a unified cgroup-v2 hierarchy")
    user_manager = shutil.which(policy.systemctl) is not None
    if not user_manager:
        return _fallback_manifest(policy, "systemctl is not installed; the user manager cannot be probed")
    if probe:
        cache = policy.cache_dir or Path(tempfile.gettempdir()) / "coding-review-agent-loop" / "containment"
        cache.mkdir(parents=True, exist_ok=True)
        report = cache / f"preflight-{uuid.uuid4().hex}.json"
        unit = f"agent-loop-preflight-{uuid.uuid4().hex[:12]}.scope"
        process = None
        stdout = ""
        stderr = ""
        try:
            # Keep the harmless probe alive long enough to inspect the actual
            # transient scope.  Looking only at /sys/fs/cgroup itself is wrong
            # on hosts where controllers are delegated below the root.
            process = subprocess.Popen(_probe_argv(policy, unit=unit, report=report), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            deadline = time.monotonic() + 5
            parsed: dict[str, object] | None = None
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    parsed = json.loads(report.read_text(encoding="utf-8")) if report.exists() else None
                except (OSError, UnicodeError, json.JSONDecodeError):
                    parsed = None
                if isinstance(parsed, dict) and parsed.get("state") in {"target-started", "target-exited"}:
                    break
                time.sleep(0.02)
            if parsed is None:
                stdout, stderr = process.communicate(timeout=1)
                return _fallback_manifest(policy, "scope probe produced no authoritative shim report", version=version, diagnostics=stderr or stdout)
            if parsed.get("state") not in {"target-started", "target-exited"}:
                stdout, stderr = process.communicate(timeout=1)
                return _fallback_manifest(policy, "scope probe did not start its target", version=version, diagnostics=stderr or stdout)
            probe_cgroup = cgroup_path_for_pid(int(parsed["pid"]), root=cgroup_root) if parsed.get("pid") else None
            supported, unavailable, required_missing = _counter_capabilities(probe_cgroup or cgroup_root)
        except (OSError, subprocess.TimeoutExpired, ValueError, KeyError) as exc:
            return _fallback_manifest(policy, f"systemd user scope probe failed: {exc}", version=version)
        finally:
            if process is not None and process.poll() is None:
                try:
                    subprocess.run((policy.systemctl, "--user", "kill", "--kill-who=all", "--signal", "TERM", unit), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=3)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            report.unlink(missing_ok=True)
    else:
        supported, unavailable, required_missing = _counter_capabilities(cgroup_root)
    if required_missing:
        return _fallback_manifest(policy, "required cgroup counters are unavailable: " + ", ".join(required_missing), version=version, diagnostics="")
    return CapabilityManifest(
        "systemd-cgroup-v2", True, version, True, True, supported, unavailable,
        reason="preflight passed",
    )


def _fallback_manifest(policy: ContainmentPolicy, reason: str, *, version: int | None = None, diagnostics: str | None = None) -> CapabilityManifest:
    manifest = CapabilityManifest(
        "process-group", False, version,
        supported_counters=(),
        unavailable_counters=tuple((*REQUIRED_COUNTERS, *DEFAULT_OPTIONAL_COUNTERS)),
        required_missing=REQUIRED_COUNTERS,
        reason=reason,
        probe_report=diagnostics or None,
    )
    if policy.mode == "required":
        raise AgentLoopError("required containment unavailable: " + reason)
    return manifest


def render_preflight(policy: ContainmentPolicy, manifest: CapabilityManifest) -> str:
    lines = ["Containment preflight", f"  mode: {policy.mode}", f"  backend: {manifest.backend}", f"  ready: {'yes' if manifest.ready else 'no (portable process-group fallback)'}", f"  aggregate: {json.dumps(policy.aggregate.to_dict(), sort_keys=True)}"]
    lines.append(f"  slice: {policy.slice_name}")
    lines.append(f"  systemd: {manifest.systemd_version or 'not available'}")
    lines.append("  supported counters: " + (", ".join(manifest.supported_counters) or "none"))
    lines.append("  unavailable counters: " + (", ".join(manifest.unavailable_counters) or "none"))
    if manifest.reason:
        lines.append(f"  reason: {manifest.reason}")
    return "\n".join(lines)


def _runtime_root(policy: ContainmentPolicy) -> Path:
    if policy.cache_dir is not None:
        root = policy.cache_dir
    else:
        root = Path(os.environ.get("XDG_RUNTIME_DIR", "")) / "agent-loop"
        if str(root) == "agent-loop":
            root = Path.home() / ".cache" / "agent-loop"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError:
        return "unknown-boot"


def _start_identity(pid: int) -> str:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        close = stat.rfind(")")
        fields = stat[close + 2 :].split()
        return fields[19] if len(fields) > 19 else "unknown"
    except (OSError, IndexError):
        return "dead"


def _pid_identity_alive(pid: int, identity: str, boot: str) -> bool:
    return boot == _boot_id() and identity not in {"dead", "unknown"} and _start_identity(pid) == identity


@dataclass
class AggregateLease:
    """Crash-recoverable lease for the per-user aggregate slice."""

    policy: ContainmentPolicy
    limits: ResourceLimits
    lease_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    _root: Path | None = field(default=None, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @classmethod
    def acquire(cls, policy: ContainmentPolicy) -> "AggregateLease":
        root = _runtime_root(policy)
        lock_path = root / "leases.lock"
        state_path = root / "leases.json"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            state = _load_leases(state_path)
            leases = [row for row in state if _pid_identity_alive(int(row.get("pid", -1)), str(row.get("start_identity", "")), str(row.get("boot_id", "")))]
            active = [ResourceLimits.from_dict(row.get("limits", {})) for row in leases]
            effective = policy.aggregate
            for limits in active:
                effective = effective.restrict(limits)
            # The newcomer may not weaken a live lease; it is admitted with the
            # strictest active aggregate and keeps that ceiling until it exits.
            row = {
                "lease_id": uuid.uuid4().hex,
                "pid": os.getpid(), "start_identity": _start_identity(os.getpid()),
                "boot_id": _boot_id(), "limits": policy.aggregate.to_dict(),
                "created": datetime.now(timezone.utc).isoformat(),
            }
            state = leases + [row]
            _write_leases(state_path, state)
        lease = cls(policy, effective, row["lease_id"], root)
        if not _apply_slice_properties(policy, effective):
            # Do not leave a lease behind when the aggregate ceiling could not
            # be applied.  In auto mode InvocationHandle.prepare turns this
            # into the explicit process-group fallback; required mode fails
            # closed in the same way as preflight.
            with lock_path.open("a+") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                _write_leases(
                    state_path,
                    [candidate for candidate in _load_leases(state_path)
                     if candidate.get("lease_id") != lease.lease_id],
                )
            raise AgentLoopError(
                f"aggregate slice properties could not be applied to {policy.slice_name}"
            )
        return lease

    def close(self) -> None:
        if self._closed:
            return
        root = self._root or _runtime_root(self.policy)
        with (root / "leases.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            state_path = root / "leases.json"
            rows = _load_leases(state_path)
            rows = [row for row in rows if row.get("lease_id") != self.lease_id]
            _write_leases(state_path, rows)
            if not rows:
                _reset_slice_properties(self.policy)
            else:
                # Re-apply the strictest ceiling still held by live leases.
                # This permits a stricter request to relax only after its own
                # lease exits, while a looser request cannot weaken a sibling.
                remaining = ResourceLimits(None, None, None, None)
                for row in rows:
                    remaining = remaining.restrict(ResourceLimits.from_dict(row.get("limits", {})))
                _apply_slice_properties(self.policy, remaining)
        self._closed = True

    def __enter__(self) -> "AggregateLease":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _load_leases(path: Path) -> list[dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _write_leases(path: Path, rows: list[dict[str, object]]) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=".leases-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(rows, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _apply_slice_properties(policy: ContainmentPolicy, limits: ResourceLimits) -> bool:
    if policy.mode == "off":
        return True
    if shutil.which(policy.systemctl) is None:
        return False
    try:
        result = subprocess.run(
            (policy.systemctl, "--user", "set-property", "--runtime", policy.slice_name, *limits.as_properties()),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _reset_slice_properties(policy: ContainmentPolicy) -> None:
    _apply_slice_properties(policy, ResourceLimits(None, None, None, None))


@dataclass(frozen=True)
class ExecReport:
    state: str
    pid: int | None = None
    returncode: int | None = None
    errno_value: int | None = None
    error: str | None = None
    cgroup_path: str | None = None

    @property
    def target_started(self) -> bool:
        return self.state in {"target-started", "target-exited"}

    @classmethod
    def read(cls, path: Path) -> "ExecReport | None":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            return None
        if not isinstance(payload, dict) or payload.get("state") not in {"target-started", "target-exited", "target-exec-error"}:
            return None
        return cls(str(payload["state"]), payload.get("pid"), payload.get("returncode"), payload.get("errno"), payload.get("error"), payload.get("cgroup_path"))


def _write_report(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(dict(payload), sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def _shim(argv: Sequence[str]) -> int:
    tokens = list(argv)
    try:
        report_index = tokens.index("--report") + 1
        report_path = Path(tokens[report_index])
    except (ValueError, IndexError):
        return 2
    try:
        remainder = tokens[tokens.index("--", report_index + 1) + 1 :]
    except ValueError:
        remainder = []
    if not remainder:
        _write_report(report_path, {"state": "target-exec-error", "errno": errno.ENOENT, "error": "target argv is empty"})
        return 127
    try:
        child = subprocess.Popen(remainder, start_new_session=False)
    except OSError as exc:
        _write_report(report_path, {"state": "target-exec-error", "errno": exc.errno, "error": str(exc)})
        return 127 if exc.errno == errno.ENOENT else 126
    target_cgroup = cgroup_path_for_pid(child.pid)
    _write_report(report_path, {"state": "target-started", "pid": child.pid, "cgroup_path": str(target_cgroup) if target_cgroup else None})
    try:
        code = child.wait()
    except BaseException:
        child.kill()
        child.wait()
        raise
    _write_report(report_path, {"state": "target-exited", "pid": child.pid, "returncode": code, "cgroup_path": str(target_cgroup) if target_cgroup else None})
    return int(code if code >= 0 else 128 + (-code))


@dataclass
class InvocationHandle:
    policy: ContainmentPolicy
    role: str
    invocation_id: str
    unit_name: str
    report_path: Path
    launcher_argv: tuple[str, ...]
    child_limits: ResourceLimits
    aggregate_limits: ResourceLimits
    backend: str = "process-group"
    capabilities: CapabilityManifest | None = None
    lease: AggregateLease | None = field(default=None, repr=False)
    cgroup_path: Path | None = None
    cleanup_confirmed: bool = False
    termination_cause: str | None = None
    applicable_limit: str | None = None
    diagnostics: tuple[str, ...] = ()

    @classmethod
    def prepare(cls, policy: ContainmentPolicy, *, role: str, target_argv: Sequence[str], env: Mapping[str, str] | None = None) -> "InvocationHandle":
        manifest = preflight_containment(policy)
        invocation_id = (env or {}).get("AGENT_LOOP_INVOCATION_ID") or uuid.uuid4().hex
        limits = policy.role_limits(role)
        diagnostics: tuple[str, ...] = ()
        if manifest.memory_ceiling_claimed:
            try:
                lease = AggregateLease.acquire(policy)
            except AgentLoopError as exc:
                if policy.mode == "required":
                    raise
                manifest = _fallback_manifest(policy, f"aggregate slice admission failed: {exc}")
                lease = None
                diagnostics = (f"{manifest.reason}; using process-group fallback",)
        else:
            lease = None
        aggregate = lease.limits if lease is not None else policy.aggregate
        limits = limits.restrict(aggregate)
        root = _runtime_root(policy)
        report = root / "reports" / f"{invocation_id}.json"
        report.unlink(missing_ok=True)
        unit = f"agent-loop-{invocation_id[:20]}.scope"
        launch = tuple(target_argv)
        backend = manifest.backend
        if manifest.memory_ceiling_claimed:
            # Re-resolve the child profile against the effective aggregate in
            # case another process holds a stricter active lease.
            launch = tuple(
                build_scope_argv(
                    policy, target_argv, report_path=report, unit_name=unit,
                    limits=limits,
                )
            )
        return cls(
            policy, role, invocation_id, unit, report, launch, limits, aggregate,
            backend, manifest, lease, diagnostics=diagnostics,
        )

    @property
    def managed(self) -> bool:
        return self.backend == "systemd-cgroup-v2"

    def refresh_report(self) -> ExecReport | None:
        report = ExecReport.read(self.report_path)
        if report and report.target_started and self.managed:
            self.cgroup_path = Path(report.cgroup_path) if report.cgroup_path else (cgroup_path_for_pid(report.pid) if report.pid else None)
        return report

    def terminate(self, *, signal_name: str = "TERM") -> None:
        self.termination_cause = "interrupt" if signal_name == "TERM" else "killed"
        if self.managed and shutil.which(self.policy.systemctl):
            try:
                subprocess.run((self.policy.systemctl, "--user", "kill", "--kill-who=all", "--signal", signal_name, self.unit_name), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=5)
            except (OSError, subprocess.TimeoutExpired) as exc:
                self.diagnostics += (f"scope termination failed: {exc}",)

    def confirm_empty(self, *, timeout: float = 2.0) -> bool:
        if not self.managed:
            self.cleanup_confirmed = True
            return True
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            report = self.refresh_report()
            if self.cgroup_path is not None:
                try:
                    members = (self.cgroup_path / "cgroup.procs").read_text(encoding="ascii").split()
                except FileNotFoundError:
                    members = []
                except OSError:
                    members = None
                if members == []:
                    self.cleanup_confirmed = True
                    return True
            if report is None or (report is not None and not report.target_started):
                if self._unit_inactive():
                    self.cleanup_confirmed = True
                    return True
            elif report.pid and not _pid_alive(report.pid) and self._unit_inactive():
                self.cleanup_confirmed = True
                return True
            time.sleep(0.05)
        report = self.refresh_report()
        self.cleanup_confirmed = self._unit_inactive() and (
            self.cgroup_path is None or not self._cgroup_members()
        )
        return self.cleanup_confirmed

    def _cgroup_members(self) -> list[str] | None:
        if self.cgroup_path is None:
            return None
        try:
            return (self.cgroup_path / "cgroup.procs").read_text(encoding="ascii").split()
        except FileNotFoundError:
            return []
        except OSError:
            return None

    def _unit_inactive(self) -> bool:
        if not self.managed or shutil.which(self.policy.systemctl) is None:
            return False
        try:
            result = subprocess.run(
                (self.policy.systemctl, "--user", "is-active", "--quiet", self.unit_name),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode != 0

    def close(self) -> None:
        if self.lease is not None:
            self.lease.close()
            self.lease = None

    def __enter__(self) -> "InvocationHandle":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _pid_alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        close = stat.rfind(")")
        if close >= 0 and stat[close + 2 :].startswith("Z "):
            return False
    except OSError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cgroup_path_for_pid(pid: int, *, root: Path = Path("/sys/fs/cgroup")) -> Path | None:
    try:
        text = Path(f"/proc/{pid}/cgroup").read_text(encoding="ascii")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("0::"):
            relative = line[3:].strip().lstrip("/")
            candidate = root / relative
            return candidate if candidate.exists() else candidate
    return None


def _read_counter(path: Path) -> dict[str, int] | int | str | None:
    try:
        text = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    if "events" in path.name or path.name == "memory.pressure":
        values: dict[str, int] = {}
        for line in text.splitlines():
            key, _, value = line.partition(" ")
            try:
                values[key] = int(value)
            except ValueError:
                continue
        return values
    try:
        return int(text)
    except ValueError:
        return text


def sample_cgroup(path: Path | None, manifest: CapabilityManifest | None = None) -> dict[str, object]:
    if path is None:
        return {}
    supported = set(manifest.supported_counters) if manifest else set(REQUIRED_COUNTERS) | set(DEFAULT_OPTIONAL_COUNTERS)
    sample: dict[str, object] = {}
    for counter in supported:
        value = _read_counter(path / counter)
        if value is not None:
            sample[counter] = value
    return sample


@dataclass(frozen=True)
class ContainmentEvidence:
    backend: str = "process-group"
    unit_name: str | None = None
    cgroup_path: str | None = None
    aggregate_limits: ResourceLimits | None = None
    child_limits: ResourceLimits | None = None
    supported_counters: tuple[str, ...] = ()
    unavailable_counters: tuple[str, ...] = ()
    before: Mapping[str, object] = field(default_factory=dict)
    after: Mapping[str, object] = field(default_factory=dict)
    termination_cause: str | None = None
    applicable_limit: str | None = None
    pressure: bool = False
    cleanup_confirmed: bool = False
    diagnostics: tuple[str, ...] = ()

    @property
    def resource_exhausted(self) -> bool:
        events = self.after.get("memory.events", {})
        pids = self.after.get("pids.events", {})
        return bool(isinstance(events, dict) and (events.get("oom_kill", 0) or events.get("oom", 0))) or bool(isinstance(pids, dict) and pids.get("max", 0))

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend, "unit": self.unit_name, "cgroup_path": self.cgroup_path,
            "aggregate_limits": self.aggregate_limits.to_dict() if self.aggregate_limits else None,
            "child_limits": self.child_limits.to_dict() if self.child_limits else None,
            "supported_counters": list(self.supported_counters),
            "unavailable_counters": list(self.unavailable_counters),
            "before": dict(self.before), "after": dict(self.after),
            "termination_cause": self.termination_cause,
            "applicable_limit": self.applicable_limit, "pressure": self.pressure,
            "cleanup_confirmed": self.cleanup_confirmed, "diagnostics": list(self.diagnostics),
        }


def evidence_for(handle: InvocationHandle, before: Mapping[str, object], after: Mapping[str, object], *, termination_cause: str | None = None, cleanup_confirmed: bool | None = None) -> ContainmentEvidence:
    events = after.get("memory.events", {})
    pids = after.get("pids.events", {})
    limit = None
    cause = termination_cause or handle.termination_cause
    if isinstance(events, dict) and (events.get("oom_kill", 0) or events.get("oom", 0)):
        cause = "oom"
        limit = "MemoryMax/MemoryHigh"
    elif isinstance(pids, dict) and pids.get("max", 0):
        cause = "tasks-max"
        limit = "TasksMax"
    if handle.termination_cause is None and handle.diagnostics:
        if any("target exec failed" in item for item in handle.diagnostics):
            cause = "target-exec-error"
    pressure = bool(isinstance(events, dict) and events.get("high", 0)) or "memory.pressure" in after
    manifest = handle.capabilities or CapabilityManifest(handle.backend, False)
    return ContainmentEvidence(
        backend=handle.backend,
        unit_name=handle.unit_name if handle.managed else None,
        cgroup_path=str(handle.cgroup_path) if handle.cgroup_path else None,
        aggregate_limits=handle.aggregate_limits,
        child_limits=handle.child_limits,
        supported_counters=manifest.supported_counters,
        unavailable_counters=manifest.unavailable_counters,
        before=before, after=after, termination_cause=cause,
        applicable_limit=limit, pressure=pressure,
        cleanup_confirmed=handle.cleanup_confirmed if cleanup_confirmed is None else cleanup_confirmed,
        diagnostics=handle.diagnostics,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if args[:1] == ["--shim"]:
        return _shim(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


# Stable descriptive aliases for integrations that use the containment module
# as a small library.  They intentionally avoid exposing any reserved response
# marker grammar.
ContainmentConfig = ContainmentPolicy
PreflightResult = CapabilityManifest
InvocationEvidence = ContainmentEvidence
ContainmentObservation = ContainmentEvidence
resolve_policy = policy_from_values
build_policy = default_policy
resolve_containment_policy = policy_from_values
build_systemd_scope_argv = build_scope_argv
preflight = preflight_containment
sample_invocation = sample_cgroup
