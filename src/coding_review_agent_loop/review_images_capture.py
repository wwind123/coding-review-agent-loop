"""Disconnected exact-head capture service. No public CLI or PR-loop dispatch."""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Literal

from .review_images_config import CaptureConfig, eligible

if TYPE_CHECKING:
    from .review_images_manifest import SnapshotSet


@dataclass(frozen=True)
class CaptureResult:
    state: Literal['disabled', 'isolation-unavailable', 'failed', 'resource-limit', 'timeout', 'success']
    diagnostic: str = ''
    snapshots: SnapshotSet | None = None
    isolation_proof: tuple[dict, ...] = ()


def _attempt(config, settings, source):
    from .containment import ContainmentPolicy, InvocationHandle, ResourceLimits
    from .review_images_isolation import sealed
    limits = ResourceLimits(settings['memory_bytes'] * 3 // 4, settings['memory_bytes'], 0,
                            settings['tasks'], settings['cpu_percent'])
    policy = ContainmentPolicy(mode='required', aggregate=ResourceLimits(None, None, 0, None), roles=(('capture', limits),))
    # Inherit the operator's existing aggregate ceiling through the shared
    # lease; never request an unbounded capture child or execution fallback.
    control = sealed(json.dumps(settings).encode())
    source_read = os.open(f'/proc/self/fd/{source.fileno()}', os.O_RDONLY | os.O_CLOEXEC)
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as result, tempfile.TemporaryFile() as errors:
        handle = None
        process = None
        try:
            command = [sys.executable, '-m', 'coding_review_agent_loop.review_images_host',
                       str(control), str(source_read), str(output.fileno()), str(result.fileno())]
            handle = InvocationHandle.prepare(policy, role='capture', target_argv=command)
            if not handle.managed:
                raise ValueError('capture requires delegated systemd cgroup-v2')
            # Only trusted monitor receives these FDs. It closes them on the
            # Bubblewrap boundary and never mounts any writable host path.
            env = {key: value for key, value in os.environ.items() if key in {
                'PATH', 'HOME', 'USER', 'LOGNAME', 'XDG_RUNTIME_DIR', 'DBUS_SESSION_BUS_ADDRESS'}}
            env['PYTHONPATH'] = str(Path(__file__).resolve().parent.parent)
            process = subprocess.Popen(handle.launcher_argv, env=env, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, stderr=errors,
                                       pass_fds=(*handle.pass_fds, control, source_read, output.fileno(), result.fileno()))
            cap = (settings['setup_timeout'] + settings['capture_timeout'] + 3 * settings['transfer_timeout'] + 10
                   if settings['mode'] == 'capture' else settings['transfer_timeout'] + 10)
            try:
                process.wait(timeout=cap)
            except subprocess.TimeoutExpired:
                handle.terminate(signal_name='KILL')
                process.wait(timeout=10)
                return {'state': 'timeout', 'error': 'whole-attempt watchdog'}, b''
            result.seek(0)
            raw = result.read(65537)
            if len(raw) > 65536:
                raise ValueError('oversized monitor reply')
            errors.seek(0)
            detail = errors.read(8192).decode(errors='replace')
            value = json.loads(raw) if raw else {'state': 'isolation-unavailable', 'error': 'capture delegated launcher refused: ' + detail}
            report = handle.refresh_report()
            if report and not report.target_started:
                value = {'state': 'isolation-unavailable', 'error': 'capture delegation/launcher unavailable'}
            if handle.refresh_resource_status() in {'oom-kill', 'resources'}:
                value = {'state': 'resource-limit', 'error': 'capture scope resource event'}
            if not handle.confirm_empty(timeout=5):
                handle.terminate(signal_name='KILL')
                if not handle.confirm_empty(timeout=5):
                    raise ValueError('capture hierarchical cleanup could not be verified')
                value = {'state': 'failed', 'error': 'capture required forced cleanup'}
            if value.get('state') == 'ok' and process.returncode:
                value = {'state': 'failed', 'error': 'capture scope failed'}
            if value.get('state') == 'ok':
                value['proof']['scope_empty'] = True
            output.seek(0)
            maximum = config.limits.total_bytes + config.limits.manifest_bytes + (config.limits.entries * 2 + 2) * 8196
            content = output.read(maximum + 1)
            if len(content) > maximum:
                raise ValueError('oversized export')
            return value, content
        finally:
            if handle is not None:
                if process is not None and process.poll() is None:
                    handle.terminate(signal_name='KILL')
                    process.wait(timeout=10)
                # Cancellation propagates only after whole-tree cleanup.
                if process is not None and not handle.confirm_empty(timeout=5):
                    handle.terminate(signal_name='KILL')
                    handle.confirm_empty(timeout=5)
                handle.close()
            os.close(control)
            os.close(source_read)


def capture(checkout: Path, head: str, config: CaptureConfig, changes: list[dict[str, str]]) -> CaptureResult:
    # Disabled operation does not load workers, probe dependencies or allocate.
    if not eligible(config, changes):
        return CaptureResult('disabled')
    if sys.platform != 'linux':
        return CaptureResult('isolation-unavailable', 'capture isolation requires Linux')
    from .errors import AgentLoopError
    from .review_images_channel import records, write_record
    from .review_images_isolation import preflight, validator_grants
    from .review_images_manifest import schema, publish
    from .review_images_worktree import stage
    try:
        bwrap = preflight(config.runtime_grants)
        png_grants = validator_grants(config.runtime_grants)
    except (OSError, ValueError, AgentLoopError, subprocess.SubprocessError) as exc:
        return CaptureResult('isolation-unavailable', str(exc)[:4096])
    try:
        settings = asdict(config)
        settings.update({'bwrap': bwrap, 'head': head, 'mode': 'capture',
                         'host_namespaces': {name: os.readlink('/proc/self/ns/' + name)
                                             for name in ('user', 'mnt', 'pid', 'ipc', 'uts', 'net')}})
        identity = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    except OSError as exc:
        return CaptureResult('isolation-unavailable', str(exc)[:4096])
    try:
        with tempfile.TemporaryFile() as source:
            stage(checkout, head, config.scratch_bytes, source)
            source.seek(0)
            settings['tree_digest'] = hashlib.file_digest(source, 'sha256').hexdigest()
            source.seek(0)
            value, exported = _attempt(config, settings, source)
        if value.get('state') != 'ok':
            return CaptureResult(value.get('state', 'failed'), value.get('error', '')[:4096])
        proof = value['proof']
        inventory = {}
        for metadata, data in records(io.BytesIO(exported), config.limits.total_bytes + config.limits.manifest_bytes,
                                      config.limits.entries * 2 + 1):
            if set(metadata) != {'path', 'size'} or metadata['path'] in inventory:
                raise ValueError('duplicate or malformed export record')
            inventory[metadata['path']] = data
        raw = inventory.pop('manifest.json')
        _, paths = schema(raw, head, config.limits)
        if set(paths) != set(inventory) or any(len(data) > config.limits.file_bytes for data in inventory.values()):
            raise ValueError('export inventory/byte mismatch')
        # Whole-set validation has one parent watchdog, no per-file resets.
        worker = {**settings, 'mode': 'validate', 'runtime_grants': png_grants, 'scratch_bytes': min(config.scratch_bytes, 8 * 1024**2),
                  'memory_bytes': min(config.memory_bytes, 256 * 1024**2), 'tasks': min(config.tasks, 16),
                  'transfer_timeout': min(config.validation_timeout, 30)}
        with tempfile.TemporaryFile() as source:
            for path, data in inventory.items():
                write_record(source, {'path': path, 'size': len(data)}, data)
            write_record(source, {'complete': True})
            source.seek(0)
            validated, _ = _attempt(config, worker, source)
        if validated.get('state') != 'ok':
            return CaptureResult(validated.get('state', 'failed'), validated.get('error', '')[:4096])
        snapshots = publish(raw, head, config.limits, inventory, validated['dimensions'], identity)
        return CaptureResult('success', value.get('diagnostic', '')[-65536:], snapshots,
                             (proof, validated['proof']))
    except (OSError, ValueError, TypeError, OverflowError, KeyError, RuntimeError, AgentLoopError, subprocess.SubprocessError) as exc:
        return CaptureResult('failed', str(exc)[:4096])
