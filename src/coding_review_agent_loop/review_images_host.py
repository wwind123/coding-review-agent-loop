"""Capture-scope host monitor: credentials, gated cgroup migration and deadlines."""
from __future__ import annotations
import json
import os
from pathlib import Path
import select
import socket
import struct
import subprocess
import sys
import time

from .containment import cgroup_path_for_pid
from .review_images_cgroup import CaptureTree
from .review_images_isolation import argv, sealed


class IsolationUnavailable(ValueError):
    pass


class ResourceLimit(RuntimeError):
    pass


def failure_state(exc):
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ResourceLimit):
        return "resource-limit"
    if isinstance(exc, IsolationUnavailable):
        return "isolation-unavailable"
    return "failed"


def start_identity(pid):
    value = Path(f'/proc/{pid}/stat').read_text()
    rest = value[value.rfind(')') + 2:].split()
    return int(rest[1]), rest[19]


def release_launcher(pid, supervisor, phase, tree, channel):
    # SO_PASSCRED translates sender PIDs to this receiver's host namespace.
    # A PID supplied in JSON is deliberately neither accepted nor consulted.
    pidfd = os.pidfd_open(pid)
    try:
        parent, started = start_identity(pid)
        if parent != supervisor or cgroup_path_for_pid(pid) != tree.root / 'control':
            raise ValueError('foreign gated launcher identity')
        tree.check()
        if not tree.empty('payload'):
            raise ValueError('previous payload remains populated')
        leaf = tree.root / 'payload' / phase
        (leaf / 'cgroup.procs').write_text(str(pid))
        if (start_identity(pid) != (parent, started) or cgroup_path_for_pid(pid) != leaf
                or select.select([pidfd], [], [], 0)[0]):
            raise ValueError('gated launcher identity/membership changed')
        channel.send(json.dumps({'release': phase}).encode())
        return {'host_pid': pid, 'start': started, 'leaf': str(leaf)}
    finally:
        os.close(pidfd)


def monitor(config, input_fd, output_fd):
    own = cgroup_path_for_pid(os.getpid())
    if own is None or own.name != 'control' or not own.parent.name.endswith('.scope'):
        raise IsolationUnavailable('capture monitor is outside delegated control leaf')
    tree = CaptureTree(own.parent)
    initial = tree.counters()
    channel, sandbox_channel = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    channel.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
    files = {}
    process = None
    proof = {'launchers': [], 'scope': str(tree.root), 'limits': {
        name: (tree.root / name).read_text().strip() for name in ('memory.max', 'memory.swap.max', 'pids.max', 'cpu.max')},
        'controllers': (tree.root / 'cgroup.subtree_control').read_text().split()}
    try:
        effective = proof['limits']
        quota, period = effective['cpu.max'].split()
        if (int(effective['memory.max']) > config['memory_bytes']
                or effective['memory.swap.max'] != '0'
                or int(effective['pids.max']) > config['tasks']
                or int(quota) * 100 > config['cpu_percent'] * int(period)):
            raise IsolationUnavailable('effective capture resource limits exceed configured bounds')
        version = subprocess.run([config['bwrap'], '--version'], stdin=subprocess.DEVNULL,
                                 capture_output=True, timeout=5, check=True,
                                 env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'})
        proof['bubblewrap_version'] = version.stdout[:256].decode().strip()
        proof['scope_identity'] = {'device': tree.identity.st_dev, 'inode': tree.identity.st_ino}
        directory = Path(__file__).parent
        names = ('worker', 'channel', 'config', 'manifest', 'png', 'mounts')
        for name in names:
            filename = 'review_images_' + name + '.py'
            source = (directory / filename).read_text().replace('from .review_images_', 'from review_images_')
            files[filename] = sealed(source.encode(), filename)
        files['config.json'] = sealed(json.dumps(config).encode())
        command = argv(config['bwrap'], config['runtime_grants'], config['scratch_bytes'], files,
                       gate_fd=sandbox_channel.fileno())
        process = subprocess.Popen(command, stdin=input_fd, stdout=output_fd if config['mode'] == 'capture' else subprocess.DEVNULL, stderr=subprocess.PIPE,
                                   pass_fds=(*files.values(), sandbox_channel.fileno()),
                                   env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'})
        sandbox_channel.close()
        deadline = time.monotonic() + config['transfer_timeout']
        supervisor = None
        gate_phase = None
        validated = None
        loaded = False
        exported = False
        diagnostic = ''
        phases = [name for name in ('setup', 'capture') if config.get(name if name == 'setup' else 'command')]
        done = []
        while True:
            if tree.counters() != initial:
                raise ResourceLimit('capture resource event')
            if time.monotonic() >= deadline:
                raise TimeoutError('capture stage deadline exceeded')
            if select.select([channel], [], [], .02)[0]:
                data, ancillary, flags, _ = channel.recvmsg(65536, socket.CMSG_SPACE(struct.calcsize('3i')))
                if not data:
                    break
                if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
                    raise ValueError('truncated supervisor control message')
                credentials = [struct.unpack('3i', value) for level, kind, value in ancillary
                               if level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS]
                if len(credentials) != 1:
                    raise ValueError('missing kernel launcher credentials')
                pid, uid, _ = credentials[0]
                message = json.loads(data)
                if supervisor is None:
                    # First trusted message is the mount probe from PID1 in the
                    # sandbox. Verify its host ancestry and control membership.
                    ancestor = pid
                    for _ in range(8):
                        if ancestor == process.pid:
                            break
                        ancestor = start_identity(ancestor)[0]
                    else:
                        raise ValueError('foreign sandbox supervisor')
                    if set(message) != {'probe'} or uid != os.getuid() or cgroup_path_for_pid(pid) != own:
                        raise ValueError('unverified sandbox supervisor/probe')
                    supervisor = pid
                    proof.update(message['probe'])
                    channel.send(b'{"probe":"accepted"}')
                    continue
                if 'gate' in message:
                    phase = message['gate']
                    if config['mode'] != 'capture' or not loaded or gate_phase or not phases or phase != phases[0]:
                        raise ValueError('unexpected gated launcher')
                    proof['launchers'].append(release_launcher(pid, supervisor, phase, tree, channel))
                    gate_phase = phase
                    deadline = time.monotonic() + config[phase + '_timeout'] + 2
                    continue
                if pid != supervisor:
                    raise ValueError('untrusted control sender')
                if message == {'loaded': True} and not loaded and config['mode'] == 'capture':
                    loaded = True
                    channel.send(b'{"loaded":"accepted"}')
                elif 'phase_done' in message and message['phase_done'] == gate_phase and gate_phase:
                    tree.kill('payload')
                    proof[gate_phase + '_empty'] = tree.empty('payload')
                    channel.send(json.dumps({'empty': gate_phase}).encode())
                    code = message.get('code')
                    if code is None:
                        raise TimeoutError(gate_phase + ' timeout')
                    if code != 0:
                        raise ValueError(gate_phase + ' nonzero exit')
                    done.append(phases.pop(0))
                    gate_phase = None
                    deadline = time.monotonic() + config['transfer_timeout']
                elif message == {'export': 'ready'} and not phases and not gate_phase and not exported:
                    tree.kill('payload')
                    proof['export_payload_empty'] = tree.empty('payload')
                    channel.send(b'{"export":"accepted"}')
                    exported = True
                    deadline = time.monotonic() + config['transfer_timeout']
                elif set(message) == {'validated'} and config['mode'] == 'validate' and validated is None:
                    validated = message['validated']
                elif set(message) == {'diagnostic'} and isinstance(message['diagnostic'], str):
                    diagnostic = (diagnostic + message['diagnostic'])[-65536:]
                else:
                    raise ValueError('unexpected supervisor framing')
            if process.poll() is not None and not select.select([channel], [], [], 0)[0]:
                break
        code = process.wait(timeout=2)
        error = process.stderr.read(8192).decode(errors='replace')
        if tree.counters() != initial:
            raise ResourceLimit('capture resource event')
        tree.kill('payload')
        proof['payload_empty'] = tree.empty('payload')
        if code or supervisor is None or (config['mode'] == 'capture' and not exported):
            failure = IsolationUnavailable if supervisor is None else ValueError
            raise failure('sandbox failed: ' + error)
        if config['mode'] == 'validate' and validated is None:
            raise ValueError('missing validator reply')
        return {'state': 'ok', 'proof': proof, 'dimensions': validated, 'diagnostic': diagnostic}
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        try:
            tree.kill('payload')
        finally:
            tree.close()
            channel.close()
            sandbox_channel.close()
            for fd in files.values():
                os.close(fd)


def main():
    config_fd, input_fd, output_fd, result_fd = map(int, sys.argv[1:])
    config = json.loads(os.pread(config_fd, 65536, 0))
    try:
        result = monitor(config, input_fd, output_fd)
    except BaseException as exc:
        message = str(exc)[:4096]
        state = failure_state(exc)
        result = {'state': state, 'error': message}
    os.write(result_fd, json.dumps(result).encode())
    return 0 if result['state'] == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
