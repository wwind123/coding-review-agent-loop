"""Capture-only delegated topology. Never infer emptiness from root procs."""
from __future__ import annotations
import os
from pathlib import Path
import time


class CaptureTree:
    def __init__(self, root: Path):
        self.root = root
        self.fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self.identity = os.fstat(self.fd)

    def check(self):
        now = self.root.stat()
        if (now.st_dev, now.st_ino) != (self.identity.st_dev, self.identity.st_ino):
            raise RuntimeError('capture cgroup identity changed')

    def close(self):
        os.close(self.fd)

    def empty(self, relative=''):
        self.check()
        fields = dict(line.split() for line in (self.root / relative / 'cgroup.events').read_text().splitlines())
        if fields.get('populated') not in {'0', '1'}:
            raise RuntimeError('missing hierarchical populated state')
        return fields['populated'] == '0'

    def kill(self, relative='', timeout=5):
        self.check()
        target = self.root / relative
        (target / 'cgroup.kill').write_text('1')
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.empty(relative):
                return
            time.sleep(.01)
        raise RuntimeError('capture subtree remains populated')

    def counters(self):
        self.check()
        result = {}
        # Older kernels and memory_localevents expose events only at leaves.
        for relative in ('', 'control', 'payload', 'payload/setup', 'payload/capture'):
            for name in ('memory.events', 'pids.events'):
                for line in (self.root / relative / name).read_text().splitlines():
                    key, value = line.split()
                    if key in {'high', 'max', 'oom', 'oom_kill', 'oom_group_kill'}:
                        result[relative + '/' + name + ':' + key] = int(value)
        return result


def delegate(root: Path):
    """Called only by the capture shim while its original scope root is pinned."""
    controllers = {'memory', 'pids', 'cpu'}
    if not controllers <= set((root / 'cgroup.controllers').read_text().split()):
        raise RuntimeError('capture requires delegated memory/pids/cpu controllers')
    control = root / 'control'
    control.mkdir()
    (control / 'cgroup.procs').write_text(str(os.getpid()))
    if (root / 'cgroup.procs').read_text().strip():
        raise RuntimeError('capture interior root is populated')
    (root / 'cgroup.subtree_control').write_text('+memory +pids +cpu')
    payload = root / 'payload'
    payload.mkdir()
    (payload / 'cgroup.subtree_control').write_text('+memory +pids +cpu')
    for phase in ('setup', 'capture'):
        (payload / phase).mkdir()
    for name in ('memory.max', 'memory.swap.max', 'pids.max', 'cpu.max'):
        value = (root / name).read_text().strip()
        if not value or value.split()[0] == 'max' or (name == 'memory.swap.max' and value != '0'):
            raise RuntimeError(f'capture limit not enforced: {name}')
    # Both teardown and hierarchical state must be available before exec.
    if not os.access(root / 'cgroup.kill', os.W_OK):
        raise RuntimeError('capture hierarchical kill unavailable')
    tree = CaptureTree(root)
    try:
        if not tree.empty('payload'):
            raise RuntimeError('capture payload starts populated')
    finally:
        tree.close()
