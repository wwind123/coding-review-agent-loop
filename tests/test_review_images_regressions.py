"""Review regressions for the capture service's host trust boundary."""
import dataclasses
import importlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

import pytest

from coding_review_agent_loop import containment
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.review_images_cgroup import CaptureTree, delegate
from coding_review_agent_loop import review_images_host as host
from coding_review_agent_loop.review_images_config import CaptureConfig
from coding_review_agent_loop.review_images_channel import write_record
from review_images_helpers import repository, manifest, png
from test_review_images_capture import config, supported, required

service = importlib.import_module('coding_review_agent_loop.review_images_capture')


@pytest.mark.parametrize('name', ['docs/a:b.md', 'a\\b.txt', 'a\u200b.txt'])
@pytest.mark.parametrize('enabled', [False, True])
def test_service_bounds_invalid_changes(tmp_path, name, enabled):
    cfg = config() if enabled else CaptureConfig()
    result = service.capture(tmp_path, 'a'*40, cfg, [{'status':'added','filename':name}])
    assert result.state == ('failed' if enabled else 'disabled')


@pytest.mark.parametrize('failure', ['missing-systemd', 'old-systemd', 'no-cgroup', 'user-manager', 'manifest'])
def test_unavailable_containment_precedes_staging(tmp_path, monkeypatch, failure):
    def preflight(policy):
        if failure != 'manifest':
            raise AgentLoopError(failure)
        return containment.CapabilityManifest('process-group', False, reason='unavailable')
    monkeypatch.setattr(containment, 'preflight_containment', preflight)
    monkeypatch.setattr('coding_review_agent_loop.review_images_worktree.stage',
                        lambda *a: pytest.fail('staged before preflight'))
    result = service.capture(tmp_path, 'a'*40, config(), [{'status':'added','filename':'a.py'}])
    assert result.state == 'isolation-unavailable'


@pytest.mark.parametrize('message', ['mount-panel.png', 'resource-limit.png', 'probe.png', 'isolation-unavailable:'])
@pytest.mark.parametrize('exception,state', [(ValueError,'failed'), (host.ResourceLimit,'resource-limit'),
                                           (host.IsolationUnavailable,'isolation-unavailable'), (TimeoutError,'timeout')])
def test_classification_uses_exception_type(message, exception, state):
    assert host.failure_state(exception(message)) == state


@pytest.mark.parametrize('relative', ['', 'control', 'payload', 'payload/setup', 'payload/capture'])
@pytest.mark.parametrize('event', ['memory.events', 'pids.events'])
def test_events_at_every_topology_node(tmp_path, relative, event):
    for node in ('', 'control', 'payload', 'payload/setup', 'payload/capture'):
        directory = tmp_path / node
        directory.mkdir(exist_ok=True, parents=True)
        for name in ('memory.events', 'pids.events'):
            (directory/name).write_text('max 0\n')
    tree = CaptureTree(tmp_path)
    try:
        before = tree.counters()
        (tmp_path/relative/event).write_text('max 1\n')
        assert tree.counters() != before
    finally:
        tree.close()


@pytest.mark.parametrize('attack', ['foreign-parent', 'wrong-initial-membership', 'reused', 'wrong-final-membership', 'exited'])
def test_gated_launcher_rejects_unverified_identity(tmp_path, monkeypatch, attack):
    from types import SimpleNamespace
    root = tmp_path
    (root/'payload'/'capture').mkdir(parents=True)
    fd = os.open('/dev/null', os.O_RDONLY)
    monkeypatch.setattr(host.os, 'pidfd_open', lambda pid: os.dup(fd))
    calls = []
    def identity(pid):
        calls.append(pid)
        return (12 if attack=='foreign-parent' else 11, 'changed' if attack=='reused' and len(calls)>1 else 'start')
    monkeypatch.setattr(host, 'start_identity', identity)
    memberships = iter([root/'foreign' if attack=='wrong-initial-membership' else root/'control',
                        root/'foreign' if attack=='wrong-final-membership' else root/'payload'/'capture'])
    monkeypatch.setattr(host, 'cgroup_path_for_pid', lambda pid: next(memberships))
    monkeypatch.setattr(host.select, 'select', lambda *args: ([fd] if attack=='exited' else [], [], []))
    tree = SimpleNamespace(root=root, check=lambda: None, empty=lambda _: True)
    channel = SimpleNamespace(send=lambda _: pytest.fail('released unverified process'))
    try:
        with pytest.raises(ValueError):
            host.release_launcher(10, 11, 'capture', tree, channel)
    finally:
        os.close(fd)


@pytest.mark.parametrize('failure', ['controllers', 'populated', 'kill', 'memory.max', 'memory.swap.max', 'pids.max', 'cpu.max'])
def test_delegation_refuses_missing_enforcement(tmp_path, monkeypatch, failure):
    (tmp_path/'cgroup.controllers').write_text('memory pids' if failure=='controllers' else 'memory pids cpu')
    (tmp_path/'cgroup.procs').write_text('42' if failure=='populated' else '')
    for name, value in [('memory.max','1024'), ('memory.swap.max','0'), ('pids.max','16'), ('cpu.max','100000 100000')]:
        (tmp_path/name).write_text('max' if name==failure else value)
    (tmp_path/'cgroup.kill').write_text('')
    real_access = os.access
    monkeypatch.setattr(os, 'access', lambda path, mode: False if failure=='kill' and Path(path).name=='cgroup.kill' else real_access(path,mode))
    with pytest.raises(RuntimeError):
        delegate(tmp_path)
    if failure in {'controllers','populated'}:
        assert not (tmp_path/'cgroup.subtree_control').exists()


def test_identity_excludes_host_namespaces_and_validator_caps_cpu(tmp_path, monkeypatch):
    root, head, _ = repository(tmp_path)
    monkeypatch.setattr(containment, 'preflight_containment', lambda _: containment.CapabilityManifest('systemd-cgroup-v2', True))
    monkeypatch.setattr('coding_review_agent_loop.review_images_isolation.preflight', lambda _: '/usr/bin/bwrap')
    monkeypatch.setattr('coding_review_agent_loop.review_images_isolation.validator_grants', lambda _: ())
    raw = json.dumps(manifest(head)).encode()
    stream = io.BytesIO()
    for path,data in [('manifest.json',raw), ('one.png',png()), ('base.png',png())]:
        write_record(stream, {'path':path,'size':len(data)}, data)
    write_record(stream, {'complete':True})
    def attempt(cfg, settings, source):
        if settings['mode']=='validate':
            assert settings['cpu_percent']==100
            return {'state':'ok','proof':{},'dimensions':{'one.png':(1,1),'base.png':(1,1)}}, b''
        return {'state':'ok','proof':{}}, stream.getvalue()
    monkeypatch.setattr(service, '_attempt', attempt)
    digests=[]
    real_readlink=os.readlink
    for boot in ('boot-a','boot-b'):
        monkeypatch.setattr(os, 'readlink', lambda path: boot if str(path).startswith('/proc/self/ns/') else real_readlink(path))
        result=service.capture(root,head,config(cpu_percent=200),[{'status':'added','filename':'capture.py'}])
        assert result.state=='success',result
        with result.snapshots as snapshot:
            digests.append(snapshot.digest)
    assert digests[0]==digests[1]


def test_host_launch_ignores_caller_modules(tmp_path, monkeypatch):
    root,head,_=repository(tmp_path)
    supported(root,head)
    trap=tmp_path/'trap'
    trap.mkdir()
    marker=tmp_path/'imported'
    (trap/'json.py').write_text('from pathlib import Path\nPath('+repr(str(marker))+').touch()\nraise RuntimeError("host import")\n')
    (trap/'coding_review_agent_loop').mkdir()
    (trap/'coding_review_agent_loop'/'__init__.py').write_text((trap/'json.py').read_text())
    monkeypatch.chdir(trap)
    result=required(service.capture(root,head,config(),[{'status':'added','filename':'capture.py'}]))
    result.snapshots.close()
    assert not marker.exists()


def test_cancellation_cleans_scope_and_propagates(tmp_path, monkeypatch):
    root,head,_=repository(tmp_path)
    supported(root,head)
    prepare=containment.InvocationHandle.prepare
    handles=[]
    def record(*args, **kwargs):
        handle=prepare(*args, **kwargs)
        handles.append(handle)
        return handle
    monkeypatch.setattr(containment.InvocationHandle, 'prepare', record)
    wait=subprocess.Popen.wait
    interrupted=False
    def interrupt(process, *args, **kwargs):
        nonlocal interrupted
        # Only interrupt the managed capture launcher, after it has started.
        if handles and process.args == handles[-1].launcher_argv and not interrupted:
            deadline=time.monotonic()+10
            while time.monotonic()<deadline:
                handles[-1].refresh_report()
                root=handles[-1].cgroup_path
                if root is not None:
                    leaf=root/'payload'/'capture'/'cgroup.procs'
                    if leaf.exists() and leaf.read_text().strip():
                        break
                time.sleep(.02)
            else:
                pytest.fail('capture never reached live payload')
            interrupted=True
            raise KeyboardInterrupt()
        return wait(process,*args,**kwargs)
    monkeypatch.setattr(subprocess.Popen, 'wait', interrupt)
    with pytest.raises(KeyboardInterrupt):
        service.capture(root,head,dataclasses.replace(config(),command='sleep 20'),[{'status':'added','filename':'capture.py'}])
    assert interrupted and handles
    assert all(handle.confirm_empty(timeout=5) for handle in handles)


def test_tmpfs_memory_event_with_zero_exit_discards_set(tmp_path):
    root,head,_=repository(tmp_path)
    supported(root,head)
    # Cross MemoryHigh (96 MiB) using tmpfs pages while remaining below max.
    # Delete the pages and return zero: counters, rather than exit status,
    # must prevent publication. The filesystem ceiling is much larger.
    script="from pathlib import Path\np=Path('/scratch/pressure')\nwith p.open('wb') as f:\n for _ in range(100): f.write(bytes(1024*1024))\np.unlink()"
    cfg=dataclasses.replace(config(),memory_bytes=128*1024**2,
                            command='/usr/bin/python3 -c '+shlex.quote(script)+'; /usr/bin/python3 capture.py')
    result=service.capture(root,head,cfg,[{'status':'added','filename':'capture.py'}])
    assert result.state=='resource-limit' and result.snapshots is None,result
