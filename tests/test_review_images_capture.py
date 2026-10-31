import dataclasses
import io
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

if sys.platform != 'linux':
    if os.environ.get('AGENT_LOOP_CAPTURE_SECURITY_REQUIRED') == '1':
        pytest.fail('required capture security evidence needs Linux', pytrace=False)
    pytest.skip('capture filesystem/security cases require Linux', allow_module_level=True)
from coding_review_agent_loop.review_images_capture import capture
from coding_review_agent_loop.review_images_config import CaptureConfig
from coding_review_agent_loop.review_images_isolation import preflight, argv
from coding_review_agent_loop.review_images_channel import records
from coding_review_agent_loop.review_images_worktree import stage
from review_images_helpers import repository


def config(**kwargs):
    return CaptureConfig(paths=('*.py',), command='/usr/bin/python3 capture.py',
                         setup='printf %s "$AGENT_LOOP_REVIEW_IMAGES_HEAD_SHA" > dependency',
                         runtime_grants=tuple(p for p in ('/usr/bin','/usr/lib','/usr/lib64') if Path(p).is_dir()), **kwargs)


def required(result):
    if result.state == 'isolation-unavailable' and os.environ.get('AGENT_LOOP_CAPTURE_SECURITY_REQUIRED') != '1':
        pytest.skip(result.diagnostic)
    assert result.state == 'success', (result.state, result.diagnostic)
    return result


def test_private_exact_head_staging(tmp_path):
    root, head, git = repository(tmp_path)
    git('config','filter.attack.smudge','touch stolen')
    (root/'capture.py').write_text('dirty source')
    stream=io.BytesIO()
    stage(root,head,4*1024**2,stream)
    entries=list(records(io.BytesIO(stream.getvalue()),4*1024**2,100))
    assert entries[0][0]['path']=='capture.py'
    assert b'import json' in entries[0][1]
    assert not (root/'stolen').exists()


def test_no_writable_host_bind_and_sizes_only_for_tmpfs():
    command=argv('/usr/bin/bwrap',('/usr/bin','/usr/lib'),2147483648,{},gate_fd=10)
    assert '--bind' not in command
    for i,arg in enumerate(command):
        if arg=='--dev-bind': assert command[i+1] in {'/dev/null','/dev/zero','/dev/full','/dev/random','/dev/urandom'}
    for index,arg in enumerate(command):
        if arg=='--size': assert command[index+2]=='--tmpfs'
    for path in ('/', '/dev', '/proc'):
        assert any(command[i:i+2]==['--remount-ro',path] for i in range(len(command)))


@pytest.mark.parametrize('grant', ['/','/home','/tmp','/etc','/run','/usr','/usr/local','/usr/share','/usr/local/etc','/usr/../home','/sys/fs/cgroup'])
def test_refuse_unsafe_grants(grant):
    with pytest.raises((ValueError, OSError)):
        preflight((grant,))


def test_supported_exact_head_roundtrip(tmp_path,monkeypatch):
    root,head,git=repository(tmp_path)
    monkeypatch.setenv('AGENT_LOOP_REVIEW_IMAGES_HEAD_SHA','hostile')
    monkeypatch.setenv('AGENT_LOOP_REVIEW_IMAGES_MANIFEST','/host-secret')
    monkeypatch.setenv('SSH_AUTH_SOCK','/host-agent')
    monkeypatch.setenv('HOST_SECRET','secret')
    for revision in range(2):
        result=required(capture(root,head,config(),[{'status':'added','filename':'capture.py'}]))
        with result.snapshots as snapshot:
            assert snapshot.head_sha==head
            assert len(snapshot.files)==2
            assert snapshot.files[0].read().startswith(b'\x89PNG')
            for proof in result.isolation_proof:
                assert proof['scope_empty'] and proof['payload_empty']
                assert proof['input_read_only'] and proof['nested_userns_disabled']
                assert proof['bubblewrap_version'].startswith('bubblewrap ')
                assert proof['runtime_version'].startswith('3.')
                assert proof['scope_identity']['inode'] > 0
                assert proof['mounts']['/scratch'][0]=='tmpfs'
                assert {'nodev','nosuid'} <= set(proof['mounts']['/scratch'][1])
                assert 'noexec' not in proof['mounts']['/scratch'][1]
                assert proof['limits']['memory.swap.max']=='0'
            assert result.isolation_proof[0]['launchers'][0]['host_pid']>1
            assert '/usr' not in result.isolation_proof[1]['mounts']
            assert '/usr/bin' not in result.isolation_proof[1]['mounts']
            assert result.isolation_proof[1]['stdout_device']
            assert set(result.isolation_proof[1]['namespaces'])=={'user','mnt','pid','ipc','uts','net'}
        (root/'next.py').write_text(str(revision))
        git('add','next.py')
        git('commit','-qm','next commit')
        head=git('rev-parse','HEAD')


def supported(root,head):
    result=required(capture(root,head,config(),[{'status':'added','filename':'capture.py'}]))
    result.snapshots.close()


@pytest.mark.parametrize('attack', ['symlink-image','symlink-parent','symlink-baseline','hardlink','fifo','wrong-head','bad-png','bad-baseline','empty','duplicate'])
def test_live_output_attacks_discard_set(tmp_path,attack):
    root,head,git=repository(tmp_path)
    supported(root,head)
    mutations={
        'symlink-image': "(o/'one.png').unlink(); (o/'one.png').symlink_to('/etc/passwd')",
        'symlink-parent': "(o/'link').symlink_to('/scratch'); m['images'][0]['path']='link/secret'",
        'symlink-baseline': "(o/'base.png').unlink(); (o/'base.png').symlink_to('/etc/passwd')",
        'hardlink': "(o/'base.png').unlink(); os.link(o/'one.png',o/'base.png')",
        'fifo': "(o/'one.png').unlink(); os.mkfifo(o/'one.png')",
        'wrong-head': "m['head_sha']='b'*40",
        'bad-png': "(o/'one.png').write_bytes(b'not png')",
        'bad-baseline': "(o/'base.png').write_bytes(b'not png')",
        'empty': "m['images']=[]",
        'duplicate': "m['images'].append(dict(m['images'][0]))",
    }
    script="import json,os; from pathlib import Path; p=Path(os.environ['AGENT_LOOP_REVIEW_IMAGES_MANIFEST']); o=p.parent; m=json.loads(p.read_text()); "+mutations[attack]+"; p.write_text(json.dumps(m))"
    altered=dataclasses.replace(config(),command='/usr/bin/python3 capture.py; /usr/bin/python3 -c '+shlex.quote(script))
    result=capture(root,head,altered,[{'status':'modified','filename':'capture.py'}])
    assert result.state!='success' and result.snapshots is None, result


@pytest.mark.parametrize('location', ['/scratch/work','/scratch/home','/tmp','/var/tmp','/dev/shm','/scratch/output'])
def test_real_tmpfs_enospc(tmp_path,location):
    root,head,git=repository(tmp_path)
    supported(root,head)
    script="""import errno,os
p=LOCATION+'/fill'
try:
 with open(p,'wb') as f:
  for _ in range(100): f.write(bytes(1024*1024))
except OSError as e:
 assert e.errno==errno.ENOSPC,e
else: raise AssertionError('filesystem was unbounded')
os.unlink(p)
""".replace('LOCATION',repr(location))
    altered=dataclasses.replace(config(scratch_bytes=8*1024**2),command='/usr/bin/python3 -c '+shlex.quote(script)+'; /usr/bin/python3 capture.py')
    result=required(capture(root,head,altered,[{'status':'modified','filename':'capture.py'}]))
    assert result.isolation_proof[0]['scratch_bytes']==8*1024**2
    result.snapshots.close()


def test_credentials_shared_paths_and_control_fds_are_unreachable(tmp_path):
    root,head,git=repository(tmp_path)
    supported(root,head)
    secret=tmp_path/'credential'
    secret.write_text('private')
    script="""import os
from pathlib import Path
assert not Path(SECRET).exists()
assert not Path('/sys/fs/cgroup').exists()
for path in ('/root','/home','/run'):
 assert not Path(path).exists()
assert set(os.listdir('/proc/self/fd')) <= {'0','1','2','3'}
try: Path('/escape').write_text('x')
except OSError: pass
else: raise AssertionError('root writable')
try: Path('/dev/escape').write_text('x')
except OSError: pass
else: raise AssertionError('dev writable')
""".replace('SECRET',repr(str(secret)))
    altered=dataclasses.replace(config(),command='/usr/bin/python3 -c '+shlex.quote(script)+' && /usr/bin/python3 capture.py')
    result=required(capture(root,head,altered,[{'status':'modified','filename':'capture.py'}]))
    result.snapshots.close()
    assert secret.read_text()=='private'


@pytest.mark.parametrize('kind', ['memory','tasks','cpu','nonzero'])
def test_real_resource_limits_and_timeout(tmp_path,kind):
    root,head,git=repository(tmp_path)
    supported(root,head)
    scripts={
        'memory': 'a=bytearray(512*1024*1024)',
        'tasks': "import os,time\ntry:\n while True:\n  if os.fork()==0: time.sleep(10); os._exit(0)\nexcept OSError: pass",
        'cpu': 'import os\nos.fork()\nwhile True: pass',
        'nonzero': 'raise SystemExit(17)',
    }
    altered=dataclasses.replace(config(),command='/usr/bin/python3 -c '+shlex.quote(scripts[kind]),
                                memory_bytes=128*1024**2,tasks=32,capture_timeout=2)
    result=capture(root,head,altered,[{'status':'modified','filename':'capture.py'}])
    assert result.snapshots is None
    if kind in {'memory','tasks'}: assert result.state=='resource-limit',result
    elif kind=='cpu': assert result.state=='timeout',result
    else: assert result.state=='failed',result


def test_background_setsid_writers_killed_before_export(tmp_path):
    root,head,git=repository(tmp_path)
    supported(root,head)
    altered=dataclasses.replace(config(),command='/usr/bin/python3 capture.py; setsid /bin/sh -c "while true; do cp /scratch/output/base.png /scratch/output/replacement; mv /scratch/output/replacement /scratch/output/one.png; sleep 0.1; done" &')
    result=required(capture(root,head,altered,[{'status':'modified','filename':'capture.py'}]))
    assert result.isolation_proof[0]['capture_empty']
    assert result.isolation_proof[0]['scope_empty']
    result.snapshots.close()


@pytest.mark.parametrize('command', ['kill -KILL 1', 'exit 19'])
def test_supervisor_or_setup_loss_never_publishes(tmp_path,command):
    root,head,git=repository(tmp_path)
    supported(root,head)
    result=capture(root,head,dataclasses.replace(config(),setup=command),[{'status':'added','filename':'capture.py'}])
    assert result.state=='failed' and result.snapshots is None,result


@pytest.mark.parametrize('reason', ['version','size'])
def test_bubblewrap_prerequisites_refused(monkeypatch, reason):
    from coding_review_agent_loop import review_images_isolation as isolation
    real=isolation.subprocess.run
    def run(command,**kwargs):
        if command[-1]=='--version':
            return subprocess.CompletedProcess(command,0,b'bubblewrap 0.8.0' if reason=='version' else b'bubblewrap 0.9.0',b'')
        if command[-1]=='--help':
            return subprocess.CompletedProcess(command,0,b'no support',b'')
        return real(command,**kwargs)
    monkeypatch.setattr(isolation.subprocess,'run',run)
    with pytest.raises(ValueError): preflight(('/usr/bin',))


@pytest.mark.parametrize('change', ['type','nodev','nosuid','noexec','size','root','dev','extra'])
def test_mount_inventory_refuses_unverified_boundaries(change):
    from coding_review_agent_loop.review_images_mounts import verify_mounts
    mounts='''1 0 0:1 / / ro,nosuid,nodev - tmpfs tmpfs rw
2 1 0:2 / /proc ro,nosuid,nodev,noexec - proc proc rw
3 1 0:3 / /dev ro,nosuid,nodev - tmpfs tmpfs rw
4 1 0:4 / /scratch rw,nosuid,nodev - tmpfs tmpfs rw,size=8192k'''
    assert verify_mounts(mounts,8388608,8388608,4096)['scratch_bytes']==8388608
    cap=8388608
    if change=='type': mounts=mounts.replace('/scratch rw,nosuid,nodev - tmpfs','/scratch rw,nosuid,nodev - ext4')
    if change=='nodev': mounts=mounts.replace('/scratch rw,nosuid,nodev','/scratch rw,nosuid')
    if change=='nosuid': mounts=mounts.replace('/scratch rw,nosuid,nodev','/scratch rw,nodev')
    if change=='noexec': mounts=mounts.replace('/scratch rw,nosuid,nodev','/scratch rw,nosuid,nodev,noexec')
    if change=='size': cap+=4096
    if change=='root': mounts=mounts.replace('/ / ro,','/ / rw,')
    if change=='dev': mounts=mounts.replace('/dev ro,','/dev rw,')
    if change=='extra': mounts+='\n5 1 0:5 / /extra rw - ext4 disk rw'
    with pytest.raises(ValueError): verify_mounts(mounts,cap,8388608,4096)


@pytest.mark.parametrize('attack',['duplicate','truncated','trailing','oversized','wrong-head','unexpected'])
def test_parent_independently_rejects_export_framing(tmp_path,monkeypatch,attack):
    import importlib,json,struct
    from coding_review_agent_loop.review_images_channel import write_record
    from review_images_helpers import manifest,png
    service=importlib.import_module('coding_review_agent_loop.review_images_capture')
    root,head,git=repository(tmp_path)
    document=manifest(head if attack!='wrong-head' else 'b'*40)
    stream=io.BytesIO()
    raw=json.dumps(document).encode()
    write_record(stream,{'path':'manifest.json','size':len(raw)},raw)
    write_record(stream,{'path':'one.png','size':len(png())},png())
    write_record(stream,{'path':'base.png','size':len(png())},png())
    if attack=='duplicate': write_record(stream,{'path':'one.png','size':len(png())},png())
    if attack=='unexpected': write_record(stream,{'path':'other.png','size':len(png())},png())
    write_record(stream,{'complete':True})
    data=stream.getvalue()
    if attack=='truncated': data=data[:-2]
    if attack=='trailing': data+=b'x'
    if attack=='oversized': data=struct.pack('>I',8193)
    monkeypatch.setattr(service,'_attempt',lambda *args: ({'state':'ok','proof':{}},data))
    monkeypatch.setattr('coding_review_agent_loop.review_images_isolation.preflight',lambda _: '/usr/bin/bwrap')
    result=capture(root,head,config(),[{'status':'modified','filename':'capture.py'}])
    assert result.state=='failed' and result.snapshots is None


def test_live_validator_corruption_and_whole_set_deadline(tmp_path,monkeypatch):
    import importlib
    service=importlib.import_module('coding_review_agent_loop.review_images_capture')
    root,head,git=repository(tmp_path)
    supported(root,head)
    real=service._attempt
    def corrupt(config,settings,source):
        if settings['mode']=='validate':
            import tempfile
            with tempfile.TemporaryFile() as broken:
                broken.write(b'bad framing')
                broken.seek(0)
                return real(config,settings,broken)
        return real(config,settings,source)
    monkeypatch.setattr(service,'_attempt',corrupt)
    result=capture(root,head,config(),[{'status':'modified','filename':'capture.py'}])
    assert result.state=='failed' and result.snapshots is None,result
    monkeypatch.setattr(service,'_attempt',real)
    result=capture(root,head,dataclasses.replace(config(),validation_timeout=.001),[{'status':'modified','filename':'capture.py'}])
    assert result.state=='timeout' and result.snapshots is None,result


def test_real_supervisor_death_aborts_export(tmp_path,monkeypatch):
    import concurrent.futures
    import signal
    import time
    from coding_review_agent_loop.containment import InvocationHandle
    root,head,git=repository(tmp_path)
    supported(root,head)
    ready=tmp_path/'supervisor.pid'
    # Instrument the trusted host monitor, keeping the actual namespace,
    # SCM_CREDENTIALS handshake, delegated scope and kill path intact.
    script='''import sys
from pathlib import Path
from coding_review_agent_loop import review_images_host as host
real=host.release_launcher
def release(pid, supervisor, phase, tree, channel):
 result=real(pid,supervisor,phase,tree,channel)
 if phase=='capture': Path(READY).write_text(str(supervisor))
 return result
host.release_launcher=release
raise SystemExit(host.main())
'''.replace('READY',repr(str(ready)))
    prepare=InvocationHandle.prepare
    def instrument(policy,**kwargs):
        target=kwargs['target_argv']
        kwargs['target_argv']=[target[0],'-c',script,*target[3:]]
        return prepare(policy,**kwargs)
    monkeypatch.setattr(InvocationHandle,'prepare',instrument)
    altered=dataclasses.replace(config(),command='/usr/bin/python3 capture.py; sleep 5',capture_timeout=8)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future=executor.submit(capture,root,head,altered,[{'status':'modified','filename':'capture.py'}])
        deadline=time.monotonic()+15
        while not ready.exists() and not future.done() and time.monotonic()<deadline:
            time.sleep(.02)
        assert ready.exists()
        os.kill(int(ready.read_text()),signal.SIGKILL)
        result=future.result(timeout=15)
    assert result.state=='failed' and result.snapshots is None,result


def test_live_confined_input_loader_rejects_truncation(tmp_path,monkeypatch):
    import importlib,tempfile
    service=importlib.import_module('coding_review_agent_loop.review_images_capture')
    root,head,git=repository(tmp_path)
    supported(root,head)
    real=service._attempt
    def corrupt(config,settings,source):
        with tempfile.TemporaryFile() as broken:
            broken.write(b'bad')
            broken.seek(0)
            return real(config,settings,broken)
    monkeypatch.setattr(service,'_attempt',corrupt)
    result=capture(root,head,config(),[{'status':'modified','filename':'capture.py'}])
    assert result.state=='failed' and result.snapshots is None,result
    assert 'truncated transfer' in result.diagnostic


def test_live_validator_address_space_budget_discards_set(tmp_path,monkeypatch):
    import importlib,tempfile
    from coding_review_agent_loop.review_images_channel import write_record
    service=importlib.import_module('coding_review_agent_loop.review_images_capture')
    root,head,git=repository(tmp_path)
    supported(root,head)
    real=service._attempt
    def exhaust(config,settings,source):
        if settings['mode']!='validate': return real(config,settings,source)
        settings={**settings,'limits':{**settings['limits'],'file_bytes':128*1024**2,'total_bytes':128*1024**2}}
        with tempfile.TemporaryFile() as oversized:
            write_record(oversized,{'path':'large.png','size':128*1024**2})
            block=bytes(65536)
            for _ in range(2048): oversized.write(block)
            write_record(oversized,{'complete':True})
            oversized.seek(0)
            return real(config,settings,oversized)
    monkeypatch.setattr(service,'_attempt',exhaust)
    result=capture(root,head,config(),[{'status':'modified','filename':'capture.py'}])
    assert result.state in {'failed','resource-limit'} and result.snapshots is None,result
    assert 'MemoryError' in result.diagnostic or result.state=='resource-limit',result


def test_staging_cleanup_refuses_foreign_replacement(tmp_path,monkeypatch):
    from coding_review_agent_loop import review_images_worktree as worktree
    attempt=tmp_path/'attempt'
    def allocate(**kwargs):
        attempt.mkdir()
        return str(attempt)
    monkeypatch.setattr(worktree.tempfile,'mkdtemp',allocate)
    with pytest.raises(ValueError,match='ownership'):
        with worktree.owned_staging() as directory:
            directory.rename(tmp_path/'original')
            directory.mkdir()
            (directory/'foreign').write_text('keep')
    assert (attempt/'foreign').read_text()=='keep'


def test_held_output_root_cannot_be_replaced(tmp_path):
    root,head,git=repository(tmp_path)
    supported(root,head)
    command='mv /scratch/output /scratch/old-output; mkdir /scratch/output; /usr/bin/python3 capture.py'
    result=capture(root,head,dataclasses.replace(config(),command=command),[{'status':'modified','filename':'capture.py'}])
    assert result.state=='failed' and result.snapshots is None,result
    assert 'held output root was replaced' in result.diagnostic


def test_payload_cannot_add_unbounded_storage_mounts(tmp_path):
    root,head,git=repository(tmp_path)
    supported(root,head)
    command='mkdir -p /scratch/extra; if unshare -Ur -m /bin/sh -c "mount -t tmpfs tmpfs /scratch/extra"; then exit 81; fi; /usr/bin/python3 capture.py'
    result=required(capture(root,head,dataclasses.replace(config(),command=command),[{'status':'modified','filename':'capture.py'}]))
    assert result.isolation_proof[0]['nested_userns_disabled']
    result.snapshots.close()
