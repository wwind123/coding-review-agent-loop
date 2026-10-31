"""Real PNG and repository fixtures; no image-library dependencies."""
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import zlib


def chunk(kind, data=b''):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(data, zlib.crc32(kind)))


def png(width=2, height=2, depth=8, color=6, raw=None, compressed=None, before=b'', after=b'', interlace=0):
    samples = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color, 1)
    if raw is None:
        raw = bytes(height * (1 + (width * samples * depth + 7) // 8))
    header = struct.pack('>IIBBBBB', width, height, depth, color, 0, 0, interlace)
    if color == 3 and not before:
        before = chunk(b'PLTE', b'\0\0\0')
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header) + before + chunk(b'IDAT', compressed if compressed is not None else zlib.compress(raw)) + after + chunk(b'IEND')


def manifest(head='a' * 40):
    return {'head_sha': head, 'images': [{'path': 'one.png', 'id': 'main', 'viewport': {'width': 2, 'height': 2},
                                        'locale': 'en', 'state': 'initial', 'baseline_path': 'base.png', 'diff_score': 0.2}]}


def repository(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), '-c', 'user.name=Capture Test',
                                        '-c', 'user.email=capture@example.invalid', *args], stdin=subprocess.DEVNULL).decode().strip()
    git('init', '-q')
    script = '''import json,os,struct,zlib
from pathlib import Path
assert Path('dependency').read_text() == os.environ['AGENT_LOOP_REVIEW_IMAGES_HEAD_SHA']
assert not Path('.git').exists()
assert not os.environ.get('SSH_AUTH_SOCK')
assert not os.environ.get('HOST_SECRET')
def chunk(k,d=b''):
 return struct.pack('>I',len(d))+k+d+struct.pack('>I',zlib.crc32(d,zlib.crc32(k)))
p=b'\\x89PNG\\r\\n\\x1a\\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',2,2,8,6,0,0,0))+chunk(b'IDAT',zlib.compress(bytes(18)))+chunk(b'IEND')
out=Path(os.environ['AGENT_LOOP_REVIEW_IMAGES_MANIFEST'])
(out.parent/'one.png').write_bytes(p)
(out.parent/'base.png').write_bytes(p)
out.write_text(json.dumps({'head_sha':os.environ['AGENT_LOOP_REVIEW_IMAGES_HEAD_SHA'],'images':[{'path':'one.png','id':'main','viewport':{'width':2,'height':2},'locale':'en','state':'ready','baseline_path':'base.png'}]}))
'''
    (root / 'capture.py').write_text(script)
    git('add', '.')
    git('commit', '-qm', 'capture fixture')
    return root, git('rev-parse', 'HEAD'), git


def stub_capture_prerequisites(monkeypatch):
    """Let service unit tests reach their branch without host prerequisites."""
    from coding_review_agent_loop import containment, review_images_isolation
    calls = []
    def containment_ready(_):
        calls.append('containment')
        return containment.CapabilityManifest('systemd-cgroup-v2', True)
    def bubblewrap_ready(_):
        calls.append('bubblewrap')
        return '/mock-runtime/bwrap'
    def validator_ready(_):
        calls.append('validator')
        return ()
    monkeypatch.setattr(containment, 'preflight_containment', containment_ready)
    monkeypatch.setattr(review_images_isolation, 'preflight', bubblewrap_ready)
    monkeypatch.setattr(review_images_isolation, 'validator_grants', validator_ready)
    return calls
