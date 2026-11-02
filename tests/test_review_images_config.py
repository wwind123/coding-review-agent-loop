import subprocess
import os
from pathlib import Path
import sys

import pytest
from coding_review_agent_loop.review_images_config import CaptureConfig, ImageLimits, eligible
from coding_review_agent_loop.review_images_capture import capture


def test_disabled_without_worker_or_io(tmp_path):
    code = """import sys
from pathlib import Path
from coding_review_agent_loop.review_images_capture import capture
from coding_review_agent_loop.review_images_config import CaptureConfig
assert capture(Path('/unavailable'), 'invalid', CaptureConfig(), []).state == 'disabled'
assert 'coding_review_agent_loop.review_images_worker' not in sys.modules
assert 'coding_review_agent_loop.review_images_png' not in sys.modules
assert 'coding_review_agent_loop.review_images_isolation' not in sys.modules
"""
    subprocess.run([sys.executable, '-c', code], stdin=subprocess.DEVNULL, check=True, timeout=10, env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')})


@pytest.mark.parametrize('status', ['added', 'modified', 'renamed', 'removed', 'deleted', 'copied', 'changed'])
def test_changed_path_matching(status):
    config = CaptureConfig(paths=('ui/*.tsx',), command='capture')
    record = {'status': status, 'filename': 'ui/view.tsx'}
    if status == 'renamed':
        record.update(filename='backend/view.py', previous_filename='ui/view.tsx')
    assert eligible(config, [record])
    assert not eligible(config, [{'status': 'modified', 'filename': 'UI/view.tsx'}])
    assert not eligible(CaptureConfig(), [record])


@pytest.mark.parametrize('path', ['/etc/passwd', '../x', 'a/../x', './x', 'a//x', '.git/config', 'a\x00x'])
def test_unsafe_paths_rejected(path):
    with pytest.raises(ValueError):
        eligible(CaptureConfig(), [{'status': 'added', 'filename': path}])


@pytest.mark.parametrize('name', ['C:x', 'a\\x', 'a\nx', 'a\u202ex'])
def test_literal_git_names_only_refuse_when_matching(name):
    changes = [{'status': 'modified', 'filename': name}]
    assert not eligible(CaptureConfig(paths=('ui/*.tsx',), command='capture'), changes)
    assert not eligible(CaptureConfig(), changes)
    with pytest.raises(ValueError, match='unsafe repository-relative'):
        eligible(CaptureConfig(paths=('*',), command='capture'), changes)


@pytest.mark.parametrize('status', [None, [], {}, True, 'unmerged'])
def test_malformed_status_is_unavailable_not_nonmatching(status):
    with pytest.raises(ValueError, match='status'):
        eligible(CaptureConfig(), [{'status': status, 'filename': 'other.py'}])


def test_delivery_policy_has_explicit_bounds_without_clamping():
    from dataclasses import replace
    from coding_review_agent_loop.review_images_config import delivery_limits
    limits = delivery_limits()
    assert (limits.entries, limits.file_bytes, limits.dimension, limits.pixels,
            limits.viewport_dimension) == (16, 2 * 1024**2, 768, 589824, 16000)
    assert delivery_limits(replace(limits, entries=1)).entries == 1
    for name in vars(limits):
        with pytest.raises(ValueError):
            delivery_limits(replace(limits, **{name: getattr(limits, name) + 1}))
    with pytest.raises(ValueError):
        delivery_limits(ImageLimits())


@pytest.mark.parametrize('field', ['setup_timeout', 'capture_timeout', 'transfer_timeout', 'validation_timeout', 'scratch_bytes', 'tasks', 'cpu_percent'])
@pytest.mark.parametrize('value', [0, -1, True, float('inf'), float('nan')])
def test_limits_require_positive_finite(field, value):
    with pytest.raises(ValueError):
        CaptureConfig(**{field: value})


@pytest.mark.parametrize('mode', ['issue', 'pr'])
@pytest.mark.parametrize('option', ['cmd', 'paths', 'seat', 'timeout', 'max-images', 'max-bytes', 'max-pixels'])
def test_public_options_rejected(mode, option):
    from coding_review_agent_loop.cli import build_parser
    parser = build_parser()
    with pytest.raises(SystemExit) as error:
        parser.parse_args([mode, '1793', '--review-images-' + option, 'value'])
    assert error.value.code == 2


@pytest.mark.parametrize('platform',['win32','darwin'])
def test_unsupported_platform_refuses_before_linux_imports_and_probes(monkeypatch,platform):
    import importlib
    service=importlib.import_module('coding_review_agent_loop.review_images_capture')
    import builtins
    real_import=builtins.__import__
    def guarded_import(name,*args,**kwargs):
        if name in {'review_images_isolation','review_images_manifest','review_images_worktree'}:
            pytest.fail('unsupported platform reached Linux imports')
        return real_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded_import)
    monkeypatch.setattr(service.sys,'platform',platform)
    result=capture(Path('/unavailable'),'a'*40,CaptureConfig(paths=('*.py',),command='capture'),
                   [{'status':'added','filename':'view.py'}])
    assert result.state=='isolation-unavailable' and result.snapshots is None


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux host namespace metadata')
def test_missing_host_namespace_metadata_refuses_before_staging(monkeypatch):
    import importlib
    from review_images_helpers import stub_capture_prerequisites
    worktree=importlib.import_module('coding_review_agent_loop.review_images_worktree')
    calls=stub_capture_prerequisites(monkeypatch)
    namespaces=[]
    monkeypatch.setattr(worktree,'stage',lambda *_:pytest.fail('namespace refusal reached staging'))
    real_readlink=os.readlink
    def readlink(path,*args,**kwargs):
        if str(path).startswith('/proc/self/ns/'):
            namespaces.append(str(path))
            raise OSError('host namespace metadata unavailable')
        return real_readlink(path,*args,**kwargs)
    monkeypatch.setattr(os,'readlink',readlink)
    result=capture(Path('/unavailable'),'a'*40,CaptureConfig(paths=('*.py',),command='capture'),
                   [{'status':'added','filename':'view.py'}])
    assert result.state=='isolation-unavailable' and 'namespace' in result.diagnostic
    assert calls==['containment','bubblewrap','validator']
    assert namespaces==['/proc/self/ns/user']
