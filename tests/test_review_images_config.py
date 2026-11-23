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
def test_public_options_accepted(mode, option):
    from coding_review_agent_loop.cli import build_parser
    parser = build_parser()
    value = '1' if option in {'timeout', 'max-images', 'max-bytes', 'max-pixels'} else 'value'
    args = parser.parse_args([mode, '1793', '--review-images-' + option, value])
    assert getattr(args, 'review_images_' + option.replace('-', '_')) is not None


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


@pytest.mark.parametrize('name,value', [('review_images_timeout', 0), ('review_images_timeout', float('nan')),
    ('review_images_timeout', 1801), ('review_images_max_images', 17), ('review_images_max_images', 1.5),
    ('review_images_max_bytes', True), ('review_images_max_pixels', 64000001),
    ('review_images_runtime_grant', ()), ('review_images_runtime_grant', ('/tmp',)),
    ('review_images_runtime_grant', ('/usr/bin', '/usr/bin/python3'))])
def test_enabled_public_settings_refuse_before_capture(tmp_path, name, value):
    from dataclasses import replace
    from agent_loop_helpers import make_config
    from coding_review_agent_loop.review_images import settings
    config = make_config(tmp_path, review_images_paths=('ui/*',), review_images_cmd='capture',
                         review_images_runtime_grant=('/usr/bin/python3',))
    with pytest.raises((ValueError, OSError)):
        settings(replace(config, **{name: value}))


def test_disabled_lifecycle_never_imports_linux_transport():
    code = '''import builtins,sys
from types import SimpleNamespace
original=builtins.__import__
def guarded(name,*args,**kwargs):
 if name in {'review_images_transport','review_images_manifest','review_images_isolation','review_images_png'}:
  raise AssertionError('disabled lifecycle imported image transport or isolation')
 return original(name,*args,**kwargs)
builtins.__import__=guarded
from coding_review_agent_loop.review_images import startup,VisualRun
config=SimpleNamespace(review_images_paths=(),dry_run=False)
startup(config)
result=VisualRun().attempt(None,config=config,metadata=None,checkout=None,items=(),managed=False,current_head=None,diagnostic=None)
assert result.diagnostic=='disabled'
assert 'coding_review_agent_loop.review_images_transport' not in sys.modules
'''
    subprocess.run([sys.executable,'-c',code],stdin=subprocess.DEVNULL,check=True,timeout=10,
                   env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src')})


# --- #2070: run-time opt-in presence detection and predicate ---

REVIEW_IMAGES_OPTIONS = [
    ('--review-images-cmd', '/usr/bin/python3 capture.py'),
    ('--review-images-paths', 'ui/*'),
    ('--review-images-seat', 'codex'),
    ('--review-images-runtime-grant', '/usr/bin/python3'),
    ('--review-images-timeout', '300'),
    ('--review-images-max-images', '16'),
    ('--review-images-max-bytes', '67108864'),
    ('--review-images-max-pixels', '64000000'),
]
RESOLVED_DEFAULTS = {'review_images_timeout': 300, 'review_images_max_images': 16,
                     'review_images_max_bytes': 67108864, 'review_images_max_pixels': 64000000}


def _parse(tmp_path, *extra):
    from coding_review_agent_loop.cli import build_parser
    return build_parser().parse_args(['pr', '77', '--repo', 'OWNER/REPO', '--codex-dir', str(tmp_path / 'codex'),
                                      '--claude-dir', str(tmp_path / 'claude'), '--dangerous-agent-permissions', *extra])


def _config(tmp_path, *extra):
    from agent_loop_helpers import FakeRunner
    from coding_review_agent_loop.config import config_from_args
    return config_from_args(_parse(tmp_path, *extra), FakeRunner())


@pytest.mark.parametrize('option,value', REVIEW_IMAGES_OPTIONS)
def test_any_single_review_images_option_opts_in(tmp_path, option, value):
    from coding_review_agent_loop.config import review_images_requested_from_args
    from coding_review_agent_loop.review_images import visual_review_requested
    assert review_images_requested_from_args(_parse(tmp_path, option, value))
    config = _config(tmp_path, option, value)
    assert config.review_images_requested and visual_review_requested(config)


def test_no_review_images_option_is_inert_and_keeps_resolved_defaults(tmp_path):
    from coding_review_agent_loop.config import review_images_requested_from_args
    from coding_review_agent_loop.review_images import visual_review_requested
    args = _parse(tmp_path)
    assert args.review_images_timeout is None and args.review_images_max_images is None
    assert not review_images_requested_from_args(args)
    config = _config(tmp_path)
    assert not config.review_images_requested and not visual_review_requested(config)
    assert {name: getattr(config, name) for name in RESOLVED_DEFAULTS} == RESOLVED_DEFAULTS
    explicit = _config(tmp_path, '--review-images-timeout', '7', '--review-images-max-images', '3')
    assert (explicit.review_images_timeout, explicit.review_images_max_images) == (7, 3)


@pytest.mark.parametrize('field,value,expected', [
    (None, None, False), ('review_images_requested', True, True), ('review_images_cmd', 'capture', True),
    ('review_images_paths', ('ui/*',), True), ('review_images_seat', 'codex', True),
    ('review_images_runtime_grant', ('/usr/bin/python3',), True)])
def test_visual_review_requested_truth_table(tmp_path, field, value, expected):
    from dataclasses import replace
    from agent_loop_helpers import make_config
    from coding_review_agent_loop.review_images import visual_review_requested
    config = make_config(tmp_path)
    assert not visual_review_requested(config)
    if field is not None:
        config = replace(config, **{field: value})
    assert visual_review_requested(config) is expected


@pytest.mark.parametrize('opted_in', [False, True])
def test_drop_visual_records_and_items_only_when_not_requested(tmp_path, opted_in):
    from dataclasses import replace
    from agent_loop_helpers import make_config
    from coding_review_agent_loop.review_images import drop_visual_items, drop_visual_records
    from coding_review_agent_loop.round_metadata import PostedRoundMetadata, PostedRoundRecord
    from coding_review_agent_loop.unresolved_items import _next_unresolved_item
    config = replace(make_config(tmp_path), review_images_requested=opted_in)
    ordinary = _next_unresolved_item(item_number=1, reviewer='Codex', source_round=1, text='ordinary', status='blocking')
    visual_item = replace(_next_unresolved_item(item_number=2, reviewer='Codex', source_round=1, text='visual',
                                                status='blocking'), origin_scope='visible-defects')
    items = [ordinary, visual_item]
    assert drop_visual_items(items, config) == (items if opted_in else [ordinary])
    assert drop_visual_items(tuple(items), config) == (tuple(items) if opted_in else (ordinary,))
    plain = PostedRoundRecord(0, PostedRoundMetadata(flow='pr', role='reviewer', agent='Codex', round_number=1,
                                                     subject='a' * 40), '')
    scoped = PostedRoundRecord(1, PostedRoundMetadata(flow='pr', role='reviewer', agent='Codex', round_number=1,
        subject='a' * 40, review_scope={'version': 1, 'kind': 'visible-defects', 'head': 'a' * 40, 'seat': 'Codex'}), '')
    lifecycle = PostedRoundRecord(2, PostedRoundMetadata(flow='pr', role='visual-intent', agent='Codex', round_number=1,
        subject='a' * 40, visual_lifecycle={'version': 99}), '')
    records = (plain, scoped, lifecycle)
    assert drop_visual_records(records, config) == (records if opted_in else (plain,))


@pytest.mark.parametrize('opted_in', [False, True])
def test_mask_visual_protocol_bodies_only_masks_visual_records(tmp_path, opted_in):
    from dataclasses import replace
    from agent_loop_helpers import make_config
    from coding_review_agent_loop.review_images import mask_visual_protocol_bodies
    from coding_review_agent_loop.round_metadata import PostedRoundMetadata, _attach_round_metadata
    from coding_review_agent_loop.unresolved_items import _next_unresolved_item
    config = replace(make_config(tmp_path), review_images_requested=opted_in)
    item = replace(_next_unresolved_item(item_number=5, reviewer='Codex', source_round=1, text='visual',
                                         status='blocking'), origin_scope='visible-defects')
    visual = _attach_round_metadata('Visual assessment.', PostedRoundMetadata(
        flow='pr', role='reviewer', agent='Codex', round_number=1, subject='a' * 40, new_items=(item,),
        review_scope={'version': 1, 'kind': 'visible-defects', 'head': 'a' * 40, 'seat': 'Codex'}))
    ordinary = _attach_round_metadata('Ordinary review.', PostedRoundMetadata(
        flow='pr', role='reviewer', agent='Codex', round_number=1, subject='a' * 40))
    bodies = (visual, ordinary, 'plain prose', None, '<!-- AGENT_LOOP_META: !!!not-base64 -->')
    masked, reserves = mask_visual_protocol_bodies(bodies, config)
    if opted_in:
        assert masked == bodies and reserves == (0,) * 5
    else:
        assert masked[0] == 'Visual assessment.' and masked[1:] == bodies[1:]
        assert reserves == (5, 0, 0, 0, 0)
