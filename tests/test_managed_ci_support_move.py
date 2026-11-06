"""Phase-five support ownership, body parity and shared diagnostics."""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from coding_review_agent_loop import managed_ci, managed_ci_api
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.runner import CommandResult
from agent_loop_helpers import make_config
from managed_ci_call_guard import managed_ci_source_paths

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'src/coding_review_agent_loop'
MANIFEST = json.loads((ROOT / 'tests/fixtures/managed_ci_support_move.json').read_text())


@pytest.mark.parametrize('path', managed_ci_source_paths(), ids=lambda path: path.stem)
@pytest.mark.parametrize('guard_module, guard_name, violation', [
    ('test_github_retry', 'test_run_gh_read_call_sites_are_read_only',
     'run_gh_read(runner, ["gh", "api", "--method", "POST"], cwd=path)'),
    ('test_managed_ci', 'test_managed_ci_gh_invocations_obey_the_245_floor',
     'runner.run([gh_cmd, "api", "--unsupported-api-flag"])'),
])
def test_gh_call_guards_reject_violations_in_every_owner(
    monkeypatch, path, guard_module, guard_name, violation,
):
    guard = getattr(importlib.import_module(guard_module), guard_name)
    original_read_text = Path.read_text

    def read_text(candidate, *args, **kwargs):
        if candidate == path:
            return violation
        return original_read_text(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read_text)
    with pytest.raises(AssertionError):
        guard()


def test_support_bodies_and_dependencies():
    for owner, records in MANIFEST['owners'].items():
        module = importlib.import_module('coding_review_agent_loop.' + owner)
        source = (PACKAGE / (owner + '.py')).read_text()
        lines = source.splitlines(keepends=True)
        definitions = {}
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                definitions[node.name] = node
            elif isinstance(node, ast.Assign):
                definitions.update((t.id, node) for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign):
                definitions[node.target.id] = node
        for name, record in records.items():
            assert getattr(module, name) is not None, name
            node = definitions[name]
            first = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
            body = ''.join(lines[first - 1:node.end_lineno])
            assert hashlib.sha256(body.encode()).hexdigest() == record['sha256'], name
        assert len(source.splitlines()) <= 6000
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in ('managed_ci', 'managed_ci_authorization')
                assert not any(a.name in ('managed_ci', 'managed_ci_authorization') for a in node.names)
            elif isinstance(node, ast.Import):
                assert not any(a.name in ('coding_review_agent_loop.managed_ci',
                                          'coding_review_agent_loop.managed_ci_authorization') for a in node.names)


@pytest.mark.parametrize('first', ['managed_ci', *MANIFEST['owners']])
def test_support_fresh_import_orders(first):
    modules = [first] + [m for m in ['managed_ci', *MANIFEST['owners']] if m != first]
    code = '\n'.join('import coding_review_agent_loop.' + m for m in modules)
    code += '\nfrom coding_review_agent_loop import managed_ci, managed_ci_api\n'
    code += 'assert managed_ci.activate_managed_ci.__globals__["_LIST_FAILURE"] is managed_ci_api._LIST_FAILURE\n'
    subprocess.run([sys.executable, '-c', code], check=True, timeout=30, cwd=ROOT,
                   env={**os.environ, 'PYTHONPATH': str(PACKAGE.parent)})


@pytest.mark.parametrize('payload', [None, '{}', '[null]'])
def test_retained_gate_and_extracted_read_share_failure_state(tmp_path, payload):
    config = make_config(tmp_path)

    class ReadRunner:
        def __init__(self):
            self.calls = []

        def run(self, command, **kwargs):
            self.calls.append(command)
            return CommandResult(tuple(command), kwargs['cwd'],
                                 '' if payload is None else payload,
                                 'gh: forbidden (HTTP 403)' if payload is None else '',
                                 1 if payload is None else 0)

    runner = ReadRunner()

    @managed_ci_api._surfaces_list_failures
    def retained_gate():
        assert managed_ci_api._active_managed_label_event(
            runner, config=config, pr_number=7) is None
        reason = managed_ci_api._last_list_failure()
        # A later successful read must not erase an earlier failure in this gate.
        class SuccessRunner:
            def run(self, command, **kwargs):
                return CommandResult(tuple(command), kwargs['cwd'], '[]', '', 0)
        assert managed_ci_api._api_list(SuccessRunner(), config, 'repos/OWNER/REPO/issues') == []
        assert managed_ci_api._last_list_failure() == ''
        raise AgentLoopError('gate refused: ' + str(bool(reason)))

    @managed_ci_api._surfaces_list_failures
    def extracted_gate():
        retained_gate()

    with pytest.raises(AgentLoopError) as caught:
        extracted_gate()
    message = str(caught.value)
    assert 'gate refused: True' in message
    assert message.count('GitHub list read failure:') == 1
    assert ('HTTP 403' if payload is None else 'non-list payload' if payload == '{}' else 'malformed entries') in message
    assert all(command[1] == 'api' and '--method' not in command for command in runner.calls)
    assert managed_ci_api._LIST_FAILURE.depth == 0
    assert managed_ci_api._LIST_FAILURE.failures is None
