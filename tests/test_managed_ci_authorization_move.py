"""Authorization body parity, final ownership and acyclic import order."""
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

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'src/coding_review_agent_loop'
MANIFEST = json.loads((ROOT / 'tests/fixtures/managed_ci_authorization_move.json').read_text())


def test_authorization_bodies_and_dependencies():
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
            elif isinstance(node, ast.ImportFrom):
                definitions.update((a.asname or a.name, node) for a in node.names)
        for name, record in records.items():
            assert getattr(module, name) is not None, name
            node = definitions[name]
            first = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
            body = ''.join(lines[first - 1:node.end_lineno])
            assert hashlib.sha256(body.encode()).hexdigest() == record['sha256'], name
        assert len(source.splitlines()) <= 6000
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in ('managed_ci', 'coding_review_agent_loop.managed_ci')
                assert not any(a.name == 'managed_ci' for a in node.names)
            elif isinstance(node, ast.Import):
                assert not any(a.name == 'coding_review_agent_loop.managed_ci' for a in node.names)


@pytest.mark.parametrize('first', [
    'managed_ci', 'managed_ci_authorization', 'managed_ci_api', 'managed_ci_protection',
])
def test_authorization_fresh_import_orders(first):
    modules = [first] + [m for m in ['managed_ci', 'managed_ci_authorization',
                                    'managed_ci_api', 'managed_ci_protection'] if m != first]
    code = '\n'.join('import coding_review_agent_loop.' + m for m in modules)
    code += '\nfrom coding_review_agent_loop import managed_ci, managed_ci_authorization, managed_ci_api\n'
    code += 'assert managed_ci_authorization.recover_issue_created_handoff.__globals__["_LIST_FAILURE"] is managed_ci_api._LIST_FAILURE\n'
    subprocess.run([sys.executable, '-c', code], check=True, timeout=30, cwd=ROOT,
                   env={**os.environ, 'PYTHONPATH': str(PACKAGE.parent)})


def test_patch_rewrite_targets_new_lookup_and_is_idempotent():
    import runpy

    rewrite = runpy.run_path(str(ROOT / 'scripts/rewrite_managed_authorization_patches.py'))['rewrite']
    source = ('from coding_review_agent_loop import managed_ci\n'
              'monkeypatch.setattr(managed_ci, "_github_proves_descendant", fake)\n'
              'patch.object(managed_ci, "_github_proves_descendant", fake)\n'
              'monkeypatch.setattr(managed_ci, "_ensure_v2_intent", fake)\n'
              'value = managed_ci.recover_issue_created_handoff\n')
    owners = {name: owner for owner, records in MANIFEST['owners'].items() for name in records}
    rewritten = rewrite(source, owners)
    assert 'monkeypatch.setattr(managed_ci_authorization, "_github_proves_descendant", fake)' in rewritten
    assert 'patch.object(managed_ci_authorization, "_github_proves_descendant", fake)' in rewritten
    assert 'monkeypatch.setattr(managed_ci, "_ensure_v2_intent", fake)' in rewritten
    assert 'value = managed_ci.recover_issue_created_handoff' in rewritten
    assert rewrite(rewritten, owners) == rewritten
