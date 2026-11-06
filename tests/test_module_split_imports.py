"""Symbol-specific guards for cleaned protocol and managed-CI support surfaces."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location('support_rewrite', ROOT / 'scripts/rewrite_protocol_support_imports.py')
_REWRITE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_REWRITE)
_RISK_SPEC = importlib.util.spec_from_file_location('risk_rewrite', ROOT / 'scripts/rewrite_protocol_risk_imports.py')
_RISK_REWRITE = importlib.util.module_from_spec(_RISK_SPEC)
_RISK_SPEC.loader.exec_module(_RISK_REWRITE)
RISK_OWNERS = _RISK_REWRITE.symbol_owners()
OWNERS = {**_REWRITE.symbol_owners(), **RISK_OWNERS}
OLD = 'coding_review_agent_loop.protocol'


def ownership_problems(source, *, driver=False, old=OLD, owners=OWNERS):
    tree = ast.parse(source)
    # Quoted forward references are executable type dependencies too. Expand
    # annotation strings (including nested tuple/list references), but never
    # count an unrelated string literal as an internal use of an import.
    nodes = list(ast.walk(tree))
    for node in list(nodes):
        annotations = []
        if isinstance(node, (ast.AnnAssign, ast.arg)):
            annotations.append(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotations.append(node.returns)
        for annotation in annotations:
            pending = list(ast.walk(annotation)) if annotation is not None else []
            while pending:
                child = pending.pop()
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    try:
                        expanded = ast.parse(child.value, mode='eval')
                    except SyntaxError:
                        continue
                    expanded_nodes = list(ast.walk(expanded))
                    for expanded_node in expanded_nodes:
                        if hasattr(expanded_node, 'lineno'):
                            expanded_node.lineno = child.lineno
                    nodes.extend(expanded_nodes)
                    pending.extend(expanded_nodes)
    bindings = {}
    problems = []

    def resolve(node):
        if isinstance(node, ast.Name):
            return bindings.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return resolve(node.value) + '.' + node.attr
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            return ''.join(resolve(value.value) if isinstance(value, ast.FormattedValue)
                           else resolve(value) for value in node.values)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return resolve(node.left) + resolve(node.right)
        return ''

    # Source order also handles local imports, module aliases and patch-prefix variables.
    for node in sorted(nodes, key=lambda n: (getattr(n, 'lineno', 0), getattr(n, 'col_offset', 0))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            if node.level and module == old.rsplit('.', 1)[1]:
                module = old
            elif node.level and not module:
                module = 'coding_review_agent_loop'
            for alias in node.names:
                bindings[alias.asname or alias.name] = module + '.' + alias.name
                if module == old and (alias.name in owners or alias.name == '*'):
                    problems.append(f'{node.lineno}: old import {alias.name}')
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = resolve(node.value)
        if isinstance(node, (ast.Attribute, ast.Constant, ast.BinOp, ast.JoinedStr)):
            value = resolve(node)
            if value.startswith(old + '.') and value[len(old)+1:].split('.')[0] in owners:
                problems.append(f'{node.lineno}: old access or patch {value}')
        if isinstance(node, ast.Call) and (
            isinstance(node.func, ast.Attribute) and node.func.attr in ('setattr', 'object')
            or isinstance(node.func, ast.Name) and node.func.id in ('getattr', 'setattr')
        ):
            if len(node.args) >= 2 and resolve(node.args[0]) == old and resolve(node.args[1]) in owners:
                problems.append(f'{node.lineno}: old object patch')
    if driver:
        loads = {node.id for node in nodes if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in owners and (alias.asname or alias.name) not in loads:
                        problems.append(f'{node.lineno}: compatibility-only export {alias.name}')
    return problems


def test_consumers_use_protocol_support_owners():
    problems = []
    for folder in ('src', 'tests'):
        for path in (ROOT / folder).rglob('*.py'):
            problems.extend(f'{path.relative_to(ROOT)}: {problem}' for problem in
                            ownership_problems(path.read_text(), driver=path.name == 'protocol.py'))
    assert problems == []


@pytest.mark.parametrize('source', [
    'from coding_review_agent_loop.protocol import ParseDegradation',
    'from .protocol import ParseDegradation as PD',
    'from . import protocol as p\np.ParseDegradation',
    'from coding_review_agent_loop.protocol import *',
    'import coding_review_agent_loop.protocol as p\np.ParseDegradation',
    'from coding_review_agent_loop import protocol as p\np.parse_structured_discuss_answer',
    'import coding_review_agent_loop.protocol\ncoding_review_agent_loop.protocol.DiscussEvidenceClaim',
    'from coding_review_agent_loop import protocol as p\nq = p\nq.ParseDegradation',
    'patch("coding_review_agent_loop.protocol.ParseDegradation")',
    'prefix = "coding_review_agent_loop.protocol."\npatch(prefix + "ParseDegradation")',
    'prefix = "coding_review_agent_loop.protocol"\npatch(f"{prefix}.ParseDegradation")',
    'from coding_review_agent_loop import protocol as p\ngetattr(p, "ParseDegradation")',
    'from coding_review_agent_loop import protocol as p\nmonkeypatch.setattr(p, "ParseDegradation", fake)',
    'from coding_review_agent_loop import protocol as p\npatch.object(p, "ParseDegradation", fake)',
])
def test_guard_rejects_old_import_alias_access_and_patch(source):
    assert ownership_problems(source)


def test_guard_rejects_compatibility_only_export():
    assert ownership_problems('from .protocol_discuss import parse_structured_discuss_answer', driver=True)


def test_guard_allows_needed_internal_import_and_correct_owner():
    assert not ownership_problems('from .protocol_common import ParseDegradation\nx = ParseDegradation', driver=True)
    assert not ownership_problems('from coding_review_agent_loop.protocol_discuss import parse_structured_discuss_answer')


def test_rewrite_handles_aliases_local_imports_and_patch_strings():
    source = ('def consumer():\n    from coding_review_agent_loop.protocol import '
              'ParseDegradation as PD, parse_structured_discuss_answer as parse, ParsedPrReview\n'
              'patch("coding_review_agent_loop.protocol.parse_structured_discuss_answer")\n')
    rewritten = _REWRITE.rewrite(source, OWNERS)
    assert not ownership_problems(rewritten)
    assert 'ParseDegradation as PD' in rewritten
    assert 'ParsedPrReview' in rewritten
    assert _REWRITE.rewrite(rewritten, OWNERS) == rewritten


def test_discussion_dependency_patch_intercepts_real_lookup(monkeypatch):
    from coding_review_agent_loop import protocol_discuss

    calls = []

    def stop_at_lookup(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError('lookup intercepted')

    monkeypatch.setattr(protocol_discuss, '_extract_structured_discuss_review_payload', stop_at_lookup)
    with pytest.raises(RuntimeError, match='lookup intercepted'):
        protocol_discuss.parse_structured_discuss_answer('{}', reviewer='probe')
    assert len(calls) == 1


@pytest.mark.parametrize('name', sorted(RISK_OWNERS))
@pytest.mark.parametrize('template', [
    'from coding_review_agent_loop.protocol import {name} as moved',
    'from . import protocol as p\nq = p\nq.{name}',
    'from . import protocol as p\nvalue: "p.{name} | None"',
    'from . import protocol as p\nvalue: tuple["p.{name}", ...]',
    'import coding_review_agent_loop.protocol as p\np.{name}',
    'patch("coding_review_agent_loop.protocol.{name}")',
    'prefix = "coding_review_agent_loop.protocol"\npatch(f"{{prefix}}.{name}")',
    'from coding_review_agent_loop import protocol as p\nmonkeypatch.setattr(p, "{name}", fake)',
    'from coding_review_agent_loop import protocol as p\npatch.object(p, "{name}", fake)',
])
def test_guard_rejects_every_old_risk_symbol(name, template):
    assert ownership_problems(template.format(name=name))


@pytest.mark.parametrize('name', sorted(RISK_OWNERS))
def test_guard_rejects_risk_compatibility_export(name):
    assert ownership_problems(f'from .protocol_risk import {name}', driver=True)
    assert not ownership_problems(f'from .protocol_risk import {name}\nx = {name}', driver=True)


def test_risk_rewrite_handles_qualified_aliases_and_object_patches():
    source = ('from coding_review_agent_loop import protocol as p\n'
              'from coding_review_agent_loop.protocol import RiskTestMatrix as Matrix, ParsedPrReview\n'
              'p.risk_test_matrix_identity(matrix)\n'
              'monkeypatch.setattr(p, "_parse_risk_evidence_citations", fake)\n'
              'patch("coding_review_agent_loop.protocol.parse_risk_test_matrix")\n')
    rewritten = _RISK_REWRITE.rewrite(source, RISK_OWNERS)
    assert not ownership_problems(rewritten)
    assert 'RiskTestMatrix as Matrix' in rewritten
    assert 'ParsedPrReview' in rewritten
    assert _RISK_REWRITE.rewrite(rewritten, RISK_OWNERS) == rewritten


@pytest.mark.parametrize('annotation', [
    'value: {name}',
    'value: "{name} | None"',
    'value: tuple["{name}", ...]',
    'value: "tuple[\'{name}\', ...]"',
    'def consumer(value: "{name}"): pass',
    'def consumer() -> "{name}": pass',
    'async def consumer() -> "{name}": pass',
])
@pytest.mark.parametrize('aliased', [False, True])
@pytest.mark.parametrize('name', ['RiskTestMatrixEvidence', 'RiskTestMatrixChange'])
def test_guard_allows_annotation_dependencies(name, aliased, annotation):
    binding = 'MovedType' if aliased else name
    imported = f'{name} as {binding}' if aliased else name
    source = f'from .protocol_risk import {imported}\n' + annotation.format(name=binding)
    assert not ownership_problems(source, driver=True)


@pytest.mark.parametrize('name', ['RiskTestMatrixEvidence', 'RiskTestMatrixChange'])
def test_guard_does_not_count_unrelated_strings_as_annotation_dependencies(name):
    source = f'from .protocol_risk import {name}\nvalue = "{name}"'
    assert ownership_problems(source, driver=True)


_MANAGED_SPEC = importlib.util.spec_from_file_location('managed_rewrite', ROOT / 'scripts/rewrite_managed_support_imports.py')
_MANAGED_REWRITE = importlib.util.module_from_spec(_MANAGED_SPEC)
_MANAGED_SPEC.loader.exec_module(_MANAGED_REWRITE)
MANAGED_OWNERS = _MANAGED_REWRITE.symbol_owners()
MANAGED_OLD = 'coding_review_agent_loop.managed_ci'


def managed_ownership_problems(source, *, driver=False):
    return ownership_problems(source, driver=driver, old=MANAGED_OLD, owners=MANAGED_OWNERS)


def test_consumers_use_managed_support_owners():
    problems = []
    for folder in ('src', 'tests'):
        for path in (ROOT / folder).rglob('*.py'):
            problems.extend(f'{path.relative_to(ROOT)}: {problem}' for problem in
                            managed_ownership_problems(path.read_text(), driver=path.name == 'managed_ci.py'))
    assert problems == []


@pytest.mark.parametrize('name', sorted(MANAGED_OWNERS))
@pytest.mark.parametrize('template', [
    'from coding_review_agent_loop.managed_ci import {name} as moved',
    'from .managed_ci import {name}',
    'from . import managed_ci as ci\nq = ci\nq.{name}',
    'import coding_review_agent_loop.managed_ci as ci\nci.{name}',
    'import coding_review_agent_loop.managed_ci\ncoding_review_agent_loop.managed_ci.{name}',
    'from coding_review_agent_loop import managed_ci as ci\nvalue: "ci.{name} | None"',
    'patch("coding_review_agent_loop.managed_ci.{name}")',
    'prefix = "coding_review_agent_loop.managed_ci"\npatch(f"{{prefix}}.{name}")',
    'from coding_review_agent_loop import managed_ci as ci\nmonkeypatch.setattr(ci, "{name}", fake)',
    'from coding_review_agent_loop import managed_ci as ci\npatch.object(ci, "{name}", fake)',
])
def test_guard_rejects_every_old_managed_support_symbol(name, template):
    assert managed_ownership_problems(template.format(name=name))


@pytest.mark.parametrize('name,owner', sorted(MANAGED_OWNERS.items()))
def test_guard_rejects_managed_support_compatibility_export(name, owner):
    assert managed_ownership_problems(f'from .{owner} import {name}', driver=True)
    assert not managed_ownership_problems(f'from .{owner} import {name}\nx = {name}', driver=True)


def test_managed_rewrite_preserves_aliases_local_imports_and_real_patch_owners():
    source = ('from coding_review_agent_loop import managed_ci as ci\n'
              'def consumer():\n    from .managed_ci import ManagedCiContract as Contract, activate_managed_ci\n'
              'ci._api_list(runner, config, endpoint)\n'
              'monkeypatch.setattr(ci, "_api_list", fake)\n'
              'patch("coding_review_agent_loop.managed_ci.assess_exact_head_protection")\n')
    rewritten = _MANAGED_REWRITE.rewrite(source, MANAGED_OWNERS)
    assert not managed_ownership_problems(rewritten)
    assert 'ManagedCiContract as Contract' in rewritten
    assert 'activate_managed_ci' in rewritten
    assert _MANAGED_REWRITE.rewrite(rewritten, MANAGED_OWNERS) == rewritten


@pytest.mark.parametrize('first', [
    'managed_ci', 'managed_ci_types', 'managed_ci_api', 'managed_ci_protection',
    'managed_ci_failure', 'managed_ci_status', 'managed_ci_reconciliation',
])
def test_managed_support_fresh_import_orders(first):
    import os
    import subprocess
    import sys

    modules = [first] + [name for name in [
        'managed_ci', 'managed_ci_types', 'managed_ci_api', 'managed_ci_protection',
        'managed_ci_failure', 'managed_ci_status', 'managed_ci_reconciliation',
    ] if name != first]
    code = '\n'.join('import coding_review_agent_loop.' + name for name in modules)
    subprocess.run([sys.executable, '-c', code], check=True, timeout=30, cwd=ROOT,
                   env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')})


def test_managed_helper_dependency_patch_intercepts_real_lookup(monkeypatch, tmp_path):
    import importlib
    from types import SimpleNamespace
    from coding_review_agent_loop import managed_ci_api, managed_ci_types
    from agent_loop_helpers import make_config

    calls = []

    def stop_at_lookup(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError('support lookup intercepted')

    monkeypatch.setattr(managed_ci_api, '_advertised_managed_actor', stop_at_lookup)
    module = importlib.import_module('coding_review_agent_loop.managed_ci_failure')
    config = make_config(tmp_path)
    contract = managed_ci_types.ManagedCiContract()
    metadata = SimpleNamespace(head_sha='a' * 40)
    with pytest.raises(RuntimeError, match='support lookup intercepted'):
        module.live_authority(None, config=config, pr_number=7, metadata=metadata, contract=contract)
    assert len(calls) == 1



def test_managed_driver_api_patch_intercepts_real_lookup(monkeypatch, tmp_path):
    from coding_review_agent_loop import managed_ci, managed_ci_api
    from agent_loop_helpers import make_config

    calls = []

    def stop_at_lookup(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError('driver API lookup intercepted')

    monkeypatch.setattr(managed_ci_api, '_api_json', stop_at_lookup)
    config = make_config(tmp_path)
    with pytest.raises(RuntimeError, match='driver API lookup intercepted'):
        managed_ci._github_proves_descendant(
            None, config=config, predecessor_head='a' * 40, live_head='b' * 40,
        )
    assert len(calls) == 1
    assert calls[0][0] == (
        None, config, f'repos/{config.repo}/compare/' + 'a' * 40 + '...' + 'b' * 40,
    )
