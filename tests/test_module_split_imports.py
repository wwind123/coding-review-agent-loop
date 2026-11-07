"""Symbol-specific guards for cleaned protocol and managed-CI support surfaces."""
from __future__ import annotations

import ast
import importlib.util
import json
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
CLEANUP_PROTOCOL_OWNERS = {
    'CITABLE_TEST_RUNNERS_SUMMARY': 'test_runtime',
    'NonRepairableEvidenceRejection': 'errors',
    'TestObservationsShapeError': 'errors',
    '_managed_test_wrapper_inner_command': 'test_runtime',
}
OWNERS.update(CLEANUP_PROTOCOL_OWNERS)
OLD = 'coding_review_agent_loop.protocol'


def ownership_problems(source, *, driver=False, old=OLD, owners=OWNERS, functions=()):
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
    # Module aliases may be imported after function definitions; those functions
    # still resolve the binding when invoked after module initialization.
    bindings = {}
    for imported in nodes:
        if isinstance(imported, ast.Import):
            for alias in imported.names:
                bindings[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
        elif isinstance(imported, ast.ImportFrom) and imported.module == 'coding_review_agent_loop':
            for alias in imported.names:
                bindings[alias.asname or alias.name] = imported.module + '.' + alias.name
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
            elif node.level:
                module = 'coding_review_agent_loop' + ('.' + module if module else '')
            for alias in node.names:
                bindings[alias.asname or alias.name] = module + '.' + alias.name
                if driver and alias.name in functions and module == 'coding_review_agent_loop.' + owners[alias.name]:
                    problems.append(f'{node.lineno}: driver-local function import {alias.name}')
                if module == old and (alias.name in owners or alias.name == '*'):
                    problems.append(f'{node.lineno}: old import {alias.name}')
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            if driver and resolve(node.value) in {
                'coding_review_agent_loop.' + owners[name] + '.' + name for name in functions
            }:
                problems.append(f'{node.lineno}: driver-local function alias')
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target]):
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
                    if node.module != '__future__' and (alias.asname or alias.name) not in loads:
                        problems.append(f'{node.lineno}: compatibility-only export {alias.name}')
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if (alias.asname or alias.name.split('.')[0]) not in loads:
                        problems.append(f'{node.lineno}: unused import {alias.name}')
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
_AUTH_SPEC = importlib.util.spec_from_file_location('authorization_rewrite', ROOT / 'scripts/rewrite_managed_authorization_imports.py')
_AUTH_REWRITE = importlib.util.module_from_spec(_AUTH_SPEC)
_AUTH_SPEC.loader.exec_module(_AUTH_REWRITE)
AUTHORIZATION_OWNERS = _AUTH_REWRITE.symbol_owners()
MANAGED_OWNERS = {**_MANAGED_REWRITE.symbol_owners(), **AUTHORIZATION_OWNERS}
CLEANUP_MANAGED_OWNERS = {
    **dict.fromkeys(('CI_MACHINE_OBLIGATION_KINDS', 'MACHINE_AUTHORITY'), 'protocol'),
    **dict.fromkeys(('KNOWN_HOST_COMMENT_FOOTER', 'MARKER_BY_TOKEN', 'scan_reserved_markers'), 'protocol_markers'),
    **dict.fromkeys(('ROUND_RESUME_MARKER_RE', 'decode_mapping', 'hydrate_mapping'), 'round_transport'),
    **dict.fromkeys(('PullRequestMergeability', 'parse_strong_issue_reference_evidence'), 'github'),
    'is_complete_code_review': 'review_scope',
    'run_gh_read': 'github_retry',
}
MANAGED_OWNERS.update(CLEANUP_MANAGED_OWNERS)
# Inventory function-valued dependencies from their defining sources, leaving
# type/constant imports and the frozen historical move manifests intact.
MANAGED_FUNCTIONS = {
    name for name, owner in MANAGED_OWNERS.items()
    if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
           for node in ast.parse((ROOT / 'src/coding_review_agent_loop' / (owner + '.py')).read_text()).body)
}
MANAGED_OLD = 'coding_review_agent_loop.managed_ci'


def managed_ownership_problems(source, *, driver=False):
    return ownership_problems(source, driver=driver, old=MANAGED_OLD, owners=MANAGED_OWNERS, functions=MANAGED_FUNCTIONS)


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
    problems = managed_ownership_problems(f'from .{owner} import {name}\nx = {name}', driver=True)
    assert bool(problems) == (name in MANAGED_FUNCTIONS)


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
    'managed_ci_authorization', 'managed_ci_recovery_commands', 'managed_ci_intent_validation',
    'managed_ci_tuple', 'managed_ci_result',
])
def test_managed_support_fresh_import_orders(first):
    import os
    import subprocess
    import sys

    modules = [first] + [name for name in [
        'managed_ci', 'managed_ci_types', 'managed_ci_api', 'managed_ci_protection',
        'managed_ci_failure', 'managed_ci_status', 'managed_ci_reconciliation',
    'managed_ci_authorization', 'managed_ci_recovery_commands', 'managed_ci_intent_validation',
    'managed_ci_tuple', 'managed_ci_result',
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
    from coding_review_agent_loop import managed_ci_authorization, managed_ci_api
    from agent_loop_helpers import make_config

    calls = []

    def stop_at_lookup(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError('driver API lookup intercepted')

    monkeypatch.setattr(managed_ci_api, '_api_json', stop_at_lookup)
    config = make_config(tmp_path)
    with pytest.raises(RuntimeError, match='driver API lookup intercepted'):
        managed_ci_authorization._github_proves_descendant(
            None, config=config, predecessor_head='a' * 40, live_head='b' * 40,
        )
    assert len(calls) == 1
    assert calls[0][0] == (
        None, config, f'repos/{config.repo}/compare/' + 'a' * 40 + '...' + 'b' * 40,
    )


def test_authorization_rewrite_migrates_helpers_and_is_idempotent():
    source = ('from coding_review_agent_loop import managed_ci as ci\n'
              'def consumer():\n    from .managed_ci import recover_issue_created_handoff as recover, activate_managed_ci\n'
              'ci.recover_issue_created_handoff(runner)\n'
              'monkeypatch.setattr(ci, "_authorization_matches", fake)\n'
              'patch("coding_review_agent_loop.managed_ci.render_managed_ci_resume_command")\n')
    rewritten = _AUTH_REWRITE.rewrite(source, AUTHORIZATION_OWNERS)
    assert not managed_ownership_problems(rewritten)
    assert 'recover_issue_created_handoff as recover' in rewritten
    assert 'activate_managed_ci' in rewritten
    assert _AUTH_REWRITE.rewrite(rewritten, AUTHORIZATION_OWNERS) == rewritten


def test_final_managed_and_protocol_sizes_and_allowlist():
    import runpy

    size_guard = runpy.run_path(str(ROOT / 'tests/test_source_file_size.py'))
    modules = {'managed_ci', 'protocol', 'protocol_common', 'protocol_discuss',
               'protocol_discuss_types', 'protocol_risk', *MANAGED_OWNERS.values()}
    for module in modules:
        path = ROOT / 'src/coding_review_agent_loop' / (module + '.py')
        assert len(path.read_text().splitlines()) <= size_guard['DEFAULT_LIMIT']
    for module in ('managed_ci', 'protocol'):
        assert f'src/coding_review_agent_loop/{module}.py' not in size_guard['ALLOWLIST']


@pytest.mark.parametrize('template', [
    'from .{owner} import {name} as helper\nhelper()',
    'from coding_review_agent_loop.{owner} import {name}\n{name}()',
    'from . import {owner} as support\nhelper = support.{name}\nhelper()',
    'from . import {owner} as support\nhelper: object = support.{name}\nhelper()',
])
@pytest.mark.parametrize('name', sorted(MANAGED_FUNCTIONS))
def test_driver_rejects_local_moved_function_bindings(name, template):
    assert managed_ownership_problems(template.format(name=name, owner=MANAGED_OWNERS[name]), driver=True)


@pytest.mark.parametrize('name', sorted(MANAGED_FUNCTIONS))
def test_driver_allows_live_owner_lookups(name):
    owner = MANAGED_OWNERS[name]
    assert not managed_ownership_problems(
        f'from . import {owner} as support\nsupport.{name}()', driver=True)


@pytest.mark.parametrize('name', sorted(CLEANUP_PROTOCOL_OWNERS))
@pytest.mark.parametrize('template', [
    'from .protocol import {name} as moved',
    'from . import protocol as p\nq = p\nq.{name}',
    'patch("coding_review_agent_loop.protocol.{name}")',
    'from . import protocol as p\nmonkeypatch.setattr(p, "{name}", fake)',
    'from . import protocol as p\npatch.object(p, "{name}", fake)',
])
def test_guard_rejects_protocol_cleanup_paths(name, template):
    assert ownership_problems(template.format(name=name))


@pytest.mark.parametrize('helper', [
    '_advertised_managed_actor', '_api_list', '_probe_json', 'assess_exact_head_protection',
])
def test_driver_owner_patch_intercepts_after_import(monkeypatch, tmp_path, helper):
    from unittest.mock import Mock
    from coding_review_agent_loop import managed_ci, managed_ci_api, managed_ci_protection
    from coding_review_agent_loop.managed_ci_types import ManagedCiProbeContext
    from agent_loop_helpers import make_config
    from coding_review_agent_loop.workdirs import active_workdir

    config = make_config(tmp_path, managed_ci_trusted_actor='agent-loop')
    context = ManagedCiProbeContext(config.repo, config.gh_cmd, active_workdir(config))
    runner = Mock()
    runner.run.side_effect = AssertionError('unexpected GitHub operation')
    intercepted = Mock(side_effect=RuntimeError('owner lookup intercepted'))
    owner = managed_ci_protection if helper == 'assess_exact_head_protection' else managed_ci_api
    monkeypatch.setattr(owner, helper, intercepted)
    if helper == '_advertised_managed_actor':
        who = Mock(return_value={'login': 'agent-loop', 'id': 1})
        monkeypatch.setattr(managed_ci_api, '_api_json', who)
    with pytest.raises(RuntimeError, match='owner lookup intercepted'):
        if helper == '_advertised_managed_actor':
            managed_ci._adoption_identity(runner, config=config)
        elif helper == '_api_list':
            managed_ci._find_resume_audit(runner, config=config, pr_number=7,
                actor_login='agent-loop', actor_id=1, base_ref='main')
        elif helper == '_probe_json':
            managed_ci.evaluate_managed_ci_readiness(runner, context=context,
                base='main', trusted_actor='agent-loop')
        else:
            managed_ci._has_exact_head_protection(runner, config=config, base_ref='main')
    if helper == '_advertised_managed_actor':
        who.assert_called_once_with(runner, config, 'user', quiet=True)
        intercepted.assert_called_once_with(runner, config)
    elif helper == '_api_list':
        intercepted.assert_called_once_with(runner, config,
            f'repos/{config.repo}/issues/7/comments?per_page=100')
    elif helper == '_probe_json':
        intercepted.assert_called_once_with(runner, context, f'repos/{config.repo}')
    else:
        intercepted.assert_called_once_with(runner, context=context, base='main')
    runner.run.assert_not_called()


def test_protocol_quoted_risk_annotations_resolve():
    import typing
    from coding_review_agent_loop import protocol

    for value in vars(protocol).values():
        if isinstance(value, type) and value.__module__ == protocol.__name__:
            typing.get_type_hints(value)


# Phase-one prompt ownership manifest; deliberately outside the frozen facade registry.

PROMPT_MOVE = json.loads((ROOT / 'tests/fixtures/prompt_support_move.json').read_text())
PROMPT_OWNERS = {name: 'prompt_support' for name in PROMPT_MOVE['symbols']}


def test_prompt_consumers_use_real_owners_and_driver_has_no_unused_imports():
    problems = []
    for folder in ('src', 'tests'):
        for path in (ROOT / folder).rglob('*.py'):
            problems.extend(f'{path.relative_to(ROOT)}: {problem}' for problem in
                            ownership_problems(path.read_text(), driver=path.name == 'prompts.py',
                                               old='coding_review_agent_loop.prompts', owners=PROMPT_OWNERS))
    assert problems == []


@pytest.mark.parametrize('first', ['prompts', 'prompt_support', 'orchestrator', 'prompt_context'])
def test_prompt_support_fresh_import_orders(first):
    import os
    import subprocess
    import sys

    modules = [first] + [m for m in ('prompts', 'prompt_support', 'orchestrator', 'prompt_context') if m != first]
    subprocess.run([sys.executable, '-c', '\n'.join('import coding_review_agent_loop.' + m for m in modules)],
                   check=True, timeout=30, cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')})


def test_prompt_support_move_preserves_function_class_and_constant_bodies():
    import hashlib

    tree = ast.parse((ROOT / 'src/coding_review_agent_loop/prompt_support.py').read_text())
    actual = {}
    for node in tree.body:
        name = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else (
            node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None)
        if name in PROMPT_OWNERS:
            actual[name] = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    assert actual == PROMPT_MOVE['body_sha256']


def missing_patch_owners(covered, reachable, name):
    """Check path-specific reachability, independently of listed-owner validation."""
    return {module.__name__ for module in reachable if hasattr(module, name)} - {
        module.__name__ for module in covered}


def test_prompt_patch_coverage_detects_omitted_owner_and_intercepts_both_calls(monkeypatch, tmp_path):
    # Both callers are imported before patching. This is a real shared lookup,
    # not an assumption that every package binding is reached by every path.
    from coding_review_agent_loop import prompts, prompt_support
    from agent_loop_helpers import make_config, patch_attr_in

    reachable = (prompts, prompt_support)
    assert missing_patch_owners((prompts,), reachable, 'sanitize_historical_text') == {prompt_support.__name__}
    assert not missing_patch_owners(reachable, reachable, 'sanitize_historical_text')
    calls = []
    def fake(text):
        calls.append(text)
        return 'intercepted'
    patch_attr_in(monkeypatch, reachable, 'sanitize_historical_text', fake)
    config = make_config(tmp_path)
    assert 'intercepted' in prompt_support._architecture_detail_dirs_text(config)
    # Retained local summary formatting also reads the shared sanitizer binding.
    assert prompts._bounded_context_text('retained probe') == 'intercepted'
    assert 'retained probe' in calls
    assert len(calls) >= 2


# Phase-two inventory is separate from the frozen historical facade registry.
ROUND_MOVE = json.loads((ROOT / 'tests/fixtures/round_metadata_move.json').read_text())
ROUND_OWNERS = {**ROUND_MOVE['support_owners'],
                **{name: 'round_metadata' for name in ROUND_MOVE['symbols']}}
OLD_ROUND = 'coding_review_agent_loop.round_state'


def test_round_consumers_use_real_owners_and_have_no_unused_driver_imports():
    problems = []
    for folder in ('src', 'tests'):
        for path in (ROOT / folder).rglob('*.py'):
            problems.extend(f'{path.relative_to(ROOT)}: {problem}' for problem in
                            ownership_problems(path.read_text(), driver=path.name in (
                                'round_state.py', 'round_metadata.py'), old=OLD_ROUND, owners=ROUND_OWNERS))
    assert problems == []


def test_round_metadata_move_preserves_bodies():
    import hashlib

    actual = {}
    tree = ast.parse((ROOT / 'src/coding_review_agent_loop/round_metadata.py').read_text())
    for node in tree.body:
        name = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else (
            node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            else node.target.id if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) else None)
        if name in ROUND_MOVE['symbols']:
            actual[name] = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    assert actual == ROUND_MOVE['body_sha256']


@pytest.mark.parametrize('first', ['round_state', 'round_metadata', 'orchestrator',
                                   'qualification_checkpoint', 'plan_round_closure'])
def test_round_metadata_fresh_import_orders(first):
    import os
    import subprocess
    import sys

    modules = [first] + [m for m in ('round_state', 'round_metadata', 'orchestrator',
                                     'qualification_checkpoint', 'plan_round_closure') if m != first]
    subprocess.run([sys.executable, '-c', '\n'.join('import coding_review_agent_loop.' + m for m in modules)],
                   check=True, timeout=30, cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')})


def test_round_metadata_sizes_and_policy():
    import runpy

    for module in ('round_state', 'round_metadata'):
        assert len((ROOT / 'src/coding_review_agent_loop' / (module + '.py')).read_text().splitlines()) <= 5000
    policy = runpy.run_path(str(ROOT / 'tests/test_source_file_size.py'))
    assert policy['DEFAULT_LIMIT'] == 6000
    assert policy['ALLOWLIST']['src/coding_review_agent_loop/pr_loop.py'] == 9945


def test_round_patch_coverage_detects_omitted_owner_and_intercepts_both_calls(monkeypatch):
    from coding_review_agent_loop import round_state, round_metadata
    from agent_loop_helpers import patch_attr_in

    # Both real paths bind decode_mapping: replay ingests an attached body,
    # while the leaf codec accepts its encoded payload directly.
    reachable = (round_state, round_metadata)
    assert missing_patch_owners((round_state,), reachable, 'decode_mapping') == {round_metadata.__name__}
    assert not missing_patch_owners(reachable, reachable, 'decode_mapping')
    metadata = round_metadata.PostedRoundMetadata('pr', 'coder', 'Codex', 1, 'head')
    body = round_metadata._attach_round_metadata('Coder.', metadata)
    encoded = round_metadata._encode_round_metadata(metadata)
    calls = []
    original = round_metadata.decode_mapping
    def decode(value):
        calls.append(value)
        return original(value)
    patch_attr_in(monkeypatch, reachable, 'decode_mapping', decode)
    from types import SimpleNamespace
    assert round_state._extract_round_metadata_records([SimpleNamespace(body=body)], flow='pr')[0].metadata == metadata
    assert round_metadata._decode_round_metadata(encoded) == metadata
    assert calls == [encoded, encoded]


def test_round_metadata_dependency_closure_and_no_reverse_import():
    import builtins
    import symtable

    for module in ('round_state', 'round_metadata'):
        source = (ROOT / 'src/coding_review_agent_loop' / (module + '.py')).read_text()
        table = symtable.symtable(source, module, 'exec')
        bound = {symbol.get_name() for symbol in table.get_symbols()
                 if symbol.is_assigned() or symbol.is_imported()}
        def global_references(scope):
            names = {symbol.get_name() for symbol in scope.get_symbols()
                     if symbol.is_global() and symbol.is_referenced()}
            for child in scope.get_children():
                names.update(global_references(child))
            return names
        assert global_references(table) - bound - set(dir(builtins)) - {'__name__'} == set()
        if module == 'round_metadata':
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom):
                    assert node.module not in ('round_state', OLD_ROUND)
                    assert all(alias.name != 'round_state' for alias in node.names)
                elif isinstance(node, ast.Import):
                    assert all(alias.name != OLD_ROUND for alias in node.names)


@pytest.mark.parametrize('name', ROUND_MOVE['symbols'])
def test_round_manifest_rejects_old_aliases_annotations_and_patch_paths(name):
    for source in (
        f'from .round_state import {name}',
        f'from . import round_state as state\nx: "state.{name}"',
        f'patch("{OLD_ROUND}.{name}")',
        f'from . import round_state as state\nmonkeypatch.setattr(state, "{name}", fake)',
    ):
        assert ownership_problems(source, old=OLD_ROUND, owners=ROUND_OWNERS)


PLAN_MOVE = json.loads((ROOT / 'tests/fixtures/plan_execution_move.json').read_text())
PLAN_OWNERS = {name: 'plan_execution' for name in PLAN_MOVE['symbols']}


def test_plan_consumers_use_real_owners_and_no_unused_driver_imports():
    problems = []
    for folder in ('src', 'tests'):
        for path in (ROOT / folder).rglob('*.py'):
            problems.extend(f'{path.relative_to(ROOT)}: {problem}' for problem in
                            ownership_problems(path.read_text(), driver=path.name in ('plan_first_loop.py', 'plan_execution.py'),
                                               old='coding_review_agent_loop.plan_first_loop', owners=PLAN_OWNERS, functions=PLAN_OWNERS))
    assert problems == []
    driver = ast.parse((ROOT / 'src/coding_review_agent_loop/plan_first_loop.py').read_text())
    definitions = {n.name for n in driver.body if isinstance(n, ast.FunctionDef)}
    definitions.update(target.id for n in driver.body if isinstance(n, ast.Assign)
                       for target in n.targets if isinstance(target, ast.Name))
    assert not definitions.intersection(PLAN_OWNERS)


@pytest.mark.parametrize('first', ['plan_execution', 'plan_first_loop', 'issue_loop', 'orchestrator'])
def test_plan_execution_fresh_import_orders(first):
    import os
    import subprocess
    import sys

    modules = [first] + [m for m in ('plan_execution', 'plan_first_loop', 'issue_loop', 'orchestrator') if m != first]
    subprocess.run([sys.executable, '-c', '\n'.join('import coding_review_agent_loop.' + m for m in modules)],
                   check=True, timeout=30, cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')})
    tree = ast.parse((ROOT / 'src/coding_review_agent_loop/plan_execution.py').read_text())
    assert not any(isinstance(n, ast.ImportFrom) and 'plan_first_loop' in (n.module or '')
                   or isinstance(n, ast.Import) and any('plan_first_loop' in a.name for a in n.names)
                   for n in ast.walk(tree))


def test_combined_split_sizes_and_original_policy():
    from test_source_file_size import DEFAULT_LIMIT, ALLOWLIST

    for module in ('prompts', 'prompt_support', 'round_state', 'round_metadata', 'plan_first_loop', 'plan_execution'):
        assert len((ROOT / f'src/coding_review_agent_loop/{module}.py').read_text().splitlines()) <= 5000
    assert DEFAULT_LIMIT == 6000
    assert ALLOWLIST['src/coding_review_agent_loop/pr_loop.py'] == 9945


def test_plan_patch_coverage_detects_omitted_owner_and_intercepts_both_calls(monkeypatch, tmp_path):
    from coding_review_agent_loop import plan_first_loop, plan_execution
    from agent_loop_helpers import make_config

    reachable = (plan_first_loop, plan_execution)
    assert missing_patch_owners((plan_first_loop,), reachable, 'get_issue_context') == {plan_execution.__name__}
    calls = []

    def intercept(*args, **kwargs):
        calls.append(kwargs['issue_number'])
        raise RuntimeError('local context intercepted')

    for module in reachable:
        monkeypatch.setattr(module, 'get_issue_context', intercept)
    config = make_config(tmp_path)
    from types import SimpleNamespace
    monkeypatch.setattr(plan_first_loop, '_infer_staged_parent_issue', lambda context: 77)
    with pytest.raises(RuntimeError, match='local context intercepted'):
        plan_first_loop._run_plan_first_loop(None, issue_number=1, config=config,
                                             memory=None, issue_context=SimpleNamespace(), usage_context=None)
    with pytest.raises(RuntimeError, match='local context intercepted'):
        plan_execution._run_child_planning_cycle(None, config=config, memory=None,
                                                usage_context=None, parent_issue=1, child_issue_number=88,
                                                run_plan_first_loop=lambda *a, **k: pytest.fail('no dispatch'))
    assert calls == [77, 88]


def test_plan_execution_move_preserves_bodies_except_explicit_callback():
    import hashlib

    class CallbackPlumbing(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            node.keywords = [k for k in node.keywords if k.arg != 'run_plan_first_loop']
            if isinstance(node.func, ast.Name) and node.func.id == 'run_plan_first_loop':
                node.func.id = '_run_plan_first_loop'
            return node

    actual = {}
    tree = ast.parse((ROOT / 'src/coding_review_agent_loop/plan_execution.py').read_text())
    for node in tree.body:
        name = node.name if isinstance(node, ast.FunctionDef) else (
            node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None)
        if name in PLAN_OWNERS:
            value = ast.Module(body=node.body, type_ignores=[]) if isinstance(node, ast.FunctionDef) else node
            value = CallbackPlumbing().visit(value)
            actual[name] = hashlib.sha256(ast.dump(value, include_attributes=False).encode()).hexdigest()
    assert actual == PLAN_MOVE['body_sha256']
    execution = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == '_execute_approved_plan')
    branch = CallbackPlumbing().visit(ast.Module(body=execution.body[1:], type_ignores=[]))
    assert hashlib.sha256(ast.dump(branch, include_attributes=False).encode()).hexdigest() == PLAN_MOVE['approved_branch_sha256']


@pytest.mark.parametrize('name', PLAN_OWNERS)
def test_plan_manifest_rejects_late_aliases_annotations_and_patch_paths(name):
    old = 'coding_review_agent_loop.plan_first_loop'
    for source in (
        f'def call():\n    loop.{name}()\nimport {old} as loop',
        f'import {old} as loop\ndef call(value: "loop.{name}"): pass',
        f'patch("{old}.{name}")',
        f'from . import plan_first_loop as loop\nmonkeypatch.setattr(loop, "{name}", fake)',
    ):
        assert ownership_problems(source, old=old, owners=PLAN_OWNERS)


def patch_owner_inventory(source, path):
    """Enumerate direct, shared-table and loop-carried patch owners statically."""
    tree = ast.parse(source)
    aliases, values = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update({a.asname or a.name: a.name.rsplit('.', 1)[-1]
                            for a in node.names if a.name.startswith('coding_review_agent_loop.')})
        elif isinstance(node, ast.ImportFrom) and node.module == 'coding_review_agent_loop':
            aliases.update({a.asname or a.name: a.name for a in node.names})
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = node.value

    def owners(node, seen=()):
        if isinstance(node, ast.Name):
            if node.id in aliases:
                return [aliases[node.id]]
            if node.id in values and node.id not in seen:
                return owners(values[node.id], (*seen, node.id))
        elif isinstance(node, (ast.Tuple, ast.List)):
            return [name for child in node.elts for name in owners(child, seen)]
        elif isinstance(node, ast.IfExp):
            return list(dict.fromkeys((*owners(node.body, seen), *owners(node.orelse, seen))))
        return []

    rows = []
    def record(node, dependency, covered):
        if not isinstance(dependency, str) or not set(covered).intersection(('plan_first_loop', 'plan_execution')):
            return
        enclosing = [f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
                     and f.lineno <= node.lineno <= f.end_lineno]
        scope = min(enclosing, key=lambda f: f.end_lineno - f.lineno).name if enclosing else '<module>'
        row = {'path': str(path) + '::' + scope, 'dependency': dependency,
               'covered': list(dict.fromkeys(covered))}
        if row not in rows:
            rows.append(row)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ('patch_attr_in', 'patch_object_in'):
                index = 1 if node.func.id == 'patch_attr_in' else 0
            elif isinstance(node.func, ast.Attribute) and node.func.attr in ('setattr', 'object'):
                index = 0
            else:
                continue
            if len(node.args) > index + 1 and isinstance(node.args[index + 1], ast.Constant):
                record(node, node.args[index + 1].value, owners(node.args[index]))
        elif isinstance(node, ast.For):
            iterable = values.get(node.iter.id, node.iter) if isinstance(node.iter, ast.Name) else node.iter
            if isinstance(iterable, (ast.Tuple, ast.List)):
                for row in iterable.elts:
                    if isinstance(row, ast.Tuple) and len(row.elts) == 2 and isinstance(row.elts[0], ast.Constant):
                        record(node, row.elts[0].value, owners(row.elts[1]))
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant):
                    record(node, key.value, owners(value))
    return sorted(rows, key=lambda r: (r['path'], r['dependency'], r['covered']))


def test_plan_patch_owner_inventory_matches_audited_paths():
    expected = json.loads((ROOT / 'tests/fixtures/plan_execution_patch_inventory.json').read_text())
    actual = [row for path in sorted((ROOT / 'tests').rglob('*.py'))
              for row in patch_owner_inventory(path.read_text(), path.relative_to(ROOT))]
    assert actual == expected


def test_patch_inventory_resolves_aliases_shared_tables_and_loop_carried_owners():
    prefix = """
from coding_review_agent_loop import plan_first_loop as retained, plan_execution as extracted
SHARED = (retained, extracted)
"""
    forms = (
        "POSTERS = {'get_issue_context': SHARED}",
        "for name, targets in (('get_issue_context', SHARED),):\n    patch_attr_in(monkeypatch, targets, name, fake)",
        "patch_object_in(SHARED, 'get_issue_context', fake)",
    )
    for form in forms:
        source = prefix + form
        complete = patch_owner_inventory(source, 'fixture.py')
        omitted = patch_owner_inventory(source.replace('SHARED = (retained, extracted)', 'SHARED = (retained,)'), 'fixture.py')
        assert complete and all(row['covered'] == ['plan_first_loop', 'plan_execution'] for row in complete)
        assert omitted != complete
        assert all(set(expected['covered']) - set(actual['covered']) == {'plan_execution'}
                   for expected, actual in zip(complete, omitted))
