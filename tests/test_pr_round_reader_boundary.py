"""Review every round-codec call site, including aliases and dynamic flows.

The adjacent inventory records input provenance per call (never per module).
Changing or adding a reader requires reviewing its acquisition boundary. Pure
helpers are intentionally usable by codec unit tests; their live PR callers
must supply the immutable observation established by PR round ingestion.
"""
import ast
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path

import pytest

parse = lru_cache(maxsize=1024)(ast.parse)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'src/coding_review_agent_loop'
CODECS = {'_extract_round_metadata_records', '_decode_round_metadata_records', '_continuity_round_records',
          'collect_reviewer_board_amendments', 'reject_misplaced_pr_amendments',
          'parse_reviewer_board_amendment_records', 'collect_child_disposition_overrides',
          'collect_child_plan_supersessions', 'find_child_plan_rebind_records',
          'authorized_replan_lineage', 'parse_child_disposition_override_records',
          'parse_child_plan_supersession_records', 'find_latest_pr_contract',
          'find_latest_remote_salvage', 'latest_salvage_context', 'restoration_rounds_from_comments',
          'latest_scope_boundary', 'parse_scope_boundary', 'scope_plan_comments',
          'approved_plan_hashes_for_issue', '_has_plan_approved_followups_marker',
          'find_existing_topology_checkpoint', 'find_topology_checkpoints_for_parent',
          'find_existing_execution_decision', 'execution_decisions_for_parent',
          'issue_execution_decision_evidence', 'find_existing_decomposition',
          'find_decompositions_for_parent', 'find_existing_phase_implementation_handoff',
          'find_phase_implementation_handoffs', 'find_phase_implementation_handoffs_for_parent',
          'find_existing_one_shot_impl_handoff', 'find_latest_one_shot_impl_handoff',
          'find_one_shot_impl_handoffs', 'find_existing_split_materialization',
          'find_existing_split_stage_handoff', 'has_unfiled_split_warning',
          'find_latest_issue_pr_handoff', 'resolve_issue_pr_handoff_lineage',
          'find_staged_completion_records', 'resolve_transaction_lineage', 'resolve_pr_contract_lineage', 'resolve_handoff_lineage',
          'recover_execution_recommendation', 'collect_board_change_links', 'parse_board_change_link',
          'plan_history_bears_lineage',
          'admitted_capture_history', '_parse_issue_human_requirements',
          '_parse_pr_human_requirements', '_authorization_comment_records',
          'parse_managed_ci_override_record', '_walk_intent_page',
          'load_actor_authorization_records', 'read_authenticated_protocol_comments'}


def calls(source, exports=None):
    tree = parse(source)
    exports = exports or CODECS
    names = set(CODECS)
    # Track imported aliases, local aliases, and public re-exports. Attribute
    # access through module aliases retains the exported symbol's last name.
    while True:
        before = set(names)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in exports or alias.name in names:
                        names.add(alias.asname or alias.name)
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                if ((isinstance(value, ast.Name) and value.id in names)
                    or (isinstance(value, ast.Attribute) and value.attr in exports)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    names.update(t.id for t in targets if isinstance(t, ast.Name))
        if names == before:
            break
    found = []
    def visit(node, scope=()):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = (*scope, node.name)
        if isinstance(node, ast.Call) and (
            (isinstance(node.func, ast.Name) and node.func.id in names)
            or (isinstance(node.func, ast.Attribute) and node.func.attr in exports)
        ):
            # Retain the callee as well as arguments: replacing one registered
            # reader with another must require review even with identical args.
            found.append(('::'.join(scope), ast.unparse(node)))
        for child in ast.iter_child_nodes(node):
            visit(child, scope)
    visit(tree)
    return sorted(found)


def sources():
    return {'.'.join(p.relative_to(PACKAGE).with_suffix('').parts): p.read_text()
            for p in PACKAGE.rglob('*.py')}


def inventory():
    return json.loads(Path(__file__).with_name('pr_round_reader_inventory.json').read_text())


def structural_readers(source):
    """Independent tripwire: durable iteration, projections and marker consumers.

    Include definitions even when no inventoried codec is called. Writers,
    presentation and offline parsers need explicit reviewed exemptions too.
    This is a bounded syntax guard, not a replacement for source review.
    """
    found = []
    def visit(node, scope=()):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = (*scope, node.name)
            # Exclude nested definitions from the enclosing function's signals.
            class Signals(ast.NodeVisitor):
                def __init__(self):
                    self.history = self.body = self.marker = self.iteration = False
                def visit_FunctionDef(self, child):
                    if child is node:
                        self.generic_visit(child)
                visit_AsyncFunctionDef = visit_FunctionDef
                def visit_For(self, child):
                    self.iteration |= any(k in ast.unparse(child.iter).lower() for k in ('comment', 'history'))
                    self.generic_visit(child)
                visit_AsyncFor = visit_For
                visit_comprehension = visit_For
                def visit_Name(self, child):
                    self.body |= child.id in {'record_body', 'protocol_comments'}
                    self.history |= any(k in child.id.lower() for k in ('comment', 'history'))
                def visit_Attribute(self, child):
                    self.body |= child.attr in {'body', 'protocol_body', 'amendment_body', 'child_control_body'}
                    self.generic_visit(child)
                def visit_Constant(self, child):
                    if isinstance(child.value, str):
                        self.body |= child.value in {'body', 'protocol_body', 'amendment_body', 'child_control_body'}
                def visit_Call(self, child):
                    if isinstance(child.func, ast.Attribute) and child.func.attr in {'finditer', 'search', 'match', 'fullmatch'}:
                        callee = ast.unparse(child.func.value)
                        self.marker |= any(k in callee for k in ('MARKER_RE', 'RESUME_MARKER', 'SIDECAR_RE', 'LINK_RE', 'SCOPE_BOUNDARY_RE'))
                        self.marker |= any(isinstance(a, ast.Constant) and isinstance(a.value, str)
                                           and 'AGENT_' in a.value for a in child.args)
                    self.generic_visit(child)
            signals = Signals()
            signals.visit(node)
            if (signals.history and (signals.body or signals.iteration)) or signals.marker:
                found.append('::'.join(scope))
        for child in ast.iter_child_nodes(node):
            visit(child, scope)
    visit(parse(source))
    return sorted(found)


def assert_inventory(source_map, reviewed=None):
    reviewed = inventory() if reviewed is None else reviewed
    rows = reviewed['calls']
    expected = [(r['module'], r['function'], r['call'], r['occurrence']) for r in rows]
    assert len(set(expected)) == len(expected), 'Duplicate call inventory'
    assert all(r['provenance'].strip() and r['family'].strip() for r in rows)
    families = reviewed['families']
    assert all(name.strip() for name in families)
    assert all(f['admission'].strip() and f['readers'] for f in families.values())
    registered = [name for family in families.values() for name in family['readers']]
    assert len(set(registered)) == len(registered), 'Duplicate family reader'
    definitions = {n.name for source in source_map.values() for n in ast.walk(parse(source))
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert CODECS <= definitions, 'Stale registered reader definition'
    assert set(registered) == CODECS, 'Stale or unregistered family reader'
    assert all(r['family'] in families for r in rows)
    exports = set(CODECS)
    while True:
        before = set(exports)
        for source in source_map.values():
            for node in ast.walk(parse(source)):
                if isinstance(node, ast.ImportFrom):
                    exports.update(a.asname or a.name for a in node.names if a.name in exports)
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    value = node.value
                    if ((isinstance(value, ast.Name) and value.id in exports)
                        or (isinstance(value, ast.Attribute) and value.attr in exports)):
                        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                        exports.update(t.id for t in targets if isinstance(t, ast.Name))
        if exports == before:
            break
    actual = []
    for module, source in source_map.items():
        counts = Counter()
        for function, call in calls(source, exports):
            counts[function, call] += 1
            actual.append((module, function, call, counts[function, call]))
    actual.sort()
    assert actual == sorted(expected), 'Reader changed: review admission provenance per call site.'
    boundaries = reviewed['boundaries']
    expected_boundaries = [(r['module'], r['function']) for r in boundaries]
    assert len(set(expected_boundaries)) == len(expected_boundaries)
    assert all(r['admission'].strip() and r['classification'] in
               {'reader', 'writer', 'offline', 'presentation', 'acquisition'} for r in boundaries)
    actual_boundaries = sorted((module, function) for module, source in source_map.items()
                               for function in structural_readers(source))
    assert actual_boundaries == sorted(expected_boundaries), 'Unreviewed durable-history consumer definition'


def test_round_reader_call_sites_have_reviewed_input_provenance():
    assert_inventory(sources())


@pytest.mark.parametrize('source', [
    'from .pr_contract import find_latest_pr_contract as read\ndef new_reader(comments):\n return read(comments)',
    'def new_reader(comments):\n return find_latest_remote_salvage(comments)',
    'from . import salvage as recovery\ndef new_reader(comments):\n read = recovery.latest_salvage_context\n return read(comments)',
    'from .decomposition import collect_child_plan_supersessions as read\ndef new_reader(comments):\n return read(comments)',
    'def new_reader(comments):\n return find_child_plan_rebind_records(comments)',
    'def new_reader(comments):\n return collect_reviewer_board_amendments(comments, flow="pr", pr_number=77)',
    'from .board_amendment import collect_reviewer_board_amendments as read\ndef new_reader(comments, flow):\n return read(comments, flow=flow)',
    'def new_reader(comments):\n return _extract_round_metadata_records(comments, flow="pr")',
    'from .round_state import _extract_round_metadata_records as read\ndef new_reader(comments, flow):\n return read(comments, flow=flow)',
    'from . import orchestrator as loop\ndef new_reader(comments, flow):\n read = loop._extract_round_metadata_records\n return read(comments, flow=flow)',
    'from .round_state import _extract_round_metadata_records as read\nparse = read\ndef new_reader(comments, flow):\n return parse(comments, flow=flow)',
])
def test_guard_rejects_new_raw_reader_including_aliases_and_dynamic_flows(source):
    augmented = sources()
    augmented['pr_loop'] += '\n' + source
    with pytest.raises(AssertionError, match='Reader changed|Unreviewed'):
        assert_inventory(augmented)


@pytest.mark.parametrize('module,source', [
    ('agents.nested.reader', 'def unseen(comments):\n return [c.body for c in comments]'),
    ('unregistered', 'def unseen(history):\n return [getattr(c, "body", "") for c in history]'),
    ('nested.restoration', 'def unseen(comments):\n return restoration_rounds_from_comments(comments, flow="plan")'),
    ('nested.dynamic', 'def unseen(comments, flow):\n return _extract_round_metadata_records(comments, flow=flow)'),
])
def test_new_reader_definitions_and_nested_modules_need_review(module, source):
    augmented = sources()
    augmented[module] = source
    with pytest.raises(AssertionError, match='Reader changed|Unreviewed'):
        assert_inventory(augmented)


def test_renamed_reexports_resolve_past_ten_modules():
    augmented = sources()
    chain = {}
    previous = '_extract_round_metadata_records'
    for index in range(14):
        alias = f'export_{index}'
        owner = 'round_state' if index == 0 else f'z{index - 1}'
        chain[f'z{index}'] = f'from .{owner} import {previous} as {alias}'
        previous = alias
    # Reverse dependency order requires more than ten fixed-point passes.
    augmented.update(reversed(list(chain.items())))
    augmented['consumer'] = (f'from .z13 import {previous} as read_current\n'
                             'def unseen(comments, flow):\n return read_current(comments, flow=flow)')
    with pytest.raises(AssertionError, match='Reader changed'):
        assert_inventory(augmented)


@pytest.mark.parametrize('mutation', ['duplicate', 'stale', 'empty-family', 'empty-admission'])
def test_inventory_rejects_invalid_review_rows(mutation):
    reviewed = inventory()
    if mutation == 'duplicate':
        reviewed['calls'].append(reviewed['calls'][0])
    elif mutation == 'stale':
        reviewed['calls'][0]['call'] = 'codec(stale)'
    elif mutation == 'empty-family':
        reviewed['calls'][0]['family'] = ''
    else:
        next(iter(reviewed['families'].values()))['admission'] = ''
    with pytest.raises(AssertionError):
        assert_inventory(sources(), reviewed)
