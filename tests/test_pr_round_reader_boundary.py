"""Review every round-codec call site, including aliases and dynamic flows.

The adjacent inventory records input provenance per call (never per module).
Changing or adding a reader requires reviewing its acquisition boundary. Pure
helpers are intentionally usable by codec unit tests; their live PR callers
must supply the immutable observation established by PR round ingestion.
"""
import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'src/coding_review_agent_loop'
CODECS = {'_extract_round_metadata_records', '_decode_round_metadata_records', '_continuity_round_records'}


def calls(source, exports=None):
    tree = ast.parse(source)
    exports = exports or CODECS
    names = set(CODECS)
    # Track imported aliases, local aliases, and public re-exports. Attribute
    # access through module aliases retains the exported symbol's last name.
    for _ in range(10):
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
            # Normalize only the callee; arguments, including dynamic flow and
            # provenance projections, remain part of the individual call site.
            found.append(('::'.join(scope), ast.unparse(ast.Call(
                func=ast.Name(id='codec'), args=node.args, keywords=node.keywords))))
        for child in ast.iter_child_nodes(node):
            visit(child, scope)
    visit(tree)
    return sorted(found)


def inventory():
    rows = json.loads((Path(__file__).with_name('pr_round_reader_inventory.json')).read_text())
    assert all(row['provenance'].strip() for row in rows)
    return sorted((row['module'], row['function'], row['call']) for row in rows)


def test_round_reader_call_sites_have_reviewed_input_provenance():
    sources = {p.stem: p.read_text() for p in PACKAGE.glob('*.py')}
    # Resolve renamed re-exports across modules before auditing their callers.
    exports = set(CODECS)
    for _ in range(10):
        before = set(exports)
        for source in sources.values():
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    exports.update(a.asname or a.name for a in node.names if a.name in exports)
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) and node.value.id in exports:
                    exports.update(t.id for t in node.targets if isinstance(t, ast.Name))
        if exports == before:
            break
    actual = sorted((module, function, call) for module, source in sources.items()
                    for function, call in calls(source, exports))
    assert actual == inventory(), 'Round reader changed: review admission provenance per call site.'


@pytest.mark.parametrize('source', [
    'def new_reader(comments):\n return _extract_round_metadata_records(comments, flow="pr")',
    'from .round_state import _extract_round_metadata_records as read\ndef new_reader(comments, flow):\n return read(comments, flow=flow)',
    'from . import orchestrator as loop\ndef new_reader(comments, flow):\n read = loop._extract_round_metadata_records\n return read(comments, flow=flow)',
    'from .round_state import _extract_round_metadata_records as read\nparse = read\ndef new_reader(comments, flow):\n return parse(comments, flow=flow)',
])
def test_guard_rejects_new_raw_reader_including_aliases_and_dynamic_flows(source):
    introduced = [('pr_loop', function, call) for function, call in calls(source)]
    assert introduced
    assert any(row not in inventory() for row in introduced)
