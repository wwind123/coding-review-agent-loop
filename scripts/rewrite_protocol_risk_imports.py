"""Migrate the phase-four risk surface using the recorded move manifest.

Run from the repository root. The manifest remains the original body inventory;
this script handles imports, qualified aliases and literal patch targets only.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
_SUPPORT = runpy.run_path(str(ROOT / 'scripts/rewrite_protocol_support_imports.py'))


def symbol_owners():
    manifest = json.loads((ROOT / 'tests/fixtures/protocol_risk_move.json').read_text())
    return {name: owner for owner, records in manifest['owners'].items() for name in records}


def rewrite(source, owners):
    source = _SUPPORT['rewrite'](source, owners)
    tree = ast.parse(source)
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'coding_review_agent_loop.protocol':
                    aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == 'coding_review_agent_loop' or (node.level and not node.module):
                aliases.update(alias.asname or alias.name for alias in node.names if alias.name == 'protocol')
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in owners and ast.unparse(node.value) in aliases:
            replacement = '_risk_owner.' + node.attr
        elif (isinstance(node, ast.Call) and len(node.args) >= 2
              and ast.unparse(node.args[0]) in aliases
              and isinstance(node.args[1], ast.Constant) and node.args[1].value in owners
              and ((isinstance(node.func, ast.Attribute) and node.func.attr in ('setattr', 'object'))
                   or (isinstance(node.func, ast.Name) and node.func.id in ('getattr', 'setattr')))):
            node = node.args[0]
            replacement = '_risk_owner'
        else:
            continue
        edits.append((offsets[node.lineno-1]+node.col_offset,
                      offsets[node.end_lineno-1]+node.end_col_offset, replacement))
    for start, end, replacement in sorted(set(edits), reverse=True):
        source = source[:start] + replacement + source[end:]
    if edits:
        # Insert after future imports/docstrings without changing local lookup scopes.
        tree = ast.parse(source)
        insertion = 0
        for node in tree.body:
            if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
                or isinstance(node, ast.ImportFrom) and node.module == '__future__'):
                insertion = node.end_lineno
            else:
                break
        lines = source.splitlines(keepends=True)
        lines.insert(insertion, '\nfrom coding_review_agent_loop import protocol_risk as _risk_owner\n')
        source = ''.join(lines)
    return source


if __name__ == '__main__':
    owners = symbol_owners()
    for folder in ('src', 'tests'):
        for path in (ROOT / folder).rglob('*.py'):
            if path.name in ('protocol.py', 'test_module_split_imports.py'):
                continue
            source = path.read_text()
            changed = rewrite(source, owners)
            if source != changed:
                path.write_text(changed)
