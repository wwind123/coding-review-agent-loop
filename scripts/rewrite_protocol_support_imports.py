"""Mechanically migrate the phase-two protocol support surface to its owners.

Run from the repository root. The move manifest and the discussion type import
inventory form the symbol-specific rewrite manifest; unrelated protocol names stay.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'tests/fixtures/protocol_support_move.json'


def symbol_owners():
    owners = {name: owner for owner, records in json.loads(MANIFEST.read_text())['owners'].items()
              for name in records}
    tree = ast.parse((ROOT / 'src/coding_review_agent_loop/protocol_discuss_types.py').read_text())
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            owners[node.name] = 'protocol_discuss_types'
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    owners[target.id] = 'protocol_discuss_types'
    return owners


def rewrite(source, owners):
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module not in ('protocol', 'coding_review_agent_loop.protocol'):
            continue
        if not any(alias.name in owners for alias in node.names):
            continue
        groups = {}
        prefix = 'coding_review_agent_loop.' if node.level == 0 else '.' * node.level
        for alias in node.names:
            owner = owners.get(alias.name, 'protocol')
            groups.setdefault(owner, []).append(ast.unparse(alias))
        indent = ' ' * node.col_offset
        replacement = ('\n' + indent).join(f'from {prefix}{owner} import (\n{indent}    ' +
                       (',\n' + indent + '    ').join(names) + f',\n{indent})'
                       for owner, names in groups.items())
        edits.append((offsets[node.lineno-1]+node.col_offset,
                      offsets[node.end_lineno-1]+node.end_col_offset, replacement))
    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    for name, owner in owners.items():
        source = re.sub(r'coding_review_agent_loop\.protocol\.' + re.escape(name) + r'\b',
                        'coding_review_agent_loop.' + owner + '.' + name, source)
    return source


if __name__ == '__main__':
    owners = symbol_owners()
    for folder in ('src', 'tests'):
        for path in (ROOT / folder).rglob('*.py'):
            if path.name == 'protocol.py':
                continue
            source = path.read_text()
            changed = rewrite(source, owners)
            if source != changed:
                path.write_text(changed)
