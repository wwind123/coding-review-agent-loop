"""Migrate the phase-six managed-CI support surface using the recorded move manifest.

Run from the repository root. The manifest remains the original body inventory;
this script handles imports, qualified aliases and literal patch targets only.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def rewrite_imports(source, owners):
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module not in ('managed_ci', 'coding_review_agent_loop.managed_ci'):
            continue
        if not any(alias.name in owners for alias in node.names):
            continue
        groups = {}
        prefix = 'coding_review_agent_loop.' if node.level == 0 else '.' * node.level
        for alias in node.names:
            owner = owners.get(alias.name, 'managed_ci')
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
        source = re.sub(r'coding_review_agent_loop\.managed_ci\.' + re.escape(name) + r'\b',
                        'coding_review_agent_loop.' + owner + '.' + name, source)
    return source


def symbol_owners():
    manifest = json.loads((ROOT / 'tests/fixtures/managed_ci_support_move.json').read_text())
    return {name: owner for owner, records in manifest['owners'].items() for name in records}


def rewrite(source, owners):
    source = rewrite_imports(source, owners)
    tree = ast.parse(source)
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'coding_review_agent_loop.managed_ci':
                    aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == 'coding_review_agent_loop' or (node.level and not node.module):
                aliases.update(alias.asname or alias.name for alias in node.names if alias.name == 'managed_ci')
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in owners and ast.unparse(node.value) in aliases:
            replacement = '_' + owners[node.attr] + '.' + node.attr
        elif (isinstance(node, ast.Call) and len(node.args) >= 2
              and ast.unparse(node.args[0]) in aliases
              and isinstance(node.args[1], ast.Constant) and node.args[1].value in owners
              and ((isinstance(node.func, ast.Attribute) and node.func.attr in ('setattr', 'object'))
                   or (isinstance(node.func, ast.Name) and node.func.id in ('getattr', 'setattr')))):
            replacement = '_' + owners[node.args[1].value]
            node = node.args[0]
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
        lines.insert(insertion, '\n' + ''.join(f'from coding_review_agent_loop import {owner} as _{owner}\n' for owner in sorted({owner for owner in owners.values() if any(replacement.startswith('_' + owner) for _, _, replacement in edits)})))
        source = ''.join(lines)
    return source


if __name__ == '__main__':
    owners = symbol_owners()
    for folder in ('src', 'tests'):
        for path in (ROOT / folder).rglob('*.py'):
            if path.name in ('managed_ci.py', 'test_module_split_imports.py', 'test_managed_ci_support_move.py'):
                continue
            source = path.read_text()
            changed = rewrite(source, owners)
            if source != changed:
                path.write_text(changed)
