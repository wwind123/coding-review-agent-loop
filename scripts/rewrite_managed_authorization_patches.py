"""Retarget moved authorization lookups during the phase-seven move window.

Ordinary consumers keep using the temporary exports until phase eight. Only
patches whose lookup moved into the new owner are rewritten here.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rewrite(source: str, owners: dict[str, str]) -> str:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in ('setattr', 'object') or len(node.args) < 2:
            continue
        target, symbol = node.args[:2]
        if (not isinstance(target, ast.Name) or target.id != 'managed_ci'
                or not isinstance(symbol, ast.Constant) or symbol.value not in owners):
            continue
        edits.append((offsets[target.lineno - 1] + target.col_offset,
                      offsets[target.end_lineno - 1] + target.end_col_offset,
                      owners[symbol.value]))
    for start, end, owner in sorted(edits, reverse=True):
        source = source[:start] + owner + source[end:]
    if edits:
        tree = ast.parse(source)
        insertion = 0
        for node in tree.body:
            if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                    or isinstance(node, ast.ImportFrom) and node.module == '__future__'):
                insertion = node.end_lineno
            else:
                break
        lines = source.splitlines(keepends=True)
        lines.insert(insertion, '\nfrom coding_review_agent_loop import ' +
                     ', '.join(sorted({owner for _, _, owner in edits})) + '\n')
        source = ''.join(lines)
    return source


if __name__ == '__main__':
    manifest = json.loads((ROOT / 'tests/fixtures/managed_ci_authorization_move.json').read_text())
    owners = {name: owner for owner, records in manifest['owners'].items() for name in records}
    for path in (ROOT / 'tests').rglob('*.py'):
        if path.name == 'test_module_split_imports.py':
            continue
        source = path.read_text()
        changed = rewrite(source, owners)
        if source != changed:
            path.write_text(changed)
