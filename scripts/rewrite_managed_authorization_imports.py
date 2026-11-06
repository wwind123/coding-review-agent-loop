"""Finish managed-CI ownership migration using the move manifest and helper inventory.

Run from the repository root. Definitions remain in their recorded owners;
only imports, qualified references and patch targets are rewritten.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
_SUPPORT = runpy.run_path(str(ROOT / 'scripts/rewrite_managed_support_imports.py'))
HELPER_OWNERS = (
    'managed_ci_recovery_commands', 'managed_ci_intent_validation',
    'managed_ci_tuple', 'managed_ci_result', 'managed_ci_reconciliation',
    'managed_ci_status', 'managed_ci_failure',
)


def symbol_owners():
    manifest = json.loads((ROOT / 'tests/fixtures/managed_ci_authorization_move.json').read_text())
    owners = {name: owner for owner, records in manifest['owners'].items() for name in records}
    # Include every helper definition/catalog, including private callback surfaces.
    for owner in HELPER_OWNERS:
        tree = ast.parse((ROOT / 'src/coding_review_agent_loop' / (owner + '.py')).read_text())
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                owners[node.name] = owner
            elif isinstance(node, ast.Assign):
                owners.update((t.id, owner) for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                owners[node.target.id] = owner
    return owners


def rewrite(source, owners):
    return _SUPPORT['rewrite'](source, owners)


def remove_unused_driver_imports(source, owners):
    tree = ast.parse(source)
    loads = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    # Include quoted annotations without mistaking unrelated strings for uses.
    for node in ast.walk(tree):
        annotations = []
        if isinstance(node, (ast.AnnAssign, ast.arg)):
            annotations.append(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotations.append(node.returns)
        for annotation in annotations:
            for child in ast.walk(annotation) if annotation is not None else []:
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    try:
                        loads.update(n.id for n in ast.walk(ast.parse(child.value, mode='eval')) if isinstance(n, ast.Name))
                    except SyntaxError:
                        pass
    lines = source.splitlines(keepends=True)
    for node in reversed(tree.body):
        if not isinstance(node, ast.ImportFrom):
            continue
        kept = [a for a in node.names if a.name not in owners or (a.asname or a.name) in loads]
        if len(kept) == len(node.names):
            continue
        lines[node.lineno-1:node.end_lineno] = ([f"from {'.' * node.level}{node.module} import (\n" +
            ''.join('    ' + ast.unparse(a) + ',\n' for a in kept) + ')\n'] if kept else [])
    return ''.join(lines)


if __name__ == '__main__':
    owners = symbol_owners()
    for folder in ('src', 'tests'):
        for path in (ROOT / folder).rglob('*.py'):
            if path.name in ('managed_ci.py', 'test_module_split_imports.py', 'test_managed_ci_authorization_move.py'):
                continue
            source = path.read_text()
            changed = rewrite(source, owners)
            if source != changed:
                path.write_text(changed)
    driver = ROOT / 'src/coding_review_agent_loop/managed_ci.py'
    driver.write_text(remove_unused_driver_imports(driver.read_text(), owners))
