"""Phase-1 ownership, verbatim-body and import-order regressions."""
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
PACKAGE = ROOT / "src" / "coding_review_agent_loop"
MANIFEST = json.loads((ROOT / "tests/fixtures/protocol_support_move.json").read_text())


@pytest.mark.parametrize("owner", MANIFEST["owners"])
def test_moved_bodies_and_owners(owner):
    new = importlib.import_module(f"coding_review_agent_loop.{owner}")
    source = (PACKAGE / f"{owner}.py").read_text()
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    definitions = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node
    for name, record in MANIFEST["owners"][owner].items():
        assert getattr(new, name) is not None, name
        node = definitions[name]
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        body = "".join(lines[first - 1:node.end_lineno])
        assert hashlib.sha256(body.encode()).hexdigest() == record["sha256"], name
    assert not any(
        isinstance(node, ast.ImportFrom) and node.level == 1 and node.module == "protocol"
        for node in ast.walk(tree)
    )


@pytest.mark.parametrize("first", ["protocol", "protocol_common", "protocol_discuss"])
def test_fresh_import_order(first):
    modules = [first] + [name for name in ("protocol", "protocol_common", "protocol_discuss") if name != first]
    code = "\n".join(f"import coding_review_agent_loop.{name}" for name in modules)
    code += "\nfrom coding_review_agent_loop import protocol, protocol_common, protocol_discuss\n"
    code += "assert protocol_common.ParseDegradation.__module__.endswith('protocol_common')\n"
    code += "assert not hasattr(protocol, 'parse_structured_discuss_answer')\n"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30,
                   env={**os.environ, "PYTHONPATH": str(PACKAGE.parent)}, cwd=ROOT)


def test_linked_audit_imports_resolve_to_the_declared_owner():
    from coding_review_agent_loop.shape_check_audit import PARSER_OWNERS

    definitions = {}
    imports = []
    for owner in PARSER_OWNERS:
        tree = ast.parse((PACKAGE / f"{owner}.py").read_text())
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assert node.name not in definitions, node.name
                definitions[node.name] = owner
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assert target.id not in definitions, target.id
                        definitions[target.id] = owner
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                assert node.target.id not in definitions, node.target.id
                definitions[node.target.id] = owner
            elif isinstance(node, ast.ImportFrom) and node.level == 1 and node.module in PARSER_OWNERS:
                imports.append(node)
    for node in imports:
        for alias in node.names:
            assert alias.asname in (None, alias.name)
            assert definitions[alias.name] == node.module
