"""Risk extraction body, ownership, lookup and acyclic import regressions."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/coding_review_agent_loop"
MANIFEST = json.loads((ROOT / "tests/fixtures/protocol_risk_move.json").read_text())


def test_risk_bodies_and_ownership():
    from coding_review_agent_loop import protocol_risk

    source = (PACKAGE / "protocol_risk.py").read_text()
    lines = source.splitlines(keepends=True)
    definitions = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node
        elif isinstance(node, ast.AnnAssign):
            definitions[node.target.id] = node
    for name, record in MANIFEST["owners"]["protocol_risk"].items():
        assert hasattr(protocol_risk, name), name
        node = definitions[name]
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        body = "".join(lines[first - 1:node.end_lineno])
        assert hashlib.sha256(body.encode()).hexdigest() == record["sha256"], name
    assert len(source.splitlines()) <= 6000
    assert len((PACKAGE / "protocol.py").read_text().splitlines()) <= 6000
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module == "protocol"
        for node in ast.walk(ast.parse(source))
    )


@pytest.mark.parametrize("first", ["protocol", "protocol_risk", "protocol_common"])
def test_risk_fresh_import_orders(first):
    modules = [first] + [name for name in ("protocol", "protocol_risk", "protocol_common") if name != first]
    code = "\n".join(f"import coding_review_agent_loop.{name}" for name in modules)
    code += "\nfrom coding_review_agent_loop import protocol_risk\n"
    for name in MANIFEST["compatibility_exports"]:
        code += f"assert hasattr(protocol_risk, {name!r})\n"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30,
                   env={**os.environ, "PYTHONPATH": str(PACKAGE.parent)}, cwd=ROOT)


def test_risk_citation_patch_intercepts_actual_lookup(monkeypatch):
    from coding_review_agent_loop import protocol_risk

    sentinel = object()
    calls = []

    def citations(value, *, context):
        calls.append((value, context))
        return sentinel

    monkeypatch.setattr(protocol_risk, "_expect_test_observations", citations)
    assert protocol_risk._parse_risk_evidence_citations([], context="evidence") is sentinel
    assert calls == [([], "evidence")]


@pytest.mark.parametrize('class_name, field_name, type_name, collection', [
    ('StructuredCoderFollowup', 'risk_test_matrix_evidence', 'RiskTestMatrixEvidence', False),
    ('StructuredIssueImplementation', 'risk_test_matrix_evidence', 'RiskTestMatrixEvidence', False),
    ('StructuredPlanRevision', 'risk_test_matrix_changes', 'RiskTestMatrixChange', True),
    ('StructuredPlanState', 'risk_test_matrix_changes', 'RiskTestMatrixChange', True),
])
def test_response_risk_annotations_resolve(class_name, field_name, type_name, collection):
    from typing import get_type_hints
    from coding_review_agent_loop import protocol, protocol_risk

    risk_type = getattr(protocol_risk, type_name)
    expected = tuple[risk_type, ...] if collection else risk_type | None
    assert get_type_hints(getattr(protocol, class_name))[field_name] == expected
