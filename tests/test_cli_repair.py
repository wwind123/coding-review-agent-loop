import importlib
import json
import re
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_loop_helpers import FakeRunner, make_config, structured_pr_review
from coding_review_agent_loop.cli import build_parser
from coding_review_agent_loop.config import DEFAULT_REASONING_EFFORT, config_from_args
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.protocol import parse_structured_pr_review
from coding_review_agent_loop.runner import CommandResult
from coding_review_agent_loop.usage import RunUsageContext


# --- #2066/#2162: the repair pass is gone; its flags were removed -------------


def _issue_args(tmp_path, *extra):
    return build_parser().parse_args([
        "issue", "1", "--repo", "OWNER/REPO",
        "--codex-dir", str(tmp_path / "codex"), "--claude-dir", str(tmp_path / "claude"),
        "--subprocess-log-dir", str(tmp_path / "logs"),
        *extra,
    ])


def test_default_antigravity_chain_uses_current_flash_generation(tmp_path):
    from coding_review_agent_loop.config import DEFAULT_ANTIGRAVITY_MODELS

    assert DEFAULT_ANTIGRAVITY_MODELS == (
        "Gemini 3.8 Flash (High)",
        "Claude Opus 5.5 (Medium)",
    )
    config = config_from_args(_issue_args(tmp_path), FakeRunner())
    assert config.antigravity_models == DEFAULT_ANTIGRAVITY_MODELS


_SRC = Path(__file__).resolve().parents[1] / "src" / "coding_review_agent_loop"
_DELETED_MODULES = ("repair", "repair_preservation", "shape_check_audit", "agents.format_repair")
# Only the retired-flag rejection table (#2162) may still name the old flag.
_RETIRED_FLAG_OWNERS = {"cli.py"}


def test_repair_backend_is_named_only_by_the_retired_flag_table():
    """`no-repair-backend-static`: grep repair_backend over src (#2092, #2162)."""
    matches = sorted(
        path.relative_to(_SRC).as_posix()
        for path in _SRC.rglob("*.py")
        if re.search(r"repair[_-]backend", path.read_text(encoding="utf-8"))
    )
    assert matches and set(matches) <= _RETIRED_FLAG_OWNERS, matches


@pytest.mark.parametrize("module", _DELETED_MODULES)
def test_deleted_repair_modules_are_neither_present_nor_imported(module):
    """`no-repair-backend-static`: no lingering reference to a deleted module."""
    assert not (_SRC / (module.replace(".", "/") + ".py")).exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"coding_review_agent_loop.{module}")
    leaf = module.rsplit(".", 1)[-1]
    import_re = re.compile(
        rf"^\s*(?:from\s+[.\w]*\b{leaf}\b\s+import|import\s+[.\w]*\b{leaf}\b"
        rf"|from\s+[.\w]+\s+import\s+(?:\(\s*)?(?:[\w\s,]*,\s*)?{leaf}\b)",
        re.MULTILINE,
    )
    offenders = [
        path.relative_to(_SRC).as_posix()
        for path in _SRC.rglob("*.py")
        if import_re.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_no_shape_check_annotation_remains_in_src():
    """The shape-check audit registry and its inline annotations are gone (#2092)."""
    assert [
        path.relative_to(_SRC).as_posix()
        for path in _SRC.rglob("*.py")
        if "# shape-check:" in path.read_text(encoding="utf-8")
    ] == []
