"""Plan-first loop tests: architecture-impact contract in planning rounds (#1543)."""

import coding_review_agent_loop.plan_execution as plan_execution
import re
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.github import IssueContext
from coding_review_agent_loop.round_metadata import (
    _decode_round_metadata,
)
import coding_review_agent_loop.architecture_contract as architecture_contract
import coding_review_agent_loop.child_plan_binding as child_plan_binding
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.execution_policy as execution_policy
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
from agent_loop_helpers import patch_attr_in, patch_object_in
from issue_loop_test_helpers import (
    _DEG_CORROBORATED,
    _DEG_UNCORROBORATED,
    _deg_with,
    _plan_round_records,
    _repaired_acknowledgement_fixtures,
    _staged_plan_config,
)
from agent_loop_helpers import (
    FakeRunner as _FakeRunner,
    make_config,
    structured_plan_state,
)
import coding_review_agent_loop.validated_agent as validated_agent_module


def test_plan_first_round_metadata_carries_a_corroborated_plan_state_record(tmp_path):
    plan = _deg_with(structured_plan_state(summary="Add schema helpers."), _DEG_CORROBORATED)
    runner = _FakeRunner(
        claude_outputs=[plan],
        codex_outputs=["Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path, plan_execution_mode="decompose-only", max_rounds=1)
    try:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    except (AgentLoopError, StopIteration, IndexError):
        pass  # only the posted planning round matters here
    bodies = [comment["body"] for comment in runner.issue_comments]
    plan_rounds = [c for c in bodies if "AGENT_LOOP_META" in c and "### Parse degradations" in c]
    assert plan_rounds, "the planning round must surface its degradation record"
    decoded = _decode_round_metadata(
        re.search(r"AGENT_LOOP_META: ([A-Za-z0-9+/=_-]+)", plan_rounds[0]).group(1)
    )
    assert [r.outcome for r in decoded.architecture_impact_degradations] == ["normalized-to-changed"]
    assert decoded.architecture_impact["status"] == "changed"
