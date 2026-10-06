"""Shared harness for the risk-matrix coverage re-ask workflow tests (#1290).

Drives the real ``_implement_approved_issue`` post-acceptance gate with a
scripted ``_run_validated_agent`` and a real temp git checkout, so the gate's
committed-tree probe, re-ask, refresh, and rendering run end to end.
"""

from __future__ import annotations

import json
import os
import subprocess
from types import SimpleNamespace

import coding_review_agent_loop.orchestrator as orchestrator_module
from agent_loop_helpers import FakeRunner, make_config, structured_issue_implementation
from coding_review_agent_loop.agent_failure import ValidatedAgentResponse
from coding_review_agent_loop.comment_rendering import render_risk_test_matrix_section
from coding_review_agent_loop.github import IssueContext
from coding_review_agent_loop.protocol import (
    parse_risk_test_matrix,
    risk_test_matrix_identity,
    validate_structured_issue_implementation,
)
from coding_review_agent_loop.round_state import make_approved_plan_context

ROWS = ("row-wf", "row-unit", "row-man")
_REAL_RUN_VALIDATED_AGENT = orchestrator_module._run_validated_agent
_REAL_RUN_AGENT_RESULT = orchestrator_module.run_agent_result


def _row(row_id: str, level: str) -> dict:
    return {
        "row_id": row_id,
        "label": f"Behaviour {row_id}",
        "entry_path_or_mode": "issue implementation",
        "initial_state": "approved plan",
        "event": "coder reports the PR",
        "expected_outcome": f"outcome {row_id}",
        "forbidden_side_effects": [f"no side effect {row_id}"],
        "proposed_test_level": level,
        "proposed_test_location": "tests/test_a.py",
        "applicability": "required",
        "related_scope_item_ids": ["scope-1"],
        "execution_owner": "one-shot",
    }


def matrix_context(*, applicable: bool = True):
    if applicable:
        payload = {
            "applicability": "applicable",
            "rows": [_row("row-wf", "workflow"), _row("row-unit", "unit"), _row("row-man", "manual review")],
            "important_exclusions": ["Planned tests are not evidence."],
        }
    else:
        payload = {
            "applicability": "not-applicable",
            "rows": [],
            "important_exclusions": [],
            "not_applicable_rationale": "Formatting-only change without any stateful path.",
        }
    matrix = parse_risk_test_matrix(payload)
    identity = risk_test_matrix_identity(matrix)
    approved_plan = "Approved implementation plan.\n\n" + render_risk_test_matrix_section(matrix)
    context = make_approved_plan_context(
        approved_plan,
        source_locator="test approved implementation plan",
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest=identity,
    )
    return approved_plan, context


def claim(row_id: str, *, level: str | None = "workflow", path: str = "tests/test_a.py", refs=True) -> dict:
    payload = {
        "row_id": row_id,
        "execution_refs": ["coder-turn:observation-1"],
        "test_identifiers": [f"{path}::test_x"] if refs else [],
        "test_locations": [path] if refs else [],
        "workflow_path_claim": "The orchestrator path was driven.",
        "outcome_assertions": ["The expected outcome holds."],
        "forbidden_effect_assertions": ["No forbidden effect."],
        "caveats": [],
    }
    if level is not None:
        payload["test_level"] = level
    return payload


def complete_claims() -> list[dict]:
    return [claim("row-wf"), claim("row-unit", level="unit"), claim("row-man", level=None)]


def response_text(*, claims=None, gaps=None, pr_number: int | None = 77) -> str:
    raw = structured_issue_implementation(pr_number=pr_number)
    payload, end = json.JSONDecoder().raw_decode(raw)
    if claims is not None:
        payload["risk_test_matrix_claims"] = claims
    if gaps is not None:
        payload["risk_test_matrix_coverage_gaps"] = gaps
    return json.dumps(payload) + raw[end:]


def gap(row_id: str) -> dict:
    return {"row_id": row_id, "reason": "cannot be driven as specified", "proposed_correction": "downgrade the row"}


class CoverageHarness:
    """A scripted implementation turn over a real temp git checkout."""

    def __init__(self, tmp_path, monkeypatch, *, applicable: bool = True, pr_body: str = "Fixes #56", pr_overrides=None, plan_context_mode: str = "default", real_agent: bool = False, claude_outputs=None, **config_overrides):
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.config = make_config(tmp_path, coder="claude", **config_overrides)
        self.repo = self.config.claude_dir
        self._git("init", "-q")
        (self.repo / "tests").mkdir(exist_ok=True)
        (self.repo / "tests" / "test_a.py").write_text("def test_x():\n    pass\n")
        self._git("add", "-A")
        self.head = self._commit("initial")
        self.runner = FakeRunner(
            pr_payload={"body": pr_body, "headRefOid": self.head, **(pr_overrides or {})},
            claude_outputs=claude_outputs,
        )
        self.real_agent = real_agent
        self.approved_plan, self.plan_context = matrix_context(applicable=applicable)
        if plan_context_mode == "none":
            self.plan_context = None
        elif plan_context_mode == "unavailable":
            self.plan_context = make_approved_plan_context(
                self.approved_plan, source_locator="unavailable matrix test plan"
            )
        self.calls: list[dict] = []
        self.script: list = []
        self.run_pr_calls: list[dict] = []
        self.agent_result_calls: list = []
        self.correction_handler = None
        self._install()

    # -- git helpers ------------------------------------------------------
    def _git(self, *args: str) -> str:
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
        }
        return subprocess.run(
            ["git", *args], cwd=self.repo, check=True, capture_output=True, text=True, env=env
        ).stdout.strip()

    def _commit(self, message: str) -> str:
        self._git("commit", "-q", "--allow-empty", "-m", message)
        return self._git("rev-parse", "HEAD")

    def commit_file(self, path: str, *, push: bool) -> str:
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("def test_new():\n    pass\n")
        self._git("add", "-A")
        sha = self._commit(f"add {path}")
        if push:
            self.runner.pr_payload["headRefOid"] = sha
        return sha

    # -- scripted agent ---------------------------------------------------
    def _install(self) -> None:
        m = self.monkeypatch

        def scripted(*args, **kwargs):
            if kwargs.get("role") not in (None, "coder"):
                return _REAL_RUN_VALIDATED_AGENT(*args, **kwargs)
            self.calls.append({"prompt": kwargs.get("prompt"), "session_id": kwargs.get("session_id")})
            step = self.script.pop(0)
            return step(self) if callable(step) else step

        if not self.real_agent:
            m.setattr(orchestrator_module, "_run_validated_agent", scripted)
        m.setattr(orchestrator_module, "resolve_canonical_pr_for_issue", lambda *_a, **_k: None)
        m.setattr(orchestrator_module, "sync_coder_base_before_implementation", lambda *_a, **_k: None)
        m.setattr(orchestrator_module, "preflight_managed_ci_creation", lambda *_a, **_k: None)
        m.setattr(orchestrator_module, "validate_assigned_head_advanced", lambda **_k: None)
        m.setattr(
            orchestrator_module, "run_pr_loop",
            lambda *_a, **kwargs: self.run_pr_calls.append(kwargs) or 0,
        )
        m.setattr(
            orchestrator_module, "reconcile_test_observations",
            lambda observations, **_kwargs: SimpleNamespace(observations=tuple(observations)),
        )
        m.setattr(
            orchestrator_module, "stable_tracked_tree_snapshot",
            lambda _workdir: SimpleNamespace(
                head=self.runner.pr_payload["headRefOid"],
                tracked_digest="tree-current", complete=True, stable=True, status_clean=True,
            ),
        )

        def no_agent_result(*a, **k):
            if k.get("label") != "semantic-evidence-correction":
                return _REAL_RUN_AGENT_RESULT(*a, **k)
            self.agent_result_calls.append((a, k))
            if self.correction_handler is not None:
                return SimpleNamespace(text=self.correction_handler(self))
            return SimpleNamespace(text="not a structured response")

        m.setattr(orchestrator_module, "run_agent_result", no_agent_result)

    def response(self, text: str, *, session_id: str = "coder-session", turn_id: str = "coder-turn", observations=()):
        parsed = validate_structured_issue_implementation(
            text, delivered_risk_test_matrix_row_ids=ROWS
        )
        assert parsed is not None
        return ValidatedAgentResponse(
            text=text,
            session_id=session_id,
            marker_value=parsed,
            acquisition_test_turn_id=turn_id,
            acquisition_test_observations=tuple(observations),
        )

    # -- run --------------------------------------------------------------
    def run(self) -> int:
        issue_context = IssueContext(
            number=56, repo="OWNER/REPO", title="Issue", body="Issue body",
            url="https://github.com/OWNER/REPO/issues/56", comments=(), human_requirements=(),
        )
        return orchestrator_module._implement_approved_issue(
            self.runner,
            issue_number=56,
            approved_plan=self.approved_plan,
            config=self.config,
            memory=None,
            issue_context=issue_context,
            coder_session_id=None,
            usage_context=orchestrator_module._new_usage_context(self.config),
            approved_plan_context=self.plan_context,
        )

    def coder_comment(self) -> str:
        raw = [
            item["body"] for item in self.runner.pr_payload.get("comments", [])
            if isinstance(item, dict) and isinstance(item.get("body"), str)
        ]
        return next(body for body in raw if "AGENT_LOOP_META: " in body)

    def all_comments(self) -> list[str]:
        return [
            item["body"] for item in self.runner.pr_payload.get("comments", [])
            if isinstance(item, dict) and isinstance(item.get("body"), str)
        ] + list(self.runner.comments)
