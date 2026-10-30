"""Non-collected issue-loop test helpers shared with other test files (#1541).

These helpers serve the split issue-loop, issue-implementation, plan-first and
module test files (#1543), which import them under the same names they had in
the former tests/test_orchestrator_issue.py.
"""

import json
from dataclasses import replace
from types import SimpleNamespace

import coding_review_agent_loop.execution_policy as execution_policy
from coding_review_agent_loop.decomposition import CreatedPhaseIssue, PlanPhase
from coding_review_agent_loop.github import IssueContext
from coding_review_agent_loop.plan_assembly import AuthenticatedPlanState, assemble_plan_revision
from coding_review_agent_loop.protocol import validate_structured_plan_state

from agent_loop_helpers import structured_plan_review, structured_v1_plan_state
import hashlib
import re

import pytest

import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.comment_rendering import render_canonical_plan_state
from coding_review_agent_loop.github import HumanReviewRequirement
from coding_review_agent_loop.managed_ci import (
    AuthenticatedIssueCreatedHandoff,
    UNPROTECTED_OVERRIDE_TRAILER,
)
from coding_review_agent_loop.panel_evidence import PostedRoundMetadata, _plan_subject
from coding_review_agent_loop.discuss_loop import _attach_round_metadata
from coding_review_agent_loop.round_state import _decode_round_metadata
from coding_review_agent_loop.prompts import HUMAN_REQUIREMENTS_ADDRESSED_MARKER
import coding_review_agent_loop.architecture_contract as architecture_contract
import coding_review_agent_loop.child_plan_binding as child_plan_binding
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.response_validation as response_validation
from agent_loop_helpers import patch_object_in
from coding_review_agent_loop.runner import CommandResult
from agent_loop_helpers import (
    FakeRunner as _FakeRunner,
    make_config,
    structured_plan_state,
)
import test_child_plan_provenance as _m936_cpp  # noqa: E402
import coding_review_agent_loop.comment_rendering as _m948_rendering  # noqa: E402
import coding_review_agent_loop.round_transport as _m948_transport  # noqa: E402
from coding_review_agent_loop.protocol import (
    validate_human_requirements_acknowledgement as _m948_validate_ack,
)
from coding_review_agent_loop.board_amendment import (  # noqa: E402
    format_reviewer_board_amendment_comment as _m943_amendment_comment,
)
import coding_review_agent_loop.validated_agent as validated_agent_module


def _fresh_child_route_fixture(disposition, *, handoff=None):
    phase = PlanPhase(
        title="Child", scope="Implement child.", non_goals="No unrelated work.",
        dependency_notes="No dependencies.", rollout_risk="low",
        validation="Run focused tests.", parent_context="Approved slice.",
        automation="agent-pr", stage_id="stage-one", position=1,
        deliverables=("Child",), non_goals_items=("Unrelated work",),
        acceptance_criteria=("Tests pass",), compatibility_constraints=("Preserve API",),
        covered_scope_item_ids=("scope-1",), execution_disposition=disposition,
        disposition_rationale="Reviewed route.",
    )
    route = execution_policy.resolve_child_execution_route(
        phase, topology_source="approved-plan-v1", recorded_handoff=handoff
    )
    issue = IssueContext(
        number=56, repo="OWNER/REPO", title="Child", body="Child", url="child-url", comments=()
    )
    parent = replace(issue, number=55, title="Parent", url="parent-url")
    return SimpleNamespace(
        parent_issue=55, plan_hash="plan-hash", plan_subject="plan-subject",
        approved_plan="Approved parent plan", parent_plan_context=SimpleNamespace(
            matrix_available=False, risk_test_matrix_payload=None
        ), recommendation=SimpleNamespace(
            strategy="staged", identity=lambda: {"recommendation_sha256": "digest"},
            child_stages=(phase,),
        ), decomposition=SimpleNamespace(phases=(phase,)),
        created=CreatedPhaseIssue(phase=phase, issue_url=issue.url, issue_number=56),
        phase_index=1, stage_id="stage-one", handoff=handoff, overrides=(), route=route,
    ), issue, parent


def _m1103_blocking_chain(
    first_round, last_round, *, base, resolve_first=True, item_offset=0
):
    """Primary blocking reviews and chained planner patches for a round range.

    Round ``n``'s primary review blocks with a fresh finding ``item-k`` (and
    resolves ``item-(k-1)``), where ``k`` is ``n - item_offset``; the
    planner's revision after it is an authenticated patch bound to the
    round-``n`` state that resolves ``item-k``.  Returns the scripted outputs
    and the next base.
    """
    codex, claude = [], []
    for n in range(first_round, last_round + 1):
        k = n - item_offset
        codex.append(
            structured_plan_review(
                state="blocking",
                summary=f"Gap {n} remains.",
                blocking_plan_issues=[f"Close gap {n}."],
                prior_plan_item_dispositions=(
                    [{"item_id": f"item-{k - 1}", "disposition": "resolved"}]
                    if k > 1 and (resolve_first or n > first_round)
                    else None
                ),
            )
        )
        patch_payload = {
            "schema_version": 1,
            "kind": "plan_revision_patch",
            "semantic_patch_contract_version": 1,
            "state": "blocking",
            "summary": f"Close gap {n}.",
            "prior_plan_item_dispositions": [
                {"item_id": f"item-{k}", "disposition": "resolved"}
            ],
            "base_round_number": n,
            "base_state_identity": base.state_identity,
            "operations": [
                {
                    "op": "replace",
                    "field": "plan_steps",
                    "value": [f"Implement the reviewed scope, revision {n}."],
                }
            ],
        }
        base = AuthenticatedPlanState.from_plan(
            assemble_plan_revision(base, patch_payload), round_number=n + 1
        )
        claude.append(
            json.dumps(patch_payload)
            + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
        )
    return codex, claude, base


def _m1103_fresh_base():
    fresh = structured_v1_plan_state()
    return fresh, AuthenticatedPlanState.from_plan(
        validate_structured_plan_state(fresh), round_number=1
    )


_STEP_BACK_MARKER = "STEP-BACK REVISION"


def _m1251_justify_patch(base, *, resolved_item, summary="Justify one-shot."):
    payload = {
        "schema_version": 1,
        "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1,
        "state": "blocking",
        "summary": summary,
        "prior_plan_item_dispositions": [
            {"item_id": resolved_item, "disposition": "resolved"}
        ],
        "base_round_number": base.round_number,
        "base_state_identity": base.state_identity,
        "operations": [
            {
                "op": "replace",
                "field": "one_shot_growth_justification",
                "value": {
                    "crossed_signals": ["revision-count"],
                    "rationale": "The revisions share one seam and cannot ship separately.",
                },
            }
        ],
    }
    next_base = AuthenticatedPlanState.from_plan(
        assemble_plan_revision(base, payload), round_number=base.round_number + 1
    )
    text = (
        json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    return text, next_base


def _fresh_staged_plan_for_recovery() -> tuple[str, str]:
    raw = structured_v1_plan_state()
    payload, end = json.JSONDecoder().raw_decode(raw.lstrip())
    recommendation = payload["execution_recommendation"]
    recommendation.update(
        {
            "strategy": "staged",
            "staging_feasibility": "safe",
            "scope_items": [
                {
                    "scope_item_id": "scope-1",
                    "requirement": "Deliver the first stage.",
                    "acceptance_criteria": ["The first stage passes."],
                },
                {
                    "scope_item_id": "scope-2",
                    "requirement": "Deliver the second stage.",
                    "acceptance_criteria": ["The second stage passes."],
                },
            ],
            "child_stages": [
                {
                    "stage_id": "stage-1",
                    "position": 1,
                    "title": "First stage",
                    "summary": "Deliver the first stage.",
                    "deliverables": ["First stage implementation."],
                    "non_goals": [],
                    "acceptance_criteria": ["The first stage passes."],
                    "depends_on_stage_ids": [],
                    "dependency_notes": "No dependencies.",
                    "automation": "agent-pr",
                    "rollout_risk": "low",
                    "compatibility_constraints": [],
                    "covered_scope_item_ids": ["scope-1"],
                },
                {
                    "stage_id": "stage-2",
                    "position": 2,
                    "title": "Second stage",
                    "summary": "Deliver the second stage.",
                    "deliverables": ["Second stage implementation."],
                    "non_goals": [],
                    "acceptance_criteria": ["The second stage passes."],
                    "depends_on_stage_ids": ["stage-1"],
                    "dependency_notes": "After stage-1.",
                    "automation": "agent-pr",
                    "rollout_risk": "low",
                    "compatibility_constraints": [],
                    "covered_scope_item_ids": ["scope-2"],
                },
            ],
        }
    )
    recommendation.pop("one_shot_delivery", None)
    staged_raw = json.dumps(payload) + raw.lstrip()[end:]
    parsed = validate_structured_plan_state(staged_raw, require_execution_strategy_contract=1)
    canonical = render_canonical_plan_state(parsed)
    return staged_raw, canonical


def _provenance_pages(message: str):
    page = {
        "data": {
            "repository": {
                "pullRequest": {
                    "headRefOid": "6367c48dd193d56ea7b0baad25b19455e529f5ee",
                    "commits": {
                        "totalCount": 1,
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                        "nodes": [{"commit": {"oid": "commit-1", "message": message}}],
                    },
                }
            }
        }
    }
    return [page, page]


def _metadata_from_public_comment(body: str):
    encoded = body.split("AGENT_LOOP_META: ", 1)[1].split(" -->", 1)[0]
    return _decode_round_metadata(encoded)


def _managed_issue_handoff(*, nonce: str) -> AuthenticatedIssueCreatedHandoff:
    return AuthenticatedIssueCreatedHandoff(
        pr_number=77,
        issue_number=56,
        repository="OWNER/REPO",
        base_ref="main",
        head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        branch="agent-loop/managed-56",
        trusted_actor_login="agent-loop",
        trusted_actor_id=1,
        protection_mode="voluntary",
        override_nonce=nonce,
    )


class FakeRunner(_FakeRunner):
    """Issue-loop fixtures model the PR created for issue #56 explicitly."""

    def __init__(self, **kwargs):
        # Most of these long-lived fixtures predate the initial plan_state
        # contract. Upgrade their two common initial-plan shorthands while
        # leaving malformed marker-only cases with non-blocking states intact
        # for their explicit rejection tests below.
        legacy_initial_plans = {
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Google Gemini",
            "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Google Gemini",
        }
        for output_key in ("claude_outputs", "codex_outputs", "gemini_outputs", "antigravity_outputs"):
            outputs = kwargs.get(output_key)
            if outputs is None:
                continue
            kwargs[output_key] = [
                structured_plan_state(summary="Initial plan.", plan_steps=["Make the change."])
                if output in legacy_initial_plans
                else output
                for output in outputs
            ]
        # Signed-requirement tests written before the typed disposition
        # contract focus on their named acknowledgement behavior. Supply the
        # otherwise-irrelevant complete attestation to their structured plan
        # fixtures, while preserving intentionally missing acknowledgement
        # markers for the tests that reject those.
        issue_body = str((kwargs.get("issue_payload") or {}).get("body") or "")
        if "-- Human Reviewer" in issue_body:
            issue_payload = kwargs.get("issue_payload") or {}
            requirement = HumanReviewRequirement(
                source_type="Issue body",
                author=str((issue_payload.get("author") or {}).get("login") or "") or None,
                created_at=issue_payload.get("createdAt"),
                url=issue_payload.get("url") or "https://github.com/OWNER/REPO/issues/56",
                body=issue_body.split("-- Human Reviewer", 1)[0].strip(),
            )
            for output_key in ("claude_outputs", "codex_outputs", "gemini_outputs", "antigravity_outputs"):
                outputs = kwargs.get(output_key)
                if outputs is None:
                    continue
                kwargs[output_key] = [
                    _add_default_requirement_disposition(
                        output,
                        requirement_id=requirement.requirement_id,
                    )
                    for output in outputs
                ]
        kwargs.setdefault("pr_payload", {"body": "Fixes #56"})
        super().__init__(**kwargs)


class _PlanDiagnosticRunner(_FakeRunner):
    def __init__(self, *, post_returncode=0, issue_number=813):
        super().__init__()
        self.post_returncode = post_returncode
        self.issue_number = issue_number
        self.diagnostic_posts = []

    def _run_locked(self, args, *, cwd, check, input_text=None):
        command = list(args)
        if command == ["gh", "api", "user"]:
            recorded, cwd_path = self._record_command(args, cwd)
            return CommandResult(
                recorded, cwd_path, json.dumps({"login": "agent", "id": 7}), "", 0
            )
        if command[:4] == ["gh", "api", "--method", "POST"]:
            endpoint = command[4] if len(command) > 4 else ""
            if endpoint == f"repos/OWNER/REPO/issues/{self.issue_number}/comments":
                recorded, cwd_path = self._record_command(args, cwd)
                body = json.loads(input_text or "{}")["body"]
                self.diagnostic_posts.append(body)
                if self.post_returncode:
                    return CommandResult(recorded, cwd_path, "", "post failed", self.post_returncode)
                return CommandResult(
                    recorded,
                    cwd_path,
                    json.dumps(
                        {
                            "id": 901,
                            "created_at": "2026-09-17T05:30:00Z",
                            "body": body,
                            "user": {"login": "agent", "id": 7},
                        }
                    ),
                    "",
                    0,
                )
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


_TRANSIENT_REPAIR_SUGGESTION = (
    "Suggestion: re-run the same command — the Antigravity repair model "
    "reported a transient model-access failure; the round is resumable "
    "and a retry may succeed."
)


def _add_default_requirement_disposition(
    output: str,
    *,
    requirement_id: str = "Requirement 1",
) -> str:
    if requirement_id.startswith("hr-"):
        output = output.replace("Requirement 1", f"Requirement {requirement_id}")
        output = output.replace(f'"Requirement {requirement_id}"', f'"{requirement_id}"')
    try:
        payload, end = json.JSONDecoder().raw_decode(output.lstrip())
    except (json.JSONDecodeError, ValueError):
        return output
    if not isinstance(payload, dict) or payload.get("kind") not in {
        "plan_state", "plan_revision", "plan_review"
    }:
        return output
    if payload.get("human_requirement_dispositions"):
        return output
    payload["human_requirement_dispositions"] = [{
        "requirement_id": requirement_id,
        "disposition": "addressed",
        "evidence": "The structured plan covers the signed requirement.",
    }]
    prefix = output[: len(output) - len(output.lstrip())]
    return prefix + json.dumps(payload) + output.lstrip()[end:]


def _initial_plan_state(*, summary: str = "Initial plan.", human_requirements: str = "") -> str:
    return structured_plan_state(summary=summary).replace(
        "\n<!-- AGENT_PLAN_STATE: blocking -->",
        human_requirements + "\n<!-- AGENT_PLAN_STATE: blocking -->",
        1,
    )


def _plan_with_requirement_disposition(
    *,
    kind: str,
    summary: str,
    plan_steps: list[str],
    evidence: str,
    state: str = "blocking",
) -> str:
    payload = {
        "schema_version": 1,
        "kind": kind,
        "state": state,
        "summary": summary,
        "plan_steps": plan_steps,
        "architecture_impact": {
            "status": "unchanged",
            "rationale": "No architectural contract changed.",
            "affected_components": [], "dependencies": [],
            "execution_data_flows": [], "persistence": [],
            "public_contracts": [], "security_boundaries": [],
            "canonical_document_action": "no-change",
            "canonical_document_path": None,
            "canonical_document_rationale": "",
        },
        "human_requirement_dispositions": [{
            "requirement_id": "Requirement 1",
            "disposition": "addressed",
            "evidence": evidence,
        }],
    }
    if kind == "plan_revision":
        payload["prior_plan_item_dispositions"] = []
    return (
        json.dumps(payload)
        + "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n### Human requirements\n"
        + f"- Requirement 1: {evidence}\n"
        + f"<!-- AGENT_PLAN_STATE: {state} -->\n-- Anthropic Claude"
    )


_RECOVERY_WORKFLOW = """
# agent-loop-managed
# expected_head_sha
# AGENT_LOOP_MANAGED_CI_V2
# AGENT_LOOP_MANAGED_CI_UNLABELED_RECOVERY_V1
name: CI
on:
  pull_request:
    types: [opened, unlabeled]
  workflow_dispatch:
    inputs:
      protocol_version: {required: true}
      pr_number: {required: true}
      expected_head_sha: {required: true}
      managed_nonce: {required: true}
jobs:
  aggregate:
    name: final-ci/exact-head
"""


class _IssueRecoveryWorkflowRunner(FakeRunner):
    """Model the server-backed issue/PR tuple used by real recovery seams."""

    def __init__(self, *, labeled, authorization_comments=None, **kwargs):
        body = (
            f"Fixes #56\n\n{UNPROTECTED_OVERRIDE_TRAILER} "
            "nonce=opening-nonce"
        )
        pr_payload = {
            "number": 77,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/77",
            "title": "Managed recovery",
            "body": body,
            "headRefName": "agent-loop/managed-56",
            "baseRefName": "main",
            "headRefOid": "6367c48dd193d56ea7b0baad25b19455e529f5ee",
            "comments": [],
            "reviews": [],
        }
        pr_payload.update(kwargs.pop("pr_payload", {}))
        super().__init__(pr_payload=pr_payload, **kwargs)
        self.rest_pr = {
            "state": "open",
            "draft": True,
            "labels": [{"name": "agent-loop-managed"}] if labeled else [],
            "body": body,
            "head": {
                "repo": {"full_name": "OWNER/REPO"},
                "sha": "6367c48dd193d56ea7b0baad25b19455e529f5ee",
                "ref": "agent-loop/managed-56",
            },
            "base": {"ref": "main", "repo": {"full_name": "OWNER/REPO"}},
            "user": {"login": "agent-loop", "id": 1},
        }
        self.authorization_comments = list(authorization_comments or [])
        self.issue_events = [{
            "id": 101,
            "event": "labeled",
            "label": {"name": "agent-loop-managed"},
            "actor": {"login": "agent-loop", "id": 1},
        }]
        self.labels_posted = False
        self.dispatch_count = 0

    @staticmethod
    def _form_value(command, name):
        prefix = f"{name}="
        return next(
            (part[len(prefix):] for part in command if part.startswith(prefix)),
            None,
        )

    def _run_locked(self, args, *, cwd, check, input_text=None):
        command = list(args)
        endpoint = next(
            (part for part in command if part.startswith("repos/")), ""
        )
        if command == ["gh", "api", "user"]:
            recorded, cwd_path = self._record_command(args, cwd)
            return CommandResult(
                recorded, cwd_path, json.dumps({"login": "agent-loop", "id": 1}), "", 0
            )
        if endpoint.endswith("/actions/variables/AGENT_LOOP_MANAGED_ACTOR"):
            recorded, cwd_path = self._record_command(args, cwd)
            return CommandResult(
                recorded, cwd_path, json.dumps({"value": "agent-loop"}), "", 0
            )
        if endpoint.startswith(
            "repos/OWNER/REPO/contents/.github/workflows/ci.yml"
        ):
            recorded, cwd_path = self._record_command(args, cwd)
            return CommandResult(recorded, cwd_path, _RECOVERY_WORKFLOW, "", 0)
        if endpoint == "repos/OWNER/REPO/pulls/77":
            recorded, cwd_path = self._record_command(args, cwd)
            return CommandResult(recorded, cwd_path, json.dumps(self.rest_pr), "", 0)
        if endpoint.startswith("repos/OWNER/REPO/issues/77/events?"):
            recorded, cwd_path = self._record_command(args, cwd)
            return CommandResult(
                recorded, cwd_path, json.dumps(self.issue_events), "", 0
            )
        if endpoint.startswith("repos/OWNER/REPO/issues/77/comments?"):
            recorded, cwd_path = self._record_command(args, cwd)
            return CommandResult(
                recorded, cwd_path, json.dumps(self.authorization_comments), "", 0
            )
        if endpoint == "repos/OWNER/REPO/issues/77/comments" and "POST" in command:
            recorded, cwd_path = self._record_command(args, cwd)
            body = self._form_value(command, "body") or ""
            comment_id = max(
                (comment["id"] for comment in self.authorization_comments),
                default=40,
            ) + 1
            comment = {
                "id": comment_id,
                "body": body,
                "user": {"login": "agent-loop", "id": 1},
            }
            self.authorization_comments.append(comment)
            self.pr_payload.setdefault("comments", []).append({
                "author": {"login": "agent-loop"},
                "createdAt": f"2026-05-23T00:00:{comment_id:02d}Z",
                "body": body,
            })
            return CommandResult(recorded, cwd_path, json.dumps(comment), "", 0)
        if endpoint.startswith("repos/OWNER/REPO/issues/56/timeline?"):
            recorded, cwd_path = self._record_command(args, cwd)
            timeline = [{
                "event": "cross-referenced",
                "source": {
                    "issue": {
                        "number": 77,
                        "repository_url": "https://api.github.test/repos/OWNER/REPO",
                        "pull_request": {"url": "https://api.github.test/pulls/77"},
                    }
                },
            }]
            return CommandResult(recorded, cwd_path, json.dumps(timeline), "", 0)
        if endpoint == "repos/OWNER/REPO/issues/77/labels" and "POST" in command:
            recorded, cwd_path = self._record_command(args, cwd)
            self.labels_posted = True
            self.rest_pr["labels"] = [{"name": "agent-loop-managed"}]
            return CommandResult(recorded, cwd_path, "{}", "", 0)
        if endpoint == "repos/OWNER/REPO/commits/main":
            recorded, cwd_path = self._record_command(args, cwd)
            return CommandResult(
                recorded, cwd_path, json.dumps({"sha": "base-sha"}), "", 0
            )
        if endpoint.endswith("/actions/workflows/ci.yml/dispatches"):
            self.dispatch_count += 1
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


def _write_salvage_summary(
    config,
    *,
    name,
    summary,
    created_at_ns,
    issue_number=56,
    scope="issue-implementation",
    approved_plan_hash_value=None,
):
    salvage_dir = config.log_dir / "salvage" / name
    salvage_dir.mkdir(parents=True)
    summary_path = salvage_dir / "salvage-summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "created_at_ns": created_at_ns,
        "repo": config.repo,
        "issue_number": issue_number,
        "scope": scope,
        "agent": "claude",
        "approved_plan_hash": approved_plan_hash_value,
        "summary": str(summary_path),
    }
    (salvage_dir / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# --- Staged primary-then-panel plan review (#905, from #841) --------------


def _staged_plan_config(tmp_path, **overrides):
    values = {
        "reviewer": ("codex", "gemini"),
        "plan_review_policy": "primary-then-panel",
        "primary_plan_reviewer": "codex",
        "max_rounds": 6,
    }
    values.update(overrides)
    return make_config(tmp_path, **values)


def _plan_round_records(runner):
    """Decode every posted planning round record in comment order."""
    records = []
    for comment in runner.issue_comments:
        body = comment["body"]
        match = re.search(r"<!-- AGENT_LOOP_META: (?P<payload>\S+) -->", body)
        if match is None:
            continue
        metadata = _decode_round_metadata(match.group("payload"))
        if metadata.flow == "plan":
            records.append(metadata)
    return records


def _repaired_acknowledgement_fixtures():
    """A staged plan whose primary approves without acknowledging the requirement.

    Repair recovers the acknowledgement, so the reviewer's verdict is unchanged
    and only the signed-requirement marker and dispositions are added.
    """
    requirement = HumanReviewRequirement(
        source_type="Issue body",
        author="maintainer",
        created_at="2026-05-17T08:00:00Z",
        url="https://github.com/OWNER/REPO/issues/56",
        body="Keep the public API unchanged.",
    )
    dispositions = [
        {
            "requirement_id": requirement.requirement_id,
            "disposition": "addressed",
            "evidence": "The plan preserves the public API.",
        }
    ]
    plan_output = structured_v1_plan_state().replace(
        '"human_requirement_dispositions": []',
        '"human_requirement_dispositions": ' + json.dumps(dispositions),
        1,
    ).replace(
        "\n<!-- AGENT_PLAN_STATE: blocking -->",
        "\n"
        f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
        "### Human requirements\n"
        f"- {requirement.requirement_id}: the plan preserves the public API.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->",
        1,
    )
    # Structurally complete, so it passes reviewer validation and is posted as
    # written; only the signed-requirement marker is missing, which is exactly
    # what the post-reconciliation acknowledgement repair recovers.
    unacknowledged = structured_plan_review(
        state="approved",
        human_requirement_dispositions=dispositions,
    )
    repaired = structured_plan_review(
        state="approved",
        human_requirements_resolved=True,
        human_requirement_dispositions=dispositions,
    )
    panel = structured_plan_review(
        state="approved",
        reviewer="Google Gemini",
        human_requirements_resolved=True,
        human_requirement_dispositions=dispositions,
    )
    return requirement, plan_output, unacknowledged, repaired, panel


def _remediation_plan_fixtures():
    """A fresh generation-1 plan plus a narrow plan-step remediation patch."""
    fresh = structured_v1_plan_state()
    base = plan_first_loop_module.AuthenticatedPlanState.from_plan(
        validate_structured_plan_state(fresh), round_number=1
    )
    patch = {
        "schema_version": 1,
        "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1,
        "state": "blocking",
        "summary": "Name the rollout owner in the plan steps.",
        "prior_plan_item_dispositions": [
            {
                "item_id": "item-1",
                "disposition": "resolved",
                "rationale": "The revised plan step names the rollout owner.",
            }
        ],
        "base_round_number": 1,
        "base_state_identity": base.state_identity,
        "operations": [
            {
                "op": "replace",
                "field": "plan_steps",
                "value": ["Implement the reviewed scope and name the rollout owner."],
            }
        ],
    }
    patch_text = (
        json.dumps(patch) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    return fresh, patch_text


def _invalid_plan_scheduler_comment(subject, *, created_at="2026-01-01T00:00:30Z"):
    """A planning scheduler record whose contract cannot be reconstructed."""
    from coding_review_agent_loop.round_transport import encode_mapping

    payload = {
        "flow": "plan",
        "role": "summary",
        "agent": "Orchestrator",
        "round_number": 1,
        "subject": subject,
        "phase": "scheduler-prelaunch",
        # Missing `required_reviewers`: the planning branch reports `invalid`.
        "scheduler_contract": {"policy": "primary-then-panel"},
        "scheduler_obligation_digest": "0" * 16,
        "scheduler_selected_reviewers": ["Codex"],
        "scheduler_paused_reviewers": [["Gemini", "primary phase"]],
        "scheduler_reasons": ["primary phase"],
        "scheduler_final_sweep": False,
        "scheduler_force_full": False,
        "scheduler_calls_avoided": 1,
    }
    return {
        "author": {"login": "bot"},
        "createdAt": created_at,
        "body": (
            "Plan review scheduling audit.\n\n-- Orchestrator\n"
            f"<!-- AGENT_LOOP_META: {encode_mapping(payload)} -->"
        ),
    }


# --- #925: the architecture_impact contract refusal seam ---------------------


_DEG_CORROBORATED = {
    "status": "modified",
    "rationale": "The parser gains a degraded status.",
    "affected_components": ["protocol parser"],
    "dependencies": ["repair preservation"],
    "execution_data_flows": ["response -> parser -> seam"],
    "persistence": ["round metadata degradation records"],
    "public_contracts": ["architecture_impact status"],
    "security_boundaries": ["agent payload trust boundary"],
    "canonical_document_action": "update",
    "canonical_document_path": "ARCHITECTURE.md",
    "canonical_document_rationale": "Document the degraded status.",
}


_DEG_UNCORROBORATED = {"status": "modified", "rationale": "Something changed."}


def _deg_with(rendered, impact, **extra):
    split = rendered.index("}\n") + 1
    payload = json.loads(rendered[:split])
    if impact is None:
        payload.pop("architecture_impact", None)
    else:
        payload["architecture_impact"] = impact
    payload.update(extra)
    return json.dumps(payload) + rendered[split:]


def _deg_task_text(impact, outcome="opened_pr"):
    payload = {
        "schema_version": 1, "kind": "task_result", "state": "blocking",
        "outcome": outcome, "summary": "Done.",
    }
    if outcome == "opened_pr":
        payload["pr_number"] = 4
    if impact is not None:
        payload["architecture_impact"] = impact
    return json.dumps(payload) + "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"


# ---------------------------------------------------------------------------
# Approved child plan that fails the inherited check before any handoff (#936)
# ---------------------------------------------------------------------------



def _m936_patch(base_plan_text, row, *, summary):
    """Semantic revision with no reviewer item: the guard mints none."""
    patch = json.loads(_m936_cpp._semantic_patch(base_plan_text, row, summary=summary).split("\n<!--", 1)[0])
    patch["prior_plan_item_dispositions"] = []
    return json.dumps(patch) + _m936_cpp.PLAN_FOOTER


# --- #948: digest-on-overflow selection at the structured plan publication sites ---


_M948_FOOTER = "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"


_M948_NOTICE = _m948_rendering.COMPACT_PLAN_DIGEST_NOTICE


_M948_DIRECT = "I checked the relevant GitHub discussion directly before responding."


def _m948_noise(seed, chars=256):
    text = ""
    while len(text) < chars:
        text += hashlib.sha256(f"{seed}-{len(text)}".encode()).hexdigest()
    return text[:chars]


def _m948_stages(prefix, count=300):
    return [
        {"title": f"{prefix} {index}", "summary": _m948_noise(f"{prefix}-{index}")}
        for index in range(count)
    ]


def _m948_plan_config(tmp_path):
    return make_config(tmp_path, max_rounds=3, plan_execution_mode="plan-only")


def _m948_bodies(runner):
    return [str(item["body"]) for item in runner.issue_comments]


def _m948_digest_anchor(runner):
    anchors = [body for body in _m948_bodies(runner) if _M948_NOTICE in body]
    assert len(anchors) == 1
    return anchors[0]


def _m948_hydrated_metadata(anchor, bodies):
    match = list(_m948_transport.ROUND_RESUME_MARKER_RE.finditer(anchor))[-1]
    hydrated, missing = _m948_transport.hydrate_mapping(
        _m948_transport.decode_mapping(match.group("payload")), bodies
    )
    assert missing == set()
    return hydrated


def _m948_ack_block(requirement_ids, *, direct=False):
    lines = ["", HUMAN_REQUIREMENTS_ADDRESSED_MARKER, "### Human requirements"]
    if direct:
        lines.append(f"- {_M948_DIRECT}")
    lines.extend(f"- {item}: covered because " + "EVIDENCE " * 300 for item in requirement_ids)
    return "\n".join(lines) + "\n"


def _m948_signed_payload(kind, requirement_ids):
    payload = json.loads(structured_plan_state(summary="Signed plan.").split("\n", 1)[0])
    payload["kind"] = kind
    payload["plan_steps"] = [f"Step {index}: {_m948_noise(index)}" for index in range(300)]
    payload["human_requirement_dispositions"] = [
        {"requirement_id": item, "disposition": "addressed", "evidence": "EVIDENCE " * 300}
        for item in requirement_ids
    ]
    if kind == "plan_revision":
        payload["prior_plan_item_dispositions"] = [
            {"item_id": "item-1", "disposition": "resolved", "note": "Addressed."}
        ]
    return payload


def _m948_assert_posted_acknowledgement(anchor, requirement_ids, *, direct):
    # Re-parse the POSTED anchor body with the existing validator.
    _m948_validate_ack(
        anchor, surfaced_requirement_ids=requirement_ids, requires_direct_discussion_ack=direct
    )
    assert anchor.count(HUMAN_REQUIREMENTS_ADDRESSED_MARKER) == 1
    assert (_M948_DIRECT in anchor) is direct
    for item in requirement_ids:
        assert f"- **{item}** — `addressed`: " in anchor
    assert "EVIDENCE " * 100 not in anchor
    assert len(anchor) <= _m948_transport.MAX_GITHUB_BODY_CHARS


# --- Signed reviewer-board amendment (#943) -------------------------------


_M943_BOARD = ("Codex", "Gemini", "Antigravity")


def _m943_runner():
    """Primary gate, then an Antigravity blocker whose remediation it never rechecks."""
    fresh, patch_text = _remediation_plan_fixtures()
    resolved = [{"item_id": "item-1", "disposition": "resolved"}]
    return _FakeRunner(
        claude_outputs=[fresh, patch_text],
        codex_outputs=[
            structured_plan_review(state="approved"),
            structured_plan_review(state="approved", prior_plan_item_dispositions=resolved),
        ],
        gemini_outputs=[
            structured_plan_review(state="approved", reviewer="Google Gemini"),
            structured_plan_review(state="approved", reviewer="Google Gemini"),
        ],
        antigravity_outputs=[
            structured_plan_review(
                state="blocking",
                reviewer="Google Antigravity",
                summary="One plan step omits the rollout owner.",
                blocking_plan_issues=["Name the rollout owner in the plan steps."],
            ),
        ],
    )


def _m943_reviewer_calls(runner):
    return [cmd[0] for cmd, _cwd in runner.commands if cmd[0] in {"codex", "gemini", "agy"}]


def _m943_decoded(body):
    match = re.search(r"<!-- AGENT_LOOP_META: (?P<payload>\S+) -->", body)
    return _decode_round_metadata(match.group("payload")) if match else None


def _m943_interrupted_history(tmp_path):
    """Round 3 is partial: Codex's remediation review is posted, Antigravity never ran."""
    runner = _m943_runner()
    config = _staged_plan_config(tmp_path, reviewer=("codex", "gemini", "antigravity"), max_rounds=8)
    real_post = architecture_contract.post_issue_comment

    def stop_after_codex_round_3(*args, **kwargs):
        result = real_post(*args, **kwargs)
        metadata = _m943_decoded(kwargs["body"])
        if (
            metadata is not None
            and metadata.role == "reviewer"
            and metadata.round_number == 3
            and metadata.agent == "Codex"
        ):
            raise KeyboardInterrupt
        return result

    with patch_object_in((architecture_contract, validated_agent_module, response_validation, discuss_loop, execution_policy, child_plan_binding, plan_first_loop_module), "post_issue_comment", side_effect=stop_after_codex_round_3):
        with pytest.raises(KeyboardInterrupt):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    return runner


def _m943_post_amendment(runner, *, effective_from_round, removed=("Antigravity",), **overrides):
    values = dict(
        flow="plan",
        issue=56,
        pr_number=None,
        original_required_reviewers=_M943_BOARD,
        policy="primary-then-panel",
        primary_reviewer="Codex",
        removed_reviewers=removed,
        effective_from_round=effective_from_round,
        rationale="Antigravity weekly quota exhausted.",
    )
    values.update(overrides)
    runner.issue_comments.append(
        {
            "author": {"login": "operator"},
            "createdAt": f"2026-05-23T00:00:{len(runner.issue_comments):02d}Z",
            "body": _m943_amendment_comment(**values),
        }
    )


def _m943_reduced_config(tmp_path):
    return _staged_plan_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=8)


# --- #1088: plain issue mode must not bypass an approved plan ---------------



_M1088_ACTOR = ("agent-bot", 4242)


def _m1088_comment(body, *, created_at="2026-05-23T00:00:02Z", author=_M1088_ACTOR, **extra):
    login, author_id = author
    return {
        "author": {"login": login},
        "_rest_author_id": author_id,
        "createdAt": created_at,
        "body": body,
        **extra,
    }


def _m1090_legacy_history(*reviews):
    """Legacy plan coder and reviewer records: no contract, binding or lineage."""
    plan = "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    comments = [
        _m1088_comment(
            _attach_round_metadata(
                plan,
                PostedRoundMetadata(
                    flow="plan", role="coder", agent="Claude", round_number=1,
                    subject=_plan_subject(plan),
                ),
            ),
            created_at="2026-05-23T00:00:00Z",
        )
    ]
    for index, (agent, signature, state) in enumerate(reviews, start=1):
        review = (
            structured_plan_review(state="approved", reviewer=signature)
            if state == "approved"
            else structured_plan_review(
                state="blocking", reviewer=signature, summary="Missing owner.",
                blocking_plan_issues=["Name the rollout owner."],
            )
        )
        comments.append(
            _m1088_comment(
                _attach_round_metadata(
                    review,
                    PostedRoundMetadata(
                        flow="plan", role="reviewer", agent=agent, round_number=1,
                        subject=_plan_subject(plan), state=state,
                    ),
                ),
                created_at=f"2026-05-23T00:00:0{index}Z",
            )
        )
    return comments


# --- Primary-phase stall stop and contract identity (#1103) ------------------



def _m1103_agent_calls(runner):
    return [cmd[0] for cmd, _cwd in runner.commands if cmd[0] in {"claude", "codex", "gemini"}]


def _m1103_stalled_runner(tmp_path, *, threshold=2, rounds=2):
    """A live run whose primary blocks every round until the stall stop."""
    fresh, base = _m1103_fresh_base()
    codex, claude, _base = _m1103_blocking_chain(1, rounds, base=base)
    runner = _FakeRunner(claude_outputs=[fresh, *claude], codex_outputs=codex)
    config = _staged_plan_config(tmp_path, max_rounds=8, plan_primary_stall_rounds=threshold)
    with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    return runner, config, excinfo.value


def _m1251_prompts(runner, agent):
    return [cmd[-1] for cmd, _cwd in runner.commands if cmd[0] == agent]


def _m1251_run(
    tmp_path, *, last_round, max_rounds, codex_overrides=None, claude_transform=None,
    **overrides,
):
    """Run a live script to `last_round`'s review; return (runner, error).

    Round `n`'s primary review blocks on a new finding `item-n`.  The planner's
    revision after round 2 publishes candidate 3, the third planner candidate,
    which crosses `revision-count`; that revision carries the justification the
    growth gate demands.  The primary's reviews of rounds 3 and 4 are therefore
    the first two new-finding blocks of a crossed plan, and the revision after
    round 4 is the step-back turn.
    """
    fresh, base0 = _m1103_fresh_base()
    codex1, claude1, base_round2 = _m1103_blocking_chain(1, 1, base=base0)
    codex2, _claude2, _base = _m1103_blocking_chain(2, 2, base=base_round2)
    justify, base_round3 = _m1251_justify_patch(base_round2, resolved_item="item-2")
    codex_rest, claude_rest, _base = _m1103_blocking_chain(3, last_round, base=base_round3)
    codex_all = [*codex1, *codex2, *codex_rest]
    for number, output in (codex_overrides or {}).items():
        codex_all[number - 1] = output
    claude_all = [fresh, *claude1, justify, *claude_rest]
    if claude_transform is not None:
        claude_all = [claude_transform(output) for output in claude_all]
    runner = _FakeRunner(
        claude_outputs=claude_all,
        codex_outputs=codex_all,
    )
    values = {
        "max_rounds": max_rounds,
        "plan_growth_max_chars": 4500,
        "plan_growth_max_revisions": 3,
    }
    values.update(overrides)
    config = _staged_plan_config(tmp_path, **values)
    with pytest.raises(AgentLoopError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    return runner, excinfo.value


# --- Staged plan under a guard-active execution mode (#1268) -----------------


def _m1268_staged_raw():
    return _fresh_staged_plan_for_recovery()[0]


def _m1268_narrowing_patch(
    base, *, round_number=1, dispositions=(), deferred=True, one_shot=True
):
    """A semantic patch narrowing the staged candidate (or leaving it staged)."""
    recommendation = json.loads(structured_v1_plan_state().split("\n<!--")[0])[
        "execution_recommendation"
    ]
    if one_shot:
        operations = [
            {"op": "replace", "field": "execution_recommendation", "value": recommendation}
        ]
    else:
        operations = [
            {"op": "replace", "field": "plan_steps", "value": ["Still staged, revised."]}
        ]
    if deferred and one_shot:
        operations.append(
            {
                "op": "replace",
                "field": "deferred_work",
                "value": [
                    {
                        "title": "Stage B follow-up",
                        "summary": "Independent remainder, filed manually.",
                    }
                ],
            }
        )
    payload = {
        "schema_version": 1,
        "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1,
        "state": "blocking",
        "summary": "Narrow to one deliverable.",
        "prior_plan_item_dispositions": list(dispositions),
        "base_round_number": round_number,
        "base_state_identity": base.state_identity,
        "operations": operations,
    }
    return json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"


# --- #1278: the step-back episode anchor -------------------------------------

_STEP_BACK_ANCHOR_PLANNER = "STEP-BACK ANCHOR"


# --- Named reviewer seats on issue/task-created PRs (#1365) ------------------


def _named_1352_seats(tmp_path):
    """The #1352 board: sol=codex, flash=antigravity, opus-low=antigravity."""
    from coding_review_agent_loop.reviewer_seats import ReviewerSeat, SeatAgent

    seats = []
    for name, backend, model in (
        ("sol", "codex", "gpt-6.1-sol"),
        ("flash", "antigravity", "Gemini 3.8 Flash (High)"),
        ("opus-low", "antigravity", "Claude Opus 5.5 (Low)"),
    ):
        workdir = tmp_path / name
        workdir.mkdir(exist_ok=True)
        seats.append(SeatAgent(ReviewerSeat(name, backend, (model,)), workdir))
    return tuple(seats)


def _queue_named_reviews(outputs, seats, review):
    from coding_review_agent_loop.agents.registry import agent_signature

    for seat in seats:
        key = "codex_outputs" if seat.backend == "codex" else "antigravity_outputs"
        outputs.setdefault(key, []).append(review(state="approved", reviewer=agent_signature(seat)))


def _serve_named_pr_rest(runner):
    """Serve PR #77's REST history from the PR's own comments, not issue #56's."""
    runner.issue_comments_by_number[77] = runner.pr_payload.setdefault("comments", [])
    return runner


def _named_pr_records(runner):
    from coding_review_agent_loop.round_state import _extract_round_metadata_records

    return _extract_round_metadata_records(
        [SimpleNamespace(body=comment["body"]) for comment in runner.pr_payload.get("comments", [])],
        flow="pr",
    )


def _assert_named_issue_created_pr(runner, config, expected_root, *, phase="authoritative"):
    """The initial coder record keeps its legacy shape; later records are anchored."""
    from coding_review_agent_loop.board_change import BoardIdentity
    from coding_review_agent_loop.reviewer_seats import reviewer_seat_binding

    records = _named_pr_records(runner)
    initial, *later = records
    assert (initial.metadata.flow, initial.metadata.role, initial.metadata.round_number) == (
        "pr", "coder", 1,
    )
    assert initial.metadata.phase == phase
    assert initial.metadata.seat_binding is None
    assert initial.metadata.pr_board_root is None
    assert initial.metadata.pr_board_identity is None
    assert initial.metadata.pr_board_binding_init is None
    reviews = [record for record in later if record.metadata.role == "reviewer"]
    assert sorted(record.metadata.agent for record in reviews) == ["flash", "opus-low", "sol"]
    assert all(record.metadata.pr_board_root is not None for record in later)
    assert {BoardIdentity.from_payload(r.metadata.pr_board_root) for r in later} == {expected_root}
    assert all(r.metadata.seat_binding == reviewer_seat_binding(config) for r in reviews)
    assert reviewer_seat_binding(config)["version"] == 1
    # The authenticated entry check read the PR and the owning issue's REST
    # history before any review was posted.
    rest_reads = [
        cmd[2] for cmd, _cwd in runner.commands
        if cmd[:2] == ["gh", "api"] and len(cmd) > 2 and "/comments?per_page=100&page=1" in cmd[2]
    ]
    assert "repos/OWNER/REPO/issues/77/comments?per_page=100&page=1" in rest_reads
