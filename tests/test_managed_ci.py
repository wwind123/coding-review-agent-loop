import ast
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import coding_review_agent_loop.managed_ci as managed_ci
import coding_review_agent_loop.orchestrator as orchestrator

from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import (
    PullRequestCheck,
    PullRequestChecks,
    PullRequestMetadata,
    merge_pr,
)
from coding_review_agent_loop.managed_ci import (
    FINAL_CONTEXT,
    MANAGED_LABEL,
    MANAGED_OPT_OUT_LABEL,
    QUALIFICATION_MARKER,
    READINESS_CONTEXT,
    ManagedCiContract,
    ManagedCiProbeContext,
    UNPROTECTED_OVERRIDE_TRAILER,
    assess_exact_head_protection,
    _dispatch_v2_qualification,
    _ensure_v2_intent,
    _v2_failed_jobs,
    _v2_correlated_status,
    _api_list,
    activate_managed_ci,
    dispatch_final_qualification,
    evaluate_managed_ci_readiness,
    intermediate_managed_checks,
    preflight_managed_ci_creation,
    prepare_v2_merge,
    publish_manual_v2_qualification,
    publish_round_readiness,
    release_adopted_managed_ci,
    revalidate_adopted_managed_ci,
    OrdinaryRecoveryCapability,
    refresh_ordinary_recovery_capability,
    _release_for_ordinary_recovery,
    wait_for_ordinary_recovery,
    wait_for_final_qualification,
)
from coding_review_agent_loop.orchestrator import (
    _finalize_ordinary_recovery_merge,
    _stop_on_terminal_without_status,
)
from coding_review_agent_loop.runner import CommandResult

from agent_loop_helpers import FakeRunner, make_config


WORKFLOW = """
name: CI
on:
  workflow_dispatch:
    inputs:
      expected_head_sha: {required: true}
jobs:
  route:
    if: contains(github.event.pull_request.labels.*.name, 'agent-loop-managed')
  aggregate:
    name: final-ci/exact-head
"""

V2_WORKFLOW = """
# agent-loop-managed
# expected_head_sha
# AGENT_LOOP_MANAGED_CI_V2
name: CI
on:
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

SUPPRESSING_V2_WORKFLOW = V2_WORKFLOW + """
# AGENT_LOOP_MANAGED_CI_UNLABELED_RECOVERY_V1
on:
  pull_request:
    types: [opened, unlabeled]
"""

SUPPRESSING_V2_WORKFLOW_WITHOUT_RECOVERY = V2_WORKFLOW + """
on:
  pull_request:
    types: [opened]
"""


class ManagedRunner(FakeRunner):
    def __init__(
        self,
        *,
        workflow=WORKFLOW,
        base_ref="main",
        handoff_completes=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.workflow = workflow
        self.base_ref = base_ref
        self.handoff_completes = handoff_completes
        self.label_applied = False

    def _run_locked(self, args, *, cwd, check):
        cmd = list(args)
        if (
            cmd[:2] == ["gh", "api"]
            and cmd[2].startswith("repos/OWNER/REPO/contents/.github/workflows/ci.yml")
        ):
            cmd, cwd_path = self._record_command(args, cwd)
            return CommandResult(cmd, cwd_path, self.workflow, "", 0)
        if cmd[:3] == ["gh", "api", "repos/OWNER/REPO/pulls/7"]:
            cmd, cwd_path = self._record_command(args, cwd)
            payload = {
                "head": {
                    "repo": {"full_name": "OWNER/REPO"},
                    "sha": "abc123",
                    "ref": "feature",
                },
                "base": {"ref": self.base_ref},
                "labels": [],
            }
            return CommandResult(cmd, cwd_path, json.dumps(payload), "", 0)
        if cmd[:3] == ["gh", "api", f"repos/OWNER/REPO/labels/{MANAGED_LABEL}"]:
            cmd, cwd_path = self._record_command(args, cwd)
            return CommandResult(cmd, cwd_path, "{}", "", 0)
        if "repos/OWNER/REPO/actions/workflows/ci.yml/runs?" in " ".join(cmd):
            cmd, cwd_path = self._record_command(args, cwd)
            runs = (
                [{"id": 2, "status": "completed", "conclusion": "success"}]
                if self.label_applied and self.handoff_completes
                else [{"id": 1, "status": "completed", "conclusion": "success"}]
            )
            return CommandResult(cmd, cwd_path, json.dumps({"workflow_runs": runs}), "", 0)
        if cmd[:4] == ["gh", "api", "--method", "DELETE"]:
            self.label_applied = False
        elif "repos/OWNER/REPO/issues/7/labels" in cmd:
            self.label_applied = True
        return super()._run_locked(args, cwd=cwd, check=check)


class V2ManagedRunner(ManagedRunner):
    def __init__(
        self,
        *,
        rest_pr=None,
        workflow_runs=None,
        intent_comments=None,
        jobs=None,
        actor_login="agent-loop",
        actor_id=1,
        advertised_actor=None,
        missing_advertised_actor=False,
        workflow_returncode=0,
        workflow_stderr="",
        issue_events=None,
        unreadable_issue_events_after_label=False,
        **kwargs,
    ):
        workflow = kwargs.pop("workflow", V2_WORKFLOW)
        super().__init__(workflow=workflow, **kwargs)
        self.rest_pr = {
            "head": {
                "repo": {"full_name": "OWNER/REPO"},
                "sha": "abc123",
                "ref": "agent-loop/managed-643",
            },
            "base": {"ref": "main"},
            "user": {"login": actor_login, "id": actor_id},
            "labels": [{"name": MANAGED_LABEL}],
            "draft": True,
        }
        if rest_pr:
            self.rest_pr.update(rest_pr)
        self.workflow_runs = list(workflow_runs or [])
        self.intent_comments = list(intent_comments or [])
        self.jobs = list(jobs or [])
        self.actor_login = actor_login
        self.actor_id = actor_id
        self.advertised_actor = advertised_actor or actor_login
        self.missing_advertised_actor = missing_advertised_actor
        self.workflow_returncode = workflow_returncode
        self.workflow_stderr = workflow_stderr
        self.issue_events = list(issue_events or [])
        self.unreadable_issue_events_after_label = unreadable_issue_events_after_label
        self.labels_posted = False

    def _run_locked(self, args, *, cwd, check):
        cmd = list(args)
        endpoint = next(
            (part for part in cmd if isinstance(part, str) and part.startswith("repos/")), ""
        )
        if cmd == ["gh", "api", "user"]:
            cmd, cwd_path = self._record_command(args, cwd)
            return CommandResult(
                cmd,
                cwd_path,
                json.dumps({"login": self.actor_login, "id": self.actor_id}),
                "",
                0,
            )
        if endpoint.endswith("/actions/variables/AGENT_LOOP_MANAGED_ACTOR"):
            cmd, cwd_path = self._record_command(args, cwd)
            if self.missing_advertised_actor:
                return CommandResult(cmd, cwd_path, "", "HTTP 404: Not Found", 1)
            return CommandResult(cmd, cwd_path, json.dumps({"value": self.advertised_actor}), "", 0)
        if endpoint.startswith("repos/OWNER/REPO/contents/.github/workflows/ci.yml"):
            cmd, cwd_path = self._record_command(args, cwd)
            return CommandResult(
                cmd,
                cwd_path,
                self.workflow if self.workflow_returncode == 0 else "",
                self.workflow_stderr,
                self.workflow_returncode,
            )
        if endpoint == "repos/OWNER/REPO/pulls/7":
            cmd, cwd_path = self._record_command(args, cwd)
            return CommandResult(cmd, cwd_path, json.dumps(self.rest_pr), "", 0)
        if endpoint.startswith("repos/OWNER/REPO/issues/7/events?"):
            cmd, cwd_path = self._record_command(args, cwd)
            if self.unreadable_issue_events_after_label and self.labels_posted:
                return CommandResult(cmd, cwd_path, "", "events unavailable", 1)
            return CommandResult(cmd, cwd_path, json.dumps(self.issue_events), "", 0)
        if endpoint == "repos/OWNER/REPO/issues/7/labels" and "POST" in cmd:
            cmd, cwd_path = self._record_command(args, cwd)
            self.labels_posted = True
            self.rest_pr["labels"] = [{"name": MANAGED_LABEL}]
            self.issue_events.append(label_event())
            return CommandResult(cmd, cwd_path, "{}", "", 0)
        if endpoint == f"repos/OWNER/REPO/issues/7/labels/{MANAGED_LABEL}" and "DELETE" in cmd:
            cmd, cwd_path = self._record_command(args, cwd)
            self.rest_pr["labels"] = []
            self.issue_events.append(label_event(event="unlabeled"))
            return CommandResult(cmd, cwd_path, "", "", 0)
        if endpoint == "repos/OWNER/REPO/commits/main":
            cmd, cwd_path = self._record_command(args, cwd)
            return CommandResult(cmd, cwd_path, json.dumps({"sha": "base-sha"}), "", 0)
        if endpoint.startswith("repos/OWNER/REPO/issues/7/comments?"):
            cmd, cwd_path = self._record_command(args, cwd)
            return CommandResult(cmd, cwd_path, json.dumps(self.intent_comments), "", 0)
        if endpoint == "repos/OWNER/REPO/issues/7/comments" and "POST" in cmd:
            cmd, cwd_path = self._record_command(args, cwd)
            return CommandResult(cmd, cwd_path, json.dumps({"id": 17}), "", 0)
        if endpoint.startswith("repos/OWNER/REPO/issues/comments/"):
            cmd, cwd_path = self._record_command(args, cwd)
            return CommandResult(cmd, cwd_path, "{}", "", 0)
        if "/actions/workflows/ci.yml/runs?event=workflow_dispatch" in endpoint:
            cmd, cwd_path = self._record_command(args, cwd)
            return CommandResult(cmd, cwd_path, json.dumps({"workflow_runs": self.workflow_runs}), "", 0)
        if endpoint.endswith("/jobs?filter=latest&per_page=100"):
            cmd, cwd_path = self._record_command(args, cwd)
            return CommandResult(cmd, cwd_path, json.dumps({"jobs": self.jobs}), "", 0)
        if endpoint.startswith("repos/OWNER/REPO/actions/runs/"):
            cmd, cwd_path = self._record_command(args, cwd)
            run_id = endpoint.rsplit("/", 1)[-1]
            run = next((run for run in self.workflow_runs if str(run.get("id")) == run_id), {})
            return CommandResult(cmd, cwd_path, json.dumps(run), "", 0)
        if cmd[:3] == ["gh", "pr", "view"] and "--jq" in cmd and ".headRefOid" in cmd:
            cmd, cwd_path = self._record_command(args, cwd)
            return CommandResult(cmd, cwd_path, f"{self.pr_payload.get('headRefOid')}\n", "", 0)
        return super()._run_locked(args, cwd=cwd, check=check)


class ManualQualificationRunner(V2ManagedRunner):
    """Model GitHub's draft/ready transition for manual qualification tests."""

    @staticmethod
    def _gh_argv_error(cmd):
        if cmd[:3] == ["gh", "pr", "ready"] and "--undo" in cmd:
            return None
        return FakeRunner._gh_argv_error(cmd)

    def _run_locked(self, args, *, cwd, check):
        cmd = list(args)
        if cmd[:3] == ["gh", "pr", "ready"]:
            cmd, cwd_path = self._record_command(args, cwd)
            self.rest_pr["draft"] = "--undo" in cmd
            return CommandResult(cmd, cwd_path, "", "", 0)
        return super()._run_locked(args, cwd=cwd, check=check)


def test_protection_assessment_distinguishes_private_free_plan_limit(tmp_path):
    runner = FakeRunner(
        pr_branch_protection_returncode=1,
        pr_branch_protection_stderr="Upgrade to GitHub Pro or make this repository public",
    )

    assessment = assess_exact_head_protection(
        runner,
        context=ManagedCiProbeContext("OWNER/REPO", "gh", tmp_path),
        base="main",
    )

    assert assessment.state == "plan_limited"


def test_private_free_plan_limits_on_both_protection_endpoints_remain_override_eligible(tmp_path):
    runner = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        repo_payload={"private": True},
        pr_branch_protection_returncode=1,
        pr_branch_protection_stderr="HTTP 403: Upgrade to GitHub Pro or make this repository public",
        pr_effective_rules_returncode=1,
        pr_effective_rules_stderr="HTTP 403: Upgrade to GitHub Pro or make this repository public",
    )

    readiness = evaluate_managed_ci_readiness(
        runner,
        context=ManagedCiProbeContext("OWNER/REPO", "gh", tmp_path),
        base="main",
        trusted_actor="agent-loop",
    )

    assert readiness.protection.state == "plan_limited"
    assert readiness.state == "override_eligible"
    assert any("/rules/branches/" in " ".join(command) for command, _ in runner.commands)


def test_protection_assessment_accepts_array_rules_and_rejects_voluntary_rulesets(tmp_path):
    context = ManagedCiProbeContext("OWNER/REPO", "gh", tmp_path)
    rule = {"ruleset_id": 8}
    required = {"contexts": []}
    active_rule = {
        "enforcement": "active", "bypass_actors": [],
        "rules": [{"type": "required_status_checks", "parameters": {"required_status_checks": [{"context": FINAL_CONTEXT}]}}],
    }
    strict = assess_exact_head_protection(
        FakeRunner(pr_branch_protection_payload=required, pr_effective_rules_payload=[rule], pr_rulesets_payload={8: active_rule}),
        context=context, base="main",
    )
    assert strict.state == "strict"

    bypassable = dict(active_rule, bypass_actors=[{"actor_id": 1}])
    voluntary = assess_exact_head_protection(
        FakeRunner(pr_branch_protection_payload=required, pr_effective_rules_payload=[rule], pr_rulesets_payload={8: bypassable}),
        context=context, base="main",
    )
    assert voluntary.state == "voluntary"

    evaluate_mode = dict(active_rule, enforcement="evaluate")
    voluntary = assess_exact_head_protection(
        FakeRunner(pr_branch_protection_payload=required, pr_effective_rules_payload=[rule], pr_rulesets_payload={8: evaluate_mode}),
        context=context, base="main",
    )
    assert voluntary.state == "voluntary"


def test_protection_assessment_never_treats_empty_admin_response_as_enforced(tmp_path):
    assessment = assess_exact_head_protection(
        FakeRunner(pr_branch_protection_payload={"contexts": [FINAL_CONTEXT]}, pr_enforce_admins_payload={}),
        context=ManagedCiProbeContext("OWNER/REPO", "gh", tmp_path), base="main",
    )

    assert assessment.state == "indeterminate"


def metadata(*, base_branch="main"):
    return PullRequestMetadata(
        number=7,
        repo="OWNER/REPO",
        title="Managed CI",
        head_branch="feature",
        base_branch=base_branch,
        head_sha="abc123",
        url="https://github.com/OWNER/REPO/pull/7",
    )


def test_readiness_resolves_default_base_and_distinguishes_missing_actor_variable(tmp_path):
    context = ManagedCiProbeContext("OWNER/REPO", "gh", tmp_path)
    ready = evaluate_managed_ci_readiness(
        V2ManagedRunner(
            workflow=SUPPRESSING_V2_WORKFLOW,
            pr_branch_protection_payload={"contexts": [FINAL_CONTEXT]},
        ),
        context=context, base=None, trusted_actor=" agent-loop ",
    )

    assert ready.state == "strict_ready"
    assert ready.base == "main"

    missing = evaluate_managed_ci_readiness(
        V2ManagedRunner(workflow=SUPPRESSING_V2_WORKFLOW, missing_advertised_actor=True),
        context=context, base="main", trusted_actor="agent-loop",
    )
    assert missing.state == "ordinary_fallback"
    assert missing.advertised_actor is None
    assert missing.remediation


def test_suppressing_v2_without_recovery_marker_is_invalid_and_cannot_create_managed_pr(tmp_path):
    runner = V2ManagedRunner(workflow=SUPPRESSING_V2_WORKFLOW_WITHOUT_RECOVERY)
    context = ManagedCiProbeContext("OWNER/REPO", "gh", tmp_path)

    readiness = evaluate_managed_ci_readiness(
        runner, context=context, base="main", trusted_actor="agent-loop"
    )

    assert readiness.state == "invalid"
    assert readiness.recovery_capable is False
    assert preflight_managed_ci_creation(
        runner,
        config=make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop"),
        issue_number=643,
    ) is None


def test_missing_workflow_is_a_deterministic_ordinary_ci_fallback(tmp_path):
    readiness = evaluate_managed_ci_readiness(
        V2ManagedRunner(
            workflow_returncode=1,
            workflow_stderr="HTTP 404: Not Found",
        ),
        context=ManagedCiProbeContext("OWNER/REPO", "gh", tmp_path),
        base="main",
        trusted_actor="agent-loop",
    )

    assert readiness.state == "ordinary_fallback"
    assert readiness.workflow_v2 is False


def test_suppressing_v2_preflight_falls_back_without_override_and_uses_nonce_with_override(tmp_path):
    runner = V2ManagedRunner(workflow=SUPPRESSING_V2_WORKFLOW)
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")

    assert preflight_managed_ci_creation(runner, config=config, issue_number=643) is None

    override_config = make_config(
        tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    intent = preflight_managed_ci_creation(runner, config=override_config, issue_number=643)

    assert intent is not None
    assert intent.audit_nonce


def test_override_activation_requires_the_preflight_nonce_and_releases_label_on_mismatch(tmp_path):
    nonce = "nonce-from-preflight"
    runner = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        rest_pr={"body": f"{UNPROTECTED_OVERRIDE_TRAILER} nonce={nonce}"},
    )
    config = make_config(
        tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True, managed_ci_expected_override_nonce=nonce,
    )

    contract = activate_managed_ci(runner, config=config, pr_number=7, metadata=metadata())

    assert contract is not None
    assert contract.audit_nonce == nonce
    assert contract.audit_comment_id == 17
    assert any(UNPROTECTED_OVERRIDE_TRAILER in " ".join(command) for command, _ in runner.commands)

    mismatch = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        rest_pr={"body": f"{UNPROTECTED_OVERRIDE_TRAILER} nonce={nonce}"},
    )
    rejected = make_config(
        tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True, managed_ci_expected_override_nonce="different",
    )
    assert activate_managed_ci(mismatch, config=rejected, pr_number=7, metadata=metadata()) is None
    assert any(
        command[:5] == ["gh", "api", "--method", "DELETE", f"repos/OWNER/REPO/issues/7/labels/{MANAGED_LABEL}"]
        for command, _ in mismatch.commands
    )


def _resume_audit(*, head="old-head", repo="OWNER/REPO", base="main"):
    return {
        "id": 41,
        "user": {"login": "agent-loop", "id": 1},
        "body": (
            f"{UNPROTECTED_OVERRIDE_TRAILER} nonce=old-nonce repo={repo} "
            f"base={base} head={head} protection=voluntary"
        ),
    }


def test_pr_mode_resumes_only_from_immutable_issue_draft_facts_and_mints_fresh_audit(tmp_path):
    config = make_config(
        tmp_path,
        auto_merge=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    runner = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        rest_pr={"draft": True},
        issue_events=[label_event()],
        intent_comments=[_resume_audit()],
    )

    contract = activate_managed_ci(runner, config=config, pr_number=7, metadata=metadata())

    assert contract is not None
    assert contract.activation_path == "managed"
    assert contract.audit_nonce and contract.audit_nonce != "old-nonce"
    assert contract.audit_comment_id == 17
    assert contract.intent_generation
    assert any("active_label_event_id=101" in " ".join(cmd) for cmd, _ in runner.commands)


def test_explicit_manual_reentry_reconstructs_ready_pr_as_draft_before_labeling(tmp_path):
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
    )
    runner = ManualQualificationRunner(
        rest_pr={"draft": False, "labels": []},
        issue_events=[],
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT]},
    )

    contract = activate_managed_ci(runner, config=config, pr_number=7, metadata=metadata())

    assert contract is not None
    assert contract.issue_created_pr is True
    assert contract.intent_generation
    commands = [command for command, _cwd in runner.commands]
    undo_index = next(index for index, command in enumerate(commands) if "--undo" in command)
    label_index = next(
        index for index, command in enumerate(commands)
        if command[:5] == ["gh", "api", "--method", "POST", "repos/OWNER/REPO/issues/7/labels"]
    )
    assert undo_index < label_index


def test_explicit_manual_reentry_fails_closed_when_ready_to_draft_transition_does_not_stick(tmp_path):
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
    )
    runner = V2ManagedRunner(rest_pr={"draft": False, "labels": []})

    with pytest.raises(AgentLoopError, match="re-entry could not make"):
        activate_managed_ci(runner, config=config, pr_number=7, metadata=metadata())


def test_pr_mode_strict_resume_does_not_require_or_write_unprotected_audit(tmp_path):
    config = make_config(
        tmp_path,
        auto_merge=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
    )
    runner = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        issue_events=[label_event()],
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT]},
        pr_enforce_admins_payload={"enabled": True},
    )

    contract = activate_managed_ci(runner, config=config, pr_number=7, metadata=metadata())

    assert contract is not None
    assert contract.activation_path == "managed"
    assert contract.audit_nonce is None
    assert not any(
        cmd[:3] == ["gh", "api", "--method"]
        and "issues/7/comments" in " ".join(cmd)
        and "AGENT_MANAGED_CI_UNPROTECTED_OVERRIDE_V1" in " ".join(cmd)
        for cmd, _cwd in runner.commands
    )


def test_pr_mode_treats_edited_or_missing_audit_as_ordinary_fallback(tmp_path):
    config = make_config(
        tmp_path,
        auto_merge=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    runner = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        issue_events=[label_event()],
        intent_comments=[_resume_audit(repo="EVIL/REPO")],
    )

    contract = activate_managed_ci(runner, config=config, pr_number=7, metadata=metadata())

    assert contract is not None
    assert contract.activation_path == "ordinary_fallback"
    assert contract.ordinary_recovery is not None
    assert any(
        cmd[:5] == ["gh", "api", "--method", "DELETE", f"repos/OWNER/REPO/issues/7/labels/{MANAGED_LABEL}"]
        for cmd, _ in runner.commands
    )


def test_pr_mode_keeps_ordinary_recovery_capability_when_label_event_is_temporarily_unreadable(tmp_path):
    config = make_config(
        tmp_path,
        auto_merge=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    runner = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        # The PR API still proves the managed draft tuple, but the timeline
        # has not yielded the label event yet.
        issue_events=[],
    )

    contract = activate_managed_ci(runner, config=config, pr_number=7, metadata=metadata())

    assert contract is not None
    assert contract.activation_path == "ordinary_fallback"
    assert contract.ordinary_recovery is not None
    assert contract.ordinary_recovery.released_label_event_id is None


def test_pr_mode_keeps_readable_non_owned_label_event_fail_closed(tmp_path):
    config = make_config(
        tmp_path,
        auto_merge=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    runner = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        issue_events=[label_event(login="collaborator", actor_id=2)],
    )

    contract = activate_managed_ci(runner, config=config, pr_number=7, metadata=metadata())

    assert contract is not None
    assert contract.activation_path == "ordinary_fallback"
    assert contract.ordinary_recovery is None
    assert any(
        cmd[:5] == ["gh", "api", "--method", "DELETE", f"repos/OWNER/REPO/issues/7/labels/{MANAGED_LABEL}"]
        for cmd, _ in runner.commands
    )


def test_resume_intent_generation_ignores_historical_same_head_ledger_and_run(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    historical = v2_intent_comment(run_id=100, run_attempt=1)
    contract = v2_contract(intent_generation="fresh-generation")
    runner = V2ManagedRunner(
        intent_comments=[historical],
        workflow_runs=[v2_run(run_id=100)],
    )

    _ensure_v2_intent(runner, config=config, pr_number=7, expected_head_sha="abc123", contract=contract)

    assert contract.nonce != "nonce-1"
    assert contract.attached_run_id is None
    assert any("issues/7/comments" in " ".join(cmd) and "POST" in cmd for cmd, _ in runner.commands)


def test_intent_history_malformed_page_fails_closed_instead_of_minting_nonce(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    contract = v2_contract(intent_generation="fresh-generation")
    runner = V2ManagedRunner(intent_comments=[{"id": 17}, "malformed-entry"])

    with pytest.raises(AgentLoopError, match="Unable to inspect managed-CI v2 intent history"):
        _ensure_v2_intent(
            runner, config=config, pr_number=7, expected_head_sha="abc123", contract=contract,
        )


def test_ordinary_fallback_readies_draft_then_merges_same_exact_head(tmp_path, monkeypatch):
    config = make_config(tmp_path, auto_merge=True)
    runner = V2ManagedRunner(issue_events=[])
    capability = OrdinaryRecoveryCapability(
        pr_number=7, repository="OWNER/REPO", base_ref="main", expected_head_sha="abc123",
        released_label_event_id=None, released_at=100, prior_run_ids=frozenset({2}),
    )
    monkeypatch.setattr(orchestrator, "refresh_ordinary_recovery_capability", lambda *args, **kwargs: capability)
    monkeypatch.setattr(
        orchestrator,
        "wait_for_ordinary_recovery",
        lambda *args, **kwargs: SimpleNamespace(status="passed"),
    )
    monkeypatch.setattr(
        orchestrator,
        "get_pr_review_context",
        lambda *args, **kwargs: SimpleNamespace(metadata=metadata()),
    )
    merged: list[tuple[int, str | None]] = []
    monkeypatch.setattr(
        orchestrator,
        "merge_pr",
        lambda _runner, _config, number, *, expected_head_sha: merged.append((number, expected_head_sha)),
    )

    _finalize_ordinary_recovery_merge(
        runner, config=config, pr_number=7, capability=capability,
    )

    assert any(command[:3] == ["gh", "pr", "ready"] for command, _ in runner.commands)
    assert merged == [(7, "abc123")]


def test_fake_runner_models_gh_parser_failure_when_check_is_true(tmp_path):
    runner = FakeRunner()

    with pytest.raises(AgentLoopError, match="unknown flag: --slurp"):
        runner.run(["gh", "api", "--paginate", "--slurp", "repos/OWNER/REPO/issues/7/events"], cwd=tmp_path)


def test_ordinary_recovery_rejects_green_checks_without_post_release_run(monkeypatch, tmp_path):
    config = make_config(tmp_path, auto_merge=True, ci_timeout_seconds=1, ci_poll_interval_seconds=1)
    capability = OrdinaryRecoveryCapability(
        pr_number=7, repository="OWNER/REPO", base_ref="main", expected_head_sha="abc123",
        released_label_event_id=101, released_at=100, prior_run_ids=frozenset({2}),
    )
    passing = checks(passing=(PullRequestCheck("test", "check_run", "success"),), required=("test",))
    monkeypatch.setattr(managed_ci, "get_pr_head_sha", lambda *args, **kwargs: "abc123")
    monkeypatch.setattr(managed_ci, "get_pr_mergeability", lambda *args, **kwargs: type("M", (), {"state": "mergeable"})())
    monkeypatch.setattr(managed_ci, "_workflow_runs_payload", lambda *args, **kwargs: [{"id": 2, "status": "completed", "conclusion": "success"}])
    monkeypatch.setattr(managed_ci, "get_pr_checks", lambda *args, **kwargs: passing)

    outcome = wait_for_ordinary_recovery(runner=FakeRunner(), config=config, capability=capability, metadata=metadata())

    assert outcome.status == "timeout"


def test_ordinary_recovery_accepts_current_head_run_without_local_clock_filter(monkeypatch, tmp_path):
    config = make_config(tmp_path, auto_merge=True, ci_timeout_seconds=1, ci_poll_interval_seconds=1)
    capability = OrdinaryRecoveryCapability(
        pr_number=7, repository="OWNER/REPO", base_ref="main", expected_head_sha="abc123",
        released_label_event_id=101, released_at=2_000, prior_run_ids=frozenset(),
    )
    passing = checks(passing=(PullRequestCheck("test", "check_run", "success"),), required=("test",))
    monkeypatch.setattr(managed_ci, "get_pr_head_sha", lambda *args, **kwargs: "abc123")
    monkeypatch.setattr(managed_ci, "get_pr_mergeability", lambda *args, **kwargs: type("M", (), {"state": "mergeable"})())
    monkeypatch.setattr(
        managed_ci,
        "_workflow_runs_payload",
        lambda *args, **kwargs: [{
            "id": 3, "status": "completed", "conclusion": "success",
            "created_at": "1970-01-01T00:00:00Z",
        }],
    )
    monkeypatch.setattr(managed_ci, "get_pr_checks", lambda *args, **kwargs: passing)

    outcome = wait_for_ordinary_recovery(
        runner=FakeRunner(), config=config, capability=capability, metadata=metadata(),
    )

    assert outcome.status == "passed"


def test_refresh_ordinary_recovery_rebinds_changed_head_and_resets_run_baseline(tmp_path):
    capability = OrdinaryRecoveryCapability(
        pr_number=7, repository="OWNER/REPO", base_ref="main", expected_head_sha="abc123",
        released_label_event_id=101, released_at=100, prior_run_ids=frozenset({2}),
    )
    runner = V2ManagedRunner(
        rest_pr={
            "head": {"repo": {"full_name": "OWNER/REPO"}, "sha": "new-head", "ref": "feature"},
            "labels": [],
        },
        issue_events=[label_event(), label_event(event="unlabeled")],
    )

    refreshed = refresh_ordinary_recovery_capability(
        runner, config=make_config(tmp_path), capability=capability,
    )

    assert refreshed is not None
    assert refreshed.expected_head_sha == "new-head"
    assert refreshed.prior_run_ids == frozenset()


def v2_contract(**overrides):
    fields = {
        "protocol_version": 2,
        "base_ref": "main",
        "trusted_actor_login": "agent-loop",
        "trusted_actor_id": 1,
        "workflow_revision": "base-sha",
        "nonce": "nonce-1",
    }
    fields.update(overrides)
    return ManagedCiContract(**fields)


def test_publish_manual_v2_qualification_releases_label_readies_and_audits_sha(tmp_path):
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
    )
    runner = ManualQualificationRunner(issue_events=[label_event()])
    contract = v2_contract(
        issue_created_pr=True,
        active_label_event_id=101,
        invocation_applied_label=True,
        protection_mode="strict",
        attached_run_id=100,
        run_attempt=2,
        intent_generation="generation-1",
    )

    qualified = publish_manual_v2_qualification(
        runner,
        config=config,
        pr_number=7,
        expected_head_sha="abc123",
        contract=contract,
        reviewers=("Codex", "Claude"),
    )

    assert qualified == "abc123"
    commands = [command for command, _cwd in runner.commands]
    release_index = next(
        index for index, command in enumerate(commands)
        if command[:5] == [
            "gh", "api", "--method", "DELETE",
            f"repos/OWNER/REPO/issues/7/labels/{MANAGED_LABEL}",
        ]
    )
    ready_index = next(index for index, command in enumerate(commands) if command[:4] == ["gh", "pr", "ready", "7"])
    assert release_index < ready_index
    assert not any(
        command[:5] == [
            "gh", "api", "--method", "POST",
            "repos/OWNER/REPO/issues/7/labels",
        ] and QUALIFICATION_MARKER not in " ".join(command)
        for command in commands
    )
    audit_commands = [command for command in commands if QUALIFICATION_MARKER in " ".join(command)]
    assert len(audit_commands) == 1
    assert "qualified_head=abc123" in " ".join(" ".join(command) for command in audit_commands)
    assert not any(command[:3] == ["gh", "pr", "merge"] for command in commands)


def test_publish_manual_v2_qualification_publishes_unprotected_residual_risk(tmp_path):
    config = make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop")
    runner = ManualQualificationRunner(
        rest_pr={"draft": False}, issue_events=[label_event()],
    )
    contract = v2_contract(
        adopted_existing_pr=True,
        active_label_event_id=101,
        protection_mode="voluntary",
    )

    publish_manual_v2_qualification(
        runner,
        config=config,
        pr_number=7,
        expected_head_sha="abc123",
        contract=contract,
        reviewers=("Codex",),
    )

    audit = next(" ".join(command) for command, _cwd in runner.commands if QUALIFICATION_MARKER in " ".join(command))
    assert "GitHub cannot force a human or other automation" in audit
    assert not any(command[:3] == ["gh", "pr", "ready"] for command, _cwd in runner.commands)


def test_publish_manual_v2_qualification_rejects_head_change_before_release(tmp_path, monkeypatch):
    config = make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop")
    runner = ManualQualificationRunner(issue_events=[label_event()])
    contract = v2_contract(
        issue_created_pr=True, active_label_event_id=101, invocation_applied_label=True,
        protection_mode="strict",
    )
    monkeypatch.setattr(managed_ci, "get_pr_head_sha", lambda *args, **kwargs: "new-head")

    with pytest.raises(AgentLoopError, match="head changed before"):
        publish_manual_v2_qualification(
            runner,
            config=config,
            pr_number=7,
            expected_head_sha="abc123",
            contract=contract,
            reviewers=("Codex",),
        )

    assert not any(command[:3] == ["gh", "pr", "ready"] for command, _cwd in runner.commands)
    assert not any(command[:4] == ["gh", "api", "--method", "DELETE"] for command, _cwd in runner.commands)


def test_publish_manual_v2_qualification_rejects_head_change_after_audit(tmp_path, monkeypatch):
    config = make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop")
    runner = ManualQualificationRunner(issue_events=[label_event()])
    contract = v2_contract(
        issue_created_pr=True, active_label_event_id=101, invocation_applied_label=True,
        protection_mode="strict",
    )
    heads = iter(("abc123", "new-head"))
    monkeypatch.setattr(managed_ci, "get_pr_head_sha", lambda *args, **kwargs: next(heads))

    with pytest.raises(AgentLoopError, match="after qualification publication"):
        publish_manual_v2_qualification(
            runner,
            config=config,
            pr_number=7,
            expected_head_sha="abc123",
            contract=contract,
            reviewers=("Codex",),
        )

    assert any(QUALIFICATION_MARKER in " ".join(command) for command, _cwd in runner.commands)
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def v2_run(
    *,
    run_id=100,
    attempt=1,
    status="completed",
    conclusion="success",
    name="managed-ci-v2 nonce=nonce-1",
    display_title=None,
    path=".github/workflows/ci.yml@main",
):
    return {
        "id": run_id,
        "run_attempt": attempt,
        "name": name,
        "display_title": display_title,
        "event": "workflow_dispatch",
        "path": path,
        "head_branch": "main",
        "head_sha": "base-sha",
        "status": status,
        "conclusion": conclusion,
    }


def v2_intent_comment(
    *, nonce="nonce-1", run_id=None, run_attempt=None, state=None,
    terminal_run_id=None, terminal_run_attempt=None,
    terminal_attempts=None,
):
    payload = {
        "repository": "OWNER/REPO",
        "pr": 7,
        "expected_head_sha": "abc123",
        "nonce": nonce,
        "created_at": 1,
        "run_id": run_id,
        "run_attempt": run_attempt,
    }
    if state is not None:
        payload["state"] = state
    if terminal_run_id is not None:
        payload["terminal_run_id"] = terminal_run_id
    if terminal_run_attempt is not None:
        payload["terminal_run_attempt"] = terminal_run_attempt
    if terminal_attempts is not None:
        payload["terminal_attempts"] = [
            {"run_id": run_id, "run_attempt": attempt}
            for run_id, attempt in terminal_attempts
        ]
    return {
        "id": 17,
        "user": {"login": "agent-loop", "id": 1},
        "body": f"<!-- AGENT_MANAGED_CI_INTENT_V2 {json.dumps(payload)} -->",
    }


def checks(*, pending=(), passing=(), failing=(), required=(FINAL_CONTEXT,), missing=()):
    return PullRequestChecks(
        state="failing" if failing else "pending" if pending else "passing",
        required_checks=required,
        passing=passing,
        pending=pending,
        failing=failing,
        missing_required=missing,
        branch_protection_status="configured",
    )


def test_activate_managed_ci_only_for_complete_supported_contract(tmp_path):
    config = make_config(tmp_path, auto_merge=True)
    runner = ManagedRunner(
        pr_payload={"headRefOid": "abc123"},
        pr_status_payload={
            "statuses": [{"context": FINAL_CONTEXT, "state": "pending"}]
        },
    )

    contract = activate_managed_ci(
        runner, config=config, pr_number=7, metadata=metadata()
    )

    assert contract == ManagedCiContract()
    assert any(
        cmd[:5] == ["gh", "api", "--method", "POST", "repos/OWNER/REPO/issues/7/labels"]
        for cmd, _cwd in runner.commands
    )


def test_activate_managed_ci_preserves_legacy_behavior_without_markers(tmp_path):
    config = make_config(tmp_path, auto_merge=True)
    runner = ManagedRunner(workflow="name: CI\n")

    assert activate_managed_ci(
        runner, config=config, pr_number=7, metadata=metadata()
    ) is None


def test_activate_managed_ci_fails_closed_for_partial_contract(tmp_path):
    config = make_config(tmp_path, auto_merge=True)
    runner = ManagedRunner(workflow=f"name: CI\n# {MANAGED_LABEL}\n")

    with pytest.raises(AgentLoopError, match="incomplete managed-CI contract"):
        activate_managed_ci(runner, config=config, pr_number=7, metadata=metadata())


def test_activate_managed_ci_accepts_resolved_non_main_base(tmp_path):
    config = make_config(tmp_path, auto_merge=True, base="release")
    runner = ManagedRunner(
        base_ref="release",
        pr_payload={"headRefOid": "abc123"},
        pr_status_payload={"statuses": [{"context": FINAL_CONTEXT, "state": "pending"}]},
    )

    assert activate_managed_ci(
        runner,
        config=config,
        pr_number=7,
        metadata=metadata(base_branch="release"),
    ) == ManagedCiContract()


def test_activate_managed_ci_removes_label_when_handoff_times_out(tmp_path):
    config = make_config(
        tmp_path,
        auto_merge=True,
        ci_timeout_seconds=1,
        ci_poll_interval_seconds=1,
    )
    runner = ManagedRunner(
        handoff_completes=False,
        pr_payload={"headRefOid": "abc123"},
    )

    with pytest.raises(AgentLoopError, match="handoff.*did not complete"):
        activate_managed_ci(runner, config=config, pr_number=7, metadata=metadata())

    assert runner.label_applied is False
    assert any(
        cmd[:5]
        == [
            "gh",
            "api",
            "--method",
            "DELETE",
            f"repos/OWNER/REPO/issues/7/labels/{MANAGED_LABEL}",
        ]
        for cmd, _cwd in runner.commands
    )


def test_intermediate_checks_remove_only_expected_final_pending_context():
    final = PullRequestCheck(FINAL_CONTEXT, "status_context", "pending")
    lint = PullRequestCheck("lint", "check_run", "failure")

    filtered = intermediate_managed_checks(
        checks(
            pending=(final,),
            failing=(lint,),
            required=(FINAL_CONTEXT, "test (pr-inline)"),
            missing=("test (pr-inline)",),
        )
    )

    assert filtered.state == "failing"
    assert filtered.required_checks == ("test (pr-inline)",)
    assert filtered.pending == ()
    assert filtered.failing == (lint,)
    assert filtered.missing_required == ()


def test_dispatch_and_wait_bind_final_qualification_to_exact_head(tmp_path):
    config = make_config(tmp_path, auto_merge=True, ci_poll_interval_seconds=1)
    final = {"context": FINAL_CONTEXT, "state": "success", "target_url": None}
    runner = ManagedRunner(
        pr_payload={"headRefOid": "abc123", "mergeable": "MERGEABLE", "mergeStateStatus": "CLEAN"},
        pr_status_payload={"statuses": [final]},
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT], "checks": []},
    )

    dispatch_final_qualification(
        runner,
        config=config,
        pr_number=7,
        expected_head_sha="abc123",
        head_ref="feature",
        contract=ManagedCiContract(),
    )
    outcome = wait_for_final_qualification(
        runner, config=config, pr_number=7, metadata=metadata()
    )

    assert outcome.status == "passed"
    dispatch = next(
        cmd for cmd, _cwd in runner.commands
        if "repos/OWNER/REPO/actions/workflows/ci.yml/dispatches" in cmd
    )
    assert "ref=feature" in dispatch
    assert "inputs[expected_head_sha]=abc123" in dispatch


def test_publish_round_readiness_posts_success_status(tmp_path):
    config = make_config(tmp_path, auto_merge=True)
    runner = FakeRunner()

    publish_round_readiness(runner, config=config, head_sha="abc123")

    assert runner.commands[-1][0] == [
        "gh",
        "api",
        "--method",
        "POST",
        "repos/OWNER/REPO/statuses/abc123",
        "-f",
        "state=success",
        "-f",
        f"context={READINESS_CONTEXT}",
        "-f",
        "description=Configured local pre-review verification passed",
    ]


def test_wait_for_final_qualification_returns_infrastructure_stall(tmp_path):
    config = make_config(
        tmp_path,
        auto_merge=True,
        ci_timeout_seconds=1,
        ci_poll_interval_seconds=1,
        ci_queued_grace_seconds=1,
    )
    runner = ManagedRunner(
        pr_payload={
            "headRefOid": "abc123",
            "mergeable": "MERGEABLE",
            "mergeStateStatus": "CLEAN",
        },
        pr_check_runs_payload={
            "check_runs": [
                {
                    "id": 99,
                    "name": "test (pr-inline)",
                    "status": "queued",
                    "conclusion": None,
                    "html_url": "https://github.com/OWNER/REPO/actions/runs/123/job/99",
                    "created_at": "2020-01-01T00:00:00Z",
                    "started_at": None,
                    "completed_at": None,
                }
            ]
        },
        pr_status_payload={
            "state": "pending",
            "statuses": [{"context": FINAL_CONTEXT, "state": "pending"}],
        },
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT], "checks": []},
    )

    outcome = wait_for_final_qualification(
        runner, config=config, pr_number=7, metadata=metadata()
    )

    assert outcome.status == "infrastructure_stall"
    assert outcome.stall is not None
    assert outcome.stall.checks[0].name == "test (pr-inline)"


@pytest.mark.parametrize("status", ["startup_failure", "stale"])
def test_wait_for_final_qualification_treats_all_terminal_failures_as_failed(
    tmp_path, status
):
    config = make_config(
        tmp_path, auto_merge=True, ci_timeout_seconds=1, ci_poll_interval_seconds=1
    )
    runner = ManagedRunner(
        pr_payload={
            "headRefOid": "abc123",
            "mergeable": "MERGEABLE",
            "mergeStateStatus": "CLEAN",
        },
        pr_check_runs_payload={
            "check_runs": [
                {
                    "name": FINAL_CONTEXT,
                    "status": "completed",
                    "conclusion": status,
                    "started_at": "2026-01-01T00:00:00Z",
                    "completed_at": "2026-01-01T00:01:00Z",
                }
            ]
        },
        pr_status_payload={"statuses": []},
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT], "checks": []},
    )

    outcome = wait_for_final_qualification(
        runner, config=config, pr_number=7, metadata=metadata()
    )

    assert outcome.status == "failed"


def test_v2_qualification_ignores_same_context_status_from_another_run(tmp_path):
    config = make_config(
        tmp_path, auto_merge=True, ci_timeout_seconds=1, ci_poll_interval_seconds=1
    )
    runner = ManagedRunner(
        pr_payload={
            "headRefOid": "abc123",
            "mergeable": "MERGEABLE",
            "mergeStateStatus": "CLEAN",
        },
        pr_status_payload={
            "statuses": [
                {
                    "context": FINAL_CONTEXT,
                    "state": "failure",
                    "description": "nonce=nonce-1;run_id=99;attempt=1",
                    "target_url": "https://github.com/OWNER/REPO/actions/runs/99",
                    "creator": {"login": "github-actions[bot]", "id": 41898282},
                }
            ]
        },
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT], "checks": []},
    )
    contract = ManagedCiContract(
        protocol_version=2,
        trusted_actor_login="agent-loop",
        trusted_actor_id=1,
        nonce="nonce-1",
        attached_run_id=100,
        run_attempt=1,
    )

    outcome = wait_for_final_qualification(
        runner, config=config, pr_number=7, metadata=metadata(), contract=contract
    )

    assert outcome.status == "timeout"


def test_v2_qualification_accepts_only_attached_run_status(tmp_path):
    config = make_config(
        tmp_path, auto_merge=True, ci_timeout_seconds=1, ci_poll_interval_seconds=1
    )
    runner = ManagedRunner(
        pr_payload={
            "headRefOid": "abc123",
            "mergeable": "MERGEABLE",
            "mergeStateStatus": "CLEAN",
        },
        pr_status_payload={
            "statuses": [
                {
                    "context": FINAL_CONTEXT,
                    "state": "success",
                    "description": "nonce=nonce-1;run_id=100;attempt=1",
                    "target_url": "https://github.com/OWNER/REPO/actions/runs/100",
                    "creator": {"login": "github-actions[bot]", "id": 41898282},
                }
            ]
        },
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT], "checks": []},
    )
    contract = ManagedCiContract(
        protocol_version=2,
        trusted_actor_login="agent-loop",
        trusted_actor_id=1,
        nonce="nonce-1",
        attached_run_id=100,
        run_attempt=1,
    )

    outcome = wait_for_final_qualification(
        runner, config=config, pr_number=7, metadata=metadata(), contract=contract
    )

    assert outcome.status == "passed"


def test_v2_correlated_status_uses_history_and_ignores_later_unrelated_status(tmp_path):
    config = make_config(tmp_path, auto_merge=True)
    creator = {"login": "github-actions[bot]", "id": 41898282}
    common = {
        "context": FINAL_CONTEXT,
        "target_url": "https://ghe.example/actions/runs/100/",
        "creator": creator,
    }
    runner = FakeRunner(pr_status_payload={
        "pages": [
            [{**common, "state": "pending", "description": "nonce=nonce-1;run_id=100;attempt=1", "created_at": "2026-08-20T10:00:00Z"}],
            [{**common, "state": "success", "description": "nonce=nonce-1;run_id=100;attempt=1", "created_at": "2026-08-20T10:01:00Z"}],
            [{**common, "state": "failure", "description": "nonce=nonce-1;run_id=999;attempt=1", "created_at": "2026-08-20T10:02:00Z"}],
        ]
    })
    contract = v2_contract(attached_run_id=100, run_attempt=1)

    result = _v2_correlated_status(runner, config=config, expected_head="abc123", contract=contract)

    assert result is not None
    assert result.status == "success"
    assert result.run_id == "100"


def test_v2_correlated_status_omits_unknown_attempt_token(tmp_path):
    config = make_config(tmp_path, auto_merge=True)
    runner = FakeRunner(pr_status_payload={"statuses": [{
        "context": FINAL_CONTEXT,
        "state": "success",
        "description": "nonce=nonce-1;run_id=100",
        "target_url": "https://github.com/OWNER/REPO/actions/runs/100",
        "creator": {"login": "github-actions[bot]", "id": 41898282},
    }]})
    result = _v2_correlated_status(
        runner, config=config, expected_head="abc123",
        contract=v2_contract(attached_run_id=100, run_attempt=None),
    )
    assert result is not None
    assert result.status == "success"


def test_v2_completed_run_without_publisher_status_stops_and_records_ledger(tmp_path, monkeypatch):
    config = make_config(
        tmp_path, auto_merge=True, ci_timeout_seconds=2, ci_poll_interval_seconds=1
    )
    runner = V2ManagedRunner(
        workflow_runs=[v2_run(status="completed", conclusion="cancelled")],
        pr_payload={"headRefOid": "abc123", "mergeable": "MERGEABLE", "mergeStateStatus": "CLEAN"},
        pr_status_payload={"statuses": []},
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT], "checks": []},
    )
    contract = v2_contract(
        attached_run_id=100, run_attempt=1, intent_comment_id=17,
        pr_number=7, expected_head_sha="abc123",
    )
    monkeypatch.setattr(
        managed_ci, "_v2_correlated_status", lambda *args, **kwargs: None
    )

    outcome = wait_for_final_qualification(
        runner, config=config, pr_number=7, metadata=metadata(), contract=contract
    )

    assert outcome.status == "terminal_without_status"
    assert (outcome.run_id, outcome.run_attempt) == (100, 1)
    assert outcome.workflow_conclusion == "cancelled"
    assert contract.intent_state == "terminal-no-status"
    assert any(
        '"state":"terminal-no-status"' in " ".join(command)
        for command, _cwd in runner.commands
        if "/issues/comments/17" in " ".join(command)
    )
    assert not any(
        "/statuses/abc123" in " ".join(command) and "POST" in command
        for command, _cwd in runner.commands
    )


def test_orchestrator_terminal_without_status_posts_resumable_diagnostic(tmp_path, capsys):
    config = make_config(tmp_path, auto_merge=True)
    runner = V2ManagedRunner()
    outcome = managed_ci.ManagedCiOutcome(
        status="terminal_without_status",
        run_id=100,
        run_attempt=1,
        workflow_conclusion="cancelled",
    )

    assert _stop_on_terminal_without_status(
        runner, config=config, pr_number=7, round_number=2, outcome=outcome
    ) == 0
    assert len(runner.comments) == 1
    assert "terminal workflow state `cancelled`" in runner.comments[0]
    assert "No terminal status was synthesized" in runner.comments[0]
    assert "higher attempt" in runner.comments[0]
    assert "terminal workflow state `cancelled`" in capsys.readouterr().out


def test_v2_cancelled_run_during_candidate_jobs_stops_without_publishing_status(tmp_path, monkeypatch):
    config = make_config(
        tmp_path, auto_merge=True, ci_timeout_seconds=2, ci_poll_interval_seconds=1
    )
    runner = V2ManagedRunner(
        workflow_runs=[v2_run(status="completed", conclusion="cancelled")],
        jobs=[{"name": "candidate", "conclusion": "cancelled"}],
        pr_payload={"headRefOid": "abc123", "mergeable": "MERGEABLE", "mergeStateStatus": "CLEAN"},
        pr_status_payload={"statuses": []},
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT], "checks": []},
    )
    contract = v2_contract(
        attached_run_id=100, run_attempt=1, intent_comment_id=17,
        pr_number=7, expected_head_sha="abc123",
    )
    monkeypatch.setattr(managed_ci, "_v2_correlated_status", lambda *args, **kwargs: None)

    outcome = wait_for_final_qualification(
        runner, config=config, pr_number=7, metadata=metadata(), contract=contract
    )

    assert outcome.status == "terminal_without_status"
    assert outcome.workflow_conclusion == "cancelled"
    assert (contract.terminal_run_id, contract.terminal_run_attempt) == (100, 1)


def test_v2_terminal_exclusion_does_not_attach_stale_cancelled_run(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = V2ManagedRunner(
        workflow_runs=[
            v2_run(run_id=100, attempt=1, status="completed", conclusion="cancelled"),
            v2_run(run_id=101, attempt=1, status="in_progress", conclusion=None),
        ],
        intent_comments=[v2_intent_comment(
            run_id=100, run_attempt=1, state="terminal-no-status",
            terminal_run_id=100, terminal_run_attempt=1,
        )],
    )
    contract = v2_contract()

    _dispatch_v2_qualification(
        runner, config=config, pr_number=7, expected_head_sha="abc123", contract=contract
    )

    assert (contract.attached_run_id, contract.run_attempt) == (101, 1)
    assert contract.intent_state == "attached"
    assert not any("/dispatches" in " ".join(cmd) for cmd, _cwd in runner.commands)


def test_v2_terminal_ledger_clears_old_attachment_before_fresh_dispatch(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = V2ManagedRunner(
        workflow_runs=[v2_run(run_id=100, attempt=1, status="completed", conclusion="cancelled")],
        intent_comments=[v2_intent_comment(
            run_id=100, run_attempt=1, state="terminal-no-status",
            terminal_run_id=100, terminal_run_attempt=1,
        )],
    )
    contract = v2_contract()

    _dispatch_v2_qualification(
        runner, config=config, pr_number=7, expected_head_sha="abc123", contract=contract
    )

    assert contract.attached_run_id is None
    assert contract.run_attempt is None
    assert any("/dispatches" in " ".join(cmd) for cmd, _cwd in runner.commands)


def test_v2_terminal_ledger_excludes_all_prior_cancelled_attempts(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = V2ManagedRunner(
        workflow_runs=[
            v2_run(run_id=100, attempt=2, status="completed", conclusion="cancelled"),
            v2_run(run_id=100, attempt=1, status="completed", conclusion="cancelled"),
            v2_run(run_id=101, attempt=1, status="in_progress", conclusion=None),
        ],
        intent_comments=[v2_intent_comment(
            run_id=100, run_attempt=2, state="terminal-no-status",
            terminal_run_id=100, terminal_run_attempt=2,
            terminal_attempts=((100, 1), (100, 2)),
        )],
    )
    contract = v2_contract()

    _dispatch_v2_qualification(
        runner, config=config, pr_number=7, expected_head_sha="abc123", contract=contract
    )

    assert (contract.attached_run_id, contract.run_attempt) == (101, 1)
    assert not any("/dispatches" in " ".join(cmd) for cmd, _cwd in runner.commands)


def test_v2_later_legitimate_rerun_attempt_is_accepted_after_terminal_stop(tmp_path):
    config = make_config(tmp_path, auto_merge=True, ci_timeout_seconds=1, ci_poll_interval_seconds=1)
    runner = V2ManagedRunner(
        workflow_runs=[
            v2_run(run_id=100, attempt=2, status="completed", conclusion="success"),
            v2_run(run_id=100, attempt=1, status="completed", conclusion="timed_out"),
        ],
        intent_comments=[v2_intent_comment(
            run_id=100, run_attempt=1, state="terminal-no-status",
            terminal_run_id=100, terminal_run_attempt=1,
        )],
        pr_status_payload={"statuses": [{
            "context": FINAL_CONTEXT,
            "state": "success",
            "description": "nonce=nonce-1;run_id=100;attempt=2",
            "target_url": "https://github.com/OWNER/REPO/actions/runs/100",
            "creator": {"login": "github-actions[bot]", "id": 41898282},
        }]},
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT], "checks": []},
    )
    contract = v2_contract()

    _dispatch_v2_qualification(
        runner, config=config, pr_number=7, expected_head_sha="abc123", contract=contract
    )
    outcome = wait_for_final_qualification(
        runner, config=config, pr_number=7, metadata=metadata(), contract=contract
    )

    assert outcome.status == "passed"
    assert (contract.attached_run_id, contract.run_attempt) == (100, 2)
    assert contract.terminal_run_attempt == 1


def test_v2_waiter_excludes_prior_non_cancelled_terminal_attempt(tmp_path):
    config = make_config(tmp_path, auto_merge=True, ci_timeout_seconds=1, ci_poll_interval_seconds=1)
    runner = V2ManagedRunner(
        workflow_runs=[v2_run(run_id=100, attempt=1, status="completed", conclusion="timed_out")],
        pr_status_payload={"statuses": []},
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT], "checks": []},
    )
    contract = v2_contract(
        terminal_run_id=100,
        terminal_run_attempt=1,
        terminal_attempts=((100, 1),),
    )

    outcome = wait_for_final_qualification(
        runner, config=config, pr_number=7, metadata=metadata(), contract=contract
    )

    assert outcome.status == "timeout"
    assert contract.attached_run_id is None
    assert not any('"state":"attached"' in " ".join(command) for command, _cwd in runner.commands)


def test_v2_refresh_keeps_known_attempt_when_payload_omits_run_attempt(tmp_path):
    config = make_config(tmp_path, auto_merge=True, ci_timeout_seconds=1, ci_poll_interval_seconds=1)
    run = v2_run(run_id=100, attempt=1)
    del run["run_attempt"]
    runner = V2ManagedRunner(
        workflow_runs=[run],
        pr_status_payload={"statuses": [{
            "context": FINAL_CONTEXT,
            "state": "success",
            "description": "nonce=nonce-1;run_id=100;attempt=1",
            "target_url": "https://github.com/OWNER/REPO/actions/runs/100",
            "creator": {"login": "github-actions[bot]", "id": 41898282},
        }]},
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT], "checks": []},
    )
    contract = v2_contract(attached_run_id=100, run_attempt=1)

    outcome = wait_for_final_qualification(
        runner, config=config, pr_number=7, metadata=metadata(), contract=contract
    )

    assert outcome.status == "passed"
    assert contract.run_attempt == 1


def test_v2_completed_run_failure_race_accepts_late_correlated_status(tmp_path, monkeypatch):
    config = make_config(
        tmp_path, auto_merge=True, ci_timeout_seconds=2, ci_poll_interval_seconds=1
    )
    runner = V2ManagedRunner(
        workflow_runs=[v2_run(status="completed", conclusion="failure")],
        jobs=[{"name": "unit", "conclusion": "failure"}],
        pr_payload={"headRefOid": "abc123", "mergeable": "MERGEABLE", "mergeStateStatus": "CLEAN"},
    )
    contract = v2_contract(attached_run_id=100, run_attempt=1)
    failure = PullRequestCheck(
        name=FINAL_CONTEXT, kind="status_context", status="failure",
        run_id="100", description="nonce=nonce-1;run_id=100;attempt=1",
    )
    responses = iter([None, failure])
    monkeypatch.setattr(
        managed_ci, "_v2_correlated_status", lambda *args, **kwargs: next(responses)
    )

    outcome = wait_for_final_qualification(
        runner, config=config, pr_number=7, metadata=metadata(), contract=contract
    )

    assert outcome.status == "failed"


def test_v2_qualification_rejects_numeric_prefix_tokens_and_run_url(tmp_path):
    config = make_config(
        tmp_path, auto_merge=True, ci_timeout_seconds=1, ci_poll_interval_seconds=1
    )
    runner = V2ManagedRunner(
        workflow_runs=[v2_run(run_id=100, attempt=1)],
        pr_payload={
            "headRefOid": "abc123",
            "mergeable": "MERGEABLE",
            "mergeStateStatus": "CLEAN",
        },
        pr_status_payload={
            "statuses": [
                {
                    "context": FINAL_CONTEXT,
                    "state": "success",
                    "description": "nonce=nonce-1;run_id=1001;attempt=10",
                    "target_url": "https://github.com/OWNER/REPO/actions/runs/1001",
                    "creator": {"login": "github-actions[bot]", "id": 41898282},
                }
            ]
        },
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT], "checks": []},
    )

    outcome = wait_for_final_qualification(
        runner,
        config=config,
        pr_number=7,
        metadata=metadata(),
        contract=v2_contract(attached_run_id=100, run_attempt=1),
    )

    assert outcome.status == "timeout"


def test_v2_qualification_refreshes_rerun_attempt_before_correlating_status(tmp_path):
    config = make_config(
        tmp_path, auto_merge=True, ci_timeout_seconds=1, ci_poll_interval_seconds=1
    )
    runner = V2ManagedRunner(
        workflow_runs=[v2_run(run_id=100, attempt=2)],
        pr_payload={
            "headRefOid": "abc123",
            "mergeable": "MERGEABLE",
            "mergeStateStatus": "CLEAN",
        },
        pr_status_payload={
            "statuses": [
                {
                    "context": FINAL_CONTEXT,
                    "state": "success",
                    "description": "nonce=nonce-1;run_id=100;attempt=2",
                    "target_url": "https://github.com/OWNER/REPO/actions/runs/100",
                    "creator": {"login": "github-actions[bot]", "id": 41898282},
                }
            ]
        },
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT], "checks": []},
    )
    contract = v2_contract(attached_run_id=100, run_attempt=1)

    outcome = wait_for_final_qualification(
        runner, config=config, pr_number=7, metadata=metadata(), contract=contract
    )

    assert outcome.status == "passed"
    assert contract.run_attempt == 2


@pytest.mark.parametrize(
    ("rest_pr", "config_overrides"),
    [
        ({"head": {"repo": {"full_name": "FORK/REPO"}, "sha": "abc123", "ref": "agent-loop/managed-643"}}, {}),
        ({"draft": False}, {}),
        ({"labels": []}, {}),
        ({"head": {"repo": {"full_name": "OWNER/REPO"}, "sha": "new-head", "ref": "agent-loop/managed-643"}}, {}),
        ({"user": {"login": "other", "id": 1}}, {}),
        ({}, {"managed_ci_trusted_actor": "other"}),
    ],
)
def test_v2_activation_rejects_untrusted_or_incomplete_opening_tuple(
    tmp_path, rest_pr, config_overrides
):
    settings = {"auto_merge": True, "managed_ci_trusted_actor": "agent-loop"}
    settings.update(config_overrides)
    config = make_config(tmp_path, **settings)
    runner = V2ManagedRunner(rest_pr=rest_pr, pr_payload={"headRefOid": "abc123"})

    assert activate_managed_ci(runner, config=config, pr_number=7, metadata=metadata()) is None


def test_v2_activation_and_preflight_require_authenticated_actor(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = V2ManagedRunner(pr_payload={"headRefOid": "abc123"})

    intent = preflight_managed_ci_creation(runner, config=config, issue_number=643)
    contract = activate_managed_ci(runner, config=config, pr_number=7, metadata=metadata())

    assert intent is not None
    assert intent.branch == "agent-loop/managed-643"
    assert contract is not None
    assert contract.protocol_version == 2
    assert contract.trusted_actor_login == "agent-loop"


def test_v2_preflight_accepts_exactly_one_reserved_direct_branch(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = V2ManagedRunner(pr_payload={"headRefOid": "abc123"})

    intent = preflight_managed_ci_creation(
        runner,
        config=config,
        branch="agent-loop/managed-direct-123-token",
    )

    assert intent is not None
    assert intent.branch == "agent-loop/managed-direct-123-token"
    with pytest.raises(AgentLoopError, match="exactly one"):
        preflight_managed_ci_creation(runner, config=config)
    with pytest.raises(AgentLoopError, match="exactly one"):
        preflight_managed_ci_creation(
            runner,
            config=config,
            issue_number=643,
            branch="agent-loop/managed-direct-123-token",
        )
    with pytest.raises(AgentLoopError, match="reserved"):
        preflight_managed_ci_creation(runner, config=config, branch="fix/not-reserved")


def test_v2_preflight_accepts_reserved_direct_branch_in_manual_mode(tmp_path):
    config = replace(
        make_config(tmp_path, auto_merge=False, managed_ci_trusted_actor="agent-loop"),
        managed_ci=True,
    )
    runner = V2ManagedRunner(pr_payload={"headRefOid": "abc123"})

    intent = preflight_managed_ci_creation(
        runner,
        config=config,
        branch="agent-loop/managed-direct-123-token",
    )

    assert intent is not None
    assert intent.branch == "agent-loop/managed-direct-123-token"


def adoption_workflow():
    return V2_WORKFLOW + "\n# AGENT_LOOP_MANAGED_CI_V2_PR_ADOPTION\n"


def label_event(event_id=101, *, event="labeled", login="agent-loop", actor_id=1):
    return {
        "id": event_id,
        "event": event,
        "label": {"name": MANAGED_LABEL},
        "actor": {"login": login, "id": actor_id},
    }


def test_existing_pr_adoption_requires_separate_marker_but_keeps_draft_v2(tmp_path):
    config = make_config(
        tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop",
        managed_ci_adopt_existing_pr=True,
    )
    runner = V2ManagedRunner(
        rest_pr={"draft": False, "user": {"login": "someone", "id": 55}},
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT]},
    )
    # A complete existing v2 contract still serves issue-created drafts, but
    # does not silently turn on adoption.
    assert activate_managed_ci(runner, config=config, pr_number=7, metadata=metadata()) is None


def test_existing_pr_adoption_rejects_null_head_repository_name(tmp_path):
    config = make_config(
        tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop",
        managed_ci_adopt_existing_pr=True,
    )
    runner = V2ManagedRunner(
        workflow=adoption_workflow(),
        rest_pr={
            "draft": False,
            "state": "open",
            "head": {"repo": {"full_name": None}, "sha": "abc123", "ref": "feature"},
            "labels": [],
        },
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT]},
    )

    assert activate_managed_ci(runner, config=config, pr_number=7, metadata=metadata()) is None


def test_existing_pr_adoption_reuses_trusted_label_and_revalidates(tmp_path):
    config = make_config(
        tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop",
        managed_ci_adopt_existing_pr=True,
    )
    runner = V2ManagedRunner(
        workflow=adoption_workflow(),
        rest_pr={
            "draft": False,
            "state": "open",
            "user": {"login": "someone", "id": 55},
            "head": {"repo": {"full_name": "OWNER/REPO"}, "sha": "abc123", "ref": "feature"},
            "labels": [{"name": MANAGED_LABEL}],
        },
        issue_events=[label_event()],
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT]},
    )

    contract = activate_managed_ci(runner, config=config, pr_number=7, metadata=metadata())

    assert contract is not None
    assert contract.adopted_existing_pr is True
    assert contract.invocation_applied_label is False
    assert contract.intent_generation is None
    assert revalidate_adopted_managed_ci(
        runner, config=config, pr_number=7, metadata=metadata(), contract=contract
    )
    runner.rest_pr["labels"].append({"name": []})
    assert revalidate_adopted_managed_ci(
        runner, config=config, pr_number=7, metadata=metadata(), contract=contract
    )
    assert release_adopted_managed_ci(runner, config=config, pr_number=7, contract=contract)
    assert not any(
        cmd[:5] == ["gh", "api", "--method", "DELETE", f"repos/OWNER/REPO/issues/7/labels/{MANAGED_LABEL}"]
        for cmd, _ in runner.commands
    )


def test_existing_pr_adoption_applies_and_releases_invocation_label(tmp_path):
    config = make_config(
        tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop",
        managed_ci_adopt_existing_pr=True,
    )
    runner = V2ManagedRunner(
        workflow=adoption_workflow(),
        rest_pr={"draft": False, "state": "open", "labels": []},
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT]},
    )

    contract = activate_managed_ci(runner, config=config, pr_number=7, metadata=metadata())

    assert contract is not None
    assert contract.invocation_applied_label is True
    assert contract.active_label_event_id == 101
    assert any(
        cmd[:5] == ["gh", "api", "--method", "POST", "repos/OWNER/REPO/issues/7/labels"]
        for cmd, _ in runner.commands
    )
    assert release_adopted_managed_ci(runner, config=config, pr_number=7, contract=contract)
    assert any(
        cmd[:5] == ["gh", "api", "--method", "DELETE", f"repos/OWNER/REPO/issues/7/labels/{MANAGED_LABEL}"]
        for cmd, _ in runner.commands
    )


def test_existing_pr_adoption_removes_unprovable_invocation_label(tmp_path):
    config = make_config(
        tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop",
        managed_ci_adopt_existing_pr=True,
    )
    runner = V2ManagedRunner(
        workflow=adoption_workflow(),
        rest_pr={"draft": False, "state": "open", "labels": []},
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT]},
        unreadable_issue_events_after_label=True,
    )

    assert activate_managed_ci(runner, config=config, pr_number=7, metadata=metadata()) is None
    assert any(
        cmd[:5] == ["gh", "api", "--method", "POST", "repos/OWNER/REPO/issues/7/labels"]
        for cmd, _ in runner.commands
    )
    assert any(
        cmd[:5] == ["gh", "api", "--method", "DELETE", f"repos/OWNER/REPO/issues/7/labels/{MANAGED_LABEL}"]
        for cmd, _ in runner.commands
    )


@pytest.mark.parametrize(
    "labels,events,protection",
    [
        ([{"name": MANAGED_OPT_OUT_LABEL}], [], {"contexts": [FINAL_CONTEXT]}),
        ([{"name": MANAGED_LABEL}], [label_event(login="collaborator", actor_id=2)], {"contexts": [FINAL_CONTEXT]}),
        ([], [], {"contexts": []}),
    ],
)
def test_existing_pr_adoption_fails_closed_before_suppression(tmp_path, labels, events, protection):
    config = make_config(
        tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop",
        managed_ci_adopt_existing_pr=True,
    )
    runner = V2ManagedRunner(
        workflow=adoption_workflow(), rest_pr={"draft": False, "labels": labels},
        issue_events=events, pr_branch_protection_payload=protection,
    )

    assert activate_managed_ci(runner, config=config, pr_number=7, metadata=metadata()) is None
    assert not any(
        cmd[:5] == ["gh", "api", "--method", "POST", "repos/OWNER/REPO/issues/7/labels"]
        for cmd, _ in runner.commands
    )


@pytest.mark.parametrize(
    "runner_kwargs",
    [
        {"actor_login": "other"},
        {"advertised_actor": "other"},
    ],
)
def test_v2_preflight_rejects_configured_or_advertised_actor_mismatch(tmp_path, runner_kwargs):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")

    assert preflight_managed_ci_creation(
        V2ManagedRunner(**runner_kwargs), config=config, issue_number=643
    ) is None


def test_v2_preflight_fails_closed_for_incomplete_markers(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = V2ManagedRunner(workflow="# AGENT_LOOP_MANAGED_CI_V2\n")

    with pytest.raises(AgentLoopError, match="incomplete managed-CI v2 contract"):
        preflight_managed_ci_creation(runner, config=config, issue_number=643)


def test_explicit_managed_mode_rejects_legacy_v1_contract(tmp_path):
    config = make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop")

    with pytest.raises(AgentLoopError, match="legacy v1"):
        activate_managed_ci(
            ManagedRunner(workflow=WORKFLOW), config=config, pr_number=7, metadata=metadata()
        )


def test_explicit_activation_release_is_terminal_and_unqualified(tmp_path):
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
    )
    runner = V2ManagedRunner(issue_events=[])

    with pytest.raises(AgentLoopError, match="did NOT qualify"):
        _release_for_ordinary_recovery(
            runner,
            config=config,
            pr_number=7,
            base_ref="main",
            expected_head_sha="abc123",
            active_event=None,
            reason="timeline unavailable",
        )
    assert any(
        command[:5] == [
            "gh", "api", "--method", "DELETE",
            f"repos/OWNER/REPO/issues/7/labels/{MANAGED_LABEL}",
        ]
        for command, _cwd in runner.commands
    )


def test_v2_intent_resumes_matching_comment_and_rejects_competing_nonce(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    comment = v2_intent_comment(run_id=100, run_attempt=1)
    runner = V2ManagedRunner(intent_comments=[comment])
    contract = v2_contract()

    _ensure_v2_intent(
        runner, config=config, pr_number=7, expected_head_sha="abc123", contract=contract
    )

    assert (contract.intent_comment_id, contract.nonce, contract.attached_run_id) == (17, "nonce-1", 100)
    competing = dict(comment)
    competing["id"] = 18
    competing["body"] = v2_intent_comment(nonce="nonce-2")["body"]
    runner = V2ManagedRunner(intent_comments=[comment, competing])
    with pytest.raises(AgentLoopError, match="Competing managed-CI v2 intent"):
        _ensure_v2_intent(
            runner, config=config, pr_number=7, expected_head_sha="abc123", contract=v2_contract()
        )


def test_v2_dispatch_discovers_existing_run_before_dispatching(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = V2ManagedRunner(workflow_runs=[v2_run()], intent_comments=[v2_intent_comment()])
    contract = v2_contract()

    _dispatch_v2_qualification(
        runner, config=config, pr_number=7, expected_head_sha="abc123", contract=contract
    )

    assert contract.attached_run_id == 100
    assert not any("/dispatches" in " ".join(cmd) for cmd, _cwd in runner.commands)


def test_v2_dispatch_discovers_run_with_display_title_and_qualified_path(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = V2ManagedRunner(
        workflow_runs=[
            v2_run(
                name="CI",
                display_title="managed-ci-v2 nonce=nonce-1",
                path="OWNER/REPO/.github/workflows/ci.yml@refs/heads/main",
            )
        ],
        intent_comments=[v2_intent_comment()],
    )
    contract = v2_contract()

    _dispatch_v2_qualification(
        runner, config=config, pr_number=7, expected_head_sha="abc123", contract=contract
    )

    assert contract.attached_run_id == 100
    assert not any("/dispatches" in " ".join(cmd) for cmd, _cwd in runner.commands)


def test_v2_dispatch_rejects_a_stale_approved_head(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = V2ManagedRunner(pr_payload={"headRefOid": "new-head"})

    with pytest.raises(AgentLoopError, match="head moved from approved SHA"):
        dispatch_final_qualification(
            runner,
            config=config,
            pr_number=7,
            expected_head_sha="abc123",
            head_ref="agent-loop/managed-643",
            contract=v2_contract(),
        )


def test_v2_qualification_reports_failed_jobs_from_the_attached_run(tmp_path):
    config = make_config(
        tmp_path, auto_merge=True, ci_timeout_seconds=1, ci_poll_interval_seconds=1
    )
    runner = V2ManagedRunner(
        workflow_runs=[v2_run()],
        jobs=[{"name": "unit", "conclusion": "failure", "html_url": "https://example.test/job/1"}],
        pr_payload={
            "headRefOid": "abc123",
            "mergeable": "MERGEABLE",
            "mergeStateStatus": "CLEAN",
        },
        pr_status_payload={
            "statuses": [
                {
                    "context": FINAL_CONTEXT,
                    "state": "failure",
                    "description": "nonce=nonce-1;run_id=100;attempt=1",
                    "target_url": "https://github.com/OWNER/REPO/actions/runs/100",
                    "creator": {"login": "github-actions[bot]", "id": 41898282},
                }
            ]
        },
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT], "checks": []},
    )

    outcome = wait_for_final_qualification(
        runner,
        config=config,
        pr_number=7,
        metadata=metadata(),
        contract=v2_contract(attached_run_id=100, run_attempt=1),
    )

    assert outcome.status == "failed"
    assert outcome.failure_details == ("unit: failure (https://example.test/job/1)",)


def test_v2_failed_jobs_and_ready_merge_recovery(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = V2ManagedRunner(
        rest_pr={"draft": False},
        jobs=[{"name": "unit", "conclusion": "failure", "html_url": "https://example.test/job/1"}],
    )

    assert _v2_failed_jobs(runner, config=config, run_id=100) == (
        "unit: failure (https://example.test/job/1)",
    )
    prepare_v2_merge(
        runner,
        config=config,
        pr_number=7,
        expected_head_sha="abc123",
        contract=v2_contract(),
    )

    assert not any(cmd[:3] == ["gh", "pr", "ready"] for cmd, _cwd in runner.commands)


def test_paginated_array_response_is_flat_and_malformed_entries_are_unavailable(tmp_path):
    config = make_config(tmp_path)
    runner = V2ManagedRunner(issue_events=[label_event()])

    assert _api_list(runner, config, "repos/OWNER/REPO/issues/7/events?per_page=100") == [label_event()]
    runner.issue_events = [label_event(), ["not an event"]]
    assert _api_list(runner, config, "repos/OWNER/REPO/issues/7/events?per_page=100") is None
    assert all("--slurp" not in command for command, _cwd in runner.commands)


def test_v2_failed_jobs_decodes_concatenated_cli_pages(tmp_path):
    class PagedJobsRunner(FakeRunner):
        def _run_locked(self, args, *, cwd, check):
            if args and str(args[-1]).endswith("/jobs?filter=latest&per_page=100"):
                cmd, cwd_path = self._record_command(args, cwd)
                return CommandResult(
                    cmd, cwd_path,
                    '{"jobs":[{"name":"unit","conclusion":"failure"}]}'
                    '{"jobs":[{"name":"lint","conclusion":"timed_out"}]}',
                    "", 0,
                )
            return super()._run_locked(args, cwd=cwd, check=check)

    details = _v2_failed_jobs(PagedJobsRunner(), config=make_config(tmp_path), run_id=100)
    assert details == ("unit: failure", "lint: timed_out")


def test_managed_ci_gh_invocations_obey_the_245_floor():
    allowed_api_flags = {
        "--paginate", "--method", "-H", "-f", "-F", "--input", "--hostname", "--jq",
        "--silent", "--verbose",
    }
    for filename in ("managed_ci.py", "orchestrator.py"):
        path = Path(__file__).parents[1] / "src" / "coding_review_agent_loop" / filename
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in {"run", "run_with_log"} or not node.args:
                continue
            command = node.args[0]
            if not isinstance(command, ast.List) or not command.elts:
                continue
            values = {elt.value for elt in command.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)}
            if "api" in values:
                api_flags = {value for value in values if value.startswith("-")}
                assert api_flags.issubset(allowed_api_flags)


def test_merge_pr_uses_expected_head_guard(tmp_path):
    config = make_config(tmp_path, auto_merge=True)
    runner = FakeRunner()

    merge_pr(runner, config, 7, expected_head_sha="abc123")

    command = runner.commands[-1][0]
    assert command[-2:] == ["--match-head-commit", "abc123"]
