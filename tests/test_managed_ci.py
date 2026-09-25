import ast
import copy
import json
import re
import shlex
import time
from dataclasses import replace
from datetime import datetime, timezone
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
    AuthenticatedManagedResume,
    FINAL_CONTEXT,
    MANAGED_LABEL,
    MANAGED_OPT_OUT_LABEL,
    QUALIFICATION_MARKER,
    READINESS_CONTEXT,
    ManagedCiContract,
    ManagedCiProbeContext,
    ManagedCiIssueAuthorization,
    UNPROTECTED_OVERRIDE_TRAILER,
    assess_exact_head_protection,
    _dispatch_v2_qualification,
    _ensure_v2_intent,
    _intent_body,
    _patch_intent,
    _v2_failed_jobs,
    _v2_correlated_status,
    _v2_terminal_attempt_excluded,
    _api_list,
    activate_managed_ci,
    authenticate_issue_created_handoff,
    authorize_fresh_issue_created_resume,
    format_issue_created_authorization_comment,
    parse_issue_created_authorization_comment,
    publish_issue_created_continuity_authorization,
    publish_issue_created_authorization,
    dispatch_final_qualification,
    evaluate_managed_ci_readiness,
    intermediate_managed_checks,
    preflight_managed_ci_creation,
    parse_managed_ci_override_record,
    render_managed_ci_resume_command,
    _render_recovery_command,
    recover_issue_created_handoff,
    revalidate_issue_created_handoff,
    prepare_v2_merge,
    publish_manual_v2_qualification,
    publish_round_readiness,
    release_adopted_managed_ci,
    revalidate_adopted_managed_ci,
    OrdinaryRecoveryCapability,
    refresh_ordinary_recovery_capability,
    _release_for_ordinary_recovery,
    _find_resume_audit,
    wait_for_ordinary_recovery,
    wait_for_final_qualification,
)
from coding_review_agent_loop.protocol_markers import (
    PR_BODY_SURFACE,
    PR_COMMENT_SURFACE,
    protocol_record_label,
)
from coding_review_agent_loop.orchestrator import (
    _finalize_ordinary_recovery_merge,
    _render_ci_rerun_command,
    _stop_on_terminal_without_status,
)
from coding_review_agent_loop.protocol import UnresolvedReviewItem
from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata
from coding_review_agent_loop.runner import CommandResult
from coding_review_agent_loop.cli import build_parser
from coding_review_agent_loop.config import resolve_base_branch

from fixtures.managed_ci import current_router, dispatch_validator, historical_router, local_router

from agent_loop_helpers import FakeRunner, make_config, structured_pr_review


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

    def _run_locked(self, args, *, cwd, check, input_text=None):
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
        issue_timeline=None,
        compare_payload=None,
        base_sha="base-sha",
        **kwargs,
    ):
        workflow = kwargs.pop("workflow", V2_WORKFLOW)
        self.base_sha = base_sha
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
        self.issue_timeline = list(issue_timeline or [{
            "event": "cross-referenced",
            "source": {"issue": {
                "number": 7,
                "repository_url": "https://api.github.test/repos/OWNER/REPO",
                "pull_request": {"url": "https://api.github.test/pulls/7"},
            }},
        }])
        self.compare_payload = compare_payload
        self.labels_posted = False
        self.intent_snapshots = []
        self.audit_comments = {}
        self.dispatch_count = 0

    def _next_comment_id(self):
        """Allocate one unused comment identity across every stored comment."""
        known = [
            int(item.get("id", 0))
            for item in self.intent_comments
            if isinstance(item, dict)
        ]
        known.extend(int(key) for key in self.audit_comments)
        return max(known, default=16) + 1

    def _stored_comment(self, comment_id, body):
        """Return the envelope GitHub would report for one stored comment."""
        for comment in self.intent_comments:
            if isinstance(comment, dict) and comment.get("id") == comment_id:
                return comment
        if comment_id in self.audit_comments:
            return self.audit_comments[comment_id]
        if body is None:
            return None
        return {
            "id": comment_id,
            "user": {"login": self.actor_login, "id": self.actor_id},
            "body": body,
        }

    def _capture_intent_body(self, body):
        marker = "AGENT_MANAGED_CI_INTENT_V2"
        if not isinstance(body, str) or marker not in body:
            return None
        try:
            encoded = body.split(marker, 1)[1].split("-->", 1)[0].strip()
            record = json.loads(encoded)
        except (IndexError, json.JSONDecodeError):
            return None
        if not isinstance(record, dict):
            return None
        snapshot = copy.deepcopy(record)
        self.intent_snapshots.append(snapshot)
        return snapshot

    @staticmethod
    def _form_value(cmd, name):
        prefix = f"{name}="
        for part in cmd:
            if isinstance(part, str) and part.startswith(prefix):
                return part[len(prefix):]
        return None

    def _run_locked(self, args, *, cwd, check, input_text=None):
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
        if endpoint.startswith("repos/OWNER/REPO/issues/643/timeline?"):
            cmd, cwd_path = self._record_command(args, cwd)
            return CommandResult(cmd, cwd_path, json.dumps(self.issue_timeline), "", 0)
        if endpoint.startswith("repos/OWNER/REPO/compare/"):
            cmd, cwd_path = self._record_command(args, cwd)
            return CommandResult(
                cmd,
                cwd_path,
                json.dumps(self.compare_payload or {}),
                "",
                0,
            )
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
            return CommandResult(cmd, cwd_path, json.dumps({"sha": self.base_sha}), "", 0)
        if endpoint.startswith("repos/OWNER/REPO/issues/7/comments?"):
            cmd, cwd_path = self._record_command(args, cwd)
            return CommandResult(cmd, cwd_path, json.dumps(self.intent_comments), "", 0)
        if endpoint == "repos/OWNER/REPO/issues/7/comments" and "POST" in cmd:
            cmd, cwd_path = self._record_command(args, cwd)
            body = self._form_value(cmd, "body")
            record = self._capture_intent_body(body)
            comment_id = self._next_comment_id()
            if record is not None:
                self.intent_comments.append({
                    "id": comment_id,
                    "user": {"login": self.actor_login, "id": self.actor_id},
                    "body": body,
                })
            else:
                self.audit_comments[comment_id] = {
                    "id": comment_id,
                    "user": {"login": self.actor_login, "id": self.actor_id},
                    "body": body,
                }
            # GitHub echoes the stored comment envelope on a successful write.
            return CommandResult(
                cmd,
                cwd_path,
                json.dumps(self._stored_comment(comment_id, body)),
                "",
                0,
            )
        if endpoint.startswith("repos/OWNER/REPO/issues/comments/"):
            cmd, cwd_path = self._record_command(args, cwd)
            comment_id = int(endpoint.rsplit("/", 1)[-1])
            body = self._form_value(cmd, "body")
            if body is None:
                stored = self._stored_comment(comment_id, None)
                if stored is None:
                    return CommandResult(cmd, cwd_path, "", "HTTP 404: Not Found", 1)
                return CommandResult(cmd, cwd_path, json.dumps(stored), "", 0)
            record = self._capture_intent_body(body)
            if record is not None:
                for comment in self.intent_comments:
                    if isinstance(comment, dict) and comment.get("id") == comment_id:
                        comment["body"] = body
                        break
            elif comment_id in self.audit_comments:
                self.audit_comments[comment_id]["body"] = body
            return CommandResult(
                cmd,
                cwd_path,
                json.dumps(self._stored_comment(comment_id, body)),
                "",
                0,
            )
        if endpoint.endswith("/actions/workflows/ci.yml/dispatches"):
            self.dispatch_count += 1
            return super()._run_locked(args, cwd=cwd, check=check)
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

    def _run_locked(self, args, *, cwd, check, input_text=None):
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
        body=f"Fixes #643\n\n{UNPROTECTED_OVERRIDE_TRAILER} nonce=nonce-643",
    )


class AuthorizationCommentRunner(V2ManagedRunner):
    """Persist the new authorization record like the GitHub comment API."""

    def __init__(self, **kwargs):
        rest_pr = dict(kwargs.pop("rest_pr", {}) or {})
        rest_pr.setdefault("state", "open")
        rest_pr.setdefault(
            "body", f"Fixes #643\n\n{UNPROTECTED_OVERRIDE_TRAILER} nonce=nonce-643"
        )
        super().__init__(rest_pr=rest_pr, **kwargs)

    def _run_locked(self, args, *, cwd, check, input_text=None):
        endpoint = next(
            (part for part in args if isinstance(part, str) and part.startswith("repos/")), ""
        )
        if endpoint == "repos/OWNER/REPO/issues/7/comments" and "POST" in args:
            body = self._form_value(args, "body") or ""
            if "AGENT_MANAGED_CI_ISSUE_AUTHORIZATION_V1" in body:
                comment_id = max(
                    (int(item.get("id", 0)) for item in self.intent_comments if isinstance(item, dict)),
                    default=16,
                ) + 1
                self.intent_comments.append({
                    "id": comment_id,
                    "user": {"login": self.actor_login, "id": self.actor_id},
                    "body": body,
                })
                cmd, cwd_path = self._record_command(args, cwd)
                return CommandResult(
                    cmd,
                    cwd_path,
                    json.dumps({
                        "id": comment_id,
                        "body": body,
                        "user": {"login": self.actor_login, "id": self.actor_id},
                    }),
                    "",
                    0,
                )
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


class IdlessAuthorizationCommentRunner(AuthorizationCommentRunner):
    def _run_locked(self, args, *, cwd, check, input_text=None):
        endpoint = next(
            (part for part in args if isinstance(part, str) and part.startswith("repos/")), ""
        )
        if endpoint == "repos/OWNER/REPO/issues/7/comments" and "POST" in args:
            body = self._form_value(list(args), "body") or ""
            cmd, cwd_path = self._record_command(args, cwd)
            return CommandResult(
                cmd, cwd_path,
                json.dumps({"body": body, "user": {"login": self.actor_login, "id": self.actor_id}}),
                "", 0,
            )
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


class FailingAuthorizationCommentRunner(AuthorizationCommentRunner):
    def _run_locked(self, args, *, cwd, check, input_text=None):
        endpoint = next(
            (part for part in args if isinstance(part, str) and part.startswith("repos/")), ""
        )
        if endpoint == "repos/OWNER/REPO/issues/7/comments" and "POST" in args:
            cmd, cwd_path = self._record_command(args, cwd)
            return CommandResult(cmd, cwd_path, "", "comment API unavailable", 1)
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


class RacedAuthorizationCommentRunner(AuthorizationCommentRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pull_reads = 0

    def _run_locked(self, args, *, cwd, check, input_text=None):
        endpoint = next(
            (part for part in args if isinstance(part, str) and part.startswith("repos/")), ""
        )
        if endpoint == "repos/OWNER/REPO/pulls/7":
            self.pull_reads += 1
            if self.pull_reads == 2:
                self.rest_pr["head"]["sha"] = "raced-head"
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


class ContinuityAuthorizationSetRaceRunner(AuthorizationCommentRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.authorization_comment_reads = 0

    def _run_locked(self, args, *, cwd, check, input_text=None):
        endpoint = next(
            (part for part in args if isinstance(part, str) and part.startswith("repos/")), ""
        )
        if endpoint == "repos/OWNER/REPO/issues/7/comments?per_page=100":
            self.authorization_comment_reads += 1
            if self.authorization_comment_reads == 4:
                root = next(
                    item
                    for item in self.intent_comments
                    if "AGENT_MANAGED_CI_ISSUE_AUTHORIZATION_V1" in item.get("body", "")
                )
                self.intent_comments.append(dict(root, id=int(root["id"]) + 100))
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


class _ActivationReached(Exception):
    """Stop an orchestrator regression immediately after real activation."""


class _ReviewReached(Exception):
    """Stop after the first real reviewer publication."""


def _stop_after_real_activation(monkeypatch, captured_resume=None):
    monkeypatch.setattr(
        orchestrator,
        "_freeze_prompt_architecture",
        lambda _runner, config, **_kwargs: config,
    )
    real_activate_managed_ci = orchestrator.activate_managed_ci

    def activate_then_stop(*args, **kwargs):
        if captured_resume is not None:
            captured_resume["managed_resume"] = kwargs.get("managed_resume")
        result = real_activate_managed_ci(*args, **kwargs)
        raise _ActivationReached(result)

    monkeypatch.setattr(
        orchestrator,
        "activate_managed_ci",
        activate_then_stop,
    )


def _stop_after_real_reviewer(monkeypatch, captured=None):
    monkeypatch.setattr(
        orchestrator,
        "_freeze_prompt_architecture",
        lambda _runner, config, **_kwargs: config,
    )
    real_activate_managed_ci = orchestrator.activate_managed_ci
    real_post_pr_comment = orchestrator.post_pr_comment

    def activate_and_capture(*args, **kwargs):
        result = real_activate_managed_ci(*args, **kwargs)
        if captured is not None:
            captured["activation"] = result
            captured["managed_resume"] = kwargs.get("managed_resume")
        return result

    def post_and_stop(*args, **kwargs):
        result = real_post_pr_comment(*args, **kwargs)
        if str(kwargs.get("body") or "").startswith("**Review verdict:"):
            raise _ReviewReached
        return result

    monkeypatch.setattr(orchestrator, "activate_managed_ci", activate_and_capture)
    monkeypatch.setattr(orchestrator, "post_pr_comment", post_and_stop)


def _workflow_runner_for_issue_authorization(
    record: ManagedCiIssueAuthorization | None,
    *,
    labeled: bool,
) -> AuthorizationCommentRunner:
    body = f"Fixes #643\n\n{UNPROTECTED_OVERRIDE_TRAILER} nonce=opening-nonce"
    comments = [] if record is None else [{
        "id": 41,
        "user": {"login": "agent-loop", "id": 1},
        "body": str(format_issue_created_authorization_comment(record)),
    }]
    return AuthorizationCommentRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        issue_payload={"number": 643},
        pr_payload={
            "number": 7,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/7",
            "title": "Managed CI",
            "body": body,
            "headRefName": "agent-loop/managed-643",
            "baseRefName": "main",
            "headRefOid": "abc123",
            "comments": [],
            "reviews": [],
        },
        rest_pr={
            "state": "open",
            "draft": True,
            "labels": [{"name": MANAGED_LABEL}] if labeled else [],
            "body": body,
        },
        issue_events=[label_event()],
        intent_comments=comments,
    )


def test_run_pr_loop_ordinary_resume_uses_real_durable_recovery_and_activation(
    tmp_path, monkeypatch,
):
    record = ManagedCiIssueAuthorization(
        kind="creation", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="abc123", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci",
        nonce="opening-nonce", label_event_id=101,
    )
    runner = _workflow_runner_for_issue_authorization(record, labeled=True)
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci",
            "--managed-ci-trusted-actor", "agent-loop",
            "--allow-unprotected-managed-ci",
        ),
    )
    _stop_after_real_activation(monkeypatch)

    with pytest.raises(_ActivationReached):
        orchestrator.run_pr_loop(
            runner, pr_number=7, config=config, workdirs_ready=True,
        )

    commands = [command for command, _cwd in runner.commands]
    assert any(
        UNPROTECTED_OVERRIDE_TRAILER in " ".join(command)
        and "issues/7/comments" in " ".join(command)
        for command in commands
    )
    assert not any(
        command[:5] == [
            "gh", "api", "--method", "POST", "repos/OWNER/REPO/issues/7/labels"
        ]
        for command in commands
    )
    assert runner.dispatch_count == 0
    assert runner.comments == []


def test_run_pr_loop_fresh_recovery_uses_real_authorization_and_activation(
    tmp_path, monkeypatch,
):
    runner = _workflow_runner_for_issue_authorization(None, labeled=False)
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_fresh_authorization=True, managed_ci_issue_number=643,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci", "--managed-ci-fresh",
            "--managed-ci-issue", "643", "--managed-ci-trusted-actor", "agent-loop",
            "--allow-unprotected-managed-ci",
        ),
    )
    _stop_after_real_activation(monkeypatch)

    with pytest.raises(_ActivationReached):
        orchestrator.run_pr_loop(
            runner, pr_number=7, config=config, workdirs_ready=True,
        )

    records = [
        record
        for comment in runner.intent_comments
        if (record := parse_issue_created_authorization_comment(comment["body"]))
        is not None
    ]
    assert len(records) == 1
    assert records[0] is not None and records[0].kind == "fresh"
    assert runner.labels_posted is True
    assert runner.dispatch_count == 0
    assert runner.comments == []


def test_run_pr_loop_fresh_retry_reuses_continuity_terminal_without_competing_grant(
    tmp_path, monkeypatch,
):
    runner = _workflow_runner_for_issue_authorization(
        None,
        labeled=True,
    )
    runner.codex_outputs = [
        structured_pr_review(
            state="approved",
            summary="Reviewed the continuity-authorized exact head.",
            reviewer="OpenAI Codex",
        )
    ]
    root = publish_issue_created_authorization(
        runner, config=make_config(
            tmp_path, managed_ci=True, managed_ci_pr_mode=True,
            managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        ),
        handoff=replace(
            _authorization_handoff(),
            override_nonce="opening-nonce",
            opening_override_nonce="opening-nonce",
        ),
        metadata=replace(metadata(), body=runner.rest_pr["body"]),
    )
    runner.intent_comments.extend([
        _round_comment(88, role="reviewer", subject="abc123", round_number=1, state="blocking"),
        _round_comment(89, role="coder", subject="next-head", round_number=2),
    ])
    runner.rest_pr["head"]["sha"] = "next-head"
    runner.pr_payload["headRefOid"] = "next-head"
    continuity = publish_issue_created_continuity_authorization(
        runner,
        config=make_config(
            tmp_path, managed_ci=True, managed_ci_pr_mode=True,
            managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        ),
        handoff=root,
        predecessor_head="abc123",
        new_head="next-head",
        round_comment_ids=(88, 89),
    )
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_fresh_authorization=True, managed_ci_issue_number=643,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci", "--managed-ci-fresh",
            "--managed-ci-issue", "643", "--managed-ci-trusted-actor", "agent-loop",
            "--allow-unprotected-managed-ci",
        ),
    )
    captured = {}
    _stop_after_real_reviewer(monkeypatch, captured)

    with pytest.raises(_ReviewReached):
        orchestrator.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)

    records = [
        (comment["id"], record)
        for comment in runner.intent_comments
        if (record := parse_issue_created_authorization_comment(comment["body"]))
        is not None
    ]
    assert [record.kind for _comment_id, record in records] == ["creation", "continuity"]
    assert root.authorization_comment_id == records[0][0]
    assert continuity.authorization_comment_id == records[1][0]
    resumed = captured["managed_resume"]
    assert resumed is not None
    assert resumed.issue_created_handoff is not None
    assert resumed.issue_created_handoff.authorization_kind == "continuity"
    assert resumed.issue_created_handoff.override_nonce == records[1][1].nonce
    assert resumed.issue_created_handoff.override_nonce != records[0][1].nonce
    assert resumed.issue_created_handoff.opening_override_nonce == "opening-nonce"
    assert captured["activation"] is not None
    reviewer_command = next(
        command for command, _cwd in runner.commands if command[:2] == ["codex", "exec"]
    )
    assert "next-head" in " ".join(reviewer_command)
    assert any(
        "Reviewed the continuity-authorized exact head." in comment
        for comment in runner.comments
    )
    assert sum(
        "AGENT_MANAGED_CI_ISSUE_AUTHORIZATION_V1" in " ".join(command)
        and "POST" in command
        for command, _cwd in runner.commands
    ) == 2
    assert runner.labels_posted is False
    assert runner.dispatch_count == 0


def _authorization_handoff(*, head="abc123"):
    return managed_ci.AuthenticatedIssueCreatedHandoff(
        pr_number=7,
        issue_number=643,
        repository="OWNER/REPO",
        base_ref="main",
        head_sha=head,
        branch="agent-loop/managed-643",
        trusted_actor_login="agent-loop",
        trusted_actor_id=1,
        protection_mode="voluntary",
        override_nonce="nonce-643",
    )


def _round_comment(comment_id, *, role, subject, round_number, state=None):
    return {
        "id": comment_id,
        "user": {"login": "agent-loop", "id": 1},
        "body": _attach_round_metadata(
            f"{role} round",
            PostedRoundMetadata(
                flow="pr", role=role, agent="agent-loop",
                round_number=round_number, subject=subject, state=state,
            ),
        ),
    }


def test_issue_authorization_round_trip_is_comment_only_and_idempotent(tmp_path):
    authorization = ManagedCiIssueAuthorization(
        kind="creation",
        repository="OWNER/REPO",
        issue_number=643,
        pr_number=7,
        base_ref="main",
        head_sha="abc123",
        actor_login="agent-loop",
        actor_id=1,
        protection="voluntary",
        waiver="allow-unprotected-managed-ci",
        nonce="nonce-643",
        label_event_id=101,
    )
    body = format_issue_created_authorization_comment(authorization)
    assert parse_issue_created_authorization_comment(str(body)) == authorization
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    handoff = _authorization_handoff()
    first = publish_issue_created_authorization(
        runner, config=config, handoff=handoff, metadata=metadata()
    )
    second = publish_issue_created_authorization(
        runner, config=config, handoff=handoff, metadata=metadata()
    )
    assert first.authorization_comment_id == second.authorization_comment_id == 17
    assert sum(
        1 for command, _cwd in runner.commands
        if "issues/7/comments" in " ".join(command) and "POST" in command
    ) == 1


def test_creation_authorization_publication_requires_verified_comment_id(tmp_path):
    runner = IdlessAuthorizationCommentRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    with pytest.raises(AgentLoopError, match="returned no comment ID"):
        publish_issue_created_authorization(
            runner, config=config, handoff=_authorization_handoff(), metadata=metadata()
        )
    assert runner.intent_comments == []


def test_creation_authorization_publication_post_failure_is_not_durable(tmp_path):
    runner = FailingAuthorizationCommentRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    with pytest.raises(AgentLoopError, match="Unable to persist.*comment API unavailable"):
        publish_issue_created_authorization(
            runner, config=config, handoff=_authorization_handoff(), metadata=metadata()
        )
    assert runner.intent_comments == []


def test_creation_authorization_race_writes_no_record(tmp_path):
    runner = RacedAuthorizationCommentRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    with pytest.raises(AgentLoopError, match="opening tuple"):
        publish_issue_created_authorization(
            runner, config=config, handoff=_authorization_handoff(), metadata=metadata()
        )
    assert runner.intent_comments == []


def test_issue_authorization_continuity_accepts_one_gap_free_head_chain(tmp_path):
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    initial = publish_issue_created_authorization(
        runner, config=config, handoff=_authorization_handoff(), metadata=metadata()
    )
    runner.intent_comments.extend([
        _round_comment(88, role="reviewer", subject="abc123", round_number=1, state="blocking"),
        _round_comment(89, role="coder", subject="next-head", round_number=2),
    ])
    runner.rest_pr["head"]["sha"] = "next-head"
    continued = publish_issue_created_continuity_authorization(
        runner,
        config=config,
        handoff=initial,
        predecessor_head="abc123",
        new_head="next-head",
        round_comment_ids=(88, 89),
    )
    audit = _find_resume_audit(
        runner,
        config=config,
        pr_number=7,
        actor_login="agent-loop",
        actor_id=1,
        base_ref="main",
        issue_number=643,
        live_head="next-head",
    )
    assert continued.authorization_kind == "continuity"
    assert audit is not None
    assert audit[0] == continued.authorization_comment_id
    assert audit[1]["head"] == "next-head"


def test_continuity_publication_rechecks_live_tuple_before_writing(tmp_path):
    root = ManagedCiIssueAuthorization(
        kind="creation", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="abc123", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci",
        nonce="nonce-643", label_event_id=101,
    )
    runner = RacedAuthorizationCommentRunner(
        issue_events=[label_event()],
        intent_comments=[
            {"id": 41, "user": {"login": "agent-loop", "id": 1},
             "body": str(format_issue_created_authorization_comment(root))},
            _round_comment(88, role="reviewer", subject="abc123", round_number=1, state="blocking"),
            _round_comment(89, role="coder", subject="next-head", round_number=2),
        ],
    )
    runner.rest_pr["head"]["sha"] = "next-head"
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    with pytest.raises(AgentLoopError, match="changed live PR tuple"):
        publish_issue_created_continuity_authorization(
            runner,
            config=config,
            handoff=_authorization_handoff(),
            predecessor_head="abc123",
            new_head="next-head",
            round_comment_ids=(88, 89),
        )

    assert not any(
        "AGENT_MANAGED_CI_ISSUE_AUTHORIZATION_V1" in " ".join(command)
        and "POST" in command
        for command, _cwd in runner.commands
    )


def test_continuity_publication_rechecks_authorization_set_before_writing(tmp_path):
    root = ManagedCiIssueAuthorization(
        kind="creation", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="abc123", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci",
        nonce="nonce-643", label_event_id=101,
    )
    runner = ContinuityAuthorizationSetRaceRunner(
        issue_events=[label_event()],
        intent_comments=[
            {"id": 41, "user": {"login": "agent-loop", "id": 1},
             "body": str(format_issue_created_authorization_comment(root))},
            _round_comment(88, role="reviewer", subject="abc123", round_number=1, state="blocking"),
            _round_comment(89, role="coder", subject="next-head", round_number=2),
        ],
    )
    runner.rest_pr["head"]["sha"] = "next-head"
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    with pytest.raises(AgentLoopError, match="authorization records changed"):
        publish_issue_created_continuity_authorization(
            runner,
            config=config,
            handoff=_authorization_handoff(),
            predecessor_head="abc123",
            new_head="next-head",
            round_comment_ids=(88, 89),
        )

    assert not any(
        "AGENT_MANAGED_CI_ISSUE_AUTHORIZATION_V1" in " ".join(command)
        and "POST" in command
        for command, _cwd in runner.commands
    )


def test_continuity_publication_rejects_distinct_predecessor_authorizations(tmp_path):
    root = ManagedCiIssueAuthorization(
        kind="creation", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="abc123", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci",
        nonce="root", label_event_id=101,
    )
    competing = replace(root, kind="fresh", nonce="competing")
    runner = AuthorizationCommentRunner(
        issue_events=[label_event()],
        intent_comments=[
            {"id": 41, "user": {"login": "agent-loop", "id": 1},
             "body": str(format_issue_created_authorization_comment(root))},
            {"id": 42, "user": {"login": "agent-loop", "id": 1},
             "body": str(format_issue_created_authorization_comment(competing))},
            _round_comment(88, role="reviewer", subject="abc123", round_number=1, state="blocking"),
            _round_comment(89, role="coder", subject="next-head", round_number=2),
        ],
    )
    runner.rest_pr["head"]["sha"] = "next-head"
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    with pytest.raises(AgentLoopError, match="conflicting prior authorizations"):
        publish_issue_created_continuity_authorization(
            runner,
            config=config,
            handoff=_authorization_handoff(),
            predecessor_head="abc123",
            new_head="next-head",
            round_comment_ids=(88, 89),
        )

    assert not any(
        "AGENT_MANAGED_CI_ISSUE_AUTHORIZATION_V1" in " ".join(command)
        and "POST" in command
        for command, _cwd in runner.commands
    )


def test_continuity_publication_rejects_different_head_fork(tmp_path):
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    initial = publish_issue_created_authorization(
        runner, config=config, handoff=_authorization_handoff(), metadata=metadata()
    )
    runner.intent_comments.extend([
        _round_comment(88, role="reviewer", subject="abc123", round_number=1, state="blocking"),
        _round_comment(89, role="coder", subject="head-a", round_number=2),
    ])
    runner.rest_pr["head"]["sha"] = "head-a"
    publish_issue_created_continuity_authorization(
        runner, config=config, handoff=initial, predecessor_head="abc123",
        new_head="head-a", round_comment_ids=(88, 89),
    )

    runner.intent_comments.extend([
        _round_comment(90, role="reviewer", subject="abc123", round_number=1, state="blocking"),
        _round_comment(91, role="coder", subject="head-b", round_number=2),
    ])
    runner.rest_pr["head"]["sha"] = "head-b"
    with pytest.raises(AgentLoopError, match="forked predecessor"):
        publish_issue_created_continuity_authorization(
            runner, config=config, handoff=initial, predecessor_head="abc123",
            new_head="head-b", round_comment_ids=(90, 91),
        )

    assert not any(
        "AGENT_MANAGED_CI_ISSUE_AUTHORIZATION_V1" in " ".join(command)
        and "POST" in command
        and "head-b" in " ".join(command)
        for command, _cwd in runner.commands
    )


def test_continuity_fork_through_duplicate_predecessor_aliases_fails_closed(tmp_path):
    root = ManagedCiIssueAuthorization(
        kind="creation", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="abc123", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="root",
        label_event_id=101,
    )
    first = ManagedCiIssueAuthorization(
        kind="continuity", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="head-a", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="first",
        label_event_id=101, predecessor_head="abc123", predecessor_comment_id=41,
        round_comment_ids=(88, 89),
    )
    second = replace(
        first, head_sha="head-b", nonce="second", predecessor_comment_id=42,
        round_comment_ids=(90, 91),
    )
    comments = [
        {"id": 41, "user": {"login": "agent-loop", "id": 1},
         "body": str(format_issue_created_authorization_comment(root))},
        {"id": 42, "user": {"login": "agent-loop", "id": 1},
         "body": str(format_issue_created_authorization_comment(root))},
        _round_comment(88, role="reviewer", subject="abc123", round_number=1, state="blocking"),
        _round_comment(89, role="coder", subject="head-a", round_number=2),
        {"id": 100, "user": {"login": "agent-loop", "id": 1},
         "body": str(format_issue_created_authorization_comment(first))},
        _round_comment(90, role="reviewer", subject="abc123", round_number=1, state="blocking"),
        _round_comment(91, role="coder", subject="head-b", round_number=2),
        {"id": 101, "user": {"login": "agent-loop", "id": 1},
         "body": str(format_issue_created_authorization_comment(second))},
    ]
    runner = AuthorizationCommentRunner(
        issue_events=[label_event()], intent_comments=comments,
    )
    runner.rest_pr["head"]["sha"] = "head-b"
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )

    with pytest.raises(AgentLoopError, match="forked predecessor"):
        publish_issue_created_continuity_authorization(
            runner, config=config, handoff=_authorization_handoff(),
            predecessor_head="abc123", new_head="head-b", round_comment_ids=(90, 91),
        )

    assert not any(
        "AGENT_MANAGED_CI_ISSUE_AUTHORIZATION_V1" in " ".join(command)
        and "POST" in command
        for command, _cwd in runner.commands
    )
    for live_head in ("head-a", "head-b"):
        assert _find_resume_audit(
            runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
            base_ref="main", issue_number=643, live_head=live_head,
        ) is None


def test_continuity_publication_rejects_missing_correlated_round_metadata(tmp_path):
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    initial = publish_issue_created_authorization(
        runner, config=config, handoff=_authorization_handoff(), metadata=metadata()
    )
    runner.rest_pr["head"]["sha"] = "next-head"
    with pytest.raises(AgentLoopError, match="correlated round metadata"):
        publish_issue_created_continuity_authorization(
            runner, config=config, handoff=initial, predecessor_head="abc123",
            new_head="next-head", round_comment_ids=(initial.authorization_comment_id,),
        )


def _conflict_round_comment(comment_id, *, subject, round_number):
    """A coder record carrying the tool-owned merge-conflict obligation."""
    conflict = UnresolvedReviewItem(
        item_id="item-merge-conflict",
        reviewer="agent-loop",
        source_round=round_number,
        text="PR has a merge conflict with main.",
        status="blocking",
        authority="machine",
        obligation_kind="merge-conflict",
        lifecycle="repair_required",
    )
    return {
        "id": comment_id,
        "user": {"login": "agent-loop", "id": 1},
        "body": _attach_round_metadata(
            "coder round",
            PostedRoundMetadata(
                flow="pr", role="coder", agent="agent-loop",
                round_number=round_number, subject=subject,
                prior_items=(conflict,),
            ),
        ),
    }


def test_conflict_resolution_round_grants_continuity_without_a_reviewer_pair(tmp_path):
    """#829: the orchestrator skips reviewers for a conflict round by design."""
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    runner.intent_comments.append(
        _conflict_round_comment(61, subject="merged-head", round_number=13)
    )
    config = make_config(tmp_path)

    selected = managed_ci.find_actor_round_metadata_comment_ids(
        runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
        predecessor_head="abc123", new_head="merged-head", round_number=12,
        after_comment_id=50,
    )

    assert selected == (61,)


def test_head_advance_without_review_or_conflict_obligation_still_fails(tmp_path):
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    runner.intent_comments.append(
        _round_comment(62, role="coder", subject="merged-head", round_number=13)
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="correlated blocking-review and coder"):
        managed_ci.find_actor_round_metadata_comment_ids(
            runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
            predecessor_head="abc123", new_head="merged-head", round_number=12,
            after_comment_id=50,
        )


def _publish_conflict_round_continuity(runner, *, config):
    """Publish a continuity grant whose only round record is the conflict coder."""
    initial = publish_issue_created_authorization(
        runner, config=config, handoff=_authorization_handoff(), metadata=metadata()
    )
    runner.intent_comments.append(
        _conflict_round_comment(61, subject="merged-head", round_number=13)
    )
    runner.rest_pr["head"]["sha"] = "merged-head"
    continued = publish_issue_created_continuity_authorization(
        runner,
        config=config,
        handoff=initial,
        predecessor_head="abc123",
        new_head="merged-head",
        round_comment_ids=(61,),
    )
    return continued


def test_conflict_round_continuity_grant_reauthenticates_on_resume(tmp_path):
    """#829: the persisted grant must survive resume with no reviewer record."""
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )

    continued = _publish_conflict_round_continuity(runner, config=config)
    audit = _find_resume_audit(
        runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
        base_ref="main", issue_number=643, live_head="merged-head",
    )

    assert continued.authorization_kind == "continuity"
    assert audit is not None
    assert audit[0] == continued.authorization_comment_id
    assert audit[1]["head"] == "merged-head"


def test_resume_rejects_a_conflict_grant_once_the_obligation_is_gone(tmp_path):
    """The obligation is the whole authority; without it the chain breaks."""
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    _publish_conflict_round_continuity(runner, config=config)

    # Same record identity and ordering, ordinary coder round metadata only.
    runner.intent_comments = [
        _round_comment(61, role="coder", subject="merged-head", round_number=13)
        if comment.get("id") == 61 else comment
        for comment in runner.intent_comments
    ]
    audit = _find_resume_audit(
        runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
        base_ref="main", issue_number=643, live_head="merged-head",
    )

    assert audit is None


def test_continuity_publication_rejects_a_lone_coder_record_without_the_obligation(tmp_path):
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    initial = publish_issue_created_authorization(
        runner, config=config, handoff=_authorization_handoff(), metadata=metadata()
    )
    runner.intent_comments.append(
        _round_comment(61, role="coder", subject="merged-head", round_number=13)
    )
    runner.rest_pr["head"]["sha"] = "merged-head"

    with pytest.raises(AgentLoopError, match="authenticated, correlated round metadata"):
        publish_issue_created_continuity_authorization(
            runner,
            config=config,
            handoff=initial,
            predecessor_head="abc123",
            new_head="merged-head",
            round_comment_ids=(61,),
        )


def test_round_metadata_selection_requires_current_ordered_transition(tmp_path):
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    runner.intent_comments.extend([
        _round_comment(18, role="reviewer", subject="abc123", round_number=1, state="blocking"),
        _round_comment(19, role="coder", subject="next-head", round_number=2),
        _round_comment(51, role="coder", subject="next-head", round_number=2),
        _round_comment(52, role="reviewer", subject="abc123", round_number=1, state="blocking"),
    ])
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="correlated blocking-review and coder"):
        managed_ci.find_actor_round_metadata_comment_ids(
            runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
            predecessor_head="abc123", new_head="next-head", round_number=1,
            after_comment_id=50,
        )


def test_continuity_resume_rejects_round_metadata_older_than_predecessor(tmp_path):
    root = ManagedCiIssueAuthorization(
        kind="creation", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="abc123", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="root",
        label_event_id=101,
    )
    continuity = ManagedCiIssueAuthorization(
        kind="continuity", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="next-head", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="next",
        label_event_id=101, predecessor_head="abc123", predecessor_comment_id=50,
        round_comment_ids=(48, 49),
    )
    comments = [
        _round_comment(48, role="reviewer", subject="abc123", round_number=1, state="blocking"),
        _round_comment(49, role="coder", subject="next-head", round_number=2),
        {"id": 50, "user": {"login": "agent-loop", "id": 1},
         "body": str(format_issue_created_authorization_comment(root))},
        {"id": 51, "user": {"login": "agent-loop", "id": 1},
         "body": str(format_issue_created_authorization_comment(continuity))},
    ]
    runner = V2ManagedRunner(intent_comments=comments)
    config = make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop")

    assert _find_resume_audit(
        runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
        base_ref="main", issue_number=643, live_head="next-head",
    ) is None


@pytest.mark.parametrize("mutation", ["unrelated", "malformed", "gapped"])
def test_continuity_resume_rejects_uncorrelated_round_metadata(tmp_path, mutation):
    root = ManagedCiIssueAuthorization(
        kind="creation", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="abc123", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="root",
        label_event_id=101,
    )
    continuation = ManagedCiIssueAuthorization(
        kind="continuity", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="next-head", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="next",
        label_event_id=101, predecessor_head="abc123", predecessor_comment_id=41,
        round_comment_ids=(42, 43),
    )
    reviewer = _round_comment(
        42, role="reviewer", subject="abc123", round_number=1, state="blocking"
    )
    coder = _round_comment(43, role="coder", subject="next-head", round_number=2)
    if mutation == "unrelated":
        reviewer = _round_comment(
            42, role="reviewer", subject="different-head", round_number=1,
            state="blocking",
        )
    elif mutation == "malformed":
        reviewer["body"] = "<!-- AGENT_ROUND_RESUME: !!! -->"
    else:
        reviewer = _round_comment(
            42, role="reviewer", subject="abc123", round_number=7, state="blocking"
        )
    comments = [
        {"id": 41, "user": {"login": "agent-loop", "id": 1},
         "body": str(format_issue_created_authorization_comment(root))},
        reviewer,
        coder,
        {"id": 44, "user": {"login": "agent-loop", "id": 1},
         "body": str(format_issue_created_authorization_comment(continuation))},
    ]
    runner = V2ManagedRunner(intent_comments=comments)
    config = make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop")

    assert _find_resume_audit(
        runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
        base_ref="main", issue_number=643, live_head="next-head",
    ) is None


def test_continuity_resume_rejects_two_valid_metadata_forks(tmp_path):
    root = ManagedCiIssueAuthorization(
        kind="creation", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="abc123", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="root",
        label_event_id=101,
    )
    first = ManagedCiIssueAuthorization(
        kind="continuity", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="next-head", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="first",
        label_event_id=101, predecessor_head="abc123", predecessor_comment_id=41,
        round_comment_ids=(42, 43),
    )
    second = replace(first, nonce="second", round_comment_ids=(44, 45))
    comments = [
        {"id": 41, "user": {"login": "agent-loop", "id": 1},
         "body": str(format_issue_created_authorization_comment(root))},
        _round_comment(42, role="reviewer", subject="abc123", round_number=1, state="blocking"),
        _round_comment(43, role="coder", subject="next-head", round_number=2),
        _round_comment(44, role="reviewer", subject="abc123", round_number=3, state="blocking"),
        _round_comment(45, role="coder", subject="next-head", round_number=4),
        {"id": 46, "user": {"login": "agent-loop", "id": 1},
         "body": str(format_issue_created_authorization_comment(first))},
        {"id": 47, "user": {"login": "agent-loop", "id": 1},
         "body": str(format_issue_created_authorization_comment(second))},
    ]
    runner = V2ManagedRunner(intent_comments=comments)
    config = make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop")

    assert _find_resume_audit(
        runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
        base_ref="main", issue_number=643, live_head="next-head",
    ) is None


@pytest.mark.parametrize("live_head", ["head-a", "head-b"])
def test_continuity_resume_rejects_different_head_fork(tmp_path, live_head):
    root = ManagedCiIssueAuthorization(
        kind="creation", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="abc123", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="root",
        label_event_id=101,
    )
    first = ManagedCiIssueAuthorization(
        kind="continuity", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="head-a", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="first",
        label_event_id=101, predecessor_head="abc123", predecessor_comment_id=41,
        round_comment_ids=(42, 43),
    )
    second = replace(
        first, head_sha="head-b", nonce="second", round_comment_ids=(44, 45)
    )
    comments = [
        {"id": 41, "user": {"login": "agent-loop", "id": 1},
         "body": str(format_issue_created_authorization_comment(root))},
        _round_comment(42, role="reviewer", subject="abc123", round_number=1, state="blocking"),
        _round_comment(43, role="coder", subject="head-a", round_number=2),
        _round_comment(44, role="reviewer", subject="abc123", round_number=1, state="blocking"),
        _round_comment(45, role="coder", subject="head-b", round_number=2),
        {"id": 46, "user": {"login": "agent-loop", "id": 1},
         "body": str(format_issue_created_authorization_comment(first))},
        {"id": 47, "user": {"login": "agent-loop", "id": 1},
         "body": str(format_issue_created_authorization_comment(second))},
    ]
    runner = V2ManagedRunner(intent_comments=comments)
    config = make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop")

    assert _find_resume_audit(
        runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
        base_ref="main", issue_number=643, live_head=live_head,
    ) is None


def test_fresh_issue_authorization_requires_explicit_scope_and_is_idempotent(tmp_path):
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    fresh_metadata = replace(
        metadata(),
        head_branch="agent-loop/managed-643",
        body="Fixes #643",
    )
    first = authorize_fresh_issue_created_resume(
        runner,
        config=config,
        pr_number=7,
        issue_number=643,
        metadata=fresh_metadata,
        approved_plan_hash="a" * 64,
    )
    second = authorize_fresh_issue_created_resume(
        runner,
        config=config,
        pr_number=7,
        issue_number=643,
        metadata=fresh_metadata,
        approved_plan_hash="a" * 64,
    )
    assert first.authorization_kind == second.authorization_kind == "fresh"
    assert first.authorization_comment_id == second.authorization_comment_id == 17
    assert first.approved_plan_hash == "a" * 64


def test_fresh_authorization_binds_plan_limited_protection_assessment(tmp_path):
    runner = AuthorizationCommentRunner(
        issue_events=[label_event()],
        repo_payload={"private": True},
        pr_branch_protection_returncode=1,
        pr_branch_protection_stderr=(
            "HTTP 403: Upgrade to GitHub Pro or make this repository public"
        ),
        pr_effective_rules_returncode=1,
        pr_effective_rules_stderr=(
            "HTTP 403: Upgrade to GitHub Pro or make this repository public"
        ),
    )
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )

    handoff = authorize_fresh_issue_created_resume(
        runner, config=config, pr_number=7, issue_number=643,
        metadata=replace(metadata(), head_branch="agent-loop/managed-643"),
    )

    assert handoff.protection_mode == "plan_limited"
    record = parse_issue_created_authorization_comment(runner.intent_comments[-1]["body"])
    assert record is not None
    assert record.protection == "plan_limited"


def test_fresh_authorization_race_writes_no_record_or_label(tmp_path):
    runner = RacedAuthorizationCommentRunner(
        issue_events=[label_event()],
        rest_pr={"labels": []},
    )
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )

    with pytest.raises(AgentLoopError, match="changed live PR tuple"):
        authorize_fresh_issue_created_resume(
            runner, config=config, pr_number=7, issue_number=643,
            metadata=replace(metadata(), head_branch="agent-loop/managed-643"),
        )

    assert runner.intent_comments == []
    assert runner.labels_posted is False
    assert runner.dispatch_count == 0


def test_fresh_authorization_reuses_existing_creation_for_same_scope(tmp_path):
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    created = publish_issue_created_authorization(
        runner, config=config, handoff=_authorization_handoff(), metadata=metadata(),
        approved_plan_hash="a" * 64,
    )
    recovered = authorize_fresh_issue_created_resume(
        runner, config=config, pr_number=7, issue_number=643,
        metadata=replace(metadata(), head_branch="agent-loop/managed-643"),
        approved_plan_hash="a" * 64,
    )
    assert recovered.authorization_kind == "creation"
    assert recovered.authorization_comment_id == created.authorization_comment_id
    assert sum(
        "POST" in command and "issues/7/comments" in " ".join(command)
        for command, _cwd in runner.commands
    ) == 1


def test_fresh_authorization_supersedes_prior_grant_for_verified_descendant(tmp_path):
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    first = authorize_fresh_issue_created_resume(
        runner, config=config, pr_number=7, issue_number=643,
        metadata=replace(metadata(), head_branch="agent-loop/managed-643"),
        approved_plan_hash="a" * 64,
    )
    runner.rest_pr["head"]["sha"] = "descendant"
    # A recovery may relabel the PR before authorizing the descendant head.
    # Historical grants remain bound to their own actor-owned label events;
    # only the terminal grant must match the current handoff event.
    runner.issue_events.append(label_event(202))
    runner.compare_payload = {
        "status": "ahead",
        "base_commit": {"sha": "abc123"},
        "merge_base_commit": {"sha": "abc123"},
    }
    advanced = authorize_fresh_issue_created_resume(
        runner, config=config, pr_number=7, issue_number=643,
        metadata=replace(
            metadata(), head_branch="agent-loop/managed-643", head_sha="descendant"
        ),
        approved_plan_hash="a" * 64,
    )

    assert advanced.authorization_kind == "fresh"
    assert advanced.head_sha == "descendant"
    assert advanced.authorization_comment_id != first.authorization_comment_id
    parsed = parse_issue_created_authorization_comment(runner.intent_comments[-1]["body"])
    assert parsed is not None
    assert parsed.predecessor_head == "abc123"
    assert parsed.predecessor_comment_id == first.authorization_comment_id
    assert parsed.label_event_id == 202

    resume = _find_resume_audit(
        runner,
        config=config,
        pr_number=7,
        actor_login="agent-loop",
        actor_id=1,
        base_ref="main",
        issue_number=643,
        live_head="descendant",
        expected_handoff=advanced,
        expected_protection="voluntary",
    )

    assert resume is not None
    assert resume[0] == advanced.authorization_comment_id
    assert resume[1]["nonce"] == advanced.override_nonce


def _rebound_plan_fresh_setup(tmp_path):
    """A pre-rebind grant under plan ``a`` and a descendant head (#993)."""
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    first = authorize_fresh_issue_created_resume(
        runner, config=config, pr_number=7, issue_number=643,
        metadata=replace(metadata(), head_branch="agent-loop/managed-643"),
        approved_plan_hash="a" * 64,
    )
    runner.rest_pr["head"]["sha"] = "descendant"
    runner.compare_payload = {
        "status": "ahead",
        "base_commit": {"sha": "abc123"},
        "merge_base_commit": {"sha": "abc123"},
    }
    live = replace(
        metadata(), head_branch="agent-loop/managed-643", head_sha="descendant"
    )
    return runner, config, first, live


def test_fresh_authorization_treats_verified_retired_plan_grant_as_history(tmp_path):
    runner, config, first, live = _rebound_plan_fresh_setup(tmp_path)

    rebound = authorize_fresh_issue_created_resume(
        runner, config=config, pr_number=7, issue_number=643, metadata=live,
        approved_plan_hash="b" * 64,
        retired_plan_hashes=frozenset({"a" * 64}),
    )

    assert rebound.authorization_kind == "fresh"
    assert rebound.approved_plan_hash == "b" * 64
    assert rebound.retired_plan_hashes == frozenset({"a" * 64})
    parsed = parse_issue_created_authorization_comment(runner.intent_comments[-1]["body"])
    assert parsed is not None
    assert parsed.approved_plan_hash == "b" * 64
    assert parsed.predecessor_comment_id == first.authorization_comment_id

    # A rerun reuses the new grant, and re-validation keeps the retired set.
    posted = len(runner.intent_comments)
    revalidated = revalidate_issue_created_handoff(
        runner, config=config, handoff=rebound, metadata=live
    )
    assert revalidated.authorization_comment_id == rebound.authorization_comment_id
    assert revalidated.retired_plan_hashes == frozenset({"a" * 64})
    assert len(runner.intent_comments) == posted


@pytest.mark.parametrize(
    "retired",
    [
        frozenset(),
        frozenset({"c" * 64}),
        # The live plan itself can never be retired.
        frozenset({"a" * 64, "b" * 64}),
    ],
)
def test_fresh_authorization_still_refuses_unexplained_plan_divergence(tmp_path, retired):
    runner, config, _first, live = _rebound_plan_fresh_setup(tmp_path)
    posted = len(runner.intent_comments)

    with pytest.raises(AgentLoopError, match="conflicting actor-owned record"):
        authorize_fresh_issue_created_resume(
            runner, config=config, pr_number=7, issue_number=643, metadata=live,
            approved_plan_hash="b" * 64,
            retired_plan_hashes=retired,
        )

    assert len(runner.intent_comments) == posted


def test_fresh_authorization_retired_plan_does_not_excuse_other_field_mismatch(tmp_path):
    runner, config, _first, live = _rebound_plan_fresh_setup(tmp_path)
    # The retired-plan grant also carries an unknown label event: still a conflict.
    runner.issue_events.clear()
    runner.issue_events.append(label_event(202))

    with pytest.raises(AgentLoopError, match="conflicting actor-owned record"):
        authorize_fresh_issue_created_resume(
            runner, config=config, pr_number=7, issue_number=643, metadata=live,
            approved_plan_hash="b" * 64,
            retired_plan_hashes=frozenset({"a" * 64}),
        )


def test_fresh_authorization_after_same_head_rebind_uses_the_retired_grant(tmp_path):
    # A signed rebind posts only an issue comment; the PR head is unchanged.
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    live = replace(metadata(), head_branch="agent-loop/managed-643")
    first = publish_issue_created_authorization(
        runner, config=config, handoff=_authorization_handoff(), metadata=metadata(),
        approved_plan_hash="a" * 64,
    )
    # GitHub reports an identical head, which never proves a descendant.
    runner.compare_payload = {"status": "identical"}

    rebound = authorize_fresh_issue_created_resume(
        runner, config=config, pr_number=7, issue_number=643, metadata=live,
        approved_plan_hash="b" * 64,
        retired_plan_hashes=frozenset({"a" * 64}),
    )

    assert rebound.authorization_kind == "fresh"
    assert rebound.head_sha == "abc123"
    parsed = parse_issue_created_authorization_comment(runner.intent_comments[-1]["body"])
    assert parsed is not None
    assert parsed.approved_plan_hash == "b" * 64
    assert parsed.predecessor_head == "abc123"
    assert parsed.predecessor_comment_id == first.authorization_comment_id

    audit_kwargs = dict(
        config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
        base_ref="main", issue_number=643, live_head="abc123",
        expected_protection="voluntary", require_actor_owned_label_event=True,
    )
    # Activation resolves the rebound grant; the retired grant is history.
    resume = _find_resume_audit(runner, expected_handoff=rebound, **audit_kwargs)
    assert resume is not None
    assert resume[0] == rebound.authorization_comment_id
    # Without the verified retired set the old grant still fails closed.
    assert _find_resume_audit(
        runner, expected_handoff=replace(rebound, retired_plan_hashes=frozenset()),
        **audit_kwargs,
    ) is None

    # A rerun reuses the rebound grant instead of posting another.
    posted = len(runner.intent_comments)
    again = authorize_fresh_issue_created_resume(
        runner, config=config, pr_number=7, issue_number=643, metadata=live,
        approved_plan_hash="b" * 64,
        retired_plan_hashes=frozenset({"a" * 64}),
    )
    assert again.authorization_comment_id == rebound.authorization_comment_id
    assert len(runner.intent_comments) == posted


def _same_head_rebound(tmp_path):
    """A creation grant under plan ``a``, then a same-head fresh grant under ``b``."""
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    first = publish_issue_created_authorization(
        runner, config=config, handoff=_authorization_handoff(), metadata=metadata(),
        approved_plan_hash="a" * 64,
    )
    runner.compare_payload = {"status": "identical"}
    rebound = authorize_fresh_issue_created_resume(
        runner, config=config, pr_number=7, issue_number=643,
        metadata=replace(metadata(), head_branch="agent-loop/managed-643"),
        approved_plan_hash="b" * 64,
        retired_plan_hashes=frozenset({"a" * 64}),
    )
    return runner, config, first, rebound


def _advance_head(runner):
    runner.intent_comments.extend([
        _round_comment(88, role="reviewer", subject="abc123", round_number=1, state="blocking"),
        _round_comment(89, role="coder", subject="next-head", round_number=2),
    ])
    runner.rest_pr["head"]["sha"] = "next-head"


def _inject_authorization(runner, comment_id, record):
    runner.intent_comments.append({
        "id": comment_id,
        "user": {"login": "agent-loop", "id": 1},
        "body": str(format_issue_created_authorization_comment(record)),
    })


def _retired_plan_record(**overrides):
    record = ManagedCiIssueAuthorization(
        kind="fresh", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="abc123", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci",
        nonce="retired-fresh", label_event_id=101, approved_plan_hash="a" * 64,
    )
    return replace(record, **overrides)


def test_continuity_after_same_head_rebind_chains_to_the_rebound_grant(tmp_path):
    runner, config, _first, rebound = _same_head_rebound(tmp_path)
    _advance_head(runner)

    continued = publish_issue_created_continuity_authorization(
        runner, config=config, handoff=rebound, predecessor_head="abc123",
        new_head="next-head", round_comment_ids=(88, 89),
    )

    assert continued.authorization_kind == "continuity"
    assert continued.retired_plan_hashes == frozenset({"a" * 64})
    parsed = parse_issue_created_authorization_comment(runner.intent_comments[-1]["body"])
    assert parsed is not None
    assert parsed.approved_plan_hash == "b" * 64
    assert parsed.predecessor_comment_id == rebound.authorization_comment_id
    audit = _find_resume_audit(
        runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
        base_ref="main", issue_number=643, live_head="next-head",
        expected_handoff=continued, expected_protection="voluntary",
        require_actor_owned_label_event=True,
    )
    assert audit is not None
    assert audit[0] == continued.authorization_comment_id


def test_continuity_after_same_head_rebind_without_retirement_refuses(tmp_path):
    runner, config, _first, rebound = _same_head_rebound(tmp_path)
    _advance_head(runner)
    posted = len(runner.intent_comments)

    with pytest.raises(AgentLoopError, match="conflicting prior authorizations"):
        publish_issue_created_continuity_authorization(
            runner, config=config,
            handoff=replace(rebound, retired_plan_hashes=frozenset()),
            predecessor_head="abc123", new_head="next-head", round_comment_ids=(88, 89),
        )
    assert len(runner.intent_comments) == posted


def test_continuity_refuses_retired_plan_record_with_inconsistent_fields(tmp_path):
    runner, config, _first, rebound = _same_head_rebound(tmp_path)
    _inject_authorization(runner, 60, _retired_plan_record(protection="plan_limited"))
    _advance_head(runner)
    posted = len(runner.intent_comments)

    with pytest.raises(AgentLoopError, match="conflicting prior authorizations"):
        publish_issue_created_continuity_authorization(
            runner, config=config, handoff=rebound, predecessor_head="abc123",
            new_head="next-head", round_comment_ids=(88, 89),
        )
    assert len(runner.intent_comments) == posted


@pytest.mark.parametrize(
    "overrides",
    [{"protection": "plan_limited"}, {"label_event_id": 999}],
)
def test_resume_audit_retires_only_the_plan_hash(tmp_path, overrides):
    runner, config, _first, rebound = _same_head_rebound(tmp_path)
    audit_kwargs = dict(
        config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
        base_ref="main", issue_number=643, live_head="abc123",
        expected_handoff=rebound, expected_protection="voluntary",
        require_actor_owned_label_event=True,
    )
    # A consistent retired-plan grant is history.
    _inject_authorization(runner, 60, _retired_plan_record())
    audit = _find_resume_audit(runner, **audit_kwargs)
    assert audit is not None and audit[0] == rebound.authorization_comment_id
    # One whose non-plan fields disagree still fails closed.
    _inject_authorization(runner, 61, _retired_plan_record(nonce="other", **overrides))
    assert _find_resume_audit(runner, **audit_kwargs) is None


def test_fresh_authorization_same_head_plan_change_without_retirement_refuses(tmp_path):
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    publish_issue_created_authorization(
        runner, config=config, handoff=_authorization_handoff(), metadata=metadata(),
        approved_plan_hash="a" * 64,
    )
    posted = len(runner.intent_comments)

    with pytest.raises(AgentLoopError, match="conflicting actor-owned record"):
        authorize_fresh_issue_created_resume(
            runner, config=config, pr_number=7, issue_number=643,
            metadata=replace(metadata(), head_branch="agent-loop/managed-643"),
            approved_plan_hash="b" * 64,
        )
    assert len(runner.intent_comments) == posted


def test_fresh_authorization_rejects_actor_owned_record_with_unknown_label_event(tmp_path):
    forged = ManagedCiIssueAuthorization(
        kind="creation", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="abc123", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="root",
        label_event_id=999,
    )
    runner = AuthorizationCommentRunner(
        issue_events=[label_event()],
        intent_comments=[{
            "id": 41,
            "user": {"login": "agent-loop", "id": 1},
            "body": str(format_issue_created_authorization_comment(forged)),
        }],
    )
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )

    with pytest.raises(AgentLoopError, match="conflicting actor-owned record"):
        authorize_fresh_issue_created_resume(
            runner,
            config=config,
            pr_number=7,
            issue_number=643,
            metadata=replace(metadata(), head_branch="agent-loop/managed-643"),
        )


def test_recovery_accepts_continuity_terminal_for_opening_body_head(tmp_path):
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    initial = publish_issue_created_authorization(
        runner, config=config, handoff=_authorization_handoff(), metadata=metadata()
    )
    runner.intent_comments.extend([
        _round_comment(88, role="reviewer", subject="abc123", round_number=1, state="blocking"),
        _round_comment(89, role="coder", subject="next-head", round_number=2),
    ])
    runner.rest_pr["head"]["sha"] = "next-head"
    continued = publish_issue_created_continuity_authorization(
        runner,
        config=config,
        handoff=initial,
        predecessor_head="abc123",
        new_head="next-head",
        round_comment_ids=(88, 89),
    )
    recovered_metadata = replace(
        metadata(),
        head_branch="agent-loop/managed-643",
        head_sha="next-head",
    )

    recovered = recover_issue_created_handoff(
        runner,
        config=config,
        pr_number=7,
        metadata=recovered_metadata,
        issue_number=643,
    )

    assert recovered is not None
    assert recovered.opening_override_nonce == "nonce-643"
    audit = _find_resume_audit(
        runner,
        config=config,
        pr_number=7,
        actor_login="agent-loop",
        actor_id=1,
        base_ref="main",
        issue_number=643,
        live_head="next-head",
        expected_handoff=recovered,
        expected_protection="voluntary",
        require_actor_owned_label_event=True,
    )
    assert audit is not None
    assert audit[0] == continued.authorization_comment_id
    assert audit[1]["kind"] == "continuity"


def test_continuity_revalidation_never_uses_terminal_nonce_for_opening_body(
    tmp_path,
):
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    handoff = replace(
        _authorization_handoff(),
        lifecycle="draft-labeled",
        authorization_kind="continuity",
        override_nonce="continuity-nonce",
        opening_override_nonce=None,
        active_label_event_id=101,
        authorization_comment_id=42,
    )

    validated = revalidate_issue_created_handoff(
        runner,
        config=config,
        handoff=handoff,
        metadata=replace(metadata(), head_branch="agent-loop/managed-643"),
    )

    # Tuple revalidation uses the opening nonce for the PR body, while the
    # continuity nonce remains the terminal authorization identity.
    assert validated.override_nonce == "continuity-nonce"
    assert validated.authorization_kind == "continuity"
    assert validated.opening_override_nonce == "nonce-643"


def _released_label_revalidation(tmp_path, issue_events, *, labels=()):
    runner = AuthorizationCommentRunner(
        issue_events=issue_events,
        rest_pr={"labels": [{"name": name} for name in labels]},
    )
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    handoff = replace(
        _authorization_handoff(),
        lifecycle="draft-unlabeled-reentry",
        authorization_kind="creation",
        opening_override_nonce=None,
        active_label_event_id=101,
        authorization_comment_id=42,
    )
    return revalidate_issue_created_handoff(
        runner,
        config=config,
        handoff=handoff,
        metadata=replace(metadata(), head_branch="agent-loop/managed-643"),
    )


def test_released_label_after_failed_activation_revalidates_for_reentry(tmp_path):
    # #997: a failed activation removes the label, so the recorded event is
    # historical rather than active.  That is the documented unlabeled
    # re-entry state, not a provenance change.
    validated = _released_label_revalidation(
        tmp_path, [label_event(101), label_event(102, event="unlabeled")]
    )

    assert validated.lifecycle == "draft-unlabeled-reentry"
    assert validated.active_label_event_id == 101


@pytest.mark.parametrize(
    ("applied_id", "login", "actor_id"),
    [
        # The recorded event does not exist in the timeline at all.
        (103, "agent-loop", 1),
        # The recorded event id was applied by a different actor.
        (101, "intruder", 9),
    ],
)
def test_released_label_reentry_still_rejects_foreign_label_event(
    tmp_path, applied_id, login, actor_id
):
    issue_events = [
        label_event(applied_id, login=login, actor_id=actor_id),
        label_event(applied_id + 1, event="unlabeled"),
    ]
    with pytest.raises(AgentLoopError, match="label provenance changed"):
        _released_label_revalidation(tmp_path, issue_events)


def test_released_label_reentry_rejects_relabeled_pr(tmp_path):
    with pytest.raises(AgentLoopError, match="opening tuple"):
        _released_label_revalidation(
            tmp_path, [label_event(101)], labels=(MANAGED_LABEL,)
        )


def test_labeled_revalidation_still_requires_the_active_label_event(tmp_path):
    runner = AuthorizationCommentRunner(
        issue_events=[label_event(101), label_event(102, event="unlabeled"), label_event(103)]
    )
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    handoff = replace(
        _authorization_handoff(),
        lifecycle="draft-labeled",
        authorization_kind="creation",
        opening_override_nonce=None,
        active_label_event_id=101,
        authorization_comment_id=42,
    )

    with pytest.raises(AgentLoopError, match="label provenance changed"):
        revalidate_issue_created_handoff(
            runner,
            config=config,
            handoff=handoff,
            metadata=replace(metadata(), head_branch="agent-loop/managed-643"),
        )


def test_public_pr_missing_authorization_reaches_parser_valid_fresh_remedy(tmp_path):
    recovered_metadata = replace(
        metadata(),
        head_branch="agent-loop/managed-643",
    )
    runner = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        rest_pr={
            "body": recovered_metadata.body,
            "state": "open",
            "labels": [],
            "draft": True,
            "head": {
                "repo": {"full_name": "OWNER/REPO"},
                "sha": "abc123",
                "ref": "agent-loop/managed-643",
            },
            "user": {"login": "agent-loop", "id": 1},
        },
        issue_events=[label_event()],
        intent_comments=[],
    )
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci",
            "--managed-ci-trusted-actor", "agent-loop",
            "--allow-unprotected-managed-ci",
        ),
    )

    handoff = recover_issue_created_handoff(
        runner,
        config=config,
        pr_number=7,
        metadata=recovered_metadata,
        issue_number=643,
    )
    assert handoff is not None

    with pytest.raises(AgentLoopError) as exc_info:
        activate_managed_ci(
            runner,
            config=config,
            pr_number=7,
            metadata=recovered_metadata,
            managed_resume=AuthenticatedManagedResume(
                origin="issue-created",
                lifecycle=handoff.lifecycle,
                issue_created_handoff=handoff,
            ),
        )

    command = str(exc_info.value).split("`", 2)[1]
    parsed = build_parser().parse_args(shlex.split(command)[1:])
    assert parsed.command == "pr"
    assert parsed.managed_ci_fresh_authorization is True
    assert parsed.managed_ci_issue == 643
    assert runner.labels_posted is False
    assert runner.dispatch_count == 0


def test_fresh_authorization_rejects_unrelated_replacement_head(tmp_path):
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    authorize_fresh_issue_created_resume(
        runner, config=config, pr_number=7, issue_number=643,
        metadata=replace(metadata(), head_branch="agent-loop/managed-643"),
    )
    runner.rest_pr["head"]["sha"] = "replacement"
    runner.compare_payload = {
        "status": "diverged",
        "base_commit": {"sha": "abc123"},
        "merge_base_commit": {"sha": "other"},
    }

    with pytest.raises(AgentLoopError, match="did not prove.*descendant"):
        authorize_fresh_issue_created_resume(
            runner, config=config, pr_number=7, issue_number=643,
            metadata=replace(
                metadata(), head_branch="agent-loop/managed-643", head_sha="replacement"
            ),
        )


def test_fresh_authorization_rejects_missing_server_issue_association(tmp_path):
    runner = AuthorizationCommentRunner(
        issue_events=[label_event()],
        issue_timeline=[{"event": "commented"}],
    )
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    with pytest.raises(AgentLoopError, match="server-observed issue-to-PR association"):
        authorize_fresh_issue_created_resume(
            runner, config=config, pr_number=7, issue_number=643,
            metadata=replace(metadata(), head_branch="agent-loop/managed-643", body="Fixes #643"),
        )


def test_fresh_authorization_rejects_same_number_foreign_repository_association(tmp_path):
    runner = AuthorizationCommentRunner(
        issue_events=[label_event()],
        issue_timeline=[{
            "event": "cross-referenced",
            "source": {"issue": {
                "number": 7,
                "repository_url": "https://api.github.com/repos/OTHER/REPO",
                "pull_request": {"url": "https://api.github.com/repos/OTHER/REPO/pulls/7"},
            }},
        }],
    )
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )

    with pytest.raises(AgentLoopError, match="server-observed issue-to-PR association"):
        authorize_fresh_issue_created_resume(
            runner, config=config, pr_number=7, issue_number=643,
            metadata=replace(metadata(), head_branch="agent-loop/managed-643", body="Fixes #643"),
        )

    assert not any(
        "AGENT_MANAGED_CI_ISSUE_AUTHORIZATION_V1" in " ".join(command)
        and "POST" in command
        for command, _cwd in runner.commands
    )


def test_creation_plus_fresh_at_same_head_selects_fresh_grant(tmp_path):
    root = ManagedCiIssueAuthorization(
        kind="creation", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="abc123", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="root",
        label_event_id=101,
    )
    fresh = replace(root, kind="fresh", nonce="fresh")
    comments = [
        {"id": 41, "user": {"login": "agent-loop", "id": 1}, "body": str(format_issue_created_authorization_comment(root))},
        {"id": 42, "user": {"login": "agent-loop", "id": 1}, "body": str(format_issue_created_authorization_comment(fresh))},
    ]
    runner = V2ManagedRunner(intent_comments=comments)
    config = make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop")
    audit = _find_resume_audit(
        runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
        base_ref="main", issue_number=643, live_head="abc123",
    )
    assert audit is not None
    assert audit[0] == 42
    assert audit[1]["kind"] == "fresh"


def test_continuity_duplicate_root_comment_ids_remain_resumable(tmp_path):
    root = ManagedCiIssueAuthorization(
        kind="creation", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="abc123", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="root",
        label_event_id=101,
    )
    continuity = ManagedCiIssueAuthorization(
        kind="continuity", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="next-head", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="next",
        label_event_id=101, predecessor_head="abc123", predecessor_comment_id=40,
        round_comment_ids=(88, 89),
    )
    comments = [
        {"id": 40, "user": {"login": "agent-loop", "id": 1}, "body": str(format_issue_created_authorization_comment(root))},
        {"id": 41, "user": {"login": "agent-loop", "id": 1}, "body": str(format_issue_created_authorization_comment(root))},
        _round_comment(88, role="reviewer", subject="abc123", round_number=1, state="blocking"),
        _round_comment(89, role="coder", subject="next-head", round_number=2),
        {"id": 90, "user": {"login": "agent-loop", "id": 1}, "body": str(format_issue_created_authorization_comment(continuity))},
    ]
    runner = V2ManagedRunner(intent_comments=comments, rest_pr={"head": {"sha": "next-head"}})
    config = make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop")
    audit = _find_resume_audit(
        runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
        base_ref="main", issue_number=643, live_head="next-head",
    )
    assert audit is not None and audit[0] == 90


@pytest.mark.parametrize("mutation", ["non-actor", "repo", "issue", "pr", "base"])
def test_resume_rejects_untrusted_or_mismatched_authorization(tmp_path, mutation):
    record = ManagedCiIssueAuthorization(
        kind="creation", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="abc123", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="root",
        label_event_id=101,
    )
    user = {"login": "agent-loop", "id": 1}
    if mutation == "non-actor":
        user = {"login": "attacker", "id": 2}
    elif mutation == "repo":
        record = replace(record, repository="OTHER/REPO")
    elif mutation == "issue":
        record = replace(record, issue_number=999)
    elif mutation == "pr":
        record = replace(record, pr_number=999)
    elif mutation == "base":
        record = replace(record, base_ref="release")
    runner = V2ManagedRunner(intent_comments=[{
        "id": 41, "user": user,
        "body": str(format_issue_created_authorization_comment(record)),
    }])
    config = make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop")
    assert _find_resume_audit(
        runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
        base_ref="main", issue_number=643, live_head="abc123",
    ) is None


@pytest.mark.parametrize(
    "mutation",
    ["actor-payload", "actor-id-payload", "nonce", "waiver", "protection", "label-event", "plan"],
)
def test_resume_rejects_every_mismatched_authorization_binding(tmp_path, mutation):
    handoff = replace(
        _authorization_handoff(), approved_plan_hash="plan-hash",
        authorization_comment_id=41,
    )
    record = ManagedCiIssueAuthorization(
        kind="creation", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="abc123", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="nonce-643",
        label_event_id=101, approved_plan_hash="plan-hash",
    )
    changes = {
        "actor-payload": {"actor_login": "attacker"},
        "actor-id-payload": {"actor_id": 2},
        "nonce": {"nonce": "wrong"},
        "waiver": {"waiver": "different-waiver"},
        "protection": {"protection": "plan_limited"},
        "label-event": {"label_event_id": 999},
        "plan": {"approved_plan_hash": "other-plan"},
    }
    record = replace(record, **changes[mutation])
    runner = V2ManagedRunner(
        intent_comments=[{
            "id": 41,
            "user": {"login": "agent-loop", "id": 1},
            "body": str(format_issue_created_authorization_comment(record)),
        }],
        issue_events=[label_event()],
    )
    config = make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop")

    assert _find_resume_audit(
        runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
        base_ref="main", issue_number=643, live_head="abc123",
        expected_handoff=handoff, expected_protection="voluntary",
    ) is None


@pytest.mark.parametrize(
    "mutation",
    ["actor-payload", "actor-id-payload", "nonce", "waiver", "protection", "label-event", "plan"],
)
def test_activation_rejects_mismatched_authorization_without_label_or_dispatch(tmp_path, mutation):
    handoff = replace(
        _authorization_handoff(), approved_plan_hash="plan-hash",
        authorization_comment_id=41,
    )
    record = ManagedCiIssueAuthorization(
        kind="creation", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="abc123", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="nonce-643",
        label_event_id=101, approved_plan_hash="plan-hash",
    )
    changes = {
        "actor-payload": {"actor_login": "attacker"},
        "actor-id-payload": {"actor_id": 2},
        "nonce": {"nonce": "wrong"},
        "waiver": {"waiver": "different-waiver"},
        "protection": {"protection": "plan_limited"},
        "label-event": {"label_event_id": 999},
        "plan": {"approved_plan_hash": "other-plan"},
    }
    record = replace(record, **changes[mutation])
    runner = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        rest_pr={"state": "open", "draft": True, "labels": []},
        issue_events=[label_event()],
        intent_comments=[{
            "id": 41,
            "user": {"login": "agent-loop", "id": 1},
            "body": str(format_issue_created_authorization_comment(record)),
        }],
    )
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci",
            "--managed-ci-trusted-actor", "agent-loop",
            "--allow-unprotected-managed-ci",
        ),
    )

    with pytest.raises(AgentLoopError, match="--managed-ci-fresh"):
        activate_managed_ci(
            runner,
            config=config,
            pr_number=7,
            metadata=metadata(),
            managed_resume=AuthenticatedManagedResume(
                origin="issue-created",
                lifecycle="draft-unlabeled-reentry",
                issue_created_handoff=handoff,
            ),
        )

    assert runner.labels_posted is False
    assert runner.dispatch_count == 0
    assert not any(
        command[:5] == [
            "gh", "api", "--method", "POST", "repos/OWNER/REPO/issues/7/labels"
        ]
        for command, _cwd in runner.commands
    )


def test_resume_rejects_unsolicited_live_head(tmp_path):
    record = ManagedCiIssueAuthorization(
        kind="creation", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="abc123", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="root",
        label_event_id=101,
    )
    runner = V2ManagedRunner(intent_comments=[{
        "id": 41, "user": {"login": "agent-loop", "id": 1},
        "body": str(format_issue_created_authorization_comment(record)),
    }])
    config = make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop")
    assert _find_resume_audit(
        runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
        base_ref="main", issue_number=643, live_head="unsolicited",
    ) is None


def test_public_resume_requires_authorized_label_event_for_versioned_record(tmp_path):
    record = ManagedCiIssueAuthorization(
        kind="creation", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="abc123", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="root",
        label_event_id=101,
    )
    config = make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop")
    comments = [{
        "id": 41, "user": {"login": "agent-loop", "id": 1},
        "body": str(format_issue_created_authorization_comment(record)),
    }]

    without_event = V2ManagedRunner(intent_comments=comments, issue_events=[])
    assert _find_resume_audit(
        without_event, config=config, pr_number=7,
        actor_login="agent-loop", actor_id=1, base_ref="main",
        issue_number=643, live_head="abc123", require_actor_owned_label_event=True,
    ) is None

    with_event = V2ManagedRunner(intent_comments=comments, issue_events=[label_event()])
    assert _find_resume_audit(
        with_event, config=config, pr_number=7,
        actor_login="agent-loop", actor_id=1, base_ref="main",
        issue_number=643, live_head="abc123", require_actor_owned_label_event=True,
    ) is not None


def test_activation_rejects_stale_authorization_before_dispatch(tmp_path, monkeypatch):
    runner = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        rest_pr={"state": "open", "draft": True, "labels": [{"name": MANAGED_LABEL}]},
        issue_events=[label_event()],
    )
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        managed_ci_issue_number=643,
    )
    handoff = _authorization_handoff()
    monkeypatch.setattr(
        managed_ci, "_find_resume_audit",
        lambda *args, **kwargs: (41, {
            "nonce": "old", "repo": "OWNER/REPO", "base": "main",
            "head": "older-head", "protection": "voluntary",
            "active_label_event_id": "101", "kind": "creation",
            "issue": "643", "pr": "7",
        }),
    )

    with pytest.raises(AgentLoopError, match="bound to an older head"):
        activate_managed_ci(
            runner, config=config, pr_number=7, metadata=metadata(),
            managed_resume=AuthenticatedManagedResume(
                origin="issue-created", lifecycle="draft-labeled",
                issue_created_handoff=handoff,
            ),
        )

    commands = [command for command, _cwd in runner.commands]
    assert any("DELETE" in command for command in commands)
    assert not any("dispatches" in " ".join(command) for command in commands)


def test_unlabeled_activation_rejects_wrong_terminal_comment_without_label_or_dispatch(
    tmp_path, monkeypatch
):
    runner = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        rest_pr={"state": "open", "draft": True, "labels": []},
        issue_events=[label_event()],
        intent_comments=[{
            "id": 42,
            "user": {"login": "agent-loop", "id": 1},
            "body": str(format_issue_created_authorization_comment(
                ManagedCiIssueAuthorization(
                    kind="creation", repository="OWNER/REPO", issue_number=643,
                    pr_number=7, base_ref="main", head_sha="abc123",
                    actor_login="agent-loop", actor_id=1,
                    protection="voluntary", waiver="allow-unprotected-managed-ci",
                    nonce="nonce-643", label_event_id=101,
                )
            )),
        }],
    )
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci",
            "--managed-ci-trusted-actor", "agent-loop",
            "--allow-unprotected-managed-ci",
        ),
    )
    handoff = replace(_authorization_handoff(), authorization_comment_id=41)

    with pytest.raises(AgentLoopError, match="--managed-ci-fresh") as exc_info:
        activate_managed_ci(
            runner,
            config=config,
            pr_number=7,
            metadata=metadata(),
            managed_resume=AuthenticatedManagedResume(
                origin="issue-created",
                lifecycle="draft-unlabeled-reentry",
                issue_created_handoff=handoff,
            ),
        )

    command = str(exc_info.value).split("`", 2)[1]
    parsed = build_parser().parse_args(shlex.split(command)[1:])
    assert parsed.command == "pr"
    assert parsed.managed_ci_fresh_authorization is True
    assert parsed.managed_ci_issue == 643
    assert runner.labels_posted is False
    assert runner.dispatch_count == 0
    assert not any(
        command[:5] == [
            "gh", "api", "--method", "POST", "repos/OWNER/REPO/issues/7/labels"
        ]
        for command, _cwd in runner.commands
    )


@pytest.mark.parametrize("source", ["non-actor-comment", "pr-body-copy"])
def test_activation_rejects_forged_authorization_surface_without_label_or_dispatch(
    tmp_path, source,
):
    record = ManagedCiIssueAuthorization(
        kind="creation", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="abc123", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="root",
        label_event_id=101,
    )
    comments = []
    body = metadata().body
    if source == "non-actor-comment":
        comments = [{
            "id": 41,
            "user": {"login": "attacker", "id": 2},
            "body": str(format_issue_created_authorization_comment(record)),
        }]
    else:
        body = str(format_issue_created_authorization_comment(record))
    runner = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        rest_pr={"state": "open", "draft": True, "labels": [], "body": body},
        issue_events=[label_event()],
        intent_comments=comments,
    )
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci",
            "--managed-ci-trusted-actor", "agent-loop",
            "--allow-unprotected-managed-ci",
        ),
    )
    handoff = replace(
        _authorization_handoff(),
        lifecycle="draft-unlabeled-reentry",
        opening_override_nonce="nonce-643",
    )

    with pytest.raises(AgentLoopError, match="--managed-ci-fresh"):
        activate_managed_ci(
            runner,
            config=config,
            pr_number=7,
            metadata=replace(metadata(), head_branch="agent-loop/managed-643"),
            managed_resume=AuthenticatedManagedResume(
                origin="issue-created",
                lifecycle="draft-unlabeled-reentry",
                issue_created_handoff=handoff,
            ),
        )

    assert runner.labels_posted is False
    assert runner.dispatch_count == 0
    assert not any(
        command[:5] == [
            "gh", "api", "--method", "POST", "repos/OWNER/REPO/issues/7/labels"
        ]
        for command, _cwd in runner.commands
    )


def test_activation_rejects_unsolicited_head_before_labeling(tmp_path):
    record = ManagedCiIssueAuthorization(
        kind="creation", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="abc123", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="root",
        label_event_id=101,
    )
    runner = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        rest_pr={
            "state": "open", "draft": True, "labels": [],
            "head": {
                "repo": {"full_name": "OWNER/REPO"},
                "sha": "unsolicited-head",
                "ref": "agent-loop/managed-643",
            },
        },
        issue_events=[label_event()],
        intent_comments=[{
            "id": 41,
            "user": {"login": "agent-loop", "id": 1},
            "body": str(format_issue_created_authorization_comment(record)),
        }],
    )
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci",
            "--managed-ci-trusted-actor", "agent-loop",
            "--allow-unprotected-managed-ci",
        ),
    )
    handoff = replace(
        _authorization_handoff(head="unsolicited-head"),
        lifecycle="draft-unlabeled-reentry",
        opening_override_nonce="nonce-643",
    )

    with pytest.raises(AgentLoopError, match="no fully bound actor-owned issue-created authorization"):
        activate_managed_ci(
            runner,
            config=config,
            pr_number=7,
            metadata=replace(
                metadata(),
                head_branch="agent-loop/managed-643",
                head_sha="unsolicited-head",
            ),
            managed_resume=AuthenticatedManagedResume(
                origin="issue-created",
                lifecycle="draft-unlabeled-reentry",
                issue_created_handoff=handoff,
            ),
        )

    assert runner.labels_posted is False
    assert runner.dispatch_count == 0
    assert not any(
        command[:5] == [
            "gh", "api", "--method", "POST", "repos/OWNER/REPO/issues/7/labels"
        ]
        for command, _cwd in runner.commands
    )


def test_activation_race_releases_label_without_dispatch_or_new_authorization(tmp_path):
    record = ManagedCiIssueAuthorization(
        kind="creation", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="abc123", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="nonce-643",
        label_event_id=101,
    )
    runner = RacedAuthorizationCommentRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        issue_events=[label_event()],
        intent_comments=[{
            "id": 41,
            "user": {"login": "agent-loop", "id": 1},
            "body": str(format_issue_created_authorization_comment(record)),
        }],
    )
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci",
            "--managed-ci-trusted-actor", "agent-loop",
            "--allow-unprotected-managed-ci",
        ),
    )

    with pytest.raises(AgentLoopError, match="immutable resume tuple changed before activation"):
        activate_managed_ci(
            runner,
            config=config,
            pr_number=7,
            metadata=metadata(),
            managed_resume=AuthenticatedManagedResume(
                origin="issue-created",
                lifecycle="draft-labeled",
                issue_created_handoff=replace(
                    _authorization_handoff(),
                    active_label_event_id=101,
                    authorization_comment_id=41,
                    opening_override_nonce="nonce-643",
                ),
            ),
        )

    assert any(
        command[:5] == [
            "gh", "api", "--method", "DELETE",
            f"repos/OWNER/REPO/issues/7/labels/{MANAGED_LABEL}",
        ]
        for command, _cwd in runner.commands
    )
    assert runner.dispatch_count == 0
    assert sum(
        "AGENT_MANAGED_CI_ISSUE_AUTHORIZATION_V1" in " ".join(command)
        and "POST" in command
        for command, _cwd in runner.commands
    ) == 0


def test_fresh_remedy_is_not_advertised_without_the_explicit_waiver(tmp_path):
    runner = V2ManagedRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=False,
    )

    with pytest.raises(AgentLoopError) as exc_info:
        _release_for_ordinary_recovery(
            runner,
            config=config,
            pr_number=7,
            base_ref="main",
            expected_head_sha="abc123",
            active_event=(101, "agent-loop", 1),
            reason="authorization provenance is unavailable",
            recovery_capable=True,
            fresh_issue_number=643,
            fresh_authorization_allowed=True,
        )

    message = str(exc_info.value)
    assert "--managed-ci-fresh" not in message
    assert "no issue-created authorization grant was inferred" in message


def test_ordinary_release_renders_recovered_issue_scope_as_parser_valid_fresh_command(tmp_path):
    runner = V2ManagedRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci",
            "--managed-ci-trusted-actor", "agent-loop",
            "--allow-unprotected-managed-ci",
        ),
    )

    with pytest.raises(AgentLoopError) as exc_info:
        _release_for_ordinary_recovery(
            runner,
            config=config,
            pr_number=7,
            base_ref="main",
            expected_head_sha="abc123",
            active_event=(101, "agent-loop", 1),
            reason="no fully bound actor-owned issue-created authorization reaches the live head",
            recovery_capable=True,
            fresh_issue_number=643,
            fresh_authorization_allowed=True,
        )

    command = str(exc_info.value).split("`", 2)[1]
    parsed = build_parser().parse_args(shlex.split(command)[1:])
    assert parsed.command == "pr"
    assert parsed.pr_number == 7
    assert parsed.managed_ci_fresh_authorization is True
    assert parsed.managed_ci_issue == 643
    assert parsed.allow_unprotected_managed_ci is True


def test_draft_unlabeled_missing_authorization_does_not_apply_label_and_prints_valid_fresh_command(
    tmp_path,
):
    runner = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        rest_pr={"state": "open", "draft": True, "labels": []},
        issue_events=[label_event()],
        intent_comments=[],
    )
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci",
            "--managed-ci-trusted-actor", "agent-loop", "--allow-unprotected-managed-ci",
        ),
    )
    handoff = _authorization_handoff()

    with pytest.raises(AgentLoopError) as exc_info:
        activate_managed_ci(
            runner, config=config, pr_number=7, metadata=metadata(),
            managed_resume=AuthenticatedManagedResume(
                origin="issue-created", lifecycle="draft-unlabeled-reentry",
                issue_created_handoff=handoff,
            ),
        )

    message = str(exc_info.value)
    assert "--managed-ci-fresh" in message
    command = message.split("`", 2)[1]
    parsed = build_parser().parse_args(shlex.split(command)[1:])
    assert parsed.command == "pr"
    assert parsed.managed_ci_issue == 643
    assert runner.labels_posted is False
    assert runner.dispatch_count == 0


def test_draft_unlabeled_stale_activation_prints_fresh_recovery_without_labeling(
    tmp_path, monkeypatch
):
    runner = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        rest_pr={"state": "open", "draft": True, "labels": []},
        issue_events=[label_event()],
    )
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci",
            "--managed-ci-trusted-actor", "agent-loop",
            "--allow-unprotected-managed-ci",
        ),
    )
    monkeypatch.setattr(
        managed_ci,
        "_find_resume_audit",
        lambda *args, **kwargs: (41, {
            "nonce": "old",
            "repo": "OWNER/REPO",
            "base": "main",
            "head": "older-head",
            "protection": "voluntary",
            "active_label_event_id": "101",
            "kind": "creation",
            "issue": "643",
            "pr": "7",
        }),
    )

    with pytest.raises(AgentLoopError, match="--managed-ci-fresh") as exc_info:
        activate_managed_ci(
            runner,
            config=config,
            pr_number=7,
            metadata=metadata(),
            managed_resume=AuthenticatedManagedResume(
                origin="issue-created",
                lifecycle="draft-unlabeled-reentry",
                issue_created_handoff=_authorization_handoff(),
            ),
        )

    command = str(exc_info.value).split("`", 2)[1]
    parsed = build_parser().parse_args(shlex.split(command)[1:])
    assert parsed.managed_ci_issue == 643
    assert runner.labels_posted is False
    assert runner.dispatch_count == 0


def test_source_managed_release_never_advertises_issue_created_fresh_authorization(tmp_path):
    runner = V2ManagedRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        pr_origin_flow="managed-pr",
    )

    with pytest.raises(AgentLoopError) as exc_info:
        managed_ci._release_for_ordinary_recovery(
            runner, config=config, pr_number=7, base_ref="main",
            expected_head_sha="abc123", active_event=(101, "agent-loop", 1),
            reason="the active managed-label event is temporarily unreadable",
            recovery_capable=True,
        )

    assert "--managed-ci-fresh" not in str(exc_info.value)
    assert "no issue-created authorization grant was inferred" in str(exc_info.value)


def test_source_managed_activation_release_never_advertises_fresh_grant(tmp_path):
    runner = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        repo_payload={"private": True},
        pr_branch_protection_returncode=1,
        pr_branch_protection_stderr="HTTP 403: Upgrade to GitHub Pro or make this repository public",
        pr_effective_rules_returncode=1,
        pr_effective_rules_stderr="HTTP 403: Upgrade to GitHub Pro or make this repository public",
        issue_events=[label_event()],
    )
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
        pr_origin_flow="managed-pr",
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci",
            "--managed-ci-trusted-actor", "agent-loop",
            "--allow-unprotected-managed-ci",
        ),
    )

    with pytest.raises(AgentLoopError) as exc_info:
        activate_managed_ci(
            runner,
            config=config,
            pr_number=7,
            metadata=metadata(),
            managed_resume=AuthenticatedManagedResume(
                origin="source-managed", lifecycle="draft-labeled"
            ),
        )

    message = str(exc_info.value)
    assert "--managed-ci-fresh" not in message
    assert "no issue-created authorization grant was inferred" in message
    assert any(
        command[:5] == [
            "gh", "api", "--method", "DELETE",
            f"repos/OWNER/REPO/issues/7/labels/{MANAGED_LABEL}",
        ]
        for command, _cwd in runner.commands
    )
    assert runner.dispatch_count == 0


def test_pr_body_authorization_copy_cannot_supply_resume_provenance(tmp_path):
    record = ManagedCiIssueAuthorization(
        kind="creation", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="abc123", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="root",
        label_event_id=101,
    )
    runner = V2ManagedRunner(
        intent_comments=[],
        rest_pr={"body": str(format_issue_created_authorization_comment(record))},
    )
    config = make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop")

    assert _find_resume_audit(
        runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
        base_ref="main", issue_number=643, live_head="abc123",
    ) is None


def test_issue_authorization_forked_head_chain_fails_closed(tmp_path):
    root = ManagedCiIssueAuthorization(
        kind="creation", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="abc123", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="root",
        label_event_id=101,
    )
    first = ManagedCiIssueAuthorization(
        kind="continuity", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="next-head", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="first",
        label_event_id=101, predecessor_head="abc123", predecessor_comment_id=41,
        round_comment_ids=(88,),
    )
    fork = replace(first, nonce="fork", round_comment_ids=(89,))
    comments = [
        {"id": 41, "user": {"login": "agent-loop", "id": 1}, "body": str(format_issue_created_authorization_comment(root))},
        {"id": 42, "user": {"login": "agent-loop", "id": 1}, "body": str(format_issue_created_authorization_comment(first))},
        {"id": 43, "user": {"login": "agent-loop", "id": 1}, "body": str(format_issue_created_authorization_comment(fork))},
    ]
    runner = V2ManagedRunner(intent_comments=comments, rest_pr={"head": {"sha": "next-head"}})
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    assert _find_resume_audit(
        runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
        base_ref="main", issue_number=643, live_head="next-head",
    ) is None


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


def test_strict_preflight_does_not_mint_unprotected_authorization_nonce(tmp_path):
    active_rule = {
        "enforcement": "active",
        "bypass_actors": [],
        "rules": [{
            "type": "required_status_checks",
            "parameters": {"required_status_checks": [{"context": FINAL_CONTEXT}]},
        }],
    }
    runner = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        rest_pr={"body": "Fixes #643"},
        pr_branch_protection_payload={"contexts": []},
        pr_effective_rules_payload=[{"ruleset_id": 8}],
        pr_rulesets_payload={8: active_rule},
    )
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )

    intent = preflight_managed_ci_creation(runner, config=config, issue_number=643)

    assert intent is not None
    assert intent.protection_mode == "strict"
    assert intent.audit_nonce is None


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
    assert contract.intent_generation
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


def test_source_managed_override_accepts_only_its_source_marker(tmp_path):
    from coding_review_agent_loop.managed_pr import _compose_body

    nonce = "nonce-from-preflight"
    body = str(_compose_body(
        "## Summary\n\nPrepared change.",
        source_branch="fix/prepared-change",
        source_sha="a" * 40,
        override_nonce=nonce,
    ))
    source_runner = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        rest_pr={"body": body},
    )
    source_config = replace(
        make_config(
            tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop",
            allow_unprotected_managed_ci=True,
            managed_ci_expected_override_nonce=nonce,
        ),
        pr_origin_flow="managed-pr",
    )

    contract = activate_managed_ci(
        source_runner, config=source_config, pr_number=7, metadata=metadata()
    )
    assert contract is not None
    assert contract.audit_nonce == nonce

    issue_runner = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        rest_pr={"body": body},
    )
    issue_config = replace(source_config, pr_origin_flow="direct-pr")
    assert activate_managed_ci(
        issue_runner, config=issue_config, pr_number=7, metadata=metadata()
    ) is None
    assert any(
        command[:5] == ["gh", "api", "--method", "DELETE", f"repos/OWNER/REPO/issues/7/labels/{MANAGED_LABEL}"]
        for command, _ in issue_runner.commands
    )


@pytest.mark.parametrize(
    ("body", "surface", "schema", "accepted"),
    [
        (f"{UNPROTECTED_OVERRIDE_TRAILER} nonce=fresh", PR_BODY_SURFACE, "body", True),
        (
            f"{UNPROTECTED_OVERRIDE_TRAILER} nonce=fresh repo=OWNER/REPO base=main head=abc protection=voluntary",
            PR_COMMENT_SURFACE,
            "audit",
            True,
        ),
        (f"{UNPROTECTED_OVERRIDE_TRAILER} nonce=fresh repo=OWNER/REPO", PR_BODY_SURFACE, "body", False),
        (f"{UNPROTECTED_OVERRIDE_TRAILER} nonce=fresh\n{UNPROTECTED_OVERRIDE_TRAILER} nonce=other", PR_BODY_SURFACE, "body", False),
        (f"{UNPROTECTED_OVERRIDE_TRAILER} nonce=fresh repo=OWNER/REPO", PR_COMMENT_SURFACE, "audit", False),
    ],
)
def test_override_record_parser_enforces_surface_schema_agreement(body, surface, schema, accepted):
    if accepted:
        parsed = parse_managed_ci_override_record(body, surface=surface, schema=schema, required=True)
        assert parsed is not None and parsed.nonce == "fresh"
    else:
        with pytest.raises(AgentLoopError):
            parse_managed_ci_override_record(body, surface=surface, schema=schema, required=True)


def test_ordinary_resume_command_preserves_merge_and_watch_mode_without_managed_ci(tmp_path):
    command = render_managed_ci_resume_command(
        make_config(
            tmp_path,
            auto_merge=True,
            watch_pending_ci=True,
            managed_ci=True,
            managed_ci_trusted_actor="agent-loop",
            allow_unprotected_managed_ci=True,
        ),
        pr_number=42,
        managed_ci=False,
    )

    assert command == "agent-loop pr 42 --repo OWNER/REPO --base main --auto-merge"


def test_managed_resume_command_retains_managed_ci_configuration(tmp_path):
    command = render_managed_ci_resume_command(
        make_config(
            tmp_path,
            auto_merge=True,
            watch_pending_ci=True,
            managed_ci=True,
            managed_ci_trusted_actor="agent-loop",
            allow_unprotected_managed_ci=True,
        ),
        pr_number=42,
        managed_ci=True,
    )

    assert command == (
        "agent-loop pr 42 --repo OWNER/REPO --base main --auto-merge "
        "--managed-ci --managed-ci-trusted-actor agent-loop --allow-unprotected-managed-ci"
    )


def test_resume_command_omits_retired_ordinary_watch_opt_out(tmp_path):
    command = render_managed_ci_resume_command(
        make_config(tmp_path, auto_merge=True, watch_pending_ci=False),
        pr_number=42,
        managed_ci=False,
    )

    assert command.endswith("--auto-merge")


def test_issue_created_handoff_authenticates_override_before_any_remote_write(tmp_path):
    body = f"Fixes #643\n\n{UNPROTECTED_OVERRIDE_TRAILER} nonce=fresh"
    runner = V2ManagedRunner(
        rest_pr={"state": "open", "body": body},
    )
    metadata = PullRequestMetadata(
        number=7,
        repo="OWNER/REPO",
        title="Managed CI",
        head_branch="agent-loop/managed-643",
        base_branch="main",
        head_sha="abc123",
        url="https://github.com/OWNER/REPO/pull/7",
        body=body,
    )
    intent = managed_ci.ManagedCiCreationIntent(
        branch="agent-loop/managed-643",
        trusted_actor="agent-loop",
        protection_mode="voluntary",
        audit_nonce="fresh",
    )

    handoff = authenticate_issue_created_handoff(
        runner,
        config=make_config(
            tmp_path,
            managed_ci=True,
            base="main",
            managed_ci_trusted_actor="agent-loop",
            allow_unprotected_managed_ci=True,
        ),
        intent=intent,
        issue_number=643,
        pr_number=7,
        metadata=metadata,
    )

    assert handoff.head_sha == "abc123"
    assert handoff.override_nonce == "fresh"
    assert not any("--method" in command for command, _ in runner.commands)


def test_direct_resume_reconstructs_issue_override_as_provenance_before_writes(tmp_path):
    body = f"Fixes #643\n\n{UNPROTECTED_OVERRIDE_TRAILER} nonce=old"
    runner = V2ManagedRunner(
        rest_pr={"state": "open", "body": body},
        issue_events=[label_event()],
    )
    metadata = PullRequestMetadata(
        number=7,
        repo="OWNER/REPO",
        title="Managed CI",
        head_branch="agent-loop/managed-643",
        base_branch="main",
        head_sha="abc123",
        url="https://github.com/OWNER/REPO/pull/7",
        body=body,
    )
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_pr_mode=True,
        base="main",
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )

    handoff = recover_issue_created_handoff(
        runner, config=config, pr_number=7, metadata=metadata,
    )
    assert handoff is not None
    assert handoff.active_label_event_id == 101
    assert revalidate_issue_created_handoff(
        runner, config=config, handoff=handoff, metadata=metadata,
    ).override_nonce == "old"
    assert not any("--method" in command for command, _ in runner.commands)


@pytest.mark.parametrize(
    ("managed_ci_pr_mode", "issue_number"),
    [(False, 643), (True, None)],
)
def test_issue_and_pr_resume_authenticate_nonce_bearing_advanced_head(
    tmp_path, managed_ci_pr_mode, issue_number
):
    body = f"Fixes #643\n\n{UNPROTECTED_OVERRIDE_TRAILER} nonce=after-coder-round"
    advanced_sha = "coder-round-head"
    runner = V2ManagedRunner(
        rest_pr={
            "state": "open",
            "draft": True,
            "labels": [{"name": MANAGED_LABEL}],
            "body": body,
            "head": {
                "repo": {"full_name": "OWNER/REPO"},
                "sha": advanced_sha,
                "ref": "agent-loop/managed-643",
            },
        },
        issue_events=[label_event()],
    )
    config = make_config(
        tmp_path,
        managed_ci=managed_ci_pr_mode,
        managed_ci_pr_mode=managed_ci_pr_mode,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    metadata = replace(_ready_issue_metadata(body=body), head_sha=advanced_sha)

    handoff = recover_issue_created_handoff(
        runner,
        config=config,
        pr_number=7,
        metadata=metadata,
        issue_number=issue_number,
    )

    assert handoff is not None
    assert handoff.head_sha == advanced_sha
    assert handoff.lifecycle == "draft-labeled"
    assert not any("--method" in command for command, _ in runner.commands)


def _ready_issue_metadata(body="Fixes #643", base_branch="main"):
    return PullRequestMetadata(
        number=7,
        repo="OWNER/REPO",
        title="Managed CI",
        head_branch="agent-loop/managed-643",
        base_branch=base_branch,
        head_sha="abc123",
        url="https://github.com/OWNER/REPO/pull/7",
        body=body,
    )


def test_strict_ready_unlabeled_issue_resume_reauthenticates_then_restores_label(tmp_path):
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
    )
    runner = ManualQualificationRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        rest_pr={
            "state": "open",
            "draft": False,
            "labels": [],
            "body": "Fixes #643",
        },
        issue_events=[],
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT]},
    )

    handoff = recover_issue_created_handoff(
        runner,
        config=config,
        pr_number=7,
        metadata=_ready_issue_metadata(),
    )
    assert handoff is not None
    assert handoff.lifecycle == "ready-unlabeled-reentry"
    resume = AuthenticatedManagedResume(
        origin="issue-created",
        lifecycle=handoff.lifecycle,
        issue_created_handoff=handoff,
    )
    contract = activate_managed_ci(
        runner,
        config=config,
        pr_number=7,
        metadata=_ready_issue_metadata(),
        managed_resume=resume,
    )

    assert contract is not None
    assert contract.origin == "issue-created"
    assert contract.lifecycle == "ready-unlabeled-reentry"
    assert contract.intent_generation
    assert contract.audit_nonce is None
    commands = [command for command, _cwd in runner.commands]
    undo_index = next(index for index, command in enumerate(commands) if "--undo" in command)
    label_index = next(
        index
        for index, command in enumerate(commands)
        if command[:5] == [
            "gh", "api", "--method", "POST", "repos/OWNER/REPO/issues/7/labels"
        ]
    )
    assert undo_index < label_index
    assert not any(UNPROTECTED_OVERRIDE_TRAILER in " ".join(command) for command in commands)


def test_explicit_managed_issue_resume_reclaims_draft_unlabeled_state(tmp_path):
    body = f"Fixes #643\n\n{UNPROTECTED_OVERRIDE_TRAILER} nonce=old"
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    runner = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        rest_pr={
            "state": "open",
            "draft": True,
            "labels": [],
            "body": body,
            "head": {
                "repo": {"full_name": "OWNER/REPO"},
                "sha": "coder-round-head",
                "ref": "agent-loop/managed-643",
            },
        },
        issue_events=[label_event()],
        intent_comments=[{
            "id": 41,
            "user": {"login": "agent-loop", "id": 1},
            "body": str(format_issue_created_authorization_comment(
                ManagedCiIssueAuthorization(
                    kind="creation", repository="OWNER/REPO", issue_number=643,
                    pr_number=7, base_ref="main", head_sha="coder-round-head",
                    actor_login="agent-loop", actor_id=1, protection="voluntary",
                    waiver="allow-unprotected-managed-ci", nonce="old",
                    label_event_id=101,
                )
            )),
        }],
    )
    metadata = replace(_ready_issue_metadata(body=body), head_sha="coder-round-head")

    handoff = recover_issue_created_handoff(
        runner,
        config=config,
        pr_number=7,
        metadata=metadata,
        issue_number=643,
    )
    assert handoff is not None
    assert handoff.lifecycle == "draft-unlabeled-reentry"

    contract = activate_managed_ci(
        runner,
        config=config,
        pr_number=7,
        metadata=metadata,
        managed_resume=AuthenticatedManagedResume(
            origin="issue-created",
            lifecycle=handoff.lifecycle,
            issue_created_handoff=handoff,
        ),
    )

    assert contract is not None
    assert contract.lifecycle == "draft-unlabeled-reentry"
    assert any(
        command[:5] == [
            "gh", "api", "--method", "POST", "repos/OWNER/REPO/issues/7/labels"
        ]
        for command, _cwd in runner.commands
    )


def test_strict_draft_unlabeled_reentry_uses_historical_label_event(tmp_path):
    runner = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        rest_pr={
            "state": "open", "draft": True, "labels": [],
            "head": {
                "repo": {"full_name": "OWNER/REPO"},
                "sha": "coder-round-head",
                "ref": "agent-loop/managed-643",
            },
        },
        issue_events=[label_event()],
        intent_comments=[],
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT]},
    )
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci",
            "--managed-ci-trusted-actor", "agent-loop", "--allow-unprotected-managed-ci",
        ),
    )

    contract = activate_managed_ci(
        runner, config=config, pr_number=7, metadata=replace(
            _ready_issue_metadata(), head_sha="coder-round-head"
        ),
        managed_resume=AuthenticatedManagedResume(
            origin="issue-created", lifecycle="draft-unlabeled-reentry",
            issue_created_handoff=replace(
                _authorization_handoff(head="coder-round-head"),
                protection_mode="strict",
            ),
        ),
    )

    assert contract is not None
    assert contract.origin == "issue-created"
    assert contract.lifecycle == "draft-unlabeled-reentry"
    assert contract.protection_mode == "strict"
    assert contract.audit_nonce is None
    assert runner.labels_posted is True
    assert runner.dispatch_count == 0
    assert any(command[:5] == [
        "gh", "api", "--method", "POST", "repos/OWNER/REPO/issues/7/labels"
    ] for command, _cwd in runner.commands)


@pytest.mark.parametrize(
    "issue_events",
    [[], [{
        "id": 101,
        "event": "labeled",
        "label": {"name": MANAGED_LABEL},
        "actor": {"login": "someone-else", "id": 2},
    }]],
)
def test_strict_draft_unlabeled_reentry_requires_actor_owned_label_history(
    tmp_path, issue_events
):
    runner = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        rest_pr={
            "state": "open", "draft": True, "labels": [],
            "head": {
                "repo": {"full_name": "OWNER/REPO"},
                "sha": "coder-round-head",
                "ref": "agent-loop/managed-643",
            },
        },
        issue_events=issue_events,
        intent_comments=[],
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT]},
    )
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci",
            "--managed-ci-trusted-actor", "agent-loop", "--allow-unprotected-managed-ci",
        ),
    )

    with pytest.raises(
        AgentLoopError,
        match="no actor-owned historical managed-label event authenticates strict re-entry",
    ):
        activate_managed_ci(
            runner, config=config, pr_number=7, metadata=replace(
                _ready_issue_metadata(), head_sha="coder-round-head"
            ),
            managed_resume=AuthenticatedManagedResume(
                origin="issue-created", lifecycle="draft-unlabeled-reentry",
                issue_created_handoff=replace(
                    _authorization_handoff(head="coder-round-head"),
                    protection_mode="strict",
                ),
            ),
        )

    assert runner.labels_posted is False
    assert runner.dispatch_count == 0
    assert not any(command[:5] == [
        "gh", "api", "--method", "POST", "repos/OWNER/REPO/issues/7/labels"
    ] for command, _cwd in runner.commands)


def test_draft_unlabeled_stale_authorization_does_not_apply_label(tmp_path):
    stale = ManagedCiIssueAuthorization(
        kind="creation", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="older-head", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="old",
        label_event_id=101,
    )
    runner = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        rest_pr={
            "state": "open", "draft": True, "labels": [],
            "head": {
                "repo": {"full_name": "OWNER/REPO"},
                "sha": "new-head",
                "ref": "agent-loop/managed-643",
            },
        },
        issue_events=[label_event()],
        intent_comments=[{
            "id": 41,
            "user": {"login": "agent-loop", "id": 1},
            "body": str(format_issue_created_authorization_comment(stale)),
        }],
    )
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci",
            "--managed-ci-trusted-actor", "agent-loop", "--allow-unprotected-managed-ci",
        ),
    )

    with pytest.raises(AgentLoopError, match="no fully bound actor-owned issue-created authorization"):
        activate_managed_ci(
            runner, config=config, pr_number=7, metadata=replace(
                _ready_issue_metadata(), head_sha="new-head"
            ),
            managed_resume=AuthenticatedManagedResume(
                origin="issue-created", lifecycle="draft-unlabeled-reentry",
                issue_created_handoff=_authorization_handoff(head="new-head"),
            ),
        )

    assert runner.labels_posted is False
    assert runner.dispatch_count == 0
    assert not any(
        command[:5] == [
            "gh", "api", "--method", "POST", "repos/OWNER/REPO/issues/7/labels"
        ]
        for command, _cwd in runner.commands
    )


@pytest.mark.parametrize("origin", ["issue-created", "source-managed"])
def test_implicit_auto_merge_leaves_ready_unlabeled_resume_unchanged(tmp_path, origin):
    config = make_config(
        tmp_path,
        auto_merge=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
        managed_ci_pr_mode=True,
    )
    runner = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        rest_pr={"state": "open", "draft": False, "labels": [], "body": "Fixes #643"},
        issue_events=[],
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT]},
    )
    resume = AuthenticatedManagedResume(origin=origin, lifecycle="ready-unlabeled-reentry")

    with pytest.raises(AgentLoopError, match="explicit `--managed-ci`"):
        activate_managed_ci(
            runner,
            config=config,
            pr_number=7,
            metadata=_ready_issue_metadata(),
            managed_resume=resume,
        )

    assert not any(
        command[:3] == ["gh", "pr", "ready"]
        or ("--method" in command and "POST" in command)
        or ("--method" in command and "PATCH" in command)
        for command, _cwd in runner.commands
    )


def test_repository_default_base_mismatch_prints_explicit_live_base_retry(tmp_path):
    runner = V2ManagedRunner(base_ref="release")
    config = make_config(
        tmp_path,
        base=None,
        invocation_argv=("agent-loop", "issue", "643", "--auto-merge"),
    )
    resolved = resolve_base_branch(
        config, runner, pr_metadata=_ready_issue_metadata(base_branch="release")
    )
    assert resolved.base == "release"
    assert resolved.base_provenance == "pr-metadata"

    inherited = resolve_base_branch(make_config(tmp_path, base=None), V2ManagedRunner())
    assert inherited.base == "main"
    assert inherited.base_provenance == "repository-default"
    with pytest.raises(AgentLoopError, match=r"repository-default.*--base release"):
        resolve_base_branch(
            inherited,
            runner,
            pr_metadata=_ready_issue_metadata(base_branch="release"),
        )


def test_recovery_renderer_retargets_both_directions_with_parser_valid_argv(tmp_path):
    parser = build_parser()
    issue_to_pr = make_config(
        tmp_path,
        invocation_argv=(
            "agent-loop", "issue", "643", "--plan-first", "--plan-execution-mode",
            "implement-one-shot", "--implementation-coder", "codex", "--split-stage", "700",
            "--reviewer", "codex", "--reviewer=gemini", "--codex-arg=--literal=$HOME;echo",
        ),
    )
    rendered = _render_recovery_command(
        issue_to_pr, target="pr", identifier=7, managed_ci=True,
    )
    args = parser.parse_args(shlex.split(rendered)[1:])
    assert args.command == "pr"
    assert args.pr_number == 7
    assert not hasattr(args, "plan_first")
    assert args.reviewer == ["codex", "gemini"]
    assert args.codex_arg == ["--literal=$HOME;echo"]

    pr_to_issue = make_config(
        tmp_path,
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci-adopt-existing-pr",
            "--managed-ci", "--managed-ci-trusted-actor=agent-loop",
            "--reviewer", "codex",
        ),
    )
    rendered = _render_recovery_command(
        pr_to_issue, target="issue", identifier=643, managed_ci=True,
    )
    args = parser.parse_args(shlex.split(rendered)[1:])
    assert args.command == "issue"
    assert args.issue_number == 643
    assert not hasattr(args, "managed_ci_adopt_existing_pr")
    assert args.managed_ci_trusted_actor == "agent-loop"

    managed_pr_to_issue = make_config(
        tmp_path,
        invocation_argv=(
            "agent-loop", "managed-pr", "--head", "feature/ref",
            "--title=Managed PR", "--body-file", "/tmp/body.md",
            "--reviewer", "codex",
        ),
    )
    rendered = _render_recovery_command(
        managed_pr_to_issue, target="issue", identifier=643, managed_ci=False,
    )
    args = parser.parse_args(shlex.split(rendered)[1:])
    assert args.command == "issue"
    assert args.issue_number == 643


def test_fresh_authorization_recovery_renderer_requires_explicit_issue_scope(tmp_path):
    parser = build_parser()
    config = make_config(
        tmp_path,
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci", "--managed-ci-trusted-actor", "agent-loop",
            "--allow-unprotected-managed-ci",
        ),
    )
    rendered = render_managed_ci_resume_command(
        config,
        pr_number=7,
        issue_number=643,
        managed_ci=True,
        fresh_authorization=True,
        fresh_issue_number=643,
    )
    args = parser.parse_args(shlex.split(rendered)[1:])
    assert args.command == "pr"
    assert args.managed_ci_fresh_authorization is True
    assert args.managed_ci_issue == 643
    assert args.allow_unprotected_managed_ci is True


def test_public_pr_fresh_recovery_renderer_uses_recovered_issue_scope(tmp_path):
    parser = build_parser()
    config = make_config(
        tmp_path,
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci",
            "--managed-ci-trusted-actor", "agent-loop",
            "--allow-unprotected-managed-ci",
        ),
    )

    rendered = render_managed_ci_resume_command(
        config,
        pr_number=7,
        issue_number=643,
        managed_ci=True,
        fresh_authorization=True,
    )
    args = parser.parse_args(shlex.split(rendered)[1:])

    assert args.command == "pr"
    assert args.pr_number == 7
    assert args.managed_ci_fresh_authorization is True
    assert args.managed_ci_issue == 643


@pytest.mark.parametrize(
    ("kind", "changes"),
    [
        ("creation", {"predecessor_head": "old-head"}),
        ("continuity", {"predecessor_head": None, "predecessor_comment_id": None}),
    ],
)
def test_issue_authorization_parser_enforces_kind_specific_schema(tmp_path, kind, changes):
    record = ManagedCiIssueAuthorization(
        kind=kind,
        repository="OWNER/REPO",
        issue_number=643,
        pr_number=7,
        base_ref="main",
        head_sha="abc123",
        actor_login="agent-loop",
        actor_id=1,
        protection="voluntary",
        waiver="allow-unprotected-managed-ci",
        nonce="nonce",
        label_event_id=101,
        predecessor_head="old-head" if kind == "continuity" else None,
        predecessor_comment_id=41 if kind == "continuity" else None,
        round_comment_ids=(88,) if kind == "continuity" else (),
    )
    payload = record.to_payload()
    payload.update(changes)
    encoded = managed_ci._encode_issue_authorization_payload(payload)
    body = f"<!-- {managed_ci.ISSUE_AUTHORIZATION_MARKER}: {encoded} -->"

    with pytest.raises(AgentLoopError, match="continuity|continuity fields"):
        parse_issue_created_authorization_comment(body)


@pytest.mark.parametrize(
    "changes",
    [
        {"round_comment_ids": [88]},
        {"predecessor_head": "old-head"},
        {"predecessor_comment_id": 41},
        {"protection": "strict"},
        {"waiver": "not-the-explicit-waiver"},
    ],
)
def test_issue_authorization_parser_rejects_noncanonical_fresh_schema(changes):
    record = ManagedCiIssueAuthorization(
        kind="fresh",
        repository="OWNER/REPO",
        issue_number=643,
        pr_number=7,
        base_ref="main",
        head_sha="abc123",
        actor_login="agent-loop",
        actor_id=1,
        protection="voluntary",
        waiver="allow-unprotected-managed-ci",
        nonce="nonce",
        label_event_id=101,
    )
    payload = record.to_payload()
    payload.update(changes)
    encoded = managed_ci._encode_issue_authorization_payload(payload)
    body = f"<!-- {managed_ci.ISSUE_AUTHORIZATION_MARKER}: {encoded} -->"

    with pytest.raises(AgentLoopError, match="fresh authorization|protection or waiver"):
        parse_issue_created_authorization_comment(body)


def test_activation_rejects_malformed_fresh_authorization_before_label_or_dispatch(tmp_path):
    record = ManagedCiIssueAuthorization(
        kind="fresh",
        repository="OWNER/REPO",
        issue_number=643,
        pr_number=7,
        base_ref="main",
        head_sha="abc123",
        actor_login="agent-loop",
        actor_id=1,
        protection="voluntary",
        waiver="allow-unprotected-managed-ci",
        nonce="fresh",
        label_event_id=101,
        predecessor_head="old-head",
    )
    encoded = managed_ci._encode_issue_authorization_payload(record.to_payload())
    malformed_body = f"<!-- {managed_ci.ISSUE_AUTHORIZATION_MARKER}: {encoded} -->"
    runner = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        rest_pr={"state": "open", "draft": True, "labels": []},
        issue_events=[label_event()],
        intent_comments=[{
            "id": 41,
            "user": {"login": "agent-loop", "id": 1},
            "body": malformed_body,
        }],
    )
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci",
            "--managed-ci-trusted-actor", "agent-loop",
            "--allow-unprotected-managed-ci",
        ),
    )
    handoff = replace(
        _authorization_handoff(),
        lifecycle="draft-unlabeled-reentry",
        opening_override_nonce="nonce-643",
        authorization_kind="fresh",
        authorization_comment_id=41,
        override_nonce="fresh",
    )

    assert _find_resume_audit(
        runner,
        config=config,
        pr_number=7,
        actor_login="agent-loop",
        actor_id=1,
        base_ref="main",
        issue_number=643,
        live_head="abc123",
        require_actor_owned_label_event=True,
    ) is None

    with pytest.raises(AgentLoopError, match="--managed-ci-fresh"):
        activate_managed_ci(
            runner,
            config=config,
            pr_number=7,
            metadata=replace(metadata(), head_branch="agent-loop/managed-643"),
            managed_resume=AuthenticatedManagedResume(
                origin="issue-created",
                lifecycle="draft-unlabeled-reentry",
                issue_created_handoff=handoff,
            ),
        )

    assert runner.labels_posted is False
    assert runner.dispatch_count == 0
    assert not any(
        command[:5] == [
            "gh", "api", "--method", "POST", "repos/OWNER/REPO/issues/7/labels"
        ]
        for command, _cwd in runner.commands
    )


def test_recovery_renderer_finds_issue_identifier_after_options_and_consumes_stdin_value(tmp_path):
    parser = build_parser()
    config = make_config(
        tmp_path,
        invocation_argv=(
            "agent-loop", "issue", "--repo", "OWNER/REPO", "--gh-cmd", "gh", "643",
            "--managed-ci", "--managed-ci-trusted-actor", "agent-loop",
        ),
    )
    rendered = render_managed_ci_resume_command(config, pr_number=7, managed_ci=True)
    args = parser.parse_args(shlex.split(rendered)[1:])
    assert args.command == "issue"
    assert args.issue_number == 643
    assert args.repo == "OWNER/REPO"
    assert args.gh_cmd == "gh"

    managed_pr = make_config(
        tmp_path,
        invocation_argv=(
            "agent-loop", "managed-pr", "--head", "feature/ref",
            "--title", "Managed PR", "--body-file", "-", "--reviewer", "codex",
        ),
    )
    rendered = _render_recovery_command(
        managed_pr, target="issue", identifier=643, managed_ci=False,
    )
    args = parser.parse_args(shlex.split(rendered)[1:])
    assert args.command == "issue"
    assert args.issue_number == 643


def test_recovery_value_option_table_covers_all_recovery_subparsers():
    parser = build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")

    for command in ("issue", "pr", "managed-pr"):
        subparser = subparsers.choices[command]
        value_options = {
            option
            for action in subparser._actions
            if action.nargs != 0
            for option in action.option_strings
            if option.startswith("--")
        }
        assert value_options <= managed_ci._RECOVERY_VALUE_OPTIONS, (
            f"{command} value-taking recovery options are not classified: "
            f"{sorted(value_options - managed_ci._RECOVERY_VALUE_OPTIONS)}"
        )

    assert "--ci-check-name" not in managed_ci._RECOVERY_VALUE_OPTIONS
    assert {
        "--ci-timeout-seconds",
        "--ci-poll-interval-seconds",
        "--ci-startup-timeout-seconds",
    } <= managed_ci._RECOVERY_VALUE_OPTIONS


def test_issue_created_tuple_actor_refusal_includes_trusted_actor_remediation(tmp_path):
    body = f"Fixes #643\n\n{UNPROTECTED_OVERRIDE_TRAILER} nonce=fresh"
    config = make_config(
        tmp_path,
        managed_ci=True,
        invocation_argv=("agent-loop", "issue", "643", "--repo", "OWNER/REPO"),
    )
    runner = V2ManagedRunner(rest_pr={"state": "open", "body": body})
    metadata = replace(_ready_issue_metadata(body=body), head_sha="abc123")
    intent = managed_ci.ManagedCiCreationIntent(
        branch="agent-loop/managed-643",
        trusted_actor="agent-loop",
        protection_mode="voluntary",
        audit_nonce="fresh",
    )

    with pytest.raises(AgentLoopError, match=r"--managed-ci-trusted-actor agent-loop"):
        authenticate_issue_created_handoff(
            runner,
            config=config,
            intent=intent,
            issue_number=643,
            pr_number=7,
            metadata=metadata,
        )


def test_issue_created_tuple_actor_refusal_replaces_existing_trusted_actor(tmp_path):
    body = f"Fixes #643\n\n{UNPROTECTED_OVERRIDE_TRAILER} nonce=fresh"
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_trusted_actor="operator-actor",
        invocation_argv=(
            "agent-loop", "issue", "643", "--repo", "OWNER/REPO", "--managed-ci",
            "--managed-ci-trusted-actor", "operator-actor",
        ),
    )
    runner = V2ManagedRunner(rest_pr={"state": "open", "body": body})
    metadata = replace(_ready_issue_metadata(body=body), head_sha="abc123")
    intent = managed_ci.ManagedCiCreationIntent(
        branch="agent-loop/managed-643",
        trusted_actor="agent-loop",
        protection_mode="voluntary",
        audit_nonce="fresh",
    )

    with pytest.raises(AgentLoopError) as error:
        authenticate_issue_created_handoff(
            runner,
            config=config,
            intent=intent,
            issue_number=643,
            pr_number=7,
            metadata=metadata,
        )

    message = str(error.value)
    assert "gh login=`agent-loop`" in message
    assert "--managed-ci-trusted-actor=`operator-actor`" in message
    assert "AGENT_LOOP_MANAGED_ACTOR=`agent-loop`" in message
    remediation = message.split("`", 7)[7]
    assert remediation.count("--managed-ci-trusted-actor") == 1
    assert "--managed-ci-trusted-actor agent-loop" in remediation
    assert "operator-actor" not in remediation


def test_issue_created_tuple_mismatch_includes_shell_quoted_resume_command(tmp_path):
    body = f"Fixes #643\n\n{UNPROTECTED_OVERRIDE_TRAILER} nonce=fresh"
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_trusted_actor="agent-loop",
        invocation_argv=(
            "agent-loop", "issue", "643", "--repo", "OWNER/REPO", "--gh-cmd", "gh",
        ),
    )
    runner = V2ManagedRunner(
        rest_pr={"state": "open", "body": body, "head": {
            "repo": {"full_name": "OWNER/REPO"},
            "sha": "changed-head",
            "ref": "agent-loop/managed-643",
        }},
    )
    metadata = replace(_ready_issue_metadata(body=body), head_sha="abc123")
    intent = managed_ci.ManagedCiCreationIntent(
        branch="agent-loop/managed-643",
        trusted_actor="agent-loop",
        protection_mode="voluntary",
        audit_nonce="fresh",
    )

    with pytest.raises(AgentLoopError) as error:
        authenticate_issue_created_handoff(
            runner,
            config=config,
            intent=intent,
            issue_number=643,
            pr_number=7,
            metadata=metadata,
        )

    message = str(error.value)
    assert "opening tuple is missing or changed" in message
    assert "agent-loop issue 643 --repo OWNER/REPO --gh-cmd gh" in message
    assert "--managed-ci-trusted-actor agent-loop" in message


def test_ci_timeout_renderer_preserves_explicit_managed_options(tmp_path):
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
        auto_merge=True,
        invocation_argv=(
            "agent-loop", "issue", "643", "--auto-merge", "--managed-ci",
            "--managed-ci-trusted-actor", "agent-loop", "--allow-unprotected-managed-ci",
        ),
    )

    command = _render_ci_rerun_command(config, pr_number=7)

    assert "--managed-ci" in command
    assert "--managed-ci-trusted-actor agent-loop" in command
    assert "--allow-unprotected-managed-ci" in command


def _resume_audit(*, head="old-head", repo="OWNER/REPO", base="main"):
    return {
        "id": 41,
        "user": {"login": "agent-loop", "id": 1},
        "body": (
            f"{UNPROTECTED_OVERRIDE_TRAILER} nonce=old-nonce repo={repo} "
            f"base={base} head={head} protection=voluntary"
        ),
    }


def test_pr_mode_resumes_only_from_durable_issue_authorization_and_mints_fresh_audit(tmp_path):
    config = make_config(
        tmp_path,
        auto_merge=True,
        managed_ci=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    resume_metadata = replace(metadata(), head_branch="agent-loop/managed-643")
    runner = V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        rest_pr={"state": "open", "draft": True, "body": resume_metadata.body},
        issue_events=[label_event()],
        intent_comments=[{
            "id": 41,
            "user": {"login": "agent-loop", "id": 1},
            "body": str(format_issue_created_authorization_comment(
                ManagedCiIssueAuthorization(
                    kind="creation", repository="OWNER/REPO", issue_number=643,
                    pr_number=7, base_ref="main", head_sha="abc123",
                    actor_login="agent-loop", actor_id=1, protection="voluntary",
                    waiver="allow-unprotected-managed-ci", nonce="nonce-643",
                    label_event_id=101,
                )
            )),
        }],
    )

    handoff = recover_issue_created_handoff(
        runner, config=config, pr_number=7, metadata=resume_metadata, issue_number=643
    )
    assert handoff is not None
    contract = activate_managed_ci(
        runner,
        config=config,
        pr_number=7,
        metadata=resume_metadata,
        managed_resume=AuthenticatedManagedResume(
            origin="issue-created",
            lifecycle=handoff.lifecycle,
            issue_created_handoff=handoff,
        ),
    )

    assert contract is not None
    assert contract.activation_path == "managed"
    assert contract.audit_nonce and contract.audit_nonce != "old-nonce"
    # The verified audit write adopts the identity GitHub stored for it.
    assert contract.audit_comment_id in runner.audit_comments
    assert contract.intent_generation
    assert contract.ordinary_recovery_capable is True
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
    contract = valid_v2_contract(intent_generation="fresh-generation")
    runner = valid_v2_runner(
        intent_comments=[historical],
        workflow_runs=[valid_v2_run(run_id=100)],
    )

    _ensure_v2_intent(runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract)

    assert contract.nonce != V2_NONCE
    assert contract.attached_run_id is None
    assert any("issues/7/comments" in " ".join(cmd) and "POST" in cmd for cmd, _ in runner.commands)


def test_intent_history_malformed_page_fails_closed_instead_of_minting_nonce(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    contract = v2_contract(intent_generation="fresh-generation")
    runner = V2ManagedRunner(intent_comments=[{"id": 17}, "malformed-entry"])

    # The workflow fails every nonce on a malformed entry, so a fresh nonce
    # is never minted while one is present (#1043).
    with pytest.raises(managed_ci.ManagedCiIntentLedgerError, match="malformed comment author"):
        _ensure_v2_intent(
            runner, config=config, pr_number=7, expected_head_sha="abc123", contract=contract,
        )
    assert not any("issues/7/comments" in " ".join(cmd) and "POST" in cmd for cmd, _ in runner.commands)


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
        lambda *args, **kwargs: SimpleNamespace(
            status="passed",
            checks=checks(
                passing=(PullRequestCheck("test", "check_run", "success"),),
                required=("test",),
            ),
            head_sha="abc123",
        ),
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


def test_ordinary_recovery_forbidden_branch_protection_never_reports_success(monkeypatch, tmp_path):
    config = make_config(tmp_path, auto_merge=True, ci_timeout_seconds=1, ci_poll_interval_seconds=1)
    capability = OrdinaryRecoveryCapability(
        pr_number=7, repository="OWNER/REPO", base_ref="main", expected_head_sha="abc123",
        released_label_event_id=101, released_at=100, prior_run_ids=frozenset(),
    )
    passing = checks(
        passing=(PullRequestCheck("test", "check_run", "success"),),
        required=("test",),
        protection="forbidden",
    )
    monkeypatch.setattr(managed_ci, "get_pr_head_sha", lambda *args, **kwargs: "abc123")
    monkeypatch.setattr(
        managed_ci,
        "get_pr_mergeability",
        lambda *args, **kwargs: type("M", (), {"state": "mergeable"})(),
    )
    monkeypatch.setattr(
        managed_ci,
        "_workflow_runs_payload",
        lambda *args, **kwargs: [{"id": 3, "status": "completed", "conclusion": "success"}],
    )
    monkeypatch.setattr(managed_ci, "get_pr_checks", lambda *args, **kwargs: passing)

    outcome = wait_for_ordinary_recovery(
        runner=FakeRunner(), config=config, capability=capability, metadata=metadata(),
    )

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


# Workflow-valid identities for seeded intent ledgers.  Rediscovery adopts
# only a record the base workflow would accept (#1043), so a seeded record
# carries 40-hex SHAs, a 32-character nonce, and the full key set.
V2_HEAD = "abc123" + "0" * 34
V2_REVISION = "ba5e" + "0" * 36
V2_NONCE = "nonce-1" + "x" * 25
V2_GENERATION = "generation-1"


def v2_intent_payload(
    *, nonce=V2_NONCE, run_id=None, run_attempt=None, state=None,
    terminal_run_id=None, terminal_run_attempt=None,
    terminal_attempts=None, terminal_outcome=None, created_at=1,
    generation=V2_GENERATION, expected_head_sha=V2_HEAD,
):
    if state is None:
        state = "attached" if run_id is not None else "dispatch-requested"
    return {
        "version": 2,
        "repository": "OWNER/REPO",
        "pr": 7,
        "expected_head_sha": expected_head_sha,
        "base_ref": "main",
        "workflow_revision": V2_REVISION,
        "generation": generation,
        "nonce": nonce,
        "created_at": created_at,
        "state": state,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "terminal_run_id": terminal_run_id,
        "terminal_run_attempt": terminal_run_attempt,
        "terminal_outcome": terminal_outcome,
        "terminal_attempts": [
            {"run_id": item_id, "run_attempt": attempt}
            for item_id, attempt in (terminal_attempts or ())
        ],
    }


def v2_intent_comment(*, comment_id=17, suffix="", **fields):
    payload = v2_intent_payload(**fields)
    return {
        "id": comment_id,
        "user": {"login": "agent-loop", "id": 1},
        "body": f"<!-- AGENT_MANAGED_CI_INTENT_V2 {json.dumps(payload)} -->{suffix}",
    }


def valid_v2_contract(**overrides):
    """A same-generation contract whose ledger the workflow would accept."""
    fields = {
        "workflow_revision": V2_REVISION,
        "nonce": V2_NONCE,
        "intent_generation": V2_GENERATION,
    }
    fields.update(overrides)
    return v2_contract(**fields)


def valid_v2_runner(**kwargs):
    kwargs.setdefault("base_sha", V2_REVISION)
    kwargs.setdefault("pr_payload", {"headRefOid": V2_HEAD})
    return V2ManagedRunner(**kwargs)


def valid_v2_run(**kwargs):
    kwargs.setdefault("name", f"managed-ci-v2 nonce={V2_NONCE}")
    run = v2_run(**kwargs)
    run["head_sha"] = V2_REVISION
    return run


def checks(
    *, pending=(), passing=(), failing=(), required=(FINAL_CONTEXT,), missing=(),
    protection="configured",
):
    return PullRequestChecks(
        state="failing" if failing else "pending" if pending else "passing",
        required_checks=required,
        passing=passing,
        pending=pending,
        failing=failing,
        missing_required=missing,
        branch_protection_status=protection,
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
    assert result is None


def test_v2_completed_run_without_attempt_waits_without_terminal_publication(tmp_path):
    config = make_config(
        tmp_path, auto_merge=True, ci_timeout_seconds=2, ci_poll_interval_seconds=1
    )
    runner = V2ManagedRunner(
        workflow_runs=[v2_run(attempt=None, status="completed", conclusion="cancelled")],
        pr_payload={"headRefOid": "abc123", "mergeable": "MERGEABLE", "mergeStateStatus": "CLEAN"},
        pr_status_payload={"statuses": []},
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT], "checks": []},
    )
    contract = v2_contract(
        attached_run_id=100, run_attempt=None, intent_comment_id=17,
        pr_number=7, expected_head_sha="abc123",
    )

    outcome = wait_for_final_qualification(
        runner, config=config, pr_number=7, metadata=metadata(), contract=contract
    )

    assert outcome.status == "timeout"
    assert contract.terminal_run_id is None
    assert contract.terminal_run_attempt is None
    assert contract.terminal_attempts == ()
    assert contract.terminal_outcome is None
    assert not any(
        "/issues/comments/17" in " ".join(command)
        for command, _cwd in runner.commands
    )


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
    assert contract.intent_state == "completed"
    assert contract.terminal_outcome == "no-status"
    assert any(
        '"state":"completed"' in " ".join(command)
        and '"terminal_outcome":"no-status"' in " ".join(command)
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
    runner = valid_v2_runner(
        workflow_runs=[
            valid_v2_run(run_id=100, attempt=1, status="completed", conclusion="cancelled"),
            valid_v2_run(run_id=101, attempt=1, status="in_progress", conclusion=None),
        ],
        intent_comments=[v2_intent_comment(
            run_id=100, run_attempt=1, state="completed", terminal_outcome="no-status",
            terminal_run_id=100, terminal_run_attempt=1, terminal_attempts=((100, 1),),
        )],
    )
    contract = valid_v2_contract()

    _dispatch_v2_qualification(
        runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract
    )

    assert (contract.attached_run_id, contract.run_attempt) == (101, 1)
    assert contract.intent_state == "attached"
    assert not any("/dispatches" in " ".join(cmd) for cmd, _cwd in runner.commands)


def test_v2_terminal_ledger_clears_old_attachment_before_fresh_dispatch(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = valid_v2_runner(
        workflow_runs=[valid_v2_run(run_id=100, attempt=1, status="completed", conclusion="cancelled")],
        intent_comments=[v2_intent_comment(
            run_id=100, run_attempt=1, state="completed", terminal_outcome="no-status",
            terminal_run_id=100, terminal_run_attempt=1, terminal_attempts=((100, 1),),
        )],
    )
    contract = valid_v2_contract()

    _dispatch_v2_qualification(
        runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract
    )

    assert contract.attached_run_id is None
    assert contract.run_attempt is None
    assert any("/dispatches" in " ".join(cmd) for cmd, _cwd in runner.commands)


def test_v2_terminal_ledger_excludes_all_prior_cancelled_attempts(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = valid_v2_runner(
        workflow_runs=[
            valid_v2_run(run_id=100, attempt=2, status="completed", conclusion="cancelled"),
            valid_v2_run(run_id=100, attempt=1, status="completed", conclusion="cancelled"),
            valid_v2_run(run_id=101, attempt=1, status="in_progress", conclusion=None),
        ],
        intent_comments=[v2_intent_comment(
            run_id=100, run_attempt=2, state="completed", terminal_outcome="no-status",
            terminal_run_id=100, terminal_run_attempt=2,
            terminal_attempts=((100, 1), (100, 2)),
        )],
    )
    contract = valid_v2_contract()

    _dispatch_v2_qualification(
        runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract
    )

    assert (contract.attached_run_id, contract.run_attempt) == (101, 1)
    assert not any("/dispatches" in " ".join(cmd) for cmd, _cwd in runner.commands)


def test_v2_later_legitimate_rerun_attempt_is_accepted_after_terminal_stop(tmp_path):
    config = make_config(tmp_path, auto_merge=True, ci_timeout_seconds=1, ci_poll_interval_seconds=1)
    runner = valid_v2_runner(
        workflow_runs=[
            valid_v2_run(run_id=100, attempt=2, status="completed", conclusion="success"),
            valid_v2_run(run_id=100, attempt=1, status="completed", conclusion="timed_out"),
        ],
        intent_comments=[v2_intent_comment(
            run_id=100, run_attempt=1, state="completed", terminal_outcome="no-status",
            terminal_run_id=100, terminal_run_attempt=1, terminal_attempts=((100, 1),),
        )],
        pr_status_payload={"statuses": [{
            "context": FINAL_CONTEXT,
            "state": "success",
            "description": f"nonce={V2_NONCE};run_id=100;attempt=2",
            "target_url": "https://github.com/OWNER/REPO/actions/runs/100",
            "creator": {"login": "github-actions[bot]", "id": 41898282},
        }]},
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT], "checks": []},
    )
    contract = valid_v2_contract()

    _dispatch_v2_qualification(
        runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract
    )
    outcome = wait_for_final_qualification(
        runner, config=config, pr_number=7,
        metadata=replace(metadata(), head_sha=V2_HEAD), contract=contract,
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



@pytest.mark.parametrize(
    ("workflow", "capable"),
    [
        (V2_WORKFLOW, False),
        (V2_WORKFLOW + "# AGENT_LOOP_MANAGED_CI_VISIBLE_INTENT_V1\n", True),
    ],
)
def test_v2_activation_derives_visible_intent_capability_from_base_workflow(
    tmp_path, workflow, capable
):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = V2ManagedRunner(workflow=workflow, pr_payload={"headRefOid": "abc123"})

    contract = activate_managed_ci(runner, config=config, pr_number=7, metadata=metadata())

    assert contract is not None
    assert contract.visible_intent_capable is capable


def _visible_intent_pr(expected_head):
    return {
        "state": "open", "draft": True, "number": 7,
        "base": {"ref": "main"}, "head": {
            "sha": expected_head, "ref": "agent-loop/managed-7",
            "repo": {"full_name": "OWNER/REPO"},
        }, "user": {"login": "agent-loop", "id": 7},
        "labels": [{"name": MANAGED_LABEL}],
    }


def test_v2_intent_body_is_bare_for_workflow_without_visible_capability():
    revision, expected_head = "a" * 40, "b" * 40
    contract = v2_contract(
        workflow_revision=revision, repository="OWNER/REPO", nonce="n" * 32, created_at=1,
        intent_generation="generation-935",
        attached_run_id=100, run_attempt=1,
    )

    body = str(_intent_body(contract, pr_number=7, expected_head_sha=expected_head, state="attached"))

    assert body.startswith("<!-- AGENT_MANAGED_CI_INTENT_V2 ")
    pages = [[{"user": {"login": "agent-loop", "id": 7}, "body": body}]]
    # Older pinned consumers, which fullmatch the bare record, keep working.
    for router in (historical_router, current_router):
        router.validate(
            _visible_intent_pr(expected_head), pages, "OWNER/REPO", "7", expected_head,
            "n" * 32, "agent-loop", revision,
        )
    local_router.validate(
        _visible_intent_pr(expected_head), pages, "OWNER/REPO", "7", expected_head,
        "n" * 32, "agent-loop", revision, 7,
    )


def test_v2_intent_body_leads_with_fixed_visible_line_for_capable_workflow():
    revision, expected_head = "a" * 40, "b" * 40
    contract = v2_contract(
        workflow_revision=revision, repository="OWNER/REPO", nonce="n" * 32, created_at=1,
        intent_generation="generation-935",
        attached_run_id=100, run_attempt=1, visible_intent_capable=True,
    )

    body = str(_intent_body(contract, pr_number=7, expected_head_sha=expected_head, state="attached"))

    visible, record = body.split("\n\n", 1)
    assert visible == f"Managed CI authorization for exact head {expected_head}."
    assert record.startswith("<!-- AGENT_MANAGED_CI_INTENT_V2 ")
    pages = [[{"user": {"login": "agent-loop", "id": 7}, "body": body}]]
    validated = local_router.validate(
        _visible_intent_pr(expected_head), pages, "OWNER/REPO", "7", expected_head,
        "n" * 32, "agent-loop", revision, 7,
    )
    assert validated["expected_head_sha"] == expected_head


def test_v2_visible_intent_body_round_trips_through_intent_rediscovery(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = valid_v2_runner()
    contract = valid_v2_contract(visible_intent_capable=True)

    _ensure_v2_intent(runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract)
    nonce = contract.nonce
    _patch_intent(runner, config=config, contract=contract, state="dispatch-requested")

    assert [snapshot["state"] for snapshot in runner.intent_snapshots] == ["prepared", "dispatch-requested"]
    resumed = valid_v2_contract(visible_intent_capable=True, nonce=None)
    _ensure_v2_intent(runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=resumed)
    assert resumed.nonce == nonce
    assert resumed.intent_state == "dispatch-requested"


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
            recovery_capable=True,
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
    runner = valid_v2_runner(intent_comments=[comment])
    contract = valid_v2_contract()

    _ensure_v2_intent(
        runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract
    )

    assert (contract.intent_comment_id, contract.nonce, contract.attached_run_id) == (17, V2_NONCE, 100)
    competing = dict(comment)
    competing["id"] = 18
    competing["body"] = v2_intent_comment(nonce="nonce-2" + "y" * 25)["body"]
    runner = valid_v2_runner(intent_comments=[comment, competing])
    with pytest.raises(AgentLoopError, match="Competing managed-CI v2 intent") as raised:
        _ensure_v2_intent(
            runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=valid_v2_contract()
        )
    # Differing nonces keep the existing, recoverable competing-intents route.
    assert not isinstance(raised.value, managed_ci.ManagedCiIntentLedgerError)


def test_v2_fresh_generation_resets_attachment_terminal_history_and_early_fields(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = V2ManagedRunner()
    contract = v2_contract(
        intent_comment_id=99,
        nonce="old-nonce",
        created_at=1,
        attached_run_id=100,
        run_attempt=2,
        intent_state="completed",
        terminal_run_id=100,
        terminal_run_attempt=1,
        terminal_attempts=((100, 1),),
        terminal_outcome="no-status",
        intent_generation="fresh-generation",
    )

    _ensure_v2_intent(
        runner, config=config, pr_number=7, expected_head_sha="abc123", contract=contract
    )

    prepared = runner.intent_snapshots[-1]
    assert prepared["state"] == "prepared"
    assert prepared["run_id"] is None
    assert prepared["run_attempt"] is None
    assert prepared["terminal_run_id"] is None
    assert prepared["terminal_run_attempt"] is None
    assert prepared["terminal_attempts"] == []
    assert prepared["terminal_outcome"] is None
    assert contract.attached_run_id is None
    assert contract.terminal_attempts == ()

    _patch_intent(runner, config=config, contract=contract, state="dispatch-requested")
    requested = runner.intent_snapshots[-1]
    assert requested["state"] == "dispatch-requested"
    assert requested["run_id"] is None
    assert requested["run_attempt"] is None


def test_v2_same_nonce_non_excluded_attachment_survives_discovery_miss_without_redispatch(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = valid_v2_runner(intent_comments=[v2_intent_comment(
        run_id=100, run_attempt=1, state="attached"
    )])
    contract = valid_v2_contract()

    _dispatch_v2_qualification(
        runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract
    )

    assert (contract.attached_run_id, contract.run_attempt) == (100, 1)
    assert runner.dispatch_count == 0
    assert runner.intent_snapshots == []


def test_v2_excluded_attachment_transitions_to_dispatch_requested_before_replacement(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = valid_v2_runner(
        workflow_runs=[valid_v2_run(run_id=101, status="in_progress", conclusion=None)],
        intent_comments=[v2_intent_comment(
            run_id=100, run_attempt=1, state="completed",
            terminal_run_id=100, terminal_run_attempt=1,
            terminal_attempts=((100, 1),), terminal_outcome="no-status",
        )],
    )
    contract = valid_v2_contract()

    _dispatch_v2_qualification(
        runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract
    )

    assert (contract.attached_run_id, contract.run_attempt) == (101, 1)
    assert runner.dispatch_count == 0
    assert [snapshot["state"] for snapshot in runner.intent_snapshots] == [
        "dispatch-requested", "attached"
    ]
    assert runner.intent_snapshots[0]["run_id"] is None


def test_v2_emitted_lifecycle_records_are_accepted_by_pinned_consumers(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    revision = "a" * 40
    expected_head = "b" * 40
    runner = V2ManagedRunner()
    contract = v2_contract(
        workflow_revision=revision,
        intent_generation="auto-merge-generation",
    )

    _ensure_v2_intent(
        runner, config=config, pr_number=7, expected_head_sha=expected_head, contract=contract
    )
    _patch_intent(runner, config=config, contract=contract, state="dispatch-requested")
    contract.attached_run_id, contract.run_attempt = 100, 1
    _patch_intent(runner, config=config, contract=contract, state="attached")
    contract.terminal_run_id, contract.terminal_run_attempt = 100, 1
    contract.terminal_attempts = ((100, 1),)
    contract.terminal_outcome = "no-status"
    _patch_intent(runner, config=config, contract=contract, state="completed")

    pr = {
        "state": "open", "draft": True, "number": 7,
        "base": {"ref": "main"}, "head": {
            "sha": expected_head, "ref": "agent-loop/managed-7",
            "repo": {"full_name": "OWNER/REPO"},
        }, "user": {"login": "agent-loop", "id": 7},
        "labels": [{"name": MANAGED_LABEL}],
    }
    for snapshot in runner.intent_snapshots:
        pages = [[{
            "user": {"login": "agent-loop", "id": 7},
            "body": f"<!-- AGENT_MANAGED_CI_INTENT_V2 {json.dumps(snapshot, separators=(',', ':'))} -->",
        }]]
        if snapshot["state"] == "prepared":
            with pytest.raises(ValueError, match="exactly one distinct qualifying intent"):
                historical_router.validate(
                    pr, pages, "OWNER/REPO", "7", expected_head, contract.nonce, "agent-loop", revision
                )
            with pytest.raises(ValueError, match="prepared intent"):
                local_router.validate(
                    pr, pages, "OWNER/REPO", "7", expected_head, contract.nonce, "agent-loop", revision, 7
                )
            with pytest.raises(ValueError, match="exactly one distinct qualifying intent"):
                current_router.validate(
                    pr, pages, "OWNER/REPO", "7", expected_head, contract.nonce, "agent-loop", revision
                )
        else:
            historical_router.validate(
                pr, pages, "OWNER/REPO", "7", expected_head, contract.nonce, "agent-loop", revision
            )
            current_router.validate(
                pr, pages, "OWNER/REPO", "7", expected_head, contract.nonce, "agent-loop", revision
            )
            local_router.validate(
                pr, pages, "OWNER/REPO", "7", expected_head, contract.nonce, "agent-loop", revision, 7
            )


def test_v2_current_pinned_consumer_scopes_lifecycle_validation_to_requested_nonce(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    revision = "a" * 40
    expected_head = "b" * 40
    runner = V2ManagedRunner()
    contract = v2_contract(workflow_revision=revision)
    _ensure_v2_intent(
        runner, config=config, pr_number=7, expected_head_sha=expected_head, contract=contract
    )
    _patch_intent(runner, config=config, contract=contract, state="dispatch-requested")
    current = runner.intent_snapshots[-1]
    unrelated = dict(current)
    unrelated.update({"nonce": "C" * 32, "state": "terminal-no-status", "run_id": 100, "run_attempt": 1})
    def body(record):
        return {"user": {"login": "agent-loop"}, "body": f"<!-- AGENT_MANAGED_CI_INTENT_V2 {json.dumps(record, separators=(',', ':'))} -->"}
    pr = {
        "state": "open", "draft": True, "number": 7,
        "base": {"ref": "main"}, "head": {
            "sha": expected_head, "ref": "agent-loop/managed-7",
            "repo": {"full_name": "OWNER/REPO"},
        }, "user": {"login": "agent-loop"},
        "labels": [{"name": MANAGED_LABEL}],
    }
    pages = [[body(current), body(unrelated)]]
    current_router.validate(pr, pages, "OWNER/REPO", "7", expected_head, contract.nonce, "agent-loop", revision)
    with pytest.raises(ValueError, match="invalid state"):
        historical_router.validate(pr, pages, "OWNER/REPO", "7", expected_head, contract.nonce, "agent-loop", revision)
    invalid_current = dict(current)
    invalid_current["state"] = "terminal-no-status"
    with pytest.raises(ValueError, match="invalid state"):
        current_router.validate(
            pr, [[body(invalid_current)]], "OWNER/REPO", "7", expected_head,
            contract.nonce, "agent-loop", revision,
        )


def test_v2_legacy_no_status_record_is_refused_instead_of_restored(tmp_path):
    """A legacy terminal-no-status state is one the base workflow fails.

    Rediscovery mirrors the workflow's verdict (#1043), so such a record is
    never adopted and no fresh intent is minted beside it.
    """
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = valid_v2_runner(intent_comments=[v2_intent_comment(
        run_id=100, run_attempt=None, state="terminal-no-status",
        terminal_run_id=100, terminal_run_attempt=None,
    )])
    contract = valid_v2_contract()

    with pytest.raises(managed_ci.ManagedCiIntentLedgerError, match="invalid state"):
        _ensure_v2_intent(
            runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract
        )

    assert contract.intent_comment_id is None
    assert runner.intent_snapshots == []


def test_v2_missing_terminal_attempt_excludes_same_run_but_allows_fresh_run():
    exclusions = ((100, None),)

    assert _v2_terminal_attempt_excluded(100, 2, exclusions)
    assert not _v2_terminal_attempt_excluded(101, 2, exclusions)


def test_v2_dispatch_discovers_existing_run_before_dispatching(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = valid_v2_runner(workflow_runs=[valid_v2_run()], intent_comments=[v2_intent_comment()])
    contract = valid_v2_contract()

    _dispatch_v2_qualification(
        runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract
    )

    assert contract.attached_run_id == 100
    assert not any("/dispatches" in " ".join(cmd) for cmd, _cwd in runner.commands)


def test_v2_dispatch_ledger_failure_uses_authenticated_ordinary_recovery_capability(tmp_path, monkeypatch):
    config = make_config(
        tmp_path,
        auto_merge=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
    )
    runner = V2ManagedRunner(issue_events=[label_event()])
    contract = v2_contract(ordinary_recovery_capable=True)

    def fail_intent(*args, **kwargs):
        raise AgentLoopError("intent ledger unavailable")

    monkeypatch.setattr(managed_ci, "_ensure_v2_intent", fail_intent)

    _dispatch_v2_qualification(
        runner, config=config, pr_number=7, expected_head_sha="abc123", contract=contract
    )

    assert contract.activation_path == "ordinary_fallback"
    assert contract.ordinary_recovery is not None
    assert any(
        command[:5] == [
            "gh", "api", "--method", "DELETE",
            f"repos/OWNER/REPO/issues/7/labels/{MANAGED_LABEL}",
        ]
        for command, _cwd in runner.commands
    )


def test_v2_dispatch_discovers_run_with_display_title_and_qualified_path(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = valid_v2_runner(
        workflow_runs=[
            valid_v2_run(
                name="CI",
                display_title=f"managed-ci-v2 nonce={V2_NONCE}",
                path="OWNER/REPO/.github/workflows/ci.yml@refs/heads/main",
            )
        ],
        intent_comments=[v2_intent_comment()],
    )
    contract = valid_v2_contract()

    _dispatch_v2_qualification(
        runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract
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
        def _run_locked(self, args, *, cwd, check, input_text=None):
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


# --- Issue #878: visible labels on tool-owned managed-CI records ------------


def _authorization_record(kind, *, head="abc123"):
    extra = (
        {
            "predecessor_head": "old-head",
            "predecessor_comment_id": 41,
            "round_comment_ids": (88, 89),
        }
        if kind == "continuity"
        else {}
    )
    return ManagedCiIssueAuthorization(
        kind=kind,
        repository="OWNER/REPO",
        issue_number=643,
        pr_number=7,
        base_ref="main",
        head_sha=head,
        actor_login="agent-loop",
        actor_id=1,
        protection="voluntary",
        waiver="allow-unprotected-managed-ci",
        nonce="nonce-643",
        label_event_id=101,
        **extra,
    )


def _authorization_marker(record):
    encoded = managed_ci._encode_issue_authorization_payload(record.to_payload())
    return f"<!-- {managed_ci.ISSUE_AUTHORIZATION_MARKER}: {encoded} -->"


def test_authorization_bodies_are_labeled_per_kind_without_touching_the_marker():
    labels = {}
    for kind in ("creation", "fresh", "continuity"):
        record = _authorization_record(kind)
        body = str(format_issue_created_authorization_comment(record))
        marker = _authorization_marker(record)
        label, separator, tail = body.partition("\n\n")
        labels[kind] = label
        assert separator == "\n\n"
        assert tail == marker
        assert label.startswith("Agent-loop managed-CI ")
        assert "issue #643" in label and "pull request #7" in label
        assert "abc123"[:7] in label
        assert parse_issue_created_authorization_comment(body) == record
        # Deterministic: a retry renders a byte-identical body.
        assert str(format_issue_created_authorization_comment(record)) == body
    assert len(set(labels.values())) == 3


def test_labeled_creation_authorization_is_published_once_and_read_back(tmp_path):
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )

    published = publish_issue_created_authorization(
        runner, config=config, handoff=_authorization_handoff(), metadata=metadata()
    )

    stored = next(
        comment for comment in runner.intent_comments
        if comment["id"] == published.authorization_comment_id
    )
    record = parse_issue_created_authorization_comment(stored["body"])
    assert record is not None and record.kind == "creation"
    assert stored["body"] == str(format_issue_created_authorization_comment(record))
    assert stored["body"].startswith("Agent-loop managed-CI authorization record")


def test_historical_marker_only_authorization_still_suppresses_republication(tmp_path):
    expected = _authorization_record("creation")
    runner = AuthorizationCommentRunner(
        issue_events=[label_event()],
        intent_comments=[{
            "id": 41,
            "user": {"login": "agent-loop", "id": 1},
            "body": _authorization_marker(expected),
        }],
    )
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )

    published = publish_issue_created_authorization(
        runner, config=config, handoff=_authorization_handoff(), metadata=metadata()
    )

    assert published.authorization_comment_id == 41
    assert not any(
        "issues/7/comments" in " ".join(command) and "POST" in command
        for command, _cwd in runner.commands
    )


class AlteredLabelAuthorizationRunner(AuthorizationCommentRunner):
    """Echo a stored body whose visible label was altered in transit."""

    def _run_locked(self, args, *, cwd, check, input_text=None):
        endpoint = next(
            (part for part in args if isinstance(part, str) and part.startswith("repos/")), ""
        )
        if endpoint == "repos/OWNER/REPO/issues/7/comments" and "POST" in args:
            body = self._form_value(list(args), "body") or ""
            cmd, cwd_path = self._record_command(args, cwd)
            return CommandResult(
                cmd, cwd_path,
                json.dumps({
                    "id": 17,
                    "body": body.replace("Agent-loop", "Agent-loop (edited)", 1),
                    "user": {"login": self.actor_login, "id": self.actor_id},
                }),
                "", 0,
            )
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


def test_altered_authorization_label_fails_read_back_and_adopts_no_identity(tmp_path):
    runner = AlteredLabelAuthorizationRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )

    with pytest.raises(AgentLoopError, match="returned a different body"):
        publish_issue_created_authorization(
            runner, config=config, handoff=_authorization_handoff(), metadata=metadata()
        )
    assert runner.intent_comments == []


def test_intent_body_stays_marker_only_for_the_base_workflow_validator(tmp_path):
    """#888: the installed workflow anchors its envelope at the body start.

    A visible #878 label ahead of the marker made every managed dispatch fail
    with "expected exactly one fresh intent for requested nonce".
    """
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = V2ManagedRunner()
    contract = v2_contract()

    _ensure_v2_intent(
        runner, config=config, pr_number=7, expected_head_sha="abc123", contract=contract
    )

    stored = next(
        comment for comment in runner.intent_comments
        if comment["id"] == contract.intent_comment_id
    )
    body = stored["body"]
    assert body.startswith(f"<!-- {managed_ci.INTENT_MARKER} ")
    # The exact envelope the base workflow applies, anchored and fullmatch.
    envelope = re.compile(
        rf"^<!-- {managed_ci.INTENT_MARKER} (?P<payload>.*?) -->$", re.S
    )
    assert envelope.match(body) is not None

    _patch_intent(runner, config=config, contract=contract, state="dispatch-requested")
    first = next(
        comment for comment in runner.intent_comments
        if comment["id"] == contract.intent_comment_id
    )["body"]
    _patch_intent(runner, config=config, contract=contract, state="dispatch-requested")
    second = next(
        comment for comment in runner.intent_comments
        if comment["id"] == contract.intent_comment_id
    )["body"]
    assert first == second
    assert envelope.match(second) is not None


def test_labeled_intent_comment_is_skipped_like_the_base_workflow_skips_it(tmp_path):
    """A free-form label ahead of the record is not a workflow envelope.

    The base workflow skips such a comment, so rediscovery must not adopt it
    either (#1043); a fresh intent is posted instead.
    """
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    historical = v2_intent_comment(run_id=100, run_attempt=1)
    labeled = dict(historical)
    labeled["body"] = (
        protocol_record_label(
            "managed_ci_intent", pr_number=7, head_sha=V2_HEAD, state="attached"
        )
        + "\n\n"
        + historical["body"]
    )
    runner = valid_v2_runner(intent_comments=[labeled])
    contract = valid_v2_contract()

    _ensure_v2_intent(
        runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract
    )

    assert contract.intent_comment_id != 17
    assert contract.nonce != V2_NONCE
    assert contract.attached_run_id is None
    assert [snapshot["state"] for snapshot in runner.intent_snapshots] == ["prepared"]


class AlteredLabelIntentRunner(V2ManagedRunner):
    """Alter the echoed intent body on create or update.

    The intent record is marker-only again (#888), so the alteration prepends
    a visible prefix instead of editing a label, which is exactly the drift the
    read-back must reject.
    """

    def __init__(self, *, alter_patch=False, **kwargs):
        super().__init__(**kwargs)
        self.alter_patch = alter_patch

    def _run_locked(self, args, *, cwd, check, input_text=None):
        result = super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)
        endpoint = next(
            (part for part in args if isinstance(part, str) and part.startswith("repos/")), ""
        )
        patching = endpoint.startswith("repos/OWNER/REPO/issues/comments/")
        creating = endpoint == "repos/OWNER/REPO/issues/7/comments" and "POST" in list(args)
        if (patching and self.alter_patch) or (creating and not self.alter_patch):
            try:
                payload = json.loads(result.stdout or "{}")
            except json.JSONDecodeError:
                return result
            if isinstance(payload, dict) and isinstance(payload.get("body"), str):
                payload["body"] = "Agent-loop (edited)\n\n" + payload["body"]
                return CommandResult(
                    result.args, result.cwd, json.dumps(payload), "", 0
                )
        return result


def test_altered_intent_body_fails_the_create_read_back(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = AlteredLabelIntentRunner()
    contract = v2_contract()

    with pytest.raises(AgentLoopError, match="returned a different body"):
        _ensure_v2_intent(
            runner, config=config, pr_number=7, expected_head_sha="abc123", contract=contract
        )
    assert contract.intent_comment_id is None
    assert contract.intent_state is None


def test_altered_intent_body_fails_the_patch_read_back_without_advancing_state(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = AlteredLabelIntentRunner(alter_patch=True)
    contract = v2_contract()
    _ensure_v2_intent(
        runner, config=config, pr_number=7, expected_head_sha="abc123", contract=contract
    )
    assert contract.intent_state == "prepared"

    with pytest.raises(AgentLoopError, match="returned a different body"):
        _patch_intent(runner, config=config, contract=contract, state="dispatch-requested")

    assert contract.intent_state == "prepared"


def _override_audit_runner(**kwargs):
    return V2ManagedRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        rest_pr={"body": f"{UNPROTECTED_OVERRIDE_TRAILER} nonce=nonce-from-preflight"},
        **kwargs,
    )


def test_override_audit_names_its_record_and_still_parses(tmp_path):
    runner = _override_audit_runner()
    config = make_config(
        tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
        managed_ci_expected_override_nonce="nonce-from-preflight",
    )

    contract = activate_managed_ci(runner, config=config, pr_number=7, metadata=metadata())

    assert contract is not None and contract.activation_path == "managed"
    audit = runner.audit_comments[contract.audit_comment_id]["body"]
    assert audit.startswith("Agent-loop managed-CI unprotected-override audit record")
    assert f"\n{UNPROTECTED_OVERRIDE_TRAILER} " in audit
    record = parse_managed_ci_override_record(
        audit, surface=PR_COMMENT_SURFACE, schema="audit", required=True
    )
    assert record is not None and record.nonce == "nonce-from-preflight"


class AlteredAuditLabelRunner(V2ManagedRunner):
    """Echo audit comments whose visible label differs from the posted one."""

    def _run_locked(self, args, *, cwd, check, input_text=None):
        result = super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)
        endpoint = next(
            (part for part in args if isinstance(part, str) and part.startswith("repos/")), ""
        )
        if endpoint != "repos/OWNER/REPO/issues/7/comments" or "POST" not in list(args):
            return result
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            return result
        body = payload.get("body") if isinstance(payload, dict) else None
        if isinstance(body, str) and body.startswith("Agent-loop managed-CI"):
            payload["body"] = body.replace("Agent-loop", "Agent-loop (edited)", 1)
            return CommandResult(result.args, result.cwd, json.dumps(payload), "", 0)
        return result


def test_unverified_override_audit_falls_back_to_ordinary_ci(tmp_path):
    runner = AlteredAuditLabelRunner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        rest_pr={"body": f"{UNPROTECTED_OVERRIDE_TRAILER} nonce=nonce-from-preflight"},
    )
    config = make_config(
        tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
        managed_ci_expected_override_nonce="nonce-from-preflight",
    )

    contract = activate_managed_ci(runner, config=config, pr_number=7, metadata=metadata())

    assert contract is None


def test_qualified_head_comment_is_labeled_and_verified(tmp_path):
    config = make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop")
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
        runner, config=config, pr_number=7, expected_head_sha="abc123",
        contract=contract, reviewers=("Codex",),
    )

    assert qualified == "abc123"
    audit = runner.audit_comments[contract.audit_comment_id]["body"]
    label, separator, tail = audit.partition("\n\n")
    assert separator == "\n\n"
    assert label.startswith("Agent-loop managed-CI qualified-head record")
    # The canonical marker span itself is unchanged; it has no parser consumer.
    assert tail.startswith(f"<!-- {QUALIFICATION_MARKER} repo=OWNER/REPO pr=7 ")
    assert "qualified_head=abc123" in tail


class AlteredQualificationLabelRunner(ManualQualificationRunner, AlteredAuditLabelRunner):
    pass


def test_unverified_qualification_audit_raises_and_adopts_no_head(tmp_path):
    config = make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop")
    runner = AlteredQualificationLabelRunner(issue_events=[label_event()])
    contract = v2_contract(
        issue_created_pr=True,
        active_label_event_id=101,
        invocation_applied_label=True,
        protection_mode="strict",
        attached_run_id=100,
        run_attempt=2,
        intent_generation="generation-1",
    )

    with pytest.raises(AgentLoopError, match="returned a different body"):
        publish_manual_v2_qualification(
            runner, config=config, pr_number=7, expected_head_sha="abc123",
            contract=contract, reviewers=("Codex",),
        )

    assert contract.audit_comment_id is None


def _resume_activation_runner(runner_class):
    """Build a PR-mode resume fixture carrying a durable issue authorization."""
    resume_metadata = replace(metadata(), head_branch="agent-loop/managed-643")
    runner = runner_class(
        workflow=SUPPRESSING_V2_WORKFLOW,
        rest_pr={"state": "open", "draft": True, "body": resume_metadata.body},
        issue_events=[label_event()],
        intent_comments=[{
            "id": 41,
            "user": {"login": "agent-loop", "id": 1},
            "body": str(format_issue_created_authorization_comment(
                ManagedCiIssueAuthorization(
                    kind="creation", repository="OWNER/REPO", issue_number=643,
                    pr_number=7, base_ref="main", head_sha="abc123",
                    actor_login="agent-loop", actor_id=1, protection="voluntary",
                    waiver="allow-unprotected-managed-ci", nonce="nonce-643",
                    label_event_id=101,
                )
            )),
        }],
    )
    return runner, resume_metadata


def _activate_resume(runner, *, config, resume_metadata):
    handoff = recover_issue_created_handoff(
        runner, config=config, pr_number=7, metadata=resume_metadata, issue_number=643
    )
    assert handoff is not None
    return activate_managed_ci(
        runner,
        config=config,
        pr_number=7,
        metadata=resume_metadata,
        managed_resume=AuthenticatedManagedResume(
            origin="issue-created",
            lifecycle=handoff.lifecycle,
            issue_created_handoff=handoff,
        ),
    )


def _resume_config(tmp_path):
    return make_config(
        tmp_path,
        auto_merge=True,
        managed_ci=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )


def test_resume_audit_names_its_record_and_still_parses(tmp_path):
    runner, resume_metadata = _resume_activation_runner(V2ManagedRunner)

    contract = _activate_resume(
        runner, config=_resume_config(tmp_path), resume_metadata=resume_metadata
    )

    assert contract is not None and contract.activation_path == "managed"
    audit = runner.audit_comments[contract.audit_comment_id]["body"]
    assert audit.startswith("Agent-loop managed-CI resume provenance audit record")
    assert f"\n{UNPROTECTED_OVERRIDE_TRAILER} " in audit
    record = parse_managed_ci_override_record(
        audit, surface=PR_COMMENT_SURFACE, schema="audit", required=True
    )
    assert record is not None and record.nonce == contract.audit_nonce


class AlteredResumeAuditLabelRunner(AlteredAuditLabelRunner):
    """Alter only the resume-provenance audit label on its way back."""

    def _run_locked(self, args, *, cwd, check, input_text=None):
        body = self._form_value(list(args), "body") or ""
        if not body.startswith("Agent-loop managed-CI resume provenance audit record"):
            return V2ManagedRunner._run_locked(
                self, args, cwd=cwd, check=check, input_text=input_text
            )
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


def test_unverified_resume_audit_falls_back_to_ordinary_recovery(tmp_path):
    runner, resume_metadata = _resume_activation_runner(AlteredResumeAuditLabelRunner)

    # The resume branch takes the same ordinary-CI fallback it already takes
    # when the audit cannot be recorded: it releases the managed label and
    # refuses to qualify, instead of activating on an unverified audit.
    with pytest.raises(AgentLoopError, match="resume audit could not be recorded and verified"):
        _activate_resume(
            runner, config=_resume_config(tmp_path), resume_metadata=resume_metadata
        )

    # The server-side comment may exist, but no managed contract adopted it.
    assert any(
        command[:5] == [
            "gh", "api", "--method", "DELETE",
            f"repos/OWNER/REPO/issues/7/labels/{MANAGED_LABEL}",
        ]
        for command, _cwd in runner.commands
    )
    assert not any(
        command[:4] == ["gh", "pr", "merge", "7"] for command, _cwd in runner.commands
    )


# --- #953: a spilled prior_items still carries the merge-conflict obligation ---


def _spilled_conflict_round_comments(*, subject, round_number, first_id):
    """A conflict coder record whose prior_items spilled into sidecar comments."""
    import base64 as _base64
    import os as _os

    import coding_review_agent_loop.round_transport as transport

    conflict = UnresolvedReviewItem(
        item_id="item-merge-conflict",
        reviewer="agent-loop",
        source_round=round_number,
        text="PR has a merge conflict with main. "
        + _base64.urlsafe_b64encode(_os.urandom(50_000)).decode("ascii"),
        status="blocking",
        authority="machine",
        obligation_kind="merge-conflict",
        lifecycle="repair_required",
    )
    prepared = transport.prepare_round_comment(
        _attach_round_metadata(
            "coder round",
            PostedRoundMetadata(
                flow="pr", role="coder", agent="agent-loop",
                round_number=round_number, subject=subject,
                prior_items=(conflict,),
            ),
        )
    )
    anchor = transport.ROUND_RESUME_MARKER_RE.search(str(prepared[-1]))
    reference = transport.decode_mapping(anchor.group("payload"))["prior_items"]
    assert isinstance(reference, dict) and "$round_transport_spill" in reference
    return [
        {"id": first_id + index, "user": {"login": "agent-loop", "id": 1}, "body": str(body)}
        for index, body in enumerate(prepared)
    ]


def _spilled_conflict_continuity():
    comments = _spilled_conflict_round_comments(
        subject="merged-head", round_number=13, first_id=58
    )
    anchor_id = comments[-1]["id"]
    authorization = ManagedCiIssueAuthorization(
        kind="continuity", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="merged-head", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="next",
        label_event_id=101, predecessor_head="abc123", predecessor_comment_id=50,
        round_comment_ids=(anchor_id,),
    )
    return comments, anchor_id, authorization


def test_m953_spilled_conflict_obligation_grants_continuity(tmp_path):
    comments, anchor_id, authorization = _spilled_conflict_continuity()

    records = managed_ci._continuity_round_records(comments)
    assert records[len(comments) - 1]["resolves_merge_conflict"] is True
    assert managed_ci._continuity_round_metadata_is_valid(
        comments, authorization=authorization
    ) is True

    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    runner.intent_comments.extend(comments)
    selected = managed_ci.find_actor_round_metadata_comment_ids(
        runner, config=make_config(tmp_path), pr_number=7, actor_login="agent-loop",
        actor_id=1, predecessor_head="abc123", new_head="merged-head", round_number=12,
        after_comment_id=50,
    )
    assert selected == (anchor_id,)


def test_m953_legacy_reader_declines_spilled_conflict_continuity(tmp_path, monkeypatch):
    """An older binary cannot see the spilled obligation and so denies, never grants."""
    import coding_review_agent_loop.round_transport as transport

    comments, _anchor_id, authorization = _spilled_conflict_continuity()
    monkeypatch.setattr(
        transport,
        "_SPILL_FIELDS",
        tuple(
            field for field in transport._SPILL_FIELDS
            if field not in transport._GROWTH_SPILL_FIELDS
        ),
    )

    records = managed_ci._continuity_round_records(comments)
    assert records[len(comments) - 1]["resolves_merge_conflict"] is False
    assert managed_ci._continuity_round_metadata_is_valid(
        comments, authorization=authorization
    ) is False

    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    runner.intent_comments.extend(comments)
    with pytest.raises(AgentLoopError, match="correlated blocking-review and coder"):
        managed_ci.find_actor_round_metadata_comment_ids(
            runner, config=make_config(tmp_path), pr_number=7, actor_login="agent-loop",
            actor_id=1, predecessor_head="abc123", new_head="merged-head",
            round_number=12, after_comment_id=50,
        )


# --- #1040: refused Actions-variable and classic-protection reads -----------

PROXY_403 = (
    "gh: Access to this GitHub Actions path is not permitted through this proxy (HTTP 403)\n"
)
INTEGRATION_403 = "gh: Resource not accessible by integration (HTTP 403)\n"
GH_404 = "gh: Not Found (HTTP 404)\n"
GH_422 = "gh: Validation Failed (HTTP 422)\n"
GH_500 = "gh: Server Error (HTTP 500)\n"
PLAN_LIMITED = "HTTP 403: Upgrade to GitHub Pro or make this repository public"
_VARIABLE = "/actions/variables/AGENT_LOOP_MANAGED_ACTOR"
_CLASSIC = "/branches/main/protection/required_status_checks"
_ADMINS = "/branches/main/protection/enforce_admins"
_RULES = "/rules/branches/main"
_BOTH_WAIVERS = "--allow-unprotected-managed-ci --allow-unreadable-protection"


def _final_rules(context=FINAL_CONTEXT):
    return [{
        "type": "required_status_checks",
        "parameters": {"required_status_checks": [{"context": context}]},
    }]


STRICT_RULESET = {"enforcement": "active", "bypass_actors": [], "rules": _final_rules()}


class _ScriptedReadsMixin:
    """Answer selected read-only endpoints with scripted gh results."""

    scripted: dict = {}

    def _run_locked(self, args, *, cwd, check, input_text=None):
        endpoint = next(
            (part for part in args if isinstance(part, str) and part.startswith("repos/")), ""
        )
        if "--method" not in args:
            for suffix, (stdout, stderr, returncode) in self.scripted.items():
                if endpoint.endswith(suffix):
                    cmd, cwd_path = self._record_command(args, cwd)
                    return CommandResult(cmd, cwd_path, stdout, stderr, returncode)
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


class CloudSessionRunner(_ScriptedReadsMixin, V2ManagedRunner):
    def __init__(self, *, scripted=None, **kwargs):
        kwargs.setdefault("workflow", SUPPRESSING_V2_WORKFLOW)
        super().__init__(**kwargs)
        self.scripted = dict(scripted or {})


class CloudAuthorizationRunner(_ScriptedReadsMixin, AuthorizationCommentRunner):
    def __init__(self, *, scripted=None, **kwargs):
        kwargs.setdefault("workflow", SUPPRESSING_V2_WORKFLOW)
        super().__init__(**kwargs)
        self.scripted = dict(scripted or {})


def _failed(stderr, stdout=""):
    return (stdout, stderr, 1)


def _cloud_scripts(*, variable=True, classic=True, rules=None):
    scripted = {}
    if variable:
        scripted[_VARIABLE] = _failed(PROXY_403)
    if classic:
        scripted[_CLASSIC] = _failed(INTEGRATION_403)
    if rules is not None:
        scripted[_RULES] = rules
    return scripted


def _probe_context(tmp_path):
    return ManagedCiProbeContext("OWNER/REPO", "gh", tmp_path)


def _mutations(runner):
    return [command for command, _cwd in runner.commands if "--method" in command]


def _cloud_config(tmp_path, *, companion=True, unreadable=True, **overrides):
    overrides.setdefault("managed_ci", True)
    return make_config(
        tmp_path,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=companion,
        allow_unreadable_protection=unreadable,
        **overrides,
    )


@pytest.mark.parametrize(
    ("stdout", "stderr", "returncode", "expected"),
    [
        ("", PROXY_403, 1, 403),
        ("", INTEGRATION_403, 1, 403),
        ("", PROXY_403, 0, None),
        ('{"message": "upstream said HTTP 403 (HTTP 403)"}', GH_500, 1, 500),
        ("", "gh: upstream said (HTTP 403) retry later (HTTP 500)\n", 1, 500),
        ("", "error: HTTP 403 forbidden by upstream\n", 1, None),
        ("", "gh: denied (HTTP 403) by policy\n", 1, None),
        ("", PROXY_403 + GH_500, 1, None),
        ("gh: Resource not accessible by integration (HTTP 403)", "", 1, None),
        ("", GH_404, 1, 404),
    ],
)
def test_http_status_reads_only_gh_trailing_stderr_status(
    tmp_path, stdout, stderr, returncode, expected
):
    result = CommandResult(["gh", "api", "repos/OWNER/REPO"], tmp_path, stdout, stderr, returncode)

    assert managed_ci._http_status(result) == expected


@pytest.mark.parametrize(
    ("stdout", "stderr", "returncode", "status"),
    [
        ('{"value": "agent-loop"}', "", 0, "readable"),
        ("", PROXY_403, 1, "unreadable"),
        ("", INTEGRATION_403, 1, "unreadable"),
        ('{"message": "Not Found", "status": "404"}', INTEGRATION_403, 1, "unreadable"),
        ("", GH_404, 1, "absent"),
        ("", "HTTP 404: Not Found", 1, "absent"),
        ('{"message": "HTTP 403"}', GH_500, 1, "error"),
        ("", "gh: upstream said (HTTP 403) retry later (HTTP 500)\n", 1, "error"),
        ("", "dial tcp: connection reset by peer", 1, "error"),
    ],
)
def test_actor_variable_read_is_asserted_only_for_a_strict_403(
    tmp_path, stdout, stderr, returncode, status
):
    runner = CloudSessionRunner(scripted={_VARIABLE: (stdout, stderr, returncode)})

    variable = managed_ci._read_managed_actor_variable(runner, "gh", "OWNER/REPO", tmp_path)

    assert variable.status == status
    expected = {
        "readable": ("agent-loop", "variable"),
        "unreadable": ("agent-loop", "asserted"),
    }.get(status, (None, None))
    assert managed_ci._resolve_advertised_actor(variable, " agent-loop ") == expected
    # Without a trusted actor nothing can stand in for the refused read.
    if status == "unreadable":
        assert managed_ci._resolve_advertised_actor(variable, "") == (None, None)


def test_readable_variable_is_never_replaced_by_the_trusted_actor(tmp_path):
    runner = CloudSessionRunner(scripted={_VARIABLE: ('{"value": "someone-else"}', "", 0)})

    variable = managed_ci._read_managed_actor_variable(runner, "gh", "OWNER/REPO", tmp_path)

    assert managed_ci._resolve_advertised_actor(variable, "agent-loop") == (
        "someone-else", "variable",
    )


@pytest.mark.parametrize(
    "variable",
    [
        _failed(PROXY_403),
        _failed(INTEGRATION_403),
        _failed(INTEGRATION_403, stdout='{"message": "Not Found", "documentation": "404"}'),
    ],
)
def test_variable_403_asserts_trusted_actor_in_readiness_and_creation(
    tmp_path, monkeypatch, variable
):
    messages = []
    monkeypatch.setattr(managed_ci, "_ASSERTED_ACTOR_LOGGED", set())
    monkeypatch.setattr(managed_ci, "log", lambda _config, message: messages.append(message))
    runner = CloudSessionRunner(
        scripted={_VARIABLE: variable},
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT]},
    )

    readiness = evaluate_managed_ci_readiness(
        runner, context=_probe_context(tmp_path), base="main", trusted_actor="agent-loop",
    )

    assert readiness.state == "strict_ready"
    assert readiness.advertised_actor == "agent-loop"
    assert readiness.advertised_actor_source == "asserted"
    rendered = managed_ci.render_managed_ci_preflight(
        readiness, repo="OWNER/REPO", base="main", trusted_actor="agent-loop",
    )
    assert "variable=agent-loop (asserted; unverified locally, enforced by workflow)" in rendered

    intent = preflight_managed_ci_creation(
        runner, config=_cloud_config(tmp_path, companion=False, unreadable=False),
        issue_number=643,
    )

    assert intent is not None
    assert intent.protection_mode == "strict"
    assert intent.audit_nonce is None
    assert any("asserted and unverified locally" in message for message in messages)
    assert any("enforces vars.AGENT_LOOP_MANAGED_ACTOR server-side" in message for message in messages)


def test_genuine_variable_404_stays_absent_and_is_not_asserted(tmp_path):
    runner = CloudSessionRunner(
        scripted={_VARIABLE: _failed(GH_404)},
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT]},
    )

    readiness = evaluate_managed_ci_readiness(
        runner, context=_probe_context(tmp_path), base="main", trusted_actor="agent-loop",
    )

    assert readiness.state == "ordinary_fallback"
    assert readiness.advertised_actor is None
    assert readiness.advertised_actor_source is None


@pytest.mark.parametrize("variable", [_failed(GH_500), _failed("connection reset by peer")])
def test_non_403_variable_failure_stays_indeterminate_and_names_the_read(tmp_path, variable):
    runner = CloudSessionRunner(
        scripted={_VARIABLE: variable},
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT]},
    )

    readiness = evaluate_managed_ci_readiness(
        runner, context=_probe_context(tmp_path), base="main", trusted_actor="agent-loop",
    )

    assert readiness.state == "indeterminate"
    assert readiness.advertised_actor is None
    assert any("actions/variables/AGENT_LOOP_MANAGED_ACTOR" in reason for reason in readiness.reasons)
    with pytest.raises(AgentLoopError, match="AGENT_LOOP_MANAGED_ACTOR could not be read") as exc_info:
        preflight_managed_ci_creation(runner, config=_cloud_config(tmp_path), issue_number=643)
    assert "no PR was created" in str(exc_info.value)
    assert _mutations(runner) == []


def test_asserted_actor_that_disagrees_with_login_fails_closed(tmp_path):
    runner = CloudSessionRunner(
        scripted=_cloud_scripts(),
        actor_login="intruder",
        actor_id=9,
    )

    readiness = evaluate_managed_ci_readiness(
        runner, context=_probe_context(tmp_path), base="main", trusted_actor="agent-loop",
    )

    assert readiness.state == "ordinary_fallback"
    assert readiness.advertised_actor_source == "asserted"
    with pytest.raises(AgentLoopError, match="no PR was created"):
        preflight_managed_ci_creation(runner, config=_cloud_config(tmp_path), issue_number=643)
    config = _cloud_config(tmp_path)
    with pytest.raises(AgentLoopError, match="must|match"):
        managed_ci._authorization_actor(runner, config=config)
    assert managed_ci._adoption_identity(runner, config=config) is None
    assert _mutations(runner) == []


def _override_metadata(body):
    return PullRequestMetadata(
        number=7,
        repo="OWNER/REPO",
        title="Managed CI",
        head_branch="agent-loop/managed-643",
        base_branch="main",
        head_sha="abc123",
        url="https://github.com/OWNER/REPO/pull/7",
        body=body,
    )


def test_readable_differing_variable_fails_closed_at_every_identity_site(tmp_path):
    body = f"Fixes #643\n\n{UNPROTECTED_OVERRIDE_TRAILER} nonce=fresh"
    runner = CloudSessionRunner(
        scripted={_CLASSIC: _failed(INTEGRATION_403)},
        advertised_actor="someone-else",
        rest_pr={"state": "open", "body": body},
    )
    config = _cloud_config(tmp_path)

    readiness = evaluate_managed_ci_readiness(
        runner, context=_probe_context(tmp_path), base="main", trusted_actor="agent-loop",
    )
    assert readiness.state == "ordinary_fallback"
    assert readiness.advertised_actor == "someone-else"
    assert readiness.advertised_actor_source == "variable"
    with pytest.raises(AgentLoopError, match="no PR was created"):
        preflight_managed_ci_creation(runner, config=config, issue_number=643)
    with pytest.raises(AgentLoopError, match="AGENT_LOOP_MANAGED_ACTOR"):
        managed_ci._authorization_actor(runner, config=config)
    assert managed_ci._adoption_identity(runner, config=config) is None
    with pytest.raises(AgentLoopError, match="AGENT_LOOP_MANAGED_ACTOR=`someone-else`"):
        authenticate_issue_created_handoff(
            runner,
            config=config,
            intent=managed_ci.ManagedCiCreationIntent(
                branch="agent-loop/managed-643", trusted_actor="agent-loop",
                protection_mode="unreadable", audit_nonce="fresh",
            ),
            issue_number=643,
            pr_number=7,
            metadata=_override_metadata(body),
        )
    assert managed_ci._activate_v2_managed_ci(
        runner, config=replace(config, managed_ci=False, auto_merge=True),
        pr_number=7, metadata=metadata(),
    ) is None
    assert _mutations(runner) == []


def test_asserted_actor_is_accepted_at_every_identity_site(tmp_path):
    body = f"Fixes #643\n\n{UNPROTECTED_OVERRIDE_TRAILER} nonce=fresh"
    runner = CloudSessionRunner(
        scripted=_cloud_scripts(),
        rest_pr={"state": "open", "body": body},
    )
    config = _cloud_config(tmp_path)

    assert managed_ci._authorization_actor(runner, config=config) == ("agent-loop", 1)
    assert managed_ci._adoption_identity(runner, config=config) == ("agent-loop", 1)
    handoff = authenticate_issue_created_handoff(
        runner,
        config=config,
        intent=managed_ci.ManagedCiCreationIntent(
            branch="agent-loop/managed-643", trusted_actor="agent-loop",
            protection_mode="unreadable", audit_nonce="fresh",
        ),
        issue_number=643,
        pr_number=7,
        metadata=_override_metadata(body),
    )
    assert handoff.trusted_actor_login == "agent-loop"
    assert handoff.protection_mode == "unreadable"
    assert _mutations(runner) == []


@pytest.mark.parametrize(
    "rules",
    [None, _failed(GH_404), _failed(GH_422)],
    ids=["empty-list", "strict-404", "strict-422"],
)
def test_classic_403_with_no_strict_rules_is_unreadable_and_override_eligible(tmp_path, rules):
    runner = CloudSessionRunner(scripted=_cloud_scripts(variable=False, rules=rules))

    protection = assess_exact_head_protection(runner, context=_probe_context(tmp_path), base="main")

    assert protection.state == "unreadable"
    assert protection.source == "classic"
    assert "not readable by this token (HTTP 403)" in protection.detail
    readiness = evaluate_managed_ci_readiness(
        runner, context=_probe_context(tmp_path), base="main", trusted_actor="agent-loop",
    )
    assert readiness.state == "override_eligible"
    assert any("--allow-unreadable-protection" in item for item in readiness.remediation)


def test_enforce_admins_403_after_readable_required_context_is_unreadable(tmp_path):
    runner = CloudSessionRunner(
        scripted={_ADMINS: _failed(INTEGRATION_403)},
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT]},
    )

    protection = assess_exact_head_protection(runner, context=_probe_context(tmp_path), base="main")

    assert protection.state == "unreadable"


@pytest.mark.parametrize(
    "rules",
    [
        _failed(INTEGRATION_403, stdout='{"message": "Not Found", "status": "404"}'),
        _failed(INTEGRATION_403, stdout='{"message": "Unprocessable", "status": "422"}'),
        _failed("HTTP 404: Not Found"),
        _failed(GH_404 + GH_422),
        _failed(GH_500),
    ],
    ids=["403-body-404", "403-body-422", "no-gh-status", "conflicting-status", "server-error"],
)
def test_classic_403_with_uninspectable_rules_is_indeterminate_even_with_both_waivers(
    tmp_path, rules
):
    runner = CloudSessionRunner(scripted=_cloud_scripts(rules=rules))

    protection = assess_exact_head_protection(runner, context=_probe_context(tmp_path), base="main")

    assert protection.state == "indeterminate"
    assert protection.detail == "effective branch rules could not be inspected"
    with pytest.raises(AgentLoopError, match="effective branch rules could not be inspected") as exc_info:
        preflight_managed_ci_creation(runner, config=_cloud_config(tmp_path), issue_number=643)
    assert "No waiver applies" in str(exc_info.value)
    assert "no PR was created" in str(exc_info.value)
    assert _mutations(runner) == []


def test_required_status_403_whose_body_mentions_404_is_never_voluntary(tmp_path):
    runner = CloudSessionRunner(scripted={
        _VARIABLE: _failed(PROXY_403),
        _CLASSIC: _failed(INTEGRATION_403, stdout='{"message": "Branch not protected", "status": "404"}'),
    })

    protection = assess_exact_head_protection(runner, context=_probe_context(tmp_path), base="main")

    assert protection.state == "unreadable"
    with pytest.raises(AgentLoopError, match="--allow-unreadable-protection") as exc_info:
        preflight_managed_ci_creation(
            runner, config=_cloud_config(tmp_path, unreadable=False), issue_number=643,
        )
    assert "no PR was created" in str(exc_info.value)
    assert _mutations(runner) == []


@pytest.mark.parametrize("classic", [_failed(GH_404), _failed("HTTP 404: Not Found")])
def test_genuine_required_status_404_stays_voluntary(tmp_path, classic):
    runner = CloudSessionRunner(scripted={_CLASSIC: classic})

    protection = assess_exact_head_protection(runner, context=_probe_context(tmp_path), base="main")

    assert protection.state == "voluntary"


@pytest.mark.parametrize("rules", [_failed(GH_404), _failed("HTTP 422: Unprocessable")])
def test_readable_classic_no_rules_response_keeps_existing_classification(tmp_path, rules):
    runner = CloudSessionRunner(scripted={_RULES: rules})

    protection = assess_exact_head_protection(runner, context=_probe_context(tmp_path), base="main")

    assert protection == managed_ci.ProtectionAssessment(
        "voluntary", "none", "final-ci/exact-head is not independently required"
    )


def test_classic_403_with_visible_empty_bypass_strict_ruleset_is_strict(tmp_path):
    runner = CloudSessionRunner(
        scripted=_cloud_scripts(variable=False),
        pr_effective_rules_payload=[{"ruleset_id": 8}],
        pr_rulesets_payload={8: STRICT_RULESET},
    )

    protection = assess_exact_head_protection(runner, context=_probe_context(tmp_path), base="main")

    assert (protection.state, protection.source) == ("strict", "ruleset")
    intent = preflight_managed_ci_creation(
        runner, config=_cloud_config(tmp_path, companion=False, unreadable=False),
        issue_number=643,
    )
    assert intent is not None
    assert intent.protection_mode == "strict"
    assert intent.audit_nonce is None


def test_classic_403_with_hidden_ruleset_bypass_actors_is_unreadable(tmp_path):
    hidden = {"enforcement": "active", "rules": _final_rules()}
    runner = CloudSessionRunner(
        scripted=_cloud_scripts(),
        pr_effective_rules_payload=[{"ruleset_id": 8}],
        pr_rulesets_payload={8: hidden},
    )

    protection = assess_exact_head_protection(runner, context=_probe_context(tmp_path), base="main")

    assert protection.state == "unreadable"
    assert "does not expose its bypass actors to this token" in protection.detail
    with pytest.raises(AgentLoopError, match="--allow-unreadable-protection"):
        preflight_managed_ci_creation(
            runner, config=_cloud_config(tmp_path, unreadable=False), issue_number=643,
        )
    assert _mutations(runner) == []
    intent = preflight_managed_ci_creation(runner, config=_cloud_config(tmp_path), issue_number=643)
    assert intent is not None
    assert intent.protection_mode == "unreadable"
    assert intent.audit_nonce


def test_hidden_bypass_controls_keep_strict_classifications(tmp_path):
    hidden = {"enforcement": "active", "rules": _final_rules()}
    with_visible = CloudSessionRunner(
        scripted=_cloud_scripts(variable=False),
        pr_effective_rules_payload=[{"ruleset_id": 8}, {"ruleset_id": 9}],
        pr_rulesets_payload={8: hidden, 9: STRICT_RULESET},
    )
    assert assess_exact_head_protection(
        with_visible, context=_probe_context(tmp_path), base="main"
    ).state == "strict"

    readable_classic = CloudSessionRunner(
        pr_effective_rules_payload=[{"ruleset_id": 8}],
        pr_rulesets_payload={8: hidden},
    )
    assert assess_exact_head_protection(
        readable_classic, context=_probe_context(tmp_path), base="main"
    ).state == "strict"


_VALID_TEAM_BYPASS = {"actor_type": "Team", "actor_id": 5, "bypass_mode": "always"}


@pytest.mark.parametrize(
    ("effective_rules", "ruleset"),
    [
        (["not-an-object"], None),
        ([{"name": "no ruleset id"}], None),
        ([{"ruleset_id": 8}], {"enforcement": "active", "bypass_actors": [], "rules": "rules"}),
        ([{"ruleset_id": 8}], {"enforcement": "active", "bypass_actors": [], "rules": ["rule"]}),
        ([{"ruleset_id": 8}], {"enforcement": "active", "bypass_actors": [], "rules": [{"parameters": {}}]}),
        ([{"ruleset_id": 8}], {
            "enforcement": "active", "bypass_actors": [],
            "rules": [{"type": "required_status_checks", "parameters": {"required_status_checks": [{"name": "x"}]}}],
        }),
        ([{"ruleset_id": 8}], {"enforcement": "actve", "bypass_actors": [], "rules": _final_rules()}),
        ([{"ruleset_id": 8}], {"enforcement": "", "bypass_actors": [], "rules": _final_rules()}),
        # Unhashable JSON values must be malformed, never a TypeError.
        ([{"ruleset_id": 8}], {"enforcement": [], "bypass_actors": [], "rules": _final_rules()}),
        ([{"ruleset_id": 8}], {"enforcement": {"mode": "active"}, "bypass_actors": [], "rules": _final_rules()}),
        ([{"ruleset_id": 8}], {
            "enforcement": "active", "rules": _final_rules(),
            "bypass_actors": [{"actor_type": ["Team"], "actor_id": 5}],
        }),
        ([{"ruleset_id": 8}], {
            "enforcement": "active", "rules": _final_rules(),
            "bypass_actors": [{"actor_type": "Team", "actor_id": 5, "bypass_mode": {"x": 1}}],
        }),
        ([{"ruleset_id": 8}], {"enforcement": "active", "bypass_actors": "all", "rules": _final_rules()}),
        ([{"ruleset_id": 8}], {"enforcement": "active", "bypass_actors": ["team"], "rules": _final_rules()}),
        ([{"ruleset_id": 8}], {"enforcement": "active", "bypass_actors": [{}], "rules": _final_rules()}),
        ([{"ruleset_id": 8}], {
            "enforcement": "active", "rules": _final_rules(),
            "bypass_actors": [{"actor_type": "Team", "actor_id": "invalid"}],
        }),
        ([{"ruleset_id": 8}], {
            "enforcement": "active", "rules": _final_rules(),
            "bypass_actors": [{"actor_type": "Wizard", "actor_id": 1}],
        }),
        ([{"ruleset_id": 8}], {
            "enforcement": "active", "rules": _final_rules(),
            "bypass_actors": [{"actor_type": "Team", "actor_id": 5, "bypass_mode": "sometimes"}],
        }),
    ],
)
def test_classic_403_with_malformed_rules_stays_indeterminate(tmp_path, effective_rules, ruleset):
    runner = CloudSessionRunner(
        scripted=_cloud_scripts(),
        pr_effective_rules_payload=effective_rules,
        pr_rulesets_payload={8: ruleset} if ruleset is not None else {},
    )

    protection = assess_exact_head_protection(runner, context=_probe_context(tmp_path), base="main")

    assert protection.state == "indeterminate"
    assert protection.source == "rulesets"
    assert "effective branch rules were malformed" in protection.detail
    if ruleset is not None:
        assert managed_ci._ruleset_detail_well_formed(ruleset) is False
    readiness = evaluate_managed_ci_readiness(
        runner, context=_probe_context(tmp_path), base="main", trusted_actor="agent-loop",
    )
    assert readiness.state == "indeterminate"
    with pytest.raises(AgentLoopError, match="no PR was created"):
        preflight_managed_ci_creation(runner, config=_cloud_config(tmp_path), issue_number=643)
    assert _mutations(runner) == []


@pytest.mark.parametrize(
    "ruleset",
    [
        {"enforcement": "evaluate", "bypass_actors": [], "rules": _final_rules()},
        {"enforcement": "disabled", "bypass_actors": [], "rules": _final_rules()},
        {"enforcement": "active", "bypass_actors": [_VALID_TEAM_BYPASS], "rules": _final_rules()},
        {
            "enforcement": "active", "rules": _final_rules(),
            "bypass_actors": [{"actor_type": "OrganizationAdmin", "actor_id": None}],
        },
    ],
)
def test_classic_403_with_well_formed_non_enforcing_ruleset_is_unreadable(tmp_path, ruleset):
    assert managed_ci._ruleset_detail_well_formed(ruleset) is True
    runner = CloudSessionRunner(
        scripted=_cloud_scripts(variable=False),
        pr_effective_rules_payload=[{"ruleset_id": 8}],
        pr_rulesets_payload={8: ruleset},
    )

    protection = assess_exact_head_protection(runner, context=_probe_context(tmp_path), base="main")

    assert protection.state == "unreadable"


def test_ruleset_bypass_state_distinguishes_hidden_empty_and_present():
    assert managed_ci._ruleset_bypass_state({"rules": []}) == "unknown"
    assert managed_ci._ruleset_bypass_state({"bypass_actors": []}) == "none"
    assert managed_ci._ruleset_bypass_state({"bypass_actors": [_VALID_TEAM_BYPASS]}) == "present"


def test_final_context_requires_exact_required_status_checks_match():
    assert managed_ci._ruleset_requires_final_context(_final_rules()) is True
    assert managed_ci._ruleset_requires_final_context(
        _final_rules("final-ci/exact-head-extra")
    ) is False
    assert managed_ci._ruleset_requires_final_context(
        [{"type": "pull_request", "parameters": {"note": FINAL_CONTEXT}}]
    ) is False
    assert managed_ci._ruleset_requires_final_context("rules") is False


@pytest.mark.parametrize(
    "rules",
    [
        _final_rules("final-ci/exact-head-extra"),
        [{"type": "pull_request", "parameters": {"required_status_checks": [{"context": FINAL_CONTEXT}]}}],
    ],
    ids=["longer-context", "unrelated-rule-type"],
)
def test_near_miss_context_is_never_strict_enforcement(tmp_path, rules):
    near_miss = {"enforcement": "active", "bypass_actors": [], "rules": rules}
    readable = CloudSessionRunner(
        pr_branch_protection_payload={"contexts": []},
        pr_effective_rules_payload=[{"ruleset_id": 8}],
        pr_rulesets_payload={8: near_miss},
    )
    assert assess_exact_head_protection(
        readable, context=_probe_context(tmp_path), base="main"
    ).state == "voluntary"

    forbidden = CloudSessionRunner(
        scripted=_cloud_scripts(),
        pr_effective_rules_payload=[{"ruleset_id": 8}],
        pr_rulesets_payload={8: near_miss},
    )
    assert assess_exact_head_protection(
        forbidden, context=_probe_context(tmp_path), base="main"
    ).state == "unreadable"
    with pytest.raises(AgentLoopError, match="--allow-unreadable-protection"):
        preflight_managed_ci_creation(
            forbidden, config=_cloud_config(tmp_path, companion=False, unreadable=False),
            issue_number=643,
        )
    intent = preflight_managed_ci_creation(forbidden, config=_cloud_config(tmp_path), issue_number=643)
    assert intent is not None and intent.protection_mode == "unreadable"


@pytest.mark.parametrize("classic_forbidden", [False, True])
def test_exact_final_context_ruleset_is_strict_on_both_paths(tmp_path, classic_forbidden):
    runner = CloudSessionRunner(
        scripted=_cloud_scripts(variable=False) if classic_forbidden else {},
        pr_branch_protection_payload={"contexts": []},
        pr_effective_rules_payload=[{"ruleset_id": 8}],
        pr_rulesets_payload={8: STRICT_RULESET},
    )

    assert assess_exact_head_protection(
        runner, context=_probe_context(tmp_path), base="main"
    ).state == "strict"


@pytest.mark.parametrize(
    ("scripted", "detail"),
    [
        ({_CLASSIC: _failed(GH_500)}, "required-status protection could not be inspected"),
        ({_CLASSIC: _failed(INTEGRATION_403), "/rulesets/8": _failed(GH_500)},
         "an applicable ruleset could not be inspected"),
    ],
)
def test_other_protection_failures_stay_indeterminate_and_name_the_read(tmp_path, scripted, detail):
    runner = CloudSessionRunner(scripted=scripted, pr_effective_rules_payload=[{"ruleset_id": 8}])

    protection = assess_exact_head_protection(runner, context=_probe_context(tmp_path), base="main")

    assert protection.state == "indeterminate"
    assert protection.detail == detail
    with pytest.raises(AgentLoopError, match=detail):
        preflight_managed_ci_creation(runner, config=_cloud_config(tmp_path), issue_number=643)
    assert _mutations(runner) == []


def test_waivable_protection_states_require_each_explicit_flag(tmp_path):
    neither = make_config(tmp_path)
    companion = make_config(tmp_path, allow_unprotected_managed_ci=True)
    both = make_config(
        tmp_path, allow_unprotected_managed_ci=True, allow_unreadable_protection=True,
    )
    # The unreadable flag alone never waives anything.
    unreadable_only = make_config(tmp_path, allow_unreadable_protection=True)

    assert managed_ci.waivable_protection_states(neither) == frozenset()
    assert managed_ci.waivable_protection_states(unreadable_only) == frozenset()
    assert managed_ci.waivable_protection_states(companion) == {"voluntary", "plan_limited"}
    assert managed_ci.waivable_protection_states(both) == {"voluntary", "plan_limited", "unreadable"}
    assert managed_ci.waiver_flags_for_protection("voluntary") == "--allow-unprotected-managed-ci"
    assert managed_ci.waiver_flags_for_protection("unreadable") == _BOTH_WAIVERS
    assert managed_ci._waiver_for_protection("plan_limited") == "allow-unprotected-managed-ci"
    assert managed_ci._waiver_for_protection("unreadable") == "allow-unreadable-protection"
    assert managed_ci._waiver_for_protection("strict") is None


@pytest.mark.parametrize(
    ("companion", "unreadable"), [(False, False), (True, False), (False, True)],
)
def test_readiness_is_flag_independent_but_creation_gate_requires_both_waivers(
    tmp_path, companion, unreadable
):
    runner = CloudSessionRunner(scripted=_cloud_scripts())

    readiness = evaluate_managed_ci_readiness(
        runner, context=_probe_context(tmp_path), base="main", trusted_actor="agent-loop",
    )
    assert readiness.state == "override_eligible"
    assert readiness.protection.state == "unreadable"
    assert any(_BOTH_WAIVERS in item for item in readiness.remediation)

    with pytest.raises(AgentLoopError, match=_BOTH_WAIVERS) as exc_info:
        preflight_managed_ci_creation(
            runner,
            config=_cloud_config(tmp_path, companion=companion, unreadable=unreadable),
            issue_number=643,
        )
    assert "no PR was created" in str(exc_info.value)
    assert _mutations(runner) == []
    # Implicit auto-merge falls back to ordinary CI instead of refusing.
    assert preflight_managed_ci_creation(
        runner,
        config=_cloud_config(
            tmp_path, companion=companion, unreadable=unreadable,
            managed_ci=False, auto_merge=True,
        ),
        issue_number=643,
    ) is None


def test_cloud_session_shape_creates_reserved_managed_pr_only_with_both_waivers(
    tmp_path, monkeypatch
):
    messages = []
    monkeypatch.setattr(managed_ci, "_ASSERTED_ACTOR_LOGGED", set())
    monkeypatch.setattr(managed_ci, "log", lambda _config, message: messages.append(message))
    runner = CloudSessionRunner(scripted=_cloud_scripts())

    with pytest.raises(AgentLoopError, match="--allow-unreadable-protection") as exc_info:
        preflight_managed_ci_creation(
            runner, config=_cloud_config(tmp_path, unreadable=False), issue_number=1040,
        )
    assert "no PR was created" in str(exc_info.value)
    assert _mutations(runner) == []

    intent = preflight_managed_ci_creation(runner, config=_cloud_config(tmp_path), issue_number=1040)

    assert intent == managed_ci.ManagedCiCreationIntent(
        branch="agent-loop/managed-1040",
        trusted_actor="agent-loop",
        protection_mode="unreadable",
        audit_nonce=intent.audit_nonce,
    )
    assert intent.audit_nonce
    assert any("asserted and unverified locally" in message for message in messages)
    assert _mutations(runner) == []


def test_voluntary_and_plan_limited_still_require_the_explicit_waiver(tmp_path):
    voluntary = CloudSessionRunner()
    with pytest.raises(AgentLoopError, match="--allow-unprotected-managed-ci") as exc_info:
        preflight_managed_ci_creation(
            voluntary, config=_cloud_config(tmp_path, companion=False, unreadable=False),
            issue_number=643,
        )
    assert "--allow-unreadable-protection" not in str(exc_info.value)
    intent = preflight_managed_ci_creation(
        voluntary, config=_cloud_config(tmp_path, unreadable=False), issue_number=643,
    )
    assert intent is not None and intent.protection_mode == "voluntary" and intent.audit_nonce

    plan_limited = CloudSessionRunner(
        repo_payload={"private": True},
        pr_branch_protection_returncode=1,
        pr_branch_protection_stderr=PLAN_LIMITED,
        pr_effective_rules_returncode=1,
        pr_effective_rules_stderr=PLAN_LIMITED,
    )
    with pytest.raises(AgentLoopError, match="--allow-unprotected-managed-ci"):
        preflight_managed_ci_creation(
            plan_limited, config=_cloud_config(tmp_path, companion=False, unreadable=False),
            issue_number=643,
        )
    assert _mutations(voluntary) == [] and _mutations(plan_limited) == []

    # The legacy non-suppressing workflow exception is a separate clause.
    legacy = CloudSessionRunner(workflow=V2_WORKFLOW)
    legacy_intent = preflight_managed_ci_creation(
        legacy, config=_cloud_config(tmp_path, companion=False, unreadable=False),
        issue_number=643,
    )
    assert legacy_intent is not None
    assert legacy_intent.audit_nonce is None


def _unreadable_record(**overrides):
    values = dict(
        kind="creation", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="abc123", actor_login="agent-loop", actor_id=1,
        protection="unreadable", waiver="allow-unreadable-protection",
        nonce="opening-nonce", label_event_id=101,
    )
    values.update(overrides)
    return ManagedCiIssueAuthorization(**values)


def _encoded_authorization(payload):
    encoded = managed_ci._encode_issue_authorization_payload(payload)
    return f"<!-- {managed_ci.ISSUE_AUTHORIZATION_MARKER}: {encoded} -->"


@pytest.mark.parametrize(
    ("protection", "waiver", "valid"),
    [
        ("unreadable", "allow-unreadable-protection", True),
        ("voluntary", "allow-unprotected-managed-ci", True),
        ("plan_limited", "allow-unprotected-managed-ci", True),
        ("unreadable", "allow-unprotected-managed-ci", False),
        ("voluntary", "allow-unreadable-protection", False),
        ("plan_limited", "allow-unreadable-protection", False),
        ("strict", "allow-unprotected-managed-ci", False),
    ],
)
def test_issue_authorization_parser_derives_waiver_from_protection(protection, waiver, valid):
    payload = _unreadable_record(protection=protection, waiver=waiver).to_payload()
    body = _encoded_authorization(payload)

    if valid:
        parsed = parse_issue_created_authorization_comment(body)
        assert parsed is not None
        assert (parsed.protection, parsed.waiver) == (protection, waiver)
    else:
        with pytest.raises(AgentLoopError, match="protection or waiver"):
            parse_issue_created_authorization_comment(body)


def test_unreadable_creation_authorization_round_trips_through_resume_audit(tmp_path):
    runner = CloudAuthorizationRunner(scripted=_cloud_scripts(), issue_events=[label_event()])
    both = _cloud_config(tmp_path, managed_ci_pr_mode=True)
    handoff = publish_issue_created_authorization(
        runner, config=both,
        handoff=replace(_authorization_handoff(), protection_mode="unreadable"),
        metadata=metadata(),
    )

    record = parse_issue_created_authorization_comment(runner.intent_comments[-1]["body"])
    assert record is not None
    assert (record.protection, record.waiver) == ("unreadable", "allow-unreadable-protection")

    def audit(config, expected_protection):
        return _find_resume_audit(
            runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
            base_ref="main", issue_number=643, live_head="abc123",
            expected_handoff=handoff, expected_protection=expected_protection,
            require_actor_owned_label_event=True,
        )

    found = audit(both, "unreadable")
    assert found is not None
    assert found[1]["protection"] == "unreadable"
    # Without the explicit flag an unreadable record is never honored.
    assert audit(_cloud_config(tmp_path, unreadable=False, managed_ci_pr_mode=True), "unreadable") is None
    # A live state change is never silently accepted.
    assert audit(both, "voluntary") is None


def test_fresh_authorization_for_unreadable_protection_requires_both_waivers(tmp_path):
    runner = CloudAuthorizationRunner(scripted=_cloud_scripts(), issue_events=[label_event()])
    fresh_metadata = replace(metadata(), head_branch="agent-loop/managed-643", body="Fixes #643")

    with pytest.raises(AgentLoopError, match=_BOTH_WAIVERS):
        authorize_fresh_issue_created_resume(
            runner, config=_cloud_config(tmp_path, unreadable=False),
            pr_number=7, issue_number=643, metadata=fresh_metadata,
        )
    assert runner.intent_comments == []

    config = _cloud_config(tmp_path)
    first = authorize_fresh_issue_created_resume(
        runner, config=config, pr_number=7, issue_number=643, metadata=fresh_metadata,
    )
    second = authorize_fresh_issue_created_resume(
        runner, config=config, pr_number=7, issue_number=643, metadata=fresh_metadata,
    )

    assert first.protection_mode == "unreadable"
    assert first.authorization_comment_id == second.authorization_comment_id
    record = parse_issue_created_authorization_comment(runner.intent_comments[-1]["body"])
    assert record is not None
    assert (record.kind, record.protection, record.waiver) == (
        "fresh", "unreadable", "allow-unreadable-protection",
    )


def test_override_activation_audits_unreadable_protection_under_the_unchanged_schema(tmp_path):
    nonce = "nonce-from-preflight"
    runner = CloudSessionRunner(
        scripted=_cloud_scripts(),
        rest_pr={"body": f"{UNPROTECTED_OVERRIDE_TRAILER} nonce={nonce}"},
    )
    config = _cloud_config(
        tmp_path, managed_ci=False, auto_merge=True,
        managed_ci_expected_override_nonce=nonce,
    )

    contract = activate_managed_ci(runner, config=config, pr_number=7, metadata=metadata())

    assert contract is not None
    assert contract.protection_mode == "unreadable"
    assert contract.audit_nonce == nonce
    audit_body = runner.audit_comments[contract.audit_comment_id]["body"]
    parsed = managed_ci._parse_override_audit(audit_body)
    assert parsed is not None
    assert parsed["protection"] == "unreadable"
    assert "waiver" not in parsed

    refused = CloudSessionRunner(
        scripted=_cloud_scripts(),
        rest_pr={"body": f"{UNPROTECTED_OVERRIDE_TRAILER} nonce={nonce}"},
    )
    assert activate_managed_ci(
        refused, config=replace(config, allow_unreadable_protection=False),
        pr_number=7, metadata=metadata(),
    ) is None
    assert any(
        command[:5] == ["gh", "api", "--method", "DELETE", f"repos/OWNER/REPO/issues/7/labels/{MANAGED_LABEL}"]
        for command, _ in refused.commands
    )
    assert refused.audit_comments == {}


def test_resume_activation_refuses_unreadable_without_its_waiver(tmp_path):
    runner = CloudSessionRunner(
        scripted=_cloud_scripts(),
        rest_pr={"state": "open", "draft": True, "labels": []},
        issue_events=[label_event()],
    )
    config = _cloud_config(tmp_path, unreadable=False, managed_ci_pr_mode=True)

    with pytest.raises(AgentLoopError, match=_BOTH_WAIVERS):
        activate_managed_ci(
            runner, config=config, pr_number=7,
            metadata=replace(_ready_issue_metadata(), head_sha="abc123"),
            managed_resume=AuthenticatedManagedResume(
                origin="issue-created", lifecycle="draft-unlabeled-reentry",
                issue_created_handoff=replace(
                    _authorization_handoff(), protection_mode="unreadable",
                ),
            ),
        )

    assert runner.labels_posted is False
    assert runner.dispatch_count == 0
    assert _mutations(runner) == []


@pytest.mark.parametrize(
    ("protection_state", "unreadable", "fresh_expected"),
    [
        ("unreadable", False, False),
        ("unreadable", True, True),
        ("voluntary", False, True),
    ],
)
def test_activation_fresh_advice_follows_the_state_specific_waiver(
    tmp_path, protection_state, unreadable, fresh_expected
):
    runner = V2ManagedRunner(issue_events=[label_event()])
    argv = [
        "agent-loop", "pr", "7", "--managed-ci",
        "--managed-ci-trusted-actor", "agent-loop", "--allow-unprotected-managed-ci",
    ]
    if unreadable:
        argv.append("--allow-unreadable-protection")
    config = _cloud_config(
        tmp_path, unreadable=unreadable, managed_ci_pr_mode=True, invocation_argv=tuple(argv),
    )

    with pytest.raises(AgentLoopError) as exc_info:
        _release_for_ordinary_recovery(
            runner,
            config=config,
            pr_number=7,
            base_ref="main",
            expected_head_sha="abc123",
            active_event=(101, "agent-loop", 1),
            reason="no fully bound actor-owned issue-created authorization reaches the live head",
            recovery_capable=True,
            fresh_issue_number=643,
            fresh_authorization_allowed=True,
            protection_state=protection_state,
        )

    message = str(exc_info.value)
    if fresh_expected:
        command = message.split("`", 2)[1]
        parsed = build_parser().parse_args(shlex.split(command)[1:])
        assert parsed.managed_ci_fresh_authorization is True
        assert parsed.allow_unprotected_managed_ci is True
        assert parsed.allow_unreadable_protection is unreadable
    else:
        assert "--managed-ci-fresh" not in message
        assert "fresh authorization is unavailable without" in message
        assert _BOTH_WAIVERS in message


def test_resume_rendering_keeps_or_strips_the_unreadable_waiver(tmp_path):
    argv = (
        "agent-loop", "issue", "1040", "--repo", "OWNER/REPO", "--managed-ci",
        "--managed-ci-trusted-actor", "agent-loop", "--allow-unprotected-managed-ci",
        "--allow-unreadable-protection",
    )
    config = _cloud_config(tmp_path, invocation_argv=argv)
    parser = build_parser()

    managed = render_managed_ci_resume_command(
        config, pr_number=7, issue_number=1040, managed_ci=True,
    )
    managed_args = parser.parse_args(shlex.split(managed)[1:])
    assert managed_args.allow_unprotected_managed_ci is True
    assert managed_args.allow_unreadable_protection is True

    fresh = render_managed_ci_resume_command(
        config, pr_number=7, issue_number=1040, managed_ci=True, fresh_authorization=True,
    )
    fresh_args = parser.parse_args(shlex.split(fresh)[1:])
    assert fresh_args.managed_ci_fresh_authorization is True
    assert fresh_args.allow_unreadable_protection is True

    ordinary = render_managed_ci_resume_command(
        config, pr_number=7, issue_number=1040, managed_ci=False,
    )
    ordinary_args = parser.parse_args(shlex.split(ordinary)[1:])
    assert ordinary_args.allow_unprotected_managed_ci is False
    assert ordinary_args.allow_unreadable_protection is False

    no_argv = render_managed_ci_resume_command(
        replace(config, invocation_argv=()), pr_number=7, managed_ci=True,
    )
    assert _BOTH_WAIVERS in no_argv


def _predicate_comment(**user):
    return {"id": 41, "user": {"login": "agent-loop", "id": 1, **user}, "body": ""}


def _predicate(record, comment=None, *, config, **overrides):
    arguments = dict(
        config=config,
        handoff=None,
        actor_login="agent-loop",
        actor_id=1,
        base_ref="main",
        pr_number=7,
        issue_number=643,
        valid_label_event_ids=None,
        check_protection=None,
        plan_scope_authenticated=True,
    )
    arguments.update(overrides)
    return managed_ci._authorization_record_valid_for_handoff(
        record, comment or _predicate_comment(), **arguments
    )


def test_shared_record_predicate_without_a_handoff(tmp_path):
    config = _cloud_config(tmp_path)
    voluntary = _unreadable_record(protection="voluntary", waiver="allow-unprotected-managed-ci")

    assert _predicate(voluntary, config=config) is True
    assert _predicate(_unreadable_record(), config=config) is True
    assert _predicate(replace(voluntary, actor_login="someone-else"), config=config) is False
    assert _predicate(replace(voluntary, actor_id=2), config=config) is False
    assert _predicate(replace(voluntary, waiver="allow-unreadable-protection"), config=config) is False
    assert _predicate(voluntary, config=config, check_protection="plan_limited") is False
    assert _predicate(voluntary, config=config, valid_label_event_ids={999}) is False
    assert _predicate(voluntary, config=config, valid_label_event_ids={101}) is True
    assert _predicate(voluntary, _predicate_comment(login="intruder"), config=config) is False
    assert _predicate(replace(voluntary, repository="OTHER/REPO"), config=config) is False
    assert _predicate(voluntary, config=config, base_ref="release") is False
    assert _predicate(voluntary, config=config, pr_number=8) is False
    assert _predicate(voluntary, config=config, issue_number=644) is False
    # An unreadable record is honored only with its explicit waiver.
    assert _predicate(
        _unreadable_record(), config=_cloud_config(tmp_path, unreadable=False),
    ) is False


def test_handoff_less_resume_audit_rejects_foreign_tuple_records(tmp_path):
    record = _unreadable_record(protection="voluntary", waiver="allow-unprotected-managed-ci")
    config = _cloud_config(tmp_path)
    for foreign in (
        replace(record, repository="OTHER/REPO"),
        replace(record, base_ref="release"),
        replace(record, pr_number=8),
        replace(record, issue_number=644),
    ):
        runner = CloudAuthorizationRunner(
            issue_events=[label_event()],
            intent_comments=[
                {"id": 41, "user": {"login": "agent-loop", "id": 1},
                 "body": str(format_issue_created_authorization_comment(record))},
                {"id": 42, "user": {"login": "agent-loop", "id": 1},
                 "body": str(format_issue_created_authorization_comment(foreign))},
            ],
        )
        assert _find_resume_audit(
            runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
            base_ref="main", issue_number=643,
        ) is None


_RECOVERY_BODY = f"Fixes #643\n\n{UNPROTECTED_OVERRIDE_TRAILER} nonce=opening-nonce"


def _recovery_metadata():
    return _override_metadata(_RECOVERY_BODY)


def _recovery_runner(records, *, scripted=None, lifecycle="draft-labeled", **kwargs):
    comments = [
        {
            "id": 41 + index,
            "user": {"login": "agent-loop", "id": 1},
            "body": str(format_issue_created_authorization_comment(record)),
        }
        for index, record in enumerate(records)
    ]
    draft = lifecycle != "ready-unlabeled"
    labeled = lifecycle == "draft-labeled"
    kwargs.setdefault("issue_events", [label_event()])
    return CloudAuthorizationRunner(
        scripted=_cloud_scripts() if scripted is None else scripted,
        issue_payload={"number": 643},
        pr_payload={
            "number": 7,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/7",
            "title": "Managed CI",
            "body": _RECOVERY_BODY,
            "headRefName": "agent-loop/managed-643",
            "baseRefName": "main",
            "headRefOid": "abc123",
            "comments": [],
            "reviews": [],
        },
        rest_pr={
            "state": "open",
            "draft": draft,
            "labels": [{"name": MANAGED_LABEL}] if labeled else [],
            "body": _RECOVERY_BODY,
        },
        intent_comments=comments,
        **kwargs,
    )


def _recover(runner, config):
    return recover_issue_created_handoff(
        runner, config=config, pr_number=7, metadata=_recovery_metadata(), issue_number=643,
    )


def test_issue_created_recovery_restores_unreadable_protection(tmp_path):
    runner = _recovery_runner([_unreadable_record()])

    handoff = _recover(runner, _cloud_config(tmp_path, managed_ci_pr_mode=True))

    assert handoff is not None
    assert handoff.protection_mode == "unreadable"
    assert _mutations(runner) == []


def test_issue_created_recovery_restores_plan_limited_protection(tmp_path):
    runner = _recovery_runner(
        [_unreadable_record(protection="plan_limited", waiver="allow-unprotected-managed-ci")],
        scripted={_CLASSIC: _failed(PLAN_LIMITED), _RULES: _failed(PLAN_LIMITED)},
    )

    handoff = _recover(runner, _cloud_config(tmp_path, unreadable=False, managed_ci_pr_mode=True))

    assert handoff is not None
    assert handoff.protection_mode == "plan_limited"


def test_issue_created_recovery_without_creation_record_uses_live_waivable_state(tmp_path):
    runner = _recovery_runner([])

    handoff = _recover(runner, _cloud_config(tmp_path, managed_ci_pr_mode=True))

    assert handoff is not None
    assert handoff.protection_mode == "unreadable"


@pytest.mark.parametrize(
    ("records", "scripted", "config_overrides", "match"),
    [
        ([_unreadable_record()], None, {"unreadable": False}, _BOTH_WAIVERS),
        ([_unreadable_record()], {}, {}, "live assessment is voluntary"),
        (
            [
                _unreadable_record(),
                _unreadable_record(protection="plan_limited", waiver="allow-unprotected-managed-ci"),
            ],
            None, {}, "disagree",
        ),
        (
            [
                _unreadable_record(),
                _unreadable_record(
                    kind="fresh", protection="voluntary",
                    waiver="allow-unprotected-managed-ci", nonce="fresh-nonce",
                ),
            ],
            None, {}, "authorization comment 42",
        ),
    ],
    ids=["missing-flag", "live-disagrees", "creation-records-disagree", "fresh-record-disagrees"],
)
def test_issue_created_recovery_refuses_without_mutation(
    tmp_path, records, scripted, config_overrides, match
):
    runner = _recovery_runner(records, scripted=scripted)

    with pytest.raises(AgentLoopError, match=match) as exc_info:
        _recover(runner, _cloud_config(tmp_path, managed_ci_pr_mode=True, **config_overrides))

    assert "The PR was left unchanged" in str(exc_info.value)
    assert _mutations(runner) == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"repository": "OTHER/REPO"},
        {"pr_number": 8},
        {"issue_number": 644},
        {"base_ref": "release"},
        {"actor_login": "someone-else"},
        {"actor_id": 2},
        {"label_event_id": 999},
    ],
)
def test_issue_created_recovery_rejects_invalid_records_without_selecting_protection(
    tmp_path, overrides
):
    runner = _recovery_runner([
        _unreadable_record(),
        _unreadable_record(**overrides),
    ])

    with pytest.raises(AgentLoopError, match="authorization comment 42"):
        _recover(runner, _cloud_config(tmp_path, managed_ci_pr_mode=True))

    assert _mutations(runner) == []


@pytest.mark.parametrize("lifecycle", ["draft-labeled", "draft-unlabeled", "ready-unlabeled"])
def test_issue_created_recovery_rejects_foreign_label_events_in_every_lifecycle(tmp_path, lifecycle):
    runner = _recovery_runner(
        [_unreadable_record(label_event_id=999)], lifecycle=lifecycle,
    )

    with pytest.raises(AgentLoopError, match="authorization comment 41"):
        _recover(runner, _cloud_config(tmp_path, managed_ci_pr_mode=True))

    assert _mutations(runner) == []


def test_issue_created_recovery_refuses_unreadable_label_history(tmp_path):
    scripted = _cloud_scripts()
    scripted["/issues/7/events?per_page=100"] = _failed(GH_500)
    runner = _recovery_runner(
        [_unreadable_record()], scripted=scripted, lifecycle="draft-unlabeled",
    )

    with pytest.raises(AgentLoopError, match="event history could not be inspected"):
        _recover(runner, _cloud_config(tmp_path, managed_ci_pr_mode=True))

    assert _mutations(runner) == []


def test_run_pr_loop_recovers_unreadable_pr_and_activates_with_both_waivers(
    tmp_path, monkeypatch,
):
    runner = _recovery_runner([_unreadable_record()])
    config = _cloud_config(
        tmp_path,
        managed_ci_pr_mode=True,
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci",
            "--managed-ci-trusted-actor", "agent-loop",
            "--allow-unprotected-managed-ci", "--allow-unreadable-protection",
        ),
    )
    captured = {}
    _stop_after_real_activation(monkeypatch, captured)

    with pytest.raises(_ActivationReached) as exc_info:
        orchestrator.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)

    contract = exc_info.value.args[0]
    assert contract is not None
    assert contract.activation_path == "managed"
    assert contract.protection_mode == "unreadable"
    resumed = captured["managed_resume"]
    assert resumed.issue_created_handoff.protection_mode == "unreadable"
    assert any(
        UNPROTECTED_OVERRIDE_TRAILER in " ".join(command)
        and "protection=unreadable" in " ".join(command)
        for command, _cwd in runner.commands
    )
    assert runner.dispatch_count == 0


def _no_lifecycle_writes(runner):
    commands = [command for command, _cwd in runner.commands]
    assert runner.labels_posted is False
    assert runner.dispatch_count == 0
    assert _mutations(runner) == []
    assert not any(command[:3] == ["gh", "pr", "ready"] for command in commands)


@pytest.mark.parametrize("lifecycle", ["draft-labeled", "draft-unlabeled", "ready-unlabeled"])
def test_recovery_refuses_a_base_that_became_strict_before_any_mutation(tmp_path, lifecycle):
    # A foreign plan hash passes recovery's deferred plan check; the strict
    # activation path would never run the resume-audit gate that checks it.
    runner = _recovery_runner(
        [_unreadable_record(approved_plan_hash="b" * 64)],
        scripted={_VARIABLE: _failed(PROXY_403)},
        lifecycle=lifecycle,
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT]},
    )

    with pytest.raises(AgentLoopError, match="live assessment is strict") as exc_info:
        _recover(runner, _cloud_config(tmp_path, managed_ci_pr_mode=True))

    assert "The PR was left unchanged" in str(exc_info.value)
    _no_lifecycle_writes(runner)


def test_run_pr_loop_refuses_strict_transition_with_foreign_plan_hash(tmp_path, monkeypatch):
    runner = _recovery_runner(
        [_unreadable_record(approved_plan_hash="b" * 64)],
        scripted={_VARIABLE: _failed(PROXY_403)},
        lifecycle="draft-unlabeled",
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT]},
    )
    config = _cloud_config(
        tmp_path,
        managed_ci_pr_mode=True,
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci",
            "--managed-ci-trusted-actor", "agent-loop",
            "--allow-unprotected-managed-ci", "--allow-unreadable-protection",
        ),
    )
    _stop_after_real_activation(monkeypatch)

    with pytest.raises(AgentLoopError, match="live assessment is strict"):
        orchestrator.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)

    _no_lifecycle_writes(runner)


def test_recovered_record_with_foreign_plan_hash_fails_the_activation_gate(tmp_path):
    runner = _recovery_runner(
        [_unreadable_record(approved_plan_hash="b" * 64)], lifecycle="draft-unlabeled",
    )
    config = _cloud_config(tmp_path, managed_ci_pr_mode=True)
    handoff = _recover(runner, config)
    assert handoff is not None and handoff.protection_mode == "unreadable"

    with pytest.raises(AgentLoopError, match="no fully bound actor-owned issue-created authorization"):
        activate_managed_ci(
            runner, config=config, pr_number=7, metadata=_recovery_metadata(),
            managed_resume=AuthenticatedManagedResume(
                origin="issue-created", lifecycle=handoff.lifecycle,
                issue_created_handoff=handoff,
            ),
        )

    assert runner.labels_posted is False
    assert runner.dispatch_count == 0
    assert _mutations(runner) == []


# --- known host comment footer (#1043) ---------------------------------------

HOST_FOOTER = managed_ci.KNOWN_HOST_COMMENT_FOOTER
_NEAR_MISS_SUFFIXES = (
    HOST_FOOTER + HOST_FOOTER,
    HOST_FOOTER + "\n",
    "\n\n---\n_Generated by [Claude Code](http://claude.ai/code)_",
    "\n\nextra prose",
)


def _footer_authorization_comment(body, *, comment_id=31):
    return {"id": comment_id, "user": {"login": "agent-loop", "id": 1}, "body": body}


def _read_authorization_records(tmp_path, comments):
    runner = V2ManagedRunner(intent_comments=comments)
    return managed_ci._authorization_comment_records(
        runner, config=make_config(tmp_path, managed_ci_trusted_actor="agent-loop"),
        pr_number=7, actor_login="agent-loop", actor_id=1,
    )


@pytest.mark.parametrize("kind", ["creation", "fresh", "continuity"])
def test_footered_authorization_authenticates_like_a_clean_record(tmp_path, kind):
    record = _authorization_record(kind)
    body = str(format_issue_created_authorization_comment(record))

    clean = _read_authorization_records(tmp_path, [_footer_authorization_comment(body)])
    footered = _read_authorization_records(
        tmp_path, [_footer_authorization_comment(body + HOST_FOOTER)]
    )

    assert clean == footered == [(31, record)]


@pytest.mark.parametrize("suffix", _NEAR_MISS_SUFFIXES)
def test_authorization_readers_reject_doubled_or_variant_footers(tmp_path, suffix):
    body = str(format_issue_created_authorization_comment(_authorization_record("creation")))

    with pytest.raises(AgentLoopError, match="authorization comment is malformed"):
        _read_authorization_records(tmp_path, [_footer_authorization_comment(body + suffix)])


def test_authorization_reader_rejects_footer_before_the_marker(tmp_path):
    record = _authorization_record("creation")
    label, _, marker = str(format_issue_created_authorization_comment(record)).partition("\n\n")
    misplaced = label + HOST_FOOTER + "\n\n" + marker

    with pytest.raises(AgentLoopError, match="authorization comment is malformed"):
        _read_authorization_records(tmp_path, [_footer_authorization_comment(misplaced)])


def test_authorization_parser_authenticates_the_whole_body_and_never_strips():
    record = _authorization_record("fresh")
    body = str(format_issue_created_authorization_comment(record))

    assert parse_issue_created_authorization_comment(body) == record
    # The pre-#878 marker-only rendering is still accepted, byte for byte.
    assert parse_issue_created_authorization_comment(_authorization_marker(record)) == record
    for tampered in (body + HOST_FOOTER, "Extra prose\n\n" + body, body + "\n", " " + body):
        with pytest.raises(AgentLoopError, match="malformed"):
            parse_issue_created_authorization_comment(tampered)


def test_authenticated_comment_ingestion_strips_one_footer_before_authorization_parse(tmp_path):
    import coding_review_agent_loop.github as github_module

    record = _authorization_record("creation")
    body = str(format_issue_created_authorization_comment(record))
    config = make_config(tmp_path)

    def envelope(text):
        return github_module._authenticated_comment_from_rest(
            {
                "id": 31, "body": text, "user": {"login": "agent-loop", "id": 1},
                "created_at": "2026-09-25T00:00:00Z", "updated_at": "2026-09-25T00:00:00Z",
            },
            surface="pr#7",
            config=config,
        )

    assert parse_issue_created_authorization_comment(envelope(body + HOST_FOOTER).body) == record
    doubled = envelope(body + HOST_FOOTER + HOST_FOOTER)
    assert doubled.body == body + HOST_FOOTER
    with pytest.raises(AgentLoopError, match="malformed"):
        parse_issue_created_authorization_comment(doubled.body)


class HostFooterRunner(V2ManagedRunner):
    """A host that appends the known footer to chosen intent writes.

    ``footer_states`` names the intent lifecycle states whose write the host
    footers.  ``server_clock`` adds an ``updated_at`` to each write response;
    ``envelope_created_at`` adds the comment's own creation time.
    """

    def __init__(
        self, *, footer_states=(), server_clock=None, envelope_created_at=None,
        after_write=None, **kwargs,
    ):
        super().__init__(**kwargs)
        self.footer_states = set(footer_states)
        self.server_clock = server_clock
        self.envelope_created_at = envelope_created_at
        self.after_write = after_write
        self.release_calls = []

    def _run_locked(self, args, *, cwd, check, input_text=None):
        result = super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)
        cmd = list(args)
        endpoint = next((part for part in cmd if isinstance(part, str) and part.startswith("repos/")), "")
        body = self._form_value(cmd, "body")
        writing = body is not None and (
            endpoint == "repos/OWNER/REPO/issues/7/comments"
            or endpoint.startswith("repos/OWNER/REPO/issues/comments/")
        )
        if not writing or "AGENT_MANAGED_CI_INTENT_V2" not in body:
            return result
        envelope = json.loads(result.stdout)
        state = json.loads(body.split("AGENT_MANAGED_CI_INTENT_V2 ", 1)[1].rsplit(" -->", 1)[0])["state"]
        if state in self.footer_states:
            envelope["body"] = body + HOST_FOOTER
            for comment in self.intent_comments:
                if comment.get("id") == envelope["id"]:
                    comment["body"] = body + HOST_FOOTER
        if self.server_clock is not None:
            stamp = datetime.fromtimestamp(self.server_clock(), tz=timezone.utc)
            envelope["updated_at"] = stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        if self.envelope_created_at is not None:
            stamp = datetime.fromtimestamp(self.envelope_created_at, tz=timezone.utc)
            envelope["created_at"] = stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        if self.after_write is not None:
            self.after_write(self, state)
        return CommandResult(result.args, result.cwd, json.dumps(envelope), "", 0)


def _spy_release(monkeypatch):
    calls = []

    def release(*args, **kwargs):
        calls.append(kwargs)
        raise AssertionError("terminal managed-CI errors must not reach ordinary recovery")

    monkeypatch.setattr(managed_ci, "_release_for_ordinary_recovery", release)
    return calls


def _intent_posts(runner):
    return [
        cmd for cmd, _cwd in runner.commands
        if "repos/OWNER/REPO/issues/7/comments" in cmd and "POST" in cmd
    ]


def _authorized_resume():
    return AuthenticatedManagedResume(
        origin="issue-created", lifecycle="draft-labeled",
        issue_created_handoff=_authorization_handoff(head=V2_HEAD),
    )


def test_rediscovery_adopts_a_footered_intent_only_when_the_workflow_admits_it(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    comment = v2_intent_comment(suffix=HOST_FOOTER)

    capable = valid_v2_contract(host_footer_capable=True)
    _ensure_v2_intent(
        valid_v2_runner(intent_comments=[dict(comment)]), config=config, pr_number=7,
        expected_head_sha=V2_HEAD, contract=capable,
    )
    assert (capable.intent_comment_id, capable.nonce) == (17, V2_NONCE)

    older = valid_v2_contract()
    runner = valid_v2_runner(intent_comments=[dict(comment)])
    _ensure_v2_intent(runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=older)
    # An older workflow would skip the footered comment, so it is not adopted.
    assert older.intent_comment_id != 17 and older.nonce != V2_NONCE
    assert len(_intent_posts(runner)) == 1


@pytest.mark.parametrize("suffix", [HOST_FOOTER + HOST_FOOTER, HOST_FOOTER + "\n", HOST_FOOTER + " "])
def test_rediscovery_skips_doubled_and_whitespace_footers(tmp_path, suffix):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = valid_v2_runner(intent_comments=[v2_intent_comment(suffix=suffix)])
    contract = valid_v2_contract(host_footer_capable=True)

    _ensure_v2_intent(runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract)

    assert contract.intent_comment_id != 17 and contract.nonce != V2_NONCE


@pytest.mark.parametrize(
    ("comments", "reason"),
    [
        (
            [v2_intent_comment(comment_id=17), v2_intent_comment(comment_id=18)],
            "expected exactly one fresh intent",
        ),
        (
            [
                v2_intent_comment(comment_id=17),
                {**v2_intent_comment(comment_id=18), "body": v2_intent_comment()["body"].replace(
                    '"version": 2', '"version": 2, "extra": 1'
                )},
            ],
            "invalid schema",
        ),
        (
            [{**v2_intent_comment(), "body": (
                f"Managed CI authorization for exact head {'c' * 40}.\n\n" + v2_intent_comment()["body"]
            )}],
            "visible intent line disagrees",
        ),
        ([v2_intent_comment(nonce="short")], "invalid nonce"),
    ],
)
def test_rediscovery_fails_closed_on_a_candidate_the_workflow_fails(tmp_path, monkeypatch, comments, reason):
    config = make_config(tmp_path, auto_merge=True, managed_ci_pr_mode=True, managed_ci_trusted_actor="agent-loop")
    releases = _spy_release(monkeypatch)
    runner = valid_v2_runner(intent_comments=comments)
    contract = valid_v2_contract(visible_intent_capable=True)

    with pytest.raises(managed_ci.ManagedCiIntentLedgerError, match=reason):
        _dispatch_v2_qualification(
            runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract
        )

    assert _intent_posts(runner) == []
    assert runner.dispatch_count == 0
    assert releases == []


@pytest.mark.parametrize(
    ("entry", "reason"),
    [
        ({"id": 40, "user": {"id": 5}, "body": "hello"}, "malformed comment author"),
        ({"id": 40, "user": "someone", "body": "hello"}, "malformed comment author"),
        ({"id": 40, "user": {"login": "agent-loop", "id": 99}, "body": "hello"}, "author ID drifted"),
        ({"id": 40, "user": {"login": "agent-loop", "id": 1}, "body": None}, "body is not text"),
        (
            {"id": 40, "user": {"login": "agent-loop", "id": 1},
             "body": "<!-- AGENT_MANAGED_CI_INTENT_V2 {not json} -->"},
            "malformed JSON",
        ),
        (
            {"id": 40, "user": {"login": "agent-loop", "id": 1},
             "body": "<!-- AGENT_MANAGED_CI_INTENT_V2 [1] -->"},
            "not an object",
        ),
        (
            {"id": 40, "user": {"login": "agent-loop", "id": 1},
             "body": "<!-- AGENT_MANAGED_CI_INTENT_V2 {\"version\": 1} -->"},
            "unsupported intent version",
        ),
    ],
)
def test_nonce_independent_fatal_page_posts_nothing(tmp_path, entry, reason):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = valid_v2_runner(intent_comments=[entry])

    with pytest.raises(managed_ci.ManagedCiIntentLedgerError, match=reason):
        _ensure_v2_intent(
            runner, config=config, pr_number=7, expected_head_sha=V2_HEAD,
            contract=valid_v2_contract(),
        )
    assert _intent_posts(runner) == []


def test_restore_rejects_a_malformed_nonce():
    with pytest.raises(managed_ci.ManagedCiIntentLedgerError, match="malformed nonce"):
        managed_ci._restore_v2_intent_fields(valid_v2_contract(), {"nonce": "nonce-1"})


@pytest.mark.parametrize("mode", ["managed-pr", "issue-created-resume"])
def test_non_dict_page_entry_reaches_rediscovery_through_the_api_seam(tmp_path, monkeypatch, mode):
    config = make_config(
        tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop",
        managed_ci_pr_mode=mode == "managed-pr",
    )
    releases = _spy_release(monkeypatch)
    runner = valid_v2_runner(intent_comments=["not a comment", v2_intent_comment()])
    contract = valid_v2_contract(
        authenticated_resume=_authorized_resume() if mode == "issue-created-resume" else None,
    )

    with pytest.raises(managed_ci.ManagedCiIntentLedgerError, match="malformed comment object"):
        _dispatch_v2_qualification(
            runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract
        )

    assert _intent_posts(runner) == []
    assert runner.dispatch_count == 0
    assert releases == []


def test_non_dict_entry_appearing_before_the_gate_blocks_dispatch(tmp_path, monkeypatch):
    config = make_config(tmp_path, auto_merge=True, managed_ci_pr_mode=True, managed_ci_trusted_actor="agent-loop")
    releases = _spy_release(monkeypatch)

    def inject(runner, state):
        if state == "dispatch-requested":
            runner.intent_comments.append(None)

    runner = HostFooterRunner(after_write=inject, base_sha=V2_REVISION, pr_payload={"headRefOid": V2_HEAD})
    contract = valid_v2_contract(nonce=None)

    with pytest.raises(managed_ci.ManagedCiIntentLedgerError, match="malformed comment object"):
        _dispatch_v2_qualification(
            runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract
        )

    assert runner.dispatch_count == 0
    assert releases == []


def test_zero_exit_empty_intent_page_is_uninspectable_not_an_empty_ledger(tmp_path, monkeypatch):
    class EmptyBody(V2ManagedRunner):
        def _run_locked(self, args, *, cwd, check, input_text=None):
            endpoint = next((part for part in args if isinstance(part, str) and part.startswith("repos/")), "")
            if endpoint.startswith("repos/OWNER/REPO/issues/7/comments?"):
                cmd, cwd_path = self._record_command(args, cwd)
                return CommandResult(cmd, cwd_path, "", "", 0)
            return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)

    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = EmptyBody(base_sha=V2_REVISION, pr_payload={"headRefOid": V2_HEAD})
    with pytest.raises(AgentLoopError, match="Unable to inspect managed-CI v2 intent history") as raised:
        _ensure_v2_intent(
            runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=valid_v2_contract(),
        )
    assert not isinstance(raised.value, managed_ci.ManagedCiIntentLedgerError)
    assert _intent_posts(runner) == []

    # At the gate, the same response is the generic inspection error, never a
    # ledger verdict, and nothing is dispatched.
    with pytest.raises(AgentLoopError, match="Unable to inspect managed-CI v2 intent history") as raised:
        managed_ci._require_workflow_dispatch_authorization(
            runner, config=config,
            contract=valid_v2_contract(pr_number=7, intent_comment_id=17), patch_result=None,
        )
    assert not isinstance(raised.value, managed_ci.ManagedCiIntentLedgerError)
    assert runner.dispatch_count == 0


def test_footer_seen_only_on_intent_rediscovery_is_logged_once(tmp_path, monkeypatch):
    import coding_review_agent_loop.github as github_module

    lines = []
    monkeypatch.setattr(github_module, "log", lambda _config, message: lines.append(message))
    github_module.reset_host_footer_log_latch()
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = valid_v2_runner(intent_comments=[v2_intent_comment(suffix=HOST_FOOTER)])
    contract = valid_v2_contract(host_footer_capable=True)

    _ensure_v2_intent(runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract)
    _ensure_v2_intent(runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract)

    assert contract.intent_comment_id == 17
    assert _intent_posts(runner) == []
    assert len(lines) == 1 and "intent re-read" in lines[0]
    # The raw page keeps its footer: the envelope mirror owns that form.
    assert runner.intent_comments[0]["body"].endswith(HOST_FOOTER)

    lines.clear()
    github_module.reset_host_footer_log_latch()
    _ensure_v2_intent(
        valid_v2_runner(intent_comments=[v2_intent_comment()]), config=config, pr_number=7,
        expected_head_sha=V2_HEAD, contract=valid_v2_contract(),
    )
    assert lines == []


@pytest.mark.parametrize(
    ("stdout", "returncode"), [("", 1), (json.dumps({"message": "rate limited"}), 0), ("{not json", 0)]
)
def test_uninspectable_intent_page_keeps_the_generic_inspection_error(tmp_path, stdout, returncode):
    class Unreadable(V2ManagedRunner):
        def _run_locked(self, args, *, cwd, check, input_text=None):
            endpoint = next((part for part in args if isinstance(part, str) and part.startswith("repos/")), "")
            if endpoint.startswith("repos/OWNER/REPO/issues/7/comments?"):
                cmd, cwd_path = self._record_command(args, cwd)
                return CommandResult(cmd, cwd_path, stdout, "", returncode)
            return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)

    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    with pytest.raises(AgentLoopError, match="Unable to inspect managed-CI v2 intent history") as raised:
        _ensure_v2_intent(
            Unreadable(), config=config, pr_number=7, expected_head_sha=V2_HEAD,
            contract=valid_v2_contract(),
        )
    assert not isinstance(raised.value, managed_ci.ManagedCiIntentLedgerError)
    # _api_list keeps its contract for every other caller.
    runner = V2ManagedRunner(intent_comments=["not a comment"])
    assert _api_list(runner, config, "repos/OWNER/REPO/issues/7/comments?per_page=100") is None


def _page_for_router(runner):
    return [[comment for comment in runner.intent_comments]]


def _router_pr():
    return {
        "state": "open", "draft": True, "number": 7, "base": {"ref": "main"},
        "head": {"sha": V2_HEAD, "ref": "agent-loop/managed-7", "repo": {"full_name": "OWNER/REPO"}},
        "user": {"login": "agent-loop", "id": 1}, "labels": [{"name": MANAGED_LABEL}],
    }


def _route(runner, nonce):
    return local_router.validate(
        _router_pr(), _page_for_router(runner), "OWNER/REPO", "7", V2_HEAD, nonce,
        "agent-loop", V2_REVISION, 1,
    )


def test_prepared_intent_is_recovered_in_place_then_authorized_at_the_gate(tmp_path, monkeypatch):
    config = make_config(tmp_path, auto_merge=True, managed_ci_pr_mode=True, managed_ci_trusted_actor="agent-loop")
    releases = _spy_release(monkeypatch)
    runner = HostFooterRunner(base_sha=V2_REVISION, pr_payload={"headRefOid": V2_HEAD})
    contract = valid_v2_contract(nonce=None)
    # The first attempt posted its prepared intent and was interrupted before
    # the dispatch-requested PATCH.
    _ensure_v2_intent(runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract)
    prepared_id, nonce = contract.intent_comment_id, contract.nonce
    with pytest.raises(ValueError, match="prepared intent is not a dispatch authorization"):
        _route(runner, nonce)
    verdict = managed_ci.classify_intent_page(
        list(runner.intent_comments), requested_nonce=nonce, trusted_login="agent-loop",
        trusted_id=1, visible_capable=False, host_footer_capable=False,
        binding=managed_ci._intent_binding(contract, nonce=nonce),
    )
    assert verdict.outcome == "recoverable"

    _dispatch_v2_qualification(
        runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract
    )

    assert contract.intent_comment_id == prepared_id and contract.nonce == nonce
    assert len(_intent_posts(runner)) == 1
    assert runner.dispatch_count == 1
    assert contract.dispatch_issued is True
    assert _route(runner, nonce)["state"] == "dispatch-requested"
    assert releases == []


def test_same_nonce_sibling_before_the_gate_blocks_dispatch(tmp_path, monkeypatch):
    config = make_config(tmp_path, auto_merge=True, managed_ci_pr_mode=True, managed_ci_trusted_actor="agent-loop")
    releases = _spy_release(monkeypatch)

    def sibling(runner, state):
        if state == "dispatch-requested":
            original = runner.intent_comments[-1]
            runner.intent_comments.append({**original, "id": original["id"] + 100})

    runner = HostFooterRunner(after_write=sibling, base_sha=V2_REVISION, pr_payload={"headRefOid": V2_HEAD})

    with pytest.raises(managed_ci.ManagedCiIntentLedgerError, match="exactly one fresh intent"):
        _dispatch_v2_qualification(
            runner, config=config, pr_number=7, expected_head_sha=V2_HEAD,
            contract=valid_v2_contract(nonce=None),
        )

    assert runner.dispatch_count == 0
    assert releases == []


def test_adopted_contract_without_generation_recovers_prepared_intent_of_its_ledger(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = valid_v2_runner(intent_comments=[v2_intent_comment(state="prepared", created_at=int(time.time()))])
    contract = valid_v2_contract(intent_generation=None)

    _ensure_v2_intent(runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract)

    assert (contract.intent_comment_id, contract.intent_state) == (17, "prepared")
    assert _intent_posts(runner) == []


def test_fresh_generation_supersedes_an_earlier_prepared_intent(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = HostFooterRunner(
        intent_comments=[v2_intent_comment(state="prepared", generation="earlier-generation")],
        base_sha=V2_REVISION, pr_payload={"headRefOid": V2_HEAD},
    )
    contract = valid_v2_contract(nonce=None, intent_generation="later-generation")

    _dispatch_v2_qualification(
        runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract
    )

    assert contract.intent_comment_id != 17 and contract.nonce != V2_NONCE
    assert runner.dispatch_count == 1
    # The superseded intent is only a nonce mismatch for the new request.
    assert _route(runner, contract.nonce)["generation"] == "later-generation"


NOW = 1_800_000_000


def _dispatch_validate(runner, *, current_time, nonce):
    return dispatch_validator.validate_dispatch(
        protocol="2", pr_number_text="7", expected_head=V2_HEAD, nonce=nonce,
        repo="OWNER/REPO", ref="refs/heads/main", configured_actor="agent-loop",
        initiating_actor="agent-loop", rerun_actor="agent-loop", current_run_id="200",
        current_run_attempt="1", current_time=current_time,
        api_json=lambda path: {
            "users/agent-loop": {"login": "agent-loop", "id": 1},
            "repos/OWNER/REPO": {"full_name": "OWNER/REPO"},
            "repos/OWNER/REPO/pulls/7": _router_pr(),
            "repos/OWNER/REPO/commits/main": {"sha": V2_REVISION},
        }[path],
        api_pages=lambda path: _page_for_router(runner),
        validate=local_router.validate,
    )


def _freshness_run(tmp_path, monkeypatch, *, created_at, local, server, envelope_created_at=None):
    monkeypatch.setattr(managed_ci.time, "time", lambda: local)
    runner = HostFooterRunner(
        intent_comments=[v2_intent_comment(state="prepared", created_at=created_at)],
        server_clock=None if server is None else (lambda: server),
        envelope_created_at=envelope_created_at,
        base_sha=V2_REVISION, pr_payload={"headRefOid": V2_HEAD},
    )
    contract = valid_v2_contract(nonce=None)
    config = make_config(tmp_path, auto_merge=True, managed_ci_pr_mode=True, managed_ci_trusted_actor="agent-loop")
    return runner, contract, config


@pytest.mark.parametrize(
    ("created_at", "control"),
    [(NOW - 1200, "managed intent record is stale"), (NOW + 1200, "too far in the future")],
)
def test_recovered_intent_is_renewed_in_place_before_dispatch(tmp_path, monkeypatch, created_at, control):
    releases = _spy_release(monkeypatch)
    runner, contract, config = _freshness_run(
        tmp_path, monkeypatch, created_at=created_at, local=NOW, server=NOW
    )

    _dispatch_v2_qualification(
        runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract
    )

    assert contract.intent_comment_id == 17 and contract.nonce == V2_NONCE
    assert _intent_posts(runner) == []
    assert runner.dispatch_count == 1
    assert runner.intent_snapshots[-1]["created_at"] == NOW
    assert _dispatch_validate(runner, current_time=NOW + 60, nonce=V2_NONCE)["record"]["created_at"] == NOW
    # Without the renewal the workflow's dispatch validator rejects the record.
    stale = json.loads(json.dumps(runner.intent_comments))
    stale[0]["body"] = stale[0]["body"].replace(f'"created_at":{NOW}', f'"created_at":{created_at}')
    runner.intent_comments = stale
    with pytest.raises(ValueError, match=control):
        _dispatch_validate(runner, current_time=NOW + 60, nonce=V2_NONCE)
    assert releases == []


@pytest.mark.parametrize("skew", [600, -720])
def test_skewed_local_clock_fails_the_freshness_gate_without_dispatch(tmp_path, monkeypatch, skew):
    releases = _spy_release(monkeypatch)
    runner, contract, config = _freshness_run(
        tmp_path, monkeypatch, created_at=NOW, local=NOW + skew, server=NOW
    )

    with pytest.raises(managed_ci.ManagedCiIntentFreshnessError, match="server updated_at"):
        _dispatch_v2_qualification(
            runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract
        )

    assert runner.dispatch_count == 0
    assert releases == []
    if skew > 0:
        with pytest.raises(ValueError, match="too far in the future"):
            _dispatch_validate(runner, current_time=NOW, nonce=V2_NONCE)


def test_patch_without_updated_at_uses_local_time_not_the_comment_creation_time(tmp_path, monkeypatch):
    releases = _spy_release(monkeypatch)
    runner, contract, config = _freshness_run(
        tmp_path, monkeypatch, created_at=NOW - 1200, local=NOW, server=None,
        envelope_created_at=NOW - 1200,
    )

    _dispatch_v2_qualification(
        runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract
    )

    assert runner.dispatch_count == 1
    assert runner.intent_snapshots[-1]["created_at"] == NOW
    assert _dispatch_validate(runner, current_time=NOW + 60, nonce=V2_NONCE)["record"]["state"] == "dispatch-requested"
    # Control: the comment's own creation time would place the renewed record
    # 20 minutes in the future and fail the gate, so it is never used as T.
    assert (NOW - 1200) - NOW < -managed_ci.INTENT_MAX_FUTURE_SKEW_SECONDS
    assert releases == []


def test_attach_path_patch_keeps_created_at(tmp_path):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    runner = valid_v2_runner(
        workflow_runs=[valid_v2_run(run_id=101, status="in_progress", conclusion=None)],
        intent_comments=[v2_intent_comment(
            run_id=100, run_attempt=1, state="completed", created_at=5,
            terminal_run_id=100, terminal_run_attempt=1,
            terminal_attempts=((100, 1),), terminal_outcome="no-status",
        )],
    )

    _dispatch_v2_qualification(
        runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=valid_v2_contract()
    )

    assert runner.dispatch_count == 0
    assert [snapshot["created_at"] for snapshot in runner.intent_snapshots] == [5, 5]


@pytest.mark.parametrize("footered_state", ["prepared", "dispatch-requested"])
def test_older_workflow_guard_stops_before_dispatch(tmp_path, monkeypatch, footered_state):
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")
    releases = _spy_release(monkeypatch)
    runner = HostFooterRunner(
        footer_states={footered_state}, base_sha=V2_REVISION, pr_payload={"headRefOid": V2_HEAD},
    )
    contract = valid_v2_contract(nonce=None, authenticated_resume=_authorized_resume())

    with pytest.raises(managed_ci.ManagedCiHostFooterIncompatibleError) as raised:
        _dispatch_v2_qualification(
            runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract
        )

    assert raised.value.phase == "pre-dispatch"
    assert managed_ci.HOST_FOOTER_INTENT_MARKER in str(raised.value)
    assert "No managed-CI run was dispatched" in str(raised.value)
    assert runner.dispatch_count == 0
    assert contract.dispatch_issued is False
    assert contract.intent_state == ("prepared" if footered_state == "dispatch-requested" else None)
    assert releases == []


def test_older_workflow_guard_after_dispatch_on_the_attached_patch(tmp_path, monkeypatch):
    config = make_config(tmp_path, auto_merge=True, managed_ci_pr_mode=True, managed_ci_trusted_actor="agent-loop")
    releases = _spy_release(monkeypatch)

    def run_appears(runner, state):
        if state == "dispatch-requested":
            runner.workflow_runs = [valid_v2_run(
                run_id=300, status="in_progress", conclusion=None,
                name=f"managed-ci-v2 nonce={runner.intent_snapshots[-1]['nonce']}",
            )]

    runner = HostFooterRunner(
        footer_states={"attached"}, after_write=run_appears,
        base_sha=V2_REVISION, pr_payload={"headRefOid": V2_HEAD},
    )
    contract = valid_v2_contract(nonce=None)

    with pytest.raises(managed_ci.ManagedCiHostFooterIncompatibleError) as raised:
        _dispatch_v2_qualification(
            runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract
        )

    assert raised.value.phase == "post-dispatch"
    assert "is not cancelled" in str(raised.value)
    assert runner.dispatch_count == 1
    assert contract.intent_state == "dispatch-requested"
    assert not any("cancel" in " ".join(cmd) for cmd, _cwd in runner.commands)
    assert releases == []


def test_older_workflow_guard_after_dispatch_claims_no_qualification(tmp_path, monkeypatch):
    config = make_config(tmp_path, auto_merge=True, ci_timeout_seconds=1, ci_poll_interval_seconds=1)
    runner = HostFooterRunner(
        footer_states={"completed"},
        workflow_runs=[valid_v2_run(run_id=100, attempt=1, status="completed", conclusion="success")],
        intent_comments=[v2_intent_comment(run_id=100, run_attempt=1, state="attached")],
        pr_status_payload={"statuses": [{
            "context": FINAL_CONTEXT,
            "state": "success",
            "description": f"nonce={V2_NONCE};run_id=100;attempt=1",
            "target_url": "https://github.com/OWNER/REPO/actions/runs/100",
            "creator": {"login": "github-actions[bot]", "id": 41898282},
        }]},
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT], "checks": []},
        base_sha=V2_REVISION, pr_payload={"headRefOid": V2_HEAD},
    )
    contract = valid_v2_contract()
    _dispatch_v2_qualification(
        runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract
    )
    assert runner.dispatch_count == 0 and contract.attached_run_id == 100

    with pytest.raises(managed_ci.ManagedCiHostFooterIncompatibleError) as raised:
        wait_for_final_qualification(
            runner, config=config, pr_number=7,
            metadata=replace(metadata(), head_sha=V2_HEAD), contract=contract,
        )

    assert raised.value.phase == "post-dispatch"
    assert contract.intent_state != "completed"
    assert runner.dispatch_count == 0
