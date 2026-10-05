import ast
import copy
import json
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import orchestrator_split_guard
import coding_review_agent_loop.managed_ci as managed_ci
import coding_review_agent_loop.orchestrator as orchestrator
from coding_review_agent_loop.workdirs import github_api_cwd

from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import (
    PullRequestCheck,
    PullRequestChecks,
    PullRequestMergeability,
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
    release_retained_managed_label,
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

from agent_loop_helpers import (
    FakeRunner,
    make_config,
    structured_coder_followup,
    structured_pr_review,
)


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
            "state": "open",
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

    def _next_event_id(self) -> int:
        ids = [
            e["id"] for e in self.issue_events
            if isinstance(e, dict) and isinstance(e.get("id"), int)
        ]
        return max(ids, default=100) + 1

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
            # GitHub event ids are unique and increasing.
            self.issue_events.append(label_event(self._next_event_id()))
            return CommandResult(cmd, cwd_path, "{}", "", 0)
        if endpoint == f"repos/OWNER/REPO/issues/7/labels/{MANAGED_LABEL}" and "DELETE" in cmd:
            cmd, cwd_path = self._record_command(args, cwd)
            self.rest_pr["labels"] = []
            self.issue_events.append(label_event(self._next_event_id(), event="unlabeled"))
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
    # head-a is still in head-b's history, so it is a live sibling rather
    # than a discarded one (#1069 retires only the latter).
    runner.compare_payload = {
        "status": "ahead",
        "base_commit": {"sha": "head-a"},
        "merge_base_commit": {"sha": "head-a"},
    }
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


# --- #1024: an exact-head CI repair round authorizes continuity on its own ---


def _advanced_ci_obligation(*, failed="abc123", candidate="ci-fix-head", round_number=12):
    """The CI obligation exactly as the orchestrator serializes it after a push."""
    from coding_review_agent_loop.unresolved_items import (
        MANAGED_CI_OBLIGATION_KIND,
        _advance_machine_obligations_for_head,
        _upsert_machine_obligation,
    )

    minted = _upsert_machine_obligation(
        [],
        item_number=1,
        kind=MANAGED_CI_OBLIGATION_KIND,
        source_round=round_number,
        text="final-ci/exact-head failed.",
        failed_head_sha=failed,
    )
    (advanced,) = _advance_machine_obligations_for_head(minted, current_head_sha=candidate)
    assert advanced.lifecycle == "awaiting_current_head_review"
    assert (advanced.failed_head_sha, advanced.candidate_head_sha) == (failed, candidate)
    return advanced


def _ci_repair_round_comment(comment_id, *, subject="ci-fix-head", round_number=13, item=None,
                             login="agent-loop", actor_id=1):
    item = item if item is not None else _advanced_ci_obligation(candidate=subject)
    return {
        "id": comment_id,
        "user": {"login": login, "id": actor_id},
        "body": _attach_round_metadata(
            "coder round",
            PostedRoundMetadata(
                flow="pr", role="coder", agent="agent-loop",
                round_number=round_number, subject=subject,
                prior_items=(item,),
            ),
        ),
    }


def _ci_repair_continuity(comment_id=61, *, predecessor_comment_id=50):
    return ManagedCiIssueAuthorization(
        kind="continuity", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="ci-fix-head", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="next",
        label_event_id=101, predecessor_head="abc123",
        predecessor_comment_id=predecessor_comment_id, round_comment_ids=(comment_id,),
    )


def test_ci_repair_round_grants_continuity_without_a_reviewer_pair(tmp_path):
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    runner.intent_comments.append(_ci_repair_round_comment(61))

    selected = managed_ci.find_actor_round_metadata_comment_ids(
        runner, config=make_config(tmp_path), pr_number=7, actor_login="agent-loop",
        actor_id=1, predecessor_head="abc123", new_head="ci-fix-head", round_number=12,
        after_comment_id=50,
    )

    assert selected == (61,)
    assert managed_ci._continuity_round_metadata_is_valid(
        [_ci_repair_round_comment(61)], authorization=_ci_repair_continuity()
    ) is True


def test_ci_repair_round_continuity_grant_reauthenticates_on_resume(tmp_path):
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    initial = publish_issue_created_authorization(
        runner, config=config, handoff=_authorization_handoff(), metadata=metadata()
    )
    runner.intent_comments.append(_ci_repair_round_comment(61))
    runner.rest_pr["head"]["sha"] = "ci-fix-head"
    continued = publish_issue_created_continuity_authorization(
        runner, config=config, handoff=initial, predecessor_head="abc123",
        new_head="ci-fix-head", round_comment_ids=(61,),
    )

    audit = _find_resume_audit(
        runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
        base_ref="main", issue_number=643, live_head="ci-fix-head",
    )

    assert continued.authorization_kind == "continuity"
    assert audit is not None
    assert audit[0] == continued.authorization_comment_id
    assert audit[1]["head"] == "ci-fix-head"


def _mismatched_ci_items():
    advanced = _advanced_ci_obligation()
    return {
        "failed-head-mismatch": _advanced_ci_obligation(failed="other-head"),
        "candidate-head-mismatch": replace(advanced, candidate_head_sha="another-head"),
        "repair-required": replace(advanced, lifecycle="repair_required"),
        "cleared": replace(advanced, lifecycle="cleared"),
        "qualification-ready": replace(advanced, lifecycle="qualification_ready"),
        "unknown-authority": replace(advanced, authority="unknown"),
        "non-ci-kind": replace(advanced, obligation_kind="alembic-migration"),
        "resolved-status": replace(advanced, status="resolved"),
        "future-status": replace(advanced, status="future"),
    }


@pytest.mark.parametrize("case", sorted(_mismatched_ci_items()))
def test_ci_repair_continuity_refuses_an_unbound_ci_obligation(tmp_path, case):
    item = _mismatched_ci_items()[case]
    comment = _ci_repair_round_comment(61, item=item)
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    runner.intent_comments.append(comment)

    with pytest.raises(AgentLoopError, match="correlated blocking-review and coder"):
        managed_ci.find_actor_round_metadata_comment_ids(
            runner, config=make_config(tmp_path), pr_number=7, actor_login="agent-loop",
            actor_id=1, predecessor_head="abc123", new_head="ci-fix-head", round_number=12,
            after_comment_id=50,
        )
    assert managed_ci._continuity_round_metadata_is_valid(
        [comment], authorization=_ci_repair_continuity()
    ) is False


def test_ci_repair_continuity_refuses_two_coder_records(tmp_path):
    comments = [_ci_repair_round_comment(61), _ci_repair_round_comment(62)]
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    runner.intent_comments.extend(comments)

    with pytest.raises(AgentLoopError, match="correlated blocking-review and coder"):
        managed_ci.find_actor_round_metadata_comment_ids(
            runner, config=make_config(tmp_path), pr_number=7, actor_login="agent-loop",
            actor_id=1, predecessor_head="abc123", new_head="ci-fix-head", round_number=12,
            after_comment_id=50,
        )
    two = replace(_ci_repair_continuity(), round_comment_ids=(61, 62))
    assert managed_ci._continuity_round_metadata_is_valid(comments, authorization=two) is False


def test_ci_repair_continuity_refuses_a_foreign_actor(tmp_path):
    comment = _ci_repair_round_comment(61, login="someone-else", actor_id=2)
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    runner.intent_comments.append(comment)

    with pytest.raises(AgentLoopError, match="correlated blocking-review and coder"):
        managed_ci.find_actor_round_metadata_comment_ids(
            runner, config=make_config(tmp_path), pr_number=7, actor_login="agent-loop",
            actor_id=1, predecessor_head="abc123", new_head="ci-fix-head", round_number=12,
            after_comment_id=50,
        )
    assert managed_ci._continuity_round_metadata_is_valid(
        [comment], authorization=_ci_repair_continuity()
    ) is False


def test_ci_repair_continuity_refuses_a_record_older_than_the_predecessor(tmp_path):
    comment = _ci_repair_round_comment(41)
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    runner.intent_comments.append(comment)

    with pytest.raises(AgentLoopError, match="correlated blocking-review and coder"):
        managed_ci.find_actor_round_metadata_comment_ids(
            runner, config=make_config(tmp_path), pr_number=7, actor_login="agent-loop",
            actor_id=1, predecessor_head="abc123", new_head="ci-fix-head", round_number=12,
            after_comment_id=50,
        )
    assert managed_ci._continuity_round_metadata_is_valid(
        [comment], authorization=_ci_repair_continuity(41)
    ) is False


def _authorization_records(runner):
    return [
        (comment["id"], record)
        for comment in runner.intent_comments
        if (record := parse_issue_created_authorization_comment(comment.get("body", "")))
        is not None
    ]


def _invalid_ci_repair_publication_cases():
    cases = {
        name: [_ci_repair_round_comment(61, item=item)]
        for name, item in _mismatched_ci_items().items()
    }
    cases["foreign-actor"] = [
        _ci_repair_round_comment(61, login="someone-else", actor_id=2)
    ]
    cases["two-coders"] = [_ci_repair_round_comment(61), _ci_repair_round_comment(62)]
    return cases


@pytest.mark.parametrize("case", sorted(_invalid_ci_repair_publication_cases()))
def test_ci_repair_continuity_publication_writes_nothing_for_an_invalid_record(tmp_path, case):
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    initial = publish_issue_created_authorization(
        runner, config=config, handoff=_authorization_handoff(), metadata=metadata()
    )
    comments = _invalid_ci_repair_publication_cases()[case]
    runner.intent_comments.extend(comments)
    runner.rest_pr["head"]["sha"] = "ci-fix-head"
    before = _authorization_records(runner)

    with pytest.raises(AgentLoopError, match="authenticated, correlated round metadata"):
        publish_issue_created_continuity_authorization(
            runner, config=config, handoff=initial, predecessor_head="abc123",
            new_head="ci-fix-head",
            round_comment_ids=tuple(comment["id"] for comment in comments),
        )

    assert _authorization_records(runner) == before
    assert [record.kind for _comment_id, record in before] == ["creation"]


class _OrchestratorRoundCommentRunner(AuthorizationCommentRunner):
    """Mirror orchestrator-posted PR comments into the actor's REST comment list.

    The coder round metadata the orchestrator serializes is exactly what the
    real continuity selection and publication then read back (#1024).
    """

    def _run_locked(self, args, *, cwd, check, input_text=None):
        cmd = [str(arg) for arg in args]
        self.rest_pr["head"]["sha"] = self.pr_payload.get("headRefOid", self.rest_pr["head"]["sha"])
        result = super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)
        if cmd[:3] == ["gh", "pr", "comment"]:
            comment_id = max(
                (int(item.get("id", 0)) for item in self.intent_comments), default=16
            ) + 1
            self.intent_comments.append({
                "id": comment_id,
                "user": {"login": self.actor_login, "id": self.actor_id},
                "body": self.pr_payload["comments"][-1]["body"],
            })
        return result


def test_issue_ci_failure_after_full_approval_repairs_and_merges_in_one_run(
    tmp_path, monkeypatch, capsys,
):
    """#1024: approve, CI fails, coder repairs, real continuity, re-approve, merge."""
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True, reviewer=("codex",), auto_merge=True,
        max_rounds=1, quiet=False,
    )
    runner = _OrchestratorRoundCommentRunner(issue_events=[label_event()])
    runner.pr_payload.update({
        "headRefName": "agent-loop/managed-643", "headRefOid": "abc123",
        "baseRefName": "main", "body": runner.rest_pr["body"],
    })
    root = publish_issue_created_authorization(
        runner, config=config, handoff=_authorization_handoff(), metadata=metadata()
    )
    runner.codex_outputs = [
        structured_pr_review(state="approved", summary="Approved."),
        structured_pr_review(
            state="approved", summary="Approved after managed CI fix.",
            prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        ),
    ]
    runner.claude_outputs = [
        structured_coder_followup(
            state="blocking", summary="Fixed managed CI.", addressed_items=["item-1"],
        )
    ]
    failed_check = PullRequestCheck(
        name="final-ci/exact-head", kind="check_run", status="failure",
        url="https://github.com/OWNER/REPO/actions/runs/555",
    )
    outcomes = iter([
        managed_ci.ManagedCiOutcome(
            status="failed", head_sha="abc123",
            checks=PullRequestChecks(
                state="failing", required_checks=("final-ci/exact-head",),
                passing=(), pending=(), failing=(failed_check,), missing_required=(),
                branch_protection_status="configured", check_query_status="ok",
            ),
        ),
        managed_ci.ManagedCiOutcome(status="passed", head_sha="abc123-coder-1"),
    ])
    qualified_heads = []
    merges = []
    handoffs = []
    monkeypatch.setattr(orchestrator, "revalidate_issue_created_handoff", lambda *_a, **_k: root)
    monkeypatch.setattr(
        orchestrator, "activate_managed_ci",
        lambda *_a, **_k: ManagedCiContract(protocol_version=2, issue_created_pr=True),
    )
    monkeypatch.setattr(orchestrator, "revalidate_adopted_managed_ci", lambda *_a, **_k: True)
    monkeypatch.setattr(orchestrator, "managed_label_present", lambda *_a, **_k: True)
    monkeypatch.setattr(
        orchestrator, "dispatch_final_qualification",
        lambda *_a, **kwargs: qualified_heads.append(kwargs["expected_head_sha"]),
    )
    monkeypatch.setattr(
        orchestrator, "wait_for_final_qualification", lambda *_a, **_k: next(outcomes)
    )
    monkeypatch.setattr(orchestrator, "merge_pr", lambda *_a, **kwargs: merges.append(kwargs))
    real_publish = orchestrator.publish_issue_created_continuity_authorization

    def publish(*args, **kwargs):
        continued = real_publish(*args, **kwargs)
        handoffs.append((kwargs, continued))
        return continued

    monkeypatch.setattr(orchestrator, "publish_issue_created_continuity_authorization", publish)

    assert orchestrator.run_pr_loop(
        runner, pr_number=7, config=config, managed_ci_handoff=root,
        managed_ci_issue_number=643,
    ) == 0

    # The real publisher wrote exactly one continuity grant, bound to the lone
    # orchestrator-serialized coder record for the A->B transition.
    records = _authorization_records(runner)
    assert [record.kind for _comment_id, record in records] == ["creation", "continuity"]
    grant_id, grant = records[1]
    assert (grant.predecessor_head, grant.head_sha) == ("abc123", "abc123-coder-1")
    assert grant.predecessor_comment_id == root.authorization_comment_id
    assert len(grant.round_comment_ids) == 1
    coder_comment = next(
        comment for comment in runner.intent_comments
        if comment["id"] == grant.round_comment_ids[0]
    )
    decoded = managed_ci._continuity_round_records([coder_comment])[0]
    assert decoded["role"] == "coder"
    assert decoded["ci_repair_transitions"] == frozenset({("abc123", "abc123-coder-1")})
    assert handoffs[0][1].authorization_comment_id == grant_id
    # Resume reauthenticates the same chain.
    audit = _find_resume_audit(
        runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
        base_ref="main", issue_number=643, live_head="abc123-coder-1",
    )
    assert audit is not None and audit[0] == grant_id
    # Re-review, qualification and merge all target the repaired head only.
    assert len([cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]) == 2
    assert qualified_heads == ["abc123", "abc123-coder-1"]
    assert merges == [{"expected_head_sha": "abc123-coder-1"}]
    err = capsys.readouterr().err
    assert "Round 1: Claude repairing failed CI" in err
    assert "addressing reviewer feedback" not in err
    # #1273: the managed exact-head repair dispatch carries the standing guidance
    # and a history block (a round-1 failure is current-round, so none is listed yet).
    repair_prompts = [
        "\n".join(cmd) for cmd, _cwd in runner.commands if cmd and cmd[0] == "claude"
    ]
    assert len(repair_prompts) == 1
    assert "Proactive generalization" in repair_prompts[0]
    assert "no earlier-round findings or fixes" in repair_prompts[0]


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


def test_fresh_authorization_supersedes_grant_on_unrelated_replacement_head(tmp_path):
    # #1065: a grant GitHub cannot chain to the live head is unbound history;
    # the explicit fresh grant retires it rather than chaining or refusing.
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    first = authorize_fresh_issue_created_resume(
        runner, config=config, pr_number=7, issue_number=643,
        metadata=replace(metadata(), head_branch="agent-loop/managed-643"),
    )
    runner.rest_pr["head"]["sha"] = "replacement"
    runner.compare_payload = {
        "status": "diverged",
        "base_commit": {"sha": "abc123"},
        "merge_base_commit": {"sha": "other"},
    }

    replacement = authorize_fresh_issue_created_resume(
        runner, config=config, pr_number=7, issue_number=643,
        metadata=replace(
            metadata(), head_branch="agent-loop/managed-643", head_sha="replacement"
        ),
    )

    assert replacement.head_sha == "replacement"
    record = parse_issue_created_authorization_comment(runner.intent_comments[-1]["body"])
    assert record is not None
    assert record.predecessor_head is None and record.predecessor_comment_id is None
    assert record.superseded_comment_ids == (first.authorization_comment_id,)


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


def test_recovery_renderer_skips_the_plan_primary_stall_rounds_value(tmp_path):
    """The stall threshold's integer value is not mistaken for the issue (#1103)."""
    parser = build_parser()
    config = make_config(
        tmp_path,
        invocation_argv=(
            "agent-loop", "issue", "--plan-review-policy", "primary-then-panel",
            "--plan-primary-stall-rounds", "3", "643",
        ),
    )
    rendered = render_managed_ci_resume_command(config, pr_number=7, managed_ci=True)
    args = parser.parse_args(shlex.split(rendered)[1:])
    assert args.command == "issue"
    assert args.issue_number == 643
    assert args.plan_primary_stall_rounds == 3


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
            mergeability=None,
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


@pytest.mark.parametrize(
    ("after_ready", "expect_merge"),
    [
        (["BLOCKED", "CLEAN"], True),
        (["CLEAN"], True),
        (["BLOCKED", "BLOCKED", "BLOCKED"], False),
        (["UNSTABLE", "UNKNOWN", "UNSTABLE"], False),
    ],
)
def test_ordinary_fallback_unreadable_protection_requires_clean_after_ready(
    tmp_path, monkeypatch, after_ready, expect_merge,
):
    # #1055: a draft reports DRAFT, so under a classic-protection 403 the
    # recovery qualifies the green draft board, marks it ready, and merges
    # only once GitHub reports CLEAN for the same exact head.
    config = make_config(
        tmp_path, auto_merge=True, ci_poll_interval_seconds=30, ci_startup_timeout_seconds=90,
    )
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
                required=(),
                protection="forbidden",
            ),
            mergeability=PullRequestMergeability("mergeable", "MERGEABLE", "DRAFT", "abc123", "main"),
            head_sha="abc123",
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "get_pr_review_context",
        lambda *args, **kwargs: SimpleNamespace(metadata=metadata()),
    )
    events: list[str] = []
    states = list(after_ready)

    def fake_mergeability(*args, **kwargs):
        assert any(command[:3] == ["gh", "pr", "ready"] for command, _ in runner.commands)
        events.append("probe")
        return PullRequestMergeability("mergeable", "MERGEABLE", states.pop(0), "abc123", "main")

    monkeypatch.setattr(orchestrator, "get_pr_mergeability", fake_mergeability)
    merged: list[tuple[int, str | None]] = []
    monkeypatch.setattr(
        orchestrator,
        "merge_pr",
        lambda _runner, _config, number, *, expected_head_sha: merged.append((number, expected_head_sha)),
    )

    if expect_merge:
        _finalize_ordinary_recovery_merge(runner, config=config, pr_number=7, capability=capability)
        assert merged == [(7, "abc123")]
    else:
        with pytest.raises(AgentLoopError, match="merge state is not CLEAN after readiness"):
            _finalize_ordinary_recovery_merge(runner, config=config, pr_number=7, capability=capability)
        assert merged == []
        # Bounded by the startup window (90s / 30s), not the CI timeout.
        assert len(events) == 3
    assert not states


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


@pytest.mark.parametrize(
    ("merge_state", "head", "expected"),
    [
        # DRAFT defers the CLEAN check to the finalizer after readiness.
        ("DRAFT", "abc123", "passed"),
        # The single poll is also the last one (1s budget, 120s startup
        # window): a green board that cannot qualify reports the unreadable
        # protection instead of a generic timeout (late-green deadline).
        ("DRAFT", "other", "protection_unreadable"),
        ("BLOCKED", "abc123", "protection_unreadable"),
        ("CLEAN", "other", "protection_unreadable"),
        ("CLEAN", "abc123", "passed"),
    ],
)
def test_ordinary_recovery_forbidden_branch_protection_requires_clean_merge_state(
    monkeypatch, tmp_path, merge_state, head, expected,
):
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
    mergeability = PullRequestMergeability("mergeable", "MERGEABLE", merge_state, head, "main")
    monkeypatch.setattr(managed_ci, "get_pr_head_sha", lambda *args, **kwargs: "abc123")
    monkeypatch.setattr(
        managed_ci,
        "get_pr_mergeability",
        lambda *args, **kwargs: mergeability,
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

    assert outcome.status == expected
    assert outcome.mergeability == mergeability


@pytest.mark.parametrize(
    "mergeability",
    [
        PullRequestMergeability("unknown", "UNKNOWN", "UNKNOWN", "abc123", "main"),
        PullRequestMergeability("unknown", None, None, None, None),
        PullRequestMergeability("mergeable", "MERGEABLE", "BLOCKED", "abc123", "main"),
    ],
)
def test_ordinary_recovery_unqualifiable_merge_state_stops_within_startup_window(
    monkeypatch, tmp_path, mergeability,
):
    # #1055 review: a green board under unreadable protection whose merge
    # state is neither same-head DRAFT nor CLEAN must not poll to the timeout.
    config = make_config(
        tmp_path, auto_merge=True, ci_timeout_seconds=1200, ci_poll_interval_seconds=30,
        ci_startup_timeout_seconds=60,
    )
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
    monkeypatch.setattr(managed_ci, "get_pr_mergeability", lambda *args, **kwargs: mergeability)
    monkeypatch.setattr(
        managed_ci,
        "_workflow_runs_payload",
        lambda *args, **kwargs: [{"id": 3, "status": "completed", "conclusion": "success"}],
    )
    probes: list[int] = []
    monkeypatch.setattr(
        managed_ci, "get_pr_checks", lambda *args, **kwargs: probes.append(1) or passing,
    )
    runner = FakeRunner()

    outcome = wait_for_ordinary_recovery(
        runner=runner, config=config, capability=capability, metadata=metadata(),
    )

    assert outcome.status == "protection_unreadable"
    assert outcome.mergeability == mergeability
    assert len(probes) == 2
    assert len([cmd for cmd, _ in runner.commands if cmd[:1] == ["sleep"]]) == 1


def test_ordinary_recovery_late_green_board_at_deadline_reports_unreadable_protection(
    monkeypatch, tmp_path,
):
    # The board turns green only on the final poll, with fewer polls left
    # than the startup window; the deadline must still name the cause.
    config = make_config(
        tmp_path, auto_merge=True, ci_timeout_seconds=90, ci_poll_interval_seconds=30,
        ci_startup_timeout_seconds=300,
    )
    capability = OrdinaryRecoveryCapability(
        pr_number=7, repository="OWNER/REPO", base_ref="main", expected_head_sha="abc123",
        released_label_event_id=101, released_at=100, prior_run_ids=frozenset(),
    )
    success = PullRequestCheck("test", "check_run", "success")
    boards = [
        checks(pending=(PullRequestCheck("test", "check_run", "in_progress"),), protection="forbidden"),
        checks(pending=(PullRequestCheck("test", "check_run", "in_progress"),), protection="forbidden"),
        checks(passing=(success,), required=("test",), protection="forbidden"),
    ]
    runs = [
        [{"id": 3, "status": "in_progress"}],
        [{"id": 3, "status": "in_progress"}],
        [{"id": 3, "status": "completed", "conclusion": "success"}],
    ]
    blocked = PullRequestMergeability("unknown", "UNKNOWN", "UNKNOWN", "abc123", "main")
    monkeypatch.setattr(managed_ci, "get_pr_head_sha", lambda *args, **kwargs: "abc123")
    monkeypatch.setattr(managed_ci, "get_pr_mergeability", lambda *args, **kwargs: blocked)
    monkeypatch.setattr(managed_ci, "_workflow_runs_payload", lambda *args, **kwargs: runs.pop(0))
    monkeypatch.setattr(managed_ci, "get_pr_checks", lambda *args, **kwargs: boards.pop(0))

    outcome = wait_for_ordinary_recovery(
        runner=FakeRunner(), config=config, capability=capability, metadata=metadata(),
    )

    assert outcome.status == "protection_unreadable"
    assert outcome.mergeability == blocked
    assert not boards


def test_ordinary_fallback_protection_unreadable_stops_with_guidance(tmp_path, monkeypatch, capsys):
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
            status="protection_unreadable",
            checks=checks(
                passing=(PullRequestCheck("test", "check_run", "success"),),
                protection="forbidden",
            ),
            mergeability=PullRequestMergeability("unknown", "UNKNOWN", "UNKNOWN", "abc123", "main"),
            head_sha="abc123",
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "get_pr_review_context",
        lambda *args, **kwargs: SimpleNamespace(metadata=metadata()),
    )
    monkeypatch.setattr(orchestrator, "merge_pr", lambda *args, **kwargs: pytest.fail("must not merge"))

    with pytest.raises(AgentLoopError, match="branch protection is unreadable"):
        _finalize_ordinary_recovery_merge(runner, config=config, pr_number=7, capability=capability)

    out = capsys.readouterr().out
    assert "HTTP 403" in out
    assert "merge state UNKNOWN" in out
    assert not any(command[:3] == ["gh", "pr", "ready"] for command, _ in runner.commands)


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
        "expected_head_sha": "abc123",
    }
    fields.update(overrides)
    return ManagedCiContract(**fields)


MANAGED_LABEL_DELETE = [
    "gh", "api", "--method", "DELETE", f"repos/OWNER/REPO/issues/7/labels/{MANAGED_LABEL}",
]


def _managed_label_deletes(runner):
    return [command for command, _cwd in runner.commands if command[:5] == MANAGED_LABEL_DELETE]


def test_publish_manual_v2_qualification_retains_label_readies_and_audits_sha(tmp_path):
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
    # Retaining the label means no `unlabeled` event re-runs ordinary CI.
    assert _managed_label_deletes(runner) == []
    assert runner.rest_pr["labels"] == [{"name": MANAGED_LABEL}]
    assert runner.rest_pr["draft"] is False
    event_reads = [
        index for index, command in enumerate(commands)
        if any(part.startswith("repos/OWNER/REPO/issues/7/events?") for part in command)
    ]
    ready_index = next(index for index, command in enumerate(commands) if command[:4] == ["gh", "pr", "ready", "7"])
    audit_index = next(
        index for index, command in enumerate(commands) if QUALIFICATION_MARKER in " ".join(command)
    )
    # Provenance is verified before readiness and again before the record.
    assert len(event_reads) == 2
    assert event_reads[0] < ready_index < event_reads[1] < audit_index
    audit_body = next(iter(runner.audit_comments.values()))["body"]
    assert "The managed label is retained so ordinary CI does not re-run" in audit_body
    assert "was released" not in audit_body
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
    assert not any(QUALIFICATION_MARKER in " ".join(command) for command, _cwd in runner.commands)
    # A failed publication returns the owned label to ordinary CI.
    assert len(_managed_label_deletes(runner)) == 1
    assert runner.rest_pr["labels"] == []


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
    # The final head check runs after the record on a ready PR; the owned
    # label is still released so the drifted head gets ordinary CI.
    assert len(_managed_label_deletes(runner)) == 1
    assert runner.rest_pr["draft"] is False


class PublicationRunner(ManualQualificationRunner):
    """Script per-read PR/event responses and label DELETE outcomes.

    ``pr_overrides`` and ``event_overrides`` map a zero-based read index to
    either raw stdout (``str``), a ``(stdout, returncode)`` tuple, or a
    callable taking the current payload and returning a replacement payload.
    ``foreign_event_before_read`` replaces the active label event with one
    applied by another actor before the given events read.
    """

    def __init__(
        self,
        *,
        pr_overrides=None,
        event_overrides=None,
        delete_outcome=None,
        ready_returncode=0,
        foreign_event_before_read=None,
        **kwargs,
    ):
        kwargs.setdefault("issue_events", [label_event()])
        super().__init__(**kwargs)
        self.pr_overrides = dict(pr_overrides or {})
        self.event_overrides = dict(event_overrides or {})
        self.delete_outcome = delete_outcome
        self.ready_returncode = ready_returncode
        self.foreign_event_before_read = foreign_event_before_read
        self.pr_reads = 0
        self.event_reads = 0

    @staticmethod
    def _scripted(override, payload):
        if callable(override):
            return json.dumps(override(copy.deepcopy(payload))), 0
        if isinstance(override, tuple):
            return override
        return override, 0

    def _run_locked(self, args, *, cwd, check, input_text=None):
        cmd = [str(arg) for arg in args]
        endpoint = next((part for part in cmd if part.startswith("repos/")), "")
        if cmd == ["gh", "api", "repos/OWNER/REPO/pulls/7"]:
            index = self.pr_reads
            self.pr_reads += 1
            if index in self.pr_overrides:
                cmd, cwd_path = self._record_command(args, cwd)
                stdout, returncode = self._scripted(self.pr_overrides[index], self.rest_pr)
                return CommandResult(cmd, cwd_path, stdout, "", returncode)
        if endpoint.startswith("repos/OWNER/REPO/issues/7/events?"):
            index = self.event_reads
            self.event_reads += 1
            if self.foreign_event_before_read == index:
                self.issue_events.append(label_event(201, event="unlabeled"))
                self.issue_events.append(label_event(202, login="someone-else", actor_id=9))
            if index in self.event_overrides:
                cmd, cwd_path = self._record_command(args, cwd)
                stdout, returncode = self._scripted(self.event_overrides[index], self.issue_events)
                return CommandResult(cmd, cwd_path, stdout, "", returncode)
        if cmd[:5] == MANAGED_LABEL_DELETE and self.delete_outcome is not None:
            if self.delete_outcome == "raise":
                self._record_command(args, cwd)
                raise RuntimeError("transport exploded")
            cmd, cwd_path = self._record_command(args, cwd)
            stderr, returncode = self.delete_outcome
            if "404" in stderr:
                self.rest_pr["labels"] = []
            return CommandResult(cmd, cwd_path, "", stderr, returncode)
        if cmd[:3] == ["gh", "pr", "ready"] and self.ready_returncode:
            cmd, cwd_path = self._record_command(args, cwd)
            return CommandResult(cmd, cwd_path, "", "ready failed", self.ready_returncode)
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


def _publish(runner, tmp_path, **contract_overrides):
    config = make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop")
    fields = {
        "issue_created_pr": True,
        "active_label_event_id": 101,
        "invocation_applied_label": True,
        "protection_mode": "strict",
    }
    fields.update(contract_overrides)
    contract = v2_contract(**fields)
    return publish_manual_v2_qualification(
        runner, config=config, pr_number=7, expected_head_sha="abc123",
        contract=contract, reviewers=("Codex",),
    ), contract


def _audit_posted(runner):
    return any(QUALIFICATION_MARKER in " ".join(command) for command, _cwd in runner.commands)


def _without_draft(payload):
    payload.pop("draft", None)
    return payload


@pytest.mark.parametrize(
    "draft_value",
    [True, "missing", None, "false", 0],
    ids=["true", "missing", "none", "string", "zero"],
)
@pytest.mark.parametrize("stage", ["initial-guard", "pre-record-reread"])
def test_publish_manual_rejects_adopted_pr_whose_draft_is_not_exactly_false(
    tmp_path, draft_value, stage,
):
    def with_draft(payload):
        if draft_value == "missing":
            payload.pop("draft", None)
        else:
            payload["draft"] = draft_value
        return payload

    # PR reads: 0 stale-label cleanup, 1 initial guard, 2 pre-record re-read.
    runner = PublicationRunner(
        rest_pr={"draft": False},
        pr_overrides={1 if stage == "initial-guard" else 2: with_draft},
    )

    with pytest.raises(AgentLoopError, match="changed before manual qualification publication"):
        _publish(runner, tmp_path, issue_created_pr=False, adopted_existing_pr=True)

    assert not _audit_posted(runner)
    assert not any(command[:3] == ["gh", "pr", "ready"] for command, _cwd in runner.commands)
    # The owned label is released so the unproven PR returns to ordinary CI.
    assert len(_managed_label_deletes(runner)) == 1


def test_publish_manual_adopted_ready_pr_retains_label_without_readying(tmp_path):
    runner = PublicationRunner(rest_pr={"draft": False})

    qualified, _contract = _publish(runner, tmp_path, issue_created_pr=False, adopted_existing_pr=True)

    assert qualified == "abc123"
    assert _managed_label_deletes(runner) == []
    assert runner.event_reads == 2
    assert not any(command[:3] == ["gh", "pr", "ready"] for command, _cwd in runner.commands)


@pytest.mark.parametrize("state", ["draft", "ready"])
def test_publish_manual_unreadable_provenance_releases_label_fail_open(tmp_path, state):
    # Event read 0 is the pre-readiness check (draft PR); read 1 is the
    # pre-record check (ready PR).  Cleanup's own read (next index) also fails.
    failing = 0 if state == "draft" else 1
    runner = PublicationRunner(
        event_overrides={failing: ("", 1), failing + 1: ("", 1)},
    )

    with pytest.raises(AgentLoopError, match="the head is not qualified") as raised:
        _publish(runner, tmp_path)

    assert not _audit_posted(runner)
    assert len(_managed_label_deletes(runner)) == 1
    assert runner.rest_pr["labels"] == []
    assert runner.rest_pr["draft"] is (state == "draft")
    assert "remove it manually" not in str(raised.value)


@pytest.mark.parametrize("before_read", [0, 1], ids=["before-readiness", "before-record"])
def test_publish_manual_replaced_label_event_fails_closed_and_is_left(tmp_path, before_read):
    runner = PublicationRunner(foreign_event_before_read=before_read)

    with pytest.raises(AgentLoopError, match="provenance changed") as raised:
        _publish(runner, tmp_path)

    assert not _audit_posted(runner)
    assert _managed_label_deletes(runner) == []
    assert "different `agent-loop-managed` label event is active" in str(raised.value)
    assert type(raised.value) is AgentLoopError


def test_publish_manual_readiness_failure_releases_owned_label(tmp_path):
    runner = PublicationRunner(ready_returncode=1)

    with pytest.raises(AgentLoopError, match="Unable to mark qualified PR #7 ready"):
        _publish(runner, tmp_path)

    assert runner.rest_pr["draft"] is True
    assert len(_managed_label_deletes(runner)) == 1
    assert not _audit_posted(runner)


def test_publish_manual_post_ready_verification_failure_releases_owned_label(tmp_path):
    # PR read 2 follows `gh pr ready`; a drifted head there fails verification.
    def drifted(payload):
        payload["head"] = dict(payload["head"], sha="other")
        return payload

    runner = PublicationRunner(pr_overrides={2: drifted})

    with pytest.raises(AgentLoopError, match="changed while being made ready"):
        _publish(runner, tmp_path)

    assert len(_managed_label_deletes(runner)) == 1
    assert not _audit_posted(runner)


def test_publish_manual_record_failure_releases_owned_label(tmp_path, monkeypatch):
    def failed_post(*args, **kwargs):
        raise AgentLoopError("record write failed")

    monkeypatch.setattr(managed_ci, "post_verified_trusted_pr_protocol_comment", failed_post)
    runner = PublicationRunner()

    with pytest.raises(AgentLoopError, match="record write failed"):
        _publish(runner, tmp_path)

    assert runner.rest_pr["draft"] is False
    assert len(_managed_label_deletes(runner)) == 1


def test_publish_manual_cleanup_confirmed_absent_label_makes_no_write(tmp_path, monkeypatch):
    monkeypatch.setattr(managed_ci, "get_pr_head_sha", lambda *args, **kwargs: "new-head")
    runner = PublicationRunner(rest_pr={"labels": [{"name": "bug"}]})

    with pytest.raises(AgentLoopError, match="head changed before"):
        _publish(runner, tmp_path)

    assert _managed_label_deletes(runner) == []
    assert runner.event_reads == 0


@pytest.mark.parametrize(
    "pr_read",
    [
        ("", 1),
        ("", 0),
        ("not json", 0),
        ("[]", 0),
        ("{}", 0),
        (json.dumps({"labels": "agent-loop-managed"}), 0),
        (json.dumps({"labels": [{"id": 3}]}), 0),
        (json.dumps({"labels": ["agent-loop-managed"]}), 0),
    ],
    ids=[
        "nonzero-exit", "empty-body", "invalid-json", "non-object", "missing-labels",
        "non-list-labels", "entry-without-name", "string-entry",
    ],
)
@pytest.mark.parametrize(
    "delete_outcome,expect_fragment",
    [(None, False), (("gh: Not Found (HTTP 404)", 1), False)],
    ids=["deleted", "already-absent-404"],
)
def test_publish_manual_cleanup_unreadable_pr_attempts_delete(
    tmp_path, monkeypatch, pr_read, delete_outcome, expect_fragment,
):
    monkeypatch.setattr(managed_ci, "get_pr_head_sha", lambda *args, **kwargs: "new-head")
    runner = PublicationRunner(pr_overrides={0: pr_read}, delete_outcome=delete_outcome)

    with pytest.raises(AgentLoopError, match="head changed before") as raised:
        _publish(runner, tmp_path)

    assert len(_managed_label_deletes(runner)) == 1
    assert ("remove it manually" in str(raised.value)) is expect_fragment


def test_publish_manual_cleanup_never_masks_the_publication_error(tmp_path, monkeypatch):
    monkeypatch.setattr(managed_ci, "get_pr_head_sha", lambda *args, **kwargs: "new-head")
    # Invalid events JSON is unreadable provenance (fail-open DELETE), and the
    # DELETE itself raises: the helper swallows both.
    runner = PublicationRunner(event_overrides={0: "not json"}, delete_outcome="raise")

    with pytest.raises(AgentLoopError) as raised:
        _publish(runner, tmp_path)

    assert type(raised.value) is AgentLoopError
    message = str(raised.value)
    assert message.startswith("PR #7 head changed before manual qualification publication")
    assert "`agent-loop-managed` could not be removed and still suppresses ordinary CI" in message


def test_publish_manual_cleanup_attaches_note_to_non_agent_loop_error(tmp_path, monkeypatch):
    def exploding(*args, **kwargs):
        raise KeyError("boom")

    monkeypatch.setattr(managed_ci, "get_pr_head_sha", exploding)
    runner = PublicationRunner(delete_outcome=("gh: Server Error (HTTP 500)", 1))

    with pytest.raises(KeyError) as raised:
        _publish(runner, tmp_path)

    assert any("remove it manually" in note for note in raised.value.__notes__)


def test_publish_manual_non_404_delete_failure_reports_manual_removal(tmp_path, monkeypatch):
    monkeypatch.setattr(managed_ci, "get_pr_head_sha", lambda *args, **kwargs: "new-head")
    runner = PublicationRunner(delete_outcome=("gh: Server Error (HTTP 500)", 1))

    with pytest.raises(AgentLoopError, match="remove it manually"):
        _publish(runner, tmp_path)

    assert runner.rest_pr["labels"] == [{"name": MANAGED_LABEL}]


class EntryNormalizationRunner(FakeRunner):
    """Serve the live PR for entry normalization from a mutable payload."""

    def __init__(
        self, live_pr, *, delete_returncode=0, readback_labels=None, delete_stderr=None,
        raise_on=None, clear_on_failed_delete=False, **kwargs,
    ):
        super().__init__(**kwargs)
        self.clear_on_failed_delete = clear_on_failed_delete
        self.live_pr = live_pr
        self.delete_returncode = delete_returncode
        self.readback_labels = readback_labels
        self.delete_stderr = delete_stderr or "gh: Server Error (HTTP 500)"
        # ``raise_on`` is "delete" or "readback": that runner call raises, as a
        # transport or subprocess failure would.
        self.raise_on = raise_on
        self.pr_reads = 0

    def _run_locked(self, args, *, cwd, check, input_text=None):
        cmd = [str(arg) for arg in args]
        if cmd == ["gh", "api", "repos/OWNER/REPO/pulls/7"]:
            cmd, cwd_path = self._record_command(args, cwd)
            self.pr_reads += 1
            if self.raise_on == "readback" and self.pr_reads == 2:
                raise OSError("transport exploded during read-back")
            if isinstance(self.live_pr, str):
                return CommandResult(cmd, cwd_path, self.live_pr, "", 0)
            return CommandResult(cmd, cwd_path, json.dumps(self.live_pr), "", 0)
        if cmd[:5] == MANAGED_LABEL_DELETE:
            cmd, cwd_path = self._record_command(args, cwd)
            if self.raise_on == "delete":
                raise OSError("transport exploded during DELETE")
            if self.delete_returncode:
                if self.clear_on_failed_delete:
                    self.live_pr["labels"] = []
                return CommandResult(cmd, cwd_path, "", self.delete_stderr, 1)
            self.live_pr["labels"] = (
                self.readback_labels if self.readback_labels is not None else [
                    item for item in self.live_pr["labels"] if item.get("name") != MANAGED_LABEL
                ]
            )
            return CommandResult(cmd, cwd_path, "", "", 0)
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


def _live_pr(*, state="open", draft=False, labels=(MANAGED_LABEL,)):
    return {"state": state, "draft": draft, "labels": [{"name": name} for name in labels]}


def test_release_retained_managed_label_releases_open_ready_labeled_pr(tmp_path):
    runner = EntryNormalizationRunner(_live_pr(labels=(MANAGED_LABEL, "bug")))
    config = make_config(tmp_path)

    assert release_retained_managed_label(runner, config=config, pr_number=7, cwd=tmp_path) is True

    assert len(_managed_label_deletes(runner)) == 1
    assert runner.live_pr["labels"] == [{"name": "bug"}]
    # Every command (read, DELETE, read-back) runs in the supplied cwd, except the
    # ownership-baseline timeline read, which is a repository-scoped API list.
    assert {
        cwd for command, cwd in runner.commands if "--paginate" not in command
    } == {tmp_path}


def test_entry_release_logs_operator_label_on_adopted_ready_pr(tmp_path, capsys):
    live_pr = _live_pr()
    live_pr["head"] = {"ref": "feature/operator-change"}
    runner = EntryNormalizationRunner(live_pr)

    assert release_retained_managed_label(
        runner, config=make_config(tmp_path, quiet=False), pr_number=7, cwd=tmp_path,
    ) is True

    diagnostic = capsys.readouterr().err
    assert "PR #7: removed `agent-loop-managed` from the ready PR at entry" in diagnostic
    assert "label origin was not checked" in diagnostic
    assert "ordinary CI resumes" in diagnostic
    # The only event read is the removal's ownership baseline (#510); with no
    # readable baseline the removal stays a single attempt.
    assert len(_managed_label_deletes(runner)) == 1


@pytest.mark.parametrize(
    "live_pr",
    [
        _live_pr(draft=True),
        _live_pr(draft=True, labels=()),
        _live_pr(labels=()),
        _live_pr(state="closed"),
        _live_pr(draft=None),
        {"state": "open", "draft": False},
        {"state": "open", "draft": False, "labels": "agent-loop-managed"},
        "not json",
        "",
    ],
    ids=[
        "draft-labeled", "draft-unlabeled", "ready-unlabeled", "closed", "draft-none",
        "missing-labels", "malformed-labels", "invalid-json", "empty",
    ],
)
def test_release_retained_managed_label_leaves_other_states_untouched(tmp_path, live_pr, capsys):
    runner = EntryNormalizationRunner(live_pr)

    assert release_retained_managed_label(
        runner, config=make_config(tmp_path, quiet=False), pr_number=7, cwd=tmp_path,
    ) is False

    assert _managed_label_deletes(runner) == []
    assert len(runner.commands) == 1
    assert "removed `agent-loop-managed`" not in capsys.readouterr().err


@pytest.mark.parametrize("failure", ["delete", "readback"])
def test_release_retained_managed_label_failure_names_manual_remedy(tmp_path, failure):
    runner = EntryNormalizationRunner(
        _live_pr(),
        delete_returncode=1 if failure == "delete" else 0,
        readback_labels=[{"name": MANAGED_LABEL}] if failure == "readback" else None,
    )

    with pytest.raises(AgentLoopError, match="Remove the label manually, then rerun"):
        release_retained_managed_label(
            runner, config=make_config(tmp_path), pr_number=7, cwd=tmp_path,
        )


@pytest.mark.parametrize("raise_on", ["delete", "readback"])
def test_release_retained_managed_label_runner_exception_names_manual_remedy(tmp_path, raise_on):
    runner = EntryNormalizationRunner(_live_pr(), raise_on=raise_on)

    with pytest.raises(AgentLoopError, match="Remove the label manually, then rerun") as raised:
        release_retained_managed_label(
            runner, config=make_config(tmp_path), pr_number=7, cwd=tmp_path,
        )

    assert type(raised.value) is AgentLoopError
    assert isinstance(raised.value.__cause__, OSError)
    assert len(_managed_label_deletes(runner)) == 1
    if raise_on == "readback":
        # The DELETE succeeded but absence was never confirmed, so the run
        # still stops rather than assuming the label is gone.
        assert runner.live_pr["labels"] == []


@pytest.mark.parametrize(
    "stderr",
    [
        "gh: Forbidden (HTTP 403)\nupstream said HTTP 404 earlier\n",
        "gh: Not Found (HTTP 404)\ngh: Forbidden (HTTP 403)\n",
        "error: HTTP 404 mentioned mid-line (HTTP 403)\n",
        "HTTP 404: Not Found",
    ],
    ids=["incidental-404-text", "conflicting-statuses", "mid-line-404", "unstructured-404"],
)
def test_release_retained_managed_label_ambiguous_404_is_not_absence(tmp_path, stderr):
    runner = EntryNormalizationRunner(_live_pr(), delete_returncode=1, delete_stderr=stderr)

    with pytest.raises(AgentLoopError, match="Remove the label manually, then rerun"):
        release_retained_managed_label(
            runner, config=make_config(tmp_path), pr_number=7, cwd=tmp_path,
        )

    assert runner.live_pr["labels"] == [{"name": MANAGED_LABEL}]


def test_release_retained_managed_label_strict_404_counts_as_absent(tmp_path):
    # Someone else removed the label between the read and the DELETE: gh's
    # strict 404 diagnostic is absence, and the read-back confirms it.
    runner = EntryNormalizationRunner(
        _live_pr(), delete_returncode=1, delete_stderr="gh: Not Found (HTTP 404)\n",
        clear_on_failed_delete=True,
    )

    assert release_retained_managed_label(
        runner, config=make_config(tmp_path), pr_number=7, cwd=tmp_path,
    ) is True
    assert runner.live_pr["labels"] == []


@pytest.mark.parametrize(
    "stderr",
    [
        "gh: Forbidden (HTTP 403)\nupstream said HTTP 404 earlier\n",
        "gh: Not Found (HTTP 404)\ngh: Forbidden (HTTP 403)\n",
        "HTTP 404: Not Found",
    ],
    ids=["incidental-404-text", "conflicting-statuses", "unstructured-404"],
)
def test_publish_manual_cleanup_ambiguous_404_reports_manual_removal(tmp_path, monkeypatch, stderr):
    monkeypatch.setattr(managed_ci, "get_pr_head_sha", lambda *args, **kwargs: "new-head")
    runner = PublicationRunner(delete_outcome=(stderr, 1))
    # The scripted DELETE clears labels for any "404" text; an ambiguous
    # diagnostic must still be reported as a failed removal.
    with pytest.raises(AgentLoopError, match="remove it manually"):
        _publish(runner, tmp_path)

    assert len(_managed_label_deletes(runner)) == 1


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

    assert publish_round_readiness(runner, config=config, head_sha="abc123") is True

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
        expected_head_sha="abc123",
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
        expected_head_sha="abc123",
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
    # Follow orchestrator code into the modules extracted from it (#1181).
    paths = [orchestrator_split_guard.module_path("managed_ci"), *orchestrator_split_guard.split_source_paths()]
    for path in paths:
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


def _spilled_ci_repair_continuity():
    """#1024: a CI repair coder record whose prior_items spilled into sidecars."""
    import base64 as _base64
    import os as _os

    import coding_review_agent_loop.round_transport as transport

    item = replace(
        _advanced_ci_obligation(),
        text="final-ci/exact-head failed. "
        + _base64.urlsafe_b64encode(_os.urandom(50_000)).decode("ascii"),
    )
    prepared = transport.prepare_round_comment(
        _attach_round_metadata(
            "coder round",
            PostedRoundMetadata(
                flow="pr", role="coder", agent="agent-loop",
                round_number=13, subject="ci-fix-head", prior_items=(item,),
            ),
        )
    )
    anchor = transport.ROUND_RESUME_MARKER_RE.search(str(prepared[-1]))
    reference = transport.decode_mapping(anchor.group("payload"))["prior_items"]
    assert isinstance(reference, dict) and "$round_transport_spill" in reference
    comments = [
        {"id": 58 + index, "user": {"login": "agent-loop", "id": 1}, "body": str(body)}
        for index, body in enumerate(prepared)
    ]
    anchor_id = comments[-1]["id"]
    return comments, anchor_id, _ci_repair_continuity(anchor_id)


def test_m1024_spilled_ci_obligation_grants_continuity(tmp_path):
    comments, anchor_id, authorization = _spilled_ci_repair_continuity()

    records = managed_ci._continuity_round_records(comments)
    assert records[len(comments) - 1]["ci_repair_transitions"] == frozenset(
        {("abc123", "ci-fix-head")}
    )
    assert managed_ci._continuity_round_metadata_is_valid(
        comments, authorization=authorization
    ) is True

    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    runner.intent_comments.extend(comments)
    selected = managed_ci.find_actor_round_metadata_comment_ids(
        runner, config=make_config(tmp_path), pr_number=7, actor_login="agent-loop",
        actor_id=1, predecessor_head="abc123", new_head="ci-fix-head", round_number=12,
        after_comment_id=50,
    )
    assert selected == (anchor_id,)


def test_m1024_legacy_reader_declines_spilled_ci_repair_continuity(tmp_path, monkeypatch):
    """An older reader cannot see the spilled CI obligation and so denies, never grants."""
    import coding_review_agent_loop.round_transport as transport

    comments, _anchor_id, authorization = _spilled_ci_repair_continuity()
    monkeypatch.setattr(
        transport,
        "_SPILL_FIELDS",
        tuple(
            field for field in transport._SPILL_FIELDS
            if field not in transport._GROWTH_SPILL_FIELDS
        ),
    )

    records = managed_ci._continuity_round_records(comments)
    assert records[len(comments) - 1]["ci_repair_transitions"] == frozenset()
    assert managed_ci._continuity_round_metadata_is_valid(
        comments, authorization=authorization
    ) is False
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    runner.intent_comments.extend(comments)
    with pytest.raises(AgentLoopError, match="correlated blocking-review and coder"):
        managed_ci.find_actor_round_metadata_comment_ids(
            runner, config=make_config(tmp_path), pr_number=7, actor_login="agent-loop",
            actor_id=1, predecessor_head="abc123", new_head="ci-fix-head", round_number=12,
            after_comment_id=50,
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
        (
            [_unreadable_record()],
            {_CLASSIC: _failed(PLAN_LIMITED), _RULES: _failed(PLAN_LIMITED)},
            {},
            "live assessment is plan_limited",
        ),
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
        (
            [
                _unreadable_record(),
                _unreadable_record(
                    kind="fresh", protection="voluntary",
                    waiver="allow-unprotected-managed-ci", nonce="fresh-nonce",
                ),
            ],
            None, {"unreadable": False}, r"records disagree on protection \(unreadable, voluntary\)",
        ),
    ],
    ids=[
        "missing-flag", "live-disagrees", "creation-records-disagree",
        "fresh-record-disagrees", "unwaived-records-disagree",
    ],
)
def test_issue_created_recovery_refuses_without_mutation(
    tmp_path, records, scripted, config_overrides, match
):
    runner = _recovery_runner(records, scripted=scripted)

    with pytest.raises(AgentLoopError, match=match) as exc_info:
        _recover(runner, _cloud_config(tmp_path, managed_ci_pr_mode=True, **config_overrides))

    assert "The PR was left unchanged" in str(exc_info.value)
    if match != _BOTH_WAIVERS:
        # #1063: no flag resolves these, so no command is offered as a remedy.
        assert _resume_command(exc_info.value) is None
        assert "No resume flag reconciles" in str(exc_info.value)
    assert _mutations(runner) == []


_FAILED_ARGV = (
    "agent-loop", "pr", "7", "--managed-ci",
    "--managed-ci-trusted-actor", "agent-loop", "--allow-unprotected-managed-ci",
)


def _resume_command(error):
    match = re.search(r"Resume with `([^`]+)`", str(error))
    return None if match is None else match.group(1)


def test_issue_created_recovery_missing_waiver_remedy_is_not_the_failing_command(tmp_path):
    # #1063: the printed remedy must add the missing flag, not replay argv.
    runner = _recovery_runner([_unreadable_record()])
    config = _cloud_config(
        tmp_path, unreadable=False, managed_ci_pr_mode=True, invocation_argv=_FAILED_ARGV,
    )

    with pytest.raises(AgentLoopError, match=_BOTH_WAIVERS) as exc_info:
        _recover(runner, config)

    resume = _resume_command(exc_info.value)
    assert resume is not None
    assert resume != shlex.join(_FAILED_ARGV)
    assert "--allow-unreadable-protection" in shlex.split(resume)
    assert _mutations(runner) == []


def test_issue_created_recovery_reconciles_persisted_unreadable_with_live_voluntary(tmp_path):
    # #1063: a PR authorized where protection was unreadable resumes on a host
    # that reads voluntary; the persisted state is kept for every record.
    runner = _recovery_runner([_unreadable_record()], scripted={})

    handoff = _recover(runner, _cloud_config(tmp_path, managed_ci_pr_mode=True))

    assert handoff is not None
    assert handoff.protection_mode == "unreadable"
    assert _mutations(runner) == []


def test_issue_created_recovery_irreconcilable_mismatch_prints_no_identical_command(tmp_path):
    argv = _FAILED_ARGV + ("--allow-unreadable-protection",)
    runner = _recovery_runner(
        [_unreadable_record()],
        scripted={_CLASSIC: _failed(PLAN_LIMITED), _RULES: _failed(PLAN_LIMITED)},
    )
    config = _cloud_config(tmp_path, managed_ci_pr_mode=True, invocation_argv=argv)

    with pytest.raises(AgentLoopError, match="live assessment is plan_limited") as exc_info:
        _recover(runner, config)

    assert _resume_command(exc_info.value) is None
    assert "No resume flag reconciles" in str(exc_info.value)
    assert _mutations(runner) == []


def test_run_pr_loop_resumes_unreadable_pr_where_protection_reads_voluntary(
    tmp_path, monkeypatch,
):
    runner = _recovery_runner([_unreadable_record()], scripted={})
    config = _cloud_config(
        tmp_path,
        managed_ci_pr_mode=True,
        invocation_argv=_FAILED_ARGV + ("--allow-unreadable-protection",),
    )
    captured = {}
    _stop_after_real_activation(monkeypatch, captured)

    with pytest.raises(_ActivationReached) as exc_info:
        orchestrator.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)

    contract = exc_info.value.args[0]
    assert contract is not None
    assert contract.activation_path == "managed"
    assert contract.protection_mode == "unreadable"
    assert captured["managed_resume"].issue_created_handoff.protection_mode == "unreadable"
    assert runner.dispatch_count == 0


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

    with pytest.raises(AgentLoopError, match="authorization comment 42") as exc_info:
        _recover(runner, _cloud_config(tmp_path, managed_ci_pr_mode=True))

    assert _resume_command(exc_info.value) is None
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

    with pytest.raises(AgentLoopError, match="event history could not be inspected") as exc_info:
        _recover(runner, _cloud_config(tmp_path, managed_ci_pr_mode=True))

    assert "may be transient; retry" in str(exc_info.value)

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


# --- #1047: qualified ready/labeled PR through real recovery and activation ---


def _qualified_ready_labeled_runner(origin="issue-created", **kwargs):
    """A strict managed PR left ready and labeled by manual qualification.

    An issue-created PR carries the canonical closing reference; a
    source-managed PR carries the loop-created source record instead.
    """
    if origin == "source-managed":
        from coding_review_agent_loop.managed_pr import _source_marker

        body = str(_source_marker(source_branch="feature/source", source_sha="source-sha"))
    else:
        body = "Fixes #643"
    return ManualQualificationRunner(
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
            "draft": False,
            "labels": [{"name": MANAGED_LABEL}],
            "body": body,
        },
        issue_events=[label_event()],
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT]},
        **kwargs,
    )


def _lifecycle_writes(runner):
    """Every label, readiness, dispatch and merge write, in command order."""
    writes = []
    for command, _cwd in runner.commands:
        if command[:5] == [
            "gh", "api", "--method", "DELETE", f"repos/OWNER/REPO/issues/7/labels/{MANAGED_LABEL}",
        ]:
            writes.append("label-delete")
        elif command[:5] == ["gh", "api", "--method", "POST", "repos/OWNER/REPO/issues/7/labels"]:
            writes.append("label-post")
        elif command[:3] == ["gh", "pr", "ready"]:
            writes.append("ready-undo" if "--undo" in command else "ready")
        elif command[:3] == ["gh", "pr", "merge"]:
            writes.append("merge")
        elif any(str(part).endswith("/actions/workflows/ci.yml/dispatches") for part in command):
            writes.append("dispatch")
    return writes


@pytest.mark.parametrize("origin", ["issue-created", "source-managed"])
def test_m1047_real_reentry_releases_at_entry_then_undo_relabels_and_preserves_abort(
    tmp_path, monkeypatch, origin,
):
    runner = _qualified_ready_labeled_runner(
        origin,
        codex_outputs=[structured_pr_review(state="approved", summary="Approved.")],
    )
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci", "--managed-ci-trusted-actor", "agent-loop",
        ),
    )
    monkeypatch.setattr(
        orchestrator, "_freeze_prompt_architecture", lambda _runner, config, **_kwargs: config,
    )
    captured = {}
    real_activate = orchestrator.activate_managed_ci
    real_source_auth = orchestrator.authenticate_source_managed_resume
    source_auth_calls = []

    def source_auth(*args, **kwargs):
        # Record the live state the real source-managed classifier observed.
        source_auth_calls.append(
            (_lifecycle_writes(runner), runner.rest_pr["draft"], list(runner.rest_pr["labels"]))
        )
        return real_source_auth(*args, **kwargs)

    def activate(*args, **kwargs):
        captured["resume"] = kwargs.get("managed_resume")
        captured["writes_before"] = _lifecycle_writes(runner)
        captured["contract"] = real_activate(*args, **kwargs)
        return captured["contract"]

    def abort_dispatch(*args, **kwargs):
        # The run fails after review and before publication.
        raise AgentLoopError("dispatch aborted")

    monkeypatch.setattr(orchestrator, "activate_managed_ci", activate)
    monkeypatch.setattr(orchestrator, "authenticate_source_managed_resume", source_auth)
    monkeypatch.setattr(orchestrator, "dispatch_final_qualification", abort_dispatch)
    monkeypatch.setattr(
        orchestrator, "publish_manual_v2_qualification",
        lambda *a, **k: pytest.fail("publication must not run"),
    )

    with pytest.raises(AgentLoopError, match="dispatch aborted"):
        orchestrator.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)

    # Entry released the retained label before recovery classified the PR,
    # so the real classifier saw ready/unlabeled.
    assert captured["writes_before"] == ["label-delete"]
    assert captured["resume"].lifecycle == "ready-unlabeled-reentry"
    contract = captured["contract"]
    assert contract is not None and contract.activation_path == "managed"
    assert captured["resume"].origin == origin
    assert contract.origin == origin
    if origin == "source-managed":
        # Real source-managed authentication ran once, after the entry
        # release, and saw the ready/unlabeled state.
        assert source_auth_calls == [(["label-delete"], False, [])]
        assert captured["resume"].source_branch == "feature/source"
    else:
        assert source_auth_calls == []
    assert orchestrator._preserve_issue_created_managed_suppression(
        contract, active_exception=AgentLoopError("dispatch aborted"),
    )
    # Exactly one release, then the existing re-entry: undo readiness first,
    # then reapply the label; the aborted run keeps it for exact resume.
    assert _lifecycle_writes(runner) == ["label-delete", "ready-undo", "label-post"]
    assert runner.rest_pr["draft"] is True
    assert runner.rest_pr["labels"] == [{"name": MANAGED_LABEL}]

    # The next invocation resumes through draft/labeled; normalization leaves
    # the preserved draft untouched.
    def stop_at_activation(*args, **kwargs):
        raise _ActivationReached(kwargs.get("managed_resume"))

    monkeypatch.setattr(orchestrator, "activate_managed_ci", stop_at_activation)
    before = len(runner.commands)
    with pytest.raises(_ActivationReached) as reached:
        orchestrator.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)

    assert reached.value.args[0].lifecycle == "draft-labeled"
    assert reached.value.args[0].origin == origin
    later = [command for command, _cwd in runner.commands[before:]]
    assert not any(command[:4] == ["gh", "api", "--method", "DELETE"] for command in later)
    assert runner.rest_pr["labels"] == [{"name": MANAGED_LABEL}]


def test_m1047_real_implicit_invocation_releases_then_stops_with_retry_command(
    tmp_path, monkeypatch,
):
    runner = _qualified_ready_labeled_runner()
    config = make_config(
        tmp_path, auto_merge=True, managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
    )
    monkeypatch.setattr(
        orchestrator, "_freeze_prompt_architecture", lambda _runner, config, **_kwargs: config,
    )

    with pytest.raises(AgentLoopError, match="ready/unlabeled re-entry state") as raised:
        orchestrator.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)

    assert "It was left unchanged; rerun with explicit `--managed-ci`" in str(raised.value)
    # One entry release, then the implicit path stops with the PR ready and
    # unlabeled: no undo, relabel, dispatch or merge.
    assert _lifecycle_writes(runner) == ["label-delete"]
    assert runner.rest_pr["draft"] is False
    assert runner.rest_pr["labels"] == []


class _RefusingStatusRunner:
    """A runner whose commit-status write is refused, as by a host proxy (#1052)."""

    def __init__(self, stderr):
        self.stderr = stderr
        self.commands = []

    def run(self, args, *, cwd=None, check=True, **kwargs):
        self.commands.append(list(args))
        return subprocess.CompletedProcess(args, 1, "", self.stderr)


@pytest.mark.parametrize(
    "stderr",
    [
        "gh: Write access to this GitHub API path is not permitted through this proxy. (HTTP 403)\n",
        "",
    ],
)
def test_refused_round_readiness_is_logged_and_does_not_abort(tmp_path, capsys, stderr):
    config = replace(make_config(tmp_path, auto_merge=True), quiet=False)
    runner = _RefusingStatusRunner(stderr)

    assert publish_round_readiness(runner, config=config, head_sha="abc123") is False

    assert runner.commands[-1][4] == "repos/OWNER/REPO/statuses/abc123"
    err = capsys.readouterr().err
    assert f"could not publish the non-required `{READINESS_CONTEXT}` status for abc123" in err
    assert ("not permitted through this proxy" in err) if stderr else ("exit 1" in err)


# --- #1065: a fresh grant supersedes accumulated unbound authorization records ---

_PLAN_1065 = "a" * 64


def _stale_unreadable_records():
    """Two actor-owned grants from an unreadable-protection host, neither at the live head."""
    return [
        _unreadable_record(head_sha="old-1", approved_plan_hash=_PLAN_1065),
        _unreadable_record(
            kind="fresh", head_sha="old-2", nonce="second", approved_plan_hash=_PLAN_1065,
        ),
    ]


def _stale_record_comments(records, *, first_id=41):
    return [
        {
            "id": first_id + index,
            "user": {"login": "agent-loop", "id": 1},
            "body": str(format_issue_created_authorization_comment(record)),
        }
        for index, record in enumerate(records)
    ]


def _fresh_1065(runner, config):
    return authorize_fresh_issue_created_resume(
        runner, config=config, pr_number=7, issue_number=643,
        metadata=replace(metadata(), head_branch="agent-loop/managed-643", body="Fixes #643"),
        approved_plan_hash=_PLAN_1065,
    )


def test_m1065_fresh_authorization_supersedes_stale_conflicting_records(tmp_path):
    # Live protection now reads voluntary; both prior grants recorded unreadable
    # and neither reaches the live head.  Previously this refused outright.
    runner = CloudAuthorizationRunner(
        scripted={},
        issue_events=[label_event()],
        intent_comments=_stale_record_comments(_stale_unreadable_records()),
    )
    config = _cloud_config(tmp_path)

    handoff = _fresh_1065(runner, config)

    assert handoff.authorization_kind == "fresh"
    assert handoff.protection_mode == "voluntary"
    assert handoff.head_sha == "abc123"
    record = parse_issue_created_authorization_comment(runner.intent_comments[-1]["body"])
    assert record is not None
    assert record.superseded_comment_ids == (41, 42)
    assert record.predecessor_head is None and record.predecessor_comment_id is None
    # The superseded records are retired, not deleted.
    assert [item["id"] for item in runner.intent_comments[:2]] == [41, 42]

    # A rerun reuses the superseding grant instead of appending another record.
    posted = len(runner.intent_comments)
    again = _fresh_1065(runner, config)
    assert again.authorization_comment_id == handoff.authorization_comment_id
    assert len(runner.intent_comments) == posted

    # The resume audit and the plan binding both see one live-head terminal.
    audit = _find_resume_audit(
        runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
        base_ref="main", issue_number=643, live_head="abc123",
        expected_handoff=handoff, expected_protection="voluntary",
        require_actor_owned_label_event=True,
    )
    assert audit is not None
    assert audit[0] == handoff.authorization_comment_id
    managed_ci.verify_managed_pr_plan_binding(
        runner, config=config, pr_number=7, issue_number=643,
        live_head="abc123", approved_plan_hash=_PLAN_1065,
    )


def test_m1065_fresh_authorization_still_refuses_non_protection_conflicts(tmp_path):
    stale = _stale_unreadable_records()
    stale[1] = replace(stale[1], label_event_id=999)
    runner = CloudAuthorizationRunner(
        scripted={},
        issue_events=[label_event()],
        intent_comments=_stale_record_comments(stale),
    )

    with pytest.raises(AgentLoopError, match="conflicting actor-owned record"):
        _fresh_1065(runner, _cloud_config(tmp_path))

    assert len(runner.intent_comments) == 2


def test_m1065_recovery_after_supersession_uses_the_superseding_grant(tmp_path):
    superseding = ManagedCiIssueAuthorization(
        kind="fresh", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="abc123", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="fresh",
        label_event_id=101, approved_plan_hash=_PLAN_1065, superseded_comment_ids=(41, 42),
    )
    runner = _recovery_runner(_stale_unreadable_records() + [superseding], scripted={})

    handoff = _recover(runner, _cloud_config(tmp_path, managed_ci_pr_mode=True))

    assert handoff is not None
    assert handoff.protection_mode == "voluntary"
    assert _mutations(runner) == []


def test_m1065_supersession_is_scoped_to_earlier_same_scope_records():
    base = _unreadable_record(approved_plan_hash=_PLAN_1065)
    foreign = replace(base, issue_number=644)
    later = replace(base, head_sha="later")
    fresh = replace(
        base, kind="fresh", nonce="fresh", superseded_comment_ids=(41, 42, 44),
    )
    ids = managed_ci._superseded_authorization_ids(
        [(41, base), (42, foreign), (43, fresh), (44, later)]
    )
    assert ids == frozenset({41})
    # A creation checkpoint never supersedes.
    assert managed_ci._superseded_authorization_ids(
        [(41, base), (43, replace(fresh, kind="creation"))]
    ) == frozenset()


@pytest.mark.parametrize(
    ("kind", "superseded", "extra"),
    [
        ("creation", [41], {}),
        ("fresh", [42, 41], {}),
        ("fresh", [41, 41], {}),
        ("fresh", [], {}),
        ("fresh", [0], {}),
        ("fresh", [True], {}),
    ],
)
def test_m1065_parser_rejects_invalid_supersession_fields(kind, superseded, extra):
    payload = _unreadable_record(kind=kind).to_payload()
    payload["superseded_comment_ids"] = superseded
    payload.update(extra)
    with pytest.raises(AgentLoopError):
        parse_issue_created_authorization_comment(_encoded_authorization(payload))


def test_m1065_superseding_record_round_trips():
    record = _unreadable_record(kind="fresh", superseded_comment_ids=(41, 42))
    parsed = parse_issue_created_authorization_comment(
        str(format_issue_created_authorization_comment(record))
    )
    assert parsed == record


def test_m1065_fresh_authorization_supersedes_same_protection_unbound_records(tmp_path):
    # Both prior grants match the live protection, base, actor, label event and
    # plan, but neither head is an ancestor of the live head.
    stale = [
        replace(
            record, protection="voluntary", waiver="allow-unprotected-managed-ci",
        )
        for record in _stale_unreadable_records()
    ]
    runner = CloudAuthorizationRunner(
        scripted={},
        issue_events=[label_event()],
        intent_comments=_stale_record_comments(stale),
    )
    runner.compare_payload = {
        "status": "diverged",
        "base_commit": {"sha": "old"},
        "merge_base_commit": {"sha": "other"},
    }
    config = _cloud_config(tmp_path)

    handoff = _fresh_1065(runner, config)

    assert handoff.authorization_kind == "fresh"
    assert handoff.protection_mode == "voluntary"
    record = parse_issue_created_authorization_comment(runner.intent_comments[-1]["body"])
    assert record is not None
    assert record.superseded_comment_ids == (41, 42)
    assert record.predecessor_head is None

    posted = len(runner.intent_comments)
    again = _fresh_1065(runner, config)
    assert again.authorization_comment_id == handoff.authorization_comment_id
    assert len(runner.intent_comments) == posted

    audit = _find_resume_audit(
        runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
        base_ref="main", issue_number=643, live_head="abc123",
        expected_handoff=handoff, expected_protection="voluntary",
        require_actor_owned_label_event=True,
    )
    assert audit is not None
    assert audit[0] == handoff.authorization_comment_id
    managed_ci.verify_managed_pr_plan_binding(
        runner, config=config, pr_number=7, issue_number=643,
        live_head="abc123", approved_plan_hash=_PLAN_1065,
    )


# --- #1069: ordinary continuity supersedes this actor's unbound records ---


def _continuity_1069_runner(extra_records):
    root = _unreadable_record(
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="nonce-643",
    )
    comments = [
        {"id": 41, "user": {"login": "agent-loop", "id": 1},
         "body": str(format_issue_created_authorization_comment(root))},
    ]
    comments.extend(_stale_record_comments(extra_records, first_id=42))
    comments.extend([
        _round_comment(88, role="reviewer", subject="abc123", round_number=1, state="blocking"),
        _round_comment(89, role="coder", subject="next-head", round_number=2),
    ])
    runner = AuthorizationCommentRunner(issue_events=[label_event()], intent_comments=comments)
    runner.rest_pr["head"]["sha"] = "next-head"
    return runner


def _continue_1069(runner, config, handoff, *, predecessor, new_head, rounds):
    return publish_issue_created_continuity_authorization(
        runner, config=config, handoff=handoff, predecessor_head=predecessor,
        new_head=new_head, round_comment_ids=rounds,
    )


def _ancestors_1069(monkeypatch, ancestors):
    monkeypatch.setattr(
        managed_ci, "_github_proves_descendant",
        lambda _runner, *, config, predecessor_head, live_head: predecessor_head in ancestors,
    )


def test_m1069_resume_across_head_advance_supersedes_stranded_record(tmp_path, monkeypatch):
    # An interrupted earlier attempt left a compatible grant at a head that the
    # live history no longer contains.
    stray = _unreadable_record(
        kind="fresh", head_sha="stray-head", nonce="stray",
        protection="voluntary", waiver="allow-unprotected-managed-ci",
    )
    runner = _continuity_1069_runner([stray])
    _ancestors_1069(monkeypatch, {"abc123", "next-head"})
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    handoff = replace(_authorization_handoff(), authorization_comment_id=41)

    first = _continue_1069(
        runner, config, handoff, predecessor="abc123", new_head="next-head", rounds=(88, 89),
    )
    first_record = parse_issue_created_authorization_comment(runner.intent_comments[-1]["body"])
    assert first_record is not None
    assert first_record.kind == "continuity"
    assert first_record.predecessor_comment_id == 41
    assert first_record.superseded_comment_ids == (42,)
    # Nothing is deleted: the stranded record stays as history.
    assert 42 in [item["id"] for item in runner.intent_comments]

    # A second resume after another head advance extends the live chain and
    # has nothing left to retire; it does not append a competing record.
    runner.intent_comments.extend([
        _round_comment(200, role="reviewer", subject="next-head", round_number=2, state="blocking"),
        _round_comment(201, role="coder", subject="third-head", round_number=3),
    ])
    runner.rest_pr["head"]["sha"] = "third-head"
    second = _continue_1069(
        runner, config, first, predecessor="next-head", new_head="third-head", rounds=(200, 201),
    )
    second_record = parse_issue_created_authorization_comment(runner.intent_comments[-1]["body"])
    assert second_record is not None
    assert second_record.predecessor_comment_id == first.authorization_comment_id
    assert second_record.superseded_comment_ids == ()

    live = managed_ci._authorization_comment_records(
        runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
    )
    assert [comment_id for comment_id, _record in live] == [
        41, first.authorization_comment_id, second.authorization_comment_id,
    ]
    audit = _find_resume_audit(
        runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
        base_ref="main", issue_number=643, live_head="third-head",
    )
    assert audit is not None
    assert audit[0] == second.authorization_comment_id


def test_m1069_continuity_supersedes_only_compatible_unbound_records(tmp_path, monkeypatch):
    voluntary = dict(protection="voluntary", waiver="allow-unprotected-managed-ci")
    extra = [
        # 42: compatible, cannot chain to the new head -> retired.
        _unreadable_record(kind="fresh", head_sha="stray-head", nonce="a", **voluntary),
        # 43: recorded a different protection -> kept; recovery adjudicates it.
        _unreadable_record(kind="fresh", head_sha="old-unreadable", nonce="b"),
        # 44: foreign managed-label provenance -> not merely stranded; kept.
        _unreadable_record(
            kind="fresh", head_sha="foreign-label", nonce="c", label_event_id=999, **voluntary,
        ),
        # 45: a different approved plan -> kept.
        _unreadable_record(
            kind="fresh", head_sha="other-plan", nonce="d", approved_plan_hash="b" * 64,
            **voluntary,
        ),
        # 46: compatible and its head is an ancestor of the new head -> kept.
        _unreadable_record(kind="fresh", head_sha="ancestor-head", nonce="e", **voluntary),
    ]
    runner = _continuity_1069_runner(extra)
    _ancestors_1069(monkeypatch, {"abc123", "ancestor-head"})
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    handoff = replace(_authorization_handoff(), authorization_comment_id=41)

    _continue_1069(
        runner, config, handoff, predecessor="abc123", new_head="next-head", rounds=(88, 89),
    )

    record = parse_issue_created_authorization_comment(runner.intent_comments[-1]["body"])
    assert record is not None
    assert record.superseded_comment_ids == (42,)


def _discarded_child_1069(head):
    return _unreadable_record(
        kind="continuity", head_sha=head, nonce="child",
        protection="voluntary", waiver="allow-unprotected-managed-ci",
        predecessor_head="abc123", predecessor_comment_id=41, round_comment_ids=(88, 89),
    )


def test_m1069_discarded_continuity_child_is_retired_not_a_fork(tmp_path, monkeypatch):
    # An interrupted attempt authorized abc123 -> discarded-head; the branch
    # then returned to abc123 and the next coder turn produced next-head.
    runner = _continuity_1069_runner([_discarded_child_1069("discarded-head")])
    _ancestors_1069(monkeypatch, {"abc123"})
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )

    continued = _continue_1069(
        runner, config, replace(_authorization_handoff(), authorization_comment_id=41),
        predecessor="abc123", new_head="next-head", rounds=(88, 89),
    )

    record = parse_issue_created_authorization_comment(runner.intent_comments[-1]["body"])
    assert record is not None
    assert record.predecessor_comment_id == 41
    assert record.superseded_comment_ids == (42,)
    audit = _find_resume_audit(
        runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
        base_ref="main", issue_number=643, live_head="next-head",
    )
    assert audit is not None
    assert audit[0] == continued.authorization_comment_id


def test_m1069_child_that_reaches_the_new_head_still_refuses_as_fork(tmp_path, monkeypatch):
    runner = _continuity_1069_runner([_discarded_child_1069("middle-head")])
    _ancestors_1069(monkeypatch, {"abc123", "middle-head"})
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    posted = len(runner.intent_comments)

    with pytest.raises(AgentLoopError, match="forked predecessor"):
        _continue_1069(
            runner, config, replace(_authorization_handoff(), authorization_comment_id=41),
            predecessor="abc123", new_head="next-head", rounds=(88, 89),
        )
    assert len(runner.intent_comments) == posted


def test_m1069_ordinary_recovery_admits_same_protection_stranded_record(tmp_path):
    # The ordinary resume must get past recovery for a continuity round to
    # retire the stranded record at all.
    stray = _unreadable_record(kind="fresh", head_sha="stray-head", nonce="stray")
    runner = _recovery_runner([_unreadable_record(), stray])

    handoff = _recover(runner, _cloud_config(tmp_path, managed_ci_pr_mode=True))

    assert handoff is not None
    assert handoff.protection_mode == "unreadable"
    assert _mutations(runner) == []


def test_m1069_protection_disagreement_is_left_to_recovery_refusal(tmp_path):
    # A record whose protection differs is not ordinary-path history: recovery
    # refuses before any continuity round, and fresh adjudicates the state.
    stray = _unreadable_record(
        kind="fresh", head_sha="stray-head", nonce="stray",
        protection="voluntary", waiver="allow-unprotected-managed-ci",
    )
    runner = _recovery_runner([_unreadable_record(), stray])

    with pytest.raises(AgentLoopError, match="refused"):
        _recover(runner, _cloud_config(tmp_path, managed_ci_pr_mode=True))
    assert _mutations(runner) == []


def test_m1069_failed_comparison_keeps_the_record(tmp_path, monkeypatch):
    stray = _unreadable_record(
        kind="fresh", head_sha="stray-head", nonce="stray",
        protection="voluntary", waiver="allow-unprotected-managed-ci",
    )
    runner = _continuity_1069_runner([stray])

    def unavailable(_runner, *, config, predecessor_head, live_head):
        raise AgentLoopError("Managed-CI API request failed: compare.")

    monkeypatch.setattr(managed_ci, "_github_proves_descendant", unavailable)
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )

    _continue_1069(
        runner, config, replace(_authorization_handoff(), authorization_comment_id=41),
        predecessor="abc123", new_head="next-head", rounds=(88, 89),
    )

    record = parse_issue_created_authorization_comment(runner.intent_comments[-1]["body"])
    assert record is not None
    assert record.superseded_comment_ids == ()


def test_m1069_continuity_supersession_is_honored_and_round_trips():
    base = _unreadable_record()
    continuity = replace(
        base, kind="continuity", head_sha="next", nonce="c", predecessor_head="abc123",
        predecessor_comment_id=40, round_comment_ids=(88, 89), superseded_comment_ids=(41,),
    )
    assert parse_issue_created_authorization_comment(
        str(format_issue_created_authorization_comment(continuity))
    ) == continuity
    assert managed_ci._superseded_authorization_ids(
        [(40, base), (41, replace(base, head_sha="stray")), (43, continuity)]
    ) == frozenset({41})


_TRUST_PAIRS = ["--human-reviewer-trusted-actor", "alice:5", "--human-reviewer-trusted-actor", "bob:6"]


def _trust_config(tmp_path, **overrides):
    from coding_review_agent_loop.github import TrustedHumanActor

    return make_config(
        tmp_path,
        human_reviewer_trusted_actors=(TrustedHumanActor("alice", 5), TrustedHumanActor("bob", 6)),
        **overrides,
    )


def _trust_values(argv):
    return [argv[i + 1] for i, token in enumerate(argv) if token == "--human-reviewer-trusted-actor"]


def test_recovery_invocation_replay_keeps_trusted_human_entries_as_option_values(tmp_path):
    """#1022: LOGIN:ID values are option payloads, never the positional."""
    parser = build_parser()
    config = _trust_config(
        tmp_path,
        invocation_argv=(
            "agent-loop", "issue", "--human-reviewer-trusted-actor", "alice:5", "643",
            "--human-reviewer-trusted-actor", "bob:6", "--managed-ci",
            "--managed-ci-trusted-actor", "agent-loop",
        ),
    )
    for target, identifier in (("issue", 643), ("pr", 7)):
        rendered = _render_recovery_command(
            config, target=target, identifier=identifier, managed_ci=target == "issue",
        )
        argv = shlex.split(rendered)
        args = parser.parse_args(argv[1:])
        assert args.command == target
        assert (args.issue_number if target == "issue" else args.pr_number) == identifier
        assert args.human_reviewer_trusted_actor == ["alice:5", "bob:6"]


@pytest.mark.parametrize("managed", [True, False])
def test_recovery_config_fallback_keeps_trusted_human_entries(tmp_path, managed):
    parser = build_parser()
    config = _trust_config(tmp_path, managed_ci_trusted_actor="agent-loop")
    rendered = _render_recovery_command(config, target="pr", identifier=7, managed_ci=managed)
    argv = shlex.split(rendered)
    assert _trust_values(argv) == ["alice:5", "bob:6"]
    args = parser.parse_args(argv[1:])
    assert args.pr_number == 7
    assert args.human_reviewer_trusted_actor == ["alice:5", "bob:6"]


@pytest.mark.parametrize("managed", [True, False])
def test_ci_rerun_command_without_invocation_keeps_trusted_human_entries(tmp_path, managed):
    parser = build_parser()
    config = _trust_config(tmp_path, managed_ci=managed, auto_merge=True)
    rendered = _render_ci_rerun_command(config, pr_number=7)
    argv = shlex.split(rendered)
    assert argv[:3] == ["agent-loop", "pr", "7"]
    assert _trust_values(argv) == ["alice:5", "bob:6"]
    args = parser.parse_args(argv[1:])
    assert args.pr_number == 7
    assert args.human_reviewer_trusted_actor == ["alice:5", "bob:6"]


def test_recovery_renderer_emits_no_trust_options_when_unconfigured(tmp_path):
    rendered = _render_recovery_command(
        make_config(tmp_path), target="pr", identifier=7, managed_ci=False, include_context=False,
    )
    assert "--human-reviewer-trusted-actor" not in rendered


# ---------------------------------------------------------------------------
# #1067: failed managed-CI activation reports its measured state and restores
# readiness only when this run's own draft conversion is proven.
# ---------------------------------------------------------------------------


class M1067Runner(ManualQualificationRunner):
    """Scriptable GitHub model for failed-activation state reporting."""

    def __init__(
        self,
        *,
        undo_returncode=0,
        undo_applies=True,
        ready_returncode=0,
        ready_applies=True,
        delete_returncode=0,
        delete_applies=True,
        post_returncode=0,
        post_applies=True,
        events_script=None,
        events_unreadable=False,
        base_commit_stdout=None,
        hooks=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.undo_returncode = undo_returncode
        self.undo_applies = undo_applies
        self.ready_returncode = ready_returncode
        self.ready_applies = ready_applies
        self.delete_returncode = delete_returncode
        self.delete_applies = delete_applies
        self.post_returncode = post_returncode
        self.post_applies = post_applies
        self.events_script = list(events_script or [])
        self.events_unreadable = events_unreadable
        self.base_commit_stdout = base_commit_stdout
        self.hooks = dict(hooks or {})
        # Queued results for plain ``pulls/7`` reads: (stdout, returncode) or
        # an exception to raise at launch.
        self.read_queue = []

    def _hook(self, name):
        hook = self.hooks.get(name)
        if hook is not None:
            hook(self)

    def _run_locked(self, args, *, cwd, check, input_text=None):
        cmd = list(args)
        endpoint = next(
            (part for part in cmd if isinstance(part, str) and part.startswith("repos/")), ""
        )
        if cmd[:3] == ["gh", "pr", "ready"]:
            cmd, cwd_path = self._record_command(args, cwd)
            undo = "--undo" in cmd
            if (undo and self.undo_applies) or (not undo and self.ready_applies):
                self.rest_pr["draft"] = undo
            self._hook("undo" if undo else "ready")
            returncode = self.undo_returncode if undo else self.ready_returncode
            return CommandResult(cmd, cwd_path, "", "" if returncode == 0 else "HTTP 502", returncode)
        if cmd == ["gh", "api", "repos/OWNER/REPO/pulls/7"] and self.read_queue:
            item = self.read_queue.pop(0)
            if isinstance(item, BaseException):
                raise item
            cmd, cwd_path = self._record_command(args, cwd)
            stdout, returncode = item
            return CommandResult(cmd, cwd_path, stdout, "", returncode)
        if endpoint.startswith("repos/OWNER/REPO/issues/7/events?") and (
            self.events_unreadable or self.events_script
        ):
            cmd, cwd_path = self._record_command(args, cwd)
            events = self.events_script.pop(0) if self.events_script else None
            if events is None:
                return CommandResult(cmd, cwd_path, "", "events unavailable", 1)
            return CommandResult(cmd, cwd_path, json.dumps(events), "", 0)
        if endpoint == "repos/OWNER/REPO/issues/7/labels" and "POST" in cmd and self.post_returncode:
            cmd, cwd_path = self._record_command(args, cwd)
            if self.post_applies:
                self.rest_pr["labels"] = [{"name": MANAGED_LABEL}]
            self._hook("post")
            return CommandResult(cmd, cwd_path, "", "HTTP 502", self.post_returncode)
        if (
            endpoint == f"repos/OWNER/REPO/issues/7/labels/{MANAGED_LABEL}"
            and "DELETE" in cmd
            and self.delete_returncode
        ):
            cmd, cwd_path = self._record_command(args, cwd)
            if self.delete_applies:
                self.rest_pr["labels"] = []
            self._hook("delete")
            return CommandResult(cmd, cwd_path, "", "HTTP 502", self.delete_returncode)
        if endpoint == "repos/OWNER/REPO/commits/main" and self.base_commit_stdout is not None:
            cmd, cwd_path = self._record_command(args, cwd)
            return CommandResult(cmd, cwd_path, self.base_commit_stdout, "", 0)
        result = super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)
        if endpoint == "repos/OWNER/REPO/issues/7/labels" and "POST" in cmd:
            self._hook("post")
        if endpoint == f"repos/OWNER/REPO/issues/7/labels/{MANAGED_LABEL}" and "DELETE" in cmd:
            self._hook("delete")
        return result


_M1067_FORBIDDEN_TEXT = (
    "Remove the label",
    "could not be removed",
    "continuing with ordinary CI",
    "is now draft and unlabeled",
    "applied the label",
    "this run applied",
    "--method DELETE",
    "gh pr view",
)


def _m1067_assert_text(text):
    """Operator-text audit (#1067 step 14) for every post-mutation exit."""
    for phrase in _M1067_FORBIDDEN_TEXT:
        assert phrase not in text, phrase
    assert text.count("Before this run PR #7 was") <= 1


def _m1067_commands(runner):
    return [command for command, _cwd in runner.commands]


def _m1067_ready_calls(runner):
    return [c for c in _m1067_commands(runner) if c[:3] == ["gh", "pr", "ready"] and "--undo" not in c]


def _m1067_undo_calls(runner):
    return [c for c in _m1067_commands(runner) if c[:3] == ["gh", "pr", "ready"] and "--undo" in c]


def _m1067_label_posts(runner):
    return [
        index for index, c in enumerate(_m1067_commands(runner))
        if c[:5] == ["gh", "api", "--method", "POST", "repos/OWNER/REPO/issues/7/labels"]
    ]


def _m1067_label_deletes(runner):
    return [
        index for index, c in enumerate(_m1067_commands(runner))
        if c[:4] == ["gh", "api", "--method", "DELETE"] and c[-1].endswith(f"/labels/{MANAGED_LABEL}")
    ]


def _m1067_config(tmp_path, **overrides):
    return make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci", "--managed-ci-trusted-actor", "agent-loop",
        ),
        **overrides,
    )


def _m1067_ready_runner(**kwargs):
    rest_pr = {"state": "open", "draft": False, "labels": [], "body": "Fixes #643"}
    rest_pr.update(kwargs.pop("rest_pr", {}))
    kwargs.setdefault("workflow", SUPPRESSING_V2_WORKFLOW)
    kwargs.setdefault("issue_events", [])
    kwargs.setdefault("pr_branch_protection_payload", {"contexts": [FINAL_CONTEXT]})
    return M1067Runner(rest_pr=rest_pr, **kwargs)


def _m1067_draft_labeled_runner(**kwargs):
    rest_pr = {"state": "open", "draft": True, "labels": [{"name": MANAGED_LABEL}], "body": "Fixes #643"}
    rest_pr.update(kwargs.pop("rest_pr", {}))
    kwargs.setdefault("workflow", SUPPRESSING_V2_WORKFLOW)
    kwargs.setdefault("pr_branch_protection_payload", {"contexts": [FINAL_CONTEXT]})
    return M1067Runner(rest_pr=rest_pr, **kwargs)


def _m1067_activate(runner, config, *, lifecycle="ready-unlabeled-reentry", metadata_override=None):
    return activate_managed_ci(
        runner,
        config=config,
        pr_number=7,
        metadata=metadata_override or _ready_issue_metadata(),
        managed_resume=AuthenticatedManagedResume(origin="issue-created", lifecycle=lifecycle),
    )


def _m1067_fail(runner, config, **kwargs):
    with pytest.raises(AgentLoopError) as raised:
        _m1067_activate(runner, config, **kwargs)
    text = str(raised.value)
    _m1067_assert_text(text)
    return raised.value, text


def test_m1067_ready_reentry_restores_readiness_after_release(tmp_path):
    runner = _m1067_ready_runner(unreadable_issue_events_after_label=True)

    error, text = _m1067_fail(runner, _m1067_config(tmp_path))

    assert type(error) is AgentLoopError
    assert text.startswith("--managed-ci requested qualification, but activation failed")
    # Review item-5: the opening current state is the post-restoration read.
    assert (
        "Before this run PR #7 was ready/unlabeled; after the failure and before the readiness "
        "attempt it was draft/unlabeled; it is now ready/unlabeled."
    ) in text
    assert "it is now draft/unlabeled" not in text
    assert "its ready-to-draft request was acknowledged" in text
    assert f"its `{MANAGED_LABEL}` label request was acknowledged" in text
    assert "Restored to ready/unlabeled as found; no qualification is claimed for this head." in text
    assert "deliberate fail-closed return to ordinary CI" in text
    assert len(_m1067_ready_calls(runner)) == 1
    assert runner.rest_pr["draft"] is False and runner.rest_pr["labels"] == []
    posts = _m1067_label_posts(runner)
    deletes = _m1067_label_deletes(runner)
    assert len(posts) == 1 and len(deletes) == 1 and posts[0] < deletes[0]
    assert "gh pr ready 7" not in text


def test_m1067_failed_readiness_command_prints_full_tuple_manual_undo(tmp_path):
    runner = _m1067_ready_runner(
        unreadable_issue_events_after_label=True, ready_returncode=1, ready_applies=False,
    )

    _error, text = _m1067_fail(runner, _m1067_config(tmp_path))

    assert "Readiness was not restored: the readiness command failed" in text
    assert "Manual undo: confirm every field with `gh api repos/OWNER/REPO/pulls/7 --jq" in text
    assert "`gh pr ready 7 --repo OWNER/REPO`" in text
    assert "no qualification is implied" in text
    for expected in (
        "head_repo=OWNER/REPO", "head_ref=agent-loop/managed-643", "head_sha=abc123",
        "base_ref=main", "author_login=agent-loop", "author_id=1", "state=open",
    ):
        assert expected in text
    assert "author_id: (.user.id? // null)" in text
    # The printed confirmation exposes exactly the snapshot field set.
    query = managed_ci._pr_inspection_command(_m1067_config(tmp_path), 7)
    names = re.findall(r"[{,] (\w+): ", shlex.split(query)[-1].replace("{", "{ ", 1))
    assert names == [name for name, _path in managed_ci._ACTIVATION_TUPLE_FIELDS]
    assert len(_m1067_ready_calls(runner)) == 1


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("head_sha", lambda r: r.rest_pr["head"].__setitem__("sha", "moved")),
        ("base_ref", lambda r: r.rest_pr.__setitem__("base", {"ref": "release"})),
        ("author_id", lambda r: r.rest_pr.__setitem__("user", {"login": "agent-loop", "id": "x"})),
        ("labels (labeled)", lambda r: r.rest_pr.__setitem__("labels", [{"name": MANAGED_LABEL}])),
    ],
)
def test_m1067_acknowledged_readiness_with_mismatched_read_back_is_unverified(
    tmp_path, field, mutate
):
    runner = _m1067_ready_runner(unreadable_issue_events_after_label=True, hooks={"ready": mutate})

    _error, text = _m1067_fail(runner, _m1067_config(tmp_path))

    assert "Readiness command acknowledged; restoration not verified" in text
    assert field in text
    assert "as found" not in text
    assert "Do not ready or unlabel the PR." in text
    assert "gh pr ready 7" not in text
    assert len(_m1067_ready_calls(runner)) == 1
    if field == "labels (labeled)":
        # Review item-5: suppression claims use the read-back, not the
        # pre-restoration measurement.
        assert "Ordinary CI can resume" not in text
        assert "is present again; suppression is still active" in text


def test_m1067_unreadable_read_back_after_readiness_attempt_is_unknown(tmp_path):
    runner = _m1067_ready_runner(
        unreadable_issue_events_after_label=True,
        hooks={"ready": lambda r: r.read_queue.append(("", 1))},
    )

    _error, text = _m1067_fail(runner, _m1067_config(tmp_path))

    assert "Readiness command acknowledged; restoration not verified (the state could not be re-read)" in text
    assert (
        "Before this run PR #7 was ready/unlabeled; after the failure and before the readiness "
        "attempt it was draft/unlabeled; it is now unknown: the current state could not be re-read."
    ) in text
    assert "it is now draft/unlabeled" not in text
    assert "draft/unlabeled re-entry state" not in text
    assert "manual-merge state is suspended" not in text
    assert "Ordinary CI can resume" not in text
    assert "Do not ready or unlabel the PR." in text
    assert len(_m1067_ready_calls(runner)) == 1


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("head_sha", lambda r: r.rest_pr["head"].__setitem__("sha", "moved")),
        ("head_ref", lambda r: r.rest_pr["head"].__setitem__("ref", "agent-loop/managed-999")),
        ("base_ref", lambda r: r.rest_pr.__setitem__("base", {"ref": "release"})),
        ("author_login", lambda r: r.rest_pr.__setitem__("user", {"login": "other", "id": 1})),
        ("head_repo", lambda r: r.rest_pr["head"].__setitem__("repo", {"full_name": "FORK/REPO"})),
        ("state", lambda r: r.rest_pr.__setitem__("state", "closed")),
    ],
)
def test_m1067_changed_tuple_blocks_restoration(tmp_path, field, mutate):
    runner = _m1067_ready_runner(unreadable_issue_events_after_label=True, hooks={"delete": mutate})

    _error, text = _m1067_fail(runner, _m1067_config(tmp_path))

    assert f"Readiness was not restored: tuple fields changed ({field}" in text
    assert "Do not ready or unlabel the PR." in text
    assert f"changed, unreadable, or labeled: {field}" in text
    assert "gh pr ready 7" not in text
    assert _m1067_ready_calls(runner) == []


@pytest.mark.parametrize("labeled", [False, True])
def test_m1067_pr_readied_by_another_actor_is_not_claimed_as_restored(tmp_path, labeled):
    def ready_elsewhere(runner):
        runner.rest_pr["draft"] = False
        if labeled:
            runner.rest_pr["labels"] = [{"name": MANAGED_LABEL}]

    runner = _m1067_ready_runner(
        unreadable_issue_events_after_label=True, hooks={"delete": ready_elsewhere}
    )

    _error, text = _m1067_fail(runner, _m1067_config(tmp_path))

    assert "the PR is already ready" in text
    assert "does not know who made it ready" in text
    assert "Restored" not in text and "as found" not in text
    assert "Do not ready or unlabel the PR." in text
    assert ("labels (labeled)" in text) is labeled
    assert _m1067_ready_calls(runner) == []


def test_m1067_foreign_label_event_preflight_refuses_restoration(tmp_path):
    runner = _m1067_draft_labeled_runner(
        issue_events=[label_event(login="someone", actor_id=9)]
    )

    _error, text = _m1067_fail(runner, _m1067_config(tmp_path), lifecycle="draft-labeled")

    assert "ownership was not the trusted actor's" in text
    assert "Before this run PR #7 was draft/labeled; it is now draft/unlabeled." in text
    assert "Do not ready or unlabel the PR." in text
    assert _m1067_ready_calls(runner) == []
    assert _m1067_label_posts(runner) == []
    assert len(_m1067_label_deletes(runner)) == 1


def test_m1067_foreign_event_after_reentry_draft_vetoes_restoration(tmp_path):
    foreign = [label_event(login="someone", actor_id=9)]
    # The first read is the label add's pre-write baseline (#510).
    runner = _m1067_ready_runner(events_script=[[], foreign, foreign])

    _error, text = _m1067_fail(runner, _m1067_config(tmp_path))

    # The post-state looks restorable, but the integrity failure vetoes it.
    assert "it is now draft/unlabeled" in text
    assert "ownership was not the trusted actor's" in text
    assert "Automatic readiness restoration was refused for ownership integrity" in text
    assert "Do not ready or unlabel the PR." in text
    assert "gh pr ready 7" not in text
    assert _m1067_ready_calls(runner) == []
    posts = _m1067_label_posts(runner)
    deletes = _m1067_label_deletes(runner)
    assert len(posts) == 1 and len(deletes) == 1 and posts[0] < deletes[0]


def test_m1067_label_event_changed_before_delete_reports_owner_not_asserted(tmp_path):
    runner = _m1067_ready_runner(
        events_script=[[label_event(101)], [label_event(102)], [label_event(103)]]
    )

    _error, text = _m1067_fail(runner, _m1067_config(tmp_path))

    assert "managed-label ownership changed before ordinary release" in text
    assert "changed or could not be verified; its owner is not asserted" in text
    assert "not the trusted actor's" not in text
    assert "it is now draft/labeled" in text
    assert "Do not ready or unlabel the PR." in text
    assert _m1067_label_deletes(runner) == []
    assert _m1067_ready_calls(runner) == []


def test_m1067_labeled_first_read_back_after_acknowledged_undo(tmp_path):
    runner = _m1067_ready_runner(
        hooks={"undo": lambda r: r.rest_pr.__setitem__("labels", [{"name": MANAGED_LABEL}])}
    )

    _error, text = _m1067_fail(runner, _m1067_config(tmp_path))

    assert "re-entry could not make PR #7 provably draft and unlabeled" in text
    assert "unchanged" not in text
    assert "Readiness was not restored: the PR is still labeled." in text
    assert _m1067_ready_calls(runner) == []


def test_m1067_failed_first_read_back_restores_after_measured_read(tmp_path):
    runner = _m1067_ready_runner(hooks={"undo": lambda r: r.read_queue.append(("", 1))})

    _error, text = _m1067_fail(runner, _m1067_config(tmp_path))

    assert "unchanged" not in text
    assert "Restored to ready/unlabeled as found" in text
    assert len(_m1067_ready_calls(runner)) == 1


def test_m1067_nonzero_undo_never_licenses_automatic_readiness(tmp_path):
    runner = _m1067_ready_runner(undo_returncode=1, undo_applies=True)

    _error, text = _m1067_fail(runner, _m1067_config(tmp_path))

    assert "its ready-to-draft request was not confirmed" in text
    assert "attribution is ambiguous" in text
    assert "`gh pr ready 7 --repo OWNER/REPO`" in text
    assert "confirm every field" in text
    assert _m1067_ready_calls(runner) == []


def test_m1067_label_create_failure_after_draft_restores_readiness(tmp_path, monkeypatch):
    monkeypatch.setattr(managed_ci, "ensure_managed_label", lambda *_a, **_k: False)
    runner = _m1067_ready_runner()

    _error, text = _m1067_fail(runner, _m1067_config(tmp_path))

    assert text.startswith(f"Unable to create the `{MANAGED_LABEL}` label.")
    assert "Restored to ready/unlabeled as found" in text
    assert len(_m1067_ready_calls(runner)) == 1
    assert _m1067_label_posts(runner) == []


def test_m1067_label_post_failure_after_draft_restores_readiness(tmp_path):
    runner = _m1067_ready_runner(post_returncode=1, post_applies=False)

    _error, text = _m1067_fail(runner, _m1067_config(tmp_path))

    assert "the label request was not confirmed" in text
    assert f"its `{MANAGED_LABEL}` label request was not confirmed" in text
    assert "Restored to ready/unlabeled as found" in text
    # A transient 502 that history proves absent is replayed within the budget (#510).
    assert len(_m1067_label_posts(runner)) == 3
    assert len(_m1067_ready_calls(runner)) == 1


@pytest.mark.parametrize(
    "failure",
    [("", 1), ("not json", 0), ("[]", 0), ("42", 0), OSError("gh could not be launched")],
)
def test_m1067_unreadable_post_failure_state_keeps_original_error(tmp_path, failure):
    runner = _m1067_ready_runner(
        unreadable_issue_events_after_label=True,
        hooks={"delete": lambda r: r.read_queue.append(failure)},
    )

    error, text = _m1067_fail(runner, _m1067_config(tmp_path))

    assert type(error) is AgentLoopError
    assert text.startswith("--managed-ci requested qualification, but activation failed")
    assert "it is now unknown: the current state could not be re-read" in text
    assert "Readiness was not restored: the current state is unknown." in text
    assert "Do not ready or unlabel the PR." in text
    assert _m1067_ready_calls(runner) == []


def test_m1067_empty_object_read_renders_every_field_unreadable(tmp_path):
    runner = _m1067_ready_runner(
        unreadable_issue_events_after_label=True,
        hooks={"delete": lambda r: r.read_queue.append(("{}", 0))},
    )

    _error, text = _m1067_fail(runner, _m1067_config(tmp_path))

    assert "could not be re-read" not in text
    assert "of unknown draft state, with unreadable labels" in text
    for name in ("head_repo", "head_ref", "head_sha", "base_ref", "author_login", "author_id", "state"):
        assert f"{name} unreadable" in text
    assert _m1067_ready_calls(runner) == []


def test_m1067_report_rendering_failure_keeps_original_error(tmp_path, monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("render failed")

    monkeypatch.setattr(managed_ci, "_build_failed_activation_report", broken)
    runner = _m1067_ready_runner(unreadable_issue_events_after_label=True)

    error, text = _m1067_fail(runner, _m1067_config(tmp_path))

    assert type(error) is AgentLoopError
    assert text.startswith("--managed-ci requested qualification, but activation failed")
    assert "Post-failure state report unavailable (RuntimeError)" in text
    assert "gh pr ready 7" not in text
    # Review item-6: the fixed fallback still gives remedy (C).
    assert "Do not ready or unlabel the PR. Inspect it with `gh api repos/OWNER/REPO/pulls/7 --jq" in text
    assert "then rerun `agent-loop pr 7" in text


def test_m1067_report_fallback_survives_inspection_rendering_failure(tmp_path, monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("render failed")

    monkeypatch.setattr(managed_ci, "_build_failed_activation_report", broken)
    monkeypatch.setattr(managed_ci, "_pr_inspection_command", broken)
    runner = _m1067_ready_runner(unreadable_issue_events_after_label=True)

    error, text = _m1067_fail(runner, _m1067_config(tmp_path))

    assert type(error) is AgentLoopError
    assert text.startswith("--managed-ci requested qualification, but activation failed")
    assert "Post-failure state report unavailable (RuntimeError)" in text
    assert "Resume with `agent-loop pr 7" in text
    assert "gh pr ready 7" not in text


def test_m1067_recovery_incapable_base_reports_without_release(tmp_path):
    runner = _m1067_ready_runner(
        workflow=SUPPRESSING_V2_WORKFLOW_WITHOUT_RECOVERY,
        unreadable_issue_events_after_label=True,
    )

    contract = _m1067_activate(runner, _m1067_config(tmp_path))

    assert contract is not None
    assert contract.activation_path == "ordinary_fallback"
    assert contract.ordinary_recovery is None
    report = contract.state_report
    assert report is not None
    _m1067_assert_text(report)
    assert f"its `{MANAGED_LABEL}` label request was acknowledged" in report
    assert "Ordinary CI can resume" not in report
    assert "base workflow has no unlabeled recovery route" in report
    assert "Do not ready or unlabel the PR." in report
    assert _m1067_label_deletes(runner) == []
    assert _m1067_ready_calls(runner) == []


def test_m1067_state_report_default_keeps_contract_equality():
    report = "Before this run PR #7 was ready/unlabeled; it is now draft/labeled."
    contract = ManagedCiContract(activation_path="ordinary_fallback", state_report=report)
    assert contract.state_report == report
    # The default keeps every existing constructor and equality; the
    # orchestrator print is exercised through run_pr_loop in
    # tests/test_orchestrator_pr.py.
    assert ManagedCiContract(activation_path="ordinary_fallback") == ManagedCiContract(
        activation_path="ordinary_fallback", state_report=None
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"labels": [5]},
        {"labels": [{"name": MANAGED_LABEL}, "junk"]},
        {"labels": [{"name": 3}]},
        {"labels": None},
        {"draft": "yes"},
        {"draft": None},
    ],
)
def test_m1067_malformed_entry_fails_closed_before_any_write(tmp_path, payload):
    runner = _m1067_ready_runner(rest_pr=payload)

    with pytest.raises(AgentLoopError, match="could not be read strictly") as raised:
        managed_ci._activate_v2_managed_ci(
            runner, config=_m1067_config(tmp_path), pr_number=7, metadata=_ready_issue_metadata(),
            managed_resume=AuthenticatedManagedResume(
                origin="issue-created", lifecycle="ready-unlabeled-reentry"
            ),
        )

    assert "Before this run" not in str(raised.value)
    assert _m1067_undo_calls(runner) == [] and _m1067_ready_calls(runner) == []
    assert _m1067_label_posts(runner) == [] and _m1067_label_deletes(runner) == []


def test_m1067_malformed_snapshot_never_renders_as_unlabeled_or_ready():
    snapshot = managed_ci._PrStateSnapshot.from_payload({"labels": [1], "draft": "x"})
    rendered = snapshot.render()
    assert "with unreadable labels" in rendered and "of unknown draft state" in rendered
    assert "unlabeled" not in rendered and "ready" not in rendered
    partial = managed_ci._PrStateSnapshot.from_payload({
        "labels": [{"name": MANAGED_LABEL}, 7], "draft": True, "state": "open",
        "head": {"repo": {"full_name": "OWNER/REPO"}, "ref": "r", "sha": "s"},
        "base": {"ref": "main"}, "user": {"login": "agent-loop", "id": 1},
    })
    assert partial.render() == "draft, with unreadable labels"


@pytest.mark.parametrize(
    "state",
    ["missing", None, "merged", "OPEN", [], ["open"], {}, {"value": "open"}],
)
def test_m1067_authenticated_pr_not_provably_open_is_refused(tmp_path, state):
    runner = _m1067_ready_runner()
    if state == "missing":
        runner.rest_pr.pop("state")
    else:
        runner.rest_pr["state"] = state

    with pytest.raises(AgentLoopError, match="is not provably open") as raised:
        managed_ci._activate_v2_managed_ci(
            runner, config=_m1067_config(tmp_path), pr_number=7, metadata=_ready_issue_metadata(),
            managed_resume=AuthenticatedManagedResume(
                origin="issue-created", lifecycle="ready-unlabeled-reentry"
            ),
        )

    assert "does not match the authenticated issue-created" not in str(raised.value)
    assert _m1067_undo_calls(runner) == [] and _m1067_ready_calls(runner) == []
    assert _m1067_label_posts(runner) == [] and _m1067_label_deletes(runner) == []


def test_m1067_closed_pr_keeps_none_route(tmp_path):
    runner = _m1067_ready_runner(rest_pr={"state": "closed"})

    assert managed_ci._activate_v2_managed_ci(
        runner, config=_m1067_config(tmp_path), pr_number=7, metadata=_ready_issue_metadata(),
        managed_resume=AuthenticatedManagedResume(
            origin="issue-created", lifecycle="ready-unlabeled-reentry"
        ),
    ) is None
    assert _m1067_undo_calls(runner) == [] and _m1067_label_posts(runner) == []


_M1067_MALFORMED = {"labels": [3], "draft": "x"}


@pytest.mark.parametrize(
    "identity",
    [
        {"user": {"login": "someone", "id": 55}},
        {"head": {"repo": None, "sha": "abc123", "ref": "agent-loop/managed-643"}},
        {"head": {"repo": {"full_name": "OWNER/REPO"}, "sha": "abc123", "ref": "feature"}},
        {"user": {"login": "agent-loop", "id": True}},
    ],
)
@pytest.mark.parametrize("state", ["missing", [], {}, {"value": "open"}])
@pytest.mark.parametrize("managed", [True, False])
def test_m1067_non_managed_prs_keep_none_route(tmp_path, identity, state, managed):
    runner = _m1067_ready_runner(rest_pr={**_M1067_MALFORMED, **identity})
    if state == "missing":
        runner.rest_pr.pop("state")
    else:
        runner.rest_pr["state"] = state
    config = _m1067_config(tmp_path) if managed else make_config(
        tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop",
    )

    assert managed_ci._activate_v2_managed_ci(
        runner, config=config, pr_number=7, metadata=_ready_issue_metadata(),
    ) is None
    assert _m1067_undo_calls(runner) == [] and _m1067_ready_calls(runner) == []
    assert _m1067_label_posts(runner) == [] and _m1067_label_deletes(runner) == []
    assert not any("comments" in " ".join(c) and "POST" in c for c in _m1067_commands(runner))


def test_m1067_non_managed_pr_implicit_caller_keeps_ordinary_outcome(tmp_path):
    runner = _m1067_ready_runner(rest_pr={**_M1067_MALFORMED, "user": {"login": "someone", "id": 55}})
    config = make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop")

    assert activate_managed_ci(
        runner, config=config, pr_number=7, metadata=_ready_issue_metadata(),
    ) is None
    assert _m1067_label_posts(runner) == [] and _m1067_label_deletes(runner) == []


def test_m1067_explicit_adoption_is_not_preempted_by_the_gate(tmp_path):
    runner = _m1067_ready_runner(rest_pr=_M1067_MALFORMED)
    config = make_config(
        tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop",
        managed_ci_adopt_existing_pr=True,
    )

    assert managed_ci._activate_v2_managed_ci(
        runner, config=config, pr_number=7, metadata=_ready_issue_metadata(),
    ) is None


def test_m1067_boolean_actor_id_never_authenticates(tmp_path):
    runner = _m1067_ready_runner(actor_id=True, rest_pr={"user": {"login": "agent-loop", "id": True}})

    assert managed_ci._activate_v2_managed_ci(
        runner, config=_m1067_config(tmp_path), pr_number=7, metadata=_ready_issue_metadata(),
        managed_resume=AuthenticatedManagedResume(
            origin="issue-created", lifecycle="ready-unlabeled-reentry"
        ),
    ) is None
    assert _m1067_undo_calls(runner) == [] and _m1067_label_posts(runner) == []


def test_m1067_strict_int_and_tuple_mismatch_reject_booleans():
    assert managed_ci._strict_int(True) is None
    assert managed_ci._strict_int(False) is None
    assert managed_ci._strict_int(7) == 7
    payload = {
        "head": {"repo": {"full_name": "OWNER/REPO"}, "ref": "r", "sha": "s"},
        "base": {"ref": "main"}, "user": {"login": "agent-loop", "id": 1},
        "state": "open", "draft": True, "labels": [],
    }
    entry = managed_ci._PrStateSnapshot.from_payload(payload)
    live = managed_ci._PrStateSnapshot.from_payload(
        {**payload, "user": {"login": "AGENT-LOOP", "id": True}}
    )
    assert managed_ci._entry_tuple_mismatches(entry, live) == ["author_id"]


def test_m1067_draft_labeled_preflight_release_reports_once(tmp_path):
    runner = _m1067_draft_labeled_runner(events_unreadable=True)

    error, text = _m1067_fail(runner, _m1067_config(tmp_path), lifecycle="draft-labeled")

    assert type(error) is AgentLoopError
    assert "the active managed-label event is temporarily unreadable" in text
    assert "Before this run PR #7 was draft/labeled; it is now draft/unlabeled." in text
    assert "deliberate fail-closed return to ordinary CI" in text
    assert "Ordinary CI can resume through the base workflow's unlabeled recovery route." in text
    assert "The PR is in the documented draft/unlabeled re-entry state." in text
    assert "Resume with `agent-loop pr 7" in text
    assert "gh pr ready" not in text
    assert _m1067_label_posts(runner) == [] and _m1067_ready_calls(runner) == []
    assert not runner.audit_comments and not runner.intent_comments


def test_m1067_relabel_after_release_reports_active_suppression(tmp_path):
    runner = _m1067_draft_labeled_runner(
        events_unreadable=True,
        hooks={"delete": lambda r: r.rest_pr.__setitem__("labels", [{"name": MANAGED_LABEL}])},
    )

    _error, text = _m1067_fail(runner, _m1067_config(tmp_path), lifecycle="draft-labeled")

    assert "is present again; suppression is still active and ordinary CI has not resumed" in text
    assert "Ordinary CI can resume" not in text
    assert "draft/unlabeled re-entry state" not in text
    assert "Do not ready or unlabel the PR." in text
    assert _m1067_label_posts(runner) == []


def test_m1067_missing_authorization_preflight_release_reports_once(tmp_path):
    stale = ManagedCiIssueAuthorization(
        kind="creation", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="older-head", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci", nonce="old",
        label_event_id=101,
    )
    runner = _m1067_draft_labeled_runner(
        rest_pr={"head": {
            "repo": {"full_name": "OWNER/REPO"}, "sha": "new-head", "ref": "agent-loop/managed-643",
        }},
        pr_branch_protection_payload=None,
        issue_events=[label_event()],
        intent_comments=[{
            "id": 41,
            "user": {"login": "agent-loop", "id": 1},
            "body": str(format_issue_created_authorization_comment(stale)),
        }],
    )
    config = _m1067_config(tmp_path, allow_unprotected_managed_ci=True)

    with pytest.raises(AgentLoopError, match="no fully bound actor-owned") as raised:
        activate_managed_ci(
            runner, config=config, pr_number=7,
            metadata=replace(_ready_issue_metadata(), head_sha="new-head"),
            managed_resume=AuthenticatedManagedResume(
                origin="issue-created", lifecycle="draft-labeled",
                issue_created_handoff=_authorization_handoff(head="new-head"),
            ),
        )

    text = str(raised.value)
    _m1067_assert_text(text)
    assert text.count("Before this run PR #7 was draft/labeled; it is now draft/unlabeled.") == 1
    command = text.split("fresh issue-created authorization command: `", 1)[1].split("`", 1)[0]
    parsed = build_parser().parse_args(shlex.split(command)[1:])
    assert parsed.managed_ci_fresh_authorization is True
    assert parsed.managed_ci_issue == 643
    assert f"Resume with `{command}`" in text
    assert _m1067_label_posts(runner) == [] and _m1067_ready_calls(runner) == []


def test_m1067_preflight_failure_without_mutation_keeps_original_message(tmp_path):
    runner = _m1067_draft_labeled_runner(
        rest_pr={"labels": []},
        pr_branch_protection_payload=None,
        repo_payload={"private": True},
        pr_branch_protection_returncode=1,
        pr_branch_protection_stderr="HTTP 403: Upgrade to GitHub Pro or make this repository public",
        pr_effective_rules_returncode=1,
        pr_effective_rules_stderr="HTTP 403: Upgrade to GitHub Pro or make this repository public",
    )

    with pytest.raises(AgentLoopError, match="was left unchanged") as raised:
        _m1067_activate(runner, _m1067_config(tmp_path), lifecycle="draft-unlabeled-reentry")

    assert "Before this run" not in str(raised.value)
    assert _m1067_label_deletes(runner) == [] and _m1067_label_posts(runner) == []


@pytest.mark.parametrize("readied", [False, True])
def test_m1067_failed_delete_on_draft_labeled_entry(tmp_path, readied):
    hooks = {"delete": lambda r: r.rest_pr.__setitem__("draft", False)} if readied else {}
    runner = _m1067_draft_labeled_runner(
        events_unreadable=True, delete_returncode=1, delete_applies=False, hooks=hooks,
    )

    _error, text = _m1067_fail(runner, _m1067_config(tmp_path), lifecycle="draft-labeled")

    assert "the label DELETE was not confirmed" in text
    assert "its label DELETE was not confirmed" in text
    assert "draft/unlabeled re-entry state" not in text
    assert "Do not ready or unlabel the PR." in text
    if readied:
        assert "it is now ready/labeled" in text
        assert "this run made no readiness change" in text
        assert "claims no qualification" in text
        assert "remains draft" not in text
    else:
        assert "It remains draft/labeled and managed qualification was abandoned." in text
    # An unconfirmed DELETE proves no removal, so no re-add is claimed.
    assert f"`{MANAGED_LABEL}` is present; suppression is still active" in text
    assert "present again" not in text
    assert _m1067_ready_calls(runner) == []


def test_m1067_unconfirmed_delete_that_took_effect_is_measured(tmp_path):
    runner = _m1067_draft_labeled_runner(events_unreadable=True, delete_returncode=1)

    _error, text = _m1067_fail(runner, _m1067_config(tmp_path), lifecycle="draft-labeled")

    assert "the label DELETE was not confirmed" in text
    assert "it is now draft/unlabeled" in text


@pytest.mark.parametrize("managed", [False, True])
def test_m1067_dispatch_time_release_reports_measured_state_only(tmp_path, capsys, managed):
    runner = _m1067_draft_labeled_runner(issue_events=[label_event()])
    config = make_config(
        tmp_path, managed_ci=managed, managed_ci_pr_mode=True, managed_ci_trusted_actor="agent-loop",
        quiet=False,
    )

    if managed:
        with pytest.raises(AgentLoopError) as raised:
            _release_for_ordinary_recovery(
                runner, config=config, pr_number=7, base_ref="main", expected_head_sha="abc123",
                active_event=(101, "agent-loop", 1), reason="fresh intent ledger could not be reconciled",
                recovery_capable=True,
            )
        text = str(raised.value)
        _m1067_assert_text(text)
        assert "PR #7 is now draft/unlabeled." in text
        assert "Before this run" not in text
    else:
        recovery = _release_for_ordinary_recovery(
            runner, config=config, pr_number=7, base_ref="main", expected_head_sha="abc123",
            active_event=(101, "agent-loop", 1), reason="fresh intent ledger could not be reconciled",
            recovery_capable=True,
        )
        assert isinstance(recovery, OrdinaryRecoveryCapability)
        assert recovery.released_label_event_id == 101
        err = capsys.readouterr().err
        assert "label DELETE acknowledged; ordinary unlabeled recovery selected" in err
        # Review item-12: the returned-capability path logs the measured state.
        assert "PR #7 is now draft/unlabeled." in err
        assert "Before this run" not in err
    assert len(_m1067_label_deletes(runner)) == 1
    assert _m1067_ready_calls(runner) == []


@pytest.mark.parametrize("labeled", [True, False])
def test_m1067_dispatch_time_recovery_incapable_release_logs_measured_state(tmp_path, capsys, labeled):
    # Review item-12: the context-free recovery-incapable return makes no
    # DELETE and logs the measured current state.
    runner = _m1067_draft_labeled_runner(issue_events=[label_event()])
    if not labeled:
        runner.rest_pr["labels"] = []  # another actor removed the label
    config = make_config(
        tmp_path, managed_ci_pr_mode=True, managed_ci_trusted_actor="agent-loop", quiet=False,
    )

    recovery = _release_for_ordinary_recovery(
        runner, config=config, pr_number=7, base_ref="main", expected_head_sha="abc123",
        active_event=(101, "agent-loop", 1), reason="ledger", recovery_capable=False,
    )

    assert recovery is None
    err = capsys.readouterr().err
    _m1067_assert_text(err)
    assert "ordinary recovery was not selected" in err
    assert f"PR #7 is now draft/{'labeled' if labeled else 'unlabeled'}." in err
    assert "Before this run" not in err
    assert _m1067_label_deletes(runner) == [] and _m1067_ready_calls(runner) == []


@pytest.mark.parametrize("applies", [True, False])
def test_m1067_dispatch_time_unconfirmed_delete_measures_label(tmp_path, applies):
    runner = _m1067_draft_labeled_runner(
        issue_events=[label_event()], delete_returncode=1, delete_applies=applies,
    )
    config = make_config(tmp_path, managed_ci_pr_mode=True, managed_ci_trusted_actor="agent-loop")

    with pytest.raises(AgentLoopError) as raised:
        _release_for_ordinary_recovery(
            runner, config=config, pr_number=7, base_ref="main", expected_head_sha="abc123",
            active_event=(101, "agent-loop", 1), reason="ledger", recovery_capable=True,
        )

    text = str(raised.value)
    _m1067_assert_text(text)
    assert "the label DELETE was not confirmed" in text
    assert f"PR #7 is now draft/{'unlabeled' if applies else 'labeled'}." in text
    assert _m1067_ready_calls(runner) == []


def _m1067_guard(runner, config, context):
    return managed_ci._failed_activation_guard(runner, config=config, pr_number=7, context=context)


def _m1067_context(runner):
    return managed_ci._FailedActivationContext(
        entry=managed_ci._PrStateSnapshot.from_payload(runner.rest_pr), recovery_capable=True,
    )


def test_m1067_guard_preserves_terminal_subclass_and_reports_once(tmp_path):
    runner = _m1067_draft_labeled_runner()
    config = _m1067_config(tmp_path)
    context = _m1067_context(runner)
    context.label_post_attempted = True
    original = managed_ci.ManagedCiHostFooterIncompatibleError(pr_number=7, phase="pre-dispatch")

    with pytest.raises(managed_ci.ManagedCiHostFooterIncompatibleError) as raised:
        with _m1067_guard(runner, config, context):
            raise original

    assert raised.value is original
    assert str(raised.value).count("Before this run PR #7 was") == 1
    # An already-reported exception passes through a second guard untouched.
    with pytest.raises(managed_ci.ManagedCiHostFooterIncompatibleError):
        with _m1067_guard(runner, config, context):
            raise original
    assert str(original).count("Before this run PR #7 was") == 1


def test_m1067_guard_notes_other_exceptions_and_skips_base_exceptions(tmp_path):
    runner = _m1067_draft_labeled_runner()
    config = _m1067_config(tmp_path)
    context = _m1067_context(runner)
    context.drafted_by_this_run = True
    original = ValueError("boom")

    with pytest.raises(ValueError) as raised:
        with _m1067_guard(runner, config, context):
            raise original

    assert raised.value is original and str(original) == "boom"
    assert any("Before this run PR #7 was" in note for note in original.__notes__)
    interrupt = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        with _m1067_guard(runner, config, managed_ci._FailedActivationContext(
            entry=context.entry, drafted_by_this_run=True,
        )):
            raise interrupt
    assert not getattr(interrupt, "__notes__", None)
    untouched = AgentLoopError("left unchanged")
    with pytest.raises(AgentLoopError):
        with _m1067_guard(runner, config, managed_ci._FailedActivationContext(entry=context.entry)):
            raise untouched
    assert str(untouched) == "left unchanged"


@pytest.mark.parametrize("repo", [None, {"full_name": 5}, "OWNER/REPO"])
def test_m1067_malformed_reconstruction_head_repo_is_a_tuple_mismatch(tmp_path, repo):
    runner = _m1067_ready_runner(
        hooks={"undo": lambda r: r.rest_pr["head"].__setitem__("repo", repo)}
    )

    error, text = _m1067_fail(runner, _m1067_config(tmp_path))

    assert type(error) is AgentLoopError
    assert "reconstruction for PR #7 failed after the ready-to-draft transition" in text
    assert "head_repo" in text
    assert _m1067_ready_calls(runner) == []


def test_m1067_invalid_base_commit_json_reports_labeled_state(tmp_path):
    runner = _m1067_ready_runner(base_commit_stdout="not json")

    _error, text = _m1067_fail(runner, _m1067_config(tmp_path))

    assert text.startswith("Managed-CI API response was invalid JSON: repos/OWNER/REPO/commits/main.")
    assert "it is now draft/labeled" in text
    assert f"its `{MANAGED_LABEL}` label request was acknowledged" in text
    assert "Readiness was not restored: the PR is still labeled." in text
    assert "Do not ready or unlabel the PR." in text
    assert _m1067_ready_calls(runner) == []


def test_m1067_concurrent_same_actor_label_is_never_attributed_to_this_run(tmp_path):
    # A concurrent invocation under the same trusted actor already labeled
    # the PR after this run's entry read; this run's POST is idempotent.
    runner = _m1067_ready_runner(
        base_commit_stdout="not json",
        hooks={"undo": lambda r: r.issue_events.append(label_event(90))},
    )

    _error, text = _m1067_fail(runner, _m1067_config(tmp_path))

    assert "label request was acknowledged" in text
    assert "owns" not in text
    assert "gh pr ready 7" not in text


def _m1067_strict_draft_unlabeled_runner(**kwargs):
    return M1067Runner(
        workflow=SUPPRESSING_V2_WORKFLOW,
        rest_pr={
            "state": "open", "draft": True, "labels": [],
            "head": {
                "repo": {"full_name": "OWNER/REPO"}, "sha": "coder-round-head",
                "ref": "agent-loop/managed-643",
            },
        },
        intent_comments=[],
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT]},
        **kwargs,
    )


def _m1067_strict_reentry(runner, tmp_path):
    return activate_managed_ci(
        runner, config=_m1067_config(tmp_path), pr_number=7,
        metadata=replace(_ready_issue_metadata(), head_sha="coder-round-head"),
        managed_resume=AuthenticatedManagedResume(
            origin="issue-created", lifecycle="draft-unlabeled-reentry",
            issue_created_handoff=replace(
                _authorization_handoff(head="coder-round-head"), protection_mode="strict",
            ),
        ),
    )


@pytest.mark.parametrize("applies", [True, False])
def test_m1067_ambiguous_label_post_on_strict_reentry_is_measured(tmp_path, applies):
    runner = _m1067_strict_draft_unlabeled_runner(
        issue_events=[label_event()], post_returncode=1, post_applies=applies,
    )

    with pytest.raises(AgentLoopError) as raised:
        _m1067_strict_reentry(runner, tmp_path)

    text = str(raised.value)
    _m1067_assert_text(text)
    assert f"its `{MANAGED_LABEL}` label request was not confirmed" in text
    if applies:
        assert "Before this run PR #7 was draft/unlabeled; it is now draft/labeled." in text
        assert "Do not ready or unlabel the PR." in text
    else:
        assert "it is now draft/unlabeled." in text
        assert "Resume with `" in text
    assert _m1067_ready_calls(runner) == [] and _m1067_label_deletes(runner) == []


def test_m1067_strict_reentry_refusal_is_an_explicit_exclusion(tmp_path):
    runner = _m1067_strict_draft_unlabeled_runner(issue_events=[])

    with pytest.raises(AgentLoopError) as raised:
        _m1067_strict_reentry(runner, tmp_path)

    text = str(raised.value)
    # Pre-mutation refusal: keeps its reviewed trusted-actor prerequisite.
    assert f"Reapply `{MANAGED_LABEL}` as the configured trusted actor" in text
    assert "Before this run" not in text
    assert _m1067_label_posts(runner) == [] and _m1067_label_deletes(runner) == []
    assert _m1067_undo_calls(runner) == [] and _m1067_ready_calls(runner) == []
    assert runner.dispatch_count == 0


def _m1067_plan_limited_runner(**kwargs):
    return _m1067_draft_labeled_runner(
        pr_branch_protection_payload=None,
        repo_payload={"private": True},
        pr_branch_protection_returncode=1,
        pr_branch_protection_stderr="HTTP 403: Upgrade to GitHub Pro or make this repository public",
        pr_effective_rules_returncode=1,
        pr_effective_rules_stderr="HTTP 403: Upgrade to GitHub Pro or make this repository public",
        issue_events=[label_event()],
        **kwargs,
    )


def _m1067_implicit_activation(runner, config):
    return managed_ci._activate_v2_managed_ci(
        runner, config=config, pr_number=7, metadata=_ready_issue_metadata(),
    )


def test_m1067_implicit_fallback_helper_logs_measured_report(tmp_path, capsys):
    runner = _m1067_plan_limited_runner()
    config = make_config(
        tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop", quiet=False,
    )

    assert _m1067_implicit_activation(runner, config) is None

    err = capsys.readouterr().err
    _m1067_assert_text(err)
    assert "label DELETE acknowledged (strict protection or the explicit override is unavailable)" in err
    assert "Before this run PR #7 was draft/labeled; it is now draft/unlabeled." in err
    assert "Ordinary CI can resume" in err
    assert len(_m1067_label_deletes(runner)) == 1


@pytest.mark.parametrize("variant", ["relabel", "no-recovery"])
def test_m1067_implicit_fallback_helper_variants_make_no_ordinary_claim(tmp_path, capsys, variant):
    kwargs = (
        {"hooks": {"delete": lambda r: r.rest_pr.__setitem__("labels", [{"name": MANAGED_LABEL}])}}
        if variant == "relabel"
        else {"workflow": SUPPRESSING_V2_WORKFLOW_WITHOUT_RECOVERY}
    )
    runner = _m1067_plan_limited_runner(**kwargs)
    config = make_config(
        tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop", quiet=False,
    )

    assert _m1067_implicit_activation(runner, config) is None

    err = capsys.readouterr().err
    _m1067_assert_text(err)
    assert "Ordinary CI can resume" not in err
    if variant == "relabel":
        assert "suppression is still active and ordinary CI has not resumed" in err
    else:
        # Review item-4: a base without the unlabeled recovery route keeps
        # the label; the fallback helper makes no DELETE.
        assert _m1067_label_deletes(runner) == []
        assert runner.rest_pr["labels"] == [{"name": MANAGED_LABEL}]
        assert f"this run made no `{MANAGED_LABEL}` label DELETE" in err
        assert "PR #7 is now draft/labeled." in err
        assert _m1067_ready_calls(runner) == []


def test_m1067_explicit_fallback_helper_on_recovery_incapable_base_retains_label(tmp_path):
    runner = _m1067_plan_limited_runner(workflow=SUPPRESSING_V2_WORKFLOW_WITHOUT_RECOVERY)
    config = make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop")

    with pytest.raises(AgentLoopError) as raised:
        _m1067_implicit_activation(runner, config)

    text = str(raised.value)
    _m1067_assert_text(text)
    assert f"this run made no `{MANAGED_LABEL}` label DELETE" in text
    assert "PR #7 is now draft/labeled." in text
    assert "did NOT qualify" in text
    assert _m1067_label_deletes(runner) == []
    assert _m1067_ready_calls(runner) == []


@pytest.mark.parametrize("delete_returncode", [0, 1])
def test_m1067_explicit_fallback_helper_raise_carries_report(tmp_path, delete_returncode):
    runner = _m1067_plan_limited_runner(delete_returncode=delete_returncode, delete_applies=False)
    config = make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop")

    with pytest.raises(AgentLoopError) as raised:
        _m1067_implicit_activation(runner, config)

    text = str(raised.value)
    _m1067_assert_text(text)
    assert text.count("Before this run PR #7 was draft/labeled") == 1
    if delete_returncode:
        assert "label DELETE was not confirmed" in text
        assert "it is now draft/labeled" in text
        assert "Do not ready or unlabel the PR." in text
    else:
        assert "did NOT qualify" in text


def test_m1067_printed_inspection_query_tolerates_malformed_payloads(tmp_path):
    jq = shutil.which("jq")
    if jq is None:
        pytest.skip("jq is not installed")
    query = shlex.split(managed_ci._pr_inspection_command(_m1067_config(tmp_path), 7))[-1]

    def evaluate(payload):
        result = subprocess.run(
            [jq, "-c", query], input=json.dumps(payload), capture_output=True, text=True, check=True,
        )
        return json.loads(result.stdout)

    assert evaluate({"head": {"repo": None}, "labels": [1, {"name": "x"}]})["labels"] == [
        {"malformed": 1}, "x",
    ]
    assert evaluate({"head": {"repo": None}, "labels": []})["head_repo"] is None
    for labels in ({}, None):
        assert evaluate({"labels": labels})["labels"] == {"malformed_container": labels}
    assert evaluate({})["labels"] == {"malformed_container": None}


def test_m1067_recovery_incapable_fallback_never_readies_after_concurrent_unlabel(tmp_path):
    # Another actor removes the label before the report read; the skipped
    # release still vetoes any automatic readiness write (review item-1).
    runner = _m1067_ready_runner(
        workflow=SUPPRESSING_V2_WORKFLOW_WITHOUT_RECOVERY,
        unreadable_issue_events_after_label=True,
        hooks={"post": lambda r: r.rest_pr.__setitem__("labels", [])},
    )

    contract = _m1067_activate(runner, _m1067_config(tmp_path))

    assert contract is not None and contract.activation_path == "ordinary_fallback"
    assert contract.ordinary_recovery is None
    report = contract.state_report
    _m1067_assert_text(report)
    assert "it is now draft/unlabeled" in report
    assert "base workflow has no unlabeled recovery route" in report
    assert "Do not ready or unlabel the PR." in report
    assert "gh pr ready 7" not in report
    assert _m1067_ready_calls(runner) == []
    assert _m1067_label_deletes(runner) == []


def test_m1067_nonzero_undo_with_pr_still_ready_gives_inspection_remedy(tmp_path):
    # Review item-3: an ambiguous ready-to-draft exit always gets remedy (C).
    runner = _m1067_ready_runner(undo_returncode=1, undo_applies=False)

    _error, text = _m1067_fail(runner, _m1067_config(tmp_path))

    assert "its ready-to-draft request was not confirmed" in text
    assert "No readiness restoration is needed: the PR is measured ready/unlabeled" in text
    assert "Do not ready or unlabel the PR. Inspect it with `gh api repos/OWNER/REPO/pulls/7" in text
    assert "Resume with `" not in text
    assert "gh pr ready 7" not in text
    assert _m1067_ready_calls(runner) == []


@pytest.mark.parametrize("managed", [False, True])
def test_m1067_recovery_incapable_fallback_helper_measures_concurrent_unlabel(tmp_path, capsys, managed):
    # Review item-7: label presence comes only from the measured read; the
    # helper states just that this run made no DELETE.
    runner = _m1067_plan_limited_runner(workflow=SUPPRESSING_V2_WORKFLOW_WITHOUT_RECOVERY)
    runner.rest_pr["labels"] = [{"name": MANAGED_LABEL}]
    config = make_config(
        tmp_path, managed_ci=managed, auto_merge=not managed,
        managed_ci_trusted_actor="agent-loop", quiet=False,
    )
    original = managed_ci._measured_state_line

    def unlabel_then_measure(runner_, config_, pr_number):
        runner_.rest_pr["labels"] = []  # another actor removed the label
        return original(runner_, config_, pr_number)

    import unittest.mock as mock
    with mock.patch.object(managed_ci, "_measured_state_line", unlabel_then_measure):
        if managed:
            with pytest.raises(AgentLoopError) as raised:
                _m1067_implicit_activation(runner, config)
            text = str(raised.value)
        else:
            assert _m1067_implicit_activation(runner, config) is None
            text = capsys.readouterr().err

    _m1067_assert_text(text)
    assert f"this run made no `{MANAGED_LABEL}` label DELETE" in text
    assert "PR #7 is now draft/unlabeled." in text
    assert "retained" not in text
    assert _m1067_label_deletes(runner) == []
    assert _m1067_ready_calls(runner) == []


def test_m1067_recovery_incapable_ready_labeled_entry_concurrent_unlabel_keeps_inspection(
    tmp_path, monkeypatch
):
    # Review item-11: a ready entry on a base without the unlabeled recovery
    # route gets remedy (C) even when the PR measures ready/unlabeled.
    runner = _m1067_draft_labeled_runner(
        workflow=SUPPRESSING_V2_WORKFLOW_WITHOUT_RECOVERY, events_unreadable=True,
        rest_pr={"draft": False},
    )
    original = managed_ci._read_failed_activation_state

    def unlabel_elsewhere(runner_, config_, pr_number):
        runner_.rest_pr["labels"] = []  # another actor removes the label
        return original(runner_, config_, pr_number)

    monkeypatch.setattr(managed_ci, "_read_failed_activation_state", unlabel_elsewhere)

    contract = _m1067_activate(runner, _m1067_config(tmp_path), lifecycle="draft-labeled")

    assert contract is not None and contract.activation_path == "ordinary_fallback"
    assert contract.ordinary_recovery is None
    report = contract.state_report
    assert report is not None
    _m1067_assert_text(report)
    assert "Before this run PR #7 was ready/labeled; it is now ready/unlabeled." in report
    assert "No readiness restoration is needed" in report
    assert "Do not ready or unlabel the PR." in report
    assert "Inspect it with `gh api repos/OWNER/REPO/pulls/7" in report
    assert "--managed-ci" in report
    assert "Resume with `" not in report
    assert "gh pr ready 7" not in report
    assert _m1067_label_deletes(runner) == [] and _m1067_ready_calls(runner) == []
    assert _m1067_label_posts(runner) == [] and _m1067_undo_calls(runner) == []


@pytest.mark.parametrize("concurrent", ["none", "readied", "unlabeled"])
def test_m1067_recovery_incapable_draft_labeled_resume_reports_measured_state(
    tmp_path, monkeypatch, capsys, concurrent
):
    # Review item-9: a draft/labeled resume on a base without the unlabeled
    # recovery route makes no DELETE, yet the fallback still carries a
    # measured report, which run_pr_loop prints instead of an unmeasured line.
    runner = _m1067_draft_labeled_runner(
        workflow=SUPPRESSING_V2_WORKFLOW_WITHOUT_RECOVERY, events_unreadable=True,
    )
    if concurrent != "none":
        original = managed_ci._read_failed_activation_state

        def change_elsewhere(runner_, config_, pr_number):
            if concurrent == "readied":
                runner_.rest_pr["draft"] = False  # another actor readies the PR
            else:
                runner_.rest_pr["labels"] = []  # another actor removes the label
            return original(runner_, config_, pr_number)

        monkeypatch.setattr(managed_ci, "_read_failed_activation_state", change_elsewhere)

    contract = _m1067_activate(runner, _m1067_config(tmp_path), lifecycle="draft-labeled")

    assert contract is not None and contract.activation_path == "ordinary_fallback"
    assert contract.ordinary_recovery is None
    report = contract.state_report
    assert report is not None
    _m1067_assert_text(report)
    assert f"This run made no `{MANAGED_LABEL}` label DELETE" in report
    if concurrent != "unlabeled":
        assert "draft/unlabeled re-entry state" not in report
    assert "Ordinary CI can resume" not in report
    assert "Do not ready or unlabel the PR." in report
    assert "gh pr ready 7" not in report
    # Review item-10: remedy (C) holds whatever the measured label state.
    assert "Inspect it with `gh api repos/OWNER/REPO/pulls/7" in report
    assert "--managed-ci" in report
    if concurrent == "unlabeled":
        assert "Before this run PR #7 was draft/labeled; it is now draft/unlabeled." in report
        assert "Resume with `" not in report
    elif concurrent == "readied":
        assert "Before this run PR #7 was draft/labeled; it is now ready/labeled." in report
        assert "this run made no readiness change" in report
        assert "claims no qualification" in report
        assert "remains draft" not in report
    else:
        assert "Before this run PR #7 was draft/labeled; it is now draft/labeled." in report
        assert "It remains draft/labeled and managed qualification was abandoned." in report
    assert _m1067_label_deletes(runner) == [] and _m1067_ready_calls(runner) == []
    assert _m1067_label_posts(runner) == []

    # The orchestrator prints this measured report, not the unmeasured line.
    monkeypatch.setattr(orchestrator, "activate_managed_ci", lambda *args, **kwargs: contract)
    loop_runner = FakeRunner()
    assert orchestrator.run_pr_loop(
        loop_runner, pr_number=7, config=make_config(tmp_path / "loop", auto_merge=True),
    ) == 0
    out = capsys.readouterr().out
    assert report in out
    assert "remains draft and unmerged" not in out
    assert not any(
        command[:3] == ["gh", "pr", "ready"] or "DELETE" in command
        for command, _cwd in loop_runner.commands
    )


def test_waiter_returns_timeout_not_passed_for_final_success_with_nonfinal_failure(tmp_path):
    """#1117: the real waiter never reports `passed` for this board; it times out with it."""
    config = make_config(
        tmp_path, auto_merge=True, ci_timeout_seconds=1, ci_poll_interval_seconds=1
    )
    runner = ManagedRunner(
        pr_payload={"headRefOid": "abc123", "mergeable": "MERGEABLE", "mergeStateStatus": "CLEAN"},
        pr_status_payload={"statuses": [{"context": FINAL_CONTEXT, "state": "success", "target_url": None}]},
        pr_check_runs_payload={
            "total_count": 1,
            "check_runs": [{"id": 3, "name": "lint", "status": "completed", "conclusion": "failure"}],
        },
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT], "checks": []},
    )

    outcome = wait_for_final_qualification(runner, config=config, pr_number=7, metadata=metadata())

    assert outcome.status == "timeout"
    assert outcome.head_sha == "abc123"
    assert [check.name for check in outcome.checks.failing] == ["lint"]


def test_draft_labeled_state_left_by_supersession_stop_is_accepted_by_real_resume_paths(tmp_path):
    """#1117: the side-effect-free draft stop leaves a lifecycle both resume entries accept."""
    from coding_review_agent_loop.managed_ci import authenticate_source_managed_resume

    runner = _recovery_runner([_unreadable_record()])
    config = _cloud_config(tmp_path, managed_ci_pr_mode=True)

    handoff = _recover(runner, config)
    assert handoff is not None
    assert _mutations(runner) == []

    source_runner = _recovery_runner([_unreadable_record()])
    resume = authenticate_source_managed_resume(
        source_runner, config=_cloud_config(tmp_path), pr_number=7,
        source_branch="feature", source_sha="abc123",
        managed_branch="agent-loop/managed-643", override_nonce="opening-nonce",
    )
    assert resume.lifecycle == "draft-labeled"
    assert _mutations(source_runner) == []


@pytest.mark.parametrize("origin", ["issue-created", "source-managed"])
@pytest.mark.parametrize("auto_merge", [True, False], ids=["auto-merge", "manual"])
def test_1117_draft_stop_then_real_resume_entry_paths_finalize_carried_obligation(
    tmp_path, monkeypatch, origin, auto_merge
):
    """#1117: stop a real draft-labeled managed PR carrying the persisted item, then resume it."""
    import dataclasses
    from test_orchestrator_pr import _carried_ci_obligations, _carried_ci_review_comment
    from coding_review_agent_loop.github import PullRequestMergeability
    from coding_review_agent_loop.managed_ci import ManagedCiOutcome

    item = dataclasses.replace(
        _carried_ci_obligations()[1], text="Failing checks: Python 3.12 full suite (failure)"
    )
    approval = structured_pr_review(
        reviewer="OpenAI Codex", state="approved",
        prior_item_dispositions=[{"item_id": item.item_id, "disposition": "resolved"}],
    )
    runner = _qualified_ready_labeled_runner(origin, codex_outputs=[approval, approval])
    runner.rest_pr["draft"] = True
    runner.pr_payload["comments"] = [
        {"author": {"login": "coding-review-agent-loop"}, "body": _carried_ci_review_comment((item,))}
    ]
    kwargs = {"auto_merge": True} if auto_merge else {}
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop", reviewer=("codex",),
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci", "--managed-ci-trusted-actor", "agent-loop",
        ),
        **kwargs,
    )
    monkeypatch.setattr(
        orchestrator, "_freeze_prompt_architecture", lambda _runner, config, **_kwargs: config,
    )

    def board(protection):
        runs = [
            PullRequestCheck(FINAL_CONTEXT, "check_run", "success"),
            PullRequestCheck("Python 3.12 full suite", "check_run", "skipped"),
        ]
        return PullRequestChecks(
            state="passing", required_checks=(), passing=tuple(runs), pending=(), failing=(),
            missing_required=(), branch_protection_status=protection, listing_complete=True,
        )

    state = {"protection": "forbidden"}
    monkeypatch.setattr(orchestrator, "dispatch_final_qualification", lambda *a, **k: None)
    monkeypatch.setattr(
        orchestrator, "wait_for_final_qualification",
        lambda *a, **k: ManagedCiOutcome(
            status="passed", checks=board(state["protection"]), head_sha="abc123"
        ),
    )
    monkeypatch.setattr(
        orchestrator, "_mergeability_for_unreadable_protection",
        lambda *a, **k: PullRequestMergeability(
            state="mergeable", mergeable_raw="MERGEABLE", merge_state_raw="DRAFT",
            head_sha="abc123", base_branch="main",
        ),
    )
    finalized = []
    monkeypatch.setattr(orchestrator, "prepare_v2_merge", lambda *a, **k: finalized.append("prepare"))
    monkeypatch.setattr(orchestrator, "_merge_with_exact_head_proof", lambda *a, **k: finalized.append("merge"))
    monkeypatch.setattr(
        orchestrator, "publish_manual_v2_qualification",
        lambda *a, **k: finalized.append("publish") or "abc123",
    )
    real_activate = orchestrator.activate_managed_ci
    lifecycles = []

    def activate(*args, **kwargs):
        lifecycles.append(kwargs["managed_resume"].lifecycle)
        return real_activate(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "activate_managed_ci", activate)

    with pytest.raises(AgentLoopError, match="administration: read"):
        orchestrator.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    assert lifecycles == ["draft-labeled"]
    assert finalized == []
    assert _lifecycle_writes(runner) == []  # no label, ready, or merge write
    assert runner.rest_pr["draft"] is True
    assert runner.rest_pr["labels"] == [{"name": MANAGED_LABEL}]
    # The persisted round metadata still holds the carried item, unchanged.
    from types import SimpleNamespace
    from coding_review_agent_loop.round_state import _extract_round_metadata_records

    persisted = [
        i
        for record in _extract_round_metadata_records(
            [SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]], flow="pr"
        )
        for i in (*record.metadata.prior_items, *record.metadata.new_items)
        if i.obligation_kind == "github-pr-checks"
    ]
    assert persisted and {(i.text, i.lifecycle, i.failed_head_sha, i.candidate_head_sha) for i in persisted} == {
        (item.text, item.lifecycle, item.failed_head_sha, item.candidate_head_sha)
    }

    # Protection becomes readable: resume the SAME PR through the same real entry path.
    state["protection"] = "configured"
    assert orchestrator.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True) == 0
    assert lifecycles == ["draft-labeled", "draft-labeled"]
    assert finalized == (["prepare", "merge"] if auto_merge else ["publish"])


@pytest.mark.parametrize("target", ["issue", "pr"])
@pytest.mark.parametrize("managed_ci", [True, False])
def test_recovery_renderer_never_replays_plan_reset_stall_streak(tmp_path, target, managed_ci):
    """`config-and-inheritance` (#1112)."""
    config = make_config(
        tmp_path,
        invocation_argv=(
            "agent-loop", "issue", "643", "--plan-first",
            "--plan-reset-stall-streak", "--plan-primary-stall-rounds", "3",
            "--reviewer", "codex",
        ),
    )
    rendered = _render_recovery_command(
        config, target=target, identifier=7, managed_ci=managed_ci,
        preserve_managed_options=True,
    )
    tokens = shlex.split(rendered)
    assert "--plan-reset-stall-streak" not in tokens
    assert "--reviewer" in tokens
    if target == "issue":
        assert tokens[tokens.index("--plan-primary-stall-rounds") + 1] == "3"


class _CheckoutSetupReached(Exception):
    """Stop the resume right after the default checkout has been cloned."""


def test_run_pr_loop_resume_with_missing_default_checkout_clones_it(tmp_path, monkeypatch):
    record = ManagedCiIssueAuthorization(
        kind="creation", repository="OWNER/REPO", issue_number=643, pr_number=7,
        base_ref="main", head_sha="abc123", actor_login="agent-loop", actor_id=1,
        protection="voluntary", waiver="allow-unprotected-managed-ci",
        nonce="opening-nonce", label_event_id=101,
    )
    # FakeRunner raises FileNotFoundError for a missing cwd, like subprocess.run,
    # and creates the target directory for `gh repo clone`.
    runner = _workflow_runner_for_issue_authorization(record, labeled=True)
    default_checkout = tmp_path / "claude" / "repo"
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = make_config(
        tmp_path, create_dirs=False, managed_ci=True, managed_ci_pr_mode=True,
        claude_dir=default_checkout, codex_dir=codex_dir,
        auto_agent_dirs=("claude",), agent_memory=False,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci",
            "--managed-ci-trusted-actor", "agent-loop",
            "--allow-unprotected-managed-ci",
        ),
    )
    assert not default_checkout.exists()

    def stop_after_clone(*_args, **_kwargs):
        raise _CheckoutSetupReached

    monkeypatch.setattr("coding_review_agent_loop.config._sync_base_branch", stop_after_clone)

    with pytest.raises(_CheckoutSetupReached):
        orchestrator.run_pr_loop(runner, pr_number=7, config=config)

    pull_reads = [
        cwd for command, cwd in runner.commands
        if command[:3] == ["gh", "api", "repos/OWNER/REPO/pulls/7"]
    ]
    assert pull_reads
    # The opening read uses the bootstrap directory (an existing agent dir);
    # the recovery reads use the neutral directory.  None may name the
    # missing default checkout.
    assert all(cwd.is_dir() and cwd != default_checkout for cwd in pull_reads)
    assert any(cwd == github_api_cwd() for cwd in pull_reads)
    assert any(
        command[:4] == ["gh", "repo", "clone", "OWNER/REPO"]
        and command[4] == str(default_checkout)
        for command, _cwd in runner.commands
    )
    assert default_checkout.is_dir()
    # No GitHub write may precede checkout setup on an ordinary resume.
    assert not [
        command for command, _cwd in runner.commands
        if command[:2] == ["gh", "api"]
        and any(part in command for part in ("POST", "PATCH", "PUT", "DELETE"))
    ]
    assert runner.comments == []


def test_run_pr_loop_fresh_resume_with_missing_default_checkout_publishes_and_clones(
    tmp_path, monkeypatch,
):
    runner = _workflow_runner_for_issue_authorization(None, labeled=False)
    default_checkout = tmp_path / "claude" / "repo"
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = make_config(
        tmp_path, create_dirs=False, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_fresh_authorization=True, managed_ci_issue_number=643,
        claude_dir=default_checkout, codex_dir=codex_dir,
        auto_agent_dirs=("claude",), agent_memory=False,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        invocation_argv=(
            "agent-loop", "pr", "7", "--managed-ci", "--managed-ci-fresh",
            "--managed-ci-issue", "643", "--managed-ci-trusted-actor", "agent-loop",
            "--allow-unprotected-managed-ci",
        ),
    )

    def stop_after_clone(*_args, **_kwargs):
        raise _CheckoutSetupReached

    monkeypatch.setattr("coding_review_agent_loop.config._sync_base_branch", stop_after_clone)

    with pytest.raises(_CheckoutSetupReached):
        orchestrator.run_pr_loop(runner, pr_number=7, config=config)

    # A new grant was actually published (a write) before checkout setup,
    # from a directory that is not the missing checkout.
    writes = [
        cwd for command, cwd in runner.commands
        if command[:2] == ["gh", "api"] and "POST" in command
        and any("issues/7/comments" in part for part in command)
    ]
    assert writes
    assert all(cwd.is_dir() and cwd != default_checkout for cwd in writes)
    assert default_checkout.is_dir()


def test_api_json_runs_from_a_directory_that_is_not_an_agent_checkout(tmp_path):
    runner = FakeRunner()
    config = make_config(tmp_path)
    config.antigravity_dir.mkdir(parents=True, exist_ok=True)

    managed_ci._api_json(runner, config, "repos/OWNER/REPO/pulls/7", quiet=True)

    (command, cwd), = [
        item for item in runner.commands if item[0][:2] == ["gh", "api"]
    ]
    assert command[2] == "repos/OWNER/REPO/pulls/7"
    assert cwd.is_dir()
    assert cwd not in (
        config.claude_dir, config.codex_dir, config.gemini_dir, config.antigravity_dir
    )


# --- #510 stage 3: label add/remove and ready reconciliation ----------------------

import json as _json510
from types import SimpleNamespace as _NS510

import coding_review_agent_loop.github as _github510
from coding_review_agent_loop.github import reconciled_pr_ready as _reconciled_ready
from coding_review_agent_loop.managed_ci import (
    CompleteLabelHistory as _Complete,
    ManagedCiContract as _Contract510,
    UnknownLabelHistory as _Unknown,
    _label_add,
    _label_remove,
    label_event_history as _label_history,
    release_adopted_managed_ci as _release510,
)

QUALIFIED_LABEL = managed_ci.QUALIFIED_LABEL
_BOT = ("agent-bot", 11)
_STRANGER = ("someone", 22)


class _LabelHub:
    """Label timeline + draft state with scripted write outcomes.

    Script entries: ``ok``; ``fail`` (502, nothing applied); ``accepted``
    (applied, client sees 502); ``fail422``.
    """

    def __init__(self, script=(), *, label=MANAGED_LABEL):
        self.label = label
        self.script = list(script)
        self.events: list[dict] = []
        self.next_event = 1000
        self.posts = 0
        self.deletes = 0
        self.readies: list[list[str]] = []
        self.draft = True
        self.head = "a" * 40
        self.events_unreadable = False
        self.events_reads = 0
        self.ready_script: list[str] = []
        self.dry_run = False

    def _latest_kind(self):
        kinds = [e["event"] for e in self.events]
        return kinds[-1] if kinds else None

    def event(self, kind, actor=_BOT):
        self.next_event += 1
        record = {
            "id": self.next_event,
            "event": kind,
            "label": {"name": self.label},
            "actor": {"login": actor[0], "id": actor[1]},
        }
        self.events.append(record)
        return record

    @staticmethod
    def _res(rc, out="", err=""):
        return _NS510(returncode=rc, stdout=out, stderr=err, args=[], cwd=None)

    def terminate_active_processes(self):  # pragma: no cover
        pass

    def run(self, args, *, cwd, check=True, input_text=None, env=None):
        cmd = [str(a) for a in args]
        joined = " ".join(cmd)
        if cmd[1:3] == ["api", "user"]:
            return self._res(0, out=_json510.dumps({"login": _BOT[0], "id": _BOT[1]}))
        if "/events" in joined:
            self.events_reads += 1
            if self.events_unreadable:
                return self._res(1, err="HTTP 404: Not Found")
            return self._res(0, out=_json510.dumps(self.events))
        if "--method" in cmd and "POST" in cmd and joined.endswith(f"labels[]={self.label}"):
            self.posts += 1
            action = self.script.pop(0) if self.script else "ok"
            if action == "fail":
                return self._res(1, err="non-200 OK status code: 502 Bad Gateway")
            if action == "fail422":
                return self._res(1, err="HTTP 422: Validation Failed")
            self.event("labeled")
            if action == "accepted":
                return self._res(1, err="non-200 OK status code: 502 Bad Gateway")
            return self._res(0)
        if "--method" in cmd and "DELETE" in cmd:
            self.deletes += 1
            action = self.script.pop(0) if self.script else "ok"
            if action == "fail":
                return self._res(1, err="HTTP 503 Service Unavailable")
            self.event("unlabeled")
            if action == "accepted":
                return self._res(1, err="HTTP 503 Service Unavailable")
            return self._res(0)
        if cmd[1:3] == ["pr", "ready"]:
            self.readies.append(cmd)
            action = self.ready_script.pop(0) if self.ready_script else "ok"
            undo = "--undo" in cmd
            if action == "fail":
                return self._res(1, err="HTTP 503 Service Unavailable")
            self.draft = undo
            if action == "accepted":
                return self._res(1, err="HTTP 503 Service Unavailable")
            return self._res(0)
        if joined.endswith("repos/OWNER/REPO/pulls/7"):
            present = self._latest_kind() == "labeled"
            payload = {"labels": [{"name": self.label}] if present else []}
            return self._res(0, out=_json510.dumps(payload))
        if cmd[1:3] == ["pr", "view"] and "isDraft,headRefOid" in cmd:
            return self._res(0, out=_json510.dumps({"isDraft": self.draft, "headRefOid": self.head}))
        return self._res(0, out="")


@pytest.fixture
def label_env(monkeypatch):
    monkeypatch.setattr(_github510, "log", lambda _c, _m: None)
    monkeypatch.setattr(managed_ci, "log", lambda _c, _m: None)
    monkeypatch.setattr(_github510, "active_workdir", lambda config: None)
    monkeypatch.setattr(managed_ci, "active_workdir", lambda config: None)


def _lcfg(tmp_path):
    return make_config(tmp_path)


@pytest.mark.parametrize("label", [MANAGED_LABEL, QUALIFIED_LABEL])
def test_label_add_accepted_but_failed_is_success_with_the_true_event_id(tmp_path, label_env, label):
    hub = _LabelHub(["accepted"], label=label)
    outcome = managed_ci.reconciled_label_write(
        hub, config=_lcfg(tmp_path), pr_number=7, label_name=label, op="add",
        write=lambda: hub.run(
            ["gh", "api", "--method", "POST", "repos/OWNER/REPO/issues/7/labels", "-f", f"labels[]={label}"],
            cwd=None, check=False,
        ),
    )
    assert outcome.result.returncode == 0
    assert outcome.event_id == hub.events[-1]["id"] and outcome.outcome == "adopted"
    assert hub.posts == 1


def test_label_add_transient_not_applied_replays_once_then_succeeds(tmp_path, label_env):
    hub = _LabelHub(["fail"])
    result = _label_add(hub, config=_lcfg(tmp_path), pr_number=7, label_name=MANAGED_LABEL).result
    assert result.returncode == 0 and hub.posts == 2


def test_label_add_foreign_newer_event_fails_closed_without_replay(tmp_path, label_env):
    hub = _LabelHub(["fail"])

    original_run = hub.run

    def run(args, **kw):
        result = original_run(args, **kw)
        if hub.posts == 1 and result.returncode != 0:
            hub.event("labeled", actor=_STRANGER)  # another actor labels meanwhile
        return result

    hub.run = run
    outcome = _label_add(hub, config=_lcfg(tmp_path), pr_number=7, label_name=MANAGED_LABEL)
    assert outcome.result.returncode != 0 and outcome.outcome == "foreign"
    assert hub.posts == 1


def test_label_add_newer_unlabeled_event_fails_closed(tmp_path, label_env):
    hub = _LabelHub(["fail"])
    original_run = hub.run

    def run(args, **kw):
        result = original_run(args, **kw)
        if hub.posts == 1 and result.returncode != 0:
            hub.event("unlabeled", actor=_BOT)
        return result

    hub.run = run
    outcome = _label_add(hub, config=_lcfg(tmp_path), pr_number=7, label_name=MANAGED_LABEL)
    assert outcome.result.returncode != 0 and hub.posts == 1


def test_label_add_unreadable_history_after_failure_is_not_replayed(tmp_path, label_env):
    hub = _LabelHub(["fail"])
    original_run = hub.run

    def run(args, **kw):
        result = original_run(args, **kw)
        if hub.posts == 1 and result.returncode != 0:
            hub.events_unreadable = True
        return result

    hub.run = run
    outcome = _label_add(hub, config=_lcfg(tmp_path), pr_number=7, label_name=MANAGED_LABEL)
    assert outcome.result.returncode != 0 and outcome.outcome == "unknown"
    assert hub.posts == 1


def test_label_add_without_a_readable_baseline_is_a_single_attempt(tmp_path, label_env):
    hub = _LabelHub(["fail"])
    hub.events_unreadable = True
    outcome = _label_add(hub, config=_lcfg(tmp_path), pr_number=7, label_name=MANAGED_LABEL)
    assert outcome.result.returncode != 0 and hub.posts == 1


def test_label_add_permanent_failure_is_not_retried(tmp_path, label_env):
    hub = _LabelHub(["fail422"])
    outcome = _label_add(hub, config=_lcfg(tmp_path), pr_number=7, label_name=MANAGED_LABEL)
    assert outcome.result.returncode != 0 and hub.posts == 1


def test_label_history_distinguishes_unknown_from_empty(tmp_path, label_env):
    hub = _LabelHub()
    assert _label_history(hub, config=_lcfg(tmp_path), pr_number=7, label_name=MANAGED_LABEL) == _Complete(None)
    hub.events_unreadable = True
    assert isinstance(_label_history(hub, config=_lcfg(tmp_path), pr_number=7, label_name=MANAGED_LABEL), _Unknown)


def _owned_contract(event_id):
    return _Contract510(
        protocol_version=2, adopted_existing_pr=True, invocation_applied_label=True,
        active_label_event_id=event_id,
    )


def test_label_remove_503_still_present_with_owned_event_replays(tmp_path, label_env):
    hub = _LabelHub(["fail"])
    owned = hub.event("labeled")["id"]
    assert _release510(hub, config=_lcfg(tmp_path), pr_number=7, contract=_owned_contract(owned)) is True
    assert hub.deletes == 2


def test_label_remove_accepted_by_actor_is_success_without_replay(tmp_path, label_env):
    hub = _LabelHub(["accepted"])
    owned = hub.event("labeled")["id"]
    assert _release510(hub, config=_lcfg(tmp_path), pr_number=7, contract=_owned_contract(owned)) is True
    assert hub.deletes == 1


@pytest.mark.parametrize("replacement_actor", [_BOT, _STRANGER])
def test_label_remove_replacement_event_is_never_deleted(tmp_path, label_env, replacement_actor):
    hub = _LabelHub(["fail"])
    owned = hub.event("labeled")["id"]
    original_run = hub.run

    def run(args, **kw):
        result = original_run(args, **kw)
        if hub.deletes == 1 and result.returncode != 0:
            hub.event("labeled", actor=replacement_actor)
        return result

    hub.run = run
    # The ownership pre-check passes on the owned event, then the DELETE 503s and
    # a replacement application appears: the remove is refused, never replayed.
    assert _release510(hub, config=_lcfg(tmp_path), pr_number=7, contract=_owned_contract(owned)) is False
    assert hub.deletes == 1


def test_label_remove_unreadable_history_is_not_replayed(tmp_path, label_env):
    hub = _LabelHub(["fail"])
    owned = hub.event("labeled")["id"]
    original_run = hub.run

    def run(args, **kw):
        result = original_run(args, **kw)
        if hub.deletes == 1 and result.returncode != 0:
            hub.events_unreadable = True
        return result

    hub.run = run
    assert _release510(hub, config=_lcfg(tmp_path), pr_number=7, contract=_owned_contract(owned)) is False
    assert hub.deletes == 1


@pytest.mark.parametrize("undo", [False, True])
def test_ready_503_same_head_in_desired_state_is_success_without_replay(tmp_path, label_env, undo):
    hub = _LabelHub()
    hub.draft = not undo
    hub.ready_script = ["accepted"]
    result = _reconciled_ready(
        hub, config=_lcfg(tmp_path), pr_number=7, expected_head_sha=hub.head, undo=undo
    )
    assert result.returncode == 0 and len(hub.readies) == 1


def test_ready_503_with_head_changed_fails_closed_without_replay(tmp_path, label_env):
    hub = _LabelHub()
    hub.ready_script = ["accepted"]
    hub.head = "b" * 40  # pushed while the transition was ambiguous
    result = _reconciled_ready(
        hub, config=_lcfg(tmp_path), pr_number=7, expected_head_sha="a" * 40
    )
    assert result.returncode != 0 and len(hub.readies) == 1


def test_ready_503_not_applied_same_head_replays(tmp_path, label_env):
    hub = _LabelHub()
    hub.ready_script = ["fail"]
    result = _reconciled_ready(hub, config=_lcfg(tmp_path), pr_number=7, expected_head_sha=hub.head)
    assert result.returncode == 0 and len(hub.readies) == 2


def test_ready_without_expected_head_is_a_single_attempt(tmp_path, label_env):
    hub = _LabelHub()
    hub.ready_script = ["fail"]
    result = _reconciled_ready(hub, config=_lcfg(tmp_path), pr_number=7, expected_head_sha=None)
    assert result.returncode != 0 and len(hub.readies) == 1


def test_label_definition_create_503_that_landed_is_success(tmp_path, label_env):
    class Definitions(_LabelHub):
        def __init__(self):
            super().__init__()
            self.defined = False
            self.creates = 0

        def run(self, args, *, cwd, check=True, **kw):
            cmd = [str(a) for a in args]
            if cmd[1:2] == ["api"] and cmd[-1] == f"repos/OWNER/REPO/labels/{MANAGED_LABEL}":
                if self.defined:
                    return self._res(0, out="{}")
                return self._res(1, err="HTTP 404: Not Found")
            if "--method" in cmd and cmd[-1:] and any(a == "repos/OWNER/REPO/labels" for a in cmd):
                self.creates += 1
                self.defined = True
                return self._res(1, err="HTTP 503 Service Unavailable")
            return super().run(args, cwd=cwd, check=check, **kw)

    hub = Definitions()
    assert managed_ci.ensure_managed_label(hub, config=_lcfg(tmp_path)) is True
    assert hub.creates == 1


# --- #510 review round 1: ownership baselines, strict history, diagnostics --------


def _hook_after_delete(hub, actor):
    original = hub.run

    def run(args, **kw):
        result = original(args, **kw)
        if hub.deletes == 1 and result.returncode != 0 and not getattr(hub, "_replaced", False):
            hub._replaced = True
            hub.event("labeled", actor=actor)  # a replacement application lands
        return result

    hub.run = run


@pytest.mark.parametrize("label", [MANAGED_LABEL, QUALIFIED_LABEL])
@pytest.mark.parametrize("replacement_actor", [_BOT, _STRANGER])
def test_remove_without_owned_id_never_deletes_a_replacement_application(
    tmp_path, label_env, label, replacement_actor
):
    hub = _LabelHub(["accepted"], label=label)
    hub.event("labeled")  # the application being removed (baseline)
    _hook_after_delete(hub, replacement_actor)

    outcome = _label_remove(hub, config=_lcfg(tmp_path), pr_number=7, label_name=label)

    assert outcome.result.returncode != 0 and outcome.outcome == "replacement"
    assert hub.deletes == 1  # the replacement was never deleted


def test_remove_without_owned_id_and_unreadable_baseline_is_a_single_attempt(tmp_path, label_env):
    hub = _LabelHub(["fail"])
    hub.event("labeled")
    hub.events_unreadable = True
    outcome = _label_remove(hub, config=_lcfg(tmp_path), pr_number=7, label_name=MANAGED_LABEL)
    assert outcome.result.returncode != 0 and hub.deletes == 1


def test_remove_without_owned_id_replays_the_same_application(tmp_path, label_env):
    hub = _LabelHub(["fail"])
    hub.event("labeled")
    outcome = _label_remove(hub, config=_lcfg(tmp_path), pr_number=7, label_name=MANAGED_LABEL)
    assert outcome.result.returncode == 0 and hub.deletes == 2


def test_remove_adoption_requires_a_newer_actor_unlabel(tmp_path, label_env):
    hub = _LabelHub(["accepted"])
    hub.event("labeled")
    outcome = _label_remove(hub, config=_lcfg(tmp_path), pr_number=7, label_name=MANAGED_LABEL)
    assert outcome.result.returncode == 0 and outcome.outcome == "adopted" and hub.deletes == 1


def test_malformed_replacement_event_never_authorizes_a_remove_replay(tmp_path, label_env):
    hub = _LabelHub(["fail"])
    owned = hub.event("labeled")["id"]
    original = hub.run

    def run(args, **kw):
        result = original(args, **kw)
        if hub.deletes == 1 and result.returncode != 0 and len(hub.events) == 1:
            # A replacement application whose label envelope is malformed.
            hub.events.append({"id": 5000, "event": "labeled", "label": None, "actor": {"login": "x", "id": 1}})
        return result

    hub.run = run
    outcome = _label_remove(
        hub, config=_lcfg(tmp_path), pr_number=7, label_name=MANAGED_LABEL, owned_event_id=owned
    )
    assert outcome.result.returncode != 0 and outcome.outcome == "unknown"
    assert hub.deletes == 1


@pytest.mark.parametrize(
    "bad",
    [
        {"id": 1001, "event": "labeled", "label": {"name": MANAGED_LABEL}},
        {"id": "x", "event": "labeled", "label": {"name": MANAGED_LABEL}, "actor": {"login": "a", "id": 1}},
        {"id": 9, "event": "unlabeled", "label": {}, "actor": {"login": "a", "id": 1}},
    ],
)
def test_label_history_rejects_malformed_transition_records(tmp_path, label_env, bad):
    hub = _LabelHub()
    hub.events.append(bad)
    assert isinstance(
        _label_history(hub, config=_lcfg(tmp_path), pr_number=7, label_name=MANAGED_LABEL), _Unknown
    )


def test_label_history_rejects_non_increasing_event_ids(tmp_path, label_env):
    hub = _LabelHub()
    hub.events.append({"id": 50, "event": "labeled", "label": {"name": MANAGED_LABEL}, "actor": {"login": "a", "id": 1}})
    hub.events.append({"id": 40, "event": "unlabeled", "label": {"name": MANAGED_LABEL}, "actor": {"login": "a", "id": 1}})
    assert isinstance(
        _label_history(hub, config=_lcfg(tmp_path), pr_number=7, label_name=MANAGED_LABEL), _Unknown
    )


def test_add_replay_requires_the_live_label_to_agree_with_history(tmp_path, label_env):
    hub = _LabelHub(["fail"])
    original = hub.run

    def run(args, **kw):
        if "repos/OWNER/REPO/pulls/7" in " ".join(str(a) for a in args):
            return hub._res(0, out=_json510.dumps({"labels": [{"name": MANAGED_LABEL}]}))
        return original(args, **kw)

    hub.run = run  # history says absent, the live PR says present
    outcome = _label_add(hub, config=_lcfg(tmp_path), pr_number=7, label_name=MANAGED_LABEL)
    assert outcome.result.returncode != 0 and outcome.outcome == "unknown" and hub.posts == 1


def test_label_failures_carry_final_diagnostic_and_attempt_history(tmp_path, label_env):
    hub = _LabelHub(["fail", "fail", "fail"])
    outcome = _label_add(hub, config=_lcfg(tmp_path), pr_number=7, label_name=MANAGED_LABEL)
    assert outcome.outcome == "exhausted" and hub.posts == 3
    assert "502 Bad Gateway" in managed_ci.failure_suffix(outcome.result)
    assert "attempt 3" in managed_ci.failure_suffix(outcome.result)


def test_ready_exhaustion_carries_attempt_history(tmp_path, label_env):
    hub = _LabelHub()
    hub.ready_script = ["fail", "fail", "fail"]
    result = _reconciled_ready(hub, config=_lcfg(tmp_path), pr_number=7, expected_head_sha=hub.head)
    assert result.returncode != 0 and len(hub.readies) == 3
    assert "503 Service Unavailable" in managed_ci.failure_suffix(result)
    assert "attempt 3" in managed_ci.failure_suffix(result)


def test_ready_unreadable_state_keeps_the_final_diagnostic(tmp_path, label_env):
    hub = _LabelHub()
    hub.ready_script = ["fail"]
    original = hub.run

    def run(args, **kw):
        if list(map(str, args))[1:3] == ["pr", "view"]:
            return hub._res(1, err="HTTP 404: Not Found")
        return original(args, **kw)

    hub.run = run
    result = _reconciled_ready(hub, config=_lcfg(tmp_path), pr_number=7, expected_head_sha=hub.head)
    assert result.returncode != 0 and len(hub.readies) == 1
    assert "503 Service Unavailable" in managed_ci.failure_suffix(result)


def test_publish_manual_readiness_refusal_carries_the_ready_diagnostic(tmp_path):
    runner = PublicationRunner(ready_returncode=1)  # the fake prints "ready failed"

    with pytest.raises(AgentLoopError, match="Unable to mark qualified PR #7 ready") as raised:
        _publish(runner, tmp_path)

    assert "ready failed" in str(raised.value)


class _ReplacedAdoptionRunner(V2ManagedRunner):
    """The label POST lands but reports 502; a same-actor replacement follows."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.event_reads = 0
        self.deletes = 0

    def _run_locked(self, args, *, cwd, check, input_text=None):
        cmd = [str(a) for a in args]
        if "POST" in cmd and "repos/OWNER/REPO/issues/7/labels" in cmd:
            super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)
            return CommandResult(cmd, cwd, "", "non-200 OK status code: 502 Bad Gateway", 1)
        if "DELETE" in cmd and any(a.endswith(f"/labels/{MANAGED_LABEL}") for a in cmd):
            self.deletes += 1
        result = super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)
        if cmd[-1].startswith("repos/OWNER/REPO/issues/7/events?"):
            self.event_reads += 1
            if self.event_reads == 3:  # after the reconciliation read, before the ownership read
                base = self._next_event_id()
                self.issue_events.append(label_event(base, event="unlabeled"))
                self.issue_events.append(label_event(base + 1))
        return result


def test_adoption_never_claims_a_replacement_of_the_recovered_label_event(tmp_path):
    config = make_config(
        tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop",
        managed_ci_adopt_existing_pr=True,
    )
    runner = _ReplacedAdoptionRunner(
        workflow=adoption_workflow(),
        rest_pr={"draft": False, "state": "open", "labels": []},
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT]},
    )

    contract = activate_managed_ci(runner, config=config, pr_number=7, metadata=metadata())

    assert contract is None  # fail closed: the observed application is not the recovered one
    assert runner.deletes == 0  # and the replacement application is left untouched


# --- #510 review round 2: recovered ownership through activation cleanup ----------


class _RecoveredCleanupRunner(M1067Runner):
    """A label add lands but reports 502; ownership reads then degrade."""

    def __init__(self, *, replacement_actor=("agent-loop", 1), **kwargs):
        super().__init__(**kwargs)
        self.post_seen = False
        self.post_events_reads = 0
        self.replacement_actor = replacement_actor

    def _run_locked(self, args, *, cwd, check, input_text=None):
        cmd = [str(a) for a in args]
        endpoint = next((p for p in cmd if p.startswith("repos/")), "")
        if endpoint.startswith("repos/OWNER/REPO/issues/7/events?") and self.post_seen:
            self.post_events_reads += 1
            record, _ = self._record_command(args, cwd)
            n = self.post_events_reads
            if n == 1:  # reconciliation read: the recovered application E
                return CommandResult(record, cwd, json.dumps([label_event(500)]), "", 0)
            if n == 2:  # the ownership read is unavailable
                return CommandResult(record, cwd, "", "events unavailable", 1)
            actor = self.replacement_actor
            events = [
                label_event(500),
                label_event(501, event="unlabeled"),
                label_event(502, login=actor[0], actor_id=actor[1]),
            ]
            return CommandResult(record, cwd, json.dumps(events), "", 0)
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


@pytest.mark.parametrize("replacement_actor", [("agent-loop", 1), ("someone", 9)])
def test_activation_cleanup_never_deletes_a_replacement_of_the_recovered_application(
    tmp_path, replacement_actor
):
    rest_pr = {"state": "open", "draft": False, "labels": [], "body": "Fixes #643"}
    runner = _RecoveredCleanupRunner(
        rest_pr=rest_pr, workflow=SUPPRESSING_V2_WORKFLOW, issue_events=[],
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT]},
        post_returncode=1, post_applies=True,
        hooks={"post": lambda r: setattr(r, "post_seen", True)},
        replacement_actor=replacement_actor,
    )

    _error, text = _m1067_fail(runner, _m1067_config(tmp_path))

    assert "ownership could not be re-established" in text
    assert _m1067_label_deletes(runner) == []  # the replacement application survives


def _adoption_config(tmp_path):
    return make_config(
        tmp_path, auto_merge=True, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        managed_ci_adopt_existing_pr=True,
    )


class _AdoptionLabelFailureRunner(V2ManagedRunner):
    def __init__(self, *, unreadable_after_post=False, **kwargs):
        super().__init__(**kwargs)
        self.unreadable_after_post = unreadable_after_post
        self.posted = False

    def _run_locked(self, args, *, cwd, check, input_text=None):
        cmd = [str(a) for a in args]
        if "POST" in cmd and "repos/OWNER/REPO/issues/7/labels" in cmd:
            self.posted = True
            record, _ = self._record_command(args, cwd)
            return CommandResult(record, cwd, "", "non-200 OK status code: 502 Bad Gateway", 1)
        if self.unreadable_after_post and self.posted and cmd[-1].startswith(
            "repos/OWNER/REPO/issues/7/events?"
        ):
            record, _ = self._record_command(args, cwd)
            return CommandResult(record, cwd, "", "HTTP 404: Not Found", 1)
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


@pytest.mark.parametrize("unreadable", [False, True], ids=["exhausted", "unreadable-reconciliation"])
def test_explicit_adoption_refusal_carries_the_label_write_diagnostic(tmp_path, unreadable):
    runner = _AdoptionLabelFailureRunner(
        workflow=adoption_workflow(),
        rest_pr={"draft": False, "state": "open", "labels": []},
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT]},
        unreadable_after_post=unreadable,
    )

    with pytest.raises(AgentLoopError, match="could not safely adopt") as raised:
        activate_managed_ci(
            runner, config=_adoption_config(tmp_path), pr_number=7, metadata=metadata(),
        )

    text = str(raised.value)
    assert "502 Bad Gateway" in text
    assert "attempt 1" in text
    if not unreadable:
        assert "attempt 3" in text


# --- #510 review round 3: event-kind validation and release diagnostics ----------


@pytest.mark.parametrize("label", [MANAGED_LABEL, QUALIFIED_LABEL])
@pytest.mark.parametrize("bad_kind", [None, 5, ""], ids=["missing", "non-string", "empty"])
def test_remove_never_replays_over_a_remove_reapply_pair_with_unusable_event_kinds(
    tmp_path, label_env, label, bad_kind
):
    hub = _LabelHub(["fail"], label=label)
    owned = hub.event("labeled")["id"]
    original = hub.run

    def run(args, **kw):
        result = original(args, **kw)
        if hub.deletes == 1 and result.returncode != 0 and len(hub.events) == 1:
            for event_id in (2001, 2002):  # remove then re-apply, kind unreadable
                envelope = {
                    "id": event_id, "label": {"name": label},
                    "actor": {"login": _BOT[0], "id": _BOT[1]},
                }
                if bad_kind is not None:
                    envelope["event"] = bad_kind
                hub.events.append(envelope)
        return result

    hub.run = run
    outcome = _label_remove(
        hub, config=_lcfg(tmp_path), pr_number=7, label_name=label, owned_event_id=owned
    )
    assert outcome.result.returncode != 0 and outcome.outcome == "unknown"
    assert hub.deletes == 1


def test_label_history_rejects_an_event_without_a_kind(tmp_path, label_env):
    hub = _LabelHub()
    hub.events.append({"id": 9, "label": {"name": MANAGED_LABEL}, "actor": {"login": "a", "id": 1}})
    assert isinstance(
        _label_history(hub, config=_lcfg(tmp_path), pr_number=7, label_name=MANAGED_LABEL), _Unknown
    )


@pytest.mark.parametrize("unreadable", [False, True], ids=["exhausted", "unreadable-reconciliation"])
def test_release_failure_carries_the_removal_diagnostic_and_attempt_history(
    tmp_path, label_env, unreadable
):
    hub = _LabelHub(["fail", "fail", "fail"])
    owned = hub.event("labeled")["id"]
    if unreadable:
        original = hub.run

        def run(args, **kw):
            result = original(args, **kw)
            if hub.deletes == 1 and result.returncode != 0:
                hub.events_unreadable = True
            return result

        hub.run = run
    contract = _owned_contract(owned)

    assert _release510(hub, config=_lcfg(tmp_path), pr_number=7, contract=contract) is False

    assert "503 Service Unavailable" in contract.release_diagnostic
    assert "attempt 1" in contract.release_diagnostic
    if not unreadable:
        assert "attempt 3" in contract.release_diagnostic


def test_successful_release_clears_the_diagnostic(tmp_path, label_env):
    hub = _LabelHub()
    owned = hub.event("labeled")["id"]
    contract = _owned_contract(owned)
    contract.release_diagnostic = "stale"
    assert _release510(hub, config=_lcfg(tmp_path), pr_number=7, contract=contract) is True
    assert contract.release_diagnostic == ""


# --- #510 review round 4: provisional cleanup diagnostics, ready through workflows --


class _ProvisionalCleanupFailureRunner(V2ManagedRunner):
    """The label add is acknowledged, provenance is unreadable and cleanup 503s."""

    def _run_locked(self, args, *, cwd, check, input_text=None):
        cmd = [str(a) for a in args]
        if "DELETE" in cmd and any(a.endswith(f"/labels/{MANAGED_LABEL}") for a in cmd):
            record, _ = self._record_command(args, cwd)
            return CommandResult(record, cwd, "", "HTTP 503 Service Unavailable", 1)
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


def test_provisional_adoption_cleanup_failure_carries_its_diagnostic(tmp_path):
    runner = _ProvisionalCleanupFailureRunner(
        workflow=adoption_workflow(),
        rest_pr={"draft": False, "state": "open", "labels": []},
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT]},
        unreadable_issue_events_after_label=True,
    )

    with pytest.raises(AgentLoopError, match="could not safely adopt") as raised:
        activate_managed_ci(
            runner, config=_adoption_config(tmp_path), pr_number=7, metadata=metadata(),
        )

    text = str(raised.value)
    assert "could not be released" in text
    assert "503 Service Unavailable" in text
    assert "attempt 1" in text


class _ReadyFlowHub(_LabelHub):
    """Qualified-label add plus draft/ready state for prepare_v2_merge."""

    def __init__(self, *args, drift_after_ready=False, **kwargs):
        super().__init__(*args, label=QUALIFIED_LABEL, **kwargs)
        self.drift_after_ready = drift_after_ready
        self.merges = 0

    def run(self, args, *, cwd, check=True, input_text=None, env=None):
        cmd = [str(a) for a in args]
        joined = " ".join(cmd)
        if cmd[1:3] == ["pr", "merge"]:
            self.merges += 1
            return self._res(0)
        if joined.endswith("repos/OWNER/REPO/pulls/7"):
            return self._res(0, out=_json510.dumps({
                "draft": self.draft,
                "labels": [{"name": QUALIFIED_LABEL}] if self._latest_kind() == "labeled" else [],
                "head": {"sha": self.head},
            }))
        if cmd[1:3] == ["pr", "view"] and "--jq" in cmd:
            return self._res(0, out=self.head + "\n")
        result = super().run(args, cwd=cwd, check=check, input_text=input_text, env=env)
        if cmd[1:3] == ["pr", "ready"] and self.drift_after_ready:
            self.head = "b" * 40  # a push lands while the transition is ambiguous
        return result


def test_prepare_v2_merge_ready_503_with_head_drift_refuses_with_one_ready_and_no_merge(
    tmp_path, label_env
):
    hub = _ReadyFlowHub(drift_after_ready=True)
    hub.ready_script = ["fail"]
    contract = managed_ci.ManagedCiContract(protocol_version=2)

    with pytest.raises(AgentLoopError, match="Unable to mark qualified PR #7 ready") as raised:
        managed_ci.prepare_v2_merge(
            hub, config=_lcfg(tmp_path), pr_number=7, expected_head_sha="a" * 40, contract=contract,
        )

    assert len(hub.readies) == 1  # never replayed against the drifted head
    assert hub.merges == 0
    assert "503 Service Unavailable" in str(raised.value)


def test_prepare_v2_merge_same_head_accepted_ready_reaches_post_write_verification(
    tmp_path, label_env
):
    hub = _ReadyFlowHub()
    hub.ready_script = ["accepted"]
    contract = managed_ci.ManagedCiContract(protocol_version=2)

    managed_ci.prepare_v2_merge(
        hub, config=_lcfg(tmp_path), pr_number=7, expected_head_sha="a" * 40, contract=contract,
    )

    assert len(hub.readies) == 1 and hub.draft is False
    assert hub.merges == 0  # prepare only readies; the head was re-verified afterwards


def test_prepare_v2_merge_post_write_verification_still_refuses_a_later_head_change(
    tmp_path, label_env
):
    hub = _ReadyFlowHub()
    hub.ready_script = ["ok"]
    original = hub.run

    def run(args, **kw):
        result = original(args, **kw)
        if [str(a) for a in args][1:3] == ["pr", "ready"]:
            hub.head = "c" * 40  # changes after a clean ready, so only the post-write check can catch it
        return result

    hub.run = run
    with pytest.raises(AgentLoopError, match="head changed while it was being readied"):
        managed_ci.prepare_v2_merge(
            hub, config=_lcfg(tmp_path), pr_number=7, expected_head_sha="a" * 40,
            contract=managed_ci.ManagedCiContract(protocol_version=2),
        )
