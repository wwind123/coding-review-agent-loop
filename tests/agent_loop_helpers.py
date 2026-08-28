"""Shared test helpers for the agent-loop test suite.

This module is NOT a test file — it contains FakeRunner, builder functions,
and utility helpers shared across test_agent_loop.py, test_protocol.py,
test_backends.py, and test_comment_rendering.py.
"""
import base64
import datetime
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import coding_review_agent_loop.orchestrator as orchestrator
from coding_review_agent_loop.agents.base import with_public_response_file_instruction
from coding_review_agent_loop.agents.claude import (
    BACKEND as CLAUDE_BACKEND,
    _normalize_claude_usage,
    _parse_claude_output,
)
from coding_review_agent_loop.agents.codex import (
    BACKEND as CODEX_BACKEND,
    _extract_codex_usage,
    _normalize_codex_usage,
)
from coding_review_agent_loop.agents.gemini import (
    BACKEND as GEMINI_BACKEND,
    PUBLIC_RESPONSE_MARKER,
    _normalize_gemini_usage,
    _parse_gemini_payload,
)
from coding_review_agent_loop.cli import (
    AgentLoopConfig,
    AgentLoopError,
    CommandResult,
    Runner,
    build_parser,
    config_from_args,
    ensure_log_dir_ignored,
    is_clarification_request,
    parse_agent_state,
    parse_pr_number,
    run_issue_loop,
    run_pr_loop,
    run_task_loop,
)
from coding_review_agent_loop.errors import QuotaResetExceededError, UnknownPriorItemDispositionError
from coding_review_agent_loop.orchestrator import (
    _decode_public_response_json_prefix,
    _format_reset_duration,
    _failure_category,
    _HumanRequirementsRecoveryContext,
    _is_transient_agent_output,
    _is_transient_public_response,
    _parse_rate_limit_reset_seconds,
    _recover_plan_revision_human_requirements_acknowledgement,
    _run_validated_agent,
    _split_reconstructable_plan_revision_response,
)
from coding_review_agent_loop.config import (
    default_agent_memory_dir,
    default_agent_workdir,
    default_cache_root,
    resolve_base_branch,
)
from coding_review_agent_loop.comment_rendering import (
    _render_public_coder_followup_comment,
    _render_public_plan_review_comment,
    _render_public_plan_revision_comment,
    _render_public_pr_review_comment,
    _render_public_issue_implementation_comment,
    normalize_freeform_signature,
)
from coding_review_agent_loop.decomposition import (
    CreatedPhaseIssue,
    MAX_DECOMPOSITION_PHASES,
    RecordedPhase,
    approved_plan_hash,
    find_existing_phase_implementation_handoff,
    format_decomposition_parent_summary,
    format_one_shot_impl_handoff_comment,
    format_phase_implementation_handoff_comment,
    parse_plan_decomposition,
)
from coding_review_agent_loop.github import (
    HumanReviewRequirement,
    IssueComment,
    IssueContext,
    PullRequestReviewContext,
    PullRequestMetadata,
    get_issue_context,
    get_pr_checks,
)
from coding_review_agent_loop.followups import (
    MAX_APPROVED_FOLLOWUP_ISSUES,
    reconcile_approved_followups,
)
from coding_review_agent_loop.memory import AgentMemoryContext
from coding_review_agent_loop.migrations import MigrationValidationResult, validate_pr_migration_topology
from coding_review_agent_loop.orchestrator import (
    ITEM_SUMMARY_LIMIT,
    HUMAN_REQUIREMENTS_ACK_ITEM_ID,
    PostedRoundMetadata,
    ValidatedAgentResponse,
    _apply_unresolved_item_dispositions,
    _attach_round_metadata,
    _collect_prior_compact_summaries,
    _decode_round_metadata,
    _encode_round_metadata,
    _format_unresolved_item_label,
    _plan_subject,
    _render_public_review_comment,
    _reconcile_human_requirements_ack_item,
    _review_freeform_summary_text,
    _resume_pr_round,
    _resume_plan_round,
    _strip_round_metadata,
    _validate_coder_followup_response,
    _validate_plan_revision_response,
    _validate_review_response,
    _validate_plan_review_response,
    render_public_agent_comment,
    render_canonical_plan_revision,
    render_canonical_plan_steps,
)
from coding_review_agent_loop.prompts import (
    COMPACT_PLANNING_VOLATILE_TAIL_MARKER,
    COMPACT_PR_REVIEW_VOLATILE_TAIL_MARKER,
    CompactPlanTailContext,
    CompactPrReviewTailContext,
    CompactPriorContext,
    HUMAN_REQUIREMENTS_ADDRESSED_MARKER,
    HUMAN_REQUIREMENTS_DIRECT_DISCUSSION_ACK,
    _build_followup_guidance,
    _build_unresolved_items_guidance,
    _phased_plan_guard,
    build_completion_recovery_prompt,
    build_followup_prompt,
    build_issue_implementation_prompt,
    build_issue_plan_prompt,
    build_issue_prompt,
    build_merge_conflict_prompt,
    build_task_prompt,
    build_same_pr_followup_prompt,
    build_plan_review_prompt,
    build_plan_revision_prompt,
    build_review_prompt,
    format_human_requirements,
    format_issue_context,
    render_coder_human_requirements_prompt_context,
)
from coding_review_agent_loop.protocol import (
    ApprovedFollowup,
    DiscussEvidenceClaim,
    _expect_string_list,
    _extract_structured_coder_followup_payload,
    _extract_structured_plan_review_payload,
    _extract_structured_plan_revision_payload,
    _extract_structured_pr_review_payload,
    normalize_response_file_structured_text,
    parse_approved_followups,
    parse_human_requirements_acknowledgement,
    parse_pr_review,
    parse_plan_item_dispositions,
    parse_plan_review,
    parse_plan_review_items,
    parse_plan_state,
    parse_structured_plan_review,
    parse_structured_pr_review,
    parse_review,
    parse_non_blocking_followups,
    parse_signed_human_requirement_body,
    parse_unresolved_item_dispositions,
    ReviewItemDisposition,
    UnresolvedReviewItem,
    validate_human_requirements_acknowledgement,
    validate_structured_coder_followup,
    validate_structured_issue_implementation,
    validate_structured_human_requirements_acknowledgement,
    validate_structured_plan_state,
    validate_structured_plan_revision,
)
from coding_review_agent_loop.workdir_guard import (
    extract_reported_tests_from_response,
    validate_checkout_inspected_evidence,
    validate_response_tests_within_workdir,
    validate_test_commands_within_workdir,
)

from unittest.mock import MagicMock, patch


class FakeRunner(Runner):
    def __init__(
        self,
        *,
        claude_outputs=None,
        codex_outputs=None,
        gemini_outputs=None,
        antigravity_outputs=None,
        antigravity_catalog_outputs=None,
        issue_payload=None,
        issue_comments=None,
        pr_payload=None,
        pr_check_runs_payload=None,
        pr_status_payload=None,
        pr_branch_protection_payload=None,
        pr_branch_protection_returncode=0,
        pr_branch_protection_stderr="",
        pr_enforce_admins_payload=None,
        pr_effective_rules_payload=None,
        pr_effective_rules_returncode=0,
        pr_effective_rules_stderr="",
        pr_rulesets_payload=None,
        repo_payload=None,
        pr_check_runs_returncode=0,
        pr_check_runs_stderr="",
        pr_status_returncode=0,
        pr_status_stderr="",
        repo_default_branch="main",
        repo_default_branch_returncode=0,
        git_status="",
        git_remote="git@github.com:OWNER/REPO.git",
        git_inside=True,
        git_head="abc123",
        tracked_files=None,
        changed_files=None,
        diff_returncode=0,
        diff_stderr="",
        git_diff="",
        git_diff_stat=None,
        git_diff_check="",
        git_diff_check_returncode=0,
        git_diff_check_stderr="",
        post_agent_git_status=None,
        post_agent_git_diff=None,
        post_agent_git_diff_stat=None,
        post_agent_git_diff_check=None,
        post_agent_git_diff_check_returncode=None,
        post_agent_git_diff_check_stderr=None,
        git_probe_results=None,
        git_probe_exceptions=None,
        issue_urls=None,
        public_response_outputs=None,
        advance_git_head_on_pr=True,
        advance_pr_head_on_coder_followup=True,
        search_issues_payload=None,
        open_prs_payload=None,
        mergeability_payloads=None,
        pr_commit_pages=None,
        pr_commit_metadata_payloads=None,
        pr_commit_query_failures=None,
    ):
        super().__init__(dry_run=False)
        self.claude_outputs = list(claude_outputs or [])
        self.codex_outputs = list(codex_outputs or [])
        self.gemini_outputs = list(gemini_outputs or [])
        self.antigravity_outputs = list(antigravity_outputs or [])
        self.antigravity_catalog_outputs = list(antigravity_catalog_outputs or [])
        self.issue_payload = {
            "number": 56,
            "state": "open",
            "is_pr": False,
            "url": "https://github.com/OWNER/REPO/issues/56",
            "title": "Fix issue-mode context",
            "body": "Original issue body.",
        }
        if issue_payload:
            self.issue_payload.update(issue_payload)
        self.issue_comments = list(issue_comments or [])
        self.pr_payload = {
            "number": 77,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/77",
            "title": "Improve review prompt context",
            "body": "PR description.",
            "headRefName": "feature/review-context",
            "baseRefName": "main",
            "headRefOid": "abc123",
            "comments": [],
            "reviews": [],
        }
        if pr_payload:
            self.pr_payload.update(pr_payload)
        self.pr_check_runs_payload = pr_check_runs_payload or {
            "check_runs": [{"name": "test", "status": "completed", "conclusion": "success"}]
        }
        self.pr_status_payload = pr_status_payload or {"state": "success", "statuses": []}
        self.pr_branch_protection_payload = pr_branch_protection_payload or {"contexts": ["test"]}
        self.pr_branch_protection_returncode = pr_branch_protection_returncode
        self.pr_branch_protection_stderr = pr_branch_protection_stderr
        self.pr_enforce_admins_payload = (
            {"enabled": True} if pr_enforce_admins_payload is None else pr_enforce_admins_payload
        )
        self.pr_effective_rules_payload = list(pr_effective_rules_payload or [])
        self.pr_effective_rules_returncode = pr_effective_rules_returncode
        self.pr_effective_rules_stderr = pr_effective_rules_stderr
        self.pr_rulesets_payload = dict(pr_rulesets_payload or {})
        self.repo_payload = {"default_branch": repo_default_branch, "private": False}
        if repo_payload:
            self.repo_payload.update(repo_payload)
        self.pr_check_runs_returncode = pr_check_runs_returncode
        self.pr_check_runs_stderr = pr_check_runs_stderr
        self.pr_status_returncode = pr_status_returncode
        self.pr_status_stderr = pr_status_stderr
        self.repo_default_branch = repo_default_branch
        self.repo_default_branch_returncode = repo_default_branch_returncode
        self.commands = []
        self.comments = []
        self.issues = []
        self.git_status = git_status
        self.git_remote = git_remote
        self.git_inside = git_inside
        self.git_head = git_head
        self.tracked_files = tracked_files or [
            "pyproject.toml",
            "README.md",
            "src/coding_review_agent_loop/cli.py",
            "tests/test_agent_loop.py",
        ]
        self.changed_files = changed_files or ["src/coding_review_agent_loop/cli.py"]
        self.diff_returncode = diff_returncode
        self.diff_stderr = diff_stderr
        self.git_diff = git_diff
        self.git_diff_stat = (
            git_diff_stat
            if git_diff_stat is not None
            else (" src/coding_review_agent_loop/cli.py | 1 +\n" if git_diff else "")
        )
        self.git_diff_check = git_diff_check
        self.git_diff_check_returncode = git_diff_check_returncode
        self.git_diff_check_stderr = git_diff_check_stderr
        self.post_agent_git_status = post_agent_git_status
        self.post_agent_git_diff = post_agent_git_diff
        self.post_agent_git_diff_stat = post_agent_git_diff_stat
        self.post_agent_git_diff_check = post_agent_git_diff_check
        self.post_agent_git_diff_check_returncode = post_agent_git_diff_check_returncode
        self.post_agent_git_diff_check_stderr = post_agent_git_diff_check_stderr
        self.git_probe_results = list(git_probe_results or [])
        self.git_probe_exceptions = list(git_probe_exceptions or [])
        self.git_probe_calls = []
        self.issue_urls = list(issue_urls) if issue_urls is not None else None
        self.public_response_outputs = list(public_response_outputs or [])
        self.advance_git_head_on_pr = advance_git_head_on_pr
        self.advance_pr_head_on_coder_followup = advance_pr_head_on_coder_followup
        self._coder_followup_counter = 0
        # Results returned by the next `gh issue list --search` call (#476).
        # A list of dicts with number/title/url/body keys; consumed one call
        # at a time when a list of lists is provided, otherwise reused as-is.
        self.search_issues_payload = search_issues_payload
        self.search_issues_calls = []
        # Results returned by `gh pr list --state open` calls, used to fake
        # `find_open_pr_closing_issue` (#495). A list of dicts with
        # number/body keys; defaults to no open PRs found.
        self.open_prs_payload = open_prs_payload if open_prs_payload is not None else []
        self.open_prs_calls = 0
        # Scripted responses for `get_pr_mergeability`'s dedicated `gh pr view
        # --json mergeable,mergeStateStatus,headRefOid,baseRefName` probe
        # (#606), consumed one call at a time. Each entry is a dict with any of
        # mergeable/mergeStateStatus/headRefOid/baseRefName; missing keys are
        # None. When exhausted (or unset), falls back to those same keys read
        # from `pr_payload`, which is None/None by default -- unknown, no-op
        # for tests that never set mergeability -- so existing suites are
        # unaffected.
        self.mergeability_payloads = (
            list(mergeability_payloads) if mergeability_payloads is not None else None
        )
        self.mergeability_calls = 0
        self.pr_commit_pages = list(
            pr_commit_pages if pr_commit_pages is not None else pr_commit_metadata_payloads or []
        )
        self.pr_commit_query_failures = list(pr_commit_query_failures or [])
        self.pr_commit_calls = 0
        self._provenance_issue_number = self.issue_payload.get("number", 56)
        self._agent_pr_counter = 0
        self._agent_command_seen = False
        # Parallel discuss debaters (#475) call run_with_log from worker
        # threads; scripted outputs stay keyed per agent, but the shared
        # bookkeeping (commands, comments, response files) needs a lock.
        self._scripted_lock = threading.RLock()

    def _normalize_legacy_agent_output(self, output: str, prompt: str) -> str:
        stripped = output.lstrip()
        if stripped.startswith("{"):
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                for key in ("response", "result"):
                    value = payload.get(key)
                    if isinstance(value, str):
                        normalized_value = self._normalize_legacy_agent_output(value, prompt)
                        if normalized_value != value:
                            payload[key] = normalized_value
                            return json.dumps(payload)
            return output
        signature_matches = re.findall(r"-- (OpenAI Codex|Google Gemini|Anthropic Claude)", prompt)
        signature = signature_matches[-1] if signature_matches else "OpenAI Codex"
        if (
            '"kind": "pr_review"' in prompt
            and '"kind": "coder_followup"' not in prompt
            and "<!-- AGENT_STATE:" in output
        ):
            parsed = parse_review(output, reviewer="OpenAI Codex")
            return structured_pr_review(
                state=parsed.state,
                summary=parsed.summary or "Review complete.",
                blocking_items=[item.text for item in parsed.blocking_items],
                same_pr_followups=[item.text for item in parsed.followups.same_pr],
                future_followups=[item.text for item in parsed.followups.future],
                prior_item_dispositions=[
                    {
                        key: value
                        for key, value in {
                            "item_id": item.item_id,
                            "disposition": item.disposition,
                            "note": item.note,
                        }.items()
                        if value is not None
                    }
                    for item in parsed.dispositions
                ],
                reviewer=signature,
                human_requirements_resolved="<!-- HUMAN_REQUIREMENTS_RESOLVED -->" in output,
            )
        if (
            '"kind": "plan_review"' in prompt
            and '"kind": "plan_revision"' not in prompt
            and "<!-- AGENT_PLAN_STATE:" in output
        ):
            state = parse_plan_state(output)
            items = parse_plan_review_items(output, reviewer="OpenAI Codex")
            future = [item.text for item in items.future] if state == "approved" else []
            blocking = [item.text for item in items.blocking]
            return structured_plan_review(
                state=state,
                summary=_review_freeform_summary_text(output) or "Plan review complete.",
                blocking_plan_issues=blocking,
                same_plan_followups=[item.text for item in items.same_plan],
                future_followups=future,
                prior_plan_item_dispositions=[
                    {
                        key: value
                        for key, value in {
                            "item_id": item.item_id,
                            "disposition": item.disposition,
                            "note": item.note,
                        }.items()
                        if value is not None
                    }
                    for item in parse_plan_item_dispositions(output, reviewer="OpenAI Codex")
                ],
                reviewer=signature,
                human_requirements_resolved="<!-- HUMAN_REQUIREMENTS_RESOLVED -->" in output,
            )
        if '"kind": "plan_revision"' in prompt and "<!-- AGENT_PLAN_STATE:" in output:
            return structured_plan_revision(
                summary=_review_freeform_summary_text(output) or "Revised the plan.",
                plan_steps=[
                    line.strip("- ").strip()
                    for line in output.splitlines()
                    if line.strip().startswith("- ")
                    and "Requirement " not in line
                    and not line.strip().startswith("--")
                ]
                or [_review_freeform_summary_text(output) or "Revised the plan."],
                human_requirements=(
                    "\n" + HUMAN_REQUIREMENTS_ADDRESSED_MARKER
                    if HUMAN_REQUIREMENTS_ADDRESSED_MARKER in output
                    else ""
                ),
                )
        if '"kind": "issue_implementation"' in prompt and (
            "<!-- AGENT_STATE:" in output or parse_pr_number(output) is not None
        ):
            pr_number = parse_pr_number(output)
            summary = _review_freeform_summary_text(output) or "Implementation completed."
            labels = sorted(
                set(re.findall(r"\bRequirement\s+\d+\b", output, re.I)),
                key=lambda value: int(value.split()[-1]),
            )
            labels = [f"Requirement {value.split()[-1]}" for value in labels]
            blocked = {
                label
                for label in labels
                if re.search(
                    rf"{re.escape(label)}.{{0,160}}(?:blocked|cannot|unavailable|impossible)",
                    output,
                    re.I | re.S,
                )
            }
            dispositions = [
                {
                    "requirement_id": label,
                    "disposition": "blocked" if label in blocked else "addressed",
                    "evidence": (
                        f"{label} is blocked according to the legacy implementation response."
                        if label in blocked
                        else f"{label} is covered by the legacy implementation response."
                    ),
                }
                for label in labels
            ]
            return structured_issue_implementation(
                state="blocking",
                summary=summary,
                pr_number=pr_number,
                human_requirement_dispositions=dispositions,
                human_requirement_ids=[label for label in labels if label not in blocked],
                checked_discussion_directly=HUMAN_REQUIREMENTS_DIRECT_DISCUSSION_ACK in output,
                tests_run=list(extract_reported_tests_from_response(output)) or None,
                reviewer=signature,
            )
        if '"kind": "coder_followup"' in prompt and "<!-- AGENT_STATE:" in output:
            item_ids = sorted(set(re.findall(r"\[(item-[A-Za-z0-9._-]+)\]", prompt)))
            item_ids = [item_id for item_id in item_ids if item_id != HUMAN_REQUIREMENTS_ACK_ITEM_ID]
            human_ids = sorted(set(re.findall(r"`(Requirement \d+)`|(?:^|\s)(Requirement \d+):", output)))
            flattened_human_ids = [first or second for first, second in human_ids]
            return structured_coder_followup(
                state=parse_agent_state(output),
                summary=_review_freeform_summary_text(output) or "Updated the PR.",
                addressed_items=item_ids,
                remaining_items=[],
                human_requirement_ids=flattened_human_ids,
            )
        return output

    def _next_agent_output(self, outputs):
        output = outputs.pop(0)
        if isinstance(output, dict):
            return output
        if isinstance(output, tuple):
            return output
        return output, 0

    def _record_command(self, args, cwd):
        cmd = [str(arg) for arg in args]
        cwd_path = Path(cwd)
        if not cwd_path.is_dir():
            raise FileNotFoundError(cwd_path)
        self.commands.append((cmd, cwd_path))
        return cmd, cwd_path

    def _maybe_write_public_response_file(self, cmd, *, prompt=None):
        if not self.public_response_outputs:
            return
        prompt = prompt or "\n".join(cmd)
        match = re.search(r"Write the final public response.*?\n\n([^\n]+/responses/[^\n]+\.md)", prompt, re.S)
        if not match:
            return
        response_path = Path(match.group(1))
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response = self.public_response_outputs.pop(0)
        if isinstance(response, dict):
            response = response.get("text", "")
        elif isinstance(response, str):
            response = self._normalize_legacy_agent_output(response, prompt)
        response_path.write_text(response, encoding="utf-8")

    def _maybe_advance_git_head_for_agent_pr(self, text: str) -> None:
        if not self.advance_git_head_on_pr:
            return
        has_pr_identity = "<!-- AGENT_PR:" in text or "github.com/OWNER/REPO/pull/" in text
        if not has_pr_identity and text.lstrip().startswith("{"):
            try:
                payload, _ = json.JSONDecoder().raw_decode(text.lstrip())
            except json.JSONDecodeError:
                payload = None
            has_pr_identity = (
                isinstance(payload, dict)
                and payload.get("kind") == "issue_implementation"
                and isinstance(payload.get("pr_number"), int)
                and not isinstance(payload.get("pr_number"), bool)
                and payload["pr_number"] > 0
            )
            if not has_pr_identity and isinstance(payload, dict):
                for key in ("response", "result"):
                    value = payload.get(key)
                    if isinstance(value, str):
                        has_pr_identity = (
                            "<!-- AGENT_PR:" in value
                            or "github.com/OWNER/REPO/pull/" in value
                        )
                        if not has_pr_identity and value.lstrip().startswith("{"):
                            try:
                                nested, _ = json.JSONDecoder().raw_decode(value.lstrip())
                            except json.JSONDecodeError:
                                nested = None
                            has_pr_identity = (
                                isinstance(nested, dict)
                                and nested.get("kind") == "issue_implementation"
                                and isinstance(nested.get("pr_number"), int)
                                and not isinstance(nested.get("pr_number"), bool)
                                and nested["pr_number"] > 0
                            )
                        if has_pr_identity:
                            break
        if not has_pr_identity:
            return
        self._agent_pr_counter += 1
        self.git_head = f"{self.git_head}-agent-{self._agent_pr_counter}"

    def _maybe_advance_pr_head_for_coder_followup(self, cmd) -> None:
        if not self.advance_pr_head_on_coder_followup:
            return
        if '"kind": "coder_followup"' not in "\n".join(cmd):
            return
        self._coder_followup_counter += 1
        head_sha = self.pr_payload.get("headRefOid", self.git_head)
        new_head_sha = f"{head_sha}-coder-{self._coder_followup_counter}"
        self.pr_payload["headRefOid"] = new_head_sha
        self.git_head = new_head_sha

    def _mark_agent_command_seen(self) -> None:
        self._agent_command_seen = True

    def _current_git_status(self) -> str:
        if self._agent_command_seen and self.post_agent_git_status is not None:
            return self.post_agent_git_status
        return self.git_status

    def _current_git_diff(self) -> str:
        if self._agent_command_seen and self.post_agent_git_diff is not None:
            return self.post_agent_git_diff
        return self.git_diff

    def _current_git_diff_stat(self) -> str:
        if self._agent_command_seen and self.post_agent_git_diff_stat is not None:
            return self.post_agent_git_diff_stat
        return self.git_diff_stat

    def _current_git_diff_check(self) -> str:
        if self._agent_command_seen and self.post_agent_git_diff_check is not None:
            return self.post_agent_git_diff_check
        return self.git_diff_check

    def _current_git_diff_check_returncode(self) -> int:
        if (
            self._agent_command_seen
            and self.post_agent_git_diff_check_returncode is not None
        ):
            return self.post_agent_git_diff_check_returncode
        return self.git_diff_check_returncode

    def _current_git_diff_check_stderr(self) -> str:
        if (
            self._agent_command_seen
            and self.post_agent_git_diff_check_stderr is not None
        ):
            return self.post_agent_git_diff_check_stderr
        return self.git_diff_check_stderr

    def run_with_log(
        self,
        args,
        *,
        cwd,
        log_path,
        label,
        progress_interval_seconds,
        check=True,
        env=None,
        input_text=None,
        use_pty=False,
        timeout_seconds=None,
    ):
        with self._scripted_lock:
            self.last_input_text = input_text
            return self._run_with_log_locked(
                args,
                cwd=cwd,
                log_path=log_path,
                check=check,
                input_text=input_text,
            )

    def _run_with_log_locked(self, args, *, cwd, log_path, check, input_text=None):
        cmd, cwd_path = self._record_command(args, cwd)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ensure_log_dir_ignored(log_path.parent)

        if cmd[:1] == ["claude"]:
            output, returncode = self._next_agent_output(self.claude_outputs)
            if isinstance(output, str):
                output = self._normalize_legacy_agent_output(output, input_text or "\n".join(cmd))
            self._maybe_write_public_response_file(cmd, prompt=input_text)
            self._maybe_advance_git_head_for_agent_pr(output)
            self._maybe_advance_pr_head_for_coder_followup(cmd)
            self._mark_agent_command_seen()
            log_path.write_text(f"$ {' '.join(cmd)}\n\n{output}", encoding="utf-8")
            return CommandResult(cmd, cwd_path, output, "", returncode)

        if cmd[:2] == ["codex", "exec"]:
            output = self._next_agent_output(self.codex_outputs)
            if isinstance(output, dict):
                public_response = output.get("public_response", "")
                stdout = output.get("stdout", "")
                returncode = output.get("returncode", 0)
            else:
                public_response, returncode = output
                stdout = public_response
            if isinstance(public_response, str):
                normalized = self._normalize_legacy_agent_output(public_response, "\n".join(cmd))
                if normalized != public_response:
                    public_response = normalized
            self._maybe_write_public_response_file(cmd, prompt=input_text)
            if "--output-last-message" in cmd:
                out_path = Path(cmd[cmd.index("--output-last-message") + 1])
                out_path.write_text(public_response, encoding="utf-8")
            self._maybe_advance_git_head_for_agent_pr(public_response)
            self._maybe_advance_pr_head_for_coder_followup(cmd)
            self._mark_agent_command_seen()
            log_path.write_text(f"$ {' '.join(cmd)}\n\ncodex completed", encoding="utf-8")
            return CommandResult(cmd, cwd_path, stdout, "", returncode)

        if cmd[:1] == ["gemini"]:
            output = self._next_agent_output(self.gemini_outputs)
            explicit_stdout = False
            if isinstance(output, dict):
                stdout = output.get("stdout", "")
                returncode = output.get("returncode", 0)
                explicit_stdout = True
            else:
                stdout, returncode = output
            output = stdout
            if isinstance(output, str) and not explicit_stdout:
                output = self._normalize_legacy_agent_output(output, input_text or "\n".join(cmd))
            self._maybe_write_public_response_file(cmd, prompt=input_text)
            self._maybe_advance_git_head_for_agent_pr(output)
            self._maybe_advance_pr_head_for_coder_followup(cmd)
            self._mark_agent_command_seen()
            log_path.write_text(f"$ {' '.join(cmd)}\n\n{output}", encoding="utf-8")
            return CommandResult(cmd, cwd_path, output, "", returncode)

        if cmd[:1] == ["agy"]:
            output = self._next_agent_output(
                self.antigravity_catalog_outputs if "models" in cmd else self.antigravity_outputs
            )
            if isinstance(output, dict):
                stdout = output.get("stdout", "")
                returncode = output.get("returncode", 0)
            else:
                stdout, returncode = output
            self._maybe_write_public_response_file(cmd)
            self._maybe_advance_git_head_for_agent_pr(stdout)
            self._maybe_advance_pr_head_for_coder_followup(cmd)
            self._mark_agent_command_seen()
            log_path.write_text(f"$ {' '.join(cmd)}\n\n{stdout}", encoding="utf-8")
            return CommandResult(cmd, cwd_path, stdout, "", returncode)

        return self.run(args, cwd=cwd, check=check)

    def run(self, args, *, cwd, input_text=None, check=True, env=None):
        with self._scripted_lock:
            cmd = [str(arg) for arg in args]
            error = self._gh_argv_error(cmd)
            if error is not None:
                cmd, cwd_path = self._record_command(cmd, cwd)
                if check:
                    raise AgentLoopError(
                        f"Command failed with exit 1: {' '.join(cmd)}\n"
                        f"stdout:\n\n\nstderr:\n{error}"
                    )
                return CommandResult(cmd, cwd_path, "", error, 1)
            return self._run_locked(args, cwd=cwd, check=check)

    @staticmethod
    def _gh_argv_error(cmd):
        """Model the pinned gh 2.45 parser so tests catch unsupported flags."""
        if len(cmd) < 2 or cmd[0] != "gh":
            return None
        if cmd[1] == "api":
            value_flags = {"--method", "-H", "-f", "-F", "--input", "--hostname", "--jq"}
            flags = {"--paginate", "--silent", "--verbose"}
            index = 2
            saw_endpoint = False
            while index < len(cmd):
                token = cmd[index]
                if token == "--slurp":
                    return "unknown flag: --slurp (unsupported by gh 2.45.0)"
                if token in value_flags:
                    if index + 1 >= len(cmd):
                        return f"flag {token} requires a value"
                    index += 2
                    continue
                if token in flags:
                    index += 1
                    continue
                if token.startswith("-"):
                    return f"unknown flag: {token}"
                if saw_endpoint:
                    return "gh api accepts one endpoint"
                saw_endpoint = True
                index += 1
            if not saw_endpoint:
                return "gh api requires an endpoint"
            return None
        allowed = {
            "ready": {"--repo"},
            "view": {"--repo", "--json", "--jq", "--comments"},
            "list": {"--repo", "--state", "--search", "--json", "--limit"},
            "comment": {"--repo", "--body", "--body-file"},
            "checks": {"--repo", "--required", "--watch", "--fail-fast"},
            "create": {"--repo", "--title", "--body", "--body-file", "--label"},
            "clone": {"--", "--depth"},
            "repo": {"--json", "--jq", "--repo"},
            "issue": {"--repo", "--body", "--body-file", "--title", "--search", "--json"},
        }
        if len(cmd) < 3 or cmd[2] not in allowed:
            return None
        accepted = allowed[cmd[2]]
        for token in cmd[3:]:
            if token.startswith("-") and token not in accepted and not token.startswith("--inputs"):
                return f"unknown flag: {token}"
        return None

    def _run_locked(self, args, *, cwd, check):
        cmd, cwd_path = self._record_command(args, cwd)

        if cmd[:1] == ["claude"]:
            output, returncode = self._next_agent_output(self.claude_outputs)
            if isinstance(output, str):
                output = self._normalize_legacy_agent_output(output, "\n".join(cmd))
            self._maybe_advance_git_head_for_agent_pr(output)
            self._maybe_advance_pr_head_for_coder_followup(cmd)
            self._mark_agent_command_seen()
            return CommandResult(cmd, cwd_path, output, "", returncode)

        if cmd[:2] == ["codex", "exec"]:
            output = self._next_agent_output(self.codex_outputs)
            if isinstance(output, dict):
                public_response = output.get("public_response", "")
                stdout = output.get("stdout", "")
                returncode = output.get("returncode", 0)
            else:
                public_response, returncode = output
                stdout = public_response
            if isinstance(public_response, str):
                normalized = self._normalize_legacy_agent_output(public_response, "\n".join(cmd))
                if normalized != public_response:
                    public_response = normalized
            if "--output-last-message" in cmd:
                out_path = Path(cmd[cmd.index("--output-last-message") + 1])
                out_path.write_text(public_response, encoding="utf-8")
            self._maybe_advance_git_head_for_agent_pr(public_response)
            self._maybe_advance_pr_head_for_coder_followup(cmd)
            self._mark_agent_command_seen()
            return CommandResult(cmd, cwd_path, stdout, "", returncode)

        if cmd[:3] == ["gh", "pr", "comment"]:
            if "--body-file" in cmd:
                body_path = Path(cmd[cmd.index("--body-file") + 1])
                raw_body = body_path.read_text(encoding="utf-8")
            elif "--body" in cmd:
                raw_body = cmd[cmd.index("--body") + 1]
            else:
                raw_body = ""
            self.comments.append(_strip_round_metadata(raw_body))
            self.pr_payload.setdefault("comments", []).append(
                {
                    "author": {"login": "coding-review-agent-loop"},
                    "createdAt": f"2026-05-23T00:00:{len(self.pr_payload.get('comments', [])):02d}Z",
                    "body": raw_body,
                }
            )
            return CommandResult(cmd, cwd_path, "", "", 0)

        if cmd[:3] == ["gh", "issue", "comment"]:
            if "--body-file" in cmd:
                body_path = Path(cmd[cmd.index("--body-file") + 1])
                raw_body = body_path.read_text(encoding="utf-8")
            elif "--body" in cmd:
                raw_body = cmd[cmd.index("--body") + 1]
            else:
                raw_body = ""
            self.comments.append(_strip_round_metadata(raw_body))
            self.issue_comments.append(
                {
                    "author": {"login": "coding-review-agent-loop"},
                    "createdAt": f"2026-05-23T00:00:{len(self.issue_comments):02d}Z",
                    "body": raw_body,
                }
            )
            return CommandResult(cmd, cwd_path, "", "", 0)

        if cmd[:3] == ["gh", "issue", "create"]:
            title = cmd[cmd.index("--title") + 1]
            if "--body-file" in cmd:
                body_path = Path(cmd[cmd.index("--body-file") + 1])
                body = body_path.read_text(encoding="utf-8")
            else:
                body = cmd[cmd.index("--body") + 1]
            self.issues.append({"title": title, "body": body})
            if self.issue_urls is None:
                issue_url = "https://github.com/OWNER/REPO/issues/99"
            else:
                issue_url = self.issue_urls.pop(0)
            return CommandResult(cmd, cwd_path, f"{issue_url or ''}\n", "", 0)

        if cmd[:3] == ["gh", "issue", "list"]:
            search = cmd[cmd.index("--search") + 1] if "--search" in cmd else ""
            self.search_issues_calls.append(search)
            payload = self.search_issues_payload
            if payload and isinstance(payload, list) and payload and isinstance(payload[0], list):
                results = payload.pop(0) if payload else []
            else:
                results = payload or []
            return CommandResult(cmd, cwd_path, json_dumps(results), "", 0)

        if cmd[:3] == ["gh", "pr", "list"]:
            self.open_prs_calls += 1
            for item in self.open_prs_payload:
                if not isinstance(item, dict):
                    continue
                body = str(item.get("body") or "")
                issue_match = re.search(
                    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#([1-9]\d*)",
                    body,
                    re.I,
                )
                if issue_match:
                    self._provenance_issue_number = int(issue_match.group(1))
                    break
            return CommandResult(cmd, cwd_path, json_dumps(self.open_prs_payload), "", 0)

        if (
            cmd[:3] == ["gh", "pr", "view"]
            and "--json" in cmd
            and cmd[cmd.index("--json") + 1] == "mergeable,mergeStateStatus,headRefOid,baseRefName"
        ):
            self.mergeability_calls += 1
            if self.mergeability_payloads:
                payload = self.mergeability_payloads.pop(0)
            else:
                payload = {
                    "mergeable": self.pr_payload.get("mergeable"),
                    "mergeStateStatus": self.pr_payload.get("mergeStateStatus"),
                    "headRefOid": self.pr_payload.get("headRefOid"),
                    "baseRefName": self.pr_payload.get("baseRefName"),
                }
            return CommandResult(cmd, cwd_path, json_dumps(payload), "", 0)

        if cmd[:3] == ["gh", "pr", "view"]:
            if "--jq" in cmd and ".headRefOid" in cmd:
                return CommandResult(cmd, cwd_path, "abc123\n", "", 0)
            return CommandResult(cmd, cwd_path, json_dumps(self.pr_payload), "", 0)

        if cmd[:3] == ["gh", "repo", "view"] and "defaultBranchRef" in cmd:
            stdout = (
                f"{self.repo_default_branch}\n"
                if self.repo_default_branch_returncode == 0 and self.repo_default_branch
                else ""
            )
            return CommandResult(
                cmd,
                cwd_path,
                stdout,
                "",
                self.repo_default_branch_returncode,
            )

        if cmd[:3] == ["gh", "issue", "view"]:
            payload = {
                "number": self.issue_payload.get("number", 56),
                "title": self.issue_payload.get("title"),
                "body": self.issue_payload.get("body"),
                "url": self.issue_payload.get("url"),
                "author": self.issue_payload.get("author"),
                "createdAt": self.issue_payload.get("createdAt"),
                "comments": self.issue_comments,
            }
            return CommandResult(cmd, cwd_path, json_dumps(payload), "", 0)

        if cmd[:2] == ["gh", "api"] and "/issues/" in cmd[2]:
            return CommandResult(cmd, cwd_path, json_dumps(self.issue_payload), "", 0)

        if cmd[:3] == ["gh", "api", "graphql"]:
            self.pr_commit_calls += 1
            if self.pr_commit_query_failures:
                failure = self.pr_commit_query_failures.pop(0)
                return CommandResult(cmd, cwd_path, "", str(failure), 1)
            if self.pr_commit_pages:
                payload = self.pr_commit_pages.pop(0)
            else:
                head_sha = self.pr_payload.get("headRefOid") or "abc123"
                issue_number = self._provenance_issue_number
                payload = {
                    "data": {
                        "repository": {
                            "pullRequest": {
                                "headRefOid": head_sha,
                                "commits": {
                                    "totalCount": 1,
                                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                                    "nodes": [
                                        {
                                            "commit": {
                                                "oid": "commit-1",
                                                "message": (
                                                    "Implement issue.\n\n"
                                                    "Agent-Issue-Provenance: v1 "
                                                    f"repo=owner/repo issue={issue_number} flow=direct"
                                                ),
                                            }
                                        }
                                    ],
                                },
                            }
                        }
                    }
                }
            if isinstance(payload, list):
                head_sha = self.pr_payload.get("headRefOid") or "abc123"
                payload = {
                    "data": {
                        "repository": {
                            "pullRequest": {
                                "headRefOid": head_sha,
                                "commits": {
                                    "totalCount": len(payload),
                                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                                    "nodes": payload,
                                },
                            }
                        }
                    }
                }
            return CommandResult(cmd, cwd_path, json_dumps(payload), "", 0)

        if cmd[:2] == ["gh", "api"] and cmd[2].endswith("/check-runs"):
            if "--jq" in cmd:
                return CommandResult(cmd, cwd_path, "success\n", "", 0)
            stdout = (
                json_dumps(self.pr_check_runs_payload) if self.pr_check_runs_returncode == 0 else ""
            )
            return CommandResult(
                cmd,
                cwd_path,
                stdout,
                self.pr_check_runs_stderr,
                self.pr_check_runs_returncode,
            )

        if cmd[:2] == ["gh", "api"] and cmd[2].endswith("/status"):
            stdout = json_dumps(self.pr_status_payload) if self.pr_status_returncode == 0 else ""
            return CommandResult(
                cmd,
                cwd_path,
                stdout,
                self.pr_status_stderr,
                self.pr_status_returncode,
            )

        if cmd[:2] == ["gh", "api"] and any(
            "/statuses?per_page=100" in part for part in cmd[2:]
        ):
            payload = self.pr_status_payload
            pages = payload.get("pages") if isinstance(payload, dict) else None
            stdout = (
                "\n".join(json_dumps(page) for page in pages)
                if isinstance(pages, list)
                else json_dumps(payload.get("statuses", []))
            )
            return CommandResult(cmd, cwd_path, stdout, self.pr_status_stderr, self.pr_status_returncode)

        if cmd[:2] == ["gh", "api"] and "/protection/required_status_checks" in cmd[2]:
            stdout = (
                json_dumps(self.pr_branch_protection_payload)
                if self.pr_branch_protection_returncode == 0
                else ""
            )
            return CommandResult(
                cmd,
                cwd_path,
                stdout,
                self.pr_branch_protection_stderr,
                self.pr_branch_protection_returncode,
            )

        if cmd[:2] == ["gh", "api"] and cmd[2].endswith("/protection/enforce_admins"):
            return CommandResult(cmd, cwd_path, json_dumps(self.pr_enforce_admins_payload), "", 0)

        if cmd[:2] == ["gh", "api"] and "/rules/branches/" in cmd[2]:
            return CommandResult(
                cmd,
                cwd_path,
                json_dumps(self.pr_effective_rules_payload)
                if self.pr_effective_rules_returncode == 0
                else "",
                self.pr_effective_rules_stderr,
                self.pr_effective_rules_returncode,
            )

        if cmd[:2] == ["gh", "api"] and "/rulesets/" in cmd[2]:
            ruleset_id = cmd[2].rsplit("/", 1)[-1]
            payload = self.pr_rulesets_payload.get(ruleset_id, self.pr_rulesets_payload.get(int(ruleset_id), {}))
            return CommandResult(cmd, cwd_path, json_dumps(payload), "", 0)

        if cmd[:2] == ["gh", "api"] and cmd[2] == "repos/OWNER/REPO":
            return CommandResult(cmd, cwd_path, json_dumps(self.repo_payload), "", 0)

        if cmd[:1] == ["sleep"]:
            return CommandResult(cmd, cwd_path, "", "", 0)

        if cmd[:3] == ["git", "rev-parse", "--is-inside-work-tree"]:
            if self.git_inside:
                return CommandResult(cmd, cwd_path, "true\n", "", 0)
            return CommandResult(cmd, cwd_path, "false\n", "", 1)

        if cmd[:3] == ["git", "rev-parse", "HEAD"]:
            self.git_probe_calls.append(tuple(cmd))
            if self.git_probe_exceptions:
                raise self.git_probe_exceptions.pop(0)
            if self.git_probe_results:
                scripted = self.git_probe_results.pop(0)
                if isinstance(scripted, CommandResult):
                    return CommandResult(
                        cmd, cwd_path, scripted.stdout, scripted.stderr, scripted.returncode
                    )
                if isinstance(scripted, dict):
                    return CommandResult(
                        cmd,
                        cwd_path,
                        scripted.get("stdout", ""),
                        scripted.get("stderr", ""),
                        scripted.get("returncode", 0),
                    )
            return CommandResult(cmd, cwd_path, f"{self.git_head}\n", "", 0)

        if cmd[:3] == ["git", "checkout", "--detach"]:
            if len(cmd) > 3 and cmd[3].startswith("refs/remotes/origin/pr/"):
                self.git_head = self.pr_payload.get("headRefOid", self.git_head)
            return CommandResult(cmd, cwd_path, "", "", 0)

        if cmd[:2] == ["git", "ls-files"]:
            return CommandResult(cmd, cwd_path, "\n".join(self.tracked_files) + "\n", "", 0)

        if cmd == ["git", "diff", "HEAD", "--binary"]:
            stdout = self._current_git_diff() if self.diff_returncode == 0 else ""
            return CommandResult(cmd, cwd_path, stdout, self.diff_stderr, self.diff_returncode)

        if cmd == ["git", "diff", "--stat", "HEAD"]:
            stdout = self._current_git_diff_stat() if self.diff_returncode == 0 else ""
            return CommandResult(cmd, cwd_path, stdout, self.diff_stderr, self.diff_returncode)

        if cmd == ["git", "diff", "--check"]:
            return CommandResult(
                cmd,
                cwd_path,
                self._current_git_diff_check(),
                self._current_git_diff_check_stderr(),
                self._current_git_diff_check_returncode(),
            )

        if cmd[:3] == ["git", "diff", "--name-only"]:
            stdout = "\n".join(self.changed_files) + "\n" if self.diff_returncode == 0 else ""
            return CommandResult(cmd, cwd_path, stdout, self.diff_stderr, self.diff_returncode)

        if cmd[:4] == ["git", "remote", "get-url", "origin"]:
            return CommandResult(cmd, cwd_path, f"{self.git_remote}\n", "", 0)

        if cmd[:3] == ["git", "status", "--porcelain"]:
            self.git_probe_calls.append(tuple(cmd))
            if self.git_probe_exceptions:
                raise self.git_probe_exceptions.pop(0)
            if self.git_probe_results:
                scripted = self.git_probe_results.pop(0)
                if isinstance(scripted, CommandResult):
                    return CommandResult(
                        cmd, cwd_path, scripted.stdout, scripted.stderr, scripted.returncode
                    )
                if isinstance(scripted, dict):
                    return CommandResult(
                        cmd,
                        cwd_path,
                        scripted.get("stdout", ""),
                        scripted.get("stderr", ""),
                        scripted.get("returncode", 0),
                    )
            return CommandResult(cmd, cwd_path, self._current_git_status(), "", 0)

        if cmd[:3] == ["git", "status", "--short"]:
            return CommandResult(cmd, cwd_path, self._current_git_status(), "", 0)

        if cmd[:3] == ["gh", "repo", "clone"]:
            Path(cmd[4]).mkdir(parents=True, exist_ok=True)
            return CommandResult(cmd, cwd_path, "", "", 0)

        return CommandResult(cmd, cwd_path, "", "", 0)


def json_dumps(value):
    import json

    return json.dumps(value) + "\n"


def command_index(commands, prefix, *, start=0):
    for index in range(start, len(commands)):
        cmd = commands[index][0]
        if cmd[: len(prefix)] == prefix:
            return index
    raise AssertionError(f"Command with prefix {prefix!r} not found.")


def read_usage_summary(log_dir: Path) -> dict:
    summary_paths = list(log_dir.glob("*-usage-summary.json"))
    assert len(summary_paths) == 1
    return json.loads(summary_paths[0].read_text(encoding="utf-8"))


def prior_item_dispositions(*lines: str) -> str:
    if not lines:
        return ""
    return "\n\n### Prior unresolved item dispositions\n" + "\n".join(f"- {line}" for line in lines)


def blocking_issues(*lines: str) -> str:
    if not lines:
        return ""
    return "\n\n### Blocking issues\n" + "\n".join(f"- {line}" for line in lines)


def prior_plan_item_dispositions(*lines: str) -> str:
    if not lines:
        return ""
    return "\n\n### Prior unresolved plan item dispositions\n" + "\n".join(
        f"- {line}" for line in lines
    )


def structured_pr_review(
    *,
    state: str = "approved",
    summary: str = "Review complete.",
    blocking_items: list[str] | None = None,
    same_pr_followups: list[str] | None = None,
    future_followups: list[str] | None = None,
    prior_item_dispositions: list[dict[str, str]] | None = None,
    reviewer: str = "OpenAI Codex",
    human_requirements_resolved: bool = False,
) -> str:
    return (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": state,
                "summary": summary,
                "blocking_items": blocking_items or [],
                "same_pr_followups": same_pr_followups or [],
                "future_followups": future_followups or [],
                "prior_item_dispositions": prior_item_dispositions or [],
            }
        )
        + ("\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->" if human_requirements_resolved else "")
        + f"\n<!-- AGENT_STATE: {state} -->\n-- {reviewer}"
    )


def structured_plan_review(
    *,
    state: str = "approved",
    summary: str = "Plan review complete.",
    blocking_plan_issues: list[str] | None = None,
    same_plan_followups: list[str] | None = None,
    future_followups: list[str] | None = None,
    prior_plan_item_dispositions: list[dict[str, str]] | None = None,
    reviewer: str = "OpenAI Codex",
    human_requirements_resolved: bool = False,
    human_requirement_dispositions: list[dict[str, str]] | None = None,
) -> str:
    return (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_review",
                "state": state,
                "summary": summary,
                "blocking_plan_issues": blocking_plan_issues or [],
                "same_plan_followups": same_plan_followups or [],
                "future_followups": future_followups or [],
                "prior_plan_item_dispositions": prior_plan_item_dispositions or [],
                "human_requirement_dispositions": human_requirement_dispositions or [],
            }
        )
        + ("\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->" if human_requirements_resolved else "")
        + f"\n<!-- AGENT_PLAN_STATE: {state} -->\n-- {reviewer}"
    )


def structured_plan_revision(
    *,
    summary: str = "Revised the plan.",
    prior_plan_item_dispositions: list[dict[str, str]] | None = None,
    plan_steps: list[str] | None = None,
    reviewer: str = "Anthropic Claude",
    human_requirements: str = "",
    human_requirement_dispositions: list[dict[str, str]] | None = None,
    deferred_stages: list[dict[str, str]] | None = None,
    child_stages: list[dict[str, str]] | None = None,
    external_dependencies: list[dict[str, str]] | None = None,
    deferred_work: list[dict[str, str]] | None = None,
    plan_actions: list[dict[str, str]] | None = None,
) -> str:
    payload = {
        "schema_version": 1,
        "kind": "plan_revision",
        "state": "blocking",
        "summary": summary,
        "prior_plan_item_dispositions": prior_plan_item_dispositions or [],
        "plan_steps": plan_steps or ["Update the plan.", "Run the relevant tests."],
        "human_requirement_dispositions": human_requirement_dispositions or (
            [{"requirement_id": "Requirement 1", "disposition": "addressed", "evidence": "The plan covers the signed requirement."}]
            if human_requirements else []
        ),
    }
    if deferred_stages is not None:
        payload["deferred_stages"] = deferred_stages
    if child_stages is not None:
        payload["child_stages"] = child_stages
    for name, value in (
        ("external_dependencies", external_dependencies),
        ("deferred_work", deferred_work),
        ("plan_actions", plan_actions),
    ):
        if value is not None:
            payload[name] = value
    return (
        json.dumps(payload)
        + human_requirements
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n"
        + f"-- {reviewer}"
    )


def structured_plan_state(
    *,
    state: str = "blocking",
    summary: str = "Implementation plan.",
    plan_steps: list[str] | None = None,
    reviewer: str = "Anthropic Claude",
    deferred_stages: list[dict[str, str]] | None = None,
    child_stages: list[dict[str, str]] | None = None,
    external_dependencies: list[dict[str, str]] | None = None,
    deferred_work: list[dict[str, str]] | None = None,
    plan_actions: list[dict[str, str]] | None = None,
    human_requirement_dispositions: list[dict[str, str]] | None = None,
) -> str:
    payload = {
        "schema_version": 1,
        "kind": "plan_state",
        "state": state,
        "summary": summary,
        "plan_steps": plan_steps or ["Update the code.", "Run the relevant tests."],
        "human_requirement_dispositions": human_requirement_dispositions or [],
    }
    if deferred_stages is not None:
        payload["deferred_stages"] = deferred_stages
    if child_stages is not None:
        payload["child_stages"] = child_stages
    for name, value in (
        ("external_dependencies", external_dependencies),
        ("deferred_work", deferred_work),
        ("plan_actions", plan_actions),
    ):
        if value is not None:
            payload[name] = value
    return (
        json.dumps(payload)
        + f"\n<!-- AGENT_PLAN_STATE: {state} -->\n"
        + f"-- {reviewer}"
    )


def structured_coder_followup(
    *,
    state: str = "blocking",
    summary: str = "Updated the PR.",
    addressed_items: list[str] | None = None,
    remaining_items: list[str] | None = None,
    addressed_item_notes: dict[str, str] | None = None,
    remaining_item_notes: dict[str, str] | None = None,
    human_requirement_ids: list[str] | None = None,
    human_requirement_dispositions: list[dict[str, str]] | None = None,
    checked_discussion_directly: bool = False,
    tests_run: list[str] | None = None,
    disputed_items: list[str] | None = None,
    dispute_evidence: dict[str, str] | None = None,
    reviewer: str = "Anthropic Claude",
) -> str:
    payload: dict = {
        "schema_version": 1,
        "kind": "coder_followup",
        "state": state,
        "summary": summary,
        "addressed_items": addressed_items or [],
        "remaining_items": remaining_items or [],
        "addressed_item_notes": addressed_item_notes or {},
        "remaining_item_notes": remaining_item_notes or {},
        "human_requirements": {
            "addressed_ids": human_requirement_ids or [],
            "checked_discussion_directly": checked_discussion_directly,
        },
        "human_requirement_dispositions": (
            human_requirement_dispositions
            if human_requirement_dispositions is not None
            else [
                {
                    "requirement_id": requirement_id,
                    "disposition": "addressed",
                    "evidence": "The follow-up addresses the surfaced requirement.",
                }
                for requirement_id in (human_requirement_ids or [])
            ]
        ),
    }
    if tests_run is not None:
        payload["tests_run"] = tests_run
    if disputed_items is not None:
        payload["disputed_items"] = disputed_items
    if dispute_evidence is not None:
        payload["dispute_evidence"] = dispute_evidence
    return json.dumps(payload) + f"\n<!-- AGENT_STATE: {state} -->\n-- {reviewer}"


def structured_issue_implementation(
    *,
    state: str = "blocking",
    summary: str = "Implemented the requested change.",
    pr_number: int | None = 77,
    human_requirement_ids: list[str] | None = None,
    human_requirement_dispositions: list[dict[str, str]] | None = None,
    checked_discussion_directly: bool = False,
    tests_run: list[str] | None = None,
    reviewer: str = "Anthropic Claude",
) -> str:
    ids = human_requirement_ids or []
    payload: dict = {
        "schema_version": 1,
        "kind": "issue_implementation",
        "state": state,
        "summary": summary,
        "pr_number": pr_number,
        "human_requirements": {
            "addressed_ids": ids,
            "checked_discussion_directly": checked_discussion_directly,
        },
        "human_requirement_dispositions": (
            human_requirement_dispositions
            if human_requirement_dispositions is not None
            else [
                {
                    "requirement_id": requirement_id,
                    "disposition": "addressed",
                    "evidence": "The implementation covers the surfaced requirement.",
                }
                for requirement_id in ids
            ]
        ),
    }
    if tests_run is not None:
        payload["tests_run"] = tests_run
    return json.dumps(payload) + f"\n<!-- AGENT_STATE: {state} -->\n-- {reviewer}"


def make_config(tmp_path, *, create_dirs=True, **overrides):
    config = {
        "repo": "OWNER/REPO",
        "claude_dir": tmp_path / "claude",
        "codex_dir": tmp_path / "codex",
        "gemini_dir": tmp_path / "gemini",
        "coder": "claude",
        "reviewer": "codex",
        "base": "main",
        "max_rounds": 5,
        "auto_merge": False,
        "dry_run": False,
        "allow_shared_dir": False,
        "claude_cmd": "claude",
        "codex_cmd": "codex",
        "gemini_cmd": "gemini",
        "gh_cmd": "gh",
        "claude_args": (),
        "codex_args": (),
        "gemini_args": (),
        "test_command": None,
        "pre_review_tests": True,
        "ci_check_name": "test",
        "ci_timeout_seconds": 1200,
        "ci_poll_interval_seconds": 30,
        "quiet": True,
        "log_dir": tmp_path / "logs",
        "progress_interval_seconds": 30,
        "agent_max_retries": 2,
        "agent_retry_backoff_seconds": (1, 1),
        "agent_memory": True,
        "refresh_agent_memory": False,
        "agent_memory_dir": tmp_path / "claude" / ".agent-loop" / "memory",
        "refresh_test_profile": False,
    }
    config.update(overrides)
    if create_dirs:
        config["claude_dir"].mkdir(parents=True, exist_ok=True)
        config["codex_dir"].mkdir(parents=True, exist_ok=True)
        config["gemini_dir"].mkdir(parents=True, exist_ok=True)
    return AgentLoopConfig(**config)


def plan_decomposition_json(*phases):
    if not phases:
        phases = (
            {
                "title": "Internal schema utilities",
                "scope": "Add internal helpers only.",
                "non_goals": "No live orchestrator behavior changes.",
                "dependency_notes": "First phase; no dependencies.",
                "rollout_risk": "low - internal only.",
                "validation": "Run parser and orchestrator tests before the next phase.",
                "parent_context": "Approved plan slice: add helpers and tests while preserving existing behavior.",
                "automation": "agent-pr",
                "depends_on": [],
            },
        )
    return json.dumps(
        {
            "schema_version": 1,
            "kind": "plan_decomposition",
            "phases": list(phases),
        }
    )


__all__ = [name for name in globals() if not name.startswith("__")]
