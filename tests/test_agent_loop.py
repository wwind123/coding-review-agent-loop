import base64
import datetime
import json
import os
import re
import subprocess
import sys
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
    _is_transient_agent_output,
    _is_transient_public_response,
    _parse_rate_limit_reset_seconds,
    _run_validated_agent,
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
    _apply_unresolved_item_dispositions,
    _attach_round_metadata,
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
    build_followup_prompt,
    build_issue_implementation_prompt,
    build_issue_plan_prompt,
    build_issue_prompt,
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
    validate_structured_human_requirements_acknowledgement,
    validate_structured_plan_state,
    validate_structured_plan_revision,
)
from coding_review_agent_loop.workdir_guard import (
    extract_reported_tests_from_response,
    validate_response_tests_within_workdir,
    validate_test_commands_within_workdir,
)

from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def _no_real_repair():
    """Prevent attempt_repair from calling the real Gemini CLI in all tests.

    Tests that explicitly test repair behaviour patch the orchestrator-level
    import themselves, which takes precedence over this fixture.  Unit tests
    for attempt_repair itself patch subprocess.run directly, so they are
    unaffected here.
    """
    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=None):
        yield


@pytest.fixture(autouse=True)
def _agent_commands_available(monkeypatch):
    """Keep config tests independent of agent CLIs installed on the test host."""
    import coding_review_agent_loop.config as config_module

    real_which = config_module.shutil.which

    def which(command):
        resolved = real_which(command)
        if resolved is not None:
            return resolved
        if command in {"claude", "codex", "gemini", "agy"}:
            return f"/mock/bin/{command}"
        return None

    monkeypatch.setattr(config_module.shutil, "which", which)


class FakeRunner(Runner):
    def __init__(
        self,
        *,
        claude_outputs=None,
        codex_outputs=None,
        gemini_outputs=None,
        antigravity_outputs=None,
        issue_payload=None,
        issue_comments=None,
        pr_payload=None,
        pr_check_runs_payload=None,
        pr_status_payload=None,
        pr_branch_protection_payload=None,
        pr_branch_protection_returncode=0,
        pr_branch_protection_stderr="",
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
        issue_urls=None,
        public_response_outputs=None,
        advance_git_head_on_pr=True,
    ):
        super().__init__(dry_run=False)
        self.claude_outputs = list(claude_outputs or [])
        self.codex_outputs = list(codex_outputs or [])
        self.gemini_outputs = list(gemini_outputs or [])
        self.antigravity_outputs = list(antigravity_outputs or [])
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
            "body": "Fixes #56",
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
        self.issue_urls = list(issue_urls) if issue_urls is not None else None
        self.public_response_outputs = list(public_response_outputs or [])
        self.advance_git_head_on_pr = advance_git_head_on_pr
        self._agent_pr_counter = 0

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

    def _maybe_write_public_response_file(self, cmd):
        if not self.public_response_outputs:
            return
        prompt = "\n".join(cmd)
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
        if "<!-- AGENT_PR:" not in text and "github.com/OWNER/REPO/pull/" not in text:
            return
        self._agent_pr_counter += 1
        self.git_head = f"{self.git_head}-agent-{self._agent_pr_counter}"

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
        use_pty=False,
    ):
        cmd, cwd_path = self._record_command(args, cwd)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ensure_log_dir_ignored(log_path.parent)

        if cmd[:1] == ["claude"]:
            output, returncode = self._next_agent_output(self.claude_outputs)
            if isinstance(output, str):
                output = self._normalize_legacy_agent_output(output, "\n".join(cmd))
            self._maybe_write_public_response_file(cmd)
            self._maybe_advance_git_head_for_agent_pr(output)
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
            self._maybe_write_public_response_file(cmd)
            if "--output-last-message" in cmd:
                out_path = Path(cmd[cmd.index("--output-last-message") + 1])
                out_path.write_text(public_response, encoding="utf-8")
            self._maybe_advance_git_head_for_agent_pr(public_response)
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
                output = self._normalize_legacy_agent_output(output, "\n".join(cmd))
            self._maybe_write_public_response_file(cmd)
            self._maybe_advance_git_head_for_agent_pr(output)
            log_path.write_text(f"$ {' '.join(cmd)}\n\n{output}", encoding="utf-8")
            return CommandResult(cmd, cwd_path, output, "", returncode)

        if cmd[:1] == ["agy"]:
            output = self._next_agent_output(self.antigravity_outputs)
            if isinstance(output, dict):
                stdout = output.get("stdout", "")
                returncode = output.get("returncode", 0)
            else:
                stdout, returncode = output
            self._maybe_write_public_response_file(cmd)
            self._maybe_advance_git_head_for_agent_pr(stdout)
            log_path.write_text(f"$ {' '.join(cmd)}\n\n{stdout}", encoding="utf-8")
            return CommandResult(cmd, cwd_path, stdout, "", returncode)

        return self.run(args, cwd=cwd, check=check)

    def run(self, args, *, cwd, input_text=None, check=True, env=None):
        cmd, cwd_path = self._record_command(args, cwd)

        if cmd[:1] == ["claude"]:
            output, returncode = self._next_agent_output(self.claude_outputs)
            if isinstance(output, str):
                output = self._normalize_legacy_agent_output(output, "\n".join(cmd))
            self._maybe_advance_git_head_for_agent_pr(output)
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

        if cmd[:1] == ["sleep"]:
            return CommandResult(cmd, cwd_path, "", "", 0)

        if cmd[:3] == ["git", "rev-parse", "--is-inside-work-tree"]:
            if self.git_inside:
                return CommandResult(cmd, cwd_path, "true\n", "", 0)
            return CommandResult(cmd, cwd_path, "false\n", "", 1)

        if cmd[:3] == ["git", "rev-parse", "HEAD"]:
            return CommandResult(cmd, cwd_path, f"{self.git_head}\n", "", 0)

        if cmd[:3] == ["git", "checkout", "--detach"]:
            if len(cmd) > 3 and cmd[3].startswith("refs/remotes/origin/pr/"):
                self.git_head = self.pr_payload.get("headRefOid", self.git_head)
            return CommandResult(cmd, cwd_path, "", "", 0)

        if cmd[:2] == ["git", "ls-files"]:
            return CommandResult(cmd, cwd_path, "\n".join(self.tracked_files) + "\n", "", 0)

        if cmd[:3] == ["git", "diff", "--name-only"]:
            stdout = "\n".join(self.changed_files) + "\n" if self.diff_returncode == 0 else ""
            return CommandResult(cmd, cwd_path, stdout, self.diff_stderr, self.diff_returncode)

        if cmd[:4] == ["git", "remote", "get-url", "origin"]:
            return CommandResult(cmd, cwd_path, f"{self.git_remote}\n", "", 0)

        if cmd[:3] == ["git", "status", "--porcelain"]:
            return CommandResult(cmd, cwd_path, self.git_status, "", 0)

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
) -> str:
    return (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_revision",
                "state": "blocking",
                "summary": summary,
                "prior_plan_item_dispositions": prior_plan_item_dispositions or [],
                "plan_steps": plan_steps or ["Update the plan.", "Run the relevant tests."],
            }
        )
        + human_requirements
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n"
        + f"-- {reviewer}"
    )


def structured_plan_state(
    *,
    state: str = "approved",
    summary: str = "Implementation plan.",
    plan_steps: list[str] | None = None,
    reviewer: str = "Anthropic Claude",
) -> str:
    return (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_state",
                "state": state,
                "summary": summary,
                "plan_steps": plan_steps or ["Update the code.", "Run the relevant tests."],
            }
        )
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
    checked_discussion_directly: bool = False,
    tests_run: list[str] | None = None,
    reviewer: str = "Anthropic Claude",
) -> str:
    return (
        json.dumps(
            {
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
                **({"tests_run": tests_run} if tests_run is not None else {}),
            }
        )
        + f"\n<!-- AGENT_STATE: {state} -->\n-- {reviewer}"
    )


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


def test_workdir_guard_rejects_outside_home_path(tmp_path):
    assigned = tmp_path / "claude" / "repo"
    assigned.mkdir(parents=True)

    with pytest.raises(AgentLoopError, match="outside the assigned checkout"):
        validate_test_commands_within_workdir(
            ("cd ~/llm-dialectic && python -m pytest",),
            assigned_workdir=assigned,
        )


def test_workdir_guard_rejects_windows_path_with_clear_message(tmp_path):
    assigned = tmp_path / "claude" / "repo"
    assigned.mkdir(parents=True)

    with pytest.raises(
        AgentLoopError,
        match="cannot be validated against the assigned Unix checkout",
    ):
        validate_test_commands_within_workdir(
            (r"cd C:\Users\dev\repo && python -m pytest",),
            assigned_workdir=assigned,
        )


def test_workdir_guard_accepts_assigned_absolute_path(tmp_path):
    assigned = tmp_path / "claude" / "repo"
    tests_dir = assigned / "tests"
    tests_dir.mkdir(parents=True)

    validate_test_commands_within_workdir(
        (f"cd {assigned} && python -m pytest {tests_dir}",),
        assigned_workdir=assigned,
    )


def test_workdir_guard_accepts_javascript_regex_closing_script_tag(tmp_path):
    assigned = tmp_path / "codex" / "repo"
    assigned.mkdir(parents=True)

    validate_test_commands_within_workdir(
        (
            r"""node -e "const fs=require('fs'); const html=fs.readFileSync('server/static/index.html','utf8'); const scripts=[...html.matchAll(/<script(?![^>]*\\bsrc=)[^>]*>([\\s\\S]*?)<\\/script>/gi)].map(m=>m[1]); scripts.forEach((code,i)=>{ try { new Function(code); } catch(e) { console.error('script '+i+' parse failed'); throw e; } }); console.log(scripts.length+' inline scripts parsed');" (failed: naive regex matched non-code text)""",
        ),
        assigned_workdir=assigned,
    )


def test_workdir_guard_accepts_relative_test_commands(tmp_path):
    assigned = tmp_path / "claude" / "repo"
    assigned.mkdir(parents=True)

    validate_test_commands_within_workdir(
        ("python -m pytest tests/test_agent_loop.py", "make test"),
        assigned_workdir=assigned,
    )


def test_workdir_guard_extracts_tests_section_only(tmp_path):
    assigned = tmp_path / "claude" / "repo"
    assigned.mkdir(parents=True)
    text = (
        "Issue context mentioned Tests: cd ~/other && pytest.\n\n"
        "Implemented.\n"
        "Tests: python -m pytest tests/test_agent_loop.py passed.\n"
        "<!-- AGENT_PR: 77 -->"
    )

    assert extract_reported_tests_from_response(text) == (
        "python -m pytest tests/test_agent_loop.py passed.",
    )
    validate_response_tests_within_workdir(text, assigned_workdir=assigned)


def test_coder_prompts_include_assigned_workdir_rule(tmp_path):
    config = make_config(tmp_path, coder="codex")
    assigned = str(config.codex_dir.resolve())

    prompts = [
        build_issue_prompt(56, config),
        build_issue_plan_prompt(56, config),
        build_issue_implementation_prompt(56, "1. Fix it.", config),
        build_task_prompt("Fix the bug.", config),
        build_followup_prompt(77, 1, "Needs tests.", config),
        build_same_pr_followup_prompt(77, 1, "Tighten docs.", config),
    ]

    for prompt in prompts:
        assert f"Assigned checkout: `{assigned}`" in prompt
        assert "`AGENT_LOOP_WORKDIR` is set to this path" in prompt
        assert "must stay in that directory" in prompt
        assert "Do not `cd` into sibling, home, deployment, or duplicate clones" in prompt
        assert "`pwd` and `git status --branch --short`" in prompt


def test_reviewer_prompts_use_reviewer_assigned_workdir_rule(tmp_path):
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer=("codex",),
        claude_args=(),
        codex_args=("--dangerously-bypass-approvals-and-sandbox",),
    )
    reviewer_assigned = str(config.codex_dir.resolve())
    coder_assigned = str(config.claude_dir.resolve())

    prompts = [
        build_plan_review_prompt(56, 1, "Plan.", config, reviewer="codex"),
        build_plan_review_prompt(
            56,
            1,
            "Plan.",
            config,
            reviewer="codex",
            compact_context=True,
        ),
        build_review_prompt(77, 1, config, reviewer="codex"),
        build_review_prompt(77, 1, config, reviewer="codex", compact_context=True),
    ]

    for prompt in prompts:
        assert f"Assigned checkout: `{reviewer_assigned}`" in prompt
        assert f"Assigned checkout: `{coder_assigned}`" not in prompt
        assert "Inspection must stay in that directory" in prompt
        assert "Dangerous agent permissions are active" in prompt


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_migration(revision: str, down_revision: str | tuple[str, ...] | None) -> str:
    return (
        f'revision = "{revision}"\n'
        f"down_revision = {repr(down_revision)}\n"
        "branch_labels = None\n"
        "depends_on = None\n"
    )


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


def _init_git_checkout_with_origin(tmp_path: Path) -> Path:
    origin = tmp_path / "origin.git"
    worktree = tmp_path / "repo"
    _git(tmp_path, "init", "--bare", str(origin))
    _git(tmp_path, "clone", str(origin), str(worktree))
    _git(worktree, "config", "user.email", "test@example.com")
    _git(worktree, "config", "user.name", "Test User")
    _git(worktree, "switch", "-c", "main")
    return worktree


def _commit_all(worktree: Path, message: str) -> None:
    _git(worktree, "add", ".")
    _git(worktree, "commit", "-m", message)


def _push_main(worktree: Path) -> None:
    _git(worktree, "push", "-u", "origin", "main")


def test_validate_pr_migration_topology_blocks_wrong_down_revision(tmp_path):
    worktree = _init_git_checkout_with_origin(tmp_path)
    _write(
        worktree / "alembic" / "versions" / "5d5f0e1a2b3c_base.py",
        _make_migration("5d5f0e1a2b3c", None),
    )
    _write(
        worktree / "alembic" / "versions" / "a6b7c8d9e0f1_add_feature.py",
        _make_migration("a6b7c8d9e0f1", "5d5f0e1a2b3c"),
    )
    _write(
        worktree / "alembic" / "versions" / "402b9e8af79b_latest.py",
        _make_migration("402b9e8af79b", "a6b7c8d9e0f1"),
    )
    _commit_all(worktree, "Base migrations")
    _push_main(worktree)

    _git(worktree, "switch", "-c", "feature/wrong-parent")
    _write(
        worktree / "alembic" / "versions" / "e4f5a6b7c8d9_add_gemini_3_5_flash_pricing.py",
        _make_migration("e4f5a6b7c8d9", "5d5f0e1a2b3c"),
    )
    _commit_all(worktree, "Add migration with stale down_revision")

    result = validate_pr_migration_topology(
        Runner(),
        config=make_config(tmp_path / "config"),
        checkout=worktree,
        pr_metadata=PullRequestMetadata(
            number=77,
            repo="OWNER/REPO",
            title="Add migration",
            head_branch="feature/wrong-parent",
            base_branch="main",
            head_sha=None,
            url=None,
        ),
    )

    assert result.ok is False
    assert result.message is not None
    assert "e4f5a6b7c8d9_add_gemini_3_5_flash_pricing.py" in result.message
    assert "`down_revision = '5d5f0e1a2b3c'`" in result.message
    assert "`402b9e8af79b`" in result.message
    assert "`e4f5a6b7c8d9`" in result.message


def test_validate_pr_migration_topology_allows_linear_head_extension(tmp_path):
    worktree = _init_git_checkout_with_origin(tmp_path)
    _write(
        worktree / "alembic" / "versions" / "5d5f0e1a2b3c_base.py",
        _make_migration("5d5f0e1a2b3c", None),
    )
    _write(
        worktree / "alembic" / "versions" / "402b9e8af79b_latest.py",
        _make_migration("402b9e8af79b", "5d5f0e1a2b3c"),
    )
    _commit_all(worktree, "Base migrations")
    _push_main(worktree)

    _git(worktree, "switch", "-c", "feature/right-parent")
    _write(
        worktree / "alembic" / "versions" / "e4f5a6b7c8d9_add_pricing.py",
        _make_migration("e4f5a6b7c8d9", "402b9e8af79b"),
    )
    _commit_all(worktree, "Add linear migration")

    result = validate_pr_migration_topology(
        Runner(),
        config=make_config(tmp_path / "config"),
        checkout=worktree,
        pr_metadata=PullRequestMetadata(
            number=77,
            repo="OWNER/REPO",
            title="Add migration",
            head_branch="feature/right-parent",
            base_branch="main",
            head_sha=None,
            url=None,
        ),
    )

    assert result.ok is True
    assert result.message is None


def test_validate_pr_migration_topology_skips_block_when_base_already_has_multiple_heads(tmp_path):
    worktree = _init_git_checkout_with_origin(tmp_path)
    _write(
        worktree / "alembic" / "versions" / "111111111111_first_head.py",
        _make_migration("111111111111", None),
    )
    _write(
        worktree / "alembic" / "versions" / "222222222222_second_head.py",
        _make_migration("222222222222", None),
    )
    _commit_all(worktree, "Base has multiple heads")
    _push_main(worktree)

    _git(worktree, "switch", "-c", "feature/merge-heads")
    _write(
        worktree / "alembic" / "versions" / "333333333333_merge_heads.py",
        _make_migration("333333333333", ("111111111111", "222222222222")),
    )
    _commit_all(worktree, "Merge existing heads")

    result = validate_pr_migration_topology(
        Runner(),
        config=make_config(tmp_path / "config"),
        checkout=worktree,
        pr_metadata=PullRequestMetadata(
            number=77,
            repo="OWNER/REPO",
            title="Merge heads",
            head_branch="feature/merge-heads",
            base_branch="main",
            head_sha=None,
            url=None,
        ),
    )

    assert result.ok is True
    assert result.message is None


def test_validate_pr_migration_topology_blocks_non_literal_changed_metadata(tmp_path):
    worktree = _init_git_checkout_with_origin(tmp_path)
    _write(
        worktree / "alembic" / "versions" / "402b9e8af79b_latest.py",
        _make_migration("402b9e8af79b", None),
    )
    _commit_all(worktree, "Base migrations")
    _push_main(worktree)

    _git(worktree, "switch", "-c", "feature/non-literal-migration")
    _write(
        worktree / "alembic" / "versions" / "e4f5a6b7c8d9_non_literal.py",
        'revision = "e4f5a6b7c8d9"\n'
        "PREVIOUS = '402b9e8af79b'\n"
        "down_revision = PREVIOUS\n"
        "branch_labels = None\n"
        "depends_on = None\n",
    )
    _commit_all(worktree, "Add non-literal migration metadata")

    result = validate_pr_migration_topology(
        Runner(),
        config=make_config(tmp_path / "config"),
        checkout=worktree,
        pr_metadata=PullRequestMetadata(
            number=77,
            repo="OWNER/REPO",
            title="Bad migration metadata",
            head_branch="feature/non-literal-migration",
            base_branch="main",
            head_sha=None,
            url=None,
        ),
    )

    assert result.ok is False
    assert result.message is not None
    assert "Could not validate Alembic revision metadata" in result.message
    assert "e4f5a6b7c8d9_non_literal.py" in result.message


def test_parse_claude_output_extracts_text_and_session_id():
    raw = json.dumps({"result": "Hello.", "session_id": "abc123"})
    text, sid, usage, raw_usage, model = _parse_claude_output(raw)
    assert text == "Hello."
    assert sid == "abc123"
    assert usage is None
    assert raw_usage is None
    assert model is None


def test_parse_claude_output_falls_back_on_plain_text():
    raw = "plain response"
    text, sid, usage, raw_usage, model = _parse_claude_output(raw)
    assert text == "plain response"
    assert sid is None
    assert usage is None
    assert raw_usage is None
    assert model is None


def test_parse_claude_output_falls_back_on_non_string_result():
    raw = json.dumps({"result": 42, "session_id": "abc"})
    text, sid, usage, raw_usage, model = _parse_claude_output(raw)
    assert text == raw  # non-string result → fall back to raw
    assert sid == "abc"
    assert usage is None
    assert raw_usage is None
    assert model is None


def test_parse_claude_output_extracts_model_from_model_usage():
    raw = json.dumps({"result": "Hi.", "modelUsage": {"claude-sonnet-4-6": {"outputTokens": 5}}})
    text, sid, usage, raw_usage, model = _parse_claude_output(raw)
    assert text == "Hi."
    assert model == "claude-sonnet-4-6"


def test_parse_claude_output_extracts_primary_model_from_model_usage():
    raw = json.dumps({
        "result": "Reviewed.",
        "modelUsage": {
            "claude-haiku-4-5-20251001": {"inputTokens": 4750, "outputTokens": 20},
            "claude-sonnet-4-6": {
                "inputTokens": 18,
                "outputTokens": 8263,
                "cacheReadInputTokens": 715975,
            },
        },
    })
    text, sid, usage, raw_usage, model = _parse_claude_output(raw)
    assert text == "Reviewed."
    assert model == "claude-sonnet-4-6"


def test_parse_claude_output_prefers_top_level_model_over_model_usage():
    raw = json.dumps({
        "result": "Reviewed.",
        "model": "claude-opus-4-1",
        "modelUsage": {
            "claude-haiku-4-5-20251001": {"outputTokens": 20},
            "claude-sonnet-4-6": {"outputTokens": 8263},
        },
    })
    text, sid, usage, raw_usage, model = _parse_claude_output(raw)
    assert text == "Reviewed."
    assert model == "claude-opus-4-1"


def test_parse_gemini_output_extracts_json_response():
    raw = json.dumps({
        "response": "Reviewed.\n<!-- AGENT_STATE: approved -->",
        "session_id": "gemini-session-1",
    })
    text, sid, usage, raw_usage, source = _parse_gemini_payload(raw)
    assert text == "Reviewed.\n<!-- AGENT_STATE: approved -->"
    assert sid == "gemini-session-1"
    assert usage is None
    assert raw_usage is None


def test_parse_gemini_output_falls_back_on_plain_text():
    text, sid, usage, raw_usage, source = _parse_gemini_payload("plain response")
    assert text == "plain response"
    assert sid is None
    assert usage is None
    assert raw_usage is None


def test_parse_gemini_output_falls_back_on_non_string_response():
    raw = json.dumps({"response": 42, "session_id": "gemini-session-1"})
    text, sid, usage, raw_usage, source = _parse_gemini_payload(raw)
    assert text == raw
    assert sid == "gemini-session-1"
    assert usage is None
    assert raw_usage is None


def test_parse_gemini_output_prefers_public_response_marker():
    raw = f"""Warning: True color (24-bit) support not detected.
YOLO mode is enabled. All tool calls will be automatically approved.
I will inspect the PR before giving the final answer.
Error executing tool read_file: Path not in workspace.
{PUBLIC_RESPONSE_MARKER}
## Review

No blocking findings.

<!-- AGENT_STATE: approved -->

-- Google Gemini
"""
    text, sid, usage, raw_usage, source = _parse_gemini_payload(raw)
    assert text.startswith("## Review")
    assert "True color" not in text
    assert "YOLO mode" not in text
    assert "I will inspect" not in text
    assert "Error executing tool" not in text
    assert "<!-- AGENT_STATE: approved -->" in text
    assert sid is None
    assert usage is None
    assert raw_usage is None


def test_parse_gemini_output_uses_last_public_response_marker():
    raw = f"""Gemini may mention {PUBLIC_RESPONSE_MARKER} while planning.
{PUBLIC_RESPONSE_MARKER}
intermediate draft
{PUBLIC_RESPONSE_MARKER}
Final answer.
<!-- AGENT_STATE: approved -->
"""
    text, _sid, usage, raw_usage, source = _parse_gemini_payload(raw)
    assert text == "Final answer.\n<!-- AGENT_STATE: approved -->\n"
    assert usage is None
    assert raw_usage is None


def test_parse_gemini_json_response_strips_public_response_marker():
    raw = json.dumps({
        "response": f"diagnostic\n{PUBLIC_RESPONSE_MARKER}\nReviewed.\n<!-- AGENT_STATE: approved -->",
        "session_id": "gemini-session-1",
    })
    text, sid, usage, raw_usage, source = _parse_gemini_payload(raw)
    assert text == "Reviewed.\n<!-- AGENT_STATE: approved -->"
    assert sid == "gemini-session-1"
    assert usage is None
    assert raw_usage is None


def test_parse_gemini_output_strips_cli_preamble_before_final_response():
    raw = """Warning: True color (24-bit) support not detected.
YOLO mode is enabled. All tool calls will be automatically approved.
Attempt 1 failed with status 429. Retrying with backoff... _GaxiosError: [{
  "error": {
    "code": 429,
    "message": "No capacity available for model gemini-3-flash-preview on the server"
  }
}]
I am now ready to provide my final response.

---

## Code Review

Looks good.

<!-- AGENT_STATE: approved -->

-- Google Gemini
"""
    text, sid, usage, raw_usage, source = _parse_gemini_payload(raw)
    assert text.startswith("## Code Review")
    assert "_GaxiosError" not in text
    assert "YOLO mode" not in text
    assert "<!-- AGENT_STATE: approved -->" in text
    assert sid is None
    assert usage is None
    assert raw_usage is None


def test_parse_gemini_output_strips_cli_preamble_before_plan_state_marker():
    raw = """Warning: True color (24-bit) support not detected.
YOLO mode is enabled.
I will now review the plan.

---

## Plan Review

Looks like a solid approach.

<!-- AGENT_PLAN_STATE: approved -->

-- Google Gemini
"""
    text, sid, usage, raw_usage, source = _parse_gemini_payload(raw)
    assert text.startswith("## Plan Review")
    assert "YOLO mode" not in text
    assert "<!-- AGENT_PLAN_STATE: approved -->" in text
    assert sid is None


def test_parse_gemini_output_preserves_markdown_rules_after_preamble():
    raw = """Warning: True color (24-bit) support not detected.
YOLO mode is enabled.

---

## Summary

Reviewed the change.

---

## Details

Still looks good.

<!-- AGENT_STATE: approved -->
"""
    text, sid, usage, raw_usage, source = _parse_gemini_payload(raw)
    assert text.startswith("## Summary")
    assert "YOLO mode" not in text
    assert "## Details" in text
    assert "\n---\n\n## Details" in text
    assert sid is None
    assert usage is None
    assert raw_usage is None


def test_parse_gemini_output_strips_preamble_before_clarification_marker():
    raw = """Warning: True color (24-bit) support not detected.
I need to ask a question.

---

    Which endpoint should I update?
<!-- AGENT_CLARIFY -->
"""
    text, sid, usage, raw_usage, source = _parse_gemini_payload(raw)
    assert text.startswith("    Which endpoint")
    assert "True color" not in text
    assert "<!-- AGENT_CLARIFY -->" in text
    assert sid is None
    assert usage is None
    assert raw_usage is None


def test_normalize_claude_usage_keeps_zero_cached_tokens_exact():
    usage = _normalize_claude_usage(
        {
            "input_tokens": 12,
            "cached_input_tokens": 0,
            "output_tokens": 8,
            "total_tokens": 20,
        }
    )

    assert usage is not None
    assert usage.mode == "exact"
    assert usage.cached_input_tokens == 0


def test_normalize_codex_usage_keeps_zero_reasoning_tokens():
    usage = _normalize_codex_usage(
        {
            "input_tokens": 12,
            "cached_input_tokens": 0,
            "output_tokens": 8,
            "reasoning_tokens": 0,
            "total_tokens": 20,
        }
    )

    assert usage is not None
    assert usage.mode == "exact"
    assert usage.reasoning_tokens == 0


def test_normalize_gemini_usage_keeps_zero_token_values_exact():
    usage = _normalize_gemini_usage(
        {
            "inputTokenCount": 0,
            "cachedInputTokenCount": 0,
            "outputTokenCount": 4,
            "totalTokenCount": 4,
        }
    )

    assert usage is not None
    assert usage.mode == "exact"
    assert usage.input_tokens == 0
    assert usage.cached_input_tokens == 0


def test_extract_codex_usage_reads_turn_completed_jsonl():
    usage, raw_usage = _extract_codex_usage(
        "\n".join(
            [
                json.dumps({"type": "turn.started"}),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {
                            "input_tokens": 120,
                            "cached_input_tokens": 30,
                            "output_tokens": 45,
                            "reasoning_tokens": 11,
                            "total_tokens": 206,
                        },
                    }
                ),
            ]
        )
    )

    assert usage is not None
    assert usage.mode == "exact"
    assert usage.input_tokens == 120
    assert usage.cached_input_tokens == 30
    assert usage.output_tokens == 45
    assert usage.reasoning_tokens == 11
    assert usage.total_tokens == 206
    assert raw_usage == {
        "input_tokens": 120,
        "cached_input_tokens": 30,
        "output_tokens": 45,
        "reasoning_tokens": 11,
        "total_tokens": 206,
    }


def test_claude_backend_prefers_response_file_over_message_text(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            json.dumps(
                {
                    "result": "stdout message text",
                    "session_id": "claude-session-1",
                }
            )
        ],
        public_response_outputs=["response file text"],
    )
    config = make_config(tmp_path)

    result = CLAUDE_BACKEND.run(runner, config, "Review this PR.", run_id="run-1")

    assert result.response_file_text == "response file text"
    assert result.message_text == "stdout message text"
    assert result.text == "response file text"
    assert result.session_id == "claude-session-1"


def test_gemini_backend_prefers_response_file_over_message_text(tmp_path):
    runner = FakeRunner(
        gemini_outputs=[
            json.dumps(
                {
                    "response": f"diagnostic\n{PUBLIC_RESPONSE_MARKER}\nstdout message text",
                    "session_id": "gemini-session-1",
                }
            )
        ],
        public_response_outputs=["response file text"],
    )
    config = make_config(tmp_path)

    result = GEMINI_BACKEND.run(runner, config, "Review this PR.", run_id="run-1")

    assert result.response_file_text == "response file text"
    assert result.message_text == "stdout message text"
    assert result.text == "response file text"
    assert result.session_id == "gemini-session-1"


def test_codex_backend_prefers_response_file_over_last_message_and_stdout(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            {
                "public_response": "last message text",
                "stdout": "\n".join(
                    [
                        "noisy stdout chatter",
                        json.dumps(
                            {
                                "type": "turn.completed",
                                "usage": {
                                    "input_tokens": 12,
                                    "cached_input_tokens": 3,
                                    "output_tokens": 4,
                                    "reasoning_tokens": 1,
                                    "total_tokens": 20,
                                },
                            }
                        ),
                    ]
                ),
            }
        ],
        public_response_outputs=["response file text"],
    )
    config = make_config(tmp_path)

    result = CODEX_BACKEND.run(runner, config, "Review this PR.", run_id="run-1")

    assert result.response_file_text == "response file text"
    assert result.message_text == "last message text"
    assert result.text == "response file text"
    assert result.usage is not None
    assert result.usage.total_tokens == 20


def test_codex_backend_prefers_last_message_over_stdout_without_response_file(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            {
                "public_response": "last message text",
                "stdout": "raw stdout fallback",
            }
        ]
    )
    config = make_config(tmp_path)

    result = CODEX_BACKEND.run(runner, config, "Review this PR.", run_id="run-1")

    assert result.response_file_text is None
    assert result.message_text == "last message text"
    assert result.text == "last message text"


def test_codex_backend_uses_stdout_when_files_are_absent_or_empty(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            {
                "public_response": "",
                "stdout": "raw stdout fallback",
            }
        ]
    )
    config = make_config(tmp_path)

    result = CODEX_BACKEND.run(runner, config, "Review this PR.", run_id="run-1")

    assert result.response_file_text is None
    assert result.message_text == "raw stdout fallback"
    assert result.text == "raw stdout fallback"


def test_codex_backend_dry_run_sets_message_text_without_response_file(tmp_path):
    runner = FakeRunner(codex_outputs=[{"stdout": "dry run stdout"}])
    config = make_config(tmp_path, dry_run=True)

    result = CODEX_BACKEND.run(runner, config, "Review this PR.", run_id="run-1")

    assert result.response_file_text is None
    assert result.message_text == "dry run stdout"
    assert result.text == "dry run stdout"


def _write_codex_rollout(
    codex_home: Path,
    thread_id: str,
    records: list[object],
    *,
    name_prefix: str = "rollout-2026-06-18T12-00-00",
) -> Path:
    rollout_path = (
        codex_home
        / "sessions"
        / "2026"
        / "06"
        / "18"
        / f"{name_prefix}-{thread_id}.jsonl"
    )
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    rollout_path.write_text(
        "\n".join(
            record if isinstance(record, str) else json.dumps(record)
            for record in records
        ),
        encoding="utf-8",
    )
    return rollout_path


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"payload": {"model": "gpt-5.5"}}, "gpt-5.5"),
        (
            {"turn": {"model": "gpt-5.5", "model_reasoning_effort": "medium"}},
            "gpt-5.5 (medium)",
        ),
    ],
)
def test_codex_backend_detects_model_from_rollout(tmp_path, monkeypatch, record, expected):
    thread_id = "019ed9d8-1111-7222-8333-444444444444"
    codex_home = tmp_path / "codex-home"
    _write_codex_rollout(codex_home, thread_id, [record])
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    runner = FakeRunner(
        codex_outputs=[
            {
                "public_response": "last message text",
                "stdout": json.dumps({"type": "thread.started", "thread_id": thread_id}),
            }
        ]
    )
    config = make_config(tmp_path)

    result = CODEX_BACKEND.run(runner, config, "Review this PR.", run_id="run-1")

    assert result.model_used == expected


def test_codex_backend_parses_current_turn_context_rollout_schema(tmp_path, monkeypatch):
    thread_id = "019ed9d8-1111-7222-8333-444444444444"
    codex_home = tmp_path / "codex-home"
    _write_codex_rollout(
        codex_home,
        thread_id,
        [
            {
                "timestamp": "2026-06-18T12:00:00.000Z",
                "type": "turn_context",
                "payload": {
                    "model": "gpt-5.5",
                    "effort": "high",
                },
            }
        ],
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    runner = FakeRunner(
        codex_outputs=[
            {
                "public_response": "last message text",
                "stdout": json.dumps({"type": "thread.started", "thread_id": thread_id}),
            }
        ]
    )

    result = CODEX_BACKEND.run(
        runner,
        make_config(tmp_path),
        "Review this PR.",
        run_id="run-1",
    )

    assert result.model_used == "gpt-5.5 (high)"


def test_codex_backend_accepts_mixed_case_uuid_for_rollout_lookup(tmp_path, monkeypatch):
    thread_id = "019ED9d8-1111-7222-8333-444444444444"
    codex_home = tmp_path / "codex-home"
    _write_codex_rollout(
        codex_home,
        thread_id,
        [{"payload": {"model": "gpt-5.5", "effort": "medium"}}],
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    runner = FakeRunner(
        codex_outputs=[
            {
                "public_response": "last message text",
                "stdout": json.dumps({"type": "thread.started", "thread_id": thread_id}),
            }
        ]
    )

    result = CODEX_BACKEND.run(
        runner,
        make_config(tmp_path),
        "Review this PR.",
        run_id="run-1",
    )

    assert result.model_used == "gpt-5.5 (medium)"


@pytest.mark.parametrize(
    ("thread_id", "declared_model", "expected"),
    [
        ("*", None, None),
        ("deadbeef-dead-beef-dead-beef", "gpt-5.4", "gpt-5.4"),
    ],
)
def test_codex_backend_rejects_non_uuid_thread_id_before_rollout_lookup(
    tmp_path,
    monkeypatch,
    thread_id,
    declared_model,
    expected,
):
    codex_home = tmp_path / "codex-home"
    _write_codex_rollout(
        codex_home,
        thread_id,
        [{"payload": {"model": "wrong-model"}}],
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    runner = FakeRunner(
        codex_outputs=[
            {
                "public_response": "last message text",
                "stdout": json.dumps({"type": "thread.started", "thread_id": thread_id}),
            }
        ]
    )

    result = CODEX_BACKEND.run(
        runner,
        make_config(tmp_path, codex_model=declared_model),
        "Review this PR.",
        run_id="run-1",
    )

    assert result.model_used == expected


def test_codex_backend_declared_model_takes_precedence_over_rollout(tmp_path, monkeypatch):
    thread_id = "019ed9d8-1111-7222-8333-444444444444"
    codex_home = tmp_path / "codex-home"
    _write_codex_rollout(
        codex_home,
        thread_id,
        [{"payload": {"model": "gpt-5.5", "effort": "high"}}],
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    runner = FakeRunner(
        codex_outputs=[
            {
                "public_response": "last message text",
                "stdout": json.dumps({"type": "thread.started", "thread_id": thread_id}),
            }
        ]
    )
    config = make_config(
        tmp_path,
        codex_model="gpt-5.4",
        codex_reasoning_effort="medium",
    )

    result = CODEX_BACKEND.run(runner, config, "Review this PR.", run_id="run-1")

    assert result.model_used == "gpt-5.4 (medium)"


@pytest.mark.parametrize(
    ("stdout", "records"),
    [
        ("not json", [{"payload": {"model": "gpt-5.5"}}]),
        (
            json.dumps(
                {
                    "type": "thread.started",
                    "thread_id": "019ed9d8-1111-7222-8333-444444444444",
                }
            ),
            None,
        ),
        (
            json.dumps(
                {
                    "type": "thread.started",
                    "thread_id": "019ed9d8-1111-7222-8333-444444444444",
                }
            ),
            ["not json", {"payload": {"model": ""}}, {"payload": {"model": 55}}],
        ),
    ],
)
def test_codex_backend_missing_or_invalid_rollout_model_falls_back_to_none(
    tmp_path,
    monkeypatch,
    stdout,
    records,
):
    thread_id = "019ed9d8-1111-7222-8333-444444444444"
    codex_home = tmp_path / "codex-home"
    if records is not None:
        _write_codex_rollout(codex_home, thread_id, records)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    runner = FakeRunner(
        codex_outputs=[{"public_response": "last message text", "stdout": stdout}]
    )
    config = make_config(tmp_path)

    result = CODEX_BACKEND.run(runner, config, "Review this PR.", run_id="run-1")

    assert result.model_used is None


def test_codex_backend_invalid_rollout_falls_back_to_declared_model(tmp_path, monkeypatch):
    thread_id = "019ed9d8-1111-7222-8333-444444444444"
    codex_home = tmp_path / "codex-home"
    _write_codex_rollout(codex_home, thread_id, ["not json"])
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    runner = FakeRunner(
        codex_outputs=[
            {
                "public_response": "last message text",
                "stdout": json.dumps({"type": "thread.started", "thread_id": thread_id}),
            }
        ]
    )
    config = make_config(tmp_path, codex_model="gpt-5.4")

    result = CODEX_BACKEND.run(runner, config, "Review this PR.", run_id="run-1")

    assert result.model_used == "gpt-5.4"


def test_parse_agent_state_accepts_html_marker():
    assert parse_agent_state("looks fine\n<!-- AGENT_STATE: approved -->") == "approved"
    assert parse_agent_state("needs work\n<!-- agent_state: BLOCKING -->") == "blocking"


def test_parse_agent_state_uses_last_marker_as_authoritative():
    text = """
    Quoting earlier review: <!-- AGENT_STATE: blocking -->

    Final decision:
    <!-- AGENT_STATE: approved -->
    """
    assert parse_agent_state(text) == "approved"


def test_parse_agent_state_requires_marker():
    with pytest.raises(AgentLoopError):
        parse_agent_state("LGTM")


def test_parse_plan_state_uses_last_marker_as_authoritative():
    text = """
    Quoting earlier plan review: <!-- AGENT_PLAN_STATE: blocking -->

    Final decision:
    <!-- AGENT_PLAN_STATE: approved -->
    """
    assert parse_plan_state(text) == "approved"


def test_parse_plan_state_requires_plan_marker():
    with pytest.raises(AgentLoopError):
        parse_plan_state("<!-- AGENT_STATE: approved -->")


def test_parse_signed_human_requirement_body_extracts_text_before_signature():
    body = parse_signed_human_requirement_body(
        "Please use the absolute URL.\n\n-- Human Reviewer\n\nExtra text ignored."
    )

    assert body == "Please use the absolute URL."


@pytest.mark.parametrize(
    "signature",
    [
        "-- Human Reviewer",
        "  -- Human Reviewer  ",
        "-- human reviewer",
        "-- HUMAN REVIEWER",
    ],
)
def test_parse_signed_human_requirement_body_accepts_standalone_signature_variants(signature):
    assert parse_signed_human_requirement_body(f"Required change.\n{signature}\n") == "Required change."


@pytest.mark.parametrize(
    "signature",
    [
        "-- OpenAI Codex",
        "-- Anthropic Claude",
        "-- Google Gemini",
        "-- coding-review-agent-loop",
        "Inline text -- Human Reviewer",
    ],
)
def test_parse_signed_human_requirement_body_rejects_agent_and_non_standalone_signatures(
    signature,
):
    assert parse_signed_human_requirement_body(f"Comment body.\n{signature}\n") is None


def test_parse_non_blocking_followups_extracts_bullets_only_from_section():
    review = """
    Looks good.

    ### Non-blocking follow-ups
    - Add `.agent-loop/` to `.gitignore`.
    1. Add regression coverage for stale memory refresh.
       Include multiple reviewers.

    ### Notes
    - This is not a follow-up.

    <!-- AGENT_STATE: approved -->
    -- OpenAI Codex
    """

    followups = parse_non_blocking_followups(review, reviewer="OpenAI Codex")

    assert [(item.reviewer, item.text) for item in followups] == [
        ("OpenAI Codex", "Add `.agent-loop/` to `.gitignore`."),
        (
            "OpenAI Codex",
            "Add regression coverage for stale memory refresh. Include multiple reviewers.",
        ),
    ]


def test_parse_approved_followups_extracts_same_pr_and_future_independently():
    review = """
    LGTM with cleanup.

    ### Same-PR follow-ups
    - Rename the helper for clarity.
      Keep the public behavior unchanged.

    ### Future follow-ups
    1. Add an integration fixture later.

    ### Non-blocking follow-ups
    - Legacy future item.

    <!-- AGENT_STATE: approved -->
    -- OpenAI Codex
    """

    followups = parse_approved_followups(review, reviewer="OpenAI Codex")

    assert isinstance(followups.same_pr, tuple)
    assert isinstance(followups.future, tuple)
    assert [(item.reviewer, item.text) for item in followups.same_pr] == [
        ("OpenAI Codex", "Rename the helper for clarity. Keep the public behavior unchanged.")
    ]
    assert [(item.reviewer, item.text) for item in followups.future] == [
        ("OpenAI Codex", "Add an integration fixture later."),
        ("OpenAI Codex", "Legacy future item."),
    ]


def test_parse_approved_followups_accepts_trailing_colons_on_headings():
    review = """
    LGTM with follow-ups.

    ### Same-PR follow-ups:
    - Rename the helper for clarity.

    ### Future follow-ups:
    - Add an integration fixture later.

    ### Non-blocking follow-ups:
    - Legacy future item.

    <!-- AGENT_STATE: approved -->
    -- Google Gemini
    """

    followups = parse_approved_followups(review, reviewer="Gemini")

    assert [(item.reviewer, item.text) for item in followups.same_pr] == [
        ("Gemini", "Rename the helper for clarity.")
    ]
    assert [(item.reviewer, item.text) for item in followups.future] == [
        ("Gemini", "Add an integration fixture later."),
        ("Gemini", "Legacy future item."),
    ]


@pytest.mark.parametrize(
    ("same_pr_heading", "future_heading", "legacy_heading"),
    [
        (
            "### **Same-PR follow-ups**",
            "### **Future follow-ups**",
            "### **Non-blocking follow-ups**",
        ),
        (
            "### **Same-PR follow-ups**:",
            "### **Future follow-ups.**",
            "### **Non-blocking follow-ups:**",
        ),
        (
            "### Same-PR follow-ups.",
            "### Future follow-ups.",
            "### Non-blocking follow-ups.",
        ),
    ],
)
def test_parse_approved_followups_accepts_common_markdown_heading_variants(
    same_pr_heading, future_heading, legacy_heading
):
    review = f"""
    LGTM with follow-ups.

    {same_pr_heading}
    - Rename the helper for clarity.

    {future_heading}
    - Add an integration fixture later.

    {legacy_heading}
    - Legacy future item.

    <!-- AGENT_STATE: approved -->
    -- Google Gemini
    """

    followups = parse_approved_followups(review, reviewer="Gemini")

    assert [(item.reviewer, item.text) for item in followups.same_pr] == [
        ("Gemini", "Rename the helper for clarity.")
    ]
    assert [(item.reviewer, item.text) for item in followups.future] == [
        ("Gemini", "Add an integration fixture later."),
        ("Gemini", "Legacy future item."),
    ]


def test_parse_approved_followups_stops_at_unrelated_bold_heading():
    review = """
    LGTM with follow-ups.

    ### Future follow-ups
    - Add an integration fixture later.

    ### **Notes**
    - This is not a follow-up.

    <!-- AGENT_STATE: approved -->
    -- Google Gemini
    """

    followups = parse_approved_followups(review, reviewer="Gemini")

    assert [(item.reviewer, item.text) for item in followups.future] == [
        ("Gemini", "Add an integration fixture later."),
    ]


def test_parse_approved_followups_extracts_bullets_and_prose_paragraphs():
    bullet_review = """
    Codex approves final pass.

    ### Future follow-ups
    - Refine token estimation for large review prompts.

    <!-- AGENT_STATE: approved -->
    -- OpenAI Codex
    """
    prose_review = """
    Claude approves final pass.

    ### Future follow-ups
    The `_parse_gemini_output` helper is dead production code and could be removed
    in a future cleanup.

    ### Same-PR follow-ups
    Rename the helper in this PR before merge.
    Keep the behavior unchanged.

    ### Notes
    This note is outside the follow-up sections.

    <!-- AGENT_STATE: approved -->
    -- Anthropic Claude
    """

    bullet_followups = parse_approved_followups(bullet_review, reviewer="Codex")
    prose_followups = parse_approved_followups(prose_review, reviewer="Claude")

    assert [(item.reviewer, item.text) for item in bullet_followups.future] == [
        ("Codex", "Refine token estimation for large review prompts."),
    ]
    assert [(item.reviewer, item.text) for item in prose_followups.future] == [
        (
            "Claude",
            "The `_parse_gemini_output` helper is dead production code and could be removed in a future cleanup.",
        ),
    ]
    assert [(item.reviewer, item.text) for item in prose_followups.same_pr] == [
        ("Claude", "Rename the helper in this PR before merge. Keep the behavior unchanged."),
    ]


def test_parse_approved_followups_keeps_multiline_markdown_finding_as_one_item():
    review = """
    Still blocked.

    ### Same-PR follow-ups
    #### Normalize `_plan_subject` whitespace handling

    Keep the helper from creating distinct round subjects for leading/trailing
    whitespace-only differences.

    ```python
    assert _plan_subject("x") == _plan_subject(" x ")
    ```

    The implementation should preserve the current hash format.

    ---

    #### Harden `_decode_round_metadata` exception handling

    Invalid base64 and invalid JSON should still become `AgentLoopError`
    consistently.

    ### Notes
    CI note outside the section.

    <!-- AGENT_STATE: blocking -->
    -- Anthropic Claude
    """

    followups = parse_approved_followups(review, reviewer="Anthropic Claude")

    assert [(item.reviewer, item.text) for item in followups.same_pr] == [
        (
            "Anthropic Claude",
            "\n".join(
                [
                    "#### Normalize `_plan_subject` whitespace handling",
                    "",
                    "Keep the helper from creating distinct round subjects for leading/trailing whitespace-only differences.",
                    "",
                    "```python",
                    'assert _plan_subject("x") == _plan_subject(" x ")',
                    "```",
                    "",
                    "The implementation should preserve the current hash format.",
                ]
            ),
        ),
        (
            "Anthropic Claude",
            "\n".join(
                [
                    "#### Harden `_decode_round_metadata` exception handling",
                    "",
                    "Invalid base64 and invalid JSON should still become `AgentLoopError` consistently.",
                ]
            ),
        ),
    ]


@pytest.mark.parametrize(
    "placeholder",
    [
        "None",
        "none.",
        "(none)",
        "(n/a)",
        "N/A",
        "No follow-ups",
        "No same-PR follow-ups.",
        "No future follow-ups",
    ],
)
def test_parse_approved_followups_ignores_empty_placeholders(placeholder):
    review = f"""
    LGTM.

    ### Same-PR follow-ups
    - {placeholder}

    ### Future follow-ups
    - {placeholder}

    <!-- AGENT_STATE: approved -->
    -- Google Gemini
    """

    followups = parse_approved_followups(review, reviewer="Gemini")

    assert followups.same_pr == ()
    assert followups.future == ()


def test_parse_approved_followups_ignores_prose_empty_placeholders():
    review = """
    LGTM.

    ### Same-PR follow-ups
    No same-PR follow-ups.

    ### Future follow-ups
    None

    ### Notes
    This sentence should not be captured.

    <!-- AGENT_STATE: approved -->
    -- Google Gemini
    """

    followups = parse_approved_followups(review, reviewer="Gemini")

    assert followups.same_pr == ()
    assert followups.future == ()


def test_parse_plan_review_items_extracts_structured_sections():
    review = """
    Plan looks sound with one required revision.

    ### Blocking plan issues
    - Cover how the plan avoids mixing `AGENT_STATE` and `AGENT_PLAN_STATE`.

    ### Same-plan follow-ups
    - Mention the exact docs pages to update.

    ### Future follow-ups
    - Consider a later helper to unify plan and PR disposition rendering.

    <!-- AGENT_PLAN_STATE: approved -->
    -- OpenAI Codex
    """

    items = parse_plan_review_items(review, reviewer="OpenAI Codex")

    assert [(item.reviewer, item.text) for item in items.blocking] == [
        (
            "OpenAI Codex",
            "Cover how the plan avoids mixing `AGENT_STATE` and `AGENT_PLAN_STATE`.",
        )
    ]
    assert [(item.reviewer, item.text) for item in items.same_plan] == [
        ("OpenAI Codex", "Mention the exact docs pages to update.")
    ]
    assert [(item.reviewer, item.text) for item in items.future] == [
        (
            "OpenAI Codex",
            "Consider a later helper to unify plan and PR disposition rendering.",
        )
    ]


def test_parse_plan_review_items_keeps_multiline_markdown_blocking_item_as_one_entry():
    review = """
    Plan needs one revision.

    ### Blocking plan issues
    #### Preserve multiline review items during tracking

    Do not split one reviewer-authored finding into separate ledger entries for
    paragraphs or code blocks.

    ```text
    item-2: heading
    item-3: paragraph
    ```

    ### Same-plan follow-ups
    - Mention the regression shape in the implementation plan.

    <!-- AGENT_PLAN_STATE: blocking -->
    -- OpenAI Codex
    """

    items = parse_plan_review_items(review, reviewer="OpenAI Codex")

    assert [(item.reviewer, item.text) for item in items.blocking] == [
        (
            "OpenAI Codex",
            "\n".join(
                [
                    "#### Preserve multiline review items during tracking",
                    "",
                    "Do not split one reviewer-authored finding into separate ledger entries for paragraphs or code blocks.",
                    "",
                    "```text",
                    "item-2: heading",
                    "item-3: paragraph",
                    "```",
                ]
            ),
        )
    ]


@pytest.mark.parametrize(
    "placeholder",
    [
        "None",
        "(none)",
        "(n/a)",
        "No blocking plan issues.",
        "No same-plan follow-ups",
        "No future follow-ups.",
    ],
)
def test_parse_plan_review_items_ignores_empty_placeholders(placeholder):
    review = f"""
    Looks good.

    ### Blocking plan issues
    - {placeholder}

    ### Same-plan follow-ups
    {placeholder}

    ### Future follow-ups
    - {placeholder}

    <!-- AGENT_PLAN_STATE: approved -->
    -- Google Gemini
    """

    items = parse_plan_review_items(review, reviewer="Gemini")

    assert items.blocking == ()
    assert items.same_plan == ()
    assert items.future == ()


def test_parse_plan_item_dispositions_extracts_same_plan_status():
    review = """
    Approved after the latest revision.

    ### Prior unresolved plan item dispositions
    - [item-1] resolved
    - [item-2] still blocking
    - [item-3] same-plan
    - [item-4] future follow-up: okay to track separately now

    <!-- AGENT_PLAN_STATE: approved -->
    -- OpenAI Codex
    """

    dispositions = parse_plan_item_dispositions(review, reviewer="OpenAI Codex")

    assert [(item.item_id, item.disposition, item.note) for item in dispositions] == [
        ("item-1", "resolved", None),
        ("item-2", "blocking", None),
        ("item-3", "same-plan", None),
        ("item-4", "future", "okay to track separately now"),
    ]


def test_parse_plan_item_dispositions_accepts_enriched_labels_with_trailing_arrow():
    review = """
    Approved after the latest revision.

    ### Prior unresolved plan item dispositions
    - [item-1] Same-plan follow-up from Google Gemini, round 1: keep the exact wording distinct -> same-plan: still need the mixed-reviewer case
    - [item-2] Blocking issue from OpenAI Codex, round 1: preserve public labels -> resolved

    <!-- AGENT_PLAN_STATE: blocking -->
    -- Anthropic Claude
    """

    dispositions = parse_plan_item_dispositions(review, reviewer="Anthropic Claude")

    assert [(item.item_id, item.disposition, item.note) for item in dispositions] == [
        ("item-1", "same-plan", "still need the mixed-reviewer case"),
        ("item-2", "resolved", None),
    ]


@pytest.mark.parametrize(
    "line",
    [
        "[item-1] same-plan: none",
        "[item-1] same-plan: N/A",
        "[item-1] same-plan: no same-plan follow-ups",
        "[item-1] still blocking: none",
        "[item-1] still blocking: no blocking plan issues",
        "[item-1] future follow-up: none",
        "[item-1] future follow-up: no future follow-ups",
    ],
)
def test_parse_plan_item_dispositions_rejects_contradictory_active_notes(line):
    review = (
        "Approved after the latest revision."
        + prior_plan_item_dispositions(line)
        + "\n\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="use `resolved` when nothing remains"):
        parse_plan_item_dispositions(review, reviewer="OpenAI Codex")


@pytest.mark.parametrize("line", ["[item-1] same-plan:", "[item-1] still blocking:"])
def test_parse_plan_item_dispositions_rejects_trailing_colon_syntax(line):
    review = (
        "Approved after the latest revision."
        + prior_plan_item_dispositions(line)
        + "\n\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="Invalid prior unresolved plan item disposition"):
        parse_plan_item_dispositions(review, reviewer="OpenAI Codex")


def test_parse_plan_item_dispositions_allows_resolved_none_and_substantive_same_plan():
    review = """
    Approved after the latest revision.
    """
    review += prior_plan_item_dispositions(
        "[item-1] resolved: none",
        "[item-2] same-plan: still need the mixed-reviewer case",
    )
    review += "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"

    dispositions = parse_plan_item_dispositions(review, reviewer="OpenAI Codex")

    assert [(item.item_id, item.disposition, item.note) for item in dispositions] == [
        ("item-1", "resolved", "none"),
        ("item-2", "same-plan", "still need the mixed-reviewer case"),
    ]


def test_parse_plan_item_dispositions_ignores_parenthesized_empty_placeholders():
    review = """
    Approved after the latest revision.

    ### Prior unresolved plan item dispositions
    - (none)
    - (n/a)

    <!-- AGENT_PLAN_STATE: approved -->
    -- OpenAI Codex
    """

    assert parse_plan_item_dispositions(review, reviewer="OpenAI Codex") == ()


def test_parse_plan_review_drops_future_followups_in_blocking_reviews():
    review = structured_plan_review(
        state="blocking",
        summary="Still blocked.",
        blocking_plan_issues=["Need clearer rollback coverage."],
        future_followups=["Do this later."],
    )

    result = parse_plan_review(review, reviewer="OpenAI Codex")
    assert result.items.future == ()
    assert result.items.blocking  # blocking item survives


def test_parse_plan_review_rejects_future_disposition_in_blocking_reviews():
    review = structured_plan_review(
        state="blocking",
        summary="Still blocked.",
        prior_plan_item_dispositions=[
            {"item_id": "item-1", "disposition": "future", "note": "maybe later"}
        ],
    )

    with pytest.raises(AgentLoopError, match="Blocking plan reviews may not downgrade"):
        parse_plan_review(review, reviewer="OpenAI Codex")


def test_parse_plan_review_rejects_contradictory_prior_plan_item_disposition():
    review = structured_plan_review(
        state="approved",
        summary="Looks good now.",
        prior_plan_item_dispositions=[
            {"item_id": "item-1", "disposition": "same-plan", "note": "none"}
        ],
    )

    with pytest.raises(AgentLoopError, match="empty placeholder"):
        parse_plan_review(review, reviewer="OpenAI Codex")


def test_parse_plan_review_rejects_approved_state_with_active_items():
    plan_review = structured_plan_review(
        state="approved",
        summary="Needs work.",
        same_plan_followups=["Add one more orchestration test."],
    )

    with pytest.raises(AgentLoopError, match="Approved plan reviews must be fully complete"):
        parse_plan_review(plan_review, reviewer="OpenAI Codex")

    with pytest.raises(AgentLoopError, match="AGENT_STATE"):
        parse_review(plan_review, reviewer="OpenAI Codex")


def test_parse_plan_review_rejects_approved_state_with_blocking_items():
    review = structured_plan_review(
        state="approved",
        summary="Needs work.",
        blocking_plan_issues=["Add one more orchestration test."],
    )

    with pytest.raises(AgentLoopError, match="Approved plan reviews must be fully complete"):
        parse_plan_review(review, reviewer="OpenAI Codex")


@pytest.mark.parametrize("line", ["[item-1] still blocking", "[item-1] same-plan"])
def test_parse_plan_review_rejects_approved_state_with_active_prior_disposition(line):
    item_id, disposition = ("item-1", "blocking") if "blocking" in line else ("item-1", "same-plan")
    review = structured_plan_review(
        state="approved",
        summary="Looks good now.",
        prior_plan_item_dispositions=[{"item_id": item_id, "disposition": disposition}],
    )

    with pytest.raises(AgentLoopError, match="Approved plan reviews must be fully complete"):
        parse_plan_review(review, reviewer="OpenAI Codex")


def test_validate_plan_review_response_rejects_duplicate_item_ids():
    review = structured_plan_review(
        state="blocking",
        summary="Still refining the plan.",
        prior_plan_item_dispositions=[
            {"item_id": "item-1", "disposition": "same-plan", "note": "keep the extra regression coverage"},
            {"item_id": "item-1", "disposition": "resolved"},
        ],
    )

    with pytest.raises(AgentLoopError, match="more than once: item-1"):
        _validate_plan_review_response(
            review,
            reviewer="OpenAI Codex",
            unresolved_items=(
                UnresolvedReviewItem(
                    item_id="item-1",
                    reviewer="Anthropic Claude",
                    source_round=1,
                    text="Keep the extra regression coverage.",
                    status="same-plan",
                ),
            ),
        )


def test_validate_plan_review_response_rejects_unknown_item_ids():
    review = structured_plan_review(
        state="approved",
        summary="Looks good now.",
        prior_plan_item_dispositions=[{"item_id": "item-9", "disposition": "resolved"}],
    )

    with pytest.raises(UnknownPriorItemDispositionError, match="item-9"):
        _validate_plan_review_response(
            review,
            reviewer="OpenAI Codex",
            unresolved_items=(
                UnresolvedReviewItem(
                    item_id="item-1",
                    reviewer="Anthropic Claude",
                    source_round=1,
                    text="Keep the extra regression coverage.",
                    status="same-plan",
                ),
            ),
        )


def test_validate_plan_review_response_rejects_unknown_item_with_empty_prior_ledger():
    review = structured_plan_review(
        state="approved",
        summary="Looks good now.",
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
    )

    with pytest.raises(UnknownPriorItemDispositionError) as exc_info:
        _validate_plan_review_response(
            review,
            reviewer="OpenAI Codex",
            unresolved_items=(),
        )

    assert exc_info.value.unknown_ids == ("item-1",)
    assert exc_info.value.allowed_ids == ()
    assert "Same-round findings are informational only" in exc_info.value.same_round_description


def test_unknown_prior_item_disposition_error_message_includes_ids():
    exc = UnknownPriorItemDispositionError(
        unknown_ids=("item-15",),
        allowed_ids=("item-12", "item-17"),
        same_round_description="Same-round findings are informational only.",
    )

    message = str(exc)
    assert "item-15" in message
    assert "item-12" in message
    assert "item-17" in message


def test_validate_plan_review_response_describes_same_round_unknown_item():
    review = structured_plan_review(
        state="approved",
        summary="Looks good now.",
        prior_plan_item_dispositions=[{"item_id": "item-2", "disposition": "resolved"}],
    )
    current_round_item = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="Google Gemini",
        source_round=2,
        text="Same-round finding.",
        status="same-plan",
    )

    with pytest.raises(UnknownPriorItemDispositionError) as exc_info:
        _validate_plan_review_response(
            review,
            reviewer="OpenAI Codex",
            unresolved_items=(),
            current_round_items=(current_round_item,),
        )

    assert "item-2" in exc_info.value.same_round_description


def test_validate_plan_review_response_accepts_structured_resolved_dispositions():
    review = structured_plan_review(
        state="approved",
        summary="Looks good now.",
        prior_plan_item_dispositions=[
            {"item_id": "item-1", "disposition": "resolved"},
            {"item_id": "item-2", "disposition": "resolved"},
        ],
    )

    parsed = _validate_plan_review_response(
        review,
        reviewer="OpenAI Codex",
        unresolved_items=(
            UnresolvedReviewItem(
                item_id="item-1",
                reviewer="Anthropic Claude",
                source_round=1,
                text="Keep the extra regression coverage.",
                status="same-plan",
            ),
            UnresolvedReviewItem(
                item_id="item-2",
                reviewer="Google Gemini",
                source_round=1,
                text="Clarify the fallback trigger.",
                status="blocking",
            ),
        ),
    )

    assert [(item.item_id, item.disposition) for item in parsed.dispositions] == [
        ("item-1", "resolved"),
        ("item-2", "resolved"),
    ]


def test_validate_plan_review_response_rejects_missing_structured_dispositions():
    review = structured_plan_review(
        state="approved",
        summary="Looks good now.",
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
    )

    with pytest.raises(
        AgentLoopError, match="did not evaluate all prior unresolved plan items: item-2"
    ):
        _validate_plan_review_response(
            review,
            reviewer="OpenAI Codex",
            unresolved_items=(
                UnresolvedReviewItem(
                    item_id="item-1",
                    reviewer="Anthropic Claude",
                    source_round=1,
                    text="Keep the extra regression coverage.",
                    status="same-plan",
                ),
                UnresolvedReviewItem(
                    item_id="item-2",
                    reviewer="Google Gemini",
                    source_round=1,
                    text="Clarify the fallback trigger.",
                    status="blocking",
                ),
            ),
        )


def test_parse_unresolved_item_dispositions_extracts_structured_updates():
    review = """
    LGTM.

    ### Prior unresolved item dispositions
    - [item-1] resolved
    - [item-2] still blocking
    - [item-3] same-pr
    - [item-4] future follow-up: split this into a separate PR

    <!-- AGENT_STATE: approved -->
    -- OpenAI Codex
    """

    dispositions = parse_unresolved_item_dispositions(review, reviewer="OpenAI Codex")

    assert [(item.item_id, item.disposition, item.note) for item in dispositions] == [
        ("item-1", "resolved", None),
        ("item-2", "blocking", None),
        ("item-3", "same-pr", None),
        ("item-4", "future", "split this into a separate PR"),
    ]


def test_parse_unresolved_item_dispositions_accepts_enriched_labels_with_trailing_arrow():
    review = """
    LGTM.

    ### Prior unresolved item dispositions
    - [item-1] Same-PR follow-up from Google Gemini, round 1: require source issue reference in PR body -> same-pr: keep the body reference
    - [item-2] Blocking issue from OpenAI Codex, round 1: rename the helper -> resolved

    <!-- AGENT_STATE: blocking -->
    -- Anthropic Claude
    """

    dispositions = parse_unresolved_item_dispositions(review, reviewer="Anthropic Claude")

    assert [(item.item_id, item.disposition, item.note) for item in dispositions] == [
        ("item-1", "same-pr", "keep the body reference"),
        ("item-2", "resolved", None),
    ]


@pytest.mark.parametrize(
    "line",
    [
        "[item-1] same-pr: none",
        "[item-1] same-pr: N/A",
        "[item-1] same-pr: no same-pr follow-ups",
        "[item-1] still blocking: none",
        "[item-1] still blocking: no blocking issues",
        "[item-1] future follow-up: none",
        "[item-1] future follow up: none",
        "[item-1] future follow-up: no future follow-ups",
        "[item-1] future follow-up: no follow-ups",
    ],
)
def test_parse_unresolved_item_dispositions_rejects_contradictory_active_notes(line):
    review = "LGTM." + prior_item_dispositions(line) + "\n\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"

    with pytest.raises(AgentLoopError, match="use `resolved` when nothing remains"):
        parse_unresolved_item_dispositions(review, reviewer="OpenAI Codex")


@pytest.mark.parametrize("line", ["[item-1] same-pr:", "[item-1] still blocking:"])
def test_parse_unresolved_item_dispositions_rejects_trailing_colon_syntax(line):
    review = "LGTM." + prior_item_dispositions(line) + "\n\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"

    with pytest.raises(
        AgentLoopError,
        match=r"Invalid prior unresolved item disposition.*section `### Prior unresolved item dispositions`, line 4",
    ):
        parse_unresolved_item_dispositions(review, reviewer="OpenAI Codex")


def test_parse_unresolved_item_dispositions_allows_resolved_none_and_substantive_same_pr():
    review = """
    LGTM.
    """
    review += prior_item_dispositions(
        "[item-1] resolved: none",
        "[item-2] same-pr: rename the helper before merge",
    )
    review += "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"

    dispositions = parse_unresolved_item_dispositions(review, reviewer="OpenAI Codex")

    assert [(item.item_id, item.disposition, item.note) for item in dispositions] == [
        ("item-1", "resolved", "none"),
        ("item-2", "same-pr", "rename the helper before merge"),
    ]


def test_parse_unresolved_item_dispositions_ignores_parenthesized_empty_placeholders():
    review = """
    LGTM.

    ### Prior unresolved item dispositions
    - (none)
    - (n/a)

    <!-- AGENT_STATE: approved -->
    -- OpenAI Codex
    """

    assert parse_unresolved_item_dispositions(review, reviewer="OpenAI Codex") == ()


def test_parse_unresolved_item_dispositions_ignores_non_bullet_prose():
    review = """
    LGTM.

    ### Prior unresolved item dispositions
    These are the remaining status calls.
    - [item-1] resolved
    Closing thought after the bullets.

    <!-- AGENT_STATE: approved -->
    -- OpenAI Codex
    """

    dispositions = parse_unresolved_item_dispositions(review, reviewer="OpenAI Codex")

    assert [(item.item_id, item.disposition, item.note) for item in dispositions] == [
        ("item-1", "resolved", None),
    ]


def test_validate_review_response_accepts_structured_resolved_dispositions():
    review = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[
            {"item_id": "item-1", "disposition": "resolved"},
            {"item_id": "item-2", "disposition": "resolved"},
        ],
    )

    parsed = _validate_review_response(
        review,
        reviewer="OpenAI Codex",
        unresolved_items=(
            UnresolvedReviewItem(
                item_id="item-1",
                reviewer="Anthropic Claude",
                source_round=1,
                text="Rename the helper.",
                status="same-pr",
            ),
            UnresolvedReviewItem(
                item_id="item-2",
                reviewer="Google Gemini",
                source_round=1,
                text="Keep the PR body issue reference.",
                status="blocking",
            ),
        ),
    )

    assert [(item.item_id, item.disposition) for item in parsed.dispositions] == [
        ("item-1", "resolved"),
        ("item-2", "resolved"),
    ]


def test_validate_review_response_rejects_unknown_item_with_empty_prior_ledger():
    review = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
    )

    with pytest.raises(UnknownPriorItemDispositionError) as exc_info:
        _validate_review_response(
            review,
            reviewer="OpenAI Codex",
            unresolved_items=(),
        )

    assert exc_info.value.unknown_ids == ("item-1",)
    assert exc_info.value.allowed_ids == ()
    assert "Same-round findings are informational only" in exc_info.value.same_round_description


def test_validate_review_response_rejects_ambiguous_blanket_prose():
    review = """
    LGTM.

    ### Prior unresolved item dispositions
    All prior items look resolved.

    <!-- AGENT_STATE: approved -->
    -- OpenAI Codex
    """

    with pytest.raises(AgentLoopError, match="required structured format"):
        _validate_review_response(
            review,
            reviewer="OpenAI Codex",
            unresolved_items=(
                UnresolvedReviewItem(
                    item_id="item-1",
                    reviewer="Anthropic Claude",
                    source_round=1,
                    text="Rename the helper.",
                    status="same-pr",
                ),
            ),
        )


def test_parse_review_drops_future_followups_in_blocking_reviews():
    review = """
    Still blocked.

    ### Same-PR follow-ups
    - Tighten the helper in this file.

    ### Future follow-ups
    - Do this later.

    <!-- AGENT_STATE: blocking -->
    -- OpenAI Codex
    """

    parsed = parse_review(review, reviewer="OpenAI Codex")

    assert parsed.state == "blocking"
    assert [item.text for item in parsed.followups.same_pr] == ["Tighten the helper in this file."]
    assert parsed.followups.future == ()


def test_parse_review_rejects_contradictory_prior_item_disposition():
    review = "LGTM." + prior_item_dispositions("[item-1] same-pr: none")
    review += "\n\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"

    with pytest.raises(AgentLoopError, match="use `resolved` when nothing remains"):
        parse_review(review, reviewer="OpenAI Codex")


def test_parse_review_rejects_approved_state_with_same_pr_followups():
    review = """
    LGTM.

    ### Same-PR follow-ups
    - Tighten the helper in this file.

    <!-- AGENT_STATE: approved -->
    -- OpenAI Codex
    """

    with pytest.raises(AgentLoopError, match="Approved reviews must be fully complete"):
        parse_review(review, reviewer="OpenAI Codex")


@pytest.mark.parametrize("line", ["[item-1] still blocking", "[item-1] same-pr"])
def test_parse_review_rejects_approved_state_with_active_prior_disposition(line):
    review = "LGTM." + prior_item_dispositions(line)
    review += "\n\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"

    with pytest.raises(AgentLoopError, match="Approved reviews must be fully complete"):
        parse_review(review, reviewer="OpenAI Codex")


def test_parse_review_populates_summary_from_legacy_markdown():
    review = """
    Blocking issue summary.

    ### Same-PR follow-ups
    - Rename the helper.

    <!-- AGENT_STATE: blocking -->
    -- OpenAI Codex
    """

    parsed = parse_review(review, reviewer="OpenAI Codex")

    assert parsed.summary == "Blocking issue summary."


def test_parse_review_round_trips_blocking_issues_section_without_polluting_summary():
    review = (
        "Blocking issue summary."
        + blocking_issues(
            "Cover the regression case in the PR test suite.",
            "Tighten the error assertion wording.",
        )
        + "\n\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )

    parsed = parse_review(review, reviewer="OpenAI Codex")

    assert parsed.summary == "Blocking issue summary."
    assert [item.text for item in parsed.blocking_items] == [
        "Cover the regression case in the PR test suite.",
        "Tighten the error assertion wording.",
    ]


def test_parse_review_dedupes_same_pr_items_that_duplicate_blocking_items():
    review = (
        "Blocking issue summary."
        + blocking_issues("`Add the missing share.html CSS update.`")
        + "\n\n### Same-PR follow-ups\n"
        + "- Add the missing share.html CSS update.\n"
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )

    parsed = parse_review(review, reviewer="OpenAI Codex")

    assert [item.text for item in parsed.blocking_items] == [
        "`Add the missing share.html CSS update.`"
    ]
    assert parsed.followups.same_pr == ()


def test_parse_review_prefers_same_pr_over_duplicate_future_followups():
    review = """
    Blocking on a local cleanup.

    ### Same-PR follow-ups
    - Fix the duplicated prompt wording introduced by this PR.

    ### Future follow-ups
    - `Fix the duplicated prompt wording introduced by this PR.`

    <!-- AGENT_STATE: blocking -->
    -- OpenAI Codex
    """

    parsed = parse_review(review, reviewer="OpenAI Codex")

    assert [item.text for item in parsed.followups.same_pr] == [
        "Fix the duplicated prompt wording introduced by this PR."
    ]
    assert parsed.followups.future == ()


def test_parse_review_prefers_blocking_over_duplicate_future_followups():
    review = (
        "Blocking issue summary."
        + blocking_issues("Fix the indentation in the touched `orchestrator.py` call.")
        + "\n\n### Future follow-ups\n"
        + "- fix the indentation in the touched orchestrator.py call\n"
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )

    parsed = parse_review(review, reviewer="OpenAI Codex")

    assert [item.text for item in parsed.blocking_items] == [
        "Fix the indentation in the touched `orchestrator.py` call."
    ]
    assert parsed.followups.future == ()


def test_parse_structured_pr_review_dedupes_exact_normalized_same_pr_duplicates():
    review = json.dumps(
        {
            "schema_version": 1,
            "kind": "pr_review",
            "state": "blocking",
            "summary": "Blocked.",
            "blocking_items": ["- Add the missing `share.html` CSS update."],
            "same_pr_followups": ["Add the missing share.html CSS update"],
            "future_followups": [],
            "prior_item_dispositions": [],
        }
    )
    review += "\n\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"

    parsed = parse_pr_review(review, reviewer="OpenAI Codex")

    assert [item.text for item in parsed.blocking_items] == [
        "- Add the missing `share.html` CSS update."
    ]
    assert parsed.followups.same_pr == ()


def test_parse_structured_pr_review_prefers_same_pr_over_duplicate_future_followups():
    review = structured_pr_review(
        state="blocking",
        summary="Blocked on local cleanup.",
        blocking_items=[],
        same_pr_followups=["Fix the duplicated prompt wording introduced by this PR."],
        future_followups=["fix the duplicated prompt wording introduced by this PR"],
    )

    parsed = parse_pr_review(review, reviewer="OpenAI Codex")

    assert parsed.blocking_items == ()
    assert [item.text for item in parsed.followups.same_pr] == [
        "Fix the duplicated prompt wording introduced by this PR."
    ]
    assert parsed.followups.future == ()


def test_parse_structured_pr_review_prefers_blocking_over_duplicate_future_followups():
    review = structured_pr_review(
        state="blocking",
        summary="Blocked.",
        blocking_items=["Fix the indentation in the touched `orchestrator.py` call."],
        same_pr_followups=[],
        future_followups=["fix the indentation in the touched orchestrator.py call"],
    )

    parsed = parse_pr_review(review, reviewer="OpenAI Codex")

    assert [item.text for item in parsed.blocking_items] == [
        "Fix the indentation in the touched `orchestrator.py` call."
    ]
    assert parsed.followups.same_pr == ()
    assert parsed.followups.future == ()


def test_parse_structured_pr_review_keeps_near_but_distinct_same_pr_items():
    review = json.dumps(
        {
            "schema_version": 1,
            "kind": "pr_review",
            "state": "blocking",
            "summary": "Blocked.",
            "blocking_items": ["Add the missing share.html CSS update."],
            "same_pr_followups": ["Add the missing share.html print CSS update."],
            "future_followups": [],
            "prior_item_dispositions": [],
        }
    )
    review += "\n\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"

    parsed = parse_pr_review(review, reviewer="OpenAI Codex")

    assert [item.text for item in parsed.blocking_items] == [
        "Add the missing share.html CSS update."
    ]
    assert [item.text for item in parsed.followups.same_pr] == [
        "Add the missing share.html print CSS update."
    ]


def test_pr_239_style_followup_classification_fixture():
    review = """
    PR #239-style cleanup classification.

    ### Same-PR follow-ups
    - orchestrator.py line 2100: subject=current_pr_subject is indented 4 extra spaces relative to sibling keyword arguments.
    - _repair_prior_item_ids_instruction duplicates the same-round warning/context in the repair prompt.

    ### Future follow-ups
    - _round_ledger_may_be_incomplete cross-subject branch could be bounded if comment history grows very large.

    <!-- AGENT_STATE: blocking -->
    -- OpenAI Codex
    """

    followups = parse_approved_followups(review, reviewer="OpenAI Codex")

    assert [item.text for item in followups.same_pr] == [
        "orchestrator.py line 2100: subject=current_pr_subject is indented 4 extra spaces relative to sibling keyword arguments.",
        "_repair_prior_item_ids_instruction duplicates the same-round warning/context in the repair prompt.",
    ]
    assert [item.text for item in followups.future] == [
        "_round_ledger_may_be_incomplete cross-subject branch could be bounded if comment history grows very large."
    ]


def test_legacy_plan_review_helpers_populate_summary_from_markdown():
    review = """
    Plan needs one more regression test.

    ### Same-plan follow-ups
    - Add a regression test matrix.

    <!-- AGENT_PLAN_STATE: blocking -->
    -- OpenAI Codex
    """

    items = parse_plan_review_items(review, reviewer="OpenAI Codex")

    assert _review_freeform_summary_text(review) == "Plan needs one more regression test."
    assert [item.text for item in items.same_plan] == ["Add a regression test matrix."]


def test_parse_plan_review_items_dedupes_plan_buckets_by_normalized_text():
    review = """
    Plan still needs cleanup.

    ### Blocking plan issues
    - Add `retry` coverage.

    ### Same-plan follow-ups
    - *add retry coverage!*
    - Add parser comment.

    ### Future follow-ups
    - ADD RETRY COVERAGE.
    - add `parser` comment
    - Add parser documentation later.

    <!-- AGENT_PLAN_STATE: blocking -->
    -- OpenAI Codex
    """

    items = parse_plan_review_items(review, reviewer="OpenAI Codex")

    assert [item.text for item in items.blocking] == ["Add `retry` coverage."]
    assert [item.text for item in items.same_plan] == ["Add parser comment."]
    assert [item.text for item in items.future] == ["Add parser documentation later."]


def test_parse_structured_pr_review_normalizes_v1_payload_with_footer_contract():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "Looks good after the latest fix.",
                "blocking_items": [],
                "same_pr_followups": [],
                "future_followups": ["Document cleanup for a later PR."],
                "prior_item_dispositions": [
                    {"item_id": "item-1", "disposition": "resolved"},
                    {
                        "item_id": "item-2",
                        "disposition": "future",
                        "note": "okay to split into follow-up work",
                    },
                ],
            }
        )
        + "\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex\n"
    )

    parsed = parse_structured_pr_review(payload, reviewer="OpenAI Codex")

    assert parsed is not None
    assert parsed.state == "approved"
    assert parsed.summary == "Looks good after the latest fix."
    assert parsed.blocking_items == ()
    assert [item.text for item in parsed.followups.future] == ["Document cleanup for a later PR."]
    assert [(item.item_id, item.disposition, item.note) for item in parsed.dispositions] == [
        ("item-1", "resolved", None),
        ("item-2", "future", "okay to split into follow-up work"),
    ]


def test_parse_structured_pr_review_tolerates_omitted_empty_collections():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "Looks good after the latest fix.",
                "prior_item_dispositions": [
                    {"item_id": "item-1", "disposition": "resolved"},
                ],
            }
        )
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex\n"
    )

    parsed = parse_structured_pr_review(payload, reviewer="OpenAI Codex")

    assert parsed is not None
    assert parsed.state == "approved"
    assert parsed.blocking_items == ()
    assert parsed.followups.same_pr == ()
    assert parsed.followups.future == ()
    assert [(item.item_id, item.disposition) for item in parsed.dispositions] == [
        ("item-1", "resolved")
    ]


def test_parse_structured_pr_review_strips_verdict_and_sections_from_json_summary():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "blocking",
                "summary": (
                    "**Review verdict:** blocking\n\n"
                    "Need one more regression test.\n\n"
                    "### Blocking issues\n"
                    "- Duplicate line that should not remain in the summary."
                ),
                "blocking_items": ["Need one more regression test."],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex\n"
    )

    parsed = parse_structured_pr_review(payload, reviewer="OpenAI Codex")

    assert parsed is not None
    assert parsed.summary == "Need one more regression test."


def test_parse_structured_pr_review_rejects_kind_mismatch():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_review",
                "state": "approved",
                "summary": "Wrong kind.",
                "blocking_plan_issues": [],
                "same_plan_followups": [],
                "future_followups": [],
                "prior_plan_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="kind mismatch"):
        parse_structured_pr_review(payload, reviewer="OpenAI Codex")


def test_parse_structured_pr_review_hard_fails_on_unsupported_schema_version():
    payload = (
        json.dumps(
            {
                "schema_version": 2,
                "kind": "pr_review",
                "state": "approved",
                "summary": "Wrong version.",
                "blocking_items": [],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="Unsupported structured response schema_version: 2"):
        parse_structured_pr_review(payload, reviewer="OpenAI Codex")


def test_parse_pr_review_rejects_markdown_when_no_structured_candidate_exists():
    review = "Looks good in markdown.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"

    with pytest.raises(AgentLoopError, match="required structured format"):
        parse_pr_review(review, reviewer="OpenAI Codex")


def test_legacy_parse_review_still_parses_markdown_for_historical_display():
    review = "Looks good in markdown.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"

    parsed = parse_review(review, reviewer="OpenAI Codex")

    assert parsed.state == "approved"
    assert parsed.summary == "Looks good in markdown."


def test_parse_pr_review_rejects_invalid_structured_candidate_instead_of_falling_back_to_markdown():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "Missing required arrays.",
                "blocking_items": [],
                "same_pr_followups": [],
                "future_followups": [],
            }
        )
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="missing required field"):
        parse_pr_review(payload, reviewer="OpenAI Codex")


def test_parse_structured_pr_review_strips_future_followups_in_blocking_reviews():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "blocking",
                "summary": "Still blocked.",
                "blocking_items": ["Needs one more test."],
                "same_pr_followups": [],
                "future_followups": ["Clean this up later."],
                "prior_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )

    result = parse_structured_pr_review(payload, reviewer="OpenAI Codex")
    assert result is not None
    assert result.followups.future == ()
    assert result.blocking_items  # blocking item survives


def test_parse_pr_review_rejects_structured_candidate_with_unknown_nested_keys():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "LGTM.",
                "blocking_items": [],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": [
                    {"item_id": "item-1", "disposition": "resolved", "extra": "nope"},
                ],
            }
        )
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="unknown field"):
        parse_pr_review(payload, reviewer="OpenAI Codex")


def test_parse_pr_review_rejects_structured_candidate_with_invalid_item_id():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "LGTM.",
                "blocking_items": [],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": [
                    {"item_id": "item 1", "disposition": "resolved"},
                ],
            }
        )
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="must match"):
        parse_pr_review(payload, reviewer="OpenAI Codex")


def test_parse_pr_review_requires_strict_structured_disposition_enums():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "LGTM.",
                "blocking_items": [],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": [
                    {"item_id": "item-1", "disposition": "still blocking"},
                ],
            }
        )
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="must be one of"):
        parse_pr_review(payload, reviewer="OpenAI Codex")


def test_parse_structured_pr_review_rejects_approved_blocking_items():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "Almost there.",
                "blocking_items": ["Still needs a regression test."],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="Approved reviews must be fully complete"):
        parse_structured_pr_review(payload, reviewer="OpenAI Codex")


@pytest.mark.parametrize(
    "suffix",
    [
        "\nExtra explanation after the payload.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        "\n```text\nextra block\n```\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        "\n- stray bullet\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
    ],
)
def test_parse_structured_pr_review_rejects_trailing_content_before_footer(suffix):
    payload = json.dumps(
        {
            "schema_version": 1,
            "kind": "pr_review",
            "state": "approved",
            "summary": "LGTM.",
            "blocking_items": [],
            "same_pr_followups": [],
            "future_followups": [],
            "prior_item_dispositions": [],
        }
    )

    with pytest.raises(
        AgentLoopError,
        match="place <!-- AGENT_STATE|may not include prose between|may not include trailing prose",
    ):
        parse_structured_pr_review(payload + suffix, reviewer="OpenAI Codex")


def test_parse_structured_pr_review_rejects_footer_state_mismatch():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "LGTM.",
                "blocking_items": [],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="must match the payload state"):
        parse_structured_pr_review(payload, reviewer="OpenAI Codex")


def test_parse_structured_pr_review_falls_back_when_json_is_embedded_in_markdown():
    review = """
    Here is an example:

    ```json
    {"schema_version": 1, "kind": "pr_review"}
    ```

    <!-- AGENT_STATE: approved -->
    -- OpenAI Codex
    """

    assert parse_structured_pr_review(review, reviewer="OpenAI Codex") is None
    assert parse_review(review, reviewer="OpenAI Codex").state == "approved"


def test_parse_structured_plan_review_normalizes_v1_payload():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_review",
                "state": "approved",
                "summary": "Plan looks good.",
                "blocking_plan_issues": [],
                "same_plan_followups": [],
                "future_followups": ["Consider a later cleanup pass."],
                "prior_plan_item_dispositions": [{"item_id": "item-1", "disposition": "resolved"}],
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"
    )

    parsed = parse_structured_plan_review(payload, reviewer="OpenAI Codex")

    assert parsed is not None
    assert parsed.summary == "Plan looks good."
    assert [item.text for item in parsed.items.future] == ["Consider a later cleanup pass."]
    assert [(item.item_id, item.disposition) for item in parsed.dispositions] == [
        ("item-1", "resolved")
    ]


def test_parse_structured_plan_review_tolerates_omitted_empty_collections():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_review",
                "state": "approved",
                "summary": "Plan looks good.",
                "prior_plan_item_dispositions": [{"item_id": "item-1", "disposition": "resolved"}],
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"
    )

    parsed = parse_structured_plan_review(payload, reviewer="OpenAI Codex")

    assert parsed is not None
    assert parsed.items.blocking == ()
    assert parsed.items.same_plan == ()
    assert parsed.items.future == ()
    assert [(item.item_id, item.disposition) for item in parsed.dispositions] == [
        ("item-1", "resolved")
    ]


def test_parse_structured_plan_review_strips_verdict_and_sections_from_json_summary():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_review",
                "state": "blocking",
                "summary": (
                    "**Review verdict:** blocking\n\n"
                    "Need clearer rollback coverage.\n\n"
                    "### Same-plan follow-ups\n"
                    "- Extra duplicate text."
                ),
                "blocking_plan_issues": ["Need clearer rollback coverage."],
                "same_plan_followups": [],
                "future_followups": [],
                "prior_plan_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex"
    )

    parsed = parse_structured_plan_review(payload, reviewer="OpenAI Codex")

    assert parsed is not None
    assert parsed.summary == "Need clearer rollback coverage."


def test_parse_structured_plan_review_dedupes_same_plan_against_blocking_items():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_review",
                "state": "blocking",
                "summary": "Still blocked.",
                "blocking_plan_issues": ["Add `retry` coverage."],
                "same_plan_followups": [
                    "*add retry coverage!*",
                    "Add retry coverage for timeout handling.",
                ],
                "future_followups": [],
                "prior_plan_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex"
    )

    parsed = parse_structured_plan_review(payload, reviewer="OpenAI Codex")

    assert parsed is not None
    assert [item.text for item in parsed.items.blocking] == ["Add `retry` coverage."]
    assert [item.text for item in parsed.items.same_plan] == [
        "Add retry coverage for timeout handling."
    ]


def test_parse_structured_plan_review_strips_blocking_future_followups():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_review",
                "state": "blocking",
                "summary": "Still blocked.",
                "blocking_plan_issues": ["Need clearer rollback coverage."],
                "same_plan_followups": [],
                "future_followups": ["Refactor the prompt later."],
                "prior_plan_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex"
    )

    result = parse_structured_plan_review(payload, reviewer="OpenAI Codex")
    assert result is not None
    assert result.items.future == ()
    assert result.items.blocking  # blocking item survives


def test_validate_structured_coder_followup_accepts_v1_payload():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Addressed the first item; one remains.",
                "addressed_items": ["item-1"],
                "remaining_items": ["item-2"],
                "human_requirements": {
                    "addressed_ids": ["Requirement 1"],
                    "checked_discussion_directly": False,
                },
                "tests_run": ["python -m pytest tests/test_agent_loop.py -k structured"],
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )

    parsed = validate_structured_coder_followup(payload)

    assert parsed is not None
    assert parsed.addressed_items == ("item-1",)
    assert parsed.remaining_items == ("item-2",)
    assert parsed.human_requirements.addressed_ids == ("Requirement 1",)
    assert parsed.addressed_item_notes == {}
    assert parsed.remaining_item_notes == {}


def test_validate_structured_coder_followup_accepts_optional_item_notes():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Addressed the parser; deferred the docs.",
                "addressed_items": ["item-1"],
                "remaining_items": ["item-2"],
                "addressed_item_notes": {"item-1": "Added parsing coverage."},
                "remaining_item_notes": {"item-2": "Deferred until the docs owner weighs in."},
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )

    parsed = validate_structured_coder_followup(payload)

    assert parsed is not None
    assert parsed.addressed_item_notes == {"item-1": "Added parsing coverage."}
    assert parsed.remaining_item_notes == {
        "item-2": "Deferred until the docs owner weighs in."
    }


def test_validate_structured_coder_followup_rejects_note_for_unlisted_item():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Addressed one item.",
                "addressed_items": ["item-1"],
                "remaining_items": [],
                "addressed_item_notes": {"item-2": "This note is stale."},
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="item-2.*not listed in coder_followup.addressed_items"):
        validate_structured_coder_followup(payload)


@pytest.mark.parametrize("bad_note", ["", "   ", 5, None])
def test_validate_structured_coder_followup_rejects_invalid_note_values(bad_note):
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Addressed one item.",
                "addressed_items": ["item-1"],
                "remaining_items": [],
                "addressed_item_notes": {"item-1": bad_note},
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="coder_followup.addressed_item_notes.item-1"):
        validate_structured_coder_followup(payload)


def test_validate_structured_coder_followup_returns_none_when_no_structured_candidate_exists():
    assert (
        validate_structured_coder_followup(
            "Implemented the fix.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
        )
        is None
    )


def test_validate_structured_coder_followup_rejects_unknown_keys_in_structured_candidate():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "approved",
                "summary": "Done.",
                "addressed_items": [],
                "remaining_items": [],
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": True,
                    "extra": "nope",
                },
            }
        )
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="unknown field\\(s\\): extra"):
        validate_structured_coder_followup(payload)


def test_validate_structured_coder_followup_rejects_footer_state_mismatch():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Done.",
                "addressed_items": [],
                "remaining_items": [],
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": True,
                },
            }
        )
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="footer AGENT_STATE must match the payload state"):
        validate_structured_coder_followup(payload)


def test_validate_structured_coder_followup_rejects_trailing_prose_after_footer():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Done.",
                "addressed_items": [],
                "remaining_items": [],
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": True,
                },
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex\nextra"
    )

    with pytest.raises(AgentLoopError, match="may not include trailing prose"):
        validate_structured_coder_followup(payload)


@pytest.mark.parametrize(
    ("addressed_ids", "checked_discussion_directly", "surfaced_ids", "requires_direct_discussion_ack", "message"),
    [
        (
            ("Requirement 1",),
            False,
            ("Requirement 1", "Requirement 2"),
            False,
            "did not address all surfaced signed human requirement IDs",
        ),
        (
            ("Requirement 1", "Requirement 1"),
            False,
            ("Requirement 1",),
            False,
            "listed signed human requirement IDs more than once",
        ),
        (
            ("Requirement 99",),
            False,
            ("Requirement 1",),
            False,
            "referenced unknown signed human requirement IDs",
        ),
        (
            (),
            False,
            (),
            True,
            "must acknowledge that the prompt omitted the detailed signed human requirements",
        ),
    ],
)
def test_validate_structured_human_requirements_acknowledgement_rejects_invalid_payloads(
    addressed_ids,
    checked_discussion_directly,
    surfaced_ids,
    requires_direct_discussion_ack,
    message,
):
    with pytest.raises(AgentLoopError, match=message):
        validate_structured_human_requirements_acknowledgement(
            addressed_ids,
            checked_discussion_directly=checked_discussion_directly,
            surfaced_requirement_ids=surfaced_ids,
            requires_direct_discussion_ack=requires_direct_discussion_ack,
        )


def test_validate_structured_plan_revision_accepts_v1_payload():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_revision",
                "state": "blocking",
                "summary": "Revised the plan to cover rollback testing.",
                "prior_plan_item_dispositions": [
                    {"item_id": "item-1", "disposition": "resolved", "note": "Covered in the new tests."}
                ],
                "plan_steps": ["Update protocol.py.", "Add regression tests."],
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex"
    )

    parsed = validate_structured_plan_revision(payload)

    assert parsed is not None
    assert parsed.state == "blocking"
    assert [(item.item_id, item.disposition) for item in parsed.prior_plan_item_dispositions] == [
        ("item-1", "resolved")
    ]
    assert parsed.plan_steps == ("Update protocol.py.", "Add regression tests.")


def test_validate_plan_revision_response_rejects_marker_only_markdown():
    with pytest.raises(AgentLoopError, match="Plan revision did not use the required structured format"):
        _validate_plan_revision_response(
            "Revised plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex"
        )


def test_validate_plan_revision_response_rejects_unknown_prior_disposition():
    revision = structured_plan_revision(
        prior_plan_item_dispositions=[
            {"item_id": "item-15", "disposition": "resolved"},
        ],
    )
    active_item = UnresolvedReviewItem(
        item_id="item-12",
        reviewer="Google Gemini",
        source_round=5,
        text="Active must-fix item.",
        status="blocking",
    )

    with pytest.raises(UnknownPriorItemDispositionError) as exc_info:
        _validate_plan_revision_response(revision, unresolved_items=(active_item,))

    assert exc_info.value.unknown_ids == ("item-15",)
    assert exc_info.value.allowed_ids == ("item-12",)


def test_render_canonical_plan_revision_rejects_unknown_prior_disposition_without_keyerror():
    revision = validate_structured_plan_revision(
        structured_plan_revision(
            prior_plan_item_dispositions=[
                {"item_id": "item-15", "disposition": "resolved"},
            ],
        )
    )
    assert revision is not None
    active_item = UnresolvedReviewItem(
        item_id="item-12",
        reviewer="Google Gemini",
        source_round=5,
        text="Active must-fix item.",
        status="blocking",
    )

    with pytest.raises(AgentLoopError, match="Renderer encountered unknown prior item ID") as exc_info:
        render_canonical_plan_revision(revision, prior_items=(active_item,))

    assert not isinstance(exc_info.value, KeyError)


@pytest.mark.parametrize(
    ("payload", "pattern"),
    [
        (
            {
                "schema_version": 1,
                "kind": "plan_revision",
                "state": "approved",
                "summary": "Wrong state.",
                "prior_plan_item_dispositions": [],
                "plan_steps": ["Update protocol.py."],
            },
            "plan_revision.state must be `blocking`",
        ),
        (
            {
                "schema_version": 1,
                "kind": "plan_revision",
                "state": "blocking",
                "summary": "",
                "prior_plan_item_dispositions": [],
                "plan_steps": ["Update protocol.py."],
            },
            "plan_revision.summary must be a non-empty string",
        ),
        (
            {
                "schema_version": 1,
                "kind": "plan_revision",
                "state": "blocking",
                "summary": "Missing steps.",
                "prior_plan_item_dispositions": [],
                "plan_steps": [],
            },
            "plan_revision.plan_steps must contain at least 1 item",
        ),
    ],
)
def test_validate_structured_plan_revision_rejects_invalid_payload(payload, pattern):
    footer_state = payload["state"]
    text = json.dumps(payload) + f"\n<!-- AGENT_PLAN_STATE: {footer_state} -->\n-- OpenAI Codex"

    with pytest.raises(AgentLoopError, match=pattern):
        validate_structured_plan_revision(text)


def test_extract_structured_plan_review_payload_rejects_embedded_json_markdown():
    review = """
    Here is an example:

    ```json
    {"schema_version": 1, "kind": "plan_review"}
    ```

    <!-- AGENT_PLAN_STATE: approved -->
    -- OpenAI Codex
    """

    assert _extract_structured_plan_review_payload(review) is None


@pytest.mark.parametrize(
    ("builder", "extractor"),
    [
        (lambda: structured_plan_review(reviewer="Google Gemini"), _extract_structured_plan_review_payload),
        (lambda: structured_pr_review(reviewer="Google Gemini"), _extract_structured_pr_review_payload),
        (lambda: structured_coder_followup(reviewer="Anthropic Claude"), _extract_structured_coder_followup_payload),
        (lambda: structured_plan_revision(reviewer="Anthropic Claude"), _extract_structured_plan_revision_payload),
    ],
)
def test_structured_extractors_recover_leading_public_response_marker(builder, extractor):
    text = f"\n\n{PUBLIC_RESPONSE_MARKER}\n{builder()}"

    payload = extractor(text)

    assert payload is not None


def test_response_file_marker_normalization_reports_unrecoverable_marker():
    text = f"{PUBLIC_RESPONSE_MARKER}\n### Review\nLooks good."

    normalized, status = normalize_response_file_structured_text(text)

    assert normalized == text
    assert status == "leading-public-response-marker-not-recoverable"


def test_public_response_json_prefix_strips_marker_variants():
    text = "==== AGENT_LOOP_PUBLIC_RESPONSE_BELOW ====\n" + json.dumps(
        {"error": {"status": 429, "message": "quota exceeded"}}
    )

    payload = _decode_public_response_json_prefix(text)

    assert payload == {"error": {"status": 429, "message": "quota exceeded"}}
    assert _is_transient_public_response(text)


def test_failure_category_threads_public_response_expected_kind():
    text = json.dumps(
        {
            "schema_version": 1,
            "kind": "plan_review",
            "message": "quota exceeded",
        }
    )

    assert _failure_category(text, public_response=True) == "deterministic"
    assert (
        _failure_category(
            text,
            public_response=True,
            repair_expected_kind="future_structured_kind",
        )
        == "transient"
    )


@pytest.mark.parametrize(
    "text",
    [
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_review",
                "state": "approved",
                "summary": PUBLIC_RESPONSE_MARKER,
                "blocking_plan_issues": [],
                "same_plan_followups": [],
                "future_followups": [],
                "prior_plan_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- Google Gemini",
        "Some prose first.\n"
        + PUBLIC_RESPONSE_MARKER
        + "\n"
        + structured_plan_review(reviewer="Google Gemini"),
    ],
)
def test_response_file_marker_not_stripped_inside_json_or_mid_prose(text):
    normalized, status = normalize_response_file_structured_text(text)

    assert normalized == text
    assert status is None


def test_extract_structured_plan_review_payload_rejects_footer_state_mismatch():
    payload = json.dumps(
        {
            "schema_version": 1,
            "kind": "plan_review",
            "state": "approved",
            "summary": "Plan looks good.",
            "blocking_plan_issues": [],
            "same_plan_followups": [],
            "future_followups": [],
            "prior_plan_item_dispositions": [],
        }
    )
    text = payload + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex"

    with pytest.raises(AgentLoopError, match="footer AGENT_PLAN_STATE must match"):
        _extract_structured_plan_review_payload(text)


def test_extract_structured_plan_review_payload_rejects_trailing_prose_after_signature():
    payload = json.dumps(
        {
            "schema_version": 1,
            "kind": "plan_review",
            "state": "approved",
            "summary": "Plan looks good.",
            "blocking_plan_issues": [],
            "same_plan_followups": [],
            "future_followups": [],
            "prior_plan_item_dispositions": [],
        }
    )
    text = payload + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex\nextra"

    with pytest.raises(AgentLoopError, match="trailing prose"):
        _extract_structured_plan_review_payload(text)


def test_parse_plan_review_hard_fails_after_top_level_json_prefix():
    review = (
        '{"schema_version":1,"kind":"plan_review","state":"approved","summary":"Plan looks good.",'
        '"blocking_plan_issues":[],"same_plan_followups":[],"future_followups":[]}\n'
        "<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="plan_review is missing required field"):
        parse_plan_review(review, reviewer="OpenAI Codex")


def test_extract_structured_plan_revision_payload_accepts_human_requirements_prefix():
    payload = json.dumps(
        {
            "schema_version": 1,
            "kind": "plan_revision",
            "state": "blocking",
            "summary": "Revised the plan.",
            "prior_plan_item_dispositions": [],
            "plan_steps": ["Update protocol.py."],
        }
    )
    text = (
        payload
        + "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n### Human requirements\n- Requirement 1: covered in step 1.\n"
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex"
    )

    assert _extract_structured_plan_revision_payload(text) is not None


def test_extract_structured_plan_revision_payload_rejects_bad_footer_ordering():
    payload = json.dumps(
        {
            "schema_version": 1,
            "kind": "plan_revision",
            "state": "blocking",
            "summary": "Revised the plan.",
            "prior_plan_item_dispositions": [],
            "plan_steps": ["Update protocol.py."],
        }
    )
    text = payload + "\n-- OpenAI Codex\n<!-- AGENT_PLAN_STATE: blocking -->"

    with pytest.raises(AgentLoopError, match="AGENT_PLAN_STATE"):
        _extract_structured_plan_revision_payload(text)


def test_expect_string_list_enforces_min_length():
    with pytest.raises(AgentLoopError, match="must contain at least 1 item"):
        _expect_string_list([], context="plan_revision.plan_steps", item_context="plan_revision.plan_steps", min_length=1)


def test_review_prompt_includes_prior_unresolved_items_and_disposition_instructions(tmp_path):
    config = make_config(tmp_path, approved_followups="fix-and-summarize")
    prompt = build_review_prompt(
        77,
        2,
        config,
        reviewer="codex",
        pr_metadata=PullRequestMetadata(
            number=77,
            repo="OWNER/REPO",
            title="Improve review prompt context",
            head_branch="feature/review-context",
            base_branch="main",
            head_sha="abc123",
            url="https://github.com/OWNER/REPO/pull/77",
        ),
        unresolved_items=(
            UnresolvedReviewItem(
                item_id="item-1",
                reviewer="Anthropic Claude",
                source_round=1,
                text="Needs a regression test before merge.",
                status="blocking",
            ),
            UnresolvedReviewItem(
                item_id="item-2",
                reviewer="OpenAI Codex",
                source_round=1,
                text="Rename the helper before merge.",
                status="same-pr",
            ),
        ),
    )

    assert "Prior unresolved review items from earlier rounds" in prompt
    assert "[item-1] blocking from Anthropic Claude in round 1" in prompt
    assert "[item-2] same-pr from OpenAI Codex in round 1" in prompt
    assert "### Prior unresolved item dispositions" not in prompt
    assert 'Disposition every listed\nitem in the JSON `prior_item_dispositions` array' in prompt
    assert '"resolved"' in prompt
    assert '"blocking"' in prompt
    assert '"same-pr"' in prompt
    assert '"future"' in prompt
    assert "do not add a separate prose section" in prompt
    assert "Only items listed under `Prior unresolved review items from earlier rounds`" in prompt
    assert "same-round findings from\nother reviewers appear elsewhere in the PR discussion" in prompt
    assert "Same-PR follow-ups may appear only in blocking reviews." in prompt
    assert "no blocking issues, no Same-PR follow-ups, and no" in prompt
    assert '"kind": "pr_review"' in prompt
    assert "After the JSON object, include only:" in prompt
    assert "Use this mandatory structured PR review format" in prompt
    assert "Markdown fallback" not in prompt


def test_compact_review_prompt_excludes_prose_disposition_heading(tmp_path):
    config = make_config(tmp_path, approved_followups="fix-and-summarize")
    prompt = build_review_prompt(
        77,
        2,
        config,
        reviewer="codex",
        compact_context=True,
        pr_metadata=PullRequestMetadata(
            number=77,
            repo="OWNER/REPO",
            title="Improve review prompt context",
            head_branch="feature/review-context",
            base_branch="main",
            head_sha="abc123",
            url="https://github.com/OWNER/REPO/pull/77",
        ),
        unresolved_items=(
            UnresolvedReviewItem(
                item_id="item-1",
                reviewer="Anthropic Claude",
                source_round=1,
                text="Needs a regression test before merge.",
                status="blocking",
            ),
        ),
    )
    assert "### Prior unresolved item dispositions" not in prompt
    assert 'Disposition every listed\nitem in the JSON `prior_item_dispositions` array' in prompt
    assert "do not add a separate prose section" in prompt


def test_review_prompt_indents_multiline_prior_unresolved_item_text(tmp_path):
    config = make_config(tmp_path, approved_followups="fix-and-summarize")
    prompt = build_review_prompt(
        77,
        2,
        config,
        reviewer="codex",
        pr_metadata=PullRequestMetadata(
            number=77,
            repo="OWNER/REPO",
            title="Improve review prompt context",
            head_branch="feature/review-context",
            base_branch="main",
            head_sha="abc123",
            url="https://github.com/OWNER/REPO/pull/77",
        ),
        unresolved_items=(
            UnresolvedReviewItem(
                item_id="item-1",
                reviewer="Anthropic Claude",
                source_round=1,
                text="Needs a regression test before merge.\n\nInclude the mixed-reviewer approval case.",
                status="blocking",
            ),
        ),
    )

    assert "  Needs a regression test before merge." in prompt
    assert "\n\n  Include the mixed-reviewer approval case." in prompt


def test_plan_review_prompt_includes_structured_sections_and_prior_items(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    prompt = build_plan_review_prompt(
        56,
        2,
        "Revise protocol parsing and add tests.",
        config,
        reviewer="codex",
        unresolved_items=(
            UnresolvedReviewItem(
                item_id="item-1",
                reviewer="Anthropic Claude",
                source_round=1,
                text="Define exact plan-review headings.",
                status="blocking",
            ),
            UnresolvedReviewItem(
                item_id="item-2",
                reviewer="Google Gemini",
                source_round=1,
                text="Add an orchestration carry-forward test.",
                status="same-plan",
            ),
        ),
    )

    assert "### Prior unresolved plan item dispositions" not in prompt
    assert "[item-1] blocking from Anthropic Claude in round 1" in prompt
    assert "[item-2] same-plan from Google Gemini in round 1" in prompt
    assert 'Disposition every listed item\nin the JSON `prior_plan_item_dispositions` array' in prompt
    assert '"resolved"' in prompt
    assert '"blocking"' in prompt
    assert '"same-plan"' in prompt
    assert "do not add a separate prose section" in prompt
    assert "Only items listed under `Prior unresolved plan items from earlier rounds`" in prompt
    assert "same-round findings from other\nreviewers appear elsewhere in the issue discussion" in prompt
    assert "Same-plan\nfollow-ups are small current-plan refinements" in prompt
    assert "must be incorporated before\nimplementation starts" in prompt
    assert "they may appear only in blocking plan reviews" in prompt
    assert "Future\nfollow-ups are independent later work" in prompt
    assert "A concern or\nparaphrase belongs in exactly one current-round list" in prompt
    assert "Do not duplicate or reclassify\nthe same concern across Same-plan and Future follow-up lists" in prompt
    assert "do not use structured Future\nfollow-ups" in prompt
    assert "no blocking plan issues, no Same-plan\nfollow-ups, and no carried-forward plan items left active" in prompt
    assert '"kind": "plan_review"' in prompt
    assert '"prior_plan_item_dispositions"' in prompt
    assert "Use this mandatory structured JSON response format" in prompt
    assert "markdown compatibility" not in prompt.lower()


def test_compact_plan_review_prompt_excludes_prose_disposition_heading(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    prompt = build_plan_review_prompt(
        56,
        2,
        "Fix the caching layer.",
        config,
        reviewer="codex",
        compact_context=True,
        unresolved_items=(
            UnresolvedReviewItem(
                item_id="item-1",
                reviewer="Anthropic Claude",
                source_round=1,
                text="Define exact plan-review headings.",
                status="blocking",
            ),
        ),
    )
    assert "### Prior unresolved plan item dispositions" not in prompt
    assert 'Disposition every listed item\nin the JSON `prior_plan_item_dispositions` array' in prompt
    assert "do not add a separate prose section" in prompt


def test_plan_revision_prompt_includes_unresolved_ledger_and_required_dispositions(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    prompt = build_plan_revision_prompt(
        56,
        2,
        "Previous plan text.",
        "OpenAI Codex plan review:\n\nNeeds a carry-forward ledger.",
        config,
        unresolved_items=(
            UnresolvedReviewItem(
                item_id="item-3",
                reviewer="OpenAI Codex",
                source_round=1,
                text="Track unresolved plan items across rounds.",
                status="blocking",
            ),
        ),
    )

    assert "Prior unresolved plan items from earlier rounds" in prompt
    assert "[item-3] blocking from OpenAI Codex in round 1" in prompt
    assert "### Prior plan review item dispositions" in prompt
    assert "- [item-id] same-plan:" in prompt
    assert "Use `same-plan`, never `same-pr`" in prompt
    assert '"kind": "plan_revision"' in prompt
    assert '"plan_steps"' in prompt
    assert "normalize structured plan revisions into canonical\nmarkdown for stored plan state" in prompt
    assert "<!-- HUMAN_REQUIREMENTS_ADDRESSED -->" in prompt
    assert "### Human requirements" in prompt
    assert "after the JSON object and before the `AGENT_PLAN_STATE` footer" in prompt
    assert "Use this mandatory structured JSON response format" in prompt
    assert "fall back to markdown" not in prompt.lower()


def test_non_compact_plan_revision_prompt_includes_workdir_guidance(tmp_path):
    """Non-compact build_plan_revision_prompt must include workdir guidance (issue #269)."""
    config = make_config(tmp_path)
    prompt = build_plan_revision_prompt(
        56,
        1,
        "Previous plan text.",
        "Claude plan review:\n\nBlocking issue found.",
        config,
        compact_context=False,
    )

    assert "Assigned checkout:" in prompt
    assert "AGENT_LOOP_WORKDIR" in prompt


def test_compact_plan_revision_prompt_includes_workdir_guidance(tmp_path):
    """Compact build_plan_revision_prompt also includes workdir guidance (regression guard)."""
    config = make_config(tmp_path)
    prompt = build_plan_revision_prompt(
        56,
        1,
        "Previous plan text.",
        "Claude plan review:\n\nBlocking issue found.",
        config,
        compact_context=True,
    )

    assert "Assigned checkout:" in prompt
    assert "AGENT_LOOP_WORKDIR" in prompt


def _compact_issue_context() -> IssueContext:
    return IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Add compact planning context",
        body="Original acceptance criteria: preserve requirements and known reproductions.",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(
            IssueComment(
                author="reviewer",
                created_at="2026-06-01T00:00:00Z",
                body="UNRELATED RAW PRIOR COMMENT PROSE should not be replayed in compact mode.",
            ),
            IssueComment(
                author="reviewer",
                created_at="2026-06-01T00:05:00Z",
                body="Unrelated future-only comment that was never elevated.",
            ),
        ),
        human_requirements=(
            HumanReviewRequirement(
                source_type="Issue comment",
                author="wwind123",
                created_at="2026-06-04T06:47:03Z",
                url="https://github.com/OWNER/REPO/issues/56#issuecomment-1",
                body="Compact mode must use a stable prefix and volatile tail.",
            ),
        ),
    )


def _compact_memory_context(tmp_path: Path) -> AgentMemoryContext:
    return AgentMemoryContext(
        memory_dir=tmp_path / "memory",
        current_commit="abc123",
        last_analyzed_commit="def456",
        changed_files=("src/coding_review_agent_loop/prompts.py",),
        repo_summary="Repo memory summary for compact prefix.",
        architecture_map=None,
        test_profile="Run `python -m pytest`.",
        toolchain=None,
    )


def test_config_and_cli_default_to_compact_planning_context(tmp_path):
    assert make_config(tmp_path).planning_context_mode == "compact"

    parser = build_parser()
    args = parser.parse_args([
        "task",
        "Fix the bug",
        "--repo",
        "OWNER/REPO",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
    ])
    assert config_from_args(args, FakeRunner()).planning_context_mode == "compact"

    args = parser.parse_args([
        "task",
        "Fix the bug",
        "--repo",
        "OWNER/REPO",
        "--planning-context-mode",
        "full",
    ])
    assert config_from_args(args, FakeRunner()).planning_context_mode == "full"

    with pytest.raises(AgentLoopError):
        make_config(tmp_path, planning_context_mode="invalid")
    with pytest.raises(SystemExit):
        parser.parse_args(["task", "Fix", "--planning-context-mode", "invalid"])


def test_compact_plan_review_prompt_preserves_canonical_context_and_omits_raw_prose(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    long_reproduction = "Known reproduction: " + ("run plan loop after idle gap. " * 80)
    prompt = build_plan_review_prompt(
        56,
        2,
        "Current plan payload.",
        config,
        reviewer="codex",
        memory=None,
        issue_context=_compact_issue_context(),
        unresolved_items=(
            UnresolvedReviewItem(
                item_id="item-1",
                reviewer="OpenAI Codex",
                source_round=1,
                text=long_reproduction,
                status="blocking",
            ),
        ),
        compact_context=True,
        compact_prior=CompactPriorContext(
            (
                "[item-2] resolved: Google Gemini same-plan item from round 1\n"
                "Original item text:\nPrior resolved item full text.\n"
                "Disposition updates:\n- Google Gemini: resolved: covered by tests.",
                "[item-3] future follow-up: OpenAI Codex blocking item from round 1\n"
                "Original item text:\nPrior future item text.",
            )
        ),
        compact_tail=CompactPlanTailContext(subject="subject-a", action="Review action."),
    )

    prefix, tail = prompt.split(COMPACT_PLANNING_VOLATILE_TAIL_MARKER, 1)
    assert "Original acceptance criteria: preserve requirements" in prefix
    assert "Compact mode must use a stable prefix and volatile tail." in prefix
    assert "Known reproduction: run plan loop after idle gap." in prefix
    assert "run plan loop after idle gap. run plan loop after idle gap. run plan loop after idle gap." in prefix
    assert "[item-2] resolved" in prefix
    assert "Prior resolved item full text." in prefix
    assert "[item-3] future follow-up" in prefix
    assert "UNRELATED RAW PRIOR COMMENT PROSE" not in prompt
    assert "Unrelated future-only comment that was never elevated." not in prompt
    assert "Current plan payload." in tail
    assert "Planning round: 2" in tail
    assert "subject-a" in tail


def test_compact_plan_revision_prompt_preserves_context_and_omits_raw_prose(tmp_path):
    config = make_config(tmp_path)
    prompt = build_plan_revision_prompt(
        56,
        2,
        "Previous plan payload.",
        "Blocking review payload.",
        config,
        memory=None,
        issue_context=_compact_issue_context(),
        unresolved_items=(
            UnresolvedReviewItem(
                item_id="item-4",
                reviewer="OpenAI Codex",
                source_round=1,
                text="Known reproduction: resume after subject change drops resolved item.",
                status="same-plan",
            ),
        ),
        compact_context=True,
        compact_prior=CompactPriorContext(("[item-5] resolved: stable old item text",)),
        compact_tail=CompactPlanTailContext(subject="subject-b", action="Revision action."),
    )

    prefix, tail = prompt.split(COMPACT_PLANNING_VOLATILE_TAIL_MARKER, 1)
    assert "Original acceptance criteria: preserve requirements" in prefix
    assert "Compact mode must use a stable prefix and volatile tail." in prefix
    assert "Known reproduction: resume after subject change" in prefix
    assert "[item-5] resolved: stable old item text" in prefix
    assert "UNRELATED RAW PRIOR COMMENT PROSE" not in prompt
    assert "Previous plan payload." in tail
    assert "Blocking review payload." in tail
    assert "Planning round: 2" in tail
    assert "subject-b" in tail
    assert "<!-- HUMAN_REQUIREMENTS_ADDRESSED -->" in prompt
    assert "### Human requirements" in prompt
    assert "after the JSON object and before the `AGENT_PLAN_STATE` footer" in prompt


def test_full_plan_prompt_still_includes_raw_issue_comments(tmp_path):
    config = make_config(tmp_path)
    prompt = build_plan_review_prompt(
        56,
        2,
        "Current plan payload.",
        config,
        reviewer="codex",
        issue_context=_compact_issue_context(),
    )

    assert "UNRELATED RAW PRIOR COMMENT PROSE" in prompt


def _compact_pr_issue_context() -> IssueContext:
    return IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Linked issue title",
        body="Linked issue body with durable requirements.",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(
            IssueComment(
                author="reviewer",
                created_at="2026-06-01T00:00:00Z",
                body="UNRELATED RAW PRIOR PR REVIEW HISTORY should not be replayed.",
            ),
        ),
        human_requirements=(
            HumanReviewRequirement(
                source_type="PR comment",
                author="wwind123",
                created_at="2026-06-04T10:00:00Z",
                url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                body="Compact PR mode must preserve human requirements.",
            ),
        ),
    )


def _compact_pr_metadata() -> PullRequestMetadata:
    return PullRequestMetadata(
        number=77,
        repo="OWNER/REPO",
        title="Compact PR context",
        head_branch="feature/context",
        base_branch="main",
        head_sha="abc123",
        url="https://github.com/OWNER/REPO/pull/77",
        body="PR body with author intent and scope.",
    )


def test_config_and_cli_default_to_full_pr_review_context(tmp_path):
    assert make_config(tmp_path).pr_review_context_mode == "full"

    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--pr-review-context-mode",
        "compact",
    ])
    assert config_from_args(args, FakeRunner()).pr_review_context_mode == "compact"

    with pytest.raises(AgentLoopError):
        make_config(tmp_path, pr_review_context_mode="invalid")
    with pytest.raises(SystemExit):
        parser.parse_args(["pr", "77", "--pr-review-context-mode", "invalid"])


def test_compact_pr_review_prompt_preserves_context_and_omits_raw_history(tmp_path):
    config = make_config(
        tmp_path,
        approved_followups="fix-and-summarize",
        reviewer=("codex", "gemini"),
    )
    issue_context = _compact_pr_issue_context()
    prompt = build_review_prompt(
        77,
        2,
        config,
        reviewer="codex",
        pr_metadata=_compact_pr_metadata(),
        pr_checks=None,
        memory=_compact_memory_context(tmp_path),
        issue_context=issue_context,
        human_requirements=issue_context.human_requirements,
        unresolved_items=(
            UnresolvedReviewItem(
                item_id="item-1",
                reviewer="OpenAI Codex",
                source_round=1,
                text="Active blocking issue remains important.",
                status="blocking",
            ),
            UnresolvedReviewItem(
                item_id="item-2",
                reviewer="Google Gemini",
                source_round=1,
                text="Active same-PR cleanup remains important.",
                status="same-pr",
            ),
            UnresolvedReviewItem(
                item_id="item-3",
                reviewer="OpenAI Codex",
                source_round=1,
                text="Unrelated future-only item should not stay active in compact prompt.",
                status="future",
            ),
        ),
        compact_context=True,
        compact_prior=CompactPriorContext(("[item-4] resolved: old resolved item",)),
        compact_coder_summary="Coder says the compact mode wiring is complete.",
        compact_coder_tests_run=("python -m pytest tests/test_agent_loop.py -k compact_pr",),
        compact_tail=CompactPrReviewTailContext(
            head_sha="abc123",
            round_number=2,
            action="Review compact PR context mode.",
        ),
    )

    prefix, tail = prompt.split(COMPACT_PR_REVIEW_VOLATILE_TAIL_MARKER, 1)
    assert "PR body with author intent and scope." in prefix
    assert "Linked issue body with durable requirements." in prefix
    assert "Compact PR mode must preserve human requirements." in prefix
    assert "Repo memory summary for compact prefix." in prefix
    assert "Active blocking issue remains important." in prefix
    assert "Active same-PR cleanup remains important." in prefix
    assert "Unrelated future-only item should not stay active" not in prefix
    assert "UNRELATED RAW PRIOR PR REVIEW HISTORY" not in prompt
    assert "[item-4] resolved: old resolved item" in prefix
    assert "Coder says the compact mode wiring is complete." in prefix
    assert "python -m pytest tests/test_agent_loop.py -k compact_pr" in prefix
    assert "Use Future follow-ups only for independent later work" in prefix
    assert "broader scaling or performance refinement\nfor very large histories" in prefix
    assert "indentation or\nstyle cleanup in touched code" in prefix
    assert "duplicated helper or prompt wording introduced by this PR" in prefix
    assert "Before approving, self-check every `future_followups` entry" in prefix
    assert "Review compact PR context mode." in tail
    assert "Head SHA: abc123" in tail
    assert "gh pr diff 77 --repo OWNER/REPO" in tail


def test_compact_pr_review_prompt_stable_prefix_is_byte_identical_across_rounds(tmp_path):
    config = make_config(tmp_path)
    metadata = _compact_pr_metadata()
    issue_context = _compact_pr_issue_context()
    unresolved_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Active PR item remains approval-critical.",
        status="blocking",
    )
    first = build_review_prompt(
        77,
        2,
        config,
        reviewer="codex",
        pr_metadata=metadata,
        memory=_compact_memory_context(tmp_path),
        issue_context=issue_context,
        human_requirements=issue_context.human_requirements,
        unresolved_items=(unresolved_item,),
        compact_context=True,
        compact_prior=CompactPriorContext(("[item-9] resolved: unchanged",)),
        compact_coder_summary="Stable coder summary.",
        compact_coder_tests_run=("pytest -k stable",),
        compact_tail=CompactPrReviewTailContext(head_sha="abc123", round_number=2, action="Review A."),
    )
    second = build_review_prompt(
        77,
        3,
        config,
        reviewer="codex",
        pr_metadata=metadata,
        memory=_compact_memory_context(tmp_path),
        issue_context=issue_context,
        human_requirements=issue_context.human_requirements,
        unresolved_items=(unresolved_item,),
        compact_context=True,
        compact_prior=CompactPriorContext(("[item-9] resolved: unchanged",)),
        compact_coder_summary="Stable coder summary.",
        compact_coder_tests_run=("pytest -k stable",),
        compact_tail=CompactPrReviewTailContext(head_sha="def456", round_number=3, action="Review B."),
    )

    first_prefix, first_tail = first.split(COMPACT_PR_REVIEW_VOLATILE_TAIL_MARKER, 1)
    second_prefix, second_tail = second.split(COMPACT_PR_REVIEW_VOLATILE_TAIL_MARKER, 1)
    assert first_prefix.encode() == second_prefix.encode()
    assert first_tail != second_tail
    assert "Active PR item remains approval-critical." in first_prefix
    assert "Stable coder summary." in first_prefix
    assert "PR body with author intent and scope." in first_prefix
    for volatile in ("round 2", "Head SHA: abc123", "Review A."):
        assert volatile not in first_prefix
        assert volatile in first_tail


def test_compact_review_prompt_stable_prefix_is_byte_identical_across_rounds(tmp_path):
    config = make_config(tmp_path)
    issue_context = _compact_issue_context()
    memory = _compact_memory_context(tmp_path)
    unresolved_item = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Active ledger item remains approval-critical.",
        status="blocking",
    )
    first = build_plan_review_prompt(
        56,
        2,
        "Plan tail A.",
        config,
        reviewer="codex",
        memory=memory,
        issue_context=issue_context,
        unresolved_items=(unresolved_item,),
        compact_context=True,
        compact_prior=CompactPriorContext(("[item-1] resolved: unchanged",)),
        compact_tail=CompactPlanTailContext(subject="subject-a", action="Review A."),
    )
    second = build_plan_review_prompt(
        56,
        3,
        "Plan tail B.",
        config,
        reviewer="codex",
        memory=memory,
        issue_context=issue_context,
        unresolved_items=(unresolved_item,),
        compact_context=True,
        compact_prior=CompactPriorContext(("[item-1] resolved: unchanged",)),
        compact_tail=CompactPlanTailContext(subject="subject-b", action="Review B."),
    )

    first_prefix, first_tail = first.split(COMPACT_PLANNING_VOLATILE_TAIL_MARKER, 1)
    second_prefix, second_tail = second.split(COMPACT_PLANNING_VOLATILE_TAIL_MARKER, 1)
    assert first_prefix.encode() == second_prefix.encode()
    assert first_tail != second_tail
    assert "Plan review response protocol" in first_prefix
    assert '"kind": "plan_review"' in first_prefix
    assert "Repo memory summary for compact prefix." in first_prefix
    assert "Original acceptance criteria: preserve requirements" in first_prefix
    assert "Compact mode must use a stable prefix and volatile tail." in first_prefix
    assert "Active ledger item remains approval-critical." in first_prefix
    assert "[item-1] resolved: unchanged" in first_prefix
    assert "<!-- HUMAN_REQUIREMENTS_RESOLVED -->" in first_prefix
    for volatile in ("Planning round: 2", "Plan tail A.", "subject-a", "Review A."):
        assert volatile not in first_prefix
        assert volatile in first_tail


def test_compact_revision_prompt_stable_prefix_is_byte_identical_across_rounds(tmp_path):
    config = make_config(tmp_path)
    issue_context = _compact_issue_context()
    memory = _compact_memory_context(tmp_path)
    unresolved_item = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Active revision ledger item remains approval-critical.",
        status="same-plan",
    )
    first = build_plan_revision_prompt(
        56,
        2,
        "Previous plan A.",
        "Review A.",
        config,
        memory=memory,
        issue_context=issue_context,
        unresolved_items=(unresolved_item,),
        compact_context=True,
        compact_prior=CompactPriorContext(("[item-1] resolved: unchanged",)),
        compact_tail=CompactPlanTailContext(subject="subject-a", action="Revision A."),
    )
    second = build_plan_revision_prompt(
        56,
        3,
        "Previous plan B.",
        "Review B.",
        config,
        memory=memory,
        issue_context=issue_context,
        unresolved_items=(unresolved_item,),
        compact_context=True,
        compact_prior=CompactPriorContext(("[item-1] resolved: unchanged",)),
        compact_tail=CompactPlanTailContext(subject="subject-b", action="Revision B."),
    )

    first_prefix, first_tail = first.split(COMPACT_PLANNING_VOLATILE_TAIL_MARKER, 1)
    second_prefix, second_tail = second.split(COMPACT_PLANNING_VOLATILE_TAIL_MARKER, 1)
    assert first_prefix.encode() == second_prefix.encode()
    assert first_tail != second_tail
    assert "Plan revision response protocol" in first_prefix
    assert '"kind": "plan_revision"' in first_prefix
    assert "Repo memory summary for compact prefix." in first_prefix
    assert "Original acceptance criteria: preserve requirements" in first_prefix
    assert "Compact mode must use a stable prefix and volatile tail." in first_prefix
    assert "Active revision ledger item remains approval-critical." in first_prefix
    assert "[item-1] resolved: unchanged" in first_prefix
    for volatile in ("Planning round: 2", "Previous plan A.", "Review A.", "subject-a", "Revision A."):
        assert volatile not in first_prefix
        assert volatile in first_tail


def test_structured_plan_review_preserves_human_requirements_resolution_marker():
    review = structured_plan_review(human_requirements_resolved=True)

    assert _extract_structured_plan_review_payload(review) is not None
    parsed = parse_plan_review(review, reviewer="OpenAI Codex")
    public = _render_public_plan_review_comment(
        parsed,
        reviewer="OpenAI Codex",
        prior_items=(),
        dispositions=(),
        human_requirements_resolved_flag=True,
    )

    assert "<!-- HUMAN_REQUIREMENTS_RESOLVED -->" in public
    assert public.index("<!-- HUMAN_REQUIREMENTS_RESOLVED -->") < public.index("<!-- AGENT_PLAN_STATE: approved -->")


def test_render_canonical_plan_steps_numbers_items():
    assert render_canonical_plan_steps(("Update protocol.py.", "Add tests.")) == (
        "1. Update protocol.py.\n2. Add tests."
    )


def test_render_canonical_plan_revision_and_public_comment():
    parsed = validate_structured_plan_revision(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_revision",
                "state": "blocking",
                "summary": "Revised the plan to cover rollback behavior.",
                "prior_plan_item_dispositions": [
                    {"item_id": "item-4", "disposition": "resolved", "note": "Added a resume-path step."}
                ],
                "plan_steps": ["Update protocol.py.", "Add orchestrator resume tests."],
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex"
    )
    assert parsed is not None
    prior_items = (
        UnresolvedReviewItem(
            item_id="item-4",
            reviewer="OpenAI Codex",
            source_round=2,
            text="Add a resume-path step.",
            status="blocking",
        ),
    )

    canonical = render_canonical_plan_revision(parsed, prior_items)
    public = _render_public_plan_revision_comment(
        parsed,
        prior_items=prior_items,
        raw_text='{"schema_version":1}\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex',
        agent="Codex",
    )

    assert canonical == (
        "Revised the plan to cover rollback behavior.\n\n"
        "### Prior plan item dispositions\n"
        "- [item-4] Blocking issue from OpenAI Codex, round 2: Add a resume-path step. -> "
        "resolved: Added a resume-path step.\n\n"
        "### Plan steps\n"
        "1. Update protocol.py.\n"
        "2. Add orchestrator resume tests."
    )
    assert public == (
        "## Revised plan\n\n"
        + canonical
        + "\n\n<!-- AGENT_PLAN_STATE: blocking -->\n\n-- OpenAI Codex"
    )
    assert '"kind": "plan_revision"' not in public


def test_render_structured_plan_state_to_public_markdown():
    raw = (
        json.dumps(
            {
                "kind": "plan_state",
                "summary": "Plan the renderer fix.",
                "plan_steps": ["Detect structured plan_state.", "Render public markdown."],
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- Google Antigravity"
    )
    parsed = validate_structured_plan_state(raw)

    assert parsed is not None
    public = render_public_agent_comment(
        kind="plan_state",
        parsed=parsed,
        agent="antigravity",
        model_used="Gemini 3.1 Pro (High)",
    )

    assert public == (
        "## Plan\n\n"
        "Plan the renderer fix.\n\n"
        "### Plan steps\n"
        "1. Detect structured plan_state.\n"
        "2. Render public markdown.\n\n"
        "<!-- AGENT_PLAN_STATE: approved -->\n\n"
        "-- Google Antigravity: Gemini 3.1 Pro (High)"
    )
    assert '"kind": "plan_state"' not in public


def test_render_public_coder_followup_comment():
    parsed = validate_structured_coder_followup(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Added the requested regression test.",
                "addressed_items": ["item-1", "item-2"],
                "remaining_items": [],
                "addressed_item_notes": {
                    "item-1": "Added coverage for the parser.",
                    "item-2": "Updated the helper.",
                },
                "human_requirements": {
                    "addressed_ids": ["Requirement 1"],
                    "checked_discussion_directly": False,
                },
                "tests_run": [
                    "python -m pytest tests/test_agent_loop.py -k coder_followup"
                ],
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    assert parsed is not None
    prior_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="OpenAI Codex",
            source_round=1,
            text="Add a regression test before merge.",
            status="blocking",
        ),
        UnresolvedReviewItem(
            item_id="item-2",
            reviewer="Google Gemini",
            source_round=2,
            text="Rename the shared helper.",
            status="same-pr",
        ),
    )

    rendered = _render_public_coder_followup_comment(
        parsed,
        agent="Claude",
        prior_items=prior_items,
    )

    assert rendered == (
        "## Coder follow-up\n\n"
        "Added the requested regression test.\n\n"
        "### Addressed items\n"
        "- item-1: Blocking issue from OpenAI Codex, round 1: Add a regression test before merge.\n"
        "  - Resolution: Added coverage for the parser.\n"
        "- item-2: Same-PR follow-up from Google Gemini, round 2: Rename the shared helper.\n"
        "  - Resolution: Updated the helper.\n\n"
        "### Remaining items\n"
        "- None.\n\n"
        "### Tests run\n"
        "- python -m pytest tests/test_agent_loop.py -k coder_followup\n\n"
        "<!-- AGENT_STATE: blocking -->\n\n"
        "-- Anthropic Claude"
    )
    assert "```json" not in rendered
    assert '"kind": "coder_followup"' not in rendered

    without_tests = validate_structured_coder_followup(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Still working through the review.",
                "addressed_items": [],
                "remaining_items": ["item-3"],
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": False,
                },
                "tests_run": [],
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    assert without_tests is not None
    rendered_without_tests = _render_public_coder_followup_comment(
        without_tests,
        agent="Claude",
    )
    assert "### Tests run" not in rendered_without_tests
    assert "### Addressed items\n- None." in rendered_without_tests
    assert (
        "### Remaining items\n"
        "- item-3: Item context unavailable in current round metadata.\n"
        "  - Reason: No reason provided by coder."
    ) in rendered_without_tests


def test_render_public_coder_followup_comment_expands_carried_items_with_notes_and_placeholders():
    parsed = validate_structured_coder_followup(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Fixed the blocker and deferred the follow-up.",
                "addressed_items": ["item-1"],
                "remaining_items": ["item-2"],
                "addressed_item_notes": {"item-1": "Restored the missing validation branch."},
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    assert parsed is not None
    prior_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="OpenAI Codex",
            source_round=2,
            text="  - Preserve structured coder follow-up metadata.\n\nExtra context should be summarized.",
            status="blocking",
        ),
        UnresolvedReviewItem(
            item_id="item-2",
            reviewer="Google Gemini",
            source_round=3,
            text="Move the rendering helper into a shared module.",
            status="same-pr",
        ),
    )

    rendered = _render_public_coder_followup_comment(
        parsed,
        agent="Claude",
        prior_items=prior_items,
    )

    assert (
        "- item-1: Blocking issue from OpenAI Codex, round 2: "
        "Preserve structured coder follow-up metadata."
    ) in rendered
    assert "  - Resolution: Restored the missing validation branch." in rendered
    assert (
        "- item-2: Same-PR follow-up from Google Gemini, round 3: "
        "Move the rendering helper into a shared module."
    ) in rendered
    assert "  - Reason: No reason provided by coder." in rendered


def test_render_public_coder_followup_comment_expands_pr_220_remaining_items():
    parsed = validate_structured_coder_followup(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Hardened markdown stripping; two follow-ups remain.",
                "addressed_items": ["item-3", "item-4"],
                "remaining_items": ["item-5", "item-6"],
                "remaining_item_notes": {
                    "item-5": "Deferred because URL canonicalization needs product confirmation.",
                    "item-6": "Deferred because the helper move should be isolated from this fix.",
                },
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    assert parsed is not None
    prior_items = (
        UnresolvedReviewItem(
            item_id="item-5",
            reviewer="Google Gemini",
            source_round=3,
            text=(
                "Update `server/static/index.html` and `server/static/landing.html` to use "
                "relative paths for `og:image` and `og:url` if possible."
            ),
            status="same-pr",
        ),
        UnresolvedReviewItem(
            item_id="item-6",
            reviewer="Google Gemini",
            source_round=3,
            text=(
                "Deduplicate `_strip_markdown` helper logic between `server/app.py` and "
                "`core/orchestrator.py` by moving it to `core/utils.py`."
            ),
            status="same-pr",
        ),
    )

    rendered = _render_public_coder_followup_comment(
        parsed,
        agent="Claude",
        prior_items=prior_items,
    )

    assert "- item-5: Same-PR follow-up from Google Gemini, round 3:" in rendered
    assert "relative paths" in rendered
    assert "  - Reason: Deferred because URL canonicalization needs product confirmation." in rendered
    assert "- item-6: Same-PR follow-up from Google Gemini, round 3:" in rendered
    assert "Deduplicate `_strip_markdown` helper logic" in rendered
    assert "  - Reason: Deferred because the helper move should be isolated from this fix." in rendered
    assert "\n- item-5\n" not in rendered
    assert "\n- item-6\n" not in rendered


def test_render_public_plan_review_comment_normalizes_sections():
    parsed = parse_structured_plan_review(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_review",
                "state": "blocking",
                "summary": "Still blocked on coverage.",
                "blocking_plan_issues": ["Add a resume coverage test."],
                "same_plan_followups": ["Mention canonical hashing explicitly."],
                "future_followups": [],
                "prior_plan_item_dispositions": [
                    {"item_id": "item-2", "disposition": "same-plan", "note": "Still needs one more prompt assertion."}
                ],
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
        reviewer="OpenAI Codex",
    )
    assert parsed is not None
    prior_items = (
        UnresolvedReviewItem(
            item_id="item-2",
            reviewer="Google Gemini",
            source_round=1,
            text="Mention canonical hashing explicitly.",
            status="same-plan",
        ),
    )

    rendered = _render_public_plan_review_comment(
        parsed,
        reviewer="OpenAI Codex",
        prior_items=prior_items,
        dispositions=parsed.dispositions,
    )

    assert rendered == (
        "**Review verdict:** Blocking\n\n"
        "Still blocked on coverage.\n\n"
        "### Blocking plan issues\n"
        "- Add a resume coverage test.\n\n"
        "### Same-plan follow-ups\n"
        "- Mention canonical hashing explicitly.\n\n"
        "### Prior unresolved plan item dispositions\n"
        "- [item-2] Same-plan follow-up from Google Gemini, round 1: Mention canonical hashing explicitly. -> "
        "same-plan: Still needs one more prompt assertion.\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n"
        "-- OpenAI Codex"
    )


def test_review_freeform_summary_text_strips_structured_followup_sections():
    review = """**Review verdict:** blocking

Blocking issue summary.

### Blocking issues
- needs one more assertion

### Prior unresolved item dispositions
- [item-1] still blocking: needs one more assertion

### Human requirements
- Requirement 1: addressed in the latest patch

### Same-PR follow-ups
- Rename helper

### Future follow-ups
- Document cleanup later

<!-- AGENT_STATE: blocking -->
-- OpenAI Codex
"""

    assert _review_freeform_summary_text(review) == "Blocking issue summary."


def test_validate_review_response_accepts_structured_pr_review():
    review = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "blocking",
                "summary": "Need one more regression test before merge.",
                "blocking_items": ["Add the mixed-history regression case to the suite."],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )

    parsed = _validate_review_response(review, reviewer="OpenAI Codex", unresolved_items=())

    assert parsed.summary == "Need one more regression test before merge."
    assert [item.text for item in parsed.blocking_items] == [
        "Add the mixed-history regression case to the suite."
    ]


def test_validate_coder_followup_response_accepts_structured_item_partition():
    unresolved_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="OpenAI Codex",
            source_round=1,
            text="Add a regression test.",
            status="blocking",
        ),
        UnresolvedReviewItem(
            item_id="item-2",
            reviewer="Anthropic Claude",
            source_round=1,
            text="Rename the helper.",
            status="same-pr",
        ),
        UnresolvedReviewItem(
            item_id=HUMAN_REQUIREMENTS_ACK_ITEM_ID,
            reviewer="Orchestrator",
            source_round=1,
            text="Ack missing.",
            status="blocking",
        ),
    )
    response = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Addressed the test, helper rename still pending.",
                "addressed_items": ["item-1"],
                "remaining_items": ["item-2"],
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )

    parsed = _validate_coder_followup_response(
        response,
        unresolved_items=unresolved_items,
        human_requirements=(),
    )

    assert parsed.addressed_items == ("item-1",)
    assert parsed.remaining_items == ("item-2",)


def test_validate_coder_followup_response_rejects_issue_acceptance_criteria_as_human_requirement():
    response = structured_coder_followup(
        state="blocking",
        addressed_items=["item-1"],
        remaining_items=[],
        human_requirement_ids=["Issue #221 acceptance criteria"],
        reviewer="OpenAI Codex",
    )

    with pytest.raises(AgentLoopError, match="issue acceptance criteria.*not signed human requirements"):
        _validate_coder_followup_response(
            response,
            unresolved_items=(
                UnresolvedReviewItem(
                    item_id="item-1",
                    reviewer="Anthropic Claude",
                    source_round=1,
                    text="Add a regression test.",
                    status="blocking",
                ),
            ),
            human_requirements=(),
        )


def test_validate_coder_followup_response_rejects_requirement_label_when_none_surfaced():
    response = structured_coder_followup(
        state="blocking",
        addressed_items=["item-1"],
        remaining_items=[],
        human_requirement_ids=["Requirement 1"],
        reviewer="OpenAI Codex",
    )

    with pytest.raises(AgentLoopError, match="no signed human requirements were surfaced"):
        _validate_coder_followup_response(
            response,
            unresolved_items=(
                UnresolvedReviewItem(
                    item_id="item-1",
                    reviewer="Anthropic Claude",
                    source_round=1,
                    text="Add a regression test.",
                    status="blocking",
                ),
            ),
            human_requirements=(),
        )


def test_validate_coder_followup_response_accepts_surfaced_requirement_label():
    response = structured_coder_followup(
        state="blocking",
        addressed_items=["item-1"],
        remaining_items=[],
        human_requirement_ids=["Requirement 1"],
        reviewer="OpenAI Codex",
    )

    parsed = _validate_coder_followup_response(
        response,
        unresolved_items=(
            UnresolvedReviewItem(
                item_id="item-1",
                reviewer="Anthropic Claude",
                source_round=1,
                text="Add a regression test.",
                status="blocking",
            ),
        ),
        human_requirements=(
            HumanReviewRequirement(
                source_type="PR comment",
                author="maintainer",
                created_at="2026-06-02T12:00:00Z",
                url="https://github.com/OWNER/REPO/pull/1#issuecomment-1",
                body="Add coverage for the rejected label case.",
            ),
        ),
    )

    assert parsed.human_requirements.addressed_ids == ("Requirement 1",)


def test_validate_coder_followup_response_rejects_mixed_valid_and_invalid_requirement_labels():
    response = structured_coder_followup(
        state="blocking",
        addressed_items=["item-1"],
        remaining_items=[],
        human_requirement_ids=["Requirement 1", "Issue #221 acceptance criteria"],
        reviewer="OpenAI Codex",
    )

    with pytest.raises(AgentLoopError, match="issue acceptance criteria.*not signed human requirements"):
        _validate_coder_followup_response(
            response,
            unresolved_items=(
                UnresolvedReviewItem(
                    item_id="item-1",
                    reviewer="Anthropic Claude",
                    source_round=1,
                    text="Add a regression test.",
                    status="blocking",
                ),
            ),
            human_requirements=(
                HumanReviewRequirement(
                    source_type="PR comment",
                    author="maintainer",
                    created_at="2026-06-02T12:00:00Z",
                    url="https://github.com/OWNER/REPO/pull/1#issuecomment-1",
                    body="Add coverage for the rejected label case.",
                ),
            ),
        )


@pytest.mark.parametrize(
    ("addressed_items", "remaining_items", "message"),
    [
        (["item-1"], ["item-1"], "listed unresolved reviewer item IDs more than once"),
        (["item-9"], [], "referenced unknown unresolved reviewer item IDs"),
        (["item-1"], [], "did not classify all unresolved reviewer items"),
    ],
)
def test_validate_coder_followup_response_rejects_invalid_structured_item_partition(
    addressed_items,
    remaining_items,
    message,
):
    unresolved_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="OpenAI Codex",
            source_round=1,
            text="Add a regression test.",
            status="blocking",
        ),
        UnresolvedReviewItem(
            item_id="item-2",
            reviewer="Anthropic Claude",
            source_round=1,
            text="Rename the helper.",
            status="same-pr",
        ),
    )
    response = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Status update.",
                "addressed_items": addressed_items,
                "remaining_items": remaining_items,
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match=message):
        _validate_coder_followup_response(
            response,
            unresolved_items=unresolved_items,
            human_requirements=(),
        )


def test_validate_coder_followup_response_rejects_marker_only_markdown():
    with pytest.raises(AgentLoopError, match="Coder response did not use the required structured format"):
        _validate_coder_followup_response(
            "Updated the PR.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            unresolved_items=(),
            human_requirements=(),
        )


def test_validate_coder_followup_response_requires_regular_synthetic_human_requirement_item():
    unresolved_items = (
        UnresolvedReviewItem(
            item_id="item-8",
            reviewer="Orchestrator",
            source_round=4,
            text="Reviewers approved without acknowledging signed human requirements.",
            status="blocking",
        ),
        UnresolvedReviewItem(
            item_id=HUMAN_REQUIREMENTS_ACK_ITEM_ID,
            reviewer="Orchestrator",
            source_round=4,
            text="Internal human requirements acknowledgement pseudo-item.",
            status="blocking",
        ),
    )
    response = structured_coder_followup(
        state="approved",
        addressed_items=[],
        remaining_items=[],
        reviewer="Anthropic Claude",
    )

    with pytest.raises(AgentLoopError, match="item-8"):
        _validate_coder_followup_response(
            response,
            unresolved_items=unresolved_items,
            human_requirements=(),
        )


def test_render_public_pr_review_comment_uses_normalized_sections_and_footer():
    parsed = parse_review(
        (
            "Need one more regression test."
            + blocking_issues("Exercise the structured-resume path.")
            + "\n\n### Same-PR follow-ups\n- Rename the helper for clarity."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
        ),
        reviewer="OpenAI Codex",
    )
    prior_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="Anthropic Claude",
            source_round=1,
            text="Add a regression test before merge.",
            status="blocking",
        ),
    )

    rendered = _render_public_pr_review_comment(
        parsed,
        reviewer="Codex",
        human_requirements_resolved_flag=True,
        prior_items=prior_items,
        dispositions=parsed.dispositions,
    )

    assert rendered == (
        "**Review verdict:** Blocking\n\n"
        "Need one more regression test.\n\n"
        "### Blocking issues\n"
        "- Exercise the structured-resume path.\n\n"
        "### Same-PR follow-ups\n"
        "- Rename the helper for clarity.\n\n"
        "### Prior unresolved item dispositions\n"
        "- [item-1] Blocking issue from Anthropic Claude, round 1: Add a regression test before merge. -> resolved\n\n"
        "<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n"
        "<!-- AGENT_STATE: blocking -->\n"
        "-- OpenAI Codex"
    )


def test_render_public_pr_review_comment_normalizes_markdown_and_structured_reviews_the_same():
    markdown_review = (
        "Need one more regression test."
        + blocking_issues("Exercise the structured-resume path.")
        + "\n\n### Same-PR follow-ups\n- Rename the helper for clarity."
        + prior_item_dispositions("[item-1] resolved")
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )
    structured_review = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "blocking",
                "summary": "Need one more regression test.",
                "blocking_items": ["Exercise the structured-resume path."],
                "same_pr_followups": ["Rename the helper for clarity."],
                "future_followups": [],
                "prior_item_dispositions": [{"item_id": "item-1", "disposition": "resolved"}],
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )
    prior_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="Anthropic Claude",
            source_round=1,
            text="Add a regression test before merge.",
            status="blocking",
        ),
    )

    markdown_rendered = _render_public_pr_review_comment(
        parse_review(markdown_review, reviewer="OpenAI Codex"),
        reviewer="Codex",
        human_requirements_resolved_flag=False,
        prior_items=prior_items,
        dispositions=parse_review(markdown_review, reviewer="OpenAI Codex").dispositions,
    )
    structured_parsed = parse_pr_review(structured_review, reviewer="OpenAI Codex")
    structured_rendered = _render_public_pr_review_comment(
        structured_parsed,
        reviewer="Codex",
        human_requirements_resolved_flag=False,
        prior_items=prior_items,
        dispositions=structured_parsed.dispositions,
    )

    assert markdown_rendered == structured_rendered


def test_render_public_pr_review_comment_includes_visible_approved_verdict():
    rendered = _render_public_pr_review_comment(
        parse_review(
            "Looks good to me.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            reviewer="OpenAI Codex",
        ),
        reviewer="Codex",
        human_requirements_resolved_flag=False,
        prior_items=(),
        dispositions=(),
    )

    assert rendered == (
        "**Review verdict:** Approved\n\n"
        "Looks good to me.\n\n"
        "<!-- AGENT_STATE: approved -->\n"
        "-- OpenAI Codex"
    )


def test_format_unresolved_item_label_normalizes_multiline_text_and_preserves_origin_status():
    item = UnresolvedReviewItem(
        item_id="item-7",
        reviewer="Google Gemini",
        source_round=1,
        text="  - require source issue reference in PR body  \n\nUpdate from Anthropic Claude: keep the wording compact",
        status="resolved",
        source_status="same-pr",
    )

    assert _format_unresolved_item_label(item) == (
        "Same-PR follow-up from Google Gemini, round 1: require source issue reference in PR body"
    )


def test_format_unresolved_item_label_truncates_at_fixed_limit():
    summary = "a" * (ITEM_SUMMARY_LIMIT + 20)
    item = UnresolvedReviewItem(
        item_id="item-8",
        reviewer="OpenAI Codex",
        source_round=2,
        text=summary,
        status="blocking",
    )

    label = _format_unresolved_item_label(item)

    assert label.startswith("Blocking issue from OpenAI Codex, round 2: ")
    assert label.endswith("...")
    rendered_summary = label.split(": ", 1)[1]
    assert len(rendered_summary) == ITEM_SUMMARY_LIMIT


def test_format_unresolved_item_label_special_cases_human_requirements_ack_item():
    item = UnresolvedReviewItem(
        item_id=HUMAN_REQUIREMENTS_ACK_ITEM_ID,
        reviewer="Orchestrator",
        source_round=3,
        text="Coder response missing required `### Human requirements` section.",
        status="blocking",
    )

    assert _format_unresolved_item_label(item) == (
        "Human-requirements acknowledgement item, round 3: "
        "Coder response missing required `### Human requirements` section."
    )


def test_render_public_review_comment_replaces_dispositions_without_exposing_same_round_new_items():
    body = """Still blocked.

### Same-PR follow-ups
- Keep the source issue reference in the PR body.

### Prior unresolved item dispositions
- [item-1] same-pr

<!-- AGENT_STATE: blocking -->
-- OpenAI Codex
"""
    prior_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="Google Gemini",
            source_round=1,
            text="Require source issue reference in PR body.\n\nUpdate from Anthropic Claude: keep the note compact",
            status="same-pr",
        ),
    )
    dispositions = parse_unresolved_item_dispositions(
        prior_item_dispositions("[item-1] Same-PR follow-up from Google Gemini, round 1: ignored by parser -> same-pr: keep the body reference"),
        reviewer="OpenAI Codex",
    )
    new_items = (
        UnresolvedReviewItem(
            item_id="item-2",
            reviewer="OpenAI Codex",
            source_round=2,
            text="Keep the source issue reference in the PR body.",
            status="same-pr",
        ),
    )

    rendered = _render_public_review_comment(
        body,
        review_kind="pr",
        prior_items=prior_items,
        dispositions=dispositions,
        new_items=new_items,
    )

    assert "### Same-PR follow-ups\n- Keep the source issue reference in the PR body." in rendered
    assert (
        "### Prior unresolved item dispositions\n"
        "- [item-1] Same-PR follow-up from Google Gemini, round 1: Require source issue reference in PR body. -> same-pr: keep the body reference"
    ) in rendered
    assert "### New tracked unresolved items" not in rendered
    assert "[item-2]" not in rendered
    assert rendered.rstrip().endswith("-- OpenAI Codex")


def test_render_public_review_comment_preserves_unknown_disposition_values():
    body = """Still blocked.

### Prior unresolved item dispositions
- [item-1] same-pr

<!-- AGENT_STATE: blocking -->
-- OpenAI Codex
"""
    prior_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="Google Gemini",
            source_round=1,
            text="Keep the parser and renderer aligned when new dispositions are added.",
            status="same-pr",
        ),
    )
    dispositions = (
        ReviewItemDisposition(
            item_id="item-1",
            reviewer="OpenAI Codex",
            disposition="deferred",
            note="tracked for a later parser update",
        ),
    )

    rendered = _render_public_review_comment(
        body,
        review_kind="pr",
        prior_items=prior_items,
        dispositions=dispositions,
        new_items=(),
    )

    assert (
        "### Prior unresolved item dispositions\n"
        "- [item-1] Same-PR follow-up from Google Gemini, round 1: "
        "Keep the parser and renderer aligned when new dispositions are added. "
        "-> deferred: tracked for a later parser update"
    ) in rendered


def test_apply_unresolved_item_dispositions_appends_disposition_notes_to_text():
    unresolved_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="OpenAI Codex",
            source_round=1,
            text="Needs regression coverage before merge.",
            status="blocking",
        ),
    )
    dispositions_by_item = {
        "item-1": [
            parse_unresolved_item_dispositions(
                prior_item_dispositions("[item-1] still blocking: include API error path too"),
                reviewer="Anthropic Claude",
            )[0]
        ]
    }

    updated_items, future_items = _apply_unresolved_item_dispositions(
        unresolved_items, dispositions_by_item
    )

    assert len(updated_items) == 1
    assert future_items == []
    assert updated_items[0].text == (
        "Needs regression coverage before merge.\n\n"
        "Update from Anthropic Claude: include API error path too"
    )
    assert updated_items[0].notes == ("Anthropic Claude: include API error path too",)


@pytest.mark.parametrize("terminator", ["<!-- AGENT_STATE: approved -->", "-- OpenAI Codex"])
def test_parse_non_blocking_followups_stops_at_final_markers(terminator):
    review = f"""
    Looks good.

    ### Non-blocking follow-ups
    - Add cleanup docs.
    {terminator}
    - This is outside the follow-up section.
    """

    followups = parse_non_blocking_followups(review, reviewer="OpenAI Codex")

    assert [(item.reviewer, item.text) for item in followups] == [
        ("OpenAI Codex", "Add cleanup docs."),
    ]


def test_parse_non_blocking_followups_returns_empty_without_section():
    review = "LGTM.\n- A normal bullet outside the section.\n<!-- AGENT_STATE: approved -->"

    assert parse_non_blocking_followups(review, reviewer="OpenAI Codex") == []


def test_parse_pr_number_accepts_marker_and_url():
    assert parse_pr_number("opened\n<!-- AGENT_PR: 61 -->") == 61
    assert parse_pr_number("https://github.com/OWNER/REPO/pull/62") == 62
    assert parse_pr_number("no pr here") is None


def test_parse_pr_number_uses_final_marker():
    # When multiple AGENT_PR markers are present, the last one is authoritative.
    assert parse_pr_number("<!-- AGENT_PR: 10 -->\n<!-- AGENT_PR: 20 -->") == 20
    # Same for PR URLs.
    assert (
        parse_pr_number(
            "https://github.com/OWNER/REPO/pull/1 and https://github.com/OWNER/REPO/pull/2"
        )
        == 2
    )
    # Marker takes precedence over URL when both present (marker checked first).
    assert parse_pr_number("https://github.com/OWNER/REPO/pull/5\n<!-- AGENT_PR: 7 -->") == 7


def test_issue_loop_creates_pr_then_alternates_until_codex_approval(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->",
            "Fixed review.\n<!-- AGENT_STATE: blocking -->",
        ],
        codex_outputs=[
            "Finding: bug remains.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    command_names = [cmd[:2] for cmd, _cwd in runner.commands]
    assert ["claude", "--print"] in command_names
    assert ["codex", "exec"] in command_names
    assert len(runner.comments) == 4
    assert runner.comments[-1].startswith("**Review verdict:** Approved\n\nLGTM.")
    assert list((tmp_path / "logs").glob("*-claude.log"))
    assert list((tmp_path / "logs").glob("*-codex.log"))
    assert (tmp_path / "logs" / ".gitignore").read_text(encoding="utf-8") == "*\n!.gitignore\n"


def test_issue_loop_syncs_coder_base_after_memory_before_coder(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->",
        ],
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)
    config.agent_memory_dir.mkdir(parents=True)
    (config.agent_memory_dir / "last-analyzed-commit").write_text("base123\n", encoding="utf-8")

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    commands = runner.commands
    issue_context_index = command_index(commands, ["gh", "issue", "view"])
    memory_index = command_index(commands, ["git", "diff", "--name-only"])
    fetch_index = command_index(commands, ["git", "fetch", "origin"])
    switch_index = command_index(commands, ["git", "switch", "main"])
    pull_index = command_index(commands, ["git", "pull", "--ff-only", "origin", "main"])
    coder_index = command_index(commands, ["claude", "--print"])

    assert issue_context_index < memory_index < fetch_index < switch_index < pull_index < coder_index


def test_issue_loop_includes_issue_comments_in_coder_and_review_prompts(tmp_path):
    runner = FakeRunner(
        issue_payload={
            "number": 56,
            "state": "open",
            "is_pr": False,
            "url": "https://github.com/OWNER/REPO/issues/56",
            "title": "Support issue comments",
            "body": "Original request.",
        },
        issue_comments=[
            {
                "author": {"login": "second-user"},
                "createdAt": "2026-05-17T10:00:00Z",
                "body": "Later comment should come second.",
            },
            {
                "author": {"login": "first-user"},
                "createdAt": "2026-05-17T09:00:00Z",
                "body": "Earlier comment refines the request.",
            },
        ],
        claude_outputs=[
            "Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->",
        ],
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    claude_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    codex_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    for prompt in (claude_prompt, codex_prompt):
        assert "Issue context from GitHub" in prompt
        assert "Later comments may refine or supersede the original issue body" in prompt
        assert "GitHub issue #56" in prompt
        assert "Title:\nSupport issue comments" in prompt
        assert "Body:\nOriginal request." in prompt
        assert "Comments, oldest to newest:" in prompt
        assert prompt.index("Comment by first-user at 2026-05-17T09:00:00Z") < prompt.index(
            "Comment by second-user at 2026-05-17T10:00:00Z"
        )
        assert "Earlier comment refines the request." in prompt
        assert "Later comment should come second." in prompt
    assert "include `Fixes #56` or another direct reference to issue #56" in claude_prompt


def test_format_issue_context_truncates_oversized_newest_comment():
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Support issue comments",
        body="Original request.",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(
            IssueComment(
                author="first-user",
                created_at="2026-05-17T09:00:00Z",
                body="Older detail should not be kept instead of the newest comment.",
            ),
            IssueComment(
                author="second-user",
                created_at="2026-05-17T10:00:00Z",
                body="Newest detail. " + ("x" * 1000),
            ),
        ),
    )

    text = format_issue_context(issue_context, max_chars=700)

    assert len(text) <= 700
    assert "Older comments omitted: 1 comment(s)" in text
    assert "Comment by second-user at 2026-05-17T10:00:00Z" in text
    assert "Newest detail." in text
    assert "[Newest comment truncated to keep this prompt bounded.]" in text
    assert "Older detail should not be kept instead of the newest comment." not in text


def test_get_issue_context_parses_signed_issue_body_and_comments(tmp_path):
    runner = FakeRunner(
        issue_payload={
            "number": 56,
            "title": "Support signed issue requirements",
            "body": "Use the stable API path.\n\n-- Human Reviewer",
            "url": "https://github.com/OWNER/REPO/issues/56",
            "author": {"login": "issue-author"},
            "createdAt": "2026-05-17T08:00:00Z",
        },
        issue_comments=[
            {
                "author": {"login": "maintainer"},
                "createdAt": "2026-05-17T09:00:00Z",
                "url": "https://github.com/OWNER/REPO/issues/56#issuecomment-1",
                "body": "Unsigned discussion remains normal context.",
            },
            {
                "author": {"login": "lead"},
                "createdAt": "2026-05-17T10:00:00Z",
                "url": "https://github.com/OWNER/REPO/issues/56#issuecomment-2",
                "body": "Add a regression test.\n\n-- Human Reviewer",
            },
        ],
    )
    config = make_config(tmp_path)

    issue_context = get_issue_context(runner, config=config, issue_number=56)

    assert [item.source_type for item in issue_context.human_requirements] == [
        "Issue body",
        "Issue comment",
    ]
    assert [item.author for item in issue_context.human_requirements] == ["issue-author", "lead"]
    assert [item.created_at for item in issue_context.human_requirements] == [
        "2026-05-17T08:00:00Z",
        "2026-05-17T10:00:00Z",
    ]
    assert issue_context.human_requirements[0].body == "Use the stable API path."
    assert issue_context.human_requirements[1].body == "Add a regression test."
    assert issue_context.comments[0].body == "Unsigned discussion remains normal context."


def test_format_human_requirements_uses_distinct_high_priority_section():
    text = format_human_requirements(
        (
            HumanReviewRequirement(
                source_type="PR comment",
                author="reviewer",
                created_at="2026-05-18T10:00:00Z",
                url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                body="Please use the absolute URL.",
            ),
        )
    )

    assert text.startswith("Signed Human Reviewer Requirements")
    assert "high-priority PR requirements" in text
    assert "latest human instruction wins" in text
    assert "- Source: PR comment" in text
    assert "- Author: reviewer" in text
    assert "Please use the absolute URL." in text


def test_format_human_requirements_supports_issue_specific_wording_and_fallback():
    text = format_human_requirements(
        (
            HumanReviewRequirement(
                source_type="Issue body",
                author="maintainer",
                created_at="2026-05-18T10:00:00Z",
                url="https://github.com/OWNER/REPO/issues/56",
                body="Keep the current CLI flag.",
            ),
        ),
        max_chars=120,
        requirement_scope="planning requirements",
        full_omission_fallback="Fetch the issue discussion directly before finalizing the plan.",
    )

    assert "high-priority planning requirements" in text
    assert "Fetch the issue discussion directly before finalizing the plan." in text


def test_format_human_requirements_preserves_entry_spacing_when_truncated():
    requirements = (
        HumanReviewRequirement(
            source_type="PR comment",
            author="reviewer",
            created_at="2026-05-18T10:00:00Z",
            url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
            body="Oldest requirement.",
        ),
        HumanReviewRequirement(
            source_type="PR comment",
            author="reviewer",
            created_at="2026-05-18T11:00:00Z",
            url="https://github.com/OWNER/REPO/pull/77#issuecomment-2",
            body="Middle requirement.",
        ),
        HumanReviewRequirement(
            source_type="PR comment",
            author="reviewer",
            created_at="2026-05-18T12:00:00Z",
            url="https://github.com/OWNER/REPO/pull/77#issuecomment-3",
            body="Newest requirement.",
        ),
    )
    full_text = format_human_requirements(requirements)

    text = format_human_requirements(requirements, max_chars=len(full_text) - 1)

    assert "Older signed human requirement(s) omitted: 1." in text
    assert "Oldest requirement." not in text
    assert "Middle requirement.\n\nRequirement 3:" in text
    assert "Newest requirement." in text


def test_render_coder_human_requirements_prompt_context_tracks_surfaced_ids_after_truncation():
    requirements = (
        HumanReviewRequirement(
            source_type="PR comment",
            author="reviewer",
            created_at="2026-05-18T10:00:00Z",
            url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
            body="Oldest requirement.",
        ),
        HumanReviewRequirement(
            source_type="PR comment",
            author="reviewer",
            created_at="2026-05-18T11:00:00Z",
            url="https://github.com/OWNER/REPO/pull/77#issuecomment-2",
            body="Middle requirement.",
        ),
        HumanReviewRequirement(
            source_type="PR comment",
            author="reviewer",
            created_at="2026-05-18T12:00:00Z",
            url="https://github.com/OWNER/REPO/pull/77#issuecomment-3",
            body="Newest requirement.",
        ),
    )
    full_text = format_human_requirements(requirements)

    context = render_coder_human_requirements_prompt_context(
        requirements,
        max_chars=len(full_text) - 1,
    )

    assert context.block.endswith("\n")
    assert "Older signed human requirement(s) omitted: 1." in context.block
    assert context.surfaced_requirement_ids == ("Requirement 2", "Requirement 3")
    assert context.requires_direct_discussion_ack is False


def test_render_coder_human_requirements_prompt_context_handles_full_omission_fallback():
    context = render_coder_human_requirements_prompt_context(
        (
            HumanReviewRequirement(
                source_type="PR comment",
                author="reviewer",
                created_at="2026-05-18T10:00:00Z",
                url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                body="Please use the absolute URL.",
            ),
        ),
        max_chars=120,
    )

    assert "All 1 signed human requirement(s) were omitted" in context.block
    assert context.surfaced_requirement_ids == ()
    assert context.requires_direct_discussion_ack is True


@pytest.mark.parametrize(
    ("builder_name", "expected_scope", "expected_guidance"),
    [
        ("issue", "high-priority implementation requirements", "how you addressed that item"),
        ("issue_plan", "high-priority planning requirements", "how the plan covers that item"),
        ("plan_revision", "high-priority planning requirements", "how the revised plan covers that item"),
        (
            "issue_implementation",
            "high-priority implementation requirements",
            "how you addressed that item",
        ),
    ],
)
def test_issue_and_plan_prompts_surface_signed_human_requirements_before_issue_context(
    tmp_path,
    builder_name,
    expected_scope,
    expected_guidance,
):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Support issue comments",
        body="Original request.",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(
            IssueComment(
                author="commenter",
                created_at="2026-05-17T10:00:00Z",
                body="General issue context.",
            ),
        ),
        human_requirements=(
            HumanReviewRequirement(
                source_type="Issue comment",
                author="maintainer",
                created_at="2026-05-17T11:00:00Z",
                url="https://github.com/OWNER/REPO/issues/56#issuecomment-1",
                body="Preserve backward compatibility.",
            ),
        ),
    )
    if builder_name == "issue":
        prompt = build_issue_prompt(56, config, issue_context=issue_context)
    elif builder_name == "issue_plan":
        prompt = build_issue_plan_prompt(56, config, issue_context=issue_context)
    elif builder_name == "plan_revision":
        prompt = build_plan_revision_prompt(
            56,
            2,
            "Old plan.",
            "Blocking review.",
            config,
            issue_context=issue_context,
        )
    else:
        prompt = build_issue_implementation_prompt(
            56,
            "Approved plan.",
            config,
            issue_context=issue_context,
        )

    assert "Signed Human Reviewer Requirements" in prompt
    assert expected_scope in prompt
    assert expected_guidance in prompt
    assert prompt.index("Signed Human Reviewer Requirements") < prompt.index("Issue context from GitHub")


def test_followup_prompt_with_no_human_requirements_guides_empty_addressed_ids(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    prompt = build_followup_prompt(
        222,
        2,
        "item-1: Add a regression test.",
        config,
        human_requirements=(),
    )

    assert '"human_requirements": {' in prompt
    assert '"addressed_ids": []' in prompt
    assert '"addressed_ids": ["Requirement 1"]' not in prompt
    assert "No signed human requirements are surfaced in this prompt" in prompt
    assert "issue acceptance criteria" in prompt
    assert "reviewer item IDs" in prompt


def test_plan_review_prompt_surfaces_signed_issue_requirements_as_approval_critical(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Support issue comments",
        body="Original request.",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(),
        human_requirements=(
            HumanReviewRequirement(
                source_type="Issue body",
                author="maintainer",
                created_at="2026-05-17T08:00:00Z",
                url="https://github.com/OWNER/REPO/issues/56",
                body="Keep the public API unchanged.",
            ),
        ),
    )

    prompt = build_plan_review_prompt(
        56,
        1,
        "Plan:\n- Update the parser.",
        config,
        reviewer="codex",
        issue_context=issue_context,
    )

    assert "Signed Human Reviewer Requirements" in prompt
    assert "high-priority planning requirements" in prompt
    assert "approval-critical issue constraints" in prompt
    assert "Verify each requirement in this set before approving." in prompt
    assert prompt.index("Signed Human Reviewer Requirements") < prompt.index("Issue context from GitHub")


@pytest.mark.parametrize("builder", [build_followup_prompt, build_same_pr_followup_prompt])
def test_coder_followup_prompts_require_human_requirements_acknowledgement_only_when_present(
    tmp_path, builder
):
    config = make_config(tmp_path)
    with_requirements = builder(
        77,
        2,
        "Fix the bug.",
        config,
        human_requirements=(
            HumanReviewRequirement(
                source_type="PR comment",
                author="reviewer",
                created_at="2026-05-18T10:00:00Z",
                url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                body="Please use the absolute URL.",
            ),
        ),
    )
    without_requirements = builder(77, 2, "Fix the bug.", config)

    assert "mandatory next-revision requirements" in with_requirements
    assert HUMAN_REQUIREMENTS_ADDRESSED_MARKER in with_requirements
    assert "### Human requirements" in with_requirements
    assert "`Requirement 1`" in with_requirements
    assert "mandatory next-revision requirements" not in without_requirements
    assert "`Requirement 1`" not in without_requirements


@pytest.mark.parametrize("builder", [build_followup_prompt, build_same_pr_followup_prompt])
def test_coder_followup_prompts_accept_precomputed_human_requirements_context(tmp_path, builder):
    config = make_config(tmp_path)
    requirements = (
        HumanReviewRequirement(
            source_type="PR comment",
            author="reviewer",
            created_at="2026-05-18T10:00:00Z",
            url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
            body="Please use the absolute URL.",
        ),
    )
    context = render_coder_human_requirements_prompt_context(requirements)

    prompt = builder(
        77,
        2,
        "Fix the bug.",
        config,
        human_requirements=requirements,
        human_requirements_context=context,
    )

    assert context.block in prompt
    assert HUMAN_REQUIREMENTS_ADDRESSED_MARKER in prompt
    assert "`Requirement 1`" in prompt


@pytest.mark.parametrize("builder", [build_followup_prompt, build_same_pr_followup_prompt])
def test_coder_followup_prompts_require_structured_json(tmp_path, builder):
    config = make_config(tmp_path)

    prompt = builder(77, 2, "Fix the bug.", config)

    assert '"kind": "coder_followup"' in prompt
    assert '"addressed_items": ["item-1"]' in prompt
    assert '"remaining_items": ["item-2"]' in prompt
    assert '"human_requirements": {' in prompt
    assert "The JSON `state` must match the `AGENT_STATE` footer exactly." in prompt
    assert "Use this mandatory structured JSON follow-up format" in prompt
    assert "compatibility fallback" not in prompt
    assert "Legacy markdown replies" not in prompt


def test_validate_human_requirements_acknowledgement_accepts_multiple_bullet_styles():
    response = f"""Implemented the fix.
{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}
### Human requirements
1. Requirement 1: updated the URL handling.
* Requirement 2: could not satisfy safely without widening scope, so I documented the limit.
<!-- AGENT_STATE: blocking -->
"""

    validate_human_requirements_acknowledgement(
        response,
        surfaced_requirement_ids=("Requirement 1", "Requirement 2"),
        requires_direct_discussion_ack=False,
    )

    parsed = parse_human_requirements_acknowledgement(response)
    assert parsed.addressed_ids == ("Requirement 1", "Requirement 2")


@pytest.mark.parametrize(
    ("response", "surfaced_ids", "requires_direct_discussion_ack", "message"),
    [
        (
            "Implemented.\n### Human requirements\n- Requirement 1: handled.\n<!-- AGENT_STATE: blocking -->",
            ("Requirement 1",),
            False,
            "missing required signed human requirements marker",
        ),
        (
            f"Implemented.\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n<!-- AGENT_STATE: blocking -->",
            ("Requirement 1",),
            False,
            "missing required `### Human requirements` section",
        ),
        (
            f"Implemented.\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n### Human requirements\n- Requirement 1: handled.\n<!-- AGENT_STATE: blocking -->",
            ("Requirement 1", "Requirement 2"),
            False,
            "did not address all surfaced signed human requirement IDs",
        ),
        (
            f"Implemented.\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n### Human requirements\n- Requirement 1: handled.\n- Requirement 1: repeated.\n<!-- AGENT_STATE: blocking -->",
            ("Requirement 1",),
            False,
            "listed signed human requirement IDs more than once",
        ),
        (
            f"Implemented.\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n### Human requirements\n- Requirement 99: handled.\n<!-- AGENT_STATE: blocking -->",
            ("Requirement 1",),
            False,
            "referenced unknown signed human requirement IDs",
        ),
        (
            f"Implemented.\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n### Human requirements\n- Prompt omitted details.\n<!-- AGENT_STATE: blocking -->",
            (),
            True,
            "must acknowledge that the prompt omitted the detailed signed human requirements",
        ),
    ],
)
def test_validate_human_requirements_acknowledgement_rejects_structural_failures(
    response,
    surfaced_ids,
    requires_direct_discussion_ack,
    message,
):
    with pytest.raises(AgentLoopError, match=message):
        validate_human_requirements_acknowledgement(
            response,
            surfaced_requirement_ids=surfaced_ids,
            requires_direct_discussion_ack=requires_direct_discussion_ack,
        )


def test_validate_human_requirements_acknowledgement_accepts_full_truncation_fallback():
    response = f"""Implemented the fix.
{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}
### Human requirements
- The prompt omitted the detailed signed human requirements, so I {HUMAN_REQUIREMENTS_DIRECT_DISCUSSION_ACK}.
<!-- AGENT_STATE: blocking -->
"""

    validate_human_requirements_acknowledgement(
        response,
        surfaced_requirement_ids=(),
        requires_direct_discussion_ack=True,
    )


def test_issue_loop_can_use_codex_as_coder_and_claude_as_reviewer(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->",
            "Fixed review.\n<!-- AGENT_STATE: blocking -->",
        ],
        claude_outputs=[
            "Finding: bug remains.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    agent_commands = [cmd[:2] for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"])]
    assert agent_commands == [
        ["codex", "exec"],
        ["claude", "--print"],
        ["codex", "exec"],
        ["claude", "--print"],
    ]
    assert len(runner.comments) == 4
    assert runner.comments[-1].startswith("**Review verdict:** Approved\n\nLGTM.")


def test_issue_loop_runs_pre_review_tests_after_coder_changes(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Created PR.\nTests: pytest passed.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->",
            "Fixed review.\nTests: pytest passed.\n<!-- AGENT_STATE: blocking -->",
        ],
        codex_outputs=[
            "Finding: bug remains.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, test_command=("pytest", "tests/test_agent_loop.py"))

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    first_coder = command_index(runner.commands, ["claude", "--print"])
    first_test = commands.index(["pytest", "tests/test_agent_loop.py"])
    first_review = command_index(runner.commands, ["codex", "exec"])
    assert first_coder < first_test < first_review
    assert commands.count(["pytest", "tests/test_agent_loop.py"]) == 3


def test_pre_review_tests_can_be_disabled(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Created PR.\nTests: pytest passed.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(
        tmp_path,
        test_command=("pytest", "tests/test_agent_loop.py"),
        pre_review_tests=False,
    )

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    first_test = commands.index(["pytest", "tests/test_agent_loop.py"])
    first_review = command_index(runner.commands, ["codex", "exec"])
    assert first_review < first_test
    assert commands.count(["pytest", "tests/test_agent_loop.py"]) == 1


def test_codex_usage_summary_records_exact_tokens_from_jsonl_and_public_response(tmp_path):
    public_response = "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    runner = FakeRunner(
        codex_outputs=[
            {
                "public_response": public_response,
                "stdout": "\n".join(
                    [
                        json.dumps({"type": "turn.started"}),
                        json.dumps(
                            {
                                "type": "turn.completed",
                                "usage": {
                                    "input_tokens": 200,
                                    "cached_input_tokens": 40,
                                    "output_tokens": 50,
                                    "reasoning_tokens": 10,
                                    "total_tokens": 300,
                                },
                            }
                        ),
                    ]
                ),
            }
        ]
    )
    config = make_config(tmp_path, reviewer="codex")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments == [f"**Review verdict:** Approved\n\n{public_response}"]
    summary = read_usage_summary(tmp_path / "logs")
    assert summary["totals"]["exact_calls"] == 1
    assert summary["totals"]["estimated_calls"] == 0
    assert summary["totals"]["input_tokens"] == 200
    assert summary["totals"]["cached_input_tokens"] == 40
    assert summary["totals"]["output_tokens"] == 50
    assert summary["totals"]["reasoning_tokens"] == 10
    assert summary["totals"]["total_tokens"] == 300
    assert summary["calls"][0]["raw_backend_usage"]["cached_input_tokens"] == 40
    assert summary["calls"][0]["validation_status"] == "validated"


def test_usage_summary_estimates_tokens_when_backend_exposes_none(tmp_path):
    public_response = "LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(gemini_outputs=[public_response])
    config = make_config(tmp_path, reviewer="gemini")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    summary = read_usage_summary(tmp_path / "logs")
    call = summary["calls"][0]
    assert call["usage"]["mode"] == "estimated"
    assert call["usage"]["input_tokens"] == max(1, (call["usage"]["input_bytes"] + 3) // 4)
    assert call["usage"]["output_tokens"] == max(1, (call["usage"]["output_bytes"] + 3) // 4)
    assert call["usage"]["output_chars"] > len(public_response)


def test_usage_summary_keeps_retry_attempts_and_marks_only_validated_call_successful(tmp_path):
    near_miss = "LGTM.\nAGENT_STATE: approved.\n-- Google Gemini"
    valid = "LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(gemini_outputs=[near_miss, valid])
    config = make_config(tmp_path, reviewer="gemini")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    summary = read_usage_summary(tmp_path / "logs")
    assert len(summary["calls"]) == 2
    assert summary["totals"]["call_count"] == 2
    assert summary["totals"]["success_count"] == 1
    assert summary["calls"][0]["validation_status"] == "invalid"
    assert summary["calls"][1]["validation_status"] == "validated"


def test_plan_first_issue_run_writes_one_summary_for_planning_implementation_and_review(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Plan:\n- Implement usage logging.\n<!-- AGENT_PLAN_STATE: approved -->\n-- Anthropic Claude",
            "Opened PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Plan reviewed.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(
        runner,
        issue_number=56,
        config=config,
        plan_first=True,
        implement_after_approval=True,
    ) == 0

    summary = read_usage_summary(tmp_path / "logs")
    assert len(list((tmp_path / "logs").glob("*-usage-summary.json"))) == 1
    assert summary["totals"]["call_count"] == 4
    assert set(summary["per_agent"]) == {"claude", "codex"}
    assert [call["agent"] for call in summary["calls"]] == ["claude", "codex", "claude", "codex"]


def test_ensure_log_dir_ignored_does_not_overwrite_existing_file(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    gitignore = log_dir / ".gitignore"
    gitignore.write_text("custom\n", encoding="utf-8")

    ensure_log_dir_ignored(log_dir)

    assert gitignore.read_text(encoding="utf-8") == "custom\n"


def test_pr_loop_runs_tests_and_merge_only_after_codex_approval(tmp_path):
    runner = FakeRunner(codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"])
    config = make_config(
        tmp_path,
        auto_merge=True,
        test_command=("pytest", "tests/test_agent_loop.py"),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    assert ["pytest", "tests/test_agent_loop.py"] in commands
    assert [
        "gh",
        "api",
        "repos/OWNER/REPO/commits/abc123/check-runs",
    ] in commands
    assert [
        "gh",
        "api",
        "repos/OWNER/REPO/commits/abc123/status",
    ] in commands
    assert [
        "gh",
        "api",
        "repos/OWNER/REPO/branches/main/protection/required_status_checks",
    ] in commands
    assert ["gh", "pr", "merge", "77", "--repo", "OWNER/REPO", "--merge"] in commands


def test_pr_loop_does_not_post_gemini_diagnostics_without_agent_state(tmp_path):
    diagnostic = "[ERROR] Invalid stream: The model returned an empty response or malformed tool call."
    runner = FakeRunner(gemini_outputs=[diagnostic, diagnostic, diagnostic])
    config = make_config(tmp_path, reviewer="gemini")

    with pytest.raises(AgentLoopError, match="No review result was recorded"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert runner.comments == []
    assert not any(diagnostic in comment for comment in runner.comments)
    sleep_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["sleep"]]
    assert sleep_commands == [["sleep", "1"], ["sleep", "1"]]


@pytest.mark.parametrize(
    "text",
    [
        "orchestrator.py lines 577-581: it currently falls back to parse_plan_state(text)",
        "orchestrator.py:577-581: it currently falls back to parse_plan_state(text)",
        "A bare 500 in diagnostic prose without HTTP context.",
    ],
)
def test_source_line_references_with_5xx_numbers_are_not_transient(text):
    assert not _is_transient_agent_output(text)
    assert _failure_category(text) == "deterministic"


@pytest.mark.parametrize(
    "text",
    [
        "Internal Server Error",
        "Bad Gateway",
        "Service Unavailable",
        "Gateway Timeout",
    ],
)
def test_explicit_server_error_phrases_remain_transient(text):
    assert _is_transient_agent_output(text)
    assert _failure_category(text) == "transient"


def test_plan_review_does_not_post_diagnostics_without_plan_state(tmp_path):
    diagnostic = "[ERROR] Invalid stream: The model returned an empty response or malformed tool call."
    runner = FakeRunner(
        claude_outputs=[
            "Plan:\n- Update the CLI.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        gemini_outputs=[diagnostic, diagnostic, diagnostic],
    )
    config = make_config(tmp_path, reviewer="gemini")

    with pytest.raises(AgentLoopError, match="AGENT_PLAN_STATE"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert len(runner.comments) == 1
    assert runner.comments[0].startswith("Plan:")
    assert not any(diagnostic in comment for comment in runner.comments)


def test_pr_loop_retries_transient_gemini_diagnostic_and_posts_only_valid_response(tmp_path):
    diagnostic = "[ERROR] Invalid stream: The model returned an empty response or malformed tool call."
    valid = "LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(gemini_outputs=[diagnostic, valid])
    config = make_config(tmp_path, reviewer="gemini")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments == [f"**Review verdict:** Approved\n\n{valid}"]
    assert diagnostic not in runner.comments[0]
    sleep_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["sleep"]]
    assert sleep_commands == [["sleep", "1"]]


@pytest.mark.parametrize("terminator", ["", "."])
def test_pr_loop_retries_plain_agent_state_near_miss_once(tmp_path, terminator):
    near_miss = f"LGTM.\nAGENT_STATE: approved{terminator}\n-- Google Gemini"
    valid = "LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(gemini_outputs=[near_miss, valid])
    config = make_config(tmp_path, reviewer="gemini")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments == [f"**Review verdict:** Approved\n\n{valid}"]
    sleep_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["sleep"]]
    assert sleep_commands == [["sleep", "1"]]


def test_plan_loop_retries_plain_agent_plan_state_near_miss_once(tmp_path):
    near_miss = "Plan looks sound.\nAGENT_PLAN_STATE: approved.\n-- Google Gemini"
    valid = "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(
        claude_outputs=[
            "Plan:\n- Update the CLI.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        gemini_outputs=[near_miss, valid],
    )
    config = make_config(tmp_path, reviewer="gemini")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert near_miss not in runner.comments
    assert any(comment == f"**Review verdict:** Approved\n\n{valid}" for comment in runner.comments)
    sleep_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["sleep"]]
    assert sleep_commands == [["sleep", "1"]]


def test_gemini_public_response_file_is_inside_git_dir(tmp_path):
    valid = "LGTM from response file.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(gemini_outputs=["stdout should be ignored"], public_response_outputs=[valid])
    config = make_config(tmp_path, reviewer="gemini")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments == [f"**Review verdict:** Approved\n\n{valid}"]
    gemini_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["gemini"]]
    assert len(gemini_commands) == 1
    prompt = "\n".join(gemini_commands[0])
    expected_prefix = str(config.gemini_dir / ".git" / "agent-loop" / "responses" / "gemini")
    assert expected_prefix in prompt
    assert "/tmp/coding-review-agent-loop/responses/" not in prompt


def test_gemini_public_response_file_resolves_worktree_git_dir(tmp_path):
    valid = "LGTM from response file.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(gemini_outputs=["stdout should be ignored"], public_response_outputs=[valid])
    config = make_config(tmp_path, reviewer="gemini")
    git_dir = tmp_path / "main-repo" / ".git" / "worktrees" / "gemini"
    git_dir.mkdir(parents=True)
    (config.gemini_dir / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments == [f"**Review verdict:** Approved\n\n{valid}"]
    gemini_call = next(cmd for cmd, _cwd in runner.commands if cmd[:1] == ["gemini"])
    assert str(git_dir / "agent-loop" / "responses" / "gemini") in gemini_call[2]
    assert str(config.gemini_dir / ".git" / "agent-loop") not in gemini_call[2]


def test_gemini_pre_marker_429_does_not_suppress_structured_review_repair(tmp_path):
    malformed_public_review = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "Found one issue.",
                "blocking_items": [],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": [],
            }
        )
        + "\nProse between JSON and footer should be repaired.\n"
        "<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    )
    raw_stdout = (
        "Attempt 1 failed with status 429. Retrying with backoff... "
        "No capacity available for model gemini-3-flash-preview on the server.\n"
        f"{PUBLIC_RESPONSE_MARKER}\n"
        f"{malformed_public_review}"
    )
    repaired_review = structured_pr_review(
        state="approved",
        summary="Review passed after repair.",
        reviewer="Google Gemini",
    )
    runner = FakeRunner(gemini_outputs=[{"stdout": raw_stdout}])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)
    captured_repairs = []

    def fake_attempt_repair(raw: str, gemini_cmd: str, *, expected_kind: str | None = None, unresolved_item_ids=None, surfaced_requirement_ids=None, allowed_prior_item_ids=None, unknown_prior_item_ids=None, same_round_context=None) -> str | None:
        captured_repairs.append(raw)
        assert expected_kind == "pr_review"
        return repaired_review

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_attempt_repair):
        assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert captured_repairs == [malformed_public_review]
    assert "429" not in captured_repairs[0]
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)
    assert any("Review passed after repair." in comment for comment in runner.comments)


def test_gemini_response_file_repair_ignores_raw_stdout_transient_diagnostics(tmp_path):
    malformed_public_review = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "Found one issue.",
                "blocking_items": ["Approved reviews cannot have blocking items."],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    )
    repaired_review = structured_pr_review(
        state="approved",
        summary="Response file review passed after repair.",
        reviewer="Google Gemini",
    )
    runner = FakeRunner(
        gemini_outputs=[
            {"stdout": "Attempt 1 failed with status 429. No capacity available, then recovered."}
        ],
        public_response_outputs=[{"text": malformed_public_review}],
    )
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)
    captured_repairs = []

    def fake_attempt_repair(raw: str, gemini_cmd: str, *, expected_kind: str | None = None, unresolved_item_ids=None, surfaced_requirement_ids=None, allowed_prior_item_ids=None, unknown_prior_item_ids=None, same_round_context=None) -> str | None:
        captured_repairs.append(raw)
        return repaired_review

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_attempt_repair):
        assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert captured_repairs == [malformed_public_review]
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)
    assert any("Response file review passed after repair." in comment for comment in runner.comments)


def test_pr_loop_exhausted_transient_retry_reports_attempt_logs(tmp_path):
    diagnostic = "[ERROR] Invalid stream: The model returned an empty response or malformed tool call."
    runner = FakeRunner(gemini_outputs=[(diagnostic, 1), (diagnostic, 1), (diagnostic, 1)])
    config = make_config(tmp_path, reviewer="gemini")

    with pytest.raises(AgentLoopError) as exc_info:
        run_pr_loop(runner, pr_number=77, config=config)

    message = str(exc_info.value)
    assert "No review result was recorded" in message
    assert "Failure category: transient" in message
    assert "Attempt logs:" in message
    assert "gemini.log" in message
    assert runner.comments == []


def test_pr_loop_retries_quota_error(tmp_path):
    quota_output = "Quota exceeded for this project."
    valid = "LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(gemini_outputs=[(quota_output, 1), valid])
    config = make_config(tmp_path, reviewer="gemini")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments == [f"**Review verdict:** Approved\n\n{valid}"]
    sleep_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["sleep"]]
    assert len(sleep_commands) == 1


def test_pr_loop_does_not_retry_normal_missing_marker_response(tmp_path):
    output = "I reviewed the PR and it looks fine."
    runner = FakeRunner(gemini_outputs=[output])
    config = make_config(tmp_path, reviewer="gemini")

    with pytest.raises(AgentLoopError, match="AGENT_STATE"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert runner.comments == []
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)


def test_pr_loop_retries_rate_limit_429(tmp_path):
    rate_limit_output = "HTTP 429 Too Many Requests: rate limit exceeded."
    valid = "LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(gemini_outputs=[(rate_limit_output, 1), valid])
    config = make_config(tmp_path, reviewer="gemini")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments == [f"**Review verdict:** Approved\n\n{valid}"]
    sleep_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["sleep"]]
    assert len(sleep_commands) == 1


def test_pr_loop_retries_claude_session_limit(tmp_path):
    session_limit_output = "Error: session_limit_exceeded — too many sessions for this project."
    valid = "LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(gemini_outputs=[(session_limit_output, 1), valid])
    config = make_config(tmp_path, reviewer="gemini")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments == [f"**Review verdict:** Approved\n\n{valid}"]
    sleep_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["sleep"]]
    assert len(sleep_commands) == 1


def test_pr_loop_retries_gemini_no_capacity(tmp_path):
    no_capacity_output = "No capacity available for model gemini-flash on the server."
    valid = "LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(gemini_outputs=[(no_capacity_output, 1), valid])
    config = make_config(tmp_path, reviewer="gemini")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments == [f"**Review verdict:** Approved\n\n{valid}"]
    sleep_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["sleep"]]
    assert len(sleep_commands) == 1


def test_diagnostic_shaped_public_response_remains_transient(tmp_path):
    public_response = (
        f"{PUBLIC_RESPONSE_MARKER}\n"
        "HTTP 429 Too Many Requests: rate limit exceeded.\n"
        "<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    )
    runner = FakeRunner(gemini_outputs=[{"stdout": public_response}])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        with pytest.raises(AgentLoopError) as exc_info:
            run_pr_loop(runner, pr_number=77, config=config)

    repair_mock.assert_not_called()
    assert "Failure category: transient" in str(exc_info.value)


def test_public_response_error_payload_remains_transient():
    assert _is_transient_public_response(
        json.dumps(
            {
                "error": {
                    "code": 429,
                    "status": "RESOURCE_EXHAUSTED",
                    "message": "Quota exceeded. Retry-After: 60",
                }
            }
        )
    )


def test_public_response_structured_json_after_known_artifact_is_not_transient():
    text = (
        f"{PUBLIC_RESPONSE_MARKER}\n"
        + structured_pr_review(
            summary="Wrong structured kind discusses 429, quota, capacity, and transient behavior.",
            reviewer="Google Gemini",
        )
    )

    assert not _is_transient_public_response(text, repair_expected_kind="coder_followup")


def test_structured_plan_review_transient_terms_with_trailing_prose_normalizes(tmp_path):
    malformed_review = (
        structured_plan_review(
            state="approved",
            summary=(
                "The plan discusses 429, quota, resource exhausted, timeout, capacity, "
                "and transient retry handling as domain text."
            ),
            reviewer="Google Gemini",
        )
        + "\nTrailing prose after the signature should be repaired."
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the plan.",
            marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
            validate=lambda text: _validate_plan_review_response(
                text,
                reviewer="Google Gemini",
                unresolved_items=(),
            ),
            use_repair=True,
            repair_expected_kind="plan_review",
        )

    assert response.text == structured_plan_review(
        state="approved",
        summary=(
            "The plan discusses 429, quota, resource exhausted, timeout, capacity, "
            "and transient retry handling as domain text."
        ),
        reviewer="Google Gemini",
    )
    repair_mock.assert_not_called()
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)


def test_structured_pr_review_transient_terms_duplicate_footer_normalizes(tmp_path):
    malformed_review = (
        structured_pr_review(
            state="approved",
            summary=(
                "The review covers capacity, timeout, 429, quota, resource-exhausted, "
                "and transient classifier behavior."
            ),
            reviewer="Google Gemini",
        )
        + "\n\n<!-- AGENT_STATE: approved -->"
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the PR.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: _validate_review_response(
                text,
                reviewer="Google Gemini",
                unresolved_items=(),
            ),
            use_repair=True,
            repair_expected_kind="pr_review",
        )

    assert response.text == structured_pr_review(
        state="approved",
        summary=(
            "The review covers capacity, timeout, 429, quota, resource-exhausted, "
            "and transient classifier behavior."
        ),
        reviewer="Google Gemini",
    )
    repair_mock.assert_not_called()
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)


def test_structured_coder_followup_transient_terms_before_footer_runs_repair(tmp_path):
    unresolved_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="OpenAI Codex",
            source_round=1,
            text="Add timeout regression coverage.",
            status="blocking",
        ),
    )
    malformed_followup = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "approved",
                "summary": "Updated timeout and capacity handling without treating prose as transient.",
                "addressed_items": ["item-1"],
                "remaining_items": [],
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n## Changes made\nMentioned timeout and capacity in prose before the footer.\n"
        "<!-- AGENT_STATE: approved -->\n-- Anthropic Claude"
    )
    repaired_followup = structured_coder_followup(
        state="approved",
        summary="Updated timeout and capacity handling.",
        addressed_items=["item-1"],
        remaining_items=[],
        reviewer="Anthropic Claude",
    )
    runner = FakeRunner(claude_outputs=[malformed_followup])
    config = make_config(tmp_path, coder="claude", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=repaired_followup) as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="claude",
            config=config,
            prompt="Address review feedback.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: _validate_coder_followup_response(
                text,
                unresolved_items=unresolved_items,
                human_requirements=(),
            ),
            use_repair=True,
            repair_expected_kind="coder_followup",
        )

    assert response.text == repaired_followup
    repair_mock.assert_called_once_with(
        malformed_followup,
        config.gemini_cmd,
        expected_kind="coder_followup",
    )
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)


def test_run_validated_agent_recovers_coder_followup_from_message_text_when_response_file_markdown(tmp_path):
    unresolved_items = (
        UnresolvedReviewItem(
            item_id="item-8",
            reviewer="Orchestrator",
            source_round=4,
            text="Acknowledge signed human requirements.",
            status="blocking",
        ),
    )
    valid_followup = structured_coder_followup(
        state="approved",
        summary="Acknowledged the signed human requirements.",
        addressed_items=["item-8"],
        remaining_items=[],
        reviewer="OpenAI Codex",
    )
    markdown_response_file = (
        "### Human requirements\n\n"
        "Acknowledged.\n\n"
        "<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n"
        "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    )
    runner = FakeRunner(
        codex_outputs=[{"public_response": valid_followup, "stdout": "diagnostic output"}],
        public_response_outputs=[{"text": markdown_response_file}],
    )
    config = make_config(tmp_path, coder="codex", agent_max_retries=0)

    response = _run_validated_agent(
        runner,
        agent="codex",
        config=config,
        prompt="Address feedback.",
        marker_description="<!-- AGENT_STATE: approved|blocking -->",
        validate=lambda text: _validate_coder_followup_response(
            text,
            unresolved_items=unresolved_items,
            human_requirements=(),
        ),
        repair_expected_kind="coder_followup",
    )

    assert response.text == valid_followup
    assert response.marker_value.addressed_items == ("item-8",)


def test_run_validated_agent_recovers_fenced_coder_followup_from_raw_stdout(tmp_path):
    unresolved_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="OpenAI Codex",
            source_round=1,
            text="Fix the bug.",
            status="blocking",
        ),
    )
    valid_followup = structured_coder_followup(
        state="approved",
        addressed_items=["item-1"],
        remaining_items=[],
        reviewer="OpenAI Codex",
    )
    json_part, footer = valid_followup.split("\n<!-- AGENT_STATE:", 1)
    fenced_stdout = f"tool diagnostic\n```json\n{json_part}\n```\n<!-- AGENT_STATE:{footer}"
    runner = FakeRunner(
        codex_outputs=[{"public_response": "legacy markdown", "stdout": fenced_stdout}],
        public_response_outputs=[{"text": "### Update\nFixed it.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"}],
    )
    config = make_config(tmp_path, coder="codex", agent_max_retries=0)

    response = _run_validated_agent(
        runner,
        agent="codex",
        config=config,
        prompt="Address feedback.",
        marker_description="<!-- AGENT_STATE: approved|blocking -->",
        validate=lambda text: _validate_coder_followup_response(
            text,
            unresolved_items=unresolved_items,
            human_requirements=(),
        ),
        repair_expected_kind="coder_followup",
    )

    assert response.text == valid_followup


def _plan_revision_validate_with_human_requirements(human_requirements):
    return lambda text: orchestrator._validate_response_with_human_requirements(
        text,
        marker_validator=lambda revised_text: _validate_plan_revision_response(
            revised_text,
            unresolved_items=(),
        ),
        human_requirements=human_requirements,
        requirement_scope="planning requirements",
        full_omission_fallback="Fetch the issue discussion directly before revising the plan.",
    )


def test_run_validated_agent_recovers_plan_revision_human_ack_from_message_text(tmp_path):
    human_requirements = (
        HumanReviewRequirement(
            source_type="Issue comment",
            author="wwind123",
            created_at="2026-06-05T00:00:00Z",
            url="https://github.com/OWNER/REPO/issues/237#issuecomment-1",
            body="Cover stdout acknowledgement recovery.",
        ),
    )
    context = render_coder_human_requirements_prompt_context(
        human_requirements,
        requirement_scope="planning requirements",
        full_omission_fallback="Fetch the issue discussion directly before revising the plan.",
    )
    response_file = structured_plan_revision(reviewer="Anthropic Claude")
    acknowledgement = (
        "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n"
        "### Human requirements\n"
        "- Requirement 1: The revised plan covers stdout acknowledgement recovery.\n"
    )
    message_text = structured_plan_revision(
        reviewer="Anthropic Claude",
        human_requirements=acknowledgement,
    )
    runner = FakeRunner(
        claude_outputs=[(message_text, 0)],
        public_response_outputs=[{"text": response_file}],
    )
    config = make_config(tmp_path, coder="claude", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="claude",
            config=config,
            prompt="Revise the plan.",
            marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
            validate=_plan_revision_validate_with_human_requirements(human_requirements),
            use_repair=True,
            repair_expected_kind="plan_revision",
            repair_surfaced_requirement_ids=context.surfaced_requirement_ids,
            repair_requires_direct_discussion_ack=context.requires_direct_discussion_ack,
        )

    repair_mock.assert_not_called()
    assert "<!-- HUMAN_REQUIREMENTS_ADDRESSED -->" in response.text
    assert "### Human requirements" in response.text
    assert response.text.index("### Human requirements") < response.text.index(
        "<!-- AGENT_PLAN_STATE: blocking -->"
    )


@pytest.mark.parametrize(
    "evidence",
    [
        "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n",
        "\n### Human requirements\n- Requirement 1: Covered.\n",
        "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n### Human requirements\n- Requirement 99: Covered.\n",
        "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n### Human requirements\n- Requirement 1: Covered.\n- Requirement 1: Covered again.\n",
    ],
)
def test_run_validated_agent_refuses_invalid_plan_revision_human_ack_evidence(tmp_path, evidence):
    human_requirements = (
        HumanReviewRequirement(
            source_type="Issue comment",
            author="wwind123",
            created_at="2026-06-05T00:00:00Z",
            url="https://github.com/OWNER/REPO/issues/237#issuecomment-1",
            body="Cover stdout acknowledgement recovery.",
        ),
    )
    context = render_coder_human_requirements_prompt_context(
        human_requirements,
        requirement_scope="planning requirements",
        full_omission_fallback="Fetch the issue discussion directly before revising the plan.",
    )
    runner = FakeRunner(
        claude_outputs=[
            (
                structured_plan_revision(
                    reviewer="Anthropic Claude",
                    human_requirements=evidence,
                ),
                0,
            )
        ],
        public_response_outputs=[{"text": structured_plan_revision(reviewer="Anthropic Claude")}],
    )
    config = make_config(tmp_path, coder="claude", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=None) as repair_mock:
        with pytest.raises(AgentLoopError, match="No review result was recorded"):
            _run_validated_agent(
                runner,
                agent="claude",
                config=config,
                prompt="Revise the plan.",
                marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
                validate=_plan_revision_validate_with_human_requirements(human_requirements),
                use_repair=True,
                repair_expected_kind="plan_revision",
                repair_surfaced_requirement_ids=context.surfaced_requirement_ids,
                repair_requires_direct_discussion_ack=context.requires_direct_discussion_ack,
            )

    repair_mock.assert_called_once()


def test_run_validated_agent_refuses_plan_revision_missing_direct_discussion_ack(tmp_path):
    context = render_coder_human_requirements_prompt_context(
        (),
        requirement_scope="planning requirements",
        full_omission_fallback="Fetch the issue discussion directly before revising the plan.",
    )
    acknowledgement = (
        "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n"
        "### Human requirements\n"
        "- The prompt omitted the detailed signed human requirements.\n"
    )
    runner = FakeRunner(
        claude_outputs=[
            (
                structured_plan_revision(
                    reviewer="Anthropic Claude",
                    human_requirements=acknowledgement,
                ),
                0,
            )
        ],
        public_response_outputs=[{"text": structured_plan_revision(reviewer="Anthropic Claude")}],
    )
    config = make_config(tmp_path, coder="claude", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=None):
        with pytest.raises(AgentLoopError, match="No review result was recorded"):
            _run_validated_agent(
                runner,
                agent="claude",
                config=config,
                prompt="Revise the plan.",
                marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
                validate=lambda text: (
                    _validate_plan_revision_response(text, unresolved_items=()),
                    validate_human_requirements_acknowledgement(
                        text,
                        surfaced_requirement_ids=(),
                        requires_direct_discussion_ack=True,
                    ),
                )[0],
                use_repair=True,
                repair_expected_kind="plan_revision",
                repair_surfaced_requirement_ids=context.surfaced_requirement_ids,
                repair_requires_direct_discussion_ack=True,
            )


def test_run_validated_agent_refuses_conflicting_plan_revision_human_ack_blocks(tmp_path):
    human_requirements = (
        HumanReviewRequirement(
            source_type="Issue comment",
            author="wwind123",
            created_at="2026-06-05T00:00:00Z",
            url="https://github.com/OWNER/REPO/issues/237#issuecomment-1",
            body="Cover stdout acknowledgement recovery.",
        ),
    )
    context = render_coder_human_requirements_prompt_context(
        human_requirements,
        requirement_scope="planning requirements",
        full_omission_fallback="Fetch the issue discussion directly before revising the plan.",
    )
    first = (
        "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n"
        "### Human requirements\n- Requirement 1: Covered by the parser step.\n"
    )
    second = (
        "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n"
        "### Human requirements\n- Requirement 1: Covered by the orchestrator step.\n"
    )
    runner = FakeRunner(
        claude_outputs=[
            (
                structured_plan_revision(
                    reviewer="Anthropic Claude",
                    human_requirements=first,
                )
                + "\n\n"
                + second,
                0,
            )
        ],
        public_response_outputs=[{"text": structured_plan_revision(reviewer="Anthropic Claude")}],
    )
    config = make_config(tmp_path, coder="claude", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=None):
        with pytest.raises(AgentLoopError, match="No review result was recorded"):
            _run_validated_agent(
                runner,
                agent="claude",
                config=config,
                prompt="Revise the plan.",
                marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
                validate=_plan_revision_validate_with_human_requirements(human_requirements),
                use_repair=True,
                repair_expected_kind="plan_revision",
                repair_surfaced_requirement_ids=context.surfaced_requirement_ids,
                repair_requires_direct_discussion_ack=context.requires_direct_discussion_ack,
            )


def test_run_validated_agent_does_not_recover_unknown_prior_item_disposition(tmp_path):
    human_requirements = (
        HumanReviewRequirement(
            source_type="Issue comment",
            author="wwind123",
            created_at="2026-06-05T00:00:00Z",
            url="https://github.com/OWNER/REPO/issues/237#issuecomment-1",
            body="Cover stdout acknowledgement recovery.",
        ),
    )
    context = render_coder_human_requirements_prompt_context(
        human_requirements,
        requirement_scope="planning requirements",
        full_omission_fallback="Fetch the issue discussion directly before revising the plan.",
    )
    acknowledgement = (
        "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n"
        "### Human requirements\n- Requirement 1: Covered.\n"
    )
    response_file = structured_plan_revision(
        reviewer="Anthropic Claude",
        prior_plan_item_dispositions=[
            {"item_id": "item-unknown", "disposition": "resolved", "note": "Covered."}
        ],
    )
    runner = FakeRunner(
        claude_outputs=[
            (
                structured_plan_revision(
                    reviewer="Anthropic Claude",
                    human_requirements=acknowledgement,
                ),
                0,
            )
        ],
        public_response_outputs=[{"text": response_file}],
    )
    config = make_config(tmp_path, coder="claude", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=None) as repair_mock:
        with pytest.raises(AgentLoopError, match="No review result was recorded"):
            _run_validated_agent(
                runner,
                agent="claude",
                config=config,
                prompt="Revise the plan.",
                marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
                validate=_plan_revision_validate_with_human_requirements(human_requirements),
                use_repair=True,
                repair_expected_kind="plan_revision",
                repair_surfaced_requirement_ids=context.surfaced_requirement_ids,
                repair_requires_direct_discussion_ack=context.requires_direct_discussion_ack,
                repair_allowed_prior_item_ids=(),
            )

    repair_mock.assert_called_once()


@pytest.mark.parametrize(
    "stdout",
    [
        structured_pr_review(reviewer="OpenAI Codex"),
        "diagnostic output without a structured response",
    ],
)
def test_run_validated_agent_refuses_unrecoverable_stdout_when_response_file_markdown(tmp_path, stdout):
    runner = FakeRunner(
        codex_outputs=[{"public_response": "legacy markdown", "stdout": stdout}],
        public_response_outputs=[{"text": "### Update\nFixed it.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"}],
    )
    config = make_config(tmp_path, coder="codex", agent_max_retries=0)

    with pytest.raises(AgentLoopError, match="No review result was recorded"):
        _run_validated_agent(
            runner,
            agent="codex",
            config=config,
            prompt="Address feedback.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: _validate_coder_followup_response(
                text,
                unresolved_items=(),
                human_requirements=(),
            ),
            repair_expected_kind="coder_followup",
        )


def test_run_validated_agent_refuses_multiple_stdout_structured_candidates(tmp_path):
    first = structured_coder_followup(
        state="approved",
        summary="First candidate.",
        reviewer="OpenAI Codex",
    )
    second = structured_coder_followup(
        state="approved",
        summary="Second candidate.",
        reviewer="OpenAI Codex",
    )
    runner = FakeRunner(
        codex_outputs=[{"public_response": "legacy markdown", "stdout": first + "\n\n" + second}],
        public_response_outputs=[{"text": "### Update\nFixed it.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"}],
    )
    config = make_config(tmp_path, coder="codex", agent_max_retries=0)

    with pytest.raises(AgentLoopError, match="No review result was recorded"):
        _run_validated_agent(
            runner,
            agent="codex",
            config=config,
            prompt="Address feedback.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: _validate_coder_followup_response(
                text,
                unresolved_items=(),
                human_requirements=(),
            ),
            repair_expected_kind="coder_followup",
        )


def test_run_validated_agent_keeps_valid_response_file_authoritative_over_noisy_stdout(tmp_path):
    valid_followup = structured_coder_followup(
        state="approved",
        summary="Response file wins.",
        reviewer="OpenAI Codex",
    )
    runner = FakeRunner(
        codex_outputs=[{"public_response": "ignored message", "stdout": "unrelated noisy diagnostics"}],
        public_response_outputs=[{"text": valid_followup}],
    )
    config = make_config(tmp_path, coder="codex", agent_max_retries=0)

    response = _run_validated_agent(
        runner,
        agent="codex",
        config=config,
        prompt="Address feedback.",
        marker_description="<!-- AGENT_STATE: approved|blocking -->",
        validate=lambda text: _validate_coder_followup_response(
            text,
            unresolved_items=(),
            human_requirements=(),
        ),
        repair_expected_kind="coder_followup",
    )

    assert response.text == valid_followup


def test_structured_plan_revision_transient_terms_before_footer_runs_repair(tmp_path):
    malformed_revision = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_revision",
                "state": "blocking",
                "summary": "Revise handling for 429, quota, resource exhausted, transient, and timeout.",
                "prior_plan_item_dispositions": [],
                "plan_steps": [
                    "Separate public-response validation from transient raw diagnostics.",
                    "Keep capacity and quota retry handling for raw provider failures.",
                ],
            }
        )
        + "\n## Revised plan\nProse before the AGENT_PLAN_STATE footer is invalid.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    repaired_revision = structured_plan_revision(
        summary="Revised transient classifier plan.",
        plan_steps=[
            "Separate public-response validation from transient raw diagnostics.",
            "Keep capacity and quota retry handling for raw provider failures.",
        ],
        reviewer="Anthropic Claude",
    )
    runner = FakeRunner(claude_outputs=[malformed_revision])
    config = make_config(tmp_path, coder="claude", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=repaired_revision) as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="claude",
            config=config,
            prompt="Revise the plan.",
            marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
            validate=_validate_plan_revision_response,
            use_repair=True,
            repair_expected_kind="plan_revision",
            repair_surfaced_requirement_ids=("Requirement 1",),
        )

    assert response.text == repaired_revision
    repair_mock.assert_called_once_with(
        malformed_revision,
        config.gemini_cmd,
        expected_kind="plan_revision",
    )
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)


def test_run_validated_agent_repairs_unknown_prior_item_disposition_when_ledger_complete(tmp_path):
    malformed_review = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the PR.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: _validate_review_response(
                text,
                reviewer="Google Gemini",
                unresolved_items=(),
            ),
            use_repair=True,
            repair_expected_kind="pr_review",
            repair_allowed_prior_item_ids=(),
            ledger_incomplete=False,
    )

    repair_mock.assert_not_called()
    parsed_response = json.loads(response.text.split("\n")[0])
    assert parsed_response["prior_item_dispositions"] == []
    assert parsed_response["state"] == "approved"
    assert parsed_response["summary"] == "LGTM."


def test_run_validated_agent_skips_unknown_prior_item_repair_when_ledger_incomplete(tmp_path):
    malformed_review = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        with pytest.raises(AgentLoopError) as exc_info:
            _run_validated_agent(
                runner,
                agent="gemini",
                config=config,
                prompt="Review the PR.",
                marker_description="<!-- AGENT_STATE: approved|blocking -->",
                validate=lambda text: _validate_review_response(
                    text,
                    reviewer="Google Gemini",
                    unresolved_items=(),
                ),
                use_repair=True,
                repair_expected_kind="pr_review",
                repair_allowed_prior_item_ids=(),
                ledger_incomplete=True,
            )

    repair_mock.assert_not_called()
    assert "item-1" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Issue #271: coder_followup path through attempt_envelope_normalization
# ---------------------------------------------------------------------------

def test_envelope_normalization_coder_followup_duplicate_state_footer():
    """attempt_envelope_normalization handles coder_followup with a duplicate AGENT_STATE footer."""
    raw = (
        structured_coder_followup(
            state="approved",
            reviewer="Anthropic Claude",
            addressed_items=["item-1"],
        )
        + "\n\n<!-- AGENT_STATE: approved -->"
    )

    normalized = attempt_envelope_normalization(raw, expected_kind="coder_followup")

    assert normalized is not None
    parsed = validate_structured_coder_followup(normalized)
    assert parsed is not None
    assert parsed.addressed_items == ("item-1",)
    assert normalized.count("<!-- AGENT_STATE: approved -->") == 1


def test_envelope_normalization_coder_followup_trailing_prose_after_signature():
    """attempt_envelope_normalization strips trailing prose after coder_followup signature."""
    raw = (
        structured_coder_followup(
            state="blocking",
            reviewer="Anthropic Claude",
            remaining_items=["item-2"],
        )
        + "\n\nExtra prose that should be stripped."
    )

    normalized = attempt_envelope_normalization(raw, expected_kind="coder_followup")

    assert normalized is not None
    assert "Extra prose" not in normalized
    parsed = validate_structured_coder_followup(normalized)
    assert parsed is not None
    assert parsed.remaining_items == ("item-2",)


def test_envelope_normalization_coder_followup_returns_none_when_prose_before_footer():
    """attempt_envelope_normalization returns None for coder_followup with prose before the footer."""
    raw = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "approved",
                "summary": "Done.",
                "addressed_items": [],
                "remaining_items": [],
                "human_requirements": {"addressed_ids": [], "checked_discussion_directly": False},
            }
        )
        + "\nSome unexpected prose here.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude"
    )

    assert attempt_envelope_normalization(raw, expected_kind="coder_followup") is None


# ---------------------------------------------------------------------------
# Issue #275: strip_unknown_prior_item_dispositions with tightly-packed input
# ---------------------------------------------------------------------------

def test_strip_unknown_prior_item_dispositions_tightly_packed_no_newline_before_footer():
    """strip_unknown_prior_item_dispositions inserts a newline when the original had none."""
    payload = {
        "schema_version": 1,
        "kind": "pr_review",
        "state": "approved",
        "summary": "LGTM.",
        "blocking_items": [],
        "same_pr_followups": [],
        "future_followups": [],
        "prior_item_dispositions": [{"item_id": "item-99", "disposition": "resolved"}],
    }
    # Tightly packed: no newline between JSON and footer
    raw = json.dumps(payload) + "<!-- AGENT_STATE: approved -->\n-- Google Gemini"

    result = strip_unknown_prior_item_dispositions(
        raw, allowed_ids=frozenset(), expected_kind="pr_review"
    )

    assert result is not None
    # Validate parses correctly even after tight packing
    parsed_payload, json_end = json.JSONDecoder().raw_decode(result.lstrip())
    assert parsed_payload["prior_item_dispositions"] == []
    tail = result.lstrip()[json_end:]
    assert "<!-- AGENT_STATE: approved -->" in tail
    assert "-- Google Gemini" in tail
    # The footer must be separated from the JSON by at least a newline
    assert tail.startswith("\n")


def test_strip_unknown_prior_item_dispositions_tightly_packed_result_validates():
    """Tight-packing case validates successfully through parse_structured_pr_review."""
    payload = {
        "schema_version": 1,
        "kind": "pr_review",
        "state": "approved",
        "summary": "LGTM.",
        "blocking_items": [],
        "same_pr_followups": [],
        "future_followups": [],
        "prior_item_dispositions": [{"item_id": "item-99", "disposition": "resolved"}],
    }
    raw = json.dumps(payload) + "<!-- AGENT_STATE: approved -->\n-- Google Gemini"

    result = strip_unknown_prior_item_dispositions(
        raw, allowed_ids=frozenset(), expected_kind="pr_review"
    )

    assert result is not None
    parsed = parse_structured_pr_review(result, reviewer="Google Gemini")
    assert parsed is not None
    assert parsed.dispositions == ()


# ---------------------------------------------------------------------------
# Issue #274: combined envelope+disposition strip via _run_validated_agent
# ---------------------------------------------------------------------------

def test_run_validated_agent_combined_envelope_and_disposition_fix(tmp_path):
    """When a response has both an envelope defect and unknown prior dispositions,
    stripping dispositions from the envelope-normalized candidate recovers it."""
    # Build a plan_review with an unknown disposition AND a duplicate footer (envelope defect).
    # strip_unknown_prior_item_dispositions on the original fails to validate because the
    # duplicate footer is still present. Envelope normalization on the original produces a
    # normalized candidate; stripping dispositions from that candidate should succeed.
    base = structured_plan_review(
        state="approved",
        summary="Plan approved.",
        prior_plan_item_dispositions=[{"item_id": "item-99", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    # Add a duplicate footer to create the envelope defect
    malformed = base + "\n\n<!-- AGENT_PLAN_STATE: approved -->"

    runner = FakeRunner(gemini_outputs=[malformed])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the plan.",
            marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
            validate=lambda text: _validate_plan_review_response(
                text,
                reviewer="Google Gemini",
                unresolved_items=(),
            ),
            use_repair=True,
            repair_expected_kind="plan_review",
            repair_allowed_prior_item_ids=(),
            ledger_incomplete=False,
        )

    repair_mock.assert_not_called()
    parsed = json.loads(response.text.split("\n")[0])
    assert parsed["prior_plan_item_dispositions"] == []
    assert parsed["state"] == "approved"


def test_run_validated_agent_rejects_repair_that_invents_prior_item_id(tmp_path):
    # item-3 is the legitimate carried prior item; item-1 is unknown and gets stripped
    # deterministically, but item-3 is then missing → re-validation fails → falls through
    # to generative repair → repair invents item-2 (also unknown) → rejected.
    carried_item_3 = UnresolvedReviewItem(
        item_id="item-3",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Must-fix prior item.",
        status="blocking",
    )
    malformed_review = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    repaired_review = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[{"item_id": "item-2", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=repaired_review):
        with pytest.raises(AgentLoopError) as exc_info:
            _run_validated_agent(
                runner,
                agent="gemini",
                config=config,
                prompt="Review the PR.",
                marker_description="<!-- AGENT_STATE: approved|blocking -->",
                validate=lambda text: _validate_review_response(
                    text,
                    reviewer="Google Gemini",
                    unresolved_items=(carried_item_3,),
                ),
                use_repair=True,
                repair_expected_kind="pr_review",
                repair_allowed_prior_item_ids=("item-3",),
            )

    assert "item-2" in str(exc_info.value)


def test_run_validated_agent_preserves_valid_disposition_when_repair_removes_unknown(tmp_path):
    carried_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Keep this prior item.",
        status="blocking",
    )
    malformed_review = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[
            {"item_id": "item-1", "disposition": "resolved"},
            {"item_id": "item-9", "disposition": "resolved"},
        ],
        reviewer="Google Gemini",
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the PR.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: _validate_review_response(
                text,
                reviewer="Google Gemini",
                unresolved_items=(carried_item,),
            ),
            use_repair=True,
            repair_expected_kind="pr_review",
            repair_allowed_prior_item_ids=("item-1",),
        )

    repair_mock.assert_not_called()
    assert response.marker_value.dispositions[0].item_id == "item-1"
    assert len(response.marker_value.dispositions) == 1


def test_run_validated_agent_repairs_unknown_plan_revision_prior_disposition(tmp_path):
    active_item = UnresolvedReviewItem(
        item_id="item-12",
        reviewer="Google Gemini",
        source_round=5,
        text="Active must-fix item.",
        status="blocking",
    )
    malformed_revision = structured_plan_revision(
        prior_plan_item_dispositions=[
            {"item_id": "item-15", "disposition": "resolved"},
        ],
    )
    runner = FakeRunner(claude_outputs=[malformed_revision])
    config = make_config(tmp_path, coder="claude", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="claude",
            config=config,
            prompt="Revise the plan.",
            marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
            validate=lambda text: _validate_plan_revision_response(
                text,
                unresolved_items=(active_item,),
            ),
            use_repair=True,
            repair_expected_kind="plan_revision",
            repair_allowed_prior_item_ids=("item-12",),
            ledger_incomplete=False,
        )

    repair_mock.assert_not_called()
    parsed_response = json.loads(response.text.split("\n")[0])
    assert parsed_response["prior_plan_item_dispositions"] == []


def test_run_validated_agent_plan_revision_unknown_prior_disposition_fails_when_ledger_incomplete(tmp_path):
    active_item = UnresolvedReviewItem(
        item_id="item-12",
        reviewer="Google Gemini",
        source_round=5,
        text="Active must-fix item.",
        status="blocking",
    )
    malformed_revision = structured_plan_revision(
        prior_plan_item_dispositions=[
            {"item_id": "item-15", "disposition": "resolved"},
            {"item_id": "item-18", "disposition": "resolved"},
        ],
    )
    runner = FakeRunner(claude_outputs=[malformed_revision])
    config = make_config(tmp_path, coder="claude", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        with pytest.raises(AgentLoopError) as exc_info:
            _run_validated_agent(
                runner,
                agent="claude",
                config=config,
                prompt="Revise the plan.",
                marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
                validate=lambda text: _validate_plan_revision_response(
                    text,
                    unresolved_items=(active_item,),
                ),
                use_repair=True,
                repair_expected_kind="plan_revision",
                repair_allowed_prior_item_ids=("item-12",),
                ledger_incomplete=True,
            )

    repair_mock.assert_not_called()
    assert "item-15" in str(exc_info.value)
    assert "item-18" in str(exc_info.value)


def test_gemini_duplicate_trailing_agent_state_marker_normalizes_without_repair(tmp_path):
    malformed_public_review = (
        structured_pr_review(
            state="approved",
            summary="Found one issue.",
            reviewer="Google Gemini",
        )
        + "\n\n<!-- AGENT_STATE: approved -->"
    )
    raw_stdout = f"{PUBLIC_RESPONSE_MARKER}\n{malformed_public_review}"
    runner = FakeRunner(gemini_outputs=[{"stdout": raw_stdout}])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        assert run_pr_loop(runner, pr_number=77, config=config) == 0

    repair_mock.assert_not_called()
    assert any("Found one issue." in comment for comment in runner.comments)
    assert all(comment.count("<!-- AGENT_STATE: approved -->") == 1 for comment in runner.comments)


def test_gemini_pre_marker_429_malformed_public_response_fails_deterministically(tmp_path):
    malformed_public_review = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "Found one issue.",
                "blocking_items": [],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": [],
            }
        )
        + "\nExtra prose before the footer.\n"
        "<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    )
    raw_stdout = (
        "Attempt 1 failed with status 429. No capacity available for model gemini.\n"
        f"{PUBLIC_RESPONSE_MARKER}\n"
        f"{malformed_public_review}"
    )
    runner = FakeRunner(gemini_outputs=[{"stdout": raw_stdout}])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value="still invalid"):
        with pytest.raises(AgentLoopError) as exc_info:
            run_pr_loop(runner, pr_number=77, config=config)

    message = str(exc_info.value)
    assert "Failure category: deterministic" in message
    assert "Failure category: transient" not in message


def test_pr_loop_does_not_retry_billing_credit_exhaustion(tmp_path):
    output = "Quota exceeded: billing credits are exhausted."
    runner = FakeRunner(gemini_outputs=[output])
    config = make_config(tmp_path, reviewer="gemini")

    with pytest.raises(AgentLoopError, match="No review result was recorded"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert runner.comments == []
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)


def test_pr_loop_does_not_retry_auth_failure(tmp_path):
    output = "Unauthorized: invalid api key provided."
    runner = FakeRunner(gemini_outputs=[output])
    config = make_config(tmp_path, reviewer="gemini")

    with pytest.raises(AgentLoopError, match="No review result was recorded"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert runner.comments == []
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)


def test_pr_loop_failure_log_distinguishes_transient_failure(tmp_path):
    rate_limit_output = "HTTP 429: rate limit exceeded."
    runner = FakeRunner(gemini_outputs=[(rate_limit_output, 1)] * 3)
    config = make_config(tmp_path, reviewer="gemini")

    with pytest.raises(AgentLoopError) as exc_info:
        run_pr_loop(runner, pr_number=77, config=config)

    message = str(exc_info.value)
    assert "transient" in message
    assert "rerun may succeed" in message


def test_pr_loop_failure_log_identifies_non_retryable(tmp_path):
    billing_output = "Your billing account has no credits remaining."
    runner = FakeRunner(gemini_outputs=[billing_output])
    config = make_config(tmp_path, reviewer="gemini")

    with pytest.raises(AgentLoopError) as exc_info:
        run_pr_loop(runner, pr_number=77, config=config)

    message = str(exc_info.value)
    assert "non-retryable" in message
    assert "credentials or billing" in message


def test_pr_loop_exits_immediately_on_long_reset_rate_limit(tmp_path):
    # "Retry-After: 3600" → 3600 s reset > 300 s threshold → must exit, not retry.
    rate_limit_output = "HTTP 429: rate limit exceeded. Retry-After: 3600"
    runner = FakeRunner(gemini_outputs=[(rate_limit_output, 1)])
    config = make_config(tmp_path, reviewer="gemini")

    with pytest.raises(QuotaResetExceededError) as exc_info:
        run_pr_loop(runner, pr_number=77, config=config)

    message = str(exc_info.value)
    assert "quota exhausted" in message.lower()
    assert "1h" in message  # 3600 s = 1h
    assert "Rerun when quota resets" in message
    # Must not have slept / retried.
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)


def test_pr_loop_exits_immediately_on_claude_session_limit_reset(tmp_path, monkeypatch):
    class FixedDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            fixed = cls(2026, 6, 3, 5, 33, 48, tzinfo=datetime.timezone.utc)
            if tz is None:
                return fixed.replace(tzinfo=None)
            return fixed.astimezone(tz)

    monkeypatch.setattr(orchestrator.datetime, "datetime", FixedDateTime)
    session_limit_output = json.dumps(
        {
            "type": "result",
            "is_error": True,
            "api_error_status": 429,
            "result": "You've hit your session limit · resets 1:30am (America/Los_Angeles)",
        }
    )
    runner = FakeRunner(claude_outputs=[(session_limit_output, 1)])
    config = make_config(tmp_path, reviewer="claude")

    with pytest.raises(QuotaResetExceededError) as exc_info:
        run_pr_loop(runner, pr_number=77, config=config)

    message = str(exc_info.value)
    assert "Claude quota exhausted" in message
    assert "2h 56m" in message
    assert "Rerun when quota resets" in message
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)


def test_pr_loop_retries_on_short_reset_rate_limit(tmp_path):
    # "Retry-After: 60" → 60 s reset ≤ 300 s threshold → retry automatically.
    rate_limit_output = "HTTP 429: rate limit exceeded. Retry-After: 60"
    valid = "LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(gemini_outputs=[(rate_limit_output, 1), valid])
    config = make_config(tmp_path, reviewer="gemini")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    sleep_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["sleep"]]
    assert len(sleep_commands) == 1


def test_pr_loop_retries_on_rate_limit_without_reset_time(tmp_path):
    # No parseable reset time → fall back to normal retry behavior.
    rate_limit_output = "HTTP 429: rate limit exceeded."
    valid = "LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(gemini_outputs=[(rate_limit_output, 1), valid])
    config = make_config(tmp_path, reviewer="gemini")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    sleep_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["sleep"]]
    assert len(sleep_commands) == 1


@pytest.mark.parametrize("text,expected_secs", [
    ("Retry-After: 3600", 3600),
    ("retry after 1800", 1800),
    ("retryDelay: '7200s'", 7200),
    ("try again in 2h 30m", 9000),
    ("try again in 45m", 2700),
    ("resets in 1h", 3600),
    ("reset in 5m", 300),
])
def test_parse_rate_limit_reset_seconds(text, expected_secs):
    assert _parse_rate_limit_reset_seconds(text) == expected_secs


def test_parse_rate_limit_reset_seconds_claude_absolute_time():
    now = datetime.datetime(2026, 6, 3, 5, 33, 48, tzinfo=datetime.timezone.utc)
    text = "You've hit your session limit · resets 1:30am (America/Los_Angeles)"

    assert _parse_rate_limit_reset_seconds(text, now_utc=now) == 10572


@pytest.mark.parametrize("text", [
    "HTTP 429: rate limit exceeded.",
    "Too many requests.",
    "quota exceeded",
])
def test_parse_rate_limit_reset_seconds_returns_none_when_unparseable(text):
    assert _parse_rate_limit_reset_seconds(text) is None


@pytest.mark.parametrize("seconds,expected", [
    (3600, "1h"),
    (7200, "2h"),
    (9000, "2h 30m"),
    (300, "5m"),
    (45, "45s"),
    (3660, "1h 1m"),
])
def test_format_reset_duration(seconds, expected):
    assert _format_reset_duration(seconds) == expected


def test_quota_reset_exceeded_error_exit_code():
    assert QuotaResetExceededError.EXIT_CODE == 3


def test_pr_loop_reinjects_blocking_item_when_human_requirement_marker_missing(tmp_path):
    # Reviewer approves without HUMAN_REQUIREMENTS_RESOLVED → synthetic blocking item,
    # loop hits max_rounds (set to 1) instead of a terminal deadlock.
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={
            "number": 77,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/77",
            "title": "Improve review prompt context",
            "headRefName": "feature/review-context",
            "baseRefName": "main",
            "headRefOid": "abc123",
            "comments": [
                {
                    "author": {"login": "maintainer"},
                    "createdAt": "2026-05-18T10:00:00Z",
                    "url": "https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                    "body": "Please use the absolute URL.\n\n-- Human Reviewer",
                }
            ],
            "reviews": [],
        },
    )
    config = make_config(
        tmp_path,
        auto_merge=True,
        test_command=("pytest", "tests/test_agent_loop.py"),
        approved_followups="summarize",
        max_rounds=1,
    )

    # The old behaviour was a terminal deadlock; now the loop continues and hits max_rounds.
    with pytest.raises(AgentLoopError, match="blocking issues after round 1"):
        run_pr_loop(runner, pr_number=77, config=config)

    commands = [cmd for cmd, _cwd in runner.commands]
    assert ["pytest", "tests/test_agent_loop.py"] not in commands
    assert ["gh", "pr", "merge", "77", "--repo", "OWNER/REPO", "--merge"] not in commands
    assert not any(comment.startswith("Approved-review future follow-ups") for comment in runner.comments)


def test_pr_loop_recovers_when_second_reviewer_includes_human_requirement_marker(tmp_path):
    # Round 1: reviewer approves without HUMAN_REQUIREMENTS_RESOLVED → blocking item injected.
    # Round 2: coder addresses it; reviewer approves with the marker → success.
    pr_payload = {
        "number": 77,
        "state": "OPEN",
        "url": "https://github.com/OWNER/REPO/pull/77",
        "title": "Improve review prompt context",
        "headRefName": "feature/review-context",
        "baseRefName": "main",
        "headRefOid": "abc123",
        "comments": [
            {
                "author": {"login": "maintainer"},
                "createdAt": "2026-05-18T10:00:00Z",
                "url": "https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                "body": "Please use the absolute URL.\n\n-- Human Reviewer",
            }
        ],
        "reviews": [],
    }
    runner = FakeRunner(
        claude_outputs=[
            # Round 2: coder addresses the re-injected blocking item and acknowledges human requirements
            "Addressed human requirements.\n"
            f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
            "### Human requirements\n"
            "- Requirement 1: used the absolute URL.\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            # Round 1: approves but forgets the marker
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            # Round 2: resolves the synthetic blocking item and acknowledges human requirements
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        pr_payload=pr_payload,
    )
    config = make_config(tmp_path, max_rounds=2)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0


def test_pr_loop_allows_approval_with_human_requirement_resolution_marker(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "LGTM.\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
        pr_payload={
            "number": 77,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/77",
            "title": "Improve review prompt context",
            "headRefName": "feature/review-context",
            "baseRefName": "main",
            "headRefOid": "abc123",
            "comments": [
                {
                    "author": {"login": "maintainer"},
                    "createdAt": "2026-05-18T10:00:00Z",
                    "url": "https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                    "body": "Please use the absolute URL.\n\n-- Human Reviewer",
                }
            ],
            "reviews": [],
        },
    )
    config = make_config(tmp_path, test_command=("pytest", "tests/test_agent_loop.py"))

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    assert ["pytest", "tests/test_agent_loop.py"] in commands


def test_pr_loop_accepts_structured_coder_followup_in_pr_round(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "coder_followup",
                    "state": "blocking",
                    "summary": "Added the requested regression test.",
                    "addressed_items": ["item-1"],
                    "remaining_items": [],
                    "addressed_item_notes": {
                        "item-1": "Added the structured coder follow-up regression case."
                    },
                    "human_requirements": {
                        "addressed_ids": [],
                        "checked_discussion_directly": False,
                    },
                    "tests_run": ["pytest tests/test_agent_loop.py -k structured_coder_followup"],
                }
            )
            + "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
        ],
        codex_outputs=[
            "Need one more regression test before merge."
            + blocking_issues("Add the structured coder follow-up regression case.")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=2)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    followup_comments = [comment for comment in runner.comments if "## Coder follow-up" in comment]
    assert len(followup_comments) == 1
    visible_followup = _strip_round_metadata(followup_comments[0])
    assert "Added the requested regression test." in visible_followup
    assert "### Addressed items\n- item-1: Blocking issue from OpenAI Codex" in visible_followup
    assert "  - Resolution: Added the structured coder follow-up regression case." in visible_followup
    assert "### Remaining items\n- None." in visible_followup
    assert (
        "### Tests run\n- pytest tests/test_agent_loop.py -k structured_coder_followup"
        in visible_followup
    )
    assert '"kind": "coder_followup"' not in visible_followup


def test_pr_loop_rejects_malformed_structured_coder_followup_before_re_review(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "coder_followup",
                    "state": "blocking",
                    "summary": "Tried to handle the feedback.",
                    "addressed_items": ["item-9"],
                    "remaining_items": [],
                    "human_requirements": {
                        "addressed_ids": [],
                        "checked_discussion_directly": False,
                    },
                }
            )
            + "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
        ],
        codex_outputs=[
            "Need one more regression test before merge."
            + blocking_issues("Add the structured coder follow-up regression case.")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer="codex",
        max_rounds=2,
        agent_max_retries=0,
    )

    with pytest.raises(
        AgentLoopError,
        match="Coder follow-up referenced unknown unresolved reviewer item IDs: item-9",
    ):
        run_pr_loop(runner, pr_number=77, config=config)


def test_reconcile_human_requirements_ack_item_surfaces_markdown_ack_blocker():
    human_requirements = (
        HumanReviewRequirement(
            source_type="PR comment",
            author="reviewer",
            created_at="2026-05-18T10:00:00Z",
            url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
            body="Please use the absolute URL.",
        ),
    )

    reconciled = _reconcile_human_requirements_ack_item(
        (),
        coder_output="Implemented fix without the extra acknowledgement.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        human_requirements=human_requirements,
        source_round=2,
    )

    assert [item.item_id for item in reconciled] == [HUMAN_REQUIREMENTS_ACK_ITEM_ID]
    assert "missing required signed human requirements marker" in reconciled[0].text


def test_reconcile_human_requirements_ack_item_clears_markdown_ack_blocker():
    human_requirements = (
        HumanReviewRequirement(
            source_type="PR comment",
            author="reviewer",
            created_at="2026-05-18T10:00:00Z",
            url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
            body="Please use the absolute URL.",
        ),
    )

    reconciled = _reconcile_human_requirements_ack_item(
        (
            UnresolvedReviewItem(
                item_id=HUMAN_REQUIREMENTS_ACK_ITEM_ID,
                reviewer="Orchestrator",
                source_round=1,
                text="Ack missing.",
                status="blocking",
            ),
        ),
        coder_output=(
            "Implemented follow-up.\n"
            f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
            "### Human requirements\n"
            "- Requirement 1: updated the URL handling.\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
        ),
        human_requirements=human_requirements,
        source_round=2,
    )

    assert reconciled == []


def test_pr_loop_revalidates_latest_coder_output_against_refreshed_human_requirements(
    tmp_path, monkeypatch
):
    runner = FakeRunner(
        claude_outputs=[
            "Implemented fix with the required acknowledgement.\n"
            f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
            "### Human requirements\n"
            "- Requirement 1: updated the URL handling.\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Blocking issue.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        pr_payload={
            "number": 77,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/77",
            "title": "Improve review prompt context",
            "headRefName": "feature/review-context",
            "baseRefName": "main",
            "headRefOid": "abc123",
            "comments": [
                {
                    "author": {"login": "maintainer"},
                    "createdAt": "2026-05-18T10:00:00Z",
                    "url": "https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                    "body": "Please use the absolute URL.\n\n-- Human Reviewer",
                }
            ],
            "reviews": [],
        },
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=2)
    metadata = PullRequestMetadata(
        number=77,
        repo="OWNER/REPO",
        title="Improve review prompt context",
        head_branch="feature/review-context",
        base_branch="main",
        head_sha="abc123",
        url="https://github.com/OWNER/REPO/pull/77",
    )
    contexts = iter(
        [
            PullRequestReviewContext(
                metadata=metadata,
                comments=(),
                human_requirements=(
                    HumanReviewRequirement(
                        source_type="PR comment",
                        author="maintainer",
                        created_at="2026-05-18T10:00:00Z",
                        url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                        body="Please use the absolute URL.",
                    ),
                ),
            ),
            PullRequestReviewContext(
                metadata=metadata,
                comments=(),
                human_requirements=(),
            ),
        ]
    )

    monkeypatch.setattr(
        "coding_review_agent_loop.orchestrator.get_pr_review_context",
        lambda *args, **kwargs: next(contexts),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    review_prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]
    assert len(review_prompts) == 2
    assert HUMAN_REQUIREMENTS_ACK_ITEM_ID not in review_prompts[1]


def test_pr_loop_routes_migration_validation_failure_through_coder_followup(tmp_path, monkeypatch):
    runner = FakeRunner(
        claude_outputs=["Fixed migration.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "LGTM again."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, test_command=("pytest", "tests/test_agent_loop.py"), max_rounds=2)
    validations = iter(
        [
            MigrationValidationResult(
                ok=False,
                message=(
                    "alembic/versions/e4f5a6b7c8d9_add_pricing.py declares `down_revision = '5d5f0e1a2b3c'`; "
                    "expected current head `402b9e8af79b`."
                ),
            ),
            MigrationValidationResult(ok=True),
        ]
    )

    monkeypatch.setattr(
        "coding_review_agent_loop.orchestrator.validate_pr_migration_topology",
        lambda *args, **kwargs: next(validations),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    coder_prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(coder_prompts) == 1
    assert "Alembic migration validation unresolved blocking item [item-1]" in coder_prompts[0]
    assert "expected current head `402b9e8af79b`" in coder_prompts[0]

    commands = runner.commands
    pytest_index = command_index(commands, ["pytest", "tests/test_agent_loop.py"])
    first_review_index = [
        index for index, (cmd, _cwd) in enumerate(commands) if cmd[:2] == ["codex", "exec"]
    ][0]
    second_review_index = [
        index for index, (cmd, _cwd) in enumerate(commands) if cmd[:2] == ["codex", "exec"]
    ][1]
    assert first_review_index < pytest_index < second_review_index


def test_review_prompt_includes_pr_metadata_and_suggested_commands(tmp_path):
    runner = FakeRunner(codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"])
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]
    assert len(prompts) == 1
    prompt = prompts[0]
    assert "PR metadata:" in prompt
    assert "- Repo: OWNER/REPO" in prompt
    assert "- PR: #77" in prompt
    assert "- Title: Improve review prompt context" in prompt
    assert "- Head branch: feature/review-context" in prompt
    assert "- Base branch: main" in prompt
    assert "- Head SHA: abc123" in prompt
    assert "Use this PR metadata as authoritative." in prompt
    assert "Do not spend time discovering the PR\nbranch." in prompt
    assert (
        "gh pr view 77 --repo OWNER/REPO --json "
        "title,body,headRefName,baseRefName,headRefOid,comments,reviews"
    ) in prompt
    assert "gh pr diff 77 --repo OWNER/REPO" in prompt
    assert "requires confirmation in non-interactive mode" in prompt
    assert "write them outside the repository checkout" in prompt
    assert "/tmp/coding-review-agent-loop/scratch/" in prompt
    assert "GitHub PR checks:" in prompt
    assert "- Overall state: passing" in prompt
    assert "- Required checks: test" in prompt
    assert "Do not say or imply that tests passed globally unless the GitHub PR checks" in prompt
    assert "ignore approved-review follow-up sections" in prompt
    assert "### Future follow-ups" not in prompt
    assert "legacy heading `### Non-blocking follow-ups`" not in prompt
    assert "verify migration topology" in prompt
    assert "Use blocking only for issues that should prevent merge." in prompt


def test_review_prompt_includes_signed_human_requirements(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "LGTM.\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
        pr_payload={
            "number": 77,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/77",
            "title": "Improve review prompt context",
            "headRefName": "feature/review-context",
            "baseRefName": "main",
            "headRefOid": "abc123",
            "comments": [
                {
                    "author": {"login": "maintainer"},
                    "createdAt": "2026-05-18T10:00:00Z",
                    "url": "https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                    "body": "Please use the absolute URL.\n\n-- Human Reviewer",
                }
            ],
            "reviews": [],
        },
    )
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "Signed Human Reviewer Requirements" in prompt
    assert "Please use the absolute URL." in prompt
    assert "Signed human reviewer requirements override AI reviewer preferences" in prompt
    assert "Verify each requirement in this set before approving." in prompt
    assert "<!-- HUMAN_REQUIREMENTS_RESOLVED -->" in prompt


def test_pr_loop_routes_failing_github_checks_through_coder_followup(tmp_path, monkeypatch):
    runner = FakeRunner(
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "Still failing upstream."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=["Investigated CI.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
    )
    config = make_config(tmp_path, max_rounds=2)
    check_states = iter(
        [
            {
                "check_runs": [
                    {"name": "tests/test_server.py", "status": "completed", "conclusion": "success"},
                    {"name": "tests/test_security.py", "status": "completed", "conclusion": "failure"},
                ]
            },
            {"check_runs": [{"name": "test", "status": "completed", "conclusion": "success"}]},
        ]
    )

    def advance_checks(*_args, **_kwargs):
        runner.pr_check_runs_payload = next(check_states)
        return original_get_pr_checks(*_args, **_kwargs)

    from coding_review_agent_loop import orchestrator as orchestrator_module

    original_get_pr_checks = orchestrator_module.get_pr_checks
    monkeypatch.setattr(orchestrator_module, "get_pr_checks", advance_checks)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert any(
        comment.startswith("GitHub PR checks are failing for PR #77.") for comment in runner.comments
    )
    followup_prompt = next(
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"]
        and "GitHub PR checks unresolved blocking item [item-1] from round 1:" in cmd[-1]
    )
    assert "Failing checks: tests/test_security.py (failure)" in followup_prompt
    assert "Do not claim global test success unless GitHub PR checks are green." in followup_prompt


def test_pr_loop_blocks_final_approval_when_github_checks_pending(tmp_path):
    runner = FakeRunner(
        codex_outputs=["Looks good locally.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_check_runs_payload={"check_runs": []},
        pr_status_payload={"state": "pending", "statuses": []},
        pr_branch_protection_payload={"contexts": ["test"]},
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="GitHub PR checks for PR #77 are pending"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert any(
        comment.startswith("GitHub PR checks are still pending for PR #77.")
        for comment in runner.comments
    )


def test_pr_loop_summarizes_approved_followups_before_pending_check_exit(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Looks good locally.\n\n### Future follow-ups\n- Add cleanup docs.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
        pr_check_runs_payload={"check_runs": []},
        pr_status_payload={"state": "pending", "statuses": []},
        pr_branch_protection_payload={"contexts": ["test"]},
    )
    config = make_config(tmp_path, approved_followups="summarize")

    with pytest.raises(AgentLoopError, match="GitHub PR checks for PR #77 are pending"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert len(runner.comments) == 3
    assert runner.comments[1].startswith("Approved-review future follow-ups for PR #77:")
    assert "- Add cleanup docs. (Codex)" in runner.comments[1]
    assert "<!-- AGENT_APPROVED_FOLLOWUPS: pr=77 head=abc123 mode=summarize -->" in runner.comments[1]
    assert runner.comments[2].startswith("GitHub PR checks are still pending for PR #77.")


def test_pr_loop_summary_marker_has_single_blank_line_before_footer_marker(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Looks good locally.\n\n### Future follow-ups\n- Add cleanup docs.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
    )
    config = make_config(tmp_path, approved_followups="summarize")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    summary = runner.comments[-1]
    assert (
        "These were mentioned in approved reviews as future work and did not block merge readiness.\n\n"
        "<!-- AGENT_APPROVED_FOLLOWUPS: pr=77 head=abc123 mode=summarize -->\n"
        "-- coding-review-agent-loop"
    ) in summary


def test_pr_loop_creates_approved_followup_issues_before_unavailable_check_exit(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Looks good locally.\n\n### Future follow-ups\n- Add cleanup docs.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
        pr_check_runs_payload={"check_runs": []},
        pr_branch_protection_returncode=1,
        pr_branch_protection_stderr="500 Internal Server Error",
        pr_check_runs_returncode=1,
        pr_check_runs_stderr="500 Internal Server Error",
        pr_status_returncode=1,
        pr_status_stderr="500 Internal Server Error",
    )
    config = make_config(tmp_path, approved_followups="issue")

    with pytest.raises(AgentLoopError, match="GitHub PR checks for PR #77 are unavailable"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert len(runner.issues) == 1
    assert runner.issues[0]["title"] == "Follow up future review note: Add cleanup docs."
    assert len(runner.comments) == 3
    assert runner.comments[1].startswith("Created approved-review future follow-up issues for PR #77:")
    assert "<!-- AGENT_APPROVED_FOLLOWUPS: pr=77 head=abc123 mode=issue -->" in runner.comments[1]
    assert runner.comments[2].startswith("GitHub PR check status is unavailable for PR #77.")


def test_pr_loop_skips_duplicate_approved_followup_issue_creation_when_marker_exists(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Codex approves.\n\n### Future follow-ups\n- Add cleanup docs.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
        pr_payload={
            "comments": [
                {
                    "author": {"login": "coding-review-agent-loop"},
                    "createdAt": "2026-05-22T10:00:00Z",
                    "body": (
                        "Created approved-review future follow-up issues for PR #77:\n\n"
                        "- https://github.com/OWNER/REPO/issues/99\n\n"
                        "These were mentioned in approved reviews as future work and did not block merge readiness.\n\n"
                        "<!-- AGENT_APPROVED_FOLLOWUPS: pr=77 head=abc123 mode=issue -->\n"
                        "-- coding-review-agent-loop"
                    ),
                }
            ]
        },
    )
    config = make_config(tmp_path, approved_followups="issue")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.issues == []
    assert runner.comments == [
        "**Review verdict:** Approved\n\n"
        "Codex approves.\n\n### Future follow-ups\n- Add cleanup docs.\n"
        "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    ]


def test_pr_loop_allows_repos_without_github_checks_when_branch_protection_404(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_check_runs_payload={"check_runs": []},
        pr_status_payload={"state": "pending", "statuses": []},
        pr_branch_protection_returncode=1,
        pr_branch_protection_stderr="404 Not Found",
    )
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert not any(comment.startswith("GitHub PR checks are") for comment in runner.comments)


def test_pr_loop_allows_repos_without_github_checks_when_branch_protection_403(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_check_runs_payload={"check_runs": []},
        pr_status_payload={"state": "pending", "statuses": []},
        pr_branch_protection_returncode=1,
        pr_branch_protection_stderr="403 Forbidden",
    )
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert not any(comment.startswith("GitHub PR checks are") for comment in runner.comments)


def test_review_prompt_includes_failing_github_check_status(tmp_path):
    runner = FakeRunner(
        codex_outputs=["Blocking.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"],
        pr_check_runs_payload={
            "check_runs": [
                {"name": "tests/test_security.py", "status": "completed", "conclusion": "failure"}
            ]
        },
        pr_branch_protection_payload={"contexts": ["tests/test_security.py"]},
    )
    config = make_config(tmp_path, max_rounds=1)

    with pytest.raises(AgentLoopError, match="blocking issues after round 1"):
        run_pr_loop(runner, pr_number=77, config=config)

    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "GitHub PR checks:" in prompt
    assert "- Overall state: failing" in prompt
    assert "- Failing checks: tests/test_security.py (failure)" in prompt


@pytest.mark.parametrize("compact_context", [False, True])
def test_review_prompt_includes_no_ci_wait_instruction(tmp_path, compact_context):
    config = make_config(tmp_path)
    prompt = build_review_prompt(77, 1, config, reviewer="codex", compact_context=compact_context)
    assert "Do not defer your review to wait for CI" in prompt
    assert "DO NOT run tests, shell commands, or compile code" in prompt


def test_review_prompt_mentions_branch_protection_forbidden_when_checks_exist(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_check_runs_payload={
            "check_runs": [{"name": "test", "status": "completed", "conclusion": "success"}]
        },
        pr_branch_protection_returncode=1,
        pr_branch_protection_stderr="403 Forbidden",
    )
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "Current GitHub token cannot inspect branch protection on the PR base branch." in prompt


def test_get_pr_checks_returns_no_checks_in_dry_run(tmp_path):
    runner = FakeRunner()
    config = make_config(tmp_path, dry_run=True)

    pr_checks = get_pr_checks(
        runner,
        config=config,
        metadata=PullRequestMetadata(
            number=77,
            repo="OWNER/REPO",
            title="Improve review prompt context",
            head_branch="feature/review-context",
            base_branch="main",
            head_sha="abc123",
            url="https://github.com/OWNER/REPO/pull/77",
        ),
    )

    assert pr_checks.state == "no_checks"
    assert pr_checks.branch_protection_status == "unavailable"
    assert pr_checks.branch_protection_note == "Dry run mode does not query live GitHub PR checks."


def test_blocking_followup_prompt_reinjects_issue_context(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Needs a fix.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            structured_coder_followup(
                state="approved",
                addressed_items=["item-1"],
                remaining_items=[],
                reviewer="Anthropic Claude",
            )
        ],
    )
    config = make_config(tmp_path)
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Support issue comments",
        body="Original request.",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(
            IssueComment(
                author="commenter",
                created_at="2026-05-17T10:00:00Z",
                body="Clarifying issue comment.",
            ),
        ),
    )

    assert run_pr_loop(runner, pr_number=77, config=config, issue_context=issue_context) == 0

    followup_prompt = next(
        cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"] and "Address the review below" in cmd[-1]
    )
    assert "Issue context from GitHub" in followup_prompt
    assert "Title:\nSupport issue comments" in followup_prompt
    assert "Clarifying issue comment." in followup_prompt
    assert "Needs a fix." in followup_prompt


def test_blocking_followup_prompt_includes_human_requirements_before_ai_feedback(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Needs a fix.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Fixed review.\n"
            f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
            "### Human requirements\n"
            "- Requirement 1: used the absolute URL.\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
        ],
        pr_payload={
            "number": 77,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/77",
            "title": "Improve review prompt context",
            "headRefName": "feature/review-context",
            "baseRefName": "main",
            "headRefOid": "abc123",
            "comments": [
                {
                    "author": {"login": "maintainer"},
                    "createdAt": "2026-05-18T10:00:00Z",
                    "url": "https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                    "body": "Please use the absolute URL.\n\n-- Human Reviewer",
                }
            ],
            "reviews": [],
        },
    )
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    followup_prompt = next(
        cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"] and "Address the review below" in cmd[-1]
    )
    assert followup_prompt.index("Signed Human Reviewer Requirements") < followup_prompt.index(
        "Codex unresolved blocking item [item-1] from round 1:"
    )
    assert "Please use the absolute URL." in followup_prompt
    assert "Needs a fix." in followup_prompt


def test_pr_loop_combines_issue_and_pr_signed_human_requirements(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "LGTM.\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
        pr_payload={
            "comments": [
                {
                    "author": {"login": "maintainer"},
                    "createdAt": "2026-05-18T10:00:00Z",
                    "url": "https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                    "body": "Use the absolute URL in the PR path.\n\n-- Human Reviewer",
                }
            ],
            "reviews": [],
        },
    )
    config = make_config(tmp_path, reviewer="codex")
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Support issue comments",
        body="Original request.",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(),
        human_requirements=(
            HumanReviewRequirement(
                source_type="Issue body",
                author="issue-author",
                created_at="2026-05-17T08:00:00Z",
                url="https://github.com/OWNER/REPO/issues/56",
                body="Preserve backward compatibility.",
            ),
        ),
    )

    assert run_pr_loop(runner, pr_number=77, config=config, issue_context=issue_context) == 0

    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "Preserve backward compatibility." in prompt
    assert "Use the absolute URL in the PR path." in prompt
    assert prompt.index("Preserve backward compatibility.") < prompt.index(
        "Use the absolute URL in the PR path."
    )


def test_review_prompt_requests_future_followups_when_processed(tmp_path):
    runner = FakeRunner(codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"])
    config = make_config(tmp_path, approved_followups="summarize")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "### Future follow-ups" in prompt
    assert "legacy heading `### Non-blocking follow-ups`" in prompt
    assert "Do not use the Same-PR follow-ups section in this mode" in prompt
    assert "Use Future follow-ups only for independent later work" in prompt
    assert "broader scaling or performance refinement\nfor very large histories" in prompt
    assert "Do not put small cleanup in touched or directly\nadjacent code under Future follow-ups" in prompt
    assert "Indentation/style cleanup in touched\ncode should be omitted unless worth requiring before merge" in prompt
    assert "duplicated helper\nor prompt wording introduced by this PR should make the review blocking" in prompt
    assert "Before returning approved, self-check that no Future follow-up is trivial or\nlocal to the current PR" in prompt
    assert "Use blocking only for issues that should prevent merge." in prompt


def test_review_prompt_allows_same_pr_followups_for_fix_modes(tmp_path):
    runner = FakeRunner(codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"])
    config = make_config(tmp_path, approved_followups="fix-and-summarize")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "### Same-PR follow-ups" in prompt
    assert "### Future follow-ups" in prompt
    assert "small, localized, low-risk cleanup" in prompt
    assert "narrow current-PR cleanup in files already\ntouched by this PR or directly adjacent code" in prompt
    assert "indentation or\nstyle cleanup in touched code" in prompt
    assert "duplicated helper or prompt wording introduced by this PR" in prompt
    assert "Use Future follow-ups only for independent later work" in prompt
    assert "broader scaling or performance refinement\nfor very large histories" in prompt
    assert "Do not put small cleanup in touched or directly\nadjacent code under Future follow-ups" in prompt
    assert "Keep `blocking_items` and `same_pr_followups` mutually exclusive." in prompt
    assert (
        "Use\n`blocking_items` for defects, missing requirements, regressions, security\n"
        "issues, or consistency gaps that make the PR not merge-ready."
    ) in prompt
    assert (
        "Use\n`same_pr_followups` only for small localized cleanup that should be handled in\n"
        "this PR but is not itself the reason the PR is blocked."
    ) in prompt
    assert "Same-PR follow-ups may appear only in blocking reviews." in prompt
    assert "will be sent back to Claude and require another review" in prompt
    assert "Approved means there are no blocking issues, no Same-PR follow-ups, and no\ncarried-forward prior unresolved items left active" in prompt
    assert "Before returning approved, self-check that no\nFuture follow-up is trivial or local to the current PR" in prompt
    assert "If you return `<!-- AGENT_STATE: blocking -->`, do not use structured Future\nfollow-ups" in prompt
    assert (
        "`blocking_items`, `same_pr_followups`, and `future_followups` have distinct\n"
        "roles."
    ) in prompt
    assert "small, localized cleanup in touched files or directly adjacent code" in prompt
    assert "`future_followups` are independent later work that remains valid after this PR\nis merge-ready." in prompt
    assert "Before approving, self-check every `future_followups` entry" in prompt
    assert (
        "`blocking_items` are merge-blocking defects, missing requirements,\n"
        "regressions, security issues, or consistency gaps."
    ) in prompt


def test_same_pr_followup_prompt_no_longer_claims_pr_was_approved(tmp_path):
    config = make_config(tmp_path)

    prompt = build_same_pr_followup_prompt(77, 2, "Rename the helper.", config)

    assert "requested same-PR follow-ups" in prompt
    assert "approved pull request" not in prompt
    assert "remains blocked pending another review round" in prompt


def test_pr_loop_keeps_blocking_review_when_future_followups_are_misclassified(tmp_path):
    runner = FakeRunner(
        gemini_outputs=[
            "Still blocked.\n\n"
            "### Same-PR follow-ups\n"
            "- Tighten the reset helper.\n\n"
            "### Future follow-ups\n"
            "- Consider a broader cleanup later.\n\n"
            "<!-- AGENT_STATE: blocking -->\n"
            "-- Google Gemini",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved", "[item-2] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- Google Gemini",
        ],
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved", "[item-2] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=["Fixed review.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
    )
    config = make_config(
        tmp_path,
        reviewer=("gemini", "codex"),
        approved_followups="fix-and-issue",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments[0].startswith("**Review verdict:** Blocking\n\nStill blocked.")
    assert "Consider a broader cleanup later." not in runner.comments[0]
    followup_prompt = next(
        cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"] and "Address the review below" in cmd[-1]
    )
    assert "Still blocked." in followup_prompt


def test_agent_memory_is_created_and_added_to_review_prompt(tmp_path):
    runner = FakeRunner(codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"])
    memory_dir = tmp_path / "memory"
    config = make_config(tmp_path, agent_memory_dir=memory_dir)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert (memory_dir / "repo-summary.md").exists()
    assert (memory_dir / "architecture-map.md").exists()
    assert (memory_dir / "module-index.json").exists()
    assert (memory_dir / "test-profile.md").exists()
    assert (memory_dir / "toolchain.json").exists()
    assert (memory_dir / "last-analyzed-commit").read_text(encoding="utf-8") == "abc123\n"
    architecture_map = (memory_dir / "architecture-map.md").read_text(encoding="utf-8")
    assert "## Top-level Layout" in architecture_map
    assert "## Python Modules" in architecture_map
    assert "## Supporting Surfaces" in architecture_map
    assert "`src/coding_review_agent_loop`" in architecture_map

    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "Agent memory context:" in prompt
    assert "Use cached repo memory and execution memory only for orientation." in prompt
    assert "inspect the actual source files and PR diff directly" in prompt
    assert "Do not search the whole filesystem for test tools." in prompt
    assert "src/coding_review_agent_loop/cli.py" in prompt


def test_agent_memory_default_parent_ignores_generated_contents(tmp_path):
    runner = FakeRunner(codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"])
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    gitignore = tmp_path / "claude" / ".agent-loop" / ".gitignore"
    assert gitignore.read_text(encoding="utf-8") == "*\n!.gitignore\n"


def test_agent_memory_does_not_ignore_custom_parent_directory(tmp_path):
    runner = FakeRunner(codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"])
    memory_dir = tmp_path / "custom-memory"
    config = make_config(tmp_path, agent_memory_dir=memory_dir)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert not (tmp_path / ".gitignore").exists()


def test_agent_memory_detects_changed_files_since_previous_commit(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        git_head="def456",
        changed_files=["src/coding_review_agent_loop/prompts.py", "tests/test_agent_loop.py"],
    )
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "last-analyzed-commit").write_text("abc123\n", encoding="utf-8")
    config = make_config(tmp_path, agent_memory_dir=memory_dir)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    diff_commands = [cmd for cmd, _cwd in runner.commands if cmd[:3] == ["git", "diff", "--name-only"]]
    assert ["git", "diff", "--name-only", "abc123..def456"] in diff_commands
    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "src/coding_review_agent_loop/prompts.py" in prompt
    assert "tests/test_agent_loop.py" in prompt
    assert (memory_dir / "last-analyzed-commit").read_text(encoding="utf-8") == "def456\n"


def test_agent_memory_logs_when_changed_file_diff_falls_back(tmp_path, capsys):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        git_head="def456",
        diff_returncode=128,
        diff_stderr="fatal: bad revision 'abc123..def456'",
    )
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "last-analyzed-commit").write_text("abc123\n", encoding="utf-8")
    config = make_config(tmp_path, agent_memory_dir=memory_dir, quiet=False)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    captured = capsys.readouterr()
    assert "Could not diff agent memory baseline abc123..def456" in captured.err
    assert "treating all tracked files as changed" in captured.err
    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "README.md" in prompt


def test_test_profile_records_provided_test_command(tmp_path):
    runner = FakeRunner(codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"])
    memory_dir = tmp_path / "memory"
    config = make_config(
        tmp_path,
        agent_memory_dir=memory_dir,
        test_command=("python", "-m", "pytest", "-q"),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    profile = (memory_dir / "test-profile.md").read_text(encoding="utf-8")
    assert "`python -m pytest -q`" in profile
    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "prefer verified test commands from the execution profile" in prompt


def test_agent_memory_can_be_disabled(tmp_path):
    runner = FakeRunner(codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"])
    memory_dir = tmp_path / "memory"
    config = make_config(tmp_path, agent_memory=False, agent_memory_dir=memory_dir)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert not memory_dir.exists()
    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "Agent memory context:" not in prompt


def test_pr_loop_requires_all_reviewers_to_approve(tmp_path):
    runner = FakeRunner(
        codex_outputs=["Codex approves.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        claude_outputs=["Claude approves.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude"],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "claude"),
        auto_merge=True,
        test_command=("pytest", "tests/test_agent_loop.py"),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    agent_commands = [cmd[:2] for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"])]
    assert agent_commands == [["codex", "exec"], ["claude", "--print"]]
    assert len(runner.comments) == 2
    commands = [cmd for cmd, _cwd in runner.commands]
    metadata_fetches = [
        cmd
        for cmd in commands
        if cmd[:3] == ["gh", "pr", "view"]
        and "--json" in cmd
        and cmd[cmd.index("--json") + 1]
        == "number,title,headRefName,baseRefName,headRefOid,url,body,comments,reviews"
    ]
    assert len(metadata_fetches) == 1
    assert ["pytest", "tests/test_agent_loop.py"] in commands
    assert ["gh", "pr", "merge", "77", "--repo", "OWNER/REPO", "--merge"] in commands


def test_pr_loop_ignores_approved_followups_by_default(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "LGTM.\n\n### Non-blocking follow-ups\n- Add cleanup docs.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
    )
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments == [
        "**Review verdict:** Approved\n\n"
        "LGTM.\n\n### Future follow-ups\n- Add cleanup docs.\n"
        "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    ]


def test_pr_loop_summarizes_approved_followups_from_multiple_reviewers(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Codex approves.\n\n### Non-blocking follow-ups\n- Add cleanup docs.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
        claude_outputs=[
            "Claude approves.\n\n### Non-blocking follow-ups\n- Add regression coverage.\n"
            "<!-- AGENT_STATE: approved -->\n-- Anthropic Claude"
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "claude"),
        approved_followups="summarize",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(runner.comments) == 3
    summary = runner.comments[-1]
    assert summary.startswith("Approved-review future follow-ups for PR #77:")
    assert "- Add cleanup docs. (Codex)" in summary
    assert "- Add regression coverage. (Claude)" in summary
    assert "future work and did not block merge readiness" in summary
    assert summary.endswith("-- coding-review-agent-loop")


def test_pr_loop_creates_issues_for_approved_followups(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Codex approves.\n\n### Non-blocking follow-ups\n- Add cleanup docs.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
        claude_outputs=[
            "Claude approves.\n\n### Non-blocking follow-ups\n- Add regression coverage.\n"
            "<!-- AGENT_STATE: approved -->\n-- Anthropic Claude"
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "claude"),
        approved_followups="issue",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(runner.comments) == 3
    assert runner.issues == [
        {
            "title": "Follow up future review note: Add cleanup docs.",
            "body": (
                "Future follow-up from approved review on PR #77.\n\n"
                "Reviewer: Codex\n\n"
                "Follow-up:\n"
                "- Add cleanup docs.\n\n"
                "Original reviewer notes:\n"
                "- Codex: Add cleanup docs.\n\n"
                "This was mentioned in an approved review as future work and did not block merge readiness."
            ),
        },
        {
            "title": "Follow up future review note: Add regression coverage.",
            "body": (
                "Future follow-up from approved review on PR #77.\n\n"
                "Reviewer: Claude\n\n"
                "Follow-up:\n"
                "- Add regression coverage.\n\n"
                "Original reviewer notes:\n"
                "- Claude: Add regression coverage.\n\n"
                "This was mentioned in an approved review as future work and did not block merge readiness."
            ),
        },
    ]
    issue_summary = runner.comments[-1]
    assert issue_summary.startswith("Created approved-review future follow-up issues for PR #77:")
    assert "- https://github.com/OWNER/REPO/issues/99" in issue_summary
    assert issue_summary.count("https://github.com/OWNER/REPO/issues/99") == 1
    assert "future work and did not block merge readiness" in issue_summary
    assert issue_summary.endswith("-- coding-review-agent-loop")


def test_pr_loop_deduplicates_approved_followup_issues_across_reviewers(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Codex approves.\n\n### Future follow-ups\n"
            "- **Remote validation**: Validate explicit workdir git remotes against the target repo.\n"
            "- Add a distinct dry-run smoke test.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
        claude_outputs=[
            "Claude approves.\n\n### Future follow-ups\n"
            "- **Remote validation**: Validate explicit workdir git remotes against the target repo.\n"
            "- Document cache cleanup behavior.\n"
            "<!-- AGENT_STATE: approved -->\n-- Anthropic Claude"
        ],
        issue_urls=[
            "https://github.com/OWNER/REPO/issues/99",
            "https://github.com/OWNER/REPO/issues/100",
            "https://github.com/OWNER/REPO/issues/101",
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "claude"),
        approved_followups="issue",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert [issue["title"] for issue in runner.issues] == [
        "Follow up future review note: **Remote validation**: Validate explicit workdir git remotes against the target repo.",
        "Follow up future review note: Add a distinct dry-run smoke test.",
        "Follow up future review note: Document cache cleanup behavior.",
    ]
    remote_body = runner.issues[0]["body"]
    assert "Reviewers:\n- Codex\n- Claude" in remote_body
    assert "Original reviewer notes:" in remote_body
    assert "- Codex: **Remote validation**" in remote_body
    assert "- Claude: **Remote validation**" in remote_body
    issue_summary = runner.comments[-1]
    assert "- https://github.com/OWNER/REPO/issues/99" in issue_summary
    assert "- https://github.com/OWNER/REPO/issues/100" in issue_summary
    assert "- https://github.com/OWNER/REPO/issues/101" in issue_summary


def test_reconcile_approved_followups_groups_semantic_duplicates_and_preserves_distinct_items():
    reconciliation = reconcile_approved_followups(
        [
            ApprovedFollowup(
                reviewer="Claude",
                text="Clarify repair-pass ownership across the flowchart and sequence diagram.",
            ),
            ApprovedFollowup(
                reviewer="Gemini",
                text="Document repair pass ownership in the flowchart and sequence diagram so the handoff is clear.",
            ),
            ApprovedFollowup(
                reviewer="Codex",
                text="Add memory freshness checks before planning starts.",
            ),
            ApprovedFollowup(
                reviewer="Claude",
                text="Add sync-before-planning coverage for reviewer workdirs.",
            ),
        ],
        issue_limit=MAX_APPROVED_FOLLOWUP_ISSUES,
    )

    assert len(reconciliation.groups) == 3
    assert reconciliation.deduplicated_count == 1
    assert reconciliation.skipped_by_cap == 0
    grouped_reviewers = [group.reviewers for group in reconciliation.groups]
    assert ("Claude", "Gemini") in grouped_reviewers
    assert any("memory freshness" in group.text for group in reconciliation.groups)
    assert any("sync-before-planning" in group.text for group in reconciliation.groups)


def test_reconcile_approved_followups_selects_more_specific_canonical_wording_and_caps():
    reconciliation = reconcile_approved_followups(
        [
            ApprovedFollowup(reviewer="Claude", text="Clarify repair-pass ownership."),
            ApprovedFollowup(
                reviewer="Gemini",
                text="Clarify repair-pass ownership in `docs/local_agent_loop.md` and the sequence diagram.",
            ),
            ApprovedFollowup(reviewer="Codex", text="Follow up two."),
            ApprovedFollowup(reviewer="Claude", text="Follow up three."),
            ApprovedFollowup(reviewer="Gemini", text="Follow up four."),
        ],
        issue_limit=3,
    )

    assert reconciliation.groups[0].text == (
        "Clarify repair-pass ownership in `docs/local_agent_loop.md` and the sequence diagram."
    )
    assert len(reconciliation.selected_groups) == 3
    assert reconciliation.skipped_by_cap == 1
    assert reconciliation.deduplicated_count == 1


def test_pr_loop_files_earlier_future_followup_not_repeated_in_final_round(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="approved",
                summary="Codex approves with a later cleanup.",
                future_followups=["Add memory freshness checks before planning starts."],
                reviewer="OpenAI Codex",
            ),
            structured_pr_review(
                state="approved",
                summary="Codex final approval.",
                prior_item_dispositions=[
                    {
                        "item_id": "item-1",
                        "disposition": "future",
                        "note": "Still useful as separate tracking.",
                    },
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
                reviewer="OpenAI Codex",
            ),
        ],
        claude_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Need one current-PR fix.",
                blocking_items=["Fix the current sync regression."],
                reviewer="Anthropic Claude",
            ),
            structured_coder_followup(
                addressed_items=["item-2"],
                remaining_items=["item-1"],
                reviewer="Anthropic Claude",
            ),
            structured_pr_review(
                state="approved",
                summary="Claude final approval.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "future", "note": "Still valid."},
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
                reviewer="Anthropic Claude",
            ),
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "claude"),
        approved_followups="issue",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(runner.issues) == 1
    assert runner.issues[0]["title"] == (
        "Follow up future review note: Add memory freshness checks before planning starts."
    )
    assert "Update from Codex: Still useful as separate tracking." in runner.issues[0]["body"]


def test_pr_loop_does_not_file_resolved_earlier_future_followup(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="approved",
                summary="Codex approves with a later cleanup.",
                future_followups=["Remove stale final-round-only follow-up handling."],
                reviewer="OpenAI Codex",
            ),
            structured_pr_review(
                state="approved",
                summary="Codex final approval.",
                prior_item_dispositions=[
                    {
                        "item_id": "item-1",
                        "disposition": "resolved",
                        "note": "Fixed in the second commit.",
                    },
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
                reviewer="OpenAI Codex",
            ),
        ],
        claude_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Need one current-PR fix.",
                blocking_items=["Fix the current sync regression."],
                reviewer="Anthropic Claude",
            ),
            structured_coder_followup(
                addressed_items=["item-2"],
                remaining_items=["item-1"],
                reviewer="Anthropic Claude",
            ),
            structured_pr_review(
                state="approved",
                summary="Claude final approval.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved", "note": "Fixed."},
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
                reviewer="Anthropic Claude",
            ),
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "claude"),
        approved_followups="issue",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.issues == []
    assert not any(comment.startswith("Created approved-review future follow-up issues") for comment in runner.comments)


def test_pr_loop_semantically_deduplicates_followup_issues_and_keeps_provenance(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="approved",
                summary="Codex approves.",
                reviewer="OpenAI Codex",
            )
        ],
        claude_outputs=[
            structured_pr_review(
                state="approved",
                summary="Claude approves.",
                future_followups=[
                    "Clarify repair-pass ownership across the flowchart and sequence diagram."
                ],
                reviewer="Anthropic Claude",
            )
        ],
        gemini_outputs=[
            structured_pr_review(
                state="approved",
                summary="Gemini approves.",
                future_followups=[
                    "Document repair pass ownership in the flowchart and sequence diagram so the handoff is clear."
                ],
                reviewer="Google Gemini",
            )
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "claude", "gemini"),
        approved_followups="issue",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(runner.issues) == 1
    body = runner.issues[0]["body"]
    assert "Reviewers:\n- Claude\n- Gemini" in body
    assert "Original reviewer notes:" in body
    assert "- Claude: Clarify repair-pass ownership" in body
    assert "- Gemini: Document repair pass ownership" in body
    assert "Reconciliation: 1 filed, 1 deduplicated, 0 skipped by cap." in runner.comments[-1]


def test_pr_loop_suppresses_followup_issue_summary_when_no_urls_returned(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Codex approves.\n\n### Future follow-ups\n- Add cleanup docs.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
        issue_urls=[None],
    )
    config = make_config(tmp_path, approved_followups="issue")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(runner.comments) == 1
    assert len(runner.issues) == 1


def test_pr_loop_creates_no_issues_without_approved_followups(tmp_path):
    runner = FakeRunner(
        codex_outputs=["Codex approves.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path, approved_followups="issue")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(runner.comments) == 1
    assert runner.issues == []


def test_pr_loop_logs_created_followup_issue_url(tmp_path, capsys):
    runner = FakeRunner(
        codex_outputs=[
            "Codex approves.\n\n### Future follow-ups\n- Add cleanup docs.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
    )
    config = make_config(tmp_path, approved_followups="issue", quiet=False)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    captured = capsys.readouterr()
    assert "Created GitHub issue: https://github.com/OWNER/REPO/issues/99" in captured.err


@pytest.mark.parametrize("mode", ["summarize", "issue"])
def test_pr_loop_treats_same_pr_followups_as_blocking_without_fix_mode(tmp_path, mode):
    runner = FakeRunner(
        codex_outputs=[
            "Codex found cleanup.\n\n"
            "### Same-PR follow-ups\n"
            "- Rename the helper before merge.\n"
            "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
        ],
    )
    config = make_config(tmp_path, approved_followups=mode, max_rounds=1)

    with pytest.raises(AgentLoopError, match="still reported blocking"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert len(runner.comments) == 1
    assert not runner.issues
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


@pytest.mark.parametrize("mode", ["summarize", "issue"])
def test_pr_loop_treats_same_pr_prose_followups_as_blocking_without_fix_mode(tmp_path, mode):
    runner = FakeRunner(
        codex_outputs=[
            "Codex found cleanup.\n\n"
            "### Same-PR follow-ups\n"
            "Rename the helper before merge.\n"
            "Keep the behavior unchanged.\n"
            "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
        ],
    )
    config = make_config(tmp_path, approved_followups=mode, max_rounds=1)

    with pytest.raises(AgentLoopError, match="still reported blocking"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert len(runner.comments) == 1
    assert not runner.issues
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def test_pr_loop_caps_approved_followup_issues(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Codex approves.\n\n### Non-blocking follow-ups\n"
            "- Follow up one.\n"
            "- Follow up two.\n"
            "- Follow up three.\n"
            "- Follow up four.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
    )
    config = make_config(tmp_path, approved_followups="issue")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert [issue["title"] for issue in runner.issues] == [
        "Follow up future review note: Follow up one.",
        "Follow up future review note: Follow up two.",
        "Follow up future review note: Follow up three.",
    ]
    assert len(runner.comments) == 2
    issue_summary = runner.comments[-1]
    assert "- https://github.com/OWNER/REPO/issues/99" in issue_summary
    assert "Skipped 1 additional item(s) to avoid issue noise" in issue_summary
    assert issue_summary.endswith("-- coding-review-agent-loop")


def test_pr_loop_fix_and_summarize_sends_same_pr_followups_to_coder_then_rereviews(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "### Same-PR follow-ups\n"
            "- Rename the helper before merge.\n"
            "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Codex approves final pass.\n\n"
            "### Future follow-ups\n"
            "- Add broader integration coverage later.\n"
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=["Renamed helper.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
    )
    config = make_config(tmp_path, approved_followups="fix-and-summarize")
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Support issue comments",
        body="Original request.",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(
            IssueComment(
                author="commenter",
                created_at="2026-05-17T10:00:00Z",
                body="Clarifying issue comment.",
            ),
        ),
    )

    assert run_pr_loop(runner, pr_number=77, config=config, issue_context=issue_context) == 0

    agent_commands = [cmd[:2] for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"])]
    assert agent_commands == [["codex", "exec"], ["claude", "--print"], ["codex", "exec"]]
    assert len(runner.comments) == 4
    followup_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "requested same-PR follow-ups" in followup_prompt
    assert "remains blocked pending another review round" in followup_prompt
    assert "Rename the helper before merge." in followup_prompt
    assert "[item-1]" in followup_prompt
    assert "Issue context from GitHub" in followup_prompt
    assert "Title:\nSupport issue comments" in followup_prompt
    assert "Clarifying issue comment." in followup_prompt
    assert "small, localized cleanup for the\ncurrent PR" in followup_prompt
    assert "Keep the change narrowly scoped to the listed items" in followup_prompt
    assert "Do not take on\nlarger redesigns or unrelated future work" in followup_prompt
    assert "Add broader integration coverage later." in runner.comments[-1]


def test_pr_loop_fix_and_issue_uses_final_round_future_followups_after_same_pr_cleanup(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "### Same-PR follow-ups\n"
            "- Tighten the validation message.\n\n"
            "### Future follow-ups\n"
            "- Stale future item from the blocking round.\n"
            "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Codex approves final pass.\n\n"
            "### Future follow-ups\n"
            "- Add a separate migration dry-run command.\n"
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=["Tightened message.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
    )
    config = make_config(tmp_path, approved_followups="fix-and-issue")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(runner.issues) == 1
    assert runner.issues[0]["title"] == "Follow up future review note: Add a separate migration dry-run command."
    assert "Stale future item from the blocking round." not in runner.issues[0]["body"]
    commands = [cmd[:3] for cmd, _cwd in runner.commands]
    assert commands.count(["gh", "issue", "create"]) == 1


def test_pr_loop_fix_and_issue_drops_blocking_round_future_followups(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "### Same-PR follow-ups\n"
            "- Tighten the validation message.\n\n"
            "### Future follow-ups\n"
            "- Stale future item from the blocking round.\n"
            "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Codex approves final pass.\n\n"
            "### Future follow-ups\n"
            "- Add a separate migration dry-run command.\n"
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=["Tightened message.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
    )
    config = make_config(tmp_path, approved_followups="fix-and-issue")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(runner.issues) == 1
    assert "Stale future item from the blocking round." not in runner.issues[0]["body"]


def test_pr_loop_fix_and_issue_uses_only_final_round_future_followups(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Codex found cleanup.\n\n"
            "### Same-PR follow-ups\n"
            "- Tighten the validation message.\n\n"
            "### Future follow-ups\n"
            "- Stale item fixed by the same-PR pass.\n"
            "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Codex approves final pass.\n\n"
            "### Future follow-ups\n"
            "- Add a separate migration dry-run command.\n"
            + prior_item_dispositions(
                "[item-1] resolved",
                "[item-2] resolved",
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=["Tightened message.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
    )
    config = make_config(tmp_path, approved_followups="fix-and-issue")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(runner.issues) == 1
    assert runner.issues[0]["title"] == "Follow up future review note: Add a separate migration dry-run command."
    assert "Stale item fixed by the same-PR pass." not in runner.issues[0]["body"]
    assert "- https://github.com/OWNER/REPO/issues/99" in runner.comments[-1]
    commands = [cmd[:3] for cmd, _cwd in runner.commands]
    assert commands.count(["gh", "issue", "create"]) == 1


def test_pr_loop_fix_and_summarize_uses_only_final_round_future_followups(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Codex found cleanup.\n\n"
            "### Same-PR follow-ups\n"
            "- Add a small assertion before merge.\n\n"
            "### Future follow-ups\n"
            "- Add Codex's larger follow-up later.\n"
            "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Codex approves final pass.\n\n"
            "### Future follow-ups\n"
            "- Add Codex's final follow-up later.\n"
            + prior_item_dispositions(
                "[item-1] resolved",
                "[item-2] resolved",
                "[item-3] resolved",
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Claude approves.\n\n"
            "### Future follow-ups\n"
            "- Add Claude's larger follow-up later.\n"
            "<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
            "Claude approves final pass.\n\n"
            "### Future follow-ups\n"
            "- Add Claude's final follow-up later.\n"
            + prior_item_dispositions(
                "[item-1] resolved",
                "[item-2] resolved",
                "[item-3] resolved",
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        gemini_outputs=["Added assertion.\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini"],
    )
    config = make_config(
        tmp_path,
        coder="gemini",
        reviewer=("codex", "claude"),
        approved_followups="fix-and-summarize",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    agent_commands = [cmd[:2] for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"], ["gemini"])]
    assert agent_commands == [
        ["codex", "exec"],
        ["claude", "--print"],
        ["gemini", "--prompt"],
        ["codex", "exec"],
        ["claude", "--print"],
    ]
    summary = runner.comments[-1]
    assert "- Add Codex's final follow-up later. (Codex)" in summary
    assert "- Add Claude's final follow-up later. (Claude)" in summary
    assert "Add Codex's larger follow-up later." not in summary
    assert "Add Claude's larger follow-up later." not in summary


def test_pr_loop_fix_and_issue_extracts_final_round_bullet_and_prose_future_followups(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Codex found cleanup.\n\n"
            "### Same-PR follow-ups\n"
            "- Tighten the validation message.\n\n"
            "### Future follow-ups\n"
            "- Stale Codex item fixed by the same-PR pass.\n"
            "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Codex approves final pass.\n\n"
            "### Future follow-ups\n"
            "- Refine token estimation for large review prompts.\n"
            + prior_item_dispositions(
                "[item-1] resolved",
                "[item-2] resolved",
                "[item-3] resolved",
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Claude approves with cleanup.\n\n"
            "### Future follow-ups\n"
            "- Stale Claude item fixed by the same-PR pass.\n"
            "<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
            "Claude approves final pass.\n\n"
            "### Future follow-ups\n"
            "The `_parse_gemini_output` helper is dead production code and could be removed\n"
            "in a future cleanup.\n\n"
            "### Same-PR follow-ups\n"
            "No same-PR follow-ups.\n"
            + prior_item_dispositions(
                "[item-1] resolved",
                "[item-2] resolved",
                "[item-3] resolved",
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        gemini_outputs=["Tightened message.\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini"],
        issue_urls=[
            "https://github.com/OWNER/REPO/issues/99",
            "https://github.com/OWNER/REPO/issues/100",
        ],
    )
    config = make_config(
        tmp_path,
        coder="gemini",
        reviewer=("codex", "claude"),
        approved_followups="fix-and-issue",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.issues[0]["title"] == (
        "Follow up future review note: Refine token estimation for large review prompts."
    )
    assert runner.issues[1]["title"].startswith(
        "Follow up future review note: The `_parse_gemini_output` helper is dead production code"
    )
    assert "could be removed in a future cleanup." in runner.issues[1]["body"]
    assert "Stale Codex item fixed by the same-PR pass." not in runner.issues[0]["body"]
    assert "Stale Claude item fixed by the same-PR pass." not in runner.issues[1]["body"]
    issue_summary = runner.comments[-1]
    assert "- https://github.com/OWNER/REPO/issues/99" in issue_summary
    assert "- https://github.com/OWNER/REPO/issues/100" in issue_summary
    assert "Stale Codex item fixed by the same-PR pass." not in issue_summary


def test_pr_loop_reruns_all_reviewers_when_any_reviewer_blocks(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Needs a regression test.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Addressed review.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Claude approves."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Codex approves first pass.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "Codex approves second pass."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer=("claude", "codex"))

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(runner.comments) == 5
    followup_prompt = next(
        cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"] and "Address the review below" in cmd[-1]
    )
    assert "Needs a regression test." in followup_prompt
    assert "Codex approves first pass." not in followup_prompt
    commands = [cmd for cmd, _cwd in runner.commands]
    metadata_fetches = [
        cmd
        for cmd in commands
        if cmd[:3] == ["gh", "pr", "view"]
        and "--json" in cmd
        and cmd[cmd.index("--json") + 1]
        == "number,title,headRefName,baseRefName,headRefOid,url,body,comments,reviews"
    ]
    assert len(metadata_fetches) == 2


def test_pr_loop_rejects_cross_reviewer_approval_without_prior_item_disposition(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Needs a regression test.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Claude resolves it."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Codex approves first pass.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "Codex approves second pass.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        gemini_outputs=["Implemented fix.\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini"],
    )
    config = make_config(tmp_path, coder="gemini", reviewer=("claude", "codex"), max_rounds=2)

    with pytest.raises(AgentLoopError, match="did not evaluate all prior unresolved items: item-1"):
        run_pr_loop(runner, pr_number=77, config=config)

    second_codex_prompt = [
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:2] == ["codex", "exec"] and "round 2" in cmd[-1]
    ][0]
    assert "Prior unresolved review items from earlier rounds" in second_codex_prompt
    assert "[item-1] blocking from Claude in round 1" in second_codex_prompt


def test_pr_loop_can_downgrade_prior_blocker_to_future_followup_only_in_approved_review(tmp_path):
    runner = FakeRunner(
        claude_outputs=["Addressed review.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
        codex_outputs=[
            "Missing docs cleanup.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM now."
            + prior_item_dispositions("[item-1] future follow-up: cleanup can wait")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, approved_followups="summarize", max_rounds=2)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    summary = runner.comments[-1]
    assert summary.startswith("Approved-review future follow-ups for PR #77:")
    assert "Missing docs cleanup." in summary


def test_pr_loop_persists_downgraded_future_followup_across_later_blocking_rounds(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Still blocked.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "LGTM."
            + prior_item_dispositions("[item-1] future follow-up: cleanup can wait", "[item-2] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Missing docs cleanup.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Implemented fix for Claude.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Looks good."
            + prior_item_dispositions("[item-1] future follow-up: cleanup can wait", "[item-2] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "claude"),
        coder="codex",
        approved_followups="summarize",
        max_rounds=2,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    summary = runner.comments[-1]
    assert summary.startswith("Approved-review future follow-ups for PR #77:")
    assert "Missing docs cleanup." in summary


def test_pr_loop_finalized_future_followup_summary_preserves_disposition_notes(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Still blocked.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Claude approves final pass."
            + prior_item_dispositions(
                "[item-1] future follow-up: cleanup can wait until after rollout",
                "[item-2] resolved",
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Missing docs cleanup.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Implemented blocker.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Codex approves final pass."
            + prior_item_dispositions(
                "[item-1] future follow-up: cleanup can wait until after rollout",
                "[item-2] resolved",
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "claude"),
        coder="codex",
        approved_followups="summarize",
        max_rounds=2,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    summary = runner.comments[-1]
    assert "Missing docs cleanup." in summary
    assert "Update from Codex: cleanup can wait until after rollout" in summary


def test_pr_loop_carries_new_future_followups_into_later_reviewer_prompts(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Codex approves with future work.\n\n"
            "### Future follow-ups\n"
            "- Document cache cleanup behavior.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "Codex approves final pass."
            + prior_item_dispositions(
                "[item-1] future follow-up: still future work",
                "[item-2] resolved",
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Claude still blocks.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Claude approves final pass."
            + prior_item_dispositions(
                "[item-1] future follow-up: still future work",
                "[item-2] resolved",
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        gemini_outputs=["Implemented blocker.\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini"],
    )
    config = make_config(
        tmp_path,
        coder="gemini",
        reviewer=("codex", "claude"),
        approved_followups="summarize",
        max_rounds=2,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    second_claude_prompt = [
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] and "round 2" in cmd[-1]
    ][0]
    assert "Document cache cleanup behavior." in second_claude_prompt
    assert "[item-1] future" in second_claude_prompt
    summary = runner.comments[-1]
    assert summary.startswith("Approved-review future follow-ups for PR #77:")
    assert "Document cache cleanup behavior." in summary


def test_pr_loop_compact_review_mode_uses_fresh_sessions_and_compact_prior_ledger(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Codex approves with future work.\n\n"
            "### Future follow-ups\n"
            "- Document cache cleanup behavior.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "Codex approves final pass."
            + prior_item_dispositions(
                "[item-1] future follow-up: still future work",
                "[item-2] resolved",
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Claude still blocks.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Claude approves final pass."
            + prior_item_dispositions(
                "[item-1] future follow-up: still future work",
                "[item-2] resolved",
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        gemini_outputs=[
            structured_coder_followup(
                state="blocking",
                summary="Implemented blocker and ran focused tests.",
                addressed_items=["item-1", "item-2"],
                remaining_items=[],
                tests_run=["python -m pytest tests/test_agent_loop.py -k compact_pr"],
                reviewer="Google Gemini",
            )
        ],
        pr_payload={"body": "PR body used by compact review mode."},
    )
    config = make_config(
        tmp_path,
        coder="gemini",
        reviewer=("codex", "claude"),
        approved_followups="summarize",
        max_rounds=2,
        pr_review_context_mode="compact",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    codex_prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]
    assert len(codex_prompts) == 2
    second_codex_prompt = codex_prompts[1]
    assert COMPACT_PR_REVIEW_VOLATILE_TAIL_MARKER in second_codex_prompt
    assert "PR body used by compact review mode." in second_codex_prompt
    assert "Implemented blocker and ran focused tests." in second_codex_prompt
    assert "python -m pytest tests/test_agent_loop.py -k compact_pr" in second_codex_prompt
    assert "Document cache cleanup behavior." not in second_codex_prompt
    assert "[item-1] future" not in second_codex_prompt
    assert "Claude still blocks." in second_codex_prompt
    assert not any("--resume" in cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])

    assert runner.comments[-1].startswith("Approved-review future follow-ups for PR #77:")
    assert "Document cache cleanup behavior." in runner.comments[-1]


def test_pr_loop_carries_prior_item_notes_without_creating_duplicate_blocker_items(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Needs regression coverage.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Still blocked."
            + prior_item_dispositions("[item-1] still blocking: include API error path too")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Looks good."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Added coverage.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Expanded coverage.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, max_rounds=3)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    second_coder_prompt = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]][1]
    assert "Latest reviewer updates:" in second_coder_prompt
    assert "Codex: include API error path too" in second_coder_prompt
    assert "[item-2]" not in second_coder_prompt


def test_pr_loop_posts_human_readable_item_labels_in_new_and_prior_sections(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Implemented the requested PR body change.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "### Same-PR follow-ups\n"
            "- Require source issue reference in PR body.\n"
            "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Looks good."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", approved_followups="fix-and-summarize", max_rounds=2)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments[0] == (
        "**Review verdict:** Blocking\n\n"
        "### Same-PR follow-ups\n"
        "- Require source issue reference in PR body.\n"
        "<!-- AGENT_STATE: blocking -->\n"
        "-- OpenAI Codex"
    )
    assert runner.comments[2] == (
        "**Review verdict:** Approved\n\n"
        "Looks good.\n\n"
        "### Prior unresolved item dispositions\n"
        "- [item-1] Same-PR follow-up from OpenAI Codex, round 1: Require source issue reference in PR body. -> resolved\n"
        "<!-- AGENT_STATE: approved -->\n"
        "-- OpenAI Codex"
    )


def test_pr_loop_tracks_only_summary_when_blocking_items_phrase_the_issue_differently(tmp_path):
    runner = FakeRunner(
        claude_outputs=["Implemented fixes.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
        codex_outputs=[
            "Needs one more regression test before merge."
            + blocking_issues("Add the mixed-history resume case to `tests/test_agent_loop.py`.")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Looks good."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=2)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    second_coder_prompt = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]][0]
    assert "Needs one more regression test before merge." in second_coder_prompt
    assert "Add the mixed-history resume case" not in second_coder_prompt
    assert runner.comments[0] == (
        "**Review verdict:** Blocking\n\n"
        "Needs one more regression test before merge.\n\n"
        "### Blocking issues\n"
        "- Add the mixed-history resume case to `tests/test_agent_loop.py`.\n"
        "<!-- AGENT_STATE: blocking -->\n"
        "-- OpenAI Codex"
    )


def test_resume_pr_round_reparses_orchestrator_rendered_blocking_issues_comment():
    carried_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Need one more regression test before merge.",
        status="blocking",
        source_status="blocking",
    )
    rendered_review = _render_public_pr_review_comment(
        parse_review(
            "Need one more regression test before merge."
            + blocking_issues("Exercise the structured-resume path.")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            reviewer="OpenAI Codex",
        ),
        reviewer="Codex",
        human_requirements_resolved_flag=False,
        prior_items=(),
        dispositions=(),
    )
    review_comment = _attach_round_metadata(
        rendered_review,
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=2,
            subject="abc123",
            prior_items=(carried_item,),
            dispositions=(),
            new_items=(),
            state="blocking",
        ),
    )
    coder_comment = _attach_round_metadata(
        "Addressed the review.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="abc123",
            prior_items=(carried_item,),
        ),
    )

    resumed = _resume_pr_round(
        [
            IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=coder_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:01:00Z", body=review_comment),
        ],
        head_sha="abc123",
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    resumed_review = parse_review(resumed.completed_reviews[0].body, reviewer="Codex")
    assert [item.text for item in resumed_review.blocking_items] == [
        "Exercise the structured-resume path."
    ]
    assert resumed_review.summary == "Need one more regression test before merge."


def test_resume_pr_round_prefers_structured_coder_followup_metadata():
    carried_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Need one more regression test before merge.",
        status="blocking",
        source_status="blocking",
    )
    raw_structured_followup = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Added the requested regression test.",
                "addressed_items": ["item-1"],
                "remaining_items": [],
                "human_requirements": {
                    "addressed_ids": ["Requirement 1"],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    parsed = validate_structured_coder_followup(raw_structured_followup)
    assert parsed is not None
    public_comment = _render_public_coder_followup_comment(parsed, agent="Claude")
    coder_comment = _attach_round_metadata(
        public_comment,
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="abc123",
            prior_items=(carried_item,),
            raw_structured_coder_response=raw_structured_followup,
        ),
    )

    resumed = _resume_pr_round(
        [IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=coder_comment)],
        head_sha="abc123",
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    assert resumed.coder_output == raw_structured_followup
    resumed_followup = validate_structured_coder_followup(resumed.coder_output)
    assert resumed_followup is not None
    assert resumed_followup.human_requirements.addressed_ids == ("Requirement 1",)
    assert '"kind": "coder_followup"' not in _strip_round_metadata(coder_comment)


def test_resume_pr_round_marks_empty_ledger_incomplete_after_same_subject_prior_new_items():
    prior_new_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Prior same-head item.",
        status="blocking",
    )
    prior_review_comment = _attach_round_metadata(
        "Prior review.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject="abc123",
            prior_items=(),
            new_items=(prior_new_item,),
            state="blocking",
        ),
    )
    current_coder_comment = _attach_round_metadata(
        "Current coder output.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="abc123",
            prior_items=(),
        ),
    )

    resumed = _resume_pr_round(
        [
            IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=prior_review_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:01:00Z", body=current_coder_comment),
        ],
        head_sha="abc123",
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    assert resumed.ledger_may_be_incomplete is True


def test_resume_pr_round_does_not_mark_ledger_incomplete_for_cross_subject_prior_new_items():
    prior_new_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Prior other-head item.",
        status="blocking",
    )
    prior_review_comment = _attach_round_metadata(
        "Prior review.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject="old-sha",
            prior_items=(),
            new_items=(prior_new_item,),
            state="blocking",
        ),
    )
    current_coder_comment = _attach_round_metadata(
        "Current coder output.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=1,
            subject="new-sha",
            prior_items=(),
        ),
    )

    resumed = _resume_pr_round(
        [
            IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=prior_review_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:01:00Z", body=current_coder_comment),
        ],
        head_sha="new-sha",
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    assert resumed.ledger_may_be_incomplete is False


def test_resume_pr_round_recovers_unrecorded_head_advance_reviewer_new_item():
    active_item = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="Google Gemini",
        source_round=1,
        text="Fix the regression before merge.",
        status="blocking",
    )
    coder_comment = _attach_round_metadata(
        "Initial PR handoff.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=1,
            subject="old-sha",
            prior_items=(),
        ),
    )
    review_comment = _attach_round_metadata(
        "Blocked.\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Gemini",
            round_number=1,
            subject="old-sha",
            prior_items=(),
            new_items=(active_item,),
            state="blocking",
        ),
    )

    resumed = _resume_pr_round(
        [
            IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=coder_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:01:00Z", body=review_comment),
        ],
        head_sha="new-sha",
        configured_reviewers=("gemini",),
    )

    assert resumed is not None
    assert resumed.unrecorded_head_advance is True
    assert resumed.ledger_may_be_incomplete is True
    assert resumed.round_number == 1
    assert resumed.completed_reviews == ()
    assert [item.item_id for item in resumed.prior_items] == ["item-2"]
    assert resumed.next_unresolved_item_number == 3


def test_resume_pr_round_recovers_coder_only_unrecorded_head_advance():
    carried_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Still needs a targeted test.",
        status="same-pr",
    )
    future_item = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Document this later.",
        status="future",
    )
    coder_comment = _attach_round_metadata(
        "Addressed prior feedback.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="old-sha",
            prior_items=(carried_item, future_item),
            compact_prior_summaries=("Older summary.",),
        ),
    )

    resumed = _resume_pr_round(
        [IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=coder_comment)],
        head_sha="new-sha",
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    assert resumed.unrecorded_head_advance is True
    assert resumed.round_number == 2
    assert [item.item_id for item in resumed.prior_items] == ["item-1"]
    assert resumed.compact_prior_summaries == ("Older summary.",)


def test_resume_pr_round_recovers_reviewer_only_with_aggregated_dispositions():
    prior_blocking = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Fix the flaky test.",
        status="blocking",
    )
    prior_same_pr = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="Google Gemini",
        source_round=1,
        text="Tighten the docs.",
        status="same-pr",
    )
    future_new_item = UnresolvedReviewItem(
        item_id="item-3",
        reviewer="OpenAI Codex",
        source_round=2,
        text="Follow up in another PR.",
        status="future",
    )
    active_new_item = UnresolvedReviewItem(
        item_id="item-4",
        reviewer="Google Gemini",
        source_round=2,
        text="Add one same-PR assertion.",
        status="same-pr",
    )
    codex_resolution = ReviewItemDisposition(
        item_id="item-1",
        reviewer="OpenAI Codex",
        disposition="resolved",
        note=None,
    )
    gemini_same_pr = ReviewItemDisposition(
        item_id="item-2",
        reviewer="Google Gemini",
        disposition="same-pr",
        note="Still needed before merge.",
    )
    codex_comment = _attach_round_metadata(
        "Codex review.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=2,
            subject="old-sha",
            prior_items=(prior_blocking, prior_same_pr),
            dispositions=(codex_resolution,),
            new_items=(future_new_item,),
            state="approved",
        ),
    )
    gemini_comment = _attach_round_metadata(
        "Gemini review.\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Gemini",
            round_number=2,
            subject="old-sha",
            prior_items=(prior_blocking, prior_same_pr),
            dispositions=(gemini_same_pr,),
            new_items=(active_new_item,),
            state="blocking",
        ),
    )

    resumed = _resume_pr_round(
        [
            IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=codex_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:01:00Z", body=gemini_comment),
        ],
        head_sha="new-sha",
        configured_reviewers=("codex", "gemini"),
    )

    assert resumed is not None
    assert resumed.unrecorded_head_advance is True
    assert [item.item_id for item in resumed.prior_items] == ["item-2", "item-4"]
    assert resumed.prior_items[0].status == "same-pr"
    assert "Still needed before merge." in resumed.prior_items[0].text


def test_resume_pr_round_ignores_unrecorded_head_advance_with_no_active_items():
    future_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Future cleanup.",
        status="future",
    )
    review_comment = _attach_round_metadata(
        "Approved with future follow-up.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject="old-sha",
            prior_items=(),
            new_items=(future_item,),
            state="approved",
        ),
    )

    assert (
        _resume_pr_round(
            [IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=review_comment)],
            head_sha="new-sha",
            configured_reviewers=("codex",),
        )
        is None
    )


def test_resume_pr_round_fails_early_for_incoherent_unrecorded_head_advance():
    bad_comment = _attach_round_metadata(
        "Bad metadata.\n<!-- AGENT_STATE: blocking -->\n-- Bot",
        PostedRoundMetadata(
            flow="pr",
            role="observer",
            agent="Bot",
            round_number=1,
            subject="old-sha",
        ),
    )

    with pytest.raises(
        AgentLoopError,
        match=(
            "PR head advanced without a recorded coder follow-up.*"
            "Current head: new-sha.*Latest recorded metadata subject: old-sha"
        ),
    ):
        _resume_pr_round(
            [IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=bad_comment)],
            head_sha="new-sha",
            configured_reviewers=("codex",),
        )


def test_resume_plan_round_marks_empty_ledger_incomplete_after_same_subject_prior_new_items():
    plan = "Plan text."
    subject = _plan_subject(plan)
    prior_new_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Prior same-plan item.",
        status="blocking",
    )
    prior_review_comment = _attach_round_metadata(
        "Prior plan review.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject=subject,
            prior_items=(),
            new_items=(prior_new_item,),
            state="blocking",
        ),
    )
    current_coder_comment = _attach_round_metadata(
        plan + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=subject,
            prior_items=(),
            canonical_plan=plan,
        ),
    )

    resumed = _resume_plan_round(
        [
            IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=prior_review_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:01:00Z", body=current_coder_comment),
        ],
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    assert resumed[1].ledger_may_be_incomplete is True


def test_resume_pr_round_prefers_latest_metadata_ledger_for_same_head_replay():
    stale_item = UnresolvedReviewItem(
        item_id="item-3",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Stale replay item.",
        status="blocking",
        source_status="blocking",
    )
    active_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Active replay item.",
        status="blocking",
        source_status="blocking",
    )
    stale_coder_comment = _attach_round_metadata(
        "Stale replay.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="abc123",
            prior_items=(stale_item,),
        ),
    )
    stale_reviewer_comment = _attach_round_metadata(
        "Still blocked."
        + prior_item_dispositions("[item-3] still blocking: stale replay")
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=2,
            subject="abc123",
            prior_items=(stale_item,),
            dispositions=(
                parse_unresolved_item_dispositions(
                    prior_item_dispositions("[item-3] still blocking: stale replay"),
                    reviewer="OpenAI Codex",
                )[0],
            ),
            state="blocking",
        ),
    )
    active_coder_comment = _attach_round_metadata(
        "Current replay.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="abc123",
            prior_items=(active_item,),
        ),
    )
    active_reviewer_comment = _attach_round_metadata(
        "Looks good."
        + prior_item_dispositions("[item-1] resolved")
        + "\n<!-- AGENT_STATE: approved -->\n-- Google Gemini",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Gemini",
            round_number=2,
            subject="abc123",
            prior_items=(active_item,),
            dispositions=(
                parse_unresolved_item_dispositions(
                    prior_item_dispositions("[item-1] resolved"),
                    reviewer="Google Gemini",
                )[0],
            ),
            state="approved",
        ),
    )
    previous_head_comment = _attach_round_metadata(
        "Older head.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=4,
            subject="old-head",
            prior_items=(
                UnresolvedReviewItem(
                    item_id="item-9",
                    reviewer="OpenAI Codex",
                    source_round=3,
                    text="Older head item.",
                    status="blocking",
                ),
            ),
        ),
    )

    resumed = _resume_pr_round(
        [
            IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=previous_head_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:01:00Z", body=stale_coder_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:02:00Z", body=stale_reviewer_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:03:00Z", body=active_coder_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:04:00Z", body=active_reviewer_comment),
        ],
        head_sha="abc123",
        configured_reviewers=("codex", "gemini"),
    )

    assert resumed is not None
    assert [item.item_id for item in resumed.prior_items] == ["item-1"]
    assert resumed.next_unresolved_item_number == 4
    assert [record.metadata.agent for record in resumed.completed_reviews] == ["Gemini"]


def test_pr_loop_resume_hybrid_history_prefers_metadata_ledger_over_legacy_markdown(tmp_path):
    carried_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Add a regression test before merge.",
        status="blocking",
        source_status="blocking",
    )
    legacy_comment = (
        "Legacy raw markdown review.\n\n"
        "### Blocking issues\n"
        "- Keep the legacy fallback path.\n"
        "<!-- AGENT_STATE: blocking -->\n"
        "-- OpenAI Codex"
    )
    coder_comment = _attach_round_metadata(
        "Updated the PR.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="abc123",
            prior_items=(carried_item,),
        ),
    )
    codex_comment = _attach_round_metadata(
        "Looks good."
        + prior_item_dispositions("[item-1] resolved")
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=2,
            subject="abc123",
            prior_items=(carried_item,),
            dispositions=(
                parse_unresolved_item_dispositions(
                    prior_item_dispositions("[item-1] resolved"),
                    reviewer="OpenAI Codex",
                )[0],
            ),
            state="approved",
        ),
    )
    runner = FakeRunner(
        gemini_outputs=[
            "Ship it."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
        ],
        pr_payload={
            "comments": [
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": legacy_comment},
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:05:00Z", "body": coder_comment},
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:06:00Z", "body": codex_comment},
            ],
        },
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    gemini_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["gemini"])
    assert "[item-1]" in gemini_prompt
    assert "Add a regression test before merge." in gemini_prompt
    assert "Keep the legacy fallback path." not in gemini_prompt


def test_pr_loop_routes_unrecorded_head_advance_through_coder_before_reviewers(tmp_path):
    old_item = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="Google Gemini",
        source_round=1,
        text="Preserve the metadata-backed unresolved item on rerun.",
        status="blocking",
    )
    old_coder_comment = _attach_round_metadata(
        "Opened the PR.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=1,
            subject="old-sha",
            prior_items=(),
        ),
    )
    old_review_comment = _attach_round_metadata(
        "Blocked.\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Gemini",
            round_number=1,
            subject="old-sha",
            prior_items=(),
            new_items=(old_item,),
            state="blocking",
        ),
    )
    runner = FakeRunner(
        claude_outputs=[
            structured_coder_followup(
                summary="Addressed the recovered prior item.",
                addressed_items=["item-2"],
                tests_run=["python -m pytest tests/test_agent_loop.py -k unrecorded_head"],
            )
        ],
        codex_outputs=[
            structured_pr_review(
                state="approved",
                summary="Recovered item is resolved.",
                prior_item_dispositions=[
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
            )
        ],
        pr_payload={
            "headRefOid": "new-sha",
            "comments": [
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": old_coder_comment},
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:01:00Z", "body": old_review_comment},
            ],
        },
    )
    config = make_config(tmp_path, reviewer="codex")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    first_coder = command_index(runner.commands, ["claude"])
    first_reviewer = command_index(runner.commands, ["codex", "exec"])
    assert first_coder < first_reviewer
    reviewer_prompt = runner.commands[first_reviewer][0][-1]
    assert "[item-2]" in reviewer_prompt
    assert "Preserve the metadata-backed unresolved item on rerun." in reviewer_prompt
    posted_coder_comment = next(
        comment["body"]
        for comment in runner.pr_payload["comments"]
        if "## Coder follow-up" in comment["body"]
    )
    match = re.search(r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", posted_coder_comment)
    assert match is not None
    metadata = _decode_round_metadata(match.group("payload"))
    assert metadata.subject == "new-sha"
    assert metadata.round_number == 2
    assert [item.item_id for item in metadata.prior_items] == ["item-2"]


def test_pr_loop_unrecorded_head_advance_prevents_empty_ledger_unknown_item_abort(tmp_path):
    old_item = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="Google Gemini",
        source_round=1,
        text="Carry this item instead of starting an empty ledger.",
        status="blocking",
    )
    old_review_comment = _attach_round_metadata(
        "Blocked.\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Gemini",
            round_number=1,
            subject="old-sha",
            prior_items=(),
            new_items=(old_item,),
            state="blocking",
        ),
    )
    runner = FakeRunner(
        claude_outputs=[
            structured_coder_followup(
                summary="Classified the recovered item.",
                addressed_items=["item-2"],
            )
        ],
        codex_outputs=[
            structured_pr_review(
                state="approved",
                summary="Old item is resolved.",
                prior_item_dispositions=[
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
            )
        ],
        pr_payload={
            "headRefOid": "new-sha",
            "comments": [
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:01:00Z", "body": old_review_comment},
            ],
        },
    )
    config = make_config(tmp_path, reviewer="codex")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert runner.claude_outputs == []
    assert runner.codex_outputs == []
    assert not any("unknown item" in comment.lower() for comment in runner.comments)


def test_reconcile_human_requirements_ack_item_accepts_stored_structured_coder_followup():
    human_requirements = (
        HumanReviewRequirement(
            source_type="PR comment",
            author="reviewer",
            created_at="2026-05-18T10:00:00Z",
            url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
            body="Please use the absolute URL.",
        ),
    )
    structured_followup = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Implemented the requested URL fix.",
                "addressed_items": ["item-1"],
                "remaining_items": [],
                "human_requirements": {
                    "addressed_ids": ["Requirement 1"],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )

    reconciled = _reconcile_human_requirements_ack_item(
        (
            UnresolvedReviewItem(
                item_id=HUMAN_REQUIREMENTS_ACK_ITEM_ID,
                reviewer="Orchestrator",
                source_round=1,
                text="Ack missing.",
                status="blocking",
            ),
        ),
        coder_output=structured_followup,
        human_requirements=human_requirements,
        source_round=2,
    )

    assert reconciled == []


def test_pr_loop_does_not_expose_same_round_item_ids_to_later_reviewers(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "### Same-PR follow-ups\n"
            "- Require source issue reference in PR body.\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Still blocked.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(
        tmp_path,
        coder="gemini",
        reviewer=("claude", "codex"),
        approved_followups="fix-and-summarize",
        max_rounds=1,
    )

    with pytest.raises(AgentLoopError, match="still reported blocking issues after round 1"):
        run_pr_loop(runner, pr_number=77, config=config)

    second_reviewer_prompt = [
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:2] == ["codex", "exec"] and "round 1" in cmd[-1]
    ][0]
    assert "Only items listed under `Prior unresolved review items from earlier rounds`" in second_reviewer_prompt
    assert "[item-1]" not in second_reviewer_prompt
    assert "### New tracked unresolved items" not in runner.comments[0]


def test_pr_loop_same_pr_items_remain_blocking_until_explicitly_resolved(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "### Same-PR follow-ups\n"
            "- Rename the helper before merge.\n"
            "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Codex still wants the rename."
            + prior_item_dispositions("[item-1] same-pr")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
        claude_outputs=["Tried a partial fix.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
    )
    config = make_config(tmp_path, approved_followups="fix-and-summarize", max_rounds=2)

    with pytest.raises(AgentLoopError, match="still reported blocking issues after round 2"):
        run_pr_loop(runner, pr_number=77, config=config)


def test_pr_loop_resumes_with_only_missing_reviewer_for_current_head(tmp_path):
    carried_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="Codex",
        source_round=1,
        text="Add a regression test before merge.",
        status="blocking",
        source_status="blocking",
    )
    coder_comment = _attach_round_metadata(
        "Updated the PR with the requested fix.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="abc123",
            prior_items=(carried_item,),
        ),
    )
    codex_comment = _attach_round_metadata(
        "Looks good."
        + prior_item_dispositions("[item-1] resolved")
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=2,
            subject="abc123",
            prior_items=(carried_item,),
            dispositions=(
                parse_unresolved_item_dispositions(
                    prior_item_dispositions("[item-1] resolved"),
                    reviewer="OpenAI Codex",
                )[0],
            ),
            state="approved",
        ),
    )
    runner = FakeRunner(
        gemini_outputs=[
            "Ship it."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
        ],
        pr_payload={
            "comments": [
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T10:00:00Z", "body": coder_comment},
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T10:05:00Z", "body": codex_comment},
            ],
        },
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    agent_commands = [cmd[0] for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"], ["gemini"])]
    assert agent_commands == ["gemini"]
    gemini_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["gemini"])
    assert "[item-1]" in gemini_prompt
    assert "Add a regression test before merge." in gemini_prompt


def test_pr_loop_resume_raises_agent_loop_error_for_missing_reconstructed_prior_item(tmp_path):
    carried_item = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Actual active carried item.",
        status="blocking",
        source_status="blocking",
    )
    coder_comment = _attach_round_metadata(
        "Updated the PR.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="abc123",
            prior_items=(carried_item,),
        ),
    )
    invalid_disposition = ReviewItemDisposition(
        item_id="item-1",
        reviewer="OpenAI Codex",
        disposition="resolved",
    )
    codex_comment = _attach_round_metadata(
        "Looks good."
        + prior_item_dispositions("[item-1] resolved")
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=2,
            subject="abc123",
            prior_items=(carried_item,),
            dispositions=(invalid_disposition,),
            state="approved",
        ),
    )
    runner = FakeRunner(
        pr_payload={
            "comments": [
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T10:00:00Z", "body": coder_comment},
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T10:05:00Z", "body": codex_comment},
            ],
        },
    )
    config = make_config(tmp_path, reviewer=("codex",))

    with pytest.raises(
        AgentLoopError,
        match=r"Resumed pr round 2 reconstructed prior items item-2, but Codex dispositioned unknown item `item-1`",
    ):
        run_pr_loop(runner, pr_number=77, config=config)


@pytest.mark.parametrize(
    "line",
    [
        "[item-1] same-pr: none",
        "[item-1] still blocking: none",
        "[item-1] future follow-up: none",
    ],
)
def test_pr_loop_rejects_contradictory_disposition_before_extra_coder_round(tmp_path, line):
    runner = FakeRunner(
        codex_outputs=[
            "Needs regression coverage.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Looks good overall."
            + prior_item_dispositions(line)
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=["Added coverage.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
    )
    config = make_config(tmp_path, approved_followups="fix-and-summarize", max_rounds=3)

    with pytest.raises(AgentLoopError, match="use `resolved` when nothing remains"):
        run_pr_loop(runner, pr_number=77, config=config)

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 1


def test_pr_loop_does_not_run_claude_after_final_blocking_round(tmp_path):
    runner = FakeRunner(codex_outputs=["Still blocked.\n<!-- AGENT_STATE: blocking -->"])
    config = make_config(tmp_path, max_rounds=1)

    with pytest.raises(AgentLoopError, match="still reported blocking"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def test_shared_workdir_requires_explicit_override(tmp_path):
    runner = FakeRunner()
    shared = tmp_path / "repo"
    shared.mkdir()
    config = make_config(tmp_path, claude_dir=shared, codex_dir=shared)

    with pytest.raises(AgentLoopError, match="same directory"):
        run_pr_loop(runner, pr_number=77, config=config)


def test_gemini_shared_workdir_requires_explicit_override(tmp_path):
    runner = FakeRunner()
    shared = tmp_path / "repo"
    shared.mkdir()
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        codex_dir=shared,
        gemini_dir=shared,
    )

    with pytest.raises(AgentLoopError, match="same directory"):
        run_pr_loop(runner, pr_number=77, config=config)


def test_missing_agent_workdirs_are_created(tmp_path):
    runner = FakeRunner(
        claude_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude"],
    )
    claude_dir = tmp_path / "missing" / "claude"
    codex_dir = tmp_path / "missing" / "codex"
    config = make_config(
        tmp_path,
        claude_dir=claude_dir,
        codex_dir=codex_dir,
        coder="codex",
        reviewer="claude",
        create_dirs=False,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert claude_dir.is_dir()
    assert codex_dir.is_dir()


def test_missing_gemini_workdir_is_created_when_configured(tmp_path):
    runner = FakeRunner(
        gemini_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"],
    )
    gemini_dir = tmp_path / "missing" / "gemini"
    config = make_config(
        tmp_path,
        reviewer="gemini",
        gemini_dir=gemini_dir,
        create_dirs=False,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert gemini_dir.is_dir()


def test_non_codex_loop_uses_active_workdir_for_github_and_tests(tmp_path):
    runner = FakeRunner(
        gemini_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"],
    )
    codex_dir = tmp_path / "inactive" / "codex"
    config = make_config(
        tmp_path,
        claude_dir=tmp_path / "missing" / "claude",
        codex_dir=codex_dir,
        gemini_dir=tmp_path / "missing" / "gemini",
        coder="claude",
        reviewer="gemini",
        test_command=("pytest", "tests/test_agent_loop.py"),
        create_dirs=False,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert not codex_dir.exists()
    github_or_test_cwds = [
        cwd
        for cmd, cwd in runner.commands
        if cmd[:1] == ["gh"] or cmd == ["pytest", "tests/test_agent_loop.py"]
    ]
    assert github_or_test_cwds
    bootstrap_pr_queries = [
        cwd
        for cmd, cwd in runner.commands
        if cmd[:3] == ["gh", "pr", "view"]
        and "number,title,headRefName,baseRefName,headRefOid,url,body,comments,reviews" in cmd
        and cwd != config.claude_dir
    ]
    assert bootstrap_pr_queries == [Path.cwd()]
    assert set(github_or_test_cwds) == {Path.cwd(), config.claude_dir}


def test_omitted_agent_dirs_default_to_repo_scoped_temp_checkouts(monkeypatch, tmp_path):
    parser = build_parser()
    cache_home = tmp_path / "cache"
    monkeypatch.setattr("coding_review_agent_loop.config.sys.platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    args = parser.parse_args([
        "task",
        "Fix the bug",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "codex",
        "--reviewer",
        "claude",
    ])

    config = config_from_args(args, FakeRunner())

    assert config.codex_dir == default_agent_workdir("OWNER/REPO", "codex").resolve()
    assert config.claude_dir == default_agent_workdir("OWNER/REPO", "claude").resolve()
    assert config.gemini_dir == default_agent_workdir("OWNER/REPO", "gemini").resolve()
    assert config.antigravity_dir == default_agent_workdir("OWNER/REPO", "antigravity").resolve()
    assert set(config.auto_agent_dirs) == {"claude", "codex", "gemini", "antigravity"}
    assert config.agent_memory_dir == (
        cache_home / "coding-review-agent-loop" / "repos" / "OWNER-REPO" / "memory"
    ).resolve()


@pytest.mark.parametrize(
    ("coder", "reviewer", "missing_command", "override_flag"),
    [
        ("claude", "codex", "missing-claude", "--claude-cmd"),
        ("claude", "gemini", "missing-gemini", "--gemini-cmd"),
    ],
)
def test_config_preflight_rejects_missing_agent_before_repo_detection(
    monkeypatch,
    coder,
    reviewer,
    missing_command,
    override_flag,
):
    parser = build_parser()
    args = parser.parse_args([
        "task",
        "Fix the bug",
        "--coder",
        coder,
        "--reviewer",
        reviewer,
        f"--{coder}-cmd",
        missing_command if override_flag == f"--{coder}-cmd" else coder,
        f"--{reviewer}-cmd",
        missing_command if override_flag == f"--{reviewer}-cmd" else reviewer,
    ])
    detection_calls = []
    monkeypatch.setattr(
        "coding_review_agent_loop.config.detect_repo",
        lambda *call_args: detection_calls.append(call_args),
    )
    monkeypatch.setattr(
        "coding_review_agent_loop.config.shutil.which",
        lambda command: None if command == missing_command else f"/bin/{command}",
    )

    with pytest.raises(
        AgentLoopError,
        match=rf"{missing_command} CLI not found on PATH.*{override_flag}",
    ):
        config_from_args(args, Runner())

    assert detection_calls == []


def test_config_preflight_checks_only_unique_configured_agents(monkeypatch, tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "task",
        "Fix the bug",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "codex",
        "--reviewer",
        "codex",
        "--codex-dir",
        str(tmp_path / "codex"),
    ])
    checked = []

    def fake_which(command):
        checked.append(command)
        return f"/bin/{command}"

    monkeypatch.setattr("coding_review_agent_loop.config.shutil.which", fake_which)

    config = config_from_args(args, Runner())

    assert config.coder == "codex"
    assert checked == ["codex"]


def test_config_preflight_accepts_custom_absolute_command(tmp_path):
    command = tmp_path / "custom-codex"
    command.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    command.chmod(0o755)
    parser = build_parser()
    args = parser.parse_args([
        "task",
        "Fix the bug",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "codex",
        "--reviewer",
        "codex",
        "--codex-cmd",
        str(command),
    ])

    config = config_from_args(args, Runner())

    assert config.codex_cmd == str(command)


def test_config_preflight_skips_dry_run_command_preview(monkeypatch):
    parser = build_parser()
    args = parser.parse_args([
        "task",
        "Fix the bug",
        "--repo",
        "OWNER/REPO",
        "--dry-run",
        "--claude-cmd",
        "missing-claude",
        "--codex-cmd",
        "missing-codex",
    ])
    monkeypatch.setattr(
        "coding_review_agent_loop.config.shutil.which",
        lambda command: pytest.fail(f"unexpected preflight for {command}"),
    )

    config = config_from_args(args, Runner(dry_run=True))

    assert config.dry_run is True


def test_omitted_cli_base_is_preserved_for_runtime_resolution(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
    ])

    assert config_from_args(args, FakeRunner()).base is None


def test_pre_review_tests_cli_defaults_on_and_can_be_disabled(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "task",
        "Fix the bug",
        "--repo",
        "OWNER/REPO",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
    ])
    config = config_from_args(args, FakeRunner())
    assert config.pre_review_tests is True

    args = parser.parse_args([
        "task",
        "Fix the bug",
        "--repo",
        "OWNER/REPO",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
        "--no-pre-review-tests",
    ])
    config = config_from_args(args, FakeRunner())
    assert config.pre_review_tests is False


@pytest.mark.parametrize("repo", ["OWNER", "OWNER/", "/REPO", "OWNER/REPO/EXTRA"])
def test_default_agent_workdir_rejects_invalid_repo_formats(repo):
    with pytest.raises(AgentLoopError, match="OWNER/REPO"):
        default_agent_workdir(repo, "codex")


def test_default_agent_memory_dir_uses_xdg_cache_and_repo_scope(monkeypatch, tmp_path):
    cache_home = tmp_path / "xdg-cache"
    monkeypatch.setattr("coding_review_agent_loop.config.sys.platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))

    assert default_agent_memory_dir("OWNER/REPO") == (
        cache_home / "coding-review-agent-loop" / "repos" / "OWNER-REPO" / "memory"
    )


def test_default_cache_root_uses_posix_home_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr("coding_review_agent_loop.config.sys.platform", "linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    assert default_cache_root() == tmp_path / ".cache" / "coding-review-agent-loop"


@pytest.mark.parametrize(
    ("platform", "home_parts"),
    [
        ("darwin", ("Library", "Caches", "coding-review-agent-loop")),
        ("win32", ("AppData", "Local", "coding-review-agent-loop", "Cache")),
    ],
)
def test_default_cache_root_uses_platform_home_fallbacks(
    monkeypatch,
    tmp_path,
    platform,
    home_parts,
):
    monkeypatch.setattr("coding_review_agent_loop.config.sys.platform", platform)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)

    assert default_cache_root() == tmp_path.joinpath(*home_parts)


def test_default_cache_root_uses_windows_local_app_data(monkeypatch, tmp_path):
    local_app_data = tmp_path / "local-app-data"
    monkeypatch.setattr("coding_review_agent_loop.config.sys.platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))

    assert default_cache_root() == local_app_data / "coding-review-agent-loop" / "Cache"


@pytest.mark.parametrize("repo", ["OWNER", "OWNER/", "/REPO", "OWNER/REPO/EXTRA"])
def test_default_agent_memory_dir_rejects_invalid_repo_formats(repo):
    with pytest.raises(AgentLoopError, match="OWNER/REPO"):
        default_agent_memory_dir(repo)


@pytest.mark.parametrize("mode", ["ignore", "summarize", "issue", "fix-and-summarize", "fix-and-issue"])
def test_approved_followups_cli_mode_is_configurable(tmp_path, mode):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--approved-followups",
        mode,
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
        "--gemini-dir",
        str(tmp_path / "gemini"),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.approved_followups == mode


@pytest.mark.parametrize(
    "mode",
    ["plan-only", "decompose-only", "implement-one-shot", "implement-by-phase"],
)
def test_plan_execution_mode_cli_is_configurable(tmp_path, mode):
    parser = build_parser()
    args = parser.parse_args([
        "issue",
        "56",
        "--repo",
        "OWNER/REPO",
        "--plan-first",
        "--plan-execution-mode",
        mode,
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
        "--gemini-dir",
        str(tmp_path / "gemini"),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.plan_execution_mode == mode


def test_explicit_agent_dirs_are_preserved_when_others_default(tmp_path):
    parser = build_parser()
    codex_dir = tmp_path / "codex"
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "codex",
        "--reviewer",
        "claude",
        "--codex-dir",
        str(codex_dir),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.codex_dir == codex_dir
    assert config.claude_dir == default_agent_workdir("OWNER/REPO", "claude").resolve()
    assert set(config.auto_agent_dirs) == {"claude", "gemini", "antigravity"}


def test_relative_log_dir_defaults_under_active_coder_workdir(tmp_path):
    parser = build_parser()
    claude_dir = tmp_path / "claude"
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "claude",
        "--reviewer",
        "gemini",
        "--claude-dir",
        str(claude_dir),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.log_dir == claude_dir / ".agent-loop-logs"


def test_agent_memory_flags_configure_memory_dir_and_refresh(tmp_path):
    parser = build_parser()
    codex_dir = tmp_path / "codex"
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "codex",
        "--reviewer",
        "claude",
        "--codex-dir",
        str(codex_dir),
        "--no-agent-memory",
        "--refresh-agent-memory",
        "--refresh-test-profile",
        "--agent-memory-dir",
        "custom-memory",
    ])

    config = config_from_args(args, FakeRunner())

    assert config.agent_memory is False
    assert config.refresh_agent_memory is True
    assert config.refresh_test_profile is True
    assert config.agent_memory_dir == codex_dir / "custom-memory"


def test_agent_memory_explicit_absolute_dir_is_resolved(tmp_path):
    parser = build_parser()
    memory_dir = tmp_path / "memory-parent" / ".." / "agent-memory"
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--agent-memory-dir",
        str(memory_dir),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.agent_memory_dir == memory_dir.resolve()


def test_agent_memory_default_ignores_active_coder_workdir(tmp_path, monkeypatch):
    parser = build_parser()
    cache_home = tmp_path / "cache"
    codex_dir = tmp_path / "codex"
    monkeypatch.setattr("coding_review_agent_loop.config.sys.platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "codex",
        "--reviewer",
        "claude",
        "--codex-dir",
        str(codex_dir),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.agent_memory_dir == (
        cache_home / "coding-review-agent-loop" / "repos" / "OWNER-REPO" / "memory"
    ).resolve()
    assert codex_dir not in config.agent_memory_dir.parents


def test_auto_created_agent_dir_is_cloned_before_use(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    codex_dir = tmp_path / "tmp-root" / "owner-repo" / "codex" / "repo"
    config = make_config(
        tmp_path,
        claude_dir=tmp_path / "explicit-claude",
        codex_dir=codex_dir,
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
    )
    config.claude_dir.mkdir(parents=True)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert ["gh", "repo", "clone", "OWNER/REPO", str(codex_dir)] in [
        cmd for cmd, _cwd in runner.commands
    ]
    assert codex_dir.is_dir()


def test_clean_existing_auto_agent_dir_is_synced(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    assert ["git", "fetch", "origin"] in commands
    assert ["git", "switch", "main"] in commands
    assert ["git", "pull", "--ff-only", "origin", "main"] in commands


def test_pr_loop_resolves_pr_base_before_workdir_setup(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={"baseRefName": "develop"},
    )
    config = make_config(
        tmp_path,
        base=None,
        reviewer="codex",
        auto_agent_dirs=("codex",),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    pr_context_index = command_index(runner.commands, ["gh", "pr", "view"])
    switch_index = command_index(runner.commands, ["git", "switch", "develop"])
    assert pr_context_index < switch_index
    assert ["git", "pull", "--ff-only", "origin", "develop"] in commands
    assert not any("origin/main" in arg for cmd in commands for arg in cmd)


def test_pr_loop_explicit_base_overrides_pr_base_without_repo_default_query(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={"baseRefName": "develop"},
    )
    config = make_config(
        tmp_path,
        base="release",
        reviewer="codex",
        auto_agent_dirs=("codex",),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    assert ["git", "switch", "release"] in commands
    assert ["git", "switch", "develop"] not in commands
    assert not any(
        cmd[:3] == ["gh", "repo", "view"] and "defaultBranchRef" in cmd
        for cmd in commands
    )


@pytest.mark.parametrize("pr_base", [None, "", "   "])
def test_pr_loop_falls_back_to_repo_default_when_pr_base_is_missing(tmp_path, pr_base):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={"baseRefName": pr_base},
        repo_default_branch="develop",
    )
    config = make_config(
        tmp_path,
        base=None,
        reviewer="codex",
        auto_agent_dirs=("codex",),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    repo_query_index = command_index(runner.commands, ["gh", "repo", "view"])
    switch_index = command_index(runner.commands, ["git", "switch", "develop"])
    assert repo_query_index < switch_index


@pytest.mark.parametrize("mode", ["issue", "task"])
def test_issue_and_task_loops_use_repo_default_when_base_is_omitted(tmp_path, mode):
    runner = FakeRunner(
        claude_outputs=[
            "Implemented.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={"baseRefName": "develop"},
        repo_default_branch="develop",
    )
    config = make_config(
        tmp_path,
        base=None,
        reviewer="codex",
        auto_agent_dirs=("claude", "codex"),
    )

    if mode == "issue":
        assert run_issue_loop(runner, issue_number=56, config=config) == 0
    else:
        assert run_task_loop(runner, task_text="Add /healthz endpoint.", config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    assert ["git", "switch", "develop"] in commands
    assert not any("origin/main" in arg for cmd in commands for arg in cmd)


def test_unresolved_base_metadata_produces_targeted_override_error(tmp_path):
    runner = FakeRunner(
        pr_payload={"baseRefName": None},
        repo_default_branch=None,
        repo_default_branch_returncode=1,
    )
    config = make_config(tmp_path, base=None, reviewer="codex")

    with pytest.raises(
        AgentLoopError,
        match=r"Unable to resolve a base branch for OWNER/REPO.*--base <branch>",
    ):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:2] == ["git", "switch"] for cmd, _cwd in runner.commands)


def test_dry_run_base_resolution_defaults_to_main_without_github_query(tmp_path):
    runner = FakeRunner()
    config = make_config(tmp_path, base=None, dry_run=True)

    resolved = resolve_base_branch(config, runner)

    assert resolved.base == "main"
    assert not any(cmd[:1] == ["gh"] for cmd, _cwd in runner.commands)


def test_reviewer_checkout_is_refreshed_to_pr_head_before_review(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path, reviewer="codex")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    review_index = command_index(runner.commands, ["codex", "exec"])
    fetch_index = command_index(runner.commands, ["git", "fetch", "origin"], start=0)
    pr_fetch_index = command_index(
        runner.commands,
        ["git", "fetch", "origin", "+pull/77/head:refs/remotes/origin/pr/77"],
    )
    checkout_index = command_index(
        runner.commands,
        ["git", "checkout", "--detach", "refs/remotes/origin/pr/77"],
    )
    head_index = command_index(runner.commands, ["git", "rev-parse", "HEAD"], start=checkout_index)

    assert commands[pr_fetch_index] == ["git", "fetch", "origin", "+pull/77/head:refs/remotes/origin/pr/77"]
    assert fetch_index < pr_fetch_index < checkout_index < head_index < review_index


def test_reviewer_checkout_refreshes_each_round_before_review(tmp_path):
    runner = FakeRunner(
        claude_outputs=["Fixed.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
        codex_outputs=[
            "Please fix it.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    pr_fetches = [
        index
        for index, cmd in enumerate(commands)
        if cmd == ["git", "fetch", "origin", "+pull/77/head:refs/remotes/origin/pr/77"]
    ]
    review_indices = [index for index, cmd in enumerate(commands) if cmd[:2] == ["codex", "exec"]]

    assert len(pr_fetches) == 3
    assert len(review_indices) == 2
    assert pr_fetches[0] < review_indices[0]
    assert pr_fetches[1] < review_indices[1]


def test_dirty_default_reviewer_checkout_is_cleaned_before_review(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        git_status=" M stale.py\n",
    )
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        coder="claude",
        reviewer="codex",
        auto_agent_dirs=("claude", "codex"),
        create_dirs=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    assert run_pr_loop(runner, pr_number=77, config=config, workdirs_ready=True) == 0

    reset_index = command_index(runner.commands, ["git", "reset", "--hard"])
    clean_index = command_index(runner.commands, ["git", "clean", "-fd"])
    review_index = command_index(runner.commands, ["codex", "exec"])

    assert reset_index < clean_index < review_index


def test_dirty_explicit_reviewer_checkout_fails_before_review_invocation(tmp_path):
    runner = FakeRunner(
        codex_outputs=["This should not run.\n<!-- AGENT_STATE: approved -->"],
        git_status=" M stale.py\n",
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", agent_memory=False)

    with pytest.raises(AgentLoopError, match="--codex-dir is dirty"):
        run_pr_loop(runner, pr_number=77, config=config, workdirs_ready=True)

    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


def test_review_prompt_warns_that_pr_head_sha_is_authoritative(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path, reviewer="codex")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    codex_command = next(cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    prompt = codex_command[-1]
    assert "The Head SHA above is the PR head this\nreview round is about." in prompt
    assert "If local files do not match that SHA, refresh/fetch the\ncheckout before reviewing." in prompt
    assert "Do not report findings based on untracked files unless those files are\npresent in the PR diff." in prompt


def test_dirty_existing_auto_agent_dir_is_cleaned_before_sync(tmp_path, capsys):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        git_status=" M file.py\n",
    )
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        coder="codex",
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
        quiet=False,
    )
    config.gemini_dir.mkdir(parents=True)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    assert ["git", "reset", "--hard"] in commands
    assert ["git", "clean", "-fd"] in commands
    assert ["git", "pull", "--ff-only", "origin", "main"] in commands
    captured = capsys.readouterr()
    assert f"Cleaning dirty default codex workdir: {codex_dir}" in captured.err


def test_dirty_explicit_agent_dir_fails_clearly(tmp_path):
    runner = FakeRunner(git_status=" M file.py\n")
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        coder="codex",
        reviewer="codex",
        create_dirs=False,
    )
    config.gemini_dir.mkdir(parents=True)

    with pytest.raises(AgentLoopError, match="--codex-dir is dirty"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


@pytest.mark.parametrize("loop_name", ["issue", "task"])
def test_dirty_explicit_coder_dir_fails_before_issue_or_task_coder_invocation(tmp_path, loop_name):
    runner = FakeRunner(
        git_status=" M file.py\n",
        codex_outputs=[
            "Implemented.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
    )
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        coder="codex",
        reviewer="codex",
        create_dirs=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    with pytest.raises(AgentLoopError, match="--codex-dir is dirty"):
        if loop_name == "issue":
            run_issue_loop(runner, issue_number=56, config=config)
        else:
            run_task_loop(runner, task_text="Add /healthz endpoint.", config=config)

    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


def test_explicit_agent_dir_must_match_requested_repo(tmp_path):
    runner = FakeRunner(git_remote="git@github.com:OTHER/REPO.git")
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        coder="codex",
        reviewer="codex",
        create_dirs=False,
    )
    config.gemini_dir.mkdir(parents=True)

    with pytest.raises(AgentLoopError, match="not 'OWNER/REPO'"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


def test_existing_auto_agent_dir_must_be_git_checkout(tmp_path):
    runner = FakeRunner(git_inside=False)
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    with pytest.raises(AgentLoopError, match="not a git checkout"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


def test_stale_default_workdir_only_logs_is_recreated(tmp_path, capsys):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    (codex_dir / ".agent-loop-logs").mkdir()

    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
        quiet=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    clone_cmds = [cmd for cmd, _cwd in runner.commands if cmd[:3] == ["gh", "repo", "clone"]]
    assert any(cmd[4] == str(codex_dir) for cmd in clone_cmds), "Expected fresh clone of stale workdir"

    captured = capsys.readouterr()
    assert "Stale default codex workdir detected" in captured.err
    assert "recreating" in captured.err


def test_stale_default_workdir_with_unknown_files_still_fails(tmp_path):
    runner = FakeRunner(git_inside=False)
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    (codex_dir / ".agent-loop-logs").mkdir()
    (codex_dir / "some-user-file.py").write_text("# user work", encoding="utf-8")

    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    with pytest.raises(AgentLoopError, match="not a git checkout"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:3] == ["gh", "repo", "clone"] for cmd, _cwd in runner.commands)


def test_stale_default_workdir_empty_is_recreated(tmp_path, capsys):
    """An empty workdir (no .git, no files) is treated as stale and re-cloned."""
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()  # exists but empty

    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
        quiet=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    clone_cmds = [cmd for cmd, _cwd in runner.commands if cmd[:3] == ["gh", "repo", "clone"]]
    assert any(cmd[4] == str(codex_dir) for cmd in clone_cmds), "Expected fresh clone of empty stale workdir"

    captured = capsys.readouterr()
    assert "Stale default codex workdir detected" in captured.err
    assert "recreating" in captured.err


def test_stale_default_workdir_git_only_is_recreated(tmp_path, capsys):
    """A workdir with only a .git dir (no working tree) is treated as stale and re-cloned."""
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    (codex_dir / ".git").mkdir()  # .git present, but no source files

    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
        quiet=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    clone_cmds = [cmd for cmd, _cwd in runner.commands if cmd[:3] == ["gh", "repo", "clone"]]
    assert any(cmd[4] == str(codex_dir) for cmd in clone_cmds), "Expected fresh clone of git-only stale workdir"

    captured = capsys.readouterr()
    assert "Stale default codex workdir detected" in captured.err
    assert "recreating" in captured.err


def test_explicit_dir_not_git_checkout_is_not_recreated(tmp_path):
    runner = FakeRunner(git_inside=False)
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    (codex_dir / ".agent-loop-logs").mkdir()

    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        reviewer="codex",
        create_dirs=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    with pytest.raises(AgentLoopError, match="not a git checkout"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:3] == ["gh", "repo", "clone"] for cmd, _cwd in runner.commands)


def test_existing_auto_agent_dir_must_match_requested_repo(tmp_path):
    runner = FakeRunner(git_remote="git@github.com:OTHER/REPO.git")
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    with pytest.raises(AgentLoopError, match="not 'OWNER/REPO'"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


def test_agent_workdir_existing_file_fails_clearly(tmp_path):
    runner = FakeRunner()
    claude_path = tmp_path / "claude-file"
    claude_path.write_text("not a dir", encoding="utf-8")
    config = make_config(tmp_path, claude_dir=claude_path, create_dirs=False)

    with pytest.raises(AgentLoopError, match="not a directory"):
        run_pr_loop(runner, pr_number=77, config=config)


def test_gemini_workdir_existing_file_fails_clearly(tmp_path):
    runner = FakeRunner()
    gemini_path = tmp_path / "gemini-file"
    gemini_path.write_text("not a dir", encoding="utf-8")
    config = make_config(
        tmp_path,
        reviewer="gemini",
        gemini_dir=gemini_path,
        create_dirs=False,
    )

    with pytest.raises(AgentLoopError, match="not a directory"):
        run_pr_loop(runner, pr_number=77, config=config)


def test_config_allows_same_coder_and_reviewer(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "codex",
        "--reviewer",
        "codex",
        "--codex-dir",
        str(tmp_path / "codex"),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.coder == "codex"
    assert config.reviewer == ("codex",)


def test_config_allows_coder_in_multiple_reviewers(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "codex",
        "--reviewer",
        "claude",
        "--reviewer",
        "codex",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.coder == "codex"
    assert config.reviewer == ("claude", "codex")


def test_config_accepts_gemini_as_coder_and_reviewer(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "gemini",
        "--reviewer",
        "claude",
        "--reviewer",
        "gemini",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--gemini-dir",
        str(tmp_path / "gemini"),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.coder == "gemini"
    assert config.reviewer == ("claude", "gemini")
    assert config.gemini_dir == tmp_path / "gemini"


@pytest.mark.parametrize(
    ("coder", "reviewer"),
    [
        ("agy", "codex"),
        ("codex", "agy"),
        ("antigravity", "codex"),
    ],
)
def test_config_normalizes_antigravity_agent_names(tmp_path, coder, reviewer):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--coder",
        coder,
        "--reviewer",
        reviewer,
        "--codex-dir",
        str(tmp_path / "codex"),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.coder == ("antigravity" if coder == "agy" else coder)
    assert config.reviewer == (
        "antigravity" if reviewer == "agy" else reviewer,
    )


def test_config_rejects_duplicate_reviewers(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--reviewer",
        "codex",
        "--reviewer",
        "codex",
        "--codex-dir",
        str(tmp_path / "codex"),
    ])

    with pytest.raises(AgentLoopError, match="same agent more than once"):
        config_from_args(args, FakeRunner())


def test_config_rejects_alias_and_canonical_duplicate_reviewers(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--reviewer",
        "agy",
        "--reviewer",
        "antigravity",
        "--codex-dir",
        str(tmp_path / "codex"),
    ])

    with pytest.raises(AgentLoopError, match="same agent more than once"):
        config_from_args(args, FakeRunner())


@pytest.mark.parametrize("max_rounds", ["0", "-1"])
def test_config_rejects_non_positive_max_rounds(tmp_path, max_rounds):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--max-rounds",
        max_rounds,
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
    ])

    with pytest.raises(AgentLoopError, match="--max-rounds must be greater than zero"):
        config_from_args(args, FakeRunner())


def test_config_defaults_do_not_bypass_agent_permissions(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
    ])
    config = config_from_args(args, FakeRunner())

    assert config.claude_args == ()
    assert config.codex_args == ()
    assert config.gemini_args == ()


def test_config_can_opt_into_dangerous_agent_permissions(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
        "--dangerous-agent-permissions",
    ])
    config = config_from_args(args, FakeRunner())

    assert config.claude_args == ("--dangerously-skip-permissions",)
    assert config.codex_args == ("--dangerously-bypass-approvals-and-sandbox",)
    assert config.gemini_args == ("--yolo", "--skip-trust")


def test_explicit_agent_args_replace_dangerous_profile(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
        "--dangerous-agent-permissions",
        "--claude-arg=--permission-mode",
        "--claude-arg=acceptEdits",
        "--codex-arg=--sandbox",
        "--codex-arg=workspace-write",
        "--gemini-arg=--approval-mode",
        "--gemini-arg=auto_edit",
    ])
    config = config_from_args(args, FakeRunner())

    assert config.claude_args == ("--permission-mode", "acceptEdits")
    assert config.codex_args == ("--sandbox", "workspace-write")
    assert config.gemini_args == ("--approval-mode", "auto_edit")


def test_issue_loop_requires_claude_to_report_pr_number(tmp_path):
    runner = FakeRunner(claude_outputs=["Created something.\n<!-- AGENT_STATE: blocking -->"])
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="PR marker"):
        run_issue_loop(runner, issue_number=56, config=config)


def test_issue_loop_rejects_missing_initial_issue_human_requirements_acknowledgement(tmp_path):
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Keep the legacy flag.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            "Created PR.\nTests: python -m pytest passed.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->"
        ],
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="missing required signed human requirements marker"):
        run_issue_loop(runner, issue_number=56, config=config)


def test_issue_loop_accepts_initial_issue_human_requirements_acknowledgement(tmp_path):
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Keep the legacy flag.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            "Created PR.\nTests: python3 -m pytest passed.\n"
            f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
            "### Human requirements\n"
            "- Requirement 1: kept the legacy flag path.\n"
            "<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->"
        ],
        codex_outputs=[
            "LGTM.\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config) == 0


def test_issue_loop_rejects_pr_number_before_running_claude(tmp_path):
    runner = FakeRunner(issue_payload={
        "number": 62,
        "state": "closed",
        "is_pr": True,
        "url": "https://github.com/OWNER/REPO/pull/62",
    })
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="pull request, not an issue"):
        run_issue_loop(runner, issue_number=62, config=config)

    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def test_issue_loop_plan_first_stops_after_approved_plan(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Plan:\n- Update the CLI.\n- Add tests.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert any(cmd[:3] == ["claude", "--print", "--output-format"] for cmd, _cwd in runner.commands)
    assert any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)
    assert not any(cmd[:3] == ["gh", "pr", "view"] for cmd, _cwd in runner.commands)
    assert len(runner.comments) == 3
    assert runner.comments[0].startswith("Plan:")
    assert runner.comments[1].startswith("**Review verdict:** Approved\n\nPlan looks sound.")
    assert "Outcome: implement" in runner.comments[2]
    assert not any(cmd[:2] == ["git", "fetch"] for cmd, _cwd in runner.commands)
    assert not any(cmd[:2] == ["git", "switch"] for cmd, _cwd in runner.commands)


def test_parse_plan_decomposition_accepts_agent_and_human_phases():
    parsed = parse_plan_decomposition(
        plan_decomposition_json(
            {
                "title": "Internal schema utilities",
                "scope": "Add helpers.",
                "non_goals": "No live switch.",
                "dependency_notes": "First phase.",
                "rollout_risk": "low - internal only.",
                "validation": "Run python -m pytest.",
                "parent_context": "Approved plan slice and invariant details.",
                "automation": "agent-pr",
                "depends_on": [],
            },
            {
                "title": "Manual rollout checkpoint",
                "scope": "Human validates the deployed behavior.",
                "non_goals": "No code changes.",
                "dependency_notes": "After Internal schema utilities.",
                "rollout_risk": "medium - live checkpoint.",
                "validation": "Human remark and closure required.",
                "parent_context": "Approved plan slice for the manual checkpoint.",
                "automation": "human-action",
                "depends_on": ["Internal schema utilities"],
            },
        )
    )

    assert [phase.title for phase in parsed.phases] == [
        "Internal schema utilities",
        "Manual rollout checkpoint",
    ]
    assert parsed.phases[1].automation == "human-action"
    assert parsed.phases[1].depends_on == ("Internal schema utilities",)


def test_parse_plan_decomposition_accepts_normalized_earlier_phase_dependency():
    parsed = parse_plan_decomposition(
        plan_decomposition_json(
            {
                "title": "Internal schema utilities",
                "scope": "Add helpers.",
                "non_goals": "No live switch.",
                "dependency_notes": "First phase.",
                "rollout_risk": "low - internal only.",
                "validation": "Run python -m pytest.",
                "parent_context": "Approved plan slice and invariant details.",
                "automation": "agent-pr",
                "depends_on": [],
            },
            {
                "title": "Manual rollout checkpoint",
                "scope": "Human validates the deployed behavior.",
                "non_goals": "No code changes.",
                "dependency_notes": "After Internal schema utilities.",
                "rollout_risk": "medium - live checkpoint.",
                "validation": "Human remark and closure required.",
                "parent_context": "Approved plan slice for the manual checkpoint.",
                "automation": "human-action",
                "depends_on": ["  internal   SCHEMA utilities  "],
            },
        )
    )

    assert parsed.phases[1].depends_on == ("internal   SCHEMA utilities",)


def test_parse_plan_decomposition_rejects_self_dependency():
    phase = {
        "title": "Internal schema utilities",
        "scope": "Add helpers.",
        "non_goals": "No live switch.",
        "dependency_notes": "First phase.",
        "rollout_risk": "low - internal only.",
        "validation": "Run python -m pytest.",
        "parent_context": "Approved plan slice and invariant details.",
        "automation": "agent-pr",
        "depends_on": ["Internal schema utilities"],
    }

    with pytest.raises(AgentLoopError, match="cannot depend on itself"):
        parse_plan_decomposition(plan_decomposition_json(phase))


def test_parse_plan_decomposition_rejects_forward_dependency():
    first_phase = {
        "title": "Internal schema utilities",
        "scope": "Add helpers.",
        "non_goals": "No live switch.",
        "dependency_notes": "First phase.",
        "rollout_risk": "low - internal only.",
        "validation": "Run python -m pytest.",
        "parent_context": "Approved plan slice and invariant details.",
        "automation": "agent-pr",
        "depends_on": ["Manual rollout checkpoint"],
    }
    second_phase = {
        "title": "Manual rollout checkpoint",
        "scope": "Human validates the deployed behavior.",
        "non_goals": "No code changes.",
        "dependency_notes": "After Internal schema utilities.",
        "rollout_risk": "medium - live checkpoint.",
        "validation": "Human remark and closure required.",
        "parent_context": "Approved plan slice for the manual checkpoint.",
        "automation": "human-action",
        "depends_on": [],
    }

    with pytest.raises(AgentLoopError, match="dependencies must reference an earlier phase"):
        parse_plan_decomposition(plan_decomposition_json(first_phase, second_phase))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda phase: phase.pop("parent_context"), "parent_context"),
        (lambda phase: phase.pop("rollout_risk"), "rollout_risk"),
        (lambda phase: phase.pop("validation"), "validation"),
        (lambda phase: phase.__setitem__("automation", "robot"), "invalid automation"),
        (lambda phase: phase.__setitem__("depends_on", ["Missing phase"]), "unknown phase"),
    ],
)
def test_parse_plan_decomposition_rejects_invalid_phase_fields(mutate, message):
    phase = {
        "title": "Internal schema utilities",
        "scope": "Add helpers.",
        "non_goals": "No live switch.",
        "dependency_notes": "First phase.",
        "rollout_risk": "low - internal only.",
        "validation": "Run python -m pytest.",
        "parent_context": "Approved plan slice and invariant details.",
        "automation": "agent-pr",
        "depends_on": [],
    }
    mutate(phase)

    with pytest.raises(AgentLoopError, match=message):
        parse_plan_decomposition(plan_decomposition_json(phase))


def test_parse_plan_decomposition_rejects_duplicates_and_over_cap():
    phase = {
        "title": "Repeated phase",
        "scope": "Add helpers.",
        "non_goals": "No live switch.",
        "dependency_notes": "First phase.",
        "rollout_risk": "low - internal only.",
        "validation": "Run python -m pytest.",
        "parent_context": "Approved plan slice and invariant details.",
        "automation": "agent-pr",
        "depends_on": [],
    }
    with pytest.raises(AgentLoopError, match="duplicate phase title"):
        parse_plan_decomposition(plan_decomposition_json(phase, dict(phase)))

    phases = [dict(phase, title=f"Phase {index}") for index in range(MAX_DECOMPOSITION_PHASES + 1)]
    with pytest.raises(AgentLoopError, match="MAX_DECOMPOSITION_PHASES"):
        parse_plan_decomposition(plan_decomposition_json(*phases))


def test_issue_loop_plan_first_rejects_missing_initial_plan_human_requirements_acknowledgement(
    tmp_path,
):
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Keep the public API unchanged.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            "Plan:\n- Update the parser.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
        ],
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="missing required signed human requirements marker"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)


def test_issue_loop_plan_first_accepts_initial_plan_human_requirements_acknowledgement(tmp_path):
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Keep the public API unchanged.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            "Plan:\n- Update the parser.\n"
            f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
            "### Human requirements\n"
            "- Requirement 1: the plan keeps the public API unchanged.\n"
            "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
        ],
        codex_outputs=[structured_plan_review(summary="Plan looks sound.", human_requirements_resolved=True)],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0


def test_issue_loop_plan_first_revises_until_all_reviewers_approve(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            structured_plan_revision(summary="Revised plan with tests."),
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Missing test strategy.",
                blocking_plan_issues=["Missing test strategy."],
            ),
            structured_plan_review(
                summary="Plan looks sound.",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0


def test_issue_loop_structured_plan_state_public_comment_renders_markdown_and_preserves_metadata(tmp_path):
    raw_structured_plan = structured_plan_state(
        summary="Plan the issue fix.",
        plan_steps=["Update the renderer.", "Add regression tests."],
        reviewer="Google Antigravity",
    )
    runner = FakeRunner(
        antigravity_outputs=[raw_structured_plan],
        codex_outputs=[structured_plan_review(summary="Plan looks sound.")],
    )
    config = make_config(tmp_path, coder="antigravity", reviewer="codex")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    public_comment = runner.comments[0]
    assert public_comment.startswith("## Plan")
    assert "### Plan steps\n1. Update the renderer.\n2. Add regression tests." in public_comment
    assert '"kind": "plan_state"' not in _strip_round_metadata(public_comment)

    raw_comment = runner.issue_comments[0]["body"]
    match = re.search(r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", raw_comment)
    assert match is not None
    metadata = _decode_round_metadata(match.group("payload"))
    assert metadata.canonical_plan == raw_structured_plan
    assert metadata.raw_structured_coder_response == raw_structured_plan


def test_issue_loop_markdown_plan_state_public_comment_passes_through(tmp_path):
    markdown_plan = "Initial markdown plan.\n<!-- AGENT_PLAN_STATE: approved -->\n-- Anthropic Claude"
    runner = FakeRunner(
        claude_outputs=[markdown_plan],
        codex_outputs=[structured_plan_review(summary="Plan looks sound.")],
    )
    config = make_config(tmp_path, reviewer="codex")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    raw_comment = runner.issue_comments[0]["body"]
    metadata_match = re.search(
        r"\n?<!--\s*AGENT_LOOP_META:\s*[A-Za-z0-9+/=_-]+\s*-->\n?",
        raw_comment,
    )
    assert metadata_match is not None
    assert raw_comment.replace(metadata_match.group(0), "\n").strip() == markdown_plan
    metadata = _decode_round_metadata(
        re.search(r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", raw_comment)
        .group("payload")
    )
    assert metadata.canonical_plan is None
    assert metadata.raw_structured_coder_response is None


def test_issue_loop_plan_revision_stores_raw_structured_metadata(tmp_path):
    raw_structured_revision = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_revision",
                "state": "blocking",
                "summary": "Revised plan with tests.",
                "prior_plan_item_dispositions": [
                    {"item_id": "item-1", "disposition": "resolved", "note": "Added the missing test step."}
                ],
                "plan_steps": ["Add the regression test.", "Run the focused suite."],
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            raw_structured_revision,
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Missing test strategy.",
                blocking_plan_issues=["Missing test strategy."],
            ),
            structured_plan_review(
                summary="Plan looks sound.",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
    )
    config = make_config(tmp_path, reviewer="codex")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert runner.comments[2].startswith("## Revised plan")
    assert '"kind": "plan_revision"' not in _strip_round_metadata(runner.comments[2])
    raw_comment = runner.issue_comments[2]["body"]
    match = re.search(r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", raw_comment)
    assert match is not None
    metadata = _decode_round_metadata(match.group("payload"))
    assert metadata.raw_structured_coder_response == raw_structured_revision


def test_issue_loop_plan_revision_rejects_missing_human_requirements_acknowledgement(tmp_path):
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Preserve backward compatibility.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            "Initial plan.\n"
            f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
            "### Human requirements\n"
            "- Requirement 1: the plan preserves backward compatibility.\n"
            "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            structured_plan_revision(summary="Revised plan."),
            "Revised plan.\n"
            f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
            "### Human requirements\n"
            "- Requirement 1: the revised plan still preserves backward compatibility.\n"
            "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Missing a regression test.",
                blocking_plan_issues=["Missing a regression test."],
            )
        ],
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="missing required signed human requirements marker"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)


def test_issue_loop_plan_revision_accepts_human_requirements_acknowledgement(tmp_path):
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Preserve backward compatibility.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            "Initial plan.\n"
            f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
            "### Human requirements\n"
            "- Requirement 1: the plan preserves backward compatibility.\n"
            "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            structured_plan_revision(
                summary="Revised plan.",
                human_requirements=(
                    f"\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
                    "### Human requirements\n"
                    "- Requirement 1: the revised plan still preserves backward compatibility.\n"
                ),
            ),
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Missing a regression test.",
                blocking_plan_issues=["Missing a regression test."],
            ),
            structured_plan_review(
                summary="Plan looks sound.",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
                human_requirements_resolved=True,
            ),
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 2
    assert "Missing a regression test." in claude_calls[1][-1]
    assert len(runner.comments) == 5
    assert runner.comments[2].startswith("## Revised plan")


def test_issue_loop_plan_revision_repair_preserves_signed_human_requirements(tmp_path):
    malformed_revision = (
        "### Prior plan review item dispositions\n"
        "- item-1: resolved by adding compatibility tests.\n\n"
        "### Revised plan\n"
        "- Preserve backward compatibility.\n"
        "- Add regression tests.\n\n"
        f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
        "### Human requirements\n"
        "- Requirement 1: the revised plan preserves backward compatibility.\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    repaired_revision = structured_plan_revision(
        summary="Revised plan with compatibility tests.",
        prior_plan_item_dispositions=[
            {
                "item_id": "item-1",
                "disposition": "resolved",
                "note": "Added compatibility tests.",
            }
        ],
        plan_steps=["Preserve backward compatibility.", "Add regression tests."],
        human_requirements=(
            f"\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
            "### Human requirements\n"
            "- Requirement 1: the revised plan preserves backward compatibility.\n"
        ),
    )
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Preserve backward compatibility.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            "Initial plan.\n"
            f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
            "### Human requirements\n"
            "- Requirement 1: the plan preserves backward compatibility.\n"
            "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            malformed_revision,
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Missing a regression test.",
                blocking_plan_issues=["Missing a regression test."],
            ),
            structured_plan_review(
                summary="Plan looks sound.",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
                human_requirements_resolved=True,
            ),
        ],
    )
    config = make_config(tmp_path, agent_max_retries=0)
    captured_repairs = []

    def fake_attempt_repair(raw: str, gemini_cmd: str, *, expected_kind: str | None = None, unresolved_item_ids=None, surfaced_requirement_ids=None, allowed_prior_item_ids=None, unknown_prior_item_ids=None, same_round_context=None) -> str | None:
        captured_repairs.append((raw, expected_kind))
        return repaired_revision

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_attempt_repair):
        assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert len(captured_repairs) == 1
    assert captured_repairs[0][1] == "plan_revision"
    assert HUMAN_REQUIREMENTS_ADDRESSED_MARKER in captured_repairs[0][0]
    public_revision = _strip_round_metadata(runner.comments[2])
    assert '"kind": "plan_revision"' not in public_revision
    assert HUMAN_REQUIREMENTS_ADDRESSED_MARKER in public_revision
    assert "### Human requirements" in public_revision


def test_issue_loop_plan_revision_repair_rejects_wrong_kind_from_human_requirements_text(tmp_path):
    malformed_revision = (
        "### Revised plan\n"
        "- Preserve backward compatibility.\n\n"
        f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
        "### Human requirements\n"
        "- Requirement 1: the revised plan preserves backward compatibility.\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    wrong_kind_repair = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Revised the plan.",
                "addressed_items": [],
                "remaining_items": [],
                "human_requirements": {
                    "addressed_ids": ["Requirement 1"],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Preserve backward compatibility.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            "Initial plan.\n"
            f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
            "### Human requirements\n"
            "- Requirement 1: the plan preserves backward compatibility.\n"
            "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            malformed_revision,
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Missing a regression test.",
                blocking_plan_issues=["Missing a regression test."],
            )
        ],
    )
    config = make_config(tmp_path, agent_max_retries=0)
    captured_kinds = []

    def fake_attempt_repair(raw: str, gemini_cmd: str, *, expected_kind: str | None = None, unresolved_item_ids=None, surfaced_requirement_ids=None, allowed_prior_item_ids=None, unknown_prior_item_ids=None, same_round_context=None) -> str | None:
        captured_kinds.append(expected_kind)
        return wrong_kind_repair

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_attempt_repair):
        with pytest.raises(AgentLoopError, match="expected `plan_revision`"):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert captured_kinds == ["plan_revision"]


def test_issue_loop_plan_revision_repair_without_human_ack_fails_clearly(tmp_path):
    malformed_revision = (
        "### Revised plan\n"
        "- Preserve backward compatibility.\n\n"
        f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
        "### Human requirements\n"
        "- Requirement 1: the revised plan preserves backward compatibility.\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    repaired_without_ack = structured_plan_revision(
        summary="Revised plan with compatibility tests.",
        plan_steps=["Preserve backward compatibility.", "Add regression tests."],
    )
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Preserve backward compatibility.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            "Initial plan.\n"
            f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
            "### Human requirements\n"
            "- Requirement 1: the plan preserves backward compatibility.\n"
            "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            malformed_revision,
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Missing a regression test.",
                blocking_plan_issues=["Missing a regression test."],
            )
        ],
    )
    config = make_config(tmp_path, agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=repaired_without_ack):
        with pytest.raises(AgentLoopError, match="missing required signed human requirements marker"):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)


def test_issue_loop_plan_first_requires_reviewers_to_disposition_prior_items(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Revised plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Second revised plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "### Blocking plan issues\n- Add parser validation tests.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
            "Still needs the test.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, max_rounds=2)

    with pytest.raises(AgentLoopError, match="did not evaluate all prior unresolved plan items"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)


def test_issue_loop_plan_first_carries_same_plan_item_across_reviewers_and_rounds(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Revised plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Second revised plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "### Same-plan follow-ups\n- Add the carry-forward orchestration test.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
            "Still needs one plan refinement."
            + prior_plan_item_dispositions("[item-1] same-plan: still need the mixed-reviewer case")
            + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
            "Plan looks sound."
            + prior_plan_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        ],
        gemini_outputs=[
            "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- Google Gemini",
            "Plan looks sound now."
            + prior_plan_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- Google Gemini",
            "Final pass."
            + prior_plan_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- Google Gemini",
        ],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=3)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert any("item-1" in call[-1] for call in claude_calls[1:])
    assert "Approved plan:" in runner.comments[-1]


def test_issue_loop_plan_first_posts_human_readable_item_labels_in_new_and_prior_sections(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Revised plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "### Blocking plan issues\n"
            "- Keep plan-review wording distinct from PR wording.\n"
            "### Same-plan follow-ups\n"
            "- Add one carry-forward plan test.\n"
            "<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
            "Plan looks sound."
            + prior_plan_item_dispositions(
                "[item-1] resolved",
                "[item-2] resolved",
            )
            + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, reviewer="codex", max_rounds=2)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert runner.comments[1] == (
        "**Review verdict:** Blocking\n\n"
        "### Blocking plan issues\n"
        "- Keep plan-review wording distinct from PR wording.\n"
        "\n"
        "### Same-plan follow-ups\n"
        "- Add one carry-forward plan test.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n"
        "-- OpenAI Codex"
    )
    assert runner.comments[3] == (
        "**Review verdict:** Approved\n\n"
        "Plan looks sound.\n\n"
        "### Prior unresolved plan item dispositions\n"
        "- [item-1] Blocking issue from OpenAI Codex, round 1: Keep plan-review wording distinct from PR wording. -> resolved\n"
        "- [item-2] Same-plan follow-up from OpenAI Codex, round 1: Add one carry-forward plan test. -> resolved\n"
        "<!-- AGENT_PLAN_STATE: approved -->\n"
        "-- OpenAI Codex"
    )


def test_issue_loop_plan_first_does_not_expose_same_round_item_ids_to_later_reviewers(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
        ],
        gemini_outputs=[
            "### Same-plan follow-ups\n"
            "- Add the carry-forward orchestration test.\n"
            "<!-- AGENT_PLAN_STATE: blocking -->\n-- Google Gemini",
        ],
        claude_outputs=[
            "Still blocked.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(
        tmp_path,
        coder="codex",
        reviewer=("gemini", "claude"),
        max_rounds=1,
    )

    with pytest.raises(AgentLoopError, match="still reported blocking plan issues after round 1"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    second_reviewer_prompt = [
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] and "planning round 1" in cmd[-1]
    ][0]
    assert "Only items listed under `Prior unresolved plan items from earlier rounds`" in second_reviewer_prompt
    assert "[item-1]" not in second_reviewer_prompt
    assert "### New tracked unresolved items" not in runner.comments[1]


def test_issue_loop_plan_first_uses_compact_context_after_round_one(tmp_path, capsys):
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            structured_plan_revision(
                summary="Resolve item one.",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved", "note": "Covered."}
                ],
                plan_steps=["Revised plan after round one."],
            ),
            structured_plan_revision(
                summary="Resolve item two.",
                prior_plan_item_dispositions=[
                    {"item_id": "item-2", "disposition": "resolved", "note": "Covered."}
                ],
                plan_steps=["Revised plan after round two."],
            ),
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                blocking_plan_issues=["Round one issue."],
            ),
            structured_plan_review(
                state="blocking",
                blocking_plan_issues=["Round two issue."],
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved", "note": "Covered."}
                ],
            ),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[
                    {"item_id": "item-2", "disposition": "resolved", "note": "Covered."}
                ],
            ),
        ],
    )
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer=("codex",),
        max_rounds=3,
        quiet=False,
    )

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"])]
    round_one_review = next(prompt for prompt in prompts if "planning round 1" in prompt)
    round_two_review = next(prompt for prompt in prompts if "Planning round: 2" in prompt and "Role: reviewer" in prompt)
    round_two_revision = next(prompt for prompt in prompts if "Planning round: 2" in prompt and "Role: coder" in prompt)
    assert COMPACT_PLANNING_VOLATILE_TAIL_MARKER not in round_one_review
    assert COMPACT_PLANNING_VOLATILE_TAIL_MARKER in round_two_review
    assert COMPACT_PLANNING_VOLATILE_TAIL_MARKER in round_two_revision

    captured = capsys.readouterr()
    assert "Planning issue #56: invoking Claude (context mode: full)" in captured.err
    assert "Planning round 2: Codex reviewing issue #56 (context mode: compact)" in captured.err
    assert "Planning round 2: Claude revising the plan (context mode: compact)" in captured.err


def test_issue_loop_plan_first_requires_reviewer_human_requirements_resolution(tmp_path, capsys):
    runner = FakeRunner(
        issue_payload={
            "body": "Keep compact context cache-aware.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            "Initial plan covers cache-aware compact context.\n"
            "<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n"
            "### Human requirements\n"
            "- Requirement 1: The plan keeps compact context cache-aware.\n"
            "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            structured_plan_revision(
                summary="Revised plan requires explicit reviewer acknowledgement.",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved", "note": "Reviewer must acknowledge."}
                ],
                plan_steps=["Keep the compact context cache-aware and require reviewer acknowledgement."],
                human_requirements=(
                    "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n"
                    "### Human requirements\n"
                    "- Requirement 1: The revised plan covers the cache-aware compact context requirement."
                ),
            ),
        ],
        codex_outputs=[
            structured_plan_review(state="approved"),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved", "note": "Acknowledged."}
                ],
                human_requirements_resolved=True,
            ),
        ],
    )
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer=("codex",),
        max_rounds=2,
        plan_execution_mode="plan-only",
        quiet=False,
    )

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert "Approved plan:" in runner.comments[-1]
    assert any(
        "approved without acknowledging the signed human requirements" in comment
        for comment in runner.comments
    )
    assert "<!-- HUMAN_REQUIREMENTS_RESOLVED -->" in runner.comments[-2]
    captured = capsys.readouterr()
    assert "approved without acknowledging signed human requirements" in captured.err


def test_issue_loop_plan_first_uses_full_context_when_plan_ledger_incomplete(tmp_path, capsys):
    old_plan = "Old plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    new_plan = "New plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    old_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Old subject item that could be missed.",
        status="blocking",
        source_status="blocking",
    )
    old_reviewer_comment = _attach_round_metadata(
        structured_plan_review(
            state="blocking",
            blocking_plan_issues=["Old subject item that could be missed."],
        ),
        PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject=_plan_subject(old_plan),
            new_items=(old_item,),
            state="blocking",
        ),
    )
    latest_coder_comment = _attach_round_metadata(
        new_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=_plan_subject(new_plan),
            prior_items=(),
        ),
    )
    runner = FakeRunner(
        issue_comments=[
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": old_reviewer_comment},
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:10:00Z", "body": latest_coder_comment},
        ],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer=("codex",),
        quiet=False,
    )

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    review_prompt = [
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:2] == ["codex", "exec"] and "planning round 2" in cmd[-1]
    ][0]
    assert COMPACT_PLANNING_VOLATILE_TAIL_MARKER not in review_prompt
    captured = capsys.readouterr()
    assert "Planning round 2: Codex reviewing issue #56 (context mode: full (ledger incomplete))" in captured.err


def test_issue_loop_plan_first_resumes_with_only_missing_reviewer_for_current_plan(tmp_path):
    current_plan = "Revised plan.\n- Add state reconstruction.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    coder_comment = _attach_round_metadata(
        current_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=_plan_subject(current_plan),
            prior_items=(),
        ),
    )
    codex_comment = _attach_round_metadata(
        "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent="Codex",
            round_number=2,
            subject=_plan_subject(current_plan),
            state="approved",
        ),
    )
    runner = FakeRunner(
        issue_comments=[
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": coder_comment},
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:05:00Z", "body": codex_comment},
        ],
        gemini_outputs=["Plan looks sound too.\n<!-- AGENT_PLAN_STATE: approved -->\n-- Google Gemini"],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    agent_commands = [cmd[0] for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"], ["gemini"])]
    assert agent_commands == ["gemini"]
    assert runner.comments[-1].startswith("Planning complete for issue #56.")


def test_resume_plan_round_prefers_latest_metadata_ledger_for_same_plan_replay():
    current_plan = "Revised plan.\n- Add the active-ledger replay test.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    subject = _plan_subject(current_plan)
    stale_item = UnresolvedReviewItem(
        item_id="item-3",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Stale plan replay item.",
        status="same-plan",
        source_status="same-plan",
    )
    active_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Active plan replay item.",
        status="same-plan",
        source_status="same-plan",
    )
    stale_coder_comment = _attach_round_metadata(
        current_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=subject,
            prior_items=(stale_item,),
            canonical_plan=current_plan,
        ),
    )
    stale_reviewer_comment = _attach_round_metadata(
        "Still needs work."
        + prior_plan_item_dispositions("[item-3] same-plan: stale replay")
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent="Codex",
            round_number=2,
            subject=subject,
            prior_items=(stale_item,),
            dispositions=(
                parse_plan_item_dispositions(
                    prior_plan_item_dispositions("[item-3] same-plan: stale replay"),
                    reviewer="OpenAI Codex",
                )[0],
            ),
            state="blocking",
        ),
    )
    active_coder_comment = _attach_round_metadata(
        current_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=subject,
            prior_items=(active_item,),
            canonical_plan=current_plan,
        ),
    )
    active_reviewer_comment = _attach_round_metadata(
        "Plan looks sound."
        + prior_plan_item_dispositions("[item-1] resolved")
        + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- Google Gemini",
        PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent="Gemini",
            round_number=2,
            subject=subject,
            prior_items=(active_item,),
            dispositions=(
                parse_plan_item_dispositions(
                    prior_plan_item_dispositions("[item-1] resolved"),
                    reviewer="Google Gemini",
                )[0],
            ),
            state="approved",
        ),
    )

    resumed = _resume_plan_round(
        [
            IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=stale_coder_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:01:00Z", body=stale_reviewer_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:02:00Z", body=active_coder_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:03:00Z", body=active_reviewer_comment),
        ],
        configured_reviewers=("codex", "gemini"),
    )

    assert resumed is not None
    current_plan_text, resumed_state = resumed
    assert current_plan_text == current_plan
    assert [item.item_id for item in resumed_state.prior_items] == ["item-1"]
    assert resumed_state.next_unresolved_item_number == 4
    assert [record.metadata.agent for record in resumed_state.completed_reviews] == ["Gemini"]


def test_resume_plan_round_prefers_canonical_plan_metadata():
    public_body = (
        "Revised plan summary.\n\n### Plan steps\n1. Public body copy.\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    canonical_plan = (
        "Revised plan summary.\n\n### Prior plan review item dispositions\n- None.\n\n"
        "### Plan steps\n1. Canonical copy."
    )
    coder_comment = _attach_round_metadata(
        public_body,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=_plan_subject(canonical_plan),
            prior_items=(),
            canonical_plan=canonical_plan,
        ),
    )

    resumed = _resume_plan_round(
        [IssueComment(author="bot", created_at="2026-05-20T09:00:00Z", body=coder_comment)],
        configured_reviewers=("codex", "gemini"),
    )

    assert resumed is not None
    current_plan, state = resumed
    assert current_plan == canonical_plan
    assert state.coder_output == canonical_plan


def test_resume_plan_round_prefers_structured_plan_revision_metadata_for_coder_output():
    public_body = (
        "## Revised plan\n\nRevised plan summary.\n\n### Plan steps\n1. Public body copy.\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    raw_structured_revision = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_revision",
                "state": "blocking",
                "summary": "Revised plan summary.",
                "prior_plan_item_dispositions": [],
                "plan_steps": ["Canonical copy."],
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    parsed = validate_structured_plan_revision(raw_structured_revision)
    assert parsed is not None
    canonical_plan = render_canonical_plan_revision(parsed, ())
    coder_comment = _attach_round_metadata(
        public_body,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=_plan_subject(canonical_plan),
            prior_items=(),
            canonical_plan=canonical_plan,
            raw_structured_coder_response=raw_structured_revision,
        ),
    )

    resumed = _resume_plan_round(
        [IssueComment(author="bot", created_at="2026-05-20T09:00:00Z", body=coder_comment)],
        configured_reviewers=("codex", "gemini"),
    )

    assert resumed is not None
    current_plan, state = resumed
    assert current_plan == canonical_plan
    assert state.coder_output == raw_structured_revision
    assert validate_structured_plan_revision(state.coder_output) is not None
    assert '"kind": "plan_revision"' not in _strip_round_metadata(coder_comment)


def test_resume_plan_round_falls_back_to_raw_body_for_markdown_plan():
    plan = "Revised plan.\n- Add state reconstruction.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    coder_comment = _attach_round_metadata(
        plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=_plan_subject(plan),
            prior_items=(),
        ),
    )

    resumed = _resume_plan_round(
        [IssueComment(author="bot", created_at="2026-05-20T09:00:00Z", body=coder_comment)],
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    current_plan, state = resumed
    assert current_plan == plan
    assert state.coder_output == plan


def test_plan_subject_ignores_trailing_whitespace_added_by_metadata_round_trip():
    plan = "Revised plan.\n- Add state reconstruction.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"

    attached = _attach_round_metadata(
        f"{plan}\n",
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=_plan_subject(f"{plan}\n"),
            prior_items=(),
        ),
    )

    assert _plan_subject(f"{plan}\n") == _plan_subject(_strip_round_metadata(attached))


def test_round_metadata_round_trip_preserves_canonical_plan():
    metadata = PostedRoundMetadata(
        flow="plan",
        role="coder",
        agent="Claude",
        round_number=2,
        subject="abc",
        canonical_plan="Summary\n\n### Plan steps\n1. Canonical step.",
    )

    assert _decode_round_metadata(_encode_round_metadata(metadata)).canonical_plan == metadata.canonical_plan


def test_round_metadata_round_trip_preserves_compact_prior_summaries():
    metadata = PostedRoundMetadata(
        flow="plan",
        role="coder",
        agent="Claude",
        round_number=2,
        subject="abc",
        compact_prior_summaries=("[item-1] resolved: full prior text",),
    )

    decoded = _decode_round_metadata(_encode_round_metadata(metadata))

    assert decoded.compact_prior_summaries == metadata.compact_prior_summaries


def test_decode_old_round_metadata_defaults_compact_prior_summaries_to_empty():
    payload = {
        "flow": "plan",
        "role": "coder",
        "agent": "Claude",
        "round_number": 2,
        "subject": "abc",
        "prior_items": [],
        "dispositions": [],
        "new_items": [],
        "state": None,
        "canonical_plan": None,
        "raw_structured_coder_response": None,
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).decode("ascii")

    assert _decode_round_metadata(encoded).compact_prior_summaries == ()


def test_resume_plan_round_restores_compact_prior_summaries_across_subject_change():
    old_plan = "Old plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    new_plan = "New plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    old_subject = _plan_subject(old_plan)
    new_subject = _plan_subject(new_plan)
    old_summary = "[item-1] resolved: old-subject resolved summary"
    old_coder_comment = _attach_round_metadata(
        old_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=old_subject,
            compact_prior_summaries=(old_summary,),
        ),
    )
    new_coder_comment = _attach_round_metadata(
        new_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=3,
            subject=new_subject,
            compact_prior_summaries=(old_summary,),
        ),
    )

    resumed = _resume_plan_round(
        [
            IssueComment(author="bot", created_at="2026-05-20T09:00:00Z", body=old_coder_comment),
            IssueComment(author="bot", created_at="2026-05-20T09:10:00Z", body=new_coder_comment),
        ],
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    _current_plan, state = resumed
    assert state.compact_prior_summaries == (old_summary,)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"flow": "plan"},
        {
            "flow": "plan",
            "role": "coder",
            "agent": "Claude",
            "round_number": "not-an-int",
            "subject": "abc",
        },
    ],
)
def test_decode_round_metadata_rejects_missing_or_invalid_required_fields(payload):
    encoded = json.dumps(payload).encode("utf-8")

    with pytest.raises(AgentLoopError, match="Invalid AGENT_LOOP_META payload"):
        _decode_round_metadata(encoded=base64.urlsafe_b64encode(encoded).decode("ascii"))


@pytest.mark.parametrize(
    "line",
    [
        "[item-1] same-plan: none",
        "[item-1] still blocking: none",
        "[item-1] future follow-up: none",
    ],
)
def test_issue_loop_plan_first_rejects_contradictory_disposition_before_extra_revision(
    tmp_path, line
):
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Revised plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "### Same-plan follow-ups\n- Add the carry-forward orchestration test.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
            "Plan looks sound now."
            + prior_plan_item_dispositions(line)
            + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, reviewer="codex", max_rounds=3)

    with pytest.raises(AgentLoopError, match="use `resolved` when nothing remains"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 2


def test_issue_loop_plan_first_plan_only_does_not_publish_approved_future_followups(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Revised plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "### Same-plan follow-ups\n- Tighten the prompt wording.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
            "Looks good."
            + prior_plan_item_dispositions("[item-1] future follow-up: document parser helper reuse separately")
            + "\n### Future follow-ups\n- Add a later cleanup to dedupe shared prompt rendering.\n"
            + "<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert runner.issues == []
    summary = runner.comments[-1]
    assert summary.startswith("Planning complete for issue #56.")
    assert "Approved plan future follow-ups:" in summary
    assert "document parser helper reuse separately" in summary
    assert "Add a later cleanup to dedupe shared prompt rendering." in summary
    assert "not carried into PR review" in summary
    assert "not PR prior review items" in summary
    assert "Filed future follow-up issues:" not in summary
    assert "<!-- AGENT_PLAN_APPROVED_FOLLOWUPS:" in summary
    assert "mode=summarize" in summary


def test_issue_loop_plan_first_decompose_only_summarizes_instead_of_filing_plan_followups(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Plan:\n- Split the implementation into phases.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            plan_decomposition_json(
                {
                    "title": "Schema helpers",
                    "scope": "Add parser dataclasses and tests.",
                    "non_goals": "No live orchestrator switch.",
                    "dependency_notes": "First phase; no dependencies.",
                    "rollout_risk": "low - internal only.",
                    "validation": "Run python -m pytest tests/test_agent_loop.py.",
                    "parent_context": "Approved plan slice: add schema helpers and preserve behavior.",
                    "automation": "agent-pr",
                    "depends_on": [],
                }
            ),
        ],
        codex_outputs=[
            structured_plan_review(
                state="approved",
                future_followups=["Add a later cleanup to dedupe shared prompt rendering."],
            ),
        ],
        issue_urls=["https://github.com/OWNER/REPO/issues/101"],
    )
    config = make_config(
        tmp_path,
        approved_followups="issue",
        plan_execution_mode="decompose-only",
    )

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert len(runner.issues) == 1
    assert runner.issues[0]["title"] == "Phase 1: Schema helpers (from #56)"
    assert not any(
        issue["title"].startswith("Follow up future plan-review note:")
        for issue in runner.issues
    )
    planning_summary = runner.comments[2]
    assert planning_summary.startswith("Planning complete for issue #56.")
    assert "Approved plan future follow-ups:" in planning_summary
    assert "Add a later cleanup to dedupe shared prompt rendering." in planning_summary
    assert "Filed future follow-up issues:" not in planning_summary
    assert "mode=summarize" in planning_summary
    assert "mode=issue" not in planning_summary


def test_issue_loop_plan_first_files_approved_future_followups_before_implementation(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(
                state="approved",
                future_followups=["Add a later cleanup to dedupe shared prompt rendering."],
            ),
            structured_pr_review(state="approved", summary="LGTM."),
        ],
        issue_urls=["https://github.com/OWNER/REPO/issues/99"],
    )
    config = make_config(tmp_path, approved_followups="issue")

    assert (
        run_issue_loop(
            runner,
            issue_number=56,
            config=config,
            plan_first=True,
            implement_after_approval=True,
        )
        == 0
    )

    assert len(runner.issues) == 1
    assert runner.issues[0]["title"] == (
        "Follow up future plan-review note: Add a later cleanup to dedupe shared prompt rendering."
    )
    issue_body = runner.issues[0]["body"]
    assert "Parent issue: #56" in issue_body
    assert "Approved plan hash:" in issue_body
    assert "Planning round(s): 1" in issue_body
    assert "Original plan item ID(s): item-1" in issue_body
    assert "Codex" in issue_body
    assert "outside the current implementation scope" in issue_body
    assert "not a PR-review prior item" in issue_body
    summary = runner.comments[2]
    assert summary.startswith("Planning complete for issue #56.")
    assert "Filed future follow-up issues:" in summary
    assert "https://github.com/OWNER/REPO/issues/99" in summary
    assert "Approved plan future follow-ups:" not in summary
    assert "<!-- AGENT_PLAN_APPROVED_FOLLOWUPS:" in summary

    issue_create_index = command_index(runner.commands, ["gh", "issue", "create"])
    second_claude_index = command_index(
        runner.commands,
        ["claude", "--print"],
        start=command_index(runner.commands, ["claude", "--print"]) + 1,
    )
    assert issue_create_index < second_claude_index


def test_issue_loop_plan_first_ignore_mode_keeps_pr_prior_ledger_clean(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(
                state="approved",
                future_followups=["Track a separate planning cleanup later."],
            ),
            structured_pr_review(state="approved", summary="LGTM."),
        ],
    )
    config = make_config(tmp_path, approved_followups="ignore")

    assert (
        run_issue_loop(
            runner,
            issue_number=56,
            config=config,
            plan_first=True,
            implement_after_approval=True,
        )
        == 0
    )

    assert runner.issues == []
    planning_summary = runner.comments[2]
    assert "Approved plan future follow-ups:" in planning_summary
    assert "Track a separate planning cleanup later." in planning_summary
    assert "not carried into PR review" in planning_summary
    pr_review_prompt = [
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:2] == ["codex", "exec"] and '"kind": "pr_review"' in cmd[-1]
    ][0]
    assert "Only items listed under `Prior unresolved review items from earlier rounds`" in pr_review_prompt
    assert "Track a separate planning cleanup later." not in pr_review_prompt
    assert "planning-stage `item-*` IDs and approved\nplan future follow-ups" in pr_review_prompt
    assert "prior_plan_item_dispositions" in pr_review_prompt


def test_issue_loop_plan_first_deduplicates_plan_followup_issues_across_reviewers(tmp_path):
    runner = FakeRunner(
        gemini_outputs=[
            "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Google Gemini",
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini",
        ],
        codex_outputs=[
            structured_plan_review(
                state="approved",
                future_followups=[
                    "**Remote validation**: Validate explicit workdir git remotes against the target repo.",
                ],
            ),
            structured_pr_review(state="approved", summary="LGTM."),
        ],
        claude_outputs=[
            structured_plan_review(
                state="approved",
                future_followups=[
                    "**Remote validation**: Validate explicit workdir git remotes against the target repo.",
                ],
                reviewer="Anthropic Claude",
            ),
            structured_pr_review(state="approved", summary="LGTM.", reviewer="Anthropic Claude"),
        ],
        issue_urls=["https://github.com/OWNER/REPO/issues/99"],
    )
    config = make_config(
        tmp_path,
        coder="gemini",
        reviewer=("codex", "claude"),
        approved_followups="issue",
    )

    assert (
        run_issue_loop(
            runner,
            issue_number=56,
            config=config,
            plan_first=True,
            implement_after_approval=True,
        )
        == 0
    )

    assert len(runner.issues) == 1
    body = runner.issues[0]["body"]
    assert "Reviewers: Codex, Claude" in body
    assert "Original plan item ID(s): item-1, item-2" in body
    assert body.count("**Remote validation**") == 3
    assert any(
        "Reconciliation: 1 filed, 1 deduplicated, 0 skipped by cap." in comment
        for comment in runner.comments
    )


def test_issue_loop_plan_first_plan_followup_marker_prevents_duplicate_issue_creation(tmp_path):
    plan = "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    plan_hash = approved_plan_hash(plan)
    future_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="Codex",
        source_round=1,
        text="Add a later cleanup to dedupe shared prompt rendering.",
        status="future",
        source_status="future",
    )
    runner = FakeRunner(
        claude_outputs=[
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_pr_review(state="approved", summary="LGTM."),
        ],
        issue_comments=[
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:00Z",
                "body": _attach_round_metadata(
                    plan,
                    PostedRoundMetadata(
                        flow="plan",
                        role="coder",
                        agent="Claude",
                        round_number=1,
                        subject=_plan_subject(plan),
                    ),
                ),
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:01Z",
                "body": _attach_round_metadata(
                    structured_plan_review(
                        state="approved",
                        future_followups=["Add a later cleanup to dedupe shared prompt rendering."],
                    ),
                    PostedRoundMetadata(
                        flow="plan",
                        role="reviewer",
                        agent="Codex",
                        round_number=1,
                        subject=_plan_subject(plan),
                        new_items=(future_item,),
                        state="approved",
                    ),
                ),
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:02Z",
                "body": (
                    "Planning complete for issue #56.\n\n"
                    "<!-- AGENT_PLAN_APPROVED_FOLLOWUPS: "
                    f"issue=56 plan={plan_hash} mode=issue -->\n"
                    "-- coding-review-agent-loop"
                ),
            },
        ],
    )
    config = make_config(tmp_path, approved_followups="issue")

    assert (
        run_issue_loop(
            runner,
            issue_number=56,
            config=config,
            plan_first=True,
            implement_after_approval=True,
        )
        == 0
    )

    assert runner.issues == []
    assert not any(
        "Filed future follow-up issues:" in comment or "Approved plan future follow-ups:" in comment
        for comment in runner.comments
    )


def test_issue_loop_plan_first_decompose_only_creates_child_issues(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Plan:\n- Add schema helpers.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            plan_decomposition_json(
                {
                    "title": "Schema helpers",
                    "scope": "Add parser dataclasses and tests.",
                    "non_goals": "No live orchestrator switch.",
                    "dependency_notes": "First phase; no dependencies.",
                    "rollout_risk": "low - internal only.",
                    "validation": "Run python -m pytest tests/test_agent_loop.py.",
                    "parent_context": "Approved plan slice: add schema helpers and preserve behavior.",
                    "automation": "agent-pr",
                    "depends_on": [],
                },
                {
                    "title": "Human rollout checkpoint",
                    "scope": "Human validates rollout readiness.",
                    "non_goals": "No code changes.",
                    "dependency_notes": "Depends on Schema helpers.",
                    "rollout_risk": "medium - manual checkpoint.",
                    "validation": "Human must add a remark and close the issue.",
                    "parent_context": "Approved plan slice: stop for human validation.",
                    "automation": "manual-close",
                    "depends_on": ["Schema helpers"],
                },
            ),
        ],
        codex_outputs=[
            "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        ],
        issue_urls=[
            "https://github.com/OWNER/REPO/issues/101",
            "https://github.com/OWNER/REPO/issues/102",
        ],
    )
    config = make_config(tmp_path, plan_execution_mode="decompose-only")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert len(runner.issues) == 2
    assert runner.issues[0]["title"] == "Phase 1: Schema helpers (from #56)"
    assert "Run `agent-loop issue <this issue number>`" in runner.issues[0]["body"]
    assert "Approved plan slice: add schema helpers" in runner.issues[0]["body"]
    assert runner.issues[1]["title"] == "[Human] Phase 2: Human rollout checkpoint (from #56)"
    assert "depends on #101: Schema helpers" in runner.issues[1]["body"]
    assert "human should add the required remark/update and close this issue" in runner.issues[1]["body"]
    summary = runner.comments[-1]
    assert summary.startswith("Approved plan decomposed for issue #56.")
    assert "Every phase above has a GitHub child issue" in summary
    assert "<!-- AGENT_PLAN_DECOMPOSITION:" in summary
    assert not any(cmd[:3] == ["gh", "pr", "view"] for cmd, _cwd in runner.commands)


def test_issue_loop_plan_first_decompose_only_is_idempotent(tmp_path):
    plan = "Plan:\n- Add schema helpers.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    summary = format_decomposition_parent_summary(
        parent_issue=56,
        mode="decompose-only",
        plan_hash=approved_plan_hash(plan),
        created=(),
    )
    runner = FakeRunner(
        issue_comments=[
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:00Z",
                "body": _attach_round_metadata(
                    plan,
                    PostedRoundMetadata(
                        flow="plan",
                        role="coder",
                        agent="Claude",
                        round_number=1,
                        subject=_plan_subject(plan),
                    ),
                ),
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:01Z",
                "body": _attach_round_metadata(
                    "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
                    PostedRoundMetadata(
                        flow="plan",
                        role="reviewer",
                        agent="Codex",
                        round_number=1,
                        subject=_plan_subject(plan),
                        state="approved",
                    ),
                ),
            },
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:02Z", "body": summary},
        ],
    )
    config = make_config(tmp_path, plan_execution_mode="decompose-only")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert runner.issues == []
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def test_issue_loop_plan_first_implement_by_phase_rerun_without_handoff_implements_once(tmp_path):
    plan = "Plan:\n- Add schema helpers.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    child = CreatedPhaseIssue(
        phase=RecordedPhase(title="Schema helpers", automation="agent-pr"),
        issue_url="https://github.com/OWNER/REPO/issues/99",
        issue_number=99,
    )
    summary = format_decomposition_parent_summary(
        parent_issue=56,
        mode="implement-by-phase",
        plan_hash=approved_plan_hash(plan),
        created=(child,),
    )
    runner = FakeRunner(
        claude_outputs=[
            "Implemented first phase.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        issue_comments=[
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:00Z",
                "body": _attach_round_metadata(
                    plan,
                    PostedRoundMetadata(
                        flow="plan",
                        role="coder",
                        agent="Claude",
                        round_number=1,
                        subject=_plan_subject(plan),
                    ),
                ),
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:01Z",
                "body": _attach_round_metadata(
                    "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
                    PostedRoundMetadata(
                        flow="plan",
                        role="reviewer",
                        agent="Codex",
                        round_number=1,
                        subject=_plan_subject(plan),
                        state="approved",
                    ),
                ),
            },
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:02Z", "body": summary},
        ],
        pr_payload={"body": "Fixes #99"},
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert runner.issues == []
    assert any("<!-- AGENT_PLAN_PHASE_IMPLEMENTATION:" in comment for comment in runner.comments)
    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 1
    assert "GitHub issue #99" in claude_calls[0][-1]


def test_issue_loop_plan_first_implement_by_phase_rerun_with_handoff_stops(tmp_path, capsys):
    plan = "Plan:\n- Add schema helpers.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    child = CreatedPhaseIssue(
        phase=RecordedPhase(title="Schema helpers", automation="agent-pr"),
        issue_url="https://github.com/OWNER/REPO/issues/99",
        issue_number=99,
    )
    summary = format_decomposition_parent_summary(
        parent_issue=56,
        mode="implement-by-phase",
        plan_hash=approved_plan_hash(plan),
        created=(child,),
    )
    handoff = format_phase_implementation_handoff_comment(
        parent_issue=56,
        mode="implement-by-phase",
        plan_hash=approved_plan_hash(plan),
        phase_index=1,
        created=child,
    )
    runner = FakeRunner(
        issue_comments=[
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:00Z",
                "body": _attach_round_metadata(
                    plan,
                    PostedRoundMetadata(
                        flow="plan",
                        role="coder",
                        agent="Claude",
                        round_number=1,
                        subject=_plan_subject(plan),
                    ),
                ),
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:01Z",
                "body": _attach_round_metadata(
                    "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
                    PostedRoundMetadata(
                        flow="plan",
                        role="reviewer",
                        agent="Codex",
                        round_number=1,
                        subject=_plan_subject(plan),
                        state="approved",
                    ),
                ),
            },
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:02Z", "body": summary},
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:03Z", "body": handoff},
        ],
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    output = capsys.readouterr().out
    assert "already handed off to child issue #99" in output
    assert "agent-loop issue 99" in output
    assert runner.issues == []
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


def test_issue_loop_plan_first_implement_by_phase_human_first_rerun_does_not_handoff(tmp_path):
    plan = "Plan:\n- Validate migration manually first.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    child = CreatedPhaseIssue(
        phase=RecordedPhase(title="Manual readiness check", automation="human-action"),
        issue_url="https://github.com/OWNER/REPO/issues/99",
        issue_number=99,
    )
    summary = format_decomposition_parent_summary(
        parent_issue=56,
        mode="implement-by-phase",
        plan_hash=approved_plan_hash(plan),
        created=(child,),
    )
    runner = FakeRunner(
        issue_comments=[
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:00Z",
                "body": _attach_round_metadata(
                    plan,
                    PostedRoundMetadata(
                        flow="plan",
                        role="coder",
                        agent="Claude",
                        round_number=1,
                        subject=_plan_subject(plan),
                    ),
                ),
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:01Z",
                "body": _attach_round_metadata(
                    "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
                    PostedRoundMetadata(
                        flow="plan",
                        role="reviewer",
                        agent="Codex",
                        round_number=1,
                        subject=_plan_subject(plan),
                        state="approved",
                    ),
                ),
            },
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:02Z", "body": summary},
        ],
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert runner.issues == []
    assert not any("<!-- AGENT_PLAN_PHASE_IMPLEMENTATION:" in comment for comment in runner.comments)
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def test_issue_loop_plan_first_implement_by_phase_stops_on_human_first_phase(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Plan:\n- Validate migration manually first.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            plan_decomposition_json(
                {
                    "title": "Manual readiness check",
                    "scope": "Human validates external readiness.",
                    "non_goals": "No agent PR.",
                    "dependency_notes": "First phase; no dependencies.",
                    "rollout_risk": "medium - manual readiness gate.",
                    "validation": "Human remark and closure required.",
                    "parent_context": "Approved plan slice: manual readiness gate.",
                    "automation": "human-action",
                    "depends_on": [],
                }
            ),
        ],
        codex_outputs=["Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert len(runner.issues) == 1
    assert runner.issues[0]["title"].startswith("[Human] Phase 1")
    assert not any("<!-- AGENT_PLAN_PHASE_IMPLEMENTATION:" in comment for comment in runner.comments)
    assert not any(cmd[:3] == ["gh", "pr", "view"] for cmd, _cwd in runner.commands)


def test_issue_loop_plan_first_implement_by_phase_implements_first_agent_phase(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Plan:\n- Add schema helpers.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            plan_decomposition_json(),
            "Implemented first phase.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        issue_urls=["https://github.com/OWNER/REPO/issues/99"],
        pr_payload={"body": "Fixes #99"},
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert len(runner.issues) == 1
    decomposition_index = next(
        index for index, comment in enumerate(runner.comments) if "<!-- AGENT_PLAN_DECOMPOSITION:" in comment
    )
    handoff_index = next(
        index for index, comment in enumerate(runner.comments) if "<!-- AGENT_PLAN_PHASE_IMPLEMENTATION:" in comment
    )
    implementation_index = next(
        index for index, comment in enumerate(runner.comments) if comment.startswith("Implemented first phase.")
    )
    assert decomposition_index < handoff_index < implementation_index
    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 3
    assert "GitHub issue #99" in claude_calls[2][-1]
    assert "Approved implementation plan" in claude_calls[2][-1]


def test_issue_loop_plan_first_implement_by_phase_missing_child_number_does_not_handoff(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Plan:\n- Add schema helpers.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            plan_decomposition_json(),
        ],
        codex_outputs=[
            "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        ],
        issue_urls=[None],
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    with pytest.raises(AgentLoopError, match="child issue number"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert not any("<!-- AGENT_PLAN_PHASE_IMPLEMENTATION:" in comment for comment in runner.comments)


def test_phase_implementation_handoff_rejects_malformed_marker():
    comment = IssueComment(
        author="bot",
        created_at="2026-05-23T00:00:00Z",
        body="<!-- AGENT_PLAN_PHASE_IMPLEMENTATION: not-valid-base64 -->",
    )

    with pytest.raises(AgentLoopError, match="Invalid AGENT_PLAN_PHASE_IMPLEMENTATION payload"):
        find_existing_phase_implementation_handoff(
            (comment,),
            parent_issue=56,
            plan_hash="abc123",
            mode="implement-by-phase",
            phase_index=1,
            child_issue_number=99,
        )


def test_issue_loop_plan_first_keeps_blocking_review_when_future_followups_are_misclassified(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Revised plan with focused tests.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Still blocked.\n\n"
            "### Blocking plan issues\n"
            "- Add parser coverage for blocking reviews with stray future follow-ups.\n\n"
            "### Same-plan follow-ups\n"
            "- Tighten the plan-review prompt wording.\n\n"
            "### Future follow-ups\n"
            "- Consider a later prompt dedupe cleanup.\n\n"
            "<!-- AGENT_PLAN_STATE: blocking -->\n"
            "-- OpenAI Codex",
            "Plan looks sound."
            + prior_plan_item_dispositions("[item-1] resolved", "[item-2] resolved")
            + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert "Add parser coverage for blocking reviews with stray future follow-ups." in claude_calls[1][-1]
    assert "Tighten the plan-review prompt wording." in claude_calls[1][-1]
    assert runner.comments[1].startswith("**Review verdict:** Blocking\n\nStill blocked.")
    assert "### Future follow-ups" not in runner.comments[1]


def test_issue_loop_plan_first_can_implement_after_approval(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)

    assert (
        run_issue_loop(
            runner,
            issue_number=56,
            config=config,
            plan_first=True,
            implement_after_approval=True,
        )
        == 0
    )

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 2
    assert "Approved implementation plan" in claude_calls[1][-1]
    assert "include `Fixes #56` or another direct reference to issue #56" in claude_calls[1][-1]
    first_claude_index = command_index(runner.commands, ["claude", "--print"])
    fetch_index = command_index(runner.commands, ["git", "fetch", "origin"])
    switch_index = command_index(runner.commands, ["git", "switch", "main"])
    second_claude_index = command_index(runner.commands, ["claude", "--print"], start=first_claude_index + 1)
    assert first_claude_index < fetch_index < switch_index < second_claude_index
    assert len(runner.comments) == 6
    assert "<!-- AGENT_PLAN_ONE_SHOT_IMPL:" in runner.comments[3]
    assert runner.comments[4].startswith("Implemented approved plan.")
    assert runner.comments[5].startswith("**Review verdict:** Approved\n\nLGTM.")


def test_issue_loop_rejects_pr_without_issue_reference_in_body(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Fixed issue.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        pr_payload={
            "number": 77,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/77",
            "body": "Summary only.",
        },
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="does not reference issue #56") as excinfo:
        run_issue_loop(runner, issue_number=56, config=config)

    assert "Edit the PR description on GitHub" in str(excinfo.value)
    assert "rerun the orchestrator as `agent-loop pr 77` to continue the review" in str(excinfo.value)


def test_issue_loop_plan_first_implementation_rejects_pr_without_issue_reference_in_body(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        ],
        pr_payload={
            "number": 77,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/77",
            "body": "Summary only.",
        },
    )
    config = make_config(tmp_path, reviewer=("codex",))

    with pytest.raises(AgentLoopError, match="does not reference issue #56") as excinfo:
        run_issue_loop(
            runner,
            issue_number=56,
            config=config,
            plan_first=True,
            implement_after_approval=True,
        )

    assert "Edit the PR description on GitHub" in str(excinfo.value)
    assert "rerun the orchestrator as `agent-loop pr 77` to continue the review" in str(excinfo.value)


def test_issue_loop_plan_first_one_shot_posts_handoff_after_pr_creation(tmp_path):
    plan = "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    runner = FakeRunner(
        claude_outputs=[
            plan,
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)

    assert (
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True)
        == 0
    )

    handoff_comments = [c for c in runner.comments if "<!-- AGENT_PLAN_ONE_SHOT_IMPL:" in c]
    assert len(handoff_comments) == 1
    assert f"Plan hash: {approved_plan_hash(plan)}" in handoff_comments[0]
    assert "Plan subject:" in handoff_comments[0]
    assert "PR #77" in handoff_comments[0]


def test_issue_loop_plan_first_one_shot_rerun_resumes_pr_loop(tmp_path):
    plan = "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    handoff = format_one_shot_impl_handoff_comment(
        parent_issue=56,
        mode="implement-one-shot",
        plan_hash=approved_plan_hash(plan),
        plan_subject=_plan_subject(plan),
        pr_number=77,
        pr_head_sha="abc123",
    )
    runner = FakeRunner(
        issue_comments=[
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:00Z",
                "body": _attach_round_metadata(
                    plan,
                    PostedRoundMetadata(
                        flow="plan",
                        role="coder",
                        agent="Claude",
                        round_number=1,
                        subject=_plan_subject(plan),
                    ),
                ),
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:01Z",
                "body": _attach_round_metadata(
                    "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
                    PostedRoundMetadata(
                        flow="plan",
                        role="reviewer",
                        agent="Codex",
                        round_number=1,
                        subject=_plan_subject(plan),
                        state="approved",
                    ),
                ),
            },
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:02Z", "body": handoff},
        ],
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True) == 0

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 0
    assert any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


def test_issue_loop_plan_first_one_shot_rerun_with_closed_pr_stops(tmp_path, capsys):
    plan = "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    handoff = format_one_shot_impl_handoff_comment(
        parent_issue=56,
        mode="implement-one-shot",
        plan_hash=approved_plan_hash(plan),
        plan_subject=_plan_subject(plan),
        pr_number=77,
        pr_head_sha="abc123",
    )
    runner = FakeRunner(
        issue_comments=[
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:00Z",
                "body": _attach_round_metadata(
                    plan,
                    PostedRoundMetadata(
                        flow="plan",
                        role="coder",
                        agent="Claude",
                        round_number=1,
                        subject=_plan_subject(plan),
                    ),
                ),
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:01Z",
                "body": _attach_round_metadata(
                    "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
                    PostedRoundMetadata(
                        flow="plan",
                        role="reviewer",
                        agent="Codex",
                        round_number=1,
                        subject=_plan_subject(plan),
                        state="approved",
                    ),
                ),
            },
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:02Z", "body": handoff},
        ],
        pr_payload={"state": "CLOSED"},
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True) == 0

    output = capsys.readouterr().out
    assert "PR #77" in output
    assert "closed" in output
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


def test_issue_loop_plan_first_one_shot_rerun_hash_mismatch_reimplements(tmp_path):
    plan = "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    old_plan = "Plan:\n- Old approach that was replaced.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    old_handoff = format_one_shot_impl_handoff_comment(
        parent_issue=56,
        mode="implement-one-shot",
        plan_hash=approved_plan_hash(old_plan),
        plan_subject=_plan_subject(old_plan),
        pr_number=99,
        pr_head_sha=None,
    )
    runner = FakeRunner(
        issue_comments=[
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:00Z",
                "body": _attach_round_metadata(
                    plan,
                    PostedRoundMetadata(
                        flow="plan",
                        role="coder",
                        agent="Claude",
                        round_number=1,
                        subject=_plan_subject(plan),
                    ),
                ),
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:01Z",
                "body": _attach_round_metadata(
                    "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
                    PostedRoundMetadata(
                        flow="plan",
                        role="reviewer",
                        agent="Codex",
                        round_number=1,
                        subject=_plan_subject(plan),
                        state="approved",
                    ),
                ),
            },
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:02Z", "body": old_handoff},
        ],
        claude_outputs=[
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True) == 0

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 1
    assert "Approved implementation plan" in claude_calls[0][-1]


def test_issue_loop_plan_first_one_shot_rerun_pr_missing_issue_reference(tmp_path):
    plan = "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    handoff = format_one_shot_impl_handoff_comment(
        parent_issue=56,
        mode="implement-one-shot",
        plan_hash=approved_plan_hash(plan),
        plan_subject=_plan_subject(plan),
        pr_number=77,
        pr_head_sha="abc123",
    )
    runner = FakeRunner(
        issue_comments=[
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:00Z",
                "body": _attach_round_metadata(
                    plan,
                    PostedRoundMetadata(
                        flow="plan",
                        role="coder",
                        agent="Claude",
                        round_number=1,
                        subject=_plan_subject(plan),
                    ),
                ),
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:01Z",
                "body": _attach_round_metadata(
                    "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
                    PostedRoundMetadata(
                        flow="plan",
                        role="reviewer",
                        agent="Codex",
                        round_number=1,
                        subject=_plan_subject(plan),
                        state="approved",
                    ),
                ),
            },
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:02Z", "body": handoff},
        ],
        pr_payload={
            "number": 77,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/77",
            "body": "No issue reference here.",
        },
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="does not reference issue #56") as excinfo:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True)

    assert "Edit the PR description on GitHub" in str(excinfo.value)
    assert "rerun the orchestrator as `agent-loop pr 77` to continue the review" in str(excinfo.value)
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def test_is_clarification_request_detects_marker():
    assert is_clarification_request("need more info\n<!-- AGENT_CLARIFY -->")
    assert is_clarification_request("<!-- agent_clarify -->")
    assert not is_clarification_request("done\n<!-- AGENT_STATE: blocking -->")


def test_is_clarification_request_state_marker_after_clarify_takes_precedence():
    # AGENT_PLAN_STATE after inline AGENT_CLARIFY example: issue #216 / #278 shape.
    # Inline (non-standalone) AGENT_CLARIFY never triggers, regardless of state markers.
    plan_with_embedded_clarify = (
        "Here is my plan.\n\n"
        "If I needed clarification I would emit <!-- AGENT_CLARIFY --> as a marker.\n\n"
        "But I have enough information, so here is the full plan:\n\n"
        "1. Do step one\n2. Do step two\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    assert not is_clarification_request(plan_with_embedded_clarify)

    # Inline AGENT_CLARIFY without any state marker: still not clarification.
    inline_only = (
        "The protocol supports <!-- AGENT_CLARIFY --> for clarification requests.\n\n"
        "Here is my fix."
    )
    assert not is_clarification_request(inline_only)

    # AGENT_STATE after inline AGENT_CLARIFY example: PR/coder blocking response.
    pr_response_with_embedded_clarify = (
        "The protocol supports <!-- AGENT_CLARIFY --> for clarification requests.\n\n"
        "Here is my fix.\n\n"
        "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    assert not is_clarification_request(pr_response_with_embedded_clarify)

    # AGENT_PR after inline AGENT_CLARIFY example: coder PR-creation response.
    pr_created_with_embedded_clarify = (
        "Use <!-- AGENT_CLARIFY --> if you need more info.\n\n"
        "Implemented the fix.\n\n"
        "<!-- AGENT_PR: 42 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    assert not is_clarification_request(pr_created_with_embedded_clarify)

    # PR URL after inline AGENT_CLARIFY: treated as final state marker.
    pr_url_with_embedded_clarify = (
        "Use <!-- AGENT_CLARIFY --> for questions.\n\n"
        "See https://github.com/OWNER/REPO/pull/99 for the PR."
    )
    assert not is_clarification_request(pr_url_with_embedded_clarify)

    # Real clarification request: standalone AGENT_CLARIFY is the final marker.
    real_clarify = "Which endpoint should I use?\n<!-- AGENT_CLARIFY -->\n-- Anthropic Claude"
    assert is_clarification_request(real_clarify)

    # Standalone AGENT_CLARIFY on its own line, after a state marker in prose.
    clarify_after_state = (
        "Round 1 ended with <!-- AGENT_STATE: blocking -->, but I still need more info.\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(clarify_after_state)

    # Standalone AGENT_CLARIFY on its own line, appearing after AGENT_PLAN_STATE in prose.
    plan_state_in_prose_clarify_last = (
        "The previous round used <!-- AGENT_PLAN_STATE: blocking --> to signal issues,\n"
        "but now I have a question:\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(plan_state_in_prose_clarify_last)


def test_is_clarification_request_standalone_marker_positional_semantics():
    # Standalone AGENT_PLAN_STATE footer appearing BEFORE a standalone AGENT_CLARIFY
    # does NOT suppress it — AGENT_CLARIFY is the final marker and wins.
    plan_footer_then_clarify_appendix = (
        "Plan content.\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n"
        "-- Anthropic Claude\n\n"
        "Appendix:\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(plan_footer_then_clarify_appendix)

    # Standalone AGENT_STATE appearing BEFORE AGENT_CLARIFY also does not suppress.
    state_then_clarify = (
        "<!-- AGENT_STATE: blocking -->\n\n"
        "Note:\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(state_then_clarify)

    # Standalone AGENT_STATE appearing AFTER AGENT_CLARIFY does suppress it.
    clarify_then_state = (
        "<!-- AGENT_CLARIFY -->\n"
        "<!-- AGENT_STATE: blocking -->"
    )
    assert not is_clarification_request(clarify_then_state)

    # Inline (non-standalone) AGENT_STATE in prose does NOT suppress AGENT_CLARIFY —
    # it may be a quoted reference to a previous round's state.
    inline_state_then_clarify = (
        "Round 1 ended with <!-- AGENT_STATE: blocking -->, but I still need more info.\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(inline_state_then_clarify)

    # Inline AGENT_PLAN_STATE in prose also does not suppress.
    inline_plan_state_then_clarify = (
        "The previous round used <!-- AGENT_PLAN_STATE: blocking --> to signal issues,\n"
        "but now I have a question:\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(inline_plan_state_then_clarify)


def test_is_clarification_request_pr_marker_takes_precedence():
    # AGENT_PR: N standalone marker appearing AFTER AGENT_CLARIFY suppresses it.
    pr_after_clarify = (
        "<!-- AGENT_CLARIFY -->\n"
        "Actually I have enough info.\n"
        "<!-- AGENT_PR: 55 -->"
    )
    assert not is_clarification_request(pr_after_clarify)

    # AGENT_PR: N standalone marker appearing BEFORE AGENT_CLARIFY does NOT suppress —
    # AGENT_CLARIFY is the final marker and wins.
    pr_before_clarify = (
        "<!-- AGENT_PR: 55 -->\n"
        "<!-- AGENT_STATE: blocking -->\n\n"
        "Note:\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(pr_before_clarify)


def test_is_clarification_request_ignores_fenced_code_block_examples():
    # AGENT_CLARIFY on its own line inside a backtick fence: not clarification.
    fenced_no_state = (
        "Here's how the marker looks:\n\n"
        "```\n"
        "<!-- AGENT_CLARIFY -->\n"
        "```\n\n"
        "That's all."
    )
    assert not is_clarification_request(fenced_no_state)

    # Fenced example with AGENT_PLAN_STATE after the block: still not clarification.
    fenced_with_plan_state = (
        "Protocol example:\n\n"
        "```\n"
        "<!-- AGENT_CLARIFY -->\n"
        "```\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    assert not is_clarification_request(fenced_with_plan_state)

    # Fenced example where the code block appears AFTER a state marker: not clarification.
    state_then_fenced = (
        "<!-- AGENT_PLAN_STATE: blocking -->\n\n"
        "Appendix:\n\n"
        "```\n"
        "<!-- AGENT_CLARIFY -->\n"
        "```"
    )
    assert not is_clarification_request(state_then_fenced)

    # Tilde fence also excluded.
    tilde_fenced = (
        "~~~\n"
        "<!-- AGENT_CLARIFY -->\n"
        "~~~"
    )
    assert not is_clarification_request(tilde_fenced)

    # Real standalone AGENT_CLARIFY outside a fence: still detected.
    outside_fence = (
        "```\n"
        "some code\n"
        "```\n\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(outside_fence)

    # AGENT_CLARIFY both inside and outside a fence: outside occurrence is active.
    inside_and_outside = (
        "```\n"
        "<!-- AGENT_CLARIFY -->\n"
        "```\n\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(inside_and_outside)


def test_is_clarification_request_requires_clarify_at_end():
    # Non-blank, non-signature content after AGENT_CLARIFY means it's embedded.
    embedded_with_trailing = (
        "<!-- AGENT_CLARIFY -->\n\n"
        "Some trailing prose that isn't a signature.\n"
        "<!-- AGENT_STATE: blocking -->"
    )
    # AGENT_STATE suppresses it via the presence-based check above.
    assert not is_clarification_request(embedded_with_trailing)

    # Standalone AGENT_CLARIFY with only blank lines after it: valid.
    clarify_then_blank = "<!-- AGENT_CLARIFY -->\n\n"
    assert is_clarification_request(clarify_then_blank)

    # Standalone AGENT_CLARIFY with only a signature after it: valid.
    clarify_then_sig = "<!-- AGENT_CLARIFY -->\n-- Anthropic Claude\n"
    assert is_clarification_request(clarify_then_sig)

    # Standalone AGENT_CLARIFY with real prose content after it (no state marker):
    # should NOT be treated as an active clarification.
    clarify_then_prose = (
        "<!-- AGENT_CLARIFY -->\n\n"
        "Continuing thoughts about the plan.\n"
    )
    assert not is_clarification_request(clarify_then_prose)

    # AGENT_CLARIFY in plan body with plan footer after: suppressed by state marker
    # (presence-based check catches it before trailing-content check).
    in_plan_body = (
        "Here are my questions:\n\n"
        "<!-- AGENT_CLARIFY -->\n\n"
        "More explanation here.\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n"
    )
    assert not is_clarification_request(in_plan_body)

    # Multiple AGENT_CLARIFY; last one has only signature trailing: valid.
    multi_clarify = (
        "First question set:\n<!-- AGENT_CLARIFY -->\n\nOther text.\n\n"
        "<!-- AGENT_CLARIFY -->\n-- Anthropic Claude\n"
    )
    assert is_clarification_request(multi_clarify)

    # Multiple AGENT_CLARIFY; last one has prose trailing: not valid.
    multi_clarify_bad = (
        "<!-- AGENT_CLARIFY -->\n\n"
        "<!-- AGENT_CLARIFY -->\n\n"
        "But wait, there's more content.\n"
    )
    assert not is_clarification_request(multi_clarify_bad)


def test_task_loop_creates_pr_then_alternates_until_codex_approval(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Implemented.\n<!-- AGENT_PR: 91 -->\n<!-- AGENT_STATE: blocking -->",
            "Fixed review.\n<!-- AGENT_STATE: blocking -->",
        ],
        codex_outputs=[
            "One nit.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        pr_payload={
            "number": 91,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/91",
        },
    )
    config = make_config(tmp_path)

    assert (
        run_task_loop(
            runner,
            task_text="Add a /healthz endpoint that returns 200 OK.",
            config=config,
        )
        == 0
    )

    command_names = [cmd[:2] for cmd, _cwd in runner.commands]
    assert ["claude", "--print"] in command_names
    assert ["codex", "exec"] in command_names
    assert len(runner.comments) == 4
    assert runner.comments[0].startswith("Implemented.")
    assert runner.comments[-1].startswith("**Review verdict:** Approved\n\nLGTM.")


def test_task_loop_syncs_coder_base_before_first_implementation_attempt(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Implemented.\n<!-- AGENT_PR: 91 -->\n<!-- AGENT_STATE: blocking -->",
        ],
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        pr_payload={
            "number": 91,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/91",
        },
    )
    config = make_config(tmp_path)
    config.agent_memory_dir.mkdir(parents=True)
    (config.agent_memory_dir / "last-analyzed-commit").write_text("base123\n", encoding="utf-8")

    assert run_task_loop(runner, task_text="Add a /healthz endpoint.", config=config) == 0

    commands = runner.commands
    memory_index = command_index(commands, ["git", "diff", "--name-only"])
    fetch_index = command_index(commands, ["git", "fetch", "origin"])
    switch_index = command_index(commands, ["git", "switch", "main"])
    pull_index = command_index(commands, ["git", "pull", "--ff-only", "origin", "main"])
    coder_index = command_index(commands, ["claude", "--print"])

    assert memory_index < fetch_index < switch_index < pull_index < coder_index


def test_task_loop_picks_up_pr_url_when_marker_missing(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Opened https://github.com/OWNER/REPO/pull/77\n"
            "<!-- AGENT_STATE: blocking -->",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path)

    assert (
        run_task_loop(
            runner,
            task_text="Tighten the rate limiter to 5 rps.",
            config=config,
        )
        == 0
    )


def test_task_loop_non_interactive_fails_on_clarification_request(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "I need to know which endpoint.\n<!-- AGENT_CLARIFY -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="non-interactive"):
        run_task_loop(
            runner,
            task_text="Add caching",
            config=config,
        )

    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)
    assert runner.comments == []


def test_task_loop_interactive_supplies_clarification_then_creates_pr(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Which endpoint and how long?\n<!-- AGENT_CLARIFY -->\n-- Anthropic Claude",
            "Implemented.\n<!-- AGENT_PR: 99 -->\n<!-- AGENT_STATE: blocking -->",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={
            "number": 99,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/99",
        },
    )
    config = make_config(tmp_path)
    answers = iter(["recent-debates endpoint, 60s TTL"])

    assert (
        run_task_loop(
            runner,
            task_text="Add caching",
            config=config,
            interactive=True,
            clarification_input=lambda: next(answers),
        )
        == 0
    )

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 2
    assert "recent-debates endpoint, 60s TTL" in claude_calls[1][-1]


def test_task_loop_interactive_aborts_after_max_clarification_rounds(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Q1?\n<!-- AGENT_CLARIFY -->",
            "Q2?\n<!-- AGENT_CLARIFY -->",
        ],
    )
    config = make_config(tmp_path)
    answers = iter(["a1", "a2"])

    with pytest.raises(AgentLoopError, match="after 1 rounds"):
        run_task_loop(
            runner,
            task_text="Refactor everything",
            config=config,
            interactive=True,
            max_clarification_rounds=1,
            clarification_input=lambda: next(answers),
        )


def test_task_loop_rejects_empty_task_text(tmp_path):
    runner = FakeRunner()
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="empty"):
        run_task_loop(runner, task_text="   ", config=config)

    assert runner.commands == []


def test_task_loop_requires_pr_or_clarification_marker(tmp_path):
    runner = FakeRunner(
        claude_outputs=["I just wrote some prose without any markers."],
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="PR marker"):
        run_task_loop(runner, task_text="Do something", config=config)


def test_pr_loop_rejects_non_open_pr_before_running_codex(tmp_path):
    runner = FakeRunner(pr_payload={
        "number": 62,
        "state": "MERGED",
        "url": "https://github.com/OWNER/REPO/pull/62",
    })
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="provide an open PR"):
        run_pr_loop(runner, pr_number=62, config=config)


def test_pr_loop_refreshes_pr_head_without_just_in_time_base_sync(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    assert ["git", "fetch", "origin"] in commands
    assert ["git", "fetch", "origin", "+pull/77/head:refs/remotes/origin/pr/77"] in commands
    assert ["git", "switch", "main"] not in commands
    assert ["git", "pull", "--ff-only", "origin", "main"] not in commands


# ---------------------------------------------------------------------------
# Reverse flow: Codex creates PR, Claude reviews
# ---------------------------------------------------------------------------


def test_codex_issue_loop_creates_pr_then_claude_approves(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Fixed issue.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Looks good.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    command_names = [cmd[:2] for cmd, _cwd in runner.commands]
    assert ["codex", "exec"] in command_names
    assert ["claude", "--print"] in command_names
    assert len(runner.comments) == 2
    assert runner.comments[0].startswith("Fixed issue.")
    assert runner.comments[1].startswith("**Review verdict:** Approved\n\nLooks good.")


def test_codex_issue_loop_alternates_until_claude_approval(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Implemented fix.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Addressed Claude's review.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Missing test.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    assert len(runner.comments) == 4
    assert runner.comments[-1].startswith("**Review verdict:** Approved\n\nLGTM.")


def test_codex_issue_loop_requires_codex_to_report_pr_number(tmp_path):
    runner = FakeRunner(
        codex_outputs=["Did some work.\n<!-- AGENT_STATE: blocking -->"],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError, match="PR marker"):
        run_issue_loop(runner, issue_number=56, config=config)

    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def test_issue_loop_rejects_outside_workdir_tests_before_posting_pr_comment(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Fixed issue.\n"
            "Tests: cd ~/llm-dialectic && python -m pytest\n"
            "<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n"
            "-- OpenAI Codex",
        ],
        claude_outputs=[
            "Looks good.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError, match="outside the assigned checkout"):
        run_issue_loop(runner, issue_number=56, config=config)

    assert runner.comments == []
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def test_issue_loop_rejects_reported_pr_when_assigned_head_unchanged(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Fixed issue.\n"
            "Tests: python -m pytest passed.\n"
            "<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n"
            "-- OpenAI Codex",
        ],
        claude_outputs=[
            "Looks good.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        advance_git_head_on_pr=False,
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError, match="HEAD did not advance"):
        run_issue_loop(runner, issue_number=56, config=config)

    assert runner.comments == []
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def test_codex_task_loop_creates_pr_then_claude_approves(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Implemented task.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Ship it.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    assert run_task_loop(runner, task_text="Add /healthz endpoint.", config=config) == 0

    assert len(runner.comments) == 2
    assert runner.comments[0].startswith("Implemented task.")
    assert runner.comments[1].startswith("**Review verdict:** Approved\n\nShip it.")


def test_pr_loop_rejects_structured_followup_outside_workdir_tests_before_posting(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Needs a test.",
                blocking_items=["Add a regression test."],
                reviewer="Anthropic Claude",
            ),
            "Looks good.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_coder_followup(
                summary="Added the test.",
                addressed_items=["item-1"],
                tests_run=["cd ~/llm-dialectic && python -m pytest"],
                reviewer="OpenAI Codex",
            ),
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError, match="outside the assigned checkout"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert len(runner.comments) == 1
    assert runner.comments[0].startswith("**Review verdict:** Blocking")
    assert not any("Added the test." in comment for comment in runner.comments)


def test_codex_task_loop_picks_up_pr_url_when_marker_missing(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Opened https://github.com/OWNER/REPO/pull/77\n"
            "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    assert run_task_loop(runner, task_text="Tighten rate limiter.", config=config) == 0


def test_gemini_issue_loop_creates_pr_then_codex_approves(tmp_path):
    runner = FakeRunner(
        gemini_outputs=[
            "Fixed issue.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini",
        ],
        codex_outputs=[
            "Looks good.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, coder="gemini", reviewer="codex")

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    agent_commands = [cmd[:2] for cmd, _cwd in runner.commands if cmd[:1] in (["gemini"], ["codex"])]
    assert agent_commands == [["gemini", "--prompt"], ["codex", "exec"]]
    assert len(runner.comments) == 2
    assert runner.comments[0].startswith("Fixed issue.")
    assert runner.comments[1].startswith("**Review verdict:** Approved\n\nLooks good.")


def test_gemini_issue_loop_resumes_session_for_followup(tmp_path):
    runner = FakeRunner(
        gemini_outputs=[
            json.dumps({
                "response": "Fixed issue.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini",
                "session_id": "gemini-session-1",
            }),
            # Plain-text output intentionally clears the tracked session; a third
            # Gemini turn would start without --resume.
            "Addressed review.\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini",
        ],
        codex_outputs=[
            "Needs a regression test.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Looks good."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(
        tmp_path,
        coder="gemini",
        reviewer="codex",
        gemini_args=("--output-format", "json"),
    )

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    gemini_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["gemini"]]
    assert len(gemini_calls) == 2
    assert "--resume" not in gemini_calls[0]
    assert gemini_calls[1][-2:] == ["--resume", "gemini-session-1"]


def test_gemini_review_loop_uses_prompt_and_extra_args(tmp_path):
    runner = FakeRunner(
        gemini_outputs=[
            json.dumps({"response": "LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"}),
        ],
    )
    config = make_config(
        tmp_path,
        reviewer="gemini",
        gemini_args=("--output-format", "json", "--model", "gemini-2.5-flash"),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    gemini_call = next(cmd for cmd, _cwd in runner.commands if cmd[:1] == ["gemini"])
    assert gemini_call[:2] == ["gemini", "--prompt"]
    assert PUBLIC_RESPONSE_MARKER in gemini_call[2]
    assert "Only content after that line will be posted to GitHub" in gemini_call[2]
    assert "--output-format" in gemini_call
    assert "--model" in gemini_call
    assert runner.comments == ["**Review verdict:** Approved\n\nLGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"]


def test_gemini_review_loop_prefers_public_response_file_over_stdout(tmp_path):
    runner = FakeRunner(
        gemini_outputs=[
            "Warning: True color (24-bit) support not detected.\n"
            "YOLO mode is enabled. All tool calls will be automatically approved.\n"
            "I will fetch the PR and inspect the diff.\n"
            "Error executing tool run_shell_command: confirmation required.\n"
            "This stdout chatter should not be posted.\n",
        ],
        public_response_outputs=[
            "LGTM from response file.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini",
        ],
    )
    config = make_config(tmp_path, reviewer="gemini")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    gemini_call = next(cmd for cmd, _cwd in runner.commands if cmd[:1] == ["gemini"])
    assert "PUBLIC RESPONSE FILE:" in gemini_call[2]
    assert str(config.gemini_dir / ".git" / "agent-loop" / "responses" / "gemini") in gemini_call[2]
    assert runner.comments == ["**Review verdict:** Approved\n\nLGTM from response file.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"]


def test_claude_review_loop_prefers_public_response_file_over_stdout(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            json.dumps(
                {
                    "result": (
                        "I will inspect the PR diff.\n"
                        "Tool output chatter should not be posted.\n"
                    ),
                    "session_id": "claude-session-1",
                }
            ),
        ],
        public_response_outputs=[
            "LGTM from response file.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, reviewer="claude")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    claude_call = next(cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "PUBLIC RESPONSE FILE:" in claude_call[-1]
    assert "/coding-review-agent-loop/responses/OWNER-REPO/claude/" in claude_call[-1]
    assert runner.comments == ["**Review verdict:** Approved\n\nLGTM from response file.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude"]


def test_public_response_file_instruction_mentions_plan_revision_human_ack_exception(tmp_path):
    prompt = with_public_response_file_instruction(
        "Review the PR.",
        tmp_path / "response.md",
    )

    assert "For structured plan revisions only" in prompt
    assert "<!-- HUMAN_REQUIREMENTS_ADDRESSED -->" in prompt
    assert "`### Human requirements` section after the JSON object" in prompt
    assert "before the\n`AGENT_PLAN_STATE` footer" in prompt


def test_codex_task_loop_rejects_empty_task_text(tmp_path):
    runner = FakeRunner()
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError, match="empty"):
        run_task_loop(runner, task_text="   ", config=config)

    assert runner.commands == []


def test_claude_review_loop_runs_tests_and_merge_only_after_approval(tmp_path):
    runner = FakeRunner(
        claude_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude"],
    )
    config = make_config(
        tmp_path,
        coder="codex",
        reviewer="claude",
        auto_merge=True,
        test_command=("pytest", "tests/test_agent_loop.py"),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    assert ["pytest", "tests/test_agent_loop.py"] in commands
    assert ["gh", "pr", "merge", "77", "--repo", "OWNER/REPO", "--merge"] in commands


def test_claude_review_loop_does_not_run_codex_after_final_blocking_round(tmp_path):
    runner = FakeRunner(
        claude_outputs=["Still blocked.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude", max_rounds=1)

    with pytest.raises(AgentLoopError, match="still reported blocking"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


def test_claude_review_loop_rejects_non_open_pr(tmp_path):
    runner = FakeRunner(pr_payload={
        "number": 62,
        "state": "CLOSED",
        "url": "https://github.com/OWNER/REPO/pull/62",
    })
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError, match="provide an open PR"):
        run_pr_loop(runner, pr_number=62, config=config)

    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)

    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


# ---------------------------------------------------------------------------
# Repair pass tests
# ---------------------------------------------------------------------------

from coding_review_agent_loop.repair import (
    _REPAIR_PROMPT,
    attempt_envelope_normalization,
    attempt_repair,
)


def test_envelope_normalization_duplicate_pr_state_footer_preserves_dispositions():
    raw = (
        structured_pr_review(
            state="approved",
            reviewer="Google Gemini",
            prior_item_dispositions=[
                {"item_id": "item-1", "disposition": "resolved"},
                {"item_id": "item-2", "disposition": "future"},
                {"item_id": "item-3", "disposition": "resolved"},
            ],
        )
        + "\n\n<!-- AGENT_STATE: approved -->"
    )

    normalized = attempt_envelope_normalization(raw, expected_kind="pr_review")

    assert normalized is not None
    parsed = parse_structured_pr_review(normalized, reviewer="Google Gemini")
    assert parsed is not None
    assert {
        disposition.item_id: disposition.disposition
        for disposition in parsed.dispositions
    } == {
        "item-1": "resolved",
        "item-2": "future",
        "item-3": "resolved",
    }
    assert normalized.count("<!-- AGENT_STATE: approved -->") == 1


def test_envelope_normalization_trailing_prose_after_signature():
    raw = (
        structured_pr_review(state="approved", reviewer="Google Gemini")
        + "\n\nExtra prose after the signature."
    )

    normalized = attempt_envelope_normalization(raw, expected_kind="pr_review")

    assert normalized is not None
    assert "Extra prose" not in normalized
    assert parse_structured_pr_review(normalized, reviewer="Google Gemini") is not None


def test_envelope_normalization_duplicate_plan_state_footer():
    raw = (
        structured_plan_review(
            state="approved",
            reviewer="Google Gemini",
            prior_plan_item_dispositions=[
                {"item_id": "item-1", "disposition": "resolved"},
            ],
        )
        + "\n\n<!-- AGENT_PLAN_STATE: approved -->"
    )

    normalized = attempt_envelope_normalization(raw, expected_kind="plan_review")

    assert normalized is not None
    parsed = parse_structured_plan_review(normalized, reviewer="Google Gemini")
    assert parsed is not None
    assert [(item.item_id, item.disposition) for item in parsed.dispositions] == [
        ("item-1", "resolved")
    ]
    assert normalized.count("<!-- AGENT_PLAN_STATE: approved -->") == 1


def test_envelope_normalization_preserves_hr_resolved_before_footer_for_reviews():
    for expected_kind, raw, parser in (
        (
            "pr_review",
            structured_pr_review(
                state="approved",
                reviewer="Google Gemini",
                human_requirements_resolved=True,
            ),
            parse_structured_pr_review,
        ),
        (
            "plan_review",
            structured_plan_review(
                state="approved",
                reviewer="Google Gemini",
                human_requirements_resolved=True,
            ),
            parse_structured_plan_review,
        ),
    ):
        normalized = attempt_envelope_normalization(
            raw + "\n\nTrailing prose.",
            expected_kind=expected_kind,
        )

        assert normalized is not None
        assert "<!-- HUMAN_REQUIREMENTS_RESOLVED -->" in normalized
        assert parser(normalized, reviewer="Google Gemini") is not None


def test_envelope_normalization_preserves_hr_resolved_after_footer_for_pr_review():
    raw = (
        structured_pr_review(state="approved", reviewer="Google Gemini").replace(
            "\n-- Google Gemini",
            "\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n-- Google Gemini",
        )
        + "\n\n<!-- AGENT_STATE: approved -->"
    )

    normalized = attempt_envelope_normalization(raw, expected_kind="pr_review")

    assert normalized is not None
    assert "<!-- HUMAN_REQUIREMENTS_RESOLVED -->" in normalized
    assert parse_structured_pr_review(normalized, reviewer="Google Gemini") is not None


def test_envelope_normalization_plan_review_drops_after_footer_hr_marker():
    raw = (
        structured_plan_review(
            state="approved",
            reviewer="Google Gemini",
            prior_plan_item_dispositions=[
                {"item_id": "item-1", "disposition": "resolved"},
                {"item_id": "item-2", "disposition": "future"},
            ],
        ).replace(
            "\n-- Google Gemini",
            "\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n-- Google Gemini",
        )
        + "\n\n<!-- AGENT_PLAN_STATE: approved -->\nTrailing prose."
    )

    normalized = attempt_envelope_normalization(raw, expected_kind="plan_review")

    assert normalized is not None
    assert "<!-- HUMAN_REQUIREMENTS_RESOLVED -->" not in normalized
    parsed = parse_structured_plan_review(normalized, reviewer="Google Gemini")
    assert parsed is not None
    assert {
        disposition.item_id: disposition.disposition
        for disposition in parsed.dispositions
    } == {
        "item-1": "resolved",
        "item-2": "future",
    }


def test_envelope_normalization_returns_none_when_no_footer():
    raw = json.dumps(
        {
            "schema_version": 1,
            "kind": "pr_review",
            "state": "approved",
            "summary": "Review complete.",
            "prior_item_dispositions": [],
        }
    )

    assert attempt_envelope_normalization(raw, expected_kind="pr_review") is None


def test_envelope_normalization_returns_none_when_no_signature():
    raw = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "Review complete.",
                "prior_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_STATE: approved -->"
    )

    assert attempt_envelope_normalization(raw, expected_kind="pr_review") is None


def test_envelope_normalization_returns_none_when_json_invalid():
    raw = '{"schema_version": 1,\n<!-- AGENT_STATE: approved -->\n-- Google Gemini'

    assert attempt_envelope_normalization(raw, expected_kind="pr_review") is None


def test_envelope_normalization_semantic_defect_still_fails_validate():
    raw = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "Review complete.",
            }
        )
        + "\n<!-- AGENT_STATE: approved -->\n-- Google Gemini\n\nTrailing prose."
    )

    normalized = attempt_envelope_normalization(raw, expected_kind="pr_review")

    assert normalized is not None
    with pytest.raises(AgentLoopError):
        parse_structured_pr_review(normalized, reviewer="Google Gemini")


def test_attempt_repair_returns_none_when_subprocess_fails(monkeypatch):
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=mock_result):
        result = attempt_repair("some malformed review text", "gemini")
    assert result is None


def test_attempt_repair_returns_none_when_subprocess_raises(monkeypatch):
    with patch("coding_review_agent_loop.repair.subprocess.run", side_effect=FileNotFoundError("gemini not found")):
        result = attempt_repair("some malformed review text", "gemini")
    assert result is None


def test_attempt_repair_calls_cli_and_returns_text():
    repaired = (
        '{"schema_version":1,"kind":"pr_review","state":"approved","summary":"OK",'
        '"blocking_items":[],"same_pr_followups":[],"future_followups":[],'
        '"prior_item_dispositions":[]}\n<!-- AGENT_STATE: approved -->\n-- Gemini'
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = repaired

    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=mock_result) as mock_run:
        result = attempt_repair("malformed review", "gemini")

    assert result == repaired
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    cmd = call_args.args[0]
    assert cmd[0] == "gemini"
    assert "--model" in cmd
    assert "gemini-3.1-flash-lite" in cmd
    assert "--prompt" in cmd
    prompt_idx = cmd.index("--prompt")
    assert "malformed review" in cmd[prompt_idx + 1]


def test_attempt_repair_includes_expected_kind_instruction():
    repaired = (
        '{"schema_version":1,"kind":"plan_revision","state":"blocking","summary":"Revised.",'
        '"prior_plan_item_dispositions":[],"plan_steps":["Add tests."]}'
        "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Gemini"
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = repaired

    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=mock_result) as mock_run:
        result = attempt_repair(
            "malformed response mentioning human requirements and addressed_items",
            "gemini",
            expected_kind="plan_revision",
        )

    assert result == repaired
    cmd = mock_run.call_args.args[0]
    prompt = cmd[cmd.index("--prompt") + 1]
    assert "You MUST repair this response as `plan_revision`" in prompt
    assert "Output no other `kind` value" in prompt


def test_attempt_repair_format_d_marks_human_requirements_optional():
    assert "omit the `<!-- HUMAN_REQUIREMENTS_ADDRESSED -->` marker" in _REPAIR_PROMPT
    assert "the `### Human requirements` section from Format D" in _REPAIR_PROMPT


def test_attempt_repair_includes_prior_item_disposition_repair_context():
    repaired = structured_plan_revision(prior_plan_item_dispositions=[])
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = repaired

    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=mock_result) as mock_run:
        result = attempt_repair(
            "malformed plan revision",
            "gemini",
            expected_kind="plan_revision",
            allowed_prior_item_ids=["item-12"],
            unknown_prior_item_ids=["item-15", "item-18"],
            same_round_context="Same-round findings are informational only.",
        )

    assert result == repaired
    cmd = mock_run.call_args.args[0]
    prompt = cmd[cmd.index("--prompt") + 1]
    assert "Prior item disposition repair" in prompt
    assert "Allowed carried prior item IDs: item-12" in prompt
    assert "Unknown prior item disposition IDs to remove: item-15, item-18" in prompt
    assert "Same-round findings are informational only" in prompt


def test_attempt_repair_prior_item_disposition_context_is_not_duplicated():
    repaired = structured_plan_revision(prior_plan_item_dispositions=[])
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = repaired

    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=mock_result) as mock_run:
        attempt_repair(
            "malformed plan revision",
            "gemini",
            expected_kind="plan_revision",
            same_round_context="item-1 matches a same-round finding, not a carried prior item.",
        )

    cmd = mock_run.call_args.args[0]
    prompt = cmd[cmd.index("--prompt") + 1]
    assert prompt.count("item-1 matches a same-round finding") == 1
    assert "Context: item-1 matches a same-round finding" not in prompt


def test_attempt_repair_includes_coder_followup_required_item_ids():
    repaired = structured_coder_followup(
        state="approved",
        addressed_items=["item-8"],
        remaining_items=[],
        reviewer="Gemini",
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = repaired

    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=mock_result) as mock_run:
        result = attempt_repair(
            "### Human requirements\nAcknowledged.\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->",
            "gemini",
            expected_kind="coder_followup",
            unresolved_item_ids=["item-8"],
        )

    assert result == repaired
    cmd = mock_run.call_args.args[0]
    prompt = cmd[cmd.index("--prompt") + 1]
    assert "Required coder follow-up item IDs" in prompt
    assert "`item-8`" in prompt
    assert "exactly one of `addressed_items` or `remaining_items`" in prompt
    assert "HUMAN_REQUIREMENTS_ADDRESSED" in prompt
    assert "do not classify regular reviewer or orchestrator-injected item-N records" in prompt


def test_attempt_repair_includes_empty_surfaced_requirement_guidance():
    repaired = structured_coder_followup(
        state="blocking",
        addressed_items=["item-1"],
        remaining_items=[],
        human_requirement_ids=[],
        reviewer="Gemini",
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = repaired

    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=mock_result) as mock_run:
        result = attempt_repair(
            '"human_requirements":{"addressed_ids":["Issue #221 acceptance criteria"],'
            '"checked_discussion_directly":false}',
            "gemini",
            expected_kind="coder_followup",
            unresolved_item_ids=["item-1"],
            surfaced_requirement_ids=[],
        )

    assert result == repaired
    cmd = mock_run.call_args.args[0]
    prompt = cmd[cmd.index("--prompt") + 1]
    assert "Surfaced signed human requirement labels for coder follow-up" in prompt
    assert "- (none)" in prompt
    assert "set `human_requirements.addressed_ids` to `[]`" in prompt
    assert "Issue #221 acceptance criteria" in prompt
    assert '"addressed_ids": []' in prompt


def test_attempt_repair_includes_surfaced_requirement_labels_for_mixed_repairs():
    repaired = structured_coder_followup(
        state="blocking",
        addressed_items=["item-1"],
        remaining_items=[],
        human_requirement_ids=["Requirement 1"],
        reviewer="Gemini",
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = repaired

    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=mock_result) as mock_run:
        result = attempt_repair(
            '"addressed_ids":["Requirement 1","Issue #221 acceptance criteria"]',
            "gemini",
            expected_kind="coder_followup",
            unresolved_item_ids=["item-1"],
            surfaced_requirement_ids=["Requirement 1"],
        )

    assert result == repaired
    cmd = mock_run.call_args.args[0]
    prompt = cmd[cmd.index("--prompt") + 1]
    assert "`Requirement 1`" in prompt
    assert "keep [\"Requirement 1\"] and drop \"Issue #221 acceptance criteria\"" in prompt


def test_attempt_repair_rejects_unresolved_item_ids_for_non_coder_kind():
    with pytest.raises(ValueError, match="unresolved_item_ids"):
        attempt_repair(
            "malformed plan review",
            "gemini",
            expected_kind="plan_review",
            unresolved_item_ids=["item-1"],
        )


def test_attempt_repair_rejects_surfaced_requirement_ids_for_non_coder_kind():
    with pytest.raises(ValueError, match="surfaced_requirement_ids"):
        attempt_repair(
            "malformed plan review",
            "gemini",
            expected_kind="plan_review",
            surfaced_requirement_ids=["Requirement 1"],
        )


def test_attempt_repair_handles_json_wrapped_cli_output():
    repaired_text = (
        '{"schema_version":1,"kind":"pr_review","state":"approved","summary":"OK",'
        '"blocking_items":[],"same_pr_followups":[],"future_followups":[],'
        '"prior_item_dispositions":[]}\n<!-- AGENT_STATE: approved -->\n-- Gemini'
    )
    json_wrapped = json.dumps({"response": repaired_text, "session_id": "s1"})
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json_wrapped

    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=mock_result):
        result = attempt_repair("malformed review", "gemini")

    assert result == repaired_text


def test_repair_prompt_contains_raw_response_placeholder():
    assert "{raw_response}" in _REPAIR_PROMPT


def test_repair_prompt_substitution_leaves_json_examples_intact():
    raw = "some {curly} braces {in} the review text"
    substituted = _REPAIR_PROMPT.replace("{raw_response}", raw, 1)
    assert raw in substituted
    assert "{raw_response}" not in substituted
    assert "schema_version" in substituted


def test_run_pr_loop_uses_repair_pass_on_format_failure(tmp_path):
    """Repair pass is invoked when schema validation fails; repaired output is used."""
    malformed_review = (
        "Looks good overall.\n\n"
        "AGENT_STATE: approved\n"
        "-- OpenAI Codex"
    )
    repaired_review = (
        '{"schema_version":1,"kind":"pr_review","state":"approved","summary":"Looks good overall.",'
        '"blocking_items":[],"same_pr_followups":[],"future_followups":[],'
        '"prior_item_dispositions":[]}'
        "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    )
    runner = FakeRunner(
        codex_outputs=[malformed_review],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", agent_max_retries=0)

    captured_repairs = []

    def fake_attempt_repair(raw: str, gemini_cmd: str, *, expected_kind: str | None = None, unresolved_item_ids=None, surfaced_requirement_ids=None, allowed_prior_item_ids=None, unknown_prior_item_ids=None, same_round_context=None) -> str | None:
        captured_repairs.append(raw)
        assert expected_kind == "pr_review"
        return repaired_review

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_attempt_repair):
        run_pr_loop(runner, pr_number=77, config=config)

    assert len(captured_repairs) == 1
    assert "AGENT_STATE: approved" in captured_repairs[0]


def test_run_validated_agent_envelope_normalization_recovers_duplicate_footer(tmp_path):
    malformed_review = (
        structured_pr_review(
            state="approved",
            reviewer="Google Gemini",
            prior_item_dispositions=[
                {"item_id": "item-1", "disposition": "resolved"},
                {"item_id": "item-2", "disposition": "future"},
                {"item_id": "item-3", "disposition": "resolved"},
            ],
        )
        + "\n\n<!-- AGENT_STATE: approved -->"
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the PR.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: parse_structured_pr_review(
                text,
                reviewer="Google Gemini",
            ).state,
            use_repair=True,
            repair_expected_kind="pr_review",
        )

    repair_mock.assert_not_called()
    parsed = parse_structured_pr_review(response.text, reviewer="Google Gemini")
    assert parsed is not None
    assert {
        disposition.item_id: disposition.disposition
        for disposition in parsed.dispositions
    } == {
        "item-1": "resolved",
        "item-2": "future",
        "item-3": "resolved",
    }


def test_run_validated_agent_envelope_normalization_semantic_defect_uses_repair(tmp_path):
    malformed_review = (
        structured_pr_review(
            state="approved",
            reviewer="Google Gemini",
            blocking_items=["This is semantically inconsistent."],
        )
        + "\n\n<!-- AGENT_STATE: approved -->"
    )
    repaired_review = structured_pr_review(state="approved", reviewer="Google Gemini")
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch(
        "coding_review_agent_loop.orchestrator.attempt_repair",
        return_value=repaired_review,
    ) as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the PR.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: parse_structured_pr_review(
                text,
                reviewer="Google Gemini",
            ).state,
            use_repair=True,
            repair_expected_kind="pr_review",
        )

    assert response.text == repaired_review
    repair_mock.assert_called_once_with(
        malformed_review,
        config.gemini_cmd,
        expected_kind="pr_review",
    )


def test_run_pr_loop_repairs_format_failure_with_5xx_source_line_reference(tmp_path):
    """A 500-series source line reference must not make deterministic format errors transient."""
    malformed_review = (
        "Looks good overall.\n\n"
        "Note: orchestrator.py:577-581 currently falls back to parse_plan_state(text).\n"
        "AGENT_STATE: approved\n"
        "-- OpenAI Codex"
    )
    repaired_review = (
        '{"schema_version":1,"kind":"pr_review","state":"approved","summary":"Looks good overall.",'
        '"blocking_items":[],"same_pr_followups":[],"future_followups":[],'
        '"prior_item_dispositions":[]}'
        "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    )
    runner = FakeRunner(
        codex_outputs=[malformed_review],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", agent_max_retries=0)

    captured_repairs = []

    def fake_attempt_repair(raw: str, gemini_cmd: str, *, expected_kind: str | None = None, unresolved_item_ids=None, surfaced_requirement_ids=None, allowed_prior_item_ids=None, unknown_prior_item_ids=None, same_round_context=None) -> str | None:
        captured_repairs.append(raw)
        assert expected_kind == "pr_review"
        return repaired_review

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_attempt_repair):
        run_pr_loop(runner, pr_number=77, config=config)

    assert len(captured_repairs) == 1
    assert "orchestrator.py:577-581" in captured_repairs[0]


def test_run_pr_loop_falls_back_to_error_when_repair_also_fails(tmp_path):
    """When repair also produces invalid output, the original error is raised."""
    malformed_review = (
        "Something went wrong with the format.\n"
        "AGENT_STATE: approved\n"
        "-- OpenAI Codex"
    )
    runner = FakeRunner(
        codex_outputs=[malformed_review],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", agent_max_retries=0)

    def fake_attempt_repair_fails(raw: str, gemini_cmd: str, *, expected_kind: str | None = None, unresolved_item_ids=None, surfaced_requirement_ids=None, allowed_prior_item_ids=None, unknown_prior_item_ids=None, same_round_context=None) -> str | None:
        assert expected_kind == "pr_review"
        return "still broken output without valid schema"

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_attempt_repair_fails):
        with pytest.raises(AgentLoopError, match="Codex"):
            run_pr_loop(runner, pr_number=77, config=config)


def test_run_pr_loop_skips_repair_when_repair_returns_none(tmp_path):
    """When attempt_repair returns None (e.g. no API key), normal error is raised."""
    malformed_review = (
        "Something went wrong.\n"
        "AGENT_STATE: approved\n"
        "-- OpenAI Codex"
    )
    runner = FakeRunner(
        codex_outputs=[malformed_review],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=None):
        with pytest.raises(AgentLoopError, match="Codex"):
            run_pr_loop(runner, pr_number=77, config=config)


def test_run_pr_loop_uses_repair_pass_on_coder_followup_format_failure(tmp_path):
    """Repair pass is invoked when coder followup schema validation fails; repaired output is used."""
    malformed_coder_followup = (
        '{"schema_version":1,"kind":"pr_review","state":"blocking","summary":"Fixed the bug.",'
        '"blocking_items":[],"same_pr_followups":[],"future_followups":[],'
        '"prior_item_dispositions":[]}'
        "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    repaired_followup = (
        '{"schema_version":1,"kind":"coder_followup","state":"blocking","summary":"Fixed the bug.",'
        '"addressed_items":["item-1"],"remaining_items":[],'
        '"human_requirements":{"addressed_ids":[],"checked_discussion_directly":false}}'
        "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = FakeRunner(
        claude_outputs=[malformed_coder_followup],
        codex_outputs=[
            "Need a fix."
            + blocking_issues("Fix the bug.")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=2, agent_max_retries=0)

    captured_repairs = []
    captured_unresolved_item_ids = []

    def fake_attempt_repair(raw: str, gemini_cmd: str, *, expected_kind: str | None = None, unresolved_item_ids=None, surfaced_requirement_ids=None, allowed_prior_item_ids=None, unknown_prior_item_ids=None, same_round_context=None) -> str | None:
        captured_repairs.append(raw)
        captured_unresolved_item_ids.append(tuple(unresolved_item_ids or ()))
        assert expected_kind == "coder_followup"
        return repaired_followup

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_attempt_repair):
        result = run_pr_loop(runner, pr_number=77, config=config)

    assert result == 0
    assert len(captured_repairs) == 1
    assert "pr_review" in captured_repairs[0]
    assert captured_unresolved_item_ids == [("item-1",)]


def test_run_pr_loop_falls_back_to_error_when_coder_followup_repair_also_fails(tmp_path):
    """When repair also produces invalid output for coder followup, the original error is raised."""
    malformed_coder_followup = (
        '{"schema_version":1,"kind":"pr_review","state":"blocking","summary":"Fixed the bug.",'
        '"blocking_items":[],"same_pr_followups":[],"future_followups":[],'
        '"prior_item_dispositions":[]}'
        "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = FakeRunner(
        claude_outputs=[malformed_coder_followup],
        codex_outputs=[
            "Need a fix."
            + blocking_issues("Fix the bug.")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=2, agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value="still broken output"):
        with pytest.raises(AgentLoopError, match="Claude"):
            run_pr_loop(runner, pr_number=77, config=config)


def test_run_pr_loop_skips_repair_when_coder_followup_repair_returns_none(tmp_path):
    """When attempt_repair returns None for coder followup, normal error is raised."""
    malformed_coder_followup = (
        '{"schema_version":1,"kind":"pr_review","state":"blocking","summary":"Fixed the bug.",'
        '"blocking_items":[],"same_pr_followups":[],"future_followups":[],'
        '"prior_item_dispositions":[]}'
        "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = FakeRunner(
        claude_outputs=[malformed_coder_followup],
        codex_outputs=[
            "Need a fix."
            + blocking_issues("Fix the bug.")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=2, agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=None):
        with pytest.raises(AgentLoopError, match="Claude"):
            run_pr_loop(runner, pr_number=77, config=config)


def test_repair_prompt_contains_coder_followup_format():
    """Repair prompt must include the coder_followup format so the model knows about it."""
    assert "coder_followup" in _REPAIR_PROMPT
    assert "addressed_items" in _REPAIR_PROMPT
    assert "remaining_items" in _REPAIR_PROMPT


def test_repair_prompt_distinguishes_item_ids_from_requirement_labels():
    """Repair prompt must warn that addressed_items uses item IDs, not requirement labels."""
    assert "Requirement 1" in _REPAIR_PROMPT
    assert "addressed_ids" in _REPAIR_PROMPT
    # The prompt must explicitly state item IDs cannot contain spaces
    assert "spaces" in _REPAIR_PROMPT or "DO NOT CONFUSE" in _REPAIR_PROMPT or "NEVER put" in _REPAIR_PROMPT


def test_repair_prompt_includes_plan_review_dedupe_guidance():
    assert "Same-plan follow-ups and Future follow-ups are mutually exclusive" in _REPAIR_PROMPT
    assert "keep blocking_plan_issues and drop the duplicate same_plan_followups entry" in _REPAIR_PROMPT
    assert (
        "keep same_plan_followups/current-plan work and drop the duplicate future_followups entry"
        in _REPAIR_PROMPT
    )
    assert "keep blocking_plan_issues and drop the duplicate future_followups entry" in _REPAIR_PROMPT


def test_repair_prompt_includes_pr_review_dedupe_guidance():
    assert "## DEDUPE RULES (Format A):" in _REPAIR_PROMPT
    assert "Same-PR follow-ups and Future follow-ups are mutually exclusive" in _REPAIR_PROMPT
    assert "keep blocking_items and drop the duplicate same_pr_followups entry" in _REPAIR_PROMPT
    assert (
        "keep same_pr_followups/current-PR work and drop the duplicate future_followups entry"
        in _REPAIR_PROMPT
    )
    assert "keep blocking_items and drop the duplicate future_followups entry" in _REPAIR_PROMPT


def test_repair_prompt_includes_skip_trust_in_cli_invocation():
    """The CLI invocation must include --skip-trust so repair works outside trusted dirs."""
    repaired = (
        '{"schema_version":1,"kind":"pr_review","state":"approved","summary":"OK",'
        '"blocking_items":[],"same_pr_followups":[],"future_followups":[],'
        '"prior_item_dispositions":[]}\n<!-- AGENT_STATE: approved -->\n-- Gemini'
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = repaired

    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=mock_result) as mock_run:
        attempt_repair("malformed review", "gemini")

    cmd = mock_run.call_args.args[0]
    assert "--skip-trust" in cmd


def test_repair_prompt_coder_followup_fenced_json_example():
    """Repair prompt must include a worked example showing fenced JSON being stripped."""
    assert "```json" in _REPAIR_PROMPT
    assert "HUMAN_REQUIREMENTS_ADDRESSED" in _REPAIR_PROMPT
    # The prompt explains the marker is not needed in structured path
    assert "NOT needed" in _REPAIR_PROMPT or "not needed" in _REPAIR_PROMPT.lower()


def test_repair_prompt_plan_revision_preserves_human_requirements_acknowledgement():
    assert "WORKED EXAMPLE 4" in _REPAIR_PROMPT
    assert "do not output coder_followup" in _REPAIR_PROMPT
    assert "preserve both after the JSON and before <!-- AGENT_PLAN_STATE: blocking -->" in _REPAIR_PROMPT


def test_repair_prompt_does_not_suggest_ack_pseudo_item_in_addressed_items():
    """The ack pseudo-item must never be suggested as a value for addressed_items.

    The orchestrator's _validate_structured_coder_followup_items explicitly excludes
    HUMAN_REQUIREMENTS_ACK_ITEM_ID from expected_ids, so any response that puts
    'item-human-requirements-acknowledgement' in addressed_items will be rejected
    as an unknown item ID.
    """
    from coding_review_agent_loop.orchestrator import HUMAN_REQUIREMENTS_ACK_ITEM_ID

    # The ack pseudo-item must not appear in the repair prompt at all, because
    # any mention of it in an addressed_items context will teach Gemini to produce
    # responses that the validator rejects.
    assert HUMAN_REQUIREMENTS_ACK_ITEM_ID not in _REPAIR_PROMPT


# ---------------------------------------------------------------------------
# New tests for issue #246: repair approved reviews with active prior dispositions
# ---------------------------------------------------------------------------

from coding_review_agent_loop.repair import (
    _reviewer_human_requirements_instruction,
)
from coding_review_agent_loop.orchestrator import _surfaced_reviewer_requirement_ids


# --- repair.py prompt content tests ---

def test_repair_prompt_blocking_state_rules_require_explicit_prior_dispositions():
    """STATE RULES for BLOCKING must prohibit omitting allowed prior items."""
    assert "ALL prior items in the allowed list must appear in prior_item_dispositions" in _REPAIR_PROMPT
    assert "No item may be omitted" in _REPAIR_PROMPT or "no item may be omitted" in _REPAIR_PROMPT.lower()
    assert '"future" is forbidden in blocking reviews' in _REPAIR_PROMPT or "future\" is forbidden in blocking" in _REPAIR_PROMPT


def test_repair_example_2_no_longer_says_omit():
    """Worked Example 2 must not instruct omission of formerly-future prior items."""
    assert "OMIT item-1 from prior_item_dispositions entirely" not in _REPAIR_PROMPT
    assert "WORKED EXAMPLE 2" in _REPAIR_PROMPT
    assert "must appear" in _REPAIR_PROMPT


def test_repair_prompt_includes_approved_plus_active_disposition_rules():
    """STATE RULES must cover approved + active same-pr/same-plan/blocking dispositions."""
    assert "APPROVED + active same-pr/same-plan/blocking prior dispositions" in _REPAIR_PROMPT
    assert 'change disposition to "resolved"' in _REPAIR_PROMPT


def test_repair_prompt_includes_approved_future_followups_current_plan_rule():
    """STATE RULES must cover approved reviews with current-plan concerns in future_followups."""
    assert "future_followups that are actually current-plan" in _REPAIR_PROMPT or \
           "future_followups" in _REPAIR_PROMPT and "required for the current plan" in _REPAIR_PROMPT


def test_repair_prompt_includes_worked_example_6_to_12():
    """Examples 6-12 must be present."""
    for n in range(6, 13):
        assert f"WORKED EXAMPLE {n}" in _REPAIR_PROMPT


def test_repair_prompt_example_12_same_round_confusion_case():
    """Example 12 must describe the same-round disposition confusion with future_followups."""
    assert "WORKED EXAMPLE 12" in _REPAIR_PROMPT
    assert "same-round" in _REPAIR_PROMPT.lower() or "same-round finding" in _REPAIR_PROMPT.lower()
    assert "future_followups" in _REPAIR_PROMPT


# --- _reviewer_human_requirements_instruction tests ---

def test_reviewer_human_requirements_instruction_pr_review():
    result = _reviewer_human_requirements_instruction("pr_review", ["Requirement 1", "Requirement 2"])
    assert "HUMAN_REQUIREMENTS_RESOLVED" in result
    assert "Requirement 1" in result
    assert "Requirement 2" in result
    assert "AGENT_STATE" in result
    assert "blocking_items" in result


def test_reviewer_human_requirements_instruction_plan_review():
    result = _reviewer_human_requirements_instruction("plan_review", ["Requirement 1"])
    assert "HUMAN_REQUIREMENTS_RESOLVED" in result
    assert "Requirement 1" in result
    assert "AGENT_PLAN_STATE" in result
    assert "blocking_plan_issues" in result


def test_reviewer_human_requirements_instruction_empty_ids():
    result = _reviewer_human_requirements_instruction("pr_review", [])
    assert "(none)" in result
    assert "HUMAN_REQUIREMENTS_RESOLVED" in result


def test_reviewer_human_requirements_instruction_returns_empty_for_none():
    assert _reviewer_human_requirements_instruction("pr_review", None) == ""
    assert _reviewer_human_requirements_instruction("plan_review", None) == ""


def test_reviewer_human_requirements_instruction_rejects_coder_kind():
    with pytest.raises(ValueError, match="reviewer_requirement_ids"):
        _reviewer_human_requirements_instruction("coder_followup", ["Requirement 1"])


def test_attempt_repair_includes_reviewer_requirement_instruction():
    """attempt_repair passes reviewer_requirement_ids into the prompt for pr_review."""
    repaired = structured_pr_review(
        state="approved",
        reviewer="OpenAI Codex",
        human_requirements_resolved=True,
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = repaired

    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=mock_result) as mock_run:
        attempt_repair(
            "malformed review",
            "gemini",
            expected_kind="pr_review",
            reviewer_requirement_ids=["Requirement 1", "Requirement 2"],
        )

    cmd = mock_run.call_args.args[0]
    prompt = cmd[cmd.index("--prompt") + 1]
    assert "Requirement 1" in prompt
    assert "Requirement 2" in prompt
    assert "HUMAN_REQUIREMENTS_RESOLVED" in prompt
    assert "AGENT_STATE" in prompt


def test_attempt_repair_reviewer_requirement_ids_not_included_for_plan_revision():
    """reviewer_requirement_ids raises for non pr_review/plan_review kinds."""
    with pytest.raises(ValueError, match="reviewer_requirement_ids"):
        attempt_repair(
            "malformed plan revision",
            "gemini",
            expected_kind="plan_revision",
            reviewer_requirement_ids=["Requirement 1"],
        )


# --- _surfaced_reviewer_requirement_ids tests ---

def test_surfaced_reviewer_requirement_ids_pr_uses_merged_requirements():
    """PR loop helper returns IDs using PR requirements scope."""
    hr = (
        HumanReviewRequirement(
            source_type="Issue comment",
            author="maintainer",
            created_at="2026-01-01T00:00:00Z",
            url="https://example.com/1",
            body="Use absolute URLs.",
        ),
    )
    ids = _surfaced_reviewer_requirement_ids(hr, requirement_scope="PR requirements")
    assert ids == ("Requirement 1",)


def test_surfaced_reviewer_requirement_ids_plan_uses_issue_requirements():
    """Plan loop helper returns IDs using planning requirements scope."""
    hr = (
        HumanReviewRequirement(
            source_type="Issue comment",
            author="maintainer",
            created_at="2026-01-01T00:00:00Z",
            url="https://example.com/1",
            body="Add regression tests.",
        ),
        HumanReviewRequirement(
            source_type="Issue comment",
            author="maintainer",
            created_at="2026-01-02T00:00:00Z",
            url="https://example.com/2",
            body="Keep backward compatibility.",
        ),
    )
    ids = _surfaced_reviewer_requirement_ids(hr, requirement_scope="planning requirements")
    assert "Requirement 1" in ids
    assert "Requirement 2" in ids


def test_surfaced_reviewer_requirement_ids_empty_for_no_requirements():
    ids = _surfaced_reviewer_requirement_ids([], requirement_scope="PR requirements")
    assert ids == ()


# --- PR loop repair-first tests ---

def _pr_payload_with_human_requirement():
    return {
        "number": 77,
        "state": "OPEN",
        "url": "https://github.com/OWNER/REPO/pull/77",
        "title": "Improve review prompt context",
        "headRefName": "feature/review-context",
        "baseRefName": "main",
        "headRefOid": "abc123",
        "comments": [
            {
                "author": {"login": "maintainer"},
                "createdAt": "2026-05-18T10:00:00Z",
                "url": "https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                "body": "Please use the absolute URL.\n\n-- Human Reviewer",
            }
        ],
        "reviews": [],
    }


def test_pr_loop_repair_missing_hr_marker_recovers_approved(tmp_path):
    """When repair returns approved + HUMAN_REQUIREMENTS_RESOLVED, no synthetic item is injected."""
    approved_without_marker = structured_pr_review(
        state="approved",
        reviewer="OpenAI Codex",
        human_requirements_resolved=False,
    )
    repaired_with_marker = structured_pr_review(
        state="approved",
        reviewer="OpenAI Codex",
        human_requirements_resolved=True,
    )
    runner = FakeRunner(
        codex_outputs=[approved_without_marker],
        pr_payload=_pr_payload_with_human_requirement(),
    )
    config = make_config(tmp_path, max_rounds=1)

    def fake_repair(raw, gemini_cmd, *, expected_kind=None, reviewer_requirement_ids=None, **kwargs):
        assert expected_kind == "pr_review"
        assert reviewer_requirement_ids == ("Requirement 1",)
        return repaired_with_marker

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_repair):
        result = run_pr_loop(runner, pr_number=77, config=config)

    assert result == 0
    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert claude_calls == [], "Coder should not be woken when repair recovers the marker"


def test_pr_loop_repair_missing_hr_marker_returns_blocking_not_synthetic(tmp_path):
    """When repair returns valid blocking, treat as reviewer blocking — no synthetic item."""
    approved_without_marker = structured_pr_review(
        state="approved",
        reviewer="OpenAI Codex",
        human_requirements_resolved=False,
    )
    repaired_blocking = structured_pr_review(
        state="blocking",
        summary="Requirement 1 not satisfied: absolute URL missing.",
        blocking_items=["Requirement 1 not satisfied: absolute URL missing."],
        reviewer="OpenAI Codex",
    )
    # Round 2: coder addresses item-1 (the repaired blocking item) + acks human requirements
    coder_response = structured_coder_followup(
        state="approved",
        addressed_items=["item-1"],
        remaining_items=[],
        human_requirement_ids=["Requirement 1"],
        reviewer="Anthropic Claude",
    )
    runner = FakeRunner(
        claude_outputs=[coder_response],
        codex_outputs=[
            approved_without_marker,
            structured_pr_review(
                state="approved",
                reviewer="OpenAI Codex",
                human_requirements_resolved=True,
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        pr_payload=_pr_payload_with_human_requirement(),
    )
    config = make_config(tmp_path, max_rounds=2)

    pr_review_repair_calls = []
    def fake_repair(raw, gemini_cmd, *, expected_kind=None, reviewer_requirement_ids=None, **kwargs):
        if expected_kind == "pr_review" and reviewer_requirement_ids is not None:
            pr_review_repair_calls.append(raw)
            return repaired_blocking
        return None  # don't interfere with coder_followup repair

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_repair):
        result = run_pr_loop(runner, pr_number=77, config=config)

    assert result == 0
    assert pr_review_repair_calls, "Repair should have been attempted for the reviewer"
    # The repaired blocking item text should appear in a coder prompt
    claude_prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert any("absolute URL missing" in p for p in claude_prompts), \
        "Repaired blocking item should appear in coder prompt"
    # There should be no synthetic Orchestrator item text
    assert not any("Orchestrator" in p and "acknowledging the signed human requirements" in p
                   for p in claude_prompts), \
        "Synthetic orchestrator item must not appear when repair returned valid blocking"


def test_pr_loop_repair_missing_hr_marker_failure_uses_synthetic(tmp_path):
    """When repair fails (returns None), synthetic blocking item is injected."""
    approved_without_marker = structured_pr_review(
        state="approved",
        reviewer="OpenAI Codex",
        human_requirements_resolved=False,
    )
    runner = FakeRunner(
        codex_outputs=[approved_without_marker],
        pr_payload=_pr_payload_with_human_requirement(),
    )
    config = make_config(tmp_path, max_rounds=1)

    def fake_repair(raw, gemini_cmd, *, expected_kind=None, **kwargs):
        return None  # repair fails

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_repair):
        with pytest.raises(AgentLoopError, match="blocking issues after round 1"):
            run_pr_loop(runner, pr_number=77, config=config)



# --- Plan loop repair-first tests ---

def _issue_with_human_requirement():
    return {
        "author": {"login": "maintainer"},
        "createdAt": "2026-05-17T08:00:00Z",
        "body": "Keep the public API unchanged.\n\n-- Human Reviewer",
    }


def test_plan_loop_repair_missing_hr_marker_recovers_approved(tmp_path):
    """Plan loop: repair returning approved+marker suppresses synthetic."""
    plan = (
        "Initial plan.\n"
        f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
        "### Human requirements\n"
        "- Requirement 1: keep the public API unchanged.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    approved_without_marker = structured_plan_review(
        state="approved",
        reviewer="OpenAI Codex",
        human_requirements_resolved=False,
    )
    repaired_with_marker = structured_plan_review(
        state="approved",
        reviewer="OpenAI Codex",
        human_requirements_resolved=True,
    )
    runner = FakeRunner(
        issue_payload=_issue_with_human_requirement(),
        claude_outputs=[plan],
        codex_outputs=[approved_without_marker],
    )
    config = make_config(tmp_path)

    def fake_repair(raw, gemini_cmd, *, expected_kind=None, reviewer_requirement_ids=None, **kwargs):
        assert expected_kind == "plan_review"
        assert reviewer_requirement_ids == ("Requirement 1",)
        return repaired_with_marker

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_repair):
        result = run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert result == 0
    # No second claude call needed (no synthetic blocking item)
    plan_revision_calls = [
        cmd for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] and runner.claude_outputs == []
    ]
    # Verify plan approved comment was posted
    assert any("Approved plan:" in comment for comment in runner.comments)


def test_plan_loop_repair_missing_hr_marker_returns_blocking_not_synthetic(tmp_path):
    """Plan loop: repair returning blocking is treated as reviewer's blocking, not synthetic."""
    plan = (
        "Initial plan.\n"
        f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
        "### Human requirements\n"
        "- Requirement 1: keep the public API unchanged.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    approved_without_marker = structured_plan_review(
        state="approved",
        reviewer="OpenAI Codex",
        human_requirements_resolved=False,
    )
    repaired_blocking = structured_plan_review(
        state="blocking",
        summary="Requirement 1 not satisfied: plan changes the public API.",
        blocking_plan_issues=["Requirement 1 not satisfied: plan changes the public API."],
        reviewer="OpenAI Codex",
    )
    revision = structured_plan_revision(
        summary="Revised plan preserving the public API.",
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        human_requirements=(
            "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n"
            "### Human requirements\n"
            "- Requirement 1: the plan preserves the public API.\n"
        ),
    )
    runner = FakeRunner(
        issue_payload=_issue_with_human_requirement(),
        claude_outputs=[plan, revision],
        codex_outputs=[
            approved_without_marker,
            structured_plan_review(
                summary="Plan looks sound.",
                human_requirements_resolved=True,
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
    )
    config = make_config(tmp_path, max_rounds=3)

    call_count = [0]
    def fake_repair(raw, gemini_cmd, *, expected_kind=None, reviewer_requirement_ids=None, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return repaired_blocking
        return None  # subsequent calls not needed

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_repair):
        result = run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert result == 0
    # Confirm the repaired blocking item text reached the coder
    claude_prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert any("plan changes the public API" in p for p in claude_prompts), \
        "Repaired blocking item text must appear in coder prompt"
    # No synthetic orchestrator text
    assert not any("Orchestrator" in p and "acknowledging the signed human requirements" in p
                   for p in claude_prompts), \
        "Synthetic orchestrator item must not appear when repair returned valid blocking"


def test_plan_loop_repair_missing_hr_marker_failure_uses_synthetic(tmp_path):
    """Plan loop: when repair fails, synthetic blocking item is injected."""
    plan = (
        "Initial plan.\n"
        f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
        "### Human requirements\n"
        "- Requirement 1: keep the public API unchanged.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    approved_without_marker = structured_plan_review(
        state="approved",
        reviewer="OpenAI Codex",
        human_requirements_resolved=False,
    )
    revision = structured_plan_revision(
        summary="Revised plan.",
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        human_requirements=(
            "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n"
            "### Human requirements\n"
            "- Requirement 1: the plan preserves the public API.\n"
        ),
    )
    runner = FakeRunner(
        issue_payload=_issue_with_human_requirement(),
        claude_outputs=[plan, revision],
        codex_outputs=[
            approved_without_marker,
            structured_plan_review(
                summary="Plan looks sound.",
                human_requirements_resolved=True,
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
    )
    config = make_config(tmp_path, max_rounds=3)

    def fake_repair(raw, gemini_cmd, *, expected_kind=None, **kwargs):
        return None  # repair fails

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_repair):
        result = run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert result == 0
    # The synthetic item must have been injected (coder was woken with it)
    claude_prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert any("acknowledging the signed human requirements" in p for p in claude_prompts), \
        "Synthetic item must appear in coder prompt when repair fails"


# --- Protocol regression tests ---

def test_parse_pr_review_rejects_approved_with_same_pr_active_disposition():
    """Approved PR review with active same-pr disposition must fail validation."""
    malformed = structured_pr_review(
        state="approved",
        prior_item_dispositions=[{"item_id": "item-1", "disposition": "same-pr", "note": "Still needed"}],
    )
    with pytest.raises(AgentLoopError, match="Approved reviews must be fully complete"):
        parse_pr_review(malformed, reviewer="OpenAI Codex")


def test_parse_plan_review_rejects_blocking_with_future_disposition_on_prior_item():
    """Blocking plan review with future prior disposition must fail validation."""
    malformed = structured_plan_review(
        state="blocking",
        blocking_plan_issues=["Something is wrong."],
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "future"}],
    )
    with pytest.raises(AgentLoopError, match="[Ff]uture"):
        parse_plan_review(malformed, reviewer="OpenAI Codex")


def test_repair_blocking_formerly_future_prior_item_explicit_disposition():
    """Repair of blocking review with formerly-future prior item produces explicit non-future disposition."""
    malformed = (
        json.dumps({
            "schema_version": 1,
            "kind": "pr_review",
            "state": "blocking",
            "summary": "Fix the memory leak.",
            "blocking_items": ["Fix the memory leak"],
            "same_pr_followups": [],
            "future_followups": [],
            "prior_item_dispositions": [
                {"item_id": "item-1", "disposition": "future"},
            ],
        })
        + "\n<!-- AGENT_STATE: blocking -->\n-- Reviewer"
    )
    repaired = structured_pr_review(
        state="blocking",
        blocking_items=["Fix the memory leak"],
        prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        reviewer="Reviewer",
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = repaired

    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=mock_result) as mock_run:
        result = attempt_repair(
            malformed,
            "gemini",
            expected_kind="pr_review",
            allowed_prior_item_ids=["item-1"],
        )

    assert result == repaired
    cmd = mock_run.call_args.args[0]
    prompt = cmd[cmd.index("--prompt") + 1]
    # Verify the prompt contains the guidance about explicitly dispositioning prior items
    assert "must appear in prior_item_dispositions" in prompt or "prior_item_dispositions" in prompt
    assert "item-1" in prompt


def test_repair_same_round_disposition_confusion_promotes_to_blocking():
    """Repair prompt instructs removing same-round dispositions and promoting current concerns."""
    malformed = structured_plan_review(
        state="approved",
        future_followups=["Reconcile repair examples and validators."],
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
    )
    repaired = structured_plan_review(
        state="blocking",
        blocking_plan_issues=["Reconcile repair examples and validators."],
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = repaired

    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=mock_result) as mock_run:
        result = attempt_repair(
            malformed,
            "gemini",
            expected_kind="plan_review",
            allowed_prior_item_ids=[],
            unknown_prior_item_ids=["item-1"],
            same_round_context="item-1 matches a same-round finding, not a carried prior item.",
        )

    assert result == repaired
    cmd = mock_run.call_args.args[0]
    prompt = cmd[cmd.index("--prompt") + 1]
    # The same-round context should appear in the prompt
    assert "same-round finding" in prompt
    # The guidance about future_followups promoting to blocking should be in STATE RULES
    assert "future_followups" in prompt


# ---------------------------------------------------------------------------
# Round 2 follow-up tests: same-pr/same-plan followup recording and FORMAT fix
# ---------------------------------------------------------------------------

def test_repair_prompt_format_rule_allows_human_requirements_resolved_for_pr_plan_review():
    """FORMAT rule must permit HUMAN_REQUIREMENTS_RESOLVED for pr_review/plan_review, not just plan_revision."""
    assert "pr_review" in _REPAIR_PROMPT or "HUMAN_REQUIREMENTS_RESOLVED" in _REPAIR_PROMPT
    # Ensure the FORMAT rule no longer says "plan_revision only"
    assert "for plan_revision only" not in _REPAIR_PROMPT


def test_pr_loop_repair_blocking_records_same_pr_followups(tmp_path):
    """When repair returns blocking with same_pr_followups, those are recorded as same-pr items."""
    approved_without_marker = structured_pr_review(
        state="approved",
        reviewer="OpenAI Codex",
        human_requirements_resolved=False,
    )
    repaired_blocking = structured_pr_review(
        state="blocking",
        same_pr_followups=["Fix the error message formatting."],
        reviewer="OpenAI Codex",
    )
    # Coder addresses item-1 (same-pr followup)
    coder_response = structured_coder_followup(
        state="approved",
        addressed_items=["item-1"],
        remaining_items=[],
        human_requirement_ids=["Requirement 1"],
        reviewer="Anthropic Claude",
    )
    runner = FakeRunner(
        claude_outputs=[coder_response],
        codex_outputs=[
            approved_without_marker,
            structured_pr_review(
                state="approved",
                reviewer="OpenAI Codex",
                human_requirements_resolved=True,
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        pr_payload=_pr_payload_with_human_requirement(),
    )
    config = make_config(tmp_path, max_rounds=2)

    def fake_repair(raw, gemini_cmd, *, expected_kind=None, reviewer_requirement_ids=None, **kwargs):
        if expected_kind == "pr_review" and reviewer_requirement_ids is not None:
            return repaired_blocking
        return None

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_repair):
        result = run_pr_loop(runner, pr_number=77, config=config)

    assert result == 0
    # The same-pr followup text must appear in the coder's prompt
    claude_prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert any("error message formatting" in p for p in claude_prompts), \
        "Same-PR followup from repaired review must appear in coder prompt"


def test_plan_loop_repair_blocking_records_same_plan_followups(tmp_path):
    """When repair returns blocking with same_plan_followups, those are recorded as same-plan items."""
    plan = (
        "Initial plan.\n"
        f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
        "### Human requirements\n"
        "- Requirement 1: keep the public API unchanged.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    approved_without_marker = structured_plan_review(
        state="approved",
        reviewer="OpenAI Codex",
        human_requirements_resolved=False,
    )
    repaired_blocking = structured_plan_review(
        state="blocking",
        same_plan_followups=["Add a regression test for the parser edge case."],
        reviewer="OpenAI Codex",
    )
    revision = structured_plan_revision(
        summary="Revised plan with regression test.",
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        human_requirements=(
            "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n"
            "### Human requirements\n"
            "- Requirement 1: the plan preserves the public API.\n"
        ),
    )
    runner = FakeRunner(
        issue_payload=_issue_with_human_requirement(),
        claude_outputs=[plan, revision],
        codex_outputs=[
            approved_without_marker,
            structured_plan_review(
                summary="Plan looks sound.",
                human_requirements_resolved=True,
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
    )
    config = make_config(tmp_path, max_rounds=3)

    call_count = [0]
    def fake_repair(raw, gemini_cmd, *, expected_kind=None, reviewer_requirement_ids=None, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1 and expected_kind == "plan_review":
            return repaired_blocking
        return None

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_repair):
        result = run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert result == 0
    # The same-plan followup text must appear in the coder's prompt
    claude_prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert any("regression test for the parser" in p for p in claude_prompts), \
        "Same-plan followup from repaired review must appear in coder prompt"


# ---------------------------------------------------------------------------
# Tests for issue #273: deterministic recovery of same-round prior-item dispositions
# ---------------------------------------------------------------------------

from coding_review_agent_loop.repair import strip_unknown_prior_item_dispositions


# --- Unit tests for strip_unknown_prior_item_dispositions ---

def test_strip_unknown_prior_item_dispositions_removes_unknown_from_empty_ledger():
    raw = structured_plan_review(
        state="approved",
        summary="Plan review complete.",
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    result = strip_unknown_prior_item_dispositions(
        raw, allowed_ids=frozenset(), expected_kind="plan_review"
    )
    assert result is not None
    payload, json_end = json.JSONDecoder().raw_decode(result.lstrip())
    assert payload["prior_plan_item_dispositions"] == []
    trailing = result.lstrip()[json_end:]
    assert "<!-- AGENT_PLAN_STATE: approved -->" in trailing
    assert "-- Google Gemini" in trailing


def test_strip_unknown_prior_item_dispositions_preserves_valid_removes_unknown():
    raw = structured_plan_review(
        state="approved",
        summary="Plan review complete.",
        prior_plan_item_dispositions=[
            {"item_id": "item-1", "disposition": "resolved"},
            {"item_id": "item-9", "disposition": "resolved"},
        ],
        reviewer="Google Gemini",
    )
    result = strip_unknown_prior_item_dispositions(
        raw, allowed_ids=frozenset({"item-1"}), expected_kind="plan_review"
    )
    assert result is not None
    payload, _ = json.JSONDecoder().raw_decode(result.lstrip())
    assert payload["prior_plan_item_dispositions"] == [{"item_id": "item-1", "disposition": "resolved"}]


def test_strip_unknown_prior_item_dispositions_preserves_all_other_fields():
    raw = structured_plan_review(
        state="approved",
        summary="Looks good overall.",
        same_plan_followups=["Consider adding benchmarks."],
        future_followups=["Improve error messages later."],
        prior_plan_item_dispositions=[{"item_id": "item-9", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    result = strip_unknown_prior_item_dispositions(
        raw, allowed_ids=frozenset(), expected_kind="plan_review"
    )
    assert result is not None
    payload, _ = json.JSONDecoder().raw_decode(result.lstrip())
    assert payload["summary"] == "Looks good overall."
    assert payload["same_plan_followups"] == ["Consider adding benchmarks."]
    assert payload["future_followups"] == ["Improve error messages later."]
    assert payload["state"] == "approved"
    assert payload["prior_plan_item_dispositions"] == []


def test_strip_unknown_prior_item_dispositions_removes_from_pr_review():
    raw = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    result = strip_unknown_prior_item_dispositions(
        raw, allowed_ids=frozenset(), expected_kind="pr_review"
    )
    assert result is not None
    payload, _ = json.JSONDecoder().raw_decode(result.lstrip())
    assert payload["prior_item_dispositions"] == []
    assert payload["kind"] == "pr_review"


def test_strip_unknown_prior_item_dispositions_returns_none_if_nothing_to_remove():
    raw = structured_plan_review(
        state="approved",
        summary="Plan review complete.",
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    result = strip_unknown_prior_item_dispositions(
        raw, allowed_ids=frozenset({"item-1"}), expected_kind="plan_review"
    )
    assert result is None


def test_strip_unknown_prior_item_dispositions_returns_none_for_markdown():
    raw = "## Plan Review\n\nLooks good.\n\n<!-- AGENT_PLAN_STATE: approved -->\n-- Google Gemini"
    result = strip_unknown_prior_item_dispositions(
        raw, allowed_ids=frozenset(), expected_kind="plan_review"
    )
    assert result is None


def test_strip_unknown_prior_item_dispositions_returns_none_for_wrong_kind():
    raw = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    result = strip_unknown_prior_item_dispositions(
        raw, allowed_ids=frozenset(), expected_kind="plan_review"
    )
    assert result is None


def test_strip_unknown_prior_item_dispositions_returns_none_for_unsupported_kind():
    raw = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    result = strip_unknown_prior_item_dispositions(
        raw, allowed_ids=frozenset(), expected_kind="coder_followup"
    )
    assert result is None


# --- Integration tests via _run_validated_agent ---

def test_run_validated_agent_deterministically_strips_unknown_plan_review_without_repair(tmp_path):
    malformed_review = structured_plan_review(
        state="approved",
        summary="Plan approved.",
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the plan.",
            marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
            validate=lambda text: _validate_plan_review_response(
                text,
                reviewer="Google Gemini",
                unresolved_items=(),
            ),
            use_repair=True,
            repair_expected_kind="plan_review",
            repair_allowed_prior_item_ids=(),
            ledger_incomplete=False,
        )

    repair_mock.assert_not_called()
    parsed = json.loads(response.text.split("\n")[0])
    assert parsed["prior_plan_item_dispositions"] == []
    assert parsed["state"] == "approved"


def test_run_validated_agent_deterministically_strips_unknown_pr_review_without_repair(tmp_path):
    malformed_review = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the PR.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: _validate_review_response(
                text,
                reviewer="Google Gemini",
                unresolved_items=(),
            ),
            use_repair=True,
            repair_expected_kind="pr_review",
            repair_allowed_prior_item_ids=(),
            ledger_incomplete=False,
        )

    repair_mock.assert_not_called()
    parsed = json.loads(response.text.split("\n")[0])
    assert parsed["prior_item_dispositions"] == []
    assert parsed["kind"] == "pr_review"


def test_run_validated_agent_deterministic_strip_preserves_valid_removes_unknown_mixed(tmp_path):
    carried_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Must-fix prior item.",
        status="blocking",
    )
    malformed_review = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[
            {"item_id": "item-1", "disposition": "resolved"},
            {"item_id": "item-9", "disposition": "resolved"},
        ],
        reviewer="Google Gemini",
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the PR.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: _validate_review_response(
                text,
                reviewer="Google Gemini",
                unresolved_items=(carried_item,),
            ),
            use_repair=True,
            repair_expected_kind="pr_review",
            repair_allowed_prior_item_ids=("item-1",),
            ledger_incomplete=False,
        )

    repair_mock.assert_not_called()
    assert response.marker_value.dispositions[0].item_id == "item-1"
    assert len(response.marker_value.dispositions) == 1


def test_run_validated_agent_deterministic_strip_logs_removed_and_allowed_ids(tmp_path, capsys):
    malformed_review = structured_plan_review(
        state="approved",
        summary="Plan approved.",
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0, quiet=False)

    _run_validated_agent(
        runner,
        agent="gemini",
        config=config,
        prompt="Review the plan.",
        marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
        validate=lambda text: _validate_plan_review_response(
            text,
            reviewer="Google Gemini",
            unresolved_items=(),
        ),
        use_repair=True,
        repair_expected_kind="plan_review",
        repair_allowed_prior_item_ids=(),
        ledger_incomplete=False,
    )

    output = capsys.readouterr().err
    assert "deterministically removed unknown prior-item disposition ID(s)" in output
    assert "item-1" in output
    assert "(none)" in output


def test_run_validated_agent_deterministic_strip_falls_through_to_repair_on_secondary_failure(tmp_path):
    missing_item = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Must-fix item not dispositioned.",
        status="blocking",
    )
    malformed_review = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[{"item_id": "item-9", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    repaired_review = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[{"item_id": "item-2", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch(
        "coding_review_agent_loop.orchestrator.attempt_repair", return_value=repaired_review
    ) as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the PR.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: _validate_review_response(
                text,
                reviewer="Google Gemini",
                unresolved_items=(missing_item,),
            ),
            use_repair=True,
            repair_expected_kind="pr_review",
            repair_allowed_prior_item_ids=("item-2",),
            ledger_incomplete=False,
        )

    repair_mock.assert_called_once()
    assert response.text == repaired_review


def test_run_validated_agent_real_264_shape_approved_plan_review_same_round_item1_future_followups(tmp_path):
    malformed_review = structured_plan_review(
        state="approved",
        summary="Plan is sound.",
        future_followups=["Consider adding retries."],
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved", "note": "Now covered."}],
        reviewer="Google Gemini",
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the plan.",
            marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
            validate=lambda text: _validate_plan_review_response(
                text,
                reviewer="Google Gemini",
                unresolved_items=(),
                current_round_items=(),
            ),
            use_repair=True,
            repair_expected_kind="plan_review",
            repair_allowed_prior_item_ids=(),
            ledger_incomplete=False,
        )

    repair_mock.assert_not_called()
    parsed = json.loads(response.text.split("\n")[0])
    assert parsed["prior_plan_item_dispositions"] == []
    assert parsed["state"] == "approved"
    assert parsed["future_followups"] == ["Consider adding retries."]
    assert parsed["summary"] == "Plan is sound."


def test_run_validated_agent_deterministic_strip_skipped_when_ledger_incomplete(tmp_path):
    malformed_review = structured_plan_review(
        state="approved",
        summary="Plan approved.",
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        with pytest.raises(AgentLoopError) as exc_info:
            _run_validated_agent(
                runner,
                agent="gemini",
                config=config,
                prompt="Review the plan.",
                marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
                validate=lambda text: _validate_plan_review_response(
                    text,
                    reviewer="Google Gemini",
                    unresolved_items=(),
                ),
                use_repair=True,
                repair_expected_kind="plan_review",
                repair_allowed_prior_item_ids=(),
                ledger_incomplete=True,
            )

    repair_mock.assert_not_called()
    assert "item-1" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Antigravity (agy) backend + Gemini retirement guidance (#215)
# ---------------------------------------------------------------------------


def test_antigravity_backend_command_and_prefers_response_file(tmp_path):
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend
    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    runner = FakeRunner(
        antigravity_outputs=[("stdout fallback text", 0)],
        public_response_outputs=["response file text"],
    )
    config = make_config(
        tmp_path,
        antigravity_dir=agy_dir,
        antigravity_cmd="agy",
        antigravity_model="Gemini 3.1 Pro (High)",
        antigravity_args=("--dangerously-skip-permissions",),
    )
    result = AntigravityBackend().run(runner, config, "Review this PR.", run_id="run-1")
    cmd = runner.commands[-1][0]
    assert cmd[0] == "agy"
    assert cmd[cmd.index("--model") + 1] == "Gemini 3.1 Pro (High)"
    assert "--dangerously-skip-permissions" in cmd
    # The prompt is the value of --print and must be the last argument (agy's
    # --print/--prompt consumes the next token, not a trailing positional).
    assert cmd[-2] == "--print"
    assert "Review this PR." in cmd[-1]
    assert result.text == "response file text"
    assert result.session_id is None  # agy --print exposes no conversation id


def test_antigravity_backend_stdout_fallback(tmp_path):
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend
    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    runner = FakeRunner(antigravity_outputs=[("plain stdout review", 0)])
    config = make_config(tmp_path, antigravity_dir=agy_dir)
    result = AntigravityBackend().run(runner, config, "Review this PR.", run_id="run-1")
    assert result.text == "plain stdout review"
    assert result.text_source == "stdout"


def test_antigravity_backend_fallback_chain_on_quota_signal(tmp_path):
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend
    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    runner = FakeRunner(
        antigravity_outputs=[
            ("quota exceeded please try again", 1),
            ("quota exceeded again", 1),
            ("ok fallback answered", 0)
        ]
    )
    config = make_config(
        tmp_path,
        antigravity_dir=agy_dir,
        antigravity_models=("ModelA", "ModelB", "ModelC"),
        antigravity_quota_signatures=("quota",)
    )
    result = AntigravityBackend().run(runner, config, "Review", run_id="r1")
    
    assert runner.commands[-3][0][runner.commands[-3][0].index("--model") + 1] == "ModelA"
    assert runner.commands[-2][0][runner.commands[-2][0].index("--model") + 1] == "ModelB"
    assert runner.commands[-1][0][runner.commands[-1][0].index("--model") + 1] == "ModelC"
    
    assert result.text == "ok fallback answered"
    assert result.model_used == "ModelC"


def test_antigravity_backend_stops_on_other_errors(tmp_path):
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend
    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    runner = FakeRunner(
        antigravity_outputs=[
            ("some regular error", 1),
            ("ok fallback answered", 0)
        ]
    )
    config = make_config(
        tmp_path,
        antigravity_dir=agy_dir,
        antigravity_models=("ModelA", "ModelB"),
        antigravity_quota_signatures=("quota",)
    )
    result = AntigravityBackend().run(runner, config, "Review", run_id="r1")
    
    assert runner.commands[-1][0][runner.commands[-1][0].index("--model") + 1] == "ModelA"
    assert result.returncode == 1
    assert result.model_used == "ModelA"


def test_antigravity_backend_ignores_partial_response_file_on_fallback(tmp_path):
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend
    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    runner = FakeRunner(
        antigravity_outputs=[
            ("quota error", 1),
            ("success", 0)
        ],
        public_response_outputs=[
            "partial failed response",
            "successful response"
        ]
    )
    config = make_config(
        tmp_path,
        antigravity_dir=agy_dir,
        antigravity_models=("ModelA", "ModelB"),
        antigravity_quota_signatures=("quota",)
    )
    result = AntigravityBackend().run(runner, config, "Review", run_id="r1")
    
    assert result.text == "successful response"
    assert result.model_used == "ModelB"


def test_antigravity_backend_writes_gemini_md_single_shot_instruction(tmp_path):
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    captured: list[str] = []

    class CapturingRunner(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            gemini_md = cwd / "GEMINI.md"
            captured.append(gemini_md.read_text(encoding="utf-8") if gemini_md.exists() else "")
            return super().run_with_log(args, cwd=cwd, **kwargs)

    runner = CapturingRunner(antigravity_outputs=[("ok", 0)])
    config = make_config(tmp_path, antigravity_dir=agy_dir)
    AntigravityBackend().run(runner, config, "Review this PR.", run_id="r1")

    assert captured, "run_with_log was not called"
    assert "Do NOT spawn background execution tasks" in captured[0]
    # prefix is stripped → file deleted (no remaining content after it)
    assert not (agy_dir / "GEMINI.md").exists()
    # Lock file must not appear in the worktree root (it lives in .git/ only)
    assert not (agy_dir / "GEMINI.md.lock").exists()


def test_antigravity_backend_preserves_existing_gemini_md_during_and_after_run(tmp_path):
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    original_content = "# My project rules\nUse tabs.\n"
    (agy_dir / "GEMINI.md").write_text(original_content, encoding="utf-8")
    captured: list[str] = []

    class CapturingRunner(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            gemini_md = cwd / "GEMINI.md"
            captured.append(gemini_md.read_text(encoding="utf-8") if gemini_md.exists() else "")
            return super().run_with_log(args, cwd=cwd, **kwargs)

    runner = CapturingRunner(antigravity_outputs=[("ok", 0)])
    config = make_config(tmp_path, antigravity_dir=agy_dir)
    AntigravityBackend().run(runner, config, "Review this PR.", run_id="r1")

    assert captured, "run_with_log was not called"
    # Instruction was prepended before the original content during the run
    assert "Do NOT spawn background execution tasks" in captured[0]
    assert "My project rules" in captured[0]
    assert captured[0].index("Do NOT spawn") < captured[0].index("My project rules")
    # Prefix stripped after the run → original content remains
    after = (agy_dir / "GEMINI.md").read_text(encoding="utf-8")
    assert after == original_content
    assert "Do NOT spawn" not in after


def test_antigravity_backend_preserves_agent_edits_to_gemini_md(tmp_path):
    """Agent (coder role) edits the content after our prefix — preserved after run."""
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    original_content = "# Original rules\n"
    (agy_dir / "GEMINI.md").write_text(original_content, encoding="utf-8")

    class EditingRunner(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            # Simulate agent appending to the file after the injected prefix
            gemini_md = cwd / "GEMINI.md"
            current = gemini_md.read_text(encoding="utf-8")
            gemini_md.write_text(current + "# New agent-added rules\n", encoding="utf-8")
            return super().run_with_log(args, cwd=cwd, **kwargs)

    runner = EditingRunner(antigravity_outputs=[("ok", 0)])
    config = make_config(tmp_path, antigravity_dir=agy_dir)
    AntigravityBackend().run(runner, config, "Review this PR.", run_id="r1")

    after = (agy_dir / "GEMINI.md").read_text(encoding="utf-8")
    # Original content and agent's new content both present; our prefix stripped
    assert "Original rules" in after
    assert "New agent-added rules" in after
    assert "Do NOT spawn" not in after


def test_antigravity_backend_cleans_up_gemini_md_on_exception(tmp_path):
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)

    class RaisingRunner(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            raise RuntimeError("subprocess failed")

    # Sub-test A: no pre-existing GEMINI.md — prefix stripped → file deleted
    runner = RaisingRunner()
    config = make_config(tmp_path, antigravity_dir=agy_dir)
    with pytest.raises(RuntimeError):
        AntigravityBackend().run(runner, config, "Review this PR.", run_id="r1")
    assert not (agy_dir / "GEMINI.md").exists()

    # Sub-test B: pre-existing GEMINI.md — prefix stripped, original content restored
    original_content = "# Existing rules\n"
    (agy_dir / "GEMINI.md").write_text(original_content, encoding="utf-8")
    runner2 = RaisingRunner()
    with pytest.raises(RuntimeError):
        AntigravityBackend().run(runner2, config, "Review this PR.", run_id="r2")
    after = (agy_dir / "GEMINI.md").read_text(encoding="utf-8")
    assert after == original_content
    assert "Do NOT spawn" not in after


def test_antigravity_backend_gemini_md_lock_serializes_concurrent_access(tmp_path):
    """flock on GEMINI.md.lock prevents a second run from starting until the first
    completes its inject→run→strip sequence."""
    import fcntl
    import threading
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    config = make_config(tmp_path, antigravity_dir=agy_dir)

    order: list[str] = []
    # Lock lives in .git/ to avoid polluting the worktree
    lock_path = agy_dir / ".git" / "GEMINI.md.lock"
    (agy_dir / ".git").mkdir(parents=True, exist_ok=True)

    # Thread A: pre-acquires the exclusive lock, records "A-holds", sleeps briefly,
    # records "A-releases", then releases the lock. This simulates another process
    # holding the lock while running agy.
    lock_acquired = threading.Event()
    lock_released = threading.Event()

    def hold_lock():
        lf = lock_path.open("a+")
        fcntl.flock(lf, fcntl.LOCK_EX)
        order.append("A-holds")
        lock_acquired.set()
        lock_released.wait()
        order.append("A-releases")
        fcntl.flock(lf, fcntl.LOCK_UN)
        lf.close()

    t = threading.Thread(target=hold_lock, daemon=True)
    t.start()
    lock_acquired.wait()

    # Thread B (main): tries to run AntigravityBackend — should block on LOCK_EX
    # until Thread A releases.
    run_started = threading.Event()

    class RecordingRunner(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            order.append("B-run")
            return super().run_with_log(args, cwd=cwd, **kwargs)

    def run_backend():
        AntigravityBackend().run(
            RecordingRunner(antigravity_outputs=[("ok", 0)]),
            config,
            "Review.",
            run_id="r1",
        )
        order.append("B-done")

    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()

    # Give the backend thread a moment to block on the lock, then release it.
    import time
    time.sleep(0.05)
    lock_released.set()
    t.join(timeout=5)
    backend_thread.join(timeout=10)

    # A-holds must precede B-run and A-releases must precede B-run
    assert "A-holds" in order
    assert "A-releases" in order
    assert "B-run" in order
    assert order.index("A-holds") < order.index("B-run")
    assert order.index("A-releases") < order.index("B-run")


def test_antigravity_module_imports_without_fcntl():
    """Antigravity module must import cleanly even when fcntl is unavailable (Windows)."""
    import importlib
    import sys

    # Remove any cached import of the module under test
    mods_to_remove = [k for k in sys.modules if "antigravity" in k]
    for m in mods_to_remove:
        del sys.modules[m]

    # Simulate a platform without fcntl by hiding it
    original = sys.modules.pop("fcntl", None)
    sys.modules["fcntl"] = None  # type: ignore[assignment]
    try:
        import coding_review_agent_loop.agents.antigravity as mod
        assert hasattr(mod, "AntigravityBackend")
    finally:
        if original is not None:
            sys.modules["fcntl"] = original
        else:
            sys.modules.pop("fcntl", None)
        # Re-remove so later tests get a clean import
        for k in list(sys.modules):
            if "antigravity" in k:
                del sys.modules[k]


def test_antigravity_backend_git_lock_path_follows_linked_worktree(tmp_path):
    """_git_lock_path resolves a file-form .git marker (linked worktree) to the real
    git dir instead of trying to mkdir the .git file."""
    from coding_review_agent_loop.agents.antigravity import _git_lock_path

    agy_dir = tmp_path / "worktree"
    agy_dir.mkdir()
    real_git_dir = tmp_path / "repo.git" / "worktrees" / "wt"
    real_git_dir.mkdir(parents=True)

    # Simulate the .git file that git worktree add creates
    (agy_dir / ".git").write_text(
        f"gitdir: {real_git_dir}\n", encoding="utf-8"
    )

    lock = _git_lock_path(agy_dir)
    assert lock.parent == real_git_dir
    assert lock.name == "GEMINI.md.lock"
    # Must not attempt to mkdir over the .git file
    assert (agy_dir / ".git").is_file()


def test_antigravity_backend_strips_public_response_marker(tmp_path):
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend
    from coding_review_agent_loop.protocol import PUBLIC_RESPONSE_MARKER
    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    stdout = (
        "I will inspect the diff and run the tests.\n"
        f"{PUBLIC_RESPONSE_MARKER}\n"
        "STATE: approved\n\nLooks good to me."
    )
    runner = FakeRunner(antigravity_outputs=[(stdout, 0)])
    config = make_config(tmp_path, antigravity_dir=agy_dir)
    result = AntigravityBackend().run(runner, config, "Review this PR.", run_id="run-1")
    assert result.text == "STATE: approved\n\nLooks good to me."
    assert result.text_source == "stdout_marker"
    assert "I will inspect" not in result.text


def test_antigravity_backend_resume_uses_conversation(tmp_path):
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend
    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    runner = FakeRunner(antigravity_outputs=[("ok", 0)])
    config = make_config(tmp_path, antigravity_dir=agy_dir)
    AntigravityBackend().run(runner, config, "x", session_id="conv-7", run_id="r")
    cmd = runner.commands[-1][0]
    assert cmd[cmd.index("--conversation") + 1] == "conv-7"


def test_runner_pty_reports_tty_and_strips_ansi(tmp_path):
    """The real PTY path: the child sees a TTY and ANSI codes are stripped."""
    import sys
    from coding_review_agent_loop.runner import Runner, strip_ansi

    assert strip_ansi("\x1b[31mred\x1b[0m\r\ndone") == "red\ndone"

    program = (
        "import sys\n"
        "sys.stdout.write('istty=%s\\n' % sys.stdout.isatty())\n"
        "sys.stdout.write('\\x1b[32mGREEN\\x1b[0m\\n')\n"
    )
    log_path = tmp_path / "logs" / "pty.log"
    result = Runner().run_with_log(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        log_path=log_path,
        label="PtyProbe",
        progress_interval_seconds=999,
        check=True,
        use_pty=True,
    )
    assert "istty=True" in result.stdout
    assert "GREEN" in result.stdout
    assert "\x1b[" not in result.stdout  # ANSI stripped from captured output
    assert result.returncode == 0


@pytest.mark.parametrize("use_pty", [False, True])
def test_runner_retries_dangling_symlink_spawn_and_recovers(
    monkeypatch,
    tmp_path,
    use_pty,
):
    import coding_review_agent_loop.runner as runner_module

    missing_target = tmp_path / "updating-agent-target"
    command = tmp_path / "agent"
    command.symlink_to(missing_target)
    runner = Runner()
    runner.remember_agent_command(str(command), str(command), "--codex-cmd")
    original_popen = runner_module.subprocess.Popen
    popen_calls = []
    sleep_calls = []

    def flaky_popen(*args, **kwargs):
        popen_calls.append(args[0])
        if len(popen_calls) == 1:
            raise FileNotFoundError(str(command))
        return original_popen(*args, **kwargs)

    def restore_command(delay):
        sleep_calls.append(delay)
        command.unlink()
        command.symlink_to(sys.executable)

    monkeypatch.setattr(runner_module.subprocess, "Popen", flaky_popen)
    monkeypatch.setattr(runner_module.time, "sleep", restore_command)

    result = runner.run_with_log(
        [str(command), "-c", "print('recovered')"],
        cwd=tmp_path,
        log_path=tmp_path / "logs" / f"retry-{use_pty}.log",
        label="Retry probe",
        progress_interval_seconds=999,
        use_pty=use_pty,
    )

    assert result.returncode == 0
    assert "recovered" in result.stdout
    assert len(popen_calls) == 2
    assert sleep_calls[0] == 2
    assert all(delay == 1 for delay in sleep_calls[1:])


@pytest.mark.parametrize("use_pty", [False, True])
def test_runner_dangling_symlink_spawn_retry_is_bounded(
    monkeypatch,
    tmp_path,
    use_pty,
):
    import coding_review_agent_loop.runner as runner_module

    command = tmp_path / "agent"
    command.symlink_to(tmp_path / "missing-target")
    runner = Runner()
    runner.remember_agent_command(str(command), str(command), "--codex-cmd")
    popen_calls = []
    sleep_calls = []

    def missing_popen(*args, **kwargs):
        popen_calls.append(args[0])
        raise FileNotFoundError(str(command))

    monkeypatch.setattr(runner_module.subprocess, "Popen", missing_popen)
    monkeypatch.setattr(
        runner_module.time,
        "sleep",
        lambda delay: sleep_calls.append(delay),
    )

    with pytest.raises(
        AgentLoopError,
        match=r"CLI not found on PATH.*--codex-cmd",
    ):
        runner.run_with_log(
            [str(command), "--version"],
            cwd=tmp_path,
            log_path=tmp_path / "logs" / f"bounded-{use_pty}.log",
            label="Bounded retry probe",
            progress_interval_seconds=999,
            use_pty=use_pty,
        )

    assert len(popen_calls) == 3
    assert sleep_calls == [2, 2]


def test_runner_missing_command_without_dangling_evidence_does_not_retry(
    monkeypatch,
    tmp_path,
):
    import coding_review_agent_loop.runner as runner_module

    popen_calls = []
    sleep_calls = []

    def missing_popen(*args, **kwargs):
        popen_calls.append(args[0])
        raise FileNotFoundError("missing-agent")

    monkeypatch.setattr(runner_module.shutil, "which", lambda command: None)
    monkeypatch.setattr(runner_module.subprocess, "Popen", missing_popen)
    monkeypatch.setattr(
        runner_module.time,
        "sleep",
        lambda delay: sleep_calls.append(delay),
    )

    with pytest.raises(AgentLoopError, match="missing-agent CLI not found on PATH"):
        Runner().run_with_log(
            ["missing-agent", "--version"],
            cwd=tmp_path,
            log_path=tmp_path / "logs" / "missing.log",
            label="Missing probe",
            progress_interval_seconds=999,
        )

    assert len(popen_calls) == 1
    assert sleep_calls == []


def test_antigravity_registry():
    from coding_review_agent_loop.agents.registry import (
        agent_display_name,
        agent_signature,
        get_backend,
    )
    assert agent_display_name("antigravity") == "Antigravity"
    assert agent_signature("antigravity") == "Google Antigravity"
    assert get_backend("antigravity").name == "antigravity"


def test_config_from_args_antigravity_defaults(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr", "123", "--repo", "OWNER/REPO",
        "--coder", "antigravity", "--reviewer", "codex",
        "--codex-dir", str(tmp_path / "codex"),
        "--dangerous-agent-permissions",
    ])
    config = config_from_args(args, FakeRunner())
    assert config.coder == "antigravity"
    assert config.antigravity_cmd == "agy"
    assert config.antigravity_model is None
    assert config.antigravity_models == ("Gemini 3.5 Flash (High)", "Gemini 3.1 Pro (High)")
    assert config.antigravity_quota_signatures == ("quota", "rate limit", "resource exhausted", "RESOURCE_EXHAUSTED", "429")
    assert config.antigravity_args == ("--dangerously-skip-permissions",)
    assert config.antigravity_dir == default_agent_workdir("OWNER/REPO", "antigravity").resolve()
    # antigravity is the coder -> primary/log dir lives under its checkout.
    assert str(config.log_dir).startswith(str(config.antigravity_dir))


def test_antigravity_quota_signatures_default_single_source(tmp_path):
    """The quota-signatures default comes from one constant — no drift across the
    dataclass field, the CLI flag, and config_from_args (#348, #350)."""
    from coding_review_agent_loop.config import DEFAULT_ANTIGRAVITY_QUOTA_SIGNATURES as DEFAULT
    # dataclass field default
    assert make_config(tmp_path).antigravity_quota_signatures == DEFAULT
    # CLI flag default and config_from_args both derive from the constant
    parser = build_parser()
    args = parser.parse_args(["pr", "123", "--repo", "OWNER/REPO", "--codex-dir", str(tmp_path / "codex")])
    assert tuple(args.antigravity_quota_signatures) == DEFAULT
    assert config_from_args(args, FakeRunner()).antigravity_quota_signatures == DEFAULT


def test_antigravity_models_default_chain_from_constant(tmp_path):
    """The default model fallback chain resolves from the named constant when
    neither a legacy model nor an explicit chain is given."""
    from coding_review_agent_loop.config import DEFAULT_ANTIGRAVITY_MODELS
    assert make_config(tmp_path).antigravity_models == DEFAULT_ANTIGRAVITY_MODELS
    parser = build_parser()
    args = parser.parse_args(["pr", "123", "--repo", "OWNER/REPO", "--codex-dir", str(tmp_path / "codex")])
    assert config_from_args(args, FakeRunner()).antigravity_models == DEFAULT_ANTIGRAVITY_MODELS


def test_cli_rejects_both_antigravity_model_flags():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "pr", "123", "--repo", "OWNER/REPO", "--antigravity-model", "Gemini", "--antigravity-models", "Gemini", "Claude"
        ])


def test_config_rejects_blank_antigravity_models(tmp_path):
    with pytest.raises(AgentLoopError, match="cannot be empty or contain blank entries"):
        make_config(tmp_path, antigravity_models=("",))


def test_config_rejects_both_model_flags(tmp_path):
    with pytest.raises(AgentLoopError, match="Cannot specify both antigravity_model and a custom antigravity_models chain"):
        make_config(tmp_path, antigravity_model="Gemini", antigravity_models=("Gemini", "Claude"))


def test_config_rejects_both_model_flags_even_if_default(tmp_path):
    with pytest.raises(AgentLoopError, match="Cannot specify both antigravity_model and a custom antigravity_models chain"):
        make_config(tmp_path, antigravity_model="Gemini", antigravity_models=("Gemini 3.1 Pro (High)", "Gemini 3.5 Flash (High)"))


def test_distinct_workdir_validation_covers_antigravity(tmp_path):
    from coding_review_agent_loop.config import ensure_distinct_workdirs
    shared = tmp_path / "shared"
    config = make_config(
        tmp_path,
        coder="antigravity",
        reviewer="codex",
        allow_shared_dir=False,
        antigravity_dir=shared,
        codex_dir=shared,
    )
    with pytest.raises(AgentLoopError, match="same directory"):
        ensure_distinct_workdirs(config)


def test_gemini_retirement_signal():
    from coding_review_agent_loop.agents.gemini import _gemini_retirement_signal
    assert _gemini_retirement_signal("Error: quota exceeded")
    assert _gemini_retirement_signal("PERMISSION_DENIED for this account")
    assert _gemini_retirement_signal("request was unauthenticated")
    assert not _gemini_retirement_signal("SyntaxError: invalid token")
    assert not _gemini_retirement_signal("connection reset by peer")


def test_gemini_date_advisory_fires_only_in_window(tmp_path, capsys, monkeypatch):
    import coding_review_agent_loop.agents.gemini as gm
    from datetime import date

    class _NearCutoff(date):
        @classmethod
        def today(cls):
            return date(2026, 6, 18)

    class _BeforeWindow(date):
        @classmethod
        def today(cls):
            return date(2026, 1, 1)

    config = make_config(tmp_path, quiet=False)

    monkeypatch.setattr(gm, "date", _NearCutoff)
    gm.BACKEND.run(FakeRunner(gemini_outputs=[("ok", 0)]), config, "Review", run_id="r1")
    assert "2026-06-18" in capsys.readouterr().err  # advisory fires near cutoff

    monkeypatch.setattr(gm, "date", _BeforeWindow)
    gm.BACKEND.run(FakeRunner(gemini_outputs=[("ok", 0)]), config, "Review", run_id="r2")
    assert "Antigravity" not in capsys.readouterr().err  # no advisory long before cutoff


def test_gemini_failure_appends_migration_guidance(tmp_path, capsys, monkeypatch):
    import coding_review_agent_loop.agents.gemini as gm
    from datetime import date

    class _BeforeWindow(date):
        @classmethod
        def today(cls):
            return date(2026, 1, 1)  # advisory off, so this isolates the failure path

    monkeypatch.setattr(gm, "date", _BeforeWindow)
    config = make_config(tmp_path, quiet=False)
    runner = FakeRunner(gemini_outputs=[{"stdout": "Error: quota exceeded", "returncode": 1}])
    result = gm.BACKEND.run(runner, config, "Review", run_id="r")
    err = capsys.readouterr().err
    assert "Antigravity" in err and "2026-06-18" in err
    # The guidance must travel with the *returned* result (raw_output and text),
    # since run_external classifies/persists failures from those, not stderr (#215).
    assert "2026-06-18" in result.raw_output
    assert "antigravity" in result.raw_output
    assert result.raw_output.startswith("Error: quota exceeded")  # original error preserved
    assert "2026-06-18" in result.text


def test_gemini_success_does_not_append_migration_guidance(tmp_path, monkeypatch):
    import coding_review_agent_loop.agents.gemini as gm
    from datetime import date

    class _BeforeWindow(date):
        @classmethod
        def today(cls):
            return date(2026, 1, 1)

    monkeypatch.setattr(gm, "date", _BeforeWindow)
    config = make_config(tmp_path)
    runner = FakeRunner(gemini_outputs=[("STATE: approved\n\nLGTM", 0)])
    result = gm.BACKEND.run(runner, config, "Review", run_id="r")
    assert "2026-06-18" not in result.raw_output  # no guidance on a clean success
    assert "2026-06-18" not in result.text


# ---------------------------------------------------------------------------
# Dynamic model-specific signatures (#332)
# ---------------------------------------------------------------------------


def test_agent_signature_generic_without_config():
    from coding_review_agent_loop.agents.registry import agent_signature
    assert agent_signature("codex") == "OpenAI Codex"
    assert agent_signature("antigravity") == "Google Antigravity"


def test_agent_signature_uses_configured_model(tmp_path):
    from coding_review_agent_loop.agents.registry import agent_signature
    config = make_config(tmp_path, codex_model="gpt-5.2-codex", codex_reasoning_effort="high")
    assert agent_signature("codex", config) == "OpenAI Codex: gpt-5.2-codex (high)"
    # antigravity model is always declared (effort already embedded).
    assert agent_signature("antigravity", config) == "Google Antigravity: Gemini 3.5 Flash (High)"
    # gemini with no declared model falls back to the generic signature.
    assert agent_signature("gemini", make_config(tmp_path)) == "Google Gemini"


def test_agent_signature_model_used_overrides_config(tmp_path):
    from coding_review_agent_loop.agents.registry import agent_signature
    config = make_config(tmp_path, antigravity_model="Gemini 3.1 Pro (High)")
    # #333 fallback: the model that actually ran wins over the configured one.
    assert (
        agent_signature("antigravity", config, "Gemini 3.5 Flash (High)")
        == "Google Antigravity: Gemini 3.5 Flash (High)"
    )


def test_config_rejects_model_arg_conflicts(tmp_path):
    for kwargs in (
        {"codex_model": "gpt-5", "codex_args": ("--model", "other")},
        {"codex_reasoning_effort": "high", "codex_args": ("-c", 'model_reasoning_effort="low"')},
        {"gemini_model": "g", "gemini_args": ("--model", "other")},
        {"claude_model": "c", "claude_args": ("--model", "other")},
        {"antigravity_args": ("--model", "x")},
    ):
        with pytest.raises(AgentLoopError, match="conflicts with"):
            make_config(tmp_path, **kwargs)


def test_config_rejects_codex_effort_without_model(tmp_path):
    # Rollout model detection is best-effort, so effort alone cannot be labeled
    # reliably and requires an explicit --codex-model.
    with pytest.raises(AgentLoopError, match="requires --codex-model"):
        make_config(tmp_path, codex_reasoning_effort="high")
    # With a model it's accepted.
    config = make_config(tmp_path, codex_model="gpt-5", codex_reasoning_effort="high")
    assert config.codex_reasoning_effort == "high"


def test_config_allows_declared_model_without_conflict(tmp_path):
    config = make_config(tmp_path, codex_model="gpt-5", gemini_model="g", claude_model="c")
    assert config.codex_model == "gpt-5"
    assert config.gemini_model == "g"
    assert config.claude_model == "c"


def test_codex_backend_passes_model_and_effort(tmp_path):
    from coding_review_agent_loop.agents.codex import CodexBackend
    runner = FakeRunner(codex_outputs=[("STATE: approved\n\nok", 0)])
    config = make_config(tmp_path, codex_model="gpt-5.2-codex", codex_reasoning_effort="high")
    result = CodexBackend().run(runner, config, "Review", run_id="r")
    cmd = runner.commands[-1][0]
    assert cmd[cmd.index("--model") + 1] == "gpt-5.2-codex"
    assert 'model_reasoning_effort="high"' in cmd
    assert result.model_used == "gpt-5.2-codex (high)"


def test_gemini_backend_passes_model_and_sets_model_used(tmp_path):
    import coding_review_agent_loop.agents.gemini as gm
    runner = FakeRunner(gemini_outputs=[("STATE: approved\n\nok", 0)])
    config = make_config(tmp_path, gemini_model="gemini-3.5-flash")
    result = gm.BACKEND.run(runner, config, "Review", run_id="r")
    cmd = runner.commands[-1][0]
    assert cmd[cmd.index("--model") + 1] == "gemini-3.5-flash"
    assert result.model_used == "gemini-3.5-flash"


def test_claude_backend_passes_model_when_declared(tmp_path):
    from coding_review_agent_loop.agents.claude import ClaudeBackend
    runner = FakeRunner(claude_outputs=[("STATE: approved\n\nok", 0)])
    config = make_config(tmp_path, claude_model="opus")
    ClaudeBackend().run(runner, config, "Review", run_id="r")
    cmd = runner.commands[-1][0]
    assert cmd[cmd.index("--model") + 1] == "opus"


def test_public_reviewer_name_config_aware_no_leakage(tmp_path):
    from coding_review_agent_loop.comment_rendering import _public_reviewer_name
    config = make_config(tmp_path, codex_model="gpt-5", antigravity_model="Gemini 3.1 Pro (High)")
    assert _public_reviewer_name("Codex", config) == "OpenAI Codex: gpt-5"
    assert _public_reviewer_name("Antigravity", config) == "Google Antigravity: Gemini 3.1 Pro (High)"
    # No declared model → generic; unknown display name → passthrough.
    assert _public_reviewer_name("Claude", config) == "Anthropic Claude"
    assert _public_reviewer_name("Codex") == "OpenAI Codex"
    assert _public_reviewer_name("Somebody") == "Somebody"


def test_render_public_agent_comment_stamps_model_for_every_kind():
    model = "Gemini 3.1 Pro (High)"

    pr_review = parse_pr_review(
        structured_pr_review(state="approved", reviewer="Google Antigravity"),
        reviewer="Google Antigravity",
    )
    plan_review = parse_plan_review(
        structured_plan_review(state="approved", reviewer="Google Antigravity"),
        reviewer="Google Antigravity",
    )
    coder_followup = validate_structured_coder_followup(
        structured_coder_followup(state="approved", reviewer="Google Antigravity")
    )
    plan_revision = validate_structured_plan_revision(
        structured_plan_revision(reviewer="Google Antigravity")
    )
    assert coder_followup is not None
    assert plan_revision is not None

    rendered = [
        render_public_agent_comment(
            kind="pr_review",
            parsed=pr_review,
            agent="Antigravity",
            dispositions=pr_review.dispositions,
            model_used=model,
        ),
        render_public_agent_comment(
            kind="plan_review",
            parsed=plan_review,
            agent="Antigravity",
            dispositions=plan_review.dispositions,
            model_used=model,
        ),
        render_public_agent_comment(
            kind="coder_followup",
            parsed=coder_followup,
            agent="antigravity",
            model_used=model,
        ),
        render_public_agent_comment(
            kind="plan_revision",
            parsed=plan_revision,
            agent="antigravity",
            raw_text=structured_plan_revision(reviewer="Google Antigravity"),
            model_used=model,
        ),
    ]

    assert all(comment.endswith(f"-- Google Antigravity: {model}") for comment in rendered)


# ---------------------------------------------------------------------------
# Antigravity prompt — turn-end requirement (#385)
# ---------------------------------------------------------------------------


def test_antigravity_prompt_includes_terminal_response_instruction():
    from coding_review_agent_loop.agents.antigravity import _with_public_response_marker_instruction
    composed = _with_public_response_marker_instruction("BASE PROMPT")
    assert "end your turn immediately" in composed
    assert "do not defer to a background task result" in composed


def test_antigravity_prompt_excludes_old_wait_instruction():
    from coding_review_agent_loop.agents.antigravity import _with_public_response_marker_instruction
    composed = _with_public_response_marker_instruction("BASE PROMPT")
    assert "Do not print the marker until you are done with all internal reasoning" not in composed


def test_base_response_file_instruction_includes_must_write_before_turn_ends(tmp_path):
    from coding_review_agent_loop.agents.base import with_public_response_file_instruction
    composed = with_public_response_file_instruction("BASE PROMPT", tmp_path / "response.md")
    assert "before your turn ends" in composed


# ── Tests: issue #400 – toolPermission: "strict" injection for reviewer ────────


def test_antigravity_backend_injects_strict_mode_for_reviewer(tmp_path, monkeypatch):
    """Reviewer run injects toolPermission:strict and expected allow-list while running."""
    import json as _json
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import (
        AntigravityBackend,
        _REVIEWER_SETTINGS_INJECTION,
    )

    agy_dir = tmp_path / "agy"
    agy_dir.mkdir(parents=True)
    settings_file = tmp_path / "settings.json"
    original = _json.dumps({"existingKey": "existingValue"})
    settings_file.write_text(original, encoding="utf-8")
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)

    captured_settings: list[dict] = []

    class CapturingRunner(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            captured_settings.append(
                _json.loads(settings_file.read_text(encoding="utf-8"))
            )
            return super().run_with_log(args, cwd=cwd, **kwargs)

    config = make_config(tmp_path, antigravity_dir=agy_dir)
    AntigravityBackend().run(
        CapturingRunner(antigravity_outputs=[("ok", 0)]),
        config, "Review PR.", role="reviewer",
    )

    assert len(captured_settings) == 1
    settings = captured_settings[0]
    assert settings["toolPermission"] == "strict"
    assert settings["permissions"] == _REVIEWER_SETTINGS_INJECTION["permissions"]
    assert settings["existingKey"] == "existingValue"


def test_antigravity_backend_restores_original_settings_after_reviewer_run(tmp_path, monkeypatch):
    """Settings file is verbatim-restored after a successful reviewer run."""
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "agy"
    agy_dir.mkdir(parents=True)
    settings_file = tmp_path / "settings.json"
    original_text = '{"key": "val", "nested": {"a": 1}}'
    settings_file.write_text(original_text, encoding="utf-8")
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)

    config = make_config(tmp_path, antigravity_dir=agy_dir)
    AntigravityBackend().run(
        FakeRunner(antigravity_outputs=[("ok", 0)]),
        config, "Review.", role="reviewer",
    )

    assert settings_file.read_text(encoding="utf-8") == original_text


def test_antigravity_backend_restores_settings_on_run_exception(tmp_path, monkeypatch):
    """Settings file is verbatim-restored even when the runner raises."""
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "agy"
    agy_dir.mkdir(parents=True)
    settings_file = tmp_path / "settings.json"
    original_text = '{"keep": "me"}'
    settings_file.write_text(original_text, encoding="utf-8")
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)

    class RaisingRunner(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            raise RuntimeError("agy crashed")

    config = make_config(tmp_path, antigravity_dir=agy_dir)
    with pytest.raises(RuntimeError, match="agy crashed"):
        AntigravityBackend().run(RaisingRunner(), config, "Review.", role="reviewer")

    assert settings_file.read_text(encoding="utf-8") == original_text


def test_antigravity_backend_restores_settings_on_injection_write_failure(tmp_path, monkeypatch):
    """If the injection write_text partially truncates then raises, original bytes are restored."""
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "agy"
    agy_dir.mkdir(parents=True)
    settings_file = tmp_path / "settings.json"
    original_text = '{"preserve": "exactly"}'
    settings_file.write_text(original_text, encoding="utf-8")
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)

    write_calls = [0]
    real_write_text = Path.write_text

    def patched_write_text(self, *args, **kwargs):
        if self == settings_file:
            write_calls[0] += 1
            if write_calls[0] == 1:
                real_write_text(self, "PARTIAL")
                raise OSError("simulated disk full")
        real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", patched_write_text)

    config = make_config(tmp_path, antigravity_dir=agy_dir)
    with pytest.raises(OSError, match="simulated disk full"):
        AntigravityBackend().run(
            FakeRunner(antigravity_outputs=[("ok", 0)]), config, "Review.", role="reviewer",
        )

    assert settings_file.read_text(encoding="utf-8") == original_text


def test_antigravity_backend_does_not_touch_settings_for_non_reviewer(tmp_path, monkeypatch):
    """Non-reviewer run holds the settings lock but does not read or modify settings.json."""
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "agy"
    agy_dir.mkdir(parents=True)
    settings_file = tmp_path / "settings.json"
    original_text = '{"untouched": true}'
    settings_file.write_text(original_text, encoding="utf-8")
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)

    captured_during: list[str] = []

    class CapturingRunner(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            captured_during.append(settings_file.read_text(encoding="utf-8"))
            return super().run_with_log(args, cwd=cwd, **kwargs)

    config = make_config(tmp_path, antigravity_dir=agy_dir)
    AntigravityBackend().run(
        CapturingRunner(antigravity_outputs=[("ok", 0)]),
        config, "Implement.", role=None,
    )

    assert captured_during[0] == original_text
    assert settings_file.read_text(encoding="utf-8") == original_text


def test_antigravity_backend_fails_fast_on_malformed_settings_json(tmp_path, monkeypatch):
    """AgentLoopError raised before any run if settings.json is not valid JSON."""
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend
    from coding_review_agent_loop.errors import AgentLoopError

    agy_dir = tmp_path / "agy"
    agy_dir.mkdir(parents=True)
    settings_file = tmp_path / "settings.json"
    settings_file.write_text("not json at all {{{", encoding="utf-8")
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)

    config = make_config(tmp_path, antigravity_dir=agy_dir)
    with pytest.raises(AgentLoopError, match="settings.json malformed"):
        AntigravityBackend().run(
            FakeRunner(antigravity_outputs=[("ok", 0)]), config, "Review.", role="reviewer",
        )


@pytest.mark.parametrize("invalid", ["[]", "null", "42"])
def test_antigravity_backend_fails_fast_on_non_object_settings_json(
    tmp_path, monkeypatch, invalid
):
    """AgentLoopError raised if settings.json root is not a JSON object."""
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend
    from coding_review_agent_loop.errors import AgentLoopError

    agy_dir = tmp_path / "agy"
    agy_dir.mkdir(parents=True)
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(invalid, encoding="utf-8")
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)

    config = make_config(tmp_path, antigravity_dir=agy_dir)
    with pytest.raises(AgentLoopError, match="not a JSON object"):
        AntigravityBackend().run(
            FakeRunner(antigravity_outputs=[("ok", 0)]), config, "Review.", role="reviewer",
        )


def test_antigravity_backend_strips_dangerously_skip_permissions_for_reviewer(
    tmp_path, monkeypatch
):
    """--dangerously-skip-permissions is removed from args when role=reviewer."""
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "agy"
    agy_dir.mkdir(parents=True)
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)

    captured_args: list[list[str]] = []

    class CapturingRunner(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            captured_args.append(list(args))
            return super().run_with_log(args, cwd=cwd, **kwargs)

    config = make_config(
        tmp_path,
        antigravity_dir=agy_dir,
        antigravity_args=("--dangerously-skip-permissions",),
    )
    AntigravityBackend().run(
        CapturingRunner(antigravity_outputs=[("ok", 0)]),
        config, "Review.", role="reviewer",
    )
    assert "--dangerously-skip-permissions" not in captured_args[-1]

    # Non-reviewer run keeps the flag
    captured_args.clear()
    AntigravityBackend().run(
        CapturingRunner(antigravity_outputs=[("ok", 0)]),
        config, "Implement.", role=None,
    )
    assert "--dangerously-skip-permissions" in captured_args[-1]


def test_antigravity_backend_lock_order_settings_outer_gemini_inner(tmp_path, monkeypatch):
    """Settings lock (outer) is acquired before GEMINI.md lock (inner);
    settings are restored before the settings lock is released."""
    import fcntl as fcntl_mod
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "agy"
    agy_dir.mkdir(parents=True)
    (agy_dir / ".git").mkdir(parents=True, exist_ok=True)
    settings_file = tmp_path / "settings.json"
    original_text = '{"v": 1}'
    settings_file.write_text(original_text, encoding="utf-8")
    settings_lock_path = str(settings_file.with_suffix(".json.lock"))
    gemini_lock_path = str(agy_dir / ".git" / "GEMINI.md.lock")
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)

    real_flock = fcntl_mod.flock
    operations: list[tuple[str, int]] = []
    settings_content_at_unlock: list[str | None] = [None]

    def tracking_flock(fd, operation):
        name = str(getattr(fd, "name", ""))
        if settings_lock_path in name:
            label = "settings"
        elif gemini_lock_path in name:
            label = "gemini"
        else:
            label = "other"
        if label == "settings" and operation == fcntl_mod.LOCK_UN:
            settings_content_at_unlock[0] = settings_file.read_text(encoding="utf-8")
        operations.append((label, operation))
        real_flock(fd, operation)

    monkeypatch.setattr(fcntl_mod, "flock", tracking_flock)

    config = make_config(tmp_path, antigravity_dir=agy_dir)
    AntigravityBackend().run(
        FakeRunner(antigravity_outputs=[("ok", 0)]),
        config, "Review.", role="reviewer",
    )

    relevant = [(label, op) for label, op in operations if label in ("settings", "gemini")]
    assert relevant == [
        ("settings", fcntl_mod.LOCK_EX),
        ("gemini", fcntl_mod.LOCK_EX),
        ("gemini", fcntl_mod.LOCK_UN),
        ("settings", fcntl_mod.LOCK_UN),
    ]
    assert settings_content_at_unlock[0] == original_text


def test_antigravity_settings_lock_serializes_reviewer_vs_reviewer(tmp_path, monkeypatch):
    """Two concurrent reviewer runs are serialized: runner B starts only after A completes."""
    import threading
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "agy"
    agy_dir.mkdir(parents=True)
    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"k": "v"}', encoding="utf-8")
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)

    config = make_config(tmp_path, antigravity_dir=agy_dir)
    order: list[str] = []
    a_in_runner = threading.Event()
    a_release = threading.Event()
    b_in_runner = threading.Event()

    class RunnerA(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            order.append("a_in_runner")
            a_in_runner.set()
            a_release.wait(timeout=10)
            return super().run_with_log(args, cwd=cwd, **kwargs)

    class RunnerB(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            order.append("b_in_runner")
            b_in_runner.set()
            return super().run_with_log(args, cwd=cwd, **kwargs)

    def run_a():
        AntigravityBackend().run(RunnerA(antigravity_outputs=[("ok", 0)]), config, "PromptA", role="reviewer")

    def run_b():
        AntigravityBackend().run(RunnerB(antigravity_outputs=[("ok", 0)]), config, "PromptB", role="reviewer")

    ta = threading.Thread(target=run_a, daemon=True)
    ta.start()
    a_in_runner.wait(timeout=10)

    tb = threading.Thread(target=run_b, daemon=True)
    tb.start()

    a_release.set()
    b_in_runner.wait(timeout=10)

    ta.join(timeout=10)
    tb.join(timeout=10)

    assert not ta.is_alive(), "Thread A deadlocked"
    assert not tb.is_alive(), "Thread B deadlocked"
    assert order == ["a_in_runner", "b_in_runner"]
    assert settings_file.read_text(encoding="utf-8") == '{"k": "v"}'


def test_antigravity_settings_lock_serializes_reviewer_vs_coder_both_orders(
    tmp_path, monkeypatch
):
    """Reviewer-coder and coder-reviewer serialization; LOCK_NB on the contender backend's
    settings-lock acquisition confirms the holder is actively holding the lock."""
    import fcntl as fcntl_mod
    import threading
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "agy"
    agy_dir.mkdir(parents=True)
    settings_file = tmp_path / "settings.json"
    settings_lock_path_str = str(settings_file.with_suffix(".json.lock"))
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)

    real_flock = fcntl_mod.flock
    _is_contender: threading.local = threading.local()
    _probe_done: threading.local = threading.local()

    for holder_role, contender_role in [("reviewer", None), (None, "reviewer")]:
        settings_file.write_text('{"initial": true}', encoding="utf-8")

        holder_in_runner = threading.Event()
        release_holder = threading.Event()
        contender_attempting_lock = threading.Event()
        contender_probe_done = threading.Event()
        contender_in_runner = threading.Event()
        nb_probe_results: list = []

        def instrumented_flock(fd, operation,
                               _slp=settings_lock_path_str,
                               _cat=contender_attempting_lock,
                               _cpd=contender_probe_done,
                               _nbr=nb_probe_results):
            if (
                getattr(_is_contender, "value", False)
                and _slp in str(getattr(fd, "name", ""))
                and operation == fcntl_mod.LOCK_EX
                and not getattr(_probe_done, "done", False)
            ):
                _probe_done.done = True
                _cat.set()
                try:
                    real_flock(fd, fcntl_mod.LOCK_EX | fcntl_mod.LOCK_NB)
                    _nbr.append(None)
                except BlockingIOError as exc:
                    _nbr.append(exc)
                _cpd.set()
                real_flock(fd, fcntl_mod.LOCK_EX)
                return
            real_flock(fd, operation)

        monkeypatch.setattr(fcntl_mod, "flock", instrumented_flock)

        config = make_config(tmp_path, antigravity_dir=agy_dir)

        class HolderRunner(FakeRunner):
            def run_with_log(self, args, *, cwd, **kwargs):
                holder_in_runner.set()
                release_holder.wait(timeout=10)
                return super().run_with_log(args, cwd=cwd, **kwargs)

        class ContenderRunner(FakeRunner):
            def run_with_log(self, args, *, cwd, **kwargs):
                contender_in_runner.set()
                return super().run_with_log(args, cwd=cwd, **kwargs)

        def run_holder(role=holder_role):
            AntigravityBackend().run(
                HolderRunner(antigravity_outputs=[("ok", 0)]),
                config, "Holder", role=role,
            )

        def run_contender(role=contender_role):
            _is_contender.value = True
            _probe_done.done = False
            AntigravityBackend().run(
                ContenderRunner(antigravity_outputs=[("ok", 0)]),
                config, "Contender", role=role,
            )

        th = threading.Thread(target=run_holder, daemon=True)
        th.start()
        holder_in_runner.wait(timeout=10)

        tc = threading.Thread(target=run_contender, daemon=True)
        tc.start()

        contender_attempting_lock.wait(timeout=10)
        contender_probe_done.wait(timeout=10)
        release_holder.set()
        contender_in_runner.wait(timeout=10)

        th.join(timeout=10)
        tc.join(timeout=10)

        assert not th.is_alive(), f"Holder deadlocked (holder_role={holder_role!r})"
        assert not tc.is_alive(), f"Contender deadlocked (contender_role={contender_role!r})"
        assert contender_in_runner.is_set()
        assert len(nb_probe_results) == 1
        assert isinstance(nb_probe_results[0], BlockingIOError), (
            f"LOCK_NB probe should have failed with BlockingIOError "
            f"but got {nb_probe_results[0]!r} "
            f"(holder_role={holder_role!r}, contender_role={contender_role!r})"
        )


def test_antigravity_settings_lock_restoration_precedes_unlock_on_exception(
    tmp_path, monkeypatch
):
    """Settings are restored before the settings lock is released, even when runner raises.

    Thread A: reviewer run; runner signals `injected_event` (settings injected, lock held),
    waits for `contention_confirmed_event`, then raises RuntimeError.
    Thread B: raw-lock contender; LOCK_NB proves A holds the lock, then blocks on LOCK_EX.
    After A's exception path restores settings and releases the lock, B unblocks and
    reads the settings file — which must already be restored to the original content.
    """
    import fcntl as fcntl_mod
    import threading
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "agy"
    agy_dir.mkdir(parents=True)
    settings_file = tmp_path / "settings.json"
    settings_lock_path = settings_file.with_suffix(".json.lock")
    original_text = '{"preserve": "this"}'
    settings_file.write_text(original_text, encoding="utf-8")
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)

    injected_event = threading.Event()
    contention_confirmed_event = threading.Event()
    thread_a_exc: list[BaseException | None] = [None]
    thread_b_result: list[str | None] = [None]
    thread_b_nb_error: list = [None]

    class InjectingRunner(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            injected_event.set()
            contention_confirmed_event.wait(timeout=10)
            raise RuntimeError("simulated reviewer failure")

    def run_a():
        try:
            AntigravityBackend().run(
                InjectingRunner(),
                make_config(tmp_path, antigravity_dir=agy_dir),
                "Review.", role="reviewer",
            )
        except RuntimeError as exc:
            thread_a_exc[0] = exc

    def run_b():
        lock_f = settings_lock_path.open("a+")
        try:
            try:
                fcntl_mod.flock(lock_f, fcntl_mod.LOCK_EX | fcntl_mod.LOCK_NB)
                thread_b_nb_error[0] = None
            except BlockingIOError as exc:
                thread_b_nb_error[0] = exc
            contention_confirmed_event.set()
            fcntl_mod.flock(lock_f, fcntl_mod.LOCK_EX)
            thread_b_result[0] = settings_file.read_text(encoding="utf-8")
        finally:
            fcntl_mod.flock(lock_f, fcntl_mod.LOCK_UN)
            lock_f.close()

    ta = threading.Thread(target=run_a, daemon=True)
    ta.start()
    injected_event.wait(timeout=10)

    tb = threading.Thread(target=run_b, daemon=True)
    tb.start()

    ta.join(timeout=10)
    tb.join(timeout=10)

    assert not ta.is_alive(), "Thread A deadlocked"
    assert not tb.is_alive(), "Thread B deadlocked"
    assert isinstance(thread_a_exc[0], RuntimeError)
    assert isinstance(thread_b_nb_error[0], BlockingIOError), (
        f"LOCK_NB should have raised BlockingIOError; got {thread_b_nb_error[0]!r}"
    )
    assert thread_b_result[0] == original_text, (
        f"Settings must be restored before the lock is released; got {thread_b_result[0]!r}"
    )


def test_reviewer_and_coder_call_sites_pass_correct_role(tmp_path):
    """_run_validated_agent propagates role= to run_agent_result correctly."""
    from unittest.mock import patch
    from coding_review_agent_loop.agents.base import AgentResult

    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)
    captured_roles: list = []

    def mock_run(runner, *, agent, config, prompt, session_id=None, run_id=None, role=None):
        captured_roles.append(role)
        return AgentResult(text="ok")

    with patch("coding_review_agent_loop.orchestrator.run_agent_result", mock_run):
        _run_validated_agent(
            FakeRunner(),
            agent="gemini",
            config=config,
            prompt="Review the plan.",
            marker_description="test",
            validate=lambda text: text,
            role="reviewer",
        )

    assert captured_roles == ["reviewer"]
    captured_roles.clear()

    with patch("coding_review_agent_loop.orchestrator.run_agent_result", mock_run):
        _run_validated_agent(
            FakeRunner(),
            agent="gemini",
            config=config,
            prompt="Implement.",
            marker_description="test",
            validate=lambda text: text,
        )

    assert captured_roles == [None]


def test_run_agent_result_passes_role_to_backend(tmp_path, monkeypatch):
    """run_agent_result threads role= through to the backend's run() method."""
    from coding_review_agent_loop.agents.registry import run_agent_result
    from coding_review_agent_loop.agents.base import AgentResult
    from coding_review_agent_loop.agents import registry as reg_mod

    captured: dict = {}

    class TrackingBackend:
        name = "gemini"
        display_name = "Gemini"
        signature = "Google Gemini"

        def workdir(self, config):
            return tmp_path

        def default_args(self, *, dangerous):
            return ()

        def run(self, runner, config, prompt, session_id=None, run_id=None, role=None):
            captured["role"] = role
            return AgentResult(text="ok")

    monkeypatch.setitem(reg_mod.BACKENDS, "gemini", TrackingBackend())
    config = make_config(tmp_path, reviewer="gemini")
    run_agent_result(FakeRunner(), agent="gemini", config=config, prompt="Test", role="reviewer")
    assert captured["role"] == "reviewer"
