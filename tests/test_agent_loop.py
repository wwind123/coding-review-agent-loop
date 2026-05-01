import json
from pathlib import Path

import pytest

from coding_review_agent_loop.agents.claude import _parse_claude_output
from coding_review_agent_loop.agents.gemini import PUBLIC_RESPONSE_MARKER, _parse_gemini_output
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
from coding_review_agent_loop.config import (
    default_agent_memory_dir,
    default_agent_workdir,
    default_cache_root,
)
from coding_review_agent_loop.protocol import parse_non_blocking_followups


class FakeRunner(Runner):
    def __init__(
        self,
        *,
        claude_outputs=None,
        codex_outputs=None,
        gemini_outputs=None,
        issue_payload=None,
        pr_payload=None,
        git_status="",
        git_remote="git@github.com:OWNER/REPO.git",
        git_inside=True,
        git_head="abc123",
        tracked_files=None,
        changed_files=None,
        diff_returncode=0,
        diff_stderr="",
    ):
        super().__init__(dry_run=False)
        self.claude_outputs = list(claude_outputs or [])
        self.codex_outputs = list(codex_outputs or [])
        self.gemini_outputs = list(gemini_outputs or [])
        self.issue_payload = issue_payload or {
            "number": 56,
            "state": "open",
            "is_pr": False,
            "url": "https://github.com/OWNER/REPO/issues/56",
        }
        self.pr_payload = pr_payload or {
            "number": 77,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/77",
            "title": "Improve review prompt context",
            "headRefName": "feature/review-context",
            "baseRefName": "main",
            "headRefOid": "abc123",
        }
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

    def _record_command(self, args, cwd):
        cmd = [str(arg) for arg in args]
        cwd_path = Path(cwd)
        if not cwd_path.is_dir():
            raise FileNotFoundError(cwd_path)
        self.commands.append((cmd, cwd_path))
        return cmd, cwd_path

    def run_with_log(
        self,
        args,
        *,
        cwd,
        log_path,
        label,
        progress_interval_seconds,
        check=True,
    ):
        cmd, cwd_path = self._record_command(args, cwd)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ensure_log_dir_ignored(log_path.parent)

        if cmd[:1] == ["claude"]:
            output = self.claude_outputs.pop(0)
            log_path.write_text(f"$ {' '.join(cmd)}\n\n{output}", encoding="utf-8")
            return CommandResult(cmd, cwd_path, output, "", 0)

        if cmd[:2] == ["codex", "exec"]:
            output = self.codex_outputs.pop(0)
            if "--output-last-message" in cmd:
                out_path = Path(cmd[cmd.index("--output-last-message") + 1])
                out_path.write_text(output, encoding="utf-8")
            log_path.write_text(f"$ {' '.join(cmd)}\n\ncodex completed", encoding="utf-8")
            return CommandResult(cmd, cwd_path, "codex completed", "", 0)

        if cmd[:1] == ["gemini"]:
            output = self.gemini_outputs.pop(0)
            log_path.write_text(f"$ {' '.join(cmd)}\n\n{output}", encoding="utf-8")
            return CommandResult(cmd, cwd_path, output, "", 0)

        return self.run(args, cwd=cwd, check=check)

    def run(self, args, *, cwd, input_text=None, check=True):
        cmd, cwd_path = self._record_command(args, cwd)

        if cmd[:1] == ["claude"]:
            return CommandResult(cmd, cwd_path, self.claude_outputs.pop(0), "", 0)

        if cmd[:2] == ["codex", "exec"]:
            output = self.codex_outputs.pop(0)
            if "--output-last-message" in cmd:
                out_path = Path(cmd[cmd.index("--output-last-message") + 1])
                out_path.write_text(output, encoding="utf-8")
            return CommandResult(cmd, cwd_path, "", "", 0)

        if cmd[:3] == ["gh", "pr", "comment"]:
            if "--body-file" in cmd:
                body_path = Path(cmd[cmd.index("--body-file") + 1])
                self.comments.append(body_path.read_text(encoding="utf-8"))
            elif "--body" in cmd:
                self.comments.append(cmd[cmd.index("--body") + 1])
            return CommandResult(cmd, cwd_path, "", "", 0)

        if cmd[:3] == ["gh", "issue", "create"]:
            title = cmd[cmd.index("--title") + 1]
            if "--body-file" in cmd:
                body_path = Path(cmd[cmd.index("--body-file") + 1])
                body = body_path.read_text(encoding="utf-8")
            else:
                body = cmd[cmd.index("--body") + 1]
            self.issues.append({"title": title, "body": body})
            return CommandResult(cmd, cwd_path, "https://github.com/OWNER/REPO/issues/99\n", "", 0)

        if cmd[:3] == ["gh", "pr", "view"]:
            if "--jq" in cmd and ".headRefOid" in cmd:
                return CommandResult(cmd, cwd_path, "abc123\n", "", 0)
            return CommandResult(cmd, cwd_path, json_dumps(self.pr_payload), "", 0)

        if cmd[:2] == ["gh", "api"] and "/issues/" in cmd[2]:
            return CommandResult(cmd, cwd_path, json_dumps(self.issue_payload), "", 0)

        if cmd[:2] == ["gh", "api"] and cmd[2].endswith("/check-runs"):
            return CommandResult(cmd, cwd_path, "success\n", "", 0)

        if cmd[:1] == ["sleep"]:
            return CommandResult(cmd, cwd_path, "", "", 0)

        if cmd[:3] == ["git", "rev-parse", "--is-inside-work-tree"]:
            if self.git_inside:
                return CommandResult(cmd, cwd_path, "true\n", "", 0)
            return CommandResult(cmd, cwd_path, "false\n", "", 1)

        if cmd[:3] == ["git", "rev-parse", "HEAD"]:
            return CommandResult(cmd, cwd_path, f"{self.git_head}\n", "", 0)

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
        "ci_check_name": "test",
        "ci_timeout_seconds": 1200,
        "ci_poll_interval_seconds": 30,
        "quiet": True,
        "log_dir": tmp_path / "logs",
        "progress_interval_seconds": 30,
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


def test_parse_claude_output_extracts_text_and_session_id():
    raw = json.dumps({"result": "Hello.", "session_id": "abc123"})
    text, sid = _parse_claude_output(raw)
    assert text == "Hello."
    assert sid == "abc123"


def test_parse_claude_output_falls_back_on_plain_text():
    raw = "plain response"
    text, sid = _parse_claude_output(raw)
    assert text == "plain response"
    assert sid is None


def test_parse_claude_output_falls_back_on_non_string_result():
    raw = json.dumps({"result": 42, "session_id": "abc"})
    text, sid = _parse_claude_output(raw)
    assert text == raw  # non-string result → fall back to raw
    assert sid == "abc"


def test_parse_gemini_output_extracts_json_response():
    raw = json.dumps({
        "response": "Reviewed.\n<!-- AGENT_STATE: approved -->",
        "session_id": "gemini-session-1",
    })
    text, sid = _parse_gemini_output(raw)
    assert text == "Reviewed.\n<!-- AGENT_STATE: approved -->"
    assert sid == "gemini-session-1"


def test_parse_gemini_output_falls_back_on_plain_text():
    text, sid = _parse_gemini_output("plain response")
    assert text == "plain response"
    assert sid is None


def test_parse_gemini_output_falls_back_on_non_string_response():
    raw = json.dumps({"response": 42, "session_id": "gemini-session-1"})
    text, sid = _parse_gemini_output(raw)
    assert text == raw
    assert sid == "gemini-session-1"


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
    text, sid = _parse_gemini_output(raw)
    assert text.startswith("## Review")
    assert "True color" not in text
    assert "YOLO mode" not in text
    assert "I will inspect" not in text
    assert "Error executing tool" not in text
    assert "<!-- AGENT_STATE: approved -->" in text
    assert sid is None


def test_parse_gemini_output_uses_last_public_response_marker():
    raw = f"""Gemini may mention {PUBLIC_RESPONSE_MARKER} while planning.
{PUBLIC_RESPONSE_MARKER}
intermediate draft
{PUBLIC_RESPONSE_MARKER}
Final answer.
<!-- AGENT_STATE: approved -->
"""
    text, _sid = _parse_gemini_output(raw)
    assert text == "Final answer.\n<!-- AGENT_STATE: approved -->\n"


def test_parse_gemini_json_response_strips_public_response_marker():
    raw = json.dumps({
        "response": f"diagnostic\n{PUBLIC_RESPONSE_MARKER}\nReviewed.\n<!-- AGENT_STATE: approved -->",
        "session_id": "gemini-session-1",
    })
    text, sid = _parse_gemini_output(raw)
    assert text == "Reviewed.\n<!-- AGENT_STATE: approved -->"
    assert sid == "gemini-session-1"


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
    text, sid = _parse_gemini_output(raw)
    assert text.startswith("## Code Review")
    assert "_GaxiosError" not in text
    assert "YOLO mode" not in text
    assert "<!-- AGENT_STATE: approved -->" in text
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
    text, sid = _parse_gemini_output(raw)
    assert text.startswith("## Summary")
    assert "YOLO mode" not in text
    assert "## Details" in text
    assert "\n---\n\n## Details" in text
    assert sid is None


def test_parse_gemini_output_strips_preamble_before_clarification_marker():
    raw = """Warning: True color (24-bit) support not detected.
I need to ask a question.

---

    Which endpoint should I update?
<!-- AGENT_CLARIFY -->
"""
    text, sid = _parse_gemini_output(raw)
    assert text.startswith("    Which endpoint")
    assert "True color" not in text
    assert "<!-- AGENT_CLARIFY -->" in text
    assert sid is None


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


def test_issue_loop_creates_pr_then_alternates_until_codex_approval(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->",
            "Fixed review.\n<!-- AGENT_STATE: blocking -->",
        ],
        codex_outputs=[
            "Finding: bug remains.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    command_names = [cmd[:2] for cmd, _cwd in runner.commands]
    assert ["claude", "--print"] in command_names
    assert ["codex", "exec"] in command_names
    assert len(runner.comments) == 4
    assert runner.comments[-1].startswith("LGTM.")
    assert list((tmp_path / "logs").glob("*-claude.log"))
    assert list((tmp_path / "logs").glob("*-codex.log"))
    assert (tmp_path / "logs" / ".gitignore").read_text(encoding="utf-8") == "*\n!.gitignore\n"


def test_issue_loop_can_use_codex_as_coder_and_claude_as_reviewer(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->",
            "Fixed review.\n<!-- AGENT_STATE: blocking -->",
        ],
        claude_outputs=[
            "Finding: bug remains.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
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
    assert runner.comments[-1].startswith("LGTM.")


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
        "--jq",
        '[.check_runs[] | select(.name == "test")] | if length == 0 then "pending" else .[0].conclusion // .[0].status end',
    ] in commands
    assert ["gh", "pr", "merge", "77", "--repo", "OWNER/REPO", "--merge"] in commands


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
    assert "### Non-blocking follow-ups" in prompt
    assert "Use blocking only for issues that should prevent merge." in prompt


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
    assert diff_commands == [["git", "diff", "--name-only", "abc123..def456"]]
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
        and cmd[cmd.index("--json") + 1] == "number,title,headRefName,baseRefName,headRefOid,url"
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
        "LGTM.\n\n### Non-blocking follow-ups\n- Add cleanup docs.\n"
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
    assert summary.startswith("Approved-review non-blocking follow-ups for PR #77:")
    assert "- Add cleanup docs. (Codex)" in summary
    assert "- Add regression coverage. (Claude)" in summary
    assert "did not block merge readiness" in summary


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

    assert len(runner.comments) == 2
    assert runner.issues == [
        {
            "title": "Follow up approved review note: Add cleanup docs.",
            "body": (
                "Non-blocking follow-up from approved review on PR #77.\n\n"
                "Reviewer: Codex\n\n"
                "Follow-up:\n"
                "- Add cleanup docs.\n\n"
                "This was mentioned in an approved review and did not block merge readiness.\n\n"
                "-- OpenAI Codex"
            ),
        },
        {
            "title": "Follow up approved review note: Add regression coverage.",
            "body": (
                "Non-blocking follow-up from approved review on PR #77.\n\n"
                "Reviewer: Claude\n\n"
                "Follow-up:\n"
                "- Add regression coverage.\n\n"
                "This was mentioned in an approved review and did not block merge readiness.\n\n"
                "-- OpenAI Codex"
            ),
        },
    ]


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
        "Follow up approved review note: Follow up one.",
        "Follow up approved review note: Follow up two.",
        "Follow up approved review note: Follow up three.",
    ]
    assert len(runner.comments) == 2
    assert "Skipped 1 additional item(s) to avoid issue noise" in runner.comments[-1]
    assert runner.comments[-1].endswith("-- OpenAI Codex")


def test_pr_loop_reruns_all_reviewers_when_any_reviewer_blocks(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Needs a regression test.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Addressed review.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Claude approves.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Codex approves first pass.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "Codex approves second pass.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
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
        and cmd[cmd.index("--json") + 1] == "number,title,headRefName,baseRefName,headRefOid,url"
    ]
    assert len(metadata_fetches) == 2


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
    assert set(github_or_test_cwds) == {config.claude_dir}


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
    assert set(config.auto_agent_dirs) == {"claude", "codex", "gemini"}
    assert config.agent_memory_dir == (
        cache_home / "coding-review-agent-loop" / "repos" / "OWNER-REPO" / "memory"
    ).resolve()


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


@pytest.mark.parametrize("mode", ["ignore", "summarize", "issue"])
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
    assert set(config.auto_agent_dirs) == {"claude", "gemini"}


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
    assert ["git", "checkout", "main"] in commands
    assert ["git", "pull", "--ff-only", "origin", "main"] in commands


def test_dirty_existing_auto_agent_dir_fails_clearly(tmp_path):
    runner = FakeRunner(git_status=" M file.py\n")
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

    with pytest.raises(AgentLoopError, match="dirty"):
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


def test_is_clarification_request_detects_marker():
    assert is_clarification_request("need more info\n<!-- AGENT_CLARIFY -->")
    assert is_clarification_request("<!-- agent_clarify -->")
    assert not is_clarification_request("done\n<!-- AGENT_STATE: blocking -->")


def test_task_loop_creates_pr_then_alternates_until_codex_approval(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Implemented.\n<!-- AGENT_PR: 91 -->\n<!-- AGENT_STATE: blocking -->",
            "Fixed review.\n<!-- AGENT_STATE: blocking -->",
        ],
        codex_outputs=[
            "One nit.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
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
    assert runner.comments[-1].startswith("LGTM.")


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
    assert runner.comments[1].startswith("Looks good.")


def test_codex_issue_loop_alternates_until_claude_approval(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Implemented fix.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Addressed Claude's review.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Missing test.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    assert len(runner.comments) == 4
    assert runner.comments[-1].startswith("LGTM.")


def test_codex_issue_loop_requires_codex_to_report_pr_number(tmp_path):
    runner = FakeRunner(
        codex_outputs=["Did some work.\n<!-- AGENT_STATE: blocking -->"],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError, match="PR marker"):
        run_issue_loop(runner, issue_number=56, config=config)

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
    assert runner.comments[1].startswith("Ship it.")


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
    assert runner.comments[1].startswith("Looks good.")


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
            "Looks good.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
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
    assert runner.comments == ["LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"]


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
