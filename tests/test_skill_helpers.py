"""Unit tests for the Claude Code skill helper CLIs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

HELPERS = Path(__file__).parent.parent / "helpers"
SRC = Path(__file__).parent.parent / "src"

# Make library importable for direct calls in this test file
sys.path.insert(0, str(SRC))


def _run(*args: str, check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, "-m", *args],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
        check=False,
        env=env,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"Command {args!r} failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    return result


def _make_fake_gh_env(fake_gh_dir: Path) -> dict:
    """Return an environment dict with a fake gh prepended to PATH."""
    env = os.environ.copy()
    env["PATH"] = str(fake_gh_dir) + ":" + env.get("PATH", "")
    return env


def _write_fake_gh(directory: Path) -> Path:
    """Write a fake gh script that returns canned responses for test isolation."""
    script = directory / "gh"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, json\n"
        "args = sys.argv[1:]\n"
        "combined = ' '.join(args)\n"
        "# _fetch_issue_comments_raw: gh issue view N --json comments → empty list\n"
        "if args[:2] == ['issue', 'view'] and '--json' in args and 'comments' in combined:\n"
        "    print(json.dumps({'comments': []}))\n"
        "# _fetch_issue_json: gh issue view N --json number,title,body,url\n"
        "elif args[:2] == ['issue', 'view']:\n"
        "    print(json.dumps({'number': 9998, 'title': 'Test Issue', 'body': 'Test body', 'url': 'https://github.com/test/repo/issues/9998'}))\n"
        "elif args[:2] == ['pr', 'view']:\n"
        "    print(json.dumps({'number': 9998, 'title': 'Test PR', 'body': 'Test body', 'url': 'https://github.com/test/repo/pull/9998', 'headRefOid': 'abc123def'}))\n"
        "elif args[:2] == ['pr', 'diff']:\n"
        "    print('diff --git a/foo.py b/foo.py')\n"
        "elif 'comment' in args:\n"
        "    # Should not be called in dry-run; fail loudly\n"
        "    print('fake-gh: unexpected comment post in dry-run', file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "else:\n"
        "    print('{}')\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


# ---------------------------------------------------------------------------
# helpers/validate_response.py
# ---------------------------------------------------------------------------

_VALID_PLAN_STATE = """\
## Plan

1. Step one

<!-- AGENT_PLAN_STATE: approved -->
-- Anthropic Claude
"""

_INVALID_PLAN_STATE = "This has no marker at all."

_VALID_PLAN_REVIEW = json.dumps(
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
) + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- Codex\n"


def _write_tmp(content: str, suffix: str = ".md") -> str:
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as f:
        f.write(content)
        return f.name


class TestValidateResponse:
    def test_valid_plan_state_accepted(self) -> None:
        path = _write_tmp(_VALID_PLAN_STATE)
        result = _run("helpers.validate_response", "--file", path, "--kind", "plan_state")
        assert "validation passed: plan_state" in result.stdout

    def test_missing_plan_state_marker_rejected(self) -> None:
        path = _write_tmp(_INVALID_PLAN_STATE)
        result = _run("helpers.validate_response", "--file", path, "--kind", "plan_state", check=False)
        assert result.returncode != 0
        assert "validation failed: plan_state" in result.stderr

    def test_valid_plan_review_accepted(self) -> None:
        path = _write_tmp(_VALID_PLAN_REVIEW)
        ctx_path = _write_tmp(
            json.dumps({"reviewer": "Codex", "prior_items": [], "current_round_items": []}),
            suffix=".json",
        )
        result = _run(
            "helpers.validate_response",
            "--file",
            path,
            "--kind",
            "plan_review",
            "--context-file",
            ctx_path,
        )
        assert "validation passed: plan_review" in result.stdout

    def test_plan_review_with_unknown_prior_item_rejected(self) -> None:
        review = json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_review",
                "state": "blocking",
                "summary": "Blocking.",
                "blocking_plan_issues": ["Something bad."],
                "same_plan_followups": [],
                "future_followups": [],
                "prior_plan_item_dispositions": [
                    {"item_id": "item-999", "disposition": "resolved"}
                ],
            }
        ) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Codex\n"

        path = _write_tmp(review)
        # Empty prior items — item-999 is unknown
        ctx_path = _write_tmp(
            json.dumps({"reviewer": "Codex", "prior_items": [], "current_round_items": []}),
            suffix=".json",
        )
        result = _run(
            "helpers.validate_response",
            "--file",
            path,
            "--kind",
            "plan_review",
            "--context-file",
            ctx_path,
            check=False,
        )
        assert result.returncode != 0

    def test_plan_revision_with_unknown_prior_item_rejected(self) -> None:
        """plan_revision must reject dispositions for item IDs not in the prior-items ledger."""
        revision = json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_revision",
                "state": "blocking",
                "summary": "Revised plan.",
                "prior_plan_item_dispositions": [
                    {"item_id": "item-unknown-99", "disposition": "resolved"}
                ],
                "plan_steps": ["Step A", "Step B"],
            }
        ) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude\n"

        path = _write_tmp(revision)
        # context has item-1 but revision references item-unknown-99
        prior_item = {
            "item_id": "item-1",
            "reviewer": "Codex",
            "source_round": 1,
            "text": "Some issue.",
            "status": "blocking",
            "source_status": "blocking",
            "notes": [],
        }
        ctx_path = _write_tmp(
            json.dumps({"prior_items": [prior_item], "current_round_items": []}),
            suffix=".json",
        )
        result = _run(
            "helpers.validate_response",
            "--file",
            path,
            "--kind",
            "plan_revision",
            "--context-file",
            ctx_path,
            check=False,
        )
        assert result.returncode != 0

    def test_plan_revision_with_known_items_accepted(self) -> None:
        """plan_revision with only known prior item IDs must be accepted."""
        prior_item = {
            "item_id": "item-1",
            "reviewer": "Codex",
            "source_round": 1,
            "text": "Some issue.",
            "status": "blocking",
            "source_status": "blocking",
            "notes": [],
        }
        revision = json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_revision",
                "state": "blocking",
                "summary": "Revised plan.",
                "prior_plan_item_dispositions": [
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
                "plan_steps": ["Step A", "Step B"],
            }
        ) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude\n"

        path = _write_tmp(revision)
        ctx_path = _write_tmp(
            json.dumps({"prior_items": [prior_item], "current_round_items": []}),
            suffix=".json",
        )
        result = _run(
            "helpers.validate_response",
            "--file",
            path,
            "--kind",
            "plan_revision",
            "--context-file",
            ctx_path,
        )
        assert "validation passed: plan_revision" in result.stdout


# ---------------------------------------------------------------------------
# helpers/state_manager.py  (session round-trip + attach-metadata)
# ---------------------------------------------------------------------------

class TestStateManager:
    def _session_path(self, repo: str, issue: int) -> Path:
        import os
        slug = repo.replace("/", "-").replace(":", "-")
        state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
        return state_home / "coding-review-agent-loop" / "skill-sessions" / slug / f"{issue}.json"

    def test_write_and_read_session(self) -> None:
        repo = "test/skill-repo"
        issue = 9999
        fields = {"last_completed_step": "post_review", "session_id": "abc123"}
        _run(
            "helpers.state_manager",
            "write-session",
            "--issue",
            str(issue),
            "--repo",
            repo,
            "--fields",
            json.dumps(fields),
        )
        result = _run("helpers.state_manager", "read-session", "--issue", str(issue), "--repo", repo)
        data = json.loads(result.stdout)
        assert data["last_completed_step"] == "post_review"
        assert data["session_id"] == "abc123"

    def test_write_and_clear_pending_comment(self) -> None:
        repo = "test/skill-repo"
        issue = 9999
        body_path = "/tmp/pending-comment-body.md"
        _run(
            "helpers.state_manager",
            "write-pending-comment",
            "--issue",
            str(issue),
            "--repo",
            repo,
            "--body",
            body_path,
        )
        result = _run("helpers.state_manager", "read-session", "--issue", str(issue), "--repo", repo)
        data = json.loads(result.stdout)
        assert data.get("pending_comment_body") == body_path

        _run(
            "helpers.state_manager",
            "clear-pending-comment",
            "--issue",
            str(issue),
            "--repo",
            repo,
        )
        result = _run("helpers.state_manager", "read-session", "--issue", str(issue), "--repo", repo)
        data = json.loads(result.stdout)
        assert "pending_comment_body" not in data

    def test_attach_metadata_produces_valid_agent_loop_meta(self) -> None:
        """attach-metadata must embed AGENT_LOOP_META that _resume_plan_round recognizes."""
        plan_body = _VALID_PLAN_STATE

        with tempfile.TemporaryDirectory() as tmpdir:
            body_file = Path(tmpdir) / "plan.md"
            body_file.write_text(plan_body, encoding="utf-8")
            output_file = Path(tmpdir) / "plan_tagged.md"

            _run(
                "helpers.state_manager",
                "attach-metadata",
                "--body-file",
                str(body_file),
                "--output",
                str(output_file),
                "--flow",
                "plan",
                "--role",
                "coder",
                "--agent",
                "Claude",
                "--round-number",
                "1",
                "--state",
                "approved",
                "--subject-plan-file",
                str(body_file),
                "--canonical-plan-file",
                str(body_file),
            )

            tagged = output_file.read_text(encoding="utf-8")
            assert "AGENT_LOOP_META" in tagged

            # Verify _resume_plan_round can reconstruct from this comment alone
            from coding_review_agent_loop.round_state import _resume_plan_round

            class _FC:
                def __init__(self, body: str) -> None:
                    self.body = body

            result = _resume_plan_round([_FC(tagged)], configured_reviewers=["codex"])
            # A coder comment with no reviewer comments → returns the round so reviewers can run
            assert result is not None, "build-resume could not find skill-posted coder round"
            _plan_text, resumed = result
            assert resumed.round_number == 1

    def test_attach_metadata_reviewer_found_by_resume(self) -> None:
        """Coder + reviewer comments both with AGENT_LOOP_META → resume finds completed reviewer."""
        plan_body = _VALID_PLAN_STATE
        reviewer_body = _VALID_PLAN_REVIEW

        with tempfile.TemporaryDirectory() as tmpdir:
            plan_file = Path(tmpdir) / "plan.md"
            plan_file.write_text(plan_body, encoding="utf-8")
            plan_tagged = Path(tmpdir) / "plan_tagged.md"
            review_file = Path(tmpdir) / "review.md"
            review_file.write_text(reviewer_body, encoding="utf-8")
            review_tagged = Path(tmpdir) / "review_tagged.md"

            # Attach coder metadata
            _run(
                "helpers.state_manager",
                "attach-metadata",
                "--body-file", str(plan_file),
                "--output", str(plan_tagged),
                "--flow", "plan", "--role", "coder", "--agent", "Claude",
                "--round-number", "1", "--state", "approved",
                "--subject-plan-file", str(plan_file),
                "--canonical-plan-file", str(plan_file),
            )

            # Attach reviewer metadata (same subject)
            _run(
                "helpers.state_manager",
                "attach-metadata",
                "--body-file", str(review_file),
                "--output", str(review_tagged),
                "--flow", "plan", "--role", "reviewer", "--agent", "Codex",
                "--round-number", "1", "--state", "approved",
                "--subject-plan-file", str(plan_file),
            )

            from coding_review_agent_loop.round_state import _resume_plan_round

            class _FC:
                def __init__(self, body: str) -> None:
                    self.body = body

            result = _resume_plan_round(
                [_FC(plan_tagged.read_text(encoding="utf-8")),
                 _FC(review_tagged.read_text(encoding="utf-8"))],
                configured_reviewers=["codex"],
            )
            assert result is not None, "build-resume did not find the round"
            _plan_text, resumed = result
            assert resumed.round_number == 1
            assert len(resumed.completed_reviews) == 1, (
                f"Expected 1 completed reviewer (Codex), got {len(resumed.completed_reviews)}"
            )


# ---------------------------------------------------------------------------
# helpers/run_external.py  (dry-run only)
# ---------------------------------------------------------------------------

class TestRunExternal:
    def test_dry_run_exits_zero_and_writes_valid_stub(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as pf:
            pf.write("Prompt text.")
            prompt_path = pf.name
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as of:
            output_path = of.name

        result = _run(
            "helpers.run_external",
            "--agent",
            "codex",
            "--prompt-file",
            prompt_path,
            "--output",
            output_path,
            "--workdir",
            "/tmp",
            "--dry-run",
        )
        assert result.returncode == 0
        content = Path(output_path).read_text(encoding="utf-8")
        # The dry-run stub must contain a valid plan_review JSON and AGENT_PLAN_STATE marker
        assert "AGENT_PLAN_STATE: approved" in content
        assert '"state": "approved"' in content

    def test_dry_run_coder_role_exits_zero_and_writes_plan_state_stub(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as pf:
            pf.write("Implement the feature.")
            prompt_path = pf.name
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as of:
            output_path = of.name

        result = _run(
            "helpers.run_external",
            "--agent",
            "codex",
            "--role",
            "coder",
            "--prompt-file",
            prompt_path,
            "--output",
            output_path,
            "--workdir",
            "/tmp",
            "--dry-run",
        )
        assert result.returncode == 0
        content = Path(output_path).read_text(encoding="utf-8")
        # Coder dry-run stub must be a valid plan_state (no JSON, just markdown + marker)
        assert "AGENT_PLAN_STATE: approved" in content
        # Must NOT be a plan_review JSON blob
        assert '"kind": "plan_review"' not in content

        # Confirm the stub passes plan_state validation
        stub_path = _write_tmp(content)
        validate_result = _run(
            "helpers.validate_response",
            "--file",
            stub_path,
            "--kind",
            "plan_state",
        )
        assert "validation passed: plan_state" in validate_result.stdout

    def test_dry_run_flow_pr_writes_pr_review_stub(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as pf:
            pf.write("Review this PR.")
            prompt_path = pf.name
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as of:
            output_path = of.name

        result = _run(
            "helpers.run_external",
            "--agent", "codex",
            "--prompt-file", prompt_path,
            "--output", output_path,
            "--workdir", "/tmp",
            "--flow", "pr",
            "--dry-run",
        )
        assert result.returncode == 0
        content = Path(output_path).read_text(encoding="utf-8")
        # PR stub must use AGENT_STATE (not AGENT_PLAN_STATE) and kind pr_review
        assert "AGENT_STATE: approved" in content
        assert "AGENT_PLAN_STATE" not in content
        assert '"kind": "pr_review"' in content

        # Confirm stub passes pr_review validation
        stub_path = _write_tmp(content)
        ctx_path = _write_tmp(
            json.dumps({"reviewer": "Codex", "prior_items": [], "current_round_items": []}),
            suffix=".json",
        )
        val = _run(
            "helpers.validate_response",
            "--file", stub_path,
            "--kind", "pr_review",
            "--context-file", ctx_path,
        )
        assert "validation passed: pr_review" in val.stdout


# ---------------------------------------------------------------------------
# helpers/skill_runner.py
# ---------------------------------------------------------------------------

_VALID_PLAN_FOR_RUNNER = """\
## Plan

1. Implement the feature.

<!-- AGENT_PLAN_STATE: approved -->
-- Anthropic Claude
"""


class TestSkillRunner:
    def test_run_plan_round_dry_run_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _write_fake_gh(tmppath)
            env = _make_fake_gh_env(tmppath)
            plan_path = tmppath / "plan.md"
            plan_path.write_text(_VALID_PLAN_FOR_RUNNER, encoding="utf-8")

            result = _run(
                "helpers.skill_runner", "run-plan-round",
                "--issue", "9998",
                "--repo", "test/skill-repo",
                "--plan-file", str(plan_path),
                "--reviewers", "codex",
                "--dry-run",
                env=env,
            )
            assert result.returncode == 0
            # The runner prints progress lines then a final multi-line JSON object.
            # Find the last '\n{' to locate the start of that object.
            stdout = result.stdout
            json_start = stdout.rfind("\n{")
            if json_start < 0:
                json_start = stdout.find("{")
            assert json_start >= 0, f"No JSON found in stdout:\n{stdout}"
            output = json.loads(stdout[json_start:].strip())
            assert "state" in output
            assert "round_number" in output
            assert "blocking_items" in output
            assert "approved_reviewers" in output

    def test_run_plan_round_dry_run_no_github_posts(self) -> None:
        """Dry-run must not invoke gh for posting (only for build-resume reads)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _write_fake_gh(tmppath)
            env = _make_fake_gh_env(tmppath)
            plan_path = tmppath / "plan.md"
            plan_path.write_text(_VALID_PLAN_FOR_RUNNER, encoding="utf-8")

            result = _run(
                "helpers.skill_runner", "run-plan-round",
                "--issue", "9998",
                "--repo", "test/skill-repo",
                "--plan-file", str(plan_path),
                "--reviewers", "codex",
                "--dry-run",
                env=env,
            )
            assert result.returncode == 0

    def test_run_plan_round_invalid_plan_exits_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _write_fake_gh(tmppath)
            env = _make_fake_gh_env(tmppath)
            bad_plan = tmppath / "bad_plan.md"
            bad_plan.write_text("No marker here at all.", encoding="utf-8")

            result = _run(
                "helpers.skill_runner", "run-plan-round",
                "--issue", "9998",
                "--repo", "test/skill-repo",
                "--plan-file", str(bad_plan),
                "--reviewers", "codex",
                "--dry-run",
                env=env,
                check=False,
            )
            assert result.returncode != 0

    def test_run_pr_round_dry_run_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _write_fake_gh(tmppath)
            env = _make_fake_gh_env(tmppath)

            result = _run(
                "helpers.skill_runner", "run-pr-round",
                "--pr", "9998",
                "--repo", "test/skill-repo",
                "--reviewers", "codex",
                "--dry-run",
                env=env,
            )
            assert result.returncode == 0
            stdout = result.stdout
            json_start = stdout.rfind("\n{")
            if json_start < 0:
                json_start = stdout.find("{")
            assert json_start >= 0, f"No JSON found in stdout:\n{stdout}"
            output = json.loads(stdout[json_start:].strip())
            assert output["state"] in ("approved", "blocking")
            assert "round_number" in output
            assert "blocking_items" in output
            assert "approved_reviewers" in output
