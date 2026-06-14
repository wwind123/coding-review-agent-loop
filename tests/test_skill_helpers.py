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


# ---------------------------------------------------------------------------
# helpers/skill_runner.py — retry-validate / repair dir
# ---------------------------------------------------------------------------

_REPAIR_BASE = Path(tempfile.gettempdir()) / "coding-review-agent-loop" / "repair"

_VALID_PLAN_REVIEW_DRY = json.dumps(
    {
        "schema_version": 1,
        "kind": "plan_review",
        "state": "approved",
        "summary": "LGTM.",
        "blocking_plan_issues": [],
        "same_plan_followups": [],
        "future_followups": [],
        "prior_plan_item_dispositions": [],
    }
) + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- Codex\n"

_VALID_CONTEXT = json.dumps(
    {"reviewer": "Codex", "prior_items": [], "current_round_items": []}
)


def _make_repair_dir(
    tmpdir: Path,
    *,
    raw_content: str = _VALID_PLAN_REVIEW_DRY,
    issue: int = 9998,
    agent: str = "codex",
    round_num: int = 1,
    dry_run: bool = True,
) -> Path:
    repair_dir = tmpdir / f"{issue}-r{round_num}-{agent}"
    repair_dir.mkdir(parents=True, exist_ok=True)
    (repair_dir / "raw.md").write_text(raw_content, encoding="utf-8")
    (repair_dir / "context.json").write_text(_VALID_CONTEXT, encoding="utf-8")
    (repair_dir / "prior_items.json").write_text("[]", encoding="utf-8")
    manifest = {
        "agent": agent,
        "agent_cap": agent.capitalize(),
        "flow": "plan",
        "issue": issue,
        "repo": "OWNER/REPO",
        "new_round_number": round_num,
        "round_subject": "abc123",
        "item_id_offset": 0,
        "validate_kind": "plan_review",
        "dry_run": dry_run,
    }
    (repair_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return repair_dir


class TestRetryValidate:
    def test_run_plan_round_saves_repair_dir(self) -> None:
        """run-plan-round --dry-run always saves a repair dir before normalization."""
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
            repair_dir = _REPAIR_BASE / "9998-r1-codex"
            assert (repair_dir / "raw.md").exists(), f"raw.md missing in {repair_dir}"
            assert (repair_dir / "manifest.json").exists(), f"manifest.json missing in {repair_dir}"
            manifest = json.loads((repair_dir / "manifest.json").read_text(encoding="utf-8"))
            assert manifest["agent"] == "codex"
            assert manifest["validate_kind"] == "plan_review"

    def test_retry_validate_repair_dir_dry_run_exits_zero(self) -> None:
        """retry-validate --repair-dir with a valid raw response exits 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _write_fake_gh(tmppath)
            env = _make_fake_gh_env(tmppath)
            repair_dir = _make_repair_dir(tmppath, dry_run=True)

            result = _run(
                "helpers.skill_runner", "retry-validate",
                "--repair-dir", str(repair_dir),
                "--dry-run",
                env=env,
                check=False,
            )
            assert result.returncode == 0, (
                f"Expected exit 0, got {result.returncode}\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
            stdout = result.stdout
            json_start = stdout.rfind("\n{")
            if json_start < 0:
                json_start = stdout.find("{")
            assert json_start >= 0, f"No JSON found in stdout:\n{stdout}"
            output = json.loads(stdout[json_start:].strip())
            assert output["state"] == "approved"

    def test_retry_validate_invalid_raw_exits_nonzero(self) -> None:
        """retry-validate with a clearly invalid raw file exits non-zero."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _write_fake_gh(tmppath)
            env = _make_fake_gh_env(tmppath)
            repair_dir = _make_repair_dir(
                tmppath,
                raw_content="This is not JSON at all.\n",
                dry_run=True,
            )

            result = _run(
                "helpers.skill_runner", "retry-validate",
                "--repair-dir", str(repair_dir),
                "--dry-run",
                env=env,
                check=False,
            )
            assert result.returncode != 0, (
                f"Expected non-zero exit, but got 0\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

    def test_retry_validate_blocking_response_returns_blocking_items(self) -> None:
        """retry-validate with a blocking reviewer response returns state=blocking and items."""
        blocking_raw = json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_review",
                "state": "blocking",
                "summary": "Has issues.",
                "blocking_plan_issues": ["Fix the thing."],
                "same_plan_followups": [],
                "future_followups": [],
                "prior_plan_item_dispositions": [],
            }
        ) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Codex\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _write_fake_gh(tmppath)
            env = _make_fake_gh_env(tmppath)
            repair_dir = _make_repair_dir(tmppath, raw_content=blocking_raw, dry_run=True)

            result = _run(
                "helpers.skill_runner", "retry-validate",
                "--repair-dir", str(repair_dir),
                "--dry-run",
                env=env,
                check=False,
            )
            assert result.returncode == 0, (
                f"Expected exit 0, got {result.returncode}\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
            stdout = result.stdout
            json_start = stdout.rfind("\n{")
            if json_start < 0:
                json_start = stdout.find("{")
            output = json.loads(stdout[json_start:].strip())
            assert output["state"] == "blocking"
            assert len(output["blocking_items"]) == 1
            assert output["new_items"][0]["item_id"] == "item-1"

    def test_retry_validate_blocking_via_same_followups_only(self, tmp_path):
        """blocking state with empty blocking_plan_issues but non-empty same_plan_followups
        surfaces the followups as blocking_items (exercises reported_blocking = new_unresolved_texts)."""
        blocking_raw = json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_review",
                "state": "blocking",
                "summary": "Unresolved followups remain.",
                "blocking_plan_issues": [],
                "same_plan_followups": ["Address the followup from last round."],
                "future_followups": [],
                "prior_plan_item_dispositions": [],
            }
        ) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Codex\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _write_fake_gh(tmppath)
            env = _make_fake_gh_env(tmppath)
            repair_dir = _make_repair_dir(tmppath, raw_content=blocking_raw, dry_run=True)

            result = _run(
                "helpers.skill_runner", "retry-validate",
                "--repair-dir", str(repair_dir),
                "--dry-run",
                env=env,
                check=False,
            )
            assert result.returncode == 0, (
                f"Expected exit 0, got {result.returncode}\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
            stdout = result.stdout
            json_start = stdout.rfind("\n{")
            if json_start < 0:
                json_start = stdout.find("{")
            output = json.loads(stdout[json_start:].strip())
            assert output["state"] == "blocking"
            assert len(output["blocking_items"]) == 1
            assert output["blocking_items"][0]["text"] == "Address the followup from last round."


# ---------------------------------------------------------------------------
# helpers/run_external.py  retry loop (in-process, monkeypatched backend)
# ---------------------------------------------------------------------------

# Ensure the repo root is importable so `helpers.run_external` resolves in-process.
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRunExternalRetries:
    def _invoke(self, monkeypatch, agent, outcomes, *, max_retries=2):
        """Drive run_external.main with a fake backend; return (calls, sleeps, output_path, exit_code)."""
        import helpers.run_external as rex
        from coding_review_agent_loop.agents.base import AgentResult  # noqa: F401

        calls = {"n": 0}

        class FakeBackend:
            def run(self, runner, config, prompt):
                i = calls["n"]
                calls["n"] += 1
                outcome = outcomes[min(i, len(outcomes) - 1)]
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

        monkeypatch.setattr("coding_review_agent_loop.agents.codex.CodexBackend", FakeBackend)
        monkeypatch.setattr("coding_review_agent_loop.agents.gemini.GeminiBackend", FakeBackend)
        sleeps: list[int] = []
        monkeypatch.setattr(rex.time, "sleep", lambda s: sleeps.append(s))

        prompt_path = _write_tmp("Review this.")
        output_path = _write_tmp("", suffix=".out.md")
        argv = [
            "run_external", "--agent", agent,
            "--prompt-file", prompt_path, "--output", output_path,
            "--workdir", "/tmp",
            "--max-retries", str(max_retries),
            "--retry-backoff-seconds", "1", "2",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        exit_code = 0
        try:
            rex.main()
        except SystemExit as exc:
            exit_code = exc.code or 0
        return calls, sleeps, output_path, exit_code

    def test_returned_transient_result_is_retried_then_succeeds(self, monkeypatch) -> None:
        from coding_review_agent_loop.agents.base import AgentResult
        outcomes = [
            AgentResult(text="", raw_output="Error: 429 Too Many Requests", returncode=1),
            AgentResult(text="", raw_output="model overloaded", returncode=1),
            AgentResult(text="final review body", returncode=0),
        ]
        calls, sleeps, output_path, exit_code = self._invoke(monkeypatch, "codex", outcomes)
        assert exit_code == 0
        assert calls["n"] == 3
        assert sleeps == [1, 2]  # backoff[0], backoff[1]
        assert Path(output_path).read_text(encoding="utf-8") == "final review body"

    def test_raised_transient_exception_is_retried_then_succeeds(self, monkeypatch) -> None:
        from coding_review_agent_loop.agents.base import AgentResult
        outcomes = [
            RuntimeError("503 Service Unavailable"),
            AgentResult(text="ok", returncode=0),
        ]
        calls, sleeps, output_path, exit_code = self._invoke(monkeypatch, "gemini", outcomes)
        assert exit_code == 0
        assert calls["n"] == 2
        assert sleeps == [1]
        assert Path(output_path).read_text(encoding="utf-8") == "ok"

    def test_non_transient_returned_result_fails_fast(self, monkeypatch) -> None:
        from coding_review_agent_loop.agents.base import AgentResult
        outcomes = [AgentResult(text="", raw_output="invalid api key", returncode=1)]
        calls, sleeps, _output_path, exit_code = self._invoke(monkeypatch, "codex", outcomes)
        assert exit_code == 1
        assert calls["n"] == 1  # no retry on non-transient failure
        assert sleeps == []

    def test_max_retries_zero_makes_single_attempt(self, monkeypatch) -> None:
        from coding_review_agent_loop.agents.base import AgentResult
        outcomes = [AgentResult(text="", raw_output="429 rate limit", returncode=1)]
        calls, sleeps, _output_path, exit_code = self._invoke(
            monkeypatch, "codex", outcomes, max_retries=0
        )
        assert exit_code == 1
        assert calls["n"] == 1  # transient, but retries disabled
        assert sleeps == []


# ---------------------------------------------------------------------------
# helpers/prompt_builders.py  checkout path embedding (#297)
# ---------------------------------------------------------------------------

class TestPromptCheckoutPath:
    _ISSUE = {
        "number": 296, "repo": "wwind123/coding-review-agent-loop",
        "title": "t", "body": "b", "url": "u",
    }

    def _no_bare_skill_runner(self, prompt: str) -> int:
        import re
        # the bare path (no -{agent} suffix) is the bug; it must not appear
        return len(re.findall(r"/coding-review-agent-loop/skill-runner(?!-)", prompt))

    def test_plan_prompt_embeds_explicit_reviewer_workdir(self) -> None:
        from helpers.prompt_builders import build_plan_review_prompt_for_skill
        wd = "/tmp/coding-review-agent-loop/skill-runner-codex"
        prompt = build_plan_review_prompt_for_skill(
            self._ISSUE, "PLAN", [], 1, "codex",
            repo="wwind123/coding-review-agent-loop",
            all_reviewers=["codex", "gemini"], workdir=wd,
        )
        assert wd in prompt
        assert self._no_bare_skill_runner(prompt) == 0

    def test_plan_prompt_default_uses_agent_suffixed_path(self) -> None:
        from helpers.prompt_builders import build_plan_review_prompt_for_skill
        prompt = build_plan_review_prompt_for_skill(
            self._ISSUE, "PLAN", [], 1, "gemini",
            repo="wwind123/coding-review-agent-loop",
            all_reviewers=["codex", "gemini"],
        )
        # default mirrors _workdir_for_agent: skill-runner-{agent}, never the bare path
        assert "skill-runner-gemini" in prompt
        assert self._no_bare_skill_runner(prompt) == 0

    def test_pr_prompt_embeds_explicit_reviewer_workdir(self) -> None:
        from helpers.prompt_builders import build_review_prompt_for_skill
        wd = "/tmp/coding-review-agent-loop/skill-runner-codex"
        prompt = build_review_prompt_for_skill(
            self._ISSUE, "diff --git a b", [], 1, "codex",
            repo="wwind123/coding-review-agent-loop", pr_number=295,
            all_reviewers=["codex", "gemini"], workdir=wd,
        )
        assert wd in prompt
        assert self._no_bare_skill_runner(prompt) == 0


# ---------------------------------------------------------------------------
# helpers/skill_runner.py  _run_test_gate (#296, increment 2)
# ---------------------------------------------------------------------------

class TestRunTestGate:
    import shlex as _shlex
    _PY = _shlex.quote(sys.executable)

    def _gate(self, command, workdir=".", *, dry_run=False):
        from helpers.skill_runner import _run_test_gate
        return _run_test_gate(command, workdir, dry_run=dry_run)

    def test_passing_command(self) -> None:
        r = self._gate(f'{self._PY} -c "import sys; sys.exit(0)"')
        assert r["passed"] is True
        assert r["exit_code"] == 0
        assert "output_tail" in r
        assert "error" not in r

    def test_failing_command_merges_stderr(self) -> None:
        r = self._gate(f'{self._PY} -c "import sys; sys.stderr.write(chr(98)+chr(111)+chr(111)+chr(109)); sys.exit(3)"')
        assert r["passed"] is False
        assert r["exit_code"] == 3
        assert "boom" in r["output_tail"]  # stderr merged into stdout

    def test_missing_executable_reports_error(self) -> None:
        r = self._gate("definitely-not-a-real-binary-xyz --flag")
        assert r["passed"] is False
        assert r["exit_code"] is None
        assert "error" in r

    def test_malformed_quoting_reports_error(self) -> None:
        r = self._gate('echo "unbalanced')
        assert r["passed"] is False
        assert r["exit_code"] is None
        assert "error" in r

    def test_empty_command_reports_error(self) -> None:
        r = self._gate("   ")
        assert r["passed"] is False
        assert "error" in r

    def test_bad_workdir_reports_error(self) -> None:
        r = self._gate(f'{self._PY} -c "pass"', workdir="/nonexistent/path/xyz-12345")
        assert r["passed"] is False
        assert r["exit_code"] is None
        assert "error" in r

    def test_dry_run_does_not_execute(self) -> None:
        # command would error if run; dry-run must skip it
        r = self._gate("definitely-not-a-real-binary-xyz", dry_run=True)
        assert r["skipped"] is True
        assert "error" not in r

    def test_test_workdir_is_honored(self) -> None:
        d = tempfile.mkdtemp()
        r = self._gate(f'{self._PY} -c "import os; print(os.getcwd())"', workdir=d)
        assert r["passed"] is True
        assert os.path.basename(d) in r["output_tail"]

    def test_maybe_gate_none_when_not_provided(self) -> None:
        from helpers.skill_runner import _maybe_test_gate
        assert _maybe_test_gate(None, ".", dry_run=False) is None

    def test_maybe_gate_reports_explicit_empty_command(self) -> None:
        # --test-command "" must be reported as a setup error, not silently skipped
        from helpers.skill_runner import _maybe_test_gate
        r = _maybe_test_gate("", ".", dry_run=False)
        assert r is not None
        assert r["passed"] is False
        assert "error" in r

    def test_invalid_utf8_output_does_not_raise(self) -> None:
        # A process emitting bytes invalid for the locale must not crash the gate.
        r = self._gate(
            f'{self._PY} -c "import sys; sys.stdout.buffer.write(bytes([255])); sys.exit(1)"'
        )
        assert r["passed"] is False
        assert r["exit_code"] == 1
        assert "output_tail" in r  # decoded with replacement rather than raising


# ---------------------------------------------------------------------------
# helpers/skill_runner.py  approved-followups publishing (#300, increment 3)
# ---------------------------------------------------------------------------

class TestApprovedFollowups:
    def test_mint_includes_future_items(self) -> None:
        from helpers.skill_runner import _mint_new_items
        items = _mint_new_items(
            blocking_texts=[], same_texts=[], future_texts=["do X later", "do Y later"],
            flow="pr", agent_cap="Codex", new_round_number=2, item_id_offset=0,
        )
        future = [i for i in items if i["status"] == "future"]
        assert len(future) == 2
        assert {i["text"] for i in future} == {"do X later", "do Y later"}
        assert all(i["reviewer"] == "Codex" and i["source_status"] == "future" for i in future)
        assert [i["item_id"] for i in items] == ["item-1", "item-2"]

    def test_mint_blocking_round_has_no_future(self) -> None:
        # Normalize clears future_followups for blocking reviews -> empty future_texts.
        from helpers.skill_runner import _mint_new_items
        items = _mint_new_items(
            blocking_texts=["must fix"], same_texts=[], future_texts=[],
            flow="pr", agent_cap="Gemini", new_round_number=1, item_id_offset=0,
        )
        assert [i["status"] for i in items] == ["blocking"]

    def test_resume_safe_collection_filter(self) -> None:
        # Mirrors cmd_run_pr_round: future items recovered via completed_reviewer_data
        # (build-resume) are picked up by the same filter as fresh-round items.
        current_round_items = [
            {"status": "blocking", "text": "b"},
            {"status": "future", "text": "f1", "reviewer": "Codex"},
            {"status": "same-pr", "text": "s"},
            {"status": "future", "text": "f2", "reviewer": "Gemini"},
        ]
        future = [i for i in current_round_items if i.get("status") == "future"]
        assert [i["text"] for i in future] == ["f1", "f2"]

    def _future_item(self, text="later", reviewer="Codex"):
        return {
            "item_id": "item-1", "reviewer": reviewer, "source_round": 1,
            "text": text, "status": "future", "source_status": "future", "notes": [],
        }

    def test_publish_ignore_mode_noop(self) -> None:
        from helpers.skill_runner import _publish_pr_followups
        r = _publish_pr_followups("o/r", 1, "sha", "ignore", [self._future_item()], [], dry_run=False)
        assert r == {"mode": "ignore", "published": False, "count": 0}

    def test_publish_dry_run_does_not_post(self, monkeypatch) -> None:
        import helpers.skill_runner as sr
        called = {"n": 0}
        monkeypatch.setattr(sr, "_publish_approved_followups", lambda *a, **k: called.__setitem__("n", called["n"] + 1) or True)
        r = sr._publish_pr_followups("o/r", 1, "sha", "summarize", [self._future_item()], [], dry_run=True)
        assert r == {"mode": "summarize", "dry_run": True, "count": 1}
        assert called["n"] == 0  # never posted in dry-run

    def test_publish_summarize_threads_mode_and_followups(self, monkeypatch) -> None:
        import helpers.skill_runner as sr
        captured = {}

        def fake_publish(runner, *, config, pr_number, head_sha, pr_comments, followups):
            captured["mode"] = config.approved_followups
            captured["pr_number"] = pr_number
            captured["head_sha"] = head_sha
            captured["followup_texts"] = [f.text for f in followups]
            captured["reviewers"] = [f.reviewer for f in followups]
            return True

        monkeypatch.setattr(sr, "_publish_approved_followups", fake_publish)
        r = sr._publish_pr_followups(
            "o/r", 42, "deadbeef", "summarize",
            [self._future_item("ship docs", "Gemini")], [], dry_run=False,
        )
        assert r == {"mode": "summarize", "published": True, "count": 1}
        assert captured["mode"] == "summarize"
        assert captured["pr_number"] == 42 and captured["head_sha"] == "deadbeef"
        assert captured["followup_texts"] == ["ship docs"]
        assert captured["reviewers"] == ["Gemini"]

    def test_publish_idempotent_with_existing_marker(self) -> None:
        # Exercises the real _publish_approved_followups: a pre-existing marker for
        # this pr/head/mode short-circuits before any network post.
        import types as _t
        from helpers.skill_runner import _publish_pr_followups
        marker = "<!-- AGENT_APPROVED_FOLLOWUPS: pr=7 head=abc123 mode=summarize -->"
        pr_comments = [_t.SimpleNamespace(body=f"prior summary\n{marker}")]
        r = _publish_pr_followups("o/r", 7, "abc123", "summarize", [self._future_item()], pr_comments, dry_run=False)
        assert r == {"mode": "summarize", "published": False, "count": 1}

    def test_pr_prompt_mode_changes_instruction(self) -> None:
        from helpers.prompt_builders import build_review_prompt_for_skill
        issue = {"number": 7, "repo": "o/r", "title": "t", "body": "b", "url": "u"}
        common = dict(repo="o/r", pr_number=7, all_reviewers=["codex", "gemini"])
        ignore_prompt = build_review_prompt_for_skill(issue, "d", [], 1, "codex", approved_followups="ignore", **common)
        summ_prompt = build_review_prompt_for_skill(issue, "d", [], 1, "codex", approved_followups="summarize", **common)
        assert ignore_prompt != summ_prompt

    def test_publish_ensures_active_workdir_exists(self, monkeypatch) -> None:
        # Regression (#300/#301): the library runs gh with cwd=active_workdir(config);
        # the Codex+Gemini flow has no Claude checkout, so the helper must create it.
        import helpers.skill_runner as sr
        from coding_review_agent_loop.workdirs import active_workdir
        seen = {}

        def fake_publish(runner, *, config, **kwargs):
            seen["exists"] = Path(active_workdir(config)).is_dir()
            return True

        monkeypatch.setattr(sr, "_publish_approved_followups", fake_publish)
        sr._publish_pr_followups("o/r", 1, "sha", "summarize", [self._future_item()], [], dry_run=False)
        assert seen["exists"] is True


# ---------------------------------------------------------------------------
# helpers/skill_runner.py  run-task-round (#302, increment 4)
# ---------------------------------------------------------------------------

import argparse as _argparse
import io as _io


class TestRunTaskRound:
    def test_resolve_task_text_inline(self) -> None:
        from helpers.skill_runner import _resolve_task_text
        args = _argparse.Namespace(task="do the thing", task_file=None)
        assert _resolve_task_text(args) == "do the thing"

    def test_resolve_task_text_from_file(self) -> None:
        from helpers.skill_runner import _resolve_task_text
        p = _write_tmp("task from file", suffix=".txt")
        args = _argparse.Namespace(task=None, task_file=p)
        assert _resolve_task_text(args).strip() == "task from file"

    def test_resolve_task_text_stdin(self, monkeypatch) -> None:
        from helpers.skill_runner import _resolve_task_text
        monkeypatch.setattr(sys, "stdin", _io.StringIO("piped task"))
        args = _argparse.Namespace(task=None, task_file="-")
        assert _resolve_task_text(args) == "piped task"

    def test_resolve_task_text_empty_errors(self) -> None:
        from helpers.skill_runner import _resolve_task_text
        with pytest.raises(SystemExit):
            _resolve_task_text(_argparse.Namespace(task="   ", task_file=None))

    def test_resolve_task_text_neither_errors(self) -> None:
        from helpers.skill_runner import _resolve_task_text
        with pytest.raises(SystemExit):
            _resolve_task_text(_argparse.Namespace(task=None, task_file=None))

    def test_title_from_task(self) -> None:
        from helpers.skill_runner import _title_from_task
        assert _title_from_task("# My Heading\nbody") == "My Heading"
        assert _title_from_task("\n\n  first real line\nmore") == "first real line"
        assert _title_from_task("   \n  ") == "Agent task"
        assert _title_from_task("x" * 200) == "x" * 80

    def test_task_index_round_trip(self, monkeypatch, tmp_path) -> None:
        import helpers.skill_runner as sr
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        sr._record_task_issue("o/r", "key123", 55)
        monkeypatch.setattr(sr, "_issue_is_open", lambda repo, n: True)
        assert sr._lookup_task_issue("o/r", "key123") == 55

    def test_task_index_missing_returns_none(self, monkeypatch, tmp_path) -> None:
        import helpers.skill_runner as sr
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        assert sr._lookup_task_issue("o/r", "absent") is None

    def test_task_index_closed_issue_returns_none(self, monkeypatch, tmp_path) -> None:
        import helpers.skill_runner as sr
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        sr._record_task_issue("o/r", "key123", 55)
        monkeypatch.setattr(sr, "_issue_is_open", lambda repo, n: False)  # closed/deleted
        assert sr._lookup_task_issue("o/r", "key123") is None

    def _task_args(self, **over):
        base = dict(task="do X", task_file=None, repo="o/r", plan_file="/tmp/p.md",
                    reviewers=["codex"], workdir=None, workdir_codex=None,
                    workdir_gemini=None, dry_run=False)
        base.update(over)
        return _argparse.Namespace(**base)

    def test_task_round_reuses_existing_issue(self, monkeypatch) -> None:
        import helpers.skill_runner as sr
        monkeypatch.setattr(sr, "_lookup_task_issue", lambda repo, key: 77)
        created = {"n": 0}
        monkeypatch.setattr(sr, "_create_task_issue", lambda *a, **k: created.__setitem__("n", created["n"] + 1) or 999)
        seen = {}
        monkeypatch.setattr(sr, "cmd_run_plan_round", lambda args: seen.__setitem__("issue", args.issue))
        sr.cmd_run_task_round(self._task_args())
        assert created["n"] == 0          # did not create a duplicate
        assert seen["issue"] == 77        # delegated with the reused issue

    def test_task_round_creates_when_absent(self, monkeypatch) -> None:
        import helpers.skill_runner as sr
        monkeypatch.setattr(sr, "_lookup_task_issue", lambda repo, key: None)
        monkeypatch.setattr(sr, "_create_task_issue", lambda repo, text, key: 123)
        seen = {}
        monkeypatch.setattr(sr, "cmd_run_plan_round", lambda args: seen.__setitem__("issue", args.issue))
        sr.cmd_run_task_round(self._task_args())
        assert seen["issue"] == 123

    def test_task_round_dry_run_creates_nothing(self, monkeypatch, capsys) -> None:
        import helpers.skill_runner as sr
        monkeypatch.setattr(sr, "_lookup_task_issue", lambda repo, key: None)
        created = {"n": 0}
        monkeypatch.setattr(sr, "_create_task_issue", lambda *a, **k: created.__setitem__("n", created["n"] + 1) or 1)
        delegated = {"n": 0}
        monkeypatch.setattr(sr, "cmd_run_plan_round", lambda args: delegated.__setitem__("n", delegated["n"] + 1))
        sr.cmd_run_task_round(self._task_args(dry_run=True))
        assert created["n"] == 0 and delegated["n"] == 0
        assert "would_create_issue" in capsys.readouterr().out

    def test_task_round_dry_run_existing_issue_does_not_delegate(self, monkeypatch, capsys) -> None:
        # Regression (#302/#303): dry-run is a pure preview even when the task
        # already maps to an open issue — it must not delegate to the plan round.
        import helpers.skill_runner as sr
        monkeypatch.setattr(sr, "_lookup_task_issue", lambda repo, key: 88)
        created = {"n": 0}
        monkeypatch.setattr(sr, "_create_task_issue", lambda *a, **k: created.__setitem__("n", created["n"] + 1) or 1)
        delegated = {"n": 0}
        monkeypatch.setattr(sr, "cmd_run_plan_round", lambda args: delegated.__setitem__("n", delegated["n"] + 1))
        sr.cmd_run_task_round(self._task_args(dry_run=True))
        assert created["n"] == 0 and delegated["n"] == 0
        out = capsys.readouterr().out
        assert "would_reuse_issue" in out and "88" in out


# ---------------------------------------------------------------------------
# agent-memory wiring (#306)
# ---------------------------------------------------------------------------

class TestAgentMemory:
    def _ctx(self):
        from coding_review_agent_loop.memory import AgentMemoryContext
        return AgentMemoryContext(
            memory_dir=Path("/tmp/mem"), current_commit="abc123",
            last_analyzed_commit=None, changed_files=(),
            repo_summary="REPO SUMMARY TEXT", architecture_map=None,
            test_profile=None, toolchain=None,
        )

    _ISSUE = {"number": 9, "repo": "o/r", "title": "t", "body": "b", "url": "u"}

    def test_plan_prompt_includes_memory(self) -> None:
        from helpers.prompt_builders import build_plan_review_prompt_for_skill
        common = dict(repo="o/r", all_reviewers=["codex", "gemini"])
        without = build_plan_review_prompt_for_skill(self._ISSUE, "PLAN", [], 1, "codex", **common)
        with_mem = build_plan_review_prompt_for_skill(self._ISSUE, "PLAN", [], 1, "codex", memory=self._ctx(), **common)
        assert with_mem != without
        assert "Cached repo memory is available" in with_mem
        assert "REPO SUMMARY TEXT" in with_mem

    def test_pr_prompt_includes_memory(self) -> None:
        from helpers.prompt_builders import build_review_prompt_for_skill
        common = dict(repo="o/r", pr_number=9, all_reviewers=["codex", "gemini"])
        without = build_review_prompt_for_skill(self._ISSUE, "diff", [], 1, "codex", **common)
        with_mem = build_review_prompt_for_skill(self._ISSUE, "diff", [], 1, "codex", memory=self._ctx(), **common)
        assert with_mem != without
        assert "Cached repo memory is available" in with_mem

    def test_prepare_memory_dry_run_is_noop(self, monkeypatch) -> None:
        import helpers.skill_runner as sr
        called = {"checkout": 0, "prepare": 0}
        monkeypatch.setattr(sr, "ensure_temp_checkout", lambda *a, **k: called.__setitem__("checkout", 1))
        monkeypatch.setattr(sr, "prepare_agent_memory", lambda *a, **k: called.__setitem__("prepare", 1))
        assert sr._prepare_skill_memory("o/r", ["codex"], "/tmp/wd", refresh=False, dry_run=True) is None
        assert called == {"checkout": 0, "prepare": 0}

    def test_prepare_memory_no_reviewers(self) -> None:
        import helpers.skill_runner as sr
        assert sr._prepare_skill_memory("o/r", [], "/tmp/wd", refresh=False, dry_run=False) is None

    def test_prepare_memory_resilient_to_errors(self, monkeypatch) -> None:
        import helpers.skill_runner as sr
        monkeypatch.setattr(sr, "ensure_temp_checkout", lambda *a, **k: None)
        def boom(*a, **k):
            raise RuntimeError("git exploded")
        monkeypatch.setattr(sr, "prepare_agent_memory", boom)
        assert sr._prepare_skill_memory("o/r", ["codex"], "/tmp/wd", refresh=False, dry_run=False) is None

    def test_prepare_memory_success_builds_enabled_config(self, monkeypatch) -> None:
        import helpers.skill_runner as sr
        from coding_review_agent_loop.workdirs import active_workdir
        captured = {}
        sentinel = object()
        monkeypatch.setattr(sr, "ensure_temp_checkout", lambda *a, **k: None)
        def fake_prepare(runner, config):
            captured["config"] = config
            return sentinel
        monkeypatch.setattr(sr, "prepare_agent_memory", fake_prepare)
        out = sr._prepare_skill_memory("o/r", ["codex", "gemini"], "/tmp/skill-runner-codex", refresh=True, dry_run=False)
        assert out is sentinel
        cfg = captured["config"]
        assert cfg.agent_memory is True and cfg.refresh_agent_memory is True
        assert str(active_workdir(cfg)) == "/tmp/skill-runner-codex"


# ---------------------------------------------------------------------------
# external-agent usage/cost tracking (#308)
# ---------------------------------------------------------------------------

class TestUsageTracking:
    def _rec(self, agent, mode, inp, out, returncode=0):
        return {
            "reviewer_name": agent.capitalize(),
            "usage": {
                "agent": agent, "session_id": f"s-{agent}", "returncode": returncode,
                "usage": {"mode": mode, "input_tokens": inp, "output_tokens": out,
                          "total_tokens": inp + out},
            },
        }

    def test_aggregate_sums_and_preserves_modes(self) -> None:
        from helpers.skill_runner import _aggregate_reviewer_usage
        out = _aggregate_reviewer_usage([
            self._rec("codex", "exact", 100, 50),
            self._rec("gemini", "estimated", 80, 20),
            {"reviewer_name": "NoUsage"},  # carries no usage → ignored
        ])
        assert out["scope"] == "external-agents-only"
        assert "note" in out
        totals = out["totals"]
        assert totals["call_count"] == 2
        assert totals["input_tokens"] == 180 and totals["output_tokens"] == 70
        assert totals["total_tokens"] == 250
        # modes preserved across reviewers
        assert totals["exact_calls"] == 1 and totals["estimated_calls"] == 1
        # success_count omitted (would be a misleading 0)
        assert "success_count" not in totals
        assert set(out["per_agent"]) == {"codex", "gemini"}
        assert "success_count" not in out["per_agent"]["codex"]

    def test_aggregate_none_when_no_usage(self) -> None:
        from helpers.skill_runner import _aggregate_reviewer_usage
        assert _aggregate_reviewer_usage([{"reviewer_name": "x"}, {"reviewer_name": "y"}]) is None
        assert _aggregate_reviewer_usage([]) is None

    def test_metadata_round_trips_usage(self) -> None:
        # PostedRoundMetadata.usage survives encode -> decode (so build-resume
        # can recover it for resumed-round aggregation).
        from coding_review_agent_loop.round_state import (
            PostedRoundMetadata, _encode_round_metadata, _decode_round_metadata,
        )
        usage = {"agent": "codex", "returncode": 0, "usage": {"mode": "exact", "total_tokens": 5}}
        meta = PostedRoundMetadata(flow="pr", role="reviewer", agent="Codex",
                                   round_number=1, subject="abc", usage=usage)
        decoded = _decode_round_metadata(_encode_round_metadata(meta))
        assert decoded.usage == usage

    def test_metadata_usage_defaults_none(self) -> None:
        from coding_review_agent_loop.round_state import (
            PostedRoundMetadata, _encode_round_metadata, _decode_round_metadata,
        )
        meta = PostedRoundMetadata(flow="plan", role="coder", agent="Claude",
                                   round_number=1, subject="x")
        assert _decode_round_metadata(_encode_round_metadata(meta)).usage is None

    def _run_external_usage(self, monkeypatch, tmp_path, agent, result):
        import helpers.run_external as rex
        class FakeBackend:
            def run(self, runner, config, prompt):
                return result
        monkeypatch.setattr("coding_review_agent_loop.agents.codex.CodexBackend", FakeBackend)
        monkeypatch.setattr("coding_review_agent_loop.agents.gemini.GeminiBackend", FakeBackend)
        usage_path = tmp_path / "u.json"
        monkeypatch.setattr(sys, "argv", [
            "run_external", "--agent", agent,
            "--prompt-file", _write_tmp("Prompt text here."),
            "--output", _write_tmp("", suffix=".out.md"),
            "--workdir", "/tmp", "--usage-output", str(usage_path),
        ])
        rex.main()
        return usage_path

    def test_run_external_writes_exact_usage(self, monkeypatch, tmp_path) -> None:
        from coding_review_agent_loop.agents.base import AgentResult
        from coding_review_agent_loop.usage import UsageMetadata
        result = AgentResult(
            text="review body", returncode=0, session_id="sess1",
            usage=UsageMetadata(mode="exact", input_tokens=10, output_tokens=5, total_tokens=15),
        )
        usage_path = self._run_external_usage(monkeypatch, tmp_path, "codex", result)
        data = json.loads(usage_path.read_text(encoding="utf-8"))
        assert data["agent"] == "codex" and data["session_id"] == "sess1"
        assert data["usage"]["mode"] == "exact" and data["usage"]["total_tokens"] == 15

    def test_run_external_estimates_usage_when_backend_omits(self, monkeypatch, tmp_path) -> None:
        from coding_review_agent_loop.agents.base import AgentResult
        result = AgentResult(text="some response text", returncode=0, usage=None)
        usage_path = self._run_external_usage(monkeypatch, tmp_path, "gemini", result)
        data = json.loads(usage_path.read_text(encoding="utf-8"))
        assert data["usage"]["mode"] == "estimated"

    def test_run_external_dry_run_writes_no_usage(self, monkeypatch, tmp_path) -> None:
        import helpers.run_external as rex
        usage_path = tmp_path / "u.json"
        monkeypatch.setattr(sys, "argv", [
            "run_external", "--agent", "codex",
            "--prompt-file", _write_tmp("Review this."),
            "--output", _write_tmp("", suffix=".out.md"),
            "--workdir", "/tmp", "--usage-output", str(usage_path), "--dry-run",
        ])
        rex.main()
        assert not usage_path.exists()
