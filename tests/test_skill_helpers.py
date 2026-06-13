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
