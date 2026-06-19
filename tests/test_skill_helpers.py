"""Unit tests for the Claude Code skill helper CLIs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
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

    def test_callable_validator_preserves_unknown_prior_item_error_type(self) -> None:
        from coding_review_agent_loop.errors import UnknownPriorItemDispositionError
        from helpers.validate_response import validate_response_text

        review = json.dumps({
            "schema_version": 1,
            "kind": "plan_review",
            "state": "blocking",
            "summary": "Blocking.",
            "blocking_plan_issues": ["Fix it."],
            "same_plan_followups": [],
            "future_followups": [],
            "prior_plan_item_dispositions": [
                {"item_id": "item-unknown", "disposition": "resolved"}
            ],
        }) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Codex\n"

        with pytest.raises(UnknownPriorItemDispositionError) as exc_info:
            validate_response_text(
                review,
                kind="plan_review",
                reviewer="Codex",
                prior_items=[],
            )
        assert exc_info.value.unknown_ids == ("item-unknown",)


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

    def test_attach_metadata_persists_compact_prior_summaries(self) -> None:
        body = _VALID_PLAN_STATE

        with tempfile.TemporaryDirectory() as tmpdir:
            body_file = Path(tmpdir) / "body.md"
            body_file.write_text(body, encoding="utf-8")
            compact_file = Path(tmpdir) / "compact.json"
            compact_file.write_text(json.dumps(["summary one", "summary two"]), encoding="utf-8")
            output_file = Path(tmpdir) / "tagged.md"

            _run(
                "helpers.state_manager",
                "attach-metadata",
                "--body-file", str(body_file),
                "--output", str(output_file),
                "--flow", "pr", "--role", "coder", "--agent", "Codex",
                "--round-number", "2", "--state", "blocking",
                "--subject", "abc123",
                "--compact-prior-summaries-file", str(compact_file),
            )

            from coding_review_agent_loop.round_state import (
                ROUND_RESUME_MARKER_RE,
                _decode_round_metadata,
            )

            tagged = output_file.read_text(encoding="utf-8")
            match = ROUND_RESUME_MARKER_RE.search(tagged)
            assert match is not None
            metadata = _decode_round_metadata(match.group("payload"))
            assert metadata.compact_prior_summaries == ("summary one", "summary two")

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


class TestStructuredSkillRecovery:
    @staticmethod
    def _prior_item(item_id: str = "item-1") -> dict:
        return {
            "item_id": item_id,
            "reviewer": "Codex",
            "source_round": 1,
            "text": "Fix the issue.",
            "status": "blocking",
            "source_status": "blocking",
            "notes": [],
        }

    def test_reviewer_envelope_and_unknown_item_are_recovered(self, monkeypatch) -> None:
        import helpers.skill_runner as sr
        from helpers.validate_response import validate_response_text

        raw = json.dumps({
            "schema_version": 1,
            "kind": "plan_review",
            "state": "blocking",
            "summary": "One issue remains.",
            "blocking_plan_issues": ["Still broken."],
            "same_plan_followups": [],
            "future_followups": [],
            "prior_plan_item_dispositions": [
                {"item_id": "item-1", "disposition": "blocking"},
                {"item_id": "item-invented", "disposition": "resolved"},
            ],
        }) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Codex\ntrailing prose"
        original = raw
        monkeypatch.setattr(sr, "attempt_repair", lambda *a, **k: pytest.fail("repair not needed"))
        validate = lambda text: validate_response_text(
            text,
            kind="plan_review",
            reviewer="Codex",
            prior_items=[self._prior_item()],
        )

        recovered, parsed = sr._recover_structured_response(
            raw,
            expected_kind="plan_review",
            validate=validate,
            allowed_prior_item_ids=["item-1"],
            reviewer_normalization=True,
        )

        assert raw == original
        assert "item-invented" not in recovered
        assert parsed.state == "blocking"

    def test_custom_gemini_command_reaches_repair(self, monkeypatch) -> None:
        import helpers.skill_runner as sr
        from helpers.validate_response import validate_response_text

        seen: dict[str, object] = {}
        repaired = _VALID_PLAN_REVIEW_DRY

        def fake_repair(raw, gemini_cmd, **kwargs):
            seen.update({"raw": raw, "gemini_cmd": gemini_cmd, "kwargs": kwargs})
            return repaired

        monkeypatch.setattr(sr, "attempt_repair", fake_repair)
        validate = lambda text: validate_response_text(
            text, kind="plan_review", reviewer="Codex",
        )
        recovered, _ = sr._recover_structured_response(
            "malformed review",
            expected_kind="plan_review",
            validate=validate,
            gemini_cmd="/opt/bin/gemini-custom",
            reviewer_normalization=True,
        )

        assert recovered == repaired
        assert seen["gemini_cmd"] == "/opt/bin/gemini-custom"
        assert seen["kwargs"]["expected_kind"] == "plan_review"

    def test_plan_revision_ack_reconstructed_from_unique_evidence(self, monkeypatch) -> None:
        import helpers.skill_runner as sr
        from coding_review_agent_loop.protocol import validate_human_requirements_acknowledgement
        from helpers.validate_response import validate_response_text

        raw = json.dumps({
            "schema_version": 1,
            "kind": "plan_revision",
            "state": "blocking",
            "summary": "Revised.",
            "prior_plan_item_dispositions": [
                {"item_id": "item-1", "disposition": "resolved"}
            ],
            "plan_steps": ["Implement the fix."],
        }) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Codex"
        acknowledgement = (
            "<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n"
            "### Human requirements\n\n"
            "- Requirement 1: preserved the required behavior."
        )
        monkeypatch.setattr(sr, "attempt_repair", lambda *a, **k: pytest.fail("repair not needed"))

        def validate(text):
            parsed = validate_response_text(
                text,
                kind="plan_revision",
                prior_items=[self._prior_item()],
            )
            validate_human_requirements_acknowledgement(
                text,
                surfaced_requirement_ids=("Requirement 1",),
                requires_direct_discussion_ack=False,
            )
            return parsed

        recovered, _ = sr._recover_structured_response(
            raw,
            expected_kind="plan_revision",
            validate=validate,
            allowed_prior_item_ids=["item-1"],
            surfaced_requirement_ids=["Requirement 1"],
            response_evidence={"message_text": f"notes\n{acknowledgement}\n-- Codex"},
        )
        assert acknowledgement in recovered

    def test_plan_revision_ambiguous_ack_evidence_falls_through(self, monkeypatch) -> None:
        import helpers.skill_runner as sr
        from coding_review_agent_loop.errors import AgentLoopError
        from coding_review_agent_loop.protocol import validate_human_requirements_acknowledgement
        from helpers.validate_response import validate_response_text

        raw = json.dumps({
            "schema_version": 1,
            "kind": "plan_revision",
            "state": "blocking",
            "summary": "Revised.",
            "prior_plan_item_dispositions": [
                {"item_id": "item-1", "disposition": "resolved"}
            ],
            "plan_steps": ["Implement the fix."],
        }) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Codex"
        first = (
            "<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n### Human requirements\n"
            "- Requirement 1: first explanation."
        )
        second = (
            "<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n### Human requirements\n"
            "- Requirement 1: different explanation."
        )
        monkeypatch.setattr(sr, "attempt_repair", lambda *a, **k: None)

        def validate(text):
            parsed = validate_response_text(
                text, kind="plan_revision", prior_items=[self._prior_item()],
            )
            validate_human_requirements_acknowledgement(
                text,
                surfaced_requirement_ids=("Requirement 1",),
                requires_direct_discussion_ack=False,
            )
            return parsed

        with pytest.raises(AgentLoopError, match="signed human requirements"):
            sr._recover_structured_response(
                raw,
                expected_kind="plan_revision",
                validate=validate,
                allowed_prior_item_ids=["item-1"],
                surfaced_requirement_ids=["Requirement 1"],
                response_evidence={"message_text": f"{first}\n-- Codex\n{second}\n-- Codex"},
            )

    def test_coder_followup_envelope_recovery(self, monkeypatch) -> None:
        import helpers.skill_runner as sr
        from helpers.validate_response import validate_response_text

        raw = json.dumps({
            "schema_version": 1,
            "kind": "coder_followup",
            "state": "approved",
            "summary": "Fixed.",
            "addressed_items": ["item-1"],
            "remaining_items": [],
            "addressed_item_notes": {"item-1": "Implemented."},
            "remaining_item_notes": {},
            "human_requirements": {
                "addressed_ids": [],
                "checked_discussion_directly": False,
            },
            "tests_run": [],
        }) + "\n<!-- AGENT_STATE: approved -->\n-- Codex\nextra"
        monkeypatch.setattr(sr, "attempt_repair", lambda *a, **k: pytest.fail("repair not needed"))
        validate = lambda text: validate_response_text(
            text,
            kind="coder_followup",
            prior_items=[self._prior_item()],
        )

        recovered, parsed = sr._recover_structured_response(
            raw,
            expected_kind="coder_followup",
            validate=validate,
            unresolved_item_ids=["item-1"],
        )
        assert recovered.endswith("-- Codex")
        assert parsed.state == "approved"


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

    def test_failure_writes_failure_text_to_output(self, monkeypatch) -> None:
        # On a non-transient agent failure, run_external exits non-zero AND writes the
        # failure text to --output so a caller can classify it (#322).
        from coding_review_agent_loop.agents.base import AgentResult
        outcomes = [AgentResult(
            text="",
            raw_output="[ERROR] Invalid stream: empty response or malformed tool call",
            returncode=1,
        )]
        _calls, _sleeps, output_path, exit_code = self._invoke(
            monkeypatch, "gemini", outcomes, max_retries=0
        )
        assert exit_code == 1
        assert "Invalid stream" in Path(output_path).read_text(encoding="utf-8")

    def test_writes_minimal_response_evidence_sidecar(self, monkeypatch, tmp_path) -> None:
        import helpers.run_external as rex
        from coding_review_agent_loop.agents.base import AgentResult

        class FakeBackend:
            def run(self, runner, config, prompt):
                return AgentResult(
                    text="public",
                    response_file_text="public",
                    message_text="captured acknowledgement",
                )

        monkeypatch.setattr("coding_review_agent_loop.agents.codex.CodexBackend", FakeBackend)
        prompt = tmp_path / "prompt.md"
        output = tmp_path / "output.md"
        evidence = tmp_path / "evidence.json"
        prompt.write_text("Review this.", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", [
            "run_external",
            "--agent", "codex",
            "--prompt-file", str(prompt),
            "--output", str(output),
            "--workdir", str(tmp_path),
            "--response-evidence-output", str(evidence),
        ])

        rex.main()

        assert json.loads(evidence.read_text(encoding="utf-8")) == {
            "response_file_text": "public",
            "message_text": "captured acknowledgement",
        }


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
        base = dict(task="do X", task_file=None, repo="o/r", coder="codex",
                    plan_file=None, reviewers=["codex"], workdir=None,
                    workdir_codex=None, workdir_gemini=None, gemini_cmd="gemini",
                    antigravity_models=None, antigravity_quota_signatures=None,
                    model=None, dry_run=False)
        base.update(over)
        return _argparse.Namespace(**base)

    def test_task_round_reuses_existing_issue(self, monkeypatch) -> None:
        import helpers.skill_runner as sr
        monkeypatch.setattr(sr, "_lookup_task_issue", lambda repo, key: 77)
        created = {"n": 0}
        monkeypatch.setattr(sr, "_create_task_issue", lambda *a, **k: created.__setitem__("n", created["n"] + 1) or 999)
        seen = {}
        monkeypatch.setattr(sr, "cmd_run_plan_round", lambda args: seen.update(vars(args)))
        sr.cmd_run_task_round(self._task_args(
            coder="antigravity",
            antigravity_models=["Model A", "Model B"],
            antigravity_quota_signatures=["quota", "429"],
        ))
        assert created["n"] == 0          # did not create a duplicate
        assert seen["issue"] == 77        # delegated with the reused issue
        assert seen["coder"] == "antigravity"
        assert seen["antigravity_models"] == ["Model A", "Model B"]
        assert seen["antigravity_quota_signatures"] == ["quota", "429"]

    def test_task_round_creates_when_absent(self, monkeypatch) -> None:
        import helpers.skill_runner as sr
        monkeypatch.setattr(sr, "_lookup_task_issue", lambda repo, key: None)
        monkeypatch.setattr(sr, "_create_task_issue", lambda repo, text, key: 123)
        seen = {}
        monkeypatch.setattr(sr, "cmd_run_plan_round", lambda args: seen.update(vars(args)))
        sr.cmd_run_task_round(self._task_args(
            coder="gemini", gemini_cmd="/opt/gemini", workdir_gemini="/tmp/gemini-workdir",
        ))
        assert seen["issue"] == 123
        assert seen["coder"] == "gemini"
        assert seen["gemini_cmd"] == "/opt/gemini"
        assert seen["workdir_gemini"] == "/tmp/gemini-workdir"

    @pytest.mark.parametrize("coder", [None, "claude"])
    def test_task_round_claude_requires_plan_before_issue_lookup(
        self, monkeypatch, capsys, coder,
    ) -> None:
        import helpers.skill_runner as sr
        looked_up = {"n": 0}
        monkeypatch.setattr(
            sr, "_lookup_task_issue",
            lambda repo, key: looked_up.__setitem__("n", looked_up["n"] + 1),
        )
        args = self._task_args(coder="claude", plan_file=None)
        if coder is None:
            del args.coder
        with pytest.raises(SystemExit):
            sr.cmd_run_task_round(args)
        assert looked_up["n"] == 0
        assert "requires --plan-file" in capsys.readouterr().err

    def test_task_round_claude_requires_existing_plan_before_issue_lookup(
        self, monkeypatch, capsys, tmp_path,
    ) -> None:
        import helpers.skill_runner as sr
        looked_up = {"n": 0}
        monkeypatch.setattr(
            sr, "_lookup_task_issue",
            lambda repo, key: looked_up.__setitem__("n", looked_up["n"] + 1),
        )
        missing = tmp_path / "missing.md"
        with pytest.raises(SystemExit):
            sr.cmd_run_task_round(self._task_args(coder="claude", plan_file=str(missing)))
        assert looked_up["n"] == 0
        assert f"plan file not found: {missing}" in capsys.readouterr().err

    def test_task_round_claude_delegation_unchanged(self, monkeypatch, tmp_path) -> None:
        import helpers.skill_runner as sr
        plan = tmp_path / "plan.md"
        plan.write_text("plan", encoding="utf-8")
        monkeypatch.setattr(sr, "_lookup_task_issue", lambda repo, key: 77)
        seen = {}
        monkeypatch.setattr(sr, "cmd_run_plan_round", lambda args: seen.update(vars(args)))
        sr.cmd_run_task_round(self._task_args(coder="claude", plan_file=str(plan)))
        assert seen["issue"] == 77
        assert seen["coder"] == "claude"
        assert seen["plan_file"] == str(plan)

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


# ---------------------------------------------------------------------------
# reversed-roles helpers, behavior-neutral pieces (#307, sub-PR 1)
# ---------------------------------------------------------------------------

class TestReverseRolesHelpers:
    _ISSUE = {"number": 5, "repo": "o/r", "title": "t", "body": "b", "url": "u"}

    def test_build_plan_prompt_for_skill_embeds_coder_workdir(self) -> None:
        from helpers.prompt_builders import build_plan_prompt_for_skill
        p = build_plan_prompt_for_skill(
            self._ISSUE, repo="o/r", coder="codex",
            reviewers=["codex", "claude"], workdir="/tmp/skill-runner-codex",
        )
        assert "/tmp/skill-runner-codex" in p
        assert "5" in p  # issue number referenced

    def test_build_plan_revision_prompt_for_skill_includes_feedback(self) -> None:
        from helpers.prompt_builders import build_plan_revision_prompt_for_skill
        p = build_plan_revision_prompt_for_skill(
            self._ISSUE, repo="o/r", coder="codex",
            reviewers=["codex", "claude"], workdir="/tmp/skill-runner-codex",
            round_number=2, previous_plan="OLD PLAN TEXT",
            reviewer_feedback="REVIEWER FEEDBACK XYZ", prior_items_raw=[],
        )
        assert "REVIEWER FEEDBACK XYZ" in p
        assert "OLD PLAN TEXT" in p

    def test_attach_metadata_persists_raw_structured_coder_response(self) -> None:
        import re as _re
        from coding_review_agent_loop.round_state import _decode_round_metadata
        body = _write_tmp("## Revised plan\nrendered markdown body")
        raw = _write_tmp('{"kind": "plan_revision", "x": 1}', suffix=".json")
        out = _write_tmp("", suffix=".md")
        _run(
            "helpers.state_manager", "attach-metadata",
            "--body-file", body, "--output", out,
            "--flow", "plan", "--role", "coder", "--agent", "Codex",
            "--round-number", "2", "--state", "approved", "--subject", "abc123",
            "--raw-structured-coder-response-file", raw,
        )
        text = Path(out).read_text(encoding="utf-8")
        m = _re.search(r"AGENT_LOOP_META:\s*([A-Za-z0-9+/=_-]+)", text)
        assert m is not None
        meta = _decode_round_metadata(m.group(1))
        assert meta.agent == "Codex"
        assert meta.raw_structured_coder_response.strip() == '{"kind": "plan_revision", "x": 1}'


# ---------------------------------------------------------------------------
# helpers/skill_runner.py — reversed roles: external coder (#307)
# ---------------------------------------------------------------------------


class TestExternalCoderPhase:
    """Unit tests for the external-coder round state machine."""

    def _phase(self, resume: dict, reviewers=("codex",)) -> str:
        from helpers.skill_runner import _external_coder_phase
        return _external_coder_phase(resume, list(reviewers))["phase"]

    def test_no_coder_record_runs_round_1(self) -> None:
        assert self._phase({}) == "coder-round-1"

    def test_plan_posted_reviewers_incomplete_resumes_reviewers(self) -> None:
        resume = {
            "current_plan_subject": "abc",
            "completed_reviewer_names": [],
            "completed_reviewer_data": [],
            "prior_items": [],
            "round_number": 1,
        }
        assert self._phase(resume) == "resume-reviewers"

    def test_round_complete_all_approved_returns_approved(self) -> None:
        resume = {
            "current_plan_subject": "abc",
            "completed_reviewer_names": ["Codex"],
            "completed_reviewer_data": [
                {"reviewer_name": "Codex", "state": "approved", "new_items": [], "dispositions": []}
            ],
            "prior_items": [],
            "round_number": 1,
        }
        assert self._phase(resume) == "approved"

    def test_round_complete_blocking_runs_next_round(self) -> None:
        resume = {
            "current_plan_subject": "abc",
            "completed_reviewer_names": ["Codex"],
            "completed_reviewer_data": [
                {"reviewer_name": "Codex", "state": "blocking", "new_items": [], "dispositions": []}
            ],
            "prior_items": [],
            "round_number": 1,
        }
        assert self._phase(resume) == "coder-round-next"


class TestExternalCoderRun:
    """Subprocess dry-run tests for run-plan-round --coder codex (#307)."""

    def test_external_coder_dry_run_exits_zero_and_runs_coder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _write_fake_gh(tmppath)
            env = _make_fake_gh_env(tmppath)

            # No --plan-file: the external coder generates the plan.
            result = _run(
                "helpers.skill_runner", "run-plan-round",
                "--issue", "9997",
                "--repo", "test/skill-repo",
                "--coder", "codex",
                "--reviewers", "codex",
                "--dry-run",
                env=env,
            )
            assert result.returncode == 0, result.stderr
            stdout = result.stdout
            json_start = stdout.rfind("\n{")
            if json_start < 0:
                json_start = stdout.find("{")
            output = json.loads(stdout[json_start:].strip())
            assert "state" in output and "round_number" in output

            # The external coder turn saved a role:coder repair dir before validating.
            repair_dir = _REPAIR_BASE / "9997-r1-codex-coder"
            assert (repair_dir / "manifest.json").exists(), f"coder repair dir missing: {repair_dir}"
            manifest = json.loads((repair_dir / "manifest.json").read_text(encoding="utf-8"))
            assert manifest["role"] == "coder"
            assert manifest["kind"] == "plan_state"

    def test_claude_coder_without_plan_file_rejected(self) -> None:
        result = _run(
            "helpers.skill_runner", "run-plan-round",
            "--issue", "9997",
            "--repo", "test/skill-repo",
            "--reviewers", "codex",
            "--dry-run",
            check=False,
        )
        assert result.returncode != 0
        assert "requires --plan-file" in result.stderr

    def test_run_task_round_external_coder_dry_run_without_plan_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env["XDG_STATE_HOME"] = tmpdir
            result = _run(
                "helpers.skill_runner", "run-task-round",
                "--task", "do a thing",
                "--repo", "test/skill-repo",
                "--coder", "codex",
                "--reviewers", "codex",
                "--dry-run",
                env=env,
            )
        assert result.returncode == 0, result.stderr
        output = json.loads(result.stdout)
        assert output["would_create_issue"] is True


class TestRetryValidateCoder:
    """retry-validate routes role:coder manifests to the coder-completion path (#307)."""

    def _make_coder_repair_dir(self, tmpdir: Path, *, kind: str = "plan_state") -> Path:
        repair_dir = tmpdir / "9997-r1-codex-coder"
        repair_dir.mkdir(parents=True, exist_ok=True)
        (repair_dir / "raw.md").write_text(_VALID_PLAN_STATE, encoding="utf-8")
        (repair_dir / "prior_items.json").write_text("[]", encoding="utf-8")
        manifest = {
            "role": "coder",
            "agent": "codex",
            "agent_cap": "Codex",
            "flow": "plan",
            "issue": 9997,
            "repo": "OWNER/REPO",
            "new_round_number": 1,
            "kind": kind,
            "dry_run": True,
        }
        (repair_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return repair_dir

    def test_retry_validate_coder_plan_state_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _write_fake_gh(tmppath)
            env = _make_fake_gh_env(tmppath)
            repair_dir = self._make_coder_repair_dir(tmppath)
            result = _run(
                "helpers.skill_runner", "retry-validate",
                "--repair-dir", str(repair_dir),
                "--dry-run",
                env=env,
                check=False,
            )
            assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
            output = json.loads(result.stdout[result.stdout.find("{"):].strip())
            assert output["role"] == "coder"
            assert output["agent"] == "Codex"

    def test_retry_validate_structured_coder_plan_state_renders_public_comment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _write_fake_gh(tmppath)
            env = _make_fake_gh_env(tmppath)
            repair_dir = self._make_coder_repair_dir(tmppath)
            raw_plan = (
                json.dumps(
                    {
                        "kind": "plan_state",
                        "summary": "Plan the renderer fix.",
                        "plan_steps": ["Detect structured plan_state.", "Render markdown."],
                    }
                )
                + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- Codex"
            )
            (repair_dir / "raw.md").write_text(raw_plan, encoding="utf-8")

            result = _run(
                "helpers.skill_runner", "retry-validate",
                "--repair-dir", str(repair_dir),
                "--dry-run",
                env=env,
                check=False,
            )

            assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
            tagged = (repair_dir / "coder-tagged.md").read_text(encoding="utf-8")
            from coding_review_agent_loop.round_state import _strip_round_metadata

            public = _strip_round_metadata(tagged)
            assert public.startswith("## Plan")
            assert "### Plan steps\n1. Detect structured plan_state.\n2. Render markdown." in public
            assert '"kind": "plan_state"' not in public

            from coding_review_agent_loop.round_state import ROUND_RESUME_MARKER_RE, _decode_round_metadata

            match = ROUND_RESUME_MARKER_RE.search(tagged)
            assert match is not None
            metadata = _decode_round_metadata(match.group("payload"))
            assert metadata.canonical_plan == raw_plan
            assert metadata.raw_structured_coder_response == raw_plan

    def test_retry_validate_markdown_coder_plan_state_passes_through(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _write_fake_gh(tmppath)
            env = _make_fake_gh_env(tmppath)
            repair_dir = self._make_coder_repair_dir(tmppath)

            result = _run(
                "helpers.skill_runner", "retry-validate",
                "--repair-dir", str(repair_dir),
                "--dry-run",
                env=env,
                check=False,
            )

            assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
            tagged = (repair_dir / "coder-tagged.md").read_text(encoding="utf-8")
            from coding_review_agent_loop.round_state import ROUND_RESUME_MARKER_RE

            match = ROUND_RESUME_MARKER_RE.search(tagged)
            assert match is not None
            original_prefix, original_suffix = _VALID_PLAN_STATE.strip().split(
                "<!-- AGENT_PLAN_STATE: approved -->",
                1,
            )
            assert tagged[: match.start()].rstrip() == original_prefix.rstrip()
            assert tagged[match.end() :].strip() == (
                "<!-- AGENT_PLAN_STATE: approved -->" + original_suffix
            ).strip()

    def test_retry_validate_coder_invalid_plan_state_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            repair_dir = self._make_coder_repair_dir(tmppath)
            (repair_dir / "raw.md").write_text("no marker here", encoding="utf-8")
            result = _run(
                "helpers.skill_runner", "retry-validate",
                "--repair-dir", str(repair_dir),
                "--dry-run",
                check=False,
            )
            assert result.returncode != 0


# ---------------------------------------------------------------------------
# helpers/skill_runner.py — reversed roles: host-as-reviewer (#307)
# ---------------------------------------------------------------------------


class TestHostReviewer:
    def _last_json(self, stdout: str) -> dict:
        start = stdout.rfind("\n{")
        if start < 0:
            start = stdout.find("{")
        return json.loads(stdout[start:].strip())

    def test_external_coder_with_claude_reviewer_is_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _write_fake_gh(tmppath)
            env = _make_fake_gh_env(tmppath)
            result = _run(
                "helpers.skill_runner", "run-plan-round",
                "--issue", "9996",
                "--repo", "test/skill-repo",
                "--coder", "codex",
                "--reviewers", "claude",
                "--dry-run",
                env=env,
            )
            assert result.returncode == 0, result.stderr
            output = self._last_json(result.stdout)
            assert output["state"] == "pending"
            assert output["pending_reviewers"] == ["Claude"]

            # A host-review request dir was written with the plan + manifest.
            request_dir = _REPAIR_BASE / "9996-r1-claude"
            assert (request_dir / "plan.md").exists()
            manifest = json.loads((request_dir / "manifest.json").read_text(encoding="utf-8"))
            assert manifest["agent"] == "claude" and manifest["role"] == "reviewer"
            assert manifest["validate_kind"] == "plan_review"

    def test_external_reviewer_runs_before_host_review(self) -> None:
        """With both gemini and claude configured, the external reviewer still runs
        and only the host review is left pending."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _write_fake_gh(tmppath)
            env = _make_fake_gh_env(tmppath)
            result = _run(
                "helpers.skill_runner", "run-plan-round",
                "--issue", "9995",
                "--repo", "test/skill-repo",
                "--coder", "codex",
                "--reviewers", "claude", "gemini",
                "--dry-run",
                env=env,
            )
            assert result.returncode == 0, result.stderr
            output = self._last_json(result.stdout)
            assert output["state"] == "pending"
            assert output["pending_reviewers"] == ["Claude"]
            # Gemini (external) ran this round despite being listed after claude.
            assert "Gemini" in output["approved_reviewers"]

    def _make_host_review_dir(self, tmpdir: Path) -> Path:
        request_dir = tmpdir / "9996-r1-claude"
        request_dir.mkdir(parents=True, exist_ok=True)
        (request_dir / "plan.md").write_text(_VALID_PLAN_STATE, encoding="utf-8")
        (request_dir / "prior_items.json").write_text("[]", encoding="utf-8")
        (request_dir / "context.json").write_text(
            json.dumps({"reviewer": "Claude", "prior_items": [], "current_round_items": []}),
            encoding="utf-8",
        )
        (request_dir / "host-review.md").write_text(_VALID_PLAN_REVIEW_DRY, encoding="utf-8")
        manifest = {
            "role": "reviewer", "agent": "claude", "agent_cap": "Claude", "flow": "plan",
            "issue": 9996, "repo": "OWNER/REPO", "new_round_number": 1,
            "round_subject": "abc123", "item_id_offset": 0,
            "validate_kind": "plan_review", "dry_run": True,
        }
        (request_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return request_dir

    def test_complete_host_review_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _write_fake_gh(tmppath)
            env = _make_fake_gh_env(tmppath)
            request_dir = self._make_host_review_dir(tmppath)
            result = _run(
                "helpers.skill_runner", "complete-host-review",
                "--dir", str(request_dir),
                "--dry-run",
                env=env,
                check=False,
            )
            assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
            start = result.stdout.find("{")
            output = json.loads(result.stdout[start:].strip())
            assert output["reviewer_name"] == "Claude"
            assert output["state"] in ("approved", "blocking")

    def test_complete_host_review_missing_review_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            request_dir = self._make_host_review_dir(tmppath)
            (request_dir / "host-review.md").unlink()
            result = _run(
                "helpers.skill_runner", "complete-host-review",
                "--dir", str(request_dir),
                "--dry-run",
                check=False,
            )
            assert result.returncode != 0
            assert "host-review.md" in result.stderr


# ---------------------------------------------------------------------------
# helpers/skill_runner.py — host-as-reviewer for the PR flow (#314)
# ---------------------------------------------------------------------------

_VALID_PR_REVIEW_DRY = json.dumps(
    {
        "schema_version": 1,
        "kind": "pr_review",
        "state": "approved",
        "summary": "Host review: PR looks good.",
        "prior_item_dispositions": [],
    }
) + "\n<!-- AGENT_STATE: approved -->\n-- Claude\n"


class TestHostReviewerPR:
    def _last_json(self, stdout: str) -> dict:
        start = stdout.rfind("\n{")
        if start < 0:
            start = stdout.find("{")
        return json.loads(stdout[start:].strip())

    def test_pr_round_with_claude_reviewer_is_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _write_fake_gh(tmppath)
            env = _make_fake_gh_env(tmppath)
            result = _run(
                "helpers.skill_runner", "run-pr-round",
                "--pr", "9994",
                "--repo", "test/skill-repo",
                "--reviewers", "claude",
                "--dry-run",
                env=env,
            )
            assert result.returncode == 0, result.stderr
            output = self._last_json(result.stdout)
            assert output["state"] == "pending"
            assert output["pending_reviewers"] == ["Claude"]

            request_dir = _REPAIR_BASE / "9994-r1-claude"
            assert (request_dir / "pr-diff.diff").exists()
            manifest = json.loads((request_dir / "manifest.json").read_text(encoding="utf-8"))
            assert manifest["agent"] == "claude" and manifest["role"] == "reviewer"
            assert manifest["flow"] == "pr"
            assert manifest["validate_kind"] == "pr_review"

    def test_external_pr_reviewer_runs_before_host_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _write_fake_gh(tmppath)
            env = _make_fake_gh_env(tmppath)
            result = _run(
                "helpers.skill_runner", "run-pr-round",
                "--pr", "9993",
                "--repo", "test/skill-repo",
                "--reviewers", "claude", "codex",
                "--dry-run",
                env=env,
            )
            assert result.returncode == 0, result.stderr
            output = self._last_json(result.stdout)
            assert output["state"] == "pending"
            assert output["pending_reviewers"] == ["Claude"]
            assert "Codex" in output["approved_reviewers"]

    def _make_pr_host_review_dir(self, tmpdir: Path) -> Path:
        request_dir = tmpdir / "9994-r1-claude"
        request_dir.mkdir(parents=True, exist_ok=True)
        (request_dir / "pr-diff.diff").write_text("diff --git a/x b/x", encoding="utf-8")
        (request_dir / "prior_items.json").write_text("[]", encoding="utf-8")
        (request_dir / "context.json").write_text(
            json.dumps({"reviewer": "Claude", "prior_items": [], "current_round_items": []}),
            encoding="utf-8",
        )
        (request_dir / "host-review.md").write_text(_VALID_PR_REVIEW_DRY, encoding="utf-8")
        manifest = {
            "role": "reviewer", "agent": "claude", "agent_cap": "Claude", "flow": "pr",
            "issue": 9994, "repo": "OWNER/REPO", "new_round_number": 1,
            "round_subject": "abc123def", "item_id_offset": 0,
            "validate_kind": "pr_review", "material_filename": "pr-diff.diff",
            "dry_run": True,
        }
        (request_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return request_dir

    def test_complete_host_review_pr_dry_run_and_flow_aware_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _write_fake_gh(tmppath)
            env = _make_fake_gh_env(tmppath)
            request_dir = self._make_pr_host_review_dir(tmppath)
            result = _run(
                "helpers.skill_runner", "complete-host-review",
                "--dir", str(request_dir),
                "--dry-run",
                env=env,
                check=False,
            )
            assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
            output = json.loads(result.stdout[result.stdout.find("{"):].strip())
            assert output["reviewer_name"] == "Claude"
            # Flow-aware hint must point at run-pr-round, not run-plan-round.
            assert "run-pr-round" in result.stderr
            assert "run-plan-round" not in result.stderr

    def test_complete_host_review_pr_missing_file_references_pr_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            request_dir = self._make_pr_host_review_dir(tmppath)
            (request_dir / "host-review.md").unlink()
            result = _run(
                "helpers.skill_runner", "complete-host-review",
                "--dir", str(request_dir),
                "--dry-run",
                check=False,
            )
            assert result.returncode != 0
            assert "pr_review" in result.stderr

    def test_complete_host_review_help_is_flow_neutral(self) -> None:
        """The --dir CLI help must not be plan-only (it serves both flows)."""
        result = _run(
            "helpers.skill_runner", "complete-host-review", "--help",
            check=False,
        )
        assert result.returncode == 0
        assert "run-pr-round" in result.stdout
        assert "pr_review" in result.stdout


# ---------------------------------------------------------------------------
# helpers/skill_runner.py — reverse implementation: run-implement (#316)
# ---------------------------------------------------------------------------

_IMPL_WITH_PR = "Implemented the plan.\n\n<!-- AGENT_PR: 5 -->\n-- Codex\n"


def _human_req():
    from coding_review_agent_loop.github import HumanReviewRequirement
    return HumanReviewRequirement(
        source_type="Issue comment", author="wwind123",
        created_at="2026-06-14T00:00:00Z", url="https://example/1",
        body="Must keep the public API backward compatible.",
    )


class TestRunExternalImplStub:
    def test_coder_pr_dry_run_writes_pr_marker_stub(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.md"
            result = _run(
                "helpers.run_external",
                "--agent", "codex", "--role", "coder", "--flow", "pr",
                "--prompt-file", _write_tmp("implement it"),
                "--output", str(out), "--workdir", tmpdir,
                "--dry-run",
            )
            assert result.returncode == 0
            assert "<!-- AGENT_PR:" in out.read_text(encoding="utf-8")


class TestRunExternalDecomposeStub:
    def test_coder_decompose_dry_run_writes_valid_decomposition_stub(self) -> None:
        from coding_review_agent_loop.decomposition import parse_plan_decomposition

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.md"
            result = _run(
                "helpers.run_external",
                "--agent", "codex", "--role", "coder", "--flow", "decompose",
                "--prompt-file", _write_tmp("decompose it"),
                "--output", str(out), "--workdir", tmpdir,
                "--dry-run",
            )
            assert result.returncode == 0
            parsed = parse_plan_decomposition(out.read_text(encoding="utf-8"))
            assert parsed.phases[0].title == "Dry-run implementation phase"


class TestImplementationPrompt:
    def test_includes_plan_workdir_and_human_requirement(self) -> None:
        from coding_review_agent_loop.github import IssueContext
        from helpers.prompt_builders import build_implementation_prompt_for_skill
        ctx = IssueContext(
            number=42, repo="o/r", title="T", body="B", url="u",
            comments=(), human_requirements=(_human_req(),),
        )
        prompt = build_implementation_prompt_for_skill(
            ctx, "APPROVED PLAN BODY XYZ",
            repo="o/r", coder="codex", workdir="/tmp/skill-runner-codex",
            base="release-x",
        )
        assert "APPROVED PLAN BODY XYZ" in prompt
        assert "/tmp/skill-runner-codex" in prompt
        assert "release-x" in prompt  # PR-base instruction threads --base
        assert "backward compatible" in prompt  # signed human requirement surfaced


class TestDecompositionPrompt:
    def test_includes_plan_issue_context_and_workdir(self) -> None:
        from coding_review_agent_loop.github import IssueContext, IssueComment
        from helpers.prompt_builders import build_plan_decomposition_prompt_for_skill
        ctx = IssueContext(
            number=42,
            repo="o/r",
            title="Decompose this",
            body="Issue body ABC",
            url="https://example/42",
            comments=(IssueComment(author="wwind123", created_at="2026-06-14T00:00:00Z", body="Comment XYZ"),),
            human_requirements=(_human_req(),),
        )
        prompt = build_plan_decomposition_prompt_for_skill(
            ctx,
            "APPROVED PLAN BODY XYZ",
            repo="o/r",
            coder="codex",
            workdir="/tmp/skill-runner-codex",
        )
        assert "APPROVED PLAN BODY XYZ" in prompt
        assert "Issue body ABC" in prompt
        assert "Comment XYZ" in prompt
        assert "/tmp/skill-runner-codex" in prompt
        assert "backward compatible" in prompt


class TestValidateCoderImplementationResponse:
    def test_well_formed_returns_pr_number(self) -> None:
        from helpers.skill_runner import _validate_coder_implementation_response
        pr = _validate_coder_implementation_response(
            _IMPL_WITH_PR, workdir="/tmp/wd", human_requirements=(),
        )
        assert pr == 5

    def test_missing_human_ack_rejected(self) -> None:
        from coding_review_agent_loop.errors import AgentLoopError
        from helpers.skill_runner import _validate_coder_implementation_response
        with pytest.raises(AgentLoopError):
            _validate_coder_implementation_response(
                _IMPL_WITH_PR, workdir="/tmp/wd", human_requirements=(_human_req(),),
            )

    def test_out_of_workdir_test_rejected(self) -> None:
        from coding_review_agent_loop.errors import AgentLoopError
        from helpers.skill_runner import _validate_coder_implementation_response
        bad = (
            "Implemented the plan.\n"
            "Tests: cd /nonexistent/other-dir && pytest passed.\n\n"
            "<!-- AGENT_PR: 5 -->\n-- Codex\n"
        )
        with pytest.raises(AgentLoopError):
            _validate_coder_implementation_response(
                bad, workdir="/tmp/wd", human_requirements=(),
            )


# ---------------------------------------------------------------------------
# helpers/skill_runner.py — reverse implementation: run-pr-fix (#338)
# ---------------------------------------------------------------------------

def _pr_fix_item(status: str = "blocking") -> dict:
    return {
        "item_id": "item-1",
        "reviewer": "Codex",
        "source_round": 1,
        "text": "Fix the broken behavior.",
        "status": status,
        "source_status": status,
        "notes": [],
    }


def _pr_fix_resume(*, state: str = "blocking", reviewer_names: list[str] | None = None) -> dict:
    names = ["Codex"] if reviewer_names is None else reviewer_names
    return {
        "round_number": 1,
        "completed_round_number": 1,
        "current_plan_subject": "head-old",
        "prior_items": [],
        "compact_prior_summaries": ["old summary"],
        "completed_reviewer_names": names,
        "completed_reviewer_data": [
            {
                "reviewer_name": names[0],
                "state": state,
                "dispositions": [],
                "new_items": [_pr_fix_item()],
            }
        ] if names else [],
    }


def _structured_pr_fix_output() -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "kind": "coder_followup",
            "state": "blocking",
            "summary": "Fixed the blocking item.",
            "addressed_items": ["item-1"],
            "remaining_items": [],
            "addressed_item_notes": {"item-1": "Implemented the requested fix."},
            "remaining_item_notes": {},
            "tests_run": ["python3 -m pytest tests/test_skill_helpers.py -k run_pr_fix"],
            "human_requirements": {
                "addressed_ids": ["Requirement 1"],
                "checked_discussion_directly": True,
            },
        }
    ) + "\n<!-- AGENT_STATE: blocking -->\n-- Codex\n"


class TestRunPrFix:
    def test_settled_gate_requires_current_subject_complete_blocking_review(self) -> None:
        from helpers.skill_runner import _pr_fix_gate

        assert _pr_fix_gate(_pr_fix_resume(), ["codex"], "head-old") == (True, "")
        allowed, reason = _pr_fix_gate(_pr_fix_resume(state="approved"), ["codex"], "head-old")
        assert not allowed and "no blocking" in reason
        allowed, reason = _pr_fix_gate(_pr_fix_resume(reviewer_names=[]), ["codex"], "head-old")
        assert not allowed and "reviewer set" in reason
        allowed, reason = _pr_fix_gate(_pr_fix_resume(), ["codex"], "head-new")
        assert not allowed and "run-pr-round" in reason

    def test_empty_reviewer_data_noops_even_with_carried_prior_items(self) -> None:
        from helpers.skill_runner import _pr_fix_gate

        resume = _pr_fix_resume(reviewer_names=["Codex"])
        resume["completed_reviewer_data"] = []
        resume["prior_items"] = [_pr_fix_item()]
        allowed, reason = _pr_fix_gate(resume, ["codex"], "head-old")
        assert not allowed
        assert "reviewer data is unavailable" in reason

    def test_closed_pr_noops_before_resume_or_workdir(self, monkeypatch, capsys) -> None:
        import helpers.skill_runner as sr

        monkeypatch.setattr(sr, "_fetch_pr_json", lambda repo, pr: {
            "number": pr, "state": "CLOSED", "headRefOid": "head-old",
        })
        monkeypatch.setattr(sr, "_build_resume", lambda *a, **k: pytest.fail("resume should not run"))
        monkeypatch.setattr(sr, "_position_pr_fix_workdir", lambda *a, **k: pytest.fail("workdir should not mutate"))

        args = types.SimpleNamespace(
            pr=7, repo="o/r", coder="codex", reviewers=["codex"],
            workdir="/tmp/wd", workdir_codex=None, dry_run=False,
        )
        sr.cmd_run_pr_fix(args)
        output = json.loads(capsys.readouterr().out)
        assert output["state"] == "noop"
        assert output["current_pr_state"] == "CLOSED"

    def test_success_posts_metadata_after_head_and_assigned_head_advance(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        import helpers.skill_runner as sr
        from coding_review_agent_loop.github import (
            IssueContext,
            PullRequestMetadata,
            PullRequestReviewContext,
        )

        pr_infos = iter([
            {
                "number": 7,
                "state": "OPEN",
                "headRefOid": "head-old",
                "headRefName": "feature/pr",
                "body": "Fixes #338",
            },
            {
                "number": 7,
                "state": "OPEN",
                "headRefOid": "head-new",
                "headRefName": "feature/pr",
                "body": "Fixes #338",
            },
        ])
        helper_calls: list[tuple[str, ...]] = []

        def fake_run_helper(*args: str, check: bool = True):
            helper_calls.append(args)
            if args[:1] == ("helpers.run_external",):
                out = Path(args[args.index("--output") + 1])
                out.write_text(_structured_pr_fix_output(), encoding="utf-8")
                usage = Path(args[args.index("--usage-output") + 1])
                usage.write_text(json.dumps({"agent": "codex", "returncode": 0}), encoding="utf-8")
            elif args[:2] == ("helpers.state_manager", "attach-metadata"):
                body = Path(args[args.index("--body-file") + 1]).read_text(encoding="utf-8")
                out = Path(args[args.index("--output") + 1])
                out.write_text(body + "\n<!-- AGENT_LOOP_META: test -->\n", encoding="utf-8")
                assert "--raw-structured-coder-response-file" in args
                assert "--compact-prior-summaries-file" in args
            elif args[:2] == ("helpers.gh_ops", "post-issue-comment"):
                posted = Path(args[args.index("--file") + 1]).read_text(encoding="utf-8")
                assert "AGENT_LOOP_META" in posted
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr(sr, "_fetch_pr_json", lambda repo, pr: next(pr_infos))
        monkeypatch.setattr(sr, "_build_resume", lambda *a, **k: _pr_fix_resume())
        monkeypatch.setattr(sr, "_reconcile_pending_comment", lambda *a, **k: None)
        monkeypatch.setattr(sr, "_position_pr_fix_workdir", lambda **kwargs: None)
        monkeypatch.setattr(sr, "_run_helper", fake_run_helper)

        seen = {"count": 0}
        def fake_git_head(workdir: str) -> str:
            seen["count"] += 1
            return "head-old" if seen["count"] == 1 else "head-new"
        monkeypatch.setattr(sr, "_git_head", fake_git_head)

        import coding_review_agent_loop.github as gh
        monkeypatch.setattr(gh, "get_issue_context", lambda runner, config, issue_number: IssueContext(
            number=issue_number,
            repo="o/r",
            title="Issue",
            body="Body",
            url=None,
            comments=(),
            human_requirements=(_human_req(),),
        ))
        monkeypatch.setattr(gh, "get_pr_review_context", lambda runner, config, pr_number: PullRequestReviewContext(
            metadata=PullRequestMetadata(
                number=pr_number,
                repo="o/r",
                title="PR",
                head_branch="feature/pr",
                base_branch="main",
                head_sha="head-old",
                url=None,
            ),
            comments=(),
            human_requirements=(),
        ))

        args = types.SimpleNamespace(
            pr=7,
            repo="o/r",
            coder="codex",
            reviewers=["codex"],
            workdir=str(tmp_path),
            workdir_codex=None,
            antigravity_models=["Model A", "Model B"],
            antigravity_quota_signatures=["Quota Hit", "429"],
            dry_run=False,
        )
        sr.cmd_run_pr_fix(args)
        output = json.loads(capsys.readouterr().out)
        assert output["state"] == "blocking"
        assert output["previous_head_sha"] == "head-old"
        assert output["head_sha"] == "head-new"
        assert output["addressed_item_count"] == 1
        assert any(call[:2] == ("helpers.gh_ops", "post-issue-comment") for call in helper_calls)
        external_call = next(call for call in helper_calls if call[:1] == ("helpers.run_external",))
        assert external_call[external_call.index("--antigravity-models") + 1:] == (
            "Model A", "Model B",
            "--antigravity-quota-signatures", "Quota Hit", "429",
        )

    def test_missing_pr_marker_rejected(self) -> None:
        from coding_review_agent_loop.errors import AgentLoopError
        from helpers.skill_runner import _validate_coder_implementation_response
        with pytest.raises(AgentLoopError):
            _validate_coder_implementation_response(
                "I implemented it but forgot the marker.", workdir="/tmp/wd",
                human_requirements=(),
            )


class TestRunImplement:
    def _last_json(self, stdout: str) -> dict:
        start = stdout.rfind("\n{")
        if start < 0:
            start = stdout.find("{")
        return json.loads(stdout[start:].strip())

    def test_dry_run_runs_coder_and_prints_pr(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _write_fake_gh(tmppath)
            env = _make_fake_gh_env(tmppath)
            plan = tmppath / "plan.md"
            plan.write_text(_VALID_PLAN_STATE, encoding="utf-8")
            result = _run(
                "helpers.skill_runner", "run-implement",
                "--issue", "9992", "--repo", "test/skill-repo",
                "--coder", "antigravity", "--plan-file", str(plan),
                "--workdir", str(tmppath), "--base", "release-x",
                "--antigravity-models", "Model A", "Model B",
                "--dry-run",
                env=env,
            )
            assert result.returncode == 0, result.stderr
            output = self._last_json(result.stdout)
            assert output["pr"] == 0 and output["issue"] == 9992
            # Launcher base behavior is observable (not only the prompt text).
            assert "release-x" in result.stdout

    def test_invalid_coder_rejected(self) -> None:
        result = _run(
            "helpers.skill_runner", "run-implement",
            "--issue", "1", "--repo", "o/r", "--coder", "claude",
            "--plan-file", "/tmp/x.md", "--dry-run",
            check=False,
        )
        assert result.returncode != 0  # argparse choices reject 'claude'

    def test_missing_workdir_rejected_for_real_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = Path(tmpdir) / "plan.md"
            plan.write_text(_VALID_PLAN_STATE, encoding="utf-8")
            result = _run(
                "helpers.skill_runner", "run-implement",
                "--issue", "1", "--repo", "o/r", "--coder", "codex",
                "--plan-file", str(plan),
                check=False,
            )
            assert result.returncode != 0
            assert "push-capable" in result.stderr


# ---------------------------------------------------------------------------
# helpers/skill_runner.py — approved-plan decomposition: run-decompose (#318)
# ---------------------------------------------------------------------------

_DECOMP_JSON = json.dumps(
    {
        "schema_version": 1,
        "kind": "plan_decomposition",
        "phases": [
            {
                "title": "Schema helpers",
                "scope": "Add parser dataclasses and tests.",
                "non_goals": "No live orchestrator switch.",
                "dependency_notes": "First phase; no dependencies.",
                "rollout_risk": "low - internal only.",
                "validation": "Run python -m pytest tests/test_agent_loop.py.",
                "parent_context": "Approved plan slice: add schema helpers.",
                "automation": "agent-pr",
                "depends_on": [],
            }
        ],
    }
)


class TestRunDecompose:
    def _last_json(self, stdout: str) -> dict:
        start = stdout.rfind("\n{")
        if start < 0:
            start = stdout.find("{")
        return json.loads(stdout[start:].strip())

    def test_dry_run_runs_coder_parses_and_prints_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _write_fake_gh(tmppath)
            env = _make_fake_gh_env(tmppath)
            plan = tmppath / "plan.md"
            plan.write_text(_VALID_PLAN_STATE, encoding="utf-8")
            result = _run(
                "helpers.skill_runner", "run-decompose",
                "--issue", "9992", "--repo", "test/skill-repo",
                "--coder", "codex", "--plan-file", str(plan),
                "--workdir", str(tmppath),
                "--dry-run",
                env=env,
            )
            assert result.returncode == 0, result.stderr
            output = self._last_json(result.stdout)
            assert output["issue"] == 9992
            assert output["mode"] == "decompose-only"
            assert output["dry_run"] is True
            assert output["reused"] is False
            assert output["phase_count"] == 1
            assert output["phases"][0]["automation"] == "agent-pr"
            assert output["would_post_parent_summary"] is True

    def test_live_path_creates_children_and_posts_parent_summary(self, monkeypatch, tmp_path) -> None:
        import helpers.skill_runner as sr
        import coding_review_agent_loop.decomposition as decomp
        import coding_review_agent_loop.github as gh
        from coding_review_agent_loop.github import IssueContext

        raw_outputs: list[str] = []
        created_calls: list[dict] = []
        posted_calls: list[dict] = []

        def fake_run_helper(*args, **_kwargs):
            raw_outputs.append(" ".join(args))
            output = Path(args[args.index("--output") + 1])
            output.write_text(_DECOMP_JSON, encoding="utf-8")
            return subprocess.CompletedProcess(args, 0, "", "")

        def fake_get_issue_context(_runner, *, config, issue_number):
            return IssueContext(
                number=issue_number, repo=config.repo, title="T", body="B", url="u",
                comments=(), human_requirements=(),
            )

        def fake_create(runner, *, config, parent_issue, approved_plan, decomposition):
            created_calls.append({
                "parent_issue": parent_issue,
                "approved_plan": approved_plan,
                "phase_count": len(decomposition.phases),
            })
            phase = decomposition.phases[0]
            return (
                decomp.CreatedPhaseIssue(
                    phase=phase,
                    issue_url="https://github.com/test/skill-repo/issues/123",
                    issue_number=123,
                ),
            )

        def fake_post(runner, *, config, parent_issue, mode, plan_hash, created):
            posted_calls.append({
                "parent_issue": parent_issue,
                "mode": mode,
                "plan_hash": plan_hash,
                "created": created,
            })

        monkeypatch.setattr(sr, "_run_helper", fake_run_helper)
        monkeypatch.setattr(gh, "get_issue_context", fake_get_issue_context)
        monkeypatch.setattr(decomp, "create_decomposition_child_issues", fake_create)
        monkeypatch.setattr(decomp, "post_decomposition_parent_summary", fake_post)
        plan = tmp_path / "plan.md"
        plan.write_text("Approved plan body", encoding="utf-8")
        sr.cmd_run_decompose(types.SimpleNamespace(
            issue=77,
            repo="test/skill-repo",
            coder="codex",
            plan_file=str(plan),
            workdir=str(tmp_path),
            workdir_codex=None,
            workdir_gemini=None,
            dry_run=False,
        ))

        assert raw_outputs and "--flow decompose" in raw_outputs[0]
        assert created_calls == [{
            "parent_issue": 77,
            "approved_plan": "Approved plan body",
            "phase_count": 1,
        }]
        assert posted_calls[0]["parent_issue"] == 77
        assert posted_calls[0]["mode"] == "decompose-only"
        assert posted_calls[0]["created"][0].issue_number == 123

    def test_existing_marker_reuses_without_running_coder(self, monkeypatch, tmp_path, capsys) -> None:
        import helpers.skill_runner as sr
        import coding_review_agent_loop.decomposition as decomp
        import coding_review_agent_loop.github as gh
        from coding_review_agent_loop.github import IssueContext, IssueComment

        plan_text = "Approved plan body"
        marker = decomp.format_decomposition_parent_summary(
            parent_issue=77,
            mode="decompose-only",
            plan_hash=decomp.approved_plan_hash(plan_text),
            created=(
                decomp.CreatedPhaseIssue(
                    phase=decomp.RecordedPhase(title="Schema helpers", automation="agent-pr"),
                    issue_url="https://github.com/test/skill-repo/issues/123",
                    issue_number=123,
                ),
            ),
        )

        def fake_get_issue_context(_runner, *, config, issue_number):
            return IssueContext(
                number=issue_number, repo=config.repo, title="T", body="B", url="u",
                comments=(IssueComment(author="bot", created_at=None, body=marker),),
                human_requirements=(),
            )

        def fail_run_helper(*_args, **_kwargs):
            raise AssertionError("coder should not run when decomposition marker exists")

        monkeypatch.setattr(gh, "get_issue_context", fake_get_issue_context)
        monkeypatch.setattr(sr, "_run_helper", fail_run_helper)
        plan = tmp_path / "plan.md"
        plan.write_text(plan_text, encoding="utf-8")
        sr.cmd_run_decompose(types.SimpleNamespace(
            issue=77,
            repo="test/skill-repo",
            coder="codex",
            plan_file=str(plan),
            workdir=str(tmp_path),
            workdir_codex=None,
            workdir_gemini=None,
            dry_run=False,
        ))

        output = self._last_json(capsys.readouterr().out)
        assert output["reused"] is True
        assert output["phase_count"] == 1
        assert output["phases"][0]["issue_number"] == 123


# ---------------------------------------------------------------------------
# helpers/skill_runner.py — implement by phase: run-implement-by-phase (#319)
# ---------------------------------------------------------------------------

class TestRunImplementByPhase:
    def _last_json(self, stdout: str) -> dict:
        start = stdout.rfind("\n{")
        if start < 0:
            start = stdout.find("{")
        return json.loads(stdout[start:].strip())

    def _phase(self, *, automation: str = "agent-pr"):
        import coding_review_agent_loop.decomposition as decomp

        return decomp.PlanPhase(
            title="Schema helpers",
            scope="Add parser dataclasses and tests.",
            non_goals="No live orchestrator switch.",
            dependency_notes="First phase; no dependencies.",
            rollout_risk="low - internal only.",
            validation="Run python -m pytest tests/test_agent_loop.py.",
            parent_context="Approved plan slice: add schema helpers.",
            automation=automation,
            depends_on=(),
        )

    def _args(self, tmp_path, *, dry_run: bool = False):
        plan = tmp_path / "plan.md"
        plan.write_text("Approved parent plan", encoding="utf-8")
        return types.SimpleNamespace(
            issue=77,
            repo="test/skill-repo",
            coder="codex",
            plan_file=str(plan),
            workdir=str(tmp_path),
            workdir_codex=None,
            workdir_gemini=None,
            base="main",
            dry_run=dry_run,
        )

    def test_live_posts_handoff_before_child_implementation(self, monkeypatch, tmp_path, capsys) -> None:
        import helpers.skill_runner as sr
        import coding_review_agent_loop.decomposition as decomp
        import coding_review_agent_loop.github as gh
        from coding_review_agent_loop.github import IssueContext

        events: list[str] = []
        phase = self._phase()
        created = (
            decomp.CreatedPhaseIssue(
                phase=phase,
                issue_url="https://github.com/test/skill-repo/issues/123",
                issue_number=123,
            ),
        )
        parent_ctx = IssueContext(
            number=77, repo="test/skill-repo", title="Parent", body="Body", url="u",
            comments=(), human_requirements=(),
        )

        def fake_decompose(**_kwargs):
            return (
                {"reused": False},
                created,
                parent_ctx,
                types.SimpleNamespace(repo="test/skill-repo"),
                object(),
                "abc123",
                "Approved parent plan",
                str(tmp_path),
            )

        def fake_post(*_args, **kwargs):
            events.append(f"handoff:{kwargs['created'].issue_number}")

        def fake_get_issue_context(_runner, *, config, issue_number):
            assert issue_number == 123
            return IssueContext(
                number=123, repo=config.repo, title="Child", body="Child body", url="child",
                comments=(), human_requirements=(),
            )

        def fake_impl(**kwargs):
            events.append(f"implement:{kwargs['issue']}")
            assert kwargs["approved_plan"] == "Approved plan slice: add schema helpers."
            assert kwargs["post_one_shot_handoff"] is False
            return {"pr": 456, "head_sha": "sha", "issue": 123}

        monkeypatch.setattr(sr, "_run_decomposition_for_skill", fake_decompose)
        monkeypatch.setattr(decomp, "post_phase_implementation_handoff_comment", fake_post)
        monkeypatch.setattr(gh, "get_issue_context", fake_get_issue_context)
        monkeypatch.setattr(sr, "_run_child_or_one_shot_implementation", fake_impl)

        sr.cmd_run_implement_by_phase(self._args(tmp_path))

        assert events == ["handoff:123", "implement:123"]
        output = self._last_json(capsys.readouterr().out)
        assert output["state"] == "implemented"
        assert output["child_implementation"]["pr"] == 456

    def test_existing_phase_handoff_stops_without_implementation(self, monkeypatch, tmp_path, capsys) -> None:
        import helpers.skill_runner as sr
        import coding_review_agent_loop.decomposition as decomp
        from coding_review_agent_loop.github import IssueComment, IssueContext

        phase = self._phase()
        created = decomp.CreatedPhaseIssue(
            phase=phase,
            issue_url="https://github.com/test/skill-repo/issues/123",
            issue_number=123,
        )
        marker = decomp.format_phase_implementation_handoff_comment(
            parent_issue=77,
            mode="implement-by-phase",
            plan_hash="abc123",
            phase_index=1,
            created=created,
        )
        parent_ctx = IssueContext(
            number=77, repo="test/skill-repo", title="Parent", body="Body", url="u",
            comments=(IssueComment(author="bot", created_at=None, body=marker),),
            human_requirements=(),
        )

        def fake_decompose(**_kwargs):
            return (
                {"reused": True},
                (created,),
                parent_ctx,
                types.SimpleNamespace(repo="test/skill-repo"),
                object(),
                "abc123",
                "Approved parent plan",
                str(tmp_path),
            )

        def fail_impl(**_kwargs):
            raise AssertionError("implementation should not run when handoff exists")

        monkeypatch.setattr(sr, "_run_decomposition_for_skill", fake_decompose)
        monkeypatch.setattr(sr, "_run_child_or_one_shot_implementation", fail_impl)

        sr.cmd_run_implement_by_phase(self._args(tmp_path))

        output = self._last_json(capsys.readouterr().out)
        assert output["state"] == "handoff-exists"
        assert output["child_issue"] == 123

    def test_human_first_phase_stops_without_handoff(self, monkeypatch, tmp_path, capsys) -> None:
        import helpers.skill_runner as sr
        import coding_review_agent_loop.decomposition as decomp
        from coding_review_agent_loop.github import IssueContext

        phase = self._phase(automation="human-action")
        created = (
            decomp.CreatedPhaseIssue(
                phase=phase,
                issue_url="https://github.com/test/skill-repo/issues/123",
                issue_number=123,
            ),
        )
        parent_ctx = IssueContext(
            number=77, repo="test/skill-repo", title="Parent", body="Body", url="u",
            comments=(), human_requirements=(),
        )

        def fake_decompose(**_kwargs):
            return (
                {"reused": False},
                created,
                parent_ctx,
                types.SimpleNamespace(repo="test/skill-repo"),
                object(),
                "abc123",
                "Approved parent plan",
                str(tmp_path),
            )

        monkeypatch.setattr(sr, "_run_decomposition_for_skill", fake_decompose)
        sr.cmd_run_implement_by_phase(self._args(tmp_path))

        output = self._last_json(capsys.readouterr().out)
        assert output["state"] == "stopped"
        assert output["automation"] == "human-action"

    def test_missing_child_issue_number_fails_before_handoff(self, monkeypatch, tmp_path) -> None:
        import helpers.skill_runner as sr
        import coding_review_agent_loop.decomposition as decomp
        from coding_review_agent_loop.errors import AgentLoopError
        from coding_review_agent_loop.github import IssueContext

        phase = self._phase()
        created = (
            decomp.CreatedPhaseIssue(
                phase=phase,
                issue_url=None,
                issue_number=None,
            ),
        )
        parent_ctx = IssueContext(
            number=77, repo="test/skill-repo", title="Parent", body="Body", url="u",
            comments=(), human_requirements=(),
        )

        def fake_decompose(**_kwargs):
            return (
                {"reused": False},
                created,
                parent_ctx,
                types.SimpleNamespace(repo="test/skill-repo"),
                object(),
                "abc123",
                "Approved parent plan",
                str(tmp_path),
            )

        monkeypatch.setattr(sr, "_run_decomposition_for_skill", fake_decompose)
        with pytest.raises(AgentLoopError):
            sr.cmd_run_implement_by_phase(self._args(tmp_path))

    def test_dry_run_runs_decomposition_and_implementation_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _write_fake_gh(tmppath)
            env = _make_fake_gh_env(tmppath)
            plan = tmppath / "plan.md"
            plan.write_text(_VALID_PLAN_STATE, encoding="utf-8")
            result = _run(
                "helpers.skill_runner", "run-implement-by-phase",
                "--issue", "9992", "--repo", "test/skill-repo",
                "--coder", "codex", "--plan-file", str(plan),
                "--workdir", str(tmppath), "--base", "release-x",
                "--dry-run",
                env=env,
            )
            assert result.returncode == 0, result.stderr
            output = self._last_json(result.stdout)
            assert output["mode"] == "implement-by-phase"
            assert output["state"] == "would-implement"
            assert output["phase"]["automation"] == "agent-pr"
            assert output["would_post_phase_handoff"] is True
            assert output["child_implementation_preview"]["pr"] == 0


# ---------------------------------------------------------------------------
# helpers/skill_runner.py — resilient reviewer turns (#322)
# ---------------------------------------------------------------------------

_GEMINI_CLI_FAILURE = (
    'Error executing tool run_shell_command: Tool "run_shell_command" not found. '
    "Did you mean one of: grep_search?\n"
    "[ERROR] Invalid stream: The model returned an empty response or malformed tool call.\n"
)


class TestAgentUnavailableClassifier:
    def test_tooling_signature_is_unavailable(self) -> None:
        from helpers.skill_runner import _is_agent_unavailable_output
        assert _is_agent_unavailable_output(_GEMINI_CLI_FAILURE) is not None

    def test_empty_output_is_unavailable(self) -> None:
        from helpers.skill_runner import _is_agent_unavailable_output
        assert _is_agent_unavailable_output("   \n  ") == "empty response"

    def test_schema_invalid_json_is_not_unavailable(self) -> None:
        from helpers.skill_runner import _is_agent_unavailable_output
        # JSON but wrong schema -> repairable, not unavailable.
        assert _is_agent_unavailable_output('{"foo": "bar"}') is None

    def test_content_bearing_prose_is_not_unavailable(self) -> None:
        from helpers.skill_runner import _is_agent_unavailable_output
        # A plain-text review with actionable findings must stay on the repair path.
        prose = "This plan is missing error handling in step 3; that is blocking."
        assert _is_agent_unavailable_output(prose) is None


class TestRoundOverallState:
    def test_precedence(self) -> None:
        from helpers.skill_runner import _round_overall_state
        f = _round_overall_state
        # pending wins over everything (incl. an unavailable external reviewer).
        assert f(pending_reviewers=["Claude"], any_reviewer_blocked=True,
                 unavailable_reviewers=["Gemini"]) == "pending"
        assert f(pending_reviewers=["Claude"], any_reviewer_blocked=False,
                 unavailable_reviewers=["Gemini"]) == "pending"
        # then blocking, then incomplete (unavailable), then approved.
        assert f(pending_reviewers=[], any_reviewer_blocked=True,
                 unavailable_reviewers=["Gemini"]) == "blocking"
        assert f(pending_reviewers=[], any_reviewer_blocked=False,
                 unavailable_reviewers=["Gemini"]) == "incomplete"
        assert f(pending_reviewers=[], any_reviewer_blocked=False,
                 unavailable_reviewers=[]) == "approved"


class TestRunReviewerUnavailable:
    def _ctx(self):
        return {"reviewer": "Gemini", "prior_items": [], "current_round_items": []}

    def _call(self, tmp_path):
        import helpers.skill_runner as sr
        return sr._run_reviewer(
            agent="gemini", prompt_text="review it", context=self._ctx(),
            round_subject="subj", next_prior_items_raw=[], new_round_number=1,
            issue=1, repo="o/r", flow="plan", role="reviewer",
            state_key="plan_review", workdir=str(tmp_path), dry_run=False,
            tmpdir=tmp_path, item_id_offset=0,
        )

    def test_agent_failure_returns_unavailable_sentinel(self, monkeypatch, tmp_path) -> None:
        import helpers.skill_runner as sr

        # Simulate run_external exiting NON-ZERO (the empty/invalid-stream path) while
        # writing its failure text to --output, and honor check= so a check=True call
        # would abort. The fix passes check=False, so _run_reviewer must reach the
        # classifier and return the unavailable sentinel rather than SystemExit (#322).
        def fake_run_helper(*args, check=True, **_kw):
            if "helpers.run_external" in args:
                out = Path(args[args.index("--output") + 1])
                out.write_text(_GEMINI_CLI_FAILURE, encoding="utf-8")
                if check:
                    raise SystemExit(1)
                return subprocess.CompletedProcess(args, 1, "", "")
            return subprocess.CompletedProcess(args, 0, "", "")

        monkeypatch.setattr(sr, "_run_helper", fake_run_helper)
        record = self._call(tmp_path)
        assert record["state"] == "unavailable"
        assert record["reason"]
        assert record["reviewer_name"] == "Gemini"

    def test_content_bearing_failure_aborts_to_repair(self, monkeypatch, tmp_path) -> None:
        import helpers.skill_runner as sr

        def fake_run_helper(*args, **_kw):
            if "helpers.run_external" in args:
                out = Path(args[args.index("--output") + 1])
                # Content-bearing but invalid review -> repair path (SystemExit), not unavailable.
                out.write_text("I reviewed this and it looks fine, ship it.", encoding="utf-8")
            return subprocess.CompletedProcess(args, 0, "", "")

        monkeypatch.setattr(sr, "_run_helper", fake_run_helper)
        with pytest.raises(SystemExit):
            self._call(tmp_path)

    def test_round_continues_with_unavailable_external_and_pending_host(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        """gemini unavailable + claude host-review pending: the round does not abort,
        reports state=pending, and surfaces BOTH reviewer lists (no false approval,
        host handoff not hidden)."""
        import helpers.skill_runner as sr

        plan = tmp_path / "plan.md"
        plan.write_text(_VALID_PLAN_STATE, encoding="utf-8")

        monkeypatch.setattr(sr, "_build_resume", lambda *a, **k: {})
        monkeypatch.setattr(
            sr, "_fetch_issue_json",
            lambda *a, **k: {"number": 1, "title": "t", "body": "b", "url": "u"},
        )

        def fake_run_helper(*args, **_kw):
            # Only the external reviewer (gemini) goes through run_external here; make
            # it the Gemini-CLI failure so it is classified unavailable.
            if "helpers.run_external" in args:
                out = Path(args[args.index("--output") + 1])
                out.write_text(_GEMINI_CLI_FAILURE, encoding="utf-8")
            return subprocess.CompletedProcess(args, 0, "", "")

        monkeypatch.setattr(sr, "_run_helper", fake_run_helper)

        sr.cmd_run_plan_round(types.SimpleNamespace(
            issue=1, repo="o/r", reviewers=["gemini", "claude"],
            coder="claude", plan_file=str(plan), dry_run=True,
            agent_memory=False, refresh_agent_memory=False,
            workdir=str(tmp_path), workdir_codex=None, workdir_gemini=None,
        ))

        out = capsys.readouterr().out
        start = out.rfind("\n{")
        if start < 0:
            start = out.find("{")
        output = json.loads(out[start:].strip())
        assert output["state"] == "pending"
        assert output["pending_reviewers"] == ["Claude"]
        assert output["unavailable_reviewers"] == ["Gemini"]


# ---------------------------------------------------------------------------
# helpers — Antigravity (agy) agent in the skill (#215)
# ---------------------------------------------------------------------------


class TestAntigravitySkill:
    def _last_json(self, stdout: str) -> dict:
        start = stdout.rfind("\n{")
        if start < 0:
            start = stdout.find("{")
        return json.loads(stdout[start:].strip())

    def test_run_external_antigravity_dry_run_plan_review_stub(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.md"
            result = _run(
                "helpers.run_external",
                "--agent", "antigravity", "--role", "reviewer", "--flow", "plan",
                "--prompt-file", _write_tmp("review it"),
                "--output", str(out), "--workdir", tmpdir,
                "--dry-run",
            )
            assert result.returncode == 0
            val = _run(
                "helpers.validate_response", "--file", str(out), "--kind", "plan_review",
                check=False,
            )
            assert val.returncode == 0, f"{out.read_text()}\n{val.stderr}"

    @pytest.mark.parametrize("coder", ["antigravity", "agy"])
    def test_run_plan_round_antigravity_external_coder_dry_run(self, coder) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _write_fake_gh(tmppath)
            env = _make_fake_gh_env(tmppath)
            result = _run(
                "helpers.skill_runner", "run-plan-round",
                "--issue", "9991", "--repo", "test/skill-repo",
                "--coder", coder, "--reviewers", "codex",
                "--antigravity-models", "Model A", "Model B",
                "--dry-run",
                env=env,
            )
            assert result.returncode == 0, result.stderr
            output = self._last_json(result.stdout)
            assert "state" in output
            repair_dir = _REPAIR_BASE / "9991-r1-antigravity-coder"
            manifest = json.loads((repair_dir / "manifest.json").read_text(encoding="utf-8"))
            assert manifest["role"] == "coder"
            assert manifest["agent"] == "antigravity"

    def test_antigravity_reviewer_display_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _write_fake_gh(tmppath)
            env = _make_fake_gh_env(tmppath)
            plan = tmppath / "plan.md"
            plan.write_text(_VALID_PLAN_STATE, encoding="utf-8")
            result = _run(
                "helpers.skill_runner", "run-plan-round",
                "--issue", "9990", "--repo", "test/skill-repo",
                "--plan-file", str(plan), "--reviewers", "agy",
                "--model", "Model X",
                "--dry-run",
                env=env,
            )
            assert result.returncode == 0, result.stderr
            output = self._last_json(result.stdout)
            assert "Antigravity" in output["approved_reviewers"]

    @pytest.mark.parametrize(
        ("extra_args", "expected_models", "expected_signatures"),
        [
            (
                (),
                ("Gemini 3.1 Pro (High)", "Gemini 3.5 Flash (High)"),
                ("quota", "rate limit", "resource exhausted", "RESOURCE_EXHAUSTED", "429"),
            ),
            (("--model", "Model X"), ("Model X",), ("quota", "rate limit", "resource exhausted", "RESOURCE_EXHAUSTED", "429")),
            (
                ("--antigravity-models", "Model A", "Model B"),
                ("Model A", "Model B"),
                ("quota", "rate limit", "resource exhausted", "RESOURCE_EXHAUSTED", "429"),
            ),
            (
                ("--antigravity-quota-signatures", "Quota Hit", "429"),
                ("Gemini 3.1 Pro (High)", "Gemini 3.5 Flash (High)"),
                ("Quota Hit", "429"),
            ),
        ],
    )
    def test_run_external_resolves_antigravity_config(
        self,
        monkeypatch,
        tmp_path,
        extra_args,
        expected_models,
        expected_signatures,
    ) -> None:
        import helpers.run_external as rex
        from coding_review_agent_loop.agents.base import AgentResult

        captured = {}

        class FakeBackend:
            def run(self, runner, config, prompt):
                captured["config"] = config
                return AgentResult(text="review complete", returncode=0)

        monkeypatch.setattr(
            "coding_review_agent_loop.agents.antigravity.AntigravityBackend",
            FakeBackend,
        )
        prompt = tmp_path / "prompt.md"
        output = tmp_path / "output.md"
        prompt.write_text("review this", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", [
            "run_external",
            "--agent", "agy",
            "--prompt-file", str(prompt),
            "--output", str(output),
            "--workdir", str(tmp_path),
            "--max-retries", "0",
            *extra_args,
        ])

        rex.main()

        config = captured["config"]
        assert config.reviewer == ("antigravity",)
        assert config.antigravity_models == expected_models
        assert config.antigravity_quota_signatures == expected_signatures

    def test_run_external_rejects_single_model_and_chain_together(self, tmp_path) -> None:
        result = _run(
            "helpers.run_external",
            "--agent", "antigravity",
            "--prompt-file", _write_tmp("review it"),
            "--output", str(tmp_path / "out.md"),
            "--workdir", str(tmp_path),
            "--model", "Model X",
            "--antigravity-models", "Model A", "Model B",
            "--dry-run",
            check=False,
        )
        assert result.returncode != 0
        assert "not allowed with argument" in result.stderr

    @pytest.mark.parametrize(
        ("namespace", "expected"),
        [
            (types.SimpleNamespace(), ()),
            (types.SimpleNamespace(model="Model X"), ("--model", "Model X")),
            (
                types.SimpleNamespace(antigravity_models=["Model A", "Model B"]),
                ("--antigravity-models", "Model A", "Model B"),
            ),
            (
                types.SimpleNamespace(
                    antigravity_models=["Model A", "Model B"],
                    antigravity_quota_signatures=["Quota Hit", "429"],
                ),
                (
                    "--antigravity-models", "Model A", "Model B",
                    "--antigravity-quota-signatures", "Quota Hit", "429",
                ),
            ),
        ],
    )
    def test_skill_runner_converts_antigravity_options(self, namespace, expected) -> None:
        from helpers.skill_runner import _run_external_antigravity_args

        assert _run_external_antigravity_args(namespace) == expected

    def test_skill_runner_rejects_single_model_and_chain_together(self) -> None:
        result = _run(
            "helpers.skill_runner", "run-plan-round",
            "--issue", "1",
            "--repo", "o/r",
            "--plan-file", _write_tmp(_VALID_PLAN_STATE),
            "--reviewers", "antigravity",
            "--model", "Model X",
            "--antigravity-models", "Model A", "Model B",
            "--dry-run",
            check=False,
        )
        assert result.returncode != 0
        assert "not allowed with argument" in result.stderr


# ---------------------------------------------------------------------------
# render_response --model stamps the dynamic signature (#332)
# ---------------------------------------------------------------------------


def test_run_task_round_forwards_antigravity_options():
    """run-task-round applies _add_antigravity_options and delegates the full args
    namespace to the plan round, so the antigravity model/chain/quota options forward
    to run_external (#347)."""
    import argparse
    import helpers.skill_runner as sr

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    p_task = sub.add_parser("run-task-round")
    sr._add_antigravity_options(p_task)

    args = parser.parse_args([
        "run-task-round",
        "--antigravity-models", "Gemini 3.1 Pro (High)", "Gemini 3.5 Flash (High)",
        "--antigravity-quota-signatures", "quota", "429",
    ])
    assert sr._run_external_antigravity_args(args) == (
        "--antigravity-models", "Gemini 3.1 Pro (High)", "Gemini 3.5 Flash (High)",
        "--antigravity-quota-signatures", "quota", "429",
    )
    # Legacy single-model override forwards as --model.
    legacy = parser.parse_args(["run-task-round", "--model", "Gemini 3.1 Pro (High)"])
    assert sr._run_external_antigravity_args(legacy) == ("--model", "Gemini 3.1 Pro (High)")


def test_model_used_from_usage_sidecar(tmp_path):
    import helpers.skill_runner as sr
    sidecar = tmp_path / "agent-usage.json"
    assert sr._model_used_from_usage(sidecar) == ""  # missing file
    sidecar.write_text(json.dumps({"model_used": "Gemini 3.1 Pro (High)"}), encoding="utf-8")
    assert sr._model_used_from_usage(sidecar) == "Gemini 3.1 Pro (High)"
    sidecar.write_text(json.dumps({"usage": {}}), encoding="utf-8")  # no model_used key
    assert sr._model_used_from_usage(sidecar) == ""
    sidecar.write_text("not json", encoding="utf-8")  # unreadable
    assert sr._model_used_from_usage(sidecar) == ""


def test_render_response_plan_revision_stamps_model_signature():
    revision = json.dumps(
        {
            "schema_version": 1,
            "kind": "plan_revision",
            "state": "blocking",
            "summary": "Revised plan.",
            "prior_plan_item_dispositions": [],
            "plan_steps": ["Step A"],
        }
    ) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Google Antigravity\n"
    path = _write_tmp(revision)
    ctx = _write_tmp(json.dumps({"prior_items": [], "current_round_items": []}), suffix=".json")
    out = _write_tmp("", suffix=".md")
    result = _run(
        "helpers.render_response",
        "--file", path, "--kind", "plan_revision",
        "--reviewer", "Antigravity", "--context-file", ctx,
        "--output", out, "--model", "Gemini 3.1 Pro (High)",
    )
    assert result.returncode == 0, result.stderr
    rendered = Path(out).read_text(encoding="utf-8")
    assert "Google Antigravity: Gemini 3.1 Pro (High)" in rendered


def test_render_response_coder_followup_stamps_model_signature():
    followup = json.dumps(
        {
            "schema_version": 1,
            "kind": "coder_followup",
            "state": "approved",
            "summary": "Addressed feedback.",
            "addressed_items": [],
            "remaining_items": [],
            "human_requirements": {"addressed_ids": [], "checked_discussion_directly": False},
        }
    ) + "\n<!-- AGENT_STATE: approved -->\n-- Google Antigravity\n"
    path = _write_tmp(followup)
    ctx = _write_tmp(json.dumps({"prior_items": [], "current_round_items": []}), suffix=".json")
    out = _write_tmp("", suffix=".md")
    result = _run(
        "helpers.render_response",
        "--file", path, "--kind", "coder_followup",
        "--reviewer", "Antigravity", "--context-file", ctx,
        "--output", out, "--model", "Gemini 3.1 Pro (High)",
    )
    assert result.returncode == 0, result.stderr
    rendered = Path(out).read_text(encoding="utf-8")
    assert "Google Antigravity: Gemini 3.1 Pro (High)" in rendered
