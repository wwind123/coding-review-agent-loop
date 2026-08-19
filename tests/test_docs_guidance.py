import re
from pathlib import Path

from coding_review_agent_loop.cli import build_parser

REPO_ROOT = Path(__file__).parent.parent
LOCAL_AGENT_LOOP_DOC = REPO_ROOT / "docs" / "local_agent_loop.md"
README = REPO_ROOT / "README.md"

HEADING_TEXT = "Phased decomposition versus split materialization"
CI_STALL_HEADING_TEXT = "External CI infrastructure stalls"
MANAGED_CI_HEADING_TEXT = "Managed exact-head CI"
LOCAL_TEST_SCOPE_HEADING_TEXT = "Focused, bounded local test selection"
SKILL_LOCAL_TEST_SCOPE_HEADING_TEXT = "Gates & guardrails"
SKILL_MODE_LOCAL_TEST_SCOPE_HEADING_TEXT = "Focused, bounded local test runs"
SKILL = REPO_ROOT / "SKILL.md"
SKILL_MODE_DOC = REPO_ROOT / "docs" / "skill_mode.md"


def _github_anchor(heading_text: str) -> str:
    """Approximate GitHub's markdown heading-to-anchor slug algorithm."""
    slug = heading_text.lower()
    slug = re.sub(r"[^\w\- ]", "", slug)
    slug = slug.replace(" ", "-")
    return slug


def test_local_agent_loop_doc_has_decision_heading_and_table():
    text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    assert f"### {HEADING_TEXT}" in text
    assert "`--plan-execution-mode decompose-only`" in text
    assert "`--plan-execution-mode implement-by-phase`" in text
    assert "`--materialize-split-issues`" in text


def test_local_agent_loop_doc_warns_about_combining_mechanisms():
    text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    section_start = text.index(f"### {HEADING_TEXT}")
    next_heading = text.index("### Split issue materialization", section_start)
    section = text[section_start:next_heading]

    assert "Do not combine" in section
    assert "decompose-only" in section
    assert "implement-by-phase" in section
    assert "duplicate" in section.lower()


def test_readme_links_to_decision_section_with_derived_anchor():
    doc_text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    assert f"### {HEADING_TEXT}" in doc_text, "heading moved; update HEADING_TEXT"
    expected_anchor = _github_anchor(HEADING_TEXT)

    readme_text = README.read_text(encoding="utf-8")
    normalized_readme_text = " ".join(readme_text.split())
    assert f"docs/local_agent_loop.md#{expected_anchor}" in readme_text
    assert "`--plan-execution-mode decompose-only`" in readme_text
    assert "`--materialize-split-issues`" in readme_text
    assert "duplicate children" in normalized_readme_text


def test_cli_help_points_to_decision_section():
    parser = build_parser()
    issue_parser = None
    discuss_parser = None
    for action in parser._actions:
        if hasattr(action, "choices") and isinstance(action.choices, dict):
            issue_parser = action.choices.get("issue", issue_parser)
            discuss_parser = action.choices.get("discuss", discuss_parser)
    assert issue_parser is not None
    assert discuss_parser is not None

    def _help_for(subparser, option_string):
        for sub_action in subparser._actions:
            if option_string in getattr(sub_action, "option_strings", []):
                return sub_action.help
        raise AssertionError(f"{option_string} not found in parser")

    plan_execution_mode_help = _help_for(issue_parser, "--plan-execution-mode")
    issue_materialize_help = _help_for(issue_parser, "--materialize-split-issues")
    discuss_materialize_help = _help_for(discuss_parser, "--materialize-split-issues")

    anchor = "phased-decomposition-versus-split-materialization"
    for help_text in (plan_execution_mode_help, issue_materialize_help, discuss_materialize_help):
        assert anchor in help_text


def test_local_agent_loop_doc_has_ci_infrastructure_stall_section():
    text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    assert f"### {CI_STALL_HEADING_TEXT}" in text
    assert "`--ci-queued-grace-seconds`" in text
    assert "queued_too_long" in text
    assert "runner_unavailable" in text


def test_readme_links_to_ci_infrastructure_stall_section():
    doc_text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    assert f"### {CI_STALL_HEADING_TEXT}" in doc_text, "heading moved; update CI_STALL_HEADING_TEXT"
    expected_anchor = _github_anchor(CI_STALL_HEADING_TEXT)

    readme_text = README.read_text(encoding="utf-8")
    assert f"docs/local_agent_loop.md#{expected_anchor}" in readme_text
    assert "`--ci-queued-grace-seconds`" in readme_text


def test_readme_links_to_managed_exact_head_ci_and_scopes_watch_mode():
    doc_text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    assert f"### {MANAGED_CI_HEADING_TEXT}" in doc_text
    expected_anchor = _github_anchor(MANAGED_CI_HEADING_TEXT)

    readme_text = README.read_text(encoding="utf-8")
    normalized_readme_text = " ".join(readme_text.split())
    assert f"docs/local_agent_loop.md#{expected_anchor}" in readme_text
    assert "`--match-head-commit`" in readme_text
    assert "merges only that qualified SHA" in normalized_readme_text
    assert (
        "`--watch-pending-ci` and `--no-watch-pending-ci` do not alter"
        in normalized_readme_text
    )


def test_managed_ci_docs_describe_explicit_existing_pr_adoption_and_opt_out():
    text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    assert "#### Optional adoption of an existing PR" in text
    assert "--managed-ci-adopt-existing-pr" in text
    assert "AGENT_LOOP_MANAGED_CI_V2_PR_ADOPTION" in text
    assert "agent-loop-managed-opt-out" in text
    assert "triage/write collaborators" in text


def test_managed_ci_docs_describe_safe_issue_draft_resume_and_fallback():
    text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    assert "agent-loop pr <number> --auto-merge" in text
    assert "supported resume, not retroactive adoption" in text
    assert "editable body nonce never grants authority" in text
    assert "previous invocation's outcome" in text
    assert "invocation-owned fallback draft" in text
    assert "remains ready" in text


def test_local_agent_loop_doc_has_focused_bounded_local_test_section():
    text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    assert f"### {LOCAL_TEST_SCOPE_HEADING_TEXT}" in text
    section_start = text.index(f"### {LOCAL_TEST_SCOPE_HEADING_TEXT}")
    next_heading = text.index("## Agent Permission Flags", section_start)
    section = " ".join(text[section_start:next_heading].split())

    assert "External CI infrastructure" in section
    assert "distinct from" in section
    assert "1,800 seconds" in section
    assert section.count("1,800 seconds") == 1
    assert "must not launch pytest in the background" in section
    assert "poll process" in section
    assert "`ps`/`kill -0`/`wait`" in section
    assert "task-output files" in section
    assert "a human or the issue explicitly asked for full-suite" in section
    assert "no-PR `AGENT_STATE: blocking`" in section
    assert "agent_unavailable" in section
    assert "three-snapshot" in section
    assert "120-second GitHub CI observation limit" in section
    assert "`--test-command` parsing and execution path" in section
    assert "`--antigravity-print-timeout-seconds` above the 1,800-second command cap" in section
    assert "default 600-second whole-invocation deadline cannot accommodate" in section
    assert "additional budget for analysis, edits, reporting, and other turn work" in section
    assert "other backend-imposed whole-turn deadline" in section


def test_skill_docs_keep_focused_bounded_local_test_policy_aligned():
    skill_text = SKILL.read_text(encoding="utf-8")
    skill_section_start = skill_text.index(f"## {SKILL_LOCAL_TEST_SCOPE_HEADING_TEXT}")
    skill_section_end = skill_text.index("\n---", skill_section_start)
    skill_section = " ".join(skill_text[skill_section_start:skill_section_end].split())

    skill_mode_text = SKILL_MODE_DOC.read_text(encoding="utf-8")
    skill_mode_section_start = skill_mode_text.index(f"## {SKILL_MODE_LOCAL_TEST_SCOPE_HEADING_TEXT}")
    skill_mode_section_end = skill_mode_text.index("## Session state", skill_mode_section_start)
    skill_mode_section = " ".join(skill_mode_text[skill_mode_section_start:skill_mode_section_end].split())

    for section in (skill_section, skill_mode_section):
        assert section.count("1,800 seconds per required test command") == 1
        assert "individually justified command" in section
        assert "not a reason to choose a broad suite" in section
        assert "foreground with visible output" in section
        assert "background" in section
        assert "`ps`/`kill -0`/`wait`" in section
        assert "naming the exact command and the timeout" in section


def test_cli_help_documents_ci_queued_grace_seconds():
    parser = build_parser()
    pr_parser = None
    for action in parser._actions:
        if hasattr(action, "choices") and isinstance(action.choices, dict):
            pr_parser = action.choices.get("pr", pr_parser)
    assert pr_parser is not None

    help_text = None
    for sub_action in pr_parser._actions:
        if "--ci-queued-grace-seconds" in getattr(sub_action, "option_strings", []):
            help_text = sub_action.help
    assert help_text is not None
    assert "1200" in help_text
