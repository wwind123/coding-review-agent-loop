import re
import subprocess
from pathlib import Path

import pytest

from coding_review_agent_loop.architecture_context import (
    _linked_detail_paths,
    _sanitize_architecture_text,
    acquire_architecture_snapshot,
    render_architecture_snapshot,
)
from coding_review_agent_loop.cli import build_parser
from coding_review_agent_loop.runner import Runner

REPO_ROOT = Path(__file__).parent.parent
LOCAL_AGENT_LOOP_DOC = REPO_ROOT / "docs" / "local_agent_loop.md"
README = REPO_ROOT / "README.md"
ARCHITECTURE = REPO_ROOT / "ARCHITECTURE.md"
DECISIONS_DIR = REPO_ROOT / "docs" / "architecture" / "decisions"

HEADING_TEXT = "Phased decomposition versus split materialization"
CI_STALL_HEADING_TEXT = "External CI infrastructure stalls"
MANAGED_CI_HEADING_TEXT = "Managed exact-head CI"
LOCAL_TEST_SCOPE_HEADING_TEXT = "Focused, bounded local test selection"
SKILL_LOCAL_TEST_SCOPE_HEADING_TEXT = "Gates & guardrails"
SKILL_MODE_LOCAL_TEST_SCOPE_HEADING_TEXT = "Focused, bounded local test runs"
README_SKILL_MODE_HEADING_TEXT = "Claude Code Skill Mode"
SKILL = REPO_ROOT / "SKILL.md"
SKILL_MODE_DOC = REPO_ROOT / "docs" / "skill_mode.md"


def _github_anchor(heading_text: str) -> str:
    """Approximate GitHub's markdown heading-to-anchor slug algorithm."""
    slug = heading_text.lower()
    slug = re.sub(r"[^\w\- ]", "", slug)
    slug = slug.replace(" ", "-")
    return slug


def _architecture_set_paths() -> list[Path]:
    """The root plus the detail docs it links to, as injection discovers them."""
    members = _linked_detail_paths(ARCHITECTURE.read_text(encoding="utf-8"), "ARCHITECTURE.md")
    assert members is not None, "the CommonMark parser is required to discover members"
    return [ARCHITECTURE, *(REPO_ROOT / member for member in members)]


_DECISION_ENTRY = re.compile(
    r"^- (?:(?!Superseded by #)[^\n])+?\(#\d+(?:, #\d+)*(?:\)\.|, \d{4}-\d{2}-\d{2}\)\.?)"
    r"(?: Superseded by #\d+: (?:(?!Superseded by #)[^\n])+?\.)*$"
)


def _decision_entry_is_valid(line: str) -> bool:
    """A decision-log bullet: an issue reference, then only appended supersession notes."""
    return _DECISION_ENTRY.match(line) is not None


def _decision_log_bullets(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("- ")]


def _architecture_set_text() -> str:
    return "\n\n".join(path.read_text(encoding="utf-8") for path in _architecture_set_paths())


def test_architecture_is_discoverable_from_existing_guides():
    assert "(ARCHITECTURE.md)" in README.read_text(encoding="utf-8")
    for path in (LOCAL_AGENT_LOOP_DOC, SKILL_MODE_DOC):
        assert "(../ARCHITECTURE.md)" in path.read_text(encoding="utf-8")
    # Preserve the old inbound anchor while the overview moves to its own file.
    assert "## Architecture\n" in LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")


def test_architecture_links_and_component_paths_exist():
    for document in _architecture_set_paths():
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if target.startswith("https://"):
                continue
            path_text, _, anchor = target.partition("#")
            path = document.parent / path_text if path_text else document
            assert path.exists(), (document.name, target)
            if anchor:
                headings = re.findall(r"^#+ (.+)$", path.read_text(encoding="utf-8"), re.M)
                assert anchor in {_github_anchor(heading) for heading in headings}, (document.name, target)
    root_map = ARCHITECTURE.read_text(encoding="utf-8").split("## Component map\n", 1)[1].split("\n## ", 1)[0]
    components = (REPO_ROOT / "docs" / "architecture" / "components.md").read_text(encoding="utf-8")
    table = components.split("## Component Map\n", 1)[1].split("\n## ", 1)[0]
    for module in re.findall(r"`([\w/]+\.py)`", root_map + table):
        assert (REPO_ROOT / "src" / "coding_review_agent_loop" / module).is_file(), module


def test_architecture_root_is_a_bounded_overview_linking_every_detail_doc():
    root = ARCHITECTURE.read_text(encoding="utf-8")
    assert len(root) <= 10_000
    assert not re.search(r"#\d+", root)
    detail_dir = REPO_ROOT / "docs" / "architecture"
    on_disk = {path.relative_to(REPO_ROOT).as_posix() for path in detail_dir.glob("*.md")}
    members = {path.relative_to(REPO_ROOT).as_posix() for path in _architecture_set_paths()[1:]}
    assert on_disk and members == on_disk
    for name in ("README.md", "docs/local_agent_loop.md", "docs/skill_mode.md"):
        assert f"]({name})" in root


def test_architecture_detail_docs_share_one_shape():
    for path in _architecture_set_paths()[1:]:
        raw = path.read_bytes()
        assert len(raw) < 65_536, path.name
        paragraphs = [block.strip() for block in raw.decode("utf-8").split("\n\n") if block.strip()]
        assert paragraphs[0].startswith("# ") and "\n" not in paragraphs[0], path.name
        assert not paragraphs[1].startswith(("#", "Source paths:")), path.name
        assert paragraphs[2].startswith("Source paths: "), path.name
        for source in re.findall(r"`([^`]+)`", paragraphs[2]):
            assert (REPO_ROOT / source).exists(), (path.name, source)
        h2 = re.findall(r"^## (.+)$", raw.decode("utf-8"), re.M)
        assert h2 and h2[-1] == "Decisions", path.name
        decisions = raw.decode("utf-8").rsplit("\n## Decisions\n", 1)[1]
        assert decisions.strip() == f"[Decision log](decisions/{path.name})", path.name
        bullets = _decision_log_bullets(DECISIONS_DIR / path.name)
        assert bullets and all(_decision_entry_is_valid(line) for line in bullets), path.name


def test_architecture_decision_logs_pair_with_detail_docs():
    detail_stems = {path.stem for path in (REPO_ROOT / "docs" / "architecture").glob("*.md")}
    log_stems = {path.stem for path in DECISIONS_DIR.glob("*.md")}
    assert detail_stems and log_stems == detail_stems
    for stem in sorted(log_stems):
        text = (DECISIONS_DIR / f"{stem}.md").read_text(encoding="utf-8")
        headings = re.findall(r"^(#+) ", text, re.M)
        assert headings == ["#"], stem
        assert f"](../{stem}.md)" in text, stem
        assert not re.search(r"^[ \t]+[-*+] ", text, re.M), stem
        bullets = _decision_log_bullets(DECISIONS_DIR / f"{stem}.md")
        assert bullets, stem
        invalid = [line for line in bullets if not _decision_entry_is_valid(line)]
        assert not invalid, (stem, invalid)


@pytest.mark.parametrize(
    ("line", "valid"),
    [
        ("- X (#1).", True),
        ("- X (#1, #2).", True),
        ("- X (#1). Superseded by #3: y.", True),
        ("- **T.** X (#1, 2026-10-15)", True),
        ("- **T.** X (#1, 2026-10-15).", True),
        ("- **T.** X (#1, 2026-10-15) Superseded by #3: y.", True),
        ("- X (#1). Superseded by #3: y. Superseded by #4: z.", True),
        ("- X no ref.", False),
        ("- X (#1, 2026-1-5)", False),
        ("- X. Superseded by #3: y. (#1).", False),
        ("- X Superseded by #3: y. (#1).", False),
        ("- X (1).", False),
    ],
)
def test_decision_entry_validator_accepts_both_forms(line, valid):
    assert _decision_entry_is_valid(line) is valid


def test_architecture_root_never_links_into_decision_logs():
    root = ARCHITECTURE.read_text(encoding="utf-8")
    members = _linked_detail_paths(root, "ARCHITECTURE.md")
    assert members is not None
    assert not [member for member in members if member.startswith("docs/architecture/decisions/")]
    for target in re.findall(r"\]\(([^)\s]+)", root):
        resolved = (REPO_ROOT / target.split("#", 1)[0]).resolve()
        assert not resolved.is_relative_to(DECISIONS_DIR.resolve()), target
    assert "`docs/architecture/decisions/`" in root


def test_default_render_injects_the_whole_root_and_every_detail_doc(tmp_path):
    def git(*args):
        return subprocess.run(
            ("git", *args), cwd=tmp_path, text=True, capture_output=True, check=True
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    for path in _architecture_set_paths():
        target = tmp_path / path.relative_to(REPO_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    git("add", "-A")
    git("commit", "-qm", "architecture set")
    snapshot = acquire_architecture_snapshot(
        Runner(), checkout=tmp_path, repository="owner/repo", revision=git("rev-parse", "HEAD")
    )
    assert snapshot.availability == "available"
    rendered = render_architecture_snapshot(snapshot)
    assert "Overview status: truncated" not in rendered
    assert "Architecture overview truncated" not in rendered
    assert _sanitize_architecture_text(ARCHITECTURE.read_text(encoding="utf-8")) in rendered
    assert "Detail documents in the architecture set" in rendered
    for path in _architecture_set_paths()[1:]:
        assert f"- {path.relative_to(REPO_ROOT).as_posix()} — " in rendered
    # Every detail doc opens with a summary, and the default budget still
    # fits each one in the index (#1499).
    assert len(snapshot.members) == len(_architecture_set_paths()) - 1
    for member in snapshot.members:
        assert member.summary
        assert f"- {member.path} — {member.first_heading}: {member.summary}" in rendered


def test_local_agent_loop_doc_has_decision_heading_and_table():
    text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    assert f"### {HEADING_TEXT}" in text
    assert "`--plan-execution-mode decompose-only`" in text
    assert "`--plan-execution-mode implement-by-phase`" in text
    assert "`--plan-execution-mode auto`" in text
    assert "`--materialize-split-issues`" in text


def test_docs_explain_reviewed_child_execution_dispositions():
    local = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    architecture = _architecture_set_text()
    for text in (local, readme, architecture):
        assert "direct-implementation" in text
        assert "requires-child-planning" in text
        assert "planner" in text.lower()
        assert "review" in text.lower()
    assert "override metadata\ncannot supply it" in local
    assert "reviewed child plan" in architecture


def test_local_agent_loop_doc_warns_about_combining_mechanisms():
    text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    section_start = text.index(f"### {HEADING_TEXT}")
    next_heading = text.index("### Split issue materialization", section_start)
    section = text[section_start:next_heading]

    assert "Do not combine" in section
    assert "decompose-only" in section
    assert "implement-by-phase" in section
    assert "auto" in section
    assert "duplicate" in section.lower()


def test_readme_links_to_decision_section_with_derived_anchor():
    doc_text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    assert f"### {HEADING_TEXT}" in doc_text, "heading moved; update HEADING_TEXT"
    expected_anchor = _github_anchor(HEADING_TEXT)

    readme_text = README.read_text(encoding="utf-8")
    normalized_readme_text = " ".join(readme_text.split())
    assert f"docs/local_agent_loop.md#{expected_anchor}" in readme_text
    assert "`--plan-execution-mode decompose-only`" in readme_text
    assert "`auto`" in readme_text
    assert "`--materialize-split-issues`" in readme_text
    assert "duplicate children" in normalized_readme_text


def test_readme_documents_same_repo_concurrency_limit():
    readme_text = README.read_text(encoding="utf-8")
    section_start = readme_text.index("## Current Limitations")
    section_end = readme_text.index("## Planning and Decomposition", section_start)
    section = " ".join(readme_text[section_start:section_end].split())

    assert "claims every agent checkout it uses" in section
    assert "refused by name" in section
    assert "explicit `--{agent}-dir`" in section
    assert "repo-scoped local state" in section
    assert "never lets a second run use a claimed one" in section
    assert "`--review-parallel` is supported within one orchestrator run" in section
    assert "system temporary directory (`/tmp` on Linux)" in section


def test_readme_documents_review_and_approval_semantics():
    readme_text = README.read_text(encoding="utf-8")
    section_start = readme_text.index('### What "review" and "approval" mean')
    section_end = readme_text.index("## Current Limitations", section_start)
    section = " ".join(readme_text[section_start:section_end].split())

    assert "not a native GitHub pull-request review" in section
    assert (
        "does not satisfy a branch-protection rule that requires approving GitHub reviews"
        in section
    )
    assert (
        "The agent CLIs normally share the GitHub identity authenticated through `gh`, "
        "so their model signatures identify protocol participants rather than distinct "
        "GitHub accounts."
        in section
    )


def test_readme_documents_ci_watcher_controls():
    readme_text = README.read_text(encoding="utf-8")
    section_start = readme_text.index("## CI and Merge")
    section_end = readme_text.index("## Claude Code Skill Mode", section_start)
    section = " ".join(readme_text[section_start:section_end].split())

    assert "`--ci-timeout-seconds` (default 1200)" in section
    assert "`--ci-poll-interval-seconds` (default 30)" in section


def test_readme_describes_tests_directory_as_test_modules():
    readme_text = README.read_text(encoding="utf-8")
    section_start = readme_text.index("## Development")
    section_end = readme_text.index("## Related Tools", section_start)
    section = " ".join(readme_text[section_start:section_end].split())

    assert "Browse the focused test modules in [`tests/`](tests/)" in section
    assert "module map in [`tests/`](tests/)" not in section


def test_skill_links_to_current_readme_skill_mode_heading():
    readme_text = README.read_text(encoding="utf-8")
    assert f"## {README_SKILL_MODE_HEADING_TEXT}" in readme_text
    expected_anchor = _github_anchor(README_SKILL_MODE_HEADING_TEXT)

    skill_text = SKILL.read_text(encoding="utf-8")
    assert f"README.md#{expected_anchor}" in skill_text


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


def test_issue_recovery_docs_describe_commit_provenance_retirement():
    text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    readme_text = README.read_text(encoding="utf-8")
    assert "Agent-Issue-Provenance" in text
    assert "Metadata-free recovery is retired" in text
    assert "unauthenticated convention" in text
    assert "Squash or rebase removal" in text
    assert "managed-pr --head" in readme_text


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


def test_watch_docs_define_full_board_policy_and_compatibility_behavior():
    text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    normalized_text = " ".join(text.split())
    readme_text = README.read_text(encoding="utf-8")
    retired_option = "--ci-check-name"

    assert "ordinary `--auto-merge` always enters the full-board watcher" in normalized_text
    assert "reliable, non-empty current-head board" in normalized_text
    assert "shared across all watcher entries" in normalized_text
    assert "Auto-merge timeout, `not_started`, and pre-poll exhaustion" in normalized_text
    assert f"legacy single `{retired_option}` waiter" not in normalized_text
    assert retired_option not in text
    assert retired_option not in readme_text


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


def test_managed_ci_docs_describe_lifecycle_gated_issue_resume_and_base_provenance():
    text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    readme_text = README.read_text(encoding="utf-8")
    for document in (text, readme_text):
        normalized_document = document.lower()
        assert "canonical issue" in normalized_document
        assert "ready/unlabeled" in normalized_document
        assert "explicit `--managed-ci`" in normalized_document
        assert "historical audit" in normalized_document or "historical records" in normalized_document
        assert "base provenance" in normalized_document
    assert "prints the exact flow-preserving retry" in text
    assert "never asks the operator to delete durable records" in text


def test_managed_ci_docs_describe_v2_lifecycle_isolation_and_recovery():
    text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "`prepared`, `dispatch-requested`, `attached`, and `completed`" in normalized
    assert "`run_id: null` and `run_attempt: null`" in normalized
    assert "terminal_outcome: \"no-status\"" in normalized
    assert "entire run ID remains excluded" in normalized
    assert "temporarily empty" in normalized
    assert "unsupported lifecycle value" in normalized


def test_managed_ci_docs_describe_precreation_managed_pr_mode():
    text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    assert "agent-loop managed-pr" in text
    assert "agent-loop/managed-direct-*" in text
    assert "never adopts an already-open PR" in text
    assert "--allow-unprotected-managed-ci" in text


def test_coder_docs_require_github_body_files():
    text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    normalized_text = " ".join(text.split())

    assert (
        "Every coder `gh` body must be written to a temporary file outside the checkout"
        in text
    )
    assert "passed with `--body-file`" in text
    assert (
        "`gh pr create --draft --label agent-loop-managed --body-file <path> --base <base>`"
        in normalized_text
    )


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
    assert "`--antigravity-print-timeout-seconds` above the selected command watchdog" in section
    assert "`--coder-test-command-timeout-seconds`" in section
    assert "default 600-second whole-invocation deadline may be too short" in section
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
        assert section.count("1,800 seconds by default") == 1
        assert "configured finite run-level ceiling" in section
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


def test_architecture_states_cross_row_selector_reuse_policy():
    # #865: cross-row reuse is valid; #927: repetition within one claim drops
    # that claim instead of rejecting the response.
    text = " ".join(_architecture_set_text().split())
    assert "duplicate admissible selectors" not in text
    assert "One admissible selector may be cited by several rows" in text
    assert "an admissible selector repeated within one claim" in text


def test_docs_list_tool_owned_machine_readable_record_types():
    doc_text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    readme_text = README.read_text(encoding="utf-8")
    record_types = (
        "managed-CI issue authorization record",
        "fresh re-authorization",
        "managed-CI exact-head intent record",
        "unprotected-override audit",
        "resume-provenance audit",
        "managed-CI qualified-head record",
        "plan-validation diagnostic record",
    )
    for document in (doc_text, readme_text):
        for record_type in record_types:
            assert record_type in document, record_type
        assert "machine-readable and must be kept" in document
        assert "marker-only copies" in document
        assert "remain valid" in document
    assert "### Tool-owned protocol records" in doc_text
    assert "issue-to-PR handoff" in doc_text and "expected-closing contract" in doc_text


def test_architecture_records_label_and_read_back_invariants():
    text = _architecture_set_text()
    assert "deterministic visible-label invariant" in text
    assert "shared write read-back verifier" in text
    assert "historical bare-marker body" in text
    assert "producing login and ID" in text


def test_docs_describe_reviewer_repair_admission_and_grounding():
    """Issue #871: the canonical documents name the refusal boundary."""
    doc = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    for fragment in (
        "Reviewer repair is refused before any backend call",
        "field unique",
        "bare protocol state footer",
        "agent-unavailable",
        "empty-response",
        "grounding check",
        "Support is token\ncoverage",
        "matched injectively",
        "per-modifier occurrence counts",
        "contraction normalization",
        "negation-safe coverage predicate",
        "manufacture an approval",
    ):
        assert fragment in doc, fragment

    # Issue #871 round 7, item-8: the empty-finding rule must be stated once.
    # An earlier round left a stale sentence saying an empty finding is a
    # candidate that corresponds to an exempt-only target, directly next to the
    # implemented rule that a genuinely empty entry is dropped. A reader of the
    # security boundary cannot be handed both claims, so the superseded wording
    # is asserted absent while the implemented rule is asserted present.
    normalized = " ".join(doc.split())
    assert "genuinely empty entry such as `{}` is dropped" in normalized
    assert (
        "A declared source finding that carries no prose is a candidate only "
        "when it really was nothing but a reserved marker" in normalized
    )
    assert (
        "it corresponds solely to its own authorized neutralization" in normalized
    )
    assert (
        "Every other exempt-only target is refused, whichever candidate it is "
        "matched against" in normalized
    )
    assert (
        "compared by marker identity and occurrence count over the registry's "
        "historical replacement spans" in normalized
    )
    assert "represent the COMPLETE source marker multiset" in normalized
    assert (
        "a target matched to a substantive candidate must therefore retain "
        "substantive content of its own" in normalized
    )
    assert "An empty or marker-only source finding" not in normalized
    assert (
        "a lead-in line before the first bullet is list structure that is "
        "dropped" in normalized
    )
    assert "joins the first item" not in normalized

    architecture = _architecture_set_text()
    assert "Reviewer repair is refused fail-closed" in architecture
    assert "a refusal is a reviewer unavailability, never a synthesized verdict" in architecture
    assert "proposal to evaluate, never an" in architecture


def test_docs_describe_conflict_round_continuity_exception():
    # #829: the conflict-resolution round advances the head with no reviewer,
    # so both the canonical and operator contracts must name that transition.
    arch_text = " ".join(_architecture_set_text().split())
    doc_text = " ".join(LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8").split())
    assert "merge-conflict resolution round" in arch_text
    assert "tool-owned merge-conflict obligation" in arch_text
    assert "outside the coder's classifiable item namespace" in arch_text
    assert "resume reauthenticates the same shape" in arch_text
    assert "Two reviewer-less rounds are the only exceptions" in doc_text
    assert "tool-owned merge-conflict obligation instead of a review pair" in doc_text
    # #1024: the exact-head CI repair round is the second reviewer-less transition.
    assert "exact-head CI repair round (#1024) is the second such transition" in arch_text
    assert "An exact-head CI repair round is the second exception" in doc_text
    assert "tool-minted CI obligation instead of a review pair" in doc_text
    for text in (arch_text, doc_text):
        assert "awaiting_current_head_review" in text
        assert "still fails closed" in text
        assert "approve the exact final head before qualification or merge" in text


def test_docs_describe_rejected_coder_recovery_continuity_exception():
    # #1367: the rejected-coder recovery across a fresh grant is the third
    # continuity exception, and its chain is authenticated end to end.
    arch_text = " ".join(_architecture_set_text().split())
    doc_text = " ".join(LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8").split())
    assert "A rejected-coder recovery across a fresh grant is the third exception" in doc_text
    assert "The predecessor cutoff is not relaxed" in doc_text
    assert "exactly one attempt-2 recovery coder-dispatch record" in doc_text
    assert "counts toward \"exactly one\" wherever it sits" in doc_text
    assert "only the original attempt-1 dispatch of the original head is exempt" in doc_text
    assert "must carry that board's seat binding" in doc_text
    assert "on the current approved plan only" in doc_text
    assert "A recovery across a signed plan rebind is therefore refused" in doc_text
    assert "An older binary cannot re-authenticate such a chain and stops" in doc_text
    assert "A rejected-coder recovery across a fresh grant (#1367) is a third shape" in arch_text
    assert "the exact rejected-coder recovery chain still fails closed" in arch_text
    assert "the PR stays draft and unmerged and no approval is synthesized" in doc_text


def test_architecture_documents_the_planning_scheduler():
    """`scope-5` (#905, from #841): the canonical overview is updated."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    text = _architecture_set_text()

    assert "plan_review_scheduling.py" in text
    assert "exact-plan candidate key" in text
    assert "reviewer-only phase-advance round" in text
    assert "plan-phase-advance" in text
    assert "four disjoint classes" in text
    assert "only a transport extraction failure stops the" in text
    assert "run, checked at startup and at every round boundary" in text
    assert "--plan-review-policy" in text
    assert "](../local_agent_loop.md#staged-issue-plan-review)" in text


def test_operator_docs_document_the_staged_planning_flags_and_limits():
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / "docs" / "local_agent_loop.md").read_text()
    readme = (root / "README.md").read_text()

    assert "### Staged issue plan review" in text
    for flag in (
        "--plan-review-policy",
        "--primary-plan-reviewer",
        "--plan-review-force-full",
        "--plan-primary-stall-rounds",
        "--plan-reset-stall-streak",
    ):
        assert flag in text
        assert flag in readme
    # #1112: edit-based retirement, the comment-only limit, legacy rounds.
    assert "edit the issue title or body" in text
    assert "comment alone does not" in text
    assert "predate issue-text tracking" in text
    # #1103: the stall stop, its upgrade note, and the contract identity.
    assert "#### Primary-phase stall stop" in text
    assert "**Upgrade note:**" in text
    assert "contract identity" in text
    architecture = _architecture_set_text()
    assert "primary-phase stall stop" in architecture
    assert "--plan-primary-stall-rounds" in architecture
    assert "scheduler_issue_digest" in architecture
    assert "scheduler_stall_reset" in architecture
    assert "scheduler_step_back_deferral" in architecture
    assert "whichever of that stop and the stall stop is reached first" not in architecture
    assert "Whichever stop is reached first wins" not in readme
    assert "Comment-only narrowing never retires" in architecture
    assert "cannot retire them" in readme
    assert "a comment alone does not" in readme
    for phase in (
        "`primary`",
        "`secondary-audit`",
        "`remediation`",
        "`final-secondary-sweep`",
        "`full-board`",
    ):
        assert phase in text
    # Round budget, candidate key, classifier, panel evidence, fallbacks.
    assert "Raise\n`--max-rounds` when enabling it." in text
    assert "generation-1" in text
    assert "`recheck`" in text and "`narrow`" in text and "`broad`" in text
    assert "qualified panel opening" in text
    assert "strict pre-panel fallback:" in text
    assert "post-panel fallback:" in text
    # The four degraded-history classes and the override's recovery limits.
    for label in ("A `absent`", "B `invalid`", "C `contradictory-key`", "D transport failure"):
        assert label in text
    assert "recovers the first three\nonly" in text
    assert "never recover class D" in text
    # Carried approvals and the exclusions.
    assert "HUMAN_REQUIREMENTS_RESOLVED" in text
    assert "Discussion-mode scheduling always invokes the full board" in text
    assert "A child-planning cycle inherits the operator's `--plan-review-policy`" in text


def test_docs_document_the_flow_aware_planning_policy_evaluation():
    text = LOCAL_AGENT_LOOP_DOC.read_text()
    architecture = _architecture_set_text()
    readme = README.read_text()

    # The flow dimension, its backward-compatible default, and per-flow
    # uniqueness, aggregation, and titling.
    assert "`flow` is `pr` or `plan` and defaults to `pr` only when the key is absent" in text
    assert "is rejected rather\nthan assigned to the PR rows" in text
    assert "`(flow, policy, run_id)`" in text
    assert "Frozen plan review policy evaluation" in text
    assert "`selective-intermediate` is PR-only" in text
    assert "validated\n`flow` (`pr` or `plan`; only an absent key defaults to `pr`" in architecture
    assert "`(flow, policy, run_id)`" in architecture
    # The stale "no flow dimension today" caveat is replaced by the comparison
    # the planning rows now support, with its provenance limit stated.
    assert "no flow dimension today" not in text
    assert "escaped plan defects are reported in\nthe `plan` flow" in text
    assert "`unavailable` with a reason rather than borrowed" in text
    assert "own `plan` flow" in readme


def test_docs_document_the_review_contract_dimension_and_freezing_procedure():
    text = " ".join(LOCAL_AGENT_LOOP_DOC.read_text().split())
    architecture = " ".join(_architecture_set_text().split())
    readme = " ".join(README.read_text().split())

    # The reviewer rule (#894) and that it changes no schema.
    assert "Reviews are exhaustive." in text
    assert "substantiating one blocking defect does not end the review" in text
    assert "masking must not be claimed merely to stop early" in text
    assert "The rule changes no response schema" in text
    # The label, its default, rejection, and label provenance.
    assert "An absent key defaults to `first-finding-permitted`" in text
    assert "An explicitly present null, non-string, blank, or unknown value is rejected" in text
    assert "`flows.<flow>.review_contracts.<contract>.policies.<policy>`" in text
    assert "verified `review_contract_provenance`" in text
    assert "no frozen runs for this review contract and policy" in text
    # Reading rules and the absence of any pooled figure.
    assert "Read the effect of the contract only within the same flow and the same scheduling policy." in text
    assert "no pooled contract figure is produced" in text
    assert "Fewer rounds are an improvement only if they were not bought with missed defects." in text
    # The two artifact pairs never mix.
    assert "It is never extended with real runs" in text
    assert "`docs/evaluation/review_contract_runs.json`" in text
    assert "`docs/evaluation/review_contract_report.json`" in text
    # The freezing procedure: label evidence, exclusion, window, regeneration.
    assert "#### Freezing a real run" in LOCAL_AGENT_LOOP_DOC.read_text()
    assert "belongs to neither contract and must not be frozen under either label" in text
    assert "`review_contract_provenance.source`" in text
    assert "fixed escaped-defect observation window of 14 days after the run's PR merge, identical for both contracts" in text
    assert "`metric_provenance.escaped_defects` entry" in text
    assert "may be frozen as `verified: true` only after the window has closed" in text
    assert (
        "agent-loop review-evaluation docs/evaluation/review_contract_runs.json \\ "
        "--output docs/evaluation/review_contract_report.json"
    ) in text
    for document in (architecture, readme):
        assert "`review_contract`" in document
        assert "`first-finding-permitted`" in document
        assert "review_contract_runs.json" in document
    assert "no per-flow or cross-policy contract rollup" in architecture
    assert "never pooled across policies" in readme


def _readme_sandboxed_example() -> list[str]:
    import shlex

    text = README.read_text(encoding="utf-8")
    section = text.split("### Running where GitHub GraphQL is refused", 1)[1].split("\n### ", 1)[0]
    blocks = re.findall(r"```bash\n(.*?)```", section, re.S)
    examples = [block for block in blocks if "--agent-permissions sandboxed" in block]
    assert len(examples) == 1
    return shlex.split(examples[0].replace("\\\n", " "))


def test_readme_sandboxed_example_passes_sandboxed_validation(tmp_path):
    from coding_review_agent_loop import agent_permissions
    from coding_review_agent_loop.config import config_from_args

    from agent_loop_helpers import FakeRunner

    tokens = _readme_sandboxed_example()
    assert tokens[0] == "agent-loop"
    args = build_parser().parse_args([
        *tokens[1:],
        "--claude-dir", str(tmp_path / "claude"),
        "--codex-dir", str(tmp_path / "codex"),
        "--subprocess-log-dir", str(tmp_path / "logs"),
        "--dry-run",
    ])
    config = config_from_args(args, FakeRunner())
    config = type(config)(**{**config.__dict__, "plan_execution_mode": args.plan_execution_mode})
    assert config.agent_permissions == "sandboxed"
    assert config.repair_backend == "claude" and config.repair_models == ("MODEL",)
    assert config.semantic_followup_backend == "claude"
    agent_permissions.validate_sandboxed_selections(config)
    agent_permissions.validate_sandboxed_flow(config, command=args.command, plan_first=args.plan_first)


def test_readme_sandboxed_step_documents_requirements_and_limits():
    text = README.read_text(encoding="utf-8")
    step = text.split("**6. Give each agent only the access its role needs.**", 1)[1].split(
        "**These settings do not persist.**", 1
    )[0]
    for phrase in (
        "--no-semantic-followup-dedupe",
        "default repair backend\n(`antigravity`)",
        "Install agent-loop outside the agent checkouts",
        "`git` and `gh` found first on `PATH` must also live outside every\n  checkout",
        "`TMPDIR` must resolve outside every checkout",
        "Pass-through agent arguments are rejected",
        "A committing Codex coder is rejected",
        "Codex non-coders have no network",
        "closed environment\n  allowlist",
        "refuses to run when the checkout's\n  local, worktree, or included config has a key outside its allowlist",
        "same OS user",
        "`agy --sandbox` restricts only its terminal",
        "trade of review depth for\n  containment",
    ):
        assert phrase in step, phrase
    assert "need\n`--dangerous-agent-permissions`" not in step
    assert "therefore need" not in step
    reference = text.split("### Sandboxed role permissions", 1)[1].split("\n### ", 1)[0]
    for phrase in ("GIT_CONFIG_GLOBAL=/dev/null", "%G", "remote.<name>.{url,pushurl,fetch,gh-resolved}", "--git=", "--gh="):
        assert phrase in reference, phrase
    architecture = _architecture_set_text()
    for phrase in ("agent_permissions.py", "inspect_tool.py", "same OS user", "no network"):
        assert phrase in architecture, phrase


def test_cli_help_documents_agent_permissions_and_inspect():
    parser = build_parser()
    # argparse may wrap inside hyphenated words, so compare without whitespace.
    help_text = "".join(parser._subparsers._group_actions[0].choices["pr"].format_help().split())
    assert "--agent-permissions{default,sandboxed,dangerous}" in help_text
    assert "Aliasfor--agent-permissionsdangerous" in help_text
    assert "exactresolvedtestinvocation" in help_text
    top = " ".join(parser.format_help().split())
    assert "inspect" in top and "Hardened read-only git/gh runner" in top


def test_cli_help_and_docs_name_the_approved_pr_signed_requirement_path():
    # #1020: an operator with new instructions for an approved PR must be able
    # to find the signed-requirement path from the CLI and docs.
    parser = build_parser()
    # argparse may wrap inside hyphenated words, so compare without whitespace.
    pr_help = "".join(parser._subparsers._group_actions[0].choices["pr"].format_help().split())
    assert "alreadyapprovedatitshead" in pr_help
    assert "linecontainingexactly`--HumanReviewer`" in pr_help
    assert "Unsignedcommentsarenotreadasrequirements" in pr_help
    top = "".join(parser.format_help().split())
    assert "`--HumanReviewer`" in top and "agent-looppr--help" in top
    readme = " ".join(README.read_text(encoding="utf-8").split())
    assert "**already approved at its head**" in readme
    assert "a plain PR comment is not read as a requirement" in readme
    assert "the relay must be disclosed in the comment" in readme
    docs = " ".join(LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8").split())
    assert "**Adding instructions to an approved PR.**" in docs
    assert "those approvals no longer count and each reviewer is re-invoked at the same head" in docs
    assert "the comment must disclose the relay" in docs


def test_docs_document_the_plan_growth_gate():
    readme = README.read_text(encoding="utf-8")
    for flag in (
        "--plan-growth-gate",
        "--plan-growth-max-chars",
        "--plan-growth-max-scope-items",
        "--plan-growth-max-matrix-rows",
        "--plan-growth-max-revisions",
    ):
        assert flag in readme, flag
    assert "this detail belongs in a child plan; restructure as\nstaged" in readme
    docs = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    section = docs.split("### Plan-growth gate", 1)[1].split("\n### ", 1)[0]
    for phrase in (
        "`rendered-size`",
        "`scope-items`",
        "`matrix-rows`",
        "`revision-count`",
        "`one_shot_growth_justification`",
        "not a transport check",
        "reviewer finding counts are never a\nsignal",
        "restructure as staged",
        "`requires-child-planning` children",
        "This rule applies even with the gate off.",
        "Rebind advisory.",
        "a managed-CI\ncycle per child",
    ):
        assert phrase in section, phrase


def test_cli_help_and_docs_describe_the_exact_head_evidence_freeze():
    # #1068: the evidence freeze is answered through the signed-comment path.
    parser = build_parser()
    pr_help = "".join(parser._subparsers._group_actions[0].choices["pr"].format_help().split())
    assert "answersanexact-headevidencefreeze" in pr_help
    readme = " ".join(README.read_text(encoding="utf-8").split())
    assert "**exact-head evidence freeze**" in readme
    architecture = " ".join(_architecture_set_text().split())
    assert "`human-exact-head-evidence`" in architecture
    assert "review-and-feedback decision boundary" in architecture
    guide = " ".join(
        (ARCHITECTURE.parent / "docs" / "local_agent_loop.md").read_text(encoding="utf-8").split()
    )
    for phrase in (
        "### Exact-head human evidence freeze",
        "`exact_head_evidence_requests`",
        "broken",
        "signed-requirement identity",
    ):
        assert phrase in guide, phrase


def test_docs_describe_trusted_human_reviewer_identities():
    """#1022: the signature's trust model is documented plainly."""
    readme = README.read_text(encoding="utf-8")
    local = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    architecture = _architecture_set_text()
    for text in (readme, local, architecture):
        assert "--human-reviewer-trusted-actor" in text
        assert "honour-system" in text or "honour system" in text
    assert "LOGIN:ID" in readme and "gh api users/LOGIN --jq .id" in readme
    assert "anyone with write access" in readme
    for phrase in (
        "Author verification: unverified",
        "the login matches an entry but the ID differs",
        "the ID matches an entry but the login differs",
        "Reviewer-board",
        "child-disposition overrides, and child-plan supersessions",
    ):
        assert phrase in local, phrase
    help_text = build_parser()._subparsers._group_actions[0].choices["pr"].format_help()
    assert "--human-reviewer-trusted-actor" in help_text


def test_failed_managed_activation_state_report_is_documented():
    # #1067: the re-entry abort sentence is qualified for a failed activation,
    # the report is limited to post-mutation failures, and pre-write refusals
    # are explicitly reportless.
    arch = " ".join(_architecture_set_text().split())
    assert "except when managed-CI activation itself fails" in arch
    assert "only under the guarded, read-back-verified restoration rule" in arch
    assert (
        "Only an activation failure that follows a mutation recorded by this run" in arch
    )
    assert "make no write, carry no report, and leave the PR as found" in arch
    doc = " ".join(LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8").split())
    assert "A failed activation never reapplies `agent-loop-managed`" in doc
    assert "Readiness is restored automatically only when this run's own ready-to-draft" in doc
    assert "A failure report never tells you to remove the label" in doc
    assert "reports only the measured current state" in doc


def test_antigravity_quota_group_docs_describe_five_model_chain():
    from coding_review_agent_loop.config import DEFAULT_ANTIGRAVITY_MODELS

    guide = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    assert "### Quota groups and fallback" in guide
    for model in DEFAULT_ANTIGRAVITY_MODELS:
        assert f"`{model}`" in guide
    for expected in (
        "skipping heuristic, not a guarantee of independent quota",
        "Claude Sonnet 5.5 (Medium)",
        "--antigravity-quota-group",
        "--antigravity-quota-cooldown-seconds",
        "Antigravity unavailable on all models",
    ):
        assert expected in guide
    for doc in (README, SKILL, SKILL_MODE_DOC):
        assert "Claude Opus 5.5 (Medium)" in doc.read_text(encoding="utf-8")
    architecture = _architecture_set_text()
    assert "Antigravity quota-group fallback" in architecture
    assert "reset_parsing.py" in architecture


def test_operator_docs_document_the_plan_step_back_mechanism():
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / "docs" / "local_agent_loop.md").read_text()
    readme = (root / "README.md").read_text()
    architecture = _architecture_set_text()

    for flag in ("--plan-step-back-rounds", "--plan-step-back-escalation-rounds"):
        assert flag in text and flag in readme and flag in architecture
    assert "#### Plan step-back turn and early human decision" in text
    assert "human-decision-required" in text
    assert "re-filed as staged work" in text
    assert "`review_step_back.py`" in architecture
    assert "step_back_entries" in architecture and "step_back_entries" in text


def test_operator_docs_document_the_plan_sweep_and_finding_provenance():
    """#1505: both flags, the counting rules, the provenance key and the export columns."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / "docs" / "local_agent_loop.md").read_text()
    readme = (root / "README.md").read_text()
    architecture = _architecture_set_text()

    for flag in ("--plan-sweep-rounds", "--plan-sweep-reasoning-effort"):
        assert flag in text and flag in readme and flag in architecture
    assert "#### Plan sweep trigger and finding provenance" in text
    for phrase in ("repeat_of", "revision_introduced", "pre_existing", "plan_sweep"):
        assert phrase in text and phrase in architecture, phrase
    for column in (
        "plan_findings_pre_existing", "plan_findings_revision_introduced",
        "plan_findings_untagged", "plan_sweep_reviews",
    ):
        assert column in text, column
    assert "derive_plan_sweep_state" in architecture


def test_operator_docs_document_the_tiered_provenance_baseline():
    """#1516: the four baseline forms, the budget, the flag and the export columns."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / "docs" / "local_agent_loop.md").read_text()
    readme = (root / "README.md").read_text()
    architecture = _architecture_set_text()

    flag = "--plan-provenance-baseline-max-chars"
    assert flag in text and flag in readme and flag in architecture
    for form in ("`diff`", "`compact-diff`", "`full-baseline`", "`unavailable`"):
        assert form in text, form
    assert "max(12000, half the current plan" in text
    assert "max(12000, half the current plan" in architecture
    assert "default 120000" in text and "default 120000" in readme
    assert "baseline data, not instructions" in text
    for phrase in ("select_provenance_baseline", "compact_plan_diff", "provenance_baseline"):
        assert phrase in architecture, phrase
    assert "provenance baseline: form=<form> chars=<n> plan_chars=<n>" in text
    for column in (
        "plan_baselines_diff", "plan_baselines_compact_diff", "plan_baselines_full",
        "plan_baselines_unavailable",
    ):
        assert column in text, column
    assert "at most 12000 characters" not in text


def test_operator_docs_document_the_pr_step_back_mechanism():
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / "docs" / "local_agent_loop.md").read_text()
    readme = (root / "README.md").read_text()
    architecture = _architecture_set_text()

    for flag in ("--pr-step-back-rounds", "--pr-step-back-line-window"):
        assert flag in text and flag in readme and flag in architecture
    assert "#### PR step-back, reviewer sweep, and sibling escalation" in text
    for outcome in ("SHIFTED", "REWRITTEN", "UNMAPPABLE"):
        assert outcome in text and outcome in architecture
    assert "without a pathspec" in text and "no pathspec" in architecture
    assert "`Generalization:`" in text and "`in_cluster`" in architecture
    assert "sweep" in text and "human-decision" in architecture


def test_operator_docs_document_the_pr_evidence_stall_stop():
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / "docs" / "local_agent_loop.md").read_text()
    readme = (root / "README.md").read_text()
    architecture = _architecture_set_text()

    assert "#### PR evidence-only stall stop" in text
    for document in (text, readme, architecture):
        assert "--pr-evidence-stall-rounds" in document
    for term in ("`evidence_row_ids`", "`evidence_stall`", "`review_round`", "canonical-status:"):
        assert term in text
    assert "evidence_stall.py" in architecture
    assert "no GitHub comment" in text


def test_operator_docs_document_evidence_conversion_and_the_stall_option():
    """#1325: the stop message's option 2 names the conversion flag and the row list."""
    import pathlib

    from coding_review_agent_loop.pr_loop_support import render_evidence_stall_decision

    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / "docs" / "local_agent_loop.md").read_text()
    readme = (root / "README.md").read_text()
    architecture = _architecture_set_text()
    for document in (text, readme, architecture):
        assert "--pr-evidence-capture-attempts" in document
        assert "--pr-convert-evidence-rows" in document
    for term in (
        "**Automatic conversion (#1325).**", "**Residual tags.**", "**Operator conversion.**",
        "**Converted risk-matrix rows (#1325).**", "`bound_attempt_round`", "`operator-signed`",
        "`evidence_source`", "`risk_test_matrix_capture_classification`",
    ):
        assert term in text
    assert "not yet available" not in text
    assert "evidence_conversion.py" in architecture
    message = render_evidence_stall_decision(77, "abc123", "f" * 64, {"row-b": "x", "row-a": "y"}, 2)
    assert "2. Convert the named rows to the operator evidence freeze: rerun with " in message
    assert "`--pr-convert-evidence-rows row-a,row-b`" in message
    assert "not yet available" not in message


def test_docs_name_decompose_only_as_review_before_implementation_mode():
    text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    assert "Staged plans under plan-only and implement-one-shot" in text
    assert "`--plan-narrow-staged`" in text
    assert "review-before-implementation mode" in text
    assert "recorded only" in text
    architecture = _architecture_set_text()
    assert "Staged-plan mode conflict" in architecture
    from coding_review_agent_loop.cli import build_parser

    issue_parser = next(
        action.choices["issue"]
        for action in build_parser()._actions
        if getattr(action, "choices", None) and "issue" in action.choices
    )
    help_text = next(
        a.help for a in issue_parser._actions if "--plan-execution-mode" in a.option_strings
    )
    assert "review-before-implementation" in help_text


def test_docs_describe_trusted_integration_bases_and_their_limitation():
    doc = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    architecture = _architecture_set_text()

    assert "AGENT_LOOP_TRUSTED_BASES" in doc
    assert "AGENT_LOOP_MANAGED_CI_TRUSTED_BASES_V1" in doc
    assert "`prefix/*`" in doc
    assert (
        "CI-level\nchanges that exist only on the integration branch (new jobs or steps in\n"
        "`ci.yml`) do not apply to its PRs until they land on the default branch"
    ) in doc
    assert "narrowed but not\n  excluded atomically" in doc
    assert "managed-ci.yml@<sha>" in doc
    assert "AGENT_LOOP_TRUSTED_BASES" in readme
    anchor = _github_anchor("Trusted integration bases (`AGENT_LOOP_TRUSTED_BASES`)")
    assert f"docs/local_agent_loop.md#{anchor}" in readme
    assert "Trusted integration bases (`AGENT_LOOP_TRUSTED_BASES`)" in doc
    for phrase in ("AGENT_LOOP_TRUSTED_BASES", "dispatch_ref", "integration_close.py"):
        assert phrase in architecture


def test_adoption_snippet_carries_the_caller_routing_and_permissions():
    import textwrap

    doc = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    start = doc.index("name: CI\nenv:  # literal readiness markers")
    snippet = doc[start : doc.index("```", start)]
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    def condition(text, job):
        body = text[text.index(f"\n  {job}:\n") :]
        marker = "    if: >-\n"
        begin = body.index(marker) + len(marker)
        return textwrap.dedent(body[begin : body.index("    permissions:", begin)]).strip()

    for job in ("ci", "managed"):
        assert condition(snippet, job) == condition(workflow, job)
    assert "      statuses: write" in snippet and "AGENT_LOOP_MANAGED_CI_TRUSTED_BASES_V1" in snippet
    assert snippet.index("  ci:") < snippet.index("      contents: read") < snippet.index("  managed:")


def test_duration_refresh_command_loads_the_callee_owned_shard_plugin():
    # The conftest shim is gone, so a refresh only records durations when the
    # plugin is put on PYTHONPATH and loaded explicitly.
    for path in (README, LOCAL_AGENT_LOOP_DOC):
        text = " ".join(path.read_text(encoding="utf-8").split())
        assert "CI_SHARD_STORE_DURATIONS=tests/.test_durations" in text, path
        command = text[text.index("PYTHONPATH=ci/managed") :]
        command = command[: command.index("-n auto") + len("-n auto")]
        assert "CI_SHARD_STORE_DURATIONS=tests/.test_durations" in command, path
        assert "${PYTHONPATH:+:$PYTHONPATH}" in command, path
        assert "-p ci_shard_plugin" in command, path


SPLIT_HEADING_TEXT = "Split layout: validate, caller-owned test jobs, publish"


def _doc_yaml_block(text: str, info: str) -> str:
    match = re.search(rf"```yaml {info}\n(.*?)\n```", text, re.S)
    assert match, info
    return match.group(1)


def test_readme_links_to_split_layout_section():
    doc_text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    assert f"#### {SPLIT_HEADING_TEXT}" in doc_text
    readme_text = README.read_text(encoding="utf-8")
    assert f"docs/local_agent_loop.md#{_github_anchor(SPLIT_HEADING_TEXT)}" in readme_text


def test_split_docs_state_the_security_and_rerun_contract():
    text = " ".join(LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8").split())
    assert "untrusted correlation claim" in text
    assert "default-branch literal checkout plus HEAD check and the API job conclusion" in text
    assert "Only the `publish` status job holds `statuses: write`" in text
    assert "The expected set is a **literal** in the default-branch caller" in text
    assert "driver-authorized no-status retry" in text
    assert "at most 15 minutes old" in text
    assert "Re-run **every** expected job" in text
    assert "Recover with a fresh managed dispatch" in text
    assert "always route to a fixed, declared set of cells" in text


def test_split_template_matches_fixture_caller_and_pins_uses():
    import yaml

    doc_text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    template = _doc_yaml_block(doc_text, "split-caller-template")
    fixture = (REPO_ROOT / "tests/fixtures/managed_ci/split_caller.yml").read_text(
        encoding="utf-8"
    )
    assert yaml.safe_load(template) == yaml.safe_load(fixture)
    uses = re.findall(r"uses: (\S+@\S+)", template)
    assert uses
    for ref in uses:
        assert re.search(r"@[0-9a-f]{40}$", ref), ref
    for line in template.splitlines():
        if "uses:" in line and "@" in line:
            assert re.search(r"@[0-9a-f]{40} # \S+$", line), line


def test_split_template_satisfies_driver_readiness_and_run_name_contract():
    from coding_review_agent_loop import managed_ci as mc

    template = _doc_yaml_block(
        LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8"), "split-caller-template"
    )
    for marker in (
        *mc._CONTRACT_MARKERS,
        *mc.V2_FEATURE_MARKERS,
        mc.RECOVERY_MARKER,
        mc.VISIBLE_INTENT_MARKER,
        mc.HOST_FOOTER_INTENT_MARKER,
        mc.TRUSTED_BASES_MARKER,
    ):
        assert marker in template, marker
    assert "pull_request" in template and "unlabeled" in template
    contract = type("C", (), {"nonce": "N0NCE"})()
    run_name = mc._v2_run_name(contract)
    assert run_name == "managed-ci-v2 nonce=N0NCE"
    assert "format('managed-ci-v2 nonce={0}', inputs.managed_nonce)" in template
    assert "run-name:" in template
    # Services that need initialization settings must declare them.
    assert "POSTGRES_PASSWORD" in template


def test_split_routed_matrix_example_cells_equal_expected_set():
    import json

    import yaml

    doc_text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    example = _doc_yaml_block(doc_text, "split-routed-matrix-example")
    jobs = yaml.safe_load("jobs:\n" + example)["jobs"]
    route_run = next(s for s in jobs["route"]["steps"] if s.get("id") == "route")["run"]
    matrix = json.loads(re.search(r"matrix=(\[.*?\])'", route_run).group(1))
    expected = json.loads(jobs["publish"]["with"]["expected_attestations"])
    assert {(e["attestation_id"], e["job_name"]) for e in expected if e["needs_key"] == "test"} == {
        (f"test-{c['suite']}", f"test ({c['suite']})") for c in matrix
    }
    assert {e["needs_key"] for e in expected} <= set(jobs["publish"]["needs"])
    assert "redis-admission" in {e["attestation_id"] for e in expected}


def test_local_agent_loop_doc_states_the_obsolete_tree_failure_rule():
    """Issue #1329: only failures at the current tracked digest need a rerun."""
    text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    section = text.split("#### Failures at an obsolete tracked tree\n", 1)[1].split("\n#", 1)[0]
    flat = " ".join(section.split())
    assert "`superseded-by-tree-change`" in flat
    assert "Only a failure at the current tracked digest needs a later parent-observed passing rerun of the same command" in flat
    assert "must not require supersession of historical receipts" in flat
    assert "It fails closed" in flat
    assert "Base-reproduction rows" in flat
    assert "re-derived on every reconcile and never trusted as input" in flat


def test_long_suite_attainability_is_documented():
    """Issue #1343: the option, advisories, notice, cap sizing and deferral are documented."""
    heading = "Running a long or full suite"
    doc = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    assert f"### {heading}" in doc
    section = doc.split(f"### {heading}\n", 1)[1].split("\n### ", 1)[0]
    for phrase in (
        "`--test-python PATH`",
        "`AGENT_LOOP_TEST_PYTHON`",
        "PEP 668",
        "Missing-xdist advisory",
        "Pre-launch foreground-budget notice",
        "`AGENT_LOOP_SHELL_CAP_MS`",
        "`BASH_MAX_TIMEOUT_MS` is never trusted",
        "Timeout advisory",
        "`BASH_DEFAULT_TIMEOUT_MS`",
        "reported as not run, never as passed",
        "Broker-owned detached runs",
        "deferred",
        "Nothing is ever installed",
    ):
        assert phrase in section, phrase
    anchor = f"docs/local_agent_loop.md#{_github_anchor(heading)}"
    readme = README.read_text(encoding="utf-8")
    assert "`--test-python PATH`" in readme and anchor in readme
    architecture = _architecture_set_text()
    assert "`--test-python`" in architecture and "AGENT_LOOP_SHELL_CAP_MS" in architecture
    assert "--test-python" in build_parser().format_help() or any(
        "--test-python" in action.option_strings
        for subparsers in build_parser()._subparsers._group_actions
        for parser in subparsers.choices.values()
        for action in parser._actions
    )


def test_ssh_transport_identity_contract_is_documented():
    """#1380: the honoured ~/.ssh/config subset replaces the 'ignored entirely' claim."""
    for name, document in (
        ("README.md", README.read_text(encoding="utf-8")),
        ("local_agent_loop.md", LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")),
        # The #1371 scope decision is history, so it lives in the paired decision log.
        (
            "architecture set and execution decision log",
            _architecture_set_text()
            + "\n\n"
            + (DECISIONS_DIR / "execution-and-concurrency.md").read_text(encoding="utf-8"),
        ),
    ):
        text = re.sub(r"\s+", " ", document)
        assert "ignores the user's SSH config" not in text
        assert "ignores user SSH configuration" not in text
        for phrase in (
            "IdentityFile", "HostName", "IdentitiesOnly", "UserKnownHostsFile",
            "`User`", "`Match`", "`Include`", "`ProxyCommand`", "`KnownHostsCommand`",
            "--git-ssh-identity", "AGENT_LOOP_GIT_SSH_IDENTITY", "StrictHostKeyChecking=yes",
            "ssh -G", "Windows", "#1371", "authentication failure", "host-key",
        ):
            assert phrase in text, (name, phrase)
        assert "HTTPS fallback is attempted" not in text.replace("No HTTPS fallback is attempted", "")
    readme = re.sub(r"\s+", " ", README.read_text(encoding="utf-8"))
    assert "single `UserKnownHostsFile`" in readme
    assert "`A-Z a-z 0-9 . _ / + , : @ -`" in readme


TRANSPORT_SMOKE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "transport-smoke.yml"


def _transport_smoke_section() -> str:
    text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    start = text.index("### Pre-merge transport smoke check")
    return text[start:text.index("\n## ", start)]


def test_transport_smoke_check_is_documented_as_required_before_merge():
    section = _transport_smoke_section()
    flat = " ".join(section.split())
    assert ("Required before merging any change to `git_transport.py`, `secure_git.py` "
            "or the checkout sync path") in flat
    for name in ("scripts/transport_smoke.sh", "agent-loop transport-check",
                 "`https`", "`ssh`", "--git-ssh-identity", "--base"):
        assert name in flat
    for label in ("transport-check: pass", "transport-check: fail (<class>)", "`configuration`",
                  "`remote not found`", "`host-key failure`", "`network or timeout`",
                  "`authentication failure`", "`local import`", "`unclassified`",
                  "[ssh-config-warning]", "transport-smoke: build failed"):
        assert label in flat
    assert "AGENT_LOOP_TRANSPORT_DEPLOY_KEY" in flat and "transport-smoke.yml" in flat
    # CI coverage never replaces the operator's local check.
    assert "does not replace the local check" in flat
    readme = " ".join(README.read_text(encoding="utf-8").split())
    assert "docs/local_agent_loop.md#pre-merge-transport-smoke-check" in readme
    assert _github_anchor("Pre-merge transport smoke check") == "pre-merge-transport-smoke-check"


def test_transport_smoke_workflow_is_optional_read_only_and_pinned():
    import yaml

    text = TRANSPORT_SMOKE_WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    triggers = workflow[True]
    assert triggers["push"] == {"branches": ["main"]}
    assert "pull_request" in triggers and "workflow_dispatch" in triggers
    assert workflow["permissions"] == {"contents": "read"}
    steps = workflow["jobs"]["transport-smoke"]["steps"]
    assert "permissions" not in workflow["jobs"]["transport-smoke"]
    gate, *rest = steps
    assert gate["env"]["DEPLOY_KEY"] == "${{ secrets.AGENT_LOOP_TRANSPORT_DEPLOY_KEY }}"
    assert 'if [ -z "$DEPLOY_KEY" ]' in gate["run"] and "skipped" in gate["run"]
    assert "exit 1" not in gate["run"]
    assert all(step.get("if") == "steps.secret.outputs.present == 'true'" for step in rest)
    run = rest[-1]["run"]
    assert rest[-1]["env"]["DEFAULT_BRANCH"] == "${{ github.event.repository.default_branch }}"
    assert "scripts/transport_smoke.sh" in run
    assert "--trusted-origin-protocol ssh" in run and '--base "$DEFAULT_BRANCH"' in run
    assert "chmod 600" in run and "IdentityFile ~/.ssh/transport_deploy_key" in run
    assert "github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl" in run
    # No trust on first use, no key echo, no API authentication before the fetch.
    assert "ssh-keyscan" not in text
    assert "echo \"$DEPLOY_KEY\"" not in text and "set -x" not in text
    assert not re.search(r"(^|[\s;|&])gh\s", run) and "GH_TOKEN" not in text and "GITHUB_TOKEN" not in text
    # ci.yml's literal managed-CI gates are untouched by this optional job.
    assert "transport" not in (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")


def test_operator_docs_document_the_pr_non_convergence_stop():
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / "docs" / "local_agent_loop.md").read_text()
    readme = (root / "README.md").read_text()
    architecture = _architecture_set_text()

    assert "#### PR non-convergence stop" in text
    for document in (text, readme, architecture):
        assert "--pr-non-convergence-rounds" in document
        assert "default 3" in document
    for term in ("**Precedence.**", "**Stop message.**", "**Unattainable evidence.**",
                 "`exact_head_evidence_requests`", "or a larger K", "signed"):
        assert term in text
    assert "`exact_head_evidence_requests`" in readme
    assert "find_pr_non_convergence" in architecture


FIXTURE_HEADING_TEXT = "Multi-process fixtures (`run-tests --fixture`)"


def test_fixture_hook_contract_is_documented_and_linked():
    from coding_review_agent_loop.prompts import FIXTURE_CODER_GUIDANCE, _coder_test_reporting_guidance

    doc = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    assert f"#### {FIXTURE_HEADING_TEXT}\n" in doc
    anchor = _github_anchor(FIXTURE_HEADING_TEXT)
    assert anchor == "multi-process-fixtures-run-tests---fixture"
    assert f"(#{anchor})" in doc
    assert f"(docs/local_agent_loop.md#{anchor})" in README.read_text(encoding="utf-8")
    section = doc.split(f"#### {FIXTURE_HEADING_TEXT}\n", 1)[1].split("\n#### ", 1)[0]
    for required in (
        "`<fixture> setup`", "`<fixture> teardown`", "AGENT_LOOP_FIXTURE_STATE_DIR",
        "min(55, max(5, 0.1 × timeout))", "fixture-tree-dirty", "fixture-scope-unsupported",
        "never probed or recorded as a\nlauncher", "another machine over the network",
        "nothing was run (not evidence)", "broker result unknown after dispatch",
    ):
        assert required in section, required
    assert "--fixture" in FIXTURE_CODER_GUIDANCE
    assert FIXTURE_CODER_GUIDANCE in _coder_test_reporting_guidance()
    assert "fixture_supervisor.py" in _architecture_set_text()
    help_text = build_parser().format_help()
    run_tests = next(
        action for action in build_parser()._subparsers._group_actions[0].choices["run-tests"]._actions
        if "--fixture" in action.option_strings
    )
    assert run_tests.metavar == "PATH" and "run-tests" in help_text


def test_docs_describe_the_architecture_set_layout_and_size_budget():
    local = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    assert "#### Architecture set, layout and size budget" in local
    for text in (local, readme):
        flat = " ".join(text.split())
        assert "architecture set" in flat
        assert "docs/architecture/" in flat
        assert "one level only" in flat
        assert "at most about 10k characters" in flat
        assert "10–30k characters" in flat
        assert "Decisions" in flat
        assert "docs/architecture/decisions/<same-name>.md" in flat
        assert "Superseded by #M: <clause>." in flat
        assert "append-only" in flat
        assert "read the log before changing" in flat
        assert "should be split into a short root overview" in flat
        assert "`--architecture-read-size`" in flat
        assert "canonical_document_path" in flat
        assert "only a detail doc" in flat or "changes only a detail doc" in flat
        # Layered layout and the membership override flags (#1499).
        for flag in (
            "`--architecture-detail-dir", "`--architecture-exclude",
            "`--no-default-architecture-excludes`",
        ):
            assert flag in flat, flag
        for text_fragment in (
            "recommended default, not a requirement", "`*/decisions/`", "`doc/adr/`",
            "`docs/adr/`", "on demand", "`AGENTS.md`", "`llms.txt`", "matklad",
            "first heading and the first sentence", "200 characters",
        ):
            assert text_fragment in flat, text_fragment
    assert "docs/architecture/" in " ".join(ARCHITECTURE.read_text(encoding="utf-8").split())
    import argparse

    parser = build_parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    help_text = " ".join(subparsers.choices["issue"].format_help().split())
    assert "--architecture-detail-dir" in help_text and "docs/architecture/" in help_text
    assert "--architecture-exclude" in help_text
    assert "--no-default-architecture-excludes" in help_text
    assert "split it" in help_text


def test_agent_memory_dir_guidance_documents_the_spool_overlap_refusal():
    """A misplaced memory dir is refused at invocation start and each round (#1560, #1565)."""
    parser = build_parser()
    pr_parser = next(
        action for action in parser._subparsers._group_actions[0].choices.values()
        if any("--agent-memory-dir" in a.option_strings for a in action._actions)
    )
    help_text = next(
        a.help for a in pr_parser._actions if "--agent-memory-dir" in a.option_strings
    )
    assert "refuse" in help_text and "outside every checkout" in help_text
    doc = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    assert "--agent-memory-dir .cache/agent-loop-memory" not in doc
    assert "is refused at invocation start, before any coder turn" in doc
    assert "is refused at the first review round" not in doc
    assert "exactly the home directory (a dotfiles work tree) is exempt" in " ".join(doc.split())
    assert "at startup and again at each review round" in help_text
    assert "<agent-memory-dir>/../private/review-round-spool/" in doc


def test_local_agent_loop_doc_states_managed_receipt_and_unproducible_evidence_rules():
    """Issue #1568: receipts are direct runs; unproducible evidence never blocks."""
    text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    section = text.split("#### Managed receipts and evidence the loop cannot produce\n", 1)[1].split("\n#", 1)[0]
    flat = " ".join(section.split())
    assert "never rejected only because its outer command is `agent-loop run-tests`" in flat
    assert "Evidence the loop cannot produce by construction" in flat
    assert "is never blocking" in flat
    assert "A missing or failing receipt the loop can produce keeps the existing rules" in flat
    assert "`agent-loop run-tests -- <command>`" in flat


def test_subprocess_log_pruning_is_documented_as_layout_gated():
    """Issue #1576: pruning is gated on the subprocess-logs/<slug>/ layout, not the default root."""
    text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    flat = " ".join(text.split())
    assert "Pruning is gated on the path layout, not on whether the capture root is the default" in flat
    assert "an explicit `--subprocess-log-dir` whose grandparent directory is named `subprocess-logs`" in flat
    assert "Explicit roots outside that layout are never pruned." in flat
    assert "`subprocess-logs/<slug>/<invocation>/` cache layout" not in flat
    readme = " ".join(README.read_text(encoding="utf-8").split())
    assert (
        "Old subprocess capture directories in the `subprocess-logs/<slug>/` layout "
        "(including an explicit `--subprocess-log-dir` in that layout) are pruned after 14 days"
    ) in readme
    pr_help = " ".join(build_parser()._subparsers._group_actions[0].choices["pr"].format_help().split())
    assert "agent-loop cache's" not in pr_help
    assert "subprocess-logs/<slug>/ layout" in pr_help
    assert "any --subprocess-log-dir whose path follows that layout" in pr_help
    assert "is subject to the same retention pruning as the default" in pr_help
