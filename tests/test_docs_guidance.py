import re
from pathlib import Path

from coding_review_agent_loop.cli import build_parser

REPO_ROOT = Path(__file__).parent.parent
LOCAL_AGENT_LOOP_DOC = REPO_ROOT / "docs" / "local_agent_loop.md"
README = REPO_ROOT / "README.md"
ARCHITECTURE = REPO_ROOT / "ARCHITECTURE.md"

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


def test_architecture_is_discoverable_from_existing_guides():
    assert "(ARCHITECTURE.md)" in README.read_text(encoding="utf-8")
    for path in (LOCAL_AGENT_LOOP_DOC, SKILL_MODE_DOC):
        assert "(../ARCHITECTURE.md)" in path.read_text(encoding="utf-8")
    # Preserve the old inbound anchor while the overview moves to its own file.
    assert "## Architecture\n" in LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")


def test_architecture_links_and_component_paths_exist():
    text = ARCHITECTURE.read_text(encoding="utf-8")
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        if target.startswith("https://"):
            continue
        path_text, _, anchor = target.partition("#")
        path = ARCHITECTURE.parent / path_text
        assert path.exists(), target
        if anchor:
            headings = re.findall(r"^#+ (.+)$", path.read_text(encoding="utf-8"), re.M)
            assert anchor in {_github_anchor(heading) for heading in headings}, target
    table = text.split("## Component Map\n", 1)[1].split("## Main Lifecycle\n", 1)[0]
    for module in re.findall(r"`([\w/]+\.py)`", table):
        assert (REPO_ROOT / "src" / "coding_review_agent_loop" / module).is_file(), module


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
    architecture = ARCHITECTURE.read_text(encoding="utf-8")
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

    assert "one active `agent-loop` invocation per repository per machine" in section
    assert "does not currently enforce a repository-wide process lock" in section
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
        "`gh pr create --draft --label agent-loop-managed --body-file <path>`"
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
    # #865: only repetition within one row is fatal; cross-row reuse is valid.
    text = " ".join(ARCHITECTURE.read_text(encoding="utf-8").split())
    assert "duplicate admissible selectors" not in text
    assert "One admissible selector may be cited by several rows" in text
    assert "an admissible selector repeated within one row" in text


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
    text = ARCHITECTURE.read_text(encoding="utf-8")
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

    architecture = ARCHITECTURE.read_text(encoding="utf-8")
    assert "Reviewer repair is refused fail-closed" in architecture
    assert "a refusal is a reviewer unavailability, never a synthesized verdict" in architecture
    assert "proposal to evaluate, never an" in architecture


def test_docs_describe_conflict_round_continuity_exception():
    # #829: the conflict-resolution round advances the head with no reviewer,
    # so both the canonical and operator contracts must name that transition.
    arch_text = " ".join(ARCHITECTURE.read_text(encoding="utf-8").split())
    doc_text = " ".join(LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8").split())
    assert "merge-conflict resolution round" in arch_text
    assert "tool-owned merge-conflict obligation" in arch_text
    assert "outside the coder's classifiable item namespace" in arch_text
    assert "resume reauthenticates the same shape" in arch_text
    assert "merge-conflict resolution round is the single exception" in doc_text
    assert "tool-owned merge-conflict obligation instead of a review pair" in doc_text
    for text in (arch_text, doc_text):
        assert "still fails closed" in text
        assert "approve the exact final head before qualification or merge" in text


def test_architecture_documents_the_planning_scheduler():
    """`scope-5` (#905, from #841): the canonical overview is updated."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / "ARCHITECTURE.md").read_text()

    assert "plan_review_scheduling.py" in text
    assert "exact-plan candidate key" in text
    assert "reviewer-only phase-advance round" in text
    assert "plan-phase-advance" in text
    assert "four disjoint classes" in text
    assert "only a transport extraction failure stops the" in text
    assert "run, checked at startup and at every round boundary" in text
    assert "--plan-review-policy" in text
    assert "docs/local_agent_loop.md#staged-issue-plan-review" in text


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
    ):
        assert flag in text
        assert flag in readme
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
    assert "recovers the first two\nonly" in text
    assert "never recover class D" in text
    # Carried approvals and the exclusions.
    assert "HUMAN_REQUIREMENTS_RESOLVED" in text
    assert "Discussion-mode scheduling and the child-planning cycle always invoke the full" in text


def test_docs_document_the_flow_aware_planning_policy_evaluation():
    text = LOCAL_AGENT_LOOP_DOC.read_text()
    architecture = ARCHITECTURE.read_text()
    readme = README.read_text()

    # The flow dimension, its backward-compatible default, and per-flow
    # uniqueness, aggregation, and titling.
    assert "`flow` is `pr` or `plan` and defaults to `pr` when omitted" in text
    assert "`(flow, policy, run_id)`" in text
    assert "Frozen plan review policy evaluation" in text
    assert "`selective-intermediate` is PR-only" in text
    assert "validated\n`flow` (`pr` or `plan`, defaulting to `pr`" in architecture
    assert "`(flow, policy, run_id)`" in architecture
    # The stale "no flow dimension today" caveat is replaced by the comparison
    # the planning rows now support, with its provenance limit stated.
    assert "no flow dimension today" not in text
    assert "escaped plan defects are reported in\nthe `plan` flow" in text
    assert "`unavailable` with a reason rather than borrowed" in text
    assert "own `plan` flow" in readme
