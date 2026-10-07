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
    text = " ".join(ARCHITECTURE.read_text(encoding="utf-8").split())
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
    architecture = (root / "ARCHITECTURE.md").read_text()
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
    architecture = ARCHITECTURE.read_text()
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
    architecture = " ".join(ARCHITECTURE.read_text().split())
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
    architecture = ARCHITECTURE.read_text(encoding="utf-8")
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
    architecture = " ".join(ARCHITECTURE.read_text(encoding="utf-8").split())
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
    architecture = ARCHITECTURE.read_text(encoding="utf-8")
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
    arch = " ".join(ARCHITECTURE.read_text(encoding="utf-8").split())
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
    architecture = ARCHITECTURE.read_text(encoding="utf-8")
    assert "Antigravity quota-group fallback" in architecture
    assert "reset_parsing.py" in architecture


def test_operator_docs_document_the_plan_step_back_mechanism():
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / "docs" / "local_agent_loop.md").read_text()
    readme = (root / "README.md").read_text()
    architecture = (root / "ARCHITECTURE.md").read_text()

    for flag in ("--plan-step-back-rounds", "--plan-step-back-escalation-rounds"):
        assert flag in text and flag in readme and flag in architecture
    assert "#### Plan step-back turn and early human decision" in text
    assert "human-decision-required" in text
    assert "re-filed as staged work" in text
    assert "`review_step_back.py`" in architecture
    assert "step_back_entries" in architecture and "step_back_entries" in text


def test_operator_docs_document_the_pr_step_back_mechanism():
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / "docs" / "local_agent_loop.md").read_text()
    readme = (root / "README.md").read_text()
    architecture = (root / "ARCHITECTURE.md").read_text()

    for flag in ("--pr-step-back-rounds", "--pr-step-back-line-window"):
        assert flag in text and flag in readme and flag in architecture
    assert "#### PR step-back, reviewer sweep, and sibling escalation" in text
    for outcome in ("SHIFTED", "REWRITTEN", "UNMAPPABLE"):
        assert outcome in text and outcome in architecture
    assert "without a pathspec" in text and "no pathspec" in architecture
    assert "`Generalization:`" in text and "`in_cluster`" in architecture
    assert "sweep" in text and "human-decision" in architecture


def test_docs_name_decompose_only_as_review_before_implementation_mode():
    text = LOCAL_AGENT_LOOP_DOC.read_text(encoding="utf-8")
    assert "Staged plans under plan-only and implement-one-shot" in text
    assert "`--plan-narrow-staged`" in text
    assert "review-before-implementation mode" in text
    assert "recorded only" in text
    architecture = (LOCAL_AGENT_LOOP_DOC.parent.parent / "ARCHITECTURE.md").read_text(
        encoding="utf-8"
    )
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
    architecture = ARCHITECTURE.read_text(encoding="utf-8")

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
        if ref.startswith("actions/"):
            continue
        assert re.search(r"@[0-9a-f]{40}$", ref), ref


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
