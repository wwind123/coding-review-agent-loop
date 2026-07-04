import json
import re
from unittest.mock import patch

import pytest

import coding_review_agent_loop.orchestrator as orchestrator_module
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.decomposition import approved_plan_hash, format_one_shot_impl_handoff_comment
from coding_review_agent_loop.errors import QuotaResetExceededError
from coding_review_agent_loop.github import get_issue_context
from coding_review_agent_loop.orchestrator import (
    PostedRoundMetadata,
    _attach_round_metadata,
    _decode_round_metadata,
    _plan_subject,
    _strip_round_metadata,
)
from coding_review_agent_loop.prompts import (
    COMPACT_PLANNING_VOLATILE_TAIL_MARKER,
    HUMAN_REQUIREMENTS_ADDRESSED_MARKER,
)
from coding_review_agent_loop.protocol import UnresolvedReviewItem
from coding_review_agent_loop.salvage import latest_salvage_summary
from agent_loop_helpers import (
    FakeRunner,
    command_index,
    make_config,
    prior_item_dispositions,
    prior_plan_item_dispositions,
    structured_plan_review,
    structured_plan_revision,
    structured_plan_state,
    structured_pr_review,
)


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

def test_issue_loop_plan_first_files_surviving_future_from_mixed_outcome_round(tmp_path):
    future_text = "Factor the shared follow-up guidance into a reusable helper."
    runner = FakeRunner(
        claude_outputs=[
            "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            structured_plan_revision(
                summary="Address the blocking test gap.",
                prior_plan_item_dispositions=[
                    {"item_id": "item-2", "disposition": "resolved", "note": "Added the test."}
                ],
                plan_steps=["Make the change.", "Add the missing regression test."],
            ),
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(
                state="approved",
                future_followups=[future_text],
            ),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[
                    {
                        "item_id": "item-1",
                        "disposition": "future",
                        "note": "Keep this as confirmed post-plan cleanup.",
                    },
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
            ),
            structured_pr_review(state="approved", summary="LGTM."),
        ],
        gemini_outputs=[
            structured_plan_review(
                state="blocking",
                blocking_plan_issues=["Add a regression test for the plan-review ledger."],
                reviewer="Google Gemini",
            ),
            structured_plan_review(
                state="approved",
                future_followups=[future_text],
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "future"},
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
                reviewer="Google Gemini",
            ),
            structured_pr_review(
                state="approved",
                summary="LGTM.",
                reviewer="Google Gemini",
            ),
        ],
        issue_urls=["https://github.com/OWNER/REPO/issues/99"],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        approved_followups="issue",
        max_rounds=2,
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

    coder_revision_prompt = next(
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] and '"kind": "plan_revision"' in cmd[-1]
    )
    assert "item-2" in coder_revision_prompt
    assert "item-1" not in coder_revision_prompt
    assert future_text not in coder_revision_prompt

    round_two_reviewer_prompts = [
        cmd[-1]
        for cmd, _cwd in runner.commands
        if "Planning round: 2" in cmd[-1] and "Role: reviewer" in cmd[-1]
    ]
    assert len(round_two_reviewer_prompts) == 2
    assert all("item-1" in prompt for prompt in round_two_reviewer_prompts)

    assert len(runner.issues) == 1
    issue_body = runner.issues[0]["body"]
    assert "Planning round(s): 1, 2" in issue_body
    assert "Reviewers: Codex, Gemini" in issue_body
    assert "Original plan item ID(s): item-1, item-3" in issue_body
    assert "Keep this as confirmed post-plan cleanup." in issue_body
    assert any(
        "Reconciliation: 1 filed, 1 deduplicated, 0 skipped by cap." in comment
        for comment in runner.comments
    )

    issue_create_index = command_index(runner.commands, ["gh", "issue", "create"])
    round_two_review_indexes = [
        index
        for index, (cmd, _cwd) in enumerate(runner.commands)
        if "Planning round: 2" in cmd[-1] and "Role: reviewer" in cmd[-1]
    ]
    assert issue_create_index > max(round_two_review_indexes)

@pytest.mark.parametrize("later_disposition", ["resolved", "same-plan", "blocking"])
def test_issue_loop_plan_first_does_not_file_future_item_after_later_lifecycle_change(
    tmp_path, later_disposition
):
    future_text = "Extract the shared plan-review formatting helper."
    promoted = later_disposition in {"same-plan", "blocking"}
    promotion_state = "blocking" if promoted else "approved"
    final_plan_dispositions = [
        {"item_id": "item-1", "disposition": later_disposition},
        {"item_id": "item-2", "disposition": "resolved"},
    ]
    claude_outputs = [
        "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        structured_plan_revision(
            summary="Address the original blocker.",
            prior_plan_item_dispositions=[{"item_id": "item-2", "disposition": "resolved"}],
        ),
    ]
    if promoted:
        claude_outputs.append(
            structured_plan_revision(
                summary="Address the promoted future item.",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            )
        )
    claude_outputs.append(
        "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n"
        "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )

    def reviewer_outputs(reviewer):
        outputs = [
            structured_plan_review(
                state="approved",
                future_followups=[future_text],
                reviewer=reviewer,
            ),
            structured_plan_review(
                state=promotion_state,
                prior_plan_item_dispositions=final_plan_dispositions,
                reviewer=reviewer,
            ),
        ]
        if promoted:
            outputs.append(
                structured_plan_review(
                    state="approved",
                    prior_plan_item_dispositions=[
                        {"item_id": "item-1", "disposition": "resolved"}
                    ],
                    reviewer=reviewer,
                )
            )
        outputs.append(structured_pr_review(state="approved", reviewer=reviewer))
        return outputs

    runner = FakeRunner(
        claude_outputs=claude_outputs,
        codex_outputs=reviewer_outputs("OpenAI Codex"),
        gemini_outputs=[
            structured_plan_review(
                state="blocking",
                blocking_plan_issues=["Add the initial ledger regression."],
                reviewer="Google Gemini",
            ),
            *reviewer_outputs("Google Gemini")[1:],
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        approved_followups="issue",
        max_rounds=3,
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

    assert runner.issues == []
    assert not any(future_text in comment for comment in runner.comments[-2:])
    if promoted:
        coder_revision_prompts = [
            cmd[-1]
            for cmd, _cwd in runner.commands
            if cmd[:1] == ["claude"] and '"kind": "plan_revision"' in cmd[-1]
        ]
        assert len(coder_revision_prompts) == 2
        assert "item-1" in coder_revision_prompts[1]
        assert future_text in coder_revision_prompts[1]
        final_round_review_indexes = [
            index
            for index, (cmd, _cwd) in enumerate(runner.commands)
            if "Planning round: 3" in cmd[-1] and "Role: reviewer" in cmd[-1]
        ]
        implementation_index = next(
            index
            for index, (cmd, _cwd) in enumerate(runner.commands)
            if cmd[:1] == ["claude"] and "Implement the approved plan" in cmd[-1]
        )
        assert implementation_index > max(final_round_review_indexes)

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


def _salvage_dirs(config):
    salvage_root = config.log_dir / "salvage"
    if not salvage_root.exists():
        return []
    return sorted(path for path in salvage_root.iterdir() if path.is_dir())


def test_failed_issue_implementation_with_diff_writes_salvage_artifacts(tmp_path):
    patch_text = (
        "diff --git a/src/coding_review_agent_loop/cli.py b/src/coding_review_agent_loop/cli.py\n"
        "--- a/src/coding_review_agent_loop/cli.py\n"
        "+++ b/src/coding_review_agent_loop/cli.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    runner = FakeRunner(
        claude_outputs=[("quota exceeded; reset in 10m", 1)],
        post_agent_git_status=" M src/coding_review_agent_loop/cli.py\n?? scratch-note.md\n",
        post_agent_git_diff=patch_text,
        post_agent_git_diff_stat=" src/coding_review_agent_loop/cli.py | 2 +-\n",
        post_agent_git_diff_check="src/coding_review_agent_loop/cli.py:1: trailing whitespace.\n",
        post_agent_git_diff_check_returncode=2,
    )
    config = make_config(tmp_path)

    with pytest.raises(QuotaResetExceededError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config)

    message = str(excinfo.value)
    assert "Implementation salvage artifacts were written to" in message
    assert "patch:" in message
    assert "No non-empty public response file was produced at expected path" in message
    assert "no result was recorded because the agent command exited with quota/session-limit status" in message
    salvage_dir = _salvage_dirs(config)[0]
    assert str(salvage_dir / "salvage-summary.md") in message
    assert (salvage_dir / "partial.patch").read_text(encoding="utf-8") == patch_text
    assert "?? scratch-note.md" in (salvage_dir / "changed-files.txt").read_text(
        encoding="utf-8"
    )
    assert "2 +-" in (salvage_dir / "diff-stat.txt").read_text(encoding="utf-8")
    assert "trailing whitespace" in (salvage_dir / "diff-check.txt").read_text(
        encoding="utf-8"
    )

    summary = (salvage_dir / "salvage-summary.md").read_text(encoding="utf-8")
    assert "No\nsuccessful response, review result, or pull request should be inferred" in summary
    assert "Public response file: missing" in summary
    assert "Required marker status: missing or invalid" in summary
    assert "Untracked files appear in `changed-files.txt`" in summary

    metadata = json.loads((salvage_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["repo"] == "OWNER/REPO"
    assert metadata["issue_number"] == 56
    assert metadata["scope"] == "issue-implementation"
    assert metadata["agent"] == "claude"
    assert metadata["failure_category"] == "transient"
    assert metadata["response_file_missing"] is True
    assert metadata["diff_check_returncode"] == 2


def test_failed_issue_implementation_salvage_oserror_preserves_original_failure(
    tmp_path, capsys, monkeypatch
):
    runner = FakeRunner(
        claude_outputs=[("quota exceeded; reset in 10m", 1)],
        post_agent_git_diff="diff --git a/file.txt b/file.txt\n",
    )
    config = make_config(tmp_path, quiet=False)

    def fail_capture(*args, **kwargs):
        raise OSError("simulated disk full")

    monkeypatch.setattr(orchestrator_module, "capture_salvage_artifacts", fail_capture)

    with pytest.raises(QuotaResetExceededError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config)

    message = str(excinfo.value)
    assert "quota exhausted" in message
    assert "Rerun when quota resets" in message
    assert "Implementation salvage was attempted for issue implementation" in message
    assert "capture failed (simulated disk full)" in message
    assert "preserving the original agent failure" in message
    assert _salvage_dirs(config) == []
    assert "salvage capture failed (simulated disk full)" in capsys.readouterr().err


def test_failed_issue_implementation_without_diff_writes_no_salvage_patch(tmp_path):
    runner = FakeRunner(
        claude_outputs=["Created local notes but forgot the required marker."],
        post_agent_git_status=" M src/coding_review_agent_loop/cli.py\n",
        post_agent_git_diff="",
    )
    config = make_config(tmp_path, agent_max_retries=0)

    with pytest.raises(AgentLoopError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config)

    message = str(excinfo.value)
    assert "Implementation salvage was attempted for issue implementation" in message
    assert "no tracked/staged `git diff HEAD --binary` existed" in message
    assert "no patch artifacts were created" in message
    assert _salvage_dirs(config) == []


def test_failed_issue_implementation_with_untracked_only_diff_reports_untracked_only(tmp_path):
    runner = FakeRunner(
        claude_outputs=["Created local notes but forgot the required marker."],
        post_agent_git_status="?? scratch-note.md\n",
        post_agent_git_diff="",
    )
    config = make_config(tmp_path, agent_max_retries=0)

    with pytest.raises(AgentLoopError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config)

    message = str(excinfo.value)
    assert "only untracked files were present" in message
    assert "no tracked/staged `git diff HEAD --binary` existed" in message
    assert _salvage_dirs(config) == []


def test_plan_revision_quota_failure_reports_response_file_without_recording(tmp_path):
    valid_revision = structured_plan_revision(
        summary="Revised plan with a regression test.",
        prior_plan_item_dispositions=[
            {"item_id": "item-1", "disposition": "resolved", "note": "Added the test step."}
        ],
        plan_steps=["Add the regression test.", "Run the focused suite."],
    )
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            ("quota exceeded; reset in 10m", 1),
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Missing a regression test.",
                blocking_plan_issues=["Missing a regression test."],
            )
        ],
        public_response_outputs=["", "", valid_revision],
    )
    config = make_config(tmp_path, agent_max_retries=0)

    with pytest.raises(QuotaResetExceededError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    message = str(excinfo.value)
    assert "No implementation salvage was attempted because this was plan revision" in message
    assert "not a mutating implementation attempt" in message
    assert "A public response file exists at" in message
    assert "no result was recorded because the agent command exited with quota/session-limit status" in message
    assert "Revised plan with a regression test" not in "".join(runner.comments)


def test_plan_review_failure_without_response_file_reports_non_mutating_skip(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=["Plan review without the required structured response."],
    )
    config = make_config(tmp_path, agent_max_retries=0)

    with pytest.raises(AgentLoopError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    message = str(excinfo.value)
    assert "No implementation salvage was attempted because this was plan review" in message
    assert "not a mutating implementation attempt" in message
    assert "No non-empty public response file was produced at expected path" in message
    assert "no result was recorded because the public response failed validation" in message


def _write_salvage_summary(
    config,
    *,
    name,
    summary,
    created_at_ns,
    issue_number=56,
    scope="issue-implementation",
    approved_plan_hash_value=None,
):
    salvage_dir = config.log_dir / "salvage" / name
    salvage_dir.mkdir(parents=True)
    summary_path = salvage_dir / "salvage-summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "created_at_ns": created_at_ns,
        "repo": config.repo,
        "issue_number": issue_number,
        "scope": scope,
        "agent": "claude",
        "approved_plan_hash": approved_plan_hash_value,
        "summary": str(summary_path),
    }
    (salvage_dir / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_issue_implementation_rerun_prompt_includes_latest_salvage_summary(tmp_path):
    config = make_config(tmp_path)
    _write_salvage_summary(
        config,
        name="old",
        summary="old failed attempt summary",
        created_at_ns=1,
    )
    _write_salvage_summary(
        config,
        name="new",
        summary="new failed attempt summary\nPartial patch: `/tmp/new.patch`",
        created_at_ns=2,
    )
    runner = FakeRunner(
        claude_outputs=[
            "Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    coder_prompt = next(
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] and "Fix GitHub issue #56" in cmd[-1]
    )
    assert "Previous failed implementation attempt salvage:" in coder_prompt
    assert "new failed attempt summary" in coder_prompt
    assert "old failed attempt summary" not in coder_prompt
    assert "Do not auto-apply the patch" in coder_prompt
    assert "cherry-pick or ignore it" in coder_prompt
    assert "selectively" in coder_prompt


def test_latest_salvage_summary_filters_approved_plan_hash(tmp_path):
    config = make_config(tmp_path)
    plan_hash = approved_plan_hash("Plan:\n- Current.")
    _write_salvage_summary(
        config,
        name="old-plan",
        summary="stale approved-plan summary",
        created_at_ns=5,
        scope="approved-plan-implementation",
        approved_plan_hash_value=approved_plan_hash("Plan:\n- Old."),
    )
    _write_salvage_summary(
        config,
        name="current-plan",
        summary="current approved-plan summary",
        created_at_ns=4,
        scope="approved-plan-implementation",
        approved_plan_hash_value=plan_hash,
    )

    summary = latest_salvage_summary(
        config.log_dir,
        repo=config.repo,
        issue_number=56,
        scope="approved-plan-implementation",
        approved_plan_hash=plan_hash,
    )

    assert summary == "current approved-plan summary"
