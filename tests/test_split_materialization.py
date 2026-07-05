import pytest

from coding_review_agent_loop.cli import AgentLoopError
from coding_review_agent_loop.github import (
    IssueComment,
    find_open_pr_referencing_issue,
    validate_pr_body_does_not_close_issue,
)
from coding_review_agent_loop.split_materialization import (
    MAX_SPLIT_CHILDREN,
    dedupe_split_stage_proposals,
    find_existing_split_materialization,
    find_existing_split_stage_handoff,
    format_split_stage_handoff_comment,
    format_unfiled_split_warning,
    has_unfiled_split_warning,
    materialize_split_proposals,
    post_split_stage_handoff_comment,
    resolve_selected_stage_child,
    split_stage_proposal_from_deferred_stage,
    split_stage_proposal_from_text,
)
from coding_review_agent_loop.protocol import DeferredStage
from agent_loop_helpers import FakeRunner, make_config


def _comment(body: str) -> IssueComment:
    return IssueComment(author="bot", created_at="2026-05-23T00:00:00Z", body=body)


def test_split_stage_proposal_from_text_uses_first_line_as_title():
    proposal = split_stage_proposal_from_text("Auth flow overhaul\n\nMore detail below.")
    assert proposal.title == "Auth flow overhaul"
    assert "More detail below." in proposal.body


def test_split_stage_proposal_from_deferred_stage_uses_title_and_summary():
    stage = DeferredStage(title="Billing follow-up", summary="Split billing reconciliation out.")
    proposal = split_stage_proposal_from_deferred_stage(stage)
    assert proposal.title == "Billing follow-up"
    assert proposal.body == "Split billing reconciliation out."


def test_dedupe_split_stage_proposals_keeps_first_occurrence():
    proposals = [
        split_stage_proposal_from_text("Auth flow"),
        split_stage_proposal_from_text("auth   FLOW"),
        split_stage_proposal_from_text("Billing flow"),
    ]
    deduped = dedupe_split_stage_proposals(proposals)
    assert [p.title for p in deduped] == ["Auth flow", "Billing flow"]


def test_materialize_split_proposals_creates_children_with_markers(tmp_path):
    runner = FakeRunner(
        issue_urls=[
            "https://github.com/OWNER/REPO/issues/101",
            "https://github.com/OWNER/REPO/issues/102",
        ]
    )
    config = make_config(tmp_path)
    proposals = [
        split_stage_proposal_from_text("Auth flow"),
        split_stage_proposal_from_text("Billing flow"),
    ]

    metadata = materialize_split_proposals(
        runner,
        config=config,
        parent_issue=56,
        subject="subject-a",
        proposals=proposals,
        rationale=[("OpenAI Codex", "Too broad for one PR.")],
        issue_comments=(),
    )

    assert metadata is not None
    assert len(runner.issues) == 2
    assert runner.issues[0]["title"] == "[#56 stage] Auth flow"
    body0 = runner.issues[0]["body"]
    assert "Part of #56" in body0
    assert "Too broad for one PR." in body0
    assert "Refs #56" in body0
    assert "Do not use closing keywords" in body0
    assert "<!-- AGENT_SPLIT_CHILD: parent=56 key=" in body0
    # Second child's body lists the first as a sibling.
    body1 = runner.issues[1]["body"]
    assert "Auth flow" in body1

    assert [child.origin for child in metadata.children] == ["created", "created"]
    parent_summary = runner.comments[-1]
    assert "<!-- AGENT_DISCUSS_SPLIT:" in parent_summary
    assert "https://github.com/OWNER/REPO/issues/101" in parent_summary
    assert "https://github.com/OWNER/REPO/issues/102" in parent_summary


def test_materialize_split_proposals_rerun_with_existing_marker_creates_nothing(tmp_path):
    runner = FakeRunner(
        issue_urls=[
            "https://github.com/OWNER/REPO/issues/101",
            "https://github.com/OWNER/REPO/issues/102",
        ]
    )
    config = make_config(tmp_path)
    proposals = [
        split_stage_proposal_from_text("Auth flow"),
        split_stage_proposal_from_text("Billing flow"),
    ]
    first = materialize_split_proposals(
        runner,
        config=config,
        parent_issue=56,
        subject="subject-a",
        proposals=proposals,
        issue_comments=(),
    )
    prior_comments = (_comment(runner.comments[-1]),)

    second = materialize_split_proposals(
        runner,
        config=config,
        parent_issue=56,
        subject="subject-a",
        proposals=proposals,
        issue_comments=prior_comments,
    )

    assert len(runner.issues) == 2  # no new issues created
    assert len(runner.comments) == 1  # no new parent comment posted
    assert second == first


def test_materialize_split_proposals_partial_failure_adopts_existing_child(tmp_path):
    proposal_auth = split_stage_proposal_from_text("Auth flow")
    proposal_billing = split_stage_proposal_from_text("Billing flow")
    runner = FakeRunner(
        issue_urls=["https://github.com/OWNER/REPO/issues/102"],
        search_issues_payload=[
            {
                "number": 101,
                "title": "[#56 stage] Auth flow",
                "url": "https://github.com/OWNER/REPO/issues/101",
                "body": f"Part of #56\n\n<!-- AGENT_SPLIT_CHILD: parent=56 key={proposal_auth.key} -->",
            }
        ],
    )
    config = make_config(tmp_path)

    metadata = materialize_split_proposals(
        runner,
        config=config,
        parent_issue=56,
        subject="subject-a",
        proposals=[proposal_auth, proposal_billing],
        issue_comments=(),
    )

    assert runner.search_issues_calls == ['"[#56 stage]" in:title']
    # Only the unmatched (billing) proposal was created; auth was adopted.
    assert len(runner.issues) == 1
    assert runner.issues[0]["title"] == "[#56 stage] Billing flow"
    origins = {child.title: child.origin for child in metadata.children}
    assert origins["Auth flow"] == "adopted"
    assert origins["Billing flow"] == "created"
    parent_summary = runner.comments[-1]
    assert "https://github.com/OWNER/REPO/issues/101" in parent_summary
    assert "adopted" in parent_summary


def test_materialize_split_proposals_caps_at_max_children(tmp_path):
    proposals = [split_stage_proposal_from_text(f"Stage {i}") for i in range(MAX_SPLIT_CHILDREN + 3)]
    runner = FakeRunner(
        issue_urls=[f"https://github.com/OWNER/REPO/issues/{100 + i}" for i in range(MAX_SPLIT_CHILDREN)]
    )
    config = make_config(tmp_path)

    metadata = materialize_split_proposals(
        runner,
        config=config,
        parent_issue=56,
        subject="subject-a",
        proposals=proposals,
        issue_comments=(),
    )

    assert len(runner.issues) == MAX_SPLIT_CHILDREN
    assert len(metadata.children) == MAX_SPLIT_CHILDREN


def test_materialize_split_proposals_rerun_does_not_exceed_cap(tmp_path):
    """A rerun with the same over-cap proposal set must not keep filing new
    children past MAX_SPLIT_CHILDREN: remaining capacity must be computed from
    children already recorded in the parent's cumulative metadata, not just
    the proposals this call still considers unresolved (#492 review)."""
    proposals = [split_stage_proposal_from_text(f"Stage {i}") for i in range(MAX_SPLIT_CHILDREN + 3)]
    config = make_config(tmp_path)

    first_runner = FakeRunner(
        issue_urls=[f"https://github.com/OWNER/REPO/issues/{100 + i}" for i in range(MAX_SPLIT_CHILDREN)]
    )
    first_metadata = materialize_split_proposals(
        first_runner,
        config=config,
        parent_issue=56,
        subject="subject-a",
        proposals=proposals,
        issue_comments=(),
    )
    assert len(first_metadata.children) == MAX_SPLIT_CHILDREN
    parent_summary_comment = _comment(first_runner.comments[-1])

    # Rerun with the identical over-cap proposal set, seeded with the parent
    # comment history left behind by the first run.
    second_runner = FakeRunner()
    second_metadata = materialize_split_proposals(
        second_runner,
        config=config,
        parent_issue=56,
        subject="subject-a",
        proposals=proposals,
        issue_comments=[parent_summary_comment],
    )

    assert second_runner.issues == []
    assert second_metadata.children == first_metadata.children
    assert len(second_metadata.children) == MAX_SPLIT_CHILDREN


def test_materialize_split_proposals_returns_none_for_no_proposals(tmp_path):
    runner = FakeRunner()
    config = make_config(tmp_path)
    assert (
        materialize_split_proposals(
            runner, config=config, parent_issue=56, subject="s", proposals=(), issue_comments=()
        )
        is None
    )


def test_materialize_split_proposals_dry_run_previews_search_and_create(tmp_path):
    runner = FakeRunner()
    config = make_config(tmp_path, dry_run=True)
    proposals = [split_stage_proposal_from_text("Auth flow")]

    materialize_split_proposals(
        runner,
        config=config,
        parent_issue=56,
        subject="subject-a",
        proposals=proposals,
        issue_comments=(),
    )

    # Dry-run still previews the `gh issue list --search` and `gh issue create`
    # commands (so the materialization path is visible), it just never
    # persists application-level state outside of GitHub CLI echo commands.
    assert runner.search_issues_calls == ['"[#56 stage]" in:title']
    assert len(runner.issues) == 1


def test_find_existing_split_materialization_rejects_invalid_marker():
    with pytest.raises(AgentLoopError, match="Invalid AGENT_DISCUSS_SPLIT payload"):
        find_existing_split_materialization(
            (_comment("<!-- AGENT_DISCUSS_SPLIT: not-valid-base64 -->"),),
            parent_issue=56,
        )


def test_format_unfiled_split_warning_lists_titles():
    proposals = [split_stage_proposal_from_text("Auth flow"), split_stage_proposal_from_text("Billing flow")]
    warning = format_unfiled_split_warning(proposals)
    assert "NOT filed as issues" in warning
    assert "Auth flow" in warning
    assert "Billing flow" in warning
    assert "--materialize-split-issues" in warning


def test_has_unfiled_split_warning_matches_issue_and_subject():
    body = (
        "### Split follow-ups are NOT filed as issues\n\n"
        "<!-- AGENT_SPLIT_UNFILED_WARNING: issue=56 subject=abc123 -->\n"
        "-- coding-review-agent-loop"
    )
    assert has_unfiled_split_warning((_comment(body),), issue_number=56, subject="abc123")
    assert not has_unfiled_split_warning((_comment(body),), issue_number=56, subject="different")
    assert not has_unfiled_split_warning((_comment(body),), issue_number=99, subject="abc123")


def test_resolve_selected_stage_child_by_split_stage_flag():
    from coding_review_agent_loop.split_materialization import MaterializedSplitChild

    children = (
        MaterializedSplitChild(title="Auth flow", key="k1", url=None, number=101, origin="created"),
        MaterializedSplitChild(title="Billing flow", key="k2", url=None, number=102, origin="created"),
    )
    selected = resolve_selected_stage_child(
        children, parent_issue=56, plan_title_or_subject="Something else entirely", split_stage_flag=102
    )
    assert selected.number == 102


def test_resolve_selected_stage_child_rejects_unknown_split_stage_flag():
    from coding_review_agent_loop.split_materialization import MaterializedSplitChild

    children = (MaterializedSplitChild(title="Auth flow", key="k1", url=None, number=101, origin="created"),)
    with pytest.raises(AgentLoopError, match="does not match any child issue"):
        resolve_selected_stage_child(
            children, parent_issue=56, plan_title_or_subject="Auth flow", split_stage_flag=999
        )


def test_resolve_selected_stage_child_by_unique_title_match():
    proposal = split_stage_proposal_from_text("Auth flow")
    from coding_review_agent_loop.split_materialization import MaterializedSplitChild

    children = (
        MaterializedSplitChild(title="Auth flow", key=proposal.key, url=None, number=101, origin="created"),
        MaterializedSplitChild(title="Billing flow", key="other-key", url=None, number=102, origin="created"),
    )
    selected = resolve_selected_stage_child(
        children, parent_issue=56, plan_title_or_subject="Auth flow", split_stage_flag=None
    )
    assert selected.number == 101


def test_resolve_selected_stage_child_fails_without_match_or_flag():
    from coding_review_agent_loop.split_materialization import MaterializedSplitChild

    children = (MaterializedSplitChild(title="Auth flow", key="k1", url=None, number=101, origin="created"),)
    with pytest.raises(AgentLoopError, match="--split-stage"):
        resolve_selected_stage_child(
            children, parent_issue=56, plan_title_or_subject="Something unrelated", split_stage_flag=None
        )


def test_split_stage_handoff_comment_roundtrip(tmp_path):
    from coding_review_agent_loop.split_materialization import MaterializedSplitChild

    runner = FakeRunner()
    config = make_config(tmp_path)
    child = MaterializedSplitChild(
        title="Auth flow",
        key="k1",
        url="https://github.com/OWNER/REPO/issues/101",
        number=101,
        origin="created",
    )
    post_split_stage_handoff_comment(runner, config=config, parent_issue=56, plan_hash="hash1", child=child)

    posted = runner.comments[-1]
    assert "handed off to split stage" in posted
    comments = (_comment(posted),)
    found = find_existing_split_stage_handoff(comments, parent_issue=56, plan_hash="hash1")
    assert found is not None
    assert found.child_issue_number == 101
    assert find_existing_split_stage_handoff(comments, parent_issue=56, plan_hash="different") is None


def test_format_split_stage_handoff_comment_requires_child_number():
    from coding_review_agent_loop.split_materialization import MaterializedSplitChild

    child = MaterializedSplitChild(title="Auth flow", key="k1", url=None, number=None, origin="created")
    with pytest.raises(AgentLoopError, match="child issue number is unavailable"):
        format_split_stage_handoff_comment(parent_issue=56, plan_hash="hash1", child=child)


def test_validate_pr_body_does_not_close_issue_rejects_closing_keyword(tmp_path):
    runner = FakeRunner(pr_payload={"body": "Closes #56\n\nRefs #56"})
    config = make_config(tmp_path)
    with pytest.raises(AgentLoopError, match="closing keyword"):
        validate_pr_body_does_not_close_issue(runner, config=config, pr_number=77, issue_number=56)


def test_validate_pr_body_does_not_close_issue_accepts_refs_only(tmp_path):
    runner = FakeRunner(pr_payload={"body": "Closes #99\n\nRefs #56"})
    config = make_config(tmp_path)
    validate_pr_body_does_not_close_issue(runner, config=config, pr_number=77, issue_number=56)


def test_validate_pr_body_does_not_close_issue_skips_in_dry_run(tmp_path):
    runner = FakeRunner(pr_payload={"body": "Fixes #56"})
    config = make_config(tmp_path, dry_run=True)
    validate_pr_body_does_not_close_issue(runner, config=config, pr_number=77, issue_number=56)


def test_find_open_pr_referencing_issue_returns_none_when_no_match(tmp_path):
    runner = FakeRunner(open_prs_payload=[{"number": 12, "body": "Unrelated change."}])
    config = make_config(tmp_path)
    assert find_open_pr_referencing_issue(runner, config=config, issue_number=56) is None


def test_find_open_pr_referencing_issue_returns_none_when_no_open_prs(tmp_path):
    runner = FakeRunner(open_prs_payload=[])
    config = make_config(tmp_path)
    assert find_open_pr_referencing_issue(runner, config=config, issue_number=56) is None


def test_find_open_pr_referencing_issue_matches_direct_reference(tmp_path):
    runner = FakeRunner(
        open_prs_payload=[
            {"number": 12, "body": "Unrelated change."},
            {"number": 492, "body": "Implements the approved plan.\n\nFixes #476"},
        ]
    )
    config = make_config(tmp_path)
    assert find_open_pr_referencing_issue(runner, config=config, issue_number=476) == 492


def test_find_open_pr_referencing_issue_matches_issue_url_reference(tmp_path):
    runner = FakeRunner(
        open_prs_payload=[
            {"number": 492, "body": "See https://github.com/OWNER/REPO/issues/476 for context."},
        ]
    )
    config = make_config(tmp_path)
    assert find_open_pr_referencing_issue(runner, config=config, issue_number=476) == 492


def test_find_open_pr_referencing_issue_raises_on_ambiguous_matches(tmp_path):
    runner = FakeRunner(
        open_prs_payload=[
            {"number": 492, "body": "Fixes #476"},
            {"number": 494, "body": "Closes #476"},
        ]
    )
    config = make_config(tmp_path)
    with pytest.raises(AgentLoopError, match=r"Multiple open PRs \(#492, #494\)"):
        find_open_pr_referencing_issue(runner, config=config, issue_number=476)


def test_find_open_pr_referencing_issue_skips_in_dry_run(tmp_path):
    runner = FakeRunner(open_prs_payload=[{"number": 492, "body": "Fixes #476"}])
    config = make_config(tmp_path, dry_run=True)
    assert find_open_pr_referencing_issue(runner, config=config, issue_number=476) is None
    assert runner.open_prs_calls == 0
