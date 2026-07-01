"""Integration tests for the discuss-mode orchestration loop."""
import hashlib
import json

import pytest

from coding_review_agent_loop.orchestrator import (
    DISCUSS_CONSENSUS_MARKER_RE,
    PostedRoundMetadata,
    ROUND_RESUME_MARKER_RE,
    _attach_round_metadata,
    _decode_round_metadata,
    _discuss_subject,
    render_public_agent_comment,
    run_discuss_loop,
)
from coding_review_agent_loop.comment_rendering import render_discuss_round_summary_comment
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import IssueComment, IssueContext
from coding_review_agent_loop.protocol import ParsedDiscussReview

from agent_loop_helpers import FakeRunner, make_config


def _discuss_review_text(
    *,
    outcome: str = "implement",
    rationale: str = "Well-scoped.",
    split_proposals: list[str] | None = None,
    rebuttal: str | None = None,
    reviewer: str = "OpenAI Codex",
) -> str:
    payload: dict = {
        "schema_version": 1,
        "kind": "discuss_review",
        "outcome": outcome,
        "rationale": rationale,
    }
    if split_proposals is not None:
        payload["split_proposals"] = split_proposals
    if rebuttal is not None:
        payload["rebuttal"] = rebuttal
    return json.dumps(payload) + f"\n<!-- AGENT_PLAN_STATE: approved -->\n-- {reviewer}"


def _issue_subject(title: str = "Fix issue-mode context", body: str = "Original issue body.") -> str:
    text = title + "\n\n" + body
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _seed_debater_comment(
    *,
    reviewer: str,
    round_number: int,
    subject: str,
    outcome: str = "implement",
    rationale: str = "Well-scoped.",
    split_proposals: list[str] | None = None,
    rebuttal: str | None = None,
    config=None,
) -> dict:
    """Build an issue-comment payload matching what `_run_discuss_loop` posts for a debater."""
    vote = ParsedDiscussReview(
        outcome=outcome,
        rationale=rationale,
        split_proposals=tuple(split_proposals or ()),
        reviewer=reviewer,
        rebuttal=rebuttal,
    )
    raw_text = _discuss_review_text(
        outcome=outcome,
        rationale=rationale,
        split_proposals=split_proposals,
        rebuttal=rebuttal,
        reviewer=reviewer,
    )
    body = render_public_agent_comment(
        kind="discuss_review",
        parsed=vote,
        agent=reviewer,
        config=config,
        round_number=round_number,
    )
    body = _attach_round_metadata(
        body,
        PostedRoundMetadata(
            flow="discuss",
            role="debater",
            agent=reviewer,
            round_number=round_number,
            subject=subject,
            raw_structured_coder_response=raw_text,
        ),
    )
    return {"author": {"login": "bot"}, "createdAt": "2026-01-01T00:00:00Z", "body": body}


def _seed_summary_comment(
    *,
    round_number: int,
    reviewer_votes: list[ParsedDiscussReview],
    is_final: bool,
    subject: str,
    outcome: str | None = None,
    consensus_kind: str | None = None,
    round_history: list[list[ParsedDiscussReview]] | None = None,
    split_proposals: list[str] | None = None,
    agenda: tuple[str, ...] = (),
) -> dict:
    """Build an issue-comment payload matching what `_run_discuss_loop` posts for a round summary."""
    body = render_discuss_round_summary_comment(
        round_number=round_number,
        reviewer_votes=reviewer_votes,
        is_final=is_final,
        subject=subject,
        outcome=outcome,
        consensus_kind=consensus_kind,
        round_history=round_history,
        split_proposals=split_proposals,
    )
    body = _attach_round_metadata(
        body,
        PostedRoundMetadata(
            flow="discuss",
            role="summary",
            agent="Orchestrator",
            round_number=round_number,
            subject=subject,
            is_final=is_final,
            consensus_kind=consensus_kind,
            agenda=agenda,
        ),
    )
    return {"author": {"login": "bot"}, "createdAt": "2026-01-01T00:00:00Z", "body": body}


def test_discuss_loop_happy_path_two_implement_votes(tmp_path):
    implement_text = _discuss_review_text(outcome="implement")
    runner = FakeRunner(
        codex_outputs=[implement_text],
        gemini_outputs=[implement_text],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))

    result = run_discuss_loop(runner, issue_number=56, config=config)

    assert result == 0
    assert len(runner.comments) == 3
    assert "Round 1: Codex position" in runner.comments[0]
    assert "Round 1: Gemini position" in runner.comments[1]
    assert "Consensus: Implement" in runner.comments[2]


def test_discuss_loop_debates_then_deadlocks_instead_of_veto(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            _discuss_review_text(outcome="implement"),
            _discuss_review_text(outcome="implement", rebuttal="The issue is scoped enough."),
        ],
        gemini_outputs=[
            _discuss_review_text(outcome="do-not-implement", rationale="Out of scope."),
            _discuss_review_text(
                outcome="do-not-implement",
                rationale="Still out of scope.",
                rebuttal="The scope objection still stands.",
            ),
        ],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))

    result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=1)

    assert result == 0
    assert len(runner.comments) == 6
    final = runner.comments[-1]
    assert "Consensus: Needs Human Review (Deadlock)" in final
    assert "Consensus kind: `deadlock` after round 2." in final
    assert "Codex held `implement`" in final
    assert "Gemini held `do-not-implement`" in final


def test_discuss_loop_idempotent_when_consensus_comment_exists(tmp_path):
    subject = _issue_subject()
    existing_body = f"## Consensus: Implement\n-- Orchestrator\n<!-- AGENT_DISCUSS_CONSENSUS: {subject} -->"
    runner = FakeRunner(
        issue_comments=[{"author": {"login": "bot"}, "createdAt": "2026-01-01T00:00:00Z", "body": existing_body}],
        codex_outputs=[],
        gemini_outputs=[],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))

    result = run_discuss_loop(runner, issue_number=56, config=config)

    assert result == 0
    assert len(runner.comments) == 0


def test_discuss_loop_reruns_when_subject_hash_differs(tmp_path):
    old_body = "## Consensus: Implement\n-- Orchestrator\n<!-- AGENT_DISCUSS_CONSENSUS: oldhashabc123 -->"
    implement_text = _discuss_review_text(outcome="implement")
    runner = FakeRunner(
        issue_comments=[{"author": {"login": "bot"}, "createdAt": "2026-01-01T00:00:00Z", "body": old_body}],
        codex_outputs=[implement_text],
        gemini_outputs=[implement_text],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))

    result = run_discuss_loop(runner, issue_number=56, config=config)

    assert result == 0
    assert len(runner.comments) == 3
    assert "Consensus: Implement" in runner.comments[-1]


def test_discuss_loop_consensus_comment_contains_subject_hash(tmp_path):
    subject = _issue_subject()
    implement_text = _discuss_review_text(outcome="implement")
    runner = FakeRunner(
        codex_outputs=[implement_text],
        gemini_outputs=[implement_text],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))

    run_discuss_loop(runner, issue_number=56, config=config)

    posted = runner.comments[-1]
    m = DISCUSS_CONSENSUS_MARKER_RE.search(posted)
    assert m is not None, "AGENT_DISCUSS_CONSENSUS marker not found in posted comment"
    assert m.group(1) == subject


def test_discuss_subject_changes_when_body_changes():
    ctx_a = IssueContext(
        number=1,
        repo="OWNER/REPO",
        title="My issue",
        body="Original body",
        url=None,
        comments=(),
    )
    ctx_b = IssueContext(
        number=1,
        repo="OWNER/REPO",
        title="My issue",
        body="Updated body",
        url=None,
        comments=(),
    )
    assert _discuss_subject(ctx_a) != _discuss_subject(ctx_b)


def test_discuss_subject_handles_none_fields():
    ctx = IssueContext(
        number=1,
        repo="OWNER/REPO",
        title=None,
        body=None,
        url=None,
        comments=(),
    )
    subject = _discuss_subject(ctx)
    expected = hashlib.sha256("".encode("utf-8")).hexdigest()
    assert subject == expected


def test_discuss_subject_changes_when_human_comment_added():
    ctx_no_comment = IssueContext(
        number=1, repo="OWNER/REPO", title="My issue", body="Body", url=None, comments=()
    )
    ctx_with_comment = IssueContext(
        number=1,
        repo="OWNER/REPO",
        title="My issue",
        body="Body",
        url=None,
        comments=(IssueComment(author="user", created_at=None, body="Clarifying comment"),),
    )
    assert _discuss_subject(ctx_no_comment) != _discuss_subject(ctx_with_comment)


def test_discuss_subject_excludes_consensus_comment_from_hash():
    ctx_base = IssueContext(
        number=1, repo="OWNER/REPO", title="My issue", body="Body", url=None, comments=()
    )
    subject = _discuss_subject(ctx_base)
    consensus_body = f"## Consensus: Implement\n-- Orchestrator\n<!-- AGENT_DISCUSS_CONSENSUS: {subject} -->"
    ctx_with_consensus = IssueContext(
        number=1,
        repo="OWNER/REPO",
        title="My issue",
        body="Body",
        url=None,
        comments=(IssueComment(author="bot", created_at=None, body=consensus_body),),
    )
    assert _discuss_subject(ctx_with_consensus) == subject


def test_discuss_subject_excludes_debater_and_summary_comments_from_hash():
    ctx_base = IssueContext(
        number=1, repo="OWNER/REPO", title="My issue", body="Body", url=None, comments=()
    )
    subject = _discuss_subject(ctx_base)
    debater_comment = _seed_debater_comment(reviewer="OpenAI Codex", round_number=1, subject=subject)
    summary_comment = _seed_summary_comment(
        round_number=1,
        reviewer_votes=[ParsedDiscussReview(outcome="implement", rationale="x", split_proposals=(), reviewer="OpenAI Codex")],
        is_final=False,
        subject=subject,
        agenda=("- OpenAI Codex held `implement`: x",),
    )
    ctx_with_rounds = IssueContext(
        number=1,
        repo="OWNER/REPO",
        title="My issue",
        body="Body",
        url=None,
        comments=(
            IssueComment(author="bot", created_at=None, body=debater_comment["body"]),
            IssueComment(author="bot", created_at=None, body=summary_comment["body"]),
        ),
    )
    assert _discuss_subject(ctx_with_rounds) == subject


def test_discuss_loop_reruns_when_human_comment_added(tmp_path):
    implement_text = _discuss_review_text(outcome="implement")
    old_subject = _issue_subject()
    old_consensus_body = f"## Consensus: Implement\n-- Orchestrator\n<!-- AGENT_DISCUSS_CONSENSUS: {old_subject} -->"
    new_comment = "Actually please also consider the auth flow."
    runner = FakeRunner(
        issue_comments=[
            {"author": {"login": "bot"}, "createdAt": "2026-01-01T00:00:01Z", "body": old_consensus_body},
            {"author": {"login": "human"}, "createdAt": "2026-01-01T00:01:00Z", "body": new_comment},
        ],
        codex_outputs=[implement_text],
        gemini_outputs=[implement_text],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))

    result = run_discuss_loop(runner, issue_number=56, config=config)

    assert result == 0
    assert len(runner.comments) == 3


def test_discuss_loop_split_consensus_includes_proposals(tmp_path):
    split_text = _discuss_review_text(
        outcome="split",
        rationale="Too broad.",
        split_proposals=["Auth flow", "Authorization checks"],
    )
    runner = FakeRunner(
        codex_outputs=[split_text],
        gemini_outputs=[split_text],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))

    result = run_discuss_loop(runner, issue_number=56, config=config)

    assert result == 0
    posted = runner.comments[-1]
    assert "Consensus: Split" in posted
    assert "Auth flow" in posted
    assert "Authorization checks" in posted


def test_discuss_loop_converges_after_debate(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            _discuss_review_text(outcome="implement"),
            _discuss_review_text(
                outcome="needs-human",
                rationale="The acceptance criteria need a product decision.",
                rebuttal="I agree the ambiguity blocks a technical verdict.",
            ),
        ],
        gemini_outputs=[
            _discuss_review_text(outcome="needs-human", rationale="Unclear requirements."),
            _discuss_review_text(
                outcome="needs-human",
                rationale="Unclear requirements remain.",
                rebuttal="The implement argument depends on unstated requirements.",
            ),
        ],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))

    result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=1)

    assert result == 0
    posted = runner.comments[-1]
    assert "Consensus: Needs Human Review" in posted
    assert "Consensus kind: `converged` after round 2." in posted
    assert "### Final rebuttals" in posted


def test_discuss_loop_zero_debate_rounds_deadlocks_after_initial_disagreement(tmp_path):
    runner = FakeRunner(
        codex_outputs=[_discuss_review_text(outcome="implement")],
        gemini_outputs=[_discuss_review_text(outcome="needs-human", rationale="Needs product input.")],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))

    result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=0)

    assert result == 0
    assert len(runner.comments) == 3
    posted = runner.comments[-1]
    assert "Consensus kind: `deadlock` after round 1." in posted
    assert "Codex held `implement`" in posted
    assert "Gemini held `needs-human`" in posted


def test_discuss_loop_passes_prior_round_snapshot_to_debate_prompts(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            _discuss_review_text(outcome="implement", rationale="Scoped."),
            _discuss_review_text(
                outcome="implement",
                rationale="Still scoped.",
                rebuttal="The split concern does not apply.",
            ),
        ],
        gemini_outputs=[
            _discuss_review_text(
                outcome="split",
                rationale="Too broad.",
                split_proposals=["Auth flow"],
            ),
            _discuss_review_text(
                outcome="implement",
                rationale="Can proceed.",
                rebuttal="I accept the scope argument.",
            ),
        ],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))

    run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=1)

    debate_commands = [cmd for cmd, _cwd in runner.commands if "debate round 2" in " ".join(cmd)]
    assert len(debate_commands) == 2
    for command in debate_commands:
        prompt = " ".join(command)
        assert "Codex: `implement`" in prompt
        assert "Gemini: `split`" in prompt
        assert "Auth flow" in prompt


def test_discuss_loop_round2_prompt_includes_round1_agenda(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            _discuss_review_text(outcome="implement", rationale="Scoped."),
            _discuss_review_text(
                outcome="implement",
                rationale="Still scoped.",
                rebuttal="The split concern does not apply.",
            ),
        ],
        gemini_outputs=[
            _discuss_review_text(outcome="do-not-implement", rationale="Out of scope entirely."),
            _discuss_review_text(
                outcome="do-not-implement",
                rationale="Still out of scope.",
                rebuttal="The scope objection still stands.",
            ),
        ],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))

    run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=1)

    debate_commands = [cmd for cmd, _cwd in runner.commands if "debate round 2" in " ".join(cmd)]
    assert len(debate_commands) == 2
    for command in debate_commands:
        prompt = " ".join(command)
        assert "Orchestrator's round summary and agenda for this round" in prompt
        assert "Out of scope entirely." in prompt


def test_discuss_loop_rejects_negative_discuss_max_rounds(tmp_path):
    runner = FakeRunner(codex_outputs=[], gemini_outputs=[])
    config = make_config(tmp_path, reviewer=("codex", "gemini"))

    with pytest.raises(AgentLoopError, match="--discuss-max-rounds must be zero or greater"):
        run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=-1)


def test_discuss_loop_resumes_missing_debater_mid_round(tmp_path):
    subject = _issue_subject()
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    seeded = _seed_debater_comment(
        reviewer="Codex",
        round_number=1,
        subject=subject,
        outcome="implement",
        rationale="Scoped.",
        config=config,
    )
    implement_text = _discuss_review_text(outcome="implement", rationale="Agreed.", reviewer="Gemini")
    runner = FakeRunner(
        issue_comments=[seeded],
        codex_outputs=[],
        gemini_outputs=[implement_text],
    )

    result = run_discuss_loop(runner, issue_number=56, config=config)

    assert result == 0
    # Only Gemini should have been invoked; Codex's posted round-1 vote is reused.
    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)
    assert len(runner.comments) == 2  # Gemini debater comment + final summary
    assert "Round 1: Gemini position" in runner.comments[0]
    assert "Consensus: Implement" in runner.comments[-1]


def test_discuss_loop_resumes_after_closed_non_final_round(tmp_path):
    subject = _issue_subject()
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    codex_vote_r1 = ParsedDiscussReview(outcome="implement", rationale="Scoped.", split_proposals=(), reviewer="Codex")
    gemini_vote_r1 = ParsedDiscussReview(outcome="do-not-implement", rationale="Out of scope.", split_proposals=(), reviewer="Gemini")
    agenda = (
        "- Codex held `implement`: Scoped.",
        "- Gemini held `do-not-implement`: Out of scope.",
    )
    seeded = [
        _seed_debater_comment(
            reviewer="Codex", round_number=1, subject=subject,
            outcome="implement", rationale="Scoped.", config=config,
        ),
        _seed_debater_comment(
            reviewer="Gemini", round_number=1, subject=subject,
            outcome="do-not-implement", rationale="Out of scope.", config=config,
        ),
        _seed_summary_comment(
            round_number=1,
            reviewer_votes=[codex_vote_r1, gemini_vote_r1],
            is_final=False,
            subject=subject,
            split_proposals=[],
            agenda=agenda,
        ),
    ]
    implement_text = _discuss_review_text(
        outcome="implement", rationale="Still scoped.", rebuttal="The split concern does not apply.",
    )
    runner = FakeRunner(
        issue_comments=seeded,
        codex_outputs=[implement_text],
        gemini_outputs=[implement_text],
    )

    result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=1)

    assert result == 0
    # Round 1 agents must not be re-invoked; only round 2's two debaters run.
    assert len(runner.comments) == 3  # 2 round-2 debater comments + final summary
    assert "Round 2: Codex position" in runner.comments[0]
    assert "Round 2: Gemini position" in runner.comments[1]
    final = runner.comments[-1]
    assert "Consensus: Implement" in final
    assert "### Round history" in final
    assert "Round 1: Codex: `implement`, Gemini: `do-not-implement`" in final
    assert "Round 2: Codex: `implement`, Gemini: `implement`" in final
    round2_commands = [
        cmd for cmd, _cwd in runner.commands if "debate round 2" in " ".join(cmd)
    ]
    assert len(round2_commands) == 2
    for command in round2_commands:
        prompt = " ".join(command)
        assert "Out of scope." in prompt


def test_discuss_loop_resume_reconstructs_full_round_history_across_multiple_closed_rounds(tmp_path):
    """Regression test for review blocking-1: the final summary after a resume must
    include every prior closed round, not just the round immediately before it."""
    subject = _issue_subject()
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    codex_vote_r1 = ParsedDiscussReview(outcome="implement", rationale="R1 scoped.", split_proposals=(), reviewer="Codex")
    gemini_vote_r1 = ParsedDiscussReview(outcome="do-not-implement", rationale="R1 out of scope.", split_proposals=(), reviewer="Gemini")
    codex_vote_r2 = ParsedDiscussReview(
        outcome="implement", rationale="R2 scoped.", split_proposals=(), reviewer="Codex",
        rebuttal="Still scoped.",
    )
    gemini_vote_r2 = ParsedDiscussReview(
        outcome="do-not-implement", rationale="R2 still out of scope.", split_proposals=(), reviewer="Gemini",
        rebuttal="Scope objection remains.",
    )
    seeded = [
        _seed_debater_comment(reviewer="Codex", round_number=1, subject=subject, outcome="implement", rationale="R1 scoped.", config=config),
        _seed_debater_comment(reviewer="Gemini", round_number=1, subject=subject, outcome="do-not-implement", rationale="R1 out of scope.", config=config),
        _seed_summary_comment(
            round_number=1,
            reviewer_votes=[codex_vote_r1, gemini_vote_r1],
            is_final=False,
            subject=subject,
            agenda=("- Codex held `implement`: R1 scoped.", "- Gemini held `do-not-implement`: R1 out of scope."),
        ),
        _seed_debater_comment(
            reviewer="Codex", round_number=2, subject=subject, outcome="implement",
            rationale="R2 scoped.", rebuttal="Still scoped.", config=config,
        ),
        _seed_debater_comment(
            reviewer="Gemini", round_number=2, subject=subject, outcome="do-not-implement",
            rationale="R2 still out of scope.", rebuttal="Scope objection remains.", config=config,
        ),
        _seed_summary_comment(
            round_number=2,
            reviewer_votes=[codex_vote_r2, gemini_vote_r2],
            is_final=False,
            subject=subject,
            agenda=("- Codex held `implement`: R2 scoped.", "- Gemini held `do-not-implement`: R2 still out of scope."),
        ),
    ]
    implement_text = _discuss_review_text(
        outcome="implement", rationale="R3 agreed.", rebuttal="Convinced by the narrowed scope.",
    )
    runner = FakeRunner(
        issue_comments=seeded,
        codex_outputs=[implement_text],
        gemini_outputs=[implement_text],
    )

    result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=2)

    assert result == 0
    assert len(runner.comments) == 3  # 2 round-3 debater comments + final summary
    final = runner.comments[-1]
    assert "Consensus: Implement" in final
    assert "Consensus kind: `converged` after round 3." in final
    assert "### Round history" in final
    assert "Round 1: Codex: `implement`, Gemini: `do-not-implement`" in final
    assert "Round 2: Codex: `implement`, Gemini: `do-not-implement`" in final
    assert "Round 3: Codex: `implement`, Gemini: `implement`" in final


def test_discuss_loop_idempotent_when_final_summary_metadata_exists(tmp_path):
    subject = _issue_subject()
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    codex_vote = ParsedDiscussReview(outcome="implement", rationale="Scoped.", split_proposals=(), reviewer="Codex")
    gemini_vote = ParsedDiscussReview(outcome="implement", rationale="Scoped.", split_proposals=(), reviewer="Gemini")
    seeded = [
        _seed_debater_comment(reviewer="Codex", round_number=1, subject=subject, outcome="implement", rationale="Scoped.", config=config),
        _seed_debater_comment(reviewer="Gemini", round_number=1, subject=subject, outcome="implement", rationale="Scoped.", config=config),
        _seed_summary_comment(
            round_number=1,
            reviewer_votes=[codex_vote, gemini_vote],
            is_final=True,
            subject=subject,
            outcome="implement",
            consensus_kind="unanimous",
            round_history=[[codex_vote, gemini_vote]],
            split_proposals=[],
        ),
    ]
    runner = FakeRunner(issue_comments=seeded, codex_outputs=[], gemini_outputs=[])

    result = run_discuss_loop(runner, issue_number=56, config=config)

    assert result == 0
    assert len(runner.comments) == 0
    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)
    assert not any(cmd[:1] == ["gemini"] for cmd, _cwd in runner.commands)


def test_discuss_loop_finalizes_from_last_completed_round_when_resumed_round_exceeds_lowered_max_rounds(tmp_path):
    """Regression test: resuming a non-final round with a since-lowered
    --discuss-max-rounds must post a final deadlock summary instead of silently
    returning 0 with no final summary (review blocking-1 on PR #471)."""
    subject = _issue_subject()
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    codex_vote_r1 = ParsedDiscussReview(outcome="implement", rationale="Scoped.", split_proposals=(), reviewer="Codex")
    gemini_vote_r1 = ParsedDiscussReview(outcome="do-not-implement", rationale="Out of scope.", split_proposals=(), reviewer="Gemini")
    seeded = [
        _seed_debater_comment(reviewer="Codex", round_number=1, subject=subject, outcome="implement", rationale="Scoped.", config=config),
        _seed_debater_comment(reviewer="Gemini", round_number=1, subject=subject, outcome="do-not-implement", rationale="Out of scope.", config=config),
        _seed_summary_comment(
            round_number=1,
            reviewer_votes=[codex_vote_r1, gemini_vote_r1],
            is_final=False,
            subject=subject,
            agenda=("- Codex held `implement`: Scoped.", "- Gemini held `do-not-implement`: Out of scope."),
        ),
    ]
    # discuss_max_rounds was previously left at the default (>=1 debate round), but this
    # run uses 0, so the resumed round 2 is no longer allowed.
    runner = FakeRunner(issue_comments=seeded, codex_outputs=[], gemini_outputs=[])

    result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=0)

    assert result == 0
    # No debaters should be re-invoked for a round that is no longer allowed.
    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)
    assert not any(cmd[:1] == ["gemini"] for cmd, _cwd in runner.commands)
    assert len(runner.comments) == 1
    final = runner.comments[-1]
    assert "Consensus: Needs Human Review (Deadlock)" in final
    assert "Consensus kind: `deadlock` after round 1." in final
    assert "Codex held `implement`: Scoped." in final
    assert "Gemini held `do-not-implement`: Out of scope." in final
    m = DISCUSS_CONSENSUS_MARKER_RE.search(final)
    assert m is not None
    assert m.group(1) == subject


def test_discuss_loop_raises_when_resumed_round_exceeds_max_rounds_with_no_completed_round(tmp_path):
    """Defensive guard: if resume ever reports a start round beyond the configured
    limit with no completed round history to finalize from (e.g. corrupted/partial
    metadata with no round-1 records at all), fail loudly instead of silently
    returning 0 with no final summary."""
    subject = _issue_subject()
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    seeded = [
        _seed_debater_comment(
            reviewer="Codex", round_number=2, subject=subject, outcome="implement",
            rationale="Scoped.", rebuttal="Still scoped.", config=config,
        ),
    ]
    runner = FakeRunner(issue_comments=seeded, codex_outputs=[], gemini_outputs=[])

    with pytest.raises(AgentLoopError, match="discuss: resumed state expects round"):
        run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=0)
