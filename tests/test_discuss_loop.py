"""Integration tests for the discuss-mode orchestration loop."""
import hashlib
import json

import pytest

from coding_review_agent_loop.orchestrator import (
    DISCUSS_CONSENSUS_MARKER_RE,
    _discuss_subject,
    run_discuss_loop,
)
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import IssueComment, IssueContext

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


def test_discuss_loop_happy_path_two_implement_votes(tmp_path):
    implement_text = _discuss_review_text(outcome="implement")
    runner = FakeRunner(
        codex_outputs=[implement_text],
        gemini_outputs=[implement_text],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))

    result = run_discuss_loop(runner, issue_number=56, config=config)

    assert result == 0
    assert len(runner.comments) == 1
    assert "Consensus: Implement" in runner.comments[0]


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
    assert len(runner.comments) == 1
    assert "Consensus: Needs Human Review (Deadlock)" in runner.comments[0]
    assert "Consensus kind: `deadlock` after round 2." in runner.comments[0]
    assert "Codex held `implement`" in runner.comments[0]
    assert "Gemini held `do-not-implement`" in runner.comments[0]


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
    assert len(runner.comments) == 1
    assert "Consensus: Implement" in runner.comments[0]


def test_discuss_loop_consensus_comment_contains_subject_hash(tmp_path):
    subject = _issue_subject()
    implement_text = _discuss_review_text(outcome="implement")
    runner = FakeRunner(
        codex_outputs=[implement_text],
        gemini_outputs=[implement_text],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))

    run_discuss_loop(runner, issue_number=56, config=config)

    posted = runner.comments[0]
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
    assert len(runner.comments) == 1


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
    posted = runner.comments[0]
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
    posted = runner.comments[0]
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
    posted = runner.comments[0]
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


def test_discuss_loop_rejects_negative_discuss_max_rounds(tmp_path):
    runner = FakeRunner(codex_outputs=[], gemini_outputs=[])
    config = make_config(tmp_path, reviewer=("codex", "gemini"))

    with pytest.raises(AgentLoopError, match="--discuss-max-rounds must be zero or greater"):
        run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=-1)
