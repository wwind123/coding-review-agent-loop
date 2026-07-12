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
    _encode_round_metadata,
    _discuss_subject,
    _validate_discuss_analyzer_agenda_fidelity,
    _validate_discuss_final_analyzer_fidelity,
    render_public_agent_comment,
    run_discuss_loop,
    _detect_discuss_answer_consensus,
)
from coding_review_agent_loop.comment_rendering import render_discuss_round_summary_comment
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import IssueComment, IssueContext
from coding_review_agent_loop.protocol import (
    DiscussAgendaDisagreement,
    DiscussSourcedFact,
    ParsedDiscussAgenda,
    ParsedDiscussReview,
    ParsedDiscussAnswer,
    DiscussUnresolvedItem,
)

from agent_loop_helpers import FakeRunner, make_config


def test_answer_consensus_does_not_escalate_for_one_early_needs_human():
    responses = [
        ParsedDiscussAnswer("needs-human", "unclear", "low", (DiscussUnresolvedItem("human-decision", "scope"),), "Codex"),
        ParsedDiscussAnswer("answer", "clear", "medium", (), "Claude", answer="Use an API."),
    ]
    assert _detect_discuss_answer_consensus(responses) is None


def test_answer_consensus_escalates_when_all_debaters_request_human_decision():
    responses = [
        ParsedDiscussAnswer("needs-human", "unclear", "low", (DiscussUnresolvedItem("human-decision", "scope"),), "Codex"),
        ParsedDiscussAnswer("needs-human", "unclear", "low", (DiscussUnresolvedItem("human-decision", "security"),), "Claude"),
    ]
    assert _detect_discuss_answer_consensus(responses) == ("needs-human", [])


def test_answer_round_metadata_round_trips_mode_and_legacy_defaults_to_triage():
    metadata = PostedRoundMetadata(
        flow="discuss", role="debater", agent="Codex", round_number=1,
        subject="subject", result_mode="answer", research_mode="auto",
    )
    decoded = _decode_round_metadata(_encode_round_metadata(metadata))
    assert decoded.result_mode == "answer"
    assert decoded.research_mode == "auto"
    legacy = _decode_round_metadata(_encode_round_metadata(
        PostedRoundMetadata(flow="discuss", role="debater", agent="Codex", round_number=1, subject="subject")
    ))
    assert legacy.result_mode == "triage"


def _discuss_review_text(
    *,
    outcome: str = "implement",
    rationale: str = "Well-scoped.",
    split_proposals: list[str] | None = None,
    rebuttal: str | None = None,
    reviewer: str = "OpenAI Codex",
    analyzer_framing: str | None = None,
    framing_note: str | None = None,
    research: dict | None = None,
    evidence: dict | None = None,
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
    if analyzer_framing is not None:
        payload["analyzer_framing"] = analyzer_framing
    if framing_note is not None:
        payload["framing_note"] = framing_note
    if research is not None:
        payload["research"] = research
    if evidence is not None:
        payload["evidence"] = evidence
    return json.dumps(payload) + f"\n<!-- AGENT_PLAN_STATE: approved -->\n-- {reviewer}"


def _checkout_inspected_evidence(source: str) -> dict:
    return {
        "claims": [
            {
                "fact": "The referenced line supports this outcome.",
                "status": "verified",
                "source": source,
                "verification_basis": "checkout-inspected",
            }
        ],
        "updates": [],
    }


def _discuss_answer_text(
    *,
    answer: str | None = "Use an API boundary.",
    position: str = "answer",
    rationale: str = "It keeps the integration replaceable.",
    confidence: str = "medium",
    open_questions: list[str] | None = None,
    unresolved_items: list[dict[str, str]] | None = None,
    rebuttal: str | None = None,
    research: dict | None = None,
    evidence: dict | None = None,
    reviewer: str = "OpenAI Codex",
) -> str:
    payload: dict = {
        "schema_version": 1,
        "kind": "discuss_answer",
        "position": position,
        "rationale": rationale,
        "confidence": confidence,
        "unresolved_items": unresolved_items if unresolved_items is not None else [
            {"status": "human-decision" if position == "needs-human" else "blocker", "text": question}
            for question in (open_questions or [])
        ],
    }
    if answer is not None:
        payload["answer"] = answer
    if rebuttal is not None:
        payload["rebuttal"] = rebuttal
    if research is not None:
        payload["research"] = research
    if evidence is not None:
        payload["evidence"] = evidence
    return json.dumps(payload) + f"\n<!-- AGENT_PLAN_STATE: approved -->\n-- {reviewer}"


def _semantic_comparison_text(
    *, classification: str, shared_recommendation: str,
    remaining_decisions: list[str] | None = None,
) -> str:
    payload = {
        "schema_version": 1,
        "kind": "discuss_semantic_comparison",
        "classification": classification,
        "shared_recommendation": shared_recommendation,
        "remaining_decisions": remaining_decisions or [],
        "evidence": [
            {"reviewer": "Codex", "supports": "Supports the shared recommendation."},
            {"reviewer": "Gemini", "supports": "Supports the shared recommendation."},
        ],
    }
    return json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- Anthropic Claude"


def _answer_confirmation_text(*, reviewer: str, decision: str = "confirm", answer: str | None = None) -> str:
    payload = {
        "schema_version": 1,
        "kind": "discuss_answer_confirmation",
        "reviewer": reviewer,
        "decision": decision,
        "rationale": "The recommendation is acceptable.",
    }
    if answer is not None:
        payload["answer"] = answer
    return json.dumps(payload) + f"\n<!-- AGENT_PLAN_STATE: approved -->\n-- {reviewer}"


def _discuss_agenda_text(
    *,
    consensus: list[str] | None = None,
    disagreements: list[dict] | None = None,
    missing_facts: list[str] | None = None,
    analyzer: str = "Anthropic Claude",
    research_required: bool | None = None,
    research_questions: list[str] | None = None,
) -> str:
    payload: dict = {
        "schema_version": 1,
        "kind": "discuss_agenda",
        "consensus": consensus if consensus is not None else ["The issue is well-motivated."],
        "disagreements": disagreements
        if disagreements is not None
        else [
            {
                "topic": "Scope of the change",
                "positions": {"Codex": "Narrow enough.", "Gemini": "Too broad."},
                "question_for_next_round": "Would splitting resolve the scope objection?",
            }
        ],
        "missing_facts": missing_facts if missing_facts is not None else [],
    }
    if research_required is not None:
        payload["research_required"] = research_required
    if research_questions is not None:
        payload["research_questions"] = research_questions
    return json.dumps(payload) + f"\n<!-- AGENT_PLAN_STATE: approved -->\n-- {analyzer}"


def _issue_subject(title: str = "Fix issue-mode context", body: str = "Original issue body.") -> str:
    text = title + "\n\n" + body
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _grounded_agenda_issue_payload() -> dict[str, str]:
    return {
        "body": (
            "Original issue body. The issue is well-motivated. Scope of the change. "
            "Would splitting resolve the scope objection? Whether the API boundary is specified. "
            "Narrow enough. Too broad. "
            "Round-one agenda marker. Round-two agenda marker. "
            "Everyone agrees to implement. Is Gemini CLI still available for enterprise users?"
        )
    }


def _seed_debater_comment(
    *,
    reviewer: str,
    round_number: int,
    subject: str,
    outcome: str = "implement",
    rationale: str = "Well-scoped.",
    split_proposals: list[str] | None = None,
    rebuttal: str | None = None,
    evidence: dict | None = None,
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
        evidence=evidence,
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


def _seed_answer_debater_comment(
    *, reviewer: str, round_number: int, subject: str, answer: str,
    rationale: str = "The evidence supports this recommendation.",
    rebuttal: str | None = None, evidence: dict | None = None, config=None, legacy: bool = False,
) -> dict:
    vote = ParsedDiscussAnswer(
        position="answer", rationale=rationale, confidence="medium",
        unresolved_items=(), reviewer=reviewer, answer=answer, rebuttal=rebuttal,
    )
    if legacy:
        raw_text = json.dumps({
            "schema_version": 1, "kind": "discuss_answer", "position": "answer",
            "answer": answer, "rationale": rationale, "confidence": "medium",
            "open_questions": ["Verify availability."],
        }) + f"\n<!-- AGENT_PLAN_STATE: approved -->\n-- {reviewer}"
    else:
        raw_text = _discuss_answer_text(
            answer=answer, rationale=rationale, rebuttal=rebuttal, reviewer=reviewer,
            evidence=evidence,
        )
    body = render_public_agent_comment(
        kind="discuss_answer", parsed=vote, agent=reviewer, config=config,
        round_number=round_number,
    )
    body = _attach_round_metadata(
        body,
        PostedRoundMetadata(
            flow="discuss", role="debater", agent=reviewer,
            round_number=round_number, subject=subject, result_mode="answer",
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
    analyzer_response: str | None = None,
    failed_debaters: tuple[tuple[str, str], ...] = (),
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
        failed_debaters=failed_debaters,
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
            analyzer_response=analyzer_response,
            failed_debaters=failed_debaters,
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


def test_discuss_loop_unanimous_answer_is_not_triage(tmp_path):
    answer_text = _discuss_answer_text()
    runner = FakeRunner(codex_outputs=[answer_text], gemini_outputs=[answer_text])
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_result_mode="answer")

    assert run_discuss_loop(runner, issue_number=56, config=config) == 0
    assert "Consensus Answer" in runner.comments[-1]
    assert "Use an API boundary." in runner.comments[-1]
    assert "Consensus: Implement" not in runner.comments[-1]


def test_answer_mode_followups_succeed_but_material_items_select_final_outcome(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_result_mode="answer")
    followups = FakeRunner(
        codex_outputs=[_discuss_answer_text(unresolved_items=[{"status": "follow-up", "text": "Refresh pricing."}])],
        gemini_outputs=[_discuss_answer_text(unresolved_items=[{"status": "follow-up", "text": "Refresh pricing."}])],
    )
    assert run_discuss_loop(followups, issue_number=56, config=config, discuss_max_rounds=0) == 0
    assert "Consensus Answer" in followups.comments[-1]
    assert "Non-blocking follow-ups" in followups.comments[-1]

    material = FakeRunner(
        codex_outputs=[_discuss_answer_text(unresolved_items=[{"status": "blocker", "text": "Verify capability."}])],
        gemini_outputs=[_discuss_answer_text(unresolved_items=[{"status": "human-decision", "text": "Choose a product tier."}])],
    )
    assert run_discuss_loop(material, issue_number=56, config=config, discuss_max_rounds=0) == 0
    final = material.comments[-1]
    assert "Needs Human Decision" in final
    assert "Blockers" in final and "Human decisions" in final


def test_answer_mode_blocker_suppresses_exact_and_semantic_convergence(tmp_path):
    runner = FakeRunner(
        codex_outputs=[_discuss_answer_text(unresolved_items=[{"status": "blocker", "text": "Verify capability."}])],
        gemini_outputs=[_discuss_answer_text(unresolved_items=[{"status": "blocker", "text": "Verify capability."}])],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_result_mode="answer")
    assert run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=0) == 0
    assert "Deadlock" in runner.comments[-1]
    assert "Verify capability." in runner.comments[-1]


def test_discuss_loop_answer_converges_after_rebuttal(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            _discuss_answer_text(answer="Use an API boundary.", reviewer="OpenAI Codex"),
            _discuss_answer_text(answer="Use an API boundary.", rebuttal="The concern is addressed.", reviewer="OpenAI Codex"),
        ],
        gemini_outputs=[
            _discuss_answer_text(answer="Use a shared library.", reviewer="Google Gemini"),
            _discuss_answer_text(answer="Use an API boundary.", rebuttal="The boundary avoids coupling.", reviewer="Google Gemini"),
        ],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_result_mode="answer")

    assert run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=1) == 0
    final = runner.comments[-1]
    assert "Converged Answer" in final
    assert "Use an API boundary." in final
    assert "Consensus kind: `converged`" in final


def test_discuss_loop_answer_disagreement_is_deadlock_and_unanimous_escalation_is_human(tmp_path):
    disagreement = FakeRunner(
        codex_outputs=[_discuss_answer_text(answer="Use an API.", reviewer="OpenAI Codex")],
        gemini_outputs=[_discuss_answer_text(answer="Use a library.", reviewer="Google Gemini")],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_result_mode="answer")
    assert run_discuss_loop(disagreement, issue_number=56, config=config, discuss_max_rounds=0) == 0
    assert "Deadlock" in disagreement.comments[-1]
    assert "Needs Human Decision" not in disagreement.comments[-1]

    escalation = FakeRunner(
        codex_outputs=[_discuss_answer_text(answer=None, position="needs-human", open_questions=["Who owns the decision?"], reviewer="OpenAI Codex")],
        gemini_outputs=[_discuss_answer_text(answer=None, position="needs-human", open_questions=["What is the target latency?"], reviewer="Google Gemini")],
    )
    assert run_discuss_loop(escalation, issue_number=56, config=config, discuss_max_rounds=0) == 0
    assert "Needs Human Decision" in escalation.comments[-1]
    assert "Who owns the decision?" in escalation.comments[-1]


def test_discuss_loop_semantic_equivalent_paraphrases_converge(tmp_path):
    runner = FakeRunner(
        codex_outputs=[_discuss_answer_text(answer="Add a bounded verification subphase.")],
        gemini_outputs=[_discuss_answer_text(answer="Introduce a capped, targeted verification phase.")],
        claude_outputs=[_semantic_comparison_text(
            classification="equivalent", shared_recommendation="Add a targeted, budget-capped verification subphase.",
        )],
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), discuss_result_mode="answer", discuss_analyzer="claude",
    )

    assert run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=0) == 0
    final = runner.comments[-1]
    assert "Converged Answer" in final
    assert "Consensus kind: `semantic-equivalent`" in final
    assert "targeted, budget-capped verification subphase" in final
    assert "Semantic comparison (advisory; not a debater vote)" in final
    assert any(
        "Analyze only these completed final-round debater responses" in " ".join(command)
        for command in _claude_commands(runner)
    )
    metadata = ROUND_RESUME_MARKER_RE.search(runner.issue_comments[-1]["body"])
    assert metadata is not None
    assert _decode_round_metadata(metadata.group("payload")).consensus_kind == "semantic-equivalent"


@pytest.mark.parametrize("refine", [False, True])
def test_discuss_loop_semantic_compatible_answers_require_debater_confirmation(tmp_path, refine):
    canonical = "Add a targeted, budget-capped verification subphase."
    runner = FakeRunner(
        codex_outputs=[
            _discuss_answer_text(answer="Add bounded verification."),
            _answer_confirmation_text(reviewer="Codex", decision="confirm"),
        ],
        gemini_outputs=[
            _discuss_answer_text(answer="Use targeted verification with a cap."),
            _answer_confirmation_text(
                reviewer="Gemini", decision="refine" if refine else "confirm",
                answer=canonical if refine else None,
            ),
        ],
        claude_outputs=[_semantic_comparison_text(
            classification="compatible_with_residual_decisions", shared_recommendation=canonical,
            remaining_decisions=["Choose the cap."],
        )],
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), discuss_result_mode="answer", discuss_analyzer="claude",
    )

    assert run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=0) == 0
    final = runner.comments[-1]
    assert "Consensus kind: `debater-confirmed`" in final
    assert "The recommendation above was explicitly confirmed by every debater." in final
    assert "Residual decisions:" in final
    assert "### Final analyzer observations" not in final
    claude_commands = _claude_commands(runner)
    assert len(claude_commands) == 1
    assert "discuss_semantic_comparison" in " ".join(claude_commands[0])
    metadata = ROUND_RESUME_MARKER_RE.search(runner.issue_comments[-1]["body"])
    assert metadata is not None
    parsed_metadata = _decode_round_metadata(metadata.group("payload"))
    assert parsed_metadata.consensus_kind == "debater-confirmed"
    assert parsed_metadata.final_analyzer_response is None


def test_discuss_loop_confirmation_failure_still_attempts_final_analyzer(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            _discuss_answer_text(answer="Add bounded verification."),
            _answer_confirmation_text(reviewer="Codex"),
        ],
        gemini_outputs=[
            _discuss_answer_text(answer="Use targeted verification with a cap."),
            "not structured",
        ],
        claude_outputs=[_semantic_comparison_text(
            classification="compatible_with_residual_decisions",
            shared_recommendation="Add a targeted, budget-capped verification subphase.",
            remaining_decisions=["Choose the cap."],
        )],
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), discuss_result_mode="answer", discuss_analyzer="claude",
    )

    assert run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=0) == 0
    assert "Consensus kind: `confirmation-failed`" in runner.comments[-1]
    assert any(
        "Analyze only these completed final-round debater responses" in " ".join(command)
        for command in _claude_commands(runner)
    )


@pytest.mark.parametrize("failed_comparison", [("comparator unavailable", 1), "not structured"])
def test_discuss_loop_semantic_material_conflict_and_failure_are_auditable_deadlocks(tmp_path, failed_comparison):
    def run_with(comparison):
        runner = FakeRunner(
            codex_outputs=[_discuss_answer_text(answer="Use an API.")],
            gemini_outputs=[_discuss_answer_text(answer="Use a library.")],
            claude_outputs=[comparison],
        )
        config = make_config(
            tmp_path, reviewer=("codex", "gemini"), discuss_result_mode="answer", discuss_analyzer="claude",
        )
        assert run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=0) == 0
        return runner

    conflict_runner = run_with(_semantic_comparison_text(
        classification="material_conflict", shared_recommendation="The alternatives materially differ.",
    ))
    conflict = conflict_runner.comments[-1]
    assert "Deadlock" in conflict
    assert "Consensus kind: `material-conflict`" in conflict
    assert "Classification: `material_conflict`" in conflict
    assert any(
        "Analyze only these completed final-round debater responses" in " ".join(command)
        for command in _claude_commands(conflict_runner)
    )
    metadata = ROUND_RESUME_MARKER_RE.search(conflict_runner.issue_comments[-1]["body"])
    assert metadata is not None
    assert _decode_round_metadata(metadata.group("payload")).consensus_kind == "material-conflict"

    failure_runner = run_with(failed_comparison)
    failure = failure_runner.comments[-1]
    assert "Deadlock" in failure
    assert "Consensus kind: `semantic-comparison-failed`" in failure
    assert "Classification: `failed`" in failure
    assert any(
        "Analyze only these completed final-round debater responses" in " ".join(command)
        for command in _claude_commands(failure_runner)
    )
    metadata = ROUND_RESUME_MARKER_RE.search(failure_runner.issue_comments[-1]["body"])
    assert metadata is not None
    assert _decode_round_metadata(metadata.group("payload")).consensus_kind == "semantic-comparison-failed"


def test_discuss_loop_triage_mode_does_not_invoke_semantic_comparator(tmp_path):
    vote = _discuss_review_text(outcome="implement")
    runner = FakeRunner(codex_outputs=[vote], gemini_outputs=[vote])
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_analyzer="claude")

    assert run_discuss_loop(runner, issue_number=56, config=config) == 0
    assert "Consensus: Implement" in runner.comments[-1]
    assert not any("discuss_semantic_comparison" in " ".join(command) for command, _ in runner.commands)
    assert any(
        "Analyze only these completed final-round debater responses" in " ".join(command)
        for command in _claude_commands(runner)
    )


def test_discuss_loop_answer_research_required_and_analyzer_prompt_are_mode_aware(tmp_path):
    sourced = {"status": "sourced", "sourced_facts": [{"fact": "Fact", "source": "https://example.test"}]}
    answer = _discuss_answer_text(research=sourced)
    runner = FakeRunner(codex_outputs=[answer], gemini_outputs=[answer])
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), discuss_result_mode="answer",
        discuss_research="required", discuss_analyzer="claude",
    )
    assert run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=0) == 0
    assert "Use an API boundary." in runner.comments[-1]
    assert all('"kind": "discuss_answer"' in " ".join(cmd) for cmd, _ in runner.commands if cmd[:1] in (["codex"], ["gemini"]))
    assert any(
        "Analyze only these completed final-round debater responses" in " ".join(command)
        for command in _claude_commands(runner)
    )


def test_discuss_loop_answer_partial_live_round_records_failure_and_deadlocks(tmp_path):
    answer = _discuss_answer_text(answer="Use an API boundary.")
    runner = FakeRunner(
        codex_outputs=[answer],
        gemini_outputs=[answer],
        claude_outputs=[("provider unavailable", 1)],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini", "claude"),
        discuss_result_mode="answer",
        discuss_on_debater_failure="partial",
        discuss_analyzer="antigravity",
    )

    assert run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=0) == 0
    final = runner.comments[-1]
    assert "Deadlock" in final
    assert "Claude" in final
    assert "Debater failures" in final
    assert "Consensus Answer" not in final
    assert any(
        "Analyze only these completed final-round debater responses" in " ".join(command)
        for command, _ in runner.commands
        if command[:1] == ["agy"]
    )


def test_discuss_loop_answer_mode_never_materializes_split_issues(tmp_path):
    answer = _discuss_answer_text(answer="Use an API boundary.")
    runner = FakeRunner(
        codex_outputs=[answer],
        gemini_outputs=[answer],
        issue_urls=["https://github.com/OWNER/REPO/issues/101"],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        discuss_result_mode="answer",
        materialize_split_issues=True,
    )

    assert run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=0) == 0
    assert runner.issues == []
    assert "Consensus Answer" in runner.comments[-1]
    assert "Consensus: Split" not in "\n".join(runner.comments)


def test_discuss_loop_rejects_resume_with_conflicting_result_mode(tmp_path):
    subject = _issue_subject()
    seeded = _seed_answer_debater_comment(
        reviewer="Codex", round_number=1, subject=subject,
        answer="Use an API boundary.",
    )
    runner = FakeRunner(issue_comments=[seeded])
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_result_mode="triage")

    with pytest.raises(
        AgentLoopError,
        match="Discuss transcript result mode conflicts with the requested mode",
    ):
        run_discuss_loop(runner, issue_number=56, config=config)


def test_discuss_loop_answer_debater_sees_analyzer_agenda_on_rebuttal_round(tmp_path):
    agenda = _discuss_agenda_text()
    answer1 = _discuss_answer_text(answer="Use an API.", reviewer="OpenAI Codex")
    answer2 = _discuss_answer_text(
        answer="Use an API.", rebuttal="The analyzer's ownership question is resolved by the API boundary.",
        reviewer="OpenAI Codex",
    )
    gemini1 = _discuss_answer_text(answer="Use a library.", reviewer="Google Gemini")
    gemini2 = _discuss_answer_text(
        answer="Use an API.", rebuttal="The ownership concern favors an API.", reviewer="Google Gemini",
    )
    runner = FakeRunner(
        codex_outputs=[answer1, answer2], gemini_outputs=[gemini1, gemini2],
        claude_outputs=[agenda], issue_payload=_grounded_agenda_issue_payload(),
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), discuss_result_mode="answer",
        discuss_analyzer="claude",
    )

    assert run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=1) == 0
    round2 = [
        " ".join(cmd) for cmd, _ in runner.commands
        if "Analyzer agenda for the prior round" in " ".join(cmd)
    ]
    assert len(round2) == 2, runner.commands
    assert all("Scope of the change" in prompt for prompt in round2)
    assert all("Would splitting resolve the scope objection?" in prompt for prompt in round2)
    round_one_summary = next(
        comment for comment in runner.comments
        if "## Round 1 summary: Answer Pending" in comment
    )
    assert "### Agenda for round 2 (analyzer: Claude)" in round_one_summary
    assert "**Scope of the change**" in round_one_summary


def test_discuss_loop_resumes_answer_partial_round_and_is_idempotent(tmp_path):
    subject = _issue_subject()
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_result_mode="answer")
    seeded = _seed_answer_debater_comment(
        reviewer="Codex", round_number=1, subject=subject,
        answer="Use an API.", config=config,
    )
    answer = _discuss_answer_text(answer="Use an API.", reviewer="Google Gemini")
    runner = FakeRunner(issue_comments=[seeded], gemini_outputs=[answer])

    assert run_discuss_loop(runner, issue_number=56, config=config) == 0
    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)
    assert "Consensus Answer" in runner.comments[-1]
    comment_count = len(runner.comments)

    assert run_discuss_loop(runner, issue_number=56, config=config) == 0
    assert len(runner.comments) == comment_count


def test_discuss_loop_resume_mid_round_answer_mode_accepts_valid_checkout_inspected_reference(tmp_path):
    subject = _issue_subject()
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_result_mode="answer")
    (config.codex_dir / "src.py").write_text("line one\nline two\n", encoding="utf-8")
    seeded = _seed_answer_debater_comment(
        reviewer="Codex", round_number=1, subject=subject,
        answer="Use an API.", evidence=_checkout_inspected_evidence("src.py:2"), config=config,
    )
    answer = _discuss_answer_text(answer="Use an API.", reviewer="Google Gemini")
    runner = FakeRunner(issue_comments=[seeded], gemini_outputs=[answer])

    assert run_discuss_loop(runner, issue_number=56, config=config) == 0
    assert "Consensus Answer" in runner.comments[-1]


def test_discuss_loop_resume_mid_round_answer_mode_rejects_invalid_checkout_inspected_reference(tmp_path):
    subject = _issue_subject()
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_result_mode="answer")
    seeded = _seed_answer_debater_comment(
        reviewer="Codex", round_number=1, subject=subject,
        answer="Use an API.", evidence=_checkout_inspected_evidence("src/missing.py:1"), config=config,
    )
    runner = FakeRunner(
        issue_comments=[seeded],
        gemini_outputs=[_discuss_answer_text(answer="Use an API.", reviewer="Google Gemini")],
    )

    with pytest.raises(AgentLoopError, match="not a file in the assigned checkout"):
        run_discuss_loop(runner, issue_number=56, config=config)


def test_discuss_loop_resumes_legacy_answer_metadata_conservatively(tmp_path):
    subject = _issue_subject()
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_result_mode="answer")
    seeded = _seed_answer_debater_comment(
        reviewer="Codex", round_number=1, subject=subject, answer="Use an API.",
        config=config, legacy=True,
    )
    runner = FakeRunner(
        issue_comments=[seeded],
        gemini_outputs=[_discuss_answer_text(answer="Use an API.", reviewer="Google Gemini")],
    )
    assert run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=0) == 0
    assert "Deadlock" in runner.comments[-1]
    assert "Verify availability." in runner.comments[-1]


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


def _validate_agenda_for_test(
    agenda: ParsedDiscussAgenda,
    *,
    issue_context: IssueContext | None = None,
    round_history: list[list[ParsedDiscussReview]] | None = None,
    prior_agenda: ParsedDiscussAgenda | None = None,
) -> None:
    _validate_discuss_analyzer_agenda_fidelity(
        agenda,
        issue_context=issue_context
        or IssueContext(
            number=1,
            repo="OWNER/REPO",
            title="Supported title",
            body="Supported body",
            url=None,
            comments=(),
        ),
        round_history=round_history or [],
        prior_agenda=prior_agenda,
        configured_reviewers=("codex", "gemini"),
        analyzer="claude",
    )


def _agenda_for_fidelity(
    *,
    consensus: tuple[str, ...] = (),
    topic: str = "Supported title",
    positions: tuple[tuple[str, str], ...] = (("Codex", "Supported body"),),
    question: str = "Supported title?",
    missing_facts: tuple[str, ...] = (),
    research_questions: tuple[str, ...] = (),
    research_question_targets: tuple[str, ...] = (),
) -> ParsedDiscussAgenda:
    return ParsedDiscussAgenda(
        consensus=consensus,
        disagreements=(
            DiscussAgendaDisagreement(
                topic=topic,
                positions=positions,
                question_for_next_round=question,
            ),
        ),
        missing_facts=missing_facts,
        research_required=bool(research_questions),
        research_questions=research_questions,
        research_question_targets=research_question_targets,
    )


def test_discuss_analyzer_fidelity_accepts_classified_supported_research_question():
    vote = ParsedDiscussReview(
        outcome="implement", rationale="Research the design.", split_proposals=(), reviewer="Codex",
        research_status="sourced", research_target="solution-design",
        research_questions=("What prior art supports this design?",),
    )
    agenda = _agenda_for_fidelity(
        research_questions=("What prior art supports this design?",),
        research_question_targets=("solution-design",),
    )
    _validate_agenda_for_test(agenda, round_history=[[vote]])


def test_discuss_analyzer_fidelity_rejects_unknown_position_key():
    agenda = _agenda_for_fidelity(positions=(("Developer", "Supported body"),))

    with pytest.raises(AgentLoopError, match="unknown debater"):
        _validate_agenda_for_test(agenda)


@pytest.mark.parametrize(
    ("field", "agenda"),
    [
        ("topic", _agenda_for_fidelity(topic="Levitation library strategy")),
        ("position", _agenda_for_fidelity(positions=(("Codex", "Use levitation libraries"),))),
        ("question", _agenda_for_fidelity(question="Should custom field manipulation win?")),
        ("missing_fact", _agenda_for_fidelity(missing_facts=("Levitation benchmark data",))),
        ("consensus", _agenda_for_fidelity(consensus=("Levitation is agreed",))),
        (
            "research_question",
            _agenda_for_fidelity(research_questions=("Which levitation library is fastest?",)),
        ),
    ],
)
def test_discuss_analyzer_fidelity_rejects_unsupported_agenda_text(field, agenda):
    with pytest.raises(AgentLoopError, match=field):
        _validate_agenda_for_test(agenda)


def test_discuss_analyzer_fidelity_accepts_issue_comment_and_body_support():
    ctx = IssueContext(
        number=1,
        repo="OWNER/REPO",
        title="API boundary",
        body="The scoped change is the implementation approach.",
        url=None,
        comments=(
            IssueComment(author="user", created_at="2026-01-01", body="Missing migration facts."),
        ),
    )
    agenda = _agenda_for_fidelity(
        consensus=("API boundary",),
        topic="scoped change",
        positions=(("Codex", "implementation approach"),),
        question="API boundary?",
        missing_facts=("Missing migration facts",),
    )

    _validate_agenda_for_test(agenda, issue_context=ctx)


def test_discuss_analyzer_fidelity_accepts_vote_research_and_prior_agenda_support():
    round_history = [
        [
            ParsedDiscussReview(
                outcome="split",
                rationale="Codex wants adapter cleanup.",
                split_proposals=("Extract parser stage",),
                reviewer="Codex",
                rebuttal="Keep adapter cleanup small.",
                analyzer_framing="misframed",
                framing_note="The API migration risk was overstated.",
                research_status="sourced",
                sourced_facts=(
                    DiscussSourcedFact(
                        fact="Gemini CLI remains available.",
                        source="https://example.com/gemini",
                    ),
                ),
            ),
            ParsedDiscussReview(
                outcome="implement",
                rationale="Gemini accepts parser stage.",
                split_proposals=(),
                reviewer="Gemini",
            ),
        ]
    ]
    prior_agenda = ParsedDiscussAgenda(
        consensus=("Prior analyzer agenda",),
        disagreements=(),
        missing_facts=("Prior missing fact",),
    )
    agenda = _agenda_for_fidelity(
        consensus=("Prior analyzer agenda",),
        topic="adapter cleanup",
        positions=(("Codex", "Extract parser stage"), ("Gemini", "accepts parser stage")),
        question="API migration risk?",
        missing_facts=("Prior missing fact",),
        research_questions=("Gemini CLI remains available?",),
    )

    _validate_agenda_for_test(agenda, round_history=round_history, prior_agenda=prior_agenda)


def test_discuss_analyzer_fidelity_accepts_empty_agenda_collections():
    agenda = ParsedDiscussAgenda(consensus=(), disagreements=(), missing_facts=())

    _validate_agenda_for_test(agenda)


def test_discuss_analyzer_fidelity_requires_exact_support_for_generic_only_text():
    agenda = _agenda_for_fidelity(topic="Scope of the change")

    with pytest.raises(AgentLoopError, match="topic"):
        _validate_agenda_for_test(agenda)

    ctx = IssueContext(
        number=1,
        repo="OWNER/REPO",
        title="My issue",
        body="Scope of the change. Supported body. Supported title.",
        url=None,
        comments=(),
    )
    _validate_agenda_for_test(agenda, issue_context=ctx)


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
    posted = runner.comments[-2]
    assert "Consensus: Split" in posted
    assert "Auth flow" in posted
    assert "Authorization checks" in posted
    # --materialize-split-issues defaults off, so the orchestrator posts an
    # explicit unfiled-split warning after the final summary (#476) instead of
    # silently leaving the proposals as unfiled text.
    warning = runner.comments[-1]
    assert "NOT filed as issues" in warning
    assert "Auth flow" in warning
    assert "Authorization checks" in warning


def test_discuss_loop_split_consensus_materializes_child_issues_when_enabled(tmp_path):
    split_text = _discuss_review_text(
        outcome="split",
        rationale="Too broad.",
        split_proposals=["Auth flow", "Authorization checks"],
    )
    runner = FakeRunner(
        codex_outputs=[split_text],
        gemini_outputs=[split_text],
        issue_urls=[
            "https://github.com/OWNER/REPO/issues/101",
            "https://github.com/OWNER/REPO/issues/102",
        ],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), materialize_split_issues=True)

    result = run_discuss_loop(runner, issue_number=56, config=config)

    assert result == 0
    assert len(runner.issues) == 2
    assert runner.issues[0]["title"] == "[#56 stage] Auth flow"
    assert runner.issues[1]["title"] == "[#56 stage] Authorization checks"
    assert "Too broad." in runner.issues[0]["body"]
    assert "Refs #56" in runner.issues[0]["body"]
    final_summary = runner.comments[-2]
    assert "Consensus: Split" in final_summary
    parent_summary = runner.comments[-1]
    assert "<!-- AGENT_DISCUSS_SPLIT:" in parent_summary
    assert "https://github.com/OWNER/REPO/issues/101" in parent_summary
    assert "https://github.com/OWNER/REPO/issues/102" in parent_summary


def test_discuss_loop_split_consensus_rerun_materializes_nothing_new(tmp_path):
    split_text = _discuss_review_text(
        outcome="split",
        rationale="Too broad.",
        split_proposals=["Auth flow"],
    )
    runner = FakeRunner(
        codex_outputs=[split_text],
        gemini_outputs=[split_text],
        issue_urls=["https://github.com/OWNER/REPO/issues/101"],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), materialize_split_issues=True)
    assert run_discuss_loop(runner, issue_number=56, config=config) == 0
    assert len(runner.issues) == 1

    # A second run reuses the same FakeRunner, which already accumulated the
    # posted comments (with their AGENT_LOOP_META metadata) from the first
    # run in `issue_comments`; it must find the existing consensus and
    # materialization markers and create nothing new.
    result = run_discuss_loop(runner, issue_number=56, config=config)

    assert result == 0
    assert len(runner.issues) == 1


def test_discuss_loop_marker_only_final_recovers_split_proposals_via_legacy_fallback(tmp_path):
    """An issue carries a bare `AGENT_DISCUSS_CONSENSUS` marker comment (from
    an old run) with no matching final-round summary metadata, so
    `resume_state.done` is False for this subject and `_run_discuss_loop`'s
    own `_resolve_final_split_proposals` must fall through to
    `_recover_final_discuss_split_proposals` -- the same shared,
    reviewer-workdir-validated recovery function used by the plan-first path
    -- to reconstruct and materialize the split proposal instead of crashing
    on the now-required `reviewer_workdirs` parameter (#541)."""
    subject = _issue_subject()
    config = make_config(tmp_path, reviewer=("codex", "gemini"), materialize_split_issues=True)
    (config.codex_dir / "src.py").write_text("line one\nline two\n", encoding="utf-8")
    seeded = [
        {
            "author": {"login": "bot"},
            "createdAt": "2026-01-01T00:00:00Z",
            "body": f"Old consensus record.\n<!-- AGENT_DISCUSS_CONSENSUS: {subject} -->",
        },
        _seed_debater_comment(
            reviewer="Codex", round_number=1, subject=subject, outcome="split",
            rationale="Too broad.", split_proposals=["Auth flow"],
            evidence=_checkout_inspected_evidence("src.py:2"), config=config,
        ),
    ]
    runner = FakeRunner(
        issue_comments=seeded, codex_outputs=[], gemini_outputs=[],
        issue_urls=["https://github.com/OWNER/REPO/issues/101"],
    )

    result = run_discuss_loop(runner, issue_number=56, config=config)

    assert result == 0
    assert len(runner.issues) == 1
    assert runner.issues[0]["title"] == "[#56 stage] Auth flow"


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


def test_discuss_loop_resume_mid_round_accepts_valid_checkout_inspected_reference(tmp_path):
    subject = _issue_subject()
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    (config.codex_dir / "src.py").write_text("line one\nline two\n", encoding="utf-8")
    seeded = _seed_debater_comment(
        reviewer="Codex",
        round_number=1,
        subject=subject,
        outcome="implement",
        rationale="Scoped.",
        evidence=_checkout_inspected_evidence("src.py:2"),
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
    assert "Consensus: Implement" in runner.comments[-1]


def test_discuss_loop_resume_mid_round_rejects_invalid_checkout_inspected_reference(tmp_path):
    subject = _issue_subject()
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    seeded = _seed_debater_comment(
        reviewer="Codex",
        round_number=1,
        subject=subject,
        outcome="implement",
        rationale="Scoped.",
        evidence=_checkout_inspected_evidence("src/missing.py:1"),
        config=config,
    )
    runner = FakeRunner(
        issue_comments=[seeded],
        codex_outputs=[],
        gemini_outputs=[_discuss_review_text(outcome="implement", rationale="Agreed.", reviewer="Gemini")],
    )

    with pytest.raises(AgentLoopError, match="not a file in the assigned checkout"):
        run_discuss_loop(runner, issue_number=56, config=config)


def test_discuss_loop_resume_mid_round_rejects_unconfigured_reviewer_debater_comment(tmp_path):
    # A debater comment posted by a reviewer who is no longer part of the
    # configured --reviewers set (e.g. dropped before resuming an interrupted
    # round) cannot be checkout-validated, since we have no known assigned
    # workdir for it; only the in-progress mid-round decode loop can ever
    # encounter this, since completed-round resume and legacy split recovery
    # both key strictly off configured reviewer names (#541).
    subject = _issue_subject()
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    seeded = [
        _seed_debater_comment(
            reviewer="Codex", round_number=1, subject=subject,
            outcome="implement", rationale="Scoped.", config=config,
        ),
        _seed_debater_comment(
            reviewer="ThirdParty", round_number=1, subject=subject,
            outcome="implement", rationale="Also scoped.", config=config,
        ),
    ]
    runner = FakeRunner(
        issue_comments=seeded,
        codex_outputs=[],
        gemini_outputs=[],
    )

    with pytest.raises(AgentLoopError, match="ThirdParty"):
        run_discuss_loop(runner, issue_number=56, config=config)


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


def test_discuss_loop_resume_completed_round_accepts_valid_checkout_inspected_reference(tmp_path):
    subject = _issue_subject()
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    (config.codex_dir / "src.py").write_text("line one\nline two\n", encoding="utf-8")
    codex_vote_r1 = ParsedDiscussReview(outcome="implement", rationale="Scoped.", split_proposals=(), reviewer="Codex")
    gemini_vote_r1 = ParsedDiscussReview(outcome="do-not-implement", rationale="Out of scope.", split_proposals=(), reviewer="Gemini")
    agenda = (
        "- Codex held `implement`: Scoped.",
        "- Gemini held `do-not-implement`: Out of scope.",
    )
    seeded = [
        _seed_debater_comment(
            reviewer="Codex", round_number=1, subject=subject,
            outcome="implement", rationale="Scoped.",
            evidence=_checkout_inspected_evidence("src.py:2"), config=config,
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
    assert "Consensus: Implement" in runner.comments[-1]


def test_discuss_loop_resume_completed_round_rejects_invalid_checkout_inspected_reference(tmp_path):
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
            outcome="implement", rationale="Scoped.",
            evidence=_checkout_inspected_evidence("src/missing.py:1"), config=config,
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
    runner = FakeRunner(
        issue_comments=seeded,
        codex_outputs=[],
        gemini_outputs=[],
    )

    with pytest.raises(AgentLoopError, match="not a file in the assigned checkout"):
        run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=1)


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


# --- analyzer-guided discuss mode tests (#467) ---


def _claude_commands(runner):
    return [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]


def test_discuss_loop_cli_parser_accepts_discuss_analyzer():
    from coding_review_agent_loop.cli import build_parser

    args = build_parser().parse_args(
        ["discuss", "56", "--repo", "OWNER/REPO", "--discuss-analyzer", "claude"]
    )
    assert args.discuss_analyzer == "claude"
    plain = build_parser().parse_args(["discuss", "56", "--repo", "OWNER/REPO"])
    assert plain.discuss_analyzer is None


def test_discuss_loop_unanimous_round1_attempts_final_only_analyzer(tmp_path):
    implement_text = _discuss_review_text(outcome="implement")
    runner = FakeRunner(
        codex_outputs=[implement_text],
        gemini_outputs=[implement_text],
        claude_outputs=[],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_analyzer="claude")

    result = run_discuss_loop(runner, issue_number=56, config=config)

    assert result == 0
    assert len(runner.comments) == 3
    assert "Consensus: Implement" in runner.comments[-1]
    assert len(_claude_commands(runner)) == 1
    assert "Analyze only these completed final-round debater responses" in " ".join(_claude_commands(runner)[0])
    assert "Analyzer" not in runner.comments[-1]


def test_discuss_loop_max_rounds_zero_runs_final_only_analyzer(tmp_path):
    runner = FakeRunner(
        codex_outputs=[_discuss_review_text(outcome="implement")],
        gemini_outputs=[_discuss_review_text(outcome="needs-human", rationale="Needs input.")],
        claude_outputs=[],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_analyzer="claude")

    result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=0)

    assert result == 0
    assert len(_claude_commands(runner)) == 1
    assert "Analyze only these completed final-round debater responses" in " ".join(_claude_commands(runner)[0])
    assert "Consensus kind: `deadlock` after round 1." in runner.comments[-1]


def test_discuss_loop_analyzer_agenda_focuses_debate_and_final_summary_distinguishes(tmp_path):
    agenda_text = _discuss_agenda_text(
        missing_facts=["Whether the API boundary is specified."],
    )
    runner = FakeRunner(
        codex_outputs=[
            _discuss_review_text(outcome="implement", rationale="Codex round-one rationale."),
            _discuss_review_text(
                outcome="implement",
                rationale="Still scoped.",
                rebuttal="Defending: the boundary is specified.",
                analyzer_framing="accurate",
            ),
        ],
        gemini_outputs=[
            _discuss_review_text(
                outcome="do-not-implement", rationale="Gemini round-one rationale."
            ),
            _discuss_review_text(
                outcome="do-not-implement",
                rationale="Still out of scope.",
                rebuttal="The scope objection stands.",
                analyzer_framing="accurate",
            ),
        ],
        claude_outputs=[agenda_text],
        issue_payload=_grounded_agenda_issue_payload(),
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_analyzer="claude")

    result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=1)

    assert result == 0
    # The analyzer ran exactly once, after the non-final round 1, with the
    # full round history in its prompt.
    analyzer_commands = [
        cmd for cmd in _claude_commands(runner) if "Summarize debate round 1" in " ".join(cmd)
    ]
    assert len(analyzer_commands) == 1
    analyzer_prompt = " ".join(analyzer_commands[0])
    assert "Codex round-one rationale." in analyzer_prompt
    assert "Gemini round-one rationale." in analyzer_prompt
    # Round-1 summary shows the attributed analyzer agenda instead of the
    # mechanical per-vote lines, including missing facts.
    round1_summary = runner.comments[2]
    assert "### Agenda for round 2 (analyzer: Claude)" in round1_summary
    assert "Scope of the change" in round1_summary
    assert "Missing facts:" in round1_summary
    assert "Whether the API boundary is specified." in round1_summary
    assert "- Codex held `implement`: Codex round-one rationale." not in round1_summary
    # Round-2 debate prompts are agenda-focused: agenda + own prior position
    # only; the other debater's rationale reaches them only via the analyzer's
    # summarized positions.
    debate_commands = [cmd for cmd, _cwd in runner.commands if "debate round 2" in " ".join(cmd)]
    assert len(debate_commands) == 2
    for command in debate_commands:
        prompt = " ".join(command)
        assert "Scope of the change" in prompt
        assert "Would splitting resolve the scope objection?" in prompt
        assert "Whether the API boundary is specified." in prompt
        assert "The analyzer is not authoritative" in prompt
        assert "Prior round reviewer positions:" not in prompt
    codex_prompt = next(
        p for p in (" ".join(c) for c in debate_commands) if "-- OpenAI Codex" in p
    )
    gemini_prompt = next(
        p for p in (" ".join(c) for c in debate_commands) if "-- Google Gemini" in p
    )
    assert "Codex round-one rationale." in codex_prompt
    assert "Gemini round-one rationale." not in codex_prompt
    assert "Gemini round-one rationale." in gemini_prompt
    assert "Codex round-one rationale." not in gemini_prompt
    # The prior agenda is historical only when the final-only analyzer fails.
    final = runner.comments[-1]
    assert "Consensus kind: `deadlock` after round 2." in final
    assert "### Final analyzer observations" not in final
    assert "### Agenda before final round" in final
    assert "Historical analyzer agenda only" in final
    assert "The issue is well-motivated." in final


def test_discuss_loop_final_analyzer_uses_only_final_responses_and_preserves_prior_agenda(tmp_path):
    prior = _discuss_agenda_text()
    final = _discuss_agenda_text(consensus=["Concession check injection is accepted."], disagreements=[])
    runner = FakeRunner(
        codex_outputs=[
            _discuss_review_text(outcome="implement", rationale="Prior scope dispute."),
            _discuss_review_text(outcome="implement", rationale="Concession check injection is accepted.", rebuttal="Resolved."),
        ],
        gemini_outputs=[
            _discuss_review_text(outcome="do-not-implement", rationale="Prior scope dispute."),
            _discuss_review_text(outcome="implement", rationale="Concession check injection is accepted.", rebuttal="Resolved."),
        ],
        claude_outputs=[prior, final], issue_payload=_grounded_agenda_issue_payload(),
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_analyzer="claude")
    assert run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=1) == 0
    final_prompt = next(" ".join(command) for command in _claude_commands(runner) if "Analyze only these completed final-round" in " ".join(command))
    assert "Concession check injection is accepted." in final_prompt
    assert "Scope of the change" not in final_prompt
    summary = runner.comments[-1]
    assert "### Final analyzer observations (analyzer: Claude; not debater-confirmed)" in summary
    assert "### Agenda before final round" in summary


def test_final_analyzer_fidelity_rejects_unknown_debater_and_unsupported_topic(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_analyzer="claude")
    votes = [
        ParsedDiscussReview(outcome="implement", rationale="Use concession checks.", split_proposals=(), reviewer="Codex"),
        ParsedDiscussReview(outcome="implement", rationale="Use concession checks.", split_proposals=(), reviewer="Gemini"),
    ]
    bad = ParsedDiscussAgenda(
        disagreements=(DiscussAgendaDisagreement("Invented synthesis", (("Claude", "Use synthesis."),), "Why?"),),
        consensus=(), missing_facts=(),
    )
    with pytest.raises(AgentLoopError, match="unknown or absent debater"):
        _validate_discuss_final_analyzer_fidelity(bad, final_votes=votes, configured_reviewers=("codex", "gemini"), analyzer="claude")
    unsupported = ParsedDiscussAgenda(consensus=("Invented synthesis injection.",), disagreements=(), missing_facts=())
    with pytest.raises(AgentLoopError, match="lacks final-round support"):
        _validate_discuss_final_analyzer_fidelity(unsupported, final_votes=votes, configured_reviewers=("codex", "gemini"), analyzer="claude")


def test_discuss_loop_debater_misframing_correction_is_rendered(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            _discuss_review_text(outcome="implement", rationale="Scoped."),
            _discuss_review_text(
                outcome="implement",
                rationale="Still scoped.",
                rebuttal="Defending scope.",
                analyzer_framing="accurate",
            ),
        ],
        gemini_outputs=[
            _discuss_review_text(outcome="do-not-implement", rationale="Out of scope."),
            _discuss_review_text(
                outcome="do-not-implement",
                rationale="Still out of scope.",
                rebuttal="The agenda mischaracterized me.",
                analyzer_framing="misframed",
                framing_note="I never proposed splitting; I questioned the motivation.",
            ),
        ],
        claude_outputs=[_discuss_agenda_text()],
        issue_payload=_grounded_agenda_issue_payload(),
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_analyzer="claude")

    result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=1)

    assert result == 0
    gemini_round2 = next(c for c in runner.comments if "Round 2: Gemini position" in c)
    assert "### Analyzer framing correction" in gemini_round2
    assert "I never proposed splitting; I questioned the motivation." in gemini_round2
    codex_round2 = next(c for c in runner.comments if "Round 2: Codex position" in c)
    assert "**Analyzer framing:** accurate" in codex_round2


def test_discuss_loop_analyzer_failure_falls_back_to_mechanical_agenda(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            _discuss_review_text(outcome="implement", rationale="Codex round-one rationale."),
            _discuss_review_text(
                outcome="implement", rationale="Still scoped.", rebuttal="Scope holds."
            ),
        ],
        gemini_outputs=[
            _discuss_review_text(
                outcome="do-not-implement", rationale="Gemini round-one rationale."
            ),
            _discuss_review_text(
                outcome="do-not-implement",
                rationale="Still out of scope.",
                rebuttal="Objection stands.",
            ),
        ],
        # The analyzer emits prose the strict parser rejects, and the repair
        # backend output is also unusable, so the analyzer fails outright.
        claude_outputs=["I could not produce a structured agenda, sorry."],
        antigravity_outputs=[("still not a structured agenda", 0)],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_analyzer="claude")

    result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=1)

    assert result == 0
    # The round-1 summary falls back to the mechanical agenda.
    round1_summary = runner.comments[2]
    assert "### Agenda for round 2" in round1_summary
    assert "(analyzer:" not in round1_summary
    assert "- Codex held `implement`: Codex round-one rationale." in round1_summary
    # Round-2 debate prompts fall back to the full prior-round transcript.
    debate_commands = [cmd for cmd, _cwd in runner.commands if "debate round 2" in " ".join(cmd)]
    assert len(debate_commands) == 2
    for command in debate_commands:
        prompt = " ".join(command)
        assert "Prior round reviewer positions:" in prompt
        assert "Codex round-one rationale." in prompt
        assert "Gemini round-one rationale." in prompt
    # The run still completes with a final summary and no analyzer section.
    final = runner.comments[-1]
    assert "Consensus kind: `deadlock` after round 2." in final
    assert "Analyzer-extracted consensus" not in final


def test_discuss_loop_rejects_repaired_hallucinated_analyzer_agenda(tmp_path):
    from unittest.mock import patch

    repaired = _discuss_agenda_text(
        disagreements=[
            {
                "topic": "Antigravity implementation strategy",
                "positions": {
                    "Developer": "Use existing levitation libraries.",
                    "Reviewer": "Implement custom field manipulation for performance.",
                },
                "question_for_next_round": (
                    "Does custom field manipulation provide significant enough performance "
                    "gains to justify the maintenance overhead?"
                ),
            }
        ],
        missing_facts=["Performance benchmarks for existing levitation libraries."],
    )
    runner = FakeRunner(
        codex_outputs=[
            _discuss_review_text(outcome="implement", rationale="Codex round-one rationale."),
            _discuss_review_text(
                outcome="implement", rationale="Still scoped.", rebuttal="Scope holds."
            ),
        ],
        gemini_outputs=[
            _discuss_review_text(
                outcome="split",
                rationale="Gemini round-one rationale.",
                split_proposals=["Extract setup work"],
            ),
            _discuss_review_text(
                outcome="split",
                rationale="Still split.",
                split_proposals=["Extract setup work"],
                rebuttal="Split still fits.",
            ),
        ],
        claude_outputs=[
            "I've drafted the round-1 agenda but the write to the required response file needs your permission."
        ],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_analyzer="claude")

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=repaired):
        result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=1)

    assert result == 0
    rendered = "\n".join(runner.comments)
    assert "Antigravity implementation strategy" not in rendered
    assert "levitation libraries" not in rendered
    assert "custom field manipulation" not in rendered
    round1_summary = runner.comments[2]
    assert "### Agenda for round 2" in round1_summary
    assert "(analyzer:" not in round1_summary
    assert "- Codex held `implement`: Codex round-one rationale." in round1_summary
    assert "- Gemini held `split`: Gemini round-one rationale." in round1_summary
    assert "### Analyzer-extracted consensus" not in round1_summary
    match = ROUND_RESUME_MARKER_RE.search(runner.issue_comments[2]["body"])
    assert match is not None
    metadata = _decode_round_metadata(match.group("payload"))
    assert metadata is not None
    assert metadata.analyzer_response is None
    debate_commands = [cmd for cmd, _cwd in runner.commands if "debate round 2" in " ".join(cmd)]
    assert len(debate_commands) == 2
    for command in debate_commands:
        prompt = " ".join(command)
        assert "Prior round reviewer positions:" in prompt
        assert "Codex round-one rationale." in prompt
        assert "Gemini round-one rationale." in prompt
        assert "Antigravity implementation strategy" not in prompt
        assert "levitation libraries" not in prompt
    final = runner.comments[-1]
    assert "Analyzer-extracted consensus" not in final


def test_discuss_loop_three_rounds_second_analyzer_prompt_includes_full_history(tmp_path):
    agenda_round1 = _discuss_agenda_text(consensus=["Round-one agenda marker."])
    agenda_round2 = _discuss_agenda_text(consensus=["Round-two agenda marker."])
    runner = FakeRunner(
        codex_outputs=[
            _discuss_review_text(outcome="implement", rationale="Codex round-one rationale."),
            _discuss_review_text(
                outcome="implement",
                rationale="Codex round-two rationale.",
                rebuttal="Codex round-two rebuttal.",
                analyzer_framing="accurate",
            ),
            _discuss_review_text(
                outcome="implement",
                rationale="Codex round-three rationale.",
                rebuttal="Holding position.",
                analyzer_framing="accurate",
            ),
        ],
        gemini_outputs=[
            _discuss_review_text(
                outcome="do-not-implement", rationale="Gemini round-one rationale."
            ),
            _discuss_review_text(
                outcome="do-not-implement",
                rationale="Gemini round-two rationale.",
                rebuttal="Gemini round-two rebuttal.",
                analyzer_framing="accurate",
            ),
            _discuss_review_text(
                outcome="implement",
                rationale="Convinced now.",
                rebuttal="Conceding the scope point.",
                analyzer_framing="accurate",
            ),
        ],
        claude_outputs=[agenda_round1, agenda_round2],
        issue_payload=_grounded_agenda_issue_payload(),
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_analyzer="claude")

    result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=2)

    assert result == 0
    round2_analyzer_commands = [
        cmd for cmd in _claude_commands(runner) if "Summarize debate round 2" in " ".join(cmd)
    ]
    assert len(round2_analyzer_commands) == 1
    prompt = " ".join(round2_analyzer_commands[0])
    # The analyzer sees every completed round, oldest first, plus its own
    # previous agenda.
    assert "Round 1 debater positions:" in prompt
    assert "Round 2 debater positions (latest round):" in prompt
    assert "Codex round-one rationale." in prompt
    assert "Gemini round-one rationale." in prompt
    assert "Codex round-two rebuttal." in prompt
    assert "Gemini round-two rebuttal." in prompt
    assert "Your previous agenda" in prompt
    assert "Round-one agenda marker." in prompt
    final = runner.comments[-1]
    assert "Consensus: Implement" in final
    assert "Consensus kind: `converged` after round 3." in final
    assert "Round-two agenda marker." in final


def test_discuss_loop_resume_restores_analyzer_agenda_from_summary_metadata(tmp_path):
    subject = _issue_subject()
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_analyzer="claude")
    codex_vote_r1 = ParsedDiscussReview(
        outcome="implement", rationale="Codex round-one rationale.", split_proposals=(), reviewer="Codex"
    )
    gemini_vote_r1 = ParsedDiscussReview(
        outcome="do-not-implement",
        rationale="Gemini round-one rationale.",
        split_proposals=(),
        reviewer="Gemini",
    )
    seeded = [
        _seed_debater_comment(
            reviewer="Codex", round_number=1, subject=subject,
            outcome="implement", rationale="Codex round-one rationale.", config=config,
        ),
        _seed_debater_comment(
            reviewer="Gemini", round_number=1, subject=subject,
            outcome="do-not-implement", rationale="Gemini round-one rationale.", config=config,
        ),
        _seed_summary_comment(
            round_number=1,
            reviewer_votes=[codex_vote_r1, gemini_vote_r1],
            is_final=False,
            subject=subject,
            agenda=(
                "- Codex held `implement`: Codex round-one rationale.",
                "- Gemini held `do-not-implement`: Gemini round-one rationale.",
            ),
            analyzer_response=_discuss_agenda_text(),
        ),
    ]
    implement_text = _discuss_review_text(
        outcome="implement", rationale="Agreed now.", rebuttal="Conceding.",
    )
    runner = FakeRunner(
        issue_comments=seeded,
        codex_outputs=[implement_text],
        gemini_outputs=[implement_text],
        claude_outputs=[],
    )

    result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=1)

    assert result == 0
    debate_commands = [cmd for cmd, _cwd in runner.commands if "debate round 2" in " ".join(cmd)]
    assert len(debate_commands) == 2
    for command in debate_commands:
        prompt = " ".join(command)
        # Resumed debate prompts are agenda-focused, not full-transcript.
        assert "Scope of the change" in prompt
        assert "Prior round reviewer positions:" not in prompt
    codex_prompt = next(
        p for p in (" ".join(c) for c in debate_commands) if "-- OpenAI Codex" in p
    )
    assert "Gemini round-one rationale." not in codex_prompt
    # The restored agenda also feeds the final summary's analyzer section.
    final = runner.comments[-1]
    assert "Consensus: Implement" in final
    assert "Analyzer-extracted consensus" in final


def test_discuss_loop_resume_legacy_summary_metadata_falls_back_to_plain_mode(tmp_path):
    subject = _issue_subject()
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_analyzer="claude")
    codex_vote_r1 = ParsedDiscussReview(
        outcome="implement", rationale="Codex round-one rationale.", split_proposals=(), reviewer="Codex"
    )
    gemini_vote_r1 = ParsedDiscussReview(
        outcome="do-not-implement",
        rationale="Gemini round-one rationale.",
        split_proposals=(),
        reviewer="Gemini",
    )
    seeded = [
        _seed_debater_comment(
            reviewer="Codex", round_number=1, subject=subject,
            outcome="implement", rationale="Codex round-one rationale.", config=config,
        ),
        _seed_debater_comment(
            reviewer="Gemini", round_number=1, subject=subject,
            outcome="do-not-implement", rationale="Gemini round-one rationale.", config=config,
        ),
        _seed_summary_comment(
            round_number=1,
            reviewer_votes=[codex_vote_r1, gemini_vote_r1],
            is_final=False,
            subject=subject,
            agenda=(
                "- Codex held `implement`: Codex round-one rationale.",
                "- Gemini held `do-not-implement`: Gemini round-one rationale.",
            ),
        ),
    ]
    implement_text = _discuss_review_text(
        outcome="implement", rationale="Agreed now.", rebuttal="Conceding.",
    )
    runner = FakeRunner(
        issue_comments=seeded,
        codex_outputs=[implement_text],
        gemini_outputs=[implement_text],
        claude_outputs=[],
    )

    result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=1)

    assert result == 0
    debate_commands = [cmd for cmd, _cwd in runner.commands if "debate round 2" in " ".join(cmd)]
    assert len(debate_commands) == 2
    for command in debate_commands:
        prompt = " ".join(command)
        assert "Prior round reviewer positions:" in prompt
        assert "Codex round-one rationale." in prompt
        assert "Gemini round-one rationale." in prompt
    assert "Analyzer-extracted consensus" not in runner.comments[-1]


def test_discuss_loop_agenda_claiming_consensus_is_forwarded_but_votes_rule(tmp_path):
    # The analyzer wrongly claims full consensus while the votes still differ:
    # the agenda is forwarded, but vote-only consensus detection rules.
    agenda_text = _discuss_agenda_text(
        consensus=["Everyone agrees to implement."], disagreements=[]
    )
    runner = FakeRunner(
        codex_outputs=[
            _discuss_review_text(outcome="implement", rationale="Scoped."),
            _discuss_review_text(
                outcome="implement", rationale="Still scoped.", rebuttal="Holding."
            ),
        ],
        gemini_outputs=[
            _discuss_review_text(outcome="do-not-implement", rationale="Out of scope."),
            _discuss_review_text(
                outcome="do-not-implement", rationale="Still out of scope.", rebuttal="Holding."
            ),
        ],
        claude_outputs=[agenda_text],
        issue_payload=_grounded_agenda_issue_payload(),
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_analyzer="claude")

    result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=1)

    assert result == 0
    round1_summary = runner.comments[2]
    assert "Everyone agrees to implement." in round1_summary
    final = runner.comments[-1]
    # Votes rule: the run still ends in a deadlock, and the divergence stays
    # visible next to the vote table.
    assert "Consensus kind: `deadlock` after round 2." in final
    assert "Everyone agrees to implement." in final
    assert "### Agenda before final round" in final
    assert "Historical analyzer agenda only" in final


# --- discuss research policy tests (#477) ---


_SOURCED_RESEARCH = {
    "status": "sourced",
    "sourced_facts": [
        {
            "fact": "Gemini CLI remains available for enterprise users.",
            "source": "https://example.com/gemini-cli-notice",
        }
    ],
}


def test_discuss_loop_cli_parser_accepts_discuss_research():
    from coding_review_agent_loop.cli import build_parser

    args = build_parser().parse_args(
        ["discuss", "56", "--repo", "OWNER/REPO", "--discuss-research", "auto"]
    )
    assert args.discuss_research == "auto"
    plain = build_parser().parse_args(["discuss", "56", "--repo", "OWNER/REPO"])
    assert plain.discuss_research == "none"
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["discuss", "56", "--repo", "OWNER/REPO", "--discuss-research", "always"]
        )


def test_discuss_loop_config_rejects_unknown_research_mode(tmp_path):
    with pytest.raises(AgentLoopError, match="--discuss-research must be one of"):
        make_config(tmp_path, reviewer=("codex", "gemini"), discuss_research="always")


def test_discuss_loop_none_mode_prompts_forbid_research(tmp_path):
    implement_text = _discuss_review_text(outcome="implement")
    runner = FakeRunner(
        codex_outputs=[implement_text],
        gemini_outputs=[implement_text],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))

    result = run_discuss_loop(runner, issue_number=56, config=config)

    assert result == 0
    review_commands = [
        cmd for cmd, _cwd in runner.commands if "vote on whether" in " ".join(cmd)
    ]
    assert len(review_commands) == 2
    for command in review_commands:
        prompt = " ".join(command)
        assert "Research policy: `none`" in prompt
        assert "sourced_facts" not in prompt
    final = runner.comments[-1]
    assert "Research policy: `none`." in final
    assert "Online research was disabled; all positions are agent judgment." in final


def test_discuss_loop_required_research_renders_sourced_facts(tmp_path):
    implement_text = _discuss_review_text(outcome="implement", research=_SOURCED_RESEARCH)
    runner = FakeRunner(
        codex_outputs=[implement_text],
        gemini_outputs=[implement_text],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_research="required")

    result = run_discuss_loop(runner, issue_number=56, config=config)

    assert result == 0
    review_commands = [
        cmd for cmd, _cwd in runner.commands if "vote on whether" in " ".join(cmd)
    ]
    assert len(review_commands) == 2
    for command in review_commands:
        prompt = " ".join(command)
        assert "Research policy: `required`" in prompt
        assert "must not be `not-needed`" in prompt
    debater_comment = runner.comments[0]
    assert "### Sourced facts" in debater_comment
    assert "https://example.com/gemini-cli-notice" in debater_comment
    final = runner.comments[-1]
    assert "### Research" in final
    assert "Research policy: `required`." in final
    assert "Sourced facts cited by debaters" in final
    assert "Gemini CLI remains available for enterprise users." in final


def test_discuss_loop_required_research_missing_research_repaired(tmp_path):
    # The debater omits the research object; the repair pass restores it, so
    # `required` mode is enforced by validation with repair as fallback.
    from unittest.mock import patch

    repaired = _discuss_review_text(outcome="implement", research=_SOURCED_RESEARCH)
    runner = FakeRunner(
        codex_outputs=[_discuss_review_text(outcome="implement")],
        gemini_outputs=[repaired],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_research="required")

    with patch(
        "coding_review_agent_loop.orchestrator.attempt_repair", return_value=repaired
    ):
        result = run_discuss_loop(runner, issue_number=56, config=config)

    assert result == 0
    assert "### Sourced facts" in runner.comments[0]


def test_discuss_loop_required_research_fails_when_repair_cannot_restore(tmp_path):
    # conftest neuters the repair pass, so the missing research object survives
    # repair and the debater invocation fails instead of being silently accepted.
    runner = FakeRunner(
        codex_outputs=[_discuss_review_text(outcome="implement")],
        gemini_outputs=[],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_research="required")

    with pytest.raises(AgentLoopError, match="research is required"):
        run_discuss_loop(runner, issue_number=56, config=config)


def test_discuss_loop_auto_mode_analyzer_brief_propagates_to_next_round(tmp_path):
    agenda_text = _discuss_agenda_text(
        research_required=True,
        research_questions=["Is Gemini CLI still available for enterprise users?"],
    )
    runner = FakeRunner(
        codex_outputs=[
            _discuss_review_text(
                outcome="implement", rationale="Scoped.", research={"status": "not-needed"}
            ),
            _discuss_review_text(
                outcome="implement",
                rationale="Still scoped.",
                rebuttal="Holding.",
                analyzer_framing="accurate",
                research=_SOURCED_RESEARCH,
            ),
        ],
        gemini_outputs=[
            _discuss_review_text(
                outcome="do-not-implement",
                rationale="Out of scope.",
                research={"status": "not-needed"},
            ),
            _discuss_review_text(
                outcome="do-not-implement",
                rationale="Still out of scope.",
                rebuttal="Holding.",
                analyzer_framing="accurate",
                research={"status": "unavailable"},
            ),
        ],
        claude_outputs=[agenda_text],
        issue_payload=_grounded_agenda_issue_payload(),
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        discuss_analyzer="claude",
        discuss_research="auto",
    )

    result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=1)

    assert result == 0
    # The analyzer was asked for a research brief.
    analyzer_prompt = " ".join(
        next(cmd for cmd in _claude_commands(runner) if "Summarize debate round 1" in " ".join(cmd))
    )
    assert "research_required" in analyzer_prompt
    # The brief propagates into round-2 debate prompts as shared context.
    debate_commands = [cmd for cmd, _cwd in runner.commands if "debate round 2" in " ".join(cmd)]
    assert len(debate_commands) == 2
    for command in debate_commands:
        prompt = " ".join(command)
        assert "Shared research brief" in prompt
        assert "Is Gemini CLI still available for enterprise users?" in prompt
    # The round-1 summary shows the research brief for auditability.
    round1_summary = runner.comments[2]
    assert "Research brief for the next round (answer with cited sources):" in round1_summary
    # The final deadlock summary distinguishes sourced facts from judgment and
    # is explicit about unavailable research.
    final = runner.comments[-1]
    assert "Research policy: `auto`." in final
    assert "Gemini CLI remains available for enterprise users." in final
    assert "Research was unavailable or inconclusive for Gemini" in final


def test_discuss_loop_auto_mode_analyzer_can_decide_no_research(tmp_path):
    agenda_text = _discuss_agenda_text(research_required=False, research_questions=[])
    runner = FakeRunner(
        codex_outputs=[
            _discuss_review_text(
                outcome="implement", rationale="Scoped.", research={"status": "not-needed"}
            ),
            _discuss_review_text(
                outcome="implement",
                rationale="Still scoped.",
                rebuttal="Holding.",
                analyzer_framing="accurate",
                research={"status": "not-needed"},
            ),
        ],
        gemini_outputs=[
            _discuss_review_text(
                outcome="do-not-implement",
                rationale="Out of scope.",
                research={"status": "not-needed"},
            ),
            _discuss_review_text(
                outcome="implement",
                rationale="Convinced.",
                rebuttal="Conceding.",
                analyzer_framing="accurate",
                research={"status": "not-needed"},
            ),
        ],
        claude_outputs=[agenda_text],
        issue_payload=_grounded_agenda_issue_payload(),
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        discuss_analyzer="claude",
        discuss_research="auto",
    )

    result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=1)

    assert result == 0
    debate_commands = [cmd for cmd, _cwd in runner.commands if "debate round 2" in " ".join(cmd)]
    assert len(debate_commands) == 2
    for command in debate_commands:
        assert "Shared research brief" not in " ".join(command)
    final = runner.comments[-1]
    assert "Consensus: Implement" in final
    assert "Research policy: `auto`." in final
    assert (
        "All debaters determined external research was unnecessary for this question."
        in final
    )


def test_discuss_loop_research_mode_recorded_in_round_metadata(tmp_path):
    implement_text = _discuss_review_text(outcome="implement", research=_SOURCED_RESEARCH)
    runner = FakeRunner(
        codex_outputs=[implement_text],
        gemini_outputs=[implement_text],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_research="required")

    result = run_discuss_loop(runner, issue_number=56, config=config)

    assert result == 0
    # runner.comments strips AGENT_LOOP_META; the raw posted bodies keep it.
    assert len(runner.issue_comments) == 3
    for comment in runner.issue_comments:
        match = ROUND_RESUME_MARKER_RE.search(comment["body"])
        assert match is not None
        metadata = _decode_round_metadata(match.group("payload"))
        assert metadata.research_mode == "required"


def test_discuss_loop_resume_is_lenient_about_prior_votes_without_research(tmp_path):
    # A transcript started without a research policy resumes cleanly under
    # `required`: enforcement applies only to newly invoked debaters.
    subject = _issue_subject()
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_research="required")
    codex_vote_r1 = ParsedDiscussReview(
        outcome="implement", rationale="Codex round-one rationale.", split_proposals=(), reviewer="Codex"
    )
    gemini_vote_r1 = ParsedDiscussReview(
        outcome="do-not-implement",
        rationale="Gemini round-one rationale.",
        split_proposals=(),
        reviewer="Gemini",
    )
    seeded = [
        _seed_debater_comment(
            reviewer="Codex", round_number=1, subject=subject,
            outcome="implement", rationale="Codex round-one rationale.", config=config,
        ),
        _seed_debater_comment(
            reviewer="Gemini", round_number=1, subject=subject,
            outcome="do-not-implement", rationale="Gemini round-one rationale.", config=config,
        ),
        _seed_summary_comment(
            round_number=1,
            reviewer_votes=[codex_vote_r1, gemini_vote_r1],
            is_final=False,
            subject=subject,
            agenda=(
                "- Codex held `implement`: Codex round-one rationale.",
                "- Gemini held `do-not-implement`: Gemini round-one rationale.",
            ),
        ),
    ]
    implement_text = _discuss_review_text(
        outcome="implement",
        rationale="Agreed now.",
        rebuttal="Conceding.",
        research=_SOURCED_RESEARCH,
    )
    runner = FakeRunner(
        issue_comments=seeded,
        codex_outputs=[implement_text],
        gemini_outputs=[implement_text],
    )

    result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=1)

    assert result == 0
    final = runner.comments[-1]
    assert "Consensus: Implement" in final
    assert "Research policy: `required`." in final


# --- parallel debater execution tests (#475) ---

import os
import sys
import threading
import time
from pathlib import Path

import coding_review_agent_loop.orchestrator as orchestrator_module
from coding_review_agent_loop.runner import CommandResult, Runner

from agent_loop_helpers import read_usage_summary


def test_discuss_loop_cli_parser_accepts_parallel_flags():
    from coding_review_agent_loop.cli import build_parser

    args = build_parser().parse_args(
        [
            "discuss", "56", "--repo", "OWNER/REPO",
            "--discuss-parallel",
            "--discuss-debater-timeout", "900",
            "--discuss-on-debater-failure", "partial",
        ]
    )
    assert args.discuss_parallel is True
    assert args.discuss_debater_timeout == 900.0
    assert args.discuss_on_debater_failure == "partial"
    plain = build_parser().parse_args(["discuss", "56", "--repo", "OWNER/REPO"])
    assert plain.discuss_parallel is False
    assert plain.discuss_debater_timeout is None
    assert plain.discuss_on_debater_failure == "fail"
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["discuss", "56", "--repo", "OWNER/REPO", "--discuss-on-debater-failure", "retry"]
        )


def test_discuss_loop_config_rejects_invalid_failure_policy_and_timeout(tmp_path):
    with pytest.raises(AgentLoopError, match="--discuss-on-debater-failure must be one of"):
        make_config(tmp_path, reviewer=("codex", "gemini"), discuss_on_debater_failure="retry")
    with pytest.raises(AgentLoopError, match="--discuss-debater-timeout must be greater than zero"):
        make_config(tmp_path, reviewer=("codex", "gemini"), discuss_debater_timeout=0)


class _ConcurrencyProbeRunner(FakeRunner):
    """Each debater blocks until the other has started: deadlocks (then fails
    fast via the wait timeout) unless same-round debaters truly overlap."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.codex_started = threading.Event()
        self.gemini_started = threading.Event()
        self.overlap_confirmed = True

    def run_with_log(self, args, *, cwd, **kwargs):
        cmd = [str(arg) for arg in args]
        if cmd[:2] == ["codex", "exec"]:
            self.codex_started.set()
            if not self.gemini_started.wait(timeout=10):
                self.overlap_confirmed = False
        elif cmd[:1] == ["gemini"]:
            self.gemini_started.set()
            if not self.codex_started.wait(timeout=10):
                self.overlap_confirmed = False
        return super().run_with_log(args, cwd=cwd, **kwargs)


def test_discuss_parallel_runs_same_round_debaters_concurrently(tmp_path):
    implement_text = _discuss_review_text(outcome="implement")
    runner = _ConcurrencyProbeRunner(
        codex_outputs=[implement_text],
        gemini_outputs=[implement_text],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_parallel=True)

    result = run_discuss_loop(runner, issue_number=56, config=config)

    assert result == 0
    assert runner.overlap_confirmed, "same-round debaters did not run concurrently"
    assert len(runner.comments) == 3
    assert "Round 1: Codex position" in runner.comments[0]
    assert "Round 1: Gemini position" in runner.comments[1]
    assert "Consensus: Implement" in runner.comments[2]


def test_discuss_parallel_matches_sequential_output(tmp_path):
    """Parity: the parallel happy path posts the same comments as sequential."""
    outputs = {
        "codex": [
            _discuss_review_text(outcome="implement", rationale="Scoped."),
            _discuss_review_text(
                outcome="implement", rationale="Still scoped.", rebuttal="Holding.",
            ),
        ],
        "gemini": [
            _discuss_review_text(outcome="do-not-implement", rationale="Out of scope."),
            _discuss_review_text(
                outcome="implement", rationale="Convinced.", rebuttal="Conceding.",
            ),
        ],
    }
    sequential_runner = FakeRunner(
        codex_outputs=list(outputs["codex"]), gemini_outputs=list(outputs["gemini"])
    )
    sequential_config = make_config(
        tmp_path / "seq", reviewer=("codex", "gemini"), log_dir=tmp_path / "seq" / "logs"
    )
    parallel_runner = FakeRunner(
        codex_outputs=list(outputs["codex"]), gemini_outputs=list(outputs["gemini"])
    )
    parallel_config = make_config(
        tmp_path / "par",
        reviewer=("codex", "gemini"),
        log_dir=tmp_path / "par" / "logs",
        discuss_parallel=True,
    )

    assert run_discuss_loop(sequential_runner, issue_number=56, config=sequential_config, discuss_max_rounds=1) == 0
    assert run_discuss_loop(parallel_runner, issue_number=56, config=parallel_config, discuss_max_rounds=1) == 0

    assert parallel_runner.comments == sequential_runner.comments
    assert "Consensus: Implement" in parallel_runner.comments[-1]


class _EventOrderRunner(FakeRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.events = []
        self._events_lock = threading.Lock()

    def _event_name(self, cmd):
        if cmd[:2] == ["codex", "exec"]:
            return "codex"
        if cmd[:1] == ["gemini"]:
            return "gemini"
        if cmd[:1] == ["claude"]:
            return "analyzer" if "Summarize debate round" in " ".join(cmd) else "claude"
        return None

    def run_with_log(self, args, *, cwd, **kwargs):
        cmd = [str(arg) for arg in args]
        name = self._event_name(cmd)
        if name is not None:
            with self._events_lock:
                self.events.append(f"{name}-start")
        result = super().run_with_log(args, cwd=cwd, **kwargs)
        if name is not None:
            with self._events_lock:
                self.events.append(f"{name}-end")
        return result


def test_discuss_parallel_analyzer_waits_for_debater_synchronization_point(tmp_path):
    runner = _EventOrderRunner(
        codex_outputs=[
            _discuss_review_text(outcome="implement", rationale="Scoped."),
            _discuss_review_text(
                outcome="implement", rationale="Still scoped.", rebuttal="Holding.",
                analyzer_framing="accurate",
            ),
        ],
        gemini_outputs=[
            _discuss_review_text(outcome="do-not-implement", rationale="Out of scope."),
            _discuss_review_text(
                outcome="do-not-implement", rationale="Still out.", rebuttal="Holding.",
                analyzer_framing="accurate",
            ),
        ],
        claude_outputs=[_discuss_agenda_text()],
        issue_payload=_grounded_agenda_issue_payload(),
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), discuss_analyzer="claude", discuss_parallel=True
    )

    result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=1)

    assert result == 0
    analyzer_start = runner.events.index("analyzer-start")
    assert runner.events.index("codex-end") < analyzer_start
    assert runner.events.index("gemini-end") < analyzer_start
    # Debater comments post after the sync point, in configured order, before
    # the round summary.
    assert "Round 1: Codex position" in runner.comments[0]
    assert "Round 1: Gemini position" in runner.comments[1]
    assert "Round 1 summary" in runner.comments[2]


def test_discuss_parallel_fail_policy_raises_after_round_settles_and_posts_survivors(tmp_path):
    runner = FakeRunner(
        codex_outputs=[_discuss_review_text(outcome="implement")],
        gemini_outputs=[("gemini exploded", 1)],
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), discuss_parallel=True, agent_max_retries=0
    )

    with pytest.raises(AgentLoopError, match="Gemini"):
        run_discuss_loop(runner, issue_number=56, config=config)

    # The surviving debater's comment is posted before the abort so a rerun
    # resumes it, and no round summary is posted.
    assert len(runner.comments) == 1
    assert "Round 1: Codex position" in runner.comments[0]


def test_discuss_parallel_partial_policy_continues_round_and_records_failure(tmp_path):
    implement_text = _discuss_review_text(outcome="implement")
    runner = FakeRunner(
        codex_outputs=[implement_text],
        claude_outputs=[implement_text],
        gemini_outputs=[("gemini exploded", 1)],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini", "claude"),
        discuss_parallel=True,
        discuss_on_debater_failure="partial",
        agent_max_retries=0,
    )

    result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=0)

    assert result == 0
    # Codex and Claude posted; Gemini did not.
    assert len(runner.comments) == 3
    final = runner.comments[-1]
    # A partial round never declares consensus even though all successful
    # debaters voted `implement`.
    assert "Consensus: Needs Human Review (Deadlock)" in final
    assert "Consensus: Implement" not in final
    assert "### Debater failures" in final
    assert "Gemini: no vote this round (deterministic)" in final
    assert "cannot declare final consensus" in final
    # The placeholder is visible in round history and recorded in metadata.
    assert "Codex: `implement`, Gemini: `failed`, Claude: `implement`" in final
    summary_raw = runner.issue_comments[-1]["body"]
    metadata = _decode_round_metadata(ROUND_RESUME_MARKER_RE.search(summary_raw).group("payload"))
    assert metadata.failed_debaters == (("Gemini", "deterministic"),)


def test_discuss_sequential_partial_policy_applies_without_parallel_flag(tmp_path):
    implement_text = _discuss_review_text(outcome="implement")
    runner = FakeRunner(
        codex_outputs=[("codex exploded", 1)],
        gemini_outputs=[implement_text],
        claude_outputs=[implement_text],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini", "claude"),
        discuss_on_debater_failure="partial",
        agent_max_retries=0,
    )

    result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=0)

    assert result == 0
    final = runner.comments[-1]
    assert "### Debater failures" in final
    assert "Codex: no vote this round (deterministic)" in final


def test_discuss_partial_policy_requires_two_successful_votes(tmp_path):
    runner = FakeRunner(
        codex_outputs=[_discuss_review_text(outcome="implement")],
        gemini_outputs=[("gemini exploded", 1)],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        discuss_parallel=True,
        discuss_on_debater_failure="partial",
        agent_max_retries=0,
    )

    with pytest.raises(AgentLoopError, match="Gemini"):
        run_discuss_loop(runner, issue_number=56, config=config)

    # The lone surviving vote is still posted for resume; no summary follows.
    assert len(runner.comments) == 1
    assert "Round 1: Codex position" in runner.comments[0]


def test_discuss_partial_round_resumes_from_failed_debater_metadata(tmp_path):
    subject = _issue_subject()
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini", "claude"),
        discuss_on_debater_failure="partial",
    )
    codex_vote_r1 = ParsedDiscussReview(outcome="implement", rationale="Scoped.", split_proposals=(), reviewer="Codex")
    claude_vote_r1 = ParsedDiscussReview(outcome="do-not-implement", rationale="Out of scope.", split_proposals=(), reviewer="Claude")
    seeded = [
        _seed_debater_comment(reviewer="Codex", round_number=1, subject=subject, outcome="implement", rationale="Scoped.", config=config),
        _seed_debater_comment(reviewer="Claude", round_number=1, subject=subject, outcome="do-not-implement", rationale="Out of scope.", config=config),
        _seed_summary_comment(
            round_number=1,
            reviewer_votes=[codex_vote_r1, claude_vote_r1],
            is_final=False,
            subject=subject,
            agenda=("- Codex held `implement`: Scoped.", "- Claude held `do-not-implement`: Out of scope."),
            failed_debaters=(("Gemini", "timeout"),),
        ),
    ]
    implement_text = _discuss_review_text(
        outcome="implement", rationale="Agreed.", rebuttal="Conceding.",
    )
    runner = FakeRunner(
        issue_comments=seeded,
        codex_outputs=[implement_text],
        gemini_outputs=[implement_text],
        claude_outputs=[implement_text],
    )

    result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=1)

    assert result == 0
    # Round 2 re-invokes all three debaters (the failed one gets a fresh turn)
    # without re-running round 1 or raising the completeness error.
    assert len(runner.comments) == 4
    final = runner.comments[-1]
    assert "Consensus: Implement" in final
    assert "Round 1: Codex: `implement`, Gemini: `failed`, Claude: `do-not-implement`" in final
    round2_commands = [cmd for cmd, _cwd in runner.commands if "debate round 2" in " ".join(cmd)]
    assert len(round2_commands) == 3
    # Round-2 debate prompts surface the placeholder position.
    for command in round2_commands:
        assert "Gemini: `failed`" in " ".join(command)


def test_discuss_resume_still_raises_when_missing_debater_has_no_failure_metadata(tmp_path):
    subject = _issue_subject()
    config = make_config(tmp_path, reviewer=("codex", "gemini", "claude"))
    codex_vote_r1 = ParsedDiscussReview(outcome="implement", rationale="Scoped.", split_proposals=(), reviewer="Codex")
    claude_vote_r1 = ParsedDiscussReview(outcome="do-not-implement", rationale="Out of scope.", split_proposals=(), reviewer="Claude")
    seeded = [
        _seed_debater_comment(reviewer="Codex", round_number=1, subject=subject, outcome="implement", rationale="Scoped.", config=config),
        _seed_debater_comment(reviewer="Claude", round_number=1, subject=subject, outcome="do-not-implement", rationale="Out of scope.", config=config),
        _seed_summary_comment(
            round_number=1,
            reviewer_votes=[codex_vote_r1, claude_vote_r1],
            is_final=False,
            subject=subject,
            agenda=("- Codex held `implement`: Scoped.",),
        ),
    ]
    runner = FakeRunner(issue_comments=seeded, codex_outputs=[], gemini_outputs=[], claude_outputs=[])

    with pytest.raises(AgentLoopError, match="metadata is inconsistent"):
        run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=1)


def test_discuss_parallel_zero_pending_resume_skips_executor(tmp_path, monkeypatch):
    subject = _issue_subject()
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_parallel=True)
    seeded = [
        _seed_debater_comment(reviewer="Codex", round_number=1, subject=subject, outcome="implement", rationale="Scoped.", config=config),
        _seed_debater_comment(reviewer="Gemini", round_number=1, subject=subject, outcome="implement", rationale="Agreed.", config=config),
    ]
    runner = FakeRunner(issue_comments=seeded, codex_outputs=[], gemini_outputs=[])

    def _fail_executor(*args, **kwargs):
        raise AssertionError("ThreadPoolExecutor must not be constructed on a zero-pending resume")

    monkeypatch.setattr(orchestrator_module, "ThreadPoolExecutor", _fail_executor)

    result = run_discuss_loop(runner, issue_number=56, config=config)

    assert result == 0
    assert len(runner.comments) == 1
    assert "Consensus: Implement" in runner.comments[0]


class _TimeoutSimulatingRunner(FakeRunner):
    """Simulates Runner.run_with_log's timeout contract (returncode=None) for gemini."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gemini_invocations = 0

    def run_with_log(self, args, *, cwd, **kwargs):
        cmd = [str(arg) for arg in args]
        if cmd[:1] == ["gemini"]:
            self.gemini_invocations += 1
            assert kwargs.get("timeout_seconds") == 45.0
            return CommandResult(cmd, Path(cwd), "", "", None)
        return super().run_with_log(args, cwd=cwd, **kwargs)


def test_discuss_debater_timeout_is_partial_result_without_transient_retries(tmp_path):
    implement_text = _discuss_review_text(outcome="implement")
    runner = _TimeoutSimulatingRunner(
        codex_outputs=[implement_text],
        claude_outputs=[implement_text],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini", "claude"),
        discuss_parallel=True,
        discuss_on_debater_failure="partial",
        discuss_debater_timeout=45.0,
        agent_max_retries=2,
    )

    result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=0)

    assert result == 0
    # A timeout is not transient: exactly one invocation despite retries being
    # configured.
    assert runner.gemini_invocations == 1
    final = runner.comments[-1]
    assert "### Debater failures" in final
    assert "Gemini: no vote this round (timeout)" in final
    summary_raw = runner.issue_comments[-1]["body"]
    metadata = _decode_round_metadata(ROUND_RESUME_MARKER_RE.search(summary_raw).group("payload"))
    assert metadata.failed_debaters == (("Gemini", "timeout"),)


def test_discuss_debater_checkout_inspected_claim_valid_reference_passes_through(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    (config.codex_dir / "src.py").write_text("line one\nline two\n", encoding="utf-8")
    valid = _discuss_review_text(
        outcome="implement", evidence=_checkout_inspected_evidence("src.py:2")
    )
    runner = FakeRunner(
        codex_outputs=[valid],
        gemini_outputs=[_discuss_review_text(outcome="implement")],
    )

    result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=0)

    assert result == 0
    assert "Consensus kind: `unanimous` after round 1." in runner.comments[-1]


def test_discuss_debater_checkout_inspected_claim_invalid_reference_triggers_repair(tmp_path):
    # A debater cites a checkout-inspected path:line that does not exist in its
    # own assigned checkout; the evidence check rejects the vote the same way
    # any other malformed structured response is rejected, so the existing
    # repair path re-checks the repaired text against the same validator (#541).
    from unittest.mock import patch

    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    (config.codex_dir / "src.py").write_text("line one\nline two\n", encoding="utf-8")
    invalid = _discuss_review_text(
        outcome="implement", evidence=_checkout_inspected_evidence("src/missing.py:1")
    )
    repaired = _discuss_review_text(
        outcome="implement", evidence=_checkout_inspected_evidence("src.py:2")
    )
    runner = FakeRunner(
        codex_outputs=[invalid],
        gemini_outputs=[_discuss_review_text(outcome="implement")],
    )

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=repaired):
        result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=0)

    assert result == 0
    assert "Consensus kind: `unanimous` after round 1." in runner.comments[-1]


def test_discuss_debater_checkout_inspected_claim_implausibly_large_line_triggers_repair(tmp_path):
    # A debater cites a checkout-inspected line number far beyond Python
    # 3.11+'s default int-conversion digit limit; this must be rejected as
    # AgentLoopError (triggering the normal repair path) rather than crashing
    # the debater turn with an uncaught ValueError (#541 review).
    from unittest.mock import patch

    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    (config.codex_dir / "src.py").write_text("line one\nline two\n", encoding="utf-8")
    huge_line = "9" * 4500
    invalid = _discuss_review_text(
        outcome="implement", evidence=_checkout_inspected_evidence(f"src.py:{huge_line}")
    )
    repaired = _discuss_review_text(
        outcome="implement", evidence=_checkout_inspected_evidence("src.py:2")
    )
    runner = FakeRunner(
        codex_outputs=[invalid],
        gemini_outputs=[_discuss_review_text(outcome="implement")],
    )

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=repaired):
        result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=0)

    assert result == 0
    assert "Consensus kind: `unanimous` after round 1." in runner.comments[-1]


def test_discuss_debater_checkout_inspected_claim_unreadable_file_triggers_repair(tmp_path):
    # The referenced file exists (passes is_file()) but cannot be opened; an
    # OSError from open() must be translated into AgentLoopError (triggering
    # the normal repair path) rather than crashing the debater turn (#541
    # review).
    from unittest.mock import patch

    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    unreadable = config.codex_dir / "secret.py"
    unreadable.write_text("line one\nline two\n", encoding="utf-8")
    unreadable.chmod(0o000)
    (config.codex_dir / "src.py").write_text("line one\nline two\n", encoding="utf-8")
    invalid = _discuss_review_text(
        outcome="implement", evidence=_checkout_inspected_evidence("secret.py:1")
    )
    repaired = _discuss_review_text(
        outcome="implement", evidence=_checkout_inspected_evidence("src.py:2")
    )
    runner = FakeRunner(
        codex_outputs=[invalid],
        gemini_outputs=[_discuss_review_text(outcome="implement")],
    )

    try:
        with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=repaired):
            result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=0)
    finally:
        unreadable.chmod(0o644)

    assert result == 0
    assert "Consensus kind: `unanimous` after round 1." in runner.comments[-1]


def test_discuss_parallel_rejects_shared_debater_workdirs_even_with_allow_shared_dir(tmp_path):
    shared = tmp_path / "shared"
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        codex_dir=shared,
        gemini_dir=shared,
        allow_shared_dir=True,
        discuss_parallel=True,
    )
    runner = FakeRunner(codex_outputs=[], gemini_outputs=[])

    with pytest.raises(AgentLoopError, match="distinct workdir per debater"):
        run_discuss_loop(runner, issue_number=56, config=config)

    assert len(runner.comments) == 0


def test_discuss_parallel_artifacts_are_isolated_by_round_and_agent(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            _discuss_review_text(outcome="implement", rationale="Scoped."),
            _discuss_review_text(outcome="implement", rationale="Still.", rebuttal="Holding.", analyzer_framing="accurate"),
        ],
        gemini_outputs=[
            _discuss_review_text(outcome="do-not-implement", rationale="No."),
            _discuss_review_text(outcome="implement", rationale="Now yes.", rebuttal="Conceding.", analyzer_framing="accurate"),
        ],
        claude_outputs=[_discuss_agenda_text()],
        issue_payload=_grounded_agenda_issue_payload(),
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), discuss_analyzer="claude", discuss_parallel=True
    )

    result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=1)

    assert result == 0
    log_names = sorted(p.name for p in config.log_dir.glob("*.log"))
    assert len([n for n in log_names if n.endswith("-codex-discuss-r1.log")]) == 1
    assert len([n for n in log_names if n.endswith("-gemini-discuss-r1.log")]) == 1
    assert len([n for n in log_names if n.endswith("-codex-discuss-r2.log")]) == 1
    assert len([n for n in log_names if n.endswith("-gemini-discuss-r2.log")]) == 1
    assert len([n for n in log_names if n.endswith("-claude-discuss-analyzer-r1.log")]) == 1
    assert len(log_names) == len(set(log_names))
    # Usage records from parallel calls are all present with unique call ids.
    summary = read_usage_summary(config.log_dir)
    call_ids = [record["call_id"] for record in summary["calls"]]
    assert len(call_ids) == len(set(call_ids))
    assert len(call_ids) == 5


# --- non-pty Runner timeout tests (#475) ---


def test_runner_non_pty_timeout_returns_none_and_keeps_log(tmp_path):
    log_path = tmp_path / "logs" / "timeout.log"
    started = time.monotonic()
    result = Runner().run_with_log(
        [sys.executable, "-c", "import sys,time; print('partial output', flush=True); time.sleep(30)"],
        cwd=tmp_path,
        log_path=log_path,
        label="timeout-test",
        progress_interval_seconds=300,
        check=False,
        timeout_seconds=0.3,
    )

    assert result.returncode is None
    assert time.monotonic() - started < 10
    assert "partial output" in result.stdout
    assert "partial output" in log_path.read_text(encoding="utf-8")


def test_runner_non_pty_timeout_kills_whole_process_group(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    log_path = tmp_path / "logs" / "group-timeout.log"
    result = Runner().run_with_log(
        ["bash", "-c", f"sleep 30 & echo $! > {pid_file}; wait"],
        cwd=tmp_path,
        log_path=log_path,
        label="group-timeout-test",
        progress_interval_seconds=300,
        check=False,
        timeout_seconds=0.3,
    )

    assert result.returncode is None
    grandchild_pid = int(pid_file.read_text(encoding="utf-8").strip())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild_pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    os.kill(grandchild_pid, 9)
    pytest.fail("grandchild process survived the process-group kill")


def test_discuss_parallel_non_final_partial_round_continues_to_next_round(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            _discuss_review_text(outcome="implement", rationale="Scoped."),
            _discuss_review_text(outcome="implement", rationale="Still scoped.", rebuttal="Holding."),
        ],
        claude_outputs=[
            _discuss_review_text(outcome="implement", rationale="Agreed."),
            _discuss_review_text(outcome="implement", rationale="Still agreed.", rebuttal="Holding."),
        ],
        gemini_outputs=[
            ("gemini exploded", 1),
            _discuss_review_text(outcome="implement", rationale="Back online.", rebuttal="Joining."),
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini", "claude"),
        discuss_parallel=True,
        discuss_on_debater_failure="partial",
        agent_max_retries=0,
    )

    result = run_discuss_loop(runner, issue_number=56, config=config, discuss_max_rounds=1)

    assert result == 0
    # Round 1: codex + claude comments, then a non-final partial summary.
    round1_summary = runner.comments[2]
    assert "Round 1 summary: Consensus Pending" in round1_summary
    assert "### Debater failures" in round1_summary
    assert "Gemini: no vote this round (deterministic)" in round1_summary
    # Round 2: the failed debater gets a fresh turn; all three converge.
    final = runner.comments[-1]
    assert "Consensus: Implement" in final
    assert "Consensus kind: `converged` after round 2." in final
    assert "Round 1: Codex: `implement`, Gemini: `failed`, Claude: `implement`" in final
    assert "Round 2: Codex: `implement`, Gemini: `implement`, Claude: `implement`" in final
