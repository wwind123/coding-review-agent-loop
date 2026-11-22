"""Plan-verification tests: plan approval classification (#1543)."""

from types import SimpleNamespace

import pytest
from issue_loop_test_helpers import _M1088_ACTOR, _m1090_legacy_history
from agent_loop_helpers import make_config


def test_1090_poster_incompleteness_proof_is_opt_in(tmp_path):
    from coding_review_agent_loop.plan_verification import classify_plan_approval

    comments = [
        SimpleNamespace(body=c["body"], author=c["author"]["login"], author_id=c["_rest_author_id"])
        for c in _m1090_legacy_history(
            ("Codex", "OpenAI Codex", "approved"),
            ("Gemini", "Google Gemini", "blocking"),
        )
    ]

    def classify(**kwargs):
        return classify_plan_approval(
            comments, config=make_config(tmp_path), issue_number=56,
            trusted_actor=_M1088_ACTOR, execution_mode_guard=False, **kwargs,
        )

    default = classify()
    assert default.state == "undetermined"
    assert "unanchored plain plan history" in default.reason
    proven = classify(prove_poster_incompleteness=True)
    assert proven.state == "unfinished"
    assert proven.reason == "the latest legacy plan round is provably incomplete"


# --- #2069: plan-approval classification keeps human-decision stops ---------


def _wrong_bound_override_stop():
    from coding_review_agent_loop.decomposition import (
        collect_child_disposition_overrides,
        format_child_disposition_override_comment,
    )
    from coding_review_agent_loop.github import IssueComment

    body = format_child_disposition_override_comment(
        parent_issue=99, plan_hash="plan-hash", stage_id="stage-one",
        disposition="requires-child-planning", rationale="Wrong binding.",
    )
    collect_child_disposition_overrides(
        parent_comments=(IssueComment(author="human", created_at="2026-01-01T00:00:00Z", body=body),),
        parent_issue=5, plan_hash="plan-hash", topology_stage_ids=("stage-one",),
        routed_stage_id="stage-one",
    )


def _wrong_bound_supersession_comments():
    from coding_review_agent_loop.decomposition import format_child_plan_supersession_comment

    return [SimpleNamespace(body=format_child_plan_supersession_comment(
        child_issue=57, parent_issue=5, stage_id="stage-one",
        superseded_plan_hash="f" * 16, rationale="Wrong child.",
    ))]


@pytest.mark.parametrize("branch", ["fresh-child-inputs", "supersession-check", "legacy-history-amendment"])
def test_plan_approval_paths_keep_human_decision_stops(tmp_path, monkeypatch, branch):
    """items 7-9: every plan-approval fallback re-raises a signed-record stop."""
    from coding_review_agent_loop import cli, execution_policy, github
    from coding_review_agent_loop.board_amendment import format_reviewer_board_amendment_comment
    from coding_review_agent_loop.decomposition import collect_child_plan_supersessions
    from coding_review_agent_loop.errors import HumanDecisionRequiredError, parse_human_decision
    from coding_review_agent_loop.plan_verification import (
        classify_plan_approval,
        require_recoverable_plan_approval,
    )

    if branch == "fresh-child-inputs":
        # Real input derivation reaches the real wrong-bound override stop.
        monkeypatch.setattr(execution_policy, "_infer_staged_parent_issue", lambda _ctx: 5)
        monkeypatch.setattr(github, "get_issue_context", lambda *_a, **_k: SimpleNamespace(
            number=5, human_requirements=(), comments=()))

        def provenance(**_kwargs):
            _wrong_bound_override_stop()

        monkeypatch.setattr(execution_policy, "_resolve_fresh_child_provenance", provenance)

        def run():
            require_recoverable_plan_approval(
                object(), config=make_config(tmp_path),
                issue_context=SimpleNamespace(number=56, human_requirements=(), comments=()),
                comments=[], trusted_actor=None, error_message="plan not approved",
            )
        expected = ("signed child-disposition override", "Found while deriving the plan approval inputs.")
    elif branch == "supersession-check":
        import test_plan_first_loop_review_scheduling as scheduling

        runner, config, _checkpoint = scheduling._approved_closed_history(tmp_path)
        comments = [SimpleNamespace(body=c["body"], author_id=c.get("_rest_author_id"))
                    for c in runner.issue_comments]

        def check(plan_hash):
            return any(record.superseded_plan_hash == plan_hash for record in collect_child_plan_supersessions(
                _wrong_bound_supersession_comments(), child_issue=56, parent_issue=5,
                stage_id="stage-one", ignored_sink=[]))

        def run():
            classify_plan_approval(comments, config=config, issue_number=56, supersession_check=check)
        expected = ("signed child-plan supersession", "Found while reading the signed child-plan supersession state.")
    else:
        comments = [
            SimpleNamespace(body=c["body"], author=c["author"]["login"], author_id=c["_rest_author_id"])
            for c in _m1090_legacy_history(
                ("Codex", "OpenAI Codex", "approved"), ("Gemini", "Google Gemini", "approved"),
            )
        ]
        # A historical scheduler contract with no board root or change digest:
        # not lineage-bearing, yet its signed amendments are still collected.
        from coding_review_agent_loop.discuss_loop import _attach_round_metadata
        from coding_review_agent_loop.plan_review_scheduling import make_plan_contract
        from coding_review_agent_loop.round_metadata import PostedRoundMetadata

        comments.insert(1, SimpleNamespace(body=_attach_round_metadata("Scheduler", PostedRoundMetadata(
            flow="plan", role="summary", agent="Orchestrator", round_number=1, subject="plan",
            phase="scheduler-prelaunch",
            scheduler_contract=make_plan_contract(("Codex", "Gemini"), "primary-then-panel", "Codex").as_dict(),
            scheduler_obligation_digest="0" * 16, scheduler_selected_reviewers=("Codex",),
            scheduler_paused_reviewers=(("Gemini", "primary phase"),),
            scheduler_reasons=("primary phase",), scheduler_final_sweep=False,
            scheduler_force_full=False, scheduler_calls_avoided=1,
            scheduler_phase="primary", scheduler_primary_reviewer="Codex",
            plan_candidate_key={
                "subject": "plan", "aggregate_plan_identity": "aggregate",
                "execution_strategy_identity": "strategy", "risk_test_matrix_identity": "matrix",
                "surfaced_requirement_id_digest": "requirements",
                "execution_strategy_contract_version": 1,
            },
        )), author="coding-review-agent-loop", author_id=_M1088_ACTOR[1]))
        comments.append(SimpleNamespace(body=format_reviewer_board_amendment_comment(
            flow="pr", issue=None, pr_number=17, original_required_reviewers=("Codex", "Gemini"),
            policy="all-reviewers", primary_reviewer=None, removed_reviewers=("Gemini",),
            effective_from_round=2, rationale="quota",
        ), author="human", author_id=7))

        def run():
            classify_plan_approval(
                comments, config=make_config(tmp_path), issue_number=56,
                trusted_actor=_M1088_ACTOR, execution_mode_guard=False,
            )
        expected = ("signed reviewer-board amendment", "Found while reading the plan board history.")

    with pytest.raises(HumanDecisionRequiredError) as caught:
        run()
    parts = parse_human_decision(str(caught.value))
    assert parts is not None and parts.safe_to_automate is False
    assert expected[0] in parts.what and parts.what.endswith(expected[1])

    def raise_stop(*_args, **_kwargs):
        raise caught.value

    monkeypatch.setattr(cli, "run_issue_loop", raise_stop)
    assert cli.main(["issue", "56", "--repo", "OWNER/REPO"]) == 4
