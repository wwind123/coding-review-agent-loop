"""Checkpoint-only round recovery across issue and fresh PR invocations (#1774)."""

import copy
import dataclasses
import json
import re
from types import SimpleNamespace

import pytest

from agent_loop_helpers import FakeRunner, make_config, structured_issue_implementation, structured_pr_review
from coding_review_agent_loop.cli import run_issue_loop, run_pr_loop
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.review_spool import ReviewRoundSpool, review_spool_root
from coding_review_agent_loop.round_state import _extract_round_metadata_records, _resume_pr_round
from pr_loop_test_helpers import _agent_sequence
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
from agent_loop_helpers import patch_attr_in


class IndependentRunner(FakeRunner):
    def _run_with_log_locked(self, args, **kwargs):
        if args[0] == "agy" and "models" not in args:
            # The provider's turn can read the live PR, so no same-round
            # secondary findings may have been published when it starts.
            assert not any(
                r.metadata.role == "reviewer" and r.metadata.agent == "Codex"
                and r.metadata.round_number == 2 for r in _records(self)
            )
        return super()._run_with_log_locked(args, **kwargs)


def _config(tmp_path, **overrides):
    values = dict(
        reviewer=("claude", "codex", "antigravity"), primary_reviewer="claude",
        pr_review_policy="primary-then-panel", review_parallel=True,
        pre_review_tests=False, agent_max_retries=0,
    )
    values.update(overrides)
    return make_config(tmp_path, **values)


def _records(runner):
    return _extract_round_metadata_records(
        [SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]], flow="pr",
    )


def _interrupted_issue(tmp_path):
    runner = IndependentRunner(
        claude_outputs=[structured_issue_implementation(pr_number=77), structured_pr_review(reviewer="Anthropic Claude")],
        codex_outputs=[structured_pr_review(reviewer="OpenAI Codex")],
        # No Antigravity output: a deterministic invocation failure, not an
        # unavailable outcome that settles the seat and may be published.
        pr_payload={"body": "Fixes #56\nFixes #57"},
    )
    config = _config(tmp_path, expected_closing_issue_ids=(56, 57))
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=config)
    records = _records(runner)
    checkpoint = records[-1].metadata
    assert checkpoint.phase == "scheduler-prelaunch"
    assert checkpoint.round_number == 2
    assert checkpoint.scheduler_phase == "secondary-audit"
    assert not any(r.metadata.agent == "Codex" for r in records)
    spool = ReviewRoundSpool(
        root=review_spool_root(config.agent_memory_dir), repo=config.repo,
        surface="pr", number=77, round_number=2, subject=checkpoint.subject,
    )
    assert spool.load("Codex") is not None
    return runner, config, spool


def _fresh(runner, **outputs):
    return IndependentRunner(
        pr_payload=copy.deepcopy(runner.pr_payload),
        issue_payload=copy.deepcopy(runner.issue_payload),
        issue_comments=copy.deepcopy(runner.issue_comments),
        git_head=runner.git_head, **outputs,
    )


@pytest.mark.parametrize("parallel", [False, True])
def test_issue_secondary_audit_replays_after_contract_refusal(tmp_path, parallel):
    runner, config, spool = _interrupted_issue(tmp_path)
    before = spool._path("Codex").read_bytes()
    fresh = _fresh(runner)
    with pytest.raises(AgentLoopError, match="expected|closing|contract"):
        run_pr_loop(fresh, pr_number=77, config=dataclasses.replace(config, expected_closing_issue_ids=(56,)))
    assert _agent_sequence(fresh) == []
    assert spool._path("Codex").read_bytes() == before
    assert fresh.pr_payload["comments"] == runner.pr_payload["comments"]
    resumed = _resume_pr_round(
        [SimpleNamespace(body=c["body"]) for c in fresh.pr_payload["comments"]],
        head_sha=spool.subject, configured_reviewers=config.reviewer,
    )
    assert resumed.round_number == 2
    assert resumed.completed_reviews == () and not resumed.reconciled
    fresh.antigravity_outputs.append(structured_pr_review(reviewer="Antigravity"))
    assert run_pr_loop(fresh, pr_number=77, config=dataclasses.replace(config, review_parallel=parallel)) == 0
    assert _agent_sequence(fresh) == ["agy"]
    reviews = [r.metadata for r in _records(fresh) if r.metadata.role == "reviewer"]
    assert [(r.agent, r.round_number) for r in reviews].count(("Codex", 2)) == 1
    assert [(r.agent, r.round_number) for r in reviews].count(("Antigravity", 2)) == 1
    assert [(r.agent, r.round_number) for r in reviews].count(("Claude", 1)) == 1
    assert not spool.directory.exists()


@pytest.mark.parametrize("parallel", [False, True])
def test_repeated_failure_then_partial_publication_resumes(tmp_path, monkeypatch, parallel):
    runner, config, spool = _interrupted_issue(tmp_path)
    before = spool._path("Codex").read_bytes()
    retry = _fresh(runner)
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_pr_loop(retry, pr_number=77, config=config)
    assert _agent_sequence(retry) == ["agy"]
    assert spool._path("Codex").read_bytes() == before
    assert not any(r.metadata.agent == "Codex" for r in _records(retry))

    retry = _fresh(retry, antigravity_outputs=[structured_pr_review(reviewer="Antigravity")])
    # Real-time authenticated publication dates let the frozen-carrier reader
    # reconcile the prefix, just as it would across separate CLI invocations.
    retry.authenticated_actor = ("coding-review-agent-loop", 424242)
    original_post = pr_loop.post_pr_comment

    def interrupt_after_first_review(*args, **kwargs):
        result = original_post(*args, **kwargs)
        if kwargs.get("publication") is not None:
            raise KeyboardInterrupt
        return result

    with monkeypatch.context() as patch:
        patch_attr_in(patch, (pr_loop, pr_loop_support), "post_pr_comment", interrupt_after_first_review)
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(retry, pr_number=77, config=config)
    assert _agent_sequence(retry) == ["agy"]
    assert spool.directory.exists()
    assert sum(r.metadata.round_number == 2 and r.metadata.role == "reviewer" for r in _records(retry)) == 1
    final = _fresh(retry)
    final.authenticated_actor = retry.authenticated_actor
    assert run_pr_loop(final, pr_number=77, config=dataclasses.replace(config, review_parallel=parallel)) == 0
    assert _agent_sequence(final) == []
    reviews = [r.metadata.agent for r in _records(final) if r.metadata.role == "reviewer"]
    assert sorted(reviews) == ["Antigravity", "Claude", "Codex"]
    assert not spool.directory.exists()


@pytest.mark.parametrize("changed_board", [False, True])
def test_first_prelaunch_interruption_keeps_startup_board_reentry(tmp_path, monkeypatch, changed_board):
    config = _config(tmp_path)
    runner = FakeRunner()
    original_post = pr_loop.post_pr_comment

    def interrupt(*args, **kwargs):
        result = original_post(*args, **kwargs)
        if "PR review scheduling audit:" in kwargs["body"]:
            raise KeyboardInterrupt
        return result

    with monkeypatch.context() as patch:
        patch_attr_in(patch, (pr_loop, pr_loop_support), "post_pr_comment", interrupt)
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=config)
    assert _agent_sequence(runner) == []
    fresh = _fresh(
        runner, claude_outputs=[structured_pr_review(reviewer="Anthropic Claude")],
        codex_outputs=[structured_pr_review(reviewer="OpenAI Codex")],
        antigravity_outputs=[structured_pr_review(reviewer="Antigravity")],
        gemini_outputs=[structured_pr_review(reviewer="Google Gemini")],
    )
    if changed_board:
        config = dataclasses.replace(config, reviewer=("claude", "codex", "gemini"))
    assert run_pr_loop(fresh, pr_number=77, config=config) == 0
    primary = [r.metadata for r in _records(fresh) if r.metadata.agent == "Claude" and r.metadata.role == "reviewer"]
    assert len(primary) == 1
    assert primary[0].round_number == (2 if changed_board else 1)
    assert ("agy" in _agent_sequence(fresh)) != changed_board


@pytest.mark.parametrize("later_round", [False, True])
def test_pending_amendment_after_prelaunch_interrupt(tmp_path, monkeypatch, later_round):
    from pr_loop_test_helpers import _m943_amendment_from_error, _m943_append

    if later_round:
        runner, config, _spool = _interrupted_issue(tmp_path)
    else:
        config = _config(tmp_path)
        runner = FakeRunner()
        original_post = pr_loop.post_pr_comment

        def interrupt(*args, **kwargs):
            result = original_post(*args, **kwargs)
            if "PR review scheduling audit:" in kwargs["body"]:
                raise KeyboardInterrupt
            return result

        with monkeypatch.context() as patch:
            patch_attr_in(patch, (pr_loop, pr_loop_support), "post_pr_comment", interrupt)
            with pytest.raises(KeyboardInterrupt):
                run_pr_loop(runner, pr_number=77, config=config)
    reduced = dataclasses.replace(config, reviewer=("claude", "codex"), min_reviewers=3)
    with pytest.raises(AgentLoopError, match="below the floor") as error:
        run_pr_loop(runner, pr_number=77, config=reduced)
    amendment = _m943_amendment_from_error(str(error.value))
    effective_round = json.loads(re.search(r"```json\s*(.*?)```", amendment, re.S).group(1))["effective_from_round"]
    _m943_append(runner, amendment)
    fresh = _fresh(
        runner, claude_outputs=[structured_pr_review(reviewer="Anthropic Claude")],
        codex_outputs=[structured_pr_review(reviewer="OpenAI Codex")],
    )
    assert run_pr_loop(fresh, pr_number=77, config=reduced) == 0
    assert "agy" not in _agent_sequence(fresh)
    activated = [r.metadata for r in _records(fresh) if r.metadata.reviewer_board_amendment_digest]
    assert activated
    assert all(tuple(r.scheduler_contract["required_reviewers"]) == ("Claude", "Codex") for r in activated if r.scheduler_contract)
    assert activated[0].phase == "scheduler-prelaunch"
    assert activated[0].round_number == effective_round


@pytest.mark.parametrize("damage", ["missing", "invalid", "malformed", "foreign-round", "foreign-head"])
def test_partial_publication_never_launches_against_unsafe_spool(tmp_path, monkeypatch, damage):
    runner, config, spool = _interrupted_issue(tmp_path)
    retry = _fresh(runner, antigravity_outputs=[structured_pr_review(reviewer="Antigravity")])
    retry.authenticated_actor = ("coding-review-agent-loop", 424242)
    original_post = pr_loop.post_pr_comment

    def interrupt(*args, **kwargs):
        result = original_post(*args, **kwargs)
        if kwargs.get("publication") is not None:
            raise KeyboardInterrupt
        return result

    with monkeypatch.context() as patch:
        patch_attr_in(patch, (pr_loop, pr_loop_support), "post_pr_comment", interrupt)
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(retry, pr_number=77, config=config)
    published = {r.metadata.agent for r in _records(retry) if r.metadata.round_number == 2 and r.metadata.role == "reviewer"}
    unpublished = ({"Codex", "Antigravity"} - published).pop()
    fields = spool.load(unpublished)
    path = spool._path(unpublished)
    foreign = None
    if damage.startswith("foreign"):
        foreign = dataclasses.replace(spool, **({"round_number": 3} if damage == "foreign-round" else {"subject": "another-head"}))
        foreign.store(unpublished, fields)
        foreign_before = foreign._path(unpublished).read_bytes()
        path.unlink()
    elif damage == "missing":
        path.unlink()
    elif damage == "invalid":
        spool.store(unpublished, {**fields, "text": "no longer a review"})
    else:
        path.write_text("{broken", encoding="utf-8")
    retained = path.read_bytes() if path.exists() else None
    fresh = _fresh(retry)
    fresh.authenticated_actor = retry.authenticated_actor
    with pytest.raises(AgentLoopError, match="partially published|malformed|validat"):
        run_pr_loop(fresh, pr_number=77, config=config)
    assert _agent_sequence(fresh) == []
    assert [r for r in _records(fresh) if r.metadata.role == "reviewer"] == [
        r for r in _records(retry) if r.metadata.role == "reviewer"
    ]
    if retained is not None:
        assert path.read_bytes() == retained
    if foreign is not None:
        assert foreign._path(unpublished).read_bytes() == foreign_before
