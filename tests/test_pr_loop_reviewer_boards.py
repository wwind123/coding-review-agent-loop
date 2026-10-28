"""PR-loop tests for board_amendment and reviewer_seats: PR reviewer boards (#1542)."""

import coding_review_agent_loop.pr_plan_recovery as pr_plan_recovery

import dataclasses
import re
from types import SimpleNamespace

import pytest
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop, run_pr_loop
from coding_review_agent_loop.managed_ci import (
    ManagedCiContract,
    ManagedCiOutcome,
)
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    _attach_round_metadata,
)
from coding_review_agent_loop.prompts import (
    COMPACT_PR_REVIEW_VOLATILE_TAIL_MARKER,
)
from coding_review_agent_loop.review_scheduling import TransitionClassification
from agent_loop_helpers import (
    FakeRunner,
    structured_plan_state,
    make_config,
    structured_coder_followup,
    structured_issue_implementation,
    structured_plan_review,
    structured_pr_review,
)
import coding_review_agent_loop.architecture_contract as architecture_contract
import coding_review_agent_loop.child_plan_binding as child_plan_binding
import coding_review_agent_loop.execution_policy as execution_policy
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.panel_evidence as panel_evidence
import coding_review_agent_loop.plan_first_loop as plan_first_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
from agent_loop_helpers import patch_attr_in
from pr_loop_test_helpers import (
    _agent_sequence,
    _m943_amendment_from_error,
    _m943_append,
    _m943_partial_pr_round,
    _posted_scheduler_metadata,
    _rewrite_pr_metadata,
    _staged_config,
    _staged_review,
    _unavailable,
)


# --- Signed reviewer-board amendment (#943) -------------------------------

from coding_review_agent_loop.board_amendment import (  # noqa: E402
    format_reviewer_board_amendment_comment as _m943_amendment_comment,
)


@pytest.mark.parametrize("linked_issue", [False, True], ids=["standalone", "issue-mode"])
def test_pr_board_amendment_resumes_and_qualifies_standalone_pr(tmp_path, monkeypatch, linked_issue):
    """Row pr-qualification, standalone (issue_context=None) and with a linked issue."""
    payload = (
        {
            "issue_payload": {"number": 56, "title": "Linked issue", "body": "Scope."},
            "pr_payload": {"body": "Fixes #56"},
        }
        if linked_issue
        else {}
    )
    runner = _m943_partial_pr_round(tmp_path, **payload)
    reduced = _staged_config(tmp_path, reviewer=("codex", "gemini"), min_reviewers=3, auto_merge=True)
    calls_before = _agent_sequence(runner)

    # No record: the drift error prints a filled PR amendment template.
    with pytest.raises(AgentLoopError, match="2 reviewer\\(s\\) is below the floor of 3") as excinfo:
        run_pr_loop(runner, pr_number=77, config=reduced)
    assert _agent_sequence(runner) == calls_before
    template = _m943_amendment_from_error(str(excinfo.value))
    assert '"flow": "pr"' in template and '"pr_number": 77' in template
    assert '"issue": null' in template

    _m943_append(runner, template)
    monkeypatch.setattr(
        pr_loop_support, "merge_pr", lambda *args, **kwargs: None
    )
    assert run_pr_loop(runner, pr_number=77, config=reduced) == 0

    # The qualification gate re-reads the fresh PR comments, including the
    # pre-amendment three-reviewer records, and accepts the amended board.
    amended_contract = pr_loop.make_contract(
        ("Codex", "Gemini"), "primary-then-panel", None, "Codex"
    )
    pr_loop_support._fresh_pr_qualification_snapshot(
        runner,
        config=reduced,
        pr_number=77,
        issue_context=None,
        parent_issue_context=None,
        scheduler_contract=amended_contract,
    )
    # A board other than the amended one is a board change the gate never
    # accepts: the run would not have reviewed with it (#1376).
    with pytest.raises(AgentLoopError, match="No qualification or merge is permitted"):
        pr_loop_support._fresh_pr_qualification_snapshot(
            runner,
            config=_staged_config(tmp_path, reviewer=("codex", "antigravity"), auto_merge=True),
            pr_number=77,
            issue_context=None,
            parent_issue_context=None,
            scheduler_contract=amended_contract,
        )
    assert "agy" not in _agent_sequence(runner)[len(calls_before):]
    assert any("Reviewer board amendment applied." in comment for comment in runner.comments)
    posted = _posted_scheduler_metadata(runner)
    # Gemini's round-2 review was reused and Codex's approval carried, yet the
    # activation round still persists a fresh digest-bound scheduler decision.
    amended = [item for item in posted if item.reviewer_board_amendment_digest is not None]
    assert amended
    assert amended[0].phase == "scheduler-prelaunch"
    assert amended[0].round_number == 2
    for item in posted:
        board = tuple(item.scheduler_contract["required_reviewers"])
        if item.reviewer_board_amendment_digest is not None:
            assert board == ("Codex", "Gemini")
        else:
            assert board == ("Codex", "Gemini", "Antigravity")
    completion_notes = [
        comment for comment in runner.comments
        if comment.startswith("Review completed on a reduced reviewer board.")
    ]
    assert len(completion_notes) == 1
    assert "required board now Codex, Gemini" in completion_notes[0]


def test_pr_board_amendment_qualification_refuses_a_stale_contract(tmp_path, monkeypatch):
    """Rows pr-qualification and digest-binding: the gate re-reads fresh PR comments."""
    runner = _m943_partial_pr_round(tmp_path)
    reduced = _staged_config(tmp_path, reviewer=("codex", "gemini"), min_reviewers=3, auto_merge=True)
    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=reduced)
    _m943_append(runner, _m943_amendment_from_error(str(excinfo.value)))
    stale_contract = pr_loop.make_contract(
        ("Codex", "Gemini", "Antigravity"), "primary-then-panel", None, "Codex"
    )
    stale = _attach_round_metadata(
        "A stale-board scheduler record posted while managed CI was running.",
        PostedRoundMetadata(
            flow="pr", role="summary", agent="Orchestrator", round_number=9, subject="abc123",
            phase="reconciliation", scheduler_contract=stale_contract.as_dict(),
            scheduler_previous_sha=None, scheduler_current_sha="abc123",
            scheduler_obligation_digest="0" * 16, scheduler_selected_reviewers=("Codex",),
            scheduler_reasons=("stale",), scheduler_final_sweep=False,
            scheduler_force_full=False, scheduler_calls_avoided=0,
            scheduler_phase="primary", scheduler_primary_reviewer="Codex",
        ),
    )
    monkeypatch.setattr(
        pr_loop, "activate_managed_ci", lambda *args, **kwargs: ManagedCiContract()
    )
    monkeypatch.setattr(pr_loop, "dispatch_final_qualification", lambda *args, **kwargs: None)

    def wait_and_inject(*args, **kwargs):
        _m943_append(runner, stale, login="bot")
        return ManagedCiOutcome(status="passed", head_sha="abc123")

    monkeypatch.setattr(pr_loop, "wait_for_final_qualification", wait_and_inject)
    monkeypatch.setattr(
        pr_loop_support, "merge_pr", lambda *args, **kwargs: pytest.fail("stale contract must block merge")
    )
    with pytest.raises(AgentLoopError, match="no qualification or merge is permitted"):
        run_pr_loop(runner, pr_number=77, config=reduced)


def test_pr_board_amendment_on_the_owning_issue_fails_closed(tmp_path):
    """Row wrong-surface: a pr amendment belongs on the PR, not the issue."""
    record = _m943_amendment_comment(
        flow="pr", issue=None, pr_number=77,
        original_required_reviewers=("Codex", "Gemini", "Antigravity"),
        policy="primary-then-panel", primary_reviewer="Codex",
        removed_reviewers=("Antigravity",), effective_from_round=1,
        rationale="Antigravity quota exhausted.",
    )
    runner = FakeRunner(
        issue_payload={"number": 56, "title": "Linked issue", "body": "Scope."},
        issue_comments=[
            {"author": {"login": "operator"}, "createdAt": "2026-06-01T00:00:00Z", "body": record}
        ],
        pr_payload={"body": "Fixes #56"},
    )
    with pytest.raises(AgentLoopError, match="Post this record on PR #77"):
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path, reviewer=("codex", "gemini")))
    assert _agent_sequence(runner) == []


@pytest.mark.parametrize("auto_merge", [True, False], ids=["auto-merge", "manual-qualification"])
def test_pr_board_amendment_managed_completion_names_the_reduced_board(
    tmp_path, monkeypatch, capsys, auto_merge
):
    """Managed-CI completion paths repeat the reduced-board note durably."""
    runner = _m943_partial_pr_round(tmp_path)
    reduced = _staged_config(tmp_path, reviewer=("codex", "gemini"), min_reviewers=3, auto_merge=auto_merge)
    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=reduced)
    _m943_append(runner, _m943_amendment_from_error(str(excinfo.value)))
    monkeypatch.setattr(
        pr_loop, "activate_managed_ci", lambda *args, **kwargs: ManagedCiContract()
    )
    monkeypatch.setattr(pr_loop, "dispatch_final_qualification", lambda *args, **kwargs: None)
    waits = []
    monkeypatch.setattr(
        pr_loop,
        "wait_for_final_qualification",
        lambda *args, **kwargs: waits.append(True) or ManagedCiOutcome(status="passed", head_sha="abc123"),
    )
    merges = []
    monkeypatch.setattr(pr_loop, "prepare_v2_merge", lambda *args, **kwargs: None)
    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop,
    ), "_merge_with_exact_head_proof", lambda *args, **kwargs: merges.append(kwargs)
    )
    monkeypatch.setattr(
        pr_loop, "publish_manual_v2_qualification", lambda *args, **kwargs: "abc123"
    )
    capsys.readouterr()
    assert run_pr_loop(runner, pr_number=77, config=reduced) == 0

    assert waits, "the managed qualification path was not exercised"
    assert bool(merges) is auto_merge
    out = capsys.readouterr().out
    assert "Reviewer board amended from round 2" in out
    assert "required board now Codex, Gemini" in out
    notes = [
        comment for comment in runner.comments
        if comment.startswith("Review completed on a reduced reviewer board.")
    ]
    assert len(notes) == 1


def test_pr_board_amendment_qualification_refuses_a_digest_on_a_contract_neutral_record(
    tmp_path, monkeypatch
):
    """Row digest-binding at the gate: a neutral record carrying a digest blocks merge."""
    from coding_review_agent_loop.board_amendment import collect_reviewer_board_amendments

    runner = _m943_partial_pr_round(tmp_path)
    reduced = _staged_config(tmp_path, reviewer=("codex", "gemini"), min_reviewers=3, auto_merge=True)
    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=reduced)
    template = _m943_amendment_from_error(str(excinfo.value))
    _m943_append(runner, template)
    (amendment,) = collect_reviewer_board_amendments(
        [SimpleNamespace(body=template)], flow="pr", pr_number=77
    )
    neutral = _attach_round_metadata(
        _staged_review(reviewer="OpenAI Codex"),
        PostedRoundMetadata(
            flow="pr", role="reviewer", agent="Codex", round_number=9, subject="abc123",
            reviewer_board_amendment_digest=amendment.digest,
        ),
    )
    assert architecture_contract._extract_round_metadata_records(
        [SimpleNamespace(body=neutral)], flow="pr"
    )[0].metadata.scheduler_metadata_status == "absent"
    monkeypatch.setattr(
        pr_loop, "activate_managed_ci", lambda *args, **kwargs: ManagedCiContract()
    )
    monkeypatch.setattr(pr_loop, "dispatch_final_qualification", lambda *args, **kwargs: None)
    injected = []

    def wait_and_inject(*args, **kwargs):
        _m943_append(runner, neutral, login="bot")
        injected.append(True)
        return ManagedCiOutcome(status="passed", head_sha="abc123")

    monkeypatch.setattr(pr_loop, "wait_for_final_qualification", wait_and_inject)
    monkeypatch.setattr(
        pr_loop_support, "merge_pr", lambda *args, **kwargs: pytest.fail("a neutral digest must block merge")
    )
    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_merge_with_exact_head_proof",
        lambda *args, **kwargs: pytest.fail("a neutral digest must block merge"),
    )
    with pytest.raises(AgentLoopError, match="(?i)no qualification or merge is permitted") as gate:
        run_pr_loop(runner, pr_number=77, config=reduced)
    assert injected
    assert "contract-neutral" in str(gate.value)


# --- Amended board derived on PR runs, CRLF records, repost recovery (#1133) ---


def _m1133_crlf(body):
    return body.replace("\n", "\r\n")


def _m1133_resume_with_amendment(tmp_path, monkeypatch, *, configured):
    """Partial 3-reviewer round, a CRLF signed amendment, then a resume."""
    runner = _m943_partial_pr_round(tmp_path)
    reduced = _staged_config(tmp_path, reviewer=("codex", "gemini"), min_reviewers=3, auto_merge=True)
    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=reduced)
    _m943_append(runner, _m1133_crlf(_m943_amendment_from_error(str(excinfo.value))))
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *args, **kwargs: None)
    resume_boards = []
    real_resume = pr_loop_support._resume_pr_round

    def spy_resume(*args, **kwargs):
        resume_boards.append(tuple(kwargs["configured_reviewers"]))
        return real_resume(*args, **kwargs)

    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_resume_pr_round", spy_resume)
    calls_before = len(runner.commands)
    config = (
        reduced
        if configured == "reduced"
        else _staged_config(tmp_path, auto_merge=True)
    )
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    new_commands = runner.commands[calls_before:]
    posted = [
        (
            tuple(item.scheduler_contract["required_reviewers"]),
            item.reviewer_board_amendment_digest,
            item.phase,
            item.round_number,
        )
        for item in _posted_scheduler_metadata(runner)
        if item.reviewer_board_amendment_digest is not None
    ]
    return runner, new_commands, posted, resume_boards


@pytest.mark.parametrize("configured", ["original", "reduced"])
def test_1133_amendment_board_used_whichever_board_is_configured(
    tmp_path, monkeypatch, configured
):
    """Rows original-board-force-full-latch, prompt-board-matches-scheduler, crlf-record-parses."""
    runner, new_commands, posted, boards = _m1133_resume_with_amendment(
        tmp_path, monkeypatch, configured=configured
    )
    assert "agy" not in [cmd[0] for cmd, _cwd in new_commands if cmd]
    assert posted and all(board == ("Codex", "Gemini") for board, *_ in posted)
    # Every later board read (prompts, follow-ups, gate) hangs off this rebound board.
    assert boards and all(board == ("codex", "gemini") for board in boards)
    assert any("Reviewer board amendment applied." in comment for comment in runner.comments)


def test_1133_original_and_reduced_boards_produce_identical_records(tmp_path, monkeypatch):
    """Row reduced-board-equivalence."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _, _, posted_original, _ = _m1133_resume_with_amendment(
        tmp_path / "a", monkeypatch, configured="original"
    )
    _, _, posted_reduced, _ = _m1133_resume_with_amendment(
        tmp_path / "b", monkeypatch, configured="reduced"
    )
    assert posted_original and posted_original == posted_reduced


def test_1133_other_configured_board_still_drifts(tmp_path, monkeypatch):
    """Row other-board-still-drifts: only C0 or the amended board is accepted.

    While the signed amendment is still pending, any other board is refused
    before any agent call; once it took effect, a different board would be a
    recorded board change (#1376).
    """
    runner = _m943_partial_pr_round(tmp_path)
    reduced = _staged_config(tmp_path, reviewer=("codex", "gemini"), min_reviewers=3, auto_merge=True)
    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=reduced)
    _m943_append(runner, _m943_amendment_from_error(str(excinfo.value)))
    reordered = _staged_config(tmp_path, reviewer=("codex", "antigravity"), auto_merge=True)
    calls_before = _agent_sequence(runner)
    with pytest.raises(AgentLoopError, match="pending signed reviewer-board amendment"):
        run_pr_loop(runner, pr_number=77, config=reordered)
    assert _agent_sequence(runner) == calls_before


def test_1133_stale_records_posted_after_unread_amendment_print_repost_round(
    tmp_path, monkeypatch
):
    """Rows stale-records-repost-hint and stale-record-no-unread-hint (PR #1124 shape)."""
    runner = _m943_partial_pr_round(tmp_path)
    reduced = _staged_config(tmp_path, reviewer=("codex", "gemini"), min_reviewers=3, auto_merge=True)
    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=reduced)
    template = _m943_amendment_from_error(str(excinfo.value))
    # Amendment comment first, then a scheduler record that still carries C0
    # and no digest (a run that never read the amendment).
    comments = runner.pr_payload["comments"]
    round_records = list(comments)
    comments.clear()
    _m943_append(runner, template)
    comments.extend(round_records)
    for config in (
        _staged_config(tmp_path, auto_merge=True),
        reduced,
    ):
        with pytest.raises(AgentLoopError) as stale:
            run_pr_loop(runner, pr_number=77, config=config)
        message = str(stale.value)
        assert "Delete the amendment comment at PR #77 comment" in message
        assert re.search(r"effective_from_round \d+", message)
        assert "was not recognized" not in message
    # Delete the unread comment and repost at the printed round: the resume
    # then proceeds on the amended board.
    round_number = int(re.search(r"effective_from_round (\d+)", message).group(1))
    comments.remove(next(c for c in comments if "Reviewer board amendment" in c["body"]))
    fresh = template.replace('"effective_from_round": 2', f'"effective_from_round": {round_number}')
    _m943_append(runner, fresh)
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *args, **kwargs: None)
    assert run_pr_loop(
        runner, pr_number=77, config=_staged_config(tmp_path, auto_merge=True)
    ) == 0


def test_1133_unreadable_amendment_after_startup_is_logged_at_the_gate(
    tmp_path, monkeypatch, capsys
):
    """Row gate-unreadable-diagnostic: logged, requirements unchanged, no board change."""
    runner, _, _, _ = _m1133_resume_with_amendment(tmp_path, monkeypatch, configured="original")
    amended = pr_loop.make_contract(("Codex", "Gemini"), "primary-then-panel", None, "Codex")
    reduced = _staged_config(tmp_path, reviewer=("codex", "gemini"), min_reviewers=3, auto_merge=True)
    before = pr_loop_support._fresh_pr_qualification_snapshot(
        runner, config=reduced, pr_number=77, issue_context=None,
        parent_issue_context=None, scheduler_contract=amended,
    )
    unreadable = (
        "Reviewer board amendment:\n\n~~~json\n"
        '{"kind": "reviewer-board-amendment"}\n~~~\n-- Human Reviewer'
    )
    _m943_append(runner, unreadable)
    capsys.readouterr()
    after = pr_loop_support._fresh_pr_qualification_snapshot(
        runner, config=dataclasses.replace(reduced, quiet=False), pr_number=77,
        issue_context=None, parent_issue_context=None, scheduler_contract=amended,
    )
    assert "fence could not be read" in capsys.readouterr().err
    assert after[1] == before[1]


def _m1133_automatic_latch(payload):
    if not payload.get("scheduler_force_full"):
        return None
    return {**payload, "scheduler_force_full_source": "automatic"}


def _m1133_latch_run(tmp_path, monkeypatch, *, configured):
    """Force-full latch in history and on the resume, CRLF amendment, real rounds.

    The first run's latch is rewritten to the automatic source; the resume keeps
    the full-board latch on, so every selection is a full-board selection.
    """
    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *args, **kwargs: None)
    runner = FakeRunner(
        codex_outputs=[_unavailable("OpenAI Codex")],
        gemini_outputs=[_unavailable("Google Gemini")],
        antigravity_outputs=[_unavailable("Antigravity")],
        claude_outputs=[],
    )
    original = _staged_config(
        tmp_path, coder="claude", pr_review_force_full=True, auto_merge=True
    )
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=original)
    _rewrite_pr_metadata(runner, _m1133_automatic_latch)
    latched = _posted_scheduler_metadata(runner)
    assert latched and any(item.scheduler_force_full_source == "automatic" for item in latched)

    reduced = _staged_config(
        tmp_path, coder="claude", reviewer=("codex", "gemini"), min_reviewers=3, auto_merge=True
    )
    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=reduced)
    _m943_append(runner, _m1133_crlf(_m943_amendment_from_error(str(excinfo.value))))

    runner.codex_outputs.extend(
        [
            _staged_review(
                reviewer="OpenAI Codex",
                state="blocking",
                blocking_items=[{"text": "worker cleanup gap", "fix_scope": ["src/worker.py"]}],
            ),
            _staged_review(
                reviewer="OpenAI Codex",
                dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ]
    )
    runner.gemini_outputs.extend(
        [
            _staged_review(reviewer="Google Gemini"),
            _staged_review(
                reviewer="Google Gemini",
                dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ]
    )
    runner.claude_outputs.append(structured_coder_followup(addressed_items=["item-1"]))
    start = len(runner.commands)
    seeded = len(runner.comments)
    config = reduced if configured == "reduced" else original
    config = dataclasses.replace(
        config, pr_review_force_full=True, pr_review_context_mode="compact"
    )
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    commands = runner.commands[start:]
    prompts = [(command[0], command[-1]) for command, _cwd in commands if command[:1] != ["git"]]
    records = [
        (
            tuple(item.scheduler_contract["required_reviewers"]),
            item.reviewer_board_amendment_digest,
            item.phase,
            item.round_number,
            item.scheduler_selected_reviewers,
            item.scheduler_force_full,
        )
        for item in _posted_scheduler_metadata(runner)
        if item.reviewer_board_amendment_digest is not None
    ]
    return runner, commands, prompts, records, runner.comments[seeded:]


@pytest.mark.parametrize("configured", ["original", "reduced"])
def test_1133_automatic_latch_with_original_board_never_reinstates_removed_reviewer(
    tmp_path, monkeypatch, configured
):
    """Rows original-board-force-full-latch and prompt-board-matches-scheduler."""
    runner, commands, prompts, records, new_comments = _m1133_latch_run(
        tmp_path, monkeypatch, configured=configured
    )
    agents = [command[0] for command, _cwd in commands if command[0] in {"claude", "codex", "gemini", "agy"}]
    assert "agy" not in agents
    assert "gemini" in agents and "claude" in agents
    assert records and all(board == ("Codex", "Gemini") for board, *_ in records)
    assert all("Antigravity" not in selected for *_, selected, _latched in records)
    assert any(latched for *_, latched in records)
    assert not any("missing required input" in c for c in new_comments)
    # Every rendered agent prompt that names the board (the run is configured for
    # compact context; whether a prompt is compact depends on session reuse)
    # names only the amended one.
    boards = []
    for _agent, prompt in prompts:
        boards.extend(re.findall(r"All configured reviewers \(([^)]*)\)", prompt))
    assert len(boards) >= 3, "reviewer prompts were not rendered"
    assert all("Antigravity" not in board for board in boards)
    assert set(boards) == {"Codex and Gemini"}
    coder_prompts = [prompt for agent, prompt in prompts if agent == "claude"]
    assert coder_prompts and all("Antigravity" not in prompt for prompt in coder_prompts)
    assert any("Reviewer board amended" in c or "reduced reviewer board" in c for c in new_comments)


def test_1133_latch_original_and_reduced_boards_are_equivalent(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _, _, prompts_a, records_a, _ = _m1133_latch_run(tmp_path / "a", monkeypatch, configured="original")
    _, _, prompts_b, records_b, _ = _m1133_latch_run(tmp_path / "b", monkeypatch, configured="reduced")
    assert records_a == records_b
    def boards(prompts):
        return [re.findall(r"All configured reviewers \(([^)]*)\)", p) for _a, p in prompts]
    assert boards(prompts_a) == boards(prompts_b)


@pytest.mark.parametrize("configured", ["original", "reduced"])
def test_1133_unreadable_amendment_posted_during_managed_ci_does_not_block_qualification(
    tmp_path, monkeypatch, capsys, configured
):
    """Row gate-unreadable-diagnostic, driven through run_pr_loop to the merge."""
    runner = _m943_partial_pr_round(tmp_path)
    reduced = _staged_config(tmp_path, reviewer=("codex", "gemini"), min_reviewers=3, auto_merge=True)
    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=reduced)
    _m943_append(runner, _m1133_crlf(_m943_amendment_from_error(str(excinfo.value))))
    unreadable = (
        "Reviewer board amendment:\n\n~~~json\n"
        '{"kind": "reviewer-board-amendment"}\n~~~\n-- Human Reviewer'
    )
    monkeypatch.setattr(
        pr_loop, "activate_managed_ci", lambda *args, **kwargs: ManagedCiContract()
    )
    monkeypatch.setattr(pr_loop, "dispatch_final_qualification", lambda *args, **kwargs: None)

    def wait_and_inject(*args, **kwargs):
        _m943_append(runner, unreadable, login="operator")
        return ManagedCiOutcome(status="passed", head_sha="abc123")

    monkeypatch.setattr(pr_loop, "wait_for_final_qualification", wait_and_inject)
    merges = []
    monkeypatch.setattr(pr_loop, "prepare_v2_merge", lambda *args, **kwargs: None)
    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop,
    ), "_merge_with_exact_head_proof", lambda *args, **kwargs: merges.append(kwargs)
    )
    config = reduced if configured == "reduced" else _staged_config(tmp_path, auto_merge=True)
    config = dataclasses.replace(config, quiet=False)
    calls_before = len(runner.commands)
    capsys.readouterr()
    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert merges, "qualification did not reach the merge"
    assert "fence could not be read" in capsys.readouterr().err
    assert "agy" not in [
        command[0] for command, _cwd in runner.commands[calls_before:] if command
    ]
    assert not any("missing required input" in comment for comment in runner.comments)


class _PlanCheckReached(Exception):
    """Raised by a patched authorization step once every plan check passed."""


def _m1133_strict_plan_run(tmp_path, monkeypatch, *, supplied):
    """Fresh managed-CI authorization through run_pr_loop, plan approved by three."""
    plan_board = ("codex", "gemini", "antigravity")
    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, pr_plan_recovery, plan_first_loop, issue_loop,
    ), "_resume_plan_round", lambda comments, **_kw: ("approved plan", object())
    )

    def require(*_args, config, error_message, **_kw):
        if tuple(config.reviewer) != plan_board:
            raise AgentLoopError(error_message)

    patch_attr_in(monkeypatch, (
        panel_evidence, pr_loop_support, pr_loop, pr_plan_recovery,
    ), "_require_complete_canonical_plan_approval", require)
    # The fixture stubs the completion predicate, including the shared
    # classification recovery now applies to real plan history (#1377).
    import coding_review_agent_loop.plan_verification as plan_verification

    monkeypatch.setattr(plan_verification, "require_recoverable_plan_approval", lambda *a, **k: None)

    def reached(*_args, **_kwargs):
        raise _PlanCheckReached

    monkeypatch.setattr(pr_loop, "authorize_fresh_issue_created_resume", reached)
    runner = _m943_partial_pr_round(
        tmp_path,
        issue_payload={"number": 56, "title": "Linked issue", "body": "Scope."},
        pr_payload={"body": "Fixes #56"},
    )
    reduced = _staged_config(tmp_path, reviewer=("codex", "gemini"), min_reviewers=3, auto_merge=True)
    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=reduced)
    _m943_append(runner, _m943_amendment_from_error(str(excinfo.value)))
    config = _staged_config(
        tmp_path,
        reviewer=supplied,
        managed_ci=True,
        managed_ci_fresh_authorization=True,
        managed_ci_issue_number=56,
        managed_ci_trusted_actor="agent-loop",
    )
    return runner, config


def test_1133_strict_plan_check_passes_with_the_original_board_despite_a_pr_amendment(
    tmp_path, monkeypatch
):
    """Row strict-managed-plan-binding-keeps-plan-board, original board."""
    runner, config = _m1133_strict_plan_run(
        tmp_path, monkeypatch, supplied=("codex", "gemini", "antigravity")
    )
    with pytest.raises(_PlanCheckReached):
        run_pr_loop(runner, pr_number=77, config=config)


def test_1133_strict_plan_check_refuses_the_reduced_board_with_a_rerun_hint(
    tmp_path, monkeypatch
):
    """Row strict-managed-plan-binding-keeps-plan-board, reduced board."""
    runner, config = _m1133_strict_plan_run(
        tmp_path, monkeypatch, supplied=("codex", "gemini")
    )
    with pytest.raises(AgentLoopError, match="Rerun with the original reviewer board"):
        run_pr_loop(runner, pr_number=77, config=config)


def _m1133_post_panel_latch_run(tmp_path, monkeypatch, *, configured):
    """Qualified panel opening, durable automatic latch, then a CRLF amendment.

    No operator force-full flag: the full board is selected only by the
    automatic latch raised after the panel opened.
    """
    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    merges = []
    monkeypatch.setattr(
        pr_loop_support, "merge_pr", lambda *args, **kwargs: merges.append(kwargs)
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[
            _staged_review(
                reviewer="Google Gemini",
                state="blocking",
                blocking_items=[{"text": "panel regression", "fix_scope": ["src/worker.py"]}],
            )
        ],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    # primary -> panel -> coder; the remediation round then runs out of output.
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path, auto_merge=True))
    assert _agent_sequence(runner)[:4] == ["codex", "gemini", "agy", "claude"]

    def latch_after_opening(payload):
        if payload.get("role") != "coder" or "scheduler_contract" not in payload:
            return None
        return {
            **payload,
            "scheduler_force_full": True,
            "scheduler_force_full_source": "automatic",
        }

    _rewrite_pr_metadata(runner, latch_after_opening)
    reduced = _staged_config(tmp_path, reviewer=("codex", "gemini"), min_reviewers=3, auto_merge=True)
    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=reduced)
    _m943_append(runner, _m1133_crlf(_m943_amendment_from_error(str(excinfo.value))))

    seeded_agents = len(_agent_sequence(runner))
    seeded_commands = len(runner.commands)
    seeded_comments = len(runner.comments)
    runner.advance_pr_head_on_coder_followup = False
    resolved = [{"item_id": "item-1", "disposition": "resolved"}]
    # Round 3 (full board): Gemini keeps the item open, so the coder runs again
    # on the same head and round 4 reuses each reviewer's round-3 session
    # (compact context).
    runner.codex_outputs.extend(
        [
            _staged_review(reviewer="OpenAI Codex", dispositions=resolved),
            _staged_review(reviewer="OpenAI Codex", dispositions=resolved),
        ]
    )
    runner.gemini_outputs.extend(
        [
            _staged_review(
                reviewer="Google Gemini",
                state="blocking",
                dispositions=[
                    {"item_id": "item-1", "disposition": "blocking", "note": "still broken"}
                ],
            ),
            _staged_review(reviewer="Google Gemini", dispositions=resolved),
        ]
    )
    runner.claude_outputs.append(structured_coder_followup(addressed_items=["item-1"]))
    config = reduced if configured == "reduced" else _staged_config(tmp_path, auto_merge=True)
    config = dataclasses.replace(config, pr_review_context_mode="compact")
    assert not config.pr_review_force_full
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    prompts = [
        (command[0], command[-1])
        for command, _cwd in runner.commands[seeded_commands:]
        if command[0] in {"codex", "gemini", "agy", "claude"}
    ]
    records = [
        (
            tuple(item.scheduler_contract["required_reviewers"]),
            item.reviewer_board_amendment_digest,
            item.phase,
            item.scheduler_selected_reviewers,
            item.scheduler_force_full,
            item.scheduler_force_full_source,
        )
        for item in _posted_scheduler_metadata(runner)
        if item.reviewer_board_amendment_digest is not None
    ]
    runner.merge_calls = merges
    return (
        runner,
        _agent_sequence(runner)[seeded_agents:],
        prompts,
        records,
        runner.comments[seeded_comments:],
    )


@pytest.mark.parametrize("configured", ["original", "reduced"])
def test_1133_post_panel_automatic_latch_uses_the_amended_board(
    tmp_path, monkeypatch, configured
):
    """Rows original-board-force-full-latch and prompt-board-matches-scheduler."""
    runner, new_agents, prompts, records, new_comments = _m1133_post_panel_latch_run(
        tmp_path, monkeypatch, configured=configured
    )
    assert set(new_agents) == {"codex", "gemini", "claude"}
    assert new_agents.count("gemini") == 2 and "agy" not in new_agents
    audits = [c for c in new_comments if c.startswith("PR review scheduling audit:")]
    assert len(audits) >= 2
    for audit in audits:
        assert "phase: full-board" in audit
        assert "force-full: True (source: automatic)" in audit
        assert "selected Codex, Gemini; paused none;" in audit
    assert records and all(board == ("Codex", "Gemini") for board, *_ in records)
    assert all("Antigravity" not in selected for _b, _d, _p, selected, *_ in records)
    assert any(latched and source == "automatic" for *_, latched, source in records)
    assert not any("missing required input" in c for c in new_comments)
    # Qualification: exactly one exact-head merge, after every latched review,
    # followed by the durable reduced-board completion note.
    assert runner.merge_calls == [{"expected_head_sha": "abc123-coder-1"}]
    notes = [c for c in new_comments if c.startswith("Review completed on a reduced reviewer board.")]
    assert len(notes) == 1 and "required board now Codex, Gemini" in notes[0]
    last_review = max(i for i, c in enumerate(new_comments) if "Google Gemini review" in c)
    assert new_comments.index(notes[0]) > last_review
    reviewer_prompts = [prompt for agent, prompt in prompts if agent in {"codex", "gemini"}]
    boards = [
        board
        for prompt in reviewer_prompts
        for board in re.findall(r"All configured reviewers \(([^)]*)\)", prompt)
    ]
    assert boards and set(boards) == {"Codex and Gemini"}
    compact = [p for p in reviewer_prompts if COMPACT_PR_REVIEW_VOLATILE_TAIL_MARKER in p]
    assert compact, "no reviewer turn entered the compact branch"
    assert all("Antigravity" not in prompt for prompt in compact)


def test_1133_post_panel_latch_original_and_reduced_boards_are_equivalent(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    *_, records_a, _ = _m1133_post_panel_latch_run(tmp_path / "a", monkeypatch, configured="original")
    *_, records_b, _ = _m1133_post_panel_latch_run(tmp_path / "b", monkeypatch, configured="reduced")
    assert records_a and records_a == records_b


def _m1133_ordinary_resume_strict(tmp_path, monkeypatch, *, supplied, entry="ordinary"):
    """Managed-CI entry (ordinary resume or fresh authorization) of an issue-created strict PR."""
    from coding_review_agent_loop.managed_ci import AuthenticatedIssueCreatedHandoff

    runner = _m943_partial_pr_round(
        tmp_path,
        issue_payload={"number": 56, "title": "Linked issue", "body": "Scope."},
        pr_payload={"body": "Fixes #56", "headRefName": "agent-loop/managed-56"},
    )
    reduced = _staged_config(tmp_path, reviewer=("codex", "gemini"), min_reviewers=3, auto_merge=True)
    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=reduced)
    _m943_append(runner, _m1133_crlf(_m943_amendment_from_error(str(excinfo.value))))
    plan_board = ("codex", "gemini", "antigravity")
    plan_checks = []
    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, pr_plan_recovery, plan_first_loop, issue_loop,
    ), "_resume_plan_round", lambda comments, **_kw: ("approved plan", object())
    )

    def require(*_args, config, error_message, **_kw):
        plan_checks.append(tuple(config.reviewer))
        if tuple(config.reviewer) != plan_board:
            raise AgentLoopError(error_message)

    patch_attr_in(monkeypatch, (
        panel_evidence, pr_loop_support, pr_loop, pr_plan_recovery,
    ), "_require_complete_canonical_plan_approval", require)
    # The fixture stubs the completion predicate, including the shared
    # classification recovery now applies to real plan history (#1377).
    import coding_review_agent_loop.plan_verification as plan_verification

    monkeypatch.setattr(plan_verification, "require_recoverable_plan_approval", lambda *a, **k: None)
    recovered_plan = execution_policy.make_approved_plan_context(
        "approved plan",
        source_locator="issue #56 canonical approved plan",
        expected_hash=panel_evidence.approved_plan_hash("approved plan"),
    )
    patch_attr_in(monkeypatch, (
        execution_policy, child_plan_binding, pr_loop_support, pr_loop, issue_loop,
    ), "recover_approved_plan_context", lambda *a, **k: recovered_plan
    )
    handoff = AuthenticatedIssueCreatedHandoff(
        pr_number=77, issue_number=56, repository="OWNER/REPO", base_ref="main",
        head_sha="abc123", branch="agent-loop/managed-56", trusted_actor_login="agent-loop",
        trusted_actor_id=1, protection_mode="strict", override_nonce=None,
    )
    monkeypatch.setattr(pr_loop, "recover_issue_created_handoff", lambda *a, **k: handoff)
    monkeypatch.setattr(
        pr_loop, "authorize_fresh_issue_created_resume", lambda *a, **k: handoff
    )
    monkeypatch.setattr(pr_loop, "revalidate_issue_created_handoff", lambda *a, **k: handoff)
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pr_loop, "activate_managed_ci", lambda *args, **kwargs: ManagedCiContract()
    )
    monkeypatch.setattr(pr_loop, "dispatch_final_qualification", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pr_loop,
        "wait_for_final_qualification",
        lambda *args, **kwargs: ManagedCiOutcome(status="passed", head_sha="abc123"),
    )
    monkeypatch.setattr(pr_loop, "prepare_v2_merge", lambda *args, **kwargs: None)
    merges = []
    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop,
    ), "_merge_with_exact_head_proof", lambda *args, **kwargs: merges.append(kwargs)
    )
    fresh = (
        {"managed_ci_fresh_authorization": True, "managed_ci_issue_number": 56}
        if entry == "fresh"
        else {}
    )
    config = _staged_config(
        tmp_path, reviewer=supplied, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        auto_merge=True, **fresh,
    )
    return runner, config, plan_checks, merges


@pytest.mark.parametrize("entry", ["ordinary", "fresh"])
def test_1133_managed_entry_and_qualification_use_the_supplied_board_for_the_plan(
    tmp_path, monkeypatch, entry
):
    """Row strict-managed-plan-binding-keeps-plan-board, original board, to the merge."""
    runner, config, plan_checks, merges = _m1133_ordinary_resume_strict(
        tmp_path, monkeypatch, supplied=("codex", "gemini", "antigravity"), entry=entry
    )
    calls_before = len(runner.commands)
    runner.codex_outputs.extend([_staged_review(reviewer="OpenAI Codex")] * 2)
    runner.gemini_outputs.extend([_staged_review(reviewer="Google Gemini")] * 2)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    # The ordinary-resume check and the qualification binding both saw the
    # supplied three-reviewer board; the PR itself ran on the amended board.
    assert len(plan_checks) >= 2
    assert set(plan_checks) == {("codex", "gemini", "antigravity")}
    assert merges
    assert "agy" not in [c[0] for c, _cwd in runner.commands[calls_before:] if c]
    amended = [
        tuple(item.scheduler_contract["required_reviewers"])
        for item in _posted_scheduler_metadata(runner)
        if item.reviewer_board_amendment_digest is not None
    ]
    assert amended and set(amended) == {("Codex", "Gemini")}


@pytest.mark.parametrize("entry", ["ordinary", "fresh"])
def test_1133_managed_entry_refuses_the_reduced_board_with_the_rerun_hint(
    tmp_path, monkeypatch, entry
):
    runner, config, plan_checks, merges = _m1133_ordinary_resume_strict(
        tmp_path, monkeypatch, supplied=("codex", "gemini"), entry=entry
    )
    with pytest.raises(AgentLoopError, match="Rerun with the original reviewer board"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert plan_checks == [("codex", "gemini")] and not merges


# --- Named reviewer seats on issue-created PRs (#1365) -----------------------

_ACTOR_1365 = ("agent-loop", 4242)
_FOREIGN_1365 = 999
_WRITE_PREFIXES_1365 = (
    ["gh", "pr", "comment"], ["gh", "pr", "edit"], ["gh", "pr", "ready"], ["gh", "pr", "merge"],
    ["gh", "issue", "comment"], ["gh", "issue", "edit"], ["gh", "label"], ["gh", "workflow"],
)


class _PrLoopStopped1365(Exception):
    pass


def _seats_1365(tmp_path, *names):
    from coding_review_agent_loop.reviewer_seats import ReviewerSeat, SeatAgent

    known = {
        "sol": ("codex", "gpt-6.1-sol"),
        "flash": ("antigravity", "Gemini 3.8 Flash (High)"),
        "opus-low": ("antigravity", "Claude Opus 5.5 (Low)"),
        "opus-high": ("antigravity", "Claude Opus 5.5 (High)"),
    }
    seats = []
    for name in names:
        workdir = tmp_path / name
        workdir.mkdir(exist_ok=True)
        backend, model = known[name]
        seats.append(SeatAgent(ReviewerSeat(name, backend, (model,)), workdir))
    return tuple(seats)


def _named_config_1365(tmp_path, seats, **overrides):
    return make_config(
        tmp_path, reviewer=seats, reviewer_seats=seats, pre_review_tests=False,
        agent_max_retries=0, **overrides,
    )


def _queue_1365(runner, seats, review=structured_pr_review):
    from coding_review_agent_loop.agents.registry import agent_signature

    for seat in seats:
        outputs = runner.codex_outputs if seat.backend == "codex" else runner.antigravity_outputs
        outputs.append(review(state="approved", reviewer=agent_signature(seat)))


def _records_1365(runner):
    from coding_review_agent_loop.round_state import _extract_round_metadata_records

    return _extract_round_metadata_records(
        [SimpleNamespace(body=c["body"]) for c in runner.pr_payload.get("comments", [])],
        flow="pr",
    )


def _agents_1365(runner, start=0):
    return [cmd[0] for cmd, _cwd in runner.commands[start:] if cmd and cmd[0] in {"claude", "codex", "agy"}]


def _writes_1365(runner, start=0):
    return [
        cmd for cmd, _cwd in runner.commands[start:]
        if any(cmd[:len(prefix)] == prefix for prefix in _WRITE_PREFIXES_1365)
        or (cmd[:2] == ["gh", "api"] and any(flag in cmd for flag in ("POST", "PATCH", "DELETE", "PUT")))
    ]


def _links_1365(runner):
    from coding_review_agent_loop.board_change import BOARD_CHANGE_LINK_HEADING

    return [
        c["body"] for c in runner.pr_payload.get("comments", [])
        if c["body"].startswith(BOARD_CHANGE_LINK_HEADING)
    ]


def _audits_1365(runner):
    from coding_review_agent_loop.plan_verification import BOARD_CHANGE_AUDIT_HEADING

    return [
        c["body"] for c in (*runner.pr_payload.get("comments", []), *runner.issue_comments)
        if BOARD_CHANGE_AUDIT_HEADING in c["body"]
    ]


def _serve_pr_rest_1365(runner):
    """Serve PR #77's REST history from the PR's own comments, not issue #56's."""
    runner.issue_comments_by_number[77] = runner.pr_payload.setdefault("comments", [])
    return runner


def _affected_issue_created_pr_1365(tmp_path, monkeypatch):
    """A #1364-shaped PR: a named plan-first run stopped after the initial coder record.

    The issue holds the actor-authored bound plan and canonical handoff
    record; the PR holds only the actor-authored unbound authoritative coder
    record at the exact head.
    """
    import coding_review_agent_loop.issue_implementation as issue_implementation

    seats = _seats_1365(tmp_path, "sol", "flash", "opus-low")
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(summary="Initial plan.", plan_steps=["Make the change."]),
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        pr_payload={"body": "Fixes #56"},
        authenticated_actor=_ACTOR_1365,
        serve_rest_issue_comments=True,
    )
    _serve_pr_rest_1365(runner)
    _queue_1365(runner, seats, structured_plan_review)
    config = _named_config_1365(tmp_path, seats)

    def stopped(*_args, **_kwargs):
        raise _PrLoopStopped1365("pre-#1375 named PR entry refusal")

    with monkeypatch.context() as patch_context:
        patch_context.setattr(issue_implementation, "run_pr_loop", stopped)
        with pytest.raises(_PrLoopStopped1365):
            run_issue_loop(
                runner, issue_number=56, config=config, plan_first=True,
                implement_after_approval=True,
            )
    (coder,) = _records_1365(runner)
    assert (coder.metadata.role, coder.metadata.phase, coder.metadata.subject) == (
        "coder", "authoritative", runner.pr_payload.get("headRefOid", "abc123"),
    )
    assert coder.metadata.seat_binding is None and coder.metadata.pr_board_root is None
    return runner, seats, config


def test_affected_issue_created_named_pr_recovers_into_review_by_rerun(tmp_path, monkeypatch):
    """Row affected-pr-recovery: a #1364-shaped PR resumes with no backfill."""
    runner, seats, config = _affected_issue_created_pr_1365(tmp_path, monkeypatch)
    history = [dict(comment) for comment in runner.pr_payload["comments"]]
    issue_history = [dict(comment) for comment in runner.issue_comments]
    _queue_1365(runner, seats)
    start = len(runner.commands)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    # The entry check read the PR and the owning issue through REST first.
    reads = [cmd[2] for cmd, _cwd in runner.commands[start:] if cmd[:2] == ["gh", "api"]]
    assert "repos/OWNER/REPO/issues/77/comments?per_page=100&page=1" in reads
    assert "repos/OWNER/REPO/issues/56/comments?per_page=100&page=1" in reads
    # The coder record at the exact head is adopted: the coder never reruns
    # and every seat reviews that head.
    assert _agents_1365(runner, start) == ["codex", "agy", "agy"]
    coder, *later = _records_1365(runner)
    assert coder.metadata.role == "coder" and coder.metadata.seat_binding is None
    assert [(r.metadata.role, r.metadata.agent, r.metadata.subject, r.metadata.state) for r in later] == [
        ("reviewer", "sol", "abc123", "approved"),
        ("reviewer", "flash", "abc123", "approved"),
        ("reviewer", "opus-low", "abc123", "approved"),
    ]
    assert all(r.metadata.pr_board_root is not None for r in later)
    # Historical comments are untouched and nothing was backfilled.
    assert runner.pr_payload["comments"][:len(history)] == history
    assert runner.issue_comments[:len(issue_history)] == issue_history
    assert not any(
        cmd[:2] == ["gh", "api"] and ("PATCH" in cmd or "DELETE" in cmd)
        for cmd, _cwd in runner.commands[start:]
    )
    assert _links_1365(runner) == [] and _audits_1365(runner) == []


def test_coder_only_named_pr_under_changed_board_posts_one_audit_and_no_link(tmp_path, monkeypatch):
    """Row coder-only-changed-board: the handoff audit, never a lineage link."""
    runner, _seats, _config = _affected_issue_created_pr_1365(tmp_path, monkeypatch)
    changed = _seats_1365(tmp_path, "sol", "flash", "opus-high")
    # The floor is enforced on the changed board and refuses it below three.
    reduced = _named_config_1365(tmp_path, changed[:2], min_reviewers=3)
    comments = len(runner.pr_payload["comments"])
    start = len(runner.commands)
    with pytest.raises(AgentLoopError, match="below the floor of 3"):
        run_pr_loop(runner, pr_number=77, config=reduced)
    assert len(runner.pr_payload["comments"]) == comments
    assert _writes_1365(runner, start) == [] and _agents_1365(runner, start) == []

    config = _named_config_1365(tmp_path, changed, min_reviewers=3)
    _queue_1365(runner, changed)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert _links_1365(runner) == []
    (audit,) = _audits_1365(runner)
    assert "opus-high" in audit
    reviews = [r.metadata for r in _records_1365(runner) if r.metadata.role == "reviewer"]
    assert sorted(m.agent for m in reviews) == ["flash", "opus-high", "sol"]
    # A rerun derives the same board: no second audit record and no link.
    _queue_1365(runner, changed)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert len(_audits_1365(runner)) == 1 and _links_1365(runner) == []


@pytest.mark.parametrize("floor", [None, 3], ids=["no-floor", "floor"])
def test_foreign_legacy_coder_record_is_refused_at_named_pr_entry(tmp_path, monkeypatch, floor):
    """Row foreign-coder-refused: the unconditional entry check, with and without a floor."""
    runner, seats, config = _affected_issue_created_pr_1365(tmp_path, monkeypatch)
    (coder,) = runner.pr_payload["comments"]
    coder["_rest_author_id"] = _FOREIGN_1365
    coder["author"] = {"login": "intruder"}
    if floor is not None:
        config = dataclasses.replace(config, min_reviewers=floor)
    _queue_1365(runner, seats)
    start = len(runner.commands)
    with pytest.raises(AgentLoopError, match="Human decision required") as refused:
        run_pr_loop(runner, pr_number=77, config=config)
    assert "coder round record" in str(refused.value) and "intruder" in str(refused.value)
    assert len(runner.pr_payload["comments"]) == 1
    assert _writes_1365(runner, start) == [] and _agents_1365(runner, start) == []


def test_foreign_canonical_handoff_record_is_refused_at_named_pr_entry(tmp_path, monkeypatch):
    """Row foreign-handoff-refused: no floor, no amendments and no links."""
    from coding_review_agent_loop.issue_pr_handoff import find_latest_issue_pr_handoff

    runner, seats, config = _affected_issue_created_pr_1365(tmp_path, monkeypatch)
    (handoff_comment,) = [
        c for c in runner.issue_comments if "AGENT_ISSUE_PR_HANDOFF" in c["body"]
    ]
    selected = find_latest_issue_pr_handoff(
        [SimpleNamespace(body=c["body"]) for c in runner.issue_comments],
        issue_number=56, repo="OWNER/REPO",
    )
    assert selected.pr_number == 77 and selected.flow == "approved-plan-implementation"
    handoff_comment["_rest_author_id"] = _FOREIGN_1365
    handoff_comment["author"] = {"login": "intruder"}
    _queue_1365(runner, seats)
    start = len(runner.commands)
    with pytest.raises(AgentLoopError, match="Human decision required") as refused:
        run_pr_loop(runner, pr_number=77, config=config)
    assert "handoff record on issue #56" in str(refused.value)
    assert len(runner.pr_payload["comments"]) == 1
    assert _writes_1365(runner, start) == [] and _agents_1365(runner, start) == []


@pytest.mark.parametrize("head", ["abc123", "def456"], ids=["exact-head", "stale-head"])
def test_named_pr_adopts_the_legacy_coder_record_only_at_its_exact_head(tmp_path, monkeypatch, head):
    """Row wrong-head-coder: a coder record at another head is not this round's output."""
    from coding_review_agent_loop.round_state import pr_resume_needs_author_admission

    runner, seats, config = _affected_issue_created_pr_1365(tmp_path, monkeypatch)
    runner.pr_payload["headRefOid"] = head
    comments = [SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]]
    # A record-less live head is the existing unrecorded-head path.
    assert pr_resume_needs_author_admission(comments, head) is (head != "abc123")
    _queue_1365(runner, seats)
    start = len(runner.commands)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    reviews = [r.metadata for r in _records_1365(runner) if r.metadata.role == "reviewer"]
    assert {m.subject for m in reviews} == {head}
    # Live PR review footers name the seat, backend, model and effort, while
    # the records keep the bare seat IDs (#1363).
    assert sorted(m.agent for m in reviews) == ["flash", "opus-low", "sol"]
    footers = [
        c["body"].rstrip().splitlines()[-1] for c in runner.pr_payload["comments"]
        if "**Review verdict:**" in c["body"]
    ]
    assert footers == [
        "-- sol (OpenAI Codex: gpt-6.1-sol (medium))",
        "-- flash (Google Antigravity: Gemini 3.8 Flash (High))",
        "-- opus-low (Google Antigravity: Claude Opus 5.5 (Low))",
    ]
    prompts = [cmd[-1] for cmd, _cwd in runner.commands[start:] if cmd[0] in {"codex", "agy"}]
    assert len(prompts) == 3
    # Only the exact-head coder record reaches the reviewers as the round's
    # coder output; the stale one is never shown for the live head.
    assert all(("Implemented approved plan." in prompt) is (head == "abc123") for prompt in prompts)


def _issue_created_named_pr_after_review_1365(tmp_path):
    """A direct-issue named PR after one approved, anchored reviewer round."""
    seats = _seats_1365(tmp_path, "sol", "flash", "opus-low")
    runner = _serve_pr_rest_1365(FakeRunner(
        claude_outputs=[structured_issue_implementation(pr_number=77)],
        pr_payload={"body": "Fixes #56"},
        authenticated_actor=_ACTOR_1365,
        serve_rest_issue_comments=True,
    ))
    _queue_1365(runner, seats)
    assert run_issue_loop(runner, issue_number=56, config=_named_config_1365(tmp_path, seats)) == 0
    coder, *later = _records_1365(runner)
    assert coder.metadata.role == "coder" and coder.metadata.pr_board_root is None
    assert later and all(r.metadata.pr_board_root is not None for r in later)
    return runner


def test_issue_created_named_pr_changes_board_mid_run_and_merges_the_approved_head(tmp_path):
    """Row midrun-board-change: one link, the new seat approves, one exact-head merge."""
    from coding_review_agent_loop.board_change import BoardIdentity

    runner = _issue_created_named_pr_after_review_1365(tmp_path)
    changed = _seats_1365(tmp_path, "sol", "flash", "opus-high")
    config = _named_config_1365(tmp_path, changed, min_reviewers=3)
    _queue_1365(runner, changed)
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    (link,) = _links_1365(runner)
    assert "opus-high" in link and "opus-low" in link
    assert "agy" in _agents_1365(runner, start)
    replacement = [
        r.metadata for r in _records_1365(runner)
        if r.metadata.role == "reviewer" and r.metadata.agent == "opus-high"
    ]
    assert replacement and all(m.state == "approved" and m.subject == "abc123" for m in replacement)
    # The initial coder record never pinned the board: every later record keeps
    # the handoff root while the run's identity is the changed board.
    roots = {
        BoardIdentity.from_payload(r.metadata.pr_board_root).names
        for r in _records_1365(runner) if r.metadata.pr_board_root is not None
    }
    assert roots == {("sol", "flash", "opus-low")}
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _cwd in runner.commands)

    # The same board with merging enabled posts no second link and merges
    # exactly the approved head.
    _queue_1365(runner, changed)
    merging = dataclasses.replace(config, auto_merge=True)
    assert run_pr_loop(runner, pr_number=77, config=merging) == 0
    assert len(_links_1365(runner)) == 1
    merges = [cmd for cmd, _cwd in runner.commands if cmd[:3] == ["gh", "pr", "merge"]]
    assert merges == [
        ["gh", "pr", "merge", "77", "--repo", "OWNER/REPO", "--merge", "--match-head-commit", "abc123"]
    ]


def test_issue_created_named_pr_below_floor_board_change_is_refused_before_any_write(tmp_path):
    """Row floor-still-enforced: an unsigned below-floor board posts nothing."""
    runner = _issue_created_named_pr_after_review_1365(tmp_path)
    reduced = _named_config_1365(tmp_path, _seats_1365(tmp_path, "sol", "flash"), min_reviewers=3)
    comments = len(runner.pr_payload["comments"])
    start = len(runner.commands)
    with pytest.raises(AgentLoopError, match="below the floor of 3"):
        run_pr_loop(runner, pr_number=77, config=reduced)
    assert len(runner.pr_payload["comments"]) == comments and _links_1365(runner) == []
    assert _writes_1365(runner, start) == [] and _agents_1365(runner, start) == []
