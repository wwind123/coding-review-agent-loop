"""PR-loop tests for pr_loop: coder follow-up handling (#1542)."""

import dataclasses
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from coding_review_agent_loop.runner import CommandResult
from coding_review_agent_loop.cli import AgentLoopError, run_pr_loop
from coding_review_agent_loop.comment_rendering import (
    _render_public_coder_followup_comment,
    render_risk_test_matrix_section,
)
from coding_review_agent_loop.github import (
    IssueComment,
    PullRequestCheck,
)
from coding_review_agent_loop.managed_ci import (
    ManagedCiContract,
    ManagedCiOutcome,
)
from coding_review_agent_loop.round_state import (
    QualificationCheckpoint,
    PostedRoundMetadata,
    _attach_round_metadata,
    _decode_round_metadata,
    _resume_pr_round,
)
from coding_review_agent_loop.agent_failure import ValidatedAgentResponse
from coding_review_agent_loop.review_scheduling import TransitionClassification
from coding_review_agent_loop.protocol import (
    ReviewItemDisposition,
    UnresolvedReviewItem,
    parse_risk_test_matrix,
    risk_test_matrix_identity,
    validate_structured_coder_followup,
)
from agent_loop_helpers import (
    FakeRunner,
    command_index,
    make_config,
    structured_coder_followup,
    structured_pr_review,
)
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.execution_policy as execution_policy
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.plan_first_loop as plan_first_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
import coding_review_agent_loop.response_validation as response_validation
import coding_review_agent_loop.triage as triage
import coding_review_agent_loop.validated_agent as validated_agent
from agent_loop_helpers import patch_attr_in
from pr_loop_test_helpers import (
    _agent_sequence,
    _followup_matrix_context,
    _followup_observation,
    _init_followup_checkout,
    _staged_config,
    _staged_review,
    _watch_check_board,
)


def test_m985_pr_loop_stops_after_two_unchanged_head_coder_turns(tmp_path):
    from coding_review_agent_loop.errors import AgentLoopError

    blocking = structured_pr_review(
        state="blocking", summary="Re-plan the issue first.", blocking_items=["Re-plan first."]
    )
    runner = FakeRunner(
        codex_outputs=[
            blocking,
            structured_pr_review(
                state="blocking",
                summary="Still needs a re-plan.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "blocking", "note": "Re-plan first."}
                ],
            ),
            structured_pr_review(state="approved"),
        ],
        claude_outputs=[
            structured_coder_followup(remaining_items=["item-1"], addressed_items=[]),
            structured_coder_followup(remaining_items=["item-1"], addressed_items=[]),
        ],
        advance_pr_head_on_coder_followup=False,
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=6)
    with pytest.raises(Exception) as excinfo:
        run_pr_loop(runner, pr_number=77, config=config)
    assert isinstance(excinfo.value, AgentLoopError)
    assert "unchanged in 2 consecutive follow-up rounds" in str(excinfo.value)
    # The identical head is not reviewed a third time.
    assert len([cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]) == 2


def test_m985_unchanged_head_count_resets_after_an_external_head_advance():
    from coding_review_agent_loop.pr_loop_support import (
        MAX_UNCHANGED_HEAD_CODER_TURNS,
        _UnchangedHeadTracker,
    )

    tracker = _UnchangedHeadTracker()
    # The coder leaves A unchanged once; another actor then pushes B.
    assert tracker.observe("A", "A") == 1
    # B has had only one no-progress follow-up, so the loop must not stop.
    assert tracker.observe("B", "B") == 1 < MAX_UNCHANGED_HEAD_CODER_TURNS
    assert tracker.observe("B", "B") == MAX_UNCHANGED_HEAD_CODER_TURNS
    # A follow-up that moves the head clears the count.
    assert tracker.observe("B", "C") == 0
    assert tracker.observe("C", "C") == 1
    # An unknown reviewed head never counts.
    assert tracker.observe(None, None) == 0


# --- #1034: head-aware PR coder follow-up reporting -------------------------

_M1034_META_RE = re.compile(r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->")
_M1034_VERIFY_INSTRUCTION = "do not credit the turn with fixes"


def _m1034_coder_comments(runner):
    return [
        comment["body"]
        for comment in runner.pr_payload["comments"]
        if "## Coder follow-up" in comment["body"]
    ]


def _m1034_metadata(body):
    match = _M1034_META_RE.search(body)
    assert match is not None
    return _decode_round_metadata(match.group("payload"))


def _m1034_reviewer_prompts(runner):
    return [cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]


def test_m1034_followup_head_log_wording():
    from coding_review_agent_loop.pr_loop_support import _coder_followup_head_log

    advanced = _coder_followup_head_log(3, "Claude", "a" * 40, "b" * 40, 0)
    assert advanced == (
        f"Round 3: PR head advanced {'a' * 12}..{'b' * 12} during Claude follow-up; re-reviewing"
    )
    unchanged = _coder_followup_head_log(3, "Codex", "c" * 40, "c" * 40, 1)
    assert unchanged == f"Round 3: Codex follow-up left PR head {'c' * 12} unchanged (1/2); re-reviewing"
    for dispatch, observed in ((None, "d" * 40), ("d" * 40, None), (None, None)):
        unknown = _coder_followup_head_log(3, "Claude", dispatch, observed, 0)
        assert unknown == "Round 3: Claude follow-up complete; PR head change could not be determined"
    for message in (advanced, unchanged, unknown):
        assert "pushed" not in message
        assert "unchanged" not in unknown


def test_m1034_followup_dispatch_head_round_trips_and_legacy_records_stay_byte_stable():
    from coding_review_agent_loop.round_state import (
        _encode_round_metadata,
        followup_head_unchanged_sha,
    )

    recorded = PostedRoundMetadata(
        flow="pr", role="coder", agent="Claude", round_number=2,
        subject="abc123", followup_dispatch_head="abc123",
    )
    decoded = _decode_round_metadata(_encode_round_metadata(recorded))
    assert decoded.followup_dispatch_head == "abc123"
    assert followup_head_unchanged_sha(decoded) == "abc123"
    # An advanced head never yields the signal.
    assert followup_head_unchanged_sha(dataclasses.replace(recorded, subject="def456")) is None
    assert followup_head_unchanged_sha(dataclasses.replace(recorded, subject="unknown")) is None
    assert followup_head_unchanged_sha(dataclasses.replace(recorded, role="reviewer")) is None
    assert followup_head_unchanged_sha(None) is None

    legacy = PostedRoundMetadata(
        flow="pr", role="coder", agent="Claude", round_number=2, subject="abc123",
    )
    encoded = _encode_round_metadata(legacy)
    legacy_decoded = _decode_round_metadata(encoded)
    assert legacy_decoded.followup_dispatch_head is None
    assert _encode_round_metadata(legacy_decoded) == encoded
    # Legacy records never infer the head-unchanged signal from other fields.
    assert followup_head_unchanged_sha(legacy_decoded) is None


def test_m1034_malformed_followup_dispatch_head_is_rejected():
    from coding_review_agent_loop.round_state import (
        _decode_round_metadata_mapping,
        _encode_round_metadata,
        decode_mapping,
    )

    with pytest.raises(ValueError):
        PostedRoundMetadata(
            flow="pr", role="coder", agent="Claude", round_number=2,
            subject="abc123", followup_dispatch_head="not a sha!",
        )
    # Placeholders and non-hex names are not Git commit SHAs.
    for bad in ("unknown", "new-sha", "arbitrary.word"):
        with pytest.raises(ValueError, match="followup_dispatch_head"):
            PostedRoundMetadata(
                flow="pr", role="coder", agent="Claude", round_number=2,
                subject=bad, followup_dispatch_head=bad,
            )
    valid = dict(decode_mapping(_encode_round_metadata(PostedRoundMetadata(
        flow="pr", role="coder", agent="Claude", round_number=2,
        subject="abc123", followup_dispatch_head="abc123",
    ))))
    assert _decode_round_metadata_mapping(valid).followup_dispatch_head == "abc123"
    for bad in (
        "", 7, "x" * 200, "bad sha", None, "unknown", "new-sha", "arbitrary.word",
        "ABC123", "abc", "a" * 65,
    ):
        with pytest.raises(AgentLoopError, match="followup_dispatch_head"):
            _decode_round_metadata_mapping({**valid, "followup_dispatch_head": bad})


def _m1034_parsed_followup():
    raw = structured_coder_followup(
        summary="Fixed both items.",
        addressed_items=["item-1"],
        remaining_items=["item-2"],
        addressed_item_notes={"item-1": "Qualified the network advice."},
        remaining_item_notes={"item-2": "Needs a design call."},
    )
    parsed = validate_structured_coder_followup(raw)
    assert parsed is not None
    return parsed


def test_m1034_renderer_frames_unchanged_head_follow_up_as_claims():
    from coding_review_agent_loop.comment_rendering import render_public_agent_comment

    parsed = _m1034_parsed_followup()
    baseline = _render_public_coder_followup_comment(parsed, agent="Claude")
    assert _render_public_coder_followup_comment(
        parsed, agent="Claude", head_unchanged_sha=None
    ) == baseline
    assert render_public_agent_comment(
        kind="coder_followup", parsed=parsed, agent="Claude"
    ) == baseline
    assert "### Addressed items" in baseline
    assert "Orchestrator note" not in baseline

    framed = render_public_agent_comment(
        kind="coder_followup", parsed=parsed, agent="Claude", head_unchanged_sha="abc123"
    )
    assert framed.startswith(
        "## Coder follow-up\n\n> Orchestrator note: the PR head was unchanged at `abc123` "
        "after this follow-up; no new commit is visible on the PR branch."
    )
    assert "no commit was made" not in framed
    assert "Coder summary (claim, unverified): Fixed both items." in framed
    assert "### Claimed already present at `abc123` (PR head unchanged)" in framed
    assert "### Addressed items" not in framed
    assert "  - Coder claim: Qualified the network advice." in framed
    assert "Resolution:" not in framed
    # Remaining items keep their labels; the coder's prose is framed, not rewritten.
    assert "### Remaining items" in framed
    assert "  - Reason: Needs a design call." in framed


def test_m1034_freeform_notice_prefixes_body():
    from coding_review_agent_loop.comment_rendering import (
        add_coder_followup_head_unchanged_notice,
    )

    framed = add_coder_followup_head_unchanged_notice("Fixed it.\n-- Claude", "abc123")
    assert framed.startswith("> Orchestrator note: the PR head was unchanged at `abc123`")
    assert framed.endswith("\n\nFixed it.\n-- Claude")


def test_m1034_reviewer_context_flags_unchanged_head_and_keeps_other_bytes():
    raw = structured_coder_followup(
        summary="Fixed both items.",
        addressed_items=["item-1"],
        addressed_item_notes={"item-1": "Qualified the network advice."},
    )
    legacy = PostedRoundMetadata(
        flow="pr", role="coder", agent="Claude", round_number=2, subject="abc123",
    )
    advanced = dataclasses.replace(legacy, followup_dispatch_head="0dd000")
    unchanged = dataclasses.replace(legacy, followup_dispatch_head="abc123")

    legacy_text = pr_loop_support._coder_followup_review_context(raw, legacy, head_sha="abc123")
    assert '"summary": "Fixed both items."' in legacy_text
    assert "pr_head_unchanged_during_followup" not in legacy_text
    # Changed-head records render exactly like historical records.
    assert pr_loop_support._coder_followup_review_context(raw, advanced, head_sha="abc123") == legacy_text

    framed = pr_loop_support._coder_followup_review_context(raw, unchanged, head_sha="abc123")
    assert '"pr_head_unchanged_during_followup": true' in framed
    assert '"followup_dispatch_head": "abc123"' in framed
    assert '"coder_summary_claim": "Fixed both items."' in framed
    assert '"claimed_addressed_items"' in framed
    assert '"claimed_addressed_item_notes"' in framed
    assert '"addressed_items"' not in framed
    assert '"summary"' not in framed
    assert _M1034_VERIFY_INSTRUCTION in framed


def test_m1034_unchanged_head_follow_up_is_logged_and_recorded_as_claims(tmp_path, capsys):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking", summary="Two fixes needed.",
                blocking_items=["Fix the README.", "Fix the docs."],
            ),
            structured_pr_review(
                state="approved", summary="Already present.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"},
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(
                summary="Fixed both items.",
                addressed_items=["item-1", "item-2"],
                addressed_item_notes={"item-1": "Qualified the network advice."},
            ),
        ],
        advance_pr_head_on_coder_followup=False,
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", quiet=False)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    err = capsys.readouterr().err
    assert "Round 1: Claude follow-up left PR head abc123 unchanged (1/2); re-reviewing" in err
    assert "pushed updates" not in err
    (coder_comment,) = _m1034_coder_comments(runner)
    assert "> Orchestrator note: the PR head was unchanged at `abc123`" in coder_comment
    assert "Coder summary (claim, unverified): Fixed both items." in coder_comment
    assert "### Claimed already present at `abc123` (PR head unchanged)" in coder_comment
    assert "  - Coder claim: Qualified the network advice." in coder_comment
    assert "### Addressed items" not in coder_comment
    metadata = _m1034_metadata(coder_comment)
    assert metadata.followup_dispatch_head == metadata.subject == "abc123"
    # The ledger meaning of addressed_items and the raw response are unchanged.
    raw = validate_structured_coder_followup(metadata.raw_structured_coder_response)
    assert raw is not None and tuple(raw.addressed_items) == ("item-1", "item-2")
    assert raw.summary == "Fixed both items."
    second_review = _m1034_reviewer_prompts(runner)[1]
    assert '"pr_head_unchanged_during_followup": true' in second_review
    assert '"coder_summary_claim": "Fixed both items."' in second_review
    assert _M1034_VERIFY_INSTRUCTION in second_review


def test_m1034_advanced_head_follow_up_keeps_legacy_framing(tmp_path, capsys):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking", summary="One fix needed.", blocking_items=["Fix the README."],
            ),
            structured_pr_review(
                state="approved", summary="Fixed.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(
                summary="Fixed the README.",
                addressed_items=["item-1"],
                addressed_item_notes={"item-1": "Rewrote the section."},
            ),
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", quiet=False)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    err = capsys.readouterr().err
    assert "Round 1: PR head advanced abc123..abc123-coder during Claude follow-up; re-reviewing" in err
    assert "pushed updates" not in err
    (coder_comment,) = _m1034_coder_comments(runner)
    assert "### Addressed items" in coder_comment
    assert "  - Resolution: Rewrote the section." in coder_comment
    assert "Orchestrator note" not in coder_comment
    assert "Claimed already present" not in coder_comment
    metadata = _m1034_metadata(coder_comment)
    assert metadata.followup_dispatch_head == "abc123"
    assert metadata.subject == "abc123-coder-1"
    second_review = _m1034_reviewer_prompts(runner)[1]
    assert "pr_head_unchanged_during_followup" not in second_review
    assert '"summary": "Fixed the README."' in second_review
    assert _M1034_VERIFY_INSTRUCTION not in second_review


def test_m1034_unchanged_head_limit_still_stops_without_push_log(tmp_path, capsys):
    blocking = structured_pr_review(
        state="blocking", summary="Re-plan the issue first.", blocking_items=["Re-plan first."]
    )
    runner = FakeRunner(
        codex_outputs=[
            blocking,
            structured_pr_review(
                state="blocking",
                summary="Still needs a re-plan.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "blocking", "note": "Re-plan first."}
                ],
            ),
            structured_pr_review(state="approved"),
        ],
        claude_outputs=[
            structured_coder_followup(remaining_items=["item-1"], addressed_items=[]),
            structured_coder_followup(remaining_items=["item-1"], addressed_items=[]),
        ],
        advance_pr_head_on_coder_followup=False,
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=6, quiet=False)
    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=config)
    assert "unchanged in 2 consecutive follow-up rounds" in str(excinfo.value)
    assert len(_m1034_reviewer_prompts(runner)) == 2
    err = capsys.readouterr().err
    assert "pushed updates for re-review" not in err
    assert "left PR head abc123 unchanged (1/2)" in err
    assert "unchanged (2/2)" not in err
    coder_comments = _m1034_coder_comments(runner)
    assert len(coder_comments) == 2
    for comment in coder_comments:
        assert "### Claimed already present at `abc123` (PR head unchanged)" in comment


def _m1034_recovery_runner(head_sha="c0ffee42"):
    old_item = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="Google Gemini",
        source_round=1,
        text="Preserve the metadata-backed unresolved item on rerun.",
        status="blocking",
    )
    old_coder_comment = _attach_round_metadata(
        "Opened the PR.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr", role="coder", agent="Claude", round_number=1,
            subject="old-sha", prior_items=(),
        ),
    )
    old_review_comment = _attach_round_metadata(
        "Blocked.\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini",
        PostedRoundMetadata(
            flow="pr", role="reviewer", agent="Gemini", round_number=1,
            subject="old-sha", prior_items=(), new_items=(old_item,), state="blocking",
        ),
    )
    return FakeRunner(
        claude_outputs=[
            structured_coder_followup(
                summary="Fixed the recovered item.",
                addressed_items=["item-2"],
                addressed_item_notes={"item-2": "The metadata-backed item is preserved."},
            )
        ],
        codex_outputs=[
            structured_pr_review(
                state="approved",
                summary="Recovered item is resolved.",
                prior_item_dispositions=[{"item_id": "item-2", "disposition": "resolved"}],
            )
        ],
        pr_payload={
            "headRefOid": head_sha,
            "comments": [
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": old_coder_comment},
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:01:00Z", "body": old_review_comment},
            ],
        },
        advance_pr_head_on_coder_followup=False,
    )


def test_m1034_recovery_round_without_commit_records_claims(tmp_path, capsys):
    runner = _m1034_recovery_runner()
    config = make_config(tmp_path, reviewer="codex", quiet=False)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    coder_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "Recovery context: the PR head `c0ffee42` was advanced by a commit" in coder_prompt
    assert "may or may not satisfy the recovered items" in coder_prompt
    assert "Check each recovered item against the current head." in coder_prompt
    assert "keep any unsatisfied item in remaining_items" in coder_prompt
    assert "already contains" not in coder_prompt
    # The full-board recovery review still follows the coder turn.
    assert command_index(runner.commands, ["claude"]) < command_index(
        runner.commands, ["codex", "exec"]
    )
    err = capsys.readouterr().err
    assert "Claude follow-up left PR head c0ffee42 unchanged (1/2)" in err
    assert "pushed updates" not in err
    (coder_comment,) = _m1034_coder_comments(runner)
    assert "> Orchestrator note: the PR head was unchanged at `c0ffee42`" in coder_comment
    assert "### Claimed already present at `c0ffee42` (PR head unchanged)" in coder_comment
    metadata = _m1034_metadata(coder_comment)
    assert metadata.followup_dispatch_head == metadata.subject == "c0ffee42"
    reviewer_prompt = _m1034_reviewer_prompts(runner)[0]
    assert '"pr_head_unchanged_during_followup": true' in reviewer_prompt
    assert _M1034_VERIFY_INSTRUCTION in reviewer_prompt


@pytest.mark.parametrize("head_sha", ["new-sha", "arbitrary.word"])
def test_m1034_non_sha_head_is_not_persisted_or_claimed_unchanged(tmp_path, head_sha):
    runner = _m1034_recovery_runner(head_sha=head_sha)
    config = make_config(tmp_path, reviewer="codex")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    (coder_comment,) = _m1034_coder_comments(runner)
    assert "Orchestrator note" not in coder_comment
    assert "Claimed already present" not in coder_comment
    assert "### Addressed items" in coder_comment
    metadata = _m1034_metadata(coder_comment)
    assert metadata.followup_dispatch_head is None
    reviewer_prompt = _m1034_reviewer_prompts(runner)[0]
    assert "pr_head_unchanged_during_followup" not in reviewer_prompt
    assert _M1034_VERIFY_INSTRUCTION not in reviewer_prompt


def test_m1034_non_recovery_follow_up_prompt_has_no_recovery_context(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(state="blocking", summary="Fix.", blocking_items=["Fix it."]),
            structured_pr_review(
                state="approved", summary="Ok.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
    )
    assert run_pr_loop(runner, pr_number=77, config=make_config(tmp_path, reviewer="codex")) == 0
    coder_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "Recovery context:" not in coder_prompt


def _m1034_resume_runner(*, dispatch_head):
    item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Fix the README.",
        status="blocking",
    )
    review_comment = _attach_round_metadata(
        structured_pr_review(state="blocking", summary="Fix it.", blocking_items=["Fix the README."]),
        PostedRoundMetadata(
            flow="pr", role="reviewer", agent="Codex", round_number=1,
            subject="abc123", prior_items=(), new_items=(item,), state="blocking",
        ),
    )
    raw = structured_coder_followup(
        summary="Fixed both items.",
        addressed_items=["item-1"],
        addressed_item_notes={"item-1": "Qualified the network advice."},
    )
    coder_comment = _attach_round_metadata(
        raw,
        PostedRoundMetadata(
            flow="pr", role="coder", agent="Claude", round_number=2,
            subject="abc123", prior_items=(item,),
            raw_structured_coder_response=raw,
            followup_dispatch_head=dispatch_head,
        ),
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="approved", summary="Verified.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            )
        ],
        pr_payload={"comments": [
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": review_comment},
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:01:00Z", "body": coder_comment},
        ]},
    )
    return runner, raw, coder_comment


def test_m1034_resume_rebuilds_unchanged_head_reviewer_context(tmp_path):
    runner, raw, coder_comment = _m1034_resume_runner(dispatch_head="abc123")
    assert run_pr_loop(runner, pr_number=77, config=make_config(tmp_path, reviewer="codex")) == 0
    reviewer_prompt = _m1034_reviewer_prompts(runner)[0]
    assert '"pr_head_unchanged_during_followup": true' in reviewer_prompt
    assert '"coder_summary_claim": "Fixed both items."' in reviewer_prompt
    assert '"claimed_addressed_item_notes"' in reviewer_prompt
    assert _M1034_VERIFY_INSTRUCTION in reviewer_prompt
    # The stored raw structured response is untouched.
    assert _m1034_metadata(coder_comment).raw_structured_coder_response == raw


def test_m1034_resume_from_historical_coder_record_keeps_legacy_context(tmp_path):
    runner, raw, _coder_comment = _m1034_resume_runner(dispatch_head=None)
    assert run_pr_loop(runner, pr_number=77, config=make_config(tmp_path, reviewer="codex")) == 0
    reviewer_prompt = _m1034_reviewer_prompts(runner)[0]
    legacy_metadata = PostedRoundMetadata(
        flow="pr", role="coder", agent="Claude", round_number=2,
        subject="abc123", raw_structured_coder_response=raw,
    )
    expected = pr_loop_support._coder_followup_review_context(raw, legacy_metadata, head_sha="abc123")
    assert "pr_head_unchanged_during_followup" not in expected
    assert expected.splitlines()[1] in reviewer_prompt
    assert '"summary": "Fixed both items."' in reviewer_prompt
    assert "pr_head_unchanged_during_followup" not in reviewer_prompt
    assert _M1034_VERIFY_INSTRUCTION not in reviewer_prompt


# --- Coder follow-up rejected after a push (#1292) ---------------------------

_ACTOR_1292 = ("agent-actor", 4242)
_LIVE_TARGET_FOLLOWUP = structured_coder_followup(
    summary="Added the test.",
    addressed_items=["item-1"],
    tests_run=["pytest tests/test_foo.py https://live.example"],
    reviewer="OpenAI Codex",
)
_VALID_FOLLOWUP = structured_coder_followup(
    summary="Added the test, in checkout.",
    addressed_items=["item-1"],
    tests_run=["pytest tests/test_foo.py"],
    reviewer="OpenAI Codex",
)


def _records_1292(runner, phase=None):
    records = []
    for comment in runner.pr_payload["comments"]:
        match = re.search(
            r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_:-]+)\s*-->", comment["body"]
        )
        if match is None:
            continue
        metadata = _decode_round_metadata(match.group("payload"))
        if phase is None or metadata.phase == phase:
            records.append(metadata)
    return records


def _blocking_review_1292(item="Add a regression test."):
    return structured_pr_review(
        state="blocking", summary="Needs a test.", blocking_items=[item],
        reviewer="Anthropic Claude",
    )


def _approving_review_1292():
    return structured_pr_review(
        state="approved", summary="Resolved.", reviewer="Anthropic Claude",
        prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
    )


def _rejected_push_runner_1292(*, coder_outputs=(_LIVE_TARGET_FOLLOWUP,), reviews=None, **kwargs):
    return FakeRunner(
        claude_outputs=list(reviews or [_blocking_review_1292(), _approving_review_1292()]),
        codex_outputs=list(coder_outputs),
        authenticated_actor=_ACTOR_1292,
        **kwargs,
    )


def _coder_prompts_1292(runner):
    return [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["codex"]]


def test_post_push_rejection_records_the_pushed_head_and_resumes_a_coder_recovery(tmp_path):
    runner = _rejected_push_runner_1292(coder_outputs=[_LIVE_TARGET_FOLLOWUP, _VALID_FOLLOWUP])
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError, match="live remote target"):
        run_pr_loop(runner, pr_number=77, config=config)

    pushed = runner.pr_payload["headRefOid"]
    assert pushed != "abc123"
    (dispatch,) = _records_1292(runner, "coder-dispatch")
    (rejection,) = _records_1292(runner, "coder-followup-rejected")
    assert (dispatch.subject, dispatch.dispatch_head, dispatch.dispatch_attempt) == ("abc123", "abc123", 1)
    assert dispatch.recovery_dispatch is False and dispatch.round_number == 2
    assert rejection.subject == pushed  # keyed to the pushed head ...
    assert rejection.dispatch_head == rejection.rejected_coder_followup_from_head == "abc123"
    assert rejection.round_number == rejection.dispatch_round == 1
    assert "live remote target" in rejection.rejected_coder_followup_reason
    assert [item.item_id for item in rejection.prior_items] == ["item-1"]
    assert rejection.recovery_round_budget.allowed_rounds == config.max_rounds
    assert not any(record.role == "coder" for record in _records_1292(runner) if record.subject == pushed)
    assert runner.pr_payload["headRefOid"] == pushed  # no reset or force-push

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    recovery_prompt = _coder_prompts_1292(runner)[1]
    assert "live remote target" in recovery_prompt
    assert "attempt 1" in recovery_prompt
    assert "`abc123`" in recovery_prompt and f"`{pushed}`" in recovery_prompt
    assert "Nothing from that attempt was reviewed or recorded" in recovery_prompt
    second_dispatch = _records_1292(runner, "coder-dispatch")[1]
    assert second_dispatch.dispatch_attempt == 2 and second_dispatch.recovery_dispatch is True
    assert second_dispatch.dispatch_round == 1  # the original slot
    assert second_dispatch.carried_rejection_reasons
    assert any(record.role == "coder" for record in _records_1292(runner))
    assert [cmd[:1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]  # reviewers ran


def test_unchanged_head_first_attempt_rejection_records_nothing_and_resume_is_unchanged(tmp_path):
    runner = _rejected_push_runner_1292(advance_pr_head_on_coder_followup=False)
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError, match="live remote target"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert _records_1292(runner, "coder-followup-rejected") == []
    (dispatch,) = _records_1292(runner, "coder-dispatch")
    assert dispatch.dispatch_attempt == 1 and dispatch.recovery_dispatch is False
    resumed = _resume_pr_round(
        runner.pr_payload["comments"] and [
            IssueComment(
                author=_ACTOR_1292[0], created_at=None, body=comment["body"], author_id=_ACTOR_1292[1]
            )
            for comment in runner.pr_payload["comments"]
        ],
        head_sha="abc123", configured_reviewers=("claude",),
        trusted_actor=_ACTOR_1292,
    )
    # The first-attempt dispatch is ignored: resume is the ordinary checkpoint-free round.
    assert resumed is None or resumed.rejected_coder_followup_reason is None


def test_rejection_record_ledger_is_the_pre_dispatch_snapshot(tmp_path):
    disputing = structured_coder_followup(
        summary="Disputed.", disputed_items=["item-1"], dispute_evidence={"item-1": "x"},
        tests_run=["pytest tests/test_foo.py https://live.example"], reviewer="OpenAI Codex",
    )
    runner = _rejected_push_runner_1292(coder_outputs=[disputing])
    config = make_config(tmp_path, coder="codex", reviewer="claude")
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=config)
    (rejection,) = _records_1292(runner, "coder-followup-rejected")
    assert [(item.item_id, item.status, item.text) for item in rejection.prior_items] == [
        ("item-1", "blocking", "Add a regression test.")
    ]


def test_dispatch_record_failure_stops_before_the_coder_runs(tmp_path):
    runner = _rejected_push_runner_1292()
    original = runner.run

    def failing_run(cmd, *args, **kwargs):
        if cmd[:3] == ["gh", "pr", "comment"]:
            body_path = Path(cmd[cmd.index("--body-file") + 1]) if "--body-file" in cmd else None
            text = body_path.read_text(encoding="utf-8") if body_path else "\n".join(cmd)
            if "coder dispatch record" in text:
                raise AgentLoopError("cannot post")
        return original(cmd, *args, **kwargs)

    runner.run = failing_run
    config = make_config(tmp_path, coder="codex", reviewer="claude")
    with pytest.raises(AgentLoopError, match="cannot post"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert _coder_prompts_1292(runner) == []


def test_rejection_record_post_failure_keeps_the_original_error_and_the_attempt_limit(tmp_path):
    runner = _rejected_push_runner_1292(
        coder_outputs=[_LIVE_TARGET_FOLLOWUP, _LIVE_TARGET_FOLLOWUP, _VALID_FOLLOWUP],
        reviews=[_blocking_review_1292(), _approving_review_1292()],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")
    with pytest.raises(AgentLoopError, match="live remote target"):
        run_pr_loop(runner, pr_number=77, config=config)
    original = runner.run

    def failing_run(cmd, *args, **kwargs):
        if cmd[:3] == ["gh", "pr", "comment"]:
            body_path = Path(cmd[cmd.index("--body-file") + 1]) if "--body-file" in cmd else None
            text = body_path.read_text(encoding="utf-8") if body_path else "\n".join(cmd)
            if "rejected coder follow-up record" in text:
                raise RuntimeError("rejection post failed")
        return original(cmd, *args, **kwargs)

    runner.run = failing_run
    # The attempt-2 rejection cannot be recorded; the original error still surfaces.
    with pytest.raises(AgentLoopError, match="live remote target"):
        run_pr_loop(runner, pr_number=77, config=config)
    runner.run = original
    assert len(_records_1292(runner, "coder-followup-rejected")) == 1
    # The attempt-2 dispatch record still carries the attempt: resume stops at the limit.
    with pytest.raises(AgentLoopError, match="--review-unrecorded-head") as raised:
        run_pr_loop(runner, pr_number=77, config=config)
    assert "consecutive" in str(raised.value)
    assert len(_coder_prompts_1292(runner)) == 2  # no third coder dispatch


_A_1292, _B_1292, _C_1292 = "a" * 40, "b" * 40, "c" * 40
from coding_review_agent_loop.protocol import UnresolvedReviewItem  # noqa: E402
from coding_review_agent_loop.round_state import RecoveryRoundBudget as _Budget1292  # noqa: E402


def _item_1292(number=1, text="Add a regression test."):
    return UnresolvedReviewItem(
        item_id=f"item-{number}", reviewer="Anthropic Claude", source_round=1, text=text,
        status="blocking",
    )


def _meta_1292(role, phase="authoritative", *, subject=_A_1292, round_number=1, items=(),
               new_items=(), **extra):
    # Reviewer records are by the configured reviewer (Codex): a historical
    # reviewer missing from the board would be a recorded board change (#1376).
    agent = {"summary": "Orchestrator", "reviewer": "Codex"}.get(role, "Claude")
    return PostedRoundMetadata(
        flow="pr", role=role, agent=agent,
        round_number=round_number, subject=subject, prior_items=tuple(items),
        new_items=tuple(new_items), state="blocking", phase=phase, **extra,
    )


def _seed_1292(*records, actor=_ACTOR_1292, forged=()):
    comments = []
    for index, metadata in enumerate(records):
        who = ("mallory", 9) if index in forged else actor
        comments.append(
            {
                "author": {"login": who[0]},
                "_rest_author_id": who[1],
                "createdAt": f"2026-05-20T09:{index:02d}:00Z",
                "body": _attach_round_metadata("record\n-- Orchestrator", metadata),
            }
        )
    return comments


def _history_1292(reviewer="Codex"):
    """A round-1 coder and a reviewer round on head A that raised item-1.

    ``reviewer`` is the configured reviewer: a historical reviewer missing
    from the configured board would be a recorded board change (#1376).
    """
    return [
        _meta_1292("coder"),
        dataclasses.replace(_meta_1292("reviewer", new_items=(_item_1292(),)), agent=reviewer),
    ]


def _dispatch_1292(subject=_A_1292, attempt=1, budget=None, **extra):
    values = dict(
        dispatch_round=1, dispatch_head=subject, dispatch_attempt=attempt,
        recovery_dispatch=attempt > 1, recovery_round_budget=budget or _Budget1292(5, False, False),
    )
    values.update(extra)
    return _meta_1292(
        "summary", "coder-dispatch", subject=subject, round_number=2, items=(_item_1292(),),
        **values,
    )


def _rejection_1292(subject=_B_1292, attempt=1, budget=None, reason="live remote target", **extra):
    values = dict(
        dispatch_round=1, dispatch_head=_A_1292, dispatch_attempt=attempt,
        recovery_dispatch=attempt > 1, rejected_coder_followup_reason=reason,
        rejected_coder_followup_from_head=_A_1292,
        recovery_round_budget=budget or _Budget1292(5, False, False),
    )
    values.update(extra)
    return _meta_1292(
        "summary", "coder-followup-rejected", subject=subject, round_number=1,
        items=(_item_1292(),), **values,
    )


def _handoff_1292(subject=_B_1292, round_number=1, source="operator", budget=None):
    return _meta_1292(
        "summary", "head-review-recovery", subject=subject, round_number=round_number,
        items=(_item_1292(),), head_review_recovery_source=source,
        recovery_round_budget=budget or _Budget1292(5, False, False),
    )


def _checkpoint_1292(subject=_A_1292, lifecycle="repair_required"):
    return _meta_1292(
        "summary", "qualification-checkpoint", subject=subject, round_number=2,
        items=(_item_1292(),),
        qualification_checkpoint=QualificationCheckpoint(
            obligation_kind="managed-exact-head-ci", obligation_identity="ci",
            lifecycle=lifecycle, failed_head_sha=_A_1292,
            candidate_head_sha=_B_1292 if lifecycle == "awaiting_current_head_review" else None,
            allowed_rounds=5, watch_failure_extension_used=False, watch_head_extension_used=False,
        ),
    )


def _stranded_runner_1292(records, head, *, forged=(), **kwargs):
    outputs = kwargs.pop("codex_outputs", None)
    return FakeRunner(
        codex_outputs=outputs if outputs is not None else [
            structured_pr_review(
                state="approved", summary="Resolved.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            )
        ],
        claude_outputs=kwargs.pop("claude_outputs", []),
        pr_payload={"headRefOid": head, "comments": _seed_1292(*records, forged=forged)},
        authenticated_actor=_ACTOR_1292,
        advance_pr_head_on_coder_followup=False,
        **kwargs,
    )


def _agent_order_1292(runner):
    return [cmd[0] for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"])]


@pytest.mark.parametrize("lifecycle", ["repair_required", "awaiting_current_head_review"])
def test_stranded_checkpoint_history_gets_an_ordinary_review_round_of_the_new_head(tmp_path, lifecycle):
    """Row legacy-checkpoint-stranded (the llm-dialectic #1370 incident shape)."""
    runner = _stranded_runner_1292([*_history_1292(), _checkpoint_1292(lifecycle=lifecycle)], _B_1292)
    config = make_config(tmp_path, reviewer="codex", coder="claude")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert _agent_order_1292(runner)[0] == "codex"  # never routed through a coder first
    assert "claude" not in _agent_order_1292(runner)
    (handoff,) = _records_1292(runner, "head-review-recovery")
    assert handoff.head_review_recovery_source == "legacy-checkpoint"
    assert handoff.round_number == 2 and handoff.subject == _B_1292  # never rewound
    assert [item.item_id for item in handoff.prior_items] == ["item-1"]
    assert handoff.recovery_round_budget == _Budget1292(5, False, False)
    assert not any(
        record.role == "coder" for record in _records_1292(runner) if record.subject == _B_1292
    )
    reviewer_prompt = runner.commands[command_index(runner.commands, ["codex", "exec"])][0][-1]
    assert "[item-1]" in reviewer_prompt


def test_stranded_checkpoint_history_with_a_forged_anchor_still_refuses(tmp_path):
    runner = _stranded_runner_1292(
        [*_history_1292(), _checkpoint_1292()], _B_1292, forged=(2,)
    )
    config = make_config(tmp_path, reviewer="codex", coder="claude")
    with pytest.raises(AgentLoopError, match="--review-unrecorded-head") as raised:
        run_pr_loop(runner, pr_number=77, config=config)
    assert "Do not hand-author" in str(raised.value)
    assert _agent_order_1292(runner) == []
    assert _records_1292(runner, "head-review-recovery") == []


def test_operator_flag_reviews_the_current_head_after_the_attempt_limit(tmp_path):
    """Rows operator-flag-recovery and rejection-retry-limit."""
    records = [
        *_history_1292(), _dispatch_1292(), _rejection_1292(),
        _dispatch_1292(subject=_B_1292, attempt=2),
        _rejection_1292(
            attempt=2, subject=_B_1292, reason="again",
            carried_rejection_reasons=("live remote target",),
        ),
    ]
    runner = _stranded_runner_1292(records, _B_1292)
    config = make_config(tmp_path, reviewer="codex", coder="claude")
    with pytest.raises(AgentLoopError, match="--review-unrecorded-head") as raised:
        run_pr_loop(runner, pr_number=77, config=config)
    assert "live remote target" in str(raised.value) and "again" in str(raised.value)
    assert _agent_order_1292(runner) == []

    flagged = make_config(tmp_path, reviewer="codex", coder="claude", review_unrecorded_head=True)
    assert run_pr_loop(runner, pr_number=77, config=flagged) == 0
    assert _agent_order_1292(runner) == ["codex"]
    (handoff,) = _records_1292(runner, "head-review-recovery")
    assert handoff.head_review_recovery_source == "operator"
    assert handoff.recovery_round_budget == _Budget1292(5, False, False)
    assert not any(record.role == "coder" and record.subject == _B_1292 for record in _records_1292(runner))


def test_head_review_handoff_interruption_resumes_without_the_flag_and_posts_no_second_record(tmp_path):
    """Row operator-record-interrupted (i)/(ii)."""
    budget = _Budget1292(6, True, False)
    records = [
        *_history_1292(), _handoff_1292(round_number=5, budget=budget),
        _meta_1292("summary", "scheduler-prelaunch", subject=_B_1292, round_number=5, items=(_item_1292(),)),
    ]
    runner = _stranded_runner_1292(records, _B_1292)
    config = make_config(tmp_path, reviewer="codex", coder="claude", max_rounds=5)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert _agent_order_1292(runner) == ["codex"]
    assert len(_records_1292(runner, "head-review-recovery")) == 1  # the original handoff
    reconciled = [
        record for record in _records_1292(runner)
        if record.role == "reviewer" and record.subject == _B_1292
    ]
    assert reconciled and all(record.round_number == 5 for record in reconciled)


def test_head_review_handoff_survives_a_further_head_advance(tmp_path):
    """Row resume-after-rejection variant (ii)."""
    budget = _Budget1292(5, False, False)
    records = [*_history_1292(), _handoff_1292(round_number=2, budget=budget)]
    runner = _stranded_runner_1292(records, _C_1292)
    config = make_config(tmp_path, reviewer="codex", coder="claude")
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    new_handoffs = [r for r in _records_1292(runner, "head-review-recovery") if r.subject == _C_1292]
    assert len(new_handoffs) == 1 and new_handoffs[0].round_number == 2
    assert [item.item_id for item in new_handoffs[0].prior_items] == ["item-1"]
    assert _agent_order_1292(runner) == ["codex"]


def test_rejection_survives_a_further_head_advance_and_keeps_its_attempt(tmp_path):
    """Row resume-after-rejection variant (i)."""
    records = [*_history_1292("Claude"), _dispatch_1292(), _rejection_1292()]
    runner = FakeRunner(
        claude_outputs=[_approving_review_1292()],
        codex_outputs=[_LIVE_TARGET_FOLLOWUP],
        pr_payload={"headRefOid": _C_1292, "comments": _seed_1292(*records)},
        authenticated_actor=_ACTOR_1292,
        advance_pr_head_on_coder_followup=False,
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")
    with pytest.raises(AgentLoopError, match="live remote target"):
        run_pr_loop(runner, pr_number=77, config=config)
    prompt = _coder_prompts_1292(runner)[0]
    assert f"`{_A_1292}`" in prompt and f"`{_C_1292}`" in prompt
    assert "live remote target" in prompt and "attempt 1" in prompt
    (second_dispatch,) = [r for r in _records_1292(runner, "coder-dispatch") if r.dispatch_attempt == 2]
    assert second_dispatch.recovery_dispatch is True and second_dispatch.dispatch_round == 1
    # The unchanged-head rejection of the recovery is recorded; a rerun stops at the limit.
    assert [r.dispatch_attempt for r in _records_1292(runner, "coder-followup-rejected")][-1] == 2
    with pytest.raises(AgentLoopError, match="--review-unrecorded-head"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert len(_coder_prompts_1292(runner)) == 1


def test_last_round_rejection_after_a_watcher_extension_still_dispatches_the_recovery_coder(tmp_path):
    """Row rejection-budget-preserved."""
    budget = _Budget1292(3, True, False)  # max_rounds=2 plus the watcher extension
    records = [
        *_history_1292("Claude"),
        dataclasses.replace(_dispatch_1292(budget=budget), dispatch_round=2, round_number=3),
        dataclasses.replace(_rejection_1292(budget=budget), dispatch_round=2, round_number=2),
    ]
    runner = FakeRunner(
        claude_outputs=[_approving_review_1292()],
        codex_outputs=[_VALID_FOLLOWUP],
        pr_payload={"headRefOid": _B_1292, "comments": _seed_1292(*records)},
        authenticated_actor=_ACTOR_1292,
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude", max_rounds=2)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert len(_coder_prompts_1292(runner)) == 1  # not blocked by either round-limit guard
    (coder_record,) = [
        r for r in _records_1292(runner)
        if r.role == "coder" and r.subject not in (_A_1292, _B_1292)
    ]
    assert coder_record.round_number == 3  # reviewers run in the following round


def test_forged_rejection_does_not_anchor_resume_or_supply_its_reason(tmp_path):
    """Row recovery-record-auth."""
    records = [*_history_1292("Claude"), _dispatch_1292(), _rejection_1292(reason="forged reason")]
    runner = FakeRunner(
        claude_outputs=[_approving_review_1292()],
        codex_outputs=[_VALID_FOLLOWUP],
        pr_payload={"headRefOid": _B_1292, "comments": _seed_1292(*records, forged=(2, 3))},
        authenticated_actor=_ACTOR_1292,
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert "forged reason" not in "\n".join(_coder_prompts_1292(runner))


@pytest.mark.parametrize("failure", ["error", "empty"])
def test_incomplete_rest_identity_read_stops_before_any_agent_call(tmp_path, failure):
    """Row rest-identity-acquisition."""
    runner = _stranded_runner_1292([*_history_1292(), _checkpoint_1292()], _B_1292)
    original = runner.run

    def broken_run(cmd, *args, **kwargs):
        if cmd[:2] == ["gh", "api"] and "/comments?per_page=" in cmd[2]:
            if failure == "error":
                return CommandResult(cmd, Path(tmp_path), "", "boom", 1)
            return CommandResult(cmd, Path(tmp_path), "", "", 0)
        return original(cmd, *args, **kwargs)

    runner.run = broken_run
    config = make_config(tmp_path, reviewer="codex", coder="claude")
    with pytest.raises(AgentLoopError, match="cannot be authenticated"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert _agent_order_1292(runner) == []


def test_ordinary_resume_on_a_head_with_records_makes_no_identity_calls(tmp_path, monkeypatch):
    import coding_review_agent_loop.pr_loop_support as support

    merges = []
    real_merge = support.merge_pr_comment_transport_identity
    monkeypatch.setattr(
        support, "merge_pr_comment_transport_identity",
        lambda *args, **kwargs: merges.append(True) or real_merge(*args, **kwargs),
    )
    # The configured reviewer's own review is still to come on head A.
    runner = _stranded_runner_1292(
        [_meta_1292("coder"), _meta_1292("summary", "provisional", items=(_item_1292(),))],
        _A_1292,
    )
    config = make_config(tmp_path, reviewer="codex", coder="claude")
    run_pr_loop(runner, pr_number=77, config=config)
    assert merges == []
    assert not any(
        cmd[:2] == ["gh", "api"] and "/comments?per_page=" in cmd[2] and "&since=" not in cmd[2]
        for cmd, _cwd in runner.commands
    )

    # A head with no records of its own needs admission, so identities are read.
    advanced = _stranded_runner_1292(
        _history_1292(), _B_1292,
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            structured_pr_review(
                state="approved", summary="Resolved.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            )
        ],
    )
    run_pr_loop(advanced, pr_number=77, config=config)
    assert merges == [True]


@pytest.mark.parametrize(
    "malformed, field",
    [
        (lambda: _rejection_1292(dispatch_attempt=None), "dispatch_attempt"),
        (lambda: _rejection_1292(dispatch_round=None), "dispatch_round"),
        (lambda: _dispatch_1292(subject=_B_1292, attempt=2, recovery_dispatch=False), "recovery_dispatch"),
    ],
)
def test_malformed_live_recovery_record_stops_before_any_agent_call_and_the_flag_recovers(
    tmp_path, malformed, field
):
    """Row metadata-compat on the production resume path."""
    record = malformed()
    head = record.subject
    runner = _stranded_runner_1292([*_history_1292(), record], head)
    config = make_config(tmp_path, reviewer="codex", coder="claude")
    with pytest.raises(AgentLoopError, match=rf"comment index 2 .*malformed.*{field}"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert _agent_order_1292(runner) == []

    flagged = make_config(tmp_path, reviewer="codex", coder="claude", review_unrecorded_head=True)
    assert run_pr_loop(runner, pr_number=77, config=flagged) == 0
    (handoff,) = _records_1292(runner, "head-review-recovery")
    assert handoff.recovery_round_budget.allowed_rounds == flagged.max_rounds  # default budget
    # The posted handoff retires the malformed record: a later restart is not blocked.
    again = _stranded_runner_1292([], head)
    again.pr_payload["comments"] = runner.pr_payload["comments"]
    resumed = _resume_pr_round(
        [
            IssueComment(author=_ACTOR_1292[0], created_at=None, body=c["body"], author_id=_ACTOR_1292[1])
            for c in runner.pr_payload["comments"]
        ],
        head_sha=head, configured_reviewers=("codex",), trusted_actor=_ACTOR_1292,
    )
    assert resumed is None or resumed.rejected_coder_followup_reason is None


def test_recovery_rejected_without_a_push_then_operator_recovery_and_interruption(tmp_path):
    """Rows recovery-rejected-no-push and rejection-superseded (the full chain)."""
    runner = _rejected_push_runner_1292(
        coder_outputs=[_LIVE_TARGET_FOLLOWUP, _LIVE_TARGET_FOLLOWUP],
        reviews=[_blocking_review_1292()],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")
    with pytest.raises(AgentLoopError, match="live remote target"):
        run_pr_loop(runner, pr_number=77, config=config)
    pushed = runner.pr_payload["headRefOid"]

    runner.advance_pr_head_on_coder_followup = False  # the recovery coder does not push
    with pytest.raises(AgentLoopError, match="live remote target"):
        run_pr_loop(runner, pr_number=77, config=config)
    second = _records_1292(runner, "coder-followup-rejected")[-1]
    assert second.subject == second.dispatch_head == second.rejected_coder_followup_from_head == pushed
    assert second.dispatch_attempt == 2 and second.recovery_dispatch is True

    with pytest.raises(AgentLoopError, match="--review-unrecorded-head") as limit:
        run_pr_loop(runner, pr_number=77, config=config)
    assert "consecutive" in str(limit.value)
    assert len(_coder_prompts_1292(runner)) == 2  # the limit stops a third dispatch

    flagged = make_config(tmp_path, coder="codex", reviewer="claude", review_unrecorded_head=True)
    runner.claude_outputs.clear()  # interrupt right after the handoff: no reviewer output
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=flagged)
    (handoff,) = _records_1292(runner, "head-review-recovery")
    assert handoff.subject == pushed and handoff.head_review_recovery_source == "operator"

    runner.claude_outputs.append(_approving_review_1292())
    assert run_pr_loop(runner, pr_number=77, config=config) == 0  # no flag needed
    assert len(_records_1292(runner, "head-review-recovery")) == 1  # resumed from the handoff
    assert len(_coder_prompts_1292(runner)) == 2  # no further coder dispatch


def test_review_unrecorded_head_flag_parses_on_pr_only_and_defaults_off(tmp_path):
    from coding_review_agent_loop.cli import build_parser
    from coding_review_agent_loop.config import config_from_args

    parser = build_parser()
    base = ["pr", "123", "--repo", "OWNER/REPO", "--codex-dir", str(tmp_path / "codex")]
    assert config_from_args(parser.parse_args(base), FakeRunner()).review_unrecorded_head is False
    flagged = parser.parse_args([*base, "--review-unrecorded-head"])
    assert config_from_args(flagged, FakeRunner()).review_unrecorded_head is True
    with pytest.raises(SystemExit):
        parser.parse_args(["issue", "5", "--repo", "OWNER/REPO", "--review-unrecorded-head"])


# --- Review feedback on #1296: authenticated budgets, numbering, matrix paths --


def _rec_1292(items=None):
    return ReviewItemDisposition(item_id="item-1", reviewer="Codex", disposition="resolved")


from coding_review_agent_loop.protocol import ReviewItemDisposition  # noqa: E402


@pytest.mark.parametrize("source", ["legacy-checkpoint", "operator"])
def test_head_review_recovery_under_primary_then_panel_gates_the_panel_and_bounds_the_coder(
    tmp_path, monkeypatch, source
):
    """Rows legacy-checkpoint-stranded / operator-flag-recovery: ordinary primary gate, no override."""
    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    if source == "legacy-checkpoint":
        records = [*_history_1292(), _checkpoint_1292()]
        flag = False
    else:
        records = [
            *_history_1292(), _dispatch_1292(), _rejection_1292(attempt=2, subject=_B_1292,
                                                                carried_rejection_reasons=("a",), reason="b"),
        ]
        flag = True
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(
                reviewer="OpenAI Codex", state="blocking",
                blocking_items=[{"text": "gap", "fix_scope": ["src/x.py"]}],
                dispositions=[{"item_id": "item-1", "disposition": "blocking", "note": "The regression test is still missing on src/x.py; add it with evidence."}],
            )
        ],
        pr_payload={"headRefOid": _B_1292, "comments": _seed_1292(*records)},
        authenticated_actor=_ACTOR_1292,
    )
    config = _staged_config(tmp_path, review_unrecorded_head=flag, coder="claude")
    with pytest.raises(Exception):  # scripted outputs run out after the coder turn
        run_pr_loop(runner, pr_number=77, config=config)
    sequence = _agent_sequence(runner)
    assert sequence[:2] == ["codex", "claude"]  # primary first, panel gated, then the coder
    assert "gemini" not in sequence[:2] and "agy" not in sequence[:2]
    (handoff,) = _records_1292(runner, "head-review-recovery")
    assert handoff.head_review_recovery_source == source
    coder_dispatches = [
        r for r in _records_1292(runner, "coder-dispatch") if r.dispatch_head == _B_1292
    ]
    assert coder_dispatches and coder_dispatches[-1].dispatch_attempt == 1
    assert coder_dispatches[-1].round_number > handoff.round_number  # after the handoff


def test_legacy_recovery_on_a_confirmed_merge_conflict_uses_the_standard_conflict_route(tmp_path):
    records = [*_history_1292(), _checkpoint_1292()]
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[_approving_review_1292().replace("Anthropic Claude", "OpenAI Codex")],
        pr_payload={
            "headRefOid": _B_1292, "comments": _seed_1292(*records),
            "mergeable": "CONFLICTING", "mergeStateStatus": "DIRTY", "baseRefName": "main",
        },
        authenticated_actor=_ACTOR_1292,
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex")
    with pytest.raises(Exception):
        run_pr_loop(runner, pr_number=77, config=config)
    assert _agent_order_1292(runner)[0] == "claude"  # conflict route, reviewers skipped
    assert len(_records_1292(runner, "head-review-recovery")) == 1
    dispatch = [r for r in _records_1292(runner, "coder-dispatch") if r.dispatch_head == _B_1292]
    assert dispatch and dispatch[0].dispatch_attempt == 1


def test_partial_publication_after_a_handoff_supersedes_the_stale_repair_checkpoint(tmp_path):
    """Row operator-record-interrupted variant (iv), through run_pr_loop."""
    budget = _Budget1292(6, True, False)
    stale = dataclasses.replace(_checkpoint_1292(subject=_B_1292), round_number=6)
    reviewed = dataclasses.replace(
        _meta_1292(
            "reviewer", subject=_B_1292, round_number=6, items=(_item_1292(),),
            dispositions=(_rec_1292(),),
        ),
        agent="Codex", state="approved",
        canonical_reviewer_response=structured_pr_review(
            state="approved", summary="Resolved.",
            prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        ),
    )
    records = [
        *_history_1292(), stale,
        _handoff_1292(round_number=6, budget=budget), reviewed,
    ]
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="approved", summary="Resolved.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            )
        ],
        gemini_outputs=[
            structured_pr_review(
                state="approved", summary="Resolved.", reviewer="Google Gemini",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            )
        ],
        pr_payload={"headRefOid": _B_1292, "comments": _seed_1292(*records)},
        authenticated_actor=_ACTOR_1292,
        advance_pr_head_on_coder_followup=False,
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), coder="claude", max_rounds=5)
    # Round 6 exceeds max_rounds: only the handoff's extended budget admits it.
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert "claude" not in _agent_sequence(runner)  # the stale repair skip did not return
    assert "gemini" in _agent_sequence(runner)
    assert len(_records_1292(runner, "head-review-recovery")) == 1


def test_operator_recovery_above_max_rounds_needs_an_authenticated_budget_source(tmp_path):
    """Row operator-flag-recovery: no rounds are granted without an admitted source."""
    checkpoint = dataclasses.replace(
        _checkpoint_1292(),
        round_number=3,
        qualification_checkpoint=QualificationCheckpoint(
            obligation_kind="managed-exact-head-ci", obligation_identity="ci",
            lifecycle="repair_required", failed_head_sha=_A_1292, candidate_head_sha=None,
            allowed_rounds=3, watch_failure_extension_used=True, watch_head_extension_used=False,
        ),
    )
    prelaunch = _meta_1292(
        "summary", "scheduler-prelaunch", round_number=3, items=(_item_1292(),)
    )
    records = [*_history_1292(), checkpoint, prelaunch]
    config = make_config(
        tmp_path, reviewer="codex", coder="claude", max_rounds=2, review_unrecorded_head=True
    )
    # Forged checkpoint (index 2): no admitted source, so the override refuses.
    forged = _stranded_runner_1292(records, _B_1292, forged=(2,))
    with pytest.raises(AgentLoopError):
        run_pr_loop(forged, pr_number=77, config=config)
    assert _agent_order_1292(forged) == []
    assert _records_1292(forged, "head-review-recovery") == []
    # Authenticated checkpoint: its used-extension budget is restored and carried.
    genuine = _stranded_runner_1292(records, _B_1292)
    assert run_pr_loop(genuine, pr_number=77, config=config) == 0
    (handoff,) = _records_1292(genuine, "head-review-recovery")
    assert handoff.round_number == 3
    assert handoff.recovery_round_budget == _Budget1292(3, True, False)


def test_rejection_after_the_live_ledger_was_rewritten_persists_the_pre_dispatch_snapshot(
    tmp_path, monkeypatch
):
    """Row rejection-ledger-snapshot: the failure comes after _apply_dispute_evidence ran."""
    import coding_review_agent_loop.pr_loop as pr_loop_module

    seen = []
    real_apply = pr_loop_module._apply_dispute_evidence

    def recording_apply(items, **kwargs):
        result = real_apply(items, **kwargs)
        seen.append((tuple(items), tuple(result)))
        return result

    def failing_derive(*args, **kwargs):
        raise AgentLoopError("derived evidence rejected")

    monkeypatch.setattr(pr_loop_module, "_apply_dispute_evidence", recording_apply)
    monkeypatch.setattr(pr_loop_module, "_derive_authenticated_risk_evidence_for_coder", failing_derive)
    disputing = structured_coder_followup(
        summary="Disputed.", disputed_items=["item-1"],
        dispute_evidence={"item-1": "The suite already covers it."}, reviewer="OpenAI Codex",
    )
    runner = _rejected_push_runner_1292(coder_outputs=[disputing])
    config = make_config(tmp_path, coder="codex", reviewer="claude")
    with pytest.raises(AgentLoopError, match="derived evidence rejected"):
        run_pr_loop(runner, pr_number=77, config=config)
    before, after = seen[0]
    assert before != after  # the live ledger really was rewritten by the rejected turn
    (rejection,) = _records_1292(runner, "coder-followup-rejected")
    assert tuple(rejection.prior_items) == before


def test_watcher_extension_survives_a_post_push_rejection_and_is_never_granted_twice(
    tmp_path, monkeypatch
):
    """Row rejection-budget-preserved, through the real watcher on the final round."""
    failed_check = PullRequestCheck(
        name="final-ci/exact-head", kind="check_run", status="failure",
        url="https://github.com/OWNER/REPO/actions/runs/555",
    )
    failed_checks = _watch_check_board("failing", failing=(failed_check,))
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(state="approved", summary="Approved."),
            structured_pr_review(
                state="approved", summary="Approved after the CI fix.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(
                summary="Fixed CI.", addressed_items=["item-1"],
                tests_run=["pytest tests/test_foo.py https://live.example"],
            ),
            structured_coder_followup(summary="Fixed CI in checkout.", addressed_items=["item-1"]),
        ],
        authenticated_actor=_ACTOR_1292,
    )
    config = make_config(tmp_path, auto_merge=True, max_rounds=1)
    qualified_heads = []

    def watcher(*args, **kwargs):
        head = runner.pr_payload["headRefOid"]  # the actual current head at each observation
        qualified_heads.append(head)
        return ManagedCiOutcome(status="failed", checks=failed_checks, head_sha=head)

    monkeypatch.setattr(
        pr_loop, "activate_managed_ci", lambda *args, **kwargs: ManagedCiContract()
    )
    monkeypatch.setattr(pr_loop, "dispatch_final_qualification", lambda *args, **kwargs: None)
    monkeypatch.setattr(pr_loop, "wait_for_final_qualification", watcher)
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *args, **kwargs: None)

    # Round 1 (the final round): the watcher grants its one extension, then the
    # CI-repair coder pushes and is rejected.
    with pytest.raises(AgentLoopError, match="live remote target"):
        run_pr_loop(runner, pr_number=77, config=config)
    pushed = runner.pr_payload["headRefOid"]
    assert qualified_heads == ["abc123"]
    (first_dispatch,) = _records_1292(runner, "coder-dispatch")
    assert first_dispatch.recovery_round_budget == _Budget1292(2, True, False)
    (rejection,) = _records_1292(runner, "coder-followup-rejected")
    assert rejection.recovery_round_budget == _Budget1292(2, True, False)

    # The restart restores that budget: the recovery coder is dispatched in the
    # original slot, its record lands one round later, the reviewers publish for
    # the new head, and the next watcher failure is refused a second extension.
    with pytest.raises(AgentLoopError, match=r"blocking issues after round 2") as refused:
        run_pr_loop(runner, pr_number=77, config=config)
    recovered_head = runner.pr_payload["headRefOid"]
    assert recovered_head != pushed
    assert qualified_heads == ["abc123", pushed, recovered_head]  # the third failure was observed
    second_dispatch = _records_1292(runner, "coder-dispatch")[1]
    assert second_dispatch.dispatch_attempt == 2 and second_dispatch.dispatch_round == 1
    assert second_dispatch.recovery_round_budget == _Budget1292(2, True, False)
    (coder_record,) = [
        r for r in _records_1292(runner) if r.role == "coder" and r.subject == recovered_head
    ]
    assert coder_record.round_number == 2
    reviews = [
        r for r in _records_1292(runner)
        if r.role == "reviewer" and r.subject == recovered_head and r.round_number == 2
    ]
    assert reviews  # the following reviewer round published for the recovered head
    assert "managed-exact-head-ci" in str(refused.value)  # refused: no second extension
    assert len([c for c, _ in runner.commands if c[:1] == ["claude"]]) == 2  # no third coder
    assert len(_records_1292(runner, "coder-dispatch")) == 2  # no second extension was granted


def test_incomplete_rest_page_that_omits_a_recovery_record_stops_before_any_agent_call(tmp_path):
    """Row rest-identity-acquisition: a valid page that lacks the handoff is incomplete."""
    records = [
        *_history_1292(), _dispatch_1292(),
        _rejection_1292(attempt=2, subject=_B_1292, reason="again",
                        carried_rejection_reasons=("first",)),
    ]
    runner = _stranded_runner_1292(records, _B_1292)
    original = runner.run

    def omitting_run(cmd, *args, **kwargs):
        result = original(cmd, *args, **kwargs)
        if cmd[:2] == ["gh", "api"] and "/comments?per_page=" in cmd[2] and "&since=" not in cmd[2]:
            page = json.loads(result.stdout)
            return CommandResult(cmd, result.cwd, json.dumps(page[:-1]), "", 0)  # drop the newest
        return result

    runner.run = omitting_run
    config = make_config(tmp_path, reviewer="codex", coder="claude")
    with pytest.raises(AgentLoopError, match="cannot be authenticated"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert _agent_order_1292(runner) == []  # no coder, no reviewer, no ordinary fallback
    assert _records_1292(runner, "head-review-recovery") == []


@pytest.mark.parametrize(
    "case",
    [
        "legacy-anchor", "operator-budget-checkpoint", "operator-budget-coder-record",
        "dispatch-and-rejection",
    ],
)
def test_rest_page_omitting_any_record_recovery_depends_on_stops_before_any_agent_call(tmp_path, case):
    """Generalizes the omitted-handoff rule to checkpoint anchors and budget sources."""
    prelaunch = _meta_1292("summary", "scheduler-prelaunch", round_number=2, items=(_item_1292(),))
    flag = False
    if case == "legacy-anchor":
        records, omit = [*_history_1292(), _checkpoint_1292()], 2
    elif case in {"operator-budget-checkpoint", "operator-budget-coder-record"}:
        used = dataclasses.replace(
            _checkpoint_1292(),
            qualification_checkpoint=QualificationCheckpoint(
                obligation_kind="managed-exact-head-ci", obligation_identity="ci",
                lifecycle="repair_required", failed_head_sha=_A_1292, candidate_head_sha=None,
                allowed_rounds=5, watch_failure_extension_used=True, watch_head_extension_used=False,
            ),
        )
        if case == "operator-budget-coder-record":
            # An authoritative coder record that carries the used-extension checkpoint.
            used = dataclasses.replace(
                _meta_1292("coder", subject=_A_1292, round_number=2, items=(_item_1292(),)),
                qualification_checkpoint=used.qualification_checkpoint,
            )
        records, omit, flag = [*_history_1292(), used, prelaunch], 2, True
    else:
        records = [
            *_history_1292(), _dispatch_1292(),
            _rejection_1292(attempt=2, subject=_B_1292, reason="again",
                            carried_rejection_reasons=("first",)),
        ]
        omit = 3
    runner = _stranded_runner_1292(records, _B_1292)
    original = runner.run

    def omitting_run(cmd, *args, **kwargs):
        result = original(cmd, *args, **kwargs)
        if cmd[:2] == ["gh", "api"] and "/comments?per_page=" in cmd[2] and "&since=" not in cmd[2]:
            page = json.loads(result.stdout)
            del page[omit]  # a successful page that lacks the needed record
            return CommandResult(cmd, result.cwd, json.dumps(page), "", 0)
        return result

    runner.run = omitting_run
    config = make_config(tmp_path, reviewer="codex", coder="claude", review_unrecorded_head=flag)
    with pytest.raises(AgentLoopError, match="cannot be authenticated"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert _agent_order_1292(runner) == []
    assert _records_1292(runner, "head-review-recovery") == []


def test_restart_after_partial_head_review_does_not_replay_pre_handoff_reviews(tmp_path):
    """Row operator-record-interrupted (iii): only post-handoff publications count."""
    stale_codex = dataclasses.replace(
        _meta_1292("reviewer", subject=_B_1292, round_number=2, items=(_item_1292(),)),
        agent="Codex", state="blocking",
        canonical_reviewer_response=structured_pr_review(
            state="blocking", summary="Old verdict.", blocking_items=["Old blocker."],
            prior_item_dispositions=[{"item_id": "item-1", "disposition": "blocking",
                                      "note": "Old note on the stale head."}],
        ),
    )
    gemini = dataclasses.replace(
        _meta_1292("reviewer", subject=_B_1292, round_number=2, items=(_item_1292(),)),
        agent="Gemini", state="approved",
        canonical_reviewer_response=structured_pr_review(
            state="approved", summary="Resolved.", reviewer="Google Gemini",
            prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        ),
    )
    records = [
        *_history_1292(), stale_codex,
        _meta_1292("summary", "reconciliation", subject=_B_1292, round_number=2, items=(_item_1292(),)),
        _handoff_1292(round_number=2), gemini,
    ]
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="approved", summary="Resolved now.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            )
        ],
        pr_payload={"headRefOid": _B_1292, "comments": _seed_1292(*records)},
        authenticated_actor=_ACTOR_1292,
        advance_pr_head_on_coder_followup=False,
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), coder="claude")
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    # Codex's pre-handoff verdict was not replayed: it was invoked afresh.
    assert "codex" in _agent_sequence(runner)
    assert len(_records_1292(runner, "head-review-recovery")) == 1


def test_manual_push_after_completed_head_review_recovery_does_not_revive_pre_handoff_verdicts(tmp_path):
    """Generalizes the item-6 boundary to prior-head reconstruction (item-7)."""
    def reviewer(agent, state, disposition):
        return dataclasses.replace(
            _meta_1292("reviewer", subject=_B_1292, round_number=2, items=(_item_1292(),)),
            agent=agent, state=state,
            dispositions=(
                ReviewItemDisposition(item_id="item-1", reviewer=agent, disposition=disposition),
            ),
        )

    records = [
        *_history_1292(),
        _meta_1292("coder", subject=_B_1292, round_number=2, items=(_item_1292(),)),
        reviewer("Codex", "blocking", "blocking"),  # published before the handoff
        _handoff_1292(round_number=2),
        reviewer("Codex", "approved", "resolved"),  # the completed fresh recovery review
    ]
    runner = _stranded_runner_1292(
        records, _C_1292,
        codex_outputs=[structured_pr_review(state="approved", summary="Nothing carried.")],
    )
    config = make_config(tmp_path, reviewer="codex", coder="claude")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    # item-1 was cleared by the fresh review: nothing is carried to C or sent to a coder.
    assert "claude" not in _agent_order_1292(runner)
    assert _agent_order_1292(runner)[:1] == ["codex"]


@pytest.mark.parametrize("mode", ["claim", "gap-only", "none", "not-applicable"])
def test_run_pr_loop_followup_comment_carries_coverage_map_without_reask(
    tmp_path, monkeypatch, mode
):
    with_matrix = mode in {"claim", "gap-only"}
    if with_matrix:
        plan_context = _followup_matrix_context()
    elif mode == "not-applicable":
        na_matrix = parse_risk_test_matrix({
            "applicability": "not-applicable", "rows": [], "important_exclusions": [],
            "not_applicable_rationale": "Formatting-only change without any stateful path.",
        })
        na_identity = risk_test_matrix_identity(na_matrix)
        plan_context = execution_policy.make_approved_plan_context(
            "Approved plan.\n\n" + render_risk_test_matrix_section(na_matrix),
            source_locator="test na plan",
            risk_test_matrix_contract_version=1,
            risk_test_matrix_payload=na_matrix.to_payload(),
            risk_test_matrix_changes_payload=(),
            risk_test_matrix_identity=na_identity,
            risk_test_matrix_boundary_digest=na_identity,
        )
    else:
        plan_context = None
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=2)
    head = _init_followup_checkout(config.claude_dir)
    current = _followup_observation(
        execution_ref="coder-turn:observation-2", receipt_id="receipt-current-head",
        head=head, timestamp="2026-01-01T00:00:01Z",
    )
    raw = structured_coder_followup(
        addressed_items=["item-1"],
        summary="The follow-up was completed and the selected workflow was exercised.",
    )
    payload, end = json.JSONDecoder().raw_decode(raw)
    payload["risk_test_matrix_claims"] = [{
        "row_id": "followup-derived-evidence",
        "execution_refs": [current.execution_ref],
        "test_identifiers": ["tests/test_orchestrator_pr.py::test_followup_workflow"],
        "test_locations": ["tests/test_orchestrator_pr.py"],
        "workflow_path_claim": "The real PR follow-up caller reached post-head derivation.",
        "outcome_assertions": ["The selected managed observation passed."],
        "forbidden_effect_assertions": ["The PR handoff was retained."],
        "caveats": [],
        "test_level": "workflow",
    }]
    if mode == "not-applicable":
        payload.pop("risk_test_matrix_claims")
    if mode == "gap-only":
        payload.pop("risk_test_matrix_claims")
        payload["risk_test_matrix_coverage_gaps"] = [{
            "row_id": "followup-derived-evidence",
            "reason": "cannot be driven through the real follow-up caller here",
            "proposed_correction": "downgrade the row to a unit test",
        }]
    text = json.dumps(payload) + raw[end:]
    parsed = validate_structured_coder_followup(
        text,
        required_architecture_impact_contract=1,
        delivered_risk_test_matrix=plan_context.risk_test_matrix_payload if plan_context else None,
        delivered_risk_test_matrix_identity=plan_context.risk_test_matrix_identity if plan_context else None,
        required_risk_test_matrix_contract=1 if plan_context else 0,
        delivered_risk_test_matrix_row_ids=("followup-derived-evidence",) if plan_context else None,
        execution_catalog=(current,),
    )
    assert parsed is not None
    coder_response = ValidatedAgentResponse(
        text=text, session_id="coder-session", marker_value=parsed,
        acquisition_test_turn_id="coder-turn", acquisition_test_observations=(current,),
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking", summary="Needs work.", blocking_items=["Exercise follow-up evidence derivation."],
            ),
            structured_pr_review(
                state="approved", summary="Covered.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        pr_payload={"headRefOid": "predecessor-head"},
        git_head="predecessor-head",
    )
    coder_calls = []
    real_validated_agent = validated_agent._run_validated_agent

    def fake_validated_agent(*args, **kwargs):
        if kwargs.get("role") == "coder":
            coder_calls.append(kwargs.get("prompt"))
            runner.pr_payload["headRefOid"] = head
            runner.simulate_agent_turn(config, config.claude_dir, head=head)
            return coder_response
        return real_validated_agent(*args, **kwargs)

    patch_attr_in(monkeypatch, (
        validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage,
        issue_loop,
    ), "_run_validated_agent", fake_validated_agent)
    monkeypatch.setattr(
        response_validation, "stable_tracked_tree_snapshot",
        lambda _workdir: SimpleNamespace(
            head=head, tracked_digest="tree-current", complete=True, stable=True, status_clean=True,
        ),
    )
    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_implementation, issue_loop,
    ), "_read_assigned_workdir_head", lambda *_a, **_k: head)

    assert run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=plan_context) == 0

    assert len(coder_calls) == 1, "the follow-up path never sends a coverage re-ask"
    coder_comment = next(
        item["body"] for item in runner.pr_payload["comments"]
        if isinstance(item, dict) and isinstance(item.get("body"), str)
        and "AGENT_LOOP_META: " in item["body"]
        and _decode_round_metadata(
            item["body"].split("AGENT_LOOP_META: ", 1)[1].split(" -->", 1)[0]
        ).role == "coder"
    )
    review_prompts = [
        cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]
    ]
    if with_matrix:
        assert "### Risk-matrix coverage map" in coder_comment
        assert f"complete at {head}" in coder_comment
        assert "Risk-matrix coverage map" in review_prompts[-1]
        assert "after one coverage re-ask" not in coder_comment
    else:
        assert "Risk-matrix coverage map" not in coder_comment


def test_run_pr_loop_followup_semantic_correction_keeps_original_gaps_and_fixes_claims(
    tmp_path, monkeypatch
):
    def row(row_id, level):
        return {
            "row_id": row_id, "label": f"Behaviour {row_id}",
            "entry_path_or_mode": "PR follow-up", "initial_state": "blocking review",
            "event": "follow-up authenticated", "expected_outcome": "map rendered",
            "forbidden_side_effects": ["No re-ask."], "proposed_test_level": level,
            "proposed_test_location": "tests/test_orchestrator_pr.py",
            "applicability": "required", "related_scope_item_ids": ["scope-1"],
            "execution_owner": "one-shot",
        }

    matrix = parse_risk_test_matrix({
        "applicability": "applicable",
        "rows": [row("followup-a", "orchestrator"), row("followup-b", "unit")],
        "important_exclusions": ["Planned tests are not evidence."],
    })
    identity = risk_test_matrix_identity(matrix)
    plan_context = execution_policy.make_approved_plan_context(
        "Approved plan.\n\n" + render_risk_test_matrix_section(matrix),
        source_locator="test two-row plan",
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest=identity,
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=2)
    head = _init_followup_checkout(config.claude_dir)
    current = _followup_observation(
        execution_ref="coder-turn:observation-2", receipt_id="receipt-current-head",
        head=head, timestamp="2026-01-01T00:00:01Z",
    )

    def claim(**extra):
        return {
            "row_id": "followup-a", "execution_refs": [current.execution_ref],
            "test_identifiers": ["tests/test_orchestrator_pr.py::test_followup_workflow"],
            "test_locations": ["tests/test_orchestrator_pr.py"],
            "workflow_path_claim": "The real follow-up caller ran.",
            "outcome_assertions": ["It passed."], "forbidden_effect_assertions": ["No re-ask."],
            "caveats": [], **extra,
        }

    def gap(reason):
        return {"row_id": "followup-b", "reason": reason, "proposed_correction": "downgrade the row"}

    raw = structured_coder_followup(addressed_items=["item-1"], summary="Follow-up with correction.")
    payload, end = json.JSONDecoder().raw_decode(raw)
    original = dict(payload)
    # Unknown selector -> actionable diagnostic; no test_level -> deficiency.
    original["risk_test_matrix_claims"] = [claim(execution_refs=["coder-turn:not-in-catalog"])]
    original["risk_test_matrix_coverage_gaps"] = [gap("original gap reason")]
    text = json.dumps(original) + raw[end:]
    corrected = dict(payload)
    corrected["risk_test_matrix_claims"] = [claim(test_level="workflow")]
    corrected["risk_test_matrix_coverage_gaps"] = [gap("CHANGED BY CORRECTION")]
    corrected_text = json.dumps(corrected) + raw[end:]
    parsed = validate_structured_coder_followup(
        text, required_architecture_impact_contract=1,
        delivered_risk_test_matrix=plan_context.risk_test_matrix_payload,
        delivered_risk_test_matrix_identity=identity, required_risk_test_matrix_contract=1,
        delivered_risk_test_matrix_row_ids=("followup-a", "followup-b"),
        execution_catalog=(current,),
    )
    assert parsed is not None
    coder_response = ValidatedAgentResponse(
        text=text, session_id="coder-session", marker_value=parsed,
        acquisition_test_turn_id="coder-turn", acquisition_test_observations=(current,),
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(state="blocking", summary="Needs work.", blocking_items=["Fix the follow-up."]),
            structured_pr_review(
                state="approved", summary="Covered.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        pr_payload={"headRefOid": "predecessor-head"}, git_head="predecessor-head",
    )
    coder_calls = []
    real_validated_agent = validated_agent._run_validated_agent

    def fake_validated_agent(*args, **kwargs):
        if kwargs.get("role") == "coder":
            coder_calls.append(1)
            runner.pr_payload["headRefOid"] = head
            runner.simulate_agent_turn(config, config.claude_dir, head=head)
            return coder_response
        return real_validated_agent(*args, **kwargs)

    corrections = []
    real_run_agent_result = validated_agent.run_agent_result

    def fake_correction(*a, **kw):
        if kw.get("label") != "semantic-evidence-correction":
            return real_run_agent_result(*a, **kw)
        corrections.append(1)
        return SimpleNamespace(text=corrected_text)

    patch_attr_in(monkeypatch, (
        validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage,
        issue_loop,
    ), "_run_validated_agent", fake_validated_agent)
    patch_attr_in(monkeypatch, (
        validated_agent, response_validation,
    ), "run_agent_result", fake_correction)
    monkeypatch.setattr(
        response_validation, "stable_tracked_tree_snapshot",
        lambda _workdir: SimpleNamespace(
            head=head, tracked_digest="tree-current", complete=True, stable=True, status_clean=True,
        ),
    )
    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_implementation, issue_loop,
    ), "_read_assigned_workdir_head", lambda *_a, **_k: head)

    assert run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=plan_context) == 0

    assert corrections == [1] and len(coder_calls) == 1
    coder_comment = next(
        item["body"] for item in runner.pr_payload["comments"]
        if isinstance(item, dict) and isinstance(item.get("body"), str)
        and "AGENT_LOOP_META: " in item["body"]
        and _decode_round_metadata(
            item["body"].split("AGENT_LOOP_META: ", 1)[1].split(" -->", 1)[0]
        ).role == "coder"
    )
    section = coder_comment.split("### Risk-matrix coverage map")[1]
    assert f"complete at {head}" in section
    assert "declared level workflow" in section
    assert "original gap reason" in section and "CHANGED BY CORRECTION" not in section
    review_prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]
    assert "original gap reason" in review_prompts[-1] and "CHANGED BY CORRECTION" not in review_prompts[-1]


def _unchanged_head_runner(*, with_failure_receipt, tmp_path):
    from coding_review_agent_loop.local_test_evidence import (
        EnvironmentIdentityRegistry,
        EvidenceScope,
        LocalTestObservation,
        TreeAttribution,
    )

    blocking = structured_pr_review(
        state="blocking", summary="Needs work.", blocking_items=["Fix it."]
    )
    runner = FakeRunner(
        codex_outputs=[
            blocking,
            structured_pr_review(
                state="blocking", summary="Still needs work.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "blocking", "note": "Fix it."}
                ],
            ),
            structured_pr_review(state="approved"),
        ],
        claude_outputs=[
            structured_coder_followup(remaining_items=["item-1"], addressed_items=[]),
            structured_coder_followup(remaining_items=["item-1"], addressed_items=[]),
        ],
        advance_pr_head_on_coder_followup=False,
    )
    if with_failure_receipt:
        registry = EnvironmentIdentityRegistry()
        runner._environment_registry = registry
        runner._local_test_observations.append(LocalTestObservation(
            command=("python3", "-m", "pytest", "tests/test_x.py", "-q", "-n", "0"),
            outcome="failed", provenance="parent-observed", scope=EvidenceScope("unknown", ()),
            receipt_id="receipt-open", turn_id="turn-opaque", cwd=str(tmp_path),
            normalized_command="python3 -m pytest tests/test_x.py -q -n 0", returncode=4,
            attribution=TreeAttribution(state="current-head", head="abc123", stable=True),
            environment_identity=registry.capture({"PATH": "/usr/bin"}),
        ))
    return runner


def test_unchanged_head_stop_names_unsuperseded_evidence_receipts(tmp_path):
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=6)
    runner = _unchanged_head_runner(with_failure_receipt=True, tmp_path=tmp_path)
    with pytest.raises(AgentLoopError) as with_receipts:
        run_pr_loop(runner, pr_number=77, config=config)
    message = str(with_receipts.value)
    assert "unchanged in 2 consecutive follow-up rounds" in message
    assert "human review required." in message
    assert "unsuperseded failure receipts" in message and "-n 0" in message
    assert "evidence bookkeeping" in message

    runner = _unchanged_head_runner(with_failure_receipt=False, tmp_path=tmp_path)
    with pytest.raises(AgentLoopError) as without:
        run_pr_loop(runner, pr_number=77, config=config)
    import re as _re

    found = _re.fullmatch(
        r"PR #77: Claude left head (\S+) unchanged in 2 consecutive follow-up rounds, so another "
        r"review of the same diff cannot change the verdict\. Stopping before round (\d+); "
        r"human review required\.",
        str(without.value),
    )
    assert found is not None, str(without.value)
