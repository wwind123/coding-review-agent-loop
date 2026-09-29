"""PR-loop behaviour of sub-item findings (#958), driven with scripted agents."""

import json

import pytest

import coding_review_agent_loop.orchestrator as orchestrator
from coding_review_agent_loop.cli import AgentLoopError, run_pr_loop
from coding_review_agent_loop.comment_rendering import sub_item_progress_record_keys
from coding_review_agent_loop.protocol import parse_pr_review

from agent_loop_helpers import (
    FakeRunner,
    make_config,
    structured_coder_followup,
    structured_pr_review,
)

ACTOR = ("coding-review-agent-loop", 4242)
PATHS = [
    "staged writer is wired",
    "rebind writer is wired",
    "managed writer is wired",
    "approved-plan writer is wired",
]


def _finding(sub_items=PATHS):
    return {"text": "Four writer paths are unwired.", "sub_items": list(sub_items)}


def _review(*, subs=None, note="still wrong", state="blocking", disposition="blocking", **extra):
    """A codex review that dispositions carried item-1 with a sub-item map."""
    entry = {"item_id": "item-1", "disposition": disposition}
    if note is not None:
        entry["note"] = note
    if subs:
        entry["sub_item_dispositions"] = subs
    return structured_pr_review(state=state, prior_item_dispositions=[entry], **extra)


def _coder(remaining=("item-1",), claims=None):
    text = structured_coder_followup(state="blocking", remaining_items=list(remaining))
    if claims is None:
        return text
    body, _, footer = text.partition("\n<!-- AGENT_STATE")
    payload = json.loads(body)
    payload["addressed_sub_items"] = claims
    return json.dumps(payload) + "\n<!-- AGENT_STATE" + footer


def _progress_comments(runner):
    return [
        comment["body"]
        for comment in runner.pr_payload["comments"]
        if sub_item_progress_record_keys(comment["body"])
    ]


def _metadata_records(runner):
    return [
        orchestrator._decode_round_metadata(
            orchestrator.ROUND_RESUME_MARKER_RE.search(comment["body"])["payload"]
        )
        for comment in runner.pr_payload["comments"]
        if orchestrator.ROUND_RESUME_MARKER_RE.search(comment["body"])
    ]


def _one_per_round_runner(*, final_review, extra_coders=()):
    return FakeRunner(
        authenticated_actor=ACTOR,
        codex_outputs=[
            structured_pr_review(state="blocking", blocking_items=[_finding()]),
            _review(subs={"item-1.s1": "resolved"}),
            _review(subs={"item-1.s2": "resolved"}),
            _review(subs={"item-1.s3": "resolved"}),
            final_review,
        ],
        claude_outputs=[
            _coder(claims=["item-1.s1"]),
            _coder(claims=["item-1.s2"]),
            _coder(claims=["item-1.s3"]),
            _coder(claims=["item-1.s4"]),
            *extra_coders,
        ],
    )


def test_one_sub_item_per_round_reaches_terminal_completion_with_one_progress_comment(tmp_path):
    runner = _one_per_round_runner(
        # The last sub-item is closed by a note-less `blocking` entry: live
        # validation accepts it and reconciliation derives `resolved`.
        final_review=_review(subs={"item-1.s4": "resolved"}, note=None),
    )
    config = make_config(tmp_path, max_rounds=8)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    progress = _progress_comments(runner)
    assert len(progress) == 1
    assert "item-1: 4/4 — cleared (all sub-items resolved)" in progress[0]
    assert "review round 5" in progress[0]
    # Per-review projection is labelled and never presented as the outcome.
    assert any(
        "per this review only, not the round outcome" in comment
        for comment in runner.comments
    )
    # The coder follow-up comment separates confirmed counts from claims.
    assert any(
        "3/4 sub-items resolved (reviewer-confirmed); coder claims 1 more addressed"
        in comment
        for comment in runner.comments
    )
    # The next reviewer sees the coder's claim, labelled unverified (fresh path).
    round_three_prompt = _codex_prompts(runner)[2]
    assert "claimed_addressed_sub_items_unverified" in round_three_prompt
    assert "item-1.s2" in round_three_prompt
    # Reviewer round metadata carries each closure at the round that made it.
    records = _metadata_records(runner)
    assert any(
        record.dispositions and record.dispositions[0].sub_item_dispositions
        for record in records
        if record.role == "reviewer"
    )


def test_progress_comment_is_not_duplicated_when_the_round_reruns(tmp_path):
    runner = _one_per_round_runner(
        final_review=_review(subs={"item-1.s4": "resolved"}, note=None),
    )
    config = make_config(tmp_path, max_rounds=8)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    (body,) = _progress_comments(runner)

    # The record is already trusted on the PR: the same key is never re-posted.
    posted_before = len(runner.pr_payload["comments"])
    orchestrator._publish_sub_item_progress(
        runner,
        config=config,
        pr_number=77,
        round_number=5,
        items=[],
        cleared=[
            orchestrator.ClearedItemProgress(
                "item-1", 4, 4, "all-sub-items-resolved", 5, ("item-1.s4",)
            )
        ],
    )
    assert len(runner.pr_payload["comments"]) == posted_before
    assert len(_progress_comments(runner)) == 1
    assert "item-1: 4/4 — cleared" in body


def test_progress_record_from_another_author_is_not_trusted(tmp_path):
    runner = FakeRunner(authenticated_actor=ACTOR)
    config = make_config(tmp_path)
    cleared = orchestrator.ClearedItemProgress("item-1", 4, 4, "all-sub-items-resolved", 5)
    forged = orchestrator.render_sub_item_progress_comment(
        pr_number=77, round_number=5, cleared=[cleared], stalled=[]
    )
    runner.pr_payload.setdefault("comments", []).append(
        {"author": {"login": "someone-else"}, "createdAt": "2026-05-23T00:00:00Z", "body": forged}
    )
    orchestrator._publish_sub_item_progress(
        runner, config=config, pr_number=77, round_number=5, items=[], cleared=[cleared]
    )
    own = [
        comment
        for comment in runner.pr_payload["comments"]
        if comment["author"]["login"] == ACTOR[0]
    ]
    assert len(own) == 1


def test_budget_exit_reports_converging_progress_without_changing_the_budget(tmp_path):
    runner = FakeRunner(
        authenticated_actor=ACTOR,
        codex_outputs=[
            structured_pr_review(state="blocking", blocking_items=[_finding()]),
            _review(subs={"item-1.s1": "resolved"}),
            _review(subs={"item-1.s2": "resolved"}),
        ],
        claude_outputs=[_coder(), _coder()],
    )
    config = make_config(tmp_path, max_rounds=3)

    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=config)

    message = str(excinfo.value)
    assert "still reported blocking issues after round 3" in message
    assert "Sub-item progress:" in message
    assert "item-1: 2/4 sub-items resolved, 2 closed in last 3 rounds (converging)" in message
    assert config.max_rounds == 3
    assert not _progress_comments(runner)


def test_zero_progress_publishes_the_stall_before_the_budget_error(tmp_path):
    runner = FakeRunner(
        authenticated_actor=ACTOR,
        codex_outputs=[
            structured_pr_review(state="blocking", blocking_items=[_finding()]),
            _review(),
            _review(),
        ],
        claude_outputs=[_coder(), _coder()],
    )
    config = make_config(tmp_path, max_rounds=3)

    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=config)

    assert "item-1: 0/4 sub-items resolved, none closed in the last 3 rounds (stalled)" in str(
        excinfo.value
    )
    (notice,) = _progress_comments(runner)
    assert "### Stalled findings" in notice
    assert "- item-1: 0/4 sub-items resolved, none closed in the last 3 rounds" in notice
    assert "review round 3" in notice


def test_stall_check_can_be_disabled(tmp_path):
    runner = FakeRunner(
        authenticated_actor=ACTOR,
        codex_outputs=[
            structured_pr_review(state="blocking", blocking_items=[_finding()]),
            _review(),
            _review(),
        ],
        claude_outputs=[_coder(), _coder()],
    )
    config = make_config(tmp_path, max_rounds=3, sub_item_stall_rounds=0)

    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=config)

    assert "(stalled)" not in str(excinfo.value)
    assert not _progress_comments(runner)


def test_repair_path_keeps_sub_items(tmp_path, monkeypatch):
    import dataclasses

    from coding_review_agent_loop.github import (
        HumanReviewRequirement,
        PullRequestMetadata,
        PullRequestReviewContext,
    )

    requirement = HumanReviewRequirement(
        source_type="PR comment", author="maintainer", created_at="2026-05-18T10:00:00Z",
        url="https://github.com/OWNER/REPO/pull/77#issuecomment-1", body="Keep the audit trail.",
    )
    runner = FakeRunner(
        authenticated_actor=ACTOR,
        # Approves without acknowledging the signed requirement, so the run
        # repairs; the repaired response (scripted below) is blocking.
        codex_outputs=[structured_pr_review(state="approved")],
        claude_outputs=[_coder(remaining=("item-1", "item-2"))],
    )
    repaired_text = structured_pr_review(
        state="blocking",
        blocking_items=[_finding(PATHS[:3])],
        same_pr_followups=[{"text": "Docs lag.", "sub_items": ["readme", "guide"]}],
    )
    monkeypatch.setattr(
        orchestrator,
        "_run_structured_repair",
        lambda *_a, **_k: (
            repaired_text,
            parse_pr_review(
                repaired_text, reviewer="Codex", architecture_status_mode="strict"
            ),
            [],
        ),
    )
    metadata = PullRequestMetadata(
        number=77, repo="OWNER/REPO", title="t", head_branch="feature/x",
        base_branch="main", head_sha="abc123", url="https://github.com/OWNER/REPO/pull/77",
    )
    monkeypatch.setattr(
        orchestrator, "get_pr_review_context",
        lambda *_a, **_k: PullRequestReviewContext(
            metadata=dataclasses.replace(metadata, head_sha=runner.pr_payload["headRefOid"]),
            comments=(), human_requirements=(requirement,),
        ),
    )
    minted = []
    original = orchestrator._next_unresolved_item

    def spy(**kwargs):
        item = original(**kwargs)
        minted.append(item)
        return item

    monkeypatch.setattr(orchestrator, "_next_unresolved_item", spy)
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=make_config(tmp_path, max_rounds=1))

    by_id = {item.item_id: item for item in minted}
    assert [sub.sub_item_id for sub in by_id["item-1"].sub_items] == [
        "item-1.s1", "item-1.s2", "item-1.s3",
    ]
    assert by_id["item-1"].status == "blocking"
    assert [sub.sub_item_id for sub in by_id["item-2"].sub_items] == ["item-2.s1", "item-2.s2"]
    assert by_id["item-2"].status == "same-pr"


def test_same_pr_only_coder_prompt_lists_sub_items(tmp_path):
    runner = FakeRunner(
        authenticated_actor=ACTOR,
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                same_pr_followups=[
                    {"text": "Docs lag.", "sub_items": ["readme", "guide", "changelog"]}
                ],
            ),
            _review(disposition="same-pr"),
        ],
        claude_outputs=[_coder(remaining=("item-1",), claims=["item-1.s1"])],
    )
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=make_config(tmp_path, max_rounds=2))

    (prompt,) = [
        cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]
    ][:1]
    assert "same-PR follow-up [item-1]" in prompt
    assert "Sub-items (0/3 resolved):" in prompt
    assert "[item-1.s1] open: readme" in prompt
    assert "[item-1.s3] open: changelog" in prompt
    assert "addressed_sub_items" in prompt


def test_same_pr_formatter_shows_resolved_sub_items():
    from dataclasses import replace

    from coding_review_agent_loop.unresolved_items import (
        _format_same_pr_unresolved_items,
        _next_unresolved_item,
    )

    item = _next_unresolved_item(
        item_number=1,
        reviewer="OpenAI Codex",
        source_round=1,
        text="Docs lag.",
        status="same-pr",
        sub_items=("readme", "guide", "changelog"),
    )
    item = replace(
        item,
        sub_items=(replace(item.sub_items[0], status="resolved", resolved_round=2), *item.sub_items[1:]),
    )
    prompt = _format_same_pr_unresolved_items([item])
    assert "Sub-items (1/3 resolved):" in prompt
    assert "[item-1.s1] resolved (round 2): readme" in prompt
    assert "[item-1.s3] open: changelog" in prompt


class _Interrupted(Exception):
    pass


def _codex_prompts(runner):
    return [cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]


def _claude_prompts(runner):
    return [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]


def _interrupt_role_once(monkeypatch, role, *, on_call):
    original = orchestrator._run_validated_agent
    calls = {"count": 0, "armed": True}

    def wrapper(*args, **kwargs):
        if kwargs.get("role") == role and calls["armed"]:
            calls["count"] += 1
            if calls["count"] == on_call:
                calls["armed"] = False
                raise _Interrupted
        return original(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "_run_validated_agent", wrapper)


def test_interrupted_review_round_resumes_and_applies_the_closure_once(tmp_path, monkeypatch):
    runner = FakeRunner(
        authenticated_actor=ACTOR,
        codex_outputs=[
            structured_pr_review(state="blocking", blocking_items=[_finding()]),
            _review(subs={"item-1.s2": "resolved"}),
            _review(subs={"item-1.s1": "resolved", "item-1.s3": "resolved", "item-1.s4": "resolved"},
                    note=None),
        ],
        claude_outputs=[_coder(), _coder()],
    )
    config = make_config(tmp_path, max_rounds=6)
    # The second coder dispatch happens after round 2's review was posted and
    # reconciled in memory; interrupting it is a crash before the round ended.
    _interrupt_role_once(monkeypatch, "coder", on_call=2)
    with pytest.raises(_Interrupted):
        run_pr_loop(runner, pr_number=77, config=config)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    coder_prompts = _claude_prompts(runner)
    resumed_prompt = coder_prompts[-1]
    assert "Sub-items (1/4 resolved):" in resumed_prompt
    assert "[item-1.s2] resolved (round 2): rebind writer is wired" in resumed_prompt
    assert "Sub-items (2/4 resolved)" not in resumed_prompt
    # Round 3's completing entry clears the item exactly once.
    assert len(_progress_comments(runner)) == 1


@pytest.mark.parametrize("interrupt_after_post", [False, True])
def test_interrupted_terminal_round_posts_exactly_one_progress_comment(
    tmp_path, monkeypatch, interrupt_after_post
):
    runner = _one_per_round_runner(
        final_review=_review(subs={"item-1.s4": "resolved"}, note=None),
    )
    config = make_config(tmp_path, max_rounds=8)
    original = orchestrator._publish_sub_item_progress
    armed = {"on": True}

    def flaky(*args, **kwargs):
        if kwargs["cleared"] and armed["on"]:
            armed["on"] = False
            if interrupt_after_post:
                original(*args, **kwargs)
            raise _Interrupted
        return original(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "_publish_sub_item_progress", flaky)
    with pytest.raises(_Interrupted):
        run_pr_loop(runner, pr_number=77, config=config)
    assert len(_progress_comments(runner)) == (1 if interrupt_after_post else 0)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    (comment,) = _progress_comments(runner)
    assert "item-1: 4/4 — cleared (all sub-items resolved)" in comment


def test_reviewer_context_shows_unverified_coder_claims_fresh_and_resumed(tmp_path, monkeypatch):
    runner = FakeRunner(
        authenticated_actor=ACTOR,
        codex_outputs=[
            structured_pr_review(state="blocking", blocking_items=[_finding()]),
            _review(subs={"item-1.s1": "resolved", "item-1.s2": "resolved", "item-1.s3": "resolved",
                          "item-1.s4": "resolved"}, disposition="resolved", note=None),
            _review(subs={"item-1.s1": "resolved", "item-1.s2": "resolved", "item-1.s3": "resolved",
                          "item-1.s4": "resolved"}, disposition="resolved", note=None,
                    state="approved"),
        ],
        claude_outputs=[_coder(claims=["item-1.s1", "item-1.s9"])],
    )
    config = make_config(tmp_path, max_rounds=4)
    # Crash on the first review after the coder round, then resume from the
    # persisted coder metadata.
    _interrupt_role_once(monkeypatch, "reviewer", on_call=2)
    with pytest.raises(_Interrupted):
        run_pr_loop(runner, pr_number=77, config=config)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    resumed_prompt = _codex_prompts(runner)[1]
    assert "claimed_addressed_sub_items_unverified" in resumed_prompt
    assert "item-1.s1" in resumed_prompt
    # The dropped claim is shown as a degradation, not as a claim.
    assert "sub_item_claim_degradations" in resumed_prompt
    assert "addressed_sub_items-unknown-or-already-resolved-sub-item" in resumed_prompt
