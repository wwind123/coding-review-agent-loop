"""Release of carried coder disputes by newer signed requirements (#2025).

A coder dispute that the reviewer keeps blocking stops the PR loop for a
human decision. A signed requirement created strictly after the dispute time
(taken from the admitted coder follow-up round record) releases that stop for
the next round only; an unknown time, a tie, or no newer requirement holds.
"""

from __future__ import annotations

import datetime

import pytest

from coding_review_agent_loop.cli import run_pr_loop
from coding_review_agent_loop.dispute_release import (
    DisputeRelease,
    coder_dispute_times,
    round_dispute_release,
    stop_for_held_disputes,
)
from coding_review_agent_loop.errors import AgentLoopError, HumanDecisionRequiredError
from coding_review_agent_loop.github import (
    HumanReviewRequirement,
    IssueComment,
    _parse_pr_human_requirements,
)
from coding_review_agent_loop.prompt_context import _format_unresolved_review_items
from coding_review_agent_loop.protocol import UnresolvedReviewItem
from coding_review_agent_loop.round_metadata import PostedRoundMetadata, _attach_round_metadata
from coding_review_agent_loop.unresolved_items import (
    CODER_DISPUTE_NOTE_PREFIX,
    _format_same_pr_unresolved_items,
    _raise_if_maintained_disputed_items,
    format_coder_followup_context,
    released_dispute_requirement_ids,
)

from agent_loop_helpers import FakeRunner, make_config, structured_coder_followup, structured_pr_review
from pr_loop_test_helpers import _rewrite_pr_metadata, _with_code_classification

ITEM_TEXT = "The gemini-3.5-flash pricing constant is wrong ($0.30 not $1.50)."
EVIDENCE = "Official docs confirm $1.50/1M tokens is correct."
REVIEWER_NOTE = "I checked and the pricing is still wrong."
HUMAN_TEXT = "Use the reviewer's pricing constant."
ACK_ID = "item-human-requirements-acknowledgement"
CONFLICTING = {
    "mergeable": "CONFLICTING",
    "mergeStateStatus": "DIRTY",
    "headRefOid": "6367c48dd193d56ea7b0baad25b19455e529f5ee",
    "baseRefName": "main",
}
T0 = datetime.datetime(2026, 5, 23, 0, 0, 2, tzinfo=datetime.timezone.utc)


# --- helpers ---------------------------------------------------------------


def _disputed_item(item_id="item-1", status="blocking"):
    return UnresolvedReviewItem(
        item_id=item_id, reviewer="Codex", source_round=1, status=status, text=ITEM_TEXT,
        notes=(f"{CODER_DISPUTE_NOTE_PREFIX}: {EVIDENCE}", f"Codex: {REVIEWER_NOTE}"),
    )


def _requirement(created_at, body="Decision."):
    return HumanReviewRequirement(
        source_type="PR comment", author="maintainer", created_at=created_at,
        url="https://github.com/OWNER/REPO/pull/77#issuecomment-9", body=body,
    )


def _iso(value):
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _coder_comment(created_at, disputed, *, round_number=1, subject="a" * 40):
    metadata = _coder_metadata(disputed, round_number=round_number, subject=subject)
    return IssueComment(
        author="coding-review-agent-loop", created_at=created_at,
        body=_attach_round_metadata("Coder follow-up.", metadata),
    )


def _coder_metadata(disputed, *, round_number=1, subject="a" * 40):
    return PostedRoundMetadata(
        flow="pr", role="coder", agent="Claude", round_number=round_number, subject=subject,
        raw_structured_coder_response=structured_coder_followup(
            disputed_items=list(disputed),
            dispute_evidence={item_id: EVIDENCE for item_id in disputed},
        ),
    )


def _dispute_runner(**kwargs):
    """Round 1 blocks item-1; the coder disputes; the reviewer keeps blocking."""
    return FakeRunner(
        claude_outputs=[_dispute_output()],
        codex_outputs=[
            structured_pr_review(state="blocking", summary="Pricing constant is wrong.",
                                 blocking_items=[ITEM_TEXT]),
            _still_blocking(),
        ],
        **kwargs,
    )


def _dispute_output():
    return structured_coder_followup(
        disputed_items=["item-1"], dispute_evidence={"item-1": EVIDENCE},
        summary="Disputing item-1.",
    )


def _still_blocking(*, requirements=None):
    dispositions = [{"item_id": "item-1", "disposition": "blocking", "note": REVIEWER_NOTE}]
    if requirements is None:
        return structured_pr_review(state="blocking", summary="Still wrong.",
                                    prior_item_dispositions=dispositions)
    dispositions.append({"item_id": ACK_ID, "disposition": "resolved"})
    return _with_code_classification(
        structured_pr_review(state="blocking", summary="Still wrong.",
                             human_requirements_resolved=True,
                             prior_item_dispositions=dispositions),
        requirements,
    )


def _approve(requirements):
    return _with_code_classification(
        structured_pr_review(
            state="approved", summary="Fixed.", human_requirements_resolved=True,
            prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"},
                                     {"item_id": ACK_ID, "disposition": "resolved"}],
        ),
        requirements,
    )


def _config(tmp_path, **overrides):
    overrides.setdefault("max_rounds", 6)
    return make_config(tmp_path, coder="claude", reviewer="codex", **overrides)


def _stop_after_dispute(tmp_path, runner):
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path))
    assert len(_coder_calls(runner)) == 1


def _sign(runner, created_at=None):
    comments = runner.pr_payload["comments"]
    comments.append({
        "author": {"login": "maintainer"},
        "createdAt": created_at or f"2026-05-23T00:00:{len(comments):02d}Z",
        "url": "https://github.com/OWNER/REPO/pull/77#issuecomment-2025",
        "body": f"{HUMAN_TEXT}\n\n-- Human Reviewer",
    })
    return _parse_pr_human_requirements(runner.pr_payload)


def _coder_records(runner):
    return [
        comment for comment in runner.pr_payload["comments"]
        if "Coder follow-up" in comment["body"] or "Disputing item-1" in comment["body"]
    ]


def _coder_calls(runner):
    return [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]


def _reviewer_calls(runner):
    return [cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]


def _merge_calls(runner):
    return [cmd for cmd, _cwd in runner.commands if cmd[:3] == ["gh", "pr", "merge"]]


def _pointer_lines(prompt):
    return [line for line in prompt.splitlines() if "Dispute on item-1 is released" in line]


# --- unit: release rule ----------------------------------------------------


def test_release_requires_a_strictly_newer_requirement():
    item = _disputed_item()
    later = _requirement(_iso(T0 + datetime.timedelta(seconds=1)))
    released = released_dispute_requirement_ids([item], {"item-1": T0}, [later])
    assert released == {"item-1": (later.requirement_id,)}


@pytest.mark.parametrize(
    "created_at",
    [_iso(T0), _iso(T0 - datetime.timedelta(seconds=1)), "not-a-time", "", None],
    ids=["equal", "earlier", "unparseable", "empty", "missing"],
)
def test_release_holds_on_ties_older_or_bad_requirement_timestamps(created_at):
    released = released_dispute_requirement_ids(
        [_disputed_item()], {"item-1": T0}, [_requirement(created_at)]
    )
    assert released == {}


def test_release_holds_when_dispute_time_is_unknown():
    later = _requirement(_iso(T0 + datetime.timedelta(days=1)))
    assert released_dispute_requirement_ids([_disputed_item()], {"item-1": None}, [later]) == {}
    assert released_dispute_requirement_ids([_disputed_item()], {}, [later]) == {}


def test_release_ignores_items_without_a_dispute_note():
    plain = UnresolvedReviewItem(item_id="item-1", reviewer="Codex", source_round=1,
                                 status="blocking", text=ITEM_TEXT)
    later = _requirement(_iso(T0 + datetime.timedelta(seconds=5)))
    assert released_dispute_requirement_ids([plain], {"item-1": T0}, [later]) == {}


def test_release_names_every_newer_requirement_deterministically():
    newer = [_requirement(_iso(T0 + datetime.timedelta(seconds=s)), body=f"Decision {s}.")
             for s in (3, 1)]
    older = _requirement(_iso(T0 - datetime.timedelta(seconds=1)), body="Old.")
    released = released_dispute_requirement_ids([_disputed_item()], {"item-1": T0}, [*newer, older])
    assert released == {"item-1": tuple(sorted(r.requirement_id for r in newer))}


# --- unit: dispute time source ---------------------------------------------


def test_dispute_time_uses_the_newest_coder_record():
    comments = [
        _coder_comment(_iso(T0), ["item-1"], round_number=1),
        _coder_comment(_iso(T0 + datetime.timedelta(seconds=9)), ["item-1"], round_number=2),
    ]
    assert coder_dispute_times(comments) == {"item-1": T0 + datetime.timedelta(seconds=9)}


def test_dispute_time_is_unknown_without_a_parseable_created_at():
    assert coder_dispute_times([_coder_comment("garbage", ["item-1"])]) == {"item-1": None}
    assert coder_dispute_times([_coder_comment(None, ["item-1"])]) == {"item-1": None}


def test_dispute_time_is_unknown_when_no_record_lists_the_item():
    comments = [_coder_comment(_iso(T0), ["item-2"])]
    later = _requirement(_iso(T0 + datetime.timedelta(seconds=5)))
    release = round_dispute_release(comments, [_disputed_item()], [later])
    assert "item-1" not in release.dispute_times
    assert release.released == {}


def test_unpublished_invocation_redispute_is_unknown_not_the_older_time():
    """A snapshot predating this run's newest coder publication fails closed."""
    comments = [_coder_comment(_iso(T0), ["item-1"], round_number=1)]
    latest = _coder_metadata(["item-1"], round_number=2, subject="b" * 40)
    later = _requirement(_iso(T0 + datetime.timedelta(seconds=1)))
    release = round_dispute_release(comments, [_disputed_item()], [later], latest_coder=latest)
    assert release.dispute_times == {"item-1": None}
    assert release.released == {}
    # Once the publication is visible, its own time is used.
    published = [*comments, _coder_comment(_iso(T0 + datetime.timedelta(seconds=2)), ["item-1"],
                                           round_number=2, subject="b" * 40)]
    release = round_dispute_release(published, [_disputed_item()], [later], latest_coder=latest)
    assert release.dispute_times == {"item-1": T0 + datetime.timedelta(seconds=2)}
    assert release.released == {}


# --- unit: stop and notice -------------------------------------------------


def test_stop_skips_released_items_and_names_only_ids_and_times():
    item = _disputed_item()
    _raise_if_maintained_disputed_items([item], prior_items=[item], released_item_ids=("item-1",))
    with pytest.raises(HumanDecisionRequiredError) as excinfo:
        _raise_if_maintained_disputed_items([item], prior_items=[item],
                                            dispute_times={"item-1": T0})
    message = str(excinfo.value)
    assert "[item-1] disputed at 2026-05-23T00:00:02Z" in message
    for echoed in (ITEM_TEXT, EVIDENCE, REVIEWER_NOTE, CODER_DISPUTE_NOTE_PREFIX):
        assert echoed not in message
    assert "rerun" in message


def test_stop_notice_post_failure_preserves_the_decision_error():
    item = _disputed_item()
    posted, logged = [], []

    def failing_post(body):
        posted.append(body)
        raise AgentLoopError("GitHub is down")

    with pytest.raises(HumanDecisionRequiredError, match="Reviewer did not resolve 1 disputed"):
        stop_for_held_disputes(
            [item], prior_items=[item],
            release=DisputeRelease(dispute_times={"item-1": T0}),
            post_notice=failing_post, log_failure=logged.append,
        )
    assert len(posted) == 1
    body = posted[0]
    assert body.startswith("## Human decision required")
    assert "[item-1] disputed at 2026-05-23T00:00:02Z" in body
    for echoed in (ITEM_TEXT, EVIDENCE, REVIEWER_NOTE):
        assert echoed not in body
    assert logged and "GitHub is down" in logged[0]


def test_stop_does_not_post_for_released_disputes():
    item = _disputed_item()
    posted = []
    stop_for_held_disputes(
        [item], prior_items=[item],
        release=DisputeRelease(dispute_times={"item-1": T0}, released={"item-1": ("hr-x",)}),
        post_notice=posted.append, log_failure=posted.append,
    )
    assert posted == []


# --- unit: prompt pointer lines --------------------------------------------


@pytest.mark.parametrize(
    "render",
    [
        lambda items, released: format_coder_followup_context(items, released),
        lambda items, released: _format_same_pr_unresolved_items(items, released),
        lambda items, released: _format_unresolved_review_items(items, released),
    ],
    ids=["coder", "same-pr", "reviewer"],
)
def test_pointer_line_names_requirement_ids_without_requirement_text(render):
    status = "same-pr"
    items = [_disputed_item(status=status), _disputed_item("item-2", status=status)]
    released = {"item-1": ("hr-aaa", "hr-bbb")}
    rendered = render(items, released)
    pointers = [line for line in rendered.splitlines() if "is released by newer" in line]
    assert len(pointers) == 1
    assert "item-1" in pointers[0] and "hr-aaa, hr-bbb" in pointers[0]
    assert "signed human requirements section" in pointers[0]
    assert render(items, {}) == render(items, None)
    assert "is released by newer" not in render(items, None)


# --- PR-loop integration ---------------------------------------------------


def test_newer_signed_requirement_releases_dispute_on_rerun(tmp_path):
    """Rows release-newer and fresh-per-round."""
    runner = _dispute_runner()
    _stop_after_dispute(tmp_path, runner)
    requirements = _sign(runner)
    hr_ids = [requirement.requirement_id for requirement in requirements]
    runner.codex_outputs += [_still_blocking(requirements=requirements), _approve(requirements)]
    runner.claude_outputs.append(structured_coder_followup(
        addressed_items=["item-1"], summary="Applied the human decision.",
        human_requirement_ids=hr_ids,
    ))

    assert run_pr_loop(runner, pr_number=77, config=_config(tmp_path)) == 0

    coder_prompts = [cmd[-1] for cmd in _coder_calls(runner)]
    assert len(coder_prompts) == 2
    (pointer,) = _pointer_lines(coder_prompts[1])
    assert hr_ids[0] in pointer
    # Requirement text appears only in the signed-requirements section.
    assert coder_prompts[1].count(HUMAN_TEXT) == 1
    assert HUMAN_TEXT not in pointer
    # The release itself approves nothing: the reviewer blocked the released
    # item before the coder ran, and only a later reviewer approval resolved it.
    reviewer_prompts = [cmd[-1] for cmd in _reviewer_calls(runner)]
    assert len(reviewer_prompts) == 4
    assert _pointer_lines(reviewer_prompts[3])
    # Nothing about the release is persisted into the public records.
    assert not any("is released by newer" in c["body"] for c in runner.pr_payload["comments"])


def test_dispute_without_newer_requirement_still_stops(tmp_path):
    """Row hold-no-newer: an equal-time requirement does not release."""
    runner = _dispute_runner()
    _stop_after_dispute(tmp_path, runner)
    (coder_record,) = _coder_records(runner)
    requirements = _sign(runner, created_at=coder_record["createdAt"])
    runner.codex_outputs.append(_still_blocking(requirements=requirements))

    with pytest.raises(HumanDecisionRequiredError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path))

    assert f"[item-1] disputed at {coder_record['createdAt']}" in str(excinfo.value)
    assert len(_coder_calls(runner)) == 1
    assert _merge_calls(runner) == []


def test_dispute_with_older_requirement_still_stops(tmp_path):
    """Row hold-no-newer: a requirement from before the dispute does not release."""
    runner = _dispute_runner()
    _stop_after_dispute(tmp_path, runner)
    requirements = _sign(runner, created_at="2026-05-01T00:00:00Z")
    runner.codex_outputs.append(_still_blocking(requirements=requirements))

    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path))

    assert len(_coder_calls(runner)) == 1
    assert _merge_calls(runner) == []


def test_unknown_dispute_time_still_stops(tmp_path):
    """Row hold-unknown-time: no admitted coder record lists the item."""
    runner = _dispute_runner()
    _stop_after_dispute(tmp_path, runner)

    def drop_dispute(payload):
        if payload.get("role") != "coder":
            return None
        payload["raw_structured_coder_response"] = structured_coder_followup(
            remaining_items=["item-1"], summary="Legacy record without a dispute list."
        )
        return payload

    _rewrite_pr_metadata(runner, drop_dispute)
    requirements = _sign(runner, created_at="2026-06-01T00:00:00Z")
    runner.codex_outputs.append(_still_blocking(requirements=requirements))

    with pytest.raises(HumanDecisionRequiredError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path))

    # The dispute note is still carried, but no time can be proven.
    assert "[item-1] disputed at unknown" in str(excinfo.value)
    assert len(_coder_calls(runner)) == 1
    assert _merge_calls(runner) == []


def test_redispute_rearms_the_stop(tmp_path):
    """Row redispute-rearms: a re-dispute after release stops again."""
    runner = _dispute_runner()
    _stop_after_dispute(tmp_path, runner)
    requirements = _sign(runner)
    # Released round: the reviewer keeps blocking, the coder disputes again,
    # and the reviewer still blocks on reconsideration.
    runner.codex_outputs += [
        _still_blocking(requirements=requirements),
        _still_blocking(requirements=requirements),
    ]
    runner.claude_outputs.append(structured_coder_followup(
        disputed_items=["item-1"], dispute_evidence={"item-1": EVIDENCE},
        summary="Disputing item-1 again.",
        human_requirement_ids=[r.requirement_id for r in requirements],
    ))

    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path))
    assert len(_coder_calls(runner)) == 2

    first, second = _coder_records(runner)
    human = next(c for c in runner.pr_payload["comments"] if c["body"].endswith("-- Human Reviewer"))
    assert first["createdAt"] < human["createdAt"] < second["createdAt"]
    # A fresh rerun sees the re-dispute record at T+2: the older requirement
    # does not release it, so the stop fires before any coder dispatch.
    runner.codex_outputs.append(_still_blocking(requirements=requirements))
    with pytest.raises(HumanDecisionRequiredError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path))
    assert f"[item-1] disputed at {second['createdAt']}" in str(excinfo.value)
    assert len(_coder_calls(runner)) == 2
    assert _merge_calls(runner) == []


def test_released_dispute_does_not_block_merge_conflict_routing(tmp_path):
    """Row merge-conflict-released."""
    runner = _dispute_runner()
    _stop_after_dispute(tmp_path, runner)
    requirements = _sign(runner)
    hr_ids = [requirement.requirement_id for requirement in requirements]
    runner.mergeability_payloads = [dict(CONFLICTING)]
    runner.codex_outputs += [_still_blocking(requirements=requirements), _approve(requirements)]
    runner.claude_outputs.append(structured_coder_followup(
        addressed_items=["item-1"], summary="Merged main and applied the decision.",
        human_requirement_ids=hr_ids,
    ))
    # The reviewer still blocks after the merge-conflict round, so the coder
    # answers that review before the final approval.
    runner.claude_outputs.append(structured_coder_followup(
        addressed_items=["item-1"], summary="Addressed the remaining review item.",
        human_requirement_ids=hr_ids,
    ))

    assert run_pr_loop(runner, pr_number=77, config=_config(tmp_path)) == 0

    # Every scripted turn was used, so the run did not exit early.
    assert runner.codex_outputs == []
    assert runner.claude_outputs == []
    coder_prompts = [cmd[-1] for cmd in _coder_calls(runner)]
    # Dispute, merge conflict, follow-up.
    assert len(coder_prompts) == 3
    assert "has a merge conflict" in coder_prompts[1]
    (pointer,) = _pointer_lines(coder_prompts[1])
    assert hr_ids[0] in pointer
    assert coder_prompts[1].count(HUMAN_TEXT) == 1


def test_held_dispute_still_stops_before_merge_conflict_routing(tmp_path):
    """Contrast for merge-conflict-released: an unreleased dispute stops first."""
    runner = _dispute_runner()
    _stop_after_dispute(tmp_path, runner)
    runner.mergeability_payloads = [dict(CONFLICTING)]
    runner.codex_outputs.append(_still_blocking())

    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path))
    assert len(_coder_calls(runner)) == 1


@pytest.mark.parametrize("compact", [False, True], ids=["full", "compact"])
def test_review_prompt_carries_pointer_beside_the_released_item(tmp_path, compact):
    from coding_review_agent_loop.prompts import build_review_prompt

    item = _disputed_item()
    kwargs = dict(reviewer="codex", unresolved_items=[item], compact_context=compact)
    released = build_review_prompt(77, 2, make_config(tmp_path), released_disputes={"item-1": ("hr-aaa",)}, **kwargs)
    held = build_review_prompt(77, 2, make_config(tmp_path), **kwargs)
    (pointer,) = _pointer_lines(released)
    assert "hr-aaa" in pointer
    assert _pointer_lines(held) == []
    # The pointer sits inside item-1's entry, after its updates.
    entry = released.split("- [item-1]", 1)[1]
    assert entry.index(REVIEWER_NOTE) < entry.index(pointer)


def test_signed_answer_between_consecutive_rounds_releases_within_one_invocation(tmp_path):
    """Row fresh-per-round: the next round of the same run sees the coder's
    dispute record and a signed answer posted after it."""
    human = {
        "author": {"login": "maintainer"},
        "createdAt": "2026-05-24T00:00:00Z",
        "url": "https://github.com/OWNER/REPO/pull/77#issuecomment-2026",
        "body": f"{HUMAN_TEXT}\n\n-- Human Reviewer",
    }
    requirements = _parse_pr_human_requirements({"comments": [human], "reviews": []})
    hr_ids = [requirement.requirement_id for requirement in requirements]
    runner = FakeRunner(
        claude_outputs=[
            _dispute_output(),
            structured_coder_followup(addressed_items=["item-1"], summary="Applied the decision.",
                                      human_requirement_ids=hr_ids),
        ],
        codex_outputs=[
            structured_pr_review(state="blocking", summary="Pricing constant is wrong.",
                                 blocking_items=[ITEM_TEXT]),
            _still_blocking(requirements=requirements),
            _approve(requirements),
        ],
    )
    original_run = runner.run

    def post_answer_after_dispute(args, *, cwd, **kwargs):
        result = original_run(args, cwd=cwd, **kwargs)
        command = [str(arg) for arg in args]
        if command[:3] == ["gh", "pr", "comment"] and _coder_records(runner) and human not in runner.pr_payload["comments"]:
            runner.pr_payload["comments"].append(human)
        return result

    runner.run = post_answer_after_dispute

    assert run_pr_loop(runner, pr_number=77, config=_config(tmp_path)) == 0

    (coder_record,) = [c for c in _coder_records(runner) if "Disputing item-1" in c["body"]]
    assert coder_record["createdAt"] < human["createdAt"]
    coder_prompts = [cmd[-1] for cmd in _coder_calls(runner)]
    assert len(coder_prompts) == 2
    (pointer,) = _pointer_lines(coder_prompts[1])
    assert hr_ids[0] in pointer
    assert coder_prompts[1].count(HUMAN_TEXT) == 1
    assert _pointer_lines([cmd[-1] for cmd in _reviewer_calls(runner)][1])
