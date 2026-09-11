import pytest

from agent_loop_helpers import FakeRunner, make_config

from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.cli import build_parser
from coding_review_agent_loop.review_scheduling import (
    GitChange,
    ReviewObligation,
    ReviewSchedulingContract,
    SchedulerSnapshot,
    TransitionClassification,
    classify_transition,
    select_reviewers,
)
from coding_review_agent_loop.protocol import ReviewItemDisposition, UnresolvedReviewItem
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    _attach_round_metadata,
    _decode_round_metadata_mapping,
    _encode_round_metadata,
)
from coding_review_agent_loop.orchestrator import (
    _all_pending_resolution_owners_unavailable,
    _fresh_pr_qualification_snapshot,
    _reviewer_history_is_reconstructible,
    _reviewer_needs_fresh_context,
    _reviewer_diff_summary,
)
from coding_review_agent_loop.round_transport import decode_mapping
from coding_review_agent_loop.unresolved_items import _apply_unresolved_item_dispositions


def _contract() -> ReviewSchedulingContract:
    return ReviewSchedulingContract(
        required_reviewers=("Claude", "Codex", "Antigravity"),
        policy="selective-intermediate",
        broad_rules=(".github/**",),
    )


def _item(*, owners=("Codex",), scope=("src/worker.py",), states=()):
    return UnresolvedReviewItem(
        item_id="item-1",
        reviewer="Codex",
        source_round=1,
        text="worker cleanup is incomplete",
        status="blocking",
        source_status="blocking",
        fix_scope=scope,
        resolution_owners=owners,
        owner_states=states or tuple((owner, "pending") for owner in owners),
    )


def test_exact_scoped_text_change_is_narrow_and_boundary_safe():
    classification = classify_transition(
        "a" * 40,
        "b" * 40,
        [GitChange("src/worker.py", status="M")],
        scopes=("src/worker.py",),
        broad_rules=("src/worker.pyx",),
    )
    assert classification.narrow

    outside = classify_transition(
        "a" * 40,
        "b" * 40,
        [GitChange("src/worker.pyx", status="M")],
        scopes=("src/worker.py",),
        broad_rules=("**/never/**",),
    )
    assert outside.broad


@pytest.mark.parametrize(
    "change",
    [
        GitChange("src/worker.py", status="D"),
        GitChange("src/worker.py", status="R"),
        GitChange("src/worker.py", status="M", binary=True),
        GitChange("src/worker.py", status="M", mode_changed=True),
    ],
)
def test_non_text_or_non_ordinary_changes_force_broad(change):
    result = classify_transition(
        "a" * 40,
        "b" * 40,
        [change],
        scopes=("src/worker.py",),
        broad_rules=("**/never/**",),
    )
    assert result.broad


def test_unavailable_history_and_unreconstructible_obligation_are_broad():
    assert classify_transition(
        "a" * 40,
        "b" * 40,
        [GitChange("src/worker.py")],
        scopes=("src/worker.py",),
        broad_rules=("**/never/**",),
        ancestor=False,
    ).broad
    obligation = ReviewObligation(
        item_id="item-1",
        status="blocking",
        scope=None,
        resolution_owners=("Codex",),
        pending_owners=("Codex",),
    )
    assert classify_transition(
        "a" * 40,
        "b" * 40,
        [GitChange("src/worker.py")],
        scopes=("src/worker.py",),
        broad_rules=("**/never/**",),
        obligations=(obligation,),
    ).broad


def test_selective_board_pauses_approvals_then_sweeps_missing_exact_head_approvals():
    snapshot = SchedulerSnapshot(
        previous_sha="a" * 40,
        current_sha="b" * 40,
        contract=_contract(),
        obligations=(
            ReviewObligation(
                item_id="item-1",
                status="blocking",
                scope=("src/worker.py",),
                resolution_owners=("Codex",),
                pending_owners=("Codex",),
            ),
        ),
    )
    narrow = classify_transition(
        snapshot.previous_sha,
        snapshot.current_sha,
        [GitChange("src/worker.py")],
        scopes=("src/worker.py",),
        broad_rules=("**/never/**",),
        obligations=snapshot.obligations,
    )
    intermediate = select_reviewers(snapshot, narrow)
    assert intermediate.selected_reviewers == ("Codex",)
    assert {name for name, _reason in intermediate.paused_reviewers} == {"Claude", "Antigravity"}
    assert intermediate.calls_avoided == 2

    sweep = select_reviewers(
        SchedulerSnapshot(
            previous_sha=snapshot.current_sha,
            current_sha=snapshot.current_sha,
            contract=snapshot.contract,
            obligations=(),
        ),
        narrow,
        qualifying_approvals=("Codex",),
        final_sweep=True,
    )
    assert sweep.selected_reviewers == ("Claude", "Antigravity")
    assert sweep.calls_avoided == 0
    assert dict(intermediate.paused_reviewers)["Claude"].startswith(
        "selective intermediate pause"
    )


def test_pause_reasons_distinguish_approval_carries_and_unavailable_reviewers():
    snapshot = SchedulerSnapshot(
        previous_sha="a" * 40,
        current_sha="b" * 40,
        contract=_contract(),
        obligations=(
            ReviewObligation(
                item_id="item-1",
                status="blocking",
                scope=("src/worker.py",),
                resolution_owners=("Codex",),
                pending_owners=("Codex",),
            ),
        ),
    )
    decision = select_reviewers(
        snapshot,
        TransitionClassification("narrow", "scoped fix"),
        qualifying_approvals=("Claude",),
        unavailable_reviewers=("Antigravity",),
    )
    paused = dict(decision.paused_reviewers)
    assert "exact-head approval" in paused["Claude"]
    assert "unavailable" in paused["Antigravity"]


def test_broad_transition_does_not_latch_future_narrow_transitions():
    item_obligation = ReviewObligation(
        item_id="item-1",
        status="blocking",
        scope=("src/worker.py",),
        resolution_owners=("Codex",),
        pending_owners=("Codex",),
    )
    broad_snapshot = SchedulerSnapshot(
        previous_sha="a" * 40,
        current_sha="b" * 40,
        contract=_contract(),
        obligations=(item_obligation,),
    )
    broad = select_reviewers(
        broad_snapshot,
        TransitionClassification("broad", "dependency change"),
    )
    assert broad.selected_reviewers == ("Claude", "Codex", "Antigravity")
    narrow = select_reviewers(
        SchedulerSnapshot(
            previous_sha="b" * 40,
            current_sha="c" * 40,
            contract=broad_snapshot.contract,
            obligations=(item_obligation,),
            force_full=False,
        ),
        TransitionClassification("narrow", "scoped fix"),
    )
    assert narrow.selected_reviewers == ("Codex",)


def test_owner_scoped_reconciliation_requires_each_owner_to_clear():
    item = _item()
    blocking, _ = _apply_unresolved_item_dispositions(
        (item,),
        {
            "item-1": [
                ReviewItemDisposition(
                    item_id="item-1", reviewer="Claude", disposition="blocking", note="still fails"
                )
            ]
        },
        reconciliation_mode="owner-scoped",
    )
    assert blocking[0].resolution_owners == ("Codex", "Claude")

    original_owner_only, _ = _apply_unresolved_item_dispositions(
        tuple(blocking),
        {
            "item-1": [
                ReviewItemDisposition(
                    item_id="item-1", reviewer="Codex", disposition="resolved"
                )
            ]
        },
        reconciliation_mode="owner-scoped",
    )
    assert original_owner_only[0].owner_states == (("Codex", "cleared"), ("Claude", "pending"))

    cleared, _ = _apply_unresolved_item_dispositions(
        tuple(original_owner_only),
        {
            "item-1": [
                ReviewItemDisposition(
                    item_id="item-1", reviewer="Claude", disposition="resolved"
                )
            ]
        },
        reconciliation_mode="owner-scoped",
    )
    assert cleared == []


def test_non_owner_resolution_is_evidence_only():
    remaining, _ = _apply_unresolved_item_dispositions(
        (_item(),),
        {
            "item-1": [
                ReviewItemDisposition(
                    item_id="item-1", reviewer="Claude", disposition="resolved", note="looks good"
                )
            ]
        },
        reconciliation_mode="owner-scoped",
    )
    assert remaining[0].owner_states == (("Codex", "pending"),)
    assert "Claude: looks good" in remaining[0].notes


def test_scheduler_metadata_roundtrips_and_malformed_state_falls_back_to_legacy():
    metadata = PostedRoundMetadata(
        flow="pr",
        role="summary",
        agent="Orchestrator",
        round_number=2,
        subject="b" * 40,
        scheduler_contract=_contract().as_dict(),
        scheduler_previous_sha="a" * 40,
        scheduler_current_sha="b" * 40,
        scheduler_obligation_digest="0" * 16,
        scheduler_selected_reviewers=("Codex",),
        scheduler_paused_reviewers=(("Claude", "narrow transition"), ("Antigravity", "narrow transition")),
        scheduler_reasons=("narrow transition",),
        scheduler_final_sweep=False,
        scheduler_force_full=False,
        scheduler_calls_avoided=2,
    )
    decoded = _decode_round_metadata_mapping(
        decode_mapping(_encode_round_metadata(metadata))
    )
    assert decoded.scheduler_contract == _contract().as_dict()
    assert decoded.scheduler_selected_reviewers == ("Codex",)
    assert decoded.scheduler_calls_avoided == 2
    assert decoded.scheduler_metadata_status == "valid"

    malformed = _decode_round_metadata_mapping(
        {
            "flow": "pr",
            "role": "summary",
            "agent": "Orchestrator",
            "round_number": 2,
            "subject": "b" * 40,
            "scheduler_contract": {"policy": "selective-intermediate"},
            **{
                key: value
                for key, value in {
                    "scheduler_previous_sha": None,
                    "scheduler_current_sha": "b" * 40,
                    "scheduler_obligation_digest": "0" * 16,
                    "scheduler_selected_reviewers": [],
                    "scheduler_paused_reviewers": [],
                    "scheduler_reasons": ["bad"],
                    "scheduler_final_sweep": False,
                    "scheduler_force_full": False,
                    "scheduler_calls_avoided": 0,
                }.items()
            },
        }
    )
    assert malformed.scheduler_contract is None
    assert malformed.scheduler_metadata_status == "invalid"

    legacy = _decode_round_metadata_mapping(
        {
            "flow": "pr",
            "role": "reviewer",
            "agent": "Codex",
            "round_number": 1,
            "subject": "a" * 40,
        }
    )
    assert legacy.scheduler_metadata_status == "absent"


def test_coder_scheduler_checkpoint_is_a_complete_decodable_record():
    metadata = PostedRoundMetadata(
        flow="pr",
        role="coder",
        agent="Claude",
        round_number=3,
        subject="c" * 40,
        scheduler_contract=_contract().as_dict(),
        scheduler_previous_sha="b" * 40,
        scheduler_current_sha="c" * 40,
        scheduler_obligation_digest="1" * 16,
        scheduler_selected_reviewers=("Codex",),
        scheduler_paused_reviewers=(
            ("Claude", "qualifying exact-head approval carried; no new turn needed"),
            ("Antigravity", "qualifying exact-head approval carried; no new turn needed"),
        ),
        scheduler_reasons=("narrow transition: pending resolution owners and co-owners", "scoped fix"),
        scheduler_final_sweep=False,
        scheduler_force_full=False,
        scheduler_calls_avoided=2,
    )
    decoded = _decode_round_metadata_mapping(
        decode_mapping(_encode_round_metadata(metadata))
    )
    assert decoded.scheduler_contract == metadata.scheduler_contract
    assert decoded.scheduler_selected_reviewers == ("Codex",)
    assert decoded.scheduler_final_sweep is False
    assert decoded.scheduler_calls_avoided == 2


def test_returning_reviewer_uses_latest_stale_record_not_exact_head_approval():
    stale = PostedRoundMetadata(
        flow="pr",
        role="reviewer",
        agent="Claude",
        round_number=1,
        subject="a" * 40,
        state="approved",
    )
    record = type("Record", (), {"metadata": stale})()
    assert _reviewer_needs_fresh_context(
        "codex",
        selective_policy=True,
        current_head_sha="b" * 40,
        current_round=2,
        latest_reviewer_records={"Claude": record},
    ) is True
    assert _reviewer_needs_fresh_context(
        "claude",
        selective_policy=True,
        current_head_sha="b" * 40,
        current_round=2,
        latest_reviewer_records={"Claude": record},
    ) is True


def test_reviewer_diff_summary_is_bounded_and_observed():
    class Runner:
        def run(self, args, *, cwd, check=False):
            return type("Result", (), {"returncode": 0, "stdout": "M\tsrc/worker.py\n"})()

    summary = _reviewer_diff_summary(
        Runner(),
        checkout=".",
        last_reviewed_sha="a" * 40,
        current_head_sha="b" * 40,
    )
    assert "src/worker.py" in summary
    assert "complete base-to-head diff" in summary


def test_returning_reviewer_history_can_span_multiple_narrow_heads_when_git_observes_it():
    class Runner:
        def __init__(self, *, ancestry=0, diff=0):
            self.ancestry = ancestry
            self.diff = diff

        def run(self, args, *, cwd, check=False):
            return type(
                "Result",
                (),
                {
                    "returncode": self.ancestry if args[1] == "merge-base" else self.diff,
                    "stdout": "",
                },
            )()

    record = type(
        "Record",
        (),
        {"metadata": PostedRoundMetadata(
            flow="pr", role="reviewer", agent="Claude", round_number=1, subject="h1"
        )},
    )()
    assert _reviewer_history_is_reconstructible(
        Runner(), checkout=".", record=record, current_head_sha="h3"
    )
    assert not _reviewer_history_is_reconstructible(
        Runner(ancestry=1), checkout=".", record=record, current_head_sha="h3"
    )
    assert not _reviewer_history_is_reconstructible(
        Runner(diff=1), checkout=".", record=record, current_head_sha="h3"
    )
    assert not _reviewer_history_is_reconstructible(
        Runner(), checkout=".", record=None, current_head_sha="h3"
    )


def test_fresh_qualification_rejects_a_scheduler_contract_change(tmp_path):
    changed_contract = ReviewSchedulingContract(
        required_reviewers=("Claude", "Codex", "Antigravity"),
        policy="all-reviewers",
        broad_rules=(".github/**",),
    )
    audit = _attach_round_metadata(
        "scheduler audit",
        PostedRoundMetadata(
            flow="pr",
            role="summary",
            agent="Orchestrator",
            round_number=2,
            subject="head",
            scheduler_contract=changed_contract.as_dict(),
            scheduler_previous_sha="base",
            scheduler_current_sha="head",
            scheduler_obligation_digest="0" * 16,
            scheduler_selected_reviewers=("Claude", "Codex", "Antigravity"),
            scheduler_paused_reviewers=(),
            scheduler_reasons=("contract changed",),
            scheduler_final_sweep=False,
            scheduler_force_full=False,
            scheduler_calls_avoided=0,
        ),
    )
    runner = FakeRunner(
        pr_payload={
            "headRefOid": "head",
            "comments": [{"author": {"login": "bot"}, "body": audit}],
        }
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini", "antigravity"),
        pr_review_policy="selective-intermediate",
    )
    with pytest.raises(AgentLoopError, match="scheduler contract changed"):
        _fresh_pr_qualification_snapshot(
            runner,
            config=config,
            pr_number=77,
            issue_context=None,
            parent_issue_context=None,
            scheduler_contract=_contract(),
        )


def test_cleared_owner_set_is_not_vacuously_unavailable():
    item = _item(owners=("Codex",), states=(("Codex", "cleared"),))
    assert not _all_pending_resolution_owners_unavailable(item, {"Codex"})
    pending = _item(owners=("Codex",), states=(("Codex", "pending"),))
    assert _all_pending_resolution_owners_unavailable(pending, {"Codex"})


def test_fix_scope_rejects_ambiguous_paths():
    from coding_review_agent_loop.review_scheduling import normalize_fix_scope

    for value in ([], ["../src/a.py"], ["src/*.py"], ["/src/a.py"], ["src/a.py", "src/a.py"]):
        with pytest.raises(AgentLoopError):
            normalize_fix_scope(value)


def test_pr_scheduler_options_are_explicit_and_repeatable():
    args = build_parser().parse_args(
        [
            "pr",
            "769",
            "--pr-review-policy",
            "selective-intermediate",
            "--pr-review-broad-rule",
            "src/generated/**",
            "--pr-review-broad-rule",
            "config.py",
            "--pr-review-force-full",
        ]
    )
    assert args.pr_review_policy == "selective-intermediate"
    assert args.pr_review_broad_rules == ["src/generated/**", "config.py"]
    assert args.pr_review_force_full is True
