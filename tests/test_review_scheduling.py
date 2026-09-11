import pytest

from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.cli import build_parser
from coding_review_agent_loop.review_scheduling import (
    GitChange,
    ReviewObligation,
    ReviewSchedulingContract,
    SchedulerSnapshot,
    classify_transition,
    select_reviewers,
)
from coding_review_agent_loop.protocol import ReviewItemDisposition, UnresolvedReviewItem
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    _decode_round_metadata_mapping,
    _encode_round_metadata,
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
