import dataclasses
from types import SimpleNamespace

import pytest

from agent_loop_helpers import FakeRunner, make_config

from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.cli import build_parser
from coding_review_agent_loop.review_scheduling import (
    OPERATOR_FORCE_FULL_REASON,
    POST_PANEL_PREFIX,
    STRICT_PRE_PANEL_PREFIX,
    GitChange,
    PrePanelSafetyError,
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
    _prior_item_ledger_signature,
)
from coding_review_agent_loop.orchestrator import (
    _all_pending_resolution_owners_unavailable,
    _fresh_pr_qualification_snapshot,
    _observe_pr_transition,
    _partition_unresolved_items,
    _round_limit_diagnostic,
    _reviewer_history_is_reconstructible,
    _reviewer_needs_fresh_context,
    _reviewer_diff_summary,
)
import coding_review_agent_loop.orchestrator as orchestrator
from coding_review_agent_loop.architecture_context import ArchitectureSnapshot
from coding_review_agent_loop.round_transport import decode_mapping
from coding_review_agent_loop.unresolved_items import (
    _advance_machine_obligations_for_head,
    _apply_unresolved_item_dispositions,
    _next_unresolved_item,
)


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


def test_observer_allows_git_create_mode_for_an_ordinary_scoped_text_addition():
    class GitObserver:
        def run(self, args, *, cwd, check=False):
            if args[:3] == ["git", "merge-base", "--is-ancestor"]:
                return type("Result", (), {"returncode": 0, "stdout": ""})()
            if args[:3] == ["git", "diff", "--name-status"]:
                return type(
                    "Result", (), {"returncode": 0, "stdout": "A\0src/new_worker.py\0"}
                )()
            if args[:3] == ["git", "diff", "--numstat"]:
                return type(
                    "Result", (), {"returncode": 0, "stdout": "12\t0\tsrc/new_worker.py\n"}
                )()
            if args[:3] == ["git", "diff", "--summary"]:
                return type(
                    "Result",
                    (),
                    {"returncode": 0, "stdout": " create mode 100644 src/new_worker.py\n"},
                )()
            raise AssertionError(f"unexpected Git probe: {args!r}")

    result = _observe_pr_transition(
        GitObserver(),
        checkout=".",
        previous_sha="a" * 40,
        current_sha="b" * 40,
        scopes=("src/new_worker.py",),
        broad_rules=(".github/**",),
    )

    assert result.narrow
    assert result.changed_paths == ("src/new_worker.py",)


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


def test_owner_scoped_future_clears_sole_owner_and_retains_future_item():
    remaining, future_items = _apply_unresolved_item_dispositions(
        (_item(),),
        {
            "item-1": [
                ReviewItemDisposition(
                    item_id="item-1",
                    reviewer="Codex",
                    disposition="future",
                    note="Defer this non-blocking cleanup.",
                )
            ]
        },
        reconciliation_mode="owner-scoped",
    )

    assert len(remaining) == 1
    assert remaining[0].status == "future"
    assert remaining[0].owner_states == (("Codex", "cleared"),)
    assert future_items == []


def test_owner_scoped_future_and_resolved_clear_all_owners():
    item = _item(owners=("Codex", "Claude"))
    remaining, future_items = _apply_unresolved_item_dispositions(
        (item,),
        {
            "item-1": [
                ReviewItemDisposition(
                    item_id="item-1", reviewer="Codex", disposition="future"
                ),
                ReviewItemDisposition(
                    item_id="item-1", reviewer="Claude", disposition="resolved"
                ),
            ]
        },
        reconciliation_mode="owner-scoped",
    )

    assert remaining[0].status == "future"
    assert remaining[0].owner_states == (
        ("Codex", "cleared"),
        ("Claude", "cleared"),
    )
    assert future_items == []


def test_owner_scoped_future_does_not_clear_another_pending_owner():
    item = _item(owners=("Codex", "Claude"))
    remaining, future_items = _apply_unresolved_item_dispositions(
        (item,),
        {
            "item-1": [
                ReviewItemDisposition(
                    item_id="item-1", reviewer="Codex", disposition="future"
                )
            ]
        },
        reconciliation_mode="owner-scoped",
    )

    assert remaining[0].status == "blocking"
    assert remaining[0].owner_states == (
        ("Codex", "cleared"),
        ("Claude", "pending"),
    )
    assert future_items == []


def test_owner_scoped_future_survives_until_other_owner_clears_in_later_round():
    item = _item(owners=("Codex", "Claude"))
    first_round, future_items = _apply_unresolved_item_dispositions(
        (item,),
        {
            "item-1": [
                ReviewItemDisposition(
                    item_id="item-1", reviewer="Codex", disposition="future"
                )
            ]
        },
        reconciliation_mode="owner-scoped",
    )

    assert future_items == []
    assert first_round[0].status == "blocking"
    assert first_round[0].owner_dispositions == (("Codex", "future"),)

    second_round, future_items = _apply_unresolved_item_dispositions(
        tuple(first_round),
        {
            "item-1": [
                ReviewItemDisposition(
                    item_id="item-1", reviewer="Claude", disposition="resolved"
                )
            ]
        },
        reconciliation_mode="owner-scoped",
    )

    assert len(second_round) == 1
    assert second_round[0].status == "future"
    assert second_round[0].owner_states == (
        ("Codex", "cleared"),
        ("Claude", "cleared"),
    )
    assert second_round[0].owner_dispositions == (
        ("Codex", "future"),
        ("Claude", "resolved"),
    )
    assert future_items == []


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


def test_fresh_qualification_rejects_scheduler_metadata_after_recovery_boundary(tmp_path):
    valid_reconciliation = _attach_round_metadata(
        "completed full-board reconciliation",
        PostedRoundMetadata(
            flow="pr",
            role="summary",
            agent="Orchestrator",
            round_number=2,
            subject="head",
            phase="reconciliation",
            scheduler_contract=_contract().as_dict(),
            scheduler_previous_sha=None,
            scheduler_current_sha="head",
            scheduler_obligation_digest="0" * 16,
            scheduler_selected_reviewers=("Claude", "Codex", "Antigravity"),
            scheduler_paused_reviewers=(),
            scheduler_reasons=("full board",),
            scheduler_final_sweep=False,
            scheduler_force_full=False,
            scheduler_calls_avoided=0,
        ),
    )
    malformed_after_recovery = _attach_round_metadata(
        "new malformed scheduler checkpoint",
        PostedRoundMetadata(
            flow="pr",
            role="summary",
            agent="Orchestrator",
            round_number=3,
            subject="head",
            scheduler_contract={"policy": "selective-intermediate"},
        ),
    )
    runner = FakeRunner(
        pr_payload={
            "headRefOid": "head",
            "comments": [
                {"author": {"login": "bot"}, "body": valid_reconciliation},
                {"author": {"login": "bot"}, "body": malformed_after_recovery},
            ],
        }
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini", "antigravity"),
        pr_review_policy="selective-intermediate",
    )

    with pytest.raises(AgentLoopError, match="Malformed or contradictory PR review scheduler metadata"):
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


def test_architecture_revalidation_schedules_once_for_changed_identity(tmp_path, monkeypatch):
    old = ArchitectureSnapshot(
        repository="OWNER/REPO", path="ARCHITECTURE.md", revision="a" * 40,
        blob_oid="b" * 40, sha256="c" * 64, availability="available",
        size=10, content="# Old\n",
    )
    new = ArchitectureSnapshot(
        repository="OWNER/REPO", path="ARCHITECTURE.md", revision="d" * 40,
        blob_oid="e" * 40, sha256="f" * 64, availability="available",
        size=12, content="# New\n",
    )
    config = make_config(tmp_path, architecture_context=old, architecture_context_enabled=True)
    metadata = SimpleNamespace(base_branch="main", head_sha="head")
    monkeypatch.setattr(
        orchestrator,
        "_freeze_prompt_architecture",
        lambda runner, config, **kwargs: dataclasses.replace(
            config, architecture_context=new
        ),
    )

    fresh, changed = orchestrator._revalidate_pr_architecture_identity(
        FakeRunner(), config=config, metadata=metadata, stored_identity=old.identity()
    )
    assert fresh == new and changed is True
    _fresh, repeated = orchestrator._revalidate_pr_architecture_identity(
        FakeRunner(), config=config, metadata=metadata, stored_identity=new.identity()
    )
    assert repeated is False


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


def test_selective_machine_ci_obligation_survives_reviewer_approval_until_new_head():
    item = _next_unresolved_item(
        item_number=30,
        reviewer="GitHub managed exact-head CI",
        source_round=13,
        text="Managed exact-head CI failed.",
        status="blocking",
        authority="machine",
        obligation_kind="managed-exact-head-ci",
        lifecycle="repair_required",
        failed_head_sha="oldhead123",
    )
    dispositions = {
        item.item_id: [
            ReviewItemDisposition(item.item_id, reviewer, "resolved")
            for reviewer in ("Claude", "Antigravity", "Codex")
        ]
    }

    same_head, _ = _apply_unresolved_item_dispositions(
        [item], dispositions, reconciliation_mode="owner-scoped"
    )
    assert len(same_head) == 1
    assert same_head[0].lifecycle == "repair_required"
    assert same_head[0].authority == "machine"

    candidate = _advance_machine_obligations_for_head(
        same_head, current_head_sha="newhead123"
    )
    assert candidate[0].lifecycle == "awaiting_current_head_review"
    assert candidate[0].candidate_head_sha == "newhead123"
    reviewed_candidate, _ = _apply_unresolved_item_dispositions(
        candidate, dispositions, reconciliation_mode="owner-scoped"
    )

    assert len(reviewed_candidate) == 1
    assert reviewed_candidate[0].failed_head_sha == "oldhead123"
    assert reviewed_candidate[0].candidate_head_sha == "newhead123"
    partitions = _partition_unresolved_items(
        reviewed_candidate, current_head_sha="newhead123"
    )
    assert partitions["coder_blockers"] == ()
    assert partitions["revalidation_candidates"] == tuple(reviewed_candidate)
    diagnostic = _round_limit_diagnostic(
        pr_number=77,
        round_number=20,
        items=reviewed_candidate,
        current_head_sha="newhead123",
    )
    assert "reviewed correction awaiting authoritative qualification" in diagnostic
    assert "reviewer" not in diagnostic.lower()


def test_machine_authority_upgrade_changes_scheduler_digest_and_forces_full_board():
    legacy = UnresolvedReviewItem(
        item_id="item-30",
        reviewer="GitHub managed exact-head CI",
        source_round=13,
        text="Managed exact-head CI failed.",
        status="blocking",
        source_status="blocking",
    )
    promoted = _next_unresolved_item(
        item_number=30,
        reviewer="GitHub managed exact-head CI",
        source_round=13,
        text="Managed exact-head CI failed.",
        status="blocking",
        authority="machine",
        obligation_kind="managed-exact-head-ci",
        lifecycle="repair_required",
        failed_head_sha="oldhead123",
    )

    assert _prior_item_ledger_signature((legacy,)) != _prior_item_ledger_signature((promoted,))
    decision = select_reviewers(
        SchedulerSnapshot(
            previous_sha="a" * 40,
            current_sha="b" * 40,
            contract=_contract(),
            obligations=(),
            force_full=True,
        ),
        TransitionClassification("narrow", "same scoped change"),
    )
    assert decision.selected_reviewers == _contract().required_reviewers


@pytest.mark.parametrize(
    "kind",
    ["alembic-migration", "merge-conflict", "human-requirements-acknowledgement"],
)
def test_non_ci_machine_obligations_remain_coder_blockers_on_a_new_head(kind):
    item = _next_unresolved_item(
        item_number=30,
        reviewer="Orchestrator",
        source_round=13,
        text=f"{kind} requires authoritative validation.",
        status="blocking",
        authority="machine",
        obligation_kind=kind,
        lifecycle="awaiting_current_head_review",
        failed_head_sha="oldhead123" if kind != "human-requirements-acknowledgement" else None,
        candidate_head_sha="newhead123",
    )

    partitions = _partition_unresolved_items(
        [item], current_head_sha="newhead123"
    )

    assert partitions["revalidation_candidates"] == ()
    assert partitions["coder_blockers"] == (item,)


def test_machine_obligation_revert_to_failed_head_returns_to_repair_required():
    item = _next_unresolved_item(
        item_number=30,
        reviewer="GitHub managed exact-head CI",
        source_round=13,
        text="Managed exact-head CI failed.",
        status="blocking",
        authority="machine",
        obligation_kind="managed-exact-head-ci",
        lifecycle="qualification_ready",
        failed_head_sha="oldhead123",
        candidate_head_sha="newhead123",
    )

    reverted = _advance_machine_obligations_for_head(
        [item], current_head_sha="oldhead123"
    )

    assert reverted[0].lifecycle == "repair_required"
    assert reverted[0].candidate_head_sha is None


# ---------------------------------------------------------------------------
# primary-then-panel staged policy (#810)
# ---------------------------------------------------------------------------


def _primary_contract() -> ReviewSchedulingContract:
    return ReviewSchedulingContract(
        required_reviewers=("Codex", "Gemini", "Antigravity"),
        policy="primary-then-panel",
        primary_reviewer="Codex",
        broad_rules=(".github/**",),
    )


def _obligation(item_id="item-1", owners=("Gemini",), scope=("src/worker.py",)):
    return ReviewObligation(
        item_id=item_id,
        status="blocking",
        scope=scope,
        resolution_owners=owners,
        pending_owners=owners,
    )


def _snapshot(
    *,
    obligations=(),
    phase=None,
    previous="a" * 40,
    current="b" * 40,
    force_full=False,
    operator_force_full=False,
    panel_evidence=False,
    fallback_reasons=(),
):
    return SchedulerSnapshot(
        previous_sha=previous,
        current_sha=current,
        contract=_primary_contract(),
        obligations=obligations,
        force_full=force_full,
        phase=phase,
        operator_force_full=operator_force_full,
        panel_evidence=panel_evidence,
        fallback_reasons=fallback_reasons,
    )


NARROW = TransitionClassification("narrow", "scoped fix")
BROAD = TransitionClassification("broad", "diff path outside obligation scopes")


def test_primary_contract_decoding_and_drift_rules():
    # Absent persisted primary decodes as None and is valid for existing policies.
    legacy = ReviewSchedulingContract.from_mapping(
        {
            "required_reviewers": ["Codex", "Gemini"],
            "policy": "selective-intermediate",
            "broad_rules": [".github/**"],
        }
    )
    assert legacy.primary_reviewer is None
    # The staged policy requires a primary that is a board member and a secondary.
    with pytest.raises(AgentLoopError, match="member"):
        ReviewSchedulingContract.from_mapping(
            {
                "required_reviewers": ["Codex", "Gemini"],
                "policy": "primary-then-panel",
                "primary_reviewer": "Claude",
                "broad_rules": [],
            }
        )
    with pytest.raises(AgentLoopError, match="at least one secondary"):
        ReviewSchedulingContract(
            required_reviewers=("Codex",), policy="primary-then-panel", primary_reviewer="Codex"
        )
    with pytest.raises(AgentLoopError, match="only be configured with primary-then-panel"):
        ReviewSchedulingContract(
            required_reviewers=("Codex", "Gemini"),
            policy="selective-intermediate",
            primary_reviewer="Codex",
        )
    # Persisted/configured primary drift is contract inequality (fails closed in the loop).
    persisted = ReviewSchedulingContract.from_mapping(
        {**_primary_contract().as_dict(), "primary_reviewer": "Gemini"}
    )
    assert persisted != _primary_contract()
    assert ReviewSchedulingContract.from_mapping(_primary_contract().as_dict()) == _primary_contract()


def test_policy_capabilities_are_named_and_existing_policies_unchanged():
    from coding_review_agent_loop.review_scheduling import policy_capabilities

    compat = policy_capabilities("all-reviewers")
    assert not compat.scheduler_enabled
    assert not compat.owner_scoped_reconciliation
    assert not compat.counts_avoided_calls
    assert not compat.recovery_latches_force_full
    selective = policy_capabilities("selective-intermediate")
    assert selective.scheduler_enabled and selective.owner_scoped_reconciliation
    assert selective.selective_pausing and selective.counts_avoided_calls
    assert not selective.phase_aware and not selective.requires_primary
    assert not selective.recovery_latches_force_full
    staged = policy_capabilities("primary-then-panel")
    assert staged.scheduler_enabled and staged.owner_scoped_reconciliation
    assert staged.selective_pausing and staged.counts_avoided_calls
    assert staged.phase_aware and staged.requires_primary
    assert staged.recovery_latches_force_full
    with pytest.raises(AgentLoopError):
        policy_capabilities("unknown")


def test_primary_blocking_loop_keeps_only_primary_until_exact_head_approval():
    initial = select_reviewers(
        _snapshot(previous=None), TransitionClassification("broad", "initial candidate")
    )
    assert initial.selected_reviewers == ("Codex",)
    assert initial.phase == "primary"
    assert initial.calls_avoided == 2
    assert dict(initial.paused_reviewers)["Gemini"].startswith("primary phase")
    # The first review is not a fallback: no strict pre-panel prefix.
    assert not initial.reason.startswith(STRICT_PRE_PANEL_PREFIX)

    # Primary blocked, coder made a narrow fix: still the primary phase, not remediation.
    recheck = select_reviewers(
        _snapshot(obligations=(_obligation(owners=("Codex",)),), phase="primary"), NARROW
    )
    assert recheck.selected_reviewers == ("Codex",)
    assert recheck.phase == "primary"
    assert "primary rechecks" in recheck.reason

    # Approval from an older head never carries: the primary is re-selected.
    stale = select_reviewers(_snapshot(phase="primary"), NARROW, qualifying_approvals=())
    assert stale.selected_reviewers == ("Codex",)
    assert stale.phase == "primary"


def test_primary_approval_opens_independent_secondary_audit_not_final_sweep():
    audit = select_reviewers(
        _snapshot(previous="b" * 40, phase="primary"),
        TransitionClassification("narrow", "same exact candidate head"),
        qualifying_approvals=("Codex",),
        final_sweep=True,
    )
    assert audit.selected_reviewers == ("Gemini", "Antigravity")
    assert audit.phase == "secondary-audit"
    assert audit.calls_avoided == 0
    # An explicit caller phase can never bypass the primary gate, and without
    # qualified panel evidence a panel-phase checkpoint does not open the panel.
    gated = select_reviewers(_snapshot(phase="secondary-audit"), NARROW, phase="secondary-audit")
    assert gated.selected_reviewers == ("Codex",)
    assert gated.phase == "primary"
    # With panel evidence the same state is post-panel remediation.
    post_panel = select_reviewers(
        _snapshot(phase="secondary-audit", panel_evidence=True), NARROW, phase="secondary-audit"
    )
    assert post_panel.selected_reviewers == ("Codex",)
    assert post_panel.phase == "remediation"


def test_scoped_remediation_selects_all_owners_and_primary_then_sweeps_missing():
    remediation = select_reviewers(
        _snapshot(
            obligations=(
                _obligation("item-1", owners=("Gemini",)),
                _obligation("item-2", owners=("Antigravity",), scope=("src/api.py",)),
            ),
            phase="secondary-audit",
            panel_evidence=True,
        ),
        NARROW,
    )
    assert remediation.selected_reviewers == ("Codex", "Gemini", "Antigravity")
    assert remediation.phase == "remediation"
    assert remediation.active_owners == ("Antigravity", "Gemini")
    assert remediation.reason.startswith(POST_PANEL_PREFIX)

    single_owner = select_reviewers(
        _snapshot(
            obligations=(_obligation(owners=("Gemini",)),),
            phase="secondary-audit",
            panel_evidence=True,
        ),
        NARROW,
    )
    assert single_owner.selected_reviewers == ("Codex", "Gemini")
    assert single_owner.calls_avoided == 1
    assert "exact-head secondary sweep follows" in dict(single_owner.paused_reviewers)["Antigravity"]

    # Owners and primary cleared: every secondary lacking exact-head approval sweeps.
    sweep = select_reviewers(
        _snapshot(previous="b" * 40, phase="remediation", panel_evidence=True),
        TransitionClassification("narrow", "same exact candidate head"),
        qualifying_approvals=("Codex", "Gemini"),
        final_sweep=True,
    )
    assert sweep.selected_reviewers == ("Antigravity",)
    assert sweep.phase == "final-secondary-sweep"
    assert sweep.calls_avoided == 0

    # A primary that failed during remediation stays outstanding before the sweep.
    primary_outstanding = select_reviewers(
        _snapshot(previous="b" * 40, phase="remediation", panel_evidence=True),
        TransitionClassification("narrow", "same exact candidate head"),
        qualifying_approvals=("Gemini",),
        final_sweep=True,
    )
    assert primary_outstanding.selected_reviewers == ("Codex",)
    assert primary_outstanding.phase == "remediation"


UNSAFE_CLASSIFICATIONS = [
    TransitionClassification("broad", "diff path outside obligation scopes"),
    TransitionClassification("broad", "the active obligation ledger is not reconstructible"),
    TransitionClassification("broad", "a returning reviewer's history could not be reconstructed"),
    TransitionClassification("broad", "binary or mode change"),
]


@pytest.mark.parametrize("classification", UNSAFE_CLASSIFICATIONS)
@pytest.mark.parametrize("phase", ["primary", "secondary-audit", "remediation", "final-secondary-sweep"])
def test_unsafe_remediation_after_panel_evidence_selects_complete_board(classification, phase):
    decision = select_reviewers(
        _snapshot(obligations=(_obligation(owners=("Gemini",)),), phase=phase, panel_evidence=True),
        classification,
    )
    assert decision.selected_reviewers == ("Codex", "Gemini", "Antigravity")
    assert decision.phase == "full-board"
    assert decision.paused_reviewers == ()
    assert decision.calls_avoided == 0
    assert decision.reason == f"{POST_PANEL_PREFIX}full board required: {classification.reason}"


@pytest.mark.parametrize("classification", UNSAFE_CLASSIFICATIONS)
@pytest.mark.parametrize("phase", [None, "primary", "full-board", "secondary-audit", "remediation"])
def test_broad_or_ambiguous_change_before_primary_approval_reinvokes_only_primary(classification, phase):
    # Row prepanel-broad-change: primary-owned findings, broad transition, no
    # qualified panel evidence (a pre-#840 panel-phase checkpoint is not one).
    decision = select_reviewers(
        _snapshot(obligations=(_obligation(owners=("Codex",)),), phase=phase), classification
    )
    assert decision.selected_reviewers == ("Codex",)
    assert decision.phase == "primary"
    assert decision.reason == (
        f"{STRICT_PRE_PANEL_PREFIX}{classification.reason}; primary re-invoked with full context"
    )
    assert decision.calls_avoided == 2
    assert all(reason.startswith("primary phase") for _name, reason in decision.paused_reviewers)


def test_missing_fix_scope_before_primary_approval_reinvokes_only_primary():
    # Row prepanel-ambiguous-scope: the primary's finding carries no scope.
    scopeless = _obligation(owners=("Codex",), scope=None)
    ambiguous = classify_transition(
        "a" * 40,
        "b" * 40,
        [GitChange("src/worker.py")],
        scopes=None,
        broad_rules=(".github/**",),
        obligations=(scopeless,),
    )
    assert ambiguous.broad
    decision = select_reviewers(_snapshot(obligations=(scopeless,)), ambiguous)
    assert decision.selected_reviewers == ("Codex",)
    assert decision.phase == "primary"
    assert decision.reason.startswith(STRICT_PRE_PANEL_PREFIX)
    assert ambiguous.reason in decision.reason
    # Even when the observed change is narrow, a scope-less obligation is a
    # strict pre-panel fallback rather than an ordinary primary recheck.
    narrow_scopeless = select_reviewers(_snapshot(obligations=(scopeless,)), NARROW)
    assert narrow_scopeless.selected_reviewers == ("Codex",)
    assert "no valid exact fix scope" in narrow_scopeless.reason


def test_automatic_fallback_before_panel_is_primary_only_and_after_panel_is_full_board():
    reasons = ("scheduler metadata recovery: current-head scheduler metadata is invalid",)
    before = select_reviewers(
        _snapshot(force_full=True, fallback_reasons=reasons, phase="primary"),
        TransitionClassification("broad", "scheduler metadata is missing or invalid"),
    )
    assert before.selected_reviewers == ("Codex",)
    assert before.phase == "primary"
    assert before.reason.startswith(STRICT_PRE_PANEL_PREFIX + reasons[0])
    assert before.reason.endswith("primary re-invoked with full context")
    # A panel-phase checkpoint without qualified evidence changes nothing.
    checkpoint_only = select_reviewers(
        _snapshot(force_full=True, phase="full-board"), NARROW
    )
    assert checkpoint_only.selected_reviewers == ("Codex",)
    assert checkpoint_only.reason.startswith(STRICT_PRE_PANEL_PREFIX)

    after = select_reviewers(
        _snapshot(force_full=True, fallback_reasons=reasons, panel_evidence=True), NARROW
    )
    assert after.selected_reviewers == ("Codex", "Gemini", "Antigravity")
    assert after.phase == "full-board"
    assert after.reason.startswith(POST_PANEL_PREFIX + "force-full latch")


def test_broad_head_change_after_panel_evidence_selects_complete_board():
    decision = select_reviewers(
        _snapshot(phase="final-secondary-sweep", panel_evidence=True), BROAD
    )
    assert decision.selected_reviewers == ("Codex", "Gemini", "Antigravity")
    assert decision.phase == "full-board"
    assert decision.reason.startswith(POST_PANEL_PREFIX + "full board required after panel evidence")
    with pytest.raises(AgentLoopError, match="phase checkpoint"):
        select_reviewers(_snapshot(phase="bogus"), NARROW)


def test_secondary_owned_finding_before_panel_stops_with_diagnostic_unless_operator():
    # Row prepanel-unsafe-stop.
    snapshot = _snapshot(
        obligations=(
            _obligation("item-3", owners=("Gemini",)),
            _obligation("item-1", owners=("Codex",)),
        ),
        phase="full-board",
    )
    for classification in (NARROW, BROAD):
        with pytest.raises(PrePanelSafetyError) as raised:
            select_reviewers(snapshot, classification, qualifying_approvals=("Codex",))
        message = str(raised.value)
        assert message.startswith("pre-panel safety cannot be established")
        assert "item-3" in message and "item-1" not in message
        assert "Gemini" in message
        assert "--pr-review-force-full" in message
        assert isinstance(raised.value, AgentLoopError)
    forced = select_reviewers(dataclasses.replace(snapshot, operator_force_full=True), NARROW)
    assert forced.selected_reviewers == ("Codex", "Gemini", "Antigravity")
    assert forced.phase == "full-board"
    assert forced.reason == OPERATOR_FORCE_FULL_REASON
    # After a qualified panel opening the same ownership is ordinary remediation.
    post_panel = select_reviewers(dataclasses.replace(snapshot, panel_evidence=True), NARROW)
    assert post_panel.phase == "remediation"


@pytest.mark.parametrize("owner", ["Orchestrator", "machine-ci", "unknown-reviewer"])
def test_non_reviewer_obligation_before_panel_is_primary_only_without_diagnostic(owner):
    # Row prepanel-machine-obligation: CI/machine/Orchestrator owners are never
    # secondary owners and never trigger the diagnostic stop.
    machine = ReviewObligation(
        item_id="item-9",
        status="blocking",
        scope=None,
        resolution_owners=(owner,),
        pending_owners=(owner,),
    )
    decision = select_reviewers(_snapshot(obligations=(machine,)), NARROW)
    assert decision.selected_reviewers == ("Codex",)
    assert decision.phase == "primary"
    assert decision.reason.startswith(STRICT_PRE_PANEL_PREFIX)
    assert "no valid exact fix scope" in decision.reason

    # Row prepanel-primary-approved-with-machine-obligation: the primary's
    # exact-head approval opens the panel while the obligation remains.
    audit = select_reviewers(
        _snapshot(obligations=(machine,)), BROAD, qualifying_approvals=("Codex",)
    )
    assert audit.selected_reviewers == ("Gemini", "Antigravity")
    assert audit.phase == "secondary-audit"
    assert "non-reviewer obligations remain: item-9" in audit.reason
    assert "post-panel rules" in audit.reason


def test_prepanel_empty_selection_invariant():
    # Only an unavailable primary empties a primary-phase selection.
    missing_primary = select_reviewers(
        _snapshot(previous=None, phase="primary"),
        BROAD,
        unavailable_reviewers=("Codex",),
    )
    assert missing_primary.selected_reviewers == ()
    assert missing_primary.phase == "primary"
    assert "unavailable" in dict(missing_primary.paused_reviewers)["Codex"]
    # Branch (iii) is empty only when every secondary is unavailable; it never
    # shrinks merely because the caller dropped premature approvals.
    all_unavailable = select_reviewers(
        _snapshot(), NARROW,
        qualifying_approvals=("Codex",),
        unavailable_reviewers=("Gemini", "Antigravity"),
    )
    assert all_unavailable.selected_reviewers == ()
    assert all_unavailable.phase == "secondary-audit"
    first_audit = select_reviewers(_snapshot(), NARROW, qualifying_approvals=("Codex",))
    assert first_audit.selected_reviewers == ("Gemini", "Antigravity")


def test_operator_force_full_overrides_primary_phase_and_unavailable_is_not_approval():
    forced = select_reviewers(_snapshot(previous=None, operator_force_full=True), BROAD)
    assert forced.selected_reviewers == ("Codex", "Gemini", "Antigravity")
    assert forced.phase == "full-board"
    assert forced.reason == OPERATOR_FORCE_FULL_REASON

    partial_sweep = select_reviewers(
        _snapshot(previous="b" * 40, phase="secondary-audit", panel_evidence=True),
        TransitionClassification("narrow", "same exact candidate head"),
        qualifying_approvals=("Codex", "Gemini"),
        unavailable_reviewers=("Antigravity",),
        final_sweep=True,
    )
    assert partial_sweep.selected_reviewers == ()
    assert "unavailable" in dict(partial_sweep.paused_reviewers)["Antigravity"]


def test_legacy_scheduler_payload_without_phase_is_valid_and_grants_no_phase_authority():
    legacy_contract = _contract()
    metadata = PostedRoundMetadata(
        flow="pr",
        role="summary",
        agent="Orchestrator",
        round_number=2,
        subject="b" * 40,
        scheduler_contract=legacy_contract.as_dict(),
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
    payload = decode_mapping(_encode_round_metadata(metadata))
    assert "scheduler_phase" not in payload
    decoded = _decode_round_metadata_mapping(payload)
    assert decoded.scheduler_metadata_status == "valid"
    assert decoded.scheduler_phase is None
    assert decoded.scheduler_primary_reviewer is None
    assert decoded.scheduler_approved_reviewers == ()
    # The scheduler itself refuses to treat a missing checkpoint as panel evidence.
    decision = select_reviewers(
        _snapshot(previous="b" * 40, phase=None),
        TransitionClassification("narrow", "same exact candidate head"),
        qualifying_approvals=("Codex",),
    )
    assert decision.phase == "secondary-audit"


def test_selective_intermediate_semantics_are_unchanged_by_the_staged_policy():
    snapshot = SchedulerSnapshot(
        previous_sha="a" * 40,
        current_sha="b" * 40,
        contract=_contract(),
        obligations=(_obligation(owners=("Codex",)),),
        phase="secondary-audit",  # ignored by the selective policy
    )
    decision = select_reviewers(snapshot, NARROW)
    assert decision.selected_reviewers == ("Codex",)
    assert decision.primary_reviewer is None
    assert decision.calls_avoided == 2
    broad = select_reviewers(snapshot, BROAD)
    assert broad.selected_reviewers == ("Claude", "Codex", "Antigravity")
    compat = select_reviewers(
        SchedulerSnapshot(
            previous_sha="a" * 40,
            current_sha="b" * 40,
            contract=ReviewSchedulingContract(required_reviewers=("Claude", "Codex")),
        ),
        NARROW,
    )
    assert compat.selected_reviewers == ("Claude", "Codex")
    assert compat.calls_avoided == 0


def test_staged_pr_scheduler_options_validate_together(tmp_path):
    args = build_parser().parse_args(
        [
            "pr", "77", "--pr-review-policy", "primary-then-panel",
            "--primary-reviewer", "codex", "--reviewer", "codex", "--reviewer", "gemini",
        ]
    )
    assert args.pr_review_policy == "primary-then-panel"
    assert args.primary_reviewer == "codex"
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        pr_review_policy="primary-then-panel",
        primary_reviewer="codex",
    )
    assert config.primary_reviewer == "codex"
    with pytest.raises(AgentLoopError, match="--primary-reviewer is required"):
        make_config(tmp_path, reviewer=("codex", "gemini"), pr_review_policy="primary-then-panel")
    with pytest.raises(AgentLoopError, match="one of the configured"):
        make_config(
            tmp_path, reviewer=("codex", "gemini"),
            pr_review_policy="primary-then-panel", primary_reviewer="claude",
        )
    with pytest.raises(AgentLoopError, match="at least one secondary"):
        make_config(
            tmp_path, reviewer=("codex",),
            pr_review_policy="primary-then-panel", primary_reviewer="codex",
        )
    with pytest.raises(AgentLoopError, match="requires --pr-review-policy primary-then-panel"):
        make_config(tmp_path, reviewer=("codex", "gemini"), primary_reviewer="codex")
    default = make_config(tmp_path, reviewer=("codex", "gemini"))
    assert default.pr_review_policy == "all-reviewers"
    assert default.primary_reviewer is None


# ---------------------------------------------------------------------------
# #840: qualified panel evidence and causal approval eligibility
# ---------------------------------------------------------------------------


def _staged_record(index, *, role="summary", agent="Orchestrator", subject="h1", state=None,
                   phase=None, approved=(), force_full=False, source=None, new_items=()):
    contract = _primary_contract()
    metadata = PostedRoundMetadata(
        flow="pr",
        role=role,
        agent=agent,
        round_number=1,
        subject=subject,
        state=state,
        new_items=tuple(new_items),
        scheduler_contract=contract.as_dict(),
        scheduler_previous_sha=None,
        scheduler_current_sha=subject,
        scheduler_obligation_digest="0" * 16,
        scheduler_selected_reviewers=contract.required_reviewers,
        scheduler_reasons=("test",),
        scheduler_final_sweep=False,
        scheduler_force_full=force_full,
        scheduler_force_full_source=source,
        scheduler_calls_avoided=0,
        scheduler_phase=phase,
        scheduler_primary_reviewer="Codex",
        scheduler_approved_reviewers=tuple(approved),
    )
    return orchestrator.PostedRoundRecord(index=index, metadata=metadata, body="")


def _evidence(records):
    return orchestrator._derive_pr_panel_evidence(
        records, primary_reviewer="Codex", required_reviewers=("Codex", "Gemini", "Antigravity")
    )


def test_panel_evidence_requires_operator_source_or_prior_same_subject_primary_approval():
    primary_approval = _staged_record(1, role="reviewer", agent="Codex", state="approved", phase="primary")
    audit = _staged_record(2, phase="secondary-audit", approved=("Codex",))
    qualified = _evidence([primary_approval, audit])
    assert qualified.opened and qualified.opening_index == 2
    assert qualified.opening_source == "primary-approval"

    operator = _evidence([_staged_record(0, phase="full-board", force_full=True, source="operator")])
    assert operator.opened and operator.opening_source == "operator"

    # A secondary-audit record whose primary approval is for another subject,
    # or appears later in comment order, is not an opening.
    other_subject = _staged_record(1, role="reviewer", agent="Codex", subject="h0", state="approved")
    assert not _evidence([other_subject, audit]).opened
    later_approval = _staged_record(3, role="reviewer", agent="Codex", state="approved")
    assert not _evidence([audit, later_approval]).opened
    # The record must also list the primary as approved.
    assert not _evidence([primary_approval, _staged_record(2, phase="secondary-audit")]).opened


def test_premature_panel_artifacts_and_legacy_latches_are_not_panel_evidence():
    premature = [
        _staged_record(0, phase="full-board", force_full=True),  # legacy unattributed latch
        _staged_record(1, role="reviewer", agent="Gemini", state="approved", phase="full-board"),
        _staged_record(2, phase="remediation", force_full=True, source="automatic"),
    ]
    evidence = _evidence(premature)
    assert not evidence.opened
    assert evidence.prepanel_latch
    assert not evidence.post_opening_automatic_latch
    assert any("Gemini review" in artifact for artifact in evidence.unqualified_artifacts)
    assert any("full-board scheduler record" in artifact for artifact in evidence.unqualified_artifacts)

    # A legacy latch after a qualified opening is restored as automatic.
    opened = _evidence(
        [
            _staged_record(1, role="reviewer", agent="Codex", state="approved", phase="primary"),
            _staged_record(2, phase="secondary-audit", approved=("Codex",)),
            _staged_record(3, phase="full-board", force_full=True),
        ]
    )
    assert opened.opened and opened.post_opening_automatic_latch and not opened.prepanel_latch


def test_secondary_approval_counts_only_after_qualified_opening():
    primary_approval = _staged_record(1, role="reviewer", agent="Codex", state="approved")
    premature_secondary = _staged_record(0, role="reviewer", agent="Gemini", state="approved")
    audit = _staged_record(2, phase="secondary-audit", approved=("Codex",))
    post_secondary = _staged_record(3, role="reviewer", agent="Gemini", state="approved")
    evidence = _evidence([premature_secondary, primary_approval, audit, post_secondary])

    def qualified(record, *, operator=False, current=evidence):
        return orchestrator._pr_record_is_panel_qualified(
            record, current, primary_reviewer="Codex", operator_force_full=operator
        )

    assert not qualified(premature_secondary)
    assert qualified(post_secondary)
    # The primary is never premature under a primary-approval opening.
    assert qualified(primary_approval)
    # Without any opening, secondaries never qualify and the primary does.
    none = _evidence([premature_secondary, primary_approval])
    assert not qualified(premature_secondary, current=none)
    assert qualified(primary_approval, current=none)
    # An operator opening (being established, or recorded) qualifies only
    # records written after it, for every reviewer.
    assert not qualified(primary_approval, operator=True, current=none)
    operator_opening = _evidence(
        [primary_approval, _staged_record(2, phase="full-board", force_full=True, source="operator")]
    )
    assert not qualified(primary_approval, current=operator_opening)
    assert qualified(post_secondary, current=operator_opening)


def test_superseded_prepanel_review_context_is_built_from_the_record():
    item = _next_unresolved_item(
        item_number=4, reviewer="Gemini", source_round=1, text="premature cache race",
        status="blocking", fix_scope=("src/worker.py",),
    )
    record = _staged_record(5, role="reviewer", agent="Gemini", state="blocking", new_items=(item,))
    superseded = orchestrator._superseded_prepanel_review(record)
    assert superseded.reviewer == "Gemini"
    assert superseded.claims == ("premature cache race",)
    assert superseded.item_ids == ("item-4",)
    assert orchestrator._superseded_prepanel_review(None) is None
    assert "Gemini (round 1, head h1, state blocking; items: item-4)" == (
        orchestrator._describe_superseded_prepanel_review(record)
    )


def test_non_staged_policies_ignore_pre_panel_rules():
    # Row other-policies-unchanged: no diagnostic, same decisions and reasons.
    secondary_owned = (_obligation(owners=("Claude",)),)
    snapshot = SchedulerSnapshot(
        previous_sha="a" * 40,
        current_sha="b" * 40,
        contract=_contract(),
        obligations=secondary_owned,
    )
    narrow = select_reviewers(snapshot, NARROW)
    assert narrow.selected_reviewers == ("Claude",)
    assert narrow.reason == "narrow transition: pending resolution owners and co-owners"
    broad = select_reviewers(snapshot, BROAD)
    assert broad.reason == f"full board required: {BROAD.reason}"
    for forced in (
        dataclasses.replace(snapshot, force_full=True),
        dataclasses.replace(snapshot, operator_force_full=True),
    ):
        decision = select_reviewers(forced, NARROW)
        assert decision.selected_reviewers == ("Claude", "Codex", "Antigravity")
        assert decision.reason == "force-full latch"
    compat = SchedulerSnapshot(
        previous_sha="a" * 40,
        current_sha="b" * 40,
        contract=ReviewSchedulingContract(required_reviewers=("Claude", "Codex")),
        obligations=secondary_owned,
        operator_force_full=True,
    )
    assert select_reviewers(compat, NARROW).reason == "compatibility policy"
