"""Round metadata compatibility for the reviewer-board amendment digest (#943)."""

import dataclasses
import json
from types import SimpleNamespace

import pytest

import coding_review_agent_loop.github as github_module
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import (
    IssueComment,
    post_verified_trusted_issue_protocol_comment,
    reset_host_footer_log_latch,
)
from coding_review_agent_loop.protocol_markers import KNOWN_HOST_COMMENT_FOOTER
from coding_review_agent_loop.review_scheduling import make_contract
from coding_review_agent_loop.round_state import (
    PlanValidationDiagnosticPayload,
    PostedRoundMetadata,
    _decode_round_metadata,
    _encode_round_metadata,
    _attach_round_metadata,
    _latest_pr_approved_reviews_for_head,
    PostedRoundRecord,
    encode_plan_validation_diagnostic_body,
    recover_plan_validation_diagnostic,
)
from coding_review_agent_loop.round_transport import decode_mapping, encode_mapping

from agent_loop_helpers import make_config
from coding_review_agent_loop.reviewer_seats import (
    ReviewerSeat, SeatAgent, reviewer_seat_binding, validate_pr_seat_bindings,
)
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
from agent_loop_helpers import patch_attr_in
import re
from dataclasses import replace
from issue_loop_test_helpers import _m1103_stalled_runner, _plan_round_records


def test_named_pr_seat_binding_and_exact_head_approval(tmp_path):
    seats = (
        SeatAgent(ReviewerSeat("flash", "antigravity", ("Model A",)), tmp_path / "flash"),
        SeatAgent(ReviewerSeat("opus", "antigravity", ("Model B",)), tmp_path / "opus"),
    )
    config = make_config(tmp_path, reviewer=seats, reviewer_seats=seats)
    binding = reviewer_seat_binding(config)

    def record(index, seat, head):
        metadata = PostedRoundMetadata(
            flow="pr", role="reviewer", agent=str(seat), round_number=index,
            subject=head, state="approved", provider="antigravity",
            configured_model=seat.model_chain[0], seat_binding=binding,
        )
        decoded = _decode_round_metadata(_encode_round_metadata(metadata))
        assert decoded.seat_binding == binding
        return PostedRoundRecord(index, decoded, _attach_round_metadata("Approved.", metadata))

    first = record(0, seats[0], "H")
    second = record(1, seats[1], "H")
    moved = record(2, seats[0], "H2")
    contract = {str(seat): (seat.model_chain[0], None, "antigravity") for seat in seats}
    lookup = lambda records, head: _latest_pr_approved_reviews_for_head(
        [SimpleNamespace(body=entry.body) for entry in records], head_sha=head,
        configured_reviewers=seats, reviewer_acquisition_contract=contract,
    )
    assert set(lookup((first, second), "H")) == {"flash", "opus"}
    assert set(lookup((first, second, moved), "H2")) == {"flash"}
    assert validate_pr_seat_bindings((first, second), config) == set()
    changed = SeatAgent(ReviewerSeat("flash", "antigravity", ("Model C",)), seats[0].workdir)
    changed_config = dataclasses.replace(config, reviewer=(changed, seats[1]), reviewer_seats=(changed, seats[1]))
    assert validate_pr_seat_bindings((first, second), changed_config) == {"flash"}
    rebound = SeatAgent(ReviewerSeat("flash", "codex", ("Model C",)), seats[0].workdir)
    rebound_config = dataclasses.replace(config, reviewer=(rebound, seats[1]), reviewer_seats=(rebound, seats[1]))
    with pytest.raises(AgentLoopError, match="changed backend"):
        validate_pr_seat_bindings((first, second), rebound_config)


def test_named_pr_binding_allows_legacy_implicit_model_on_other_backend(tmp_path):
    seat = SeatAgent(ReviewerSeat("flash", "antigravity", ("Model A",)), tmp_path / "flash")
    config = make_config(
        tmp_path, reviewer=("codex", seat), reviewer_seats=(seat,),
        codex_model=None, reviewer_codex_model=None,
    )
    binding = reviewer_seat_binding(config)
    assert binding["seats"][0]["model_chain"] == [None]
    record = PostedRoundRecord(
        0,
        PostedRoundMetadata(
            flow="pr", role="reviewer", agent="flash", round_number=1,
            subject="head", seat_binding=binding,
        ),
        "",
    )
    assert validate_pr_seat_bindings((record,), config) == set()


def test_named_pr_prelaunch_binding_is_checked_before_first_reviewer(tmp_path):
    seat = SeatAgent(ReviewerSeat("flash", "antigravity", ("Model A",)), tmp_path / "flash")
    config = make_config(tmp_path, reviewer=(seat,), reviewer_seats=(seat,))
    binding = reviewer_seat_binding(config)
    checkpoint = PostedRoundRecord(
        0, _checkpoint(seat_binding=binding), "",
    )
    assert validate_pr_seat_bindings((checkpoint,), config) == set()
    changed = SeatAgent(ReviewerSeat("flash", "antigravity", ("Model B",)), seat.workdir)
    changed_config = dataclasses.replace(config, reviewer=(changed,), reviewer_seats=(changed,))
    assert validate_pr_seat_bindings((checkpoint,), changed_config) == {"flash"}
    rebound = SeatAgent(ReviewerSeat("flash", "codex", ("Model B",)), seat.workdir)
    rebound_config = dataclasses.replace(config, reviewer=(rebound,), reviewer_seats=(rebound,))
    with pytest.raises(AgentLoopError, match="changed backend"):
        validate_pr_seat_bindings((checkpoint,), rebound_config)
    with pytest.raises(AgentLoopError, match="unbound historical recovery record"):
        validate_pr_seat_bindings((PostedRoundRecord(0, _checkpoint(), ""),), config)


def test_named_pr_evidence_and_qualification_summaries_resume_with_binding(tmp_path, monkeypatch):
    import coding_review_agent_loop.pr_loop_support as support

    seat = SeatAgent(ReviewerSeat("flash", "antigravity", ("Model A",)), tmp_path / "flash")
    config = make_config(tmp_path, reviewer=(seat,), reviewer_seats=(seat,))
    bodies = []
    monkeypatch.setattr(support, "post_pr_comment", lambda *args, **kwargs: bodies.append(kwargs["body"]))
    common = dict(runner=None, config=config, pr_number=77, round_number=1, head_sha="head")
    support._publish_evidence_freeze(
        **common, items=(), signed_requirement_ids=(), allowed_rounds=3,
        watch_failure_extension_used=False, watch_head_extension_used=False,
    )
    support._publish_evidence_release(
        **common, items=(), reason="evidence-cleared", surfaced_requirement_ids=(),
        allowed_rounds=3, watch_failure_extension_used=False,
        watch_head_extension_used=False,
    )
    support._persist_qualification_checkpoint(
        **common, unresolved_items=(), checkpoint=_legacy_checkpoint(), message="Checkpoint.",
    )
    from coding_review_agent_loop.round_state import _extract_round_metadata_records
    records = _extract_round_metadata_records(
        [SimpleNamespace(body=body) for body in bodies], flow="pr",
    )
    assert [record.metadata.phase for record in records] == [
        "evidence-freeze", "evidence-release", "qualification-checkpoint",
    ]
    assert all(record.metadata.seat_binding == reviewer_seat_binding(config) for record in records)
    assert validate_pr_seat_bindings(records, config) == set()


def test_named_pr_resume_records_rebound_prelaunch_as_board_change(tmp_path, monkeypatch):
    """A backend change on a named seat is a recorded board change (#1376).

    It was refused as drift before; now one orchestrator link is posted
    before any reviewer turn, and the rebound seat reviews on its new backend.
    """
    from agent_loop_helpers import FakeRunner, structured_pr_review

    seat = SeatAgent(ReviewerSeat("flash", "antigravity", ("Model A",)), tmp_path / "flash")
    seat.workdir.mkdir()
    config = make_config(
        tmp_path, reviewer=(seat,), reviewer_seats=(seat,),
        pr_review_policy="selective-intermediate", pre_review_tests=False,
    )
    runner = FakeRunner()
    original_post = pr_loop_support.post_pr_comment

    def interrupt_after_prelaunch(*args, **kwargs):
        result = original_post(*args, **kwargs)
        if "PR review scheduling audit:" in kwargs["body"]:
            raise KeyboardInterrupt
        return result

    with monkeypatch.context() as patch:
        patch_attr_in(patch, (pr_loop_support, pr_loop, issue_loop), "post_pr_comment", interrupt_after_prelaunch)
        with pytest.raises(KeyboardInterrupt):
            pr_loop.run_pr_loop(runner, pr_number=77, config=config)
    assert not any(command[0] == "agy" for command, _ in runner.commands)
    rebound = SeatAgent(ReviewerSeat("flash", "codex", ("Model B",)), seat.workdir)
    rebound_config = dataclasses.replace(config, reviewer=(rebound,), reviewer_seats=(rebound,))
    runner.codex_outputs.append(structured_pr_review(reviewer="flash (OpenAI Codex: Model B)"))
    assert pr_loop.run_pr_loop(runner, pr_number=77, config=rebound_config) == 0
    links = [c for c in runner.comments if c.startswith("Reviewer board change recorded.")]
    assert len(links) == 1
    assert "flash (antigravity: Model A) -> flash (codex: Model B)" in links[0]
    assert not any(command[0] == "agy" for command, _ in runner.commands)


def _checkpoint(**overrides):
    values = dict(
        flow="pr",
        role="summary",
        agent="Orchestrator",
        round_number=2,
        subject="abc123",
        phase="scheduler-prelaunch",
        scheduler_contract=make_contract(
            ("Codex", "Claude"), "selective-intermediate", None
        ).as_dict(),
        scheduler_previous_sha=None,
        scheduler_current_sha="abc123",
        scheduler_obligation_digest="0" * 16,
        scheduler_selected_reviewers=("Codex", "Claude"),
        scheduler_reasons=("full board",),
        scheduler_final_sweep=False,
        scheduler_force_full=False,
        scheduler_calls_avoided=0,
    )
    values.update(overrides)
    return PostedRoundMetadata(**values)


def test_legacy_record_encoding_is_unchanged_and_decodes_without_the_field():
    legacy = _checkpoint()
    encoded = _encode_round_metadata(legacy)
    assert "reviewer_board_amendment_digest" not in decode_mapping(encoded)
    decoded = _decode_round_metadata(encoded)
    assert decoded.scheduler_metadata_status == "valid"
    assert decoded.reviewer_board_amendment_digest is None
    assert _decode_round_metadata(_encode_round_metadata(decoded)) == decoded


def test_amendment_digest_round_trips():
    digest = "a" * 64
    encoded = _encode_round_metadata(_checkpoint(reviewer_board_amendment_digest=digest))
    assert decode_mapping(encoded)["reviewer_board_amendment_digest"] == digest
    decoded = _decode_round_metadata(encoded)
    assert decoded.reviewer_board_amendment_digest == digest
    assert _decode_round_metadata(_encode_round_metadata(decoded)) == decoded


def test_invalid_amendment_digest_is_rejected():
    with pytest.raises(ValueError, match="reviewer board amendment digest"):
        _checkpoint(reviewer_board_amendment_digest="not-a-digest")
    tampered = _encode_round_metadata(_checkpoint(reviewer_board_amendment_digest="b" * 64))
    # Re-encode a payload whose digest is present but malformed.
    payload = decode_mapping(tampered)
    payload["reviewer_board_amendment_digest"] = "XYZ"
    with pytest.raises(AgentLoopError):
        _decode_round_metadata(encode_mapping(payload))
    payload["reviewer_board_amendment_digest"] = ""
    with pytest.raises(AgentLoopError):
        _decode_round_metadata(encode_mapping(payload))


# --- plan-validation diagnostic recovery with the host footer (#1043) -------

FOOTER = KNOWN_HOST_COMMENT_FOOTER


def _diagnostic_body(attempt=1):
    return encode_plan_validation_diagnostic_body(
        PlanValidationDiagnosticPayload(
            repository="OWNER/REPO",
            issue_number=1043,
            planning_generation=1,
            target_coder_round=1,
            prior_plan_subject=None,
            candidate_kind="plan_state",
            architecture_contract_version=1,
            execution_strategy_contract_version=1,
            risk_test_matrix_contract_version=1,
            expected_producer_login="agent",
            expected_producer_id=7,
            failure_attempt=attempt,
            candidate_digest=str(attempt) * 64,
            category="deterministic",
            diagnostic="missing audit",
        )
    )


def _recover(comments, observed=None):
    return recover_plan_validation_diagnostic(
        comments,
        repository="OWNER/REPO", issue_number=1043,
        expected_author_login="agent", expected_author_id=7,
        planning_generation=1, target_coder_round=1,
        prior_plan_subject=None, candidate_kind="plan_state",
        architecture_contract_version=1,
        execution_strategy_contract_version=1,
        risk_test_matrix_contract_version=1,
        on_host_footer=None if observed is None else observed.append,
    )


def _issue_comment(body, *, comment_id, created_at="2026-09-25T00:00:00Z"):
    return IssueComment(
        author="agent", created_at=created_at, body=body, comment_id=comment_id, author_id=7,
    )


class _FooteringHost:
    """Store every posted issue comment with the known host footer."""

    def run(self, args, *, cwd, check=True, input_text=None, **_kwargs):
        if input_text is None:
            # Reconciliation baseline reads (#510) are unsupported here.
            return SimpleNamespace(returncode=1, stdout="", stderr="unsupported read")
        posted = json.loads(input_text)["body"]
        envelope = {
            "id": 900,
            "body": posted + FOOTER,
            "user": {"login": "agent", "id": 7},
            "created_at": "2026-09-25T00:00:00Z",
        }
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(envelope))


def test_footered_diagnostic_publishes_and_recovers_like_a_clean_one(tmp_path, monkeypatch):
    monkeypatch.setattr(github_module, "active_workdir", lambda config: None)
    monkeypatch.setattr(github_module, "log", lambda *_args: None)
    reset_host_footer_log_latch()
    body = _diagnostic_body(attempt=2)

    published = post_verified_trusted_issue_protocol_comment(
        _FooteringHost(), config=make_config(tmp_path), issue_number=1043, body=body,
        expected_author_login="agent", expected_author_id=7,
    )
    assert published.body == str(body)

    # A later run fetches the stored, footered comment from the issue context.
    observed: list[str] = []
    footered = _recover(
        (_issue_comment(str(_diagnostic_body(1)) + FOOTER, comment_id=899),
         _issue_comment(str(body) + FOOTER, comment_id=900)),
        observed,
    )
    clean = _recover(
        (_issue_comment(str(_diagnostic_body(1)), comment_id=899),
         _issue_comment(str(body), comment_id=900)),
    )

    assert footered is not None and clean is not None
    assert footered.payload == clean.payload
    assert footered.failure_attempt == clean.failure_attempt == 2
    assert footered.server_comment_id == clean.server_comment_id == 900
    assert footered.exact_live_body == str(body)
    assert observed and set(observed) == {"plan-validation diagnostic recovery"}


@pytest.mark.parametrize(
    ("suffix", "footer_observed"),
    [
        (FOOTER + FOOTER, True),
        (FOOTER + "\n", False),
        ("\n\n---\n_Generated by [Claude Code](http://claude.ai/code)_", False),
        ("\n\nextra prose", False),
    ],
)
def test_doubled_or_variant_footer_diagnostic_stays_ineligible(suffix, footer_observed):
    observed: list[str] = []

    assert _recover((_issue_comment(str(_diagnostic_body()) + suffix, comment_id=900),), observed) is None
    # A doubled footer is stripped once at ingestion (and that one exact footer
    # is reported) and then rejected by the exact-canonical decoder; the
    # decoder itself never strips.  Variant suffixes are not the host footer.
    assert observed == (["plan-validation diagnostic recovery"] if footer_observed else [])


def test_footer_is_observed_on_malformed_diagnostic_re_read():
    observed: list[str] = []
    marker_only = str(_diagnostic_body())
    # Corrupt the encoded payload but keep the diagnostic marker shape.
    malformed = marker_only.replace(marker_only[-20:-10], "!!!!!!!!!!")
    assert malformed != marker_only

    assert _recover((_issue_comment(malformed + FOOTER, comment_id=900),), observed) is None
    assert observed == ["plan-validation diagnostic recovery"]


def test_footer_is_observed_before_identity_checks():
    observed: list[str] = []
    stranger = IssueComment(
        author="someone", created_at="2026-09-25T00:00:00Z",
        body=str(_diagnostic_body()) + FOOTER, comment_id=900, author_id=8,
    )

    assert _recover((stranger,), observed) is None
    assert observed == ["plan-validation diagnostic recovery"]


def test_clean_diagnostic_reports_no_footer():
    observed: list[str] = []

    assert _recover((_issue_comment(str(_diagnostic_body()), comment_id=900),), observed) is not None
    assert observed == []


def test_bodies_differing_only_by_the_exact_footer_are_not_a_conflict():
    body = str(_diagnostic_body())

    selected = _recover(
        (_issue_comment(body, comment_id=900), _issue_comment(body + FOOTER, comment_id=900)),
    )

    assert selected is not None and selected.server_comment_id == 900


# --- Human-only exact-head evidence records (#1068) --------------------------

from coding_review_agent_loop.round_state import (  # noqa: E402
    EvidenceFreezeRecord,
    EvidenceReleaseRecord,
    _attach_round_metadata,
    _deserialize_unresolved_item,
    _drop_unterminated_evidence_response_records,
    _extract_round_metadata_records,
    _resume_pr_round,
)
from coding_review_agent_loop.unresolved_items import (  # noqa: E402
    _upsert_evidence_obligation,
    freeze_evidence_obligations,
)

_FROZEN_HEAD = "abc123"


def _evidence_ledger(*, frozen=True):
    ledger, _ = _upsert_evidence_obligation(
        [], item_number=1, reviewer="Codex", text="Attach the live run.",
        source_round=1, current_head_sha=_FROZEN_HEAD,
    )
    return tuple(freeze_evidence_obligations(ledger, head_sha=_FROZEN_HEAD) if frozen else ledger)


def _freeze_metadata(ledger, **payload_overrides):
    payload = dict(
        frozen_head=_FROZEN_HEAD,
        evidence_identities=tuple(item.obligation_identity for item in ledger),
        signed_requirement_ids_at_freeze=(),
        allowed_rounds=5,
        watch_failure_extension_used=False,
        watch_head_extension_used=False,
    )
    payload.update(payload_overrides)
    return PostedRoundMetadata(
        flow="pr", role="summary", agent="Orchestrator", round_number=1,
        subject=_FROZEN_HEAD, prior_items=tuple(ledger), state="blocking",
        phase="evidence-freeze", evidence_freeze=EvidenceFreezeRecord(**payload),
    )


def _comments(*metadata):
    return [SimpleNamespace(body=_attach_round_metadata("record", item)) for item in metadata]


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"lifecycle": "evidence_unknown"}, "unknown lifecycle"),
        ({"lifecycle": "evidence_frozen", "candidate_head_sha": None}, "frozen without head"),
    ],
)
def test_malformed_persisted_evidence_record_becomes_unknown_blocker(overrides, reason):
    """Row malformed-evidence-record: never dropped, never a reviewer finding."""
    payload = {
        "item_id": "item-1", "reviewer": "Codex", "source_round": 1,
        "text": "Attach the live run.", "status": "blocking", "source_status": "blocking",
        "notes": [], "authority": "machine", "obligation_kind": "human-exact-head-evidence",
        "lifecycle": "evidence_deferred", "obligation_identity": "human-exact-head-evidence:x",
        **overrides,
    }
    payload = {key: value for key, value in payload.items() if value is not None}
    item = _deserialize_unresolved_item(payload)
    assert item.obligation_kind == "unknown", reason
    assert item.is_machine_obligation
    assert item.status == "blocking"


def test_evidence_metadata_round_trips_and_legacy_records_omit_it():
    ledger = _evidence_ledger()
    metadata = _freeze_metadata(ledger)
    metadata = PostedRoundMetadata(
        **{**metadata.__dict__, "evidence_clearances": (("id-x", _FROZEN_HEAD),)}
    )
    decoded = _decode_round_metadata(_encode_round_metadata(metadata))
    assert decoded.evidence_freeze == metadata.evidence_freeze
    assert decoded.evidence_clearances == (("id-x", _FROZEN_HEAD),)
    assert decoded.prior_items == metadata.prior_items
    reviewer = PostedRoundMetadata(
        flow="pr", role="reviewer", agent="Codex", round_number=1, subject=_FROZEN_HEAD,
        evidence_requests=("Attach the live run.",),
    )
    assert _decode_round_metadata(_encode_round_metadata(reviewer)).evidence_requests == (
        "Attach the live run.",
    )
    plain = decode_mapping(_encode_round_metadata(_checkpoint()))
    for key in ("evidence_requests", "evidence_freeze", "evidence_release", "evidence_clearances"):
        assert key not in plain
    release = EvidenceReleaseRecord.from_mapping(
        {"released_head": _FROZEN_HEAD, "reason": "findings", "allowed_rounds": 3,
         "watch_failure_extension_used": False, "watch_head_extension_used": False}
    )
    assert release.valid and release.signed_requirement_ids_surfaced is None
    assert not EvidenceReleaseRecord.from_mapping({"released_head": "x", "reason": "nope"}).valid
    budget_only = EvidenceFreezeRecord.from_mapping(
        {**_freeze_metadata(ledger).evidence_freeze.as_dict(), "allowed_rounds": True}
    )
    assert budget_only.valid and not budget_only.budget_valid


def test_resume_at_a_contradictory_freeze_record_adds_unknown_blocker():
    ledger = _evidence_ledger()
    comments = _comments(_freeze_metadata(ledger, evidence_identities=("someone-else",)))
    resumed = _resume_pr_round(comments, head_sha=_FROZEN_HEAD, configured_reviewers=("codex",))
    assert resumed is not None and resumed.evidence_boundary is not None
    unknown = [item for item in resumed.prior_items if item.obligation_kind == "unknown"]
    assert unknown and unknown[0].obligation_identity == "invalid-evidence-record"
    # The frozen ledger itself is kept, never treated as absent.
    assert any(item.lifecycle == "evidence_frozen" for item in resumed.prior_items)


def test_unterminated_evidence_response_records_are_ignored_by_recovery():
    """Row evidence-response-interrupted (recovery unit)."""
    ledger = _evidence_ledger()
    freeze = _freeze_metadata(ledger)
    response = PostedRoundMetadata(
        flow="pr", role="reviewer", agent="Codex", round_number=1, subject=_FROZEN_HEAD,
        prior_items=tuple(ledger), state="approved", phase="evidence-response",
    )
    records = _extract_round_metadata_records(_comments(freeze, response), flow="pr")
    assert [record.metadata.phase for record in _drop_unterminated_evidence_response_records(records)] == [
        "evidence-freeze"
    ]
    resumed = _resume_pr_round(
        _comments(freeze, response), head_sha=_FROZEN_HEAD, configured_reviewers=("codex",)
    )
    assert resumed.evidence_boundary.phase == "evidence-freeze"
    assert resumed.completed_reviews == ()
    assert resumed.prior_items == tuple(ledger)
    # Once a terminal record follows, the pass's records are kept.
    terminated = _extract_round_metadata_records(
        _comments(freeze, response, freeze), flow="pr"
    )
    assert len(_drop_unterminated_evidence_response_records(terminated)) == 3


def test_external_push_after_freeze_releases_evidence_and_marks_broken_head():
    """Row external-push-breaks-freeze (recovery unit)."""
    ledger = _evidence_ledger()
    resumed = _resume_pr_round(
        _comments(_freeze_metadata(ledger)), head_sha="def456", configured_reviewers=("codex",)
    )
    assert resumed is not None
    assert resumed.unrecorded_head_advance is True
    assert resumed.broken_evidence_freeze_head == _FROZEN_HEAD
    assert [item.lifecycle for item in resumed.prior_items] == ["evidence_deferred"]
    assert resumed.evidence_boundary.evidence_freeze.allowed_rounds == 5


@pytest.mark.parametrize(
    "phases, expected",
    [
        (("scheduler-prelaunch",), False),
        (("scheduler-prelaunch", "reconciliation"), True),
        (("authoritative",), True),  # legacy summary without a phase label
    ],
)
def test_pr_round_reconciled_only_by_a_reconciliation_summary(phases, expected):
    """Row leftover-reconciliation (#1142): prelaunch is a pre-reviewer checkpoint."""
    records = [
        PostedRoundMetadata(
            flow="pr", role="reviewer", agent="Codex", round_number=1,
            subject="head", state="approved", phase="publication",
        ),
        *(
            PostedRoundMetadata(
                flow="pr", role="summary", agent="Orchestrator", round_number=1,
                subject="head", phase=phase,
            )
            for phase in phases
        ),
    ]
    resumed = _resume_pr_round(_comments(*records), head_sha="head", configured_reviewers=("codex",))
    assert resumed is not None and resumed.reconciled is expected


def test_plan_execution_mode_fields_round_trip_and_are_omitted_when_absent():
    from coding_review_agent_loop.round_state import prior_plan_execution_mode

    base = dict(flow="plan", role="coder", agent="Claude", round_number=1, subject="s")
    legacy = PostedRoundMetadata(**base)
    assert "plan_execution_mode" not in decode_mapping(_encode_round_metadata(legacy))
    assert _decode_round_metadata(_encode_round_metadata(legacy)).plan_execution_mode is None

    recorded = PostedRoundMetadata(**base, plan_execution_mode="plan-only")
    decoded = _decode_round_metadata(_encode_round_metadata(recorded))
    assert decoded.plan_execution_mode == "plan-only"

    payload = decode_mapping(_encode_round_metadata(recorded))
    payload["plan_execution_mode"] = "not-a-mode"
    assert _decode_round_metadata(encode_mapping(payload)).plan_execution_mode is None

    from coding_review_agent_loop.round_state import PostedRoundRecord

    records = [
        PostedRoundRecord(index=0, metadata=recorded, body=""),
        PostedRoundRecord(index=1, metadata=legacy, body=""),
    ]
    assert prior_plan_execution_mode(records) == "plan-only"
    assert prior_plan_execution_mode([records[1]]) is None


def test_planning_only_scheduler_mode_is_invalid_on_a_pr_flow_checkpoint():
    assert "scheduler_execution_mode" not in decode_mapping(
        _encode_round_metadata(_checkpoint())
    )
    payload = decode_mapping(_encode_round_metadata(_checkpoint()))
    payload["scheduler_execution_mode"] = "plan-only"
    assert _decode_round_metadata(encode_mapping(payload)).scheduler_metadata_status == "invalid"


# --- Coder-recovery and head-review-recovery records (#1292) -----------------

from coding_review_agent_loop.protocol import ReviewItemDisposition, UnresolvedReviewItem  # noqa: E402
from coding_review_agent_loop.round_state import (  # noqa: E402
    CODER_DISPATCH_PHASE,
    CODER_FOLLOWUP_REJECTED_PHASE,
    HEAD_REVIEW_RECOVERY_PHASE,
    MAX_REJECTED_DISPATCH_ATTEMPTS,
    QualificationCheckpoint,
    RecoveryRoundBudget,
    _recovery_record_problems,
    pr_resume_needs_author_admission,
    sanitize_recovery_reason,
)

_ACTOR = ("agent-actor", 4242)
_OLD, _NEW, _NEWER = "a" * 40, "b" * 40, "c" * 40
_BUDGET = RecoveryRoundBudget(6, True, False)


def _item(number=1, text="Fix the thing."):
    return UnresolvedReviewItem(
        item_id=f"item-{number}", reviewer="Codex", source_round=1, text=text, status="blocking"
    )


def _comment(metadata, *, author_id=_ACTOR[1], author=_ACTOR[0]):
    return SimpleNamespace(
        body=_attach_round_metadata("record", metadata), author=author, author_id=author_id
    )


def _history(*metadata, forged=()):
    """Comments authored by the actor; indexes in ``forged`` come from someone else."""
    return [
        _comment(item, **({"author_id": 9, "author": "mallory"} if index in forged else {}))
        for index, item in enumerate(metadata)
    ]


def _coder(subject=_OLD, round_number=1, items=()):
    return PostedRoundMetadata(
        flow="pr", role="coder", agent="Claude", round_number=round_number,
        subject=subject, prior_items=tuple(items),
    )


def _reviewer(subject=_OLD, round_number=1, items=(), new_items=(), agent="Codex"):
    return PostedRoundMetadata(
        flow="pr", role="reviewer", agent=agent, round_number=round_number,
        subject=subject, prior_items=tuple(items), new_items=tuple(new_items), state="blocking",
    )


def _summary(phase, subject, round_number, items=(), **extra):
    return PostedRoundMetadata(
        flow="pr", role="summary", agent="Orchestrator", round_number=round_number,
        subject=subject, prior_items=tuple(items), state="blocking", phase=phase, **extra,
    )


def _dispatch(subject=_OLD, dispatch_round=1, items=(_item(),), attempt=1, reasons=(), **extra):
    values = dict(
        dispatch_round=dispatch_round, dispatch_head=subject, dispatch_attempt=attempt,
        recovery_dispatch=attempt > 1, carried_rejection_reasons=tuple(reasons),
        recovery_round_budget=_BUDGET,
    )
    values.update(extra)
    return _summary(CODER_DISPATCH_PHASE, subject, (dispatch_round or 1) + 1, items, **values)


def _rejection(subject=_NEW, dispatch_head=_OLD, dispatch_round=1, items=(_item(),), attempt=1,
               reason="live remote target", reasons=(), **extra):
    values = dict(
        dispatch_round=dispatch_round, dispatch_head=dispatch_head, dispatch_attempt=attempt,
        recovery_dispatch=attempt > 1, carried_rejection_reasons=tuple(reasons),
        rejected_coder_followup_reason=reason, rejected_coder_followup_from_head=dispatch_head,
        recovery_round_budget=_BUDGET,
    )
    values.update(extra)
    return _summary(CODER_FOLLOWUP_REJECTED_PHASE, subject, dispatch_round or 1, items, **values)


def _handoff(subject=_NEW, round_number=1, items=(_item(),), source="operator", budget=_BUDGET):
    return _summary(
        HEAD_REVIEW_RECOVERY_PHASE, subject, round_number, items,
        head_review_recovery_source=source, recovery_round_budget=budget,
    )


def _base_history():
    """Round-1 coder on A and a reviewer that raised item-1."""
    return [_coder(), _reviewer(new_items=(_item(),))]


def _resume(history, head, *, actor=_ACTOR, flag=False, forged=()):
    return _resume_pr_round(
        _history(*history, forged=forged), head_sha=head, configured_reviewers=("codex",),
        trusted_actor=actor, review_unrecorded_head=flag,
    )


def _legacy_checkpoint(lifecycle="repair_required", **extra):
    values = dict(
        obligation_kind="managed-exact-head-ci", obligation_identity="ci", lifecycle=lifecycle,
        failed_head_sha=_OLD, candidate_head_sha=None, allowed_rounds=6,
        watch_failure_extension_used=True, watch_head_extension_used=False,
    )
    if lifecycle == "awaiting_current_head_review":
        values["candidate_head_sha"] = _NEW
    values.update(extra)
    return QualificationCheckpoint(**values)


def _checkpoint_summary(round_number=2, items=(_item(),), **checkpoint_extra):
    return _summary(
        "qualification-checkpoint", _OLD, round_number, items,
        qualification_checkpoint=_legacy_checkpoint(**checkpoint_extra),
    )


def test_stranded_checkpoint_anchor_refuses_without_admission_and_names_the_flag():
    """Reproduces the #1370 incident: the pre-handoff checkpoint is the only prior-head anchor."""
    history = [*_base_history(), _checkpoint_summary()]
    with pytest.raises(AgentLoopError, match="PR head advanced without a recorded coder follow-up") as raised:
        _resume(history, _NEW, actor=None)
    assert "--review-unrecorded-head" in str(raised.value)
    # A forged anchor (authored by someone else) is not admitted either.
    with pytest.raises(AgentLoopError, match="PR head advanced"):
        _resume(history, _NEW, forged=(2,))


@pytest.mark.parametrize("lifecycle", ["repair_required", "awaiting_current_head_review"])
def test_legacy_checkpoint_anchor_resumes_as_an_ordinary_head_review(lifecycle):
    history = [*_base_history(), _checkpoint_summary(lifecycle=lifecycle)]
    resumed = _resume(history, _NEW)
    assert resumed is not None
    assert resumed.round_number == 2  # never rewound
    assert [item.item_id for item in resumed.prior_items] == ["item-1"]
    assert resumed.head_review_recovery == "legacy-checkpoint"
    assert resumed.head_review_recovery_post_required is True
    assert resumed.unrecorded_head_advance is False  # never routed through a coder first
    assert resumed.qualification_checkpoint is None  # the old head's checkpoint is not reused
    assert resumed.coder_output is None
    assert resumed.recovery_round_budget == RecoveryRoundBudget(6, True, False)
    assert resumed.next_unresolved_item_number == 2


def test_checkpoint_anchor_with_no_active_items_gets_a_full_fresh_review():
    history = [_coder(), _reviewer(), _checkpoint_summary(items=())]
    assert _resume(history, _NEW) is None


def test_post_push_rejection_resumes_a_coder_recovery_in_the_original_slot():
    history = [*_base_history(), _dispatch(), _rejection()]
    resumed = _resume(history, _NEW)
    assert resumed is not None
    assert resumed.unrecorded_head_advance is True
    assert resumed.round_number == 1  # the dispatch slot, not slot + 1
    assert resumed.rejected_coder_followup_reason == "live remote target"
    assert resumed.rejected_coder_followup_from_head == _OLD
    assert resumed.dispatch_attempt == 1
    assert resumed.recovery_round_budget == _BUDGET
    assert resumed.qualification_checkpoint is None
    assert resumed.carried_rejection_reasons == ("live remote target",)
    assert [item.item_id for item in resumed.prior_items] == ["item-1"]
    assert resumed.next_unresolved_item_number == 2


def test_unchanged_head_recovery_rejection_resumes_with_its_attempt():
    history = [
        *_base_history(), _dispatch(), _rejection(),
        _dispatch(subject=_NEW, attempt=2, reasons=("live remote target",)),
        _rejection(subject=_NEW, dispatch_head=_NEW, attempt=2, reason="again", reasons=("live remote target",)),
    ]
    with pytest.raises(AgentLoopError, match="--review-unrecorded-head") as raised:
        _resume(history, _NEW)
    assert "live remote target" in str(raised.value) and "again" in str(raised.value)


def test_recovery_dispatch_rejected_without_a_push_resumes_in_its_slot_with_the_attempt():
    history = [*_base_history(), _dispatch(), _rejection(attempt=1, reason="first")]
    # Rejection of attempt 1 on an unchanged head cannot exist; attempt-2 dispatch only:
    history = [*_base_history(), _dispatch(subject=_OLD, attempt=2, reasons=("first",))]
    with pytest.raises(AgentLoopError, match="consecutive"):
        _resume(history, _OLD)


def test_dispatch_record_alone_after_a_push_recovers_as_not_recorded():
    history = [*_base_history(), _dispatch()]
    resumed = _resume(history, _NEW)
    assert resumed is not None and resumed.unrecorded_head_advance is True
    assert resumed.dispatch_attempt == 1 and resumed.round_number == 1
    assert "not recorded" in resumed.rejected_coder_followup_reason


def test_valid_first_attempt_dispatch_on_the_current_head_is_ignored_by_resume():
    base = [*_base_history(), _checkpoint_summary()]
    with_dispatch = [*base, _dispatch(subject=_OLD, dispatch_round=2)]
    for head in (_OLD,):
        without = _resume(base, head)
        assert _resume(with_dispatch, head) == without


def test_later_progress_retires_a_rejection_record():
    rejected = [*_base_history(), _dispatch(), _rejection()]
    after_coder = [*rejected, _coder(subject=_NEW, round_number=2, items=(_item(),))]
    resumed = _resume(after_coder, _NEW)
    assert resumed is not None and resumed.rejected_coder_followup_reason is None
    assert resumed.unrecorded_head_advance is False
    after_reviewer = [*rejected, _reviewer(subject=_NEW, round_number=1, items=(_item(),))]
    resumed = _resume(after_reviewer, _NEW)
    assert resumed is not None and resumed.rejected_coder_followup_reason is None


def test_forged_recovery_records_are_ignored():
    history = [*_base_history(), _dispatch(), _rejection()]
    # Resume behaves as if the forged records were absent: the ordinary reviewer
    # recovery, with no rejection reason, attempt or budget supplied by a forgery.
    for kwargs in ({"forged": (2, 3)}, {"actor": None}):
        ignored = _resume(history, _NEW, **kwargs)
        assert ignored is not None
        assert ignored.rejected_coder_followup_reason is None
        assert ignored.dispatch_attempt is None and ignored.recovery_round_budget is None
    # Only the genuine dispatch survives a forged rejection: not-recorded recovery.
    survivor = _resume(history, _NEW, forged=(3,))
    assert survivor is not None and "not recorded" in survivor.rejected_coder_followup_reason


def test_rejection_budget_and_ledger_come_from_the_dispatch_snapshot():
    late = RecoveryRoundBudget(7, True, True)
    history = [
        *_base_history(),
        _rejection(items=(_item(1), _item(2, "Second.")), recovery_round_budget=late),
    ]
    resumed = _resume(history, _NEW)
    assert resumed.recovery_round_budget == late
    assert [item.item_id for item in resumed.prior_items] == ["item-1", "item-2"]


def test_rejection_on_prior_subject_survives_a_further_head_advance():
    history = [*_base_history(), _dispatch(), _rejection()]
    resumed = _resume(history, _NEWER)
    assert resumed is not None
    assert resumed.unrecorded_head_advance is True and resumed.round_number == 1
    assert resumed.rejected_coder_followup_from_head == _OLD
    assert resumed.dispatch_attempt == 1
    exhausted = [
        *_base_history(), _dispatch(),
        _rejection(attempt=2, reasons=("one",), reason="two"),
    ]
    with pytest.raises(AgentLoopError, match="consecutive"):
        _resume(exhausted, _NEWER)


def test_head_review_handoff_on_a_prior_subject_resumes_after_a_further_advance():
    budget = RecoveryRoundBudget(7, True, False)
    history = [*_base_history(), _handoff(subject=_NEW, round_number=3, budget=budget)]
    resumed = _resume(history, _NEWER)
    assert resumed is not None
    assert resumed.round_number == 3 and resumed.recovery_round_budget == budget
    assert resumed.head_review_recovery == "operator"
    assert resumed.head_review_recovery_post_required is True
    assert resumed.unrecorded_head_advance is False


def test_head_review_handoff_on_the_current_head_survives_later_pre_review_summaries():
    budget = RecoveryRoundBudget(7, True, False)
    history = [
        *_base_history(), _handoff(subject=_NEW, round_number=3, budget=budget),
        _summary("scheduler-prelaunch", _NEW, 3, (_item(),)),
        _summary("qualification-checkpoint", _NEW, 3, (_item(),),
                 qualification_checkpoint=_legacy_checkpoint(lifecycle="repair_required")),
    ]
    resumed = _resume(history, _NEW)
    assert resumed is not None and resumed.round_number == 3
    assert resumed.recovery_round_budget == budget
    assert resumed.qualification_checkpoint is None
    assert resumed.head_review_recovery_post_required is False
    assert resumed.reconciled is False


def test_partially_published_head_review_supersedes_the_older_checkpoint():
    budget = RecoveryRoundBudget(7, True, False)
    stale = _summary(
        "qualification-checkpoint", _NEW, 3, (_item(),),
        qualification_checkpoint=_legacy_checkpoint(lifecycle="repair_required", failed_head_sha=_NEW),
    )
    history = [
        *_base_history(), stale, _handoff(subject=_NEW, round_number=3, budget=budget),
        _reviewer(subject=_NEW, round_number=3, items=(_item(),)),
    ]
    resumed = _resume(history, _NEW)
    assert resumed is not None and resumed.round_number == 3
    assert resumed.qualification_checkpoint is None
    assert resumed.recovery_round_budget == budget
    assert resumed.reconciled is False
    assert [record.metadata.agent for record in resumed.completed_reviews] == ["Codex"]


def test_attempt_limit_and_malformed_records_are_overridable_only_by_the_operator_flag():
    exhausted = [*_base_history(), _dispatch(), _rejection(attempt=2, reason="two", reasons=("one",))]
    resumed = _resume(exhausted, _NEW, flag=True)
    assert resumed is not None
    assert resumed.head_review_recovery == "operator"
    assert resumed.head_review_recovery_post_required is True
    assert resumed.qualification_checkpoint is None
    assert resumed.recovery_round_budget == _BUDGET
    assert [item.item_id for item in resumed.prior_items] == ["item-1"]
    # An unrefused history ignores the flag.
    clean = [*_base_history()]
    assert _resume(clean, _OLD, flag=True) == _resume(clean, _OLD)


def test_operator_flag_takes_no_budget_from_a_malformed_record():
    broken = _dispatch(attempt=2, recovery_dispatch=False, recovery_round_budget=RecoveryRoundBudget(7, True, True))
    history = [*_base_history(), broken]
    with pytest.raises(AgentLoopError, match="malformed"):
        _resume(history, _OLD)
    resumed = _resume(history, _OLD, flag=True)
    assert resumed is not None and resumed.recovery_round_budget is None


def test_operator_flag_without_any_budget_source_leaves_defaults():
    history = [*_base_history(), _summary("scheduler-prelaunch", _OLD, 2, (_item(),))]
    resumed = _resume(history, _NEW, flag=True)
    assert resumed is not None
    assert resumed.round_number == 2 and resumed.recovery_round_budget is None


@pytest.mark.parametrize(
    "record, fields",
    [
        (_dispatch(attempt=2, recovery_dispatch=False), "recovery_dispatch"),
        (_dispatch(attempt=1, recovery_dispatch=True), "recovery_dispatch"),
        (_dispatch(dispatch_attempt=None), "dispatch_attempt"),
        (_dispatch(dispatch_round=None), "dispatch_round"),
        (_dispatch(recovery_round_budget=RecoveryRoundBudget.invalid()), "recovery_round_budget"),
        (_rejection(rejected_coder_followup_from_head=_NEWER), "rejected_coder_followup_from_head"),
        (_rejection(rejected_coder_followup_reason=None), "rejected_coder_followup_reason"),
        (_rejection(dispatch_attempt=3, recovery_dispatch=True), "dispatch_attempt"),
        (_handoff(source="nobody"), "head_review_recovery_source"),
    ],
)
def test_malformed_live_recovery_record_raises_before_any_field_is_used(record, fields):
    history = [*_base_history(), record]
    head = record.subject
    with pytest.raises(AgentLoopError, match=rf"comment index 2 .*malformed: .*{fields}") as raised:
        _resume(history, head)
    assert "--review-unrecorded-head" in str(raised.value)


def test_dispatch_whose_subject_differs_from_its_dispatch_head_is_malformed():
    bad = _dispatch()
    bad = PostedRoundMetadata(**{**bad.__dict__, "subject": _NEW})
    assert "subject" in _recovery_record_problems(bad)
    wrong_round = PostedRoundMetadata(**{**_dispatch().__dict__, "round_number": 1})
    assert "round_number" in _recovery_record_problems(wrong_round)


def test_both_rejection_shapes_and_writer_outputs_are_structurally_valid():
    assert _recovery_record_problems(_rejection()) == ()
    unchanged = _rejection(subject=_OLD, dispatch_head=_OLD, attempt=2, reasons=("x",))
    assert _recovery_record_problems(unchanged) == ()
    assert _recovery_record_problems(_dispatch()) == ()
    assert _recovery_record_problems(_handoff()) == ()


def test_a_retired_malformed_record_no_longer_affects_resume():
    broken = _dispatch(attempt=2, recovery_dispatch=False)
    history = [*_base_history(), broken, _handoff(subject=_OLD, round_number=1)]
    resumed = _resume(history, _OLD)
    assert resumed is not None and resumed.head_review_recovery == "operator"
    with_coder = [*_base_history(), broken, _coder(subject=_OLD, round_number=2, items=(_item(),))]
    assert _resume(with_coder, _OLD) is not None


def test_exhausted_coder_handoff_never_overrides_a_later_head_review_handoff():
    history = [
        *_base_history(), _dispatch(),
        _rejection(attempt=2, reason="two", reasons=("one",)),
        _handoff(subject=_NEW),
    ]
    resumed = _resume(history, _NEW)
    assert resumed is not None and resumed.head_review_recovery == "operator"


def test_reconciled_head_review_is_not_resumed_after_a_manual_push():
    history = [
        *_base_history(), _dispatch(),
        _rejection(attempt=2, reason="two", reasons=("one",)),
        _handoff(subject=_NEW),
        _reviewer(subject=_NEW, round_number=1, items=(_item(),)),
        _summary("reconciliation", _NEW, 1, (_item(),)),
    ]
    resumed = _resume(history, _NEWER)
    # The retired handoffs are skipped; B's reconciled ledger is rebuilt at its round.
    assert resumed is not None
    assert resumed.unrecorded_head_advance is True and resumed.round_number == 1
    assert resumed.rejected_coder_followup_reason is None


def test_item_numbering_continues_after_every_recovery_branch():
    high = (_item(7, "Seventh."),)
    for history, head in (
        ([_coder(), _reviewer(new_items=high), _dispatch(items=high), _rejection(items=high)], _NEW),
        ([_coder(), _reviewer(new_items=high), _checkpoint_summary(items=high)], _NEW),
        ([_coder(), _reviewer(new_items=high), _handoff(subject=_NEW, items=high)], _NEW),
    ):
        resumed = _resume(history, head)
        assert resumed is not None and resumed.next_unresolved_item_number == 8


def test_recovery_fields_round_trip_and_legacy_encodings_are_byte_identical():
    legacy = _checkpoint()
    payload = decode_mapping(_encode_round_metadata(legacy))
    for key in (
        "dispatch_round", "dispatch_head", "dispatch_attempt", "recovery_dispatch",
        "carried_rejection_reasons", "rejected_coder_followup_reason",
        "rejected_coder_followup_from_head", "head_review_recovery_source",
        "recovery_round_budget",
    ):
        assert key not in payload
    record = _rejection(attempt=2, reasons=("one",), reason="two")
    decoded = _decode_round_metadata(_encode_round_metadata(record))
    assert decoded == record
    assert decoded.recovery_dispatch is True
    assert _recovery_record_problems(decoded) == ()


def test_malformed_optional_recovery_fields_decode_as_absent_never_raise():
    payload = decode_mapping(_encode_round_metadata(_dispatch()))
    payload.update(
        dispatch_attempt="two", dispatch_round=None, recovery_dispatch="yes",
        recovery_round_budget={"allowed_rounds": "x"}, carried_rejection_reasons="nope",
        head_review_recovery_source=3,
    )
    decoded = _decode_round_metadata(encode_mapping(payload))
    assert decoded.dispatch_attempt is None and decoded.dispatch_round is None
    assert decoded.recovery_dispatch is None
    assert decoded.recovery_round_budget == RecoveryRoundBudget.invalid()
    assert decoded.carried_rejection_reasons == ()
    problems = _recovery_record_problems(decoded)
    assert {"dispatch_attempt", "dispatch_round", "recovery_dispatch", "recovery_round_budget"} <= set(problems)


def test_rejection_reasons_are_bounded_and_redacted():
    reason = sanitize_recovery_reason("failed with ghp_abcdefghijklmnop " + "x" * 5000)
    assert len(reason) <= 2000
    assert "ghp_abcdefghijklmnop" not in reason and "[redacted]" in reason
    assert sanitize_recovery_reason("   ") == "rejected without a reason"
    assert "AGENT_STATE" not in sanitize_recovery_reason("<!-- AGENT_STATE: approved -->")


def test_author_admission_is_needed_only_for_new_records_head_advances_or_the_flag():
    ordinary = _history(*_base_history())
    assert pr_resume_needs_author_admission(ordinary, _OLD) is False
    assert pr_resume_needs_author_admission(ordinary, _NEW) is True  # head has no records
    assert pr_resume_needs_author_admission(ordinary, _OLD, True) is True
    assert pr_resume_needs_author_admission(_history(*_base_history(), _dispatch()), _OLD) is True
    assert pr_resume_needs_author_admission([], _OLD) is False


@pytest.mark.parametrize("carrier", ["checkpoint", "ordinary-phase-budget"])
def test_operator_budget_is_never_taken_from_a_forged_record(carrier):
    """A record the authenticated actor did not author cannot grant rounds or reset extension flags."""
    genuine = _checkpoint_summary(round_number=2, allowed_rounds=6)
    if carrier == "checkpoint":
        forged = _checkpoint_summary(
            round_number=2, allowed_rounds=7, watch_failure_extension_used=False
        )
    else:
        forged = _summary(
            "scheduler-prelaunch", _OLD, 2, (_item(),),
            recovery_round_budget=RecoveryRoundBudget(7, False, False),
        )
    history = [*_base_history(), genuine, forged]
    resumed = _resume(history, _NEW, flag=True, forged=(3,))
    assert resumed is not None
    assert resumed.recovery_round_budget == RecoveryRoundBudget(6, True, False)
    # With the genuine checkpoint also absent no budget source exists at all.
    only_forged = [*_base_history(), forged]
    assert _resume(only_forged, _NEW, flag=True, forged=(2,)).recovery_round_budget is None
    # An ordinary phase never supplies a recovery budget, even when authored by the actor.
    if carrier == "ordinary-phase-budget":
        assert _resume(only_forged, _NEW, flag=True).recovery_round_budget is None


def test_partially_published_recovery_keeps_the_global_item_number_high_water_mark():
    high = tuple(_item(n, f"Item {n}.") for n in range(1, 8))
    history = [
        _coder(), _reviewer(new_items=high),
        _handoff(subject=_NEW, items=(_item(1),)),
        _reviewer(subject=_NEW, round_number=1, items=(_item(1),)),
    ]
    resumed = _resume(history, _NEW)
    assert resumed is not None and resumed.next_unresolved_item_number == 8


@pytest.mark.parametrize("stage", ["recovery-coder-record", "coder-then-reviewer"])
def test_item_numbers_stay_global_after_a_successful_recovery_push(stage):
    """Generalizes the partial-publication rule: no reconstruction restarts numbering per head."""
    high = tuple(_item(n, f"Item {n}.") for n in range(1, 8))
    history = [
        _coder(), _reviewer(new_items=high),
        _dispatch(items=(_item(1),)), _rejection(items=(_item(1),)),
        _coder(subject=_NEWER, round_number=2, items=(_item(1),)),
    ]
    if stage == "coder-then-reviewer":
        history.append(_reviewer(subject=_NEWER, round_number=2, items=(_item(1),)))
    resumed = _resume(history, _NEWER)
    assert resumed is not None and resumed.next_unresolved_item_number == 8


def test_partial_head_review_recovery_ignores_records_published_before_its_handoff():
    """Pre-handoff reviewer/coder records sharing head, round and ledger are not replayed."""
    items = (_item(),)
    history = [
        _coder(subject=_NEW, round_number=3, items=items),
        _reviewer(subject=_NEW, round_number=3, items=items, agent="Codex"),
        _summary("reconciliation", _NEW, 3, items),
        _handoff(subject=_NEW, round_number=3, items=items),
        _reviewer(subject=_NEW, round_number=3, items=items, agent="Gemini"),
    ]
    resumed = _resume_pr_round(
        _history(*history), head_sha=_NEW, configured_reviewers=("codex", "gemini"),
        trusted_actor=_ACTOR,
    )
    assert resumed is not None
    assert [r.metadata.agent for r in resumed.completed_reviews] == ["Gemini"]
    assert resumed.coder_output is None and resumed.coder_metadata is None
    assert resumed.reconciled is False


def test_prior_head_reconstruction_stops_at_the_head_review_handoff_boundary():
    """Cleared items stay cleared and pre-handoff coder context is absent after a head advance."""
    items = (_item(),)

    def verdict(disposition):
        return dataclasses.replace(
            _reviewer(subject=_NEW, round_number=3, items=items),
            dispositions=(
                ReviewItemDisposition(item_id="item-1", reviewer="Codex", disposition=disposition),
            ),
        )

    history = [
        _coder(subject=_NEW, round_number=3, items=items),
        verdict("blocking"),  # published before the handoff
        _handoff(subject=_NEW, round_number=3, items=items),
        verdict("resolved"),  # the completed fresh recovery review
    ]
    # The fresh review cleared item-1; a manual push to C must not revive it.
    assert _resume(history, _NEWER) is None
    # With the fresh review still blocking, the ledger carries over but the
    # pre-handoff coder response is never restored.
    resumed = _resume([*history[:3], verdict("blocking")], _NEWER)
    assert resumed is not None and resumed.coder_output is None


def test_durable_sanitizer_strips_coverage_gaps_like_claims():
    import json
    from coding_review_agent_loop.round_state import _sanitize_durable_coder_response

    payload = {
        "kind": "issue_implementation",
        "summary": "s",
        "risk_test_matrix_claims": [{"row_id": "r"}],
        "risk_test_matrix_coverage_gaps": [{"row_id": "r", "reason": "x", "proposed_correction": "y"}],
    }
    sanitized = _sanitize_durable_coder_response(json.dumps(payload) + "\n<!-- AGENT_STATE: blocking -->")
    body = json.loads(sanitized.split("\n<!--")[0])
    assert "risk_test_matrix_claims" not in body
    assert "risk_test_matrix_coverage_gaps" not in body
    gaps_only = {"kind": "coder_followup", "risk_test_matrix_coverage_gaps": []}
    assert "risk_test_matrix_coverage_gaps" not in _sanitize_durable_coder_response(json.dumps(gaps_only))


# --- evidence-only stall persistence (#1324) -------------------------------

def _stall_payload(**overrides):
    payload = {
        "version": 1, "review_round": 2, "review_head": "a" * 40, "matrix_identity": "b" * 64,
        "qualifies": True, "unsatisfied_row_ids": ["row-b", "row-a"], "reasons": [],
    }
    payload.update(overrides)
    return payload


def _coder_record(**overrides):
    values = dict(flow="pr", role="coder", agent="Claude", round_number=3, subject="a" * 40)
    values.update(overrides)
    return PostedRoundMetadata(**values)


def test_evidence_stall_snapshot_round_trips_sorted_and_is_omitted_when_none():
    legacy = _coder_record()
    assert "evidence_stall" not in decode_mapping(_encode_round_metadata(legacy))
    assert _decode_round_metadata(_encode_round_metadata(legacy)) == legacy

    record = _coder_record(evidence_stall=_stall_payload())
    decoded = _decode_round_metadata(_encode_round_metadata(record))
    assert decoded.evidence_stall["unsatisfied_row_ids"] == ["row-a", "row-b"]
    assert _decode_round_metadata(_encode_round_metadata(decoded)) == decoded


@pytest.mark.parametrize(
    "bad",
    [
        _stall_payload(review_round=3),  # must equal enclosing round - 1
        _stall_payload(qualifies="yes"),
        _stall_payload(unsatisfied_row_ids="row-a"),
        _stall_payload(version=2),
    ],
)
def test_malformed_or_misnumbered_stall_snapshot_decodes_invalid_not_error(bad):
    decoded = _decode_round_metadata(_encode_round_metadata(_coder_record(evidence_stall=bad)))
    assert decoded.evidence_stall == {"invalid": True}


def test_evidence_row_ids_tag_round_trips_and_legacy_items_are_byte_stable():
    from coding_review_agent_loop.protocol import UnresolvedReviewItem

    plain = UnresolvedReviewItem(item_id="item-1", reviewer="Codex", source_round=1, text="t", status="blocking")
    tagged = dataclasses.replace(plain, item_id="item-2", evidence_row_ids=("row-a", "row-b"))
    encoded_plain = _encode_round_metadata(_coder_record(prior_items=(plain,)))
    assert "evidence_row_ids" not in encoded_plain and "evidence_row_ids" not in decode_mapping(encoded_plain)["prior_items"][0]
    decoded = _decode_round_metadata(_encode_round_metadata(_coder_record(prior_items=(plain, tagged))))
    assert [item.evidence_row_ids for item in decoded.prior_items] == [(), ("row-a", "row-b")]


@pytest.mark.parametrize("bad", ["row-a", [], [1], ["a", "a"], [""]])
def test_malformed_stored_tag_degrades_to_an_ordinary_finding(bad):
    from coding_review_agent_loop.protocol import UnresolvedReviewItem
    from coding_review_agent_loop.round_state import _deserialize_unresolved_item, _serialize_unresolved_item

    item = UnresolvedReviewItem(item_id="item-1", reviewer="Codex", source_round=1, text="t", status="blocking")
    payload = _serialize_unresolved_item(item)
    payload["evidence_row_ids"] = bad
    assert _deserialize_unresolved_item(payload).evidence_row_ids == ()


def test_non_mapping_stall_value_in_stored_payload_decodes_invalid():
    stored = decode_mapping(_encode_round_metadata(_coder_record()))
    stored["evidence_stall"] = "not-a-mapping"
    assert _decode_round_metadata(encode_mapping(stored)).evidence_stall == {"invalid": True}


_CONTRADICTORY_STALL_SNAPSHOTS = [
    pytest.param(_stall_payload(reasons=["checks-failing"]), id="qualifies-with-checks-failing"),
    pytest.param(_stall_payload(reasons=["untagged-finding"]), id="qualifies-with-untagged-finding"),
    pytest.param(_stall_payload(qualifies=False, reasons=["made-up-reason"]), id="unknown-reason"),
    pytest.param(_stall_payload(qualifies=False, reasons=[]), id="non-qualifying-without-reason"),
    pytest.param(_stall_payload(unsatisfied_row_ids=[]), id="qualifies-without-rows"),
    pytest.param(_stall_payload(unsatisfied_row_ids=["row-a", "row-a"]), id="duplicate-rows"),
    pytest.param(_stall_payload(review_head=""), id="qualifies-without-head"),
    pytest.param(_stall_payload(matrix_identity=""), id="qualifies-without-identity"),
    pytest.param(
        _stall_payload(qualifies=False, reasons=["checks-failing", "checks-failing"]),
        id="duplicate-reasons",
    ),
]


@pytest.mark.parametrize("bad", _CONTRADICTORY_STALL_SNAPSHOTS)
def test_contradictory_stall_snapshot_decodes_invalid_and_never_counts_toward_k(bad):
    """`stall-legacy-compat` / `stall-window-reset`: corrupt history is non-qualifying."""
    from coding_review_agent_loop.evidence_stall import StallRoundSnapshot, stall_window

    decoded = _decode_round_metadata(_encode_round_metadata(_coder_record(evidence_stall=bad)))
    assert decoded.evidence_stall == {"invalid": True}
    current = StallRoundSnapshot(
        review_round=3, review_head="a" * 40, matrix_identity="b" * 64, qualifies=True,
        unsatisfied_row_ids=("row-a", "row-b"), reasons=(),
    )
    # Neither the decoded value nor the raw contradictory payload counts toward K.
    for prior in (decoded.evidence_stall, bad):
        window = stall_window(current, {2: prior}, 2)
        assert window.stop is False and window.length == 1


def test_consistent_non_qualifying_stall_snapshot_round_trips():
    record = _coder_record(
        evidence_stall=_stall_payload(qualifies=False, reasons=["untagged-finding", "checks-failing"])
    )
    decoded = _decode_round_metadata(_encode_round_metadata(record))
    assert decoded.evidence_stall["reasons"] == ["checks-failing", "untagged-finding"]
    assert decoded.evidence_stall["qualifies"] is False


# --- plan board identity fields are flow-scoped (#1377) --------------------------------


_PLAN_ROOT = {
    "policy": "all-reviewers", "primary": None, "broad_rules": [],
    "seats": [{"id": "Codex", "backend": "codex", "model_chain": None, "effort": None}],
}


@pytest.mark.parametrize(
    "flow,fields",
    [
        ("plan", {"pr_board_root": _PLAN_ROOT, "pr_board_identity": _PLAN_ROOT}),
        ("pr", {"plan_board_root": _PLAN_ROOT, "plan_board_identity": _PLAN_ROOT}),
        ("plan", {"plan_board_root": _PLAN_ROOT}),
        ("plan", {"plan_board_identity": _PLAN_ROOT}),
        ("plan", {"plan_board_root": "not-an-object", "plan_board_identity": _PLAN_ROOT}),
    ],
    ids=["pr-fields-on-plan", "plan-fields-on-pr", "root-only", "identity-only", "malformed"],
)
def test_board_identity_fields_are_flow_scoped_and_paired(flow, fields):
    """Row plan-metadata-cross-flow: the wrong flow's fields, or half a pair, are malformed."""
    from coding_review_agent_loop.round_state import (
        PostedRoundMetadata,
        _attach_round_metadata,
        _decode_round_metadata_mapping,
        _extract_round_metadata_records,
    )

    payload = {"flow": flow, "role": "reviewer", "agent": "Codex", "round_number": 1,
               "subject": "abc123", **fields}
    with pytest.raises(AgentLoopError, match="Invalid AGENT_LOOP_META payload"):
        _decode_round_metadata_mapping(payload)
    with pytest.raises(ValueError):
        PostedRoundMetadata(flow=flow, role="reviewer", agent="Codex", round_number=1,
                            subject="abc123", **fields)
    # A well-formed plan record round-trips its plan board fields through the encoder.
    good = PostedRoundMetadata(
        flow="plan", role="reviewer", agent="Codex", round_number=1, subject="abc123",
        plan_board_root=_PLAN_ROOT, plan_board_identity=_PLAN_ROOT,
    )
    (record,) = _extract_round_metadata_records(
        [SimpleNamespace(body=_attach_round_metadata("Review.\n-- OpenAI Codex", good))], flow="plan",
    )
    assert record.metadata.plan_board_root == _PLAN_ROOT
    assert record.metadata.pr_board_root is None


def test_plan_board_stamp_applies_only_to_plan_flow_records():
    """Row plan-metadata-cross-flow: the plan stamp never touches PR-flow records."""
    from coding_review_agent_loop.board_change import PLAN_SURFACE, PrBoardStamp
    from coding_review_agent_loop.round_state import (
        PostedRoundMetadata,
        _attach_round_metadata,
        _extract_round_metadata_records,
        reset_plan_board_stamp,
        set_plan_board_stamp,
    )

    token = set_plan_board_stamp(PrBoardStamp(
        root=_PLAN_ROOT, identity=_PLAN_ROOT, change_digest=None, amendment_digest=None,
        surface=PLAN_SURFACE,
    ))
    try:
        bodies = [
            _attach_round_metadata("x\n-- Orchestrator", PostedRoundMetadata(
                flow=flow, role="summary", agent="Orchestrator", round_number=1, subject="s",
            ))
            for flow in ("plan", "pr")
        ]
    finally:
        reset_plan_board_stamp(token)
    (plan,) = _extract_round_metadata_records([SimpleNamespace(body=bodies[0])], flow="plan")
    (pr,) = _extract_round_metadata_records([SimpleNamespace(body=bodies[1])], flow="pr")
    assert plan.metadata.plan_board_root == _PLAN_ROOT
    assert pr.metadata.plan_board_root is None and pr.metadata.pr_board_root is None


# --- Deferred publication replay (#1389) ------------------------------------

from coding_review_agent_loop.protocol import ReviewItemDisposition  # noqa: E402
from coding_review_agent_loop.round_state import (  # noqa: E402
    UnresolvedReviewItem,
    canonical_deferred_history_items,
    canonical_history_item_outcomes,
)


def _deferred_item(item_id, status):
    return UnresolvedReviewItem(
        item_id=item_id, reviewer="Codex", source_round=1,
        text=f"finding {item_id}", status=status,
    )


def _deferred_record(index, role, round_number, subject, **kw):
    return PostedRoundRecord(
        index=index,
        metadata=PostedRoundMetadata(
            flow="pr", role=role, agent=kw.pop("agent", "Codex"),
            round_number=round_number, subject=subject, **kw,
        ),
        body="",
    )


def test_canonical_deferred_history_items_tracks_eligibility_separately():
    x = _deferred_item("item-1", "future")      # new future follow-up on a blocking review
    y = _deferred_item("item-2", "blocking")    # blocker later dispositioned future
    z = _deferred_item("item-3", "future")      # deferred, then re-promoted by the final review
    w = _deferred_item("item-4", "future")      # full-context ledger future, later resolved
    round_1 = _deferred_record(0, "reviewer", 1, "h0", state="blocking", new_items=(x, y, z, w))
    ledger = (x, y, z, w)
    coder_2 = _deferred_record(1, "coder", 2, "h1", agent="Claude", prior_items=ledger)
    final_review = _deferred_record(
        2, "reviewer", 2, "h1", state="blocking", prior_items=ledger,
        dispositions=(
            ReviewItemDisposition("item-2", "Codex", "future", "off-scope hardening"),
            ReviewItemDisposition("item-3", "Codex", "blocking", "must be fixed now"),
            ReviewItemDisposition("item-4", "Codex", "resolved"),
        ),
    )
    records = [round_1, coder_2, final_review]
    before = canonical_history_item_outcomes(
        records, reconciliation_mode="aggregate", same_status="same-pr"
    )

    deferred = canonical_deferred_history_items(
        records, reconciliation_mode="aggregate", same_status="same-pr"
    )

    assert [item.item_id for item in deferred] == ["item-1", "item-2"]
    assert all(item.status == "future" for item in deferred)
    # The legacy outcomes map is unchanged by the new replay (it does not reset
    # a kept re-promotion, which is why eligibility is tracked separately).
    assert before == {
        "item-1": "deferred", "item-2": "deferred", "item-3": "deferred", "item-4": "resolved",
    }
    assert canonical_history_item_outcomes(
        records, reconciliation_mode="aggregate", same_status="same-pr"
    ) == before


def test_canonical_deferred_history_items_drops_reappearing_mandatory_item_and_machine_items():
    y = _deferred_item("item-2", "blocking")
    ci = UnresolvedReviewItem(
        item_id="ci-item", reviewer="Orchestrator", source_round=1, text="checks",
        status="future", authority="machine", obligation_kind="github-pr-checks",
        lifecycle="repair_required", failed_head_sha="c" * 40, obligation_identity="ci-1",
    )
    round_1 = _deferred_record(0, "reviewer", 1, "h0", state="blocking", new_items=(y, ci))
    downgrade = _deferred_record(
        1, "reviewer", 2, "h1", state="blocking", prior_items=(y,),
        dispositions=(ReviewItemDisposition("item-2", "Codex", "future", "later"),),
    )
    assert [
        item.item_id
        for item in canonical_deferred_history_items(
            [round_1, downgrade], reconciliation_mode="aggregate", same_status="same-pr"
        )
    ] == ["item-2"]
    reintroduced = _deferred_record(2, "coder", 3, "h2", agent="Claude", prior_items=(y,))
    assert canonical_deferred_history_items(
        [round_1, downgrade, reintroduced], reconciliation_mode="aggregate", same_status="same-pr"
    ) == ()


# --- evidence conversion records (#1325) -----------------------------------------

_CONV_IDENTITY = "c" * 64
_CONV_HEAD = "a" * 40


def _conversion_fixtures():
    from coding_review_agent_loop.evidence_conversion import (
        ConversionAuditRow,
        EvidenceConversionAudit,
    )
    from coding_review_agent_loop.protocol import SignedEvidenceReference, UnresolvedReviewItem
    from coding_review_agent_loop.unresolved_items import (
        _upsert_converted_evidence_obligation,
        converted_evidence_obligation_identity,
    )

    _ledger, obligation, _ = _upsert_converted_evidence_obligation(
        [], item_number=4, row_id="row-a", matrix_identity=_CONV_IDENTITY, head_sha=_CONV_HEAD,
        owners=("Codex", "Gemini"), source_round=2, explanation="canonical-status:missing",
    )
    finding = UnresolvedReviewItem(
        item_id="item-1", reviewer="Codex", source_round=1, text="Capture evidence",
        status="converted", evidence_row_ids=("row-a",), converted_row_ids=("row-a",),
    )
    partial = UnresolvedReviewItem(
        item_id="item-2", reviewer="Gemini", source_round=1, text="Capture more evidence",
        status="blocking", evidence_row_ids=("row-a", "row-b"), converted_row_ids=("row-a",),
    )
    audit = EvidenceConversionAudit(
        source="auto", matrix_identity=_CONV_IDENTITY, head=_CONV_HEAD,
        rows=(ConversionAuditRow(
            row_id="row-a",
            obligation_identity=converted_evidence_obligation_identity("row-a", _CONV_IDENTITY),
            obligation_item_id="item-4", owners=("Codex", "Gemini"),
            converted_item_ids=("item-1", "item-2"), attempt_record_rounds=(1, 2),
        ),),
    )
    reference = SignedEvidenceReference.build(
        matrix_identity=_CONV_IDENTITY, row_id="row-a", head=_CONV_HEAD,
        obligation_identity=obligation.obligation_identity, obligation_item_id="item-4",
        owners=("Codex", "Gemini"), signed_requirement_ids=("hr-1",),
        clearing_reviewers=("Codex", "Gemini"), cleared_round=3,
    )
    return obligation, finding, partial, audit, reference


def _conversion_record(**overrides):
    obligation, finding, partial, audit, reference = _conversion_fixtures()
    values = dict(
        flow="pr", role="summary", agent="Orchestrator", round_number=2, subject=_CONV_HEAD,
        prior_items=(finding, partial, obligation), state="blocking", phase="evidence-conversion",
        evidence_conversion=audit,
        conversion_stall_snapshot=_stall_payload(
            review_round=2, bound_attempt_round=2, unsatisfied_row_ids=["row-a", "row-b"]
        ),
        conversion_review_round=2,
        signed_evidence_references=(reference,),
    )
    values.update(overrides)
    return PostedRoundMetadata(**values)


def test_conversion_record_fields_round_trip_identically_after_restart():
    """`convert-restart`: audit, snapshot, converted status and references decode identically."""
    record = _conversion_record()
    decoded = _decode_round_metadata(_encode_round_metadata(record))
    assert decoded == record
    obligation, finding, partial, audit, reference = _conversion_fixtures()
    assert decoded.evidence_conversion == audit
    assert decoded.signed_evidence_references == (reference,)
    assert decoded.conversion_stall_snapshot["bound_attempt_round"] == 2
    items = {item.item_id: item for item in decoded.prior_items}
    assert items["item-1"].status == "converted" and items["item-1"].is_inert_ledger_record
    assert items["item-2"].residual_evidence_row_ids == ("row-b",)
    assert items["item-4"].converted_row_id == "row-a"
    assert items["item-4"].obligation_identity == obligation.obligation_identity


def test_invalid_conversion_fields_decode_as_unconverted_without_raising():
    stored = decode_mapping(_encode_round_metadata(_conversion_record()))
    stored["evidence_conversion"]["rows"][0]["obligation_identity"] = "forged"
    stored["signed_evidence_references"][0]["head"] = "b" * 40
    stored["conversion_review_round"] = 7
    stored["conversion_stall_snapshot"]["review_round"] = 1
    decoded = _decode_round_metadata(encode_mapping(stored))
    assert decoded.evidence_conversion is None
    assert decoded.signed_evidence_references == ()
    assert decoded.conversion_review_round is None
    assert decoded.conversion_stall_snapshot == {"invalid": True}


def test_malformed_converted_item_fields_fail_safe():
    from coding_review_agent_loop.round_state import _deserialize_unresolved_item, _serialize_unresolved_item

    obligation, finding, partial, _audit, _reference = _conversion_fixtures()
    bad_binding = _serialize_unresolved_item(obligation)
    bad_binding["converted_row_id"] = "row-z"  # no longer reproduces the identity
    assert _deserialize_unresolved_item(bad_binding).converted_row_id is None
    bad_subset = _serialize_unresolved_item(finding)
    bad_subset["converted_row_ids"] = ["row-z"]
    restored = _deserialize_unresolved_item(bad_subset)
    # An inconsistent converted finding is restored as an open blocker.
    assert (restored.status, restored.converted_row_ids) == ("blocking", ())


def test_legacy_records_without_conversion_fields_decode_and_reencode_unchanged():
    """`legacy-evidence-compat`: no new required fields; reviewer identities unchanged."""
    from coding_review_agent_loop.protocol import UnresolvedReviewItem
    from coding_review_agent_loop.unresolved_items import (
        _upsert_evidence_obligation,
        evidence_obligation_identity,
    )

    ledger, _consumed = _upsert_evidence_obligation(
        [], item_number=1, reviewer="Codex", text="Run the live suite at this head",
        source_round=1, current_head_sha=_CONV_HEAD,
    )
    [request] = ledger
    assert request.obligation_identity == evidence_obligation_identity("Run the live suite at this head", "Codex")
    assert ":row:" not in request.obligation_identity and request.converted_row_id is None
    legacy = _coder_record(
        prior_items=(request, UnresolvedReviewItem(
            item_id="item-2", reviewer="Codex", source_round=1, text="t", status="blocking",
        )),
        evidence_stall=_stall_payload(unsatisfied_row_ids=["row-a", "row-b"]),
    )
    encoded = _encode_round_metadata(legacy)
    payload = decode_mapping(encoded)
    for key in (
        "risk_test_matrix_capture_classification", "evidence_conversion",
        "conversion_stall_snapshot", "conversion_review_round", "signed_evidence_references",
    ):
        assert key not in payload
    for item in payload["prior_items"]:
        assert not {"converted_row_ids", "converted_row_id", "converted_matrix_identity"} & set(item)
    decoded = _decode_round_metadata(encoded)
    assert decoded == legacy
    assert _encode_round_metadata(decoded) == encoded
    assert "bound_attempt_round" not in decoded.evidence_stall


def test_resume_adopts_an_admitted_conversion_record_and_drops_a_foreign_one():
    from coding_review_agent_loop.round_state import _resume_pr_round

    coder = PostedRoundMetadata(
        flow="pr", role="coder", agent="Claude", round_number=2, subject=_CONV_HEAD,
        raw_structured_coder_response="{}",
    )
    record = _conversion_record()
    comments = [
        IssueComment(author="coding-review-agent-loop", created_at="2026-10-01T00:00:01Z", body=_attach_round_metadata("Coder.", coder), author_id=424242),
        IssueComment(author="coding-review-agent-loop", created_at="2026-10-01T00:00:02Z", body=_attach_round_metadata("Converted.", record), author_id=424242),
    ]
    trusted = ("coding-review-agent-loop", 424242)
    resumed = _resume_pr_round(
        comments, head_sha=_CONV_HEAD, configured_reviewers=("codex",), trusted_actor=trusted,
    )
    assert resumed.round_number == 2
    assert {item.item_id for item in resumed.prior_items} == {"item-1", "item-2", "item-4"}
    assert resumed.completed_reviews == ()
    foreign = [
        comments[0],
        IssueComment(author="mallory", created_at="2026-10-01T00:00:02Z", body=comments[1].body, author_id=999),
    ]
    resumed_foreign = _resume_pr_round(
        foreign, head_sha=_CONV_HEAD, configured_reviewers=("codex",), trusted_actor=trusted,
    )
    assert resumed_foreign is None or not any(
        item.converted_row_id for item in resumed_foreign.prior_items
    )


# --- Plan finding provenance, repeat marker and sweep marker (#1505) -----------


def _plan_reviewer_metadata(**overrides):
    from coding_review_agent_loop.round_state import PostedRoundMetadata

    values = dict(
        flow="plan", role="reviewer", agent="Codex", round_number=2,
        subject="plan", state="blocking",
    )
    values.update(overrides)
    return PostedRoundMetadata(**values)


def test_provenance_repeat_and_sweep_marker_round_trip_and_legacy_stays_byte_stable():
    """`persistence-roundtrip`: set values round-trip, unset values are omitted."""
    from coding_review_agent_loop.protocol import UnresolvedReviewItem

    plain = UnresolvedReviewItem(item_id="item-1", reviewer="Codex", source_round=2, text="t", status="blocking")
    tagged = dataclasses.replace(
        plain, item_id="item-2", provenance="revision_introduced", repeat_of="item-1"
    )
    legacy = _plan_reviewer_metadata(new_items=(plain,))
    encoded_legacy = _encode_round_metadata(legacy)
    stored_legacy = decode_mapping(encoded_legacy)
    assert "plan_sweep" not in stored_legacy
    assert {"provenance", "repeat_of"}.isdisjoint(stored_legacy["new_items"][0])
    assert _encode_round_metadata(_decode_round_metadata(encoded_legacy)) == encoded_legacy

    swept = _plan_reviewer_metadata(new_items=(plain, tagged), plan_sweep=True)
    encoded = _encode_round_metadata(swept)
    assert decode_mapping(encoded)["plan_sweep"] is True
    decoded = _decode_round_metadata(encoded)
    assert decoded.plan_sweep is True
    assert [(item.provenance, item.repeat_of) for item in decoded.new_items] == [
        (None, None), ("revision_introduced", "item-1"),
    ]
    assert _encode_round_metadata(decoded) == encoded


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("provenance", "unknown"), ("provenance", 3), ("provenance", ["pre_existing"]),
        ("repeat_of", ""), ("repeat_of", 7), ("repeat_of", ["item-1"]),
    ],
)
def test_malformed_stored_provenance_or_repeat_decodes_as_absent(key, bad, caplog):
    from coding_review_agent_loop.protocol import UnresolvedReviewItem
    from coding_review_agent_loop.round_state import _deserialize_unresolved_item, _serialize_unresolved_item

    item = UnresolvedReviewItem(item_id="item-1", reviewer="Codex", source_round=1, text="t", status="blocking")
    payload = _serialize_unresolved_item(item)
    payload[key] = bad
    with caplog.at_level("WARNING"):
        restored = _deserialize_unresolved_item(payload)
    assert getattr(restored, key) is None
    assert restored.status == "blocking" and restored.text == "t"
    assert any(key in message for message in caplog.messages)


@pytest.mark.parametrize(
    ("value", "overrides"),
    [("yes", {}), (1, {}), (False, {}), (True, {"role": "coder"}), (True, {"flow": "pr"})],
)
def test_malformed_plan_sweep_marker_decodes_as_absent_without_degrading(value, overrides, caplog):
    stored = decode_mapping(_encode_round_metadata(_plan_reviewer_metadata(**overrides)))
    stored["plan_sweep"] = value
    with caplog.at_level("WARNING"):
        decoded = _decode_round_metadata(encode_mapping(stored))
    assert decoded.plan_sweep is False
    # A malformed sweep marker never degrades the step-back or stall history.
    assert decoded.step_back_status == "absent"
    assert any("plan_sweep" in message for message in caplog.messages)


def test_provenance_baseline_field_round_trips_and_is_omitted_when_unset():
    """`record-field-roundtrip` (#1516)."""
    unset = _encode_round_metadata(_plan_reviewer_metadata())
    assert "provenance_baseline" not in decode_mapping(unset)
    assert _decode_round_metadata(unset).provenance_baseline_form is None
    for form, chars in (("compact-diff", 27812), ("full-baseline", 0), ("unavailable", 0)):
        metadata = _plan_reviewer_metadata(
            provenance_baseline_form=form, provenance_baseline_chars=chars, plan_sweep=True,
        )
        encoded = _encode_round_metadata(metadata)
        assert decode_mapping(encoded)["provenance_baseline"] == {"form": form, "chars": chars}
        decoded = _decode_round_metadata(encoded)
        assert (decoded.provenance_baseline_form, decoded.provenance_baseline_chars) == (
            form, chars,
        )
        assert decoded.plan_sweep is True
        assert _encode_round_metadata(decoded) == encoded


@pytest.mark.parametrize(
    ("value", "overrides"),
    [
        ("diff", {}),
        ({"form": "patch", "chars": 1}, {}),
        ({"form": "diff", "chars": -1}, {}),
        ({"form": "diff", "chars": "10"}, {}),
        ({"form": "diff", "chars": True}, {}),
        ({"form": "diff"}, {}),
        ({"form": "diff", "chars": 1, "extra": 1}, {}),
        ({"form": 3, "chars": 1}, {}),
        ({"form": "diff", "chars": 1}, {"role": "coder"}),
        ({"form": "diff", "chars": 1}, {"flow": "pr"}),
    ],
)
def test_malformed_provenance_baseline_decodes_as_absent_without_degrading(
    value, overrides, caplog
):
    stored = decode_mapping(
        _encode_round_metadata(_plan_reviewer_metadata(plan_sweep=True, **overrides))
        if not overrides
        else _encode_round_metadata(_plan_reviewer_metadata(**overrides))
    )
    stored["provenance_baseline"] = value
    with caplog.at_level("WARNING"):
        decoded = _decode_round_metadata(encode_mapping(stored))
    assert decoded.provenance_baseline_form is None
    assert decoded.provenance_baseline_chars is None
    # A malformed baseline field never degrades step-back, stall or sweep history.
    assert decoded.step_back_status == "absent"
    assert decoded.plan_sweep is (not overrides)
    assert any("provenance_baseline" in message for message in caplog.messages)


def _m1112_first_primary_payload(runner):
    for comment in runner.issue_comments:
        match = re.search(r"<!-- AGENT_LOOP_META: (?P<payload>\S+) -->", comment["body"])
        if match is None:
            continue
        payload = decode_mapping(match.group("payload"))
        if payload.get("phase") == "scheduler-prelaunch":
            return payload
    raise AssertionError("no scheduler checkpoint")


def test_issue_digest_and_reset_fields_are_strictly_validated(tmp_path):
    """`malformed-fields` and `all-reviewers-unaffected` at the codec."""
    from coding_review_agent_loop.round_state import (
        _decode_round_metadata_mapping,
        _encode_round_metadata,
    )

    runner, _config, _error = _m1103_stalled_runner(tmp_path, threshold=2)
    base = _m1112_first_primary_payload(runner)
    assert _decode_round_metadata_mapping(base).scheduler_metadata_status == "valid"
    good_digest = base["scheduler_issue_digest"]
    assert re.fullmatch(r"[0-9a-f]{16}", good_digest)

    def status(**changes):
        payload = {**base, **changes}
        return _decode_round_metadata_mapping(payload).scheduler_metadata_status

    assert status(scheduler_stall_reset=True) == "valid"
    assert status(scheduler_issue_digest="XYZ") == "invalid"
    assert status(scheduler_issue_digest=None) == "invalid"
    assert status(scheduler_stall_reset=False) == "invalid"
    assert status(scheduler_stall_reset="yes") == "invalid"
    assert status(scheduler_stall_reset=True, scheduler_phase="full-board") == "invalid"
    # Plan board identity fields are flow-scoped (#1377): a PR-flow record
    # carrying them is malformed, so compare the scheduler status without them.
    with pytest.raises(AgentLoopError, match="only valid on plan-flow records"):
        status(flow="pr")
    unstamped = {
        key: value for key, value in base.items() if not key.startswith("plan_board_")
    }
    assert _decode_round_metadata_mapping(
        {**unstamped, "flow": "pr"}
    ).scheduler_metadata_status == "invalid"

    # Unset fields serialize with no new keys (byte-stable legacy shape).
    decoded = _decode_round_metadata_mapping(base)
    unset = replace(decoded, scheduler_issue_digest=None, scheduler_stall_reset=False)
    legacy = decode_mapping(_encode_round_metadata(unset))
    assert "scheduler_issue_digest" not in legacy
    assert "scheduler_stall_reset" not in legacy
    # A record carrying only the new keys is not classified absent.
    only_new = {"flow": "plan", "role": "summary", "agent": "Orchestrator",
                "round_number": 1, "subject": "s", "scheduler_stall_reset": True}
    assert _decode_round_metadata_mapping(only_new).scheduler_metadata_status == "invalid"


def test_step_back_deferral_marker_is_strictly_validated(tmp_path):
    """`deferral-marker-roundtrip` at the codec (#1275)."""
    from coding_review_agent_loop.round_state import (
        _decode_round_metadata_mapping,
        _encode_round_metadata,
    )

    runner, _config, _error = _m1103_stalled_runner(tmp_path, threshold=2)
    base = _m1112_first_primary_payload(runner)

    def decoded(**changes):
        return _decode_round_metadata_mapping({**base, **changes})

    assert decoded().scheduler_step_back_deferral is False
    marked = decoded(scheduler_step_back_deferral=True)
    assert marked.scheduler_metadata_status == "valid"
    assert marked.scheduler_step_back_deferral is True
    assert (
        decode_mapping(_encode_round_metadata(marked))["scheduler_step_back_deferral"]
        is True
    )
    assert "scheduler_step_back_deferral" not in decode_mapping(
        _encode_round_metadata(decoded())
    )
    for bad in (False, "yes", 1, None):
        assert decoded(scheduler_step_back_deferral=bad).scheduler_metadata_status == "invalid"
    assert (
        decoded(
            scheduler_step_back_deferral=True, scheduler_phase="full-board"
        ).scheduler_metadata_status
        == "invalid"
    )


def test_planning_checkpoint_mode_round_trips_and_tolerates_unknown_values(tmp_path):
    """`metadata-roundtrip` on a valid planning checkpoint."""
    from coding_review_agent_loop.round_state import _encode_round_metadata
    from coding_review_agent_loop.round_transport import encode_mapping

    runner, _config, _error = _m1103_stalled_runner(tmp_path, threshold=2)
    checkpoint = next(r for r in _plan_round_records(runner) if r.phase == "scheduler-prelaunch")
    assert checkpoint.scheduler_metadata_status == "valid"
    assert checkpoint.scheduler_execution_mode == "plan-only"
    encoded = _encode_round_metadata(checkpoint)
    assert _decode_round_metadata(encoded).scheduler_execution_mode == "plan-only"

    payload = decode_mapping(encoded)
    payload["scheduler_execution_mode"] = "not-a-mode"
    unknown = _decode_round_metadata(encode_mapping(payload))
    assert unknown.scheduler_execution_mode is None
    assert unknown.scheduler_metadata_status == "valid"

    payload.pop("scheduler_execution_mode")
    legacy = _decode_round_metadata(encode_mapping(payload))
    assert legacy.scheduler_execution_mode is None
    assert legacy.scheduler_metadata_status == "valid"
    assert "scheduler_execution_mode" not in decode_mapping(_encode_round_metadata(legacy))


# --- #1553: a reconciled PR round is closed --------------------------------------------


def _closed_round_history(*, stray_prior_items=(), stray=True, reopen=False):
    """Round 2 reconciled with two reviewer records, then a stray round-2 checkpoint."""
    from coding_review_agent_loop.protocol import UnresolvedReviewItem

    finding = UnresolvedReviewItem(
        item_id="item-1", reviewer="Gemini", source_round=2, text="Guard the race.",
        status="blocking",
    )

    def summary(round_number, phase, prior_items=(), **extra):
        return PostedRoundMetadata(
            flow="pr", role="summary", agent="Orchestrator", round_number=round_number,
            subject="head", prior_items=tuple(prior_items), phase=phase, **extra,
        )

    history = [
        summary(1, "scheduler-prelaunch"),
        PostedRoundMetadata(
            flow="pr", role="reviewer", agent="Codex", round_number=1, subject="head",
            state="approved", phase="publication",
        ),
        summary(1, "reconciliation"),
        summary(2, "scheduler-prelaunch"),
        PostedRoundMetadata(
            flow="pr", role="reviewer", agent="Codex", round_number=2, subject="head",
            state="approved", phase="publication",
        ),
        PostedRoundMetadata(
            flow="pr", role="reviewer", agent="Gemini", round_number=2, subject="head",
            state="blocking", phase="publication",
        ),
        summary(2, "reconciliation", new_items=(finding,)),
    ]
    if reopen:
        history.append(summary(2, "evidence-conversion"))
    if stray:
        history.append(summary(2, "scheduler-prelaunch", prior_items=stray_prior_items))
    return history, finding


def test_1553_post_reconciliation_checkpoint_indexes():
    """Row stray-checkpoint-recovery (round_state unit): only closed-round checkpoints."""
    from coding_review_agent_loop.round_state import (
        _extract_round_metadata_records,
        drop_inert_post_reconciliation_checkpoints,
        inert_post_reconciliation_checkpoint_indexes,
    )

    history, _finding = _closed_round_history()
    records = _extract_round_metadata_records(_comments(*history), flow="pr")
    # Round 2's own opening checkpoint (index 3) stays; the stray (index 7) is inert.
    assert inert_post_reconciliation_checkpoint_indexes(records) == frozenset({7})
    assert [record.index for record in drop_inert_post_reconciliation_checkpoints(records)] == [
        0, 1, 2, 3, 4, 5, 6,
    ]
    # A conversion repeat reopens the round: its checkpoint is not inert.
    reopened, _finding = _closed_round_history(reopen=True)
    records = _extract_round_metadata_records(_comments(*reopened), flow="pr")
    assert inert_post_reconciliation_checkpoint_indexes(records) == frozenset()


def test_1553_stray_checkpoint_does_not_change_visibility_or_panel_evidence():
    """Row stray-checkpoint-recovery: panel opening and checkpoint index are unchanged."""
    from coding_review_agent_loop.panel_evidence import _derive_pr_panel_evidence
    from coding_review_agent_loop.round_state import (
        _extract_round_metadata_records,
        drop_inert_post_reconciliation_checkpoints,
    )
    from coding_review_agent_loop.round_visibility import latest_round_checkpoint_index

    without, _finding = _closed_round_history(stray=False)
    with_stray, _finding = _closed_round_history()
    base = _extract_round_metadata_records(_comments(*without), flow="pr")
    filtered = drop_inert_post_reconciliation_checkpoints(
        _extract_round_metadata_records(_comments(*with_stray), flow="pr")
    )
    assert latest_round_checkpoint_index(
        filtered, flow="pr", round_number=2, subject="head"
    ) == latest_round_checkpoint_index(base, flow="pr", round_number=2, subject="head") == 3
    kwargs = dict(primary_reviewer="Codex", required_reviewers=("Codex", "Gemini"))
    assert _derive_pr_panel_evidence(filtered, **kwargs) == _derive_pr_panel_evidence(base, **kwargs)


@pytest.mark.parametrize("ledger_differs", [False, True])
def test_1553_resume_anchors_on_the_reconciled_transcript(ledger_differs):
    """Rows activation-latest-ledger-differs / existing-activation-checkpoint-kept (unit)."""
    from coding_review_agent_loop.protocol import UnresolvedReviewItem
    from coding_review_agent_loop.round_state import _resume_pr_round

    ack = UnresolvedReviewItem(
        item_id="human-requirements-ack", reviewer="Human requirements", source_round=2,
        text="Acknowledge the signed requirements.", status="blocking",
    )
    history, finding = _closed_round_history(stray_prior_items=(ack,) if ledger_differs else ())
    resumed = _resume_pr_round(
        _comments(*history), head_sha="head", configured_reviewers=("codex", "gemini"),
    )
    assert resumed is not None
    assert resumed.round_number == 2 and resumed.reconciled
    assert [record.metadata.agent for record in resumed.completed_reviews] == ["Codex", "Gemini"]
    assert resumed.reconciliation_record is not None
    assert resumed.reconciliation_record.index == 6
    assert resumed.reconciliation_record.metadata.new_items == (finding,)
    # Item IDs are never reused after rollover.
    assert resumed.next_unresolved_item_number == 2


def test_1553_unreconciled_round_has_no_reconciliation_record():
    from coding_review_agent_loop.round_state import _resume_pr_round

    history, _finding = _closed_round_history(stray=False)
    resumed = _resume_pr_round(
        _comments(*history[:-1]), head_sha="head", configured_reviewers=("codex", "gemini"),
    )
    assert resumed is not None and not resumed.reconciled
    assert resumed.reconciliation_record is None


# --- #1559: an amendment activation still carries the force-full latch ----------------


_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64


def test_1559_activation_classification_of_inert_records():
    """Row activation-classifier-unit: only inert first carriers are activations."""
    from coding_review_agent_loop.round_state import (
        _extract_round_metadata_records,
        force_full_latch_records,
        post_reconciliation_amendment_activation_indexes,
    )

    def summary(round_number, phase, digest=None):
        return PostedRoundMetadata(
            flow="pr", role="summary", agent="Orchestrator", round_number=round_number,
            subject="head", phase=phase, reviewer_board_amendment_digest=digest,
        )

    history, _finding = _closed_round_history(stray=False)
    history += [
        summary(2, "scheduler-prelaunch", _DIGEST_A),  # 7: first carrier of A
        summary(2, "scheduler-prelaunch"),  # 8: null-digest stray
        summary(2, "scheduler-prelaunch", _DIGEST_A),  # 9: repeats A
        summary(3, "scheduler-prelaunch", _DIGEST_A),  # 10: open round, not inert
        summary(3, "reconciliation", _DIGEST_A),  # 11
        summary(3, "scheduler-prelaunch", _DIGEST_B),  # 12: first carrier of B
    ]
    records = _extract_round_metadata_records(_comments(*history), flow="pr")
    snapshot = list(records)

    assert post_reconciliation_amendment_activation_indexes(records) == frozenset({7, 12})
    assert [record.index for record in force_full_latch_records(records)] == [
        0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12,
    ]
    assert list(records) == snapshot

    # A digest first carried by an earlier non-inert record is not an activation.
    early = list(history[:3]) + [summary(2, "scheduler-prelaunch", _DIGEST_A)] + list(history[4:7])
    early.append(summary(2, "scheduler-prelaunch", _DIGEST_A))
    records = _extract_round_metadata_records(_comments(*early), flow="pr")
    assert post_reconciliation_amendment_activation_indexes(records) == frozenset()
    assert [record.index for record in force_full_latch_records(records)] == list(range(7))


def test_1559_helpers_return_input_unchanged_without_inert_records():
    from coding_review_agent_loop.round_state import (
        _extract_round_metadata_records,
        force_full_latch_records,
        post_reconciliation_amendment_activation_indexes,
    )

    history, _finding = _closed_round_history(stray=False)
    records = _extract_round_metadata_records(_comments(*history), flow="pr")
    assert post_reconciliation_amendment_activation_indexes(records) == frozenset()
    assert force_full_latch_records(records) == tuple(records)


def test_1559_startup_latch_reads_activation_but_not_stray():
    from coding_review_agent_loop.round_state import (
        _extract_round_metadata_records,
        startup_force_full_latch,
    )

    def latched(digest, source):
        return _checkpoint(
            subject="head", scheduler_current_sha="head",
            reviewer_board_amendment_digest=digest,
            scheduler_force_full=True, scheduler_force_full_source=source,
        )

    history, _finding = _closed_round_history(stray=False)
    for source, staged, expected in (
        ("operator", True, (True, False)),
        ("operator", False, (True, False)),
        ("automatic", True, (False, False)),
        ("automatic", False, (False, True)),
    ):
        activation = _extract_round_metadata_records(
            _comments(*history, latched(_DIGEST_A, source)), flow="pr"
        )
        assert startup_force_full_latch(activation, requires_primary=staged) == expected
        stray = _extract_round_metadata_records(
            _comments(*history, latched(None, source)), flow="pr"
        )
        assert startup_force_full_latch(stray, requires_primary=staged) == (False, False)


def test_legacy_contradictory_approval_is_not_carried_or_resumed_and_keeps_evidence(tmp_path):
    from coding_review_agent_loop.protocol import (
        ParsedReview, ApprovedFollowups, ReviewItemDisposition, parse_review,
    )
    from coding_review_agent_loop.comment_rendering import render_public_agent_comment
    from coding_review_agent_loop.round_state import _resume_pr_round
    from coding_review_agent_loop.response_validation import _resumed_pr_reviewer_matches_requirements
    ledger = _evidence_ledger(frozen=False)
    disposition = ReviewItemDisposition(item_id="item-1", reviewer="Codex", disposition="blocking", note="Evidence remains needed.")
    contradictory = ParsedReview(state="approved", summary="Code approved, evidence deferred.",
                                 blocking_items=(), followups=ApprovedFollowups(same_pr=(), future=()), dispositions=(disposition,))
    body = render_public_agent_comment(kind="pr_review", parsed=contradictory, agent="codex",
                                      prior_items=ledger, dispositions=(disposition,), config=make_config(tmp_path))
    with pytest.raises(AgentLoopError, match="fully complete"):
        parse_review(body, reviewer="Codex")
    metadata = PostedRoundMetadata(flow="pr", role="reviewer", agent="Codex", round_number=2,
                                   subject=_FROZEN_HEAD, state="approved", prior_items=ledger, dispositions=(disposition,))
    comments = [SimpleNamespace(body=_attach_round_metadata(body, metadata))]
    assert _latest_pr_approved_reviews_for_head(comments, head_sha=_FROZEN_HEAD, configured_reviewers=("codex",)) == {}
    resumed = _resume_pr_round(comments, head_sha=_FROZEN_HEAD, configured_reviewers=("codex",))
    assert resumed is not None and resumed.completed_reviews == ()
    assert resumed.prior_items == ledger
    assert resumed.prior_items[0].resolution_owners == ledger[0].resolution_owners
    record = PostedRoundRecord(metadata=metadata, body=body, index=0)
    assert not _resumed_pr_reviewer_matches_requirements(record, ())


@pytest.mark.parametrize("qualification", [False, True])
@pytest.mark.parametrize("round_number", [1, 2])
def test_open_prelaunch_checkpoint_preserves_later_round_identity(round_number, qualification):
    history = []
    if round_number > 1:
        history = [
            _reviewer(subject="abc123", new_items=(_item(8),)),
            _summary("reconciliation", "abc123", 1),
        ]
    if qualification:
        history.append(_summary(
            "qualification-checkpoint", "abc123", 1,
            qualification_checkpoint=_legacy_checkpoint(),
        ))
    history.append(_checkpoint(round_number=round_number, prior_items=(_item(),)))
    resumed = _resume(history, "abc123")
    if round_number == 1 and not qualification:
        assert resumed is None
        return
    assert resumed.round_number == round_number
    assert resumed.completed_reviews == ()
    assert resumed.reconciled == (round_number == 1 and qualification)
    assert resumed.coder_output is None
    if round_number > 1:
        assert resumed.prior_items == (_item(),)
        assert resumed.next_unresolved_item_number == 9
