"""Round metadata compatibility for the reviewer-board amendment digest (#943)."""

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
    encode_plan_validation_diagnostic_body,
    recover_plan_validation_diagnostic,
)
from coding_review_agent_loop.round_transport import decode_mapping, encode_mapping

from agent_loop_helpers import make_config


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
