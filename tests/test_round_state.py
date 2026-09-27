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
