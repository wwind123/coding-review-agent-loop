"""Approval-time plan-growth verdict in the approved-plan handoff record (#1074)."""

import base64
import json
from types import SimpleNamespace

import pytest

import coding_review_agent_loop.orchestrator as orchestrator
from agent_loop_helpers import FakeRunner, make_config, structured_v1_plan_state
from coding_review_agent_loop.cli import run_pr_loop
from coding_review_agent_loop.comment_rendering import render_canonical_plan_state
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.issue_pr_handoff import (
    AGENT_ISSUE_PR_HANDOFF_RE,
    find_latest_issue_pr_handoff,
    format_issue_pr_handoff_comment,
)
from coding_review_agent_loop.plan_assembly import (
    make_assembled_plan_sidecar,
    structured_plan_revision_to_payload,
)
from coding_review_agent_loop.plan_growth import (
    PlanGrowthApprovalVerdict,
    PlanGrowthThresholds,
    assess_plan_growth,
    handoff_growth_verdict_violation,
    plan_growth_approval_verdict,
)
from coding_review_agent_loop.protocol import validate_structured_plan_state
from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata

FOOTER = "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
PR_URL = "https://github.com/OWNER/REPO/pull/77"


def _verdict(status="compliant", *, gate="enforce", crossed=()):
    return PlanGrowthApprovalVerdict(
        gate=gate, status=status, crossed_signals=tuple(crossed),
        thresholds=PlanGrowthThresholds(),
    )


def _handoff(plan_hash, verdict=None, *, flow="approved-plan-implementation"):
    return format_issue_pr_handoff_comment(
        issue_number=56, pr_number=77, pr_url=PR_URL, pr_head_sha="abc123",
        flow=flow, plan_hash=plan_hash, plan_growth_verdict=verdict,
    )


def _payload_of(comment):
    encoded = AGENT_ISSUE_PR_HANDOFF_RE.search(comment).group("payload")
    return json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")))


def _recorded(comment):
    return find_latest_issue_pr_handoff(
        [SimpleNamespace(body=comment)], issue_number=56, repo="OWNER/REPO"
    )


# --- verdict type and rules ---------------------------------------------------


def test_verdict_round_trips_and_parses_strictly():
    verdict = _verdict("non-compliant", gate="off", crossed=("scope-items", "revision-count"))
    assert PlanGrowthApprovalVerdict.from_payload(verdict.to_payload(), context="v") == verdict
    good = verdict.to_payload()
    for bad in (
        {**good, "extra": 1},
        {**good, "gate": "warn"},
        {**good, "status": "maybe"},
        {**good, "crossed_signals": ["revision-count", "scope-items"]},
        {**good, "crossed_signals": ["scope-items", "scope-items"]},
        {**good, "crossed_signals": ["unknown"]},
        {**good, "status": "not-applicable"},
        {**good, "thresholds": {**good["thresholds"], "max_chars": 0}},
        {**good, "thresholds": {**good["thresholds"], "max_chars": True}},
        {key: value for key, value in good.items() if key != "thresholds"},
    ):
        with pytest.raises(AgentLoopError):
            PlanGrowthApprovalVerdict.from_payload(bad, context="v")


def test_approval_verdict_is_computed_whatever_the_gate_mode(tmp_path):
    payload = json.loads(structured_v1_plan_state().split("\n", 1)[0])
    low = PlanGrowthThresholds(max_scope_items=1)
    assessment = assess_plan_growth(
        payload, rendered_chars=100, revision_count=1, thresholds=low
    )
    for gate in ("enforce", "off"):
        config = make_config(tmp_path, plan_growth_gate=gate, plan_growth_max_scope_items=1)
        verdict = plan_growth_approval_verdict(config, payload, assessment)
        assert (verdict.gate, verdict.status, verdict.crossed_signals) == (
            gate, "non-compliant", ("scope-items",)
        )
        assert verdict.thresholds == low
    compliant = plan_growth_approval_verdict(
        make_config(tmp_path),
        payload,
        assess_plan_growth(
            payload, rendered_chars=100, revision_count=1, thresholds=PlanGrowthThresholds()
        ),
    )
    assert (compliant.status, compliant.crossed_signals) == ("compliant", ())
    assert plan_growth_approval_verdict(make_config(tmp_path), None, None).status == "not-applicable"


def test_resume_rule_uses_the_recorded_verdict_not_todays_thresholds(tmp_path):
    enforced_low = make_config(tmp_path, plan_growth_max_scope_items=1)
    # Legacy: a handoff without a verdict predates it and passes.
    assert handoff_growth_verdict_violation(None, enforced_low) is None
    # Compliant when approved stays compliant after thresholds are lowered.
    assert handoff_growth_verdict_violation(_verdict("compliant"), enforced_low) is None
    assert handoff_growth_verdict_violation(_verdict("not-applicable"), enforced_low) is None
    # Non-compliant at approval refuses while the gate is enforced now.
    for gate in ("enforce", "off"):
        noncompliant = _verdict("non-compliant", gate=gate, crossed=("scope-items",))
        assert "failed the plan-growth gate" in handoff_growth_verdict_violation(
            noncompliant, make_config(tmp_path)
        )
        assert handoff_growth_verdict_violation(
            noncompliant, make_config(tmp_path, plan_growth_gate="off")
        ) is None


# --- handoff record codec -----------------------------------------------------


def test_handoff_record_carries_the_verdict_and_legacy_records_stay_unchanged():
    verdict = _verdict("non-compliant", gate="off", crossed=("scope-items",))
    comment = _handoff("0123456789abcdef", verdict)
    assert "Plan-growth gate at approval: gate off, non-compliant (crossed: scope-items)." in comment
    assert _payload_of(comment)["plan_growth_verdict"] == verdict.to_payload()
    assert _recorded(comment).plan_growth_verdict == verdict

    legacy = _handoff("0123456789abcdef")
    assert "plan_growth_verdict" not in _payload_of(legacy)
    assert "Plan-growth gate at approval" not in legacy
    assert _recorded(legacy).plan_growth_verdict is None


def test_verdict_is_rejected_on_a_direct_issue_handoff():
    with pytest.raises(AgentLoopError, match="only valid for approved-plan-implementation"):
        _handoff(None, _verdict(), flow="issue-implementation")


# --- measuring the approved candidate -----------------------------------------


def _plan_record(canonical_plan, payload, *, round_number=1):
    sidecar = make_assembled_plan_sidecar(
        payload, round_number=round_number, response_form="fresh-plan-state",
        rendered_plan=canonical_plan,
    )
    return SimpleNamespace(body=_attach_round_metadata(
        canonical_plan + FOOTER,
        PostedRoundMetadata(
            flow="plan", role="coder", agent="Claude", round_number=round_number,
            subject="s", canonical_plan=canonical_plan, response_form="fresh-plan-state",
            aggregate_plan_identity=sidecar.aggregate_identity,
            assembled_plan_sidecar=sidecar.to_payload(),
        ),
    ))


def test_verdict_for_hash_measures_the_authenticated_candidate(tmp_path):
    parsed = validate_structured_plan_state(structured_v1_plan_state())
    payload = structured_plan_revision_to_payload(parsed)
    canonical = render_canonical_plan_state(parsed)
    plan_hash = orchestrator.approved_plan_hash(canonical)
    comments = [_plan_record(canonical, payload)]

    compliant = orchestrator._plan_growth_verdict_for_hash(
        make_config(tmp_path), plan_hash=plan_hash, comment_sources=(comments,)
    )
    assert (compliant.gate, compliant.status) == ("enforce", "compliant")
    off_low = orchestrator._plan_growth_verdict_for_hash(
        make_config(tmp_path, plan_growth_gate="off", plan_growth_max_scope_items=1),
        plan_hash=plan_hash,
        # A missing first source falls through to the one holding the plan.
        comment_sources=(None, [], comments),
    )
    assert (off_low.gate, off_low.status, off_low.crossed_signals) == (
        "off", "non-compliant", ("scope-items",)
    )
    missing = orchestrator._plan_growth_verdict_for_hash(
        make_config(tmp_path), plan_hash="0" * 16, comment_sources=(comments,)
    )
    assert missing.status == "not-applicable"


# --- handoff-backed resume ------------------------------------------------------


def _plan_comments(plan):
    subject = orchestrator._plan_subject(plan)
    return [
        {
            "author": {"login": "coding-review-agent-loop"},
            "createdAt": "2026-05-01T00:00:00Z",
            "body": _attach_round_metadata(
                plan,
                PostedRoundMetadata(
                    flow="plan", role="coder", agent="Claude", round_number=1,
                    subject=subject, canonical_plan=plan,
                    raw_structured_coder_response=plan,
                ),
            ),
        },
        {
            "author": {"login": "coding-review-agent-loop"},
            "createdAt": "2026-05-01T00:01:00Z",
            "body": _attach_round_metadata(
                "Approved.",
                PostedRoundMetadata(
                    flow="plan", role="reviewer", agent="Codex",
                    round_number=1, subject=subject, state="approved",
                ),
            ),
        },
    ]


PLAN = "Approved plan.\n\n### Plan steps\n1. Preserve the trust boundary."
NONCOMPLIANT = _verdict("non-compliant", gate="off", crossed=("revision-count",))


def _runner(verdict, **pr_payload):
    handoff = {
        "author": {"login": "coding-review-agent-loop"},
        "createdAt": "2026-05-01T00:02:00Z",
        "body": _handoff(orchestrator.approved_plan_hash(PLAN), verdict),
    }
    return FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        issue_comments=[*_plan_comments(PLAN), handoff],
        issue_payload={"number": 56, "title": "Issue", "body": "Original issue."},
        pr_payload={"number": 77, "url": PR_URL, "body": "Fixes #56", **pr_payload},
    )


@pytest.mark.parametrize("verdict", [None, _verdict("compliant")])
def test_ordinary_pr_recovery_accepts_legacy_and_compliant_handoffs(tmp_path, verdict):
    runner = _runner(verdict)
    # Today's thresholds would fail almost any plan; only the record counts.
    config = make_config(tmp_path, plan_growth_max_scope_items=1, plan_growth_max_chars=1)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


def test_ordinary_pr_recovery_refuses_a_noncompliant_handoff(tmp_path):
    runner = _runner(NONCOMPLIANT)
    with pytest.raises(AgentLoopError, match="PR #77 recovery: .*failed the plan-growth gate"):
        run_pr_loop(runner, pr_number=77, config=make_config(tmp_path))
    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)
    # The operator opt-out still accepts it.
    runner = _runner(NONCOMPLIANT)
    assert run_pr_loop(
        runner, pr_number=77, config=make_config(tmp_path, plan_growth_gate="off")
    ) == 0


class _Reached(Exception):
    pass


def _managed_config(tmp_path, **overrides):
    return make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        reviewer=("codex",), **overrides,
    )


_MANAGED_PR = {"headRefName": "agent-loop/managed-56", "headRefOid": "abc123", "baseRefName": "main"}


def _no_retired_plans(monkeypatch):
    monkeypatch.setattr(
        orchestrator, "_managed_ci_retired_plan_hashes",
        lambda *_a, **kwargs: (frozenset(), kwargs.get("parent_issue_context")),
    )


@pytest.mark.parametrize("verdict,refused", [(None, False), (NONCOMPLIANT, True)])
def test_managed_fresh_authorization_applies_the_recorded_verdict(
    tmp_path, monkeypatch, verdict, refused
):
    runner = _runner(verdict, **_MANAGED_PR)
    _no_retired_plans(monkeypatch)
    monkeypatch.setattr(
        orchestrator, "authorize_fresh_issue_created_resume",
        lambda *_a, **_k: (_ for _ in ()).throw(_Reached()),
    )
    config = _managed_config(
        tmp_path, managed_ci_fresh_authorization=True, managed_ci_issue_number=56
    )
    expected = (
        pytest.raises(AgentLoopError, match="Managed-CI fresh authorization: .*failed the plan-growth")
        if refused else pytest.raises(_Reached)
    )
    with expected:
        run_pr_loop(runner, pr_number=77, config=config)


@pytest.mark.parametrize("verdict,refused", [(None, False), (NONCOMPLIANT, True)])
def test_managed_ordinary_resume_applies_the_recorded_verdict(
    tmp_path, monkeypatch, verdict, refused
):
    runner = _runner(verdict, **_MANAGED_PR)
    handoff = orchestrator.AuthenticatedIssueCreatedHandoff(
        pr_number=77, issue_number=56, repository="OWNER/REPO", base_ref="main",
        head_sha="abc123", branch="agent-loop/managed-56",
        trusted_actor_login="agent-loop", trusted_actor_id=1,
        protection_mode="voluntary", override_nonce="opening-nonce",
    )
    monkeypatch.setattr(orchestrator, "recover_issue_created_handoff", lambda *_a, **_k: handoff)
    _no_retired_plans(monkeypatch)
    monkeypatch.setattr(
        orchestrator, "revalidate_issue_created_handoff",
        lambda *_a, **_k: (_ for _ in ()).throw(_Reached()),
    )
    expected = (
        pytest.raises(AgentLoopError, match="Managed-CI ordinary resume: .*failed the plan-growth")
        if refused else pytest.raises(_Reached)
    )
    with expected:
        run_pr_loop(runner, pr_number=77, config=_managed_config(tmp_path))
