"""Degraded and synthesized coder records after a post-push rejection (#2086)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import coding_review_agent_loop.coder_followup_degradation as degradation
import coding_review_agent_loop.unresolved_items as unresolved_items_module
from coding_review_agent_loop.coder_followup_degradation import (
    DEGRADED_DISPOSITION_EVIDENCE,
    CoderFollowupRecovery,
    degrade_rejected_followup,
)
from coding_review_agent_loop.errors import AgentLoopError, RejectedCandidate
from coding_review_agent_loop.github import HumanReviewRequirement
from coding_review_agent_loop.prompt_support import (
    CoderHumanRequirementsPromptContext,
    render_coder_human_requirements_prompt_context,
)
from coding_review_agent_loop.protocol import (
    HUMAN_REQUIREMENT_DISPOSITION_VALUES,
    UnresolvedReviewItem,
    validate_human_requirement_dispositions,
    validate_structured_human_requirements_acknowledgement,
)
from coding_review_agent_loop.response_validation import (
    _current_test_turn_observations,
    _validate_followup_workdir_report,
)
from coding_review_agent_loop.round_metadata import PostedRoundMetadata, _attach_round_metadata
from coding_review_agent_loop.round_state import _resume_pr_round
from coding_review_agent_loop.unresolved_items import (
    HUMAN_REQUIREMENTS_ACK_ITEM_ID,
    _reconcile_human_requirements_ack_item,
    _validate_coder_followup_response,
)

from agent_loop_helpers import make_config, structured_coder_followup
from pr_loop_test_helpers import _followup_observation

HEAD = "b" * 40
OLD = "a" * 40
_REQUIREMENT = HumanReviewRequirement(
    source_type="PR comment",
    author="reviewer",
    created_at="2026-05-18T10:00:00Z",
    url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
    body="Please use the absolute URL.",
)
REQUIREMENTS = (_REQUIREMENT,)


def _item(number):
    return UnresolvedReviewItem(
        item_id=f"item-{number}", reviewer="Codex", source_round=1,
        text=f"Fix defect {number}.", status="blocking",
    )


def _ack_item():
    return UnresolvedReviewItem(
        item_id=HUMAN_REQUIREMENTS_ACK_ITEM_ID, reviewer="Orchestrator", source_round=1,
        text="Acknowledge the signed requirements.", status="blocking",
    )


ITEMS = (_item(1), _item(2))


class _Runner:
    """Its latest test turn is a repair turn; degradation must never read it."""

    def __init__(self, repair_catalog=()):
        self.latest_test_turn_id = "repair-turn"
        self._repair_catalog = tuple(repair_catalog)

    def current_test_turn_observations(self):
        return self._repair_catalog


def _factory(runner, tmp_path, *, requirements=(), items=ITEMS, seen=None):
    config = make_config(tmp_path)

    def factory(mode, observations_mode, degraded_unacknowledged=False):
        def validate(text):
            catalog = _current_test_turn_observations(runner)
            if seen is not None:
                seen.append(tuple(catalog))
            return _validate_followup_workdir_report(
                _validate_coder_followup_response(
                    text,
                    unresolved_items=items,
                    human_requirements=requirements,
                    required_architecture_impact_contract=1,
                    authoritative_test_observations=catalog,
                    execution_catalog=catalog,
                    architecture_status_mode=mode,
                    test_observations_mode=observations_mode,
                    degraded_unacknowledged=degraded_unacknowledged,
                ),
                config=config,
            )

        return validate

    return factory


def _response(payload_changes=None, *, drop=(), **kwargs):
    kwargs.setdefault("reviewer", "OpenAI Codex")
    text = structured_coder_followup(**kwargs)
    payload = json.loads(text.split("\n", 1)[0])
    payload.update(payload_changes or {})
    for key in drop:
        payload.pop(key, None)
    return json.dumps(payload) + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"


def _degrade(text, runner, tmp_path, *, catalog=(), requirements=(), seen=None, **extra):
    context = render_coder_human_requirements_prompt_context(requirements)
    return degrade_rejected_followup(
        RejectedCandidate(text=text, test_turn_id="coder-turn", test_turn_observations=tuple(catalog)),
        validator_factory=_factory(runner, tmp_path, requirements=requirements, seen=seen),
        runner=runner,
        allowed_item_ids=[item.item_id for item in ITEMS],
        surfaced_requirement_ids=context.surfaced_requirement_ids,
        requires_direct_discussion_ack=context.requires_direct_discussion_ack,
        **extra,
    )


def _payload(text):
    return json.loads(text.split("\n", 1)[0])


def _resumed_coder_record(text, **metadata_fields):
    metadata = PostedRoundMetadata(
        flow="pr", role="coder", agent="OpenAI Codex", round_number=2, subject=HEAD,
        prior_items=(*ITEMS, _ack_item()), raw_structured_coder_response=text, **metadata_fields,
    )
    body = _attach_round_metadata("## Coder follow-up\n\nDegraded.", metadata)
    resumed = _resume_pr_round(
        [SimpleNamespace(body=body)], head_sha=HEAD, configured_reviewers=("claude",)
    )
    assert resumed is not None and resumed.unrecorded_head_advance is False
    return resumed


def _ack_open(items):
    return any(item.item_id == HUMAN_REQUIREMENTS_ACK_ITEM_ID for item in items)


def test_missing_dispositions_normalize_to_blocked_and_keep_the_ack_open_same_run_and_resume(tmp_path):
    """Row degrade-missing-dispositions."""
    requirement_id = _REQUIREMENT.requirement_id
    text = _response(
        summary="Fixed item 1.", addressed_items=["item-1"], remaining_items=["item-2"],
        human_requirement_ids=[requirement_id], drop=("human_requirement_dispositions",),
    )
    runner = _Runner()
    degraded = _degrade(text, runner, tmp_path, requirements=REQUIREMENTS)

    assert degraded is not None  # never synthesis: the unrelated claim survives
    assert degraded.marker_value.addressed_items == ("item-1",)
    assert degraded.human_requirements_degraded is True
    (disposition,) = degraded.marker_value.human_requirement_dispositions
    assert disposition.requirement_id == requirement_id
    assert disposition.disposition == "blocked"
    assert disposition.disposition in HUMAN_REQUIREMENT_DISPOSITION_VALUES
    assert disposition.evidence == DEGRADED_DISPOSITION_EVIDENCE
    assert degraded.marker_value.human_requirements.addressed_ids == ()
    assert degraded.marker_value.human_requirements.checked_discussion_directly is False
    # The unchanged strict validators accept the normalized payload.
    validate_human_requirement_dispositions(
        degraded.marker_value.human_requirement_dispositions,
        surfaced_requirement_ids=(requirement_id,),
    )
    validate_structured_human_requirements_acknowledgement(
        degraded.marker_value.human_requirements.addressed_ids,
        dispositions=degraded.marker_value.human_requirement_dispositions,
        checked_discussion_directly=False,
        surfaced_requirement_ids=(requirement_id,),
        requires_direct_discussion_ack=False,
    )
    assert any("blocked and not addressed" in note for note in degraded.notes)

    # Same run: the real reconciliation keeps the acknowledgement item open.
    ledger = [*ITEMS, _ack_item()]
    same_run = _reconcile_human_requirements_ack_item(
        ledger, coder_output=degraded.text, human_requirements=REQUIREMENTS, source_round=2,
        human_requirements_degraded=degraded.human_requirements_degraded,
    )
    assert _ack_open(same_run)
    # Without the orchestrator flag, the valid payload would clear it.
    assert not _ack_open(_reconcile_human_requirements_ack_item(
        ledger, coder_output=degraded.text, human_requirements=REQUIREMENTS, source_round=2,
    ))

    # Resume: the flag is read from the coder record's metadata.
    resumed = _resumed_coder_record(
        degraded.text, coder_record_source="degraded",
        coder_record_degradations=degraded.notes, human_requirements_degraded=True,
    )
    assert resumed.coder_output == degraded.text
    on_resume = _reconcile_human_requirements_ack_item(
        resumed.prior_items, coder_output=resumed.coder_output, human_requirements=REQUIREMENTS,
        source_round=3, human_requirements_degraded=resumed.coder_metadata.human_requirements_degraded,
    )
    assert _ack_open(on_resume)
    assert {"item-1", "item-2"} <= {item.item_id for item in on_resume}


def _direct_ack_context(monkeypatch):
    context = CoderHumanRequirementsPromptContext(
        block="", surfaced_requirement_ids=(), requires_direct_discussion_ack=True,
        unverified_authors=False,
    )
    monkeypatch.setattr(
        unresolved_items_module, "render_coder_human_requirements_prompt_context",
        lambda *_args, **_kwargs: context,
    )
    return context


def test_missing_direct_acknowledgement_is_recorded_not_acknowledged(tmp_path, monkeypatch):
    """Row degrade-direct-ack-missing."""
    _direct_ack_context(monkeypatch)
    text = _response(
        summary="Fixed item 1.", addressed_items=["item-1"], remaining_items=["item-2"],
        checked_discussion_directly=False,
    )
    runner = _Runner()
    validate = _factory(runner, tmp_path, requirements=REQUIREMENTS)
    # Only the degraded flag lets the unacknowledged payload validate.
    with pytest.raises(AgentLoopError, match="must acknowledge"):
        validate("degradable", "degrade")(text)
    degraded = degrade_rejected_followup(
        RejectedCandidate(text=text, test_turn_id="coder-turn"),
        validator_factory=validate, runner=runner,
        allowed_item_ids=["item-1", "item-2"],
        surfaced_requirement_ids=(), requires_direct_discussion_ack=True,
    )

    assert degraded is not None and degraded.human_requirements_degraded is True
    assert degraded.marker_value.human_requirements.checked_discussion_directly is False  # never fabricated
    assert _payload(degraded.text)["human_requirements"]["checked_discussion_directly"] is False
    assert any("recorded as not acknowledged" in note for note in degraded.notes)
    resumed = _resumed_coder_record(
        degraded.text, coder_record_source="degraded",
        coder_record_degradations=degraded.notes, human_requirements_degraded=True,
    )
    on_resume = _reconcile_human_requirements_ack_item(
        resumed.prior_items, coder_output=resumed.coder_output, human_requirements=REQUIREMENTS,
        source_round=3, human_requirements_degraded=resumed.coder_metadata.human_requirements_degraded,
    )
    assert _ack_open(on_resume)


def test_degraded_acknowledgement_flag_never_applies_to_ordinary_records(tmp_path, monkeypatch):
    """Row degrade-direct-ack-missing: an ordinary record still needs the acknowledgement."""
    _direct_ack_context(monkeypatch)
    text = _response(summary="Fixed.", addressed_items=["item-1", "item-2"], checked_discussion_directly=False)
    with pytest.raises(AgentLoopError, match="must acknowledge"):
        _factory(_Runner(), tmp_path, requirements=REQUIREMENTS)("strict", "repair-first")(text)


def test_ordinary_acknowledgement_still_clears_and_a_synthesized_record_keeps_it_open(tmp_path):
    """Row ack-reconcile-ordinary-unchanged."""
    requirement_id = _REQUIREMENT.requirement_id
    valid = _response(
        summary="Fixed.", addressed_items=["item-1", "item-2"], human_requirement_ids=[requirement_id],
    )
    ledger = [*ITEMS, _ack_item()]
    # Same run: no flag, unchanged clearing.
    assert not _ack_open(_reconcile_human_requirements_ack_item(
        ledger, coder_output=valid, human_requirements=REQUIREMENTS, source_round=2,
    ))
    # Resume: an ordinary record carries no flag; coder text cannot set it.
    resumed = _resumed_coder_record(valid)
    assert resumed.coder_metadata.human_requirements_degraded is False
    assert not _ack_open(_reconcile_human_requirements_ack_item(
        resumed.prior_items, coder_output=resumed.coder_output, human_requirements=REQUIREMENTS,
        source_round=3, human_requirements_degraded=resumed.coder_metadata.human_requirements_degraded,
    ))
    # A synthesized record with surfaced requirements carries the flag.
    recovery = CoderFollowupRecovery(
        _Runner(), config=make_config(tmp_path), pr_number=77, pre_turn_head=OLD, ledger=ledger,
        coder_name="OpenAI Codex", record_round=2, validator_factory=_factory(_Runner(), tmp_path),
        human_requirements_context=render_coder_human_requirements_prompt_context(REQUIREMENTS),
        workdir=tmp_path, observe_head=lambda: HEAD,
    )
    response = recovery._recover(AgentLoopError("no structured object"))
    assert recovery.source == "synthesized" and recovery.human_requirements_degraded is True
    assert response.marker_value == "blocking"  # no structured coder claims
    assert _ack_open(_reconcile_human_requirements_ack_item(
        ledger, coder_output=response.text, human_requirements=REQUIREMENTS, source_round=2,
        human_requirements_degraded=recovery.human_requirements_degraded,
    ))


def _observation(receipt, execution_ref):
    """A passing, parent-observed broker receipt."""
    return _followup_observation(
        execution_ref=execution_ref, receipt_id=receipt, head=HEAD, timestamp="2026-11-22T00:00:00Z",
    )


def test_degradation_uses_the_rejected_candidates_own_acquisition_catalog(tmp_path):
    """Row degrade-frozen-acquisition-catalog: repair-only receipts never become evidence."""
    coder = _observation("rcpt-coder", "exec-coder")
    repair = _observation("rcpt-repair", "exec-repair")
    runner = _Runner(repair_catalog=(repair,))
    text = _response(
        {"test_observations": [
            {"command": "python3 -m pytest tests/test_orchestrator_pr.py -q", "receipt_id": "rcpt-coder", "claim": "current-result"},
            {"command": "python3 -m pytest tests/test_orchestrator_pr.py -q", "receipt_id": "rcpt-repair", "claim": "current-result"},
        ]},
        summary="Fixed.", addressed_items=["item-1", "item-2"],
    )
    seen = []
    degraded = _degrade(text, runner, tmp_path, catalog=(coder,), seen=seen)

    assert degraded is not None
    assert seen and all(catalog == (coder,) for catalog in seen)  # never runner.latest_test_turn_id
    receipts = [citation.receipt_id for citation in degraded.marker_value.test_observations]
    assert receipts == ["rcpt-coder"]
    assert "rcpt-repair" not in degraded.text
    assert any("`rcpt-repair`" in note for note in degraded.notes)


def test_malformed_architecture_impact_is_dropped_and_not_persisted(tmp_path):
    """Row degrade-architecture-impact."""
    from coding_review_agent_loop.architecture_contract import _architecture_metadata_fields

    text = _response(
        {"architecture_impact": {"status": "sideways", "rationale": 7}},
        summary="Fixed.", addressed_items=["item-1", "item-2"],
    )
    degraded = _degrade(text, _Runner(), tmp_path)

    assert degraded is not None
    assert "architecture_impact" not in _payload(degraded.text)
    assert degraded.marker_value.architecture_impact is None
    assert any("architecture_impact" in note for note in degraded.notes)
    fields = _architecture_metadata_fields(make_config(tmp_path), result=degraded.marker_value)
    assert fields["architecture_impact"] is None


def test_outside_checkout_tests_and_inadmissible_selectors_are_dropped(tmp_path):
    """Row degrade-outside-checkout: nothing from the offending claims becomes evidence."""
    coder = _observation("rcpt-coder", "exec-coder")
    text = _response(
        {"risk_test_matrix_claims": [{
            "row_id": "row-a", "execution_refs": ["exec-unknown", "exec-coder"],
            "test_identifiers": ["tests/test_x.py::test_a"], "test_locations": ["tests/test_x.py"],
            "workflow_path_claim": "pr loop", "outcome_assertions": ["passes"],
            "forbidden_effect_assertions": ["no raise"], "caveats": [], "test_level": "unit",
        }]},
        summary="Fixed.", addressed_items=["item-1", "item-2"],
        tests_run=["pytest tests/test_foo.py", "pytest tests/test_foo.py https://live.example"],
    )
    from coding_review_agent_loop.config import active_workdir

    workdir = active_workdir(make_config(tmp_path))
    degraded = _degrade(
        text, _Runner(), tmp_path, catalog=(coder,),
        tests_run_admissible=lambda entry: degradation.tests_run_entry_in_checkout(entry, workdir),
    )

    assert degraded is not None
    assert degraded.marker_value.tests_run == ("pytest tests/test_foo.py",)
    assert "https://live.example" not in degraded.text
    (claim,) = _payload(degraded.text)["risk_test_matrix_claims"]
    assert claim["execution_refs"] == ["exec-coder"]
    joined = " ".join(degraded.notes)
    assert "tests_run" in joined and "`exec-unknown`" in joined


@pytest.mark.parametrize("case", ["duplicate-ids", "dispute-without-evidence", "unparseable", "wrong-kind"])
def test_a_rejection_naming_no_droppable_field_requests_synthesis(tmp_path, case):
    """Row degrade-unrecognised-rejection: never a partially validated response."""
    text = {
        "duplicate-ids": _response(summary="Fixed.", addressed_items=["item-1"], remaining_items=["item-1", "item-2"]),
        "dispute-without-evidence": _response(
            {"dispute_evidence": {}}, summary="Disputed.", addressed_items=["item-2"], disputed_items=["item-1"],
        ),
        "unparseable": "I pushed a fix but forgot the format.",
        "wrong-kind": json.dumps({"schema_version": 1, "kind": "pr_review"}) + "\n-- OpenAI Codex",
    }[case]
    assert _degrade(text, _Runner(), tmp_path) is None


def test_exceeding_the_drop_bound_requests_synthesis(tmp_path, monkeypatch):
    """Row degrade-unrecognised-rejection: the drop loop is bounded."""
    monkeypatch.setattr(degradation, "MAX_DEGRADATION_STEPS", 1)
    text = _response(
        summary="Fixed.", addressed_items=["item-1", "item-2"],
        tests_run=["pytest tests/test_foo.py https://live.example"],
    )
    assert _degrade(text, _Runner(), tmp_path) is None


def test_synthesized_text_bounds_the_listing_and_the_diffstat():
    listing = degradation.synthesize_coder_record_text(
        old_head=OLD, new_head=HEAD, item_ids=[f"item-{n}" for n in range(1, 2001)],
        diffstat=degradation.bound_diffstat_text("\n".join(f" f{n}.py | 1 +" for n in range(100)) + "\n 100 files changed"),
    )
    assert "more item(s) omitted from this listing; all remain open" in listing
    assert "more line(s) omitted" in listing and "100 files changed" in listing
    assert listing.startswith(degradation.SYNTHESIZED_RECORD_HEADING)


def test_unchanged_or_unobservable_head_keeps_todays_rejection(tmp_path):
    """Row unchanged-head-rejection at the seam helper."""
    for observe in (lambda: OLD, lambda: (_ for _ in ()).throw(AgentLoopError("gh down"))):
        recovery = CoderFollowupRecovery(
            _Runner(), config=make_config(tmp_path), pr_number=77, pre_turn_head=OLD, ledger=ITEMS,
            coder_name="OpenAI Codex", record_round=2, validator_factory=_factory(_Runner(), tmp_path),
            human_requirements_context=None, workdir=tmp_path, observe_head=observe,
        )
        error = AgentLoopError("rejected")
        with pytest.raises(AgentLoopError, match="rejected"):
            recovery.run(lambda: (_ for _ in ()).throw(error))
        assert recovery.source is None and recovery.metadata_fields() == {}


# --- record_failed_push guard and the untransportable notice (#2153) --------

import hashlib  # noqa: E402

import coding_review_agent_loop.round_transport as round_transport  # noqa: E402
from coding_review_agent_loop.errors import QuotaResetExceededError  # noqa: E402
from coding_review_agent_loop.pr_head_sync import PrHeadObservationError  # noqa: E402
from coding_review_agent_loop.protocol_markers import scan_reserved_markers  # noqa: E402
from coding_review_agent_loop.round_state import _extract_round_metadata_records  # noqa: E402


def _recovery_2153(tmp_path, *, posts, observe=lambda: HEAD, ledger=ITEMS, fail_post=False):
    def post_body(body):
        posts.append(body)
        if fail_post:
            raise AgentLoopError("GitHub write failed")

    return CoderFollowupRecovery(
        _Runner(), config=make_config(tmp_path), pr_number=77, pre_turn_head=OLD, ledger=ledger,
        coder_name="OpenAI Codex", record_round=2, validator_factory=_factory(_Runner(), tmp_path),
        human_requirements_context=None, workdir=tmp_path, observe_head=observe,
        metadata_factory=lambda **fields: PostedRoundMetadata(**fields), post_body=post_body,
    )


def _posted_records(posts):
    return [
        record.metadata
        for record in _extract_round_metadata_records([SimpleNamespace(body=body) for body in posts], flow="pr")
    ]


def test_synthesized_recovery_then_later_failure_posts_one_record(tmp_path):
    """Row synthesized-then-later-failure."""
    posts: list[str] = []
    recovery = _recovery_2153(tmp_path, posts=posts)
    response = recovery.run(lambda: (_ for _ in ()).throw(AgentLoopError("unparseable")))
    assert recovery.source == "synthesized" and posts == []
    later = AgentLoopError("derived evidence rejected")
    recovery.record_failed_push(later)
    recovery.record_failed_push(later)  # a second failure path never posts again
    (record,) = _posted_records(posts)
    assert record.subject == HEAD and record.coder_record_source == "synthesized"
    assert tuple(record.prior_items) == ITEMS
    notes = " ".join(record.coder_record_degradations)
    assert "could not be parsed or degraded" in notes and "AgentLoopError" in notes
    assert "coder_followup" not in (record.raw_structured_coder_response or "")
    assert response.marker_value == "blocking"


def test_degraded_recovery_then_later_failure_keeps_the_degraded_notes(tmp_path):
    """Row degraded-then-later-failure at the seam helper."""
    posts: list[str] = []
    recovery = _recovery_2153(tmp_path, posts=posts)
    recovery.source, recovery.notes = "degraded", ("Dropped unknown item ID(s) `item-99`; they are not addressed.",)
    recovery.record_failed_push(AgentLoopError("derived evidence rejected"))
    (record,) = _posted_records(posts)
    assert record.coder_record_source == "synthesized"
    assert record.coder_record_degradations[0].startswith("Dropped unknown item ID(s)")
    assert "stopped (AgentLoopError)" in record.coder_record_degradations[-1]


def test_failed_best_effort_attempt_is_not_retried(tmp_path):
    """Row attempted-not-retried."""
    posts: list[str] = []
    recovery = _recovery_2153(tmp_path, posts=posts, fail_post=True)
    stop = QuotaResetExceededError("quota reset is too far away")
    with pytest.raises(QuotaResetExceededError) as raised:
        recovery.run(lambda: (_ for _ in ()).throw(stop))
    assert raised.value is stop
    assert len(posts) == 1 and recovery.record_attempted and not recovery.record_posted
    recovery.record_failed_push(stop)  # pr_loop's except block
    assert len(posts) == 1


@pytest.mark.parametrize("case", ["unchanged-head", "head-observation", "checkout-verification"])
def test_fail_closed_or_unchanged_head_posts_nothing(tmp_path, case):
    """Row fail-closed-and-unchanged."""
    posts: list[str] = []
    error = {
        "unchanged-head": AgentLoopError("later"),
        "head-observation": PrHeadObservationError("PR-head verification failed"),
        "checkout-verification": degradation.CheckoutVerificationError("checkout changed"),
    }[case]
    observe = (lambda: OLD) if case == "unchanged-head" else (lambda: HEAD)
    recovery = _recovery_2153(tmp_path, posts=posts, observe=observe)
    recovery.record_failed_push(error)
    if case == "unchanged-head":
        recovery.post_untransportable_notice(error)
    assert posts == []
    unchanged = _recovery_2153(tmp_path, posts=posts, observe=lambda: OLD)
    unchanged.post_untransportable_notice(AgentLoopError("untransportable"))
    assert posts == []


def _incompressible(chars, seed=""):
    return "".join(
        hashlib.sha256(f"{seed}:{n}".encode()).hexdigest() for n in range(chars // 64 + 1)
    )[:chars]


def _oversized_ledger_2153():
    return tuple(
        UnresolvedReviewItem(
            item_id=f"item-{n}", reviewer="Codex", source_round=1,
            text=_incompressible(2_000, seed=n), status="blocking",
        )
        for n in range(1, 11)
    )


def _assert_untransportable_notice(body, item_ids):
    assert scan_reserved_markers(body) == () and "AGENT_LOOP_META" not in body
    assert "<!--" not in body
    assert _posted_records([body]) == []
    assert "**Not claimed (every carried item stays open):**" in body
    for item_id in item_ids:
        assert f"`{item_id}`" in body
    assert "round-metadata transport bound" in body
    assert degradation.UNTRANSPORTABLE_NOTICE_TEXT in body


def test_untransportable_record_on_the_best_effort_path_posts_the_notice(tmp_path, monkeypatch):
    """Row untransportable-best-effort-path: real oversized ledger, lowered limits."""
    monkeypatch.setattr(round_transport, "_MAX_COMPRESSED", 4_000)
    posts: list[str] = []
    ledger = _oversized_ledger_2153()
    recovery = _recovery_2153(tmp_path, posts=posts, ledger=ledger)
    stop = QuotaResetExceededError("quota reset is too far away")
    with pytest.raises(QuotaResetExceededError) as raised:
        recovery.run(lambda: (_ for _ in ()).throw(stop))
    assert raised.value is stop
    (body,) = posts
    _assert_untransportable_notice(body, [item.item_id for item in ledger])
    recovery.record_failed_push(stop)
    assert len(posts) == 1


def test_failed_notice_post_never_masks_the_original_error(tmp_path, monkeypatch, capsys):
    """A failing notice post is logged; the stopping error still re-raises."""
    monkeypatch.setattr(round_transport, "_MAX_COMPRESSED", 4_000)
    posts: list[str] = []
    recovery = _recovery_2153(tmp_path, posts=posts, ledger=_oversized_ledger_2153(), fail_post=True)
    stop = QuotaResetExceededError("quota reset is too far away")
    with pytest.raises(QuotaResetExceededError) as raised:
        recovery.run(lambda: (_ for _ in ()).throw(stop))
    assert raised.value is stop and len(posts) == 1


def test_bound_reraises_the_recovery_original_error_after_the_notice(tmp_path):
    """The ordinary-path wrapper: notice, then the recovery's original error."""
    from coding_review_agent_loop.coder_comment_budget import CoderRecordUntransportableError

    posts: list[str] = []
    recovery = _recovery_2153(tmp_path, posts=posts)
    original = AgentLoopError("unparseable")
    recovery.run(lambda: (_ for _ in ()).throw(original))
    untransportable = CoderRecordUntransportableError("too large", head_sha=HEAD, old_head=OLD, item_ids=())

    def make_bounded():
        raise untransportable

    with pytest.raises(AgentLoopError) as raised:
        recovery.bound(make_bounded)
    assert raised.value is original and raised.value.__cause__ is untransportable
    (body,) = posts
    _assert_untransportable_notice(body, [item.item_id for item in ITEMS])
