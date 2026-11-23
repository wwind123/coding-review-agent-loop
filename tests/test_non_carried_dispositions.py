"""Dispositions for non-carried item IDs are dropped, not rejected (#2100)."""

from __future__ import annotations

import dataclasses
import json
import re

import pytest

import coding_review_agent_loop.architecture_contract as architecture_contract
import coding_review_agent_loop.child_plan_binding as child_plan_binding
import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
import coding_review_agent_loop.review_rounds as review_rounds
import coding_review_agent_loop.round_transport as round_transport
from agent_loop_helpers import (
    FakeRunner,
    make_config,
    structured_plan_review,
    structured_plan_revision,
    structured_pr_review,
)
from coding_review_agent_loop.comment_rendering import render_public_agent_comment
from coding_review_agent_loop.errors import AgentLoopError, MissingPriorItemDispositionError
from coding_review_agent_loop.native_evidence import publication_dispositions
from coding_review_agent_loop.protocol import (
    ParsedPlanReview,
    ParsedReview,
    PlanRevisionPatch,
    UnresolvedReviewItem,
    parse_pr_review,
    parse_structured_plan_review,
    parse_structured_pr_review,
)
from coding_review_agent_loop.protocol_common import partition_raw_dispositions
from coding_review_agent_loop.response_normalization import remove_prior_item_dispositions
from coding_review_agent_loop.response_validation import (
    _validate_plan_revision_patch_payload,
    _validate_plan_revision_patch_response,
    _validate_plan_revision_response,
)
from coding_review_agent_loop.review_spool import ReviewRoundSpool
from coding_review_agent_loop.round_metadata import (
    IGNORED_DISPOSITIONS_HEADING,
    IGNORED_DISPOSITIONS_LINE_BUDGET,
    PostedRoundMetadata,
    _attach_round_metadata,
    _decode_round_metadata,
    _encode_round_metadata,
    render_ignored_dispositions_section,
)
from coding_review_agent_loop.unresolved_items import (
    _validate_plan_review_response,
    _validate_review_response,
)
from coding_review_agent_loop.validated_agent import _run_validated_agent

PLAN_FOOTER = "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
NOTE_PREFIX = "Ignored dispositions for non-carried IDs: "


def _item(item_id: str, *, status: str = "blocking", aliases=()) -> UnresolvedReviewItem:
    return UnresolvedReviewItem(
        item_id=item_id,
        reviewer="OpenAI Codex",
        source_round=1,
        text=f"Finding {item_id}.",
        status=status,
        alias_item_ids=tuple(aliases),
    )


def _patch_text(dispositions) -> str:
    payload = {
        "schema_version": 1,
        "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1,
        "state": "blocking",
        "summary": "Address the reviewer findings.",
        "prior_plan_item_dispositions": dispositions,
        "base_round_number": 2,
        "base_state_identity": "a" * 64,
        "operations": [{"op": "replace", "field": "summary", "value": "Revised."}],
    }
    return json.dumps(payload) + PLAN_FOOTER


def _resolved(*ids: str) -> list[dict[str, str]]:
    return [{"item_id": item_id, "disposition": "resolved", "note": "Done."} for item_id in ids]


def _payload(text: str) -> dict:
    return json.JSONDecoder().raw_decode(text.lstrip())[0]


def _meta(**fields) -> PostedRoundMetadata:
    base = dict(flow="plan", role="reviewer", agent="OpenAI Codex", round_number=2, subject="a" * 64)
    base.update(fields)
    return PostedRoundMetadata(**base)


def _attached(body: str, metadata: PostedRoundMetadata, **kwargs) -> str:
    return str(_attach_round_metadata(body, metadata, **kwargs))


def _decoded(body: str) -> PostedRoundMetadata:
    match = list(round_transport.ROUND_RESUME_MARKER_RE.finditer(body))[-1]
    return _decode_round_metadata(match.group("payload"))


# --- Partition helper ------------------------------------------------------


def test_partition_drops_only_well_formed_non_carried_ids():
    raw = [
        {"item_id": "item-1", "disposition": "resolved"},
        {"item_id": " item-9 ", "disposition": "resolved"},
        {"item_id": "item-9", "disposition": "blocking"},
        {"item_id": "alias-1", "disposition": "resolved"},
        {"item_id": "bad id!", "disposition": "resolved"},
        "not-an-object",
        {"item_id": 7},
    ]
    kept, dropped = partition_raw_dispositions(raw, frozenset({"item-1"}), frozenset({"alias-1"}))
    assert dropped == ("item-9",)
    assert kept == [raw[0], raw[3], raw[4], raw[5], raw[6]]
    # A non-array is passed through for the parser to reject as today.
    assert partition_raw_dispositions("nope", frozenset(), frozenset()) == ("nope", ())


# --- Semantic patch and legacy revision (plan candidates) ------------------


def test_semantic_patch_extras_are_dropped_by_payload_and_full_validators():
    carried = (_item("item-17"), _item("item-18"), _item("item-19"))
    text = _patch_text(_resolved("item-17", "item-1", "item-18", "item-19", "item-13"))

    full = _validate_plan_revision_patch_response(text, unresolved_items=carried)
    payload_only = _validate_plan_revision_patch_payload(_payload(text), unresolved_items=carried)

    for parsed in (full, payload_only):
        assert isinstance(parsed, PlanRevisionPatch)
        assert parsed.ignored_prior_disposition_ids == ("item-1", "item-13")
        assert [d.item_id for d in parsed.prior_plan_item_dispositions] == [
            "item-17", "item-18", "item-19",
        ]
        wire = parsed.to_payload()
        assert "ignored_prior_disposition_ids" not in wire
        assert [d["item_id"] for d in wire["prior_plan_item_dispositions"]] == [
            "item-17", "item-18", "item-19",
        ]
        assert parsed.base_state_identity == "a" * 64


def test_semantic_patch_with_unbounded_extras_is_accepted():
    carried = (_item("item-1"),)
    many = [f"item-{index}" for index in range(100, 300)]
    long_id = "x" * 2000
    text = _patch_text(_resolved("item-1", *many, long_id))
    parsed = _validate_plan_revision_patch_response(text, unresolved_items=carried)
    assert parsed.ignored_prior_disposition_ids == tuple(sorted([*many, long_id]))
    assert [d.item_id for d in parsed.prior_plan_item_dispositions] == ["item-1"]


def test_plan_candidates_missing_a_carried_disposition_behave_as_on_main():
    carried = (_item("item-4"), _item("item-5"))
    patch = _validate_plan_revision_patch_response(
        _patch_text(_resolved("item-4", "item-1")), unresolved_items=carried
    )
    revision = _validate_plan_revision_response(
        structured_plan_revision(prior_plan_item_dispositions=_resolved("item-4", "item-1")),
        unresolved_items=carried,
        architecture_status_mode="legacy",
    )
    for parsed in (patch, revision):
        # Accepted, item-5 simply stays carried, item-1 is recorded as ignored.
        assert [d.item_id for d in parsed.prior_plan_item_dispositions] == ["item-4"]
        assert parsed.ignored_prior_disposition_ids == ("item-1",)


# --- Reviewer validators ---------------------------------------------------


def test_reviewer_missing_carried_disposition_still_fails_with_an_extra():
    carried = (_item("item-4"), _item("item-5"))
    with pytest.raises(MissingPriorItemDispositionError) as error:
        _validate_review_response(
            structured_pr_review(state="blocking", prior_item_dispositions=_resolved("item-4", "item-1")),
            reviewer="OpenAI Codex",
            unresolved_items=carried,
            architecture_status_mode="legacy",
        )
    assert error.value.missing_ids == ("item-5",)
    with pytest.raises(AgentLoopError, match="did not evaluate all prior unresolved plan items: item-5"):
        _validate_plan_review_response(
            structured_plan_review(state="blocking", prior_plan_item_dispositions=_resolved("item-4", "item-1")),
            reviewer="OpenAI Codex",
            unresolved_items=carried,
            architecture_status_mode="legacy",
        )


def test_reviewer_extras_are_dropped_and_alias_still_maps_to_its_item():
    carried = (_item("item-2", aliases=("legacy-2",)), _item("item-3"))
    pr = _validate_review_response(
        structured_pr_review(
            state="blocking",
            prior_item_dispositions=[
                *_resolved("legacy-2", "item-3", "item-7"),
            ],
        ),
        reviewer="OpenAI Codex",
        unresolved_items=carried,
        architecture_status_mode="legacy",
    )
    assert [d.item_id for d in pr.dispositions] == ["legacy-2", "item-3"]
    assert pr.ignored_prior_disposition_ids == ("item-7",)

    plan = _validate_plan_review_response(
        structured_plan_review(state="blocking", prior_plan_item_dispositions=_resolved("item-2", "item-3", "item-7")),
        reviewer="OpenAI Codex",
        unresolved_items=carried,
        architecture_status_mode="legacy",
    )
    assert [d.item_id for d in plan.dispositions] == ["item-2", "item-3"]
    assert plan.ignored_prior_disposition_ids == ("item-7",)

    # The public reviewer comment renders the note from round metadata.
    for parsed, kind, flow in ((pr, "pr_review", "pr"), (plan, "plan_review", "plan")):
        body = render_public_agent_comment(
            kind=kind, parsed=parsed, agent="codex", prior_items=carried,
            dispositions=parsed.dispositions, config=make_config_for_rendering(),
        )
        published = _attached(
            body,
            _meta(flow=flow, **architecture_contract.ignored_disposition_metadata_fields(parsed)),
        )
        assert IGNORED_DISPOSITIONS_HEADING in published
        assert f"{NOTE_PREFIX}`item-7`" in published
        assert _decoded(published).ignored_prior_disposition_ids == ("item-7",)


def make_config_for_rendering():
    import tempfile
    from pathlib import Path

    return make_config(Path(tempfile.mkdtemp()))


@pytest.mark.parametrize("carried_ids", [("item-8",), ()])
def test_approved_review_with_only_a_non_carried_active_disposition_is_accepted(carried_ids):
    carried = tuple(_item(item_id) for item_id in carried_ids)
    active = [{"item_id": "item-1", "disposition": "blocking", "note": "Still broken here."}]
    pr = _validate_review_response(
        structured_pr_review(state="approved", prior_item_dispositions=[*_resolved(*carried_ids), *active]),
        reviewer="OpenAI Codex",
        unresolved_items=carried,
        architecture_status_mode="legacy",
    )
    assert pr.state == "approved" and pr.ignored_prior_disposition_ids == ("item-1",)
    plan_active = [{"item_id": "item-1", "disposition": "same-plan", "note": "Still open."}]
    plan = _validate_plan_review_response(
        structured_plan_review(
            state="approved", prior_plan_item_dispositions=[*_resolved(*carried_ids), *plan_active]
        ),
        reviewer="OpenAI Codex",
        unresolved_items=carried,
        architecture_status_mode="legacy",
    )
    assert plan.state == "approved" and plan.ignored_prior_disposition_ids == ("item-1",)


def test_approval_with_an_active_carried_disposition_is_still_rejected():
    carried = (_item("item-8"),)
    with pytest.raises(AgentLoopError, match="Approved reviews must be fully complete"):
        _validate_review_response(
            structured_pr_review(
                state="approved",
                prior_item_dispositions=[{"item_id": "item-8", "disposition": "blocking", "note": "Broken."}],
            ),
            reviewer="OpenAI Codex",
            unresolved_items=carried,
            architecture_status_mode="legacy",
        )
    with pytest.raises(AgentLoopError, match="Approved plan reviews must be fully complete"):
        _validate_plan_review_response(
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[{"item_id": "item-8", "disposition": "blocking", "note": "Broken."}],
            ),
            reviewer="OpenAI Codex",
            unresolved_items=carried,
            architecture_status_mode="legacy",
        )
    # Historical parsing (no carried set) keeps today's behaviour.
    with pytest.raises(AgentLoopError, match="Approved reviews must be fully complete"):
        parse_pr_review(
            structured_pr_review(
                state="approved",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "blocking", "note": "Broken."}],
            ),
            reviewer="OpenAI Codex",
        )


def test_duplicates_split_between_dropped_and_rejected():
    carried = (_item("item-1"),)
    pr = _validate_review_response(
        structured_pr_review(state="blocking", prior_item_dispositions=_resolved("item-1", "item-9", "item-9")),
        reviewer="OpenAI Codex", unresolved_items=carried, architecture_status_mode="legacy",
    )
    assert pr.ignored_prior_disposition_ids == ("item-9",)
    plan = _validate_plan_review_response(
        structured_plan_review(state="blocking", prior_plan_item_dispositions=_resolved("item-1", "item-9", "item-9")),
        reviewer="OpenAI Codex", unresolved_items=carried, architecture_status_mode="legacy",
    )
    assert plan.ignored_prior_disposition_ids == ("item-9",)
    patch = _validate_plan_revision_patch_response(
        _patch_text(_resolved("item-1", "item-9", "item-9")), unresolved_items=carried
    )
    assert patch.ignored_prior_disposition_ids == ("item-9",)
    revision = _validate_plan_revision_response(
        structured_plan_revision(prior_plan_item_dispositions=_resolved("item-1", "item-9", "item-9")),
        unresolved_items=carried, architecture_status_mode="legacy",
    )
    assert revision.ignored_prior_disposition_ids == ("item-9",)

    with pytest.raises(AgentLoopError, match="more than once: item-1"):
        _validate_review_response(
            structured_pr_review(state="blocking", prior_item_dispositions=_resolved("item-1", "item-1")),
            reviewer="OpenAI Codex", unresolved_items=carried, architecture_status_mode="legacy",
        )
    with pytest.raises(AgentLoopError, match="more than once: item-1"):
        _validate_plan_review_response(
            structured_plan_review(state="blocking", prior_plan_item_dispositions=_resolved("item-1", "item-1")),
            reviewer="OpenAI Codex", unresolved_items=carried, architecture_status_mode="legacy",
        )


# --- Acceptance boundary ---------------------------------------------------


def _pr_validator(carried):
    return lambda text: _validate_review_response(
        text, reviewer="OpenAI Codex", unresolved_items=carried, architecture_status_mode="legacy"
    )


def _plan_review_validator(carried, mode="legacy"):
    return lambda text: _validate_plan_review_response(
        text, reviewer="OpenAI Codex", unresolved_items=carried, architecture_status_mode=mode
    )


def test_accept_boundary_canonicalizes_all_four_kinds():
    carried = (_item("item-1"),)
    cases = [
        (
            structured_pr_review(state="approved", prior_item_dispositions=_resolved("item-1", "item-9")),
            _pr_validator(carried),
            "prior_item_dispositions",
        ),
        (
            structured_plan_review(state="approved", prior_plan_item_dispositions=_resolved("item-1", "item-9")),
            _plan_review_validator(carried),
            "prior_plan_item_dispositions",
        ),
        (
            structured_plan_revision(prior_plan_item_dispositions=_resolved("item-1", "item-9")),
            lambda text: _validate_plan_revision_response(
                text, unresolved_items=carried, architecture_status_mode="legacy"
            ),
            "prior_plan_item_dispositions",
        ),
        (
            _patch_text(_resolved("item-1", "item-9")),
            lambda text: _validate_plan_revision_patch_response(text, unresolved_items=carried),
            "prior_plan_item_dispositions",
        ),
    ]
    for text, validate, field in cases:
        parsed = validate(text)
        accepted = architecture_contract._accept_candidate(
            text, parsed, strict_revalidate=validate, runner=None, acquisition=None,
        )
        assert [entry["item_id"] for entry in _payload(accepted.text)[field]] == ["item-1"]
        assert accepted.text.rstrip().endswith(text.rstrip().rsplit("\n", 2)[-1])
        assert "<!--" in accepted.text.split("\n", 1)[1]
        assert accepted.marker_value.ignored_prior_disposition_ids == ("item-9",)
        assert type(accepted.marker_value) is type(parsed)
        # Ignore-free candidates pass through byte-identical.
        clean = text.replace(', {"item_id": "item-9", "disposition": "resolved", "note": "Done."}', "")
        clean_parsed = validate(clean)
        clean_accepted = architecture_contract._accept_candidate(
            clean, clean_parsed, strict_revalidate=None, runner=None, acquisition=None,
        )
        assert clean_accepted.text == clean and clean_accepted.marker_value is clean_parsed


def test_accept_boundary_without_strict_reparse_uses_the_fallback_validate():
    carried = (_item("item-1"),)
    text = _patch_text(_resolved("item-1", "item-9"))

    def validate(candidate):
        return _validate_plan_revision_patch_response(candidate, unresolved_items=carried)

    parsed = validate(text)
    accepted = architecture_contract._accept_candidate(
        text, parsed, strict_revalidate=None, runner=None, acquisition=None,
        fallback_revalidate=validate,
    )
    assert accepted.marker_value.ignored_prior_disposition_ids == ("item-9",)
    with pytest.raises(architecture_contract._AcceptedTextCanonicalizationError):
        architecture_contract._accept_candidate(
            text, parsed, strict_revalidate=None, runner=None, acquisition=None,
        )


def test_accept_boundary_combines_architecture_records_and_ignored_ids():
    carried = (_item("item-1"),)
    payload = _payload(
        structured_plan_review(state="approved", prior_plan_item_dispositions=_resolved("item-1", "item-9"))
    )
    payload["architecture_impact"] = {"status": "modified", "rationale": "Changed one module."}
    text = json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"
    degradable = _plan_review_validator(carried, "degradable")
    strict = _plan_review_validator(carried, "strict")

    parsed = degradable(text)
    assert parsed.architecture_impact_degradations
    accepted = architecture_contract._accept_candidate(
        text, parsed, strict_revalidate=strict, runner=None, acquisition=None,
    )
    rewritten = _payload(accepted.text)
    assert [entry["item_id"] for entry in rewritten["prior_plan_item_dispositions"]] == ["item-1"]
    assert "architecture_impact" not in rewritten or rewritten["architecture_impact"]["status"] != "modified"
    assert accepted.marker_value.ignored_prior_disposition_ids == ("item-9",)
    assert accepted.marker_value.architecture_impact_degradations == parsed.architecture_impact_degradations


def test_parser_records_stay_consistent_across_the_drop():
    carried = (dataclasses.replace(_item("item-2"), sub_items=()),)
    payload = _payload(structured_pr_review(state="blocking"))
    payload["prior_item_dispositions"] = [
        {"item_id": "item-9", "disposition": "resolved", "sub_item_dispositions": []},
        {
            "item_id": "item-2", "disposition": "blocking", "note": "Still broken in module X.",
            "sub_item_dispositions": {"s1": "maybe"},
        },
    ]
    text = json.dumps(payload) + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    validate = _pr_validator(carried)
    parsed = validate(text)
    paths = [record.element_path for record in parsed.sub_item_degradations]
    assert paths == ["pr_review.prior_item_dispositions[0].sub_item_dispositions.s1"]
    accepted = architecture_contract._accept_candidate(
        text, parsed, strict_revalidate=validate, runner=None, acquisition=None,
    )
    assert accepted.marker_value.sub_item_degradations == parsed.sub_item_degradations
    assert accepted.marker_value.ignored_prior_disposition_ids == ("item-9",)


def test_accept_boundary_kind_map_covers_reviewer_types_and_refuses_unmapped():
    assert architecture_contract._IGNORED_ID_CARRIER_KINDS[ParsedReview] == "pr_review"
    assert architecture_contract._IGNORED_ID_CARRIER_KINDS[ParsedPlanReview] == "plan_review"
    carried = (_item("item-1"),)
    text = structured_pr_review(state="approved", prior_item_dispositions=_resolved("item-1", "item-9"))
    parsed = _pr_validator(carried)(text)

    @dataclasses.dataclass(frozen=True)
    class UnmappedReview(ParsedReview):
        pass

    unmapped = UnmappedReview(**{f.name: getattr(parsed, f.name) for f in dataclasses.fields(parsed)})
    with pytest.raises(architecture_contract._AcceptedTextCanonicalizationError, match="no wire kind"):
        architecture_contract._accept_candidate(
            text, unmapped, strict_revalidate=_pr_validator(carried), runner=None, acquisition=None,
        )


def test_remove_helper_uses_the_parser_normalization():
    text = _patch_text([
        {"item_id": " item-9 ", "disposition": "resolved"},
        {"item_id": "item-1", "disposition": "resolved"},
    ])
    rewritten = remove_prior_item_dispositions(text, drop_ids=("item-9",), expected_kind="plan_revision_patch")
    assert [e["item_id"] for e in _payload(rewritten)["prior_plan_item_dispositions"]] == ["item-1"]
    assert rewritten.endswith(PLAN_FOOTER)


# --- Live turn, publication and resume -------------------------------------


def _run_reviewer(tmp_path, text, validate):
    runner = FakeRunner(codex_outputs=[text])
    config = make_config(tmp_path, reviewer="codex", agent_max_retries=0)
    return _run_validated_agent(
        runner, agent="codex", config=config, prompt="Review.",
        marker_description="<!-- AGENT_STATE: approved|blocking -->", validate=validate,
    )


def test_approved_review_with_dropped_entry_publishes_and_resumes(tmp_path):
    carried = (_item("item-8"),)
    active = [{"item_id": "item-1", "disposition": "blocking", "note": "Still broken here."}]

    pr_text = structured_pr_review(state="approved", prior_item_dispositions=[*_resolved("item-8"), *active])
    pr_response = _run_reviewer(tmp_path / "pr", pr_text, _pr_validator(carried))
    assert "item-1" not in json.dumps(_payload(pr_response.text)["prior_item_dispositions"])
    parsed = pr_response.marker_value
    # publication_dispositions reparses the accepted text without a carried set.
    kept = publication_dispositions(parsed, pr_response.text, "OpenAI Codex", carried)
    assert [d.item_id for d in kept] == ["item-8"]
    published = _attached(
        "Approved.\n\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        _meta(
            flow="pr", state="approved", dispositions=kept, phase="publication",
            canonical_reviewer_response=pr_response.text,
            **architecture_contract.ignored_disposition_metadata_fields(parsed),
        ),
    )
    assert f"{NOTE_PREFIX}`item-1`" in published
    resumed = _decoded(published)
    reparsed = parse_structured_pr_review(
        resumed.canonical_reviewer_response, reviewer="OpenAI Codex", architecture_status_mode="legacy"
    )
    assert reparsed.state == "approved"
    assert [d.item_id for d in reparsed.dispositions] == ["item-8"]
    assert resumed.ignored_prior_disposition_ids == ("item-1",)

    plan_active = [{"item_id": "item-1", "disposition": "same-plan", "note": "Still open."}]
    plan_text = structured_plan_review(
        state="approved", prior_plan_item_dispositions=[*_resolved("item-8"), *plan_active]
    )
    plan_response = _run_reviewer(tmp_path / "plan", plan_text, _plan_review_validator(carried))
    published = _attached(
        "Approved.\n\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        _meta(
            state="approved", dispositions=plan_response.marker_value.dispositions,
            phase="publication", canonical_reviewer_response=plan_response.text,
            **architecture_contract.ignored_disposition_metadata_fields(plan_response.marker_value),
        ),
    )
    resumed = _decoded(published)
    reparsed = parse_structured_plan_review(
        resumed.canonical_reviewer_response, reviewer="OpenAI Codex", architecture_status_mode="legacy"
    )
    assert reparsed.state == "approved"
    assert [d.item_id for d in reparsed.dispositions] == ["item-8"]
    assert resumed.ignored_prior_disposition_ids == ("item-1",)


@pytest.mark.parametrize("kind", ["pr_review", "plan_review"])
def test_spool_replay_reattaches_ignored_ids_and_publishes_them(tmp_path, kind):
    carried = (_item("item-8"),)
    if kind == "pr_review":
        text = structured_pr_review(state="blocking", prior_item_dispositions=_resolved("item-8", "item-1"))
        validate = _pr_validator(carried)
    else:
        text = structured_plan_review(state="blocking", prior_plan_item_dispositions=_resolved("item-8", "item-1"))
        validate = _plan_review_validator(carried)
    response = _run_reviewer(tmp_path / "turn", text, validate)
    fields = review_rounds._spooled_response_fields(response)
    assert fields["ignored_prior_disposition_ids"] == ["item-1"]

    spool = ReviewRoundSpool(
        root=tmp_path / "spool", repo="OWNER/REPO", surface="plan", number=1,
        round_number=2, subject="a" * 64,
    )
    spool.store("OpenAI Codex", fields)
    loaded = spool.load("OpenAI Codex")
    assert loaded["ignored_prior_disposition_ids"] == ["item-1"]

    config = make_config(tmp_path, reviewer="codex")
    replayed = review_rounds._replay_spooled_review(
        None, config=config, reviewer="codex", fields=loaded,
        validators={"validate": validate, "strict_revalidate": validate},
    )
    assert replayed is not None
    marker = replayed.response.marker_value
    assert marker.ignored_prior_disposition_ids == ("item-1",)
    published = _attached(
        "Review.\n\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        _meta(**architecture_contract.ignored_disposition_metadata_fields(marker)),
    )
    assert f"{NOTE_PREFIX}`item-1`" in published
    assert _decoded(published).ignored_prior_disposition_ids == ("item-1",)

    malformed = dict(loaded, ignored_prior_disposition_ids=["bad id"])
    assert review_rounds._replay_spooled_review(
        None, config=config, reviewer="codex", fields=malformed,
        validators={"validate": validate, "strict_revalidate": validate},
    ) is None


def test_spool_record_without_ignored_ids_is_unchanged(tmp_path):
    spool = ReviewRoundSpool(
        root=tmp_path / "spool", repo="OWNER/REPO", surface="plan", number=1,
        round_number=2, subject="a" * 64,
    )
    spool.store("OpenAI Codex", {"text": "x"})
    stored = json.loads(spool._path("OpenAI Codex").read_text())
    assert "ignored_prior_disposition_ids" not in stored["response"]
    assert "ignored_prior_disposition_ids" not in spool.load("OpenAI Codex")


# --- Round metadata, transport and the visible note ------------------------


def test_metadata_without_ignored_ids_encodes_byte_identically_and_round_trips():
    plain = _meta()
    encoded = _encode_round_metadata(plain)
    assert "ignored_prior_disposition_ids" not in json.dumps(
        round_transport.decode_mapping(encoded)
    )
    assert _encode_round_metadata(_decode_round_metadata(encoded)) == encoded
    populated = _meta(ignored_prior_disposition_ids=("item-1", "item-2"))
    assert _decode_round_metadata(_encode_round_metadata(populated)).ignored_prior_disposition_ids == (
        "item-1", "item-2",
    )


def test_malformed_ignored_id_metadata_fails_closed():
    mapping = round_transport.decode_mapping(_encode_round_metadata(_meta()))
    mapping["ignored_prior_disposition_ids"] = ["ok-1", "bad id"]
    with pytest.raises((AgentLoopError, ValueError)):
        _decode_round_metadata(round_transport.encode_mapping(mapping))


def test_note_lists_every_id_within_budget_then_counts():
    section = render_ignored_dispositions_section(("item-1", "item-2"))
    assert section == f"{IGNORED_DISPOSITIONS_HEADING}\n\n{NOTE_PREFIX}`item-1`, `item-2`"
    assert render_ignored_dispositions_section(()) is None
    ids = tuple(f"item-{index:04d}-{'q' * 40}" for index in range(200))
    overflow = render_ignored_dispositions_section(ids)
    line = overflow.splitlines()[-1]
    assert len(line) <= IGNORED_DISPOSITIONS_LINE_BUDGET
    assert line.startswith(f"{NOTE_PREFIX}200 ID(s): `item-0000-")
    assert "complete list in round metadata" in line
    long_line = render_ignored_dispositions_section(("x" * 5000,)).splitlines()[-1]
    assert "0 of 1 shown; complete list in round metadata" in long_line
    assert len(long_line) <= IGNORED_DISPOSITIONS_LINE_BUDGET


def test_unbounded_ignored_ids_spill_losslessly_through_the_transport():
    import hashlib

    # Incompressible IDs, so the compressed anchor genuinely overflows.
    many = tuple(sorted(
        hashlib.sha256(str(index).encode()).hexdigest() for index in range(3000)
    )) + ("x" * 2000,)
    body = _attached(
        "Revision.\n\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        _meta(role="coder", agent="Anthropic Claude", ignored_prior_disposition_ids=many),
    )
    assert len(body) > round_transport.MAX_GITHUB_BODY_CHARS
    prepared = round_transport.prepare_round_comment(body)
    bodies = [str(item) for item in prepared]
    anchor = bodies[-1]
    assert len(anchor) <= round_transport.MAX_GITHUB_BODY_CHARS
    match = list(round_transport.ROUND_RESUME_MARKER_RE.finditer(anchor))[-1]
    hydrated, missing = round_transport.hydrate_mapping(
        round_transport.decode_mapping(match.group("payload")), bodies
    )
    assert missing == set()
    assert tuple(hydrated["ignored_prior_disposition_ids"]) == many
    assert "complete list in round metadata" in anchor


def _big_revision(prior):
    from issue_loop_test_helpers import _m948_noise

    payload = _payload(structured_plan_revision(prior_plan_item_dispositions=prior))
    payload["plan_steps"] = [f"Step {index}: {_m948_noise(index)}" for index in range(300)]
    return json.dumps(payload) + PLAN_FOOTER


@pytest.mark.parametrize("oversized", [True, False])
def test_overflow_digest_uses_compact_indicator_and_keeps_the_record(tmp_path, oversized):
    config = make_config(tmp_path, max_rounds=3, plan_execution_mode="plan-only")
    carried = (_item("item-1"),)
    raw_text = (
        _big_revision(_resolved("item-1", "item-13"))
        if oversized
        else structured_plan_revision(prior_plan_item_dispositions=_resolved("item-1", "item-13"))
    )
    parsed = _validate_plan_revision_response(raw_text, unresolved_items=carried, architecture_status_mode="legacy")
    assert parsed.ignored_prior_disposition_ids == ("item-13",)
    canonical_plan = plan_first_loop_module.render_canonical_plan_revision(parsed, carried, config)
    metadata = _meta(
        role="coder", agent="Anthropic Claude", canonical_plan=canonical_plan,
        raw_structured_coder_response=raw_text,
        **architecture_contract.ignored_disposition_metadata_fields(parsed),
    )
    full_comment = render_public_agent_comment(
        kind="plan_revision", parsed=parsed, agent=config.coder, prior_items=carried,
        raw_text=raw_text, config=config,
    )
    body = str(child_plan_binding._assemble_structured_plan_round_body(
        config=config, issue_number=56, kind="plan_revision", parsed_plan=parsed,
        full_comment=full_comment, metadata=metadata, raw_text=raw_text, prior_items=carried,
        model_used=None, surfaced_requirement_ids=(), requires_direct_discussion_ack=False,
    ))
    bodies = [str(item) for item in round_transport.prepare_round_comment(body)]
    anchor = bodies[-1]
    match = list(round_transport.ROUND_RESUME_MARKER_RE.finditer(anchor))[-1]
    hydrated, missing = round_transport.hydrate_mapping(
        round_transport.decode_mapping(match.group("payload")), bodies
    )
    assert missing == set()
    assert hydrated["ignored_prior_disposition_ids"] == ["item-13"]
    if oversized:
        assert "Ignored dispositions for non-carried IDs: 1 (complete list in round metadata)" in anchor
        assert IGNORED_DISPOSITIONS_HEADING not in anchor
    else:
        assert f"{IGNORED_DISPOSITIONS_HEADING}\n\n{NOTE_PREFIX}`item-13`" in anchor
        assert "(complete list in round metadata)" not in anchor
    assert re.search(r"item-13", canonical_plan) is None
