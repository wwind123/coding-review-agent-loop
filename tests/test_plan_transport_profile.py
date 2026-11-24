"""Canonical plan size against published round transport (#1075)."""

import base64
import json
import os
from types import SimpleNamespace

import coding_review_agent_loop.orchestrator as orchestrator_module
import coding_review_agent_loop.plan_transport_profile as profile
import coding_review_agent_loop.round_transport as transport
from agent_loop_helpers import make_config, structured_v1_plan_state
from coding_review_agent_loop.comment_rendering import (
    COMPACT_PLAN_DIGEST_BUDGET_CHARS,
    COMPACT_PLAN_DIGEST_NOTICE,
    render_canonical_plan_state,
    render_public_agent_comment,
)
from coding_review_agent_loop.config import AgentLoopConfig
from coding_review_agent_loop.plan_growth import (
    DEFAULT_PLAN_GROWTH_MAX_CHARS,
    DEFAULT_PLAN_GROWTH_MAX_REVISIONS,
    PLAN_TRANSPORT_MEASURED_LARGEST_METADATA_COMPRESSED_BYTES,
    PLAN_TRANSPORT_MEASURED_METADATA_FLOOR_CHARS,
    PLAN_TRANSPORT_MEASURED_VISIBLE_RATIO,
    PLAN_TRANSPORT_PROJECTED_DIGEST_TRANSITION_CHARS,
)
from coding_review_agent_loop.protocol import validate_structured_plan_state
from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata
from coding_review_agent_loop.plan_transport_profile import (
    PlanTransportSample,
    measure_plan_rounds,
    metadata_floor_chars,
    parse_paginated_pages,
    summarize,
)


def _random_text(size: int) -> str:
    return base64.urlsafe_b64encode(os.urandom(size)).decode("ascii")


def _comment(payload: dict[str, object], body: str) -> str:
    return f"{body}\n<!-- AGENT_LOOP_META: {transport.encode_mapping(payload)} -->"


def _plan_payload(canonical: str, **extra: object) -> dict[str, object]:
    return {"flow": "plan", "role": "coder", "round_number": 3, "canonical_plan": canonical, **extra}


def _publish(payload: dict[str, object], body: str) -> list[str]:
    return [str(item) for item in transport.prepare_round_comment(_comment(payload, body))]


def test_default_size_threshold_is_derived_from_the_digest_transition():
    assert PLAN_TRANSPORT_MEASURED_VISIBLE_RATIO == 17_791 / 68_956
    assert PLAN_TRANSPORT_MEASURED_METADATA_FLOOR_CHARS == 24_995
    assert 135_000 < PLAN_TRANSPORT_PROJECTED_DIGEST_TRANSITION_CHARS < 136_500
    assert DEFAULT_PLAN_GROWTH_MAX_CHARS == 120_000
    # At least 10% under the transition, and the combining revision signal
    # starts below the known-good #886 plan (68,956 characters, 7 candidates).
    assert DEFAULT_PLAN_GROWTH_MAX_CHARS <= PLAN_TRANSPORT_PROJECTED_DIGEST_TRANSITION_CHARS * 0.9
    assert DEFAULT_PLAN_GROWTH_MAX_CHARS // 2 <= 68_956
    assert DEFAULT_PLAN_GROWTH_MAX_REVISIONS <= 7
    assert AgentLoopConfig.__dataclass_fields__["plan_growth_max_chars"].default == (
        DEFAULT_PLAN_GROWTH_MAX_CHARS
    )


def test_inferred_codec_usage_is_far_below_the_codec_cap():
    """#871's final candidate, the largest plan seen, against the cap.

    The figure sums its independently compressed spilled fields, inferred
    from its nine spill sidecar comment lengths, and its residual anchor
    metadata: strip the sidecar label and marker framing and the JSON
    envelope, then undo two base64 layers.  It is not the compressed size of
    the complete metadata as one payload, which was not measured.
    """
    sidecar_lengths = [10_379, 10_690, 53_869, 3_085, 25_375, 25_908, 53_853, 12_877, 10_371]
    framing, envelope, residual_metadata = 180, 260, 24_995
    packed = sum(((length - framing) * 3 // 4 - envelope) * 3 // 4 for length in sidecar_lengths)
    estimate = packed + residual_metadata * 3 // 4
    assert abs(estimate - PLAN_TRANSPORT_MEASURED_LARGEST_METADATA_COMPRESSED_BYTES) < 1_000
    assert PLAN_TRANSPORT_MEASURED_LARGEST_METADATA_COMPRESSED_BYTES * 50 < transport._MAX_COMPRESSED


def _long_plan(config, steps: int):
    payload = json.loads(structured_v1_plan_state().split("\n", 1)[0])
    payload["plan_steps"] = [f"Step {index} " + "detail " * 150 for index in range(steps)]
    raw = json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    plan = validate_structured_plan_state(raw)
    canonical = render_canonical_plan_state(plan, config)
    metadata = PostedRoundMetadata(
        flow="plan", role="coder", agent="Claude", round_number=2, subject="s",
        canonical_plan=canonical, raw_structured_coder_response=raw,
    )
    full = render_public_agent_comment(kind="plan_state", parsed=plan, agent="claude", config=config)
    body = orchestrator_module._assemble_structured_plan_round_body(
        config=config, issue_number=1075, kind="plan_state", parsed_plan=plan,
        full_comment=full, metadata=metadata, raw_text=raw, prior_items=(), model_used=None,
        surfaced_requirement_ids=(), requires_direct_discussion_ack=False,
    )
    published = [str(item) for item in transport.prepare_round_comment(body)]
    return canonical, _attach_round_metadata(full, metadata), published


def test_assembler_overflow_publishes_a_digest_the_profile_recognises(tmp_path):
    """The real full-comment-to-digest transition, measured end to end.

    A plan whose full comment overflows goes through the orchestrator's
    assembler, which selects the bounded digest; the published digest is
    profiled as past the transition, with headroom to its own overflow.
    A small plan through the same assembler stays a full comment.
    """
    config = make_config(tmp_path, max_rounds=4)
    canonical, full_body, published = _long_plan(config, steps=80)

    assert not transport.round_comment_fits(full_body)
    assert COMPACT_PLAN_DIGEST_NOTICE in published[-1]
    [sample] = measure_plan_rounds(published)
    assert sample.canonical_chars == len(canonical) > transport.MAX_GITHUB_BODY_CHARS
    assert sample.digest is True
    assert sample.projected_digest_transition_chars is None
    assert sample.digest_headroom_chars is not None and sample.digest_headroom_chars > 0
    assert sample.anchor_visible_chars < transport.MAX_GITHUB_BODY_CHARS // 2

    small_canonical, small_full, small_published = _long_plan(config, steps=2)
    assert transport.round_comment_fits(small_full)
    [small] = measure_plan_rounds(small_published)
    assert small.digest is False and small.canonical_chars == len(small_canonical)
    assert small.digest_headroom_chars is None


def test_spilled_plan_round_measures_visible_text_floor_and_sidecars():
    canonical = _random_text(90_000)
    visible = "## Revised plan\n\n" + "Step detail. " * 1_500
    bodies = ["unrelated human comment", *_publish(_plan_payload(canonical), visible)]

    [sample] = measure_plan_rounds(bodies, issue_number=886)

    anchor = bodies[-1]
    match = transport.ROUND_RESUME_MARKER_RE.search(anchor)
    assert sample.issue_number == 886 and sample.round_number == 3
    assert sample.canonical_chars == len(canonical)
    assert sample.anchor_chars == len(anchor) <= transport.MAX_GITHUB_BODY_CHARS
    assert sample.anchor_metadata_chars == len(match.group("payload"))
    assert sample.anchor_visible_chars == len(anchor) - len(match.group("payload"))
    assert sample.spilled_fields == ("canonical_plan",)
    assert sample.sidecar_comments == len(bodies) - 2 >= 2
    # Random text does not compress, so encoding inflates it.
    assert sample.compression_ratio > 1
    assert sample.metadata_floor_chars <= sample.anchor_metadata_chars + 200
    assert sample.digest is False and sample.digest_headroom_chars is None
    assert sample.projected_digest_transition_chars == (
        len(canonical)
        * (transport.MAX_GITHUB_BODY_CHARS - sample.metadata_floor_chars)
        // sample.anchor_visible_chars
    )


def test_inline_plan_round_reports_compression_and_no_sidecars():
    # Repetitive prose compresses away; the random half does not.
    canonical = "Add the parser seam and its tests. " * 240 + _random_text(6_000)
    bodies = _publish(_plan_payload(canonical), "## Plan\n\nShort plan.")

    [sample] = measure_plan_rounds(bodies)

    assert sample.spilled_fields == ()
    assert sample.sidecar_comments == 0
    assert 0.4 < sample.compression_ratio < 0.7
    # The floor replaces the inline canonical text with a small reference.
    assert sample.metadata_floor_chars < sample.anchor_metadata_chars


def test_unhydratable_and_non_plan_rounds_are_skipped():
    spilled = _publish(_plan_payload(_random_text(90_000)), "## Plan\n\nBody.")
    reviewer = _comment(
        {"flow": "plan", "role": "reviewer", "round_number": 3, "canonical_plan": "x"}, "Review"
    )
    pr_coder = _comment({"flow": "pr", "role": "coder", "canonical_plan": "x"}, "PR")

    assert measure_plan_rounds([spilled[-1], reviewer, pr_coder]) == []


def test_metadata_floor_spills_every_growing_field():
    small = {"flow": "plan", "role": "coder", "compact_prior_summaries": ["kept inline"]}
    grown = {
        **small,
        "canonical_plan": _random_text(40_000),
        "raw_structured_coder_response": _random_text(40_000),
        "assembled_plan_sidecar": {"canonical_json": {"bulk": _random_text(40_000)}},
        "prior_items": [{"id": _random_text(20_000)}],
    }

    floor = metadata_floor_chars(grown)

    assert floor < 3_000
    assert metadata_floor_chars(small) == len(transport.encode_mapping(small))


def _marker(token: str, payload: dict[str, object]) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    return f"<!-- {token}: {encoded} -->"


def test_inline_recommendation_or_matrix_leaves_the_anchor_unsettled():
    inline = "## Plan\n\n" + _marker("AGENT_EXECUTION_RECOMMENDATION", {"strategy": "one-shot"})
    reference = {
        "$round_transport_risk_test_matrix": "a" * 24,
        "field": "risk_test_matrix_marker",
        "parts": 1,
        "sha256": "b" * 64,
        "spill": "c" * 64,
    }
    settled = "## Plan\n\n" + _marker("AGENT_RISK_TEST_MATRIX", reference)
    payload = _plan_payload("Plan text. " * 50)

    [unsettled_sample] = measure_plan_rounds([_comment(payload, inline)])
    [settled_sample] = measure_plan_rounds([_comment(payload, settled)])

    assert unsettled_sample.transport_settled is False
    assert unsettled_sample.projected_digest_transition_chars is None
    assert settled_sample.transport_settled is True
    assert settled_sample.projected_digest_transition_chars is not None
    summary = summarize([unsettled_sample, settled_sample])
    assert summary.samples == 2 and summary.projected_samples == 1
    assert summary.lowest_projected_digest_transition_chars == (
        settled_sample.projected_digest_transition_chars
    )
    only_unsettled = summarize([unsettled_sample])
    assert only_unsettled.lowest_projected_digest_transition_chars is None
    assert only_unsettled.median_visible_ratio is None


def test_digest_notice_lead_matches_the_renderer():
    assert COMPACT_PLAN_DIGEST_NOTICE.startswith(profile._DIGEST_NOTICE_LEAD)


def test_digest_anchor_is_past_the_transition_and_reports_its_headroom():
    """A plan whose full comment overflows publishes as a bounded digest.

    The digest's visible text does not grow with the plan, so the sample
    projects no transition; it reports how far the digest is from its own
    overflow instead.
    """
    canonical = _random_text(150_000)
    digest_body = (
        "## Revised plan\n\n"
        + COMPACT_PLAN_DIGEST_NOTICE
        + "\n\n"
        + "Digest step. " * (COMPACT_PLAN_DIGEST_BUDGET_CHARS // 20)
    )
    full_body = "## Revised plan\n\n" + "Step detail. " * 3_000
    payload = _plan_payload(canonical)
    digest_bodies = _publish(payload, digest_body)

    [digest_sample] = measure_plan_rounds(digest_bodies)
    [full_sample] = measure_plan_rounds(_publish(payload, full_body))

    assert digest_sample.digest is True and full_sample.digest is False
    assert digest_sample.projected_digest_transition_chars is None
    assert full_sample.projected_digest_transition_chars is not None
    assert digest_sample.digest_headroom_chars == (
        transport.MAX_GITHUB_BODY_CHARS
        - digest_sample.anchor_visible_chars
        - digest_sample.metadata_floor_chars
    ) > 0
    # Same canonical size, bounded visible text: the digest is far smaller.
    assert digest_sample.anchor_visible_chars < full_sample.anchor_visible_chars
    summary = summarize([digest_sample, full_sample])
    assert summary.digest_samples == 1 and summary.projected_samples == 1
    assert summary.largest_digest_visible_chars == digest_sample.anchor_visible_chars
    assert summary.lowest_digest_headroom_chars == digest_sample.digest_headroom_chars
    assert summary.median_visible_ratio == round(full_sample.visible_ratio, 3)


def test_paginated_pages_parse_without_slurp():
    stdout = '[{"body": "a"}, {"body": "b\\nc"}]\n[{"body": "d"}]\n'

    assert parse_paginated_pages(stdout) == [{"body": "a"}, {"body": "b\nc"}, {"body": "d"}]
    assert parse_paginated_pages("") == []


def test_fetch_does_not_pass_slurp(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout='[{"body": "x"}][{"body": null}]')

    monkeypatch.setattr(profile.subprocess, "run", fake_run)

    assert profile._fetch_issue_bodies("o/r", 886, "gh") == ["x", ""]
    assert "--slurp" not in calls[0] and "--paginate" in calls[0]


def test_summary_reports_the_lowest_projected_transition():
    def sample(canonical: int, visible: int, floor: int) -> PlanTransportSample:
        return PlanTransportSample(
            issue_number=None,
            round_number=1,
            canonical_chars=canonical,
            canonical_encoded_chars=canonical // 2,
            anchor_chars=visible + 30_000,
            anchor_metadata_chars=30_000,
            metadata_floor_chars=floor,
            spilled_fields=(),
            sidecar_comments=0,
        )

    summary = summarize([sample(68_956, 17_791, 24_995), sample(40_000, 20_000, 10_000)])

    assert summary.samples == 2
    assert summary.largest_published_chars == 68_956
    assert summary.largest_metadata_floor_chars == 24_995
    assert summary.lowest_projected_digest_transition_chars == 100_000
    assert summary.median_compression_ratio == 0.5
    assert summary.digest_samples == 0 and summary.lowest_digest_headroom_chars is None
