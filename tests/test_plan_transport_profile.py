"""Canonical plan size against published round transport (#1075)."""

import base64
import json
import os
from types import SimpleNamespace

import coding_review_agent_loop.plan_transport_profile as profile
import coding_review_agent_loop.round_transport as transport
from coding_review_agent_loop.config import AgentLoopConfig
from coding_review_agent_loop.plan_growth import (
    DEFAULT_PLAN_GROWTH_MAX_CHARS,
    DEFAULT_PLAN_GROWTH_MAX_REVISIONS,
    PLAN_TRANSPORT_MEASURED_METADATA_FLOOR_CHARS,
    PLAN_TRANSPORT_MEASURED_VISIBLE_RATIO,
    PLAN_TRANSPORT_PROJECTED_CLIFF_CHARS,
)
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


def test_default_size_threshold_is_derived_from_the_measured_cliff():
    assert PLAN_TRANSPORT_MEASURED_VISIBLE_RATIO == 17_791 / 68_956
    assert PLAN_TRANSPORT_MEASURED_METADATA_FLOOR_CHARS == 24_995
    assert 135_000 < PLAN_TRANSPORT_PROJECTED_CLIFF_CHARS < 136_500
    assert DEFAULT_PLAN_GROWTH_MAX_CHARS == 120_000
    # At least 10% under the cliff, and the combining revision signal starts
    # below the known-good #886 plan (68,956 characters, 7 candidates).
    assert DEFAULT_PLAN_GROWTH_MAX_CHARS <= PLAN_TRANSPORT_PROJECTED_CLIFF_CHARS * 0.9
    assert DEFAULT_PLAN_GROWTH_MAX_CHARS // 2 <= 68_956
    assert DEFAULT_PLAN_GROWTH_MAX_REVISIONS <= 7
    assert AgentLoopConfig.__dataclass_fields__["plan_growth_max_chars"].default == (
        DEFAULT_PLAN_GROWTH_MAX_CHARS
    )


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
    assert sample.projected_cliff_chars == (
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
    assert unsettled_sample.projected_cliff_chars is None
    assert settled_sample.transport_settled is True
    assert settled_sample.projected_cliff_chars is not None
    summary = summarize([unsettled_sample, settled_sample])
    assert summary.samples == 2 and summary.settled_samples == 1
    assert summary.lowest_projected_cliff_chars == settled_sample.projected_cliff_chars
    assert summarize([unsettled_sample]).lowest_projected_cliff_chars is None


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


def test_summary_reports_the_lowest_projected_cliff():
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
    assert summary.lowest_projected_cliff_chars == 100_000
    assert summary.median_compression_ratio == 0.5
