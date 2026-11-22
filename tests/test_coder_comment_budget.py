"""Bounded PR-loop coder round comment (#2059)."""

from __future__ import annotations

import json
from dataclasses import replace
import re
from types import SimpleNamespace

import pytest

from coding_review_agent_loop.coder_comment_budget import (
    FALLBACK_LEVEL,
    FALLBACK_NOTE,
    FREEFORM_OMISSION_NOTE,
    bounded_coder_round_comment,
)
from coding_review_agent_loop.comment_rendering import render_public_agent_comment
from coding_review_agent_loop.protocol import (
    ArchitectureImpact,
    StructuredCoderFollowup,
    StructuredHumanRequirementsPayload,
    UnresolvedReviewItem,
)
from coding_review_agent_loop.protocol_common import ParseDegradation
from coding_review_agent_loop.risk_coverage_map import (
    CoverageAssessment,
    CoverageDeficiency,
    CoverageRowEntry,
    extract_coverage_map_section,
    render_coverage_map,
)
from coding_review_agent_loop.round_metadata import (
    COMPACT_DEGRADATIONS_HEADING,
    PostedRoundMetadata,
    _attach_round_metadata,
)
from coding_review_agent_loop.round_state import _extract_round_metadata_records, _resume_pr_round
from coding_review_agent_loop.round_transport import (
    MAX_GITHUB_BODY_CHARS,
    RoundCommentOverflowError,
    prepare_round_comment,
)
from coding_review_agent_loop.unresolved_items import _upsert_evidence_obligation

from agent_loop_helpers import make_config

HEAD = "4" * 40
PRIOR = "d" * 40
OTHER_HEADS = ("1" * 40, "2" * 40, "3" * 40)
PR = 2310


def _code_item(number, text=None):
    return UnresolvedReviewItem(
        item_id=f"item-{number}", reviewer="Codex", source_round=1,
        text=text or f"Fix defect number {number}: " + "the parser drops the trailing field. " * 20,
        status="blocking",
    )


def _evidence_item(number, *, reviewer, head):
    """A native exact-head evidence request built by the real ledger writer."""
    ledger, _ = _upsert_evidence_obligation(
        [], item_number=number, reviewer=reviewer,
        text=f"Attach a human-only exact-head capture of the release smoke run at {head}.",
        source_round=2, current_head_sha=head, known_heads=OTHER_HEADS,
    )
    (item,) = ledger
    assert item.item_id == f"item-{number}"
    return item


def _followup(addressed, remaining=(), *, summary="Addressed the round findings.", note="", **extra):
    return StructuredCoderFollowup(
        schema_version=1, kind="coder_followup", state="blocking", summary=summary,
        addressed_items=tuple(addressed), remaining_items=tuple(remaining),
        human_requirements=StructuredHumanRequirementsPayload((), False),
        addressed_item_notes={item_id: note for item_id in addressed if note},
        remaining_item_notes={item_id: note for item_id in remaining if note},
        **extra,
    )


def _raw(followup):
    return json.dumps({
        "schema_version": 1, "kind": "coder_followup", "state": "blocking",
        "summary": followup.summary, "addressed_items": list(followup.addressed_items),
        "remaining_items": list(followup.remaining_items),
        "addressed_item_notes": followup.addressed_item_notes,
        "remaining_item_notes": followup.remaining_item_notes,
        "tests_run": list(followup.tests_run or ()),
    })


def _metadata(items, *, raw=None, **extra):
    return PostedRoundMetadata(
        flow="pr", role="coder", agent="Anthropic Claude", round_number=17, subject=HEAD,
        prior_items=tuple(items), state="blocking", raw_structured_coder_response=raw, **extra,
    )


def _kwargs(followup, items, config, **extra):
    return dict(
        kind="coder_followup", parsed=followup, agent="claude", prior_items=tuple(items),
        config=config, model_used=None, **extra,
    )


def _post(followup, items, config, metadata=None, fallback_old_head=None, fallback_diffstat=None, **render_extra):
    kwargs = _kwargs(followup, items, config, **render_extra)
    metadata = metadata or _metadata(items, raw=_raw(followup))
    full = render_public_agent_comment(**kwargs)
    result = bounded_coder_round_comment(
        full, metadata, pr_number=PR, head_sha=HEAD, render_kwargs=kwargs,
        evidence_heads=(HEAD, PRIOR, *OTHER_HEADS),
        fallback_old_head=fallback_old_head, fallback_diffstat=fallback_diffstat,
    )
    parts = [str(part) for part in prepare_round_comment(result.posted_body)]
    assert all(len(part) <= MAX_GITHUB_BODY_CHARS for part in parts)
    return full, result, parts


def _section(body, heading):
    start = body.index(heading)
    following = re.search(r"\n\n(?:###|##|<!--|--) ", body[start + len(heading):])
    end = start + len(heading) + following.start() if following else len(body)
    return body[start:end]


def _ids(text):
    return set(re.findall(r"\bitem-\d+\b", text))


def _hydrated(parts):
    (record,) = _extract_round_metadata_records([SimpleNamespace(body=part) for part in parts], flow="pr")
    return record


def _sixty_item_ledger():
    """About 60 items; many are repeated native human-only evidence requests."""
    items = [_code_item(number) for number in range(1, 21)]
    reviewers = ("Codex", "Anthropic Claude", "Google Antigravity")
    number = 21
    for head in OTHER_HEADS:  # each head's request repeats once per reviewer
        for reviewer in reviewers:
            items.append(_evidence_item(number, reviewer=reviewer, head=head))
            number += 1
    # Same request text repeated at different heads by one reviewer: one key.
    for head in (HEAD, PRIOR, *OTHER_HEADS):
        for _ in range(6):
            items.append(_evidence_item(number, reviewer="Codex", head=head))
            number += 1
    singleton, _ = _upsert_evidence_obligation(
        [], item_number=number, reviewer="Codex", text="Attach the signed onboarding walkthrough log.",
        source_round=2, current_head_sha=HEAD,
    )
    items.append(singleton[0])
    # A row-converted obligation sharing a repeated request's text is never grouped.
    items.append(replace(
        items[21], item_id=f"item-{number + 1}", converted_row_id="row-a", converted_matrix_identity="m" * 64
    ))
    return items


def test_fitting_comment_is_byte_identical(tmp_path):
    """Row coder-body-fits."""
    config = make_config(tmp_path)
    items = [_code_item(1), _code_item(2)]
    followup = _followup(["item-1"], ["item-2"], note="Done.")
    metadata = _metadata(items, raw=_raw(followup), architecture_impact_degradations=(
        ParseDegradation("architecture_impact.rationale", "type", "123", "degraded-to-undetermined"),
    ))
    full, result, _parts = _post(followup, items, config, metadata=metadata)
    assert result.level == 0
    assert result.posted_body == _attach_round_metadata(full, metadata)
    assert "shortened" not in result.posted_body
    assert "### Parse degradations" in result.posted_body  # full section, as today


def test_oversized_structured_body_posts_bounded_with_every_item_id(tmp_path):
    """Rows coder-body-oversized and coder-body-evidence-grouping."""
    config = make_config(tmp_path)
    items = _sixty_item_ledger()
    ids = [item.item_id for item in items]
    addressed, remaining = ids[:20], ids[20:]
    long_note = "Explained in depth. " * 120
    followup = _followup(addressed, remaining, summary="Round 17 summary. " * 40, note=long_note)
    full, result, parts = _post(followup, items, config)
    assert len(full) > MAX_GITHUB_BODY_CHARS
    assert result.level >= 1
    anchor = parts[-1]
    assert f"Head: `{HEAD}`" in anchor
    assert "Round 17 summary." in anchor
    addressed_section = _section(anchor, "### Addressed items")
    remaining_section = _section(anchor, "### Remaining items")
    assert _ids(addressed_section) == set(addressed)
    assert _ids(remaining_section) == set(remaining)
    assert _ids(addressed_section) | _ids(remaining_section) == set(ids)
    # The complete response and ledger hydrate from round metadata.
    record = _hydrated(parts)
    assert record.metadata.raw_structured_coder_response == _raw(followup)
    assert [item.item_id for item in record.metadata.prior_items] == ids


def test_evidence_grouping_collapses_only_repeated_native_requests(tmp_path):
    """Row coder-body-evidence-grouping at an explicit bounded level."""
    config = make_config(tmp_path)
    items = _sixty_item_ledger()
    ids = [item.item_id for item in items]
    followup = _followup(ids[:20], ids[20:])
    body = render_public_agent_comment(
        **_kwargs(followup, items, config), bound_level=2, head_sha=HEAD,
        evidence_heads=(HEAD, PRIOR, *OTHER_HEADS),
    )
    group_lines = [line for line in body.splitlines() if line.startswith("- Human-only exact-head evidence,")]
    assert group_lines, body
    grouped = set().union(*(_ids(line) for line in group_lines))
    # A deferred request's raw text is never published by a group line (#1068).
    assert "release smoke run" not in "\n".join(group_lines)
    native_repeated = {item.item_id for item in items[20:-2]}
    assert grouped == native_repeated
    # Each group lists every member and they share one request identity.
    assert all(len(_ids(line)) > 1 for line in group_lines)
    # Code findings, the singleton request and the row-converted obligation keep their own lines.
    for item_id in (*ids[:20], ids[-2], ids[-1]):
        assert re.search(rf"^- {item_id}: ", body, re.MULTILINE), item_id
    assert _ids(_section(body, "### Addressed items")) | _ids(_section(body, "### Remaining items")) == set(ids)
    # No requirement identifier is invented for a group.
    assert "hr-" not in "\n".join(group_lines)
    # Level 1 does not group.
    level_one = render_public_agent_comment(**_kwargs(followup, items, config), bound_level=1, head_sha=HEAD)
    assert "Human-only exact-head evidence, one request repeated" not in level_one


def _degradations(prefix, count=8):
    return tuple(
        ParseDegradation(f"{prefix}[{index}]" + "p" * 150, "rule-" + "r" * 190, "x" * 120, "claim-dropped")
        for index in range(count)
    )


def test_ancillary_and_attachment_sections_compact_with_complete_records(tmp_path):
    """Row coder-body-ancillary-overflow: tests, sub-items and degradation records."""
    from coding_review_agent_loop.protocol import ReviewSubItem as SubItem

    config = make_config(tmp_path)
    sub_items = tuple(SubItem(sub_item_id=f"item-1.{n}", text="sub " * 30, status="open") for n in range(1, 40))
    items = [_code_item(1), *(_code_item(n) for n in range(2, 4))]
    items[0] = UnresolvedReviewItem(**{**items[0].__dict__, "sub_items": sub_items})
    tests = tuple(f"python -m pytest tests/test_case_{n}.py::test_" + "long_name_" * 25 for n in range(400))
    followup = _followup(
        ["item-1", "item-2"], ["item-3"], tests_run=tests,
        out_of_checkout_tests_run=tuple(f"baseline run {n} " + "y" * 150 for n in range(60)),
        test_observation_degradations=_degradations("test_observations"),
    )
    metadata = _metadata(
        items, raw=_raw(followup),
        architecture_impact_degradations=_degradations("architecture_impact"),
        test_observation_degradations=_degradations("test_observations"),
        sub_item_degradations=_degradations("addressed_sub_items"),
    )
    _full, result, parts = _post(followup, items, config, metadata=metadata)
    assert result.level >= 3
    anchor = parts[-1]
    if result.level == 3:
        assert "Tests run: 400 command(s) (full list in round metadata)" in anchor
        assert "Sub-item progress: 1 item(s) with sub-items" in anchor
    assert COMPACT_DEGRADATIONS_HEADING in anchor
    assert "Architecture-impact parse degradations: 8 element(s) degraded" in anchor
    assert "### Parse degradations\n" not in anchor
    assert "### Test observation parse degradations" not in anchor
    assert "### Sub-item parse degradations" not in anchor
    assert _ids(anchor) >= {"item-1", "item-2", "item-3"}
    record = _hydrated(parts).metadata
    assert record.architecture_impact_degradations == metadata.architecture_impact_degradations
    assert record.test_observation_degradations == metadata.test_observation_degradations
    assert record.sub_item_degradations == metadata.sub_item_degradations


def test_degradations_with_a_tight_budget_use_compact_indicators(tmp_path):
    """Row coder-body-ancillary-overflow (b2): the attachment honours the bound level."""
    config = make_config(tmp_path)
    items = [_code_item(n, text=f"Finding {n}: " + "detail " * 30) for n in range(1, 151)]
    ids = [item.item_id for item in items]
    followup = _followup(ids, note="Fixed it carefully. " * 30)
    metadata = _metadata(
        items, raw=_raw(followup),
        architecture_impact_degradations=_degradations("architecture_impact"),
        test_observation_degradations=_degradations("test_observations"),
        sub_item_degradations=_degradations("addressed_sub_items"),
    )
    _full, result, parts = _post(followup, items, config, metadata=metadata)
    assert result.level >= 3
    anchor = parts[-1]
    assert anchor.count(COMPACT_DEGRADATIONS_HEADING) == 1
    for heading in ("### Parse degradations\n", "### Test observation parse degradations",
                    "### Sub-item parse degradations\n"):
        assert heading not in anchor
    assert _ids(_section(anchor, "### Addressed items")) == set(ids)
    assert _hydrated(parts).metadata.sub_item_degradations == metadata.sub_item_degradations


def _big_assessment():
    entries = []
    for number in range(24):
        kind = "gap" if number % 6 == 0 else "claim"
        entries.append(CoverageRowEntry(
            row_id=f"row-{number}", label=f"Row {number}", required_level="unit",
            required_level_text="unit", kind=kind,
            test_references=tuple(
                f"tests/test_module_{number}.py::test_" + "very_long_reference_" * 12 + str(ref)
                for ref in range(12)
            ) if kind == "claim" else (),
            declared_level="unit" if kind == "claim" else None,
            gap_reason="reason " * 200 if kind == "gap" else "",
            gap_correction="correction " * 200 if kind == "gap" else "",
        ))
    return CoverageAssessment(
        revision=HEAD, deficiencies=(CoverageDeficiency("row-0", "declared-gap"),), entries=tuple(entries)
    )


def test_oversized_coverage_map_compacts_and_resume_extracts_it(tmp_path):
    """Row coder-body-resume-reads."""
    config = make_config(tmp_path)
    assessment = _big_assessment()
    assert len(render_coverage_map(assessment)) > MAX_GITHUB_BODY_CHARS
    items = [_code_item(1)]
    followup = _followup(["item-1"], note="Done.")
    _full, result, parts = _post(followup, items, config, coverage_assessment=assessment)
    assert result.level >= 3
    record = _hydrated(parts)
    section = extract_coverage_map_section(record.body, expected_revision=HEAD)
    assert section is not None
    assert section.startswith("### Risk-matrix coverage map")
    assert f"- **Status:** incomplete at {HEAD}" in section
    assert "row-0" in section
    # Resume at the same head reads the bounded record as the coder record of
    # this head (not an unrecorded advance) and hydrates the full response.
    resumed = _resume_pr_round(
        [SimpleNamespace(body=part) for part in parts], head_sha=HEAD, configured_reviewers=("codex",)
    )
    assert resumed is not None and resumed.unrecorded_head_advance is False
    assert resumed.coder_output == _raw(followup)
    assert extract_coverage_map_section(resumed.coder_comment_body, expected_revision=HEAD) == section


@pytest.mark.parametrize("compact", ["compact", "minimal"])
def test_compact_coverage_map_forms_keep_the_extraction_contract(compact):
    assessment = _big_assessment()
    rendered = render_coverage_map(assessment, compact=compact)
    assert len(rendered) < 10_000
    body = "\n\n".join(["## Coder follow-up", "### Addressed items\n- item-1", rendered, "-- Anthropic Claude"])
    section = extract_coverage_map_section(body, expected_revision=HEAD)
    assert section == rendered
    assert rendered.splitlines()[-1] == render_coverage_map(assessment).splitlines()[-1]


def test_full_coverage_map_render_is_unchanged():
    assessment = _big_assessment()
    assert render_coverage_map(assessment) == render_coverage_map(assessment, compact="full")


def _signed(text):
    return text + "\n-- Anthropic Claude: claude-opus-5-5 (medium)"


def test_oversized_freeform_output_keeps_whole_lines_and_the_full_text(tmp_path):
    """Row coder-body-freeform."""
    original = "\n".join(f"Line {n}: " + "free-form coder narration " * 3 for n in range(2000))
    full_body = _signed(original)
    metadata = _metadata([_code_item(1)])
    result = bounded_coder_round_comment(
        full_body, metadata, pr_number=PR, head_sha=HEAD, freeform_text=original,
    )
    parts = [str(part) for part in prepare_round_comment(result.posted_body)]
    assert all(len(part) <= MAX_GITHUB_BODY_CHARS for part in parts)
    anchor = parts[-1]
    assert FREEFORM_OMISSION_NOTE in anchor
    assert anchor.rstrip().endswith("-- Anthropic Claude: claude-opus-5-5 (medium)")
    visible_lines = [line for line in anchor.splitlines() if line.startswith("Line ")]
    expected = [line.rstrip() for line in original.splitlines()[: len(visible_lines)]]
    assert visible_lines and [line.rstrip() for line in visible_lines] == expected
    assert _hydrated(parts).metadata.raw_structured_coder_response == original
    # Resume after a further push yields the original text as the coder output.
    resumed = _resume_pr_round(
        [SimpleNamespace(body=part) for part in parts], head_sha="9" * 40, configured_reviewers=("codex",)
    )
    assert resumed is not None and resumed.coder_output == original


def test_single_oversized_freeform_line_is_cut_at_a_character_boundary():
    """Row coder-body-freeform-long-line."""
    original = "é" * 30_000 + "z" * 80_000
    result = bounded_coder_round_comment(
        _signed(original), _metadata([_code_item(1)]), pr_number=PR, head_sha=HEAD,
        freeform_text=original,
    )
    parts = [str(part) for part in prepare_round_comment(result.posted_body)]
    assert all(len(part) <= MAX_GITHUB_BODY_CHARS for part in parts)
    assert "…" in parts[-1] and FREEFORM_OMISSION_NOTE in parts[-1]
    assert _hydrated(parts).metadata.raw_structured_coder_response == original


def test_fitting_freeform_output_is_unchanged():
    original = "Short free-form follow-up."
    metadata = _metadata([_code_item(1)])
    result = bounded_coder_round_comment(
        _signed(original), metadata, pr_number=PR, head_sha=HEAD, freeform_text=original,
    )
    assert result.level == 0 and result.metadata is metadata
    assert result.posted_body == _attach_round_metadata(_signed(original), metadata)


def _fallback_resume(parts):
    return _resume_pr_round(
        [SimpleNamespace(body=part) for part in parts], head_sha=HEAD, configured_reviewers=("codex",)
    )


def test_oversize_at_every_bound_level_posts_the_synthesized_fallback(tmp_path):
    """Row oversize-never-terminal: no bound level fits; the record still posts.

    The caller adopts a claim-free synthesized record; the transport overflow
    error never escapes and no recovery command is named.
    """
    config = make_config(tmp_path)
    ids = [f"item-{n}" for n in range(1, 9001)]
    followup = _followup(ids)
    old = "a" * 40
    _full, result, parts = _post(
        followup, (), config, metadata=_metadata(()), fallback_old_head=old,
        fallback_diffstat=lambda new: f" src/a.py | 2 +-\n 1 file changed ({new[:7]})",
    )
    assert result.level == FALLBACK_LEVEL
    assert "--review-unrecorded-head" not in result.posted_body
    assert result.coder_output is not None
    assert "orchestrator synthesized this coder record from git" in result.coder_output
    assert f"`{old}`" in result.coder_output and f"`{HEAD}`" in result.coder_output
    assert "src/a.py" in result.coder_output
    # Claim-free: none of the rejected response's structured claims survive.
    assert _ids(result.coder_output) == set()
    assert result.metadata.coder_record_source == "synthesized"
    assert result.metadata.raw_structured_coder_response is None
    assert FALLBACK_NOTE in result.metadata.coder_record_degradations
    record = _hydrated(parts)
    assert record.metadata.coder_record_source == "synthesized"
    assert "## Orchestrator-synthesized coder record" in record.body
    assert "### Orchestrator degradation notes" in record.body
    resumed = _fallback_resume(parts)
    assert resumed is not None and resumed.unrecorded_head_advance is False
    assert resumed.coder_metadata.coder_record_source == "synthesized"


def _summaries(count=400):
    """Large, poorly compressible, non-spillable residual metadata."""
    import hashlib

    return tuple(
        f"Round {n}: " + "".join(hashlib.sha256(f"{n}-{k}".encode()).hexdigest() for k in range(8))
        for n in range(count)
    )


def test_fallback_omits_residual_metadata_and_keeps_the_ledger(tmp_path):
    """Row fallback-residual-metadata-overflow, including resume."""
    config = make_config(tmp_path)
    items = [_code_item(n) for n in range(1, 4)]
    followup = _followup(["item-1"], ["item-2", "item-3"])
    metadata = _metadata(items, raw=_raw(followup), compact_prior_summaries=_summaries())
    _full, result, parts = _post(followup, items, config, metadata=metadata)
    assert result.level == FALLBACK_LEVEL
    assert result.metadata.compact_prior_summaries == ()
    assert result.metadata.raw_structured_coder_response is None
    assert tuple(result.metadata.prior_items) == tuple(items)  # never dropped or truncated
    record = _hydrated(parts)
    assert [item.item_id for item in record.metadata.prior_items] == ["item-1", "item-2", "item-3"]
    resumed = _fallback_resume(parts)
    assert resumed is not None and resumed.unrecorded_head_advance is False
    assert [item.item_id for item in resumed.prior_items] == ["item-1", "item-2", "item-3"]
    assert all(item.status == "blocking" for item in resumed.prior_items)
    assert "Not claimed" in resumed.coder_output


def _noise(label, count):
    import hashlib

    return "".join(hashlib.sha256(f"{label}-{n}".encode()).hexdigest() for n in range(count))


def _residual(case):
    """Large, poorly compressible residual metadata that never spills."""
    if case == "architecture-impact":
        from coding_review_agent_loop.architecture_contract import sanitize_architecture_impact

        return {"architecture_impact": sanitize_architecture_impact(ArchitectureImpact(
            status="changed", rationale=_noise("rationale", 1200),
            affected_components=tuple(_noise(f"component-{n}", 20) for n in range(40)),
            dependencies=("d",), execution_data_flows=("f",), persistence=("p",),
            public_contracts=("c",), security_boundaries=("s",),
            canonical_document_action="update",
            canonical_document_path="docs/architecture/round-state-and-resume.md",
            canonical_document_rationale="r",
        )), "architecture_contract_version": 1}
    if case == "compact-summaries":
        return {"compact_prior_summaries": _summaries()}
    # Outside the first-tier cleared set: only the minimal tier drops it.
    return {"surfaced_reviewer_requirement_ids": tuple(
        "hr-" + _noise(f"hr-{n}", 1) for n in range(1500)
    )}


@pytest.mark.parametrize("case", ["architecture-impact", "compact-summaries", "minimal-tier"])
def test_fallback_always_returns_a_verified_postable_record_and_resumes(tmp_path, case):
    """Rows oversize-never-terminal and fallback-residual-metadata-overflow (#2086 item-1)."""
    config = make_config(tmp_path)
    items = [_code_item(n) for n in range(1, 4)]
    followup = _followup(["item-1"], ["item-2", "item-3"])
    metadata = _metadata(items, raw=_raw(followup), **_residual(case))
    _full, result, parts = _post(followup, items, config, metadata=metadata)  # asserts every part fits
    assert result.level == FALLBACK_LEVEL
    assert result.metadata.architecture_impact is None
    assert result.metadata.compact_prior_summaries == ()
    assert tuple(result.metadata.prior_items) == tuple(items)
    assert result.metadata.surfaced_reviewer_requirement_ids == ()  # tier 1 or minimal tier
    if case == "minimal-tier":
        # The first tier keeps this field; only the minimal tier makes it fit.
        assert metadata.surfaced_reviewer_requirement_ids
    record = _hydrated(parts)
    assert record.metadata.coder_record_source == "synthesized"
    assert record.metadata.architecture_impact is None
    resumed = _fallback_resume(parts)
    assert resumed is not None and resumed.unrecorded_head_advance is False
    assert [item.item_id for item in resumed.prior_items] == ["item-1", "item-2", "item-3"]


def test_fallback_rolls_back_dispute_annotations_of_the_discarded_response(tmp_path):
    """#2086 item-2: the fallback ledger carries no claim of the discarded response."""
    from coding_review_agent_loop.coder_followup_degradation import rollback_discarded_annotations
    from coding_review_agent_loop.unresolved_items import _apply_dispute_evidence, _is_disputed_item

    config = make_config(tmp_path)
    earlier = replace(_code_item(2), notes=("Coder disputes this item: an earlier round's evidence.",))
    pre_turn = [_code_item(1), earlier]
    annotated = _apply_dispute_evidence(
        pre_turn, disputed_items=["item-1", "item-2"], dispute_evidence={"item-1": "x", "item-2": "y"},
    )
    assert _is_disputed_item(annotated[0])
    followup = _followup((), disputed_items=("item-1", "item-2"), dispute_evidence={"item-1": "x", "item-2": "y"})
    metadata = _metadata(annotated, raw=_raw(followup), compact_prior_summaries=_summaries())
    kwargs = _kwargs(followup, annotated, config)
    result = bounded_coder_round_comment(
        render_public_agent_comment(**kwargs), metadata, pr_number=PR, head_sha=HEAD,
        render_kwargs=kwargs,
        fallback_ledger=lambda ledger: rollback_discarded_annotations(ledger, pre_turn),
    )
    assert result.level == FALLBACK_LEVEL
    by_id = {item.item_id: item for item in result.metadata.prior_items}
    assert not _is_disputed_item(by_id["item-1"])  # the discarded dispute is gone
    assert by_id["item-2"].notes == earlier.notes  # pre-turn state is kept
    resumed = _fallback_resume([str(part) for part in prepare_round_comment(result.posted_body)])
    assert not _is_disputed_item({i.item_id: i for i in resumed.prior_items}["item-1"])


def test_fallback_caps_an_oversized_id_listing_and_keeps_every_item_open(tmp_path):
    """Row fallback-oversized-ledger: the ID-only listing alone exceeds the limit."""
    config = make_config(tmp_path)
    items = [
        UnresolvedReviewItem(
            item_id=f"item-{n}", reviewer="Codex", source_round=1, text="Short.", status="blocking"
        )
        for n in range(1, 9001)
    ]
    assert len(", ".join(f"`{item.item_id}`" for item in items)) > MAX_GITHUB_BODY_CHARS
    followup = _followup([item.item_id for item in items])
    metadata = _metadata(items, raw=_raw(followup), compact_prior_summaries=_summaries())
    _full, result, parts = _post(followup, items, config, metadata=metadata)
    assert result.level == FALLBACK_LEVEL
    assert "more item(s) omitted from this listing; all remain open" in result.coder_output
    resumed = _fallback_resume(parts)
    assert resumed is not None
    assert len(resumed.prior_items) == 9000
    assert {item.status for item in resumed.prior_items} == {"blocking"}


def test_many_distinct_repeated_evidence_groups_still_fit_id_only(tmp_path):
    """Item-1: a group line never costs more than its member IDs at levels 3-4."""
    config = make_config(tmp_path)
    items = []
    for request in range(300):
        for offset, reviewer in enumerate(("Codex", "Anthropic Claude")):
            ledger, _ = _upsert_evidence_obligation(
                [], item_number=2 * request + offset + 1, reviewer=reviewer,
                text=f"Attach a human-only exact-head capture of release scenario {request} "
                     + "with the full operator console transcript and timing table. " * 2,
                source_round=2, current_head_sha=HEAD,
            )
            items.append(ledger[0])
    ids = [item.item_id for item in items]
    followup = _followup((), ids)
    _full, result, parts = _post(followup, items, config)
    assert result.level >= 3
    remaining = _section(parts[-1], "### Remaining items")
    assert _ids(remaining) == set(ids)
    group_lines = [line for line in remaining.splitlines() if "," in line]
    assert len(group_lines) == 300
    assert all(re.fullmatch(r"- item-\d+, item-\d+", line) for line in group_lines)


def test_grouping_needs_the_admitted_historical_heads(tmp_path):
    """Item-2: requests quoting older PR heads collapse only when those heads are known."""
    config = make_config(tmp_path)
    items = [
        _evidence_item(number, reviewer=reviewer, head=head)
        for number, (reviewer, head) in enumerate(
            [("Codex", OTHER_HEADS[0]), ("Anthropic Claude", OTHER_HEADS[1]), ("Codex", OTHER_HEADS[2])], 1
        )
    ]
    assert all(item.candidate_head_sha is None for item in items)
    followup = _followup((), [item.item_id for item in items])
    kwargs = _kwargs(followup, items, config)
    only_current = render_public_agent_comment(**kwargs, bound_level=2, head_sha=HEAD, evidence_heads=(HEAD,))
    assert "one request repeated" not in only_current
    admitted = render_public_agent_comment(
        **kwargs, bound_level=2, head_sha=HEAD, evidence_heads=(HEAD, *OTHER_HEADS)
    )
    assert "one request repeated: item-1, item-2, item-3" in admitted


def test_evidence_heads_are_resolved_lazily_only_when_grouping(tmp_path):
    config = make_config(tmp_path)
    calls = []
    items = [_code_item(1)]
    followup = _followup(["item-1"], note="Done.")
    kwargs = _kwargs(followup, items, config)
    bounded_coder_round_comment(
        render_public_agent_comment(**kwargs), _metadata(items, raw=_raw(followup)), pr_number=PR,
        head_sha=HEAD, render_kwargs=kwargs, evidence_heads=lambda: calls.append(1) or (HEAD,),
    )
    assert calls == []  # a fitting comment never reads history


def test_freeform_trailing_long_comment_is_bounded_as_content():
    """Item-3: only the short footer is immutable; a huge trailing comment is content."""
    original = "Short narration.\n<!-- " + "n" * 70_000 + " -->"
    full_body = _signed(original)
    result = bounded_coder_round_comment(
        full_body, _metadata([_code_item(1)]), pr_number=PR, head_sha=HEAD, freeform_text=original,
    )
    parts = [str(part) for part in prepare_round_comment(result.posted_body)]
    assert all(len(part) <= MAX_GITHUB_BODY_CHARS for part in parts)
    anchor = parts[-1]
    assert "Short narration." in anchor and "n" * 1000 not in anchor
    assert anchor.rstrip().endswith("-- Anthropic Claude: claude-opus-5-5 (medium)")
    assert _hydrated(parts).metadata.raw_structured_coder_response == original


@pytest.mark.parametrize(
    "tail",
    [
        "<!-- " + "n" * 70_000 + " -->",  # one long trailing comment
        "\n".join(f"<!-- note {n}: " + "c" * 180 + " -->" for n in range(400)),  # many short comments
        "\n".join(f"-- Signature-shaped line {n} " + "s" * 150 for n in range(400)),  # many signature-shaped lines
    ],
    ids=["one-long-comment", "many-short-comments", "many-signature-shaped-lines"],
)
def test_freeform_trailing_lines_of_any_shape_are_budgeted_content(tail):
    """Item-3 (generalized): only the final signature and one state marker are immutable."""
    original = "Short narration.\n" + tail + "\n<!-- AGENT_STATE: blocking -->"
    result = bounded_coder_round_comment(
        _signed(original), _metadata([_code_item(1)]), pr_number=PR, head_sha=HEAD, freeform_text=original,
    )
    parts = [str(part) for part in prepare_round_comment(result.posted_body)]
    assert all(len(part) <= MAX_GITHUB_BODY_CHARS for part in parts)
    anchor = parts[-1]
    assert "Short narration." in anchor and FREEFORM_OMISSION_NOTE in anchor
    assert anchor.rstrip().endswith("<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude: claude-opus-5-5 (medium)")
    assert _hydrated(parts).metadata.raw_structured_coder_response == original
