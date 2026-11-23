"""Unit tests for ``--pr-human-evidence`` (#2063): config, records, ledger and parsing."""

import ast
import dataclasses
import json
from pathlib import Path

import pytest

from coding_review_agent_loop.cli import build_parser
from coding_review_agent_loop.config import config_from_args
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.human_evidence_mode import (
    HUMAN_CHECKS_FLOW,
    HumanCheck,
    HumanEvidenceRun,
    decode_human_check_entries,
    normalize_human_evidence_ledger,
    restore_checks_as_obligations,
)
from coding_review_agent_loop.managed_ci_recovery_commands import _find_recovery_positional_index
from coding_review_agent_loop.protocol import (
    EVIDENCE_OBLIGATION_KIND,
    MACHINE_AUTHORITY,
    UnresolvedReviewItem,
    parse_pr_review,
    parse_structured_pr_review,
)
from coding_review_agent_loop.round_metadata import (
    EvidenceReleaseRecord,
    PostedRoundMetadata,
    _attach_round_metadata,
    _decode_round_metadata,
    _encode_round_metadata,
    reset_human_checks_stamp,
    set_human_checks_stamp,
)
from coding_review_agent_loop.unresolved_items import (
    _upsert_evidence_obligation,
    converted_evidence_obligation_identity,
)
from agent_loop_helpers import FakeRunner, make_config
from pr_loop_test_helpers import _association, _classified_review

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "src" / "coding_review_agent_loop"
HEAD = "a" * 40
OTHER = "b" * 40


def _native(text="Run the live CLI suite at this head.", reviewer="Codex", number=1):
    items, _ = _upsert_evidence_obligation(
        [], item_number=number, reviewer=reviewer, text=text, source_round=1, current_head_sha=HEAD)
    return items[0]


def _converted(row="row-a", owners=("Codex", "Claude"), number=7):
    return UnresolvedReviewItem(
        item_id=f"item-{number}", reviewer="Orchestrator", source_round=2,
        text=f"Converted row {row}: supply live evidence.", status="blocking", source_status="blocking",
        resolution_owners=owners, owner_states=((owners[0], "resolved"), (owners[1], "pending")),
        authority=MACHINE_AUTHORITY, obligation_kind=EVIDENCE_OBLIGATION_KIND,
        lifecycle="evidence_frozen", candidate_head_sha=HEAD,
        obligation_identity=converted_evidence_obligation_identity(row, "e" * 64),
        converted_row_id=row, converted_matrix_identity="e" * 64,
    )


def _screenshot(number=9):
    from coding_review_agent_loop import review_images_coverage as coverage

    request = {"request_id": "mobile", "kind": "screenshot", "description": "Show the mobile landing page"}
    items, _ = coverage.upsert([], repo="OWNER/REPO", pr=77, requester="Codex", request=request,
                               contract="c" * 64, source=100, number=number, round_number=1, head=HEAD)
    return items[0]


def _code(number=3):
    return UnresolvedReviewItem(item_id=f"item-{number}", reviewer="Codex", source_round=1,
                                text="Fix the retry bound.", status="blocking")


# --- cli-default -------------------------------------------------------------------------


def _parsed(tmp_path, *extra, command=("pr", "77")):
    argv = [*command, "--repo", "OWNER/REPO", "--reviewer", "codex",
            "--codex-dir", str(tmp_path / "codex"), "--dangerous-agent-permissions", *extra]
    return config_from_args(build_parser().parse_args(argv), FakeRunner())


def test_cli_default_is_advisory_and_each_value_resolves(tmp_path):
    """Row cli-default."""
    assert _parsed(tmp_path).pr_human_evidence == "advisory"
    for mode in ("freeze", "advisory", "off"):
        assert _parsed(tmp_path, "--pr-human-evidence", mode).pr_human_evidence == mode
    # The issue entry point carries the same option into its PR loop.
    assert _parsed(tmp_path, "--pr-human-evidence", "off", command=("issue", "12")).pr_human_evidence == "off"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["pr", "77", "--pr-human-evidence", "maybe"])


def test_invalid_mode_errors_through_the_enum_check(tmp_path):
    with pytest.raises(AgentLoopError, match="--pr-human-evidence must be one of: freeze, advisory, off"):
        make_config(tmp_path, pr_human_evidence="maybe")
    for mode in ("freeze", "advisory", "off"):
        assert make_config(tmp_path, pr_human_evidence=mode).pr_human_evidence == mode


def test_operator_conversion_requires_freeze(tmp_path):
    """Row advisory-stall-conversion-bypass (operator conversion at config validation)."""
    with pytest.raises(AgentLoopError, match="requires --pr-human-evidence freeze"):
        _parsed(tmp_path, "--pr-convert-evidence-rows", "row-a")
    assert _parsed(tmp_path, "--pr-convert-evidence-rows", "row-a",
                   "--pr-human-evidence", "freeze").pr_convert_evidence_rows == ("row-a",)


def test_recovery_parsing_consumes_the_mode_value_before_the_positional():
    argv = ["agent-loop", "pr", "--pr-human-evidence", "freeze", "77"]
    assert _find_recovery_positional_index(argv, 1) == 4


# --- metadata-roundtrip --------------------------------------------------------------------


def _check(item, head=HEAD):
    return HumanCheck(item.reviewer, item.text, item.source_round, head,
                      "converted" if item.converted_row_id else "native", item)


def test_historical_records_are_byte_stable_and_new_fields_roundtrip():
    """Row metadata-roundtrip."""
    legacy = PostedRoundMetadata(flow="pr", role="summary", agent="Orchestrator", round_number=1, subject=HEAD)
    encoded = _encode_round_metadata(legacy)
    assert "human_checks" not in encoded
    assert _encode_round_metadata(_decode_round_metadata(encoded)) == encoded
    checks = tuple(_check(item).as_dict() for item in (_native(), _converted()))
    release = EvidenceReleaseRecord(HEAD, "advisory-mode", ("r1",), 3, False, False)
    new = dataclasses.replace(legacy, human_checks=checks, phase="evidence-release", evidence_release=release)
    decoded = _decode_round_metadata(_encode_round_metadata(new))
    assert decoded.human_checks == checks and decoded.evidence_release.valid
    assert decoded.evidence_release.reason == "advisory-mode"
    restored = [HumanCheck.from_mapping(entry) for entry in decoded.human_checks]
    assert restored[1].origin_item == _converted()
    assert _encode_round_metadata(decoded) == _encode_round_metadata(new)


def test_malformed_human_checks_entries_are_dropped_without_failing_the_record():
    good = _check(_native()).as_dict()
    bad = [{"reviewer": "Codex"}, "text", {**good, "kind": "other"}, {**good, "origin_item": 3}]
    assert decode_human_check_entries([good, *bad]) == (good,)
    assert decode_human_check_entries("nope") == ()


def test_stamp_applies_only_to_orchestrator_pr_records_while_advisory_list_is_non_empty(tmp_path):
    run = HumanEvidenceRun(make_config(tmp_path, pr_human_evidence="advisory"), 77)
    run.checks = (_check(_native()),)
    token = set_human_checks_stamp(run)
    try:
        def decoded(**fields):
            body = _attach_round_metadata("body", PostedRoundMetadata(round_number=1, subject=HEAD, **fields))
            return _decode_round_metadata(body.split("AGENT_LOOP_META: ", 1)[1].split(" -->")[0])

        assert len(decoded(flow="pr", role="summary", agent="Orchestrator").human_checks) == 1
        assert decoded(flow="pr", role="reviewer", agent="Codex").human_checks == ()
        assert decoded(flow="plan", role="summary", agent="Orchestrator").human_checks == ()
        run.mode = "freeze"
        assert decoded(flow="pr", role="summary", agent="Orchestrator").human_checks == ()
    finally:
        reset_human_checks_stamp(token)


# --- the normalization choke point ---------------------------------------------------------


def test_advisory_moves_native_and_converted_items_and_keeps_screenshots_and_code():
    """Rows advisory-stall-conversion-bypass (historical converted rows) and screenshot-channel-preserved."""
    native, converted, screenshot, code = _native(), _converted(), _screenshot(), _code()
    result = normalize_human_evidence_ledger([native, converted, screenshot, code], (), mode="advisory", head_sha=OTHER)
    assert result.items == [screenshot, code]
    assert [(c.kind, c.origin_item) for c in result.checks] == [("native", native), ("converted", converted)]
    assert result.checks[0].head_sha == OTHER and result.checks[1].head_sha == HEAD
    again = normalize_human_evidence_ledger([native], result.checks, mode="advisory")
    assert again.checks == result.checks and again.added == 0


def test_off_drops_and_freeze_is_the_identity():
    items = [_native(), _converted(), _code()]
    off = normalize_human_evidence_ledger(items, (), mode="off")
    assert off.items == [_code()] and off.dropped == 2 and off.checks == ()
    freeze = normalize_human_evidence_ledger(items, (), mode="freeze")
    assert freeze.items == items and freeze.dropped == 0


def test_restore_keeps_converted_provenance_once_and_skips_cleared_identities():
    """Row advisory-to-freeze-carried (converted row provenance and dedupe)."""
    native, converted = _native(), _converted()
    checks = (_check(native), _check(converted))
    restored = restore_checks_as_obligations([_code()], checks)
    row = next(item for item in restored if item.converted_row_id)
    assert row == dataclasses.replace(converted, lifecycle="evidence_deferred", candidate_head_sha=None)
    assert row.resolution_owners == ("Codex", "Claude") and row.owner_states == converted.owner_states
    # Already restored by historical reconciliation: never duplicated.
    assert restore_checks_as_obligations(restored, checks) == restored
    reconciled = dataclasses.replace(converted, item_id="item-99")
    assert sum(bool(i.converted_row_id) for i in restore_checks_as_obligations([reconciled], checks)) == 1
    cleared = restore_checks_as_obligations([], checks, clearances=((native.obligation_identity, HEAD),), head_sha=HEAD)
    assert [item.item_id for item in cleared] == [converted.item_id]


def test_freeze_run_restores_the_seeded_list_once(tmp_path):
    from types import SimpleNamespace

    run = HumanEvidenceRun(make_config(tmp_path, pr_human_evidence="freeze"), 77)
    record = SimpleNamespace(index=4, metadata=SimpleNamespace(
        agent="Orchestrator", subject=HEAD, human_checks=(_check(_native()).as_dict(),)))
    run.seed([record], [])
    assert run.restore_pending
    first = run.normalize([], round_number=2, head_sha=HEAD)
    assert [item.text for item in first] == [_native().text]
    assert run.normalize(first, round_number=2, head_sha=HEAD) == first and not run.checks


# --- parse-level association bypass and the all-call-sites guard ---------------------------


def _inconsistent_review():
    return _classified_review([_association("R1", "human-only", request="never emitted")])


def _malformed_review():
    text = _classified_review([])
    raw, footer = text.split("\n<!--", 1)
    payload = json.loads(raw)
    payload["requirement_evidence"] = [{"requirement_id": "R1"}]
    return json.dumps(payload) + "\n<!--" + footer


def test_parse_flag_ignores_requirement_evidence_outside_freeze():
    """Row advisory-signed-classification-bypass (protocol unit)."""
    with pytest.raises(AgentLoopError):
        parse_structured_pr_review(_malformed_review(), reviewer="Codex")
    for text in (_malformed_review(), _inconsistent_review()):
        parsed = parse_structured_pr_review(text, reviewer="Codex", parse_requirement_evidence=False)
        assert parsed.requirement_evidence == ()
        assert parse_pr_review(text, reviewer="Codex", parse_requirement_evidence=False).requirement_evidence == ()
    assert parse_structured_pr_review(_inconsistent_review(), reviewer="Codex").requirement_evidence


def test_publication_dispositions_reparse_honours_the_flag():
    from coding_review_agent_loop.native_evidence import publication_dispositions

    text = _malformed_review()
    parsed = parse_structured_pr_review(text, reviewer="Codex", parse_requirement_evidence=False)
    assert publication_dispositions(parsed, text, "Codex", (), parse_requirement_evidence=False) == ()
    with pytest.raises(AgentLoopError):
        publication_dispositions(parsed, text, "Codex", ())


_PARSE_ENTRY_POINTS = {"parse_structured_pr_review", "parse_pr_review", "_validate_review_response"}


def test_every_reviewer_response_parse_site_passes_the_flag_explicitly():
    """Row advisory-signed-classification-bypass (AST guard over every parse site)."""
    missing = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
            if name not in _PARSE_ENTRY_POINTS:
                continue
            if any(keyword.arg == "parse_requirement_evidence" for keyword in node.keywords):
                continue
            missing.append(f"{path.relative_to(ROOT)}:{node.lineno} {name}")
    assert missing == []


def test_human_checks_flow_is_not_a_pr_round_flow():
    assert HUMAN_CHECKS_FLOW != "pr"
