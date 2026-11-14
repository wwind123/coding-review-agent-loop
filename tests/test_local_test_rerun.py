"""Issue #1999: exact ``run-tests --rerun`` replay and supersession rejection reasons.

Covers the private rerun catalog and broker replay, the local coder-prompt
block, closed-vocabulary rejection reasons, capacity-safe transport of the
supersession diagnostics, rerun citations and public redaction.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import socket
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import coding_review_agent_loop.local_test_evidence as evidence_module
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.local_test_evidence import (
    MAX_ROUND_BYTES,
    MAX_SUPERSESSION_CAVEAT_BYTES,
    MAX_SUPERSESSION_CAVEATS,
    REASON_COMMAND_DIFFERS,
    REASON_FIXTURE_DIFFERS,
    REASON_OUT_OF_CHECKOUT,
    REASON_TREE_CHANGED,
    EnvironmentIdentityRegistry,
    EvidenceScope,
    FixtureRecord,
    LocalTestEvidence,
    LocalTestObservation,
    TestBrokerClient as BrokerClient,
    TestBrokerServer as BrokerServer,
    TrackedTreeSnapshot,
    TreeAttribution,
    _can_supersede,
    _can_supersede_across_restart,
    _CrossRestartContext,
    bounded_evidence_for_round,
    canonicalize_bounded_evidence,
    classify_supersession_diagnostic,
    decode_bounded_evidence,
    is_supersession_reason,
    observation_from_mapping,
    reconcile_test_observations,
    redact_test_command,
    stable_tracked_tree_snapshot,
    superseded_by_current_turn,
    unsuperseded_receipts_sentence,
)
from coding_review_agent_loop.local_test_rerun import (
    IDENTICAL_RERUN_RULE,
    NO_RERUN_HANDLE_REASON,
    RERUN_CATALOG_UNAVAILABLE,
    RERUN_RECEIPT_UNKNOWN,
    RerunCatalog,
    unsuperseded_failure_rerun_block,
)
from coding_review_agent_loop.test_runtime import parse_rerun_citation

# A ``-k`` value long enough that the public redactor hashes it.
RAW_K = "durable_retry_backoff_private_selection_and_not_slow"
TEMP_NAME = "RERUN_TEMP_ASSIGNMENT_1999"
TEMP_VALUE = "temporary-assignment-value-1999"
PREFIX = ("/opt/agent-loop/bin/agent-loop", "run-tests")


def _obs(
    receipt: str,
    outcome: str,
    minute: int,
    *,
    registry: EnvironmentIdentityRegistry,
    command: tuple[str, ...] = ("python", "-m", "pytest", "tests/test_x.py", "-q"),
    environment: dict[str, str] | None = None,
    digest: str = "tree-a",
    state: str = "current-head",
    fixture: FixtureRecord | None = None,
    rerun_of: str | None = None,
    caveats: tuple[str, ...] = (),
    provenance: str = "parent-observed",
) -> LocalTestObservation:
    from coding_review_agent_loop.test_runtime import normalize_test_command

    return LocalTestObservation(
        command=command,
        outcome=outcome,
        provenance=provenance,
        scope=EvidenceScope("suite", ("tests",)),
        receipt_id=receipt,
        turn_id="turn-1999",
        timestamp=f"2026-11-14T10:{minute:02d}:00+00:00",
        cwd="/checkout",
        normalized_command=normalize_test_command(command, cwd=Path("/checkout")),
        returncode=0 if outcome == "passed" else 1,
        attribution=TreeAttribution(
            state=state, head="head-a", pre_digest="p", post_digest="p",
            tracked_digest=digest, stable=True,
        ),
        environment_state="not-compared",
        environment_identity=registry.capture(environment or {"PATH": "/usr/bin"}),
        fixture=fixture,
        rerun_of=rerun_of,
        caveats=caveats,
    )


def _cmd(index: int) -> tuple[str, ...]:
    return ("python", "-m", "pytest", f"tests/test_selection_{index}.py", "-q")


def _diagnostic_texts(evidence: LocalTestEvidence) -> list[str]:
    return [text for _failure, _kind, text in evidence.supersession_diagnostics]


# --- Rejection reasons ---------------------------------------------------------


def test_every_rejection_branch_returns_a_closed_vocabulary_reason():
    registry = EnvironmentIdentityRegistry()
    failure = _obs("f", "failed", 0, registry=registry)
    passing = _obs("p", "passed", 1, registry=registry)
    record_a = FixtureRecord(path="fixture.py", digest="100755:" + "a" * 40)
    record_b = FixtureRecord(path="fixture.py", digest="100755:" + "b" * 40)
    unstable = replace(failure, attribution=replace(failure.attribution, stable=False))
    snapshot = TrackedTreeSnapshot(
        root="/checkout", head="head-a", digest="d", tracked_digest="tree-a",
        status_clean=True, complete=True, stable=True,
    )
    context = _CrossRestartContext(
        current_head="head-a", current_snapshot=snapshot, snapshot_head_matches=True,
        refreshed_restored=frozenset({0}), entry_heads=("head-a", "head-a"),
    )
    cases = {
        "pair": (passing, passing, None),
        "pass-telemetry": (failure, replace(passing, provenance="telemetry-unverified"), None),
        "command": (failure, _obs("p", "passed", 1, registry=registry, command=_cmd(9)), None),
        "fixture": (
            _obs("f", "failed", 0, registry=registry, fixture=record_a),
            _obs("p", "passed", 1, registry=registry, fixture=record_b),
            None,
        ),
        "scope": (failure, replace(passing, scope=EvidenceScope("subset", ("tests",))), None),
        "attribution": (failure, replace(passing, attribution=replace(passing.attribution, state="stale")), None),
        "failure-not-current": (
            replace(failure, attribution=replace(failure.attribution, state="stale")),
            replace(passing, attribution=replace(passing.attribution, state="stale")),
            None,
        ),
        "snapshot": (failure, replace(passing, attribution=replace(passing.attribution, stable=None)), None),
        "tree": (failure, replace(passing, attribution=replace(passing.attribution, tracked_digest="tree-b")), None),
        "environment-different": (
            failure, _obs("p", "passed", 1, registry=registry, environment={"PATH": "/x"}), None,
        ),
        "environment-unknown": (failure, replace(passing, environment_identity=None), None),
        "unstable-not-parent": (replace(unstable, provenance="telemetry-unverified"), passing, None),
        "unstable-not-later": (unstable, replace(passing, timestamp=unstable.timestamp), None),
        "unstable-base": (unstable, replace(passing, claim="base-reproduction"), None),
        "unstable-head-unknown": (unstable, passing, None),
        "restored-head-unknown": (replace(failure, restored=True), passing, None),
        "restored-not-parent": (replace(failure, restored=True, provenance="self-reported"), passing, context),
        "restored-out-of-checkout": (
            replace(
                failure, restored=True, restart_command_digest="0" * 32,
                caveats=(evidence_module.OUT_OF_CHECKOUT_CONTEXT_CAVEAT,),
            ),
            replace(passing, restart_command_digest="0" * 32),
            context,
        ),
    }
    reasons = {}
    for name, (left, right, ctx) in cases.items():
        allowed, reason = _can_supersede(
            left, right, registry=registry, context=ctx, failure_index=0, passing_index=1,
        )
        assert allowed is False, name
        assert reason is not None and is_supersession_reason(reason), (name, reason)
        reasons[name] = reason
    assert reasons["command"] == REASON_COMMAND_DIFFERS
    assert reasons["fixture"] == REASON_FIXTURE_DIFFERS
    assert reasons["tree"] == REASON_TREE_CHANGED
    assert reasons["environment-different"] == "environment identity is different"
    assert reasons["environment-unknown"] == "environment identity is unknown"
    assert reasons["restored-out-of-checkout"] == REASON_OUT_OF_CHECKOUT
    assert reasons["unstable-base"] == evidence_module.REASON_BASE_REPRODUCTION
    assert reasons["unstable-not-later"] == evidence_module.REASON_NOT_LATER
    # The launch-failure branch of the restart path names its own reason.
    launch = replace(
        failure, restored=True, restart_command_digest="0" * 32, outcome="launch-failed",
        caveats=(evidence_module.PRE_COLLECTION_LAUNCH_FAILURE_PREFIX + ": missing",),
    )
    allowed, reason = _can_supersede_across_restart(
        launch, replace(passing, restart_command_digest="0" * 32),
        context=context, failure_index=0, passing_index=1,
    )
    assert (allowed, reason) == (False, evidence_module.REASON_LAUNCH_FAILURE)
    # An exact pass still supersedes: allow decisions are unchanged.
    assert _can_supersede(failure, passing, registry=registry) == (True, None)


def test_more_than_three_rejections_keep_two_earliest_the_latest_and_a_summary():
    registry = EnvironmentIdentityRegistry()
    rows = [_obs("fail-1", "failed", 0, registry=registry)]
    rows += [_obs(f"pass-{i}", "passed", i + 1, registry=registry, command=_cmd(i)) for i in range(5)]
    evidence = reconcile_test_observations(rows, registry=registry)
    assert evidence.authoritative_failures == ("fail-1",)
    texts = _diagnostic_texts(evidence)
    assert texts == [
        f"fail-1 retained against pass-0: {REASON_COMMAND_DIFFERS}",
        f"fail-1 retained against pass-1: {REASON_COMMAND_DIFFERS}",
        f"fail-1 retained against pass-4: {REASON_COMMAND_DIFFERS} (latest)",
        "fail-1: 2 other later pass(es) also rejected",
    ]
    assert evidence.supersession_rejections == (("fail-1", "pass-4", REASON_COMMAND_DIFFERS, 5),)
    # Never part of the generic caveats or of any projection.
    assert not any("retained against" in item for item in evidence.caveats)
    assert "supersession_diagnostics" not in evidence.to_dict()
    assert "supersession_rejections" not in json.dumps(evidence.to_dict())


def test_different_selection_pass_keeps_the_failure_and_records_command_differs():
    registry = EnvironmentIdentityRegistry()
    failure = _obs("narrow-failure", "failed", 0, registry=registry)
    broader = _obs(
        "broader-pass", "passed", 1, registry=registry,
        command=("python", "-m", "pytest", "tests", "-q"),
    )
    evidence = reconcile_test_observations([failure, broader], registry=registry)
    assert evidence.observations[0].superseded_by is None
    assert evidence.authoritative_failures == ("narrow-failure",)
    expected = f"narrow-failure retained against broader-pass: {REASON_COMMAND_DIFFERS} (latest)"
    assert _diagnostic_texts(evidence) == [expected]
    encoded = bounded_evidence_for_round(evidence)
    assert expected in json.loads(encoded)["caveats"]
    assert "command differs" in expected


def test_fixture_identity_mismatch_produces_a_retained_caveat():
    registry = EnvironmentIdentityRegistry()
    record = FixtureRecord(path="fixture.py", digest="100755:" + "a" * 40)
    failure = _obs("fixture-failure", "failed", 0, registry=registry, fixture=record)
    plain = _obs("plain-pass", "passed", 1, registry=registry)
    evidence = reconcile_test_observations([failure, plain], registry=registry)
    assert evidence.authoritative_failures == ("fixture-failure",)
    assert _diagnostic_texts(evidence) == [
        f"fixture-failure retained against plain-pass: {REASON_FIXTURE_DIFFERS} (latest)"
    ]


def test_out_of_checkout_pass_is_recorded_as_a_rejection():
    registry = EnvironmentIdentityRegistry()
    failure = _obs("in-failure", "failed", 0, registry=registry)
    outside = _obs(
        "outside-pass", "passed", 1, registry=registry,
        caveats=(evidence_module.OUT_OF_CHECKOUT_CONTEXT_CAVEAT,),
    )
    evidence = reconcile_test_observations([failure, outside], registry=registry)
    assert evidence.authoritative_failures == ("in-failure",)
    assert evidence.supersession_rejections == (("in-failure", "outside-pass", REASON_OUT_OF_CHECKOUT, 1),)


def test_replay_after_a_tracked_tree_change_does_not_supersede():
    registry = EnvironmentIdentityRegistry()
    failure = _obs("tree-failure", "failed", 0, registry=registry, digest="tree-a")
    replay = _obs("tree-replay", "passed", 1, registry=registry, digest="tree-b", rerun_of="tree-failure")
    evidence = reconcile_test_observations([failure, replay], registry=registry)
    assert evidence.observations[0].superseded_by is None
    assert evidence.authoritative_failures == ("tree-failure",)
    assert _diagnostic_texts(evidence) == [
        f"tree-failure retained against tree-replay: {REASON_TREE_CHANGED} (latest)"
    ]


@pytest.mark.parametrize("variant", ["different", "unknown"])
def test_replay_in_a_non_equivalent_environment_does_not_supersede(variant):
    registry = EnvironmentIdentityRegistry()
    failure = _obs(
        "env-failure", "failed", 0, registry=registry,
        environment={"PATH": "/usr/bin", TEMP_NAME: TEMP_VALUE},
    )
    replay = _obs("env-replay", "passed", 1, registry=registry, rerun_of="env-failure")
    if variant == "unknown":
        replay = replace(replay, environment_identity=None)
    evidence = reconcile_test_observations([failure, replay], registry=registry)
    assert evidence.authoritative_failures == ("env-failure",)
    expected = f"env-failure retained against env-replay: environment identity is {variant} (latest)"
    assert _diagnostic_texts(evidence) == [expected]
    encoded = bounded_evidence_for_round(evidence)
    assert expected in encoded
    for text in (encoded, *_diagnostic_texts(evidence), *evidence.caveats):
        assert TEMP_NAME not in text and TEMP_VALUE not in text


def test_latest_reason_survives_many_rejections_and_reaches_the_prompt(tmp_path):
    registry = EnvironmentIdentityRegistry()
    rows = [_obs("latest-failure", "failed", 0, registry=registry, environment={"PATH": "/a"})]
    rows += [_obs(f"wrong-{i}", "passed", i + 1, registry=registry, command=_cmd(i)) for i in range(3)]
    rows.append(_obs("exact-replay", "passed", 9, registry=registry, environment={"PATH": "/b"},
                     rerun_of="latest-failure"))
    evidence = reconcile_test_observations(rows, registry=registry)
    latest = [text for _f, kind, text in evidence.supersession_diagnostics if kind == "latest"]
    assert latest == [
        "latest-failure retained against exact-replay: environment identity is different (latest)"
    ]
    catalog = RerunCatalog()
    block = unsuperseded_failure_rerun_block(evidence, catalog, tmp_path, PREFIX)
    assert (
        "Latest rejected later pass `exact-replay`: environment identity is different "
        "(4 later pass(es) rejected in total)."
    ) in block
    assert REASON_COMMAND_DIFFERS not in block


# --- Coder prompt block --------------------------------------------------------


def _k_failure(registry, receipt="k-failure", minute=0):
    return _obs(
        receipt, "failed", minute, registry=registry,
        command=("python", "-m", "pytest", "tests/test_durable.py", "-k", RAW_K, "-q", "-n", "0"),
    )


def _record(catalog, receipt, root, argv=("python", "-m", "pytest", "-k", RAW_K), env=None):
    assert catalog.record(
        receipt_id=receipt, argv=argv, cwd=root.resolve(), root=root, fixture=None,
        environment=env or {"PATH": "/usr/bin", TEMP_NAME: TEMP_VALUE},
    )


def test_prompt_block_carries_the_exact_rerun_handle_without_the_raw_k_value(tmp_path):
    from agent_loop_helpers import make_config
    from coding_review_agent_loop.prompts import build_followup_prompt, build_same_pr_followup_prompt

    registry = EnvironmentIdentityRegistry()
    failure = _k_failure(registry)
    broader = _obs("broad-pass", "passed", 1, registry=registry, command=("python", "-m", "pytest", "-q"))
    evidence = reconcile_test_observations([failure, broader], registry=registry)
    catalog = RerunCatalog()
    _record(catalog, "k-failure", tmp_path)
    block = unsuperseded_failure_rerun_block(evidence, catalog, tmp_path, PREFIX)
    redacted = redact_test_command(failure.command)[0]
    assert "<arg-sha256:" in redacted and RAW_K not in redacted
    handle = "/opt/agent-loop/bin/agent-loop run-tests --rerun k-failure"
    assert f"Exact rerun: `{handle}`" in block
    assert f"`{redacted}`" in block
    assert "Receipt `k-failure` (failed)" in block
    assert f"Latest rejected later pass `broad-pass`: {REASON_COMMAND_DIFFERS}" in block
    assert IDENTICAL_RERUN_RULE in block
    assert RAW_K not in block and TEMP_NAME not in block and TEMP_VALUE not in block
    config = make_config(tmp_path)
    for builder in (build_followup_prompt, build_same_pr_followup_prompt):
        prompt = builder(7, 2, "review text", config, local_test_failure_context=block)
        assert handle in prompt and IDENTICAL_RERUN_RULE in prompt and RAW_K not in prompt
        # Without a block the prompt is unchanged.
        assert builder(7, 2, "review text", config) == builder(
            7, 2, "review text", config, local_test_failure_context=""
        )
    # Public projections keep today's redaction.
    encoded = bounded_evidence_for_round(evidence)
    assert RAW_K not in encoded
    assert RAW_K not in unsuperseded_receipts_sentence(encoded)


def test_restored_failure_has_no_handle_and_the_broker_refuses_it(tmp_path):
    registry = EnvironmentIdentityRegistry()
    prior = bounded_evidence_for_round(reconcile_test_observations([_k_failure(registry)], registry=registry))
    restored = decode_bounded_evidence(prior)
    evidence = reconcile_test_observations(restored.observations, registry=EnvironmentIdentityRegistry())
    assert evidence.observations[0].restored
    catalog = RerunCatalog()
    block = unsuperseded_failure_rerun_block(evidence, catalog, tmp_path, PREFIX)
    assert NO_RERUN_HANDLE_REASON in block
    assert "--rerun" not in block.split("Receipt-to-handle map")[0].replace(
        "`agent-loop run-tests --rerun <receipt-id>`", ""
    )
    assert "- `k-failure` -> no exact rerun handle in this invocation" in block
    assert RAW_K not in block
    calls, server = _injected_server(tmp_path, catalog=catalog)
    try:
        result = BrokerClient(_client_env(server)).run([], timeout_seconds=5, cwd=tmp_path, rerun_of="k-failure")
    finally:
        server.stop()
    assert result.error_code == RERUN_RECEIPT_UNKNOWN
    assert result.receipt_id == "" and calls == [] and server.journal == ()


def test_every_eligible_failure_gets_a_handle_line_beyond_eight(tmp_path):
    registry = EnvironmentIdentityRegistry()
    catalog = RerunCatalog()
    rows = []
    for index in range(10):
        rows.append(_obs(f"failure-{index}", "failed", index, registry=registry, command=_cmd(index)))
        if index != 9:
            _record(catalog, f"failure-{index}", tmp_path)
    evidence = reconcile_test_observations(rows, registry=registry)
    block = unsuperseded_failure_rerun_block(evidence, catalog, tmp_path, PREFIX)
    assert block.count("- Receipt `") == 8
    mapping = block.split("Receipt-to-handle map")[1]
    for index in range(9):
        assert f"- `failure-{index}` -> `/opt/agent-loop/bin/agent-loop run-tests --rerun failure-{index}`" in mapping
    assert "- `failure-9` -> no exact rerun handle in this invocation" in mapping
    assert "--rerun failure-9" not in block
    assert RAW_K not in block


def test_catalog_is_root_bound_bounded_and_private(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    catalog = RerunCatalog(limit=2)
    _record(catalog, "a", tmp_path)
    assert catalog.contains("a", tmp_path)
    assert not catalog.contains("a", other)
    assert catalog.resolve("a", other) is None
    _record(catalog, "b", tmp_path)
    _record(catalog, "c", tmp_path)
    assert not catalog.contains("a", tmp_path) and len(catalog) == 2
    assert RAW_K not in repr(catalog) and RAW_K not in repr(catalog.resolve("b", tmp_path))
    assert TEMP_VALUE not in repr(catalog.resolve("b", tmp_path))
    assert not catalog.record(
        receipt_id="bad id", argv=("x",), cwd=tmp_path, root=tmp_path, fixture=None, environment={},
    )
    assert not catalog.record(
        receipt_id="outside", argv=("x",), cwd=other.parent.parent, root=tmp_path, fixture=None, environment={},
    )


# --- Capacity-safe transport ---------------------------------------------------


def _many_failure_evidence(*, padding: int = 0, cited: bool = True, failures: int = 20, passes: int = 4):
    registry = EnvironmentIdentityRegistry()
    rows = [
        _obs(f"fail{index:02d}", "failed", index, registry=registry, command=_cmd(index))
        for index in range(failures)
    ]
    rows += [
        _obs(f"pass{index:02d}", "passed", 30 + index, registry=registry, command=_cmd(100 + index))
        for index in range(passes)
    ]
    evidence = reconcile_test_observations(rows, registry=registry)
    if padding:
        evidence = replace(
            evidence,
            caveats=evidence.caveats + tuple(f"generic padding caveat {i:04d} " + "x" * 200 for i in range(padding)),
        )
    return evidence, ({f"pass{passes - 1:02d}"} if cited else set())


def _identity(encoded: str):
    decoded = decode_bounded_evidence(encoded)
    return (
        tuple(row.receipt_id for row in decoded.observations),
        tuple(item.to_dict() for item in decoded.cited_receipt_references),
        decoded.authoritative_failures,
    )


def _near_cap_padding() -> int:
    best = 0
    for padding in range(0, 80):
        evidence, protected = _many_failure_evidence(padding=padding)
        plain = replace(evidence, supersession_diagnostics=())
        encoded = bounded_evidence_for_round(plain, protected_receipt_ids=protected)
        payload = json.loads(encoded)
        if "counts" in payload or "local evidence details truncated before transport" in payload["caveats"]:
            break
        best = padding
    return best


@pytest.mark.parametrize("variant", ["near-cap", "roomy"])
def test_diagnostics_use_only_leftover_capacity_through_strict_recanonicalization(variant):
    padding = _near_cap_padding() if variant == "near-cap" else 0
    failures = 20 if variant == "near-cap" else 5
    evidence, protected = _many_failure_evidence(padding=padding, failures=failures)
    # The live map keeps every failure's latest reason, whatever transport keeps.
    assert len(evidence.supersession_rejections) == failures
    assert all(count == 4 for _f, _p, _r, count in evidence.supersession_rejections)
    plain = replace(evidence, supersession_diagnostics=())
    with_diag = bounded_evidence_for_round(evidence, protected_receipt_ids=protected)
    without = bounded_evidence_for_round(plain, protected_receipt_ids=protected)
    assert len(with_diag.encode()) <= MAX_ROUND_BYTES
    assert _identity(with_diag) == _identity(without)
    assert "pass03" in _identity(with_diag)[0]
    current_with, current_without = with_diag, without
    for _round in range(2):
        current_with = canonicalize_bounded_evidence(current_with, strict=True)
        current_without = canonicalize_bounded_evidence(current_without, strict=True)
        assert len(current_with.encode()) <= MAX_ROUND_BYTES
        assert _identity(current_with) == _identity(current_without) == _identity(without)
    payload = json.loads(with_diag)
    assert "supersession_diagnostics" not in payload and "supersession_rejections" not in payload
    appended = [item for item in payload["caveats"] if classify_supersession_diagnostic(item)]
    lines = [item for item in appended if classify_supersession_diagnostic(item)[1] != "omission"]
    assert len(lines) <= MAX_SUPERSESSION_CAVEATS
    assert sum(len(item.encode()) for item in lines) <= MAX_SUPERSESSION_CAVEAT_BYTES
    decoded = decode_bounded_evidence(with_diag)
    assert not any(classify_supersession_diagnostic(item) for item in decoded.caveats)
    assert [text for _f, _k, text in decoded.supersession_diagnostics] == appended
    retained = set(_identity(with_diag)[0])
    order = [row.receipt_id for row in decoded.observations if row.is_failure]
    named = [classify_supersession_diagnostic(item)[0] for item in lines]
    assert set(named) <= retained
    latest_order = [
        classify_supersession_diagnostic(item)[0] for item in lines
        if classify_supersession_diagnostic(item)[1] == "latest"
    ]
    assert latest_order == [receipt for receipt in order if receipt in latest_order]
    if variant == "roomy":
        assert latest_order[:3] == order[:3]
        assert any(classify_supersession_diagnostic(item)[1] == "omission" for item in appended)
        sentence = unsuperseded_receipts_sentence(with_diag)
        for receipt in order[:3]:
            assert f"receipt `{receipt}`" in sentence
        assert sentence.count(f"latest later pass rejected: {REASON_COMMAND_DIFFERS}") == 3
        assert "run-tests --rerun <receipt>" in sentence
    else:
        assert padding > 0
        # Near the cap the diagnostics shrink; the base was within ~300 bytes.
        assert len(without.encode()) > MAX_ROUND_BYTES - 600


def test_protected_compaction_and_capacity_stop_are_unchanged_by_diagnostics():
    registry = EnvironmentIdentityRegistry()
    big = tuple(f"tests/test_{'y' * 30}_{index}.py" for index in range(40))
    cited = [
        replace(
            _obs(f"cited{index:02d}", "passed", 30 + index, registry=registry, command=("pytest", *big)),
            identifiers=tuple(f"id-{index}-{n}-" + "z" * 200 for n in range(8)),
        )
        for index in range(24)
    ]
    failures = [_obs(f"fail{index:02d}", "failed", index, registry=registry, command=_cmd(index)) for index in range(6)]
    evidence = reconcile_test_observations([*failures, *cited], registry=registry)
    assert evidence.supersession_diagnostics
    protected = {row.receipt_id for row in cited}
    plain = replace(evidence, supersession_diagnostics=())
    with_diag = bounded_evidence_for_round(evidence, protected_receipt_ids=protected)
    without = bounded_evidence_for_round(plain, protected_receipt_ids=protected)
    assert json.loads(without).get("cited_receipt_references") or any(
        not row.get("identifiers") for row in json.loads(without)["observations"]
    )
    assert _identity(with_diag) == _identity(without)
    assert canonicalize_bounded_evidence(with_diag, strict=True) is not None
    assert _identity(canonicalize_bounded_evidence(with_diag, strict=True)) == _identity(
        canonicalize_bounded_evidence(without, strict=True)
    )
    # A capacity stop happens with and without diagnostics alike.
    huge = [
        replace(
            _obs(f"cfail{index:02d}", "failed", index, registry=registry, command=("pytest", *big)),
            caveats=tuple("c" * 250 for _ in range(4)),
        )
        for index in range(32)
    ]
    late = [_obs(f"later{index}", "passed", 50 + index, registry=registry, command=_cmd(index)) for index in range(4)]
    stop_evidence = reconcile_test_observations([*huge, *late], registry=registry)
    stop_protected = {row.receipt_id for row in huge}
    outcomes = []
    for candidate in (stop_evidence, replace(stop_evidence, supersession_diagnostics=())):
        try:
            bounded_evidence_for_round(candidate, protected_receipt_ids=stop_protected)
        except AgentLoopError as exc:
            outcomes.append(("stop", str(exc)))
        else:
            outcomes.append(("ok", None))
    assert outcomes[0] == outcomes[1]


def test_forced_fixed_point_mismatch_returns_the_diagnostics_free_output(monkeypatch):
    evidence, protected = _many_failure_evidence()
    without = bounded_evidence_for_round(replace(evidence, supersession_diagnostics=()), protected_receipt_ids=protected)
    original = evidence_module.canonicalize_bounded_evidence

    def mismatching(value, *, strict=False, _verify_fixed_point=True):
        if not _verify_fixed_point and "retained against" in str(value):
            return bounded_evidence_for_round(LocalTestEvidence(observations=()), _verify_fixed_point=False)
        return original(value, strict=strict, _verify_fixed_point=_verify_fixed_point)

    monkeypatch.setattr(evidence_module, "canonicalize_bounded_evidence", mismatching)
    assert bounded_evidence_for_round(evidence, protected_receipt_ids=protected) == without


def test_legacy_retained_caveat_decodes_as_a_generic_caveat():
    payload = json.loads(bounded_evidence_for_round(LocalTestEvidence(observations=())))
    payload["caveats"] = [
        "abc retained: environment identity is unknown",
        "abc retained against def: not a closed reason",
        "abc retained against def: tracked tree changed (latest)",
        "3 further supersession caveat(s) omitted",
    ]
    decoded = decode_bounded_evidence(json.dumps(payload))
    assert decoded.caveats == (
        "abc retained: environment identity is unknown",
        "abc retained against def: not a closed reason",
    )
    assert [kind for _f, kind, _t in decoded.supersession_diagnostics] == ["latest", "omission"]
    assert decoded.latest_diagnostic_reason_for("abc") == "tracked tree changed"


# --- Broker replay -------------------------------------------------------------


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout


def _git_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    (root / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-qm", "initial")
    return root


def _injected_server(root, *, catalog=None, outcomes=None, registry=None):
    calls = []
    sequence = list(outcomes or [])

    def execute(argv, cwd, timeout, environment, stream):
        calls.append((tuple(argv), Path(cwd), dict(environment)))
        outcome = sequence.pop(0) if sequence else "failed"
        return SimpleNamespace(
            outcome=outcome, returncode=0 if outcome == "passed" else 1,
            elapsed_seconds=0.01, output_tail="", suite_start="verified",
            wrapper_bootstrap="verified", inner_exec="started",
        )

    server = BrokerServer(
        root=root, turn_id="turn-rerun-1999", execute=execute, rerun_catalog=catalog,
        environment_registry=registry,
    ).start()
    return calls, server


def _client_env(server, **extra):
    return {
        "PATH": os.environ.get("PATH", ""),
        **server.environment,
        "AGENT_LOOP_INVOCATION_ID": server.turn_id,
        **extra,
    }


K_ARGV = (sys.executable, "-m", "pytest", "tests/test_durable.py", "-k", RAW_K, "-q", "-n", "0")


def test_exact_rerun_replays_the_recorded_failure_and_supersedes_it(tmp_path):
    root = _git_repo(tmp_path / "repo")
    registry = EnvironmentIdentityRegistry()
    catalog = RerunCatalog()
    calls, server = _injected_server(root, catalog=catalog, outcomes=["failed", "passed"], registry=registry)
    try:
        first = BrokerClient(_client_env(server, **{TEMP_NAME: TEMP_VALUE})).run(
            list(K_ARGV), timeout_seconds=5, cwd=root,
        )
        assert first.outcome == "failed" and catalog.contains(first.receipt_id, root)
        replay = BrokerClient(_client_env(server)).run([], timeout_seconds=5, cwd=root, rerun_of=first.receipt_id)
        journal = list(server.journal)
    finally:
        server.stop()
    # The recorded argv, cwd and temporary assignment were replayed.
    assert calls[1][0] == K_ARGV and calls[1][1] == calls[0][1]
    assert calls[1][2][TEMP_NAME] == TEMP_VALUE
    assert calls[1][2]["AGENT_LOOP_INVOCATION_ID"] == server.turn_id
    assert replay.rerun_of == first.receipt_id and replay.outcome == "passed"
    assert RAW_K not in replay.rerun_command and "<arg-sha256:" in replay.rerun_command
    assert all(RAW_K not in item for item in replay.executed_argv)
    failure, passing = journal
    assert passing.rerun_of == failure.receipt_id
    assert passing.normalized_command == failure.normalized_command
    assert passing.scope.normalized() == failure.scope.normalized()
    assert passing.fixture == failure.fixture
    assert passing.restart_command_digest == failure.restart_command_digest is not None
    assert passing.environment_identity.compare(failure.environment_identity) == "equivalent"
    head = _git(root, "rev-parse", "HEAD").strip()
    evidence = reconcile_test_observations(
        journal, current_head=head, current_snapshot=stable_tracked_tree_snapshot(root), registry=registry,
    )
    assert evidence.observations[0].superseded_by == passing.receipt_id
    assert evidence.authoritative_failures == ()
    assert superseded_by_current_turn(evidence, {passing.receipt_id}) == {passing.receipt_id}
    encoded = bounded_evidence_for_round(evidence)
    assert RAW_K not in encoded and TEMP_NAME not in encoded and TEMP_VALUE not in encoded
    assert json.loads(encoded)["observations"][1]["rerun_of"] == failure.receipt_id


def test_rerun_refusals_run_nothing_and_record_nothing(tmp_path):
    root = _git_repo(tmp_path / "repo")
    other = _git_repo(tmp_path / "other")
    catalog = RerunCatalog()
    _record(catalog, "other-root", other)
    calls, server = _injected_server(root, catalog=catalog, outcomes=["passed"])
    try:
        client = BrokerClient(_client_env(server))
        passing = client.run([sys.executable, "-c", "pass"], timeout_seconds=5, cwd=root)
        assert passing.outcome == "passed" and not catalog.contains(passing.receipt_id, root)
        refused = [
            client.run([], timeout_seconds=5, cwd=root, rerun_of=receipt)
            for receipt in ("unknown-receipt", passing.receipt_id, "other-root")
        ]
        with pytest.raises(evidence_module.BrokerNotDispatched):
            client.run(["pytest"], timeout_seconds=5, cwd=root, rerun_of="x")
        with pytest.raises(evidence_module.BrokerNotDispatched):
            client.run([], timeout_seconds=5, cwd=root, rerun_of="x", fixture="fixture.py")
        raw = _signed_request(server, root)
        for request in (
            {**raw, "rerun_of": "x"},
            {**{k: v for k, v in raw.items() if k != "argv"}, "rerun_of": "x", "fixture": "f.py"},
            {**{k: v for k, v in raw.items() if k != "argv"}, "rerun_of": "bad id"},
        ):
            assert _raw_request(server, request)["type"] == "error"
        journal = list(server.journal)
    finally:
        server.stop()
    assert [item.error_code for item in refused] == [RERUN_RECEIPT_UNKNOWN] * 3
    assert all(item.receipt_id == "" and item.returncode == 2 for item in refused)
    assert all("nothing was run (not evidence)" in item.diagnostic for item in refused)
    assert len(calls) == 1 and len(journal) == 1
    # No catalog: refused the same way.
    calls, server = _injected_server(root, catalog=None)
    try:
        result = BrokerClient(_client_env(server)).run([], timeout_seconds=5, cwd=root, rerun_of="any")
    finally:
        server.stop()
    assert result.error_code == RERUN_CATALOG_UNAVAILABLE and calls == [] and server.journal == ()


def test_launch_failures_and_out_of_checkout_failures_are_not_catalogued(tmp_path, monkeypatch):
    root = _git_repo(tmp_path / "repo")
    catalog = RerunCatalog()

    def launch_failure(argv, cwd, timeout, environment, stream):
        return SimpleNamespace(
            outcome="failed", returncode=1, elapsed_seconds=0.0, output_tail="",
            suite_start="not-started", pre_collection_launch_failure="no module named pytest",
        )

    server = BrokerServer(root=root, turn_id="turn-launch", execute=launch_failure, rerun_catalog=catalog).start()
    try:
        client = BrokerClient(_client_env(server))
        launch = client.run([sys.executable, "-m", "pytest", "-q"], timeout_seconds=5, cwd=root)
        outside = client.run(
            [sys.executable, "-m", "pytest", str(tmp_path / "elsewhere" / "test_x.py")],
            timeout_seconds=5, cwd=root,
        )
    finally:
        server.stop()
    assert launch.receipt_id and not catalog.contains(launch.receipt_id, root)
    assert not catalog.contains(outside.receipt_id, root)
    assert len(catalog) == 0


def _signed_request(server, root):
    raw_nonce = "c" * 32
    nonce = raw_nonce + "." + hmac.new(
        server.capability.encode("ascii"), raw_nonce.encode("ascii"), hashlib.sha256
    ).hexdigest()
    return {
        "turn_id": server.turn_id, "nonce": nonce, "argv": [sys.executable, "-c", "pass"],
        "timeout_seconds": 5, "cwd": str(root),
        "environment": {"PATH": os.environ.get("PATH", ""), evidence_module.BROKER_CLIENT_ACKS_ENV: "1"},
    }


def _raw_request(server, request):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(10)
        connection.connect(server.endpoint)
        evidence_module._send_frame(connection, request)
        while True:
            response = evidence_module._recv_frame(connection)
            if response.get("type") != "output":
                return response


# --- CLI -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["run-tests", "--rerun", "abc", "--", "pytest", "-q"],
        ["run-tests", "--rerun", "abc", "--fixture", "fixture.py"],
        ["run-tests", "--rerun", "abc", "--preflight"],
        ["run-tests", "--rerun", "bad id"],
    ],
)
def test_cli_rerun_conflicts_are_refused_before_dispatch(argv, monkeypatch, capsys):
    import coding_review_agent_loop.cli as cli_module

    monkeypatch.setattr(cli_module, "broker_client_from_environment", lambda: pytest.fail("dispatched"))
    monkeypatch.setattr(cli_module, "run_foreground_test", lambda *a, **k: pytest.fail("ran locally"))
    assert cli_module.main(argv) == 2
    assert "nothing was run (not evidence)" in capsys.readouterr().err


def test_cli_rerun_without_a_broker_never_runs_locally(monkeypatch, capsys):
    import coding_review_agent_loop.cli as cli_module

    monkeypatch.setattr(cli_module, "broker_client_from_environment", lambda: None)
    monkeypatch.setattr(cli_module, "run_foreground_test", lambda *a, **k: pytest.fail("ran locally"))
    assert cli_module.main(["run-tests", "--rerun", "abc"]) == 1
    assert cli_module.RERUN_NOT_DISPATCHED_MESSAGE in capsys.readouterr().err


def test_cli_rerun_prints_the_receipt_redacted_command_and_citation(tmp_path, monkeypatch, capsys):
    import coding_review_agent_loop.cli as cli_module

    root = _git_repo(tmp_path / "repo")
    catalog = RerunCatalog()
    calls, server = _injected_server(root, catalog=catalog, outcomes=["failed", "passed"])
    try:
        first = BrokerClient(_client_env(server)).run(list(K_ARGV), timeout_seconds=5, cwd=root)
        env = _client_env(server)
        monkeypatch.setattr(cli_module, "broker_client_from_environment", lambda: BrokerClient(env))
        monkeypatch.chdir(root)
        code = cli_module.main(["run-tests", "--timeout-seconds", "5", "--rerun", first.receipt_id])
    finally:
        server.stop()
    err = capsys.readouterr().err
    assert code == 0
    assert f"cite this receipt with the command `agent-loop run-tests --rerun {first.receipt_id}`" in err
    assert "agent-loop test observation receipt:" in err
    assert RAW_K not in err and calls[1][0] == K_ARGV


# --- Citations and public text -------------------------------------------------


def test_parse_rerun_citation_is_strict():
    assert parse_rerun_citation("agent-loop run-tests --rerun abc123") == "abc123"
    assert parse_rerun_citation('"${AGENT_LOOP_EXECUTABLE:-agent-loop}" run-tests --rerun abc') == "abc"
    assert parse_rerun_citation("/opt/x/agent-loop run-tests --timeout-seconds 60 --rerun=abc") == "abc"
    assert parse_rerun_citation("python3 -m coding_review_agent_loop.cli run-tests --rerun abc") == "abc"
    for text in (
        "agent-loop run-tests --rerun abc -- pytest",
        "agent-loop run-tests --rerun abc --rerun def",
        "agent-loop run-tests --rerun 'bad id'",
        "agent-loop run-tests -- pytest -q",
        "other run-tests --rerun abc",
        "agent-loop run-tests --fixture f.py --rerun abc",
        None,
    ):
        assert parse_rerun_citation(text) is None, text


def _risk_matrix_pieces(command):
    from coding_review_agent_loop.protocol_risk import parse_risk_test_matrix, risk_test_matrix_identity

    matrix = parse_risk_test_matrix({
        "applicability": "applicable",
        "rows": [{
            "row_id": "row-rerun", "label": "Rerun", "entry_path_or_mode": "run-tests --rerun",
            "initial_state": "failure", "event": "replay", "expected_outcome": "superseded",
            "forbidden_side_effects": ["raw argv"], "proposed_test_level": "unit",
            "proposed_test_location": "tests/test_local_test_rerun.py",
            "applicability": "applicable", "related_scope_item_ids": ["scope"],
            "execution_owner": "one-shot",
        }],
        "important_exclusions": ["none"],
    })
    evidence = {
        "matrix_identity": risk_test_matrix_identity(matrix),
        "rows": [{
            "row_id": "row-rerun", "status": "verified",
            "test_identifiers": ["test_x"], "test_locations": ["tests/test_x.py"],
            "workflow_path_claim": "replay", "outcome_assertions": ["superseded"],
            "forbidden_effect_assertions": ["no raw argv"],
            "evidence_citations": [{"command": command, "receipt_id": "replay-r", "claim": "current-result"}],
        }],
    }
    return matrix, evidence


def _replay_observation(rerun_of="failure-f"):
    return SimpleNamespace(
        receipt_id="replay-r", claim=None, rerun_of=rerun_of,
        normalized_command="python -m pytest tests/test_durable.py -q", outcome="passed",
        provenance="parent-observed",
        public_projection=lambda: {"command": "python -m pytest tests/test_durable.py -q"},
    )


def test_risk_matrix_matcher_accepts_a_rerun_citation_only_for_its_replay_row():
    from coding_review_agent_loop.protocol_risk import parse_risk_test_matrix_evidence

    matrix, evidence = _risk_matrix_pieces("agent-loop run-tests --rerun failure-f")
    parsed = parse_risk_test_matrix_evidence(
        evidence, matrix=matrix, authoritative_test_observations=[_replay_observation()],
    )
    assert parsed.rows[0].status == "verified"
    assert parsed.rows[0].evidence_citations[0].receipt_id == "replay-r"
    for observations in ([_replay_observation("failure-g")], [_replay_observation(None)], []):
        with pytest.raises(AgentLoopError, match="authoritative"):
            parse_risk_test_matrix_evidence(
                evidence, matrix=matrix, authoritative_test_observations=observations,
            )
    matrix, wrong = _risk_matrix_pieces("agent-loop run-tests --rerun failure-g")
    with pytest.raises(AgentLoopError, match="authoritative"):
        parse_risk_test_matrix_evidence(
            wrong, matrix=matrix, authoritative_test_observations=[_replay_observation()],
        )


def test_public_rendering_and_projection_stay_redacted_and_compatible():
    from coding_review_agent_loop.comment_rendering import _render_test_observation_citations

    registry = EnvironmentIdentityRegistry()
    failure = _k_failure(registry, receipt="k-failure")
    replay = replace(_k_failure(registry, receipt="k-replay", minute=1), outcome="passed", rerun_of="k-failure")
    evidence = reconcile_test_observations([failure, replay], registry=registry)
    encoded = bounded_evidence_for_round(evidence)
    assert RAW_K not in encoded
    citation = SimpleNamespace(command="agent-loop run-tests --rerun k-failure", receipt_id="k-replay", claim="current-result")
    rendered = _render_test_observation_citations(
        [citation], local_test_evidence=encoded, current_test_turn_id="turn-1999",
    )
    redacted = redact_test_command(failure.command)[0]
    assert f"- `{redacted}` — receipt `k-replay` — `current-result` (verified against the parent journal" in rendered
    assert RAW_K not in rendered
    wrong = SimpleNamespace(command="agent-loop run-tests --rerun other", receipt_id="k-replay", claim="current-result")
    assert "unverified" in _render_test_observation_citations(
        [wrong], local_test_evidence=encoded, current_test_turn_id="turn-1999",
    )
    # ``rerun_of`` is projected only when set, and decoded strictly.
    assert "rerun_of" not in failure.public_projection()
    assert replay.public_projection()["rerun_of"] == "k-failure"
    projection = replay.public_projection()
    assert observation_from_mapping(projection).rerun_of == "k-failure"
    for bad in ("bad id", "x" * 200, 7, "../escape"):
        assert observation_from_mapping({**projection, "rerun_of": bad}).rerun_of is None
    # The stop sentence names receipt ids and reasons, never raw argv.
    broader = _obs("k-broad", "passed", 2, registry=registry, command=("python", "-m", "pytest", "-q"))
    open_evidence = reconcile_test_observations([failure, broader], registry=registry)
    sentence = unsuperseded_receipts_sentence(bounded_evidence_for_round(open_evidence))
    assert "receipt `k-failure`" in sentence and REASON_COMMAND_DIFFERS in sentence
    assert RAW_K not in sentence and "run-tests --rerun <receipt>" in sentence


# --- Runner: replay across per-turn brokers -----------------------------------


def _src_env(**extra):
    source = str(Path(__file__).resolve().parents[1] / "src")
    existing = os.environ.get("PYTHONPATH")
    return {"PYTHONPATH": source + (os.pathsep + existing if existing else ""), **extra}


_TURN_CODE = "import os, sys; sys.exit(0 if os.path.exists(os.environ['RERUN_MARKER_1999']) else 1)"


def test_failure_from_an_earlier_coder_turn_is_replayed_by_the_next_turn_broker(tmp_path, monkeypatch):
    from coding_review_agent_loop.containment import default_policy
    from coding_review_agent_loop.runner import Runner

    if not sys.platform.startswith("linux"):
        pytest.skip("the test broker requires Linux")
    source = str(Path(__file__).resolve().parents[1] / "src")
    monkeypatch.setenv("PYTHONPATH", source + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
    root = _git_repo(tmp_path / "repo")
    logs = tmp_path / "logs"
    logs.mkdir()
    marker = tmp_path / "marker"
    runner = Runner(containment_policy=default_policy(mode="off", cache_dir=tmp_path / ".runtime"))
    turn_one = (
        "from coding_review_agent_loop.cli import main; import sys; "
        "raise SystemExit(main(['run-tests','--timeout-seconds','30','--',"
        f"sys.executable,'-c',{_TURN_CODE!r},'-k',{RAW_K!r}]))"
    )
    first = runner.run_with_log(
        [sys.executable, "-c", turn_one], cwd=root, log_path=logs / "turn-1.log", label="coder",
        progress_interval_seconds=1, check=False,
        env=_src_env(RERUN_MARKER_1999=str(marker), **{TEMP_NAME: TEMP_VALUE}),
    )
    assert first.returncode == 1
    (failure,) = runner.local_test_observations()
    assert failure.outcome == "failed" and failure.provenance == "parent-observed"
    head = _git(root, "rev-parse", "HEAD").strip()
    assert runner.render_local_test_evidence(current_head=head, cwd=root) is not None
    block = unsuperseded_failure_rerun_block(runner.last_reconciled_local_test_evidence, runner.rerun_catalog, root, PREFIX)
    assert f"--rerun {failure.receipt_id}`" in block
    assert RAW_K not in block and TEMP_NAME not in block and TEMP_VALUE not in block
    marker.write_text("fixed", encoding="utf-8")
    turn_two = (
        "from coding_review_agent_loop.cli import main; "
        f"raise SystemExit(main(['run-tests','--timeout-seconds','30','--rerun',{failure.receipt_id!r}]))"
    )
    # The second turn has neither the marker variable nor the temporary assignment.
    second = runner.run_with_log(
        [sys.executable, "-c", turn_two], cwd=root, log_path=logs / "turn-2.log", label="coder",
        progress_interval_seconds=1, check=False, env=_src_env(),
    )
    log_text = (logs / "turn-2.log").read_text(encoding="utf-8", errors="replace")
    assert second.returncode == 0, log_text
    assert RAW_K not in log_text and TEMP_VALUE not in log_text and TEMP_NAME not in log_text
    rendered = runner.render_local_test_evidence(current_head=head, cwd=root)
    evidence = runner.last_reconciled_local_test_evidence
    replay = [row for row in evidence.observations if row.rerun_of == failure.receipt_id]
    assert len(replay) == 1 and replay[0].outcome == "passed"
    assert replay[0].turn_id != failure.turn_id
    assert evidence.observations[0].superseded_by == replay[0].receipt_id
    assert evidence.authoritative_failures == ()
    assert RAW_K not in rendered and TEMP_VALUE not in rendered and TEMP_NAME not in rendered
