import json
import os
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from coding_review_agent_loop.local_test_evidence import (
    ENVIRONMENT_EXCLUSIONS,
    EvidenceScope,
    EnvironmentIdentityRegistry,
    LocalTestObservation,
    TestBrokerClient as BrokerClient,
    TestBrokerServer as BrokerServer,
    TreeAttribution,
    attribute_base_reproduction,
    bounded_evidence_for_round,
    canonical_environment_bytes,
    decode_bounded_evidence,
    environment_comparison_for_restart,
    parse_legacy_tests_run,
    reconcile_test_observations,
    redact_test_command,
)
from coding_review_agent_loop.containment import open_confined_cwd


def _observation(
    *,
    outcome: str,
    timestamp: str,
    receipt_id: str,
    registry: EnvironmentIdentityRegistry,
    environment: dict[str, str] | None = None,
    scope: EvidenceScope | None = None,
    state: str = "current-head",
    tracked_digest: str | None = "tree-a",
    provenance: str = "parent-observed",
) -> LocalTestObservation:
    cwd = Path("/checkout")
    argv = (sys.executable, "-m", "pytest", "tests/test_protocol.py", "-q")
    return LocalTestObservation(
        command=argv,
        outcome=outcome,
        provenance=provenance,
        scope=scope or EvidenceScope("suite", ("tests/test_protocol.py",)),
        receipt_id=receipt_id,
        turn_id="turn-opaque",
        timestamp=timestamp,
        cwd=str(cwd),
        normalized_command="python -m pytest tests/test_protocol.py -q",
        returncode=0 if outcome == "passed" else 1,
        attribution=TreeAttribution(
            state=state,
            head="head-a",
            pre_digest="pre-a",
            post_digest="post-a",
            tracked_digest=tracked_digest,
            stable=True,
        ),
        environment_state="not-compared",
        environment_identity=registry.capture(environment or {"PATH": "/usr/bin"}),
    )


def test_failed_full_suite_then_subset_remains_visible_and_exact_rerun_supersedes():
    registry = EnvironmentIdentityRegistry()
    full = _observation(
        outcome="failed",
        timestamp="2026-09-10T10:00:00+00:00",
        receipt_id="full-failure",
        registry=registry,
        scope=EvidenceScope("suite", ("all",)),
    )
    subset = _observation(
        outcome="passed",
        timestamp="2026-09-10T10:01:00+00:00",
        receipt_id="subset-pass",
        registry=registry,
        scope=EvidenceScope("subset", ("tests/test_protocol.py",)),
    )
    rerun = _observation(
        outcome="passed",
        timestamp="2026-09-10T10:02:00+00:00",
        receipt_id="full-pass",
        registry=registry,
        scope=EvidenceScope("suite", ("all",)),
    )

    evidence = reconcile_test_observations([full, subset, rerun], registry=registry)

    assert evidence.observations[0].superseded_by == "full-pass"
    assert evidence.observations[1].superseded_by is None
    assert evidence.observations[1].outcome == "passed"
    assert evidence.authoritative_failures == ()


def test_environment_divergence_and_unknown_do_not_supersede_a_failure():
    registry = EnvironmentIdentityRegistry()
    failure = _observation(
        outcome="failed",
        timestamp="2026-09-10T10:00:00+00:00",
        receipt_id="failure",
        registry=registry,
        environment={"TEST_INPUT": "one"},
    )
    different = _observation(
        outcome="passed",
        timestamp="2026-09-10T10:01:00+00:00",
        receipt_id="different",
        registry=registry,
        environment={"TEST_INPUT": "two"},
    )
    unknown = _observation(
        outcome="passed",
        timestamp="2026-09-10T10:02:00+00:00",
        receipt_id="unknown",
        registry=registry,
    )
    unknown = LocalTestObservation(
        **{**unknown.__dict__, "environment_identity": None}
    )

    evidence = reconcile_test_observations(
        [failure, different, unknown], registry=registry
    )

    assert evidence.observations[0].superseded_by is None
    assert evidence.observations[1].environment_state == "different"
    assert evidence.observations[2].environment_state == "unknown"
    assert evidence.authoritative_failures == ("failure",)


def test_duplicate_and_conflicting_receipts_are_explicit():
    registry = EnvironmentIdentityRegistry()
    first = _observation(
        outcome="failed",
        timestamp="2026-09-10T10:00:00+00:00",
        receipt_id="same",
        registry=registry,
    )
    duplicate = first
    conflicting = _observation(
        outcome="passed",
        timestamp="2026-09-10T10:01:00+00:00",
        receipt_id="same",
        registry=registry,
    )

    evidence = reconcile_test_observations(
        [first, duplicate, conflicting], registry=registry
    )

    assert len(evidence.observations) == 2
    assert any("idempotent duplicate" in item for item in evidence.caveats)
    assert evidence.observations[1].outcome == "incomplete"
    assert evidence.observations[1].provenance == "telemetry-unverified"


def test_legacy_tests_run_uses_shell_parsing_and_marks_capture_limits(tmp_path):
    rows = parse_legacy_tests_run(
        ["python -m pytest tests/test_protocol.py -q", "pytest tests && echo done", "'unterminated"],
        cwd=tmp_path,
    )

    assert rows[0].outcome == "passed"
    assert rows[1].outcome == "incomplete"
    assert rows[2].outcome == "incomplete"
    assert all(row.provenance == "self-reported" for row in rows)
    assert all("capture" in " ".join(row.caveats) for row in rows[1:])


def test_environment_identity_uses_exact_exclusions_and_keeps_other_variables():
    base = {name: "volatile" for name in ENVIRONMENT_EXCLUSIONS}
    base.update({"PATH": "/usr/bin", "TEST_SECRET_VALUE": "first"})
    equivalent = dict(base)
    equivalent["TERM"] = "different"
    assert canonical_environment_bytes(base) == canonical_environment_bytes(equivalent)

    changed = dict(base)
    changed["TEST_SECRET_VALUE"] = "second"
    assert canonical_environment_bytes(base) != canonical_environment_bytes(changed)
    with pytest.raises(Exception):
        canonical_environment_bytes({"Path": "one", "PATH": "two"}, case_insensitive=True)


def test_restart_evidence_is_identity_unknown_and_never_supersedes():
    registry = EnvironmentIdentityRegistry()
    failure = _observation(
        outcome="failed",
        timestamp="2026-09-10T10:00:00+00:00",
        receipt_id="old-failure",
        registry=registry,
    )
    restored = decode_bounded_evidence(
        bounded_evidence_for_round(reconcile_test_observations([failure]))
    )
    assert restored is not None
    assert restored.observations[0].environment_state == environment_comparison_for_restart()

    passing = _observation(
        outcome="passed",
        timestamp="2026-09-10T10:01:00+00:00",
        receipt_id="new-pass",
        registry=registry,
    )
    evidence = reconcile_test_observations(
        [*restored.observations, passing], registry=registry
    )
    assert evidence.observations[0].superseded_by is None
    assert evidence.observations[0].environment_state == "identity-unknown"


def test_public_projection_redacts_credentials_paths_and_diagnostics(tmp_path):
    command, identifiers, caveats = redact_test_command(
        [
            sys.executable,
            "--token",
            "super-secret",
            f"postgres://user:password@db.example.test:5432/app",
            str(tmp_path / "private-output.txt"),
        ],
        cwd=tmp_path,
    )
    assert "super-secret" not in command
    assert "password@" not in command
    assert str(tmp_path) not in command
    assert identifiers or "redacted" in command
    assert caveats

    encoded = bounded_evidence_for_round(
        {
            "observations": [
                {
                    "argv": [sys.executable, "--password", "secret-value"],
                    "outcome": "failed",
                    "provenance": "parent-observed",
                    "environment": "unknown",
                    "diagnostic": "secret-value",
                }
            ]
        }
    )
    assert encoded is not None
    assert "secret-value" not in encoded
    assert str(tmp_path) not in encoded


def test_broker_authenticates_turn_and_forwards_only_snapshot_environment(tmp_path):
    observed: dict[str, object] = {}

    def execute(argv, cwd, timeout, environment, stream):
        observed.update(argv=argv, cwd=cwd, timeout=timeout, environment=environment)
        stream("broker output\n")
        return SimpleNamespace(outcome="passed", returncode=0, elapsed_seconds=0.01, output_tail="")

    server = BrokerServer(
        root=tmp_path,
        turn_id="turn-761",
        execute=execute,
    ).start()
    try:
        environment = {
            **server.environment,
            "AGENT_LOOP_INVOCATION_ID": "turn-761",
            "PATH": "/usr/bin",
            "TEST_VARIABLE": "kept",
        }
        result = BrokerClient(environment).run(
            [sys.executable, "-c", "pass"],
            timeout_seconds=5,
            cwd=tmp_path,
        )
        assert result.outcome == "passed"
        forwarded = observed["environment"]
        assert isinstance(forwarded, dict)
        assert forwarded["AGENT_LOOP_INVOCATION_ID"] == "turn-761"
        assert forwarded["TEST_VARIABLE"] == "kept"
        assert not any(name.startswith("AGENT_LOOP_TEST_BROKER_") for name in forwarded)

        bad = dict(environment)
        bad["AGENT_LOOP_TEST_BROKER_CAPABILITY"] = "wrong"
        with pytest.raises(Exception):
            BrokerClient(bad).run([sys.executable, "-c", "pass"], timeout_seconds=5, cwd=tmp_path)
    finally:
        server.stop()


def test_runner_broker_preserves_turn_binding_and_collects_receipt(tmp_path):
    from coding_review_agent_loop.containment import default_policy
    from coding_review_agent_loop.runner import Runner

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    (tmp_path / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "initial"], check=True)
    runner = Runner(containment_policy=default_policy(mode="off", cache_dir=tmp_path / ".runtime"))
    script = (
        "from coding_review_agent_loop.cli import main; import sys; "
        "raise SystemExit(main(['run-tests','--timeout-seconds','5','--',"
        "sys.executable,'-c','print(123)']))"
    )
    result = runner.run_with_log(
        [sys.executable, "-c", script], cwd=tmp_path, log_path=tmp_path / "coder.log",
        label="coder", progress_interval_seconds=1,
    )
    assert result.returncode == 0
    observations = runner.local_test_observations()
    assert len(observations) == 1
    assert observations[0].outcome == "passed"
    assert observations[0].turn_id


def test_runner_exceptional_teardown_preserves_broker_journal(tmp_path):
    from coding_review_agent_loop.containment import default_policy
    from coding_review_agent_loop.runner import Runner

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    runner = Runner(containment_policy=default_policy(mode="off", cache_dir=tmp_path / ".runtime"))
    script = (
        "from coding_review_agent_loop.cli import main; import sys; "
        "assert main(['run-tests','--timeout-seconds','5','--',sys.executable,'-c','print(1)']) == 0; "
        "raise SystemExit(9)"
    )
    with pytest.raises(Exception, match="Command failed"):
        runner.run_with_log(
            [sys.executable, "-c", script], cwd=tmp_path, log_path=tmp_path / "coder.log",
            label="coder", progress_interval_seconds=1,
        )
    assert [item.outcome for item in runner.local_test_observations()] == ["passed"]


def test_confined_cwd_rejects_final_symlink_and_outside_escape(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    inside = root / "inside"
    inside.mkdir()
    (inside / "nested").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "safe-link").symlink_to(inside, target_is_directory=True)
    (root / "final-link").symlink_to(inside, target_is_directory=True)
    (root / "escape").symlink_to(outside, target_is_directory=True)

    original_cwd = Path.cwd()
    try:
        with open_confined_cwd(root, root / "safe-link" / "nested") as confined:
            assert confined.path == inside / "nested"
            confined.fchdir()
            assert Path.cwd() == inside / "nested"
    finally:
        os.chdir(original_cwd)
    with pytest.raises(Exception):
        open_confined_cwd(root, root / "final-link")
    with pytest.raises(Exception):
        open_confined_cwd(root, root / "escape" / "missing")


def test_confined_cwd_repeated_name_does_not_treat_intermediate_as_final(tmp_path):
    root = tmp_path / "checkout"
    final = root / "target" / "b" / "a"
    final.mkdir(parents=True)
    (root / "a").symlink_to(root / "target", target_is_directory=True)
    with open_confined_cwd(root, root / "a" / "b" / "a") as confined:
        assert confined.path == final


def test_base_reproduction_requires_clean_complete_stable_snapshots():
    from coding_review_agent_loop.local_test_evidence import TrackedTreeSnapshot

    snapshot = TrackedTreeSnapshot(
        root="/checkout",
        head="base",
        digest="all-a",
        tracked_digest="tracked-a",
        status_clean=True,
        complete=True,
        stable=True,
    )
    result = attribute_base_reproduction(snapshot, snapshot, base_commit="base")
    assert result.state == "base-reproduction"

    dirty = TrackedTreeSnapshot(
        **{**snapshot.__dict__, "status_clean": False}
    )
    assert attribute_base_reproduction(dirty, dirty, base_commit="base").state == "unknown"
