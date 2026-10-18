"""PR-loop tests for coder_ci_guard: the coder CI-dispatch guard (#1542)."""

import datetime
import json
from types import SimpleNamespace

import pytest

from coding_review_agent_loop.runner import CommandResult
from coding_review_agent_loop.cli import AgentLoopError, run_pr_loop
from coding_review_agent_loop.comment_rendering import (
    render_risk_test_matrix_section,
)
from coding_review_agent_loop.github import (
    PullRequestCheck,
)
from coding_review_agent_loop.local_test_evidence import (
    LocalTestObservation,
)
from coding_review_agent_loop.managed_ci import (
    ManagedCiContract,
    ManagedCiOutcome,
)
from coding_review_agent_loop.agent_failure import ValidatedAgentResponse
from coding_review_agent_loop.protocol import (
    parse_risk_test_matrix,
    risk_test_matrix_identity,
    validate_structured_coder_followup,
)
from agent_loop_helpers import (
    FakeRunner,
    make_config,
    structured_coder_followup,
    structured_pr_review,
)
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.execution_policy as execution_policy
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.plan_first_loop as plan_first_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
import coding_review_agent_loop.response_validation as response_validation
import coding_review_agent_loop.triage as triage
import coding_review_agent_loop.validated_agent as validated_agent
from agent_loop_helpers import patch_attr_in
from pr_loop_test_helpers import (
    _followup_observation,
    _init_followup_checkout,
    _watch_check_board,
)


# ---- coder CI-dispatch guard and post-turn audit (#1366) ----

import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402

_GUARD_DIR_ENV = "AGENT_LOOP_CODER_GUARD_DIR"


class _GuardedCoderRunner(FakeRunner):
    """Records launch envs, has the coder try an empty-input ordinary dispatch
    through whatever ``gh`` its env installs, seeds the coder turn's managed
    full-suite receipt, records orchestrator dispatches, and scripts the
    dispatch audit."""

    def __init__(self, *, audit_runs=(), audit_returncode=0, seed_receipt=True,
                 audit_failing_queries=(), dispatch_returncode=0, first_dispatch_runs=None,
                 **kwargs):
        super().__init__(**kwargs)
        # Query substrings whose listing fails (#1414).
        self.audit_failing_queries = tuple(audit_failing_queries)
        self.dispatch_returncode = dispatch_returncode
        # Called with the dispatched head on the first orchestrator dispatch;
        # the runs it returns become visible to later audits (#1414).
        self.first_dispatch_runs = first_dispatch_runs
        self.launch_envs = []
        self.coder_dispatch_attempts = []
        self.coder_turn_receipts = []
        self.audit_queries = []
        self.orchestrator_dispatches = []
        self.audit_runs = list(audit_runs)
        self.audit_returncode = audit_returncode
        self.seed_receipt = seed_receipt
        self._coder_turns = 0

    def run_with_log(self, args, *, cwd, log_path, label, progress_interval_seconds,
                     check=True, env=None, input_text=None, use_pty=False,
                     timeout_seconds=None, **_kwargs):
        cmd = [str(arg) for arg in args]
        merged = {**os.environ, **(env or {})}
        self.launch_envs.append((cmd[0], dict(env or {})))
        if cmd[:1] == ["claude"]:
            if self.seed_receipt:
                # The coder's managed `agent-loop run-tests` full suite passed
                # in this turn: a real broker receipt in the turn catalog.
                self._coder_turns += 1
                turn_id = f"coder-turn-{self._coder_turns}"
                self._latest_test_turn_id = turn_id
                self._local_test_observations.append(LocalTestObservation(
                    command=("python", "-m", "pytest", "tests/", "-q"),
                    outcome="passed",
                    provenance="parent-observed",
                    receipt_id=f"full-suite-{turn_id}",
                    turn_id=turn_id,
                    timestamp="2026-10-11T22:00:00+00:00",
                    cwd=str(cwd),
                ))
            self.coder_turn_receipts.append(tuple(self.current_test_turn_observations()))
            # The coder follow-up: a skipped synchronize check, a local
            # full-suite receipt, and no GitHub green check -- it reaches for
            # `gh workflow run` with empty inputs, as on PR #1364.
            gh = shutil.which("gh", path=merged.get("PATH", ""))
            self.coder_dispatch_attempts.append(subprocess.run(
                [gh, "workflow", "run", "ci.yml", "--ref", self.pr_payload["headRefName"]],
                env=merged, capture_output=True, text=True, timeout=60,
            ))
        return super().run_with_log(
            args, cwd=cwd, log_path=log_path, label=label,
            progress_interval_seconds=progress_interval_seconds, check=check, env=env,
            input_text=input_text, use_pty=use_pty, timeout_seconds=timeout_seconds,
        )

    def run(self, args, *, cwd, input_text=None, check=True, env=None):
        cmd = [str(arg) for arg in args]
        if cmd[:3] == ["gh", "api", "--paginate"] and "/actions/runs?" in cmd[3]:
            self.audit_queries.append(cmd[3])
            cmd, cwd_path = self._record_command(cmd, cwd)
            if self.audit_returncode or any(part in cmd[3] for part in self.audit_failing_queries):
                return CommandResult(cmd, cwd_path, "", "HTTP 502", self.audit_returncode or 1)
            payload = {"total_count": len(self.audit_runs), "workflow_runs": self.audit_runs}
            return CommandResult(cmd, cwd_path, json.dumps(payload), "", 0)
        if cmd[:2] == ["gh", "api"] and any(part.endswith("/dispatches") for part in cmd):
            # The orchestrator's real exact-head dispatch, at the GitHub boundary.
            self.orchestrator_dispatches.append((cmd, dict(env or {})))
            if self.first_dispatch_runs is not None and len(self.orchestrator_dispatches) == 1:
                head = next(
                    part.split("=", 1)[1] for part in cmd
                    if part.startswith("inputs[expected_head_sha]=")
                )
                self.audit_runs.extend(self.first_dispatch_runs(head))
            cmd, cwd_path = self._record_command(cmd, cwd)
            if self.dispatch_returncode:
                return CommandResult(cmd, cwd_path, "", "HTTP 500", self.dispatch_returncode)
            return CommandResult(cmd, cwd_path, "", "", 0)
        return super().run(args, cwd=cwd, input_text=input_text, check=check, env=env)


def _fake_real_gh(tmp_path, monkeypatch):
    from coding_review_agent_loop import scratch

    bin_dir = tmp_path / "real-gh-bin"
    bin_dir.mkdir()
    record = bin_dir / "calls.txt"
    gh = bin_dir / "gh"
    gh.write_text(f"#!/bin/sh\necho \"$@\" >> '{record}'\nexit 0\n", encoding="utf-8")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setattr(scratch, "scratch_root", lambda: tmp_path / "scratch")
    return record


_V2_NONCE = "nonce-1366" + "x" * 22


def _install_v2_dispatch_boundary(monkeypatch):
    """A v2 contract whose real dispatcher runs to its nonce-bearing POST.

    Only the intent-ledger bookkeeping around the dispatch is stubbed; the
    real ``dispatch_final_qualification`` and ``_dispatch_v2_qualification``
    build and send the workflow dispatch through the runner boundary.
    """
    from coding_review_agent_loop import managed_ci as managed_ci_module

    contract = ManagedCiContract(
        protocol_version=2, base_ref="main", dispatch_ref="main",
        trusted_actor_login="agent-loop", trusted_actor_id=1, nonce=_V2_NONCE,
    )
    discovered = []
    monkeypatch.setattr(managed_ci_module, "_contract_dispatch_ref", lambda *_a, **_k: "main")
    real_api_json = managed_ci_module._api_json
    monkeypatch.setattr(
        managed_ci_module, "_api_json",
        lambda runner, config, path, **kw: (
            {} if path.endswith("/commits/main") else real_api_json(runner, config, path, **kw)
        ),
    )
    def mint_intent(*_args, contract, **_kwargs):
        # The loop resets the generation per head; the ledger mints its nonce.
        if contract.nonce is None:
            contract.nonce = _V2_NONCE

    monkeypatch.setattr(managed_ci_module, "_ensure_v2_intent", mint_intent)
    monkeypatch.setattr(managed_ci_module, "_patch_intent", lambda *_a, **_k: None)
    monkeypatch.setattr(
        managed_ci_module, "_require_workflow_dispatch_authorization", lambda *_a, **_k: None
    )
    # Nothing is visible before the dispatch; the dispatched run after it.
    monkeypatch.setattr(
        managed_ci_module, "_discover_v2_run",
        lambda *_a, **_k: (discovered.append(1) and None) if len(discovered) == 0 else (555, 1),
    )
    monkeypatch.setattr(
        pr_loop, "publish_manual_v2_qualification",
        lambda *_a, **kwargs: kwargs["expected_head_sha"],
    )
    return contract


def _managed_guard_loop(
    tmp_path, monkeypatch, *, managed=True, mode="implicit", audit_runs=(), audit_returncode=0,
    claude_outputs=None, antigravity_outputs=None, real_repair=False, expect_error=None,
):
    record = _fake_real_gh(tmp_path, monkeypatch)
    runner = _GuardedCoderRunner(
        audit_runs=audit_runs,
        audit_returncode=audit_returncode,
        codex_outputs=[
            structured_pr_review(
                state="blocking", summary="Fix it.", blocking_items=["Fix the edge case."],
            ),
            structured_pr_review(
                state="approved", summary="Fixed.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=claude_outputs or [structured_coder_followup(
            addressed_items=["item-1"], tests_run=["python -m pytest tests/ -q"],
        )],
        antigravity_outputs=antigravity_outputs or [],
        pr_payload={"headRefName": "agent-loop/managed-77", "headRefOid": "abc123", "baseRefName": "main"},
        pr_check_runs_payload={"total_count": 1, "check_runs": [{
            "id": 1, "name": "ci", "status": "completed", "conclusion": "skipped",
            "app": {"slug": "github-actions"}, "check_suite": {"id": 5},
        }]},
    )
    if real_repair:
        # Undo the autouse repair stub so the configured repair chain runs.
        from coding_review_agent_loop import validated_agent
        from coding_review_agent_loop.agents import antigravity as agy_mod

        monkeypatch.setattr(validated_agent, "attempt_repair", validated_agent._ORIGINAL_ATTEMPT_REPAIR)
        monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: tmp_path / "agy-settings.json")
    contract = None
    if managed and mode in {"implicit", "explicit"}:
        contract = ManagedCiContract()
    elif managed and mode == "v2":
        contract = _install_v2_dispatch_boundary(monkeypatch)
    elif managed and mode == "revalidation-drop":
        contract = ManagedCiContract(adopted_existing_pr=True, base_ref="main")
        monkeypatch.setattr(pr_loop, "revalidate_adopted_managed_ci", lambda *_a, **_k: False)
        monkeypatch.setattr(pr_loop, "release_adopted_managed_ci", lambda *_a, **_k: True)
    monkeypatch.setattr(pr_loop, "activate_managed_ci", lambda *_a, **_k: contract)
    monkeypatch.setattr(
        pr_loop, "wait_for_final_qualification",
        lambda *_a, **_k: ManagedCiOutcome(status="passed", head_sha=runner.pr_payload["headRefOid"]),
    )
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *_a, **_k: None)
    explicit = (
        {"managed_ci": True, "managed_ci_trusted_actor": "agent-loop"}
        if mode in {"explicit", "v2"} else {}
    )
    config = make_config(
        # v2 runs as explicit managed CI without a merge: the merge gate is
        # out of scope, the dispatch is what this regression observes.
        tmp_path, auto_merge=managed and mode != "v2", reviewer=("codex",), max_rounds=2,
        **explicit,
    )
    started = datetime.datetime.now(datetime.timezone.utc)
    if expect_error is not None or (managed and contract is None) or mode == "revalidation-drop":
        # Activation fallback / revalidation drop end in the ordinary watcher,
        # which this fake board never satisfies; the coder turn already ran.
        with pytest.raises(AgentLoopError):
            run_pr_loop(runner, pr_number=77, config=config)
    else:
        assert run_pr_loop(runner, pr_number=77, config=config) == 0
    return runner, record, runner.orchestrator_dispatches, started


def _coder_comment(runner):
    return next(
        comment["body"] for comment in runner.pr_payload.get("comments", [])
        if '"kind": "coder_followup"' in comment["body"] or "Updated the PR." in comment["body"]
    )


def _assert_single_v2_dispatch(runner, dispatches, head):
    """Exactly one nonce-correlated v2 exact-head dispatch, after approval."""
    _assert_single_correlated_dispatch(runner, dispatches, head)
    cmd, _env = dispatches[0]
    assert "inputs[protocol_version]=2" in cmd
    assert f"inputs[managed_nonce]={_V2_NONCE}" in cmd
    assert "ref=main" in cmd


def _assert_single_correlated_dispatch(runner, dispatches, head):
    """The real v1 dispatcher posted exactly one exact-head workflow dispatch,
    correlated by PR number and expected head, outside every agent env."""
    assert len(dispatches) == 1
    cmd, env = dispatches[0]
    assert cmd[:4] == ["gh", "api", "--method", "POST"]
    assert cmd[4].endswith("/dispatches")
    assert f"inputs[expected_head_sha]={head}" in cmd
    assert "inputs[pr_number]=77" in cmd
    assert "ref=agent-loop/managed-77" in cmd or "inputs[managed_nonce]=" in " ".join(cmd)
    assert "AGENT_LOOP_CODER_CI_DISPATCH_GUARD" not in env
    # Dispatched only after the approving review of that head.
    names = [cmd[0] if cmd[:2] != ["codex", "exec"] else "review" for cmd, _cwd in runner.commands]
    last_review = max(index for index, name in enumerate(names) if name == "review")
    dispatch_index = next(
        index for index, (cmd, _cwd) in enumerate(runner.commands)
        if cmd[:2] == ["gh", "api"] and cmd[4:5] and cmd[4].endswith("/dispatches")
    )
    assert dispatch_index > last_review


@pytest.mark.parametrize("mode", ["implicit", "explicit"])
def test_managed_coder_dispatch_is_refused_before_approval_and_orchestrator_dispatches_after(
    tmp_path, monkeypatch, mode,
):
    runner, record, dispatches, _ = _managed_guard_loop(tmp_path, monkeypatch, mode=mode)

    # The coder held a passing managed full-suite receipt in its turn catalog
    # while the only GitHub check was the skipped synchronize run.
    [receipts] = runner.coder_turn_receipts
    assert [(obs.command, obs.outcome) for obs in receipts] == [
        (("python", "-m", "pytest", "tests/", "-q"), "passed")
    ]
    # Its empty-input ordinary dispatch was refused by the guard and never
    # reached the real gh: no workflow_dispatch before approval.
    [attempt] = runner.coder_dispatch_attempts
    assert attempt.returncode == 3
    assert "refused `gh workflow run`" in attempt.stderr
    assert not record.exists()
    assert not any(cmd[:3] == ["gh", "workflow", "run"] for cmd, _cwd in runner.commands)
    # Guard env on the coder launch only; reviewer launches stay unguarded.
    coder_envs = [env for name, env in runner.launch_envs if name == "claude"]
    reviewer_envs = [env for name, env in runner.launch_envs if name == "codex"]
    assert coder_envs and all(env.get("AGENT_LOOP_CODER_CI_DISPATCH_GUARD") == "managed" for env in coder_envs)
    assert reviewer_envs and not any(_GUARD_DIR_ENV in env for env in reviewer_envs)
    # The coder prompt carries the matching guidance.
    coder_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "This pull request uses managed CI." in coder_prompt
    # After every reviewer approves, the orchestrator's real correlated
    # exact-head dispatch still happens exactly once.
    _assert_single_correlated_dispatch(runner, dispatches, "abc123-coder-1")
    # Nothing was dispatched by the coder, so the audit adds no note.
    assert runner.audit_queries
    assert "Coder-turn CI dispatch audit" not in _coder_comment(runner)


def test_unmanaged_coder_turn_has_no_guard_prompt_or_audit(tmp_path, monkeypatch):
    runner, record, dispatches, _ = _managed_guard_loop(tmp_path, monkeypatch, managed=False)

    [attempt] = runner.coder_dispatch_attempts
    # Without managed CI the coder's gh is the real one (guard not installed).
    assert attempt.returncode == 0
    assert record.read_text(encoding="utf-8").startswith("workflow run ci.yml")
    assert not any(_GUARD_DIR_ENV in env for _name, env in runner.launch_envs)
    coder_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "This pull request uses managed CI." not in coder_prompt
    assert runner.audit_queries == []
    assert dispatches == []


def _dispatch_run(
    run_id, head_sha, *, created_at, name="CI", branch="agent-loop/managed-77",
    path=".github/workflows/ci.yml", head_branch=None,
):
    return {
        "id": run_id, "name": name, "display_title": name, "event": "workflow_dispatch",
        "head_sha": head_sha, "head_branch": head_branch or branch, "created_at": created_at,
        "path": path,
        "html_url": f"https://github.com/OWNER/REPO/actions/runs/{run_id}",
        "repository": {"full_name": "OWNER/REPO"}, "check_suite_id": run_id + 1,
    }


def test_bypass_dispatch_on_intermediate_head_is_reported_once_as_diagnostic(tmp_path, monkeypatch):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    runs = [
        # A bypass (absolute gh / curl) on intermediate head B, superseded by a later push.
        _dispatch_run(38032663783, "intermediate-head-b", created_at=now),
        # Managed runs and runs older than the turn are never attributed.
        _dispatch_run(900, "abc123-coder-1", created_at=now, name="managed-ci-v2 nonce=x"),
        _dispatch_run(901, "abc123", created_at="2020-01-01T00:00:00Z"),
    ]
    runner, _record, dispatches, _ = _managed_guard_loop(tmp_path, monkeypatch, audit_runs=runs)

    body = _coder_comment(runner)
    assert "ordinary workflow_dispatch created during coder turn before managed approval" in body
    assert "non-authoritative diagnostic, not qualification" in body
    # Branch-scoped plus per-head lookups return it repeatedly; it is reported once.
    assert body.count("run 38032663783 on `intermediate-head-b`") == 1
    assert "run 900 " not in body and "run 901 " not in body
    assert any("branch=agent-loop%2Fmanaged-77" in query and "created=%3E%3D" in query
               for query in runner.audit_queries)
    # Never treated as qualification: the managed dispatch still runs after approval.
    _assert_single_correlated_dispatch(runner, dispatches, "abc123-coder-1")


def test_dispatch_audit_lookup_failure_is_only_a_diagnostic_note(tmp_path, monkeypatch):
    runner, _record, dispatches, _ = _managed_guard_loop(
        tmp_path, monkeypatch, audit_returncode=1,
    )

    body = _coder_comment(runner)
    assert "diagnostic only, does not block this round" in body
    assert "HTTP 502" in body
    assert len(dispatches) == 1


def test_audit_coder_turn_dispatches_unit_filters_and_dedupes(tmp_path):
    from coding_review_agent_loop.github import (
        audit_coder_turn_dispatches,
        render_coder_dispatch_audit_note,
    )

    started = datetime.datetime(2026, 10, 11, 12, 0, tzinfo=datetime.timezone.utc)
    runs = [
        _dispatch_run(1, "b", created_at="2026-10-11T12:00:10Z"),
        _dispatch_run(2, "c", created_at="2026-10-11T11:59:50Z"),  # inside skew
        _dispatch_run(3, "c", created_at="2026-10-11T11:00:00Z"),  # before the turn
        _dispatch_run(4, "c", created_at="2026-10-11T12:01:00Z", name="managed-ci-v2 nonce=n"),
        {**_dispatch_run(5, "c", created_at="2026-10-11T12:01:00Z"), "repository": {"full_name": "x/y"}},
        {**_dispatch_run(6, "c", created_at="2026-10-11T12:01:00Z"), "event": "pull_request"},
    ]
    runner = _GuardedCoderRunner(audit_runs=runs)
    audit = audit_coder_turn_dispatches(
        runner, make_config(tmp_path), branch="agent-loop/managed-77",
        head_shas=("a", "c", "c", None), turn_started_at=started,
    )
    assert [run.run_id for run in audit.runs] == [1, 2]
    assert audit.lookup_notes == ()
    # One branch query plus one per distinct known head.
    assert len(runner.audit_queries) == 3
    note = render_coder_dispatch_audit_note(audit)
    assert "run 1 on `b` (https://github.com/OWNER/REPO/actions/runs/1)" in note
    assert render_coder_dispatch_audit_note(type(audit)()) == ""


# ---- orchestrator v1 self-dispatch correlation in the coder-turn audit (#1414) ----

_UTC = datetime.timezone.utc


def _at(clock):
    hour, minute, second = (int(part) for part in clock.split(":"))
    return datetime.datetime(2026, 10, 11, hour, minute, second, tzinfo=_UTC)


def _v1_record(requested, returned, *, head="rec-head", workflow_file="ci.yml"):
    from coding_review_agent_loop.managed_ci import ManagedV1DispatchRecord

    return ManagedV1DispatchRecord(
        workflow_file=workflow_file, ref="agent-loop/managed-77", pr_number=77,
        expected_head_sha=head, requested_at=_at(requested), returned_at=_at(returned),
    )


def _audit_1414(tmp_path, runs, *, turn, records=(), failing=()):
    from coding_review_agent_loop.github import audit_coder_turn_dispatches

    runner = _GuardedCoderRunner(audit_runs=runs, audit_failing_queries=failing)
    audit = audit_coder_turn_dispatches(
        runner, make_config(tmp_path), branch="agent-loop/managed-77",
        head_shas=("known-head",), turn_started_at=_at(turn),
        orchestrator_dispatches=records,
    )
    return runner, audit


def _ambiguous(audit):
    return {entry.run.run_id: entry.reasons for entry in audit.ambiguous_runs}


def test_coder_turn_audit_claims_the_sole_matching_orchestrator_v1_run(tmp_path):
    from coding_review_agent_loop.github import render_coder_dispatch_audit_note

    run = _dispatch_run(10, "rec-head", created_at="2026-10-11T12:00:00Z")
    record = _v1_record("12:00:00", "12:00:01")
    # (a) Inside the turn skew window and the record's only match: claimed.
    runner, audit = _audit_1414(tmp_path, [run], turn="12:00:10", records=(record,))
    assert audit.runs == () and audit.ambiguous_runs == ()
    # (g) The branch and per-head listings both return it; deduplicated once.
    assert len(runner.audit_queries) == 3
    assert [r.run_id for r in audit.excluded_orchestrator_runs] == [10]
    assert render_coder_dispatch_audit_note(audit) == ""
    # (b) Without a record the same run is reported exactly as before.
    _runner, unrecorded = _audit_1414(tmp_path, [run], turn="12:00:10")
    assert [r.run_id for r in unrecorded.runs] == [10]
    assert unrecorded.ambiguous_runs == () and unrecorded.excluded_orchestrator_runs == ()


def test_coder_turn_audit_never_picks_an_owner_by_chronological_order(tmp_path):
    # (c) An ordinary run before the orchestrator run; both match the record.
    runs = [
        _dispatch_run(11, "rec-head", created_at="2026-10-11T11:59:55Z"),
        _dispatch_run(12, "rec-head", created_at="2026-10-11T12:00:00Z"),
    ]
    _runner, audit = _audit_1414(
        tmp_path, runs, turn="12:00:10", records=(_v1_record("12:00:00", "12:00:01"),),
    )
    assert audit.runs == ()
    assert audit.excluded_orchestrator_runs == ()
    assert _ambiguous(audit) == {11: ("multiple-matches",), 12: ("multiple-matches",)}


def test_coder_turn_audit_later_coder_duplicate_makes_the_record_ambiguous(tmp_path):
    runs = [
        _dispatch_run(13, "rec-head", created_at="2026-10-11T12:00:01Z"),
        _dispatch_run(14, "rec-head", created_at="2026-10-11T12:00:20Z"),
    ]
    _runner, audit = _audit_1414(
        tmp_path, runs, turn="12:00:10", records=(_v1_record("12:00:00", "12:00:01"),),
    )
    assert audit.runs == () and audit.excluded_orchestrator_runs == ()
    assert _ambiguous(audit) == {13: ("multiple-matches",), 14: ("multiple-matches",)}


def test_coder_turn_audit_pre_cutoff_record_run_never_hides_an_in_window_duplicate(tmp_path):
    # (d) Association runs before the turn cutoff over the widened discovery.
    runs = [
        _dispatch_run(15, "rec-head", created_at="2026-10-11T12:00:01Z"),
        _dispatch_run(16, "rec-head", created_at="2026-10-11T12:00:36Z"),
    ]
    record = _v1_record("12:00:00", "12:00:10")
    _runner, audit = _audit_1414(tmp_path, runs, turn="12:00:35", records=(record,))
    assert audit.runs == () and audit.excluded_orchestrator_runs == ()
    assert _ambiguous(audit) == {16: ("multiple-matches",)}
    # (e) The orchestrator run alone, before the cutoff: claimed, nothing reported.
    _runner, alone = _audit_1414(tmp_path, runs[:1], turn="12:00:35", records=(record,))
    assert alone.runs == () and alone.ambiguous_runs == ()
    assert [r.run_id for r in alone.excluded_orchestrator_runs] == [15]


def test_coder_turn_audit_reports_runs_that_do_not_match_the_record(tmp_path):
    # (f) The record's workflow is managed.yml, so ci.yml is "another path".
    runs = [
        _dispatch_run(20, "rec-head", created_at="2026-10-11T12:00:00Z",
                      path=".github/workflows/managed.yml"),
        _dispatch_run(21, "other-head", created_at="2026-10-11T12:00:00Z",
                      path=".github/workflows/managed.yml"),
        _dispatch_run(22, "rec-head", created_at="2026-10-11T12:00:00Z"),
        _dispatch_run(23, "rec-head", created_at="2026-10-11T12:00:35Z",
                      path=".github/workflows/managed.yml"),
    ]
    record = _v1_record("12:00:00", "12:00:01", workflow_file="managed.yml")
    _runner, audit = _audit_1414(tmp_path, runs, turn="12:00:10", records=(record,))
    assert [r.run_id for r in audit.excluded_orchestrator_runs] == [20]
    assert [r.run_id for r in audit.runs] == [21, 22, 23]
    assert audit.ambiguous_runs == ()


def test_coder_turn_audit_record_head_lookup_failure_claims_nothing(tmp_path):
    from coding_review_agent_loop.github import render_coder_dispatch_audit_note

    # (h) The branch listing returns the sole match; the record head's fails.
    run = _dispatch_run(30, "rec-head", created_at="2026-10-11T12:00:00Z")
    _runner, audit = _audit_1414(
        tmp_path, [run], turn="12:00:10", records=(_v1_record("12:00:00", "12:00:01"),),
        failing=("head_sha=rec-head&",),
    )
    assert audit.runs == () and audit.excluded_orchestrator_runs == ()
    assert _ambiguous(audit) == {30: ("incomplete-lookup",)}
    assert len(audit.lookup_notes) == 1 and "head `rec-head`" in audit.lookup_notes[0]
    # (j) Neutral heading; no claim that another run matched.
    note = render_coder_dispatch_audit_note(audit)
    assert "Coder-turn CI dispatch audit (attribution undetermined)" in note
    assert "could not confirm which run the orchestrator created" in note
    assert (
        "run 30 on `rec-head` (https://github.com/OWNER/REPO/actions/runs/30): the run listing "
        "for the dispatched head was unavailable, so the match could not be confirmed"
    ) in note
    assert "more than one run" not in note
    assert "at least one other run" not in note
    assert "ordinary workflow_dispatch created during coder turn" not in note


def test_coder_turn_audit_widens_discovery_to_the_record_window_and_head(tmp_path):
    # (i) The branch floor drops to the record's start minus the skew.
    runner, _audit = _audit_1414(
        tmp_path, [], turn="12:00:10", records=(_v1_record("12:00:00", "12:00:01"),),
    )
    assert any(
        "branch=agent-loop%2Fmanaged-77" in query and "created=%3E%3D2026-10-11T11%3A59%3A30Z" in query
        for query in runner.audit_queries
    )
    assert any("head_sha=rec-head&" in query for query in runner.audit_queries)
    assert any("head_sha=known-head&" in query for query in runner.audit_queries)
    # Without a record the floor is the turn start minus the skew.
    runner, _audit = _audit_1414(tmp_path, [], turn="12:00:10")
    assert any("created=%3E%3D2026-10-11T11%3A59%3A40Z" in query for query in runner.audit_queries)
    assert not any("rec-head" in query for query in runner.audit_queries)


def test_coder_turn_audit_run_shared_by_two_records_is_unattributed(tmp_path):
    from coding_review_agent_loop.github import render_coder_dispatch_audit_note

    run = _dispatch_run(40, "rec-head", created_at="2026-10-11T12:00:03Z")
    records = (_v1_record("12:00:00", "12:00:01"), _v1_record("12:00:05", "12:00:06"))
    _runner, audit = _audit_1414(tmp_path, [run], turn="12:00:10", records=records)
    assert audit.runs == () and audit.excluded_orchestrator_runs == ()
    assert _ambiguous(audit) == {40: ("shared-match",)}
    note = render_coder_dispatch_audit_note(audit)
    assert "this run matches more than one orchestrator dispatch" in note
    assert "more than one run matches" not in note


def test_coder_turn_audit_renders_multiple_match_and_combined_reasons(tmp_path):
    from coding_review_agent_loop.github import (
        CoderTurnAmbiguousRun,
        CoderTurnDispatchAudit,
        CoderTurnDispatchRun,
        render_coder_dispatch_audit_note,
    )

    runs = [
        _dispatch_run(11, "rec-head", created_at="2026-10-11T11:59:55Z"),
        _dispatch_run(12, "rec-head", created_at="2026-10-11T12:00:00Z"),
    ]
    _runner, audit = _audit_1414(
        tmp_path, runs, turn="12:00:10", records=(_v1_record("12:00:00", "12:00:01"),),
    )
    note = render_coder_dispatch_audit_note(audit)
    assert note.count("more than one run matches the same orchestrator dispatch") == 2
    assert "ordinary workflow_dispatch created during coder turn" not in note
    run = CoderTurnDispatchRun(run_id=9, head_sha="h", url=None, created_at="x")
    combined = render_coder_dispatch_audit_note(CoderTurnDispatchAudit(ambiguous_runs=(
        CoderTurnAmbiguousRun(run=run, reasons=("multiple-matches", "incomplete-lookup")),
    )))
    assert (
        "- run 9 on `h`: more than one run matches the same orchestrator dispatch; the run "
        "listing for the dispatched head was unavailable, so the match could not be confirmed"
    ) in combined


def test_rejected_coder_turn_audit_posts_nothing_for_a_claimed_orchestrator_run(tmp_path):
    from coding_review_agent_loop.github import publish_rejected_coder_turn_dispatch_audit

    run = _dispatch_run(50, "rec-head", created_at="2026-10-11T12:00:00Z")
    runner = _GuardedCoderRunner(audit_runs=[run], pr_payload={
        "headRefName": "agent-loop/managed-77", "headRefOid": "rec-head", "baseRefName": "main",
    })
    config = make_config(tmp_path)
    kwargs = dict(
        pr_number=77, branch="agent-loop/managed-77", head_shas=("rec-head",),
        turn_started_at=_at("12:00:10"),
    )
    note = publish_rejected_coder_turn_dispatch_audit(
        runner, config, orchestrator_dispatches=(_v1_record("12:00:00", "12:00:01"),), **kwargs,
    )
    assert note == ""
    assert not runner.pr_payload.get("comments")
    # The same run without the record is still published.
    note = publish_rejected_coder_turn_dispatch_audit(runner, config, **kwargs)
    assert "run 50 on `rec-head`" in note
    assert len(runner.pr_payload["comments"]) == 1


def test_v1_dispatch_records_only_a_successful_post(tmp_path):
    from coding_review_agent_loop.managed_ci import dispatch_final_qualification

    payload = {"headRefName": "agent-loop/managed-77", "headRefOid": "abc123", "baseRefName": "main"}
    config = make_config(tmp_path)
    contract = ManagedCiContract()
    before = datetime.datetime.now(_UTC)
    dispatch_final_qualification(
        _GuardedCoderRunner(pr_payload=dict(payload)), config=config, pr_number=77,
        expected_head_sha="abc123", head_ref="agent-loop/managed-77", contract=contract,
    )
    [record] = contract.v1_dispatch_records
    assert (record.workflow_file, record.ref, record.pr_number, record.expected_head_sha) == (
        "ci.yml", "agent-loop/managed-77", 77, "abc123",
    )
    assert before <= record.requested_at <= record.returned_at <= datetime.datetime.now(_UTC)
    # A failed POST created no run, so it records nothing and still raises.
    failing = ManagedCiContract()
    with pytest.raises(AgentLoopError, match="Unable to dispatch managed final CI"):
        dispatch_final_qualification(
            _GuardedCoderRunner(pr_payload=dict(payload), dispatch_returncode=1), config=config,
            pr_number=77, expected_head_sha="abc123", head_ref="agent-loop/managed-77",
            contract=failing,
        )
    assert failing.v1_dispatch_records == ()
    # The record list stays bounded to the newest entries.
    for _ in range(12):
        dispatch_final_qualification(
            _GuardedCoderRunner(pr_payload=dict(payload)), config=config, pr_number=77,
            expected_head_sha="abc123", head_ref="agent-loop/managed-77", contract=contract,
        )
    assert len(contract.v1_dispatch_records) == 8
    assert contract.v1_dispatch_records[-1].requested_at >= record.requested_at


def test_v2_dispatch_never_appends_a_v1_record(tmp_path, monkeypatch):
    from coding_review_agent_loop.managed_ci import dispatch_final_qualification

    contract = _install_v2_dispatch_boundary(monkeypatch)
    runner = _GuardedCoderRunner(pr_payload={
        "headRefName": "agent-loop/managed-77", "headRefOid": "abc123", "baseRefName": "main",
    })
    dispatch_final_qualification(
        runner, config=make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop"),
        pr_number=77, expected_head_sha="abc123", head_ref="agent-loop/managed-77",
        contract=contract,
    )
    assert len(runner.orchestrator_dispatches) == 1
    assert contract.v1_dispatch_records == ()


def _failed_qualification_fix_loop(
    tmp_path, monkeypatch, *, coder_run=True, claude_outputs=None, antigravity_outputs=(),
    real_repair=False, expect_error=False,
):
    """Approval, a real v1 dispatch, a failed qualification, then a guarded
    coder fix turn whose audit sees the orchestrator's own run (#1414)."""
    from coding_review_agent_loop import pr_loop as pr_loop_module

    _fake_real_gh(tmp_path, monkeypatch)

    def created_by_first_dispatch(head):
        now = datetime.datetime.now(_UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        runs = [_dispatch_run(7001, head, created_at=now)]
        if coder_run:
            runs.append(_dispatch_run(
                7002, "abc123", created_at=now, path=".github/workflows/other.yml",
            ))
        return runs

    runner = _GuardedCoderRunner(
        first_dispatch_runs=created_by_first_dispatch,
        codex_outputs=[
            structured_pr_review(state="approved", summary="Approved."),
            structured_pr_review(
                state="approved", summary="Approved after managed CI fix.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=claude_outputs or [structured_coder_followup(
            state="blocking", summary="Fixed managed CI.", addressed_items=["item-1"],
        )],
        antigravity_outputs=list(antigravity_outputs),
        pr_payload={"headRefName": "agent-loop/managed-77", "headRefOid": "abc123", "baseRefName": "main"},
        pr_check_runs_payload={"total_count": 1, "check_runs": [{
            "id": 1, "name": "ci", "status": "completed", "conclusion": "skipped",
            "app": {"slug": "github-actions"}, "check_suite": {"id": 5},
        }]},
    )
    if real_repair:
        from coding_review_agent_loop import validated_agent
        from coding_review_agent_loop.agents import antigravity as agy_mod

        monkeypatch.setattr(validated_agent, "attempt_repair", validated_agent._ORIGINAL_ATTEMPT_REPAIR)
        monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: tmp_path / "agy-settings.json")
    contract = ManagedCiContract()
    monkeypatch.setattr(pr_loop, "activate_managed_ci", lambda *_a, **_k: contract)
    failed_check = PullRequestCheck(
        name="final-ci/exact-head", kind="check_run", status="failure",
        url="https://github.com/OWNER/REPO/actions/runs/7001",
    )
    outcomes = iter([
        ManagedCiOutcome(
            status="failed", checks=_watch_check_board("failing", failing=(failed_check,)),
            head_sha="abc123",
        ),
        ManagedCiOutcome(status="passed", head_sha="abc123-coder-1"),
    ])
    monkeypatch.setattr(pr_loop, "wait_for_final_qualification", lambda *_a, **_k: next(outcomes))
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *_a, **_k: None)
    rejected_calls = []
    real_rejected = pr_loop_module.publish_rejected_coder_turn_dispatch_audit

    def spy_rejected(*args, **kwargs):
        rejected_calls.append(kwargs)
        return real_rejected(*args, **kwargs)

    monkeypatch.setattr(pr_loop_module, "publish_rejected_coder_turn_dispatch_audit", spy_rejected)
    config = make_config(tmp_path, auto_merge=True, reviewer=("codex",), max_rounds=1)
    if expect_error:
        with pytest.raises(AgentLoopError):
            run_pr_loop(runner, pr_number=77, config=config)
    else:
        assert run_pr_loop(runner, pr_number=77, config=config) == 0
    return runner, contract, rejected_calls


def test_failed_v1_qualification_then_guarded_fix_turn_skips_the_orchestrator_run(
    tmp_path, monkeypatch,
):
    from coding_review_agent_loop import github as github_module

    logged = []
    real_log = github_module.log
    monkeypatch.setattr(
        github_module, "log", lambda config, message: (logged.append(message), real_log(config, message)),
    )
    runner, contract, _rejected = _failed_qualification_fix_loop(tmp_path, monkeypatch)

    # The orchestrator's v1 dispatch on the approved head was recorded.
    assert contract.v1_dispatch_records[0].expected_head_sha == "abc123"
    assert len(runner.orchestrator_dispatches) == 2
    # The guarded fix turn's audit reports only the ordinary coder run.
    body = next(
        comment["body"] for comment in runner.pr_payload.get("comments", [])
        if "## Coder follow-up" in comment["body"]
    )
    assert "ordinary workflow_dispatch created during coder turn before managed approval" in body
    assert "run 7002 on `abc123`" in body
    assert "run 7001" not in body
    assert "attribution undetermined" not in body
    assert any("head_sha=abc123&" in query for query in runner.audit_queries)
    assert (
        "Coder-turn dispatch audit: run 7001 on abc123 matches the orchestrator's own "
        "managed v1 dispatch; not attributed to the coder"
    ) in logged


def test_rejected_guarded_fix_turn_passes_records_and_posts_nothing_for_own_run(
    tmp_path, monkeypatch,
):
    malformed = _prose_before_footer(structured_coder_followup(addressed_items=["item-1"]))
    runner, contract, rejected = _failed_qualification_fix_loop(
        tmp_path, monkeypatch, coder_run=False, claude_outputs=[malformed],
        antigravity_outputs=["still not a response"] * 12, real_repair=True, expect_error=True,
    )

    [call] = rejected
    assert call["orchestrator_dispatches"] == contract.v1_dispatch_records
    assert len(call["orchestrator_dispatches"]) == 1
    assert not any(
        "Coder-turn CI dispatch audit" in comment["body"]
        for comment in runner.pr_payload.get("comments", [])
    )


@pytest.mark.parametrize("mode", ["fallback", "revalidation-drop"])
def test_activation_fallback_or_revalidation_drop_leaves_coder_unguarded(tmp_path, monkeypatch, mode):
    runner, record, dispatches, _ = _managed_guard_loop(tmp_path, monkeypatch, mode=mode)

    # Only the live managed decision installs the guard -- never auto-merge alone.
    assert runner.coder_dispatch_attempts
    assert not any(_GUARD_DIR_ENV in env for _name, env in runner.launch_envs)
    coder_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "This pull request uses managed CI." not in coder_prompt
    assert runner.audit_queries == []
    assert dispatches == []


def test_coder_retry_inherits_the_guard_and_reviewer_stays_unguarded(tmp_path, monkeypatch):
    runner, record, dispatches, _ = _managed_guard_loop(
        tmp_path, monkeypatch,
        claude_outputs=[
            ("Quota exceeded for this project.", 1),
            structured_coder_followup(addressed_items=["item-1"]),
        ],
    )

    names = [name for name, _env in runner.launch_envs]
    coder_envs = [env for name, env in runner.launch_envs if name == "claude"]
    assert len(coder_envs) == 2  # the original attempt and its retry
    assert all(env.get("AGENT_LOOP_CODER_CI_DISPATCH_GUARD") == "managed" for env in coder_envs)
    assert [attempt.returncode for attempt in runner.coder_dispatch_attempts] == [3, 3]
    assert not record.exists()
    # The reviewer launched right after the guarded coder is unguarded.
    last_coder = max(index for index, name in enumerate(names) if name == "claude")
    after = [env for name, env in runner.launch_envs[last_coder + 1:] if name == "codex"]
    assert after and not any(_GUARD_DIR_ENV in env for env in after)
    assert len(dispatches) == 1


def _prose_before_footer(response):
    """Markdown between the JSON object and the footer needs format repair."""
    head, footer, tail = response.partition("<!-- AGENT_STATE:")
    return f"{head.rstrip()}\n## Changes made\nProse the envelope forbids.\n{footer}{tail}"


def test_loop_driven_format_repair_of_guarded_coder_inherits_the_guard(tmp_path, monkeypatch):
    """A malformed guarded coder response reaches the real repair chain; the
    repair subprocess launch carries the guard, the next reviewer does not."""
    malformed = _prose_before_footer(structured_coder_followup(addressed_items=["item-1"]))
    repaired = structured_coder_followup(addressed_items=["item-1"])
    runner, record, dispatches, _ = _managed_guard_loop(
        tmp_path, monkeypatch, claude_outputs=[malformed], antigravity_outputs=[repaired],
        real_repair=True,
    )

    names = [name for name, _env in runner.launch_envs]
    repair_indexes = [
        index for index, (name, env) in enumerate(runner.launch_envs)
        # The `agy models` catalog probe has no role and runs no tools.
        if name == "agy" and "agy-catalog-" not in env.get("AGENT_LOOP_WORKDIR", "")
    ]
    assert repair_indexes, names
    for index in repair_indexes:
        env = runner.launch_envs[index][1]
        assert env.get("AGENT_LOOP_CODER_CI_DISPATCH_GUARD") == "managed"
        assert env["PATH"].split(os.pathsep)[0] == env[_GUARD_DIR_ENV]
    after = [env for name, env in runner.launch_envs[max(repair_indexes) + 1:] if name == "codex"]
    assert after and not any(_GUARD_DIR_ENV in env for env in after)
    assert not record.exists()
    _assert_single_correlated_dispatch(runner, dispatches, runner.pr_payload["headRefOid"])


def test_rejected_guarded_coder_turn_still_publishes_dispatch_audit(tmp_path, monkeypatch):
    """#1366 item-3: a bypass dispatch followed by a response that fails
    validation and repair is still surfaced, without masking the rejection."""
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    malformed = _prose_before_footer(structured_coder_followup(addressed_items=["item-1"]))
    runner, _record, dispatches, _ = _managed_guard_loop(
        tmp_path, monkeypatch, claude_outputs=[malformed],
        antigravity_outputs=["still not a response"] * 12, real_repair=True,
        audit_runs=[_dispatch_run(38032663783, "abc123-coder-1", created_at=now)],
        expect_error=True,
    )

    notes = [
        comment["body"] for comment in runner.pr_payload.get("comments", [])
        if "Coder-turn CI dispatch audit" in comment["body"]
    ]
    assert len(notes) == 1
    assert "run 38032663783 on `abc123-coder-1`" in notes[0]
    assert "non-authoritative diagnostic, not qualification" in notes[0]
    assert "coder response for this turn was rejected" in notes[0]
    assert dispatches == []


def test_v2_guarded_coder_turn_then_nonce_correlated_dispatch_after_approval(tmp_path, monkeypatch):
    """#1366 item-5: the same guarded-coder regression on a v2 contract; the
    real dispatcher sends one nonce-correlated exact-head dispatch after
    approval, outside the agent guard."""
    runner, record, dispatches, _ = _managed_guard_loop(tmp_path, monkeypatch, mode="v2")

    [attempt] = runner.coder_dispatch_attempts
    assert attempt.returncode == 3
    assert not record.exists()
    _assert_single_v2_dispatch(runner, dispatches, "abc123-coder-1")


def _guarded_semantic_correction_loop(
    tmp_path, monkeypatch, *, correction="ok", audit_runs_after_correction=(),
):
    """A managed PR follow-up whose authenticated claims force the bounded
    semantic-evidence correction, launched through the real coder backend
    (#1366 item-8/9/10)."""
    from coding_review_agent_loop.errors import CheckoutVerificationError

    def row(row_id, level):
        return {
            "row_id": row_id, "label": f"Behaviour {row_id}",
            "entry_path_or_mode": "PR follow-up", "initial_state": "blocking review",
            "event": "follow-up authenticated", "expected_outcome": "map rendered",
            "forbidden_side_effects": ["No re-ask."], "proposed_test_level": level,
            "proposed_test_location": "tests/test_orchestrator_pr.py",
            "applicability": "required", "related_scope_item_ids": ["scope-1"],
            "execution_owner": "one-shot",
        }

    matrix = parse_risk_test_matrix({
        "applicability": "applicable",
        "rows": [row("followup-a", "unit")],
        "important_exclusions": ["Planned tests are not evidence."],
    })
    identity = risk_test_matrix_identity(matrix)
    plan_context = execution_policy.make_approved_plan_context(
        "Approved plan.\n\n" + render_risk_test_matrix_section(matrix),
        source_locator="test guarded correction plan",
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest=identity,
    )
    record = _fake_real_gh(tmp_path, monkeypatch)
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=2, auto_merge=True)
    head = _init_followup_checkout(config.claude_dir)
    current = _followup_observation(
        execution_ref="coder-turn:observation-2", receipt_id="receipt-current-head",
        head=head, timestamp="2026-01-01T00:00:01Z",
    )

    def claim(**extra):
        return {
            "row_id": "followup-a", "execution_refs": [current.execution_ref],
            "test_identifiers": ["tests/test_orchestrator_pr.py::test_followup_workflow"],
            "test_locations": ["tests/test_orchestrator_pr.py"],
            "workflow_path_claim": "The real follow-up caller ran.",
            "outcome_assertions": ["It passed."], "forbidden_effect_assertions": ["No re-ask."],
            "caveats": [], "test_level": "unit", **extra,
        }

    raw = structured_coder_followup(addressed_items=["item-1"], summary="Follow-up with correction.")
    payload, end = json.JSONDecoder().raw_decode(raw)
    original = dict(payload)
    # An unknown selector is an actionable diagnostic: it forces the correction.
    original["risk_test_matrix_claims"] = [claim(execution_refs=["coder-turn:not-in-catalog"])]
    text = json.dumps(original) + raw[end:]
    corrected = dict(payload)
    corrected["risk_test_matrix_claims"] = [claim()]
    corrected_text = json.dumps(corrected) + raw[end:]
    parsed = validate_structured_coder_followup(
        text, required_architecture_impact_contract=1,
        delivered_risk_test_matrix=plan_context.risk_test_matrix_payload,
        delivered_risk_test_matrix_identity=identity, required_risk_test_matrix_contract=1,
        delivered_risk_test_matrix_row_ids=("followup-a",),
        execution_catalog=(current,),
    )
    coder_response = ValidatedAgentResponse(
        text=text, session_id="coder-session", marker_value=parsed,
        acquisition_test_turn_id="coder-turn", acquisition_test_observations=(current,),
    )

    class CorrectionRunner(_GuardedCoderRunner):
        correction_launched = False

        def run_with_log(self, args, **kwargs):
            if str(args[0]) == "claude":
                # Only the semantic correction reaches the claude backend here.
                self.correction_launched = True
                if correction == "raise":
                    self.launch_envs.append(("claude", dict(kwargs.get("env") or {})))
                    raise CheckoutVerificationError("assigned checkout corrupted during correction")
            return super().run_with_log(args, **kwargs)

        def run(self, args, *, cwd, input_text=None, check=True, env=None):
            # The bypass run becomes visible only once the correction ran.
            self.audit_runs = list(audit_runs_after_correction) if self.correction_launched else []
            return super().run(args, cwd=cwd, input_text=input_text, check=check, env=env)

    runner = CorrectionRunner(
        seed_receipt=False,
        claude_outputs=[corrected_text],
        codex_outputs=[
            structured_pr_review(state="blocking", summary="Needs work.", blocking_items=["Fix the follow-up."]),
            structured_pr_review(
                state="approved", summary="Covered.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        pr_payload={"headRefOid": "predecessor-head", "headRefName": "agent-loop/managed-77"},
        git_head="predecessor-head",
    )
    coder_configs = []
    real_validated_agent = validated_agent._run_validated_agent

    def fake_validated_agent(*args, **kwargs):
        if kwargs.get("role") == "coder":
            coder_configs.append(kwargs["config"])
            runner.pr_payload["headRefOid"] = head
            runner.simulate_agent_turn(config, config.claude_dir, head=head)
            return coder_response
        return real_validated_agent(*args, **kwargs)

    patch_attr_in(monkeypatch, (
        validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage,
        issue_loop,
    ), "_run_validated_agent", fake_validated_agent)
    monkeypatch.setattr(
        response_validation, "stable_tracked_tree_snapshot",
        lambda _workdir: SimpleNamespace(
            head=head, tracked_digest="tree-current", complete=True, stable=True, status_clean=True,
        ),
    )
    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_implementation, issue_loop,
    ), "_read_assigned_workdir_head", lambda *_a, **_k: head)
    monkeypatch.setattr(pr_loop, "activate_managed_ci", lambda *_a, **_k: ManagedCiContract())
    monkeypatch.setattr(
        pr_loop, "wait_for_final_qualification",
        lambda *_a, **_k: ManagedCiOutcome(status="passed", head_sha=runner.pr_payload["headRefOid"]),
    )
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *_a, **_k: None)
    if correction == "raise":
        with pytest.raises(CheckoutVerificationError):
            run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=plan_context)
    else:
        # The coder comment and the approving review are published first;
        # this fixture models no issue handoff, so final qualification then
        # stops fail-closed.  Dispatch after approval is covered by the
        # v1/v2 guarded-coder regressions above.
        with pytest.raises(AgentLoopError, match="handoff identity is missing"):
            run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=plan_context)
    return runner, record, coder_configs, head


def test_semantic_correction_of_guarded_coder_turn_is_guarded(tmp_path, monkeypatch):
    runner, record, coder_configs, head = _guarded_semantic_correction_loop(tmp_path, monkeypatch)

    assert [cfg.coder_ci_dispatch_guard for cfg in coder_configs] == [True]
    assert runner.correction_launched
    correction_envs = [env for name, env in runner.launch_envs if name == "claude"]
    assert correction_envs
    assert all(env.get("AGENT_LOOP_CODER_CI_DISPATCH_GUARD") == "managed" for env in correction_envs)
    # The correction's `gh workflow run` was refused and never reached GitHub.
    assert [attempt.returncode for attempt in runner.coder_dispatch_attempts] == [3]
    assert not record.exists()
    correction_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "bounded semantic coverage defect" in correction_prompt
    assert "This pull request uses managed CI." in correction_prompt
    # The reviewer launched after the correction stays unguarded.
    last = max(index for index, (name, _env) in enumerate(runner.launch_envs) if name == "claude")
    after = [env for name, env in runner.launch_envs[last + 1:] if name == "codex"]
    assert after and not any(_GUARD_DIR_ENV in env for env in after)
    assert runner.orchestrator_dispatches == []


def test_audit_runs_after_semantic_correction_and_names_its_dispatch_once(tmp_path, monkeypatch):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    runner, _record, _configs, head = _guarded_semantic_correction_loop(
        tmp_path, monkeypatch,
        audit_runs_after_correction=[_dispatch_run(4711, "correction-head", created_at=now)],
    )

    coder_comments = [
        comment["body"] for comment in runner.pr_payload.get("comments", [])
        if "Coder-turn CI dispatch audit" in comment["body"]
    ]
    assert len(coder_comments) == 1
    assert coder_comments[0].count("run 4711 on `correction-head`") == 1
    assert "non-authoritative diagnostic, not qualification" in coder_comments[0]
    assert runner.orchestrator_dispatches == []


def test_rejection_after_semantic_correction_still_publishes_the_audit_note(tmp_path, monkeypatch):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    runner, _record, _configs, _head = _guarded_semantic_correction_loop(
        tmp_path, monkeypatch, correction="raise",
        audit_runs_after_correction=[_dispatch_run(4712, "bypass-head", created_at=now)],
    )

    notes = [
        comment["body"] for comment in runner.pr_payload.get("comments", [])
        if "Coder-turn CI dispatch audit" in comment["body"]
    ]
    assert len(notes) == 1
    assert "run 4712 on `bypass-head`" in notes[0]
    assert "coder response for this turn was rejected" in notes[0]
    assert runner.orchestrator_dispatches == []
