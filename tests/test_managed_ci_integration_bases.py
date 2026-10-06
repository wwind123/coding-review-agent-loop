"""Tool-side trusted integration bases (#1285)."""

import json
from pathlib import Path

import pytest

import coding_review_agent_loop.managed_ci as managed_ci
from coding_review_agent_loop.config import resolve_base_branch
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import IssueContext
from coding_review_agent_loop.integration_close import (
    close_child_after_integration_merge,
    require_child_closed_or_report,
)
from coding_review_agent_loop.managed_ci import QUALIFICATION_MARKER as QUALIFICATION_MARKER_TEXT
from coding_review_agent_loop.managed_ci import (
    QUALIFIED_LABEL,
    TRUSTED_BASES_MARKER,
    ManagedCiContract,
    _contract_dispatch_ref,
    _is_v2_intent_run,
    enforce_trusted_base_at_startup,
    require_recorded_base,
    trusted_base_problem,
)
from coding_review_agent_loop.runner import CommandResult, Runner

from agent_loop_helpers import make_config

WORKFLOW = f"{TRUSTED_BASES_MARKER}: enabled\nAGENT_LOOP_MANAGED_CI_V2: enabled\n"


def intent_record(base, *, pr=7, **overrides):
    """A complete, schema-valid managed handoff record (what the real writer posts)."""
    record = {
        "version": 2, "repository": "OWNER/REPO", "pr": pr, "expected_head_sha": "a" * 40,
        "base_ref": base, "workflow_revision": "b" * 40, "generation": "gen-1",
        "nonce": "n" * 32, "created_at": 1700000000, "state": "dispatch-requested",
        "run_id": None, "run_attempt": None, "terminal_run_id": None,
        "terminal_run_attempt": None, "terminal_outcome": None, "terminal_attempts": [],
    }
    record.update(overrides)
    return record


def intent_comment(base, *, login="agent-loop", user_id=1, pr=7, body=None, **overrides):
    record = intent_record(base, pr=pr, **overrides)
    text = body if body is not None else (
        "<!-- AGENT_MANAGED_CI_INTENT_V2 " + json.dumps(record, separators=(",", ":")) + " -->"
    )
    return {
        "id": 900, "created_at": "2026-01-01T00:00:00Z", "user": {"login": login, "id": user_id},
        "body": text,
    }


class ScriptRunner(Runner):
    """Answers ``gh api <endpoint>`` calls from a dict; records every command."""

    def __init__(self, *, default="main", variable="refactor/*", variable_error=None,
                 workflow=WORKFLOW, pr=None, issue_state="open", close_rc=0, merged=True,
                 authorized="refactor/1181", comments=None):
        super().__init__(dry_run=False)
        self.commands: list[list[str]] = []
        self.default = default
        self.variable = variable
        self.variable_error = variable_error
        self.workflow = workflow
        self.issue_state = issue_state
        self.close_rc = close_rc
        # Durable evidence: the trusted actor's signed handoff record naming the authorized base.
        self.comments = comments if comments is not None else (
            [intent_comment(authorized)] if authorized else []
        )
        self.pr = pr if pr is not None else {
            "number": 7, "merged": merged, "merged_at": "2026-01-01T00:00:00Z" if merged else None,
            "base": {"ref": "refactor/1181", "repo": {"full_name": "OWNER/REPO"}},
            "head": {"sha": "abc"}, "body": "Closes #42",
        }

    def run(self, args, *, cwd, input_text=None, check=True, env=None):
        cmd = [str(a) for a in args]
        self.commands.append(cmd)
        ok = lambda out: CommandResult(cmd, Path(cwd), out, "", 0)  # noqa: E731
        if cmd[:3] == ["gh", "issue", "close"]:
            return CommandResult(cmd, Path(cwd), "", "boom" if self.close_rc else "", self.close_rc)
        if cmd[:3] == ["gh", "pr", "view"]:
            return ok("abc\n")
        if cmd[:2] != ["gh", "api"]:
            return ok("")
        endpoint = cmd[2]
        if endpoint == "repos/OWNER/REPO":
            return ok(json.dumps({"default_branch": self.default, "full_name": "OWNER/REPO"}))
        if endpoint.endswith("/actions/variables/AGENT_LOOP_TRUSTED_BASES"):
            if self.variable_error:
                return CommandResult(cmd, Path(cwd), "", f"gh: Forbidden (HTTP {self.variable_error})", 1)
            if self.variable is None:
                return CommandResult(cmd, Path(cwd), "", "gh: Not Found (HTTP 404)", 1)
            return ok(json.dumps({"value": self.variable}))
        if "/contents/.github/workflows/" in endpoint:
            return ok(self.workflow) if self.workflow is not None else CommandResult(cmd, Path(cwd), "", "gh: Not Found (HTTP 404)", 1)
        if endpoint.startswith("repos/OWNER/REPO/issues/7/comments?"):
            return ok(json.dumps(self.comments if "page=1" in endpoint else []))
        if endpoint.endswith("/actions/variables/AGENT_LOOP_MANAGED_ACTOR"):
            return ok(json.dumps({"value": "agent-loop"}))
        if endpoint == "users/agent-loop":
            return ok(json.dumps({"login": "agent-loop", "id": 1}))
        if endpoint == "repos/OWNER/REPO/pulls/7":
            return ok(json.dumps(self.pr))
        if endpoint == "repos/OWNER/REPO/issues/42":
            return ok(json.dumps({"state": self.issue_state}))
        return ok("{}")

    def commands_matching(self, *prefix):
        return [c for c in self.commands if c[: len(prefix)] == list(prefix)]


def _config(tmp_path, **overrides):
    values = {"managed_ci": True, "managed_ci_trusted_actor": "agent-loop", "base": "refactor/1181"}
    values.update(overrides)
    return make_config(tmp_path, **values)


# --- startup refusal (row startup-refusal) -------------------------------------------


def test_trusted_base_starts(tmp_path):
    runner = ScriptRunner()
    enforce_trusted_base_at_startup(runner, _config(tmp_path))
    assert resolve_base_branch(_config(tmp_path), runner).base == "refactor/1181"


@pytest.mark.parametrize(
    ("kwargs", "cause"),
    [
        ({"variable": "other/*"}, "not trusted"),
        ({"variable": None}, "currently unset"),
        ({"variable": "refactor/**"}, "not trusted"),
        ({"variable_error": 403}, "could not be read"),
        ({"workflow": "AGENT_LOOP_MANAGED_CI_V2: enabled\n"}, TRUSTED_BASES_MARKER),
        ({"workflow": None}, TRUSTED_BASES_MARKER),
    ],
)
def test_untrusted_unreadable_or_unsupported_base_is_refused_before_any_work(tmp_path, kwargs, cause):
    runner = ScriptRunner(**kwargs)
    with pytest.raises(AgentLoopError, match="AGENT_LOOP_TRUSTED_BASES") as raised:
        resolve_base_branch(_config(tmp_path), runner)
    assert "refactor/1181" in str(raised.value) and cause in str(raised.value)
    # Read-only probes only: no label, comment, dispatch or agent call.
    assert all(c[:2] == ["gh", "api"] and "--method" not in c for c in runner.commands)


def test_default_branch_and_non_managed_runs_are_never_refused(tmp_path):
    runner = ScriptRunner(variable=None, workflow=None)
    enforce_trusted_base_at_startup(runner, _config(tmp_path, base="main"))
    enforce_trusted_base_at_startup(runner, _config(tmp_path, managed_ci=False, auto_merge=False))
    enforce_trusted_base_at_startup(runner, _config(tmp_path, dry_run=True))


def test_trusted_base_problem_never_assumes_trust_on_read_failure(tmp_path):
    runner = ScriptRunner(variable_error=403)
    problem = trusted_base_problem(
        runner, gh_cmd="gh", repo="OWNER/REPO", cwd=tmp_path, base="refactor/1",
        default_branch="main", workflow_text=WORKFLOW,
    )
    assert problem and "AGENT_LOOP_TRUSTED_BASES" in problem and "never assumed" in problem


# --- merge boundary (row merge-base-retarget) ----------------------------------------


def _guard(runner, tmp_path, **kwargs):
    values = dict(
        config=_config(tmp_path), pr_number=7, base_ref="refactor/1181",
        repository="OWNER/REPO", head_sha="abc", action="merge",
    )
    values.update(kwargs)
    return require_recorded_base(runner, **values)


def test_guard_accepts_the_recorded_trusted_base(tmp_path):
    runner = ScriptRunner()
    _guard(runner, tmp_path)
    assert not runner.commands_matching("gh", "api", "--method")


@pytest.mark.parametrize(
    ("live_base", "variable", "message"),
    [
        ("refactor/other", "refactor/*", "differs from the qualified base"),
        ("feature/x", "refactor/*", "differs from the qualified base"),
        ("main", "refactor/*", "differs from the qualified base"),
        ("refactor/1181", "other/*", "is not trusted"),
        ("refactor/1181", None, "is not trusted"),
    ],
)
def test_guard_refuses_retarget_or_revocation_and_releases_the_label(tmp_path, live_base, variable, message):
    pr = {
        "number": 7, "base": {"ref": live_base, "repo": {"full_name": "OWNER/REPO"}},
        "head": {"sha": "abc"},
    }
    runner = ScriptRunner(pr=pr, variable=variable)
    with pytest.raises(AgentLoopError, match=message) as raised:
        _guard(runner, tmp_path)
    assert "re-qualify" in str(raised.value)
    deletes = [c for c in runner.commands if "DELETE" in c and any(QUALIFIED_LABEL in part for part in c)]
    assert len(deletes) == 1


def test_guard_refuses_a_moved_head_and_a_foreign_base_repository(tmp_path):
    moved = ScriptRunner(pr={"base": {"ref": "refactor/1181", "repo": {"full_name": "OWNER/REPO"}}, "head": {"sha": "zzz"}})
    with pytest.raises(AgentLoopError, match="live head differs"):
        _guard(moved, tmp_path)
    foreign = ScriptRunner(pr={"base": {"ref": "refactor/1181", "repo": {"full_name": "EVIL/REPO"}}, "head": {"sha": "abc"}})
    with pytest.raises(AgentLoopError, match="base repository differs"):
        _guard(foreign, tmp_path)


def test_default_branch_base_needs_no_variable(tmp_path):
    pr = {"base": {"ref": "main", "repo": {"full_name": "OWNER/REPO"}}, "head": {"sha": "abc"}}
    runner = ScriptRunner(pr=pr, variable=None)
    _guard(runner, tmp_path, base_ref="main")


def test_prepare_v2_merge_and_manual_qualification_use_the_guard(tmp_path):
    pr = {"base": {"ref": "feature/x", "repo": {"full_name": "OWNER/REPO"}}, "head": {"sha": "abc"}, "draft": True, "labels": []}
    contract = ManagedCiContract(
        protocol_version=2, base_ref="refactor/1181", dispatch_ref="main",
        trusted_actor_login="agent-loop", trusted_actor_id=1, repository="OWNER/REPO",
    )
    runner = ScriptRunner(pr=pr)
    with pytest.raises(AgentLoopError, match="readiness for merge refused"):
        managed_ci.prepare_v2_merge(
            runner, config=_config(tmp_path), pr_number=7, expected_head_sha="abc", contract=contract
        )
    # Nothing was readied or labeled as qualified before the refusal.
    assert not runner.commands_matching("gh", "pr", "ready")
    assert not [c for c in runner.commands if "POST" in c]
    runner = ScriptRunner(pr=pr)
    with pytest.raises(AgentLoopError, match="manual qualification refused"):
        managed_ci._publish_manual_v2_qualification(
            runner, config=_config(tmp_path), pr_number=7, expected_head_sha="abc",
            contract=contract, reviewers=("Codex",),
        )
    assert not runner.commands_matching("gh", "pr", "ready")


# --- dispatch ref (row tool-dispatch-from-default / resume-legacy-contract) ----------


def test_workflow_source_ref_is_the_default_branch_not_the_base(tmp_path):
    runner = ScriptRunner()
    assert managed_ci._workflow_source_ref(runner, _config(tmp_path), "refactor/1181") == "main"


def test_legacy_contract_without_dispatch_ref_resumes_only_on_the_default_branch(tmp_path):
    config = _config(tmp_path)
    ok = ManagedCiContract(protocol_version=2, base_ref="main")
    assert _contract_dispatch_ref(ScriptRunner(), config, ok) == "main"
    assert ok.dispatch_ref == "main"
    with pytest.raises(AgentLoopError, match="never dispatched from a\nnon-default ref|non-default ref"):
        _contract_dispatch_ref(ScriptRunner(), config, ManagedCiContract(protocol_version=2, base_ref="refactor/1181"))
    assert _contract_dispatch_ref(
        ScriptRunner(), config,
        ManagedCiContract(protocol_version=2, base_ref="refactor/1181", dispatch_ref="main"),
    ) == "main"


def test_run_discovery_matches_the_dispatch_ref_not_the_base():
    contract = ManagedCiContract(
        protocol_version=2, base_ref="refactor/1181", dispatch_ref="main", nonce="n",
        workflow_revision="a" * 40, trusted_actor_login="agent-loop", trusted_actor_id=1,
    )
    name = managed_ci._v2_run_name(contract)
    run = {"name": name, "event": "workflow_dispatch", "head_branch": "main", "head_sha": "a" * 40}
    assert _is_v2_intent_run(run, contract=contract, run_name=name)
    assert not _is_v2_intent_run({**run, "head_branch": "refactor/1181"}, contract=contract, run_name=name)


# --- child issue close (row close-child-non-default) ---------------------------------


def _issue_context():
    return IssueContext(number=42, repo="OWNER/REPO", title="child", body="", url=None, comments=())


def test_child_is_closed_with_a_comment_after_a_confirmed_integration_merge(tmp_path):
    runner = ScriptRunner()
    outcome = close_child_after_integration_merge(
        runner, config=_config(tmp_path), issue_context=_issue_context(), pr_number=7
    )
    assert outcome == "closed"
    (close,) = runner.commands_matching("gh", "issue", "close")
    assert close[3] == "42" and "PR #7 merged into `refactor/1181`" in close[close.index("--comment") + 1]


def test_close_is_a_noop_when_already_closed_and_never_for_default_or_unmerged(tmp_path):
    cfg, ctx = _config(tmp_path), _issue_context()
    closed = ScriptRunner(issue_state="closed")
    assert close_child_after_integration_merge(closed, config=cfg, issue_context=ctx, pr_number=7) == "already-closed"
    assert not closed.commands_matching("gh", "issue", "close")
    default_merge = ScriptRunner(authorized="main", pr={
        "merged": True, "merged_at": "t", "base": {"ref": "main", "repo": {"full_name": "OWNER/REPO"}}, "body": "Closes #42",
    })
    assert close_child_after_integration_merge(default_merge, config=cfg, issue_context=ctx, pr_number=7) == "default-branch"
    unmerged = ScriptRunner(merged=False)
    assert close_child_after_integration_merge(unmerged, config=cfg, issue_context=ctx, pr_number=7) == "not-merged"
    no_reference = ScriptRunner(pr={
        "merged": True, "merged_at": "t", "base": {"ref": "refactor/1181", "repo": {"full_name": "OWNER/REPO"}}, "body": "Refs #42",
    })
    assert close_child_after_integration_merge(no_reference, config=cfg, issue_context=ctx, pr_number=7) == "unresolved"
    for runner in (default_merge, unmerged, no_reference):
        assert not runner.commands_matching("gh", "issue", "close")


def test_unreadable_post_merge_evidence_is_unresolved_never_complete(tmp_path, capsys):
    cfg, ctx = _config(tmp_path), _issue_context()

    class Broken(ScriptRunner):
        def __init__(self, fail, **kw):
            super().__init__(**kw)
            self.fail = fail

        def run(self, args, *, cwd, **kw):
            cmd = [str(a) for a in args]
            if cmd[:2] == ["gh", "api"] and cmd[2] == self.fail:
                self.commands.append(cmd)
                return CommandResult(cmd, Path(cwd), "", "gh: boom (HTTP 502)", 1)
            return super().run(args, cwd=cwd, **kw)

    for fail in ("repos/OWNER/REPO/pulls/7", "repos/OWNER/REPO", "repos/OWNER/REPO/issues/42"):
        runner = Broken(fail)
        assert close_child_after_integration_merge(runner, config=cfg, issue_context=ctx, pr_number=7) == "unresolved"
        assert not require_child_closed_or_report(runner, config=cfg, issue_context=ctx, pr_number=7)
        assert not runner.commands_matching("gh", "issue", "close")
    assert "could not be" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("merged_base", "variable", "recorded"),
    [
        ("refactor/other", "refactor/*", "refactor/1181"),   # a different trusted base than recorded
        ("feature/x", "refactor/*", None),                    # an untrusted base
        ("refactor/1181", None, None),                        # allow-list revoked
    ],
)
def test_close_authenticates_the_recorded_base_before_mutating(tmp_path, merged_base, variable, recorded):
    pr = {
        "merged": True, "merged_at": "t", "body": "Closes #42",
        "base": {"ref": merged_base, "repo": {"full_name": "OWNER/REPO"}},
    }
    runner = ScriptRunner(pr=pr, variable=variable)
    outcome = close_child_after_integration_merge(
        runner, config=_config(tmp_path, base="refactor/1181", base_provenance="explicit"),
        issue_context=_issue_context(), pr_number=7, expected_base=recorded,
    )
    assert outcome == "unresolved"
    assert not runner.commands_matching("gh", "issue", "close")


def test_failed_close_is_reported_unresolved_and_never_replays_the_merge(tmp_path, capsys):
    runner = ScriptRunner(close_rc=1)
    assert not require_child_closed_or_report(
        runner, config=_config(tmp_path), issue_context=_issue_context(), pr_number=7
    )
    assert "could not be confirmed" in capsys.readouterr().out
    assert not runner.commands_matching("gh", "pr", "merge")
    # A later run retries the closure from the authenticated MERGED PR.
    retry = ScriptRunner()
    assert require_child_closed_or_report(
        retry, config=_config(tmp_path), issue_context=_issue_context(), pr_number=7
    )
    assert not retry.commands_matching("gh", "pr", "merge")


def test_preflight_reports_the_same_refusal_read_only(tmp_path):
    runner = ScriptRunner(variable="other/*")
    context = managed_ci.ManagedCiProbeContext("OWNER/REPO", "gh", tmp_path)
    result = managed_ci.evaluate_managed_ci_readiness(
        runner, context=context, base="refactor/1181", trusted_actor="agent-loop"
    )
    assert result.state == "invalid"
    assert any("AGENT_LOOP_TRUSTED_BASES" in reason for reason in result.reasons)
    assert "AGENT_LOOP_TRUSTED_BASES" in managed_ci.render_managed_ci_preflight(
        result, repo="OWNER/REPO", base="refactor/1181", trusted_actor="agent-loop"
    )
    # Read-only: GETs only, and the workflow was read from the default branch.
    assert all("--method" not in c for c in runner.commands)
    assert any("contents/.github/workflows/ci.yml?ref=main" in c[2] for c in runner.commands if len(c) > 2)
    assert not any("ref=refactor/1181" in " ".join(c) for c in runner.commands)


def test_dispatch_reads_revision_and_dispatches_from_the_default_branch(tmp_path):
    contract = ManagedCiContract(
        protocol_version=2, base_ref="refactor/1181", dispatch_ref="main",
        trusted_actor_login="agent-loop", trusted_actor_id=1, workflow_revision="a" * 40,
        nonce="n" * 32,
    )
    runner = ScriptRunner()
    # The revision moved on the default branch: refuse before any intent or dispatch.
    original = runner.run

    def run(args, **kwargs):
        if list(args)[2:3] == ["repos/OWNER/REPO/commits/main"]:
            runner.commands.append([str(a) for a in args])
            return CommandResult([str(a) for a in args], Path(kwargs["cwd"]), json.dumps({"sha": "b" * 40}), "", 0)
        return original(args, **kwargs)

    runner.run = run
    with pytest.raises(AgentLoopError, match="workflow revision changed"):
        managed_ci._dispatch_v2_qualification(
            runner, config=_config(tmp_path), pr_number=7, expected_head_sha="abc", contract=contract
        )
    reads = [c[2] for c in runner.commands if len(c) > 2 and "/commits/" in c[2]]
    assert reads == ["repos/OWNER/REPO/commits/main"]
    assert not [c for c in runner.commands if any("dispatches" in part for part in c)]


# --- fail closed without an established default branch (round-1 item 1) ----------------


class _NoMetadata(ScriptRunner):
    """Repository metadata is unreadable; an integration workflow differs from the default."""

    def run(self, args, *, cwd, **kw):
        cmd = [str(a) for a in args]
        if cmd[:3] == ["gh", "api", "repos/OWNER/REPO"]:
            self.commands.append(cmd)
            return CommandResult(cmd, Path(cwd), "", "gh: Bad Gateway (HTTP 502)", 1)
        return super().run(args, cwd=cwd, **kw)


@pytest.mark.parametrize("base", ["refactor/1181", "main"])
def test_unreadable_repository_metadata_refuses_startup_for_every_base(tmp_path, base):
    runner = _NoMetadata()
    with pytest.raises(AgentLoopError, match="AGENT_LOOP_TRUSTED_BASES"):
        resolve_base_branch(_config(tmp_path, base=base), runner)
    assert all("--method" not in c for c in runner.commands)


def test_workflow_source_and_trust_never_fall_back_to_the_integration_base(tmp_path):
    runner = _NoMetadata()
    with pytest.raises(AgentLoopError, match="cannot be established"):
        managed_ci._workflow_source_ref(runner, _config(tmp_path), "refactor/1181")
    assert trusted_base_problem(
        runner, gh_cmd="gh", repo="OWNER/REPO", cwd=tmp_path, base="refactor/1181",
        default_branch=None, workflow_text=WORKFLOW,
    )
    # No workflow or revision read was ever issued against the integration branch.
    assert not any("refactor/1181" in " ".join(c) for c in runner.commands)


def test_activation_paths_read_only_from_the_default_branch_or_fail_closed(tmp_path):
    from coding_review_agent_loop.github import PullRequestMetadata

    metadata = PullRequestMetadata(
        number=7, repo="OWNER/REPO", title="t", head_branch="agent-loop/managed-1",
        base_branch="refactor/1181", head_sha="abc", url="u", body="",
    )
    for runner in (ScriptRunner(), _NoMetadata()):
        config = _config(tmp_path)
        try:
            managed_ci.activate_managed_ci(runner, config=config, pr_number=7, metadata=metadata)
        except AgentLoopError:
            pass
        reads = [" ".join(c) for c in runner.commands]
        assert not any("ref=refactor/1181" in r or "commits/refactor/1181" in r for r in reads)
        assert not [c for c in runner.commands if any("dispatches" in part for part in c)]


def test_preflight_creation_reads_the_workflow_from_the_default_branch(tmp_path):
    runner = ScriptRunner()
    try:
        managed_ci.preflight_managed_ci_creation(runner, config=_config(tmp_path), issue_number=42)
    except AgentLoopError:
        pass
    reads = [c[2] for c in runner.commands if len(c) > 2 and "/contents/.github/workflows/" in c[2]]
    assert reads and all(r.endswith("ref=main") for r in reads)


# --- interrupted closure (round-1 items 2-4) --------------------------------------------


def test_issue_resume_reconciles_a_merged_integration_pr_without_replaying_the_merge(tmp_path, monkeypatch, capsys):
    import coding_review_agent_loop.issue_pr_handoff as handoff
    from coding_review_agent_loop.integration_close import reconcile_merged_integration_child

    class Auth:
        pr_number, state = 7, "MERGED"

    monkeypatch.setattr(handoff, "authenticate_canonical_issue_pr", lambda *a, **k: Auth())
    runner = ScriptRunner(issue_state="open")
    assert reconcile_merged_integration_child(
        runner, config=_config(tmp_path), issue_number=42, issue_context=_issue_context()
    )
    assert runner.commands_matching("gh", "issue", "close") and not runner.commands_matching("gh", "pr", "merge")
    # A failed close never reports completion; an OPEN PR or default-branch merge defers to normal resume.
    failing = ScriptRunner(close_rc=1)
    with pytest.raises(AgentLoopError, match="could not be authenticated"):
        reconcile_merged_integration_child(
            failing, config=_config(tmp_path), issue_number=42, issue_context=_issue_context()
        )
    Auth.state = "OPEN"
    assert not reconcile_merged_integration_child(
        ScriptRunner(), config=_config(tmp_path), issue_number=42, issue_context=_issue_context()
    )


def test_run_issue_loop_reconciles_before_canonical_pr_resolution():
    import inspect

    import coding_review_agent_loop.issue_loop as issue_loop

    source = inspect.getsource(issue_loop.run_issue_loop)
    reconcile = source.index("reconcile_merged_integration_child(")
    # Before every route, including the staged direct-child dispatch, so the
    # approved-implementation resume path (which rejects MERGED) is never reached first.
    assert reconcile < source.index("_resolve_fresh_child_provenance(")
    assert reconcile < source.index("_dispatch_decomposition_child(")
    assert reconcile < source.index("resolve_canonical_pr_for_issue(")


def test_parent_completion_is_not_recorded_when_post_merge_evidence_is_unreadable(tmp_path, monkeypatch):
    import coding_review_agent_loop.execution_policy as policy

    def explode(*a, **k):
        raise AssertionError("parent completion must not proceed")

    monkeypatch.setattr(policy, "get_issue_context", explode)
    monkeypatch.setattr(policy, "resolve_staged_phase_progress", explode)
    monkeypatch.setattr(policy, "record_staged_completion", explode)

    class PrUnreadable(ScriptRunner):
        def run(self, args, *, cwd, **kw):
            cmd = [str(a) for a in args]
            if cmd[:3] == ["gh", "api", "repos/OWNER/REPO/pulls/7"]:
                return CommandResult(cmd, Path(cwd), "", "gh: boom (HTTP 502)", 1)
            return super().run(args, cwd=cwd, **kw)

    policy._record_staged_parent_completion_after_merge(
        PrUnreadable(), config=_config(tmp_path), issue_context=_issue_context(), pr_number=7
    )


# --- orchestration rows through the real boundaries (round-1 item 5) -----------------------


class SeqRunner(ScriptRunner):
    """pulls/7 answers from a sequence (last repeated); other endpoints as ScriptRunner."""

    def __init__(self, prs, **kw):
        super().__init__(**kw)
        self.prs = list(prs)

    def run(self, args, *, cwd, **kw):
        cmd = [str(a) for a in args]
        if cmd[:3] == ["gh", "api", "repos/OWNER/REPO/pulls/7"]:
            self.commands.append(cmd)
            pr = self.prs.pop(0) if len(self.prs) > 1 else self.prs[0]
            return CommandResult(cmd, Path(cwd), json.dumps(pr), "", 0)
        return super().run(args, cwd=cwd, **kw)


def _live(base):
    return {"state": "open", "draft": False, "labels": [], "head": {"sha": "abc"},
            "base": {"ref": base, "repo": {"full_name": "OWNER/REPO"}}}


@pytest.mark.parametrize(
    ("after", "variable"),
    [("refactor/b", "refactor/*"), ("feature/x", "refactor/*"), ("main", "refactor/*"), ("refactor/a", "other/*")],
    ids=["trusted-retarget", "untrusted-retarget", "default-retarget", "revoked"],
)
def test_merge_boundary_refuses_a_change_after_the_initial_check(tmp_path, after, variable):
    from coding_review_agent_loop.pr_loop_support import ExactHeadCiProof, _merge_with_exact_head_proof

    # The intent was granted for refactor/a; the live base changes between the first and the final read.
    runner = SeqRunner([_live(after)], variable="refactor/*" if variable == "refactor/*" else variable)
    runner.variable = variable
    # Allow-list revocation leaves the base itself unchanged.
    if after == "refactor/a":
        runner.prs = [_live("refactor/a")]
    with pytest.raises(AgentLoopError, match="merge refused"):
        _merge_with_exact_head_proof(
            runner, config=_config(tmp_path), pr_number=7,
            proof=ExactHeadCiProof(head_sha="abc", source="managed exact-head", base_ref="refactor/a", repository="OWNER/REPO"),
        )
    assert not [c for c in runner.commands if "merge" in " ".join(c) and c[:2] != ["gh", "api"]]
    assert not [c for c in runner.commands if "--method" in c and "merge" in " ".join(c)]
    assert any("DELETE" in c and any(QUALIFIED_LABEL in part for part in c) for c in runner.commands)


@pytest.mark.parametrize("adopted", [False, True], ids=["issue-created", "adopted"])
def test_manual_qualification_refuses_a_retarget_after_the_initial_check(tmp_path, adopted):
    contract = ManagedCiContract(
        protocol_version=2, base_ref="refactor/a", dispatch_ref="main",
        trusted_actor_login="agent-loop", trusted_actor_id=1, repository="OWNER/REPO",
        adopted_existing_pr=adopted, issue_created_pr=not adopted,
    )
    # Reads: labels, initial guard (both still refactor/a), then the recorded-base guard sees feature/x.
    runner = SeqRunner([_live("refactor/a"), _live("refactor/a"), _live("feature/x")], variable="refactor/*")
    with pytest.raises(AgentLoopError, match="manual qualification refused"):
        managed_ci._publish_manual_v2_qualification(
            runner, config=_config(tmp_path, base="refactor/a"), pr_number=7, expected_head_sha="abc",
            contract=contract, reviewers=("Codex",),
        )
    assert not runner.commands_matching("gh", "pr", "ready")
    assert not [c for c in runner.commands if any(QUALIFICATION_MARKER_TEXT in part for part in c)]


# --- authorized base recovered from authenticated durable evidence (round-2 item 4) --------


def _merged(base, body="Closes #42"):
    return {"merged": True, "merged_at": "t", "body": body, "base": {"ref": base, "repo": {"full_name": "OWNER/REPO"}}}


@pytest.mark.parametrize(
    ("label", "runner_kwargs", "expected"),
    [
        # No --base on the rerun: the merge target must still equal the authorized base.
        ("different-trusted-target", dict(pr=_merged("refactor/other"), authorized="refactor/1181"), "unresolved"),
        ("default-target-vs-integration-record", dict(pr=_merged("main"), authorized="refactor/1181"), "unresolved"),
        ("integration-merge-vs-default-record", dict(pr=_merged("refactor/1181"), authorized="main"), "unresolved"),
        ("no-durable-evidence", dict(pr=_merged("refactor/1181"), authorized=None), "unresolved"),
        ("forged-by-other-author", dict(pr=_merged("refactor/1181"), authorized=None,
                                        comments=[intent_comment("refactor/1181", login="mallory", user_id=9)]), "unresolved"),
        ("right-login-wrong-id", dict(pr=_merged("refactor/1181"), authorized=None,
                                      comments=[intent_comment("refactor/1181", user_id=99)]), "unresolved"),
        ("contradictory-records", dict(pr=_merged("refactor/1181"), authorized=None,
                                       comments=[intent_comment("refactor/1181"), intent_comment("refactor/other")]), "unresolved"),
        ("authorized-and-merged-agree", dict(pr=_merged("refactor/1181"), authorized="refactor/1181"), "closed"),
        ("unmanaged-default-merge", dict(pr=_merged("main"), authorized=None), "default-branch"),
    ],
)
def test_recovery_authenticates_the_originally_authorized_base_without_an_explicit_base(
    tmp_path, label, runner_kwargs, expected
):
    runner = ScriptRunner(**runner_kwargs)
    # A parent/child rerun carries no explicit --base (provenance is the repository default).
    config = _config(tmp_path, base="main", base_provenance="repository-default")
    outcome = close_child_after_integration_merge(
        runner, config=config, issue_context=_issue_context(), pr_number=7
    )
    assert outcome == expected, label
    assert bool(runner.commands_matching("gh", "issue", "close")) == (expected == "closed")


def test_a_later_explicit_base_cannot_stand_in_for_the_authorization(tmp_path):
    runner = ScriptRunner(pr=_merged("refactor/other"), authorized="refactor/1181")
    config = _config(tmp_path, base="refactor/other", base_provenance="explicit")
    assert close_child_after_integration_merge(
        runner, config=config, issue_context=_issue_context(), pr_number=7
    ) == "unresolved"
    assert not runner.commands_matching("gh", "issue", "close")


# --- canonical envelope/schema evidence only; mentions never authorize or block (round-3 items 4, 7) ---

_EMBEDDED = (
    "Example for reviewers:\n```\n<!-- AGENT_MANAGED_CI_INTENT_V2 "
    + json.dumps(intent_record("refactor/1181")) + " -->\n```"
)


@pytest.mark.parametrize(
    ("label", "comments"),
    [
        ("embedded-example-in-prose", [intent_comment("refactor/1181", body=_EMBEDDED)]),
        ("partial-four-field-record", [intent_comment("refactor/1181", body=(
            '<!-- AGENT_MANAGED_CI_INTENT_V2 {"version":2,"repository":"OWNER/REPO","pr":7,'
            '"base_ref":"refactor/1181"} -->'))]),
        ("wrong-version", [intent_comment("refactor/1181", version=1)]),
        ("bad-lifecycle", [intent_comment("refactor/1181", state="exploded")]),
        ("malformed-json", [intent_comment("refactor/1181", body="<!-- AGENT_MANAGED_CI_INTENT_V2 {nope} -->")]),
        ("wrong-pr", [intent_comment("refactor/1181", pr=8)]),
        ("wrong-repository", [intent_comment("refactor/1181", repository="EVIL/REPO")]),
    ],
)
def test_non_canonical_or_invalid_evidence_never_authorizes_integration_closure(tmp_path, label, comments):
    runner = ScriptRunner(pr=_merged("refactor/1181"), comments=comments)
    config = _config(tmp_path, base="main", base_provenance="repository-default")
    outcome = close_child_after_integration_merge(
        runner, config=config, issue_context=_issue_context(), pr_number=7
    )
    assert outcome == "unresolved", label
    assert not runner.commands_matching("gh", "issue", "close")


def test_a_complete_valid_record_authorizes_and_footered_form_is_accepted(tmp_path):
    footer = "\n\n---\n_Generated by [Claude Code](https://claude.ai/code)_"
    record = intent_comment("refactor/1181")
    record["body"] += footer
    runner = ScriptRunner(pr=_merged("refactor/1181"), comments=[record])
    config = _config(tmp_path, base="main", base_provenance="repository-default")
    assert close_child_after_integration_merge(
        runner, config=config, issue_context=_issue_context(), pr_number=7
    ) == "closed"


@pytest.mark.parametrize(
    ("label", "comments", "variable_unset"),
    [
        ("unrelated-author-mention", [{
            "id": 5, "created_at": "2026-01-01T00:00:00Z", "user": {"login": "reviewer", "id": 9},
            "body": "FYI AGENT_MANAGED_CI_INTENT_V2 records are posted by the bot.",
        }], False),
        ("unrelated-author-envelope", [intent_comment("refactor/1181", login="mallory", user_id=9)], False),
        ("no-comments", [], False),
        ("unrelated-envelope-with-actor-variable-unset",
         [intent_comment("refactor/1181", login="mallory", user_id=9)], True),
        ("trusted-looking-envelope-with-actor-variable-unset",
         [intent_comment("refactor/1181")], True),
        ("mention-with-actor-variable-unset", [{
            "id": 5, "created_at": "2026-01-01T00:00:00Z", "user": {"login": "reviewer", "id": 9},
            "body": "AGENT_MANAGED_CI_INTENT_V2",
        }], True),
    ],
)
def test_default_branch_completion_ignores_unauthenticated_mentions(tmp_path, label, comments, variable_unset):
    class NoActorVariable(ScriptRunner):
        def run(self, args, *, cwd, **kw):
            cmd = [str(a) for a in args]
            if variable_unset and cmd[:2] == ["gh", "api"] and cmd[2].endswith("/actions/variables/AGENT_LOOP_MANAGED_ACTOR"):
                self.commands.append(cmd)
                return CommandResult(cmd, Path(cwd), "", "gh: Not Found (HTTP 404)", 1)
            return super().run(args, cwd=cwd, **kw)

    runner = NoActorVariable(pr=_merged("main"), comments=comments)
    config = _config(tmp_path, base="main", base_provenance="repository-default")
    outcome = close_child_after_integration_merge(
        runner, config=config, issue_context=_issue_context(), pr_number=7
    )
    assert outcome == "default-branch", label
    assert not runner.commands_matching("gh", "issue", "close")
    # ...but the same unauthenticated material never authorizes a non-default close,
    # and a genuine recorded integration base still rejects a default-branch merge.
    integration = NoActorVariable(pr=_merged("refactor/1181"), comments=comments)
    assert close_child_after_integration_merge(
        integration, config=config, issue_context=_issue_context(), pr_number=7
    ) == "unresolved"
    assert not integration.commands_matching("gh", "issue", "close")


def test_authenticated_integration_record_still_rejects_a_default_branch_merge(tmp_path):
    runner = ScriptRunner(pr=_merged("main"), authorized="refactor/1181")
    config = _config(tmp_path, base="main", base_provenance="repository-default")
    assert close_child_after_integration_merge(
        runner, config=config, issue_context=_issue_context(), pr_number=7
    ) == "unresolved"


def test_unreadable_actor_variable_still_fails_closed_for_envelopes(tmp_path):
    class ForbiddenVariable(ScriptRunner):
        def run(self, args, *, cwd, **kw):
            cmd = [str(a) for a in args]
            if cmd[:2] == ["gh", "api"] and cmd[2].endswith("/actions/variables/AGENT_LOOP_MANAGED_ACTOR"):
                self.commands.append(cmd)
                return CommandResult(cmd, Path(cwd), "", "gh: Forbidden (HTTP 403)", 1)
            return super().run(args, cwd=cwd, **kw)

    runner = ForbiddenVariable(pr=_merged("main"), comments=[intent_comment("refactor/1181", login="mallory", user_id=9)])
    config = _config(tmp_path, base="main", base_provenance="repository-default", managed_ci_trusted_actor=None)
    assert close_child_after_integration_merge(
        runner, config=config, issue_context=_issue_context(), pr_number=7
    ) == "unresolved"


def test_staged_completion_guard_accepts_default_merge_with_unset_actor_variable(tmp_path):
    from coding_review_agent_loop.integration_close import require_child_closed_or_report

    class NoActor(ScriptRunner):
        def run(self, args, *, cwd, **kw):
            cmd = [str(a) for a in args]
            if cmd[:2] == ["gh", "api"] and cmd[2].endswith("/actions/variables/AGENT_LOOP_MANAGED_ACTOR"):
                self.commands.append(cmd)
                return CommandResult(cmd, Path(cwd), "", "gh: Not Found (HTTP 404)", 1)
            return super().run(args, cwd=cwd, **kw)

    runner = NoActor(pr=_merged("main"), comments=[intent_comment("refactor/1181", login="mallory", user_id=9)], issue_state="closed")
    config = _config(tmp_path, base="main", base_provenance="repository-default")
    # The real guard the staged-completion hook uses: complete, nothing closed or posted.
    assert require_child_closed_or_report(
        runner, config=config, issue_context=_issue_context(), pr_number=7
    )
    assert not runner.commands_matching("gh", "issue", "close")
