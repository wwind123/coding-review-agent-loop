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


class ScriptRunner(Runner):
    """Answers ``gh api <endpoint>`` calls from a dict; records every command."""

    def __init__(self, *, default="main", variable="refactor/*", variable_error=None,
                 workflow=WORKFLOW, pr=None, issue_state="open", close_rc=0, merged=True):
        super().__init__(dry_run=False)
        self.commands: list[list[str]] = []
        self.default = default
        self.variable = variable
        self.variable_error = variable_error
        self.workflow = workflow
        self.issue_state = issue_state
        self.close_rc = close_rc
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
    # Unreadable repository metadata cannot affect the default-branch path either.
    broken = ScriptRunner()
    broken.default = None
    enforce_trusted_base_at_startup(broken, _config(tmp_path, base="main"))


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
    assert managed_ci._workflow_source_ref(ScriptRunner(default=None), _config(tmp_path), "x") == "x"


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
    default_merge = ScriptRunner(pr={
        "merged": True, "merged_at": "t", "base": {"ref": "main", "repo": {"full_name": "OWNER/REPO"}}, "body": "Closes #42",
    })
    assert close_child_after_integration_merge(default_merge, config=cfg, issue_context=ctx, pr_number=7) == "skipped"
    unmerged = ScriptRunner(merged=False)
    assert close_child_after_integration_merge(unmerged, config=cfg, issue_context=ctx, pr_number=7) == "skipped"
    no_reference = ScriptRunner(pr={
        "merged": True, "merged_at": "t", "base": {"ref": "refactor/1181", "repo": {"full_name": "OWNER/REPO"}}, "body": "Refs #42",
    })
    assert close_child_after_integration_merge(no_reference, config=cfg, issue_context=ctx, pr_number=7) == "skipped"
    for runner in (default_merge, unmerged, no_reference):
        assert not runner.commands_matching("gh", "issue", "close")


def test_failed_close_is_reported_unresolved_and_never_replays_the_merge(tmp_path, capsys):
    runner = ScriptRunner(close_rc=1)
    assert not require_child_closed_or_report(
        runner, config=_config(tmp_path), issue_context=_issue_context(), pr_number=7
    )
    assert "could not be closed" in capsys.readouterr().out
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
