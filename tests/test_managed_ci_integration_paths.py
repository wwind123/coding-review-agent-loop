"""Successful integration-mode entry paths (#1285 matrix: tool-dispatch-from-default).

The default branch is ``main`` and its ``ci.yml`` is the only trusted workflow;
``refactor/1181``'s ``ci.yml`` differs (and must never be read).  Each path
must succeed, read and dispatch only from ``main``, and keep the real base.
"""

import json
from dataclasses import replace

import pytest

from coding_review_agent_loop.managed_ci import (
    FINAL_CONTEXT,
    TRUSTED_BASES_MARKER,
    ManagedCiProbeContext,
    _dispatch_v2_qualification,
    activate_managed_ci,
    evaluate_managed_ci_readiness,
    preflight_managed_ci_creation,
)
from coding_review_agent_loop.runner import CommandResult

from agent_loop_helpers import make_config
from test_managed_ci import (
    V2_HEAD,
    V2_REVISION,
    V2_WORKFLOW,
    V2ManagedRunner,
    adoption_workflow,
    label_event,
    metadata,
)

INTEGRATION = "refactor/1181"
# The integration branch's workflow differs: it lacks every marker.
INTEGRATION_WORKFLOW = "name: CI\non: push\njobs: {}\n"


class IntegrationRunner(V2ManagedRunner):
    """main serves the trusted workflow; any other ref serves a different one."""

    def __init__(self, *, trusted_workflow=None, variable="refactor/*", **kwargs):
        trusted = (trusted_workflow or V2_WORKFLOW) + f"\n# {TRUSTED_BASES_MARKER}\n"
        super().__init__(workflow=trusted, **kwargs)
        self.trusted_workflow = trusted
        self.variable = variable
        self.reads = []

    def _run_locked(self, args, *, cwd, check, input_text=None):
        cmd = list(args)
        endpoint = next((p for p in cmd if isinstance(p, str) and p.startswith("repos/")), "")
        self.reads.append(endpoint)
        if "/contents/.github/workflows/ci.yml" in endpoint:
            ref = endpoint.partition("?ref=")[2]
            cmd, cwd_path = self._record_command(args, cwd)
            text = self.trusted_workflow if ref == "main" else INTEGRATION_WORKFLOW
            return CommandResult(cmd, cwd_path, text, "", 0)
        if endpoint.endswith("/actions/variables/AGENT_LOOP_TRUSTED_BASES"):
            cmd, cwd_path = self._record_command(args, cwd)
            return CommandResult(cmd, cwd_path, json.dumps({"value": self.variable}), "", 0)
        if endpoint == f"repos/OWNER/REPO/commits/{INTEGRATION}":
            cmd, cwd_path = self._record_command(args, cwd)
            return CommandResult(cmd, cwd_path, json.dumps({"sha": "integration-sha"}), "", 0)
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)

    def integration_reads(self):
        # Workflow-source and revision reads only; branch protection is rightly read on the real base.
        return [r for r in self.reads if INTEGRATION in r and ("/contents/" in r or "/commits/" in r)]


def _config(tmp_path, **kw):
    kw.setdefault("base", INTEGRATION)
    return make_config(tmp_path, auto_merge=True, managed_ci_trusted_actor="agent-loop", **kw)


def _integration_pr(**extra):
    pr = {"base": {"ref": INTEGRATION}}
    pr.update(extra)
    return pr


def test_issue_and_branch_creation_preflight_reads_main_only(tmp_path):
    runner = IntegrationRunner(rest_pr=_integration_pr(), pr_payload={"headRefOid": "abc123"})
    for kwargs in ({"issue_number": 643}, {"branch": "agent-loop/managed-direct-1-tok"}):
        intent = preflight_managed_ci_creation(runner, config=_config(tmp_path), **kwargs)
        assert intent is not None
    reads = [r for r in runner.reads if "/contents/.github/workflows/ci.yml" in r]
    assert reads and all(r.endswith("ref=main") for r in reads)
    assert runner.integration_reads() == []


def test_readiness_for_an_integration_base_reads_main_and_checks_trust(tmp_path):
    runner = IntegrationRunner()
    readiness = evaluate_managed_ci_readiness(
        runner, context=ManagedCiProbeContext("OWNER/REPO", "gh", tmp_path),
        base=INTEGRATION, trusted_actor="agent-loop",
    )
    assert readiness.state != "invalid"
    assert all(r.endswith("ref=main") for r in runner.reads if "/contents/.github/workflows/ci.yml" in r)
    assert runner.integration_reads() == []


def test_primary_activation_then_dispatch_use_main_and_record_the_real_base(tmp_path):
    runner = IntegrationRunner(
        rest_pr=_integration_pr(head={
            "repo": {"full_name": "OWNER/REPO"}, "sha": V2_HEAD, "ref": "agent-loop/managed-643",
        }),
        pr_payload={"headRefOid": V2_HEAD}, base_sha=V2_REVISION,
    )
    config = _config(tmp_path)
    contract = activate_managed_ci(
        runner, config=config, pr_number=7,
        metadata=replace(metadata(base_branch=INTEGRATION), head_sha=V2_HEAD),
    )
    assert contract is not None
    assert (contract.base_ref, contract.dispatch_ref) == (INTEGRATION, "main")
    assert contract.workflow_revision == runner.base_sha  # revision of main, not of the integration head

    _dispatch_v2_qualification(
        runner, config=config, pr_number=7, expected_head_sha=V2_HEAD, contract=contract
    )
    dispatches = [c for c, _ in runner.commands if any(str(p).endswith("/dispatches") for p in c)]
    assert len(dispatches) == 1 and "ref=main" in dispatches[0]
    assert runner.intent_snapshots and all(s["base_ref"] == INTEGRATION for s in runner.intent_snapshots)
    assert runner.integration_reads() == []


@pytest.mark.parametrize("fresh_resume", [False, True], ids=["optional-adoption", "resumed-authorization"])
def test_existing_pr_adoption_and_resume_read_main_only(tmp_path, fresh_resume):
    config = _config(tmp_path, managed_ci_adopt_existing_pr=True)
    runner = IntegrationRunner(
        trusted_workflow=adoption_workflow(),
        rest_pr={
            "draft": False, "state": "open", "base": {"ref": INTEGRATION},
            "user": {"login": "someone", "id": 55},
            "head": {"repo": {"full_name": "OWNER/REPO"}, "sha": "abc123", "ref": "feature"},
            "labels": [{"name": "agent-loop-managed"}] if fresh_resume else [],
        },
        issue_events=[label_event()] if fresh_resume else [],
        pr_branch_protection_payload={"contexts": [FINAL_CONTEXT]},
    )
    contract = activate_managed_ci(
        runner, config=config, pr_number=7, metadata=metadata(base_branch=INTEGRATION)
    )
    assert contract is not None and contract.adopted_existing_pr
    assert (contract.base_ref, contract.dispatch_ref) == (INTEGRATION, "main")
    assert runner.integration_reads() == []
    assert all(
        r.endswith("ref=main") for r in runner.reads if "/contents/.github/workflows/ci.yml" in r
    )


# --- manual qualification: drive through to the final pre-publication guard -------------------

from agent_loop_helpers import FakeRunner  # noqa: E402
from coding_review_agent_loop.errors import AgentLoopError  # noqa: E402
from coding_review_agent_loop.managed_ci import QUALIFIED_LABEL  # noqa: E402
from test_managed_ci import PublicationRunner, _audit_posted, _publish  # noqa: E402


class IntegrationPublicationRunner(PublicationRunner):
    """PublicationRunner whose repository allow-lists ``refactor/*`` (mutable)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.variable = "refactor/*"

    def _run_locked(self, args, *, cwd, check, input_text=None):
        cmd = list(args)
        if cmd[:2] == ["gh", "api"] and cmd[2].endswith("/actions/variables/AGENT_LOOP_TRUSTED_BASES"):
            cmd, cwd_path = self._record_command(args, cwd)
            return CommandResult(cmd, cwd_path, json.dumps({"value": self.variable}), "", 0)
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)

    def qualified_label_deletes(self):
        return [c for c, _ in self.commands if "DELETE" in c and any(QUALIFIED_LABEL in p for p in c)]


def _late(runner, *, retarget=None, revoke=False):
    def override(payload):
        if retarget is not None:
            payload["base"] = {"ref": retarget, "repo": {"full_name": "OWNER/REPO"}}
        if revoke:
            runner.variable = "other/*"
        return payload
    return override


@pytest.mark.parametrize("adopted", [False, True], ids=["issue-created", "adopted"])
@pytest.mark.parametrize(
    "event", [{"retarget": "feature/x"}, {"retarget": "main"}, {"revoke": True}],
    ids=["untrusted-retarget", "default-retarget", "allow-list-revoked"],
)
def test_manual_qualification_refuses_a_late_change_before_the_audit_record(tmp_path, adopted, event):
    base = {"ref": INTEGRATION, "repo": {"full_name": "OWNER/REPO"}}
    runner = IntegrationPublicationRunner(rest_pr={"draft": False, "base": base} if adopted else {"base": base})
    # PR reads: 0 stale-label cleanup, 1 initial guard, 2 recorded-base guard (passes),
    # 3 post-readiness re-read, 4 final pre-publication guard (the change lands here).
    runner.pr_overrides = {4: _late(runner, **event)}
    with pytest.raises(AgentLoopError, match="manual qualification record refused"):
        _publish(
            runner, tmp_path, base_ref=INTEGRATION, dispatch_ref="main",
            issue_created_pr=not adopted, adopted_existing_pr=adopted,
        )
    assert not _audit_posted(runner)
    assert runner.qualified_label_deletes()
    # The earlier checks passed, so readiness for the issue-created draft happened
    # legitimately before the change; it is never repeated and nothing was published.
    ready = [c for c, _ in runner.commands if c[:3] == ["gh", "pr", "ready"]]
    assert len(ready) == (0 if adopted else 1)
