"""Terminal retirement through real issue/planning/execution recovery paths."""
import json
from dataclasses import replace

import pytest

from coding_review_agent_loop.cli import run_issue_loop
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.runner import CommandResult
from coding_review_agent_loop import decomposition as d
from coding_review_agent_loop import split_materialization as s
from coding_review_agent_loop.issue_pr_handoff import format_issue_pr_handoff_comment
from coding_review_agent_loop.child_topology import parent_child_search_queries
from agent_loop_helpers import structured_v1_plan_state, structured_plan_review, make_config
from issue_loop_test_helpers import _fresh_staged_plan_for_recovery
from test_plan_first_loop_execution_strategy import (
    _M1087Runner, _M1087_STALE_HASH, _m1087_config, _m1087_stale_decision_comment,
    _m1087_posted_decisions, _m1087_implementation_dispatched,
)


def comment(body, second=1):
    return {"author": {"login": "bot"}, "createdAt": f"2026-05-22T00:00:{second:02d}Z", "body": body}


def child(number=90, state="CLOSED", body=""):
    return {"number": number, "state": state, "title": "[#56 stage] Old work", "body": body,
            "url": f"https://github.com/OWNER/REPO/issues/{number}"}


def pr(number=82, state="MERGED"):
    return {"number": number, "state": state, "body": "Fixes #56", "headRefName": "agent-loop/managed-56"}


class RetirementRunner(_M1087Runner):
    def __init__(self, *, children=(), prs=(), fault_source=None, fault="invalid-json", branch_error=None, **kwargs):
        self.children = list(children)
        self.prs = list(prs)
        self.fault_source = fault_source
        self.fault = fault
        self.branch_error = branch_error
        kwargs.setdefault("serve_rest_issue_comments", True)
        kwargs.setdefault("issue_comments", [_m1087_stale_decision_comment()])
        kwargs.setdefault("claude_outputs", [structured_v1_plan_state(),
            "Implemented.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"])
        kwargs.setdefault("codex_outputs", [structured_plan_review(state="approved"),
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"])
        kwargs.setdefault("issue_urls", ["https://github.com/OWNER/REPO/issues/99", "https://github.com/OWNER/REPO/issues/100"])
        super().__init__(search_issues_payload=self.children, **kwargs)

    def run(self, args, *, cwd, input_text=None, check=True, env=None):
        cmd = [str(a) for a in args]
        source = None
        if cmd[:2] == ["gh", "api"] and cmd[2].startswith("repos/OWNER/REPO/pulls/") and cmd[2].split("/")[-1].isdigit():
            cmd, cwd_path = self._record_command(cmd, cwd)
            return CommandResult(cmd, cwd_path, json.dumps({"merged": True, "merged_at": "2026-05-21T00:00:00Z",
                "base": {"ref": "main", "repo": {"full_name": "OWNER/REPO"}}}), "", 0)
        if cmd[:3] == ["gh", "api", "repos/OWNER/REPO"]:
            cmd, cwd_path = self._record_command(cmd, cwd)
            return CommandResult(cmd, cwd_path, json.dumps({"default_branch": "main"}), "", 0)
        if cmd[:3] == ["gh", "issue", "list"] and "state" in cmd[cmd.index("--json") + 1].split(","):
            source = cmd[cmd.index("--search") + 1]
            payload = self.children
        elif cmd[:3] == ["gh", "pr", "list"]:
            payload = self.prs
            if "state" in cmd[cmd.index("--json") + 1].split(","):
                source = "prs"
            else:
                payload = [p for p in payload if p.get("state") == "OPEN"]
        elif cmd[:2] == ["gh", "api"] and "/branches/" in cmd[2] and self.branch_error:
            cmd, cwd_path = self._record_command(cmd, cwd)
            return CommandResult(cmd, cwd_path, "", self.branch_error, 1)
        elif cmd[:3] in (["gh", "issue", "view"], ["gh", "pr", "view"]) and cmd[cmd.index("--json") + 1] == "number,state":
            cmd, cwd_path = self._record_command(cmd, cwd)
            number = int(cmd[3])
            mapping = self.issue_payloads_by_number if cmd[1] == "issue" else self.pr_payloads_by_number
            item = mapping.get(number)
            if item is None and cmd[1] == "pr" and number == 77:
                item = self.pr_payload
            return CommandResult(cmd, cwd_path, json.dumps(item), "" if item else "not found", 0 if item else 1)
        else:
            return super().run(args, cwd=cwd, input_text=input_text, check=check, env=env)
        cmd, cwd_path = self._record_command(cmd, cwd)
        if source is not None and source == self.fault_source:
            if self.fault == "failure":
                return CommandResult(cmd, cwd_path, "", "permission denied", 1)
            if self.fault == "invalid-json":
                return CommandResult(cmd, cwd_path, "not json", "", 0)
            if self.fault == "shape":
                return CommandResult(cmd, cwd_path, "{}", "", 0)
            if self.fault == "saturated":
                payload = [child()] * 100000 if source != "prs" else [pr()] * 100000
            elif self.fault == "number":
                payload = [dict(pr() if source == "prs" else child(), number=True)]
            elif self.fault == "state":
                payload = [dict(pr() if source == "prs" else child(), state="mystery")]
            elif self.fault == "state-object":
                payload = [dict(pr() if source == "prs" else child(), state={})]
            elif self.fault == "missing-state":
                payload = [pr() if source == "prs" else child()]
                payload[0].pop("state")
            elif self.fault == "association":
                payload = [dict(pr(), headRefName=None)] if source == "prs" else [dict(child(), title=None)]
        return CommandResult(cmd, cwd_path, json.dumps(payload), "", 0)


def run(runner, tmp_path, **options):
    return run_issue_loop(runner, issue_number=56, config=replace(_m1087_config(tmp_path), **options), plan_first=True)


def no_dispatch(runner):
    assert _m1087_posted_decisions(runner) == []
    assert not _m1087_implementation_dispatched(runner)
    assert not runner.issues
    assert not any("AGENT_PLAN_ONE_SHOT_IMPL" in b for b in runner.comments)


@pytest.mark.parametrize("state", ["CLOSED", "MERGED"])
def test_terminal_inventory_dispatches_new_work(tmp_path, state):
    runner = RetirementRunner(children=[child()], prs=[pr(state=state)])
    assert run(runner, tmp_path) == 0
    assert _m1087_posted_decisions(runner)[0].retires_plan_hashes == (_M1087_STALE_HASH,)
    assert _m1087_implementation_dispatched(runner)
    assert runner.issue_comments[0]["body"] == _m1087_stale_decision_comment()["body"]


@pytest.mark.parametrize("live", ["child", "pr", "branch-pr"])
def test_mixed_inventory_names_only_live_blockers(tmp_path, live):
    children = [child(), child(91, "OPEN")] if live == "child" else [child()]
    prs = [pr(), pr(83, "OPEN")] if live != "child" else [pr()]
    if live == "branch-pr":
        prs[-1]["body"] = "No parent closing reference."
    runner = RetirementRunner(children=children, prs=prs)
    with pytest.raises(AgentLoopError) as error:
        run(runner, tmp_path)
    assert ("child issue #91" if live == "child" else "PR #83") in str(error.value)
    assert "#90" not in str(error.value) and "#82" not in str(error.value)
    no_dispatch(runner)


@pytest.mark.parametrize("terminal_pr", [False, True])
def test_branch_must_be_absent_even_after_merge(tmp_path, terminal_pr):
    runner = RetirementRunner(prs=[pr()] if terminal_pr else [], branch_exists=True)
    with pytest.raises(AgentLoopError, match="delete the branch manually") as error:
        run(runner, tmp_path)
    assert "close that work" not in str(error.value)
    assert not any(cmd[:3] == ["gh", "api", "--method"] for cmd, _ in runner.commands)
    no_dispatch(runner)


@pytest.mark.parametrize("error", ["HTTP 403", "transport failed", "Not Found without a status"])
def test_unknown_branch_state_never_means_absence(tmp_path, error):
    runner = RetirementRunner(branch_error=error)
    with pytest.raises(AgentLoopError, match="could not be checked"):
        run(runner, tmp_path)
    no_dispatch(runner)


@pytest.mark.parametrize("source", [*parent_child_search_queries(56), "prs"])
@pytest.mark.parametrize("fault", ["failure", "invalid-json", "shape", "saturated", "number", "state", "state-object", "missing-state", "association"])
def test_uncertain_inventory_refuses_before_execution(tmp_path, source, fault):
    runner = RetirementRunner(fault_source=source, fault=fault)
    with pytest.raises(AgentLoopError, match="Cannot verify"):
        run(runner, tmp_path)
    no_dispatch(runner)


def old_record(kind):
    if kind == "pr":
        return d.format_one_shot_impl_handoff_comment(parent_issue=56, mode="implement-one-shot",
            plan_hash=_M1087_STALE_HASH, plan_subject="old", pr_number=82, pr_head_sha="abc123")
    if kind == "canonical":
        return format_issue_pr_handoff_comment(issue_number=56, pr_number=82,
            pr_url="https://github.com/OWNER/REPO/pull/82", pr_head_sha="abc123",
            flow="approved-plan-implementation", plan_hash=_M1087_STALE_HASH)
    if kind == "split":
        metadata = s.SplitMaterializationMetadata(56, "old", (s.MaterializedSplitChild("Old", "a" * 64,
            "https://github.com/OWNER/REPO/issues/90", 90, "created"),))
        return s.format_split_materialization_summary(parent_issue=56, metadata=metadata)
    if kind == "phase":
        return d.format_phase_implementation_handoff_comment(parent_issue=56, mode="implement-by-phase",
            plan_hash=_M1087_STALE_HASH, phase_index=1,
            created=d.CreatedPhaseIssue(d.RecordedPhase("Old", "agent-pr"), "https://github.com/OWNER/REPO/issues/90", 90, "created"))
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["pr", "canonical", "split", "phase"])
@pytest.mark.parametrize("state", ["CLOSED", "OPEN", "unreadable"])
def test_comment_references_are_resolved_when_search_misses(tmp_path, kind, state):
    runner = RetirementRunner(issue_comments=[_m1087_stale_decision_comment(), comment(old_record(kind))],
        issue_payloads_by_number={90: child(state=state)} if state != "unreadable" else {},
        pr_payloads_by_number={82: dict(pr(state=state), url="https://github.com/OWNER/REPO/pull/82", headRefOid="abc123", comments=[])} if state != "unreadable" else {},
        malformed_pr_view_numbers=[82] if state == "unreadable" else [])
    if state == "CLOSED":
        assert run(runner, tmp_path) == 0
        assert _m1087_implementation_dispatched(runner)
    else:
        with pytest.raises(AgentLoopError):
            run(runner, tmp_path)
        no_dispatch(runner)


@pytest.mark.parametrize("old_strategy", ["one-shot", "staged"])
@pytest.mark.parametrize("new_strategy", ["one-shot", "staged"])
@pytest.mark.parametrize("policy", ["auto", "explicit"])
def test_strategy_transitions_route_past_terminal_history(tmp_path, old_strategy, new_strategy, policy):
    record = old_record("pr" if old_strategy == "one-shot" else "phase")
    raw = structured_v1_plan_state() if new_strategy == "one-shot" else _fresh_staged_plan_for_recovery()[0]
    if new_strategy == "staged":
        payload, end = json.JSONDecoder().raw_decode(raw)
        for stage in payload["execution_recommendation"]["child_stages"]:
            stage["execution_disposition"] = {
                "disposition": "direct-implementation", "rationale": "Reviewed bounded stage.",
                "unresolved_design_decisions": [],
            }
            stage["non_goals"] = ["No unrelated work."]
            stage["compatibility_constraints"] = ["Preserve public interfaces."]
        raw = json.dumps(payload) + raw[end:]
    earlier = _m1087_stale_decision_comment()
    if old_strategy == "staged":
        marker = d.EXECUTION_DECISION_MARKER_RE.search(earlier["body"])
        decision = d._decode_execution_decision(marker["payload"])
        earlier["body"] = d.format_execution_decision(replace(
            decision, strategy="staged", requested_policy="implement-by-phase",
            current_action="implement-by-phase", stage_ids=("old-stage",),
        ))
    runner = RetirementRunner(children=[child()], prs=[pr()], issue_comments=[earlier, comment(record)],
        claude_outputs=[raw, "Implemented.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"])
    if new_strategy == "staged":
        runner.pr_payload["body"] = "Fixes #99"
    mode = "auto" if policy == "auto" else "implement-one-shot" if new_strategy == "one-shot" else "decompose-only"
    assert run(runner, tmp_path, plan_execution_mode=mode) == 0
    assert _m1087_posted_decisions(runner)[0].strategy == new_strategy
    assert not any(issue["title"] == "Old" for issue in runner.issues)
    if new_strategy == "staged":
        assert len(runner.issues) == 2
    else:
        assert _m1087_implementation_dispatched(runner)


def test_restart_rebuilds_history_with_current_open_pr(tmp_path):
    runner = RetirementRunner(children=[child()], issue_comments=[_m1087_stale_decision_comment(), comment(old_record("split"))])
    assert run(runner, tmp_path) == 0
    decision_count = len(_m1087_posted_decisions(runner))
    runner.prs = [dict(pr(77, "OPEN"), headRefName="agent-loop/managed-56")]
    runner.branch_exists = True
    assert run(runner, tmp_path) == 0
    assert len(_m1087_posted_decisions(runner)) == decision_count
    runner.children[0]["state"] = "OPEN"
    with pytest.raises(AgentLoopError, match="child issue #90"):
        run(runner, tmp_path)
    assert len(_m1087_posted_decisions(runner)) == decision_count


@pytest.mark.parametrize("mode,dry", [("plan-only", False), ("auto", True)])
def test_nonmutating_modes_do_not_retire_or_dispatch(tmp_path, mode, dry):
    runner = RetirementRunner(children=[child(90, "OPEN")], branch_exists=True)
    assert run(runner, tmp_path, plan_execution_mode=mode, dry_run=dry) == 0
    no_dispatch(runner)


@pytest.mark.parametrize("state", ["CLOSED", "MERGED"])
def test_terminal_canonical_entry_defers_only_for_proven_new_plan(tmp_path, state):
    runner = RetirementRunner(issue_comments=[_m1087_stale_decision_comment(), comment(old_record("canonical"))],
        pr_payloads_by_number={82: dict(pr(state=state), url="https://github.com/OWNER/REPO/pull/82", headRefOid="abc123", comments=[])})
    assert run(runner, tmp_path) == 0
    assert _m1087_implementation_dispatched(runner)
    assert _m1087_posted_decisions(runner)[0].retires_plan_hashes == (_M1087_STALE_HASH,)
    assert any("PR #77" in body for body in runner.comments)


@pytest.mark.parametrize("state", ["CLOSED", "MERGED"])
def test_terminal_canonical_direct_mode_still_refuses(tmp_path, state):
    body = format_issue_pr_handoff_comment(issue_number=56, pr_number=82,
        pr_url="https://github.com/OWNER/REPO/pull/82", pr_head_sha="abc123",
        flow="issue-implementation", plan_hash=None)
    runner = RetirementRunner(issue_comments=[comment(body)],
        pr_payloads_by_number={82: dict(pr(state=state), url="https://github.com/OWNER/REPO/pull/82", headRefOid="abc123", comments=[])})
    with pytest.raises(AgentLoopError, match="not OPEN"):
        run_issue_loop(runner, issue_number=56, config=make_config(tmp_path))
    no_dispatch(runner)
    assert len(runner.claude_outputs) == 2


def test_terminal_canonical_without_retirement_authority_still_refuses(tmp_path):
    runner = RetirementRunner(issue_comments=[comment(old_record("canonical"))],
        pr_payloads_by_number={82: dict(pr(state="MERGED"), url="https://github.com/OWNER/REPO/pull/82", headRefOid="abc123", comments=[])})
    with pytest.raises(AgentLoopError, match="not OPEN"):
        run(runner, tmp_path)
    no_dispatch(runner)
    assert len(runner.claude_outputs) == 1  # planning ran, entry alone granted nothing


@pytest.mark.parametrize("state", ["CLOSED", "MERGED"])
def test_same_plan_terminal_canonical_does_not_dispatch_again(tmp_path, state):
    runner = RetirementRunner()
    assert run(runner, tmp_path) == 0
    runner.pr_payload["state"] = state
    before = len(_m1087_posted_decisions(runner))
    with pytest.raises(AgentLoopError, match="not OPEN"):
        run(runner, tmp_path)
    assert len(_m1087_posted_decisions(runner)) == before
    assert not runner.claude_outputs


def test_restart_between_decision_and_replacement_handoff(tmp_path):
    runner = RetirementRunner(children=[child()], issue_comments=[_m1087_stale_decision_comment(), comment(old_record("split")), comment(old_record("canonical"), 2)],
        pr_payloads_by_number={82: dict(pr(), url="https://github.com/OWNER/REPO/pull/82", headRefOid="abc123", comments=[])})
    assert run(runner, tmp_path) == 0
    # Simulate death immediately after decision publication: planning and the
    # decision are durable, but all implementation handoffs have not happened.
    runner.issue_comments = [c for c in runner.issue_comments if not (
        "AGENT_PLAN_ONE_SHOT_IMPL" in c["body"] or ("AGENT_ISSUE_PR_HANDOFF" in c["body"] and "PR #77" in c["body"]))]
    runner.comments.clear()
    runner.claude_outputs = ["Implemented.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"]
    runner.codex_outputs = ["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"]
    assert run(runner, tmp_path) == 0
    assert _m1087_posted_decisions(runner) == []
    assert sum("AGENT_PLAN_ONE_SHOT_IMPL" in b for b in runner.comments) == 1


@pytest.mark.parametrize("kind", ["summary", "checkpoint"])
def test_staged_history_resolves_children_and_unrealized_checkpoints(tmp_path, kind):
    phase = d.PlanPhase("Old", "Old scope", "None", "None", "low", "Tests", "Parent", "agent-pr")
    created = d.CreatedPhaseIssue(phase, "https://github.com/OWNER/REPO/issues/90", 90)
    body = d.format_decomposition_parent_summary(parent_issue=56, mode="implement-by-phase", plan_hash=_M1087_STALE_HASH,
        created=(created,)) if kind == "summary" else d.format_topology_checkpoint(d.TopologyCheckpoint(
            56, _M1087_STALE_HASH, "implement-by-phase", "model", (phase,)))
    runner = RetirementRunner(children=[child()] if kind == "summary" else [],
        issue_comments=[_m1087_stale_decision_comment(), comment(body)])
    assert run(runner, tmp_path) == 0
    assert _m1087_implementation_dispatched(runner)


def test_current_ownership_conflict_is_not_hidden_by_retirement(tmp_path):
    runner = RetirementRunner(children=[child()], issue_comments=[_m1087_stale_decision_comment(), comment(old_record("split"))])
    assert run(runner, tmp_path) == 0
    current = _m1087_posted_decisions(runner)[0]
    body = d.format_phase_implementation_handoff_comment(parent_issue=56, mode="implement-by-phase",
        plan_hash=current.plan_hash, phase_index=1,
        created=d.CreatedPhaseIssue(d.RecordedPhase("Wrong strategy", "agent-pr"), "https://github.com/OWNER/REPO/issues/90", 90))
    runner.issue_comments.append(comment(body, 59))
    with pytest.raises(AgentLoopError, match="Ambiguous ownership"):
        run(runner, tmp_path)
    assert len(_m1087_posted_decisions(runner)) == 1


def staged_raw():
    raw = _fresh_staged_plan_for_recovery()[0]
    payload, end = json.JSONDecoder().raw_decode(raw)
    for stage in payload["execution_recommendation"]["child_stages"]:
        stage["execution_disposition"] = {"disposition": "direct-implementation",
            "rationale": "The slice is bounded.", "unresolved_design_decisions": []}
        stage["non_goals"] = ["No unrelated work."]
        stage["compatibility_constraints"] = ["Preserve public interfaces."]
    return json.dumps(payload) + raw[end:]


def test_partial_materialization_reuses_current_child_beside_closed_split_history(tmp_path):
    runner = RetirementRunner(children=[child()],
        issue_comments=[_m1087_stale_decision_comment(), comment(old_record("split"))],
        claude_outputs=[staged_raw()])
    assert run(runner, tmp_path, plan_execution_mode="decompose-only") == 0
    current = _m1087_posted_decisions(runner)[0]
    first_child_body = runner.issues[0]["body"]
    first_child_title = runner.issues[0]["title"]
    # The approved sidecar and first child survive; publication of the cumulative
    # summary and the second child did not happen.
    runner.issue_comments = [c for c in runner.issue_comments if not any(
        d._decode_metadata(m["payload"]).plan_hash == current.plan_hash
        for m in d.DECOMPOSITION_MARKER_RE.finditer(c["body"]))]
    runner.children.append(dict(child(99, "OPEN", first_child_body), title=first_child_title))
    runner.issue_urls = ["https://github.com/OWNER/REPO/issues/100"]
    runner.issues.clear()
    runner.comments.clear()
    assert run(runner, tmp_path, plan_execution_mode="decompose-only") == 0
    assert len(runner.issues) == 1
    assert "Second stage" in runner.issues[0]["title"]
    assert _m1087_posted_decisions(runner) == []


def test_restart_unattributed_open_child_blocks_without_new_dispatch(tmp_path):
    runner = RetirementRunner(children=[child()], issue_comments=[_m1087_stale_decision_comment(), comment(old_record("split"))])
    assert run(runner, tmp_path) == 0
    runner.children.append(child(91, "OPEN"))
    with pytest.raises(AgentLoopError, match="child issue #91"):
        run(runner, tmp_path)
    assert len(_m1087_posted_decisions(runner)) == 1


@pytest.mark.parametrize("body", ["unmapped-split", "malformed-summary"])
def test_unresolved_comment_only_topology_never_becomes_history(tmp_path, body):
    if body == "unmapped-split":
        raw = "<!-- AGENT_SPLIT_CHILD: parent=56 key=" + "a" * 64 + " -->"
    else:
        raw = "<!-- AGENT_PLAN_DECOMPOSITION: e30= -->"
    runner = RetirementRunner(issue_comments=[_m1087_stale_decision_comment(), comment(raw)])
    with pytest.raises(AgentLoopError):
        run(runner, tmp_path)
    no_dispatch(runner)


def test_restart_hash_recurrence_uses_record_order_even_in_one_comment(tmp_path):
    runner = RetirementRunner()
    assert run(runner, tmp_path) == 0
    current = _m1087_posted_decisions(runner)[0]
    old_a = _m1087_stale_decision_comment(current.plan_hash)["body"]
    old_pr = d.format_one_shot_impl_handoff_comment(parent_issue=56, mode="implement-one-shot",
        plan_hash=current.plan_hash, plan_subject=current.plan_subject, pr_number=82, pr_head_sha="abc123")
    middle_b = _m1087_stale_decision_comment(retires=[current.plan_hash])["body"]
    runner.issue_comments[0] = comment("\n".join([old_a, old_pr, middle_b]), 0)
    runner.pr_payloads_by_number[82] = dict(pr(state="CLOSED"),
        url="https://github.com/OWNER/REPO/pull/82", headRefOid="abc123", comments=[])
    runner.prs = [pr(77, "OPEN")]
    runner.branch_exists = True
    assert run(runner, tmp_path) == 0
    assert len(_m1087_posted_decisions(runner)) == 1
    assert runner.issue_comments[0]["body"].count(old_pr) == 1
    runner.pr_payloads_by_number[82]["state"] = "OPEN"
    with pytest.raises(AgentLoopError, match="PR #82"):
        run(runner, tmp_path)
    assert len(_m1087_posted_decisions(runner)) == 1


@pytest.mark.parametrize("state", ["CLOSED", "OPEN"])
def test_hashless_split_marker_requires_a_resolved_terminal_child(tmp_path, state):
    marker = "<!-- AGENT_SPLIT_CHILD: parent=56 key=" + "a" * 64 + " -->"
    runner = RetirementRunner(children=[child(state=state, body=marker)],
        issue_comments=[_m1087_stale_decision_comment(), comment(marker)])
    if state == "CLOSED":
        assert run(runner, tmp_path) == 0
    else:
        with pytest.raises(AgentLoopError, match="child issue #90"):
            run(runner, tmp_path)
        no_dispatch(runner)


def test_restart_recovers_provenanced_pr_created_before_replacement_handoff(tmp_path):
    runner = RetirementRunner(issue_comments=[_m1087_stale_decision_comment(), comment(old_record("canonical"))],
        pr_payloads_by_number={82: dict(pr(), url="https://github.com/OWNER/REPO/pull/82", headRefOid="abc123", comments=[])})
    assert run(runner, tmp_path) == 0
    runner.issue_comments = [c for c in runner.issue_comments if not (
        "AGENT_PLAN_ONE_SHOT_IMPL" in c["body"] or ("AGENT_ISSUE_PR_HANDOFF" in c["body"] and "PR #77" in c["body"]))]
    runner.prs = [pr(77, "OPEN")]
    runner.branch_exists = True
    from issue_loop_test_helpers import _provenance_pages
    from coding_review_agent_loop.issue_pr_provenance import IssuePrProvenanceScope, format_issue_pr_provenance
    current = _m1087_posted_decisions(runner)[0]
    runner.pr_commit_pages = _provenance_pages(format_issue_pr_provenance(
        IssuePrProvenanceScope("OWNER/REPO", 56, "approved", current.plan_hash))) * 10
    runner.comments.clear()
    # No scripted coder output remains: only recovery of this PR can succeed.
    assert run(runner, tmp_path) == 0
    assert _m1087_posted_decisions(runner) == []
    assert not runner.claude_outputs
