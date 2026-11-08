"""Process-isolated fake GitHub CLI and orchestration driver for #1882.

The server persists only externally visible GitHub data, never handoff objects.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

from agent_loop_helpers import FakeRunner, make_config, structured_plan_review, structured_plan_state
from issue_loop_test_helpers import _IssueRecoveryWorkflowRunner
from coding_review_agent_loop import managed_ci, pr_loop, issue_loop
from coding_review_agent_loop.runner import CommandResult

FIELDS = ("rest_pr", "pr_payload", "issue_comments", "authorization_comments", "issue_events", "pr_branch_protection_payload")


def server(path, command, cwd, input_text):
    fixture = _IssueRecoveryWorkflowRunner(labeled=True, authenticated_actor=("agent-loop", 1))
    state = json.loads(path.read_text())
    for field in FIELDS:
        setattr(fixture, field, state[field])
    if not state.get("opened") and any("/issues/56/timeline?" in part for part in command):
        return {"stdout": "[]", "stderr": "", "returncode": 0}
    if any("/issues/77/comments?" in part and "&page=" in part for part in command):
        result = FakeRunner._run_locked(fixture, command, cwd=Path(cwd), check=False, input_text=input_text)
        comments = {c["id"]: c for c in json.loads(result.stdout)}
        for comment in fixture.authorization_comments:
            comments.setdefault(comment["id"], comment)
        result = CommandResult(result.args, result.cwd, json.dumps(list(comments.values())), "", 0)
    else:
        result = fixture.run(command, cwd=Path(cwd), check=False, input_text=input_text)
    path.write_text(json.dumps({**{field: getattr(fixture, field) for field in FIELDS}, "opened": state.get("opened", False)}))
    return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}


class ExternalRunner(_IssueRecoveryWorkflowRunner):
    def __init__(self, path, *, create):
        super().__init__(labeled=True, authenticated_actor=("agent-loop", 1),
            claude_outputs=[structured_plan_state(summary="Preserve the recovery boundary."),
                "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"] if create else [],
            codex_outputs=[structured_plan_review(state="approved")] if create else [],
            git_head="1" * 40, advance_git_head_on_pr=False)
        self.path = path
        if create:
            path.write_text(json.dumps({**{field: getattr(self, field) for field in FIELDS}, "opened": False}))
        else:
            state = json.loads(path.read_text())
            self.pr_payload = state["pr_payload"]
        self.agent_calls = 0

    def _run_locked(self, args, *, cwd, check, input_text=None):
        command = list(args)
        if command[0] == "gh":
            recorded, cwd_path = self._record_command(args, cwd)
            # Model the CLI at its command seam. Only durable JSON state
            # crosses the creation/resume interpreter boundary; starting a new
            # interpreter for every API call adds no restart coverage.
            with contextlib.redirect_stdout(io.StringIO()):
                response = server(self.path, command, str(cwd), input_text)
            return CommandResult(recorded, cwd_path, **response)
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)

    def _record_command(self, args, cwd, **kwargs):
        if args[0] in {"claude", "codex", "gemini", "agy"}:
            self.agent_calls += 1
        return super()._record_command(args, cwd, **kwargs)

    def _maybe_advance_git_head_for_agent_pr(self, text):
        if '"kind": "issue_implementation"' in text or "<!-- AGENT_PR:" in text:
            state = json.loads(self.path.read_text())
            assert not state["opened"], "replacement PR creation"
            state["opened"] = True
            self.path.write_text(json.dumps(state))
            self._move_head(state["pr_payload"]["headRefOid"], cwd=self._last_agent_cwd)


class ReviewBoundary(Exception):
    pass


def drive(path, workdir, *, create, direct, strict):
    from coding_review_agent_loop import git_transport, workdir_claims, validated_agent
    # As in conftest, an invalid fake agent response must fail locally rather
    # than invoking a real provider from this independent interpreter.
    validated_agent.attempt_repair = lambda *args, **kwargs: None
    workdir_claims._set_claim_root_for_tests(workdir / "claims")
    def import_ref(destination, source_ref, tracking_ref, *, runner, **kwargs):
        if source_ref.startswith("refs/pull/"):
            number = source_ref.split("/")[2]
            runner.run(("git", "fetch", "origin", f"+pull/{number}/head:{tracking_ref}"), cwd=destination)
            return runner.pr_payload["headRefOid"]
        runner.run(("git", "fetch", "origin"), cwd=destination)
        return runner.git_head
    git_transport.import_ref = import_ref
    runner = ExternalRunner(path, create=create)
    if create and strict:
        state = json.loads(path.read_text())
        state["pr_branch_protection_payload"] = {"contexts": ["final-ci/exact-head"]}
        state["rest_pr"]["body"] = state["pr_payload"]["body"] = "Fixes #56"
        path.write_text(json.dumps(state))
    config = make_config(workdir, managed_ci=True, managed_ci_pr_mode=direct,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=not strict,
        max_rounds=1)
    # Deterministic external entropy; all authentication and publication remain real.
    managed_ci.secrets.token_urlsafe = lambda _: "opening-nonce"
    def boundary(*args, **kwargs):
        raise ReviewBoundary("transport interruption" if create else "review reached")
    pr_loop.sync_reviewer_pr_before_review = boundary
    captured = {}
    real_activate = pr_loop.activate_managed_ci
    def activate(*args, **kwargs):
        result = real_activate(*args, **kwargs)
        handoff = kwargs["managed_resume"].issue_created_handoff
        assert handoff is not None and handoff.issue_number == 56
        captured["authorization_plan_hash"] = handoff.approved_plan_hash
        captured["authorization_comment_id"] = handoff.authorization_comment_id
        return result
    pr_loop.activate_managed_ci = activate
    # Stop after real recovery/activation has reached the reviewer transport boundary.
    try:
        if direct:
            pr_loop.run_pr_loop(runner, pr_number=77, config=config, managed_ci_issue_number=56)
        else:
            issue_loop.run_issue_loop(runner, issue_number=56, config=config,
                                      plan_first=True, implement_after_approval=True)
    except ReviewBoundary:
        from coding_review_agent_loop.github import get_issue_context
        from coding_review_agent_loop.issue_pr_handoff import find_latest_issue_pr_handoff
        issue = get_issue_context(runner, config=config, issue_number=56)
        binding = find_latest_issue_pr_handoff(issue.comments, issue_number=56, repo=config.repo)
        assert binding is not None and binding.pr_number == 77
        plan = pr_loop.recover_approved_plan_context(issue.comments, expected_hash=binding.plan_hash)
        assert plan.is_available
        captured["plan_hash"] = plan.plan_hash
        if not strict:
            assert captured["authorization_plan_hash"] == plan.plan_hash
        return {"boundary": True, "pid": os.getpid(), "agent_calls": runner.agent_calls, **captured,
                "commands": [cmd for cmd, _ in runner.commands]}
    raise AssertionError("orchestration did not reach the review boundary")


if __name__ == "__main__":
    mode, path = sys.argv[1:3]
    if mode == "cli":
        request = json.load(sys.stdin)
        with contextlib.redirect_stdout(io.StringIO()):
            response = server(Path(path), **request)
        print(json.dumps(response))
    else:
        response = drive(Path(path), Path(sys.argv[3]), create=mode == "create",
                         direct=sys.argv[4] == "direct", strict=sys.argv[5] == "strict")
        print("PROCESS_RESULT=" + json.dumps(response))
