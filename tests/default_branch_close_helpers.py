"""Scripted live GitHub evidence for staged-parent missed-close recovery."""

import json
from pathlib import Path

from agent_loop_helpers import FakeRunner, child_pr_handoff_comment, pr_payload_for_state
from coding_review_agent_loop.runner import CommandResult

MERGE_SHA = "c" * 40


class DefaultCloseRunner(FakeRunner):
    def __init__(self, *, child=42, pr_number=7):
        super().__init__(
            issue_payloads_by_number={child: {"state": "open"}},
            issue_comments_by_number={child: [child_pr_handoff_comment(child, pr_number)]},
            pr_payloads_by_number={pr_number: pr_payload_for_state(pr_number, "MERGED")},
        )
        self.child = child
        self.pr_number = pr_number
        self.live_pr = {
            "merged": True, "merged_at": "2026-01-01T00:00:00Z",
            "base": {"ref": "main", "repo": {"full_name": "OWNER/REPO"}},
            "body": f"Fixes #{child}",
        }
        self.default = {"default_branch": "main"}
        self.commit = {"mergeCommit": {"oid": MERGE_SHA}}
        self.state = "open"
        self.close_rc = 0
        self.confirm = True
        self.unreadable = set()
        self.closes = []
        self.evidence_commands = []

    def run(self, args, *, cwd, **kwargs):
        cmd = [str(a) for a in args]
        self.evidence_commands.append(cmd)
        def answer(payload, rc=0):
            return CommandResult(cmd, Path(cwd), json.dumps(payload), "", rc)
        if cmd[:3] == ["gh", "issue", "close"]:
            self.closes.append(cmd)
            if self.close_rc == 0 and self.confirm:
                self.state = "closed"
            return answer({}, self.close_rc)
        if cmd[:3] == ["gh", "pr", "view"] and "mergeCommit" in cmd:
            return answer(self.commit, int("commit" in self.unreadable))
        if cmd[:2] == ["gh", "api"]:
            endpoint = cmd[2]
            if endpoint in self.unreadable:
                return answer(None, 1)
            if endpoint == f"repos/OWNER/REPO/pulls/{self.pr_number}":
                return answer(self.live_pr)
            if endpoint == "repos/OWNER/REPO":
                return answer(self.default)
            if endpoint == f"repos/OWNER/REPO/issues/{self.child}":
                return answer({"state": self.state})
            if endpoint.startswith(f"repos/OWNER/REPO/issues/{self.pr_number}/comments?"):
                return answer([])
        return super().run(args, cwd=cwd, **kwargs)
