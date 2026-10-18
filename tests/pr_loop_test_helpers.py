"""Non-collected PR-loop test helpers shared with other test files (#1541).

These helpers used to live in tests/test_orchestrator_pr.py, and other test
files imported them from there.  tests/test_orchestrator_pr.py imports them
back under the same names.
"""

import json
import re
from types import SimpleNamespace

import pytest

import coding_review_agent_loop.architecture_contract as architecture_contract
from coding_review_agent_loop.cli import AgentLoopError, run_pr_loop
from coding_review_agent_loop.protocol import ReviewItemDisposition, UnresolvedReviewItem
from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata
from coding_review_agent_loop.round_transport import decode_mapping, encode_mapping
from coding_review_agent_loop.runner import CommandResult

from agent_loop_helpers import (
    FakeRunner,
    make_config,
    structured_coder_followup,
    structured_pr_review,
)


def _carried_ci_obligations() -> tuple[UnresolvedReviewItem, ...]:
    return tuple(
        UnresolvedReviewItem(
            item_id=f"item-{index}",
            reviewer=reviewer,
            source_round=1,
            text=f"{kind} failed on the previous head.",
            status="blocking",
            source_status="blocking",
            authority="machine",
            obligation_kind=kind,
            lifecycle="qualification_ready",
            failed_head_sha="old-head",
            candidate_head_sha="abc123",
            obligation_identity=f"{kind}:item-{index}",
        )
        for index, (reviewer, kind) in enumerate(
            (
                ("GitHub managed exact-head CI", "managed-exact-head-ci"),
                ("GitHub PR checks", "github-pr-checks"),
            ),
            start=1,
        )
    )


def _carried_ci_review_comment(items):
    dispositions = tuple(
        ReviewItemDisposition(item.item_id, "Codex", "resolved") for item in items
    )
    return _attach_round_metadata(
        structured_pr_review(
            reviewer="OpenAI Codex",
            state="approved",
            summary="The reviewed repair is approved.",
            prior_item_dispositions=[
                {"item_id": item.item_id, "disposition": "resolved"}
                for item in items
            ],
        ),
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject="abc123",
            prior_items=items,
            dispositions=dispositions,
            state="approved",
        ),
    )


def _staged_review(*, reviewer, state="approved", blocking_items=None, dispositions=None, resolved=False):
    return (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": state,
                "summary": f"{reviewer} review",
                "blocking_items": blocking_items or [],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": dispositions or [],
            }
        )
        + ("\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->" if resolved else "")
        + f"\n<!-- AGENT_STATE: {state} -->\n-- {reviewer}"
    )


def _staged_config(tmp_path, **overrides):
    values = dict(
        reviewer=("codex", "gemini", "antigravity"),
        pr_review_policy="primary-then-panel",
        primary_reviewer="codex",
        max_rounds=6,
    )
    values.update(overrides)
    return make_config(tmp_path, **values)


def _audit_phases(runner):
    phases = []
    for comment in runner.comments:
        if comment.startswith("PR review scheduling audit:"):
            match = re.search(r"phase: ([a-z-]+); head: ([^;]+);", comment)
            assert match is not None, comment
            phases.append((match.group(1), match.group(2)))
    return phases


def _posted_scheduler_metadata(runner):
    return [
        record.metadata
        for record in architecture_contract._extract_round_metadata_records(
            [
                SimpleNamespace(body=comment["body"])
                for comment in runner.pr_payload.get("comments", [])
            ],
            flow="pr",
        )
        if record.metadata.scheduler_metadata_status != "absent"
    ]


def _rewrite_pr_metadata(runner, transform):
    """Rewrite persisted round-metadata payloads in place (legacy simulation).

    ``transform`` receives the decoded payload mapping and returns a new
    mapping, or ``None`` to leave the comment unchanged.
    """
    from coding_review_agent_loop.round_transport import (
        ROUND_RESUME_MARKER_RE,
        decode_mapping,
        encode_mapping,
    )

    for comment in runner.pr_payload.get("comments", []):
        body = comment["body"]
        matches = list(ROUND_RESUME_MARKER_RE.finditer(body))
        if not matches:
            continue
        match = matches[-1]
        payload = transform(dict(decode_mapping(match.group("payload"))))
        if payload is None:
            continue
        comment["body"] = (
            body[: match.start("payload")] + encode_mapping(payload) + body[match.end("payload"):]
        )


def _m943_partial_pr_round(tmp_path, **payload):
    """Codex and Gemini review; Antigravity's backend fails before it posts."""
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[],
        **payload,
    )
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path))
    posted = _posted_scheduler_metadata(runner)
    assert posted and all(
        tuple(item.scheduler_contract["required_reviewers"]) == ("Codex", "Gemini", "Antigravity")
        for item in posted
    )
    return runner


def _m943_amendment_from_error(message):
    start = message.index("Reviewer board amendment:")
    template = message[start:]
    return template.replace(
        "<why the removed reviewer cannot be reached>", "Antigravity quota exhausted."
    )


def _m943_append(runner, body, login="operator"):
    comments = runner.pr_payload.setdefault("comments", [])
    comments.append(
        {
            "author": {"login": login},
            "createdAt": f"2026-05-23T00:00:{len(comments):02d}Z",
            "body": body,
        }
    )


_EVIDENCE_TEXT = (
    "Run tests/test_sandboxed_permissions_live.py against authenticated CLIs at this exact "
    "head and attach the output"
)


def _evidence_review(
    *,
    state="approved",
    evidence=(),
    dispositions=(),
    blocking=(),
    hr=False,
    reviewer="OpenAI Codex",
    summary="Reviewed.",
):
    payload = {
        "schema_version": 1,
        "kind": "pr_review",
        "state": state,
        "summary": summary,
        "blocking_items": list(blocking),
        "same_pr_followups": [],
        "future_followups": [],
        "prior_item_dispositions": list(dispositions),
    }
    if evidence:
        payload["exact_head_evidence_requests"] = list(evidence)
    return (
        json.dumps(payload)
        + ("\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->" if hr else "")
        + f"\n<!-- AGENT_STATE: {state} -->\n-- {reviewer}"
    )


def _signed(body, index):
    return {
        "author": {"login": "maintainer"},
        "createdAt": f"2026-05-24T00:{index:02d}:00Z",
        "url": f"https://github.com/OWNER/REPO/pull/77#issuecomment-{900 + index}",
        "body": f"{body}\n\n-- Human Reviewer",
    }


def _resolve(item_id="item-1", note="The signed live run covers this exact head."):
    return {"item_id": item_id, "disposition": "resolved", "note": note}


SB_HEAD = "a" * 40


class _StepBackRunner(FakeRunner):
    """FakeRunner with Git-SHA-shaped heads and scripted ``git diff`` output."""

    def __init__(self, *, name_status="M\tsrc/spool.py\n", hunks="", shift_head=None, **kwargs):
        kwargs.setdefault("pr_payload", {"headRefOid": SB_HEAD})
        kwargs.setdefault("git_head", SB_HEAD)
        super().__init__(**kwargs)
        self.scripted_name_status = name_status
        self.scripted_hunks = hunks
        # The scripted hunks describe the step-back coder's push only; diffs between
        # earlier heads stay empty so older locations are not shifted by them.
        self.shift_head = shift_head or f"{3:040x}"

    def _maybe_advance_pr_head_for_coder_followup(self, cmd) -> None:
        if '"kind": "coder_followup"' not in "\n".join(cmd):
            return
        self._coder_followup_counter += 1
        new_head = f"{self._coder_followup_counter:040x}"
        self.pr_payload["headRefOid"] = new_head
        self._move_head(new_head, cwd=self._last_agent_cwd)

    def run(self, args, *, cwd, input_text=None, check=True, env=None):
        cmd = [str(arg) for arg in args]
        if cmd[:1] == ["git"] and "diff" in cmd and "--name-status" in cmd and "-M" in cmd:
            cmd, cwd_path = self._record_command(cmd, cwd)
            return CommandResult(cmd, cwd_path, self.scripted_name_status, "", 0)
        if cmd[:1] == ["git"] and "diff" in cmd and "-U0" in cmd:
            cmd, cwd_path = self._record_command(cmd, cwd)
            touches = self.shift_head in cmd
            return CommandResult(cmd, cwd_path, self.scripted_hunks if touches else "", "", 0)
        return super().run(args, cwd=cwd, input_text=input_text, check=check, env=env)


def _step_back_coders(count=3):
    return [
        structured_coder_followup(addressed_items=[f"item-{n}"]) for n in range(1, count)
    ] + [
        structured_coder_followup(
            addressed_items=["item-3"],
            summary="Generalization: any spool record that fails a validation step is malformed.",
        )
    ]
