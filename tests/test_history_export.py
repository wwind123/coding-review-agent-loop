"""Outcome history export (#1433): authenticated, read-only, never estimated."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from coding_review_agent_loop.agents.registry import agent_signature
from coding_review_agent_loop.checks import (
    _ci_infrastructure_stop_message,
    _format_ci_infrastructure_comment,
)
from coding_review_agent_loop.ci_health import CiInfrastructureStall, StalledCheck
from coding_review_agent_loop.cli import main
from coding_review_agent_loop.comment_rendering import render_agent_unavailable_comment
from coding_review_agent_loop.followups import _append_plan_approved_followups_marker
from coding_review_agent_loop.github import (
    AuthenticatedComment,
    AuthenticatedCommentView,
    IssueOutcomeHistory,
    LinkedPullRequest,
)
from coding_review_agent_loop.history_export import (
    KIND_HUMAN_DECISION_STOP,
    KIND_INFRASTRUCTURE,
    KIND_PLANNING_STOP,
    KIND_PR_STOP,
    derive_history_row,
    export_history,
    load_tool_commits,
    read_only_history_config,
    recognize_diagnostic,
    render_csv,
    render_json,
)
from coding_review_agent_loop.panel_evidence import (
    plan_issue_text_digest_for,
    plan_primary_stall_message,
)
from coding_review_agent_loop.plan_first_loop import render_plan_scheduling_diagnostic
from coding_review_agent_loop.pr_loop import render_pr_scheduling_diagnostic
from coding_review_agent_loop.protocol import AgentUnavailable
from coding_review_agent_loop.protocol_markers import named_reserved_marker_tokens
from coding_review_agent_loop.review_step_back import render_step_back_human_decision
from coding_review_agent_loop.issue_pr_handoff import format_issue_pr_handoff_comment
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    PostedRoundRecord,
    _approved_plan_hash,
    _attach_round_metadata,
)
from coding_review_agent_loop.runner import CommandResult

REPO = "OWNER/REPO"
ISSUE = 41
PR = 77
ACTOR = ("agent-loop-bot", 4242)
FOREIGN = ("mallory", 666)
CODEX, CLAUDE = "Codex", "Claude"


def at(minute: int, hour: int = 10) -> str:
    return f"2026-10-01T{hour:02d}:{minute:02d}:00Z"


def after_cutoff(minute: int, hour: int = 10) -> str:
    """A time after the step-back feature cutoff (2026-10-04T00:00:00Z)."""
    return f"2026-10-06T{hour:02d}:{minute:02d}:00Z"


def record(flow: str, role: str, agent: str, round_number: int, **fields) -> str:
    fields.setdefault("subject", f"{flow}-subject-{round_number}")
    metadata = PostedRoundMetadata(
        flow=flow, role=role, agent=agent, round_number=round_number, **fields
    )
    return str(_attach_round_metadata(f"{agent} {role} round {round_number}", metadata))


def plan_coder(round_number: int, plan: str = "Plan text.", **fields) -> str:
    fields.setdefault("plan_execution_mode", "plan-only")
    return record("plan", "coder", "Claude", round_number, canonical_plan=plan, **fields)


def plan_review(agent: str, round_number: int, state: str, **fields) -> str:
    fields.setdefault("plan_execution_mode", "plan-only")
    return record("plan", "reviewer", agent, round_number, state=state, **fields)


def pr_record(role: str, agent: str, round_number: int, state: str | None = None) -> str:
    return record(
        "pr", role, agent, round_number, state=state,
        pr_board_root={"seats": [CODEX]}, pr_board_identity={"seats": [CODEX]},
    )


def comment(
    comment_id: int, body: str, *, surface: str = f"issue#{ISSUE}", author=ACTOR,
    created: str | None = None, updated: str | None = None,
) -> AuthenticatedComment:
    created = created or at(comment_id % 60, 10 + comment_id // 60)
    return AuthenticatedComment(
        surface=surface, comment_id=comment_id, author_login=author[0], author_id=author[1],
        created_at=created, updated_at=updated or created, body=body,
    )


def view(surface: str, *comments: AuthenticatedComment) -> AuthenticatedCommentView:
    ordered = sorted(comments, key=lambda item: item.comment_id)
    return AuthenticatedCommentView(
        surface=surface, actor_login=ACTOR[0], actor_id=ACTOR[1],
        authored=tuple(item for item in ordered if item.author_id == ACTOR[1]),
        ignored_foreign=tuple(
            item for item in ordered
            if item.author_id != ACTOR[1] and named_reserved_marker_tokens(item.body)
        ),
    )


def issue_view(*comments):
    return view(f"issue#{ISSUE}", *comments)


def pr_view(*comments, number: int = PR):
    return view(f"pr#{number}", *comments)


def pr_comment(comment_id: int, body: str, **kwargs) -> AuthenticatedComment:
    return comment(comment_id, body, surface=f"pr#{kwargs.pop('number', PR)}", **kwargs)


def history(
    *, title="Issue title", body="Issue body", state="OPEN", created=at(0, 9), closed=None,
    body_revisions=(), title_renames=(), linked=(),
) -> IssueOutcomeHistory:
    return IssueOutcomeHistory(
        number=ISSUE, title=title, body=body, state=state, created_at=created,
        closed_at=closed, body_revisions=body_revisions, title_renames=title_renames,
        linked_prs=tuple(linked),
    )


def merged_pr(number: int = PR, when: str = at(50, 12)) -> LinkedPullRequest:
    return LinkedPullRequest(number=number, state="MERGED", merged_at=when, closed_at=when)


def open_pr(number: int = PR) -> LinkedPullRequest:
    return LinkedPullRequest(number=number, state="OPEN", merged_at=None, closed_at=None)


def derive(hist, issue, prs=None, **kwargs):
    return derive_history_row(repo=REPO, history=hist, issue_view=issue, pr_views=prs or {}, **kwargs)


def handoff(comment_id: int, plan: str = "Plan text.") -> AuthenticatedComment:
    """The orchestrator's issue-to-PR handoff record for an accepted plan."""
    return comment(comment_id, format_issue_pr_handoff_comment(
        issue_number=ISSUE, pr_number=PR, pr_url=f"https://github.com/{REPO}/pull/{PR}",
        pr_head_sha="a" * 40, flow="approved-plan-implementation",
        plan_hash=_approved_plan_hash(plan),
    ))


def approval_announcement(comment_id: int, plan: str = "Plan text.") -> AuthenticatedComment:
    """The plan-approval announcement an approved plan-first round posts (plan-only too)."""
    return comment(comment_id, _append_plan_approved_followups_marker(
        f"Issue #{ISSUE} plan approved.\n-- coding-review-agent-loop",
        issue_number=ISSUE, plan_hash=_approved_plan_hash(plan), mode="ignore",
    ))


def approved_plan_comments(start: int = 1, round_number: int = 1, **coder_fields):
    return (
        comment(start, plan_coder(round_number, **coder_fields)),
        comment(start + 1, plan_review(CODEX, round_number, "approved")),
        handoff(start + 2),
    )


def step_back_stop_message() -> str:
    return render_step_back_human_decision(
        phase="plan", measurements="Plan-growth measurements: 3 steps.", step_back_round=2,
        blocks=2, threshold=2, alternative="Narrow the scope.",
    )


# --- export-authenticated-only ------------------------------------------------


def test_foreign_copies_of_records_never_count():
    foreign_round = plan_review(CODEX, 2, "blocking")
    row = derive(
        history(state="CLOSED", closed=at(30, 11)),
        issue_view(
            *approved_plan_comments(),
            comment(13, plan_coder(2), author=FOREIGN),
            comment(14, foreign_round, author=FOREIGN),
            comment(15, render_plan_scheduling_diagnostic("round 2", step_back_stop_message()), author=FOREIGN),
        ),
    )
    assert row["plan_rounds"] == {"value": 1, "evidence": "verified"}
    assert row["human_decision_stops"] == {"value": 0, "evidence": "verified"}
    assert row["ignored_foreign_records"] == 2
    assert row["route_taken"] == ["plan-first"]


class _ReadOnlyRunner:
    """Answers the export's GitHub reads and records every command it is given."""

    dry_run = False

    def __init__(self, issue_comments, pr_comments, outcome):
        self.commands: list[list[str]] = []
        self._issue_comments = issue_comments
        self._pr_comments = pr_comments
        self._outcome = outcome

    def run(self, command, *, cwd, check=False, **_kwargs):
        self.commands.append(list(command))
        joined = " ".join(command)
        if command[1:3] == ["api", "user"]:
            payload = {"login": ACTOR[0], "id": ACTOR[1]}
        elif command[1:3] == ["api", "graphql"]:
            payload = {"data": {"repository": {"issue": self._outcome}}}
        elif f"issues/{ISSUE}/comments" in joined:
            payload = self._issue_comments
        elif f"issues/{PR}/comments" in joined:
            payload = self._pr_comments
        else:
            raise AssertionError(f"unexpected command {command}")
        return CommandResult(args=list(command), cwd=cwd, stdout=json.dumps(payload), stderr="", returncode=0)


def _rest(comment_id: int, body: str, author=ACTOR, created: str | None = None):
    created = created or at(comment_id)
    return {
        "id": comment_id, "user": {"login": author[0], "id": author[1]},
        "created_at": created, "updated_at": created, "body": body,
    }


def test_export_driver_reads_only_and_ignores_foreign_records():
    runner = _ReadOnlyRunner(
        issue_comments=[
            _rest(1, plan_coder(1)),
            _rest(2, plan_review(CODEX, 1, "approved")),
            _rest(3, plan_coder(2), author=FOREIGN),
        ],
        pr_comments=[
            _rest(10, pr_record("coder", "Claude", 1)),
            _rest(11, pr_record("reviewer", CODEX, 1, "approved")),
        ],
        outcome={
            "number": ISSUE, "title": "T", "body": "B", "state": "CLOSED",
            "createdAt": at(0, 9), "closedAt": at(50, 12), "lastEditedAt": None,
            "includesCreatedEdit": False,
            "userContentEdits": {"totalCount": 0, "nodes": []},
            "timelineItems": {"totalCount": 0, "nodes": []},
            "closedByPullRequestsReferences": {
                "totalCount": 1,
                "nodes": [{"number": PR, "state": "MERGED", "mergedAt": at(50, 12), "closedAt": at(50, 12)}],
            },
        },
    )
    rows = export_history(runner, config=read_only_history_config(REPO), issue_numbers=(ISSUE,))
    assert [row["issue_number"] for row in rows] == [ISSUE]
    assert rows[0]["plan_rounds"] == {"value": 1, "evidence": "verified"}
    assert rows[0]["pr_rounds"] == {"value": 1, "evidence": "verified"}
    assert rows[0]["merged"] == {"value": True, "evidence": "verified"}
    assert rows[0]["ignored_foreign_records"] == 1
    for command in runner.commands:
        assert command[:2] == ["gh", "api"]
        assert not {"-X", "--method", "POST", "PATCH", "DELETE", "--input"} & set(command)
        if command[2] == "graphql":
            query = next(arg for arg in command if arg.startswith("query="))
            assert query.startswith("query=query(") and "mutation" not in query


# --- export-route-classification ------------------------------------------------


def test_route_plan_first_with_pr():
    row = derive(
        history(state="CLOSED", closed=at(50, 12), linked=[merged_pr()],
                body="Labels say direct; prose must never pick the route."),
        issue_view(
            comment(1, plan_coder(1)), comment(2, plan_review(CODEX, 1, "blocking")),
            comment(3, plan_coder(2)), comment(4, plan_review(CODEX, 2, "approved")),
        ),
        {PR: pr_view(
            pr_comment(20, pr_record("coder", "Claude", 1)),
            pr_comment(21, pr_record("reviewer", CODEX, 1, "blocking")),
            pr_comment(22, pr_record("coder", "Claude", 2)),
            pr_comment(23, pr_record("reviewer", CODEX, 2, "approved")),
        )},
    )
    assert row["route"] == "plan-first"
    assert row["plan_rounds"] == {"value": 2, "evidence": "verified"}
    assert row["pr_rounds"] == {"value": 2, "evidence": "verified"}
    assert row["final_plan_size"] == {"value": len("Plan text."), "evidence": "verified"}
    assert row["outcome_complete"] is True


def test_route_direct_with_pr():
    row = derive(
        history(state="CLOSED", closed=at(50, 12), linked=[merged_pr()], title="Plan-first: please"),
        issue_view(comment(1, "Operator note: plan-first would be nice.")),
        {PR: pr_view(
            pr_comment(20, pr_record("coder", "Claude", 1)),
            pr_comment(21, pr_record("reviewer", CODEX, 1, "approved")),
        )},
    )
    assert row["route_taken"] == ["direct"]
    assert row["plan_rounds"] == {"value": 0, "evidence": "verified"}
    assert row["pr_rounds"] == {"value": 1, "evidence": "verified"}


def _discuss(comment_id: int, round_number: int, *, final: bool = False):
    role = "summary" if final else "debater"
    return comment(comment_id, record("discuss", role, CODEX, round_number, is_final=final))


def test_route_discuss_only_and_discuss_then_plan():
    discuss_only = derive(history(), issue_view(_discuss(1, 1), _discuss(2, 1, final=True)))
    assert discuss_only["route_taken"] == ["discuss"]
    assert "discuss-final" in discuss_only["terminal_states"]
    assert discuss_only["plan_rounds"] == {"value": 0, "evidence": "verified"}

    then_plan = derive(
        history(),
        issue_view(_discuss(1, 1), _discuss(2, 1, final=True), *approved_plan_comments(start=5)),
    )
    assert then_plan["route"] == "discuss then plan-first"
    assert then_plan["plan_rounds"] == {"value": 1, "evidence": "verified"}


# --- export-legacy-fields ------------------------------------------------------


def test_legacy_records_render_unavailable_without_failing():
    # Written after the step-back cutoff, yet without era fields: ambiguous.
    legacy_coder = record("plan", "coder", "Claude", 1, canonical_plan="Old plan.")
    legacy_review = record("plan", "reviewer", CODEX, 1, state="approved")
    row = derive(
        history(state="CLOSED", closed=after_cutoff(30, 11)),
        issue_view(
            comment(1, legacy_coder, created=after_cutoff(1)),
            comment(2, legacy_review, created=after_cutoff(2)),
        ),
    )
    assert row["plan_rounds"] == {"value": 1, "evidence": "verified"}
    assert row["step_backs"] == {"value": None, "evidence": "unavailable"}
    assert row["tool_commit"] == {"value": None, "evidence": "unavailable"}
    assert row["snapshot"]["plan_issue_digest"] is None
    assert row["snapshot"]["status"] == "never-edited"


def test_tool_commit_requires_an_explicit_attribution_join(tmp_path):
    (tmp_path / "r1-usage-summary.json").write_text(
        json.dumps({"run_id": "r1", "tool_provenance": {"commit": "abc1234"}})
    )
    (tmp_path / "r2-usage-summary.json").write_text(
        json.dumps({"run_id": "r2", "tool_provenance": {"commit": "def5678"}})
    )
    attribution = [
        json.dumps({"record": "run-start", "run_id": "r1", "repo": REPO, "issue_number": ISSUE, "pr_number": None}),
        json.dumps({"record": "run-start", "run_id": "r2", "repo": "OTHER/REPO", "issue_number": ISSUE}),
        "not json",
    ]
    index = load_tool_commits(tmp_path, attribution, repo=REPO)
    assert index == {("issue", ISSUE): {"abc1234"}}
    row = derive(
        history(state="CLOSED", closed=at(30, 11)), issue_view(*approved_plan_comments()),
        tool_commits=index[("issue", ISSUE)],
    )
    assert row["tool_commit"] == {"value": ["abc1234"], "evidence": "verified"}


# --- snapshot rows -------------------------------------------------------------


def test_snapshot_digest_verified_selects_text_the_run_received():
    v1, v2 = "Body v1", "Body v2 edited during the planner turn"
    row = derive(
        history(body=v2, created=at(0, 9), body_revisions=((at(0, 9), v1), (at(5, 10), v2))),
        issue_view(comment(30, plan_coder(1, plan_issue_digest=plan_issue_text_digest_for("Issue title", v1)))),
    )
    snapshot = row["snapshot"]
    assert snapshot["status"] == "digest-verified"
    assert snapshot["body"] == v1
    assert snapshot["as_of"] == at(0, 9)


def test_snapshot_digest_tie_exports_normalized_text_only():
    v1, v2, v3 = "a\nb", "a  \r\nb", "something else"
    digest = plan_issue_text_digest_for("Issue title", v1)
    tie = derive(
        history(body=v3, body_revisions=((at(0, 9), v1), (at(1, 10), v2), (at(9, 11), v3))),
        issue_view(comment(30, plan_coder(1, plan_issue_digest=digest))),
    )["snapshot"]
    assert tie["status"] == "digest-normalized"
    assert tie["body"] == "a\nb" and tie["body"] != v2
    assert tie["body"] != v3

    a, b = "Text A", "Text B"
    repeat = derive(
        history(body=a, body_revisions=((at(0, 9), a), (at(1, 10), b), (at(2, 10), a))),
        issue_view(comment(30, plan_coder(1, plan_issue_digest=plan_issue_text_digest_for("Issue title", a)))),
    )["snapshot"]
    assert repeat["status"] == "digest-verified"
    assert repeat["body"] == a


def test_snapshot_fallbacks():
    never = derive(history(body="Original"), issue_view(*approved_plan_comments()))["snapshot"]
    assert never == {
        "status": "never-edited", "title": "Issue title", "body": "Original",
        "as_of": at(0, 9), "plan_issue_digest": None,
    }

    direct = derive(
        history(body="v3", body_revisions=((at(0, 9), "v1"), (at(5, 10), "v2"), (at(30, 11), "v3")),
                linked=[open_pr()]),
        issue_view(),
        {PR: pr_view(pr_comment(20, pr_record("coder", "Claude", 1), created=at(10, 10)))},
    )["snapshot"]
    assert direct["status"] == "time-inferred" and direct["body"] == "v2"

    mismatch = derive(
        history(body="v2", body_revisions=((at(0, 9), "v1"), (at(5, 10), "v2"))),
        issue_view(comment(30, plan_coder(1, plan_issue_digest="0123456789abcdef"))),
    )["snapshot"]
    assert mismatch["status"] == "unavailable"
    assert mismatch["title"] is None and mismatch["body"] is None

    missing = derive(history(body_revisions=None), issue_view(*approved_plan_comments()))["snapshot"]
    assert missing["status"] == "unavailable" and missing["body"] is None


def test_title_rename_chain_must_reconcile():
    renamed = derive(
        history(title="New", title_renames=((at(5, 10), "Old", "New"),)),
        issue_view(comment(1, plan_coder(1, plan_issue_digest=plan_issue_text_digest_for("Old", "Issue body")))),
    )["snapshot"]
    assert renamed["status"] == "digest-verified" and renamed["title"] == "Old"
    broken = derive(
        history(title="New", title_renames=((at(5, 10), "Old", "Mid"), (at(6, 10), "Other", "New"))),
        issue_view(*approved_plan_comments()),
    )["snapshot"]
    assert broken["status"] == "unavailable"


# --- export-human-stop -----------------------------------------------------------


STEP_BACK = ({"phase": "plan", "reviewer": CODEX, "trigger_round": 1},)


def test_human_decision_stops_from_recognized_diagnostics():
    converges = derive(
        history(),
        issue_view(
            comment(1, plan_coder(1)),
            comment(2, plan_review(CODEX, 1, "blocking", step_back_entries=STEP_BACK)),
            comment(3, plan_coder(2)),
            comment(4, plan_review(CODEX, 2, "approved", step_back_entries=STEP_BACK)),
            handoff(5),
        ),
    )
    assert converges["human_decision_stops"] == {"value": 0, "evidence": "verified"}
    assert converges["step_backs"] == {"value": 1, "evidence": "verified"}

    stops = derive(
        history(),
        issue_view(
            comment(1, plan_coder(1)),
            comment(2, plan_review(CODEX, 1, "blocking", step_back_entries=STEP_BACK)),
            comment(3, plan_coder(2)),
            comment(4, plan_review(CODEX, 2, "blocking", step_back_entries=STEP_BACK)),
            comment(5, render_plan_scheduling_diagnostic("round 3", step_back_stop_message())),
            comment(6, "Parking this issue (operator comment from the same account)."),
        ),
    )
    assert stops["human_decision_stops"] == {"value": 1, "evidence": "verified"}
    assert stops["outcome_complete"] is True
    assert stops["plan_rounds"] == {"value": 2, "evidence": "verified"}

    open_ended = derive(
        history(),
        issue_view(
            comment(1, plan_coder(1)),
            comment(2, plan_review(CODEX, 1, "blocking", step_back_entries=STEP_BACK)),
        ),
    )
    assert open_ended["human_decision_stops"] == {"value": None, "evidence": "unavailable"}
    assert open_ended["step_backs"] == {"value": None, "evidence": "unavailable"}
    assert open_ended["outcome_complete"] is False


def test_recognizers_accept_the_bytes_each_poster_renders():
    step_back = recognize_diagnostic(render_plan_scheduling_diagnostic("round 3", step_back_stop_message()))
    assert (step_back.kind, step_back.reason, step_back.round_number) == (
        KIND_HUMAN_DECISION_STOP, "plan-step-back", 3,
    )
    stall = recognize_diagnostic(render_plan_scheduling_diagnostic(
        "round 4", plan_primary_stall_message(streak=3, threshold=3, plan_chars=1200)
    ))
    assert (stall.kind, stall.reason) == (KIND_HUMAN_DECISION_STOP, "plan-primary-stall")
    other = recognize_diagnostic(render_plan_scheduling_diagnostic("startup", "history is undecodable"))
    assert (other.kind, other.round_number) == (KIND_PLANNING_STOP, None)
    pr_stop = recognize_diagnostic(render_pr_scheduling_diagnostic("round 2", "panel unsafe"))
    assert pr_stop.kind == KIND_PR_STOP

    stall_check = CiInfrastructureStall(checks=(StalledCheck(name="test", kind="check_run", reason="queued_too_long", check_id=7, run_id="9", url=None, age_seconds=9000.0),))
    for body in (
        _format_ci_infrastructure_comment(PR, stall_check),
        _ci_infrastructure_stop_message(PR, stall_check, ["[item-1] carried"]),
    ):
        ci = recognize_diagnostic(body)
        assert (ci.kind, ci.reason) == (KIND_INFRASTRUCTURE, "ci-infrastructure-stall")

    unavailable = render_agent_unavailable_comment(
        AgentUnavailable(
            schema_version=1, kind="agent_unavailable", retryable=False, category="tooling",
            summary="Completion recovery exhausted its bounded pass.",
            suggested_action="Inspect the completion-recovery log.",
        ),
        signature=agent_signature("claude"),
    )
    recovered = recognize_diagnostic(unavailable)
    assert recovered.kind == KIND_INFRASTRUCTURE
    assert recovered.reason == "agent-unavailable-tooling"
    assert recovered.seat_signature == "Anthropic Claude"

    assert recognize_diagnostic(plan_coder(1)) is None
    assert recognize_diagnostic("human decision required: the planner stepped back at round 2") is None


# --- export-outcome-completeness ------------------------------------------------


def test_outcome_completeness_and_availability():
    open_pr_row = derive(
        history(linked=[open_pr()]),
        issue_view(*approved_plan_comments()),
        {PR: pr_view(pr_comment(20, pr_record("coder", "Claude", 1)))},
    )
    assert "plan-approved" in open_pr_row["terminal_states"]
    assert open_pr_row["outcome_complete"] is False
    assert open_pr_row["pr_rounds"]["evidence"] == "unavailable"

    t1, t2 = at(20, 11), at(45, 12)
    merged_late = derive(
        history(linked=[merged_pr(when=t2)]),
        issue_view(*approved_plan_comments()),
        {PR: pr_view(pr_comment(20, pr_record("coder", "Claude", 1), created=t1))},
    )
    assert merged_late["outcome_available_at"] == t2
    assert merged_late["merged"] == {"value": True, "evidence": "verified"}

    t3 = at(59, 15)
    edited = derive(
        history(),
        issue_view(comment(1, plan_coder(1)), comment(2, plan_review(CODEX, 1, "approved"), updated=t3)),
    )
    assert edited["outcome_available_at"] >= t3

    closed_pr = LinkedPullRequest(number=PR + 1, state="CLOSED", merged_at=None, closed_at=at(55, 14))
    done = derive(
        history(state="CLOSED", closed=at(40, 14), linked=[merged_pr(when=at(30, 14)), closed_pr]),
        issue_view(*approved_plan_comments()),
        {
            PR: pr_view(pr_comment(20, pr_record("coder", "Claude", 1))),
            PR + 1: pr_view(number=PR + 1),
        },
    )
    assert done["outcome_complete"] is True
    assert done["outcome_available_at"] == at(55, 14)


# --- export-infra-flagged ------------------------------------------------------------


def _unavailable(summary: str, category: str, agent: str = "claude") -> str:
    return render_agent_unavailable_comment(
        AgentUnavailable(
            schema_version=1, kind="agent_unavailable", retryable=False, category=category,
            summary=summary, suggested_action="Retry later.",
        ),
        signature=agent_signature(agent),
    )


def test_infrastructure_failures_are_flagged_not_counted():
    stall = CiInfrastructureStall(checks=(StalledCheck(name="test", kind="check_run", reason="queued_too_long", check_id=7, run_id="9", url=None, age_seconds=9000.0),))
    row = derive(
        history(linked=[open_pr()]),
        issue_view(
            comment(1, plan_coder(1)),
            comment(2, plan_review(CODEX, 1, "blocking")),
            comment(3, plan_review(CLAUDE, 1, "blocking")),
            comment(4, plan_coder(2)),
            comment(5, plan_review(CODEX, 2, "blocking")),
            comment(6, _unavailable("Claude authentication expired; run claude login.", "permissions")),
            comment(7, plan_coder(3)),
            comment(8, plan_review(CODEX, 3, "blocking")),
            comment(9, plan_coder(4)),
            comment(10, plan_review(CODEX, 4, "approved")),
            comment(11, plan_review(CLAUDE, 4, "approved")),
            comment(12, _unavailable("Completion recovery exhausted its bounded pass.", "tooling")),
        ),
        {PR: pr_view(
            pr_comment(20, pr_record("coder", "Claude", 1)),
            pr_comment(21, pr_record("reviewer", CODEX, 1, "approved")),
            pr_comment(22, _format_ci_infrastructure_comment(PR, stall)),
            pr_comment(23, _ci_infrastructure_stop_message(PR, stall, [])),
        )},
    )
    categories = sorted(entry["category"] for entry in row["infrastructure_failures"])
    assert categories == [
        "agent-unavailable-tooling", "auth-expiry",
        "ci-infrastructure-stall", "ci-infrastructure-stall",
    ]
    assert row["infrastructure_rounds"] == [{"surface": f"issue#{ISSUE}", "flow": "plan", "round": 2}]
    assert row["plan_rounds"] == {"value": 3, "evidence": "verified"}
    assert row["missing_reviewer_records"] == [
        {"surface": f"issue#{ISSUE}", "flow": "plan", "round": 2, "reviewer": CLAUDE, "cause": "auth-expiry"},
        {"surface": f"issue#{ISSUE}", "flow": "plan", "round": 3, "reviewer": CLAUDE, "cause": "unknown"},
    ]
    assert row["infrastructure_coverage"] == "partial"


def test_coverage_complete_when_every_round_is_explained():
    row = derive(
        history(state="CLOSED", closed=at(30, 11)),
        issue_view(
            comment(1, plan_coder(1)),
            comment(2, plan_review(CODEX, 1, "blocking")),
            comment(3, plan_review(CLAUDE, 1, "blocking")),
            comment(4, plan_coder(2)),
            comment(5, plan_review(CODEX, 2, "approved")),
            comment(6, _unavailable("Provider returned 503 Service Unavailable.", "provider")),
        ),
    )
    assert row["infrastructure_failures"][0]["category"] == "backend-outage"
    assert row["missing_reviewer_records"][0]["cause"] == "backend-outage"
    assert row["infrastructure_coverage"] == "complete"
    assert row["plan_rounds"] == {"value": 1, "evidence": "verified"}


# --- rendering and CLI -------------------------------------------------------------


def test_json_and_csv_rendering_are_deterministic():
    row = derive(history(state="CLOSED", closed=at(30, 11)), issue_view(*approved_plan_comments()))
    rendered = render_json([row], repo=REPO)
    assert rendered == render_json([row], repo=REPO)
    payload = json.loads(rendered)
    assert payload["schema_version"] == 1 and payload["rows"][0]["issue_number"] == ISSUE
    csv_text = render_csv([row])
    header, line = csv_text.splitlines()[:2]
    assert "plan_rounds_evidence" in header.split(",")
    assert line.startswith(f"1,{REPO},{ISSUE},closed,plan-first,1,verified")


def test_cli_history_export_writes_requested_format(tmp_path, monkeypatch):
    captured = {}

    def fake_export(runner, *, config, issue_numbers, since, tool_commit_index):
        captured.update(repo=config.repo, issues=issue_numbers, since=since, index=tool_commit_index)
        return [derive(history(state="CLOSED", closed=at(30, 11)), issue_view(*approved_plan_comments()))]

    monkeypatch.setattr("coding_review_agent_loop.history_export.export_history", fake_export)
    output = tmp_path / "rows.csv"
    code = main([
        "history", "export", "--repo", REPO, "--issue", str(ISSUE), "--format", "csv",
        "--since", "2026-10-01", "--output", str(output),
    ])
    assert code == 0
    assert captured["repo"] == REPO and captured["issues"] == (ISSUE,)
    assert captured["since"].isoformat() == "2026-10-01T00:00:00+00:00"
    assert captured["index"] is None
    assert output.read_text().startswith("schema_version,repo,issue_number")
    assert main(["history", "export", "--repo", REPO, "--since", "yesterday"]) == 1


def test_history_export_is_documented():
    root = Path(__file__).resolve().parents[1]
    for path in ("README.md", "docs/local_agent_loop.md"):
        assert "history export" in (root / path).read_text(encoding="utf-8"), path
    architecture_set = [root / "ARCHITECTURE.md", *sorted((root / "docs" / "architecture").glob("*.md"))]
    assert "history export" in "\n\n".join(path.read_text(encoding="utf-8") for path in architecture_set)


# --- review round 1: real publication and failure paths -----------------------


from coding_review_agent_loop.errors import AgentInvocationError  # noqa: E402
from coding_review_agent_loop.plan_review_scheduling import make_plan_contract  # noqa: E402
from coding_review_agent_loop.review_rounds import _format_incomplete_pr_review_comment  # noqa: E402


def _plan_subject_review(agent: str, round_number: int, state: str, subject: str, **fields) -> str:
    return plan_review(agent, round_number, state, subject=subject, **fields)


def _primary_scheduler(round_number: int, subject: str) -> str:
    return record(
        "plan", "summary", "Orchestrator", round_number, subject=subject,
        phase="scheduler-prelaunch",
        scheduler_contract=make_plan_contract((CODEX, CLAUDE), "primary-then-panel", CODEX).as_dict(),
        scheduler_obligation_digest="0" * 16,
        scheduler_selected_reviewers=(CODEX,),
        scheduler_paused_reviewers=((CLAUDE, "primary phase"),),
        scheduler_reasons=("primary phase",),
        scheduler_final_sweep=False, scheduler_force_full=False, scheduler_calls_avoided=1,
        scheduler_phase="primary", scheduler_primary_reviewer=CODEX,
        plan_candidate_key={
            "subject": subject, "aggregate_plan_identity": "aggregate",
            "execution_strategy_identity": "strategy", "risk_test_matrix_identity": "matrix",
            "surfaced_requirement_id_digest": "requirements",
            "execution_strategy_contract_version": 1,
        },
    )


def test_reviewer_votes_alone_are_not_plan_approval():
    """Generalization: only the orchestrator's post-approval records settle a plan.

    Each case is a set of authenticated votes the planning loop can still
    reject; none is terminal without an acceptance record for the final plan.
    """
    subject = "plan-subject-1"
    publication = (
        comment(1, plan_coder(1, subject=subject)),
        comment(2, plan_review(CODEX, 1, "approved", subject=subject, phase="publication")),
        comment(3, plan_review(CLAUDE, 1, "approved", subject=subject, phase="publication")),
    )
    reconciled = (*publication, comment(4, record(
        "plan", "summary", "Orchestrator", 1, subject=subject, phase="reconciliation",
    )))
    primary_only = (
        comment(1, _primary_scheduler(1, subject)),
        comment(2, plan_coder(1, subject=subject)),
        comment(3, plan_review(CODEX, 1, "approved", subject=subject)),
    )
    secondary_without_opening = (*primary_only, comment(4, plan_review(CLAUDE, 2, "approved", subject=subject)))
    outstanding_obligation = (
        comment(1, plan_coder(1, subject=subject)),
        comment(2, plan_review(CODEX, 1, "approved", subject=subject)),
        comment(3, plan_review(CLAUDE, 1, "approved", subject=subject)),
        comment(4, plan_coder(2, subject=subject, plan="Plan text, revised for the growth gate.")),
    )
    for case in (publication, reconciled, primary_only, secondary_without_opening, outstanding_obligation):
        row = derive(history(), issue_view(*case))
        assert "plan-approved" not in row["terminal_states"]
        assert row["plan_rounds"] == {"value": None, "evidence": "unavailable"}
        assert row["human_decision_stops"] == {"value": None, "evidence": "unavailable"}

    stale = derive(history(), issue_view(*reconciled, handoff(9, plan="An older plan.")))
    assert "plan-approved" not in stale["terminal_states"]

    stale_announcement = derive(history(), issue_view(*reconciled, approval_announcement(9, plan="An older plan.")))
    assert "plan-approved" not in stale_announcement["terminal_states"]
    assert stale_announcement["plan_rounds"] == {"value": None, "evidence": "unavailable"}

    # Each orchestrator acceptance record for the final candidate settles the
    # plan: the plan-only approval announcement and the implementation handoff.
    for acceptance in (approval_announcement(9), handoff(9)):
        accepted = derive(history(), issue_view(*reconciled, acceptance))
        assert "plan-approved" in accepted["terminal_states"]
        assert accepted["plan_rounds"] == {"value": 1, "evidence": "verified"}
        assert accepted["final_plan_size"] == {"value": len("Plan text."), "evidence": "verified"}
        assert accepted["human_decision_stops"] == {"value": 0, "evidence": "verified"}
        assert accepted["outcome_complete"] is False


def _board(*seats: str) -> dict:
    return {
        "policy": "all-reviewers", "primary": None, "broad_rules": [],
        "seats": [{"id": seat, "backend": seat.lower(), "model_chain": None, "effort": None} for seat in seats],
    }


def test_expected_seats_come_from_the_recorded_board():
    board = _board(CODEX, CLAUDE)

    def pr(role, agent, round_number, state=None):
        return record("pr", role, agent, round_number, state=state, pr_board_root=board, pr_board_identity=board)

    never_posted = derive(
        history(state="CLOSED", closed=at(50, 12), linked=[merged_pr()]),
        issue_view(),
        {PR: pr_view(
            pr_comment(20, pr("coder", "Claude", 1)),
            pr_comment(21, pr("reviewer", CODEX, 1, "approved")),
        )},
    )
    assert never_posted["missing_reviewer_records"] == [
        {"surface": f"pr#{PR}", "flow": "pr", "round": 1, "reviewer": CLAUDE, "cause": "unknown"},
    ]
    assert never_posted["infrastructure_coverage"] == "partial"

    failed = derive(
        history(state="CLOSED", closed=at(50, 12), linked=[merged_pr()]),
        issue_view(),
        {PR: pr_view(
            pr_comment(20, pr("coder", "Claude", 1)),
            pr_comment(21, pr("reviewer", CODEX, 1, "approved")),
            pr_comment(22, _unavailable("Claude authentication expired.", "permissions")),
        )},
    )
    assert failed["missing_reviewer_records"][0]["cause"] == "auth-expiry"
    assert failed["infrastructure_rounds"] == [{"surface": f"pr#{PR}", "flow": "pr", "round": 1}]
    assert failed["pr_rounds"] == {"value": 0, "evidence": "verified"}
    assert failed["infrastructure_coverage"] == "complete"


def _incomplete_review(round_number: int, failures: dict) -> str:
    return _format_incomplete_pr_review_comment(
        pr_number=PR,
        unavailable_reviewers={
            agent: AgentInvocationError("down", failure_category=category)
            for agent, category in failures.items()
        },
        approved_reviewer_names=[CODEX],
        detection_round=round_number,
    )


def test_incomplete_pr_review_stop_is_recognized_and_explains_seats():
    body = _incomplete_review(2, {"claude": "non-retryable", "gemini": "transient"})
    diagnostic = recognize_diagnostic(body)
    assert diagnostic.kind == KIND_INFRASTRUCTURE and diagnostic.terminal
    assert diagnostic.round_number == 2
    assert diagnostic.seat_failures == (("Claude", "auth-or-billing"), ("Gemini", "backend-outage"))

    row = derive(
        history(state="CLOSED", closed=at(50, 12), linked=[merged_pr()]),
        issue_view(),
        {PR: pr_view(
            pr_comment(20, pr_record("coder", "Claude", 1)),
            pr_comment(21, pr_record("reviewer", CODEX, 1, "blocking")),
            pr_comment(22, pr_record("reviewer", CLAUDE, 1, "blocking")),
            pr_comment(23, pr_record("coder", "Claude", 2)),
            pr_comment(24, pr_record("reviewer", CODEX, 2, "approved")),
            pr_comment(25, _incomplete_review(2, {"claude": "non-retryable"})),
            pr_comment(26, pr_record("coder", "Claude", 3)),
            pr_comment(27, pr_record("reviewer", CODEX, 3, "approved")),
            pr_comment(28, pr_record("reviewer", CLAUDE, 3, "approved")),
        )},
    )
    assert {"surface": f"pr#{PR}", "flow": "pr", "round": 2} in row["infrastructure_rounds"]
    assert row["pr_rounds"] == {"value": 2, "evidence": "verified"}
    failure = next(entry for entry in row["infrastructure_failures"] if entry["source"] == "incomplete-review")
    assert (failure["seat"], failure["category"], failure["terminal"]) == (CLAUDE, "auth-or-billing", True)
    assert {"surface": f"pr#{PR}", "flow": "pr", "round": 2, "reviewer": CLAUDE,
            "cause": "auth-or-billing"} in row["missing_reviewer_records"]
    assert row["infrastructure_coverage"] == "complete"


def test_round_ends_in_infrastructure_unless_the_failed_seat_recovers():
    base = (
        comment(1, plan_coder(1)),
        comment(2, plan_review(CODEX, 1, "blocking")),
        comment(3, plan_review(CLAUDE, 1, "blocking")),
        comment(4, plan_coder(2)),
        comment(5, _unavailable("Claude authentication expired.", "permissions")),
        comment(6, plan_review(CODEX, 2, "blocking")),
    )
    later_peer = derive(history(state="CLOSED", closed=at(30, 11)), issue_view(*base))
    assert later_peer["infrastructure_rounds"] == [{"surface": f"issue#{ISSUE}", "flow": "plan", "round": 2}]
    assert later_peer["plan_rounds"] == {"value": 1, "evidence": "verified"}

    recovered = derive(
        history(state="CLOSED", closed=at(30, 11)),
        issue_view(*base, comment(7, plan_review(CLAUDE, 2, "blocking"))),
    )
    assert recovered["infrastructure_rounds"] == []
    assert recovered["plan_rounds"] == {"value": 2, "evidence": "verified"}


def test_terminal_infrastructure_diagnostic_bounds_an_open_issue():
    row = derive(
        history(),
        issue_view(
            comment(1, plan_coder(1)),
            comment(2, plan_review(CODEX, 1, "blocking")),
            comment(3, _unavailable("Completion recovery exhausted its bounded pass.", "tooling")),
        ),
    )
    assert "recognized-stop" in row["terminal_states"]
    assert row["outcome_complete"] is True
    assert row["human_decision_stops"] == {"value": 0, "evidence": "verified"}
    assert row["plan_rounds"] == {"value": 1, "evidence": "verified"}

    stall = CiInfrastructureStall(checks=(StalledCheck(name="test", kind="check_run", reason="queued_too_long", check_id=7, run_id="9", url=None, age_seconds=9000.0),))
    advisory = recognize_diagnostic(_format_ci_infrastructure_comment(PR, stall))
    stop = recognize_diagnostic(_ci_infrastructure_stop_message(PR, stall, []))
    assert (advisory.terminal, stop.terminal) == (False, True)


def test_unexplained_seat_in_the_latest_unbounded_round_is_pending():
    row = derive(
        history(),
        issue_view(
            comment(1, plan_coder(1)),
            comment(2, plan_review(CODEX, 1, "blocking")),
            comment(3, plan_review(CLAUDE, 1, "blocking")),
            comment(4, plan_coder(2)),
            comment(5, plan_review(CODEX, 2, "blocking")),
        ),
    )
    assert row["missing_reviewer_records"] == [
        {"surface": f"issue#{ISSUE}", "flow": "plan", "round": 2, "reviewer": CLAUDE, "cause": "pending"},
    ]
    assert row["infrastructure_coverage"] == "partial"


def _debater(comment_id: int, agent: str, round_number: int):
    return comment(comment_id, record("discuss", "debater", agent, round_number))


def _discuss_summary(comment_id: int, round_number: int, *, failed=(), final=False):
    return comment(comment_id, record(
        "discuss", "summary", "Orchestrator", round_number, failed_debaters=failed, is_final=final,
    ))


def test_discussion_failed_debaters_are_infrastructure_failures():
    explained = derive(
        history(),
        issue_view(
            _debater(1, CODEX, 1), _debater(2, CLAUDE, 1), _discuss_summary(3, 1),
            _debater(4, CODEX, 2), _discuss_summary(5, 2, failed=((CLAUDE, "transient"),), final=True),
        ),
    )
    assert explained["route_taken"] == ["discuss"]
    failure = explained["infrastructure_failures"][0]
    assert (failure["seat"], failure["category"], failure["source"]) == (
        CLAUDE, "backend-outage", "discuss-failed-debater",
    )
    assert explained["missing_reviewer_records"] == [
        {"surface": f"issue#{ISSUE}", "flow": "discuss", "round": 2, "reviewer": CLAUDE,
         "cause": "backend-outage"},
    ]
    assert explained["infrastructure_coverage"] == "complete"

    unexplained = derive(
        history(),
        issue_view(
            _debater(1, CODEX, 1), _debater(2, CLAUDE, 1), _discuss_summary(3, 1),
            _debater(4, CODEX, 2), _discuss_summary(5, 2, final=True),
        ),
    )
    assert unexplained["infrastructure_failures"] == []
    assert unexplained["missing_reviewer_records"][0]["cause"] == "unknown"
    assert unexplained["infrastructure_coverage"] == "partial"


def test_snapshot_uses_only_the_first_planner_record_digest():
    v1, v2 = "Body v1", "Body v2"
    row = derive(
        history(body=v2, body_revisions=((at(0, 9), v1), (at(30, 10), v2))),
        issue_view(
            comment(1, plan_coder(1)),
            comment(2, plan_review(CODEX, 1, "blocking")),
            comment(45, plan_coder(2, plan_issue_digest=plan_issue_text_digest_for("Issue title", v2))),
        ),
    )
    snapshot = row["snapshot"]
    assert snapshot["plan_issue_digest"] is None
    assert snapshot["status"] == "time-inferred"
    assert snapshot["body"] == v1


def test_mixed_era_step_backs_stay_unavailable():
    # The legacy round is after the step-back cutoff, so it is an era round
    # missing its record rather than a pre-feature round.
    legacy = (
        comment(1, record("plan", "coder", "Claude", 1, canonical_plan="Old plan."), created=after_cutoff(1)),
        comment(2, record("plan", "reviewer", CODEX, 1, state="blocking"), created=after_cutoff(2)),
    )
    mixed = derive(
        history(state="CLOSED", closed=after_cutoff(30, 11)),
        issue_view(
            *legacy,
            comment(3, plan_coder(2), created=after_cutoff(3)),
            comment(4, plan_review(CODEX, 2, "approved"), created=after_cutoff(4)),
        ),
    )
    assert mixed["step_backs"] == {"value": None, "evidence": "unavailable"}
    modern = derive(
        history(state="CLOSED", closed=at(30, 11)),
        issue_view(
            comment(1, plan_coder(1)), comment(2, plan_review(CODEX, 1, "blocking")),
            comment(3, plan_coder(2)), comment(4, plan_review(CODEX, 2, "approved")),
        ),
    )
    assert modern["step_backs"] == {"value": 0, "evidence": "verified"}


# --- step-backs before the step-back feature (#1442) ---------------------------


PRE_FEATURE = {"value": 0, "evidence": "not-applicable-pre-feature"}
UNAVAILABLE = {"value": None, "evidence": "unavailable"}


def _legacy_coder(round_number: int, **kwargs) -> str:
    return record("plan", "coder", "Claude", round_number, canonical_plan="Old plan.", **kwargs)


def _legacy_review(round_number: int, state: str) -> str:
    return record("plan", "reviewer", CODEX, round_number, state=state)


def _legacy_pr(role: str, round_number: int, state: str | None = None) -> str:
    return record("pr", role, CODEX if role == "reviewer" else "Claude", round_number, state=state)


def test_pre_feature_only_flow_exports_a_provable_zero():
    row = derive(
        history(state="CLOSED", closed=at(30, 11)),
        issue_view(
            comment(1, _legacy_coder(1)), comment(2, _legacy_review(1, "blocking")),
            comment(3, _legacy_coder(2)), comment(4, _legacy_review(2, "approved")),
        ),
    )
    assert row["step_backs"] == PRE_FEATURE
    assert row["plan_rounds"] == {"value": 2, "evidence": "verified"}
    assert row["human_decision_stops"] == {"value": 0, "evidence": "verified"}
    assert row["final_plan_size"] == {"value": len("Old plan."), "evidence": "verified"}


def test_post_cutoff_round_without_era_evidence_stays_unavailable():
    row = derive(
        history(state="CLOSED", closed=after_cutoff(30, 11)),
        issue_view(
            comment(1, _legacy_coder(1)), comment(2, _legacy_review(1, "blocking")),
            comment(3, _legacy_coder(2), created=after_cutoff(3)),
            comment(4, _legacy_review(2, "approved"), created=after_cutoff(4)),
        ),
    )
    assert row["step_backs"] == UNAVAILABLE


def test_mixed_flow_counts_era_step_backs_and_pre_era_rounds_add_zero():
    def mixed(entries):
        return derive(
            history(state="CLOSED", closed=after_cutoff(30, 11)),
            issue_view(
                comment(1, _legacy_coder(1)), comment(2, _legacy_review(1, "blocking")),
                comment(3, plan_coder(2), created=after_cutoff(3)),
                comment(4, plan_review(CODEX, 2, "blocking", **entries), created=after_cutoff(4)),
                comment(5, plan_coder(3), created=after_cutoff(5)),
                comment(6, plan_review(CODEX, 3, "approved"), created=after_cutoff(6)),
            ),
        )

    stepped = {"phase": "plan", "reviewer": CODEX, "trigger_round": 2}
    assert mixed({"step_back_entries": (stepped,)})["step_backs"] == {"value": 1, "evidence": "verified"}
    assert mixed({})["step_backs"] == {"value": 0, "evidence": "verified"}


def test_round_straddling_the_cutoff_without_era_evidence_stays_unavailable():
    row = derive(
        history(state="CLOSED", closed=after_cutoff(30, 11)),
        issue_view(
            comment(1, _legacy_coder(1)),
            comment(2, _legacy_review(1, "approved"), created=after_cutoff(2)),
        ),
    )
    assert row["step_backs"] == UNAVAILABLE


def test_pre_feature_plan_ledger_with_a_linked_pr_ledger():
    plan = (comment(1, _legacy_coder(1)), comment(2, _legacy_review(1, "approved")))
    closed = history(state="CLOSED", closed=after_cutoff(50, 12), linked=[merged_pr(when=after_cutoff(50, 12))])
    era_pr = derive(closed, issue_view(*plan), {PR: pr_view(
        pr_comment(20, pr_record("coder", "Claude", 1), created=after_cutoff(20)),
        pr_comment(21, pr_record("reviewer", CODEX, 1, "approved"), created=after_cutoff(21)),
    )})
    assert era_pr["step_backs"] == {"value": 0, "evidence": "verified"}
    uncovered_pr = derive(closed, issue_view(*plan), {PR: pr_view(
        pr_comment(20, _legacy_pr("coder", 1), created=after_cutoff(20)),
        pr_comment(21, _legacy_pr("reviewer", 1, "approved"), created=after_cutoff(21)),
    )})
    assert uncovered_pr["step_backs"] == UNAVAILABLE
    pre_feature_pr = derive(
        history(state="CLOSED", closed=at(50, 12), linked=[merged_pr()]),
        issue_view(*plan),
        {PR: pr_view(
            pr_comment(20, _legacy_pr("coder", 1)), pr_comment(21, _legacy_pr("reviewer", 1, "approved")),
        )},
    )
    assert pre_feature_pr["step_backs"] == PRE_FEATURE


def test_unbounded_pre_feature_history_stays_unavailable():
    row = derive(
        history(),
        issue_view(comment(1, _legacy_coder(1)), comment(2, _legacy_review(1, "blocking"))),
    )
    assert row["step_backs"] == UNAVAILABLE


def test_pre_feature_proof_requires_a_known_edit_time_before_the_cutoff():
    def derive_review(review: AuthenticatedComment):
        return derive(
            history(state="CLOSED", closed=after_cutoff(30, 11)),
            issue_view(comment(1, _legacy_coder(1)), review),
        )

    edited_after = comment(2, _legacy_review(1, "approved"), updated=after_cutoff(2))
    assert derive_review(edited_after)["step_backs"] == UNAVAILABLE
    # The current body was replaced by a different round's record after the cutoff.
    replaced = derive(
        history(state="CLOSED", closed=after_cutoff(30, 11)),
        issue_view(
            comment(1, _legacy_coder(1)), comment(2, _legacy_review(1, "approved")),
            comment(3, _legacy_coder(2), updated=after_cutoff(3)),
        ),
    )
    assert replaced["step_backs"] == UNAVAILABLE
    edited_before = comment(2, _legacy_review(1, "approved"), updated="2026-10-02T10:00:00Z")
    assert derive_review(edited_before)["step_backs"] == PRE_FEATURE
    unknown_edit = dataclasses.replace(comment(2, _legacy_review(1, "approved")), updated_at=None)
    assert derive_review(unknown_edit)["step_backs"] == UNAVAILABLE
    era_edit = comment(2, plan_review(CODEX, 1, "approved"), updated=after_cutoff(2))
    assert derive_review(era_edit)["step_backs"] == {"value": 0, "evidence": "verified"}


# --- Plan finding provenance and sweep columns (#1505) ------------------------

from coding_review_agent_loop.protocol import UnresolvedReviewItem  # noqa: E402

_PROVENANCE_COLUMNS = (
    "plan_findings_pre_existing", "plan_findings_revision_introduced",
    "plan_findings_untagged", "plan_sweep_reviews",
)


def _finding(n: int, *, provenance=None, repeat_of=None, status="blocking", reviewer=CODEX):
    return UnresolvedReviewItem(
        item_id=f"item-{n}", reviewer=reviewer, source_round=1, text=f"Gap {n}.",
        status=status, provenance=provenance, repeat_of=repeat_of,
    )


def _provenance_comments(*, round_two_phase="authoritative"):
    round_two_items = (
        _finding(3, provenance="pre_existing"),
        _finding(4, provenance="revision_introduced"),
        _finding(5),
        _finding(6, provenance="pre_existing", repeat_of="item-1"),
        _finding(7, provenance="pre_existing", status="future"),
    )
    return (
        comment(1, plan_coder(1, plan="Plan v1.")),
        # The first review's findings never count, whatever their tags.
        comment(2, plan_review(CODEX, 1, "blocking", new_items=(
            _finding(1, provenance="pre_existing"), _finding(2),
        ))),
        comment(3, plan_coder(2, plan="Plan v2.")),
        comment(4, plan_review(
            CODEX, 2, "blocking", plan_sweep=True, phase=round_two_phase,
            new_items=() if round_two_phase == "publication" else round_two_items,
        )),
        *approved_plan_comments(start=5, round_number=3),
    )


def test_history_export_reports_provenance_buckets_and_sweep_reviews():
    """`export-counts`: 1/1/1 with the repeat, future and first-review items excluded."""
    row = derive(
        history(state="CLOSED", closed=at(30, 11)),
        issue_view(
            *_provenance_comments(),
            # A foreign copy of a swept, tagged review never counts.
            comment(20, plan_review(
                CODEX, 3, "blocking", plan_sweep=True,
                new_items=(_finding(9, provenance="pre_existing"),),
            ), author=FOREIGN),
        ),
    )
    assert row["plan_findings_pre_existing"] == {"value": 1, "evidence": "verified"}
    assert row["plan_findings_revision_introduced"] == {"value": 1, "evidence": "verified"}
    assert row["plan_findings_untagged"] == {"value": 1, "evidence": "verified"}
    assert row["plan_sweep_reviews"] == {"value": 1, "evidence": "verified"}
    assert row["schema_version"] == 1
    header = render_csv([row]).splitlines()[0].split(",")
    for column in _PROVENANCE_COLUMNS:
        assert column in header and f"{column}_evidence" in header


def test_history_export_provenance_uses_the_shared_filter_without_the_trigger_rule():
    """`export-counts`: a revision-introduced-only review breaks the trigger, yet exports."""
    from coding_review_agent_loop.review_step_back import (
        CLASS_REPEAT_ONLY,
        classify_sweep_review,
    )

    items = (_finding(4, provenance="revision_introduced"),)
    metadata = PostedRoundMetadata(
        flow="plan", role="reviewer", agent=CODEX, round_number=2, subject="s",
        state="blocking", new_items=items,
    )
    assert classify_sweep_review(metadata, items, first_review=False) == CLASS_REPEAT_ONLY
    row = derive(
        history(state="CLOSED", closed=at(30, 11)),
        issue_view(
            comment(1, plan_coder(1, plan="Plan v1.")),
            comment(2, plan_review(CODEX, 1, "blocking", new_items=(_finding(1),))),
            comment(3, plan_coder(2, plan="Plan v2.")),
            comment(4, plan_review(CODEX, 2, "blocking", new_items=items)),
            *approved_plan_comments(start=5, round_number=3),
        ),
    )
    assert row["plan_findings_revision_introduced"] == {"value": 1, "evidence": "verified"}
    assert row["plan_findings_pre_existing"]["value"] == 0
    assert row["plan_findings_untagged"]["value"] == 0
    assert row["plan_sweep_reviews"]["value"] == 0


def test_history_export_provenance_is_unavailable_for_unbounded_or_unresolvable_history():
    unbounded = derive(history(), issue_view(*_provenance_comments()[:4]))
    for column in _PROVENANCE_COLUMNS:
        assert unbounded[column] == {"value": None, "evidence": "unavailable"}
    # A provisional publication record with no later reconciliation is unknowable.
    unresolvable = derive(
        history(state="CLOSED", closed=at(30, 11)),
        issue_view(*_provenance_comments(round_two_phase="publication")),
    )
    for column in _PROVENANCE_COLUMNS[:3]:
        assert unresolvable[column] == {"value": None, "evidence": "unavailable"}
    assert unresolvable["plan_sweep_reviews"] == {"value": 1, "evidence": "verified"}


# --- Plan provenance baseline form columns (#1516) ----------------------------

_BASELINE_COLUMNS = (
    "plan_baselines_diff", "plan_baselines_compact_diff", "plan_baselines_full",
    "plan_baselines_unavailable",
)


def _baseline(form: str, chars: int = 10) -> dict:
    return {"provenance_baseline_form": form, "provenance_baseline_chars": chars}


def _baseline_comments():
    return (
        comment(1, plan_coder(1, plan="Plan v1.")),
        # First reviews never count, whatever form they record.
        comment(2, plan_review(CODEX, 1, "blocking", **_baseline("first-review", 0))),
        comment(3, plan_review(CLAUDE, 1, "blocking", **_baseline("unavailable", 0))),
        comment(4, plan_coder(2, plan="Plan v2.")),
        # Superseded by the later record of the same reviewer and round.
        comment(5, plan_review(CODEX, 2, "blocking", **_baseline("unavailable", 0))),
        comment(6, plan_review(CODEX, 2, "blocking", **_baseline("compact-diff", 900))),
        # A pre-feature record without the field counts in no form.
        comment(7, plan_review(CLAUDE, 2, "blocking")),
        comment(8, plan_coder(3, plan="Plan v3.")),
        comment(9, plan_review(CODEX, 3, "blocking", **_baseline("full-baseline", 5000))),
        comment(10, plan_review(CLAUDE, 3, "blocking", **_baseline("unavailable", 0))),
        comment(11, plan_coder(4, plan="Plan v4.")),
        comment(12, plan_review(CODEX, 4, "approved", **_baseline("diff", 300))),
        comment(13, plan_review(CLAUDE, 4, "approved", **_baseline("identical", 0))),
        handoff(14),
    )


def test_history_export_counts_baseline_forms_of_non_first_reviews():
    """`export-form-counts`: latest non-first records only, plan_evidence rule."""
    row = derive(
        history(state="CLOSED", closed=at(30, 11)),
        issue_view(
            *_baseline_comments(),
            comment(20, plan_review(CODEX, 5, "blocking", **_baseline("diff")), author=FOREIGN),
        ),
    )
    assert row["plan_baselines_diff"] == {"value": 1, "evidence": "verified"}
    assert row["plan_baselines_compact_diff"] == {"value": 1, "evidence": "verified"}
    assert row["plan_baselines_full"] == {"value": 1, "evidence": "verified"}
    assert row["plan_baselines_unavailable"] == {"value": 1, "evidence": "verified"}
    assert row["schema_version"] == 1
    header = render_csv([row]).splitlines()[0].split(",")
    for column in _BASELINE_COLUMNS:
        assert column in header and f"{column}_evidence" in header
    assert header.index("plan_baselines_diff") > header.index("plan_sweep_reviews")


def test_history_export_baseline_forms_follow_the_plan_evidence_rule():
    unbounded = derive(history(), issue_view(*_baseline_comments()[:7]))
    for column in _BASELINE_COLUMNS:
        assert unbounded[column] == {"value": None, "evidence": "unavailable"}
    legacy = derive(
        history(state="CLOSED", closed=at(30, 11)),
        issue_view(*_provenance_comments()),
    )
    for column in _BASELINE_COLUMNS:
        assert legacy[column] == {"value": 0, "evidence": "verified"}


def _approach_alternative_stop_message() -> str:
    from coding_review_agent_loop import review_step_back as sb

    records = (
        PostedRoundRecord(
            index=1,
            metadata=PostedRoundMetadata(
                flow="pr", role="coder", agent="Claude", round_number=4, subject="b" * 40,
                step_back_entries=({
                    "phase": "pr", "reviewer": CODEX, "trigger_round": 3,
                    "trigger_head": "a" * 40,
                    "anchor": {"path": "src/spool.py", "start": 124, "end": 140},
                },),
                raw_structured_coder_response=json.dumps(
                    {"summary": "Approach alternative: drop the predecessor check."}
                ),
            ),
            body="",
        ),
    )
    alternative = sb.pr_step_back_approach_alternative(records, head_sha="b" * 40)
    return sb.render_pr_step_back_approach_alternative_decision(PR, alternative)


def test_pr_approach_alternative_stop_is_a_human_decision_stop():
    """#1628: the rendered message, including its ``PR #N: `` prefix, is recognized."""
    message = _approach_alternative_stop_message()
    assert message.startswith(f"PR #{PR}: ")
    stop = recognize_diagnostic(render_pr_scheduling_diagnostic("round 5", message))
    assert (stop.kind, stop.reason, stop.round_number) == (
        KIND_HUMAN_DECISION_STOP, "pr-step-back-approach-alternative", 5,
    )
    # Without the PR prefix it is recognized too; other PR stops stay generic.
    bare = message.split(": ", 1)[1]
    assert recognize_diagnostic(
        render_pr_scheduling_diagnostic("round 5", bare)
    ).reason == "pr-step-back-approach-alternative"
    other = recognize_diagnostic(render_pr_scheduling_diagnostic("round 5", "PR #9: panel unsafe"))
    assert (other.kind, other.reason) == (KIND_PR_STOP, "scheduling-stop")
