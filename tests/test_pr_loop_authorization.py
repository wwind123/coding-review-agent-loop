"""PR-loop tests for pr_loop_support and issue_pr_handoff: PR authorization (#1542)."""

import base64
import json
import re
from types import SimpleNamespace

import pytest
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop, run_pr_loop
from coding_review_agent_loop.comment_rendering import (
    render_risk_test_matrix_section,
)
from coding_review_agent_loop.followups import (
    PlanApprovedFollowupSource,
    _followup_issue_body,
    _format_approved_followup_summary,
    _format_plan_approval_summary_with_followups,
    _plan_followup_issue_body,
    reconcile_approved_followups,
    reconcile_plan_approved_followups,
)
from coding_review_agent_loop.github import (
    IssueComment,
    IssueContext,
)
from coding_review_agent_loop.issue_pr_handoff import (
    format_issue_pr_handoff_comment,
)
from coding_review_agent_loop.memory import AgentMemoryContext
import coding_review_agent_loop.prompts as prompts_module
from coding_review_agent_loop.managed_ci import (
    ManagedCiContract,
)
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    _attach_round_metadata,
)
from coding_review_agent_loop.pr_loop_support import (
    _preserve_issue_created_managed_suppression,
)
from coding_review_agent_loop.prompts import (
    build_followup_prompt,
)
import coding_review_agent_loop.test_runtime as runtime
from coding_review_agent_loop.protocol import (
    ApprovedFollowup,
    parse_risk_test_matrix,
    risk_test_matrix_identity,
)
from agent_loop_helpers import (
    FakeRunner,
    structured_plan_state,
    make_config,
    prior_item_dispositions,
    structured_coder_followup,
    structured_plan_review,
    structured_pr_review,
)
import coding_review_agent_loop.child_plan_binding as child_plan_binding
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.execution_policy as execution_policy
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.panel_evidence as panel_evidence
import coding_review_agent_loop.plan_first_loop as plan_first_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
from agent_loop_helpers import patch_attr_in


@pytest.mark.parametrize("origin", ["issue-created", "source-managed"])
def test_interrupted_tool_created_managed_pr_retains_suppression(origin):
    contract = ManagedCiContract(origin=origin, issue_created_pr=origin == "issue-created")

    assert _preserve_issue_created_managed_suppression(
        contract,
        active_exception=AgentLoopError("malformed coder handoff"),
    )
    assert not _preserve_issue_created_managed_suppression(
        contract,
        active_exception=None,
    )


def test_interrupted_existing_pr_adoption_does_not_claim_durable_suppression():
    contract = ManagedCiContract(adopted_existing_pr=True)

    assert not _preserve_issue_created_managed_suppression(
        contract,
        active_exception=AgentLoopError("review failed"),
    )


def _approved_issue_plan_comments(plan: str) -> list[dict[str, object]]:
    subject = panel_evidence._plan_subject(plan)
    return [
        {
            "author": {"login": "coding-review-agent-loop"},
            "createdAt": "2026-05-01T00:00:00Z",
            "body": _attach_round_metadata(
                plan,
                PostedRoundMetadata(
                    flow="plan", role="coder", agent="Claude", round_number=1,
                    subject=subject, canonical_plan=plan,
                    # Versioned (empty) dispositions back the keyless approval (#1377).
                    raw_structured_coder_response=structured_plan_state(),
                ),
            ),
        },
        {
            "author": {"login": "coding-review-agent-loop"},
            "createdAt": "2026-05-01T00:01:00Z",
            "body": _attach_round_metadata(
                "Approved.",
                PostedRoundMetadata(
                    flow="plan", role="reviewer", agent="Codex",
                    round_number=1, subject=subject, state="approved",
                ),
            ),
        },
    ]


class _FreshScopeCaptured(Exception):
    pass


def test_pr_fresh_authorization_binds_server_recovered_approved_plan(tmp_path, monkeypatch):
    plan = "Approved plan.\n\n### Plan steps\n1. Preserve the trust boundary."
    runner = FakeRunner(
        issue_comments=_approved_issue_plan_comments(plan),
        pr_payload={
            "headRefName": "agent-loop/managed-56", "headRefOid": "abc123",
            "baseRefName": "main", "body": "Fixes #56",
        },
    )
    captured = {}

    def authorize(*args, **kwargs):
        captured.update(kwargs)
        raise _FreshScopeCaptured

    monkeypatch.setattr(pr_loop, "authorize_fresh_issue_created_resume", authorize)
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_fresh_authorization=True, managed_ci_issue_number=56,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        reviewer=("codex",),
    )

    with pytest.raises(_FreshScopeCaptured):
        run_pr_loop(runner, pr_number=77, config=config)

    assert captured["issue_number"] == 56
    assert captured["approved_plan_hash"] == panel_evidence.approved_plan_hash(plan)
    assert not any(command[:1] in (["claude"], ["codex"]) for command, _cwd in runner.commands)


def test_pr_ordinary_resume_binds_server_recovered_approved_plan_before_activation(
    tmp_path, monkeypatch,
):
    plan = "Approved plan.\n\n### Plan steps\n1. Preserve the trust boundary."
    runner = FakeRunner(
        issue_comments=_approved_issue_plan_comments(plan),
        pr_payload={
            "headRefName": "agent-loop/managed-56", "headRefOid": "abc123",
            "baseRefName": "main", "body": "Fixes #56",
        },
    )
    handoff = pr_loop_support.AuthenticatedIssueCreatedHandoff(
        pr_number=77, issue_number=56, repository="OWNER/REPO", base_ref="main",
        head_sha="abc123", branch="agent-loop/managed-56",
        trusted_actor_login="agent-loop", trusted_actor_id=1,
        protection_mode="voluntary", override_nonce="opening-nonce",
    )
    captured = {}
    monkeypatch.setattr(
        pr_loop, "recover_issue_created_handoff", lambda *_a, **_k: handoff
    )

    def revalidate(*_args, **kwargs):
        captured.update(kwargs)
        raise _FreshScopeCaptured

    monkeypatch.setattr(pr_loop, "revalidate_issue_created_handoff", revalidate)
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        reviewer=("codex",),
    )

    with pytest.raises(_FreshScopeCaptured):
        run_pr_loop(runner, pr_number=77, config=config)

    assert captured["handoff"].approved_plan_hash == panel_evidence.approved_plan_hash(plan)
    assert not any(command[:1] in (["claude"], ["codex"]) for command, _cwd in runner.commands)


def test_pr_fresh_authorization_rejects_mismatched_supplied_plan_scope(tmp_path, monkeypatch):
    canonical = "Approved plan.\n\n### Plan steps\n1. Preserve the trust boundary."
    supplied = execution_policy.make_approved_plan_context(
        "Different plan.\n\n### Plan steps\n1. Change the boundary.", source_locator="test"
    )
    runner = FakeRunner(
        issue_comments=_approved_issue_plan_comments(canonical),
        pr_payload={"headRefName": "agent-loop/managed-56", "body": "Fixes #56"},
    )
    monkeypatch.setattr(
        pr_loop, "authorize_fresh_issue_created_resume",
        lambda *args, **kwargs: (_ for _ in ()).throw(_FreshScopeCaptured()),
    )
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_fresh_authorization=True, managed_ci_issue_number=56,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        reviewer=("codex",),
    )

    with pytest.raises(AgentLoopError, match="does not match the canonical issue plan"):
        run_pr_loop(
            runner, pr_number=77, config=config, approved_plan_context=supplied
        )


def test_pr_fresh_authorization_rejects_incomplete_plan_scope(tmp_path, monkeypatch):
    plan = "Unapproved plan."
    subject = panel_evidence._plan_subject(plan)
    runner = FakeRunner(
        issue_comments=[{
            "author": {"login": "coding-review-agent-loop"},
            "createdAt": "2026-05-01T00:00:00Z",
            "body": _attach_round_metadata(
                plan,
                PostedRoundMetadata(
                    flow="plan", role="reviewer", agent="OpenAI Codex",
                    round_number=1, subject=subject, state="blocking",
                ),
            ),
        }],
        pr_payload={"headRefName": "agent-loop/managed-56", "body": "Fixes #56"},
    )
    monkeypatch.setattr(
        pr_loop, "authorize_fresh_issue_created_resume",
        lambda *args, **kwargs: (_ for _ in ()).throw(_FreshScopeCaptured()),
    )
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_fresh_authorization=True, managed_ci_issue_number=56,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        reviewer=("codex",),
    )

    with pytest.raises(AgentLoopError, match="complete canonical approved plan"):
        run_pr_loop(runner, pr_number=77, config=config)


def test_pr_fresh_authorization_allows_authenticated_planless_issue_scope(tmp_path, monkeypatch):
    runner = FakeRunner(
        pr_payload={"headRefName": "agent-loop/managed-56", "body": "Fixes #56"},
    )
    captured = {}

    def authorize(*args, **kwargs):
        captured.update(kwargs)
        raise _FreshScopeCaptured

    monkeypatch.setattr(pr_loop, "authorize_fresh_issue_created_resume", authorize)
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_fresh_authorization=True, managed_ci_issue_number=56,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        reviewer=("codex",),
    )

    with pytest.raises(_FreshScopeCaptured):
        run_pr_loop(runner, pr_number=77, config=config)

    assert captured["approved_plan_hash"] is None


def test_managed_issue_fix_round_publishes_correlated_head_continuity(
    tmp_path, monkeypatch,
):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking", summary="A repair is required.",
                blocking_items=["Fix the managed recovery edge case."],
            ),
            structured_pr_review(
                state="approved", summary="The repair is complete.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        pr_payload={
            "headRefName": "agent-loop/managed-56", "headRefOid": "abc123",
            "baseRefName": "main", "body": "Fixes #56",
        },
    )
    handoff = pr_loop_support.AuthenticatedIssueCreatedHandoff(
        pr_number=77, issue_number=56, repository="OWNER/REPO", base_ref="main",
        head_sha="abc123", branch="agent-loop/managed-56",
        trusted_actor_login="agent-loop", trusted_actor_id=1,
        protection_mode="voluntary", override_nonce="root",
        authorization_kind="creation", authorization_comment_id=17,
    )
    selected = []
    published = []
    monkeypatch.setattr(pr_loop, "revalidate_issue_created_handoff", lambda *_a, **_k: handoff)
    monkeypatch.setattr(
        pr_loop, "activate_managed_ci",
        lambda *_a, **_k: ManagedCiContract(protocol_version=2, issue_created_pr=True),
    )
    monkeypatch.setattr(pr_loop, "revalidate_adopted_managed_ci", lambda *_a, **_k: True)
    monkeypatch.setattr(pr_loop, "managed_label_present", lambda *_a, **_k: True)
    monkeypatch.setattr(
        pr_loop, "find_actor_round_metadata_comment_ids",
        lambda *_a, **kwargs: selected.append(kwargs) or (88, 89),
    )
    def publish(*_args, **kwargs):
        published.append(kwargs)
        raise _FreshScopeCaptured

    monkeypatch.setattr(
        pr_loop, "publish_issue_created_continuity_authorization", publish
    )
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True, reviewer=("codex",), max_rounds=2,
    )

    with pytest.raises(_FreshScopeCaptured):
        run_pr_loop(
            runner, pr_number=77, config=config, managed_ci_handoff=handoff,
            managed_ci_issue_number=56,
        )

    assert len(selected) == len(published) == 1
    assert selected[0]["predecessor_head"] == "abc123"
    assert selected[0]["new_head"] == "abc123-coder-1"
    assert selected[0]["after_comment_id"] == 17
    assert published[0]["round_comment_ids"] == (88, 89)


def test_pr_resume_prompt_uses_current_wrapper_health_without_dropping_command(
    tmp_path, monkeypatch
):
    config = make_config(
        tmp_path,
        test_command=("pytest", "tests/test_protocol.py", "-q"),
    )
    memory = AgentMemoryContext(
        memory_dir=tmp_path / "memory",
        current_commit=None,
        last_analyzed_commit=None,
        changed_files=(),
        repo_summary=None,
        architecture_map=None,
        test_profile=None,
        toolchain=None,
    )
    monkeypatch.setattr(
        prompts_module,
        "preflight_wrapper_candidates",
        lambda **_kwargs: (runtime.LauncherProbeResult(("/opt/agent-loop", "run-tests"), "failed", "import failed"),),
    )
    prompt = build_followup_prompt(768, 2, "Repair the launcher path.", config, memory=memory)
    assert "pytest tests/test_protocol.py -q" in prompt
    assert "no wrapper candidate verified" in prompt


def _issue_view_commands(runner):
    return [cmd for cmd, _cwd in runner.commands if cmd[:3] == ["gh", "issue", "view"]]


def test_direct_pr_resolves_single_linked_issue_context_for_reviewer(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "LGTM.\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
        issue_payload={
            "number": 56,
            "title": "Original acceptance criteria",
            "body": "Keep compatibility.\n\n-- Human Reviewer",
        },
        issue_comments=[
            {"author": {"login": "maintainer"}, "createdAt": "2026-06-01T00:00:00Z", "body": "Later clarification."}
        ],
        pr_payload={"body": "Fixes #56"},
    )
    config = make_config(tmp_path, quiet=False)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(_issue_view_commands(runner)) == 1
    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "Original acceptance criteria" in prompt
    assert "Keep compatibility." in prompt
    assert "Later clarification." in prompt
    assert "Human requirements" in prompt


def test_pr_review_runs_when_pr_and_issue_text_name_a_reserved_token(tmp_path, capsys):
    """Issue #891: naming a record in PR or issue prose must not abort the run.

    Both surfaces carry the token here, and the run must review normally with
    the token defanged in the reviewer prompt.
    """
    token = "AGENT_PLAN_APPROVED_FOLLOWUPS"
    label = "[protocol PLAN_APPROVED_FOLLOWUPS record]"
    # A `well-formed-only` entry has no bare-name fallback in the scanner, so
    # cover one on the title surfaces too.
    strict_token = "AGENT_ISSUE_PR_HANDOFF"
    strict_label = "[protocol ISSUE_PR_HANDOFF record]"
    runner = FakeRunner(
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
        issue_payload={
            "number": 56,
            "title": f"Reserved {strict_token} label",
            "body": f"The planner writes {token} when it approves follow-ups.",
        },
        issue_comments=[
            {
                "author": {"login": "maintainer"},
                "createdAt": "2026-09-19T00:00:00Z",
                "body": f"The {token} name also appears in a comment.",
            }
        ],
        pr_payload={
            "title": f"Rename the {strict_token} label",
            "body": f"Fixes #56\n\nThis PR renames the {token} record label.",
        },
    )
    config = make_config(tmp_path, quiet=False)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert token not in prompt
    assert strict_token not in prompt
    assert label in prompt
    assert f"- Title: Rename the {strict_label} label" in prompt
    # Surrounding prose survives, so the reviewer still sees the request.
    assert "when it approves follow-ups." in prompt
    assert "name also appears in a comment." in prompt
    # The run says which surfaces named the tokens, titles included, and that
    # they grant no authority.
    logged = capsys.readouterr().err
    assert (
        f"Issue #56 text names reserved protocol marker(s) {strict_token}, {token}"
        in logged
    )
    assert (
        f"Pull request #77 text names reserved protocol marker(s) {strict_token}, {token}"
        in logged
    )
    assert logged.count("They carry no authority.") == 2


def test_plain_pr_recovery_accepts_loop_created_managed_pr_body(tmp_path):
    encoded = base64.urlsafe_b64encode(
        json.dumps(
            {"source_branch": "feature/review-context", "source_sha": "abc123"},
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).decode()
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={
            "body": f"<!-- AGENT_MANAGED_PR_SOURCE_V1 {encoded} -->",
            "headRefName": "agent-loop/managed-direct-token",
        },
    )

    assert run_pr_loop(runner, pr_number=77, config=make_config(tmp_path)) == 0
    assert runner.comments == ["**Review verdict:** Approved\n\nLGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex: unknown model (medium)"]


def test_plain_pr_recovery_accepts_managed_pr_head_after_creation_sha_advanced(tmp_path):
    encoded = base64.urlsafe_b64encode(
        json.dumps(
            {"source_branch": "feature/review-context", "source_sha": "abc123"},
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).decode()
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={
            "body": f"<!-- AGENT_MANAGED_PR_SOURCE_V1 {encoded} -->",
            "headRefName": "agent-loop/managed-direct-token",
            "headRefOid": "abc123-coder-1",
        },
    )

    assert run_pr_loop(runner, pr_number=77, config=make_config(tmp_path)) == 0
    assert runner.comments == ["**Review verdict:** Approved\n\nLGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex: unknown model (medium)"]


def test_in_process_managed_pr_handoff_still_requires_creation_sha(tmp_path):
    encoded = base64.urlsafe_b64encode(
        json.dumps(
            {"source_branch": "feature/review-context", "source_sha": "abc123"},
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).decode()
    runner = FakeRunner(
        pr_payload={
            "body": f"<!-- AGENT_MANAGED_PR_SOURCE_V1 {encoded} -->",
            "headRefName": "agent-loop/managed-direct-token",
            "headRefOid": "abc123-coder-1",
        },
    )

    with pytest.raises(AgentLoopError, match="Managed PR head SHA does not match"):
        run_pr_loop(
            runner,
            pr_number=77,
            config=make_config(tmp_path),
            managed_pr_origin=(
                "feature/review-context",
                "abc123",
                "agent-loop/managed-direct-token",
                None,
            ),
        )


def test_approved_followup_public_renderings_sanitize_historical_marker_mentions():
    followup = ApprovedFollowup(
        reviewer="Claude",
        text="Document AGENT_APPROVED_FOLLOWUPS handling.",
    )
    reconciliation = reconcile_approved_followups([followup])

    summary = _format_approved_followup_summary(77, reconciliation)
    issue_body = _followup_issue_body(77, reconciliation.selected_groups[0])

    assert "AGENT_APPROVED_FOLLOWUPS" not in summary
    assert "AGENT_APPROVED_FOLLOWUPS" not in issue_body

    source = PlanApprovedFollowupSource(
        item_id="item-1",
        reviewer="Claude",
        source_round=1,
        text="Document AGENT_PLAN_APPROVED_FOLLOWUPS handling.",
        notes=("AGENT_SPLIT_UNFILED_WARNING was also discussed.",),
    )
    plan_reconciliation = reconcile_plan_approved_followups([source])
    plan_body = _plan_followup_issue_body(
        issue_number=56,
        plan_hash="abc123",
        plan_subject="Plan AGENT_MANAGED_PR_SOURCE_V1 handling.",
        followup=plan_reconciliation.selected_groups[0],
    )
    plan_summary = _format_plan_approval_summary_with_followups(
        56,
        "Plan AGENT_MANAGED_PR_SOURCE_V1 handling.",
        reconciliation=plan_reconciliation,
    )

    for rendered in (plan_body, plan_summary):
        assert "AGENT_PLAN_APPROVED_FOLLOWUPS" not in rendered
        assert "AGENT_SPLIT_UNFILED_WARNING" not in rendered
        assert "AGENT_MANAGED_PR_SOURCE_V1" not in rendered


def test_direct_pr_keeps_linked_issue_context_for_coder_followup(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Needs a small correction.\n\n### Same-PR follow-ups\n"
            "- Correct the implementation.\n\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=["Corrected it.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
        issue_payload={"title": "Linked issue title", "body": "Linked issue body"},
        pr_payload={"body": "Fixes #56"},
    )
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    followup_prompt = next(
        cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]
    )
    assert "Linked issue title" in followup_prompt
    assert "Linked issue body" in followup_prompt


@pytest.mark.parametrize(
    "body, issue_views",
    [
        ("https://github.com/owner/repo/issues/56", 1),
        ("No issue reference.", 0),
        ("Fixes #56; Resolves #57", 0),
        ("Fixes #56; https://github.com/OWNER/REPO/issues/56", 1),
        ("Fixes other/repo#56 https://github.com/other/repo/issues/57", 0),
    ],
)
def test_direct_pr_linked_issue_resolution_handles_urls_absence_and_ambiguity(
    tmp_path, body, issue_views
):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={"body": body},
    )
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(_issue_view_commands(runner)) == issue_views


def test_issue_mode_context_is_not_replaced_by_pr_link(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        issue_payload={"title": "Fresh issue", "body": "Fresh issue body."},
        pr_payload={"body": "Fixes #99"},
    )
    config = make_config(tmp_path)
    supplied_context = IssueContext(
        number=56, repo="OWNER/REPO", title="Caller issue", body="Caller body", url=None, comments=()
    )

    assert run_pr_loop(runner, pr_number=77, config=config, issue_context=supplied_context) == 0

    assert len(_issue_view_commands(runner)) == 1
    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "Fresh issue" in prompt
    assert "Caller issue" not in prompt


def test_direct_pr_ignores_unrelated_older_issue_handoff(tmp_path):
    older_handoff = format_issue_pr_handoff_comment(
        issue_number=56,
        pr_number=66,
        pr_url="https://github.com/OWNER/REPO/pull/66",
        pr_head_sha="older-head",
        flow="issue-implementation",
        plan_hash=None,
    )
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        issue_comments=[
            {
                "author": {"login": "coding-review-agent-loop"},
                "createdAt": "2026-05-01T00:00:00Z",
                "body": older_handoff,
            }
        ],
        pr_payload={"body": "Fixes #56"},
    )

    assert run_pr_loop(runner, pr_number=77, config=make_config(tmp_path)) == 0

    assert any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)
    assert not any("handoff selects PR #66" in comment for comment in runner.comments)


def test_direct_pr_resume_uses_handoff_bound_plan_when_later_plan_exists(tmp_path):
    old_matrix = parse_risk_test_matrix({
        "applicability": "applicable",
        "rows": [{
            "row_id": "row-review-only-recovery",
            "label": "Review-only recovery reaches the intended branch",
            "entry_path_or_mode": "ordinary / review-only",
            "initial_state": "review complete, recovery pending",
            "event": "repaired head passes",
            "expected_outcome": "Complete without requiring a watcher",
            "forbidden_side_effects": ["Do not merge a stale head"],
            "proposed_test_level": "orchestrator",
            "proposed_test_location": "tests/test_orchestrator_pr.py::test_review_only_recovery",
            "applicability": "applicable",
            "related_scope_item_ids": ["scope-review-recovery"],
            "execution_owner": "one-shot",
        }],
        "important_exclusions": ["Unrelated managed-CI combinations."],
    })
    later_matrix = parse_risk_test_matrix({
        **old_matrix.to_payload(),
        "rows": [{
            **old_matrix.rows[0].to_payload(),
            "row_id": "row-unrelated-later-plan",
            "expected_outcome": "Use a watcher before completion",
        }],
    })
    old_identity = risk_test_matrix_identity(old_matrix)
    later_identity = risk_test_matrix_identity(later_matrix)
    old_plan = "Preserve the old API.\n\nApproved old plan.\n\n" + render_risk_test_matrix_section(old_matrix)
    later_plan = "Replace the old API.\n\nUnrelated later plan.\n\n" + render_risk_test_matrix_section(later_matrix)
    old_context = execution_policy.make_approved_plan_context(
        old_plan,
        source_locator="test old approved plan",
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=old_matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=old_identity,
        risk_test_matrix_boundary_digest=old_identity,
    )
    later_context = execution_policy.make_approved_plan_context(
        later_plan,
        source_locator="test later plan",
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=later_matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=later_identity,
        risk_test_matrix_boundary_digest=later_identity,
    )
    old_plan_comment = _attach_round_metadata(
        old_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=1,
            subject=old_context.plan_subject or "old-plan",
            canonical_plan=old_plan,
            raw_structured_coder_response=old_plan,
            risk_test_matrix_contract_version=1,
            risk_test_matrix_payload=old_matrix.to_payload(),
            risk_test_matrix_identity=old_identity,
            risk_test_matrix_boundary_digest=old_identity,
        ),
    )
    later_plan_comment = _attach_round_metadata(
        later_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=later_context.plan_subject or "later-plan",
            canonical_plan=later_plan,
            raw_structured_coder_response=later_plan,
            risk_test_matrix_contract_version=1,
            risk_test_matrix_payload=later_matrix.to_payload(),
            risk_test_matrix_identity=later_identity,
            risk_test_matrix_boundary_digest=later_identity,
        ),
    )
    handoff = format_issue_pr_handoff_comment(
        issue_number=56,
        pr_number=77,
        pr_url="https://github.com/OWNER/REPO/pull/77",
        pr_head_sha="abc123",
        flow="approved-plan-implementation",
        plan_hash=panel_evidence.approved_plan_hash(old_plan),
    )
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        issue_comments=[
            {"author": {"login": "bot"}, "createdAt": "2026-05-01T00:00:00Z", "body": old_plan_comment},
            {"author": {"login": "bot"}, "createdAt": "2026-05-01T00:01:00Z", "body": handoff},
            {"author": {"login": "bot"}, "createdAt": "2026-05-01T00:02:00Z", "body": later_plan_comment},
        ],
        issue_payload={"number": 56, "title": "Issue", "body": "Original issue."},
        pr_payload={
            "number": 77,
            "url": "https://github.com/OWNER/REPO/pull/77",
            "body": "Fixes #56",
        },
    )

    assert run_pr_loop(runner, pr_number=77, config=make_config(tmp_path)) == 0

    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    plan_block = prompt.split("Approved implementation plan context", 1)[1].split(
        "Target child/primary issue context", 1
    )[0]
    assert "Preserve the old API." in plan_block
    assert "Replace the old API." not in plan_block
    assert "row-review-only-recovery" in plan_block
    assert "row-unrelated-later-plan" not in plan_block


def _plan_comment(plan: str) -> IssueComment:
    return IssueComment(
        author="coding-review-agent-loop",
        created_at="2026-05-01T00:00:00Z",
        body=_attach_round_metadata(
            plan,
            PostedRoundMetadata(
                flow="plan", role="coder", agent="Claude", round_number=1,
                subject=panel_evidence._plan_subject(plan), canonical_plan=plan,
                raw_structured_coder_response=plan,
            ),
        ),
    )


def _handoff_comment(plan_hash: str) -> IssueComment:
    return IssueComment(
        author="coding-review-agent-loop",
        created_at="2026-05-01T00:02:00Z",
        body=format_issue_pr_handoff_comment(
            issue_number=56, pr_number=77,
            pr_url="https://github.com/OWNER/REPO/pull/77",
            pr_head_sha="abc123", flow="approved-plan-implementation",
            plan_hash=plan_hash,
        ),
    )


def _issue(number: int, comments: tuple[IssueComment, ...]) -> IssueContext:
    return IssueContext(
        number=number, repo="OWNER/REPO", title=f"Issue {number}",
        body="Scope.", url=f"https://github.com/OWNER/REPO/issues/{number}",
        comments=comments,
    )


def _managed_resume_runner() -> FakeRunner:
    return FakeRunner(
        pr_payload={
            "headRefName": "agent-loop/managed-56", "headRefOid": "abc123",
            "baseRefName": "main", "body": "Fixes #56",
        },
    )


def _ordinary_managed_config(tmp_path, **overrides):
    return make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        reviewer=("codex",), **overrides,
    )


def _issue_created_handoff() -> pr_loop_support.AuthenticatedIssueCreatedHandoff:
    return pr_loop_support.AuthenticatedIssueCreatedHandoff(
        pr_number=77, issue_number=56, repository="OWNER/REPO", base_ref="main",
        head_sha="abc123", branch="agent-loop/managed-56",
        trusted_actor_login="agent-loop", trusted_actor_id=1,
        protection_mode="voluntary", override_nonce="opening-nonce",
    )


def test_pr_ordinary_resume_without_parent_context_still_fails_closed(tmp_path, monkeypatch):
    """Plain PR-mode resume has no parent identity and keeps the hard stop."""
    plan = "Approved plan.\n\n### Plan steps\n1. Preserve the trust boundary."
    child = _issue(56, (_handoff_comment(panel_evidence.approved_plan_hash(plan)),))
    patch_attr_in(monkeypatch, (
        discuss_loop, pr_loop, issue_loop,
    ), "validate_open_issue", lambda *_a, **_k: None)
    patch_attr_in(monkeypatch, (
        discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop,
        plan_first_loop, issue_loop,
    ), "get_issue_context", lambda *_a, **_k: child
    )
    monkeypatch.setattr(
        pr_loop, "recover_issue_created_handoff",
        lambda *_a, **_k: _issue_created_handoff(),
    )
    monkeypatch.setattr(
        pr_loop, "revalidate_issue_created_handoff",
        lambda *_a, **_k: (_ for _ in ()).throw(_FreshScopeCaptured()),
    )
    runner = _managed_resume_runner()

    with pytest.raises(
        AgentLoopError,
        match="Managed-CI ordinary resume could not recover the canonical approved plan.",
    ):
        run_pr_loop(runner, pr_number=77, config=_ordinary_managed_config(tmp_path))

    assert not any(command[:1] in (["claude"], ["codex"]) for command, _cwd in runner.commands)


def test_pr_ordinary_resume_ignores_parent_when_child_plan_recovers(tmp_path, monkeypatch):
    """A recoverable child plan wins and the parent comments are never read."""
    plan = "Approved plan.\n\n### Plan steps\n1. Preserve the trust boundary."
    decoy = "Decoy plan.\n\n### Plan steps\n1. Do something else entirely."
    plan_hash = panel_evidence.approved_plan_hash(plan)
    child = _issue(56, (_plan_comment(plan), _handoff_comment(plan_hash)))
    parent = _issue(55, (_plan_comment(decoy),))
    read_comments: list[int] = []

    real_recover = execution_policy.recover_approved_plan_context

    def recording_recover(comments, **kwargs):
        read_comments.append(len(comments))
        return real_recover(comments, **kwargs)

    patch_attr_in(monkeypatch, (
        execution_policy, child_plan_binding, pr_loop_support, pr_loop, issue_loop,
    ), "recover_approved_plan_context", recording_recover)
    patch_attr_in(monkeypatch, (
        discuss_loop, pr_loop, issue_loop,
    ), "validate_open_issue", lambda *_a, **_k: None)
    patch_attr_in(monkeypatch, (
        discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop,
        plan_first_loop, issue_loop,
    ), "get_issue_context",
        lambda _runner, *, config, issue_number: child if issue_number == 56 else parent,
    )
    monkeypatch.setattr(
        pr_loop, "recover_issue_created_handoff",
        lambda *_a, **_k: _issue_created_handoff(),
    )
    captured: dict[str, object] = {}

    def revalidate(*_args, **kwargs):
        captured.update(kwargs)
        raise _FreshScopeCaptured

    monkeypatch.setattr(pr_loop, "revalidate_issue_created_handoff", revalidate)
    runner = _managed_resume_runner()

    with pytest.raises(_FreshScopeCaptured):
        run_pr_loop(
            runner, pr_number=77, config=_ordinary_managed_config(tmp_path),
            parent_issue_context=parent,
        )

    assert captured["handoff"].approved_plan_hash == plan_hash
    # Only the child comments were consulted by the managed-CI recovery.
    assert read_comments == [len(child.comments)]


def test_pr_fresh_authorization_recovers_parent_held_plan(tmp_path, monkeypatch):
    """A staged child under --managed-ci-fresh recovers the parent-held plan."""
    plan = "Approved plan.\n\n### Plan steps\n1. Preserve the trust boundary."
    plan_hash = panel_evidence.approved_plan_hash(plan)
    child = _issue(56, (_handoff_comment(plan_hash),))
    parent = _issue(55, (_plan_comment(plan),))
    patch_attr_in(monkeypatch, (
        discuss_loop, pr_loop, issue_loop,
    ), "validate_open_issue", lambda *_a, **_k: None)
    patch_attr_in(monkeypatch, (
        discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop,
        plan_first_loop, issue_loop,
    ), "get_issue_context",
        lambda _runner, *, config, issue_number: child if issue_number == 56 else parent,
    )
    captured: dict[str, object] = {}

    def authorize(*_args, **kwargs):
        captured.update(kwargs)
        raise _FreshScopeCaptured

    monkeypatch.setattr(pr_loop, "authorize_fresh_issue_created_resume", authorize)
    config = _ordinary_managed_config(
        tmp_path, managed_ci_fresh_authorization=True, managed_ci_issue_number=56,
    )
    runner = _managed_resume_runner()

    with pytest.raises(_FreshScopeCaptured):
        run_pr_loop(
            runner, pr_number=77, config=config, parent_issue_context=parent,
        )

    assert captured["approved_plan_hash"] == plan_hash
    assert not any(command[:1] in (["claude"], ["codex"]) for command, _cwd in runner.commands)


def test_pr_fresh_authorization_without_parent_context_still_fails_closed(tmp_path, monkeypatch):
    plan = "Approved plan.\n\n### Plan steps\n1. Preserve the trust boundary."
    child = _issue(56, (_handoff_comment(panel_evidence.approved_plan_hash(plan)),))
    patch_attr_in(monkeypatch, (
        discuss_loop, pr_loop, issue_loop,
    ), "validate_open_issue", lambda *_a, **_k: None)
    patch_attr_in(monkeypatch, (
        discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop,
        plan_first_loop, issue_loop,
    ), "get_issue_context", lambda *_a, **_k: child)
    monkeypatch.setattr(
        pr_loop, "authorize_fresh_issue_created_resume",
        lambda *_a, **_k: (_ for _ in ()).throw(_FreshScopeCaptured()),
    )
    config = _ordinary_managed_config(
        tmp_path, managed_ci_fresh_authorization=True, managed_ci_issue_number=56,
    )

    with pytest.raises(
        AgentLoopError,
        match="could not recover the canonical approved plan for the explicit issue scope",
    ):
        run_pr_loop(_managed_resume_runner(), pr_number=77, config=config)


# --- Approval recovery reads the current planning scope (#1390) ---------------


def _m1390_patch(base, round_number, *, item_id, summary):
    """A semantic planner patch bound to ``base`` that resolves ``item_id``."""
    from coding_review_agent_loop.plan_assembly import (
        AuthenticatedPlanState,
        assemble_plan_revision,
    )

    payload = {
        "schema_version": 1,
        "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1,
        "state": "blocking",
        "summary": summary,
        "prior_plan_item_dispositions": [{"item_id": item_id, "disposition": "resolved"}],
        "base_round_number": round_number,
        "base_state_identity": base.state_identity,
        "operations": [
            {"op": "replace", "field": "plan_steps", "value": [f"{summary} Revision."]}
        ],
    }
    next_base = AuthenticatedPlanState.from_plan(
        assemble_plan_revision(base, payload), round_number=round_number + 1
    )
    return json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude", next_base


def _m1390_scoped_approval_history(tmp_path):
    """Archived round-3 plan with open item-3; boundary at 4; fresh plan approved at 5."""
    from agent_loop_helpers import structured_v1_plan_state
    from coding_review_agent_loop.plan_assembly import AuthenticatedPlanState
    from coding_review_agent_loop.protocol import validate_structured_plan_state

    actor = ("agent-loop", 4242)
    fresh = structured_v1_plan_state()
    base = AuthenticatedPlanState.from_plan(validate_structured_plan_state(fresh), round_number=1)
    planner, reviews = [fresh], []
    for n in (1, 2, 3):
        reviews.append(structured_plan_review(
            state="blocking", blocking_plan_issues=[f"Archived gap {n}."],
            prior_plan_item_dispositions=(
                [{"item_id": f"item-{n - 1}", "disposition": "resolved"}] if n > 1 else None
            ),
        ))
        patch_text, base = _m1390_patch(base, n, item_id=f"item-{n}", summary=f"Close {n}.")
        planner.append(patch_text)
    first = FakeRunner(
        claude_outputs=planner, codex_outputs=reviews,
        authenticated_actor=actor, serve_rest_issue_comments=True,
    )
    with pytest.raises(AgentLoopError):
        run_issue_loop(
            first, issue_number=56, config=make_config(tmp_path, max_rounds=3), plan_first=True,
        )

    fresh_base = AuthenticatedPlanState.from_plan(
        validate_structured_plan_state(fresh), round_number=4
    )
    revision, _base = _m1390_patch(fresh_base, 4, item_id="item-1", summary="Fresh scope fix.")
    edited = FakeRunner(
        issue_comments=list(first.issue_comments),
        issue_payload={"body": "Narrowed scope: a two-command design."},
        claude_outputs=[fresh, revision],
        codex_outputs=[
            structured_plan_review(state="blocking", blocking_plan_issues=["Fresh gap."]),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        authenticated_actor=actor, serve_rest_issue_comments=True,
    )
    config = make_config(tmp_path, max_rounds=5)
    assert run_issue_loop(edited, issue_number=56, config=config, plan_first=True) == 0
    return edited, config


def test_m1390_approval_recovery_and_strict_qualification_read_the_current_scope(tmp_path):
    """`approval-recovery-scoped`."""
    from coding_review_agent_loop.decomposition import approved_plan_hash
    from coding_review_agent_loop.github import get_issue_context
    from coding_review_agent_loop.plan_board_state import admitted_issue_comments, plan_resume_round
    from coding_review_agent_loop.plan_verification import (
        classify_plan_approval,
        require_recoverable_plan_approval,
    )
    from coding_review_agent_loop.pr_loop import _canonical_plan_without_handoff
    from coding_review_agent_loop.pr_loop_support import _verify_strict_managed_plan_binding
    from coding_review_agent_loop.round_state import _decode_round_metadata

    runner, config = _m1390_scoped_approval_history(tmp_path)
    records = []
    for comment in runner.issue_comments:
        match = re.search(r"<!-- AGENT_LOOP_META: (?P<payload>\S+) -->", comment["body"])
        if match is not None:
            records.append(_decode_round_metadata(match.group("payload")))
    coders = [r for r in records if r.flow == "plan" and r.role == "coder"]
    assert [r.round_number for r in coders] == [1, 2, 3, 4, 5]
    # The archived round-3 review published item-3, which nothing carried or cleared.
    assert any(
        item.item_id == "item-3" and item.status == "blocking"
        for r in records if r.role == "reviewer" and r.round_number == 3
        for item in r.new_items
    )
    approved_plan = coders[-1].canonical_plan
    expected_hash = approved_plan_hash(approved_plan)

    issue = get_issue_context(runner, config=config, issue_number=56)
    comments, actor = admitted_issue_comments(
        runner, config=config, issue_number=56, comments=issue.comments,
    )
    assert actor is not None
    # Every reader derives scope start 4 from the admitted boundary.
    assert classify_plan_approval(
        comments, config=config, issue_number=56, trusted_actor=actor,
        poster_completion_proven=True, execution_mode_guard=False,
    ).approved
    require_recoverable_plan_approval(
        runner, config=config, issue_context=issue, comments=comments,
        trusted_actor=actor, error_message="not approved",
    )
    recovered = _canonical_plan_without_handoff(
        config, issue, source_locator="issue #56 plan", error_message="not approved",
        runner=runner,
    )
    assert recovered is not None and recovered.plan_hash == expected_hash
    _verify_strict_managed_plan_binding(
        config=config, pr_number=7, issue_context=issue,
        metadata=SimpleNamespace(head_branch="agent-loop/managed-56", head_sha="head-1"),
        expected_plan_hash=expected_hash, runner=runner,
    )
    # Issue-loop comment-only recovery re-enters the fresh scope's round.
    assert plan_resume_round(comments, config, trusted_actor=actor) == 5

    # An unadmitted boundary is never honoured: the archived item-3 is then an
    # unreconstructible obligation and recovery fails closed.
    for comment in runner.issue_comments:
        if "AGENT_PLAN_SCOPE_BOUNDARY" in comment["body"]:
            comment["author"] = {"login": "mallory"}
            comment["_rest_author_id"] = 999
    forged_issue = get_issue_context(runner, config=config, issue_number=56)
    forged_comments, forged_actor = admitted_issue_comments(
        runner, config=config, issue_number=56, comments=forged_issue.comments,
    )
    with pytest.raises(AgentLoopError, match="item-3"):
        require_recoverable_plan_approval(
            runner, config=config, issue_context=forged_issue, comments=forged_comments,
            trusted_actor=forged_actor, error_message="not approved",
        )
