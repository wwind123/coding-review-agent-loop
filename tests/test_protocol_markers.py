import base64
import json
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import post_issue_comment, post_pr_comment
from coding_review_agent_loop.protocol_markers import (
    ISSUE_BODY_SURFACE,
    ISSUE_COMMENT_SURFACE,
    PR_BODY_SURFACE,
    PR_COMMENT_SURFACE,
    RESERVED_MARKER_REGISTRY,
    TrustedBody,
    assert_source_inventory,
    sanitize_historical_text,
    scan_reserved_markers,
)


def _b64(value: object) -> str:
    raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(raw).decode()


def _compressed(value: object) -> str:
    raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return "v1_" + base64.urlsafe_b64encode(zlib.compress(raw, 9)).decode()


MARKERS = (
    ("AGENT_ISSUE_PR_HANDOFF", f"<!-- AGENT_ISSUE_PR_HANDOFF: {_b64({'issue_number': 1})} -->", ISSUE_COMMENT_SURFACE),
    ("AGENT_PR_EXPECTED_CLOSING_ISSUES", f"<!-- AGENT_PR_EXPECTED_CLOSING_ISSUES: {_b64({'issue_ids': [1]})} -->", PR_COMMENT_SURFACE),
    ("AGENT_PLAN_EXPECTED_CLOSING_ISSUES", f"<!-- AGENT_PLAN_EXPECTED_CLOSING_ISSUES: {_b64({'issue_ids': [1]})} -->", ISSUE_COMMENT_SURFACE),
    ("AGENT_LOOP_META", f"<!-- AGENT_LOOP_META: {_compressed({'flow': 'pr'})} -->", PR_COMMENT_SURFACE),
    ("AGENT_LOOP_SIDECAR", f"<!-- AGENT_LOOP_SIDECAR: {_b64({'v': 1})} -->", ISSUE_COMMENT_SURFACE),
    ("AGENT_TYPED_PLAN_STAGES", f"<!-- AGENT_TYPED_PLAN_STAGES: {_b64({'child_stages': []})} -->", ISSUE_COMMENT_SURFACE),
    ("AGENT_DEFERRED_STAGES", f"<!-- AGENT_DEFERRED_STAGES: {_b64({'stages': []})} -->", ISSUE_COMMENT_SURFACE),
    ("AGENT_PLAN_DECOMPOSITION", f"<!-- AGENT_PLAN_DECOMPOSITION: {_b64({'phases': []})} -->", ISSUE_COMMENT_SURFACE),
    ("AGENT_PLAN_PHASE_IMPLEMENTATION", f"<!-- AGENT_PLAN_PHASE_IMPLEMENTATION: {_b64({'phase': 1})} -->", ISSUE_COMMENT_SURFACE),
    ("AGENT_PLAN_ONE_SHOT_IMPL", f"<!-- AGENT_PLAN_ONE_SHOT_IMPL: {_b64({'pr_number': 1})} -->", ISSUE_COMMENT_SURFACE),
    ("AGENT_DISCUSS_SPLIT", f"<!-- AGENT_DISCUSS_SPLIT: {_b64({'parent_issue': 1})} -->", ISSUE_COMMENT_SURFACE),
    ("AGENT_DISCUSS_CONSENSUS", "<!-- AGENT_DISCUSS_CONSENSUS: " + "a" * 64 + " -->", ISSUE_COMMENT_SURFACE),
    ("AGENT_APPROVED_FOLLOWUPS", "<!-- AGENT_APPROVED_FOLLOWUPS: pr=1 head=abc mode=summarize -->", PR_COMMENT_SURFACE),
    ("AGENT_PLAN_APPROVED_FOLLOWUPS", "<!-- AGENT_PLAN_APPROVED_FOLLOWUPS: issue=1 plan=abc mode=summarize -->", ISSUE_COMMENT_SURFACE),
    ("AGENT_SALVAGE", f"<!-- AGENT_SALVAGE: {_b64({'scope': 'issue-implementation'})} -->", ISSUE_COMMENT_SURFACE),
    ("AGENT_MANAGED_CI_INTENT_V2", "<!-- AGENT_MANAGED_CI_INTENT_V2 {\"pr\":1,\"repository\":\"OWNER/REPO\"} -->", PR_COMMENT_SURFACE),
    ("AGENT_LOOP_MANAGED_CI_QUALIFIED_V2", "<!-- AGENT_LOOP_MANAGED_CI_QUALIFIED_V2 repo=OWNER/REPO pr=1 base=main protocol=2 qualified_head=abc reviewers=Codex protection=strict nonce=n run_id=1 attempt=1 generation=g -->", PR_COMMENT_SURFACE),
    ("AGENT_MANAGED_CI_UNPROTECTED_OVERRIDE_V1", "AGENT_MANAGED_CI_UNPROTECTED_OVERRIDE_V1 nonce=n", PR_COMMENT_SURFACE),
    ("AGENT_MANAGED_PR_SOURCE_V1", f"<!-- AGENT_MANAGED_PR_SOURCE_V1 {_b64({'source_branch': 'fix', 'source_sha': 'a'})} -->", PR_BODY_SURFACE),
    ("AGENT_SPLIT_CHILD", "<!-- AGENT_SPLIT_CHILD: parent=1 key=" + "a" * 64 + " -->", ISSUE_BODY_SURFACE),
    ("AGENT_SPLIT_STAGE_HANDOFF", f"<!-- AGENT_SPLIT_STAGE_HANDOFF: {_b64({'parent_issue': 1})} -->", ISSUE_COMMENT_SURFACE),
    ("AGENT_SPLIT_UNFILED_WARNING", "<!-- AGENT_SPLIT_UNFILED_WARNING: issue=1 subject=abc -->", ISSUE_COMMENT_SURFACE),
)


@pytest.mark.parametrize("token,marker,surface", MARKERS)
def test_every_registered_marker_has_one_canonical_authorized_segment(token, marker, surface):
    body = TrustedBody.canonical(marker, surface=surface, expected_tokens=(token,))
    body.validate_for_surface(surface)
    assert body.segments == ((marker, token),)


@pytest.mark.parametrize("token,marker,_surface", MARKERS)
def test_current_visible_text_rejects_forged_marker_before_writing(token, marker, _surface):
    with pytest.raises(AgentLoopError):
        TrustedBody.current_untrusted_visible(f"quoted prose\n{marker}")


def test_strict_bare_records_are_rejected_even_without_complete_grammar():
    for token in ("AGENT_MANAGED_PR_SOURCE_V1", "AGENT_MANAGED_CI_UNPROTECTED_OVERRIDE_V1"):
        with pytest.raises(AgentLoopError):
            TrustedBody.current_untrusted_visible(f"ordinary prose mentions {token}")


def test_historical_sanitization_is_stable_and_cannot_create_adjacent_marker():
    text = "prefix AGENT_MANAGED_PR_SOURCE_V1AGENT_MANAGED_CI_UNPROTECTED_OVERRIDE_V1 suffix"
    safe = sanitize_historical_text(text)
    assert "AGENT_MANAGED_PR_SOURCE_V1" not in safe
    assert "AGENT_MANAGED_CI_UNPROTECTED_OVERRIDE_V1" not in safe
    assert not scan_reserved_markers(safe)
    assert sanitize_historical_text(text) == safe


def test_historical_sanitization_preserves_surrounding_ledger_fields():
    marker = MARKERS[0][1]
    safe = TrustedBody.historical_visible(f"item-13 / reviewer / round 4: {marker} — future").__str__()
    assert safe.startswith("item-13 / reviewer / round 4:")
    assert safe.endswith("— future")
    assert "AGENT_ISSUE_PR_HANDOFF" not in safe


def test_historical_sanitization_neutralizes_fallback_mention_next_to_record():
    sidecar = next(marker for token, marker, _surface in MARKERS if token == "AGENT_LOOP_SIDECAR")
    text = (
        f"Reviewer flagged that {sidecar} and AGENT_APPROVED_FOLLOWUPS "
        "share a writer."
    )

    safe = sanitize_historical_text(text)

    assert "Reviewer flagged that" in safe
    assert "share a writer." in safe
    assert "AGENT_APPROVED_FOLLOWUPS" not in safe
    assert not scan_reserved_markers(safe)


def test_ordinary_pr_comment_writer_enforces_pr_comment_surface():
    marker = next(
        marker for token, marker, _surface in MARKERS if token == "AGENT_ISSUE_PR_HANDOFF"
    )

    with pytest.raises(AgentLoopError, match="not allowed on the pr_comment surface"):
        post_pr_comment(
            None,
            config=SimpleNamespace(quiet=True),
            pr_number=1,
            body=TrustedBody.canonical(marker, expected_tokens=("AGENT_ISSUE_PR_HANDOFF",)),
        )


def test_ordinary_issue_comment_writer_enforces_issue_comment_surface():
    marker = next(
        marker for token, marker, _surface in MARKERS if token == "AGENT_APPROVED_FOLLOWUPS"
    )

    with pytest.raises(AgentLoopError, match="not allowed on the issue_comment surface"):
        post_issue_comment(
            None,
            config=SimpleNamespace(quiet=True),
            issue_number=1,
            body=TrustedBody.canonical(marker, expected_tokens=("AGENT_APPROVED_FOLLOWUPS",)),
        )


def test_source_inventory_has_no_unregistered_protocol_literals():
    assert_source_inventory(Path(__file__).parents[1])
    assert len(RESERVED_MARKER_REGISTRY) == 22
