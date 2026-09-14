"""Offline contract tests for the repository's managed-CI workflow."""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import pytest

from fixtures.managed_ci import current_router


ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
CURRENT_FIXTURE = ROOT / "tests" / "fixtures" / "managed_ci" / "current_router.py"


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _validator_block(text: str) -> str:
    marker = text.index("# BEGIN MANAGED_CI_V2_VALIDATOR")
    start = text.rfind("\n", 0, marker) + 1
    end = text.index("# END MANAGED_CI_V2_VALIDATOR", start)
    end = text.index("\n", end) + 1
    return textwrap.dedent(text[start:end])


def _pr(**overrides):
    value = {
        "state": "open",
        "draft": True,
        "number": 7,
        "base": {"ref": "main"},
        "head": {
            "sha": "b" * 40,
            "ref": "agent-loop/managed-803",
            "repo": {"full_name": "OWNER/REPO"},
        },
        "user": {"login": "agent-loop", "id": 7},
        "labels": [{"name": "agent-loop-managed"}],
    }
    value.update(overrides)
    return value


def _record(state="dispatch-requested", **overrides):
    value = {
        "version": 2,
        "repository": "OWNER/REPO",
        "pr": 7,
        "expected_head_sha": "b" * 40,
        "base_ref": "main",
        "workflow_revision": "a" * 40,
        "generation": "generation-803",
        "nonce": "n" * 32,
        "created_at": 1,
        "state": state,
        "run_id": None,
        "run_attempt": None,
        "terminal_run_id": None,
        "terminal_run_attempt": None,
        "terminal_outcome": None,
        "terminal_attempts": [],
    }
    if state in {"attached", "completed"}:
        value.update(run_id=100, run_attempt=1)
    value.update(overrides)
    return value


def _pages(*records, actor="agent-loop", actor_id=7):
    return [[
        {
            "user": {"login": actor, "id": actor_id},
            "body": "<!-- AGENT_MANAGED_CI_INTENT_V2 "
            + json.dumps(record, separators=(",", ":"))
            + " -->",
        }
        for record in records
    ]]


def _validate(record, *, pr=None, pages=None, actor="agent-loop", actor_id=7):
    return current_router.validate(
        pr or _pr(),
        pages if pages is not None else _pages(record),
        "OWNER/REPO",
        "7",
        "b" * 40,
        "n" * 32,
        actor,
        "a" * 40,
        actor_id,
    )


def test_workflow_advertises_exact_activation_contract():
    workflow = _workflow_text()

    assert "AGENT_LOOP_MANAGED_CI_V2: enabled" in workflow
    assert "AGENT_LOOP_MANAGED_CI_UNLABELED_RECOVERY_V1: enabled" in workflow
    for required in (
        "workflow_dispatch",
        "protocol_version",
        "pr_number",
        "expected_head_sha",
        "managed_nonce",
        "final-ci/exact-head",
        "run-name: managed-ci-v2 nonce=${{ inputs.managed_nonce }}",
    ):
        assert required in workflow

    pull_request = re.search(r"  pull_request:\n    types: \[([^\]]+)\]", workflow)
    assert pull_request is not None
    assert [item.strip() for item in pull_request.group(1).split(",")] == [
        "opened", "synchronize", "reopened", "unlabeled"
    ]
    assert "labeled" not in [item.strip() for item in pull_request.group(1).split(",")]
    assert "ready_for_review" not in pull_request.group(1)
    assert "converted_to_draft" not in pull_request.group(1)


def test_validator_fixture_is_an_extraction_of_production_workflow():
    assert _validator_block(_workflow_text()) == _validator_block(
        CURRENT_FIXTURE.read_text(encoding="utf-8")
    )


@pytest.mark.parametrize("state", ["dispatch-requested", "attached", "completed"])
def test_validator_accepts_each_dispatch_lifecycle_state(state):
    assert _validate(_record(state))["state"] == state


def test_validator_rejects_prepared_duplicate_and_foreign_records():
    with pytest.raises(ValueError, match="prepared intent"):
        _validate(_record("prepared"))
    with pytest.raises(ValueError, match="exactly one fresh"):
        _validate(_record(), pages=_pages(_record(), _record()))

    foreign = _record(expected_head_sha="c" * 40)
    with pytest.raises(ValueError, match="trusted intent binding drifted"):
        _validate(foreign)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repository", "OTHER/REPO"),
        ("pr", 8),
        ("expected_head_sha", "c" * 40),
        ("base_ref", "release"),
        ("workflow_revision", "c" * 40),
        ("generation", ""),
        ("nonce", "x" * 32),
    ],
)
def test_validator_rejects_every_binding_drift(field, value):
    record = _record(**{field: value})
    with pytest.raises(ValueError):
        _validate(record)


@pytest.mark.parametrize(
    "pr",
    [
        _pr(base={"ref": "release"}),
        _pr(head={"sha": "c" * 40, "ref": "agent-loop/managed-803", "repo": {"full_name": "OWNER/REPO"}}),
        _pr(head={"sha": "b" * 40, "ref": "feature", "repo": {"full_name": "OWNER/REPO"}}),
        _pr(head={"sha": "b" * 40, "ref": "agent-loop/managed-803", "repo": {"full_name": "someone/REPO"}}),
        _pr(user={"login": "other", "id": 7}),
        _pr(user={"login": "agent-loop", "id": 8}),
        _pr(labels=[]),
        _pr(draft=False),
    ],
)
def test_validator_fails_closed_for_live_pr_tuple_drift(pr):
    with pytest.raises(ValueError):
        _validate(_record(), pr=pr)


def test_validator_rejects_unauthorized_comment_author_and_malformed_terminal_history():
    with pytest.raises(ValueError, match="exactly one fresh"):
        _validate(_record(), pages=_pages(_record(), actor="other", actor_id=99))

    record = _record(terminal_attempts=[{"run_id": 100, "run_attempt": 1}, {"run_id": 100, "run_attempt": 1}])
    with pytest.raises(ValueError, match="duplicate terminal"):
        _validate(record)

    record = _record(terminal_run_attempt=2)
    with pytest.raises(ValueError, match="no terminal run"):
        _validate(record)


def _ordinary_route(action, pr, trusted_actor):
    """Reference the workflow's fail-open four-event routing matrix."""
    if action == "unlabeled":
        return True
    managed_tuple = (
        pr.get("base", {}).get("ref") == "main"
        and pr.get("base", {}).get("repo", {}).get("full_name", "OWNER/REPO") == "OWNER/REPO"
        and pr.get("head", {}).get("repo", {}).get("full_name") == "OWNER/REPO"
        and pr.get("head", {}).get("ref", "").startswith("agent-loop/managed-")
        and pr.get("draft") is True
        and bool(trusted_actor)
        and pr.get("user", {}).get("login") == trusted_actor
    )
    if action == "opened":
        return not managed_tuple
    return not (
        managed_tuple
        and action in {"synchronize", "reopened"}
        and any(label.get("name") == "agent-loop-managed" for label in pr.get("labels", []))
    )


def test_routing_matrix_is_label_race_safe_and_fail_open():
    opening = _pr(labels=[])
    labeled = _pr()
    assert _ordinary_route("opened", opening, "agent-loop") is False
    assert _ordinary_route("synchronize", opening, "agent-loop") is True
    assert _ordinary_route("reopened", labeled, "agent-loop") is False
    assert _ordinary_route("synchronize", labeled, "wrong-actor") is True
    assert _ordinary_route("unlabeled", labeled, "agent-loop") is True
    assert _ordinary_route("synchronize", _pr(head={"sha": "b" * 40, "ref": "feature", "repo": {"full_name": "fork/REPO"}}), "agent-loop") is True


def test_workflow_keeps_exact_checkout_suite_and_safe_terminal_publisher():
    workflow = _workflow_text()
    assert "ref: ${{ needs.validate-managed.outputs.target_sha }}" in workflow
    assert 'test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD_SHA"' in workflow
    assert workflow.count("- run: python -m pytest") >= 2
    assert "permissions:\n      statuses: write" in workflow
    assert "Authorization failed before an exact target was established; no status written." in workflow
    assert "context': 'final-ci/exact-head'" in workflow
    assert "target_url'" in workflow
    assert "nonce=' + os.environ['NONCE'] + ';run_id='" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "GH_REF: ${{ github.ref }}" in workflow
    assert "managed dispatch must execute the base workflow from main" in workflow
