"""Offline contract tests for the repository's managed-CI workflow."""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import pytest

from fixtures.managed_ci import dispatch_validator, local_router


ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
LOCAL_FIXTURE = ROOT / "tests" / "fixtures" / "managed_ci" / "local_router.py"
DISPATCH_FIXTURE = ROOT / "tests" / "fixtures" / "managed_ci" / "dispatch_validator.py"


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _extraction_block(text: str, marker_name: str) -> str:
    marker = text.index(f"# BEGIN {marker_name}")
    start = text.rfind("\n", 0, marker) + 1
    end = text.index(f"# END {marker_name}", start)
    end = text.index("\n", end) + 1
    return textwrap.dedent(text[start:end])


def _validator_block(text: str) -> str:
    return _extraction_block(text, "MANAGED_CI_V2_VALIDATOR")


def _dispatch_block(text: str) -> str:
    return _extraction_block(text, "MANAGED_CI_V2_DISPATCH_VALIDATOR")


def _job_if_expression(text: str) -> str:
    marker = "    if: >-\n"
    start = text.index(marker) + len(marker)
    end = text.index("    name: Python 3.12 full suite", start)
    return textwrap.dedent(text[start:end]).strip()


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
    return local_router.validate(
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
        "run-name: ${{ github.event_name == 'workflow_dispatch' && inputs.managed_nonce != '' && format('managed-ci-v2 nonce={0}', inputs.managed_nonce) || github.workflow }}",
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
        LOCAL_FIXTURE.read_text(encoding="utf-8")
    )


def test_dispatch_validator_fixture_is_an_extraction_of_production_workflow():
    assert _dispatch_block(_workflow_text()) == _dispatch_block(
        DISPATCH_FIXTURE.read_text(encoding="utf-8")
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
    assert _job_if_expression(_workflow_text()) == """github.event_name == 'push' ||
(github.event_name == 'workflow_dispatch' &&
 inputs.protocol_version == '' && inputs.pr_number == '' &&
 inputs.expected_head_sha == '' && inputs.managed_nonce == '') ||
(github.event_name == 'pull_request' &&
 (github.event.action == 'unlabeled' ||
  !(github.event.pull_request.base.ref == 'main' &&
    github.event.pull_request.base.repo.full_name == github.repository &&
    github.event.pull_request.head.repo.full_name == github.repository &&
    startsWith(github.event.pull_request.head.ref, 'agent-loop/managed-') &&
    github.event.pull_request.draft == true &&
    vars.AGENT_LOOP_MANAGED_ACTOR != '' &&
    github.event.pull_request.user.login == vars.AGENT_LOOP_MANAGED_ACTOR &&
    (github.event.action == 'opened' ||
     ((github.event.action == 'synchronize' || github.event.action == 'reopened') &&
      contains(github.event.pull_request.labels.*.name, 'agent-loop-managed'))))))""".strip()
    opening = _pr(labels=[])
    labeled = _pr()
    assert _ordinary_route("opened", opening, "agent-loop") is False
    assert _ordinary_route("synchronize", opening, "agent-loop") is True
    assert _ordinary_route("reopened", labeled, "agent-loop") is False
    assert _ordinary_route("synchronize", labeled, "wrong-actor") is True
    assert _ordinary_route("unlabeled", labeled, "agent-loop") is True
    assert _ordinary_route("synchronize", _pr(head={"sha": "b" * 40, "ref": "feature", "repo": {"full_name": "fork/REPO"}}), "agent-loop") is True


def _dispatch_validate(**overrides):
    values = {
        "protocol": "2",
        "pr_number_text": "7",
        "expected_head": "b" * 40,
        "nonce": "n" * 32,
        "repo": "OWNER/REPO",
        "ref": "refs/heads/main",
        "configured_actor": "agent-loop",
        "initiating_actor": "agent-loop",
        "rerun_actor": "agent-loop",
    }
    values.update(overrides)
    calls = []
    revision = "a" * 40
    records = {
        "users/agent-loop": {"login": "agent-loop", "id": 7},
        "repos/OWNER/REPO": {"full_name": "OWNER/REPO"},
        "repos/OWNER/REPO/pulls/7": _pr(),
        "repos/OWNER/REPO/commits/main": {"sha": revision},
    }

    def api_json(path):
        calls.append(path)
        return records[path]

    def api_pages(path):
        calls.append(path)
        return _pages(_record())

    result = dispatch_validator.validate_dispatch(
        **values,
        api_json=api_json,
        api_pages=api_pages,
        validate=local_router.validate,
    )
    return result, calls


def test_dispatch_validator_resolves_named_actor_and_returns_exact_target():
    result, calls = _dispatch_validate()
    assert result["target_sha"] == "b" * 40
    assert result["record"]["state"] == "dispatch-requested"
    assert calls[:2] == ["users/agent-loop", "repos/OWNER/REPO"]
    assert "user" not in calls
    assert not any("actions/variables" in path for path in calls)


@pytest.mark.parametrize(
    "overrides",
    [
        {"protocol": ""},
        {"pr_number_text": ""},
        {"expected_head": ""},
        {"nonce": ""},
        {"ref": "refs/heads/feature"},
        {"configured_actor": ""},
        {"initiating_actor": "other"},
        {"rerun_actor": "other"},
    ],
)
def test_dispatch_validator_rejects_partial_trust_and_actor_inputs(overrides):
    with pytest.raises(ValueError):
        _dispatch_validate(**overrides)


def test_dispatch_validator_propagates_api_failure_without_authorizing_target():
    values = {
        "protocol": "2",
        "pr_number_text": "7",
        "expected_head": "b" * 40,
        "nonce": "n" * 32,
        "repo": "OWNER/REPO",
        "ref": "refs/heads/main",
        "configured_actor": "agent-loop",
        "initiating_actor": "agent-loop",
        "rerun_actor": "agent-loop",
    }

    def failing_api(_path):
        raise ValueError("GitHub API read failed")

    with pytest.raises(ValueError, match="GitHub API read failed"):
        dispatch_validator.validate_dispatch(
            **values,
            api_json=failing_api,
            api_pages=failing_api,
            validate=local_router.validate,
        )


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
