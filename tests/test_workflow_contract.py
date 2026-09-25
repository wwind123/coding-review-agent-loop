"""Offline contract tests for the repository's managed-CI workflow."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import textwrap
from pathlib import Path

import pytest

import coding_review_agent_loop.managed_ci as managed_ci
from coding_review_agent_loop.protocol_markers import KNOWN_HOST_COMMENT_FOOTER
from fixtures.managed_ci import dispatch_validator, local_router, publisher


ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
LOCAL_FIXTURE = ROOT / "tests" / "fixtures" / "managed_ci" / "local_router.py"
DISPATCH_FIXTURE = ROOT / "tests" / "fixtures" / "managed_ci" / "dispatch_validator.py"
PUBLISHER_FIXTURE = ROOT / "tests" / "fixtures" / "managed_ci" / "publisher.py"


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


def _publisher_block(text: str) -> str:
    return _extraction_block(text, "MANAGED_CI_V2_PUBLISHER")


def _fixture_source_digest(text: str) -> str:
    match = re.search(r"^Source block SHA-256: ([0-9a-f]{64})$", text, re.MULTILINE)
    assert match is not None
    return match.group(1)


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


def test_publisher_fixture_is_an_extraction_of_production_workflow():
    assert _publisher_block(_workflow_text()) == _publisher_block(
        PUBLISHER_FIXTURE.read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    ("fixture", "marker"),
    [
        (DISPATCH_FIXTURE, "MANAGED_CI_V2_DISPATCH_VALIDATOR"),
        (PUBLISHER_FIXTURE, "MANAGED_CI_V2_PUBLISHER"),
    ],
)
def test_fixture_provenance_digest_contains_the_extracted_workflow_block(fixture, marker):
    fixture_text = fixture.read_text(encoding="utf-8")
    source_digest = _fixture_source_digest(fixture_text)
    fixture_block = _extraction_block(fixture_text, marker)
    workflow_block = _extraction_block(_workflow_text(), marker)
    assert hashlib.sha256(fixture_block.encode("utf-8")).hexdigest() == source_digest
    assert hashlib.sha256(workflow_block.encode("utf-8")).hexdigest() == source_digest


@pytest.mark.parametrize("state", ["dispatch-requested", "attached", "completed"])
def test_validator_accepts_each_dispatch_lifecycle_state(state):
    assert _validate(_record(state))["state"] == state



def test_workflow_advertises_visible_intent_capability():
    assert "AGENT_LOOP_MANAGED_CI_VISIBLE_INTENT_V1: enabled" in _workflow_text()


def _visible_pages(record, *, prefix, actor="agent-loop", actor_id=7):
    return [[{
        "user": {"login": actor, "id": actor_id},
        "body": prefix + "<!-- AGENT_MANAGED_CI_INTENT_V2 "
        + json.dumps(record, separators=(",", ":"))
        + " -->",
    }]]


def test_validator_accepts_fixed_visible_line_bound_to_the_authorized_head():
    record = _record()
    prefix = f"Managed CI authorization for exact head {'b' * 40}.\n\n"

    assert _validate(record, pages=_visible_pages(record, prefix=prefix))["nonce"] == "n" * 32
    # Surrounding whitespace is stripped before the whole-body match.
    assert _validate(record, pages=_visible_pages(record, prefix="\n" + prefix))["state"] == record["state"]


def test_validator_still_accepts_bare_record_during_transition():
    assert _validate(_record())["state"] == "dispatch-requested"


def test_validator_rejects_visible_line_that_disagrees_with_payload_head():
    record = _record()
    prefix = f"Managed CI authorization for exact head {'c' * 40}.\n\n"

    with pytest.raises(ValueError, match="visible intent line disagrees"):
        _validate(record, pages=_visible_pages(record, prefix=prefix))


@pytest.mark.parametrize(
    "prefix",
    [
        "Managed CI authorization for exact head.\n\n",
        f"Managed CI authorization for exact head {'b' * 40}\n\n",
        f"Managed CI authorization for exact head {'b' * 40}.\n",
        f"Managed CI authorization for exact head {'b' * 40}.\n\n\n",
        f"Managed CI authorization for exact head {'B' * 40}.\n\n",
        f"managed CI authorization for exact head {'b' * 40}.\n\n",
        f"Managed CI authorization for exact head {'b' * 40}. Approved!\n\n",
        f"Hello\nManaged CI authorization for exact head {'b' * 40}.\n\n",
        f"Managed CI authorization for exact head {'b' * 40}.\n\n"
        f"Managed CI authorization for exact head {'b' * 40}.\n\n",
        "Ordinary prose ahead of the record.\n\n",
    ],
)
def test_validator_ignores_any_body_beyond_the_fixed_visible_template(prefix):
    record = _record()

    # A non-conforming trusted body is not an intent, so the requested nonce
    # has no authorization at all.
    with pytest.raises(ValueError, match="exactly one fresh"):
        _validate(record, pages=_visible_pages(record, prefix=prefix))


def test_validator_rejects_trailing_content_after_visible_record():
    record = _record()
    pages = _visible_pages(record, prefix=f"Managed CI authorization for exact head {'b' * 40}.\n\n")
    pages[0][0]["body"] += "\n\nsmuggled"

    with pytest.raises(ValueError, match="exactly one fresh"):
        _validate(record, pages=pages)


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


def _dispatch_validate(*, record=None, **overrides):
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
        "current_run_id": "200",
        "current_run_attempt": "1",
        "current_time": 100,
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
        return _pages(record or _record())

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


def test_dispatch_validator_accepts_initial_and_current_run_bound_retry_records():
    initial, _ = _dispatch_validate()
    assert initial["record"]["state"] == "dispatch-requested"

    attached, _ = _dispatch_validate(
        record=_record("attached", run_id=200, run_attempt=1),
    )
    assert attached["record"]["run_id"] == 200

    completed, _ = _dispatch_validate(
        record=_record(
            "completed",
            run_id=200,
            run_attempt=1,
            terminal_run_id=200,
            terminal_run_attempt=1,
            terminal_attempts=[{"run_id": 200, "run_attempt": 1}],
            terminal_outcome="no-status",
        ),
        current_run_attempt="2",
    )
    assert completed["record"]["terminal_outcome"] == "no-status"


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "record": _record(
                "completed",
                run_id=200,
                run_attempt=1,
                terminal_run_id=200,
                terminal_run_attempt=1,
                terminal_attempts=[{"run_id": 200, "run_attempt": 1}],
                terminal_outcome="no-status",
            ),
            "current_run_attempt": "1",
        },
        {
            "record": _record(
                "completed",
                run_id=200,
                run_attempt=1,
                terminal_run_id=200,
                terminal_run_attempt=1,
                terminal_attempts=[{"run_id": 200, "run_attempt": 1}],
                terminal_outcome="no-status",
            ),
            "current_run_attempt": "3",
        },
        {
            "record": _record(
                "completed",
                run_id=200,
                run_attempt=1,
                terminal_run_id=200,
                terminal_run_attempt=1,
                terminal_attempts=[{"run_id": 200, "run_attempt": 1}],
                terminal_outcome="no-status",
            ),
            "current_run_id": "201",
            "current_run_attempt": "2",
        },
        {
            "record": _record(
                "completed",
                run_id=200,
                run_attempt=1,
                terminal_run_id=200,
                terminal_run_attempt=1,
                terminal_attempts=[
                    {"run_id": 200, "run_attempt": 1},
                    {"run_id": 200, "run_attempt": 3},
                ],
                terminal_outcome="no-status",
            ),
            "current_run_attempt": "2",
        },
    ],
)
def test_dispatch_validator_rejects_incoherent_completed_retry_transitions(overrides):
    with pytest.raises(ValueError, match="retry run pair|executing Actions run"):
        _dispatch_validate(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"record": _record("attached", run_id=201, run_attempt=1)},
        {"record": _record("attached", run_id=200, run_attempt=2)},
        {
            "record": _record(
                "completed",
                run_id=200,
                run_attempt=1,
                terminal_run_id=201,
                terminal_run_attempt=1,
                terminal_attempts=[{"run_id": 201, "run_attempt": 1}],
                terminal_outcome="no-status",
            ),
        },
    ],
)
def test_dispatch_validator_rejects_foreign_or_inconsistent_run_pairs(overrides):
    with pytest.raises(ValueError, match="run pair|executing Actions run|terminal run"):
        _dispatch_validate(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"current_run_id": ""},
        {"current_run_attempt": "0"},
        {"current_run_id": "not-a-run"},
    ],
)
def test_dispatch_validator_requires_current_run_identity(overrides):
    with pytest.raises(ValueError, match="run ID|run attempt"):
        _dispatch_validate(**overrides)


def test_dispatch_validator_enforces_bounded_intent_freshness_and_clock_skew():
    window = dispatch_validator.MAX_INTENT_AGE_SECONDS
    skew = dispatch_validator.MAX_INTENT_FUTURE_SKEW_SECONDS

    assert _dispatch_validate(
        record=_record(created_at=100), current_time=100 + window
    )[0]["target_sha"] == "b" * 40
    with pytest.raises(ValueError, match="stale"):
        _dispatch_validate(record=_record(created_at=100), current_time=100 + window + 1)

    assert _dispatch_validate(
        record=_record(created_at=100 + skew), current_time=100
    )[0]["target_sha"] == "b" * 40
    with pytest.raises(ValueError, match="future"):
        _dispatch_validate(record=_record(created_at=100 + skew + 1), current_time=100)


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
        "current_run_id": "200",
        "current_run_attempt": "1",
        "current_time": 100,
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
    assert "'nonce=' + nonce + ';run_id='" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "GH_REF: ${{ github.ref }}" in workflow
    assert "managed dispatch must execute the base workflow from main" in workflow


def _publisher_payload(**overrides):
    values = {
        "target_sha": "b" * 40,
        "validation_result": "success",
        "test_result": "success",
        "nonce": "n" * 32,
        "run_id": "200",
        "run_attempt": "1",
        "server_url": "https://github.com",
        "api_url": "https://api.github.com",
        "repository": "OWNER/REPO",
    }
    values.update(overrides)
    return values


def test_terminal_publisher_decision_is_production_derived_and_safely_correlated():
    success = publisher.build_status_request(**_publisher_payload())
    assert success == {
        "target_sha": "b" * 40,
        "url": "https://api.github.com/repos/OWNER/REPO/statuses/" + "b" * 40,
        "payload": {
            "state": "success",
            "context": "final-ci/exact-head",
            "description": "nonce=" + "n" * 32 + ";run_id=200;attempt=1;result=success",
            "target_url": "https://github.com/OWNER/REPO/actions/runs/200",
        },
    }

    failure = publisher.build_status_request(
        **_publisher_payload(test_result="failure", run_attempt="2")
    )
    assert failure["target_sha"] == "b" * 40
    assert failure["payload"]["state"] == "failure"
    assert "attempt=2" in failure["payload"]["description"]
    assert failure["payload"]["target_url"].endswith("/actions/runs/200")

    assert publisher.build_status_request(
        **_publisher_payload(validation_result="failure")
    ) is None
    assert publisher.build_status_request(**_publisher_payload(target_sha="")) is None


@pytest.mark.parametrize(
    "field, value",
    [
        ("nonce", "short"),
        ("run_id", "0"),
        ("run_attempt", ""),
        ("server_url", ""),
    ],
)
def test_terminal_publisher_does_not_write_for_unbound_correlation(field, value):
    assert publisher.build_status_request(
        **_publisher_payload(**{field: value})
    ) is None


def test_terminal_publisher_request_uses_only_the_validated_target():
    request = publisher.build_status_request(**_publisher_payload())
    assert request["url"].endswith("/statuses/" + request["target_sha"])
    assert request["target_sha"] == "b" * 40
    assert publisher.build_status_request(
        **_publisher_payload(api_url="")
    ) is None


# --- host footer envelope and the agent-side intent mirror (#1043) ----------

FOOTER = KNOWN_HOST_COMMENT_FOOTER
VISIBLE = f"Managed CI authorization for exact head {'b' * 40}.\n\n"


def _record_text(record):
    return "<!-- AGENT_MANAGED_CI_INTENT_V2 " + json.dumps(record, separators=(",", ":")) + " -->"


def _comment(body, *, login="agent-loop", actor_id=7, comment_id=1):
    return {"id": comment_id, "user": {"login": login, "id": actor_id}, "body": body}


def test_workflow_advertises_host_footer_capability():
    assert f"{managed_ci.HOST_FOOTER_INTENT_MARKER}: enabled" in _workflow_text()


def test_workflow_host_footer_literal_is_the_agent_constant():
    block = _validator_block(_workflow_text())
    literal = re.search(r"host_footer = (\'.*\')\n", block)
    assert literal is not None
    assert ast.literal_eval(literal.group(1)) == FOOTER


@pytest.mark.parametrize("prefix", ["", VISIBLE])
def test_validator_accepts_exact_footer_through_the_raw_body_route(prefix):
    record = _record()
    pages = [[_comment(prefix + _record_text(record) + FOOTER)]]

    assert _validate(record, pages=pages)["nonce"] == "n" * 32


@pytest.mark.parametrize(
    "body",
    [
        _record_text(_record()) + FOOTER + FOOTER,
        _record_text(_record()) + FOOTER + "\n",
        _record_text(_record()) + FOOTER + " ",
        "\n" + _record_text(_record()) + FOOTER,
        " " + VISIBLE + _record_text(_record()) + FOOTER,
        _record_text(_record()) + "\n\n---\n_Generated by [Claude Code](http://claude.ai/code)_",
        _record_text(_record()) + "\n---\n_Generated by [Claude Code](https://claude.ai/code)_",
        FOOTER.lstrip("\n") + "\n\n" + _record_text(_record()),
        _record_text(_record()) + FOOTER + "\n\nsmuggled",
    ],
)
def test_validator_skips_doubled_whitespace_and_variant_footers(body):
    # Neither route matches, so the comment is not an intent at all and the
    # requested nonce has no authorization.
    with pytest.raises(ValueError, match="exactly one fresh"):
        _validate(_record(), pages=[[_comment(body)]])


def test_pinned_older_routers_skip_a_footered_intent():
    """Why agent-loop guards older base workflows against the footer."""
    from fixtures.managed_ci import current_router, historical_router

    record = _record()
    pages = [[_comment(_record_text(record) + FOOTER)]]
    for router in (current_router, historical_router):
        with pytest.raises(ValueError):
            router.validate(_pr(), pages, "OWNER/REPO", "7", "b" * 40, "n" * 32, "agent-loop", "a" * 40)


def test_agent_freshness_constants_mirror_the_dispatch_validator():
    block = _dispatch_block(_workflow_text())
    for name, value in (
        ("MAX_INTENT_AGE_SECONDS", managed_ci.INTENT_MAX_AGE_SECONDS),
        ("MAX_INTENT_FUTURE_SKEW_SECONDS", managed_ci.INTENT_MAX_FUTURE_SKEW_SECONDS),
    ):
        declared = re.search(rf"^{name} = (.+)$", block, re.MULTILINE)
        assert declared is not None
        assert eval(declared.group(1), {}) == value  # noqa: S307 - literal arithmetic
        assert getattr(dispatch_validator, name) == value
    assert 0 < managed_ci.DISPATCH_START_MARGIN_SECONDS < managed_ci.INTENT_MAX_AGE_SECONDS


def _classify(page, nonce="n" * 32):
    return managed_ci.classify_intent_page(
        page,
        requested_nonce=nonce,
        trusted_login="agent-loop",
        trusted_id=7,
        visible_capable=True,
        host_footer_capable=True,
        binding=managed_ci.IntentBinding(
            repository="OWNER/REPO", pr=7, expected_head_sha="b" * 40, base_ref="main",
            workflow_revision="a" * 40, nonce=nonce,
        ),
    )


def _router(page, nonce="n" * 32):
    try:
        return local_router.validate(
            _pr(), [page], "OWNER/REPO", "7", "b" * 40, nonce, "agent-loop", "a" * 40, 7
        ), None
    except ValueError as exc:
        return None, str(exc)


def _assert_parity(page, nonce="n" * 32):
    accepted, reason = _router(page, nonce)
    verdict = _classify(page, nonce)
    if accepted is not None:
        assert verdict.outcome == "authorized", verdict
        assert verdict.record == accepted
    elif reason == "prepared intent is not a dispatch authorization":
        assert verdict.outcome == "recoverable", verdict
    else:
        assert verdict.outcome == "fail", (reason, verdict)
    fatal = managed_ci.intent_page_pre_nonce_fatal(
        page, trusted_login="agent-loop", trusted_id=7,
        visible_capable=True, host_footer_capable=True,
    )
    if fatal is not None:
        # A nonce-independent failure also fails a freshly minted nonce.
        assert _router(page, "f" * 32)[0] is None
        assert _classify(page, "f" * 32).outcome == "fail"
    return verdict


def _single(record=None, *, prefix="", suffix="", raw=None):
    body = raw if raw is not None else prefix + _record_text(record or _record()) + suffix
    return [_comment(body)]


_PARITY_PAGES = {
    "bare": _single(),
    "visible": _single(prefix=VISIBLE),
    "bare-footer": _single(suffix=FOOTER),
    "visible-footer": _single(prefix=VISIBLE, suffix=FOOTER),
    "doubled-footer": _single(suffix=FOOTER + FOOTER),
    "footer-newline": _single(suffix=FOOTER + "\n"),
    "footer-space": _single(suffix=FOOTER + " "),
    "leading-whitespace-footer": _single(prefix="\n", suffix=FOOTER),
    "http-footer": _single(suffix="\n\n---\n_Generated by [Claude Code](http://claude.ai/code)_"),
    "missing-blank-line": _single(suffix="\n---\n_Generated by [Claude Code](https://claude.ai/code)_"),
    "footer-before-marker": _single(prefix=FOOTER.lstrip("\n") + "\n\n"),
    "visible-head-mismatch": _single(prefix=f"Managed CI authorization for exact head {'c' * 40}.\n\n"),
    "visible-head-mismatch-footer": _single(
        prefix=f"Managed CI authorization for exact head {'c' * 40}.\n\n", suffix=FOOTER
    ),
    "short-nonce": _single(_record(nonce="n" * 31)),
    "extra-key": _single(_record(extra=True)),
    "missing-key": _single({k: v for k, v in _record().items() if k != "terminal_attempts"}),
    "early-run-fields": _single(_record(run_id=100, run_attempt=1)),
    "attached-without-attempt": _single(_record("attached", run_attempt=None)),
    "invalid-terminal-outcome": _single(_record(terminal_outcome="failure")),
    "malformed-terminal-attempts": _single(_record(terminal_attempts=[{"run_id": 1}])),
    "base-ref-drift": _single(_record(base_ref="release")),
    "revision-drift": _single(_record(workflow_revision="c" * 40)),
    "malformed-json": _single(raw="<!-- AGENT_MANAGED_CI_INTENT_V2 {not json} -->"),
    "non-object": _single(raw="<!-- AGENT_MANAGED_CI_INTENT_V2 [1, 2] -->"),
    "wrong-version": _single(_record(version=3)),
    "prepared": _single(_record("prepared")),
    "hours-old": _single(_record(created_at=1)),
    "duplicate-same-nonce": [
        _comment(_record_text(_record()), comment_id=1),
        _comment(_record_text(_record()), comment_id=2),
    ],
    "valid-plus-malformed-same-nonce": [
        _comment(_record_text(_record()), comment_id=1),
        _comment(_record_text(_record(extra=True)), comment_id=2),
    ],
    "valid-plus-other-nonce": [
        _comment(_record_text(_record()), comment_id=1),
        _comment(_record_text(_record(nonce="o" * 32)), comment_id=2),
    ],
    "malformed-json-beside-valid": [
        _comment("<!-- AGENT_MANAGED_CI_INTENT_V2 {not json} -->", comment_id=1),
        _comment(_record_text(_record()), comment_id=2),
    ],
    "non-object-beside-valid": [
        _comment("<!-- AGENT_MANAGED_CI_INTENT_V2 [1] -->", comment_id=1),
        _comment(_record_text(_record()), comment_id=2),
    ],
    "wrong-version-beside-valid": [
        _comment(_record_text(_record(version=1, nonce="o" * 32)), comment_id=1),
        _comment(_record_text(_record()), comment_id=2),
    ],
    "trusted-actor-id-drift": [_comment(_record_text(_record()), actor_id=8)],
    "trusted-body-none": [_comment(None)],
    "trusted-body-not-text": [_comment(["list"])],
    "author-missing": [{"id": 1, "body": _record_text(_record())}],
    "author-not-dict": [{"id": 1, "user": "agent-loop", "body": "hi"}],
    "author-login-not-text": [{"id": 1, "user": {"login": 7, "id": 7}, "body": "hi"}],
    "foreign-author-malformed-login": [
        {"id": 1, "user": {"id": 99}, "body": "hi"},
        _comment(_record_text(_record()), comment_id=2),
    ],
    "non-dict-entry": ["not a comment", _comment(_record_text(_record()))],
    "null-entry": [None, _comment(_record_text(_record()))],
    "foreign-author-ignored": [
        _comment(_record_text(_record()), login="someone", actor_id=99, comment_id=1),
        _comment(_record_text(_record()), comment_id=2),
    ],
}


@pytest.mark.parametrize("name", sorted(_PARITY_PAGES))
def test_agent_intent_classifier_matches_the_workflow_router(name):
    _assert_parity(_PARITY_PAGES[name])


def test_parity_corpus_exercises_every_outcome():
    outcomes = {_classify(page).outcome for page in _PARITY_PAGES.values()}
    assert outcomes == {"fail", "recoverable", "authorized"}
    assert _classify(_PARITY_PAGES["bare-footer"]).outcome == "authorized"
    assert _classify(_PARITY_PAGES["hours-old"]).outcome == "authorized"
    assert _classify(_PARITY_PAGES["prepared"]).outcome == "recoverable"
    assert _classify(_PARITY_PAGES["doubled-footer"]).outcome == "fail"
    assert _classify(_PARITY_PAGES["valid-plus-other-nonce"]).outcome == "authorized"


def test_agent_envelope_admits_footer_only_when_the_workflow_advertises_it():
    body = _record_text(_record()) + FOOTER
    assert managed_ci.match_intent_envelope(body, visible_capable=True, host_footer_capable=True)
    assert managed_ci.match_intent_envelope(body, visible_capable=True, host_footer_capable=False) is None
    visible = VISIBLE + _record_text(_record())
    assert managed_ci.match_intent_envelope(visible, visible_capable=False, host_footer_capable=True) is None
