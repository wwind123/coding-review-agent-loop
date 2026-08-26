import base64

import pytest

from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import (
    IssueComment,
    PullRequestMetadata,
    find_open_pr_closing_issue,
    read_pull_request_commit_metadata,
)
from coding_review_agent_loop.issue_pr_provenance import (
    IssuePrProvenanceScope,
    compare_issue_pr_provenance,
    format_issue_pr_provenance,
    parse_issue_pr_provenance_messages,
)
from coding_review_agent_loop.issue_pr_handoff import (
    _decode_issue_pr_handoff_metadata,
    _encode_issue_pr_handoff_metadata,
    _validate_issue_pr_handoff_url,
    IssuePrHandoffMetadata,
    find_latest_issue_pr_handoff,
    format_issue_pr_handoff_comment,
    require_pr_metadata_for_handoff,
)
from agent_loop_helpers import FakeRunner, make_config


def _comment(body: str) -> IssueComment:
    return IssueComment(author="bot", created_at="2026-05-23T00:00:00Z", body=body)


def _metadata(**overrides) -> IssuePrHandoffMetadata:
    defaults = dict(
        schema_version=1,
        issue_number=56,
        pr_number=77,
        pr_url="https://github.com/OWNER/REPO/pull/77",
        pr_head_sha="abc123",
        flow="issue-implementation",
        plan_hash=None,
    )
    defaults.update(overrides)
    return IssuePrHandoffMetadata(**defaults)


def _commit_page(messages, *, head="abc123", total=None, has_next=False, cursor=None):
    nodes = [
        {"commit": {"oid": f"commit-{index}", "message": message}}
        for index, message in enumerate(messages, start=1)
    ]
    return {
        "data": {
            "repository": {
                "pullRequest": {
                    "headRefOid": head,
                    "commits": {
                        "totalCount": len(nodes) if total is None else total,
                        "pageInfo": {"hasNextPage": has_next, "endCursor": cursor},
                        "nodes": nodes,
                    },
                }
            }
        }
    }


def _provenance_pages(message, *, issue=56, flow="direct", plan=None):
    scope = IssuePrProvenanceScope("OWNER/REPO", issue, flow, plan)
    page = _commit_page([f"Change\n\n{format_issue_pr_provenance(scope)}"])
    return [page, page]


def test_round_trips_metadata_through_marker():
    metadata = _metadata()
    encoded = _encode_issue_pr_handoff_metadata(metadata)
    assert _decode_issue_pr_handoff_metadata(encoded) == metadata


def test_round_trips_approved_plan_flow_with_plan_hash():
    metadata = _metadata(flow="approved-plan-implementation", plan_hash="deadbeef01234567")
    encoded = _encode_issue_pr_handoff_metadata(metadata)
    assert _decode_issue_pr_handoff_metadata(encoded) == metadata


def test_round_trips_complete_expected_closing_set_and_supersession_lineage():
    metadata = _metadata(
        expected_closing_issue_ids=(56, 57),
        supersedes_hash="a" * 64,
    )

    encoded = _encode_issue_pr_handoff_metadata(metadata)

    assert _decode_issue_pr_handoff_metadata(encoded) == metadata
    rendered = format_issue_pr_handoff_comment(
        issue_number=56,
        pr_number=77,
        pr_url="https://github.com/OWNER/REPO/pull/77",
        pr_head_sha="abc123",
        flow="issue-implementation",
        plan_hash=None,
        expected_closing_issue_ids=(56, 57),
        supersedes_hash="a" * 64,
    )
    assert "Expected closing issues: #56, #57." in rendered


def test_find_latest_issue_pr_handoff_returns_newest_when_multiple_markers_present():
    older = format_issue_pr_handoff_comment(
        issue_number=56,
        pr_number=77,
        pr_url="https://github.com/OWNER/REPO/pull/77",
        pr_head_sha="oldsha",
        flow="issue-implementation",
        plan_hash=None,
    )
    newer = format_issue_pr_handoff_comment(
        issue_number=56,
        pr_number=90,
        pr_url="https://github.com/OWNER/REPO/pull/90",
        pr_head_sha="newsha",
        flow="issue-implementation",
        plan_hash=None,
    )
    comments = [_comment(older), _comment(newer)]

    found = find_latest_issue_pr_handoff(comments, issue_number=56, repo="OWNER/REPO")

    assert found is not None
    assert found.pr_number == 90
    assert found.pr_head_sha == "newsha"


def test_find_latest_issue_pr_handoff_ignores_records_for_other_issues():
    comment = format_issue_pr_handoff_comment(
        issue_number=99,
        pr_number=77,
        pr_url="https://github.com/OWNER/REPO/pull/77",
        pr_head_sha="abc123",
        flow="issue-implementation",
        plan_hash=None,
    )

    found = find_latest_issue_pr_handoff([_comment(comment)], issue_number=56, repo="OWNER/REPO")

    assert found is None


def test_find_latest_issue_pr_handoff_returns_none_with_no_matching_comments():
    assert find_latest_issue_pr_handoff([_comment("just talk")], issue_number=56, repo="OWNER/REPO") is None


def test_decode_raises_on_malformed_base64_payload():
    with pytest.raises(AgentLoopError, match="Invalid AGENT_ISSUE_PR_HANDOFF payload"):
        _decode_issue_pr_handoff_metadata("!!!not-base64!!!")


def test_decode_raises_on_malformed_json_payload():
    encoded = base64.urlsafe_b64encode(b"not json").decode("ascii")
    with pytest.raises(AgentLoopError, match="Invalid AGENT_ISSUE_PR_HANDOFF payload"):
        _decode_issue_pr_handoff_metadata(encoded)


def test_decode_raises_on_unsupported_schema_version():
    metadata = _metadata(schema_version=2)
    with pytest.raises(AgentLoopError, match="unsupported schema_version"):
        _decode_issue_pr_handoff_metadata(_encode_issue_pr_handoff_metadata(metadata))


def test_decode_raises_on_fractional_schema_version():
    metadata = _metadata(schema_version=1.9)
    with pytest.raises(AgentLoopError, match="`schema_version` must be an integer"):
        _decode_issue_pr_handoff_metadata(_encode_issue_pr_handoff_metadata(metadata))


def test_decode_raises_on_boolean_schema_version():
    # `True` coerces to `1` (== SCHEMA_VERSION) via a naive `int(...)` cast; the decoder
    # must reject bool outright rather than silently accepting it as a valid version.
    metadata = _metadata(schema_version=True)
    with pytest.raises(AgentLoopError, match="`schema_version` must be an integer"):
        _decode_issue_pr_handoff_metadata(_encode_issue_pr_handoff_metadata(metadata))


def test_decode_raises_on_unknown_flow():
    encoded = _encode_issue_pr_handoff_metadata(_metadata())
    payload = base64.urlsafe_b64decode(encoded)
    import json as _json

    data = _json.loads(payload)
    data["flow"] = "something-else"
    bad_encoded = base64.urlsafe_b64encode(_json.dumps(data).encode("utf-8")).decode("ascii")
    with pytest.raises(AgentLoopError, match="unknown flow"):
        _decode_issue_pr_handoff_metadata(bad_encoded)


@pytest.mark.parametrize("field,value", [("issue_number", 0), ("pr_number", -1)])
def test_decode_raises_on_non_positive_identifiers(field, value):
    encoded = _encode_issue_pr_handoff_metadata(_metadata(**{field: value}))
    with pytest.raises(AgentLoopError, match="must be positive"):
        _decode_issue_pr_handoff_metadata(encoded)


@pytest.mark.parametrize("field", ["issue_number", "pr_number"])
def test_decode_raises_on_non_coercible_identifiers(field):
    encoded = _encode_issue_pr_handoff_metadata(_metadata())
    import json as _json

    data = _json.loads(base64.urlsafe_b64decode(encoded))
    data[field] = "not-a-number"
    bad_encoded = base64.urlsafe_b64encode(_json.dumps(data).encode("utf-8")).decode("ascii")
    with pytest.raises(AgentLoopError, match="must be integers"):
        _decode_issue_pr_handoff_metadata(bad_encoded)


@pytest.mark.parametrize("field", ["issue_number", "pr_number"])
def test_decode_raises_on_boolean_identifiers(field):
    encoded = _encode_issue_pr_handoff_metadata(_metadata(**{field: True}))
    with pytest.raises(AgentLoopError, match="must be integers"):
        _decode_issue_pr_handoff_metadata(encoded)


@pytest.mark.parametrize("field", ["issue_number", "pr_number"])
def test_decode_raises_on_fractional_identifiers(field):
    encoded = _encode_issue_pr_handoff_metadata(_metadata(**{field: 56.5}))
    with pytest.raises(AgentLoopError, match="must be integers"):
        _decode_issue_pr_handoff_metadata(encoded)


@pytest.mark.parametrize("pr_url", [None, ""])
def test_decode_raises_on_missing_or_empty_pr_url(pr_url):
    encoded = _encode_issue_pr_handoff_metadata(_metadata(pr_url=pr_url or ""))
    import json as _json

    data = _json.loads(base64.urlsafe_b64decode(encoded))
    data["pr_url"] = pr_url
    bad_encoded = base64.urlsafe_b64encode(_json.dumps(data).encode("utf-8")).decode("ascii")
    with pytest.raises(AgentLoopError, match="`pr_url` must be a non-empty string"):
        _decode_issue_pr_handoff_metadata(bad_encoded)


@pytest.mark.parametrize("pr_head_sha", [None, ""])
def test_decode_raises_on_missing_or_empty_pr_head_sha(pr_head_sha):
    encoded = _encode_issue_pr_handoff_metadata(_metadata())
    import json as _json

    data = _json.loads(base64.urlsafe_b64decode(encoded))
    data["pr_head_sha"] = pr_head_sha
    bad_encoded = base64.urlsafe_b64encode(_json.dumps(data).encode("utf-8")).decode("ascii")
    with pytest.raises(AgentLoopError, match="`pr_head_sha` must be a non-empty string"):
        _decode_issue_pr_handoff_metadata(bad_encoded)


def test_decode_raises_when_plan_hash_present_for_issue_implementation_flow():
    encoded = _encode_issue_pr_handoff_metadata(_metadata())
    import json as _json

    data = _json.loads(base64.urlsafe_b64decode(encoded))
    data["plan_hash"] = "deadbeef01234567"
    bad_encoded = base64.urlsafe_b64encode(_json.dumps(data).encode("utf-8")).decode("ascii")
    with pytest.raises(AgentLoopError, match="`plan_hash` must be absent"):
        _decode_issue_pr_handoff_metadata(bad_encoded)


def test_decode_raises_when_plan_hash_absent_for_approved_plan_flow():
    metadata = _metadata(flow="approved-plan-implementation", plan_hash=None)
    encoded = _encode_issue_pr_handoff_metadata(metadata)
    with pytest.raises(AgentLoopError, match="`plan_hash` is required"):
        _decode_issue_pr_handoff_metadata(encoded)


def test_validate_url_accepts_matching_url():
    _validate_issue_pr_handoff_url(
        "https://github.com/OWNER/REPO/pull/77", repo="OWNER/REPO", pr_number=77
    )


def test_validate_url_rejects_spoofed_host():
    with pytest.raises(AgentLoopError, match="does not match"):
        _validate_issue_pr_handoff_url(
            "https://evil.example.com/OWNER/REPO/pull/77", repo="OWNER/REPO", pr_number=77
        )


def test_validate_url_rejects_prefix_collision():
    with pytest.raises(AgentLoopError, match="does not match"):
        _validate_issue_pr_handoff_url(
            "https://github.com/OWNER/REPO/pull/770", repo="OWNER/REPO", pr_number=77
        )


def test_validate_url_rejects_non_https_scheme():
    with pytest.raises(AgentLoopError, match="does not match"):
        _validate_issue_pr_handoff_url(
            "http://github.com/OWNER/REPO/pull/77", repo="OWNER/REPO", pr_number=77
        )


def test_require_pr_metadata_for_handoff_raises_on_missing_url():
    metadata = PullRequestMetadata(
        number=77,
        repo="OWNER/REPO",
        title="Title",
        head_branch="feature",
        base_branch="main",
        head_sha="abc123",
        url=None,
    )
    with pytest.raises(AgentLoopError, match="PR URL is unavailable"):
        require_pr_metadata_for_handoff(metadata)


def test_require_pr_metadata_for_handoff_raises_on_missing_head_sha():
    metadata = PullRequestMetadata(
        number=77,
        repo="OWNER/REPO",
        title="Title",
        head_branch="feature",
        base_branch="main",
        head_sha=None,
        url="https://github.com/OWNER/REPO/pull/77",
    )
    with pytest.raises(AgentLoopError, match="PR head SHA is unavailable"):
        require_pr_metadata_for_handoff(metadata)


def test_require_pr_metadata_for_handoff_returns_tuple_when_present():
    metadata = PullRequestMetadata(
        number=77,
        repo="OWNER/REPO",
        title="Title",
        head_branch="feature",
        base_branch="main",
        head_sha="abc123",
        url="https://github.com/OWNER/REPO/pull/77",
    )
    assert require_pr_metadata_for_handoff(metadata) == (
        "https://github.com/OWNER/REPO/pull/77",
        "abc123",
    )


def test_provenance_parser_collapses_identical_claims_across_commits():
    scope = IssuePrProvenanceScope("OWNER/REPO", 56, "direct")
    messages = [
        f"first\n\n{format_issue_pr_provenance(scope)}",
        f"second\n\n{format_issue_pr_provenance(scope)}",
    ]

    claims = parse_issue_pr_provenance_messages(messages)

    assert compare_issue_pr_provenance(claims, expected=scope) == scope


def test_legacy_recovery_requires_complete_matching_commit_provenance(tmp_path):
    runner = FakeRunner(
        open_prs_payload=[{"number": 77, "body": "Fixes #56"}],
        pr_commit_pages=_provenance_pages("ignored"),
    )
    config = make_config(tmp_path)

    found = find_open_pr_closing_issue(runner, config=config, issue_number=56)

    assert found is not None
    assert found.pr_number == 77


@pytest.mark.parametrize(
    "pages",
    [
        [_commit_page([]), _commit_page([])],
        [
            _commit_page(
                ["Change\n\nAgent-Issue-Provenance: v1 repo=owner/repo issue=56 flow=direct\n"
                 "Agent-Issue-Provenance: malformed"]
            ),
            _commit_page(["ignored"]),
        ],
        [
            _commit_page(
                [
                    "Change\n\nAgent-Issue-Provenance: v1 repo=owner/repo issue=56 flow=direct",
                    "Other\n\nAgent-Issue-Provenance: v1 repo=owner/repo issue=57 flow=direct",
                ],
                total=2,
            ),
            _commit_page(["ignored"], total=2),
        ],
    ],
)
def test_invalid_single_candidate_recovery_includes_both_safe_remedies(tmp_path, pages):
    runner = FakeRunner(
        open_prs_payload=[{"number": 77, "body": "Fixes #56"}],
        pr_commit_pages=pages,
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError) as exc_info:
        find_open_pr_closing_issue(runner, config=config, issue_number=56)

    message = str(exc_info.value)
    assert "Remove the closing reference" in message
    assert "close the unrelated PR" in message
    assert "agent-loop pr 77" in message


def test_commit_connection_rejects_count_truncation(tmp_path):
    scope = IssuePrProvenanceScope("OWNER/REPO", 56, "direct")
    page = _commit_page([format_issue_pr_provenance(scope)], total=2)
    runner = FakeRunner(pr_commit_pages=[page, page])
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="truncated"):
        read_pull_request_commit_metadata(runner, config=config, pr_number=77)
