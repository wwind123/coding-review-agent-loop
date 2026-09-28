"""Trusted-author verification for signed human requirements (#1022)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from coding_review_agent_loop.board_amendment import (
    collect_reviewer_board_amendments,
    format_reviewer_board_amendment_comment,
)
from coding_review_agent_loop.cli import build_parser
from coding_review_agent_loop.config import parse_human_reviewer_trusted_actors
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import (
    HumanReviewRequirement,
    TrustedHumanActor,
    get_issue_context,
    get_pr_review_context,
    human_requirement_id,
)
from coding_review_agent_loop.prompts import render_coder_human_requirements_prompt_context
from coding_review_agent_loop.protocol import (
    HumanRequirementDisposition,
    validate_human_requirement_dispositions,
    validate_structured_human_requirements_acknowledgement,
)
from coding_review_agent_loop.runner import CommandResult

from agent_loop_helpers import make_config


REPO = "OWNER/REPO"
SIGNATURE = "\n\n-- Human Reviewer"
PR = 77
ISSUE = 42
TRUSTED = (TrustedHumanActor(login="maintainer", user_id=101),)


def _pr_comment_url(comment_id: int) -> str:
    return f"https://github.com/{REPO}/pull/{PR}#issuecomment-{comment_id}"


def _issue_comment_url(comment_id: int) -> str:
    return f"https://github.com/{REPO}/issues/{ISSUE}#issuecomment-{comment_id}"


class RoutedRunner:
    """Minimal runner that serves `gh pr/issue view` and REST reads."""

    def __init__(self, *, projection: dict, rest: dict[str, object] | None = None):
        self.projection = projection
        self.rest = rest or {}
        self.commands: list[list[str]] = []

    def run(self, cmd, cwd=None, check=True, **_kwargs):
        cmd = list(cmd)
        self.commands.append(cmd)
        if cmd[1] in {"pr", "issue"} and cmd[2] == "view":
            return CommandResult(cmd, Path("."), json.dumps(self.projection), "", 0)
        if cmd[1] == "api":
            path = cmd[2]
            for prefix, payload in self.rest.items():
                if path == prefix or path.startswith(prefix + "?"):
                    if isinstance(payload, Exception):
                        return CommandResult(cmd, Path("."), "", "boom", 1)
                    if callable(payload):
                        payload = payload(path)
                    return CommandResult(cmd, Path("."), json.dumps(payload), "", 0)
            raise AssertionError(f"unexpected REST read: {path}")
        raise AssertionError(f"unexpected command: {cmd}")

    @property
    def rest_reads(self) -> list[str]:
        return [cmd[2] for cmd in self.commands if cmd[1] == "api"]


def _pr_projection(*, comments=(), reviews=()):
    return {
        "number": PR,
        "state": "OPEN",
        "url": f"https://github.com/{REPO}/pull/{PR}",
        "title": "t",
        "headRefName": "feature",
        "baseRefName": "main",
        "headRefOid": "abc123",
        "body": "",
        "comments": list(comments),
        "reviews": list(reviews),
    }


def _gql_comment(comment_id, login, body, *, url=None, created="2026-09-01T00:00:00Z"):
    return {
        "id": f"IC_{comment_id}",
        "author": {"login": login},
        "createdAt": created,
        "url": url if url is not None else _pr_comment_url(comment_id),
        "body": body,
    }


def _rest_comment(comment_id, login, user_id, body, *, url=None, created="2026-09-01T00:00:00Z"):
    return {
        "id": comment_id,
        "user": {"login": login, "id": user_id},
        "created_at": created,
        "html_url": url if url is not None else _pr_comment_url(comment_id),
        "body": body,
    }


def _config(tmp_path, trusted=TRUSTED, **overrides):
    return make_config(tmp_path, human_reviewer_trusted_actors=tuple(trusted), **overrides)


def _comments_path(number=PR):
    return f"repos/{REPO}/issues/{number}/comments"


def _logged(capsys) -> str:
    return capsys.readouterr().err


def _assert_response_omitting_excluded_id_validates(requirements):
    """A coder ledger naming only the admitted requirements validates."""
    prompt = render_coder_human_requirements_prompt_context(requirements)
    admitted_ids = [r.requirement_id for r in requirements]
    assert list(prompt.surfaced_requirement_ids) == admitted_ids
    dispositions = [
        HumanRequirementDisposition(requirement_id=item, disposition="addressed", evidence="Done.")
        for item in admitted_ids
    ]
    validate_structured_human_requirements_acknowledgement(
        admitted_ids,
        dispositions=dispositions,
        checked_discussion_directly=False,
        surfaced_requirement_ids=prompt.surfaced_requirement_ids,
        requires_direct_discussion_ack=prompt.requires_direct_discussion_ack,
    )
    validate_human_requirement_dispositions(
        dispositions, surfaced_requirement_ids=prompt.surfaced_requirement_ids
    )


# --- unconfigured parity ---------------------------------------------------


def test_unconfigured_trust_set_keeps_every_requirement_unverified_without_rest_reads(tmp_path):
    comments = [
        _gql_comment(1, "maintainer", "Use absolute URLs." + SIGNATURE),
        _gql_comment(2, "agent-bot", "Also add a test." + SIGNATURE),
    ]
    reviews = [{"id": "PRR_9", "author": {"login": "other"}, "body": "Keep it small." + SIGNATURE,
                "submittedAt": "2026-09-02T00:00:00Z"}]
    runner = RoutedRunner(projection=_pr_projection(comments=comments, reviews=reviews))
    config = make_config(tmp_path)
    assert config.human_reviewer_trusted_actors == ()

    context = get_pr_review_context(runner, config=config, pr_number=PR)

    assert runner.rest_reads == []
    assert len(context.human_requirements) == 3
    assert {r.author_verification for r in context.human_requirements} == {"unverified"}
    assert all(r.author_id is None for r in context.human_requirements)
    legacy = [
        HumanReviewRequirement(
            source_type=r.source_type, author=r.author, created_at=r.created_at, url=r.url, body=r.body
        )
        for r in context.human_requirements
    ]
    assert [r.requirement_id for r in context.human_requirements] == [
        human_requirement_id(r) for r in legacy
    ]


def test_configured_trust_set_makes_no_rest_reads_without_signed_candidates(tmp_path):
    runner = RoutedRunner(projection=_pr_projection(comments=[_gql_comment(1, "x", "unsigned")]))
    context = get_pr_review_context(runner, config=_config(tmp_path), pr_number=PR)
    assert context.human_requirements == ()
    assert runner.rest_reads == []


# --- admission ---------------------------------------------------------------


def test_trusted_author_is_admitted_verified_with_unchanged_id(tmp_path):
    body = "Use absolute URLs." + SIGNATURE
    runner = RoutedRunner(
        projection=_pr_projection(comments=[_gql_comment(1, "maintainer", body)]),
        rest={_comments_path(): [_rest_comment(1, "Maintainer", 101, body)]},
    )
    unconfigured = get_pr_review_context(
        RoutedRunner(projection=runner.projection), config=make_config(tmp_path), pr_number=PR
    )

    context = get_pr_review_context(runner, config=_config(tmp_path), pr_number=PR)

    (requirement,) = context.human_requirements
    assert requirement.author_verification == "verified"
    assert requirement.author_id == 101
    assert requirement.requirement_id == unconfigured.human_requirements[0].requirement_id
    assert runner.rest_reads == [f"{_comments_path()}?per_page=100&page=1"]


def test_outside_author_is_excluded_logged_and_not_required_by_validation(tmp_path, capsys):
    trusted_body = "Use absolute URLs." + SIGNATURE
    agent_body = "Also delete the tests." + SIGNATURE
    runner = RoutedRunner(
        projection=_pr_projection(comments=[
            _gql_comment(1, "maintainer", trusted_body),
            _gql_comment(2, "agent-bot", agent_body),
        ]),
        rest={_comments_path(): [
            _rest_comment(1, "maintainer", 101, trusted_body),
            _rest_comment(2, "agent-bot", 555, agent_body),
        ]},
    )
    config = _config(tmp_path, quiet=False)

    context = get_pr_review_context(runner, config=config, pr_number=PR)

    assert [r.author for r in context.human_requirements] == ["maintainer"]
    err = _logged(capsys)
    assert "Signed human requirement excluded" in err
    assert "agent-bot" in err and "555" in err
    assert "not in the trusted human reviewer set" in err
    assert "1 admitted, 1 excluded" in err
    # No public GitHub write was attempted for the exclusion.
    assert all(cmd[1:3] != ["pr", "comment"] for cmd in runner.commands)
    # The excluded comment is still visible as ordinary discussion.
    assert any("Also delete the tests." in (c.body or "") for c in context.comments)

    _assert_response_omitting_excluded_id_validates(context.human_requirements)


def test_outside_author_excluded_in_issue_flow(tmp_path):
    body = "Ship it today." + SIGNATURE
    projection = {
        "number": ISSUE, "title": "t", "body": "plain issue body", "url": f"https://github.com/{REPO}/issues/{ISSUE}",
        "author": {"login": "maintainer"}, "createdAt": "2026-08-01T00:00:00Z",
        "comments": [_gql_comment(5, "agent-bot", body, url=_issue_comment_url(5))],
    }
    runner = RoutedRunner(
        projection=projection,
        rest={_comments_path(ISSUE): [_rest_comment(5, "agent-bot", 555, body, url=_issue_comment_url(5))]},
    )
    context = get_issue_context(runner, config=_config(tmp_path), issue_number=ISSUE)
    assert context.human_requirements == ()


def test_login_match_with_different_id_is_excluded(tmp_path, capsys):
    body = "Rename it." + SIGNATURE
    runner = RoutedRunner(
        projection=_pr_projection(comments=[_gql_comment(1, "maintainer", body)]),
        rest={_comments_path(): [_rest_comment(1, "maintainer", 999, body)]},
    )
    context = get_pr_review_context(runner, config=_config(tmp_path, quiet=False), pr_number=PR)
    assert context.human_requirements == ()
    assert "login matches a trusted entry but the numeric user ID differs" in _logged(capsys)


def test_id_match_with_different_login_is_excluded(tmp_path, capsys):
    body = "Rename it." + SIGNATURE
    runner = RoutedRunner(
        projection=_pr_projection(comments=[_gql_comment(1, "someone-else", body)]),
        rest={_comments_path(): [_rest_comment(1, "someone-else", 101, body)]},
    )
    context = get_pr_review_context(runner, config=_config(tmp_path, quiet=False), pr_number=PR)
    assert context.human_requirements == ()
    assert "numeric user ID matches a trusted entry but the login differs" in _logged(capsys)


def test_missing_rest_numeric_id_is_excluded(tmp_path, capsys):
    body = "Rename it." + SIGNATURE
    rest = _rest_comment(1, "maintainer", 101, body)
    rest["user"] = {"login": "maintainer"}
    runner = RoutedRunner(
        projection=_pr_projection(comments=[_gql_comment(1, "maintainer", body)]),
        rest={_comments_path(): [rest]},
    )
    context = get_pr_review_context(runner, config=_config(tmp_path, quiet=False), pr_number=PR)
    assert context.human_requirements == ()
    assert "no numeric author ID" in _logged(capsys)


def test_pr_reviews_are_verified_by_node_id(tmp_path):
    trusted_body = "Keep the API stable." + SIGNATURE
    outside_body = "Drop the API." + SIGNATURE
    reviews = [
        {"id": "PRR_1", "author": {"login": "maintainer"}, "body": trusted_body,
         "submittedAt": "2026-09-02T00:00:00Z"},
        {"id": "PRR_2", "author": {"login": "outsider"}, "body": outside_body,
         "submittedAt": "2026-09-03T00:00:00Z"},
    ]
    rest_reviews = [
        {"id": 1, "node_id": "PRR_1", "user": {"login": "maintainer", "id": 101}, "body": trusted_body,
         "html_url": f"https://github.com/{REPO}/pull/{PR}#pullrequestreview-1"},
        {"id": 2, "node_id": "PRR_2", "user": {"login": "outsider", "id": 7}, "body": outside_body,
         "html_url": f"https://github.com/{REPO}/pull/{PR}#pullrequestreview-2"},
    ]
    runner = RoutedRunner(
        projection=_pr_projection(reviews=reviews),
        rest={f"repos/{REPO}/pulls/{PR}/reviews": rest_reviews},
    )
    context = get_pr_review_context(runner, config=_config(tmp_path), pr_number=PR)
    assert [(r.source_type, r.author, r.author_verification) for r in context.human_requirements] == [
        ("PR review", "maintainer", "verified")
    ]
    assert runner.rest_reads == [f"repos/{REPO}/pulls/{PR}/reviews?per_page=100&page=1"]


def _issue_projection(*, author="maintainer", body="Do the thing." + SIGNATURE, comments=()):
    return {
        "number": ISSUE, "title": "t", "body": body,
        "url": f"https://github.com/{REPO}/issues/{ISSUE}",
        "author": {"login": author}, "createdAt": "2026-08-01T00:00:00Z",
        "comments": list(comments),
    }


def _rest_issue(*, login="maintainer", user_id=101, body="Do the thing." + SIGNATURE):
    return {
        "number": ISSUE, "node_id": "I_1", "user": {"login": login, "id": user_id}, "body": body,
        "html_url": f"https://github.com/{REPO}/issues/{ISSUE}",
    }


def test_issue_body_is_verified_through_issue_rest_record(tmp_path):
    runner = RoutedRunner(
        projection=_issue_projection(),
        rest={f"repos/{REPO}/issues/{ISSUE}": _rest_issue()},
    )
    context = get_issue_context(runner, config=_config(tmp_path), issue_number=ISSUE)
    (requirement,) = context.human_requirements
    assert requirement.source_type == "Issue body"
    assert requirement.author_verification == "verified"
    assert requirement.author_id == 101


def test_issue_body_from_outside_author_is_excluded(tmp_path):
    runner = RoutedRunner(
        projection=_issue_projection(author="outsider"),
        rest={f"repos/{REPO}/issues/{ISSUE}": _rest_issue(login="outsider", user_id=7)},
    )
    context = get_issue_context(runner, config=_config(tmp_path), issue_number=ISSUE)
    assert context.human_requirements == ()


# --- fail-closed reads -------------------------------------------------------


def test_failed_rest_read_fails_closed(tmp_path):
    body = "Rename it." + SIGNATURE
    runner = RoutedRunner(
        projection=_pr_projection(comments=[_gql_comment(1, "maintainer", body)]),
        rest={_comments_path(): RuntimeError("fail")},
    )
    with pytest.raises(AgentLoopError, match="cannot be verified against the configured trusted"):
        get_pr_review_context(runner, config=_config(tmp_path), pr_number=PR)


def test_failed_review_read_fails_closed(tmp_path):
    reviews = [{"id": "PRR_1", "author": {"login": "maintainer"}, "body": "x" + SIGNATURE}]
    runner = RoutedRunner(
        projection=_pr_projection(reviews=reviews),
        rest={f"repos/{REPO}/pulls/{PR}/reviews": RuntimeError("fail")},
    )
    with pytest.raises(AgentLoopError, match="review read is incomplete"):
        get_pr_review_context(runner, config=_config(tmp_path), pr_number=PR)


def test_incomplete_review_page_fails_closed(tmp_path):
    reviews = [{"id": "PRR_1", "author": {"login": "maintainer"}, "body": "x" + SIGNATURE}]
    runner = RoutedRunner(
        projection=_pr_projection(reviews=reviews),
        rest={f"repos/{REPO}/pulls/{PR}/reviews": {"message": "not a list"}},
    )
    with pytest.raises(AgentLoopError, match="non-list page"):
        get_pr_review_context(runner, config=_config(tmp_path), pr_number=PR)


# --- locator join ------------------------------------------------------------


def test_candidate_without_locator_is_never_cross_attributed(tmp_path, capsys):
    body = "Same words." + SIGNATURE
    comments = [
        _gql_comment(1, "maintainer", body, url=""),
        _gql_comment(2, "outsider", body, url=""),
    ]
    runner = RoutedRunner(
        projection=_pr_projection(comments=comments),
        rest={_comments_path(): [
            _rest_comment(1, "maintainer", 101, body),
            _rest_comment(2, "outsider", 7, body),
        ]},
    )
    context = get_pr_review_context(runner, config=_config(tmp_path, quiet=False), pr_number=PR)
    assert context.human_requirements == ()
    assert "no verifiable record locator" in _logged(capsys)


def test_candidate_url_absent_from_rest_is_excluded(tmp_path, capsys):
    body = "Same words." + SIGNATURE
    runner = RoutedRunner(
        projection=_pr_projection(comments=[_gql_comment(1, "maintainer", body)]),
        rest={_comments_path(): [_rest_comment(3, "maintainer", 101, body)]},
    )
    context = get_pr_review_context(runner, config=_config(tmp_path, quiet=False), pr_number=PR)
    assert context.human_requirements == ()
    assert "absent from the complete REST read" in _logged(capsys)


def test_duplicate_rest_locator_fails_closed(tmp_path):
    body = "Same words." + SIGNATURE
    runner = RoutedRunner(
        projection=_pr_projection(comments=[_gql_comment(1, "maintainer", body)]),
        rest={_comments_path(): [
            _rest_comment(1, "maintainer", 101, body),
            _rest_comment(2, "outsider", 7, body, url=_pr_comment_url(1)),
        ]},
    )
    with pytest.raises(AgentLoopError, match="sharing one locator"):
        get_pr_review_context(runner, config=_config(tmp_path), pr_number=PR)


# --- content drift -----------------------------------------------------------


@pytest.mark.parametrize(
    "rest_login, rest_body",
    [
        ("maintainer", "Different words." + SIGNATURE),
        ("maintainer", "Use absolute URLs."),
        ("impostor", "Use absolute URLs." + SIGNATURE),
    ],
    ids=["body-changed", "signature-removed", "author-differs"],
)
def test_issue_comment_drift_fails_closed(tmp_path, rest_login, rest_body):
    body = "Use absolute URLs." + SIGNATURE
    runner = RoutedRunner(
        projection=_issue_projection(body="plain", comments=[
            _gql_comment(5, "maintainer", body, url=_issue_comment_url(5))
        ]),
        rest={_comments_path(ISSUE): [
            _rest_comment(5, rest_login, 101, rest_body, url=_issue_comment_url(5))
        ]},
    )
    with pytest.raises(AgentLoopError, match="changed during author verification"):
        get_issue_context(runner, config=_config(tmp_path), issue_number=ISSUE)


@pytest.mark.parametrize(
    "rest_kwargs",
    [
        {"body": "Do another thing." + SIGNATURE},
        {"body": "Do the thing."},
        {"login": "impostor"},
    ],
    ids=["body-changed", "signature-removed", "author-differs"],
)
def test_issue_body_drift_fails_closed(tmp_path, rest_kwargs):
    runner = RoutedRunner(
        projection=_issue_projection(),
        rest={f"repos/{REPO}/issues/{ISSUE}": _rest_issue(**rest_kwargs)},
    )
    with pytest.raises(AgentLoopError, match="changed during author verification"):
        get_issue_context(runner, config=_config(tmp_path), issue_number=ISSUE)


@pytest.mark.parametrize(
    "rest_login, rest_body",
    [
        ("maintainer", "Other." + SIGNATURE),
        ("maintainer", "Keep the API stable."),
        ("impostor", "Keep the API stable." + SIGNATURE),
    ],
    ids=["body-changed", "signature-removed", "author-differs"],
)
def test_pr_review_drift_fails_closed(tmp_path, rest_login, rest_body):
    reviews = [{"id": "PRR_1", "author": {"login": "maintainer"},
                "body": "Keep the API stable." + SIGNATURE}]
    runner = RoutedRunner(
        projection=_pr_projection(reviews=reviews),
        rest={f"repos/{REPO}/pulls/{PR}/reviews": [
            {"id": 1, "node_id": "PRR_1", "user": {"login": rest_login, "id": 101}, "body": rest_body}
        ]},
    )
    with pytest.raises(AgentLoopError, match="changed during author verification"):
        get_pr_review_context(runner, config=_config(tmp_path), pr_number=PR)


def test_bot_login_spellings_are_one_identity(tmp_path):
    body = "Relayed decision." + SIGNATURE
    runner = RoutedRunner(
        projection=_pr_projection(comments=[_gql_comment(1, "relay-app", body)]),
        rest={_comments_path(): [_rest_comment(1, "relay-app[bot]", 300, body)]},
    )
    config = _config(tmp_path, trusted=(TrustedHumanActor(login="relay-app[bot]", user_id=300),))
    (requirement,) = get_pr_review_context(runner, config=config, pr_number=PR).human_requirements
    assert requirement.author_verification == "verified"


# --- config ------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, message",
    [
        ("alice", "LOGIN:ID"),
        ("alice:0", "positive numeric"),
        ("alice:-3", "positive numeric"),
        ("alice:x", "positive numeric"),
        (":5", "blank or malformed login"),
        ("alice:", "positive numeric"),
    ],
)
def test_malformed_trust_entries_are_rejected(value, message):
    with pytest.raises(AgentLoopError, match=message):
        parse_human_reviewer_trusted_actors([value])


def test_conflicting_logins_for_one_id_are_rejected():
    with pytest.raises(AgentLoopError, match="conflicting logins"):
        parse_human_reviewer_trusted_actors(["alice:5", "bob:5"])


def test_trust_entries_parse_in_order_and_empty_is_unconfigured():
    assert parse_human_reviewer_trusted_actors(None) == ()
    assert parse_human_reviewer_trusted_actors(["alice:5", "Bob:6", "ALICE:5"]) == (
        TrustedHumanActor("alice", 5),
        TrustedHumanActor("Bob", 6),
    )


@pytest.mark.parametrize("command", [["issue", "12"], ["pr", "7"]])
def test_cli_accepts_repeatable_trust_option(command):
    args = build_parser().parse_args([
        *command,
        "--human-reviewer-trusted-actor", "alice:5",
        "--human-reviewer-trusted-actor=bob:6",
    ])
    assert args.human_reviewer_trusted_actor == ["alice:5", "bob:6"]
    assert parse_human_reviewer_trusted_actors(args.human_reviewer_trusted_actor) == (
        TrustedHumanActor("alice", 5),
        TrustedHumanActor("bob", 6),
    )


def test_config_from_args_rejects_malformed_entry_before_orchestration(tmp_path, monkeypatch):
    from coding_review_agent_loop import config as config_module

    args = build_parser().parse_args(["pr", "7", "--repo", REPO, "--human-reviewer-trusted-actor", "alice"])
    with pytest.raises(AgentLoopError, match="LOGIN:ID"):
        config_module.parse_human_reviewer_trusted_actors(args.human_reviewer_trusted_actor)


# --- other signed records ----------------------------------------------------


def test_signed_board_amendment_is_unchanged_and_triggers_no_verification(tmp_path):
    amendment = format_reviewer_board_amendment_comment(
        flow="plan",
        issue=ISSUE,
        pr_number=None,
        original_required_reviewers=("Codex", "Claude", "Antigravity"),
        policy="primary-then-panel",
        primary_reviewer="Codex",
        removed_reviewers=("Antigravity",),
        effective_from_round=3,
        rationale="Quota exhausted.",
    )
    runner = RoutedRunner(
        projection=_issue_projection(body="plain", comments=[
            _gql_comment(9, "outsider", amendment, url=_issue_comment_url(9))
        ]),
    )
    context = get_issue_context(runner, config=_config(tmp_path), issue_number=ISSUE)
    assert context.human_requirements == ()
    assert runner.rest_reads == []
    amendments = collect_reviewer_board_amendments(
        [SimpleNamespace(body=amendment)], flow="plan", issue_number=ISSUE
    )
    assert len(amendments) == 1
