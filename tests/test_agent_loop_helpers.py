"""The explicit multi-owner patch helpers in agent_loop_helpers (#1186)."""
import types

import pytest

from agent_loop_helpers import patch_attr_in, patch_object_in


def _modules():
    first = types.ModuleType("owner_one")
    second = types.ModuleType("owner_two")
    stray = types.ModuleType("not_an_owner")
    first.target = second.target = "real"
    return first, second, stray


def test_patch_attr_in_sets_every_listed_owner_and_restores_them():
    first, second, _stray = _modules()
    monkeypatch = pytest.MonkeyPatch()
    patch_attr_in(monkeypatch, (first, second), "target", "fake")
    assert (first.target, second.target) == ("fake", "fake")
    monkeypatch.undo()
    assert (first.target, second.target) == ("real", "real")


def test_patch_attr_in_rejects_a_module_that_does_not_bind_the_name(monkeypatch):
    first, second, stray = _modules()
    with pytest.raises(AttributeError, match="not_an_owner does not bind 'target'"):
        patch_attr_in(monkeypatch, (first, stray, second), "target", "fake")
    assert (first.target, second.target) == ("real", "real")
    assert "target" not in vars(stray)


def test_patch_object_in_shares_one_replacement_and_restores_every_owner():
    first, second, _stray = _modules()
    with patch_object_in((first, second), "target", return_value=7) as replacement:
        assert first.target is replacement and second.target is replacement
        assert second.target() == 7
    assert (first.target, second.target) == ("real", "real")
    with patch_object_in((first, second), "target", "explicit") as replacement:
        assert replacement == "explicit" and (first.target, second.target) == ("explicit", "explicit")
    assert (first.target, second.target) == ("real", "real")


def test_patch_object_in_rejects_a_module_that_does_not_bind_the_name():
    first, _second, stray = _modules()
    with pytest.raises(AttributeError, match="not_an_owner does not bind 'target'"):
        patch_object_in((first, stray), "target")
    assert first.target == "real" and "target" not in vars(stray)


@pytest.mark.parametrize("flow", ["plan", "pr"])
@pytest.mark.parametrize("actor", [None, ("loop", 42)])
@pytest.mark.parametrize("identity", ["legacy", "rest-id", "author-id", "foreign", "missing-rest-id", "missing-author-id"])
@pytest.mark.parametrize("author_shape", ["mapping", "string"])
def test_fixture_comment_identity_is_consistent_across_surfaces(tmp_path, flow, actor, identity, author_shape):
    """Legacy publications get the actor; explicit foreign identities stay foreign."""
    from agent_loop_helpers import FakeRunner, make_config
    from coding_review_agent_loop import github
    from coding_review_agent_loop.board_amendment import (
        collect_reviewer_board_amendments, format_reviewer_board_amendment_comment,
    )

    comment = {"author": {"login": "bot"}, "body": format_reviewer_board_amendment_comment(
        flow=flow, issue=56 if flow == "plan" else None,
        pr_number=77 if flow == "pr" else None,
        original_required_reviewers=("Codex", "Gemini"), policy="all-reviewers",
        primary_reviewer=None, removed_reviewers=("Gemini",), effective_from_round=1,
        rationale="Quota exhausted.",
    )}
    expected_actor = actor or ("coding-review-agent-loop", 424242)
    if identity == "rest-id":
        comment["_rest_author_id"] = 81
    elif identity == "author-id":
        comment["author"]["id"] = 81
    elif identity in {"missing-rest-id", "missing-author-id"}:
        comment["author"]["login"] = expected_actor[0]
        if identity == "missing-rest-id":
            comment["_rest_author_id"] = None
        else:
            comment["author"]["id"] = None
    elif identity == "foreign":
        comment["author"]["login"] = "outsider"
    expected = expected_actor if identity == "legacy" else (
        comment["author"]["login"], 81 if identity != "foreign" else 987654
    )
    if author_shape == "string":
        # String logins have no nested ID; carry explicit identity overrides
        # through the REST fixture field, including deliberately absent IDs.
        if "id" in comment["author"]:
            comment["_rest_author_id"] = comment["author"]["id"]
        comment["author"] = comment["author"]["login"]
    runner = FakeRunner(authenticated_actor=actor, issue_comments=[comment],
                        pr_payload={"comments": [comment]})
    config = make_config(tmp_path)
    if identity in {"missing-rest-id", "missing-author-id"}:
        from coding_review_agent_loop.errors import AgentLoopError
        # Both surfaces must fail closed instead of assigning the actor ID.
        with pytest.raises(AgentLoopError):
            if flow == "plan":
                github.get_issue_context(runner, config=config, issue_number=56)
            else:
                github.get_pr_review_context(runner, config=config, pr_number=77)
        assert not runner.comments
        return
    observations = []
    for _ in range(2):
        if flow == "plan":
            comments = github.get_issue_context(runner, config=config, issue_number=56).comments
            target = {"issue_number": 56}
        else:
            comments = github.get_pr_review_context(runner, config=config, pr_number=77).comments
            target = {"pr_number": 77}
        assert len(comments) == 1
        observed = comments[0]
        assert (observed.author, observed.author_id) == expected
        amendments = collect_reviewer_board_amendments(comments, flow=flow, **target)
        assert bool(amendments) == (identity == "legacy")
        observations.append(observed.comment_id)
    assert observations[0] == observations[1]
    assert not runner.comments
