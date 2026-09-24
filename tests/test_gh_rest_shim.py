"""Tests for the agent-loop-gh REST shim and GraphQL-refusal fallbacks (#1029)."""

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import coding_review_agent_loop.gh_rest_shim as shim
import coding_review_agent_loop.github_transport as transport
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import (
    IssueComment,
    get_issue_context,
    get_pr_mergeability,
    get_pr_review_context,
    human_requirement_id,
    read_pull_request_commit_metadata,
    strip_bot_login_suffix,
)
from coding_review_agent_loop.round_state import (
    encode_plan_validation_diagnostic_body,
    recover_plan_validation_diagnostic,
)
from coding_review_agent_loop.runner import Runner

from agent_loop_helpers import make_config
from test_round_transport import _diagnostic_payload

REPO = "OWNER/REPO"
REFUSAL = (
    "HTTP 403: GitHub GraphQL is not available from Claude Code sessions; use the REST API "
    "(gh api repos/{owner}/{repo}/...)."
)


def completed(argv, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(argv, returncode, stdout, stderr)


class FakeApi:
    """Routes `gh api PATH [-X M] [--input -] [-H ...]` to canned responses."""

    def __init__(self, routes=None):
        self.routes = dict(routes or {})
        self.calls = []

    def __call__(self, argv, input=None, capture_output=True, text=True, check=False):
        assert argv[1] == "api", argv
        path = argv[2]
        method = argv[argv.index("-X") + 1] if "-X" in argv else "GET"
        accept = argv[argv.index("-H") + 1] if "-H" in argv else None
        payload = json.loads(input) if input else None
        self.calls.append((method, path, payload, accept))
        key = (method, path)
        if key not in self.routes:
            return completed(argv, 1, json.dumps({"message": "Not Found"}), "gh: Not Found (HTTP 404)")
        value = self.routes[key]
        if callable(value):
            value = value(payload)
        if isinstance(value, tuple):
            code, body = value
            return completed(argv, code, body if isinstance(body, str) else json.dumps(body), "gh: error")
        return completed(argv, 0, value if isinstance(value, str) else json.dumps(value))

    def paths(self, method="GET"):
        return [path for m, path, _payload, _accept in self.calls if m == method]


def run_shim(argv, api, *, env=None, stdin="", probe=None, git=None, tmp_path=None):
    out, err = [], []
    values = {"AGENT_LOOP_GH_TRANSPORT": "rest", "AGENT_LOOP_REAL_GH": "/usr/bin/real-gh"}
    if tmp_path is not None:
        values["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    values.update(env or {})
    passthrough_calls = []

    def passthrough(real_gh, args):
        passthrough_calls.append((real_gh, list(args)))
        return 0

    code = shim.run(
        argv,
        env=values,
        passthrough=passthrough,
        api_run=api,
        probe=probe,
        out=out.append,
        err=err.append,
        stdin=lambda: stdin,
        git=git,
        own=Path("/opt/agent-loop-gh"),
    )
    return SimpleNamespace(code=code, out="".join(out), err="".join(err), passthrough=passthrough_calls)


def page(path):
    return f"{path}{'&' if '?' in path else '?'}per_page=100&page=1"


def user(login, *, uid=1, bot=False):
    return {"login": login, "id": uid, "node_id": f"U_{uid}", "type": "Bot" if bot else "User"}


def rest_comment(cid, body, *, login="alice", uid=1, bot=False, created="2026-09-01T00:00:00Z"):
    return {
        "id": cid,
        "node_id": f"IC_{cid}",
        "user": user(login, uid=uid, bot=bot),
        "author_association": "OWNER",
        "body": body,
        "created_at": created,
        "html_url": f"https://github.com/{REPO}/issues/1#issuecomment-{cid}",
    }


def rest_pr(number=5, **overrides):
    raw = {
        "id": 900 + number,
        "number": number,
        "title": "Title",
        "body": "Closes #1",
        "html_url": f"https://github.com/{REPO}/pull/{number}",
        "state": "open",
        "merged": False,
        "draft": False,
        "user": user("alice"),
        "created_at": "2026-09-01T00:00:00Z",
        "labels": [],
        "head": {"ref": "feature", "sha": "a" * 40},
        "base": {"ref": "main"},
        "mergeable": True,
        "mergeable_state": "clean",
        "commits": 2,
    }
    raw.update(overrides)
    return raw


# ------------------------------------------------------------ dispatch ---


def test_api_calls_always_pass_through_even_in_rest_mode():
    api = FakeApi()
    result = run_shim(["api", "graphql", "-f", "query={viewer{login}}"], api)
    assert result.passthrough == [("/usr/bin/real-gh", ["api", "graphql", "-f", "query={viewer{login}}"])]
    result = run_shim(["api", f"repos/{REPO}/issues/1", "--jq", '{n:.number,is_pr:has("pull_request")}'], api)
    assert result.passthrough and not api.calls


def test_unsupported_commands_pass_through_without_probing():
    probed = []
    result = run_shim(
        ["auth", "status"], FakeApi(), env={"AGENT_LOOP_GH_TRANSPORT": "auto"}, probe=probed.append
    )
    assert result.passthrough == [("/usr/bin/real-gh", ["auth", "status"])]
    assert probed == []


def test_graphql_mode_passes_supported_commands_through():
    result = run_shim(["pr", "view", "5", "--repo", REPO, "--json", "number"], FakeApi(),
                      env={"AGENT_LOOP_GH_TRANSPORT": "graphql"})
    assert result.passthrough


def test_unsupported_flag_or_field_fails_with_rest_hint():
    api = FakeApi({("GET", f"repos/{REPO}/pulls/5"): rest_pr()})
    result = run_shim(["pr", "view", "5", "--repo", REPO, "--json", "statusCheckRollup"], api)
    assert result.code == shim.EXIT_USAGE
    assert "not supported in REST mode" in result.err and "gh api repos/" in result.err
    result = run_shim(["pr", "view", "5", "--repo", REPO, "--json", "number", "--web"], api)
    assert result.code == shim.EXIT_USAGE


def test_invalid_transport_value_is_a_usage_error():
    result = run_shim(["pr", "view", "5", "--json", "number"], FakeApi(),
                      env={"AGENT_LOOP_GH_TRANSPORT": "sometimes"})
    assert result.code == shim.EXIT_USAGE
    assert "AGENT_LOOP_GH_TRANSPORT" in result.err


# ----------------------------------------------------------- transport ---


def probe_result(code, stderr=""):
    return lambda argv: completed(argv, code, "", stderr)


def test_auto_probe_selects_rest_only_for_the_exact_refusal(tmp_path):
    env = {"AGENT_LOOP_GH_TRANSPORT": "auto", "XDG_CACHE_HOME": str(tmp_path)}
    assert transport.select_transport("/gh", env=env, probe=probe_result(1, REFUSAL)) == "rest"
    env2 = {"AGENT_LOOP_GH_TRANSPORT": "auto", "XDG_CACHE_HOME": str(tmp_path / "b")}
    assert transport.select_transport("/gh", env=env2, probe=probe_result(0)) == "graphql"
    env3 = {"AGENT_LOOP_GH_TRANSPORT": "auto", "XDG_CACHE_HOME": str(tmp_path / "c")}
    assert transport.select_transport("/gh", env=env3, probe=probe_result(1, "HTTP 502 Bad Gateway")) == "graphql"


def test_probe_positive_outcomes_are_cached_but_transient_failures_are_not(tmp_path):
    env = {"AGENT_LOOP_GH_TRANSPORT": "auto", "XDG_CACHE_HOME": str(tmp_path)}
    calls = []

    def probe(argv):
        calls.append(argv)
        return completed(argv, 1, "", "connection reset")

    assert transport.select_transport("/gh", env=env, probe=probe, now=1000) == "graphql"
    assert transport.select_transport("/gh", env=env, probe=probe, now=1001) == "graphql"
    assert len(calls) == 2  # transient failure never cached

    assert transport.select_transport("/gh", env=env, probe=probe_result(1, REFUSAL), now=1000) == "rest"
    assert transport.select_transport("/gh", env=env, probe=probe, now=1000 + 60) == "rest"
    assert len(calls) == 2  # cache hit, no probe
    # expired entry re-probes
    assert transport.select_transport(
        "/gh", env=env, probe=probe, now=1000 + transport.PROBE_CACHE_TTL_SECONDS + 1
    ) == "graphql"
    assert len(calls) == 3


def test_probe_cache_key_changes_with_proxy_and_real_gh(tmp_path):
    env = {"AGENT_LOOP_GH_TRANSPORT": "auto", "XDG_CACHE_HOME": str(tmp_path), "HTTPS_PROXY": "http://a"}
    assert transport.select_transport("/gh", env=env, probe=probe_result(1, REFUSAL), now=10) == "rest"
    changed = dict(env, HTTPS_PROXY="http://b")
    assert transport.select_transport("/gh", env=changed, probe=probe_result(0), now=11) == "graphql"
    assert transport.select_transport("/other-gh", env=env, probe=probe_result(0), now=11) == "graphql"
    assert transport.select_transport("/gh", env=env, probe=probe_result(0), now=11) == "rest"


def test_corrupt_or_unwritable_cache_is_ignored(tmp_path):
    cache_dir = tmp_path / "agent-loop-gh"
    cache_dir.mkdir()
    (cache_dir / "transport.json").write_text("{not json", encoding="utf-8")
    env = {"AGENT_LOOP_GH_TRANSPORT": "auto", "XDG_CACHE_HOME": str(tmp_path)}
    assert transport.select_transport("/gh", env=env, probe=probe_result(1, REFUSAL)) == "rest"
    blocked = tmp_path / "file-not-dir"
    blocked.write_text("x", encoding="utf-8")
    env = {"AGENT_LOOP_GH_TRANSPORT": "auto", "XDG_CACHE_HOME": str(blocked)}
    assert transport.select_transport("/gh", env=env, probe=probe_result(1, REFUSAL)) == "rest"


def test_rest_mode_never_probes():
    def probe(argv):
        raise AssertionError("rest mode must not probe")

    assert transport.select_transport("/gh", env={"AGENT_LOOP_GH_TRANSPORT": "rest"}, probe=probe) == "rest"


def test_real_gh_discovery_skips_the_shim_and_guards_recursion(tmp_path):
    own = tmp_path / "shim" / "agent-loop-gh"
    own.parent.mkdir()
    own.write_text("#!/bin/sh\n", encoding="utf-8")
    own.chmod(0o755)
    link_dir = tmp_path / "bin"
    link_dir.mkdir()
    (link_dir / "gh").symlink_to(own)
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    real = real_dir / "gh"
    real.write_text("#!/bin/sh\n", encoding="utf-8")
    real.chmod(0o755)
    path = os.pathsep.join([str(link_dir), str(real_dir)])
    assert shim.find_real_gh({"PATH": path}, own=own.resolve()) == str(real)
    with pytest.raises(shim.ShimError) as excinfo:
        shim.find_real_gh({"PATH": str(link_dir)}, own=own.resolve())
    assert excinfo.value.code == shim.EXIT_NO_GH
    with pytest.raises(shim.ShimError):
        shim.find_real_gh({"AGENT_LOOP_REAL_GH": str(link_dir / "gh")}, own=own.resolve())


# ------------------------------------------------------------ repo ---


def test_repo_view_projection_and_jq():
    api = FakeApi({("GET", f"repos/{REPO}"): {"full_name": REPO, "name": "REPO", "default_branch": "trunk"}})
    result = run_shim(["repo", "view", REPO, "--json", "defaultBranchRef", "--jq", ".defaultBranchRef.name"], api)
    assert result.code == 0 and result.out == "trunk\n"
    result = run_shim(["repo", "view", "--repo", REPO, "--json", "nameWithOwner", "--jq", ".nameWithOwner"], api)
    assert result.out == f"{REPO}\n"


def test_jq_beyond_dotted_paths_is_refused():
    api = FakeApi({("GET", f"repos/{REPO}"): {"full_name": REPO}})
    result = run_shim(["repo", "view", REPO, "--json", "nameWithOwner", "--jq", ".nameWithOwner | ascii_downcase"], api)
    assert result.code == shim.EXIT_USAGE


def test_repo_resolution_from_origin_remote_and_gh_repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "remote", "add", "origin", "git@github.com:acme/widget.git"], check=True)
    assert shim.resolve_repo(None, {}, tmp_path) == "acme/widget"
    assert shim.resolve_repo(None, {"GH_REPO": "o/r"}, tmp_path) == "o/r"
    assert shim.resolve_repo("https://github.com/o/r.git", {}) == "o/r"
    with pytest.raises(shim.UsageError):
        shim.resolve_repo("o/r", {"GH_HOST": "ghe.example.com"})


def test_repo_clone_uses_git_and_adds_upstream_for_forks():
    api = FakeApi({("GET", f"repos/{REPO}"): {"fork": True, "parent": {"full_name": "UP/REPO"}}})
    git_calls = []

    def git(argv, **kwargs):
        git_calls.append(argv)
        return completed(argv)

    result = run_shim(["repo", "clone", REPO, "/tmp/x", "--", "--depth", "1"], api, git=git)
    assert result.code == 0
    assert git_calls[0] == ["git", "clone", f"https://github.com/{REPO}.git", "/tmp/x", "--depth", "1"]
    assert git_calls[1][-2:] == ["upstream", "https://github.com/UP/REPO.git"]


# ------------------------------------------------------------ issue view ---


def issue_routes(comments, *, number=1, body="Issue body", issue_user=None):
    return {
        ("GET", f"repos/{REPO}/issues/{number}"): {
            "id": 1, "number": number, "title": "Issue", "body": body, "state": "open",
            "html_url": f"https://github.com/{REPO}/issues/{number}", "user": issue_user or user("alice"),
            "created_at": "2026-08-01T00:00:00Z", "labels": [{"name": "bug"}],
        },
        ("GET", page(f"repos/{REPO}/issues/{number}/comments")): comments,
    }


def test_issue_view_matches_gh_graphql_shape():
    comments = [rest_comment(11, "hi"), rest_comment(12, "bot says", login="helper[bot]", uid=2, bot=True)]
    api = FakeApi(issue_routes(comments))
    result = run_shim(
        ["issue", "view", "1", "--repo", REPO, "--comments", "--json", "number,title,body,url,author,createdAt,comments,state"],
        api,
    )
    data = json.loads(result.out)
    assert data["state"] == "OPEN"
    assert data["url"] == f"https://github.com/{REPO}/issues/1"
    assert data["author"]["login"] == "alice"
    assert data["comments"][0] == {
        "id": "IC_11", "author": {"login": "alice"}, "authorAssociation": "OWNER", "body": "hi",
        "createdAt": "2026-09-01T00:00:00Z", "url": f"https://github.com/{REPO}/issues/1#issuecomment-11",
    }
    # GraphQL spells app logins without `[bot]`.
    assert data["comments"][1]["author"] == {"login": "helper"}


def test_bot_suffix_is_only_stripped_for_bot_accounts():
    assert shim.graphql_login(user("helper[bot]", bot=True)) == "helper"
    assert shim.graphql_login(user("helper[bot]", bot=False)) == "helper[bot]"
    assert shim.graphql_login(user("helper", bot=True)) == "helper"


def test_comments_are_read_to_completion_across_pages():
    first = [rest_comment(i, f"c{i}") for i in range(1, 101)]
    second = [rest_comment(101, "last")]
    routes = issue_routes(first)
    routes[("GET", f"repos/{REPO}/issues/1/comments?per_page=100&page=2")] = second
    result = run_shim(["issue", "view", "1", "--repo", REPO, "--json", "comments"], FakeApi(routes))
    assert len(json.loads(result.out)["comments"]) == 101


@pytest.mark.parametrize(
    "second_page",
    [
        (1, {"message": "Server Error"}),  # failure after a valid page
        [rest_comment(5, "dup")],  # repeated identity
        [{"id": 777}, "not-an-object"],  # malformed record
        {"not": "a list"},
    ],
)
def test_incomplete_or_malformed_pages_fail_instead_of_truncating(second_page):
    routes = issue_routes([rest_comment(i, f"c{i}") for i in range(1, 101)])
    routes[("GET", f"repos/{REPO}/issues/1/comments?per_page=100&page=2")] = second_page
    result = run_shim(["issue", "view", "1", "--repo", REPO, "--json", "comments"], FakeApi(routes))
    assert result.code != 0
    assert result.out == ""


def test_comment_without_numeric_id_fails():
    routes = issue_routes([{"id": None, "node_id": "IC_x", "body": "x", "user": user("a")}])
    result = run_shim(["issue", "view", "1", "--repo", REPO, "--json", "comments"], FakeApi(routes))
    assert result.code != 0


# ------------------------------------------------------------ pr view ---


@pytest.mark.parametrize(
    ("raw", "state", "mergeable", "merge_state"),
    [
        ({}, "OPEN", "MERGEABLE", "CLEAN"),
        ({"merged": True, "state": "closed"}, "MERGED", "MERGEABLE", "CLEAN"),
        ({"state": "closed"}, "CLOSED", "MERGEABLE", "CLEAN"),
        ({"mergeable": False, "mergeable_state": "dirty"}, "OPEN", "CONFLICTING", "DIRTY"),
        ({"mergeable": None, "mergeable_state": "unknown"}, "OPEN", "UNKNOWN", "UNKNOWN"),
        ({"mergeable": None, "mergeable_state": None}, "OPEN", "UNKNOWN", "UNKNOWN"),
        ({"mergeable_state": "blocked"}, "OPEN", "MERGEABLE", "BLOCKED"),
    ],
)
def test_pr_state_and_mergeability_mapping(raw, state, mergeable, merge_state):
    api = FakeApi({("GET", f"repos/{REPO}/pulls/5"): rest_pr(**raw)})
    result = run_shim(
        ["pr", "view", "5", "--repo", REPO, "--json", "state,mergeable,mergeStateStatus,headRefOid,baseRefName"], api
    )
    data = json.loads(result.out)
    assert (data["state"], data["mergeable"], data["mergeStateStatus"]) == (state, mergeable, merge_state)
    assert data["headRefOid"] == "a" * 40 and data["baseRefName"] == "main"


def test_pr_view_fetches_comments_and_reviews_only_when_requested():
    api = FakeApi({("GET", f"repos/{REPO}/pulls/5"): rest_pr()})
    run_shim(["pr", "view", "5", "--repo", REPO, "--json", "number,headRefOid"], api)
    assert api.paths() == [f"repos/{REPO}/pulls/5"]


def test_review_projection_has_no_url_and_uses_graphql_login():
    reviews = [{
        "id": 31, "node_id": "PRR_31", "user": user("ci[bot]", uid=9, bot=True), "body": "LGTM",
        "state": "APPROVED", "submitted_at": "2026-09-02T00:00:00Z", "commit_id": "b" * 40,
        "html_url": "https://example/review", "author_association": "NONE",
    }]
    api = FakeApi({
        ("GET", f"repos/{REPO}/pulls/5"): rest_pr(),
        ("GET", page(f"repos/{REPO}/pulls/5/reviews")): reviews,
    })
    result = run_shim(["pr", "view", "5", "--repo", REPO, "--json", "reviews"], api)
    review = json.loads(result.out)["reviews"][0]
    assert "url" not in review
    assert review["author"] == {"login": "ci"}
    assert review["submittedAt"] == "2026-09-02T00:00:00Z"


def test_pr_list_applies_limit_and_state():
    prs = [rest_pr(n) for n in range(1, 4)]
    api = FakeApi({("GET", f"repos/{REPO}/pulls?state=open&per_page=100&page=1"): prs})
    result = run_shim(["pr", "list", "--repo", REPO, "--state", "open", "--json", "number,body", "--limit", "2"], api)
    assert [row["number"] for row in json.loads(result.out)] == [1, 2]


# ------------------------------------------------------------ search ---


def rest_issue(number, title, body="", *, state="open", pr=False):
    raw = {"id": 5000 + number, "number": number, "title": title, "body": body, "state": state,
           "html_url": f"https://github.com/{REPO}/issues/{number}", "user": user("alice")}
    if pr:
        raw["pull_request"] = {"url": "x"}
    return raw


def search(query, issues, *, state="all", limit="100000"):
    api = FakeApi({("GET", f"repos/{REPO}/issues?state={state}&per_page=100&page=1"): issues})
    result = run_shim(
        ["issue", "list", "--repo", REPO, "--search", query, "--state", state, "--limit", limit,
         "--json", "number,title,url,body"],
        api,
    )
    return result, ([row["number"] for row in json.loads(result.out)] if result.code == 0 else None)


def test_parent_child_title_queries_match_token_sequences():
    issues = [
        rest_issue(20, "Stage 1 (from #12)"),
        rest_issue(21, "Stage 2 (from #123)"),
        rest_issue(22, "[#12 stage] Two"),
        rest_issue(23, "Unrelated", body="(from #12) in body only"),
        rest_issue(24, "PR (from #12)", pr=True),
    ]
    assert search('"(from #12)" in:title', issues)[1] == [20]
    assert search('"[#12 stage]" in:title', issues)[1] == [22]


def test_escaped_title_phrase_matches():
    issues = [rest_issue(30, 'Say "hello" world'), rest_issue(31, "Say hello")]
    assert search('"Say \\"hello\\" world" in:title', issues)[1] == [30]


def test_followup_queries_support_their_qualifiers_and_filter_before_limit():
    noise = [rest_issue(100 + n, "Noise", body=f"a follow-up somewhere (item {chr(65 + n % 26)})") for n in range(40)]
    tracker = rest_issue(99, "Tracker", body="Future follow-up for #7 cache eviction")
    closed = rest_issue(98, "Old", body="future follow-up #7", state="closed")
    issues = noise + [tracker, closed]
    query = f'repo:{REPO} is:issue is:open "#7" "follow-up"'
    result, found = search(query, issues, state="open", limit="20")
    assert result.code == 0
    assert found == [99]
    query = f'repo:{REPO} is:issue is:open "future follow-up" "cache eviction"'
    assert search(query, issues, state="open", limit="20")[1] == [99]


def test_search_rejects_foreign_repo_and_unknown_qualifiers():
    assert search('repo:other/repo "x"', [])[0].code == shim.EXIT_USAGE
    assert search('label:bug "x"', [])[0].code == shim.EXIT_USAGE
    assert search('"unterminated', [])[0].code == shim.EXIT_USAGE


# ------------------------------------------------------------ writes ---


def test_comment_and_issue_create_accept_body_or_body_file(tmp_path):
    body_file = tmp_path / "body.md"
    body_file.write_text("from `file`\n", encoding="utf-8")
    api = FakeApi({
        ("POST", f"repos/{REPO}/issues/3/comments"): {"html_url": "https://c/1"},
        ("POST", f"repos/{REPO}/issues"): {"html_url": f"https://github.com/{REPO}/issues/77", "number": 77},
    })
    assert run_shim(["pr", "comment", "3", "--repo", REPO, "--body-file", str(body_file)], api).out == "https://c/1\n"
    assert run_shim(["issue", "comment", "3", "--repo", REPO, "--body", "inline"], api).code == 0
    created = run_shim(["issue", "create", "--repo", REPO, "--title", "T", "--body", "B"], api)
    assert created.out.endswith("/issues/77\n")
    posted = [payload for method, _path, payload, _accept in api.calls if method == "POST"]
    assert posted == [{"body": "from `file`\n"}, {"body": "inline"}, {"title": "T", "body": "B"}]
    assert run_shim(["issue", "comment", "3", "--repo", REPO, "--body", "a", "--body-file", str(body_file)], api).code == shim.EXIT_USAGE


def pr_create_routes(*, existing=None, label_ok=True, label_post=None):
    routes = {
        ("GET", f"repos/{REPO}"): {"default_branch": "main"},
        ("GET", f"repos/{REPO}/branches/feature"): {"name": "feature"},
        ("GET", f"repos/{REPO}/pulls?state=open&head=OWNER%3Afeature&base=main&per_page=100&page=1"): existing or [],
        ("POST", f"repos/{REPO}/pulls"): {"number": 44, "html_url": f"https://github.com/{REPO}/pull/44"},
    }
    if label_ok:
        routes[("GET", f"repos/{REPO}/labels/agent-loop-managed")] = {"name": "agent-loop-managed"}
    routes[("POST", f"repos/{REPO}/issues/44/labels")] = label_post if label_post is not None else [{"name": "agent-loop-managed"}]
    return routes


def pr_create_argv(tmp_path, *extra):
    body = tmp_path / "pr.md"
    body.write_text("Closes #1\n", encoding="utf-8")
    return ["pr", "create", "--repo", REPO, "--head", "feature", "--title", "T", "--body-file", str(body), *extra]


def test_pr_create_draft_with_label(tmp_path):
    api = FakeApi(pr_create_routes())
    result = run_shim(pr_create_argv(tmp_path, "--draft", "--label", "agent-loop-managed"), api)
    assert result.code == 0 and result.out.endswith("/pull/44\n")
    create = next(p for m, path, p, _a in api.calls if m == "POST" and path.endswith("/pulls"))
    assert create == {"title": "T", "head": "feature", "base": "main", "body": "Closes #1\n", "draft": True}


def test_pr_create_refuses_when_same_head_and_base_pr_exists(tmp_path):
    existing = [{"id": 1, "number": 40, "html_url": f"https://github.com/{REPO}/pull/40"}]
    api = FakeApi(pr_create_routes(existing=existing))
    result = run_shim(pr_create_argv(tmp_path), api)
    assert result.code == 1
    assert "already exists" in result.err and result.out.endswith("/pull/40\n")
    assert not api.paths("POST")


def test_pr_create_allows_same_head_into_a_different_base(tmp_path):
    routes = pr_create_routes()
    routes[("GET", f"repos/{REPO}/pulls?state=open&head=OWNER%3Afeature&base=release&per_page=100&page=1")] = []
    api = FakeApi(routes)
    result = run_shim(pr_create_argv(tmp_path, "--base", "release"), api)
    assert result.code == 0


def test_pr_create_preflights_missing_label_and_unpushed_branch(tmp_path):
    api = FakeApi(pr_create_routes(label_ok=False))
    result = run_shim(pr_create_argv(tmp_path, "--label", "agent-loop-managed"), api)
    assert result.code == 1 and "does not exist" in result.err
    assert not api.paths("POST")
    routes = pr_create_routes()
    del routes[("GET", f"repos/{REPO}/branches/feature")]
    result = run_shim(pr_create_argv(tmp_path), FakeApi(routes))
    assert result.code == 1 and "push it first" in result.err


def test_pr_create_label_failure_reports_created_pr_and_retry_does_not_duplicate(tmp_path):
    routes = pr_create_routes(label_post=(1, {"message": "Validation Failed"}))
    api = FakeApi(routes)
    result = run_shim(pr_create_argv(tmp_path, "--label", "agent-loop-managed"), api)
    assert result.code == 1
    assert result.out.endswith("/pull/44\n")
    assert "do not create another PR" in result.err
    routes[("GET", f"repos/{REPO}/pulls?state=open&head=OWNER%3Afeature&base=main&per_page=100&page=1")] = [
        {"id": 1, "number": 44, "html_url": f"https://github.com/{REPO}/pull/44"}
    ]
    retry = FakeApi(routes)
    again = run_shim(pr_create_argv(tmp_path, "--label", "agent-loop-managed"), retry)
    assert again.code == 1 and "already exists" in again.err
    assert f"repos/{REPO}/pulls" not in retry.paths("POST")


def test_pr_create_requires_title_and_rejects_fill(tmp_path):
    api = FakeApi(pr_create_routes())
    assert run_shim(["pr", "create", "--repo", REPO, "--head", "feature", "--body", "b"], api).code == shim.EXIT_USAGE
    assert run_shim(["pr", "create", "--repo", REPO, "--fill"], api).code == shim.EXIT_USAGE


@pytest.mark.parametrize(("argv", "route", "draft_after"), [
    (["pr", "ready", "5"], "ready_for_review", False),
    (["pr", "ready", "--undo", "5"], "convert_to_draft", True),
])
def test_pr_ready_uses_ccr_routes_and_verifies_draft_state(argv, route, draft_after):
    api = FakeApi({
        ("POST", f"repos/{REPO}/pulls/5/ccr/{route}"): {},
        ("GET", f"repos/{REPO}/pulls/5"): rest_pr(draft=draft_after),
    })
    assert run_shim([*argv, "--repo", REPO], api).code == 0
    wrong = FakeApi({
        ("POST", f"repos/{REPO}/pulls/5/ccr/{route}"): {},
        ("GET", f"repos/{REPO}/pulls/5"): rest_pr(draft=not draft_after),
    })
    assert run_shim([*argv, "--repo", REPO], wrong).code == 1


def test_pr_merge_requires_merged_true_and_passes_head_sha():
    api = FakeApi({("PUT", f"repos/{REPO}/pulls/5/merge"): {"merged": True, "sha": "c" * 40}})
    result = run_shim(["pr", "merge", "5", "--repo", REPO, "--merge", "--match-head-commit", "a" * 40], api)
    assert result.code == 0
    assert api.calls[0][2] == {"merge_method": "merge", "sha": "a" * 40}
    refused = FakeApi({("PUT", f"repos/{REPO}/pulls/5/merge"): {"merged": False, "message": "nope"}})
    assert run_shim(["pr", "merge", "5", "--repo", REPO, "--merge"], refused).code == 1
    moved = FakeApi({("PUT", f"repos/{REPO}/pulls/5/merge"): (1, {"message": "Head branch was modified"})})
    result = run_shim(["pr", "merge", "5", "--repo", REPO, "--merge", "--match-head-commit", "a" * 40], moved)
    assert result.code == 1 and "Head branch was modified" in result.err
    assert run_shim(["pr", "merge", "5", "--repo", REPO, "--squash"], api).code == shim.EXIT_USAGE


def test_pr_diff_uses_diff_media_type():
    api = FakeApi({("GET", f"repos/{REPO}/pulls/5"): "diff --git a/x b/x\n"})
    result = run_shim(["pr", "diff", "5", "--repo", REPO], api)
    assert result.out == "diff --git a/x b/x\n"
    assert api.calls[0][3] == f"Accept: {shim.DIFF_MEDIA_TYPE}"


@pytest.mark.parametrize(("runs", "statuses", "code"), [
    ([{"id": 1, "name": "ci", "status": "completed", "conclusion": "success"}], [], 0),
    ([{"id": 1, "name": "ci", "status": "in_progress", "conclusion": None}], [], shim.EXIT_PENDING),
    ([{"id": 1, "name": "ci", "status": "completed", "conclusion": "failure"}], [{"context": "ext", "state": "pending"}], 1),
])
def test_pr_checks_exit_codes(runs, statuses, code):
    sha = "a" * 40
    api = FakeApi({
        ("GET", f"repos/{REPO}/pulls/5"): rest_pr(),
        ("GET", f"repos/{REPO}/commits/{sha}/check-runs?per_page=100&page=1"): {"total_count": len(runs), "check_runs": runs},
        ("GET", f"repos/{REPO}/commits/{sha}/status"): {"statuses": statuses, "total_count": len(statuses), "state": "success"},
        ("GET", f"repos/{REPO}/commits/{sha}/status?per_page=100&page=1"): {"statuses": [dict(s, id=i) for i, s in enumerate(statuses)]},
    })
    result = run_shim(["pr", "checks", "5", "--repo", REPO], api)
    assert result.code == code
    assert result.out.startswith("ci\t")


# ----------------------------------------- end to end through real parsers ---


FAKE_GH = r'''#!{python}
import json, os, sys
routes = json.load(open(os.environ["FAKE_GH_ROUTES"]))
args = sys.argv[1:]
assert args[0] == "api", args
path = args[1]
method = args[args.index("-X") + 1] if "-X" in args else "GET"
value = routes.get(method + " " + path)
if value is None:
    sys.stdout.write(json.dumps({{"message": "Not Found"}}))
    sys.stderr.write("gh: Not Found (HTTP 404)\n")
    sys.exit(1)
sys.stdout.write(value if isinstance(value, str) else json.dumps(value))
'''


@pytest.fixture
def shim_config(tmp_path, monkeypatch):
    """A real Runner driving the real shim, whose real `gh` serves REST fixtures."""
    fake = tmp_path / "real-gh"
    fake.write_text(FAKE_GH.format(python=sys.executable), encoding="utf-8")
    fake.chmod(0o755)
    wrapper = tmp_path / "agent-loop-gh"
    wrapper.write_text(
        f"#!/bin/sh\nexec {sys.executable} -m coding_review_agent_loop.gh_rest_shim \"$@\"\n", encoding="utf-8"
    )
    wrapper.chmod(0o755)
    routes_file = tmp_path / "routes.json"
    monkeypatch.setenv("AGENT_LOOP_REAL_GH", str(fake))
    monkeypatch.setenv("AGENT_LOOP_GH_TRANSPORT", "rest")
    monkeypatch.setenv("FAKE_GH_ROUTES", str(routes_file))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(sys.path))
    config = make_config(tmp_path, gh_cmd=str(wrapper))

    def serve(routes):
        routes_file.write_text(json.dumps({f"{m} {p}": v for (m, p), v in routes.items()}), encoding="utf-8")

    return config, serve


def _gql_comment(cid, body, login, created):
    return {"id": f"IC_{cid}", "author": {"login": login}, "body": body, "createdAt": created,
            "url": f"https://github.com/{REPO}/issues/1#issuecomment-{cid}"}


def test_bot_authored_diagnostic_recovers_through_shim_and_parsers(shim_config):
    config, serve = shim_config
    payload = replace(_diagnostic_payload(), repository=REPO, issue_number=1,
                      expected_producer_login="agent-app[bot]", expected_producer_id=77)
    body = str(encode_plan_validation_diagnostic_body(payload))
    comments = [rest_comment(501, body, login="agent-app[bot]", uid=77, bot=True, created="2026-09-01T00:00:05Z")]
    routes = issue_routes(comments)
    routes[("GET", f"repos/{REPO}/issues/1/comments?per_page=100&page=1")] = comments
    serve(routes)

    context = get_issue_context(Runner(), config=config, issue_number=1)

    merged = [c for c in context.comments if c.body == body]
    assert len(merged) == 1  # the transport record was matched, not appended twice
    assert (merged[0].author, merged[0].author_id, merged[0].comment_id) == ("agent-app[bot]", 77, 501)
    selected = recover_plan_validation_diagnostic(
        context.comments, repository=REPO, issue_number=1,
        expected_author_login="agent-app[bot]", expected_author_id=77,
        planning_generation=1, target_coder_round=1, prior_plan_subject=None,
        candidate_kind="plan_state", architecture_contract_version=1,
        execution_strategy_contract_version=1, risk_test_matrix_contract_version=1,
    )
    assert selected is not None and selected.server_comment_id == 501


def test_signed_requirement_ids_match_across_transports(shim_config):
    config, serve = shim_config
    signed = "Must keep the cache.\n\n-- Human Reviewer"
    comments = [rest_comment(601, signed, login="reviewer-app[bot]", uid=5, bot=True)]
    routes = issue_routes(comments, body=signed, issue_user=user("reviewer-app[bot]", uid=5, bot=True))
    routes[("GET", f"repos/{REPO}/pulls/1")] = rest_pr(1, body=signed)
    routes[("GET", f"repos/{REPO}/pulls/1/reviews?per_page=100&page=1")] = [{
        "id": 71, "node_id": "PRR_71", "user": user("reviewer-app[bot]", uid=5, bot=True), "body": signed,
        "state": "COMMENTED", "submitted_at": "2026-09-03T00:00:00Z", "commit_id": "a" * 40,
    }]
    serve(routes)
    runner = Runner()
    issue = get_issue_context(runner, config=config, issue_number=1)
    pr = get_pr_review_context(runner, config=config, pr_number=1)

    from coding_review_agent_loop.github import _parse_issue_human_requirements, _parse_pr_human_requirements

    graphql_issue = {
        "body": signed, "author": {"login": "reviewer-app"}, "createdAt": "2026-08-01T00:00:00Z",
        "url": f"https://github.com/{REPO}/issues/1",
        "comments": [_gql_comment(601, signed, "reviewer-app", "2026-09-01T00:00:00Z")],
    }
    graphql_pr = {
        "comments": [_gql_comment(601, signed, "reviewer-app", "2026-09-01T00:00:00Z")],
        "reviews": [{"id": "PRR_71", "author": {"login": "reviewer-app"}, "body": signed,
                     "state": "COMMENTED", "submittedAt": "2026-09-03T00:00:00Z"}],
    }
    assert issue.human_requirements, "fixture must contain signed requirements"
    assert [human_requirement_id(r) for r in issue.human_requirements] == [
        human_requirement_id(r) for r in _parse_issue_human_requirements(graphql_issue)
    ]
    assert [human_requirement_id(r) for r in pr.human_requirements] == [
        human_requirement_id(r) for r in _parse_pr_human_requirements(graphql_pr)
    ]


def test_mergeability_through_shim_keeps_unknown_unknown(shim_config):
    config, serve = shim_config
    config = replace(config, mergeability_poll_attempts=1)
    serve({("GET", f"repos/{REPO}/pulls/5"): rest_pr(mergeable=None, mergeable_state="unknown")})
    assert get_pr_mergeability(Runner(), config=config, pr_number=5).state == "unknown"
    serve({("GET", f"repos/{REPO}/pulls/5"): rest_pr(mergeable=False, mergeable_state="dirty")})
    assert get_pr_mergeability(Runner(), config=config, pr_number=5).state == "conflicted"


# ------------------------------------- transport identity merge (github.py) ---


class RoutedRunner:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def run(self, args, *, cwd=None, check=True, **kwargs):
        self.calls.append(list(args))
        for predicate, result in self.responses:
            if predicate(args):
                return completed(args, *result) if isinstance(result, tuple) else completed(args, 0, result)
        raise AssertionError(f"unexpected command {args}")


def _projection_runner(projection_comments, rest_comments):
    projection = json.dumps({"number": 1, "title": "t", "body": "", "url": "u", "author": {"login": "a"},
                             "createdAt": "2026-01-01T00:00:00Z", "comments": projection_comments})
    return RoutedRunner([
        (lambda a: a[1:3] == ["issue", "view"], projection),
        (lambda a: a[1] == "api" and "/comments?" in a[2], json.dumps(rest_comments)),
    ])


def test_graphql_projection_bot_diagnostic_authenticates_without_the_shim(tmp_path):
    payload = replace(_diagnostic_payload(), repository="OWNER/REPO", issue_number=1,
                      expected_producer_login="agent-app[bot]", expected_producer_id=77)
    body = str(encode_plan_validation_diagnostic_body(payload))
    runner = _projection_runner(
        [_gql_comment(501, body, "agent-app", "2026-09-01T00:00:05Z")],
        [rest_comment(501, body, login="agent-app[bot]", uid=77, bot=True, created="2026-09-01T00:00:05Z")],
    )
    context = get_issue_context(runner, config=make_config(tmp_path), issue_number=1)
    matched = [c for c in context.comments if c.body == body]
    assert len(matched) == 1
    assert (matched[0].author, matched[0].author_id, matched[0].comment_id) == ("agent-app[bot]", 77, 501)


def test_ambiguous_bot_and_user_protocol_comments_fail_closed(tmp_path):
    body = "<!-- AGENT_LOOP_META: x -->"
    created = "2026-09-01T00:00:05Z"
    # A diagnostic elsewhere on the issue makes the REST identity merge run.
    payload = replace(_diagnostic_payload(), repository="OWNER/REPO", issue_number=1)
    diagnostic = str(encode_plan_validation_diagnostic_body(payload))
    runner = _projection_runner(
        [_gql_comment(1, body, "agent-app", created),
         _gql_comment(3, diagnostic, "agent", "2026-09-01T00:00:09Z")],
        [rest_comment(1, body, login="agent-app", uid=3, created=created),
         rest_comment(2, body, login="agent-app[bot]", uid=77, bot=True, created=created),
         rest_comment(3, diagnostic, login="agent", uid=7, created="2026-09-01T00:00:09Z")],
    )
    with pytest.raises(AgentLoopError, match="differ only by an app"):
        get_issue_context(runner, config=make_config(tmp_path), issue_number=1)


def test_human_copy_of_bot_marker_is_not_given_the_bot_identity(tmp_path):
    payload = replace(_diagnostic_payload(), repository="OWNER/REPO", issue_number=1,
                      expected_producer_login="agent-app[bot]", expected_producer_id=77)
    body = str(encode_plan_validation_diagnostic_body(payload))
    runner = _projection_runner(
        [_gql_comment(9, body, "mallory", "2026-09-01T00:00:06Z")],
        [rest_comment(9, body, login="mallory", uid=66, created="2026-09-01T00:00:06Z")],
    )
    context = get_issue_context(runner, config=make_config(tmp_path), issue_number=1)
    assert recover_plan_validation_diagnostic(
        context.comments, repository="OWNER/REPO", issue_number=1,
        expected_author_login="agent-app[bot]", expected_author_id=77,
        planning_generation=1, target_coder_round=1, prior_plan_subject=None,
        candidate_kind="plan_state", architecture_contract_version=1,
        execution_strategy_contract_version=1, risk_test_matrix_contract_version=1,
    ) is None


def test_strip_bot_login_suffix():
    assert strip_bot_login_suffix("x[bot]") == "x"
    assert strip_bot_login_suffix("x") == "x"
    assert strip_bot_login_suffix(None) is None


# ------------------------------------------- commit provenance fallback ---


def _rest_commits(n, start=0):
    return [{"sha": f"{i:040x}", "commit": {"message": f"m{i}"}} for i in range(start, start + n)]


def _provenance_runner(total, pages, *, head="a" * 40, final=None, graphql=(1, "", REFUSAL)):
    heads = iter([(head, total), final or (head, total)])

    def pr_object(args):
        current_head, current_total = next(heads)
        return json.dumps({"head": {"sha": current_head}, "commits": current_total})

    responses = [(lambda a: a[1:3] == ["api", "graphql"], graphql)]
    runner = RoutedRunner(responses)
    original = runner.run

    def run(args, **kwargs):
        if args[1] == "api" and args[2] == f"repos/{REPO}/pulls/7":
            runner.calls.append(list(args))
            return completed(args, 0, pr_object(args))
        if args[1] == "api" and args[2].startswith(f"repos/{REPO}/pulls/7/commits"):
            runner.calls.append(list(args))
            number = int(args[2].rsplit("page=", 1)[1])
            return completed(args, 0, json.dumps(pages[number - 1] if number <= len(pages) else []))
        return original(args, **kwargs)

    runner.run = run
    return runner


def test_commit_provenance_falls_back_to_rest_on_refusal(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_LOOP_GH_TRANSPORT", "auto")
    pages = [_rest_commits(100), _rest_commits(100, 100), _rest_commits(50, 200)]
    commits = read_pull_request_commit_metadata(
        _provenance_runner(250, pages), config=make_config(tmp_path), pr_number=7
    )
    assert len(commits) == 250 and commits[0].message == "m0"


def test_commit_provenance_rest_fails_closed_above_250(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_LOOP_GH_TRANSPORT", "rest")
    with pytest.raises(AgentLoopError, match="at most 250"):
        read_pull_request_commit_metadata(_provenance_runner(251, []), config=make_config(tmp_path), pr_number=7)


@pytest.mark.parametrize(("pages", "total", "final", "match"), [
    ([_rest_commits(2)], 3, None, "truncated"),
    ([_rest_commits(1) + _rest_commits(1)], 2, None, "repeated"),
    ([_rest_commits(2)], 2, ("b" * 40, 2), "changed during provenance"),
    ([_rest_commits(2)], 2, ("a" * 40, 3), "changed during provenance"),
])
def test_commit_provenance_rest_invariants(tmp_path, monkeypatch, pages, total, final, match):
    monkeypatch.setenv("AGENT_LOOP_GH_TRANSPORT", "rest")
    with pytest.raises(AgentLoopError, match=match):
        read_pull_request_commit_metadata(
            _provenance_runner(total, pages, final=final), config=make_config(tmp_path), pr_number=7
        )


def test_commit_provenance_non_refusal_graphql_error_still_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_LOOP_GH_TRANSPORT", "auto")
    runner = _provenance_runner(2, [_rest_commits(2)], graphql=(1, "", "HTTP 502 Bad Gateway"))
    with pytest.raises(AgentLoopError, match="query failed"):
        read_pull_request_commit_metadata(runner, config=make_config(tmp_path), pr_number=7)
    assert not any("/commits" in call[2] for call in runner.calls if call[1] == "api")


def test_commit_provenance_graphql_mode_does_not_fall_back(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_LOOP_GH_TRANSPORT", "graphql")
    with pytest.raises(AgentLoopError, match="query failed"):
        read_pull_request_commit_metadata(_provenance_runner(2, [_rest_commits(2)]), config=make_config(tmp_path), pr_number=7)


def test_discussion_agenda_tokens_ignore_the_rest_bot_suffix():
    from coding_review_agent_loop.github import IssueContext
    from coding_review_agent_loop.orchestrator import _build_discuss_agenda_support_corpus

    def corpus(author):
        context = IssueContext(
            number=1, repo=REPO, title="t", body="b", url=None, human_requirements=(),
            comments=(IssueComment(author=author, created_at="2026-01-01T00:00:00Z", body="hello"),),
        )
        return _build_discuss_agenda_support_corpus(
            issue_context=context, round_history=(), prior_agenda=None,
            configured_reviewers=("codex",), analyzer="codex",
        ).tokens

    assert corpus("helper[bot]") == corpus("helper")


def test_search_issues_and_create_issue_through_shim(shim_config):
    from coding_review_agent_loop.github import create_issue, search_issues

    config, serve = shim_config
    serve({
        ("GET", f"repos/{REPO}/issues?state=all&per_page=100&page=1"): [
            rest_issue(20, "Stage 1 (from #12)", body="child"),
            rest_issue(21, "Stage 2 (from #123)"),
            rest_issue(22, "PR (from #12)", pr=True),
        ],
        ("POST", f"repos/{REPO}/issues"): {"number": 30, "html_url": f"https://github.com/{REPO}/issues/30"},
    })
    found = search_issues(Runner(), config=config, search='"(from #12)" in:title')
    assert [issue.number for issue in found] == [20]
    assert create_issue(Runner(), config=config, title="T", body="B") == f"https://github.com/{REPO}/issues/30"


def test_skill_helper_fetchers_through_shim(shim_config):
    from helpers.skill_runner import _fetch_issue_context, _fetch_pr_json

    config, serve = shim_config
    routes = issue_routes([rest_comment(11, "hi")])
    routes[("GET", f"repos/{REPO}/pulls/5")] = rest_pr()
    routes[("GET", f"repos/{REPO}/issues/5/comments?per_page=100&page=1")] = []
    routes[("GET", f"repos/{REPO}/pulls/5/reviews?per_page=100&page=1")] = []
    serve(routes)
    pr = _fetch_pr_json(REPO, 5, gh_cmd=config.gh_cmd)
    assert (pr["headRefOid"], pr["state"], pr["reviews"]) == ("a" * 40, "OPEN", [])
    issue = _fetch_issue_context(REPO, 1, gh_cmd=config.gh_cmd)
    assert [c.body for c in issue.comments] == ["hi"]


# ------------------------------------- code-review round 1 regressions ---


def test_incomplete_record_cannot_vanish_into_a_filtered_empty_result():
    incomplete = {"id": 1, "number": 20, "title": "Stage 1 (from #12)", "state": "open",
                  "html_url": "u", "user": user("a")}  # no `body` key at all
    result, _found = search('"(from #12)" in:title', [incomplete])
    assert result.code == 1 and result.out == ""
    unrelated_incomplete = dict(incomplete, title="Unrelated")
    assert search('"(from #12)" in:title', [unrelated_incomplete])[0].code == 1


def test_null_body_is_legitimate_but_missing_body_is_not():
    null_body = dict(rest_pr(), body=None)
    api = FakeApi({("GET", f"repos/{REPO}/pulls/5"): null_body})
    result = run_shim(["pr", "view", "5", "--repo", REPO, "--json", "body"], api)
    assert result.code == 0 and json.loads(result.out) == {"body": ""}
    missing = rest_pr()
    del missing["body"]
    api = FakeApi({("GET", f"repos/{REPO}/pulls/5"): missing})
    assert run_shim(["pr", "view", "5", "--repo", REPO, "--json", "number"], api).code == 1


def test_comment_with_only_an_id_is_rejected():
    routes = issue_routes([{"id": 777}])
    result = run_shim(["issue", "view", "1", "--repo", REPO, "--json", "comments"], FakeApi(routes))
    assert result.code == 1 and result.out == ""


@pytest.mark.parametrize("draft", ["missing", None, "false", 0])
def test_pr_ready_requires_a_real_boolean_draft_state(draft):
    refreshed = rest_pr()
    if draft == "missing":
        del refreshed["draft"]
    else:
        refreshed["draft"] = draft
    api = FakeApi({
        ("POST", f"repos/{REPO}/pulls/5/ccr/ready_for_review"): {},
        ("GET", f"repos/{REPO}/pulls/5"): refreshed,
    })
    assert run_shim(["pr", "ready", "5", "--repo", REPO], api).code == 1


def test_failing_status_is_not_masked_by_a_same_named_check_run():
    sha = "a" * 40
    api = FakeApi({
        ("GET", f"repos/{REPO}/pulls/5"): rest_pr(),
        ("GET", f"repos/{REPO}/commits/{sha}/check-runs?per_page=100&page=1"): {
            "check_runs": [{"id": 1, "name": "ci", "status": "completed", "conclusion": "success"}]},
        ("GET", f"repos/{REPO}/commits/{sha}/status"): {"total_count": 1, "state": "failure"},
        ("GET", f"repos/{REPO}/commits/{sha}/status?per_page=100&page=1"): {"statuses": [{"id": 1, "context": "ci", "state": "failure"}]},
    })
    result = run_shim(["pr", "checks", "5", "--repo", REPO], api)
    assert result.code == 1
    assert result.out.splitlines() == ["ci\tpass\t\t", "ci\tfail\t\t"]


@pytest.mark.parametrize("argv", [
    ["issue", "list", "--repo", REPO, "--json", "bogus"],
    ["pr", "list", "--repo", REPO, "--json", "bogus"],
    ["issue", "view", "1", "--repo", REPO, "--json", "bogus"],
])
def test_unknown_fields_are_rejected_before_any_request(argv):
    api = FakeApi()
    result = run_shim(argv, api)
    assert result.code == shim.EXIT_USAGE
    assert api.calls == []


@pytest.mark.parametrize("argv", [
    ["pr", "merge", "5", "--repo", REPO, "--mer"],
    ["pr", "ready", "5", "--repo", REPO, "--und"],
    ["issue", "comment", "5", "--repo", REPO, "--bod", "x"],
])
def test_abbreviated_flags_are_refused_without_mutation(argv):
    api = FakeApi({("PUT", f"repos/{REPO}/pulls/5/merge"): {"merged": True}})
    result = run_shim(argv, api)
    assert result.code == shim.EXIT_USAGE
    assert api.calls == []


def test_pending_review_without_submitted_at_is_accepted():
    reviews = [{"id": 31, "node_id": "PRR_31", "user": user("bob"), "body": "", "state": "PENDING", "commit_id": "b" * 40}]
    api = FakeApi({
        ("GET", f"repos/{REPO}/pulls/5"): rest_pr(),
        ("GET", page(f"repos/{REPO}/pulls/5/reviews")): reviews,
    })
    result = run_shim(["pr", "view", "5", "--repo", REPO, "--json", "reviews"], api)
    assert result.code == 0
    assert json.loads(result.out)["reviews"][0]["submittedAt"] is None
    submitted_missing = [dict(reviews[0], state="APPROVED")]
    api = FakeApi({
        ("GET", f"repos/{REPO}/pulls/5"): rest_pr(),
        ("GET", page(f"repos/{REPO}/pulls/5/reviews")): submitted_missing,
    })
    assert run_shim(["pr", "view", "5", "--repo", REPO, "--json", "reviews"], api).code == 1


def _status_api(first_page, second_page, *, total, state):
    sha = "a" * 40
    return FakeApi({
        ("GET", f"repos/{REPO}/pulls/5"): rest_pr(),
        ("GET", f"repos/{REPO}/commits/{sha}/check-runs?per_page=100&page=1"): {"check_runs": []},
        ("GET", f"repos/{REPO}/commits/{sha}/status"): {"total_count": total, "state": state},
        ("GET", f"repos/{REPO}/commits/{sha}/status?per_page=100&page=1"): {"statuses": first_page},
        ("GET", f"repos/{REPO}/commits/{sha}/status?per_page=100&page=2"): {"statuses": second_page},
    })


def test_status_failure_on_a_later_page_is_reported():
    first = [{"id": i, "context": f"ok{i}", "state": "success"} for i in range(100)]
    second = [{"id": 100, "context": "late", "state": "failure"}]
    result = run_shim(["pr", "checks", "5", "--repo", REPO], _status_api(first, second, total=101, state="failure"))
    assert result.code == 1
    assert "late\tfail" in result.out


def test_aggregate_status_state_is_never_reported_as_better():
    only = [{"id": 1, "context": "x", "state": "success"}]
    result = run_shim(["pr", "checks", "5", "--repo", REPO], _status_api(only, [], total=1, state="failure"))
    assert result.code == 1
    result = run_shim(["pr", "checks", "5", "--repo", REPO], _status_api(only, [], total=1, state="pending"))
    assert result.code == shim.EXIT_PENDING


def test_status_count_mismatch_fails_closed():
    only = [{"id": 1, "context": "x", "state": "success"}]
    result = run_shim(["pr", "checks", "5", "--repo", REPO], _status_api(only, [], total=31, state="success"))
    assert result.code == 1 and result.out == ""


def test_same_named_check_runs_from_different_suites_are_all_reported():
    sha = "a" * 40
    api = FakeApi({
        ("GET", f"repos/{REPO}/pulls/5"): rest_pr(),
        ("GET", f"repos/{REPO}/commits/{sha}/check-runs?per_page=100&page=1"): {"check_runs": [
            {"id": 1, "name": "ci", "status": "completed", "conclusion": "failure", "check_suite": {"id": 10}},
            {"id": 2, "name": "ci", "status": "completed", "conclusion": "success", "check_suite": {"id": 11}},
        ]},
        ("GET", f"repos/{REPO}/commits/{sha}/status"): {"total_count": 0, "state": "pending"},
        ("GET", f"repos/{REPO}/commits/{sha}/status?per_page=100&page=1"): {"statuses": []},
    })
    result = run_shim(["pr", "checks", "5", "--repo", REPO], api)
    assert result.code == 1
    assert sorted(result.out.splitlines()) == ["ci\tfail\t\t", "ci\tpass\t\t"]
