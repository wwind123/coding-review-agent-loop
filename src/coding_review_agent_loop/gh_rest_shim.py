"""``agent-loop-gh``: a ``gh``-compatible shim for hosts that refuse GraphQL (#1029).

Install it as ``gh`` earlier on ``PATH`` (``agent-loop-gh shim-install DIR``) or
pass ``--gh-cmd agent-loop-gh``.  It emulates only the GraphQL-backed ``gh``
porcelain forms agent-loop, its skill helpers, and its coder prompts use,
calling repository-scoped REST through the real ``gh api`` and printing the
same ``--json`` shape ``gh`` prints.  Everything else, including every
``gh api`` call, passes through to the real ``gh`` unchanged.

Transport is chosen before a command runs (see ``github_transport``), so a
write is never attempted twice.  Emulation fails closed: an unsupported flag
or field, a malformed or incomplete page, or a failed request exits non-zero
instead of printing partial output.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from .github_transport import TransportConfigError, select_transport

EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_PENDING = 8
EXIT_NO_GH = 127
PAGE_SIZE = 100
MAX_PAGES = 10_000
REST_HINT = "use `gh api repos/{owner}/{repo}/...` instead"
DIFF_MEDIA_TYPE = "application/vnd.github.diff"


class ShimError(Exception):
    def __init__(self, message: str, code: int = EXIT_ERROR) -> None:
        super().__init__(message)
        self.code = code


class UsageError(ShimError):
    def __init__(self, message: str) -> None:
        super().__init__(f"{message}; {REST_HINT}", EXIT_USAGE)


# ---------------------------------------------------------------- real gh ---


def _own_path() -> Path:
    return Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else Path(__file__).resolve()


def find_real_gh(env: Mapping[str, str] | None = None, *, own: Path | None = None) -> str:
    values = os.environ if env is None else env
    own_path = own if own is not None else _own_path()
    explicit = values.get("AGENT_LOOP_REAL_GH")
    if explicit:
        resolved = Path(explicit).resolve()
        if resolved == own_path:
            raise ShimError("AGENT_LOOP_REAL_GH points at agent-loop-gh itself.", EXIT_NO_GH)
        return explicit
    for directory in (values.get("PATH") or "").split(os.pathsep):
        if not directory:
            continue
        candidate = Path(directory) / "gh"
        if candidate.is_file() and os.access(candidate, os.X_OK) and candidate.resolve() != own_path:
            return str(candidate)
    raise ShimError(
        "agent-loop-gh could not find the real `gh` on PATH; install GitHub CLI or set AGENT_LOOP_REAL_GH.",
        EXIT_NO_GH,
    )


# --------------------------------------------------------------- REST API ---


class GhApi:
    """Repository-scoped REST through the real ``gh api`` (credentials via gh/proxy)."""

    def __init__(self, real_gh: str, *, run: Callable[..., subprocess.CompletedProcess[str]] | None = None) -> None:
        self.real_gh = real_gh
        self._run = run or subprocess.run

    def _call(self, path: str, *, method: str = "GET", payload: object = None, accept: str | None = None) -> str:
        argv = [self.real_gh, "api", path]
        if method != "GET":
            argv += ["-X", method]
        if accept:
            argv += ["-H", f"Accept: {accept}"]
        stdin = None
        if payload is not None:
            argv += ["--input", "-"]
            stdin = json.dumps(payload)
        result = self._run(argv, input=stdin, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise ShimError(f"GitHub REST {method} {path} failed: {_error_message(result)}")
        return result.stdout or ""

    def json(self, path: str, *, method: str = "GET", payload: object = None) -> object:
        text = self._call(path, method=method, payload=payload)
        try:
            return json.loads(text or "null")
        except json.JSONDecodeError as exc:
            raise ShimError(f"GitHub REST {method} {path} returned malformed JSON.") from exc

    def obj(self, path: str, *, method: str = "GET", payload: object = None) -> dict:
        value = self.json(path, method=method, payload=payload)
        if not isinstance(value, dict):
            raise ShimError(f"GitHub REST {method} {path} did not return an object.")
        return value

    def text(self, path: str, *, accept: str) -> str:
        return self._call(path, accept=accept)

    def pages(
        self,
        path: str,
        *,
        key: str | None = None,
        identity: Callable[[dict], object] | None = None,
        want: int | None = None,
        keep: Callable[[dict], bool] | None = None,
        validate: Callable[[dict, str], None] | None = None,
    ) -> list[dict]:
        """Read a list endpoint to completion (or until ``want`` kept items).

        Every page must be a list of objects; a repeated identity or a page
        that fails after earlier pages is an error, never a shorter list.
        """
        separator = "&" if "?" in path else "?"
        kept: list[dict] = []
        seen: set[object] = set()
        for page in range(1, MAX_PAGES + 1):
            value = self.json(f"{path}{separator}per_page={PAGE_SIZE}&page={page}")
            items = value.get(key) if key and isinstance(value, dict) else value
            if not isinstance(items, list):
                raise ShimError(f"GitHub REST {path} page {page} is not a list.")
            for item in items:
                if not isinstance(item, dict):
                    raise ShimError(f"GitHub REST {path} page {page} contains a malformed record.")
                # Validate before filtering so an incomplete record can never
                # disappear into a shorter, successful result.
                if validate is not None:
                    validate(item, f"{path} page {page}")
                if identity is not None:
                    marker = identity(item)
                    if marker is None:
                        raise ShimError(f"GitHub REST {path} page {page} contains a record without an identity.")
                    if marker in seen:
                        raise ShimError(f"GitHub REST {path} pagination repeated a record.")
                    seen.add(marker)
                if keep is None or keep(item):
                    kept.append(item)
                    if want is not None and len(kept) >= want:
                        return kept
            if len(items) < PAGE_SIZE:
                return kept
        raise ShimError(f"GitHub REST {path} exceeded the pagination bound.")


def _error_message(result: subprocess.CompletedProcess[str]) -> str:
    try:
        parsed = json.loads(result.stdout or "")
        if isinstance(parsed, dict) and isinstance(parsed.get("message"), str):
            return parsed["message"]
    except json.JSONDecodeError:
        pass
    detail = (result.stderr or "").strip().splitlines()
    return detail[-1] if detail else f"exit {result.returncode}"


def _require_int(item: dict, field: str, where: str) -> int:
    value = item.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ShimError(f"GitHub REST {where} returned a record without `{field}`.")
    return value


# ------------------------------------------------------------- validation ---


def _need(raw: dict, where: str, field: str, *kinds: type, nullable: bool = False) -> None:
    """Require ``field`` to be present with one of ``kinds`` (``None`` only if nullable)."""
    if field not in raw:
        raise ShimError(f"GitHub REST {where} returned a record without `{field}`.")
    value = raw[field]
    if value is None and nullable:
        return
    if isinstance(value, bool) and bool not in kinds:
        raise ShimError(f"GitHub REST {where} returned a malformed `{field}`.")
    if not isinstance(value, kinds):
        raise ShimError(f"GitHub REST {where} returned a malformed `{field}`.")


def validate_issue(raw: dict, where: str) -> None:
    _need(raw, where, "number", int)
    _need(raw, where, "title", str)
    _need(raw, where, "body", str, nullable=True)
    _need(raw, where, "state", str)
    _need(raw, where, "html_url", str)
    _need(raw, where, "user", dict, nullable=True)


def validate_comment(raw: dict, where: str) -> None:
    _need(raw, where, "id", int)
    _need(raw, where, "body", str)
    _need(raw, where, "created_at", str)
    _need(raw, where, "html_url", str)
    _need(raw, where, "user", dict, nullable=True)


def validate_pr(raw: dict, where: str) -> None:
    validate_issue(raw, where)
    _need(raw, where, "draft", bool)
    _need(raw, where, "head", dict)
    _need(raw, where, "base", dict)
    _need(raw["head"], where + " head", "sha", str)
    _need(raw["head"], where + " head", "ref", str)
    _need(raw["base"], where + " base", "ref", str)


def validate_review(raw: dict, where: str) -> None:
    _need(raw, where, "id", int)
    _need(raw, where, "body", str, nullable=True)
    _need(raw, where, "state", str)
    _need(raw, where, "user", dict, nullable=True)
    # GitHub omits `submitted_at` for a pending (unsubmitted) review.
    if not (raw.get("state") == "PENDING" and "submitted_at" not in raw):
        _need(raw, where, "submitted_at", str, nullable=True)


# ------------------------------------------------------------ projections ---


def graphql_login(user: object) -> str | None:
    """Return a login as gh's GraphQL projection spells it (bots lose ``[bot]``)."""
    if not isinstance(user, dict):
        return None
    login = user.get("login")
    if not isinstance(login, str):
        return None
    if user.get("type") == "Bot" and login.endswith("[bot]"):
        return login[: -len("[bot]")]
    return login


def _author(user: object) -> dict | None:
    login = graphql_login(user)
    if login is None:
        return None
    return {"login": login}


def _upper(value: object) -> str | None:
    return value.upper() if isinstance(value, str) else None


def _comment(raw: dict) -> dict:
    return {
        "id": raw.get("node_id"),
        "author": _author(raw.get("user")),
        "authorAssociation": _upper(raw.get("author_association")),
        "body": raw.get("body") or "",
        "createdAt": raw.get("created_at"),
        "url": raw.get("html_url"),
    }


def _review(raw: dict) -> dict:
    # gh's review projection has no URL; adding one would change
    # signed-requirement identities across transports.
    commit_id = raw.get("commit_id")
    return {
        "id": raw.get("node_id"),
        "author": _author(raw.get("user")),
        "authorAssociation": _upper(raw.get("author_association")),
        "body": raw.get("body") or "",
        "state": raw.get("state"),
        "submittedAt": raw.get("submitted_at"),
        "commit": {"oid": commit_id} if isinstance(commit_id, str) else None,
    }


def _labels(raw: dict) -> list[dict]:
    return [{"name": label.get("name")} for label in raw.get("labels") or [] if isinstance(label, dict)]


def pr_state(raw: dict) -> str | None:
    if raw.get("merged") is True or raw.get("merged_at"):
        return "MERGED"
    return _upper(raw.get("state"))


def pr_mergeable(raw: dict) -> str:
    value = raw.get("mergeable")
    if value is True:
        return "MERGEABLE"
    if value is False:
        return "CONFLICTING"
    return "UNKNOWN"


def _issue_like_author(user: object) -> dict | None:
    login = graphql_login(user)
    if login is None:
        return None
    return {"id": user.get("node_id"), "is_bot": user.get("type") == "Bot", "login": login, "name": ""}  # type: ignore[union-attr]


# ----------------------------------------------------------------- output ---

_JQ_PATH = re.compile(r"^(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")


def render(value: object, jq: str | None) -> str:
    if jq is not None:
        expr = jq.strip()
        if expr != "." and not _JQ_PATH.fullmatch(expr):
            raise UsageError(f"--jq {jq!r} is not supported in REST mode (only dotted paths)")
        if expr != ".":
            for part in expr.split(".")[1:]:
                value = value.get(part) if isinstance(value, dict) else None
        if value is None:
            return "\n"
        if isinstance(value, str):
            return value + "\n"
        return json.dumps(value) + "\n"
    return json.dumps(value) + "\n"


def require_fields(fields: str, allowed: Iterable[str], form: str) -> None:
    """Reject unknown ``--json`` fields before fetching, even for empty results."""
    _select(fields, {name: (lambda: None) for name in allowed}, form)


def _select(fields: str, available: Mapping[str, Callable[[], object]], form: str) -> dict:
    names = [name.strip() for name in fields.split(",") if name.strip()]
    if not names:
        raise UsageError(f"{form} requires --json fields")
    unknown = [name for name in names if name not in available]
    if unknown:
        raise UsageError(f"{form} field(s) {', '.join(unknown)} are not supported in REST mode")
    return {name: available[name]() for name in names}


# ------------------------------------------------------------ arg parsing ---


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # pragma: no cover - exercised via UsageError
        raise UsageError(f"{self.prog}: {message}")


def _parser(prog: str) -> _Parser:
    parser = _Parser(prog=prog, add_help=False, allow_abbrev=False)
    parser.add_argument("-R", "--repo")
    return parser


def _parse(parser: _Parser, argv: Sequence[str]) -> argparse.Namespace:
    return parser.parse_args(list(argv))


_REMOTE_RE = re.compile(r"github\.com[:/]+(?P<repo>[^/\s]+/[^/\s]+?)(?:\.git)?/?$")


def resolve_repo(explicit: str | None, env: Mapping[str, str], cwd: Path | None = None) -> str:
    host = env.get("GH_HOST")
    if host and host != "github.com":
        raise UsageError(f"GH_HOST={host} is not supported in REST mode")
    repo = explicit or env.get("GH_REPO")
    if not repo:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"], cwd=cwd, capture_output=True, text=True, check=False
        )
        match = _REMOTE_RE.search((result.stdout or "").strip()) if result.returncode == 0 else None
        if not match:
            raise UsageError("could not determine the repository; pass --repo OWNER/NAME")
        repo = match.group("repo")
    repo = repo.removeprefix("https://github.com/").removeprefix("github.com/").strip("/").removesuffix(".git")
    if repo.count("/") != 1 or not all(repo.split("/")):
        raise UsageError(f"repository {repo!r} is not OWNER/NAME")
    return repo


def _read_body(body: str | None, body_file: str | None, stdin: Callable[[], str]) -> str | None:
    if body is not None and body_file is not None:
        raise UsageError("pass only one of --body and --body-file")
    if body_file is None:
        return body
    if body_file == "-":
        return stdin()
    try:
        return Path(body_file).read_text(encoding="utf-8")
    except OSError as exc:
        raise ShimError(f"could not read --body-file {body_file}: {exc}") from exc


# ------------------------------------------------------------- commands ---


@dataclass
class Context:
    api: GhApi
    env: Mapping[str, str]
    out: Callable[[str], None]
    err: Callable[[str], None]
    stdin: Callable[[], str]
    cwd: Path | None = None
    git: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run

    def repo(self, explicit: str | None) -> str:
        return resolve_repo(explicit, self.env, self.cwd)


def _issue_comments(ctx: Context, repo: str, number: int) -> list[dict]:
    raw = ctx.api.pages(
        f"repos/{repo}/issues/{number}/comments",
        identity=lambda item: item.get("id"),
        validate=validate_comment,
    )
    return [_comment(item) for item in raw]


def cmd_repo_view(ctx: Context, argv: Sequence[str]) -> int:
    parser = _parser("gh repo view")
    parser.add_argument("repo_arg", nargs="?")
    parser.add_argument("--json", required=True)
    parser.add_argument("-q", "--jq")
    args = _parse(parser, argv)
    require_fields(args.json, REPO_VIEW_FIELDS, "repo view")
    repo = ctx.repo(args.repo_arg or args.repo)
    raw = ctx.api.obj(f"repos/{repo}")
    data = _select(
        args.json,
        {
            "nameWithOwner": lambda: raw.get("full_name"),
            "name": lambda: raw.get("name"),
            "url": lambda: raw.get("html_url"),
            "defaultBranchRef": lambda: {"name": raw.get("default_branch")},
        },
        "repo view",
    )
    ctx.out(render(data, args.jq))
    return 0


def cmd_repo_clone(ctx: Context, argv: Sequence[str]) -> int:
    argv = list(argv)
    git_args: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, git_args = argv[:split], argv[split + 1 :]
    if not argv or len(argv) > 2 or any(part.startswith("-") for part in argv):
        raise UsageError("gh repo clone supports only `REPO [DIR] [-- GITFLAGS]` in REST mode")
    repo = resolve_repo(argv[0], ctx.env)
    target = argv[1] if len(argv) == 2 else repo.split("/")[1]
    clone = ctx.git(
        ["git", "clone", f"https://github.com/{repo}.git", target, *git_args],
        cwd=ctx.cwd, check=False,
    )
    if clone.returncode != 0:
        raise ShimError(f"git clone of {repo} failed (exit {clone.returncode}).")
    raw = ctx.api.obj(f"repos/{repo}")
    parent = raw.get("parent") if raw.get("fork") else None
    if isinstance(parent, dict) and isinstance(parent.get("full_name"), str):
        ctx.git(
            ["git", "-C", target, "remote", "add", "upstream", f"https://github.com/{parent['full_name']}.git"],
            cwd=ctx.cwd, check=False,
        )
    return 0


def cmd_issue_view(ctx: Context, argv: Sequence[str]) -> int:
    parser = _parser("gh issue view")
    parser.add_argument("number", type=int)
    parser.add_argument("--json", required=True)
    parser.add_argument("-q", "--jq")
    parser.add_argument("-c", "--comments", action="store_true")
    args = _parse(parser, argv)
    require_fields(args.json, ISSUE_VIEW_FIELDS, "issue view")
    repo = ctx.repo(args.repo)
    raw = ctx.api.obj(f"repos/{repo}/issues/{args.number}")
    validate_issue(raw, f"issues/{args.number}")
    data = _select(
        args.json,
        {
            "number": lambda: raw.get("number"),
            "title": lambda: raw.get("title"),
            "body": lambda: raw.get("body") or "",
            "url": lambda: raw.get("html_url"),
            "state": lambda: _upper(raw.get("state")),
            "author": lambda: _issue_like_author(raw.get("user")),
            "createdAt": lambda: raw.get("created_at"),
            "updatedAt": lambda: raw.get("updated_at"),
            "closedAt": lambda: raw.get("closed_at"),
            "labels": lambda: _labels(raw),
            "comments": lambda: _issue_comments(ctx, repo, args.number),
        },
        "issue view",
    )
    ctx.out(render(data, args.jq))
    return 0


def cmd_pr_view(ctx: Context, argv: Sequence[str]) -> int:
    parser = _parser("gh pr view")
    parser.add_argument("number", type=int)
    parser.add_argument("--json", required=True)
    parser.add_argument("-q", "--jq")
    parser.add_argument("-c", "--comments", action="store_true")
    args = _parse(parser, argv)
    require_fields(args.json, PR_VIEW_FIELDS, "pr view")
    repo = ctx.repo(args.repo)
    raw = ctx.api.obj(f"repos/{repo}/pulls/{args.number}")
    validate_pr(raw, f"pulls/{args.number}")
    head = raw.get("head") if isinstance(raw.get("head"), dict) else {}
    base = raw.get("base") if isinstance(raw.get("base"), dict) else {}

    def reviews() -> list[dict]:
        items = ctx.api.pages(
            f"repos/{repo}/pulls/{args.number}/reviews",
            identity=lambda item: item.get("id"),
            validate=validate_review,
        )
        return [_review(item) for item in items]

    data = _select(
        args.json,
        {
            "number": lambda: raw.get("number"),
            "title": lambda: raw.get("title"),
            "body": lambda: raw.get("body") or "",
            "url": lambda: raw.get("html_url"),
            "state": lambda: pr_state(raw),
            "isDraft": lambda: bool(raw.get("draft")),
            "author": lambda: _issue_like_author(raw.get("user")),
            "createdAt": lambda: raw.get("created_at"),
            "labels": lambda: _labels(raw),
            "headRefName": lambda: head.get("ref"),
            "headRefOid": lambda: head.get("sha"),
            "baseRefName": lambda: base.get("ref"),
            "mergeable": lambda: pr_mergeable(raw),
            # Upper-cased verbatim; an unknown or missing state is never
            # promoted to a confident value.
            "mergeStateStatus": lambda: _upper(raw.get("mergeable_state")) or "UNKNOWN",
            "comments": lambda: _issue_comments(ctx, repo, args.number),
            "reviews": reviews,
        },
        "pr view",
    )
    ctx.out(render(data, args.jq))
    return 0


def _limit(value: str) -> int:
    limit = int(value)
    if limit <= 0:
        raise UsageError("--limit must be positive")
    return limit


def cmd_pr_list(ctx: Context, argv: Sequence[str]) -> int:
    parser = _parser("gh pr list")
    parser.add_argument("-s", "--state", default="open", choices=("open", "closed", "all"))
    parser.add_argument("--json", required=True)
    parser.add_argument("-L", "--limit", type=_limit, default=30)
    parser.add_argument("-q", "--jq")
    args = _parse(parser, argv)
    require_fields(args.json, PR_LIST_FIELDS, "pr list")
    repo = ctx.repo(args.repo)
    items = ctx.api.pages(
        f"repos/{repo}/pulls?state={args.state}",
        identity=lambda item: item.get("id"),
        want=args.limit,
        validate=validate_pr,
    )
    rows = []
    for raw in items:
        head = raw.get("head") if isinstance(raw.get("head"), dict) else {}
        base = raw.get("base") if isinstance(raw.get("base"), dict) else {}
        rows.append(
            _select(
                args.json,
                {
                    "number": lambda raw=raw: raw.get("number"),
                    "title": lambda raw=raw: raw.get("title"),
                    "body": lambda raw=raw: raw.get("body") or "",
                    "url": lambda raw=raw: raw.get("html_url"),
                    "state": lambda raw=raw: pr_state(raw),
                    "isDraft": lambda raw=raw: bool(raw.get("draft")),
                    "headRefName": lambda head=head: head.get("ref"),
                    "baseRefName": lambda base=base: base.get("ref"),
                },
                "pr list",
            )
        )
    ctx.out(render(rows, args.jq))
    return 0


# --- issue search (search/issues is refused, so match locally) ---

_TOKEN_RE = re.compile(r"[^\w]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.split(text.casefold()) if token]


def _contains(haystack: list[str], needle: list[str]) -> bool:
    if not needle:
        return True
    width = len(needle)
    return any(haystack[index : index + width] == needle for index in range(len(haystack) - width + 1))


@dataclass(frozen=True)
class IssueQuery:
    phrases: tuple[tuple[str, ...], ...]
    in_title: bool
    in_body: bool
    state: str | None


def parse_search(query: str, *, repo: str) -> IssueQuery:
    """Parse the issue-search subset agent-loop generates; reject anything else."""
    phrases: list[tuple[str, ...]] = []
    scopes: set[str] = set()
    state: str | None = None
    index = 0
    while index < len(query):
        char = query[index]
        if char.isspace():
            index += 1
            continue
        if char == '"':
            index += 1
            chunk: list[str] = []
            while index < len(query) and query[index] != '"':
                if query[index] == "\\" and index + 1 < len(query):
                    index += 1
                chunk.append(query[index])
                index += 1
            if index >= len(query):
                raise UsageError("unterminated quoted phrase in --search")
            index += 1
            words = tuple(_tokens("".join(chunk)))
            if words:
                phrases.append(words)
            continue
        end = index
        while end < len(query) and not query[end].isspace():
            end += 1
        word = query[index:end]
        index = end
        key, sep, value = word.partition(":")
        if sep and key in {"repo", "is", "in"}:
            lowered = value.casefold()
            if key == "repo":
                if lowered != repo.casefold():
                    raise UsageError(f"--search qualifier repo:{value} does not match {repo}")
            elif key == "is":
                if lowered == "issue":
                    pass
                elif lowered in {"open", "closed"}:
                    state = lowered
                else:
                    raise UsageError(f"--search qualifier is:{value} is not supported in REST mode")
            else:
                if lowered not in {"title", "body"}:
                    raise UsageError(f"--search qualifier in:{value} is not supported in REST mode")
                scopes.add(lowered)
            continue
        if sep and re.fullmatch(r"[A-Za-z-]+", key):
            raise UsageError(f"--search qualifier {key}: is not supported in REST mode")
        words = tuple(_tokens(word))
        phrases.extend((part,) for part in words)
    return IssueQuery(
        phrases=tuple(phrases),
        in_title=not scopes or "title" in scopes,
        in_body=not scopes or "body" in scopes,
        state=state,
    )


def issue_matches(query: IssueQuery, raw: dict) -> bool:
    if query.state is not None and str(raw.get("state") or "").casefold() != query.state:
        return False
    haystack: list[str] = []
    if query.in_title:
        haystack += _tokens(str(raw.get("title") or ""))
    if query.in_body:
        haystack += ["\x00"] + _tokens(str(raw.get("body") or ""))
    return all(_contains(haystack, list(phrase)) for phrase in query.phrases)


def cmd_issue_list(ctx: Context, argv: Sequence[str]) -> int:
    parser = _parser("gh issue list")
    parser.add_argument("-S", "--search")
    parser.add_argument("-s", "--state", default="open", choices=("open", "closed", "all"))
    parser.add_argument("-L", "--limit", type=_limit, default=30)
    parser.add_argument("--json", required=True)
    parser.add_argument("-q", "--jq")
    args = _parse(parser, argv)
    require_fields(args.json, ISSUE_LIST_FIELDS, "issue list")
    repo = ctx.repo(args.repo)
    query = parse_search(args.search, repo=repo) if args.search else None

    def keep(item: dict) -> bool:
        if "pull_request" in item:
            return False
        return query is None or issue_matches(query, item)

    items = ctx.api.pages(
        f"repos/{repo}/issues?state={args.state}",
        identity=lambda item: item.get("id"),
        want=args.limit,
        keep=keep,
        validate=validate_issue,
    )
    rows = []
    for raw in items:
        rows.append(
            _select(
                args.json,
                {
                    "number": lambda raw=raw: raw.get("number"),
                    "title": lambda raw=raw: raw.get("title"),
                    "body": lambda raw=raw: raw.get("body") or "",
                    "url": lambda raw=raw: raw.get("html_url"),
                    "state": lambda raw=raw: _upper(raw.get("state")),
                },
                "issue list",
            )
        )
    ctx.out(render(rows, args.jq))
    return 0


def cmd_issue_create(ctx: Context, argv: Sequence[str]) -> int:
    parser = _parser("gh issue create")
    parser.add_argument("-t", "--title", required=True)
    parser.add_argument("-b", "--body")
    parser.add_argument("-F", "--body-file")
    parser.add_argument("-l", "--label", action="append", default=[])
    args = _parse(parser, argv)
    repo = ctx.repo(args.repo)
    body = _read_body(args.body, args.body_file, ctx.stdin)
    if body is None:
        raise UsageError("gh issue create requires --body or --body-file in REST mode")
    payload: dict[str, object] = {"title": args.title, "body": body}
    if args.label:
        payload["labels"] = args.label
    created = ctx.api.obj(f"repos/{repo}/issues", method="POST", payload=payload)
    ctx.out(f"{created.get('html_url')}\n")
    return 0


def cmd_comment(ctx: Context, argv: Sequence[str], *, kind: str) -> int:
    parser = _parser(f"gh {kind} comment")
    parser.add_argument("number", type=int)
    parser.add_argument("-b", "--body")
    parser.add_argument("-F", "--body-file")
    args = _parse(parser, argv)
    repo = ctx.repo(args.repo)
    body = _read_body(args.body, args.body_file, ctx.stdin)
    if body is None:
        raise UsageError(f"gh {kind} comment requires --body or --body-file in REST mode")
    created = ctx.api.obj(
        f"repos/{repo}/issues/{args.number}/comments", method="POST", payload={"body": body}
    )
    ctx.out(f"{created.get('html_url')}\n")
    return 0


def _current_branch(ctx: Context) -> str:
    result = ctx.git(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ctx.cwd, capture_output=True, text=True, check=False
    )
    branch = (result.stdout or "").strip()
    if result.returncode != 0 or not branch or branch == "HEAD":
        raise UsageError("could not determine the current branch; pass --head")
    return branch


def _open_prs_for(ctx: Context, repo: str, head: str, base: str | None) -> list[dict]:
    owner = repo.split("/")[0]
    path = f"repos/{repo}/pulls?state=open&head={quote(owner + ':' + head, safe='')}"
    if base:
        path += f"&base={quote(base, safe='')}"
    return ctx.api.pages(path, identity=lambda item: item.get("id"))


def cmd_pr_create(ctx: Context, argv: Sequence[str]) -> int:
    parser = _parser("gh pr create")
    parser.add_argument("-t", "--title")
    parser.add_argument("-b", "--body")
    parser.add_argument("-F", "--body-file")
    parser.add_argument("-B", "--base")
    parser.add_argument("-H", "--head")
    parser.add_argument("-d", "--draft", action="store_true")
    parser.add_argument("-l", "--label", action="append", default=[])
    parser.add_argument("-f", "--fill", action="store_true")
    args = _parse(parser, argv)
    if args.fill:
        raise UsageError("gh pr create --fill is not supported in REST mode")
    if not args.title:
        raise UsageError("gh pr create requires --title in REST mode")
    body = _read_body(args.body, args.body_file, ctx.stdin)
    if body is None:
        raise UsageError("gh pr create requires --body or --body-file in REST mode")
    repo = ctx.repo(args.repo)
    head = args.head or _current_branch(ctx)
    if ":" in head:
        raise UsageError("cross-repository --head OWNER:BRANCH is not supported in REST mode")
    base = args.base or ctx.api.obj(f"repos/{repo}").get("default_branch")
    if not isinstance(base, str) or not base:
        raise ShimError(f"could not determine the default branch of {repo}.")
    try:
        ctx.api.obj(f"repos/{repo}/branches/{quote(head, safe='')}")
    except ShimError as exc:
        raise ShimError(f"branch {head!r} is not on {repo}; push it first ({exc}).") from exc
    existing = _open_prs_for(ctx, repo, head, base)
    if existing:
        ctx.out(f"{existing[0].get('html_url')}\n")
        raise ShimError(
            f"a pull request for branch {head!r} into {base!r} already exists: {existing[0].get('html_url')}"
        )
    for label in args.label:
        try:
            ctx.api.obj(f"repos/{repo}/labels/{quote(label, safe='')}")
        except ShimError as exc:
            raise ShimError(f"label {label!r} does not exist on {repo}; create it first ({exc}).") from exc
    created = ctx.api.obj(
        f"repos/{repo}/pulls",
        method="POST",
        payload={"title": args.title, "head": head, "base": base, "body": body, "draft": bool(args.draft)},
    )
    number = _require_int(created, "number", "pull request create")
    url = created.get("html_url")
    if args.label:
        try:
            ctx.api.json(f"repos/{repo}/issues/{number}/labels", method="POST", payload={"labels": args.label})
        except ShimError as exc:
            ctx.out(f"{url}\n")
            raise ShimError(
                f"PR #{number} was created but labels failed ({exc}); do not create another PR; add labels with "
                f"`gh api -X POST repos/{repo}/issues/{number}/labels`."
            ) from exc
    ctx.out(f"{url}\n")
    return 0


def cmd_pr_ready(ctx: Context, argv: Sequence[str]) -> int:
    parser = _parser("gh pr ready")
    parser.add_argument("number", type=int)
    parser.add_argument("--undo", action="store_true")
    args = _parse(parser, argv)
    repo = ctx.repo(args.repo)
    route = "convert_to_draft" if args.undo else "ready_for_review"
    ctx.api.json(f"repos/{repo}/pulls/{args.number}/ccr/{route}", method="POST")
    refreshed = ctx.api.obj(f"repos/{repo}/pulls/{args.number}")
    draft = refreshed.get("draft")
    if not isinstance(draft, bool) or draft != args.undo:
        raise ShimError(
            f"PR #{args.number} draft state did not change to {'draft' if args.undo else 'ready'}."
        )
    ctx.err(
        f"Pull request #{args.number} is {'converted to draft' if args.undo else 'marked as ready for review'}\n"
    )
    return 0


def cmd_pr_merge(ctx: Context, argv: Sequence[str]) -> int:
    parser = _parser("gh pr merge")
    parser.add_argument("number", type=int)
    parser.add_argument("-m", "--merge", action="store_true")
    parser.add_argument("--match-head-commit")
    args = _parse(parser, argv)
    if not args.merge:
        raise UsageError("gh pr merge supports only --merge in REST mode")
    repo = ctx.repo(args.repo)
    payload: dict[str, object] = {"merge_method": "merge"}
    if args.match_head_commit:
        payload["sha"] = args.match_head_commit
    result = ctx.api.obj(f"repos/{repo}/pulls/{args.number}/merge", method="PUT", payload=payload)
    if result.get("merged") is not True:
        raise ShimError(f"PR #{args.number} was not merged: {result.get('message') or 'no confirmation'}.")
    ctx.err(f"Merged pull request #{args.number} ({result.get('sha')})\n")
    return 0


def cmd_pr_diff(ctx: Context, argv: Sequence[str]) -> int:
    parser = _parser("gh pr diff")
    parser.add_argument("number", type=int)
    args = _parse(parser, argv)
    repo = ctx.repo(args.repo)
    ctx.out(ctx.api.text(f"repos/{repo}/pulls/{args.number}", accept=DIFF_MEDIA_TYPE))
    return 0


_CHECK_BUCKETS = {
    "success": "pass", "neutral": "pass", "skipped": "skipping",
    "failure": "fail", "timed_out": "fail", "cancelled": "fail",
    "action_required": "fail", "startup_failure": "fail", "stale": "fail",
    "error": "fail", "pending": "pending",
}


def cmd_pr_checks(ctx: Context, argv: Sequence[str]) -> int:
    parser = _parser("gh pr checks")
    parser.add_argument("number", type=int, nargs="?")
    args = _parse(parser, argv)
    repo = ctx.repo(args.repo)
    if args.number is None:
        found = _open_prs_for(ctx, repo, _current_branch(ctx), None)
        if len(found) != 1:
            raise ShimError("could not find exactly one open pull request for the current branch.")
        number = _require_int(found[0], "number", "pull request list")
    else:
        number = args.number
    pr = ctx.api.obj(f"repos/{repo}/pulls/{number}")
    sha = (pr.get("head") or {}).get("sha") if isinstance(pr.get("head"), dict) else None
    if not isinstance(sha, str):
        raise ShimError(f"PR #{number} has no head commit.")
    # Check-runs and commit statuses are separate namespaces: a status must
    # never be hidden by a check-run that happens to share its name.
    # The endpoint's default `filter=latest` already keeps only the newest run
    # per check within each suite; runs sharing a name across apps or suites
    # are distinct checks, so every returned run is reported.
    run_entries: list[tuple[str, tuple[str, str]]] = []
    runs = ctx.api.pages(f"repos/{repo}/commits/{sha}/check-runs", key="check_runs", identity=lambda item: item.get("id"))
    for run in runs:
        name = str(run.get("name") or "")
        status = run.get("conclusion") if run.get("status") == "completed" else "pending"
        run_entries.append((name, (_CHECK_BUCKETS.get(str(status), "fail"), str(run.get("html_url") or ""))))
    statuses_by_context: dict[str, tuple[str, str]] = {}
    status_path = f"repos/{repo}/commits/{sha}/status"
    summary = ctx.api.obj(status_path)
    statuses = ctx.api.pages(status_path, key="statuses", identity=lambda item: item.get("id"))
    total = summary.get("total_count")
    if isinstance(total, int) and not isinstance(total, bool) and total != len(statuses):
        raise ShimError(f"GitHub REST {status_path} returned {len(statuses)} of {total} statuses.")
    for status in statuses:
        statuses_by_context.setdefault(
            str(status.get("context") or ""),
            (_CHECK_BUCKETS.get(str(status.get("state")), "fail"), str(status.get("target_url") or "")),
        )
    entries = sorted(run_entries) + sorted(statuses_by_context.items())
    for name, (bucket, url) in entries:
        ctx.out(f"{name}\t{bucket}\t\t{url}\n")
    buckets = {bucket for _name, (bucket, _url) in entries}
    # The aggregate state is authoritative too; never report better than it.
    aggregate = summary.get("state")
    if statuses and aggregate in {"failure", "error"}:
        buckets.add("fail")
    elif statuses and aggregate == "pending":
        buckets.add("pending")
    if "fail" in buckets:
        return EXIT_ERROR
    if "pending" in buckets:
        return EXIT_PENDING
    return 0


REPO_VIEW_FIELDS = ("nameWithOwner", "name", "url", "defaultBranchRef")
ISSUE_VIEW_FIELDS = (
    "number", "title", "body", "url", "state", "author", "createdAt", "updatedAt", "closedAt", "labels", "comments",
)
PR_VIEW_FIELDS = (
    "number", "title", "body", "url", "state", "isDraft", "author", "createdAt", "labels", "headRefName",
    "headRefOid", "baseRefName", "mergeable", "mergeStateStatus", "comments", "reviews",
)
PR_LIST_FIELDS = ("number", "title", "body", "url", "state", "isDraft", "headRefName", "baseRefName")
ISSUE_LIST_FIELDS = ("number", "title", "body", "url", "state")

COMMANDS: dict[tuple[str, str], Callable[[Context, Sequence[str]], int]] = {
    ("repo", "view"): cmd_repo_view,
    ("repo", "clone"): cmd_repo_clone,
    ("issue", "view"): cmd_issue_view,
    ("issue", "list"): cmd_issue_list,
    ("issue", "create"): cmd_issue_create,
    ("issue", "comment"): lambda ctx, argv: cmd_comment(ctx, argv, kind="issue"),
    ("pr", "view"): cmd_pr_view,
    ("pr", "list"): cmd_pr_list,
    ("pr", "comment"): lambda ctx, argv: cmd_comment(ctx, argv, kind="pr"),
    ("pr", "create"): cmd_pr_create,
    ("pr", "ready"): cmd_pr_ready,
    ("pr", "merge"): cmd_pr_merge,
    ("pr", "diff"): cmd_pr_diff,
    ("pr", "checks"): cmd_pr_checks,
}


def emulated_command(argv: Sequence[str]) -> Callable[[Context, Sequence[str]], int] | None:
    if len(argv) < 2:
        return None
    return COMMANDS.get((argv[0], argv[1]))


# ------------------------------------------------------------------ main ---


def shim_install(directory: str) -> int:
    target = Path(directory).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    link = target / "gh"
    source = _own_path()
    if link.is_symlink() or link.exists():
        if link.resolve() == source:
            print(f"{link} already points at {source}")
            return 0
        print(f"agent-loop-gh: {link} exists and is not this shim; remove it first.", file=sys.stderr)
        return EXIT_ERROR
    link.symlink_to(source)
    print(f"Installed {link} -> {source}\nPut {target} before the real gh on PATH, e.g.:\n  export PATH=\"{target}:$PATH\"")
    return 0


def run(
    argv: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    passthrough: Callable[[str, Sequence[str]], int] | None = None,
    api_run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    probe: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    out: Callable[[str], None] | None = None,
    err: Callable[[str], None] | None = None,
    stdin: Callable[[], str] | None = None,
    cwd: Path | None = None,
    git: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    own: Path | None = None,
) -> int:
    values = dict(os.environ if env is None else env)
    write_err = err or sys.stderr.write
    try:
        real_gh = find_real_gh(values, own=own)
        handler = emulated_command(argv)
        if handler is None or select_transport(real_gh, env=values, probe=probe) == "graphql":
            return (passthrough or _exec_passthrough)(real_gh, argv)
        ctx = Context(
            api=GhApi(real_gh, run=api_run),
            env=values,
            out=out or sys.stdout.write,
            err=write_err,
            stdin=stdin or sys.stdin.read,
            cwd=cwd,
            git=git or subprocess.run,
        )
        return handler(ctx, list(argv[2:]))
    except TransportConfigError as exc:
        write_err(f"agent-loop-gh: {exc}\n")
        return EXIT_USAGE
    except ShimError as exc:
        write_err(f"agent-loop-gh: {exc}\n")
        return exc.code


def _exec_passthrough(real_gh: str, argv: Sequence[str]) -> int:
    os.execv(real_gh, [real_gh, *argv])
    return EXIT_ERROR  # pragma: no cover - execv does not return


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) == 2 and args[0] == "shim-install":
        return shim_install(args[1])
    return run(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
