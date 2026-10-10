"""Hardened read-only git/gh runner behind ``agent-loop inspect`` (#1035).

In ``--agent-permissions sandboxed`` mode this is the only shell command a
Claude non-coder (reviewer, planner, analyzer, ...) may run.  The Bash grant
pins its prefix to the orchestrator's interpreter in isolated mode and to the
startup-verified absolute ``git`` and ``gh`` paths::

    <python> -I -m coding_review_agent_loop.cli inspect --git=<git> [--gh=<gh>] git|gh ...

Every request is checked against closed allowlists before anything runs:

* argument allowlist: exact subcommands, exact option tokens, a closed set of
  ``--format`` placeholders, and positionals that never start with ``-``;
* environment allowlist: git and gh get an environment built from an empty
  mapping with a PATH made only of the pinned executables' directories, so no
  inherited ``GIT_*`` (including every trace variable), ``GH_DEBUG``, or PATH
  entry reaches them;
* repository-config gate: the checkout's effective local/worktree/included git
  config may only contain allowlisted keys, so a coder-planted filter,
  signature program, include, alias, hook, or trace target cannot make a
  read-only inspection run a program or write a file.

This module deliberately imports only the standard library: it runs inside
the pinned interpreter with ``-I`` and must not import code a coder could have
changed in its checkout.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

EXIT_REJECTED = 2

GIT_SUBCOMMANDS = ("diff", "log", "show", "status", "rev-parse", "ls-files")
GH_SUBCOMMANDS = (("issue", "view"), ("pr", "view"), ("pr", "diff"), ("pr", "checks"))

# Exact flag tokens per git subcommand.  Valued options are handled below.
_GIT_DIFF_FLAGS = frozenset(
    {"--stat", "--name-only", "--name-status", "--numstat", "--patch", "-p"}
)
_GIT_FLAGS: dict[str, frozenset[str]] = {
    "diff": _GIT_DIFF_FLAGS,
    "log": _GIT_DIFF_FLAGS | {"--oneline"},
    "show": _GIT_DIFF_FLAGS | {"--oneline"},
    "status": frozenset({"--porcelain", "--short", "--branch"}),
    "rev-parse": frozenset({"--abbrev-ref", "--short"}),
    "ls-files": frozenset(),
}
_GIT_UNIFIED = frozenset({"diff", "log", "show"})
_GIT_FORMAT = frozenset({"log", "show"})
_GIT_MAX_COUNT = frozenset({"log"})
# These subcommands can render content through diff drivers and textconv.
_GIT_DIFF_RENDERING = frozenset({"diff", "log", "show"})
# Subcommands that would otherwise recurse into populated submodules.  A
# submodule's own local config is outside the repository-config gate, so its
# status or diff could run a coder-planted clean filter.
_GIT_SUBMODULE_IGNORING = frozenset({"diff", "log", "show", "status"})

NAMED_FORMATS = frozenset({"oneline", "short", "medium", "full", "fuller", "reference"})
# Longest tokens first so ``%an`` is not read as ``%a`` + ``n``.
FORMAT_PLACEHOLDERS = (
    "an", "ae", "ad", "ar", "at", "aI",
    "cn", "ce", "cd", "cr", "ct", "cI",
    "H", "h", "T", "t", "P", "p", "s", "b", "B", "d", "D", "n", "%",
)

_GH_FLAGS: dict[tuple[str, str], frozenset[str]] = {
    ("issue", "view"): frozenset({"--comments"}),
    ("pr", "view"): frozenset({"--comments"}),
    ("pr", "diff"): frozenset({"--name-only"}),
    ("pr", "checks"): frozenset(),
}
_GH_VALUED = frozenset({"--repo", "--json"})
_GH_REPO_RE = re.compile(r"^(?:[A-Za-z0-9.-]+/)?[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_GH_JSON_RE = re.compile(r"^[A-Za-z0-9_]+(?:,[A-Za-z0-9_]+)*$")
_DIGITS_RE = re.compile(r"^[0-9]{1,6}$")

# Forced git configuration.  The command scope overrides repository config,
# so these neutralize pagers, fsmonitor hooks, external diff drivers,
# signature verification programs, transports, hooks, submodule recursion,
# and Trace2 targets.  The Trace2 entries must stay last.
FORCED_GIT_CONFIG = (
    "core.pager=cat",
    "core.fsmonitor=false",
    "diff.external=",
    "log.showSignature=false",
    "gpg.program=false",
    "gpg.ssh.program=false",
    "gpg.x509.program=false",
    "protocol.allow=never",
    "core.hooksPath=/dev/null",
    "diff.ignoreSubmodules=all",
    "status.submoduleSummary=false",
    "submodule.recurse=false",
    "trace2.eventTarget=",
    "trace2.perfTarget=",
    "trace2.normalTarget=",
)
_TRACE2_OVERRIDES = FORCED_GIT_CONFIG[-3:]

# Closed environment allowlist.  PATH is never copied; it is constructed from
# the pinned executables' directories.  No GIT_* variable is ever copied.
ENV_ALLOWLIST = frozenset(
    {
        "HOME", "USER", "LOGNAME", "LANG", "LANGUAGE", "LC_ALL", "TZ", "TERM", "TMPDIR",
        "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME",
        "SSL_CERT_FILE", "SSL_CERT_DIR",
        "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY",
        "https_proxy", "http_proxy", "all_proxy", "no_proxy",
        "GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN",
        "GH_HOST", "GH_CONFIG_DIR",
        # Transport mode selector for the agent-loop-gh REST shim; a closed
        # enum value, never a path or program.
        "AGENT_LOOP_GH_TRANSPORT",
    }
)
ENV_ALLOWLIST_PREFIXES = ("LC_",)
FORCED_ENV = {
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_PAGER": "cat",
    "PAGER": "cat",
    "GH_PAGER": "cat",
    "GH_BROWSER": "false",
    "GH_PROMPT_DISABLED": "1",
    "GH_NO_UPDATE_NOTIFIER": "1",
}

# Repository-config gate allowlist: (section, variable, subsection).  The
# subsection is ``False`` for two-part keys and ``True`` for three-part keys
# such as ``remote.<name>.url``; keys are matched on their first and last
# components so branch names containing dots work.
CONFIG_KEY_ALLOWLIST = frozenset(
    {
        ("core", "repositoryformatversion", False),
        ("core", "filemode", False),
        ("core", "bare", False),
        ("core", "logallrefupdates", False),
        ("core", "ignorecase", False),
        ("core", "precomposeunicode", False),
        ("core", "symlinks", False),
        ("core", "autocrlf", False),
        ("core", "eol", False),
        ("core", "safecrlf", False),
        ("core", "quotepath", False),
        ("core", "abbrev", False),
        ("extensions", "objectformat", False),
        ("extensions", "worktreeconfig", False),
        ("extensions", "refstorage", False),
        ("remote", "url", True),
        ("remote", "pushurl", True),
        ("remote", "fetch", True),
        ("remote", "gh-resolved", True),
        ("branch", "remote", True),
        ("branch", "merge", True),
        ("branch", "rebase", True),
        ("user", "name", False),
        ("user", "email", False),
    }
)


class InspectRejected(Exception):
    """A request refused before any git or gh subprocess ran."""


@dataclass(frozen=True)
class InspectRequest:
    git: str
    gh: str | None
    tool: str
    argv: tuple[str, ...]


@dataclass(frozen=True)
class ExecResult:
    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""


# (argv, env, cwd, capture) -> ExecResult.  Tests substitute a recorder.
Executor = Callable[[Sequence[str], Mapping[str, str], str, bool], ExecResult]


def _subprocess_executor(
    argv: Sequence[str], env: Mapping[str, str], cwd: str, capture: bool
) -> ExecResult:
    command = list(argv)
    if command and os.path.basename(command[0]) == "git":
        # The inspect interpreter still imports only this stdlib-only module.
        # A pinned isolated helper applies the shared local Git boundary.
        helper = os.path.join(os.path.dirname(__file__), "inspect_git_subprocess.py")
        command = [sys.executable, "-I", "-S", helper, *command]
    completed = subprocess.run(
        command,
        env=dict(env),
        cwd=cwd,
        shell=False,
        stdin=subprocess.DEVNULL,
        capture_output=capture,
        check=False,
    )
    return ExecResult(
        completed.returncode,
        completed.stdout if capture else b"",
        completed.stderr if capture else b"",
    )


def _validate_pinned_path(option: str, value: str) -> str:
    if not value or not os.path.isabs(value):
        raise InspectRejected(f"{option} must name an absolute executable path.")
    target = os.path.realpath(value)
    try:
        info = os.stat(target)
    except OSError as exc:
        raise InspectRejected(f"{option} path {value} does not exist: {exc.strerror}.") from None
    if not stat.S_ISREG(info.st_mode) or not os.access(target, os.X_OK):
        raise InspectRejected(f"{option} path {value} is not an executable regular file.")
    return value


def parse_request(argv: Sequence[str]) -> InspectRequest:
    """Split the pinned leading options from the tool request."""
    tokens = list(argv)
    if not tokens or not tokens[0].startswith("--git="):
        raise InspectRejected("inspect requires a leading --git=<absolute path> option.")
    git = tokens.pop(0)[len("--git="):]
    gh: str | None = None
    if tokens and tokens[0].startswith("--gh="):
        gh = tokens.pop(0)[len("--gh="):]
    if any(token.startswith(("--git=", "--gh=", "--git", "--gh")) for token in tokens):
        raise InspectRejected(
            "--git= and --gh= are accepted only once each, in the leading positions."
        )
    _validate_pinned_path("--git", git)
    if gh is not None:
        _validate_pinned_path("--gh", gh)
    if not tokens:
        raise InspectRejected("inspect requires a git or gh subcommand.")
    tool = tokens.pop(0)
    if tool not in {"git", "gh"}:
        raise InspectRejected(f"inspect runs only git or gh, not {tool!r}.")
    if tool == "gh" and gh is None:
        raise InspectRejected("gh inspection is unavailable: no gh executable was pinned for this run.")
    return InspectRequest(git=git, gh=gh, tool=tool, argv=tuple(tokens))


def validate_format(value: str) -> None:
    """Accept only named formats or format strings from the closed placeholder set."""
    if value in NAMED_FORMATS:
        return
    for prefix in ("format:", "tformat:"):
        if value.startswith(prefix):
            body = value[len(prefix):]
            break
    else:
        raise InspectRejected(
            f"--format/--pretty value {value!r} is not allowed; use one of "
            f"{', '.join(sorted(NAMED_FORMATS))} or a format:/tformat: string."
        )
    index = 0
    while index < len(body):
        if body[index] != "%":
            index += 1
            continue
        rest = body[index + 1:]
        for token in FORMAT_PLACEHOLDERS:
            if rest.startswith(token):
                index += 1 + len(token)
                break
        else:
            raise InspectRejected(
                f"format placeholder %{rest[:2]!s} is not allowed; allowed placeholders: "
                + " ".join(f"%{token}" for token in FORMAT_PLACEHOLDERS)
                + "."
            )


def _positional(token: str) -> str:
    if token.startswith("-"):
        raise InspectRejected(f"option {token!r} is not allowed.")
    if "\x00" in token or "\n" in token:
        raise InspectRejected("arguments may not contain NUL or newline characters.")
    return token


def validate_git_args(argv: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    if not argv:
        raise InspectRejected("inspect git requires a subcommand.")
    sub, *rest = argv
    if sub not in GIT_SUBCOMMANDS:
        raise InspectRejected(
            f"git {sub!r} is not allowed; allowed: {', '.join(GIT_SUBCOMMANDS)}."
        )
    flags = _GIT_FLAGS[sub]
    out: list[str] = []
    index = 0
    while index < len(rest):
        token = rest[index]
        index += 1
        if token == "--":
            paths = rest[index:]
            if any("\x00" in path or "\n" in path for path in paths):
                raise InspectRejected("paths may not contain NUL or newline characters.")
            out += ["--", *paths]
            break
        if token in flags:
            out.append(token)
            continue
        if sub in _GIT_UNIFIED and (
            (token.startswith("-U") and _DIGITS_RE.match(token[2:]))
            or (token.startswith("--unified=") and _DIGITS_RE.match(token[len("--unified="):]))
        ):
            out.append(token)
            continue
        if sub in _GIT_FORMAT and token.startswith(("--format=", "--pretty=")):
            validate_format(token.split("=", 1)[1])
            out.append(token)
            continue
        if sub in _GIT_MAX_COUNT:
            if token.startswith("--max-count=") and _DIGITS_RE.match(token[len("--max-count="):]):
                out.append(token)
                continue
            if token == "-n":
                if index >= len(rest) or not _DIGITS_RE.match(rest[index]):
                    raise InspectRejected("-n requires a numeric count.")
                out += ["-n", rest[index]]
                index += 1
                continue
        out.append(_positional(token))
    return sub, tuple(out)


def validate_gh_args(argv: Sequence[str]) -> tuple[str, ...]:
    if len(argv) < 2 or (argv[0], argv[1]) not in _GH_FLAGS:
        allowed = ", ".join(" ".join(pair) for pair in GH_SUBCOMMANDS)
        rendered = " ".join(argv[:2]) or "(none)"
        raise InspectRejected(f"gh {rendered!r} is not allowed; allowed: {allowed}.")
    key = (argv[0], argv[1])
    flags = _GH_FLAGS[key]
    out: list[str] = [argv[0], argv[1]]
    rest = list(argv[2:])
    index = 0
    while index < len(rest):
        token = rest[index]
        index += 1
        if token in flags:
            out.append(token)
            continue
        if token in _GH_VALUED:
            if index >= len(rest):
                raise InspectRejected(f"{token} requires a value.")
            value = rest[index]
            index += 1
            pattern = _GH_REPO_RE if token == "--repo" else _GH_JSON_RE
            if not pattern.match(value):
                raise InspectRejected(f"{token} value {value!r} is not allowed.")
            out += [token, value]
            continue
        out.append(_positional(token))
    return tuple(out)


def build_subprocess_env(
    environ: Mapping[str, str], pinned_dirs: Sequence[str]
) -> dict[str, str]:
    """Build the closed git/gh environment from an empty mapping."""
    env: dict[str, str] = {}
    for key, value in environ.items():
        if key in ENV_ALLOWLIST or key.startswith(ENV_ALLOWLIST_PREFIXES):
            env[key] = value
    directories: list[str] = []
    for directory in pinned_dirs:
        if directory and directory not in directories:
            directories.append(directory)
    env["PATH"] = os.pathsep.join(directories)
    env.update(FORCED_ENV)
    return env


def git_argv(git: str, sub: str, args: Sequence[str]) -> list[str]:
    argv = [git, "--no-pager"]
    for item in FORCED_GIT_CONFIG:
        argv += ["-c", item]
    argv.append(sub)
    if sub in _GIT_DIFF_RENDERING:
        argv += ["--no-ext-diff", "--no-textconv"]
    if sub in _GIT_SUBMODULE_IGNORING:
        # The command-line option outranks submodule.<name>.ignore from
        # .gitmodules, which the config gate does not scan.
        argv.append("--ignore-submodules=all")
    argv += list(args)
    return argv


def config_gate_argv(git: str) -> list[str]:
    argv = [git]
    for item in _TRACE2_OVERRIDES:
        argv += ["-c", item]
    argv += ["config", "--list", "--null", "--show-origin", "--show-scope", "--includes"]
    return argv


def _config_key_allowed(key: str, value: str) -> bool:
    parts = key.split(".")
    if len(parts) < 2:
        return False
    section, variable = parts[0].lower(), parts[-1].lower()
    has_subsection = len(parts) > 2
    if (section, variable, has_subsection) not in CONFIG_KEY_ALLOWLIST:
        return False
    if (section, variable) == ("core", "bare") and value.strip().lower() != "false":
        return False
    return True


def parse_config_listing(payload: bytes) -> list[tuple[str, str, str, str]]:
    """Parse ``--null --show-scope --show-origin`` output into (scope, origin, key, value)."""
    fields = payload.split(b"\x00")
    if fields and fields[-1] == b"":
        fields.pop()
    if len(fields) % 3:
        raise InspectRejected("repository config listing could not be parsed; refusing to run.")
    entries: list[tuple[str, str, str, str]] = []
    for index in range(0, len(fields), 3):
        scope, origin, item = (field.decode("utf-8", "replace") for field in fields[index:index + 3])
        key, _, value = item.partition("\n")
        entries.append((scope, origin, key, value))
    return entries


def check_repository_config(entries: Sequence[tuple[str, str, str, str]]) -> None:
    for scope, origin, key, value in entries:
        if scope == "command":
            continue
        if not _config_key_allowed(key, value):
            raise InspectRejected(
                f"repository git config key {key!r} from {origin or scope} is outside the "
                "inspect allowlist; refusing to run git or gh in this checkout. Remove the "
                "key, read files directly with Read/Grep/Glob, or run without "
                "--agent-permissions sandboxed."
            )


def run_config_gate(git: str, env: Mapping[str, str], cwd: str, execute: Executor) -> None:
    """Refuse (``InspectRejected``) unless the checkout's config is allowlisted."""
    gate = execute(config_gate_argv(git), env, cwd, True)
    if gate.returncode != 0:
        detail = gate.stderr.decode("utf-8", "replace").strip()
        raise InspectRejected(
            "repository config gate could not read git config; refusing to run"
            + (f": {detail}" if detail else ".")
        )
    check_repository_config(parse_config_listing(gate.stdout))


def run_hardened_git(
    git: str,
    sub: str,
    args: Sequence[str],
    *,
    cwd: str,
    environ: Mapping[str, str] | None = None,
    executor: Executor | None = None,
) -> ExecResult:
    """Run one fixed git probe with inspect's gate, closed env, and forced config.

    For agent-loop's own probes of a shared checkout (for example the
    workdir snapshot before a sandboxed turn); ``sub`` and ``args`` come from
    agent-loop, not from an agent, so they bypass the argument allowlist.
    """
    execute = executor or _subprocess_executor
    env = build_subprocess_env(os.environ if environ is None else environ, [os.path.dirname(git)])
    run_config_gate(git, env, cwd, execute)
    return execute(git_argv(git, sub, args), env, cwd, True)


def run_inspect(
    argv: Sequence[str],
    *,
    environ: Mapping[str, str] | None = None,
    cwd: str | None = None,
    executor: Executor | None = None,
    stdout=None,
    stderr=None,
) -> int:
    values = os.environ if environ is None else environ
    execute = executor or _subprocess_executor
    out = stdout or sys.stdout.buffer
    err = stderr or sys.stderr
    try:
        request = parse_request(argv)
        if request.tool == "git":
            sub, args = validate_git_args(request.argv)
            command = git_argv(request.git, sub, args)
        else:
            assert request.gh is not None
            command = [request.gh, *validate_gh_args(request.argv)]
        pinned_dirs = [os.path.dirname(request.git)]
        if request.gh is not None:
            pinned_dirs.append(os.path.dirname(request.gh))
        env = build_subprocess_env(values, pinned_dirs)
        workdir = cwd or values.get("AGENT_LOOP_WORKDIR") or os.getcwd()
        run_config_gate(request.git, env, workdir, execute)
    except InspectRejected as exc:
        print(f"agent-loop inspect: rejected: {exc}", file=err)
        return EXIT_REJECTED
    result = execute(command, env, workdir, True)
    if result.stdout:
        out.write(result.stdout)
        out.flush()
    if result.stderr:
        err.write(result.stderr.decode("utf-8", "replace"))
    return result.returncode


def main(argv: Sequence[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] == "inspect":
        tokens = tokens[1:]
    return run_inspect(tokens)
