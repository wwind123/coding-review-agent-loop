"""Validation helpers for keeping coder-reported work inside the assigned checkout."""

from __future__ import annotations

import fnmatch
import glob
import ipaddress
import re
import shlex
from itertools import islice
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Literal, Sequence
from urllib.parse import urlsplit

from .errors import AgentLoopError, ReportWorkdirRefusal
from .scratch import scratch_root
from .protocol_discuss_types import (
    DiscussEvidenceClaim,
)
from .test_runtime import (
    ManagedWrapperTraversal,
    TestRuntimeConfigurationError,
    managed_wrapper_traversal,
    parse_managed_test_command,
)


TEST_SECTION_RE = re.compile(r"(?im)^\s*tests(?:\s+run)?\s*:\s*(?P<body>.*)$")
WINDOWS_PATH_RE = re.compile(r"(?<![\w.-])[A-Za-z]:\\[^\s`'\"|;&)<>]+")
URL_SCHEME_RE = re.compile(r"(?i)https?://")
# Stops at a comma and refuses to consume into a subsequent http(s) scheme, so
# a concatenated/delimited string such as "http://127.0.0.1/foo,http://evil.com"
# yields two separate URL values instead of one that only reveals its first host.
URL_VALUE_RE = re.compile(r"(?i)https?://(?:(?!https?://)[^\s'\"`;&|()<>,])+")
BACKTICK_SPAN_RE = re.compile(r"`([^`]*)`")
VAR_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
INTERPRETER_BASENAME_RE = re.compile(r"^(python|pypy)\d*(\.\d+)?$")

# Origin of a reported test-command string.
Origin = Literal["structured", "response"]

# These probes are deliberately kept in the workdir guard module so the
# Claude recovery path and the coder-reported-PR HEAD guard share the same
# read-only Git command contract without sharing exception policy.
GitProbeKind = Literal["available", "command-failed", "blank", "exception"]


@dataclass(frozen=True)
class GitProbeResult:
    """The result of one bounded, read-only Git probe."""

    command: tuple[str, ...]
    kind: GitProbeKind
    value: str | None = None
    returncode: int | None = None
    exception_type: str | None = None
    detail: str | None = None

    @property
    def available(self) -> bool:
        return self.kind == "available"


@dataclass(frozen=True)
class WorkdirSnapshot:
    """HEAD and porcelain status captured from one assigned checkout."""

    head: GitProbeResult
    status: GitProbeResult

    @property
    def available(self) -> bool:
        return self.head.available and self.status.available


@dataclass(frozen=True)
class WorkdirReplayEvidence:
    """Replacement evidence plus an optional fail-closed replay refusal."""

    reason: str
    replay_refusal_kind: str | None = None
    replay_refusal_detail: str | None = None


def _workdir_probe_label(snapshot: WorkdirSnapshot) -> str:
    def describe(probe: GitProbeResult) -> str:
        if probe.kind == "available":
            return repr(probe.value)
        if probe.kind == "exception":
            return f"{probe.exception_type}: {probe.detail}"
        if probe.kind == "command-failed":
            return f"returncode={probe.returncode!r}"
        return "blank"

    return f"HEAD={describe(snapshot.head)}, status={describe(snapshot.status)}"


def gate_workdir_replay(
    provider_label: str,
    reason: str,
    *,
    before_snapshot: WorkdirSnapshot | None,
    after_snapshot: WorkdirSnapshot | None,
) -> WorkdirReplayEvidence:
    """Allow replay only when both checkout snapshots are available and equal.

    The provider label is deliberately supplied by each backend so refusal
    diagnostics identify the actual backend while the metadata contract stays
    provider-neutral.
    """
    if before_snapshot is None and after_snapshot is None:
        return WorkdirReplayEvidence(reason)
    if before_snapshot is None or not before_snapshot.available:
        detail = (
            f"{provider_label} replay refused: before workdir snapshot unavailable "
            f"({ _workdir_probe_label(before_snapshot) if before_snapshot else 'missing'})."
        )
        return WorkdirReplayEvidence(reason, "before-unavailable", detail)
    if after_snapshot is None or not after_snapshot.available:
        detail = (
            f"{provider_label} replay refused: after workdir snapshot unavailable "
            f"({ _workdir_probe_label(after_snapshot) if after_snapshot else 'missing'})."
        )
        return WorkdirReplayEvidence(reason, "after-unavailable", detail)
    if before_snapshot.head.value != after_snapshot.head.value:
        detail = (
            f"{provider_label} replay refused: assigned checkout HEAD changed during "
            f"invocation (before={before_snapshot.head.value!r}, "
            f"after={after_snapshot.head.value!r})."
        )
        return WorkdirReplayEvidence(reason, "changed-head", detail)
    if before_snapshot.status.value != after_snapshot.status.value:
        detail = (
            f"{provider_label} replay refused: dirty workdir status changed during "
            f"invocation (before={before_snapshot.status.value!r}, "
            f"after={after_snapshot.status.value!r})."
        )
        return WorkdirReplayEvidence(reason, "changed-status", detail)
    return WorkdirReplayEvidence(reason)


# (args, workdir) -> an object with returncode/stdout/stderr, replacing the
# runner's bare ``git`` for probes that must not honor checkout config.
GitProbeRunner = Callable[[tuple[str, ...], Path], object]


def _git_probe(
    runner: object,
    *,
    workdir: Path,
    args: tuple[str, ...],
    tolerate_exceptions: bool,
    git_runner: GitProbeRunner | None = None,
) -> GitProbeResult:
    command = ("git", *args)
    try:
        if git_runner is not None:
            result = git_runner(args, workdir)
        else:
            result = runner.run(command, cwd=workdir, check=False)  # type: ignore[attr-defined]
    except (AgentLoopError, OSError) as exc:
        if not tolerate_exceptions:
            raise
        return GitProbeResult(
            command=command,
            kind="exception",
            exception_type=type(exc).__name__,
            detail=str(exc),
        )
    if result.returncode != 0:
        return GitProbeResult(
            command=command,
            kind="command-failed",
            returncode=result.returncode,
            detail=result.stderr or None,
        )
    if args == ("rev-parse", "HEAD"):
        value = result.stdout.strip()
        if not value:
            return GitProbeResult(command=command, kind="blank", returncode=result.returncode)
        return GitProbeResult(
            command=command,
            kind="available",
            value=value,
            returncode=result.returncode,
        )
    # `git status --porcelain` uses an empty stdout as the authoritative clean
    # worktree state. Preserve all non-empty bytes so dirty-status comparisons
    # remain exact and do not normalize away meaningful changes.
    return GitProbeResult(
        command=command,
        kind="available",
        value=result.stdout,
        returncode=result.returncode,
    )


def read_workdir_head(
    runner: object,
    workdir: Path,
    *,
    tolerate_exceptions: bool = False,
    git_runner: GitProbeRunner | None = None,
) -> GitProbeResult:
    """Read only ``git rev-parse HEAD`` from ``workdir``."""
    return _git_probe(
        runner,
        workdir=workdir,
        args=("rev-parse", "HEAD"),
        tolerate_exceptions=tolerate_exceptions,
        git_runner=git_runner,
    )


def capture_workdir_snapshot(
    runner: object,
    workdir: Path,
    *,
    tolerate_exceptions: bool = False,
    git_runner: GitProbeRunner | None = None,
) -> WorkdirSnapshot:
    """Capture HEAD and ``git status --porcelain`` without mutating the checkout.

    ``git_runner`` replaces the bare ``git`` (sandboxed mode passes the
    hardened, config-gated runner so a coder-planted hook or filter in a
    shared checkout never runs before a turn).
    """
    return WorkdirSnapshot(
        head=read_workdir_head(
            runner,
            workdir,
            tolerate_exceptions=tolerate_exceptions,
            git_runner=git_runner,
        ),
        status=_git_probe(
            runner,
            workdir=workdir,
            args=("status", "--porcelain"),
            tolerate_exceptions=tolerate_exceptions,
            git_runner=git_runner,
        ),
    )

# ---------------------------------------------------------------------------
# Toolchain / runner recognition (path pass, program-role tokens only).
# ---------------------------------------------------------------------------

RUNNER_BASENAMES = {
    "python", "pypy", "pytest", "py.test", "tox", "nox", "node", "npm", "npx",
    "yarn", "pnpm", "deno", "ruby", "bundle", "rake", "go", "cargo", "java",
    "gradle", "mvn", "dotnet", "make", "uv", "poetry", "pipenv", "pdm",
    "hatch", "sh", "bash", "zsh",
}

# Network clients that count as command heads for URL classification purposes.
NETWORK_CLIENT_BASENAMES = {"curl", "wget", "http", "httpie"}

WRAPPER_PROGRAMS = {
    "env", "sudo", "nohup", "nice", "time", "timeout", "stdbuf", "command", "xargs",
}

_EXACT_TOOLCHAIN_DIRS = {
    "/bin", "/sbin", "/usr/bin", "/usr/sbin", "/usr/local/bin", "/usr/local/sbin",
    "/opt/homebrew/bin", "/opt/local/bin", "/snap/bin",
}

_TOOLCHAIN_PARENT_RES = [
    re.compile(r"(^|/)(\.venv|venv|\.env|env|virtualenv)/(bin|Scripts)$"),
    re.compile(r"(^|/)node_modules/\.bin$"),
    re.compile(r"(^|/)\.pyenv/versions/[^/]+/bin$"),
    re.compile(r"(^|/)\.rbenv/versions/[^/]+/bin$"),
    re.compile(r"(^|/)\.nvm/versions/node/[^/]+/bin$"),
    re.compile(r"(^|/)(conda|miniconda3|anaconda3)/bin$"),
    re.compile(r"(^|/)(conda|miniconda3|anaconda3)/envs/[^/]+/bin$"),
    re.compile(r"(^|/)nix/store/[^/]+/bin$"),
]

# Working-directory flags: strictly validated, no exemption ever applies.
WORKDIR_FLAGS = {"cd", "pushd", "-C", "--directory", "--chdir", "--cwd", "--rootdir"}
WORKDIR_FLAG_PREFIXES = ("--directory=", "--chdir=", "--cwd=", "--rootdir=")

# interpreter_value role: flags/env-vars whose *value* names an interpreter or
# library/toolchain path rather than a test location.
INTERPRETER_VALUE_FLAGS = {"--python", "--interpreter", "--with-python", "--python-executable"}
INTERPRETER_VALUE_ENV_VARS = {"PYTHONPATH", "VIRTUAL_ENV", "NODE_PATH", "JAVA_HOME", "PATH"}

# Report destinations are not execution inputs.  Keep this vocabulary
# deliberately explicit so an arbitrary outside path cannot be hidden behind
# a generic option.
OUTPUT_VALUE_FLAGS = {
    "--output",
    "--output-dir",
    "--output-file",
    "--report",
    "--report-file",
}
OUTPUT_VALUE_FLAG_PREFIXES = tuple(f"{flag}=" for flag in OUTPUT_VALUE_FLAGS)

# ---------------------------------------------------------------------------
# Package acquisition (URL pass exclusion).
# ---------------------------------------------------------------------------

DIRECT_PACKAGE_MANAGERS = {
    "pip", "pip3", "uv", "poetry", "pipenv", "npm", "yarn", "pnpm", "gem",
    "bundle", "cargo", "go", "apt", "apt-get", "brew", "git",
}
MODULE_PACKAGE_MANAGERS = {"pip", "uv", "poetry", "pipenv", "ensurepip", "installer", "build"}
ACQUISITION_SUBCOMMANDS = {
    "install", "add", "download", "wheel", "get", "fetch", "clone", "pull", "sync", "restore",
}

# ---------------------------------------------------------------------------
# URL attachment (narrative / prose) vocabulary.
# ---------------------------------------------------------------------------

EXECUTION_VERBS = {
    "ran", "run", "runs", "running", "executed", "executing", "invoked", "invoking",
    "hit", "hitting", "curled", "pinged", "queried", "tested", "retested",
    "exercised", "smoke-tested",
}
TARGET_PREPOSITIONS = {"against", "at", "on", "onto", "to", "via", "toward", "towards", "targeting"}
NEGATION_TOKENS = {
    "not", "never", "no", "none", "without", "skipped", "skipping", "avoided", "unable",
}
DETERMINERS = {"the", "a", "our"}

# Windows-specific toolchain recognition (used only for the WINDOWS_PATH_RE branch).
_WINDOWS_RUNNER_BASENAMES = {"python.exe", "python3.exe", "pytest.exe", "py.exe"}
_WINDOWS_TOOLCHAIN_PARENT_RES = [
    re.compile(r"(?i)\\python\d*(\.\d+)?$"),
    re.compile(r"(?i)\\scripts$"),
    re.compile(r"(?i)\\node_modules\\\.bin$"),
]


def _canonical(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _is_inside(path: Path, assigned_workdir: Path) -> bool:
    try:
        path.relative_to(assigned_workdir)
        return True
    except ValueError:
        return False


def _normalize_reported_path(raw_path: str) -> Path | None:
    cleaned = raw_path.strip().strip("`'\".,")
    if not cleaned:
        return None
    if cleaned.startswith("$HOME/"):
        cleaned = str(Path.home()) + cleaned[len("$HOME") :]
    if cleaned.startswith("~/") or cleaned.startswith("/"):
        return _canonical(Path(cleaned))
    if WINDOWS_PATH_RE.fullmatch(cleaned):
        return Path(cleaned)
    return None


# ---------------------------------------------------------------------------
# Token-level predicates shared by the path pass and the URL pass.
# ---------------------------------------------------------------------------


def _strip_wrap(token: str) -> str:
    return token.strip("`'\"")


def _is_url_token(token: str) -> bool:
    return bool(URL_SCHEME_RE.search(_strip_wrap(token)))


def _url_values(token: str) -> tuple[str, ...]:
    return tuple(
        match.group(0).rstrip(".,")
        for match in URL_VALUE_RE.finditer(_strip_wrap(token))
    )


def _has_unmatched_url_scheme(token: str) -> bool:
    """True if a `https?://` occurrence in ``token`` is not the start of any
    `URL_VALUE_RE` match.

    This catches both a bare/incomplete scheme with no host at all
    (`_url_values` yields nothing for it, e.g. a lone "http://") and a
    scheme immediately followed by a *nested* scheme (e.g.
    "http://https://localhost"): the outer "http://" can supply zero
    characters to `URL_VALUE_RE`'s mandatory `+` group because the negative
    lookahead refuses to let it consume into the nested "https://", so the
    only match `_url_values` finds is the *inner* URL. Evaluating just that
    inner fragment for loopback-ness (as opposed to the outer scheme's own,
    unrelated host) would let a non-loopback outer target hide behind a
    loopback-looking inner fragment.
    """
    text = _strip_wrap(token)
    matched_starts = {match.start() for match in URL_VALUE_RE.finditer(text)}
    return any(
        match.start() not in matched_starts for match in URL_SCHEME_RE.finditer(text)
    )


def _is_loopback_url(url: str) -> bool:
    # WHATWG URL consumers (Node, browsers, and browser-driven E2E clients)
    # treat a backslash as a path separator for special schemes like
    # http(s), so "http://live.example\@localhost" resolves to live.example
    # there even though `urlsplit` reports "localhost" as the hostname.
    # Refuse to classify authority syntax that could disagree between
    # parsers instead of trusting `urlsplit` alone.
    if "\\" in url:
        return False
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return False
    if host is None:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_path_like_token(token: str) -> bool:
    if _is_url_token(token):
        return False
    cleaned = _strip_wrap(token)
    if not cleaned:
        return False
    return "/" in cleaned or "\\" in cleaned or cleaned.startswith("~") or cleaned.startswith("$HOME/")


def _is_path_shaped(token: str) -> bool:
    return token.startswith("/") or token.startswith("~/") or token == "~" or token.startswith("$HOME/")


def _word(token: str) -> str:
    return token.strip("`'\".,;:!?()").lower()


def _is_bare_word(token: str) -> bool:
    if not token:
        return False
    if token.startswith("-"):
        return False
    if VAR_ASSIGNMENT_RE.match(token):
        return False
    if _is_path_like_token(token) or _is_url_token(token):
        return False
    return True


def _is_negation_word(word: str) -> bool:
    return word in NEGATION_TOKENS or word.endswith("n't")


def _is_runner_basename(name: str) -> bool:
    if name in RUNNER_BASENAMES:
        return True
    return bool(INTERPRETER_BASENAME_RE.match(name))


def _is_command_shaped(token: str) -> bool:
    if _is_path_like_token(token):
        return True
    name = Path(_strip_wrap(token)).name.rstrip(".,;:!?")
    return _is_runner_basename(name) or name in NETWORK_CLIENT_BASENAMES


def _is_toolchain_executable(token: str) -> bool:
    cleaned = _strip_wrap(token).strip(".,")
    if not cleaned:
        return False
    p = Path(cleaned)
    name = p.name
    if _is_runner_basename(name):
        return True
    parent_str = str(p.parent)
    if parent_str in _EXACT_TOOLCHAIN_DIRS:
        return True
    for pattern in _TOOLCHAIN_PARENT_RES:
        if pattern.search(parent_str):
            return True
    return False


def _is_windows_toolchain_executable(token: str) -> bool:
    cleaned = _strip_wrap(token).strip(".,")
    if not cleaned:
        return False
    name = cleaned.rsplit("\\", 1)[-1].lower()
    if name in _WINDOWS_RUNNER_BASENAMES:
        return True
    parent = cleaned.rsplit("\\", 1)[0] if "\\" in cleaned else ""
    return any(pattern.search(parent) for pattern in _WINDOWS_TOOLCHAIN_PARENT_RES)


_WRAPPER_VALUE_OPTIONS = {
    "timeout": {"-k", "--kill-after", "-s", "--signal"},
    "sudo": {"-u", "-g"},
    "nice": {"-n"},
    "stdbuf": {"-i", "-o", "-e"},
    "env": {"-u", "--unset", "-C", "--chdir"},
    "xargs": {"-n", "-P", "-I"},
}

_TIMEOUT_DURATION_RE = re.compile(
    r"(?P<number>[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?P<suffix>[smhd])?$",
    re.ASCII,
)


def _program_basename(token: str) -> str:
    """Return a platform-neutral executable basename without treating /foo as an option."""
    return _strip_wrap(token).rstrip(".,;:!?").replace("\\", "/").rsplit("/", 1)[-1].lower()


def _consume_wrapper_options(tokens: Sequence[str], start: int, wrapper: str) -> int | None:
    """Consume GNU-style wrapper options, preserving unknown options as valueless."""
    i = start
    value_options = _WRAPPER_VALUE_OPTIONS.get(wrapper, set())
    while i < len(tokens):
        token = tokens[i]
        if token == "--":
            return i + 1
        # Deliberately do not recognize slash-prefixed Windows tokens as options.
        if not token.startswith("-") or _is_path_like_token(token):
            return i
        option, sep, option_value = token.partition("=")
        if option in value_options:
            if sep:
                i += 1
                continue
            # Attached short values, such as -k10s or -n10, are valid.
            if option.startswith("-") and not option.startswith("--") and len(token) > len(option):
                i += 1
                continue
            if i + 1 >= len(tokens):
                return None
            i += 2
            continue
        if token.startswith("-") and not token.startswith("--"):
            attached_option = next(
                (known for known in value_options if token.startswith(known) and len(token) > len(known)),
                None,
            )
            if attached_option is not None:
                i += 1
                continue
        # Ordinary classification retains its historical permissive traversal.
        i += 1
    return i


def _parse_wrapper(tokens: Sequence[str], index: int, wrapper: str) -> int | None:
    next_index = _consume_wrapper_options(tokens, index + 1, wrapper)
    if next_index is None:
        return None
    if wrapper == "timeout":
        if next_index >= len(tokens) or not _TIMEOUT_DURATION_RE.fullmatch(tokens[next_index]):
            return None
        next_index += 1
    if next_index >= len(tokens):
        return None
    return next_index


def _wrapper_traversal(
    tokens: Sequence[str], *, managed: bool = False,
) -> ManagedWrapperTraversal:
    if managed:
        return managed_wrapper_traversal(tokens)
    positions: set[int] = set()
    i = 0
    recognized_prefix = False
    assignments_allowed = True
    while i < len(tokens):
        if VAR_ASSIGNMENT_RE.match(tokens[i]) and (not managed or assignments_allowed):
            i += 1
            continue
        wrapper = _program_basename(tokens[i])
        if wrapper not in WRAPPER_PROGRAMS:
            positions.add(i)
            return ManagedWrapperTraversal(recognized_prefix, positions, i)
        recognized_prefix = True
        positions.add(i)
        next_index = _parse_wrapper(tokens, i, wrapper)
        if next_index is None:
            return ManagedWrapperTraversal(True, positions, None)
        i = next_index
        # Shell assignments are transparent before the initial command, and
        # env accepts them as operands before its command.  Other execution
        # prefixes treat an assignment-shaped operand as the program to run.
        assignments_allowed = wrapper == "env"
    return ManagedWrapperTraversal(recognized_prefix, positions, None)


def _program_position_indices(tokens: Sequence[str]) -> tuple[set[int], int | None]:
    traversal = _wrapper_traversal(tokens)
    return traversal.program_positions, traversal.effective_head_index


def _effective_head_index(tokens: Sequence[str]) -> int | None:
    return _wrapper_traversal(tokens).effective_head_index


# ---------------------------------------------------------------------------
# Clause splitting.
# ---------------------------------------------------------------------------

ClauseMode = Literal["structured", "code", "narrative"]
_SHELL_PUNCTUATION = ";&|()"
_SHELL_PUNCTUATION_CHARS = frozenset(_SHELL_PUNCTUATION)


@dataclass
class _Clause:
    tokens: list[str]
    mode: ClauseMode
    command_by_contract: bool
    # Where a working-directory change stops applying: the nesting level of a
    # shell -c script, and whether the text has a subshell or substitution.
    shell_depth: int = 0
    in_subshell: bool = False
    # A ``sh -c`` launcher (or one of its wrappers) that is not a PATH name or
    # system toolchain program: the script's measurements are never plain.
    launcher_untrusted: bool = False


def _posix_tokens(text: str) -> list[str]:
    """Shell-lex ``text``; raises ``ValueError`` for malformed quoting."""
    # ``shlex.split`` only separates operators surrounded by whitespace.
    # A shell does not impose that restriction, so use punctuation-aware
    # lexing to keep adjacent commands from hiding in an argument token.
    lexer = shlex.shlex(text, posix=True, punctuation_chars=_SHELL_PUNCTUATION)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def _lexes_cleanly(text: str) -> bool:
    """Whether :func:`_tokenize` can lex ``text`` without its fallback (#2052)."""
    try:
        _posix_tokens(text)
    except ValueError:
        return False
    return True


def _tokenize(text: str) -> list[str]:
    try:
        return _posix_tokens(text)
    except ValueError:
        # The fallback is observable: _validate_command_contents marks a
        # display-partition entry it applied to as parse-failed (#2052).
        return text.split()


def _split_into_clauses(text: str, mode: ClauseMode) -> list[_Clause]:
    tokens = _tokenize(text)
    in_subshell = any(
        token and set(token) <= _SHELL_PUNCTUATION_CHARS and ("(" in token or ")" in token)
        for token in tokens
    )
    clauses: list[_Clause] = []
    current: list[str] = []
    command_by_contract = True

    def flush(next_by_contract: bool) -> None:
        nonlocal current, command_by_contract
        if current:
            clauses.append(
                _Clause(
                    tokens=current,
                    mode=mode,
                    command_by_contract=command_by_contract if mode == "structured" else False,
                    in_subshell=in_subshell,
                )
            )
        current = []
        command_by_contract = next_by_contract

    for token in tokens:
        if token and set(token) <= _SHELL_PUNCTUATION_CHARS:
            flush(True)
            continue
        current.append(token)
        if not token:
            continue
        last_char = token[-1]
        if last_char in ".!?;" and token not in {".", ".."} and not _is_path_like_token(token):
            flush(False)
        elif mode == "narrative" and last_char == "," and not _is_path_like_token(token):
            flush(False)
    flush(False)
    return clauses


def _segments_for_entry(command: str, origin: Origin) -> list[tuple[str, ClauseMode]]:
    """Return (text, mode) segments; mode is 'structured', 'code', or 'narrative'."""
    if origin == "structured":
        return [(command, "structured")]
    segments: list[tuple[str, ClauseMode]] = []
    last_end = 0
    for match in BACKTICK_SPAN_RE.finditer(command):
        if match.start() > last_end:
            segments.append((command[last_end : match.start()], "narrative"))
        segments.append((match.group(1), "code"))
        last_end = match.end()
    if last_end < len(command):
        segments.append((command[last_end:], "narrative"))
    return segments


def _clauses_for_entry(command: str, origin: Origin) -> list[_Clause]:
    clauses: list[_Clause] = []
    for text, mode in _segments_for_entry(command, origin):
        for clause in _split_into_clauses(text, mode):
            clauses.extend(_expand_shell_clause(clause))
    return clauses


def _shell_command_index(tokens: Sequence[str]) -> int | None:
    head = _effective_head_index(tokens)
    if head is None or _program_basename(tokens[head]) not in {"sh", "bash", "zsh"}:
        return None
    index = head + 1
    while index < len(tokens):
        option = tokens[index]
        if option in {"--login", "--noprofile", "--norc"}:
            index += 1
            continue
        if re.fullmatch(r"-[elux]*o", option):
            # ``-o`` takes an option name, including when combined with
            # operand-free flags as in ``-euo pipefail``.
            if index + 1 >= len(tokens):
                raise AgentLoopError("Cannot validate shell -o option without its operand.")
            index += 2
            continue
        if re.fullmatch(r"-[celux]+", option):
            if "c" in option:
                return index + 1 if index + 1 < len(tokens) else None
            index += 1
            continue
        if not option.startswith("-"):
            # A shell operand is normally a script path. A quoted command-like
            # blob is neither a safely identifiable script path nor an explicit
            # command-string invocation, so refuse it instead of allowing paths
            # embedded in the blob to bypass token-level validation.
            if any(character.isspace() for character in option):
                raise AgentLoopError("Cannot validate multi-word shell script operand.")
            return None
        if re.fullmatch(r"-[A-Za-z]+", option) and "c" in option:
            # ``c`` makes the next operand executable text, even when it is
            # grouped with a shell flag that this scanner does not support.
            # Refuse the ambiguous cluster instead of treating that text as an
            # ordinary non-path argument and skipping its embedded paths.
            raise AgentLoopError(
                "Cannot validate shell command string after an unsupported shell option."
            )
        if any(re.fullmatch(r"-[A-Za-z]*c[A-Za-z]*", later) for later in tokens[index + 1 :]):
            raise AgentLoopError(
                "Cannot validate shell command string after an unsupported shell option."
            )
        return None
    return None


def _expand_shell_clause(clause: _Clause, depth: int = 0) -> list[_Clause]:
    index = _shell_command_index(clause.tokens)
    if index is None:
        return [clause]
    if depth >= 8:
        raise AgentLoopError("Test report shell-command nesting exceeds the validation limit.")
    script = clause.tokens[index]
    try:
        shlex.split(script)
    except ValueError as exc:
        raise AgentLoopError("Cannot validate malformed quoted shell command in test report.") from exc
    # Keep the launcher, environment, and positional arguments under the normal
    # checks. Only the -c operand is executable text rather than a path argument.
    outer = _Clause(
        tokens=clause.tokens[:index] + clause.tokens[index + 1 :],
        mode=clause.mode,
        command_by_contract=clause.command_by_contract,
        shell_depth=clause.shell_depth,
        in_subshell=clause.in_subshell,
    )
    expanded = [outer]
    for inner in _split_into_clauses(script, "structured"):
        inner = replace(
            inner,
            shell_depth=clause.shell_depth + 1,
            in_subshell=inner.in_subshell or clause.in_subshell,
        )
        expanded.extend(_expand_shell_clause(inner, depth + 1))
    launcher_untrusted = clause.launcher_untrusted or not _launch_is_plain(
        clause.tokens, *_program_position_indices(clause.tokens)
    )
    if launcher_untrusted:
        expanded[1:] = [replace(inner, launcher_untrusted=True) for inner in expanded[1:]]
    return expanded


# ---------------------------------------------------------------------------
# Path pass: assign a role to every path-shaped token in a clause.
# ---------------------------------------------------------------------------


def _grep_pattern_indices(tokens: Sequence[str], head: int | None) -> set[int]:
    """Find grep patterns so a slash in a regex is not mistaken for a path."""
    if head is None:
        return set()
    program = _program_basename(tokens[head])
    start = head + 1
    if program == "git":
        while start < len(tokens):
            token = tokens[start]
            if token in {"-C", "-c", "--git-dir", "--work-tree"}:
                start += 2
            elif token.startswith("-"):
                start += 1
            else:
                break
        if start >= len(tokens) or tokens[start] != "grep":
            return set()
        start += 1
    elif program not in {"grep", "egrep", "fgrep"}:
        return set()

    patterns: set[int] = set()
    has_pattern = False
    index = start
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            # Options end here. If no option supplied a pattern, the next
            # positional token is the pattern; later tokens are file operands.
            if not has_pattern and index + 1 < len(tokens):
                patterns.add(index + 1)
            break
        if token in {"-e", "--regexp"}:
            if index + 1 >= len(tokens):
                break
            patterns.add(index + 1)
            has_pattern = True
            index += 2
            continue
        if token.startswith("--regexp=") or re.match(r"^-[A-Za-z]*e.+", token):
            patterns.add(index)
            has_pattern = True
            index += 1
            continue
        if token in {"-f", "--file"}:
            has_pattern = True
            index += 2
            continue
        if token.startswith("--file=") or re.match(r"^-[A-Za-z]*f.+", token):
            has_pattern = True
            index += 1
            continue
        if token in {
            "-m", "--max-count", "-A", "--after-context", "-B", "--before-context",
            "-C", "--context", "-D", "--devices", "-d", "--directories",
            "--binary-files", "--exclude", "--exclude-dir",
            "--include", "--label", "--group-separator", "--threads",
        }:
            # These options consume the next token, which cannot be the
            # positional pattern. Keep that token subject to path checks.
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        if not has_pattern:
            patterns.add(index)
            has_pattern = True
        index += 1
    return patterns


def _scratch_option_values(tokens: Sequence[str], head: int | None) -> dict[int, str]:
    """Eligible occurrences only; never grant a path-wide exemption."""
    root = scratch_root()
    confined = root / "scratch"
    values: dict[int, str] = {}
    try:
        if confined.resolve() != root.parent.resolve() / root.name / "scratch":
            return values
        index = (head + 1) if head is not None else len(tokens)
        while index < len(tokens):
            token = tokens[index]
            if token == "--":
                break
            flag, equals, value = token.partition("=")
            guarded = (
                _VALUE_TAKING_OPTIONS | WORKDIR_FLAGS | INTERPRETER_VALUE_FLAGS
                | OUTPUT_VALUE_FLAGS | _BOOLEAN_LONG_FLAGS
            )
            if token.startswith("--") and flag not in guarded:
                value_index = index if equals else index + 1
                if not equals and value_index < len(tokens):
                    value = tokens[value_index]
                    if value.startswith("-"):
                        index += 1
                        continue
                path = Path(value).expanduser()
                if (
                    value and path.is_absolute() and ".." not in path.parts
                    and path.is_relative_to(confined)
                    and path.resolve().is_relative_to(confined.resolve())
                ):
                    values[value_index] = value
                if not equals:
                    index += 1
            index += 1
    except (OSError, RuntimeError, ValueError):
        return {}
    return values


def _npm_forwarding_separator(tokens: Sequence[str], head: int | None) -> int | None:
    """Recognize npm run/run-script NAME -- after known wrappers.

    A second separator belongs to the script and keeps terminator semantics.
    Unrecognized npm layouts and other package managers remain strict.
    """
    if head is None or len(tokens) <= head + 3:
        return None
    if (_program_basename(tokens[head]) == "npm"
            and tokens[head + 1] in {"run", "run-script"}
            and not tokens[head + 2].startswith("-")
            and tokens[head + 3] == "--"):
        return head + 3
    return None


def _path_roles(clause: _Clause, *, display_partition: bool = False) -> list[tuple[str, str]]:
    tokens = clause.tokens
    n = len(tokens)
    program_positions, head = _program_position_indices(tokens)
    grep_patterns = _grep_pattern_indices(tokens, head)
    forwarding = _npm_forwarding_separator(tokens, head)

    first_path_idx = next((i for i, t in enumerate(tokens) if _is_path_shaped(t)), None)
    promoted_idx: int | None = None
    if clause.mode == "narrative" and first_path_idx is not None:
        if all(_is_bare_word(t) for t in tokens[:first_path_idx]) and _is_toolchain_executable(
            tokens[first_path_idx]
        ):
            promoted_idx = first_path_idx

    results: list[tuple[str, str]] = []
    option_values = _scratch_option_values(tokens, head) if display_partition else {}
    idx = 0
    terminated = False
    while idx < n:
        token = tokens[idx]
        if idx in grep_patterns:
            idx += 1
            continue
        if idx == forwarding:
            idx += 1
            continue
        if token == "--" and (head is None or idx > head):
            terminated = True
            idx += 1
            continue
        if terminated:
            value = token.split("=", 1)[-1] if token.startswith("-") else token
            if _is_path_shaped(value):
                results.append((value, "argument"))
            idx += 1
            continue
        if idx in option_values:
            results.append((option_values[idx], "scratch_input"))
            idx += 1
            continue

        matched_prefix = False
        for prefix in WORKDIR_FLAG_PREFIXES:
            if token.startswith(prefix):
                results.append((token[len(prefix) :], "working_directory"))
                matched_prefix = True
                break
        if matched_prefix:
            idx += 1
            continue
        for flag in INTERPRETER_VALUE_FLAGS:
            if token.startswith(flag + "="):
                results.append((token[len(flag) + 1 :], "interpreter_value"))
                matched_prefix = True
                break
        if matched_prefix:
            idx += 1
            continue

        for prefix in OUTPUT_VALUE_FLAG_PREFIXES:
            if token.startswith(prefix):
                results.append((token[len(prefix) :], "output"))
                matched_prefix = True
                break
        if matched_prefix:
            idx += 1
            continue

        env_match = VAR_ASSIGNMENT_RE.match(token)
        if env_match:
            eq_idx = token.index("=")
            var_name = token[:eq_idx]
            value = token[eq_idx + 1 :]
            if var_name in INTERPRETER_VALUE_ENV_VARS:
                parts = value.split(":") if var_name == "PATH" else [value]
                for part in parts:
                    if part:
                        results.append((part, "interpreter_value"))
            idx += 1
            continue

        if token in WORKDIR_FLAGS and idx + 1 < n:
            results.append((tokens[idx + 1], "working_directory"))
            idx += 2
            continue

        if token in INTERPRETER_VALUE_FLAGS and idx + 1 < n:
            results.append((tokens[idx + 1], "interpreter_value"))
            idx += 2
            continue

        if token in OUTPUT_VALUE_FLAGS and idx + 1 < n:
            results.append((tokens[idx + 1], "output"))
            idx += 2
            continue

        # Forwarded options retain strict equals-value extraction after npm's
        # separator; known destinations were already handled above.
        if ((forwarding is not None and idx > forwarding and token.startswith("-"))
                or token.partition("=")[0] in {
            "-c", "--config", "--config-file", "--confcutdir",
            "--ignore", "--ignore-glob", "--cov-config",
        }) and "=" in token:
            value = token.split("=", 1)[1]
            if _is_path_shaped(value):
                results.append((value, "argument"))
                idx += 1
                continue

        if _is_path_shaped(token):
            if idx in program_positions or idx == promoted_idx:
                results.append((token, "program"))
            else:
                results.append((token, "argument"))
        idx += 1

    return results


def _check_path_role(
    raw_path: str,
    role: str,
    *,
    command: str,
    assigned: Path,
    path_violations: list[str] | None = None,
) -> None:
    path = _normalize_reported_path(raw_path)
    if path is None:
        return
    if path == assigned or _is_inside(path, assigned):
        return
    if role == "program" and _is_toolchain_executable(raw_path):
        return
    if role == "interpreter_value":
        return
    if role in {"output", "scratch_input"}:
        return
    if path_violations is not None:
        # The caller degrades this report to out-of-checkout context instead
        # of rejecting the whole payload; every other check still applies.
        path_violations.append(raw_path)
        if role == "program" and isinstance(path_violations, _PathLog):
            path_violations.program_outside = True
        return
    raise (_OutsideProgramError if role == "program" else AgentLoopError)(
        "Coder reported tests from outside the assigned checkout: "
        f"{raw_path!r} in command {command!r}. Assigned checkout: {assigned}"
    )


class _OutsideProgramError(AgentLoopError):
    """An interpreter or executable outside the checkout; may carry a remedy (#1651)."""


def _with_dependency_hint(exc: AgentLoopError, dependency_hint: str | None) -> AgentLoopError | None:
    """The same refusal with the advisory remedy appended, or None to re-raise unchanged."""
    if not dependency_hint or not isinstance(exc, _OutsideProgramError):
        return None
    from .worktree_hints import with_remedy

    return _OutsideProgramError(with_remedy(str(exc), dependency_hint))


def _windows_path_is_exempt(command: str, raw_windows: str, origin: Origin) -> bool:
    # Windows paths use backslashes, which `shlex` treats as POSIX escape
    # characters and would otherwise mangle; use plain whitespace splitting
    # here instead of the shared shlex-based clause tokenizer.
    if not _is_windows_toolchain_executable(raw_windows):
        return False
    for segment_text, _mode in _segments_for_entry(command, origin):
        tokens = segment_text.split()
        idx = next((i for i, t in enumerate(tokens) if t.strip("`'\".,") == raw_windows), None)
        if idx is None:
            continue
        traversal = _wrapper_traversal(tokens)
        if traversal.effective_head_index != idx:
            return False
        if idx > 0 and tokens[idx - 1] in WORKDIR_FLAGS:
            return False
        return True
    return False


# ---------------------------------------------------------------------------
# URL pass: package-acquisition exemption.
# ---------------------------------------------------------------------------


def _acquisition_exempt(clause: _Clause) -> bool:
    tokens = clause.tokens
    n = len(tokens)
    idx = _effective_head_index(tokens)
    if idx is None or idx >= n:
        return False
    head = tokens[idx]
    head_name = Path(_strip_wrap(head)).name

    def next_is_acquisition_subcommand(start: int) -> bool:
        j = start
        while j < n and tokens[j].startswith("-"):
            j += 1
        return j < n and tokens[j] in ACQUISITION_SUBCOMMANDS

    if head_name in DIRECT_PACKAGE_MANAGERS:
        return next_is_acquisition_subcommand(idx + 1)

    if INTERPRETER_BASENAME_RE.match(head_name):
        j = idx + 1
        module: str | None = None
        while j < n:
            tok = tokens[j]
            if tok == "-m" and j + 1 < n:
                module = tokens[j + 1]
                j += 2
                break
            if tok.startswith("-m") and tok != "-m":
                module = tok[2:]
                j += 1
                break
            if tok.startswith("-"):
                j += 1
                continue
            break
        if module is not None and module in MODULE_PACKAGE_MANAGERS:
            return next_is_acquisition_subcommand(j)
    return False


# ---------------------------------------------------------------------------
# URL pass: clause classification.
# ---------------------------------------------------------------------------

UrlMode = Literal["STRICT_COMMAND", "NARRATIVE_COMMAND", "PROSE"]


def _is_command_clause(clause: _Clause) -> bool:
    if clause.mode == "structured" and clause.command_by_contract:
        return True
    tokens = clause.tokens
    traversal = _wrapper_traversal(tokens)
    if traversal.recognized_prefix and traversal.effective_head_index is None:
        # A malformed recognized wrapper remains command syntax.  It gets no
        # nested-program exemption, but must not hide live URLs as prose.
        return True
    idx = traversal.effective_head_index
    if idx is None or idx >= len(tokens):
        return False
    return _is_command_shaped(tokens[idx])


def _url_mode(clause: _Clause) -> UrlMode:
    if not _is_command_clause(clause):
        return "PROSE"
    if clause.mode == "structured" and clause.command_by_contract:
        return "STRICT_COMMAND"
    if clause.mode == "code":
        return "STRICT_COMMAND"
    return "NARRATIVE_COMMAND"


# ---------------------------------------------------------------------------
# URL pass: narrative/prose attachment (rules 1-4) with verb negation.
# ---------------------------------------------------------------------------


def _walk_span(tokens: Sequence[str], start: int, n: int) -> list[str]:
    targets: list[str] = []
    j = start
    still_leading = True
    while j < n:
        tok = tokens[j]
        if _is_url_token(tok):
            targets.append(tok)
            j += 1
            still_leading = False
            continue
        if tok.startswith("-"):
            still_leading = False
            j += 1
            if j < n and not tokens[j].startswith("-") and not _is_url_token(tokens[j]):
                j += 1
            continue
        if _is_path_like_token(tok) or "=" in tok:
            still_leading = False
            j += 1
            continue
        if still_leading:
            j += 1
            continue
        break
    return targets


def _is_negated_occurrence(tokens: Sequence[str], idx: int) -> bool:
    start = max(0, idx - 3)
    for k in range(start, idx):
        if _is_negation_word(_word(tokens[k])):
            return True
    return False


def _execution_phrase_end(
    tokens: Sequence[str],
    start: int,
    n: int,
    *,
    stop_at_negation: bool = True,
) -> int:
    """End (exclusive) of the phrase attached to an execution verb.

    The phrase normally stops at a negation word or at the next execution verb,
    so a later negated or separately-reported clause never lends its URL to this
    verb (and vice versa: the next verb gets its own phrase, subject to its
    own negation check). Wrapper traversal passes ``stop_at_negation=False``
    because a wrapper option operand can itself be a negation word.
    """
    j = start
    while j < n:
        word = _word(tokens[j])
        if word in EXECUTION_VERBS or (stop_at_negation and _is_negation_word(word)):
            break
        j += 1
    return j


def _malformed_wrapper_recovery_targets(
    tokens: Sequence[str],
    start: int,
    n: int,
    *,
    wrapper_positions: set[int] | None = None,
) -> list[str]:
    """Recover a command-shaped head without making later URLs commands.

    A malformed wrapper prefix is still useful evidence that a nearby command
    head was intended, but only before the same negation-aware boundary used by
    prose attachment. Once a head is recovered, its command span remains
    clause-wide so incidental words such as ``no proxy`` cannot hide a URL.
    """
    phrase_end = _execution_phrase_end(tokens, start, n)
    if wrapper_positions is None:
        traversal = _wrapper_traversal(tokens[start:phrase_end])
        wrapper_positions = {
            start + position for position in traversal.program_positions
        }
    head_idx = next(
        (
            idx
            for idx in range(start, phrase_end)
            if idx not in wrapper_positions
            and _program_basename(tokens[idx]) not in WRAPPER_PROGRAMS
            and not _is_url_token(tokens[idx])
            and _is_command_shaped(tokens[idx])
        ),
        None,
    )
    if head_idx is None:
        return []
    return _walk_span(tokens, head_idx, n)


def _head_opened_span_targets(clause: _Clause) -> list[str]:
    tokens = clause.tokens
    n = len(tokens)
    traversal = _wrapper_traversal(tokens)
    head_idx = traversal.effective_head_index
    if head_idx is not None and head_idx < n:
        if _is_url_token(tokens[head_idx]):
            return [tokens[head_idx]]
        if _is_command_shaped(tokens[head_idx]):
            return _walk_span(tokens, head_idx, n)
        return []
    if traversal.recognized_prefix:
        return _malformed_wrapper_recovery_targets(
            tokens,
            0,
            n,
            wrapper_positions=traversal.program_positions,
        )
    return []


def _prepositional_url_targets(tokens: Sequence[str], start: int, end: int) -> list[str]:
    """URLs attached to any target preposition within ``tokens[start:end]``.

    Scanning every preposition in the phrase -- not just the first one --
    matters because an earlier, non-URL prepositional object would otherwise
    hide the real target ("ran the suite against the production environment
    at https://live.example").
    """
    found: list[str] = []
    for k in range(start, end):
        if _word(tokens[k]) not in TARGET_PREPOSITIONS:
            continue
        if _is_negated_occurrence(tokens, k):
            continue
        m = k + 1
        if m < end and _word(tokens[m]) in DETERMINERS:
            m += 1
        if m < end and _is_url_token(tokens[m]):
            found.append(tokens[m])
    return found


def _verb_based_targets(clause: _Clause) -> list[str]:
    tokens = clause.tokens
    n = len(tokens)
    targets: list[str] = []
    for idx, raw in enumerate(tokens):
        if _word(raw) not in EXECUTION_VERBS:
            continue
        if _is_negated_occurrence(tokens, idx):
            continue

        start = idx + 1
        j = start
        if j < n and _word(tokens[j]) in DETERMINERS:
            j += 1

        wrapper_end = _execution_phrase_end(tokens, j, n, stop_at_negation=False)
        traversal = _wrapper_traversal(tokens[j:wrapper_end])
        if traversal.effective_head_index is not None:
            head_idx = j + traversal.effective_head_index
            if _is_url_token(tokens[head_idx]):
                targets.append(tokens[head_idx])
            elif _is_command_shaped(tokens[head_idx]):
                targets.extend(_walk_span(tokens, head_idx, n))

        phrase_end = _execution_phrase_end(tokens, start, n)
        attached = _prepositional_url_targets(tokens, start, phrase_end)
        if attached:
            targets.extend(attached)
            continue
        if traversal.recognized_prefix and traversal.effective_head_index is None:
            targets.extend(
                _malformed_wrapper_recovery_targets(
                    tokens,
                    j,
                    n,
                    wrapper_positions={
                        j + position for position in traversal.program_positions
                    },
                )
            )
    return targets


def _url_targets_in_clause(clause: _Clause) -> list[str]:
    mode = _url_mode(clause)
    if mode == "STRICT_COMMAND":
        if _acquisition_exempt(clause):
            return []
        return [t for t in clause.tokens if _is_url_token(t)]
    targets: list[str] = []
    if mode == "NARRATIVE_COMMAND":
        targets.extend(_head_opened_span_targets(clause))
    targets.extend(_verb_based_targets(clause))
    return targets


# ---------------------------------------------------------------------------
# Top-level validation entry points.
# ---------------------------------------------------------------------------


class _PathLog(list):
    """Out-of-checkout path violations plus where the run's test targets lie."""

    def __init__(self) -> None:
        super().__init__()
        self.inside_target = False
        self.outside_target = False
        self.cwd_outside = False
        # The current directory can no longer be proven: a popd, ``cd -``, a
        # relative or unresolvable cd, or a cd whose scope may have ended.
        self.cwd_unknown = False
        self.cwd_depth = 0
        # A target whose location cannot be proven blocks the context label.
        self.ambiguous_target = False
        # An interpreter or executable outside the checkout was named.
        self.program_outside = False
        # The tracked working directory; ``None`` means the assigned checkout
        # until a resolvable move, and is cleared whenever cwd_unknown is set.
        self.cwd_path: Path | None = None
        # A read-only measurement (du, ls, ...) named only outside operands.
        self.measured_outside = False
        # Measurement operands whose location cannot be proved, such as an
        # unexpanded variable or command substitution (#2052).
        self.unresolved_inputs: list[str] = []
        # The entry needed the whitespace fallback because shell lexing failed.
        self.parse_failed = False
        # ``PATH`` was reassigned, so a bare program name may resolve anywhere.
        self.path_modified = False
        # Nesting depth of managed commands launched by an untrusted sh -c.
        self.inherited_untrusted = 0

    def note_outside(self, location: str) -> None:
        """Record an outside location once, however the command spelled it."""
        if location not in self:
            self.append(location)


# Options whose following operand is a value (config file, plugin, expression,
# worker count, report path, ...) rather than a test target.  The list can
# never be complete, so an absolute path after an *unknown* option is also
# ignored; see _record_test_targets for why that is the fail-safe direction.
_VALUE_TAKING_OPTIONS = frozenset({
    "-c", "-p", "-k", "-m", "-o", "-n", "-r", "-W", "--config", "--config-file",
    "--confcutdir", "--basetemp", "--cache-dir", "--ignore", "--ignore-glob",
    "--deselect", "--cov", "--cov-config", "--cov-report", "--override-ini",
    "--junitxml", "--junit-xml", "--html", "--log-file", "--log-level",
    "--import-mode", "--numprocesses", "--dist", "--maxfail", "--tb",
    "--timeout", "--timeout-method", "--durations", "--capture", "--color",
    "--rootdir",
})
# Flags known to take no value, so a following path is a test operand.
_BOOLEAN_SHORT_FLAGS_RE = re.compile(r"-[qvxslh]+")
_BOOLEAN_LONG_FLAGS = frozenset({
    "--quiet", "--verbose", "--exitfirst", "--lf", "--last-failed", "--ff",
    "--failed-first", "--sw", "--stepwise", "--no-header", "--strict-markers",
    "--showlocals", "--collect-only", "--co",
})
# Options that change the process working directory.  pytest's --rootdir is
# deliberately absent: it selects configuration, not where operands resolve.
_CWD_FLAGS = frozenset({"-C", "--directory", "--chdir", "--cwd"})
_CWD_FLAG_PREFIXES = ("--directory=", "--chdir=", "--cwd=")
# Clause heads that never run tests themselves.  A shell's own clause is here
# because its -c script is expanded into separate clauses.
_NON_TEST_HEADS = frozenset({
    "pwd", "echo", "printf", "true", "false", "export", "set",
    "source", ".", "mkdir", "rm", "cp", "mv", "which", "git", "grep",
    "egrep", "fgrep",
    "sh", "bash", "zsh",
})


# Read-only inspection programs whose whole command line the guard can parse
# (#2052).  A clause headed by one of these is a *measurement*: it tests
# nothing, so an entry whose every measurement operand lies outside the
# checkout is out-of-checkout context instead of an ambiguous test run.  The
# grammar is closed: an option it does not list, or one that reads its file
# list from another file, makes the clause ambiguous, which keeps the entry
# from ever becoming context.
@dataclass(frozen=True)
class _MeasurementGrammar:
    short_booleans: str = ""
    long_booleans: frozenset[str] = frozenset()
    # Value-taking options: short letters, and long names (separate or ``=``).
    short_values: str = ""
    long_values: frozenset[str] = frozenset()
    # Long options whose value is optional and only given in ``=`` form.
    optional_long_values: frozenset[str] = frozenset()
    # Options naming a control file that may itself list checkout files.
    indirect: frozenset[str] = frozenset()
    # ``head -5`` style numeric counts.
    numeric_count: bool = False


_MEASUREMENT_GRAMMARS: dict[str, _MeasurementGrammar] = {
    "du": _MeasurementGrammar(
        short_booleans="abchkmsxlLHPDS0",
        long_booleans=frozenset({
            "--all", "--apparent-size", "--bytes", "--total", "--dereference-args",
            "--human-readable", "--si", "--count-links", "--dereference",
            "--no-dereference", "--separate-dirs", "--summarize", "--one-file-system",
            "--null", "--inodes",
        }),
        short_values="dBt",
        long_values=frozenset({
            "--max-depth", "--block-size", "--threshold", "--exclude", "--time-style",
        }),
        indirect=frozenset({"--files0-from", "-X", "--exclude-from"}),
    ),
    "ls": _MeasurementGrammar(
        short_booleans="aAlhdR1trSFinogGpQUXvs",
        long_booleans=frozenset({
            "--all", "--almost-all", "--human-readable", "--directory", "--recursive",
            "--classify", "--inode", "--numeric-uid-gid", "--reverse", "--size",
            "--color",
        }),
        short_values="Iw",
        long_values=frozenset({"--ignore", "--width", "--sort", "--time-style"}),
        optional_long_values=frozenset({"--color"}),
    ),
    "wc": _MeasurementGrammar(
        short_booleans="cmlLw",
        long_booleans=frozenset({
            "--bytes", "--chars", "--lines", "--max-line-length", "--words",
        }),
        indirect=frozenset({"--files0-from"}),
    ),
    "stat": _MeasurementGrammar(
        short_booleans="Lft",
        long_booleans=frozenset({"--dereference", "--file-system", "--terse"}),
        short_values="c",
        long_values=frozenset({"--format", "--printf"}),
    ),
    "file": _MeasurementGrammar(
        short_booleans="biLhzks",
        long_booleans=frozenset({
            "--brief", "--mime", "--mime-type", "--mime-encoding", "--dereference",
            "--no-dereference", "--uncompress", "--keep-going", "--special-files",
        }),
        indirect=frozenset({"-f", "--files-from", "-m", "--magic-file"}),
    ),
    "head": _MeasurementGrammar(
        short_booleans="qv",
        long_booleans=frozenset({"--quiet", "--silent", "--verbose"}),
        short_values="nc",
        long_values=frozenset({"--lines", "--bytes"}),
        numeric_count=True,
    ),
    "tail": _MeasurementGrammar(
        short_booleans="qv",
        long_booleans=frozenset({"--quiet", "--silent", "--verbose"}),
        short_values="nc",
        long_values=frozenset({"--lines", "--bytes"}),
        numeric_count=True,
    ),
    "cat": _MeasurementGrammar(
        short_booleans="AbeEnstTuv",
        long_booleans=frozenset({
            "--show-all", "--number-nonblank", "--number", "--squeeze-blank",
            "--show-ends", "--show-tabs", "--show-nonprinting",
        }),
    ),
    # ``find`` has its own expression grammar; see _find_operands.
    "find": _MeasurementGrammar(),
}
# find predicates that take one value (a pattern, number or format), never a
# file the predicate reads.
_FIND_VALUE_PREDICATES = frozenset({
    "-name", "-iname", "-path", "-ipath", "-regex", "-type", "-maxdepth",
    "-mindepth", "-size", "-mtime", "-mmin", "-perm", "-user", "-group", "-printf",
})
_FIND_PLAIN_TOKENS = frozenset({
    "-print", "-print0", "-prune", "-empty", "-ls", "-o", "-a", "-or", "-and",
    "-not", "(", ")", "!",
})


def _find_operands(tokens: Sequence[str], start: int) -> list[str] | None:
    """find's starting points, or ``None`` when its expression is not read-only.

    Executors (``-exec``, ``-ok``, ``-delete``, ...), output files
    (``-fprint``), and predicates naming a reference file (``-newer``,
    ``-samefile``, ...) all fall outside the closed grammar.
    """
    index = start
    while index < len(tokens) and tokens[index] in {"-H", "-L", "-P"}:
        index += 1
    starts: list[str] = []
    while index < len(tokens) and not tokens[index].startswith(("-", "(", ")", "!")):
        starts.append(tokens[index])
        index += 1
    while index < len(tokens):
        token = tokens[index]
        if token in _FIND_VALUE_PREDICATES:
            if index + 1 >= len(tokens):
                return None
            index += 2
        elif token in _FIND_PLAIN_TOKENS:
            index += 1
        else:
            return None
    return starts


def _measurement_operands(
    tokens: Sequence[str], start: int, grammar: _MeasurementGrammar
) -> list[str] | None:
    """File operands of a measurement clause, or ``None`` if not fully recognised."""
    operands: list[str] = []
    index = start
    terminated = False
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if terminated:
            operands.append(token)
            continue
        if token == "--":
            terminated = True
            continue
        if token == "-":
            # Standard input names no file.
            continue
        if token.startswith("--"):
            name, equals, _ = token.partition("=")
            if name in grammar.indirect:
                return None
            if name in grammar.long_values:
                if not equals:
                    if index >= len(tokens):
                        return None
                    index += 1
                continue
            if name in grammar.optional_long_values or (
                not equals and name in grammar.long_booleans
            ):
                continue
            return None
        if token.startswith("-"):
            body = token[1:]
            if grammar.numeric_count and body.isdigit():
                continue
            for position, letter in enumerate(body):
                if f"-{letter}" in grammar.indirect:
                    return None
                if letter in grammar.short_values:
                    if position + 1 == len(body):
                        # The value is the next token, never an operand.
                        if index >= len(tokens):
                            return None
                        index += 1
                    break
                if letter not in grammar.short_booleans:
                    return None
            continue
        operands.append(token)
    return operands


def _is_unexpanded(token: str) -> bool:
    """Whether a token still holds shell expansion the guard cannot resolve."""
    if "`" in token:
        return True
    remainder = token[len("$HOME/"):] if token.startswith("$HOME/") else token
    return "$" in remainder


# Shell expansion the guard cannot evaluate: it may produce any text,
# including ``/`` and ``..``.  Glob characters only match names within one
# directory, so a glob's literal parent directory bounds its matches.
_UNKNOWN_EXPANSION_CHARS = frozenset("$`{")
_GLOB_CHARS = frozenset("*?[")
_GLOB_MATCH_LIMIT = 256
OperandLocation = Literal["inside", "outside", "ambiguous"]


def _operand_location(
    raw: str, cwd: Path | None, assigned: Path, *, selector: bool = False
) -> tuple[OperandLocation, str | None]:
    """Where a reported operand can lie, and an outside location to record.

    A literal operand is canonicalised against ``cwd``.  An operand the shell
    would expand is outside only when every possible expansion is provably
    outside (#2052): an unknown expansion (``$X``, a substitution, a brace)
    or a ``..`` after a glob is ambiguous, and so is a glob whose literal
    parent is an ancestor of the checkout (``../*/tests`` from the checkout)
    or whose current matches reach it.  The second value is the outside
    location to record as a violation, if any.
    """

    def inside(path: Path) -> bool:
        return path == assigned or _is_inside(path, assigned)

    text = raw
    if selector and "::" in text and text.split("::", 1)[0]:
        # A test runner's node selector (``path::name[param]``) runs its file
        # part; a measurement keeps the whole filename (#2052).
        text, _, suffix = text.partition("::")
        if "$" in suffix or "`" in suffix:
            # The shell may split the suffix into further operands.
            _, note = _operand_location(text, cwd, assigned)
            return "ambiguous", note
    if text.startswith("$HOME/"):
        text = str(Path.home()) + text[len("$HOME"):]
    elif text == "~" or text.startswith("~/"):
        text = str(Path.home()) + text[1:]
    if text.startswith("/"):
        path, offset = Path(text), 0
    elif cwd is None:
        return "ambiguous", None
    else:
        path, offset = cwd / text, len(cwd.parts)
    parts = path.parts
    first = next(
        (
            index for index in range(offset, len(parts))
            if set(parts[index]) & (_UNKNOWN_EXPANSION_CHARS | _GLOB_CHARS)
        ),
        None,
    )
    try:
        if first is None:
            resolved = _canonical(path)
            return ("inside", None) if inside(resolved) else ("outside", str(resolved))
        prefix = _canonical(Path(*parts[:first]))
        note = None if inside(prefix) else str(prefix)
        remainder = parts[first:]
        if ".." in remainder or any(set(part) & _UNKNOWN_EXPANSION_CHARS for part in remainder):
            return "ambiguous", note
        if inside(prefix):
            return "inside", None
        if _is_inside(assigned, prefix):
            # The glob's parent contains the checkout: a match may be the
            # checkout, an ancestor of it, or a path inside it, unless some
            # component cannot match the checkout's own path below the parent.
            below = assigned.relative_to(prefix).parts
            if all(
                fnmatch.fnmatchcase(name, pattern)
                for name, pattern in zip(below, remainder)
            ):
                return "ambiguous", note
        pattern = str(prefix.joinpath(*remainder))
        matches = list(islice(glob.iglob(pattern), _GLOB_MATCH_LIMIT + 1))
        if len(matches) > _GLOB_MATCH_LIMIT or any(
            inside(_canonical(Path(match))) for match in matches
        ):
            return "ambiguous", note
        return "outside", pattern
    except (OSError, RuntimeError, ValueError):
        return "ambiguous", None


# Prefix wrappers that neither supply operands nor move the working
# directory; ``xargs`` (operands from stdin) and ``sudo`` are excluded, and
# ``env`` is accepted only without options (``env -C``/``-S``).
_MEASUREMENT_SAFE_WRAPPERS = frozenset({
    "env", "nohup", "nice", "time", "timeout", "stdbuf", "command",
})


def _launch_is_plain(tokens: Sequence[str], positions: set[int], head: int | None) -> bool:
    """Whether a clause runs its head program directly, from PATH or the toolchain.

    Every program position (wrappers and the head) must be a bare name or an
    executable in a system toolchain directory, and every prefix wrapper must
    neither supply operands nor move the working directory, so neither an
    outside nor a checkout-located program can hide behind a measurement
    basename (#2052).
    """
    if head is None:
        return False
    env_options_closed = False
    for index in range(head):
        token = tokens[index]
        if index in positions:
            if _program_basename(token) not in _MEASUREMENT_SAFE_WRAPPERS:
                return False
            env_options_closed = _program_basename(token) == "env"
            continue
        if VAR_ASSIGNMENT_RE.match(token):
            if token.split("=", 1)[0] in INTERPRETER_VALUE_ENV_VARS:
                return False
            continue
        if env_options_closed and token.startswith("-"):
            return False
    for index in positions:
        program = _strip_wrap(tokens[index])
        if "/" not in program and "\\" not in program:
            continue
        if not _is_path_shaped(program):
            return False
        try:
            located = _canonical(Path(_normalize_reported_path(program) or program))
        except (OSError, RuntimeError, ValueError):
            return False
        if str(located.parent) not in _EXACT_TOOLCHAIN_DIRS:
            return False
    return True


def _is_shell_literal(token: str) -> bool:
    """Whether a lexed token means to the shell exactly what it reads as.

    The lexer drops quotes, so expansion syntax that survives lexing may or
    may not have been expanded: ``$`` and backticks may add words, run
    commands, or name any directory, and a leading ``~`` may be a home path
    or a quoted literal directory.  Such a token never proves a location
    anywhere in a clause -- operand, option value, ``find`` predicate value
    or selector suffix (#2052).  An assignment value is not word-split, so
    only a substitution there runs anything.
    """
    if VAR_ASSIGNMENT_RE.match(token):
        value = token.split("=", 1)[1]
        return "$(" not in value and "`" not in value
    return "$" not in token and "`" not in token and not token.startswith("~")


_SELECTOR_RUNNERS = frozenset({"pytest", "py.test"})


def _runs_pytest(tokens: Sequence[str], head: int) -> bool:
    """Whether a clause runs pytest, whose operands may be node selectors."""
    if _program_basename(tokens[head]) in _SELECTOR_RUNNERS:
        return True
    return any(
        _program_basename(token) in _SELECTOR_RUNNERS and tokens[index - 1] in {"-m", "run"}
        for index, token in enumerate(tokens[head + 1 :], start=head + 1)
    )


def _literal_directory(raw: str) -> Path | None:
    """A working-directory target, or ``None`` when the shell would expand it.

    ``cd /work/$NAME`` or ``cd /work/*`` may land anywhere, including the
    checkout, so such a target leaves the directory unknown (#2052).
    """
    if not _is_shell_literal(raw) or set(raw) & (_UNKNOWN_EXPANSION_CHARS | _GLOB_CHARS):
        return None
    return _normalize_reported_path(raw)


def _record_test_targets(clause: _Clause, *, assigned: Path, log: _PathLog) -> None:
    """Note whether a clause's test targets lie inside and/or outside the checkout.

    This is fail-safe: a run counts as out-of-checkout context only with
    positive evidence that everything it tested lies elsewhere.  Every
    positional operand that is not absolute is resolved against the tracked
    working directory -- relative paths, bare words such as ``tests``, node
    IDs such as ``test_a.py::t`` -- and classified by where it lands, so
    ``../other`` from the checkout is an outside location and
    ``checkout/tests`` from its parent is an inside one (#2052).  Every
    outside location reaches the violation list.  The directory is the
    confined in-checkout ``cwd`` unless a ``cd``/``pushd`` clause or a real
    working-directory option (``-C``, ``--chdir``, ...) moved it;
    ``--rootdir`` does not.  A test clause with no positional operand tests
    its current directory.  An outside location after a value-taking or
    unknown option is not a target, so an unrecognized report or config path
    can never turn an in-checkout run into context; an inside path there
    still counts, and every token after ``--`` is an operand.

    A clause headed by a read-only measurement program (``du``, ``ls``, ...)
    is classified through its closed grammar instead: its file operands set
    ``measured_outside`` or ``inside_target``, and anything the grammar does
    not recognise is ambiguous.

    The directory is tracked only through resolvable absolute moves outside
    any subshell.  After ``popd``, ``cd -``, a relative or unresolvable
    target, a subshell or substitution, or the end of the ``sh -c`` script
    that moved it, the directory is unknown and a relative or implicit target
    is ambiguous, which keeps the run authoritative.
    """

    def inside(path: Path) -> bool:
        return path == assigned or _is_inside(path, assigned)

    def outside(raw: str) -> bool | None:
        path = _normalize_reported_path(raw)
        if path is None:
            return None
        return not inside(path)

    def forget_cwd() -> None:
        log.cwd_unknown = True
        log.cwd_path = None

    tokens = clause.tokens
    if log.cwd_depth > clause.shell_depth:
        # The shell -c script that moved the directory has exited.
        forget_cwd()
    if any(token.startswith("PATH=") for token in tokens):
        # A later bare program name may now resolve anywhere.
        log.path_modified = True
    if not all(_is_shell_literal(token) for token in tokens):
        # Only what the shell takes literally can prove a location (#2052).
        log.ambiguous_target = True
    positions, head = _program_position_indices(tokens)
    if head is None or head >= len(tokens):
        return
    head_name = _program_basename(tokens[head])
    forwarding = _npm_forwarding_separator(tokens, head)
    if head_name in {"cd", "pushd", "popd"}:
        target = (
            _literal_directory(tokens[head + 1]) if head + 1 < len(tokens) else None
        )
        if target is None or clause.in_subshell or head_name == "popd":
            # Only one resolvable move outside any subshell is tracked; a
            # restore or unresolvable target leaves the directory unknown.
            forget_cwd()
        else:
            log.cwd_outside = not inside(target)
            log.cwd_path = target
            log.cwd_depth = max(log.cwd_depth, clause.shell_depth)
        return

    cwd: Path | None = None if log.cwd_unknown else (log.cwd_path or assigned)
    # Only pytest gives ``path::name`` selector meaning; any other program
    # sees the whole token as a filename (#2052).
    selector_runner = _runs_pytest(tokens, head)

    def locate(operand: str, *, measurement: bool = False) -> OperandLocation:
        kind, note = _operand_location(
            operand, cwd, assigned, selector=not measurement and selector_runner
        )
        if note is not None and not _is_path_shaped(operand):
            # The role pass already records (or, for an eligible scratch
            # option value, deliberately exempts) absolute and home paths.
            log.note_outside(note)
        return kind

    def record_relative(operand: str | None, *, measurement: bool = False) -> None:
        if operand is None:
            kind: OperandLocation = (
                "ambiguous" if cwd is None else "inside" if inside(cwd) else "outside"
            )
            if kind == "outside":
                log.note_outside(str(cwd))
        else:
            kind = locate(operand, measurement=measurement)
        if kind == "ambiguous":
            log.ambiguous_target = True
        elif kind == "inside":
            log.inside_target = True
        elif measurement:
            log.measured_outside = True
        else:
            log.outside_target = True

    grammar = _MEASUREMENT_GRAMMARS.get(head_name)
    if grammar is not None:
        if log.parse_failed:
            # A fallback-tokenized clause is never classified by the grammar.
            log.ambiguous_target = True
            return
        if (
            log.path_modified
            or clause.launcher_untrusted
            or log.inherited_untrusted
            or not _launch_is_plain(tokens, positions, head)
        ):
            # Operand-supplying wrappers, PATH changes and path-named
            # programs make the clause ambiguous on its own (#2052); its
            # operands are still located below so outside ones are recorded.
            log.ambiguous_target = True
        for token in tokens[head + 1 :]:
            # The role pass skips ``--option=/path`` values; an outside one
            # (``--files0-from=/x/list``) is still an outside location.
            value = token.partition("=")[2] if token.startswith("--") else ""
            if _is_path_shaped(value):
                path = _normalize_reported_path(value)
                if path is not None and not inside(path):
                    log.note_outside(value)
        operands = (
            _find_operands(tokens, head + 1)
            if head_name == "find"
            else _measurement_operands(tokens, head + 1, grammar)
        )
        if operands is None:
            log.ambiguous_target = True
            return
        if not operands:
            record_relative(None, measurement=True)
        for raw in operands:
            if _is_unexpanded(raw):
                log.unresolved_inputs.append(raw)
                log.ambiguous_target = True
                continue
            cleaned = _strip_wrap(raw)
            if cleaned:
                record_relative(cleaned, measurement=True)
        return
    if head_name in _NON_TEST_HEADS:
        return

    index = head + 1
    while index < len(tokens):
        token = tokens[index]
        if index == forwarding:
            index += 1
            continue
        if token == "--":
            break
        if token in _CWD_FLAGS and index + 1 < len(tokens):
            cwd = _literal_directory(tokens[index + 1])
            index += 2
            continue
        for prefix in _CWD_FLAG_PREFIXES:
            if token.startswith(prefix):
                cwd = _literal_directory(token[len(prefix) :])
        index += 1

    saw_positional = False
    terminated = False
    for index in range(head + 1, len(tokens)):
        token = tokens[index]
        previous = tokens[index - 1]
        if terminated:
            # After ``--`` every token is an operand, never an option value.
            cleaned = _strip_wrap(token)
            if not cleaned:
                continue
            saw_positional = True
            record_relative(cleaned)
            continue
        if index == forwarding:
            continue
        if token == "--":
            terminated = True
            continue
        if (
            token.startswith("-")
            or VAR_ASSIGNMENT_RE.match(token)
            or _is_url_token(token)
            or previous in _CWD_FLAGS
            or previous in WORKDIR_FLAGS
            or previous in _VALUE_TAKING_OPTIONS
            or previous in INTERPRETER_VALUE_FLAGS
            or previous in OUTPUT_VALUE_FLAGS
        ):
            continue
        cleaned = _strip_wrap(token)
        if not cleaned:
            continue
        after_unknown_option = previous.startswith("-") and not (
            _BOOLEAN_SHORT_FLAGS_RE.fullmatch(previous) or previous in _BOOLEAN_LONG_FLAGS
        )
        if cleaned.isdigit() and after_unknown_option and (
            cwd is None or not (cwd / cleaned).exists()
        ):
            # A number after an unknown option is taken as its value only when
            # no such path exists to be a test target (#2052); otherwise it is
            # an operand like any other.
            continue
        if after_unknown_option:
            kind = locate(cleaned)
            if kind == "outside":
                # Possibly an option value: an outside location is a
                # violation but never a target; an inside path still counts.
                continue
            if kind == "ambiguous":
                # Possibly an option value, possibly an operand that reaches
                # the checkout: never context, and the implicit cwd remains.
                log.ambiguous_target = True
                continue
        saw_positional = True
        record_relative(cleaned)
    if not saw_positional:
        record_relative(None)


def _validate_command_contents(
    command: str,
    *,
    assigned: Path,
    origin: Origin,
    path_violations: list[str] | None = None,
    display_partition: bool = False,
) -> None:
    for raw_windows in WINDOWS_PATH_RE.findall(command):
        if _windows_path_is_exempt(command, raw_windows, origin):
            continue
        raise AgentLoopError(
            "Coder reported tests from a Windows-style path that cannot be "
            "validated against the assigned Unix checkout: "
            f"{raw_windows!r} in command {command!r}. Assigned checkout: {assigned}"
        )

    if isinstance(path_violations, _PathLog) and not _lexes_cleanly(command):
        # Malformed quoting falls back to whitespace tokens, which can hide a
        # quoted outside path from the role pass.  Such an entry is never
        # context, and one naming an outside path is refused (#2052).
        path_violations.parse_failed = True
        for token in command.split():
            if not any(quote in token for quote in "'\""):
                continue
            cleaned = token.strip("'\"`;&|()")
            path = _normalize_reported_path(cleaned) if _is_path_shaped(cleaned) else None
            if path is not None and not (path == assigned or _is_inside(path, assigned)):
                path_violations.note_outside(cleaned)

    clauses = _clauses_for_entry(command, origin)
    # A preflight (pwd/git status) can precede a managed test command. Apply
    # the exact wrapper contract per clause, not just to the whole report.
    # Clauses are classified in execution order so a managed command sees the
    # working directory an earlier ``cd`` clause established (#2052).
    ordinary_clauses: list[_Clause] = []
    for clause in clauses:
        inherit = isinstance(path_violations, _PathLog) and clause.launcher_untrusted
        if inherit:
            # The managed inner argv is re-parsed into fresh clauses; an
            # untrusted sh -c launcher above it still taints them (#2052).
            path_violations.inherited_untrusted += 1
        try:
            managed = _validate_managed_command(
                shlex.join(clause.tokens), assigned=assigned,
                origin="structured" if clause.mode in {"structured", "code"} else "response",
                path_violations=path_violations, display_partition=display_partition,
            )
        finally:
            if inherit:
                path_violations.inherited_untrusted -= 1
        if managed:
            continue
        ordinary_clauses.append(clause)
        for raw_path, role in _path_roles(clause, display_partition=display_partition):
            _check_path_role(
                raw_path,
                role,
                command=command,
                assigned=assigned,
                path_violations=path_violations,
            )
        if isinstance(path_violations, _PathLog):
            _record_test_targets(clause, assigned=assigned, log=path_violations)

    for clause in ordinary_clauses:
        for target in _url_targets_in_clause(clause):
            if _has_unmatched_url_scheme(target):
                # A scheme occurrence that produced no (or the wrong)
                # matched URL value -- a bare "http://" with no host, or an
                # outer scheme immediately followed by a nested one such as
                # "http://https://localhost" -- is unverifiable rather than
                # provably loopback; treat it as a live target instead of
                # silently skipping it or judging it by an unrelated inner
                # fragment.
                raise AgentLoopError(
                    "Coder reported tests run against a live remote target: "
                    f"{target!r} in command {command!r}. Assigned checkout: {assigned}"
                )
            for url in _url_values(target):
                if _is_loopback_url(url):
                    continue
                raise AgentLoopError(
                    "Coder reported tests run against a live remote target: "
                    f"{url!r} in command {command!r}. Assigned checkout: {assigned}"
                )


def _validate_managed_command(
    command: str,
    *,
    assigned: Path,
    origin: Origin,
    path_violations: list[str] | None = None,
    display_partition: bool = False,
) -> bool:
    # The managed wrapper itself is an absolute executable outside the checkout
    # by design, and its memory directory is an output location.  Once the
    # exact contract is recognized, validate leading shell assignments and the
    # inner executable/targets so the wrapper cannot hide an escape.
    try:
        tokens = shlex.split(command)
        prefix_len = _wrapper_traversal(tokens, managed=True).effective_head_index
        if prefix_len is None:
            return False
        managed = parse_managed_test_command(
            tokens, allow_command_name_launcher=True
        )
    except ValueError:
        if isinstance(path_violations, _PathLog):
            path_violations.parse_failed = True
        managed = None
    except TestRuntimeConfigurationError:
        managed = None
    if managed is not None:
        # Leading assignments are command syntax even when extracted from a
        # prose response; the inner report retains its original source mode.
        if prefix_len:
            _validate_command_contents(
                shlex.join(tokens[:prefix_len]),
                assigned=assigned,
                origin="structured",
                path_violations=path_violations, display_partition=display_partition,
            )
        _validate_single_command(
            shlex.join(managed.inner_argv),
            assigned=assigned,
            origin=origin,
            path_violations=path_violations, display_partition=display_partition,
        )
        return True
    return False


def _validate_single_command(
    command: str,
    *,
    assigned: Path,
    origin: Origin,
    path_violations: list[str] | None = None,
    display_partition: bool = False,
) -> None:
    if not _validate_managed_command(
        command, assigned=assigned, origin=origin, path_violations=path_violations, display_partition=display_partition
    ):
        _validate_command_contents(
            command, assigned=assigned, origin=origin, path_violations=path_violations, display_partition=display_partition
        )


def validate_test_commands_within_workdir(
    tests_run: Sequence[str] | None,
    *,
    assigned_workdir: Path,
    origin: Origin = "structured",
    dependency_hint: str | None = None,
) -> None:
    """Refuse out-of-checkout test commands.

    ``dependency_hint`` is advisory text appended to an out-of-checkout
    interpreter/executable refusal; it never changes whether a command is refused.
    """
    if not tests_run:
        return
    assigned = _canonical(assigned_workdir)
    for command in tests_run:
        try:
            _validate_single_command(command, assigned=assigned, origin=origin)
        except AgentLoopError as exc:
            hinted = _with_dependency_hint(exc, dependency_hint)
            if hinted is None:
                raise
            raise hinted from exc


@dataclass(frozen=True)
class ReportedTestsPartition:
    """Display-only ``tests_run`` split by whether it ran in the assigned checkout."""

    in_checkout: tuple[str, ...] | None
    out_of_checkout: tuple[str, ...] = ()


def partition_reported_tests_by_workdir(
    tests_run: Sequence[str] | None, *, assigned_workdir: Path,
    dependency_hint: str | None = None,
) -> ReportedTestsPartition:
    try:
        return _partition_reported_tests_by_workdir(
            tests_run, assigned_workdir=assigned_workdir, dependency_hint=dependency_hint
        )
    except AgentLoopError as exc:
        raise ReportWorkdirRefusal(str(exc)) from exc


def _partition_reported_tests_by_workdir(
    tests_run: Sequence[str] | None,
    *,
    assigned_workdir: Path,
    dependency_hint: str | None = None,
) -> ReportedTestsPartition:
    """Separate out-of-checkout context runs from a display-only test report.

    A coder may honestly report a baseline run on a clean copy of the base
    branch.  That run is context, not evidence: it is split out so it never
    becomes a self-reported evidence row, and it does not reject the whole
    hand-off.  Only a run proven to test *only* outside the checkout degrades
    this way.  A command that names an outside path but may also test the
    checkout -- mixed operands, an outside config or ignore path, a directory
    the guard cannot track -- is ambiguous and refused with guidance to run
    the two separately.  A read-only measurement (``du``, ``ls``, ``wc``,
    ``stat``, ``find`` without executors, ...) whose every operand lies
    outside is context too (#2052); one with an unexpanded operand, an
    unparseable entry naming an outside path, and an indirect-input or
    unrecognised option are refused.  Live remote targets, unvalidatable
    Windows paths, and malformed shell text still fail closed, as do receipt
    citations, which keep using the strict validator.
    """
    if not tests_run:
        return ReportedTestsPartition(in_checkout=None if tests_run is None else ())
    assigned = _canonical(assigned_workdir)
    kept: list[str] = []
    context: list[str] = []
    for command in tests_run:
        violations = _PathLog()
        _validate_single_command(
            command, assigned=assigned, origin="structured", path_violations=violations,
            display_partition=True,
        )
        if not violations and not violations.unresolved_inputs:
            kept.append(command)
        elif _tests_only_outside(violations):
            context.append(command)
        else:
            named = ", ".join(repr(p) for p in violations)
            if violations.parse_failed:
                message = (
                    "Coder reported a test command that could not be parsed "
                    "(unbalanced quoting) and names a path outside the assigned "
                    f"checkout ({named}), in command {command!r}. Report each "
                    "command with balanced quotes, and report in-checkout tests "
                    "and any run outside the checkout as separate commands. "
                    f"Assigned checkout: {assigned}"
                )
            elif violations.unresolved_inputs:
                unresolved = ", ".join(repr(p) for p in violations.unresolved_inputs)
                message = (
                    "Coder reported a command whose operand location cannot be "
                    f"proved ({unresolved}: an unexpanded variable or command "
                    f"substitution), in command {command!r}. Name literal paths, "
                    "and report in-checkout tests and any run outside the "
                    "checkout as separate commands. Assigned checkout: "
                    f"{assigned}"
                )
            else:
                message = (
                    "Coder reported a test command that names a path outside the "
                    f"assigned checkout ({named}) "
                    "but may also test the checkout itself, in command "
                    f"{command!r}. Report in-checkout tests and any baseline run "
                    "outside the checkout as separate commands; a baseline command "
                    "must name only outside test targets, and a measurement of an "
                    "outside directory (for example du or ls) must be its own "
                    "entry naming only outside paths. Assigned checkout: "
                    f"{assigned}"
                )
            if violations.program_outside and dependency_hint:
                from .worktree_hints import with_remedy

                message = with_remedy(message, dependency_hint)
            raise AgentLoopError(message)
    return ReportedTestsPartition(in_checkout=tuple(kept), out_of_checkout=tuple(context))


_PACKAGE_MANAGER_HEADS = frozenset({"npm", "pnpm", "yarn"})


def _argv_tokens(argv: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(argv, str):
        try:
            return tuple(shlex.split(argv))
        except ValueError:
            return (argv,)
    return tuple(str(item) for item in argv)


def is_package_manager_command(argv: Sequence[str] | str) -> bool:
    """Find npm/pnpm/yarn through the same wrappers as checkout validation."""
    command = argv if isinstance(argv, str) else shlex.join(_argv_tokens(argv))
    try:
        clauses = _clauses_for_entry(command, "structured")
    except AgentLoopError:
        # The path guard independently refuses unvalidatable shell syntax.
        return False
    for clause in clauses:
        tokens = clause.tokens
        try:
            managed = parse_managed_test_command(tokens, allow_command_name_launcher=True)
        except TestRuntimeConfigurationError:
            managed = None
        if managed is not None:
            if is_package_manager_command(managed.inner_argv):
                return True
            continue
        head = _effective_head_index(tokens)
        if head is not None and _program_basename(tokens[head]) in _PACKAGE_MANAGER_HEADS:
            return True
    return False


def command_is_admissible_evidence(
    argv: Sequence[str] | str,
    *,
    assigned_workdir: Path,
    executed_argv: Sequence[str] | None = None,
) -> bool:
    """Return whether a recorded test command may back an evidence selector.

    A command that targets a location outside the assigned checkout is
    context, not evidence, even when the managed broker ran it from inside the
    checkout (#991).  Anything the guard cannot validate is inadmissible too.

    A package-manager command (``npm run <script>``) hides its operands in
    package.json, so it is admissible only with the ``executed_argv`` the
    runner froze at resolution time (#1294); that argv is validated as well.
    """
    if is_package_manager_command(argv) and not executed_argv:
        return False
    violations = _command_path_violations(argv, assigned_workdir=assigned_workdir)
    if violations is None or violations:
        return False
    if executed_argv:
        executed = _command_path_violations(executed_argv, assigned_workdir=assigned_workdir)
        return executed is not None and not executed
    return True


def command_targets_outside_workdir(
    argv: Sequence[str] | str,
    *,
    assigned_workdir: Path,
    executed_argv: Sequence[str] | None = None,
) -> bool:
    """Return whether a validatable command tests *only* outside the checkout.

    Unlike :func:`command_is_admissible_evidence`, a command the guard cannot
    validate is *not* reported as outside: callers use this to relabel a run
    as context, and an unvalidatable failure must stay authoritative.  A mixed
    run that also names an in-checkout test target, or whose only outside
    path is an option value such as a config or ignore path, is not context
    either: its failure may be a real in-checkout failure.  When
    ``executed_argv`` is supplied it is the argv judged (#1294).
    """
    log = _command_path_violations(
        executed_argv if executed_argv else argv, assigned_workdir=assigned_workdir
    )
    return isinstance(log, _PathLog) and _tests_only_outside(log)


def _tests_only_outside(log: "_PathLog") -> bool:
    """Whether a validated command names outside paths and tests nothing else.

    A read-only measurement of outside paths qualifies too (#2052), unless
    its program itself lies outside the checkout.  Unresolved operands and
    unparseable entries never do.
    """
    return (
        bool(log)
        and not log.inside_target
        and not log.ambiguous_target
        and not log.unresolved_inputs
        and not log.parse_failed
        and (log.outside_target or (log.measured_outside and not log.program_outside))
    )


def _command_path_violations(
    argv: Sequence[str] | str,
    *,
    assigned_workdir: Path,
) -> "_PathLog | None":
    """Out-of-checkout paths in ``argv``, or ``None`` when unvalidatable."""
    command = argv if isinstance(argv, str) else shlex.join(str(item) for item in argv)
    if not command.strip():
        return None
    violations = _PathLog()
    try:
        _validate_single_command(
            command,
            assigned=_canonical(assigned_workdir),
            origin="structured",
            path_violations=violations,
        )
    except AgentLoopError:
        return None
    return violations


def validate_test_observation_citations_within_workdir(
    citations: Sequence[object] | None,
    *,
    assigned_workdir: Path,
    dependency_hint: str | None = None,
) -> None:
    """Apply the same checkout guard to structured receipt citations."""
    if not citations:
        return
    commands = tuple(
        str(getattr(citation, "command", ""))
        for citation in citations
    )
    try:
        validate_test_commands_within_workdir(
            commands, assigned_workdir=assigned_workdir, dependency_hint=dependency_hint
        )
    except AgentLoopError as exc:
        raise ReportWorkdirRefusal(
            "Receipt citations must be admissible checkout evidence; scratch-input "
            "display acceptance grants no receipt authority. Supply admissible replacement "
            "evidence without hiding failing observations. " + str(exc)
        ) from exc


def extract_reported_tests_from_response(text: str) -> tuple[str, ...]:
    """Return the coder's public test-report lines, avoiding quoted issue context."""
    lines = text.splitlines()
    reports: list[str] = []
    for index, line in enumerate(lines):
        match = TEST_SECTION_RE.match(line)
        if not match:
            continue
        body = match.group("body").strip()
        if body:
            reports.append(body)
            continue
        continuation: list[str] = []
        for next_line in lines[index + 1 :]:
            stripped = next_line.strip()
            if not stripped:
                if continuation:
                    break
                continue
            if stripped.startswith("<!-- AGENT_") or stripped.startswith("-- "):
                break
            if re.match(r"^#{1,6}\s+", stripped):
                break
            continuation.append(stripped.removeprefix("- ").strip())
        if continuation:
            reports.extend(continuation)
    return tuple(reports)


def validate_response_tests_within_workdir(
    text: str, *, assigned_workdir: Path, dependency_hint: str | None = None
) -> None:
    validate_test_commands_within_workdir(
        extract_reported_tests_from_response(text),
        assigned_workdir=assigned_workdir,
        origin="response",
        dependency_hint=dependency_hint,
    )


def validate_checkout_inspected_evidence(
    claims: Sequence[DiscussEvidenceClaim],
    *,
    assigned_workdir: Path,
) -> None:
    """Reject a ``checkout-inspected`` evidence claim whose ``path:line``
    source does not resolve to a real, in-range line inside the reviewer's
    assigned checkout right now.

    This is structural containment/existence checking only -- it does not
    verify the claimed fact is actually supported by that line's content.
    Claims with any other (or no) verification_basis are left untouched.
    """
    assigned = _canonical(assigned_workdir)
    for claim in claims:
        if claim.verification_basis != "checkout-inspected":
            continue
        # The parser (protocol.py) already guarantees `source` fullmatches
        # `[^\s:][^:]*:\d+`, i.e. a single colon separating a path with no
        # embedded colons from a trailing line number.
        source = claim.source or ""
        path_part, _, line_part = source.rpartition(":")
        if path_part.startswith("/") or path_part.startswith("~"):
            raise AgentLoopError(
                "checkout-inspected evidence claim used an absolute path outside "
                f"the assigned checkout: {source!r} for fact {claim.fact!r}."
            )
        if any(segment == ".." for segment in Path(path_part).parts):
            raise AgentLoopError(
                "checkout-inspected evidence claim used a path traversal segment: "
                f"{source!r} for fact {claim.fact!r}."
            )
        # No real file has anywhere near this many lines; reject before ever
        # calling int() on it. Python 3.11+ raises ValueError (not
        # AgentLoopError) for int()/str() conversions beyond its default
        # 4300-digit limit, and an unbounded `\d+` source is otherwise free
        # to carry an arbitrarily long digit string.
        if len(line_part) > 15:
            raise AgentLoopError(
                "checkout-inspected evidence claim references an implausibly "
                f"large line number: {source!r} for fact {claim.fact!r}."
            )
        resolved = _canonical(assigned / path_part)
        if not _is_inside(resolved, assigned):
            raise AgentLoopError(
                "checkout-inspected evidence claim resolves outside the assigned "
                f"checkout: {source!r} for fact {claim.fact!r}."
            )
        if not resolved.is_file():
            raise AgentLoopError(
                "checkout-inspected evidence claim references a path that is not "
                f"a file in the assigned checkout: {source!r} for fact {claim.fact!r}."
            )
        line_number = int(line_part)
        line_count = 0
        try:
            with resolved.open("rb") as handle:
                for line_count, _ in enumerate(handle, start=1):
                    pass
        except OSError as exc:
            # The already-validated path can still fail to open (deleted or
            # made unreadable between is_file() and open()); translate this
            # the same way as any other unresolvable reference instead of
            # letting an OSError escape past the AgentLoopError/repair
            # contract every other call site of this function relies on.
            raise AgentLoopError(
                "checkout-inspected evidence claim references a path that could "
                f"not be read: {source!r} for fact {claim.fact!r} ({exc})."
            ) from exc
        if not (1 <= line_number <= line_count):
            raise AgentLoopError(
                "checkout-inspected evidence claim references a line number "
                f"outside the file's range: {source!r} for fact {claim.fact!r} "
                f"(file has {line_count} lines)."
            )


def validate_assigned_head_advanced(
    *,
    before_head: str | None,
    after_head: str | None,
    assigned_workdir: Path,
) -> None:
    if not before_head or not after_head:
        return
    if before_head == after_head:
        raise AgentLoopError(
            "Coder reported a PR, but the assigned checkout HEAD did not advance. "
            f"Assigned checkout: {_canonical(assigned_workdir)}; HEAD: {after_head}"
        )
