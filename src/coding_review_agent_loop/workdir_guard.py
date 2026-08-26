"""Validation helpers for keeping coder-reported work inside the assigned checkout."""

from __future__ import annotations

import ipaddress
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence
from urllib.parse import urlsplit

from .errors import AgentLoopError
from .protocol import DiscussEvidenceClaim


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


def _git_probe(
    runner: object,
    *,
    workdir: Path,
    args: tuple[str, ...],
    tolerate_exceptions: bool,
) -> GitProbeResult:
    command = ("git", *args)
    try:
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
) -> GitProbeResult:
    """Read only ``git rev-parse HEAD`` from ``workdir``."""
    return _git_probe(
        runner,
        workdir=workdir,
        args=("rev-parse", "HEAD"),
        tolerate_exceptions=tolerate_exceptions,
    )


def capture_workdir_snapshot(
    runner: object,
    workdir: Path,
    *,
    tolerate_exceptions: bool = False,
) -> WorkdirSnapshot:
    """Capture HEAD and ``git status --porcelain`` without mutating the checkout."""
    return WorkdirSnapshot(
        head=read_workdir_head(
            runner,
            workdir,
            tolerate_exceptions=tolerate_exceptions,
        ),
        status=_git_probe(
            runner,
            workdir=workdir,
            args=("status", "--porcelain"),
            tolerate_exceptions=tolerate_exceptions,
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
    "env": {"-u", "-C"},
    "xargs": {"-n", "-P", "-I"},
}
_TIMEOUT_DURATION_RE = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[smhd])?$")


@dataclass(frozen=True)
class _WrapperTraversal:
    """Wrapper-prefix recognition is deliberately distinct from a valid head."""

    recognized_prefix: bool
    program_positions: set[int]
    effective_head_index: int | None


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
        option, sep, _value = token.partition("=")
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
        # Every other dash-prefixed option remains a no-value option.
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


def _wrapper_traversal(tokens: Sequence[str]) -> _WrapperTraversal:
    positions: set[int] = set()
    i = 0
    recognized_prefix = False
    while i < len(tokens):
        if VAR_ASSIGNMENT_RE.match(tokens[i]):
            i += 1
            continue
        wrapper = _program_basename(tokens[i])
        if wrapper not in WRAPPER_PROGRAMS:
            positions.add(i)
            return _WrapperTraversal(recognized_prefix, positions, i)
        recognized_prefix = True
        positions.add(i)
        next_index = _parse_wrapper(tokens, i, wrapper)
        if next_index is None:
            return _WrapperTraversal(True, positions, None)
        i = next_index
    return _WrapperTraversal(recognized_prefix, positions, None)


def _program_position_indices(tokens: Sequence[str]) -> tuple[set[int], int | None]:
    traversal = _wrapper_traversal(tokens)
    return traversal.program_positions, traversal.effective_head_index


def _effective_head_index(tokens: Sequence[str]) -> int | None:
    return _wrapper_traversal(tokens).effective_head_index


# ---------------------------------------------------------------------------
# Clause splitting.
# ---------------------------------------------------------------------------

ClauseMode = Literal["structured", "code", "narrative"]
_SEPARATOR_TOKENS = {"&&", "||", ";", "|", "&", "(", ")"}


@dataclass
class _Clause:
    tokens: list[str]
    mode: ClauseMode
    command_by_contract: bool


def _tokenize(text: str) -> list[str]:
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def _split_into_clauses(text: str, mode: ClauseMode) -> list[_Clause]:
    tokens = _tokenize(text)
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
                )
            )
        current = []
        command_by_contract = next_by_contract

    for token in tokens:
        if token in _SEPARATOR_TOKENS:
            flush(True)
            continue
        current.append(token)
        if not token:
            continue
        last_char = token[-1]
        if last_char in ".!?;" and not _is_path_like_token(token):
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
        clauses.extend(_split_into_clauses(text, mode))
    return clauses


# ---------------------------------------------------------------------------
# Path pass: assign a role to every path-shaped token in a clause.
# ---------------------------------------------------------------------------


def _path_roles(clause: _Clause) -> list[tuple[str, str]]:
    tokens = clause.tokens
    n = len(tokens)
    program_positions, _ = _program_position_indices(tokens)

    first_path_idx = next((i for i, t in enumerate(tokens) if _is_path_shaped(t)), None)
    promoted_idx: int | None = None
    if clause.mode == "narrative" and first_path_idx is not None:
        if all(_is_bare_word(t) for t in tokens[:first_path_idx]) and _is_toolchain_executable(
            tokens[first_path_idx]
        ):
            promoted_idx = first_path_idx

    results: list[tuple[str, str]] = []
    idx = 0
    while idx < n:
        token = tokens[idx]

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

        if _is_path_shaped(token):
            if idx in program_positions or idx == promoted_idx:
                results.append((token, "program"))
            else:
                results.append((token, "argument"))
        idx += 1

    return results


def _check_path_role(raw_path: str, role: str, *, command: str, assigned: Path) -> None:
    path = _normalize_reported_path(raw_path)
    if path is None:
        return
    if path == assigned or _is_inside(path, assigned):
        return
    if role == "program" and _is_toolchain_executable(raw_path):
        return
    if role == "interpreter_value":
        return
    if role == "output":
        return
    raise AgentLoopError(
        "Coder reported tests from outside the assigned checkout: "
        f"{raw_path!r} in command {command!r}. Assigned checkout: {assigned}"
    )


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


def _validate_single_command(command: str, *, assigned: Path, origin: Origin) -> None:
    for raw_windows in WINDOWS_PATH_RE.findall(command):
        if _windows_path_is_exempt(command, raw_windows, origin):
            continue
        raise AgentLoopError(
            "Coder reported tests from a Windows-style path that cannot be "
            "validated against the assigned Unix checkout: "
            f"{raw_windows!r} in command {command!r}. Assigned checkout: {assigned}"
        )

    clauses = _clauses_for_entry(command, origin)

    for clause in clauses:
        for raw_path, role in _path_roles(clause):
            _check_path_role(raw_path, role, command=command, assigned=assigned)

    for clause in clauses:
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


def validate_test_commands_within_workdir(
    tests_run: Sequence[str] | None,
    *,
    assigned_workdir: Path,
    origin: Origin = "structured",
) -> None:
    if not tests_run:
        return
    assigned = _canonical(assigned_workdir)
    for command in tests_run:
        _validate_single_command(command, assigned=assigned, origin=origin)


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


def validate_response_tests_within_workdir(text: str, *, assigned_workdir: Path) -> None:
    validate_test_commands_within_workdir(
        extract_reported_tests_from_response(text),
        assigned_workdir=assigned_workdir,
        origin="response",
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
