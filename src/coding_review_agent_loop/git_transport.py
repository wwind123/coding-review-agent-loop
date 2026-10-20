"""Fetch trusted refs privately and import their reachable objects into Git state.

No transport command runs with an agent-controlled checkout as its repository.
Callers hold their checkout/store lock through the import and ref update.
"""

from __future__ import annotations

import base64
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlsplit

from .errors import AgentLoopError
from .secure_git import _first_trusted_executable, _git_path, _trusted_executable, local_command

_SHA = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


def default_origin(repo: str, *, protocol: str = "https",
                   local_origin: Path | None = None) -> str:
    """Choose a new checkout's endpoint solely from operator configuration."""
    if local_origin is not None:
        origin = str(local_origin.resolve(strict=True))
    else:
        parts = repo.split("/")
        if len(parts) == 2:
            host, owner, name = "github.com", *parts
        elif len(parts) == 3:
            host, owner, name = parts
        else:
            raise AgentLoopError(f"Invalid operator repository identity: {repo!r}.")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+(?::[0-9]{1,5})?", host) or not all(
            re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in (owner, name)
        ):
            raise AgentLoopError(f"Invalid operator repository identity: {repo!r}.")
        if protocol == "https":
            origin = f"https://{host}/{owner}/{name}.git"
        elif protocol == "ssh":
            origin = (f"ssh://git@{host}/{owner}/{name}.git" if ":" in host else
                      f"git@{host}:{owner}/{name}.git")
        else:
            raise AgentLoopError(f"Unsupported trusted origin protocol: {protocol}.")
    return trusted_url(repo, origin, local_origin=local_origin)


def trusted_url(repo: str, observed: str, *, local_origin: Path | None = None) -> str:
    """Compare an unexpanded origin with the exact operator-selected endpoint."""
    parts = repo.split("/")
    if len(parts) == 2:
        host, owner, name = "github.com", *parts
    elif len(parts) == 3:
        host, owner, name = parts
    else:
        raise AgentLoopError(f"Invalid operator repository identity: {repo!r}.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+(?::[0-9]{1,5})?", host) or not all(
        re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in (owner, name)
    ):
        raise AgentLoopError(f"Invalid operator repository identity: {repo!r}.")
    suffix = f"{owner}/{name}"
    allowed = {
        f"https://{host}/{suffix}", f"https://{host}/{suffix}.git",
        f"git@{host}:{suffix}", f"git@{host}:{suffix}.git",
        f"ssh://git@{host}/{suffix}", f"ssh://git@{host}/{suffix}.git",
    }
    if local_origin is not None:
        trusted = str(local_origin.resolve(strict=True))
        if observed == trusted:
            return trusted
        raise AgentLoopError(
            f"Checkout origin does not match the operator-trusted endpoint for {repo}; refusing transport."
        )
    if observed not in allowed:
        raise AgentLoopError(
            f"Checkout origin does not match the operator-trusted endpoint for {repo}; refusing transport."
        )
    return observed


class TrustedFile(NamedTuple):
    """A path that passed the operator trust walk.

    ``identity`` is ``(st_dev, st_ino)`` for an existing file and ``None`` for
    an accepted missing file (``allow_missing=True``).
    """

    path: Path
    identity: tuple[int, int] | None


class _UntrustedPath(Exception):
    """A refused path.  Deliberately carries no path content."""


# Every config-derived or override value reaching the SSH command must match
# this in its final (expanded, resolved) form.  Anything else is refused, never
# escaped, so neither the shell, ssh's ``-o`` grammar nor ssh's token, tilde or
# environment expansion can change its meaning.
_SSH_SAFE_VALUE = re.compile(r"[A-Za-z0-9._/+,:@-]+")
_SSH_HOSTNAME = re.compile(r"[A-Za-z0-9.-]+|[0-9A-Fa-f:.]*:[0-9A-Fa-f:.]*")
_SSH_CONFIG_MAX_BYTES = 256 * 1024
_SSH_CONFIG_KEYS = {
    "identityfile": "IdentityFile", "hostname": "HostName", "port": "Port",
    "identitiesonly": "IdentitiesOnly", "userknownhostsfile": "UserKnownHostsFile",
}
_SSH_CONFIG_IGNORED = (
    "Ignoring ~/.ssh/config for trusted Git transport: it is not an operator-owned, "
    "agent-inaccessible regular file."
)
_SSH_IDENTITY_REFUSED = (
    "Trusted SSH identity must be an operator-owned, agent-inaccessible regular file "
    "with a plain path."
)
_SSH_IDENTITY_WINDOWS = "Explicit SSH identity selection is not supported on Windows."
_EMITTED_WARNINGS: set[str] = set()


def _warn(message: str) -> None:
    """Print a fixed, content-free warning once per process."""
    if message not in _EMITTED_WARNINGS:
        _EMITTED_WARNINGS.add(message)
        print(message, file=sys.stderr)


def _ssh_value_warning(key: str) -> str:
    return (f"Ignoring ~/.ssh/config {_SSH_CONFIG_KEYS[key]} for trusted Git transport: "
            "the value is unsupported or is not an operator-owned, agent-inaccessible plain path.")


_MAX_SYMLINK_HOPS = 40


def _forbidden_root_forms(forbidden_roots: tuple[Path, ...]) -> tuple[str, ...]:
    forms: dict[str, None] = {}
    for root in forbidden_roots:
        forms[os.path.abspath(root)] = None
        forms[os.path.realpath(root)] = None
    return tuple(forms)


def _reject_forbidden(candidate: Path, roots: tuple[str, ...]) -> None:
    for root in roots:
        try:
            if os.path.commonpath((root, str(candidate))) == root:
                raise _UntrustedPath()
        except ValueError:
            continue


def _check_component(path: Path, info: os.stat_result, *, uid: int, final: bool) -> None:
    """Owner must be ``uid`` or root; a non-symlink must not be group/world-writable.

    Root-owned sticky directories (``/tmp``) are exempt from the write check
    unless they are the final component or the parent of an accepted missing file.
    """
    if info.st_uid not in (uid, 0):
        raise _UntrustedPath()
    if not stat.S_ISLNK(info.st_mode) and info.st_mode & 0o022:
        sticky_root_dir = (info.st_uid == 0 and stat.S_ISDIR(info.st_mode)
                           and info.st_mode & stat.S_ISVTX)
        if final or not sticky_root_dir:
            raise _UntrustedPath()


def _trusted_resolve(path: Path, *, uid: int, roots: tuple[str, ...],
                     allow_missing: bool) -> tuple[Path, os.stat_result | None]:
    """Resolve ``path`` one component at a time, checking every component traversed.

    Each symlink is expanded in place, so every intermediate link, every
    directory it passes through and every selection path the walk visits must
    be operator- or root-owned, writable by nobody else and outside every
    forbidden root.  Returns the resolved path and its lstat (None if missing).
    """
    _check_component(Path("/"), Path("/").lstat(), uid=uid, final=False)
    current = Path("/")
    pending = list(path.parts[1:])
    hops = 0
    while pending:
        name = pending.pop(0)
        if name in ("", "."):
            continue
        if name == "..":
            current = current.parent
            continue
        candidate = current / name
        _reject_forbidden(candidate, roots)
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            if not allow_missing or pending:
                raise _UntrustedPath() from None
            parent_info = current.lstat()
            if not stat.S_ISDIR(parent_info.st_mode):
                raise _UntrustedPath() from None
            _check_component(current, parent_info, uid=uid, final=True)
            return candidate, None
        _check_component(candidate, info, uid=uid, final=not pending)
        if stat.S_ISLNK(info.st_mode):
            hops += 1
            if hops > _MAX_SYMLINK_HOPS:
                raise _UntrustedPath()
            target = Path(os.readlink(candidate))
            if target.is_absolute():
                current = Path("/")
                pending = list(target.parts[1:]) + pending
            else:
                pending = list(target.parts) + pending
            continue
        current = candidate
    # Describe the resolved final component itself, even when no component
    # was traversed (``/``) or the last hop resolved to a directory such as ``/``.
    return current, current.lstat()


def _trusted_operator_file(path: Path, *, forbidden_roots: tuple[Path, ...],
                           allow_missing: bool = False) -> TrustedFile:
    """Validate an operator file that a same-user agent must not control (POSIX).

    Every component traversed while resolving the path, including each
    intermediate symlink and the directories its target passes through, must
    be owned by the current uid or root, must not be group/world-writable
    (symlinks excepted) and must lie outside every forbidden root.  The target
    must be a regular file.  With ``allow_missing`` a missing final file is
    accepted when its resolved parent passes.
    """
    if sys.platform == "win32" or not path.is_absolute() or ".." in path.parts:
        raise _UntrustedPath()
    roots = _forbidden_root_forms(forbidden_roots)
    _reject_forbidden(path, roots)
    try:
        resolved, info = _trusted_resolve(path, uid=os.getuid(), roots=roots,
                                          allow_missing=allow_missing)
    except OSError as exc:
        raise _UntrustedPath() from exc
    if info is None:
        return TrustedFile(path=resolved, identity=None)
    if not stat.S_ISREG(info.st_mode):
        raise _UntrustedPath()
    return TrustedFile(path=resolved, identity=(info.st_dev, info.st_ino))


def _ssh_safe_value(value: str) -> str:
    if not _SSH_SAFE_VALUE.fullmatch(value) or value.startswith("-"):
        raise _UntrustedPath()
    return value


def _read_ssh_config(home: Path, forbidden_roots: tuple[Path, ...]) -> str | None:
    """Read ~/.ssh/config only when the operator alone controls it."""
    if sys.platform == "win32":
        return None
    config = home / ".ssh" / "config"
    try:
        config.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        _warn(_SSH_CONFIG_IGNORED)
        return None
    try:
        trusted = _trusted_operator_file(config, forbidden_roots=forbidden_roots)
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(trusted.path, flags)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != trusted.identity:
                raise _UntrustedPath()
            chunks: list[bytes] = []
            remaining = _SSH_CONFIG_MAX_BYTES + 1
            while remaining > 0:
                chunk = os.read(fd, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
        finally:
            os.close(fd)
        data = b"".join(chunks)
        if len(data) > _SSH_CONFIG_MAX_BYTES:
            raise _UntrustedPath()
        return data.decode("utf-8")
    except (_UntrustedPath, OSError, UnicodeDecodeError):
        _warn(_SSH_CONFIG_IGNORED)
        return None


def _split_ssh_args(text: str) -> list[str] | None:
    """Split an ssh_config argument list: whitespace, double quotes, ``#`` comments."""
    args: list[str] = []
    index, length = 0, len(text)
    while index < length:
        if text[index] in " \t":
            index += 1
            continue
        if text[index] == "#":
            break
        token: list[str] = []
        quoted = False
        while index < length:
            char = text[index]
            if quoted:
                if char == '"':
                    quoted = False
                else:
                    token.append(char)
            elif char in " \t":
                break
            elif char == '"':
                quoted = True
            else:
                token.append(char)
            index += 1
        if quoted:
            return None
        args.append("".join(token))
    return args


def _host_pattern_matches(pattern: str, host: str) -> bool:
    regex = "".join(".*" if char == "*" else "." if char == "?" else re.escape(char)
                    for char in pattern)
    return re.fullmatch(regex, host, flags=re.IGNORECASE | re.DOTALL) is not None


def _host_block_applies(patterns: list[str], host: str) -> bool:
    matched = False
    for pattern in patterns:
        if pattern.startswith("!"):
            if _host_pattern_matches(pattern[1:], host):
                return False
        elif _host_pattern_matches(pattern, host):
            matched = True
    return matched


_SSH_CONFIG_LINE = re.compile(r"([^\s=]+)(?:[ \t]*=[ \t]*|[ \t]+|$)(.*)")


def _ssh_host_options(text: str, host: str) -> dict[str, Any]:
    """Resolve the honoured ssh_config subset for ``host`` without I/O.

    Only ``Host`` blocks and IdentityFile, HostName, Port, IdentitiesOnly and
    UserKnownHostsFile are honoured.  A ``Match`` block is skipped until the
    next ``Host`` header; ``Include`` is never opened, and every other key
    (User, ProxyCommand, ProxyJump, LocalCommand, KnownHostsCommand, ...) is
    dropped.  Scalars keep their first value; IdentityFile accumulates.
    Values are the raw argument lists for later validation.
    """
    options: dict[str, Any] = {"identityfile": []}
    active = True
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SSH_CONFIG_LINE.fullmatch(line)
        if match is None:
            continue
        key = match.group(1).lower()
        args = _split_ssh_args(match.group(2))
        if key == "host":
            active = bool(args) and _host_block_applies(args, host)
            continue
        if key == "match":
            active = False
            continue
        if not active or key not in _SSH_CONFIG_KEYS:
            continue
        value = tuple(args) if args else None
        if key == "identityfile":
            options["identityfile"].append(value)
        elif key not in options:
            options[key] = value
    return options


def _expand_operator_path(raw: str, home: Path | None) -> Path:
    if "%" in raw or "${" in raw:
        raise _UntrustedPath()
    if raw.startswith("~/"):
        if home is None:
            raise _UntrustedPath()
        raw = str(home) + raw[1:]
    path = Path(raw)
    if not path.is_absolute():
        raise _UntrustedPath()
    _ssh_safe_value(str(path))
    return path


def _trusted_ssh_path(raw: str, *, home: Path | None, forbidden_roots: tuple[Path, ...],
                      allow_missing: bool = False) -> str:
    path = _expand_operator_path(raw, home)
    trusted = _trusted_operator_file(path, forbidden_roots=forbidden_roots,
                                     allow_missing=allow_missing)
    return _ssh_safe_value(str(trusted.path))


def _single(value: tuple[str, ...] | None) -> str:
    if value is None or len(value) != 1:
        raise _UntrustedPath()
    return value[0]


def _resolve_ssh_options(parsed: dict[str, Any], *, url_has_port: bool, home: Path | None,
                         forbidden_roots: tuple[Path, ...],
                         explicit_identity: Path | None) -> list[str]:
    """Validate parsed values into ssh arguments; refuse rather than escape."""
    override: str | None = None
    if explicit_identity is not None:
        if sys.platform == "win32":
            raise AgentLoopError(_SSH_IDENTITY_WINDOWS)
        try:
            override = _trusted_ssh_path(str(explicit_identity), home=None,
                                         forbidden_roots=forbidden_roots)
        except _UntrustedPath:
            raise AgentLoopError(_SSH_IDENTITY_REFUSED) from None
    options: list[str] = []

    def resolve(key: str, check: Any) -> None:
        if key not in parsed:
            return
        try:
            value = check(_single(parsed[key]))
        except _UntrustedPath:
            _warn(_ssh_value_warning(key))
            return
        options.extend(("-o", f"{_SSH_CONFIG_KEYS[key]}={value}"))

    def hostname(value: str) -> str:
        if "%" in value or not _SSH_HOSTNAME.fullmatch(value):
            raise _UntrustedPath()
        return _ssh_safe_value(value)

    def port(value: str) -> str:
        if not re.fullmatch(r"[0-9]{1,5}", value) or not 1 <= int(value) <= 65535:
            raise _UntrustedPath()
        return str(int(value))

    def identities_only(value: str) -> str:
        if value.lower() not in ("yes", "no"):
            raise _UntrustedPath()
        return value.lower()

    def known_hosts(value: str) -> str:
        return _trusted_ssh_path(value, home=home, forbidden_roots=forbidden_roots,
                                 allow_missing=True)

    resolve("hostname", hostname)
    if not url_has_port:
        resolve("port", port)
    if override is not None:
        options.extend(("-o", "IdentitiesOnly=yes"))
    else:
        resolve("identitiesonly", identities_only)
    resolve("userknownhostsfile", known_hosts)
    if override is not None:
        options.extend(("-i", override))
        return options
    for entry in parsed.get("identityfile", ()):
        try:
            identity = _trusted_ssh_path(_single(entry), home=home,
                                         forbidden_roots=forbidden_roots)
        except _UntrustedPath:
            _warn(_ssh_value_warning("identityfile"))
            continue
        options.extend(("-i", identity))
    return options


def _ssh_url_host(url: str) -> tuple[str, bool]:
    """Return the literal host of a trusted SSH URL and whether it names a port."""
    if url.startswith("ssh://"):
        parts = urlsplit(url)
        return parts.hostname or "", parts.port is not None
    return url.split("@", 1)[1].split(":", 1)[0], False


def _transport_env(url: str, gh_cmd: str | None, *, ssh_identity: Path | None = None,
                   forbidden_roots: tuple[Path, ...] = ()) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key in {
        "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TZ", "TMPDIR",
        "SSL_CERT_FILE", "SSL_CERT_DIR", "SSH_AUTH_SOCK",
    }}
    env.update({
        "PATH": os.path.dirname(_git_path()),
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "/bin/false",
        "GIT_PAGER": "cat", "GIT_NO_LAZY_FETCH": "1",
    })
    if url.startswith("ssh://") or url.startswith("git@"):
        if sys.platform == "win32":
            if ssh_identity is not None:
                raise AgentLoopError(_SSH_IDENTITY_WINDOWS)
            ssh = r"C:\Program Files\Git\usr\bin\ssh.exe"
        else:
            ssh = "/usr/bin/ssh"
        if not os.path.isfile(ssh):
            raise AgentLoopError("Trusted SSH transport requires an operator-owned SSH client.")
        try:
            trusted_ssh = _trusted_executable(ssh)
        except (OSError, AgentLoopError) as exc:
            raise AgentLoopError("Trusted SSH transport requires an agent-inaccessible SSH client.") from exc
        # Git interprets this value through a shell. An explicit empty config
        # preserves agent/default-key authentication without user ProxyCommand,
        # LocalCommand, Match exec, or Include directives.
        quoted_ssh = f'"{trusted_ssh}"' if sys.platform == "win32" else shlex.quote(trusted_ssh)
        command = (
            f"{quoted_ssh} -F {os.devnull} -o ProxyCommand=none "
            "-o PermitLocalCommand=no -o ClearAllForwardings=yes"
        )
        if sys.platform != "win32":
            # Unknown or changed host keys fail closed without prompting;
            # pinned before any config-derived option.  BatchMode stays unset
            # so an encrypted key can still prompt for its passphrase.
            command += " -o StrictHostKeyChecking=yes"
            raw_home = os.environ.get("HOME", "")
            home = Path(raw_home) if raw_home and os.path.isabs(raw_home) else None
            host, url_has_port = _ssh_url_host(url)
            text = _read_ssh_config(home, forbidden_roots) if home is not None else None
            parsed = _ssh_host_options(text, host) if text is not None else {}
            options = _resolve_ssh_options(
                parsed, url_has_port=url_has_port, home=home,
                forbidden_roots=forbidden_roots, explicit_identity=ssh_identity,
            )
            if options:
                command += " " + " ".join(shlex.quote(option) for option in options)
        env["GIT_SSH_COMMAND"] = command
    if url.startswith("https://") and gh_cmd:
        # Public repositories need no token.  A private-repository retry below
        # obtains one from the trusted GitHub CLI context only when necessary.
        pass
    return env


# Most specific evidence first: Git appends its generic "correct access rights"
# footer to network and not-found failures too, so it is only a fallback.
_FAILURE_EVIDENCE: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("remote not found", ("repository not found", "does not appear to be a git repository",
                          "returned error: 404")),
    ("host-key failure", ("host key verification failed",
                          "remote host identification has changed", "host key is known")),
    ("network or timeout", ("could not resolve host", "connection timed out",
                            "connection refused", "network is unreachable", "failed to connect",
                            "operation timed out", "connection closed by remote host",
                            "kex_exchange_identification")),
    ("authentication failure", ("permission denied (publickey", "authentication failed",
                                "could not read username", "returned error: 401",
                                "returned error: 403")),
    ("authentication failure", ("correct access rights",)),
)
_SSH_FAILURE_HINTS = {
    "authentication failure": (
        " The SSH server rejected the offered keys; select the key with --git-ssh-identity"
        " PATH (or AGENT_LOOP_GIT_SSH_IDENTITY), or set IdentityFile for this host in"
        " ~/.ssh/config."
    ),
    "host-key failure": (
        " The server's host key must already be in ~/.ssh/known_hosts (or the trusted"
        " UserKnownHostsFile); unknown or changed keys are refused without prompting."
    ),
}


class TransportFailure(AgentLoopError):
    """A failed or timed-out transport Git command.

    ``label`` is the fixed, non-secret failure class (``remote not found``,
    ``host-key failure``, ``network or timeout``, ``authentication failure``
    or ``unclassified``), so callers need not parse the message (#1383).
    """

    def __init__(self, message: str, *, label: str):
        super().__init__(message)
        self.label = label


def _classify_transport_failure(stderr: bytes) -> str:
    text = stderr.decode("utf-8", "replace").lower()
    for label, needles in _FAILURE_EVIDENCE:
        if any(needle in text for needle in needles):
            return label
    return "unclassified"


def _private_git(args: tuple[str, ...], *, cwd: Path, env: dict[str, str],
                 input_bytes: bytes | None = None, timeout: int = 90,
                 ssh: bool = False) -> bytes:
    command = (_git_path(), "-c", "credential.helper=", "-c", "core.hooksPath=/dev/null",
               "-c", "core.fsmonitor=false", "-c", "protocol.allow=never",
               "-c", "protocol.https.allow=always", "-c", "protocol.ssh.allow=always",
               "-c", "protocol.file.allow=always", "-c", "http.followRedirects=false", *args)
    try:
        # No transport prompt may read the caller's stdin.
        stdin = {"input": input_bytes} if input_bytes is not None else {"stdin": subprocess.DEVNULL}
        result = subprocess.run(command, cwd=cwd, env=env, **stdin,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                check=False, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise TransportFailure("Trusted Git transport timed out (network or timeout).",
                               label="network or timeout") from exc
    if result.returncode:
        # Do not expose command arguments or errors: either may contain a
        # credential returned by a remote server.  Only a fixed class is shown.
        label = _classify_transport_failure(result.stderr or b"")
        hint = _SSH_FAILURE_HINTS.get(label, "") if ssh else ""
        raise TransportFailure(
            f"Trusted Git transport failed with exit {result.returncode} ({label}).{hint}",
            label=label,
        )
    return result.stdout


def _token_from_gh(gh_cmd: str, destination: Path, host: str) -> str:
    if os.path.isabs(gh_cmd):
        candidates: tuple[str, ...] = (gh_cmd,)
    elif gh_cmd == "gh":
        candidates = ((r"C:\Program Files\GitHub CLI\gh.exe",) if sys.platform == "win32"
                      else ("/usr/bin/gh", "/usr/local/bin/gh", "/opt/homebrew/bin/gh"))
    else:
        candidates = ()
    resolved = _first_trusted_executable(
        candidates,
        missing="Trusted GitHub authentication is unavailable.",
        untrusted="Trusted GitHub CLI must be in an agent-inaccessible installation.",
    )
    try:
        if os.path.commonpath((resolved, os.path.realpath(destination))) == os.path.realpath(destination):
            raise AgentLoopError("Trusted GitHub CLI must be outside the agent checkout.")
    except ValueError:
        pass
    try:
        result = subprocess.run((resolved, "auth", "token", "--hostname", host), cwd=tempfile.gettempdir(),
                                env={key: value for key, value in os.environ.items()
                                     if key in {"HOME", "USER", "LOGNAME", "GH_HOST", "GH_CONFIG_DIR",
                                                "GH_TOKEN", "GITHUB_TOKEN"}},
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AgentLoopError("Trusted GitHub authentication is unavailable.") from exc
    token = result.stdout.decode("utf-8", "replace").strip()
    if result.returncode or not token or "\n" in token or "\r" in token:
        raise AgentLoopError("Trusted GitHub authentication is unavailable.")
    return token


def _origin_from_checkout(destination: Path, runner: Any) -> str:
    result = runner.run(("git", "config", "--local", "--no-includes", "--get", "remote.origin.url"),
                        cwd=destination, check=False)
    if result.returncode or not result.stdout.strip():
        raise AgentLoopError(f"Checkout {destination} has no origin URL.")
    return result.stdout.strip()


def _destination_tips(destination: Path, runner: Any) -> list[str]:
    """Return the destination's distinct well-formed ref tips (exclusion hints only)."""
    try:
        listed = runner.run(("git", "for-each-ref", "--format=%(objectname)"),
                            cwd=destination, check=False)
    except AgentLoopError:
        return []
    if listed.returncode:
        return []
    return list(dict.fromkeys(line.strip() for line in listed.stdout.splitlines()
                              if _SHA.fullmatch(line.strip())))


def _private_commits(private: Path, env: dict[str, str], candidates: list[str]) -> list[str]:
    """Keep only candidates the private repository holds as commits.

    ``pack-objects --revs`` fails on a negative it cannot resolve, so a
    local-only commit, tag or tree is never passed.
    """
    if not candidates:
        return []
    try:
        report = _private_git(("cat-file", "--batch-check"), cwd=private, env=env,
                              input_bytes="".join(f"{tip}\n" for tip in candidates).encode("ascii"))
    except AgentLoopError:
        return []
    kept = []
    for line in report.decode("ascii", "replace").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[1] == "commit" and fields[0] in candidates:
            kept.append(fields[0])
    return kept


def _closure_complete(destination: Path, sha: str, runner: Any) -> bool:
    """Return True only when destination holds every object reachable from sha.

    No history is excluded: destination refs are hints, never proof of
    completeness, so this is the single gate before the tracking-ref CAS.
    """
    try:
        walked = runner.run(("git", "rev-list", "--objects", "--missing=error", "--quiet", sha),
                            cwd=destination, check=False)
    except AgentLoopError:
        return False
    return walked.returncode == 0


def _stream_pack(private: Path, env: dict[str, str], destination: Path,
                 revs: tuple[str, ...], *, thin: bool) -> bool:
    """Stream ``pack-objects`` for revs from private into a confined ``index-pack``."""
    local_argv, local_env, pass_fds = local_command(
        ("index-pack", "--stdin", "--fix-thin"), checkout=destination
    )
    export = subprocess.Popen(
        (_git_path(), "pack-objects", "--stdout", "--revs", *(("--thin",) if thin else ())),
        cwd=private, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert export.stdin is not None and export.stdout is not None
        export.stdin.write("".join(f"{rev}\n" for rev in revs).encode("ascii"))
        export.stdin.close()
        imported = subprocess.run(local_argv, cwd=destination, env=local_env,
                                  pass_fds=pass_fds, stdin=export.stdout,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  check=False, timeout=90)
        export.stdout.close()
        export.wait(timeout=90)
    finally:
        if export.poll() is None:
            export.kill()
            export.wait()
    return not (imported.returncode or export.returncode)


def import_ref(destination: Path, source_ref: str, tracking_ref: str, *,
               repo: str, runner: Any, gh_cmd: str | None = None,
               local_origin: Path | None = None, ssh_identity: Path | None = None,
               forbidden_roots: tuple[Path, ...] = ()) -> str:
    """Fetch one ref, stream its pack into destination, then CAS tracking_ref.

    The pack excludes history reachable from destination ref tips that the
    fetched history also holds, and is skipped when sha is already a ref tip.
    Every path verifies sha's complete closure before the CAS and falls back
    once to a full export when the incremental import is incomplete.

    ``forbidden_roots`` names every agent-controlled checkout, store and run
    root; SSH config, identity and known-hosts files there are refused.
    """
    if not _REF.fullmatch(source_ref) or not _REF.fullmatch(tracking_ref):
        raise AgentLoopError("Unsafe Git ref for private transport.")
    observed = _origin_from_checkout(destination, runner)
    url = trusted_url(repo, observed, local_origin=local_origin)
    with tempfile.TemporaryDirectory(prefix="agent-loop-transport-") as raw:
        private = Path(raw)
        is_ssh = url.startswith("ssh://") or url.startswith("git@")
        env = _transport_env(url, gh_cmd, ssh_identity=ssh_identity,
                             forbidden_roots=(*forbidden_roots, destination))
        _private_git(("init", "--bare", "-q"), cwd=private, env=env)
        refspec = f"+{source_ref}:refs/agent-loop/import"
        try:
            _private_git(("fetch", "--no-tags", "--no-recurse-submodules", url, refspec),
                         cwd=private, env=env, ssh=is_ssh)
        except AgentLoopError:
            if not (url.startswith("https://") and gh_cmd):
                raise
            host = urlsplit(url).hostname
            if not host:
                raise AgentLoopError("Trusted HTTPS origin has no host.")
            token = _token_from_gh(gh_cmd, destination, host)
            header = "AUTHORIZATION: basic " + base64.b64encode(
                f"x-access-token:{token}".encode()).decode()
            env.update({"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0":
                        f"http.{url}.extraheader", "GIT_CONFIG_VALUE_0": header})
            _private_git(("fetch", "--no-tags", "--no-recurse-submodules", url, refspec),
                         cwd=private, env=env)
            for key in ("GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0"):
                env.pop(key, None)
        sha = _private_git(("rev-parse", "refs/agent-loop/import^{commit}"),
                           cwd=private, env=env).decode("ascii").strip()
        if not _SHA.fullmatch(sha):
            raise AgentLoopError("Trusted fetch returned an invalid commit identity.")
        tips = _destination_tips(destination, runner)
        # An existing complete ref tip needs no pack; otherwise try a thin
        # pack excluding usable tips, then fall back once to a full export.
        complete = sha in tips and _closure_complete(destination, sha, runner)
        if not complete and sha not in tips:
            negatives = _private_commits(private, env, tips)
            if negatives:
                revs = (sha, *(f"^{tip}" for tip in negatives))
                try:
                    complete = (_stream_pack(private, env, destination, revs, thin=True)
                                and _closure_complete(destination, sha, runner))
                except (OSError, subprocess.TimeoutExpired):
                    complete = False
        if not complete and not (
                _stream_pack(private, env, destination, (sha,), thin=False)
                and _closure_complete(destination, sha, runner)):
            raise AgentLoopError("Verified object-pack import failed.")
        verified = runner.run(("git", "rev-parse", "--verify", f"{sha}^{{commit}}"),
                              cwd=destination)
        if verified.stdout.strip() != sha:
            raise AgentLoopError("Imported commit did not verify in destination object database.")
        previous = runner.run(("git", "show-ref", "--verify", "--hash", tracking_ref),
                              cwd=destination, check=False)
        old = previous.stdout.strip() if previous.returncode == 0 else "0" * len(sha)
        runner.run(("git", "update-ref", tracking_ref, sha, old), cwd=destination)
        return sha
