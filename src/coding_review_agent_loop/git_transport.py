"""Fetch trusted refs privately and import their reachable objects into Git state.

No transport command runs with an agent-controlled checkout as its repository.
Callers hold their checkout/store lock through the import and ref update.
"""

from __future__ import annotations

import base64
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .errors import AgentLoopError
from .secure_git import _git_path, _trusted_executable, local_command

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
            origin = f"git@{host}:{owner}/{name}.git"
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
    if observed not in allowed:
        raise AgentLoopError(
            f"Checkout origin does not match the operator-trusted endpoint for {repo}; refusing transport."
        )
    return observed


def _transport_env(url: str, gh_cmd: str | None) -> dict[str, str]:
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
            ssh = r"C:\Program Files\Git\usr\bin\ssh.exe"
        else:
            ssh = "/usr/bin/ssh"
        if not os.path.isfile(ssh):
            raise AgentLoopError("Trusted SSH transport requires an operator-owned SSH client.")
        try:
            env["GIT_SSH_COMMAND"] = _trusted_executable(ssh)
        except (OSError, AgentLoopError) as exc:
            raise AgentLoopError("Trusted SSH transport requires an agent-inaccessible SSH client.") from exc
    if url.startswith("https://") and gh_cmd:
        # Public repositories need no token.  A private-repository retry below
        # obtains one from the trusted GitHub CLI context only when necessary.
        pass
    return env


def _private_git(args: tuple[str, ...], *, cwd: Path, env: dict[str, str],
                 input_bytes: bytes | None = None, timeout: int = 90) -> bytes:
    command = (_git_path(), "-c", "credential.helper=", "-c", "core.hooksPath=/dev/null",
               "-c", "core.fsmonitor=false", "-c", "protocol.allow=never",
               "-c", "protocol.https.allow=always", "-c", "protocol.ssh.allow=always",
               "-c", "protocol.file.allow=always", "-c", "http.followRedirects=false", *args)
    try:
        result = subprocess.run(command, cwd=cwd, env=env, input=input_bytes,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                check=False, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise AgentLoopError("Trusted Git transport timed out.") from exc
    if result.returncode:
        # Do not expose command arguments or errors: either may contain a
        # credential returned by a remote server.
        raise AgentLoopError(f"Trusted Git transport failed with exit {result.returncode}.")
    return result.stdout


def _token_from_gh(gh_cmd: str, destination: Path, host: str) -> str:
    if os.path.isabs(gh_cmd):
        candidate = gh_cmd
    elif gh_cmd == "gh":
        candidates = ([r"C:\Program Files\GitHub CLI\gh.exe"] if sys.platform == "win32"
                      else ["/usr/bin/gh", "/usr/local/bin/gh", "/opt/homebrew/bin/gh"])
        candidate = next((path for path in candidates if os.path.isfile(path)), "")
    else:
        candidate = ""
    if not candidate:
        raise AgentLoopError("Trusted GitHub authentication is unavailable.")
    try:
        resolved = _trusted_executable(candidate)
    except (OSError, AgentLoopError) as exc:
        raise AgentLoopError("Trusted GitHub CLI must be in an agent-inaccessible installation.") from exc
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


def import_ref(destination: Path, source_ref: str, tracking_ref: str, *,
               repo: str, runner: Any, gh_cmd: str | None = None,
               local_origin: Path | None = None) -> str:
    """Fetch one ref, stream its pack into destination, then CAS tracking_ref."""
    if not _REF.fullmatch(source_ref) or not _REF.fullmatch(tracking_ref):
        raise AgentLoopError("Unsafe Git ref for private transport.")
    observed = _origin_from_checkout(destination, runner)
    url = trusted_url(repo, observed, local_origin=local_origin)
    with tempfile.TemporaryDirectory(prefix="agent-loop-transport-") as raw:
        private = Path(raw)
        env = _transport_env(url, gh_cmd)
        _private_git(("init", "--bare", "-q"), cwd=private, env=env)
        refspec = f"+{source_ref}:refs/agent-loop/import"
        try:
            _private_git(("fetch", "--no-tags", "--no-recurse-submodules", url, refspec),
                         cwd=private, env=env)
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
        local_argv, local_env, pass_fds = local_command(
            ("index-pack", "--stdin", "--fix-thin"), checkout=destination
        )
        export = subprocess.Popen(
            (_git_path(), "pack-objects", "--stdout", "--revs"), cwd=private,
            env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            assert export.stdin is not None and export.stdout is not None
            export.stdin.write((sha + "\n").encode("ascii"))
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
        if imported.returncode or export.returncode:
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
