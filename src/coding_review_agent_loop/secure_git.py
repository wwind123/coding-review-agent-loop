"""Command-time boundary for agent-loop-owned local Git operations.

The checkout and its linked administrative directory are input, never a source
of executable policy.  A missing execution-denial backend is a hard error.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import stat
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Mapping, Sequence

from .errors import AgentLoopError

_LOCK = threading.Lock()
_PINNED_GIT: str | None = None
_PINNED_GIT_HASH: bytes | None = None
_GUARD_FD: int | None = None
_WINDOWS_VERIFIED = False
_HELPER_HASHES = {
    "git_exec_guard.c": "09286e7b31740b482a2a3f867723bcc3716d27601c9bd1cbbe4ff8bcd533980c",
    "git_exec_guard_macos.c": "2890ee03e52bb997023b5ba1cadbdf555d78f72249a4fab9cdbc56d0ef0f0b46",
    "git_windows_launcher.py": "49a10e4312dfa420f6abe540e7dca3e282def3377219ae5b705d9928149631d7",
}

_CONFIG = (
    "core.hooksPath=/dev/null", "core.fsmonitor=false", "core.pager=cat",
    "core.sshCommand=ssh", "diff.external=", "diff.trustExitCode=false",
    "log.showSignature=false", "gpg.program=false", "gpg.ssh.program=false",
    "gpg.x509.program=false", "protocol.allow=never", "protocol.ext.allow=never",
    "submodule.recurse=false", "status.submoduleSummary=false",
    "diff.ignoreSubmodules=all", "gc.auto=0", "credential.helper=",
    "trace2.eventTarget=", "trace2.perfTarget=", "trace2.normalTarget=",
)
_ENV_KEYS = frozenset({
    "HOME", "USER", "LOGNAME", "LANG", "LANGUAGE", "LC_ALL", "TZ",
    "TMPDIR", "SSL_CERT_FILE", "SSL_CERT_DIR",
})


def _trusted_path(path: str, *, executable: bool) -> str:
    """Require an installation that a same-user unrestricted agent cannot replace."""
    if not os.path.isabs(path):
        raise AgentLoopError("Trusted executable must have an absolute path.")
    resolved = os.path.realpath(path)
    if sys.platform == "win32":
        # POSIX ownership does not describe Windows ACLs. Restrict launch to
        # the standard system installation roots, never a caller-supplied
        # per-user path or a junction resolving outside those roots.
        protected = (r"C:\Program Files", r"C:\Program Files (x86)")
        if not any(os.path.commonpath((candidate, resolved)).casefold() == candidate.casefold()
                   for candidate in protected):
            raise AgentLoopError("Trusted executable must be in a system installation.")
        if not os.path.isfile(resolved) or (executable and not os.access(resolved, os.X_OK)):
            raise AgentLoopError("Trusted executable is unavailable.")
        return resolved
    for name in (path, resolved):
        current = Path(name)
        while True:
            info = current.lstat()
            if info.st_uid != 0 or (not stat.S_ISLNK(info.st_mode) and info.st_mode & 0o022):
                raise AgentLoopError("Trusted executable must be in a root-owned installation.")
            if current.parent == current:
                break
            current = current.parent
    info = os.stat(resolved)
    if not stat.S_ISREG(info.st_mode) or (executable and not os.access(resolved, os.X_OK)):
        raise AgentLoopError("Trusted executable is unavailable.")
    return resolved


def _trusted_executable(path: str) -> str:
    return _trusted_path(path, executable=True)


def _trusted_helper(name: str) -> bytes:
    """Read a helper only when it matches the digest in loaded controller code."""
    expected = _HELPER_HASHES.get(name)
    if expected is None:
        raise AgentLoopError("Unsupported Git confinement: unknown helper.")
    try:
        source = Path(__file__).with_name(name).read_bytes()
    except OSError as exc:
        raise AgentLoopError("Unsupported Git confinement: helper is unavailable.") from exc
    if hashlib.sha256(source).hexdigest() != expected:
        raise AgentLoopError("Unsupported Git confinement: helper identity changed.")
    return source


def _git_path() -> str:
    global _PINNED_GIT, _PINNED_GIT_HASH
    if _PINNED_GIT is None:
        # Resolve independently of checkout-controlled PATH.  The operator's
        # system installation is the supported Linux backend.
        if sys.platform == "win32":
            candidate = next((item for item in (
                r"C:\Program Files\Git\mingw64\bin\git.exe",
                r"C:\Program Files\Git\bin\git.exe",
                r"C:\Program Files (x86)\Git\mingw64\bin\git.exe",
            ) if os.path.isfile(item)), None)
        elif sys.platform == "darwin":
            candidate = next((item for item in (
                "/opt/homebrew/bin/git", "/usr/local/bin/git", "/usr/bin/git"
            ) if os.path.isfile(item)), None)
        else:
            candidate = next((item for item in ("/usr/bin/git", "/bin/git", "/usr/local/bin/git")
                              if os.path.isfile(item)), None)
        if not candidate:
            raise AgentLoopError("Unsupported Git confinement: no pinned Git executable.")
        try:
            path = _trusted_executable(candidate)
        except (OSError, AgentLoopError) as exc:
            raise AgentLoopError("Unsupported Git confinement: pinned Git is not in a trusted installation.") from exc
        _PINNED_GIT = path
        _PINNED_GIT_HASH = hashlib.sha256(Path(path).read_bytes()).digest()
    elif hashlib.sha256(Path(_PINNED_GIT).read_bytes()).digest() != _PINNED_GIT_HASH:
        raise AgentLoopError("Unsupported Git confinement: pinned Git changed during the run.")
    return _PINNED_GIT


def _guard_fd() -> int:
    global _GUARD_FD
    with _LOCK:
        if _GUARD_FD is not None:
            return _GUARD_FD
        if sys.platform not in {"linux", "darwin"}:
            raise AgentLoopError(
                "Unsupported Git confinement: no verified command-time execution-denial backend."
            )
        try:
            source = _trusted_helper(
                "git_exec_guard_macos.c" if sys.platform == "darwin" else "git_exec_guard.c"
            )
        except (OSError, AgentLoopError) as exc:
            raise AgentLoopError("Unsupported Git confinement: guard source identity could not be verified.") from exc
        try:
            with tempfile.TemporaryDirectory(prefix="agent-loop-git-guard-") as directory:
                library = Path(directory) / ("guard.dylib" if sys.platform == "darwin" else "guard.so")
                compiler = next((item for item in ("/usr/bin/cc", "/usr/bin/clang")
                                 if os.path.isfile(item)), None)
                if not compiler:
                    raise OSError("C compiler unavailable")
                compiler = _trusted_executable(compiler)
                built = subprocess.run(
                    (compiler, *( ("-dynamiclib",) if sys.platform == "darwin" else ("-shared", "-fPIC") ),
                    "-O2", "-x", "c", "-", "-o", str(library)),
                    input=source, capture_output=True, timeout=30, check=False,
                )
                if built.returncode:
                    raise OSError("execution guard could not be built")
                if sys.platform == "darwin":
                    fd = os.open(library, os.O_RDONLY)
                else:
                    import fcntl

                    fd = os.memfd_create("agent-loop-git-guard", os.MFD_ALLOW_SEALING)
                    os.write(fd, library.read_bytes())
                    fcntl.fcntl(fd, fcntl.F_ADD_SEALS,
                                fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW |
                                fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
            _verify_guard(fd)
        except (OSError, subprocess.TimeoutExpired) as exc:
            if "fd" in locals():
                os.close(fd)
            raise AgentLoopError(f"Unsupported Git confinement: {exc}") from exc
        _GUARD_FD = fd
        return fd


def _verify_guard(fd: int) -> None:
    def denied(probe: subprocess.CompletedProcess[bytes]) -> bool:
        detail = probe.stderr.lower()
        return probe.returncode != 0 and (
            b"operation not permitted" in detail or b"cannot fork" in detail
            or b"cannot exec" in detail
        )

    read_fd, write_fd = os.pipe()
    try:
        env = _closed_env({}, fd)
        env["AGENT_LOOP_GIT_GUARD_PROBE_FD"] = str(write_fd)
        probe = subprocess.run(
            (_git_path(), "--version"), env=env, pass_fds=(fd, write_fd),
            capture_output=True, timeout=5, check=False,
        )
        os.close(write_fd)
        write_fd = -1
        if probe.returncode != 0 or os.read(read_fd, 1) != b"1":
            raise OSError("pinned Git did not load the execution guard")
        # A Git external alias requests a child process.  It must fail while
        # the same preload constructor is active in the real Git process.
        canary = subprocess.run(
            (_git_path(), "-c", "alias.guard=!/usr/bin/true", "guard"),
            env=_closed_env({}, fd), pass_fds=(fd,), capture_output=True,
            timeout=5, check=False,
        )
        if not denied(canary):
            raise OSError("pinned Git launched a child under the execution guard")
        same_path = subprocess.run(
            (_git_path(), "-c", f"alias.guard=!{shlex.quote(_git_path())} --version", "guard"),
            env=_closed_env({}, fd), pass_fds=(fd,), capture_output=True,
            timeout=5, check=False,
        )
        if not denied(same_path):
            raise OSError("pinned Git could relaunch itself under the execution guard")
        if sys.platform == "linux":
            network = subprocess.run(
                (sys.executable, "-I", "-S", "-c",
                 "import errno,socket,sys\n"
                 "try: socket.socket()\n"
                 "except OSError as e: sys.exit(0 if e.errno in (errno.EPERM,errno.EACCES) else 2)\n"
                 "sys.exit(3)"),
                env=_closed_env({}, fd), pass_fds=(fd,), capture_output=True,
                timeout=5, check=False,
            )
            if network.returncode != 0:
                raise OSError("network creation was not denied under the execution guard")
    finally:
        if write_fd >= 0:
            os.close(write_fd)
        os.close(read_fd)


def _closed_env(environ: Mapping[str, str], fd: int) -> dict[str, str]:
    env = {key: value for key, value in environ.items() if key in _ENV_KEYS or key.startswith("LC_")}
    if sys.platform == "win32":
        env.update({key: value for key, value in environ.items()
                    if key in {"SystemRoot", "WINDIR", "TEMP", "TMP"}})
    else:
        env[("DYLD_INSERT_LIBRARIES" if sys.platform == "darwin" else "LD_PRELOAD")] = (
            f"/dev/fd/{fd}" if sys.platform == "darwin" else f"/proc/self/fd/{fd}"
        )
    env.update({
        "PATH": os.path.dirname(_git_path()),
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0", "GIT_NO_LAZY_FETCH": "1",
        "GIT_PAGER": "cat", "PAGER": "cat",
    })
    return env


def _verify_windows() -> None:
    global _WINDOWS_VERIFIED
    with _LOCK:
        if _WINDOWS_VERIFIED:
            return
        launcher = _trusted_helper("git_windows_launcher.py").decode("utf-8")
        if not os.path.isfile(sys.executable):
            raise AgentLoopError("Unsupported Git confinement: Windows launcher is unavailable.")
        python = _trusted_executable(sys.executable)
        prefix = (python, "-I", "-S", "-c", launcher, _git_path())
        env = _closed_env(os.environ, -1)
        try:
            version = subprocess.run((*prefix, "--version"), env=env,
                                     capture_output=True, timeout=10, check=False)
            blocked = subprocess.run((*prefix, "-c", "alias.guard=!git --version", "guard"),
                                     env=env, capture_output=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AgentLoopError("Unsupported Git confinement: Windows Job Object preflight failed.") from exc
        if version.returncode != 0 or b"git version" not in version.stdout or blocked.returncode == 0:
            raise AgentLoopError("Unsupported Git confinement: Windows Job Object child denial was not verified.")
        _WINDOWS_VERIFIED = True


def local_command(
    args: Sequence[str], *, environ: Mapping[str, str] | None = None,
    checkout: Path | None = None,
) -> tuple[list[str], dict[str, str], tuple[int, ...]]:
    """Build one confined Git invocation; preflight occurs before repo access."""
    if checkout is not None:
        root = os.path.realpath(checkout)
        for protected in (os.path.realpath(__file__), os.path.realpath(_git_path()),
                          os.path.realpath(sys.executable)):
            try:
                if os.path.commonpath((root, protected)) == root:
                    raise AgentLoopError(
                        "Unsupported Git confinement: tool code or executable lies in the agent checkout."
                    )
            except ValueError:
                pass
    if sys.platform == "win32":
        _verify_windows()
        fd = -1
    else:
        fd = _guard_fd()
    values = os.environ if environ is None else environ
    rest = [str(value) for value in args]
    prefix: list[str] = []
    while rest and rest[0].startswith("-"):
        option = rest.pop(0)
        prefix.append(option)
        if option in {"-c", "-C", "--git-dir", "--work-tree"}:
            if not rest:
                raise AgentLoopError("Malformed Git global option in local command.")
            prefix.append(rest.pop(0))
        elif option not in {"--literal-pathspecs", "--no-pager", "--no-optional-locks"}:
            raise AgentLoopError(f"Unsupported Git global option in local command: {option}.")
    command = [_git_path(), "--no-pager", *prefix]
    if checkout is not None and rest and rest[0] not in {"init", "clone"}:
        # A repository's core.worktree is agent writable. Bind both Git's
        # work-tree option and the effective config to the checkout root.
        # Some read callers start in a package directory inside that root.
        worktree_path = Path(os.path.abspath(checkout))
        for candidate in (worktree_path, *worktree_path.parents):
            if os.path.lexists(candidate / ".git"):
                worktree_path = candidate
                break
        worktree = str(worktree_path)
        command.extend(("--work-tree", worktree, "-c", f"core.worktree={worktree}"))
    for item in _CONFIG:
        command.extend(("-c", item))
    if rest and rest[0] in {"diff", "show", "log"}:
        rest[1:1] = ["--no-ext-diff", "--no-textconv", "--ignore-submodules=all"]
    elif rest and rest[0] == "status":
        rest.insert(1, "--ignore-submodules=all")
    command.extend(rest)
    if sys.platform == "win32":
        launcher = _trusted_helper("git_windows_launcher.py").decode("utf-8")
        command = [_trusted_executable(sys.executable), "-I", "-S", "-c", launcher, *command]
        return command, _closed_env(values, fd), ()
    return command, _closed_env(values, fd), (fd,)
