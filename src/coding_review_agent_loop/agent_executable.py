"""Pin every agent subprocess to the orchestrator's own ``agent-loop`` (#1632).

Agents are told to run ``agent-loop run-tests -- <command>``.  Without a pin
their shell resolves the first ``agent-loop`` on its inherited ``PATH``, which
can be a different, stale install whose receipts the running orchestrator then
judges.  This module installs a private, content-addressed ``agent-loop``
launcher that re-executes the running interpreter in isolated mode (``-I``)
with the orchestrator's effective import path frozen at install time, so a
coder's ``PYTHONPATH``, its cwd and user-site installs cannot select other
code.  The launcher directory goes first on every agent's ``PATH`` and the
launcher path is exported as :data:`EXECUTABLE_ENV`.

The pin is fail-soft like the coder CI-dispatch guard: when the launcher
cannot be installed the agent keeps its inherited ``PATH`` and a warning is
logged.  Windows is out of scope; the helpers return nothing there.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

EXECUTABLE_ENV = "AGENT_LOOP_EXECUTABLE"
LAUNCHER_NAME = "agent-loop"
LAUNCHER_DIR_NAME = "agent-loop-exe"


def package_path() -> Path:
    """The resolved directory of the running ``coding_review_agent_loop`` package."""
    return Path(__file__).resolve().parent


def package_digest(path: str | os.PathLike[str] | None = None) -> str:
    """Short sha256 of a resolved package path (12 hex), for comparisons."""
    value = str(path if path is not None else package_path())
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:12]


def running_install_identity() -> dict[str, str]:
    """The running install's resolved package path and its short digest."""
    path = package_path()
    return {"package_path": str(path), "package_digest": package_digest(path)}


def _frozen_import_path() -> list[str]:
    """The orchestrator's effective import path, without cwd-relative entries."""
    first = str(package_path().parent)
    frozen = [first]
    for entry in sys.path:
        if entry and os.path.isabs(entry) and entry not in frozen:
            frozen.append(entry)
    return frozen


def launcher_text() -> str:
    """The POSIX ``sh`` launcher that re-executes this install in isolated mode."""
    code = (
        f"import sys; sys.path[:0] = {_frozen_import_path()!r}; "
        "sys.argv[0] = 'agent-loop'; "
        "from coding_review_agent_loop.cli import main; "
        "sys.exit(main())"
    )
    # The lexical interpreter keeps a virtualenv's ``bin/python`` symlink, so
    # anything run-tests spawns through ``sys.executable`` stays in the venv.
    return f"#!/bin/sh\nexec {shlex.quote(sys.executable)} -I -c {shlex.quote(code)} \"$@\"\n"


def install_agent_loop_launcher(root: Path | None = None) -> Path:
    """Write the launcher into a private, content-addressed directory; return its path.

    The digest covers the whole launcher text, so different interpreters or
    import paths never share a directory.  Raises ``OSError`` on failure.
    """
    from .scratch import make_private_dirs, scratch_root

    if os.name != "posix":
        raise OSError("the agent-loop launcher requires a POSIX shell")
    text = launcher_text()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    directory = make_private_dirs((root or scratch_root() / LAUNCHER_DIR_NAME) / digest)
    launcher = directory / LAUNCHER_NAME
    if not launcher.is_file() or launcher.read_text(encoding="utf-8") != text:
        staging = directory / f".{LAUNCHER_NAME}.{os.getpid()}.tmp"
        staging.write_text(text, encoding="utf-8")
        staging.chmod(0o700)
        os.replace(staging, launcher)
    return launcher


_LAUNCHER_CACHE: dict[str, Path] = {}
_WARNED: set[str] = set()


def agent_loop_launcher(root: Path | None = None) -> Path | None:
    """The installed launcher, or ``None`` when it cannot be installed (warned once)."""
    key = str(root) if root is not None else ""
    cached = _LAUNCHER_CACHE.get(key)
    if cached is not None and cached.is_file():
        return cached
    try:
        launcher = install_agent_loop_launcher(root)
    except OSError as exc:
        _warn_once(
            f"agent-loop executable pin not installed ({exc}); agents keep their inherited PATH"
        )
        return None
    _LAUNCHER_CACHE[key] = launcher
    return launcher


def _kept_entries(path_value: str, directory: str) -> list[str]:
    """Every ``PATH`` entry except those naming ``directory``.

    Empty POSIX entries mean the current directory, so leading, trailing and
    interior empty entries are preserved exactly.
    """
    try:
        excluded = os.path.realpath(directory)
    except OSError:
        excluded = directory
    return [
        entry for entry in path_value.split(os.pathsep)
        if not entry or (entry != directory and os.path.realpath(entry) != excluded)
    ]


def agent_executable_environment(
    ambient: Mapping[str, str] | None = None, *, root: Path | None = None
) -> dict[str, str]:
    """``PATH`` and ``AGENT_LOOP_EXECUTABLE`` overrides for one agent launch, or ``{}``.

    Only the launcher's own directory is prepended (never the whole venv
    ``bin``, which would shadow ``python``/``pytest``), and an earlier copy of
    it is stripped so repeated launches never stack it.
    """
    values = os.environ if ambient is None else ambient
    launcher = agent_loop_launcher(root)
    if launcher is None:
        return {}
    directory = str(launcher.parent)
    inherited = values.get("PATH")
    kept = [] if inherited is None else _kept_entries(inherited, directory)
    # Only entries naming the launcher dir are removed, so an inherited
    # PATH's other entries (including empty current-directory entries) keep
    # their meaning and order behind it.
    path = os.pathsep.join([directory, *kept])
    return {"PATH": path, EXECUTABLE_ENV: str(launcher)}


def pinned_agent_environment(
    overrides: Mapping[str, str] | None = None,
    *,
    ambient: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The complete environment for an agent launch outside ``Runner.run_with_log``.

    Merges ``overrides`` (for example the coder CI-dispatch guard) over the
    ambient environment, then applies the executable pin last, exactly as the
    runner's launch sites do.  Every direct agent spawn uses this helper.
    """
    launch = {**(os.environ if ambient is None else ambient), **(overrides or {})}
    launch.update(agent_executable_environment(launch))
    return launch


def _warn_once(message: str) -> None:
    if message in _WARNED:
        return
    _WARNED.add(message)
    sys.stderr.write(f"agent-loop: Warning: {message}\n")
    sys.stderr.flush()


def preflight_messages(
    ambient: Mapping[str, str] | None = None,
    *,
    root: Path | None = None,
    which: Callable[..., str | None] = shutil.which,
) -> list[str]:
    """Startup preflight lines about which ``agent-loop`` agents will resolve.

    Returns a warning when the agent-visible ``agent-loop`` is not this
    install's launcher (or the launcher is unavailable), and an informational
    line when the inherited ``PATH``'s first ``agent-loop`` is another file.
    Never raises; the caller only logs the lines.
    """
    values = dict(os.environ if ambient is None else ambient)
    lines: list[str] = []
    try:
        identity = running_install_identity()
        launcher = agent_loop_launcher(root)
        inherited = which("agent-loop", path=values.get("PATH"))
        if launcher is None:
            lines.append(
                "Warning: agents will resolve agent-loop to "
                f"{inherited or '<not found>'}, not this install (launcher unavailable -> "
                f"{identity['package_path']})"
            )
            return lines
        agent_env = agent_executable_environment(values, root=root)
        found = which("agent-loop", path=agent_env.get("PATH", values.get("PATH")))
        if found is None or os.path.realpath(found) != os.path.realpath(launcher):
            lines.append(
                f"Warning: agents will resolve agent-loop to {found or '<not found>'}, not this "
                f"install ({launcher} -> {identity['package_path']})"
            )
        if inherited is not None and os.path.realpath(inherited) != os.path.realpath(launcher):
            lines.append(
                f"agent-loop on the inherited PATH is {inherited}; agents are pinned to the "
                f"running install ({launcher} -> {identity['package_path']})"
            )
    except Exception as exc:  # noqa: BLE001 - the preflight must never block a run
        lines.append(f"Warning: agent-loop executable preflight failed ({type(exc).__name__})")
    return lines


def log_preflight(config: object) -> None:
    """Log the startup preflight lines; never blocks the run."""
    from .logging import log

    for line in preflight_messages():
        try:
            log(config, line)  # type: ignore[arg-type]
        except Exception:  # pragma: no cover - logging must never block a run
            sys.stderr.write(f"agent-loop: {line}\n")
