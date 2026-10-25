"""Turn-scoped ``gh`` guard against coder-initiated CI dispatch (#1366).

While a managed-CI pull request is waiting for review, the ordinary
``pull_request`` CI run is skipped on purpose and the orchestrator dispatches
correlated exact-head qualification only after every required reviewer has
approved the current head.  A coder or format-repair turn that dispatches the
ordinary workflow itself to get a green check defeats that saving.

This module is installed as ``gh`` on the coder's ``PATH`` for exactly those
turns.  It refuses the CI-triggering forms (``gh workflow run``, ``gh run
rerun`` and POSTs through ``gh api`` to dispatch or rerun endpoints) and
``exec``s every other command, unchanged, through the downstream ``gh`` named
by :data:`TARGET_ENV`.  It is defense in depth, not a sandbox: an absolute
``gh`` path, ``curl`` or the web UI bypasses it, and the orchestrator's
post-turn audit makes such a bypass visible.

The guard uses its own variables only.  It never reads or rewrites
``AGENT_LOOP_REAL_GH``; when the downstream target is the ``agent-loop-gh``
REST shim, the installer preserves (or derives against the pre-guard
``PATH``) the shim's own real-gh resolution, and the guard removes its own
directory from ``PATH`` before ``exec`` so nothing downstream can resolve back
into it.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

GUARD_ENV = "AGENT_LOOP_CODER_CI_DISPATCH_GUARD"
GUARD_ACTIVE = "managed"
TARGET_ENV = "AGENT_LOOP_CODER_GUARD_TARGET_GH"
GUARD_DIR_ENV = "AGENT_LOOP_CODER_GUARD_DIR"
REAL_GH_ENV = "AGENT_LOOP_REAL_GH"
GUARD_ROLES = frozenset({"coder", "repair"})
SHIM_NAME = "agent-loop-gh"
EXIT_REFUSED = 3
EXIT_NO_TARGET = 127

REFUSAL_MESSAGE = (
    "agent-loop: refused `{trigger}`: this pull request uses managed CI. A skipped or neutral "
    "pull_request CI check on the managed draft is the expected approval-gated state, not a "
    "failure to fix. Your managed `agent-loop run-tests` receipt is the evidence before "
    "approval; after every required reviewer approves the current head, the orchestrator "
    "dispatches the correlated exact-head qualification itself. If you believe early GitHub CI "
    "is needed, say so in your remaining items so a human operator can decide; an operator can "
    "still run the workflow from their own shell as a non-authoritative diagnostic.\n"
)

# gh parses flags with spf13/pflag: long flags as ``--name value`` or
# ``--name=value``; shorthands may be grouped (``-iX POST``), and a value
# shorthand takes the rest of its token (``-XPOST``, ``-X=POST``) or the next
# token.  Every form below is decoded by one tokenizer so no spelling of a
# method or body flag slips past the classifier (#1366).
# Persistent value flags outside ``gh api`` (shorthand chars, long names).
_SHORT_VALUE = frozenset("R")
_LONG_VALUE = frozenset({"repo"})
# ``gh api`` value flags.
_API_SHORT_VALUE = frozenset("XfFHqtp")
_API_LONG_VALUE = frozenset({
    "method", "raw-field", "field", "header", "input", "jq", "template",
    "hostname", "cache", "preview",
})
_API_SHORT_BODY = frozenset("fF")
_API_LONG_BODY = frozenset({"raw-field", "field", "input"})
_TRIGGER_ENDPOINT = re.compile(
    r"(?:^|/)repos/[^/]+/[^/]+/actions/"
    r"(?:workflows/[^/]+/dispatches|runs/[^/]+/(?:rerun|rerun-failed-jobs)|jobs/[^/]+/rerun)"
    r"/?(?:[?#].*)?$"
)


def _parse_flags(
    argv: Sequence[str], short_value: frozenset[str], long_value: frozenset[str]
) -> tuple[list[str], list[tuple[str, str | None]], list[int]]:
    """Split ``argv`` like pflag into positionals and ``(flag, value)`` pairs.

    Short flags are reported as their single character, long flags as their
    name without dashes; boolean flags carry ``None``.  The third element is
    each positional's index in ``argv``.
    """
    positionals: list[str] = []
    positional_indexes: list[int] = []
    flags: list[tuple[str, str | None]] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        index += 1
        if token == "--":
            positional_indexes.extend(range(index, len(argv)))
            positionals.extend(argv[index:])
            break
        if token.startswith("--"):
            name, eq, value = token[2:].partition("=")
            if eq:
                flags.append((name, value))
            elif name in long_value:
                flags.append((name, argv[index] if index < len(argv) else ""))
                index += 1
            else:
                flags.append((name, None))
            continue
        if token.startswith("-") and len(token) > 1:
            group = token[1:]
            while group:
                char, group = group[0], group[1:]
                if char not in short_value:
                    flags.append((char, None))
                    continue
                if group.startswith("="):
                    group = group[1:]
                if group:
                    flags.append((char, group))
                else:
                    flags.append((char, argv[index] if index < len(argv) else ""))
                    index += 1
                break
            continue
        positional_indexes.append(index - 1)
        positionals.append(token)
    return positionals, flags, positional_indexes


def _classify_api(argv: Sequence[str]) -> str | None:
    positionals, flags, _ = _parse_flags(argv, _API_SHORT_VALUE, _API_LONG_VALUE)
    method: str | None = None
    has_body = False
    for name, value in flags:
        if name in {"X", "method"}:
            method = value or ""
        elif name in _API_SHORT_BODY or name in _API_LONG_BODY:
            has_body = True
    endpoint = positionals[0] if positionals else None
    if endpoint is None or not _TRIGGER_ENDPOINT.search(endpoint):
        return None
    effective = (method or ("POST" if has_body else "GET")).upper()
    if effective != "POST":
        return None
    return f"gh api -X POST {endpoint}"


def classify_ci_trigger(argv: Sequence[str]) -> str | None:
    """Name the CI-triggering ``gh`` form in ``argv`` (program name excluded), or None."""
    args = list(argv)
    positionals, _, indexes = _parse_flags(args, _SHORT_VALUE, _LONG_VALUE)
    if not positionals:
        return None
    if positionals[:2] == ["workflow", "run"]:
        return "gh workflow run"
    if positionals[:2] == ["run", "rerun"]:
        return "gh run rerun"
    if positionals[0] == "api":
        # ``api`` is the first positional, so everything after it is the
        # api command's own argv (persistent flags before it carry no body).
        return _classify_api(args[indexes[0] + 1:])
    return None


def _strip_dir(path_value: str, directory: str | None) -> str:
    if not directory:
        return path_value
    try:
        excluded = os.path.realpath(directory)
    except OSError:
        excluded = directory
    kept = [
        entry for entry in path_value.split(os.pathsep)
        if entry and os.path.realpath(entry) != excluded
    ]
    return os.pathsep.join(kept)


def guard_main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    """Refuse a CI trigger while the guard is active; otherwise exec the target."""
    args = list(sys.argv[1:] if argv is None else argv)
    values = dict(os.environ if env is None else env)
    if values.get(GUARD_ENV) == GUARD_ACTIVE:
        trigger = classify_ci_trigger(args)
        if trigger is not None:
            sys.stderr.write(REFUSAL_MESSAGE.format(trigger=trigger))
            return EXIT_REFUSED
    target = values.get(TARGET_ENV) or ""
    guard_dir = values.get(GUARD_DIR_ENV)
    if not target or not os.path.isfile(target) or not os.access(target, os.X_OK):
        sys.stderr.write(
            f"agent-loop: coder gh guard has no usable downstream gh ({TARGET_ENV}={target!r}).\n"
        )
        return EXIT_NO_TARGET
    if guard_dir and os.path.realpath(os.path.dirname(os.path.realpath(target))) == os.path.realpath(guard_dir):
        sys.stderr.write("agent-loop: coder gh guard target points back at the guard itself.\n")
        return EXIT_NO_TARGET
    values["PATH"] = _strip_dir(values.get("PATH", ""), guard_dir)
    os.execve(target, [target, *args], values)
    return EXIT_NO_TARGET  # pragma: no cover - execve does not return


def _launcher_text() -> str:
    package_parent = str(Path(__file__).resolve().parent.parent)
    code = (
        f"import sys; sys.path.insert(0, {package_parent!r}); "
        "from coding_review_agent_loop.coder_ci_guard import guard_main; "
        "sys.exit(guard_main())"
    )
    return f"#!/bin/sh\nexec {shlex.quote(sys.executable)} -I -c {shlex.quote(code)} \"$@\"\n"


def install_guard_launcher(root: Path | None = None) -> Path:
    """Write the ``gh`` launcher into a private, content-addressed directory.

    The launcher is identical for every invocation of this interpreter and
    package; the per-invocation target travels in the environment.  It lives
    under the private scratch root, outside every checkout.
    """
    from .scratch import make_private_dirs, scratch_root

    text = _launcher_text()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    directory = make_private_dirs((root or scratch_root() / "coder-ci-guard") / digest)
    launcher = directory / "gh"
    if not launcher.is_file() or launcher.read_text(encoding="utf-8") != text:
        # Same per-writer staging as the agent-loop launcher (#1632), so
        # concurrent coder and repair launches never share a temporary file.
        from .agent_executable import _atomic_write_executable

        _atomic_write_executable(directory, launcher, text)
    return directory


def coder_ci_guard_environment(
    config: object,
    provider: str,
    role: str | None,
    *,
    ambient: Mapping[str, str] | None = None,
    root: Path | None = None,
) -> dict[str, str]:
    """Guard environment for one coder/repair launch, or ``{}`` when unguarded.

    Every subprocess launch with role ``coder`` or ``repair`` obtains its
    guard environment from this one helper.  It is keyed only on the
    invocation-scoped ``coder_ci_dispatch_guard`` flag, so reviewer, planner
    and unmanaged launches never see it.  When the launcher cannot be
    installed the turn continues with prompt-only protection.
    """
    del provider  # every backend shares the same guard
    if not getattr(config, "coder_ci_dispatch_guard", False) or role not in GUARD_ROLES:
        return {}
    values = os.environ if ambient is None else ambient
    pre_guard_path = _strip_dir(values.get("PATH", ""), values.get(GUARD_DIR_ENV))
    gh_cmd = str(getattr(config, "gh_cmd", "gh") or "gh")
    target = shutil.which(gh_cmd, path=pre_guard_path)
    if target is None:
        _warn(config, f"coder gh guard not installed: {gh_cmd!r} was not found on PATH; prompt-only protection")
        return {}
    target = os.path.abspath(target)
    try:
        directory = install_guard_launcher(root)
    except OSError as exc:
        _warn(config, f"coder gh guard not installed ({exc}); prompt-only protection")
        return {}
    environment = {
        GUARD_ENV: GUARD_ACTIVE,
        TARGET_ENV: target,
        GUARD_DIR_ENV: str(directory),
        "PATH": os.pathsep.join(entry for entry in (str(directory), pre_guard_path) if entry),
    }
    if Path(target).resolve().name == SHIM_NAME and not values.get(REAL_GH_ENV):
        from .gh_rest_shim import ShimError, find_real_gh

        try:
            environment[REAL_GH_ENV] = find_real_gh(
                {"PATH": pre_guard_path}, own=Path(target).resolve()
            )
        except ShimError:
            # The shim reports its own missing-gh error on first use.
            pass
    return environment


def _warn(config: object, message: str) -> None:
    from .logging import log

    try:
        log(config, message)  # type: ignore[arg-type]
    except Exception:  # pragma: no cover - logging must never block a turn
        sys.stderr.write(f"agent-loop: {message}\n")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(guard_main())
