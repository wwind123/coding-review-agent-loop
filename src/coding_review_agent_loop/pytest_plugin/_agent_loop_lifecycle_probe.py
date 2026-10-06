"""agent-loop pytest lifecycle probe (injected by ``agent-loop run-tests``).

This module is stdlib-only apart from pytest itself and never imports
``coding_review_agent_loop``.  It is a cooperative observation channel, not a
sandbox: it records, in a parent-created report file, whether pytest finished
option parsing and whether any file that could hold test code was opened,
compiled, executed or imported before that point.  The parent uses the report
only as positive provenance that a run exited before any test could run.

The private ``AGENT_LOOP_LIFECYCLE_PROBE_SPEC`` variable (report path and
nonce) is read once at import and removed from ``os.environ`` so child
processes and nested pytest runs are never armed.  Every probe code path
swallows its own exceptions: a probe failure never changes the target's exit
status and only ever withdraws provenance (a missing or non-clean ``final``
record makes the parent fail closed).
"""

from __future__ import annotations

import atexit
import fnmatch
import json
import os
import sys
import threading

import pytest

_SPEC_ENV = "AGENT_LOOP_LIFECYCLE_PROBE_SPEC"
_SELF = "_agent_loop_lifecycle_probe"
_DEFAULT_PATTERNS = ("test_*.py", "*_test.py")
_MAX_RETAINED = 4096
_MAX_PATTERNS = 16


class _State:
    armed = False
    fd = -1
    nonce = ""
    pid = 0
    invalid = False
    write_failures = 0
    vetoed = False
    parsed = False
    phase = 1
    patterns = _DEFAULT_PATTERNS
    protected = frozenset()
    retained = set()
    touched_written = False


_S = _State()
_TLS = threading.local()


def _emit(kind: str, *, force: bool = False, **fields) -> bool:
    """Write one record with a single ``os.write``; never raise."""
    if _S.fd < 0 or (_S.invalid and not force) or getattr(_TLS, "busy", False):
        return False
    _TLS.busy = True
    try:
        record = {"kind": kind, "nonce": _S.nonce, "pid": _S.pid}
        record.update(fields)
        data = (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8")
        written = os.write(_S.fd, data)
        if written != len(data):
            raise OSError("short write")
        return True
    except Exception:
        _S.invalid = True
        _S.write_failures += 1
        return False
    finally:
        _TLS.busy = False


def _source_of(path: str) -> str:
    """Map a ``.pyc`` path to the source file it was compiled from."""
    if not path.endswith(".pyc"):
        return path
    directory, name = os.path.split(path)
    if os.path.basename(directory) == "__pycache__":
        return os.path.join(os.path.dirname(directory), name.split(".", 1)[0] + ".py")
    return path[:-1]


def _pattern_matches(pattern: str, source: str) -> bool:
    """Match like pytest's ``fnmatch_ex``: pathname patterns see the whole path."""
    seps = [os.sep] + ([os.altsep] if os.altsep else [])
    if not any(sep in pattern for sep in seps):
        return fnmatch.fnmatch(os.path.basename(source), pattern)
    if not os.path.isabs(pattern):
        pattern = "*" + os.sep + pattern
    candidates = {os.path.abspath(source)}
    try:
        candidates.add(os.path.realpath(source))
    except Exception:
        pass
    return any(fnmatch.fnmatch(candidate, pattern) for candidate in candidates)


def _matches_patterns(path: str) -> bool:
    source = _source_of(path)
    try:
        return any(_pattern_matches(pattern, source) for pattern in _S.patterns)
    except Exception:
        return True


def _is_protected(path: str) -> bool:
    if not _S.protected:
        return False
    for candidate in (path, _source_of(path)):
        try:
            if os.path.realpath(candidate) in _S.protected:
                return True
        except Exception:
            return True
    return False


def _veto() -> None:
    _S.vetoed = True
    if not _S.touched_written:
        _S.touched_written = True
        _emit("test-code-touched")


def _see(path) -> None:
    if isinstance(path, bytes):
        path = os.fsdecode(path)
    if not isinstance(path, str) or not path:
        return
    if _S.phase == 1:
        # Every path is retained, whatever its extension: an explicit target
        # (for example ``cases.spec``) can only be recognised once the resolved
        # arguments are known.  Overflow is a veto, never a silent drop.
        if path not in _S.retained:
            if len(_S.retained) >= _MAX_RETAINED:
                _veto()
                return
            _S.retained.add(path)
        if _matches_patterns(path):
            _veto()
    elif _matches_patterns(path) or _is_protected(path):
        _veto()


def _audit(event, args) -> None:
    try:
        if _S.vetoed or _S.parsed or _S.invalid or getattr(_TLS, "busy", False):
            return
        if event == "open" or event == "import":
            _see(args[0] if event == "open" else (args[1] if len(args) > 1 else None))
        elif event == "compile":
            _see(args[1] if len(args) > 1 else None)
        elif event == "exec":
            _see(getattr(args[0], "co_filename", None))
    except Exception:
        pass


def _final() -> None:
    try:
        if os.getpid() != _S.pid:
            return
        _emit(
            "final", force=True, invalid=_S.invalid, write_failures=_S.write_failures,
            vetoed=_S.vetoed, parsed=_S.parsed,
        )
    except Exception:
        pass


def _arm() -> None:
    raw = os.environ.pop(_SPEC_ENV, None)
    if not raw:
        return
    try:
        spec = json.loads(raw)
        report = spec["report"]
        nonce = spec["nonce"]
        if not isinstance(report, str) or not report or not isinstance(nonce, str) or not nonce:
            return
        fd = os.open(report, os.O_WRONLY | os.O_APPEND)
    except Exception:
        return
    _S.fd = fd
    _S.nonce = nonce
    _S.pid = os.getpid()
    _S.armed = True
    _emit("loaded")
    try:
        atexit.register(_final)
        sys.addaudithook(_audit)
    except Exception:
        _S.invalid = True


_arm()


def _flag_tokens(args):
    """Return (early_plugin, blocked_later) for the resolved argument list."""
    index = None
    for position, token in enumerate(args):
        if token == "-p" and position + 1 < len(args) and args[position + 1] == _SELF:
            index = position
            break
        if token in (f"-p{_SELF}", f"-p={_SELF}"):
            index = position
            break
    early = False
    blocked = False
    position = 0
    while position < len(args):
        token = args[position]
        value = None
        if token == "-p" and position + 1 < len(args):
            value = args[position + 1]
        elif token.startswith("-p") and len(token) > 2:
            value = token[2:].lstrip("=")
        if value is not None:
            if value == f"no:{_SELF}":
                blocked = True
            elif index is None or position < index:
                if value != _SELF and not value.startswith("no:"):
                    early = True
        position += 1
    return early or index is None, blocked


def _candidate_paths(args, early_config):
    """Over-approximate every path the resolved arguments could select."""
    tokens = []
    for token in args:
        token = str(token)
        tokens.append(token)
        if "=" in token:
            tokens.append(token.split("=", 1)[1])
        stripped = token.lstrip("-")
        if stripped != token:
            tokens.append(stripped)
        if len(token) > 2:
            tokens.append(token[2:])
    invocation = str(early_config.invocation_params.dir)
    bases = [invocation]
    candidates = []
    for token in tokens:
        candidates.append((token, bases))
    try:
        rootdir = str(early_config.rootpath)
        for value in early_config.getini("testpaths") or ():
            candidates.append((str(value), [rootdir, invocation]))
    except Exception:
        pass
    return candidates


def _compute_protected(args, early_config):
    protected = set()
    unprovable = False
    if any(str(token).startswith("@") for token in args) or any(
        str(token) == "--pyargs" or str(token).startswith("--pyargs=") for token in args
    ):
        unprovable = True
    for token, bases in _candidate_paths(args, early_config):
        target = token.split("::", 1)[0]
        if not target:
            continue
        for base in bases:
            try:
                resolved = os.path.realpath(os.path.join(base, target))
                if os.path.isfile(resolved):
                    protected.add(resolved)
            except Exception:
                unprovable = True
    return frozenset(protected), unprovable


@pytest.hookimpl(tryfirst=True)
def pytest_load_initial_conftests(early_config, parser, args):
    if not _S.armed or _S.invalid:
        return
    try:
        args = [str(token) for token in args]
        try:
            patterns = tuple(str(item) for item in early_config.getini("python_files") or ())
        except Exception:
            patterns = ()
        if not patterns:
            patterns = _DEFAULT_PATTERNS
        early, blocked = _flag_tokens(args)
        protected, unprovable = _compute_protected(args, early_config)
        _S.patterns = patterns
        _S.protected = protected
        # Retrospective re-check of everything retained before the patterns
        # and explicit targets were known.
        if not _S.vetoed:
            for path in list(_S.retained):
                if _matches_patterns(path) or _is_protected(path):
                    _veto()
                    break
        doctest = any("doctest" in token for token in args)
        _emit(
            "args-seen",
            patterns=list(patterns)[:_MAX_PATTERNS],
            protected=len(protected),
            doctest_requested=doctest,
            early_plugin=early,
            probe_blocked_later=blocked,
            explicit_targets_unprovable=unprovable,
        )
        _S.phase = 2
    except Exception:
        _S.invalid = True


def _mark_parsed() -> None:
    if _S.armed and not _S.parsed:
        _S.parsed = True
        _emit("parsed")


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config):
    try:
        _mark_parsed()
    except Exception:
        pass


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    try:
        _mark_parsed()
    except Exception:
        pass
