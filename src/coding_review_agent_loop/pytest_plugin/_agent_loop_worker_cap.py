"""agent-loop worker-budget cap for pytest-xdist (injected by ``agent-loop run-tests``).

This module is stdlib-only apart from pytest itself and never imports
``coding_review_agent_loop``.  It is a prompt-slip safety net, not a
sandbox: a repository can unregister or outrun it, and cgroup memory limits
remain the hard boundary.

The wrapper passes the budget, mode and a report path in the private
``AGENT_LOOP_WORKER_CAP_SPEC`` variable.  The spec is read once at import and
removed from ``os.environ`` so child processes (xdist workers, pytester, tests
that run pytest) are never enforced.  Each top-level ``pytest.main`` claims
the armed spec for its own Config and reports under its own session id;
Configs nested inside an active claim and child processes that inherit the
plugin environment only write a best-effort ``nested`` marker.

The wrapper's ``PYTEST_PLUGINS`` entry is withdrawn from ``os.environ`` for
the whole of every claimed session, so a test that launches a
nested pytest with a replaced ``PYTHONPATH`` never inherits an entry it cannot
import (issue #1008).  It is restored when a claim ends so a later sequential
``pytest.main`` in the same launcher process still loads the plugin.
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytest

_SPEC_ENV = "AGENT_LOOP_WORKER_CAP_SPEC"
_NESTED_ENV = "AGENT_LOOP_WORKER_CAP_NESTED"
_MODES = ("clamp", "refuse")
_PLUGINS_ENV = "PYTEST_PLUGINS"
_SELF = "_agent_loop_worker_cap"


def _load_spec():
    raw = os.environ.pop(_SPEC_ENV, None)
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(value, dict):
        return None
    budget = value.get("budget")
    mode = value.get("mode")
    report = value.get("report")
    if (
        isinstance(budget, bool)
        or not isinstance(budget, int)
        or budget < 1
        or mode not in _MODES
        or not isinstance(report, str)
        or not report
    ):
        return None
    return {"budget": budget, "mode": mode, "report": report}


def _plugin_entries(raw):
    return [entry.strip() for entry in (raw or "").split(",") if entry.strip()]


def _withdraw_env_entry() -> bool:
    """Drop this plugin from ``PYTEST_PLUGINS`` so child processes never inherit it."""
    entries = _plugin_entries(os.environ.get(_PLUGINS_ENV))
    if _SELF not in entries:
        return False
    kept = [entry for entry in entries if entry != _SELF]
    if kept:
        os.environ[_PLUGINS_ENV] = ",".join(kept)
    else:
        os.environ.pop(_PLUGINS_ENV, None)
    return True


def _restore_env_entry() -> None:
    entries = _plugin_entries(os.environ.get(_PLUGINS_ENV))
    if _SELF not in entries:
        os.environ[_PLUGINS_ENV] = ",".join([*entries, _SELF])


_ARMED = _load_spec()
if _ARMED is not None:
    # Only the report path is published; never the budget, mode or spec.
    os.environ[_NESTED_ENV] = _ARMED["report"]
    _NESTED_REPORT = None
else:
    _NESTED_REPORT = os.environ.get(_NESTED_ENV) or None

_CLAIM = None  # the _Session currently holding the armed spec


def _pluggy_supports_wrappers() -> bool:
    try:
        import pluggy

        parts = []
        for piece in str(getattr(pluggy, "__version__", "0")).split(".")[:2]:
            digits = "".join(ch for ch in piece if ch.isdigit())
            parts.append(int(digits or 0))
        while len(parts) < 2:
            parts.append(0)
        return tuple(parts) >= (1, 2)
    except Exception:
        return False


def _emit(path, record) -> bool:
    """Append one JSON line; never raises and never changes pytest's status."""
    try:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
        return True
    except (OSError, ValueError, TypeError):
        try:
            sys.stderr.write("agent-loop worker cap: report not written\n")
            sys.stderr.flush()
        except Exception:
            pass
        return False


def spec_is_popen(spec) -> bool:
    """Whether an xdist gateway spec (XSpec or string) is a local popen."""
    popen = getattr(spec, "popen", None)
    if popen is not None and not isinstance(spec, str):
        return bool(popen)
    text = str(getattr(spec, "_spec", spec))
    for part in text.split("//"):
        key = part.split("=", 1)[0].strip()
        if key == "popen":
            return True
    return False


def count_specs(specs):
    """Return (total, remote) for an expanded spec list; every type counts."""
    total = len(specs)
    remote = sum(1 for spec in specs if not spec_is_popen(spec))
    return total, remote


class _Session:
    def __init__(self, spec, config):
        self.env_entry = False  # whether this claim withdrew the PYTEST_PLUGINS entry
        self.budget = spec["budget"]
        self.mode = spec["mode"]
        self.report = spec["report"]
        self.config = config
        self.session = f"{os.getpid()}-{time.monotonic_ns()}"
        self.seq = 0
        self.requested = None
        self.auto_raw = None
        self.planned = None
        self.planned_remote = None
        self.lowered = False
        self.counting = False
        self.gateways = 0
        self.remote_gateways = 0
        self.confirmed = False
        self.executed = False
        self.notices = []

    def emit(self, record) -> bool:
        self.seq += 1
        payload = {"session": self.session, "seq": self.seq, "pid": os.getpid(), "budget": self.budget, "mode": self.mode}
        payload.update(record)
        return _emit(self.report, payload)

    def refuse(self, stage, requested, *, total=None, remote=None):
        record = {
            "kind": "refused",
            "action": "refused",
            "stage": stage,
            "requested": requested,
            "auto_raw": self.auto_raw,
            "total": total,
            "remote": remote,
            "source": "pytest-resolved",
        }
        self.emit(record)
        raise pytest.UsageError(
            f"agent-loop worker budget refused: {stage} requested {requested} worker(s) "
            f"above the budget of {self.budget} (refuse mode)"
        )


def _state(config):
    claim = _CLAIM
    if claim is not None and claim.config is config:
        return claim
    return None


def _write_line(config, text):
    reporter = None
    try:
        reporter = config.pluginmanager.get_plugin("terminalreporter")
    except Exception:
        reporter = None
    if reporter is not None:
        try:
            reporter.write_line(text)
            return
        except Exception:
            pass
    try:
        sys.stderr.write(text + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def _is_worker(config) -> bool:
    return hasattr(config, "workerinput")


if _pluggy_supports_wrappers():

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_cmdline_main(config):
        global _CLAIM
        if _is_worker(config):
            return (yield)
        state = None
        if _ARMED is not None:
            if _CLAIM is None:
                state = _Session(_ARMED, config)
                _CLAIM = state
                state.env_entry = _withdraw_env_entry()
            else:
                _emit(
                    _ARMED["report"],
                    {"kind": "nested", "where": "in-process", "parent_session": _CLAIM.session, "pid": os.getpid()},
                )
        elif _NESTED_REPORT:
            _emit(_NESTED_REPORT, {"kind": "nested", "where": "child", "pid": os.getpid()})
        if state is None:
            return (yield)
        try:
            _option_gate(state, config)
            return (yield)
        finally:
            if _CLAIM is state:
                _CLAIM = None
                if state.env_entry:
                    _restore_env_entry()

    def _option_gate(state, config):
        option = config.option
        numprocesses = getattr(option, "numprocesses", None)
        maxprocesses = getattr(option, "maxprocesses", None)
        state.requested = numprocesses
        if isinstance(numprocesses, int) and not isinstance(numprocesses, bool):
            if state.mode == "clamp":
                if numprocesses > state.budget:
                    lowered = state.budget if state.budget > 1 else 0
                    option.numprocesses = lowered
                    state.lowered = True
                    state.notices.append(
                        f"agent-loop worker budget: lowered -n {numprocesses} to {lowered} "
                        f"(budget {state.budget})"
                    )
            else:
                effective = min(numprocesses, maxprocesses) if maxprocesses else numprocesses
                if effective > state.budget:
                    state.refuse("cmdline", effective)
        elif numprocesses in ("auto", "logical"):
            if state.mode == "clamp":
                if state.budget == 1:
                    option.numprocesses = 0
                    state.lowered = True
                    state.notices.append(
                        f"agent-loop worker budget: lowered -n {numprocesses} to serial (budget 1)"
                    )
                else:
                    cap = min(maxprocesses, state.budget) if maxprocesses else state.budget
                    if cap != maxprocesses:
                        # Capping the auto resolution is itself a clamp; the
                        # confirmation still reports the gateways created.
                        option.maxprocesses = cap
                        state.lowered = True

    @pytest.hookimpl(wrapper=True, optionalhook=True)
    def pytest_xdist_auto_num_workers(config):
        result = yield
        state = _state(config)
        if state is None or isinstance(result, bool) or not isinstance(result, int):
            return result
        state.auto_raw = result
        if state.mode == "clamp":
            if result > state.budget:
                state.lowered = True
                state.notices.append(
                    f"agent-loop worker budget: lowered auto-resolved {result} worker(s) to {state.budget}"
                )
                return state.budget
            return result
        maxprocesses = getattr(config.option, "maxprocesses", None)
        effective = min(result, maxprocesses) if maxprocesses else result
        if effective > state.budget:
            state.refuse("auto", effective)
        return result

    @pytest.hookimpl(wrapper=True, tryfirst=True, optionalhook=True)
    def pytest_xdist_setupnodes(config, specs):
        result = yield
        state = _state(config)
        if state is None:
            return result
        total, remote = count_specs(specs)
        if total > state.budget:
            if state.mode == "clamp":
                del specs[state.budget:]
                # xdist's schedulers count nodes from ``--tx``; keep it in
                # step with the trimmed spec list or the run waits forever.
                try:
                    config.option.tx = [str(getattr(spec, "_spec", spec)) for spec in specs]
                except Exception:
                    pass
                state.lowered = True
                state.notices.append(
                    f"agent-loop worker budget: trimmed {total} gateway spec(s) to {state.budget}"
                )
            else:
                state.refuse("setupnodes", total, total=total, remote=remote)
        planned, planned_remote = count_specs(specs)
        state.planned = planned
        state.planned_remote = planned_remote
        state.emit(
            {
                "kind": "decision",
                "planned": planned,
                "remote": planned_remote,
                "action": "clamped" if state.lowered else "unchanged",
            }
        )
        state.counting = True
        return result

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_newgateway(gateway):
        state = _CLAIM
        if state is None or not state.counting or state.confirmed:
            return
        state.gateways += 1
        if not spec_is_popen(getattr(gateway, "spec", "")):
            state.remote_gateways += 1

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_runtestloop(session):
        state = _state(session.config)
        if state is not None and not state.confirmed:
            state.counting = False
            state.confirmed = True
            effective = state.gateways
            exceeded = effective > state.budget or (
                state.planned is not None and effective > state.planned
            )
            action = "exceeded" if exceeded else ("clamped" if state.lowered else "unchanged")
            requested = state.requested
            if requested is not None and not isinstance(requested, (int, str)):
                requested = str(requested)
            state.emit(
                {
                    "kind": "confirmed",
                    "requested": requested,
                    "auto_raw": state.auto_raw,
                    "planned": state.planned,
                    "effective": effective,
                    "remote": state.remote_gateways,
                    "action": action,
                    "source": "pytest-resolved",
                }
            )
            for notice in state.notices:
                _write_line(session.config, notice)
            if exceeded:
                _write_line(
                    session.config,
                    f"agent-loop worker budget WARNING: {effective} worker gateway(s) were created above "
                    f"the budget of {state.budget}; the worker budget was not enforced.",
                )
        return (yield)

    def pytest_runtest_logstart(nodeid, location):
        state = _CLAIM
        if state is None or state.executed or not state.confirmed:
            return
        state.executed = True
        state.emit({"kind": "executed", "first_nodeid": nodeid})

else:  # pragma: no cover - exercised only with pluggy < 1.2
    try:
        sys.stderr.write(
            "agent-loop worker cap: pluggy >= 1.2 (pytest >= 8) is required; "
            "worker-budget enforcement is unavailable\n"
        )
    except Exception:
        pass
