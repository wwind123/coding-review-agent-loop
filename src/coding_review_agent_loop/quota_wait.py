"""Opt-in waiting for a parsed usage-limit reset (#1601).

``--wait-for-quota-reset DURATION`` lets a validated turn sleep until a parsed
quota reset instead of stopping the run.  Waiting is bounded per logical run
(at most :data:`QUOTA_WAIT_MAX_WAITS` waits and at most DURATION in total),
holds no agent subprocess or settings lock, and is interruptible: a process
level waiter registry lets the main thread cancel waits running in reviewer
or debater worker threads when SIGINT/SIGTERM arrives.
"""

from __future__ import annotations

import datetime
import os
import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass

from . import workdir_claims
from .agent_failure import _format_reset_duration
from .errors import QuotaResetExceededError
from .logging import log

QUOTA_WAIT_MAX_WAITS = 2
QUOTA_WAIT_MARGIN_SECONDS = 60
QUOTA_WAIT_SLICE_SECONDS = 30


@dataclass(frozen=True)
class QuotaWaitReservation:
    sleep_seconds: int
    wait_number: int
    max_waits: int


class QuotaWaitBudget:
    """Thread-safe per-run allowance of quota waits."""

    def __init__(self, limit_seconds: int, max_waits: int = QUOTA_WAIT_MAX_WAITS) -> None:
        self.limit_seconds = limit_seconds
        self.max_waits = max_waits
        self._lock = threading.Lock()
        self.waits_used = 0
        self.seconds_waited = 0

    def try_reserve(self, reset_seconds: int) -> QuotaWaitReservation | None:
        """Reserve a wait of reset plus margin, or None when it does not fit."""
        sleep_seconds = max(0, int(reset_seconds)) + QUOTA_WAIT_MARGIN_SECONDS
        with self._lock:
            if self.waits_used >= self.max_waits:
                return None
            if self.seconds_waited + sleep_seconds > self.limit_seconds:
                return None
            self.waits_used += 1
            self.seconds_waited += sleep_seconds
            return QuotaWaitReservation(sleep_seconds, self.waits_used, self.max_waits)


_run_budgets: dict[str, QuotaWaitBudget] = {}
_run_budgets_lock = threading.Lock()


def quota_wait_budget_for_current_run(config) -> QuotaWaitBudget:
    """Budget owned by the active logical run, or a fresh local one with no owner.

    Mirrors ``quota_memory_for_current_run``: nested loops and context-propagated
    reviewer threads share it, and a run finalizer drops it.
    """
    limit = config.wait_for_quota_reset_seconds
    owner = workdir_claims.current_claim_owner()
    if owner is None:
        return QuotaWaitBudget(limit)
    with _run_budgets_lock:
        budget = _run_budgets.get(owner.run_id)
        if budget is None:
            budget = QuotaWaitBudget(limit)
            _run_budgets[owner.run_id] = budget
            run_id = owner.run_id

            def _drop() -> None:
                with _run_budgets_lock:
                    _run_budgets.pop(run_id, None)

            workdir_claims.register_run_finalizer("quota-wait-budget", _drop)
    return budget


# --- waiter registry -------------------------------------------------------

_waiters_lock = threading.Lock()
_active_waits: set[int] = set()
_next_wait_token = 0
_cancel_event = threading.Event()
_interrupted_message: str | None = None


def _event_sleep(seconds: float) -> None:
    _cancel_event.wait(seconds)


# Tests substitute this; it is called once per slice of at most
# QUOTA_WAIT_SLICE_SECONDS and must return early once cancellation is set.
_sleeper: Callable[[float], None] = _event_sleep


def active_wait_count() -> int:
    with _waiters_lock:
        return len(_active_waits)


def cancel_active_waits() -> None:
    """Wake every active (and any about-to-start) wait; it raises a quota stop."""
    global _interrupted_message
    with _waiters_lock:
        if _active_waits and _interrupted_message is None:
            _interrupted_message = _interrupted_text("agent")
    _cancel_event.set()


def wait_was_interrupted() -> bool:
    with _waiters_lock:
        return _interrupted_message is not None


def interrupted_message() -> str | None:
    with _waiters_lock:
        return _interrupted_message


def reset_wait_state() -> None:
    """Clear cancellation state; the CLI calls it around each command."""
    global _interrupted_message
    with _waiters_lock:
        _interrupted_message = None
    _cancel_event.clear()


def _interrupted_text(agent_name: str) -> str:
    return (
        f"{agent_name} quota reset wait interrupted; rerun to resume once the quota resets, "
        "or switch to a different API key / model."
    )


def _latch_interrupted(agent_name: str) -> str:
    global _interrupted_message
    message = _interrupted_text(agent_name)
    with _waiters_lock:
        # A specific agent name is more useful than the generic latch set by
        # cancel_active_waits() for the same interruption.
        _interrupted_message = message
    return message


def wait_for_reset(config, agent_name: str, reservation: QuotaWaitReservation) -> None:
    """Sleep for the reserved wait, or raise QuotaResetExceededError when interrupted."""
    global _next_wait_token
    sleep_seconds = reservation.sleep_seconds
    until = datetime.datetime.now().astimezone() + datetime.timedelta(seconds=sleep_seconds)
    with _waiters_lock:
        _next_wait_token += 1
        token = _next_wait_token
        _active_waits.add(token)
    try:
        log(
            config,
            f"{agent_name} quota exhausted; waiting until {until.strftime('%Y-%m-%d %H:%M %Z')} "
            f"({_format_reset_duration(sleep_seconds)}) for reset "
            f"(wait {reservation.wait_number} of {reservation.max_waits})",
        )
        remaining = float(sleep_seconds)
        try:
            while remaining > 0:
                if _cancel_event.is_set():
                    break
                chunk = min(float(QUOTA_WAIT_SLICE_SECONDS), remaining)
                _sleeper(chunk)
                remaining -= chunk
        except KeyboardInterrupt:
            raise QuotaResetExceededError(_latch_interrupted(agent_name)) from None
        if _cancel_event.is_set():
            raise QuotaResetExceededError(_latch_interrupted(agent_name))
        log(config, f"{agent_name} quota reset wait finished; retrying the same turn")
    finally:
        with _waiters_lock:
            _active_waits.discard(token)


# --- CLI SIGTERM integration ----------------------------------------------

def install_sigterm_handler():
    """Install the quota-wait SIGTERM handler; return the previous handler.

    Returns None (and installs nothing) off the main thread.  With no active
    wait, SIGTERM keeps the previous behaviour.
    """
    if threading.current_thread() is not threading.main_thread():
        return None
    previous = signal.getsignal(signal.SIGTERM)
    if previous is None:  # installed outside Python; nothing to chain to
        previous = signal.SIG_DFL

    def _handler(signum, frame):
        if active_wait_count():
            cancel_active_waits()
            raise KeyboardInterrupt
        if callable(previous):
            previous(signum, frame)
            return
        if previous == signal.SIG_IGN:
            return
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)

    signal.signal(signal.SIGTERM, _handler)
    return previous


def restore_sigterm_handler(previous) -> None:
    if previous is None or threading.current_thread() is not threading.main_thread():
        return
    signal.signal(signal.SIGTERM, previous)
