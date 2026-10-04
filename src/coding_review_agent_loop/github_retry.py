"""Bounded retry policy for transient GitHub (``gh``) failures.

Dependency-light on purpose (like ``transient.py``).  Only read-only ``gh``
invocations are routed through :func:`run_gh_read`; a write must never be
replayed without operation-specific reconciliation, so no write helper lives
here.  Agent-subprocess retries stay in ``transient.py``.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from typing import Literal, Sequence

from .errors import AgentLoopError
from .runner import CommandResult, Runner

Classification = Literal["transient", "permanent"]

_TRANSIENT_RE = re.compile(
    r"HTTP\s*50[0234]\b|status code:?\s*50[0234]\b|\b50[0234]\s+(?:Bad Gateway|Service Unavailable|Gateway Timeout|Internal Server Error)"
    r"|Bad Gateway|Service Unavailable|Gateway Timeout"
    r"|couldn'?t respond to your request in time"
    r"|connection reset|connection refused|i/o timeout|TLS handshake timeout"
    r"|unexpected EOF|\bEOF\b",
    re.I,
)
_PERMANENT_RE = re.compile(
    r"HTTP\s*4\d\d\b|status code:?\s*4\d\d\b|\b4\d\d\s+(?:Not Found|Unauthorized|Forbidden|Unprocessable)"
    r"|Not Found|validation|Resource not accessible|gh auth login|authentication|bad credentials"
    r"|unauthorized|forbidden|billing|rate limit|secondary rate|abuse detection|unprocessable",
    re.I,
)

_ATTEMPT_STDERR_LIMIT = 300


def classify_gh_failure(result: CommandResult) -> Classification:
    """Return ``transient`` only for genuine 5xx/transport failures.

    Any 4xx/auth/billing/validation/rate-limit marker wins over a transient
    phrase appearing in the same text.
    """
    if result.returncode == 0:
        return "permanent"
    text = f"{result.stderr or ''}\n{result.stdout or ''}"
    if _PERMANENT_RE.search(text):
        return "permanent"
    if _TRANSIENT_RE.search(text):
        return "transient"
    return "permanent"


@dataclass(frozen=True)
class GitHubAttempt:
    number: int
    returncode: int | None
    classification: Classification
    stderr: str
    started_monotonic: float

    def describe(self) -> str:
        return (
            f"attempt {self.number}: exit {self.returncode} ({self.classification}): "
            f"{self.stderr or '<no stderr>'}"
        )


@dataclass(frozen=True)
class GitHubRetryPolicy:
    attempts: int = 3
    backoff_seconds: tuple[float, ...] = (2.0, 5.0, 15.0)
    jitter_fraction: float = 0.25

    def delay_before_retry(self, failed_attempts: int) -> float:
        index = min(max(failed_attempts - 1, 0), len(self.backoff_seconds) - 1)
        base = self.backoff_seconds[index]
        jitter = base * self.jitter_fraction
        return max(0.0, base + random.uniform(-jitter, jitter))


DEFAULT_POLICY = GitHubRetryPolicy()


@dataclass(frozen=True)
class RetriedCommandResult(CommandResult):
    attempts: tuple[GitHubAttempt, ...] = ()
    exhausted: bool = False


class GitHubTransientExhaustedError(AgentLoopError):
    """Every attempt of a transient-failing gh command failed."""

    def __init__(self, message: str, attempts: tuple[GitHubAttempt, ...]):
        super().__init__(message)
        self.attempts = attempts


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _trim(text: str | None) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) > _ATTEMPT_STDERR_LIMIT:
        return collapsed[:_ATTEMPT_STDERR_LIMIT] + "..."
    return collapsed


def describe_gh_failure(result: CommandResult) -> str:
    """Render the final stderr plus attempt history (if any)."""
    stderr = (result.stderr or "").strip()
    attempts = getattr(result, "attempts", ())
    if not attempts:
        return stderr
    history = "\n".join(f"  {attempt.describe()}" for attempt in attempts)
    suffix = " (retry budget exhausted)" if getattr(result, "exhausted", False) else ""
    return f"{stderr}\nGitHub attempt history{suffix}:\n{history}"


def _failure_message(result: CommandResult) -> str:
    return (
        f"Command failed with exit {result.returncode}: {' '.join(result.args)}\n"
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )


def run_gh_read(
    runner: Runner,
    args: Sequence[str],
    *,
    cwd,
    check: bool = True,
    policy: GitHubRetryPolicy | None = None,
) -> CommandResult:
    """Run a caller-asserted read-only ``gh`` command, retrying transient failures.

    Non-transient failures are never retried and keep today's error shape.
    ``check=False`` callers receive a :class:`RetriedCommandResult` carrying
    the attempt history.  Dry-run bypasses the policy.
    """
    command = [str(a) for a in args]
    if getattr(runner, "dry_run", False):
        return runner.run(command, cwd=cwd, check=check)
    active = policy or DEFAULT_POLICY
    history: list[GitHubAttempt] = []
    for number in range(1, active.attempts + 1):
        started = time.monotonic()
        result = runner.run(command, cwd=cwd, check=False)
        if result.returncode == 0:
            if not history:
                return result
            return _with_history(result, history, exhausted=False)
        classification = classify_gh_failure(result)
        history.append(
            GitHubAttempt(number, result.returncode, classification, _trim(result.stderr), started)
        )
        if classification == "permanent":
            if len(history) == 1:
                # Untouched shape for the single-attempt permanent failure.
                if check:
                    raise AgentLoopError(_failure_message(result))
                return result
            break
        if number < active.attempts:
            _sleep(active.delay_before_retry(number))
    final = _with_history(result, history, exhausted=history[-1].classification == "transient")
    if check:
        if final.exhausted:
            raise GitHubTransientExhaustedError(
                f"GitHub command failed after {len(history)} transient attempt(s): "
                f"{' '.join(command)}\n{describe_gh_failure(final)}",
                tuple(history),
            )
        raise AgentLoopError(_failure_message(result))
    return final


def _with_history(
    result: CommandResult, history: Sequence[GitHubAttempt], *, exhausted: bool
) -> RetriedCommandResult:
    return RetriedCommandResult(
        args=result.args,
        cwd=result.cwd,
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
        observation=result.observation,
        capture_diagnostics=result.capture_diagnostics,
        launcher_args=result.launcher_args,
        containment=result.containment,
        attempts=tuple(history),
        exhausted=exhausted,
    )
