"""Bounded post-approval plumbing routing; dependencies supplied by the loop."""
from __future__ import annotations
from .runner import Runner
from .config import AgentLoopConfig
from .managed_ci import FINAL_CONTEXT

# --- Post-approval plumbing routing (#1477) ---------------------------------


class _ManagedRerunContinue:
    """Watch the managed qualification again from ``min_attempt``."""

    def __init__(self, min_attempt: int) -> None:
        self.min_attempt = min_attempt


MANAGED_PUBLISHED_FAILURE_RECOVERY = (
    "No automatic retry exists for a managed run that published a failure status. "
    "Rerunning this command on the same head reattaches the same failed run and stops "
    "here again. Recovery needs a new head commit on the PR (current-head review then "
    "mints a fresh head-scoped managed intent and dispatch) or the managed resume and "
    "ordinary-recovery options described in docs/local_agent_loop.md."
)


def _route_managed_plumbing_outcome(
    runner: Runner,
    api,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    round_number: int,
    head_sha: str | None,
    outcome,
    contract,
    reruns_used: set[str],
    budget_left: bool,
    deadline: float | None = None,
):
    """Apply the plumbing rerun-once rule to one managed qualification outcome.

    Returns ``None`` to keep the existing outcome handling, ``"coder"`` for a
    classified test failure (board or run jobs; test precedence), an exit code
    for an infrastructure stop, or ``_ManagedRerunContinue`` after a verified
    rerun.  Plumbing-only also requires an authoritative complete PR-head board.
    """
    if outcome.status not in {"terminal_without_status", "failed"} or not head_sha:
        return None
    names = api.plumbing_check_names(config)
    board = api.classify_managed_board(outcome.checks, names)
    jobs = api.classify_managed_jobs(outcome.jobs, names)
    if board == "test-failure" or jobs == "test-failure":
        return "coder"
    if board != "ok" or jobs != "plumbing-only":
        return None
    failed_jobs = [
        f"{name}: {conclusion}" + (f" ({url})" if url else "")
        for name, conclusion, url in (outcome.jobs or ())
        if conclusion not in {"success", "skipped", "neutral"}
    ]
    run_url = api.managed_run_url(config, outcome.run_id)
    run_urls = (run_url,) if run_url else ()

    def stop(reason: str, final_state: str, recovery: str | None = None) -> int:
        return api._stop_on_ci_plumbing_failure(
            runner,
            config=config,
            pr_number=pr_number,
            round_number=round_number,
            reason=reason,
            run_urls=run_urls,
            plumbing=failed_jobs,
            final_line=api.final_context_state_line(None, managed_status=final_state),
            recovery=recovery,
        )

    if outcome.status == "failed":
        # The record is plain ``completed``; the dispatch validator rejects
        # attempt N+1 against it, so no rerun and no record rewrite (#1477).
        return stop(
            f"Managed run {outcome.run_id} published a failure status although only "
            "plumbing jobs failed and every test job passed.",
            "failure",
            MANAGED_PUBLISHED_FAILURE_RECOVERY,
        )
    if head_sha in reruns_used:
        return stop(
            "Plumbing failed again after the single automatic managed rerun.",
            "absent (run completed without a correlated status)",
        )
    if not budget_left:
        # The shared attempts or deadline are spent: never issue a rerun that
        # nothing will observe; the existing terminal handling applies.
        return None
    reruns_used.add(head_sha)
    rerun = api.rerun_managed_attempt(
        runner, config=config, contract=contract, expected_head_sha=head_sha,
        deadline=deadline,
    )
    if rerun.status == "expired":
        # Verification reads used up the shared deadline: no write; the
        # existing terminal handling applies with no coder or merge.
        return None
    if rerun.status != "rerun_started" or rerun.prior_attempt is None:
        return stop(
            f"The single automatic managed rerun could not be performed: {rerun.reason}.",
            "absent (run completed without a correlated status)",
        )
    message = (
        f"PR #{pr_number}: managed run {rerun.run_url or rerun.run_id} attempt "
        f"{rerun.prior_attempt} completed without a correlated `{FINAL_CONTEXT}` status "
        "and only plumbing jobs failed; re-ran the whole run once (validation runs again "
        "in the new attempt); no coder round."
    )
    api.log(config, f"Round {round_number}: {message}")
    api.post_pr_comment(
        runner, config=config, pr_number=pr_number,
        body=message + "\n\n-- coding-review-agent-loop",
    )
    return _ManagedRerunContinue(rerun.prior_attempt + 1)
