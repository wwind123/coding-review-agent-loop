"""High-level issue, task, and PR orchestration loops."""

from __future__ import annotations

import sys

from .agents.base import AgentName
from .agents.registry import agent_display_name, run_agent
from .config import AgentLoopConfig, ensure_agent_workdirs, reviewers
from .errors import AgentLoopError
from .github import (
    merge_pr,
    post_pr_comment,
    validate_open_issue,
    validate_open_pr,
    wait_for_ci,
)
from .logging import log
from .prompts import (
    build_followup_prompt,
    build_issue_prompt,
    build_review_prompt,
    build_task_clarification_prompt,
    build_task_prompt,
    format_agent_list,
)
from .protocol import is_clarification_request, parse_agent_state, parse_pr_number
from .runner import Runner
from .workdirs import active_workdir


def run_optional_tests(runner: Runner, config: AgentLoopConfig) -> None:
    if not config.test_command:
        return
    log(config, f"Running local test command: {' '.join(config.test_command)}")
    runner.run(config.test_command, cwd=active_workdir(config))
    log(config, "Local test command passed")


def run_issue_loop(runner: Runner, *, issue_number: int, config: AgentLoopConfig) -> int:
    ensure_agent_workdirs(config, runner)
    log(config, f"Validating issue #{issue_number}")
    validate_open_issue(runner, config=config, issue_number=issue_number)

    coder_output, coder_session_id = run_agent(
        runner,
        agent=config.coder,
        config=config,
        prompt=build_issue_prompt(issue_number, config),
    )
    pr_number = parse_pr_number(coder_output)
    if pr_number is None:
        raise AgentLoopError(
            f"{agent_display_name(config.coder)} output did not include a PR marker or PR URL."
        )
    log(config, f"{agent_display_name(config.coder)} reported PR #{pr_number}; validating it is open")
    validate_open_pr(runner, config=config, pr_number=pr_number)

    post_pr_comment(runner, config=config, pr_number=pr_number, body=coder_output)
    return run_pr_loop(
        runner,
        pr_number=pr_number,
        config=config,
        coder_session_id=coder_session_id,
        workdirs_ready=True,
    )


def _read_clarification_from_stdin() -> str:
    print(
        "\nProvide clarification (one entry per line; finish with a single '.' line or Ctrl+D):",
        file=sys.stderr,
        flush=True,
    )
    lines: list[str] = []
    try:
        while True:
            line = input()
            if line.strip() == ".":
                break
            lines.append(line)
    except EOFError:
        pass
    return "\n".join(lines)


def run_task_loop(
    runner: Runner,
    *,
    task_text: str,
    config: AgentLoopConfig,
    interactive: bool = False,
    max_clarification_rounds: int = 3,
    clarification_input=None,
) -> int:
    ensure_agent_workdirs(config, runner)
    if not task_text.strip():
        raise AgentLoopError("Task text is empty; provide a non-empty description.")
    if max_clarification_rounds < 0:
        raise AgentLoopError("--max-clarification-rounds must be zero or positive.")

    history: list[tuple[str, str]] = []
    prompt = build_task_prompt(task_text, config)
    read_clarification = clarification_input or _read_clarification_from_stdin
    coder_name = agent_display_name(config.coder)
    session_id: str | None = None

    for attempt in range(max_clarification_rounds + 1):
        log(config, f"Task attempt {attempt + 1}: invoking {coder_name}")
        coder_output, session_id = run_agent(
            runner,
            agent=config.coder,
            config=config,
            prompt=prompt,
            session_id=session_id,
        )
        if not coder_output.strip():
            raise AgentLoopError(f"{coder_name} produced an empty response.")

        pr_number = parse_pr_number(coder_output)
        if pr_number is not None:
            log(config, f"{coder_name} reported PR #{pr_number}; validating it is open")
            validate_open_pr(runner, config=config, pr_number=pr_number)
            post_pr_comment(runner, config=config, pr_number=pr_number, body=coder_output)
            return run_pr_loop(
                runner,
                pr_number=pr_number,
                config=config,
                coder_session_id=session_id,
                workdirs_ready=True,
            )

        if not is_clarification_request(coder_output):
            raise AgentLoopError(
                f"{coder_name} output did not include a PR marker, PR URL, "
                "or clarification marker."
            )

        if not interactive:
            raise AgentLoopError(
                f"{coder_name} requested clarification but the loop is non-interactive. "
                "Add the missing details to the task text or rerun with --interactive.\n\n"
                f"{coder_name}'s questions:\n{coder_output}"
            )

        if attempt >= max_clarification_rounds:
            raise AgentLoopError(
                f"{coder_name} still requested clarification after "
                f"{max_clarification_rounds} rounds; "
                "human intervention required."
            )

        log(config, f"{coder_name} requested clarification (round {attempt + 1}); awaiting user input")
        print(coder_output, flush=True)
        answers = read_clarification()
        if not answers.strip():
            raise AgentLoopError("Empty clarification reply; aborting task.")
        history.append((coder_output, answers))
        prompt = build_task_clarification_prompt(task_text, history, config)

    raise AgentLoopError("run_task_loop exited unexpectedly without producing a PR.")


def run_pr_loop(
    runner: Runner,
    *,
    pr_number: int,
    config: AgentLoopConfig,
    coder_session_id: str | None = None,
    reviewer_session_id: str | None = None,
    workdirs_ready: bool = False,
) -> int:
    if not workdirs_ready:
        ensure_agent_workdirs(config, runner)
    log(config, f"Validating PR #{pr_number}")
    validate_open_pr(runner, config=config, pr_number=pr_number)
    reviewer_session_ids: dict[AgentName, str | None] = {}
    configured_reviewers = reviewers(config)
    if reviewer_session_id is not None and configured_reviewers:
        # Backward-compatible single-reviewer resume support: older callers
        # pass one reviewer session, so attach it to the first configured reviewer.
        reviewer_session_ids[configured_reviewers[0]] = reviewer_session_id

    for round_number in range(1, config.max_rounds + 1):
        coder_name = agent_display_name(config.coder)
        blocking_reviews: list[tuple[str, str]] = []
        for reviewer in configured_reviewers:
            reviewer_name = agent_display_name(reviewer)
            log(config, f"Round {round_number}: {reviewer_name} reviewing PR #{pr_number}")
            review_output, new_session_id = run_agent(
                runner,
                agent=reviewer,
                config=config,
                prompt=build_review_prompt(
                    pr_number,
                    round_number,
                    config,
                    reviewer=reviewer,
                ),
                session_id=reviewer_session_ids.get(reviewer),
            )
            reviewer_session_ids[reviewer] = new_session_id
            if not review_output.strip():
                raise AgentLoopError(f"{reviewer_name} produced an empty response.")

            post_pr_comment(runner, config=config, pr_number=pr_number, body=review_output)
            review_state = parse_agent_state(review_output)
            log(config, f"Round {round_number}: {reviewer_name} state is {review_state}")
            if review_state == "blocking":
                blocking_reviews.append((reviewer_name, review_output))

        if not blocking_reviews:
            run_optional_tests(runner, config)
            if config.auto_merge:
                wait_for_ci(runner, config, pr_number)
                merge_pr(runner, config, pr_number)
            print(f"PR #{pr_number} approved by {format_agent_list(configured_reviewers)}.")
            return 0
        if round_number == config.max_rounds:
            raise AgentLoopError(
                f"One or more reviewers still reported blocking issues after round {round_number}; "
                "human review required."
            )

        combined_review = "\n\n".join(
            f"{name} review:\n\n{review}" for name, review in blocking_reviews
        )
        log(config, f"Round {round_number}: {coder_name} addressing reviewer feedback")
        coder_output, coder_session_id = run_agent(
            runner,
            agent=config.coder,
            config=config,
            prompt=build_followup_prompt(pr_number, round_number, combined_review, config),
            session_id=coder_session_id,
        )
        if not coder_output.strip():
            raise AgentLoopError(f"{coder_name} produced an empty response.")

        post_pr_comment(runner, config=config, pr_number=pr_number, body=coder_output)
        # Validate marker presence; reviewer remains the merge gate on the next round.
        parse_agent_state(coder_output)
        log(config, f"Round {round_number}: {coder_name} pushed updates for re-review")

    raise AgentLoopError(
        f"Reached max rounds ({config.max_rounds}) for PR #{pr_number}; human review required."
    )
