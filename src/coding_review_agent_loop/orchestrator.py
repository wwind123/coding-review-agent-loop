"""High-level issue, task, and PR orchestration loops."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from .agents.base import AgentName
from .agents.registry import agent_display_name, run_agent
from .config import AgentLoopConfig, ensure_agent_workdirs, reviewers
from .errors import AgentLoopError
from .github import (
    create_issue,
    get_pr_metadata,
    merge_pr,
    post_pr_comment,
    validate_open_issue,
    validate_open_pr,
    wait_for_ci,
)
from .logging import log
from .memory import prepare_agent_memory
from .prompts import (
    build_followup_prompt,
    build_issue_prompt,
    build_review_prompt,
    build_same_pr_followup_prompt,
    build_task_clarification_prompt,
    build_task_prompt,
    format_agent_list,
)
from .protocol import is_clarification_request, parse_agent_state, parse_pr_number
from .protocol import ApprovedFollowup, parse_approved_followups
from .runner import Runner
from .workdirs import active_workdir

MAX_APPROVED_FOLLOWUP_ISSUES = 3


def run_optional_tests(runner: Runner, config: AgentLoopConfig) -> None:
    if not config.test_command:
        return
    log(config, f"Running local test command: {' '.join(config.test_command)}")
    runner.run(config.test_command, cwd=active_workdir(config))
    log(config, "Local test command passed")


def _format_approved_followup_summary(pr_number: int, followups: list[ApprovedFollowup]) -> str:
    lines = [
        f"Approved-review future follow-ups for PR #{pr_number}:",
        "",
    ]
    for followup in followups:
        lines.append(f"- {followup.text} ({followup.reviewer})")
    lines.extend(
        [
            "",
            "These were mentioned in approved reviews as future work and did not block merge readiness.",
            "",
            "-- coding-review-agent-loop",
        ]
    )
    return "\n".join(lines)


def _followup_issue_title(followup: ApprovedFollowup) -> str:
    text = " ".join(followup.text.split())
    title = f"Follow up future review note: {text}"
    return title[:120]


def _followup_issue_body(pr_number: int, followup: ApprovedFollowup) -> str:
    return "\n".join(
        [
            f"Future follow-up from approved review on PR #{pr_number}.",
            "",
            f"Reviewer: {followup.reviewer}",
            "",
            "Follow-up:",
            f"- {followup.text}",
            "",
            "This was mentioned in an approved review as future work and did not block merge readiness.",
            "",
            "-- OpenAI Codex",
        ]
    )


def _create_approved_followup_issues(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    followups: list[ApprovedFollowup],
) -> None:
    selected_followups = followups[:MAX_APPROVED_FOLLOWUP_ISSUES]
    for followup in selected_followups:
        create_issue(
            runner,
            config=config,
            title=_followup_issue_title(followup),
            body=_followup_issue_body(pr_number, followup),
        )
    skipped_count = len(followups) - len(selected_followups)
    if skipped_count <= 0:
        return
    post_pr_comment(
        runner,
        config=config,
        pr_number=pr_number,
        body=(
            f"Created follow-up issues for the first {len(selected_followups)} "
            f"approved-review future items. Skipped {skipped_count} additional "
            "item(s) to avoid issue noise; reviewers should reserve this section for "
            "substantial independent follow-up work.\n\n-- coding-review-agent-loop"
        ),
    )


def _format_same_pr_followups(followups: Sequence[ApprovedFollowup]) -> str:
    lines: list[str] = []
    for followup in followups:
        lines.append(f"{followup.reviewer} same-PR follow-up:")
        lines.append(f"- {followup.text}")
        lines.append("")
    return "\n".join(lines).strip()


def run_issue_loop(runner: Runner, *, issue_number: int, config: AgentLoopConfig) -> int:
    ensure_agent_workdirs(config, runner)
    log(config, f"Validating issue #{issue_number}")
    validate_open_issue(runner, config=config, issue_number=issue_number)
    memory = prepare_agent_memory(runner, config)

    coder_output, coder_session_id = run_agent(
        runner,
        agent=config.coder,
        config=config,
        prompt=build_issue_prompt(issue_number, config, memory),
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
    if not task_text.strip():
        raise AgentLoopError("Task text is empty; provide a non-empty description.")
    if max_clarification_rounds < 0:
        raise AgentLoopError("--max-clarification-rounds must be zero or positive.")
    ensure_agent_workdirs(config, runner)
    memory = prepare_agent_memory(runner, config)

    history: list[tuple[str, str]] = []
    prompt = build_task_prompt(task_text, config, memory)
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
        prompt = build_task_clarification_prompt(task_text, history, config, memory)

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
    memory = prepare_agent_memory(runner, config)
    reviewer_session_ids: dict[AgentName, str | None] = {}
    configured_reviewers = reviewers(config)
    if reviewer_session_id is not None and configured_reviewers:
        # Backward-compatible single-reviewer resume support: older callers
        # pass one reviewer session, so attach it to the first configured reviewer.
        reviewer_session_ids[configured_reviewers[0]] = reviewer_session_id
    pending_future_followups: list[ApprovedFollowup] = []

    for round_number in range(1, config.max_rounds + 1):
        coder_name = agent_display_name(config.coder)
        blocking_reviews: list[tuple[str, str]] = []
        same_pr_followups: list[ApprovedFollowup] = []
        round_future_followups: list[ApprovedFollowup] = []
        pr_metadata = get_pr_metadata(runner, config=config, pr_number=pr_number)
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
                    pr_metadata=pr_metadata,
                    memory=memory,
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
            elif config.approved_followups != "ignore":
                followups = parse_approved_followups(review_output, reviewer=reviewer_name)
                round_future_followups.extend(followups.future)
                if followups.same_pr:
                    if config.approved_followups.startswith("fix-and-"):
                        same_pr_followups.extend(followups.same_pr)
                    else:
                        blocking_reviews.append(
                            (
                                reviewer_name,
                                "\n".join(
                                    [
                                        "Approved review included Same-PR follow-ups, "
                                        f"but --approved-followups={config.approved_followups} "
                                        "does not enable a same-PR fix path.",
                                        "",
                                        _format_same_pr_followups(followups.same_pr),
                                    ]
                                ),
                            )
                        )

        if not blocking_reviews and not same_pr_followups:
            approved_followups = [*pending_future_followups, *round_future_followups]
            if config.approved_followups in ("summarize", "fix-and-summarize") and approved_followups:
                body = _format_approved_followup_summary(pr_number, approved_followups)
                post_pr_comment(runner, config=config, pr_number=pr_number, body=body)
            elif config.approved_followups in ("issue", "fix-and-issue") and approved_followups:
                _create_approved_followup_issues(
                    runner,
                    config=config,
                    pr_number=pr_number,
                    followups=approved_followups,
                )
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

        if same_pr_followups and not blocking_reviews:
            pending_future_followups.extend(round_future_followups)
            combined_review = _format_same_pr_followups(same_pr_followups)
            followup_prompt = build_same_pr_followup_prompt(
                pr_number,
                round_number,
                combined_review,
                config,
                memory,
            )
        else:
            # Future follow-ups are only retained for fully approved same-PR fix
            # rounds. If any reviewer blocks, future-work suggestions from that
            # round are discarded so reviewers can restate still-relevant items
            # after the blocking issues have been resolved.
            if same_pr_followups:
                blocking_reviews.append(
                    (
                        "Approved same-PR follow-ups",
                        _format_same_pr_followups(same_pr_followups),
                    )
                )
            combined_review = "\n\n".join(
                f"{name} review:\n\n{review}" for name, review in blocking_reviews
            )
            followup_prompt = build_followup_prompt(
                pr_number,
                round_number,
                combined_review,
                config,
                memory,
            )
        log(config, f"Round {round_number}: {coder_name} addressing reviewer feedback")
        coder_output, coder_session_id = run_agent(
            runner,
            agent=config.coder,
            config=config,
            prompt=followup_prompt,
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
