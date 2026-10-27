"""Qualification restarts preserve the distinction between head and identity."""

from .logging import log


def restart_reason(previous_head, context):
    changes = []
    if previous_head != context.metadata.head_sha:
        changes.append("PR head changed")
    if context.architecture_identity_changed:
        changes.append("architecture identity changed")
    if getattr(context, "converted_capture_review_pending", False):
        changes.append("converted capture requires fresh semantic review")
    return " and ".join(changes)


def checkpoint_restart(
    runner, *, config, pr_number, round_number, previous_head, context,
    items, checkpoint, managed_contract, allowed_rounds,
    watch_failure_extension_used, watch_head_extension_used, phase,
):
    from .pr_loop_support import _machine_obligation_checkpoint, _persist_qualification_checkpoint
    from .unresolved_items import _advance_machine_obligations_for_head

    reason = restart_reason(previous_head, context)
    log(config, f"Round {round_number}: {reason} {phase}; restarting exact-head scheduling")
    # Architecture-only invalidation is not a new candidate for head-bound work.
    if previous_head != context.metadata.head_sha:
        items = _advance_machine_obligations_for_head(items, current_head_sha=context.metadata.head_sha)
    checkpoint = _machine_obligation_checkpoint(
        items, previous_checkpoint=checkpoint, managed_contract=managed_contract,
        current_head_sha=context.metadata.head_sha,
        base_branch=context.metadata.base_branch or config.base,
        allowed_rounds=allowed_rounds, watch_failure_extension_used=watch_failure_extension_used,
        watch_head_extension_used=watch_head_extension_used,
        lifecycle="awaiting_current_head_review",
    )
    _persist_qualification_checkpoint(
        runner, config=config, pr_number=pr_number, round_number=round_number + 1,
        head_sha=context.metadata.head_sha, unresolved_items=items, checkpoint=checkpoint,
        message=f"PR #{pr_number} qualification checkpoint: {reason} {phase}; fresh review is required.",
    )
    return items, checkpoint


def review_inputs_changed(previous_head, context):
    """All qualification readers return to scheduling on these fresh inputs."""
    return (
        previous_head != context.metadata.head_sha
        or context.architecture_identity_changed
        or getattr(context, "converted_capture_review_pending", False)
    )
