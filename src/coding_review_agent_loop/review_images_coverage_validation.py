"""Single text acquisition with no retry, fallback, code review or new image input."""
from __future__ import annotations

import json

from .agent_failure import ValidatedAgentResponse
from .errors import AgentLoopError
from .review_images_coverage import decisions, validate_scope
from .review_images_usage import accounting


def validate_coverage_turn(runner, *, coverage_scope, coverage_report, run_agent, agent, config,
                           prompt, usage_context=None, timeout_seconds=300, coverage_acquisition_id=None, **unused):
    validate_scope(coverage_scope)
    with accounting(usage_context, agent=agent, kind='requester-coverage', prompt=prompt, acquisition_id=coverage_acquisition_id) as audit:
        result = run_agent(runner, agent=agent, config=config, prompt=prompt, role='reviewer',
                           session_id=None, timeout_seconds=timeout_seconds,
                           run_id=usage_context.run_id if usage_context else None,
                           label='screenshot-coverage')
        audit['result'] = result
        if result.returncode != 0:
            audit['outcome'] = 'timeout' if result.returncode is None else 'nonzero_exit'
            raise AgentLoopError('Screenshot coverage acquisition failed; its budget remains spent')
        if result.containment is not None and result.containment.managed_cleanup_unconfirmed:
            raise AgentLoopError('Screenshot coverage cleanup could not be verified; retry is blocked')
        # Only deterministic envelope cleanup is permitted. No second acquisition.
        from .repair import attempt_envelope_normalization
        artifact = result.response_file_text
        if not artifact:
            raise AgentLoopError('Screenshot coverage requires the acquisition response artifact')
        normalized = attempt_envelope_normalization(artifact, expected_kind='screenshot-coverage')
        if normalized is None:
            raise AgentLoopError('Screenshot coverage response cannot be extracted')
        value = json.loads(normalized)
        decisions(value, coverage_scope, coverage_report)
        audit['outcome'] = 'succeeded'
        return ValidatedAgentResponse(text=json.dumps(value), marker_value='coverage',
            session_id=result.session_id, usage=result.usage, acquisition_outcome='success',
            acquisition_returncode=0)


def guard_ordinary_validation(runner, kwargs):
    from .protocol import ParsedReview
    from .review_images_coverage_run import validate_ordinary, history_for, restore
    config = kwargs.get('config')
    if config is None:
        return
    for name in ('validate', 'strict_revalidate'):
        original = kwargs.get(name)
        if original is None:
            continue
        def checked(text, original=original):
            parsed = original(text)
            if not isinstance(parsed, ParsedReview) or not (
                    parsed.screenshot_requests or parsed.screenshot_coverage_acceptances):
                return parsed
            pr = getattr(runner, '_screenshot_coverage_pr', None)
            if not config.review_images_paths or pr is None:
                raise AgentLoopError('Screenshot authority is not enabled for this turn')
            from .agents.registry import agent_display_name
            from .pr_head_sync import get_observed_pr_review_context
            head = get_observed_pr_review_context(runner, config=config, pr_number=pr).metadata.head_sha
            history = history_for(runner, config, pr)
            items, _ = restore(history, config=config, pr=pr, head=head, items=(), number=1)
            return validate_ordinary(parsed, runner=runner, config=config, pr=pr, head=head,
                requester=agent_display_name(kwargs['agent']), items=items)
        kwargs[name] = checked
