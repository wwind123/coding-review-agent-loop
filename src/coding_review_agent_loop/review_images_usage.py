"""One accounting owner for single-acquisition image and text coverage turns."""
from __future__ import annotations

from contextlib import contextmanager
import json

from .usage import UsageMetadata, estimate_usage, NUMERIC_USAGE_FIELDS


def _provider_usage(provider, raw):
    """Bound provider metadata separately from proof; it grants no delivery authority."""
    if not isinstance(raw, str) or len(raw.encode()) > 4 * 1024 * 1024:
        return None
    usage = None
    if provider == 'codex':
        from .agents.codex import _extract_codex_usage
        usage, _ = _extract_codex_usage(raw)
    elif provider == 'claude':
        from .agents.claude import _normalize_claude_usage
        for line in raw.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get('type') == 'result':
                usage = _normalize_claude_usage(event.get('usage'))
    if usage is not None and any(v is not None and (type(v) is not int or not 0 <= v <= 10**12)
                                 for v in (getattr(usage, k) for k in NUMERIC_USAGE_FIELDS)):
        return None
    return usage


def provider_usage(provider, raw):
    try:
        return _provider_usage(provider, raw)
    except (ValueError, TypeError, RecursionError, OverflowError):
        return None


@contextmanager
def accounting(context, *, agent, kind, prompt, acquisition_id=None):
    """Record once, including proof/validation failure and unknown interrupted usage."""
    state = {'result': None, 'outcome': 'invalid_output'}
    if context is not None:
        context.note_agent_dispatch()
    try:
        yield state
    finally:
        if context is not None:
            result = state['result']
            usage = getattr(result, 'usage', None)
            estimated = usage is None
            if result is None:
                usage = UsageMetadata(mode='partial')
            elif usage is None:
                usage = estimate_usage(prompt, result.text)
            record = context.add_record(agent=agent, session_id=getattr(result, 'session_id', None),
                returncode=getattr(result, 'returncode', None), usage=usage,
                turn_role='visual-' + kind, outcome=state['outcome'],
                model=getattr(result, 'model_used', None))
            record.validation_status = "validated" if state["outcome"] == "succeeded" else "invalid"
            record.visual_kind = kind
            record.visual_acquisition_id = acquisition_id
            record.visual_usage_provenance = ('unreported' if result is None else 'text-estimate' if estimated else 'provider')
            from .reviewer_seats import seat_backend
            record.backend = seat_backend(agent)
            record.image_payload_unknown = kind == 'image-assessment' and (estimated or usage.input_tokens is None)
