"""Conservative, single-acquisition validated image turns."""
from __future__ import annotations

from .agent_failure import ValidatedAgentResponse, _response_identity_fields
from .architecture_contract import _accept_candidate
from .errors import AgentLoopError
from .repair import attempt_envelope_normalization


def validate_image_turn(runner, *, image_inputs, run_agent, agent, config, prompt,
                        validate, role=None, session_id=None, timeout_seconds=None,
                        label=None, use_repair=False, repair_expected_kind=None,
                        strict_revalidate=None, **_unused):
    """Repair the text envelope only; never reassess or resume the image turn.

    Missing proof is terminal before any candidate, file recovery, normalization
    or diagnostic enters the ordinary text recovery state machine.
    """
    binding = image_inputs.binding
    with image_inputs:
        result = run_agent(runner, agent=agent, config=config, prompt=prompt,
                           role=role, session_id=session_id, label=label,
                           timeout_seconds=timeout_seconds, image_inputs=image_inputs)
        if result.containment is not None and result.containment.managed_cleanup_unconfirmed:
            raise AgentLoopError("Image turn cleanup-failed: managed scope emptiness was not confirmed; retry is blocked")
        proof = result.image_delivery_proof
        if proof is None:
            raise AgentLoopError("Image turn has no complete acquisition delivery proof")
        try:
            proof.require(binding)
        except ValueError:
            raise AgentLoopError("Image delivery/model binding rejected") from None
        text = result.text
        try:
            parsed = validate(text)
        except (AgentLoopError, ValueError):
            if not use_repair:
                raise
            # This deliberately has no backend, prompt, path or image access.
            # The immutable original proof remains out of band.
            normalized = attempt_envelope_normalization(text, expected_kind=repair_expected_kind)
            if normalized is None:
                raise AgentLoopError("Image response requires unsupported format repair") from None
            text = normalized
            parsed = validate(text)
        proof.require(binding)
        candidate = _accept_candidate(text, parsed, strict_revalidate=strict_revalidate,
                                      runner=runner, acquisition=result)
        return ValidatedAgentResponse(
            text=candidate.text, marker_value=candidate.marker_value,
            session_id=result.session_id, usage=result.usage, model_used=result.model_used,
            image_delivery_proof=proof,
            acquisition_outcome=("success" if result.returncode == 0 else
                                 "accepted_timeout" if result.returncode is None else "accepted_nonzero_exit"),
            acquisition_returncode=result.returncode, **_response_identity_fields(result))
