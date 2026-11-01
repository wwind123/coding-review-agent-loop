"""One fresh image acquisition, using the existing reviewer execution boundary."""
from __future__ import annotations

import base64
from dataclasses import replace
import json
import math
import time

from .agents.base import AgentResult, public_response_path
from .agent_permissions import (is_sandboxed, prepare_sandboxed_spawn,
                                role_permission_args, role_permission_env)
from .config import resolve_invocation
from .logging import agent_log_path
from .review_images_capabilities import require_image_invocation, executable_identity
from .review_images_proof import claude_proof, codex_proof
from .test_runtime import agent_shell_environment
from .worktree_hints import dependency_hint_environment


def run_image_adapter(runner, config, prompt, images, *, provider, role, session_id,
                      run_id, label, timeout_seconds, attempt_suffix):
    """No retry, fallback, resume, raw-text fallback or proof from model output."""
    try:
        return _acquire(runner, config, prompt, images, provider=provider, role=role,
                        session_id=session_id, run_id=run_id, label=label,
                        timeout_seconds=timeout_seconds, attempt_suffix=attempt_suffix)
    finally:
        images.close()


def _acquire(runner, config, prompt, images, *, provider, role, session_id,
             run_id, label, timeout_seconds, attempt_suffix):
    config = require_image_invocation(images, config, images.binding.seat,
                                      role=role, session_id=session_id)
    if provider != images.binding.provider or config.dry_run:
        raise ValueError("image contract requires the selected provider and actual execution")
    if timeout_seconds is None or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 1800:
        raise ValueError("image acquisition requires a bounded timeout, at most 1800 seconds")
    if len(prompt.encode()) > 256 * 1024:
        raise ValueError("image prompt exceeds contract bound")
    invocation = resolve_invocation(config, provider=provider, role="reviewer")
    cwd = getattr(config, f"{provider}_dir")
    if is_sandboxed(config):
        response = prepare_sandboxed_spawn(config, provider, "reviewer")
        permissions = role_permission_args(config, provider, "reviewer", response_dir=response.parent)
    else:
        response = public_response_path(config, provider)
        permissions = ()
    prompt = f"Visual acquisition {images.binding.acquisition_id}.\n{prompt}\nReturn your assessment in the final response."
    executable = images.capability.executable
    if provider == "claude":
        args = [executable, "--print", "--input-format", "stream-json", "--output-format",
                "stream-json", "--verbose", "--replay-user-messages", "--model", invocation.configured_model,
                "--effort", invocation.resolved_effort, *config.claude_args, *permissions,
                *(() if is_sandboxed(config) else ("--tools", "")),
                "--no-session-persistence", "--setting-sources", ""]
        blocks = [{"type": "text", "text": prompt}]
        blocks.extend({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                       "data": base64.b64encode(images.read(i)).decode()}}
                      for i in range(len(images.entries)))
        stdin = json.dumps({"type": "user", "session_id": "", "uuid": images.binding.acquisition_id,
                            "message": {"role": "user", "content": blocks}}) + "\n"
    else:
        from .agents.codex import _codex_model_args
        args = [executable, "exec", "--cd", str(cwd), "--json", "--output-last-message", str(response),
                *_codex_model_args(config, role="reviewer"), *config.codex_args, *permissions,
                *(f"--image={path}" for path in images.paths()), "--", "-"]
        stdin = prompt
    log_path = agent_log_path(config, provider, run_id=run_id, label=label, attempt_suffix=attempt_suffix)
    env = {"AGENT_LOOP_WORKDIR": str(cwd.resolve()),
           **dependency_hint_environment(config, cwd),
           **role_permission_env(config, provider, "reviewer"),
           **agent_shell_environment(config, provider, "reviewer")}
    if provider == "claude":
        env["CLAUDE_CODE_EFFORT_LEVEL"] = invocation.resolved_effort
    identity = executable_identity(executable)
    if identity != (images.capability.executable_identity, images.capability.executable_digest):
        raise ValueError("image CLI identity changed before spawn")
    started = time.time()
    result = runner.run_with_log(args, cwd=cwd, log_path=log_path, label=provider.title(),
                                 progress_interval_seconds=config.progress_interval_seconds,
                                 check=False, env=env, input_text=stdin,
                                 stdin_log_summary=f"[immutable image acquisition {images.binding.acquisition_id}; payload redacted]",
                                 timeout_seconds=timeout_seconds, containment_role="reviewer")
    ended = time.time()
    if result.containment is not None and result.containment.managed_cleanup_unconfirmed:
        sanitized = replace(result, stdout="", stderr="", capture_diagnostics=())
        return AgentResult(text="Image acquisition cleanup-failed: managed scope emptiness was not confirmed.",
                           raw_output="", message_text=None, returncode=result.returncode or 1,
                           provider=provider, role="reviewer", log_path=log_path,
                           command_result=sanitized)
    # On every failure exit only adapter-generated diagnostics leave this module.
    # Replay bytes stay in the private invocation log, never recovery candidates.
    try:
        if executable_identity(executable) != identity:
            raise ValueError("image CLI identity changed during acquisition")
        if provider == "claude":
            text, proof = claude_proof(result.stdout, images)
        else:
            from .agents.codex import _codex_session_root
            text, proof = codex_proof(result.stdout, images, session_root=_codex_session_root(),
                                      cwd=cwd, started=started, ended=ended)
    except (ValueError, OSError, KeyError, TypeError, AttributeError, UnicodeError):
        return AgentResult(text="Image acquisition lacks complete delivery/model evidence.",
                           raw_output="", message_text=None, returncode=result.returncode or 1,
                           provider=provider, role="reviewer", log_path=log_path)
    sanitized = replace(result, stdout=text, stderr="", capture_diagnostics=())
    return AgentResult(text=text, raw_output=text, message_text=text, session_id=proof.session_id,
                       log_path=log_path, returncode=result.returncode,
                       provider=provider, role="reviewer", configured_model=invocation.configured_model,
                       configured_effort=invocation.resolved_effort, effort_source=invocation.effort_source,
                       observed_model=invocation.configured_model,
                       observation_provenance=proof.model_evidence_source,
                       image_delivery_proof=proof, command_result=sanitized)
