"""Local, unpaid admission of concrete image CLI contracts and reviewer seats."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

from .config import DEFAULT_REASONING_EFFORT, resolve_invocation
from .reviewer_seats import SeatAgent, seat_backend


# Admission requires the opt-in real CLI suite and committed replay provenance.
# Versions are exact, not minimums. No alias expansion or implicit fallback.
CATALOG_VERSION = 1
EXECUTABLE_DIGESTS = {
    "codex-native-v1": "b8a11fedf24b2dfb59a70c9e52b030f9a508dcd49cd2dd0d41956f4c785156b8",
    "claude-stream-v1": "fd13c70000fec3f2116dfbcbb6121210b4a29328bf0e46210a59dd594a67d824",
}
CONTRACTS = {
    ("codex", "codex-cli 0.166.0", "gpt-6.1-sol"): "codex-native-v1",
    ("claude", "2.1.316 (Claude Code)", "claude-sonnet-4-6"): "claude-stream-v1",
}


@dataclass(frozen=True)
class ImageCapability:
    seat: str
    provider: str
    model: str
    effort: str
    model_chain: tuple[str, ...]
    contract_id: str
    executable: str
    executable_digest: str
    executable_identity: tuple[int, int, int, int]
    version: str
    permission_mode: str
    invocation_view: object


def reviewer_view(config, selection=None):
    selection = config.primary_reviewer if selection is None else selection
    matches = [agent for agent in config.reviewer if agent == selection]
    if selection is None or len(matches) != 1:
        raise ValueError("image review requires exactly one configured reviewer")
    agent = matches[0]
    provider = seat_backend(agent)
    if provider not in {"codex", "claude"}:
        raise ValueError("reviewer backend has no admitted image transport")
    if isinstance(agent, SeatAgent):
        overrides = {f"{provider}_dir": agent.workdir, "active_reviewer_seat_id": str(agent),
                     f"reviewer_{provider}_model": agent.model_chain[0]}
        overrides["reviewer_codex_reasoning_effort" if provider == "codex" else "reviewer_claude_effort"] = agent.effort or DEFAULT_REASONING_EFFORT
        config = replace(config, **overrides)
        chain = agent.model_chain
    else:
        config = replace(config, active_reviewer_seat_id=None)
        chain = (resolve_invocation(config, provider=provider, role="reviewer").configured_model,)
    invocation = resolve_invocation(config, provider=provider, role="reviewer")
    if not invocation.configured_model or not all(chain):
        raise ValueError("image review requires an explicit configured model")
    return agent, config, invocation, chain


def executable_identity(path):
    info = os.stat(path)
    if not os.path.isfile(path) or not os.access(path, os.X_OK):
        raise ValueError("image CLI is not executable")
    with open(path, "rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns), checksum


def preflight_images(config, selection=None) -> ImageCapability:
    """No model call. Results belong to this invocation and are rechecked at spawn."""
    agent, view, invocation, chain = reviewer_view(config, selection)
    provider = seat_backend(agent)
    args = getattr(view, f"{provider}_args")
    allowed = {"--dangerously-skip-permissions"} if provider == "claude" else {
        "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check"}
    if any(arg not in allowed for arg in args):
        raise ValueError("image transport refuses unverified CLI passthrough options")
    command = shutil.which(getattr(view, f"{provider}_cmd"))
    if command is None:
        raise ValueError("image CLI is unavailable")
    executable = str(Path(command).resolve(strict=True))
    identity, checksum = executable_identity(executable)
    # Probe with isolated configuration. The pinned release can emit benign
    # alias-setup diagnostics under a temporary home; bounded stderr is not
    # capability evidence. Admission still requires exact version and hash.
    with tempfile.TemporaryDirectory(prefix="image-cli-probe-") as probe_home:
        probe = subprocess.run([executable, "--version"], stdin=subprocess.DEVNULL,
                               capture_output=True, timeout=5, check=True,
                               env={"PATH": "/usr/bin:/bin", "HOME": probe_home, "LC_ALL": "C"})
    if len(probe.stdout) > 1024 or len(probe.stderr) > 4096:
        raise ValueError("unrecognized image CLI version output")
    version = probe.stdout.decode().strip()
    contracts = [CONTRACTS.get((provider, version, model)) for model in chain]
    if not all(contracts) or len(set(contracts)) != 1:
        raise ValueError("unverified executable/model/fallback image contract")
    if checksum != EXECUTABLE_DIGESTS[contracts[0]]:
        raise ValueError("image executable lacks concrete conformance admission")
    return ImageCapability(str(agent), provider, invocation.configured_model,
                           invocation.resolved_effort, tuple(chain), contracts[0], executable,
                           checksum, identity, version, view.agent_permissions, view)


def require_image_invocation(images, config, agent, *, role, session_id):
    if role != "reviewer" or session_id is not None:
        raise ValueError("images require a fresh scoped reviewer acquisition")
    current = preflight_images(config, agent)
    if images.capability != current:
        raise ValueError("image capability or invocation identity changed")
    binding = images.binding
    if (binding.seat != current.seat or binding.provider != current.provider
            or binding.configured_model != current.model or binding.effort != current.effort
            or binding.executable_digest != current.executable_digest
            or binding.contract_id != current.contract_id or binding.scope != "visible-defects"):
        raise ValueError("image binding does not match selected seat")
    if tuple((entry.path, entry.digest) for entry in images.entries) != binding.inventory:
        raise ValueError("image inventory binding changed")
    for index in range(len(images.entries)):
        images.read(index)
    return current.invocation_view
