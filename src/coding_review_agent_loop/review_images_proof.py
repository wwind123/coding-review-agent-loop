"""Bounded CLI-origin proof parsing, separate from generated response channels."""
from __future__ import annotations

import base64
import datetime
import json
import os
from pathlib import Path
import stat

from .review_images_transport import ImageDeliveryProof, digest

MAX_STREAM_BYTES = 96 * 1024**2


def records(raw):
    if len(raw.encode()) > MAX_STREAM_BYTES:
        raise ValueError("image CLI stream exceeds contract limit")
    try:
        result = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except (ValueError, TypeError):
        raise ValueError("malformed image CLI stream") from None
    if not result or any(not isinstance(event, dict) for event in result):
        raise ValueError("malformed image CLI envelope")
    return result


def image_digest(value):
    try:
        return digest(base64.b64decode(value, validate=True))
    except (ValueError, TypeError):
        raise ValueError("malformed CLI image block") from None


def reported_models(event):
    found = []
    if isinstance(event, dict):
        if "canonicalModel" in event:
            found.append(event["canonicalModel"])
        if "model" in event:
            model = event["model"]
            if isinstance(model, dict) and "model" in model:
                pass  # Codex world-state model object; recursively checked below.
            elif not isinstance(model, str) or not model:
                raise ValueError("malformed CLI model evidence")
            else:
                found.append(model)
        for key, value in event.items():
            if key in {"modelUsage", "model_usage_since_last_result"}:
                if not isinstance(value, dict):
                    raise ValueError("malformed CLI model usage")
                found.extend(value)
                for usage in value.values():
                    found.extend(reported_models(usage))
            # Inspect CLI metadata recursively, including release-specific settings.
            # Content blocks are lists and are never model evidence.
            elif isinstance(value, dict):
                found.extend(reported_models(value))
    return found


def finish(images, digests, session, turn, models, source, text):
    proof = ImageDeliveryProof(images.binding, tuple(digests), session, turn,
                               tuple(models), source)
    proof.require(images.binding)
    if not text:
        raise ValueError("image acquisition has no generated response")
    return text, proof


def claude_proof(raw, images):
    events = records(raw)
    acquisition = images.binding.acquisition_id
    init = [e for e in events if e.get("type") == "system" and e.get("subtype") == "init"]
    if len(init) != 1 or not isinstance(init[0].get("session_id"), str):
        raise ValueError("missing fresh Claude session")
    session = init[0]["session_id"]
    acks = [e for e in events if e.get("type") == "user"]
    if (len(acks) != 1 or acks[0].get("uuid") != acquisition
            or acks[0].get("session_id") != session or acks[0].get("isReplay") is not True):
        raise ValueError("missing or duplicate Claude image acknowledgement")
    blocks = acks[0].get("message", {}).get("content", [])
    digests = []
    for block in blocks:
        if block.get("type") == "image":
            source = block.get("source", {})
            if source.get("type") != "base64" or source.get("media_type") != "image/png":
                raise ValueError("Claude image substitution")
            digests.append(image_digest(source.get("data")))
    assistants = []
    completions = []
    models = []
    message_ids = set()
    for event in events:
        if event.get("type") != "user":
            if (event.get("session_id") != session
                    or event.get("user_message_uuid", acquisition) != acquisition):
                raise ValueError("uncorrelated Claude session evidence")
            models.extend(reported_models(event))
        if event.get("type") not in {"assistant", "result"}:
            continue
        message = event.get("message", {})
        message_id = message.get("id")
        event_acquisition = event.get("user_message_uuid")
        # Claude emits one assistant envelope per content block; only the
        # first block carries the acquisition UUID. Later blocks must share
        # its CLI message ID, never just the session or model label.
        correlated_chunk = (event["type"] == "assistant" and event_acquisition is None
                            and isinstance(message_id, str) and message_id in message_ids)
        if (event.get("session_id") != session
                or (event_acquisition != acquisition and not correlated_chunk)
                or event.get("parent_tool_use_id") is not None):
            raise ValueError("uncorrelated Claude image response")
        if event["type"] == "assistant":
            if isinstance(message_id, str) and message_id:
                message_ids.add(message_id)
            if message.get("role") != "assistant":
                raise ValueError("unexpected Claude response role")
            content = message.get("content", [])
            if any(b.get("type") not in {"text", "thinking", "redacted_thinking"} for b in content):
                raise ValueError("image assessment attempted additional tools")
            text = "\n".join(b["text"] for b in content
                             if b.get("type") == "text" and isinstance(b.get("text"), str))
            if text:
                assistants.append(text)
        else:
            completions.append(event)
    if (len(completions) != 1 or completions[0].get("subtype") != "success"
            or completions[0].get("is_error") is not False
            or completions[0].get("num_turns") != 1 or not assistants):
        raise ValueError("Claude image completion is uncertain")
    text = completions[0].get("result")
    if not isinstance(text, str) or text != assistants[-1]:
        raise ValueError("Claude completion differs from generated response")
    return finish(images, digests, session, acquisition, models,
                  "Claude current-turn model/modelUsage events", text)


def open_codex_records(root: Path, thread: str, *, started: float, ended: float):
    """Open only the fresh CLI-named session, with no-follow on every component."""
    # The released CLI names its directory and rollout using local time.
    dates = {datetime.datetime.fromtimestamp(t).strftime("%Y/%m/%d")
             for t in (started, ended)}
    candidates = []
    for date in sorted(dates):
        directory = root / "sessions" / date
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        handle = os.open("/", flags)
        try:
            for part in directory.absolute().parts[1:]:
                next_handle = os.open(part, flags, dir_fd=handle)
                os.close(handle)
                handle = next_handle
            names = os.listdir(handle)
            if len(names) > 4096:
                raise ValueError("Codex session directory exceeds bound")
            for name in names:
                if name.startswith("rollout-") and name.endswith(f"-{thread}.jsonl"):
                    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=handle)
                    try:
                        info = os.fstat(fd)
                        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                                or not 0 < info.st_size <= MAX_STREAM_BYTES
                                or info.st_mtime < started - 1):
                            raise ValueError("unsafe or stale Codex session artifact")
                        raw = os.pread(fd, MAX_STREAM_BYTES + 1, 0).decode()
                        candidates.append(records(raw))
                    finally:
                        os.close(fd)
        except FileNotFoundError:
            pass
        finally:
            os.close(handle)
    if len(candidates) != 1:
        raise ValueError("missing or ambiguous invocation-specific Codex session")
    return candidates[0]


def codex_proof(raw, images, *, session_root, cwd, started, ended):
    from .agents.codex import _extract_codex_thread_id
    thread = _extract_codex_thread_id(raw)
    if thread is None:
        raise ValueError("missing Codex thread identity")
    events = open_codex_records(session_root, thread, started=started, ended=ended)
    metadata = [e["payload"] for e in events if e.get("type") == "session_meta"]
    if (len(metadata) != 1 or metadata[0].get("id") != thread
            or metadata[0].get("cwd") != str(cwd)
            or metadata[0].get("cli_version") != "0.166.0"):
        raise ValueError("wrong Codex session identity")
    timestamp = datetime.datetime.fromisoformat(metadata[0]["timestamp"].replace("Z", "+00:00")).timestamp()
    if not started - 1 <= timestamp <= ended + 1:
        raise ValueError("stale Codex session creation")
    starts = [e["payload"] for e in events if e.get("type") == "event_msg"
              and e.get("payload", {}).get("type") == "task_started"]
    if len(starts) != 1 or not starts[0].get("turn_id"):
        raise ValueError("Codex image acquisition is not one fresh turn")
    turn = starts[0]["turn_id"]
    digests, models, texts, complete, inventories = [], [], [], [], []
    for event in events:
        payload = event.get("payload", {})
        # Exactly one fresh turn owns unscoped session metadata, including
        # context emitted before task_started or after task_complete. Explicit
        # foreign turn IDs never supply evidence or override current IDs.
        containers = [event, payload, *(base.get(k, {}) for base in (event, payload)
                      for k in ("turn", "turn_context", "turn_attribution"))]
        turn_ids = {c["turn_id"] for c in containers if isinstance(c, dict) and c.get("turn_id")}
        if turn in turn_ids and turn_ids != {turn}:
            raise ValueError("conflicting Codex turn attribution")
        if any(c.get("thread_id", thread) != thread for c in (event, payload)):
            raise ValueError("foreign Codex thread metadata")
        if not turn_ids or turn_ids == {turn}:
            if event.get("type") != "response_item" or payload.get("role") != "user":
                models.extend(reported_models(event))
        if event.get("type") == "response_item" and payload.get("role") == "user":
            attribution = payload.get("internal_chat_message_metadata_passthrough", {})
            content = payload.get("content", [])
            image_blocks = [b for b in content if b.get("type") == "input_image"]
            if image_blocks:
                if attribution.get("turn_id") != turn or digests:
                    raise ValueError("duplicate or foreign Codex image input")
                if not any(images.binding.acquisition_id in b.get("text", "") for b in content):
                    raise ValueError("Codex input acquisition mismatch")
                for block in image_blocks:
                    prefix = "data:image/png;base64,"
                    value = block.get("image_url", "")
                    if not value.startswith(prefix):
                        raise ValueError("Codex image downgrade or transformation")
                    digests.append(image_digest(value[len(prefix):]))
        if event.get("type") != "event_msg" or payload.get("turn_id") != turn:
            continue
        if payload.get("type") == "item_completed":
            item = payload.get("item", {})
            if item.get("type") == "UserMessage":
                inventories.append(tuple(b["path"] for b in item.get("content", [])
                                         if b.get("type") == "local_image"))
            elif item.get("type") == "AgentMessage":
                texts.append("\n".join(b["text"] for b in item.get("content", []) if b.get("type") == "Text"))
            elif item.get("type") == "Reasoning":
                pass  # Generated reasoning is neither a tool nor a response candidate.
            else:
                raise ValueError("image assessment attempted additional tools")
        elif payload.get("type") == "task_complete":
            complete.append(payload)
    if inventories != [images.paths()] or len(complete) != 1 or not texts:
        raise ValueError("Codex image delivery/completion is incomplete")
    text = texts[-1]
    if complete[0].get("last_agent_message") != text:
        raise ValueError("Codex completion differs from generated response")
    return finish(images, digests, thread, turn, models,
                  "Codex current-turn session record", text)
