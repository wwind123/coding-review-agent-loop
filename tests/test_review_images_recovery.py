"""Current-turn Codex identity and single-acquisition validated recovery."""
from __future__ import annotations

import base64
from dataclasses import replace
import datetime
import json
import os
from pathlib import Path
import time

import pytest

from agent_loop_helpers import make_config
from coding_review_agent_loop.agents.base import AgentResult
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop import review_images_proof as proof_parser
from coding_review_agent_loop.review_images_transport import ImageDeliveryProof
from coding_review_agent_loop import validated_agent
from test_review_images_transport import inputs

THREAD = "01a1964c-23b0-70d3-b7af-6401d253d683"
TURN = "current-turn"


def codex_records(images, models):
    records = [{"type": "session_meta", "payload": {"id": THREAD, "cwd": str(Path.cwd()),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "cli_version": "0.166.0"}},
               {"type": "event_msg", "payload": {"type": "task_started", "turn_id": TURN}}]
    records.extend({"type": "turn_context", "payload": {"turn_id": TURN, "model": model}} for model in models)
    records.extend([
        {"type": "response_item", "payload": {"type": "message", "role": "user",
         "internal_chat_message_metadata_passthrough": {"turn_id": TURN},
         "content": [{"type": "input_text", "text": images.binding.acquisition_id},
                     *({"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(images.read(i)).decode()}
                       for i in range(len(images.entries)))]}},
        {"type": "event_msg", "payload": {"type": "item_completed", "turn_id": TURN,
         "item": {"type": "UserMessage", "content": [{"type": "local_image", "path": path} for path in images.paths()]}}},
        {"type": "event_msg", "payload": {"type": "item_completed", "turn_id": TURN,
         "item": {"type": "AgentMessage", "content": [{"type": "Text", "text": "VISUAL_OK"}]}}},
        {"type": "event_msg", "payload": {"type": "task_complete", "turn_id": TURN, "last_agent_message": "VISUAL_OK"}},
    ])
    return records


@pytest.mark.parametrize("models,accepted", [(["gpt-6.1-sol"], True), ([], False), (["other"], False),
                                           (["other", "gpt-6.1-sol"], False)])
def test_codex_current_turn_every_model_without_configured_fallback(monkeypatch, models, accepted):
    snapshot, images = inputs("codex")
    events = codex_records(images, models)
    monkeypatch.setattr(proof_parser, "open_codex_records", lambda *a, **k: events)
    try:
        def parse():
            return proof_parser.codex_proof(json.dumps({"type": "thread.started", "thread_id": THREAD}), images,
                                            session_root=Path("/unused"), cwd=Path.cwd(),
                                            started=time.time() - 1, ended=time.time() + 1)
        if accepted:
            text, proof = parse()
            assert text == "VISUAL_OK"
            assert proof.reported_models == ("gpt-6.1-sol",)
        else:
            with pytest.raises(ValueError): parse()
    finally:
        images.close(); snapshot.close()


@pytest.mark.parametrize("fault", ["other-turn-model", "partial", "duplicate", "wrong-path", "wrong-digest",
                                   "wrong-turn", "wrong-session", "stale", "uncertain", "extra-tool"])
def test_codex_inventory_and_current_turn_provenance_fail_closed(monkeypatch, fault):
    snapshot, images = inputs("codex")
    events = codex_records(images, ["gpt-6.1-sol"])
    if fault == "other-turn-model": events[2]["payload"]["turn_id"] = "older-turn"
    elif fault == "partial": events[3]["payload"]["content"].pop()
    elif fault == "duplicate": events.insert(3, events[3])
    elif fault == "wrong-path": events[4]["payload"]["item"]["content"][0]["path"] = "/untrusted/path"
    elif fault == "wrong-digest": events[3]["payload"]["content"][1]["image_url"] = "data:image/png;base64,eA=="
    elif fault == "wrong-turn": events[3]["payload"]["internal_chat_message_metadata_passthrough"]["turn_id"] = "older-turn"
    elif fault == "wrong-session": events[0]["payload"]["id"] = "older-session"
    elif fault == "stale": events[0]["payload"]["timestamp"] = "2020-01-01T00:00:00Z"
    elif fault == "uncertain": events.pop()
    elif fault == "extra-tool": events[5]["payload"]["item"]["type"] = "ToolCall"
    monkeypatch.setattr(proof_parser, "open_codex_records", lambda *a, **k: events)
    try:
        with pytest.raises(ValueError):
            proof_parser.codex_proof(json.dumps({"type": "thread.started", "thread_id": THREAD}), images,
                                      session_root=Path("/unused"), cwd=Path.cwd(),
                                      started=time.time() - 1, ended=time.time() + 1)
    finally:
        images.close(); snapshot.close()


def test_codex_session_open_is_bounded_invocation_identified_and_no_follow(tmp_path):
    now = time.time()
    folder = tmp_path / "sessions" / datetime.datetime.fromtimestamp(now).strftime("%Y/%m/%d")
    folder.mkdir(parents=True)
    path = folder / f"rollout-current-{THREAD}.jsonl"
    path.write_text('{"type":"session_meta"}\n')
    assert proof_parser.open_codex_records(tmp_path, THREAD, started=now - 1, ended=now + 1)
    path.unlink()
    outside = tmp_path / "foreign"
    outside.write_text('{"type":"session_meta"}\n')
    path.symlink_to(outside)
    with pytest.raises(OSError):
        proof_parser.open_codex_records(tmp_path, THREAD, started=now - 1, ended=now + 1)


@pytest.mark.parametrize("returncode", [0, 1, None])
@pytest.mark.parametrize("proof_present", [False, True])
def test_all_validated_acceptance_paths_gate_original_proof_and_never_reask(tmp_path, monkeypatch, returncode, proof_present):
    snapshot, images = inputs()
    snapshot.close()
    proof = ImageDeliveryProof(images.binding, tuple(e.digest for e in images.entries), "session", "turn",
                               (images.binding.configured_model,), "CLI current-turn model")
    calls = []
    def acquire(*a, **kwargs):
        calls.append(kwargs)
        return AgentResult(text='{"visible": true}', returncode=returncode,
                           image_delivery_proof=proof if proof_present else None)
    monkeypatch.setattr(validated_agent, "run_agent_result", acquire)
    def run():
        return validated_agent._run_validated_agent(
            object(), agent="claude", config=make_config(tmp_path), prompt="image assessment",
            marker_description="visual", role="reviewer", timeout_seconds=30, image_inputs=images,
            validate=json.loads, use_repair=True, reask_on_evidence_rejection=True,
            completion_recovery=object())
    if proof_present:
        response = run()
        assert response.image_delivery_proof is proof
        assert response.marker_value == {"visible": True}
    else:
        with pytest.raises(AgentLoopError, match="proof"): run()
    assert len(calls) == 1 and images._closed


def test_format_repair_only_sees_generated_text_and_keeps_frozen_proof(tmp_path, monkeypatch):
    from coding_review_agent_loop import review_images_validation as validation
    snapshot, images = inputs()
    snapshot.close()
    proof = ImageDeliveryProof(images.binding, tuple(e.digest for e in images.entries), "session", "turn",
                               (images.binding.configured_model,), "CLI current-turn model")
    monkeypatch.setattr(validated_agent, "run_agent_result", lambda *a, **k:
                        AgentResult(text="malformed generated response", image_delivery_proof=proof))
    seen = []
    def normalize(text, *, expected_kind):
        seen.append(text)
        return '{"visible": true}'
    monkeypatch.setattr(validation, "attempt_envelope_normalization", normalize)
    response = validated_agent._run_validated_agent(
        object(), agent="claude", config=make_config(tmp_path), prompt="untrusted prompt and image bytes",
        marker_description="visual", role="reviewer", timeout_seconds=30, image_inputs=images,
        validate=json.loads, use_repair=True, repair_expected_kind="pr_review")
    assert seen == ["malformed generated response"]
    assert response.image_delivery_proof is proof and images._closed
