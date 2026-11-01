"""Sealed ownership, CLI proof provenance and explicit real-release conformance."""
from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from review_images_helpers import manifest, png
from coding_review_agent_loop.review_images_config import ImageLimits
from coding_review_agent_loop.review_images_manifest import publish
from coding_review_agent_loop.review_images_transport import acquire_images, ImageDeliveryProof
from coding_review_agent_loop.review_images_proof import claude_proof

ROOT = Path(__file__).resolve().parents[1]


def inputs(provider="claude", *, multiple=True):
    value = manifest()
    if multiple:
        value["images"].append({**value["images"][0], "path": "twin.png", "id": "twin"})
    paths = {image[key] for image in value["images"] for key in ("path", "baseline_path") if key in image}
    snapshot = publish(json.dumps(value).encode(), value["head_sha"], ImageLimits(),
                       {path: png() for path in paths}, {path: (2, 2) for path in paths}, "config")
    cap = SimpleNamespace(seat=provider, provider=provider,
                          model="claude-sonnet-4-6" if provider == "claude" else "gpt-6.1-sol",
                          effort="medium", contract_id="test", executable_digest="test")
    acquired = acquire_images(snapshot, cap, head_sha=snapshot.head_sha,
                              config_digest=snapshot.config_digest, snapshot_digest=snapshot.digest)
    return snapshot, acquired


def claude_events(images, *, models=None, content="VISUAL_OK"):
    binding = images.binding
    models = [binding.configured_model] if models is None else models
    blocks = [{"type": "text", "text": "replayed prompt: overload retry {schema-example}"}]
    blocks.extend({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                   "data": base64.b64encode(images.read(i)).decode()}}
                  for i in range(len(images.entries)))
    common = {"session_id": "fresh", "user_message_uuid": binding.acquisition_id}
    assistant = {"role": "assistant", "content": [{"type": "text", "text": content}]}
    if models:
        assistant["model"] = models[0]
    return [{"type": "system", "subtype": "init", "session_id": "fresh"},
            {"type": "user", "uuid": binding.acquisition_id, "session_id": "fresh", "isReplay": True,
             "message": {"role": "user", "content": blocks}},
            {"type": "assistant", **common, "message": assistant},
            {"type": "result", **common, "subtype": "success", "is_error": False, "num_turns": 1,
             "modelUsage": {m: {} for m in models}, "result": content}]


def encode(events):
    return "\n".join(json.dumps(event) for event in events)


def test_native_sealed_parent_paths_need_no_inheritance_and_survive_caller_close():
    snapshot, images = inputs("codex")
    snapshot.close()
    try:
        script = '''import os,sys,hashlib
path=sys.argv[1]
fd=os.open(path,os.O_RDONLY)
print(hashlib.sha256(os.read(fd,100000)).hexdigest())
os.close(fd)
try:
 fd=os.open(path,os.O_RDWR)
 try: os.write(fd,b'x')
 finally: os.close(fd)
except OSError: pass
else: raise AssertionError('sealed input became writable')
'''
        for path, entry in zip(images.paths(), images.entries):
            result = subprocess.run([sys.executable, "-c", script, path], cwd=ROOT,
                                    stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5)
            assert result.returncode == 0, result.stderr
            assert result.stdout.strip() == entry.digest
    finally:
        images.close()


@pytest.mark.parametrize("exit_kind", ["timeout", "interrupt"])
def test_acquisition_closes_descriptors_on_timeout_or_interrupt(exit_kind):
    snapshot, images = inputs()
    snapshot.close()
    with pytest.raises((subprocess.TimeoutExpired, KeyboardInterrupt)):
        with images:
            if exit_kind == "interrupt":
                raise KeyboardInterrupt()
            subprocess.run([sys.executable, "-c", "import time; time.sleep(2)"], cwd=ROOT,
                           stdin=subprocess.DEVNULL, capture_output=True, timeout=0.05)
    assert images._closed
    for entry in images.entries:
        with pytest.raises(OSError): os.fstat(entry.fd)


def test_runner_redacts_stdin_log_without_changing_delivery(tmp_path):
    from coding_review_agent_loop.runner import Runner
    payload = "secret-image-payload"
    checksum = hashlib.sha256(payload.encode()).hexdigest()
    script = f"import sys,hashlib; assert hashlib.sha256(sys.stdin.read().encode()).hexdigest() == '{checksum}'; print('ok')"
    log = tmp_path / "redacted.log"
    result = Runner().run_with_log([sys.executable, "-c", script], cwd=ROOT, log_path=log,
                                    label="reviewer", progress_interval_seconds=30,
                                    input_text=payload, stdin_log_summary="[redacted image stdin]",
                                    timeout_seconds=10, containment_role="reviewer")
    assert result.returncode == 0 and "ok" in result.stdout
    assert payload not in log.read_text() and "[redacted image stdin]" in log.read_text()


@pytest.mark.parametrize("provider", ["claude", "codex"])
@pytest.mark.parametrize("scenario", ["", "-reasoning"])
def test_committed_actual_cli_replay_and_dropped_equal_byte_twin(monkeypatch, provider, scenario):
    import datetime
    import re
    import time
    from coding_review_agent_loop import review_images_proof as parser
    snapshot, images = inputs(provider)
    fixture = json.loads((ROOT / f"tests/fixtures/review_images/{provider}{scenario}-replay.json").read_text())
    raw = json.dumps(fixture).replace("<CWD>", str(ROOT))
    if provider == "claude":
        acquisition = next(e["uuid"] for e in fixture["records"] if e.get("type") == "user")
    else:
        acquisition = re.search(r"Visual acquisition ([a-f0-9-]+)", raw).group(1)
        old_paths = list(dict.fromkeys(re.findall(r"/proc/\d+/fd/\d+", raw)))
        for old, new in zip(old_paths, images.paths()):
            raw = raw.replace(old, new)
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        raw = raw.replace("<TIMESTAMP>", timestamp)
    raw = raw.replace(acquisition, images.binding.acquisition_id)
    fixture = json.loads(raw)
    def parse():
        if provider == "claude":
            return parser.claude_proof(encode(fixture["records"]), images)
        monkeypatch.setattr(parser, "open_codex_records", lambda *a, **k: fixture["records"])
        return parser.codex_proof(fixture["stdout"], images, session_root=Path("/unused"),
                                  cwd=ROOT, started=time.time() - 1, ended=time.time() + 1)
    try:
        text, proof = parse()
        assert text == "VISUAL_OK"
        assert len(proof.inventory_digests) == 3
        proof.require(images.binding)
        if provider == "claude":
            next(e for e in fixture["records"] if e.get("type") == "user")["message"]["content"].pop()
        else:
            message = next(e for e in fixture["records"] if e.get("type") == "response_item"
                           and any(b.get("type") == "input_image" for b in e["payload"].get("content", [])))
            block = next(b for b in message["payload"]["content"] if b.get("type") == "input_image")
            message["payload"]["content"].remove(block)
        with pytest.raises(ValueError): parse()
    finally:
        images.close(); snapshot.close()


def test_sealed_ownership_shared_baseline_equal_bytes_and_cleanup():
    snapshot, images = inputs()
    snapshot.close()
    assert len(images.entries) == 3
    assert len({entry.digest for entry in images.entries}) == 1
    assert images.binding.references == (("main", 1, 0), ("twin", 2, 0))
    for i, entry in enumerate(images.entries):
        assert images.read(i) == png()
        with pytest.raises(OSError):
            os.write(entry.fd, b"x")
        with pytest.raises(OSError):
            os.ftruncate(entry.fd, 0)
    images.close()
    images.close()
    for entry in images.entries:
        with pytest.raises(OSError):
            os.fstat(entry.fd)


@pytest.mark.parametrize("fault", ["head", "config", "digest", "mapping", "extra", "closed", "writable", "content"])
def test_forged_snapshot_refused_and_caller_ownership_preserved(fault):
    snapshot, images = inputs()
    images.close()
    expected = dict(head_sha=snapshot.head_sha, config_digest=snapshot.config_digest, snapshot_digest=snapshot.digest)
    original = snapshot
    extra_fd = None
    if fault == "head": expected["head_sha"] = "b" * 40
    elif fault == "config": expected["config_digest"] = "other"
    elif fault == "digest": expected["snapshot_digest"] = "other"
    elif fault == "mapping": snapshot = replace(snapshot, manifest_json=snapshot.manifest_json.replace(b"twin.png", b"gone.png"))
    elif fault == "extra": snapshot = replace(snapshot, files=snapshot.files + (snapshot.files[0],))
    elif fault == "closed": snapshot.close()
    elif fault in {"writable", "content"}:
        extra_fd = os.memfd_create("unsealed")
        os.write(extra_fd, png())
        file = replace(snapshot.files[0], fd=extra_fd) if fault == "writable" else replace(snapshot.files[0], size=10)
        snapshot = replace(snapshot, files=(file, *snapshot.files[1:]))
    try:
        with pytest.raises((ValueError, OSError)):
            acquire_images(snapshot, images.capability, **expected)
    finally:
        original.close()
        if extra_fd is not None:
            os.close(extra_fd)


@pytest.mark.parametrize("models,accepted", [
    (["claude-sonnet-4-6"], True), ([], False), (["other"], False),
    (["other", "claude-sonnet-4-6"], False), (["claude-sonnet-4-6", "helper"], False)])
def test_claude_every_reported_model_required(models, accepted):
    snapshot, images = inputs()
    try:
        raw = encode(claude_events(images, models=models))
        if accepted:
            text, proof = claude_proof(raw, images)
            assert text == "VISUAL_OK" and "replayed prompt" not in text
            proof.require(images.binding)
        else:
            with pytest.raises(ValueError):
                claude_proof(raw, images)
    finally:
        images.close(); snapshot.close()


@pytest.mark.parametrize("fault", ["missing", "partial", "duplicate", "wrong-session", "wrong-acquisition",
                                   "wrong-digest", "downgrade", "malformed", "other-turn-model", "no-assistant"])
def test_clean_text_never_supplies_claude_image_proof(fault):
    snapshot, images = inputs()
    events = claude_events(images)
    try:
        if fault == "missing": events.pop(1)
        elif fault == "partial": events[1]["message"]["content"].pop()
        elif fault == "duplicate": events.insert(1, events[1])
        elif fault == "wrong-session": events[1]["session_id"] = "stale"
        elif fault == "wrong-acquisition": events[1]["uuid"] = "stale"
        elif fault == "wrong-digest": events[1]["message"]["content"][1]["source"]["data"] = base64.b64encode(b"wrong").decode()
        elif fault == "downgrade": events[1]["message"]["content"][1] = {"type": "text", "text": "[image omitted]"}
        elif fault == "other-turn-model": events[2]["user_message_uuid"] = "stale"
        elif fault == "no-assistant": events.pop(2)
        raw = encode(events) if fault != "malformed" else encode(events) + "\nnot-json BASE64"
        with pytest.raises(ValueError):
            claude_proof(raw, images)
    finally:
        images.close(); snapshot.close()


@pytest.mark.image_cli_contract
@pytest.mark.parametrize("provider", ["codex", "claude"])
@pytest.mark.parametrize("scenario", ["single", "multiple", "reasoning-messages"])
@pytest.mark.parametrize("mode", ["default", "sandboxed"])
def test_real_cli_image_request_contract(tmp_path, provider, scenario, mode):
    binary = os.environ.get("IMAGE_CONTRACT_" + provider.upper())
    assert binary and Path(binary).is_file(), f"IMAGE_CONTRACT_{provider.upper()} must name the pinned CLI"
    bwrap = shutil.which("bwrap")
    assert bwrap, "bubblewrap network isolation is required for explicit conformance"
    scratch = tmp_path / "contract"
    scratch.mkdir()
    target = scratch / "cli"
    target.touch()
    common_git = subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                         cwd=ROOT, stdin=subprocess.DEVNULL, text=True).strip()
    command = [bwrap, "--unshare-net", "--ro-bind", "/", "/", "--tmpfs", "/tmp",
               "--ro-bind", str(ROOT), str(ROOT), "--ro-bind", common_git, common_git,
               "--bind", str(scratch), str(scratch),
               "--ro-bind", str(Path(binary).resolve()), str(target), "--proc", "/proc", "--dev", "/dev",
               "--", sys.executable, str(ROOT / "tests/fixtures/review_images/harness.py"),
               provider, str(target), str(scratch), scenario, mode]
    run = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True,
                         text=True, timeout=60, env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"})
    assert run.returncode == 0, run.stdout + run.stderr
    evidence = json.loads(run.stdout)
    from coding_review_agent_loop.review_images_capabilities import CONTRACTS, EXECUTABLE_DIGESTS
    contract = CONTRACTS[(provider, evidence["version"], evidence["model"])]
    assert evidence["executable_sha256"] == EXECUTABLE_DIGESTS[contract]
    assert len(evidence["delivered"]) == (1 if scenario == "single" else 3)
    assert all(model == evidence["model"] for model in evidence["reported_models"])
    assert evidence["reported_models"]
    assert evidence["provider_substitution"] is (provider == "codex")
    print(json.dumps(evidence, sort_keys=True))
