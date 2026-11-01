"""Provider-free actual-release conformance. Launched under bwrap --unshare-net.

The fake provider lives in this same network namespace. No inherited credentials,
user configuration, provider traffic or paid call is available to either CLI.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import provider as fake

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from review_images_helpers import png, manifest
from coding_review_agent_loop.review_images_config import ImageLimits
from coding_review_agent_loop.review_images_manifest import publish
from coding_review_agent_loop.review_images_transport import acquire_images
from coding_review_agent_loop.review_images_proof import claude_proof, codex_proof


def image_blocks(value, provider):
    result = []
    if isinstance(value, dict):
        if provider == "codex" and value.get("type") == "input_image":
            result.append(value["image_url"])
        elif provider == "claude" and value.get("type") == "image":
            result.append("data:image/png;base64," + value["source"]["data"])
        else:
            for child in value.values():
                result.extend(image_blocks(child, provider))
    elif isinstance(value, list):
        for child in value:
            result.extend(image_blocks(child, provider))
    return result


def main():
    provider, binary, scratch, scenario = sys.argv[1:]
    scratch = Path(scratch)
    home = scratch / "home"
    home.mkdir()
    (home / "codex").mkdir()
    fake.provider = provider
    server = fake.ThreadingHTTPServer(("127.0.0.1", 0), fake.Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    model = "gpt-6.1-sol" if provider == "codex" else "claude-sonnet-4-6"
    capability = SimpleNamespace(seat=provider, provider=provider, model=model, effort="medium",
                                 contract_id=provider + "-conformance", executable_digest="fixture")
    value = manifest()
    if scenario == "multiple":
        value["images"].append({**value["images"][0], "path": "twin.png", "id": "twin"})
    elif scenario == "single":
        value["images"][0].pop("baseline_path")
    paths = {image[k] for image in value["images"] for k in ("path", "baseline_path") if k in image}
    content = {path: png() for path in paths}
    snapshot = publish(json.dumps(value).encode(), value["head_sha"], ImageLimits(), content,
                       {path: (2, 2) for path in paths}, "config")
    images = acquire_images(snapshot, capability, head_sha=snapshot.head_sha,
                            config_digest=snapshot.config_digest, snapshot_digest=snapshot.digest)
    snapshot.close()
    url = f"http://127.0.0.1:{server.server_port}"
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "CODEX_HOME": str(home / "codex"),
           "ANTHROPIC_API_KEY": "inert-contract-key", "ANTHROPIC_BASE_URL": url,
           "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_NONESSENTIAL_TRAFFIC": "1",
           "CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY": "1", "AGENT_LOOP_WORKDIR": str(ROOT)}
    prompt = "Visual acquisition " + images.binding.acquisition_id + ". Inspect visible defects. Say VISUAL_OK."
    if provider == "codex":
        # Local, isolated configuration supplies the inert endpoint. Production
        # capability preflight refuses endpoint/passthrough overrides.
        (home / "codex" / "config.toml").write_text(
            'model_provider="fixture"\n[model_providers.fixture]\nname="fixture"\n'
            f'base_url="{url}/v1"\nwire_api="responses"\nrequires_openai_auth=false\n')
        args = [binary, "exec", "--skip-git-repo-check", "--json", "--model", model,
                "-c", 'model_reasoning_effort="medium"',
                *(f"--image={path}" for path in images.paths()), "--", "-"]
        stdin = prompt
    else:
        args = [binary, "--print", "--input-format", "stream-json", "--output-format", "stream-json",
                "--verbose", "--replay-user-messages", "--model", model, "--effort", "medium",
                "--tools", "", "--no-session-persistence", "--setting-sources", ""]
        blocks = [{"type": "text", "text": prompt}]
        blocks.extend({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                       "data": base64.b64encode(images.read(i)).decode()}}
                      for i in range(len(images.entries)))
        stdin = json.dumps({"type": "user", "session_id": "", "uuid": images.binding.acquisition_id,
                            "message": {"role": "user", "content": blocks}}) + "\n"
    try:
        started = time.time()
        result = subprocess.run(args, cwd=ROOT, input=stdin, text=True, capture_output=True,
                                timeout=40, env=env)
        ended = time.time()
        assert result.returncode == 0, result.stderr[:500]
        assert len(fake.requests) == 1, "unexpected model-request count"
        delivered = image_blocks(fake.requests[0], provider)
        actual = [hashlib.sha256(base64.b64decode(block.split(",", 1)[1], validate=True)).hexdigest()
                  for block in delivered]
        expected = [entry.digest for entry in images.entries]
        assert actual == expected, "actual model-request inventory mismatch"
        if provider == "claude":
            text, proof = claude_proof(result.stdout, images)
            replay = result.stdout
        else:
            text, proof = codex_proof(result.stdout, images, session_root=home / "codex",
                                      cwd=ROOT, started=started, ended=ended)
            replay = next((home / "codex").glob("sessions/**/*.jsonl")).read_text()
        assert text == "VISUAL_OK"
        proof.require(images.binding)
        (scratch / "replay.json").write_text(json.dumps({"stdout": result.stdout, "records": replay}))
        print(json.dumps({"provider": provider, "model": model, "delivered": actual,
                          "reported_models": proof.reported_models,
                          "model_evidence_source": proof.model_evidence_source,
                          "provider_substitution": provider == "codex",
                          "version": subprocess.check_output([binary, "--version"], env=env).decode().strip(),
                          "executable_sha256": hashlib.file_digest(open(binary, "rb"), "sha256").hexdigest()}))
    finally:
        images.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
