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
import contextlib
import tempfile

import provider as fake

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from review_images_helpers import png, manifest
from coding_review_agent_loop.review_images_config import ImageLimits
from coding_review_agent_loop.review_images_manifest import publish
from coding_review_agent_loop.review_images_transport import acquire_images
from coding_review_agent_loop.review_images_capabilities import preflight_images
from coding_review_agent_loop.review_images_adapter import run_image_adapter
from coding_review_agent_loop.runner import Runner
from agent_loop_helpers import make_config


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
    provider, binary, scratch, scenario, mode = sys.argv[1:]
    scratch = Path(scratch)
    home = scratch / "home"
    home.mkdir()
    (home / "codex").mkdir()
    fake.tool_call = scenario == "tool-call"
    fake.provider = provider
    fake.reasoning_messages = scenario == "reasoning-messages"
    server = fake.ThreadingHTTPServer(("127.0.0.1", 0), fake.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    model = "gpt-6.1-sol" if provider == "codex" else "claude-sonnet-4-6"
    value = manifest()
    if scenario != "single":
        value["images"].append({**value["images"][0], "path": "twin.png", "id": "twin"})
    elif scenario == "single":
        value["images"][0].pop("baseline_path")
    paths = {image[k] for image in value["images"] for k in ("path", "baseline_path") if k in image}
    content = {path: png() for path in paths}
    snapshot = publish(json.dumps(value).encode(), value["head_sha"], ImageLimits(), content,
                       {path: (2, 2) for path in paths}, "config")
    url = f"http://127.0.0.1:{server.server_port}"
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "CODEX_HOME": str(home / "codex"),
           "ANTHROPIC_API_KEY": "inert-contract-key", "ANTHROPIC_BASE_URL": url,
           "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_NONESSENTIAL_TRAFFIC": "1",
           "CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY": "1", "AGENT_LOOP_WORKDIR": str(ROOT),
           "TMPDIR": str(scratch / "tmp")}
    (scratch / "tmp").mkdir()
    os.environ.clear()
    os.environ.update(env)
    tempfile.tempdir = None
    if provider == "codex":
        # Local, isolated configuration supplies the inert endpoint. Production
        # capability preflight refuses endpoint/passthrough overrides.
        (home / "codex" / "config.toml").write_text(
            'model_provider="fixture"\n[model_providers.fixture]\nname="fixture"\n'
            f'base_url="{url}/v1"\nwire_api="responses"\nrequires_openai_auth=false\n')
    config = make_config(scratch, agent_permissions=mode, reviewer=(provider,),
                         repair_backend="codex", repair_models=("gpt-6.1-sol",),
                         semantic_followup_backend="codex",
                         **{f"{provider}_dir": ROOT, f"{provider}_cmd": binary,
                            f"reviewer_{provider}_model": model,
                            f"{provider}_args": (("--skip-git-repo-check",) if provider == "codex" and mode == "default"
                                                   else (("--dangerously-skip-permissions",) if mode == "default" else ()))})
    capability = preflight_images(config, provider)
    images = acquire_images(snapshot, capability, head_sha=snapshot.head_sha,
                            config_digest=snapshot.config_digest, snapshot_digest=snapshot.digest)
    snapshot.close()
    expected = [entry.digest for entry in images.entries]

    class CaptureRunner(Runner):
        def run_with_log(self, args, **kwargs):
            self.args = args
            self.env = kwargs["env"]
            self.result = super().run_with_log(args, **kwargs)
            return self.result

    runner = CaptureRunner()
    runner.configure_from_config(config)
    runner.set_containment_role("reviewer")
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = run_image_adapter(runner, config, "Inspect visible defects. Say VISUAL_OK.", images,
                                       provider=provider, role="reviewer", session_id=None,
                                       run_id=None, label=None, timeout_seconds=40, attempt_suffix=None)
        if fake.tool_call:
            assert result.image_delivery_proof is None
            assert result.returncode != 0
            assert result.raw_output == "" and result.message_text is None
            delivered = image_blocks(fake.requests[0], provider)
            assert [hashlib.sha256(base64.b64decode(b.split(",", 1)[1])).hexdigest()
                    for b in delivered] == expected
            assert len(fake.requests) == 2, "tool call must be followed by a final model turn"
            replay = next((home / "codex").glob("sessions/**/*.jsonl")).read_text()
            assert '"type":"function_call"' in replay
            assert '"type":"function_call_output"' in replay
            assert '"type":"task_complete"' in replay
            (scratch / "replay.json").write_text(json.dumps({"stdout": runner.result.stdout, "records": replay}))
            print(json.dumps({"provider": provider, "mode": mode, "tool_refused": True}))
            return
        assert result.returncode == 0, (runner.result.stderr[:500], result.text)
        assert result.image_delivery_proof is not None, result.text
        if provider == "claude":
            assert runner.args.count("--tools") == 1
            assert runner.env["CLAUDE_CODE_EFFORT_LEVEL"] == "medium"
            if mode == "sandboxed":
                from coding_review_agent_loop.agent_permissions import CLAUDE_READ_ONLY_TOOLS
                assert runner.args[runner.args.index("--tools") + 1] == CLAUDE_READ_ONLY_TOOLS
            else:
                assert "--dangerously-skip-permissions" in runner.args
        else:
            assert runner.args[runner.args.index("--cd") + 1] == str(ROOT)
            assert "--output-last-message" in runner.args
            if mode == "sandboxed":
                assert runner.args[runner.args.index("--sandbox") + 1] == "read-only"
        assert len(fake.requests) == 1, "unexpected model-request count"
        delivered = image_blocks(fake.requests[0], provider)
        actual = [hashlib.sha256(base64.b64decode(block.split(",", 1)[1], validate=True)).hexdigest()
                  for block in delivered]
        assert actual == expected, "actual model-request inventory mismatch"
        proof = result.image_delivery_proof
        assert result.text == "VISUAL_OK"
        proof.require(images.binding)
        replay = (runner.result.stdout if provider == "claude" else
                  next((home / "codex").glob("sessions/**/*.jsonl")).read_text())
        (scratch / "replay.json").write_text(json.dumps({"stdout": runner.result.stdout, "records": replay}))
        print(json.dumps({"provider": provider, "mode": mode, "model": model, "delivered": actual,
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
