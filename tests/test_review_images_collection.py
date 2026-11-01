"""Opt-in image CLI tests are removed before xdist and managed sharding."""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import pytest
import conftest as repository_conftest

pytest_plugins = ["pytester"]
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("sharded", [False, True])
def test_image_cli_contract_deselection_precedes_sharding(pytester, monkeypatch, parallel, sharded):
    pytester.makeconftest("import pytest\n" + inspect.getsource(repository_conftest.pytest_addoption) + "\n" +
                         inspect.getsource(repository_conftest.pytest_collection_modifyitems) +
                         "\ndef host_specific_node_ids(*args): return []\n")
    pytester.makeini("[pytest]\nmarkers = image_cli_contract: explicit CLI conformance\n")
    pytester.makepyfile('''import pytest
def test_replay(): pass
@pytest.mark.image_cli_contract
def test_pinned_cli(): pytest.fail('default collection ran conformance')
''')
    args = ["-q"]
    if parallel:
        args.extend(["-n", "2"])
    if sharded:
        monkeypatch.setenv("CI_SHARD_INDEX", "1")
        monkeypatch.setenv("CI_SHARD_COUNT", "1")
        monkeypatch.setenv("CI_SHARD_MANIFEST", str(pytester.path / "manifest.json"))
        monkeypatch.syspath_prepend(str(ROOT / "ci/managed"))
        monkeypatch.setenv("PYTHONPATH", str(ROOT / "ci/managed") + os.pathsep + os.environ.get("PYTHONPATH", ""))
        args.extend(["-p", "ci_shard_plugin"])
    result = pytester.runpytest_subprocess(*args, timeout=30)
    result.assert_outcomes(passed=1)
    if sharded:
        value = json.loads((pytester.path / "manifest.json").read_text())
        assert "test_pinned_cli" not in json.dumps(value)


def test_explicit_contract_option_runs_and_fails_missing_prerequisites(pytester):
    pytester.makeconftest("import pytest\n" + inspect.getsource(repository_conftest.pytest_addoption) + "\n" +
                         inspect.getsource(repository_conftest.pytest_collection_modifyitems) +
                         "\ndef host_specific_node_ids(*args): return []\n")
    pytester.makeini("[pytest]\nmarkers = image_cli_contract: explicit CLI conformance\n")
    pytester.makepyfile('''import pytest
@pytest.mark.image_cli_contract
def test_pinned_cli(): assert False, 'pinned CLI prerequisite missing'
''')
    result = pytester.runpytest("--run-image-cli-contract", "-m", "image_cli_contract", "-q")
    result.assert_outcomes(failed=1)


def test_visible_prompt_is_bounded_pure_and_has_no_clearance_authority():
    from test_review_images_transport import inputs
    from coding_review_agent_loop.review_images_prompts import build_visual_prompt
    snapshot, images = inputs()
    try:
        prompt = build_visual_prompt(images, "untrusted diff" * 1000, diff_bytes=20)
        assert "Diff truncated: true" in prompt
        assert "visible defects" in prompt and "untrusted data" in prompt
        assert "cannot approve code or grant screenshot clearance" in prompt
        assert all(entry.digest in prompt and entry.path in prompt for entry in images.entries)
        assert images.binding.head_sha in prompt
        assert len(prompt) < 4000
    finally:
        images.close(); snapshot.close()
