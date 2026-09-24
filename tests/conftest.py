from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _no_real_repair():
    """Prevent attempt_repair from calling the real Gemini CLI in all tests.

    Tests that explicitly test repair behaviour patch the orchestrator-level
    import themselves, which takes precedence over this fixture.  Unit tests
    for attempt_repair itself patch subprocess.run directly, so they are
    unaffected here.
    """
    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=None):
        yield


@pytest.fixture(autouse=True)
def _agent_commands_available(monkeypatch):
    """Keep config tests independent of agent CLIs installed on the test host."""
    import coding_review_agent_loop.config as config_module

    real_which = config_module.shutil.which

    def which(command, *args, **kwargs):
        resolved = real_which(command, *args, **kwargs)
        if resolved is not None:
            return resolved
        if command in {"claude", "codex", "gemini", "agy"}:
            return f"/mock/bin/{command}"
        return None

    monkeypatch.setattr(config_module.shutil, "which", which)


_WORKER_BUDGET_ISOLATED_ENV = (
    "AGENT_LOOP_INVOCATION_ID",
    "AGENT_LOOP_TEST_WORKERS",
    "AGENT_LOOP_TEST_WORKER_ENFORCEMENT",
    "AGENT_LOOP_WORKER_CAP_SPEC",
    "AGENT_LOOP_WORKER_CAP_NESTED",
    "AGENT_LOOP_TEST_BROKER_ENDPOINT",
    "AGENT_LOOP_TEST_BROKER_CAPABILITY",
    "AGENT_LOOP_TEST_BROKER_PROTOCOL",
)


@pytest.fixture(autouse=True)
def _isolate_worker_budget_environment(monkeypatch):
    """Never inherit an outer agent-loop invocation or worker budget (#848).

    When this suite runs through ``agent-loop run-tests`` inside a coder
    scope, in-process ``run-tests`` calls and child interpreters must key
    their worker-budget lock on their own cwd instead of contending with the
    outer command.  Tests that need these variables set them explicitly.
    """
    for name in _WORKER_BUDGET_ISOLATED_ENV:
        monkeypatch.delenv(name, raising=False)
    # Host-wide worker sharing (#987) would otherwise count the real loops on
    # this host; tests that exercise it opt back in explicitly.
    monkeypatch.setenv("AGENT_LOOP_TEST_WORKER_HOST_SHARING", "off")
