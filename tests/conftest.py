from unittest.mock import patch

import pytest

from orchestrator_split_guard import install_patch_propagation

# Patches set on the orchestrator facade must keep reaching code that the
# split (#1181) moves into extracted modules; installed once per process, so
# every xdist worker gets it too.
install_patch_propagation()


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
def _no_real_github_backoff(monkeypatch):
    """The GitHub transient-retry backoff never really sleeps under test (#510).

    Tests asserting the backoff sequence patch ``_sleep`` themselves, which
    takes precedence over this fixture.
    """
    from coding_review_agent_loop import github_retry

    monkeypatch.setattr(github_retry, "_sleep", lambda _seconds: None)


@pytest.fixture(autouse=True)
def _reset_checkout_baselines():
    """The checkout-verification ledger is process-global; isolate every test (#1130)."""
    from coding_review_agent_loop.checkout_verification import reset_checkout_baselines

    reset_checkout_baselines()
    yield
    reset_checkout_baselines()


STUB_TOOL_PROVENANCE = {
    "package_path": "/stub/src/coding_review_agent_loop",
    "checkout_root": "/stub",
    "commit": "0123456789abcdef0123456789abcdef01234567",
    "dirty": False,
    "dirty_paths_sample": [],
    "error": None,
    "captured_at": "2026-01-01T00:00:00+00:00",
    "process_started_at": "2026-01-01T00:00:00+00:00",
}


@pytest.fixture(autouse=True)
def _stub_tool_provenance(monkeypatch):
    """Keep tests deterministic: no real git probe of the tool checkout (#1111)."""
    import coding_review_agent_loop.tool_provenance as tp

    monkeypatch.setattr(tp, "_PROCESS_PROVENANCE", dict(STUB_TOOL_PROVENANCE))


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
    "AGENT_LOOP_RUN_REPO",
    "AGENT_LOOP_RUN_ID",
    "AGENT_LOOP_RUN_ISSUE",
    "AGENT_LOOP_RUN_PR",
)


@pytest.fixture(autouse=True)
def _isolate_worker_budget_environment(monkeypatch, tmp_path_factory):
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
    # Reservation telemetry (#1107) must never write the real host log.
    monkeypatch.setenv(
        "AGENT_LOOP_WORKER_TELEMETRY_LOG",
        str(tmp_path_factory.mktemp("worker-telemetry") / "worker-reservations.jsonl"),
    )


@pytest.fixture(autouse=True)
def _isolate_workdir_claims(tmp_path_factory):
    """Keep checkout claims (#1127) off the real host lock root."""
    from coding_review_agent_loop import workdir_claims

    workdir_claims._set_claim_root_for_tests(tmp_path_factory.mktemp("workdir-claims"))
    yield
    workdir_claims._set_claim_root_for_tests(None)


@pytest.fixture(autouse=True)
def _isolate_store_locks(tmp_path_factory):
    """Keep shared-store locks and owner records (#1162) off the real host lock root."""
    from coding_review_agent_loop import run_worktrees

    run_worktrees._set_store_lock_root_for_tests(tmp_path_factory.mktemp("workdir-stores"))
    yield
    run_worktrees._set_store_lock_root_for_tests(None)


pytest_plugins = ["_ci_shard"]


class _InertRoundPublication:
    """Stands in for a freeze/resume hook when a fake runner models no actor surface."""

    def prepare(self, prepared):
        from coding_review_agent_loop.publication_resume import PreparedPublication

        return PreparedPublication(tuple(str(body) for body in prepared))

    def gate(self, *, published=False):
        return None


@pytest.fixture(autouse=True)
def _round_publication_needs_an_actor_surface(monkeypatch):
    """Freeze spooled bodies only for runners that model the REST actor surface (#1258).

    A frozen carrier is actor-bound with a complete baseline listing, and
    publication refuses to start without one.  The historical orchestrator fakes
    model neither, so they get an inert hook; tests that opt in with
    ``authenticated_actor`` and ``serve_rest_issue_comments`` exercise the real one.
    """
    import coding_review_agent_loop.plan_first_loop as plan_first_loop
    import coding_review_agent_loop.pr_loop as pr_loop

    real = pr_loop.round_publication

    def maybe(runner, **kwargs):
        if (
            getattr(runner, "authenticated_actor", None) is not None
            and getattr(runner, "serve_rest_issue_comments", False)
        ):
            return real(runner, **kwargs)
        return _InertRoundPublication()

    monkeypatch.setattr(pr_loop, "round_publication", maybe)
    monkeypatch.setattr(plan_first_loop, "round_publication", maybe)
