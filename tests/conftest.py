import platform
import sys
from unittest.mock import patch

import pytest


def _release_specific_interpreter_paths():
    # Only a path naming this exact patch release (a hosted toolcache path such
    # as .../Python/3.12.14/x64/bin/python) can differ between CI runners; a
    # generic /usr/bin/python3 also appears as a literal in many test ids.
    release = platform.python_version()
    return tuple(path for path in (sys.executable,) if path and release in path)


def host_specific_node_ids(node_ids, host_paths=None):
    """Return node ids that embed a release-specific interpreter path.

    Managed CI shards verify that every shard collected the same node ids, and
    shards can land on runners with different Python patch releases, so an id
    built from ``sys.executable`` splits the collection between shards.
    """
    paths = _release_specific_interpreter_paths() if host_paths is None else host_paths
    return sorted(node_id for node_id in node_ids if any(path in node_id for path in paths))


def pytest_addoption(parser):
    parser.addoption("--run-image-cli-contract", action="store_true",
                     help="Run provider-free pinned image CLI conformance (prerequisites required)")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-image-cli-contract"):
        excluded = [item for item in items if item.get_closest_marker("image_cli_contract")]
        if excluded:
            items[:] = [item for item in items if item not in excluded]
            config.hook.pytest_deselected(items=excluded)
    unstable = host_specific_node_ids(item.nodeid for item in items)
    if unstable:
        raise pytest.UsageError(
            "test ids must not embed the interpreter path; give the parametrization "
            "explicit ids: " + ", ".join(unstable[:5])
        )


@pytest.fixture(autouse=True)
def _no_real_repair():
    """Prevent attempt_repair from calling the real Gemini CLI in all tests.

    The stub sits on ``validated_agent``, the only module that binds
    ``attempt_repair`` (#1186).  Tests that explicitly test repair behaviour
    patch that binding themselves, which takes precedence over this fixture.
    Unit tests for attempt_repair itself patch subprocess.run directly, so
    they are unaffected here.
    """
    with patch("coding_review_agent_loop.validated_agent.attempt_repair", return_value=None):
        yield


@pytest.fixture
def private_response_tempdir(tmp_path, monkeypatch):
    """Give a test its own response root instead of $TMPDIR's shared one.

    Every xdist worker otherwise hardens the same
    ``$TMPDIR/coding-review-agent-loop/responses/OWNER-REPO`` tree, and one
    worker's brief mode change can make another's listing fail (#1653 CI).
    """
    import tempfile

    from coding_review_agent_loop import agent_permissions

    private = tmp_path / "private-response-tmp"
    private.mkdir()
    private.chmod(0o700)
    monkeypatch.setattr(tempfile, "tempdir", str(private))
    agent_permissions.reset_sandbox_state()
    yield private
    agent_permissions.reset_sandbox_state()


@pytest.fixture(autouse=True)
def _isolated_agy_log_dir(monkeypatch, tmp_path_factory):
    """Never list or read the host's real agy CLI log directory (#1604)."""
    from coding_review_agent_loop.agents import antigravity

    log_dir = tmp_path_factory.mktemp("agy-cli-log")
    monkeypatch.setattr(antigravity, "_antigravity_log_dir", lambda: log_dir)
    return log_dir


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


@pytest.fixture(autouse=True)
def _fake_runner_private_transport(monkeypatch):
    """Keep orchestration fakes at their command seam; real Git tests use import_ref."""
    from agent_loop_helpers import FakeRunner
    from coding_review_agent_loop import git_transport

    real_import = git_transport.import_ref

    def import_for_fake(destination, source_ref, tracking_ref, *, runner, **kwargs):
        if not isinstance(runner, FakeRunner):
            return real_import(destination, source_ref, tracking_ref, runner=runner, **kwargs)
        if source_ref.startswith("refs/pull/"):
            number = source_ref.split("/")[2]
            runner.run(("git", "fetch", "origin", f"+pull/{number}/head:{tracking_ref}"),
                       cwd=destination)
            return runner.pr_payload.get("headRefOid", runner.git_head)
        runner.run(("git", "fetch", "origin"), cwd=destination)
        return runner.git_head

    monkeypatch.setattr(git_transport, "import_ref", import_for_fake)


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
    # The run-tests client's identity stamp (#1632) uses the same stub, so a
    # client-side capture never probes git either.
    monkeypatch.setattr(
        tp, "capture_client_tool_provenance", lambda **_kwargs: dict(STUB_TOOL_PROVENANCE)
    )


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
    hooks = dict(workdir_claims._first_run_start_hooks)
    yield
    workdir_claims._set_claim_root_for_tests(None)
    # Drop hooks a test registered and keep the production order (#1609).
    workdir_claims._first_run_start_hooks.clear()
    workdir_claims._first_run_start_hooks.update(hooks)
    assert workdir_claims.live_run_count() == 0, "a test leaked a live workdir_claim_scope"


@pytest.fixture(autouse=True)
def _reset_quota_wait_state():
    """A cancelled quota wait (#1601) must not cancel a later test's waits."""
    from coding_review_agent_loop import quota_wait

    quota_wait.reset_wait_state()
    yield
    quota_wait.reset_wait_state()


@pytest.fixture(autouse=True)
def _isolate_store_locks(tmp_path_factory):
    """Keep shared-store locks and owner records (#1162) off the real host lock root."""
    from coding_review_agent_loop import run_worktrees

    run_worktrees._set_store_lock_root_for_tests(tmp_path_factory.mktemp("workdir-stores"))
    yield
    run_worktrees._set_store_lock_root_for_tests(None)


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


import os as _os
import stat as _stat

# Captured at import: a test's monkeypatched os functions are still active
# while this cleanup runs.
_REAL_LSTAT, _REAL_CHMOD, _REAL_LISTDIR = _os.lstat, _os.chmod, _os.listdir


_REAL_JOIN = _os.path.join
_REAL_EUID = _os.geteuid() if hasattr(_os, "geteuid") else None


def _directory_walk(root, *, restore):
    errors = []
    pending = [str(root)]
    while pending:
        current = pending.pop()
        try:
            info = _REAL_LSTAT(current)
        except FileNotFoundError:
            continue
        except OSError as exc:
            errors.append(f"lstat {current}: {exc}")
            continue
        if not _stat.S_ISDIR(info.st_mode):
            continue  # lstat never follows a symlink, including the selected root.
        if _REAL_EUID is not None and info.st_uid != _REAL_EUID:
            continue
        if (info.st_mode & 0o700) != 0o700:
            if restore:
                try:
                    _REAL_CHMOD(current, _stat.S_IMODE(info.st_mode) | 0o700)
                except OSError as exc:
                    errors.append(f"chmod {current}: {exc}")
            else:
                errors.append(f"owner rwx missing: {current}")
        try:
            names = _REAL_LISTDIR(current)
        except OSError as exc:
            errors.append(f"listdir {current}: {exc}")
            continue
        pending.extend(_REAL_JOIN(current, name) for name in names)
    return errors


def _restore_owner_listing(root):
    return _directory_walk(root, restore=True)


def _verify_owner_listing(root):
    return _directory_walk(root, restore=False)


@pytest.fixture(name="tmp_path")
def _tmp_path_stays_removable(tmp_path):
    # Wrapping pytest's fixture makes every tmp_path-dependent resource finish
    # before release, even resources that re-harden paths in their finalizer.
    yield tmp_path
    restoration_errors = _restore_owner_listing(tmp_path)
    verification_errors = _verify_owner_listing(tmp_path)
    if restoration_errors or verification_errors:
        pytest.fail("tmp_path cleanup failed:\n" + "\n".join(
            restoration_errors + verification_errors
        ), pytrace=False)
