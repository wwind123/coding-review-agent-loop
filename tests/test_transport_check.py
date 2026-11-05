"""``agent-loop transport-check`` and ``scripts/transport_smoke.sh`` (#1383)."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from coding_review_agent_loop import cli, git_transport, github, transport_check
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.git_transport import TransportFailure
from test_git_transport_network import requires_ssh, requires_sshd, ssh_origin  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE = REPO_ROOT / "scripts" / "transport_smoke.sh"
REPO = "OWNER/REPO"
SECRET = "SECRETSENTINEL"
LABELS = ("remote ref missing", "remote not found", "host-key failure", "network or timeout",
          "authentication failure", "unclassified")


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(path), *args), text=True).strip()


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Trusted HOME without an agent, and a private TMPDIR to observe cleanup."""
    tmp_path.chmod(0o700)
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    home.chmod(0o700)
    (home / ".ssh").chmod(0o700)
    scratch_tmp = tmp_path / "tmp"
    scratch_tmp.mkdir(mode=0o700)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    monkeypatch.delenv("AGENT_LOOP_GIT_SSH_IDENTITY", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(scratch_tmp))
    monkeypatch.setattr(git_transport, "_EMITTED_WARNINGS", set())
    calls: dict[str, list] = {"lookup": [], "import": []}

    def lookup(runner, *, config, cwd):
        calls["lookup"].append(config.repo)
        return "main"

    monkeypatch.setattr(github, "get_repo_default_branch", lookup)

    class Env:
        pass

    state = Env()
    state.home, state.tmp, state.calls, state.tmp_path = home, scratch_tmp, calls, tmp_path
    return state


def _patch_import(monkeypatch, env, outcome=None):
    def fake_import(destination, source_ref, tracking_ref, **kwargs):
        env.calls["import"].append((destination, source_ref, tracking_ref, kwargs))
        assert destination.is_dir() and str(destination).startswith(str(env.tmp))
        if outcome is not None:
            raise outcome
        return "0" * 40

    monkeypatch.setattr(git_transport, "import_ref", fake_import)


def _run(capfd, *args: str) -> tuple[int, str, str]:
    code = cli.main(["transport-check", *args])
    out, err = capfd.readouterr()
    return code, out, err


def _assert_clean(env, out: str, err: str, line: str) -> None:
    assert out == line + "\n"
    assert err == ""
    assert SECRET not in out
    assert list(env.tmp.iterdir()) == []


def _local_origin(tmp_path: Path) -> Path:
    bare = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    subprocess.run(("git", "init", "-q", "--bare", "-b", "main", str(bare)), check=True)
    subprocess.run(("git", "init", "-q", "-b", "main", str(seed)), check=True)
    (seed / "file.txt").write_text("fixture\n")
    _git(seed, "add", "file.txt")
    _git(seed, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture")
    _git(seed, "push", "-q", str(bare), "main")
    return bare


def _redirect_fetch(monkeypatch, bare: Path) -> list[tuple[str, ...]]:
    """Serve the trusted URL's fetch from a local bare repository.

    Only the fetch endpoint changes; init, rev-parse, the pack export, the
    confined index-pack import and the ref update all run for real.
    """
    original = git_transport._private_git
    fetched: list[tuple[str, ...]] = []

    def private_git(args, **kwargs):
        if args[0] == "fetch":
            fetched.append(args)
            args = (*args[:-2], str(bare), args[-1])
        return original(args, **kwargs)

    monkeypatch.setattr(git_transport, "_private_git", private_git)
    return fetched


# -- pass ------------------------------------------------------------------


def test_pass_prints_one_line_and_removes_scratch(env, monkeypatch, capfd):
    _patch_import(monkeypatch, env)
    code, out, err = _run(capfd, "--repo", REPO, "--base", "main")
    assert code == 0
    _assert_clean(env, out, err, "transport-check: pass")
    destination, source, tracking, kwargs = env.calls["import"][0]
    assert (source, tracking) == ("refs/heads/main", "refs/remotes/origin/main")
    assert kwargs["repo"] == REPO and kwargs["forbidden_roots"] == (destination,)
    assert env.calls["lookup"] == []


def test_pass_through_real_import_ref_into_scratch_repository(env, monkeypatch, capfd, tmp_path):
    bare = _local_origin(tmp_path)
    fetched = _redirect_fetch(monkeypatch, bare)
    code, out, err = _run(capfd, "--repo", REPO)
    assert code == 0
    _assert_clean(env, out, err, "transport-check: pass")
    assert env.calls["lookup"] == [REPO]
    assert fetched and fetched[0][-2] == "https://github.com/OWNER/REPO.git"
    assert fetched[0][-1] == "+refs/heads/main:refs/agent-loop/import"


@requires_sshd
def test_pass_end_to_end_against_local_sshd(env, ssh_origin, capfd):  # noqa: F811
    origin = ssh_origin
    (origin.home / ".ssh" / "config").write_text(
        "Host localhost\n  IdentityFile ~/.ssh/github_key\n  IdentitiesOnly yes\n"
        "  UserKnownHostsFile ~/.ssh/known_hosts_test\n"
    )
    (origin.home / ".ssh" / "config").chmod(0o600)
    code, out, err = _run(capfd, "--repo", origin.repo, "--trusted-origin-protocol", "ssh",
                          "--base", "main")
    assert (code, out, err) == (0, "transport-check: pass\n", "")
    assert origin.commands and f"-i {origin.key}" in origin.commands[-1]
    assert str(origin.key) not in out


# -- fail classes ----------------------------------------------------------


@pytest.mark.parametrize("label", LABELS)
def test_transport_failure_reports_its_label(env, monkeypatch, capfd, label):
    _patch_import(monkeypatch, env, TransportFailure(f"boom {SECRET}", label=label))
    code, out, err = _run(capfd, "--repo", REPO, "--base", "main")
    assert code == 1
    _assert_clean(env, out, err, f"transport-check: fail ({label})")


@pytest.mark.parametrize(("exc", "label"), [
    (subprocess.TimeoutExpired(["git", SECRET], 90), "local import"),
    (OSError(f"disk {SECRET}"), "local import"),
    (AgentLoopError(f"plain {SECRET}"), "local import"),
    (RuntimeError(f"bug {SECRET}"), "unclassified"),
])
def test_non_transport_exceptions_report_fixed_fallback(env, monkeypatch, capfd, exc, label):
    _patch_import(monkeypatch, env, exc)
    code, out, err = _run(capfd, "--repo", REPO, "--base", "main")
    assert code == 1
    _assert_clean(env, out, err, f"transport-check: fail ({label})")
    assert "Traceback" not in out + err


def test_real_index_pack_failure_after_fetch_is_local_import(env, monkeypatch, capfd, tmp_path):
    bare = _local_origin(tmp_path)
    _redirect_fetch(monkeypatch, bare)
    monkeypatch.setattr(git_transport, "local_command",
                        lambda *_a, **_k: (["/bin/sh", "-c", f"echo {SECRET} >&2; exit 7"], {}, ()))
    code, out, err = _run(capfd, "--repo", REPO, "--base", "main")
    assert code == 1
    _assert_clean(env, out, err, "transport-check: fail (local import)")


def test_exporter_wait_timeout_after_fetch_is_local_import(env, monkeypatch, capfd, tmp_path):
    bare = _local_origin(tmp_path)
    _redirect_fetch(monkeypatch, bare)
    real_popen = subprocess.Popen

    class Exporter(real_popen):
        def wait(self, timeout=None):
            if timeout == 90 and "pack-objects" in tuple(self.args):
                raise subprocess.TimeoutExpired(self.args, timeout)
            return super().wait(timeout=timeout)

    monkeypatch.setattr(subprocess, "Popen", Exporter)
    code, out, err = _run(capfd, "--repo", REPO, "--base", "main")
    assert code == 1
    _assert_clean(env, out, err, "transport-check: fail (local import)")


# -- HTTPS credential retry (real import_ref) ------------------------------


def _fail_fetches(monkeypatch, *failures: TransportFailure) -> list[tuple[str, ...]]:
    original = git_transport._private_git
    pending = list(failures)
    fetched: list[tuple[str, ...]] = []

    def private_git(args, **kwargs):
        if args[0] == "fetch":
            fetched.append(args)
            raise pending.pop(0)
        return original(args, **kwargs)

    monkeypatch.setattr(git_transport, "_private_git", private_git)
    return fetched


def test_https_retry_without_gh_token_reports_fetch_class(env, monkeypatch, capfd):
    fetched = _fail_fetches(monkeypatch, TransportFailure("x", label="authentication failure"))

    def no_token(*_args):
        raise AgentLoopError(f"Trusted GitHub authentication is unavailable. {SECRET}")

    monkeypatch.setattr(git_transport, "_token_from_gh", no_token)
    code, out, err = _run(capfd, "--repo", REPO, "--base", "main", "--gh-cmd", "gh")
    assert code == 1
    _assert_clean(env, out, err, "transport-check: fail (authentication failure)")
    assert len(fetched) == 1


def test_https_retry_fetch_failure_reports_its_own_class(env, monkeypatch, capfd):
    fetched = _fail_fetches(monkeypatch, TransportFailure("x", label="authentication failure"),
                            TransportFailure("y", label="remote not found"))
    monkeypatch.setattr(git_transport, "_token_from_gh", lambda *_args: SECRET)
    code, out, err = _run(capfd, "--repo", REPO, "--base", "main")
    assert code == 1
    _assert_clean(env, out, err, "transport-check: fail (remote not found)")
    assert len(fetched) == 2


def test_classify_failure_walks_cause_and_context_chain():
    def raised_while_handling(inner: BaseException, outer: BaseException) -> BaseException:
        try:
            try:
                raise inner
            except BaseException:
                raise outer
        except BaseException as exc:
            return exc

    chained = raised_while_handling(TransportFailure("x", label="host-key failure"),
                                    AgentLoopError("no token"))
    assert transport_check.classify_failure(chained) == "host-key failure"
    explicit = AgentLoopError("wrapped")
    explicit.__cause__ = TransportFailure("x", label="network or timeout")
    assert transport_check.classify_failure(explicit) == "network or timeout"
    assert transport_check.classify_failure(AgentLoopError("plain")) == "local import"
    cyclic = AgentLoopError("a")
    other = AgentLoopError("b")
    cyclic.__context__, other.__context__ = other, cyclic
    assert transport_check.classify_failure(cyclic) == "local import"
    assert transport_check.classify_failure(ValueError("v")) == "unclassified"


# -- configuration and argument errors -------------------------------------


def test_invalid_repo_is_configuration_before_any_network(env, monkeypatch, capfd):
    _patch_import(monkeypatch, env)
    code, out, err = _run(capfd, "--repo", f"not/a/valid/{SECRET}")
    assert code == 1
    _assert_clean(env, out, err, "transport-check: fail (configuration)")
    assert env.calls == {"lookup": [], "import": []}


@requires_ssh
@pytest.mark.parametrize("kind", ["missing", "group-writable", "relative"])
def test_untrusted_ssh_identity_is_configuration_before_any_network(env, monkeypatch, capfd, kind):
    _patch_import(monkeypatch, env)
    key = env.home / ".ssh" / f"key_{SECRET}"
    if kind == "group-writable":
        key.write_text("placeholder\n")
        key.chmod(0o660)
    argument = f"key_{SECRET}" if kind == "relative" else str(key)
    code, out, err = _run(capfd, "--repo", REPO, "--trusted-origin-protocol", "ssh",
                          "--git-ssh-identity", argument)
    assert code == 1
    _assert_clean(env, out, err, "transport-check: fail (configuration)")
    assert env.calls == {"lookup": [], "import": []}


@requires_ssh
def test_untrusted_identity_from_environment_is_configuration(env, monkeypatch, capfd):
    _patch_import(monkeypatch, env)
    monkeypatch.setenv("AGENT_LOOP_GIT_SSH_IDENTITY", str(env.home / ".ssh" / "absent"))
    code, out, err = _run(capfd, "--repo", REPO, "--trusted-origin-protocol", "ssh")
    assert code == 1
    _assert_clean(env, out, err, "transport-check: fail (configuration)")
    assert env.calls == {"lookup": [], "import": []}


def test_unsafe_base_is_configuration_without_fetch(env, monkeypatch, capfd):
    _patch_import(monkeypatch, env)
    code, out, err = _run(capfd, "--repo", REPO, "--base", "../main")
    assert code == 1
    _assert_clean(env, out, err, "transport-check: fail (configuration)")
    assert env.calls == {"lookup": [], "import": []}


@pytest.mark.parametrize("lookup_result", ["raise", None, "", "bad name", "-x"])
def test_default_branch_lookup_failure_is_configuration_without_fetch(
        env, monkeypatch, capfd, lookup_result):
    _patch_import(monkeypatch, env)

    def lookup(runner, *, config, cwd):
        env.calls["lookup"].append(config.repo)
        if lookup_result == "raise":
            raise AgentLoopError(f"gh failed {SECRET}")
        return lookup_result

    monkeypatch.setattr(github, "get_repo_default_branch", lookup)
    code, out, err = _run(capfd, "--repo", REPO)
    assert code == 1
    _assert_clean(env, out, err, "transport-check: fail (configuration)")
    assert env.calls == {"lookup": [REPO], "import": []}


def test_default_branch_lookup_selects_the_fetched_branch(env, monkeypatch, capfd):
    _patch_import(monkeypatch, env)
    monkeypatch.setattr(github, "get_repo_default_branch", lambda *_a, **_k: "trunk")
    code, out, err = _run(capfd, "--repo", REPO)
    assert code == 0
    _assert_clean(env, out, err, "transport-check: pass")
    assert env.calls["import"][0][1:3] == ("refs/heads/trunk", "refs/remotes/origin/trunk")


@pytest.mark.parametrize("argv", [
    [],
    ["--base", "main"],
    ["--repo", REPO, "--trusted-origin-protocol", "ftp"],
    ["--repo", REPO, f"--token={SECRET}"],
    ["--repo", REPO, SECRET],
    ["--repo"],
])
def test_argument_errors_print_only_configuration(env, monkeypatch, capfd, argv):
    _patch_import(monkeypatch, env)
    code, out, err = _run(capfd, *argv)
    assert code == 1
    _assert_clean(env, out, err, "transport-check: fail (configuration)")
    assert "usage" not in out.lower()
    assert env.calls == {"lookup": [], "import": []}


def test_help_still_prints_and_exits_zero(capfd):
    with pytest.raises(SystemExit) as raised:
        cli.main(["transport-check", "--help"])
    assert raised.value.code == 0
    out, _ = capfd.readouterr()
    assert "--repo" in out and "--trusted-origin-protocol" in out


def test_other_subcommands_keep_argparse_behaviour(capfd):
    with pytest.raises(SystemExit) as raised:
        cli.main(["containment-preflight", "--no-such-option"])
    assert raised.value.code == 2
    assert "usage" in capfd.readouterr().err
    assert "transport-check" in cli.build_parser().format_help()


@pytest.mark.parametrize("failure", ["mkdtemp", "runner", "output-isolation"])
def test_local_setup_failures_print_only_a_fixed_class(env, monkeypatch, capfd, failure):
    """No setup step runs outside a fixed-class boundary (review item-1)."""
    _patch_import(monkeypatch, env)
    original_set = git_transport._EMITTED_WARNINGS
    fds_before = sorted(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else None

    def refuse(*_args, **_kwargs):
        raise OSError(f"No space left on device: {env.tmp}/{SECRET}")

    if failure == "mkdtemp":
        monkeypatch.setattr(tempfile, "mkdtemp", refuse)
        expected = "configuration"
    elif failure == "runner":
        from coding_review_agent_loop import runner as runner_module

        monkeypatch.setattr(runner_module, "Runner", refuse)
        expected = "configuration"
    else:
        class FailingDevnull:
            def __getattr__(self, name):
                return getattr(os, name)

            @staticmethod
            def open(path, flags, *args):
                if path == os.devnull:
                    refuse()
                return os.open(path, flags, *args)

        monkeypatch.setattr(transport_check, "os", FailingDevnull())
        expected = "unclassified"
    code, out, err = _run(capfd, "--repo", REPO)
    assert code == 1
    _assert_clean(env, out, err, f"transport-check: fail ({expected})")
    assert str(env.tmp) not in out and "Traceback" not in out + err
    assert env.calls == {"lookup": [], "import": []}
    assert git_transport._EMITTED_WARNINGS is original_set
    if fds_before is not None:
        assert sorted(os.listdir("/proc/self/fd")) == fds_before


# -- output silencing and warning-state isolation --------------------------


def _refused_identity_config(home: Path) -> None:
    key = home / ".ssh" / "group_writable_key"
    key.write_text("placeholder\n")
    key.chmod(0o660)
    config = home / ".ssh" / "config"
    config.write_text("Host github.com\n  IdentityFile ~/.ssh/group_writable_key\n")
    config.chmod(0o600)


def _loop_side_call() -> None:
    git_transport._transport_env("git@github.com:OWNER/REPO.git", None)


@requires_ssh
def test_warning_seen_earlier_still_marks_each_check(env, monkeypatch, capfd):
    _refused_identity_config(env.home)
    _patch_import(monkeypatch, env)
    process_set = git_transport._EMITTED_WARNINGS
    _loop_side_call()
    assert "Ignoring ~/.ssh/config IdentityFile" in capfd.readouterr().err
    remembered = set(process_set)
    for _ in range(2):
        code, out, err = _run(capfd, "--repo", REPO, "--trusted-origin-protocol", "ssh",
                              "--base", "main")
        assert code == 0
        _assert_clean(env, out, err, "transport-check: pass [ssh-config-warning]")
        assert git_transport._EMITTED_WARNINGS is process_set
        assert process_set == remembered


@requires_ssh
def test_warning_first_seen_inside_check_still_prints_for_loop(env, monkeypatch, capfd):
    _refused_identity_config(env.home)
    _patch_import(monkeypatch, env)
    process_set = git_transport._EMITTED_WARNINGS
    code, out, err = _run(capfd, "--repo", REPO, "--trusted-origin-protocol", "ssh",
                          "--base", "main")
    assert code == 0
    _assert_clean(env, out, err, "transport-check: pass [ssh-config-warning]")
    assert git_transport._EMITTED_WARNINGS is process_set and process_set == set()
    _loop_side_call()
    assert "Ignoring ~/.ssh/config IdentityFile" in capfd.readouterr().err


def test_subprocess_and_python_output_inside_check_is_silenced(env, monkeypatch, capfd):
    def noisy_import(*_args, **_kwargs):
        print(f"stdout {SECRET}")
        print(f"stderr {SECRET}", file=__import__("sys").stderr)
        subprocess.run(("/bin/sh", "-c", f"echo child {SECRET}; echo child {SECRET} >&2"), check=True)
        raise TransportFailure(SECRET, label="network or timeout")

    monkeypatch.setattr(git_transport, "import_ref", noisy_import)
    code, out, err = _run(capfd, "--repo", REPO, "--base", "main")
    assert code == 1
    _assert_clean(env, out, err, "transport-check: fail (network or timeout)")


# -- scripts/transport_smoke.sh --------------------------------------------


_STUB_PYTHON = r"""#!/bin/bash
if [ -n "${STUB_VENV_FAIL:-}" ]; then
  echo "venv stdout noise"; echo "venv stderr noise" >&2; exit 1
fi
[ "$1" = "-m" ] && [ "$2" = "venv" ] || exit 64
dir="$3"
mkdir -p "$dir/bin"
cat > "$dir/bin/pip" <<'STUB'
#!/bin/bash
printf '%s\n' "$@" > "$STUB_PIP_ARGV"
echo "pip stdout noise"; echo "pip stderr noise" >&2
exit "${STUB_PIP_STATUS:-0}"
STUB
cat > "$dir/bin/agent-loop" <<'STUB'
#!/bin/bash
printf '%s\n' "$@" > "$STUB_CHECK_ARGV"
if [ "${STUB_CHECK_STATUS:-0}" = 0 ]; then
  echo "transport-check: pass"
else
  echo "transport-check: fail (authentication failure)"
fi
exit "${STUB_CHECK_STATUS:-0}"
STUB
chmod 755 "$dir/bin/pip" "$dir/bin/agent-loop"
"""


@pytest.fixture
def smoke(tmp_path):
    stub = tmp_path / "stub-python"
    stub.write_text(_STUB_PYTHON)
    stub.chmod(0o755)
    scratch = tmp_path / "smoke-tmp"
    scratch.mkdir()
    base_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "TMPDIR": str(scratch),
        "AGENT_LOOP_SMOKE_PYTHON": str(stub),
        "AGENT_LOOP_SMOKE_SOURCE": str(tmp_path / "pr head"),
        "STUB_PIP_ARGV": str(tmp_path / "pip-argv"),
        "STUB_CHECK_ARGV": str(tmp_path / "check-argv"),
    }

    def run(*args: str, **extra: str) -> subprocess.CompletedProcess:
        result = subprocess.run(("bash", str(SMOKE), *args), env={**base_env, **extra},
                                capture_output=True, text=True, timeout=60)
        assert list(scratch.iterdir()) == []
        return result

    run.tmp_path = tmp_path
    return run


def test_smoke_script_is_strict_and_executable():
    text = SMOKE.read_text()
    assert "set -euo pipefail" in text
    assert "trap 'rm -rf \"$venv\"' EXIT" in text
    assert os.access(SMOKE, os.X_OK)


def test_smoke_success_forwards_arguments_verbatim(smoke):
    args = ("--repo", "OWNER/REPO", "--git-ssh-identity", "/keys/with space", "--base", "main")
    result = smoke(*args)
    assert (result.returncode, result.stdout, result.stderr) == (0, "transport-check: pass\n", "")
    check_argv = (smoke.tmp_path / "check-argv").read_text().splitlines()
    assert check_argv == ["transport-check", *args]
    pip_argv = (smoke.tmp_path / "pip-argv").read_text().splitlines()
    assert pip_argv == ["install", "--quiet", str(smoke.tmp_path / "pr head")]


def test_smoke_transport_check_failure_propagates(smoke):
    result = smoke("--repo", "OWNER/REPO", STUB_CHECK_STATUS="1")
    assert result.returncode == 1
    assert result.stdout == "transport-check: fail (authentication failure)\n"
    assert result.stderr == ""


@pytest.mark.parametrize("failure", ["pip", "venv", "missing-interpreter"])
def test_smoke_build_failures_print_only_fixed_line(smoke, failure):
    extra = {"pip": {"STUB_PIP_STATUS": "1"}, "venv": {"STUB_VENV_FAIL": "1"},
             "missing-interpreter": {"AGENT_LOOP_SMOKE_PYTHON": str(smoke.tmp_path / "absent" / "python")}}
    result = smoke("--repo", "OWNER/REPO", **extra[failure])
    assert (result.returncode, result.stdout, result.stderr) == (3, "transport-smoke: build failed\n", "")
    assert not (smoke.tmp_path / "check-argv").exists()


def test_smoke_refuses_without_repo_before_creating_a_venv(smoke):
    result = smoke("--base", "main")
    assert result.returncode == 2
    assert result.stdout == ""
    assert not (smoke.tmp_path / "pip-argv").exists()


def test_smoke_accepts_repo_equals_form_and_forwards_verbatim(smoke):
    result = smoke("--repo=OWNER/REPO", "--base", "main")
    assert (result.returncode, result.stdout, result.stderr) == (0, "transport-check: pass\n", "")
    check_argv = (smoke.tmp_path / "check-argv").read_text().splitlines()
    assert check_argv == ["transport-check", "--repo=OWNER/REPO", "--base", "main"]


@pytest.mark.parametrize("args", [
    ("--repo=",),
    ("--repository=OWNER/REPO",),
    ("--base", "--repo-ish"),
])
def test_smoke_refuses_empty_or_lookalike_repo_before_creating_a_venv(smoke, args):
    result = smoke(*args)
    assert (result.returncode, result.stdout, result.stderr) == (
        2, "", "transport-smoke: --repo OWNER/REPO is required\n")
    assert not (smoke.tmp_path / "pip-argv").exists()
