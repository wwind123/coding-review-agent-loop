"""Physical release, coder shell and real receipt integration for issue #1922."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace
import venv

import pytest

from coding_review_agent_loop import agent_executable as ae
from coding_review_agent_loop.containment import default_policy
from coding_review_agent_loop.local_test_evidence import TestBrokerServer as BrokerServer
from coding_review_agent_loop.runner import Runner
import coding_review_agent_loop.local_test_evidence as evidence


@pytest.mark.parametrize("use_pty", [False, True])
@pytest.mark.parametrize("retarget,foreign", [(True, False), (False, True), (True, True)])
def test_physical_release_and_exported_command_survive_shell_mutations(tmp_path, monkeypatch, use_pty, retarget, foreign):
    release = tmp_path / "A"
    release.mkdir()
    venv.EnvBuilder(system_site_packages=True).create(release / ".venv")
    package = release / "src" / "coding_review_agent_loop"
    shutil.copytree(ae.package_path(), package, ignore=shutil.ignore_patterns("__pycache__"))
    (release / "deps").mkdir()
    (release / "deps" / "release_probe.py").write_text("RELEASE = 'A'\n")
    # Instrument the copied CLI, preserving its real main/run-tests implementation.
    cli = package / "cli.py"
    probe = ("import json as _j, os as _o, sys as _s, release_probe as _d\n"
             "from pathlib import Path as _P\n"
             "with _P(_o.environ['RUNTIME_PROBE']).open('a') as _f:\n"
             "    _f.write(_j.dumps({'executable': _s.executable, 'prefix': _s.prefix, "
             "'dependency': _d.__file__, 'release': _d.RELEASE}) + '\\n')\n")
    cli.write_text(cli.read_text().replace("from __future__ import annotations\n", "from __future__ import annotations\n" + probe, 1))
    current = tmp_path / "current"
    current.symlink_to(release, target_is_directory=True)
    replacement = tmp_path / "C"
    replacement.mkdir()
    venv.EnvBuilder(system_site_packages=True).create(replacement / ".venv")
    (replacement / "deps").mkdir()
    (replacement / "deps" / "release_probe.py").write_text("RELEASE = 'C'\n")
    paths = [str(current / "src"), str(current / "deps"), *[p for p in sys.path if p and Path(p).is_absolute()]]
    startup = ("import sys; sys.path[:0] = " + repr(paths) + "; "
               "from coding_review_agent_loop.agent_executable import install_agent_loop_launcher; "
               "print(install_agent_loop_launcher(" + repr(str(tmp_path / "launchers")) + "))")
    # install expects a Path root, not a string.
    startup = startup.replace("print(install_agent_loop_launcher(", "from pathlib import Path; print(install_agent_loop_launcher(Path(").replace("))", ")))")
    installed = subprocess.run([str(current / ".venv/bin/python"), "-c", startup],
        check=True, text=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=30)
    launcher = Path(installed.stdout.strip())
    assert str(release / ".venv/bin/python") in launcher.read_text()
    assert str(current / ".venv/bin/python") not in launcher.read_text()
    monkeypatch.setattr(ae, "agent_loop_launcher", lambda root=None: launcher)
    monkeypatch.setattr(evidence, "_orchestrator_tool_provenance", lambda: {"commit": None, "dirty": False, "package_path": str(package)})
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "test_release.py").write_text("def test_real_receipt():\n    assert 2 + 2 == 4\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True, stdin=subprocess.DEVNULL)
    hostile = tmp_path / "foreign-bin"
    hostile.mkdir()
    (hostile / "agent-loop").write_text("#!/bin/sh\necho FOREIGN_CLIENT_B\n")
    (hostile / "agent-loop").chmod(0o755)
    runtime = tmp_path / "runtime.jsonl"
    executed = []

    def execute(argv, cwd, timeout, environment, stream):
        result = subprocess.run(argv, cwd=cwd, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=timeout)
        executed.append(result)
        stream(result.stdout)
        return SimpleNamespace(outcome="passed" if result.returncode == 0 else "failed",
            returncode=result.returncode, elapsed_seconds=0.1, output_tail=result.stdout,
            wrapper_bootstrap="verified", inner_exec="started", suite_start="verified")

    server = BrokerServer(root=repo, turn_id="release-pin", execute=execute).start()
    try:
        shell = ["set -eu", 'test "$CODER_GUARD" = intact',
            'test "$(command -v python3)" = "$EXPECTED_PYTHON"']
        if foreign:
            shell += ["export PATH=" + shlex.quote(str(hostile)) + ':"$PATH"',
                      'test "$(agent-loop)" = FOREIGN_CLIENT_B']
        command = '\"$AGENT_LOOP_EXECUTABLE\" run-tests --timeout-seconds 30 -- ' + shlex.join([
            sys.executable, "-m", "pytest", "-q", "-n", "0", "-p", "no:cacheprovider", "test_release.py"])
        shell.append(command)
        if retarget:
            shell += ["rm " + shlex.quote(str(current)), "ln -s " + shlex.quote(str(replacement)) + " " + shlex.quote(str(current))]
        shell.append(command)
        result = Runner(containment_policy=default_policy(mode="off")).run_with_log(
            ["/bin/sh", "-c", "\n".join(shell)], cwd=repo, log_path=tmp_path / "shell.log",
            label="Claude", containment_role="reviewer", use_pty=use_pty,
            progress_interval_seconds=999, timeout_seconds=90,
            env={**server.environment, "AGENT_LOOP_INVOCATION_ID": "release-pin",
                 "RUNTIME_PROBE": str(runtime), "CODER_GUARD": "intact",
                 "EXPECTED_PYTHON": shutil.which("python3")})
        assert result.returncode == 0
        rows = list(server.journal)
    finally:
        server.stop()
    assert len(executed) == len(rows) == 2
    assert all("1 passed" in r.stdout for r in executed)
    assert all(r.outcome == "passed" and r.tool_identity.status == "unknown" for r in rows)
    assert all(r.tool_identity.client_package_digest == ae.package_digest(package) for r in rows)
    probes = [json.loads(line) for line in runtime.read_text().splitlines()]
    assert len(probes) == 2
    assert all(p == {"executable": str(release / ".venv/bin/python"), "prefix": str(release / ".venv"),
                     "dependency": str(release / "deps/release_probe.py"), "release": "A"} for p in probes)
