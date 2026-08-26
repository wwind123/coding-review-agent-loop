from agent_loop_helpers import *  # noqa: F403


def test_workdir_guard_rejects_outside_home_path(tmp_path):
    assigned = tmp_path / "claude" / "repo"
    assigned.mkdir(parents=True)

    with pytest.raises(AgentLoopError, match="outside the assigned checkout"):
        validate_test_commands_within_workdir(
            ("cd ~/llm-dialectic && python -m pytest",),
            assigned_workdir=assigned,
        )

def test_workdir_guard_rejects_outside_absolute_cwd(tmp_path):
    assigned = tmp_path / "claude" / "repo"
    assigned.mkdir(parents=True)

    with pytest.raises(AgentLoopError, match="outside the assigned checkout"):
        validate_test_commands_within_workdir(
            ("cd /outside && python -m pytest",),
            assigned_workdir=assigned,
        )

def test_workdir_guard_rejects_windows_path_with_clear_message(tmp_path):
    assigned = tmp_path / "claude" / "repo"
    assigned.mkdir(parents=True)

    with pytest.raises(
        AgentLoopError,
        match="cannot be validated against the assigned Unix checkout",
    ):
        validate_test_commands_within_workdir(
            (r"cd C:\Users\dev\repo && python -m pytest",),
            assigned_workdir=assigned,
        )

def test_workdir_guard_accepts_assigned_absolute_path(tmp_path):
    assigned = tmp_path / "claude" / "repo"
    tests_dir = assigned / "tests"
    tests_dir.mkdir(parents=True)

    validate_test_commands_within_workdir(
        (f"cd {assigned} && python -m pytest {tests_dir}",),
        assigned_workdir=assigned,
    )

def test_workdir_guard_accepts_absolute_test_path_inside_checkout(tmp_path):
    assigned = tmp_path / "claude" / "repo"
    test_file = assigned / "tests" / "test_agent_loop.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")

    validate_test_commands_within_workdir(
        (f"python -m pytest {test_file}",),
        assigned_workdir=assigned,
    )

def test_workdir_guard_ignores_environment_prose_in_tests_line(tmp_path):
    assigned = tmp_path / "claude" / "repo"
    assigned.mkdir(parents=True)
    text = (
        "Tests: `python -m pytest` run from a fresh `pip install -e '.[dev]'` venv "
        "(matching CI exactly) — 1332 passed, 0 failed."
    )

    validate_response_tests_within_workdir(text, assigned_workdir=assigned)

def test_workdir_guard_ignores_url_like_prose_in_tests_line(tmp_path):
    assigned = tmp_path / "claude" / "repo"
    assigned.mkdir(parents=True)
    text = (
        "Tests: `.venv/bin/python -m pytest tests/test_server.py tests/test_security.py "
        "tests/test_pricing.py tests/test_summarizer.py tests/test_language.py "
        "tests/test_providers.py tests/test_attachments.py -q` (the mandated CI command) - "
        "435 passed. Also ran `tests/test_database.py`, `tests/test_sitemap.py`, and the new "
        "`tests/test_routes.py` - all passed. Did not run the Playwright E2E suite "
        "(`test_debate_e2e.js`, `test_csrf_redirect_e2e.js`); those target a live "
        "`https://dev.aispar.app` session cookie and are not runnable from this sandboxed "
        "checkout - recommend running them before any prod deploy, per project policy."
    )

    validate_response_tests_within_workdir(text, assigned_workdir=assigned)

def test_workdir_guard_accepts_javascript_regex_closing_script_tag(tmp_path):
    assigned = tmp_path / "codex" / "repo"
    assigned.mkdir(parents=True)

    validate_test_commands_within_workdir(
        (
            r"""node -e "const fs=require('fs'); const html=fs.readFileSync('server/static/index.html','utf8'); const scripts=[...html.matchAll(/<script(?![^>]*\\bsrc=)[^>]*>([\\s\\S]*?)<\\/script>/gi)].map(m=>m[1]); scripts.forEach((code,i)=>{ try { new Function(code); } catch(e) { console.error('script '+i+' parse failed'); throw e; } }); console.log(scripts.length+' inline scripts parsed');" (failed: naive regex matched non-code text)""",
        ),
        assigned_workdir=assigned,
    )

def test_workdir_guard_accepts_relative_test_commands(tmp_path):
    assigned = tmp_path / "claude" / "repo"
    assigned.mkdir(parents=True)

    validate_test_commands_within_workdir(
        ("python -m pytest tests/test_agent_loop.py", "make test"),
        assigned_workdir=assigned,
    )

def test_workdir_guard_extracts_tests_section_only(tmp_path):
    assigned = tmp_path / "claude" / "repo"
    assigned.mkdir(parents=True)
    text = (
        "Issue context mentioned Tests: cd ~/other && pytest.\n\n"
        "Implemented.\n"
        "Tests: python -m pytest tests/test_agent_loop.py passed.\n"
        "<!-- AGENT_PR: 77 -->"
    )

    assert extract_reported_tests_from_response(text) == (
        "python -m pytest tests/test_agent_loop.py passed.",
    )
    validate_response_tests_within_workdir(text, assigned_workdir=assigned)


# ---------------------------------------------------------------------------
# Issue #584: role-aware test-location validation (interpreter/toolchain
# paths in program position, package acquisition, and live-remote-target
# detection in both structured `tests_run` entries and `Tests:` prose).
# ---------------------------------------------------------------------------


def _assigned(tmp_path):
    assigned = tmp_path / "claude" / "repo"
    (assigned / "tests").mkdir(parents=True)
    (assigned / ".venv" / "bin").mkdir(parents=True)
    return assigned


ACCEPTED_STRUCTURED_PATH_COMMANDS = [
    "/usr/bin/python3 -m pytest tests/test_foo.py",
    "/usr/bin/python3 -m pytest tests/test_foo.py -q",
    "/other/checkout/.venv/bin/pytest tests/test_foo.py",
    "/usr/bin/env python3 -m pytest tests/test_foo.py",
    "sudo /usr/bin/python3 -m pytest tests/test_foo.py",
    "~/.pyenv/versions/3.12.1/bin/python -m pytest tests/test_foo.py",
    "/nix/store/abc123-python3-3.12.1/bin/python3 -m pytest tests/test_foo.py",
    "PYTHONPATH=/usr/lib/python3.12 pytest tests/test_foo.py",
    "tox --python=/usr/bin/python3.12",
]


@pytest.mark.parametrize("command", ACCEPTED_STRUCTURED_PATH_COMMANDS)
def test_accepts_interpreter_and_toolchain_paths(tmp_path, command):
    assigned = _assigned(tmp_path)
    validate_test_commands_within_workdir((command,), assigned_workdir=assigned)


def test_accepts_assigned_venv_bin_pytest(tmp_path):
    assigned = _assigned(tmp_path)
    command = f"{assigned}/.venv/bin/pytest tests/test_foo.py"
    validate_test_commands_within_workdir((command,), assigned_workdir=assigned)


def test_accepts_cd_then_absolute_interpreter(tmp_path):
    assigned = _assigned(tmp_path)
    command = f"cd {assigned} && /usr/bin/python3 -m pytest tests/test_foo.py"
    validate_test_commands_within_workdir((command,), assigned_workdir=assigned)


def test_accepts_windows_interpreter_absolute_path(tmp_path):
    assigned = _assigned(tmp_path)
    command = r"C:\Python311\python.exe -m pytest tests\test_foo.py"
    validate_test_commands_within_workdir((command,), assigned_workdir=assigned)


# ---------------------------------------------------------------------------
# Issue #616: operand-aware wrapper traversal.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", [
    "timeout 180s /other/checkout/.venv/bin/python -m pytest tests/test_foo.py",
    "timeout 180s /other/checkout/.venv/bin/pytest tests/test_foo.py",
    "timeout -k 10s -s TERM 180s /other/checkout/.venv/bin/python -m pytest tests/test_foo.py",
    "timeout -k10s -sTERM 180s /other/checkout/.venv/bin/python -m pytest tests/test_foo.py",
    "timeout --kill-after=10s --signal=TERM 180 /other/checkout/.venv/bin/python -m pytest tests/test_foo.py",
    "timeout -- 0.5 /other/checkout/.venv/bin/python -m pytest tests/test_foo.py",
    "timeout --foreground 1h /other/checkout/.venv/bin/python -m pytest tests/test_foo.py",
    "timeout 180s env VAR=1 /other/checkout/.venv/bin/python -m pytest tests/test_foo.py",
    "sudo -u root /usr/bin/python3 -m pytest tests/test_foo.py",
    "sudo -E /usr/bin/python3 -m pytest tests/test_foo.py",
    "nice -n 10 /usr/bin/python3 -m pytest tests/test_foo.py",
    "nice -10 /usr/bin/python3 -m pytest tests/test_foo.py",
    "stdbuf -o L /usr/bin/python3 -m pytest tests/test_foo.py",
    "env -i /usr/bin/python3 -m pytest tests/test_foo.py",
    "xargs -n 1 /usr/bin/python3 -m pytest tests/test_foo.py",
])
def test_accepts_operand_aware_wrapper_interpreters(tmp_path, command):
    validate_test_commands_within_workdir((command,), assigned_workdir=_assigned(tmp_path))


def test_accepts_gnu_timeout_windows_interpreter(tmp_path):
    validate_test_commands_within_workdir(
        (r"timeout 180s C:\Python311\python.exe -m pytest tests\test_foo.py",),
        assigned_workdir=_assigned(tmp_path),
    )


@pytest.mark.parametrize("command", [
    "timeout 180s /outside/checkout/run_tests.sh",
    "timeout 180s pytest /outside/checkout/tests/test_foo.py",
    "timeout 180s cd /outside/checkout && pytest tests/test_foo.py",
    "timeout 180s pytest --rootdir /outside/checkout tests/test_foo.py",
    "timeout 180s pytest tests/test_foo.py > /outside/results.log",
    "timeout -k /outside/value 180s /usr/bin/python3 -m pytest tests/test_foo.py",
    "timeout /outside/checkout/run_tests.sh",
])
def test_rejects_timeout_wrapped_outside_paths(tmp_path, command):
    with pytest.raises(AgentLoopError, match="outside the assigned checkout"):
        validate_test_commands_within_workdir((command,), assigned_workdir=_assigned(tmp_path))


@pytest.mark.parametrize("command", [
    "python scripts/qualify_localization.py --output-dir /tmp/localization-report",
    "python scripts/qualify_localization.py --output=/tmp/localization-report",
    "python scripts/qualify_localization.py --report-file /tmp/localization-report.json",
])
def test_accepts_explicit_report_outputs_outside_checkout(tmp_path, command):
    validate_test_commands_within_workdir((command,), assigned_workdir=_assigned(tmp_path))


@pytest.mark.parametrize("command", [
    "python /outside/checkout/scripts/qualify.py --output-dir /tmp/report",
    "python tests/test_foo.py --rootdir /outside/checkout",
])
def test_report_output_exemption_does_not_allow_outside_execution_or_workdir(tmp_path, command):
    with pytest.raises(AgentLoopError, match="outside the assigned checkout"):
        validate_test_commands_within_workdir((command,), assigned_workdir=_assigned(tmp_path))


@pytest.mark.parametrize("command", [
    "timeout curl https://live.example",
    "timeout badduration curl https://live.example",
    "timeout -k",
    "timeout 180s",
    "sudo -u",
])
def test_malformed_wrappers_do_not_grant_exemptions(tmp_path, command):
    assigned = _assigned(tmp_path)
    if "https://" in command:
        with pytest.raises(AgentLoopError, match="live remote target"):
            validate_test_commands_within_workdir((command,), assigned_workdir=assigned)
    else:
        validate_test_commands_within_workdir((command,), assigned_workdir=assigned)


def test_rejects_windows_native_timeout_option(tmp_path):
    with pytest.raises(AgentLoopError, match="cannot be validated"):
        validate_test_commands_within_workdir(
            (r"timeout /t 180 C:\Python311\python.exe -m pytest tests\test_foo.py",),
            assigned_workdir=_assigned(tmp_path),
        )


def test_timeout_wrapped_urls_in_response_are_command_classified(tmp_path):
    assigned = _assigned(tmp_path)
    for command in (
        "timeout 180s curl https://live.example",
        "timeout curl https://live.example",
        "timeout badduration curl https://live.example",
    ):
        with pytest.raises(AgentLoopError, match="live remote target"):
            validate_response_tests_within_workdir(f"Tests: `{command}`", assigned_workdir=assigned)


NARRATIVE_WRAPPED_LIVE_TARGETS = [
    "Tests: ran timeout -k 5 180s curl https://live.example",
    "Tests: ran sudo -u none curl https://live.example",
    "Tests: ran nohup --verbose curl https://live.example",
    "Tests: ran nice -n 5 curl https://live.example",
    "Tests: ran time -p curl https://live.example",
    "Tests: ran stdbuf -o L curl https://live.example",
    "Tests: ran command -p curl https://live.example",
    "Tests: ran env -u NO curl https://live.example",
    "Tests: ran xargs -n 1 curl https://live.example",
    "Tests: ran timeout --foreground 180s curl https://live.example",
    "Tests: ran sudo -E nohup nice -10 time -p stdbuf -p command -p env -i xargs -r curl https://live.example",
    "Tests: ran timeout 180s sudo -u none env -u NO curl https://live.example",
    "Tests: ran /usr/bin/timeout 180s curl https://live.example",
    "Tests: ran the curl https://live.example",
    "Tests: ran the timeout 180s curl https://live.example",
    "Tests: ran https://live.example",
    "Tests: ran the https://live.example",
    "Tests: ran FOO=1 https://live.example",
]


@pytest.mark.parametrize("text", NARRATIVE_WRAPPED_LIVE_TARGETS)
def test_rejects_non_backticked_wrapped_live_targets(tmp_path, text):
    with pytest.raises(AgentLoopError, match="live remote target"):
        validate_response_tests_within_workdir(text, assigned_workdir=_assigned(tmp_path))


@pytest.mark.parametrize("text", [
    "Tests: ran timeout curl https://live.example",
    "Tests: timeout curl https://live.example",
    "Tests: ran timeout badduration curl https://live.example",
    "Tests: ran /usr/bin/timeout badduration curl https://live.example",
    "Tests: /usr/bin/timeout badduration curl https://live.example",
    "Tests: ran timeout --kill-after curl https://live.example",
    "Tests: ran sudo -u curl https://live.example",
    "Tests: ran env -u curl https://live.example",
    "Tests: ran xargs -n curl https://live.example",
])
def test_malformed_wrapper_recovery_rejects_only_when_a_head_is_recoverable(tmp_path, text):
    with pytest.raises(AgentLoopError, match="live remote target"):
        validate_response_tests_within_workdir(text, assigned_workdir=_assigned(tmp_path))


@pytest.mark.parametrize("text", [
    "Tests: time constraints meant we relied on the report at https://ci.example/123",
    "Tests: env variables are documented at https://ci.example/123",
    "Tests: Command output was saved to https://ci.example/123",
])
def test_accepts_wrapper_word_narrative_with_unattached_url(tmp_path, text):
    validate_response_tests_within_workdir(text, assigned_workdir=_assigned(tmp_path))


def test_timeout_wrapped_package_acquisition_keeps_path_checks(tmp_path):
    assigned = _assigned(tmp_path)
    validate_test_commands_within_workdir(
        ("timeout 180s python -m pip install https://packages.example/pkg.whl",),
        assigned_workdir=assigned,
    )
    with pytest.raises(AgentLoopError, match="outside the assigned checkout"):
        validate_test_commands_within_workdir(
            ("timeout 180s python -m pip install --target /outside/site https://packages.example/pkg.whl",),
            assigned_workdir=assigned,
        )


def test_accepts_ran_absolute_interpreter_through_response_path(tmp_path):
    assigned = _assigned(tmp_path)
    text = "Tests: ran /usr/bin/python3 -m pytest tests/test_foo.py - 12 passed."
    validate_response_tests_within_workdir(text, assigned_workdir=assigned)


def test_accepts_backticked_interpreter_reproducer_through_response_path(tmp_path):
    assigned = _assigned(tmp_path)
    text = (
        "Tests: `/usr/bin/python3 -m pytest tests/test_durable_jobs.py -q` "
        "- 12 passed, run from the assigned checkout."
    )
    validate_response_tests_within_workdir(text, assigned_workdir=assigned)


REJECTED_STRUCTURED_PATH_COMMANDS = [
    "cd /outside && python -m pytest",
    "/usr/bin/python3 -m pytest /outside/checkout/tests/test_foo.py",
    "pytest /outside/bin/tests/test_foo.py",
    "pytest /other/checkout/.venv/tests/test_foo.py",
    "pytest /nix/store/abc/tests/test_foo.py",
    "pytest tests/test_foo.py | tee /outside/bin/results.log",
    "pytest tests/test_foo.py > /outside/bin/results.log",
    "/usr/local/src/other-checkout/run_tests.sh",
    "/usr/share/other-checkout/run_tests.sh",
    "/outside/.venv/tests/run_e2e.sh",
    "/nix/store/abc/other-checkout/run_tests.sh",
    "/outside/bin/run_tests.sh",
    "/outside/checkout/run_tests.sh",
    "pytest --rootdir=/outside/checkout tests/test_foo.py",
    "python -m pip install --target /outside/site-packages requests",
]


@pytest.mark.parametrize("command", REJECTED_STRUCTURED_PATH_COMMANDS)
def test_rejects_genuine_out_of_checkout_paths(tmp_path, command):
    assigned = _assigned(tmp_path)
    with pytest.raises(AgentLoopError, match="outside the assigned checkout"):
        validate_test_commands_within_workdir((command,), assigned_workdir=assigned)


def test_rejects_chained_cd_back_to_outside(tmp_path):
    assigned = _assigned(tmp_path)
    command = f"cd {assigned} && pytest tests/test_foo.py && cd /outside && pytest"
    with pytest.raises(AgentLoopError, match="outside the assigned checkout"):
        validate_test_commands_within_workdir((command,), assigned_workdir=assigned)


def test_rejects_did_not_run_outside_script_through_response_path(tmp_path):
    assigned = _assigned(tmp_path)
    text = "Tests: Did not run /outside/checkout/run_tests.sh"
    with pytest.raises(AgentLoopError, match="outside the assigned checkout"):
        validate_response_tests_within_workdir(text, assigned_workdir=assigned)


REJECTED_STRICT_COMMAND_URLS = [
    "pytest -k not --base-url https://live.example tests/test_e2e.py",
    "curl --data no https://live.example",
    "python tests/e2e.py staging https://live.example",
    "curl -sf https://dev.aispar.app/health",
    "pytest --base-url https://dev.aispar.app tests/test_e2e.py",
    "python tests/e2e.py https://live.example",
    "pytest -k smoke https://live.example",
    "python -m pytest https://live.example",
    "python -m pip https://live.example",
    "pytest --maxfail 1 https://live.example",
]


@pytest.mark.parametrize("command", REJECTED_STRICT_COMMAND_URLS)
def test_rejects_strict_command_live_targets(tmp_path, command):
    assigned = _assigned(tmp_path)
    with pytest.raises(AgentLoopError, match="live remote target"):
        validate_test_commands_within_workdir((command,), assigned_workdir=assigned)


def test_rejects_strict_command_code_span_live_target(tmp_path):
    assigned = _assigned(tmp_path)
    text = "Tests: `npx playwright test --base-url https://dev.aispar.app` - 4 passed."
    with pytest.raises(AgentLoopError, match="live remote target"):
        validate_response_tests_within_workdir(text, assigned_workdir=assigned)


@pytest.mark.parametrize("url", [
    "http://localhost:8765",
    "http://127.0.0.1:8765",
    "http://127.42.0.9:8765",
    "http://[::1]:8765",
])
def test_accepts_loopback_test_targets(tmp_path, url):
    validate_test_commands_within_workdir(
        (f"E2E_BASE={url} node tests/test_e2e.js",),
        assigned_workdir=_assigned(tmp_path),
    )


def test_accepts_loopback_target_in_nested_shell_command(tmp_path):
    command = (
        "timeout 180 bash -c 'ADMISSION_BACKEND=local DEPLOYMENT_MODE=single "
        "python3 -m server --host 127.0.0.1 --port 8765 >/tmp/server.log 2>&1 & "
        "server_pid=$!; trap \"kill $server_pid 2>/dev/null || true\" EXIT; sleep 4; "
        "E2E_BASE=http://127.0.0.1:8765 timeout 120 node tests/test_e2e.js'"
    )
    validate_test_commands_within_workdir((command,), assigned_workdir=_assigned(tmp_path))


@pytest.mark.parametrize("url", [
    "http://0.0.0.0:8765",
    "http://192.168.1.10:8765",
    "https://live.example",
])
def test_loopback_exemption_does_not_accept_other_network_targets(tmp_path, url):
    with pytest.raises(AgentLoopError, match=re.escape(url)):
        validate_test_commands_within_workdir(
            (f"E2E_BASE={url} node tests/test_e2e.js",),
            assigned_workdir=_assigned(tmp_path),
        )


def test_loopback_target_does_not_hide_remote_target_in_same_command(tmp_path):
    command = "E2E_BASE=http://127.0.0.1:8765 node tests/test_e2e.js && curl https://live.example"
    with pytest.raises(AgentLoopError, match="https://live.example"):
        validate_test_commands_within_workdir((command,), assigned_workdir=_assigned(tmp_path))


def test_rejects_backslash_ambiguous_loopback_authority(tmp_path):
    # WHATWG URL consumers (Node, browsers) treat `\` as a path separator for
    # http(s) and would resolve this to live.example, not localhost, even
    # though a strict URL parse reports "localhost" as the hostname.
    command = "E2E_BASE='http://live.example\\@localhost' node tests/test_e2e.js"
    with pytest.raises(AgentLoopError, match="live remote target"):
        validate_test_commands_within_workdir((command,), assigned_workdir=_assigned(tmp_path))


def test_rejects_live_target_concatenated_with_loopback_url(tmp_path):
    command = "E2E_BASE=http://127.0.0.1:8765/foo,http://evil.com node tests/test_e2e.js"
    with pytest.raises(AgentLoopError, match=re.escape("http://evil.com")):
        validate_test_commands_within_workdir((command,), assigned_workdir=_assigned(tmp_path))


def test_rejects_incomplete_url_scheme_token(tmp_path):
    command = "curl http:// tests/test_e2e.js"
    with pytest.raises(AgentLoopError, match="live remote target"):
        validate_test_commands_within_workdir((command,), assigned_workdir=_assigned(tmp_path))


def test_rejects_nested_scheme_hiding_behind_loopback_fragment(tmp_path):
    # The outer "http://" scheme can't supply any characters to its own URL
    # value because the next chars start a nested "https://" scheme, so a
    # naive "only judge whatever _url_values found" check would evaluate
    # just the inner "https://localhost" fragment (loopback) and miss that
    # the outer target itself is unverifiable.
    command = "E2E_BASE=http://https://localhost node tests/test_e2e.js"
    with pytest.raises(AgentLoopError, match="live remote target"):
        validate_test_commands_within_workdir((command,), assigned_workdir=_assigned(tmp_path))


REJECTED_NARRATIVE_URLS = [
    "Tests: hit https://live.example/health",
    "Tests: curled https://live.example/health",
    "Tests: ran the e2e suite against https://live.example only in staging.",
    "Tests: ran the smoke suite at https://live.example.",
    "Tests: ran python tests/e2e.py https://live.example",
    "Tests: also ran pytest tests/e2e.py https://live.example",
    "Tests: `pytest tests/unit` passed; also ran curl https://live.example/health",
    "Tests: ran curl https://live.example/health to verify.",
    "Tests: did not run unit tests but ran curl https://live.example",
    "Tests: ran curl with no proxy https://live.example",
    "Tests: ran curl skipped nothing https://live.example",
    "Tests: ran timeout 180s curl with no proxy https://live.example",
    # The attached URL sits behind an earlier, non-URL prepositional object;
    # only scanning the first preposition would read this as benign prose.
    "Tests: ran the suite against the production environment at https://live.example",
    "Tests: ran the smoke suite on staging targeting https://live.example",
    "Tests: retested the checkout suite via the shared runner against https://live.example",
]


@pytest.mark.parametrize("text", REJECTED_NARRATIVE_URLS)
def test_rejects_narrative_live_targets(tmp_path, text):
    assigned = _assigned(tmp_path)
    with pytest.raises(AgentLoopError, match="live remote target"):
        validate_response_tests_within_workdir(text, assigned_workdir=assigned)


ACCEPTED_ACQUISITION_COMMANDS = [
    "pip install git+https://github.com/org/pkg && pytest tests/test_foo.py",
    "python -m pip install https://packages.example/pkg.whl",
    "/usr/bin/python3 -m pip install https://packages.example/pkg.whl",
    "python -m pip install https://packages.example/pkg.whl && python -m pytest tests/test_foo.py",
]


@pytest.mark.parametrize("command", ACCEPTED_ACQUISITION_COMMANDS)
def test_accepts_package_acquisition_urls(tmp_path, command):
    assigned = _assigned(tmp_path)
    validate_test_commands_within_workdir((command,), assigned_workdir=assigned)


def test_accepts_package_acquisition_url_through_response_path(tmp_path):
    assigned = _assigned(tmp_path)
    text = (
        "Tests: `python -m pip install https://packages.example/pkg.whl` "
        "then `pytest tests/test_foo.py` - 12 passed."
    )
    validate_response_tests_within_workdir(text, assigned_workdir=assigned)


def test_acquisition_exclusion_is_per_clause_not_per_entry(tmp_path):
    assigned = _assigned(tmp_path)
    command = "python -m pip install https://packages.example/pkg.whl && pytest https://live.example"
    with pytest.raises(AgentLoopError, match="live remote target"):
        validate_test_commands_within_workdir((command,), assigned_workdir=assigned)


def test_module_acquisition_requires_package_manager_module(tmp_path):
    assigned = _assigned(tmp_path)
    with pytest.raises(AgentLoopError, match="live remote target"):
        validate_test_commands_within_workdir(
            ("python -m pytest https://live.example",), assigned_workdir=assigned
        )


def test_module_acquisition_requires_subcommand(tmp_path):
    assigned = _assigned(tmp_path)
    with pytest.raises(AgentLoopError, match="live remote target"):
        validate_test_commands_within_workdir(
            ("python -m pip https://live.example",), assigned_workdir=assigned
        )


ACCEPTED_NARRATIVE_TEXTS = [
    "Tests: Did not run curl https://live.example because it is live.",
    "Tests: Did not run the E2E suite; it is configured with --base-url https://live.example.",
    "Tests: pytest tests/test_foo.py -q - did not run the https://dev.aispar.app e2e suite.",
    "Tests: pytest tests/test_foo.py -q passed. Deployment notes live at https://dev.aispar.app/docs.",
    "Tests: ran pytest tests/test_foo.py -q, release notes at https://dev.aispar.app/notes.",
    "Tests: ran timeout regression checks and attached https://docs.example/run",
    "Tests: ran timeout notes and no curl https://live.example",
    "Tests: ran timeout regression checks and ran release notes about https://docs.example/run.",
    "Tests: Timeout tuning notes live in https://pkg.go.dev/net/http",
    "Tests: ran timeout tuning notes and published https://pkg.go.dev/net/http",
    # Negation still wins over the full-phrase preposition scan, both when the
    # verb itself is negated and when the negation precedes the preposition.
    "Tests: Did not run the suite against the production environment at https://live.example.",
    "Tests: ran the unit suite against the local stub and never against https://live.example.",
    "Tests: Timeout tuning did not make https://ci.example/123 flaky",
    "Tests: Timeout budgets were not enough to go https://ci.example/123",
    "Tests: Timeout tuning means no curl runs against https://ci.example/123",
]


@pytest.mark.parametrize("text", ACCEPTED_NARRATIVE_TEXTS)
def test_accepts_narrative_controls(tmp_path, text):
    assigned = _assigned(tmp_path)
    validate_response_tests_within_workdir(text, assigned_workdir=assigned)


def test_accepts_bare_backticked_url_in_negated_sentence(tmp_path):
    assigned = _assigned(tmp_path)
    text = (
        "Tests: pytest tests/test_foo.py -q - 40 passed. Did not run the Playwright "
        "E2E suite; it is only reachable at `https://dev.aispar.app` in this sandbox."
    )
    validate_response_tests_within_workdir(text, assigned_workdir=assigned)


def test_same_text_is_strict_under_structured_origin(tmp_path):
    assigned = _assigned(tmp_path)
    command = "pytest tests/test_foo.py https://live.example"
    with pytest.raises(AgentLoopError, match="live remote target"):
        validate_test_commands_within_workdir((command,), assigned_workdir=assigned)


def test_same_text_is_narrative_and_accepted_under_response_origin_when_negated(tmp_path):
    assigned = _assigned(tmp_path)
    text = "Tests: Did not run pytest tests/test_foo.py https://live.example because it is live."
    validate_response_tests_within_workdir(text, assigned_workdir=assigned)


def test_default_origin_is_structured(tmp_path):
    assigned = _assigned(tmp_path)
    with pytest.raises(AgentLoopError, match="live remote target"):
        validate_test_commands_within_workdir(
            ("pytest tests/test_foo.py https://live.example",),
            assigned_workdir=assigned,
        )


def _checkout_claim(source, *, verification_basis="checkout-inspected", status="verified", fact="fact"):
    return DiscussEvidenceClaim(fact=fact, status=status, source=source, verification_basis=verification_basis)


def test_checkout_inspected_evidence_accepts_valid_in_range_reference(tmp_path):
    (tmp_path / "src.py").write_text("line1\nline2\nline3\n", encoding="utf-8")

    validate_checkout_inspected_evidence(
        [_checkout_claim("src.py:2")], assigned_workdir=tmp_path
    )


def test_checkout_inspected_evidence_rejects_missing_file(tmp_path):
    with pytest.raises(AgentLoopError, match="not a file"):
        validate_checkout_inspected_evidence(
            [_checkout_claim("nope.py:1")], assigned_workdir=tmp_path
        )


def test_checkout_inspected_evidence_rejects_out_of_range_line(tmp_path):
    (tmp_path / "src.py").write_text("line1\nline2\n", encoding="utf-8")

    with pytest.raises(AgentLoopError, match="outside the file's range"):
        validate_checkout_inspected_evidence(
            [_checkout_claim("src.py:99")], assigned_workdir=tmp_path
        )


def test_checkout_inspected_evidence_rejects_implausibly_large_line_number_without_crashing(tmp_path):
    # Python 3.11+ raises ValueError (not AgentLoopError) converting a
    # digit string beyond its default int-conversion digit limit; the
    # parser's `\d+` source pattern is otherwise unbounded, so this must be
    # rejected as AgentLoopError before ever calling int() on it.
    (tmp_path / "src.py").write_text("line1\nline2\n", encoding="utf-8")
    huge_line = "9" * 4500

    with pytest.raises(AgentLoopError, match="implausibly large line number"):
        validate_checkout_inspected_evidence(
            [_checkout_claim(f"src.py:{huge_line}")], assigned_workdir=tmp_path
        )


def test_checkout_inspected_evidence_rejects_unreadable_file_without_crashing(tmp_path):
    # is_file() only checks the path resolves to a regular file; it does not
    # guarantee the file can actually be opened (deleted or made unreadable
    # between the check and open() -- e.g. a permission race). An OSError
    # from open() must be translated into AgentLoopError like every other
    # unresolvable reference, not left to escape and bypass the repair path.
    target = tmp_path / "src.py"
    target.write_text("line1\nline2\n", encoding="utf-8")
    target.chmod(0o000)
    try:
        with pytest.raises(AgentLoopError, match="could not be read"):
            validate_checkout_inspected_evidence(
                [_checkout_claim("src.py:1")], assigned_workdir=tmp_path
            )
    finally:
        target.chmod(0o644)


def test_checkout_inspected_evidence_rejects_line_zero(tmp_path):
    (tmp_path / "src.py").write_text("line1\n", encoding="utf-8")

    with pytest.raises(AgentLoopError, match="outside the file's range"):
        validate_checkout_inspected_evidence(
            [_checkout_claim("src.py:0")], assigned_workdir=tmp_path
        )


def test_checkout_inspected_evidence_rejects_absolute_path(tmp_path):
    with pytest.raises(AgentLoopError, match="absolute path"):
        validate_checkout_inspected_evidence(
            [_checkout_claim("/etc/passwd:1")], assigned_workdir=tmp_path
        )


def test_checkout_inspected_evidence_rejects_parent_traversal(tmp_path):
    outside = tmp_path.parent / "outside.py"
    outside.write_text("secret\n", encoding="utf-8")
    assigned = tmp_path / "repo"
    assigned.mkdir()

    with pytest.raises(AgentLoopError, match="traversal"):
        validate_checkout_inspected_evidence(
            [_checkout_claim("../outside.py:1")], assigned_workdir=assigned
        )


def test_checkout_inspected_evidence_rejects_symlink_escape(tmp_path):
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    target = outside_dir / "secret.py"
    target.write_text("secret line\n", encoding="utf-8")
    assigned = tmp_path / "repo"
    assigned.mkdir()
    link = assigned / "escape.py"
    link.symlink_to(target)

    with pytest.raises(AgentLoopError, match="outside the assigned checkout"):
        validate_checkout_inspected_evidence(
            [_checkout_claim("escape.py:1")], assigned_workdir=assigned
        )


def test_checkout_inspected_evidence_leaves_external_source_claim_alone(tmp_path):
    validate_checkout_inspected_evidence(
        [_checkout_claim("https://example.com/spec:1", verification_basis="external-source-inspected")],
        assigned_workdir=tmp_path,
    )


def test_checkout_inspected_evidence_leaves_missing_status_claim_alone(tmp_path):
    claim = DiscussEvidenceClaim(fact="fact", status="missing", source=None, verification_basis=None)

    validate_checkout_inspected_evidence([claim], assigned_workdir=tmp_path)
