"""canonical_document_path membership in the architecture set (#1454)."""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_loop_helpers import (
    FakeRunner,
    make_config,
    structured_coder_followup,
    structured_issue_implementation,
)
from coding_review_agent_loop.architecture_context import architecture_set_paths, freeze_architecture_context
from coding_review_agent_loop.errors import AgentInvocationError
from coding_review_agent_loop.runner import Runner
import coding_review_agent_loop.architecture_contract as architecture_contract
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.response_validation as response_validation
import coding_review_agent_loop.validated_agent as validated_agent

SRC = Path(__file__).resolve().parents[1] / "src" / "coding_review_agent_loop"
BILLING = "docs/architecture/billing.md"
MEMBERS = frozenset({"ARCHITECTURE.md", BILLING})
SECTION = "## Previous response not accepted: canonical_document_path"


def _impact(path):
    if path is None:
        return {
            "status": "unchanged", "rationale": "No architectural contract changed.",
            "canonical_document_action": "no-change", "canonical_document_path": None,
        }
    return {
        "status": "changed",
        "rationale": "The billing component changed.",
        "affected_components": ["Billing"],
        "dependencies": [],
        "execution_data_flows": [],
        "persistence": [],
        "public_contracts": [],
        "security_boundaries": [],
        "canonical_document_action": "update",
        "canonical_document_path": path,
        "canonical_document_rationale": "The billing detail doc owns this component.",
    }


def _with_impact(rendered: str, path) -> str:
    split = rendered.index("}\n") + 1
    payload = json.loads(rendered[:split])
    payload["architecture_impact"] = _impact(path)
    return json.dumps(payload) + rendered[split:]


def _task_output(path) -> str:
    payload = {
        "schema_version": 1, "kind": "task_result", "state": "blocking",
        "outcome": "opened_pr", "summary": "Done.", "pr_number": 4,
        "architecture_impact": _impact(path),
    }
    return json.dumps(payload) + "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"


def _issue_validators():
    return architecture_contract._architecture_mode_validators(
        lambda mode: lambda text: response_validation._validate_issue_implementation_response(
            text, human_requirements=(), require_architecture_impact=True,
            architecture_status_mode=mode,
        )
    )


def _task_validators():
    return architecture_contract._architecture_mode_validators(
        lambda mode: lambda text: response_validation._require_task_implementation_result(
            text, required_architecture_impact_contract=1, architecture_status_mode=mode,
        )
    )


def _followup_validators():
    return architecture_contract._architecture_mode_validators(
        lambda mode: lambda text: pr_loop._validate_coder_followup_response(
            text, unresolved_items=(), human_requirements=(),
            required_architecture_impact_contract=1, architecture_status_mode=mode,
        )
    )


# Each coder invocation that yields a validated carrier: task-mode
# run_task_loop, issue implementation, and the PR coder follow-up.
MODES = {
    "task": (_task_output, _task_validators),
    "issue_implementation": (
        lambda path: _with_impact(structured_issue_implementation(), path), _issue_validators,
    ),
    "coder_followup": (
        lambda path: _with_impact(structured_coder_followup(), path), _followup_validators,
    ),
}


def _prompts(runner):
    return ["\n".join(cmd) for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]


def _run(runner, tmp_path, mode, hook, *, retries=2, **extra):
    _output, validators = MODES[mode]
    return validated_agent._run_validated_agent(
        runner, agent="claude", config=make_config(tmp_path, agent_max_retries=retries),
        prompt="Implement the change.",
        marker_description="structured coder result",
        require_architecture_impact_contract=True,
        architecture_document_paths=hook,
        **validators(), **extra,
    )


def _path_of(response):
    carrier = architecture_contract._unwrap_architecture_result(response.marker_value)
    return carrier.architecture_impact.canonical_document_path


@pytest.mark.parametrize("mode", sorted(MODES))
@pytest.mark.parametrize(
    "path",
    [BILLING, f"./{BILLING}", f" {BILLING} ", "ARCHITECTURE.md", "docs/./architecture/billing.md", None],
    ids=["member", "dot-prefixed", "whitespace", "root", "dot-segment", "null"],
)
def test_member_spellings_and_null_are_accepted_on_the_first_attempt(tmp_path, mode, path):
    output, _validators = MODES[mode]
    runner = FakeRunner(claude_outputs=[output(path)])
    response = _run(runner, tmp_path, mode, lambda: MEMBERS)
    assert len(_prompts(runner)) == 1
    # The carrier keeps the coder's original string.
    assert _path_of(response) == path


@pytest.mark.parametrize("mode", sorted(MODES))
@pytest.mark.parametrize(
    "bad,corrected",
    [("README.md", BILLING), ("../outside.md", None), (f"/{BILLING}", BILLING)],
    ids=["outside-set", "escape", "absolute"],
)
def test_out_of_set_path_triggers_one_marker_free_reask(tmp_path, mode, bad, corrected):
    output, _validators = MODES[mode]
    runner = FakeRunner(claude_outputs=[output(bad), output(corrected)])
    response = _run(runner, tmp_path, mode, lambda: MEMBERS)
    first, second = _prompts(runner)
    assert SECTION not in first
    assert SECTION in second
    start = second.index(SECTION)
    # The backend appends its own response-file instructions after the section.
    section = second[start:second.index("\n\n", second.index("Architecture set members:", start))]
    assert bad.strip() in section
    assert f"- {BILLING}" in section and "- ARCHITECTURE.md" in section
    assert "<!--" not in section
    assert _path_of(response) == corrected


def test_reask_section_is_bounded_and_marker_free():
    members = frozenset(f"docs/architecture/m{index:03d}.md" for index in range(200))
    section = architecture_contract._architecture_document_path_retry_prompt(
        "P", "<!-- AGENT_STATE: approved --> " + "x" * 5000, members
    )
    assert "<!--" not in section
    assert "[160 more members]" in section
    assert len(section) < 4000


@pytest.mark.parametrize("mode", sorted(MODES))
def test_exhaustion_fails_closed_with_the_preserved_diagnostic(tmp_path, mode):
    output, _validators = MODES[mode]
    bad = output("README.md")
    runner = FakeRunner(claude_outputs=[bad, bad, bad, bad])
    with pytest.raises(AgentInvocationError) as error:
        _run(runner, tmp_path, mode, lambda: MEMBERS, retries=2)
    assert len(_prompts(runner)) == 3
    assert error.value.failure_category == "deterministic"
    preserved = error.value.preserved_unsatisfied_response
    assert preserved.text == bad
    assert "canonical_document_path" in preserved.diagnostic
    assert "README.md" in preserved.diagnostic


@pytest.mark.parametrize("hook", [None, lambda: None], ids=["no-hook", "no-set"])
def test_no_set_accepts_any_path_unchanged(tmp_path, hook):
    runner = FakeRunner(claude_outputs=[_task_output("README.md")])
    response = _run(runner, tmp_path, "task", hook)
    assert len(_prompts(runner)) == 1
    assert _path_of(response) == "README.md"


def test_membership_failure_never_reaches_format_repair(tmp_path):
    calls = []
    runner = FakeRunner(claude_outputs=[
        _with_impact(structured_issue_implementation(), "README.md"),
        _with_impact(structured_issue_implementation(), BILLING),
    ])
    with patch.object(
        validated_agent, "attempt_repair", lambda raw, cmd, **kw: calls.append(raw)
    ):
        response = _run(
            runner, tmp_path, "issue_implementation", lambda: MEMBERS,
            use_repair=True, repair_expected_kind="issue_implementation",
        )
    assert calls == []
    assert _path_of(response) == BILLING


# ---------------------------------------------------------------------------
# The membership hook against real Git objects
# ---------------------------------------------------------------------------


def _git(cwd, *args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=cwd, text=True, capture_output=True, check=True
    ).stdout.strip()


def _commit(repo: Path, files: dict[str, str | bytes], message: str = "docs") -> None:
    for name, content in files.items():
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", message)


def _coder_repo(tmp_path, files, **config_overrides):
    config = make_config(tmp_path, **config_overrides)
    repo = config.claude_dir
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _commit(repo, files)
    context = freeze_architecture_context(
        Runner(), checkout=repo, repository=config.repo,
        path=config.architecture_path, read_size=config.architecture_read_size,
        policy=config.architecture_membership_policy,
    )
    from dataclasses import replace

    return replace(config, architecture_context=context), repo


def _hook(config):
    return architecture_contract._architecture_document_paths_hook(Runner(), config)


def test_a_member_created_and_linked_at_the_coder_head_is_accepted(tmp_path):
    config, repo = _coder_repo(tmp_path, {
        "ARCHITECTURE.md": f"# Root\n\n- [Billing]({BILLING})\n",
        BILLING: "# Billing\n",
    })
    assert "docs/architecture/new.md" not in architecture_set_paths(
        config.architecture_context
    )
    _commit(repo, {
        "ARCHITECTURE.md": f"# Root\n\n- [Billing]({BILLING})\n- [New](docs/architecture/new.md)\n",
        "docs/architecture/new.md": "# New\n",
    }, "new member")
    hook = _hook(config)
    assert "docs/architecture/new.md" in hook()
    runner = FakeRunner(claude_outputs=[
        _with_impact(structured_issue_implementation(), "docs/architecture/new.md"),
    ])
    response = _run(runner, tmp_path, "issue_implementation", hook)
    assert len(_prompts(runner)) == 1
    assert _path_of(response) == "docs/architecture/new.md"


def test_member_of_a_headingless_oversized_root_is_accepted(tmp_path):
    root = "Plain overview without headings. " * 20 + "\nSee [component](docs/architecture/component.md).\n"
    config, _repo = _coder_repo(
        tmp_path, {"ARCHITECTURE.md": root, "docs/architecture/component.md": "# Component\n"},
        architecture_read_size=64,
    )
    assert config.architecture_context.availability == "oversized"
    runner = FakeRunner(claude_outputs=[_task_output("docs/architecture/component.md")])
    response = _run(runner, tmp_path, "task", _hook(config))
    assert len(_prompts(runner)) == 1
    assert _path_of(response) == "docs/architecture/component.md"


def test_the_seventieth_member_of_an_over_cap_set_is_accepted(tmp_path):
    names = [f"docs/architecture/m{index:02d}.md" for index in range(1, 71)]
    files: dict[str, str | bytes] = {name: f"# Member {name}\n" for name in names}
    files["ARCHITECTURE.md"] = "# Root\n\n" + "\n".join(f"- [{name}]({name})" for name in names) + "\n"
    config, _repo = _coder_repo(tmp_path, files)
    runner = FakeRunner(claude_outputs=[_task_output(names[-1])])
    response = _run(runner, tmp_path, "task", _hook(config))
    assert len(_prompts(runner)) == 1
    assert _path_of(response) == names[-1]


def test_a_path_named_only_by_an_unused_reference_definition_is_rejected(tmp_path):
    config, _repo = _coder_repo(tmp_path, {
        "ARCHITECTURE.md": (
            f"# Root\n\nSee [billing].\n\n[billing]: {BILLING}\n"
            "[unused]: docs/architecture/unused.md\n"
        ),
        BILLING: "# Billing\n",
        "docs/architecture/unused.md": "# Unused\n",
    })
    runner = FakeRunner(claude_outputs=[
        _task_output("docs/architecture/unused.md"), _task_output(BILLING),
    ])
    response = _run(runner, tmp_path, "task", _hook(config))
    first, second = _prompts(runner)
    assert SECTION in second
    assert _path_of(response) == BILLING


@pytest.mark.parametrize(
    "files,overrides",
    [
        ({"README.md": "# Readme\n"}, {}),
        ({"ARCHITECTURE.md": b"# bad\x00 binary\n"}, {}),
        ({"ARCHITECTURE.md": b"# bad\x00" + b"x" * 200}, {"architecture_read_size": 64}),
    ],
    ids=["missing-root", "binary-root", "unscanned-oversized-root"],
)
def test_unknowable_set_skips_validation(tmp_path, files, overrides):
    config, _repo = _coder_repo(tmp_path, files, **overrides)
    hook = _hook(config)
    assert hook() is None
    runner = FakeRunner(claude_outputs=[_task_output("README.md")])
    response = _run(runner, tmp_path, "task", hook)
    assert len(_prompts(runner)) == 1
    assert _path_of(response) == "README.md"


def test_disabled_architecture_context_has_no_hook(tmp_path):
    config = make_config(tmp_path, architecture_context_enabled=False)
    assert architecture_contract._architecture_document_paths_hook(Runner(), config) is None


# ---------------------------------------------------------------------------
# Wiring: no coder invocation may omit the hook unnoticed
# ---------------------------------------------------------------------------

_CODER_VALIDATORS = (
    "_validate_issue_implementation_response",
    "_require_task_implementation_result",
    "_validate_coder_followup_response",
)


def test_every_coder_invocation_wires_the_membership_hook():
    wired = 0
    for path in sorted(SRC.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name != "_run_validated_agent":
                continue
            segment = ast.get_source_segment(source, node) or ""
            if not any(validator in segment for validator in _CODER_VALIDATORS):
                continue
            keywords = {keyword.arg for keyword in node.keywords}
            assert "architecture_document_paths" in keywords, (
                f"{path.name}:{node.lineno} invokes a coder without the membership hook"
            )
            wired += 1
    # task, direct issue, approved-plan issue implementation, PR follow-up.
    assert wired >= 4


def test_a_parenthesized_member_is_accepted_and_a_code_span_link_is_rejected(tmp_path):
    member = "docs/architecture/component(v2).md"
    config, _repo = _coder_repo(tmp_path, {
        "ARCHITECTURE.md": (
            f"# Root\n\nSee [component]({member}).\n\n"
            "Code `start\n[demo](docs/architecture/demo.md) end`.\n"
        ),
        member: "# Component\n",
        "docs/architecture/demo.md": "# Demo\n",
    })
    runner = FakeRunner(claude_outputs=[_task_output("docs/architecture/demo.md"), _task_output(member)])
    response = _run(runner, tmp_path, "task", _hook(config))
    first, second = _prompts(runner)
    assert SECTION in second
    assert _path_of(response) == member


@pytest.mark.parametrize(
    "root,member,accepted",
    [
        ("\\` [c](docs/architecture/c.md) \\`\n", "docs/architecture/c.md", True),
        ("[c](<docs/architecture/c\\(v2\\).md>)\n", "docs/architecture/c(v2).md", True),
        ('[c](docs/architecture/c.md "The \\"c\\" service")\n', "docs/architecture/c.md", True),
        ("> See [c].\n>\n> [c]: docs/architecture/c.md\n", "docs/architecture/c.md", True),
        ("\\[c](docs/architecture/c.md)\n", "docs/architecture/c.md", False),
        ("> ~~~\n> [c](docs/architecture/c.md)\n> ~~~\n", "docs/architecture/c.md", False),
        ("Para.\n\n    [c](docs/architecture/c.md)\n", "docs/architecture/c.md", False),
    ],
    ids=[
        "escaped-backticks", "angle-escaped-parens", "escaped-title-quote", "blockquote-definition",
        "escaped-bracket", "blockquote-fence", "indented-code",
    ],
)
def test_round_two_forms_drive_canonical_path_validation(tmp_path, root, member, accepted):
    config, _repo = _coder_repo(tmp_path, {"ARCHITECTURE.md": "# Root\n\n" + root, member: "# C\n"})
    outputs = [_task_output(member)] if accepted else [_task_output(member), _task_output(None)]
    runner = FakeRunner(claude_outputs=outputs)
    response = _run(runner, tmp_path, "task", _hook(config))
    prompts = _prompts(runner)
    if accepted:
        assert len(prompts) == 1 and _path_of(response) == member
    else:
        assert len(prompts) == 2 and SECTION in prompts[1] and _path_of(response) is None


@pytest.mark.parametrize(
    "root,member,accepted",
    [
        ("[c](docs/architecture/c&#46;md)\n", "docs/architecture/c.md", True),
        ("[c](docs/architecture/c.md&#35;section)\n", "docs/architecture/c.md", True),
        ("[c](docs/architecture/`c`.md)\n", "docs/architecture/`c`.md", True),
        ("\\![c]\n\n[c]: docs/architecture/c.md\n", "docs/architecture/c.md", True),
        ("<pre>\n[c](docs/architecture/c.md)\n</pre>\n", "docs/architecture/c.md", False),
        ("<!-- [c](docs/architecture/c.md) -->\n", "docs/architecture/c.md", False),
    ],
    ids=["char-ref", "char-ref-fragment", "backtick-destination", "escaped-image", "html-pre", "html-comment"],
)
def test_round_three_forms_drive_canonical_path_validation(tmp_path, root, member, accepted):
    config, _repo = _coder_repo(tmp_path, {"ARCHITECTURE.md": "# Root\n\n" + root, member: "# C\n"})
    outputs = [_task_output(member)] if accepted else [_task_output(member), _task_output(None)]
    runner = FakeRunner(claude_outputs=outputs)
    response = _run(runner, tmp_path, "task", _hook(config))
    prompts = _prompts(runner)
    if accepted:
        assert len(prompts) == 1 and _path_of(response) == member
    else:
        assert len(prompts) == 2 and SECTION in prompts[1] and _path_of(response) is None


@pytest.mark.parametrize(
    "root",
    [
        "![diagram [c](docs/architecture/c.md)](diagram.png)\n",
        "![diagram [c]](diagram.png)\n\n[c]: docs/architecture/c.md\n",
    ],
    ids=["inline-link-in-image", "reference-link-in-image"],
)
def test_a_link_inside_an_image_description_is_rejected(tmp_path, root):
    # #1459 item 18: such links render only as alt text, so they name no member.
    member = "docs/architecture/c.md"
    config, _repo = _coder_repo(tmp_path, {"ARCHITECTURE.md": "# Root\n\n" + root, member: "# C\n"})
    runner = FakeRunner(claude_outputs=[_task_output(member), _task_output(None)])
    response = _run(runner, tmp_path, "task", _hook(config))
    prompts = _prompts(runner)
    assert len(prompts) == 2 and SECTION in prompts[1] and _path_of(response) is None


# ---------------------------------------------------------------------------
# Configured membership policy (#1499)
# ---------------------------------------------------------------------------

DECISIONS = "docs/architecture/decisions/billing.md"


def test_an_excluded_decision_log_is_rejected_as_canonical_document(tmp_path):
    config, _repo = _coder_repo(tmp_path, {
        "ARCHITECTURE.md": f"# Root\n\n- [Billing]({BILLING})\n- [Log]({DECISIONS})\n",
        BILLING: "# Billing\n",
        DECISIONS: "# Billing decisions\n",
    })
    assert architecture_set_paths(config.architecture_context) == MEMBERS
    hook = _hook(config)
    assert DECISIONS not in hook()
    runner = FakeRunner(claude_outputs=[_task_output(DECISIONS), _task_output(BILLING)])
    response = _run(runner, tmp_path, "task", hook)
    _first, second = _prompts(runner)
    assert SECTION in second
    assert _path_of(response) == BILLING


def test_a_custom_detail_dir_member_is_accepted_through_the_coder_head_hook(tmp_path):
    design = "docs/design/a.md"
    config, repo = _coder_repo(
        tmp_path,
        {"ARCHITECTURE.md": f"# Root\n\n- [A]({design})\n", design: "# A\n"},
        architecture_detail_dirs=("docs/design",),
    )
    assert architecture_set_paths(config.architecture_context) == {"ARCHITECTURE.md", design}
    _commit(repo, {
        "ARCHITECTURE.md": f"# Root\n\n- [A]({design})\n- [B](docs/design/b.md)\n",
        "docs/design/b.md": "# B\n",
    }, "new member")
    # The coder-HEAD acquisition uses the configured policy, not the default.
    hook = _hook(config)
    assert hook() == {"ARCHITECTURE.md", design, "docs/design/b.md"}
    runner = FakeRunner(claude_outputs=[_task_output("docs/design/b.md")])
    response = _run(runner, tmp_path, "task", hook)
    assert len(_prompts(runner)) == 1
    assert _path_of(response) == "docs/design/b.md"
    # The same repository under the default policy has no design members.
    default_config, _ = _coder_repo(tmp_path / "default", {
        "ARCHITECTURE.md": f"# Root\n\n- [A]({design})\n", design: "# A\n",
    })
    assert architecture_set_paths(default_config.architecture_context) == frozenset({"ARCHITECTURE.md"})
