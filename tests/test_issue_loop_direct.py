"""Issue-loop tests: direct issue mode without a plan (#1543)."""

import pytest
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.prompts import HUMAN_REQUIREMENTS_ADDRESSED_MARKER
from issue_loop_test_helpers import FakeRunner
from agent_loop_helpers import command_index, make_config, prior_item_dispositions


def test_issue_loop_creates_pr_then_alternates_until_codex_approval(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->",
            "Fixed review.\n<!-- AGENT_STATE: blocking -->",
        ],
        codex_outputs=[
            "Finding: bug remains.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    command_names = [cmd[:2] for cmd, _cwd in runner.commands]
    assert ["claude", "--print"] in command_names
    assert ["codex", "exec"] in command_names
    assert len(runner.comments) == 6
    assert runner.comments[-1].startswith("**Review verdict:** Approved\n\nLGTM.")
    assert list((tmp_path / "logs").glob("*-claude-attempt1.log"))
    assert list((tmp_path / "logs").glob("*-codex.log"))
    assert (tmp_path / "logs" / ".gitignore").read_text(encoding="utf-8") == "*\n!.gitignore\n"


def test_issue_loop_refuses_a_corrupted_checkout_at_the_next_turn_1127(tmp_path, monkeypatch):
    """The #1127 incident: another writer left the coder checkout on main with edited files."""
    from coding_review_agent_loop import checkout_verification as cv
    from coding_review_agent_loop.errors import CheckoutVerificationError
    from coding_review_agent_loop.runner import BinaryCommandResult

    class CorruptibleRunner(FakeRunner):
        corrupted = False
        coder_dir = None

        def run_binary(self, args, *, cwd, **kwargs):
            cmd = [str(arg) for arg in args]
            hit = self.corrupted and cwd == self.coder_dir
            if hit and cmd[1:3] == ["rev-parse", "--abbrev-ref"]:
                return BinaryCommandResult(cmd, cwd, b"main\n", b"", 0)
            if hit and cmd[1:2] == ["status"]:
                return BinaryCommandResult(
                    cmd, cwd, b" M orchestrator.py\0 M protocol.py\0", b"", 0
                )
            if cmd[1:3] == ["rev-parse", "--abbrev-ref"]:
                return BinaryCommandResult(cmd, cwd, b"agent-loop/managed-1112\n", b"", 0)
            return super().run_binary(args, cwd=cwd, **kwargs)

    runner = CorruptibleRunner(
        claude_outputs=[
            "Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->",
            "Fixed review.\n<!-- AGENT_STATE: blocking -->",
        ],
        codex_outputs=[
            "Finding: bug remains.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)
    runner.coder_dir = config.claude_dir
    (config.claude_dir / "orchestrator.py").write_text("edited by someone else\n")
    (config.claude_dir / "protocol.py").write_text("edited by someone else\n")
    real_record = cv.record_checkout_baseline

    def record(cfg, run, path, *, source):
        real_record(cfg, run, path, source=source)
        if "coder turn" in source and path == run.coder_dir:
            run.corrupted = True

    monkeypatch.setattr(cv, "record_checkout_baseline", record)

    with pytest.raises(CheckoutVerificationError) as info:
        run_issue_loop(runner, issue_number=56, config=config)

    assert "observed main" in str(info.value)
    assert "orchestrator.py" in str(info.value) and "protocol.py" in str(info.value)
    claude_turns = [cmd for cmd, _cwd in runner.commands if cmd[:2] == ["claude", "--print"]]
    assert len(claude_turns) == 1  # the corrupted checkout was never given a second turn
    assert not runner.comments or "Fixed review" not in runner.comments[-1]


def test_issue_loop_syncs_coder_base_after_memory_before_coder(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->",
        ],
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)
    config.agent_memory_dir.mkdir(parents=True)
    (config.agent_memory_dir / "last-analyzed-commit").write_text("base123\n", encoding="utf-8")

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    commands = runner.commands
    issue_context_index = command_index(commands, ["gh", "issue", "view"])
    memory_index = command_index(commands, ["git", "diff", "--name-only"])
    fetch_index = command_index(commands, ["git", "fetch", "origin"])
    switch_index = command_index(commands, ["git", "switch", "main"])
    reset_index = command_index(commands, ["git", "reset", "--hard", runner.git_head])
    coder_index = command_index(commands, ["claude", "--print"])

    assert issue_context_index < memory_index < fetch_index < switch_index < reset_index < coder_index


def test_issue_loop_can_use_codex_as_coder_and_claude_as_reviewer(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->",
            "Fixed review.\n<!-- AGENT_STATE: blocking -->",
        ],
        claude_outputs=[
            "Finding: bug remains.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    agent_commands = [cmd[:2] for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"])]
    assert agent_commands == [
        ["codex", "exec"],
        ["claude", "--print"],
        ["codex", "exec"],
        ["claude", "--print"],
    ]
    assert len(runner.comments) == 6
    assert runner.comments[-1].startswith("**Review verdict:** Approved\n\nLGTM.")


def test_issue_loop_runs_pre_review_tests_after_coder_changes(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Created PR.\nTests: pytest passed.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->",
            "Fixed review.\nTests: pytest passed.\n<!-- AGENT_STATE: blocking -->",
        ],
        codex_outputs=[
            "Finding: bug remains.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, test_command=("pytest", "tests/test_agent_loop.py"))

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    first_coder = command_index(runner.commands, ["claude", "--print"])
    first_test = commands.index(["pytest", "tests/test_agent_loop.py"])
    first_review = command_index(runner.commands, ["codex", "exec"])
    assert first_coder < first_test < first_review
    assert commands.count(["pytest", "tests/test_agent_loop.py"]) == 3


def test_issue_loop_requires_claude_to_report_pr_number(tmp_path):
    runner = FakeRunner(claude_outputs=["Created something.\n<!-- AGENT_STATE: blocking -->"])
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="valid PR"):
        run_issue_loop(runner, issue_number=56, config=config)


def test_issue_loop_rejects_missing_initial_issue_human_requirements_acknowledgement(tmp_path):
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Keep the legacy flag.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            "Created PR.\nTests: python -m pytest passed.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->"
        ],
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="missing requirement ID"):
        run_issue_loop(runner, issue_number=56, config=config)


def test_issue_loop_accepts_initial_issue_human_requirements_acknowledgement(tmp_path):
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Keep the legacy flag.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            "Created PR.\nTests: python3 -m pytest passed.\n"
            f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
            "### Human requirements\n"
            "- Requirement 1: kept the legacy flag path.\n"
            "<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->"
        ],
        codex_outputs=[
            "LGTM.\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config) == 0


def test_issue_loop_rejects_pr_number_before_running_claude(tmp_path):
    runner = FakeRunner(issue_payload={
        "number": 62,
        "state": "closed",
        "is_pr": True,
        "url": "https://github.com/OWNER/REPO/pull/62",
    })
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="pull request, not an issue"):
        run_issue_loop(runner, issue_number=62, config=config)

    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


@pytest.mark.parametrize('invalid', [False, True])
def test_new_issue_pr_observation_precedes_head_bound_publication(tmp_path, invalid):
    from coding_review_agent_loop.pr_head_sync import PrHeadObservationError
    runner = FakeRunner(claude_outputs=['Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->'])
    runner.branch_tip = 'b' * 40
    if invalid: runner.pr_payload['headRefOid'] = 'invalid'
    with pytest.raises(PrHeadObservationError):
        run_issue_loop(runner, issue_number=56, config=make_config(tmp_path))
    assert not runner.pr_payload.get('comments')
    assert not any(cmd[:3] == ['gh', 'pr', 'merge'] for cmd, _ in runner.commands)


def test_new_issue_pr_transient_convergence_binds_latest_context(tmp_path, monkeypatch):
    import coding_review_agent_loop.issue_loop as issue_loop
    runner = FakeRunner(claude_outputs=['Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->'])
    before = runner.pr_payload['headRefOid']
    runner.branch_tip = 'b' * 40
    original = runner.run

    def run(args, **kwargs):
        if args == ['sleep', '2']:
            runner.pr_payload.update(headRefOid='b' * 40, url='https://example.test/latest', title='Latest context')
        return original(args, **kwargs)

    runner.run = run
    metadata = []
    attach = issue_loop._attach_round_metadata

    def capture(body, record):
        metadata.append(record)
        return attach(body, record)

    monkeypatch.setattr(issue_loop, '_attach_round_metadata', capture)
    delegated = []
    monkeypatch.setattr(issue_loop, 'run_pr_loop', lambda *a, **k: delegated.append(k) or 0)
    assert run_issue_loop(runner, issue_number=56, config=make_config(tmp_path)) == 0
    assert delegated
    handoffs = [body for body in runner.comments if 'AGENT_ISSUE_PR_HANDOFF' in body]
    assert len(handoffs) == 1
    assert 'b' * 40 in handoffs[0] and 'https://example.test/latest' in handoffs[0]
    assert before not in handoffs[0]
    coder = [body for body in runner.comments if body.startswith('## Issue implementation')]
    assert len(coder) == 1
    assert metadata and all(record.subject == 'b' * 40 for record in metadata)
