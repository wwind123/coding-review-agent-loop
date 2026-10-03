"""A reviewer that omits a carried-item disposition is re-asked once (#1167)."""
import threading
from unittest.mock import patch

import pytest

import coding_review_agent_loop.orchestrator as orchestrator
from coding_review_agent_loop.cli import run_pr_loop
from coding_review_agent_loop.errors import AgentLoopError, MissingPriorItemDispositionError
from coding_review_agent_loop.protocol import UnresolvedReviewItem
from coding_review_agent_loop.review_spool import ReviewRoundSpool
from coding_review_agent_loop.unresolved_items import _validate_review_response
from agent_loop_helpers import (
    FakeRunner,
    make_config,
    structured_coder_followup,
    structured_pr_review,
)

ADDENDUM_HEADING = "## Previous response not accepted: prior item dispositions"
OMITTED = structured_pr_review(summary="COMPLETE-NOT Approves without dispositions.")
COMPLETE = structured_pr_review(
    summary="COMPLETE Approves with dispositions.",
    prior_item_dispositions=[{"item_id": "item-4", "disposition": "resolved"}],
)


def _validate(text):
    if "COMPLETE-NOT" in text:
        raise MissingPriorItemDispositionError(("item-4",))
    if "COMPLETE" not in text:
        raise AgentLoopError("malformed review")
    return text


def _repair_recorder(calls):
    def fake_repair(raw, **kwargs):
        calls.append(raw)
        return None, None, []

    return fake_repair


def _run(runner, config, *, reask=True):
    return orchestrator._run_validated_agent(
        runner,
        agent="codex",
        config=config,
        prompt="ORIGINAL PROMPT",
        session_id="sess-1",
        marker_description="<!-- AGENT_STATE: approved|blocking -->",
        validate=_validate,
        use_repair=True,
        repair_expected_kind="pr_review",
        role="reviewer",
        operation_description="PR review",
        reask_on_prior_disposition_omission=reask,
    )


def _agent_commands(runner, binary):
    return [cmd for cmd, _cwd in runner.commands if cmd[:1] == [binary]]


def _is_sleep(cmd):
    return cmd[:1] == ["sleep"]


@pytest.mark.parametrize("backend", ["codex", "antigravity"])
def test_omission_is_reasked_once_on_same_model_and_accepted(tmp_path, backend):
    if backend == "codex":
        runner = FakeRunner(codex_outputs=[OMITTED, COMPLETE])
        config = make_config(tmp_path, reviewer=("codex",), agent_max_retries=0)
        binary = "codex"
    else:
        runner = FakeRunner(antigravity_outputs=[(OMITTED, 0), (COMPLETE, 0)])
        config = make_config(
            tmp_path,
            reviewer=("antigravity",),
            agent_max_retries=0,
            antigravity_models=("ModelA", "ModelB", "ModelC"),
        )
        binary = "agy"
    repair_calls: list[str] = []
    logs: list[str] = []
    with patch.object(orchestrator, "_run_structured_repair", _repair_recorder(repair_calls)), \
            patch.object(orchestrator, "log", lambda _cfg, msg: logs.append(msg)), \
            patch.object(orchestrator.time, "sleep", side_effect=AssertionError("slept")):
        response = orchestrator._run_validated_agent(
            runner,
            agent="codex" if backend == "codex" else "antigravity",
            config=config,
            prompt="ORIGINAL PROMPT",
            session_id=None,
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=_validate,
            use_repair=True,
            repair_expected_kind="pr_review",
            role="reviewer",
            operation_description="PR review",
            reask_on_prior_disposition_omission=True,
        )

    assert "COMPLETE Approves" in response.text
    commands = _agent_commands(runner, binary)
    assert len(commands) == 2
    flat = ["\n".join(cmd) for cmd in commands]
    assert "ORIGINAL PROMPT" in flat[0] and ADDENDUM_HEADING not in flat[0]
    assert "ORIGINAL PROMPT" in flat[1] and ADDENDUM_HEADING in flat[1] and "item-4" in flat[1]
    if backend == "antigravity":
        models = [cmd[cmd.index("--model") + 1] for cmd in commands if "--model" in cmd]
        assert len(models) == 2 and models[0] == models[1] == "ModelA"
    assert repair_calls == []
    assert not any(_is_sleep(cmd) for cmd, _cwd in runner.commands)
    assert any("re-asking the same reviewer once (no repair)" in line for line in logs)
    assert not any("retry" in line.lower() and "re-asking" not in line for line in logs)


def test_second_omission_follows_existing_path_with_no_third_invocation(tmp_path):
    runner = FakeRunner(codex_outputs=[OMITTED, OMITTED, OMITTED, OMITTED])
    config = make_config(tmp_path, reviewer=("codex",), agent_max_retries=0)
    repair_calls: list[str] = []
    with patch.object(orchestrator, "_run_structured_repair", _repair_recorder(repair_calls)):
        with pytest.raises(AgentLoopError, match="did not evaluate all prior unresolved items: item-4"):
            _run(runner, config)

    assert len(_agent_commands(runner, "codex")) == 2
    # The unchanged path is allowed to attempt (and be refused by) repair for
    # the second omission; it never fabricates the missing disposition.
    assert len(repair_calls) <= 1


def test_malformed_response_goes_to_repair_without_reask(tmp_path):
    runner = FakeRunner(codex_outputs=["not json at all", "not json at all"])
    config = make_config(tmp_path, reviewer=("codex",), agent_max_retries=0)
    repair_calls: list[str] = []
    with patch.object(orchestrator, "_run_structured_repair", _repair_recorder(repair_calls)):
        with pytest.raises(AgentLoopError):
            _run(runner, config)

    assert len(repair_calls) == 1
    commands = _agent_commands(runner, "codex")
    assert not any(ADDENDUM_HEADING in "\n".join(cmd) for cmd in commands)


def test_flag_off_does_not_reask(tmp_path):
    runner = FakeRunner(codex_outputs=[OMITTED, OMITTED, OMITTED])
    config = make_config(tmp_path, reviewer=("codex",), agent_max_retries=0)
    with patch.object(orchestrator, "_run_structured_repair", _repair_recorder([])):
        with pytest.raises(AgentLoopError):
            _run(runner, config, reask=False)

    assert not any(
        ADDENDUM_HEADING in "\n".join(cmd) for cmd in _agent_commands(runner, "codex")
    )


def test_validator_raises_typed_omission_with_unchanged_message():
    items = [
        UnresolvedReviewItem(
            item_id="item-4", reviewer="OpenAI Codex", source_round=1, text="Fix it.",
            status="blocking",
        )
    ]
    with pytest.raises(MissingPriorItemDispositionError) as info:
        _validate_review_response(
            structured_pr_review(summary="No dispositions."),
            reviewer="OpenAI Codex",
            unresolved_items=items,
            architecture_status_mode="legacy",
        )
    assert info.value.missing_ids == ("item-4",)
    assert str(info.value) == "Review did not evaluate all prior unresolved items: item-4"


class _ReaskRunner(FakeRunner):
    """Codex omits item-1 in round 2; gemini finishes between codex's two turns."""

    def __init__(self, *, parallel, **kwargs):
        super().__init__(**kwargs)
        self.parallel = parallel
        self.codex_calls = 0
        self.codex_first_done = threading.Event()
        self.gemini_done = threading.Event()
        self.codex_prompts: list[str] = []
        self.codex_cmds: list[list[str]] = []
        self.at_reask_start: dict | None = None
        self.events: list[tuple[str, str]] = []
        self.round2_events_start: int | None = None

    def run_with_log(self, args, *, cwd, **kwargs):
        cmd = [str(arg) for arg in args]
        if cmd[:1] == ["codex"]:
            self.codex_calls += 1
            number = self.codex_calls
            if number >= 2:
                self.codex_prompts.append(kwargs.get("input_text") or "")
                self.codex_cmds.append(cmd)
            if number == 2:
                self.round2_events_start = len(self.events)
            if number == 3:
                if self.parallel:
                    assert self.gemini_done.wait(10)
                start = self.round2_events_start or 0
                self.at_reask_start = {"events": list(self.events[start:])}
            result = super().run_with_log(args, cwd=cwd, **kwargs)
            if number == 2:
                self.codex_first_done.set()
            return result
        if cmd[:1] == ["gemini"] and self.parallel and self._gemini_round2():
            assert self.codex_first_done.wait(10)
            result = super().run_with_log(args, cwd=cwd, **kwargs)
            self.gemini_done.set()
            return result
        return super().run_with_log(args, cwd=cwd, **kwargs)

    def _gemini_round2(self):
        self._gemini_calls = getattr(self, "_gemini_calls", 0) + 1
        return self._gemini_calls == 2


def _reask_outputs(parallel):
    runner = _ReaskRunner(
        parallel=parallel,
        codex_outputs=[
            structured_pr_review(
                state="blocking", summary="Codex found a blocker.", blocking_items=["Blocker."]
            ),
            structured_pr_review(summary="Codex omits the carried item."),
            structured_pr_review(
                summary="Codex approves after re-ask.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        gemini_outputs=[
            structured_pr_review(summary="Gemini approves.", reviewer="Google Gemini"),
            structured_pr_review(
                summary="GEMINI-ROUND2-UNIQUE approves.",
                reviewer="Google Gemini",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(summary="Fixed it.", addressed_items=["item-1"])
        ],
    )
    return runner


def _run_reask_loop(tmp_path, parallel):
    runner = _reask_outputs(parallel)
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), review_parallel=parallel, agent_max_retries=0
    )
    real_post = orchestrator.post_pr_comment
    real_store = ReviewRoundSpool.store

    def recording_post(*args, **kwargs):
        runner.events.append(("post", kwargs.get("body", "")))
        return real_post(*args, **kwargs)

    def recording_store(self, reviewer_name, fields):
        runner.events.append(("spool", reviewer_name))
        return real_store(self, reviewer_name, fields)

    with patch.object(orchestrator, "post_pr_comment", side_effect=recording_post), \
            patch.object(ReviewRoundSpool, "store", recording_store):
        assert run_pr_loop(runner, pr_number=77, config=config) == 0
    return runner


@pytest.mark.parametrize("parallel", [True, False])
def test_reask_reuses_frozen_prompt_behind_publication_barrier(tmp_path, parallel):
    runner = _run_reask_loop(tmp_path, parallel)

    first, reask = runner.codex_prompts[0], runner.codex_prompts[1]
    # The backend appends a per-invocation response-file footer after the
    # orchestrator's prompt; compare the orchestrator-owned part only.
    marker = "PUBLIC RESPONSE FILE"
    first_body = first.split(marker)[0]
    assert reask.startswith(first_body)
    addendum = reask[len(first_body):].split(marker)[0]
    assert addendum.lstrip().startswith(ADDENDUM_HEADING) and "item-1" in addendum
    assert ADDENDUM_HEADING not in first
    assert "GEMINI-ROUND2-UNIQUE" not in reask
    # Same backend invocation shape (same model/session flags); only the
    # per-invocation temp artifact paths and the prompt itself differ.
    def shape(cmd):
        return [arg for arg in cmd[:-1] if not arg.startswith("/tmp/tmp")]

    assert shape(runner.codex_cmds[0]) == shape(runner.codex_cmds[1])

    # Nothing from this round was spooled or published while the re-ask ran
    # (parallel) or between A's two turns (sequential).
    assert runner.at_reask_start is not None
    assert runner.at_reask_start["events"] == []

    kinds = [kind for kind, _detail in runner.events]
    gemini_post = next(
        index for index, (kind, detail) in enumerate(runner.events)
        if kind == "post" and "GEMINI-ROUND2-UNIQUE" in detail
    )
    if parallel:
        assert "spool" in kinds[:gemini_post]
