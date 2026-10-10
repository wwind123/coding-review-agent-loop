"""Antigravity quota-group fallback, run-owned memory and early stop (#1236)."""
import contextvars
import threading
from unittest.mock import patch

from agent_loop_helpers import *  # noqa: F403
from coding_review_agent_loop import validated_agent, workdir_claims
from coding_review_agent_loop.agents import antigravity as agy
from coding_review_agent_loop.config import (
    DEFAULT_ANTIGRAVITY_MODELS,
    parse_antigravity_quota_group_overrides,
)
from coding_review_agent_loop.errors import AgentInvocationError, QuotaResetExceededError
from coding_review_agent_loop.reviewer_seats import ReviewerSeat, SeatAgent


def test_named_seat_fallback_stays_local_and_usage_is_separate(tmp_path):
    import json
    from coding_review_agent_loop.cli import run_pr_loop

    first = SeatAgent(ReviewerSeat("flash", "antigravity", ("Model A", "Model B")), tmp_path / "flash")
    second = SeatAgent(ReviewerSeat("opus", "antigravity", ("Model C",)), tmp_path / "opus")
    first.workdir.mkdir()
    second.workdir.mkdir()
    runner = FakeRunner(antigravity_outputs=[
        ("quota exceeded", 1),
        (structured_pr_review(reviewer="flash (Google Antigravity: Model B)"), 0),
        (structured_pr_review(reviewer="opus (Google Antigravity: Model C)"), 0),
    ])
    config = make_config(
        tmp_path, reviewer=(first, second), reviewer_seats=(first, second),
        agent_max_retries=0, pre_review_tests=False,
    )
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    reviews = [
        comment["body"] for comment in runner.pr_payload["comments"]
        if "**Review verdict:**" in comment["body"]
    ]
    assert len(reviews) == 2
    assert any(body.endswith("-- flash (Google Antigravity: Model B)") for body in reviews)
    assert any(body.endswith("-- opus (Google Antigravity: Model C)") for body in reviews)
    models = [cmd[cmd.index("--model") + 1] for cmd, _ in runner.commands
              if cmd and cmd[0] == "agy" and "--model" in cmd]
    assert models == ["Model A", "Model B", "Model C"]
    summary = json.loads(next(config.log_dir.glob("*-usage-summary.json")).read_text())
    assert [call["agent"] for call in summary["calls"]] == ["flash", "flash", "opus"]
    assert [call["backend"] for call in summary["calls"]] == ["antigravity"] * 3
    assert sum(call["validation_status"] == "validated" for call in summary["calls"]) == 2
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert len([cmd for cmd, _ in runner.commands if cmd and cmd[0] == "agy" and "--model" in cmd]) == 3


def test_named_seats_share_backend_quota_memory_without_model_substitution(tmp_path):
    first = SeatAgent(ReviewerSeat("flash", "antigravity", (GEM1, OPUS)), tmp_path / "flash")
    second = SeatAgent(ReviewerSeat("other", "antigravity", ("Gemini 3.7 Flash (High)",)), tmp_path / "other")
    first.workdir.mkdir()
    second.workdir.mkdir()
    config = make_config(
        tmp_path, reviewer=(first, second), reviewer_seats=(first, second),
        agent_max_retries=0,
    )
    runner = FakeRunner(antigravity_outputs=[_fail(LIVE_SAMPLE), ("approved", 0)])
    with workdir_claims.workdir_claim_scope():
        response = validated_agent._run_validated_agent(
            runner, agent=first, config=config, prompt="Review", marker_description="none",
            validate=lambda text: text,
        )
        assert response.model_used == OPUS
        with pytest.raises(AgentInvocationError, match="tried no models"):
            validated_agent._run_validated_agent(
                runner, agent=second, config=config, prompt="Review", marker_description="none",
                validate=lambda text: text,
            )
    models = [cmd[cmd.index("--model") + 1] for cmd, _ in runner.commands
              if cmd and cmd[0] == "agy" and "--model" in cmd]
    assert models == [GEM1, OPUS]

LIVE_SAMPLE = (
    "error: RESOURCE_EXHAUSTED (code 429): Resource has been exhausted (e.g. check quota).\n"
    'AGY_ERROR: {"status":"RESOURCE_EXHAUSTED","error_code":429,"message":"exhausted","retryable":true}'
)
QUOTA_4H = "Error: quota exceeded, try again in 4h"
INDENTED_4H = "Error: quota exceeded\n    daily limit reached\n    plan: pro\n    try again in 4h"
JSON_4H = (
    '{\n  "error": {\n    "code": 429,\n    "status": "RESOURCE_EXHAUSTED",\n'
    '    "details": [\n      {"retryDelay": "14400s"}\n    ]\n  }\n}'
)
HIGH_TRAFFIC = "Error: high traffic, try again in a minute"
OPUS = "Claude Opus 5.5 (Medium)"
GEM1 = "Gemini 3.8 Flash (High)"


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _fail(text):
    return (text, 1)


def _turn(tmp_path, outputs, *, agent_max_retries=1, runner=None, **overrides):
    runner = runner or FakeRunner(antigravity_outputs=list(outputs))
    config = make_config(tmp_path, agent_max_retries=agent_max_retries, **overrides)
    logs: list[str] = []
    try:
        with patch("coding_review_agent_loop.validated_agent.log", lambda _c, m: logs.append(m)), \
                patch.object(agy, "log", lambda _c, m: logs.append(m)):
            response = validated_agent._run_validated_agent(
                runner,
                agent="antigravity",
                config=config,
                prompt="PROMPT",
                marker_description="none",
                validate=lambda text: text,
            )
        error = None
    except (AgentInvocationError, QuotaResetExceededError) as exc:
        response, error = None, exc
    cmds = [c for c, _cwd in runner.commands if c[:1] == ["agy"]]
    models = [c[c.index("--model") + 1] for c in cmds if "--model" in c]
    sleeps = [c for c, _cwd in runner.commands if c[:1] == ["sleep"]]
    return response, error, models, sleeps, logs


# --- config / groups -------------------------------------------------------

def test_default_chain_has_opus_last():
    assert len(DEFAULT_ANTIGRAVITY_MODELS) == 5
    assert DEFAULT_ANTIGRAVITY_MODELS[-1] == OPUS


def test_group_derivation_and_override():
    assert agy.antigravity_quota_group("Gemini 3.1 Pro (High)") == "gemini"
    assert agy.antigravity_quota_group(" claude Sonnet 5.5 (Medium)") == "claude"
    assert agy.antigravity_quota_group("Other X") == "model:Other X"
    assert agy.antigravity_quota_group(OPUS, ((OPUS, "gemini"),)) == "gemini"


def test_config_validation_of_cooldown_and_overrides(tmp_path):
    for bad in (0, -1, 3601, True):
        with pytest.raises(AgentLoopError):
            make_config(tmp_path, antigravity_quota_cooldown_seconds=bad)
    with pytest.raises(AgentLoopError):
        make_config(tmp_path, antigravity_quota_groups=(("a", ""),))
    with pytest.raises(AgentLoopError):
        make_config(tmp_path, antigravity_quota_groups=(("a", "g"), ("a", "h")))
    for bad in ("nogroup", "=g", "m="):
        with pytest.raises(AgentLoopError):
            parse_antigravity_quota_group_overrides([bad])
    assert parse_antigravity_quota_group_overrides(["A B (X)=g"]) == (("A B (X)", "g"),)


def test_cli_override_must_name_a_chain_model(tmp_path):
    parser = build_parser()
    base = ["pr", "1", "--repo", "O/R", "--coder", "antigravity", "--reviewer", "codex",
            "--codex-dir", str(tmp_path / "codex"), "--dangerous-agent-permissions"]
    with pytest.raises(AgentLoopError):
        config_from_args(parser.parse_args([*base, "--antigravity-quota-group", "Nope=g"]), FakeRunner())
    config = config_from_args(
        parser.parse_args([*base, "--antigravity-quota-group", f"{OPUS}=gemini"]), FakeRunner()
    )
    assert config.antigravity_quota_groups == ((OPUS, "gemini"),)
    # Narrowing to a single model keeps every override and constructs cleanly.
    state = agy.AntigravityAttemptState.from_config(config, 1)
    assert state.groups[-1] == "gemini"
    assert state.singleton_config(config).antigravity_quota_groups == ((OPUS, "gemini"),)


def test_override_only_on_fallback_entry_survives_singleton(tmp_path):
    config = make_config(tmp_path, antigravity_quota_groups=((OPUS, "x"),))
    state = agy.AntigravityAttemptState.from_config(config, 1)
    assert state.groups == ("gemini",) * 4 + ("x",)
    narrowed = state.singleton_config(config)
    assert narrowed.antigravity_models == (GEM1,)
    assert narrowed.antigravity_quota_groups == ((OPUS, "x"),)


# --- validated turns -------------------------------------------------------

def test_quota_on_first_link_skips_gemini_group_without_sleep(tmp_path):
    response, error, models, sleeps, logs = _turn(tmp_path, [_fail(LIVE_SAMPLE), ("ok review", 0)])
    assert error is None and response.text == "ok review"
    assert models == [GEM1, OPUS]
    assert sleeps == []
    assert any("capacity failure after" in line for line in logs)
    assert any("skipping quota group gemini" in line for line in logs)
    assert response.model_used == OPUS


def test_quota_exhaustion_does_not_consume_retry_budget(tmp_path):
    _r, error, models, sleeps, _l = _turn(
        tmp_path, [_fail(LIVE_SAMPLE), _fail(HIGH_TRAFFIC), ("done", 0)], agent_max_retries=1
    )
    assert error is None
    assert models == [GEM1, OPUS, OPUS]
    assert len(sleeps) == 1  # only the transient retry sleeps


def test_transient_failure_retries_same_model_first(tmp_path):
    _r, error, models, sleeps, _l = _turn(tmp_path, [_fail(HIGH_TRAFFIC), ("done", 0)])
    assert error is None
    assert models == [GEM1, GEM1]
    assert len(sleeps) == 1


def test_run_memory_next_turn_and_expiry_with_fake_clock(tmp_path):
    clock = FakeClock()
    with patch.object(agy, "_now", clock), workdir_claims.workdir_claim_scope():
        _t1 = _turn(tmp_path, [_fail(LIVE_SAMPLE), ("one", 0)])
        r2, e2, models2, _s, logs2 = _turn(tmp_path, [("two", 0)])
        assert e2 is None and models2 == [OPUS]
        assert not any("exhausted until" in line for line in logs2)
        clock.now += 601
        r3, e3, models3, _s, logs3 = _turn(tmp_path, [("three", 0)])
        assert models3 == [GEM1]
        assert sum("eligible again" in line for line in logs3) == 1


def test_parsed_long_reset_variants_fall_back_and_expire(tmp_path):
    for frame in (QUOTA_4H, INDENTED_4H, JSON_4H):
        clock = FakeClock()
        with patch.object(agy, "_now", clock), workdir_claims.workdir_claim_scope():
            _r, error, models, sleeps, _l = _turn(tmp_path, [_fail(frame), ("fine", 0)])
            assert error is None, frame
            assert models == [GEM1, OPUS] and sleeps == []
            entry = agy.quota_memory_for_current_run().entry("gemini")
            assert entry.source == "parsed"
            assert 14390 <= entry.expires_at - clock.now <= 14400
            clock.now += 14401
            assert not agy.quota_memory_for_current_run().is_exhausted("gemini")


def test_all_groups_parsed_long_reset_raises_quota_reset(tmp_path):
    with patch.object(agy, "_now", FakeClock()), workdir_claims.workdir_claim_scope():
        agy.quota_memory_for_current_run().mark_exhausted(
            "gemini", 14400, "parsed", QUOTA_4H, 0.0, cooldown_seconds=600
        )
        _r, error, models, _s, _l = _turn(tmp_path, [_fail(QUOTA_4H)])
        assert isinstance(error, QuotaResetExceededError)
        assert models == [OPUS]
        assert "quota exhausted. Reset in" in str(error)


def test_cooldown_only_exhaustion_stops_with_unavailable_message(tmp_path):
    with patch.object(agy, "_now", FakeClock()), workdir_claims.workdir_claim_scope():
        agy.quota_memory_for_current_run().mark_exhausted(
            "gemini", None, "cooldown", LIVE_SAMPLE, 0.0, cooldown_seconds=600
        )
        _r, error, models, sleeps, _l = _turn(tmp_path, [_fail(LIVE_SAMPLE)])
        assert type(error) is AgentInvocationError
        assert str(error).startswith("Antigravity unavailable on all models")
        assert models == [OPUS] and sleeps == []


def test_turn_start_with_every_group_cooling_makes_no_invocation(tmp_path):
    with patch.object(agy, "_now", FakeClock()), workdir_claims.workdir_claim_scope():
        memory = agy.quota_memory_for_current_run()
        for group in ("gemini", "claude"):
            memory.mark_exhausted(group, None, "cooldown", LIVE_SAMPLE, 0.0, cooldown_seconds=600)
        _r, error, models, sleeps, _l = _turn(tmp_path, [("never", 0)])
        assert str(error).startswith("Antigravity unavailable on all models")
        assert models == [] and sleeps == []


def test_second_exhaustion_after_jump_is_shared_limit_and_third_group_untried(tmp_path):
    chain = (GEM1, OPUS, "Model X")
    _r, error, models, sleeps, _l = _turn(
        tmp_path, [_fail("Error: quota exceeded"), _fail(LIVE_SAMPLE), ("never", 0)],
        antigravity_models=chain,
    )
    assert type(error) is AgentInvocationError
    assert models == [GEM1, OPUS]
    assert "failed the same way" in str(error)
    assert sleeps == []


def test_transient_walk_to_opus_then_exhaustion_jumps_back_to_gemini(tmp_path):
    _r, error, models, _s, _l = _turn(
        tmp_path,
        [_fail(HIGH_TRAFFIC)] * 4 + [_fail(QUOTA_4H), _fail(HIGH_TRAFFIC), _fail(HIGH_TRAFFIC)],
        agent_max_retries=0,
    )
    assert models[:5] == list(DEFAULT_ANTIGRAVITY_MODELS)
    assert models[5] == GEM1  # Claude group exhausted; jump back, no raise
    assert not isinstance(error, QuotaResetExceededError)


def test_unverified_transcript_reset_at_end_of_chain_does_not_raise_quota_reset(tmp_path):
    noise = "reviewing: the quota note says try again in 4h\nmore transcript\n"
    outputs = [_fail(HIGH_TRAFFIC)] * 5 + [_fail(noise + HIGH_TRAFFIC)]
    with workdir_claims.workdir_claim_scope():
        _r, error, models, _s, _l = _turn(tmp_path, outputs, agent_max_retries=1)
        memory = agy.quota_memory_for_current_run()
        assert all(memory.entry(g) is None for g in ("gemini", "claude"))
    # one retry is spent on Gemini 3.8, then the walk reaches Opus, which fails last
    assert models == [GEM1, GEM1, *DEFAULT_ANTIGRAVITY_MODELS[1:]]
    assert type(error) is AgentInvocationError
    assert not str(error).startswith("Antigravity unavailable")


def test_sparse_newline_head_header_never_becomes_exhaustion(tmp_path):
    text = "Error: quota exceeded, try again in 4h\n" + "x" * 3000
    capacity = validated_agent.classify_antigravity_capacity(
        text, returncode=1, empty_response=False, signatures=make_config(tmp_path).antigravity_quota_signatures
    )
    assert capacity.is_capacity and capacity.frame == ""
    with workdir_claims.workdir_claim_scope():
        _r, error, models, sleeps, _l = _turn(
            tmp_path, [_fail(text), ("ok", 0)], agent_max_retries=0
        )
        assert error is None
        assert models == [GEM1, "Gemini 3.7 Flash (High)"]  # transient sibling fallback, no skip
        assert agy.quota_memory_for_current_run().entry("gemini") is None


def test_shared_limit_message_names_the_actual_last_failure(tmp_path):
    claude_frame = "Error: quota exceeded (claude frame)"
    gemini_frame = "Error: quota exceeded (gemini frame)"
    outputs = [_fail(HIGH_TRAFFIC)] * 4 + [_fail(claude_frame), _fail(gemini_frame)]
    _r, error, models, _s, _l = _turn(tmp_path, outputs, agent_max_retries=0)
    assert models[-2:] == [OPUS, GEM1]
    assert type(error) is AgentInvocationError
    assert "gemini frame" in str(error) and "claude frame" not in str(error)


def test_remembered_later_group_does_not_override_current_failure_frame(tmp_path):
    with patch.object(agy, "_now", FakeClock()), workdir_claims.workdir_claim_scope():
        agy.quota_memory_for_current_run().mark_exhausted(
            "claude", None, "cooldown", "Error: quota exceeded (old claude frame)", 0.0,
            cooldown_seconds=600,
        )
        _r, error, _m, _s, _l = _turn(tmp_path, [_fail("Error: quota exceeded (current gemini frame)")])
        assert "current gemini frame" in str(error) and "old claude" not in str(error)


def test_turn_start_stop_reports_most_recent_remembered_frame(tmp_path):
    with patch.object(agy, "_now", FakeClock()), workdir_claims.workdir_claim_scope():
        memory = agy.quota_memory_for_current_run()
        memory.mark_exhausted("claude", None, "cooldown", "Error: quota (first, claude)", 0.0, cooldown_seconds=600)
        memory.mark_exhausted("gemini", None, "cooldown", "Error: quota (latest, gemini)", 0.0, cooldown_seconds=600)
        _r, error, models, _s, _l = _turn(tmp_path, [("never", 0)])
        assert models == [] and "latest, gemini" in str(error)
        assert error.failure_category == "transient"


def test_expiry_while_opus_runs_then_quota_jumps_back_to_gemini_once(tmp_path):
    clock = FakeClock()

    class ClockAdvancingRunner(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            if list(args)[:1] == ["agy"] and not getattr(self, "advanced", False):
                self.advanced = True
                clock.now += 601  # Gemini's 600 s reset passes while Opus is running
            return super().run_with_log(args, cwd=cwd, **kwargs)

    runner = ClockAdvancingRunner(antigravity_outputs=[_fail(LIVE_SAMPLE), ("fine", 0)])
    with patch.object(agy, "_now", clock), workdir_claims.workdir_claim_scope():
        agy.quota_memory_for_current_run().mark_exhausted(
            "gemini", 600, "parsed", "Error: quota 10m", 0.0, cooldown_seconds=600
        )
        response, error, models, sleeps, logs = _turn(tmp_path, [], runner=runner)
    assert error is None and response.text == "fine"
    assert models == [OPUS, GEM1] and sleeps == []
    assert sum("eligible again" in line for line in logs) == 1


def test_all_groups_quoted_json_long_reset_raises_and_ignores_unrelated_groups(tmp_path):
    with patch.object(agy, "_now", FakeClock()), workdir_claims.workdir_claim_scope():
        memory = agy.quota_memory_for_current_run()
        memory.mark_exhausted("model:unrelated", 60, "parsed", "f", 0.0, cooldown_seconds=600)
        memory.mark_exhausted("gemini", 14400, "parsed", JSON_4H, 0.0, cooldown_seconds=600)
        _r, error, models, _s, _l = _turn(tmp_path, [_fail(JSON_4H.replace("14400s", "18000s"))])
        assert isinstance(error, QuotaResetExceededError)
        assert models == [OPUS]
        assert "Reset in 4h" in str(error)  # earliest chain reset (Gemini's 14400 s), not 60 s


def test_parallel_memory_through_production_reviewer_launcher(tmp_path):
    from coding_review_agent_loop import review_rounds

    config = make_config(tmp_path)
    quota = validated_agent.classify_antigravity_quota_exhaustion(
        validated_agent.classify_antigravity_capacity(
            LIVE_SAMPLE, returncode=1, empty_response=False,
            signatures=config.antigravity_quota_signatures,
        )
    )
    logs: list[str] = []
    seen = []
    barrier = threading.Barrier(2)

    def run_turn(reviewer):
        state = agy.AntigravityAttemptState.from_config(config, 1)
        seen.append(state.memory)
        barrier.wait(timeout=10)
        state.next_after_quota_exhaustion(quota, 0.0)
        return review_rounds._ReviewerTurnResult(reviewer_name=str(reviewer))

    with patch.object(agy, "_now", FakeClock()), patch.object(agy, "log", lambda _c, m: logs.append(m)), \
            workdir_claims.workdir_claim_scope():
        early = agy.AntigravityAttemptState.from_config(config, 1)
        review_rounds._launch_reviewer_turns(
            FakeRunner(), ["codex", "gemini"], thread_name_prefix="t", run_turn=run_turn
        )
        assert seen[0] is seen[1] is early.memory
        assert sum("exhausted until" in line for line in logs) == 1
        assert early.ensure_eligible_before_attempt() == "ok"
        assert early.models[early.model_index] == OPUS


def test_multiple_overrides_and_out_of_chain_repair_model(tmp_path, monkeypatch):
    import sys

    from coding_review_agent_loop import repair as repair_module
    from coding_review_agent_loop.repair import execute_repair

    # test_antigravity_module_imports_without_fcntl re-imports the module, so
    # the module-level ``agy`` can be stale on a shared worker; patch the live
    # module whose backend class the repair chain really uses.
    agy = sys.modules[repair_module.AntigravityBackend.__module__]

    monkeypatch.setattr(agy, "_antigravity_settings_path", lambda: tmp_path / "settings.json")
    overrides = ((GEM1, "g1"), ("Gemini 3.7 Flash (High)", "g1"), (OPUS, "g2"))
    config = make_config(tmp_path, antigravity_quota_groups=overrides, repair_models=("Out Of Chain Model",))
    state = agy.AntigravityAttemptState.from_config(config, 1)
    assert state.groups == ("g1", "g1", "gemini", "gemini", "g2")
    captured = []
    real_run = agy.AntigravityBackend.run

    def spy(self, runner, cfg, *args, **kwargs):
        captured.append(cfg)
        return real_run(self, runner, cfg, *args, **kwargs)

    memories_before = dict(agy._run_memories)
    with patch.object(agy.AntigravityBackend, "run", spy):
        execute_repair(
            malformed_pr_review_source(state="approved"),
            runner=FakeRunner(antigravity_outputs=[("not structured", 0)] * 8),
            config=config, run_id="r", usage_context=None,
            validate=lambda t: (_ for _ in ()).throw(AgentLoopError("invalid")),
            expected_kind="pr_review",
        )
    assert captured and all(c.antigravity_quota_groups == overrides for c in captured)
    assert captured[0].antigravity_models == ("Out Of Chain Model",)
    assert agy._run_memories == memories_before


def test_interleaved_custom_chain_never_invokes_exhausted_group(tmp_path):
    chain = ("Gemini A", "Claude B", "Gemini C")
    with patch.object(agy, "_now", FakeClock()), workdir_claims.workdir_claim_scope():
        agy.quota_memory_for_current_run().mark_exhausted(
            "gemini", None, "cooldown", LIVE_SAMPLE, 0.0, cooldown_seconds=600
        )
        _r, error, models, _s, _l = _turn(
            tmp_path, [_fail(HIGH_TRAFFIC)] * 4, agent_max_retries=1, antigravity_models=chain
        )
        assert set(models) == {"Claude B"}
        assert error is not None


def test_non_antigravity_agent_keeps_long_reset_raise(tmp_path):
    runner = FakeRunner(codex_outputs=[("Error: rate limit exceeded, try again in 4h", 1)])
    config = make_config(tmp_path, agent_max_retries=1)
    with pytest.raises(QuotaResetExceededError):
        validated_agent._run_validated_agent(
            runner, agent="codex", config=config, prompt="P", marker_description="none",
            validate=lambda t: t,
        )


# --- run ownership / threads ----------------------------------------------

def test_memory_is_owned_by_the_logical_run():
    with patch.object(agy, "_now", FakeClock()):
        with workdir_claims.workdir_claim_scope():
            first = agy.quota_memory_for_current_run()
            first.mark_exhausted("gemini", None, "cooldown", "f", 0.0, cooldown_seconds=600)
            with workdir_claims.workdir_claim_scope():  # nested loop joins the run
                assert agy.quota_memory_for_current_run() is first
                assert agy.quota_memory_for_current_run().is_exhausted("gemini")
        with workdir_claims.workdir_claim_scope():
            assert not agy.quota_memory_for_current_run().is_exhausted("gemini")
    assert agy._run_memories == {}


def test_no_owner_gives_local_memory_per_state(tmp_path):
    config = make_config(tmp_path)
    a = agy.AntigravityAttemptState.from_config(config, 1)
    b = agy.AntigravityAttemptState.from_config(config, 1)
    assert a.memory is not b.memory


def test_parallel_threads_share_run_memory_and_log_once(tmp_path):
    config = make_config(tmp_path)
    logs: list[str] = []
    with patch.object(agy, "_now", FakeClock()), patch.object(agy, "log", lambda _c, m: logs.append(m)), \
            workdir_claims.workdir_claim_scope():
        early = agy.AntigravityAttemptState.from_config(config, 1)
        quota = validated_agent.classify_antigravity_quota_exhaustion(
            validated_agent.classify_antigravity_capacity(
                LIVE_SAMPLE, returncode=1, empty_response=False,
                signatures=config.antigravity_quota_signatures,
            )
        )
        barrier = threading.Barrier(2)
        states = []

        def worker():
            state = agy.AntigravityAttemptState.from_config(config, 1)
            states.append(state)
            barrier.wait()
            state.next_after_quota_exhaustion(quota, 0.0)

        threads = [
            threading.Thread(target=contextvars.copy_context().run, args=(worker,)) for _ in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert states[0].memory is states[1].memory is early.memory
        assert sum("exhausted until" in line for line in logs) == 1
        assert early.ensure_eligible_before_attempt() == "ok"
        assert early.models[early.model_index] == OPUS


def test_unverified_reset_with_retry_budget_on_opus_retries_same_model_then_fails(tmp_path):
    noise = "reviewing: the quota note says try again in 4h\nmore transcript\n"
    with workdir_claims.workdir_claim_scope():
        _r, error, models, sleeps, _l = _turn(
            tmp_path, [_fail(noise + HIGH_TRAFFIC)] * 2, agent_max_retries=1,
            antigravity_models=(OPUS,),
        )
        memory = agy.quota_memory_for_current_run()
        assert memory.entry("claude") is None
    assert models == [OPUS, OPUS]  # same-model retry inside the budget
    assert len(sleeps) == 1
    assert type(error) is AgentInvocationError
    assert not isinstance(error, QuotaResetExceededError)


def test_format_repair_never_reads_or_marks_quota_memory(tmp_path, monkeypatch):
    from coding_review_agent_loop.repair import execute_repair

    monkeypatch.setattr(agy, "_antigravity_settings_path", lambda: tmp_path / "settings.json")
    touched = []

    def boom(*_a, **_k):
        touched.append("access")
        raise AssertionError("repair touched quota memory")

    config = make_config(tmp_path, repair_models=("Out Of Chain Model",))
    with workdir_claims.workdir_claim_scope(), \
            patch.object(agy, "quota_memory_for_current_run", boom), \
            patch.object(agy.AntigravityQuotaGroupMemory, "mark_exhausted", boom), \
            patch.object(agy.AntigravityQuotaGroupMemory, "is_exhausted", boom), \
            patch.object(agy.AntigravityAttemptState, "from_config", boom):
        execute_repair(
            malformed_pr_review_source(state="approved"),
            runner=FakeRunner(antigravity_outputs=[(LIVE_SAMPLE, 1)] * 8),
            config=config, run_id="r", usage_context=None,
            validate=lambda t: (_ for _ in ()).throw(AgentLoopError("invalid")),
            expected_kind="pr_review",
        )
    assert touched == []


def test_every_unavailable_stop_is_a_transient_category(tmp_path):
    _r, shared, *_ = _turn(
        tmp_path, [_fail("Error: quota exceeded"), _fail(LIVE_SAMPLE)],
        antigravity_models=(GEM1, OPUS, "Model X"),
    )
    assert shared.failure_category == "transient"
    with patch.object(agy, "_now", FakeClock()), workdir_claims.workdir_claim_scope():
        _r, error, *_ = _turn(tmp_path, [_fail(LIVE_SAMPLE)] * 2)
        assert str(error).startswith("Antigravity unavailable")
        assert error.failure_category == "transient"


def test_exhausted_antigravity_is_marked_unavailable_not_fatal_in_a_pr_round(tmp_path):
    from test_orchestrator_pr import structured_pr_review  # noqa: F401

    runner = FakeRunner(
        codex_outputs=[structured_pr_review(summary="Codex ok.")],
        antigravity_outputs=[_fail(LIVE_SAMPLE)] * 3,
    )
    config = make_config(tmp_path, reviewer=("antigravity", "codex"), review_parallel=False)
    with workdir_claims.workdir_claim_scope():
        with pytest.raises(AgentLoopError) as excinfo:
            run_pr_loop(runner, pr_number=77, config=config)
    # The round finished Codex's review and then stopped for the unavailable reviewer.
    assert any("Codex ok." in c for c in runner.comments)
    assert "unavailable" in str(excinfo.value).lower()
