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


# Bound for the two reviewer turns to meet at the launcher barrier (#1471).
# Barrier.wait returns as soon as both parties arrive, so a green run pays
# nothing; a serialized launcher still raises BrokenBarrierError after it.
_PEER_BARRIER_TIMEOUT_SECONDS = 60.0


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
        barrier.wait(timeout=_PEER_BARRIER_TIMEOUT_SECONDS)
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
    from agent_loop_helpers import structured_pr_review  # noqa: F401

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


# --- credits-balance exhaustion and auth precedence (#1598) -------------------

CREDITS_OUTPUT = (
    "error: Your AI credits balance is too low to continue.\n"
    'AGY_ERROR: {"short_error":"RESOURCE_EXHAUSTED (code 429): Resource has been exhausted '
    '(e.g. check quota).","status":"RESOURCE_EXHAUSTED","error_code":429,'
    '"code_kind":"http","retryable":true}'
)
RETRYABLE_429 = (
    'AGY_ERROR: {"status":"RESOURCE_EXHAUSTED","error_code":429,'
    '"message":"Resource has been exhausted","retryable":true}'
)


def _gemini_marked():
    return agy.quota_memory_for_current_run().entry("gemini") is not None


def test_credits_exhaustion_falls_back_to_next_group_without_sleep(tmp_path):
    with workdir_claims.workdir_claim_scope():
        response, error, models, sleeps, logs = _turn(
            tmp_path, [_fail(CREDITS_OUTPUT), ("ok review", 0)],
            antigravity_models=(GEM1, OPUS),
        )
        assert error is None and response.model_used == OPUS
        assert models == [GEM1, OPUS]
        assert sleeps == []
        assert _gemini_marked()
        assert agy.quota_memory_for_current_run().entry("gemini").source == "cooldown"
    assert f'Antigravity model "{GEM1}" out of credits; falling back to "{OPUS}"' in logs
    assert any("skipping quota group gemini" in line for line in logs)
    assert not any("non-retryable" in line for line in logs)


def test_credits_exhaustion_on_single_model_chain_is_transient_unavailable(tmp_path):
    _r, error, models, sleeps, _logs = _turn(
        tmp_path, [_fail(CREDITS_OUTPUT)], antigravity_models=(GEM1,)
    )
    assert type(error) is AgentInvocationError
    assert error.failure_category == "transient"
    assert str(error).startswith("Antigravity unavailable on all models")
    assert "check credentials or billing" not in str(error)
    assert models == [GEM1] and sleeps == []


@pytest.mark.parametrize(
    "output",
    [
        "error: invalid api key",
        'AGY_ERROR: {"status":"X","error_code":401,"message":"nope"}',
        'AGY_ERROR: {"status":"X","error_code":403,"message":"nope"}',
        "Error: request timed out (code 429)\n"
        'AGY_ERROR: {"status":"X","error_code":401,"message":"timeout"}',
        "error: Your AI credits balance is too low to continue.\n"
        'AGY_ERROR: {"status":"X","error_code":401,"retryable":true}',
    ],
)
def test_auth_failure_is_one_invocation_without_fallback(tmp_path, output):
    with workdir_claims.workdir_claim_scope():
        _r, error, models, sleeps, logs = _turn(
            tmp_path, [_fail(output), ("never", 0)],
            agent_max_retries=2, antigravity_models=(GEM1, OPUS),
        )
        assert not _gemini_marked()
        assert agy.quota_memory_for_current_run().entry("claude") is None
    assert type(error) is AgentInvocationError
    assert error.failure_category == "non-retryable"
    assert models == [GEM1] and sleeps == []
    assert not any("out of credits" in line for line in logs)


def test_transcript_only_credits_wording_records_no_credits_cooldown(tmp_path):
    output = (
        "The issue quotes: Your AI credits balance is too low to continue.\n"
        "Reviewing the classifier next.\n"
        f"Error: high traffic\n{RETRYABLE_429}"
    )
    with workdir_claims.workdir_claim_scope():
        _r, error, models, sleeps, logs = _turn(
            tmp_path, [_fail(output), ("never", 0)], antigravity_models=(GEM1, OPUS),
        )
        assert not _gemini_marked()
    assert error is not None and error.failure_category == "non-retryable"
    assert models == [GEM1] and sleeps == []
    assert not any("out of credits" in line for line in logs)


def test_dirty_checkout_beats_credits_exemption_in_turn(tmp_path):
    with workdir_claims.workdir_claim_scope():
        _r, error, models, sleeps, logs = _turn(
            tmp_path, [_fail(f"Error: dirty checkout\n{CREDITS_OUTPUT}"), ("never", 0)],
            antigravity_models=(GEM1, OPUS),
        )
        assert not _gemini_marked()
    assert error is not None and error.failure_category == "non-retryable"
    assert models == [GEM1] and sleeps == []
    assert not any("out of credits" in line for line in logs)


def test_nonterminal_status_403_json_keeps_capacity_path(tmp_path):
    output = "\n".join([
        '{"message":"Resource not accessible","status":"403"}',
        "Reading the diff to draft the review...",
        "Checking tests next.",
        "Still drafting.",
        "Almost there.",
        "Error: RESOURCE_EXHAUSTED",
        RETRYABLE_429,
    ])
    with workdir_claims.workdir_claim_scope():
        response, error, models, sleeps, logs = _turn(
            tmp_path, [_fail(output), ("ok review", 0)], antigravity_models=(GEM1, OPUS),
        )
        assert _gemini_marked()
    assert error is None and response.model_used == OPUS
    assert models == [GEM1, OPUS] and sleeps == []
    assert any("capacity failure after" in line for line in logs)


@pytest.mark.parametrize(
    "auth_text",
    ["unauthorized", 'AGY_ERROR: {"status":"X","error_code":401,"message":"nope"}'],
)
def test_typed_target_exec_retry_outranks_antigravity_auth_verdict(tmp_path, auth_text):
    from coding_review_agent_loop.agents.base import AgentResult
    from coding_review_agent_loop.containment import ContainmentEvidence
    from coding_review_agent_loop.runner import CommandResult

    class TargetExecRunner(FakeRunner):
        def target_exec_retry_decision(self, command, errno_value):
            return True, "agy temporarily unavailable/being replaced after successful preflight"

    failed = AgentResult(
        text="",
        raw_output=auth_text,
        returncode=127,
        model_used=GEM1,
        command_result=CommandResult(["agy", "exec"], tmp_path, "", "", 127),
        containment=ContainmentEvidence(
            backend="systemd-cgroup-v2",
            termination_cause="target-exec-error",
            target_exec_errno=2,
            cleanup_confirmed=True,
            diagnostics=("target exec failed (2): missing",),
        ),
    )
    ok = AgentResult(text="ok review", raw_output="ok review", returncode=0, model_used=GEM1)
    from coding_review_agent_loop import response_validation

    with patch_object_in(
        (validated_agent, response_validation), "run_agent_result", side_effect=[failed, ok],
    ) as run_mock:
        response, error, _models, sleeps, logs = _turn(
            tmp_path, [], runner=TargetExecRunner(), antigravity_models=(GEM1, OPUS),
        )
    assert error is None and response.text == "ok review"
    assert run_mock.call_count == 2
    assert len(sleeps) == 1
    assert any("transient failure" in line for line in logs)
    assert not any("non-retryable" in line for line in logs)


def test_bare_json_retryable_credits_frame_causes_no_quota_marking_or_fallback(tmp_path):
    output = (
        "error: Your AI credits balance is too low to continue.\n"
        '{"message":"credits balance too low","code":402,"retryable":true}'
    )
    with workdir_claims.workdir_claim_scope():
        _r, error, models, sleeps, logs = _turn(
            tmp_path, [_fail(output), ("never", 0)], antigravity_models=(GEM1, OPUS),
        )
        assert not _gemini_marked()
    assert error is not None and error.failure_category == "non-retryable"
    assert models == [GEM1] and sleeps == []
    assert not any("out of credits" in line for line in logs)


# --- #1601: opt-in wait for a verified all-groups parsed reset --------------

SIX_HOURS = 6 * 3600


@pytest.fixture
def advancing_sleeper(monkeypatch):
    """Injected quota-wait sleeper that advances the quota-memory clock."""
    from coding_review_agent_loop import quota_wait

    clock = FakeClock()
    slept: list[float] = []

    def sleep(seconds):
        slept.append(seconds)
        clock.now += seconds

    wait_logs: list[str] = []
    monkeypatch.setattr(quota_wait, "_sleeper", sleep)
    monkeypatch.setattr(quota_wait, "log", lambda _c, m: wait_logs.append(m))
    return clock, slept, wait_logs


def test_wait_pre_dispatch_all_parsed_builds_the_normal_first_invocation(tmp_path, advancing_sleeper):
    clock, slept, wait_logs = advancing_sleeper
    with patch.object(agy, "_now", clock), workdir_claims.workdir_claim_scope(), \
            patch.object(validated_agent, "run_agent_result", wraps=validated_agent.run_agent_result) as invoke:
        memory = agy.quota_memory_for_current_run()
        memory.mark_exhausted("gemini", 3600, "parsed", QUOTA_4H, 0.0, cooldown_seconds=600)
        memory.mark_exhausted("claude", 7200, "parsed", QUOTA_4H, 0.0, cooldown_seconds=600)
        response, error, models, sleeps, logs = _turn(
            tmp_path, [("first", 0)], agent_max_retries=0, wait_for_quota_reset_seconds=SIX_HOURS,
        )
    assert error is None and response.text == "first"
    assert models == [GEM1]  # zero dispatches before the wait, one after
    assert invoke.call_count == 1
    assert invoke.call_args.kwargs["prompt"] == "PROMPT"
    assert invoke.call_args.kwargs["session_id"] is None
    assert "attempt_suffix" not in invoke.call_args.kwargs  # not a quota replay
    assert sum(slept) == 3600 + 60 and sleeps == []
    assert any(
        "quota exhausted; waiting until" in line and "(wait 1 of 2)" in line
        for line in wait_logs
    )


def test_wait_pre_dispatch_keeps_a_pending_judgement_reask(tmp_path, advancing_sleeper):
    from coding_review_agent_loop.errors import MissingJudgementFieldError

    clock, slept, wait_logs = advancing_sleeper
    config = make_config(tmp_path, agent_max_retries=0, wait_for_quota_reset_seconds=SIX_HOURS)
    runner = FakeRunner(antigravity_outputs=[("no verdict", 0), ("with verdict", 0)])
    calls = []

    def validate(text):
        calls.append(text)
        if text == "no verdict":
            # A peer reviewer thread exhausts the remaining group meanwhile.
            agy.quota_memory_for_current_run().mark_exhausted(
                "claude", 3600, "parsed", QUOTA_4H, 0.0, cooldown_seconds=600
            )
            raise MissingJudgementFieldError(
                "missing verdict", field_path="verdict", allowed_values=("approved", "blocking")
            )
        return text

    with patch.object(agy, "_now", clock), workdir_claims.workdir_claim_scope(), \
            patch.object(validated_agent, "run_agent_result", wraps=validated_agent.run_agent_result) as invoke:
        agy.quota_memory_for_current_run().mark_exhausted(
            "gemini", 1800, "parsed", QUOTA_4H, 0.0, cooldown_seconds=600
        )
        response = validated_agent._run_validated_agent(
            runner, agent="antigravity", config=config, prompt="PROMPT", marker_description="none",
            validate=validate, session_id="S1", reask_on_missing_judgement_field=True,
        )
    assert response.text == "with verdict"
    assert invoke.call_count == 2
    first, second = (call.kwargs for call in invoke.call_args_list)
    assert first["prompt"] == "PROMPT"
    assert second["prompt"] != first["prompt"]  # the pending re-ask, not a stale replay
    assert second["prompt"].startswith("PROMPT") and "verdict" in second["prompt"][len("PROMPT"):]
    assert second["session_id"] == first["session_id"] == "S1"
    assert sum(slept) == 1800 + 60


def test_wait_after_fresh_two_group_shared_limit_replays_failed_invocation(tmp_path, advancing_sleeper):
    clock, slept, wait_logs = advancing_sleeper
    with patch.object(agy, "_now", clock), workdir_claims.workdir_claim_scope(), \
            patch.object(validated_agent, "run_agent_result", wraps=validated_agent.run_agent_result) as invoke:
        response, error, models, sleeps, _logs = _turn(
            tmp_path, [_fail(QUOTA_4H), _fail(QUOTA_4H.replace("4h", "5h")), ("after reset", 0)],
            agent_max_retries=0, wait_for_quota_reset_seconds=SIX_HOURS,
        )
    assert error is None and response.text == "after reset"
    # Waits for the earliest (Gemini) reset, then replays on that group.
    assert models == [GEM1, OPUS, GEM1]
    assert sum(slept) == 14400 + 60 and sleeps == []
    prompts = [call.kwargs["prompt"] for call in invoke.call_args_list]
    assert prompts[2] == prompts[1]


@pytest.mark.parametrize("case", ["untried-group", "cooldown-only", "beyond-duration"])
def test_no_wait_unless_every_group_is_parsed_within_duration(tmp_path, advancing_sleeper, case):
    clock, slept, wait_logs = advancing_sleeper
    with patch.object(agy, "_now", clock), workdir_claims.workdir_claim_scope():
        if case == "untried-group":
            _r, error, models, _s, _l = _turn(
                tmp_path, [_fail(QUOTA_4H), _fail(QUOTA_4H), ("never", 0)],
                antigravity_models=(GEM1, OPUS, "Model X"), wait_for_quota_reset_seconds=SIX_HOURS,
            )
            assert type(error) is AgentInvocationError and "failed the same way" in str(error)
        elif case == "cooldown-only":
            agy.quota_memory_for_current_run().mark_exhausted(
                "gemini", None, "cooldown", LIVE_SAMPLE, 0.0, cooldown_seconds=600
            )
            _r, error, models, _s, _l = _turn(
                tmp_path, [_fail(QUOTA_4H)], wait_for_quota_reset_seconds=SIX_HOURS,
            )
            assert type(error) is AgentInvocationError
            assert str(error).startswith("Antigravity unavailable on all models")
        else:
            _r, error, models, _s, _l = _turn(
                tmp_path, [_fail(QUOTA_4H), _fail(QUOTA_4H)], wait_for_quota_reset_seconds=3600,
            )
            assert isinstance(error, QuotaResetExceededError)
    assert slept == []


# --- silent quota failure read from agy's own CLI log (#1604) -----------------

SILENT = (AGY_SILENT_WARNING + "\n", 0)
AGY_429_LOG = AGY_LOG_STARTUP_LINES + AGY_LOG_429_LINE + "\n"


def _live_agy():
    """The adapter module the backend runs from (another test re-imports it)."""
    import importlib

    return importlib.import_module("coding_review_agent_loop.agents.antigravity")


def _strict_validate(text):
    if not text.startswith("ok"):
        raise AgentLoopError("missing required format")
    return text


def _silent_turn(tmp_path, log_dir, outputs, agy_logs, *, spawn_wall_time=None, **overrides):
    runner = AgyLogRunner(
        antigravity_outputs=list(outputs), log_dir=log_dir, agy_logs=agy_logs,
        spawn_wall_time=spawn_wall_time,
    )
    config = make_config(tmp_path, agent_max_retries=overrides.pop("agent_max_retries", 1), **overrides)
    logs: list[str] = []
    repairs: list[object] = []

    def record_repair(*_a, **_k):
        repairs.append(1)
        return None

    try:
        with patch("coding_review_agent_loop.validated_agent.log", lambda _c, m: logs.append(m)), \
                patch.object(_live_agy(), "log", lambda _c, m: logs.append(m)), \
                patch("coding_review_agent_loop.validated_agent.attempt_repair", record_repair):
            response = validated_agent._run_validated_agent(
                runner, agent="antigravity", config=config, prompt="PROMPT",
                marker_description="none", validate=_strict_validate, use_repair=True,
            )
        error = None
    except (AgentInvocationError, QuotaResetExceededError) as exc:
        response, error = None, exc
    cmds = [c for c, _cwd in runner.commands if c[:1] == ["agy"]]
    models = [c[c.index("--model") + 1] for c in cmds if "--model" in c]
    return response, error, models, logs, repairs, runner


def test_silent_attempt_with_agy_log_429_falls_back_in_the_same_turn(tmp_path, _isolated_agy_log_dir):
    with workdir_claims.workdir_claim_scope():
        response, error, models, logs, repairs, runner = _silent_turn(
            tmp_path, _isolated_agy_log_dir, [SILENT, ("ok review", 0)], [AGY_429_LOG, None],
            antigravity_models=(GEM1, OPUS),
        )
        assert error is None and response.model_used == OPUS
        assert models == [GEM1, OPUS]
        entry = agy.quota_memory_for_current_run().entry("gemini")
        assert entry is not None and entry.source == "cooldown"
    assert repairs == []
    name = runner.written_agy_logs[0].name
    assert (
        f"Antigravity printed no output; agy log {name} shows RESOURCE_EXHAUSTED (code 429); "
        "treating as quota exhaustion"
    ) in logs
    assert any("skipping quota group gemini" in line for line in logs)
    assert not any("retrying in" in line for line in logs)


def test_silent_agy_log_429_on_every_group_stops_unavailable_on_all_models(tmp_path, _isolated_agy_log_dir):
    with workdir_claims.workdir_claim_scope():
        _r, error, models, _logs, repairs, _runner = _silent_turn(
            tmp_path, _isolated_agy_log_dir, [SILENT, SILENT], [AGY_429_LOG, AGY_429_LOG],
            antigravity_models=(GEM1, OPUS),
        )
    assert type(error) is AgentInvocationError
    assert str(error).startswith("Antigravity unavailable on all models")
    assert error.failure_category == "transient"
    assert models == [GEM1, OPUS]
    assert repairs == []


@pytest.mark.parametrize(
    "agy_log", [None, AGY_LOG_STARTUP_LINES + "I1022 02:03:52.000000 1 run.go:1] done\n"],
    ids=["no-log", "log-without-429"],
)
def test_silent_attempt_without_usable_agy_log_keeps_format_failure_path(
    tmp_path, _isolated_agy_log_dir, agy_log
):
    with workdir_claims.workdir_claim_scope():
        _r, error, models, logs, repairs, runner = _silent_turn(
            tmp_path, _isolated_agy_log_dir, [SILENT], [agy_log],
            antigravity_models=(GEM1, OPUS), agent_max_retries=0,
        )
        assert agy.quota_memory_for_current_run().entry("gemini") is None
    assert isinstance(error, AgentInvocationError)
    assert "Antigravity unavailable" not in str(error)
    assert error.failure_category != "transient"
    assert models == [GEM1]
    assert repairs == [1]
    assert not any("treating as quota exhaustion" in line for line in logs)
    if agy_log is None:
        assert any("no new matching agy log" in line for line in logs)
        assert "agy log:" not in str(error)
    else:
        path = runner.written_agy_logs[0]
        assert any(f"agy log {path.name} has no RESOURCE_EXHAUSTED 429 line" in line for line in logs)
        assert f"agy log: {path}" in str(error)


def test_silent_attempt_with_unreadable_agy_log_dir_keeps_today_behavior(tmp_path, monkeypatch):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setattr(_live_agy(), "_antigravity_log_dir", lambda: blocker / "log")
    with workdir_claims.workdir_claim_scope():
        _r, error, models, _logs, repairs, _runner = _silent_turn(
            tmp_path, tmp_path / "unused", [SILENT], [None],
            antigravity_models=(GEM1, OPUS), agent_max_retries=0,
        )
        assert agy.quota_memory_for_current_run().entry("gemini") is None
    assert isinstance(error, AgentInvocationError)
    assert models == [GEM1] and repairs == [1]


def test_agy_log_lookup_exception_never_fails_the_turn(tmp_path, _isolated_agy_log_dir, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("lookup exploded")

    monkeypatch.setattr(_live_agy(), "find_antigravity_attempt_log", boom)
    with workdir_claims.workdir_claim_scope():
        _r, error, models, logs, repairs, _runner = _silent_turn(
            tmp_path, _isolated_agy_log_dir, [SILENT], [AGY_429_LOG],
            antigravity_models=(GEM1, OPUS), agent_max_retries=0,
        )
        assert agy.quota_memory_for_current_run().entry("gemini") is None
    assert "lookup exploded" not in str(error)
    assert any("agy log lookup failed: lookup exploded" in line for line in logs)
    assert models == [GEM1] and repairs == [1]


def test_prior_attempt_429_log_near_spawn_leaves_quota_unchanged(tmp_path, _isolated_agy_log_dir):
    import os
    import time

    spawn = time.time()
    prior = _isolated_agy_log_dir / agy_log_name(spawn - 1)
    prior.write_text(AGY_429_LOG, encoding="utf-8")
    os.utime(prior, (spawn - 0.5, spawn - 0.5))
    # Inside both tolerance windows: only the pre-spawn inventory rejects it.
    assert agy.find_antigravity_attempt_log(
        _isolated_agy_log_dir, frozenset(), spawn, spawn + 1
    ).log_path == prior
    with workdir_claims.workdir_claim_scope():
        _r, error, models, logs, repairs, _runner = _silent_turn(
            tmp_path, _isolated_agy_log_dir, [SILENT], [None],
            spawn_wall_time=spawn, antigravity_models=(GEM1, OPUS), agent_max_retries=0,
        )
        assert agy.quota_memory_for_current_run().entry("gemini") is None
        assert agy.quota_memory_for_current_run().entry("claude") is None
    assert models == [GEM1]
    assert "Antigravity unavailable" not in str(error)
    assert repairs == [1]
    assert not any("treating as quota exhaustion" in line for line in logs)
    assert any("no new matching agy log" in line for line in logs)


_AGY_LONG_RESET_LOG = AGY_LOG_STARTUP_LINES + AGY_LOG_429_LINE + " retryDelay: 14400s\n"
_AGY_NO_429_LOG = AGY_LOG_STARTUP_LINES + "I1022 02:03:52.000000 1 run.go:1] done\n"


@pytest.mark.parametrize(
    "agy_logs,error_type,prefix,max_retries",
    [
        ([AGY_429_LOG, AGY_429_LOG], AgentInvocationError, "Antigravity unavailable on all models", 1),
        ([_AGY_LONG_RESET_LOG, _AGY_LONG_RESET_LOG], QuotaResetExceededError, "", 1),
        ([_AGY_NO_429_LOG], AgentInvocationError, "", 0),
    ],
    ids=["all-models-stop", "parsed-reset-stop", "ordinary-failure"],
)
def test_every_terminal_error_names_each_located_agy_log(
    tmp_path, _isolated_agy_log_dir, agy_logs, error_type, prefix, max_retries
):
    with workdir_claims.workdir_claim_scope():
        _r, error, _models, _logs, _repairs, runner = _silent_turn(
            tmp_path, _isolated_agy_log_dir, [SILENT] * len(agy_logs), agy_logs,
            antigravity_models=(GEM1, OPUS), agent_max_retries=max_retries,
        )
    assert type(error) is error_type
    assert str(error).startswith(prefix)
    assert len(runner.written_agy_logs) == len(agy_logs)
    for path in runner.written_agy_logs:
        assert f"agy log: {path}." in str(error)


# --- the post-exit agy log lookup is taken under the settings lock (#1614) ----

def _spy_attempt_log_lookup(monkeypatch, on_call):
    live = _live_agy()
    real = live.find_antigravity_attempt_log
    calls: list[tuple] = []

    def spy(*args, **kwargs):
        calls.append(args)
        on_call()
        return real(*args, **kwargs)

    monkeypatch.setattr(live, "find_antigravity_attempt_log", spy)
    return calls


def test_silent_attempt_lists_agy_logs_while_the_settings_lock_is_held(
    tmp_path, _isolated_agy_log_dir, monkeypatch
):
    import fcntl

    live = _live_agy()
    settings_path = tmp_path / "agy-settings" / "settings.json"
    monkeypatch.setattr(live, "_antigravity_settings_path", lambda: settings_path)
    probes: list[str] = []

    def probe_lock():
        with settings_path.with_suffix(".json.lock").open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                probes.append("held")
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
                probes.append("free")

    calls = _spy_attempt_log_lookup(monkeypatch, probe_lock)
    with workdir_claims.workdir_claim_scope():
        response, error, models, logs, _repairs, runner = _silent_turn(
            tmp_path, _isolated_agy_log_dir, [SILENT, ("ok review", 0)], [AGY_429_LOG, None],
            antigravity_models=(GEM1, OPUS),
        )
        assert agy.quota_memory_for_current_run().entry("gemini") is not None
    assert error is None and response.model_used == OPUS
    assert len(calls) == 1 and probes == ["held"]
    assert any(
        f"agy log {runner.written_agy_logs[0].name} shows RESOURCE_EXHAUSTED" in line
        for line in logs
    )


def test_agy_log_created_after_the_locked_lookup_is_never_a_candidate(
    tmp_path, _isolated_agy_log_dir, monkeypatch
):
    import os
    import time

    live = _live_agy()
    monkeypatch.setattr(
        live, "_antigravity_settings_path", lambda: tmp_path / "agy-settings" / "settings.json"
    )
    spawn = time.time()
    intruder = _isolated_agy_log_dir / agy_log_name(spawn + 1)
    real_remove = live._remove_own_settings_lock_holder

    def plant_then_release(*args, **kwargs):
        # Another agent-loop turn's 429 log, inside this attempt's windows,
        # appears after the lookup but before the settings lock is released.
        intruder.write_text(AGY_429_LOG, encoding="utf-8")
        os.utime(intruder, (spawn + 0.5, spawn + 0.5))
        return real_remove(*args, **kwargs)

    monkeypatch.setattr(live, "_remove_own_settings_lock_holder", plant_then_release)
    with workdir_claims.workdir_claim_scope():
        _r, error, models, logs, repairs, _runner = _silent_turn(
            tmp_path, _isolated_agy_log_dir, [SILENT], [None],
            spawn_wall_time=spawn, antigravity_models=(GEM1, OPUS), agent_max_retries=0,
        )
        assert agy.quota_memory_for_current_run().entry("gemini") is None
    assert intruder.exists()
    # The same file would have been the single candidate for a later listing.
    assert live.find_antigravity_attempt_log(
        _isolated_agy_log_dir, frozenset(), spawn, spawn + 1
    ).log_path == intruder
    assert isinstance(error, AgentInvocationError)
    assert "Antigravity unavailable" not in str(error)
    assert intruder.name not in str(error)
    assert models == [GEM1] and repairs == [1]
    assert not any("treating as quota exhaustion" in line for line in logs)
    assert any("no new matching agy log" in line for line in logs)


def test_non_silent_turn_does_not_list_agy_logs(tmp_path, _isolated_agy_log_dir, monkeypatch):
    calls = _spy_attempt_log_lookup(monkeypatch, lambda: None)
    with workdir_claims.workdir_claim_scope():
        response, error, models, _logs, repairs, _runner = _silent_turn(
            tmp_path, _isolated_agy_log_dir, [("ok review", 0)], [None],
            antigravity_models=(GEM1, OPUS),
        )
    assert error is None and response.model_used == GEM1
    assert models == [GEM1] and repairs == []
    assert calls == []
