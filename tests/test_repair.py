from agent_loop_helpers import *  # noqa: F403


from coding_review_agent_loop.repair import (
    _REPAIR_PROMPT,
    _build_repair_prompt,
    attempt_envelope_normalization,
    attempt_repair,
    execute_repair,
)
from coding_review_agent_loop.protocol import validate_structured_discuss_answer


def test_answer_repair_prompt_has_mode_specific_schema_and_examples():
    prompt = _build_repair_prompt(
        '{"kind":"discuss_review","outcome":"implement","rationale":"Use an adapter."}',
        expected_kind="discuss_answer",
    )
    assert '"kind": "discuss_answer"' in prompt
    assert '"position": "needs-human"' in prompt
    assert '"open_questions"' in prompt
    assert "Do not repair answer mode into `discuss_review`" in prompt
    assert "split_proposals" in prompt


def test_answer_repair_shape_preserves_answer_and_rejects_triage_fields():
    valid = json.dumps({
        "schema_version": 1, "kind": "discuss_answer", "position": "answer",
        "answer": "Use an adapter.", "rationale": "It isolates policy.",
        "confidence": "medium", "open_questions": [],
    }) + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- Reviewer"
    assert validate_structured_discuss_answer(valid, reviewer="Reviewer").position == "answer"
    malformed = json.dumps({
        "schema_version": 1, "kind": "discuss_answer", "position": "answer",
        "answer": "Use an adapter.", "rationale": "Reason", "confidence": "medium",
        "open_questions": [], "outcome": "implement",
    }) + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- Reviewer"
    with pytest.raises(AgentLoopError, match="unknown field.*outcome"):
        validate_structured_discuss_answer(malformed, reviewer="Reviewer")

def test_envelope_normalization_duplicate_pr_state_footer_preserves_dispositions():
    raw = (
        structured_pr_review(
            state="approved",
            reviewer="Google Gemini",
            prior_item_dispositions=[
                {"item_id": "item-1", "disposition": "resolved"},
                {"item_id": "item-2", "disposition": "future"},
                {"item_id": "item-3", "disposition": "resolved"},
            ],
        )
        + "\n\n<!-- AGENT_STATE: approved -->"
    )

    normalized = attempt_envelope_normalization(raw, expected_kind="pr_review")

    assert normalized is not None
    parsed = parse_structured_pr_review(normalized, reviewer="Google Gemini")
    assert parsed is not None
    assert {
        disposition.item_id: disposition.disposition
        for disposition in parsed.dispositions
    } == {
        "item-1": "resolved",
        "item-2": "future",
        "item-3": "resolved",
    }
    assert normalized.count("<!-- AGENT_STATE: approved -->") == 1

def test_envelope_normalization_trailing_prose_after_signature():
    raw = (
        structured_pr_review(state="approved", reviewer="Google Gemini")
        + "\n\nExtra prose after the signature."
    )

    normalized = attempt_envelope_normalization(raw, expected_kind="pr_review")

    assert normalized is not None
    assert "Extra prose" not in normalized
    assert parse_structured_pr_review(normalized, reviewer="Google Gemini") is not None

def test_envelope_normalization_duplicate_plan_state_footer():
    raw = (
        structured_plan_review(
            state="approved",
            reviewer="Google Gemini",
            prior_plan_item_dispositions=[
                {"item_id": "item-1", "disposition": "resolved"},
            ],
        )
        + "\n\n<!-- AGENT_PLAN_STATE: approved -->"
    )

    normalized = attempt_envelope_normalization(raw, expected_kind="plan_review")

    assert normalized is not None
    parsed = parse_structured_plan_review(normalized, reviewer="Google Gemini")
    assert parsed is not None
    assert [(item.item_id, item.disposition) for item in parsed.dispositions] == [
        ("item-1", "resolved")
    ]
    assert normalized.count("<!-- AGENT_PLAN_STATE: approved -->") == 1

def test_envelope_normalization_preserves_hr_resolved_before_footer_for_reviews():
    for expected_kind, raw, parser in (
        (
            "pr_review",
            structured_pr_review(
                state="approved",
                reviewer="Google Gemini",
                human_requirements_resolved=True,
            ),
            parse_structured_pr_review,
        ),
        (
            "plan_review",
            structured_plan_review(
                state="approved",
                reviewer="Google Gemini",
                human_requirements_resolved=True,
            ),
            parse_structured_plan_review,
        ),
    ):
        normalized = attempt_envelope_normalization(
            raw + "\n\nTrailing prose.",
            expected_kind=expected_kind,
        )

        assert normalized is not None
        assert "<!-- HUMAN_REQUIREMENTS_RESOLVED -->" in normalized
        assert parser(normalized, reviewer="Google Gemini") is not None

def test_envelope_normalization_preserves_hr_resolved_after_footer_for_pr_review():
    raw = (
        structured_pr_review(state="approved", reviewer="Google Gemini").replace(
            "\n-- Google Gemini",
            "\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n-- Google Gemini",
        )
        + "\n\n<!-- AGENT_STATE: approved -->"
    )

    normalized = attempt_envelope_normalization(raw, expected_kind="pr_review")

    assert normalized is not None
    assert "<!-- HUMAN_REQUIREMENTS_RESOLVED -->" in normalized
    assert parse_structured_pr_review(normalized, reviewer="Google Gemini") is not None

def test_envelope_normalization_plan_review_drops_after_footer_hr_marker():
    raw = (
        structured_plan_review(
            state="approved",
            reviewer="Google Gemini",
            prior_plan_item_dispositions=[
                {"item_id": "item-1", "disposition": "resolved"},
                {"item_id": "item-2", "disposition": "future"},
            ],
        ).replace(
            "\n-- Google Gemini",
            "\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n-- Google Gemini",
        )
        + "\n\n<!-- AGENT_PLAN_STATE: approved -->\nTrailing prose."
    )

    normalized = attempt_envelope_normalization(raw, expected_kind="plan_review")

    assert normalized is not None
    assert "<!-- HUMAN_REQUIREMENTS_RESOLVED -->" not in normalized
    parsed = parse_structured_plan_review(normalized, reviewer="Google Gemini")
    assert parsed is not None
    assert {
        disposition.item_id: disposition.disposition
        for disposition in parsed.dispositions
    } == {
        "item-1": "resolved",
        "item-2": "future",
    }

@pytest.mark.parametrize(
    "expected_kind",
    ["pr_review", "plan_review", "coder_followup", "plan_revision"],
)
def test_envelope_normalization_recovers_reversed_signature_before_footer(expected_kind):
    """Signature placed before the AGENT_STATE footer is reordered deterministically."""
    if expected_kind == "pr_review":
        json_obj = {
            "schema_version": 1,
            "kind": "pr_review",
            "state": "blocking",
            "summary": "Found issues.",
            "blocking_items": ["Fix the bug"],
            "same_pr_followups": [],
            "future_followups": [],
            "prior_item_dispositions": [],
        }
        footer = "<!-- AGENT_STATE: blocking -->"
        signature = "-- OpenAI Codex"
    elif expected_kind == "plan_review":
        json_obj = {
            "schema_version": 1,
            "kind": "plan_review",
            "state": "blocking",
            "summary": "Plan needs work.",
            "blocking_plan_issues": ["Missing tests"],
            "same_plan_followups": [],
            "future_followups": [],
            "prior_plan_item_dispositions": [],
        }
        footer = "<!-- AGENT_PLAN_STATE: blocking -->"
        signature = "-- OpenAI Codex"
    elif expected_kind == "coder_followup":
        json_obj = {
            "schema_version": 1,
            "kind": "coder_followup",
            "state": "approved",
            "summary": "All done.",
            "addressed_items": [],
            "remaining_items": [],
            "addressed_item_notes": {},
            "remaining_item_notes": {},
            "human_requirements": {"addressed_ids": [], "checked_discussion_directly": False},
        }
        footer = "<!-- AGENT_STATE: approved -->"
        signature = "-- Anthropic Claude"
    else:  # plan_revision
        json_obj = {
            "schema_version": 1,
            "kind": "plan_revision",
            "state": "blocking",
            "summary": "Revised.",
            "prior_plan_item_dispositions": [],
            "plan_steps": ["Do the thing."],
        }
        footer = "<!-- AGENT_PLAN_STATE: blocking -->"
        signature = "-- Anthropic Claude"

    # Reversed: signature comes before footer
    raw = f"{json.dumps(json_obj)}\n{signature}\n{footer}"
    normalized = attempt_envelope_normalization(raw, expected_kind=expected_kind)

    assert normalized is not None, f"Expected normalization to succeed for {expected_kind}"
    # Canonical order: JSON → footer → signature
    assert normalized.endswith(f"\n{footer}\n{signature}"), (
        f"Expected footer before signature; got:\n{normalized}"
    )
    assert normalized.index(footer) < normalized.index(signature)

def test_envelope_normalization_reversed_signature_state_mismatch_returns_none():
    """Reversed-signature recovery must not proceed when state mismatches."""
    json_obj = {
        "schema_version": 1,
        "kind": "pr_review",
        "state": "approved",  # JSON says approved
        "summary": "All good.",
        "blocking_items": [],
        "same_pr_followups": [],
        "future_followups": [],
        "prior_item_dispositions": [],
    }
    # Footer says blocking — mismatch
    raw = f"{json.dumps(json_obj)}\n-- OpenAI Codex\n<!-- AGENT_STATE: blocking -->"

    assert attempt_envelope_normalization(raw, expected_kind="pr_review") is None

def test_envelope_normalization_reversed_signature_with_extra_prose_returns_none():
    """Reversed-signature recovery must not fire when extra prose precedes the footer."""
    json_obj = {
        "schema_version": 1,
        "kind": "pr_review",
        "state": "blocking",
        "summary": "Issues found.",
        "blocking_items": ["Fix it"],
        "same_pr_followups": [],
        "future_followups": [],
        "prior_item_dispositions": [],
    }
    # Extra prose between JSON and signature — not a clean reversed envelope
    raw = (
        f"{json.dumps(json_obj)}\nExtra prose here.\n"
        "-- OpenAI Codex\n<!-- AGENT_STATE: blocking -->"
    )

    assert attempt_envelope_normalization(raw, expected_kind="pr_review") is None

def test_envelope_normalization_returns_none_when_no_footer():
    raw = json.dumps(
        {
            "schema_version": 1,
            "kind": "pr_review",
            "state": "approved",
            "summary": "Review complete.",
            "prior_item_dispositions": [],
        }
    )

    assert attempt_envelope_normalization(raw, expected_kind="pr_review") is None

def test_envelope_normalization_returns_none_when_no_signature():
    raw = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "Review complete.",
                "prior_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_STATE: approved -->"
    )

    assert attempt_envelope_normalization(raw, expected_kind="pr_review") is None

def test_envelope_normalization_returns_none_when_json_invalid():
    raw = '{"schema_version": 1,\n<!-- AGENT_STATE: approved -->\n-- Google Gemini'

    assert attempt_envelope_normalization(raw, expected_kind="pr_review") is None

def test_envelope_normalization_semantic_defect_still_fails_validate():
    raw = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "Review complete.",
            }
        )
        + "\n<!-- AGENT_STATE: approved -->\n-- Google Gemini\n\nTrailing prose."
    )

    normalized = attempt_envelope_normalization(raw, expected_kind="pr_review")

    assert normalized is not None
    with pytest.raises(AgentLoopError):
        parse_structured_pr_review(normalized, reviewer="Google Gemini")

def test_attempt_repair_returns_none_when_subprocess_fails(monkeypatch):
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=mock_result):
        result = attempt_repair("some malformed review text", "gemini")
    assert result is None

def test_attempt_repair_returns_none_when_subprocess_raises(monkeypatch):
    with patch("coding_review_agent_loop.repair.subprocess.run", side_effect=FileNotFoundError("gemini not found")):
        result = attempt_repair("some malformed review text", "gemini")
    assert result is None

def test_attempt_repair_calls_cli_and_returns_text():
    repaired = (
        '{"schema_version":1,"kind":"pr_review","state":"approved","summary":"OK",'
        '"blocking_items":[],"same_pr_followups":[],"future_followups":[],'
        '"prior_item_dispositions":[]}\n<!-- AGENT_STATE: approved -->\n-- Gemini'
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = repaired

    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=mock_result) as mock_run:
        result = attempt_repair("malformed review", "gemini")

    assert result == repaired
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    cmd = call_args.args[0]
    assert cmd[0] == "gemini"
    assert "--model" in cmd
    assert "gemini-3.1-flash-lite" in cmd
    assert "--prompt" in cmd
    prompt_idx = cmd.index("--prompt")
    assert "malformed review" in cmd[prompt_idx + 1]

def test_attempt_repair_includes_expected_kind_instruction():
    repaired = (
        '{"schema_version":1,"kind":"plan_revision","state":"blocking","summary":"Revised.",'
        '"prior_plan_item_dispositions":[],"plan_steps":["Add tests."]}'
        "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Gemini"
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = repaired

    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=mock_result) as mock_run:
        result = attempt_repair(
            "malformed response mentioning human requirements and addressed_items",
            "gemini",
            expected_kind="plan_revision",
        )

    assert result == repaired
    cmd = mock_run.call_args.args[0]
    prompt = cmd[cmd.index("--prompt") + 1]
    assert "You MUST repair this response as `plan_revision`" in prompt
    assert "Output no other `kind` value" in prompt

def test_attempt_repair_format_d_marks_human_requirements_optional():
    assert "omit the `<!-- HUMAN_REQUIREMENTS_ADDRESSED -->` marker" in _REPAIR_PROMPT
    assert "the `### Human requirements` section from Format D" in _REPAIR_PROMPT


def test_repair_prompt_makes_research_intent_conditional_on_active_status():
    assert '`target` and `questions` are conditional intent fields' in _REPAIR_PROMPT
    assert 'omit both for `status: "not-needed"`' in _REPAIR_PROMPT
    assert 'For `status: "not-needed"`,' in _REPAIR_PROMPT

def test_attempt_repair_includes_prior_item_disposition_repair_context():
    repaired = structured_plan_revision(prior_plan_item_dispositions=[])
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = repaired

    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=mock_result) as mock_run:
        result = attempt_repair(
            "malformed plan revision",
            "gemini",
            expected_kind="plan_revision",
            allowed_prior_item_ids=["item-12"],
            unknown_prior_item_ids=["item-15", "item-18"],
            same_round_context="Same-round findings are informational only.",
        )

    assert result == repaired
    cmd = mock_run.call_args.args[0]
    prompt = cmd[cmd.index("--prompt") + 1]
    assert "Prior item disposition repair" in prompt
    assert "Allowed carried prior item IDs: item-12" in prompt
    assert "Unknown prior item disposition IDs to remove: item-15, item-18" in prompt
    assert "Same-round findings are informational only" in prompt

def test_attempt_repair_prior_item_disposition_context_is_not_duplicated():
    repaired = structured_plan_revision(prior_plan_item_dispositions=[])
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = repaired

    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=mock_result) as mock_run:
        attempt_repair(
            "malformed plan revision",
            "gemini",
            expected_kind="plan_revision",
            same_round_context="item-1 matches a same-round finding, not a carried prior item.",
        )

    cmd = mock_run.call_args.args[0]
    prompt = cmd[cmd.index("--prompt") + 1]
    assert prompt.count("item-1 matches a same-round finding") == 1
    assert "Context: item-1 matches a same-round finding" not in prompt

def test_attempt_repair_includes_coder_followup_required_item_ids():
    repaired = structured_coder_followup(
        state="approved",
        addressed_items=["item-8"],
        remaining_items=[],
        reviewer="Gemini",
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = repaired

    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=mock_result) as mock_run:
        result = attempt_repair(
            "### Human requirements\nAcknowledged.\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->",
            "gemini",
            expected_kind="coder_followup",
            unresolved_item_ids=["item-8"],
        )

    assert result == repaired
    cmd = mock_run.call_args.args[0]
    prompt = cmd[cmd.index("--prompt") + 1]
    assert "Required coder follow-up item IDs" in prompt
    assert "`item-8`" in prompt
    assert "exactly one of `addressed_items` or `remaining_items`" in prompt
    assert "HUMAN_REQUIREMENTS_ADDRESSED" in prompt
    assert "do not classify regular reviewer or orchestrator-injected item-N records" in prompt

def test_attempt_repair_includes_empty_surfaced_requirement_guidance():
    repaired = structured_coder_followup(
        state="blocking",
        addressed_items=["item-1"],
        remaining_items=[],
        human_requirement_ids=[],
        reviewer="Gemini",
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = repaired

    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=mock_result) as mock_run:
        result = attempt_repair(
            '"human_requirements":{"addressed_ids":["Issue #221 acceptance criteria"],'
            '"checked_discussion_directly":false}',
            "gemini",
            expected_kind="coder_followup",
            unresolved_item_ids=["item-1"],
            surfaced_requirement_ids=[],
        )

    assert result == repaired
    cmd = mock_run.call_args.args[0]
    prompt = cmd[cmd.index("--prompt") + 1]
    assert "Surfaced signed human requirement labels for coder follow-up" in prompt
    assert "- (none)" in prompt
    assert "set `human_requirements.addressed_ids` to `[]`" in prompt
    assert "Issue #221 acceptance criteria" in prompt
    assert '"addressed_ids": []' in prompt

def test_attempt_repair_includes_surfaced_requirement_labels_for_mixed_repairs():
    repaired = structured_coder_followup(
        state="blocking",
        addressed_items=["item-1"],
        remaining_items=[],
        human_requirement_ids=["Requirement 1"],
        reviewer="Gemini",
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = repaired

    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=mock_result) as mock_run:
        result = attempt_repair(
            '"addressed_ids":["Requirement 1","Issue #221 acceptance criteria"]',
            "gemini",
            expected_kind="coder_followup",
            unresolved_item_ids=["item-1"],
            surfaced_requirement_ids=["Requirement 1"],
        )

    assert result == repaired
    cmd = mock_run.call_args.args[0]
    prompt = cmd[cmd.index("--prompt") + 1]
    assert "`Requirement 1`" in prompt
    assert "keep [\"Requirement 1\"] and drop \"Issue #221 acceptance criteria\"" in prompt

def test_attempt_repair_rejects_unresolved_item_ids_for_non_coder_kind():
    with pytest.raises(ValueError, match="unresolved_item_ids"):
        attempt_repair(
            "malformed plan review",
            "gemini",
            expected_kind="plan_review",
            unresolved_item_ids=["item-1"],
        )

def test_attempt_repair_rejects_surfaced_requirement_ids_for_non_coder_kind():
    with pytest.raises(ValueError, match="surfaced_requirement_ids"):
        attempt_repair(
            "malformed plan review",
            "gemini",
            expected_kind="plan_review",
            surfaced_requirement_ids=["Requirement 1"],
        )

def test_attempt_repair_handles_json_wrapped_cli_output():
    repaired_text = (
        '{"schema_version":1,"kind":"pr_review","state":"approved","summary":"OK",'
        '"blocking_items":[],"same_pr_followups":[],"future_followups":[],'
        '"prior_item_dispositions":[]}\n<!-- AGENT_STATE: approved -->\n-- Gemini'
    )
    json_wrapped = json.dumps({"response": repaired_text, "session_id": "s1"})
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json_wrapped

    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=mock_result):
        result = attempt_repair("malformed review", "gemini")

    assert result == repaired_text

def test_repair_prompt_contains_raw_response_placeholder():
    assert "{raw_response}" in _REPAIR_PROMPT

def test_repair_prompt_substitution_leaves_json_examples_intact():
    raw = "some {curly} braces {in} the review text"
    substituted = _REPAIR_PROMPT.replace("{raw_response}", raw, 1)
    assert raw in substituted
    assert "{raw_response}" not in substituted
    assert "schema_version" in substituted

def test_execute_repair_defaults_to_isolated_antigravity_and_records_usage(
    tmp_path, monkeypatch
):
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.usage import RunUsageContext

    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)
    valid = structured_pr_review(state="approved", reviewer="Google Antigravity")
    captured = {}

    class RepairRunner(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            captured["args"] = list(args)
            captured["cwd"] = Path(cwd)
            captured["env"] = kwargs["env"]
            captured["timeout"] = kwargs["timeout_seconds"]
            captured["settings"] = json.loads(settings_file.read_text(encoding="utf-8"))
            captured["gemini_md"] = (Path(cwd) / "GEMINI.md").read_text(encoding="utf-8")
            assert not (Path(cwd) / ".git").exists()
            return super().run_with_log(args, cwd=cwd, **kwargs)

    config = make_config(
        tmp_path,
        antigravity_args=("--dangerously-skip-permissions",),
        repair_models=("Gemini 3 Flash",),
    )
    usage = RunUsageContext("run-1", tmp_path / "usage.json")
    repaired, marker, attempts = execute_repair(
        "malformed",
        runner=RepairRunner(antigravity_outputs=[(valid, 0)]),
        config=config,
        run_id="run-1",
        usage_context=usage,
        validate=lambda text: parse_structured_pr_review(
            text, reviewer="Google Antigravity"
        ),
        expected_kind="pr_review",
    )

    assert repaired == valid
    assert marker is not None
    assert attempts[0].model == "Gemini 3 Flash"
    assert captured["timeout"] == 120
    assert captured["env"]["AGENT_LOOP_WORKDIR"] == str(captured["cwd"])
    assert not captured["cwd"].exists()
    assert "--dangerously-skip-permissions" not in captured["args"]
    assert captured["settings"]["permissions"]["allow"] == []
    assert "Do not inspect files" in captured["gemini_md"]
    assert usage.records[0].role == "repair"
    assert usage.records[0].outcome == "succeeded"
    assert usage.records[0].validation_status == "validated"

def test_execute_repair_explicit_chain_falls_back_after_failure_and_invalid_output(
    tmp_path, monkeypatch
):
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.usage import RunUsageContext

    monkeypatch.setattr(
        agy_mod, "_antigravity_settings_path", lambda: tmp_path / "settings.json"
    )
    valid = structured_pr_review(state="approved", reviewer="Google Antigravity")
    config = make_config(
        tmp_path,
        repair_models=("Gemini 3 Flash", "Gemini 3.1 Pro (High)"),
    )
    usage = RunUsageContext("run-2", tmp_path / "usage.json")
    repaired, _, attempts = execute_repair(
        "malformed",
        runner=FakeRunner(antigravity_outputs=[("not structured", 0), (valid, 0)]),
        config=config,
        run_id="run-2",
        usage_context=usage,
        validate=lambda text: (
            parse_structured_pr_review(text, reviewer="Google Antigravity")
            or (_ for _ in ()).throw(AgentLoopError("invalid repaired output"))
        ),
        expected_kind="pr_review",
    )

    assert repaired == valid
    assert [attempt.outcome for attempt in attempts] == ["invalid_output", "succeeded"]
    assert [record.outcome for record in usage.records] == ["invalid_output", "succeeded"]
    assert usage.records[0].fallback_planned is True
    assert usage.records[1].fallback_planned is False

@pytest.mark.parametrize(
    ("output", "returncode", "expected"),
    [
        ("fatal auth error", 41, "nonzero_exit"),
        ("", 0, "empty_output"),
        (PUBLIC_RESPONSE_MARKER, 0, "empty_output"),
        ("", None, "timeout"),
    ],
)
def test_execute_repair_records_antigravity_failure_outcomes(
    tmp_path, monkeypatch, output, returncode, expected
):
    from coding_review_agent_loop.agents import antigravity as agy_mod

    monkeypatch.setattr(
        agy_mod, "_antigravity_settings_path", lambda: tmp_path / "settings.json"
    )
    repaired, _, attempts = execute_repair(
        "malformed",
        runner=FakeRunner(antigravity_outputs=[(output, returncode)]),
        config=make_config(tmp_path),
        run_id="run-3",
        usage_context=None,
        validate=lambda text: text,
        expected_kind="pr_review",
    )
    assert repaired is None
    assert attempts[0].outcome == expected

def test_execute_repair_records_spawn_error_with_log_path(tmp_path, monkeypatch):
    from coding_review_agent_loop.agents import antigravity as agy_mod

    monkeypatch.setattr(
        agy_mod, "_antigravity_settings_path", lambda: tmp_path / "settings.json"
    )

    class SpawnErrorRunner(FakeRunner):
        def run_with_log(self, *args, **kwargs):
            raise AgentLoopError("agy executable missing")

    repaired, _, attempts = execute_repair(
        "malformed",
        runner=SpawnErrorRunner(),
        config=make_config(tmp_path),
        run_id="run-spawn",
        usage_context=None,
        validate=lambda text: text,
        expected_kind="pr_review",
    )
    assert repaired is None
    assert attempts[0].outcome == "spawn_error"
    assert attempts[0].log_path is not None
    assert attempts[0].log_path.exists()

def test_execute_repair_uses_configured_legacy_gemini_override(tmp_path):
    valid = structured_pr_review(state="approved", reviewer="Google Gemini")
    proc = MagicMock(returncode=0, stdout=valid, stderr="")
    config = make_config(
        tmp_path,
        repair_backend="gemini",
        repair_models=("gemini-enterprise-flash",),
    )
    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=proc) as run:
        repaired, _, attempts = execute_repair(
            "malformed",
            runner=FakeRunner(),
            config=config,
            run_id="run-4",
            usage_context=None,
            validate=lambda text: parse_structured_pr_review(
                text, reviewer="Google Gemini"
            ),
            expected_kind="pr_review",
        )
    assert repaired == valid
    assert attempts[0].backend == "gemini"
    assert "gemini-enterprise-flash" in run.call_args.args[0]

def test_runner_pty_timeout_is_opt_in_and_retains_combined_log(tmp_path):
    log_path = tmp_path / "logs" / "timeout.log"
    result = Runner().run_with_log(
        [
            sys.executable,
            "-c",
            "import sys,time; print('stderr diagnostic', file=sys.stderr, flush=True); time.sleep(5)",
        ],
        cwd=tmp_path,
        log_path=log_path,
        label="timeout-test",
        progress_interval_seconds=30,
        check=False,
        use_pty=True,
        timeout_seconds=0.2,
    )

    assert result.returncode is None
    assert result.stderr == ""
    assert "stderr diagnostic" in result.stdout
    assert "stderr diagnostic" in log_path.read_text(encoding="utf-8")

def test_run_pr_loop_uses_repair_pass_on_format_failure(tmp_path):
    """Repair pass is invoked when schema validation fails; repaired output is used."""
    malformed_review = (
        "Looks good overall.\n\n"
        "AGENT_STATE: approved\n"
        "-- OpenAI Codex"
    )
    repaired_review = (
        '{"schema_version":1,"kind":"pr_review","state":"approved","summary":"Looks good overall.",'
        '"blocking_items":[],"same_pr_followups":[],"future_followups":[],'
        '"prior_item_dispositions":[]}'
        "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    )
    runner = FakeRunner(
        codex_outputs=[malformed_review],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", agent_max_retries=0)

    captured_repairs = []

    def fake_attempt_repair(raw: str, gemini_cmd: str, *, expected_kind: str | None = None, unresolved_item_ids=None, surfaced_requirement_ids=None, allowed_prior_item_ids=None, unknown_prior_item_ids=None, same_round_context=None) -> str | None:
        captured_repairs.append(raw)
        assert expected_kind == "pr_review"
        return repaired_review

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_attempt_repair):
        run_pr_loop(runner, pr_number=77, config=config)

    assert len(captured_repairs) == 1
    assert "AGENT_STATE: approved" in captured_repairs[0]

def test_run_validated_agent_envelope_normalization_recovers_duplicate_footer(tmp_path):
    malformed_review = (
        structured_pr_review(
            state="approved",
            reviewer="Google Gemini",
            prior_item_dispositions=[
                {"item_id": "item-1", "disposition": "resolved"},
                {"item_id": "item-2", "disposition": "future"},
                {"item_id": "item-3", "disposition": "resolved"},
            ],
        )
        + "\n\n<!-- AGENT_STATE: approved -->"
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the PR.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: parse_structured_pr_review(
                text,
                reviewer="Google Gemini",
            ).state,
            use_repair=True,
            repair_expected_kind="pr_review",
        )

    repair_mock.assert_not_called()
    parsed = parse_structured_pr_review(response.text, reviewer="Google Gemini")
    assert parsed is not None
    assert {
        disposition.item_id: disposition.disposition
        for disposition in parsed.dispositions
    } == {
        "item-1": "resolved",
        "item-2": "future",
        "item-3": "resolved",
    }

def test_run_validated_agent_envelope_normalization_semantic_defect_uses_repair(tmp_path):
    malformed_review = (
        structured_pr_review(
            state="approved",
            reviewer="Google Gemini",
            blocking_items=["This is semantically inconsistent."],
        )
        + "\n\n<!-- AGENT_STATE: approved -->"
    )
    repaired_review = structured_pr_review(state="approved", reviewer="Google Gemini")
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)
    normalized_review = attempt_envelope_normalization(malformed_review, expected_kind="pr_review")

    with patch(
        "coding_review_agent_loop.orchestrator.attempt_repair",
        return_value=repaired_review,
    ) as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the PR.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: parse_structured_pr_review(
                text,
                reviewer="Google Gemini",
            ).state,
            use_repair=True,
            repair_expected_kind="pr_review",
        )

    assert response.text == repaired_review
    repair_mock.assert_called_once_with(
        normalized_review,
        config.gemini_cmd,
        expected_kind="pr_review",
    )

def test_run_validated_agent_attempt_repair_uses_envelope_normalized(tmp_path, monkeypatch):
    raw_text = structured_pr_review(state="blocking", reviewer="Google Gemini") + "\ngarbage"
    normalized_text = structured_pr_review(state="blocking", reviewer="Google Gemini")
    repaired_text = structured_pr_review(state="approved", reviewer="Google Gemini")

    monkeypatch.setattr(
        "coding_review_agent_loop.orchestrator.attempt_envelope_normalization",
        lambda text, expected_kind: normalized_text,
    )
    repair_inputs = []
    monkeypatch.setattr(
        "coding_review_agent_loop.orchestrator.attempt_repair",
        lambda text, gemini_cmd, **kwargs: repair_inputs.append(text) or repaired_text,
    )

    runner = FakeRunner(gemini_outputs=[(raw_text, 0)])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    def validate(text):
        if text == repaired_text:
            return parse_structured_pr_review(text, reviewer="Google Gemini").state
        raise AgentLoopError("invalid")

    response = _run_validated_agent(
        runner,
        agent="gemini",
        config=config,
        prompt="Review the PR.",
        marker_description="<!-- AGENT_STATE: approved|blocking -->",
        validate=validate,
        use_repair=True,
        repair_expected_kind="pr_review",
    )

    assert response.text == repaired_text
    assert repair_inputs == [normalized_text]

def test_run_validated_agent_attempt_repair_falls_back_to_text_when_no_normalization(tmp_path, monkeypatch):
    raw_text = structured_pr_review(state="blocking", reviewer="Google Gemini")
    repaired_text = structured_pr_review(state="approved", reviewer="Google Gemini")

    monkeypatch.setattr(
        "coding_review_agent_loop.orchestrator.attempt_envelope_normalization",
        lambda text, expected_kind: None,
    )
    repair_inputs = []
    monkeypatch.setattr(
        "coding_review_agent_loop.orchestrator.attempt_repair",
        lambda text, gemini_cmd, **kwargs: repair_inputs.append(text) or repaired_text,
    )

    runner = FakeRunner(gemini_outputs=[(raw_text, 0)])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    def validate(text):
        if text == repaired_text:
            return parse_structured_pr_review(text, reviewer="Google Gemini").state
        raise AgentLoopError("invalid")

    response = _run_validated_agent(
        runner,
        agent="gemini",
        config=config,
        prompt="Review the PR.",
        marker_description="<!-- AGENT_STATE: approved|blocking -->",
        validate=validate,
        use_repair=True,
        repair_expected_kind="pr_review",
    )

    assert response.text == repaired_text
    assert repair_inputs == [raw_text]

def _plan_revision_validate_with_human_requirements(human_requirements):
    return lambda text: orchestrator._validate_response_with_human_requirements(
        text,
        marker_validator=lambda revised_text: _validate_plan_revision_response(
            revised_text,
            unresolved_items=(),
        ),
        human_requirements=human_requirements,
        requirement_scope="planning requirements",
        full_omission_fallback="Fetch the issue discussion directly before revising the plan.",
    )

def test_recover_plan_revision_ack_text_override_uses_stripped_as_base(tmp_path):
    from coding_review_agent_loop.agents.base import AgentResult

    human_requirements = (
        HumanReviewRequirement(
            source_type="Issue comment",
            author="wwind123",
            created_at="2026-06-05T00:00:00Z",
            url="https://github.com/OWNER/REPO/issues/1#issuecomment-1",
            body="Requirement 1: cover the stripped-base case.",
        ),
    )
    context = render_coder_human_requirements_prompt_context(
        human_requirements,
        requirement_scope="planning requirements",
        full_omission_fallback="Fetch the issue directly before revising the plan.",
    )
    ack = (
        "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n"
        "### Human requirements\n"
        "- Requirement 1: The revised plan covers the stripped-base case.\n"
    )
    dirty_text = structured_plan_revision(
        prior_plan_item_dispositions=[{"item_id": "unknown-prior-item-1", "disposition": "resolved"}],
    )
    stripped_text = structured_plan_revision()
    message_text = structured_plan_revision(human_requirements=ack)

    result = AgentResult(
        text=dirty_text,
        message_text=message_text,
        response_file_text=dirty_text,
    )
    config = make_config(tmp_path, coder="claude", agent_max_retries=0)
    validate = _plan_revision_validate_with_human_requirements(human_requirements)

    recovered = _recover_plan_revision_human_requirements_acknowledgement(
        result,
        text=stripped_text,
        validate=validate,
        context=_HumanRequirementsRecoveryContext(
            surfaced_requirement_ids=context.surfaced_requirement_ids,
            requires_direct_discussion_ack=context.requires_direct_discussion_ack,
        ),
        config=config,
        agent_name="TestAgent",
    )

    assert recovered is not None
    recovered_text, _ = recovered
    # JSON prefix comes from stripped_text (no unknown disposition), not dirty_text
    assert "unknown-prior-item-1" not in recovered_text
    assert "<!-- HUMAN_REQUIREMENTS_ADDRESSED -->" in recovered_text
    assert "### Human requirements" in recovered_text
    validate(recovered_text)

def test_run_validated_agent_strip_path_ack_recovery(tmp_path, monkeypatch):
    human_requirements = (
        HumanReviewRequirement(
            source_type="Issue comment",
            author="wwind123",
            created_at="2026-06-05T00:00:00Z",
            url="https://github.com/OWNER/REPO/issues/1#issuecomment-1",
            body="Requirement 1: cover the strip-path ack recovery.",
        ),
    )
    context = render_coder_human_requirements_prompt_context(
        human_requirements,
        requirement_scope="planning requirements",
        full_omission_fallback="Fetch the issue directly before revising the plan.",
    )
    ack = (
        "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n"
        "### Human requirements\n"
        "- Requirement 1: The revised plan covers the strip-path case.\n"
    )
    response_file = structured_plan_revision(
        prior_plan_item_dispositions=[{"item_id": "unknown-prior-item-1", "disposition": "resolved"}],
    )
    message_text_with_ack = structured_plan_revision(human_requirements=ack)
    recovered_text = structured_plan_revision(human_requirements=ack)

    recovery_calls = []

    def mock_recovery(result, *, text=None, **kwargs):
        recovery_calls.append(text)
        return (recovered_text, "blocking")

    monkeypatch.setattr(
        "coding_review_agent_loop.orchestrator._recover_plan_revision_human_requirements_acknowledgement",
        mock_recovery,
    )

    runner = FakeRunner(
        claude_outputs=[(message_text_with_ack, 0)],
        public_response_outputs=[{"text": response_file}],
    )
    config = make_config(tmp_path, coder="claude", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="claude",
            config=config,
            prompt="Revise the plan.",
            marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
            validate=_plan_revision_validate_with_human_requirements(human_requirements),
            use_repair=True,
            repair_expected_kind="plan_revision",
            repair_surfaced_requirement_ids=context.surfaced_requirement_ids,
            repair_requires_direct_discussion_ack=context.requires_direct_discussion_ack,
        )

    repair_mock.assert_not_called()
    assert response.text == recovered_text
    # Recovery was called once for the stripped variant (no unknown disposition)
    assert len(recovery_calls) == 1
    assert recovery_calls[0] is not None
    assert "unknown-prior-item-1" not in recovery_calls[0]

def test_run_validated_agent_combined_strip_path_ack_recovery(tmp_path, monkeypatch):
    raw_text = "non-structured plan revision text"
    normalized_text = structured_plan_revision(
        prior_plan_item_dispositions=[{"item_id": "unknown-item-1", "disposition": "resolved"}],
    )
    stripped_from_normalized = structured_plan_revision()
    recovered_text = structured_plan_revision(
        human_requirements=(
            "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n"
            "### Human requirements\n- Req 1: covered.\n"
        ),
    )

    human_requirements = (
        HumanReviewRequirement(
            source_type="Issue comment",
            author="wwind123",
            created_at="2026-06-05T00:00:00Z",
            url="https://github.com/OWNER/REPO/issues/1#issuecomment-1",
            body="Req 1: cover the combined-strip-path case.",
        ),
    )
    context = render_coder_human_requirements_prompt_context(
        human_requirements,
        requirement_scope="planning requirements",
        full_omission_fallback="Fetch the issue directly.",
    )

    def fake_validate(text):
        if text == raw_text:
            raise AgentLoopError("not structured")
        if text == normalized_text:
            raise UnknownPriorItemDispositionError(
                unknown_ids=("unknown-item-1",),
                allowed_ids=(),
                same_round_description="not a valid prior item",
            )
        if text == stripped_from_normalized:
            raise AgentLoopError("missing ack")
        return "blocking"

    monkeypatch.setattr(
        "coding_review_agent_loop.orchestrator.attempt_envelope_normalization",
        lambda text, expected_kind: normalized_text if text == raw_text else None,
    )
    monkeypatch.setattr(
        "coding_review_agent_loop.orchestrator.strip_unknown_prior_item_dispositions",
        lambda text, allowed_ids, expected_kind: (
            stripped_from_normalized if text == normalized_text else None
        ),
    )
    monkeypatch.setattr(
        "coding_review_agent_loop.orchestrator._plan_revision_missing_human_acknowledgement",
        lambda text, context: text == stripped_from_normalized,
    )

    recovery_calls = []

    def mock_recovery(result, *, text=None, **kwargs):
        recovery_calls.append(text)
        return (recovered_text, "blocking")

    monkeypatch.setattr(
        "coding_review_agent_loop.orchestrator._recover_plan_revision_human_requirements_acknowledgement",
        mock_recovery,
    )

    runner = FakeRunner(
        claude_outputs=[(raw_text, 0)],
        public_response_outputs=[{"text": raw_text}],
    )
    config = make_config(tmp_path, coder="claude", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="claude",
            config=config,
            prompt="Revise the plan.",
            marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
            validate=fake_validate,
            use_repair=True,
            repair_expected_kind="plan_revision",
            repair_surfaced_requirement_ids=context.surfaced_requirement_ids,
            repair_requires_direct_discussion_ack=context.requires_direct_discussion_ack,
        )

    repair_mock.assert_not_called()
    assert response.text == recovered_text
    assert len(recovery_calls) == 1
    assert recovery_calls[0] is stripped_from_normalized

def test_run_pr_loop_repairs_format_failure_with_5xx_source_line_reference(tmp_path):
    """A 500-series source line reference must not make deterministic format errors transient."""
    malformed_review = (
        "Looks good overall.\n\n"
        "Note: orchestrator.py:577-581 currently falls back to parse_plan_state(text).\n"
        "AGENT_STATE: approved\n"
        "-- OpenAI Codex"
    )
    repaired_review = (
        '{"schema_version":1,"kind":"pr_review","state":"approved","summary":"Looks good overall.",'
        '"blocking_items":[],"same_pr_followups":[],"future_followups":[],'
        '"prior_item_dispositions":[]}'
        "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    )
    runner = FakeRunner(
        codex_outputs=[malformed_review],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", agent_max_retries=0)

    captured_repairs = []

    def fake_attempt_repair(raw: str, gemini_cmd: str, *, expected_kind: str | None = None, unresolved_item_ids=None, surfaced_requirement_ids=None, allowed_prior_item_ids=None, unknown_prior_item_ids=None, same_round_context=None) -> str | None:
        captured_repairs.append(raw)
        assert expected_kind == "pr_review"
        return repaired_review

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_attempt_repair):
        run_pr_loop(runner, pr_number=77, config=config)

    assert len(captured_repairs) == 1
    assert "orchestrator.py:577-581" in captured_repairs[0]

def test_run_pr_loop_falls_back_to_error_when_repair_also_fails(tmp_path):
    """When repair also produces invalid output, the original error is raised."""
    malformed_review = (
        "Something went wrong with the format.\n"
        "AGENT_STATE: approved\n"
        "-- OpenAI Codex"
    )
    runner = FakeRunner(
        codex_outputs=[malformed_review],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", agent_max_retries=0)

    def fake_attempt_repair_fails(raw: str, gemini_cmd: str, *, expected_kind: str | None = None, unresolved_item_ids=None, surfaced_requirement_ids=None, allowed_prior_item_ids=None, unknown_prior_item_ids=None, same_round_context=None) -> str | None:
        assert expected_kind == "pr_review"
        return "still broken output without valid schema"

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_attempt_repair_fails):
        with pytest.raises(AgentLoopError, match="Codex"):
            run_pr_loop(runner, pr_number=77, config=config)

def test_run_pr_loop_skips_repair_when_repair_returns_none(tmp_path):
    """When attempt_repair returns None (e.g. no API key), normal error is raised."""
    malformed_review = (
        "Something went wrong.\n"
        "AGENT_STATE: approved\n"
        "-- OpenAI Codex"
    )
    runner = FakeRunner(
        codex_outputs=[malformed_review],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=None):
        with pytest.raises(AgentLoopError, match="Codex"):
            run_pr_loop(runner, pr_number=77, config=config)

def test_run_pr_loop_uses_repair_pass_on_coder_followup_format_failure(tmp_path):
    """Repair pass is invoked when coder followup schema validation fails; repaired output is used."""
    malformed_coder_followup = (
        '{"schema_version":1,"kind":"pr_review","state":"blocking","summary":"Fixed the bug.",'
        '"blocking_items":[],"same_pr_followups":[],"future_followups":[],'
        '"prior_item_dispositions":[]}'
        "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    repaired_followup = (
        '{"schema_version":1,"kind":"coder_followup","state":"blocking","summary":"Fixed the bug.",'
        '"addressed_items":["item-1"],"remaining_items":[],'
        '"human_requirements":{"addressed_ids":[],"checked_discussion_directly":false}}'
        "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = FakeRunner(
        claude_outputs=[malformed_coder_followup],
        codex_outputs=[
            "Need a fix."
            + blocking_issues("Fix the bug.")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=2, agent_max_retries=0)

    captured_repairs = []
    captured_unresolved_item_ids = []

    def fake_attempt_repair(raw: str, gemini_cmd: str, *, expected_kind: str | None = None, unresolved_item_ids=None, surfaced_requirement_ids=None, allowed_prior_item_ids=None, unknown_prior_item_ids=None, same_round_context=None) -> str | None:
        captured_repairs.append(raw)
        captured_unresolved_item_ids.append(tuple(unresolved_item_ids or ()))
        assert expected_kind == "coder_followup"
        return repaired_followup

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_attempt_repair):
        result = run_pr_loop(runner, pr_number=77, config=config)

    assert result == 0
    assert len(captured_repairs) == 1
    assert "pr_review" in captured_repairs[0]
    assert captured_unresolved_item_ids == [("item-1",)]

def test_run_pr_loop_falls_back_to_error_when_coder_followup_repair_also_fails(tmp_path):
    """When repair also produces invalid output for coder followup, the original error is raised."""
    malformed_coder_followup = (
        '{"schema_version":1,"kind":"pr_review","state":"blocking","summary":"Fixed the bug.",'
        '"blocking_items":[],"same_pr_followups":[],"future_followups":[],'
        '"prior_item_dispositions":[]}'
        "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = FakeRunner(
        claude_outputs=[malformed_coder_followup],
        codex_outputs=[
            "Need a fix."
            + blocking_issues("Fix the bug.")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=2, agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value="still broken output"):
        with pytest.raises(AgentLoopError, match="Claude"):
            run_pr_loop(runner, pr_number=77, config=config)

def test_run_pr_loop_skips_repair_when_coder_followup_repair_returns_none(tmp_path):
    """When attempt_repair returns None for coder followup, normal error is raised."""
    malformed_coder_followup = (
        '{"schema_version":1,"kind":"pr_review","state":"blocking","summary":"Fixed the bug.",'
        '"blocking_items":[],"same_pr_followups":[],"future_followups":[],'
        '"prior_item_dispositions":[]}'
        "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = FakeRunner(
        claude_outputs=[malformed_coder_followup],
        codex_outputs=[
            "Need a fix."
            + blocking_issues("Fix the bug.")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=2, agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=None):
        with pytest.raises(AgentLoopError, match="Claude"):
            run_pr_loop(runner, pr_number=77, config=config)

def test_repair_prompt_contains_coder_followup_format():
    """Repair prompt must include the coder_followup format so the model knows about it."""
    assert "coder_followup" in _REPAIR_PROMPT
    assert "addressed_items" in _REPAIR_PROMPT
    assert "remaining_items" in _REPAIR_PROMPT

def test_repair_prompt_distinguishes_item_ids_from_requirement_labels():
    """Repair prompt must warn that addressed_items uses item IDs, not requirement labels."""
    assert "Requirement 1" in _REPAIR_PROMPT
    assert "addressed_ids" in _REPAIR_PROMPT
    # The prompt must explicitly state item IDs cannot contain spaces
    assert "spaces" in _REPAIR_PROMPT or "DO NOT CONFUSE" in _REPAIR_PROMPT or "NEVER put" in _REPAIR_PROMPT

def test_repair_prompt_includes_plan_review_dedupe_guidance():
    assert "Same-plan follow-ups and Future follow-ups are mutually exclusive" in _REPAIR_PROMPT
    assert "keep blocking_plan_issues and drop the duplicate same_plan_followups entry" in _REPAIR_PROMPT
    assert (
        "keep same_plan_followups/current-plan work and drop the duplicate future_followups entry"
        in _REPAIR_PROMPT
    )
    assert "keep blocking_plan_issues and drop the duplicate future_followups entry" in _REPAIR_PROMPT

def test_repair_prompt_includes_pr_review_dedupe_guidance():
    assert "## DEDUPE RULES (Format A):" in _REPAIR_PROMPT
    assert "Same-PR follow-ups and Future follow-ups are mutually exclusive" in _REPAIR_PROMPT
    assert "keep blocking_items and drop the duplicate same_pr_followups entry" in _REPAIR_PROMPT
    assert (
        "keep same_pr_followups/current-PR work and drop the duplicate future_followups entry"
        in _REPAIR_PROMPT
    )
    assert "keep blocking_items and drop the duplicate future_followups entry" in _REPAIR_PROMPT

def test_repair_prompt_includes_skip_trust_in_cli_invocation():
    """The CLI invocation must include --skip-trust so repair works outside trusted dirs."""
    repaired = (
        '{"schema_version":1,"kind":"pr_review","state":"approved","summary":"OK",'
        '"blocking_items":[],"same_pr_followups":[],"future_followups":[],'
        '"prior_item_dispositions":[]}\n<!-- AGENT_STATE: approved -->\n-- Gemini'
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = repaired

    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=mock_result) as mock_run:
        attempt_repair("malformed review", "gemini")

    cmd = mock_run.call_args.args[0]
    assert "--skip-trust" in cmd

def test_repair_prompt_coder_followup_fenced_json_example():
    """Repair prompt must include a worked example showing fenced JSON being stripped."""
    assert "```json" in _REPAIR_PROMPT
    assert "HUMAN_REQUIREMENTS_ADDRESSED" in _REPAIR_PROMPT
    # The prompt explains the marker is not needed in structured path
    assert "NOT needed" in _REPAIR_PROMPT or "not needed" in _REPAIR_PROMPT.lower()

def test_repair_prompt_plan_revision_preserves_human_requirements_acknowledgement():
    assert "WORKED EXAMPLE 4" in _REPAIR_PROMPT
    assert "do not output coder_followup" in _REPAIR_PROMPT
    assert "preserve both after the JSON and before <!-- AGENT_PLAN_STATE: blocking -->" in _REPAIR_PROMPT

def test_repair_prompt_does_not_suggest_ack_pseudo_item_in_addressed_items():
    """The ack pseudo-item must never be suggested as a value for addressed_items.

    The orchestrator's _validate_structured_coder_followup_items explicitly excludes
    HUMAN_REQUIREMENTS_ACK_ITEM_ID from expected_ids, so any response that puts
    'item-human-requirements-acknowledgement' in addressed_items will be rejected
    as an unknown item ID.
    """
    from coding_review_agent_loop.orchestrator import HUMAN_REQUIREMENTS_ACK_ITEM_ID

    # The ack pseudo-item must not appear in the repair prompt at all, because
    # any mention of it in an addressed_items context will teach Gemini to produce
    # responses that the validator rejects.
    assert HUMAN_REQUIREMENTS_ACK_ITEM_ID not in _REPAIR_PROMPT

from coding_review_agent_loop.repair import (
    _reviewer_human_requirements_instruction,
)

from coding_review_agent_loop.orchestrator import _surfaced_reviewer_requirement_ids

def test_repair_prompt_blocking_state_rules_require_explicit_prior_dispositions():
    """STATE RULES for BLOCKING must prohibit omitting allowed prior items."""
    assert "ALL prior items in the allowed list must appear in prior_item_dispositions" in _REPAIR_PROMPT
    assert "No item may be omitted" in _REPAIR_PROMPT or "no item may be omitted" in _REPAIR_PROMPT.lower()
    assert '"future" is forbidden in blocking reviews' in _REPAIR_PROMPT or "future\" is forbidden in blocking" in _REPAIR_PROMPT

def test_repair_example_2_no_longer_says_omit():
    """Worked Example 2 must not instruct omission of formerly-future prior items."""
    assert "OMIT item-1 from prior_item_dispositions entirely" not in _REPAIR_PROMPT
    assert "WORKED EXAMPLE 2" in _REPAIR_PROMPT
    assert "must appear" in _REPAIR_PROMPT

def test_repair_prompt_includes_approved_plus_active_disposition_rules():
    """STATE RULES must cover approved + active same-pr/same-plan/blocking dispositions."""
    assert "APPROVED + active same-pr/same-plan/blocking prior dispositions" in _REPAIR_PROMPT
    assert 'change disposition to "resolved"' in _REPAIR_PROMPT

def test_repair_prompt_includes_approved_future_followups_current_plan_rule():
    """STATE RULES must cover approved reviews with current-plan concerns in future_followups."""
    assert "future_followups that are actually current-plan" in _REPAIR_PROMPT or \
           "future_followups" in _REPAIR_PROMPT and "required for the current plan" in _REPAIR_PROMPT

def test_repair_prompt_includes_worked_example_6_to_12():
    """Examples 6-12 must be present."""
    for n in range(6, 13):
        assert f"WORKED EXAMPLE {n}" in _REPAIR_PROMPT

def test_repair_prompt_example_12_same_round_confusion_case():
    """Example 12 must describe the same-round disposition confusion with future_followups."""
    assert "WORKED EXAMPLE 12" in _REPAIR_PROMPT
    assert "same-round" in _REPAIR_PROMPT.lower() or "same-round finding" in _REPAIR_PROMPT.lower()
    assert "future_followups" in _REPAIR_PROMPT

def test_reviewer_human_requirements_instruction_pr_review():
    result = _reviewer_human_requirements_instruction("pr_review", ["Requirement 1", "Requirement 2"])
    assert "HUMAN_REQUIREMENTS_RESOLVED" in result
    assert "Requirement 1" in result
    assert "Requirement 2" in result
    assert "AGENT_STATE" in result
    assert "blocking_items" in result

def test_reviewer_human_requirements_instruction_plan_review():
    result = _reviewer_human_requirements_instruction("plan_review", ["Requirement 1"])
    assert "HUMAN_REQUIREMENTS_RESOLVED" in result
    assert "Requirement 1" in result
    assert "AGENT_PLAN_STATE" in result
    assert "blocking_plan_issues" in result

def test_reviewer_human_requirements_instruction_empty_ids():
    result = _reviewer_human_requirements_instruction("pr_review", [])
    assert "(none)" in result
    assert "HUMAN_REQUIREMENTS_RESOLVED" in result

def test_reviewer_human_requirements_instruction_returns_empty_for_none():
    assert _reviewer_human_requirements_instruction("pr_review", None) == ""
    assert _reviewer_human_requirements_instruction("plan_review", None) == ""

def test_reviewer_human_requirements_instruction_rejects_coder_kind():
    with pytest.raises(ValueError, match="reviewer_requirement_ids"):
        _reviewer_human_requirements_instruction("coder_followup", ["Requirement 1"])

def test_attempt_repair_includes_reviewer_requirement_instruction():
    """attempt_repair passes reviewer_requirement_ids into the prompt for pr_review."""
    repaired = structured_pr_review(
        state="approved",
        reviewer="OpenAI Codex",
        human_requirements_resolved=True,
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = repaired

    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=mock_result) as mock_run:
        attempt_repair(
            "malformed review",
            "gemini",
            expected_kind="pr_review",
            reviewer_requirement_ids=["Requirement 1", "Requirement 2"],
        )

    cmd = mock_run.call_args.args[0]
    prompt = cmd[cmd.index("--prompt") + 1]
    assert "Requirement 1" in prompt
    assert "Requirement 2" in prompt
    assert "HUMAN_REQUIREMENTS_RESOLVED" in prompt
    assert "AGENT_STATE" in prompt

def test_attempt_repair_reviewer_requirement_ids_not_included_for_plan_revision():
    """reviewer_requirement_ids raises for non pr_review/plan_review kinds."""
    with pytest.raises(ValueError, match="reviewer_requirement_ids"):
        attempt_repair(
            "malformed plan revision",
            "gemini",
            expected_kind="plan_revision",
            reviewer_requirement_ids=["Requirement 1"],
        )

def test_surfaced_reviewer_requirement_ids_pr_uses_merged_requirements():
    """PR loop helper returns IDs using PR requirements scope."""
    hr = (
        HumanReviewRequirement(
            source_type="Issue comment",
            author="maintainer",
            created_at="2026-01-01T00:00:00Z",
            url="https://example.com/1",
            body="Use absolute URLs.",
        ),
    )
    ids = _surfaced_reviewer_requirement_ids(hr, requirement_scope="PR requirements")
    assert ids == ("Requirement 1",)

def test_surfaced_reviewer_requirement_ids_plan_uses_issue_requirements():
    """Plan loop helper returns IDs using planning requirements scope."""
    hr = (
        HumanReviewRequirement(
            source_type="Issue comment",
            author="maintainer",
            created_at="2026-01-01T00:00:00Z",
            url="https://example.com/1",
            body="Add regression tests.",
        ),
        HumanReviewRequirement(
            source_type="Issue comment",
            author="maintainer",
            created_at="2026-01-02T00:00:00Z",
            url="https://example.com/2",
            body="Keep backward compatibility.",
        ),
    )
    ids = _surfaced_reviewer_requirement_ids(hr, requirement_scope="planning requirements")
    assert "Requirement 1" in ids
    assert "Requirement 2" in ids

def test_surfaced_reviewer_requirement_ids_empty_for_no_requirements():
    ids = _surfaced_reviewer_requirement_ids([], requirement_scope="PR requirements")
    assert ids == ()

def _pr_payload_with_human_requirement():
    return {
        "number": 77,
        "state": "OPEN",
        "url": "https://github.com/OWNER/REPO/pull/77",
        "title": "Improve review prompt context",
        "headRefName": "feature/review-context",
        "baseRefName": "main",
        "headRefOid": "abc123",
        "comments": [
            {
                "author": {"login": "maintainer"},
                "createdAt": "2026-05-18T10:00:00Z",
                "url": "https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                "body": "Please use the absolute URL.\n\n-- Human Reviewer",
            }
        ],
        "reviews": [],
    }

def test_pr_loop_repair_missing_hr_marker_recovers_approved(tmp_path):
    """When repair returns approved + HUMAN_REQUIREMENTS_RESOLVED, no synthetic item is injected."""
    approved_without_marker = structured_pr_review(
        state="approved",
        reviewer="OpenAI Codex",
        human_requirements_resolved=False,
    )
    repaired_with_marker = structured_pr_review(
        state="approved",
        reviewer="OpenAI Codex",
        human_requirements_resolved=True,
    )
    runner = FakeRunner(
        codex_outputs=[approved_without_marker],
        pr_payload=_pr_payload_with_human_requirement(),
    )
    config = make_config(tmp_path, max_rounds=1)

    def fake_repair(raw, gemini_cmd, *, expected_kind=None, reviewer_requirement_ids=None, **kwargs):
        assert expected_kind == "pr_review"
        assert reviewer_requirement_ids == ("Requirement 1",)
        return repaired_with_marker

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_repair):
        result = run_pr_loop(runner, pr_number=77, config=config)

    assert result == 0
    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert claude_calls == [], "Coder should not be woken when repair recovers the marker"

def test_pr_loop_repair_missing_hr_marker_returns_blocking_not_synthetic(tmp_path):
    """When repair returns valid blocking, treat as reviewer blocking — no synthetic item."""
    approved_without_marker = structured_pr_review(
        state="approved",
        reviewer="OpenAI Codex",
        human_requirements_resolved=False,
    )
    repaired_blocking = structured_pr_review(
        state="blocking",
        summary="Requirement 1 not satisfied: absolute URL missing.",
        blocking_items=["Requirement 1 not satisfied: absolute URL missing."],
        reviewer="OpenAI Codex",
    )
    # Round 2: coder addresses item-1 (the repaired blocking item) + acks human requirements
    coder_response = structured_coder_followup(
        state="approved",
        addressed_items=["item-1"],
        remaining_items=[],
        human_requirement_ids=["Requirement 1"],
        reviewer="Anthropic Claude",
    )
    runner = FakeRunner(
        claude_outputs=[coder_response],
        codex_outputs=[
            approved_without_marker,
            structured_pr_review(
                state="approved",
                reviewer="OpenAI Codex",
                human_requirements_resolved=True,
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        pr_payload=_pr_payload_with_human_requirement(),
    )
    config = make_config(tmp_path, max_rounds=2)

    pr_review_repair_calls = []
    def fake_repair(raw, gemini_cmd, *, expected_kind=None, reviewer_requirement_ids=None, **kwargs):
        if expected_kind == "pr_review" and reviewer_requirement_ids is not None:
            pr_review_repair_calls.append(raw)
            return repaired_blocking
        return None  # don't interfere with coder_followup repair

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_repair):
        result = run_pr_loop(runner, pr_number=77, config=config)

    assert result == 0
    assert pr_review_repair_calls, "Repair should have been attempted for the reviewer"
    # The repaired blocking item text should appear in a coder prompt
    claude_prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert any("absolute URL missing" in p for p in claude_prompts), \
        "Repaired blocking item should appear in coder prompt"
    # There should be no synthetic Orchestrator item text
    assert not any("Orchestrator" in p and "acknowledging the signed human requirements" in p
                   for p in claude_prompts), \
        "Synthetic orchestrator item must not appear when repair returned valid blocking"

def test_pr_loop_repair_missing_hr_marker_failure_uses_synthetic(tmp_path):
    """When repair fails (returns None), synthetic blocking item is injected."""
    approved_without_marker = structured_pr_review(
        state="approved",
        reviewer="OpenAI Codex",
        human_requirements_resolved=False,
    )
    runner = FakeRunner(
        codex_outputs=[approved_without_marker],
        pr_payload=_pr_payload_with_human_requirement(),
    )
    config = make_config(tmp_path, max_rounds=1)

    def fake_repair(raw, gemini_cmd, *, expected_kind=None, **kwargs):
        return None  # repair fails

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_repair):
        with pytest.raises(AgentLoopError, match="blocking issues after round 1"):
            run_pr_loop(runner, pr_number=77, config=config)

def _issue_with_human_requirement():
    return {
        "author": {"login": "maintainer"},
        "createdAt": "2026-05-17T08:00:00Z",
        "body": "Keep the public API unchanged.\n\n-- Human Reviewer",
    }

def test_plan_loop_repair_missing_hr_marker_recovers_approved(tmp_path):
    """Plan loop: repair returning approved+marker suppresses synthetic."""
    plan = (
        "Initial plan.\n"
        f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
        "### Human requirements\n"
        "- Requirement 1: keep the public API unchanged.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    approved_without_marker = structured_plan_review(
        state="approved",
        reviewer="OpenAI Codex",
        human_requirements_resolved=False,
    )
    repaired_with_marker = structured_plan_review(
        state="approved",
        reviewer="OpenAI Codex",
        human_requirements_resolved=True,
    )
    runner = FakeRunner(
        issue_payload=_issue_with_human_requirement(),
        claude_outputs=[plan],
        codex_outputs=[approved_without_marker],
    )
    config = make_config(tmp_path)

    def fake_repair(raw, gemini_cmd, *, expected_kind=None, reviewer_requirement_ids=None, **kwargs):
        assert expected_kind == "plan_review"
        assert reviewer_requirement_ids == ("Requirement 1",)
        return repaired_with_marker

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_repair):
        result = run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert result == 0
    # No second claude call needed (no synthetic blocking item)
    plan_revision_calls = [
        cmd for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] and runner.claude_outputs == []
    ]
    # Verify plan approved comment was posted
    assert any("Approved plan:" in comment for comment in runner.comments)

def test_plan_loop_repair_missing_hr_marker_returns_blocking_not_synthetic(tmp_path):
    """Plan loop: repair returning blocking is treated as reviewer's blocking, not synthetic."""
    plan = (
        "Initial plan.\n"
        f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
        "### Human requirements\n"
        "- Requirement 1: keep the public API unchanged.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    approved_without_marker = structured_plan_review(
        state="approved",
        reviewer="OpenAI Codex",
        human_requirements_resolved=False,
    )
    repaired_blocking = structured_plan_review(
        state="blocking",
        summary="Requirement 1 not satisfied: plan changes the public API.",
        blocking_plan_issues=["Requirement 1 not satisfied: plan changes the public API."],
        reviewer="OpenAI Codex",
    )
    revision = structured_plan_revision(
        summary="Revised plan preserving the public API.",
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        human_requirements=(
            "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n"
            "### Human requirements\n"
            "- Requirement 1: the plan preserves the public API.\n"
        ),
    )
    runner = FakeRunner(
        issue_payload=_issue_with_human_requirement(),
        claude_outputs=[plan, revision],
        codex_outputs=[
            approved_without_marker,
            structured_plan_review(
                summary="Plan looks sound.",
                human_requirements_resolved=True,
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
    )
    config = make_config(tmp_path, max_rounds=3)

    call_count = [0]
    def fake_repair(raw, gemini_cmd, *, expected_kind=None, reviewer_requirement_ids=None, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return repaired_blocking
        return None  # subsequent calls not needed

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_repair):
        result = run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert result == 0
    # Confirm the repaired blocking item text reached the coder
    claude_prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert any("plan changes the public API" in p for p in claude_prompts), \
        "Repaired blocking item text must appear in coder prompt"
    # No synthetic orchestrator text
    assert not any("Orchestrator" in p and "acknowledging the signed human requirements" in p
                   for p in claude_prompts), \
        "Synthetic orchestrator item must not appear when repair returned valid blocking"

def test_plan_loop_repair_missing_hr_marker_failure_uses_synthetic(tmp_path):
    """Plan loop: when repair fails, synthetic blocking item is injected."""
    plan = (
        "Initial plan.\n"
        f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
        "### Human requirements\n"
        "- Requirement 1: keep the public API unchanged.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    approved_without_marker = structured_plan_review(
        state="approved",
        reviewer="OpenAI Codex",
        human_requirements_resolved=False,
    )
    revision = structured_plan_revision(
        summary="Revised plan.",
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        human_requirements=(
            "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n"
            "### Human requirements\n"
            "- Requirement 1: the plan preserves the public API.\n"
        ),
    )
    runner = FakeRunner(
        issue_payload=_issue_with_human_requirement(),
        claude_outputs=[plan, revision],
        codex_outputs=[
            approved_without_marker,
            structured_plan_review(
                summary="Plan looks sound.",
                human_requirements_resolved=True,
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
    )
    config = make_config(tmp_path, max_rounds=3)

    def fake_repair(raw, gemini_cmd, *, expected_kind=None, **kwargs):
        return None  # repair fails

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_repair):
        result = run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert result == 0
    # The synthetic item must have been injected (coder was woken with it)
    claude_prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert any("acknowledging the signed human requirements" in p for p in claude_prompts), \
        "Synthetic item must appear in coder prompt when repair fails"

def test_parse_pr_review_rejects_approved_with_same_pr_active_disposition():
    """Approved PR review with active same-pr disposition must fail validation."""
    malformed = structured_pr_review(
        state="approved",
        prior_item_dispositions=[{"item_id": "item-1", "disposition": "same-pr", "note": "Still needed"}],
    )
    with pytest.raises(AgentLoopError, match="Approved reviews must be fully complete"):
        parse_pr_review(malformed, reviewer="OpenAI Codex")

def test_parse_plan_review_rejects_blocking_with_future_disposition_on_prior_item():
    """Blocking plan review with future prior disposition must fail validation."""
    malformed = structured_plan_review(
        state="blocking",
        blocking_plan_issues=["Something is wrong."],
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "future"}],
    )
    with pytest.raises(AgentLoopError, match="[Ff]uture"):
        parse_plan_review(malformed, reviewer="OpenAI Codex")

def test_repair_blocking_formerly_future_prior_item_explicit_disposition():
    """Repair of blocking review with formerly-future prior item produces explicit non-future disposition."""
    malformed = (
        json.dumps({
            "schema_version": 1,
            "kind": "pr_review",
            "state": "blocking",
            "summary": "Fix the memory leak.",
            "blocking_items": ["Fix the memory leak"],
            "same_pr_followups": [],
            "future_followups": [],
            "prior_item_dispositions": [
                {"item_id": "item-1", "disposition": "future"},
            ],
        })
        + "\n<!-- AGENT_STATE: blocking -->\n-- Reviewer"
    )
    repaired = structured_pr_review(
        state="blocking",
        blocking_items=["Fix the memory leak"],
        prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        reviewer="Reviewer",
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = repaired

    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=mock_result) as mock_run:
        result = attempt_repair(
            malformed,
            "gemini",
            expected_kind="pr_review",
            allowed_prior_item_ids=["item-1"],
        )

    assert result == repaired
    cmd = mock_run.call_args.args[0]
    prompt = cmd[cmd.index("--prompt") + 1]
    # Verify the prompt contains the guidance about explicitly dispositioning prior items
    assert "must appear in prior_item_dispositions" in prompt or "prior_item_dispositions" in prompt
    assert "item-1" in prompt

def test_repair_same_round_disposition_confusion_promotes_to_blocking():
    """Repair prompt instructs removing same-round dispositions and promoting current concerns."""
    malformed = structured_plan_review(
        state="approved",
        future_followups=["Reconcile repair examples and validators."],
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
    )
    repaired = structured_plan_review(
        state="blocking",
        blocking_plan_issues=["Reconcile repair examples and validators."],
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = repaired

    with patch("coding_review_agent_loop.repair.subprocess.run", return_value=mock_result) as mock_run:
        result = attempt_repair(
            malformed,
            "gemini",
            expected_kind="plan_review",
            allowed_prior_item_ids=[],
            unknown_prior_item_ids=["item-1"],
            same_round_context="item-1 matches a same-round finding, not a carried prior item.",
        )

    assert result == repaired
    cmd = mock_run.call_args.args[0]
    prompt = cmd[cmd.index("--prompt") + 1]
    # The same-round context should appear in the prompt
    assert "same-round finding" in prompt
    # The guidance about future_followups promoting to blocking should be in STATE RULES
    assert "future_followups" in prompt

def test_repair_prompt_format_rule_allows_human_requirements_resolved_for_pr_plan_review():
    """FORMAT rule must permit HUMAN_REQUIREMENTS_RESOLVED for pr_review/plan_review, not just plan_revision."""
    assert "pr_review" in _REPAIR_PROMPT or "HUMAN_REQUIREMENTS_RESOLVED" in _REPAIR_PROMPT
    # Ensure the FORMAT rule no longer says "plan_revision only"
    assert "for plan_revision only" not in _REPAIR_PROMPT

def test_pr_loop_repair_blocking_records_same_pr_followups(tmp_path):
    """When repair returns blocking with same_pr_followups, those are recorded as same-pr items."""
    approved_without_marker = structured_pr_review(
        state="approved",
        reviewer="OpenAI Codex",
        human_requirements_resolved=False,
    )
    repaired_blocking = structured_pr_review(
        state="blocking",
        same_pr_followups=["Fix the error message formatting."],
        reviewer="OpenAI Codex",
    )
    # Coder addresses item-1 (same-pr followup)
    coder_response = structured_coder_followup(
        state="approved",
        addressed_items=["item-1"],
        remaining_items=[],
        human_requirement_ids=["Requirement 1"],
        reviewer="Anthropic Claude",
    )
    runner = FakeRunner(
        claude_outputs=[coder_response],
        codex_outputs=[
            approved_without_marker,
            structured_pr_review(
                state="approved",
                reviewer="OpenAI Codex",
                human_requirements_resolved=True,
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        pr_payload=_pr_payload_with_human_requirement(),
    )
    config = make_config(tmp_path, max_rounds=2)

    def fake_repair(raw, gemini_cmd, *, expected_kind=None, reviewer_requirement_ids=None, **kwargs):
        if expected_kind == "pr_review" and reviewer_requirement_ids is not None:
            return repaired_blocking
        return None

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_repair):
        result = run_pr_loop(runner, pr_number=77, config=config)

    assert result == 0
    # The same-pr followup text must appear in the coder's prompt
    claude_prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert any("error message formatting" in p for p in claude_prompts), \
        "Same-PR followup from repaired review must appear in coder prompt"

def test_plan_loop_repair_blocking_records_same_plan_followups(tmp_path):
    """When repair returns blocking with same_plan_followups, those are recorded as same-plan items."""
    plan = (
        "Initial plan.\n"
        f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
        "### Human requirements\n"
        "- Requirement 1: keep the public API unchanged.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    approved_without_marker = structured_plan_review(
        state="approved",
        reviewer="OpenAI Codex",
        human_requirements_resolved=False,
    )
    repaired_blocking = structured_plan_review(
        state="blocking",
        same_plan_followups=["Add a regression test for the parser edge case."],
        reviewer="OpenAI Codex",
    )
    revision = structured_plan_revision(
        summary="Revised plan with regression test.",
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        human_requirements=(
            "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n"
            "### Human requirements\n"
            "- Requirement 1: the plan preserves the public API.\n"
        ),
    )
    runner = FakeRunner(
        issue_payload=_issue_with_human_requirement(),
        claude_outputs=[plan, revision],
        codex_outputs=[
            approved_without_marker,
            structured_plan_review(
                summary="Plan looks sound.",
                human_requirements_resolved=True,
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
    )
    config = make_config(tmp_path, max_rounds=3)

    call_count = [0]
    def fake_repair(raw, gemini_cmd, *, expected_kind=None, reviewer_requirement_ids=None, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1 and expected_kind == "plan_review":
            return repaired_blocking
        return None

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_repair):
        result = run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert result == 0
    # The same-plan followup text must appear in the coder's prompt
    claude_prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert any("regression test for the parser" in p for p in claude_prompts), \
        "Same-plan followup from repaired review must appear in coder prompt"

from coding_review_agent_loop.repair import strip_unknown_prior_item_dispositions

def test_strip_unknown_prior_item_dispositions_removes_unknown_from_empty_ledger():
    raw = structured_plan_review(
        state="approved",
        summary="Plan review complete.",
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    result = strip_unknown_prior_item_dispositions(
        raw, allowed_ids=frozenset(), expected_kind="plan_review"
    )
    assert result is not None
    payload, json_end = json.JSONDecoder().raw_decode(result.lstrip())
    assert payload["prior_plan_item_dispositions"] == []
    trailing = result.lstrip()[json_end:]
    assert "<!-- AGENT_PLAN_STATE: approved -->" in trailing
    assert "-- Google Gemini" in trailing

def test_strip_unknown_prior_item_dispositions_preserves_valid_removes_unknown():
    raw = structured_plan_review(
        state="approved",
        summary="Plan review complete.",
        prior_plan_item_dispositions=[
            {"item_id": "item-1", "disposition": "resolved"},
            {"item_id": "item-9", "disposition": "resolved"},
        ],
        reviewer="Google Gemini",
    )
    result = strip_unknown_prior_item_dispositions(
        raw, allowed_ids=frozenset({"item-1"}), expected_kind="plan_review"
    )
    assert result is not None
    payload, _ = json.JSONDecoder().raw_decode(result.lstrip())
    assert payload["prior_plan_item_dispositions"] == [{"item_id": "item-1", "disposition": "resolved"}]

def test_strip_unknown_prior_item_dispositions_preserves_all_other_fields():
    raw = structured_plan_review(
        state="approved",
        summary="Looks good overall.",
        same_plan_followups=["Consider adding benchmarks."],
        future_followups=["Improve error messages later."],
        prior_plan_item_dispositions=[{"item_id": "item-9", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    result = strip_unknown_prior_item_dispositions(
        raw, allowed_ids=frozenset(), expected_kind="plan_review"
    )
    assert result is not None
    payload, _ = json.JSONDecoder().raw_decode(result.lstrip())
    assert payload["summary"] == "Looks good overall."
    assert payload["same_plan_followups"] == ["Consider adding benchmarks."]
    assert payload["future_followups"] == ["Improve error messages later."]
    assert payload["state"] == "approved"
    assert payload["prior_plan_item_dispositions"] == []

def test_strip_unknown_prior_item_dispositions_removes_from_pr_review():
    raw = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    result = strip_unknown_prior_item_dispositions(
        raw, allowed_ids=frozenset(), expected_kind="pr_review"
    )
    assert result is not None
    payload, _ = json.JSONDecoder().raw_decode(result.lstrip())
    assert payload["prior_item_dispositions"] == []
    assert payload["kind"] == "pr_review"

def test_strip_unknown_prior_item_dispositions_returns_none_if_nothing_to_remove():
    raw = structured_plan_review(
        state="approved",
        summary="Plan review complete.",
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    result = strip_unknown_prior_item_dispositions(
        raw, allowed_ids=frozenset({"item-1"}), expected_kind="plan_review"
    )
    assert result is None

def test_strip_unknown_prior_item_dispositions_returns_none_for_markdown():
    raw = "## Plan Review\n\nLooks good.\n\n<!-- AGENT_PLAN_STATE: approved -->\n-- Google Gemini"
    result = strip_unknown_prior_item_dispositions(
        raw, allowed_ids=frozenset(), expected_kind="plan_review"
    )
    assert result is None

def test_strip_unknown_prior_item_dispositions_returns_none_for_wrong_kind():
    raw = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    result = strip_unknown_prior_item_dispositions(
        raw, allowed_ids=frozenset(), expected_kind="plan_review"
    )
    assert result is None

def test_strip_unknown_prior_item_dispositions_returns_none_for_unsupported_kind():
    raw = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    result = strip_unknown_prior_item_dispositions(
        raw, allowed_ids=frozenset(), expected_kind="coder_followup"
    )
    assert result is None

def test_run_validated_agent_deterministically_strips_unknown_plan_review_without_repair(tmp_path):
    malformed_review = structured_plan_review(
        state="approved",
        summary="Plan approved.",
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the plan.",
            marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
            validate=lambda text: _validate_plan_review_response(
                text,
                reviewer="Google Gemini",
                unresolved_items=(),
            ),
            use_repair=True,
            repair_expected_kind="plan_review",
            repair_allowed_prior_item_ids=(),
            ledger_incomplete=False,
        )

    repair_mock.assert_not_called()
    parsed = json.loads(response.text.split("\n")[0])
    assert parsed["prior_plan_item_dispositions"] == []
    assert parsed["state"] == "approved"

def test_run_validated_agent_deterministically_strips_unknown_pr_review_without_repair(tmp_path):
    malformed_review = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the PR.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: _validate_review_response(
                text,
                reviewer="Google Gemini",
                unresolved_items=(),
            ),
            use_repair=True,
            repair_expected_kind="pr_review",
            repair_allowed_prior_item_ids=(),
            ledger_incomplete=False,
        )

    repair_mock.assert_not_called()
    parsed = json.loads(response.text.split("\n")[0])
    assert parsed["prior_item_dispositions"] == []
    assert parsed["kind"] == "pr_review"

def test_run_validated_agent_deterministic_strip_preserves_valid_removes_unknown_mixed(tmp_path):
    carried_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Must-fix prior item.",
        status="blocking",
    )
    malformed_review = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[
            {"item_id": "item-1", "disposition": "resolved"},
            {"item_id": "item-9", "disposition": "resolved"},
        ],
        reviewer="Google Gemini",
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the PR.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: _validate_review_response(
                text,
                reviewer="Google Gemini",
                unresolved_items=(carried_item,),
            ),
            use_repair=True,
            repair_expected_kind="pr_review",
            repair_allowed_prior_item_ids=("item-1",),
            ledger_incomplete=False,
        )

    repair_mock.assert_not_called()
    assert response.marker_value.dispositions[0].item_id == "item-1"
    assert len(response.marker_value.dispositions) == 1

def test_run_validated_agent_deterministic_strip_logs_removed_and_allowed_ids(tmp_path, capsys):
    malformed_review = structured_plan_review(
        state="approved",
        summary="Plan approved.",
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0, quiet=False)

    _run_validated_agent(
        runner,
        agent="gemini",
        config=config,
        prompt="Review the plan.",
        marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
        validate=lambda text: _validate_plan_review_response(
            text,
            reviewer="Google Gemini",
            unresolved_items=(),
        ),
        use_repair=True,
        repair_expected_kind="plan_review",
        repair_allowed_prior_item_ids=(),
        ledger_incomplete=False,
    )

    output = capsys.readouterr().err
    assert "deterministically removed unknown prior-item disposition ID(s)" in output
    assert "item-1" in output
    assert "(none)" in output

def test_run_validated_agent_deterministic_strip_falls_through_to_repair_on_secondary_failure(tmp_path):
    missing_item = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Must-fix item not dispositioned.",
        status="blocking",
    )
    malformed_review = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[{"item_id": "item-9", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    repaired_review = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[{"item_id": "item-2", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch(
        "coding_review_agent_loop.orchestrator.attempt_repair", return_value=repaired_review
    ) as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the PR.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: _validate_review_response(
                text,
                reviewer="Google Gemini",
                unresolved_items=(missing_item,),
            ),
            use_repair=True,
            repair_expected_kind="pr_review",
            repair_allowed_prior_item_ids=("item-2",),
            ledger_incomplete=False,
        )

    repair_mock.assert_called_once()
    assert response.text == repaired_review

def test_run_validated_agent_real_264_shape_approved_plan_review_same_round_item1_future_followups(tmp_path):
    malformed_review = structured_plan_review(
        state="approved",
        summary="Plan is sound.",
        future_followups=["Consider adding retries."],
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved", "note": "Now covered."}],
        reviewer="Google Gemini",
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the plan.",
            marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
            validate=lambda text: _validate_plan_review_response(
                text,
                reviewer="Google Gemini",
                unresolved_items=(),
                current_round_items=(),
            ),
            use_repair=True,
            repair_expected_kind="plan_review",
            repair_allowed_prior_item_ids=(),
            ledger_incomplete=False,
        )

    repair_mock.assert_not_called()
    parsed = json.loads(response.text.split("\n")[0])
    assert parsed["prior_plan_item_dispositions"] == []
    assert parsed["state"] == "approved"
    assert parsed["future_followups"] == ["Consider adding retries."]
    assert parsed["summary"] == "Plan is sound."

def test_run_validated_agent_deterministic_strip_skipped_when_ledger_incomplete(tmp_path):
    malformed_review = structured_plan_review(
        state="approved",
        summary="Plan approved.",
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        with pytest.raises(AgentLoopError) as exc_info:
            _run_validated_agent(
                runner,
                agent="gemini",
                config=config,
                prompt="Review the plan.",
                marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
                validate=lambda text: _validate_plan_review_response(
                    text,
                    reviewer="Google Gemini",
                    unresolved_items=(),
                ),
                use_repair=True,
                repair_expected_kind="plan_review",
                repair_allowed_prior_item_ids=(),
                ledger_incomplete=True,
            )

    repair_mock.assert_not_called()
    assert "item-1" in str(exc_info.value)

def test_run_validated_agent_recovers_plan_revision_human_ack_from_message_text(tmp_path):
    human_requirements = (
        HumanReviewRequirement(
            source_type="Issue comment",
            author="wwind123",
            created_at="2026-06-05T00:00:00Z",
            url="https://github.com/OWNER/REPO/issues/237#issuecomment-1",
            body="Cover stdout acknowledgement recovery.",
        ),
    )
    context = render_coder_human_requirements_prompt_context(
        human_requirements,
        requirement_scope="planning requirements",
        full_omission_fallback="Fetch the issue discussion directly before revising the plan.",
    )
    response_file = structured_plan_revision(reviewer="Anthropic Claude")
    acknowledgement = (
        "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n"
        "### Human requirements\n"
        "- Requirement 1: The revised plan covers stdout acknowledgement recovery.\n"
    )
    message_text = structured_plan_revision(
        reviewer="Anthropic Claude",
        human_requirements=acknowledgement,
    )
    runner = FakeRunner(
        claude_outputs=[(message_text, 0)],
        public_response_outputs=[{"text": response_file}],
    )
    config = make_config(tmp_path, coder="claude", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="claude",
            config=config,
            prompt="Revise the plan.",
            marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
            validate=_plan_revision_validate_with_human_requirements(human_requirements),
            use_repair=True,
            repair_expected_kind="plan_revision",
            repair_surfaced_requirement_ids=context.surfaced_requirement_ids,
            repair_requires_direct_discussion_ack=context.requires_direct_discussion_ack,
        )

    repair_mock.assert_not_called()
    assert "<!-- HUMAN_REQUIREMENTS_ADDRESSED -->" in response.text
    assert "### Human requirements" in response.text
    assert response.text.index("### Human requirements") < response.text.index(
        "<!-- AGENT_PLAN_STATE: blocking -->"
    )

@pytest.mark.parametrize(
    "evidence",
    [
        "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n",
        "\n### Human requirements\n- Requirement 1: Covered.\n",
        "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n### Human requirements\n- Requirement 99: Covered.\n",
        "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n### Human requirements\n- Requirement 1: Covered.\n- Requirement 1: Covered again.\n",
    ],
)
def test_run_validated_agent_refuses_invalid_plan_revision_human_ack_evidence(tmp_path, evidence):
    human_requirements = (
        HumanReviewRequirement(
            source_type="Issue comment",
            author="wwind123",
            created_at="2026-06-05T00:00:00Z",
            url="https://github.com/OWNER/REPO/issues/237#issuecomment-1",
            body="Cover stdout acknowledgement recovery.",
        ),
    )
    context = render_coder_human_requirements_prompt_context(
        human_requirements,
        requirement_scope="planning requirements",
        full_omission_fallback="Fetch the issue discussion directly before revising the plan.",
    )
    runner = FakeRunner(
        claude_outputs=[
            (
                structured_plan_revision(
                    reviewer="Anthropic Claude",
                    human_requirements=evidence,
                ),
                0,
            )
        ],
        public_response_outputs=[{"text": structured_plan_revision(reviewer="Anthropic Claude")}],
    )
    config = make_config(tmp_path, coder="claude", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=None) as repair_mock:
        with pytest.raises(AgentLoopError, match="No review result was recorded"):
            _run_validated_agent(
                runner,
                agent="claude",
                config=config,
                prompt="Revise the plan.",
                marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
                validate=_plan_revision_validate_with_human_requirements(human_requirements),
                use_repair=True,
                repair_expected_kind="plan_revision",
                repair_surfaced_requirement_ids=context.surfaced_requirement_ids,
                repair_requires_direct_discussion_ack=context.requires_direct_discussion_ack,
            )

    repair_mock.assert_called_once()

def test_run_validated_agent_refuses_plan_revision_missing_direct_discussion_ack(tmp_path):
    context = render_coder_human_requirements_prompt_context(
        (),
        requirement_scope="planning requirements",
        full_omission_fallback="Fetch the issue discussion directly before revising the plan.",
    )
    acknowledgement = (
        "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n"
        "### Human requirements\n"
        "- The prompt omitted the detailed signed human requirements.\n"
    )
    runner = FakeRunner(
        claude_outputs=[
            (
                structured_plan_revision(
                    reviewer="Anthropic Claude",
                    human_requirements=acknowledgement,
                ),
                0,
            )
        ],
        public_response_outputs=[{"text": structured_plan_revision(reviewer="Anthropic Claude")}],
    )
    config = make_config(tmp_path, coder="claude", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=None):
        with pytest.raises(AgentLoopError, match="No review result was recorded"):
            _run_validated_agent(
                runner,
                agent="claude",
                config=config,
                prompt="Revise the plan.",
                marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
                validate=lambda text: (
                    _validate_plan_revision_response(text, unresolved_items=()),
                    validate_human_requirements_acknowledgement(
                        text,
                        surfaced_requirement_ids=(),
                        requires_direct_discussion_ack=True,
                    ),
                )[0],
                use_repair=True,
                repair_expected_kind="plan_revision",
                repair_surfaced_requirement_ids=context.surfaced_requirement_ids,
                repair_requires_direct_discussion_ack=True,
            )

def test_run_validated_agent_refuses_conflicting_plan_revision_human_ack_blocks(tmp_path):
    human_requirements = (
        HumanReviewRequirement(
            source_type="Issue comment",
            author="wwind123",
            created_at="2026-06-05T00:00:00Z",
            url="https://github.com/OWNER/REPO/issues/237#issuecomment-1",
            body="Cover stdout acknowledgement recovery.",
        ),
    )
    context = render_coder_human_requirements_prompt_context(
        human_requirements,
        requirement_scope="planning requirements",
        full_omission_fallback="Fetch the issue discussion directly before revising the plan.",
    )
    first = (
        "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n"
        "### Human requirements\n- Requirement 1: Covered by the parser step.\n"
    )
    second = (
        "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n"
        "### Human requirements\n- Requirement 1: Covered by the orchestrator step.\n"
    )
    runner = FakeRunner(
        claude_outputs=[
            (
                structured_plan_revision(
                    reviewer="Anthropic Claude",
                    human_requirements=first,
                )
                + "\n\n"
                + second,
                0,
            )
        ],
        public_response_outputs=[{"text": structured_plan_revision(reviewer="Anthropic Claude")}],
    )
    config = make_config(tmp_path, coder="claude", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=None):
        with pytest.raises(AgentLoopError, match="No review result was recorded"):
            _run_validated_agent(
                runner,
                agent="claude",
                config=config,
                prompt="Revise the plan.",
                marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
                validate=_plan_revision_validate_with_human_requirements(human_requirements),
                use_repair=True,
                repair_expected_kind="plan_revision",
                repair_surfaced_requirement_ids=context.surfaced_requirement_ids,
                repair_requires_direct_discussion_ack=context.requires_direct_discussion_ack,
            )

def test_run_validated_agent_strip_unknown_disposition_then_ack_recovery_succeeds(tmp_path):
    # When the response file has both an unknown prior-item disposition AND a missing ack,
    # block 2 strips the disposition deterministically and the new ack recovery path (#403)
    # then reconstructs the ack from message_text — both issues fixed, no repair pass needed.
    human_requirements = (
        HumanReviewRequirement(
            source_type="Issue comment",
            author="wwind123",
            created_at="2026-06-05T00:00:00Z",
            url="https://github.com/OWNER/REPO/issues/237#issuecomment-1",
            body="Cover stdout acknowledgement recovery.",
        ),
    )
    context = render_coder_human_requirements_prompt_context(
        human_requirements,
        requirement_scope="planning requirements",
        full_omission_fallback="Fetch the issue discussion directly before revising the plan.",
    )
    acknowledgement = (
        "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n"
        "### Human requirements\n- Requirement 1: Covered.\n"
    )
    response_file = structured_plan_revision(
        reviewer="Anthropic Claude",
        prior_plan_item_dispositions=[
            {"item_id": "item-unknown", "disposition": "resolved", "note": "Covered."}
        ],
    )
    runner = FakeRunner(
        claude_outputs=[
            (
                structured_plan_revision(
                    reviewer="Anthropic Claude",
                    human_requirements=acknowledgement,
                ),
                0,
            )
        ],
        public_response_outputs=[{"text": response_file}],
    )
    config = make_config(tmp_path, coder="claude", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="claude",
            config=config,
            prompt="Revise the plan.",
            marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
            validate=_plan_revision_validate_with_human_requirements(human_requirements),
            use_repair=True,
            repair_expected_kind="plan_revision",
            repair_surfaced_requirement_ids=context.surfaced_requirement_ids,
            repair_requires_direct_discussion_ack=context.requires_direct_discussion_ack,
            repair_allowed_prior_item_ids=(),
        )

    repair_mock.assert_not_called()
    assert "<!-- HUMAN_REQUIREMENTS_ADDRESSED -->" in response.text
    assert "item-unknown" not in response.text

@pytest.mark.parametrize(
    "stdout",
    [
        structured_pr_review(reviewer="OpenAI Codex"),
        "diagnostic output without a structured response",
    ],
)
def test_run_validated_agent_refuses_unrecoverable_stdout_when_response_file_markdown(tmp_path, stdout):
    runner = FakeRunner(
        codex_outputs=[{"public_response": "legacy markdown", "stdout": stdout}],
        public_response_outputs=[{"text": "### Update\nFixed it.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"}],
    )
    config = make_config(tmp_path, coder="codex", agent_max_retries=0)

    with pytest.raises(AgentLoopError, match="No review result was recorded"):
        _run_validated_agent(
            runner,
            agent="codex",
            config=config,
            prompt="Address feedback.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: _validate_coder_followup_response(
                text,
                unresolved_items=(),
                human_requirements=(),
            ),
            repair_expected_kind="coder_followup",
        )

def test_run_validated_agent_refuses_multiple_stdout_structured_candidates(tmp_path):
    first = structured_coder_followup(
        state="approved",
        summary="First candidate.",
        reviewer="OpenAI Codex",
    )
    second = structured_coder_followup(
        state="approved",
        summary="Second candidate.",
        reviewer="OpenAI Codex",
    )
    runner = FakeRunner(
        codex_outputs=[{"public_response": "legacy markdown", "stdout": first + "\n\n" + second}],
        public_response_outputs=[{"text": "### Update\nFixed it.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"}],
    )
    config = make_config(tmp_path, coder="codex", agent_max_retries=0)

    with pytest.raises(AgentLoopError, match="No review result was recorded"):
        _run_validated_agent(
            runner,
            agent="codex",
            config=config,
            prompt="Address feedback.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: _validate_coder_followup_response(
                text,
                unresolved_items=(),
                human_requirements=(),
            ),
            repair_expected_kind="coder_followup",
        )

def test_run_validated_agent_keeps_valid_response_file_authoritative_over_noisy_stdout(tmp_path):
    valid_followup = structured_coder_followup(
        state="approved",
        summary="Response file wins.",
        reviewer="OpenAI Codex",
    )
    runner = FakeRunner(
        codex_outputs=[{"public_response": "ignored message", "stdout": "unrelated noisy diagnostics"}],
        public_response_outputs=[{"text": valid_followup}],
    )
    config = make_config(tmp_path, coder="codex", agent_max_retries=0)

    response = _run_validated_agent(
        runner,
        agent="codex",
        config=config,
        prompt="Address feedback.",
        marker_description="<!-- AGENT_STATE: approved|blocking -->",
        validate=lambda text: _validate_coder_followup_response(
            text,
            unresolved_items=(),
            human_requirements=(),
        ),
        repair_expected_kind="coder_followup",
    )

    assert response.text == valid_followup

def test_structured_plan_revision_transient_terms_before_footer_runs_repair(tmp_path):
    malformed_revision = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_revision",
                "state": "blocking",
                "summary": "Revise handling for 429, quota, resource exhausted, transient, and timeout.",
                "prior_plan_item_dispositions": [],
                "plan_steps": [
                    "Separate public-response validation from transient raw diagnostics.",
                    "Keep capacity and quota retry handling for raw provider failures.",
                ],
            }
        )
        + "\n## Revised plan\nProse before the AGENT_PLAN_STATE footer is invalid.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    repaired_revision = structured_plan_revision(
        summary="Revised transient classifier plan.",
        plan_steps=[
            "Separate public-response validation from transient raw diagnostics.",
            "Keep capacity and quota retry handling for raw provider failures.",
        ],
        reviewer="Anthropic Claude",
    )
    runner = FakeRunner(claude_outputs=[malformed_revision])
    config = make_config(tmp_path, coder="claude", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=repaired_revision) as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="claude",
            config=config,
            prompt="Revise the plan.",
            marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
            validate=_validate_plan_revision_response,
            use_repair=True,
            repair_expected_kind="plan_revision",
            repair_surfaced_requirement_ids=("Requirement 1",),
        )

    assert response.text == repaired_revision
    repair_mock.assert_called_once_with(
        malformed_revision,
        config.gemini_cmd,
        expected_kind="plan_revision",
    )
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)

def test_run_validated_agent_repairs_unknown_prior_item_disposition_when_ledger_complete(tmp_path):
    malformed_review = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the PR.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: _validate_review_response(
                text,
                reviewer="Google Gemini",
                unresolved_items=(),
            ),
            use_repair=True,
            repair_expected_kind="pr_review",
            repair_allowed_prior_item_ids=(),
            ledger_incomplete=False,
    )

    repair_mock.assert_not_called()
    parsed_response = json.loads(response.text.split("\n")[0])
    assert parsed_response["prior_item_dispositions"] == []
    assert parsed_response["state"] == "approved"
    assert parsed_response["summary"] == "LGTM."

def test_run_validated_agent_skips_unknown_prior_item_repair_when_ledger_incomplete(tmp_path):
    malformed_review = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        with pytest.raises(AgentLoopError) as exc_info:
            _run_validated_agent(
                runner,
                agent="gemini",
                config=config,
                prompt="Review the PR.",
                marker_description="<!-- AGENT_STATE: approved|blocking -->",
                validate=lambda text: _validate_review_response(
                    text,
                    reviewer="Google Gemini",
                    unresolved_items=(),
                ),
                use_repair=True,
                repair_expected_kind="pr_review",
                repair_allowed_prior_item_ids=(),
                ledger_incomplete=True,
            )

    repair_mock.assert_not_called()
    assert "item-1" in str(exc_info.value)

def test_envelope_normalization_coder_followup_duplicate_state_footer():
    """attempt_envelope_normalization handles coder_followup with a duplicate AGENT_STATE footer."""
    raw = (
        structured_coder_followup(
            state="approved",
            reviewer="Anthropic Claude",
            addressed_items=["item-1"],
        )
        + "\n\n<!-- AGENT_STATE: approved -->"
    )

    normalized = attempt_envelope_normalization(raw, expected_kind="coder_followup")

    assert normalized is not None
    parsed = validate_structured_coder_followup(normalized)
    assert parsed is not None
    assert parsed.addressed_items == ("item-1",)
    assert normalized.count("<!-- AGENT_STATE: approved -->") == 1

def test_envelope_normalization_coder_followup_trailing_prose_after_signature():
    """attempt_envelope_normalization strips trailing prose after coder_followup signature."""
    raw = (
        structured_coder_followup(
            state="blocking",
            reviewer="Anthropic Claude",
            remaining_items=["item-2"],
        )
        + "\n\nExtra prose that should be stripped."
    )

    normalized = attempt_envelope_normalization(raw, expected_kind="coder_followup")

    assert normalized is not None
    assert "Extra prose" not in normalized
    parsed = validate_structured_coder_followup(normalized)
    assert parsed is not None
    assert parsed.remaining_items == ("item-2",)

def test_envelope_normalization_coder_followup_returns_none_when_prose_before_footer():
    """attempt_envelope_normalization returns None for coder_followup with prose before the footer."""
    raw = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "approved",
                "summary": "Done.",
                "addressed_items": [],
                "remaining_items": [],
                "human_requirements": {"addressed_ids": [], "checked_discussion_directly": False},
            }
        )
        + "\nSome unexpected prose here.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude"
    )

    assert attempt_envelope_normalization(raw, expected_kind="coder_followup") is None

def test_strip_unknown_prior_item_dispositions_tightly_packed_no_newline_before_footer():
    """strip_unknown_prior_item_dispositions inserts a newline when the original had none."""
    payload = {
        "schema_version": 1,
        "kind": "pr_review",
        "state": "approved",
        "summary": "LGTM.",
        "blocking_items": [],
        "same_pr_followups": [],
        "future_followups": [],
        "prior_item_dispositions": [{"item_id": "item-99", "disposition": "resolved"}],
    }
    # Tightly packed: no newline between JSON and footer
    raw = json.dumps(payload) + "<!-- AGENT_STATE: approved -->\n-- Google Gemini"

    result = strip_unknown_prior_item_dispositions(
        raw, allowed_ids=frozenset(), expected_kind="pr_review"
    )

    assert result is not None
    # Validate parses correctly even after tight packing
    parsed_payload, json_end = json.JSONDecoder().raw_decode(result.lstrip())
    assert parsed_payload["prior_item_dispositions"] == []
    tail = result.lstrip()[json_end:]
    assert "<!-- AGENT_STATE: approved -->" in tail
    assert "-- Google Gemini" in tail
    # The footer must be separated from the JSON by at least a newline
    assert tail.startswith("\n")

def test_strip_unknown_prior_item_dispositions_tightly_packed_result_validates():
    """Tight-packing case validates successfully through parse_structured_pr_review."""
    payload = {
        "schema_version": 1,
        "kind": "pr_review",
        "state": "approved",
        "summary": "LGTM.",
        "blocking_items": [],
        "same_pr_followups": [],
        "future_followups": [],
        "prior_item_dispositions": [{"item_id": "item-99", "disposition": "resolved"}],
    }
    raw = json.dumps(payload) + "<!-- AGENT_STATE: approved -->\n-- Google Gemini"

    result = strip_unknown_prior_item_dispositions(
        raw, allowed_ids=frozenset(), expected_kind="pr_review"
    )

    assert result is not None
    parsed = parse_structured_pr_review(result, reviewer="Google Gemini")
    assert parsed is not None
    assert parsed.dispositions == ()

def test_run_validated_agent_combined_envelope_and_disposition_fix(tmp_path):
    """When a response has both an envelope defect and unknown prior dispositions,
    stripping dispositions from the envelope-normalized candidate recovers it."""
    # Build a plan_review with an unknown disposition AND a duplicate footer (envelope defect).
    # strip_unknown_prior_item_dispositions on the original fails to validate because the
    # duplicate footer is still present. Envelope normalization on the original produces a
    # normalized candidate; stripping dispositions from that candidate should succeed.
    base = structured_plan_review(
        state="approved",
        summary="Plan approved.",
        prior_plan_item_dispositions=[{"item_id": "item-99", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    # Add a duplicate footer to create the envelope defect
    malformed = base + "\n\n<!-- AGENT_PLAN_STATE: approved -->"

    runner = FakeRunner(gemini_outputs=[malformed])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the plan.",
            marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
            validate=lambda text: _validate_plan_review_response(
                text,
                reviewer="Google Gemini",
                unresolved_items=(),
            ),
            use_repair=True,
            repair_expected_kind="plan_review",
            repair_allowed_prior_item_ids=(),
            ledger_incomplete=False,
        )

    repair_mock.assert_not_called()
    parsed = json.loads(response.text.split("\n")[0])
    assert parsed["prior_plan_item_dispositions"] == []
    assert parsed["state"] == "approved"

def test_run_validated_agent_rejects_repair_that_invents_prior_item_id(tmp_path):
    # item-3 is the legitimate carried prior item; item-1 is unknown and gets stripped
    # deterministically, but item-3 is then missing → re-validation fails → falls through
    # to generative repair → repair invents item-2 (also unknown) → rejected.
    carried_item_3 = UnresolvedReviewItem(
        item_id="item-3",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Must-fix prior item.",
        status="blocking",
    )
    malformed_review = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    repaired_review = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[{"item_id": "item-2", "disposition": "resolved"}],
        reviewer="Google Gemini",
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=repaired_review):
        with pytest.raises(AgentLoopError) as exc_info:
            _run_validated_agent(
                runner,
                agent="gemini",
                config=config,
                prompt="Review the PR.",
                marker_description="<!-- AGENT_STATE: approved|blocking -->",
                validate=lambda text: _validate_review_response(
                    text,
                    reviewer="Google Gemini",
                    unresolved_items=(carried_item_3,),
                ),
                use_repair=True,
                repair_expected_kind="pr_review",
                repair_allowed_prior_item_ids=("item-3",),
            )

    assert "item-2" in str(exc_info.value)

def test_run_validated_agent_preserves_valid_disposition_when_repair_removes_unknown(tmp_path):
    carried_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Keep this prior item.",
        status="blocking",
    )
    malformed_review = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[
            {"item_id": "item-1", "disposition": "resolved"},
            {"item_id": "item-9", "disposition": "resolved"},
        ],
        reviewer="Google Gemini",
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the PR.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: _validate_review_response(
                text,
                reviewer="Google Gemini",
                unresolved_items=(carried_item,),
            ),
            use_repair=True,
            repair_expected_kind="pr_review",
            repair_allowed_prior_item_ids=("item-1",),
        )

    repair_mock.assert_not_called()
    assert response.marker_value.dispositions[0].item_id == "item-1"
    assert len(response.marker_value.dispositions) == 1

def test_run_validated_agent_repairs_unknown_plan_revision_prior_disposition(tmp_path):
    active_item = UnresolvedReviewItem(
        item_id="item-12",
        reviewer="Google Gemini",
        source_round=5,
        text="Active must-fix item.",
        status="blocking",
    )
    malformed_revision = structured_plan_revision(
        prior_plan_item_dispositions=[
            {"item_id": "item-15", "disposition": "resolved"},
        ],
    )
    runner = FakeRunner(claude_outputs=[malformed_revision])
    config = make_config(tmp_path, coder="claude", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="claude",
            config=config,
            prompt="Revise the plan.",
            marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
            validate=lambda text: _validate_plan_revision_response(
                text,
                unresolved_items=(active_item,),
            ),
            use_repair=True,
            repair_expected_kind="plan_revision",
            repair_allowed_prior_item_ids=("item-12",),
            ledger_incomplete=False,
        )

    repair_mock.assert_not_called()
    parsed_response = json.loads(response.text.split("\n")[0])
    assert parsed_response["prior_plan_item_dispositions"] == []

def test_run_validated_agent_plan_revision_unknown_prior_disposition_fails_when_ledger_incomplete(tmp_path):
    active_item = UnresolvedReviewItem(
        item_id="item-12",
        reviewer="Google Gemini",
        source_round=5,
        text="Active must-fix item.",
        status="blocking",
    )
    malformed_revision = structured_plan_revision(
        prior_plan_item_dispositions=[
            {"item_id": "item-15", "disposition": "resolved"},
            {"item_id": "item-18", "disposition": "resolved"},
        ],
    )
    runner = FakeRunner(claude_outputs=[malformed_revision])
    config = make_config(tmp_path, coder="claude", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        with pytest.raises(AgentLoopError) as exc_info:
            _run_validated_agent(
                runner,
                agent="claude",
                config=config,
                prompt="Revise the plan.",
                marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
                validate=lambda text: _validate_plan_revision_response(
                    text,
                    unresolved_items=(active_item,),
                ),
                use_repair=True,
                repair_expected_kind="plan_revision",
                repair_allowed_prior_item_ids=("item-12",),
                ledger_incomplete=True,
            )

    repair_mock.assert_not_called()
    assert "item-15" in str(exc_info.value)
    assert "item-18" in str(exc_info.value)


# --- discuss_agenda repair support tests (#467) ---


def _agenda_json() -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "kind": "discuss_agenda",
            "consensus": ["The issue is well-motivated."],
            "disagreements": [
                {
                    "topic": "Scope",
                    "positions": {"Codex": "Narrow enough.", "Gemini": "Too broad."},
                    "question_for_next_round": "Would splitting resolve the objection?",
                }
            ],
            "missing_facts": [],
        }
    )


def test_envelope_normalization_discuss_agenda_reversed_footer_and_signature():
    raw = _agenda_json() + "\n-- Anthropic Claude\n<!-- AGENT_PLAN_STATE: approved -->"

    normalized = attempt_envelope_normalization(raw, expected_kind="discuss_agenda")

    assert normalized is not None
    from coding_review_agent_loop.protocol import parse_structured_discuss_agenda

    parsed = parse_structured_discuss_agenda(normalized)
    assert parsed is not None
    assert parsed.disagreements[0].topic == "Scope"


def test_build_repair_prompt_accepts_discuss_agenda_expected_kind():
    from coding_review_agent_loop.repair import _build_repair_prompt

    prompt = _build_repair_prompt("garbage", expected_kind="discuss_agenda")
    assert "You MUST repair this response as `discuss_agenda`." in prompt
    assert "Valid Format F — Discuss Agenda" in prompt
    assert "question_for_next_round" in prompt
    assert "Never invent topics, debater positions, facts, or questions" in prompt
    assert "If no agenda content is recoverable" in prompt
