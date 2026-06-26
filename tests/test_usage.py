from agent_loop_helpers import *  # noqa: F403


def test_codex_usage_summary_records_exact_tokens_from_jsonl_and_public_response(tmp_path):
    public_response = "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    runner = FakeRunner(
        codex_outputs=[
            {
                "public_response": public_response,
                "stdout": "\n".join(
                    [
                        json.dumps({"type": "turn.started"}),
                        json.dumps(
                            {
                                "type": "turn.completed",
                                "usage": {
                                    "input_tokens": 200,
                                    "cached_input_tokens": 40,
                                    "output_tokens": 50,
                                    "reasoning_tokens": 10,
                                    "total_tokens": 300,
                                },
                            }
                        ),
                    ]
                ),
            }
        ]
    )
    config = make_config(tmp_path, reviewer="codex")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments == [f"**Review verdict:** Approved\n\n{public_response}"]
    summary = read_usage_summary(tmp_path / "logs")
    assert summary["totals"]["exact_calls"] == 1
    assert summary["totals"]["estimated_calls"] == 0
    assert summary["totals"]["input_tokens"] == 200
    assert summary["totals"]["cached_input_tokens"] == 40
    assert summary["totals"]["output_tokens"] == 50
    assert summary["totals"]["reasoning_tokens"] == 10
    assert summary["totals"]["total_tokens"] == 300
    assert summary["calls"][0]["raw_backend_usage"]["cached_input_tokens"] == 40
    assert summary["calls"][0]["validation_status"] == "validated"

def test_usage_summary_estimates_tokens_when_backend_exposes_none(tmp_path):
    public_response = "LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(gemini_outputs=[public_response])
    config = make_config(tmp_path, reviewer="gemini")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    summary = read_usage_summary(tmp_path / "logs")
    call = summary["calls"][0]
    assert call["usage"]["mode"] == "estimated"
    assert call["usage"]["input_tokens"] == max(1, (call["usage"]["input_bytes"] + 3) // 4)
    assert call["usage"]["output_tokens"] == max(1, (call["usage"]["output_bytes"] + 3) // 4)
    assert call["usage"]["output_chars"] > len(public_response)

def test_usage_summary_keeps_retry_attempts_and_marks_only_validated_call_successful(tmp_path):
    near_miss = "LGTM.\nAGENT_STATE: approved.\n-- Google Gemini"
    valid = "LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(gemini_outputs=[near_miss, valid])
    config = make_config(tmp_path, reviewer="gemini")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    summary = read_usage_summary(tmp_path / "logs")
    assert len(summary["calls"]) == 2
    assert summary["totals"]["call_count"] == 2
    assert summary["totals"]["success_count"] == 1
    assert summary["calls"][0]["validation_status"] == "invalid"
    assert summary["calls"][1]["validation_status"] == "validated"

def test_plan_first_issue_run_writes_one_summary_for_planning_implementation_and_review(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Plan:\n- Implement usage logging.\n<!-- AGENT_PLAN_STATE: approved -->\n-- Anthropic Claude",
            "Opened PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Plan reviewed.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(
        runner,
        issue_number=56,
        config=config,
        plan_first=True,
        implement_after_approval=True,
    ) == 0

    summary = read_usage_summary(tmp_path / "logs")
    assert len(list((tmp_path / "logs").glob("*-usage-summary.json"))) == 1
    assert summary["totals"]["call_count"] == 4
    assert set(summary["per_agent"]) == {"claude", "codex"}
    assert [call["agent"] for call in summary["calls"]] == ["claude", "codex", "claude", "codex"]
