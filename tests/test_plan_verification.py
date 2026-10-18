"""Plan-verification tests: plan approval classification (#1543)."""

from types import SimpleNamespace
from issue_loop_test_helpers import _M1088_ACTOR, _m1090_legacy_history
from agent_loop_helpers import make_config


def test_1090_poster_incompleteness_proof_is_opt_in(tmp_path):
    from coding_review_agent_loop.plan_verification import classify_plan_approval

    comments = [
        SimpleNamespace(body=c["body"], author=c["author"]["login"], author_id=c["_rest_author_id"])
        for c in _m1090_legacy_history(
            ("Codex", "OpenAI Codex", "approved"),
            ("Gemini", "Google Gemini", "blocking"),
        )
    ]

    def classify(**kwargs):
        return classify_plan_approval(
            comments, config=make_config(tmp_path), issue_number=56,
            trusted_actor=_M1088_ACTOR, execution_mode_guard=False, **kwargs,
        )

    default = classify()
    assert default.state == "undetermined"
    assert "unanchored plain plan history" in default.reason
    proven = classify(prove_poster_incompleteness=True)
    assert proven.state == "unfinished"
    assert proven.reason == "the latest legacy plan round is provably incomplete"
