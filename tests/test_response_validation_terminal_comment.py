"""Response-validation tests: structured no-PR terminal comments (#1543)."""

import json
from coding_review_agent_loop.protocol import validate_structured_issue_implementation
import coding_review_agent_loop.response_validation as response_validation
from issue_loop_test_helpers import FakeRunner
from agent_loop_helpers import make_config, structured_issue_implementation


def test_959_structured_no_pr_terminal_comment_omits_matrix_evidence(tmp_path):
    raw = structured_issue_implementation(pr_number=None, summary="Blocked before a PR.")
    payload, end = json.JSONDecoder().raw_decode(raw)
    payload["risk_test_matrix_claims"] = [{
        "row_id": "row-0",
        "execution_refs": ["turn:observation-1"],
        "test_identifiers": ["tests/test_x.py::test_y"],
        "test_locations": ["tests/test_x.py"],
        "workflow_path_claim": "Semantic claim only.",
        "outcome_assertions": ["outcome"],
        "forbidden_effect_assertions": ["forbidden"],
        "caveats": [],
    }]
    parsed = validate_structured_issue_implementation(json.dumps(payload) + raw[end:])
    assert parsed is not None and parsed.pr_number is None
    assert parsed.risk_test_matrix_evidence is None
    runner = FakeRunner()
    config = make_config(tmp_path, coder="claude", reviewer="codex")

    response_validation._post_structured_issue_implementation_terminal_comment(
        runner, config=config, issue_number=56, parsed=parsed, model_used=None
    )

    assert runner.comments
    body = runner.comments[-1]
    assert "Blocked before a PR." in body
    assert "matrix evidence" not in body
    assert "<details>" not in body
