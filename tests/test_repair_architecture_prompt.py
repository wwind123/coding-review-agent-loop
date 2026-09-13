import copy
import json

import pytest

from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.protocol_markers import sanitize_historical_text
from coding_review_agent_loop.repair import _build_repair_prompt
from coding_review_agent_loop.repair_preservation import validate_repair_preservation


@pytest.mark.parametrize('kind', [
    'plan_state', 'plan_revision', 'plan_review', 'pr_review',
    'coder_followup', 'issue_implementation',
])
def test_repair_prompt_preserves_optional_architecture_assessment(kind):
    prompt = _build_repair_prompt('{}', expected_kind=kind)
    assert 'not exhaustive allowlists' in prompt
    assert 'copy the COMPLETE `architecture_impact` object' in prompt
    assert 'invent an assessment when the source has none' in prompt
    assert 'compare `architecture_impact` recursively' in prompt


def test_marker_only_repair_cannot_drop_architecture_assessment():
    original = {
        'schema_version': 1, 'kind': 'pr_review', 'state': 'blocking',
        'summary': 'The unfiled-scope warning is suppressed.',
        'blocking_items': ['Keep the `AGENT_SPLIT_UNFILED_WARNING` warning for one-shot plans.'],
        'same_pr_followups': [], 'future_followups': [],
        'prior_item_dispositions': [],
        'architecture_impact': {
            'status': 'changed', 'rationale': 'Routing changes.',
            'affected_components': ['orchestrator.py'], 'dependencies': [],
            'execution_data_flows': ['Approval then routing'],
            'persistence': ['Decision checkpoint'], 'public_contracts': ['auto'],
            'security_boundaries': [], 'canonical_document_action': 'updated-in-pr',
            'canonical_document_path': 'ARCHITECTURE.md',
            'canonical_document_rationale': 'Documents routing.',
            'uncertainty': ['Resume coverage is missing.'],
        },
    }
    repaired = copy.deepcopy(original)
    repaired['blocking_items'] = [sanitize_historical_text(original['blocking_items'][0])]
    raw = json.dumps(original)
    validate_repair_preservation(raw, json.dumps(repaired))
    del repaired['architecture_impact']
    with pytest.raises(AgentLoopError, match='architecture_impact'):
        validate_repair_preservation(raw, json.dumps(repaired))


def test_repair_prompt_distinguishes_reserved_syntax_from_bare_identifiers():
    raw = json.dumps({'summary': 'AGENT_SPLIT_UNFILED_WARNING; AGENT_PLAN_EXECUTION_DECISION'})
    prompt = _build_repair_prompt(raw, expected_kind='pr_review')
    section = prompt.split('## Registry-detected reserved syntax in this source:\n')[1]
    detected = json.loads(section.splitlines()[0])
    assert detected == ['AGENT_SPLIT_UNFILED_WARNING']
    assert 'Do not rename other bare code identifiers' in section
