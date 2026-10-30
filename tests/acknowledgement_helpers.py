"""Durable historical PR acknowledgement fixtures (no live GitHub effects)."""
from dataclasses import replace
from coding_review_agent_loop.protocol import UnresolvedReviewItem, UNKNOWN_MACHINE_AUTHORITY
from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata
from coding_review_agent_loop.unresolved_items import _upsert_human_requirements_ack_item

ACK_TEXT = (
    "Reviewer(s) Codex, Gemini approved without acknowledging the signed human requirements. "
    "Coder must address the human requirements and ensure the reviewer explicitly resolves them before approval."
)
ACK_ID = "item-human-requirements-acknowledgement"
UNBOUND_NOTE = "Known machine item could not be bound to a failed head."
UNTRUSTED_NOTE = "Synthetic machine record lacked trusted orchestrator lineage."


def legacy_ack(**overrides):
    item = UnresolvedReviewItem(
        item_id="item-13", reviewer="Orchestrator", source_round=1,
        text=ACK_TEXT, status="blocking", source_status="blocking",
        resolution_owners=("Orchestrator",), owner_states=(("Orchestrator", "pending"),),
    )
    return replace(item, **overrides)


def promoted_ack(item=None, **overrides):
    item = item or legacy_ack()
    return replace(item, **{
        "authority": UNKNOWN_MACHINE_AUTHORITY, "obligation_kind": "unknown",
        "obligation_identity": f"unknown:{item.item_id}", "lifecycle": "repair_required",
        "resolution_owners": (), "owner_states": (), "notes": (UNBOUND_NOTE,),
        **overrides,
    })


def dedicated_ack():
    return _upsert_human_requirements_ack_item((), source_round=1, text=ACK_TEXT)[0]


def unrelated_item():
    return UnresolvedReviewItem(
        item_id="item-2", reviewer="Codex", source_round=1,
        text="Keep the independent regression blocker.", status="blocking", source_status="blocking",
        authority=UNKNOWN_MACHINE_AUTHORITY, obligation_kind="unknown", lifecycle="repair_required",
        obligation_identity="unknown:item-2",
    )


def durable_record(items, *, role="coder", agent="Claude", round_number=2,
                   dispositions=(), new_items=(), body=None, **kwargs):
    if body is None:
        body = "Historical checkpoint.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    return _attach_round_metadata(body, PostedRoundMetadata(
        flow="pr", role=role, agent=agent, round_number=round_number, subject="abc123",
        prior_items=tuple(items), new_items=tuple(new_items), dispositions=tuple(dispositions), **kwargs,
    ))


def pr_payload(history=()):
    return {
        "number": 77, "state": "OPEN", "url": "https://github.com/OWNER/REPO/pull/77",
        "title": "Acknowledgement recovery", "headRefName": "fix/ack", "baseRefName": "main",
        "headRefOid": "abc123", "reviews": [],
        "comments": [
            {"author": {"login": "maintainer"}, "createdAt": "2026-05-18T10:00:00Z",
             "body": "Please preserve the URL.\n\n-- Human Reviewer"},
            *[{"author": {"login": "bot"}, "createdAt": f"2026-05-20T00:{i:02}:00Z", "body": body}
              for i, body in enumerate(history)],
        ],
    }


def ack_resume_history(*, promoted=False, coder_ack=False, conflict=False, unrelated=True, bad_id=False):
    from coding_review_agent_loop.github import _parse_pr_human_requirements
    from coding_review_agent_loop.protocol import ReviewItemDisposition
    from agent_loop_helpers import structured_coder_followup, structured_pr_review
    requirement_id = _parse_pr_human_requirements(pr_payload())[0].requirement_id
    alias = promoted_ack() if promoted else legacy_ack()
    other = (unrelated_item(),) if unrelated else ()
    raw_coder = structured_coder_followup(
        remaining_items=["item-2"] if unrelated else [],
        human_requirement_ids=[requirement_id] if coder_ack else [],
    ) if coder_ack else None
    original = durable_record((alias, *other))
    # Real historical items first persist through prior_items, with no new_items.
    later_alias = promoted_ack(notes=(UNTRUSTED_NOTE, "Codex: Acknowledged.")) if promoted else alias
    prior = (later_alias, dedicated_ack(), *other) if promoted else (alias, *other)
    coder = durable_record(prior, raw_structured_coder_response=raw_coder) if promoted or coder_ack else original
    dispositions = [ReviewItemDisposition(reviewer="Codex", item_id="item-13", disposition="resolved", note="Alias reviewed.")]
    if conflict:
        dispositions.append(ReviewItemDisposition(reviewer="Codex", item_id=ACK_ID, disposition="blocking", note="Keep ack gate."))
    if unrelated:
        dispositions.append(ReviewItemDisposition(reviewer="Codex", item_id="item-2", disposition="resolved", note="Ordinary disposition cannot clear a machine blocker."))
    if bad_id:
        dispositions.append(ReviewItemDisposition(reviewer="Codex", item_id="item-999", disposition="resolved"))
    review = durable_record(
        prior, role="reviewer", agent="Codex", dispositions=dispositions, state="blocking" if conflict else "approved",
        surfaced_reviewer_requirement_ids=(requirement_id,),
        body=structured_pr_review(state="blocking" if conflict else "approved", human_requirements_resolved=True),
    )
    return [original, coder, review]


