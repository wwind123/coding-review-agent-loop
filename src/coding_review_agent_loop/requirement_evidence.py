"""Reviewer-owned signed-source routing; classification never grants acceptance."""
from dataclasses import dataclass

from .errors import AgentLoopError, MissingJudgementFieldError
from .github import HumanReviewRequirement


def parse_associations(value):
    if not isinstance(value, (list, tuple)):
        raise AgentLoopError("requirement_evidence must be an array")
    result = []
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {
            "requirement_id", "role", "request", "answered_item_ids", "leftover_instructions"
        }:
            raise AgentLoopError("requirement_evidence entries require source, role, request, answers and leftover instructions")
        if not isinstance(entry["role"], str) or entry["role"] not in {"code", "human-only", "evidence-only", "mixed"}:
            raise AgentLoopError("Unknown requirement_evidence role")
        if any(not isinstance(entry[k], str) for k in ("requirement_id", "request", "leftover_instructions")):
            raise AgentLoopError("requirement_evidence source, request and instructions must be strings")
        answers = entry["answered_item_ids"]
        if not isinstance(answers, list) or any(not isinstance(i, str) or not i for i in answers) or len(set(answers)) != len(answers):
            raise AgentLoopError("requirement_evidence answers must be distinct item IDs")
        role = entry["role"]
        if not entry["requirement_id"] or (role == "human-only") != bool(entry["request"].strip()):
            raise AgentLoopError("Human-only requirements must link an evidence request")
        if (role in {"mixed", "evidence-only"}) != bool(answers):
            raise AgentLoopError("Evidence responses must identify answered items")
        if (role == "mixed") != bool(entry["leftover_instructions"].strip()):
            raise AgentLoopError("Only mixed sources require explicit leftover instructions")
        if any(e["requirement_id"] == entry["requirement_id"] for e in result):
            raise AgentLoopError("Repeated requirement_evidence source")
        result.append(dict(entry))
    return tuple(result)


def decode_associations(payload):
    """Adapt semantic shape errors to the round transport's invalid-record path."""
    try:
        return parse_associations(payload.get("requirement_evidence", []))
    except AgentLoopError as exc:
        raise ValueError(str(exc)) from exc


def validate_associations(parsed, requirements, items, reviewer):
    def invalid(message):
        # Reuse the bounded semantic-judgement re-ask, never format repair.
        return MissingJudgementFieldError(message, field_path="requirement_evidence",
            allowed_values=("surfaced sources with linked emitted requests and requester-owned native answers",),
            observed_preview=message)

    live = {r.requirement_id for r in requirements}
    for entry in parsed.requirement_evidence:
        if entry["requirement_id"] not in live:
            raise invalid("Requirement/evidence association names an unsurfaced source")
        if entry["role"] == "human-only" and entry["request"] not in parsed.exact_head_evidence_requests:
            raise invalid("Human-only association must name this requester's emitted evidence request")
        for item_id in entry["answered_item_ids"]:
            targets = [i for i in items if item_id in (i.item_id, *i.alias_item_ids)]
            from .native_evidence import native
            if len(targets) != 1 or not native(targets[0]) or targets[0].reviewer != reviewer:
                raise invalid("Evidence response must name an existing item owned by this requester")
    return parsed


@dataclass(frozen=True)
class ActionableRequirement(HumanReviewRequirement):
    original_id: str = ""

    @property
    def requirement_id(self):
        return self.original_id


def project(requirements, reviews, items):
    """Keep original signed identity, and fail closed on disagreeing classifiers.

    Called only on actual validated reviewer results or admitted requester records.
    Absence of a classification never consumes a signed source.
    """
    by_source = {}
    invalid_sources = set()
    for reviewer, parsed in reviews.items():
        from types import SimpleNamespace
        for entry in parsed.requirement_evidence:
            try:
                validate_associations(SimpleNamespace(requirement_evidence=(entry,),
                    exact_head_evidence_requests=parsed.exact_head_evidence_requests), requirements, items, reviewer)
            except AgentLoopError:
                invalid_sources.add(entry["requirement_id"])
                continue
            by_source.setdefault(entry["requirement_id"], []).append(entry)
    result = []
    for source in requirements:
        entries = by_source.get(source.requirement_id, [])
        if not entries or source.requirement_id in invalid_sources:
            result.append(source)
            continue
        signatures = {(e["role"], e["leftover_instructions"]) for e in entries}
        if len(signatures) != 1:
            result.append(source)
            continue
        entry = entries[0]
        if entry["role"] in {"human-only", "evidence-only"}:
            continue
        if entry["role"] == "mixed":
            from dataclasses import fields
            values = {f.name: getattr(source, f.name) for f in fields(HumanReviewRequirement)}
            values["body"] = entry["leftover_instructions"]
            source = ActionableRequirement(**values, original_id=source.requirement_id)
        result.append(source)
    return tuple(result)


def classify_review(parsed, text, *, requirements, items, reviewer_name, invoke, previous=None):
    """One bounded semantic continuation when evidence lacks source classification."""
    from .protocol import human_requirements_resolved
    from .native_evidence import open_native
    classified = {e["requirement_id"] for e in parsed.requirement_evidence}
    missing = {r.requirement_id for r in requirements} - classified
    previous_entries = [e for review in (previous or {}).values() for e in review.requirement_evidence]
    conflicts = [e for e in parsed.requirement_evidence if any(
        p["requirement_id"] == e["requirement_id"] and
        (p["role"], p["leftover_instructions"]) != (e["role"], e["leftover_instructions"])
        for p in previous_entries)]
    evidence_context = parsed.exact_head_evidence_requests or any(open_native(i) for i in items)
    if conflicts or (requirements and missing and evidence_context
            and (parsed.requirement_evidence or not human_requirements_resolved(text))):
        import json
        from .errors import HumanDecisionRequiredError
        sources = [{"requirement_id": r.requirement_id, "body": r.body} for r in requirements]
        prompt = ("Continue the PR review with semantic signed-source classification. Return a complete "
            "structured pr_review and its normal footer. Preserve code findings. Classify every signed "
            "source in requirement_evidence using exactly requirement_id, role (code, human-only, "
            "evidence-only, mixed), request (your exact emitted request for human-only, otherwise empty), "
            "answered_item_ids (your own prior evidence items only, otherwise []), leftover_instructions "
            "(explicit leftover code instructions for mixed, otherwise empty). Do not infer acceptance "
            "from a head mention. Only requester dispositions with evidence_requirement_ids attest adequacy. "
            "This is a real semantic reviewer turn; do not invent instructions.\nSigned sources: "
            + json.dumps(sources) + "\nPrior classifications to reconcile: " + repr(previous_entries) + "\nPrior items: " + repr(items) + "\nPrevious review:\n" + text)
        response = invoke(prompt)
        parsed, text = response.marker_value, response.text
        if {r.requirement_id for r in requirements} - {e["requirement_id"] for e in parsed.requirement_evidence}:
            raise HumanDecisionRequiredError("Signed-source classification remained absent; no coder was dispatched for human evidence.")
    # Live turns validate associations before publication. Legacy replay may carry
    # invalid entries; project retains their full source rather than aborting.
    return parsed, text


def historical_projection(requirements, records):
    """Routing is replayed only from admitted requester records, never prose.

    Routing does not fulfill evidence, so a push retains the human-only role
    while the native evidence gate independently reopens exact-head acceptance.
    """
    from types import SimpleNamespace
    live = {r.requirement_id for r in requirements}
    latest = {}
    targets = {}
    for record in records:
        m = record.metadata
        if m.role != "reviewer":
            continue
        entries = tuple(e for e in m.requirement_evidence if e["requirement_id"] in live)
        latest[m.agent] = SimpleNamespace(requirement_evidence=entries,
            exact_head_evidence_requests=m.evidence_requests)
        for item in m.prior_items:
            targets[item.item_id] = item
    return project(requirements, latest, tuple(targets.values()))
