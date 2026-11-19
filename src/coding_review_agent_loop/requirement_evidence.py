"""Reviewer-owned signed-source routing; classification never grants acceptance."""
from dataclasses import dataclass

from .errors import AgentLoopError, MissingJudgementFieldError
from .github import HumanReviewRequirement


def normalize_empty_text(entry):
    """Copy an association, canonicalizing empty text and redundant non-mixed leftovers."""
    result = dict(entry)
    for field in ("request", "leftover_instructions"):
        value = result[field]
        if value is None or (type(value) is list and not value):
            result[field] = ""
    if (result.get("role") in ("code", "human-only", "evidence-only")
            and isinstance(result["leftover_instructions"], str)):
        result["leftover_instructions"] = ""
    return result


def parse_associations(value):
    if not isinstance(value, (list, tuple)):
        raise AgentLoopError("requirement_evidence must be an array")
    result = []
    for index, entry in enumerate(value):
        if not isinstance(entry, dict) or set(entry) != {
            "requirement_id", "role", "request", "answered_item_ids", "leftover_instructions"
        }:
            raise AgentLoopError("requirement_evidence entries require requirement_id, role, request, answered_item_ids and leftover_instructions")
        if not isinstance(entry["role"], str) or entry["role"] not in {"code", "human-only", "evidence-only", "mixed"}:
            raise AgentLoopError("Unknown requirement_evidence role")
        entry = normalize_empty_text(entry)
        for field in ("requirement_id", "request", "leftover_instructions"):
            if not isinstance(entry[field], str):
                raise AgentLoopError(
                    f"requirement_evidence[{index}].{field} must be a string "
                    f"(got {type(entry[field]).__name__})")
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


def _merged_mixed_body(source, attributed):
    """Verbatim signed source plus every reviewer's leftover, attributed and name-ordered.

    Only signed-source text, reviewer-authored leftovers and fixed labels appear,
    so the coder never receives orchestrator-invented instructions. Sorting by
    reviewer name keeps live projection and historical replay byte-identical.
    """
    lines = [f"- {reviewer}: {leftover.strip()}" for reviewer, leftover in sorted(attributed)]
    return ("Signed source (verbatim):\n" + source.body + "\n\n"
            "Reviewer leftover instructions (classification only; the signed source is authoritative):\n"
            + "\n".join(lines))


def project(requirements, reviews, items):
    """Keep original signed identity, and fail closed on disagreeing classifiers.

    Agreement is decided by role; leftover wording differences merge instead of
    conflicting. Called only on actual validated reviewer results or admitted
    requester records. Absence of a classification never consumes a signed source.
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
            by_source.setdefault(entry["requirement_id"], []).append((reviewer, entry))
    result = []
    for source in requirements:
        entries = by_source.get(source.requirement_id, [])
        if not entries or source.requirement_id in invalid_sources:
            result.append(source)
            continue
        roles = {e["role"] for _, e in entries}
        if len(roles) != 1:
            result.append(source)
            continue
        (role,) = roles
        if role in {"human-only", "evidence-only"}:
            continue
        if role == "mixed":
            from dataclasses import fields
            values = {f.name: getattr(source, f.name) for f in fields(HumanReviewRequirement)}
            values["body"] = _merged_mixed_body(
                source, [(reviewer, e["leftover_instructions"]) for reviewer, e in entries])
            source = ActionableRequirement(**values, original_id=source.requirement_id)
        result.append(source)
    return tuple(result)


def _role_conflict_message(conflicts, previous_entries, reviewer_name):
    """Name every reviewer's role per conflicting source and the clearing operator input."""
    details = []
    for entry in conflicts:
        rid = entry["requirement_id"]
        roles = [(name, p["role"]) for name, p in previous_entries if p["requirement_id"] == rid]
        roles.append((reviewer_name, entry["role"]))
        details.append(f"{rid}: " + ", ".join(f"{name}: {role}" for name, role in sorted(roles)))
    return ("Signed-source classification remained conflicting; retained signed sources and pending evidence "
            "require reviewer agreement or a human decision before coder dispatch. Reviewer roles per source: "
            + "; ".join(details) + ". Post a follow-up signed comment that separates exact-head evidence "
            "answers from code requirements, or have reviewers converge on one role on rerun; "
            "no coder was dispatched.")


def classify_review(parsed, text, *, requirements, items, reviewer_name, invoke, previous=None):
    """Settle unacknowledged source routing with one bounded semantic continuation."""
    from .protocol import human_requirements_resolved
    classified = {e["requirement_id"] for e in parsed.requirement_evidence}
    missing = {r.requirement_id for r in requirements} - classified
    previous_entries = [(name, e) for name, review in (previous or {}).items()
                        for e in review.requirement_evidence]
    def conflicts_with_previous(review):
        # Agreement is decided by role; leftover wording alone never conflicts.
        return [e for e in review.requirement_evidence if any(
            p["requirement_id"] == e["requirement_id"] and p["role"] != e["role"]
            for _, p in previous_entries)]

    if conflicts_with_previous(parsed) or (requirements and missing
            and (parsed.requirement_evidence or not human_requirements_resolved(text))):
        import json
        from .errors import HumanDecisionRequiredError
        sources = [{"requirement_id": r.requirement_id, "body": r.body} for r in requirements]
        prior = [{"requirement_id": p["requirement_id"], "reviewer": name, "role": p["role"]}
                 for name, p in previous_entries]
        prompt = ("Continue the PR review with semantic signed-source classification. Return a complete "
            "structured pr_review and its normal footer. Preserve code findings in blocking_items. Classify every signed "
            "source in requirement_evidence using exactly requirement_id, role (code, human-only, "
            "evidence-only, mixed), request (a string: your exact emitted request for human-only, otherwise \"\"), "
            "answered_item_ids (a list: your own prior evidence items only, otherwise []), leftover_instructions "
            "(a string: explicit leftover code instructions for mixed, otherwise \"\"). Do not infer acceptance "
            "from a head mention. Only requester dispositions with evidence_requirement_ids attest adequacy. "
            "This is a real semantic reviewer turn; do not invent instructions. Agreement is decided by role; "
            "leftover wording need not match other reviewers.\nSigned sources: "
            + json.dumps(sources) + "\nPrior classifications to reconcile: " + json.dumps(prior) + "\nPrior items: " + repr(items) + "\nPrevious review:\n" + text)
        response = invoke(prompt)
        parsed, text = response.marker_value, response.text
        if {r.requirement_id for r in requirements} - {e["requirement_id"] for e in parsed.requirement_evidence}:
            raise HumanDecisionRequiredError("Signed-source classification remained absent; no coder was dispatched for human evidence.")
        conflicts = conflicts_with_previous(parsed)
        if conflicts:
            raise HumanDecisionRequiredError(_role_conflict_message(conflicts, previous_entries, reviewer_name))
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


def reviewer_owned_ids(requirements, projected):
    """Return normalized labels of live signed sources the projection routed away from the coder.

    Option A (#2020): evidence-only and human-only sources are reviewer-owned
    exact-head evidence and never coder-dispositioned. Mixed sources keep their
    original identity in the projection and therefore stay coder-owned, as do
    sources with conflicting or missing classifications.
    """
    kept = {str(r.requirement_id).strip().lower() for r in projected or ()}
    result = []
    for source in requirements or ():
        label = str(source.requirement_id).strip().lower()
        if label and label not in kept and label not in result:
            result.append(label)
    return tuple(result)
