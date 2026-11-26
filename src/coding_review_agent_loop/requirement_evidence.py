"""Reviewer-owned signed-source routing; classification never grants acceptance."""
from dataclasses import dataclass

from .errors import AgentLoopError, MissingJudgementFieldError, render_human_decision
from .github import HumanReviewRequirement
from .protocol_markers import sanitize_untrusted_prose


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


ROLES = frozenset({"code", "human-only", "evidence-only", "mixed", "superseded"})

_HUMAN_ONLY_RULE = "human-only must name its exact emitted evidence request in request"
_NON_HUMAN_REQUEST_RULE = ("only human-only names a request; code, evidence-only, mixed and superseded "
                           "use request \"\"")
_ANSWER_RULES = {
    "mixed": ("mixed requires at least one answered_item_id; if no prior evidence item of yours "
              "is answered, use code or human-only"),
    "evidence-only": ("evidence-only requires at least one answered_item_id; if no prior evidence "
                      "item of yours is answered, use code or human-only"),
    "code": "code uses answered_item_ids []; use evidence-only or mixed to answer your prior evidence items",
    "human-only": ("human-only uses answered_item_ids []; use evidence-only or mixed to answer "
                   "your prior evidence items"),
    "superseded": "superseded uses answered_item_ids []; a replaced source answers nothing",
}
_LEFTOVER_RULE = ("mixed requires explicit leftover_instructions; if nothing is left for the coder, "
                  "use evidence-only")
_SUPERSEDED_LEFTOVER_RULE = ('superseded uses leftover_instructions ""; if leftover code work remains, '
                             "use mixed or code")
# The valid choices for a reviewer that emitted no exact_head_evidence_requests
# this turn (#2101): human-only cannot validate without a linked request. Every
# rejection carries it as its own re-ask value, kept within the re-ask
# renderer's per-value bound so it always reaches the reviewer intact.
VALID_CHOICES_WITHOUT_REQUESTS = (
    "no exact_head_evidence_requests emitted: use superseded if a newer signed source from the "
    "same author replaces it, evidence-only with your own prior evidence item IDs, or code")
# Rule variants that do not offer human-only when no request was emitted.
_NO_REQUEST_RULES = {
    "human-only": "this role needs a request linked to an emitted evidence request",
    "request": 'request must be ""',
    "mixed": "mixed requires at least one answered_item_id",
    "evidence-only": "evidence-only requires at least one answered_item_id",
    "human-only-answers": "answered_item_ids must be []",
}
_SUPERSEDED_PRECONDITION = ("superseded requires another surfaced signed source from the same author "
                            "with a strictly later created_at")


def _reask_values(rule, no_requests):
    """Re-ask values: the specific rule, plus the valid choices when no request was emitted."""
    return (rule, VALID_CHOICES_WITHOUT_REQUESTS) if no_requests else (rule,)


def _inconsistent(index, entry, rule, no_requests=False):
    """A role/field consistency violation, routed to the reviewer judgement re-ask."""
    rid = " ".join(str(entry["requirement_id"]).split())[:120]
    message = f"requirement_evidence[{index}] (requirement_id {rid}, role {entry['role']}): {rule}"
    if no_requests:
        message += "; " + VALID_CHOICES_WITHOUT_REQUESTS
    message += "."
    return MissingJudgementFieldError(message, field_path="requirement_evidence",
                                      allowed_values=_reask_values(rule, no_requests),
                                      observed_preview=message)


_ENTRY_KEYS = ("requirement_id", "role", "request", "answered_item_ids", "leftover_instructions")
_PREVIEW_CHARS = 60


def _value_preview(value: object) -> str:
    """Bounded, single-line, marker-free echo of a rejected value (#2128)."""
    text = value if isinstance(value, str) else type(value).__name__
    text = " ".join(sanitize_untrusted_prose(text).replace("`", "'").split())
    if len(text) > _PREVIEW_CHARS:
        text = text[:_PREVIEW_CHARS] + "\u2026"
    return f"`{text}`"


def _entry_keys_message(index: int, entry: dict) -> str:
    """Name the missing and unexpected keys of one requirement_evidence entry (#2128)."""
    keys = {key for key in entry if isinstance(key, str)}
    missing = [key for key in _ENTRY_KEYS if key not in keys]
    unexpected = sorted(_value_preview(key) for key in entry if key not in _ENTRY_KEYS)
    parts = [f"requirement_evidence[{index}] entries require exactly {', '.join(_ENTRY_KEYS)}"]
    if missing:
        parts.append(f"missing: {', '.join(missing)}")
    if unexpected:
        parts.append(f"unexpected: {', '.join(unexpected[:10])}")
    return "; ".join(parts)


def parse_associations(value, *, emitted_requests=None):
    """Parse reviewer classifications; role/field violations take the judgement re-ask.

    ``emitted_requests`` is this turn's exact_head_evidence_requests, or None when
    unknown (replay). With none emitted, re-ask rules never offer human-only.
    """
    no_requests = emitted_requests is not None and not emitted_requests
    if not isinstance(value, (list, tuple)):
        raise AgentLoopError("requirement_evidence must be an array")
    result = []
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise AgentLoopError(
                f"requirement_evidence[{index}] must be an object (got {type(entry).__name__}); "
                f"entries require {', '.join(_ENTRY_KEYS)}")
        if set(entry) != set(_ENTRY_KEYS):
            raise AgentLoopError(_entry_keys_message(index, entry))
        if not isinstance(entry["role"], str) or entry["role"] not in ROLES:
            raise AgentLoopError(
                f"Unknown requirement_evidence[{index}].role "
                f"(got {_value_preview(entry['role'])}); allowed values: {', '.join(sorted(ROLES))}")
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
        if not entry["requirement_id"]:
            raise AgentLoopError("Human-only requirements must link an evidence request")
        # Role/field consistency is the reviewer's judgement (#2035): it takes
        # the bounded judgement re-ask, never format repair or a fatal stop.
        if role == "superseded" and not entry["leftover_instructions"].strip():
            entry["leftover_instructions"] = ""
        if (role == "human-only") != bool(entry["request"].strip()):
            if no_requests:
                rule = _NO_REQUEST_RULES["human-only" if role == "human-only" else "request"]
            else:
                rule = _HUMAN_ONLY_RULE if role == "human-only" else _NON_HUMAN_REQUEST_RULE
            raise _inconsistent(index, entry, rule, no_requests)
        if (role in {"mixed", "evidence-only"}) != bool(answers):
            rule = _ANSWER_RULES[role]
            if no_requests and role in {"mixed", "evidence-only"}:
                rule = _NO_REQUEST_RULES[role]
            elif no_requests and role == "human-only":
                rule = _NO_REQUEST_RULES["human-only-answers"]
            raise _inconsistent(index, entry, rule, no_requests)
        if role == "superseded" and entry["leftover_instructions"]:
            # Never cleared by normalize_empty_text: a superseded source with
            # leftover code work would otherwise silently drop it (#2101).
            raise _inconsistent(index, entry, _SUPERSEDED_LEFTOVER_RULE, no_requests)
        if (role == "mixed") != bool(entry["leftover_instructions"].strip()):
            raise _inconsistent(index, entry, _LEFTOVER_RULE, no_requests)
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


def _created_at(source):
    """Parse a source's created_at as an aware timestamp; None fails closed."""
    from datetime import datetime, timezone
    raw = getattr(source, "created_at", None)
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _same_author(left, right):
    """Compare author_id when both carry it, otherwise the normalized login."""
    if left.author_id is not None and right.author_id is not None:
        return left.author_id == right.author_id
    login = lambda source: str(source.author or "").strip().lower()
    return bool(login(left)) and login(left) == login(right)


def is_superseded(source, requirements):
    """True when another surfaced source from the same author is strictly newer (#2101)."""
    when = _created_at(source)
    if when is None:
        return False
    for other in requirements:
        if other.requirement_id == source.requirement_id or not _same_author(source, other):
            continue
        newer = _created_at(other)
        if newer is not None and newer > when:
            return True
    return False


def validate_associations(parsed, requirements, items, reviewer):
    no_requests = not parsed.exact_head_evidence_requests

    def invalid(message):
        # Reuse the bounded semantic-judgement re-ask, never format repair.
        # With no emitted request every rejection also names the valid choices.
        if no_requests:
            message += "; " + VALID_CHOICES_WITHOUT_REQUESTS
        return MissingJudgementFieldError(message, field_path="requirement_evidence",
            allowed_values=_reask_values(
                "surfaced sources with linked emitted requests and requester-owned native answers", no_requests),
            observed_preview=message)

    live = {r.requirement_id: r for r in requirements}
    for entry in parsed.requirement_evidence:
        if entry["requirement_id"] not in live:
            raise invalid("Requirement/evidence association names an unsurfaced source")
        if entry["role"] == "human-only" and entry["request"] not in parsed.exact_head_evidence_requests:
            raise invalid("Human-only association must name this requester's emitted evidence request")
        if entry["role"] == "superseded" and not is_superseded(live[entry["requirement_id"]], requirements):
            message = (f"Superseded association for {entry['requirement_id']} is invalid: "
                       f"{_SUPERSEDED_PRECONDITION}")
            if not no_requests:
                message += "; use code, evidence-only, or human-only with your linked emitted request"
            raise invalid(message)
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


def _agreed_classifications(requirements, reviews, items):
    """Map each source to its single agreed role and entries; invalid or split sources are absent."""
    from types import SimpleNamespace
    by_source = {}
    invalid_sources = set()
    for reviewer, parsed in reviews.items():
        for entry in parsed.requirement_evidence:
            try:
                validate_associations(SimpleNamespace(requirement_evidence=(entry,),
                    exact_head_evidence_requests=parsed.exact_head_evidence_requests), requirements, items, reviewer)
            except AgentLoopError:
                invalid_sources.add(entry["requirement_id"])
                continue
            by_source.setdefault(entry["requirement_id"], []).append((reviewer, entry))
    agreed = {}
    for source_id, entries in by_source.items():
        roles = {e["role"] for _, e in entries}
        if source_id not in invalid_sources and len(roles) == 1:
            agreed[source_id] = (next(iter(roles)), entries)
    return agreed


def project(requirements, reviews, items):
    """Keep original signed identity, and fail closed on disagreeing classifiers.

    Agreement is decided by role; leftover wording differences merge instead of
    conflicting. Called only on actual validated reviewer results or admitted
    requester records. Absence of a classification never consumes a signed source.
    An agreed superseded source leaves the projection like evidence-only (#2101).
    """
    agreed = _agreed_classifications(requirements, reviews, items)
    result = []
    for source in requirements:
        if source.requirement_id not in agreed:
            result.append(source)
            continue
        role, entries = agreed[source.requirement_id]
        if role in {"human-only", "evidence-only", "superseded"}:
            continue
        if role == "mixed":
            from dataclasses import fields
            values = {f.name: getattr(source, f.name) for f in fields(HumanReviewRequirement)}
            values["body"] = _merged_mixed_body(
                source, [(reviewer, e["leftover_instructions"]) for reviewer, e in entries])
            source = ActionableRequirement(**values, original_id=source.requirement_id)
        result.append(source)
    return tuple(result)


def superseded_ids(requirements, reviews, items):
    """Normalized labels of sources whose agreed, valid role is superseded (a subset of reviewer_owned_ids)."""
    agreed = _agreed_classifications(requirements, reviews, items)
    result = []
    for source in requirements or ():
        label = str(source.requirement_id).strip().lower()
        if agreed.get(source.requirement_id, ("",))[0] == "superseded" and label and label not in result:
            result.append(label)
    return tuple(result)


def _role_conflict_message(conflicts, previous_entries, reviewer_name):
    """Name every reviewer's role per conflicting source and the clearing operator input."""
    details = []
    for entry in conflicts:
        rid = entry["requirement_id"]
        roles = [(name, p["role"]) for name, p in previous_entries if p["requirement_id"] == rid]
        roles.append((reviewer_name, entry["role"]))
        details.append(f"{rid}: " + ", ".join(f"{name}: {role}" for name, role in sorted(roles)))
    what = ("Signed-source classification remained conflicting; retained signed sources and pending evidence "
            "require reviewer agreement or a human decision before coder dispatch. Reviewer roles per source: "
            + "; ".join(details) + ". Post a new follow-up signed comment (do not edit the existing "
            "signed comment) that separates exact-head evidence answers from code requirements; the next "
            "resume opens a fresh review round against the added source. Or have reviewers converge on "
            "one role on rerun; no coder was dispatched.")
    return render_human_decision(
        what=what,
        clear=("Post a new follow-up signed comment (do not edit the existing one) that separates "
               "exact-head evidence answers from code requirements, then rerun the same command."),
        safe_to_automate=False,
        automation_note="the clarifying signed comment must come from a human.")


# How to route a signed source a later signed source replaces (#2101).
SUPERSEDED_GUIDANCE = (
    "Use superseded (request \"\", answered_item_ids [], leftover_instructions \"\") for a signed "
    "source that a later signed source from the same author replaces, such as an older exact-head "
    "approval or a note limiting it; it creates no coder work and no evidence, and it is valid only "
    "when a strictly newer signed source from the same author is surfaced. Use human-only only when "
    "you emit its linked exact_head_evidence_request in this same response.")


# The signed-source classification continuation's validated-turn options. A
# consistency violation is re-asked once; an exhausted re-ask takes the typed
# judgement-failure path (#2035). No repair kind is set, so the turn keeps its
# backend output channel and never enters format repair.
CLASSIFICATION_INVOKE_OPTIONS = {
    "marker_description": "structured PR review classification",
    "role": "reviewer",
    "operation_description": "Signed-source classification",
    "reviewer_judgement_turn": True,
}


def classify_review(parsed, text, *, requirements, items, reviewer_name, invoke, previous=None):
    """Settle unacknowledged source routing with one bounded semantic continuation.

    A published review is never restored for a classification-only
    continuation: resume rejects it and its round closes as incomplete (#2064).
    """
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
        sources = [{"requirement_id": r.requirement_id, "author": r.author, "created_at": r.created_at,
                    "body": r.body} for r in requirements]
        prior = [{"requirement_id": p["requirement_id"], "reviewer": name, "role": p["role"]}
                 for name, p in previous_entries]
        prompt = ("Continue the PR review with semantic signed-source classification. Return a complete "
            "structured pr_review and its normal footer. Preserve code findings in blocking_items. Classify every signed "
            "source in requirement_evidence using exactly requirement_id, role (code, human-only, "
            "evidence-only, mixed, superseded), request (a string: your exact emitted request for human-only, otherwise \"\"), "
            "answered_item_ids (a list: your own prior evidence items only, otherwise []), leftover_instructions "
            "(a string: explicit leftover code instructions for mixed, otherwise \"\"). Do not infer acceptance "
            "from a head mention. Only requester dispositions with evidence_requirement_ids attest adequacy. "
            "This is a real semantic reviewer turn; do not invent instructions. Agreement is decided by role; "
            "leftover wording need not match other reviewers. " + SUPERSEDED_GUIDANCE + "\nSigned sources: "
            + json.dumps(sources) + "\nPrior classifications to reconcile: " + json.dumps(prior) + "\nPrior items: " + repr(items) + "\nPrevious review:\n" + text)
        response = invoke(prompt)
        parsed, text = response.marker_value, response.text
        if {r.requirement_id for r in requirements} - {e["requirement_id"] for e in parsed.requirement_evidence}:
            raise HumanDecisionRequiredError.decision(
                what="Signed-source classification remained absent; no coder was dispatched for human evidence.",
                clear=("Post a new follow-up signed comment that separates exact-head evidence answers "
                       "from code requirements, then rerun the same command."),
                safe_to_automate=False,
                automation_note="the clarifying signed comment must come from a human.")
        conflicts = conflicts_with_previous(parsed)
        if conflicts:
            raise HumanDecisionRequiredError(_role_conflict_message(conflicts, previous_entries, reviewer_name))
    # Live turns validate associations before publication. Legacy replay may carry
    # invalid entries; project retains their full source rather than aborting.
    return parsed, text


def _historical_inputs(requirements, records):
    """Latest admitted reviewer classifications and their prior items, keyed for project()."""
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
    return latest, tuple(targets.values())


def historical_projection(requirements, records):
    """Routing is replayed only from admitted requester records, never prose.

    Routing does not fulfill evidence, so a push retains the human-only role
    while the native evidence gate independently reopens exact-head acceptance.
    """
    return project(requirements, *_historical_inputs(requirements, records))


def historical_superseded_ids(requirements, records):
    """Replay twin of superseded_ids over admitted requester records."""
    return superseded_ids(requirements, *_historical_inputs(requirements, records))


def reviewer_owned_ids(requirements, projected):
    """Return normalized labels of live signed sources the projection routed away from the coder.

    Option A (#2020): evidence-only and human-only sources are reviewer-owned
    exact-head evidence and never coder-dispositioned; agreed superseded
    sources (#2101) are reviewer-owned and inert. Mixed sources keep their
    original identity in the projection and therefore stay coder-owned, as do
    sources with conflicting or missing classifications.

    The projection kept here is also the PR reviewer acknowledgement scope
    (#2083): the PR loop's acknowledgement creation and clearing checks key on
    it, and the freeze-mode reviewer prompt lists these complementary IDs as
    outside the approval marker's scope.
    """
    kept = {str(r.requirement_id).strip().lower() for r in projected or ()}
    result = []
    for source in requirements or ():
        label = str(source.requirement_id).strip().lower()
        if label and label not in kept and label not in result:
            result.append(label)
    return tuple(result)
