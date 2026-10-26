"""Discussion response records, constants and aliases."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StructuredDiscussReview:
    schema_version: int
    kind: str
    outcome: str
    rationale: str
    split_proposals: tuple[str, ...]
    rebuttal: str | None = None
    analyzer_framing: str | None = None
    framing_note: str | None = None


@dataclass(frozen=True)
class DiscussUnresolvedItem:
    """A classified unresolved concern from an answer-mode discuss vote.

    This is intentionally distinct from ``UnresolvedReviewItem``, which is the
    PR/plan-review ledger type and has unrelated lifecycle semantics.
    """

    status: str
    text: str


DISCUSS_UNRESOLVED_ITEM_STATUS_VALUES = frozenset(
    {"blocker", "human-decision", "follow-up"}
)


@dataclass(frozen=True)
class StructuredDiscussAnswer:
    schema_version: int
    kind: str
    position: str
    rationale: str
    confidence: str
    unresolved_items: tuple[DiscussUnresolvedItem, ...]
    answer: str | None = None
    rebuttal: str | None = None
    analyzer_framing: str | None = None
    framing_note: str | None = None


DISCUSS_EVIDENCE_STATUS_VALUES = frozenset({"verified", "reported-but-unverified", "missing"})
DISCUSS_EVIDENCE_UPDATE_ACTIONS = frozenset({"retract", "supersede"})
DISCUSS_VERIFICATION_BASES = frozenset({"external-source-inspected", "checkout-inspected"})


@dataclass(frozen=True)
class DiscussEvidenceClaim:
    """A debater-authored observation.  IDs are assigned by the orchestrator."""

    fact: str
    status: str
    source: str | None = None
    verification_basis: str | None = None


@dataclass(frozen=True)
class DiscussEvidenceUpdate:
    action: str
    target_observation_id: str
    reason: str
    replacement_claim_index: int | None = None


@dataclass(frozen=True)
class ParsedDiscussAnswer:
    position: str
    rationale: str
    confidence: str
    unresolved_items: tuple[DiscussUnresolvedItem, ...]
    reviewer: str
    answer: str | None = None
    rebuttal: str | None = None
    analyzer_framing: str | None = None
    framing_note: str | None = None
    research_status: str | None = None
    sourced_facts: tuple[DiscussSourcedFact, ...] = ()
    research_target: str | None = None
    research_questions: tuple[str, ...] = ()
    evidence_claims: tuple[DiscussEvidenceClaim, ...] = ()
    evidence_updates: tuple[DiscussEvidenceUpdate, ...] = ()


@dataclass(frozen=True)
class DiscussSemanticEvidence:
    reviewer: str
    supports: str


@dataclass(frozen=True)
class ParsedDiscussSemanticComparison:
    classification: str
    shared_recommendation: str
    remaining_decisions: tuple[str, ...]
    evidence: tuple[DiscussSemanticEvidence, ...]


@dataclass(frozen=True)
class ParsedDiscussAnswerConfirmation:
    reviewer: str
    decision: str
    rationale: str
    answer: str | None = None


@dataclass(frozen=True)
class ParsedDiscussEvidenceReconciliation:
    groups: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class ParsedFailedDiscussResponse:
    reviewer: str
    category: str
    result_mode: str = "triage"


@dataclass(frozen=True)
class DiscussSourcedFact:
    fact: str
    source: str


@dataclass(frozen=True)
class ParsedDiscussReview:
    outcome: str
    rationale: str
    split_proposals: tuple[str, ...]
    reviewer: str
    rebuttal: str | None = None
    analyzer_framing: str | None = None
    framing_note: str | None = None
    # Research policy fields (#477). None means the response carried no
    # `research` object (legacy transcripts and `none`-mode responses).
    research_status: str | None = None
    sourced_facts: tuple[DiscussSourcedFact, ...] = ()
    research_target: str | None = None
    research_questions: tuple[str, ...] = ()
    evidence_claims: tuple[DiscussEvidenceClaim, ...] = ()
    evidence_updates: tuple[DiscussEvidenceUpdate, ...] = ()


ParsedDiscussResponse = ParsedDiscussReview | ParsedDiscussAnswer | ParsedFailedDiscussResponse


DISCUSS_OUTCOME_VALUES = frozenset({"implement", "do-not-implement", "needs-human", "split"})
DISCUSS_ANALYZER_FRAMING_VALUES = frozenset({"accurate", "misframed"})
DISCUSS_RESEARCH_STATUS_VALUES = frozenset({"sourced", "not-needed", "unavailable", "inconclusive"})
DISCUSS_RESEARCH_TARGET_VALUES = frozenset(
    {
        "example-validation",
        "solution-design",
        "cost-latency",
        "implementation-feasibility",
        "policy/legal/current-facts",
    }
)

# Internal-only outcome for debaters that failed or timed out in a partial
# round (#475). Deliberately NOT in DISCUSS_OUTCOME_VALUES: real agent
# responses cannot claim it. Placeholders are built only by the orchestrator
# and by resume; their presence in a round blocks false-positive consensus.
DISCUSS_FAILED_OUTCOME = "failed"


@dataclass(frozen=True)
class DiscussAgendaDisagreement:
    topic: str
    positions: tuple[tuple[str, str], ...]
    question_for_next_round: str


DISCUSS_SYNTHESIS_MAX_ENTRIES = 8
DISCUSS_SYNTHESIS_MAX_NEXT_ROUND_FOCUS = 5
DISCUSS_SYNTHESIS_MAX_TEXT_BYTES = 512
DISCUSS_SYNTHESIS_MAX_CANONICAL_BYTES = 16_000
DISCUSS_SYNTHESIS_CHANGE_KINDS = frozenset(
    {"introduced", "resolved", "reopened", "retracted", "refined"}
)
DISCUSS_SYNTHESIS_CLASSIFICATIONS = frozenset(
    {"consensus", "near_consensus", "material_deadlock"}
)


@dataclass(frozen=True)
class DiscussSynthesisResponseReference:
    """A response that independently supports a synthesized statement."""

    reviewer: str
    round: int


@dataclass(frozen=True)
class DiscussSynthesisConsensus:
    text: str
    references: tuple[DiscussSynthesisResponseReference, ...]


@dataclass(frozen=True)
class DiscussSynthesisPosition:
    reviewers: tuple[str, ...]
    position: str


@dataclass(frozen=True)
class DiscussSynthesisDisagreement:
    topic: str
    positions: tuple[DiscussSynthesisPosition, ...]
    decision_needed: str


@dataclass(frozen=True)
class DiscussSynthesisChange:
    kind: str
    topic: str
    text: str
    references: tuple[DiscussSynthesisResponseReference, ...]


@dataclass(frozen=True)
class ParsedDiscussRoundSynthesis:
    """Validated cumulative state for one completed answer-mode round."""

    consensus: tuple[DiscussSynthesisConsensus, ...]
    disagreements: tuple[DiscussSynthesisDisagreement, ...]
    changes: tuple[DiscussSynthesisChange, ...]
    missing_facts: tuple[str, ...]
    next_round_focus: tuple[str, ...]
    responding_reviewers: tuple[str, ...]


@dataclass(frozen=True)
class ParsedDiscussFinalSynthesis:
    """Validated executive conclusion for an answer-mode final result."""

    classification: str
    agreed_conclusions: tuple[DiscussSynthesisConsensus, ...]
    remaining_disagreements: tuple[DiscussSynthesisDisagreement, ...]
    next_action: str


# Short aliases keep the public protocol vocabulary easy to discover while
# retaining the explicit response-reference name in serialized models.
DiscussSynthesisReference = DiscussSynthesisResponseReference
DiscussRoundSynthesis = ParsedDiscussRoundSynthesis
DiscussFinalSynthesis = ParsedDiscussFinalSynthesis


@dataclass(frozen=True)
class ParsedDiscussAgenda:
    consensus: tuple[str, ...]
    disagreements: tuple[DiscussAgendaDisagreement, ...]
    missing_facts: tuple[str, ...]
    # Shared research brief (#477). Emitted by the analyzer when a research
    # policy is active; forwarded to the next round's debaters so parallel
    # turns do not duplicate work.
    research_required: bool = False
    research_questions: tuple[str, ...] = ()
    research_question_targets: tuple[str, ...] = ()
    # Optional answer-mode presentation state. Legacy agendas deliberately
    # remain valid and have no synthesis value.
    round_synthesis: ParsedDiscussRoundSynthesis | None = None
