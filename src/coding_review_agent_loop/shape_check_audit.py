"""The protocol.py shape-check audit registry (#924 stage 3, #927).

Every function, method and nested def in ``protocol.py`` that can still
reach an ``AgentLoopError``-family raise after the stage-3 conversions is
classified here as ``fatal``, ``mixed`` or ``delegated``, and every raise
site and raise-reaching call site in it carries an inline
``# shape-check: <value>`` annotation:

* ``fatal:<clause>`` -- a propagating site in a ``fatal`` or ``mixed`` unit,
  naming its reserved clause from ``FATAL_CLAUSES``;
* ``handled`` -- a site inside a ``try`` whose handlers cover every class the
  site can raise without re-raising (the error becomes a degradation);
* ``delegated`` -- a propagating site inside a generic helper whose
  consequence is classified at each caller's annotated site.

A ``mixed`` unit keeps at least one fatal site and also degrades: it has a
handled site, calls a ``DEGRADING_PARSERS`` unit, calls another ``mixed``
unit, or builds a ``ParseDegradation`` record in its own body.  Record
construction reaches ``sanitize_historical_text``, so a unit that builds
records is never a degrading parser.

``tests/test_shape_check_audit.py`` recomputes the transitive inventory from
source and fails on any disagreement.  protocol.py never imports this
module; it has no runtime effect.

Caveat: calls routed through attributes of other modules' objects are
outside the inventory.  protocol.py imports only bare names from package
modules, so today the caveat covers nothing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShapeCheckClassification:
    disposition: str  # "fatal", "mixed" or "delegated"
    clauses: frozenset[str]
    justification: str


FATAL_CLAUSES = frozenset({
    # JSON extraction, footer and signature consumption, and the top-level
    # object or type of a response.
    "unparseable-envelope",
    # kind, schema_version and contract-version checks.
    "kind-or-version-mismatch",
    # Marker grammar and marker neutralization, identity digests, any
    # agent-supplied field naming orchestrator-owned authority, and canonical
    # risk_test_matrix_evidence rows including their evidence_citations.
    "authentication-or-forgery",
    # Hard caps and count bounds that keep an accepted payload bounded.
    "payload-bound",
    # Selectors resolving to real broker handles whose recorded outcome
    # contradicts the claim (#990).
    "authority-decision",
    # Decoders of state the orchestrator wrote, where a violation is
    # corruption rather than an agent defect.
    "orchestrator-authored",
    # Elements whose removal would weaken an obligation, a finding, a
    # disposition, approved topology or a discussion answer.
    "no-conservative-reading",
})

# Non-raising defect classifiers with no propagating site at all, including
# no record construction.
DEGRADING_PARSERS: tuple[str, ...] = ("_test_observation_item_defect",)

# Bare-name functions imported into protocol.py from package modules, with
# the AgentLoopError-family classes each can raise.  The guard verifies each
# entry against the source module.
IMPORTED_RAISERS: dict[str, frozenset[str]] = {
    # Raises when neutralization would still leave a reserved marker
    # (protocol_markers.py): the marker-neutralization defence.
    "sanitize_historical_text": frozenset({"AgentLoopError"}),
    "normalize_fix_scope": frozenset({"AgentLoopError"}),
    "parse_managed_test_command": frozenset({"TestRuntimeConfigurationError"}),
}
IMPORTED_NON_RAISERS: frozenset[str] = frozenset()


SHAPE_CHECK_AUDIT: dict[str, ShapeCheckClassification] = {
    "ExecutionDisposition.to_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "ExecutionStrategyRecommendation.identity": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "ExecutionStrategyRecommendation.to_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "ParseDegradation.build": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Record construction marker-neutralizes every field; a reserved marker "
            "surviving neutralization must reject, never be swallowed. Fatal sites: "
            "authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "ParseDegradation.to_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "PlanRevisionPatch.to_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "PlanRevisionPatchOperation._raw_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "PlanRevisionPatchOperation.to_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "PostAuthClaimDiagnostic.to_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "RiskTestMatrix.to_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "RiskTestMatrixChange.to_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "RiskTestMatrixEvidence.to_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "RiskTestMatrixEvidenceRow.to_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "RiskTestMatrixMetadata.to_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "RiskTestMatrixRow.to_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "SemanticRiskCoverageClaim.to_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "SemanticRiskCoverageClaims.to_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "_bounded_discuss_synthesis_list": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading", "payload-bound"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim; payload-bound: a hard cap or count bound that keeps an "
            "accepted payload bounded."
        ),
    ),
    "_bounded_discuss_synthesis_text": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading", "payload-bound"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim; payload-bound: a hard cap or count bound that keeps an "
            "accepted payload bounded."
        ),
    ),
    "_bounded_single_line": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Marker neutralization of a record field through sanitize_historical_text. "
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "_bounded_synthesis_object_list": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading", "payload-bound"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim; payload-bound: a hard cap or count bound that keeps an "
            "accepted payload bounded."
        ),
    ),
    "_check_architecture_status_mode": ShapeCheckClassification(
        "fatal",
        frozenset({"orchestrator-authored"}),
        (
            "Fatal sites: orchestrator-authored: state the orchestrator wrote or "
            "configured, where a violation is corruption rather than an agent defect."
        ),
    ),
    "_check_dropped_value_bound": ShapeCheckClassification(
        "fatal",
        frozenset({"payload-bound"}),
        (
            "DROPPED_VALUE_MAX_BYTES: no unbounded raw value is accepted by being dropped "
            "with a preview. Fatal sites: payload-bound: a hard cap or count bound that "
            "keeps an accepted payload bounded."
        ),
    ),
    "_citation_defect_observed": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "_claim_row_id_or_degradation": ShapeCheckClassification(
        "mixed",
        frozenset({"authentication-or-forgery", "payload-bound"}),
        (
            "Converts a malformed row ID into a claim-scope defect through a handled "
            "_validate_risk_row_id call; a row ID beyond the hard cap stays payload-bound. "
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows; payload-bound: a hard cap or count bound "
            "that keeps an accepted payload bounded."
        ),
    ),
    "_consume_agent_unavailable_footer_and_signature": ShapeCheckClassification(
        "fatal",
        frozenset({"unparseable-envelope"}),
        (
            "Fatal sites: unparseable-envelope: the response is not readable as the "
            "declared structured object (JSON extraction, footer and signature, top-level "
            "type)."
        ),
    ),
    "_consume_structured_footer_and_signature": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading", "unparseable-envelope"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim; unparseable-envelope: the response is not readable as the "
            "declared structured object (JSON extraction, footer and signature, top-level "
            "type)."
        ),
    ),
    "_degradable_response_impact": ShapeCheckClassification(
        "mixed",
        frozenset({"no-conservative-reading", "orchestrator-authored"}),
        (
            "Carries the #925 architecture-impact degradation for every response validator; "
            "strict assessment defects stay fatal. Fatal sites: no-conservative-reading: "
            "removing the element would weaken an obligation, finding, disposition, "
            "approved topology or discussion answer rather than a claim; "
            "orchestrator-authored: state the orchestrator wrote or configured, where a "
            "violation is corruption rather than an agent defect."
        ),
    ),
    "_degradable_test_observations": ShapeCheckClassification(
        "mixed",
        frozenset({"authentication-or-forgery", "payload-bound"}),
        (
            "Drops a malformed follow-up citation with its own citation-dropped record; "
            "more than CITATION_DEGRADATION_MAX_DROPS drops or an over-bound dropped value "
            "rejects under payload-bound, and record construction propagates "
            "marker-neutralization failures. Fatal sites: authentication-or-forgery: marker "
            "neutralization through sanitize_historical_text, identity digests, "
            "orchestrator-owned authority fields or canonical evidence rows; payload-bound: "
            "a hard cap or count bound that keeps an accepted payload bounded."
        ),
    ),
    "_degraded_row_claim_message": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "_dropped_execution_refs_caveat": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "_dropped_ref_preview": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "_expect_bool": ShapeCheckClassification(
        "delegated",
        frozenset(),
        (
            "Generic shape helper; the consequence is classified at each caller's annotated "
            "site."
        ),
    ),
    "_expect_deferred_stage_list": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_expect_disposition_list": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_expect_exact_keys": ShapeCheckClassification(
        "delegated",
        frozenset(),
        (
            "Generic shape helper; the consequence is classified at each caller's annotated "
            "site."
        ),
    ),
    "_expect_execution_allocation": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_expect_execution_disposition": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_expect_execution_recommendation": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_expect_human_requirement_dispositions": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_expect_int": ShapeCheckClassification(
        "delegated",
        frozenset(),
        (
            "Generic shape helper; the consequence is classified at each caller's annotated "
            "site."
        ),
    ),
    "_expect_item_id": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_expect_item_id_list": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_expect_item_note_map": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_expect_non_empty_string": ShapeCheckClassification(
        "delegated",
        frozenset(),
        (
            "Generic shape helper; the consequence is classified at each caller's annotated "
            "site."
        ),
    ),
    "_expect_object": ShapeCheckClassification(
        "delegated",
        frozenset(),
        (
            "Generic shape helper; the consequence is classified at each caller's annotated "
            "site."
        ),
    ),
    "_expect_optional_issue_id_list": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_expect_optional_string_list": ShapeCheckClassification(
        "delegated",
        frozenset(),
        (
            "Generic shape helper; the consequence is classified at each caller's annotated "
            "site."
        ),
    ),
    "_expect_plan_review_finding_list": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_expect_requirement_id_list": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_expect_review_finding_list": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "A dropped finding changes a verdict non-monotonically; the rewrapped "
            "normalize_fix_scope call and its re-raise both propagate. Fatal sites: "
            "no-conservative-reading: removing the element would weaken an obligation, "
            "finding, disposition, approved topology or discussion answer rather than a "
            "claim."
        ),
    ),
    "_expect_state": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_expect_string_list": ShapeCheckClassification(
        "delegated",
        frozenset(),
        (
            "Generic shape helper; the consequence is classified at each caller's annotated "
            "site."
        ),
    ),
    "_expect_test_observations": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Strict citation parser behind canonical evidence_citations: a verified "
            "canonical row must never stand on fewer citations than it declared. Fatal "
            "sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "_expect_typed_plan_stages": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_expect_typed_plan_stages.parse_child_stages": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_expect_typed_plan_stages.recorded_stages": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_extract_json_object_prefix": ShapeCheckClassification(
        "fatal",
        frozenset({"unparseable-envelope"}),
        (
            "Fatal sites: unparseable-envelope: the response is not readable as the "
            "declared structured object (JSON extraction, footer and signature, top-level "
            "type)."
        ),
    ),
    "_extract_structured_coder_followup_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"unparseable-envelope"}),
        (
            "Fatal sites: unparseable-envelope: the response is not readable as the "
            "declared structured object (JSON extraction, footer and signature, top-level "
            "type)."
        ),
    ),
    "_extract_structured_discuss_agenda_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"unparseable-envelope"}),
        (
            "Fatal sites: unparseable-envelope: the response is not readable as the "
            "declared structured object (JSON extraction, footer and signature, top-level "
            "type)."
        ),
    ),
    "_extract_structured_discuss_review_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"unparseable-envelope"}),
        (
            "Fatal sites: unparseable-envelope: the response is not readable as the "
            "declared structured object (JSON extraction, footer and signature, top-level "
            "type)."
        ),
    ),
    "_extract_structured_issue_implementation_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"unparseable-envelope"}),
        (
            "Fatal sites: unparseable-envelope: the response is not readable as the "
            "declared structured object (JSON extraction, footer and signature, top-level "
            "type)."
        ),
    ),
    "_extract_structured_plan_review_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"unparseable-envelope"}),
        (
            "Fatal sites: unparseable-envelope: the response is not readable as the "
            "declared structured object (JSON extraction, footer and signature, top-level "
            "type)."
        ),
    ),
    "_extract_structured_plan_revision_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"unparseable-envelope"}),
        (
            "Fatal sites: unparseable-envelope: the response is not readable as the "
            "declared structured object (JSON extraction, footer and signature, top-level "
            "type)."
        ),
    ),
    "_extract_structured_plan_state_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"unparseable-envelope"}),
        (
            "Fatal sites: unparseable-envelope: the response is not readable as the "
            "declared structured object (JSON extraction, footer and signature, top-level "
            "type)."
        ),
    ),
    "_extract_structured_pr_review_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"unparseable-envelope"}),
        (
            "Fatal sites: unparseable-envelope: the response is not readable as the "
            "declared structured object (JSON extraction, footer and signature, top-level "
            "type)."
        ),
    ),
    "_finalize_parsed_plan_review": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_finalize_parsed_review": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_flatten_plan_review_finding": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_hard_capped_claim_string": ShapeCheckClassification(
        "fatal",
        frozenset({"payload-bound"}),
        (
            "Fatal sites: payload-bound: a hard cap or count bound that keeps an accepted "
            "payload bounded."
        ),
    ),
    "_normalize_architecture_status": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery", "no-conservative-reading"}),
        (
            "Legacy closed-enum decode: both enum_error raises stay fatal; #925 settled "
            "which near misses degrade. Fatal sites: authentication-or-forgery: marker "
            "neutralization through sanitize_historical_text, identity digests, "
            "orchestrator-owned authority fields or canonical evidence rows; "
            "no-conservative-reading: removing the element would weaken an obligation, "
            "finding, disposition, approved topology or discussion answer rather than a "
            "claim."
        ),
    ),
    "_normalize_disposition": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_normalize_requirement_label": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_optional_semantic_fact_list": ShapeCheckClassification(
        "fatal",
        frozenset({"payload-bound"}),
        (
            "Fatal sites: payload-bound: a hard cap or count bound that keeps an accepted "
            "payload bounded."
        ),
    ),
    "_optional_semantic_fact_string": ShapeCheckClassification(
        "fatal",
        frozenset({"payload-bound"}),
        (
            "Fatal sites: payload-bound: a hard cap or count bound that keeps an accepted "
            "payload bounded."
        ),
    ),
    "_parse_architecture_impact": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_parse_architecture_impact_degradable": ShapeCheckClassification(
        "mixed",
        frozenset({"authentication-or-forgery", "no-conservative-reading"}),
        (
            "Mixed: degrades through classify_architecture_status_near_miss. Fatal sites: "
            "authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows; no-conservative-reading: removing the "
            "element would weaken an obligation, finding, disposition, approved topology or "
            "discussion answer rather than a claim."
        ),
    ),
    "_parse_architecture_impact_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_parse_complete_risk_test_matrix_row": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_parse_discuss_agenda_disagreement": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_parse_discuss_evidence": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_parse_discuss_research": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_parse_discuss_synthesis_change": ShapeCheckClassification(
        "fatal",
        frozenset({"kind-or-version-mismatch", "no-conservative-reading"}),
        (
            "Fatal sites: kind-or-version-mismatch: a kind, schema_version or contract "
            "version this parser does not read; no-conservative-reading: removing the "
            "element would weaken an obligation, finding, disposition, approved topology or "
            "discussion answer rather than a claim."
        ),
    ),
    "_parse_discuss_synthesis_consensus": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_parse_discuss_synthesis_disagreement": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading", "payload-bound"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim; payload-bound: a hard cap or count bound that keeps an "
            "accepted payload bounded."
        ),
    ),
    "_parse_discuss_synthesis_position": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_parse_discuss_synthesis_reference": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_parse_discuss_synthesis_references": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading", "payload-bound"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim; payload-bound: a hard cap or count bound that keeps an "
            "accepted payload bounded."
        ),
    ),
    "_parse_discuss_unresolved_items": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_parse_execution_contract_fields": ShapeCheckClassification(
        "fatal",
        frozenset({"kind-or-version-mismatch", "no-conservative-reading"}),
        (
            "Fatal sites: kind-or-version-mismatch: a kind, schema_version or contract "
            "version this parser does not read; no-conservative-reading: removing the "
            "element would weaken an obligation, finding, disposition, approved topology or "
            "discussion answer rather than a claim."
        ),
    ),
    "_parse_final_synthesis_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"kind-or-version-mismatch", "no-conservative-reading"}),
        (
            "Fatal sites: kind-or-version-mismatch: a kind, schema_version or contract "
            "version this parser does not read; no-conservative-reading: removing the "
            "element would weaken an obligation, finding, disposition, approved topology or "
            "discussion answer rather than a claim."
        ),
    ),
    "_parse_matrix_metadata": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_parse_plan_patch_field_value": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_parse_plan_revision_patch_operation": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Dropping a patch operation would silently change the revision, the inherited "
            "forbidden side effect. Fatal sites: no-conservative-reading: removing the "
            "element would weaken an obligation, finding, disposition, approved topology or "
            "discussion answer rather than a claim."
        ),
    ),
    "_parse_review_item_disposition_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_parse_risk_evidence_citations": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Canonical risk_test_matrix_evidence citations stay strict; a dropped citation "
            "could leave a verified row standing on less support. Fatal sites: "
            "authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "_parse_risk_test_matrix": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading", "payload-bound"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim; payload-bound: a hard cap or count bound that keeps an "
            "accepted payload bounded."
        ),
    ),
    "_parse_risk_test_matrix_changes": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading", "payload-bound"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim; payload-bound: a hard cap or count bound that keeps an "
            "accepted payload bounded."
        ),
    ),
    "_parse_risk_test_matrix_contract_fields": ShapeCheckClassification(
        "fatal",
        frozenset({"kind-or-version-mismatch", "no-conservative-reading"}),
        (
            "Fatal sites: kind-or-version-mismatch: a kind, schema_version or contract "
            "version this parser does not read; no-conservative-reading: removing the "
            "element would weaken an obligation, finding, disposition, approved topology or "
            "discussion answer rather than a claim."
        ),
    ),
    "_parse_round_synthesis_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"kind-or-version-mismatch", "no-conservative-reading"}),
        (
            "Fatal sites: kind-or-version-mismatch: a kind, schema_version or contract "
            "version this parser does not read; no-conservative-reading: removing the "
            "element would weaken an obligation, finding, disposition, approved topology or "
            "discussion answer rather than a claim."
        ),
    ),
    "_parse_semantic_risk_coverage_claims": ShapeCheckClassification(
        "mixed",
        frozenset({"authentication-or-forgery", "authority-decision", "orchestrator-authored", "payload-bound"}),
        (
            "Drops one claim per claim-scope defect with one bounded record (#926, #927). "
            "Stays fatal for CLAIM_RESERVED_AUTHORITY_KEYS, the row, ref, caveat and "
            "16,384-byte hard caps, an over-bound dropped value, catalog collisions and "
            "non-passing or launch-integrity-unknown in-catalog selectors; previews are "
            "marker-neutralized. Fatal sites: authentication-or-forgery: marker "
            "neutralization through sanitize_historical_text, identity digests, "
            "orchestrator-owned authority fields or canonical evidence rows; "
            "authority-decision: an in-catalog broker handle whose recorded outcome or "
            "launch integrity contradicts the claim (#990); orchestrator-authored: state "
            "the orchestrator wrote or configured, where a violation is corruption rather "
            "than an agent defect; payload-bound: a hard cap or count bound that keeps an "
            "accepted payload bounded."
        ),
    ),
    "_parse_structured_discuss_answer": ShapeCheckClassification(
        "fatal",
        frozenset({"kind-or-version-mismatch", "no-conservative-reading", "unparseable-envelope"}),
        (
            "Fatal sites: kind-or-version-mismatch: a kind, schema_version or contract "
            "version this parser does not read; no-conservative-reading: removing the "
            "element would weaken an obligation, finding, disposition, approved topology or "
            "discussion answer rather than a claim; unparseable-envelope: the response is "
            "not readable as the declared structured object (JSON extraction, footer and "
            "signature, top-level type)."
        ),
    ),
    "_parse_unresolved_item_dispositions": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_patch_value_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "_reject_legacy_requirement_labels": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_require_supported_schema_version": ShapeCheckClassification(
        "fatal",
        frozenset({"kind-or-version-mismatch"}),
        (
            "Fatal sites: kind-or-version-mismatch: a kind, schema_version or contract "
            "version this parser does not read."
        ),
    ),
    "_risk_bounded_string": ShapeCheckClassification(
        "delegated",
        frozenset(),
        (
            "Generic shape helper; the consequence is classified at each caller's annotated "
            "site."
        ),
    ),
    "_risk_bounded_string_list": ShapeCheckClassification(
        "delegated",
        frozenset(),
        (
            "Generic shape helper; the consequence is classified at each caller's annotated "
            "site."
        ),
    ),
    "_semantic_execution_ref_list": ShapeCheckClassification(
        "fatal",
        frozenset({"payload-bound"}),
        (
            "Fatal sites: payload-bound: a hard cap or count bound that keeps an accepted "
            "payload bounded."
        ),
    ),
    "_unapproved_row_claim_message": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "_validate_risk_owner": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "_validate_risk_row_id": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "classify_architecture_status_near_miss": ShapeCheckClassification(
        "mixed",
        frozenset({"authentication-or-forgery"}),
        (
            "Builds the #925 near-miss record directly; its construction propagates "
            "marker-neutralization failures. Fatal sites: authentication-or-forgery: marker "
            "neutralization through sanitize_historical_text, identity digests, "
            "orchestrator-owned authority fields or canonical evidence rows."
        ),
    ),
    "derive_risk_test_matrix_evidence": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery", "orchestrator-authored"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows; orchestrator-authored: state the "
            "orchestrator wrote or configured, where a violation is corruption rather than "
            "an agent defect."
        ),
    ),
    "parse_agent_state": ShapeCheckClassification(
        "fatal",
        frozenset({"unparseable-envelope"}),
        (
            "Fatal sites: unparseable-envelope: the response is not readable as the "
            "declared structured object (JSON extraction, footer and signature, top-level "
            "type)."
        ),
    ),
    "parse_agent_unavailable": ShapeCheckClassification(
        "fatal",
        frozenset({"kind-or-version-mismatch", "unparseable-envelope"}),
        (
            "Fatal sites: kind-or-version-mismatch: a kind, schema_version or contract "
            "version this parser does not read; unparseable-envelope: the response is not "
            "readable as the declared structured object (JSON extraction, footer and "
            "signature, top-level type)."
        ),
    ),
    "parse_architecture_impact": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "parse_architecture_impact_degradable": ShapeCheckClassification(
        "mixed",
        frozenset({"no-conservative-reading"}),
        (
            "Mixed: degrades through _parse_architecture_impact_degradable. Fatal sites: "
            "no-conservative-reading: removing the element would weaken an obligation, "
            "finding, disposition, approved topology or discussion answer rather than a "
            "claim."
        ),
    ),
    "parse_canonical_discuss_final_synthesis": ShapeCheckClassification(
        "fatal",
        frozenset({"orchestrator-authored"}),
        (
            "Fatal sites: orchestrator-authored: state the orchestrator wrote or "
            "configured, where a violation is corruption rather than an agent defect."
        ),
    ),
    "parse_canonical_discuss_round_synthesis": ShapeCheckClassification(
        "fatal",
        frozenset({"orchestrator-authored"}),
        (
            "Fatal sites: orchestrator-authored: state the orchestrator wrote or "
            "configured, where a violation is corruption rather than an agent defect."
        ),
    ),
    "parse_degradation_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery", "orchestrator-authored"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows; orchestrator-authored: state the "
            "orchestrator wrote or configured, where a violation is corruption rather than "
            "an agent defect."
        ),
    ),
    "parse_execution_recommendation_payload": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "parse_historical_structured_coder_followup": ShapeCheckClassification(
        "mixed",
        frozenset({"no-conservative-reading"}),
        (
            "Mixed: degrades through validate_structured_coder_followup. Fatal sites: "
            "no-conservative-reading: removing the element would weaken an obligation, "
            "finding, disposition, approved topology or discussion answer rather than a "
            "claim."
        ),
    ),
    "parse_historical_structured_issue_implementation": ShapeCheckClassification(
        "mixed",
        frozenset({"no-conservative-reading"}),
        (
            "Mixed: degrades through validate_structured_issue_implementation. Fatal sites: "
            "no-conservative-reading: removing the element would weaken an obligation, "
            "finding, disposition, approved topology or discussion answer rather than a "
            "claim."
        ),
    ),
    "parse_human_requirements_acknowledgement": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "parse_legacy_structured_discuss_answer": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "parse_plan_item_dispositions": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "parse_plan_review": ShapeCheckClassification(
        "mixed",
        frozenset({"unparseable-envelope"}),
        (
            "Mixed: degrades through parse_structured_plan_review. Fatal sites: "
            "unparseable-envelope: the response is not readable as the declared structured "
            "object (JSON extraction, footer and signature, top-level type)."
        ),
    ),
    "parse_plan_revision_patch": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery", "kind-or-version-mismatch", "no-conservative-reading", "payload-bound"}),
        (
            "The base-state identity digest is authentication-or-forgery; every "
            "operation-shape site is no-conservative-reading. Fatal sites: "
            "authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows; kind-or-version-mismatch: a kind, "
            "schema_version or contract version this parser does not read; "
            "no-conservative-reading: removing the element would weaken an obligation, "
            "finding, disposition, approved topology or discussion answer rather than a "
            "claim; payload-bound: a hard cap or count bound that keeps an accepted payload "
            "bounded."
        ),
    ),
    "parse_plan_state": ShapeCheckClassification(
        "fatal",
        frozenset({"unparseable-envelope"}),
        (
            "Fatal sites: unparseable-envelope: the response is not readable as the "
            "declared structured object (JSON extraction, footer and signature, top-level "
            "type)."
        ),
    ),
    "parse_pr_review": ShapeCheckClassification(
        "mixed",
        frozenset({"unparseable-envelope"}),
        (
            "Mixed: degrades through parse_structured_pr_review. Fatal sites: "
            "unparseable-envelope: the response is not readable as the declared structured "
            "object (JSON extraction, footer and signature, top-level type)."
        ),
    ),
    "parse_review": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading", "unparseable-envelope"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim; unparseable-envelope: the response is not readable as the "
            "declared structured object (JSON extraction, footer and signature, top-level "
            "type)."
        ),
    ),
    "parse_risk_test_matrix": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery", "no-conservative-reading", "payload-bound"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows; no-conservative-reading: removing the "
            "element would weaken an obligation, finding, disposition, approved topology or "
            "discussion answer rather than a claim; payload-bound: a hard cap or count "
            "bound that keeps an accepted payload bounded."
        ),
    ),
    "parse_risk_test_matrix_changes": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "parse_risk_test_matrix_evidence": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery", "no-conservative-reading", "payload-bound"}),
        (
            "Canonical evidence asserts verified coverage: identity and citation checks are "
            "authentication-or-forgery and row-count checks are payload-bound. Fatal sites: "
            "authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows; no-conservative-reading: removing the "
            "element would weaken an obligation, finding, disposition, approved topology or "
            "discussion answer rather than a claim; payload-bound: a hard cap or count "
            "bound that keeps an accepted payload bounded."
        ),
    ),
    "parse_risk_test_matrix_row": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "parse_structured_discuss_agenda": ShapeCheckClassification(
        "fatal",
        frozenset({"kind-or-version-mismatch", "no-conservative-reading", "unparseable-envelope"}),
        (
            "Fatal sites: kind-or-version-mismatch: a kind, schema_version or contract "
            "version this parser does not read; no-conservative-reading: removing the "
            "element would weaken an obligation, finding, disposition, approved topology or "
            "discussion answer rather than a claim; unparseable-envelope: the response is "
            "not readable as the declared structured object (JSON extraction, footer and "
            "signature, top-level type)."
        ),
    ),
    "parse_structured_discuss_answer": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "parse_structured_discuss_final_synthesis": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading", "unparseable-envelope"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim; unparseable-envelope: the response is not readable as the "
            "declared structured object (JSON extraction, footer and signature, top-level "
            "type)."
        ),
    ),
    "parse_structured_discuss_review": ShapeCheckClassification(
        "fatal",
        frozenset({"kind-or-version-mismatch", "no-conservative-reading", "unparseable-envelope"}),
        (
            "Fatal sites: kind-or-version-mismatch: a kind, schema_version or contract "
            "version this parser does not read; no-conservative-reading: removing the "
            "element would weaken an obligation, finding, disposition, approved topology or "
            "discussion answer rather than a claim; unparseable-envelope: the response is "
            "not readable as the declared structured object (JSON extraction, footer and "
            "signature, top-level type)."
        ),
    ),
    "parse_structured_discuss_round_synthesis": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading", "unparseable-envelope"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim; unparseable-envelope: the response is not readable as the "
            "declared structured object (JSON extraction, footer and signature, top-level "
            "type)."
        ),
    ),
    "parse_structured_plan_review": ShapeCheckClassification(
        "mixed",
        frozenset({"kind-or-version-mismatch", "no-conservative-reading", "unparseable-envelope"}),
        (
            "Mixed: degrades through _degradable_response_impact. Fatal sites: "
            "kind-or-version-mismatch: a kind, schema_version or contract version this "
            "parser does not read; no-conservative-reading: removing the element would "
            "weaken an obligation, finding, disposition, approved topology or discussion "
            "answer rather than a claim; unparseable-envelope: the response is not readable "
            "as the declared structured object (JSON extraction, footer and signature, "
            "top-level type)."
        ),
    ),
    "parse_structured_pr_review": ShapeCheckClassification(
        "mixed",
        frozenset({"kind-or-version-mismatch", "no-conservative-reading", "unparseable-envelope"}),
        (
            "Mixed: degrades through _degradable_response_impact. Fatal sites: "
            "kind-or-version-mismatch: a kind, schema_version or contract version this "
            "parser does not read; no-conservative-reading: removing the element would "
            "weaken an obligation, finding, disposition, approved topology or discussion "
            "answer rather than a claim; unparseable-envelope: the response is not readable "
            "as the declared structured object (JSON extraction, footer and signature, "
            "top-level type)."
        ),
    ),
    "parse_unresolved_item_dispositions": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "risk_test_matrix_identity": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows."
        ),
    ),
    "sanitize_architecture_impact": ShapeCheckClassification(
        "fatal",
        frozenset({"orchestrator-authored"}),
        (
            "Fatal sites: orchestrator-authored: state the orchestrator wrote or "
            "configured, where a violation is corruption rather than an agent defect."
        ),
    ),
    "sanitize_architecture_impact.clean": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery", "orchestrator-authored"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows; orchestrator-authored: state the "
            "orchestrator wrote or configured, where a violation is corruption rather than "
            "an agent defect."
        ),
    ),
    "sanitize_risk_test_matrix": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery", "orchestrator-authored"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows; orchestrator-authored: state the "
            "orchestrator wrote or configured, where a violation is corruption rather than "
            "an agent defect."
        ),
    ),
    "serialize_discuss_final_synthesis": ShapeCheckClassification(
        "fatal",
        frozenset({"orchestrator-authored"}),
        (
            "Fatal sites: orchestrator-authored: state the orchestrator wrote or "
            "configured, where a violation is corruption rather than an agent defect."
        ),
    ),
    "serialize_discuss_round_synthesis": ShapeCheckClassification(
        "fatal",
        frozenset({"orchestrator-authored"}),
        (
            "Fatal sites: orchestrator-authored: state the orchestrator wrote or "
            "configured, where a violation is corruption rather than an agent defect."
        ),
    ),
    "validate_human_requirement_dispositions": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "validate_human_requirements_acknowledgement": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "validate_risk_test_matrix_revision": ShapeCheckClassification(
        "fatal",
        frozenset({"authentication-or-forgery", "no-conservative-reading"}),
        (
            "Fatal sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows; no-conservative-reading: removing the "
            "element would weaken an obligation, finding, disposition, approved topology or "
            "discussion answer rather than a claim."
        ),
    ),
    "validate_structured_coder_followup": ShapeCheckClassification(
        "mixed",
        frozenset({"authentication-or-forgery", "authority-decision", "kind-or-version-mismatch", "no-conservative-reading", "payload-bound", "unparseable-envelope"}),
        (
            "Mixed: degrades through _degradable_response_impact, "
            "_degradable_test_observations, _parse_semantic_risk_coverage_claims. Fatal "
            "sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows; authority-decision: an in-catalog broker "
            "handle whose recorded outcome or launch integrity contradicts the claim "
            "(#990); kind-or-version-mismatch: a kind, schema_version or contract version "
            "this parser does not read; no-conservative-reading: removing the element would "
            "weaken an obligation, finding, disposition, approved topology or discussion "
            "answer rather than a claim; payload-bound: a hard cap or count bound that "
            "keeps an accepted payload bounded; unparseable-envelope: the response is not "
            "readable as the declared structured object (JSON extraction, footer and "
            "signature, top-level type)."
        ),
    ),
    "validate_structured_discuss_agenda": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "validate_structured_discuss_answer": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "validate_structured_discuss_answer_confirmation": ShapeCheckClassification(
        "fatal",
        frozenset({"kind-or-version-mismatch", "no-conservative-reading", "unparseable-envelope"}),
        (
            "Fatal sites: kind-or-version-mismatch: a kind, schema_version or contract "
            "version this parser does not read; no-conservative-reading: removing the "
            "element would weaken an obligation, finding, disposition, approved topology or "
            "discussion answer rather than a claim; unparseable-envelope: the response is "
            "not readable as the declared structured object (JSON extraction, footer and "
            "signature, top-level type)."
        ),
    ),
    "validate_structured_discuss_evidence_reconciliation": ShapeCheckClassification(
        "fatal",
        frozenset({"kind-or-version-mismatch", "no-conservative-reading", "payload-bound", "unparseable-envelope"}),
        (
            "Fatal sites: kind-or-version-mismatch: a kind, schema_version or contract "
            "version this parser does not read; no-conservative-reading: removing the "
            "element would weaken an obligation, finding, disposition, approved topology or "
            "discussion answer rather than a claim; payload-bound: a hard cap or count "
            "bound that keeps an accepted payload bounded; unparseable-envelope: the "
            "response is not readable as the declared structured object (JSON extraction, "
            "footer and signature, top-level type)."
        ),
    ),
    "validate_structured_discuss_final_synthesis": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "validate_structured_discuss_review": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "validate_structured_discuss_round_synthesis": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "validate_structured_discuss_semantic_comparison": ShapeCheckClassification(
        "fatal",
        frozenset({"kind-or-version-mismatch", "no-conservative-reading", "unparseable-envelope"}),
        (
            "Fatal sites: kind-or-version-mismatch: a kind, schema_version or contract "
            "version this parser does not read; no-conservative-reading: removing the "
            "element would weaken an obligation, finding, disposition, approved topology or "
            "discussion answer rather than a claim; unparseable-envelope: the response is "
            "not readable as the declared structured object (JSON extraction, footer and "
            "signature, top-level type)."
        ),
    ),
    "validate_structured_human_requirements_acknowledgement": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim."
        ),
    ),
    "validate_structured_issue_implementation": ShapeCheckClassification(
        "mixed",
        frozenset({"authentication-or-forgery", "authority-decision", "kind-or-version-mismatch", "no-conservative-reading", "payload-bound", "unparseable-envelope"}),
        (
            "Mixed: degrades through _degradable_response_impact, "
            "_degradable_test_observations, _parse_semantic_risk_coverage_claims. Fatal "
            "sites: authentication-or-forgery: marker neutralization through "
            "sanitize_historical_text, identity digests, orchestrator-owned authority "
            "fields or canonical evidence rows; authority-decision: an in-catalog broker "
            "handle whose recorded outcome or launch integrity contradicts the claim "
            "(#990); kind-or-version-mismatch: a kind, schema_version or contract version "
            "this parser does not read; no-conservative-reading: removing the element would "
            "weaken an obligation, finding, disposition, approved topology or discussion "
            "answer rather than a claim; payload-bound: a hard cap or count bound that "
            "keeps an accepted payload bounded; unparseable-envelope: the response is not "
            "readable as the declared structured object (JSON extraction, footer and "
            "signature, top-level type)."
        ),
    ),
    "validate_structured_plan_revision": ShapeCheckClassification(
        "mixed",
        frozenset({"kind-or-version-mismatch", "no-conservative-reading", "unparseable-envelope"}),
        (
            "Mixed: degrades through _degradable_response_impact. Fatal sites: "
            "kind-or-version-mismatch: a kind, schema_version or contract version this "
            "parser does not read; no-conservative-reading: removing the element would "
            "weaken an obligation, finding, disposition, approved topology or discussion "
            "answer rather than a claim; unparseable-envelope: the response is not readable "
            "as the declared structured object (JSON extraction, footer and signature, "
            "top-level type)."
        ),
    ),
    "validate_structured_plan_revision_patch": ShapeCheckClassification(
        "fatal",
        frozenset({"no-conservative-reading", "unparseable-envelope"}),
        (
            "Fatal sites: no-conservative-reading: removing the element would weaken an "
            "obligation, finding, disposition, approved topology or discussion answer "
            "rather than a claim; unparseable-envelope: the response is not readable as the "
            "declared structured object (JSON extraction, footer and signature, top-level "
            "type)."
        ),
    ),
    "validate_structured_plan_state": ShapeCheckClassification(
        "mixed",
        frozenset({"kind-or-version-mismatch", "no-conservative-reading", "unparseable-envelope"}),
        (
            "Mixed: degrades through _degradable_response_impact. Fatal sites: "
            "kind-or-version-mismatch: a kind, schema_version or contract version this "
            "parser does not read; no-conservative-reading: removing the element would "
            "weaken an obligation, finding, disposition, approved topology or discussion "
            "answer rather than a claim; unparseable-envelope: the response is not readable "
            "as the declared structured object (JSON extraction, footer and signature, "
            "top-level type)."
        ),
    ),
    "validate_structured_task_result": ShapeCheckClassification(
        "mixed",
        frozenset({"kind-or-version-mismatch", "no-conservative-reading", "unparseable-envelope"}),
        (
            "Mixed: degrades through _degradable_response_impact. Fatal sites: "
            "kind-or-version-mismatch: a kind, schema_version or contract version this "
            "parser does not read; no-conservative-reading: removing the element would "
            "weaken an obligation, finding, disposition, approved topology or discussion "
            "answer rather than a claim; unparseable-envelope: the response is not readable "
            "as the declared structured object (JSON extraction, footer and signature, "
            "top-level type)."
        ),
    ),
}
