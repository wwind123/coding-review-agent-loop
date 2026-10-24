# Testing and document maintenance

How the test suite exercises the orchestrator, when the architecture set must change, how architecture context is injected, and how semantic-claim validation degrades the narrowest offending element: architecture-impact parser modes and canonicalization, the reviewer status re-ask, `test_observations` citation degradation, and the `protocol.py` shape-check audit.

Source paths: `src/coding_review_agent_loop/architecture_context.py`, `src/coding_review_agent_loop/architecture_contract.py`, `src/coding_review_agent_loop/protocol.py`, `src/coding_review_agent_loop/validated_agent.py`, `src/coding_review_agent_loop/repair_preservation.py`, `src/coding_review_agent_loop/shape_check_audit.py`, `tests/`, `tests/test_architecture_context.py`, `tests/test_architecture_set_validation.py`, `tests/test_shape_check_audit.py`, `tests/test_docs_guidance.py`.

## Tests and Document Maintenance

[`tests/`](../../tests/) contains focused tests for the orchestrator flows, protocol,
resume/transport, scheduling, CI contracts, backend adapters, process execution,
and skill helpers. Fake subprocess/GitHub runners let lifecycle tests run
without real provider turns or GitHub mutations. They cannot fully reproduce
provider behavior, GitHub eventual consistency, or every host capability.

When changing component ownership, cross-component flows, persistence, trust
boundaries, or execution topology, update the architecture set and the affected detailed
guide in the same PR. An internal refactor or ordinary bug fix need not change
the document if those contracts stay the same. Keep diagrams textual and paths
relative; avoid copying volatile CLI defaults or protocol schemas here.

Architecture-context injection is bounded and advisory on the supported prompt
paths, and fresh turns with an acquired architecture snapshot use the versioned
architecture-impact assessment. Legacy records remain decodable without
inventing a snapshot or assessment; source inspection remains required.

The architecture path names the root of an architecture set: the root plus
the `.md` files it links to under the configured detail directories (default
`docs/architecture/`), one level only. An `ArchitectureMembershipPolicy`
built from the configuration decides membership: a linked document is a
member when it lies under a detail directory and under no history exclusion
(default `*/decisions/`, `doc/adr/`, `docs/adr/`), and an exclusion always
wins. The prompt freeze and the coder-HEAD membership hook use the same
policy, and managed recovery preserves the flags. The policy is never part of
an identity: member blob identities join the snapshot and PR-pair identity
only when members exist, so root-only and default-layout identities are
unchanged. The detail-doc index carries each member's path, first heading and
a bounded first-sentence summary; summaries are rendering detail only, and
they are dropped before entries, section indexes or overview text. An oversized root is injected as
identity, size and heading index with a split recommendation; coder
`canonical_document_path` values are validated against the set by a semantic
re-ask; and the PR comparison reports a detail-doc-only edit as a set change.
See `docs/local_agent_loop.md` for the layout and size budget.

Each detail doc's decision history lives in a paired, append-only log at
`docs/architecture/decisions/<same-name>.md`; the detail doc's final
Decisions section only links to it. Record new decisions there, keep the body
current, and read the log before changing a component's behaviour so a
rejected alternative is not proposed again. A replaced entry gets a
supersession note appended rather than being rewritten. The root never links
into the logs, so they stay outside the architecture set and out of prompts.
`tests/test_docs_guidance.py` checks the pairing and that every entry carries
an issue reference; `docs/local_agent_loop.md` gives the entry shape.

Semantic-claim validation degrades the narrowest offending element instead of
rejecting the whole response. The architecture-impact parser has three explicit
modes. `strict` is the default and accepts only `changed` and `unchanged`.
`legacy` is the synonym normalization, selected explicitly only where
stored or historical text is decoded: plan re-authentication, both topology
checkpoint decodes, and re-parses of posted or resumed records. `degradable`
is selected only by the agent-response validation of the seven
required-contract invocations and the four live review invocations. It honors
exactly one alias: `modified` becomes `changed` when the payload corroborates a
change, meaning every changed-only list, including the literal
`execution_data_flows` key, is non-empty, and the payload names a real
canonical-document action. Every other synonym, and an uncorroborated
`modified`, becomes the parser-only `undetermined` status, which has no wire
form and never resolves to `unchanged`. In every mode, an object item inside
one of the assessment's list fields is flattened deterministically before
validation to `key: value` text in the object's own key order, pairs joined by
`; ` and list values by `, `, and the parser logs one normalization line per
flattened item. The same renderability rule, exported from `protocol.py`, is
what counts as positive evidence for the `modified` near miss and what repair
preservation pins, so a renderable object survives a repair as the identical
object or as its flattened string. Nested objects, nested lists, non-string
scalars, empty values and empty objects still fail closed with an error that
names the field and the offending key; string-list fields outside the
assessment keep rejecting object items.

Accepted text is canonicalized at one acceptance boundary that every
successful exit of `_run_validated_agent` and of completion recovery goes
through. The status is rewritten to `changed`, or an undetermined optional
review assessment is removed. The rewritten text is then re-parsed strictly
under the candidate's own test-turn context, and the degradation records are
reattached, so later strict re-parses of posted text still succeed.

A required assessment that is omitted, removed, or `undetermined` does not
satisfy the contract. The validators return it without raising, and
`_run_validated_agent` refuses it. The refusal is deterministic and makes no
repair call. It takes an ordinary retry whose prompt names `architecture_impact`
and its accepted values, within the existing `--agent-max-retries` budget and
attempt bound. This applies to a primary response, a response-file artifact
from a timeout or nonzero exit, a completion-recovery response, and a repair
candidate. The refused response is kept on the final error.

A reviewer's own `architecture_impact.status` that is the sole defect in its
assessment (absent, null, blank, non-string, or outside `changed | unchanged`)
is a different transition from that ordinary contract retry. The
validator raises a typed `MissingJudgementFieldError` with the unchanged message,
and `_run_validated_agent` re-asks the same reviewer once with the frozen prompt
and original session plus a short section naming the field and its allowed
values, so peer visibility matches the original turn and nothing is published or
spooled between the two turns. The re-ask shares the single slot of the carried
disposition re-ask, so one invocation never re-asks twice, and it consumes
neither `--agent-max-retries` nor the Antigravity model fallback. A second
failure, or a flag-off invocation, makes no repair call and ends as a
deterministic reviewer failure that names the field and the allowed values.
Repair is skipped because it cannot supply the reviewer's decision. Any repair
still reached for a reviewer, such as when a structural defect masks the status,
is refused by repair preservation if the candidate introduces, changes, or drops
an `architecture_impact.status` that the source did not decide, whether or not
the source carries a `kind`. Mixed defects, near-miss aliases, and planner
responses keep their existing routing.

Semantic patch values stay strict. Canonical plan assembly and topology
checkpoint publication refuse a degraded status. Repair normalizes a near miss
before the repair prompt is built. It pins the absence of any removed
assessment, and of an omitted required one, and it refuses an unsatisfied
candidate before counting it as a success. Acknowledgement-only review repairs
validate new output strictly and keep the accepted assessment. Every
degradation produces a bounded, parser-derived record. Records travel on the
parsed result and in an optional round-metadata field, from which resumed
review carriers are rebuilt, and they appear in the round summary. A
decomposition surfaces its records in a plain parent-issue comment, whether it
was accepted or refused.

Model-authored `test_observations` citations in coder follow-ups and issue
implementations also degrade per element. A malformed citation is dropped with
its own `citation-dropped` record, and valid citations survive. More than
eight drops, the same limit the degradation renderer shows in full, reject the
response, so every kept record is stored and rendered. The records travel in
an optional `test_observation_degradations` round-metadata field, appear in
the follow-up comment and the round summary, and are restored onto the coder
carrier on resume. Canonical `risk_test_matrix_evidence` citations stay
strict: a verified row must never stand on fewer citations than it declared.

The `test_observations` field as a whole is decoded first and repaired before
it degrades (#1626). In every parse mode, a string that decodes to a JSON
array is used as that array, with one `citations-decoded-from-string` record
that renders on its own line ahead of the drop list. The parser takes a
`test_observations_mode`. The default, `degrade`, keeps today's single field
drop and is used by every historical, resume and legacy re-parse. Fresh coder
follow-up and issue-implementation turns build their validators with
`_architecture_and_test_observation_mode_validators`. Their primary pair is
`repair-first`: a value that is still not an array, or more than eight
malformed items, raises the repairable `TestObservationsShapeError`. The
validated turn then runs the structured repair with a targeted instruction
and the acquisition's tool-owned run-tests receipts. Both receipt
prohibitions are narrowed to canonical matrix-evidence envelopes. A scoped
preservation rule classifies the source as the parser does. A native or
decoded array pins its valid citations and lets malformed ones be corrected
or removed. Prose must be moved into `summary` or an item note, never
deleted. If repair yields no accepted candidate, a non-array value falls back
to the `degrade-after-repair` pair. That pair accepts the original through
the same acceptance boundary, with a drop record whose rule says repair was
attempted, and never spends an agent retry. The over-bound case stays fatal.
The validated turn classifies the shape defect from the source payload
(`source_has_test_observations_shape_defect`), not from the first validation
error, so a response whose other repairable defect is reported first still
gets the targeted repair. When the original still carries that other defect,
the fallback also tries a rejected repair candidate. The candidate's own
`test_observations` must still be the non-array field
(`candidate_keeps_degradable_test_observations`); a candidate that made it an
array was judged by the scoped preservation rule and is never rescued. The
candidate must also pass the unflagged preservation rules against the source,
it inherits the source's architecture near-miss record, and it is accepted
only with the after-repair drop record.

Every function, method and nested def in `protocol.py` that can still reach
an `AgentLoopError`-family raise is classified in `shape_check_audit.py` as
`fatal`, `mixed` or `delegated`. Each raise site and raise-reaching call site
carries an inline annotation. A propagating site names one clause from a
closed set: unparseable envelope, kind or version mismatch, authentication or
forgery, payload bound, authority decision, orchestrator-authored, or no
conservative reading. A site inside a `try` whose handlers cover every class
it can raise without re-raising is annotated as handled. Sites inside generic
helpers are annotated as delegated to their callers. Imported raisers, such
as the marker-neutralization check in `sanitize_historical_text`, are listed
with the classes they raise. `tests/test_shape_check_audit.py` recomputes the
transitive inventory from source, following bare names, same-module
`Class.method`, `cls`, `self` and `super()` calls, constructors, and untyped
receivers. It fails on an unclassified unit, a stale entry, a missing or
mismatched clause, or a degrading parser that can raise or builds records.

## Decisions

[Decision log](decisions/testing-and-doc-maintenance.md)
