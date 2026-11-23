# Testing and document maintenance

How the test suite exercises the orchestrator, when the architecture set must change, how architecture context is injected, and how semantic-claim validation degrades the narrowest offending element: architecture-impact parser modes and canonicalization, the reviewer status re-ask, `test_observations` citation degradation, and the `protocol.py` shape-check audit.

Source paths: `src/coding_review_agent_loop/prompts.py`, `src/coding_review_agent_loop/prompt_support.py`, `src/coding_review_agent_loop/review_diff.py`, `src/coding_review_agent_loop/architecture_context.py`, `src/coding_review_agent_loop/architecture_contract.py`, `src/coding_review_agent_loop/protocol.py`, `src/coding_review_agent_loop/protocol_common.py`, `src/coding_review_agent_loop/protocol_discuss.py`, `src/coding_review_agent_loop/protocol_risk.py`, `src/coding_review_agent_loop/validated_agent.py`, `src/coding_review_agent_loop/response_reask.py`, `src/coding_review_agent_loop/response_normalization.py`, `src/coding_review_agent_loop/repair_preservation.py`, `src/coding_review_agent_loop/shape_check_audit.py`, `tests/`, `tests/test_architecture_context.py`, `tests/test_architecture_set_validation.py`, `tests/test_shape_check_audit.py`, `tests/test_docs_guidance.py`, `tests/test_source_file_size.py`.

## Reader admission integration

`tests/test_pr_round_reader_boundary.py` owns the reader admission guard. Its reviewed family registry and per-call provenance live in `tests/pr_round_reader_inventory.json`, alongside explicit structural-reader exemptions for writers, presentation and offline codecs. Recursive dotted module discovery covers subpackages; alias and renamed re-export resolution runs to a fixed point. Calls retain arguments (including dynamic flow) and occurrence identity. A complementary structural scan detects new durable-history consumers even without a registered codec call. Synthetic additions run through the same production inventory assertion; duplicate, stale or empty review entries fail. The inventory is a review tripwire, not proof against arbitrary Python syntax: new readers require source inspection and an acquisition-owner description, never a generic raw-history exemption. PR amendment emitters (start-round lookup and activation template) read one shared record set, `_pr_amendment_comments`, admitted whenever a runner is available (#1986). Restart regressions retain fake remote history while reacquiring projections through real PR, plan and staged parent/child entry paths. See the canonical [per-family replay policy](round-state-and-resume.md#per-family-replay-policy).

## Tests and Document Maintenance

[`tests/`](../../tests/) contains focused tests for the orchestrator flows, protocol,
resume/transport, scheduling, CI contracts, backend adapters, and process execution. Fake subprocess/GitHub runners let lifecycle tests run
without real provider turns or GitHub mutations. They cannot fully reproduce
provider behavior, GitHub eventual consistency, or every host capability.

When changing component ownership, cross-component flows, persistence, trust
boundaries, or execution topology, update the architecture set and the affected detailed
guide in the same PR. An internal refactor or ordinary bug fix need not change
the document if those contracts stay the same. Keep diagrams textual and paths
relative; avoid copying volatile CLI defaults or protocol schemas here.

`prompt_support.py` owns shared prompt context dataclasses and reusable architecture,
memory, runtime/test, workdir, human-requirement and reviewer guidance. Role-specific
builders in `prompts.py` call this owner directly, so post-import patches intercept
shared helper calls without compatibility exports. The symbol manifest and unchanged
body digests live in `tests/fixtures/prompt_support_move.json`;
`tests/test_module_split_imports.py` audits old imports, aliases, annotations,
patch targets, unused builder imports, fresh import orders and shared patch coverage.
The prompt owner/path inventory is in `docs/prompt_support_ownership.md`; the
combined inventory and execution-boundary audit are in `docs/plan_execution_ownership.md`.

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

`tests/test_source_file_size.py` caps every `.py` file under
`src/coding_review_agent_loop/` at `DEFAULT_LIMIT` (6,000) lines, counted as
`len(text.splitlines())`. Files already over the limit when the guard landed
sit in its `ALLOWLIST` with their measured size as their own ceiling, so each
entry is a ratchet: the file may shrink but not grow. A failure names the
file, its size and limit, and whether the limit is the default or an
allowlist ceiling, and points to #1623. A second test rejects an entry whose
file is gone, is back at or under the default, or whose ceiling is not above
it. To split a file, move cohesive groups to sibling modules without behavior
changes, re-export the moved names, list the moved symbols in the PR, and
lower or delete the entry in the same PR.

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
modes. `strict` is the default and accepts only `changed` and `unchanged`
(after the shared case/separator fold described under lossless shape
normalization, so `Unchanged` reads as `unchanged` in every mode).
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
`_run_validated_agent` refuses it as a deterministic validation failure. This
applies to a primary response, a response-file artifact from a timeout or
nonzero exit, and a completion-recovery response. The refused response is
kept on the final error.

### Lossless shape normalization (#2109)

Mechanical slips whose meaning is unambiguous are normalized inside the
parsers, on the parsed value only, each with a warning log line; the stored
response text and semantic-patch bytes are never rewritten, so a re-parse of
the same text yields the same dataclass. The classes, all in
`protocol_common.py` or at the named parser site, are:

- `_expect_non_empty_string` unwraps a one-element list of a string;
  multi-element, empty and nested lists, objects, numbers and null still
  reject.
- `_expect_string_list` turns a lone non-empty string into a one-item list
  and `null` into `[]` when no minimum length applies; list items are checked
  strictly, so `["a", ["b"]]` still rejects.
- `fold_closed_enum` is the one closed-enum fold (strip, lowercase, `[\s_]+`
  to `-`, matched only against the allowed tokens, no synonym tables). It is
  applied to every closed enum, including the payload `state` before the
  footer comparison and the ordinary discussion enums (research status and
  target, outcome, analyzer framing, answer position and confidence,
  unresolved-item status, confirmation decision, semantic and final-synthesis
  classifications, agenda research targets); `normalize_staging_feasibility`
  is rebuilt on it and keeps its audited safe-alias set. Discussion evidence
  claim status, verification basis and update action stay exact. The architecture status near-miss classifier
  treats a fold of a declared status as no near miss, so `Unchanged` reaches
  normal parsing and an optional review assessment is no longer removed;
  real synonyms (`modified`, `no-change`, `none`, ...) keep the degrade and
  patch-reject handling above.
- `architecture_impact` list fields accept `null` or a lone string, an empty
  `canonical_document_path` is `null`, and a `null` action is `no-change`.
  The documented `no-change` action token is folded (`NO_CHANGE`); other
  actions are open text and kept as written.
- A `none` allocation (or a `null` one) gets its empty lists; a `required`
  allocation is never defaulted. A `null` `one_shot_delivery` on a staged
  recommendation is dropped.
- `_expect_int_like` accepts `^\s*#?\d+\s*$` strings for stage `position`
  and `issue_implementation`/`task_result` `pr_number`; `_expect_int` stays
  strict elsewhere.
- Review finding lists accept `null` or a lone string, and a PR finding
  object without `text` takes exactly one of `description`/`issue` as its
  text and wraps a lone-string `fix_scope`. This runs once on the parsed
  payload copy, before every reader of those lists; collision checks are
  unchanged.

The strict boundary is `_expect_strict_string` and the `strict_shape`
list option: item, stage, scope, constraint and risk-row IDs, signed
requirement labels, risk-matrix text and evidence citations, semantic-patch
`op`/`field`/`base_state_identity`, the `agent_unavailable` envelope, the
payload `state` (an envelope field compared with the footer, so a list never
bypasses that comparison), discussion reviewer and observation IDs, synthesis
reviewer references and reviewer lists, and synthesis change `kind` stay
exact. The only identifier-list
outer-shape exception is the opt-in `normalize_outer_shape` for
`depends_on_stage_ids`; each element is still a strict ID.

### Harmless extras (#2114)

Extra content that asserts nothing is dropped or aliased instead of costing a
re-ask. The rule applies to the parsed value only: stored text is never
rewritten, so a resume re-parse gives the same result. Identities and digests
come from parsed values or the orchestrator's canonical payload, never from
raw agent JSON.

- `_expect_known_keys_lenient` (in `protocol_common.py`) is an opt-in twin of
  `_expect_exact_keys`. It drops an unknown key with a warning only when the
  value is a string and the name is not protected. Any other unknown key
  raises the existing unknown-field error. Protected names are
  `PROTECTED_RESPONSE_KEY_NAMES` (envelope fields, `item_id`/`id`/
  `disposition`, requirement labels, `pr_number`, head SHAs, evidence and
  matrix identity fields, receipt, dispatch, managed-CI and tool-stamped
  recommendation fields). Names that contain `<`/`>`, `agent_`, `signature`,
  `managed_ci`, `provenance`, `marker`, `nonce` or `dispatch` are also
  protected. These sites opt in: the `pr_review`, `plan_review`,
  `coder_followup`, `issue_implementation`, `task_result`, `plan_revision` and
  `plan_state` envelopes, the `human_requirements` objects, review item
  dispositions, human-requirement dispositions, execution child stages, and
  the complete risk-matrix envelope and rows.
- These sites stay strict: `agent_unavailable`, semantic plan-patch operations
  (parsed inside `strict_response_keys`, so every shared parser they reach,
  such as replacement requirement dispositions or execution-recommendation
  child stages, keeps exact keys with no drop, rename or duplicate collapse)
  and their matrix rows and metadata, `risk_test_matrix_changes`,
  `architecture_impact`, every authentication-or-forgery parser, and records
  the orchestrator writes.
- Renames come in two kinds. `blocking_items` and `same_pr_followups` in a plan
  review are renamed to the plan keys before finding normalization, so the
  findings are kept; an alias next to its canonical key rejects. In review item
  dispositions, `reason` and then `rationale` become `note` when `note` is
  absent. A string alias next to an existing `note` is dropped with a warning.
  A non-string alias (null, a list, a number) always rejects, with or without
  `note`. A `rationale` that is itself a known key is never renamed.
- A plan-review disposition loses `sub_item_dispositions` with a
  `ParseDegradation` record (`ParsedPlanReview.disposition_degradations`), and
  the parent disposition stays authoritative. A plan finding drops a string or
  string-list `fix_scope`.
- Human requirements: a surplus `checked_discussion_directly: true` is a
  warning. When dispositions are present, `addressed_ids` is derived from the
  surfaced `addressed` dispositions. Live parsers carry the derived tuple,
  while a historical decode keeps the submitted list.
  `filter_surfaced_requirement_dispositions` drops dispositions for unsurfaced
  IDs before any blocked-state or conflict decision. Legacy labels,
  reviewer-owned IDs, missing coverage and conflicting duplicates still reject.
  Every caller that keeps a parsed record (the plan-review, plan-state and
  plan-revision acceptance paths, and their recovery reader) stores the kept
  tuple that `validate_human_requirement_dispositions` returns. Semantic-patch
  dispositions stay strict: assembly publishes them verbatim, so an unsurfaced
  ID there still rejects.
- Context-dependent checks run only in live validators. Every context-free
  reader of an already-accepted response passes `historical_decode=True`:
  `parse_historical_structured_*`, and the post-authentication
  claim-correction reader (`_parse_fresh_correction_claims`), which takes only
  the claim set. So an accepted approved follow-up beside an unsurfaced
  blocked disposition, or an implementation PR beside one, decodes with a
  warning.
- Duplicates: entries that are equal on every parsed field collapse with a
  warning. This covers review dispositions (including
  `evidence_requirement_ids`), requirement dispositions, IDs repeated within
  one coder bucket and repeated `addressed_ids`. Any differing field, or the
  same ID in two coder buckets, still rejects.
- On the coder side, a note keyed to an unlisted ID is dropped. In the live
  validator, an ID outside the ledger is dropped through
  `partition_item_id_buckets` (shared with the post-push degradation path),
  together with its notes, sub-items and dispute evidence. It is recorded in
  `StructuredCoderFollowup.dropped_unknown_items` and logged by the PR loop.
  Missing classification still rejects.

### The same-agent re-ask rule (#2066)

No second model reformats a rejected response. A deterministic validation
failure of any structured response goes back to the agent that produced it:
the same agent, the model that produced the response (a Claude, Codex or Gemini
response whose observed model differs from the configured one is re-asked on
the observed model for every later re-ask of the turn, replacing a `--model`
passthrough argument; a named seat is re-issued with that model first, and an
Antigravity chain is not advanced), and the response's own session when there is one (otherwise
the full original prompt). The re-ask prompt is the original prompt plus a
bounded, marker-free copy of the validator message
(`response_reask.build_reask_prompt`) and an instruction to resend the
complete response only. A turn gets at most two re-asks
(`MAX_SAME_AGENT_REASKS`). They replace the former repair pass, content
preservation and the separate omission, judgement, evidence,
architecture-contract and risk-matrix replay slots. The re-asks consume neither
`--agent-max-retries` nor the Antigravity model fallback, and transient,
provider, quota, timeout and unsupported-model failures keep their own retry
paths without spending a re-ask. A plain-prose response that names an auth or
billing stop is a provider failure, not a re-ask cause.

Typed defects keep a targeted hint section ahead of the generic re-ask
appendix: the missing carried item IDs (`MissingPriorItemDispositionError`),
the judgement field and its allowed values (`MissingJudgementFieldError`),
the architecture contract field or the out-of-set canonical document path,
and the evidence rejection reason.

A coder re-ask (role `coder`, or a `coder_followup`/`issue_implementation`
turn) also carries an instruction not to modify, commit or push the checkout.
The assigned checkout head is read before and after each coder re-ask. An
unreadable baseline means no re-ask; a moved or unreadable head after the
re-ask fails the turn with a deterministic error naming both heads, never
accepts the re-asked response, and makes no further re-ask. pr_loop's
post-turn head comparison and the post-push coder record (#2086) then handle
the moved head.

The evidence scope is decided by the instruction. One active acquisition is
kept, initially the primary dispatch. A `NonRepairableEvidenceRejection`
whose reason is `non-passing-selector`, `launch-integrity` or
`report-workdir` makes the old observations non-citable: that re-ask keeps
the rerun-tests prompt, still forbids commits and pushes, and its dispatch
becomes the active acquisition. Every other cause, including a refused
`tests_run` entry, is a resend-only re-ask that inherits the active
acquisition unchanged. Validation, strict re-parsing, accepted and
rejected-candidate metadata and pr_loop's evidence derivation all read the
active acquisition, so receipts produced during a resend-only dispatch are
never admitted and catalogs are never merged.

When both re-asks fail, the turn fails as before: the same
`AgentInvocationError`, the deterministic category, the plan-validation
exhaustion capture for the last planner candidate and the bounded replan for
semantic-patch payload rejections. The message is prefixed with
`response rejected after 2 same-agent re-asks`. A reviewer response that
still carries no review substance (narration, an envelope-only payload) is a
reviewer availability failure (`agent-unavailable`) that keeps the ordinary
retry and Antigravity format-fallback policy, never a verdict.

Purely mechanical normalization stays in code and runs for every turn with
no model call (`response_normalization.py`): envelope trimming, the
lone-string risk-matrix list fields and the semantic-patch `rationale` alias.
Image turns use the same envelope normalization. Architecture
near-miss canonicalization (`canonicalize_architecture_near_miss_text`) also
lives there and runs at the shared acceptance boundary.

Prior-item dispositions for IDs outside the live carried set (or its PR-review
aliases) are dropped, never rejected (#2100). The plan semantic patch (payload
and full validators), the legacy full-state plan revision, the plan review and
the PR review all pass the carried set to the parser, and
`partition_raw_dispositions` (`protocol_common.py`) removes those entries from
the raw array before it is parsed. Per-entry parser records such as sub-item
degradations therefore exist only for kept entries, with indices into the
filtered array. The verdict-consistency check sees only kept entries, so an
approval whose only active disposition names a non-carried ID is accepted.
There is no count or length bound, so non-carried IDs alone never cause a
rejection, re-ask or bounded replan. Duplicate carried IDs are rejected by all
four validators; for a semantic patch that is a payload rejection.
Missing dispositions keep their semantics: reviewers still fail on a missing
carried disposition, and plan candidates, which never had a completeness check,
gain none. The dropped IDs travel on the parsed result
(`ignored_prior_disposition_ids`, never part of a payload, canonical plan or
sidecar identity). The acceptance boundary (`_accept_candidate`) removes the
dropped entries from the accepted text with `remove_prior_item_dispositions`,
strictly re-parses it, compares under a projection that clears the ignored IDs,
and reattaches them. Publication, canonical reviewer responses and resume
parsers that read the text without a carried set therefore agree with live
validation. The wire kind comes from a fixed type map, because the reviewer
result types have no `kind` field. Spool records persist the IDs (only when
present), and replay reattaches them. The optional round-metadata field
`ignored_prior_disposition_ids` is in the lossless sidecar spill set. It renders
as an `### Ignored prior-item dispositions` note that lists every ID up to a
display budget and then a count. The compact projection, which the overflow plan
digest now uses, shows a count indicator instead. Every section the round
metadata attaches is inserted above any trailing execution-recommendation or
risk-matrix section, because the transport may replace those sections with a
bounded summary. The response-normalization
strip helpers stay but are unreachable for live turns. The
`--repair-backend`, `--repair-model`, `--repair-reasoning-effort`,
`--repair-timeout-seconds` and `--containment-repair-*` flags are deprecated
no-ops: they still parse, print one deprecation notice, are never validated
and no longer require the agy CLI; containment drops the repair-only overrides
before policy parsing. `repair.py`, `repair_preservation.py` and
`shape_check_audit.py` are unreachable at runtime and are deleted in stage 2.

Semantic patch values stay strict. Canonical plan assembly and topology
checkpoint publication refuse a degraded status. Every degradation produces a
bounded, parser-derived record. Records travel on the parsed result and in an
optional round-metadata field, from which resumed review carriers are
rebuilt, and they appear in the round summary. A decomposition surfaces its
records in a plain parent-issue comment, whether it was accepted or refused.

Model-authored `test_observations` citations in coder follow-ups and issue
implementations also degrade per element. A malformed citation is dropped with
its own `citation-dropped` record, and valid citations survive. More than
eight drops, the same limit the degradation renderer shows in full, reject the
response, so every kept record is stored and rendered. The records travel in
an optional `test_observation_degradations` round-metadata field, appear in
the follow-up comment and the round summary, and are restored onto the coder
carrier on resume. Canonical `risk_test_matrix_evidence` citations stay
strict: a verified row must never stand on fewer citations than it declared.

The `test_observations` field as a whole is decoded first and re-asked before
it degrades (#1626, #2066). In every parse mode, a string that decodes to a
JSON array is used as that array, with one `citations-decoded-from-string`
record that renders on its own line ahead of the drop list. The parser takes
a `test_observations_mode`. The default, `degrade`, keeps today's single field
drop and is used by every historical, resume and legacy re-parse. Fresh coder
follow-up and issue-implementation turns build their validators with
`_architecture_and_test_observation_mode_validators`. Their primary pair is
`repair-first`: a value that is still not an array, or more than eight
malformed items, raises `TestObservationsShapeError`, which is re-asked. If
the re-asks yield no accepted candidate, a non-array value falls back to the
`degrade-after-repair` pair (the historical rule name stays parseable). That
pair accepts the latest response through the same acceptance boundary, with a
drop record, and never spends an agent retry. It runs only after both re-asks
were spent; a coder head-guard refusal fails the turn instead. When envelope
normalization exposes the defect, the normalized candidate and its own
rejection carry forward to the re-ask and the degradation. The over-bound case
stays fatal.

Every function, method and nested def in `protocol.py`, `protocol_common.py`, `protocol_discuss.py` and `protocol_risk.py` that can still reach
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
transitive inventory from all three owners as one linked parser graph, following bare names, same-module
`Class.method`, `cls`, `self` and `super()` calls, constructors, and untyped
receivers. It fails on an unclassified unit, a stale entry, a missing or
mismatched clause, or a degrading parser that can raise or builds records.

The owner list is explicit in `shape_check_audit.PARSER_OWNERS`. The inventory
links internal imports to their definitions while retaining external imported-raiser
verification, degradation coverage and the closed fatal-clause set.
`tests/fixtures/protocol_support_move.json` records moved symbols, original
body hashes, global references, decorators and consumer/lookup candidates for
the cleanup rewrite in `scripts/rewrite_protocol_support_imports.py`.
`tests/test_protocol_support_move.py` checks verbatim bodies and fresh-process
import orders. `tests/test_module_split_imports.py` guards the cleaned support,
discussion parser and discussion type symbols against old imports, aliases,
qualified accesses, patch strings and compatibility-only driver exports. A
consumer regression patches the dependency at the discussion parser lookup.
Necessary internal imports remain permitted; downstream consumers use the owners.

`protocol_risk.py` owns risk matrix records, identities and revisions, semantic
claims/gaps, test-observation citation parsing and canonical evidence derivation,
including signed-reference handling and matrix row/metadata patch parsing.
`protocol.py` retains response-composition adapters and only their necessary
risk imports; all risk consumers use the definition owner directly. The new owner imports shared support directly
and never imports the response driver. Bounds, diagnostics and evidence authority
are unchanged. `tests/fixtures/protocol_risk_move.json` records original symbol
bodies, global references, decorators and consumer/patch candidates for the paired
cleanup via `scripts/rewrite_protocol_risk_imports.py`. The symbol guard covers
all risk names, including aliases, patch strings and compatibility-only exports.
`tests/test_protocol_risk_move.py` checks body hashes, definition ownership,
fresh import orders, source sizes and interception at the moved citation lookup;
the linked shape audit includes the risk owner and tests missing annotations there.


The PR-review `requirement_evidence` field records semantic signed-source
classification (#1922). Every surfaced signed source must be classified whenever
the review does not acknowledge it as resolved, including ordinary work with role
`code`. Missing classification triggers one bounded reviewer continuation; a
repeated omission stops the run before coder dispatch. Entries contain requirement_id, role (code, human-only,
evidence-only, mixed, superseded), request, answered_item_ids and leftover_instructions.
The parser checks shape; routing checks surfaced source identity, native target
existence and requester ownership. Only `request` and `leftover_instructions`
normalize empty lists or null to empty strings before strict validation; invalid
text types report the indexed field and observed type (#1943). String leftovers
for `code`, `human-only` and `evidence-only` are cleared on copies (#1954), without
relocating text into findings or requests; mixed leftovers must remain nonblank.
`superseded` (#2101) is excluded from that clearing: a nonblank superseded
leftover is a judgement re-ask, while null, `[]` or whitespace-only leftovers
canonicalize to empty. The role is valid only when a strictly newer surfaced
signed source from the same author exists.
Repair compares copies under this same role-aware normalization only when both collections
parse, retaining raw equality otherwise. Collection shape, order and all other
content stay protected. A real bounded reviewer continuation owns missing
classification. Mixed-source prompt and disposition validation use the original
signed ID with the verbatim source and attributed leftover scope; agreement is by role (#2000). Classification is separate from native
evidence authorization and does not waive live citation validation. Report-side
command parsing recognizes the exported executable spelling while continuing to
validate inner test targets against the assigned checkout.


## Decisions

[Decision log](decisions/testing-and-doc-maintenance.md)
