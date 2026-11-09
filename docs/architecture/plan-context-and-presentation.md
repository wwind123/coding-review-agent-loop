# Plan context and presentation

How plans are assembled, presented and scoped: the semantic `plan_revision_patch` and the deterministic plan assembler, the bounded visible digest used when a structured plan comment overflows, the collapsed delta presentation of coder matrix evidence, and the fresh planning scope started when the issue text changes under an unapproved plan.

Source paths: `src/coding_review_agent_loop/plan_assembly.py`, `src/coding_review_agent_loop/round_transport.py`, `src/coding_review_agent_loop/comment_rendering.py`, `src/coding_review_agent_loop/plan_issue_scope.py`, `src/coding_review_agent_loop/plan_board_state.py`, `src/coding_review_agent_loop/panel_evidence.py`, `tests/test_plan_assembly.py`, `tests/test_comment_rendering.py`, `tests/test_plan_issue_scope.py`.

## Semantic planning foundation

Generation-1 planning remains the downstream wire contract. A semantic
`plan_revision_patch` v1 is an intent-only input: the model may replace an
approved field in the bounded allow-list or declare a complete matrix row
operation, but it cannot write kinds, contract versions, audit entries,
ordering, identities, sidecars, or protocol records.

`plan_assembly.py` is the deterministic boundary. It authenticates the base
round and aggregate identity, validates every operation before mutation, and
applies row edits, retirements, splits, merges, and additions simultaneously.
Split replacement is ordered, merge replacement uses the lowest source slot,
and additions occupy declared final positions. Audit rows are derived in the
fixed order add, change, retire, split, merge; row IDs are normalized
lexically. A metadata-only transition over an empty matrix uses `matrix` as
its complete-scope sentinel.

The assembled object is emitted as the existing validated
`StructuredPlanRevision`. Its canonical JSON sidecar and aggregate SHA-256
identity are the restart authority; raw semantic patches are provenance only.
The identity explicitly covers the full plan plus architecture, execution,
matrix, closing, human-disposition, and typed-category data. Rendered
Markdown remains a downstream presentation surface.

Publication now seeds that sidecar prospectively for fresh `plan_state` and
new validated full-state `plan_revision` records. A new unapproved round with
a complete authenticated matrix is pinned to `semantic-patch-v1`; its full,
compact, retry, and restart paths all hydrate the same sidecar and invoke the
same assembler. The response form is durable round metadata, so a legacy or
in-flight round cannot silently switch forms. Matrix-less and pre-rollout
records remain on the legacy path and are never backfilled.

## Compact-on-overflow plan presentation

The transport spills round *metadata* into authenticated sidecars, but the
visible Markdown of a structured plan comment had no spill path, so a plan whose
prose alone exceeded the 60,000-character body budget (typically a separately
planned child that must preserve many inherited matrix rows verbatim) could not
be posted. Structured `plan_state` and `plan_revision` coder comments therefore
have two presentations of the same plan:

- **Full form** (default). Unchanged and byte-identical to earlier releases. It
  is used whenever `round_transport.round_comment_fits` reports that the
  assembled comment can be transported, with or without the existing spills.
- **Bounded visible digest.** Selected by the shared publication helper
  `_assemble_structured_plan_round_body` only when the fit check reports the
  dedicated `RoundCommentOverflowError`. Every other transport failure
  (malformed structured record, non-serializable metadata, provenance failure,
  oversized sidecar part) propagates and aborts publication; it never selects
  the digest. Round metadata is always built from the full canonical rendering
  before either presentation is chosen, so `canonical_plan`, the plan subject,
  the rendered-plan identity, the assembled plan sidecar and the matrix identity
  are identical for both forms.

The digest is a pure deterministic function of the parsed plan. Its budgeted
sections (summary, prior plan item dispositions as ID plus disposition, plan
steps, additional closing issue IDs, deferred stages and each typed stage
category by title) share one aggregate budget,
`COMPACT_PLAN_DIGEST_BUDGET_CHARS` (12,000 characters). Each section owns a
fixed share that already includes its heading and its omitted-entry line; entries
are emitted in canonical order, each sanitized and clipped to a fixed per-entry
ceiling, until the next entry would exceed the share, and the section then ends
with one line stating how many of how many entries were omitted. Unused share is
never redistributed, so the bound holds by construction for any entry count or
string length, and the renderer asserts it.

Record contract of the digest:

- *Visible-anchor set* (always present): the risk-matrix record with its
  renderer boundary, the execution-recommendation record with its renderer
  boundary, the signed-requirements acknowledgement record when the raw response
  carried one, the plan-state footer, the signature, and the round metadata
  record. The matrix and recommendation sections come from the unchanged section
  renderers; the transport's existing authenticated reference rewrites of those
  two payloads and of the round metadata payload are the only rewrites allowed.
- *Canonical-metadata-only set* (deliberately absent): the deferred-stage,
  typed-stage and expected-closing payload records. Their sections appear only
  as bounded indexes. They remain lossless because identical records stay inside
  `canonical_plan` in authenticated (spillable) round metadata and the assembled
  plan sidecar carries the same collections as structured state. Every reader of
  these records (`_extract_current_deferred_stages`,
  `_extract_current_expected_closing_issue_ids` and the typed-stage extractors
  in `execution_policy.py`) takes the canonical plan text, never the visible body of
  a plan coder round comment.
- *Signed requirement IDs are never omitted.* The acknowledgement block keeps
  its record, heading, every requirement ID line and the direct-discussion
  sentence; the dispositions section lists every requirement ID. Only
  explanatory evidence text is clipped, and a clip never cuts inside an ID. These
  sections are outside the omitted-entry policy and bounded only by the prompt's
  surfaced-ID cap. For a fresh `plan_state` the parsed plan does not retain the
  raw acknowledgement, so the compact path alone reads the validated raw
  response (`raw_text`); the full rendering ignores it. Before posting, the
  helper revalidates the digest with the same surfaced IDs and direct-discussion
  flag that validated the raw response, and refuses to post a digest that fails.

On resume, authenticated round metadata and its sidecars are the sole source of
everything the digest omitted: plan resume reads `metadata.canonical_plan`,
reviewers are prompted with the complete plan, and child matrix validation reads
the hydrated structured matrix. The digest is never parsed back as plan text.
Free-form (unstructured) plans are not compacted, because their visible text is
their only representation. Reviewer, PR-review and discuss comments keep their
current rendering.

When even the digest cannot fit (unspilled round metadata, the signed-requirement
sections at their cap, or a free-form plan), publication fails before any
comment or sidecar is posted. The overflow error keeps its leading sentence and
adds size attribution: visible characters outside the round metadata record,
residual encoded metadata characters, and the names and encoded sizes of the
largest unspilled metadata fields. It reports names and integers only, never
content.

Repair runs before assembly and is lossless for semantic patches: it may fix
only the response envelope and cannot add operations, change rationales, or
alter base bindings. A cross-form, approved, stale, incomplete, or identity-
mismatched response fails before publication. The assembled generation-1
Markdown and its existing subject/hash remain the downstream review and
implementation surface; the sidecar is the durable authority used for
hydration.

## Delta matrix-evidence presentation

The visible `### Risk-based mode and transition test matrix evidence` section of
coder comments is presentation only: reviewers and resume read the canonical
`risk_test_matrix_evidence` in round metadata (and the matrix sidecar), which
always carries every row. The section therefore keeps its heading and
`- Matrix identity:` line visible but wraps the row list in a renderer-emitted
`<details><summary>…</summary>` block, padded with blank lines so GitHub renders
the Markdown inside it collapsed. The summary text is built only from counts and
integer round numbers; row text is still sanitized.

`comment_rendering.resolve_matrix_evidence_render` is the single place that
chooses the presentation. It returns a frozen `MatrixEvidenceRenderDecision`
(`mode`, `anchor_round`, `previous_evidence`, `previous_round`), never raises,
and `_render_risk_test_matrix_evidence` follows the decision without comparing
anything itself:

- **Full** (anchor = the coder record's own round number). Chosen when there is
  no previous coder metadata, the previous evidence is absent or fails
  `parse_risk_test_matrix_evidence`, the matrix identity differs, the row-ID sets
  differ, or the previous anchor is unusable. A full decision carries no
  previous evidence. Every row is rendered.
- **Delta**. Chosen only for the same identity and the same row-ID set with a
  usable anchor. Only rows whose comparison key changed are shown, followed by
  `N rows unchanged since round P; full matrix in round F.`; with no changes that
  line is the only content. The comparison key is the row payload with each
  citation reduced to `(command, claim)`, so a fresh per-turn receipt ID alone is
  not a change. The anchor is carried forward unchanged.

The PR coder follow-up computes the posted record's round number once
(`round_number + 1`) and uses it both as the helper's `current_round` and as the
metadata `round_number`; it passes the same decision to
`render_public_agent_comment(matrix_evidence_render_decision=...)` and persists
its `anchor_round`. When the derived current evidence is `None` the helper is not
called, no decision is passed, the section is omitted and no anchor is written;
a later round with evidence then renders full. Issue-implementation comments that
carry canonical evidence always render the full list, and both issue-to-PR
metadata sites persist their own round as the anchor. The structured no-PR and
rejected-conflict terminal comments carry no canonical evidence and omit the
section.

The anchor is the optional `PostedRoundMetadata.risk_test_matrix_evidence_full_round`
integer, written only when not `None` (so earlier encodings stay byte-identical)
and set if and only if the record persists matrix evidence. It is a small
unspillable scalar. A non-serialized
`risk_test_matrix_evidence_full_round_status` distinguishes decode quality, in
the same style as `scheduler_metadata_status`: a missing key decodes as
`absent`; a positive non-bool integer decodes as `valid`; any other present value
(bool, zero, negative, string, null, float) decodes as anchor `None` with status
`invalid`, without raising. Construction with an anchor promotes the status to
`valid`, so in-process and decoded records agree, and an anchor combined with
`invalid` is rejected. The helper treats a `valid` anchor as usable only when it
is a positive integer no later than the previous record's round; an `absent`
status on a record with evidence is a record from before delta presentation, which rendered the full
list itself, so its own round is the anchor; `invalid` forces a full render.


## Planning scopes across an edited issue

A plan-first resume compares the normalized issue title/body digest
(`panel_evidence.plan_issue_text_digest`; comments never contribute) with the
digest the current plan was written against. Every plan-flow planner (coder)
record carries that digest as the optional `plan_issue_digest` field. It is not
a `scheduler_*` field, so a planner record is never read as a scheduler
checkpoint, and it is omitted when unset so legacy records stay byte-stable.
`plan_issue_scope.recorded_plan_issue_digest` prefers the newest in-scope
planner digest, then the newest in-scope `scheduler_issue_digest`, then the
admitted boundary's current digest. Legacy history with none of these resumes
as before and logs that an edit cannot be detected.

When the digests differ for a plan that is not completely approved, plan
startup (`plan_issue_scope.plan_scope_decision`, shared with the issue floor
preflight) logs both digests and the archived round R. By default it then:

1. runs the read-only plan-board floor check at the new scope start R+1, so a
   refused board publishes nothing;
2. posts one orchestrator-only **scope-boundary record**
   (`AGENT_PLAN_SCOPE_BOUNDARY`, a reserved marker agent output cannot emit),
   naming both digests, the archived round and plan subject, and
   `scope_start_round = R + 1`;
3. starts a fresh plan at round R+1 with no prior plan and no carried items.
   The planning round budget restarts there: the limit is
   `scope start + --max-rounds - 1` for every budget exit.

Round numbers stay monotonic, so board lineage, freshness and primary epochs
keep their meaning. Old plan comments are never edited or deleted.
`--plan-resume-keep-on-issue-edit` (per invocation, never inherited by child
planning, isolated providers or rendered recovery commands) keeps today's
resume instead. An approved plan is never reopened, nor is one the issue's
handoff record already hands to implementation (`plan_issue_scope.plan_handed_off`),
so a signed child-plan supersession re-plans it in place. The issue floor
preflight and plan startup call the same decision function, so they always
agree on the scope.

**Admission.** `plan_board_state.admitted_issue_comments` reads REST author
identities and the authenticated actor whenever a boundary or a plan
board-change link is present. A boundary is honoured only when its author is
that actor; a lookalike by anyone else is ignored and logged, and retires
nothing. Callers admit through `admitted_issue_comments` and pass the returned
actor; `resolve_plan_scope` no longer acquires history itself and takes no
runner (#1987). Issue recovery admits only when a boundary-shaped record is
present, so a history without one makes no network call, and an incomplete
read fails closed in that admission. A boundary that reaches
`resolve_plan_scope` without an authenticated actor fails closed rather than
resolving to scope 1.

**Two inputs.** Reviewer-board lineage (root anchor, binding initialization,
links, signed amendments) is always resolved from the full admitted history.
Planning candidates, the obligation ledger, panel evidence, the stall streak,
step-back state, force-full and calls-avoided replay, growth revision counts,
validation-diagnostic recovery and the planner/reviewer prompt context read
the current scope. `scope_plan_comments` blanks earlier-scope plan round
records and validation diagnostics in place (indices stay comparable with the
lineage); the prompt view drops them, keeping human comments, signed
requirements and the boundary audit text. Every shared reader derives the
scope itself when no explicit scope is given: `plan_verification_inputs`,
`classify_plan_approval`, `require_recoverable_plan_approval`,
`plan_resume_round` and issue-loop comment-only recovery, so managed-CI
recovery and strict qualification never report an archived item as an
unreconstructible obligation. `_resume_plan_round` takes `min_round_number`.

The approval classification has a distinct **`empty-scope`** state: a scope
start above 1 with no in-scope plan candidate. Consumers treat it like
`no-plan`, except the issue floor preflight, which floor-checks the plan board
at the scope start instead of deriving the archived plan's handoff board. A
restart after the boundary but before the first fresh planner record reopens
the same scope without a second boundary and skips board-anchor re-entry.

## Decisions

[Decision log](decisions/plan-context-and-presentation.md)
