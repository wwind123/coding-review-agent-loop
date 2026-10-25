# Findings and convergence

How findings are carried and closed, and how the PR loop stops when it does not converge: stable finding IDs and dispositions, reviewer scope discipline and future follow-ups, human-only exact-head evidence and risk-matrix row conversion, conjunctive sub-items, progress publication, the PR step-back, sweep and sibling escalation, the non-convergence stop, the evidence-only stall stop and evidence conversion.

Source paths: `src/coding_review_agent_loop/unresolved_items.py`, `src/coding_review_agent_loop/review_step_back.py`, `src/coding_review_agent_loop/evidence_stall.py`, `src/coding_review_agent_loop/evidence_conversion.py`, `src/coding_review_agent_loop/pr_loop.py`, `src/coding_review_agent_loop/pr_loop_support.py`, `src/coding_review_agent_loop/scope_declarations.py`, `src/coding_review_agent_loop/protocol.py`, `tests/test_review_step_back.py`, `tests/test_evidence_stall.py`, `tests/test_evidence_conversion.py`, `tests/test_pr_loop_review_ledger.py`, `tests/test_pr_loop_convergence.py`, `tests/test_pr_loop_followups.py`, `tests/test_pr_loop_evidence_freeze.py`.

## Findings, evidence and convergence

Findings have stable IDs, provenance, dispositions, and, where applicable,
resolution ownership. The coder reports addressed, remaining, and disputed
items; it does not silently redefine a finding or resolve another participant's
obligation. Human requirements and approved plans are separate from that ledger.
Original requirements, valid later human instructions, and safety constraints
outrank the plan; plan conflicts must be surfaced explicitly.

Reviewer scope discipline. Plan and PR reviewer prompts limit blocking
findings to the issue's stated requirements or signed human requirements,
regressions in touched behaviour, and introduced security or data-loss risk;
other hardening is routed to future follow-ups. Future follow-ups and `future`
prior-item dispositions are accepted on blocking as well as approved plan and
PR reviews (`protocol._finalize_parsed_review` and
`_finalize_parsed_plan_review` keep only the approved-state invariants). They
are recorded under the same follow-up-mode gate as approved-review items and
are never coder or planner work: the general coder handoff and its validator
exclude ledger `future` items, and compact PR review validation does not
require dispositions for the future items the compact prompt hides. In compact
PR context a carried item downgraded to `future` leaves the ledger, so
`pr_loop` keeps a publication-only pending set. It is seeded at loop start or
resume from `round_state.canonical_deferred_history_items`, a replay of the
posted round records that tracks deferral eligibility separately from the
legacy `canonical_history_item_outcomes` map. After each reconciliation it
drops IDs the carried ledger owned and adds the round's newly deferred items.
The set feeds only approval-time publication, adds no persisted fields, and the
publish-once audit record keeps filing to exactly once. Operator-declared
`## Out of scope` / `## Non-goals` sections in the issue body and signed human
comments are extracted by the pure `scope_declarations.py` helper and rendered
as a deterministic, bounded binding-scope block next to the issue context in
planner and reviewer prompts, so compact stable prefixes stay byte-identical;
compliance is enforced by the prompt, not by a deterministic classifier.

Human-only exact-head evidence is a separate channel from findings.
Reviewers list it in `exact_head_evidence_requests`, and the ledger records it
as a `human-exact-head-evidence` machine obligation. The obligation is owned by
the requesting reviewer and never goes to the coder or the review scheduler. It
stays deferred across heads while any finding or machine gate remains. Once
everything else is clean at a head H, the finalization gate re-reads the live
head and signed input and posts one freeze comment at H. That comment is a
review-and-feedback decision boundary (exit code 4): no coder, qualification,
or merge runs while it stands. A signed human response re-runs the full board
at H inside the same round, so it spends no round budget. Only the requester
can clear its evidence there. New findings or a failing machine gate release
the freeze with one release comment, and an external push breaks it.

Risk-matrix rows whose evidence cannot be tool-captured are converted into the
same obligation kind, keyed by (row, approved matrix identity) and owned
by the reviewers whose evidence-only finding named the row (or the whole board).
Conversion happens automatically after N consecutive capture-limited coder
attempts, evaluated from the persisted per-attempt classification and the review
snapshots' `bound_attempt_round` before the evidence-only stall window, or on
operator request (`--pr-convert-evidence-rows`) before any agent runs. One
`evidence-conversion` summary record persists the converted ledger and an audit;
the complete board then re-reviews the same head fresh inside the same round.
Owner clearance at the frozen head after new signed input mints a head-bound
signed-evidence reference carried cumulatively on every terminal freeze or
release record; an orchestrator-only overlay turns it into an `operator-signed`
verified row. Cleared converted obligations remain as tombstones that live-head
reconciliation (round start and the finalization gate) reopens with the same item
ID at any other head, unless tool-captured evidence bound to that head verifies
the row.

Conjunctive findings carry sub-items. A PR-review blocking or same-PR
finding may declare an ordered `sub_items` array of two to twelve distinct
statements; a single-obligation finding is unchanged. The orchestrator mints
the IDs `item-N.s1`..`item-N.sK` on every minting path, including the
reviewer-repair path, and each carried item persists every sub-item's text,
status and closing round in round metadata. The persisted fields are optional
and emitted only when present, so legacy records, dispositions and the
prior-item ledger signature are byte-identical. A malformed persisted payload
decodes to no sub-items and never clears or weakens the item. Malformed
sub-items in a reviewer response degrade to a plain finding with a recorded
degradation instead of rejecting the envelope, and a plan-review
finding's well-formed `sub_items` are flattened into its text with no ledger
structure.

Sub-item status is reviewer-authoritative. A reviewer disposition may carry
`sub_item_dispositions`, mapping a sub-item ID to `resolved` or `unresolved`;
only reviewers whose entries count toward the item under the current mode
(every disposing reviewer in aggregate mode, owners plus reviewers added as
owners this round in owner-scoped mode) can close or reopen a sub-item, and
machine obligations never accept them. Coder `addressed_sub_items` are
advisory claims that never change persisted status: they are validated against
the open sub-items of `remaining_items` parents, persisted with the coder
round, and shown to the next reviewer as unverified. An item is resolved when
all of its sub-items are. The item-level outcome logic is unchanged and runs
on effective dispositions: a `blocking`/`same-pr` entry that resolves every
remaining sub-item (a completing entry) is exempt from the actionable-note rule
and derives to `resolved`. A post-outcome invariant defers the completing
closure whenever the item would otherwise stay open with every sub-item
resolved, the outcome is recomputed from the literal dispositions so no owner
is cleared on its strength, and a note-less completing entry that does not take
effect always leaves a synthesized explanation naming the open sub-item and
why. The literal item-level `resolved` shortcut changes sub-item status only
when the item actually clears. Reconciliation is a deterministic function of
the ledger and the persisted dispositions, so a resumed round applies each
closure exactly once.

Progress is published three ways: a labelled per-review projection on each
reviewer comment, carried `k/K` counts (reviewer-confirmed and coder-claimed
counts labelled separately) in the coder follow-up comment, and one
orchestrator progress comment per round, posted right after reconciliation
and before any coder turn, approval or budget exit. That comment lists cleared
items and newly stalled findings and carries an `AGENT_SUB_ITEM_PROGRESS`
audit record keyed by PR, round and digest; only records in comments written by
the authenticated actor suppress a repost, which keeps it exactly-once across
resume. A pure helper classifies each item as `new`, `converging`, `stalled`,
`complete` or `cleared` from persisted rounds. `--sub-item-stall-rounds`
(default 3, 0 disables) sets the stall window, and the max-rounds diagnostics
add a per-item progress block. The signal is advisory: it never changes
`--max-rounds`, `allowed_rounds` or a checkpoint bound.

**PR step-back, reviewer sweep, and sibling escalation** (`--pr-step-back-rounds` K, default 3, `0` disables; `--pr-step-back-line-window`,
default 40). The PR fix loop reuses the plan-phase classification
(`classify_review(..., phase="pr")`: `blocking` and `same-pr` items the reviewer
owns are mandatory, future follow-ups never count). Each reviewer is tracked
independently, or only the primary under `primary-then-panel`. Findings carry
`path:start[-end]` references; `project_finding_locations` reads them from the
parent text, the `fix_scope` entries, and every open sub-item statement (the
sub-items are stored apart from the parent) while keeping the parent identity,
and every location is bound to the head its review was taken on. Locations are
compared across heads by `map_anchor`, which pairs renames with
`git diff --name-status -M` and no pathspec, then reads `git diff -U0` hunks for
the old and new paths through the runner. It returns exactly one of three
outcomes: `SHIFTED` (the range lies in unchanged lines and moves by the
cumulative hunk offset), `REWRITTEN` (changed hunks overlap it, their old side
stays within the anchor +/- the window, and the replacement span is the new
position; ranges are never widened to a whole hunk), or `UNMAPPABLE` (a deleted
file with no rename pair, a rewrite that spills beyond the window, a pure
deletion or move, or a failed diff). At the coder dispatch, after the round's
review is read back from fresh history, a tracked reviewer with K consecutive
`new-finding` reviews (and no active episode) forms a cluster when one path has a
mapped finding in every one of those K reviews and the mapped ranges connect with
gaps no larger than the window; unrelated paths, repeat-only reviews, findings
without a line reference, and `UNMAPPABLE` locations never contribute. All
reviewers that trigger at the same dispatch share one step-back coder turn
(`step_back_context` replaces the same-PR small-cleanup framing and offers two
choices: a single rule, a parametrized test, and a summary beginning
`Generalization:`; or, when the approved approach itself generates the class, a
summary beginning `Approach alternative:` that names a materially smaller
alternative without redesigning beyond the plan; the existing `coder_followup`
JSON carries it, no new section), and the
orchestrator records one reviewer-owned `step_back_entries` entry per reviewer
(`phase: "pr"`, `trigger_round`, `trigger_head`, `anchor`) on the coder record. The
sweep binds to that record's entry reviewers and its resulting head only: the
review of that head receives `sweep_context` (full and compact prompts), asking
for every remaining instance, in the existing sub-item shape (none, one plain
finding, one finding with 2-12 sub-items, or several findings of at most 12). A
live run carries the entries across the iteration; a resumed run rebuilds them
from the durable record and never re-requests a published sweep review. One
predicate, `in_cluster`, decides membership for clustering, escalation, and
clearance: for a mappable anchor, the mapped path within the mapped span +/- the
window; for an `UNMAPPABLE` anchor, the logged conservative fallback of any
finding on the original or rename-destination path. If the sweep review, or any
later review in the active episode, introduces a new mandatory finding that is a
member (located in the parent text or only in a sub-item), the loop raises
`HumanDecisionRequiredError` (exit code 4) before any further coder invocation, naming the cluster, the
mapping outcome, the siblings, and the coder's `Generalization:` excerpt rebuilt
from the stored structured response. A carried item is not a sibling; it keeps the
episode open while it is a member. The episode closes on the reviewer's approval
or when no unresolved mandatory item of theirs is a member. A malformed entry
suppresses the trigger and the stop with a log line.

An `Approach alternative:` answer (#1628) stops instead of continuing:
`pr_step_back_approach_alternative` reads the newest valid step-back coder record
and matches only when its decoded summary opens with that prefix, its resulting
head equals the live head, and no review of that head was recorded after it. The
live check runs after the coder record is posted and after the issue-created
managed-CI continuity grant for the new head, so a resume activates managed CI
before stopping; the same check runs at every round start before any review, so
a resumed run stops before reviewing the proposal head. The loop raises
`HumanDecisionRequiredError` (exit code 4) quoting the alternative and offering a
scope change, a stated limitation, or continued generalization with
`--pr-step-back-rounds 0`, which also disables this stop; history export classifies the
message as a human-decision stop. A pushed head, a
`Generalization:` answer, or an unreadable record never stops. The sibling
escalation message also names the scope-change route (approve the scope change,
push the replacement and rerun).

**PR non-convergence stop** (`--pr-non-convergence-rounds` K, default 3,
`0` disables, independent of `--pr-step-back-rounds`). The clustered trigger counts
only new findings near one file span, so items that persist across files never
fire it. `review_step_back.find_pr_non_convergence` therefore joins a tracked
reviewer's reviews at rounds N-K..N with the coder records between them using the
loop's numbering (review r, then coder record r+1 whose subject is the head
reviewed at r+1). Every join must have moved the head and the next review must be
on that head, so a no-op turn, a missing record, or an external push breaks the
chain. Each review's blocking set is `_remaining_mandatory_items` with only the
reviewer's own dispositions, which drops machine and human exact-head evidence
obligations and items whose last sub-item a completing disposition resolved;
tagged findings count like any finding; an unresolvable
`--review-parallel` publication breaks the chain. An item persists when it is in
all K+1 blocking sets and its open sub-item count never drops between two of them
(resolving a sub-item is progress). Any persisting item raises `HumanDecisionRequiredError` (exit 4) inside
`_pr_step_back_decision`, after a clustered-sibling escalation and before a fresh
clustered step-back turn, the round-limit exit, and any coder invocation. When
`defers_to_evidence_stall` holds (every persisting item is tagged with rows
unsatisfied at the dispatch head, read from the head-bound coder record as the
stall reads them), the decision is returned as `deferred_stop` instead: the PR
loop raises it right after `_pr_evidence_stall_decision`, so a stall that
qualifies on the combined ledger and its window history wins, and otherwise the
non-convergence stop fires before the coder; on the final round or a
merge-conflict round, where the stall is not evaluated, it is raised before the
round-limit exit. The
snapshot pre-check only needs the reviews N-K..N-1 to be contiguous blocks, because
the live snapshot predates coder record N; the joins are verified on the fresh
read. The decision text lists each item with its attempt count and first line, the
`git diff --shortstat` growth since the head of the earliest counted attempt (the
attempt scan may reach past the K window) and the PR size from the merge
base (each `unavailable` when Git fails), flags items asking for evidence, and
offers the override options. Nothing is persisted, so a resumed run stops with the
same text.

**Unchanged-head stop and evidence progress.** `_UnchangedHeadTracker` counts
consecutive PR follow-ups that leave one head unchanged and stops at
`MAX_UNCHANGED_HEAD_CODER_TURNS` (#985). It has one bounded exception
(#1626). Each tracked head gets a single extra review round when the follow-up
at the limit made cited, authoritative evidence progress, meaning a receipt
that passes three checks:
- it is in the coder acquisition's frozen catalog
  (`acquisition_test_observations`, captured before any repair launch can
  repoint the runner's latest test turn);
- the live, unbounded reconciliation retained by
  `Runner.render_local_test_evidence` names it as `superseded_by` on a
  parent-observed failure (`superseded_by_current_turn`);
- a surviving `test_observations` citation of the accepted follow-up cites it.

A failed citation repair therefore never earns the grant. A different head
resets it, and like the count it is in memory only. The bounded round evidence
projection stays display-only; no gate or count reads it. When the stop fires
after the last follow-up's `test_observations` were dropped as a non-array,
the message names that, says whether repair was attempted, and gives the
number of uncited receipts from the frozen catalog.

**PR evidence-only stall stop** (`--pr-evidence-stall-rounds` K, default 2,
`0` disables). A `pr_review` blocking item may carry `evidence_row_ids`, parsed in
`protocol.py` (blocking items only, bounded unique strings), validated against the
enforceable rows of the approved matrix inside the review validation and repair loop
(`unresolved_items._validate_review_response`; the repair preserves valid tags and
invents none), and fixed on the stable `UnresolvedReviewItem` when it is minted.
`evidence_stall.py` is pure: `unsatisfied_rows` derives `row_id -> explanation` from the
persisted `risk_test_matrix_evidence` and diagnostics of the coder record whose subject
is the current head, requiring the approved matrix identity (`canonical-status:<status>`
plus the row's own sorted diagnostic codes; absent or other-identity evidence yields
`evidence-absent` or `evidence-matrix-mismatch`, never "all rows unverified");
`classify_round` returns a `StallRoundSnapshot` whose closed reason set disqualifies a
round (untagged finding, tag naming a satisfied row, non-evidence machine obligation,
failing or stalled checks, live freeze, unbound or mismatched evidence); `stall_window`
walks back over consecutive qualifying snapshots with identical row sets and matrix
identity. `pr_loop_support._pr_evidence_stall_decision` runs on the ordinary dispatch
path in `pr_loop.py`, after the existing check-board fetch and CI-failure
reconciliation and before any coder invocation; it is skipped on a merge-conflict
round and adds no check fetch. It reads prior snapshots from freshly fetched round
history and raises `HumanDecisionRequiredError` (exit 4) when the window reaches K.
Otherwise it returns the snapshot, which `PostedRoundMetadata.evidence_stall` persists on
the PR coder follow-up record (omitted when absent). The coder record for loop round r
is numbered r+1, so the snapshot carries an explicit `review_round`; the decoder marks it
invalid unless it equals the record number minus one and its content is consistent
(`qualifies` exactly when `reasons` is empty, reasons from the closed set, unique row IDs,
and a qualifying snapshot names its head, identity, and rows). An invalid or missing
snapshot ends the window walk. History that cannot be decoded suppresses the stop with a log
line. State is orchestrator-owned, so a resumed run recomputes the same decision.

**Evidence conversion** (`--pr-evidence-capture-attempts` N, default 2, `0`
disables; `--pr-convert-evidence-rows`). `protocol.derive_risk_test_matrix_evidence`
records `row_capture_classifications` in the loop that emits diagnostics, and both
coder-record writers persist it as `risk_test_matrix_capture_classification` (bound to
the record head and evidence identity; malformed or mis-bound decodes as no
classification). Stall snapshots gain `bound_attempt_round`. `_pr_evidence_stall_decision`
first computes the current snapshot, then `evidence_conversion.auto_conversion_rows`,
and only without a conversion evaluates the stall window. A conversion mints
row-keyed obligations (`unresolved_items._upsert_converted_evidence_obligation`),
applies the residual-tag transition (`converted_row_ids`, terminal `converted` status),
posts one `evidence-conversion` record (audit, `conversion_stall_snapshot`,
`conversion_review_round`, cumulative references), and `pr_loop.py` repeats the round
at the same head with `conversion_pass`, which forces the complete board and blocks
carried approvals; approvals older than the latest conversion at the head are dropped.
The admitted conversion record is also a same-round visibility boundary (#1664),
resolved separately in each analysed history (base snapshot, post-checkpoint fresh read,
REST recovery history) by `pr_loop_support.conversion_visibility_boundary` from admitted
records only, and matched across histories by comment identity, never by index.
Publications before it belong to the pre-conversion pass, which the repeat reads through
the converted ledger: they are not same-round peers for the partial-publish guard
(`round_visibility.same_round_public_publications` takes the boundary and combines it with
the coder boundary by max), are not resumed as completed repeat reviews, and are never
recovery deletion targets, orphan-attachment attributions or verification blockers in
`partial_round_recovery`. Isolation still holds within the repeat: a repeat-pass
publication is a peer, so a reviewer still needing a fresh turn after it is refused.
A foreign-authored or audit-rejected record yields no boundary. Staging a repeat discards
the round's review spool, so a pre-conversion withheld outcome is never replayed, and an
operator conversion refreshes the round snapshot (or, if that read fails, the visibility
fallback carries the process's own conversion record). An unfinished admitted operator
conversion (`conversion_review_round` unset) is recognized by its round on resume, so it
restores the full-board repeat just as an auto conversion does.
The evidence-response pass mints `SignedEvidenceReference`s for cleared converted rows
(`mint_signed_evidence_references`), keeps a durable tombstone, and threads the
cumulative set onto every terminal record. `reconcile_converted_obligations` runs after
the round-start ledger assignment and inside `_evidence_freeze_gate`.

## Decisions

[Decision log](decisions/findings-and-convergence.md)
