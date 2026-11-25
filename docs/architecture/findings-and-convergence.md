# Findings and convergence

How findings are carried and closed, and how the PR loop stops when it does not converge: stable finding IDs and dispositions, reviewer scope discipline, future follow-ups and plan-review implementation notes, human-only exact-head evidence and risk-matrix row conversion, conjunctive sub-items, progress publication, the single PR no-progress stop, the PR step-back prompt and sweep, the evidence-only stall stop and evidence conversion.

Source paths: `src/coding_review_agent_loop/pr_head_sync.py`, `src/coding_review_agent_loop/converted_capture.py`, `src/coding_review_agent_loop/pr_restart.py`, `src/coding_review_agent_loop/unresolved_items.py`, `src/coding_review_agent_loop/dispute_release.py`, `src/coding_review_agent_loop/native_evidence.py`, `src/coding_review_agent_loop/human_evidence_mode.py`, `src/coding_review_agent_loop/review_step_back.py`, `src/coding_review_agent_loop/no_progress.py`, `src/coding_review_agent_loop/evidence_stall.py`, `src/coding_review_agent_loop/evidence_conversion.py`, `src/coding_review_agent_loop/pr_loop.py`, `src/coding_review_agent_loop/pr_loop_support.py`, `src/coding_review_agent_loop/scope_declarations.py`, `src/coding_review_agent_loop/protocol.py`, `src/coding_review_agent_loop/plan_implementation_notes.py`, `tests/test_review_step_back.py`, `tests/test_no_progress.py`, `tests/test_evidence_stall.py`, `tests/test_evidence_conversion.py`, `tests/test_pr_loop_review_ledger.py`, `tests/test_dispute_release.py`, `tests/test_pr_loop_convergence.py`, `tests/test_pr_loop_followups.py`, `tests/test_pr_loop_evidence_freeze.py`, `tests/test_human_evidence_mode.py`, `tests/test_pr_loop_human_evidence_mode.py`, `tests/test_plan_implementation_notes.py`, `tests/test_plan_implementation_notes_inventory.py`.

## Findings, evidence and convergence

Findings have stable IDs, provenance, dispositions, and, where applicable,
resolution ownership. The coder reports addressed, remaining, and disputed
items; it does not silently redefine a finding or resolve another participant's
obligation. Human requirements and approved plans are separate from that ledger.
Visual findings always use owner-scoped reconciliation, including sub-items and
restart replay, independently of the ordinary code-review policy. A verified
visual seat's resolved or future disposition clears only its own share; the
finding stays active while another resolution owner is pending.
Original requirements, valid later human instructions, and safety constraints
outrank the plan; plan conflicts must be surfaced explicitly.

Coder classification normalization uses trusted context from the complete current
ledger, separately from the dispatch-selected required items. Evidence obligations,
retained future work and inert provenance records listed as addressed or remaining
are dropped with internal diagnostics, including their notes and advisory sub-item
claims. Duplicates are checked first; unknown IDs and excluded disputes still fail.
This does not clear evidence, resolve findings or change future publication eligibility.
Reviewer prose can name records omitted from the rendered item list, so prompt
filtering alone cannot enforce this boundary.

Disputed items and their release (#2025). A coder dispute that the reviewer
keeps `blocking` or `same-pr` after one reconsideration stops the PR loop with
`HumanDecisionRequiredError` before any coder, CI, qualification or merge.
`dispute_release.py` observes, once per round after `round_start` and from the
freshly read PR history, whether a human has answered. While the carried
ledger holds a disputed item, the next round refetches the admitted PR context
instead of reusing the pre-publication snapshot, so requirements and dispute
times come from one fresh observation. The dispute time is the
`created_at` of the newest admitted coder follow-up round record that lists the
item in `disputed_items`. A dispute is released when the loop's existing
effective signed-requirement set contains a requirement whose `created_at` is
strictly later; no new authority path is added. An unknown dispute time (no
admitted record lists the item, or this invocation's newer coder publication
is not yet in the observed snapshot), a tie, or an unparseable requirement
timestamp never releases. Any newer requirement releases every older dispute:
release only routes the item back to the coder and reviewers, never approves or
merges, and a later re-dispute re-arms the stop because its newer record moves
the dispute time. Nothing about a release is persisted or cached. Coder,
same-PR, merge-conflict and reviewer prompts carry one pointer line per
released item naming the newer `hr-` IDs; requirement text stays only in the
signed-requirements section. Because a released item no longer triggers the
stop, a conflicting PR reaches merge-conflict routing; a held dispute still
stops first. The stop notice lists only item IDs, dispute times and recovery
steps, never item text or dispute evidence, and is posted best-effort: a
posting failure is logged and the original `HumanDecisionRequiredError` is
raised.

Validation and repair share the same required IDs and optional allowance with
ledger-proven exclusions removed, including an explicitly empty required set.
Repair's complete-valid-set instruction and preservation exemptions permit removal
of excluded classifications, notes and dispute evidence. Generic optional claims
without ledger exclusion proof retain their existing protection. Diagnostics are
internal and unparsed; no persisted schema changes. Carried ledger records, rather
than raw coder classifications retained for audit, own lifecycle and replay state.

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

Plan-review implementation notes (#2145). A plan review may also return
`implementation_notes`: in-scope findings that need no change to the plan's
design, steps or acceptance criteria and that building the change or running
its tests would surface anyway (exact-inventory test updates, pinned fixture
sets, snapshot or golden-file refreshes). When in doubt a finding still
blocks, and a qualifying item placed in `same_plan_followups` stays valid.
`protocol.parse_structured_plan_review` carries notes on
`ParsedPlanReview.implementation_notes`, never on `PlanReviewItems`, so the
unresolved-item ledger, dispositions, scheduling, no-progress and step-back
logic never see them. Notes never block, are allowed in approved and blocking
reviews, never become planner revision input, and are never filed as issues.
Each review is bounded deterministically (10 notes of at most 1000
characters; a note duplicating the same review's blocking or same-plan item
is dropped) with recorded degradations instead of a rejection, and repair
preservation keeps every note while letting a blocking duplicate win. Notes
persist on the reviewer's plan round record (`implementation_notes`, omitted
when empty so legacy encodings stay byte-identical, spillable to sidecars).
`plan_implementation_notes.py` collects them at approval from settled plan
reviewer records (authoritative records, or a parallel round's publication
checkpoints followed by its reconciliation summary, minus failed seats) in
the approved plan scope: from the newest authenticated scope boundary at or
before the approving round through that round. They are de-duplicated with
merged attribution, approving round first, and bounded to 20 notes and 6000
characters with an omitted count. If the plan history cannot be admitted or
the trusted actor is unknown, the notes are reported unavailable with a
reason, which never blocks or fails the run. The collected notes are listed in
the plan approval comment. When the announcement for identical plan text
belongs to an archived scope, one notes-only supplement for the current scope
is posted instead; it carries no approval record, files nothing, and is
skipped when an identical orchestrator comment already follows the boundary.
The notes ride on `ApprovedPlanContext` fields excluded from equality, so the
plan hash, subject, matrix identity and handoff validation are unchanged, and
`format_approved_plan_context` renders them in a fixed-bounded, non-binding
section for the implementer and the PR reviewers; an unaddressed note is an
ordinary PR finding. Every PR-facing construction or recovery site attaches
them through one helper (fresh approval, the handoff and parent fallbacks in
`run_pr_loop`, managed-CI recoveries, the qualification replacement and the
canonical no-handoff plan), guarded by an inventory test. A direct-execution
staged child inherits the parent plan's notes labeled with their origin;
planning-route children see only their own scope's notes, and notes are never
written into child issues or handoff records.

An approving PR reviewer whose review lacks the signed-requirements
acknowledgement is never reformatted by a repair model (#2066): the review
upserts the dedicated human-requirements acknowledgement machine obligation and immediately persists
its ledger in an Orchestrator summary with an explicit failure-boundary phase.
Same-head restart retains the round and full ledger and reruns reviewers against
that ledger rather than treating the summary as reconciliation. Repeated failures replace that obligation
without consuming an ordinary item number. Existing valid coder acknowledgement
can clear it after follow-up or at resumed round start. Complete required-board
acknowledgement also clears it; partial acknowledgement and ordinary reviewer
resolved dispositions do not. The fresh reviewer acknowledgement gate still
runs before qualification: an approving reviewer missing acknowledgement
re-creates the dedicated blocker even after valid coder clearance. Unrelated findings and machine gates retain their own lifecycle.
The obligation is minted and kept only for the projected acknowledgement scope
described under coder-disposition scope below (#2083): when that set is empty,
for example under freeze with only human-only sources left, an approval without
the marker mints nothing, a carried obligation is cleared in that round, and the
human-only evidence stays the deferred final barrier. Although its lifecycle is
`repair_required`, a new head never clears it, so its stop message names
current-head reviewer acknowledgement or coder `human_requirement_dispositions`
instead of asking for a strictly different head.
An issue implementation that reports a PR beside blocked surfaced signed
requirements is a valid handoff, not a terminal conflict (#2120). Round-start
reconciliation of that implementation carrier upserts the same acknowledgement
obligation, listing each blocked surfaced requirement with the coder's reason
(sanitized and bounded per entry and in total), so reviewers and the operator
see it in the first review round. Decisions use only the surfaced tuple that
`validate_human_requirement_dispositions` returns, so unsurfaced entries kept
by historical decode never mint or appear in it. Clearing and merge behaviour
is the #2083 path above; PR-loop coder follow-ups that report `blocked` still
clear it as before.
A coder that lists the tool-owned acknowledgement record in its addressed,
remaining or disputed items has it stripped by the coder follow-up parser, with
one accepted-response log line; the listing never clears or blocks the
obligation, which stays decided by the signed acknowledgement fields. A `pr_review` carrying the coder-only
`human_requirement_dispositions` field has it dropped (and logged) by
`parse_structured_pr_review` before the exact-key check; other unknown fields
remain fatal (#2026).
A reviewer seat that fails in a round where another seat blocked no longer
stops the round; its carried items are frozen against every healthy seat's
disposition before settlement (see
[Failed reviewer seats in a blocked round](review-scheduling.md#failed-reviewer-seats-in-a-blocked-round)).

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

Native obligations also admit pre-supplied evidence at the final gate. A
same-head requester resolution cites `evidence_requirement_ids` from that
reviewer's surfaced signed input and attests adequate acceptance for H. The gate
reads this round's reviewer records plus the same-head approvals the round's
scheduler counted as qualifying (carried through the unchanged-head skip or
paused as already approved, #2019). A carried record counts only when the gate's
authenticated read holds an identical record (metadata and body) at its index,
and the gate evaluates that admitted copy. The requester's newest admitted
complete review at H from any round governs, so a later blocking or
non-resolving review (including one from an intervening round) overrides an
older resolution; that newest review attests only when it is from this round or
was carried. A review at another head never clears.
The gate uses the live effective signed objects, checks content-bound IDs and
unambiguous complete 7–40 character head references against current/admitted PR
head history, and publishes a single evidence-cleared release without a freeze.
Changed/new input refreshes the same round; missing input cannot authorize
clearance. Partial acceptance rides on the remaining freeze. A removed requester
requires every current reviewer. Head changes invalidate the acceptance.

`native_evidence.py` shares conservative matching and alias logic with live
request upserts and reviewer-record replay. Only recognized commit tokens are
normalized for contextual equivalence; arbitrary hex and complete digests remain
significant. Valid affirmative consolidation clauses may name only the owner's
open native items. The earliest persisted identity and item ID survive, with
optional historical aliases for disposition routing and head-scoped clearance.
Conflicting alias votes cannot clear an obligation. Legacy records default to
empty citation/alias fields. Replay uses admitted heads available at each ingestion
boundary, never later records retroactively. Push recovery replays requester
requests after each round's snapshots to retain consolidation text and aliases
even when the same response clears the survivor before another snapshot. Recovery
uses the authenticated subset for both request head recognition and ledger
consolidation; retained foreign ordinary records cannot introduce known heads.
Frozen-response ingestion retains pre-clearance targets to validate consolidation
reemissions, suppressing them only when every native target belongs to the
requester and has an authorized clearance at the current head.
Converted rows and screenshot requests remain separate channels.

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
`complete` or `cleared` from persisted rounds. `--no-progress-rounds`
(default 3, minimum 1) sets the stall window (the retired
`--sub-item-stall-rounds` is ignored), and the max-rounds diagnostics
add a per-item progress block. The signal is advisory: it never changes
`--max-rounds`, `allowed_rounds` or a checkpoint bound.

**PR no-progress stop** (`--no-progress-rounds` K, default 3, minimum 1; #2097,
stage 1 of #2067). One pure, record-derived detector in `no_progress.py`
replaces the PR step-back sibling escalation, the non-convergence stop, the
approach-alternative stop and the in-memory unchanged-head coder counter.
`no_progress_decision(observations, live_subject=..., live_blocking_ids=...,
k=..., through_round=...)` fires only when the last K observations have
contiguous round numbers ending at `through_round`, the same non-empty subject
and the same non-empty blocking ID set, and the live candidate equals that
repeated observation; the live candidate is compared, never appended, so an
unreviewed head never counts as a completed round. `pr_no_progress_observations`
derives one observation per completed round N from the newest PR coder record
numbered N+1, which every review mode (including the default sequential
all-reviewers mode, which writes no reconciliation summary) already persists:
subject = its `followup_dispatch_head`, or for a historical record without one
the single subject of round N's complete reviewer records; IDs = its
`prior_items` filtered by `is_blocking_ledger_item` (`blocking` and `same-pr`
findings; never future items, inert converted records or tombstones, or machine
and human exact-head obligations). Round N is no observation, and breaks the
window, when it has a closure boundary or an unvalidated closure summary
(`is_round_closure_boundary` / `is_unvalidated_closure_summary`, covering
closed-incomplete records and legacy signed-input supersession boundaries),
when `closed_round_reviewer_indexes` leaves it no complete review, when no coder
record N+1 exists, or when that record's dispatch head differs from the reviewed
head. No producer change to round metadata was needed.

The PR loop checks it at two sites with the same functions. At the coder
dispatch, `pr_loop_support._pr_dispatch_convergence` reads fresh history (the
live snapshot predates coder record N; a cheap snapshot pre-check needs only
complete reviews for the previous K-1 rounds) and adds the just-settled round
from the in-memory ledger as `current`; the live subject is the freshly
observed PR head. This runs before the round-limit exit and any coder
invocation. At round start, `_pr_round_start_no_progress` uses durable
observations only, through the previous round, against the observed head and
the in-memory ledger, before any checkpoint, architecture refresh or reviewer
prompt, so a resume after K completed rounds stops before another review while
a moved head is reviewed first. A stop raises `HumanDecisionRequiredError`
(exit 4) through `stop_for_human_decision` and writes no round metadata. When
`no_progress.defers_to_evidence_stall` holds (every repeated item is tagged with
rows unsatisfied at the dispatch head, read by `_evidence_owned_rows` from the
head-bound coder record as the stall reads them, and `--pr-evidence-stall-rounds`
is enabled), the dispatch returns the built error as
`PrStepBackDecision.deferred_stop`: the loop raises it right after
`_pr_evidence_stall_decision` when the stall did not stop, or before the
round-limit exit on the final or a merge-conflict round. Round start applies the
same predicate (`_no_progress_defers_to_evidence_stall`) to its durable records,
so a tag naming a satisfied row or a head without usable matrix evidence never
defers; only an evidence-owned window is left to the dispatch. A closure boundary
excludes its round number at both sites, including the just-settled `current`
round, so a replacement review that reuses the number never counts.

`render_no_progress_message` (registered in `HUMAN_DECISION_RENDERERS`) opens
with `NO_PROGRESS_PREFIX_TEXT` after the `PR #N: ` lead and lists the sorted IDs
(first 20, then `and N more`) with each item's first line. Every operator
route or diagnostic of a retired stop is kept as an optional section, emitted
only when present: the PR size from `_pr_size_measurement` (merge-base
`git diff --shortstat`, never raising; omitted when unavailable), the newest
PR coder record's `unsuperseded_receipts_sentence` and `tool_identity_clause`,
the non-array citation diagnosis read back from that record
(`durable_citation_diagnosis`: its persisted `test_observation_degradations`
and the tool-owned `uncited_acquisition_receipts` count the loop writes on the
coder record from the frozen original acquisition catalog, before evidence
bounding, so repair-turn receipts never count and evicted rows are never lost),
so a resumed stop renders the same section as the live one, the
newest `Approach alternative:` excerpt at the stop head
(`review_step_back.pr_approach_alternative_at_head`, selected even after later
reviews of that head), and the child-plan supersession route for a PR bound to
a planning child (formerly appended by the unchanged-head stop, #985). The
retired non-convergence diff-growth line is dropped on purpose: every window
round has the same head, so growth across it is always "no change". The
per-head evidence-progress grant (#1626) is retired with the counter: evidence
progress that makes a reviewer resolve an item changes the ID set and restarts
the window.

**Legacy step-back fields are inert.** `round_metadata` still decodes
`step_back_entries` and `step_back_status` (absent, valid or invalid) without
error, and PR coder records keep writing `step_back_entries` purely as an audit
record that history export's `step_backs` metric counts. No PR decision reads
them: not the detector, the step-back trigger or its suppression, the sweep, the
approach-alternative excerpt, or any degraded-history log. A static AST guard in
`tests/test_no_progress.py` forbids attribute loads of those fields and the
retired entry readers in `pr_loop.py`, `pr_loop_support.py` and
`no_progress.py` while allowing the writer keyword. The retired flags
`--pr-step-back-rounds`, `--pr-step-back-line-window`,
`--pr-non-convergence-rounds` and `--sub-item-stall-rounds` are accepted and
ignored for one release (`config.warn_deprecated_convergence_flags` logs one
notice); their config fields and cross-validation are removed.

**Plan no-progress stop** (same `--no-progress-rounds` K; #2098, stage 2 of
#2067). The plan loop reuses the same `no_progress_decision` and replaces the
plan step-back escalation stop (round start and post-review) and the
primary-phase stall stop. The subject is the plan subject (the canonical plan
hash); `blocking_item_ids(..., statuses=PLAN_BLOCKING_STATUSES)` counts
`blocking` and `same-plan` items (the PR default stays `blocking`/`same-pr`).
`plan_no_progress_observations` observes round N through the newest plan
planner record numbered N+1: subject = its `prior_plan_subject`, IDs = its
`prior_items`, the round-end ledger the planner was asked to fix. No scheduler
record is needed. Round N is no
observation, and breaks the window, when it has a closure boundary or an
unvalidated closure summary, no complete non-inert review before that planner
record, review subjects other than exactly `{prior_plan_subject}`, a planner
`plan_issue_digest` that differs from the live issue digest, or a round number
below the reset floor (the newest valid `scheduler_stall_reset` prelaunch
checkpoint's round, in any phase). A reviewer-only phase-advance round has no
planner record N+1, so it never observes. Requirement binding is conservative:
with a non-empty live surfaced planning-requirement set every review must carry a
`plan_candidate_key` whose `surfaced_requirement_id_digest` equals the live one
(keyless legacy rounds are excluded); with an empty live set a review whose key
records another digest is excluded. A requirement signed in a comment therefore
restarts the window even though the issue digest excludes comments.

`plan_first_loop` runs the check once per round, outside the staged scheduler
branch so it applies under every review policy, after the staged history is
classified and before the scheduler prelaunch checkpoint and every agent turn.
Under `primary-then-panel` it is skipped with a log line unless the history class
is intact; it is also skipped on the reset round, ownership, staged-mode,
force-full and approved-shortcut rounds, a resumed round holding completed
reviews, and an empty live blocking set.
The live candidate is the current plan subject plus the blocking IDs of the
pre-round ledger. The message (lead empty, subject label `plan`) adds the plan
size (characters plus the growth-assessment measurements) and the newest
step-back candidate's summary, replayed per primary segment so it survives the
panel opening, as the approach alternative, and
`stop_plan_pre_panel_decision` posts it once as a planning diagnostic and raises
`PlanPrePanelHumanDecisionError` (exit 4) without a round record. The plan step-back and sweep
stay prompts with internal thresholds (`PLAN_STEP_BACK_ROUNDS`,
`PLAN_SWEEP_ROUNDS`), and the step-back episode is replay-derived (see
[review scheduling](review-scheduling.md)). The four plan threshold flags
`--plan-step-back-rounds`, `--plan-step-back-escalation-rounds`,
`--plan-primary-stall-rounds` and `--plan-sweep-rounds` join
`DEPRECATED_CONVERGENCE_FLAG_DESTS`: accepted, ignored and named in the one
notice. History export labels the stop `plan-no-progress` and keeps `plan-step-back` and `plan-primary-stall` for old comments. The
legacy-field AST guard also covers the plan modules.

**PR step-back turn and reviewer sweep** (a prompt technique only). The PR fix
loop reuses the plan-phase classification (`classify_review(..., phase="pr")`:
`blocking` and `same-pr` items the reviewer owns are mandatory, future
follow-ups never count). Each reviewer is tracked independently, or only the
primary under `primary-then-panel`. Findings carry `path:start[-end]`
references; `project_finding_locations` reads them from the parent text, the
`fix_scope` entries, and every open sub-item statement while keeping the parent
identity, and every location is bound to the head its review was taken on.
Locations are compared across heads by `map_anchor`, which pairs renames with
`git diff --name-status -M` and no pathspec, then reads `git diff -U0` hunks for
the old and new paths through the runner. It returns exactly one of three
outcomes: `SHIFTED` (unchanged lines moved by the cumulative hunk offset),
`REWRITTEN` (changed hunks overlap the range within the window; the replacement
span is the new position, never a whole hunk), or `UNMAPPABLE` (a deleted file
with no rename pair, a rewrite that spills beyond the window, a pure deletion
or move, or a failed diff). `derived_pr_trigger_rounds` scans a reviewer's
per-head review records in round order: a trigger at round r needs
`PR_STEP_BACK_ROUNDS` (3) contiguous `new-finding` reviews after the previous
trigger, whose mapped locations cluster on one path with gaps no larger than
`PR_STEP_BACK_LINE_WINDOW` (40) under the `in_cluster`/`find_cluster` rules;
locations map to the head the coder follow-up was dispatched from
(`pr_dispatch_heads`, or the dispatch head for the current round), so a live
and a resumed run agree, and unmappable locations never contribute. A step-back
turn is requested at round r exactly when r is a derived trigger round; all
reviewers triggering at one dispatch share one coder turn (`step_back_context`
offers a `Generalization:` rule with a parametrized test, or an
`Approach alternative:` proposal that goes to review like any revision). The
review of the head that follows a trigger, and only it, receives
`sweep_context` for the triggering reviewer
(`_pr_step_back_sweep_contexts`, rebuilt from the trigger cluster; within a run
the previous dispatch head is carried because the round snapshot can predate
the newest coder record). A step-back never stops the run.

**Durable human-decision diagnostics** (#1640). The no-progress stop (at round
start, at the dispatch, or deferred) goes through `pr_loop`'s
`stop_for_human_decision` closure, which posts one plain
`PR review scheduling diagnostic (round N): ...` comment and re-raises the
original exception object, only where a stop actually raises, so a deferred stop
that the evidence stall owns posts nothing. It carries no round metadata and
resume never reads it; a rerun that stops again posts one more. The body is
`render_pr_human_decision_diagnostic`: the unchanged message within
`PR_HUMAN_DECISION_DIAGNOSTIC_LIMIT` (40,000 characters), otherwise the
template, the first line and as many following whole lines as fit, then an
omission note; the exception and CLI text stay complete. The post is
best-effort: an `AgentLoopError` or `OSError` is logged as a warning and never
changes the stop or exit 4. History export classifies the stop as
`pr-no-progress` and still recognizes the retired stops' recorded comments
(`pr-step-back-sibling`, `pr-non-convergence`,
`pr-step-back-approach-alternative`) from the prefix constants kept in
`review_step_back`; the legacy non-convergence match reads only the first line.
Evidence-stall and evidence-freeze coverage is described below;
disputed-items and finalization stops are unchanged.

**Uniform human-decision stops** (#2069). Every human-decision stop raises
`HumanDecisionRequiredError` (exit 4); a plain `AgentLoopError` never carries
"human decision" text. The message is `render_human_decision` output in
`errors.py`: the verbatim, possibly multi-line "what happened" part, then
exactly two final lines, `To clear: <one action>` and
`Safe to automate: yes|no - <reason>`. `parse_human_decision` inverts it.
Errors are built through `HumanDecisionRequiredError.decision(...)` or from a
registered full-message renderer listed in `HUMAN_DECISION_RENDERERS`, directly
or through a same-function local. Each renderer keeps the leading prefix that
history export and triage recognize. A stop that crosses functions carries the
built error object: `PrStepBackDecision.deferred_stop` holds the deferred
no-progress error, and the plan no-progress stop builds a
`PlanPrePanelHumanDecisionError` (both `PlanPrePanelSafetyError` and
`HumanDecisionRequiredError`) that `stop_plan_pre_panel_decision` posts once
and re-raises. Post-merge context is added with `exc.with_context`, which keeps
it inside the "what" part. The bounded PR diagnostic shortens only the "what"
part, so the two final lines still end the posted body.
A handler that re-wraps a caught `AgentLoopError` around a call that can raise
a human-decision stop first calls `reraise_human_decision(exc, context)`. That
call re-raises the stop with the context inside 'what', so the qualification
board check and the plan-lineage and plan-approval readers keep exit 4. The
guard test checks this with a call graph built from the parsed source.
Fallbacks that turn a caught error into a result, such as the plan-approval
input derivation, legacy plan-history classification and signed-supersession
reads, historical PR board-binding validation and the managed-CI
source-authority fallback, also call it first, so their degraded results never
carry or hide a stop. A second guard resolves calls through imports, flags any
such fallback that can reach a stop, and lists the agent-turn failure
policies and advisory previews that keep their degraded outcome, each with a
reason.
`tests/test_human_decision_stops.py` enforces this statically from the parsed
source class graph: the phrase scan, the provenance rule and the renderer
bodies. The hyphenated `human-decision` protocol status token is not a stop.
The staged needs-human JSON outcome keeps exit 2.

**Unchanged-head stop and evidence progress.** `_UnchangedHeadTracker` counts
consecutive PR follow-ups that leave one head unchanged and stops at
`MAX_UNCHANGED_HEAD_CODER_TURNS` (#985). It has one bounded exception
(#1626). Each tracked head gets a single extra review round when the follow-up
at the limit made cited, authoritative evidence progress, meaning a receipt
that passes three checks:
- it is in the coder acquisition's frozen catalog
  (`acquisition_test_observations`, captured before any repair launch can
  repoint the runner's latest test turn);
- either the live, unbounded reconciliation retained by
  `Runner.render_local_test_evidence` names it as `superseded_by` on a
  parent-observed failure (`superseded_by_current_turn`), or authenticated
  canonical derivation newly verifies a previously unsatisfied row requested
  by a receipt-only blocker, with the same approved matrix and head as the
  prior coder record (`canonical_receipt_progress`);
- a surviving `test_observations` citation of the accepted follow-up cites it.

Canonical row progress requires every remaining blocker to carry evidence-row
IDs; an untagged implementation blocker cannot earn this new allowance.
Repeated passes and unchanged canonical statuses earn nothing. Evidence does
not resolve findings automatically: the next reviewer must disposition them,
and CI qualification remains independent.

A failed citation repair therefore never earns the grant. A different head
resets it, and like the count it is in memory only. The live reconciliation's
prior-round input is the previous round's bounded projection
(`prior_local_test_evidence`, decoded in `Runner.render_local_test_evidence`),
so after a resume a parent-observed failure evicted from that projection cannot
be seen as superseded and cannot earn the grant; this is conservative (#1639).
The bounded round evidence projection supplies citation display and persisted
reviewer context; admission and progress use the frozen acquisition catalog,
authenticated canonical derivation and live reconciliation, never projection
membership. `Runner.render_local_test_evidence` reserves accepted citations
matching parent-observed, unrestored receipts in the original acquisition
before older rows under both the 32-row and 16-KiB bounds. Format repair and
post-authentication correction cannot replace that acquisition identity.
If the protected set itself exceeds either capacity, projection raises an
explicit capacity error and stops the handoff rather than publishing known
receipts as unknown. Unprotected failures retain priority over other rows;
truncation preserves unresolved failure IDs and capture-incomplete caveats,
so omitted details never imply supersession. Current legacy `tests_run`
declarations replace prior display-only declarations instead of accumulating
or colliding at reused indices. Legacy declarations remain self-reported with
unknown attribution and cannot earn canonical satisfaction or progress.
When the stop fires
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

Evidence-stall and evidence-freeze human-decision stops use the shared
best-effort bounded diagnostic publisher in `pr_loop_support.py`, bound locally
by `pr_loop.py`. Every actual stop, including a frozen rerun without new signed
input and a still-frozen response pass, attempts one plain audit comment and
re-raises the identical full exception with exit 4. The mandatory freeze write
precedes the diagnostic; a freeze-write failure retains its failure semantics
and never produces a successful-freeze diagnostic. An audit-post failure only
warns and cannot resume work. Non-stopping gate actions remain unchanged.
`history_export.py` recognizes the anchored first-line evidence openings as
`pr-evidence-stall` and `pr-evidence-freeze`. Each stopped invocation counts,
including repeated frozen reruns, without manufacturing review rounds. These
comments carry no authoritative state, authorization or budget; only the freeze
record restores the boundary. Failed posts and historical stderr-only stops
cannot be recovered retroactively. Other human-decision families are unchanged.

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
the coder boundary by max), and are not resumed as completed repeat reviews. Isolation
still holds within the repeat: a repeat-pass publication is a peer, so a reviewer still
needing a fresh turn after it stops fail-closed (the repeat keeps its own route and is
never closed as incomplete).
A foreign-authored or audit-rejected record yields no boundary. Staging a repeat discards
the round's review spool, so a pre-conversion withheld outcome is never replayed, and an
operator conversion refreshes the round snapshot (or, if that read fails, the visibility
fallback carries the process's own conversion record). An unfinished admitted operator
conversion (`conversion_review_round` unset) is recognized by its round on resume, so it
restores the full-board repeat just as an auto conversion does. When the complete board
overrides a staged scheduler decision (a conversion repeat or an evidence pass), the
decision itself is rewritten to that board, so the checkpoint, reviewer and
reconciliation records never list a reviewer as both selected and paused. A conversion
boundary that neither the REST history nor the snapshot can admit (for example during an
identity-read outage) stops retryably; no comment is ever listed for deletion (#2064).
The evidence-response pass mints `SignedEvidenceReference`s for cleared converted rows
(`mint_signed_evidence_references`), keeps a durable tombstone, and threads the
cumulative set onto every terminal record. `reconcile_converted_obligations` runs after
the round-start ledger assignment and inside `_evidence_freeze_gate`.

After an external push breaks a converted-row freeze, `converted_capture.py`
derives one bounded capture opportunity from the admitted source freeze,
conversion audits and ordered coder records. The normal coder feedback names
the approved rows; code fixes can share the turn without manufacturing a code
finding. A completed publication with dispatch/result and converted-ledger
bindings consumes the opportunity even without evidence or after a coder push.
Legacy or uncertain history grants no retry. Tool evidence at the live head
supersedes conversions through existing reconciliation; unresolved rows keep
their identities and owners and reach the final freeze after normal gates.
Optional recovery capture is admitted only when a coder follow-up fits the
restored effective round ceiling, including authorized watcher allowances. At
the ceiling it is skipped without consuming the opportunity: unresolved rows
reach the normal clean-head evidence freeze and human-decision stop. Ordinary
code blockers retain the round-limit stop. Native evidence and live freezes
never dispatch capture, and round ceilings remain in force. No new recovery
payload is persisted.

`converted_capture_state.py` owns the pure capture eligibility rules shared by
round-state approval readers, qualification and capture dispatch.
`pr_loop_support.py` owns authenticated capture-history admission;
`converted_capture.py` composes those lower layers for recovery dispatch.
This keeps qualification and dispatch on the same rules without a cyclic
dependency between their orchestration modules.

A completed converted capture with new tool-verified evidence establishes a
semantic-review boundary. Same-head complete code approvals survive independently;
qualification also requires bound published acceptance of the captured claims.
Eligible captures select the primary or per-row active requesters using the
[post-capture assignment and scope contract](review-scheduling.md#post-capture-evidence-acceptance).
Evidence-only records grant capture authority only and are excluded from
complete-review accounting at every consumer, including code-round completion,
panel/sweep qualification, step-back triggers and no-progress observations. Scoped findings
remain actionable, but dispositions are restricted to assigned reviewer-owned
evidence findings. Machine supersession alone supplies no semantic acceptance.
Replacement evidence reopens acceptance, while identical claims and citations
retain it. Partial captures retain every unresolved row's final barrier.

`review_rounds._normalize_evidence_only_approval` constructs a public approval
carrier after the original evidence dispositions have been retained for ledger
settlement. It omits only eligible active evidence dispositions, retaining
requests and final barriers without resolving them for serialization. Both
publication modes use this carrier; generic model approval completeness stays
strict. `pr_restart.py` checkpoints qualification restarts with their actual
head and architecture causes, advancing head-bound obligations only when the
head moves.

## GitHub PR-head observation

`pr_head_sync.py` rejects closed or merged PRs with the existing not-open
error before reading a branch tip or diagnosing lag. Comment-only completion
deduplication after merge uses the plain context reader, so automatic branch
deletion cannot interrupt merge completion bookkeeping. Optional conversion,
scheduler-checkpoint, step-back and converted-evidence visibility refreshes keep
their existing fallback when the plain context read fails. When that read
succeeds, head observation runs outside the fallback handler: closed-PR stops,
desync and verification failures propagate before the refreshed history is used.

`pr_head_sync.py` observes the PR projection, REST PR head and independent tip
of the exact head repository/ref before lifecycle publications, entry activation
and fresh grants, selected review rounds and post-coder progress accounting.
Issue-to-PR entry includes direct and approved-plan implementation, legacy
resume and plan-first legacy-closing-reference resume before extracting handoff
metadata. Callers consume the returned complete context, including its URL.

At most three samples, two seconds apart, permit transient convergence. Full
commit SHAs and repository/ref/base identity are mandatory. Equality avoids an
ancestry call; differing immutable commits use the head repository's comparison.
Only an ahead result with matching base and merge base proves descendant lag.
Behind/diverged tips retain existing handling. Missing data, uncertain ancestry
and exhausted projection disagreement raise observation verification errors.

Persistent descendant lag raises a distinct observation stop before unchanged
accounting, accepted or rejected coder publication, continuity evidence or
qualification reuse. Stops can enrich their operator recovery text from already authenticated origin
and issue scope without granting authority or publishing records. Diagnostic origin
uses explicit contract origin first, then authenticated adoption when the origin
is absent; adopted PRs receive existing managed recovery advice rather than
conditional issue-created fresh-authorization advice. Normal
interruption cleanup still runs. The branch tip
never replaces the PR-record head as authority for review, CI or merge. Reads
cannot be atomic with later pushes; existing exact-head gates remain necessary.

## Human evidence mode

`--pr-human-evidence freeze|advisory|off` (default `advisory`, #2063) gates the
whole native human-evidence channel; every branch keys off the configured mode,
and persisted state only supplies the migration trigger and list contents.
`human_evidence_mode.py` holds one normalization choke point. In `advisory` and
`off` it removes every open native or converted `human-exact-head-evidence`
obligation (never a screenshot obligation) from the ledger. It runs after each
ledger-construction path: the round-start resume assignment, restore for new
input, and the round's request ingestion. It runs once more inside the
finalization gate wrapper, so no such obligation reaches partitioning, dispatch
or finalization. `advisory` moves each removed item onto a deduplicated
"Human checks requested" list and `off` drops it with a log line. The list is
stamped, through a context-variable stamp at the single round-metadata encoder,
on every orchestrator PR record while it is non-empty (`human_checks`; omitted
when empty; malformed individual entries are dropped; a spillable
round-transport field, so a large list moves to sidecars, and a missing or
corrupt sidecar fails the record closed instead of reading as an empty list). It is also written on the human-checks comment,
whose record uses its own flow so PR round reconstruction never reads it. Each
entry keeps the complete removed item, so a later `freeze` resume restores the
original typed obligation (converted-row identity, owners and owner states)
instead of re-creating it; identities cleared at the current head are skipped.

Outside `freeze`, signed-source classification, association validation and
parsing, historical association routing, the evidence-only stall and automatic
conversion are skipped, and operator conversion is refused at config
validation. The finalization gate reconciles converted rows only in `freeze`, and
outside it a native evidence item reaching the freeze write fails closed
instead of freezing. Every reviewer-response parse site passes the mode explicitly, and
a static guard test enforces this. The finalization gate still runs whenever the
advisory list is non-empty, so the final head and signed-input reads, the
screenshot final gate and `_ensure_finalization_ready` all precede approval. At
approval the list is posted once per head and identity set. With `--auto-merge`
and a non-empty list the merge and post-merge completion are skipped and the run
exits 0 with a notice. A non-freeze resume of a native evidence freeze publishes
a release with reason `advisory-mode` (unless screenshot obligations stay
frozen), sets revalidation against the freeze's signed baseline, and continues
without reviewers.

## Requester-owned screenshot obligations

Typed screenshot requests are separate machine evidence obligations with stable
requester-bound identities; legacy strings remain human-only. Typed-only complete
reviews normalize to code-approved/evidence-pending and never create a summary
code blocker or coder call. An authenticated current-head report, even one with
unresolved visible defects, may establish unrelated requested views only with
explicit bound requester acceptance. Other seats cannot accept the request;
removed requesters require all current-board votes. Partial/mixed obligations
stay pending. Full satisfaction publishes a verified screenshot coverage release
and returns to real same-head qualification; machine and signed-input gates retain
precedence. Visual findings survive independent of coverage artifact availability.
Both ordinary typed fields preserve presence and complete decoded values through
repair, including malformed/extra content and order. Dedicated coverage is a
strict versioned decision set and cannot become a code review. Reports and receipts
supply no signed-human or risk-matrix conversion authority.

Requester coverage restoration and final gates consume trusted in-process signed
human clearances together with clearances carried by authenticated freeze and
release records. These clearances are bound to obligation identity and exact
head; they do not require a machine coverage receipt or available image cache.
Partial clearance remains durable on the next freeze without reopening the
cleared request, while a new head requires fresh evidence. Disabling acquisition
does not disable restoration or final-gate evidence revalidation: unavailable
machine artifacts reopen typed obligations, while exact-head human clearances
remain valid. Converted-row reconciliation preserves the shared caller ledger
so coverage resolution reaches same-invocation finalization.
Frozen recovery is selected from the authenticated boundary even when restored
receipts have already cleared its ledger. Until the terminal release is durable,
the same-head handoff retains completed code reviews and passes the existing
machine and signed-input gates before publication-only recovery.

## Post-merge timing

`post_merge_obligations.py` separates strict typed proposals from admission.
`post_merge_runtime.py` admits selected-reviewer declarations and unanimous
resolution-owner transitions, or deterministic scheduling directives from
live-verified signed PR comments. Operator-only converted rows and signed
requirements cannot obtain authority from model claims or repair output.
Whole-block validation binds target identity and full requirement text; tracker
preflight requires an existing open same-repository issue outside the closing
set. Rejection retains the original evidence barrier and reports a target-bound
remedy. Owners are free text, without assignment or account validation.

A cumulative admission registry is published and read back before eligibility
changes. Source-bound unchanged scheduling survives head changes and remains
visible in reviewer prompts. Unscheduled evidence and machine gates remain
active; scheduling never means verification or fulfillment. Converted-row
timing is separate from approved matrix identity and evidence status. Shared
merge completion publishes tracking records only after confirmed merge; manual
completion uses the explicit recovery entry described in the CLI guide.

Declaration matching may release only a sole-requester evidence item; shared
items require explicit target-bound consent from every resolution owner. Rejected
new declarations retain a fallback evidence barrier unless that same requester
has an active declaration admission for the exact payload; another requester
with identical text and tracking details cannot suppress the barrier.
Successful admission clears only diagnostics for its own target, preserving
unrelated rejection remedies in durable metadata and subsequent prompts.
Tracker problems after admission use retained-obligation diagnostics: required
post-merge work stays scheduled and does not become a merge blocker. Replay
refreshes only its own target diagnostics, clearing them when its tracker recovers.
If publishing a diagnostic refresh fails without changing the admission registry,
the admitted control input stays consumed and surrounding signed prose remains
visible. The run reports the transport failure and retries on replay; new admissions
still require verified publication before they can change eligibility.

Signed-source routing (#1922) is owned by `requirement_evidence.py`. Validated
reviewer associations link surfaced signed IDs to that requester's native
human-only requests or classify answers as evidence-only/mixed. The shared
projection precedes acknowledgement repair and coder prompt/validation. Human-only
requirements remain pending evidence; evidence-only answers leave coder work;
mixed answers keep their original ID. Reviewers agree on a source when they agree
on its role (#2000); each entry's linkage is still validated against that
requester's own requests and native items, and leftover wording never conflicts.
An agreed mixed source carries the verbatim signed source plus every reviewer's
`leftover_instructions`, each attributed to its reviewer and ordered by reviewer
name, so live projection and `historical_projection` replay produce identical bodies.
The merged leftover is ordinary code work: the coder may receive it while the
answers are still unaccepted, and the requester obligations remain the deferred
finalization barrier (#1068), stopping at the evidence freeze before merge.
Every validated review contributes, including a blocking review with code work.
Unacknowledged signed sources receive one real semantic classification continuation,
including standalone sources with no evidence ledger or emitted request, never
format repair. The continuation shows prior classifications as reviewer and role
only. Missing classifications or role conflicts after that continuation stop
before acknowledgement reinjection or coder dispatch; a role-conflict stop names
each reviewer's role per source and the clearing input, a follow-up signed comment
that separates exact-head evidence answers from code requirements. The signed source and already
published pending evidence remain available for reviewer agreement or a human decision. Invalid associations are semantic judgement errors re-asked within validated reviewer turns; replay conservatively retains the complete source. Classification
cannot authorize acceptance: native requester citations and final live head/source
reads retain that authority before and after freeze.

A fifth role, `superseded` (#2101), routes a signed source that a later signed
source from the same author replaces, such as an older exact-head approval or a
note limiting it. Its fields are `request` `""`, `answered_item_ids` `[]` and
`leftover_instructions` `""`. A nonblank superseded leftover is not normalized
away; like any other nonblank field, it takes the bounded judgement re-ask.
`validate_associations` accepts it only when another surfaced live source has the
same author (`author_id` when both sides carry it, otherwise the normalized login)
and a strictly later parsed `created_at`. A missing author or timestamp fails
closed, so a reviewer cannot use the role to drop the newest requirement. An agreed superseded
source leaves the projection like evidence-only: it is reviewer-owned, needs no coder
disposition or acknowledgement, and creates no evidence request. Live and
replayed rounds derive the inert subset through `superseded_ids` and
`historical_superseded_ids`, and prompts render those IDs in a separate inert
sentence rather than under the exact-head evidence-path instruction. Two
reviewers who both pick `superseded` agree by role, while a split against another role
still stops with the role-conflict message. When a reviewer emitted no
`exact_head_evidence_requests` that turn, re-ask messages no longer offer
`human-only`, which cannot validate without a linked request; they name
superseded, evidence-only with the reviewer's own item IDs, or code.

Coder-disposition scope (#2020) follows the same projection. A signed source is
the coder's to dispose exactly when the projection keeps it; every other live
signed source (evidence-only, human-only or superseded) is reviewer-owned and never
coder-dispositioned. `requirement_evidence.reviewer_owned_ids` derives that set,
and the PR loop threads it into the coder prompt (a read-only notice that lists
the bare IDs and says reviewer demands for their dispositions are non-actionable),
coder follow-up validation, acknowledgement reconciliation, and reviewer guidance
(a scope line telling reviewers to accept those IDs only through their own
exact-head evidence path). The approval-marker rule is mode-scoped (#2083): under
`--pr-human-evidence freeze` the marker covers only the projected, non-reviewer-owned
set, so a reviewer emits it when every in-scope signed requirement is resolved,
including when the review blocks only on human-only exact-head evidence or nothing
is in scope; under `advisory` or `off` classification is discarded and the marker
covers the whole signed set. A coder disposition or `addressed_ids` entry naming a reviewer-owned ID gets
a distinct out-of-scope message, not "unknown requirement ID". The set is seeded each
round, including the first resumed round, from the single `historical_projection`
result used for reconciliation, and refreshed after each live projection, so live
and resumed rounds derive it identically. Conflicting or missing classifications
keep the source coder-owned. Plan-mode and standalone issue implementation have
no projection and keep the empty set.


## Decisions

[Decision log](decisions/findings-and-convergence.md)
