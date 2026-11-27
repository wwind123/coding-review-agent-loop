# Review scheduling

How reviewers are chosen for each PR head and each candidate plan: the shared pre-round snapshot and no-execution policy, selective intermediate and `primary-then-panel` PR review, signed board amendments, staged issue plan review with its exact-plan candidate key and reviewer-only phase-advance round, the transition classifier, the plan no-progress bound and reset window, the staged-plan mode conflict, the replay-derived plan step-back prompt, planning scheduler metadata, frozen policy evaluation and the shared reviewer exhaustiveness rule.

Source paths: `src/coding_review_agent_loop/capture_acceptance.py`, `src/coding_review_agent_loop/capture_dispatch.py`, `src/coding_review_agent_loop/review_scope.py`, `src/coding_review_agent_loop/code_approval.py`, `src/coding_review_agent_loop/review_scheduling.py`, `src/coding_review_agent_loop/plan_review_scheduling.py`, `src/coding_review_agent_loop/review_rounds.py`, `src/coding_review_agent_loop/board_amendment.py`, `src/coding_review_agent_loop/review_step_back.py`, `src/coding_review_agent_loop/review_evaluation.py`, `src/coding_review_agent_loop/prompts.py`, `tests/test_review_scheduling.py`, `tests/test_plan_review_scheduling.py`, `tests/test_review_step_back.py`, `tests/test_review_evaluation.py`, `tests/test_pr_loop_review_scheduling.py`.

## Post-capture evidence acceptance

After an admitted converted recovery capture, complete code approval and semantic
capture acceptance are independent gates. `capture_acceptance.py` derives stable
row assignments from conversion audit ownership and the effective board. Eligible
same-head captures use the configured primary for `primary-then-panel`; other
policies choose each row's first active requesting owner in board order, then
deduplicate the reviewer set. Every required complete code approval must already
qualify at that head, with the current plan, requirement, architecture,
acquisition and seat contracts and any panel/sweep obligations satisfied.
Operator force-full, missing approvals, uncertain ownership, a changed code head
or unrelated code findings retain ordinary policy review. Existing automatic
code-review recovery remains in force for code work; capture freshness alone
does not create another complete-board obligation after that work qualifies.

`capture_dispatch.py` shares scope, prompt and response validation across
sequential and parallel publication. Full, compact and summary-only contexts
retain the assigned row requirements, canonical claims, tool citations and
related findings. Scoped verdicts accept evidence only; they cannot open the
panel, supply a code sweep, resolve unrelated findings or clear machine evidence.
Semantic coverage gaps become actionable findings tagged with the assigned rows.
Reviewer failures retain the existing bounded availability and board-amendment
handling and never imply acceptance.

`review_scope.is_complete_code_review` is the common authority boundary for
decoded records and raw recovery mappings: only a reviewer record with absent
scope carries complete-review authority. Explicit null, malformed and unknown
scopes remain present through decoding and re-serialization and fail closed.
Scoped turns are excluded from complete approvals, panel/sweep derivation,
complete-round reconciliation, step-back triggers and no-progress observations, legacy board
participation inference, architecture approval and managed-CI recovery authority.
They still undergo effective-seat and board-integrity validation. Bound scoped
findings and their permitted dispositions use the capture reconciliation path.

Assignments bind the publication episode, canonical evidence, resulting head,
matrix, assigned rows and reviewer contract. The publication validation digest
includes scope, so private spool output cannot qualify a changed assignment.
Resume replays admitted scoped publication responses through ordinary finding-ID
assignment and settlement for blocking evidence findings, even when
private spool state is lost. Reconciled scoped rounds retain their assigned IDs;
these recovered responses still have no complete-code-review authority. Resume
reuses only admitted published matching acceptance. Identical evidence
keeps its binding; replacement claims or citations reopen acceptance. Ordinary
complete reviews can satisfy both gates when their durable context binding
proves that their prompt included the current capture. Ambiguous legacy history
requires fresh capture-bound ordinary reviews from the required board. Conflicting
requester histories disable narrow assignment without discarding the evidence
binding. Conservative pending capture work requests the existing automatic
scheduler fallback, including when transition classification cancels provisional
scopes; carry and resumed-review eligibility are recalculated at that point.
Strict pre-panel primary ordering still applies. Final CI/merge qualification
checks the same semantic predicate against live inputs. Capture attempt limits and unresolved
partial-capture barriers remain unchanged; no new CLI option is needed.

## Planning rollover and activation

A reconciled planning round accepts no fresh reviewer turns. Necessary fresh
reviews move to N+1 after qualified settlement, before publication or spool
preflight. Contradictory candidate history after a qualified panel opening keeps
the conservative full-board fallback. Unopened automatic fallback selects only
the primary and records no automatic force-full latch. Approval-only secondary
history permits ordinary later panel progression; premature blocking secondary
history retains the existing safety stop after a current-key checkpoint clears
degradation. A current operator override authorizes the full board in N+1;
an idle historical operator checkpoint supplies no opening or operator latch,
even when it is the first carrier of an amendment digest. The raw carrier
remains available to amendment-use and lineage validation.

History projection discards only idle post-reconciliation scheduling decisions.
Idle checkpoints retain negative candidate-key and invalid-metadata evidence
until a legitimate scheduler recovery checkpoint supersedes it. A later idle
matching checkpoint cannot restore intact history or completed approvals.
A legitimate force-full or contradictory-key recovery prelaunch retires its
selected reviewers' approvals from earlier rounds, including across restart.
This also covers primary-only unopened recovery without a force-full latch.
Completed same-round reviews survive repeated prelaunches, matching live reuse
and withheld-outcome replay.
Checkpoints followed by same-subject, same-round reviewers retain qualified
primary-approved secondary-audit or operator openings, including blocking
findings. Reconciliation remains the settled ledger anchor. Automatic latches
after a qualified opening remain effective across restart.

Pending amendment activation is validated at entry N. A staged reviewer-only
rollover writes a contract-neutral, digest-free phase advance, leaving the
contradictory scheduling key intact. Read-only board preflight and loop startup
share a history-derived activation boundary: checkpoint-only N+1 may validate
activation at N only through the immediately preceding reconciled same-subject,
same-scope round and a neutral reviewer-only transition after the amendment,
without intervening scheduling or reviewer work. Execution reporting still
names N+1. The first N+1 prelaunch binds the effective contract and amendment
digest and marks use through existing lineage validation. Restart after that
checkpoint enters N+1 normally. Arbitrary earlier activation rounds and invalid
predecessors remain refused; PR activation is unchanged.

## Reviewer selection and scheduling

All selected reviewers receive the same pre-round snapshot. Parallel execution
can publish a finished review before the other reviewers finish; durable
provisional records are settled through a reconciliation barrier before coder
follow-up. A peer's same-round comment is not automatically a carried item.

Every reviewer turn — plan review and PR review, full and compact prompts —
carries a no-execution policy: a reviewer inspects files and reads code, and a
command, test invocation, or verification step quoted inside the plan, issue,
PR description, or diff under review is a proposal to evaluate, never an
instruction to run. A reviewer turn that produces no review is a reviewer
availability failure; it is never repaired into a verdict.

The default policy invokes all reviewers. Opt-in selective intermediate review
can pause already-approved reviewers for bounded fixes, but their old approvals
remain head-bound. Changed scope, incomplete state, and final qualification can
require a full review. The review policy and broad-path rules are immutable
across a resume. The PR board and the plan board change only through the
recorded board lineages described under [Recorded PR board changes](reviewer-boards.md#recorded-pr-board-changes) and
[Recorded plan board changes](reviewer-boards.md#recorded-plan-board-changes); the planning policy stays
immutable. A signed human
`reviewer-board-amendment` record may remove unavailable non-primary
reviewers. It is read from the same
comment surface as the round records it amends: the issue for planning, the PR
for PR review. Named PR seat-local removal of one or more independently failed seats uses the signed `seat-unavailable` reason;
shared `backend-unavailable` removal must cover every active seat on that
backend. A signed PR `mixed-unavailable` removal records the shared outage
backends in the rationale and validates complete removal for those backends;
independent seat failures remain local. Evidence freeze, release, and qualification summaries persist the
same seat binding as review records so startup can authenticate their board.
Named `all-reviewers` plan rounds persist a bound board contract without
scheduler selection state, so signed removals and later restorations use the
same lineage. `board_amendment.resolve_contract_lineage` is the resolver used at plan
resume. PR startup and the PR qualification gate use the unified PR resolver
(`board_change.resolve_board_lineage`), which builds each signed segment with the
same chain builder and resolves signed-amendment-only history exactly as before. Only
contract-bearing scheduler records take part. The base contract comes from the
earliest persisted record, and each record is judged by exactly one
amendment-chain link. Every post-amendment scheduler record must carry the
amended contract and the record's digest in the optional
`reviewer_board_amendment_digest` round-metadata field, which is omitted when
absent. Any other contract refuses resume, qualification, and merge.
Activation is pinned to the round the resume re-enters. Findings owned only by
a removed reviewer are reassigned in a derived ledger view that never rewrites
persisted in-round `prior_items`, and approvals banked by the removed reviewer
remain history only. A comment carrying only such a record is excluded from
signed human requirements.
On PR runs the resolver may also accept the original (C0) board as the
configured contract; startup then rebinds the run's config, scheduler contract,
and reviewer set to the amended board, so every full-board selection, prompt, and
gate check sees it. Strict managed-CI issue-plan checks keep the operator-supplied
board. Amendment parsing normalizes CRLF, and an amendment-shaped comment whose
fence cannot be read is logged and excluded from signed requirements.
The opt-in `primary-then-panel` policy adds a phase-aware contract with one
primary reviewer and a non-empty secondary panel. It keeps the primary as the
only normal reviewer until exact-head approval, then dispatches an independent
complete-diff audit to all secondaries from one frozen snapshot. Scoped
remediation rechecks finding owners plus the primary before a mandatory exact-
head secondary sweep. Before the first exact-head primary approval the phase is
strictly primary-only: broad or ambiguous changes and every automatic recovery
reason re-invoke just the primary with full context (`strict pre-panel
fallback:`) and latch nothing. The panel opens only at a qualified panel
opening derived from comment order: a `secondary-audit` record preceded by the
primary's approval of the same head, or an operator-sourced force-full record.
Secondary approvals count only after that opening. After it, unsafe
scope/history, head changes, and the durable automatic latch use the complete
board (`post-panel fallback:`), and such a full-board decision raises the
monotonic automatic latch. A pre-panel state that cannot be made safe without
the panel stops with a diagnostic, and `--pr-review-force-full` (source
`operator`) is the explicit override; undecodable history stops in both flag
states. Phase,
owner, selection, approval-head, and scheduler-policy call-accounting metadata
are optional extensions to the legacy scheduler core, so old records remain
decodable and grant no staged phase authority.
Converted recovery captures add a freshness boundary shared by approval carry,
resumed reviews and qualification. Selected reviewers whose approvals predate
the admitted capture must review the captured row claims and tool evidence.
Published fresh approvals can be reused after restart; a stale local response
checkpoint cannot bypass the boundary. A fresh qualification read returns to
scheduling if post-capture semantic approval is incomplete. The force-full
latch selects policy-authorized scope only: ordinary same-head coder follow-ups
and non-capture obligation-digest changes preserve otherwise eligible carry,
including approvals without a recorded digest. Persistent latches do not
re-invalidate fresh post-capture approvals. Automatic pre-panel recovery stays
primary-only; post-panel and explicit operator scope rules still apply.

See [selective and staged review](../local_agent_loop.md#selective-intermediate-pr-review).

Issue plan review has its own scheduling component, `plan_review_scheduling.py`,
selected independently of the PR policy by `--plan-review-policy`. Full-board
planning stays the compatibility default. The staged planning policy binds every
decision to one canonical **exact-plan candidate key**: the ordered tuple of the
plan subject, aggregate plan identity, execution-strategy identity,
risk-test-matrix identity, and a digest of the surfaced planning-requirement
IDs. Staged planning requires a generation-1 plan, because a legacy unversioned
plan cannot form that key. The key is persisted on planning scheduler records
and on every plan reviewer record, so a later round compares a stored approval
component-for-component instead of inferring equivalence from the subject.

The plan-first lifecycle gains a **reviewer-only phase-advance round**. When no
must-fix plan item remains but a required reviewer still lacks a qualifying
exact-key approval, the loop posts a `plan-phase-advance` record, increments the
round number, and runs the secondary panel (later the final sweep) against a
byte-identical candidate plan with no planner turn. The advance record and the
round-budget diagnostic name the phase that is still outstanding, projected by
running the scheduler over the unchanged candidate key and the post-round
approvals, not the phase of the board that just finished. Resume anchors on the latest
coder record and the highest round number for that plan subject, so a round with
a phase-advance record and no coder record is a legitimate reviewer-only round
and never synthesizes a planner turn. A planning round counts as reconciled only
when it holds an actual reconciliation record: the `scheduler-prelaunch` and
`plan-phase-advance` summaries are pre-reviewer checkpoints, so an interruption
at either one resumes as an unsettled round that still reconstructs each
published reviewer's numbered items, owners, and obligations. A PR round follows
the same rule: only an actual reconciliation summary marks it reconciled, and its
`scheduler-prelaunch` summary (also the secondary-audit panel opening) is a
pre-reviewer checkpoint. The round-closure decision relies on this to treat a
half-published unreconciled round as incomplete and roll it over (#2064). Resume also
rebuilds the transition classifier's authenticated inputs — the durable
`semantic-patch-v1` payload and the cross-cutting contracts of the state that
patch was bound to — from the coder records themselves, so an interruption
between a remediation planner turn and its scheduler checkpoint keeps the same
narrow classification instead of latching the complete board; anything that
cannot be re-verified stays broad. Under staged planning, and only there,
ledger completeness is judged the same way: an item recorded under an earlier
plan subject stops counting as a missing obligation once the run has carried or
minted that item, or once the recorded history proves it was canonically
cleared. Full-board planning keeps the conservative reading it had before this
policy existed, where any cross-subject item at all makes the ledger
unreconstructible, so the compatibility default's context-mode selection and
posted bodies are unchanged. That second proof has to come from
the durable history rather than the running process, because a reviewer-only
advance records an empty carried ledger — after remediation there is nothing
left to carry — and a restart on exactly that seam would otherwise rediscover
the cleared item and read the ledger as unreconstructible. With both proofs, a
resumed run's later phase advance carries the approvals it just recorded into
the final sweep instead of re-invoking the complete board. Because each
advance costs a round, staged
planning consumes strictly more rounds than full-board planning, and exhausting
`--max-rounds` during a pending advance is reported distinctly from reviewer
blocking issues.

The transition classifier compares cross-cutting contracts, not prose. Its
execution-recommendation contract identity digests the whole recommendation
minus only its non-executable narrative: the top-level rationale, the caveats,
and each coupling constraint's rationale. Scope-item acceptance criteria, stage
fields, allocations, and any unknown key stay in the identity, because they are
copied into execution and child-issue data. A semantic patch that replaces the
recommendation is narrow only when that identity is unchanged, and a broad
reason names the changed components. The identity is recomputed from sidecars
on both the live and the resume path and is never persisted. The exact-plan
candidate key stays full-content, so a narrative edit still invalidates carried
approvals. Before a qualified opening, classification changes only the audit
reason, never the primary-only board. A primary reviewer that never approves
could therefore hold a run in the primary phase indefinitely. Since #2098 the
**plan no-progress detector** bounds that under every review policy (see
[findings and convergence](findings-and-convergence.md)):
the primary-phase stall stop and the step-back escalation stop were removed, and
`--plan-primary-stall-rounds` was removed (#2162) and is a startup error. Two
optional planning-only checkpoint fields remain: `scheduler_issue_digest` (issue
title/body digest, read by the step-back and sweep retirement rules) and
`scheduler_stall_reset`. The one-shot `--plan-reset-stall-streak` stamps
`scheduler_stall_reset` on the invocation's first staged prelaunch checkpoint in
whichever phase receives it (primary or post-panel; the decoder accepts the
marker on any valid phase), skips the detector on that round, and makes every
later run drop observations numbered below the newest reset checkpoint's round.
Both fields are omitted when unset and invalid on PR-flow records.
`--plan-review-force-full` stays the only way to convene the panel without an
exact-plan primary approval.

**Staged-plan mode conflict.** `plan-only` and `implement-one-shot` apply
the phased-delivery guard (`config.phased_delivery_guard_active`). At the start of
every planning round, before the growth seam and the scheduler block (so every
review policy is covered), a candidate whose v1 recommendation is `staged` stops
for a human decision through `stop_plan_pre_panel`: no reviewer or revision turn,
and no checkpoint, approval, or panel-opening record. `--plan-narrow-staged`
replaces the stop with an orchestrator-owned revision obligation modeled on the
growth guard: it blocks approval and reviewer-only advance, skips the plan
no-progress detector for that round, is delivered once per invocation, and is
not a reviewer item. Planner and reviewer round records carry
`plan_execution_mode` and planning checkpoints carry `scheduler_execution_mode`
(both omitted when unset, unknown values decode to none).

**Plan step-back prompt** (internal threshold
`review_step_back.PLAN_STEP_BACK_ROUNDS` = 2; `--plan-step-back-rounds` and
`--plan-step-back-escalation-rounds` were removed in #2162).
Step-back targets the "new edge case every round" pattern and is a prompt
technique only: it never stops the run (#2098). Under `primary-then-panel`,
`review_step_back.classify_review` reads the primary's *historical* review record
for a round (never the surviving unresolved ledger) and calls it `new-finding`
when the review blocked and introduced at least one `blocking` or `same-plan`
item it owns, `repeat-only` when it blocked with only carried-forward items or
future follow-ups, and `approved` otherwise. Growth crossings are recomputed from
each candidate's canonical plan and assembled sidecar and are never persisted.
`derive_plan_step_back_state` **replays** the counted primary reviews in round
order with at most one open episode. At each review round r it first applies the
closure rules to an open episode (an exact-plan primary approval at or after its
candidate round, a changed `scheduler_issue_digest`, a `scheduler_stall_reset`
checkpoint after the candidate record, or the panel opening), counting only
records before review r so a prefix replay and a full replay agree. While an
episode stays open no trigger is admitted, so the original episode and its anchor
persist. Otherwise one shared predicate, `plan_step_back_trigger_holds`, applies
from max(crossing, last closed episode's candidate round): the new-finding streak
since then reaches K, or the streak before r reached K and review r is blocking.
A trigger at r opens an episode with candidate round r+1 only when a planner
record numbered r+1 exists. The live post-review decision runs the same replay
with the current round as the live round, so a fresh run and a resume route
identically. A trigger makes the next planner turn a simplify-or-re-scope
revision: `step_back_context` replaces the "address the reviewer items" framing,
`step_back_review_notice` tells every reviewer of that candidate to judge the
alternative on its merits, and an open episode adds the anchor guidance to later
revisions. Legacy `step_back_entries`, `step_back_status` and the
`scheduler_step_back_deferral` checkpoint marker are still decoded, never read
by a plan decision, and no longer written; a malformed legacy entry no longer
degrades planning history. Replay may treat a round as a step-back candidate when
an orchestrator guard revision displaced the step-back turn; the effect is
prompt-only and identical on fresh and resumed runs.

**Plan sweep trigger** (internal threshold `review_step_back.PLAN_SWEEP_ROUNDS` =
2; `--plan-sweep-rounds` was removed in #2162;
`--plan-sweep-reasoning-effort`, off by default). The step-back arms only on a
crossed one-shot plan, so a primary can drip-feed new findings on a plan that
never crosses a growth signal. Under `primary-then-panel`,
`review_step_back.derive_plan_sweep_state` counts consecutive sweep-counting
primary reviews through the same counted-review walk as
`derive_plan_step_back_state` (`_walk_primary_reviews`: latest record per round,
a required same-round primary checkpoint, issue-digest and
`scheduler_stall_reset` retirement, the panel opening, phase-advance and invalid
scheduler boundaries, and publication records resolved through
`effective_new_items`), so the two derivations share their retirement rules and
differ only in the classifier. One eligibility filter,
`eligible_new_mandatory_items` (primary-owned `blocking`/`same-plan` items not
marked `repeat_of`), is shared with the history export; `count_sweep_items`
additionally drops findings tagged `revision_introduced`, but only on reviews
after the reviewer's first. When `streak_before(R) >= K` and the panel has not
opened, the primary's round-R prompt carries `render_plan_sweep_guidance` (the
finding classes from the streak's first round, bounded to 12 lines) in the
volatile part of the full and compact plan review prompts, and the orchestrator
writes the optional reviewer-record field `plan_sweep` (strict `true`, omitted
when unset, malformed decodes as absent). There is no episode, no planner turn and no stop, and the step-back
counts are unaffected when both apply. The effort override builds a turn-local
config (or seat copy) for that one primary turn in a fresh provider session; it
is validated against the primary's backend before any agent turn and ignored with
a log line for backends without effort control. Under the same policy, each
reviewer's later prompts carry a previous-plan baseline
(`reviewer_previous_plan_baseline`, resolved from the reviewer's latest earlier
review subject and the hash-checked planner record of that subject), against
which reviewers tag new mandatory findings with an advisory
`provenance` (`revision_introduced` or `pre_existing`). Minting persists `provenance` and the orchestrator-computed
`repeat_of` (a normalized-text match against the carried ledger) as optional
item fields, omitted when unset. The pure `select_provenance_baseline` chooses
the baseline form (#1516): the unified line diff, then a deterministic
sentence/cell/word-level compact diff (`compact_plan_diff`), each within the
diff budget max(12000, half the current plan), then the full previous reviewed
plan fenced as baseline data, then `unavailable`. Every form is bounded by
`--plan-provenance-baseline-max-chars` (default 120000, `0` disables), which
joins the primary-then-panel validation, child-planning inheritance and the
approved-plan board copy boundary beside the sweep flags. The plan loop selects
once per reviewer per round from pre-round durable records, so the volatile
prompt block, the per-reviewer log line and the optional reviewer-record field
`provenance_baseline` (`form`, `chars`; omitted when unset, malformed decodes
as absent) always agree, and a resumed round recomputes the same form.

Planning scheduler metadata is a flow-discriminated branch of the same durable
round-metadata transport. A planning record carries the candidate key in place
of the PR-only SHA pair and omits the PR-only broad-rule and scope-digest state,
so a planning record and a PR record can never be decoded as one another even
when both carry `policy: primary-then-panel`. Degraded planning history is
partitioned into exactly four disjoint classes with one outcome each: absent,
invalid, and contradictory-key history always continue under a conservative
fallback (strict primary-only before a qualified opening, complete board with an
automatic latch after one), and only a transport extraction failure stops the
run, checked at startup and at every round boundary. Degradation is scoped to
the latest valid planning scheduler checkpoint, so an older invalid record stays
auditable without pinning later rounds to the fallback, and the persisted
planning contract is immutable under both policies. A carried exact-key
approval is honored only when it also carried the acknowledgement for exactly
the currently surfaced planning-requirement ID set, so the signed-requirement
gate cannot be satisfied vacuously. When a same-agent re-ask recovers a missing
acknowledgement, staged planning persists the re-asked reviewer result as an
amended record before the phase advance, because the record written from the
original text carries no requirement IDs and a later record for the same
reviewer supersedes the earlier one; otherwise the very approval that the re-ask
just recovered would be rejected on the next round. Discussion-mode and child-planning cycles
keep the full board by configuration reset. See
[staged issue plan review](../local_agent_loop.md#staged-issue-plan-review).

Frozen policy evaluation is a local read-only boundary. `review-evaluation`
validates artifacts and deterministically reports severity-weighted marginal
coverage and process/cost/CI outcomes per flow: each run carries a validated
`flow` (`pr` or `plan`; only an absent key defaults to `pr`, so PR-only
artifacts keep loading while an explicit null is rejected),
run identity is unique per `(flow, policy, run_id)`, and aggregation and report
titling are per flow, so PR and planning runs that share a policy name are
never pooled. Every measurement
is `verified` only with trustworthy run/label provenance; otherwise it is
explicitly `unavailable`, findings are namespaced by run with unique per-policy
run identities, validated severity labels, and validated non-empty contributor
arrays (a valid finding is never silently dropped from coverage), and the
evaluator has no GitHub or reviewer-call dependency.

PR and plan reviewer prompts, in both full and compact forms, share one static
exhaustiveness rule: report every independently substantiated defect on the
reviewed head or plan in one response, enumerate same-path defects together,
and state genuine masking in the finding text and `summary`. It changes no
response schema and sits inside the compact stable prefixes. To measure it,
runs may carry a validated `review_contract` label (`first-finding-permitted`,
the absent-key default, or `exhaustive`) with optional
`review_contract_provenance`. The report adds
`flows.<flow>.review_contracts.<contract>.policies.<policy>` cells holding
rounds, reviewer calls, and escaped defects per run; a cell is `verified` only
when every run has verified metric provenance and verified label provenance,
and there is deliberately no per-flow or cross-policy contract rollup, because
scheduling policy itself drives calls, rounds, and escapes. Two artifact pairs
are checked in under `docs/evaluation/`: the synthetic regression fixture
(`frozen_review_artifacts.json` and `frozen_review_report.json`), which never
receives real runs, and the real-run pair (`review_contract_runs.json` and
`review_contract_report.json`), which ships empty and is extended only by the
documented data-only
[freezing procedure](../local_agent_loop.md#freezing-a-real-run).

## Failed reviewer seats in a blocked round

A reviewer seat *fails* in a round when its response is still rejected after
its same-agent re-asks (a deterministic failure), when it is unavailable
(transient or agent-unavailable), or when it reports an incomplete review. In
both the PR loop and the plan loop, parallel or sequential, in any reviewer
order, a failed seat no longer stops the round when another seat in the same
round blocked: the remaining seats still run, and the round continues to the
coder (PR) or planner revision (plan). The shared rule lives in
`failed_seats.py`.

- **Carried-item freeze.** Every disposition this round of an item owned or
  co-owned by a failed seat (by item ID or alias, including sub-item
  dispositions) is dropped before settlement by
  `unresolved_items.preserve_failed_seat_items`, so a healthy seat can never
  resolve, defer or edit a failed seat's item. The filtered map feeds both the
  settled ledger and the round's reconciliation record, which therefore agree.
- **Durable record.** The round's reconciliation record names the failed seats
  in the bounded `failed_reviewer_seats` metadata field and the comment lists
  each seat's failure category and frozen items. A continued sequential PR
  round posts that record too, and a sequential plan round posts it whenever
  a seat failed, including on the stop path. Resume, history replay and the
  board-change carry-over apply the same freeze from that field (see
  [Round state and resume](round-state-and-resume.md)).
- **Never approving.** A failed seat never enters an approval set; the
  all-approved, qualification and merge gates still require an exact-head
  approval from every required seat. It is treated like an unavailable seat for
  the carried-panel no-dispatch shortcut and the migration re-probe.
- **Invoked again.** The seat is not quarantined for the run: the next round
  invokes it again, and a selective (PR or staged plan) scheduler force-selects
  it with a logged reason. A PR restart rebuilds the pending retry from the
  latest reconciliation record until the seat publishes a review.
- **Plan resume.** A rerun of a reconciled sequential plan round whose only
  missing reviews are the recorded failed seats re-invokes those seats in the
  same round, resumes the healthy reviews and posts a fresh reconciliation
  record; the latest record of a round is the one recovery reads. A retry
  interrupted after the failed seat published but before that fresh record is
  finished on the next rerun before the round advances; reviews published
  after the failed-seat record belong to that unfinished retry, so their new
  findings are minted exactly once. Eligibility comes from the recorded round,
  never from an invocation's parallel flag: a round whose original reviews
  (before the failed-seat record) were published in parallel keeps the
  reviewer-only rollover even when rerun sequentially, because a same-round retry would let the failed seat read its
  peers' published verdicts.
- **Parallel independence.** A parallel round normally withholds healthy
  peers' outcomes until a failed seat completes an independent rerun. When a
  healthy peer returned a completed blocking review (a new finding, a carried
  item kept blocking, or a substantive summary) and continuation is allowed,
  the failure settles in the round instead; settings-lock waits keep the
  withheld rerun.
- **Stop kept.** The existing error, message and incomplete-review comment
  are unchanged for single-reviewer boards, when no healthy seat blocked (every
  other seat approved), for quota-reset errors, in evidence passes, in the
  final allowed round, and when settlement leaves no must-fix item. A
  sequential PR round that stops keeps today's shape and posts no
  reconciliation record.

## Completely approved resume

Completely approved plans are an exception to fresh-review rollover. Startup
proves approval on the approving board, then resumes the live approval gate
without selecting reviewers, writing a scheduling decision or recovering a
fresh-turn spool. Current-round dispositions still settle the ledger; paused
seats count only through qualifying exact-plan carried approvals. Neither the
reconciled-round rollover nor the later missing-approval advance may write or
consume another round under this shortcut. Live qualification disagreements
refuse before either write, including at the round limit, with the plan, round,
missing seats and their seat/restoration/parent-requirement bindings. Requirement,
obligation, inherited-matrix and growth checks still govern execution.

A matching opt-in visual pass follows applicable bounded CI-start observation
and existing execution/freeze gates. It never authorizes final managed CI.
Bound visible-defects publications have no scheduler metadata or code approval,
panel, sweep, capture acceptance or evidence-boundary authority. Authority readers
exclude those records before checking history presence or selecting current/prior
state, while full history preserves visual finding IDs and owner obligations.
Only proof-verified assigned visual dispositions settle visual findings; ordinary
code turns cannot do so. Failed evidence permits ordinary work, then unresolved
visual blockers with no disposition opportunity require a human decision before
merge or another futile coder dispatch (including unchanged-head retries).
A successful pass offers that opportunity only while its selected owner remains
pending after reconciliation. Clearing that seat's share of a multi-owner finding
leaves other pending owners intact and requires the same human stop. If the
selected visual seat changes while a blocker's owner remains on the board,
the blocker survives and the human stop names its owners and the current seat;
an unrelated successful pass cannot authorize another coder turn.

Visual findings are excluded from code scheduler obligations and every scheduler
ledger digest, including coder checkpoints and qualification comparisons. A fresh
visual pass adding or settling a finding cannot invalidate a code checkpoint,
select an image-less owner turn, or latch a full-board recovery.

Visual dispatch runs once before reviewer fanout in sequential, parallel and
primary-then-panel scheduling. Its invocation-local guard also suppresses repeated
listing, CI observation, diff and capture after pre-intent failures. Only stale
pre-capture dispatch inputs release that local guard; durable spend is never
released. Restart can retry pre-intent work but cannot repeat an intent-backed
assessment. Only a fresh, verified published assessment in the same invocation
can offer the existing single coder opportunity while the selected owner is
pending. Recovered results and skips never replenish that opportunity.

Fresh captures on a new head compare only with the latest authenticated terminal
source and its original assessment. Matching images and reviewer contracts may
skip assessment while retaining the new head's capture binding. Source coverage
includes prior assignments and findings raised by that assessment, with their
owner shares. New pending shares outside that coverage prohibit reuse. A source's
own blocker does not prohibit reuse: it carries unchanged and requires a human
when no fresh disposition opportunity exists. Skips have no dispositions and no
code approval, screenshot-evidence, panel, sweep or scheduler authority.

## Requester-owned coverage scheduling

The merged image seat and at-most-once image lifecycle are unchanged.
`review_images_coverage_run.py` batches pending typed screenshot assignments per
requester seat/head at existing evidence gates, after machine gates and signed
input response ordering. Paused requesters and carried code approvers can execute
text-only coverage without code review or opening a panel. Removed requesters
require every current-board vote. Intent publication/read-back spends one attempt
per repository/PR/head/executing seat and one assignment per request/voter/head;
contract changes and late requests cannot reset it. Failed/unknown attempts never
retry. Original outcomes may be replayed only for publication. Neither scoped
coverage nor its attachments qualify code reviews, sweeps, panels, CI or merges.

## Decisions

[Decision log](decisions/review-scheduling.md)
