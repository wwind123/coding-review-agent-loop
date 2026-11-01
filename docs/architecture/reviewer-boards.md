# Reviewer seats and boards

How reviewer boards are configured and changed: named reviewer seat preflight, the per-invocation PR board and its handoff rule, the reviewer floor, and the recorded board lineages that keep every reviewer on the board in force approving the exact final PR head or plan across in-run board changes.

Source paths: `src/coding_review_agent_loop/reviewer_seats.py`, `src/coding_review_agent_loop/reviewer_floor.py`, `src/coding_review_agent_loop/board_change.py`, `src/coding_review_agent_loop/board_amendment.py`, `src/coding_review_agent_loop/pr_board_state.py`, `src/coding_review_agent_loop/plan_board_state.py`, `src/coding_review_agent_loop/plan_verification.py`, `tests/test_reviewer_floor.py`, `tests/test_board_change.py`, `tests/test_board_amendment.py`, `tests/test_pr_loop_reviewer_boards.py`.

## Named reviewer seat preflight

`reviewer_seats.py` defines an immutable seat configuration with a stable ID,
backend, ordered model chain, optional supported effort, and optional workdir.
The CLI validates named seats before command lookup, repository detection,
workdir claims, and agent invocation. Validation covers reserved identities,
model overlap within one backend, and explicit checkout collisions. Sandboxed
permission checks use the seat's resolved backend and identify the rejected
seat. A same-backend legacy reviewer with an unknown implicit CLI model is
rejected until its model is explicit; backend model and effort options require
an active coder or legacy reviewer to own them. PR review uses a string-valued
seat role key carrying backend, model chain, effort, and checkout. Common
review scheduling and ledger code stores that seat ID in reviewer and agent
fields. Backend adapters receive an immutable seat-specific invocation view;
response files and logs use the seat ID while Antigravity quota state and the
host settings lock remain shared. The settings lock is held for a whole agy
turn, so the adapter publishes a token-owned holder record (PID, run ID, seat,
role, turn bound) after acquiring it, and a waiter allows each observed holder
its advertised turn bound plus a margin under an overall cap. An expired wait
raises `AgentSettingsLockWaitTimeout`, a transient `AgentInvocationError`,
before settings.json is touched; `run_validated_agent` and the direct
Antigravity repair path each retry it once in place without spending retries
or model fallback. A second expiry stays seat-local (repair records
`settings_lock_contention`) and is never treated as a backend outage; a
parallel PR round treats it as retry-bound, withholding healthy peers in the
round spool so a rerun completes the round independently. PR round metadata, including scheduler and
coder recovery records, stores a versioned board binding. Recovery verifies
the binding before using any recorded checkpoint; a changed
model chain starts a fresh round, retaining reconciled open findings but
invalidating prior approvals and unpublished response checkpoints. Legacy
default seats keep historical signatures and metadata bytes. Issue-plan review
writes the same versioned seat binding into coder, reviewer, and scheduling
round records. Resume validates the binding before reading
approvals or amendments; a model-chain change starts a fresh review round
while preserving verifiable seat-owned findings.

## Per-invocation PR boards and the reviewer floor

The PR reviewer board is a per-invocation choice. A plan approved by board A
can hand off to PR review by board B, so the handoff no longer requires the
plan's named seats, seat set, or seat models. `plan_verification.py` derives one
plan verification context from the plan's owning issue: the persisted plan
contract and its signed plan amendment lineage (C0 to Cn), the plan seat
binding, and the planning policy and primary. Plain all-reviewers plans persist
neither, so their board is every reviewer that reviewed the plan. The context
validates the plan's complete seat-binding history. Mixed bound and unbound
rounds, a changed backend, a changed recorded board, or a reviewer outside its
own binding fail closed. A model chain or effort change between consecutive
bound records is a binding transition. Every approval recorded before the
latest transition is superseded, even when a later record returns to the
earlier model (A to B to A). Approvals bound to a model the latest binding
replaced never count either. A malformed staged checkpoint that hides
the persisted policy also fails closed; it is never downgraded to
all-reviewers verification. Every
plan-approval reader uses this context and never reads the PR invocation's
board: managed-CI fresh authorization, ordinary resume without an issue-side
handoff, the handoff seam, and strict managed-CI qualification. For a staged
child whose plan records do not match the handoff plan, the seam reads the
parent issue that supplied the approved plan.

`reviewer_seats.resolve_plan_handoff_board` applies the handoff board rule. If
the invocation's seat-ID set equals the plan's C0 or Cn, the PR board is
restricted to Cn, so a signed plan removal stays removed and a signed
restoration is included. Any other seat set is an operator reconfiguration, and
the PR board is exactly the invocation's board. When the derived PR board
differs from Cn in membership, backend, model chain, or effort, the PR loop
posts one plain-text board-change audit record on the PR before round 1. The
record lists the old and new boards, effective round 1, and reason
`operator-reconfigured`. A digest over that content identifies it, so a rerun
posts nothing new. It carries no protocol marker or signature grammar and
cannot be read as a signed amendment or human requirement. An unchanged board
posts nothing. Plan approvals never count as PR approvals: every PR board
reviewer must approve the PR's exact head.

`--min-reviewers` and `--min-distinct-providers` set an optional
per-invocation floor. Config validates only that the values are positive
integers. `AgentLoopConfig` construction and `dataclasses.replace` never compare
the board with the floor, so amendment rebinding cannot trip it.
`reviewer_floor.check_reviewer_floor` checks the effective board after
read-only history and lineage resolution. It runs before any workdir setup,
managed-CI activation, PR-side write, or agent invocation. That covers
issue-run start (before base resolution and workdir setup, using the parent's plan
board when a direct-implementation staged child implements the parent's plan),
`managed-pr` (before
the branch and draft PR are created), and PR start or handoff. Providers are
counted by seat backend. Only a validated, non-empty signed amendment chain
supplies a floor exception. At handoff that requires an inherited seat set. On
the PR, the signed PR amendment lineage is resolved read-only at the same early
point. That includes the backend-outage completeness check and the activation
round of pending amendments, so malformed or not-yet-activatable amendment
history also fails before any write and never supplies a floor exception.
Malformed round history fails closed under every review policy. The first PR
write can be the release of a retained managed label on a ready PR or a
managed-CI fresh authorization record. Whenever signed PR amendments exist, or a
floor is set, the complete lineage and the floor are resolved at PR entry,
before either write. Signed amendments are validated there even without floor
flags. That resolution uses the authoritative approved plan. The owning issue is
the caller's issue, an explicit managed-CI issue scope, the PR-side contract's
primary issue, the issue named by the managed branch (ordinary managed-CI
recovery), or the single issue the PR body links. The plan identity comes from the caller's approved plan or from
the issue-side handoff that names this PR. The board then comes from the issue
holding exactly that plan, either the child or its staged parent. With no
handoff record, only the managed-CI recovery scopes (fresh authorization with
an explicit issue, or ordinary resume from the managed branch) bind the owning
issue's completely approved canonical plan. That scope is decided from the
invocation's managed-CI settings and the managed branch, not from which lookup
found the owning issue, so a caller-supplied issue or a PR-side contract does
not hide it. It is verified by the same helper
those recovery paths use, and that issue's board is used. An ordinary PR whose
linked issue has no handoff record naming it binds no plan, so unfinished
planning on that issue is never consulted. The read-only PR preflight also validates persisted PR seat bindings
against the lineage segment that governs each record before it grants a signed
floor exception. This is the
same selection the handoff seam applies, and the seam repeats the check. The
exception covers only the measures the chain
lowered from C0 to Cn, and never goes below the signed board's own counts with
its recorded bindings. A binding swap therefore cannot inherit a provider
exception. In-run PR board changes are recorded as described in [Recorded PR board changes](#recorded-pr-board-changes),
and in-run plan board changes as described in [Recorded plan board changes](#recorded-plan-board-changes).

## Recorded PR board changes

The PR reviewer board is a per-invocation choice under every review policy,
including the default `all-reviewers` policy, which persists no scheduler
contract. `board_change.py` keeps the guarantee that every reviewer on the
board in force approved the exact final artifact through a durable, auditable
board lineage instead of a frozen roster. `pr_board_state.resolve_pr_board_state`
resolves it read-only at PR entry, at startup and at the qualification gate.

- **Board identity and root anchor.** A board identity is the policy, primary,
  broad-path rules and ordered seats (ID, backend, model chain, effort). Every
  PR round record written by the PR loop carries `pr_board_root` (the immutable
  root identity C0) and `pr_board_identity` (the identity of the lineage
  element that governs it). They are stamped in the single round-metadata
  encoder, so every writer is covered. The root is the root carried by the
  earliest identity-bearing record or link, and every later one must carry the
  same root. It is never re-derived from the effective board.
- **Legacy migration.** With no anchor, the root is derived once, read-only:
  the earliest persisted scheduler contract (with the latest legacy seat
  binding when one exists), else the recorded named seat binding, else for
  plain `all-reviewers` history the earliest signed amendment's original board,
  or the configured board plus every historical reviewer poster. Plain
  membership is never inferred from round completeness, so an unchanged board
  resumes with no record whether a reviewer crashed, was unavailable or hit a
  fatal stop. Plain history records no models: an unknown binding is
  initialized from configuration by the first durable artifact
  (`pr_board_binding_init`) and compared exactly afterwards. Before that, it
  is compared compatibly (seat ID, order, primary and backend only).
- **Authenticated named legacy entry.** Issue, plan-first and task
  hand-offs post an initial PR coder record with no seat binding and no root,
  so a named board derives its legacy root from the configured board and the
  first PR round anchors it. Before that derivation adopts unanchored history,
  `pr_board_state.authenticate_named_legacy_pr_entry` runs once at
  `run_pr_loop` entry. It is a top-level call after base resolution and before
  the floor/amendment block and the first PR write, with no other gate. It
  applies only when named seats are configured and the PR has unanchored coder,
  reviewer or summary records and no link. The owning issue comes from
  `pr_board_state.resolve_pr_owning_issue`, the same resolver
  `pr_loop._entry_plan_handoff_board` uses: the caller's issue, else the
  explicit managed-CI issue scope, else the PR contract's primary issue, else
  the `agent-loop/managed-N` branch issue during managed-CI recovery, else the
  single linked issue. It reads the PR, and that owning issue, with
  `github.read_authenticated_protocol_comments`. Each such record, the PR
  contract record, and the latest handoff record naming the PR (compared with
  the unfiltered `resolve_issue_pr_handoff_lineage` selection) must be authored
  by the authenticated actor; otherwise it refuses for a human decision.
  Validators and initial coder-record shapes are unchanged.
- **Board-change links.** A configured board that differs from the effective
  board in membership, primary, backend, model chain or effort becomes one
  orchestrator-authored link comment: an audit text plus the reserved
  `AGENT_PR_BOARD_CHANGE` record. It carries the root, the predecessor digest,
  the old and new boards and seat bindings, the added, removed and rebound
  seats, `anchor_round`, `advance` and `effective_from_round`, and the reason
  `operator-reconfigured`. Only a comment by the authenticated orchestrator
  actor (the REST identity path used for PR recovery records) is admitted. A
  marker-bearing comment by anyone else, a malformed payload, a digest, root or
  predecessor mismatch, or an old board that differs from the derived one fails
  closed at startup and at qualification. Any record-shaped span naming the
  record counts, even one the strict grammar cannot parse. Agent text naming
  the marker is
  neutralized by the reserved-marker registry. Policy and broad-rule changes
  stay contract drift.
- **One ordered chain.** Links and signed amendments are ordered by comment
  position with strictly increasing effective rounds. The only exception is a
  link that supersedes a never-activated pending link: it copies that link's
  anchor round, advance flag and effective round. With link history, a
  pending signed amendment takes effect in the round after the highest
  recorded round, with the same carry-over as a link, and its printed template
  names that round. A signed amendment after a link amends the link's new board. A signed restoration can only re-add a seat
  a signed amendment removed since the latest link. Each round record is
  governed by the latest element whose comment precedes it and whose effective
  round is at or before its round. It must carry that element's digest in the
  matching field (`reviewer_board_amendment_digest` or
  `reviewer_board_change_digest`), and its identity and contract.
- **Floor authorization.** `lineage_floor_authorizations` is the only source of
  floor authorizations, at preflight, link candidacy, every round and
  qualification. Each signed segment is frozen from the identity immediately
  before it, plus its signed membership changes. A link never creates or
  enlarges an exception. A candidate link is floor-checked against the lineage
  that would exist after posting it, so a refused change posts nothing. A
  removal refusal prints the filled-in signed amendment.
- **Re-entry and carry-over.** The anchor round is the highest recorded PR round
  of any role or phase (a scheduler-prelaunch-only round included), and
  `advance` is set when it holds any record. The anchor round's records are
  collected under the predecessor board, so an outgoing reviewer's findings are
  kept, and a finding whose author is not its publisher, or a repeated item ID
  with different content, fails closed. Its persisted dispositions are
  reconciled with the existing rules, its new findings merged, and its evidence
  requests rebuilt with the head-scoped clearances accumulated by durable
  records up to the anchor round plus those its reconciliation produces.
  Replay never clears human-only evidence from an unreconciled
  evidence-response pass. An evidence freeze at the anchor round's head stays
  in force across the change, so the new board answers new signed input at
  that head in an evidence-response pass. A request whose requester left the
  board then needs every configured reviewer. A release boundary is dropped,
  so the new board reviews the head itself. The run then re-enters the
  effective round once, with the anchor round's coder output as the artifact. A link
  posted before a crash is recovered from durable history alone. A resume whose
  anchor no longer matches fails closed before any agent invocation or post.
- **Approval and ledger rules.** Required reviewers are the effective board
  only, so removed reviewers' approvals never count. Added, rebound, restored
  and role-changed seats must approve from the link's effective round
  (`fresh_from_rounds`). Removed reviewers' unresolved items are reassigned to
  the remaining board by the ledger view; settled items stay settled.
- **Primary restart.** A change of the primary seat, its binding, or a seat's
  role under `primary-then-panel` sets a primary epoch. Panel openings and
  approvals before it never count. Active secondary-owned items from before the
  epoch are carried obligations. They are exempt only from the pre-panel safety
  stop, keep their owners, and block qualification. When the restarted primary
  approves, the panel reopens at the same head so their owners disposition them.
  Only items first published after a qualified panel opening before the epoch
  (judged with the replaced primary and board), or published by the replaced
  primary itself, are carried; any other secondary-owned item still stops.
  Each epoch closes one window of history, judged by the board that governed
  it, and eligibility is fixed at an item's first publication. An item
  therefore stays carried through successive primary changes, and through a
  change that supersedes a pending one at the same round.
- **Managed CI.** Once a PR carries board-change records, historical
  rejected-recovery reviews and every round record an ordinary continuity
  ancestor references are checked against the seat binding of the lineage
  element that governed them (by comment position and round), resolved through
  the same validated lineage, including each link's recorded previous binding.
  Any lineage defect fails the chain. Current-head checks use the current board.

## Recorded plan board changes

Issue plan-review runs use the same board lineage as PR runs, kept on the issue
that owns the plan. `board_change.py` describes each surface with a small
`BoardSurface` descriptor. `PR_SURFACE` keeps the PR link marker, payload,
digests, audit text and round-metadata fields byte-identical. `PLAN_SURFACE`
uses the reserved `AGENT_PLAN_BOARD_CHANGE` marker (issue comments only), a
payload naming `issue_number` instead of `pr_number`, the planning policies and
no broad-path rules. `plan_board_state.resolve_plan_board_state` is the
read-only plan resolver, called by the issue-entry floor preflight and by plan
startup.

- **Approval classification first.** `plan_verification.classify_plan_approval`
  decides read-only, from history alone, whether the current canonical plan is
  provably completely approved. It runs before any configuration-dependent
  root migration, policy-drift check, link candidacy or planning floor, and
  returns `no-plan`, `unfinished`, `undetermined` or `approved`. Its inputs
  are the approving plan identity from the plan verification context (never
  the invocation's reviewers, planning policy or primary), the reconstructed
  latest round, and the reconciled ledger. That ledger includes a removed
  reviewer's reassigned items, and any open blocking, same-plan or machine
  item means not approved. The current signed requirement set is checked
  against each counted approval's requirement binding. Other inputs are the
  inherited-matrix admissibility of a planning child, any pending signed
  supersession, the execution-mode guard and the plan-growth gate. Any input
  it cannot derive means not approved. `build_plan_approval_inputs` derives
  the live inputs read-only for the issue preflight, managed-CI recovery and
  strict qualification. These are the issue and authoritative-parent
  requirement sets, and for a planning child its inherited admissibility and
  signed supersession state. Plan startup supplies the same fields from its
  own state. The ledger must also be reconstructible. A blocking or same-plan
  item that history published but that is neither in the resumed ledger nor
  provably cleared means not approved. Recorded reconciliation evidence, not
  the resume reader's conservative incompleteness flag, tells cleared history
  from a lost obligation. The rule reads the complete publication and
  reconciliation history: a primary epoch, a seat-binding change or a stale
  approval neutralizes approval evidence, never an obligation. The live
  approval boundary applies the same rule to its freshly re-read history and
  fails closed. With requirements surfaced, the
  plan's own coder dispositions must name exactly that set (a semantic patch
  inherits them from its authenticated assembled state), or the plan is
  `unfinished`. Approving reviewers never stand in for the coder's
  dispositions.
- **Requirement-binding evidence.** Reviewer records of generation-1 plans now
  persist `plan_candidate_key` (and, when acknowledged,
  `surfaced_reviewer_requirement_ids`) under every planning policy. Under
  `all-reviewers` these fields are evidence only. Full-board scheduling and
  approval never read them. A recorded key binds only when it equals the key
  recomputed against the current requirement set. A legacy keyless approval
  binds only when the plan's own versioned dispositions cover exactly the
  current set, even an empty one (an unversioned plan never proves its set),
  and every approval text carries the acknowledgement when that set is
  non-empty. With no requirement surfaced now, an approval text that carries
  the acknowledgement was given under a since-withdrawn requirement, so it is
  stale. Otherwise the binding is unproven and the plan is `undetermined`.
  Reviewer records of a staged child also persist the authoritative parent's
  signed requirement IDs (`parent_requirement_ids`). A counted approval binds
  only to the current parent set, and a record without that evidence binds
  only while the parent set is empty. The planning requirement set is the
  issue's own signed requirements merged with the authoritative parent's, for
  prompts, candidate keys, acknowledgements, the approval gate and the
  classification alike. The pre-implementation refresh still compares the
  issue's own set and the parent's set separately. The live loop never reuses
  a resumed approval whose recorded key, keyless legacy fallback or parent set
  is stale, so its reviewer reviews again. Staged carried approvals bind the
  parent set the same way. An acknowledgement recovered by repair is
  persisted as a new reviewer record wherever approvals carry this evidence,
  including generation-1 `all-reviewers` plans, so a restart or qualification
  binds it instead of reading the earlier unacknowledged record.
- **Legacy plain history.** Unanchored plain `all-reviewers` history proves
  membership by its reviewer posters, never completion. Without an issue-side
  implementation handoff bound to the current plan hash it is `undetermined`
  and resumes exactly as before, so a configured reviewer that never reviewed
  still reviews.
- **An approved plan is never reopened.** An `approved` plan takes no link,
  no planning-policy drift check and no planning candidate floor. The run is
  re-pointed at the approving board and enters the existing approval boundary
  with no planner or reviewer turn. That boundary still checks the exact key,
  inherited rows, supersession, growth, narrowing and the pre-implementation
  requirement refresh. A plan-only run reports the approved outcome. An
  implement-after-approval run restores the invocation's board, and the stage
  1 handoff rule alone governs the PR board and its floor. A board change
  recorded after approval is not activated. If the live boundary would
  invoke a reviewer or start a revision, the run fails closed instead. A
  staged restart of a completed round applies the recorded dispositions of
  reviewers paused on their carried exact-plan approvals, so items they
  resolved in that round are not reopened.
- **Plan root anchor and legacy root.** Substantive plan round records carry
  `plan_board_root` and `plan_board_identity`, stamped at the single
  round-metadata encoder (`set_plan_board_stamp`). Board identity fields are
  flow-scoped: `plan_board_*` fields on a PR-flow record, `pr_board_*` fields on
  a plan-flow record, or half a pair, decode as malformed. Unanchored history
  derives its root read-only, the same way as on a PR. Contract-neutral plan
  history only ever happened under `all-reviewers`, so its root policy is
  `all-reviewers` with no primary, whatever the invocation's planning flags.
  Resuming it as `primary-then-panel` is refused as drift before any post or
  agent invocation. The configured policy and primary seed the root only for a
  plan with no round records at all, so even coder-only legacy history keeps
  the all-reviewers root. Persisted rounds are never rewritten.
- **Neutral staged advances.** An identity-less orchestrator summary with phase
  `plan-phase-advance` is admitted read-only, including existing persisted
  history, when it claims no scheduler metadata, board fields, lineage digests,
  new findings, dispositions or evidence clearances. Carried prior items remain
  obligations. In index-ordered projected plan history, the immediately preceding
  contiguous same-subject round must be N before the advance to N+1 and contain
  reconciliation (including legacy authoritative reconciliation). Supported used
  scheduler openings and qualified reviews after reconciliation are permitted;
  they retain their own validation and obligations. Idle checkpoints removed by
  projection do not affect the boundary. Different subjects or rounds cannot be
  bridged, and an optional advance candidate key must agree with available keys
  in that segment. Only eligible advances are excluded from legacy seat-binding
  comparisons, including under the root or successive links; their bindings
  also cannot seed legacy root derivation. Binding structure remains validated.
  They cannot initialize seats, consume links or amendments, claim scheduling or supply
  approval. Raw history remains available; records after a still-pending link
  remain refused. This exception does not broaden amendment activation.
- **Plan board-change links.** For a plan that is not approved, a configured
  plan board that differs from the effective one is posted as one
  orchestrator-authored link on the issue. That covers membership, primary,
  backend, model chain and effort. The link is posted after the floor check
  and before any agent invocation, and an identical link is never reposted.
  Only links by the authenticated actor are admitted, through
  `admitted_issue_comments`. It reads REST author IDs and the actor whenever
  a plan link record is present, and an incomplete read fails closed. Links
  and signed plan amendments form one ordered chain. Plan records carry the
  governing element's digest. Changing the planning policy stays contract
  drift. The old refusals are gone: a named plan board change and a changed
  seat model are now recorded links. A signed plan amendment no longer needs
  contract-bearing history, because the root supplies the segment base.
- **Re-entry, approvals and ledger.** A link anchors at the highest recorded
  plan round. The anchor round's records are collected under the predecessor
  board and carried over (`carry_over_anchor_round` with `same-plan` status).
  The run then re-enters the effective round once, as a reviewer-only round on
  the byte-identical plan. Required plan reviewers are the effective board
  only. Added, rebound, restored and role-changed seats need an exact-plan
  approval from the effective round, applied through the staged carried
  approvals. Removed reviewers' unresolved items are reassigned by the derived
  plan ledger view. The audit and scheduler summaries carry a one-line
  board-change note.
- **Primary epoch.** Under `primary-then-panel`, a change of the primary seat,
  its binding or a seat's role sets a plan primary epoch at the link's
  effective round. Plan history before it is excluded from panel evidence,
  carried approvals, the primary stall streak and the step-back counters. So
  the next round is primary-only on the new primary, and the old primary's
  blocking reviews never trip `--plan-primary-stall-rounds`. Eligible
  pre-epoch secondary-owned items are carried obligations
  (`PlanSchedulerSnapshot.carried_obligations`). They are exempt only from the
  pre-panel safety stop and keep blocking approval.
- **Plan floor.** The issue-entry preflight branches on the classification.
  An approved plan resumed plan-only selects no board, so no floor applies.
  An approved plan resumed for implementation, or no plan, keeps the stage 1
  behavior: the handoff board rule against the approving plan context. Any other plan is resolved through
  `resolve_plan_board_state`, so a below-floor plan board change is refused
  before any agent invocation or comment post, with the filled-in signed plan
  amendment template. Plan startup repeats the resolution, and the floor is
  checked again at every planning round. Floor authorizations come only from
  signed plan segments. Once a round's reviews are in, before its outcome
  creates the next round (a planner revision or a reviewer-only phase-advance
  record) or reaches approval, again after each acknowledgement repair (an
  agent turn) before its result is published, and at each later round, the
  run re-reads the issue, re-admits link records
  through the authenticated actor and re-resolves the plan lineage.
  Inconsistent history, or any board change recorded while the run was
  active, refuses before another agent turn, next-round record or approval; a
  rerun resolves and applies it. The approval boundary's signed-requirement
  refresh uses the latest of these reads, so no agent turn separates it from
  approval.
- **Plan verification.** `derive_plan_verification_context` resolves plan
  history that carries a root anchor or a link through the plan lineage,
  never from the invocation's planning configuration. The approving identity
  is the board of the latest element that took effect. `original_reviewers`
  is that segment's base board. Every signed segment of that prefix keeps its
  frozen floor authorization at the handoff, also when a later link (for
  example a model-only rebind) follows it. A record carrying a board-change
  digest is always judged through the lineage, so a rootless digest with no
  admitted link fails closed on every recovery and qualification path.
  Approvals by removed seats, by added, rebound
  or role-changed seats before their effective round, and pre-epoch reviews
  are neutralized. A forged, non-orchestrator or digest-mismatched link, or a
  record carrying an unknown change digest, raises. Admitted comments and the
  actor are threaded through managed-CI fresh authorization, ordinary resume
  without an issue-side handoff, the handoff seam, strict managed-CI
  qualification and the issue floor preflight. Plans without links or root
  anchors verify exactly as before. Managed-CI no-handoff recovery and strict
  qualification also run the shared classification
  (`require_recoverable_plan_approval`) and refuse to bind a plan it does not
  approve. That covers a stale or unproven requirement binding, an open or
  unreconstructible obligation, inherited inadmissibility and a pending
  signed supersession.

## Approved resume routing

Approved-plan resume retains the approving board even across a recovered unused
neutral advance. Historical board-change links remain effective; the shortcut
log says that no **new** link is recorded. Pending post-approval links remain
inactive. The live approval gate must still qualify every approving seat before
restoring the invocation's PR board for execution.

## Decisions

[Decision log](decisions/reviewer-boards.md)
