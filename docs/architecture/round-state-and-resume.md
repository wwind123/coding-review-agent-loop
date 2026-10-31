# Round state and resume

Where workflow state lives and how a run resumes: the GitHub and local state table, durable comment carriers and write read-back, transient GitHub failure handling, the known host comment footer, evidence freeze records, rejected coder follow-up recovery, and the context and trust contracts that every prompt and resume path must preserve.

Source paths: `src/coding_review_agent_loop/approved_plan_resume.py`, `src/coding_review_agent_loop/review_approval.py`, `src/coding_review_agent_loop/round_state.py`, `src/coding_review_agent_loop/legacy_machine_items.py`, `src/coding_review_agent_loop/qualification_checkpoint.py`, `src/coding_review_agent_loop/managed_ci_result.py`, `src/coding_review_agent_loop/round_state_scheduler.py`, `src/coding_review_agent_loop/round_transport.py`, `src/coding_review_agent_loop/comment_rendering.py`, `src/coding_review_agent_loop/github.py`, `src/coding_review_agent_loop/github_issue_references.py`, `src/coding_review_agent_loop/github_retry.py`, `src/coding_review_agent_loop/protocol_markers.py`, `src/coding_review_agent_loop/partial_round_recovery.py`, `src/coding_review_agent_loop/publication_resume.py`, `src/coding_review_agent_loop/prompts.py`, `src/coding_review_agent_loop/memory.py`, `src/coding_review_agent_loop/architecture_context.py`, `tests/test_round_state.py`, `tests/test_round_transport.py`, `tests/test_protocol_markers.py`, `tests/test_pr_loop_resume.py`, `tests/test_pr_loop_closed_round.py`.

Scheduler metadata decoding and its key sets live in `round_state_scheduler.py`,
with explicit re-exports from `round_state.py`. Round entry points, encoding,
authentication and reconstruction remain in `round_state.py`; the extraction
preserves serialized bytes, legacy defaults and conservative invalid-record
handling, with no change to persistence or approval authority.

Pure issue-reference parsing and its evidence types live in
`github_issue_references.py`, with explicit compatibility re-exports from
`github.py`. The leaf depends only on the standard library and `errors.py`;
GitHub I/O, authorization, provenance decisions, publication/recovery entry
points and patched-name lookups remain in `github.py`. The extraction preserves
reference grammar, Markdown exclusions and existing import identities without
changing the trust or persistence contracts.

## Legacy PR acknowledgement recovery

`legacy_machine_items.py` owns legacy machine promotion and planning demotion,
re-exported by `round_state.py`. Before PR promotion, it indexes the earliest
durable appearance of each numbered acknowledgement candidate across both
`prior_items` and `new_items` in history order. Recovery transport carriers
(dispatch, rejected follow-up, and head-review recovery) are lineage-neutral
at this stage because their actor admission happens later; the next ordinary
carrier establishes origin. Transport alone cannot grant
authority, and a following reviewer origin still rejects recovery.
Only an eligible coder
checkpoint or Orchestrator summary with a consistent source round can establish
lineage. Coder evidence fields, including matrix capture classifications, do
not change producer eligibility; evidence-conversion handoffs cannot establish
an origin. Evidence authority still requires its separate actor admission.
Later trusted carries cannot authorize a reviewer origin; later
reviewer copies with the legacy untrusted-lineage diagnostic do not revoke an
already trusted origin.

Recovery requires the complete historical PR acknowledgement template and
consistent identity across copies. It accepts ordinary findings and precise
persisted unknown promotion shapes, including their valid authority/head,
lifecycle, empty ownership and diagnostic/reviewer-note combinations. Ordinary
blocking and same-PR copies may retain valid reconciliation owners, pending or
cleared owner states, dispositions and matching reviewer evidence. Promotion
may retain that evidence after clearing ownership; malformed fields still veto
recovery. Invalid
persisted fields, near matches, conflicting identities and unrelated unknowns
remain blockers. The unknown-obligation repair guard is unchanged.

Recognized aliases become the dedicated, head-independent acknowledgement
obligation. Each historical ledger coalesces duplicates and retains notes;
the original numbered IDs still contribute to the global numbering high-water
mark. Dispositions are rewritten only for aliases proven in that record's
prior ledger. Alias/dedicated duplicates coalesce in durable order, retaining
notes and preferring an unresolved disposition on conflict. Ordinary resolved
dispositions do not clear this machine obligation. If round-start coder
reconciliation already cleared the target, replay consumes only dispositions
for that canonical acknowledgement present in the completed review's prior
ledger; unrelated unknown IDs still fail validation. Normalization transforms
historical records in memory without adding old items into newer ledgers, so
later cleared checkpoints remain cleared on repeated restart. Existing comments
are never edited or deleted. Planning retains its separate ordinary-finding
and legacy-demotion behavior and receives no PR alias/disposition migration.

An exhausted acknowledgement repair publishes an explicit
`acknowledgement-repair-failed` boundary. Same-head resume preserves its round,
full carried ledger and the history-wide item-number high-water mark, including
round 1. Earlier reviews saw a different ledger and are rerun; the boundary is
not reconciliation or approval. Parallel-review peer visibility and partial-round
recovery use the same latest same-head, same-round boundary in the history being
read: earlier publications are historical context, and neither they nor the
boundary are deletion targets. Publications after the boundary remain peers.
It reopens scheduler checkpoint admission after
a prior reconciliation. A subsequent unrecorded push carries its active ledger,
round and history-wide numbering into a fresh review on the new head without
requiring the operator recovery override.

## Closed planning rounds

`plan_round_closure.py` projects admitted planning history before resume ledger
selection and scheduling. Reconciliation (including legacy authoritative
summaries) closes a subject's round and owns its settled ledger. A later
scheduler checkpoint is idle when no later admitted reviewer on that same
subject and round follows it. Idle checkpoints cannot replace the ledger,
open a panel, or supply approvals; reviewer-backed checkpoints retain their
historical opening authority under the existing candidate and approval checks.
Raw history remains available for board lineage and amendment-use validation.
Idle checkpoints retain negative candidate-key and invalid-metadata evidence
until a legitimate scheduler recovery checkpoint supersedes it. A later idle
matching checkpoint cannot restore intact history or completed approvals.
A legitimate force-full or contradictory-key recovery prelaunch retires its
selected reviewers' approvals from earlier rounds, including across restart.
This also covers primary-only unopened recovery without a force-full latch.
Completed same-round reviews survive repeated prelaunches, matching live reuse
and frozen publication replay.
Even an idle checkpoint carrying the first use of an amendment digest cannot
restore an operator force-full latch or scheduler savings; digest admission
and scheduling authority are separate.
An open next-round phase advance or prelaunch remains a resume anchor.

When a closed round needs fresh reviews, planning settles qualified findings
and dispositions and advances to the next round before spool preflight,
scheduling publication or invocation. Canonical plan bytes and assembled state
stay unchanged, with no planner turn. Superseded pre-panel claims establish no
obligation; an operator override sends that context only to its original author.
Finding identifiers retain their high-water mark. Round limits still apply:
without room for the next round, planning stops before fresh publication.
Qualifying completed approvals keep the existing shortcut. Unreconciled partial
rounds still require independent withheld spool outcomes or refuse new turns.

Planning recovery checks reconciliation before collecting deletion targets,
including when REST fails or is incomplete and the snapshot supplies history.
Closed rounds return no reviewer, summary or attachment deletion targets and
advise resuming fresh reviews in the next round. PR conversion recovery keeps
its existing boundary rules.

## State and Recovery

There is no application database or always-running server. State is split
between GitHub and local artifacts:

| State | Location and purpose |
| --- | --- |
| Source and candidate identity | Git commits/branches and live GitHub PR metadata. |
| Cross-invocation workflow history | GitHub comments with round metadata, canonical issue/PR handoffs, plan identities, and managed-CI intents; oversized payloads use sidecar comments. |
| Invocation results | Unique response files, each in a private per-turn directory removed after the validated turn ingests it, and external subprocess logs; validate the current attempt before accepting an artifact. |
| Parallel review-round spool | `<agent-memory-dir>/../private/review-round-spool/<round digest>/`; settled outcomes withheld until the round publishes (#1025, #1258). |
| Work in progress | Tool-owned per-run worktrees (removed at run end) or explicitly supplied checkouts; run artifacts (a relative `--log-dir`) live under `<cache>/run-artifacts/OWNER-REPO/<run-token>/`, a relative `--agent-memory-dir` under `<cache>/repos/OWNER-REPO/`, and worktree owner records under the host lock root; tracked diffs can be preserved as salvage on supported failure paths. |
| Local evidence | Test observations/receipts, runtime recommendations, usage summaries, and containment evidence. |
| Evidence-conversion authority | Capture classifications, conversion audits, attempt pairings and signed-evidence references in round metadata; admitted only from comments authored by the authenticated orchestrator actor, with REST identity hydration on resume and on every rereading consumer. |
| Orientation only | Repo-scoped memory, file inventories, and cached execution profiles; these may be stale. |

Durable comment carriers persist a bounded visible label outside the unchanged
hidden record, composed only from trusted vocabulary and already-authenticated
record fields, so labels are deterministic and historical marker-only comments
stay readable. The plan-validation diagnostic decoder accepts exactly two
canonical forms: the historical bare-marker body, and the marker introduced by
its own regenerated label. Every tool-owned comment write — create and update
alike — is verified by comparing the server's stored body for that comment with
the exact posted carrier and checking the producing login and ID, fetching the
comment when the write response carries no envelope; a rejected read-back takes
that seam's existing failure path rather than being accepted on an exit status.

Tool-owned GitHub writes never replay blindly after a transient (5xx or
transport) failure. Bounded retry (three attempts, backoff with jitter) applies
to read-only `gh` calls directly; every write — comments, issue creation, label
add and remove, ready transitions — first reconciles complete live state
(baseline-aware exact-body or identity matching by the authenticated actor,
event-history ownership, draft state and exact head) and fails closed with the
attempt history when that state is unreadable or ambiguous. A merge is never
replayed in-process: only an already-merged exact head counts as recovered, and
any other state ends the run so a rerun re-executes every merge gate.
Ambiguous-write recovery is distinct from cross-invocation spool replay, which
republishes a preserved reviewer acquisition from a frozen carrier. See
`docs/local_agent_loop.md` ("Transient GitHub failures").

The review-round spool is kept out of reviewer reach (#1560). Its root is
`private/review-round-spool/` beside the agent memory directory, not a direct
sibling of the prompt-advertised memory directory; `private/` and the root are
mode 0300 (never listable), round directories 0700 and record files exactly
0600 whatever the umask, so a record never reads back as malformed. Once at
invocation start for `issue`, `task`, `pr` and `managed-pr` (in
`establish_sandboxed_run`, before any agent turn or directory hardening) and
again at each review-round start, before any spool state is created or
migrated, both the new root and the legacy root
(`<agent-memory-dir>/../review-round-spool/`) are checked not to overlap, in
either direction, the response root, any provider checkout, any named reviewer
seat's workdir, or the repository checkout; an overlap raises. Both checks share
one read-only helper (#1565). A repository checkout that is exactly the
resolved home directory (a dotfiles work tree) is not part of the spool overlap
set; an unresolvable or root home keeps it, and the sandboxed response-root
boundary still uses the unexempted repository checkout. An existing legacy root is then re-moded to 0300 so retained
rounds cannot be listed (including the group-writable mode older releases
created under `umask 002`; a symlinked or foreign-owned legacy root raises),
and it is never created. A round directory found at
the legacy location is atomically renamed into the new root before its first
use and keeps its publication classification (fresh, carrier, legacy,
malformed). If both locations hold the round, or the rename fails, the run
fails closed instead of treating a possibly frozen carrier as absent. Residual:
these are same-UID mode bits that stop incidental discovery, not an
adversarial agent that computes the digest path from source.

Some hosts append a fixed footer to every posted comment (Claude Code cloud
sessions). The one byte-exact suffix `KNOWN_HOST_COMMENT_FOOTER` in
`protocol_markers.py` is tolerated once and nowhere else:

- Read-back accepts the stored body when it equals the posted body, or the
  posted body plus that suffix. Each write reports whether the footer was
  observed and the server's `updated_at`, never the comment's `created_at`.
- The suffix is stripped exactly once, where a raw comment body enters a strict
  reader. There are three such boundaries: `_authenticated_comment_from_rest`,
  managed-CI `_normalized_comment_body`, and plan-validation diagnostic
  recovery. Strict parsers never strip, so doubled or variant footers are still
  rejected.
- The managed-CI issue authorization parser requires the whole body to equal
  its canonical rendering, or the historical marker-only rendering.
- The base workflow admits a footered intent only through a raw-body match, and
  advertises that with `AGENT_LOOP_MANAGED_CI_HOST_FOOTER_V1`. Agent-loop's
  intent rediscovery and pre-dispatch gate mirror that workflow's page
  decision. The mirror returns `authorized`, `recoverable` (a lone `prepared`
  record), or `fail`, plus nonce-independent fatal comments.
- Before a dispatch, agent-loop renews the intent's `created_at` in place and
  checks it against the workflow's age window.
- Against an older workflow, an observed footer on an intent write raises a
  terminal, recovery-exempt error. That is `pre-dispatch` unless a dispatch was
  issued or a run attached.
- Observation is logged once per owned usage context.

Managed repair checkpoints can retain an optional typed failed-run reference
and a verified later-attempt result (#1709). The checkpoint type lives in
`qualification_checkpoint.py`, re-exported by `round_state.py`; shared immutable
result types live in `managed_ci_result.py`. Partial references, nonpositive IDs
or attempts, and contradictory head/base/obligation bindings invalidate a
checkpoint. Existing checkpoint shapes remain readable, and consumed watcher
extensions and qualification attempt identities retain their meanings. The
failed-head guard is unchanged: a repair checkpoint cannot create a qualification
candidate equal to its failed head.

For an older repair checkpoint without a reference, reconciliation reads
actor-authenticated history. It requires one completed intent between the
source round's authenticated reviewer records and its first matching repair
checkpoint. Repository, PR, target head, base and obligation must match, and the
workflow-mirrored intent validator must accept the record. Missing or ambiguous
history and model prose cannot establish correlation. The recovered reference
is observation evidence and grants no authority to dispatch an old generation.

A successful later attempt records settlement and its source audit linkage in
round metadata together with the remaining ledger and spent budget, before
readiness, publication or merge. A restart before that write observes again; a
restart after it reconstructs provenance and rechecks the live run, current
attempt, counted board, authorization and review/input identities. Recorded
clearance alone cannot qualify a head. If the live result is no longer valid,
the matching repair obligation is restored. Other blockers remain active, and
already completed merges use the existing post-merge recovery path.

An evidence freeze or release comment is a PR-round handoff boundary like a
qualification checkpoint. Its round metadata carries the ledger, round number,
round budget, signed-requirement baseline, and head-scoped clearances.
Reviewer records from an evidence-response pass count only once a terminal
freeze or release record follows them, so an interrupted pass is replayed in
full. A deferred request is re-derived from the reviewer record that raised
it. A malformed evidence record becomes the non-bypassable `unknown`
obligation.

Converted-freeze capture recovery uses existing admitted freeze, conversion and
coder records, with `followup_dispatch_head`, result subject and carried ledger
identities establishing its bound. Completed ordinary code turns also consume
the opportunity, including failed or subset capture and coder pushes. Missing
causal bindings or uncertain publication history preserve barriers and stop
conservatively; interruption before publication uses existing recovery without
an exactly-once process guarantee. A new independent terminal freeze can define
a later recovery boundary; a coder push cannot do so alone.

Historical derived approvals that contain active dispositions are excluded
before carried-approval and completed-review rereads. Their admitted ledger
evidence is retained and fresh review is required; historical comments are not
edited and arbitrary parse failures never become approvals. New evidence-only
approvals strictly round-trip while deferred evidence remains in the ledger.

A coder record's architecture identity describes its prompt acquisition. If its
result subject advances beyond that identity's candidate revision, it is not
selected as an observation of the new head. Fresh reviewer and checkpoint
observations remain authoritative for invalidation. Qualification diagnostics
identify the selected record's role, round and subject and all differing identity
fields. Empty pushes still change candidate revision authority even with equal
trees; legitimate architecture transitions invalidate approvals, and persisting
the fresh observation makes repeated unchanged gates stable.

A rejected coder follow-up is recoverable. A coder turn can push and
then be rejected by a post-response validator, which discards the response but
not the push. Before every PR coder dispatch the orchestrator therefore posts a
`coder-dispatch` summary (dispatch round and head, immutable ledger snapshot,
bounded round budget, attempt number, carried reasons); on a rejection after a
push, or any rejected recovery dispatch, it posts a `coder-followup-rejected`
summary keyed to the observed head with a bounded, redacted reason. Ordinary
selection never sees dispatch records. Resume admits these records only from
the authenticated GitHub actor, using author identities fetched from REST and
only when needed, validates each live record per phase before reading any
recovery field (a malformed live record stops resume before any agent call),
and uses one newest-first walk, shared by the current head and the latest prior
head, to pick the live handoff. Later coder, reviewer or head-review records
retire a handoff. A coder handoff resumes a coder round in its original slot
with the restored bounded budget and a prompt that names both heads and the
reason, up to `MAX_REJECTED_DISPATCH_ATTEMPTS`. Head-review recovery (a legacy
qualification-checkpoint anchor, or `agent-loop pr N --review-unrecorded-head`)
is an ordinary review round of the current head: it carries only the ledger, the
bounded budget and a `head-review-recovery` handoff that supersedes earlier
checkpoints, and never overrides review scheduling or merge-conflict routing.
Nothing force-pushes or writes a coder-role record on the operator's behalf.

A PR round closed by its reconciliation summary is final. It receives no new
scheduling or reviewer record; its only possible later record is the
digest-bound activation checkpoint of a signed reviewer-board amendment that is
effective from that round. When a resume re-enters a reconciled round and the
scheduler's selection needs a fresh reviewer turn (for example, a new signed
human requirement invalidates the primary's carried approval), the PR loop
settles the round's ledger by replaying that round's reconciliation record
(its carried ledger, dispositions and settled new items) through the ordinary
round-end settlement, then opens round N+1 under the normal round bound. If
the round still needs fresh turns while that activation is due, it is posted
first as an activation-only record that selects nobody. A reconciled resume
with no fresh turns continues in place and posts no scheduler checkpoint.
Recovery, qualification, evidence, conflict, conversion, board-change re-entry
and pre-panel history recovery resumes keep their own routes. A
`scheduler-prelaunch` record posted into a reconciled round (a stray checkpoint
from an older tool version, or an amendment activation) is inert: resume anchor
selection, visibility, panel evidence and every scheduler-state reader ignore
it, and only the amendment-lineage reader counts it. One exception keeps the
monotonic force-full latch: the startup latch reconstruction still reads a
lineage-visible amendment activation, the earliest record carrying its
amendment digest, so an operator latch recorded only on an activation-only
record survives a resume without the flag (#1559). A stray, whose digest is
null or repeats an earlier carrier, stays fully inert, and calls-avoided still
ignores every inert record. A conversion repeat or a
head-review recovery handoff reopens the round, so a checkpoint after one is
not inert. Nothing is deleted, so such PRs resume into round N+1 as they are.

A live, valid scheduler-prelaunch checkpoint for an open round N > 1 anchors
resume at N even when no coder or reviewer has published in that round. Its
ledger and finding-ID high-water mark survive; completed reviews stay empty
and the round is unreconciled. An older same-head qualification checkpoint
cannot replace that open round identity. Round-1 checkpoint-only startup keeps
its existing no-resume result without a qualification checkpoint, and its
existing qualification recovery when one exists, preserving reviewer-board
link fallback and named-seat rebinding. Pending board amendments still re-enter
their effective round. Dedicated recovery and closed-round boundaries retain
precedence. Qualification lookup helpers live in `qualification_checkpoint.py`
and remain re-exported from `round_state.py`.

Issue implementation delegates PR review to the same PR loop used by a fresh
PR invocation. Both use the repository, PR surface, PR number, recorded round
and current head as the private review spool identity. Restoring N therefore
reuses withheld responses without searching other heads or rounds. Fatal
reviewer failure and preflight contract refusal retain the spool; publication
cleans it only after success. Replay still validates responses and frozen
publication context, and refuses unsafe fresh turns when same-round peers are
public. The spool schema, private permissions and migration rules are unchanged;
recovery still requires the original host-local spool.

Resume reconstructs state from recorded evidence and then checks it against the
live PR, issue, plan, requirements, and policy. GitHub metadata is durable but
not a blanket authorization token. A lost local response or uncommitted patch
cannot necessarily be recovered from GitHub alone. Default checkouts can be
cleaned on rerun; inspect and preserve salvage before reusing them.

Salvage is partial implementation evidence, not approval or a guaranteed full
backup. Usage can be exact, partial, or estimated. Test receipts must retain
failure and scope caveats; a passing subset does not erase a broader failure.
A failure restored from an earlier run can be superseded at an unchanged head
and tracked tree only by a gated cross-restart rule. That rule matches commands
by a broker-computed digest over the checkout-relative cwd and raw argv, never
compares environments or persists environment material, and leaves the visible
caveat `superseded across a restart; environment not compared` on the failure.

## Context and Trust

The initial prompt does not contain the whole codebase. Agents receive a
verified checkout, task/PR identity, available plan and human requirements,
feedback, checks, and bounded advisory memory. Reviewers are told to inspect
the complete base-to-head diff and use read-only source exploration for related
code. There is no automatic semantic dependency-retrieval engine.

The generated memory architecture map is primarily a directory/file inventory,
not the architecture set and not a semantic model of the repository. Prompts may embed
a bounded revision-bound snapshot of `ARCHITECTURE.md` as advisory context;
missing, unsafe, or opted-out context preserves the legacy path. `full` versus `compact` review
context controls review-history presentation, not whole-repository ingestion.

Architecture snapshots are acquired from regular blobs in committed Git
objects, with literal repository-relative lookup, bounded UTF-8 reads, and
immutable revision/blob/hash identity. PR reviews show established base and
candidate documents separately; candidate prose is a proposal and cannot
replace the baseline. The snapshot is untrusted orientation, never a remote
fetch permission, protocol authority, whole-codebase audit, or correctness
guarantee. Fresh review acquisitions revalidate the candidate, target,
merge-base, and document identity at qualification and merge gates; each
review prompt uses the immutable acquisition frozen for its current round.

Key contracts to preserve when changing the implementation:

- Validate acquisition, structured fields, footer state, IDs, and provenance
  before treating a response as an authoritative workflow result. Repair is
  bounded format correction and is revalidated, not a new substantive review.
- Signed human requirements (`-- Human Reviewer`) are honour-system unless
  `--human-reviewer-trusted-actor LOGIN:ID` entries are configured. With a
  trust set, `github.py` admits a signed issue body, issue/PR comment, or PR
  review only after joining it by exact locator (URL, or review node ID) to its
  live REST record, binding the REST body/author to the loaded text, and
  matching both login and numeric user ID. Otherwise the record is excluded
  with a logged diagnostic, and an incomplete read or a drifted record fails
  closed. Unconfigured, no extra reads happen and prompts label every author
  unverified. Verification fields are provenance only, and requirement IDs do
  not depend on them. Reviewer-board amendments, child-disposition overrides,
  and child-plan supersessions are not covered and remain textual.
- Keep untrusted issue/PR prose and model output separate from tool-owned
  protocol markers. A marker or model signature alone is not authentication.
  Untrusted text that merely names a reserved token is neutralized into the
  registry's descriptive label when rendered into prompts, never refused.
  Rejection is narrower than neutralization: record-shaped spans fail closed at
  the pull-request-body authorization gate, in a current agent response, and in
  tool-owned publications. Other untrusted surfaces are neutralized, and
  authority there still depends on the typed record parsers' own checks.
- Re-check the live head and relevant qualification identities at finalization;
  old approvals and CI from another head must not authorize a new one.
- Do not equate missing required input, interrupted commands, or infrastructure
  failures with approval. Distinguish them from actionable code defects.
- Permissions and resource containment are different controls. Permission bypass
  flags do not make fetched content or agent commands trustworthy. Sandboxed
  role grants never widen with operator pass-through arguments, and a failed
  pre-spawn boundary or provenance re-check is never retried with the same
  grant.

These are design contracts and validation responsibilities, not a claim that
every integration path is bug-free. Use source inspection and regression tests
to check them when a change crosses module boundaries.

## Approved planning resume

For completely approved planning history, `approved_plan_resume.py` supplies a
read-only predecessor view when the final record is a single unused same-plan
reviewer-only advance immediately after reconciliation. Independent complete
approval must still hold against the full history. A later reviewer, candidate
or scheduling checkpoint is substantive and prevents this recovery. The raw
comments remain available to lineage and amendment validation; no persisted
format or global rollover projection changes.

The predecessor's ledger and current-round dispositions remain authoritative.
Existing lifecycle replay reconstructs future follow-ups from the predecessor
and earlier rounds, preserving identities, provenance and numbering while
excluding resolved or promoted deferrals. Conflicting provenance or
unreconstructible obligations fail closed. Direct reconciled resume uses the
same future-item replay. Existing publication policy and follow-up deduplication
apply to both views: plan-only summarizes; implementation may file issues once;
ignore retains its existing summary behavior. Non-approved rollover semantics
are unchanged.

## Decisions

[Decision log](decisions/round-state-and-resume.md)
