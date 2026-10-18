# Round state and resume

Where workflow state lives and how a run resumes: the GitHub and local state table, durable comment carriers and write read-back, transient GitHub failure handling, the known host comment footer, evidence freeze records, rejected coder follow-up recovery, and the context and trust contracts that every prompt and resume path must preserve.

Source paths: `src/coding_review_agent_loop/round_state.py`, `src/coding_review_agent_loop/round_transport.py`, `src/coding_review_agent_loop/comment_rendering.py`, `src/coding_review_agent_loop/github.py`, `src/coding_review_agent_loop/github_retry.py`, `src/coding_review_agent_loop/protocol_markers.py`, `src/coding_review_agent_loop/partial_round_recovery.py`, `src/coding_review_agent_loop/publication_resume.py`, `src/coding_review_agent_loop/prompts.py`, `src/coding_review_agent_loop/memory.py`, `src/coding_review_agent_loop/architecture_context.py`, `tests/test_round_state.py`, `tests/test_round_transport.py`, `tests/test_protocol_markers.py`, `tests/test_pr_loop_resume.py`, `tests/test_pr_loop_closed_round.py`.

## State and Recovery

There is no application database or always-running server. State is split
between GitHub and local artifacts:

| State | Location and purpose |
| --- | --- |
| Source and candidate identity | Git commits/branches and live GitHub PR metadata. |
| Cross-invocation workflow history | GitHub comments with round metadata, canonical issue/PR handoffs, plan identities, and managed-CI intents; oversized payloads use sidecar comments. |
| Invocation results | Unique response files and external subprocess logs; validate the current attempt before accepting an artifact. |
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

An evidence freeze or release comment is a PR-round handoff boundary like a
qualification checkpoint. Its round metadata carries the ledger, round number,
round budget, signed-requirement baseline, and head-scoped clearances.
Reviewer records from an evidence-response pass count only once a terminal
freeze or release record follows them, so an interrupted pass is replayed in
full. A deferred request is re-derived from the reviewer record that raised
it. A malformed evidence record becomes the non-bypassable `unknown`
obligation.

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
it, and only the amendment-lineage reader counts it. A conversion repeat or a
head-review recovery handoff reopens the round, so a checkpoint after one is
not inert. Nothing is deleted, so such PRs resume into round N+1 as they are.

Resume reconstructs state from recorded evidence and then checks it against the
live PR, issue, plan, requirements, and policy. GitHub metadata is durable but
not a blanket authorization token. A lost local response or uncommitted patch
cannot necessarily be recovered from GitHub alone. Default checkouts can be
cleaned on rerun; inspect and preserve salvage before reusing them.

Salvage is partial implementation evidence, not approval or a guaranteed full
backup. Usage can be exact, partial, or estimated. Test receipts must retain
failure and scope caveats; a passing subset does not erase a broader failure.

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

## Decisions

[Decision log](decisions/round-state-and-resume.md)
