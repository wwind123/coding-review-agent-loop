# Managed CI and merge

How agent-loop observes CI and merges: the counted check board, managed exact-head CI and its hosted workflow boundary, managed-label handling, base-workflow dispatch authorization, sharded and split layouts, integration targets, preflight 403 fallbacks and protection states, issue-created managed-CI authorization, the authorization continuity chain across coder, merge-conflict, CI-repair and rejected-coder recovery heads, and the coder CI-dispatch guard.

Source paths: `src/coding_review_agent_loop/checks.py`, `src/coding_review_agent_loop/ci_health.py`, `src/coding_review_agent_loop/managed_ci.py`, `src/coding_review_agent_loop/managed_ci_reconciliation.py`, `src/coding_review_agent_loop/managed_ci_result.py`, `src/coding_review_agent_loop/managed_ci_status.py`, `src/coding_review_agent_loop/pr_loop_plumbing.py`, `src/coding_review_agent_loop/managed_ci_recovery_commands.py`, `src/coding_review_agent_loop/managed_ci_intent_validation.py`, `src/coding_review_agent_loop/managed_ci_bases.py`, `src/coding_review_agent_loop/managed_pr.py`, `src/coding_review_agent_loop/coder_ci_guard.py`, `src/coding_review_agent_loop/integration_close.py`, `src/coding_review_agent_loop/github.py`, `.github/workflows/ci.yml`, `.github/workflows/managed-ci.yml`, `ci/managed/ci_shard_verify.py`, `tests/test_managed_ci.py`, `tests/test_ci_sharding.py`, `tests/test_managed_ci_integration_bases.py`, `tests/test_pr_loop_ci_gate.py`, `tests/test_pr_loop_managed_ci.py`, `tests/test_pr_loop_merge.py`, `tests/test_pr_loop_coder_ci_guard.py`.

## CI and Merge

Recovery command rendering lives in the pure `managed_ci_recovery_commands.py`
sibling, and workflow-mirrored intent envelope, schema and lifecycle validation
lives in the pure `managed_ci_intent_validation.py` sibling. Both import
independently without a back-reference to `managed_ci.py` or the orchestrator.
`managed_ci.py` explicitly re-exports their definitions and catalogs, retains
all GitHub I/O, freshness and dispatch authorization gates, and keeps callers
of patched names in their original namespace. Persistence formats and
authorization decisions are unchanged by this pure move (#1692).

Ordinary CI observes the current-head check board. Known failures can join
review feedback without waiting for unfinished checks. Post-approval watching
and automatic merging use bounded gates, including infrastructure-failure
classification and fresh mergeability checks.

Post-approval failures are classified before any coder routing (#1477). The
plumbing catalog lives once in `managed_ci.py` (`MANAGED_PLUMBING_JOB_NAMES`
plus `final-ci/exact-head`, extended only by `--ci-plumbing-check`). Matching
is by exact whole name or final ` / ` segment; a workflow-contract test pins
the job names to the managed workflows. `ci_health.classify_watch_failure` is
pure. It gives test failures precedence, and it requires an authoritative
complete board (query `ok`, complete listing, readable protection, no missing
required check) before it reports plumbing-only. Plumbing that failed while
tests still run is held, not routed. `github.watch_pr_checks` also confirms
stuck-pending required statuses: their target-URL Actions run reads
`completed` on two consecutive polls. Such statuses are plumbing by cause. It
polls through evidence from before a rerun. Every rerun decision reads the
producing run's complete job listing (`github.read_run_jobs`,
`ci_health.classify_run_jobs`), never its aggregate conclusion, and one budget
decision gates watcher and approval-snapshot reruns alike, before the shared
watch budget is spent. The ordinary route re-runs the one resolved PR-head
run's failed jobs once (`rerun_ordinary_failed_jobs`). The managed route also
requires an authoritative complete PR-head board
(`managed_ci.classify_managed_board`), and a board or job test failure routes
to the coder even when no status was published. The
managed route re-runs the attached v2 run fully (`rerun_managed_attempt`),
only after re-reading the authenticated audit record's completed `no-status`
transition for the current attempt, because the split publisher rejects
carried-over validation. A managed failure that published a failure status is
refused. Second plumbing failures and refused reruns end in
`pr_loop_support._stop_on_ci_plumbing_failure`. That stop records no coder
obligation, attempts no merge, and reports the run URL and the
`final-ci/exact-head` state. The rerun budget is invocation-local per head and
adds no persistence. `--watch-pending-ci` stays opt-in; an ordinary
non-auto-merge pending stop names it as a hint.

The counted board is the current-head board minus check runs positively
identified as ad hoc `workflow_dispatch` runs (operator diagnostics such as a
deliberate negative-evidence dispatch). `github.get_pr_checks` applies one
per-observation predicate: a check run is excluded only when it comes from the
GitHub Actions app, its own `check_suite.id` belongs to a same-repository,
non-managed `workflow_dispatch` run on the head, and any URL-derived run id
agrees. The managed qualification run (`managed-ci-v2 nonce=`) and every commit
status context, including `final-ci/exact-head`, always count. When a candidate
exists, the board is rebuilt from the paginated `filter=all` history (decoded
without `--slurp`); the predicate runs before any collapsing, only fully
identified same-suite reruns are collapsed, and conflicting or identity-less
observations stay independent so they still reach `shadowed`. If the lookup
fails nothing is excluded; if history is unusable while a candidate exists the
board is non-authoritative (partial query, unavailable unless failing) and can
neither pass the gate nor qualify managed success. Exclusions are recorded in
`PullRequestChecks.excluded`/`exclusion_notes`, rendered separately and logged.
Cost: one dispatch-run read per fetch whenever Actions check runs are present,
plus the history read only when a candidate exists.

Managed CI is a repository-integrated alternative: intermediate runs may be
suppressed, then a reviewed candidate receives an authenticated final workflow
dispatch. Version-2 intent records track generation, nonce, run, and attempt.
A fresh process never adopts an earlier generation's attempt; a matching
`qualifying` checkpoint resumes as `qualification_ready` and dispatches a fresh
exact-head attempt instead of re-running reviewers.
Qualification requires correlated exact-head evidence and the full counted
check board, not a model's statement that CI passed. Head changes invalidate the
proof. Automatic merge uses `--match-head-commit`; explicit managed CI can
qualify for manual merging instead.

A repair checkpoint has a separate observation path in
`managed_ci_reconciliation.py` (#1709). `FailedManagedRun` binds the original
completed audit record, nonce, run, failed attempt, generation and workflow
revision to the repository, PR, target head, base and obligation. Before repair
routing and immediately before a coder handoff, a bounded read can verify a
later successful attempt of that same run. It uses the shared status predicates
in `managed_ci_status.py`, validates the original authenticated record and
workflow actors/ref/revision, follows the run's current positive attempt, and
requires the complete passing counted board and live managed authorization.
The workflow run's default-branch head is distinct from the target PR head.
Pending, incomplete, excluded, superseded or unrelated observations cannot
settle the obligation. A pending or unreadable rerun stops before another
coder handoff, retaining the obligation and its budget. Observation never
renews an audit record or dispatches CI.

`VerifiedManagedResult` is immutable result provenance, separate from the fresh
activation contract. Settlement persists it with the remaining ledger before
finalization. Both `prepare_v2_merge` and manual qualification accept it through
an explicit optional `verified_result` argument and re-observe it before their
side effects. The settlement evidence, manual qualification record and
`ExactHeadCiProof.managed_result` name the observed nonce, run, successful
attempt, original generation and source audit-record reference. The fresh
contract continues to govern live authorization, protection, label ownership
and cleanup; its nonce, attachment, attempt, generation and intent/audit
references are never replaced with historical result identity. Manual publication
keeps its publication receipt separate on this path. Existing approval, plan,
requirements, architecture, evidence and head-guarded merge gates still apply,
and settlement clears only the matching managed obligation. Unrelated qualifying
checkpoints retain the fresh-generation dispatch behavior above. Post-approval
plumbing routing is extracted into `pr_loop_plumbing.py` with the loop's existing
dependencies supplied explicitly; retry policy is unchanged.

The installed `CI` workflow is the hosted execution boundary for this
repository's managed route. Its literal v2 and unlabeled-recovery declarations
activate a four-event pull-request matrix: trusted same-repository managed
draft openings may be suppressed before the label race settles,
`synchronize`/`reopened` suppression requires the active managed label, and
`unlabeled` always restores ordinary Python 3.12 CI. Forks, malformed or
unavailable payloads, trust mismatches, non-drafts, and non-managed branches
fail open. Pushes to `main` and all-empty manual dispatches remain full-suite
paths; label-addition, readiness, and draft-conversion events are deliberately
not subscribed to.

Explicit manual qualification keeps the managed label on the PR it readies, so
no `unlabeled` event re-runs ordinary CI on the qualified head. A ready PR is
never suppressed, and `unlabeled` still restores ordinary CI whenever the label
is removed. Publication checks twice, before readiness and before the audit
record, that the active label event is the contract's event and actor. Any
publication failure releases the label, whether the PR is still draft or
already ready, and regardless of the orchestrator's interrupted-run
preservation. The label is removed when the event is owned or unprovable
(fail-open). A readable label event from someone else is left and reported.
The next agent-loop invocation removes a retained label from an open ready PR
at PR-loop entry, in the bootstrap directory, before lifecycle authentication
and workdir setup. Every lifecycle classifier therefore still sees only the
draft/labeled and ready/unlabeled states. A re-entered run that aborts before
publication is kept draft/labeled for exact resume, like any other interrupted
managed run, except when managed-CI activation itself fails. In that case the
label may be released under the fail-closed rule, and a ready/unlabeled PR that
this run's acknowledged draft conversion drafted is restored to ready/unlabeled
only under the guarded, read-back-verified restoration rule. Only an activation
failure that follows a mutation recorded by this run (a label release attempt,
a draft conversion, or a label POST attempt) carries the measured before/now
report. Pre-write refusals (identity or adoption `None` returns, the strict
pre-write gate, and the strict draft/unlabeled re-entry refusal) make no write,
carry no report, and leave the PR as found. Residual: if a human converts a qualified PR back to
draft and pushes, that push is suppressed until the label is removed. Strict
protection still requires `final-ci/exact-head` for the new head.

Managed dispatch is authorized in base-workflow code. It resolves the
configured `AGENT_LOOP_MANAGED_ACTOR` to the live identity, requires both the
initiating and re-run actors to match, validates the live PR and current base
workflow revision, and accepts exactly one fresh generation-scoped handoff
record. Freshness is bounded to 15 minutes of age with a 5-minute future-skew
allowance, and a no-status retry must be the immediate next attempt of the same
run after its recorded terminal attempt. Only after validation does the
exact-head job receive the target SHA;
it verifies checkout, installs editable development dependencies, and runs the
complete pytest suite once, split across three duration-balanced shard legs
that each test the same validated SHA. A small aggregate keeps the
original `Test validated exact head` check name: it fails unless every shard
succeeded, using a workflow-literal result gate before any repository code,
and then runs the callee-owned `ci/managed/ci_shard_verify.py`, checked out
from the reusable workflow's own repository and revision, to prove from the
per-shard manifests that the shards covered the full collection exactly once.
The whole boundary is a reusable `workflow_call` workflow
(`.github/workflows/managed-ci.yml`, plus a read-only
`managed-ci-ordinary.yml`) called by a thin `ci.yml` whose caller jobs grant
exactly the permissions the callee jobs need; the dispatch must execute from the
live default branch, whose head the intent's `workflow_revision` is bound to.
The ordinary `Python 3.12 full suite` check is sharded the same way.
Split layout: an adopter whose tests need services or a routed matrix
composes `managed-ci-validate.yml` (read-only; its validator step is byte-identical
to the single-job workflow), caller-owned read-only test jobs, and
`managed-ci-publish.yml`. Every test job checks out the literal validated SHA,
asserts `HEAD`, and ends with the pinned `managed-ci-attest` action. Publish has a
read-only `verify` job (current-attempt attestation artifacts, the jobs API
conclusion per expected job name, and the caller's `toJSON(needs)`, against a
literal expected set declared in the default-branch caller) and a status job that
alone holds `statuses: write`. The status job re-checks that the forwarded
inputs, target SHA and nonce equal the dispatch event, and that validation ran in
this very run attempt (the caller forwards validate's `run_id`/`attempt` outputs, so a
validate job carried over by "Re-run failed jobs" never publishes), before feeding
`needs.verify.result` to the unchanged status builder. Attestations are untrusted
correlation claims; trust rests on the default-branch composition and API
conclusions. The validator, driver and single-job workflows are unchanged.
Integration targets: the live PR base must equal the intent's
`base_ref` and be the default branch or match the owner-controlled
`AGENT_LOOP_TRUSTED_BASES` repository variable (exact names or `prefix/*`,
parsed by one matcher block shared verbatim by the workflow and
`managed_ci_bases.py`; unset or invalid trusts the default branch only). The
workflow, its revision and the dispatch ref always come from the default
branch (`ManagedCiContract.dispatch_ref`), while `base_ref` carries the real
base, so a branch cannot supply the workflow that qualifies it. Startup (and
`managed-ci preflight`) refuses an untrusted, unreadable or unsupported
(`AGENT_LOOP_MANAGED_CI_TRUSTED_BASES_V1` absent) non-default `--base` before
any agent work. `prepare_v2_merge`, manual qualification and the merge call
re-read the live repository/base/head and re-check trust, releasing the
qualified label on a retarget or revocation; GitHub's merge API guards only
the head SHA, so the final retarget race is narrowed, not eliminated. After a
confirmed merge into a non-default base the child issue is closed
(`integration_close.py`), and an interrupted closure is retried from the
authenticated MERGED PR by resume and `phase_progress` without replaying the
merge. CI-level changes only on an integration branch do not apply to its PRs
until they land on the default branch. An
always-evaluated publisher writes the
`final-ci/exact-head` status only for that validated SHA, correlating nonce,
run ID, attempt, and Actions URL. Authorization failures before target
establishment produce no status; downstream checkout/test failures cannot
produce success.

Every managed-CI GitHub API read and status POST goes through one bounded
retry helper, the `MANAGED_CI_HTTP_RETRY` block (#1476). It is inlined
byte-identically in the validate, verify and publish steps, outside the pinned
publisher, guard and validator blocks. Transient failures retry with backoff:
network, timeout, 5xx, 429, a rate-limited 403, and an empty or truncated body.
Only their exhaustion counts as infrastructure. A definitive 4xx is a
rejection after one call. When initial validation cannot read the API, no
status is written, because there is no validated target. The job fails with a
distinct `managed-ci infrastructure` annotation and the existing no-status
retry path recovers. When an authorized split target's verify jobs listing is
exhausted, the publisher posts `error` instead of `failure`, and it chooses
`error` only after the correlation guard passes. A status POST that still
fails after retries fails the job with the URL and HTTP status. Having the
loop's CI watch classify these runs as infrastructure is a companion loop-side
change. See
[GitHub API retry and outcomes](../local_agent_loop.md#managed-ci-github-api-retry-and-outcomes).

This adds no application persistence or public Python API: the handoff record,
workflow run metadata, and terminal status remain GitHub audit records, while
the local producer continues to own lifecycle creation and exact-head merge
guards. The workflow's read-only validation boundary and narrowly scoped
status-write permission are separate from the ordinary test job. Until the
sole-maintainer rollout is completed, the queue stays on ordinary CI; the
explicit unprotected waiver for `wwind123` is per invocation and is not a
default change, existing-PR adoption grant, or substitute for head-guarded
merge enforcement.

Machine-owned CI failures and reviewer-owned findings are different kinds of
obligation. Their reconciliation must preserve CI authority without preventing
fresh qualification of an approved correction. This boundary has a known
selective-policy integration gap tracked in [#776](https://github.com/wwind123/coding-review-agent-loop/issues/776).

The unprotected managed-CI waiver is a voluntary tool gate, not a replacement
for GitHub enforcement. Historical audit markers cannot grant fresh authority;
the live actor, repository, PR lifecycle, and provenance must be revalidated.
See [managed CI](../local_agent_loop.md#managed-exact-head-ci).

Two preflight reads can be refused where the rest of managed CI works, for
example in Claude Code cloud sessions. Each fallback is keyed to a
strict HTTP 403, taken only from gh's own trailing stderr status line; response
bodies are never searched for status digits.

- A refused `AGENT_LOOP_MANAGED_ACTOR` read lets `--managed-ci-trusted-actor`
  stand in as an *asserted* actor at every identity site. The authenticated
  login must still equal it. The boundary is the workflow: it enforces the real
  variable server-side, so a wrong assertion leaves ordinary CI unsuppressed
  and fails managed dispatch validation rather than failing open. A readable
  variable that differs, a 404, or any other error keeps failing closed.
- A refused classic-protection read (`required_status_checks`, or
  `enforce_admins` behind a required context) makes protection indeterminate,
  not absent. It becomes the distinct `unreadable` state only when the
  effective rules were fully inspected: a list, or gh's strict 404/422
  no-rules response. It also requires that no ruleset proves strict
  enforcement. After such a 403, a ruleset proves strict only when it requires
  exactly `final-ci/exact-head` and its `bypass_actors` are visible and empty.
  Malformed rule or ruleset data, and any other rules failure, stay
  `indeterminate`, which nothing can waive. On readable classic paths only
  the context matching changed: it is now an exact
  `required_status_checks` match, which can only remove a false `strict`.

Readiness maps `voluntary`, `plan_limited` and `unreadable` to
override-eligible without consulting invocation flags. Only the authorization
gates consult `waivable_protection_states(config)`: creation, fresh
authorization, resume, activation, recovery, and fresh-command advice.
`--allow-unprotected-managed-ci` waives `voluntary` and `plan_limited`, and
`unreadable` additionally requires `--allow-unreadable-protection`. The
legacy non-suppressing workflow exception stays a separate clause.
Authorization records derive their persisted waiver from protection through
one mapping (`allow-unreadable-protection` for `unreadable`), and the parser
rejects crossed pairings. The override audit trailer keeps its field set and
simply carries `protection=unreadable`. Issue-created recovery restores the
persisted state read-only. A selection pass validates every actor-authored
authorization record with protection and plan-scope comparisons deferred, and
the creation records supply the state. A confirmation pass then requires every
record, and the live assessment, to agree. A base that has since become
`strict` also refuses, because activation's strict path does not run the
resume-audit plan-scope gate that recovery deferred to. Plan-scope checks are enforced by
the activation-time resume-audit gate, after the canonical plan is bound and
before any mutation.

Issue-created managed-CI authorization is a durable orchestration protocol,
not coder evidence. Once a structurally accepted implementation response
exposes a PR number, the orchestrator authenticates the live repository, issue,
PR, actor, base, reserved branch, exact head, and managed-label event, then
persists a versioned authorization record only as an actor-authored PR comment
before validating reported test locations. Rejected test evidence remains
rejected and cannot produce handoff, approval, readiness, dispatch, or
qualification state. A response rejected before the PR number is accepted is
never parsed for authority; recovery on a voluntary or plan-limited base
requires the explicit fresh operator grant in issue or PR mode, with PR mode
naming the issue scope. A strictly protected base instead uses ordinary
same-PR issue/PR discovery and resume; a draft/unlabeled re-entry additionally
requires the authenticated strict PR tuple and an actor-owned historical
managed-label event before the suppression label is reapplied. It does not
mint or accept an unprotected grant merely because the waiver flag was supplied.
PR-mode recovery fetches that issue from GitHub, resolves any canonical
approved-plan identity from its durable comments (for a fresh decomposition
child, from the staged parent its own fresh-phase identity names and
authenticates, never from PR body text), and requires a server-side
issue timeline association to the exact PR; candidate-authored closing text is
not the source of authorization. When no PR-side closing-contract record exists,
the authenticated issue/plan scope supplies the expected set, and the current
body is checked for unapproved same-repository closing references before managed
activation can write the suppression label.
Creation and fresh-authorization publication re-read the live tuple, managed
label event, and authorization-comment state immediately before writing, and
fresh grants bind the observed voluntary or plan-limited protection state.
Legacy PR discovery during explicit managed issue recovery does not synthesize
a canonical handoff from an implementation response that failed validation.
Fresh-authorization guidance is emitted only for an authenticated
issue-created missing/stale authorization with a known issue scope and explicit
unprotected waiver. Label-event, protection, tuple/publication, intent-ledger,
adoption, and source-managed failures retain their own prerequisite-specific
diagnostics rather than advertising an inapplicable grant.

Automatic authority across a coder repair head is a narrow execution/data-flow
transition: the orchestrator must have persisted the blocking round, invoked
the coder for that round, validated the response and assigned-checkout head
advance, re-read the exact live PR tuple, and persist one continuity comment
binding predecessor head, new head, predecessor authorization, and round
metadata identities. Those referenced comments are parsed again on resume and
must be newer than the predecessor authorization, with the blocking reviewer
record for the predecessor head ordered before the coder record for the
immediately following round and new head. Resume accepts only
one unique gap-free chain. Missing,
forked, stale, unsolicited, or raced links fail closed and direct an eligible
operator to fresh authorization. PR bodies, branches, labels, draft state,
commits, and coder-authored comments remain corroboration only; they are not
authority or a persistence substitute.

The same chain is also a PR-qualification input. Managed recovery does not
synthesize the issue-side handoff, so when a managed approved-plan PR has none,
the fresh qualification snapshot reads the plan binding from the PR side
instead. On a voluntary or plan-limited base it re-reads the PR comments
and requires every authorization record to be authored by the configured trusted
actor (the comment author ID must match the record) and to name this
repository, issue, PR, and base. It also requires exactly one distinct terminal
at the live head. The only accepted exception follows resume: a single fresh
grant may supersede the creation record at the same head. That terminal must
link through round-backed continuity records to a creation or fresh root, the
chain must not fork, and every record in it must carry the approved plan hash.
Records retired by a later fresh grant or continuity record
are history and are ignored by this check; a continuity record retires only
the actor's otherwise compatible records that cannot reach its new head, never
its own predecessor chain.
A strictly protected base publishes no authorization record, so the binding
comes from the same sources its resume used: the reserved managed branch for
the issue, plus the issue's completely approved canonical plan, whose hash must
still equal the bound plan. Any gap fails qualification closed. Runs that do
have an issue-side handoff, and all non-managed runs, keep the issue-side check.

Two automatic transitions have no reviewer to correlate: the merge-conflict
resolution round and the exact-head CI repair round. When the live head conflicts with the base branch the
orchestrator skips reviewers by construction and routes the round to the coder,
so no blocking reviewer record for the predecessor head can ever exist. That
head advance is authorized instead by the tool-owned merge-conflict obligation:
exactly one coder round metadata record for the new exact head and the
immediately following round, authored by the bound actor, newer than the
predecessor authorization, carrying the orchestrator-minted machine-authority
merge-conflict obligation in its round items. The obligation counts only with
machine authority, an open `blocking` or `same-pr` status, and a
`repair_required` (as minted) or `awaiting_current_head_review` (advanced for
the pushed head) lifecycle; a resolved, deferred, cleared or qualification-stage
item does not, and its raw fields are type-checked before comparison (#1057).
That obligation is minted by the
tool and sits outside the coder's classifiable item namespace, so an agent
response cannot introduce it. The continuity comment binds that single record,
and resume reauthenticates the same shape rather than a reviewer pair that never
existed.

The exact-head CI repair round (#1024) is the second such transition. When
managed exact-head CI fails on a head every reviewer already approved, the
failure comes from CI rather than a reviewer, so the coder round that repairs it
has no blocking review to pair with. That head advance is authorized by the
tool-minted CI obligation in the state the coder record carries after the push:
exactly one coder round metadata record for the new exact head and the
immediately following round, authored by the bound actor, newer than the
predecessor authorization, carrying a machine-authority `managed-exact-head-ci`
or `github-pr-checks` obligation in `awaiting_current_head_review` whose failed
head is the predecessor head and whose candidate head is the new head. The
orchestrator mints the obligation and advances its lifecycle and heads; an agent
response cannot introduce it or set those fields. Resume reauthenticates the
same shape, bound to both heads of the transition.

A rejected-coder recovery across a fresh grant (#1367) is a third shape, with a
real reviewer. A coder follow-up pushes a stranded head and is rejected, an
operator fresh grant is then minted at that head, and the recovery dispatch
pushes a new head. The fresh grant's immutable predecessor link names the
original authorization, so continuity binds the blocking review of the original
head (newer than that original authorization, never merely older than the fresh
grant), the rejected-follow-up record, the attempt-2 recovery dispatch record
and the one coder record, in that strict order. Correlation, publication, resume
and plan binding share one chain authentication (`_authenticate_authorization_chain`):
a recovery-shaped chain is walked through the fresh grant to the original
authorization's root, and every record on it must pass the same actor, scope,
waiver, config-aware protection and label-event checks on the current approved
plan only.

A head advance carrying none of an ordered blocking-review/coder pair, the
merge-conflict obligation, a CI obligation bound to exactly that transition, or
the exact rejected-coder recovery chain still fails closed, and every reviewer must still approve the exact final head
before qualification or merge.

Coder and format-repair turns on a managed PR cannot dispatch CI before
approval. `pr_loop.py` evaluates `managed_ci_active` once per coder
turn. Issue and approved-plan implementation use the managed creation intent
instead. That boolean travels as `coder_ci_dispatch_guard` on a
`dataclasses.replace`d config that is passed only to the coder prompt builder
and the coder `_run_validated_agent` call. Retries, format repair and the semantic-evidence correction
inherit it, and reviewers never see it. The same flag emits the prompt guidance and
installs `coder_ci_guard.py`'s `gh` guard. Every coder or repair launch site
calls `coder_ci_guard_environment`: the four backends, `run_cli_repair`, and the
Gemini repair subprocess. The guard refuses `gh workflow run`, `gh run rerun`,
and dispatch/rerun POSTs. It `exec`s everything else through
`AGENT_LOOP_CODER_GUARD_TARGET_GH` with its own directory removed from `PATH`,
and preserves the REST shim's `AGENT_LOOP_REAL_GH`. After the turn,
`github.audit_coder_turn_dispatches` lists `workflow_dispatch` runs on the PR
branch created since the turn started, and every known head, so intermediate
heads are covered. Any non-managed run it finds is prepended to the coder
comment as a non-authoritative diagnostic. The audit runs after the turn's
last coder continuation. On a rejected guarded turn,
`publish_rejected_coder_turn_dispatch_audit` posts the note on its own and
never masks the rejection. Managed v2 runs are excluded by their
`managed-ci-v2 nonce=` run name. A v1 run carries only the plain workflow
name, so `dispatch_final_qualification` appends a `ManagedV1DispatchRecord`
(workflow file, ref, PR number, expected head, request and return times) to
the in-process `ManagedCiContract.v1_dispatch_records` after each successful
v1 POST (#1414). The records are never persisted. Both PR-loop audit paths
pass them in. Discovery then widens to each record's head and window, and
records are matched by workflow path, head, branch and time window against
the full deduplicated candidate set before the turn cutoff applies. A record
claims a run only when it is that record's sole match, no other record
matches it, and the record head's listing succeeded. A claimed run is only
logged. Any other matched run inside the turn window goes to a separate
"attribution undetermined" section that names no creator and gives a typed
reason: `multiple-matches`, `shared-match` or `incomplete-lookup`.
Chronological order never decides ownership. The classifier decodes arguments with one
pflag-style tokenizer, so grouped and equals shorthands cannot slip past
it. Gating keeps excluding such runs as
ad hoc diagnostics, and the workflow and its `unlabeled` recovery are
unchanged.

## Decisions

[Decision log](decisions/managed-ci-and-merge.md)
