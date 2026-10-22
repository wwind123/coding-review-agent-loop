# Execution and concurrency

How agent and Git processes run: per-run worktrees over a shared store, the agent-loop-owned Git boundary and private fetch transport with SSH identity selection, pre-turn checkout verification, process-tree containment, test interpreter and worker budgets, pre-collection launch-failure provenance, the Antigravity quota-group fallback, the opt-in quota reset wait, and the other entry paths outside the review loop.

Source paths: `src/coding_review_agent_loop/run_worktrees.py`, `src/coding_review_agent_loop/workdirs.py`, `src/coding_review_agent_loop/workdir_claims.py`, `src/coding_review_agent_loop/quota_wait.py`, `src/coding_review_agent_loop/checkout_verification.py`, `src/coding_review_agent_loop/secure_git.py`, `src/coding_review_agent_loop/git_transport.py`, `src/coding_review_agent_loop/transport_check.py`, `src/coding_review_agent_loop/runner.py`, `src/coding_review_agent_loop/containment.py`, `src/coding_review_agent_loop/test_workers.py`, `src/coding_review_agent_loop/test_runtime.py`, `src/coding_review_agent_loop/lifecycle_probe.py`, `src/coding_review_agent_loop/agents/antigravity.py`, `src/coding_review_agent_loop/discuss_loop.py`, `helpers/`, `tests/test_run_worktrees.py`, `tests/test_containment.py`, `tests/test_test_workers.py`.

## Execution and Concurrency

Concurrent default-dir invocations on one repository are supported.
Each run with an omitted `--<agent>-dir` gets its own detached `git worktree`
under `<scratch>/OWNER-REPO/<agent>/runs/<run-token>`, backed by one shared
clone per repo/agent (the *store*, `<scratch>/OWNER-REPO/<agent>/repo`) that
owns the object store. Explicit `--<agent>-dir` paths are untouched. Optional `--worktree-link PATH` (repeatable, off by default) symlinks (never copies) an untracked repository-relative path from the shared store, for example `.venv`, into each per-run worktree. The path must exist in the store (otherwise the run fails, naming it) and its parent directories must be tracked. Explicit `--<agent>-dir` checkouts ignore it. The link is exempted from checkout verification, sync cleaning and test-evidence attribution by agent-loop's in-process registry only (no git ignore state is written); a removed, retargeted, staged or committed link is refused. Linked state is shared and mutable across concurrent runs. Keep it ignored with `/.venv` (a dir-only `.venv/` rule does not match a symlink). `.venv/bin/python` works through the link; bare `python` reaches the linked venv only if the agent environment activates it. Parallel
reviewers within one invocation are supported; the orchestrator verifies
distinct reviewer workdirs. Coder and reviewer turns are separate lifecycle
stages even when they use the same CLI.

Named seats with no explicit `--seat-dir` use the same per-run worktree lifecycle,
with a separate shared store and finalizer keyed by each seat ID.

**Per-run worktrees (`run_worktrees.py`).** Store mutations (initialization,
private fetch and pack import, local-base fast-forward, pack compaction,
administrative link creation, prune, remove) run under a short,
non-reentrant host `flock` (`<lock root>/workdir-stores/`) that is never held
across an agent turn; a nested acquisition fails fast. A probe of the store
path's own run claim (`workdir_claims.probe_free_claim`) interlocks with an
explicit-dir or older-version run that checked out a branch in the store: while
it is held the store's HEAD, branch, index and tree are untouched and the run
only imports objects and updates tracking refs. Every worktree has a durable owner record (`<store-key>/<token>.json`,
pending then ready, with the root and admin-dir identities) under the lock
root. Base and PR syncs check out an immutable SHA pinned under the store lock,
so a concurrent fetch cannot move another run's checkout; the PR path still
compares the pin with the advertised head. The local `<base>` ref moves only by
a guarded compare-and-swap `git update-ref` fast-forward, skipped when `<base>` is checked out in
any worktree. The outermost `workdir_claim_scope` runs registered finalizers
(identity-verified worktree removal) before releasing claims, on success and
failure. At startup a run prunes worktrees whose owner record matches and whose
claim lock is free; unrecognized entries are logged and left untouched.

**Incremental import.** `import_ref` exports only objects not reachable from
destination ref tips that the private fetch repository also holds as commits:
those tips are passed to `pack-objects --revs --thin` as negative revs and the
confined `index-pack --fix-thin` completes the thin pack. When the fetched
commit is already a ref tip whose closure is complete, no pack is written.
Destination tips are exclusion hints only, never proof of completeness: every
path verifies the fetched commit's full reachable closure in the destination
(`rev-list --objects --missing=error`, no history excluded, with the
destination's shallow file, grafts and replace refs ignored) before the
tracking-ref compare-and-swap, and an incomplete closure or a failed
incremental export falls back once to a full export, which repairs missing
history. If that also fails or raises, the fixed import error is raised and
the ref is not moved. The network fetch into a fresh private repository is unchanged.

**Pack compaction.** Each import writes at most one pack and the hardened config keeps
`gc.auto=0`, so after the last import in each locked fetch (`prepare_store`,
`pin_pr_head`, and the explicit-checkout base and PR syncs in `config.py`)
`compact_packs_if_needed` runs while the store or checkout `flock` is still
held. Above 50 non-kept packs it writes one replacement pack with a single
confined `git pack-objects --stdout --no-use-bitmap-index --keep-unreachable`
into a private scratch directory; `git repack` cannot be used because it execs
`pack-objects`, which the guard denies. A confined `index-pack` indexes it,
`show-index` verifies that it holds every object of the snapshot packs, and
only then are the snapshot packs deleted. Kept, promisor and post-snapshot
packs are never touched and nothing is pruned, so SHA-pinned PR heads survive.
A missing lock raises; any other failure is logged and never fails the fetch.
Incremental imports shrink each pack, but compaction remains the bound on
pack count. A timeout of `pack-objects`, `index-pack` or `show-index` (and
only a timeout) records a private backoff marker for the store beside its
lock file in the per-user store-lock root, keyed by the same store key, so no
checkout can plant or remove it. While its deadline is ahead, later locked
fetches still run the cheap `rev-parse --git-common-dir` lookup and pack
listing but none of the expensive pack commands, so the flock is not held for
the full timeout on every fetch. The window is 1 h, doubles per consecutive
timeout and is capped at 24 h; the exponent saturates and the failure count
is bounded. A successful compaction, or a pack count at or below the
threshold (for example after a manual repack), clears the marker. The marker
is opened non-blocking without following symlinks and validated by `fstat`
(regular, owned by the user) before reading; an unreadable, non-regular,
oversized, malformed (including any parse failure such as deep nesting or an
overflowing number), out-of-range or more-than-24-h-future marker fails open
to a normal attempt, so no marker suppresses compaction for more than 24 h.
Both the timeout log and the backoff skip log carry a fixed hint, with the
checkout shell-quoted, to run
`git -C <checkout> repack -a -d --keep-unreachable` while no run uses the
store.
A multi-pack-index written by an operator or `git maintenance` names the
packs compaction deletes, so, as `git repack -d` does, once compaction passes
the threshold and backoff checks it removes every multi-pack-index artifact
(the `multi-pack-index` file, its `.bitmap` and `.rev` sidecars, and a
`multi-pack-index.d/` chain) before the `show-index` inventory and before any
pack is written, and therefore before any snapshot pack is deleted. Only
exactly-named entries are removed, links are unlinked and never followed, and
removal spawns no process. The multi-pack-index is never rewritten; Git
rebuilds it on the next maintenance run. A removal failure writes no pack,
runs no pack command, keeps the old packs, logs a fixed reason and starts no
backoff, so a persistent failure repeats only the cheap listing and removal
attempt. With no multi-pack-index present the command sequence is unchanged.

**Agent-loop-owned Git boundary (`secure_git.py`, `git_transport.py`).** Every
local text, binary, and cancellable Git call that can read an agent checkout
uses the pinned executable, a closed child environment, and command-scoped
neutralizers. Linux loads a sealed preload library whose constructor installs
no-new-privileges and an inherited seccomp filter denying exec and network;
macOS loads a private library whose constructor applies Seatbelt before Git
main, denying process exec, fork, and network without a pinned-Git exception;
Windows starts Git suspended, assigns a kill-on-close one-process Job Object,
then resumes it. Each backend performs a real canary before the first checkout
probe and fails closed if confinement cannot be verified. Existing process
group containment is not an execution boundary. The isolated inspector keeps
its standard-library-only main process and delegates Git through verified
helper bytes executed by an isolated interpreter using the same local boundary.
Guard C source and the Windows launcher are checked against hashes held by the
already loaded controller before use. The compiler receives verified C bytes
through stdin, and Windows passes verified launcher code directly to a trusted
Python executable, leaving no helper file to replace between check and launch.
Each local Git invocation also binds `--work-tree` and `core.worktree` to its
assigned checkout, so repository config cannot redirect status or cleanup to
another directory. Repository initialization has no existing worktree to bind.

Fetches run only in a fresh private transport repository, with clean Git
configuration and an exact operator-selected GitHub HTTPS or SSH URL. An
explicit `--trusted-local-origin` is required for local paths. Private HTTPS
may obtain a token from an operator-owned GitHub CLI installation resolved
independently of inherited `PATH`, passed only as an in-memory scoped
header to that private fetch. On POSIX hosts, both pinned executable paths,
including their parent directories, must be root-owned so an unrestricted
same-user agent cannot replace them before launch.
Transport failures raise `TransportFailure`, whose `label` carries the fixed
class. `agent-loop transport-check` (wrapped by `scripts/transport_smoke.sh`)
runs a local configuration preflight and then one production `import_ref` fetch
into a private scratch repository with the operator's credentials. It reports
only that class or a fixed fallback (`configuration`, `local import`,
`unclassified`). It is required before merging transport changes.
The `--trusted-origin-protocol` setting selects HTTPS or SSH when creating a
new default checkout; the pinned SSH client is validated before transport.
Its command keeps `-F /dev/null` and executable-option hardening, retains
default keys and `SSH_AUTH_SOCK`, and pins `StrictHostKeyChecking=yes` before
any other option, so an unknown or changed host key fails closed without a
prompt (`BatchMode` stays unset for passphrase prompts). A trusted host with a
port uses an `ssh://` origin.
SSH identity selection: on POSIX, `git_transport` reads a non-executing
subset of `~/.ssh/config` for the literal URL host (`Host` blocks only; never
`ssh -G`). It honours `IdentityFile`, `HostName`, `IdentitiesOnly`, a single
`UserKnownHostsFile`, and `Port` only when the URL has no port. `User`,
`Match`, `Include`, `ProxyCommand`, `ProxyJump`, `LocalCommand`,
`KnownHostsCommand` and all other keys are ignored. Values are emitted as
explicit `-o`/`-i` options only when their final form (after `~/` expansion and
realpath) matches `[A-Za-z0-9._/+,:@-]+` without a leading `-`; anything else is
refused rather than escaped, so neither the shell nor ssh's option grammar or
`%`/`${}`/`~` expansion can reinterpret it. The config file, identity files and
the known-hosts file pass one trust walk (`_trusted_operator_file`). It
resolves the path component by component, expanding each symlink in place
(bounded hops), so every component traversed, including intermediate links
and the directories their targets pass through, is owned by the current uid
or root and, unless it is a symlink, not group- or world-writable. Root-owned
sticky directories are exempt, but never the final file or the parent of an
accepted missing file. The supplied path and every traversed component must
lie outside `transport_forbidden_roots`, which covers every agent checkout,
named-seat workdir, store, run-worktree root and import destination, so an
agent-controlled link cannot select an otherwise trusted file. The target must
be a regular file. The config
is opened `O_NOFOLLOW|O_NONBLOCK`, inode-checked against the walk and read with
a 256 KiB bound. An explicit `--git-ssh-identity` / `AGENT_LOOP_GIT_SSH_IDENTITY`
replaces config identities with exactly one `-i` and `IdentitiesOnly=yes`, and
an untrusted override is a fixed error. Windows does not read the config, keeps
the previous command byte-for-byte and refuses the override. Transport failures
report a fixed class chosen by evidence precedence (`remote not found`,
`host-key failure`, `network or timeout`, then `authentication failure` from
specific evidence before Git's generic access-rights footer, else
`unclassified`) with no stderr or argv; SSH authentication and host-key
failures add a fixed hint. No HTTPS fallback is attempted after SSH failure. Same-UID live races against `~/.ssh` remain out of
scope.
The complete reachable pack for a pinned commit
is streamed to confined `index-pack --stdin`; the destination verifies the
commit, then compare-and-swaps the tracking ref while its checkout or store
lock is held. Base ancestry and advertised PR head are checked before cleaning
or checkout. Git 2.43 spawns `update-ref` even for `worktree add --no-checkout`;
fresh linked worktrees therefore create only the administrative link under
`store_lock`, run a separate confined `reset --hard` at the pinned SHA, and
publish a ready owner record only after administrative identity, HEAD, and
clean status verify. Any setup failure removes only the recorded root and
administrative inode identities; incomplete cleanup is reported as an error.

**Pre-turn checkout verification.** A claim protects against another
agent-loop run, not against an operator, editor or stray script writing to the
same directory. `run_agent_result` therefore verifies the assigned checkout
before every checked agent turn (every attempt, including retries,
executable-replacement replays and Antigravity fallbacks) and fails closed with
`CheckoutVerificationError`, naming the expected and observed branch/HEAD and
each unexpected path. The in-process ledger in `checkout_verification.py` keeps
one fingerprint per checkout: branch, HEAD, staged blob identity and a content
identity (sha256, or stat identity beyond the 16 MiB per-file / 512 MiB total
hash budget) for every dirty path, with each untracked file listed
individually and paths handled as raw bytes. Probes use `--no-renames`, go
through the hardened git runner in sandboxed mode, and never open a symlinked
ancestor, FIFO, socket or device. Verification never resets, cleans, stashes
or deletes foreign changes. The baseline is established by startup preparation,
verified before every between-turn sync (before any reset or clean) and every
real test gate, and refreshed only afterwards; a gate adopts only new
artifacts. Each completed turn refreshes it, and a post-turn capture that
fails, or exceeds the 100,000-path cap, poisons the entry so a later
clean-but-wrong checkout is refused rather than adopted. `semantic-dedupe` and
`repair` (tool-owned temporary directories) and dry-run are exempt. The
Antigravity backend treats `semantic-dedupe` like `repair`: a GEMINI lock
beside the temporary directory instead of in git metadata, no checkout
snapshot or replacement classification, and the strict empty-allowlist
settings injection for the turn. Explicit
agent directories must be existing git checkouts. Foreign writes *during* a
turn are absorbed into the refreshed baseline; excluding them is the claim's
responsibility.

The Antigravity `GEMINI.md` injection carries a self-describing header
(`orig`, body byte length and sha256) so a leftover from a killed run is
recognised as tool-owned in both the per-turn check and the startup dirty
check, and only the verified header and body are removed. Every agent-loop
access to `GEMINI.md` goes through one fd-relative accessor that touches only a
regular, non-symlink file, and stale recovery takes the backend's non-blocking
`GEMINI.md` lock first.

Subprocess capture roots live outside managed checkouts and each holds a
lifetime `flock` lease. Capture-root retention is operator-configured
(`--subprocess-log-retention-days`, default 14 days, `0` disables pruning). A
sweep in the `subprocess-logs/<slug>/` layout considers only real
(non-symlink), marked, expired and unlocked roots; leased roots are never
pruned. An eligible root is removed whole without following symlinks, with the
marker and lease removed last so a partial failure leaves the root marked and
expired for a later retry. See
[`--subprocess-log-retention-days`](../local_agent_loop.md).

On supported Linux/systemd/cgroup-v2 hosts, containment bounds aggregate and
role-specific process trees. The portable process-group fallback can terminate
descendants but does not impose memory ceilings. Test-wrapper lane locks prevent
recognized duplicate commands, not all overlapping work in arbitrary shells.
Backend turn timeouts, whole-test-command watchdogs, and framework per-test
timeouts are independent limits. See [containment](../local_agent_loop.md#process-tree-containment).

An operator-configured `--test-python` interpreter is validated, exported to
every coder as `AGENT_LOOP_TEST_PYTHON`, import-checked at each coder-prompt
render and, in sandboxed mode, becomes the single granted test invocation. Claude
coder turns get `BASH_DEFAULT_TIMEOUT_MS`/`BASH_MAX_TIMEOUT_MS` sized by
`run_tests_foreground_budget_seconds` (watchdog ceiling, host-wait bound,
parent overhead, headroom); every other backend and role carries an `unknown`
shell-cap fact. Broker-owned detached runs are deferred. See
[Running a long or full suite](../local_agent_loop.md#running-a-long-or-full-suite).

A containment-aware test-worker budget (`test_workers.py`) is derived after
admission from the limits that apply on each path (managed handle limits,
cgroup ancestry, usable host memory, CPU affinity/quota) and exported to coder
and repair agents as `AGENT_LOOP_TEST_WORKERS`. The parent-owned broker is the
enforcement boundary: it applies the stricter of the parent and client values
and injects a stdlib-only pytest plugin that clamps or refuses the final
resolved pytest-xdist worker count and reports the gateways actually created.
The local fallback is advisory by comparison, and the plugin is a prompt-slip
safety net, not a sandbox. In clamp and refuse one test command per invocation
holds a worker-budget lock. Host-wide admission is exclusive with a bounded
wait (`AGENT_LOOP_TEST_WORKER_HOST_WAIT_SECONDS`): a run waits for the shared
pool inside `run_foreground_test`, around the test command only, and on timeout
falls back to the free workers (or one oversubscribed worker) instead of
refusing; the fallback lowers launched workers after the configured-budget
judgement, so refuse-mode enforcement is not weakened. A broker request's
pending admission is cancelled when the broker stops, synchronised with target
launch so no target starts after the owning turn ends. See
[Parallel test-worker budget](../local_agent_loop.md#parallel-test-worker-budget).

The same test-execution boundary distinguishes positive lifecycle provenance
from vetoes when deciding that a direct-pytest run exited before any test could
run. Provenance is parent-owned: a launch eligibility the parent validated for a
native, non-script, non-isolated interpreter that passed a probe-import
preflight, the `_agent_loop_lifecycle_probe` plugin's strictly parsed report
(injected ahead of later plugins; option parsing never finished and no test-code
file was touched first; a clean `final` record), or an independent parent
importability check when pytest itself is absent. The whole-stream output scan
and the worker-session report are vetoes only, and a missing, blocked or
unreadable signal always fails closed. The probe is a cooperative observation
channel, not a sandbox. A classified run is a persisted non-evidence
`launch-failed` row; see
[Pre-collection launch failures](../local_agent_loop.md#pre-collection-launch-failures).

**Antigravity quota-group fallback.** `agents/antigravity.py` derives a
quota group per chain entry (an explicit override or the model-name family) and
keeps `AntigravityQuotaGroupMemory`, a locked map of exhausted groups with
finite monotonic expiries: the parsed reset, or a bounded
`--antigravity-quota-cooldown-seconds` cooldown. The memory is owned by the
logical run: it is keyed by the `workdir_claims` owner's `run_id`, created on
first use, dropped by a run finalizer when the outermost scope ends, shared by
nested loops, and reached from reviewer threads because `review_rounds.py` and
`discuss_loop.py` submit with `contextvars.copy_context()`. With no owner
(library use, the skill helper process) the memory is local to one attempt
state. Exhaustion is classified in `transient.py` only from the matched
provider error frame, extracted from raw unstripped lines (never the whole
transcript), with capacity wording taking precedence. Auth (auth words or a
numeric 401/403 in a terminal frame) returns a typed `auth_failure` verdict that
both callers enforce as non-retryable with no fallback, unless a typed runner
retry decision applies; dirty checkouts stay non-retryable; billing wording is
exempt only when agy's credits-balance error is correlated with a terminal
retryable `429 RESOURCE_EXHAUSTED` frame, which then cools its group down like
any other exhaustion. agy can also end a quota-exhausted turn silently (no
response file; stdout empty or only agy's `run ended with no output and no
recorded error` warning). `AntigravityBackend.run` then reads agy's own CLI
log (`~/.gemini/antigravity-cli/log/cli-<stamp>.log`). Newness is proven by a
directory inventory taken under the shared settings lock, which serializes
agent-loop agy turns host-wide, just before spawn: only a name absent from it
qualifies. The name-stamp and mtime windows around the attempt then only
narrow among new regular files, and exactly one candidate must remain. An
error-severity executor `RESOURCE_EXHAUSTED` `429` line in that log becomes
typed `AntigravityLogQuotaEvidence` on `AgentResult`. `validated_agent.py`
routes it, before the nonzero-exit, empty-response and format/repair branches,
into the same quota path as a printed frame (reason `agy-log`), unless a typed
runner retry decision applies. Any lookup failure leaves no evidence and keeps
the earlier behavior. `AntigravityAttemptState`
is the single policy shared by `validated_agent.py` and `helpers/run_external.py`:
transient failures retry then fall back to a later eligible entry; a verified
exhaustion skips the whole group with no retry or sleep and makes at most one
cross-group jump per turn; a second verified exhaustion is a shared-limit stop;
and an all-groups-cooling state stops before invoking. Only when every group
carries a parsed long reset is `QuotaResetExceededError` raised. Memory is not
persisted across processes.

**Quota reset wait.** `--wait-for-quota-reset DURATION` (off by default) turns
a parsed long-reset stop into a bounded wait. `quota_wait.py` owns a
`QuotaWaitBudget` (at most 2 waits and DURATION in total). Like the quota
memory, it is keyed by the `workdir_claims` owner's `run_id`, dropped by a run
finalizer, shared by nested loops and context-propagated reviewer threads, and
local to the turn with no owner. `validated_agent.py` reserves from the budget
at the long-reset raise sites and sleeps between invocations, outside
`run_agent_result`, so no subprocess or settings lock is held. It then
replays the frozen failed invocation (prompt, session, timeout and settings).
Each wait extends the turn's attempt ceiling by the quota-failed dispatches it
reopens, so the replay and any renewed Antigravity fallback never consume the
ordinary allowance. For Antigravity it waits only when
`stop_is_verified_long_reset()` holds and the latest group reset is within
DURATION. A pre-dispatch all-exhausted wait builds the next invocation
normally, and a post-failure wait replays the failed one. Waits sleep in
slices of at most 30 s on a process-level cancel event in a waiter registry.
The CLI installs a SIGTERM handler for the command only when the option is
set. The handler is lock-free (it reads the registry and sets a flag), because
it may run while the main thread is inside the registry. The handler, and SIGINT's own `KeyboardInterrupt`, reach the main
thread. There, a waiting main thread converts the interrupt to
`QuotaResetExceededError`, and the reviewer (`review_rounds.py`) and debater
(`discuss_loop.py`) executors cancel waiting workers before terminating
processes and joining. `cli.main` maps an interrupt that cancelled a wait to
exit code 3. SIGTERM with no active wait keeps its previous behaviour.
The cancel latch is process-level. A cancel always sets it, so a wait that is
about to start in an interrupted run still stops. It persists while any logical
run is live and is cleared when a new run starts after all runs ended with no
active wait (#1609). `cli.main` also clears it on every exit.
`workdir_claims` counts live outermost `workdir_claim_scope` runs and runs
registered first-run-start hooks on the zero-to-one transition; `quota_wait`
registers its stale-latch reset there. A run-lifecycle lock, separate from the
claim registry lock, covers the increment, the hooks and the decrement, so no
other run is admitted before the reset finishes. Waits outside any scope have
no run boundary, so their callers call `reset_wait_state()` themselves.

## Other Entry Paths

- `discuss` coordinates non-implementation debate, with optional research,
  agenda/synthesis, and evidence reconciliation. It does not implicitly enter
  implementation or merge a PR.
- `managed-pr` prepares a managed PR from an existing branch before opening it;
  it is distinct from explicitly adopting an already-open PR.
- `managed-ci preflight`, `containment-preflight`, and `run-tests` expose
  supporting checks and execution helpers outside the full review loop.
- `history export` is a read-only reporting path: it reads issue and linked-PR
  conversations, edit history and closing-PR state from GitHub and writes one
  local JSON or CSV file. It shares no state with a running loop and never
  writes to GitHub.
- `issue --triage` adds one advisory model call before an issue run's first
  agent dispatch and one issue comment. It reads an optional local history
  export file but never changes the route, execution mode or policy the
  operator selected.
- Claude Code skill mode uses an interactive host plus [`helpers/`](../../helpers/)
  for validation, external agents, GitHub operations, and local session state.
  It reuses library functionality but is not an identical CLI lifecycle and
  never auto-merges. Host-session work is outside the external-agent process
  boundary. See [skill architecture](../skill_mode.md#architecture).

## Decisions

[Decision log](decisions/execution-and-concurrency.md)
