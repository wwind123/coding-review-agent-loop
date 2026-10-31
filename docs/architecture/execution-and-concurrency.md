# Execution and concurrency

How agent and Git processes run: per-run worktrees over a shared store, the agent-loop-owned Git boundary and private fetch transport with SSH identity selection, pre-turn checkout verification, process-tree containment, test interpreter and worker budgets, pre-collection launch-failure provenance, the Antigravity quota-group fallback, the opt-in quota reset wait, the agent-loop executable pin with its receipt tool identity, and the other entry paths outside the review loop.

Source paths: `src/coding_review_agent_loop/run_worktrees.py`, `src/coding_review_agent_loop/worktree_setup.py`, `src/coding_review_agent_loop/worktree_hints.py`, `src/coding_review_agent_loop/worktree_pr_env.py`, `src/coding_review_agent_loop/workdirs.py`, `src/coding_review_agent_loop/workdir_claims.py`, `src/coding_review_agent_loop/quota_wait.py`, `src/coding_review_agent_loop/checkout_verification.py`, `src/coding_review_agent_loop/secure_git.py`, `src/coding_review_agent_loop/git_transport.py`, `src/coding_review_agent_loop/transport_check.py`, `src/coding_review_agent_loop/runner.py`, `src/coding_review_agent_loop/agent_executable.py`, `src/coding_review_agent_loop/containment.py`, `src/coding_review_agent_loop/test_workers.py`, `src/coding_review_agent_loop/test_runtime.py`, `src/coding_review_agent_loop/lifecycle_probe.py`, `src/coding_review_agent_loop/agents/antigravity.py`, `src/coding_review_agent_loop/discuss_loop.py`, `helpers/`, `tests/test_run_worktrees.py`, `tests/test_worktree_setup.py`, `tests/test_worktree_dependency_hints.py`, `tests/test_worktree_pr_env.py`, `tests/test_containment.py`, `tests/test_test_workers.py`, `tests/test_agent_executable.py`.

## Execution and Concurrency

Concurrent default-dir invocations on one repository are supported.
Each run with an omitted `--<agent>-dir` gets its own detached `git worktree`
under `<scratch>/OWNER-REPO/<agent>/runs/<run-token>`, backed by one shared
clone per repo/agent (the *store*, `<scratch>/OWNER-REPO/<agent>/repo`) that
owns the object store. Explicit `--<agent>-dir` paths are untouched. Optional `--worktree-link PATH` (repeatable, off by default) symlinks (never copies) an untracked repository-relative path from the shared store, for example `.venv`, into each per-run worktree. The path must exist in the store (otherwise the run fails, naming the shared clone and suggesting `--worktree-setup` or creating the path there) and its parent directories must be tracked. Explicit `--<agent>-dir` checkouts ignore it. The link is exempted from checkout verification, sync cleaning and test-evidence attribution by agent-loop's in-process registry only (no git ignore state is written); a removed, retargeted, staged or committed link is refused. Linked state is shared and mutable across concurrent runs. Keep every link ignored with a root-anchored pattern without a trailing slash, for example `/.venv` or `/node_modules` in `.gitignore` or the store's `.git/info/exclude` (shared by every worktree); a dir-only `.venv/` or `node_modules/` rule does not match a symlink. agent-loop warns, and never writes ignore state, when a created link is not ignored. `.venv/bin/python` works through the link; bare `python` reaches the linked venv only if the agent environment activates it. Parallel
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

**Shared link source health (`worktree_link_integrity.py`, #1775).**

Shared link sources now require nonempty directory outputs. Agent-loop records
bounded structural health in private `link-health.json` beside `setup.json`,
binding the store identity, configured links, source identities, generation,
selected probes and stale reason. Publication is atomic, mode 0600 and fsynced.
Startup and the second store-lock hold before worktree creation check health;
every checked backend attempt checks still-shared sources before launch. The
post-turn check inspects the original sources even after link replacement,
reconciliation failure or sticky checkout poison. Under the store lock it
revalidates current health, preserving healthy newer setup generations, and
marks damage stale without deleting the generated-key ledger. No lock spans
an agent turn. A poisoned run remains poisoned after source repair.

Checks read no dependency-file contents: at most 64 selected directory counts,
128 children per count (8192 entries total), 128 saved path/type witnesses and
probe depth 4 (counted children can reach depth 5). The scan payload is at most
32 KiB and complete record at most 64 KiB. Roots reserve counts first; optional
probes use breadth-first selection. Large healthy trees reduce coverage and
log limitations instead of being refused. Baselines are never resampled on
reuse. Additions and content/mtime changes do not rebase counts. Counts capped
at 128 detect decreases only below their recorded value. Unselected/deeper
deletion, same-count substitution, overwritten bytes and damage behind interior
symlinks exceed the guarantee. Interior symlinks and substituted probe ancestors
are not traversed. Unsafe, unreadable, unstable or malformed required health
refuses reuse; more than 64 directory roots is a configuration error.

Legacy setup-backed stores rebuild once to establish health. Inspectable,
nonempty manually managed sources bootstrap with a warning: historical partial
damage cannot be proven absent. Empty sources, damaged health and durable stale
state force operator setup even if the setup digest matches. Without setup,
agent-loop refuses before creating a worktree or launching an agent, naming the
absolute clone/source, relative link and reason. Initial capture failures preserve
the failing source through root and optional-probe checks; failed setup output
validation, including missing or dangling outputs, retains that source in the
pending recovery record and repair message across restart.
Rerun once with a trusted
`--worktree-setup CMD` that reinstalls or validates from the named clone root;
for node_modules use `--worktree-setup 'npm ci'`. After manual repair use a real
validating command that exits nonzero for incomplete dependencies. Do not use a
no-op command or delete health records. Only successful setup with validated
nonempty outputs clears stale state; setup success is published last and must
match the health generation before setup-backed sources can be reused. Pending
recovery records retain store/configuration identity but cannot authorize reuse. Failed or
interrupted recovery retains health history and generated-key protections.

Shared symlinks still permit write-through. Same-user chmod/symlinks cannot
provide per-turn isolation, and permissive backend launches have no common
read-only mount boundary. Agent-loop does not recursively chmod dependencies or
install mount infrastructure. Use `--worktree-setup-pr-local` for opt-in writable
per-run dependencies. Already-running consumers can observe damage before the
next check. Bounded post-turn lock waits and intentionally empty shared caches
remain future work. Explicit directories and dry-run retain their behavior;
private paths are excluded from shared-source attribution.

**Operator worktree setup (`worktree_setup.py`).** `--worktree-setup CMD`
is an operator CLI flag, never read from repository configuration. Inside
`prepare_store`'s single store-lock hold, after `_fetch_and_pin` returns the
base SHA and before pack compaction, `ensure_store_setup` decides whether the
setup for that same SHA is current, so both `prepare_store` callers (worktree
creation and every later per-run base sync) never hand out a pin with an
out-of-date setup; `pin_pr_head` never runs setup. The digest hashes the
command string's SHA-256 and each `--worktree-setup-key` entry (`git ls-tree`
mode, type and OID at the SHA, or an `absent` sentinel), never working-tree
bytes. The success record is `<lock root>/workdir-stores/<store-key>/setup.json`
(private, outside the clone, written atomically with mode 0600 only after a
zero exit) and holds the digest, command hash, base SHA and the store root's
dev/ino as a secondary check. Setup is required when the record is missing,
unreadable or different, or when any `--worktree-link` source fails the shared
`link_source_usable` predicate (a dangling source symlink counts as missing),
or persisted source health requires recovery; `_prepare_store_locked` deletes the record and the generated-key ledger before it
removes a stale store or clones a missing one. With `--worktree-setup`, a store whose
index or HEAD commit still holds tracked paths (only an unborn HEAD with an
empty index counts as artifact-only, and an inspection failure counts as
tracked) is never treated as stale, so staged or unstaged tracked deletions
reach the setup gate instead of being re-cloned over. A required setup needs a free
store claim (otherwise the held-store message), then refuses any locally
modified, staged or deleted tracked path, any index entry flagged
assume-unchanged or skip-worktree (which status and diff trust), any dirty or drifted initialized
submodule (separate confined per-submodule probes, the same inspection as
test-evidence snapshots; parent probes keep the confined
`--ignore-submodules=all`) and any untracked file at a key absent at the SHA
that the generated-key ledger does not exempt, all before the record is
touched. The ledger, `setup-generated.json` beside `setup.json` (private, mode
0600, written atomically, bound to the store dev/ino), maps each key absent at
the SHA that exists as a regular file after a successful setup to its SHA-256.
It is replaced just before the success record is written, is left untouched by
pre-setup record invalidation and by failed or timed-out setups, and is
deleted together with `setup.json` by `invalidate_store_setup_state` whenever
`_prepare_store_locked` removes a stale store or clones a missing one. A
malformed, foreign-identity, wrong-version, symlinked or foreign-owned ledger,
or one naming an unconfigured key or a non-hex hash, exempts nothing. A key is
exempt only when it opens without following a symlink as an operator-owned
regular file inside the store whose streamed SHA-256 equals the ledger entry;
PR-local worktrees never use the exemption. Untracked files at key paths the
new SHA tracks but HEAD does not are classified in the same pre-invalidation
check: a content-matching ledger entry is unlinked after the record is deleted
and immediately before the checkout (re-verified first), and anything else is
refused. Every remaining absent-key refusal says the file may have been
created by the setup command itself. It then deletes the record and fsyncs its
directory (failing closed), runs a non-forced
`checkout --detach --no-overwrite-ignore <sha>` (so Git refuses rather than
silently replacing an ignored untracked file), re-verifies HEAD and the
tracked tree, and runs the command through `/bin/sh -c` in the store as the
operator in its own process group, bounded by the operator-only
`--worktree-setup-timeout` (default 3600 s, independent of the coder
test-command timeout, kept by managed-CI recovery commands; the PR-local
rebuild uses the same limit). The leader is observed with `waitid(WNOWAIT)`
and left unreaped until the whole group has been killed, so however the command
ends no background child survives the store-lock release and the group ID
cannot have been reused when it is signalled; on a host without `os.waitid`
setup is refused before anything launches. A failure or timeout raises
with the output tail, leaving no record; a success that still leaves a link
source missing or dangling raises the common source-specific repair diagnostic,
retaining the failing link in pending recovery without a setup-success record. Dry-run
only logs. After linking, a read-only `git check-ignore --no-index` probe in the
new worktree warns when a link is not ignored as a symlink (never failing,
never writing ignore state), and with neither setup nor link flags a one-line
hint per run names dependency manifests found in the pinned tree.

**Dependency remedy text (`worktree_hints.py`).** Launch-failure and refusal
diagnostics add one sentence naming `--worktree-link` and `--worktree-setup`
only for a default per-run worktree with no configured link, decided from
parent-owned config (`run_worktree_store` and `config.worktree_links`), never
from the process-local link registry. Parent-side sites (the coder test gate,
the workdir guard's out-of-checkout executable refusal on structured, citation
and free-form reports) read the config directly. Every agent launch sets
`AGENT_LOOP_WORKTREE_DEPENDENCY_HINT` explicitly: `unlinked-run-worktree` under
the condition and empty otherwise, so a value inherited by the orchestrator is
neutralized. The Runner passes the same parent-built decision to the test
broker, which attaches it as a separate `advisory` field to launch-failure
result frames (never to the journal or the stored diagnostic); the `run-tests`
client (including `--fixture` runs) prints that advisory, or the remedy selected by its own variable, once
after a launch failure or a pre-collection failure. The variable is excluded
from evidence environment identity, and nothing reads it for classification,
outcome, evidence or exit code. A replaced-link refusal names the link source
in the shared clone. When an issue run stops (a checkout refusal or a terminal
no-PR result) while the implementing coder's per-run worktree is poisoned and its branch
has a remote-tracking ref, the stop's message gains a `managed-pr --head`
resume and the manual worktree-removal command; the exception type, exit code
and refusal are unchanged and the worktree is left in place.

**PR-local dependency environments (`worktree_pr_env.py`).** The operator
flag `--worktree-setup-pr-local` (requires `--worktree-setup`, keys and links;
never read from repository configuration) extends each run worktree's
in-process link registration in `checkout_verification` with an environment
state: still-shared `links`, `private` paths (link name to the setup digest
they were built for and the (st_dev, st_ino) recorded after the build, kept
allocated by an open descriptor so a recreated path cannot reuse it), the base
digest of the shared environment, a `writable` flag, a generation and an
orchestrator-generated nonce. A worktree is writable unless the run is
sandboxed and its agent holds only CLI-enforced read-only roles. Re-preparation
of an already registered path keeps the registration, validates its links and
private paths, and never recreates links. `reconcile_pr_local_environment`
runs only at orchestrator-owned points outside turns and before the baseline
capture: `record_checkout_baseline` (a failure poisons, never raises),
`establish_initial_baseline` and `refresh_after_sync` (a failure raises
`CheckoutVerificationError`). It privatizes a writable registration that still
holds links (so no writable turn ever runs with a link into the shared clone)
and otherwise rebuilds when the setup digest at HEAD differs from the
environment's digest. Before any mutation it validates the root against the
recorded alias and identity, requires the tracked tree to equal HEAD (the
stage-1 predicate, including index flags and submodules, plus the absent-key
shadow check; otherwise it defers and records the offending paths for the
remedy), refuses staged or HEAD-owned paths, removes each path
descriptor-relative (symlinks and files unlinked, directories by fd-based
`rmtree`), runs the setup command in the worktree with the stage-1 process-group
handling, and requires each path to be a real directory or file before
`mark_private`. A pre-turn backstop in `agents/registry.py` refuses a turn
while links remain if the registration is writable or the live turn is not a
sandboxed read-only turn. Fingerprints, protected sync cleaning (selected when
links or private paths exist) and test-evidence snapshots validate private
paths with `private_path_intact` before exempting them, and the snapshot adds
one `worktree-env` record (nonce, generation and digest) to the tracked digest,
so evidence from another environment at the same HEAD is neither current-head
nor able to supersede. With the flag off no record is added and tracked
digests are unchanged.

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
launch so no target starts after the owning turn ends. Each owned broker
request also has its own cancel token and a request-lifetime client monitor:
the client acknowledges every heartbeat, and EOF, a reset, a wedged send or no
acknowledged progress for 30 s marks the client lost. A lost client's request
is cancelled while it waits for host capacity, in probing or fixture setup
(nothing launches, and its lane, worker-budget lock and reservation are
released) or while its target runs (the whole process group, including
descendants that ignore SIGTERM, is confirmed gone before the lane is released,
and a telemetry-unverified, non-evidence row is journaled). A client that does
not announce acknowledgements gets a visible no-test refusal before any lock is
taken. No-test refusals reach the client's own output and name a same-broker
lock holder. See
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
requested slices of at most one second on a process-level cancel event in a
waiter registry. Short timed slices bound deferred pending-signal observation
when the blocking wait misses an interrupt. Main-thread reviewer observation
uses finite `wait(..., FIRST_COMPLETED)` polls over the remaining futures;
discuss uses finite all-completed polls before collecting results in configured
order. Both use `QUOTA_WAIT_SLICE_SECONDS`, because slicing worker quota
waits alone cannot bound a main thread blocked in executor observation. Empty
polls are neither failed turns nor overall deadlines. Reviewer results are
consumed once in completion-driven order, and discuss retains its publication
barrier. Scheduling and executor cleanup
can add overhead; this is not an absolute one-second wall-clock shutdown guarantee.
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

### Agent-loop executable pin

Agents are told to run `agent-loop run-tests -- <command>`, so every agent
subprocess must resolve the orchestrator's own install, not the first
`agent-loop` on its inherited `PATH` (#1632). `agent_executable.py` writes a
private, content-addressed POSIX `sh` launcher named `agent-loop` under
`<scratch>/agent-loop-exe/<digest>/`, using the same atomic-replace pattern as
the coder `gh` guard; both write through a unique `mkstemp` staging file, and
a process-wide lock lets parallel reviewer turns share one first install. It execs the lexical `sys.executable` with `-I` and
prepends the orchestrator's effective import path, frozen at install time
without empty or relative entries, so a coder's `PYTHONPATH`, its cwd and
user-site installs cannot select other code. Both launch-environment sites in
`runner.py` (pipe and PTY) apply `agent_executable_environment` after the
backend env merge and before the broker environment and containment
preparation. The two direct agent spawns outside the runner, the Gemini
repair backend and the legacy repair pass in `repair.py`, build their whole
environment with `pinned_agent_environment`, which merges the guard
environment and then applies the same pin. It puts only the launcher
directory first on `PATH` (never the whole venv `bin`), removes only earlier
copies of that directory so launches never stack it, keeps every other
inherited entry in order (including empty current-directory entries), and
exports the launcher path as `AGENT_LOOP_EXECUTABLE`. Because the pin is
applied last it composes with the coder CI-dispatch guard: the final `PATH`
is `<launcher dir>:<guard dir>:<pre-guard PATH>`, and the guard still strips
only its own directory. The read-only inspector builds its own closed `PATH`
and is unchanged. Installation is fail-soft: an `OSError` logs one warning
and the agent keeps its inherited `PATH`. Windows is out of scope.

The launcher is the only run-tests wrapper candidate. `_wrapper_candidates` and
`resolve_wrapper_prefix` in `test_runtime.py` return neither an ambient-`PATH`
`agent-loop` nor the unpinned `sys.executable -m coding_review_agent_loop.cli`
spelling. When the launcher is unavailable or fails its `run-tests --preflight`
probe, the sandboxed coder grant is omitted with the existing warning and no
wrapper is rendered. Report-side parsing still recognizes coder-typed `-m` and
bare spellings. At startup, issue, task, pr, discuss and managed-pr log a
warning naming both paths when the agent-visible `agent-loop` is not the
launcher, and an informational line when the inherited `PATH` holds another
`agent-loop`. Neither line blocks the run. The preflight is read-only: it
computes the content-addressed launcher path and checks that it is installed
or installable without writing to the scratch root, because a run writes
nothing there before it needs to; the first agent launch installs the
launcher.

**Receipt tool identity.** The `run-tests` client captures its own tool
commit, dirty flag and package path (bounded at 3 s, never raising, and
refusing a Git executable inside the assigned workdir or the cwd's enclosing
checkout, found by a `.git` entry without running Git) and sends
them to the broker as the reserved `AGENT_LOOP_RUN_TESTS_TOOL_COMMIT`,
`AGENT_LOOP_RUN_TESTS_TOOL_DIRTY` and `AGENT_LOOP_RUN_TESTS_TOOL_PACKAGE`
environment entries, so the broker request schema is unchanged and old and
new clients and brokers interoperate. The broker pops them before the target
launches; they are also environment-identity exclusions. Each journaled
observation gets a `ToolIdentity` comparing the client with the orchestrator's
process capture: `match` (equal commits), `mismatch` (different commits, or
different package digests when a commit is unknown), `unidentified` (an older
client sent nothing) or `unknown`. Only commits and 12-hex package digests are
stored; raw package paths go only to the orchestrator's stderr log. The record
is projected as its own `tool_identity` field outside the four-caveat cap and
decoded defensively, so legacy rows are unchanged and a malformed value
decodes as absent. It is unauthenticated diagnostic data. It never changes
provenance, supersession or the evidence gate: the broker, not the client,
executes the target, so a flagged receipt stays parent-observed. The coder
follow-up receipts section, the unchanged-head stop and the evidence-stall
decision render the derived `run-tests-tool-mismatch` or
`run-tests-tool-unidentified` diagnostic. Standalone (broker-less) telemetry
rows record `tool_commit`, `tool_dirty` and `tool_package_digest` and stay
telemetry-unverified.


Structured test reports accept non-execution option values confined both lexically
and canonically to `scratch_root() / "scratch"` for display (#1738). Generic
options such as `--input`, `--baseline`, and `--sarif` support separate and equals
forms, including managed wrappers and nested shells. Traversal and symlink
escapes fail closed; repository stores and run checkouts under the shared root
are excluded. Each occurrence is checked: targets, operands after `--`, programs,
working directories, and execution-affecting config/ignore options retain their
guards. Fresh handoffs and resumed reviewer context share this display policy.
It grants no receipt or matrix authority. Strict citations, freeform reports,
and failure reconciliation retain their existing checkout requirements.
Generic equals-form values such as `--junitxml=/tmp/report.xml` and
`--basetemp=/tmp/pytest` retain their prior classification; the display policy
does not turn them into execution inputs. Known config and ignore option values
remain guarded in both separate and equals forms.

Report guards, protocol receipt matching and receipt rendering share the strict
`test_runtime.py` managed wrapper grammar (#1768). Rendering projects original
citation text before redaction and compares canonical redacted inner argv, also
using that key for conflict detection. Equivalent wrapped and bare citations
match broker journal argv without revealing wrapper memory locations; distinct
inner argv or claims still conflict. No journal migration or path equivalence
is introduced. Prefix execution inputs and sibling shell clauses remain guarded;
malformed wrappers fail closed. Unknown long options without `=` never consume
an option-shaped next token: `node --test --require <scratch-shim> tests/test.js`
therefore retains its confined scratch value for display. The inner `--` still
ends option scanning, and positional, config, cwd, program, traversal and symlink
guards remain intact. Display acceptance grants no strict citation eligibility.

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

## Internal exact-head image capture

`review_images_config.py` and `review_images_capture.py` expose a disconnected
library service. There are no public visual flags, provider calls, PR dispatch,
CI policy changes, screenshot clearance or durable visual history in this phase.
`CaptureConfig` defaults to no path patterns or commands. `eligible` performs
case-sensitive matching of normalized changed paths, including both rename
names and deleted names, without filesystem access. Disabled calls allocate
nothing and do not import the capture workers.

The operator supplies a command, optional setup and explicit read-only offline
runtime grants. Grants are restricted to trusted system runtime installations;
checkout configuration cannot select grants. Whole installation prefixes such as
`/usr`, `/usr/local` and `/usr/share` are refused; choose specific binary/library
directories or individual runtime assets instead. Capture requires Linux, trusted
Bubblewrap >=0.9.0 with `--size`, user/mount/PID/IPC/UTS/network namespaces and
systemd >=253 with delegated cgroup-v2 memory/pids/cpu enforcement. There is no
process-group or unsandboxed fallback. Missing prerequisites return
`isolation-unavailable`. No host tmpfs provisioning or scratch environment
variable is required. No runtime, test or optional image-library dependency is
added. Shell/browser/Python runtimes, libraries, fonts and offline caches must
be available through the operator's grants; downloads and external networking
are unavailable.

`review_images_worktree.py` verifies a caller-pinned full commit OID with
`secure_git.local_command`, imports only local objects into a private independent
repository, and checks out a fresh detached worktree. The staging root identity
is held until cleanup; an unexpected replacement is left untouched. It never prepares a shared
store or dependency link. Confined Git commands cannot launch hooks, filters,
submodules or network helpers. Staging limits both compressed and expanded
input. A bounded descriptor channel transfers at most 100000 entries, paths of
1024 characters and the configured scratch byte limit. The trusted sandbox
loader refuses duplicate/unsafe layouts, follows no directory symlinks, preserves
executable modes and symlink text, and checks a service-pinned transfer digest
before setup. Git administration and host staging paths are never mounted into
the sandbox.

`review_images_isolation.py`, `review_images_host.py`,
`review_images_worker.py`, `review_images_mounts.py` and
`review_images_channel.py` own the launch and trusted channels. One sandbox
creates its own scratch using `--size <bytes> --tmpfs /scratch`, default 2 GiB.
Worktree, dependencies, output, HOME, temporary paths and `/dev/shm` are all
subdirectories or symlinks into that mount. A separate 64 KiB device-construction
tmpfs is read-only before execution; only named character devices are writable.
There are no writable host-disk binds. The internal root and `/proc` are also
remounted read-only. The trusted probe inventories every mount and verifies
scratch tmpfs type, nodev/nosuid, execution permission and page-rounded capacity
before loading input or running setup. Nested user namespaces are disabled and
probed, so payloads cannot gain mount capabilities to add writable filesystems. An executable probe and distinct namespace
identities must also pass. Failed checks refuse execution without fallback.

Capture alone uses `InvocationHandle`'s delegated scope path and optional CPU
quota. Existing roles retain their scope arguments and defaults. Defaults are
MemoryHigh=768 MiB, MemoryMax=1 GiB, zero swap, TasksMax=128 and CPUQuota=100%,
restricted by active aggregate limits. The shim pins its original scope root,
moves to a control leaf and enables controllers only on empty interior nodes.
The protected supervisor and Bubblewrap stay in control; fresh gated launchers
move to setup/capture payload leaves. The host receives kernel SCM_CREDENTIALS
across the PID namespace, pins a pidfd/start identity, verifies ancestry and
membership, migrates the blocked launcher, verifies again, then releases it.
Payload exec closes control descriptors and uses a minimal environment. The
non-dumpable supervisor denies payload memory/descriptor inspection. Supervisor
loss aborts publication; parent-pipe loss kills the whole hierarchical scope.

`review_images_cgroup.py` uses pinned cgroup identities, hierarchical
`cgroup.events` and `cgroup.kill`. Empty top-level `cgroup.procs` never proves
capture emptiness. The host kills and verifies the entire payload subtree after
each phase, before capture and before export. Only trusted control processes
remain for export; whole-scope emptiness precedes PNG validation and publication.
Memory high/max/OOM and task-limit events prohibit success even on zero exit.
Counters are checked at every control, interior and phase cgroup as well as the
scope root, including kernels with local-only event reporting.
CPU quota and deadlines bound saturation. The 2 GiB tmpfs is a filesystem ceiling,
not an allocation promise: tmpfs pages share the lower memory budget with all
processes. Small-tmpfs ENOSPC and memory-first failures are distinct tests.

Setup and capture run sequentially in the same sandbox, preserving private
setup dependencies. Each defaults to a separate 300-second deadline; input and
export each have a 30-second deadline. Diagnostics retain only a 64 KiB tail on
a channel separate from output framing. The monitor further trims that tail to fit
the 64 KiB encoded JSON reply budget, including escaping and proof metadata.
Supervisor diagnostic chunks also fit their control-channel budget under
worst-case JSON escaping, including non-BMP characters.
After each phase, resource counters are checked after payload teardown and before
evaluating the exit status; any resource event takes precedence over nonzero
exit or timeout classification. Positive finite limits exclude booleans.
Containment and Bubblewrap preflight precede staging. Trusted capture preflight
and monitor launches use `/` and a scrubbed environment to prevent imports from
the caller checkout. Failure classes come from trusted exception types and monitor
state, never diagnostic text or output filenames.
Failures are bounded typed outcomes (`isolation-unavailable`, `failed`,
`resource-limit`, `timeout`); cancellation tears down the scope and propagates.
Ordinary-review continuation remains the caller's responsibility.

Both stages receive service-owned `AGENT_LOOP_REVIEW_IMAGES_HEAD_SHA` (the
verified caller-pinned OID) and `AGENT_LOOP_REVIEW_IMAGES_MANIFEST` (absolute
`/scratch/output/manifest.json`). Inherited values cannot override these.
The manifest requires `head_sha` and nonempty `images` with unique `id`, relative
`path`, `viewport: {width, height}`, `locale` and `state`. Optional `baseline_path`
has identical confinement; finite numeric `diff_score` is advisory. Duplicate
JSON keys, control characters, unsafe paths and malformed metadata are rejected.
Text limits are ID 128, locale 64, state 256 and path 1024 characters. Viewport
dimensions are positive bounded integers, independent of PNG dimensions.

The supervisor pins scratch and output root descriptors before repository
execution. Only after payload emptiness does it read output relative to that
held output descriptor; replacing the output root invalidates the attempt. Every path component uses no-follow opens. Symlinks, hardlinks,
special files, size overruns and changed file identities/metadata are refused.
Export consists of bounded records and an explicit completion, never payload
stdout. Parent validation independently checks the pinned head/schema, inventory,
record counts, sizes, duplicates, truncation and trailing data. Whole-scope
teardown must pass before these copied bytes reach validation.

`review_images_png.py` checks PNG structure with only `struct` and `zlib`.
A second no-network Bubblewrap worker receives bounded copied bytes, never
capture/snapshot paths or writable snapshot descriptors. Input descriptors are
reopened read-only; validator stdout uses the null device rather than a host file. Its runtime allowlist is
narrowed from the operator grants to root-owned system Python, its standard
library and their ELF dependencies (inventoried with trusted system `ldd`);
browser binaries and broad capture grants are absent. It has its own verified
8 MiB scratch, read-only root/dev, at most 256 MiB address space/cgroup memory,
zero swap, 16 tasks, 100% CPU quota and a single 30-second whole-set deadline.
No capture worktree is present. It checks signature, every chunk length/type/CRC,
first single IHDR, legal bit-depth/color-type combinations, PLTE ordering/size,
one consecutive IDAT run and final empty IEND without trailing bytes. Interlacing,
APNG and unknown critical chunks are refused. Incremental zlib output is bounded
by `height * (1 + ceil(width * samples * bit_depth / 8))`, including filter bytes.
Fixed buffers, valid scanline filters, exact decompressed size, EOF and absence of
unused/trailing compressed data are required. Pixel reconstruction and baseline
comparison belong to the repository's capture/compare command.

Manifest defaults are 16 entries, 256 KiB JSON, 8 MiB per image/baseline, 64 MiB
aggregate encoded bytes, 16 million pixels per file, 64 million aggregate pixels
and maximum dimension 16000. Baseline references count toward aggregate limits.
Any invalid image, worker crash/reply, resource event, deadline or uncertain
cleanup discards the entire unpublished set. `review_images_manifest.py` hashes
actual copied bytes and binds the head, command/config identity, metadata and
file digests into a deterministic set digest. Per-host namespace identities are
probe evidence and are excluded from configuration identity. Publication exposes sealed,
read-only memory descriptors and immutable metadata. `SnapshotSet` survives all
sandbox/worktree cleanup and belongs to its caller, which must `close` it or use
its context manager. This does not establish durable retention/resume authority.

Security tests use `AGENT_LOOP_CAPTURE_SECURITY_REQUIRED=1` to make unavailable
host prerequisites fail rather than skip. Real setup/capture/export/PNG-worker
and delegated cleanup receipts are required evidence; ordinary CI skips or mock
launches cannot demonstrate confinement.


## Decisions

[Decision log](decisions/execution-and-concurrency.md)
