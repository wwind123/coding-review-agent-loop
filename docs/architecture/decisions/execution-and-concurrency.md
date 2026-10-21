# Execution and concurrency — decisions

Decision log for [Execution and concurrency](../execution-and-concurrency.md). Entries are append-only: a
replaced entry gets a `Superseded by #M: <clause>.` note appended and is
never rewritten or deleted. The detail doc describes the current design.

- Concurrent default-dir invocations on one repository are supported by giving each run its own detached worktree over one shared store per repository and agent (#1162).
- `agent-loop transport-check` runs one production fetch with the operator's credentials and is required before merging transport changes (#1383).
- SSH transport honours a non-executing subset of `~/.ssh/config` for the literal URL host, refusing rather than escaping any unsafe value, and never runs `ssh -G` (#1380).
- Same-UID live races against `~/.ssh` are left out of scope for the SSH transport (#1371).
- Antigravity tracks quota exhaustion per quota group for the logical run, so a verified exhaustion skips the whole group with no retry instead of retrying each model (#1236).
- The Antigravity `semantic-dedupe` turn reuses the isolated `repair` path (sibling GEMINI lock, no checkout snapshot, strict empty allowlist) with a stdout-only classification contract, so a classifier never creates git metadata in its temporary directory or reads a stale response file (#1472).
- Capture-root retention is an operator setting (default 14 days, 0 disables); pruning never follows symlinks, removes expired roots whole, and deletes the marker and lease last so partial failures are retried (#1572).
- Managed checkouts are compacted under the store flock once they exceed 50 non-kept packs, by one confined exec-free `pack-objects` plus superset verification and snapshot-only deletion, because `git repack` needs exec and `gc.auto=0` stays; failures are logged and never fail the fetch (#1578).
- `import_ref` exports a thin pack excluding destination ref tips the fetched history also holds, skips the pack for a complete existing tip, and gates the CAS on a full-closure check with one full-export fallback, so destination refs are hints and never proof of completeness (#1581).
- A compaction timeout records a private per-store backoff marker beside the store lock (1 h doubling to a 24 h cap, cleared on success or at or below the threshold, failing open on any invalid or too-far-future marker), and timeout and backoff skips log a manual-repack hint (#1580).
- Pack compaction removes any multi-pack-index file, sidecar or chain, without following links, after the threshold and backoff checks and before any pack command, as `git repack -d` does; it never rewrites one, and a removal failure writes no pack and starts no backoff (#1579).
