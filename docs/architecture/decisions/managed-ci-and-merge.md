# Managed CI and merge — decisions

Decision log for [Managed CI and merge](../managed-ci-and-merge.md). Entries are append-only: a
replaced entry gets a `Superseded by #M: <clause>.` note appended and is
never rewritten or deleted. The detail doc describes the current design.

- Managed exact-head CI runs the complete suite as three duration-balanced shard legs of the same validated SHA, with an aggregate that keeps the original check name and proves full coverage exactly once (#1235). Superseded by #1863: the repository callers select six shards; only the shard count is superseded.
- Adopters whose tests need services or a routed matrix compose separate validate, test and publish workflows, where trust rests on the default-branch composition and API conclusions, not on attestations (#1313).
- Managed CI may target a non-default base only when the owner-controlled trusted-bases variable allows it, while the workflow always comes from the default branch, so a branch cannot supply the workflow that qualifies it (#1285).
- The selective-policy integration gap between machine-owned CI failures and reviewer-owned findings is tracked in #776 rather than solved here (#776).
- Two preflight reads fall back only on a strict HTTP 403, so managed CI works where they are refused, such as Claude Code cloud sessions, while the workflow still enforces the real variable server-side (#1040).
- A managed approved-plan PR without an issue-side handoff takes its plan binding from the PR-side authorization chain at qualification (#966).
- Records retired by a later fresh grant are history and are ignored by the qualification chain check (#1065).
- Records retired by a continuity record are likewise ignored, and a continuity record retires only records that cannot reach its new head (#1069).
- The exact-head CI repair round is authorized by the tool-minted CI obligation, because it has no blocking review to pair with (#1024).
- A rejected-coder recovery across a fresh grant is a third accepted continuity shape, walked through one shared chain authentication (#1367).
- Coder and format-repair turns on a managed PR run under a `gh` guard that refuses CI dispatch before approval, plus a post-turn dispatch audit (#1366).
- Post-approval CI watching stays opt-in without auto-merge (an end-of-run hint names `--watch-pending-ci`). Plumbing-only failures on a complete board are re-run once before any coder round, then stop on infrastructure (#1477).
- A fresh process resumes a matching `qualifying` managed checkpoint as `qualification_ready` and dispatches a fresh attempt under its own generation, rather than adopting the prior generation's attempt or re-running reviewers (#1177).
- The merge-conflict continuity transition counts only an open, live machine-authority merge-conflict obligation, matching the CI-repair predicate's authority and status checks (#1057).
- The post-turn dispatch audit claims a run as the orchestrator's own v1 dispatch only when an in-process dispatch record uniquely matches it; unconfirmed matches are listed with a typed reason and no named creator (#1414).
- PR-mode managed-CI resume derives a fresh decomposition child's staged parent from the child's own authenticated fresh-phase identity before the hash-only plan fallback; non-fresh children keep issue-mode-only parent recovery, and refusals and interrupted-run hints name the command that resumes the work (#1662).
- Recovery rendering and workflow-mirrored intent validation are pure sibling modules; `managed_ci.py` retains I/O, freshness and authorization gates and re-exports every moved binding (#1692).

- Same-head managed repair recovery observes an authenticated later successful attempt of the recorded run through a separate immutable result, preserving fresh-generation dispatch authority and carrying original result identity into automatic proof and manual publication (#1709).

- Reusable ordinary and managed shard jobs share an optional consumer-selected job budget; named pytest timeouts and the outer shard ceiling address different hangs while preserving aggregate and authorization gates, and local CI watching remains independent (#1769).

- The local watch defaults to 2400 seconds; authenticated current-attempt test failures can authorize early repair through fresh live job evidence while active intents remain attached and historical clearance retains completed-result gates (#1790).

- Timing coverage is advisory balance bookkeeping, with no CI or review threshold; retain mean-weight fallback, exactly-once guards and real-failure gates, and refresh measured durations separately from ordinary feature work (#1850).

- Both repository CI callers select six duration-balanced shards following the completed [#1851 measurements](https://github.com/wwind123/coding-review-agent-loop/issues/1851). Two ordinary all-empty `workflow_dispatch` runs per count, run one at a time on 2026-11-02 against main at `c8a887221ca0` and a trial branch differing only in shard count, all passed (#1863).

  | Shards | Run | Wall time (minutes) | Longest shard (minutes) | Runner minutes |
  | --- | --- | --- | --- | --- |
  | 3 | [41413783786](https://github.com/wwind123/coding-review-agent-loop/actions/runs/41413783786) | 12.3 | 12.0 | 34.8 |
  | 6 | [41415293654](https://github.com/wwind123/coding-review-agent-loop/actions/runs/41415293654) | 7.8 | 7.4 | 34.8 |
  | 3 | [41416254058](https://github.com/wwind123/coding-review-agent-loop/actions/runs/41416254058) | 12.0 | 11.7 | 27.9 |
  | 6 | [41417759212](https://github.com/wwind123/coding-review-agent-loop/actions/runs/41417759212) | 7.1 | 6.7 | 36.6 |

  These samples showed roughly 40% lower wall time, from about 12 minutes to about 7.5 minutes, with broadly similar runner minutes. They do not guarantee a universal speedup or runner cost and do not directly benchmark managed dispatch. Reusable single-shard defaults, budgets, routing, authorization, permissions, aggregate names, manifest formats and duration data remain unchanged. No further timing trials are needed; #1847 remains reference-only, and the operator closes #1851 after merge.

- Issue-created opening-tuple recovery distinguishes bounded unavailable or malformed PR observations from safe phase-specific field mismatches while retaining exact raw-body authentication (#1882).

- Managed-CI support consumers now import their definition owners directly; only internally used driver imports remain. Manifest-driven rewriting, symbol/patch guards and fresh-process imports protect the support cleanup without changing runtime or authority (#1906).

- Issue-created authorization, continuity and recovery now live in an acyclic authorization owner with temporary identity-preserving driver imports. Comment ingestion is shared through the API owner; body hashes and import-order checks protect the move window (#1907).
