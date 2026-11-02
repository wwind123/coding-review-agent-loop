# Managed CI and merge — decisions

Decision log for [Managed CI and merge](../managed-ci-and-merge.md). Entries are append-only: a
replaced entry gets a `Superseded by #M: <clause>.` note appended and is
never rewritten or deleted. The detail doc describes the current design.

- Managed exact-head CI runs the complete suite as three duration-balanced shard legs of the same validated SHA, with an aggregate that keeps the original check name and proves full coverage exactly once (#1235).
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
