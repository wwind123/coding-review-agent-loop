# Round state and resume — decisions

Decision log for [Round state and resume](../round-state-and-resume.md). Entries are append-only: a
replaced entry gets a `Superseded by #M: <clause>.` note appended and is
never rewritten or deleted. The detail doc describes the current design.

- Evidence-conversion authority lives in round metadata and is admitted only from comments authored by the authenticated orchestrator actor (#1325).
- Exactly one byte-exact host comment footer is tolerated, once, at read-back and at three strict-reader boundaries, because some hosts such as Claude Code cloud sessions append it to every comment (#1043).
- A coder follow-up rejected after it pushed is recoverable through `coder-dispatch` and `coder-followup-rejected` records, because the rejection discards the response but not the push (#1292).
- Signed human requirements are honour-system unless trusted actors are configured, and board amendments, child-disposition overrides and child-plan supersessions stay textual (#1022).
- A reconciled PR round is closed: a later decision that needs fresh reviewer turns opens round N+1 with the ledger settled from the round's reconciliation record, and scheduler checkpoints posted into a closed round are inert except to the amendment-lineage reader, so PRs with a stray checkpoint resume without deleting comments (#1553).
- The review-round spool moved under a non-listable `private/` directory beside the agent memory directory, a spool overlapping a reviewer-reachable tree is refused, and a legacy round is migrated by atomic rename that fails closed when both locations hold it, so resume and replay keep the #1258 classification (#1560).
- 2026-10-21: The startup force-full latch reconstruction reads a lineage-visible amendment activation (the earliest record carrying its amendment digest) even when it was posted into a reconciled round, so an operator latch recorded only on an activation-only record survives resume; strays with a null or repeated digest stay fully inert (#1559).
- 2026-10-24: A parent-observed failure restored from an earlier run may be superseded by a later parent-observed pass at the same head and tracked tree through a gated cross-restart rule. Command identity is a broker-computed digest over the checkout-relative cwd and raw argv, and the superseded failure carries a visible caveat. A keyed cross-process environment comparison was rejected because it would persist secret-derived material (#1633).
- Scheduler metadata decoders and five key sets moved verbatim from `round_state.py` to `round_state_scheduler.py` with explicit re-exports. Round entry points and patched-name callers retain their original namespace; runtime, persistence and trust contracts remain unchanged. Component source ownership is reflected in the root source-path cell and detail docs (#1695).
- Pure issue-reference parsers, regexes and evidence types moved verbatim to `github_issue_references.py` with explicit `github.py` compatibility re-exports. GitHub I/O, authorization, provenance decisions, patched-name callers and all read-only source-scan call sites remain in `github.py`; execution, persistence and trust contracts are unchanged, and its source-size allowlist entry is removed (#1697).

- Managed repair checkpoints retain optional typed failed-run correlation and verified settlement provenance; old shapes use unique authenticated source-round chronology, and restart revalidates live evidence without resetting spent budgets (#1709).

- Live scheduler-prelaunch checkpoints for open PR rounds after round 1 retain their recorded round and ledger on resume, preserving private same-round replay across issue and PR modes without changing first-round startup, recovery precedence or spool safeguards (#1774).

- 2026-10-30: PR acknowledgement aliases recover only from earliest trusted durable lineage and exact historical shapes, with per-record disposition normalization and cleared-target replay; original numbering and idempotent restart remain intact (#1791).

- 2026-10-30: Coder-dispatch records are lineage-neutral until actor admission; acknowledgement recovery uses the next ordinary carrier and accepts validated ordinary reconciliation copies. Exhausted acknowledgement repair has an explicit open-round boundary preserving ledger and numbering on same-head restart (#1791).

- 2026-10-30: All recovery transport phases are lineage-neutral for legacy acknowledgement origin selection; ordinary coder evidence fields do not veto producer eligibility, while conversion origins remain excluded. The acknowledgement failure boundary also hands off active state after an unrecorded push and reopens scheduler checkpoint admission after reconciliation (#1791).
