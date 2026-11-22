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

- 2026-10-30: Same-head failed acknowledgement repair bounds peer visibility and partial-round recovery with the same predicate as resume; earlier publications and the ledger boundary are retained, while later publications remain peers (#1791).
- An independently approved plan may recover an unused neutral advance through its reconciled predecessor and lifecycle-replayed future ledger without changing durable records (#1805).

- Approved publication resumes retain settlement-time item minting; refusal diagnostics use scoped unfiltered history to explain excluded approval bindings (#1805).

- Approved resume preserves an owed same-model reconciliation note, while settled rounds suppress duplicates (#1805).

- 2026-11-01: Valid terminal implementation stops preserve tracked/staged salvage before reporting and cleanup; category-aware rendering distinguishes valid responses, while default rerun discovery still requires an eligible breadcrumb or a reused stable absolute log directory (#1833).

- 2026-11-06: Reviewer round metadata has additive `requirement_evidence` associations with empty legacy defaults. Each entry preserves the original signed ID, role, requester-owned request or answered item IDs, and explicit mixed-source leftover instructions. Resume restores associations alongside the requester's own evidence requests, then projects only against current admitted signed sources and owned native items. Missing legacy metadata never consumes an instruction. Acceptance still depends on current live content-bound citations and exact-head requester attestations; terminal publication, partial acceptance and push invalidation use the existing evidence release/freeze records (#1922).

- 2026-11-06: Cross-round classification conflicts cannot prevent fresh reviewer turns on resume; invalid replay entries retain conservative source enforcement. (#1922, #1924).

- 2026-11-07: Live PR observations bind round candidates to complete REST identities before decoding, excluding foreign anchors and chunks together while retaining raw prose and index joins. A per-call-site AST inventory guards the live-reader boundary; owned corruption and downstream evidence, board, approval and publication gates remain strict (#1860).

- Plan and decomposition readers use the shared author-bound family projections, preserving separate parent/child surfaces, raw discussion and stable indices; historical foreign controls are ignored and logged before payload parsing (#1965).

- Legacy PR contracts and remote salvage select invocation-only body projections before decoding; capture reconstruction has no reviewer-role exemption. Complete authenticated acquisition preserves raw discussion, trusted transaction semantics and local salvage fallback (#1966).

- 2026-11-08: Final integration records per-family admission ownership and historical replay consequences, with recursive reader inventory review and repeated lifecycle restart regressions; existing author sets and root invariants remain unchanged (#1976, #1959).

- 2026-11-08: A same-head reconciled PR round anchors resume even when all settlements were carried exact-head approvals. PR amendment round suggestions share the admitted resume reader without a round-1 fallback and a resumability check, and say no safe round exists instead of printing an unsafe template; the partial-round peer-independence guard is unchanged (#1981).

- 2026-11-20: Signed-source role/field consistency violations route through the bounded reviewer-judgement re-ask instead of a fatal stop. A published review whose only gap is the signed-source classification resumes as classification-pending or classification-complete in an unreconciled round with an unchanged signed requirement set; only the classification continuation reruns, pinned to the published verdict and findings, so recovery no longer needs deleting published reviews. Other rejections keep the deletion refusal with a named cause (#2035).

- 2026-11-21: Pure signed-set growth after publication no longer forces deleting a round's reviews. When every published review of an unreconciled same-head round (on a one-reviewer board only when its carrier is still spooled) is a strict digest-superset rejection with matching identity and proves its publication unedited (complete frozen carrier, or a never-edited GitHub edit history), a tool-owned `signed-input-supersession` summary closes the round before any launch and the closed-round rollover opens round N+1. Superseded reviews are inert history and never deletion targets; the carried ledger keeps the durable evidence requests resume restores for invalidated reviews (#1922); edits, withdrawals, unproven publications and every other cause keep the refusal. The approval reader skips superseded reviews independently of the live signed set (#2054).

- 2026-11-21: `--review-unrecorded-head` converts every coder-recovery or unrecorded-advance resume outcome into an operator head review at any dispatch attempt through a single conversion point that keeps evidence-boundary ledgers and fields; a new `head_advanced_past_boundary` field keeps moved-head semantics for the release-revalidation and live-boundary predicates. The PR-loop coder follow-up comment is posted through a whole-body budget with bound levels instead of exiting on overflow; full responses (free-form text included) and degradation records stay in round metadata, and the residual overflow names the recovery command (#2059).

- 2026-11-22: A tool-owned `round-closed-incomplete` summary record and its readers land dormant (stage 1 of #2064). One closure predicate covers the new record on PR and plan flows and the legacy PR `signed-input-supersession` record (read-only for one release window). Closed-round reviewer records are inert for resume, panel evidence, scheduler history, approval carry and plan projections; the record carries the ledger, including evidence obligations at reserved item numbers, so restart rebuilds it without the inert reviews. Author authentication is unchanged; no writer exists yet (#2079, #2064).

- 2026-11-22: A closure-phase summary that does not validate (malformed payload, or a reserved obligation number naming a different ledger item) neither closes nor reconciles its round. Unrecorded-head-advance recovery after a closure boundary takes the record's ledger, including reserved evidence obligations (#2079, #2064).

- 2026-11-22: A validated closure boundary counts as a coherent head-advance handoff even with an empty ledger and no coder record, so a push after it resumes as a fresh review without admitting the closed reviews (#2079, #2064).

- 2026-11-22: In plan approval carry a closed-incomplete plan review supersedes that reviewer's earlier carried approval with a non-qualifying entry, matching the PR approval reader, so a stale approval never revives (#2079, #2064).

- 2026-11-22: The planning loop's approval-carry call sites pass raw, scope-filtered plan history so a closed-incomplete review can still supersede an older approval; carry applies the projection itself (#2079, #2064).

- 2026-11-22: A rejected PR coder follow-up that moved the head posts a coder record instead of only a rejected-follow-up summary (stage 1 of #2065). The rejection carries its final candidate and frozen acquisition catalog; droppable fields are dropped or conservatively normalized and revalidated against that catalog, otherwise the record is synthesized from git, and review runs on the new head. Optional `coder_record_source`, `coder_record_degradations` and `human_requirements_degraded` metadata keep historical records byte-stable; the flag keeps the acknowledgement obligation open. The coder comment budget ends in a claim-free synthesized fallback, so overflow never escapes. The coder-dispatch writer, recovery readers and `--review-unrecorded-head` are unchanged until stage 2 (#2086, #2065).
