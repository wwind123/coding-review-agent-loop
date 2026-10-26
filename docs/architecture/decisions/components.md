# System boundaries and components — decisions

Decision log for [System boundaries and components](../components.md). Entries are append-only: a
replaced entry gets a `Superseded by #M: <clause>.` note appended and is
never rewritten or deleted. The detail doc describes the current design.

- Lifecycle modules were extracted from `orchestrator.py` by pure moves, leaving a docstring-and-imports re-export facade, so each layer imports alone in an acyclic chain while the historical import surface used by `cli.py` and tests keeps working (#1181).
- A reviewer whose `architecture_impact.status` is the only defect is re-asked once through the shared re-ask slot, and repair may never introduce that status, because repair cannot supply the reviewer's decision (#1185).
- Step-back bookkeeping in `review_step_back.py` is pure and history-derived for both the plan and the PR fix loops, with the git reads kept in `pr_loop_support.py` (#1251).
- The step-back episode anchor takes its dissolved items and trade-offs from the step-back candidate record (#1278).
- Finding history is advisory, bounded and in-memory per loop, seeded from round records the loop already extracted, so it informs prompts without persisting anything or inferring status from historical votes (#1273).
- Coverage-map completeness is a pure, deterministic check in `risk_coverage_map.py`, run once by the approved-plan implementation seam with a single pre-review re-ask (#1290).
- Evidence-only stall classification lives in the pure `evidence_stall.py`; the dispatch-path decision and stop message live in `pr_loop_support.py` (#1324).
- Evidence conversion is pure in `evidence_conversion.py`, with I/O and the authenticated comment accessor in `pr_loop_support.py`, and `apply_signed_evidence_references` is the only producer of `operator-signed` rows, so provenance cannot come from agent text (#1325).
- `coder_ci_guard.py` is a turn-scoped `gh` guard, so managed-CI coder and repair turns cannot dispatch CI before approval (#1366).
- Operator-declared out-of-scope and non-goal sections are extracted by the pure `scope_declarations.py` into a binding-scope prompt block (#1389).
- The shared write read-back verifier in `github.py` tolerates exactly one known host comment footer (#1043).
- `agent-loop history export` is a read-only outcome export derived only from authenticated comment views; it posts nothing and makes no model call (#1433).
- `agent-loop issue --triage` is one advisory validated turn armed at the fresh-dispatch point; it never reads or changes a routing argument and every failure is ignored (#1434).
- `agent-loop triage backtest` is offline and report-only over local history-export files, so it makes no GitHub call and never changes routing (#1435).
- The workflow transaction model landed as stage A of #827: model only, with no writer emitting its records and no orchestration call site consuming them yet (#827).
- Semantic follow-up dedupe defaults to the Antigravity backend; an unresolvable helper CLI skips semantic matching with one log line instead of failing the run or adding the helper to the required agent-command preflight, and the per-call timeout is resolved at use time from the backend (#1472).
- Tests patch the extracted modules that own each name instead of the `orchestrator.py` facade, the conftest patch-propagation shim is retired, and a static guard rejects test imports and patch targets of the facade outside the split guard and its tests (#1186).
- Peer responses are hidden from incidental listing rather than sandboxed: each agent turn writes into its own 0700 `turn-<uuid4 hex>` directory under 0300 ancestors, the whole repo response tree is hardened before any launch with a mode-bit (not `os.access`) post-condition that refuses a listable child, and the validated turn removes its directories; an OS-level filesystem sandbox stays separate work (#1560).
- Gemini's git-dir response root is hidden like the shared root and released for deletion just before worktree removal or prune, and per-turn cleanup is scoped at `run_agent_result` so direct backend invocations outside a validated turn are cleaned up too (#1560).
- A launched run-tests observation whose suite start stayed `unknown` persists why (an unrecognized launcher or an unavailable probe) as a diagnostic-only closed-set field, so the post-auth diagnostic and the receipt notice need not re-classify argv; legacy rows decode to no reason and keep the generic wording (#1661, 2026-10-25).
- Stopped private trees are released explicitly through `release_unlistable_tree` before separate operator deletion; normal completion does not release shared ancestors because concurrent invocations may still use them. Test-owned trees restore and independently verify owner access only after dependent resource finalizers finish (#1688).
- Pure PR stop diagnostics, dispatch-record access and reconciliation-disposition indexing moved from `pr_loop.py` to its existing support module with re-exports, reducing the size ceiling rather than raising it when the required guidance check exposed pre-existing growth (#1688).
