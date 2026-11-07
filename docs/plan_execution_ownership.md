# Combined split ownership audit

The three symbol manifests are the complete old-to-new owner inventory:

- `tests/fixtures/prompt_support_move.json`: `prompts` to `prompt_support`.
- `tests/fixtures/round_metadata_move.json`: `round_state` to `round_metadata`.
- `tests/fixtures/plan_execution_move.json`: `plan_first_loop` to `plan_execution`.

The final execution stage moves all eight listed helpers/constants. The approved
terminal branch becomes `_execute_approved_plan`, with explicit live inputs.
The planning wrapper still scopes and resets the board stamp; execution restores
the invocation board and passes recursive planning as an explicit callback.
No reverse driver import, compatibility export, registry extension or baseline
regeneration is needed. The existing global size ceilings remain unchanged.

`test_module_split_imports.py` uses the symbol-specific AST audit across source
and tests, including aliases, quoted annotations, string and object patch targets,
and unused driver imports. Body hashes check the moved helpers after normalizing
only callback plumbing. Fresh-process imports check both orders and the facade.
Existing prompt and metadata omitted-owner regressions remain part of the combined
check; the execution regression imports both callers before patching and exercises
actual retained and extracted context reads with a local intercept.

## Patch lookup inventory

The detailed tuple inventory is `tests/fixtures/plan_execution_patch_inventory.json`.
It records each affected test path, dependency and covered owners, including
loop-carried tuples, shared owner tables and conditional aliases. A static
regression compares the committed inventory against the test callers. A tuple naming
the former driver moves to execution when review no longer binds that dependency;
shared review/execution dependencies retain both lookups. Direct object patches
use the same rule. Callback tests patch the entry point supplied by the caller.

| Path family | Reached lookups | Coverage decision |
| --- | --- | --- |
| Planning workflow fixtures in `test_plan_first_loop_*.py`, `test_board_change.py`, `test_plan_growth.py`, `test_plan_ownership.py` | Review context reads followed by approved execution reads; execution policy and implementation dispatch | Keep both context bindings; move dispatch and execution-only lookups to execution. Retain both bindings of `_current_execution_recommendation` and `log` where the fixture intercepts both paths. |
| Child disposition, staged phase and execution-policy fixtures | Child context reads, handoff publication, progress and recursive/direct dispatch | Patch execution lookups after importing the caller; supply the planning callback explicitly. Human/dry-run guards retain forbidden-call assertions. |
| Issue resume/recovery and dependency guard fixtures | Entry checks, approved execution, PR resume and child planning | Cover `issue_loop`, execution-policy helpers and `plan_execution` where each call path binds the dependency; keep the caller's planning callback patched. |
| PR workflow fixtures | PR-owned context/authentication helpers; issue/plan transitions when reached | Preserve original PR owners, replace removed driver bindings with execution bindings so existing multi-owner replacements remain coherent on transitions. |
| `test_same_model_note.py` shared `_POSTER_OWNERS` and conditional log owner | Review reconciliation publication and model-note logging | Retain the review owner: execution binds no publication function, and these fixtures intercept review logging before approval. |
| `test_issue_implementation_managed_pr.py` loop-carried gate tuples | PR context, handoff publication and PR-loop gates after an implementation conflict | Replace removed driver bindings with execution; the existing listed-owner guard still checks every row of the loop. |
| Publication/architecture/review-round fixtures | `post_issue_comment` during review publication | Retain review and existing helper owners. Execution has no `post_issue_comment` binding; its handoff helpers own publication. |

Other package modules merely importing a dependency are not presumed reachable.
Listed-owner validation is preserved. The separate omitted-owner regression proves
that validation alone cannot detect an unlisted reachable owner and that complete
coverage intercepts both actual calls without live GitHub/provider activity.

## Measured sizes

- `prompts.py`: 4674 lines.
- `prompt_support.py`: 1328 lines.
- `round_state.py`: 2749 lines.
- `round_metadata.py`: 3261 lines.
- `plan_first_loop.py`: 4785 lines.
- `plan_execution.py`: 1310 lines.

## Shared issue recovery (#1939)

`issue_recovery.py` owns plan selection, inherited-row/supersession validation,
retirement and canonical-PR recovery. `issue_loop` and resumed planning-child
dispatch call it without another entry-point claim or telemetry scope. Planning
continues via the caller's explicit callback. Test patches follow moved recovery
lookups; dependencies still used by issue/task creation retain both owners.
Only the deliberately changed dispatch and child-planning body hashes are updated.
