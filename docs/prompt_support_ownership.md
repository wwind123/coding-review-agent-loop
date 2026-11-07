# Prompt support ownership audit

Phase 1 of #1927, implemented by #1931, moves the 54 symbols recorded in
[`prompt_support_move.json`](../tests/fixtures/prompt_support_move.json) from
`prompts.py` to `prompt_support.py`. The manifest records original AST body
hashes; the regression checks every moved function, decorator, class and constant
against those hashes. Role builders retain their order and text and use explicit
`prompt_support` lookups. The frozen orchestrator registry and baseline stay intact;
the facade imports moved surface names directly from their new owner.

## Patch owner and reachable path inventory

The audit inspected source/test imports, qualified aliases, quoted annotations,
literal/object patches, monkeypatch calls and shared tuple variables. No existing
`patch_attr_in` or `patch_object_in` tuple lists `prompts`; no existing mock.patch
or patch.object target names a moved prompt symbol. Existing direct monkeypatches
reach these bindings:

| Test path | Patched name | Reached binding / covered owner |
| --- | --- | --- |
| `tests/test_prompts.py`: runtime memory guidance, wrapper diagnostics, compact stable-prefix guidance and exported executable guidance | `preflight_wrapper_candidates` | `_memory_block` in `prompt_support` |
| `tests/test_pr_loop_authorization.py`: prompt wrapper diagnostics | `preflight_wrapper_candidates` | `_memory_block` in `prompt_support` |
| `tests/test_prompts.py`: root-only golden prompt fixtures | `preliminary_worker_budget` | `parallel_test_worker_guidance` in `prompt_support` |
| `tests/test_prompts.py`: inherited enforceable/delta budgets | `INHERITED_OBLIGATIONS_ENFORCEABLE_MAX_BYTES`, `INHERITED_COVERAGE_DELTA_MAX_BYTES` | retained matrix renderers in `prompts` |
| `tests/test_module_split_imports.py`: shared-binding regression | `sanitize_historical_text` | `_bounded_context_text` in `prompts` and `_architecture_detail_dirs_text` in `prompt_support`; both owners covered |

The first three patches move to the new owner because retained builders reach the
support helpers by module-qualified calls; they no longer bind those dependencies.
The inherited-budget patches stay on the builder module. Dynamic guidance-name
access in the executable fallback regression also moves to the support owner.
Other modules with the same dependency names are not reached by these test paths.

The shared-binding regression imports both callers before patching, detects an
omitted reachable owner separately from the existing listed-owner validation, then
uses the existing multi-owner helper to intercept actual calls in both modules.
All calls are local formatting calls; no provider or GitHub action is needed.
The AST ownership audit scans all source and test Python files, including aliases,
annotation strings and object/literal patch targets; it also rejects unused imports
in `prompts.py`. Fresh subprocess imports exercise both owner orders and the facade.
A focused size guard caps both prompt modules at 5,000 lines without changing the
6,000 global default or the 9,945-line PR-loop ceiling. Later phases own the other
large-file splits and final combined full-suite validation.
