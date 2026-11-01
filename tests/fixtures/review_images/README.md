# Pinned image CLI contracts (#1794)

Provider-free Linux x86-64 conformance was captured on 2026-11-01 using the
committed harness, inert credentials, isolated homes/config and bubblewrap's
unshared network namespace. The fake provider thread and CLI share only that
namespace's loopback interface. No remote provider is reachable.

| CLI | Explicit model | Executable SHA-256 |
| --- | --- | --- |
| Codex 0.166.0 | `gpt-6.1-sol` | `b8a11fedf24b2dfb59a70c9e52b030f9a508dcd49cd2dd0d41956f4c785156b8` |
| Claude 2.1.316 | `claude-sonnet-4-6` | `fd13c70000fec3f2116dfbcbb6121210b4a29328bf0e46210a59dd594a67d824` |

Codex's released source revision is
`593389380740f7b4762771573c447ccfdc59742f` (`rust-v0.166.0`). The parser contract
is in `codex-rs/exec/src/cli.rs` and
`codex-rs/utils/cli/src/shared_options.rs`: the shared image option accepts one
or more values, so use one equals-form option per source and an explicit option
terminator before the single stdin positional. The runtime input contract is
the session's current-turn user message with native `input_image` blocks,
correlated local-image inventory, turn-context model and task completion.
The installed release's help and actual-request harness confirmed this grammar.
The npm `@openai/codex@0.166.0` integrity is
`sha512-2Szlp1l1cz5TX1ccpX4Zok3DZrot9xeZe0km+6/1v8JmqxADwyJxKBJah4dBymkwtS25j/u6dipJusTOI97Z8Q==`.

Claude's executable was extracted from the published
`@anthropic-ai/claude-code-linux-x64@2.1.316` package (tarball SHA-1
`52585a7fe1432d036822f769720a36b8e43dfd27`). Its released structured-input
contract uses print mode, stream-json input/output, verbose events and replayed
user messages. Replays identify the exact submitted UUID and image inventory;
assistant/result events carry that user UUID and session. Model identity uses
all current-turn reported model/modelUsage entries. Mechanism documentation
alone does not admit another executable or release.

The `*-replay.json` fixtures retain the proof-bearing fields of actual
production-adapter multiple-image runs, including fresh-session model metadata.
The `*-reasoning-replay.json` captures add non-tool thinking/reasoning and
earlier commentary followed by final text. Unrelated system/tool descriptions
and developer prompts are omitted. Checkout and timestamp values are placeholders; tests rebind acquisition
IDs and native descriptor paths. The inert PNG bytes, inventory order, equal-byte
multiplicity, shared baseline references and CLI envelope shapes are preserved.
No fixture is a replacement for explicit real-release conformance.

The actual requests contained one or three identical PNG image blocks as
expected. Three distinct source paths retained multiplicity; two logical images
shared one baseline. Both request inventory and CLI proof were checked. Codex's
fake provider deliberately returned a different model string, which its CLI did
not record; matching current-turn session evidence admitted the turn. Model
identity is CLI-record evidence, not provider attestation. Unrecorded server-side
substitution is an accepted limitation; missing or conflicting CLI records fail.
Runtime comparison requires exact delivered PNG digests; unverified image
normalization or downgrade is refused rather than inferred from final text.

Run the explicit command documented in `docs/local_agent_loop.md`, setting
`IMAGE_CONTRACT_CODEX` and `IMAGE_CONTRACT_CLAUDE` to these exact executables.
Missing tools fail instead of skip. Default CI deselects this marker before
sharding and keeps the replay and negative tests.

The harness now calls `preflight_images` and `run_image_adapter` with the real
`Runner`, reviewer role permissions and model/effort environment. Twelve cases
cover both default and sandboxed modes for single, shared-baseline/equal-byte,
and reasoning/multiple-message requests. Codex uses production `--cd` and
`--output-last-message`, including real configuration passthrough handling.
Claude uses production settings isolation and exactly one `--tools` option:
the reviewer grant in sandboxed mode, and the empty tool set in default mode.
Version probes use production's fresh private home and restricted environment;
bounded alias-setup stderr warnings do not replace exact version/hash admission.
Request inventory is checked independently of successful adapter proof. Codex's
unrecorded provider substitution remains intentional in every scenario.

The `codex-tool-replay.json` capture retains a real `function_call` to
`exec_command`, its `function_call_output` containing `TOOL_CALLED`, a
`CommandExecution` completion and final assistant text. Developer tool
declarations have their descriptions removed. The fake provider emits the
harmless command once, then final text; production adapters refuse this turn in
both permission modes despite complete image requests and completion. The
explicit harness now covers fourteen cases, including these two refusals.
