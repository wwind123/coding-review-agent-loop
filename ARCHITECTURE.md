# Architecture

`coding-review-agent-loop` is a local Python orchestrator around authenticated
agent CLIs, Git, and GitHub. Its primary job is to turn an issue or existing PR
into a bounded code-review loop with explicit evidence, resumable state, and
optional CI qualification and merge. It is not a hosted agent service or a
model API gateway.

This file is the root overview of the architecture set: the component map,
system-wide invariants, the main flows, and an index of the detail documents
under `docs/architecture/`. Start with the [README](README.md) for installation
and examples; use the [CLI guide](docs/local_agent_loop.md) for exact options,
schemas, and recovery contracts. Diagrams in the detail documents show responsibility and
data flow, not a strictly enforced Python import hierarchy.

## Component map

Source paths are relative to `src/coding_review_agent_loop/`. Each detail
document holds the full description.

| Component | Role | Main source paths | Detail doc |
| --- | --- | --- | --- |
| Entry points and configuration | Parse modes; resolve role-specific models, effort, base, and policy before invocation. | `cli.py`, `config.py`, `agent_permissions.py` | [components](docs/architecture/components.md) |
| Lifecycle loops | Issue, task, plan-first and PR loops behind a re-export facade. | `orchestrator.py`, `issue_loop.py`, `plan_first_loop.py`, `issue_implementation.py`, `pr_loop.py`, `pr_loop_decisions.py`, `pr_loop_support.py`, `discuss_loop.py` | [planning](docs/architecture/planning.md) |
| Validated agent turns and response contracts | Run every role through one validated turn; validate structured responses; bounded repair. | `validated_agent.py`, `response_validation.py`, `protocol.py`, `protocol_discuss_types.py`, `repair.py`, `repair_preservation.py` | [testing and doc maintenance](docs/architecture/testing-and-doc-maintenance.md) |
| Plan assembly and presentation | Assemble semantic plan patches; present large plans and matrix evidence; scope plans to the issue text. | `plan_assembly.py`, `comment_rendering.py`, `plan_issue_scope.py` | [plan context and presentation](docs/architecture/plan-context-and-presentation.md) |
| Review scheduling | Decide which reviewers inspect a PR head or a candidate plan. | `review_scheduling.py`, `plan_review_scheduling.py`, `review_rounds.py` | [review scheduling](docs/architecture/review-scheduling.md) |
| Findings, evidence and convergence | Carry findings and dispositions; detect step-back, non-convergence and evidence stalls. | `unresolved_items.py`, `review_step_back.py`, `evidence_stall.py`, `evidence_conversion.py`, `finding_history.py` | [findings and convergence](docs/architecture/findings-and-convergence.md) |
| Reviewer seats and boards | Named seats, per-invocation boards, the reviewer floor, recorded board lineages. | `reviewer_seats.py`, `reviewer_floor.py`, `board_change.py`, `board_amendment.py`, `plan_verification.py` | [reviewer boards](docs/architecture/reviewer-boards.md) |
| Durable round state and transport | Reconstruct rounds from GitHub comments; persist bounded sidecars. | `round_state.py`, `round_state_scheduler.py`, `round_transport.py`, `issue_body_limits.py` | [round state and resume](docs/architecture/round-state-and-resume.md) |
| GitHub and protocol trust | Live GitHub reads, verified writes, and separation of untrusted text from tool-owned records. | `github.py`, `github_issue_references.py`, `protocol_markers.py` | [round state and resume](docs/architecture/round-state-and-resume.md) |
| Issue/PR association | Bind the issue set, approved plan, and canonical PR. | `issue_pr_handoff.py`, `issue_pr_provenance.py`, `pr_contract.py`, `managed_pr.py` | [managed CI and merge](docs/architecture/managed-ci-and-merge.md) |
| CI and merge gates | Interpret the check board, qualify exact heads, guard merges. | `checks.py`, `ci_health.py`, `managed_ci.py`, `managed_ci_recovery_commands.py`, `managed_ci_intent_validation.py`, `coder_ci_guard.py` | [managed CI and merge](docs/architecture/managed-ci-and-merge.md) |
| Checkouts and process execution | Per-run worktrees, the Git boundary, containment, test workers. | `run_worktrees.py`, `worktree_link_integrity.py`, `secure_git.py`, `git_transport.py`, `runner.py`, `containment.py`, `test_workers.py` | [execution and concurrency](docs/architecture/execution-and-concurrency.md) |
| Architecture and prompt context | Render issue, plan, feedback and bounded advisory architecture context. | `prompts.py`, `prompt_context.py`, `memory.py`, `architecture_context.py`, `architecture_contract.py` | [testing and doc maintenance](docs/architecture/testing-and-doc-maintenance.md) |
| Local evidence, history and triage | Test evidence, salvage, usage, history export and advisory triage. | `test_runtime.py`, `local_test_evidence.py`, `salvage.py`, `usage.py`, `history_export.py`, `triage.py` | [components](docs/architecture/components.md) |

## System-wide invariants

- The orchestrator validates every agent response. Model responses propose
  verdicts and actions; they are not authoritative Git state, CI results, or
  permission grants, and repair is bounded format correction.
- Durable workflow state lives on GitHub. There is no application database;
  local state is invocation evidence or advisory orientation.
- Untrusted issue/PR prose and model output stay separate from tool-owned
  protocol records. A marker or model signature alone is not authentication.
- Approvals and CI are head-bound. Every required reviewer must approve the
  exact final head, and merges are head-guarded.
- Missing input, interrupted commands, and infrastructure failures are never
  equated with approval.
- Architecture text is advisory orientation, never a permission channel or a
  correctness waiver; source inspection remains required.
- `orchestrator.py` is a re-export facade over lifecycle modules that form an
  acyclic import chain.

## Main flows

- **Issue run.** Load and validate live context, implement the issue, validate
  the reported PR, post the canonical issue-to-PR handoff, and enter the PR
  loop. See [planning](docs/architecture/planning.md).
- **Plan review.** In plan-first mode a planner and plan reviewers iterate on a
  canonical plan until approval; plan-only and decomposition-only runs stop
  with a recorded outcome. See
  [review scheduling](docs/architecture/review-scheduling.md).
- **PR loop.** Reviewers inspect the verified head, the orchestrator
  reconciles findings, and the coder addresses feedback and CI failures until
  the required reviews are satisfied or a bounded stop applies. See
  [findings and convergence](docs/architecture/findings-and-convergence.md).
- **Managed CI.** A reviewed candidate receives an authenticated exact-head
  workflow dispatch, and qualification requires correlated exact-head evidence
  before a head-guarded merge. See
  [managed CI and merge](docs/architecture/managed-ci-and-merge.md).
- **Resume.** A rerun reconstructs state from authenticated GitHub records and
  revalidates it against the live PR, issue, plan, requirements, and policy.
  See [round state and resume](docs/architecture/round-state-and-resume.md).

## Detail documents

- [System boundaries and components](docs/architecture/components.md)
- [Planning and implementation](docs/architecture/planning.md)
- [Review scheduling](docs/architecture/review-scheduling.md)
- [Findings and convergence](docs/architecture/findings-and-convergence.md)
- [Managed CI and merge](docs/architecture/managed-ci-and-merge.md)
- [Round state and resume](docs/architecture/round-state-and-resume.md)
- [Execution and concurrency](docs/architecture/execution-and-concurrency.md)
- [Testing and document maintenance](docs/architecture/testing-and-doc-maintenance.md)
- [Plan context and presentation](docs/architecture/plan-context-and-presentation.md)
- [Reviewer seats and boards](docs/architecture/reviewer-boards.md)

Each detail doc's decision history lives in a paired append-only log under
`docs/architecture/decisions/`. The logs are not architecture-set members.
