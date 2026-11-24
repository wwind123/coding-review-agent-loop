# Plan context and presentation — decisions

Decision log for [Plan context and presentation](../plan-context-and-presentation.md). Entries are append-only: a
replaced entry gets a `Superseded by #M: <clause>.` note appended and is
never rewritten or deleted. The detail doc describes the current design.

- A structured plan comment that overflows the body budget is posted as a bounded visible digest, while round metadata and sidecars stay the complete authority, because visible plan Markdown had no spill path (#948).
- Coder matrix evidence is shown collapsed and, on PR follow-ups, as a delta against the previous round, while canonical evidence stays complete; records written before this rendered the full list themselves (#959).
- When the issue title or body changes under a plan that is not completely approved, a plan-first resume posts a scope-boundary record and starts a fresh plan at the next round, unless `--plan-resume-keep-on-issue-edit` is given (#1390).
- An approved plan reopens only through the explicit per-invocation `--reopen-approved-plan` option: a pre-recovery gate refuses a handed-off plan (fail closed on lookup errors) and otherwise posts a schema-version-2 scope boundary with reason `operator-reopen` naming the superseded plan hash, because implicit reopening on an issue edit would discard reviewed approvals (#2010).
- An issue title/body edit under an approved plan that was not handed off stops a plan-first relaunch before recovery or any agent call, with a deduplicated issue note, unless the per-invocation `--keep-approved-plan` keeps the plan or `--reopen-approved-plan` reopens it; one shared gate runs at both plan-first dispatch entries (issue loop and parent-driven child planning), because continuing silently implemented a plan the issue no longer states (#2138).
