# Findings and convergence — decisions

Decision log for [Findings and convergence](../findings-and-convergence.md). Entries are append-only: a
replaced entry gets a `Superseded by #M: <clause>.` note appended and is
never rewritten or deleted. The detail doc describes the current design.

- Reviewer prompts limit blocking findings to stated or signed requirements, regressions in touched behaviour and introduced security or data-loss risk, and future follow-ups are accepted on blocking reviews but never become coder or planner work (#1389).
- Human-only exact-head evidence is a separate channel from findings: an obligation owned by the requesting reviewer, answered through a freeze boundary at a clean head (#1068).
- Risk-matrix rows whose evidence cannot be tool-captured convert into the same obligation kind, automatically after N capture-limited attempts or on operator request, with signed-evidence references minted only on owner clearance at the frozen head (#1325).
- Conjunctive findings carry ordered sub-items whose status is reviewer-authoritative, while coder `addressed_sub_items` stay advisory claims (#958).
- Malformed sub-items in a reviewer response degrade to a plain finding with a recorded degradation instead of rejecting the envelope (#924).
- The PR fix loop reuses the plan step-back classification, clustering findings by mapped location across heads, so repeated new findings near one span trigger one generalizing coder turn and a reviewer sweep (#1251).
- A separate non-convergence stop catches items that persist across K+1 reviews regardless of file, because the clustered trigger only counts new findings near one span (#1372).
- The evidence-only stall stop ends a run whose only blocking items are tagged with unsatisfied risk-matrix rows for K rounds, derived from persisted evidence so a resume recomputes the same decision (#1324).
- The unchanged-head stop grants one extra review round per head when the follow-up superseded a parent-observed failure with a receipt from its own frozen acquisition catalog that a surviving citation also cites; uncited supersession and absence from the bounded projection never count, so the exception stays bounded and tool-decided (#1626, 2026-10-24).
- The PR step-back coder turn may answer `Approach alternative:` instead of generalizing; that head-bound answer stops for a human decision after the managed-CI continuity grant and again at round start on resume, because redesigning beyond the approved plan is a human scope decision (#1628, 2026-10-24).
- An admitted evidence-conversion record bounds same-round peer visibility, resume and partial-round recovery, so a same-head conversion repeat no longer trips the partial-publish guard on its own pre-conversion pass, while repeat-pass publications stay peers; the round is not advanced, because round budgets and resume would change for no extra isolation (#1664, 2026-10-25).
- A full-board pass that overrides a staged scheduler decision rewrites the decision to the board that runs, and a recovery whose requested conversion boundary cannot be located advises no deletion, so a tool-created conversion repeat neither fails qualification on contradictory scheduler metadata nor lists its earlier pass for deletion (#1664, 2026-10-25).
