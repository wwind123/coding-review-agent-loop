# Review scheduling — decisions

Decision log for [Review scheduling](../review-scheduling.md). Entries are append-only: a
replaced entry gets a `Superseded by #M: <clause>.` note appended and is
never rewritten or deleted. The detail doc describes the current design.

- A signed human `reviewer-board-amendment` record may remove unavailable non-primary reviewers, read from the same comment surface as the round records it amends (#943).
- On PR runs the amendment resolver may accept the original board as the configured contract and rebinds the run to the amended board, so every selection, prompt and gate sees it (#1133).
- Only an actual reconciliation record marks a round reconciled, which lets the partial-round refusal list the comments whose deletion lets a half-published round rerun (#1142).
- Two optional planning checkpoint fields, the issue digest and the one-shot reset, retire the primary stall streak; comment-only narrowing never does, because loop-posted comments share the operator's account and cannot be authenticated as operator intent (#1112).
- A staged recommendation under `plan-only` or `implement-one-shot` stops for a human decision, or with `--plan-narrow-staged` becomes an orchestrator-owned revision obligation (#1268).
- A plan step-back episode triggers a simplify-or-re-scope revision after K consecutive new-finding primary reviews and escalates to a human decision at M, because the stall stop fires only after the time is spent (#1251).
- The PR and plan boards change only through recorded board lineages, while the planning policy stays immutable (#1376, #1377).
- A growth-independent plan sweep asks the primary for one exhaustive pass after K consecutive new-finding blocks, sharing the step-back's retirement walk; findings carry an advisory, orchestrator-persisted provenance tag so drip-feeding can be measured in the history export (#1505).
- The provenance baseline is chosen by a tiered selector (unified diff, sentence-level compact diff, full previous plan as data, unavailable) under a plan-relative diff budget and a hard character budget, because long canonical plan lines made a fixed 12k line-diff cap always fail (#1516).

- Converted captures invalidate only preceding approvals; scope-only latches preserve carry across non-capture digest changes and fresh semantic approval survives restart (#1734).
