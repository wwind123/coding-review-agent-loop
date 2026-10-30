# Planning and implementation — decisions

Decision log for [Planning and implementation](../planning.md). Entries are append-only: a
replaced entry gets a `Superseded by #M: <clause>.` note appended and is
never rewritten or deleted. The detail doc describes the current design.

- Implementation claims may declare `test_level` and coverage gaps, and one post-acceptance gate checks the coverage map against the pushed head with at most one same-session re-ask, never through repair or the retry budget (#1290).
- Row-ID defects in matrix claims are claim defects, dropped with one degradation record instead of rejecting the envelope, and every copy of a duplicate is dropped so no arbitrary copy becomes coverage (#920, #926).
- The remaining claim-scope defects also drop the claim with one record, while authority violations and hard bounds stay fatal and are checked first (#927). Superseded by #1300: an over-long `execution_refs` list is no longer a fatal hard bound; it is de-duplicated and trimmed with a caveat.
- An `execution_refs` list of more than eight well-typed refs is de-duplicated and trimmed to its first eight distinct selectors with a caveat, rather than being fatal (#1300).
- Catalog-authority rejections never route to structured repair, since reformatting cannot change them; the run stops once as a `semantic-evidence-rejection` (#990). Superseded by #1240: the approved-plan implementation hand-off first gets one coder re-ask for a non-passing or non-authoritative selector rejection, and only a second rejection stops.
- The approved-plan implementation hand-off gets exactly one coder re-ask for non-passing or non-authoritative selector rejections, and a local `npx playwright test` spelling is recognized only when the local binary exists, which is then launched directly (#1240).
- Package-script resolution reads `package.json` statically and adopts only a single simple command the recognizer already accepts; package.json logic is never executed (#1294).
- Hierarchical execution, where a planning child recommends another staged topology, is owned by #720, so such a child stops before decision persistence or decomposition (#720).
- Direct child entry checks every earlier phase in the recorded order, not only declared dependencies, because the parent's ordered-prefix rule rejects any out-of-order handoff; completion reuses the parent's staged-progress resolution read-only, and an undelivered phase stops with exit code 5 before any handoff or agent turn (#1331).
- Planner prompts require a `Smallest alternative:` plan step and plan reviewers block on a missing or weakly rejected one; enforcement is the reviewer flag rather than a schema field, so legacy and approved plans stay valid (#1628, 2026-10-24).
- A passing selector whose only launch defect is `suite_start=unknown` is refused non-terminally at post-authentication with the `suite-start-unverified` capture disqualifier, instead of stopping the run as a `semantic-evidence-rejection`; the evidence rule is unchanged, and `run-tests` and the coder prompts say which runners are citable (#1659, 2026-10-25).

- PR follow-up report partition and citation workdir refusals use the existing single shared semantic-evidence correction allowance inside validated acquisition (#1738).

- Semantic refusals exposed by the post-repair test-observations fallback use the same shared correction allowance and terminal classification as direct and repaired candidates (#1738).

- Initial and coverage-reask implementation acquisitions share the PR-follow-up typed report/selector correction policy, one allowance per acquisition; initial exhaustion stops before PR authentication, while coverage exhaustion discards the candidate and refreshes the original PR with post-authentication audit and recovery safeguards (#1768).

- Re-approved plans may retire terminal execution history using complete inventories and resolved references; plan-first terminal handoffs defer to post-approval checks, and restart partitions current work before verified history without deleting durable records or branches (#1792, 2026-10-30).
