# Testing and document maintenance — decisions

Decision log for [Testing and document maintenance](../testing-and-doc-maintenance.md). Entries are append-only: a
replaced entry gets a `Superseded by #M: <clause>.` note appended and is
never rewritten or deleted. The detail doc describes the current design.

- The architecture path names the root of an architecture set, the root plus the `docs/architecture/*.md` files it links to one level deep, so an oversized single document can be split while root-only identities stay unchanged (#1454).
- The `legacy` architecture-impact parser mode keeps the earlier synonym normalization, selected only where stored or historical text is decoded (#916).
- A reviewer's own missing or invalid `architecture_impact.status` is re-asked once with the frozen prompt and session instead of being repaired, because repair cannot supply the reviewer's decision (#1185).
- Architecture-set membership follows a configurable list of detail directories (default `docs/architecture/`) minus history exclusions (default `*/decisions/`, `doc/adr/`, `docs/adr/`) that always win, and the detail-doc index adds a bounded first-sentence summary that is dropped first and never joins an identity; an explicit directory list was chosen over admitting any linked `.md` so default-layout identities stay byte-identical and stray root links never become members, and a deeper membership walk was rejected because it would change the identity (#1499, 2026-10-15).
- A non-array coder `test_observations` is decoded when it is a JSON-array string and otherwise sent to the structured format repair before any citation is dropped; today's #927 drop remains only as the fallback after a failed repair, with a record saying so, and as the default for historical and resume re-parses, because the field is how an evidence-only follow-up reaches the reviewer (#1626, 2026-10-24).
