"""Bounding the compact prior item ledger carried in round metadata (#1003)."""

from __future__ import annotations

from coding_review_agent_loop.unresolved_items import (
    COMPACT_PRIOR_DETAILS_OMITTED_SUFFIX,
    COMPACT_PRIOR_SUMMARIES_MAX_CHARS,
    _compact_prior_summaries_size,
    bound_compact_prior_summaries,
)


def _summary(round_number: int, index: int, body_chars: int = 1_500) -> str:
    return "\n".join(
        [
            f"[P{round_number}-{index}] resolved: codex blocking item from round {round_number}",
            "Original item text:",
            "x" * body_chars,
            "Disposition updates:",
            "- codex: resolved: fixed",
        ]
    )


def test_small_ledger_is_unchanged() -> None:
    summaries = (_summary(1, 1), _summary(1, 2))

    assert bound_compact_prior_summaries(summaries) == summaries


def test_ledger_stays_bounded_across_many_rounds() -> None:
    ledger: tuple[str, ...] = ()
    for round_number in range(1, 301):
        ledger = bound_compact_prior_summaries(
            [*ledger, *(_summary(round_number, index) for index in range(1, 4))]
        )
        assert _compact_prior_summaries_size(ledger) <= COMPACT_PRIOR_SUMMARIES_MAX_CHARS

    # The newest summaries are kept verbatim; older ones degrade first.
    assert ledger[-1] == _summary(300, 3)
    assert any(entry.endswith(COMPACT_PRIOR_DETAILS_OMITTED_SUFFIX) for entry in ledger)
    assert ledger[0].startswith("[compacted] ")


def test_oldest_bodies_collapse_to_headers_before_dropping() -> None:
    summaries = [_summary(1, index, body_chars=3_000) for index in range(1, 7)]

    bounded = bound_compact_prior_summaries(summaries, max_chars=10_000)

    assert bounded[0] == (
        "[P1-1] resolved: codex blocking item from round 1"
        + COMPACT_PRIOR_DETAILS_OMITTED_SUFFIX
    )
    assert bounded[-1] == summaries[-1]
    assert len(bounded) == len(summaries)
    assert _compact_prior_summaries_size(bounded) <= 10_000


def test_bounding_is_idempotent_and_merges_omission_counts() -> None:
    summaries = [_summary(1, index, body_chars=10) for index in range(1, 40)]

    once = bound_compact_prior_summaries(summaries, max_chars=1_000)
    twice = bound_compact_prior_summaries(once, max_chars=1_000)
    assert once == twice
    assert once[0].startswith("[compacted] ")
    omitted_first = int(once[0].split()[1])

    grown = bound_compact_prior_summaries(
        [*once, *(_summary(2, index, body_chars=10) for index in range(1, 10))],
        max_chars=1_000,
    )
    assert grown[0].startswith("[compacted] ")
    assert int(grown[0].split()[1]) > omitted_first
    assert sum(1 for entry in grown if entry.startswith("[compacted] ")) == 1
    assert _compact_prior_summaries_size(grown) <= 1_000
