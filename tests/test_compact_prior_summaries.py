"""Bounding the compact prior item ledger carried in round metadata (#1003)."""

from __future__ import annotations

import random

from coding_review_agent_loop.prompts import (
    _canonical_plan_ledger_rules,
    _canonical_pr_review_ledger_rules,
    _compact_prior_ledger_block,
    CompactPriorContext,
)
from coding_review_agent_loop.round_transport import encode_mapping
from coding_review_agent_loop.unresolved_items import (
    COMPACT_PRIOR_DETAILS_OMITTED_SUFFIX,
    COMPACT_PRIOR_SUMMARIES_MAX_BYTES,
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
        assert _compact_prior_summaries_size(ledger) <= COMPACT_PRIOR_SUMMARIES_MAX_BYTES

    # The newest summaries are kept verbatim; older ones degrade first.
    assert ledger[-1] == _summary(300, 3)
    assert any(entry.endswith(COMPACT_PRIOR_DETAILS_OMITTED_SUFFIX) for entry in ledger)
    assert ledger[0].startswith("[compacted] ")


def test_oldest_bodies_collapse_to_headers_before_dropping() -> None:
    summaries = [_summary(1, index, body_chars=3_000) for index in range(1, 7)]

    bounded = bound_compact_prior_summaries(summaries, max_bytes=10_000)

    assert bounded[0] == (
        "[P1-1] resolved: codex blocking item from round 1"
        + COMPACT_PRIOR_DETAILS_OMITTED_SUFFIX
    )
    assert bounded[-1] == summaries[-1]
    assert len(bounded) == len(summaries)
    assert _compact_prior_summaries_size(bounded) <= 10_000


def test_bounding_is_idempotent_and_merges_omission_counts() -> None:
    summaries = [_summary(1, index, body_chars=10) for index in range(1, 40)]

    once = bound_compact_prior_summaries(summaries, max_bytes=1_000)
    twice = bound_compact_prior_summaries(once, max_bytes=1_000)
    assert once == twice
    assert once[0].startswith("[compacted] ")
    omitted_first = int(once[0].split()[1])

    grown = bound_compact_prior_summaries(
        [*once, *(_summary(2, index, body_chars=10) for index in range(1, 10))],
        max_bytes=1_000,
    )
    assert grown[0].startswith("[compacted] ")
    assert int(grown[0].split()[1]) > omitted_first
    assert sum(1 for entry in grown if entry.startswith("[compacted] ")) == 1
    assert _compact_prior_summaries_size(grown) <= 1_000


def _high_entropy_non_ascii_summary(rng: random.Random, round_number: int) -> str:
    # Four-byte UTF-8 code points drawn at random compress poorly.
    body = "".join(chr(rng.randrange(0x10000, 0x10FFFF)) for _ in range(2_000))
    return "\n".join(
        [
            f"[P{round_number}] resolved: codex blocking item from round {round_number}",
            "Original item text:",
            body,
        ]
    )


def test_non_ascii_incompressible_ledger_encodes_well_inside_comment_budget() -> None:
    rng = random.Random(1003)
    ledger: tuple[str, ...] = ()
    for round_number in range(1, 41):
        ledger = bound_compact_prior_summaries(
            [*ledger, _high_entropy_non_ascii_summary(rng, round_number)]
        )
        encoded = encode_mapping({"compact_prior_summaries": list(ledger)})
        # The round-comment budget is 60,000 characters; the bounded ledger
        # must leave most of it for the visible body and other metadata.
        assert len(encoded) <= 22_000, len(encoded)
    assert _compact_prior_summaries_size(ledger) <= COMPACT_PRIOR_SUMMARIES_MAX_BYTES


def test_compact_ledger_prompts_describe_bounded_lossy_history() -> None:
    for rules in (_canonical_plan_ledger_rules(), _canonical_pr_review_ledger_rules()):
        assert "append-only" not in rules.lower()
        assert "(details compacted)" in rules
        assert "[compacted] N earlier prior item summaries omitted" in rules
    block = _compact_prior_ledger_block(CompactPriorContext(("[item-1] resolved: x",)))
    assert block.startswith("Compact prior item ledger (bounded)")
