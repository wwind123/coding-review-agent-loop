"""Measure how canonical plan text relates to what round transport publishes (#1075).

The plan-growth size threshold compares a count of canonical plan characters,
but GitHub's limit applies to the published anchor comment.  Round transport
compresses and base64-encodes metadata and spills large fields into sidecar
comments, so the two quantities are not directly comparable.  This module
measures published plan rounds so the threshold can be anchored to evidence:

* the compression ratio ``encode_mapping`` achieves on the canonical text;
* which metadata fields spilled and how many sidecar comments the round took;
* the published anchor size, split into visible text and residual metadata.

Spill stops as soon as the anchor fits, so a published plan anchor sits just
under the limit whatever its size; the anchor size itself says little.  What
transport cannot spill is the anchor's visible text (the rendered plan outside
round metadata, with large recommendation and matrix sections already
compacted) plus the metadata floor left once every spillable field is a
reference.  When the full plan comment no longer fits, the orchestrator posts
a bounded visible digest instead (#948), whose size does not grow with the
plan.  There are therefore two points:

* the digest transition, where the full comment stops fitting and readers see
  only the digest.  ``projected_digest_transition_chars`` extrapolates a
  full-comment sample linearly to it;
* the digest's own overflow, when digest text plus the metadata floor exceed
  ``MAX_GITHUB_BODY_CHARS``.  Neither term grows with canonical text, which
  spills, so no canonical size predicts it; ``digest_headroom_chars`` reports
  how far a digest sample is from it.

The measurement functions are pure.  ``main`` fetches a repository's issue
comments with ``gh`` so the corpus can be re-measured:

    python -m coding_review_agent_loop.plan_transport_profile \\
        --repo OWNER/NAME ISSUE [ISSUE ...]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .errors import AgentLoopError
from .round_transport import (
    MAX_GITHUB_BODY_CHARS,
    ROUND_RESUME_MARKER_RE,
    ROUND_TRANSPORT_SIDECAR_RE,
    _EXECUTION_RECOMMENDATION_RE,
    _RISK_TEST_MATRIX_RE,
    _SPILL_FIELDS,
    decode_mapping,
    encode_mapping,
    hydrate_mapping,
)

# Leading words of comment_rendering.COMPACT_PLAN_DIGEST_NOTICE, which every
# digest anchor carries; a test pins the two together.
_DIGEST_NOTICE_LEAD = "> **Compact plan digest.**"

_REFERENCE_KEYS = (
    "$round_transport_spill",
    "$round_transport_execution_recommendation",
    "$round_transport_risk_test_matrix",
)


@dataclass(frozen=True)
class PlanTransportSample:
    """One published planner round, measured."""

    issue_number: int | None
    round_number: int
    canonical_chars: int
    canonical_encoded_chars: int
    anchor_chars: int
    anchor_metadata_chars: int
    metadata_floor_chars: int
    spilled_fields: tuple[str, ...]
    sidecar_comments: int
    # Every execution-recommendation and risk-matrix marker in the anchor is
    # already a transport reference, so its section is compacted and the
    # visible text is the residual transport cannot shrink further.
    transport_settled: bool = True
    # The anchor shows the bounded compact digest, not the full plan.
    digest: bool = False

    @property
    def anchor_visible_chars(self) -> int:
        return self.anchor_chars - self.anchor_metadata_chars

    @property
    def compression_ratio(self) -> float:
        """Encoded characters per canonical character (lower compresses better)."""
        return self.canonical_encoded_chars / self.canonical_chars

    @property
    def visible_ratio(self) -> float:
        """Unspillable visible anchor characters per canonical character."""
        return self.anchor_visible_chars / self.canonical_chars

    @property
    def projected_digest_transition_chars(self) -> int | None:
        """Canonical size at which this plan's full comment would stop fitting.

        Past it the round still publishes, as the bounded digest.  Assumes the
        visible text grows in proportion to the canonical text while the
        metadata floor stays fixed: every field that grows with the plan is
        spillable.  ``None`` for a digest anchor, which is already past the
        transition, and for an unsettled anchor, whose visible text still
        holds an inline recommendation or matrix that transport would spill
        and compact as the plan grows.
        """
        if self.digest or not self.transport_settled:
            return None
        budget = MAX_GITHUB_BODY_CHARS - self.metadata_floor_chars
        return self.canonical_chars * budget // max(self.anchor_visible_chars, 1)

    @property
    def digest_headroom_chars(self) -> int | None:
        """Characters left before a digest anchor itself would overflow."""
        if not self.digest:
            return None
        return MAX_GITHUB_BODY_CHARS - self.anchor_visible_chars - self.metadata_floor_chars


def _reference_anchors(value: object) -> set[str]:
    anchors: set[str] = set()
    if isinstance(value, Mapping):
        for key in _REFERENCE_KEYS:
            if key in value:
                anchors.add(str(value[key]))
    return anchors


def _marker_references(body: str) -> tuple[set[str], bool]:
    """Transport anchors referenced by recommendation and matrix markers.

    Also reports whether every such marker is a transport reference, which
    is when transport has nothing left to spill or compact in the body.
    """
    anchors: set[str] = set()
    settled = True
    for pattern in (_EXECUTION_RECOMMENDATION_RE, _RISK_TEST_MATRIX_RE):
        for match in pattern.finditer(body):
            try:
                parsed = json.loads(base64.urlsafe_b64decode(match.group("payload")))
            except ValueError:
                settled = False
                continue
            found = _reference_anchors(parsed)
            settled = settled and bool(found)
            anchors |= found
    return anchors, settled


def _sidecar_anchors(body: str) -> set[str]:
    anchors: set[str] = set()
    for match in ROUND_TRANSPORT_SIDECAR_RE.finditer(body):
        try:
            item = json.loads(base64.urlsafe_b64decode(match.group("payload")))
        except ValueError:
            continue
        if isinstance(item, Mapping) and "anchor" in item:
            anchors.add(str(item["anchor"]))
    return anchors


def metadata_floor_chars(payload: Mapping[str, object]) -> int:
    """Encoded size of ``payload`` with every spillable field a spill reference.

    Like transport, a field is replaced only when that shrinks the encoding.
    Reference digests are real SHA-256 values of the field so the floor
    carries the same incompressible bytes a published reference does.
    """
    floor = dict(payload)
    size = len(encode_mapping(floor))
    for field in _SPILL_FIELDS:
        value = floor.get(field)
        if value is None or _reference_anchors(value):
            continue
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
        trial = dict(floor)
        trial[field] = {
            "$round_transport_spill": hashlib.sha256(b"anchor" + raw).hexdigest()[:24],
            "field": field,
            "parts": 1,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "spill": hashlib.sha256(b"spill" + raw).hexdigest(),
            "encoding": "text" if isinstance(value, str) else "json",
        }
        trial_size = len(encode_mapping(trial))
        if trial_size < size:
            floor, size = trial, trial_size
    return size


def measure_plan_rounds(
    bodies: Sequence[str], *, issue_number: int | None = None
) -> list[PlanTransportSample]:
    """Measure every published planner round among one issue's comment bodies.

    Rounds with a spilled field that cannot be hydrated, or a legacy round
    without canonical text, are skipped rather than guessed.
    """
    sidecar_anchors = [_sidecar_anchors(body) for body in bodies]
    samples: list[PlanTransportSample] = []
    for body in bodies:
        matches = list(ROUND_RESUME_MARKER_RE.finditer(body))
        if not matches:
            continue
        match = matches[-1]
        try:
            payload = decode_mapping(match.group("payload"))
        except AgentLoopError:
            continue
        if payload.get("flow") != "plan" or payload.get("role") != "coder":
            continue
        hydrated, missing = hydrate_mapping(payload, bodies)
        canonical = hydrated.get("canonical_plan")
        if missing or not isinstance(canonical, str) or not canonical:
            continue
        spilled = tuple(
            sorted(field for field, value in payload.items() if _reference_anchors(value))
        )
        anchors, settled = _marker_references(body)
        for value in payload.values():
            anchors |= _reference_anchors(value)
        round_number = payload.get("round_number")
        samples.append(
            PlanTransportSample(
                issue_number=issue_number,
                round_number=round_number if isinstance(round_number, int) else 0,
                canonical_chars=len(canonical),
                canonical_encoded_chars=len(encode_mapping({"canonical_plan": canonical})),
                anchor_chars=len(body),
                anchor_metadata_chars=len(match.group("payload")),
                metadata_floor_chars=metadata_floor_chars(hydrated),
                spilled_fields=spilled,
                sidecar_comments=sum(1 for found in sidecar_anchors if found & anchors),
                transport_settled=settled,
                digest=_DIGEST_NOTICE_LEAD in body[: match.start()],
            )
        )
    return samples


@dataclass(frozen=True)
class PlanTransportSummary:
    samples: int
    largest_published_chars: int
    median_canonical_chars: int
    median_compression_ratio: float
    largest_metadata_floor_chars: int
    # Visible ratio and transitions come from settled full-comment samples
    # only; ``None`` when there are none.
    projected_samples: int
    median_visible_ratio: float | None
    lowest_projected_digest_transition_chars: int | None
    median_projected_digest_transition_chars: int | None
    # Digest anchors are past the transition; their headroom is how far each
    # is from the digest's own overflow.
    digest_samples: int
    largest_digest_visible_chars: int | None
    lowest_digest_headroom_chars: int | None


def summarize(samples: Sequence[PlanTransportSample]) -> PlanTransportSummary:
    if not samples:
        raise AgentLoopError("No measurable planner rounds.")
    projected = [
        sample for sample in samples if sample.projected_digest_transition_chars is not None
    ]
    transitions = [sample.projected_digest_transition_chars or 0 for sample in projected]
    digests = [sample for sample in samples if sample.digest]
    return PlanTransportSummary(
        samples=len(samples),
        largest_published_chars=max(sample.canonical_chars for sample in samples),
        median_canonical_chars=int(statistics.median(s.canonical_chars for s in samples)),
        median_compression_ratio=round(
            statistics.median(sample.compression_ratio for sample in samples), 3
        ),
        largest_metadata_floor_chars=max(sample.metadata_floor_chars for sample in samples),
        projected_samples=len(projected),
        median_visible_ratio=(
            round(statistics.median(sample.visible_ratio for sample in projected), 3)
            if projected
            else None
        ),
        lowest_projected_digest_transition_chars=min(transitions) if transitions else None,
        median_projected_digest_transition_chars=(
            int(statistics.median(transitions)) if transitions else None
        ),
        digest_samples=len(digests),
        largest_digest_visible_chars=(
            max(sample.anchor_visible_chars for sample in digests) if digests else None
        ),
        lowest_digest_headroom_chars=(
            min(sample.digest_headroom_chars or 0 for sample in digests) if digests else None
        ),
    )


def parse_paginated_pages(stdout: str) -> list[object]:
    """Items of ``gh api --paginate`` output: JSON arrays printed back to back.

    Older gh releases (2.45, pinned by this repository's tests) have no
    ``--slurp``, so the pages are decoded one after another instead.
    """
    decoder = json.JSONDecoder()
    items: list[object] = []
    index = 0
    while True:
        while index < len(stdout) and stdout[index].isspace():
            index += 1
        if index >= len(stdout):
            return items
        page, index = decoder.raw_decode(stdout, index)
        if not isinstance(page, list):
            raise AgentLoopError("Expected a JSON array page from gh api --paginate.")
        items.extend(page)


def _fetch_issue_bodies(repo: str, issue_number: int, gh_cmd: str) -> list[str]:
    result = subprocess.run(
        [
            gh_cmd, "api", "--paginate",
            f"repos/{repo}/issues/{issue_number}/comments?per_page=100",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        str(comment.get("body") or "")
        for comment in parse_paginated_pages(result.stdout)
        if isinstance(comment, Mapping)
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure canonical plan size against published round transport (#1075)."
    )
    parser.add_argument("--repo", required=True, help="OWNER/NAME")
    parser.add_argument("--gh-cmd", default="gh")
    parser.add_argument("issues", nargs="+", type=int)
    args = parser.parse_args(argv)
    samples: list[PlanTransportSample] = []
    for issue in args.issues:
        samples.extend(
            measure_plan_rounds(_fetch_issue_bodies(args.repo, issue, args.gh_cmd), issue_number=issue)
        )
    for sample in samples:
        print(
            json.dumps(
                {
                    "issue": sample.issue_number,
                    "round": sample.round_number,
                    "canonical_chars": sample.canonical_chars,
                    "compression_ratio": round(sample.compression_ratio, 3),
                    "anchor_chars": sample.anchor_chars,
                    "anchor_visible_chars": sample.anchor_visible_chars,
                    "anchor_metadata_chars": sample.anchor_metadata_chars,
                    "metadata_floor_chars": sample.metadata_floor_chars,
                    "visible_ratio": round(sample.visible_ratio, 3),
                    "spilled_fields": list(sample.spilled_fields),
                    "sidecar_comments": sample.sidecar_comments,
                    "transport_settled": sample.transport_settled,
                    "digest": sample.digest,
                    "projected_digest_transition_chars": (
                        sample.projected_digest_transition_chars
                    ),
                    "digest_headroom_chars": sample.digest_headroom_chars,
                }
            )
        )
    if samples:
        print(json.dumps({"summary": summarize(samples).__dict__}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
