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
reference.  A plan round fails closed when those two exceed
``MAX_GITHUB_BODY_CHARS``.  ``projected_cliff_chars`` extrapolates each sample
linearly to that point.

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
    def projected_cliff_chars(self) -> int:
        """Canonical size at which this plan's anchor would stop fitting.

        Assumes the visible text grows in proportion to the canonical text
        while the metadata floor stays fixed: every field that grows with the
        plan is spillable.
        """
        budget = MAX_GITHUB_BODY_CHARS - self.metadata_floor_chars
        return self.canonical_chars * budget // max(self.anchor_visible_chars, 1)


def _reference_anchors(value: object) -> set[str]:
    anchors: set[str] = set()
    if isinstance(value, Mapping):
        for key in _REFERENCE_KEYS:
            if key in value:
                anchors.add(str(value[key]))
    return anchors


def _marker_reference_anchors(body: str) -> set[str]:
    anchors: set[str] = set()
    for pattern in (_EXECUTION_RECOMMENDATION_RE, _RISK_TEST_MATRIX_RE):
        for match in pattern.finditer(body):
            try:
                parsed = json.loads(base64.urlsafe_b64decode(match.group("payload")))
            except ValueError:
                continue
            anchors |= _reference_anchors(parsed)
    return anchors


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
        anchors = _marker_reference_anchors(body)
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
            )
        )
    return samples


@dataclass(frozen=True)
class PlanTransportSummary:
    samples: int
    largest_published_chars: int
    median_canonical_chars: int
    median_compression_ratio: float
    median_visible_ratio: float
    largest_metadata_floor_chars: int
    lowest_projected_cliff_chars: int
    median_projected_cliff_chars: int


def summarize(samples: Sequence[PlanTransportSample]) -> PlanTransportSummary:
    if not samples:
        raise AgentLoopError("No measurable planner rounds.")
    return PlanTransportSummary(
        samples=len(samples),
        largest_published_chars=max(sample.canonical_chars for sample in samples),
        median_canonical_chars=int(statistics.median(s.canonical_chars for s in samples)),
        median_compression_ratio=round(
            statistics.median(sample.compression_ratio for sample in samples), 3
        ),
        median_visible_ratio=round(
            statistics.median(sample.visible_ratio for sample in samples), 3
        ),
        largest_metadata_floor_chars=max(sample.metadata_floor_chars for sample in samples),
        lowest_projected_cliff_chars=min(sample.projected_cliff_chars for sample in samples),
        median_projected_cliff_chars=int(
            statistics.median(sample.projected_cliff_chars for sample in samples)
        ),
    )


def _fetch_issue_bodies(repo: str, issue_number: int, gh_cmd: str) -> list[str]:
    result = subprocess.run(
        [
            gh_cmd, "api", "--paginate", "--slurp",
            f"repos/{repo}/issues/{issue_number}/comments?per_page=100",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    pages = json.loads(result.stdout)
    return [str(comment.get("body") or "") for page in pages for comment in page]


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
                    "projected_cliff_chars": sample.projected_cliff_chars,
                }
            )
        )
    if samples:
        print(json.dumps({"summary": summarize(samples).__dict__}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
