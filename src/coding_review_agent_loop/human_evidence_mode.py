"""Exact-head human evidence mode: ``--pr-human-evidence freeze|advisory|off`` (#2063).

``freeze`` keeps the blocking exact-head evidence freeze exactly as before.
``advisory`` (the default) and ``off`` remove native and converted human
evidence obligations from the PR ledger at one normalization choke point:
``advisory`` moves each removed item onto a durable, non-blocking "Human checks
requested" list, and ``off`` drops it with a log line.  Screenshot-coverage
obligations share the evidence obligation kind but are a separate channel and
are never touched here.

The advisory list keeps each removed item verbatim (``origin_item``), so a
later ``freeze`` resume restores the original typed obligation, including a
converted row's identity and owners, instead of re-creating it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from .logging import log
from .protocol import UnresolvedReviewItem

PR_HUMAN_EVIDENCE_MODES = ("freeze", "advisory", "off")
DEFAULT_PR_HUMAN_EVIDENCE_MODE = "advisory"
# The human-checks comment's record flow: deliberately not ``pr``, so PR round
# reconstruction never treats it as a round record.
HUMAN_CHECKS_FLOW = "pr-human-checks"
HUMAN_CHECKS_HEADING = "## Human checks requested"
# Bound on the rendered comment only; the durable list is lossless.
HUMAN_CHECKS_RENDER_LIMIT = 50
_TEXT_LIMIT = 600
_OPEN_STATUSES = frozenset({"blocking", "same-pr"})

_logger = logging.getLogger(__name__)


def mode_of(config: object) -> str:
    return getattr(config, "pr_human_evidence", DEFAULT_PR_HUMAN_EVIDENCE_MODE)


def freeze_mode(config: object) -> bool:
    """Whether the blocking evidence freeze (and its classification) runs."""
    return mode_of(config) == "freeze"


@dataclass(frozen=True)
class HumanCheck:
    """One advisory human check and the complete ledger item it replaced."""

    reviewer: str
    text: str
    source_round: int
    head_sha: str | None
    kind: str  # "native" | "converted"
    origin_item: UnresolvedReviewItem

    @property
    def identity(self) -> str:
        return check_identity(self.origin_item)

    def as_dict(self) -> dict[str, object]:
        from .round_metadata import _serialize_unresolved_item

        return {
            "reviewer": self.reviewer,
            "text": self.text,
            "source_round": self.source_round,
            "head_sha": self.head_sha,
            "kind": self.kind,
            "origin_item": _serialize_unresolved_item(self.origin_item),
        }

    @classmethod
    def from_mapping(cls, value: object) -> "HumanCheck | None":
        """Decode one entry; anything malformed decodes as ``None``."""
        from .round_metadata import _deserialize_unresolved_item

        if not isinstance(value, Mapping):
            return None
        reviewer, text, round_number = value.get("reviewer"), value.get("text"), value.get("source_round")
        head, kind = value.get("head_sha"), value.get("kind")
        if (
            not isinstance(reviewer, str) or not reviewer
            or not isinstance(text, str) or not text
            or not isinstance(round_number, int) or isinstance(round_number, bool)
            or not (head is None or (isinstance(head, str) and head))
            or not isinstance(kind, str) or kind not in {"native", "converted"}
        ):
            return None
        try:
            origin = _deserialize_unresolved_item(value.get("origin_item"))
        except Exception:  # noqa: BLE001 - advisory data never fails a record
            return None
        if not is_native_human_evidence(origin):
            return None
        return cls(reviewer, text, round_number, head, kind, origin)


def decode_human_check_entries(value: object) -> tuple[dict, ...]:
    """Lenient decode of a persisted ``human_checks`` list.

    Valid entries are kept verbatim (so a re-encode is byte-stable); malformed
    entries are dropped and logged without failing the record.
    """
    if not isinstance(value, list):
        _logger.warning("Ignoring a malformed human_checks field on a persisted round record")
        return ()
    kept = tuple(dict(entry) for entry in value if HumanCheck.from_mapping(entry) is not None)
    if len(kept) != len(value):
        _logger.warning(
            "Dropped %d malformed human_checks entr(y/ies) from a persisted round record",
            len(value) - len(kept),
        )
    return kept


def is_native_human_evidence(item: UnresolvedReviewItem) -> bool:
    """A native reviewer request or a converted risk-matrix row, never a screenshot."""
    from .unresolved_items import _is_evidence_obligation

    return _is_evidence_obligation(item) and not item.screenshot_request


def check_identity(item: UnresolvedReviewItem) -> str:
    if item.converted_row_id is not None:
        return f"converted:{item.converted_row_id}:{item.converted_matrix_identity or ''}"
    return item.obligation_identity or item.item_id


def _check_from_item(item: UnresolvedReviewItem, head_sha: str | None) -> HumanCheck:
    return HumanCheck(
        reviewer=item.reviewer,
        text=item.text,
        source_round=item.source_round,
        head_sha=item.candidate_head_sha or head_sha,
        kind="converted" if item.converted_row_id is not None else "native",
        origin_item=item,
    )


@dataclass(frozen=True)
class Normalization:
    items: list[UnresolvedReviewItem]
    checks: tuple[HumanCheck, ...]
    dropped: int = 0
    added: int = 0


def normalize_human_evidence_ledger(
    items: Sequence[UnresolvedReviewItem],
    checks: Sequence[HumanCheck],
    *,
    mode: str,
    head_sha: str | None = None,
) -> Normalization:
    """The single choke point: no open native human evidence survives a non-freeze ledger."""
    if mode == "freeze":
        return Normalization(list(items), tuple(checks))
    kept: list[UnresolvedReviewItem] = []
    moved: list[UnresolvedReviewItem] = []
    for item in items:
        if is_native_human_evidence(item) and item.status in _OPEN_STATUSES:
            moved.append(item)
        else:
            kept.append(item)
    if mode == "off":
        return Normalization(kept, (), dropped=len(moved))
    result = list(checks)
    known = {check.identity for check in result}
    added = 0
    for item in moved:
        identity = check_identity(item)
        if identity in known:
            continue
        known.add(identity)
        result.append(_check_from_item(item, head_sha))
        added += 1
    return Normalization(kept, tuple(result), added=added)


def restore_checks_as_obligations(
    items: Sequence[UnresolvedReviewItem],
    checks: Sequence[HumanCheck],
    *,
    clearances: Sequence[tuple[str, str]] = (),
    head_sha: str | None = None,
) -> list[UnresolvedReviewItem]:
    """advisory -> freeze: put each check's original item back, deduplicated.

    The origin item is restored verbatim with its lifecycle reset to
    ``evidence_deferred`` (the reset a freeze release applies), so a converted
    row keeps its row identity, resolution owners and owner states.  A check
    whose identity was cleared at ``head_sha`` is not restored.
    """
    from .native_evidence import identities

    result = list(items)
    cleared = set(clearances)
    item_ids = {key for item in result for key in (item.item_id, *item.alias_item_ids)}
    obligation_ids = {
        identity for item in result for identity in identities(item) if identity
    }
    rows = {
        (item.converted_row_id, item.converted_matrix_identity)
        for item in result if item.converted_row_id is not None
    }
    for check in checks:
        origin = check.origin_item
        keys = {origin.item_id, *origin.alias_item_ids}
        if keys & item_ids or set(identities(origin)) & obligation_ids:
            continue
        if origin.converted_row_id is not None and (
            (origin.converted_row_id, origin.converted_matrix_identity) in rows
        ):
            continue
        if head_sha and any((identity, head_sha) in cleared for identity in identities(origin)):
            continue
        result.append(replace(origin, lifecycle="evidence_deferred", candidate_head_sha=None))
        item_ids |= keys
        obligation_ids |= {identity for identity in identities(origin) if identity}
        if origin.converted_row_id is not None:
            rows.add((origin.converted_row_id, origin.converted_matrix_identity))
    return result


def render_human_checks_comment(
    checks: Sequence[HumanCheck], *, pr_number: int, head_sha: str
) -> str:
    from .comment_rendering import _normalize_item_summary

    lines = [
        HUMAN_CHECKS_HEADING,
        "",
        f"Reviewers approved PR #{pr_number} at head `{head_sha}`. They asked for the "
        "human checks below. Under `--pr-human-evidence advisory` these are non-blocking: "
        "the tool does not wait for them and does not merge while they are listed, so a "
        "human decides whether to check and merge.",
        "",
    ]
    for check in checks[:HUMAN_CHECKS_RENDER_LIMIT]:
        text = _normalize_item_summary(check.text, limit=_TEXT_LIMIT)
        if check.kind == "converted":
            label = f"risk-matrix row `{check.origin_item.converted_row_id}`"
        else:
            label = f"{_normalize_item_summary(check.reviewer, limit=80)} (round {check.source_round})"
        lines.append(f"- {label}: {text}")
    if len(checks) > HUMAN_CHECKS_RENDER_LIMIT:
        lines.append(f"- ... and {len(checks) - HUMAN_CHECKS_RENDER_LIMIT} more; every check is kept "
                     "in this comment's round metadata.")
    lines.extend(["", "-- coding-review-agent-loop"])
    return "\n".join(lines)


def notice_line(*, pr_number: int, count: int, head_sha: str) -> str:
    return (
        f"PR #{pr_number} approved but not merged: {count} human check(s) requested; see the "
        f"'Human checks requested' comment on PR #{pr_number} for head {head_sha}."
    )


class HumanEvidenceRun:
    """Per-run state of the human evidence mode inside one PR loop."""

    def __init__(self, config: object, pr_number: int) -> None:
        self.config = config
        self.mode = mode_of(config)
        self.pr_number = pr_number
        self.checks: tuple[HumanCheck, ...] = ()
        self.restore_pending = False
        self.posted: set[tuple[str, tuple[str, ...]]] = set()

    @property
    def freeze(self) -> bool:
        return self.mode == "freeze"

    @property
    def advisory(self) -> bool:
        return self.mode == "advisory"

    def stamp(self) -> tuple[dict, ...]:
        """The list stamped on orchestrator PR records (advisory only)."""
        return tuple(check.as_dict() for check in self.checks) if self.advisory else ()

    def seed(self, pr_records: Sequence[object], check_records: Sequence[object]) -> None:
        """Rebuild the list from the latest orchestrator record that carries one.

        The latest orchestrator PR record or human-checks record is
        authoritative (an empty list on it is authoritative too).  Requests
        recorded only on later reviewer records come back through the normal
        reconstruction plus normalization.
        """
        for record in check_records:
            metadata = record.metadata
            decoded = [HumanCheck.from_mapping(entry) for entry in metadata.human_checks]
            self.posted.add((metadata.subject, tuple(sorted(c.identity for c in decoded if c))))
        candidates = [
            record for record in pr_records if getattr(record.metadata, "agent", None) == "Orchestrator"
        ] + list(check_records)
        if not candidates:
            return
        latest = max(candidates, key=lambda record: record.index)
        checks = tuple(
            check for check in map(HumanCheck.from_mapping, latest.metadata.human_checks) if check
        )
        if self.freeze:
            self.checks, self.restore_pending = checks, bool(checks)
        elif self.advisory:
            self.checks = checks
        elif checks:
            log(self.config, f"Dropped {len(checks)} recorded human check(s) (--pr-human-evidence off)")

    def normalize(
        self,
        items: Sequence[UnresolvedReviewItem],
        *,
        round_number: int,
        head_sha: str | None = None,
        clearances: Sequence[tuple[str, str]] = (),
    ) -> list[UnresolvedReviewItem]:
        if self.freeze:
            if not self.restore_pending:
                return list(items)
            restored = restore_checks_as_obligations(
                items, self.checks, clearances=clearances, head_sha=head_sha
            )
            log(
                self.config,
                f"Round {round_number}: restored {len(restored) - len(items)} advisory human "
                "check(s) as exact-head evidence obligations (--pr-human-evidence freeze)",
            )
            self.checks, self.restore_pending = (), False
            return restored
        result = normalize_human_evidence_ledger(
            items, self.checks, mode=self.mode, head_sha=head_sha
        )
        self.checks = result.checks
        if result.dropped:
            log(
                self.config,
                f"Round {round_number}: dropped {result.dropped} exact-head evidence "
                "request(s) (--pr-human-evidence off)",
            )
        if result.added:
            log(self.config, f"Round {round_number}: {result.added} human check(s) recorded as advisory")
        return result.items

    def gate_needed(self, items: Sequence[UnresolvedReviewItem]) -> bool:
        """The finalization gate trigger, plus a pending advisory list (#2063)."""
        from .pr_loop_support import _evidence_gate_needed

        return _evidence_gate_needed(items) or (self.advisory and bool(self.checks))

    def releases_freeze(self, items: Sequence[UnresolvedReviewItem]) -> bool:
        """A non-freeze resume releases a freeze unless screenshot items stay frozen."""
        from .unresolved_items import _frozen_evidence_obligations

        return not self.freeze and not _frozen_evidence_obligations(items)

    def hold(self, runner: object, *, head_sha: str | None, round_number: int, merging: bool) -> bool:
        """At approval: post the list once per head; True skips the merge.

        Runs only after the finalization gate and ``_ensure_finalization_ready``.
        With an empty list (or outside advisory) it does nothing.
        """
        if not self.advisory or not self.checks or not head_sha:
            return False
        key = (head_sha, tuple(sorted(check.identity for check in self.checks)))
        if key not in self.posted:
            from .github import post_pr_comment
            from .round_metadata import PostedRoundMetadata, _attach_round_metadata

            post_pr_comment(
                runner,
                config=self.config,
                pr_number=self.pr_number,
                body=_attach_round_metadata(
                    render_human_checks_comment(self.checks, pr_number=self.pr_number, head_sha=head_sha),
                    PostedRoundMetadata(
                        flow=HUMAN_CHECKS_FLOW, role="summary", agent="Orchestrator",
                        round_number=round_number, subject=head_sha, state="approved",
                        human_checks=self.stamp(),
                    ),
                ),
            )
            self.posted.add(key)
        if not merging:
            return False
        print(notice_line(pr_number=self.pr_number, count=len(self.checks), head_sha=head_sha))
        return True
