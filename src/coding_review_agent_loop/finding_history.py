"""Bounded finding history and standing generalization guidance (#1273).

The step-back trigger (#1251) is reactive and external.  This module is its
proactive complement: it gives the planner and the coder a compact list of the
findings and fixes of earlier rounds of *this run* (resolved ones included, each
with its ``path:line`` reference) plus standing guidance to compare the newest
finding against them and fix a whole class at its root.

Everything here is advisory prompt context.  Status is never inferred from
historical votes: open/deferred/resolved is read from the loop's authoritative
post-reconciliation ledger (``observe_reconciled``), and, on resume, from the
read-only replay in ``round_state.canonical_history_item_outcomes``.  A failure
inside the history never stops a run; it degrades to an "unavailable" notice
while the guidance stays in the prompt.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .protocol_markers import sanitize_untrusted_prose
from .review_step_back import effective_new_items, project_finding_locations

if TYPE_CHECKING:
    from .round_state import PostedRoundRecord

FINDING_HISTORY_MAX_ROUNDS = 6
FINDING_HISTORY_MAX_CHARS = 6000
FINDING_HISTORY_LINE_LIMIT = 240
FINDING_HISTORY_CI_DETAIL_LINES = 3
FINDING_HISTORY_MAX_FIX_ITEMS = 8
_CI_DETAIL_LIMIT = 120
_MAX_LOCATIONS = 4

PHASE_PLAN = "plan"
PHASE_PR = "pr"

STATUS_OPEN = "open"
STATUS_RESOLVED = "resolved"
STATUS_DEFERRED = "deferred"
STATUS_SUPERSEDED = "superseded"

_MANDATORY = frozenset({"blocking", "same-pr", "same-plan"})
_CI_KINDS = frozenset({"managed-exact-head-ci", "github-pr-checks"})

VIEW_POPULATED = "populated"
VIEW_EMPTY = "empty"
VIEW_UNAVAILABLE = "unavailable"

LogFn = Callable[[str], None]


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 1)].rstrip() + "…"


def _neutral(text: str, limit: int) -> str:
    return _clip(sanitize_untrusted_prose(str(text or "")), limit)


@dataclass(frozen=True)
class HistoryFinding:
    key: tuple
    kind: str  # "reviewer" | "machine"
    source_round: int
    reviewer: str
    item_id: str
    summary_line: str
    locations: tuple[str, ...] = ()
    ci_details: tuple[str, ...] = ()
    identity: str = ""
    failed_head: str | None = None


@dataclass(frozen=True)
class HistoryFix:
    published_round: int
    agent: str
    summary_excerpt: str
    addressed: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class FindingHistoryView:
    """What a prompt renders: one of three distinct states."""

    state: str
    body: str = ""
    reason: str = ""


def _first_line(text: str) -> str:
    for line in str(text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def _location_labels(item: object) -> tuple[str, ...]:
    labels: list[str] = []
    for location in project_finding_locations(item):
        label = (
            f"{location.path}:{location.start}"
            if location.start == location.end
            else f"{location.path}:{location.start}-{location.end}"
        )
        label = _neutral(label, FINDING_HISTORY_LINE_LIMIT)
        if label not in labels:
            labels.append(label)
    return tuple(labels[:_MAX_LOCATIONS])


def _ci_detail_lines(text: str) -> tuple[str, ...]:
    """The first ``- `` detail lines after the generic CI headline."""
    details: list[str] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("- ") or stripped.lower().startswith("- reviewed head"):
            continue
        details.append(_neutral(stripped, _CI_DETAIL_LIMIT))
        if len(details) >= FINDING_HISTORY_CI_DETAIL_LINES:
            break
    return tuple(details)


def project_finding(item: object) -> HistoryFinding | None:
    """Project a ledger item; ``None`` for machine kinds other than CI."""
    item_id = str(getattr(item, "item_id", ""))
    source_round = int(getattr(item, "source_round", 0) or 0)
    kind = getattr(item, "obligation_kind", None)
    if getattr(item, "is_machine_obligation", False):
        if kind not in _CI_KINDS:
            return None
        identity = str(getattr(item, "obligation_identity", None) or kind)
        head = getattr(item, "failed_head_sha", None)
        text = str(getattr(item, "text", "") or "")
        if head:
            key: tuple = ("machine", identity, head)
        else:
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
            key = ("machine", identity, None, source_round, digest)
        label = f"CI {kind} on {head[:12]}" if head else f"CI {kind}"
        return HistoryFinding(
            key=key,
            kind="machine",
            source_round=source_round,
            reviewer="orchestrator",
            item_id=item_id,
            summary_line=label,
            ci_details=_ci_detail_lines(text),
            identity=identity,
            failed_head=head or None,
        )
    reviewer = str(getattr(item, "reviewer", ""))
    return HistoryFinding(
        key=("reviewer", reviewer, item_id),
        kind="reviewer",
        source_round=source_round,
        reviewer=reviewer,
        item_id=item_id,
        summary_line=_neutral(
            _first_line(str(getattr(item, "text", "") or "")), FINDING_HISTORY_LINE_LIMIT
        ),
        locations=_location_labels(item),
    )


def _fix_from_parts(
    *,
    published_round: int,
    agent: str,
    summary: object,
    addressed: Iterable[tuple[str, str]],
) -> HistoryFix | None:
    summary_text = summary if isinstance(summary, str) else ""
    pairs = tuple(
        (_neutral(item_id, 60), _neutral(note, FINDING_HISTORY_LINE_LIMIT))
        for item_id, note in list(addressed)[:FINDING_HISTORY_MAX_FIX_ITEMS]
    )
    if not summary_text.strip() and not pairs:
        return None
    return HistoryFix(
        published_round=published_round,
        agent=_neutral(agent, 80),
        summary_excerpt=_neutral(summary_text, FINDING_HISTORY_LINE_LIMIT),
        addressed=pairs,
    )


def _addressed_from_mapping(payload: Mapping) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    notes = payload.get("addressed_item_notes")
    notes = notes if isinstance(notes, Mapping) else {}
    addressed = payload.get("addressed_items")
    if isinstance(addressed, list):
        for item_id in addressed:
            pairs.append((str(item_id), str(notes.get(item_id, "") or "")))
    for entry in payload.get("prior_plan_item_dispositions") or ():
        if isinstance(entry, Mapping) and entry.get("disposition") == "resolved":
            pairs.append((str(entry.get("item_id", "")), str(entry.get("note") or "")))
    return pairs


def fix_from_payload(
    payload: object, *, published_round: int, agent: str
) -> HistoryFix | None:
    if not isinstance(payload, Mapping):
        raise ValueError("structured response is not an object")
    return _fix_from_parts(
        published_round=published_round,
        agent=agent,
        summary=payload.get("summary"),
        addressed=_addressed_from_mapping(payload),
    )


def fix_from_parsed(parsed: object, *, published_round: int, agent: str) -> HistoryFix | None:
    """Project a validated ``StructuredCoderFollowup`` / planner revision object."""
    addressed: list[tuple[str, str]] = []
    notes = getattr(parsed, "addressed_item_notes", None) or {}
    for item_id in getattr(parsed, "addressed_items", None) or ():
        addressed.append((str(item_id), str(notes.get(item_id, "") or "")))
    for entry in getattr(parsed, "prior_plan_item_dispositions", None) or ():
        if getattr(entry, "disposition", None) == "resolved":
            addressed.append((str(entry.item_id), str(entry.note or "")))
    return _fix_from_parts(
        published_round=published_round,
        agent=agent,
        summary=getattr(parsed, "summary", None),
        addressed=addressed,
    )


_GENERALIZATION_RE = re.compile(
    r"(?im)(?:^|(?<=[.!?]\s))[\s\-*>]*(generalization:[^\n]*)|(generalizes the fix for[^\n]*)"
)


def detect_generalization(text: str | None) -> str | None:
    """A <=240-char excerpt of a declared generalization, or ``None``."""
    if not text:
        return None
    match = _GENERALIZATION_RE.search(str(text))
    if match is None:
        return None
    return _clip(match.group(1) or match.group(2), FINDING_HISTORY_LINE_LIMIT)


def generalization_texts(parsed: object) -> list[str]:
    """Texts a response may carry a declaration in (summary and item notes)."""
    texts: list[str] = []
    summary = getattr(parsed, "summary", None)
    if isinstance(summary, str):
        texts.append(summary)
    notes = getattr(parsed, "addressed_item_notes", None)
    if isinstance(notes, Mapping):
        texts.extend(str(note) for note in notes.values())
    for entry in getattr(parsed, "prior_plan_item_dispositions", None) or ():
        note = getattr(entry, "note", None)
        if note:
            texts.append(str(note))
    return texts


def log_declared_generalization(
    parsed: object,
    *,
    log: LogFn,
    round_number: int,
    agent: str,
    step_back_directed: bool,
) -> str | None:
    """Log one line when the response declares a generalization (logging only)."""
    for text in generalization_texts(parsed):
        excerpt = detect_generalization(text)
        if excerpt is not None:
            tag = "step-back-directed" if step_back_directed else "proactive"
            log(f"Round {round_number}: {agent} declared a generalization ({tag}): {excerpt}")
            return excerpt
    return None


def render_generalization_guidance(phase: str) -> str:
    """Standing guidance; independent of whether any history is available."""
    artifact = "plan" if phase == PHASE_PLAN else "patch"
    coverage = (
        "make the design and its test-matrix row cover the whole class"
        if phase == PHASE_PLAN
        else "cover every branch of the class with one parametrized test"
    )
    return (
        "Proactive generalization (orchestrator guidance, not a reviewer finding):\n"
        "Before patching a finding, compare it with the earlier-round findings listed "
        "below (and with the findings you can see in this prompt). If it is another "
        f"instance of a class you have already fixed, fix the whole class at its root; "
        f"{coverage}, not just the newest instance. State the generalization explicitly "
        "in your `summary` or the item note as `Generalization: this generalizes the fix "
        "for [<earlier-item-id>]: <the rule>` so reviewers can check it. Stay inside this "
        f"issue's scope: if the class needs a broader refactor than the {artifact} "
        "should carry, record it as a future follow-up instead of including it.\n"
    )


def render_finding_history_block(view: FindingHistoryView | None) -> str:
    if view is None:
        return ""
    if view.state == VIEW_UNAVAILABLE:
        return (
            f"Earlier-round history is unavailable this turn ({view.reason or 'unknown'}); "
            "the guidance above still applies to findings you can see in this prompt.\n"
        )
    if view.state == VIEW_EMPTY:
        return "(no earlier-round findings or fixes in this run yet)\n"
    return view.body


@dataclass
class FindingHistoryLedger:
    """In-memory per-run accumulator; every method is advisory and never raises."""

    phase: str
    log: LogFn | None = None
    _findings: dict[tuple, HistoryFinding] = field(default_factory=dict)
    _fixes: dict[tuple[int, str], HistoryFix] = field(default_factory=dict)
    _live_open: dict[tuple, int] = field(default_factory=dict)
    _live_machine: dict[str, str | None] = field(default_factory=dict)
    _deferred: set[tuple] = field(default_factory=set)
    _ledger_future: set[tuple] = field(default_factory=set)
    unavailable_reason: str | None = None

    def _note(self, message: str) -> None:
        if self.log is not None:
            self.log(message)

    def _fail(self, exc: Exception) -> None:
        self.unavailable_reason = _clip(f"{type(exc).__name__}: {exc}", 120)
        self._note(f"finding history unavailable: {self.unavailable_reason}")

    def add_finding(self, item: object) -> HistoryFinding | None:
        finding = project_finding(item)
        if finding is None:
            return None
        return self._findings.setdefault(finding.key, finding)

    def add_fix(self, fix: HistoryFix | None) -> None:
        if fix is not None:
            self._fixes.setdefault((fix.published_round, fix.agent), fix)

    def observe_reconciled(
        self, unresolved_items: Sequence[object], future_items: Sequence[object] = ()
    ) -> None:
        """Record the FULL post-reconciliation ledger (never a filtered subset)."""
        try:
            live_open: dict[tuple, int] = {}
            live_machine: dict[str, str | None] = {}
            seen_keys: set[tuple] = set()
            ledger_future: set[tuple] = set()
            for item in (*unresolved_items, *future_items):
                finding = self.add_finding(item)
                if finding is None:
                    continue
                seen_keys.add(finding.key)
                status = getattr(item, "status", None)
                if finding.kind == "machine":
                    if getattr(item, "lifecycle", None) != "cleared" and status != "future":
                        live_machine[finding.identity] = finding.failed_head
                    continue
                if status in _MANDATORY:
                    open_subs = sum(
                        1
                        for sub in getattr(item, "sub_items", None) or ()
                        if getattr(sub, "status", "open") != "resolved"
                    )
                    live_open[finding.key] = open_subs
                    self._deferred.discard(finding.key)
                elif status == "future":
                    self._deferred.add(finding.key)
                    if any(item is carried for carried in unresolved_items):
                        ledger_future.add(finding.key)
            # A future item that stays in the canonical ledger (full context mode,
            # planner) and then leaves it was cleared by the reconciler: explicit
            # clearance overrides the deferral.  An item that left the ledger by
            # moving to the separate future collection (compact PR mode) is not
            # in `_ledger_future`, so mere absence keeps it deferred.
            for key in self._ledger_future - seen_keys:
                self._deferred.discard(key)
            self._ledger_future = ledger_future
            self._live_open = live_open
            self._live_machine = live_machine
        except Exception as exc:  # advisory: never stop the run
            self._fail(exc)

    def note_carried_ledger(self, prior_items: Sequence[object]) -> None:
        """Record the authoritative ledger a round starts from, before reconciliation.

        On a resumed run the first observation happens only after the round's
        reconciliation, so the future items already carried in the stored ledger
        must be known beforehand: if the reconciler then clears one, its absence
        is explicit clearance rather than the compact-mode move to the separate
        future collection.
        """
        try:
            for item in prior_items:
                finding = self.add_finding(item)
                if finding is None or finding.kind != "reviewer":
                    continue
                if getattr(item, "status", None) == "future":
                    self._deferred.add(finding.key)
                    self._ledger_future.add(finding.key)
        except Exception as exc:
            self._fail(exc)

    def record_fix(self, parsed: object, *, published_round: int, agent: str) -> None:
        try:
            self.add_fix(fix_from_parsed(parsed, published_round=published_round, agent=agent))
        except Exception as exc:
            self._note(f"finding history: skipped a fix entry ({type(exc).__name__}: {exc})")

    def seed(
        self,
        records: Sequence["PostedRoundRecord"],
        *,
        outcomes: Mapping[str, str] | None = None,
    ) -> None:
        """Seed once from round records the loop already extracted."""
        try:
            for record in records:
                metadata = record.metadata
                items = effective_new_items(records, record) or ()
                for item in items:
                    self.add_finding(item)
                for item in metadata.prior_items:
                    if getattr(item, "is_machine_obligation", False):
                        self.add_finding(item)
                if metadata.role == "coder" and metadata.raw_structured_coder_response:
                    try:
                        payload, _end = json.JSONDecoder().raw_decode(
                            metadata.raw_structured_coder_response.lstrip()
                        )
                        self.add_fix(
                            fix_from_payload(
                                payload,
                                published_round=metadata.round_number,
                                agent=metadata.agent,
                            )
                        )
                    except Exception as exc:  # one bad payload skips one fix
                        self._note(
                            "finding history: skipped an undecodable fix payload on round "
                            f"{metadata.round_number} ({type(exc).__name__}: {exc})"
                        )
            for key, finding in self._findings.items():
                if finding.kind == "reviewer" and (outcomes or {}).get(finding.item_id) == "deferred":
                    self._deferred.add(key)
        except Exception as exc:
            self._fail(exc)

    def seed_from_records(
        self,
        records: Sequence["PostedRoundRecord"],
        *,
        reconciliation_mode: str,
        same_status: str,
    ) -> None:
        """Seed from already-extracted records, replaying deferral outcomes."""
        from .round_state import canonical_history_item_outcomes

        try:
            outcomes = canonical_history_item_outcomes(
                records, reconciliation_mode=reconciliation_mode, same_status=same_status
            )
        except Exception as exc:
            self._fail(exc)
            return
        self.seed(records, outcomes=outcomes)

    def _status(self, finding: HistoryFinding) -> str:
        if finding.kind == "machine":
            live_head = self._live_machine.get(finding.identity, "")
            if finding.identity not in self._live_machine:
                return STATUS_RESOLVED
            if finding.failed_head is None or live_head is None or live_head == finding.failed_head:
                return STATUS_OPEN
            return f"{STATUS_SUPERSEDED} by the failure on {str(live_head)[:12]}"
        if finding.key in self._live_open:
            subs = self._live_open[finding.key]
            return f"{STATUS_OPEN}, {subs} open sub-item(s)" if subs else STATUS_OPEN
        if finding.key in self._deferred:
            return STATUS_DEFERRED
        return STATUS_RESOLVED

    def view(self, round_number: int) -> FindingHistoryView:
        """The bounded history for the prompt of review round ``round_number``.

        Findings first raised in ``round_number`` are in the review payload and
        are excluded; fixes already published up to it are included.
        """
        if self.unavailable_reason is not None:
            return FindingHistoryView(VIEW_UNAVAILABLE, reason=self.unavailable_reason)
        try:
            return self._build_view(round_number)
        except Exception as exc:
            self._fail(exc)
            return FindingHistoryView(VIEW_UNAVAILABLE, reason=self.unavailable_reason or "")

    def _build_view(self, round_number: int) -> FindingHistoryView:
        groups: dict[int, list[str]] = {}
        for finding in self._findings.values():
            if finding.source_round >= round_number:
                continue
            groups.setdefault(finding.source_round, []).append(self._finding_entry(finding))
        for fix in self._fixes.values():
            if fix.published_round > round_number:
                continue
            groups.setdefault(fix.published_round, []).append(_fix_entry(fix))
        if not groups:
            return FindingHistoryView(VIEW_EMPTY)
        rounds = sorted(groups)
        total_rounds = len(rounds)
        kept = rounds[-FINDING_HISTORY_MAX_ROUNDS:]
        # Whole oldest rounds are dropped until the block fits; an oversized
        # newest round is omitted too rather than cut mid-entry.
        while kept:
            text = _render_groups(kept, groups, total_rounds - len(kept))
            if len(text) <= FINDING_HISTORY_MAX_CHARS:
                break
            kept = kept[1:]
        else:
            text = _render_groups([], groups, total_rounds)
        return FindingHistoryView(VIEW_POPULATED, body=text)

    def _finding_entry(self, finding: HistoryFinding) -> str:
        status = self._status(finding)
        if finding.kind == "machine":
            lines = [f"- {_neutral(finding.item_id, 60)}: {finding.summary_line} ({status})"]
            lines.extend(f"    {detail}" for detail in finding.ci_details)
            return "\n".join(lines)
        where = ", ".join(finding.locations) if finding.locations else "(no location)"
        # Item IDs are deliberately not bracketed: `[item-N]` marks the
        # actionable ledger of the current dispatch and must stay unambiguous.
        return (
            f"- {_neutral(finding.reviewer, 80)} finding {_neutral(finding.item_id, 60)} "
            f"({status}) {where}: "
            f"{finding.summary_line}"
        )


def _fix_entry(fix: HistoryFix) -> str:
    lines = [f"- fix by {fix.agent}: {fix.summary_excerpt or '(no summary)'}"]
    lines.extend(f"    {item_id}: {note or '(no note)'}" for item_id, note in fix.addressed)
    return "\n".join(lines)


def _render_groups(
    rounds: Sequence[int],
    groups: Mapping[int, Sequence[str]],
    omitted_rounds: int,
) -> str:
    out = [
        "Earlier-round history for this run (orchestrator-collected context, not reviewer "
        "findings; most recent rounds only; status is read from the live ledger):"
    ]
    if omitted_rounds:
        out.append(f"({omitted_rounds} earlier round(s) omitted)")
    for number in rounds:
        out.append(f"Round {number}:")
        out.extend(groups[number])
    return "\n".join(out) + "\n"
