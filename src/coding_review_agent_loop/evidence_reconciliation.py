"""Deterministic reconciliation for final discuss evidence ledgers (#535)."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Sequence

from .protocol_discuss_types import (
    ParsedDiscussResponse,
)

MAX_RECONCILIATION_CANDIDATES = 64
MAX_RECONCILIATION_BYTES = 24_000
MAX_RENDERED_EVIDENCE_ENTRIES = 50
MAX_RENDERED_EVIDENCE_BYTES = 16_000
MAX_FACT_BYTES = 512
MAX_SOURCE_BYTES = 256
MAX_REASON_BYTES = 512


def _clip(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    raw = value.encode("utf-8")
    if len(raw) <= limit:
        return value
    return raw[: max(0, limit - 3)].decode("utf-8", "ignore") + "..."


def _normalize(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).casefold()


@dataclass(frozen=True)
class EvidenceObservation:
    observation_id: str
    round_number: int
    reviewer: str
    fact: str
    status: str
    source: str | None = None
    verification_basis: str | None = None


def observation_id(subject: str, round_number: int, reviewer: str, index: int) -> str:
    return f"{subject}-r{round_number}-{reviewer}-c{index}"


def collect_evidence_observations(
    subject: str, round_history: Sequence[Sequence[ParsedDiscussResponse]],
) -> tuple[list[EvidenceObservation], list[dict[str, object]]]:
    """Collect claims and ordered updates while preserving legacy citations."""
    observations: list[EvidenceObservation] = []
    updates: list[dict[str, object]] = []
    for round_number, votes in enumerate(round_history, start=1):
        for vote in votes:
            if not hasattr(vote, "evidence_claims"):
                continue
            claims = list(vote.evidence_claims)
            # Legacy citations are intentionally never promoted to verified.
            claims.extend(type("Legacy", (), {
                "fact": fact.fact, "status": "reported-but-unverified", "source": fact.source,
                "verification_basis": None,
            })() for fact in vote.sourced_facts)
            for index, claim in enumerate(claims):
                observations.append(EvidenceObservation(
                    observation_id=observation_id(subject, round_number, vote.reviewer, index),
                    round_number=round_number, reviewer=vote.reviewer, fact=claim.fact,
                    status=claim.status, source=getattr(claim, "source", None),
                    verification_basis=getattr(claim, "verification_basis", None),
                ))
            for update in vote.evidence_updates:
                updates.append({
                    "round_number": round_number, "reviewer": vote.reviewer, "action": update.action,
                    "target_observation_id": update.target_observation_id, "reason": update.reason,
                    "replacement_claim_index": update.replacement_claim_index,
                })
    return observations, updates


def bounded_reconciliation_candidates(observations: Sequence[EvidenceObservation], updates: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Newest-first bounded input; relation targets are included atomically."""
    by_id = {item.observation_id: item for item in observations}
    selected: list[EvidenceObservation] = []
    selected_ids: set[str] = set()
    # Updates/targets first, then all claims newest-first.  Invalid old targets
    # are audit-only and never suppress a current item.
    priority_ids = [str(item["target_observation_id"]) for item in reversed(updates)]
    priority_ids += [item.observation_id for item in reversed(observations)]
    for candidate_id in priority_ids:
        item = by_id.get(candidate_id)
        if item is None or candidate_id in selected_ids:
            continue
        candidate = {
            "id": item.observation_id, "fact": _clip(item.fact, MAX_FACT_BYTES),
            "status": item.status, "source": _clip(item.source, MAX_SOURCE_BYTES),
            "contributors": [item.reviewer],
        }
        trial = selected + [item]
        payload = [{"id": x.observation_id, "fact": _clip(x.fact, MAX_FACT_BYTES), "status": x.status,
                    "source": _clip(x.source, MAX_SOURCE_BYTES)} for x in trial]
        if len(trial) > MAX_RECONCILIATION_CANDIDATES or len(json.dumps(payload, ensure_ascii=False).encode()) > MAX_RECONCILIATION_BYTES:
            continue
        selected.append(item)
        selected_ids.add(candidate_id)
    return [{"id": item.observation_id, "fact": _clip(item.fact, MAX_FACT_BYTES), "status": item.status,
             "source": _clip(item.source, MAX_SOURCE_BYTES), "contributors": [item.reviewer]} for item in selected]


def reconcile_evidence(
    subject: str, round_history: Sequence[Sequence[ParsedDiscussResponse]],
    semantic_groups: Sequence[Sequence[str]] = (),
) -> dict[str, object]:
    """Apply explicit updates and exact-match grouping without analyzer inference."""
    observations, updates = collect_evidence_observations(subject, round_history)
    by_id = {item.observation_id: item for item in observations}
    inactive: dict[str, dict[str, object]] = {}
    warnings: list[str] = []
    for update in updates:
        target = str(update["target_observation_id"])
        if target not in by_id:
            warnings.append(f"dangling update target: {target}")
            continue
        inactive[target] = update
    # Keep semantic-group membership separate from the canonical ID mapping.
    # An identity alias cannot tell a group leader from an observation that was
    # never grouped, so using it as the condition would leave the leader in an
    # exact-match bucket while its peers use the semantic bucket.
    semantic_aliases: dict[str, str] = {}
    for group in semantic_groups:
        if group:
            for item in group:
                semantic_aliases[item] = group[0]
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    history: list[dict[str, object]] = []
    for item in observations:
        if item.observation_id in inactive:
            update = inactive[item.observation_id]
            history.append({"id": item.observation_id, "fact": _clip(item.fact, MAX_FACT_BYTES),
                            "action": update["action"], "reason": _clip(str(update["reason"]), MAX_REASON_BYTES),
                            "target_round": item.round_number, "reviewer": item.reviewer,
                            "replacement_claim_index": update["replacement_claim_index"]})
            continue
        # Analyzer grouping may merge paraphrases, but only after strict ID and
        # status validation in the protocol parser.
        key = (("semantic", semantic_aliases[item.observation_id], item.status)
               if item.observation_id in semantic_aliases
               else (_normalize(item.fact), _normalize(item.source), item.status))
        entry = grouped.setdefault(key, {"ids": [], "fact": _clip(item.fact, MAX_FACT_BYTES), "status": item.status,
                                         "sources": [], "contributors": [], "rounds": []})
        entry["ids"].append(item.observation_id)
        if item.source and item.source not in entry["sources"]:
            entry["sources"].append(_clip(item.source, MAX_SOURCE_BYTES))
        if item.reviewer not in entry["contributors"]:
            entry["contributors"].append(item.reviewer)
        if item.round_number not in entry["rounds"]:
            entry["rounds"].append(item.round_number)
    entries = list(grouped.values())
    # Current evidence gets stable status order; later observations win ties.
    order = {"verified": 0, "reported-but-unverified": 1, "missing": 2}
    entries.sort(key=lambda entry: (order[str(entry["status"])], -max(entry["rounds"]), str(entry["fact"])))
    history.sort(key=lambda item: (-int(item["target_round"]), str(item["id"])))
    candidates = bounded_reconciliation_candidates(observations, updates)
    material = {"entries": entries, "history": history, "warnings": warnings}
    digest = hashlib.sha256(json.dumps(material, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    rendered: list[dict[str, object]] = []
    used = 0
    for item in entries + history:
        encoded = len(json.dumps(item, ensure_ascii=False).encode())
        if len(rendered) >= MAX_RENDERED_EVIDENCE_ENTRIES or used + encoded > MAX_RENDERED_EVIDENCE_BYTES:
            continue
        rendered.append(item)
        used += encoded
    return {
        "entries": entries, "history": history, "rendered": rendered,
        "omitted_entries": len(entries) + len(history) - len(rendered),
        "candidate_omitted": len(observations) - len(candidates), "digest": digest,
        "warnings": warnings, "candidate_observations": candidates,
        "observation_count": len(observations), "update_count": len(updates),
    }
