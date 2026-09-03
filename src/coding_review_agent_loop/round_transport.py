"""Bounded, dependency-leaf transport for durable round comments."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import zlib
from collections.abc import Mapping, Sequence

from .errors import AgentLoopError
from .protocol_markers import TrustedBody, scan_reserved_markers

MAX_GITHUB_BODY_CHARS = 60_000
ROUND_RESUME_MARKER_RE = re.compile(
    r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_:-]+)\s*-->", re.I
)
ROUND_TRANSPORT_SIDECAR_RE = re.compile(
    r"<!--\s*AGENT_LOOP_SIDECAR:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", re.I
)
# Spill reviewer checkpoints first: they are often the largest metadata field
# and are required to safely resume a provisional parallel-review round.
_SPILL_FIELDS = (
    "canonical_reviewer_response",
    "raw_structured_coder_response",
    "canonical_plan",
    "round_synthesis",
    "final_synthesis",
    "analyzer_response",
    "final_analyzer_response",
    "raw_synthesis_response",
)
_MAX_COMPRESSED = 8_000_000
_MAX_DECOMPRESSED = 16_000_000
_PART_CHARS = 40_000


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value.encode("ascii"))


def _decompress_bounded(packed: bytes) -> bytes:
    if len(packed) > _MAX_COMPRESSED:
        raise ValueError("compressed payload too large")
    decompressor = zlib.decompressobj()
    raw = decompressor.decompress(packed, _MAX_DECOMPRESSED + 1)
    if (
        len(raw) > _MAX_DECOMPRESSED
        or decompressor.unconsumed_tail
        or decompressor.unused_data
    ):
        raise ValueError("decompressed payload too large")
    raw += decompressor.flush()
    if len(raw) > _MAX_DECOMPRESSED:
        raise ValueError("decompressed payload too large")
    return raw


def encode_mapping(payload: Mapping[str, object]) -> str:
    raw = json.dumps(
        dict(payload), separators=(",", ":"), sort_keys=True, ensure_ascii=False
    ).encode()
    compressed = zlib.compress(raw, 9)
    if len(compressed) > _MAX_COMPRESSED:
        raise AgentLoopError("Round metadata is too large to transport safely.")
    return "v1_" + _b64(compressed)


def decode_mapping(encoded: str) -> dict[str, object]:
    try:
        raw = (
            _decompress_bounded(_unb64(encoded[3:]))
            if encoded.startswith("v1_")
            else _unb64(encoded)
        )
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("mapping required")
        return value
    except Exception as exc:
        raise AgentLoopError("Invalid AGENT_LOOP_META payload.") from exc


def is_round_transport_sidecar(body: str) -> bool:
    return bool(ROUND_TRANSPORT_SIDECAR_RE.search(body))


def _sidecar(payload: Mapping[str, object]) -> TrustedBody:
    encoded = _b64(
        json.dumps(dict(payload), separators=(",", ":"), sort_keys=True).encode()
    )
    return TrustedBody.canonical(
        f"<!-- AGENT_LOOP_SIDECAR: {encoded} -->",
        expected_tokens=("AGENT_LOOP_SIDECAR",),
    )


def prepare_round_comment(body: str | TrustedBody) -> tuple[TrustedBody, ...]:
    """Return sidecars followed by an anchor; non-round bodies are strictly bounded."""
    if isinstance(body, TrustedBody):
        carrier = body
    else:
        occurrences = scan_reserved_markers(body)
        carrier = (
            TrustedBody.current_untrusted_visible(body)
            if not occurrences
            else TrustedBody.canonical(
                body,
                expected_tokens=tuple(item.definition.token for item in occurrences),
            )
        )
    body_text = str(carrier)
    matches = list(ROUND_RESUME_MARKER_RE.finditer(body_text))
    if len(body_text) > MAX_GITHUB_BODY_CHARS and not matches:
        raise AgentLoopError(
            f"GitHub comment body exceeds {MAX_GITHUB_BODY_CHARS} characters; shorten the response."
        )
    if not matches:
        return (carrier,)

    # Resume reads the last marker when a legacy comment contains more than one.
    match = matches[-1]
    payload = decode_mapping(match.group("payload"))
    sidecars: list[str] = []
    anchor_id = hashlib.sha256(body_text.encode()).hexdigest()[:24]

    def render_anchor(mapping: Mapping[str, object]) -> str:
        return body_text[: match.start("payload")] + encode_mapping(mapping) + body_text[match.end("payload") :]

    for field in _SPILL_FIELDS:
        current_anchor = render_anchor(payload)
        if len(current_anchor) <= MAX_GITHUB_BODY_CHARS:
            break
        value = payload.get(field)
        if not isinstance(value, str):
            continue
        packed = zlib.compress(value.encode(), 9)
        if len(packed) > _MAX_COMPRESSED:
            raise AgentLoopError(f"Round metadata field {field} is too large to spill safely.")
        raw_digest = hashlib.sha256(value.encode()).hexdigest()
        packed_digest = hashlib.sha256(packed).hexdigest()
        encoded_packed = _b64(packed)
        chunks = [
            encoded_packed[index : index + _PART_CHARS]
            for index in range(0, len(encoded_packed), _PART_CHARS)
        ]
        reference = {
            "$round_transport_spill": anchor_id,
            "field": field,
            "parts": len(chunks),
            "sha256": raw_digest,
            "spill": packed_digest,
        }
        trial = dict(payload)
        trial[field] = reference
        if len(render_anchor(trial)) >= len(current_anchor):
            continue
        payload[field] = reference
        for index, chunk in enumerate(chunks):
            sidecars.append(
                _sidecar(
                    {
                        "v": 1,
                        "anchor": anchor_id,
                        "spill": packed_digest,
                        "field": field,
                        "index": index,
                        "count": len(chunks),
                        "sha256": raw_digest,
                        "data": chunk,
                    }
                )
            )

    anchor = render_anchor(payload)
    if len(anchor) > MAX_GITHUB_BODY_CHARS:
        raise AgentLoopError(
            f"Round comment exceeds {MAX_GITHUB_BODY_CHARS} characters even after metadata spill; "
            "shorten the visible response or metadata."
        )
    if any(len(item) > MAX_GITHUB_BODY_CHARS for item in sidecars):
        raise AgentLoopError("Round metadata sidecar exceeds GitHub body budget.")
    expected = tuple(item.definition.token for item in scan_reserved_markers(anchor))
    return (*sidecars, TrustedBody.canonical(anchor, expected_tokens=expected))


def hydrate_mapping(
    payload: Mapping[str, object], bodies: Sequence[str]
) -> tuple[dict[str, object], set[str]]:
    """Hydrate references from an unordered whole comment list; report missing fields."""
    parts: dict[tuple[str, str], dict[int, dict[str, object]]] = {}
    for body in bodies:
        for match in ROUND_TRANSPORT_SIDECAR_RE.finditer(body):
            try:
                item = json.loads(_unb64(match.group("payload")).decode())
                if not isinstance(item, dict):
                    continue
                key = (str(item["anchor"]), str(item["field"]))
                index = int(item["index"])
                count = int(item["count"])
                if index < 0 or count < 1 or index >= count:
                    continue
                old = parts.setdefault(key, {}).get(index)
                if old is None or old == item:
                    parts[key][index] = item
                else:
                    parts[key].pop(index, None)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue

    result = dict(payload)
    missing: set[str] = set()
    for field in _SPILL_FIELDS:
        ref = result.get(field)
        if not isinstance(ref, dict) or "$round_transport_spill" not in ref:
            continue
        try:
            anchor = str(ref["$round_transport_spill"])
            count = int(ref["parts"])
            entries = parts.get((anchor, field), {})
            if count < 1 or len(entries) != count or any(index not in entries for index in range(count)):
                raise ValueError("missing parts")
            ordered = [entries[index] for index in range(count)]
            if any(
                str(item.get("anchor")) != anchor
                or str(item.get("field")) != field
                or int(item.get("count", -1)) != count
                or str(item.get("sha256")) != str(ref["sha256"])
                or str(item.get("spill")) != str(ref["spill"])
                for item in ordered
            ):
                raise ValueError("inconsistent parts")
            packed = _unb64("".join(str(item["data"]) for item in ordered))
            if hashlib.sha256(packed).hexdigest() != str(ref["spill"]):
                raise ValueError("corrupt payload")
            raw = _decompress_bounded(packed)
            if hashlib.sha256(raw).hexdigest() != str(ref["sha256"]):
                raise ValueError("corrupt payload")
            result[field] = raw.decode("utf-8")
        except (KeyError, TypeError, UnicodeDecodeError, ValueError, zlib.error):
            missing.add(field)
            result[field] = None
    return result, missing
